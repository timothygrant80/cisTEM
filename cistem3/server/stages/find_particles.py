"""Find Particles: the `find_particles` adapter.

Mirrors MyFindParticlesPanel in cisTEM (src/gui/FindParticlesPanel.cpp). The
task argument list is the 28 positional arguments find_particles'
DoCalculation() reads, in the order StartPickingClick() packs them
("tffffffffbtbiffftiifbbffbiib"). Only cisTEM's "ab initio" algorithm is
offered -- a soft-edged disc template generated from the characteristic
radius, matched with Sigworth's (2004) filter -- because that is the one
algorithm cisTEM's own panel offers.

Every image needs an active CTF estimate first (the panel's "Please run
CTF estimation on this group before picking particles"): the CTF is part
of the matched filter, and comes from ESTIMATED_CTF_PARAMETERS through
IMAGE_ASSETS.CTF_ESTIMATION_ID exactly as GetCTFParameters() reads it.

Results are five floats per pick: x and y in Angstroms from the image
centre (particle_finder.cpp: pixel_size * (box centre - peak)), the peak
height, the template index and the template rotation. finalize() writes
them as WriteResultToDataBase() does: a PARTICLE_PICKING_LIST row per
image, a PARTICLE_PICKING_RESULTS_<job> table with every pick of the job
(the history), and PARTICLE_POSITION_ASSETS holding the *active* picks of
each image -- this job's, replacing any earlier job's for the same images.
One deviation: IMAGE_ASSETS gains ACTIVE_PICKING_ID so that "which pick
job is active for this image" is stored rather than inferred from which
asset rows survived (an image with zero picks would otherwise be
ambiguous). It is the same idea as CTF_ESTIMATION_ID, one stage later.
"""

import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import db
import job_protocol as jp

PROGRAM = {"name": "find_particles", "executable": "find_particles"}

BACKGROUND_ALGORITHMS = ("Lowest variance", "Variance near mode")  # AlgorithmToFindBackgroundChoice, in index order

CTF_REQUIRED_MESSAGE = "Please run CTF estimation on this group before picking particles"


def _num(params, key, default, cast):
    v = params.get(key)
    if v is None or v == "":
        return default
    try:
        return cast(v)
    except (TypeError, ValueError):
        return default


def _flag(params, key, default):
    v = params.get(key)
    if v is None or v == "":
        return default
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _ensure_dirs(project_id):
    # cisTEM: Project::CreateNewProject() makes Assets/ParticlePosition.
    d = db.project_dir(project_id) / "Assets" / "ParticlePosition"
    d.mkdir(parents=True, exist_ok=True)
    return d


def images_without_ctf(conn, group_id):
    """Members of an image group with no active CTF estimate -- the ones that
    make the group unpickable (cisTEM's can_be_picked)."""
    return [r["IMAGE_ASSET_ID"] for r in conn.execute(
        "SELECT ia.IMAGE_ASSET_ID FROM IMAGE_ASSETS ia JOIN IMAGE_GROUP_MEMBERS m ON m.IMAGE_ASSET_ID = ia.IMAGE_ASSET_ID "
        "LEFT JOIN ESTIMATED_CTF_PARAMETERS ce ON ce.CTF_ESTIMATION_ID = ia.CTF_ESTIMATION_ID "
        "WHERE m.GROUP_ID = ? AND ce.CTF_ESTIMATION_ID IS NULL", (int(group_id),))]


def build_tasks(conn, project_id, params):
    group_id = params.get("image_group_id")
    if group_id is None:
        raise ValueError("image_group_id is required")
    if images_without_ctf(conn, group_id):
        raise ValueError(CTF_REQUIRED_MESSAGE)
    rows = conn.execute(
        "SELECT ia.*, ce.VOLTAGE AS CTF_VOLTAGE, ce.SPHERICAL_ABERRATION AS CTF_CS, ce.AMPLITUDE_CONTRAST, ce.DEFOCUS1, "
        "ce.DEFOCUS2, ce.DEFOCUS_ANGLE, ce.ADDITIONAL_PHASE_SHIFT "
        "FROM IMAGE_ASSETS ia JOIN IMAGE_GROUP_MEMBERS m ON m.IMAGE_ASSET_ID = ia.IMAGE_ASSET_ID "
        "JOIN ESTIMATED_CTF_PARAMETERS ce ON ce.CTF_ESTIMATION_ID = ia.CTF_ESTIMATION_ID "
        "WHERE m.GROUP_ID = ? ORDER BY ia.IMAGE_ASSET_ID", (int(group_id),)).fetchall()
    if not rows:
        raise ValueError("image group {} has no images".format(group_id))

    out_dir = _ensure_dirs(project_id)

    settings = _settings_from_params(params)
    tasks = []
    for index, image in enumerate(rows):
        asset_id = image["IMAGE_ASSET_ID"]
        previous = conn.execute("SELECT COUNT(*) FROM PARTICLE_PICKING_LIST WHERE PARENT_IMAGE_ASSET_ID=?", (asset_id,)).fetchone()[0]
        # cisTEM: <image stem>_COOS_<number of previous picks>.mrc; the
        # candidate stack itself is skipped (box size 0), as the GUI does.
        output_stack = str(out_dir / "{}_COOS_{}.mrc".format(Path(image["FILENAME"]).stem, previous))
        tasks.append({"index": index, "ref": asset_id, "args": _image_args(image, image["FILENAME"], image["PIXEL_SIZE"] or 1.0, output_stack, settings)})
    return tasks


def _settings_from_params(params):
    """The panel's controls -> one dict, ResetDefaults() values when absent."""
    algo = params.get("background_algorithm", BACKGROUND_ALGORITHMS[0])
    restrain = _flag(params, "avoid_low_variance", True)
    return dict(
        maximum_radius=_num(params, "maximum_radius_a", 120.0, float),
        characteristic_radius=_num(params, "characteristic_radius_a", 80.0, float),
        threshold=_num(params, "threshold_peak_height", 6.0, float),
        avoid_low_variance=restrain,
        low_variance_threshold=_num(params, "low_variance_threshold", -0.5, float),
        avoid_high_variance=_flag(params, "avoid_high_variance", False),
        high_variance_threshold=_num(params, "high_variance_threshold", 2.0, float),
        highest_resolution=_num(params, "highest_resolution_a", 15.0, float),
        # cisTEM computes the edge distance from the exclusion radius unless
        # "Set minimum distance from edges" is ticked (ReturnDefaultMinimumDistanceFromEdges);
        # None here means "compute it per image" in _image_args().
        min_edge_distance=_num(params, "min_edge_distance_px", 128, int) if _flag(params, "set_min_edge_distance", False) else None,
        avoid_abnormal_mean=_flag(params, "avoid_abnormal_mean", True),
        background_boxes=_num(params, "background_boxes", 40, int),
        background_algorithm=BACKGROUND_ALGORITHMS.index(algo) if algo in BACKGROUND_ALGORITHMS else _num(params, "background_algorithm", 0, int),
    )


def _image_args(image, input_file, pixel_size, output_stack, st):
    """The 28 arguments for one image (a row of IMAGE_ASSETS joined with its
    active CTF estimate), in StartPickingClick()'s order."""
    A = jp.arg
    # cisTEM: int(exclusion radius / pixel size) + 1, from the group's first
    # image; here from each image's own, which is the same for a group of
    # one pixel size and right for a mixed one.
    edge = st["min_edge_distance"] if st["min_edge_distance"] is not None else default_edge_distance(st["maximum_radius"], pixel_size)
    args = [
        A("text", input_file),                              # 0  micrograph
        A("float", pixel_size),                             # 1
        A("float", image["CTF_VOLTAGE"] or image["VOLTAGE"] or 300.0),  # 2  kV
        A("float", image["CTF_CS"] or image["SPHERICAL_ABERRATION"] or 2.7),  # 3  mm
        A("float", image["AMPLITUDE_CONTRAST"] or 0.07),    # 4
        A("float", image["ADDITIONAL_PHASE_SHIFT"] or 0.0),  # 5  rad
        A("float", image["DEFOCUS1"] or 0.0),               # 6  A
        A("float", image["DEFOCUS2"] or 0.0),               # 7  A
        A("float", image["DEFOCUS_ANGLE"] or 0.0),          # 8  deg
        A("bool", False),                                   # 9  already have templates -- ab initio only
        A("text", "no_templates.mrc"),                      # 10
        A("bool", False),                                   # 11 average templates radially
        A("int", 1),                                        # 12 template rotations
        A("float", st["characteristic_radius"]),                  # 13 typical radius (A)
        A("float", st["maximum_radius"]),                         # 14 maximum radius (A)
        A("float", st["highest_resolution"]),                     # 15
        A("text", output_stack),                            # 16 candidate stack (not written: box size 0)
        A("int", 0),                                        # 17 output stack box size
        A("int", int(edge)),                                # 18 px
        A("float", st["threshold"]),                              # 19 picking threshold
        A("bool", st["avoid_low_variance"]),                      # 20
        A("bool", st["avoid_high_variance"]),                     # 21
        A("float", st["low_variance_threshold"]),                 # 22 FWHM
        A("float", st["high_variance_threshold"]),                # 23 FWHM
        A("bool", st["avoid_abnormal_mean"]),                     # 24
        A("int", st["background_algorithm"]),                     # 25 0 lowest variance, 1 near mode
        A("int", st["background_boxes"]),                         # 26
        A("bool", bool(image["PROTEIN_IS_WHITE"])),         # 27
    ]
    assert len(args) == 28
    return args


def default_edge_distance(maximum_radius, pixel_size):
    """MyFindParticlesPanel::ReturnDefaultMinimumDistanceFromEdges()."""
    return int(float(maximum_radius) / float(pixel_size or 1.0)) + 1


def _arg_values(task):
    return [a["value"] for a in task["args"]]


def _positions(task_row):
    """The result's five floats per pick -> [{x, y, peak_height, template, rotation}] (x, y in A from the image origin, y up).
    An image with no candidate particles sends no result at all (cisTEM's
    OnSocketJobResultMsg ignores an empty one), so a finished task without
    one is a legitimate zero picks, not a failure."""
    if task_row["STATUS"] == "ok" and not task_row["RESULT_JSON"]:
        return []
    result = json.loads(task_row["RESULT_JSON"]) if task_row["RESULT_JSON"] else None
    data = (result or {}).get("data") if (result or {}).get("kind") == "floats" else None
    if data is None:
        return None
    n = len(data) // 5
    return [{"x": float(data[5 * i]), "y": float(data[5 * i + 1]), "peak_height": float(data[5 * i + 2]),
             "template": int(data[5 * i + 3]), "rotation": float(data[5 * i + 4])} for i in range(n)]


def results_table(job_id):
    """cisTEM names it by the integer picking job id; ours are hex strings, which are still valid identifiers here."""
    return "PARTICLE_PICKING_RESULTS_{}".format(job_id)


_RESULTS_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS {}(POSITION_ID INTEGER PRIMARY KEY, PICKING_ID INTEGER, PARENT_IMAGE_ASSET_ID INTEGER, "
    "X_POSITION REAL, Y_POSITION REAL, PEAK_HEIGHT REAL, TEMPLATE_ASSET_ID INTEGER, TEMPLATE_PSI REAL, TEMPLATE_THETA REAL, TEMPLATE_PHI REAL)"
)


def _replace_active_picks(conn, image_id, picking_id, job_id):
    """Make one picking the image's active one: its picks become the image's
    PARTICLE_POSITION_ASSETS rows (any earlier job's for this image go, and
    with them their group memberships), and IMAGE_ASSETS.ACTIVE_PICKING_ID
    records the choice."""
    old = [r[0] for r in conn.execute("SELECT PARTICLE_POSITION_ASSET_ID FROM PARTICLE_POSITION_ASSETS WHERE PARENT_IMAGE_ASSET_ID=?", (image_id,))]
    if old:
        conn.executemany("DELETE FROM PARTICLE_POSITION_GROUP_MEMBERS WHERE PARTICLE_POSITION_ASSET_ID=?", [(i,) for i in old])
        conn.execute("DELETE FROM PARTICLE_POSITION_ASSETS WHERE PARENT_IMAGE_ASSET_ID=?", (image_id,))
    conn.execute(
        "INSERT INTO PARTICLE_POSITION_ASSETS(PARENT_IMAGE_ASSET_ID, PICKING_ID, PICK_JOB_ID, X_POSITION, Y_POSITION, PEAK_HEIGHT, "
        "TEMPLATE_ASSET_ID, TEMPLATE_PSI, TEMPLATE_THETA, TEMPLATE_PHI) "
        "SELECT PARENT_IMAGE_ASSET_ID, PICKING_ID, ?, X_POSITION, Y_POSITION, PEAK_HEIGHT, TEMPLATE_ASSET_ID, TEMPLATE_PSI, TEMPLATE_THETA, TEMPLATE_PHI "
        "FROM {} WHERE PICKING_ID=?".format(results_table(job_id)), (job_id, picking_id))
    conn.execute(
        "INSERT OR IGNORE INTO PARTICLE_POSITION_GROUP_MEMBERS(GROUP_ID, PARTICLE_POSITION_ASSET_ID) "
        "SELECT 0, PARTICLE_POSITION_ASSET_ID FROM PARTICLE_POSITION_ASSETS WHERE PARENT_IMAGE_ASSET_ID=?", (image_id,))
    conn.execute("UPDATE IMAGE_ASSETS SET ACTIVE_PICKING_ID=? WHERE IMAGE_ASSET_ID=?", (picking_id, image_id))


def finalize(conn, project_id, job, sent_tasks, task_rows, log):
    by_index = {t["index"]: t for t in sent_tasks}
    written = skipped = picked = 0
    now = db.now_epoch()
    table = results_table(job["id"])
    with conn:
        conn.execute(_RESULTS_TABLE_SQL.format(table))
        for row in task_rows:
            task = by_index.get(row["TASK_INDEX"])
            positions = _positions(row) if row["STATUS"] == "ok" and task is not None else None
            if positions is None:
                skipped += 1
                if row["STATUS"] == "ok":
                    log("task {}: no usable picking result; skipped".format(row["TASK_INDEX"]), level="error")
                continue
            v = _arg_values(task)
            image_id = int(row["REF"]) if row["REF"] is not None else int(task["ref"])
            if conn.execute("SELECT 1 FROM IMAGE_ASSETS WHERE IMAGE_ASSET_ID=?", (image_id,)).fetchone() is None:
                log("task {}: image asset {} no longer exists; skipped".format(row["TASK_INDEX"], image_id), level="error")
                skipped += 1
                continue
            cur = conn.execute(
                "INSERT INTO PARTICLE_PICKING_LIST(DATETIME_OF_RUN, PICKING_JOB_ID, PARENT_IMAGE_ASSET_ID, PICKING_ALGORITHM, "
                "CHARACTERISTIC_RADIUS, MAXIMUM_RADIUS, THRESHOLD_PEAK_HEIGHT, HIGHEST_RESOLUTION_USED_IN_PICKING, "
                "MIN_DIST_FROM_EDGES, AVOID_HIGH_VARIANCE, AVOID_HIGH_LOW_MEAN, NUM_BACKGROUND_BOXES, MANUAL_EDIT) "
                "VALUES (?,?,?,0,?,?,?,?,?,?,?,?,0)",
                (now, job["id"], image_id, v[13], v[14], v[19], v[15], int(v[18]), 1 if v[21] else 0, 1 if v[24] else 0, int(v[26])))
            picking_id = cur.lastrowid
            conn.executemany(
                "INSERT INTO {}(PICKING_ID, PARENT_IMAGE_ASSET_ID, X_POSITION, Y_POSITION, PEAK_HEIGHT, TEMPLATE_ASSET_ID, "
                "TEMPLATE_PSI, TEMPLATE_THETA, TEMPLATE_PHI) VALUES (?,?,?,?,?,?,?,?,?)".format(table),
                [(picking_id, image_id, p["x"], p["y"], p["peak_height"], p["template"], p["rotation"], 0.0, 0.0) for p in positions])
            _replace_active_picks(conn, image_id, picking_id, job["id"])
            written += 1
            picked += len(positions)
    return {"pickings_written": written, "particles_picked": picked, "tasks_skipped": skipped}


def describe_summary(summary):
    return "wrote picks for {} image{} to the project database: {} particles".format(
        summary.get("pickings_written", 0), "" if summary.get("pickings_written") == 1 else "s", summary.get("particles_picked", 0))


def activate_picking(conn, picking_id):
    """MyPickingResultsPanel::OnValueChanged(): another job's picks become
    the image's positions. Returns the image asset id, None if unknown."""
    row = conn.execute("SELECT PICKING_JOB_ID, PARENT_IMAGE_ASSET_ID FROM PARTICLE_PICKING_LIST WHERE PICKING_ID=?", (picking_id,)).fetchone()
    if row is None:
        return None
    if conn.execute("SELECT 1 FROM IMAGE_ASSETS WHERE IMAGE_ASSET_ID=?", (row["PARENT_IMAGE_ASSET_ID"],)).fetchone() is None:
        raise ValueError("the image these picks belong to no longer exists")
    if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (results_table(row["PICKING_JOB_ID"]),)).fetchone() is None:
        raise ValueError("this job's picks were never recorded (a simulated run)")
    with conn:
        _replace_active_picks(conn, row["PARENT_IMAGE_ASSET_ID"], picking_id, row["PICKING_JOB_ID"])
    return row["PARENT_IMAGE_ASSET_ID"]


def replace_picks(conn, picking_id, positions):
    """MyPickingResultsPanel's save of a manual edit (the block under
    UserHasEditedParticleCoordinates()): the picking's rows in the job's
    results table are replaced by the edited list -- added picks get new
    ids and a peak height of 0, as cisTEM's ParticlePositionAsset does --
    MANUAL_EDIT is set, and when this picking is the image's active one its
    position assets are rebuilt from it. Returns the image asset id, None
    if the picking is unknown; ValueError when the job recorded no picks."""
    row = conn.execute("SELECT PICKING_JOB_ID, PARENT_IMAGE_ASSET_ID FROM PARTICLE_PICKING_LIST WHERE PICKING_ID=?", (picking_id,)).fetchone()
    if row is None:
        return None
    job_id, image_id = row["PICKING_JOB_ID"], row["PARENT_IMAGE_ASSET_ID"]
    table = results_table(job_id)
    if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is None:
        raise ValueError("this job's picks were never recorded (a simulated run)")
    with conn:
        conn.execute("DELETE FROM {} WHERE PICKING_ID=?".format(table), (picking_id,))
        conn.executemany(
            "INSERT INTO {}(PICKING_ID, PARENT_IMAGE_ASSET_ID, X_POSITION, Y_POSITION, PEAK_HEIGHT, TEMPLATE_ASSET_ID, "
            "TEMPLATE_PSI, TEMPLATE_THETA, TEMPLATE_PHI) VALUES (?,?,?,?,?,?,?,?,?)".format(table),
            [(picking_id, image_id, float(p["x"]), float(p["y"]), float(p.get("peak_height") or 0.0), -1, 0.0, 0.0, 0.0) for p in positions])
        conn.execute("UPDATE PARTICLE_PICKING_LIST SET MANUAL_EDIT=1 WHERE PICKING_ID=?", (picking_id,))
        active = conn.execute("SELECT ACTIVE_PICKING_ID FROM IMAGE_ASSETS WHERE IMAGE_ASSET_ID=?", (image_id,)).fetchone()
        if active is not None and active[0] == picking_id:
            _replace_active_picks(conn, image_id, picking_id, job_id)
    return image_id


def activate_job_results(conn, job_id):
    ids = [r["PICKING_ID"] for r in conn.execute("SELECT PICKING_ID FROM PARTICLE_PICKING_LIST WHERE PICKING_JOB_ID=? ORDER BY PICKING_ID", (job_id,))]
    failed = []
    for pid in ids:
        try:
            activate_picking(conn, pid)
        except ValueError as exc:
            failed.append({"id": pid, "reason": str(exc)})
    return len(ids), failed


def picks_for(conn, picking_id):
    """The positions of one picking, from the job's results table."""
    job = conn.execute("SELECT PICKING_JOB_ID FROM PARTICLE_PICKING_LIST WHERE PICKING_ID=?", (picking_id,)).fetchone()
    if job is None or conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (results_table(job[0]),)).fetchone() is None:
        return []
    return [{"x": r["X_POSITION"], "y": r["Y_POSITION"], "peak_height": r["PEAK_HEIGHT"]} for r in conn.execute(
        "SELECT X_POSITION, Y_POSITION, PEAK_HEIGHT FROM {} WHERE PICKING_ID=? ORDER BY POSITION_ID".format(results_table(job[0])), (picking_id,))]


def live_result(conn, task, task_row):
    """What the panel draws as each image finishes: its picks over the image."""
    positions = _positions(task_row)
    if positions is None:
        return None
    v = _arg_values(task)
    image_id = int(task_row["REF"]) if task_row["REF"] is not None else int(task["ref"])
    image = conn.execute("SELECT NAME, X_SIZE, Y_SIZE, PIXEL_SIZE FROM IMAGE_ASSETS WHERE IMAGE_ASSET_ID=?", (image_id,)).fetchone()
    ctf = conn.execute("SELECT ICINESS FROM ESTIMATED_CTF_PARAMETERS ce JOIN IMAGE_ASSETS ia ON ia.CTF_ESTIMATION_ID = ce.CTF_ESTIMATION_ID "
                       "WHERE ia.IMAGE_ASSET_ID=?", (image_id,)).fetchone()
    return {
        "kind": "picks",
        "image_asset_id": image_id,
        "image_name": image["NAME"] if image else Path(v[0]).name,
        "x_size": image["X_SIZE"] if image else None, "y_size": image["Y_SIZE"] if image else None,
        "pixel_size": v[1],
        "maximum_radius": v[14], "characteristic_radius": v[13], "threshold_peak_height": v[19],
        "defocus1": v[6], "defocus2": v[7], "iciness": ctf["ICINESS"] if ctf else None,
        "positions": positions,
        "pick_count": len(positions),
        "image_file_exists": os.path.isfile(v[0]),
    }


def live_result_files(task):
    v = _arg_values(task)
    return {"image": v[0]}


# ---------------------------------------------------------------------------
# Preview: the picker on one image, now, without a job. cisTEM's panel does
# this in-process (Preview / Auto preview) so parameters can be tuned before
# a run; here the same find_particles binary is run directly, fed its
# interactive prompts on stdin, and the .plt it writes when run that way is
# read back. One at a time per server: previews are quick, and a second
# click shouldn't start a second search.
# ---------------------------------------------------------------------------

_preview_lock = threading.Lock()


def _yn(v):
    return "yes" if v else "no"


def _interactive_answers(args):
    """The answers DoInteractiveUserInput() asks for, in its order, from the
    same 28 arguments the socket path packs. Without templates it skips
    the template questions, so those arguments (9-12) have no prompt."""
    v = [a["value"] for a in args]
    return [
        v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8],
        "no",                      # supply templates? -- ab initio
        v[13], v[14], v[15], v[16], v[17], v[18], v[19],
        _yn(v[20]), _yn(v[21]), v[22], v[23], _yn(v[24]), v[25], v[26], _yn(v[27]),
    ]


def preview(conn, project_id, image_id, params, executable, timeout=120.0):
    """Pick one image with the panel's current parameters and return what
    the Results view draws (positions in A from the image origin, y up,
    in the full image's frame). Uses the scaled sum Align Movies wrote
    (Assets/Images/Scaled/<same name>) when its pixel size still satisfies
    the highest resolution asked for -- MyFindParticlesPanel's rule -- so
    a preview takes a fraction of a second."""
    image = conn.execute(
        "SELECT ia.*, ce.VOLTAGE AS CTF_VOLTAGE, ce.SPHERICAL_ABERRATION AS CTF_CS, ce.AMPLITUDE_CONTRAST, ce.DEFOCUS1, "
        "ce.DEFOCUS2, ce.DEFOCUS_ANGLE, ce.ADDITIONAL_PHASE_SHIFT, ce.ICINESS "
        "FROM IMAGE_ASSETS ia LEFT JOIN ESTIMATED_CTF_PARAMETERS ce ON ce.CTF_ESTIMATION_ID = ia.CTF_ESTIMATION_ID "
        "WHERE ia.IMAGE_ASSET_ID = ?", (int(image_id),)).fetchone()
    if image is None:
        raise LookupError("no such image asset")
    if image["DEFOCUS1"] is None:
        raise ValueError(CTF_REQUIRED_MESSAGE)
    if not image["FILENAME"] or not os.path.isfile(image["FILENAME"]):
        raise ValueError("the image file is missing: {}".format(image["FILENAME"]))

    st = _settings_from_params(params)
    pixel_size = image["PIXEL_SIZE"] or 1.0
    x_size, y_size = image["X_SIZE"], image["Y_SIZE"]
    input_file, input_ps, input_y, used_scaled = image["FILENAME"], pixel_size, y_size, False
    scaled = Path(image["FILENAME"]).parent / "Scaled" / Path(image["FILENAME"]).name
    if scaled.is_file() and x_size:
        try:
            from imageheaders import read_image_header
            hdr = read_image_header(str(scaled))
            scale = float(x_size) / float(hdr["x_size"])
            scaled_ps = pixel_size * scale
            if scaled_ps <= st["highest_resolution"] / 2.0:
                input_file, input_ps, input_y, used_scaled = str(scaled), scaled_ps, hdr["y_size"], True
                edge_full = st["min_edge_distance"] if st["min_edge_distance"] is not None else default_edge_distance(st["maximum_radius"], pixel_size)
                st = dict(st, min_edge_distance=max(1, int(round(edge_full / scale))))
        except Exception:  # noqa: BLE001 -- an unreadable scaled copy just means the full image is used
            pass

    started = time.time()
    with _preview_lock, tempfile.TemporaryDirectory(prefix="pick-preview-") as tmp:
        out_stack = os.path.join(tmp, "preview.mrc")
        args = _image_args(image, input_file, input_ps, out_stack, st)
        answers = "\n".join(str(a) for a in _interactive_answers(args)) + "\n"
        try:
            proc = subprocess.run([executable], input=answers, capture_output=True, text=True, cwd=tmp, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise TimeoutError("find_particles took longer than {:.0f} s on this image".format(timeout))
        plt = os.path.join(tmp, "preview.plt")
        # Run standalone, find_particles opens its .plt on the first pick and
        # deletes the handle on exit regardless, so with *no* picks it
        # segfaults once the search is complete ("Finding peaks ..." has been
        # printed) and before it reports a count. That is a clean zero-pick
        # result here, not a failure -- the socket path the real job uses has
        # no such step. Any other non-zero exit, or one that left no trace of
        # the search having run, is reported.
        searched = "Finding peaks" in (proc.stdout or "")
        if os.path.isfile(plt):
            pass
        elif proc.returncode in (-11, 139) and searched:
            pass  # zero picks
        elif proc.returncode != 0:
            tail = [l for l in (proc.stdout or "").strip().splitlines() if l.strip() and "%" not in l][-3:]
            raise RuntimeError("find_particles exited with {}: {}".format(proc.returncode, " | ".join(tail) or (proc.stderr or "").strip()[-300:]))
        positions = []
        if os.path.isfile(plt):
            for line in open(plt):
                parts = line.split()
                if len(parts) < 6 or parts[0].startswith("#"):
                    continue
                # The .plt has (row, column) in the *input* image's pixels,
                # 1-based, rows counted from the top: turn that back into the
                # socket result's convention, Angstroms from the origin, y up.
                row_px, col_px, peak = float(parts[0]) - 1.0, float(parts[1]) - 1.0, float(parts[5])
                positions.append({"x": col_px * input_ps, "y": (input_y - row_px) * input_ps, "peak_height": peak})
    return {
        "kind": "picks",
        "image_asset_id": image["IMAGE_ASSET_ID"],
        "image_name": image["NAME"],
        "x_size": x_size, "y_size": y_size, "pixel_size": pixel_size,
        "maximum_radius": st["maximum_radius"], "characteristic_radius": st["characteristic_radius"],
        "threshold_peak_height": st["threshold"],
        "defocus1": image["DEFOCUS1"], "defocus2": image["DEFOCUS2"], "iciness": image["ICINESS"],
        "positions": positions, "pick_count": len(positions),
        "used_scaled_image": used_scaled, "input_file": os.path.basename(input_file),
        "elapsed_s": round(time.time() - started, 2),
    }
