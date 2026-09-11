"""Find CTF: the `ctffind` adapter.

Mirrors MyFindCTFPanel in cisTEM (src/gui/FindCTFPanel.cpp). The task
argument list is the 41 positional arguments ctffind's DoCalculation()
reads, in the order the panel's AddJob() packs them -- the format string
there, "sbisffffifffffbfbfffbffbbsbsbfffbfffbiiib", is the contract, and
the installed ctffind must agree with it (a binary older than Nov 2023
lacks the final "weight down low resolution" flag).

Results come back as ten floats per image (ProcessResult()): defocus 1 and
2 (A), astigmatism angle (deg), additional phase shift (rad), score, the
resolution to which Thon rings were fit, the resolution at which aliasing
was detected (0 = none), iciness, tilt angle and tilt axis. finalize()
writes them the way WriteResultToDataBase() does: one ESTIMATED_CTF_PARAMETERS
row per image and IMAGE_ASSETS.CTF_ESTIMATION_ID pointing at the new row --
that pointer is the *active* estimate, what downstream stages use.

ctffind writes its diagnostic image (the filtered spectrum with the fit in
the lower-left quadrant) to Assets/CTF/<image>_CTF_<n>.mrc, and beside it
<same>_avrot.txt, the 1D curves cisTEM's CTF1DPanel plots.
"""

import json
import os
from pathlib import Path

import db
import job_protocol as jp

# package.program (docs/job-protocol.md section 6.2): the name workers report and the executable the run commands launch.
PROGRAM = {"name": "ctffind", "executable": "ctffind"}
DEV_NULL = "/dev/null"

# One-dimensional smoothing cisTEM applies to the rotational average before
# plotting it (Curve::FitSavitzkyGolayToData(7, 3)): a 7-point cubic
# Savitzky-Golay window has these fixed coefficients.
_SG7 = (-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0)
_SG7_NORM = 21.0


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
    # cisTEM: Project::CreateNewProject() makes Assets/CTF next to Assets/Images.
    ctf_dir = db.project_dir(project_id) / "Assets" / "CTF"
    ctf_dir.mkdir(parents=True, exist_ok=True)
    return ctf_dir


def build_tasks(conn, project_id, params):
    """One task per image in the chosen image group, with the arguments
    MyFindCTFPanel::StartEstimationClick() packs. `params` are the
    Actions panel's fields; defaults are ResetDefaults()'s."""
    group_id = params.get("image_group_id")
    if group_id is None:
        raise ValueError("image_group_id is required")
    images = conn.execute(
        "SELECT ia.* FROM IMAGE_ASSETS ia JOIN IMAGE_GROUP_MEMBERS m ON m.IMAGE_ASSET_ID = ia.IMAGE_ASSET_ID "
        "WHERE m.GROUP_ID = ? ORDER BY ia.IMAGE_ASSET_ID", (int(group_id),)).fetchall()
    if not images:
        raise ValueError("image group {} has no images".format(group_id))

    ctf_dir = _ensure_dirs(project_id)

    # Images when absent: the panel's default (cisTEM's ResetDefaults() picks Movies; the aligned sums are what most runs want).
    use_movies = str(params.get("estimate_using", "Images")).lower().startswith("movie")
    frames_to_average = _num(params, "frames_to_average", 3, int)
    box_size = _num(params, "box_size", 512, int)
    amplitude_contrast = _num(params, "amplitude_contrast", 0.07, float)
    determine_tilt = _flag(params, "search_tilt", False)
    min_res = _num(params, "min_resolution_a", 30.0, float)
    max_res = _num(params, "max_resolution_a", 5.0, float)
    min_defocus = _num(params, "min_defocus_a", 5000.0, float)
    max_defocus = _num(params, "max_defocus_a", 50000.0, float)
    defocus_step = _num(params, "defocus_step_a", 100.0, float)
    slower_search = _flag(params, "slower_search", False)
    restrain_astigmatism = _flag(params, "restrain_astigmatism", False)
    # cisTEM passes -100 for "no restraint"; the sign is what ctffind reads.
    tolerated_astigmatism = _num(params, "tolerated_astigmatism_a", 500.0, float) if restrain_astigmatism else -100.0
    find_phase_shift = _flag(params, "find_phase_shift", False)
    deg2rad = 3.14159265358979 / 180.0
    if find_phase_shift:
        min_phase = _num(params, "min_phase_shift_deg", 0.0, float) * deg2rad
        max_phase = _num(params, "max_phase_shift_deg", 180.0, float) * deg2rad
        phase_step = _num(params, "phase_shift_step_deg", 10.0, float) * deg2rad
    else:
        min_phase = max_phase = phase_step = 0.0
    filter_lowres = _flag(params, "filter_lowres_signal", True)

    A = jp.arg
    tasks = []
    for index, image in enumerate(images):
        asset_id = image["IMAGE_ASSET_ID"]
        movie = None
        if use_movies and image["PARENT_MOVIE_ID"] is not None and image["PARENT_MOVIE_ID"] >= 0:
            movie = conn.execute("SELECT * FROM MOVIE_ASSETS WHERE MOVIE_ASSET_ID=?", (image["PARENT_MOVIE_ID"],)).fetchone()
        # An imported micrograph has no movie to go back to: estimate on
        # the image itself, as cisTEM would if you picked Images.
        on_movie = movie is not None
        input_file = movie["FILENAME"] if on_movie else image["FILENAME"]
        pixel_size = (movie["PIXEL_SIZE"] if on_movie else image["PIXEL_SIZE"]) or 1.0

        previous = conn.execute(
            "SELECT COUNT(*) FROM ESTIMATED_CTF_PARAMETERS WHERE IMAGE_ASSET_ID=?", (asset_id,)).fetchone()[0]
        # cisTEM: <image stem>_CTF_<number of previous estimations>.mrc under Assets/CTF.
        diagnostic = str(ctf_dir / "{}_CTF_{}.mrc".format(Path(image["FILENAME"]).stem, previous))

        if on_movie:
            gain = movie["GAIN_FILENAME"] or ""
            dark = movie["DARK_FILENAME"] or ""
            correct_mag = bool(movie["CORRECT_MAG_DISTORTION"])
            mag_angle = movie["MAG_DISTORTION_ANGLE"] or 0.0
            mag_major = movie["MAG_DISTORTION_MAJOR_SCALE"] or 1.0
            mag_minor = movie["MAG_DISTORTION_MINOR_SCALE"] or 1.0
            eer_frames = movie["EER_FRAMES_PER_IMAGE"] or 25
            eer_super = movie["EER_SUPER_RES_FACTOR"] or 1
        else:
            gain = dark = ""
            correct_mag = False
            mag_angle, mag_major, mag_minor = 0.0, 1.0, 1.0
            eer_frames, eer_super = 0, 1

        args = [
            A("text", input_file),                       # 0  input image or movie
            A("bool", on_movie),                         # 1  input is a movie
            A("int", frames_to_average if on_movie else 1),  # 2
            A("text", diagnostic),                       # 3  output diagnostic image
            A("float", pixel_size),                      # 4
            A("float", image["VOLTAGE"] or 300.0),       # 5  kV
            A("float", image["SPHERICAL_ABERRATION"] or 2.7),  # 6  mm
            A("float", amplitude_contrast),              # 7
            A("int", box_size),                          # 8
            A("float", min_res),                         # 9
            A("float", max_res),                         # 10
            A("float", min_defocus),                     # 11
            A("float", max_defocus),                     # 12
            A("float", defocus_step),                    # 13
            A("bool", slower_search),                    # 14 large astigmatism expected
            A("float", tolerated_astigmatism),           # 15 negative = unrestrained
            A("bool", find_phase_shift),                 # 16
            A("float", min_phase),                       # 17 rad
            A("float", max_phase),                       # 18 rad
            A("float", phase_step),                      # 19 rad
            A("bool", False),                            # 20 astigmatism is known -- not in the GUI
            A("float", 0.0),                             # 21
            A("float", 0.0),                             # 22
            A("bool", True),                             # 23 resample if pixel too small
            A("bool", gain == ""),                       # 24 movie is gain corrected
            A("text", gain or DEV_NULL),                 # 25
            A("bool", dark == ""),                       # 26 movie is dark corrected
            A("text", dark or DEV_NULL),                 # 27
            A("bool", correct_mag),                      # 28
            A("float", float(mag_angle)),                # 29
            A("float", float(mag_major)),                # 30
            A("float", float(mag_minor)),                # 31
            A("bool", False),                            # 32 defocus is known -- not in the GUI
            A("float", 0.0),                             # 33
            A("float", 0.0),                             # 34
            A("float", 0.0),                             # 35
            A("bool", determine_tilt),                   # 36
            A("int", 1),                                 # 37 threads (the worker's command line overrides)
            A("int", int(eer_frames)),                   # 38
            A("int", int(eer_super)),                    # 39
            A("bool", filter_lowres),                    # 40 weight down low resolution signal
        ]
        assert len(args) == 41
        tasks.append({"index": index, "ref": asset_id, "args": args})
    return tasks


def _arg_values(task):
    return [a["value"] for a in task["args"]]


def _result_floats(task_row):
    result = json.loads(task_row["RESULT_JSON"]) if task_row["RESULT_JSON"] else None
    data = (result or {}).get("data") if (result or {}).get("kind") == "floats" else None
    return data if data and len(data) >= 7 else None


def _avrot_path(diagnostic_file):
    p = Path(diagnostic_file)
    return p.with_name(p.stem + "_avrot.txt")


def read_avrot(diagnostic_file):
    """The 1D curves ShowCTFResultsPanel::Draw() plots, from the
    <diagnostic>_avrot.txt ctffind writes: six lines per micrograph --
    spatial frequency (1/A), rotational average assuming no astigmatism,
    rotational average, CTF fit, cross-correlation between spectrum and
    fit, 2 sigma of the expected noise correlation. cisTEM plots the
    Savitzky-Golay-smoothed rotational average, the fit and the quality.
    None if the file isn't there."""
    path = _avrot_path(diagnostic_file)
    try:
        lines = [l for l in path.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]
    except OSError:
        return None
    if len(lines) < 5:
        return None
    rows = []
    for line in lines[:6]:
        try:
            rows.append([float(x) for x in line.split()])
        except ValueError:
            return None
    n = min(len(r) for r in rows[:5])
    freq, amplitude, fit, quality = rows[0][:n], rows[2][:n], rows[3][:n], rows[4][:n]
    smoothed = []
    for i in range(n):
        acc = 0.0
        for k, c in enumerate(_SG7):
            j = min(max(i + k - 3, 0), n - 1)
            acc += c * amplitude[j]
        smoothed.append(acc / _SG7_NORM)
    return {"frequency": freq, "amplitude": smoothed, "fit": fit, "quality": quality}


def _point_image_asset(conn, image_asset_id, ctf_estimation_id):
    """Make an estimate the image's *active* one: IMAGE_ASSETS.CTF_ESTIMATION_ID
    is where cisTEM keeps that, and what Find Particles / refinement read."""
    conn.execute("UPDATE IMAGE_ASSETS SET CTF_ESTIMATION_ID=? WHERE IMAGE_ASSET_ID=?", (ctf_estimation_id, image_asset_id))


def finalize(conn, project_id, job, sent_tasks, task_rows, log):
    """MyFindCTFPanel::WriteResultToDataBase(): one ESTIMATED_CTF_PARAMETERS
    row per finished image, and the image asset pointed at it. Returns the
    counts the job's metrics carry."""
    by_index = {t["index"]: t for t in sent_tasks}
    written = skipped = aliasing_in_range = 0
    now = db.now_epoch()
    with conn:
        for row in task_rows:
            task = by_index.get(row["TASK_INDEX"])
            data = _result_floats(row) if row["STATUS"] == "ok" and task is not None else None
            if data is None:
                skipped += 1
                if row["STATUS"] == "ok":
                    log("task {}: no usable CTF result; skipped".format(row["TASK_INDEX"]), level="error")
                continue
            v = _arg_values(task)
            image_id = int(row["REF"]) if row["REF"] is not None else int(task["ref"])
            if conn.execute("SELECT 1 FROM IMAGE_ASSETS WHERE IMAGE_ASSET_ID=?", (image_id,)).fetchone() is None:
                log("task {}: image asset {} no longer exists; skipped".format(row["TASK_INDEX"], image_id), level="error")
                skipped += 1
                continue
            data = list(data) + [0.0] * (10 - len(data))
            restrain = v[15] >= 0
            cur = conn.execute(
                "INSERT INTO ESTIMATED_CTF_PARAMETERS("
                "CTF_ESTIMATION_JOB_ID, DATETIME_OF_RUN, IMAGE_ASSET_ID, ESTIMATED_ON_MOVIE_FRAMES, VOLTAGE, "
                "SPHERICAL_ABERRATION, PIXEL_SIZE, AMPLITUDE_CONTRAST, BOX_SIZE, MIN_RESOLUTION, MAX_RESOLUTION, "
                "MIN_DEFOCUS, MAX_DEFOCUS, DEFOCUS_STEP, RESTRAIN_ASTIGMATISM, TOLERATED_ASTIGMATISM, "
                "FIND_ADDITIONAL_PHASE_SHIFT, MIN_PHASE_SHIFT, MAX_PHASE_SHIFT, PHASE_SHIFT_STEP, DEFOCUS1, DEFOCUS2, "
                "DEFOCUS_ANGLE, ADDITIONAL_PHASE_SHIFT, SCORE, DETECTED_RING_RESOLUTION, DETECTED_ALIAS_RESOLUTION, "
                "OUTPUT_DIAGNOSTIC_FILE, NUMBER_OF_FRAMES_AVERAGED, LARGE_ASTIGMATISM_EXPECTED, ICINESS, TILT_ANGLE, TILT_AXIS) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    job["id"], now, image_id, 1 if v[1] else 0, v[5], v[6], v[4], v[7], int(v[8]), v[9], v[10],
                    v[11], v[12], v[13], 1 if restrain else 0, v[15] if restrain else 0.0,
                    1 if v[16] else 0, v[17] if v[16] else 0.0, v[18] if v[16] else 0.0, v[19] if v[16] else 0.0,
                    float(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5]), float(data[6]),
                    v[3], int(v[2]), 1 if v[14] else 0, float(data[7]), float(data[8]), float(data[9]),
                ),
            )
            _point_image_asset(conn, image_id, cur.lastrowid)
            written += 1
            if data[6] > v[10]:
                aliasing_in_range += 1
    if aliasing_in_range:
        # cisTEM's warning, verbatim in substance.
        log("For {} of {} micrographs, CTF aliasing was detected within the fit range. Aliasing may affect the "
            "detected fit resolution and/or the quality of the defocus estimates. To reduce aliasing, use a larger "
            "box size.".format(aliasing_in_range, written))
    return {"ctf_estimates_written": written, "tasks_skipped": skipped}


def describe_summary(summary):
    n = summary.get("ctf_estimates_written", 0)
    return "wrote {} CTF estimate{} to the project database".format(n, "" if n == 1 else "s")


def activate_estimate(conn, ctf_estimation_id):
    """MyFindCTFResultsPanel::OnValueChanged(): checking another job's cell
    for an image points the asset at that estimate. Returns the image
    asset id, or None if the estimate doesn't exist."""
    row = conn.execute("SELECT IMAGE_ASSET_ID FROM ESTIMATED_CTF_PARAMETERS WHERE CTF_ESTIMATION_ID=?",
                       (ctf_estimation_id,)).fetchone()
    if row is None:
        return None
    if conn.execute("SELECT 1 FROM IMAGE_ASSETS WHERE IMAGE_ASSET_ID=?", (row["IMAGE_ASSET_ID"],)).fetchone() is None:
        raise ValueError("the image this estimate belongs to no longer exists")
    with conn:
        _point_image_asset(conn, row["IMAGE_ASSET_ID"], ctf_estimation_id)
    return row["IMAGE_ASSET_ID"]


def activate_job_results(conn, job_id):
    """Every estimate of one job becomes its image's active one."""
    ids = [r["CTF_ESTIMATION_ID"] for r in conn.execute(
        "SELECT CTF_ESTIMATION_ID FROM ESTIMATED_CTF_PARAMETERS WHERE CTF_ESTIMATION_JOB_ID=? ORDER BY CTF_ESTIMATION_ID", (job_id,))]
    failed = []
    for eid in ids:
        try:
            activate_estimate(conn, eid)
        except ValueError as exc:
            failed.append({"id": eid, "reason": str(exc)})
    return len(ids), failed


def live_result(conn, task, task_row):
    """What MyFindCTFPanel::ProcessResult() draws as each image finishes:
    the numbers, the diagnostic image and the 1D fit curves."""
    data = _result_floats(task_row)
    if data is None:
        return None
    data = list(data) + [0.0] * (10 - len(data))
    v = _arg_values(task)
    image_id = int(task_row["REF"]) if task_row["REF"] is not None else int(task["ref"])
    image = conn.execute("SELECT NAME FROM IMAGE_ASSETS WHERE IMAGE_ASSET_ID=?", (image_id,)).fetchone()
    return {
        "kind": "ctf",
        "image_asset_id": image_id,
        "image_name": image["NAME"] if image else Path(v[0]).name,
        "defocus1": data[0], "defocus2": data[1], "defocus_angle": data[2], "additional_phase_shift": data[3],
        "score": data[4], "detected_ring_resolution": data[5], "detected_alias_resolution": data[6],
        "iciness": data[7], "tilt_angle": data[8], "tilt_axis": data[9],
        "find_additional_phase_shift": bool(v[16]),
        "output_diagnostic_file": v[3],
        "diagnostic_file_exists": os.path.isfile(v[3]),
        "plot": read_avrot(v[3]),
    }


def live_result_files(task):
    v = _arg_values(task)
    return {"diagnostic": v[3]}
