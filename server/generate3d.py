"""Generate 3D: cisTEM's Generate3DPanel (src/gui/Generate3DPanel.cpp) as a
driver in the parent/child shape of refine3d.py, with the shortest cycle of
them all: reconstruct3d x N -> merge3d x classes, once, from a chosen
refinement's parameters and nothing else -- no refinement, no schedule.

What it is for is what the panel's text says: a 3D from imported or edited
parameters, a look at how the reconstruction settings (mask radii, score
weighting and threshold, resolution limit, cropping, Ewald sphere
correction) change the map, or the half maps saved. Each class's map
becomes a volume asset "Generated from #<refinement> - Class #<k>"
(Assets/Volumes/generate3d_volume_<n>_<refinement>_<k>.mrc, `_map1` /
`_map2` beside it when the half maps are kept) with a RECONSTRUCTION_LIST
row; with Overwrite Statistics on, the input refinement's resolution
statistics are replaced by the new FSC and its details point at the new
volume, as ProcessAllJobsFinished() does. The package's current references
are left alone -- cisTEM's panel does not touch them either.

One deliberate difference from cisTEM: its StartReconstructionClick() hands
reconstruct3d 0 (no correction) when "Apply Ewald Sphere correction" is
Yes and +/-1 when it is No -- the answers are crossed. Here Yes means
corrected (+1, or -1 with "Apply For Inverse Hand") and No means 0, which
is what reconstruct3d's own prompt documents.
"""
import json
import os
import threading
from pathlib import Path

import db
import refinements
import starfile
import volumes
from abinitio import (_new_child, _parent_row, _profile, _task, _load_state, _save, now_iso, particle_range, read_statistics,
                      pooled_part_ssnr)
from refine3d import _num, _flag, PLEASE_CREATE_PACKAGE_MESSAGE, CHILD_RECON, CHILD_MERGE
from stages import merge3d, reconstruct3d

STAGE = "generate3d"

# Generate3DPanel::SetDefaults(); the mask radius follows the package.
DEFAULTS = {
    "inner_mask_radius_a": 0.0, "score_to_weight_constant": 2.0, "adjust_score_for_defocus": True, "score_threshold": 0.0,
    "reconstruction_resolution_limit_a": 0.0, "autocrop_images": False, "save_half_maps": False, "overwrite_statistics": True,
    "apply_ewald_correction": False, "ewald_inverse_hand": False,
}


def package_defaults(pkg):
    size = float(pkg["PARTICLE_SIZE"] or 150.0)
    return {"mask_radius_a": round(size * 0.6, 2)}


def settings_from_params(params, pkg):
    s = {}
    for k, v in DEFAULTS.items():
        s[k] = _flag(params, k, v) if isinstance(v, bool) else _num(params, k, v)
    for k, v in package_defaults(pkg).items():
        s[k] = _num(params, k, v)
    return s


def ewald_flag(apply_correction, inverse_hand):
    """reconstruct3d's correct_ewald_sphere: 0 = no, 1 = correct hand, -1 = wrong hand."""
    if not apply_correction:
        return 0
    return -1 if inverse_hand else 1


def validate(conn, params):
    package_id = params.get("refinement_package_id")
    if package_id in (None, ""):
        raise ValueError(PLEASE_CREATE_PACKAGE_MESSAGE)
    pkg = conn.execute("SELECT * FROM REFINEMENT_PACKAGE_ASSETS WHERE REFINEMENT_PACKAGE_ASSET_ID=?", (int(package_id),)).fetchone()
    if pkg is None:
        raise ValueError("refinement package {} does not exist".format(package_id))
    if not pkg["STACK_FILENAME"] or not os.path.isfile(pkg["STACK_FILENAME"]):
        raise ValueError("the particle stack {} is missing".format(pkg["STACK_FILENAME"]))
    rid = params.get("input_refinement_id")
    if rid in (None, ""):
        rid = pkg["LAST_REFINEMENT_ID"]
    ref = refinements.refinement_row(conn, rid) if rid is not None else None
    if ref is None:
        raise ValueError("the input parameters (refinement {}) do not exist".format(rid))
    if ref["REFINEMENT_PACKAGE_ASSET_ID"] != pkg["REFINEMENT_PACKAGE_ASSET_ID"]:
        raise ValueError("refinement {} belongs to a different package".format(rid))
    return pkg, ref


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------

_runtime = None
_lock = threading.Lock()


def configure(runtime):
    global _runtime
    _runtime = runtime


def _log(project_id, job_id, text, level="info"):
    _runtime.append_log(project_id, job_id, text, level=level)


def scratch_dir(project_id, job_id):
    d = db.project_dir(project_id) / "Scratch" / "Generate3D" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def start(conn, project_id, job_id, params, profile):
    """StartReconstructionClick(): the run bar's profile is cisTEM's
    Reconstruction Run Profile -- the only one this panel has."""
    pkg, ref = validate(conn, params)
    if not profile or profile.get("total_jobs", 0) <= 0:
        raise ValueError("run profile {!r} has no run commands, so it can't launch anything".format((profile or {}).get("name")))
    s = settings_from_params(params, pkg)
    classes = max(1, int(ref["NUMBER_OF_CLASSES"] or pkg["NUMBER_OF_CLASSES"] or 1))
    n = int(ref["NUMBER_OF_PARTICLES"])
    scratch = scratch_dir(project_id, job_id)
    for p in scratch.iterdir():
        try:
            p.unlink()
        except OSError:
            pass
    state = {
        "phase": None, "settings": s,
        "package_id": pkg["REFINEMENT_PACKAGE_ASSET_ID"], "package_name": pkg["NAME"], "stack_filename": pkg["STACK_FILENAME"],
        "pixel_size": float(pkg["OUTPUT_PIXEL_SIZE"] or 1.0), "box_size": int(pkg["STACK_BOX_SIZE"]), "invert_contrast": bool(pkg["STACK_HAS_WHITE_PROTEIN"]),
        "symmetry": pkg["SYMMETRY"] or "C1", "molecular_weight": float(pkg["MOLECULAR_WEIGHT"] or 300.0),
        "number_of_particles": n, "number_of_classes": classes,
        "reconstruction_profile": profile["name"], "reconstruction_jobs": int(profile["total_jobs"]),
        "input_refinement_id": ref["REFINEMENT_ID"], "input_refinement_name": ref["NAME"],
        "child_job_id": None, "child_task_count": 0, "child_done": 0, "history": [], "started_at": now_iso(), "scratch": str(scratch),
        "volume_ids": [], "reference_files": [],
    }
    with conn:
        conn.execute("UPDATE JOBS SET STATUS='running', STARTED_AT=?, PROGRESS=0 WHERE JOB_ID=?", (now_iso(), job_id))
    _log(project_id, job_id, "Generate 3D of {!r} from {!r}: {} particles, {} class{}, mask {:.1f} Å, profile {!r}".format(
        pkg["NAME"], ref["NAME"], n, classes, "" if classes == 1 else "es", s["mask_radius_a"], profile["name"]))
    _launch_reconstruction(conn, project_id, job_id, state)
    _save(conn, job_id, state)
    return state


def _launch_reconstruction(conn, project_id, job_id, state):
    """SetupReconstructionJob() + RunReconstructionJob()."""
    s = state["settings"]
    n, classes = state["number_of_particles"], state["number_of_classes"]
    rid = state["input_refinement_id"]
    scratch = Path(state["scratch"])
    written = []
    for k in range(1, classes + 1):
        p = str(scratch / "generate3d_par_{}_{}.star".format(rid, k))
        starfile.write_star(p, refinements.load_rows(conn, rid, k))
        written.append(p)
    jobs = max(1, min(n, state["reconstruction_jobs"]))
    ewald = ewald_flag(s["apply_ewald_correction"], s["ewald_inverse_hand"])
    tasks = []
    index = 0
    for k in range(classes):
        for j in range(1, jobs + 1):
            first, last = particle_range(j, jobs, n)
            values = [state["stack_filename"], written[k], "/dev/null", "/dev/null", "/dev/null", "/dev/null", "/dev/null", state["symmetry"], first, last,
                      state["pixel_size"], state["molecular_weight"], s["inner_mask_radius_a"], s["mask_radius_a"],
                      s["reconstruction_resolution_limit_a"], 0.0, s["score_to_weight_constant"], s["score_threshold"], 1.0, 1.0,
                      True, s["adjust_score_for_defocus"], state["invert_contrast"], False, s["autocrop_images"], False, False,
                      False, True, True,
                      str(scratch / "dump_file_{}_{}_odd_{}.dump".format(rid, k, j)), str(scratch / "dump_file_{}_{}_even_{}.dump".format(rid, k, j)), ewald, 1]
            tasks.append(_task(reconstruct3d, index, k * 1000 + j, values))
            index += 1
    parent = _parent_row(conn, job_id)
    child = _new_child(conn, job_id, CHILD_RECON, "{} · reconstruct3d".format(parent["NAME"]), parent)
    state.update({"phase": "recon", "child_job_id": child, "child_task_count": len(tasks), "child_done": 0, "number_of_dump_files": jobs})
    _log(project_id, job_id, "Calculating {} ({} task{}{}) — child job {}".format("reconstructions" if classes > 1 else "reconstruction", len(tasks), "" if len(tasks) == 1 else "s",
                                                                                 ", Ewald sphere correction {}".format("for the inverse hand" if ewald < 0 else "on") if ewald else "", child))
    _runtime.submit_child(project_id, child, reconstruct3d, tasks, _profile(state["reconstruction_profile"]))


def _launch_merge(conn, project_id, job_id, state):
    """SetupMerge3dJob() + RunMerge3dJob(): generate3d_volume_<n>_<refinement>_<class>.mrc."""
    s = state["settings"]
    classes = state["number_of_classes"]
    rid = state["input_refinement_id"]
    scratch = Path(state["scratch"])
    vol_dir = volumes.volume_dir(project_id)
    n_3d = conn.execute("SELECT COUNT(*) FROM RECONSTRUCTION_LIST").fetchone()[0] + 1
    tasks, outputs, stats, halves = [], [], [], []
    for k in range(classes):
        stem = "generate3d_volume_{}_{}_{}".format(n_3d, rid, k + 1)
        out = str(vol_dir / (stem + ".mrc"))
        st = str(scratch / "volume_stats_{}_{}.txt".format(rid, k + 1))
        half = [str(vol_dir / (stem + "_map1.mrc")), str(vol_dir / (stem + "_map2.mrc"))] if s["save_half_maps"] else ["/dev/null", "/dev/null"]
        outputs.append(out)
        stats.append(st)
        halves.append(half if s["save_half_maps"] else ["", ""])
        tasks.append(_task(merge3d, k, k + 1, [half[0], half[1], out, st, state["molecular_weight"], s["inner_mask_radius_a"], s["mask_radius_a"],
                                                str(scratch / "dump_file_{}_{}_odd_.dump".format(rid, k)), str(scratch / "dump_file_{}_{}_even_.dump".format(rid, k)),
                                                k + 1, False, "", int(state.get("number_of_dump_files") or 1), 1.0, 5.0]))
    parent = _parent_row(conn, job_id)
    child = _new_child(conn, job_id, CHILD_MERGE, "{} · merge3d".format(parent["NAME"]), parent)
    state.update({"phase": "merge", "child_job_id": child, "child_task_count": len(tasks), "child_done": 0,
                  "pending_volume_files": outputs, "pending_stats_files": stats, "pending_half_maps": halves})
    _log(project_id, job_id, "Merging and filtering {} — child job {}".format("reconstructions" if classes > 1 else "reconstruction", child))
    _runtime.submit_child(project_id, child, merge3d, tasks, _profile(state["reconstruction_profile"]))


def _progress_percent(state):
    frac = float(state.get("child_done", 0)) / float(max(state.get("child_task_count", 1), 1))
    phase = state.get("phase")
    if phase == "recon":
        return int(80 * frac)
    if phase == "merge":
        return int(80 + 20 * frac)
    if phase == "finished":
        return 100
    return 0


def progress_info(state):
    if not state:
        return {}
    finished = [h["finished_at"] for h in state.get("history", []) if h.get("finished_at")]
    return {"task_count": 1, "tasks_done": len(finished), "first_task_finished_at": finished[0] if finished else None,
            "last_task_finished_at": finished[-1] if finished else None, "round": 0, "rounds": 1, "phase": state["phase"]}


def child_progress(conn, parent_id, child_id, done_count, task_count):
    state = _load_state(conn, parent_id)
    if not state or state.get("child_job_id") != child_id:
        return
    state["child_done"] = done_count
    state["child_task_count"] = task_count or state.get("child_task_count", 1)
    _save(conn, parent_id, state, _progress_percent(state))


def child_finished(project_id, child_row, status, error=None):
    threading.Thread(target=_child_finished, args=(project_id, child_row["JOB_ID"], child_row["PARENT_JOB_ID"], status, error),
                     daemon=True, name="generate3d-" + child_row["PARENT_JOB_ID"]).start()


def _child_finished(project_id, child_id, parent_id, status, error):
    with _lock:
        conn = db.get_conn(project_id)
        try:
            state = _load_state(conn, parent_id)
            parent = _parent_row(conn, parent_id)
            if not state or parent is None or state.get("child_job_id") != child_id or parent["STATUS"] not in ("queued", "running"):
                return
            if parent["CANCEL_REQUESTED"] or status == "cancelled":
                _finish(conn, project_id, parent_id, state, "cancelled", "cancelled during {}".format(state["phase"]))
                return
            if status != "completed":
                _finish(conn, project_id, parent_id, state, "failed", "{} failed{}".format(state["phase"], ": " + error if error else ""))
                return
            try:
                if state["phase"] == "recon":
                    _launch_merge(conn, project_id, parent_id, state)
                    _save(conn, parent_id, state, _progress_percent(state))
                elif state["phase"] == "merge":
                    _record(conn, project_id, parent_id, state)
                    state["phase"] = "finished"
                    state["child_job_id"] = None
                    _log(project_id, parent_id, "Reconstruction is finished!")
                    _finish(conn, project_id, parent_id, state, "completed", None)
                else:
                    raise ValueError("unexpected phase {!r}".format(state["phase"]))
            except Exception as exc:  # noqa: BLE001
                _finish(conn, project_id, parent_id, state, "failed", "could not continue: {}".format(exc))
        finally:
            conn.close()


def _record(conn, project_id, parent_id, state):
    """The MERGE branch of ProcessAllJobsFinished()."""
    s = state["settings"]
    classes = state["number_of_classes"]
    rid = state["input_refinement_id"]
    for p in state["pending_volume_files"]:
        if not os.path.isfile(p):
            raise ValueError("merge3d did not write {}".format(p))
    class_rows = [refinements.load_rows(conn, rid, k) for k in range(1, classes + 1)]
    stats = [read_statistics(p) if os.path.isfile(p) else [] for p in state["pending_stats_files"]]
    avgs = pooled_part_ssnr(stats, class_rows)
    volume_ids, est_res = [], []
    for k in range(classes):
        recon_id = refinements.next_reconstruction_id(conn)
        half = state.get("pending_half_maps", [["", ""]] * classes)[k]
        vid = volumes.add_volume_asset(conn, "Generated from #{} - Class #{}".format(rid, k + 1), state["pending_volume_files"][k], state["pixel_size"],
                                       state["box_size"], state["box_size"], state["box_size"], reconstruction_job_id=recon_id,
                                       half_map_1=half[0] if os.path.isfile(half[0] or "") else "", half_map_2=half[1] if os.path.isfile(half[1] or "") else "")
        refinements.add_reconstruction_job(conn, recon_id, state["package_id"], rid, "", s["inner_mask_radius_a"], s["mask_radius_a"],
                                           s["reconstruction_resolution_limit_a"], s["score_to_weight_constant"], s["adjust_score_for_defocus"],
                                           s["autocrop_images"], s["save_half_maps"], False, 1.0, k + 1, vid)
        volume_ids.append(vid)
        est_res.append(refinements.estimated_resolution(stats[k], state["pixel_size"]) if stats[k] else None)
        if s["overwrite_statistics"]:
            refinements.update_details(conn, rid, k + 1, RECONSTRUCTED_VOLUME_ASSET_ID=vid, RECONSTRUCTION_ID=recon_id, ESTIMATED_RESOLUTION=est_res[k] or 0.0)
    if s["overwrite_statistics"]:
        with conn:
            for k in range(classes):
                if stats[k]:
                    refinements.write_statistics(conn, rid, k + 1, stats[k])
            conn.execute("UPDATE REFINEMENT_LIST SET RESOLUTION_STATISTICS_ARE_GENERATED=0 WHERE REFINEMENT_ID=?", (rid,))
    for k in range(classes):
        if classes > 1:
            _log(project_id, parent_id, "Est. Res. Class {:2d} = {} Å ({:.2f} %)".format(k + 1, "{:.2f}".format(est_res[k]) if est_res[k] else "n/a", avgs[k]))
        else:
            _log(project_id, parent_id, "Est. Res. = {} Å".format("{:.2f}".format(est_res[k]) if est_res[k] else "n/a"))
    state["reference_files"] = list(state["pending_volume_files"])
    state["volume_ids"] = volume_ids
    state["history"].append({"round": 1, "refinement_id": rid, "label": "Reconstruction", "estimated_resolution": est_res, "finished_at": now_iso()})
    _log(project_id, parent_id, "Volume asset{} {} written{}".format("" if classes == 1 else "s", ", ".join("#{}".format(v) for v in volume_ids),
                                                                     "; the statistics of Refinement #{} were replaced".format(rid) if s["overwrite_statistics"] else ""))


def _remove_scratch(state):
    d = Path(state.get("scratch") or "")
    if d.is_dir():
        for p in d.glob("*"):
            try:
                p.unlink()
            except OSError:
                pass
        try:
            d.rmdir()
        except OSError:
            pass


def _finish(conn, project_id, parent_id, state, status, error):
    _remove_scratch(state)
    metrics = {"refinement_id": state.get("input_refinement_id"), "volume_ids": state.get("volume_ids", [])}
    if state.get("history"):
        est = [v for v in (state["history"][-1].get("estimated_resolution") or []) if v]
        if est:
            metrics["resolution_a"] = min(est)
    cpu = conn.execute("SELECT COALESCE(SUM(json_extract(METRICS_JSON, '$.cpu_ms')), 0) FROM JOBS WHERE PARENT_JOB_ID=?", (parent_id,)).fetchone()[0]
    metrics["cpu_ms"] = cpu or 0
    with conn:
        conn.execute("UPDATE JOBS SET STATUS=?, ERROR=?, FINISHED_AT=?, PROGRESS=?, METRICS_JSON=?, STATE_JSON=? WHERE JOB_ID=?",
                     (status, error, now_iso(), 100 if status == "completed" else _progress_percent(state), json.dumps(metrics), json.dumps(state), parent_id))
    if status != "completed":
        _log(project_id, parent_id, "job {}: {}".format(status, error), level="error")


def cancel(conn, project_id, parent_id):
    state = _load_state(conn, parent_id)
    with conn:
        conn.execute("UPDATE JOBS SET CANCEL_REQUESTED=1 WHERE JOB_ID=?", (parent_id,))
    child_id = (state or {}).get("child_job_id")
    if child_id and _runtime.cancel(child_id):
        _log(project_id, parent_id, "cancel requested; stopping child job {}".format(child_id))
        return True
    if state is not None:
        _finish(conn, project_id, parent_id, state, "cancelled", "cancelled")
    return False


def resume(project_id, parent_row):
    conn = db.get_conn(project_id)
    try:
        state = _load_state(conn, parent_row["JOB_ID"])
        if not state:
            with conn:
                conn.execute("UPDATE JOBS SET STATUS='failed', ERROR='Server restarted before this run recorded its plan', FINISHED_AT=? WHERE JOB_ID=?",
                             (now_iso(), parent_row["JOB_ID"]))
            return
        child_id = state.get("child_job_id")
        child = conn.execute("SELECT * FROM JOBS WHERE JOB_ID=?", (child_id,)).fetchone() if child_id else None
        if child is None:
            _finish(conn, project_id, parent_row["JOB_ID"], state, "failed", "Server restarted and the running step could not be found")
            return
        _log(project_id, parent_row["JOB_ID"], "server restarted during {}; waiting on child job {}".format(state["phase"], child_id))
        if child["STATUS"] in ("completed", "failed", "cancelled"):
            child_finished(project_id, child, child["STATUS"], child["ERROR"])
    finally:
        conn.close()


def live_result(conn, row):
    """The Jobs tab's Latest Result: the FSC and the reconstruction, once merged."""
    state = _load_state(conn, row["JOB_ID"])
    if not state or not state.get("history"):
        return None
    last = state["history"][-1]
    rid = last["refinement_id"]
    stats = refinements.load_statistics(conn, rid, 1) if state["settings"].get("overwrite_statistics") else []
    files = [p for p in (state.get("reference_files") or []) if p and os.path.isfile(p)]
    return {
        "kind": "generate3d", "task_index": 10 + (1 if state.get("phase") == "finished" else 0),
        "label": last["label"], "refinement_id": rid, "phase": state["phase"], "round": 0, "rounds": 1,
        "number_of_classes": state["number_of_classes"], "has_picture": bool(files),
        "estimated_resolution": last.get("estimated_resolution"), "history": state["history"],
        "fsc": [{"resolution": s["resolution"], "fsc": s["fsc"], "part_fsc": s["part_fsc"]} for s in stats if s["resolution"]],
        "pixel_size": state["pixel_size"], "volume_ids": state.get("volume_ids", []), "package_name": state.get("package_name"),
    }


def current_picture(conn, row, class_index=0):
    state = _load_state(conn, row["JOB_ID"])
    if not state:
        return None
    files = [p for p in (state.get("reference_files") or []) if p and os.path.isfile(p)]
    if not files:
        return None
    path = files[min(class_index, len(files) - 1)]
    png, meta = volumes.orthogonal_views_png(path, state["settings"]["mask_radius_a"])
    return png, meta, path
