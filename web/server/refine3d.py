"""Refine 3D: cisTEM's RefinementManager (src/gui/MyRefine3DPanel.cpp) as a
server-side driver, in the parent/child shape of classification.py and
abinitio.py.

Per round: (mask the reference) -> refine3d x N -> reconstruct3d x N ->
merge3d x classes, for `number_of_rounds` rounds, starting from an earlier
refinement's parameters (the package's "Random Parameters" start, or the
output of a previous run) and the package's *current references* -- the
volume assets REFINEMENT_PACKAGE_CURRENT_REFERENCES_<pkg> names, normally
what ab-initio or the last refinement produced. A class whose reference is
still "generate from parameters" (-1) gets an initial reconstruction from
the input parameters first (start_with_reconstruction), exactly as cisTEM
does.

Unlike ab-initio there is no schedule: every round uses the panel's limits,
and every round writes a full refinement -- REFINEMENT_LIST,
REFINEMENT_DETAILS, per-class results, resolution statistics, angular
distribution (refinements.py) -- and one volume asset per class
(Assets/Volumes/volume_<refinement>_<class>.mrc) recorded in
RECONSTRUCTION_LIST and made the package's current reference, so the next
run picks up where this one stopped. The reference is masked before each
round either automatically (volumes.auto_mask) or with a supplied volume
(volumes.apply_mask, the Multiply3DMaskerThread), with the panel's edge,
outside weight and optional low-pass outside the mask.

Children carry STAGE abinitio_refine3d / abinitio_reconstruct3d /
abinitio_merge3d -- the same adapters ab-initio uses; refine3d and its
statistics files go through the run bar's profile, reconstruct3d and
merge3d through the panel's Reconstruction Run Profile.
"""
import json
import logging
import math
import os
import random
import sys
import threading
from pathlib import Path

import numpy as np

import blush
import blush_phase
import db
import progress_store
import refinements
import starfile
import volumes
from abinitio import (merge_profile, _new_child, _parent_row, _profile, _task, _load_state, _save, now_iso, particle_range,
                      default_statistics, write_statistics, read_statistics, update_occupancies, pooled_part_ssnr,
                      average_sigma, angular_step as calculate_angular_step)
from stages import merge3d, reconstruct3d, refine3d as refine3d_adapter

log = logging.getLogger(__name__)

STAGE = "refine3d"
CHILD_REFINE = "abinitio_refine3d"
CHILD_RECON = "abinitio_reconstruct3d"
CHILD_MERGE = "abinitio_merge3d"

# MyRefine3DPanel::SetDefaults(); the size-dependent ones follow the package.
DEFAULTS = {
    "number_of_rounds": 1, "refinement_type": "Local Refinement", "high_resolution_limit_a": 30.0,
    "refine_psi": True, "refine_theta": True, "refine_phi": True, "refine_x_shift": True, "refine_y_shift": True, "refine_occupancies": True,
    "signed_cc_resolution_limit_a": 0.0, "percent_used": 100.0, "inner_mask_radius_a": 0.0,
    "number_of_results_to_refine": 20, "also_refine_input": True,
    "focused_classification": False, "sphere_x_a": 0.0, "sphere_y_a": 0.0, "sphere_z_a": 0.0, "sphere_radius_a": 0.0,
    "refine_ctf": False, "defocus_search_range_a": 500.0, "defocus_search_step_a": 50.0,
    "score_to_weight_constant": 2.0, "adjust_score_for_defocus": True, "score_threshold": 0.0, "reconstruction_resolution_limit_a": 0.0,
    "autocrop_images": False, "apply_blurring": False, "smoothing_factor": 1.0, "autocenter": False,
    "use_mask": False, "auto_mask": True, "mask_edge_a": 10.0, "outside_mask_weight": 0.0, "low_pass_outside_mask": False, "mask_filter_resolution_a": 20.0,
    # Blush regularisation of the reference between rounds (blush.py); not in cisTEM's panel.
    "use_blush": False, "blush_input": "Unfiltered half-map sum",
}
PLEASE_CREATE_PACKAGE_MESSAGE = ("Please create a refinement package (in the assets panel) in order to perform a "
                                 "3D refinement.")


def package_defaults(pkg):
    """SetDefaults(): the numbers that follow the package's largest dimension."""
    size = float(pkg["PARTICLE_SIZE"] or 150.0)
    local_mask = size * 0.65
    return {"low_resolution_limit_a": round(min(size * 1.5, 300.0), 2), "high_resolution_limit_a": 30.0,
            "mask_radius_a": round(local_mask, 2), "global_mask_radius_a": round(size * 0.8, 2),
            "angular_step_deg": round(calculate_angular_step(30.0, local_mask), 2),
            "search_range_x_a": round(size * 0.15, 2), "search_range_y_a": round(size * 0.15, 2),
            "classification_high_resolution_limit_a": 30.0}


def _num(params, key, default, cast=float):
    v = params.get(key)
    if v is None or v == "":
        return cast(default)
    try:
        return cast(v)
    except (TypeError, ValueError):
        return cast(default)


def _flag(params, key, default):
    v = params.get(key)
    if v is None or v == "":
        return bool(default)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def settings_from_params(params, pkg):
    d = package_defaults(pkg)
    s = {}
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            s[k] = _flag(params, k, v)
        elif isinstance(v, str):
            s[k] = str(params.get(k) or v)
        elif isinstance(v, int):
            s[k] = _num(params, k, v, int)
        else:
            s[k] = _num(params, k, v)
    for k, v in d.items():
        s[k] = _num(params, k, v)
    s["global"] = s["refinement_type"].strip().lower().startswith("global")
    s["number_of_rounds"] = max(1, s["number_of_rounds"])
    s["blush_unfiltered"] = not s["blush_input"].strip().lower().startswith("filtered")
    return s


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
    classes = max(1, int(pkg["NUMBER_OF_CLASSES"] or 1))
    refs = refinements.current_references(conn, pkg["REFINEMENT_PACKAGE_ASSET_ID"])
    reference_ids = [refs.get(k, -1) if refs.get(k) is not None else -1 for k in range(1, classes + 1)]
    for k, vid in enumerate(reference_ids, 1):
        if vid is not None and vid >= 0:
            vol = conn.execute("SELECT * FROM VOLUME_ASSETS WHERE VOLUME_ASSET_ID=?", (vid,)).fetchone()
            if vol is None or not vol["FILENAME"] or not os.path.isfile(vol["FILENAME"]):
                raise ValueError("the reference volume of class {} (asset {}) is missing".format(k, vid))
            if vol["X_SIZE"] != pkg["STACK_BOX_SIZE"] or abs(float(vol["PIXEL_SIZE"] or 0) - float(pkg["OUTPUT_PIXEL_SIZE"] or 0)) > 0.001:
                raise ValueError("Reference volume has different dimensions / pixel size from the input stack. This will currently not work.")
    mask = None
    if _flag(params, "use_mask", False):
        mid = params.get("mask_volume_id")
        if mid in (None, ""):
            raise ValueError("Pick a mask volume, or untick Use a Mask.")
        mask = conn.execute("SELECT * FROM VOLUME_ASSETS WHERE VOLUME_ASSET_ID=?", (int(mid),)).fetchone()
        if mask is None or not mask["FILENAME"] or not os.path.isfile(mask["FILENAME"]):
            raise ValueError("the mask volume is missing")
        if mask["X_SIZE"] != pkg["STACK_BOX_SIZE"]:
            raise ValueError("the mask volume has a different box size from the particle stack")
    if _flag(params, "use_blush", False):
        info = blush.availability()
        if not info["available"]:
            raise ValueError("Blush cannot run on this server: {}".format(info["reason"]))
    return pkg, ref, reference_ids, mask


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------

_runtime = None
_locks = {}
_locks_guard = threading.Lock()


def _job_lock(job_id):
    """One lock per parent job: its steps run one at a time (a child ending
    before start() has saved the state waits for it), but a job of this stage
    being submitted or finishing a step is not held behind another job's
    minutes of bookkeeping."""
    with _locks_guard:
        return _locks.setdefault(job_id, threading.Lock())


def configure(runtime):
    global _runtime
    _runtime = runtime


def _log(project_id, job_id, text, level="info"):
    _runtime.append_log(project_id, job_id, text, level=level)


def scratch_dir(project_id, job_id):
    d = db.project_dir(project_id) / "Scratch" / "Refine3D" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _class_star(state, tag, k):
    """The scratch copy of class k's parameters under `tag`: cisTEM's binary form."""
    return str(Path(state["scratch"]) / "refinement_{}_class{}.cistem".format(tag, k + 1))


def _store_rows(state, tag, class_rows):
    for k, rows in enumerate(class_rows):
        starfile.write_params(_class_star(state, tag, k), refinements.as_table(rows))


def _load_rows(state, tag):
    """The per-class parameter tables stored under `tag`."""
    return [starfile.read_params(_class_star(state, tag, k), as_table=True) for k in range(state["number_of_classes"])]



def start(conn, project_id, job_id, params, profile):
    """The first step, under the driver's lock: the child it launches can end
    before this returns (a controller that fails to launch does so inside
    submit()), and _child_finished() must then find the parent's state saved,
    not a parent that never hears of it and stays "running" forever."""
    with _job_lock(job_id):
        return _start(conn, project_id, job_id, params, profile)

def _start(conn, project_id, job_id, params, profile):
    """BeginRefinementCycle()."""
    pkg, ref, reference_ids, mask = validate(conn, params)
    recon_profile = _profile(params.get("reconstruction_run_profile") or params.get("run_profile")) or profile
    if not profile or profile.get("total_jobs", 0) <= 0:
        raise ValueError("run profile {!r} has no run commands, so it can't launch anything".format((profile or {}).get("name")))
    if not recon_profile or recon_profile.get("total_jobs", 0) <= 0:
        raise ValueError("reconstruction run profile {!r} has no run commands".format((recon_profile or {}).get("name")))
    s = settings_from_params(params, pkg)
    classes = max(1, int(pkg["NUMBER_OF_CLASSES"] or 1))
    n = int(ref["NUMBER_OF_PARTICLES"])
    scratch = scratch_dir(project_id, job_id)
    for p in scratch.iterdir():
        try:
            p.unlink()
        except OSError:
            pass
    reference_files = []
    for vid in reference_ids:
        vol = conn.execute("SELECT FILENAME FROM VOLUME_ASSETS WHERE VOLUME_ASSET_ID=?", (vid,)).fetchone() if vid >= 0 else None
        reference_files.append(vol["FILENAME"] if vol else None)
    state = {
        "phase": None, "round": 0, "rounds": s["number_of_rounds"], "settings": s,
        "package_id": pkg["REFINEMENT_PACKAGE_ASSET_ID"], "package_name": pkg["NAME"], "stack_filename": pkg["STACK_FILENAME"],
        "pixel_size": float(pkg["OUTPUT_PIXEL_SIZE"] or 1.0), "box_size": int(pkg["STACK_BOX_SIZE"]), "invert_contrast": bool(pkg["STACK_HAS_WHITE_PROTEIN"]),
        "symmetry": pkg["SYMMETRY"] or "C1", "molecular_weight": float(pkg["MOLECULAR_WEIGHT"] or 300.0), "particle_size": float(pkg["PARTICLE_SIZE"] or 150.0),
        "number_of_particles": n, "number_of_classes": classes,
        "refinement_profile": profile["name"], "refinement_jobs": int(profile["total_jobs"]),
        "reconstruction_profile": recon_profile["name"], "reconstruction_jobs": int(recon_profile["total_jobs"]),
        "input_refinement_id": ref["REFINEMENT_ID"], "input_refinement_name": ref["NAME"], "output_refinement_id": None,
        "start_with_reconstruction": any(v < 0 for v in reference_ids),
        "reference_files": reference_files, "reference_asset_ids": list(reference_ids),
        "mask_file": mask["FILENAME"] if mask is not None else None, "mask_asset_id": mask["VOLUME_ASSET_ID"] if mask is not None else -1,
        "child_job_id": None, "child_task_count": 0, "child_done": 0, "history": [], "started_at": now_iso(), "scratch": str(scratch),
        "initial": bool(any(v < 0 for v in reference_ids)), "volume_ids": [],
    }
    with conn:
        conn.execute("UPDATE JOBS SET STATUS='running', STARTED_AT=?, PROGRESS=0 WHERE JOB_ID=?", (now_iso(), job_id))
    _log(project_id, job_id, "Refine 3D of {!r} from {!r}: {} particles, {} class{}, {} refinement, {} round{}, {:.1f} Å limit, refinement profile {!r}, reconstruction profile {!r}".format(
        pkg["NAME"], ref["NAME"], n, classes, "" if classes == 1 else "es", "global search" if s["global"] else "local", s["number_of_rounds"],
        "" if s["number_of_rounds"] == 1 else "s", s["high_resolution_limit_a"], profile["name"], recon_profile["name"]))
    # The input refinement's rows go to scratch once, as the "output" tables every later step reads (the initial
    # reconstruction, each round's refine3d inputs and merge), in place of re-reading the database each round.
    _store_rows(state, "output", [refinements.load_table(conn, ref["REFINEMENT_ID"], k) for k in range(1, classes + 1)])
    if state["initial"]:
        state["output_refinement_id"] = ref["REFINEMENT_ID"]
        _log(project_id, job_id, "A class has no reference volume yet: reconstructing one from the input parameters first")
        _launch_reconstruction(conn, project_id, job_id, state)
    else:
        _prepare_references_then_refine(conn, project_id, job_id, state)
    _save(conn, job_id, state)
    return state


def _launch_reconstruction(conn, project_id, job_id, state):
    """SetupReconstructionJob() + RunReconstructionJob()."""
    s = state["settings"]
    n, classes = state["number_of_particles"], state["number_of_classes"]
    rid = state["output_refinement_id"]
    rng = np.random.default_rng()
    class_rows = _load_rows(state, "output")
    written = []
    for k, rows in enumerate(class_rows):
        out = refinements.as_table(rows).copy()
        if state["initial"]:
            # WritecisTEMStarFiles(percent_used / 100, sigma_override = 1)
            if s["percent_used"] < 100.0:
                out["image_is_active"] = np.where(rng.uniform(-1.0, 1.0, len(out)) < 1.0 - 2.0 * s["percent_used"] / 100.0, -1, 1)
            out["sigma"] = 1.0
        p = str(Path(state["scratch"]) / "recon_input_{}_class{}.cistem".format(rid, k + 1))
        starfile.write_params(p, out)
        written.append(p)
    jobs = max(1, min(n, state["reconstruction_jobs"]))
    scratch = Path(state["scratch"])
    tasks = []
    index = 0
    for k in range(classes):
        use_ref = bool(s["apply_blurring"] and state["reference_files"][k])
        for j in range(1, jobs + 1):
            first, last = particle_range(j, jobs, n)
            values = [state["stack_filename"], written[k], state["reference_files"][k] if use_ref else "/dev/null",
                      "/dev/null", "/dev/null", "/dev/null", "/dev/null", state["symmetry"], first, last,
                      state["pixel_size"], state["molecular_weight"], s["inner_mask_radius_a"], s["mask_radius_a"],
                      s["reconstruction_resolution_limit_a"], s["high_resolution_limit_a"], s["score_to_weight_constant"], s["score_threshold"],
                      s["smoothing_factor"], 1.0,
                      True, s["adjust_score_for_defocus"], state["invert_contrast"], False, s["autocrop_images"], False, s["autocenter"],
                      use_ref, True, True,
                      str(scratch / "dump_file_{}_{}_odd_{}.dump".format(rid, k, j)), str(scratch / "dump_file_{}_{}_even_{}.dump".format(rid, k, j)), 0, 1]
            tasks.append(_task(reconstruct3d, index, k * 1000000 + j, values))
            index += 1
    parent = _parent_row(conn, job_id)
    child = _new_child(conn, job_id, CHILD_RECON, "{} · {}reconstruction {}".format(parent["NAME"], "initial " if state["initial"] else "", rid), parent)
    state.update({"phase": "initial_recon" if state["initial"] else "recon", "child_job_id": child, "child_task_count": len(tasks), "child_done": 0,
                  "number_of_dump_files": jobs})
    _log(project_id, job_id, "Calculating {}{} ({} task{}) — child job {}".format("initial " if state["initial"] else "", "reconstructions" if classes > 1 else "reconstruction",
                                                                                   len(tasks), "" if len(tasks) == 1 else "s", child))
    _runtime.submit_child(project_id, child, reconstruct3d, tasks, _profile(state["reconstruction_profile"]))


def _launch_merge(conn, project_id, job_id, state):
    """SetupMerge3dJob() + RunMerge3dJob(): the merged volume goes straight to Assets/Volumes."""
    s = state["settings"]
    classes = state["number_of_classes"]
    rid = state["output_refinement_id"]
    scratch = Path(state["scratch"])
    vol_dir = volumes.volume_dir(project_id)
    tasks, outputs, stats, halves = [], [], [], []
    keep_halves = s["use_blush"] and s["blush_unfiltered"]   # after the merge, Blush denoises the half maps' average (the unfiltered reconstruction)
    for k in range(classes):
        out = str(vol_dir / "volume_{}_{}.mrc".format(rid, k + 1))
        st = str(scratch / "volume_stats_{}_{}.txt".format(rid, k + 1))
        half = [str(scratch / "half_{}_{}_{}.mrc".format(rid, k + 1, h)) for h in (1, 2)] if keep_halves else ["/dev/null", "/dev/null"]
        outputs.append(out)
        stats.append(st)
        halves.append(half)
        tasks.append(_task(merge3d, k, k + 1, [half[0], half[1], out, st, state["molecular_weight"], s["inner_mask_radius_a"], s["mask_radius_a"],
                                                str(scratch / "dump_file_{}_{}_odd_.dump".format(rid, k)), str(scratch / "dump_file_{}_{}_even_.dump".format(rid, k)),
                                                k + 1, False, "", _required_count(state, "number_of_dump_files"), 1.0, 5.0]))
    parent = _parent_row(conn, job_id)
    child = _new_child(conn, job_id, CHILD_MERGE, "{} · merge {}".format(parent["NAME"], rid), parent)
    state.update({"phase": "initial_merge" if state["initial"] else "merge", "child_job_id": child, "child_task_count": len(tasks), "child_done": 0,
                  "pending_volume_files": outputs, "pending_stats_files": stats, "pending_half_maps": halves if keep_halves else []})
    _log(project_id, job_id, "Merging and filtering {} — child job {}".format("reconstructions" if classes > 1 else "reconstruction", child))
    _runtime.submit_child(project_id, child, merge3d, tasks, merge_profile(_profile(state["reconstruction_profile"]), len(tasks)))


def _launch_refinement(conn, project_id, job_id, state):
    """SetupRefinementJob() + RunRefinementJob()."""
    s = state["settings"]
    n, classes = state["number_of_particles"], state["number_of_classes"]
    input_id = state["input_refinement_id"]
    rid = refinements.next_refinement_id(conn)
    state["output_refinement_id"] = rid
    scratch = Path(state["scratch"])
    defaults = default_statistics(state["molecular_weight"], state["pixel_size"], state["box_size"])
    star_files, stats_files = [], []
    stored = _load_rows(state, "output")   # the round's input rows, kept in scratch (see _merge_output_stars)
    for k in range(1, classes + 1):
        p = str(scratch / "input_par_{}_{}.cistem".format(input_id, k))
        starfile.write_params(p, stored[k - 1])
        star_files.append(p)
        # WriteStatisticsToFile() writes shells 1..box/2 only; the stored
        # curve (and a package's synthetic one) also has shell 0 and the
        # corners beyond Nyquist, which refine3d's reader rejects.
        stats = [st for st in refinements.load_statistics(conn, input_id, k) if 1 <= st["shell"] <= state["box_size"] // 2] or defaults
        sp = str(scratch / "input_stats_{}_{}.txt".format(input_id, k))
        write_statistics(sp, stats, state["pixel_size"])
        stats_files.append(sp)
    jobs = max(1, min(n, state["refinement_jobs"]))
    tasks, outputs = [], []
    index = 0
    for k in range(classes):
        for j in range(1, jobs + 1):
            first, last = particle_range(j, jobs, n)
            out_star = str(scratch / "refine_output_{}_class{}_{}.cistem".format(rid, k + 1, j))
            outputs.append(out_star)
            values = [state["stack_filename"], star_files[k], state["reference_files"][k], stats_files[k], True, "", out_star, "/dev/null",
                      state["symmetry"], first, last, s["percent_used"] / 100.0, state["pixel_size"], state["molecular_weight"],
                      s["inner_mask_radius_a"], s["mask_radius_a"], s["low_resolution_limit_a"], s["high_resolution_limit_a"],
                      s["signed_cc_resolution_limit_a"], s["classification_high_resolution_limit_a"], s["global_mask_radius_a"],
                      s["high_resolution_limit_a"], s["angular_step_deg"], int(s["number_of_results_to_refine"]),
                      s["search_range_x_a"], s["search_range_y_a"], s["sphere_x_a"], s["sphere_y_a"], s["sphere_z_a"], s["sphere_radius_a"],
                      s["defocus_search_range_a"], s["defocus_search_step_a"], 1.0,
                      s["global"], not s["global"], s["refine_psi"], s["refine_theta"], s["refine_phi"], s["refine_x_shift"], s["refine_y_shift"],
                      False, s["focused_classification"], s["refine_ctf"], True, state["invert_contrast"], False, not s["apply_blurring"], True,
                      1, False, k, not s["also_refine_input"], False]
            tasks.append(_task(refine3d_adapter, index, k * 1000000 + j, values))
            index += 1
    parent = _parent_row(conn, job_id)
    child = _new_child(conn, job_id, CHILD_REFINE, "{} · round {} refine3d".format(parent["NAME"], state["round"] + 1), parent)
    state.update({"phase": "refine", "child_job_id": child, "child_task_count": len(tasks), "child_done": 0, "pending_output_stars": outputs,
                  "refinement_jobs_this_round": jobs})
    _log(project_id, job_id, "Running refinement round {:2d} of {:2d} ({} search, {:.1f} Å) -> Refinement #{} — child job {}".format(
        state["round"] + 1, state["rounds"], "global" if s["global"] else "local", s["high_resolution_limit_a"], rid, child))
    _runtime.submit_child(project_id, child, refine3d_adapter, tasks, _profile(state["refinement_profile"]))


def _prepare_references_then_refine(conn, project_id, job_id, state):
    """What happens to the references between a merge and the next refine3d:
    Blush (its own phase, on a thread) when asked for, then masking."""
    if state["settings"]["use_blush"]:
        _blush_then_mask(conn, project_id, job_id, state)
    else:
        _mask_then_refine(conn, project_id, job_id, state)


def blushed_file(project_id, reference_file):
    """Where the Blush companion of a volume asset lives (blush_phase.assets_companion)."""
    return blush_phase.assets_companion(project_id, reference_file)


def _round_statistics(conn, state, k):
    """Class k's statistics for the volumes the state currently points at: the
    merge's file when those are this round's volumes, else the input refinement's."""
    stats_files = state.get("pending_stats_files") or []
    if stats_files and k < len(stats_files) and os.path.isfile(stats_files[k]) and state["reference_files"] == state.get("pending_volume_files"):
        return read_statistics(stats_files[k])
    return refinements.load_statistics(conn, state["input_refinement_id"], k + 1)


def _half_maps(state, k):
    halves = state.get("pending_half_maps") or []
    return halves[k] if k < len(halves) else []


# How this driver describes itself to blush_phase: the companions live beside the volume
# assets (Assets/Volumes/Blushed), the FSC is the round's or the input refinement's.
_BLUSH = blush_phase.Host(module=sys.modules[__name__],
                          continue_pre=lambda conn, project_id, job_id, state: _mask_then_refine(conn, project_id, job_id, state),
                          continue_post=lambda conn, project_id, job_id, state: _cycle(conn, project_id, job_id, state),
                          pixel_size=lambda state: state["pixel_size"],
                          mask_radius=lambda state: state["settings"]["mask_radius_a"],
                          statistics=_round_statistics,
                          companion_path=lambda project_id, state, ref: blush_phase.assets_companion(project_id, ref),
                          half_maps=_half_maps,
                          unfiltered=lambda state: state["settings"]["blush_unfiltered"])


def _blush_then_mask(conn, project_id, job_id, state):
    """Before a round (blush_phase.PRE): each class refines against its reference's Blush companion, made now if there is none."""
    blush_phase.start(_BLUSH, conn, project_id, job_id, state, blush_phase.PRE)


def _blush_after_merge(conn, project_id, job_id, state):
    """After a round's merge (blush_phase.POST): the new volume's companion from the half maps, for the next round or run."""
    blush_phase.start(_BLUSH, conn, project_id, job_id, state, blush_phase.POST)


def _mask_then_refine(conn, project_id, job_id, state):
    """DoMasking(): the supplied mask (Multiply3DMaskerThread) or the auto-mask, then refine."""
    s = state["settings"]
    scratch = Path(state["scratch"])
    if s["use_mask"] and state.get("mask_file"):
        _log(project_id, job_id, "Masking reference reconstruction with selected mask")
        mask, _ps = volumes.read_mrc_volume(state["mask_file"])
        masked = []
        for k, ref in enumerate(state["reference_files"]):
            vol, ps = volumes.read_mrc_volume(ref)
            ps = ps or state["pixel_size"]
            out = str(scratch / (Path(ref).stem + "_masked_class{}.mrc".format(k + 1)))
            # No filter (None) when the low-pass is off: the outside weight still applies, as in the GUI, where the thread's
            # pixel size / 0 gives ApplyMask an infinite radius; a radius of 0 would have it ignore the weight instead.
            low_pass = state["pixel_size"] / s["mask_filter_resolution_a"] if s["low_pass_outside_mask"] and s["mask_filter_resolution_a"] > 0 else None
            volumes.write_mrc_volume(out, volumes.apply_mask(vol, mask, s["mask_edge_a"] / state["pixel_size"], s["outside_mask_weight"], low_pass, state["pixel_size"] / 40.0), state["pixel_size"])
            masked.append(out)
        state["reference_files"] = masked
    elif s["auto_mask"]:
        _log(project_id, job_id, "Automasking reference reconstruction")
        masked = []
        for k, ref in enumerate(state["reference_files"]):
            vol, _ps = volumes.read_mrc_volume(ref)
            out = str(scratch / (Path(ref).stem + "_masked_class{}.mrc".format(k + 1)))
            volumes.write_mrc_volume(out, volumes.auto_mask(vol, state["pixel_size"], state["particle_size"] * 0.75), state["pixel_size"])
            masked.append(out)
        state["reference_files"] = masked
    _launch_refinement(conn, project_id, job_id, state)


def _merge_output_stars(conn, state):
    n_classes = state["number_of_classes"]
    jobs = _required_count(state, "refinement_jobs_this_round")
    outputs = state.get("pending_output_stars") or []
    # The round's input rows are the tables in scratch (the input refinement's at the start, each round's merged
    # rows after that -- what the record step writes to the database), read in a couple of seconds for eight
    # classes; the same rows from REFINEMENT_RESULT_<id>_<k> took 16 s a class at 3.3 million particles.
    stored = _load_rows(state, "output")
    class_rows = []
    for k in range(1, n_classes + 1):
        paths = [outputs[(k - 1) * jobs + j] if (k - 1) * jobs + j < len(outputs) else None for j in range(jobs)]
        try:
            class_rows.append(starfile.merge_task_outputs(stored[k - 1], paths))
        except ValueError as exc:
            raise ValueError("{} (class {})".format(exc, k))
    return class_rows


def _progress_percent(state):
    total = float(state["rounds"]) + (1.0 if state["start_with_reconstruction"] else 0.0)
    done = float(state["round"]) + (1.0 if state["start_with_reconstruction"] and not state["initial"] else 0.0)
    frac = float(state.get("child_done", 0)) / float(max(state.get("child_task_count", 1), 1))
    phase = state.get("phase")
    if phase == "initial_recon":
        done = 0.7 * frac
    elif phase == "initial_merge":
        done = 0.7 + 0.3 * frac
    elif phase == "refine":
        done += 0.6 * frac
    elif phase == "recon":
        done += 0.6 + 0.3 * frac
    elif phase == "merge":
        done += 0.9 + 0.1 * frac
    elif phase == "finished":
        done = total
    return max(0, min(100, int(100.0 * done / max(total, 1.0))))


def progress_info(state):
    if not state:
        return {}
    total = int(state["rounds"]) + (1 if state["start_with_reconstruction"] else 0)
    finished = [h["finished_at"] for h in state.get("history", []) if h.get("finished_at")]
    return {"task_count": total, "tasks_done": len(finished), "first_task_finished_at": finished[0] if finished else None,
            "last_task_finished_at": finished[-1] if finished else None, "round": state["round"], "rounds": state["rounds"], "phase": state["phase"]}



def _required_count(state, key):
    """A count the launch step recorded for the merge step. Missing means the
    state was overwritten in between (or this is a run from before the key
    existed): failing is right, since defaulting to 1 would merge the wrong
    files -- one dump file of N, or class k's rows from class 0's task k."""
    value = state.get(key)
    if not value:
        raise ValueError("the job's state has no {}; the round cannot be merged".format(key))
    return int(value)

def child_progress(conn, parent_id, child_id, done_count, task_count):
    """One task of the running child done: its two counters written into
    STATE_JSON in place (json_set), not load-modify-save -- this runs on the
    runner's thread while _child_finished() may be rewriting the whole state
    under the driver's lock, and a whole-state write here could put back a
    stale copy of everything else (child ids, file lists, counts)."""
    with conn:
        if task_count:
            n = conn.execute("UPDATE JOBS SET STATE_JSON = json_set(STATE_JSON, '$.child_done', ?, '$.child_task_count', ?) "
                             "WHERE JOB_ID = ? AND json_extract(STATE_JSON, '$.child_job_id') = ?",
                             (int(done_count), int(task_count), parent_id, child_id)).rowcount
        else:
            n = conn.execute("UPDATE JOBS SET STATE_JSON = json_set(STATE_JSON, '$.child_done', ?) "
                             "WHERE JOB_ID = ? AND json_extract(STATE_JSON, '$.child_job_id') = ?",
                             (int(done_count), parent_id, child_id)).rowcount
    if n:
        state = _load_state(conn, parent_id)
        if state:
            with conn:
                conn.execute("UPDATE JOBS SET PROGRESS=? WHERE JOB_ID=?", (_progress_percent(state), parent_id))


def child_finished(project_id, child_row, status, error=None):
    threading.Thread(target=_child_finished_cleared, args=(project_id, child_row["JOB_ID"], child_row["PARENT_JOB_ID"], status, error),
                     daemon=True, name="refine3d-" + child_row["PARENT_JOB_ID"]).start()


def _child_finished_cleared(project_id, child_id, parent_id, status, error):
    """_child_finished() with the parent's progress note cleared however it ends:
    the bookkeeping it does notes its phases in progress_store for the Jobs tab."""
    try:
        _child_finished(project_id, child_id, parent_id, status, error)
    finally:
        progress_store.clear(parent_id)


def _child_finished(project_id, child_id, parent_id, status, error):
    with _job_lock(parent_id):
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
                _advance(conn, project_id, parent_id, state)
            except Exception as exc:  # noqa: BLE001
                _finish(conn, project_id, parent_id, state, "failed", "could not continue: {}".format(exc))
        finally:
            conn.close()


def _advance(conn, project_id, parent_id, state):
    """ProcessAllJobsFinished() + CycleRefinement()."""
    phase = state["phase"]
    s = state["settings"]
    if phase == "refine":
        progress_store.note(parent_id, 0, 0, "merging the round's results")
        class_rows = _merge_output_stars(conn, state)
        if s["refine_occupancies"]:
            update_occupancies(class_rows)
        _store_rows(state, "output", class_rows)
        for p in state.get("pending_output_stars") or []:
            try:
                os.remove(p)
            except OSError:
                pass
        _launch_reconstruction(conn, project_id, parent_id, state)
    elif phase in ("initial_recon", "recon"):
        _launch_merge(conn, project_id, parent_id, state)
    elif phase in ("initial_merge", "merge"):
        progress_store.note(parent_id, 0, 0, "writing the refinement")
        _record_round(conn, project_id, parent_id, state)
        for p in Path(state["scratch"]).glob("dump_file_*.dump"):
            try:
                p.unlink()
            except OSError:
                pass
        if s["use_blush"]:
            _blush_after_merge(conn, project_id, parent_id, state)
        else:
            _cycle(conn, project_id, parent_id, state)
    else:
        raise ValueError("unexpected phase {!r}".format(phase))
    if state["phase"] != "finished":
        _save(conn, parent_id, state, _progress_percent(state))


def _record_round(conn, project_id, parent_id, state):
    """The MERGE branch of ProcessAllJobsFinished(): volume assets,
    reconstruction records, the package's references, and the refinement
    itself (or, for the initial reconstruction, the input refinement's
    statistics)."""
    s = state["settings"]
    classes = state["number_of_classes"]
    rid = state["output_refinement_id"]
    for p in state["pending_volume_files"]:
        if not os.path.isfile(p):
            raise ValueError("merge3d did not write {}".format(p))
    class_rows = _load_rows(state, "output")
    stats = [read_statistics(p) if os.path.isfile(p) else [] for p in state["pending_stats_files"]]
    avgs = pooled_part_ssnr(stats, class_rows)
    volume_ids, recon_ids, est_res = [], [], []
    previous_refs = list(state["reference_asset_ids"])
    for k in range(classes):
        if state["initial"]:
            name = "Start Params #{} - Class #{}".format(rid, k + 1)
        else:
            name = "{} #{} (Rnd. {}) - Class #{}".format("Global" if s["global"] else "Local", rid, state["round"] + 1, k + 1)
        recon_id = refinements.next_reconstruction_id(conn)
        vid = volumes.add_volume_asset(conn, name, state["pending_volume_files"][k], state["pixel_size"], state["box_size"], state["box_size"], state["box_size"],
                                       reconstruction_job_id=recon_id)
        refinements.add_reconstruction_job(conn, recon_id, state["package_id"], rid, "", s["inner_mask_radius_a"], s["mask_radius_a"],
                                           s["reconstruction_resolution_limit_a"], s["score_to_weight_constant"], s["adjust_score_for_defocus"],
                                           s["autocrop_images"], False, s["apply_blurring"], s["smoothing_factor"], k + 1, vid)
        refinements.set_current_reference(conn, state["package_id"], k + 1, vid)
        volume_ids.append(vid)
        recon_ids.append(recon_id)
        est_res.append(refinements.estimated_resolution(stats[k], state["pixel_size"]) if stats[k] else None)
    state["reference_files"] = list(state["pending_volume_files"])
    state["reference_asset_ids"] = volume_ids
    state["volume_ids"] = volume_ids
    volumes.prepare_orth_views(state["reference_files"], state["settings"]["mask_radius_a"], lambda t: _log(project_id, parent_id, t, "warning"))
    for k in range(classes):
        if classes > 1:
            _log(project_id, parent_id, "Est. Res. Class {:2d} = {} Å ({:.2f} %)".format(k + 1, "{:.2f}".format(est_res[k]) if est_res[k] else "n/a", avgs[k]))
        else:
            _log(project_id, parent_id, "Est. Res. = {} Å".format("{:.2f}".format(est_res[k]) if est_res[k] else "n/a"))
    if state["initial"]:
        # The input refinement gains real statistics and points at its reconstruction.
        with conn:
            for k in range(classes):
                refinements.write_statistics(conn, rid, k + 1, stats[k])
                conn.execute("UPDATE REFINEMENT_LIST SET RESOLUTION_STATISTICS_ARE_GENERATED=0 WHERE REFINEMENT_ID=?", (rid,))
        for k in range(classes):
            refinements.update_details(conn, rid, k + 1, RECONSTRUCTED_VOLUME_ASSET_ID=volume_ids[k], RECONSTRUCTION_ID=recon_ids[k],
                                       ESTIMATED_RESOLUTION=est_res[k] or 0.0)
        state["history"].append({"round": 0, "refinement_id": rid, "label": "Iter. #0", "estimated_resolution": est_res, "finished_at": now_iso()})
        return
    details = []
    for k in range(classes):
        avg_occ = refinements.average_occupancy(class_rows[k])
        details.append({
            "REFERENCE_VOLUME_ASSET_ID": previous_refs[k], "LOW_RESOLUTION_LIMIT": s["low_resolution_limit_a"], "HIGH_RESOLUTION_LIMIT": s["high_resolution_limit_a"],
            "MASK_RADIUS": s["mask_radius_a"], "SIGNED_CC_RESOLUTION_LIMIT": s["signed_cc_resolution_limit_a"], "GLOBAL_RESOLUTION_LIMIT": s["high_resolution_limit_a"],
            "GLOBAL_MASK_RADIUS": s["global_mask_radius_a"], "NUMBER_RESULTS_TO_REFINE": int(s["number_of_results_to_refine"]),
            "ANGULAR_SEARCH_STEP": s["angular_step_deg"], "SEARCH_RANGE_X": s["search_range_x_a"], "SEARCH_RANGE_Y": s["search_range_y_a"],
            "CLASSIFICATION_RESOLUTION_LIMIT": s["classification_high_resolution_limit_a"], "SHOULD_FOCUS_CLASSIFY": 1 if s["focused_classification"] else 0,
            "SPHERE_X_COORD": s["sphere_x_a"], "SPHERE_Y_COORD": s["sphere_y_a"], "SPHERE_Z_COORD": s["sphere_z_a"], "SPHERE_RADIUS": s["sphere_radius_a"],
            "SHOULD_REFINE_CTF": 1 if s["refine_ctf"] else 0, "DEFOCUS_SEARCH_RANGE": s["defocus_search_range_a"], "DEFOCUS_SEARCH_STEP": s["defocus_search_step_a"],
            "AVERAGE_OCCUPANCY": avg_occ, "ESTIMATED_RESOLUTION": est_res[k] or 0.0, "RECONSTRUCTED_VOLUME_ASSET_ID": volume_ids[k], "RECONSTRUCTION_ID": recon_ids[k],
            "SHOULD_AUTOMASK": 1 if s["auto_mask"] else 0, "SHOULD_REFINE_INPUT_PARAMS": 0 if s["also_refine_input"] else 1,   # cisTEM stores AlsoRefineInputNoRadio: 1 = do not
            "SHOULD_USE_SUPPLIED_MASK": 1 if s["use_mask"] else 0, "MASK_ASSET_ID": state.get("mask_asset_id", -1), "MASK_EDGE_WIDTH": s["mask_edge_a"],
            "OUTSIDE_MASK_WEIGHT": s["outside_mask_weight"], "SHOULD_LOWPASS_OUTSIDE_MASK": 1 if s["low_pass_outside_mask"] else 0,
            "MASK_FILTER_RESOLUTION": s["mask_filter_resolution_a"],
        })
    ref = {"refinement_id": rid, "refinement_package_asset_id": state["package_id"],
           "name": "{} #{}".format("Global Search" if s["global"] else "Local Refinement", rid), "resolution_statistics_are_generated": False,
           "starting_refinement_id": state["input_refinement_id"], "number_of_particles": state["number_of_particles"], "number_of_classes": classes,
           "resolution_statistics_box_size": state["box_size"], "resolution_statistics_pixel_size": state["pixel_size"], "percent_used": s["percent_used"],
           "job_id": parent_id}
    refinements.add_refinement(conn, ref, class_rows, stats, details, symmetry=state["symmetry"])
    state["history"].append({"round": state["round"] + 1, "refinement_id": rid, "label": "Iter. #{}".format(state["round"] + 1), "estimated_resolution": est_res,
                             "average_sigma": average_sigma(class_rows), "finished_at": now_iso()})
    _log(project_id, parent_id, "Refinement #{} written ({}); volume asset{} {}".format(rid, ref["name"], "" if classes == 1 else "s", ", ".join("#{}".format(v) for v in volume_ids)))


ACTIONS = {"finish": "Finish After This Round"}


def available_actions(state):
    """Finish early: the run stops as completed once the round in progress
    has written its refinement, instead of going on to the next."""
    if not state or state.get("phase") in (None, "finished") or state.get("finish_requested") or state.get("round", 0) + 1 >= state.get("rounds", 1):
        return []
    return [{"name": "finish", "label": ACTIONS["finish"]}]


def perform_action(conn, project_id, parent_id, name):
    if name != "finish":
        raise ValueError("unknown action {!r}".format(name))
    state = _load_state(conn, parent_id)
    if not available_actions(state):
        raise ValueError("Finish is not available right now")
    state["finish_requested"] = True
    with conn:   # the flag alone, in place: a whole-state write from this request thread could put back a stale step
        conn.execute("UPDATE JOBS SET STATE_JSON = json_set(STATE_JSON, '$.finish_requested', json('true')) WHERE JOB_ID = ?", (parent_id,))
    _log(project_id, parent_id, "Finish requested: the run will stop once round {} has written its refinement".format(state["round"] + 1))


def _cycle(conn, project_id, parent_id, state):
    """CycleRefinement()."""
    if state["initial"]:
        state["initial"] = False
        _prepare_references_then_refine(conn, project_id, parent_id, state)
        return
    state["round"] += 1
    if state.get("finish_requested") and state["round"] < state["rounds"]:
        state["rounds"] = state["round"]
        _log(project_id, parent_id, "Finished at the user's request after {} round{}.".format(state["round"], "" if state["round"] == 1 else "s"))
    if state["round"] < state["rounds"]:
        state["input_refinement_id"] = state["output_refinement_id"]
        _prepare_references_then_refine(conn, project_id, parent_id, state)
        return
    state["phase"] = "finished"
    state["child_job_id"] = None
    _log(project_id, parent_id, "All refinement cycles are finished!")
    _finish(conn, project_id, parent_id, state, "completed", None)


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
    metrics = {"rounds_run": state.get("round", 0), "refinement_ids": [h["refinement_id"] for h in state.get("history", []) if h.get("round")],
               "volume_ids": state.get("volume_ids", []), "final_refinement_id": state.get("output_refinement_id") if status == "completed" else None}
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
    if state is not None and state.get("phase") in blush_phase.PHASES and blush_phase.cancel(parent_id):
        _log(project_id, parent_id, "cancel requested; stopping Blush")   # the Blush thread finishes the job as cancelled at its next block
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
        if state.get("phase") in blush_phase.PHASES:
            blush_phase.resume(_BLUSH, project_id, parent_row["JOB_ID"], state["phase"])
            return
        child_id = state.get("child_job_id")
        child = conn.execute("SELECT * FROM JOBS WHERE JOB_ID=?", (child_id,)).fetchone() if child_id else None
        if child is None:
            _finish(conn, project_id, parent_row["JOB_ID"], state, "failed", "Server restarted and the running step could not be found")
            return
        _log(project_id, parent_row["JOB_ID"], "server restarted during {} (round {}); waiting on child job {}".format(state["phase"], state["round"] + 1, child_id))
        if child["STATUS"] in ("completed", "failed", "cancelled"):
            child_finished(project_id, child, child["STATUS"], child["ERROR"])
    finally:
        conn.close()


def live_result(conn, row):
    """The Jobs tab's Latest Result: the newest refinement's FSC curve and
    the current reconstruction (ShowRefinementResultsPanel)."""
    state = _load_state(conn, row["JOB_ID"])
    if not state or not state.get("history"):
        return None
    last = state["history"][-1]
    rid = last["refinement_id"]
    stats = refinements.load_statistics(conn, rid, 1)
    files = [p for p in (state.get("reference_files") or []) if p and os.path.isfile(p)]
    return {
        "kind": "refine3d", "task_index": len(state["history"]) * 10 + (1 if state.get("phase") == "finished" else 0),
        "label": last["label"], "refinement_id": rid, "phase": state["phase"], "round": state["round"], "rounds": state["rounds"],
        "number_of_classes": state["number_of_classes"], "has_picture": bool(files), "current_volume": files[0] if files else None,
        "estimated_resolution": last.get("estimated_resolution"), "history": state["history"],
        "fsc": [{"resolution": s["resolution"], "fsc": s["fsc"], "part_fsc": s["part_fsc"]} for s in stats if s["resolution"]],
        "angular_distribution": refinements.load_angular_distribution(conn, rid, 1),
        "pixel_size": state["pixel_size"], "volume_ids": state.get("volume_ids", []), "package_name": state.get("package_name"),
    }


def volume_file(conn, row, class_index=0, output_number=None):
    """The current reconstruction's file for the live view's Download: (path, label) or None."""
    state = _load_state(conn, row["JOB_ID"])
    if not state:
        return None
    files = [p for p in (state.get("reference_files") or []) if p and os.path.isfile(p)]
    if not files:
        return None
    k = min(class_index, len(files) - 1)
    last = (state.get("history") or [{}])[-1]
    label = "refinement{}".format(last.get("refinement_id", "")) if last.get("refinement_id") is not None else "current"
    return files[k], label + ("_class{}".format(k + 1) if len(files) > 1 else "")


def current_picture(conn, row, class_index=0):
    state = _load_state(conn, row["JOB_ID"])
    if not state:
        return None
    files = [p for p in (state.get("reference_files") or []) if p and os.path.isfile(p)]
    if not files:
        return None
    path = files[min(class_index, len(files) - 1)]
    png, meta = volumes.orthogonal_views_png_cached(path, state["settings"]["mask_radius_a"])
    return png, meta, path
