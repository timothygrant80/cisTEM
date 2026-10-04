"""Auto Refine: cisTEM's AutoRefinementManager (src/gui/AutoRefine3dPanel.cpp)
as a server-side driver, in the parent/child shape of refine3d.py.

The user gives a package, a starting reference and an initial resolution
limit; everything else is decided round by round the way the manager does
it:

* The input parameters are the package's first refinement with its Euler
  angles randomised, shifts zeroed and default statistics -- the reference
  is what carries the information, not the parameters -- and the first
  round is a global search of every particle at the initial limit (one
  class only; with several classes the results are copied to every class
  with random occupancies, so classification starts from noise).
* Each later round refines a subset: the percentage of particles needed for
  8000 * exp(75 / res^2) asymmetric units at the current resolution, never
  below the starting percentage, and refine3d is handed three times that
  so reconstruct3d can keep the best-scoring third. When the resolution
  stalls the ceiling is tripled; under 7 A everything is used.
* Per particle, refine3d is told to search globally (image_is_active 0) or
  locally (1): never-aligned particles always search; otherwise a random
  draw whose likelihood grows with the resolution gained since that
  particle's last global search and shrinks with how many it has had
  (SetupRefinementJob()'s "very arbitrary" formula, kept as is).
* The next round's high-resolution limit per class comes from the FSC of
  the round just finished: the larger of "a few shells past the current
  limit" and "the 0.5 crossing, backed off by the mask's bleed shells",
  never coarser than the current limit (CycleRefinement()). The
  reconstruction is limited to a few shells past the alignment limit,
  never relaxed from one round to the next, and unlimited in the final
  round.
* It stops when, after at least 5 rounds (10 with several classes), all
  particles are in use, the occupancies have settled and the estimated
  resolution has not improved over the last two rounds; one more round on
  every particle with an unlimited reconstruction is then the final one.

Every round writes a refinement "Auto #<id> - Round <n>" through
refinements.py and one volume asset per class ("Auto #<id> (Rnd. <n>) -
Class #<k>", Assets/Volumes/volume_<id>_<k>.mrc) that becomes the
package's current reference, so the Refine 3D results page and a following
manual refinement pick up where it stopped. The per-particle bookkeeping
(global-alignment counts, rounds since, resolution of the last one) lives
in the job's scratch directory beside the star files, not in STATE_JSON.

Children carry the same abinitio_refine3d / abinitio_reconstruct3d /
abinitio_merge3d stages as the other 3D drivers.
"""
import json
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
import refinements
import starfile
import volumes
from abinitio import (_new_child, _parent_row, _profile, _task, _load_state, _save, now_iso, particle_range,
                      default_statistics, write_statistics, read_statistics, update_occupancies, pooled_part_ssnr,
                      average_sigma, angular_step as calculate_angular_step)
from refine3d import _num, _flag, PLEASE_CREATE_PACKAGE_MESSAGE, CHILD_REFINE, CHILD_RECON, CHILD_MERGE
from stages import merge3d, reconstruct3d, refine3d as refine3d_adapter
import symmetry as symmetry_module

STAGE = "auto_refine3d"

# AutoRefine3DPanel::SetDefaults(); the size-dependent ones follow the package.
DEFAULTS = {
    "high_resolution_limit_a": 20.0, "inner_mask_radius_a": 0.0, "number_of_results_to_refine": 20,
    "autocrop_images": False, "apply_blurring": False, "smoothing_factor": 1.0, "autocenter": True,
    "use_mask": False, "auto_mask": True, "mask_edge_a": 10.0, "outside_mask_weight": 0.0, "low_pass_outside_mask": False,
    "mask_filter_resolution_a": 20.0,
    # Blush regularisation of the reference between rounds (blush.py / blush_phase.py); not in cisTEM's panel.
    "use_blush": False, "blush_input": "Unfiltered half-map sum",
}
ASYM_UNITS_CONSTANT = 8000.0  # estimated_required_asym_units = 8000 * exp(75 / res^2)


def package_defaults(pkg):
    """SetDefaults(): the numbers that follow the package's largest dimension."""
    size = float(pkg["PARTICLE_SIZE"] or 150.0)
    return {"low_resolution_limit_a": round(min(size * 1.5, 300.0), 2), "mask_radius_a": round(size * 0.65, 2),
            "global_mask_radius_a": round(size * 0.8, 2), "search_range_x_a": round(size * 0.15, 2), "search_range_y_a": round(size * 0.15, 2)}


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
    if s["use_mask"]:
        s["auto_mask"] = False  # OnUseMaskCheckBox(): a supplied mask switches auto-masking off
        s["autocenter"] = False
    s["blush_unfiltered"] = not s["blush_input"].strip().lower().startswith("filtered")
    return s


def asymmetric_units(symbol):
    """ReturnNumberofAsymmetricUnits()."""
    try:
        kind, n = symmetry_module.parse_symbol(symbol)
    except ValueError:
        return 1
    return {"C": n, "D": 2 * n, "T": 12, "O": 24, "I": 60}.get(kind, 1)


def percent_for_resolution(resolution_a, number_of_particles, symbol, number_of_classes):
    """The percentage of particles holding 8000 * exp(75 / res^2) asymmetric
    units per class (BeginRefinementCycle() / CycleRefinement()), capped at 100."""
    number_of_asym_units = max(1, number_of_particles) * asymmetric_units(symbol)
    wanted = int(round(ASYM_UNITS_CONSTANT * math.exp(75.0 / (resolution_a ** 2)))) * number_of_classes
    return min(100.0, wanted / float(number_of_asym_units) * 100.0)


# ---------------------------------------------------------------------------
# ResolutionStatistics helpers over the [{shell, resolution, fsc, part_fsc,
# ...}] lists refinements.py / abinitio.py pass around (index 0 = shell 1,
# the first shell cisTEM's files hold).
# ---------------------------------------------------------------------------

def resolution_at(stats, threshold, pixel_size, use_part_fsc=False):
    """Return0p5Resolution() / ReturnEstimatedResolution(): the resolution
    midway between the shells where the (part) FSC first drops below
    `threshold`, never better than Nyquist."""
    key = "part_fsc" if use_part_fsc else "fsc"
    est = 0.0
    for i in range(1, len(stats)):
        if stats[i][key] < threshold:
            est = (stats[i - 1]["resolution"] + stats[i]["resolution"]) / 2.0
            break
    return max(est, 2.0 * float(pixel_size))


def resolution_n_shells_after(stats, wanted_resolution, number_of_shells, pixel_size):
    """ReturnResolutionNShellsAfter(): 0 if no shell is finer than the wanted
    resolution; Nyquist if the step runs off the end."""
    shell = next((i for i in range(1, len(stats)) if stats[i]["resolution"] < wanted_resolution), -1)
    if shell == -1:
        return 0.0
    shell += number_of_shells
    if shell >= len(stats):
        return 2.0 * float(pixel_size)
    return stats[shell]["resolution"]


def resolution_n_shells_before(stats, wanted_resolution, number_of_shells):
    """ReturnResolutionNShellsBefore(): 0 when it would step before shell 1."""
    shell = next((i for i in range(len(stats) - 1, 0, -1) if stats[i]["resolution"] > wanted_resolution), -1)
    if shell == -1:
        return 0.0
    shell -= number_of_shells
    if shell < 1:
        return 0.0
    return stats[shell]["resolution"]


def next_class_limit(stats, current_limit, box_size, pixel_size, mask_radius_a):
    """CycleRefinement()'s per-class high-resolution limit for the next
    round, from the round's statistics."""
    bleed_shells = int(math.ceil(box_size / (mask_radius_a / pixel_size)))
    res_0p5 = resolution_at(stats, 0.5, pixel_size, False)
    part_0p5 = resolution_at(stats, 0.5, pixel_size, True)
    minus_bleed = resolution_n_shells_before(stats, res_0p5, bleed_shells + 1) or current_limit
    part_minus_bleed = resolution_n_shells_before(stats, part_0p5, bleed_shells + 1) or current_limit
    average_minus_bleed = (minus_bleed + part_minus_bleed) * 0.5
    min_shells_after = resolution_n_shells_after(stats, current_limit, box_size // 15, pixel_size)
    res = max(min_shells_after, average_minus_bleed)
    return min(res, current_limit)


def choose_global(rounds_run, last_resolution, lowest_alignment_res, last_global_res, globals_so_far, rounds_since_global,
                  reference_contains_all, final_round, rng):
    """SetupRefinementJob()'s per-particle decision: refine3d's image_is_active
    (0 = global search, 1 = local refinement) for one particle."""
    if globals_so_far == 0:
        return 0
    if final_round or rounds_since_global == 0:
        return 1
    if rounds_run == 0:
        do_global = True
    elif last_resolution < 5.0 and lowest_alignment_res <= 8.0 and last_global_res > 9.0 and reference_contains_all and rounds_run > 2:
        do_global = True
    else:
        round_adjust = max(1.0, (globals_so_far - math.floor(rounds_since_global / 3.0)) ** 2)
        res_adjust = last_global_res - lowest_alignment_res
        if last_global_res <= 5.0:
            likelihood = -5.0
        elif res_adjust == 0.0:
            likelihood = 0.0
        else:
            likelihood = lowest_alignment_res ** 2 / ((1000.0 / res_adjust) * round_adjust)
        do_global = abs(rng.uniform(-1.0, 1.0)) < likelihood
    return 0 if do_global else 1


def should_stop(resolution_per_round, max_percent_used, change_in_occupancies, number_of_classes):
    """CycleRefinement()'s convergence test."""
    min_rounds = 5 if number_of_classes == 1 else 10
    if len(resolution_per_round) < min_rounds or max_percent_used <= 99.0 or change_in_occupancies >= 1.0:
        return False
    round_resolution = resolution_per_round[-3]
    return all(r >= round_resolution - 0.001 for r in resolution_per_round[-2:])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(conn, params):
    package_id = params.get("refinement_package_id")
    if package_id in (None, ""):
        raise ValueError(PLEASE_CREATE_PACKAGE_MESSAGE)
    pkg = conn.execute("SELECT * FROM REFINEMENT_PACKAGE_ASSETS WHERE REFINEMENT_PACKAGE_ASSET_ID=?", (int(package_id),)).fetchone()
    if pkg is None:
        raise ValueError("refinement package {} does not exist".format(package_id))
    if not pkg["STACK_FILENAME"] or not os.path.isfile(pkg["STACK_FILENAME"]):
        raise ValueError("the particle stack {} is missing".format(pkg["STACK_FILENAME"]))
    first = conn.execute("SELECT REFINEMENT_ID FROM REFINEMENT_PACKAGE_REFINEMENTS_LIST_{} ORDER BY REFINEMENT_NUMBER LIMIT 1".format(int(package_id))).fetchone()
    ref = refinements.refinement_row(conn, first[0]) if first else None
    if ref is None:
        raise ValueError("the package has no starting parameters")
    vid = params.get("reference_volume_id")
    if vid in (None, ""):
        raise ValueError("Pick a starting reference volume.")
    vol = conn.execute("SELECT * FROM VOLUME_ASSETS WHERE VOLUME_ASSET_ID=?", (int(vid),)).fetchone()
    if vol is None or not vol["FILENAME"] or not os.path.isfile(vol["FILENAME"]):
        raise ValueError("the starting reference volume is missing")
    if vol["X_SIZE"] != pkg["STACK_BOX_SIZE"] or abs(float(vol["PIXEL_SIZE"] or 0) - float(pkg["OUTPUT_PIXEL_SIZE"] or 0)) > 0.01:
        raise ValueError("Error: The reference volume ({}, {}, {}; psize: {}) has different dimensions / pixel size from the input stack ({}; psize: {}).  This will currently not work.".format(
            vol["X_SIZE"], vol["Y_SIZE"], vol["Z_SIZE"], vol["PIXEL_SIZE"], pkg["STACK_BOX_SIZE"], pkg["OUTPUT_PIXEL_SIZE"]))
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
    if _num(params, "high_resolution_limit_a", DEFAULTS["high_resolution_limit_a"]) <= 0:
        raise ValueError("the initial resolution limit must be positive")
    if _flag(params, "use_blush", False):
        info = blush.availability()
        if not info["available"]:
            raise ValueError("Blush cannot run on this server: {}".format(info["reason"]))
    return pkg, ref, vol, mask


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
    d = db.project_dir(project_id) / "Scratch" / "AutoRefine3D" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _class_star(state, tag, k):
    return str(Path(state["scratch"]) / "refinement_{}_class{}.cistem".format(tag, k + 1))


def _class_stats(state, tag, k):
    return str(Path(state["scratch"]) / "statistics_{}_class{}.txt".format(tag, k + 1))


def _store_rows(state, tag, class_rows):
    for k, rows in enumerate(class_rows):
        starfile.write_params(_class_star(state, tag, k), refinements.as_table(rows))


def _load_rows(state, tag):
    """The per-class parameter tables stored under `tag`."""
    return [starfile.read_params(_class_star(state, tag, k), as_table=True) for k in range(state["number_of_classes"])]


def _store_stats(state, tag, class_stats):
    for k, stats in enumerate(class_stats):
        write_statistics(_class_stats(state, tag, k), stats, state["pixel_size"])


def _load_stats(state, tag):
    return [read_statistics(_class_stats(state, tag, k)) for k in range(state["number_of_classes"])]


def _tracking_path(state):
    return str(Path(state["scratch"]) / "tracking.npz")


def _load_tracking(state):
    """The per-particle bookkeeping (global-alignment count, rounds since, resolution of the last one) as arrays."""
    with np.load(_tracking_path(state)) as data:
        return {k: data[k] for k in ("globals", "since_global", "last_global_res")}


def _store_tracking(state, tracking):
    np.savez(_tracking_path(state), globals=np.asarray(tracking["globals"], dtype=np.int64), since_global=np.asarray(tracking["since_global"], dtype=np.int64),
             last_global_res=np.asarray(tracking["last_global_res"], dtype=np.float64))



def start(conn, project_id, job_id, params, profile):
    """The first step, under the driver's lock: the child it launches can end
    before this returns (a controller that fails to launch does so inside
    submit()), and _child_finished() must then find the parent's state saved,
    not a parent that never hears of it and stays "running" forever."""
    with _job_lock(job_id):
        return _start(conn, project_id, job_id, params, profile)

def _start(conn, project_id, job_id, params, profile):
    """BeginRefinementCycle()."""
    pkg, ref, vol, mask = validate(conn, params)
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
    sym = pkg["SYMMETRY"] or "C1"
    start_percent = percent_for_resolution(s["high_resolution_limit_a"], n, sym, classes)
    state = {
        "phase": None, "round": 0, "settings": s,
        "package_id": pkg["REFINEMENT_PACKAGE_ASSET_ID"], "package_name": pkg["NAME"], "stack_filename": pkg["STACK_FILENAME"],
        "pixel_size": float(pkg["OUTPUT_PIXEL_SIZE"] or 1.0), "box_size": int(pkg["STACK_BOX_SIZE"]), "invert_contrast": bool(pkg["STACK_HAS_WHITE_PROTEIN"]),
        "symmetry": sym, "molecular_weight": float(pkg["MOLECULAR_WEIGHT"] or 300.0), "particle_size": float(pkg["PARTICLE_SIZE"] or 150.0),
        "number_of_particles": n, "number_of_classes": classes,
        "refinement_profile": profile["name"], "refinement_jobs": int(profile["total_jobs"]),
        "reconstruction_profile": recon_profile["name"], "reconstruction_jobs": int(recon_profile["total_jobs"]),
        "input_refinement_id": ref["REFINEMENT_ID"], "output_refinement_id": None,
        "starting_reference_id": vol["VOLUME_ASSET_ID"], "reference_files": [vol["FILENAME"]] * classes, "reference_asset_ids": [vol["VOLUME_ASSET_ID"]] * classes,
        "mask_file": mask["FILENAME"] if mask is not None else None, "mask_asset_id": mask["VOLUME_ASSET_ID"] if mask is not None else -1,
        "class_high_res_limits": [s["high_resolution_limit_a"]] * classes, "high_res_limit_per_round": [s["high_resolution_limit_a"]],
        "resolution_per_round": [], "percent_used_per_round": [],
        "start_percent_used": start_percent, "current_percent_used": start_percent, "max_percent_used": start_percent,
        "last_round_reconstruction_resolution": None, "final_round": False, "reference_contains_all": False,
        "child_job_id": None, "child_task_count": 0, "child_done": 0, "history": [], "started_at": now_iso(), "scratch": str(scratch),
        "volume_ids": [],
    }
    # The input refinement: the package's first parameters with random
    # angles, zero shifts, equal occupancies and default statistics.
    rng = np.random.default_rng()
    class_rows = []
    for k in range(1, classes + 1):
        table = refinements.load_table(conn, ref["REFINEMENT_ID"], k)
        table["occupancy"] = 100.0 / classes
        table["phi"] = rng.uniform(-180.0, 180.0, len(table)); table["theta"] = rng.uniform(-180.0, 180.0, len(table)); table["psi"] = rng.uniform(-180.0, 180.0, len(table))
        table["x_shift"] = 0.0; table["y_shift"] = 0.0; table["score"] = 0.0; table["image_is_active"] = 1; table["sigma"] = 1.0
        class_rows.append(table)
    _store_rows(state, "input", class_rows)
    _store_stats(state, "input", [default_statistics(state["molecular_weight"], state["pixel_size"], state["box_size"]) for _ in range(classes)])
    _store_tracking(state, {"globals": [0] * n, "since_global": [0] * n, "last_global_res": [100.0] * n})
    with conn:
        conn.execute("UPDATE JOBS SET STATUS='running', STARTED_AT=?, PROGRESS=0 WHERE JOB_ID=?", (now_iso(), job_id))
    _log(project_id, job_id, "Auto Refine of {!r} from volume #{} {!r}: {} particles, {} class{}, initial limit {:.1f} Å, starting with {:.1f}% of the particles; refinement profile {!r}, reconstruction profile {!r}".format(
        pkg["NAME"], vol["VOLUME_ASSET_ID"], vol["NAME"], n, classes, "" if classes == 1 else "es", s["high_resolution_limit_a"], start_percent, profile["name"], recon_profile["name"]))
    _prepare_references_then_refine(conn, project_id, job_id, state)
    _save(conn, job_id, state)
    return state


def _prepare_references_then_refine(conn, project_id, job_id, state):
    """Between a merge and the next refine3d: Blush (its own phase) when asked for, then masking."""
    if state["settings"]["use_blush"]:
        blush_phase.start(_BLUSH, conn, project_id, job_id, state, blush_phase.PRE)
    else:
        _mask_then_refine(conn, project_id, job_id, state)


def _round_statistics(conn, state, k):
    """Class k's statistics for the volumes the state points at: this round's merge, else the input's (none for the starting reference)."""
    stats_files = state.get("pending_stats_files") or []
    if stats_files and k < len(stats_files) and os.path.isfile(stats_files[k]) and state["reference_files"] == state.get("pending_volume_files"):
        return read_statistics(stats_files[k])
    try:
        return _load_stats(state, "input")[k]
    except (IndexError, TypeError, OSError, ValueError):
        return []


def _half_maps(state, k):
    halves = state.get("pending_half_maps") or []
    return halves[k] if k < len(halves) else []


_BLUSH = blush_phase.Host(module=sys.modules[__name__],
                          continue_pre=lambda conn, project_id, job_id, state: _mask_then_refine(conn, project_id, job_id, state),
                          continue_post=lambda conn, project_id, job_id, state: _cycle(conn, project_id, job_id, state),
                          pixel_size=lambda state: state["pixel_size"],
                          mask_radius=lambda state: state["settings"]["mask_radius_a"],
                          statistics=_round_statistics,
                          companion_path=lambda project_id, state, ref: blush_phase.assets_companion(project_id, ref),
                          half_maps=_half_maps,
                          unfiltered=lambda state: state["settings"]["blush_unfiltered"])


def _mask_then_refine(conn, project_id, job_id, state):
    """DoMasking(): the supplied mask (Multiply3DMaskerThread) or the auto-mask, then refine."""
    s = state["settings"]
    scratch = Path(state["scratch"])
    if s["use_mask"] and state.get("mask_file"):
        _log(project_id, job_id, "Masking reference reconstruction with selected mask")
        mask, _ps = volumes.read_mrc_volume(state["mask_file"])
        masked = []
        for ref in state["reference_files"]:
            vol, _ps = volumes.read_mrc_volume(ref)
            out = str(scratch / (Path(ref).stem + "_masked.mrc"))
            low_pass = state["pixel_size"] / s["mask_filter_resolution_a"] if s["low_pass_outside_mask"] and s["mask_filter_resolution_a"] > 0 else 0.0
            volumes.write_mrc_volume(out, volumes.apply_mask(vol, mask, s["mask_edge_a"] / state["pixel_size"], s["outside_mask_weight"], low_pass, state["pixel_size"] / 40.0), state["pixel_size"])
            masked.append(out)
        state["reference_files"] = masked
    elif s["auto_mask"]:
        _log(project_id, job_id, "Automasking reference reconstruction")
        # As cisTEM's AutoMaskerThread::Entry: it is handed a filter resolution by
        # DoMasking() but never reads it -- the mask comes from ConvertToAutoMask's
        # fixed 7 A binning, which is what volumes.auto_mask does.
        masked = []
        for ref in state["reference_files"]:
            vol, _ps = volumes.read_mrc_volume(ref)
            out = str(scratch / (Path(ref).stem + "_masked.mrc"))
            volumes.write_mrc_volume(out, volumes.auto_mask(vol, state["pixel_size"], s["mask_radius_a"]), state["pixel_size"])
            masked.append(out)
        state["reference_files"] = masked
    _launch_refinement(conn, project_id, job_id, state)


def _classification_limit(state):
    """SetupRefinementJob(): 20 A falling to 8 A over nine rounds, never
    finer than the alignment limit."""
    lowest = min(state["class_high_res_limits"])
    return max(20.0 + (8.0 - 20.0) * (state["round"] / 9.0), lowest)


def _launch_refinement(conn, project_id, job_id, state):
    """SetupRefinementJob() + RunRefinementJob()."""
    s = state["settings"]
    n, classes = state["number_of_particles"], state["number_of_classes"]
    first_round = state["round"] == 0
    tracking = _load_tracking(state)
    lowest = min(state["class_high_res_limits"])
    last_res = state["resolution_per_round"][-1] if state["resolution_per_round"] else 100.0
    flags = choose_global_array(state["round"], last_res, lowest, tracking["last_global_res"], tracking["globals"], tracking["since_global"],
                                state["reference_contains_all"], state["final_round"], np.random.default_rng())
    class_rows = _load_rows(state, "input")
    for table in class_rows:
        table["image_is_active"] = flags[:len(table)] if len(flags) >= len(table) else np.concatenate([flags, np.ones(len(table) - len(flags), dtype=int)])
    _store_rows(state, "input", class_rows)
    class_stats = _load_stats(state, "input")
    rid = refinements.next_refinement_id(conn)
    state["output_refinement_id"] = rid
    scratch = Path(state["scratch"])
    # The first round refines one class only; its results seed every class.
    classes_to_run = 1 if first_round else classes
    star_files, stats_files = [], []
    for k in range(classes_to_run):
        p = str(scratch / "auto_input_par_{}_{}.cistem".format(rid, k + 1))
        starfile.write_params(p, class_rows[k])
        star_files.append(p)
        sp = str(scratch / "auto_input_stats_{}_{}.txt".format(rid, k + 1))
        write_statistics(sp, [st for st in class_stats[k] if 1 <= st["shell"] <= state["box_size"] // 2], state["pixel_size"])
        stats_files.append(sp)
    percent_used = 1.0 if first_round else min(1.0, state["current_percent_used"] * 3.0 / 100.0)
    classification_limit = _classification_limit(state)
    jobs = max(1, min(n, state["refinement_jobs"]))
    tasks, outputs = [], []
    index = 0
    for k in range(classes_to_run):
        limit = state["class_high_res_limits"][k]
        step = max(calculate_angular_step(limit, s["mask_radius_a"]), calculate_angular_step(8.0, s["mask_radius_a"]))
        for j in range(1, jobs + 1):
            first, last = particle_range(j, jobs, n)
            out_star = str(scratch / "refine_output_{}_class{}_{}.cistem".format(rid, k + 1, j))
            outputs.append(out_star)
            values = [state["stack_filename"], star_files[k], state["reference_files"][k], stats_files[k], True, "", out_star, "/dev/null",
                      state["symmetry"], first, last, percent_used, state["pixel_size"], state["molecular_weight"],
                      s["inner_mask_radius_a"], s["mask_radius_a"], s["low_resolution_limit_a"], limit,
                      0.0, classification_limit, s["global_mask_radius_a"], limit, step, int(s["number_of_results_to_refine"]),
                      s["search_range_x_a"], s["search_range_y_a"], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                      False, False, True, True, True, True, True,
                      False, False, False, True, state["invert_contrast"], False, not s["apply_blurring"], True,
                      1, True, k, False, False]
            tasks.append(_task(refine3d_adapter, index, k * 1000000 + j, values))
            index += 1
    parent = _parent_row(conn, job_id)
    child = _new_child(conn, job_id, CHILD_REFINE, "{} · round {} refine3d".format(parent["NAME"], state["round"] + 1), parent)
    state.update({"phase": "refine", "child_job_id": child, "child_task_count": len(tasks), "child_done": 0, "pending_output_stars": outputs,
                  "refinement_jobs_this_round": jobs, "classes_this_round": classes_to_run})
    n_global = sum(1 for f in flags if f == 0)
    _log(project_id, job_id, "Running refinement round {:2d} ({} of {} particles globally, refining {:.1f}%; limit{} {} Å) -> Refinement #{} — child job {}".format(
        state["round"] + 1, n_global, n, percent_used * 100.0, "s" if classes > 1 else "", ", ".join("{:.2f}".format(v) for v in state["class_high_res_limits"]), rid, child))
    _runtime.submit_child(project_id, child, refine3d_adapter, tasks, _profile(state["refinement_profile"]))


def choose_global_array(rounds_run, last_resolution, lowest_alignment_res, last_global_res, globals_so_far, rounds_since_global,
                        reference_contains_all, final_round, rng):
    """choose_global() for every particle at once (arrays in, int array of image_is_active out)."""
    last_global_res = np.asarray(last_global_res, dtype=float)
    globals_so_far = np.asarray(globals_so_far, dtype=float)
    rounds_since_global = np.asarray(rounds_since_global, dtype=float)
    n = len(globals_so_far)
    flags = np.ones(n, dtype=np.int64)
    never = globals_so_far == 0
    if rounds_run == 0:
        do_global = np.ones(n, dtype=bool)
    else:
        forced = (last_resolution < 5.0 and lowest_alignment_res <= 8.0 and bool(reference_contains_all) and rounds_run > 2)
        round_adjust = np.maximum(1.0, (globals_so_far - np.floor(rounds_since_global / 3.0)) ** 2)
        res_adjust = last_global_res - lowest_alignment_res
        with np.errstate(divide="ignore", invalid="ignore"):
            likelihood = np.where(last_global_res <= 5.0, -5.0, np.where(res_adjust == 0.0, 0.0, lowest_alignment_res ** 2 / ((1000.0 / np.where(res_adjust == 0.0, 1.0, res_adjust)) * round_adjust)))
        do_global = np.abs(rng.uniform(-1.0, 1.0, n)) < likelihood
        if forced:
            do_global |= last_global_res > 9.0
    flags[do_global] = 0
    if final_round:
        flags[:] = 1
    else:
        flags[rounds_since_global == 0] = 1
    flags[never] = 0
    return flags


def _merge_output_stars(state):
    """The refinement results back into every particle's table, from the
    per-task output parameter files (abinitio.merge_task_outputs)."""
    inputs = _load_rows(state, "input")
    jobs = _required_count(state, "refinement_jobs_this_round")
    outputs = state.get("pending_output_stars") or []
    classes_run = _required_count(state, "classes_this_round")
    class_rows = []
    for k in range(classes_run):
        paths = [outputs[k * jobs + j] if k * jobs + j < len(outputs) else None for j in range(jobs)]
        try:
            class_rows.append(starfile.merge_task_outputs(inputs[k], paths))
        except ValueError as exc:
            raise ValueError("{} (class {})".format(exc, k + 1))
    return class_rows


def _average_occupancy(rows):
    """Refinement::UpdateAverageOccupancy(): the mean occupancy over the active particles (0 when none)."""
    table = refinements.as_table(rows)
    active = table["image_is_active"] >= 0
    return float(table["occupancy"][active].astype(float).mean()) if active.any() else 0.0


def _rows_for_reconstruction(rows):
    """The table as reconstruct3d should rank it: a copy in which every
    particle refine3d left inactive this round has a score one below the
    lowest refined score, so the percentage threshold ranks only this
    round's refined particles (reconstruct3d skips inactive rows regardless,
    so the score is used for nothing else). The stored refinement keeps the
    real scores."""
    table = refinements.as_table(rows).copy()
    active = table["image_is_active"] >= 0
    floor = (float(table["score"][active].min()) if active.any() else 0.0) - 1.0
    table["score"][~active] = floor
    return table


def _launch_reconstruction(conn, project_id, job_id, state):
    """SetupReconstructionJob() + RunReconstructionJob()."""
    s = state["settings"]
    n, classes = state["number_of_particles"], state["number_of_classes"]
    rid = state["output_refinement_id"]
    box, ps = state["box_size"], state["pixel_size"]
    class_rows = _load_rows(state, "output")
    input_stats = _load_stats(state, "input")
    written = []
    for k, rows in enumerate(class_rows):
        if state["class_high_res_limits"][k] > 10.0:
            rows["sigma"] = 1.0
        p = str(Path(state["scratch"]) / "auto_output_par_{}_{}.cistem".format(rid, k + 1))
        starfile.write_params(p, _rows_for_reconstruction(rows))
        written.append(p)
    _store_rows(state, "output", class_rows)
    jobs = max(1, min(n, state["reconstruction_jobs"]))
    scratch = Path(state["scratch"])
    last_res = state["resolution_per_round"][-1] if state["resolution_per_round"] else None
    limits_rec, weights = [], []
    for k in range(classes):
        limit = state["class_high_res_limits"][k]
        if state["final_round"]:
            rec_limit = 0.0
        else:
            padded = resolution_n_shells_after(input_stats[k], limit, box // 10, ps)
            if last_res is not None and abs(last_res - padded) < last_res * 0.05:
                rec_limit = resolution_n_shells_after(input_stats[k], last_res, box // 10, ps)
            else:
                rec_limit = padded
        if state["last_round_reconstruction_resolution"] is not None and rec_limit > state["last_round_reconstruction_resolution"]:
            rec_limit = state["last_round_reconstruction_resolution"]
        state["last_round_reconstruction_resolution"] = rec_limit
        limits_rec.append(rec_limit)
        weights.append(2.0 if limit < 8.0 else 0.0)
    # cisTEM passes 0.333 here while the round refines three times the percentage
    # used: "the top third of what was refined". reconstruct3d's percentile, though,
    # ranks every row, and the two thirds refine3d skipped keep the score of the round
    # that last refined them -- at a different resolution limit, so not comparable.
    # On a large dataset (a few percent used, so rounds 2-5 refine a minority) the
    # stale majority set the cut and the map was built from a handful of particles,
    # worse each round. With the inactive rows ranked below every refined one
    # (_rows_for_reconstruction) the percentage of *all* rows that is meant is the
    # percentage used itself, and the threshold picks the top third of the refined.
    # The same departure abinitio.reconstruction_rows() makes, for the same reason.
    # Round one refined every particle, so its scores are on one scale and cisTEM's
    # 0.333 -- the top third of all of them -- stands as it is.
    if state["round"] == 0:
        score_threshold = 0.333
    elif state["current_percent_used"] * 3.0 < 100.0:
        score_threshold = max(state["current_percent_used"] / 100.0, 1e-4)
    else:
        score_threshold = min(1.0, state["current_percent_used"] / 100.0)
    state["reconstruction_limits"] = limits_rec
    state["score_weights"] = weights
    tasks = []
    index = 0
    for k in range(classes):
        use_ref = bool(s["apply_blurring"] and state["reference_asset_ids"][k] != -1 and state["reference_files"][k])
        for j in range(1, jobs + 1):
            first, last = particle_range(j, jobs, n)
            values = [state["stack_filename"], written[k], state["reference_files"][k] if use_ref else "/dev/null",
                      "/dev/null", "/dev/null", "/dev/null", "/dev/null", state["symmetry"], first, last,
                      ps, state["molecular_weight"], s["inner_mask_radius_a"], s["mask_radius_a"],
                      limits_rec[k], state["class_high_res_limits"][k], weights[k], score_threshold, s["smoothing_factor"], 1.0,
                      True, True, state["invert_contrast"], False, s["autocrop_images"], False, s["autocenter"],
                      use_ref, True, True,
                      str(scratch / "dump_file_{}_{}_odd_{}.dump".format(rid, k, j)), str(scratch / "dump_file_{}_{}_even_{}.dump".format(rid, k, j)), 0, 1]
            tasks.append(_task(reconstruct3d, index, k * 1000000 + j, values))
            index += 1
    parent = _parent_row(conn, job_id)
    child = _new_child(conn, job_id, CHILD_RECON, "{} · round {} reconstruct3d".format(parent["NAME"], state["round"] + 1), parent)
    state.update({"phase": "recon", "child_job_id": child, "child_task_count": len(tasks), "child_done": 0, "number_of_dump_files": jobs})
    _log(project_id, job_id, "Calculating {} (limit{} {} Å, keeping {:.0f}% by score; {} task{}) — child job {}".format(
        "reconstructions" if classes > 1 else "reconstruction", "s" if classes > 1 else "",
        ", ".join("max" if v == 0 else "{:.2f}".format(v) for v in limits_rec), score_threshold * 100.0, len(tasks), "" if len(tasks) == 1 else "s", child))
    _runtime.submit_child(project_id, child, reconstruct3d, tasks, _profile(state["reconstruction_profile"]))


def _launch_merge(conn, project_id, job_id, state):
    """SetupMerge3dJob() + RunMerge3dJob()."""
    s = state["settings"]
    classes = state["number_of_classes"]
    rid = state["output_refinement_id"]
    scratch = Path(state["scratch"])
    vol_dir = volumes.volume_dir(project_id)
    tasks, outputs, stats, halves = [], [], [], []
    keep_halves = s["use_blush"] and s["blush_unfiltered"]   # after the merge, Blush denoises the half maps' average
    for k in range(classes):
        out = str(vol_dir / "volume_{}_{}.mrc".format(rid, k + 1))
        st = str(scratch / "volume_stats_{}_{}.txt".format(rid, k + 1))
        half = [str(scratch / "half_{}_{}_{}.mrc".format(rid, k + 1, h)) for h in (1, 2)] if keep_halves else ["/dev/null", "/dev/null"]
        outputs.append(out)
        stats.append(st)
        halves.append(half)
        tasks.append(_task(merge3d, k, k + 1, [half[0], half[1], out, st, state["molecular_weight"], s["inner_mask_radius_a"], s["mask_radius_a"],
                                                str(scratch / "dump_file_{}_{}_odd_.dump".format(rid, k)), str(scratch / "dump_file_{}_{}_even_.dump".format(rid, k)),
                                                k + 1, False, "", _required_count(state, "number_of_dump_files"), 1.0, state["class_high_res_limits"][k]]))
    parent = _parent_row(conn, job_id)
    child = _new_child(conn, job_id, CHILD_MERGE, "{} · round {} merge3d".format(parent["NAME"], state["round"] + 1), parent)
    state.update({"phase": "merge", "child_job_id": child, "child_task_count": len(tasks), "child_done": 0,
                  "pending_volume_files": outputs, "pending_stats_files": stats, "pending_half_maps": halves if keep_halves else []})
    _log(project_id, job_id, "Merging and filtering {} — child job {}".format("reconstructions" if classes > 1 else "reconstruction", child))
    _runtime.submit_child(project_id, child, merge3d, tasks, _profile(state["reconstruction_profile"]))


def _progress_percent(state):
    """cisTEM's bar pulses through each round; this counts the round's steps."""
    frac = float(state.get("child_done", 0)) / float(max(state.get("child_task_count", 1), 1))
    phase = state.get("phase")
    if phase == "refine":
        done = 0.6 * frac
    elif phase == "recon":
        done = 0.6 + 0.3 * frac
    elif phase == "merge":
        done = 0.9 + 0.1 * frac
    elif phase == "finished":
        done = 1.0
    else:
        done = 0.0
    return max(0, min(100, int(100.0 * done)))


def progress_info(state):
    if not state:
        return {}
    finished = [h["finished_at"] for h in state.get("history", []) if h.get("finished_at")]
    return {"task_count": None, "tasks_done": len(finished), "first_task_finished_at": finished[0] if finished else None,
            "last_task_finished_at": finished[-1] if finished else None, "round": state["round"], "rounds": None, "phase": state["phase"]}



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
    threading.Thread(target=_child_finished, args=(project_id, child_row["JOB_ID"], child_row["PARENT_JOB_ID"], status, error),
                     daemon=True, name="autorefine-" + child_row["PARENT_JOB_ID"]).start()


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
    """ProcessAllJobsFinished()."""
    phase = state["phase"]
    classes = state["number_of_classes"]
    if phase == "refine":
        class_rows = _merge_output_stars(state)
        if state["round"] == 0 and classes > 1:
            # Random occupancies around the one refined class, so classification starts from noise.
            rng = np.random.default_rng()
            seed = refinements.as_table(class_rows[0])
            class_rows = []
            for k in range(classes):
                table = seed.copy()
                table["occupancy"] = np.abs(rng.uniform(-1.0, 1.0, len(table)) * (200.0 / classes))
                class_rows.append(table)
        elif classes > 1:
            update_occupancies(class_rows, use_old_occupancies=state["current_percent_used"] >= 99.99)
        _store_rows(state, "output", class_rows)
        for p in state.get("pending_output_stars") or []:
            try:
                os.remove(p)
            except OSError:
                pass
        _launch_reconstruction(conn, project_id, parent_id, state)
    elif phase == "recon":
        _launch_merge(conn, project_id, parent_id, state)
    elif phase == "merge":
        _record_round(conn, project_id, parent_id, state)
        for p in Path(state["scratch"]).glob("dump_file_*.dump"):
            try:
                p.unlink()
            except OSError:
                pass
        if state["settings"]["use_blush"]:
            blush_phase.start(_BLUSH, conn, project_id, parent_id, state, blush_phase.POST)
        else:
            _cycle(conn, project_id, parent_id, state)
    else:
        raise ValueError("unexpected phase {!r}".format(phase))
    if state["phase"] != "finished":
        _save(conn, parent_id, state, _progress_percent(state))


def _record_round(conn, project_id, parent_id, state):
    """The MERGE branch of ProcessAllJobsFinished(): volume assets, reconstruction
    records, the package's references, and the refinement itself."""
    s = state["settings"]
    classes = state["number_of_classes"]
    rid = state["output_refinement_id"]
    for p in state["pending_volume_files"]:
        if not os.path.isfile(p):
            raise ValueError("merge3d did not write {}".format(p))
    class_rows = _load_rows(state, "output")
    stats = [read_statistics(p) if os.path.isfile(p) else [] for p in state["pending_stats_files"]]
    for k in range(classes):
        if not stats[k]:
            raise ValueError("merge3d wrote no resolution statistics for class {}".format(k + 1))
    avgs = pooled_part_ssnr(stats, class_rows)
    _store_stats(state, "output", stats)
    volume_ids, recon_ids, est_res = [], [], []
    previous_refs = list(state["reference_asset_ids"])
    limits_rec = state.get("reconstruction_limits") or [0.0] * classes
    weights = state.get("score_weights") or [0.0] * classes
    for k in range(classes):
        name = "Auto #{} (Rnd. {}) - Class #{}".format(rid, state["round"] + 1, k + 1)
        recon_id = refinements.next_reconstruction_id(conn)
        vid = volumes.add_volume_asset(conn, name, state["pending_volume_files"][k], state["pixel_size"], state["box_size"], state["box_size"], state["box_size"],
                                       reconstruction_job_id=recon_id)
        refinements.add_reconstruction_job(conn, recon_id, state["package_id"], rid, "", s["inner_mask_radius_a"], s["mask_radius_a"],
                                           limits_rec[k], weights[k], False, s["autocrop_images"], False, s["apply_blurring"], s["smoothing_factor"], k + 1, vid)
        refinements.set_current_reference(conn, state["package_id"], k + 1, vid)
        volume_ids.append(vid)
        recon_ids.append(recon_id)
        est_res.append(refinements.estimated_resolution(stats[k], state["pixel_size"]))
    state["reference_files"] = list(state["pending_volume_files"])
    state["reference_asset_ids"] = volume_ids
    state["volume_ids"] = state.get("volume_ids", []) + volume_ids
    volumes.prepare_orth_views(state["reference_files"], state["settings"]["mask_radius_a"], lambda t: _log(project_id, parent_id, t, "warning"))
    for k in range(classes):
        if classes > 1:
            _log(project_id, parent_id, "Est. Res. Class {:2d} = {:.2f} Å ({:.2f} %)".format(k + 1, est_res[k], avgs[k]))
        else:
            _log(project_id, parent_id, "Est. Res. = {:.2f} Å".format(est_res[k]))
    details = []
    for k in range(classes):
        avg_occ = _average_occupancy(class_rows[k])
        limit = state["class_high_res_limits"][k]
        details.append({
            "REFERENCE_VOLUME_ASSET_ID": previous_refs[k], "LOW_RESOLUTION_LIMIT": s["low_resolution_limit_a"], "HIGH_RESOLUTION_LIMIT": limit,
            "MASK_RADIUS": s["mask_radius_a"], "SIGNED_CC_RESOLUTION_LIMIT": 0.0, "GLOBAL_RESOLUTION_LIMIT": limit,
            "GLOBAL_MASK_RADIUS": s["global_mask_radius_a"], "NUMBER_RESULTS_TO_REFINE": int(s["number_of_results_to_refine"]),
            "ANGULAR_SEARCH_STEP": max(calculate_angular_step(limit, s["mask_radius_a"]), calculate_angular_step(8.0, s["mask_radius_a"])),
            "SEARCH_RANGE_X": s["search_range_x_a"], "SEARCH_RANGE_Y": s["search_range_y_a"],
            "CLASSIFICATION_RESOLUTION_LIMIT": _classification_limit(state), "SHOULD_FOCUS_CLASSIFY": 0,
            "SPHERE_X_COORD": 0.0, "SPHERE_Y_COORD": 0.0, "SPHERE_Z_COORD": 0.0, "SPHERE_RADIUS": 0.0,
            "SHOULD_REFINE_CTF": 0, "DEFOCUS_SEARCH_RANGE": 0.0, "DEFOCUS_SEARCH_STEP": 0.0,
            "AVERAGE_OCCUPANCY": avg_occ, "ESTIMATED_RESOLUTION": est_res[k], "RECONSTRUCTED_VOLUME_ASSET_ID": volume_ids[k], "RECONSTRUCTION_ID": recon_ids[k],
            "SHOULD_AUTOMASK": 1 if s["auto_mask"] else 0, "SHOULD_REFINE_INPUT_PARAMS": 1,
            "SHOULD_USE_SUPPLIED_MASK": 1 if s["use_mask"] else 0, "MASK_ASSET_ID": state.get("mask_asset_id", -1), "MASK_EDGE_WIDTH": s["mask_edge_a"],
            "OUTSIDE_MASK_WEIGHT": s["outside_mask_weight"], "SHOULD_LOWPASS_OUTSIDE_MASK": 1 if s["low_pass_outside_mask"] else 0,
            "MASK_FILTER_RESOLUTION": s["mask_filter_resolution_a"],
        })
    ref = {"refinement_id": rid, "refinement_package_asset_id": state["package_id"],
           "name": "Auto #{} - Round {}".format(rid, state["round"] + 1), "resolution_statistics_are_generated": False,
           "starting_refinement_id": state["input_refinement_id"], "number_of_particles": state["number_of_particles"], "number_of_classes": classes,
           "resolution_statistics_box_size": state["box_size"], "resolution_statistics_pixel_size": state["pixel_size"], "percent_used": state["current_percent_used"],
           "job_id": parent_id}
    refinements.add_refinement(conn, ref, class_rows, stats, details, symmetry=state["symmetry"])
    state["history"].append({"round": state["round"] + 1, "refinement_id": rid, "label": "Iter. #{}".format(state["round"] + 1), "estimated_resolution": est_res,
                             "high_res_limits": list(state["class_high_res_limits"]), "percent_used": state["current_percent_used"],
                             "average_sigma": average_sigma(class_rows), "finished_at": now_iso()})
    _log(project_id, parent_id, "Refinement #{} written ({}); volume asset{} {}".format(rid, ref["name"], "" if classes == 1 else "s", ", ".join("#{}".format(v) for v in volume_ids)))
    if state["current_percent_used"] > 99.9:
        state["reference_contains_all"] = True


def plan_next_round(state, output_stats, input_rows, output_rows):
    """CycleRefinement()'s bookkeeping, on the state in place: the global-
    alignment tracking, the classes' next limits, the particle percentage,
    and whether this was the last round. Returns (stop_now, final_next)."""
    classes = state["number_of_classes"]
    n = state["number_of_particles"]
    box, ps = state["box_size"], state["pixel_size"]
    mask = state["settings"]["mask_radius_a"]
    state["percent_used_per_round"].append(state["current_percent_used"])
    tracking = _load_tracking(state)
    for i in range(n):
        if i < len(input_rows[0]) and i < len(output_rows[0]) and input_rows[0][i].get("image_is_active", 1) == 0 and output_rows[0][i].get("image_is_active", 1) == 1:
            tracking["globals"][i] += 1
            tracking["since_global"][i] = 0
            tracking["last_global_res"][i] = state["high_res_limit_per_round"][-1]
        else:
            tracking["since_global"][i] += 1
    _store_tracking(state, tracking)
    best_p143 = None
    for k in range(classes):
        state["class_high_res_limits"][k] = next_class_limit(output_stats[k], state["class_high_res_limits"][k], box, ps, mask)
        est = resolution_at(output_stats[k], 0.143, ps, False)
        best_p143 = est if best_p143 is None else min(best_p143, est)
    state["high_res_limit_per_round"].append(min(state["class_high_res_limits"]))
    if state["resolution_per_round"]:
        if best_p143 > state["resolution_per_round"][-1] - 0.1:  # no improvement: let more particles in
            state["max_percent_used"] = min(100.0, state["max_percent_used"] * 3.0)
        if best_p143 < 7.0:
            state["max_percent_used"] = 100.0
    state["resolution_per_round"].append(best_p143)
    wanted = percent_for_resolution(best_p143, n, state["symmetry"], classes)
    current = max(wanted, state["start_percent_used"])
    current = min(current, 100.0)
    if current < state["max_percent_used"]:
        current = state["max_percent_used"]
    else:
        state["max_percent_used"] = current
    state["current_percent_used"] = current
    state["round"] += 1
    if classes == 1:
        change = 0.0
    else:
        change = sum(abs(_average_occupancy(output_rows[k]) - _average_occupancy(input_rows[k])) for k in range(classes))
    stop = should_stop(state["resolution_per_round"], state["max_percent_used"], change, classes)
    if state["final_round"]:
        return True, False
    if stop:
        state["final_round"] = True
        state["current_percent_used"] = 100.0
    return False, stop


# A Finish that cisTEM's panel only offers once the manager has stopped by
# itself: here it can be asked for while running, and takes effect once the
# round in progress has written its refinement.
ACTIONS = {"finish": "Finish After This Round"}


def available_actions(state):
    if not state or state.get("phase") in (None, "finished") or state.get("finish_requested"):
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
    if state.get("finish_requested"):
        state["round"] += 1
        state["phase"] = "finished"
        state["child_job_id"] = None
        _log(project_id, parent_id, "Finished at the user's request after {} round{}.".format(state["round"], "" if state["round"] == 1 else "s"))
        _finish(conn, project_id, parent_id, state, "completed", None)
        return
    output_stats = _load_stats(state, "output")
    input_rows = _load_rows(state, "input")
    output_rows = _load_rows(state, "output")
    stop_now, final_next = plan_next_round(state, output_stats, input_rows, output_rows)
    if stop_now:
        state["phase"] = "finished"
        state["child_job_id"] = None
        _log(project_id, parent_id, "Resolution is stable - Auto refine is stopping.")
        _finish(conn, project_id, parent_id, state, "completed", None)
        return
    if final_next:
        _log(project_id, parent_id, "Resolution is stable - one final round on every particle with an unlimited reconstruction")
    _log(project_id, parent_id, "Next round: limit{} {} Å, {:.1f}% of the particles (est. {:.2f} Å so far)".format(
        "s" if state["number_of_classes"] > 1 else "", ", ".join("{:.2f}".format(v) for v in state["class_high_res_limits"]),
        state["current_percent_used"], state["resolution_per_round"][-1]))
    # The output becomes the input.
    _store_rows(state, "input", output_rows)
    _store_stats(state, "input", output_stats)
    state["input_refinement_id"] = state["output_refinement_id"]
    _prepare_references_then_refine(conn, project_id, parent_id, state)


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
    metrics = {"rounds_run": state.get("round", 0), "refinement_ids": [h["refinement_id"] for h in state.get("history", [])],
               "volume_ids": state.get("volume_ids", []), "final_refinement_id": state.get("output_refinement_id") if status == "completed" else None,
               "resolution_per_round": state.get("resolution_per_round", []), "percent_used_per_round": state.get("percent_used_per_round", [])}
    if state.get("resolution_per_round"):
        metrics["resolution_a"] = state["resolution_per_round"][-1]
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
        _log(project_id, parent_id, "cancel requested; stopping Blush")
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
    """The Jobs tab's Latest Result: the newest refinement's FSC, the
    resolution and percentage per round, and the current reconstruction."""
    state = _load_state(conn, row["JOB_ID"])
    if not state:
        return None
    files = [p for p in (state.get("reference_files") or []) if p and os.path.isfile(p)]
    if not state.get("history"):
        return None
    last = state["history"][-1]
    rid = last["refinement_id"]
    stats = refinements.load_statistics(conn, rid, 1)
    return {
        "kind": "auto_refine3d", "task_index": len(state["history"]) * 10 + (1 if state.get("phase") == "finished" else 0),
        "label": last["label"], "refinement_id": rid, "phase": state["phase"], "round": state["round"], "rounds": None,
        "final_round": state.get("final_round"), "number_of_classes": state["number_of_classes"], "has_picture": bool(files), "current_volume": files[0] if files else None,
        "estimated_resolution": last.get("estimated_resolution"), "history": state["history"],
        "resolution_per_round": state.get("resolution_per_round", []), "percent_used_per_round": state.get("percent_used_per_round", []),
        "high_res_limits": state.get("class_high_res_limits"), "current_percent_used": state.get("current_percent_used"),
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
