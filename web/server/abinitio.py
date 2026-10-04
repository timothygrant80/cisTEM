"""Ab-initio 3D reconstruction: cisTEM's AbInitioManager (src/gui/AbInitio3DPanel.cpp)
as a server-side driver, in the parent/child shape of server/classification.py.

The cycle, for `number_of_starts` starts of `number_of_rounds` rounds each:

    [PREPARE STACK]  prepare_stack x N   when the final resolution allows binning
    INITIAL RECON    reconstruct3d x N   random angles, a few percent of the particles
    INITIAL MERGE    merge3d x classes   -> Scratch/Startup/startup3d_initial_0_<k>.mrc
    per round:  (auto-mask the reference)  refine3d x N  ->  reconstruct3d x N  ->  merge3d x classes
    at 3/4 of the first start, for a symmetric particle not refined with symmetry
    from the start: align_symmetry, then symmetry is applied from there on.

Each program run is a hidden child job (STAGE "abinitio_<program>",
PARENT_JOB_ID set) that the runner launches through the run profile the
panel names -- the refinement profile for prepare_stack / refine3d /
align_symmetry, the reconstruction profile for reconstruct3d / merge3d, as
cisTEM has two pickers. Between children the driver does what the GUI does
in its own threads: writes the input star and statistics files, masks the
reference (volumes.auto_mask), merges the per-task output star files back
into the refinement, updates occupancies and the pooled particle SSNR for
several classes, and at the end resamples the final reconstruction to the
package's box size and registers it as a volume asset (VOLUME_ASSETS +
STARTUP_LIST, Database::AddStartupJob). align_symmetry is run directly by
the driver rather than through the runner, whole angular range at once,
because in socket mode the program only reports the alignment and leaves
applying it to the GUI, while run locally it writes the aligned,
symmetrised volume itself.

The refinement -- one row per particle per class with the Euler angles,
shifts, occupancy, logP, sigma and score -- lives in star files under
<project>/Scratch/Startup/<job>/ rather than in memory, so the parent can
resume from JOBS.STATE_JSON after a restart. The schedules (resolution
ramp, percent used, Wiener nominator, signed-CC limit) follow
BeginRefinementCycle() / CycleRefinement() / Setup*Job() line for line.

The panel's other input is a **2D class selection** (classification.py's
selection manager): prepare_stack_classaverage then builds, for each
selected class, `number_of_2d_classes` CTF-corrected averages of
`images_per_class` randomly drawn members -- whitened, aligned by the
classification's angles and shifts, binned to the final resolution -- and
those averages are the "particles" the cycle refines (no CTF: defocus 0,
300 kV, 2.7 mm, 0.07, alternating half-sets), with the 3D class count and
symmetry taken from the panel rather than the package. The count of
averages per class follows BeginRefinementCycle(): the smallest selected
class divided by images-per-class, clamped so the whole stack holds
between 2500 and 20000 averages.
"""
import json
import math
import os
import re
import random
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import blush
import blush_phase
import db
import progress_store
import refinements
import job_protocol as jp
import job_runner
import refinement_packages
import starfile
import volumes
from stages import merge3d, prepare_stack, prepare_stack_classaverage, reconstruct3d, refine3d

STAGE = "ab_initio_3d"
CHILD_PREPARE = "abinitio_prepare_stack"
CHILD_PREPARE_CLASSAVG = "abinitio_prepare_stack_classaverage"
CHILD_REFINE = "abinitio_refine3d"
CHILD_RECON = "abinitio_reconstruct3d"
CHILD_MERGE = "abinitio_merge3d"

# AbInitio3DPanel::SetDefaults(); the mask radius and search range follow the package.
DEFAULTS = {
    "number_of_starts": 2,
    "number_of_rounds": 40,
    "initial_resolution_limit_a": 20.0,
    "final_resolution_limit_a": 8.0,
    "inner_mask_radius_a": 0.0,
    "auto_mask": True,
    "use_blush": True,   # Blush regularisation of the reference between rounds (blush.py / blush_phase.py); not in cisTEM's panel. On by default here, when the server can run it
    "auto_percent_used": True,
    "start_percent_used": 10.0,
    "end_percent_used": 10.0,
    "always_apply_symmetry": False,
    "apply_blurring": False,
    "smoothing_factor": 1.0,
    "images_per_class": 5,
    "number_of_classes": 1,
}
PLEASE_CREATE_PACKAGE_MESSAGE = ("Please create a refinement package (in the assets panel) in order to perform a "
                                 "3D refinement.")


# ---------------------------------------------------------------------------
# The pure pieces
# ---------------------------------------------------------------------------

def asymmetric_units(symmetry):
    """ReturnNumberofAsymmetricUnits()."""
    s = str(symmetry or "C1").strip().upper()
    kind, num = s[0], s[1:]
    n = int(num) if num.isdigit() else 0
    if kind == "C":
        return max(n, 1)
    if kind == "D":
        return 2 * n
    if kind == "T":
        return 12
    if kind == "O":
        return 24
    if kind == "I":
        return 60
    return 1


def percent_used_plan(number_of_particles, number_of_classes, symmetry, auto, start_percent, end_percent):
    """BeginRefinementCycle(): with Auto Percent Used, enough particles for
    2500 asymmetric units per class at the start and 10000 at the end,
    with and without the symmetry counted; else the user's numbers."""
    if auto:
        sym_n = asymmetric_units(symmetry)
        want_start = 2500 * number_of_classes
        want_end = 10000 * number_of_classes
        n = max(number_of_particles, 1)
        return {
            "start": min(100.0, want_start / n * 100.0),
            "end": min(100.0, want_end / n * 100.0),
            "sym_start": min(100.0, want_start / (n * sym_n) * 100.0),
            "sym_end": min(100.0, want_end / (n * sym_n) * 100.0),
        }
    return {"start": start_percent, "end": end_percent, "sym_start": start_percent, "sym_end": start_percent}


def round_schedule(rounds_run, rounds, start_res, end_res, plan, apply_symmetry):
    """CycleRefinement()'s numbers for round `rounds_run` (0-based) of
    `rounds`: the refinement resolution limit (ramping from start to end,
    reset to the start every ~tenth round for the first 65%), the next
    round's limit (the reconstruction's), and the percent of particles."""
    denom = float(max(rounds - 1, 1))
    frac = float(rounds_run) / denom
    current = start_res + (end_res - start_res) * frac
    step = max(1, int(math.floor(rounds / 10.0 + 0.5)))
    if rounds_run % step <= 1 and rounds_run <= rounds * 0.65:
        current = start_res
    nxt = max(0.0, start_res + (end_res - start_res) * float(rounds_run + 1) / denom)
    if apply_symmetry:
        percent = plan["sym_start"] + (plan["sym_end"] - plan["sym_start"]) * frac
    else:
        percent = plan["start"] + (plan["end"] - plan["start"]) * frac
    return {"high_res": current, "next_high_res": nxt, "percent_used": percent}


def wiener_nominator(rounds_run, rounds, starts_run):
    """SetupMerge3dJob(): 500 for the initial reconstruction, ramping 200 -> 10 over the first start, 10 after."""
    if rounds_run == 0 and starts_run == 0:
        return 500.0
    if starts_run == 0:
        return max(10.0, 200.0 + (10.0 - 200.0) * float(rounds_run) / float(rounds))
    return 10.0


def signed_cc_limit(rounds_run, rounds):
    """SetupRefinementJob(): 0 (no limit) on odd rounds and the last, 15 A otherwise."""
    return 0.0 if (rounds_run % 2 == 1 or rounds_run == rounds - 1) else 15.0


def angular_step(resolution_a, radius_a=75.0):
    """CalculateAngularStep(resolution, 75)."""
    return math.degrees(2.0 * resolution_a / radius_a)


def particle_range(job_number, number_of_jobs, number_of_particles):
    return __import__("classification").particle_range(job_number, number_of_jobs, number_of_particles)


def prepare_stack_jobs(number_of_particles, total_jobs):
    """SetupPrepareStackJob(): as many jobs as processes, but at least 100 particles each."""
    jobs = max(1, int(total_jobs))
    if number_of_particles / float(jobs) < 100:
        jobs = 1 if number_of_particles < 100 else int(number_of_particles / 100.0)
    return max(1, jobs)


def binned_box_size(box_size, binning_factor):
    """ReturnClosestFactorizedUpper(ReturnSafeBinnedBoxSize(box, bin), 3, true)."""
    safe = int(math.floor(float(box_size) / binning_factor + 0.5))
    return refinement_packages.closest_factorized_upper(safe, 3, True)


def class_averages_per_class(smallest_class_size, images_per_class, number_of_selected_classes):
    """BeginRefinementCycle(), class-average input: how many averages to
    make of each selected class -- the smallest class's members divided by
    the images per average, kept between 2500 and 20000 averages in all."""
    n = int(smallest_class_size) // max(int(images_per_class), 1)
    lo = 2500 // max(int(number_of_selected_classes), 1)
    hi = 20000 // max(int(number_of_selected_classes), 1)
    return max(lo, min(hi, n))


def classaverage_job_ranges(number_of_selected_classes, total_jobs):
    """SetupPrepareStackJob(): the selected classes (0-based indices into
    the selection) split over the run profile's processes, at least one
    class per job -> [(first, last)]."""
    jobs = max(1, min(int(total_jobs), int(number_of_selected_classes)))
    ranges = []
    for j in range(1, jobs + 1):
        first, last = particle_range(j, jobs, number_of_selected_classes)
        ranges.append((first - 1, last - 1))
    return ranges


def random_angles(rng):
    """BeginRefinementCycle()'s re-randomisation: phi and psi uniform on
    [-180, 180), theta from acos(2|u| - 1) so views are spread evenly."""
    return (rng.uniform(-1.0, 1.0) * 180.0,
            math.degrees(math.acos(max(-1.0, min(1.0, 2.0 * abs(rng.uniform(-1.0, 1.0)) - 1.0)))),
            rng.uniform(-1.0, 1.0) * 180.0)


def _tables(class_rows):
    """The per-class parameter tables from either form (dict rows are converted once)."""
    return [refinements.as_table(c) for c in class_rows]


def update_occupancies(class_rows, use_old_occupancies=True):
    """Refinement::UpdateOccupancies() for class_rows[k] (class k's table, the
    same particles in the same order in every class), in place: each
    particle's occupancies from its per-class logP, weighted by the classes'
    old average occupancies (or equally, when `use_old_occupancies` is off --
    Auto Refine's choice while it still refines a subset of the particles).
    cisTEM then gives every class the particle's occupancy-weighted sigma:
    one noise estimate per particle, which refine3d and reconstruct3d weight
    by; left per class, the classes' likelihoods drift apart round by round.
    Vectorised over particles; `class_rows` must hold tables (the drivers do)."""
    n_classes = len(class_rows)
    if n_classes <= 1:
        return
    if class_rows and not isinstance(class_rows[0], np.ndarray):
        # Dict rows (tests, older callers): compute on tables and write the two columns back.
        tables = _tables(class_rows)
        update_occupancies(tables, use_old_occupancies)
        for rows, t in zip(class_rows, tables):
            for r, occ, sig in zip(rows, t["occupancy"].tolist(), t["sigma"].tolist()):
                r["occupancy"] = occ
                r["sigma"] = sig
        return
    logp = np.stack([t["logp"].astype(float) for t in class_rows], 0)          # classes x particles
    sigma = np.stack([t["sigma"].astype(float) for t in class_rows], 0)
    if use_old_occupancies:
        avg = np.array([t["occupancy"].astype(float).mean() if len(t) else 0.0 for t in class_rows])
    else:
        avg = np.full(n_classes, 100.0 / n_classes)
    max_logp = logp.max(axis=0)
    within = (max_logp - logp) < 10.0
    prob = np.where(within, np.exp(np.minimum(logp - max_logp, 0.0)) * avg[:, None], 0.0)
    total = prob.sum(axis=0)
    occ = np.where(total > 0, prob / np.where(total > 0, total, 1.0) * 100.0, 0.0)
    average_sigma = (sigma * occ / 100.0).sum(axis=0)
    for k, t in enumerate(class_rows):
        t["occupancy"] = occ[k]
        t["sigma"] = average_sigma


def pooled_part_ssnr(stats_per_class, class_rows):
    """Refinement::UpdatePSSNR(): occupancy-weighted average of the classes'
    particle SSNR curves, written back to every class. Returns the average
    occupancies (100 for one class)."""
    if len(class_rows) <= 1:
        return [100.0]
    avgs = []
    for t in _tables(class_rows):
        active = t["image_is_active"] >= 0
        avgs.append(float(t["occupancy"][active].astype(float).mean()) if active.any() else 0.0)
    total = sum(avgs) or 1.0
    n_points = min(len(s) for s in stats_per_class)
    for i in range(n_points):
        pooled = sum(stats_per_class[k][i]["part_ssnr"] * avgs[k] for k in range(len(class_rows))) / total
        for s in stats_per_class:
            s[i]["part_ssnr"] = pooled
    return avgs


def average_sigma(class_rows):
    """The occupancy-weighted mean sigma over the active particles of every class (None when none)."""
    n_active = 0.0
    total = 0.0
    for t in _tables(class_rows):
        active = t["image_is_active"] >= 0
        w = t["occupancy"][active].astype(float) * 0.01
        n_active += float(w.sum())
        total += float((t["sigma"][active].astype(float) * w).sum())
    return total / n_active if n_active > 0 else None


def default_statistics(molecular_weight_kda, pixel_size, box_size):
    """[{shell, resolution, fsc, part_fsc, part_ssnr, rec_ssnr}] for shells 1..number_of_bins-1."""
    rows = refinement_packages.default_statistics(molecular_weight_kda, pixel_size, box_size)
    number_of_bins = int(box_size) // 2 + 1
    return [{"shell": r[0], "resolution": r[1], "fsc": r[2], "part_fsc": r[3], "part_ssnr": r[4], "rec_ssnr": r[5]}
            for r in rows[1:number_of_bins]]


def write_statistics(path, stats, pixel_size):
    with open(path, "w") as fh:
        fh.write("C        SHELL     RESOLUTION    RING_RADIUS            FSC       Part_FSC  Part_SSNR^0.5   Rec_SSNR^0.5\n")
        for s in stats:
            res = s["resolution"] if s["resolution"] else 1e-6
            fh.write("{:14.0f} {:14.4f} {:14.4f} {:14.4f} {:14.4f} {:14.4f} {:14.4f}\n".format(
                s["shell"] + 1, s["resolution"], pixel_size / res, s["fsc"], s["part_fsc"],
                math.sqrt(max(s["part_ssnr"], 0.0)), math.sqrt(max(s["rec_ssnr"], 0.0))))
    return path


def read_statistics(path):
    out = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts or parts[0].startswith("C") or parts[0].startswith("#"):
                continue
            if len(parts) < 7:
                continue
            try:
                vals = [float(p) for p in parts[:7]]
            except ValueError:
                continue
            out.append({"shell": int(vals[0]) - 1, "resolution": vals[1], "fsc": vals[3], "part_fsc": vals[4],
                        "part_ssnr": vals[5] ** 2, "rec_ssnr": vals[6] ** 2})
    return out


def cap_part_ssnr(stats, defaults):
    """SetupRefinementJob() from round 3 on: the measured particle SSNR may not exceed the synthetic one."""
    by_shell = {d["shell"]: d["part_ssnr"] for d in defaults}
    for s in stats:
        cap = by_shell.get(s["shell"])
        if cap is not None and s["part_ssnr"] > cap:
            s["part_ssnr"] = cap
    return stats


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

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
    size = float(pkg["PARTICLE_SIZE"] or 150.0)
    return {
        "number_of_starts": max(1, _num(params, "number_of_starts", DEFAULTS["number_of_starts"], int)),
        "number_of_rounds": max(1, _num(params, "number_of_rounds", DEFAULTS["number_of_rounds"], int)),
        "initial_resolution_limit": _num(params, "initial_resolution_limit_a", DEFAULTS["initial_resolution_limit_a"]),
        "final_resolution_limit": _num(params, "final_resolution_limit_a", DEFAULTS["final_resolution_limit_a"]),
        "mask_radius": _num(params, "mask_radius_a", size * 0.75),
        "inner_mask_radius": _num(params, "inner_mask_radius_a", DEFAULTS["inner_mask_radius_a"]),
        "search_range_x": _num(params, "search_range_x_a", size * 0.4),
        "search_range_y": _num(params, "search_range_y_a", size * 0.4),
        "auto_mask": _flag(params, "auto_mask", DEFAULTS["auto_mask"]),
        # The default applies only where Blush can run, so a request that leaves the field out is not refused on a server without torch or the weights.
        "use_blush": _flag(params, "use_blush", DEFAULTS["use_blush"] and blush.availability()["available"]),
        "auto_percent_used": _flag(params, "auto_percent_used", DEFAULTS["auto_percent_used"]),
        "start_percent_used": _num(params, "start_percent_used", DEFAULTS["start_percent_used"]),
        "end_percent_used": _num(params, "end_percent_used", DEFAULTS["end_percent_used"]),
        "always_apply_symmetry": _flag(params, "always_apply_symmetry", DEFAULTS["always_apply_symmetry"]),
        "apply_blurring": _flag(params, "apply_blurring", DEFAULTS["apply_blurring"]),
        "smoothing_factor": _num(params, "smoothing_factor", DEFAULTS["smoothing_factor"]),
        "symmetry": str(params.get("symmetry") or pkg["SYMMETRY"] or "C1").strip().upper(),
        "use_class_averages": str(params.get("input_mode") or "images").strip().lower().startswith("class"),
        "images_per_class": max(1, _num(params, "images_per_class", DEFAULTS["images_per_class"], int)),
        "number_of_classes": max(1, _num(params, "number_of_classes", pkg["NUMBER_OF_CLASSES"] or 1, int)),
    }


def package_defaults(pkg):
    """What the panel prefills from the package (SetDefaults())."""
    size = float(pkg["PARTICLE_SIZE"] or 150.0)
    return {"symmetry": pkg["SYMMETRY"] or "C1", "mask_radius_a": round(size * 0.75, 2),
            "search_range_x_a": round(size * 0.4, 2), "search_range_y_a": round(size * 0.4, 2),
            "number_of_classes": pkg["NUMBER_OF_CLASSES"] or 1}


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


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def _log(project_id, job_id, text, level="info"):
    _runtime.append_log(project_id, job_id, text, level=level)


def _load_state(conn, job_id):
    row = conn.execute("SELECT STATE_JSON FROM JOBS WHERE JOB_ID=?", (job_id,)).fetchone()
    return json.loads(row["STATE_JSON"]) if row is not None and row["STATE_JSON"] else None


def _save(conn, job_id, state, progress=None):
    with conn:
        if progress is None:
            conn.execute("UPDATE JOBS SET STATE_JSON=? WHERE JOB_ID=?", (json.dumps(state), job_id))
        else:
            conn.execute("UPDATE JOBS SET STATE_JSON=?, PROGRESS=? WHERE JOB_ID=?", (json.dumps(state), progress, job_id))


def scratch_dir(project_id, job_id):
    d = db.project_dir(project_id) / "Scratch" / "Startup" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parent_row(conn, job_id):
    return conn.execute("SELECT * FROM JOBS WHERE JOB_ID=?", (job_id,)).fetchone()


def _new_child(conn, parent_id, stage, name, parent_row):
    child_id = uuid.uuid4().hex[:10]
    with conn:
        conn.execute("INSERT INTO JOBS(JOB_ID, STAGE, JOB_NUMBER, NAME, PARAMS_JSON, STATUS, PROGRESS, CREATED_AT, PARENT_JOB_ID) "
                     "VALUES (?, ?, ?, ?, ?, 'queued', 0, ?, ?)",
                     (child_id, stage, parent_row["JOB_NUMBER"], name, parent_row["PARAMS_JSON"], now_iso(), parent_id))
    return child_id


def _profile(name):
    sys_conn = db.get_system_conn()
    try:
        return db.load_run_profile_by_name(sys_conn, name)
    finally:
        sys_conn.close()


def _task(adapter, index, ref, values):
    kinds = {"t": "text", "i": "int", "f": "float", "b": "bool"}
    if len(values) != len(adapter.ARGUMENT_TYPES):
        raise ValueError("{} expects {} arguments, got {}".format(adapter.PROGRAM["name"], len(adapter.ARGUMENT_TYPES), len(values)))
    return {"index": index, "ref": ref, "args": [jp.arg(kinds[t], v) for t, v in zip(adapter.ARGUMENT_TYPES, values)]}


def selection_for(conn, params):
    """The class selection of a class-average run: (selection dict with its
    classes, the member count of each class, the classification row)."""
    import classification
    sid = params.get("classification_selection_id")
    if sid in (None, ""):
        raise ValueError("Pick a class selection to build the reconstruction from.")
    sel = classification.get_selection(conn, int(sid))
    if sel is None:
        raise ValueError("class selection {} does not exist".format(sid))
    if not sel["classes"]:
        raise ValueError("the class selection {!r} has no classes in it".format(sel["name"]))
    table = classification.results_table(sel["classification_id"])
    counts = {k: conn.execute("SELECT COUNT(*) FROM {} WHERE BEST_CLASS = ?".format(table), (k,)).fetchone()[0] for k in sel["classes"]}
    if min(counts.values()) == 0:
        raise ValueError("a selected class of {!r} has no members".format(sel["name"]))
    cls = conn.execute("SELECT * FROM CLASSIFICATION_LIST WHERE CLASSIFICATION_ID=?", (sel["classification_id"],)).fetchone()
    return sel, counts, cls


def validate(conn, params):
    use_class_averages = str(params.get("input_mode") or "images").strip().lower().startswith("class")
    package_id = params.get("refinement_package_id")
    if use_class_averages:
        sel, _counts, _cls = selection_for(conn, params)
        package_id = sel["refinement_package_id"]
    if package_id in (None, ""):
        raise ValueError(PLEASE_CREATE_PACKAGE_MESSAGE)
    pkg = conn.execute("SELECT * FROM REFINEMENT_PACKAGE_ASSETS WHERE REFINEMENT_PACKAGE_ASSET_ID=?", (int(package_id),)).fetchone()
    if pkg is None:
        raise ValueError("refinement package {} does not exist".format(package_id))
    table = "REFINEMENT_PACKAGE_CONTAINED_PARTICLES_{}".format(int(package_id))
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is None:
        raise ValueError("refinement package {!r} has no contained particles".format(pkg["NAME"]))
    contained = conn.execute("SELECT * FROM {} ORDER BY POSITION_IN_STACK".format(table)).fetchall()
    if not contained:
        raise ValueError("refinement package {!r} contains no particles".format(pkg["NAME"]))
    if not pkg["STACK_FILENAME"] or not os.path.isfile(pkg["STACK_FILENAME"]):
        raise ValueError("the particle stack {} is missing".format(pkg["STACK_FILENAME"]))
    sym = str(params.get("symmetry") or pkg["SYMMETRY"] or "C1").strip().upper()
    if sym not in refinement_packages.SYMMETRIES:
        raise ValueError("unknown symmetry {!r}".format(sym))
    if _flag(params, "use_blush", False):
        info = blush.availability()
        if not info["available"]:
            raise ValueError("Blush cannot run on this server: {}".format(info["reason"]))
    return pkg, contained


def _initial_rows(contained, pixel_size, rng):
    """BeginRefinementCycle(): the package's particles with random angles,
    shifts within +-5 A, everything active, sigma 1."""
    rows = []
    for c in contained:
        phi, theta, psi = random_angles(rng)
        rows.append({
            "position_in_stack": c["POSITION_IN_STACK"], "image_is_active": 1, "psi": psi, "theta": theta, "phi": phi,
            "x_shift": rng.uniform(-1.0, 1.0) * 5.0, "y_shift": rng.uniform(-1.0, 1.0) * 5.0,
            "defocus_1": c["DEFOCUS_1"] or 0.0, "defocus_2": c["DEFOCUS_2"] or 0.0, "defocus_angle": c["DEFOCUS_ANGLE"] or 0.0,
            "phase_shift": c["PHASE_SHIFT"] or 0.0, "occupancy": 100.0, "logp": 0.0, "sigma": 1.0, "score": 0.0,
            "pixel_size": pixel_size, "voltage": c["MICROSCOPE_VOLTAGE"] or 300.0, "cs": c["SPHERICAL_ABERRATION"] or 2.7,
            "amplitude_contrast": c["AMPLITUDE_CONTRAST"] or 0.07, "beam_tilt_x": 0.0, "beam_tilt_y": 0.0,
            "image_shift_x": 0.0, "image_shift_y": 0.0, "assigned_subset": c["ASSIGNED_SUBSET"] or 1,
        })
    return rows


def _class_star(state, tag, k):
    """The scratch copy of class k's parameters under `tag`: cisTEM's binary form, read and written in one call."""
    return str(Path(state["scratch"]) / "refinement_{}_class{}.cistem".format(tag, k + 1))


def _load_rows(state, tag):
    """The per-class parameter tables stored under `tag`."""
    return [starfile.read_params(_class_star(state, tag, k), as_table=True) for k in range(state["number_of_classes"])]


def _store_rows(state, tag, class_rows):
    for k, rows in enumerate(class_rows):
        starfile.write_params(_class_star(state, tag, k), refinements.as_table(rows))



def start(conn, project_id, job_id, params, profile):
    """The first step, under the driver's lock: the child it launches can end
    before this returns (a controller that fails to launch does so inside
    submit()), and _child_finished() must then find the parent's state saved,
    not a parent that never hears of it and stays "running" forever."""
    with _job_lock(job_id):
        return _start(conn, project_id, job_id, params, profile)

def _start(conn, project_id, job_id, params, profile):
    """BeginRefinementCycle()."""
    pkg, contained = validate(conn, params)
    recon_profile = _profile(params.get("reconstruction_run_profile") or params.get("run_profile")) or profile
    use_class_averages = str(params.get("input_mode") or "images").strip().lower().startswith("class")
    selection = counts = classification_row = None
    if use_class_averages:
        selection, counts, classification_row = selection_for(conn, params)
    if not profile or profile.get("total_jobs", 0) <= 0:
        raise ValueError("run profile {!r} has no run commands, so it can't launch anything".format((profile or {}).get("name")))
    if not recon_profile or recon_profile.get("total_jobs", 0) <= 0:
        raise ValueError("reconstruction run profile {!r} has no run commands".format((recon_profile or {}).get("name")))
    s = settings_from_params(params, pkg)
    if use_class_averages:
        n2d = class_averages_per_class(min(counts.values()), s["images_per_class"], len(selection["classes"]))
        n = len(selection["classes"]) * n2d
        classes = s["number_of_classes"]
    else:
        n2d = 0
        n = len(contained)
        classes = max(1, int(pkg["NUMBER_OF_CLASSES"] or 1))
    scratch = scratch_dir(project_id, job_id)
    for p in scratch.iterdir():
        try:
            p.unlink()
        except OSError:
            pass
    pixel_size = float(pkg["OUTPUT_PIXEL_SIZE"] or contained[0]["PIXEL_SIZE"] or 1.0)
    box = int(pkg["STACK_BOX_SIZE"])
    plan = percent_used_plan(n, classes, s["symmetry"], s["auto_percent_used"], s["start_percent_used"], s["end_percent_used"])
    apply_symmetry = bool(s["always_apply_symmetry"] and s["symmetry"] != "C1")
    mask_radius = min(s["mask_radius"], box * 0.45 * pixel_size)
    state = {
        "phase": None, "start": 0, "round": 0, "starts": s["number_of_starts"], "rounds": s["number_of_rounds"],
        "settings": dict(s, mask_radius=mask_radius), "package_id": pkg["REFINEMENT_PACKAGE_ASSET_ID"], "package_name": pkg["NAME"],
        "stack_filename": pkg["STACK_FILENAME"], "pixel_size": pixel_size, "box_size": box,
        "invert_contrast": bool(pkg["STACK_HAS_WHITE_PROTEIN"]), "number_of_particles": n, "number_of_classes": classes,
        "molecular_weight": float(pkg["MOLECULAR_WEIGHT"] or 300.0), "particle_size": float(pkg["PARTICLE_SIZE"] or 150.0),
        "refinement_profile": profile["name"], "refinement_jobs": int(profile["total_jobs"]),
        "reconstruction_profile": recon_profile["name"], "reconstruction_jobs": int(recon_profile["total_jobs"]),
        "active_stack": pkg["STACK_FILENAME"], "active_pixel_size": pixel_size, "active_box": box, "stack_bin_factor": 1.0,
        "stack_precomputed": False, "apply_symmetry": apply_symmetry, "plan": plan,
        "current_percent_used": plan["sym_start"] if apply_symmetry else plan["start"],
        "current_high_res": s["initial_resolution_limit"], "next_high_res": s["initial_resolution_limit"],
        "reference_files": [None] * classes, "display_files": [None] * classes, "stats_files": [None] * classes,
        "child_job_id": None, "child_task_count": 0, "child_done": 0, "history": [], "started_at": now_iso(),
        "scratch": str(scratch), "initial": True, "iteration": 0,
        "use_class_averages": use_class_averages, "selection_id": selection["selection_id"] if selection else None,
        "selection_classes": selection["classes"] if selection else None, "classification_id": selection["classification_id"] if selection else None,
        "class_averages_per_class": n2d,
    }
    rng = random.Random()
    if use_class_averages:
        rows = _classaverage_rows(n, pixel_size, rng)
        _store_rows(state, "input", [list(r) for r in ([rows] * classes)])
    else:
        _store_rows(state, "input", [_initial_rows(contained, pixel_size, rng) for _ in range(classes)])
    with conn:
        conn.execute("UPDATE JOBS SET STATUS='running', STARTED_AT=?, PROGRESS=0 WHERE JOB_ID=?", (now_iso(), job_id))
    _log(project_id, job_id, "Ab-initio 3D of {!r}: {} {}, {} class{}, symmetry {}, {} start{} x {} round{}, refinement profile {!r}, reconstruction profile {!r}".format(
        pkg["NAME"], n, "class averages" if use_class_averages else "particles", classes, "" if classes == 1 else "es", s["symmetry"],
        s["number_of_starts"], "" if s["number_of_starts"] == 1 else "s", s["number_of_rounds"], "" if s["number_of_rounds"] == 1 else "s",
        profile["name"], recon_profile["name"]))
    if use_class_averages:
        _log(project_id, job_id, "From selection {!r} of {}: {} class{} ({} members in the smallest), {} averages of {} images per class".format(
            selection["name"], classification_row["NAME"] if classification_row else "classification {}".format(selection["classification_id"]),
            len(selection["classes"]), "" if len(selection["classes"]) == 1 else "es", min(counts.values()), n2d, s["images_per_class"]))
    if s["auto_percent_used"]:
        _log(project_id, job_id, "Percent used: {:.2f}% at the start, {:.2f}% at the end{}".format(
            plan["start"], plan["end"], " ({:.2f}% -> {:.2f}% once symmetry is applied)".format(plan["sym_start"], plan["sym_end"]) if s["symmetry"] != "C1" else ""))
    if use_class_averages:
        _launch_prepare_classaverages(conn, project_id, job_id, state)
    elif s["final_resolution_limit"] > pixel_size * 3.0:
        _launch_prepare_stack(conn, project_id, job_id, state)
    else:
        _launch_reconstruction(conn, project_id, job_id, state)
    _save(conn, job_id, state)
    return state


def _classaverage_rows(n, pixel_size, rng):
    """The refinement a class-average run starts from: n averages, no CTF
    (they are CTF-corrected sums), 300 kV / 2.7 mm / 0.07, half-sets
    alternating (SetAssignedSubsetToEvenOdd), random angles."""
    rows = []
    for i in range(1, n + 1):
        phi, theta, psi = random_angles(rng)
        rows.append({
            "position_in_stack": i, "image_is_active": 1, "psi": psi, "theta": theta, "phi": phi,
            "x_shift": rng.uniform(-1.0, 1.0) * 5.0, "y_shift": rng.uniform(-1.0, 1.0) * 5.0,
            "defocus_1": 0.0, "defocus_2": 0.0, "defocus_angle": 0.0, "phase_shift": 0.0, "occupancy": 100.0, "logp": 0.0,
            "sigma": 1.0, "score": 0.0, "pixel_size": pixel_size, "voltage": 300.0, "cs": 2.7, "amplitude_contrast": 0.07,
            "beam_tilt_x": 0.0, "beam_tilt_y": 0.0, "image_shift_x": 0.0, "image_shift_y": 0.0, "assigned_subset": 1 if (i - 1) % 2 == 0 else 2,
        })
    return rows


def _launch_prepare_classaverages(conn, project_id, job_id, state):
    """SetupPrepareStackJob(), class-average branch: the selection as a text
    file of class numbers, the classification as a star file, and one
    prepare_stack_classaverage task per share of the selected classes."""
    import classification
    s = state["settings"]
    scratch = Path(state["scratch"])
    classes = state["selection_classes"]
    selection_file = str(scratch / "class_average_selection.txt")
    with open(selection_file, "w") as fh:
        for k in classes:
            fh.write("{:f}\n".format(float(k)))
    particles = classification.package_particle_table(conn, state["package_id"])
    star = classification.write_params(scratch / "classification_star_{}.cistem".format(state["classification_id"]),
                                       classification.classification_rows(conn, state["classification_id"], particles))
    binning = (s["final_resolution_limit"] / 2.0) / state["pixel_size"]
    wanted_box = binned_box_size(state["box_size"], binning)
    resample = wanted_box < state["box_size"]
    out_stack = str(scratch / "temp_stack.mrc")
    if os.path.exists(out_stack):
        os.remove(out_stack)
    tasks = []
    for idx, (first, last) in enumerate(classaverage_job_ranges(len(classes), state["refinement_jobs"])):
        tasks.append(_task(prepare_stack_classaverage, idx, idx + 1, [state["stack_filename"], out_stack, star, selection_file, state["pixel_size"],
                                                                     state["particle_size"] * 0.6, resample, wanted_box, state["class_averages_per_class"],
                                                                     s["images_per_class"], True, True, first, last]))
    parent = _parent_row(conn, job_id)
    child = _new_child(conn, job_id, CHILD_PREPARE_CLASSAVG, "{} · prepare class averages".format(parent["NAME"]), parent)
    state.update({"phase": "prepare", "child_job_id": child, "child_task_count": len(tasks), "child_done": 0, "temp_stack": out_stack,
                  "wanted_box": wanted_box if resample else state["box_size"]})
    _log(project_id, job_id, "Preparing {} class averages ({} task{}, box {} -> {} px) — child job {}".format(
        state["number_of_particles"], len(tasks), "" if len(tasks) == 1 else "s", state["box_size"], wanted_box if resample else state["box_size"], child))
    _runtime.submit_child(project_id, child, prepare_stack_classaverage, tasks, _profile(state["refinement_profile"]))


def _launch_prepare_stack(conn, project_id, job_id, state):
    s = state["settings"]
    n = state["number_of_particles"]
    jobs = prepare_stack_jobs(n, state["refinement_jobs"])
    binning = (s["final_resolution_limit"] / 2.0) / state["pixel_size"]
    wanted_box = binned_box_size(state["box_size"], binning)
    # ONLY WRITING FIRST CLASS FOR PIXEL SIZES (cisTEM's own comment).
    star = str(Path(state["scratch"]) / "prepare_stack_input.cistem")
    starfile.write_params(star, _load_rows(state, "input")[0])
    out_stack = str(Path(state["scratch"]) / "temp_stack.mrc")
    if os.path.exists(out_stack):
        os.remove(out_stack)
    tasks = []
    for k in range(1, jobs + 1):
        first, last = particle_range(k, jobs, n)
        tasks.append(_task(prepare_stack, k - 1, k, [state["stack_filename"], star, out_stack, state["pixel_size"], s["mask_radius"],
                                                      True, wanted_box, True, first, last]))
    parent = _parent_row(conn, job_id)
    child = _new_child(conn, job_id, CHILD_PREPARE, "{} · prepare stack".format(parent["NAME"]), parent)
    state.update({"phase": "prepare", "child_job_id": child, "child_task_count": jobs, "child_done": 0, "temp_stack": out_stack, "wanted_box": wanted_box})
    _log(project_id, job_id, "Preparing input stack ({} task{}, box {} -> {} px) — child job {}".format(jobs, "" if jobs == 1 else "s", state["box_size"], wanted_box, child))
    _runtime.submit_child(project_id, child, prepare_stack, tasks, _profile(state["refinement_profile"]))


def _output_number(state):
    return state["rounds"] * state["start"] + state["round"]


RECON_PERCENT_MULTIPLIER = 5.0   # SetupRefinementJob() refines this many times the percent used; the reconstruction keeps a fifth of them


def reconstruction_rows(rows, initial, current_percent_used, rng):
    """The rows reconstruct3d reads for one class. The initial reconstruction
    draws its particles at random and writes sigma 10 (WritecisTEMStarFiles'
    percent_used_override / sigma_override); a refinement round hands over
    refine3d's output with sigma 1 -- and gives every particle refine3d left
    inactive a score one below the lowest refined score. reconstruct3d's
    percentage threshold ranks *all* rows by score, and an inactive row
    keeps the score of whichever earlier round last refined it; at a new
    start those are 8 A scores against this round's 20 A ones, and in the
    run that showed this up they crowded out the refined particles until
    the map was built from nine of them, then none. Pushed below the refined
    minimum they still count in the denominator but never win a place, so
    the threshold ranks only this round's refined particles, as the comment
    beside cisTEM's 0.2 says it means to. (reconstruct3d skips inactive
    rows regardless, so their score is never used for anything else.)"""
    table = refinements.as_table(rows).copy()
    n = len(table)
    if initial:
        draws = _uniform_draws(rng, n)
        table["image_is_active"] = np.where(draws < 1.0 - 2.0 * current_percent_used / 100.0, -1, 1)
        table["sigma"] = 10.0
    else:
        table["sigma"] = 1.0
        active = table["image_is_active"] >= 0
        floor = (float(table["score"][active].min()) if active.any() else 0.0) - 1.0
        table["score"][~active] = floor
    return table


def _uniform_draws(rng, n):
    """n draws in [-1, 1) from `rng`: a numpy Generator, or a random.Random seeding one (the drivers' choice)."""
    if isinstance(rng, np.random.Generator):
        return rng.uniform(-1.0, 1.0, n)
    return np.random.default_rng(rng.getrandbits(64)).uniform(-1.0, 1.0, n)


def reconstruction_score_threshold(current_percent_used, initial=False):
    """reconstruct3d's `score_threshold` (< 1 is a fraction of the rows, by
    score). cisTEM's SetupReconstructionJob() passes 0.2 while the
    refinement refines fewer than all particles (RECON_PERCENT_MULTIPLIER x
    the percent used) and 1.0 once it refines them all. With the inactive
    rows ranked below the refined ones (reconstruction_rows), the fraction
    that keeps the top fifth of the refined particles -- what the 0.2 was
    for -- is the percent used itself; the initial reconstruction keeps
    cisTEM's literal values (all its scores are zero anyway)."""
    if current_percent_used * RECON_PERCENT_MULTIPLIER >= 100.0:
        return 1.0
    if initial:
        return 0.2
    return max(current_percent_used / 100.0, 1e-4)


def _launch_reconstruction(conn, project_id, job_id, state):
    """SetupReconstructionJob() + RunReconstructionJob()."""
    s = state["settings"]
    n, classes = state["number_of_particles"], state["number_of_classes"]
    initial = state["initial"]
    rng = random.Random()
    class_rows = _load_rows(state, "input" if initial else "output")
    written = []
    for k, rows in enumerate(class_rows):
        out = reconstruction_rows(rows, initial, state["current_percent_used"], rng)
        p = str(Path(state["scratch"]) / "recon_input_{}_class{}.cistem".format(_output_number(state), k + 1))
        starfile.write_params(p, out)
        written.append(p)
    jobs = max(1, min(n, state["reconstruction_jobs"]))
    score_threshold = reconstruction_score_threshold(state["current_percent_used"], initial)
    symmetry = s["symmetry"] if state["apply_symmetry"] else "C1"
    scratch = Path(state["scratch"])
    tasks = []
    index = 0
    for k in range(classes):
        for j in range(1, jobs + 1):
            first, last = particle_range(j, jobs, n)
            use_ref = bool(s["apply_blurring"] and not initial and state["reference_files"][k])
            values = [state["active_stack"], written[k], state["reference_files"][k] if use_ref else "/dev/null",
                      "/dev/null", "/dev/null", "/dev/null", "/dev/null", symmetry, first, last,
                      state["active_pixel_size"], state["molecular_weight"], s["inner_mask_radius"], s["mask_radius"],
                      state["next_high_res"], state["current_high_res"], 0.0, score_threshold, s["smoothing_factor"], 1.0,
                      (not state["stack_precomputed"]) or state.get("use_class_averages", False), False, state["invert_contrast"], False, False, False, True,
                      use_ref, False, True,
                      str(scratch / "startup_dump_file_{}_odd_{}.dump".format(k, j)),
                      str(scratch / "startup_dump_file_{}_even_{}.dump".format(k, j)), 0, 1]
            tasks.append(_task(reconstruct3d, index, k * 1000000 + j, values))
            index += 1
    parent = _parent_row(conn, job_id)
    label = "initial reconstruction" if initial else "reconstruction {}".format(_output_number(state) + 1)
    child = _new_child(conn, job_id, CHILD_RECON, "{} · {}".format(parent["NAME"], label), parent)
    state.update({"phase": "initial_recon" if initial else "recon", "child_job_id": child, "child_task_count": len(tasks), "child_done": 0,
                  "number_of_dump_files": jobs})
    _log(project_id, job_id, "Calculating {}{} ({} task{}) — child job {}".format("initial " if initial else "", "reconstructions" if classes > 1 else "reconstruction",
                                                                                   len(tasks), "" if len(tasks) == 1 else "s", child))
    _runtime.submit_child(project_id, child, reconstruct3d, tasks, _profile(state["reconstruction_profile"]))


def _launch_merge(conn, project_id, job_id, state):
    """SetupMerge3dJob() + RunMerge3dJob()."""
    s = state["settings"]
    classes = state["number_of_classes"]
    scratch = Path(state["scratch"])
    n_out = _output_number(state)
    outer = min(s["mask_radius"], state["active_box"] * 0.45 * state["active_pixel_size"])
    wiener = wiener_nominator(state["round"], state["rounds"], state["start"])
    tasks, outputs, stats, halves = [], [], [], []
    for k in range(classes):
        name = "startup3d_initial_{}_{}.mrc".format(n_out, k) if state["initial"] else "startup3d_{}_{}.mrc".format(n_out, k)
        out = str(scratch / name)
        st = str(scratch / "startup3d_stats_{}_{}.txt".format(n_out, k))
        # With Blush on, the half maps are kept for the round: their average is the unfiltered reconstruction it denoises.
        half = [str(scratch / "startup3d_half_{}_{}_{}.mrc".format(n_out, k, h)) for h in (1, 2)] if s["use_blush"] else ["/dev/null", "/dev/null"]
        outputs.append(out)
        stats.append(st)
        halves.append(half)
        tasks.append(_task(merge3d, k, k + 1, [half[0], half[1], out, st, state["molecular_weight"], s["inner_mask_radius"], outer,
                                                str(scratch / "startup_dump_file_{}_odd_.dump".format(k)), str(scratch / "startup_dump_file_{}_even_.dump".format(k)),
                                                k + 1, False, "", _required_count(state, "number_of_dump_files"), wiener, state["current_high_res"]]))
    parent = _parent_row(conn, job_id)
    child = _new_child(conn, job_id, CHILD_MERGE, "{} · merge {}".format(parent["NAME"], "initial" if state["initial"] else n_out + 1), parent)
    state.update({"phase": "initial_merge" if state["initial"] else "merge", "child_job_id": child, "child_task_count": len(tasks), "child_done": 0,
                  "pending_reference_files": outputs, "pending_stats_files": stats, "pending_half_maps": halves if s["use_blush"] else []})
    _log(project_id, job_id, "Merging and filtering {} (Wiener nominator {:.0f}) — child job {}".format("reconstructions" if classes > 1 else "reconstruction", wiener, child))
    _runtime.submit_child(project_id, child, merge3d, tasks, _profile(state["reconstruction_profile"]))


def _launch_refinement(conn, project_id, job_id, state):
    """SetupRefinementJob() + RunRefinementJob(), after the masking DoMasking() does first."""
    s = state["settings"]
    n, classes = state["number_of_particles"], state["number_of_classes"]
    rng = random.Random()
    scratch = Path(state["scratch"])
    class_rows = _load_rows(state, "input")
    defaults = default_statistics(state["molecular_weight"], state["active_pixel_size"], state["active_box"])
    star_files, stats_files = [], []
    for k, rows in enumerate(class_rows):
        for r in rows:
            r["occupancy"] = 100.0 if classes == 1 else 100.0 / classes
            r["phi"], r["theta"], r["psi"] = rng.uniform(-1, 1) * 180.0, rng.uniform(-1, 1) * 180.0, rng.uniform(-1, 1) * 180.0
            r["x_shift"] = r["y_shift"] = 0.0
        p = str(scratch / "refine_input_{}_class{}.cistem".format(_output_number(state), k + 1))
        starfile.write_params(p, rows)
        star_files.append(p)
        if state["round"] < 3 or not state["stats_files"][k] or not os.path.isfile(state["stats_files"][k]):
            stats = defaults
        else:
            stats = cap_part_ssnr(read_statistics(state["stats_files"][k]), defaults)
        sp = str(scratch / "refine_stats_{}_class{}.txt".format(_output_number(state), k + 1))
        write_statistics(sp, stats, state["active_pixel_size"])
        stats_files.append(sp)
    jobs = max(1, min(n, state["refinement_jobs"]))
    percent = min(100.0, state["current_percent_used"] * 5.0) / 100.0
    symmetry = s["symmetry"] if state["apply_symmetry"] else "C1"
    tasks, outputs = [], []
    index = 0
    for k in range(classes):
        for j in range(1, jobs + 1):
            first, last = particle_range(j, jobs, n)
            out_star = str(scratch / "refine_output_{}_class{}_{}.cistem".format(_output_number(state), k + 1, j))
            outputs.append(out_star)
            values = [state["active_stack"], star_files[k], state["reference_files"][k], stats_files[k], True, "", out_star, "/dev/null",
                      symmetry, first, last, percent, state["active_pixel_size"], state["molecular_weight"], s["inner_mask_radius"], s["mask_radius"],
                      state["particle_size"], state["current_high_res"], signed_cc_limit(state["round"], state["rounds"]), 8.0, s["mask_radius"],
                      state["current_high_res"], angular_step(state["current_high_res"]), -100000, s["search_range_x"], s["search_range_y"],
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                      True, False, True, True, True, True, True, False, False, False,
                      (not state["stack_precomputed"]) or state.get("use_class_averages", False), state["invert_contrast"], False, not s["apply_blurring"], False,
                      1, False, k, False, False]
            tasks.append(_task(refine3d, index, k * 1000000 + j, values))
            index += 1
    parent = _parent_row(conn, job_id)
    child = _new_child(conn, job_id, CHILD_REFINE, "{} · start {} round {}".format(parent["NAME"], state["start"] + 1, state["round"] + 1), parent)
    state.update({"phase": "refine", "child_job_id": child, "child_task_count": len(tasks), "child_done": 0, "pending_output_stars": outputs,
                  "refinement_jobs_this_round": jobs})
    _log(project_id, job_id, "Running refinement round {:2d} of {:2d} ({:.2f} Å / {:.2f} %) - Start {:2d} of {:2d} — child job {}".format(
        state["round"] + 1, state["rounds"], state["current_high_res"], state["current_percent_used"], state["start"] + 1, state["starts"], child))
    _runtime.submit_child(project_id, child, refine3d, tasks, _profile(state["refinement_profile"]))


def _prepare_references_then_refine(conn, project_id, job_id, state):
    """Between a merge and the next refine3d: Blush (its own phase) when asked for, then masking."""
    if state["settings"]["use_blush"]:
        blush_phase.start(_BLUSH, conn, project_id, job_id, state, blush_phase.PRE)
    else:
        _mask_then_refine(conn, project_id, job_id, state)


def _round_statistics(conn, state, k):
    """Class k's statistics for the volumes the state points at (the merge's file, set at the same time as the references)."""
    files = state.get("stats_files") or []
    if k < len(files) and files[k] and os.path.isfile(files[k]):
        return read_statistics(files[k])
    return []


def _half_maps(state, k):
    halves = state.get("pending_half_maps") or []
    return halves[k] if k < len(halves) else []


def _blush_mask_radius(state):
    s = state["settings"]
    return min(s["mask_radius"], state["active_box"] * 0.45 * state["active_pixel_size"])


# Ab-initio's round volumes live in scratch and are rewritten under the same name every round, so a
# companion sits beside its volume (blush_phase.sibling_companion) and counts only while it is newer
# than the volume; the exported startup volume gets one in Assets/Volumes/Blushed (_take_files).
_BLUSH = blush_phase.Host(module=sys.modules[__name__],
                          continue_pre=lambda conn, project_id, job_id, state: _mask_then_refine(conn, project_id, job_id, state),
                          continue_post=lambda conn, project_id, job_id, state: _cycle(conn, project_id, job_id, state),
                          pixel_size=lambda state: state["active_pixel_size"],
                          mask_radius=_blush_mask_radius,
                          statistics=_round_statistics,
                          companion_path=lambda project_id, state, ref: blush_phase.sibling_companion(ref),
                          half_maps=_half_maps,
                          unfiltered=lambda state: True)


def _mask_then_refine(conn, project_id, job_id, state):
    """DoMasking() then the refinement (or the refinement straight away)."""
    if state["settings"]["auto_mask"]:
        _log(project_id, job_id, "Automasking reference reconstruction")
        masked = []
        for k, ref in enumerate(state["reference_files"]):
            vol, _ = volumes.read_mrc_volume(ref)   # the prepared stack's pixel size, not the file header's, is what the run works in
            out = str(Path(ref).with_suffix("")) + "_masked.mrc"
            volumes.write_mrc_volume(out, volumes.auto_mask(vol, state["active_pixel_size"], state["settings"]["mask_radius"]), state["active_pixel_size"])
            masked.append(out)
        state["reference_files"] = masked
    _launch_refinement(conn, project_id, job_id, state)


def _align_symmetry(project_id, job_id, state):
    """SetupAlignSymmetryJob() + ImposeAlignmentAndSymmetryThread, as one
    local run of align_symmetry per class, which writes the aligned and
    symmetrised volume itself."""
    # From where the refinement profile's commands say the workers live (a directory on
    # $command), else PATH -- the same rule the picking preview uses.
    candidate = job_runner.local_program_of(_profile(state["refinement_profile"]), "align_symmetry")
    exe = shutil.which(candidate)
    if not exe:
        raise ValueError("align_symmetry was not found (looked for {!r}, from run profile {!r})".format(candidate, state["refinement_profile"]))
    _log(project_id, job_id, "Aligning to {} symmetry and applying it from here on".format(state["settings"]["symmetry"]))
    progress_store.note(job_id, 0, 0, "aligning symmetry (align_symmetry)")
    out_files = []
    for k, ref in enumerate(state["reference_files"]):
        base = str(Path(ref).with_suffix(""))
        no_sym, with_sym = base + "_ali.mrc", base + "_sym.mrc"
        # align_symmetry's questions, in order: input volume, symmetry, aligned output, symmetrised
        # output, start and end angle of the search on each axis, initial angular step. The angles
        # are AbInitioManager::SetupAlignSymmetryJob()'s whole range (it splits it across processes;
        # one local run covers all of it) and its 4 degree step.
        answers = "\n".join([ref, state["settings"]["symmetry"], no_sym, with_sym, "-90.0", "90.0", "4.0"]) + "\n"
        proc = subprocess.run([exe], input=answers, capture_output=True, text=True, cwd=state["scratch"], timeout=3600)
        if proc.returncode != 0 or not os.path.isfile(with_sym):
            tail = (proc.stdout or "").strip().splitlines()[-3:]
            raise ValueError("align_symmetry failed on class {}: {}".format(k + 1, " | ".join(tail) or (proc.stderr or "")[-300:]))
        for line in (proc.stdout or "").splitlines():
            if "Rot =" in line or "Shift =" in line:
                _log(project_id, job_id, "  " + line.strip())
        out_files.append(with_sym)
    state["reference_files"] = out_files
    state["display_files"] = out_files
    volumes.prepare_orth_views(out_files, state["settings"]["mask_radius"], lambda t: _log(project_id, job_id, t, "warning"))
    state["apply_symmetry"] = True
    sched = round_schedule(state["round"], state["rounds"], state["settings"]["initial_resolution_limit"], state["settings"]["final_resolution_limit"], state["plan"], True)
    state["current_percent_used"] = sched["percent_used"]


def _merge_output_stars(state):
    """The per-task output parameter files of a refinement round -> one table per class."""
    n_classes = state["number_of_classes"]
    inputs = _load_rows(state, "input")
    jobs = _required_count(state, "refinement_jobs_this_round")
    outputs = state.get("pending_output_stars") or []
    class_rows = []
    for k in range(n_classes):
        paths = [outputs[k * jobs + j] if k * jobs + j < len(outputs) else None for j in range(jobs)]
        try:
            class_rows.append(starfile.merge_task_outputs(inputs[k], paths))
        except ValueError as exc:
            raise ValueError("{} (class {})".format(exc, k + 1)) if "class" not in str(exc) else exc
    return class_rows


def _progress_percent(state):
    total = float(state["starts"] * state["rounds"]) + 1.0
    done = 1.0 if not state["initial"] else 0.0
    done += float(state["start"] * state["rounds"] + state["round"])
    frac = float(state.get("child_done", 0)) / float(max(state.get("child_task_count", 1), 1))
    phase = state.get("phase")
    if phase in ("prepare",):
        done = 0.0
    elif phase == "initial_recon":
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
    return max(0, min(100, int(100.0 * done / total)))


def progress_info(state):
    if not state:
        return {}
    total = int(state["starts"] * state["rounds"]) + 1
    finished = [h["finished_at"] for h in state.get("history", []) if h.get("finished_at")]
    return {"task_count": total, "tasks_done": len(finished),
            "first_task_finished_at": finished[0] if finished else None,
            "last_task_finished_at": finished[-1] if finished else None,
            "round": state["round"], "rounds": state["rounds"], "start": state["start"], "starts": state["starts"], "phase": state["phase"]}



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
                     daemon=True, name="abinitio-" + child_row["PARENT_JOB_ID"]).start()


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
            if state.get("pending_action"):
                # Take Current / Take Last Start stopped this step on purpose.
                _perform_pending_action(conn, project_id, parent_id, state)
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
    if phase == "prepare":
        # The controller's master writes the prepared stack slice by slice
        # and its header last; cisTEM sleeps a second here for the same
        # reason. Wait until the header counts every particle.
        hdr = volumes.read_mrc_header(state["temp_stack"])
        for _ in range(60):
            if hdr["nz"] >= state["number_of_particles"]:
                break
            time.sleep(0.5)
            hdr = volumes.read_mrc_header(state["temp_stack"])
        if hdr["nz"] < state["number_of_particles"]:
            raise ValueError("the prepared stack holds {} of {} particles".format(hdr["nz"], state["number_of_particles"]))
        new_box = hdr["nx"]
        state["stack_bin_factor"] = float(state["box_size"]) / float(new_box)
        state["active_pixel_size"] = state["pixel_size"] * state["stack_bin_factor"]
        state["active_box"] = new_box
        state["active_stack"] = state["temp_stack"]
        state["stack_precomputed"] = True
        rows = _load_rows(state, "input")
        for cls in rows:
            for r in cls:
                r["pixel_size"] = state["active_pixel_size"]
        _store_rows(state, "input", rows)
        _log(project_id, parent_id, "Stack prepared: {} particles at {} px, {:.4f} Å/px (bin {:.3f})".format(hdr["nz"], new_box, state["active_pixel_size"], state["stack_bin_factor"]))
        _launch_reconstruction(conn, project_id, parent_id, state)
    elif phase in ("initial_recon", "recon"):
        _launch_merge(conn, project_id, parent_id, state)
    elif phase in ("initial_merge", "merge"):
        for p in state["pending_reference_files"]:
            if not os.path.isfile(p):
                raise ValueError("merge3d did not write {}".format(p))
        state["reference_files"] = list(state["pending_reference_files"])
        state["display_files"] = list(state["pending_reference_files"])
        volumes.prepare_orth_views(state["display_files"], state["settings"]["mask_radius"], lambda t: _log(project_id, parent_id, t, "warning"))
        state["stats_files"] = list(state["pending_stats_files"])
        for p in Path(state["scratch"]).glob("startup_dump_file_*.dump"):
            try:
                p.unlink()
            except OSError:
                pass
        progress_store.note(parent_id, 0, 0, "recording the round")
        if not state["initial"]:
            class_rows = _load_rows(state, "output")
            stats = [read_statistics(p) if os.path.isfile(p) else [] for p in state["stats_files"]]
            avgs = pooled_part_ssnr(stats, class_rows)
            if state["number_of_classes"] > 1:
                for k, a in enumerate(avgs):
                    _log(project_id, parent_id, "   Occupancy for Class {:2d} = {:.2f} %".format(k + 1, a))
                for k, p in enumerate(state["stats_files"]):
                    if stats[k]:
                        write_statistics(p, stats[k], state["active_pixel_size"])
        if s["use_blush"]:
            blush_phase.start(_BLUSH, conn, project_id, parent_id, state, blush_phase.POST)
        else:
            _cycle(conn, project_id, parent_id, state)
    elif phase == "refine":
        progress_store.note(parent_id, 0, 0, "merging the round's results")
        class_rows = _merge_output_stars(state)
        update_occupancies(class_rows)
        _store_rows(state, "output", class_rows)
        for p in state.get("pending_output_stars") or []:
            try:
                os.remove(p)
            except OSError:
                pass
        _launch_reconstruction(conn, project_id, parent_id, state)
    else:
        raise ValueError("unexpected phase {!r}".format(phase))
    if state["phase"] != "finished":
        _save(conn, parent_id, state, _progress_percent(state))


def _cycle(conn, project_id, parent_id, state):
    """CycleRefinement()."""
    s = state["settings"]
    if state["initial"]:
        state["initial"] = False
        state["history"].append({"iteration": 0, "label": "Random Start", "average_sigma": None, "finished_at": now_iso(),
                                 "high_res": state["current_high_res"], "percent_used": state["current_percent_used"]})
        _prepare_references_then_refine(conn, project_id, parent_id, state)
        return
    state["round"] += 1
    class_rows = _load_rows(state, "output")
    sigma = average_sigma(class_rows)
    iteration = state["rounds"] * state["start"] + state["round"]
    state["history"].append({"iteration": iteration, "label": "Iter. #{},{}".format(state["start"], state["round"]), "average_sigma": sigma,
                             "finished_at": now_iso(), "high_res": state["current_high_res"], "percent_used": state["current_percent_used"]})
    _log(project_id, parent_id, "Round {} of {} (start {}) done — average sigma {}".format(
        state["round"], state["rounds"], state["start"] + 1, "{:.4f}".format(sigma) if sigma is not None else "n/a"))
    if state["round"] < state["rounds"]:
        _store_rows(state, "input", class_rows)
        sched = round_schedule(state["round"], state["rounds"], s["initial_resolution_limit"], s["final_resolution_limit"], state["plan"], state["apply_symmetry"])
        state.update({"current_high_res": sched["high_res"], "next_high_res": sched["next_high_res"], "current_percent_used": sched["percent_used"]})
        at_three_quarters = state["round"] == int(math.floor(state["rounds"] * 0.75 + 0.5))
        if at_three_quarters and s["symmetry"] != "C1" and not s["always_apply_symmetry"] and state["start"] == 0:
            # align_symmetry runs here for minutes: record that the merge is consumed and
            # the alignment is what is running, so a server restart resumes at the
            # alignment instead of finishing the round a second time.
            state["phase"] = "align_symmetry"; state["child_job_id"] = None
            _save(conn, parent_id, state)
            _align_symmetry(project_id, parent_id, state)
        _prepare_references_then_refine(conn, project_id, parent_id, state)
        return
    state["start"] += 1
    if state["start"] < state["starts"]:
        state["round"] = 0
        _store_rows(state, "input", class_rows)
        sched = round_schedule(0, state["rounds"], s["initial_resolution_limit"], s["final_resolution_limit"], state["plan"], state["apply_symmetry"])
        state.update({"current_high_res": sched["high_res"], "next_high_res": sched["next_high_res"], "current_percent_used": sched["percent_used"]})
        _log(project_id, parent_id, "Start {} of {} finished; restarting from its result".format(state["start"], state["starts"]))
        _prepare_references_then_refine(conn, project_id, parent_id, state)
        return
    state["start"] -= 1
    _take_current(conn, project_id, parent_id, state)


# The buttons AbInitio3DPanel shows beside Terminate while it runs
# (TakeCurrentClicked / TakeLastStartClicked): stop the job and keep a
# reconstruction as the run's result instead of nothing.
ACTIONS = {"take_current": "Take Current Result", "take_last_start": "Take Last Start Result"}


def available_actions(state):
    """OnUpdateUI()'s rule: Take Current once a round has finished, Take
    Last Start once a whole start has."""
    if not state or state.get("phase") in (None, "finished") or state.get("pending_action"):
        return []
    out = []
    if (state.get("round", 0) > 0 or state.get("start", 0) > 0) and any(state.get("display_files") or []):
        out.append({"name": "take_current", "label": ACTIONS["take_current"]})
    if state.get("start", 0) > 0:
        out.append({"name": "take_last_start", "label": ACTIONS["take_last_start"]})
    return out


def last_start_files(state):
    """TakeLastStart(): the reconstructions at the end of the previous start,
    startup3d_<rounds * starts_run - 1>_<class>.mrc in the scratch directory."""
    n = state["rounds"] * state["start"] - 1
    return [str(Path(state["scratch"]) / "startup3d_{}_{}.mrc".format(n, k)) for k in range(state["number_of_classes"])]


def perform_action(conn, project_id, parent_id, name):
    """Record the wish and stop the running step; _child_finished() carries it
    out once the controller has wound down (or now, if nothing is running)."""
    if name not in ACTIONS:
        raise ValueError("unknown action {!r}".format(name))
    state = _load_state(conn, parent_id)
    if not any(a["name"] == name for a in available_actions(state)):
        raise ValueError("{} is not available right now".format(ACTIONS[name]))
    state["pending_action"] = name
    with conn:   # the flag alone, in place: a whole-state write from this request thread could put back a stale step
        conn.execute("UPDATE JOBS SET STATE_JSON = json_set(STATE_JSON, '$.pending_action', ?) WHERE JOB_ID = ?", (name, parent_id))
    _log(project_id, parent_id, "Terminating job, and importing the {}.".format("current result" if name == "take_current" else "result at the end of the previous start"))
    child_id = state.get("child_job_id")
    if child_id and _runtime.cancel(child_id):
        return
    _perform_pending_action(conn, project_id, parent_id, state)


def _perform_pending_action(conn, project_id, parent_id, state):
    name = state.pop("pending_action", None)
    if name == "take_last_start":
        files = last_start_files(state)
        missing = [f for f in files if not os.path.isfile(f)]
        if missing:
            _finish(conn, project_id, parent_id, state, "failed", "the previous start's reconstruction is gone: {}".format(missing[0]))
            return
        _take_files(conn, project_id, parent_id, state, files, "the reconstruction at the end of start {}".format(state["start"]))
    else:
        files = [f for f in (state.get("display_files") or []) if f and os.path.isfile(f)]
        if len(files) != state["number_of_classes"]:
            _finish(conn, project_id, parent_id, state, "failed", "no current reconstruction to keep")
            return
        _take_files(conn, project_id, parent_id, state, files, "the current reconstruction")


def _take_current(conn, project_id, parent_id, state):
    """TakeCurrent() at the natural end of the run."""
    _take_files(conn, project_id, parent_id, state, list(state["display_files"]), None)


def _take_files(conn, project_id, parent_id, state, files, what):
    """TakeCurrent() + OnVolumeResampled(): reconstructions resampled to the
    package's box, registered as volume assets, and the run recorded in
    STARTUP_LIST. `what` names an early result in the log; None is the
    natural end of the run."""
    s = state["settings"]
    startup_id = volumes.next_startup_id(conn)
    vol_dir = volumes.volume_dir(project_id)
    volume_ids = []
    for k, ref in enumerate(files):
        vol, _ps = volumes.read_mrc_volume(ref)
        resampled = volumes.fourier_resize(vol, state["box_size"])
        out = str(vol_dir / "startup_volume_{}_{}.mrc".format(startup_id, k + 1))
        volumes.write_mrc_volume(out, resampled, state["pixel_size"])
        companion = blush_phase.sibling_companion(ref)
        if s.get("use_blush") and blush_phase.companion_is_current(companion, ref):
            # The round's Blush companion goes with the exported volume, resampled the same way, so a
            # refinement started from this volume with Blush on finds it ready.
            cvol, _cps = volumes.read_mrc_volume(companion)
            cout = blush_phase.assets_companion(project_id, out)
            Path(cout).parent.mkdir(parents=True, exist_ok=True)
            volumes.write_mrc_volume(cout, volumes.fourier_resize(cvol, state["box_size"]), state["pixel_size"])
        vid = volumes.add_volume_asset(conn, "Volume From Startup #{} - Class #{}".format(startup_id, k + 1), out, state["pixel_size"],
                                       state["box_size"], state["box_size"], state["box_size"])
        volume_ids.append(vid)
    volumes.add_startup_job(conn, startup_id, state["package_id"], "Refinement #{}".format(startup_id), s, volume_ids, job_id=parent_id)
    state.update({"phase": "finished", "child_job_id": None, "startup_id": startup_id, "volume_ids": volume_ids})
    for p in Path(state["scratch"]).glob("*"):
        if p.name.startswith(("refine_output_", "recon_input_", "refine_input_", "refine_stats_", "startup_dump_file_")):
            try:
                p.unlink()
            except OSError:
                pass
    _log(project_id, parent_id, "{} Volume{} {} written to Assets/Volumes (Startup #{})".format(
        "All refinement cycles are finished!" if what is None else "Kept {} at the user's request.".format(what),
        "" if len(volume_ids) == 1 else "s", ", ".join("#{}".format(v) for v in volume_ids), startup_id))
    _finish(conn, project_id, parent_id, state, "completed", None)


def _finish(conn, project_id, parent_id, state, status, error):
    metrics = {"starts_run": state.get("start", 0) + (1 if status == "completed" else 0), "rounds_run": state.get("round", 0),
               "iterations": len(state.get("history", [])), "startup_id": state.get("startup_id"), "volume_ids": state.get("volume_ids", [])}
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


def _resume_alignment(project_id, parent_id):
    with _job_lock(parent_id):
        conn = db.get_conn(project_id)
        try:
            state = _load_state(conn, parent_id)
            parent = _parent_row(conn, parent_id)
            if not state or parent is None or state.get("phase") != "align_symmetry" or parent["STATUS"] not in ("queued", "running"):
                return
            try:
                _align_symmetry(project_id, parent_id, state)
                _prepare_references_then_refine(conn, project_id, parent_id, state)
                _save(conn, parent_id, state, _progress_percent(state))   # the child it launched is only ours if the state says so
            except Exception as exc:  # noqa: BLE001
                _finish(conn, project_id, parent_id, state, "failed", "could not continue after the restart: {}".format(exc))
        finally:
            conn.close()


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
        if state.get("phase") == "align_symmetry":
            # Died while align_symmetry ran (its round is already recorded): run it again and carry on.
            _log(project_id, parent_row["JOB_ID"], "server restarted during the symmetry alignment (start {}, round {}); running it again".format(
                state["start"] + 1, state["round"]))
            threading.Thread(target=_resume_alignment, args=(project_id, parent_row["JOB_ID"]), daemon=True,
                             name="abinitio-" + parent_row["JOB_ID"]).start()
            return
        child_id = state.get("child_job_id")
        child = conn.execute("SELECT * FROM JOBS WHERE JOB_ID=?", (child_id,)).fetchone() if child_id else None
        if child is None:
            _finish(conn, project_id, parent_row["JOB_ID"], state, "failed", "Server restarted and the running step could not be found")
            return
        _log(project_id, parent_row["JOB_ID"], "server restarted during {} (start {}, round {}); waiting on child job {}".format(
            state["phase"], state["start"] + 1, state["round"] + 1, child_id))
        if child["STATUS"] in ("completed", "failed", "cancelled"):
            child_finished(project_id, child, child["STATUS"], child["ERROR"])
    finally:
        conn.close()


_ROUND_FILE = re.compile(r"^startup3d_(\d+)_(\d+)\.mrc$")


def rounds_available(state):
    """The reconstructions still in the scratch directory, one per finished
    round: [{n, start, round, label, files}] in order, `n` the output number
    (rounds x start + round, 0-based) startup3d_<n>_<class>.mrc was written
    under, `files` its per-class volumes. What lets the live view look back
    at how a run developed -- cisTEM keeps these files too, but its panel
    only ever shows the newest."""
    scratch = state.get("scratch")
    if not scratch or not os.path.isdir(scratch):
        return []
    by_n = {}
    for name in os.listdir(scratch):
        m = _ROUND_FILE.match(name)
        if m:
            by_n.setdefault(int(m.group(1)), {})[int(m.group(2))] = os.path.join(scratch, name)
    rounds = max(1, int(state.get("rounds") or 1))
    out = []
    for n in sorted(by_n):
        files = [by_n[n].get(k) for k in range(int(state.get("number_of_classes") or 1))]
        if not files or files[0] is None:
            continue
        out.append({"n": n, "start": n // rounds, "round": n % rounds,
                    "label": "Start {} \u00b7 Round {}".format(n // rounds + 1, n % rounds + 1), "files": files})
    return out


def live_result(conn, row):
    """The Jobs tab's Latest Result: the current reconstruction's orthogonal
    views and the sigma-per-iteration plot (AbInitio3DPanel's
    OrthResultsPanel + PlotPanel)."""
    state = _load_state(conn, row["JOB_ID"])
    if not state:
        return None
    display = [p for p in (state.get("display_files") or []) if p and os.path.isfile(p)]
    if not display and not state.get("volume_ids"):
        return None
    history = state.get("history", [])
    return {
        "kind": "abinitio",
        "task_index": len(history) * 10 + (1 if state.get("phase") == "finished" else 0),
        "label": history[-1]["label"] if history else "Random Start",
        "phase": state["phase"], "start": state["start"], "starts": state["starts"], "round": state["round"], "rounds": state["rounds"],
        "high_res": state.get("current_high_res"), "percent_used": state.get("current_percent_used"),
        "number_of_classes": state["number_of_classes"], "has_picture": bool(display), "current_volume": display[0] if display else None,
        "history": [h for h in history if h.get("average_sigma") is not None],
        "volume_ids": state.get("volume_ids", []), "startup_id": state.get("startup_id"),
        "package_name": state.get("package_name"),
        # every round still on disk, so the view can step back through them
        "available_rounds": [{"n": r["n"], "start": r["start"], "round": r["round"], "label": r["label"], "volume": r["files"][0]} for r in rounds_available(state)],
    }


def volume_file(conn, row, class_index=0, output_number=None):
    """The file behind current_picture(): the current reconstruction's, or an
    earlier round's with `output_number`. (path, label) or None."""
    state = _load_state(conn, row["JOB_ID"])
    if not state:
        return None
    if output_number is not None:
        match = [r for r in rounds_available(state) if r["n"] == output_number]
        if not match:
            return None
        files, label = match[0]["files"], "start{}_round{}".format(match[0]["start"] + 1, match[0]["round"] + 1)
    else:
        files, label = state.get("display_files") or [], "current"
    files = [p for p in files if p and os.path.isfile(p)]
    if not files:
        return None
    k = min(class_index, len(files) - 1)
    return files[k], label + ("_class{}".format(k + 1) if len(files) > 1 else "")


def current_picture(conn, row, class_index=0, output_number=None):
    """PNG of the current reconstruction's orthogonal views, for the live
    view -- or, with `output_number`, of the reconstruction an earlier round
    wrote (rounds_available())."""
    state = _load_state(conn, row["JOB_ID"])
    if not state:
        return None
    if output_number is not None:
        match = [r for r in rounds_available(state) if r["n"] == output_number]
        files = [p for p in (match[0]["files"] if match else []) if p and os.path.isfile(p)]
    else:
        files = [p for p in (state.get("display_files") or []) if p and os.path.isfile(p)]
    if not files:
        return None
    path = files[min(class_index, len(files) - 1)]
    png, meta = volumes.orthogonal_views_png_cached(path, state["settings"]["mask_radius"])
    return png, meta, path
