"""Refine CTF: cisTEM's CTFRefinementManager (src/gui/RefineCTFPanel.cpp)
as a driver in the parent/child shape of refine3d.py.

One pass, not a cycle: (mask the reference) -> refine_ctf x N -> [estimate
beam tilt x N] -> reconstruct3d x N -> merge3d x classes, from a chosen
refinement's parameters against the package's current references, writing
one refinement "Defocus Refinement #<id>" / "Beam Tilt Refinement #<id>" /
"Defocus & Beam Tilt Refinement #<id>" and one volume asset per class ("CTF
Refine #<id> - Class #<k>") that becomes the package's current reference.

refine_ctf refines each particle's defocus against its highest-occupancy
class (SetupRefinementJob() writes one star file for all classes with the
reference named per particle) and, when beam tilt is asked for, sums the
phase differences between the aligned images and their projections. Under
a controller it writes no output star: the refined defocus of each particle
arrives as an intermediate result, which the sink hands to
stages/refine_ctf.py's on_task_progress(); the phase-difference sums go to
the program's master, which writes their normalised, masked sum to the
phase-difference image path when the last worker reports. estimate_beamtilt
then searches that image over FindBeamTilt()'s 290 880 positions, split
across the profile's processes, and the lowest score wins.
ProcessAllJobsFinished()'s significance test decides whether that tilt is
real (Image::CalculateBeamTiltImage + ReturnBeamTiltSignificanceScore,
ported to numpy here): a beam tilt whose binarised phase pattern does not
match the measured one well enough is set to zero. The tilt (mrad) and the
particle shift (A) are written into every particle's row, as cisTEM does.

The output refinement keeps the input's angles, shifts and occupancies; its
statistics come from the new reconstruction and its angular distribution
is recomputed (identical to cisTEM's copy, since the angles did not move).

Two simplifications, both documented in CLAUDE.md: every class needs a
current reference (cisTEM would reconstruct one first), and the Signed CC
limit the panel shows is left out, since BeginRefinementCycle() forces it
to zero regardless of the box.
"""
import json
import math
import os
import re
import threading
from pathlib import Path

import numpy as np

import db
import progress_store
import aberrations
import refinements
import starfile
import volumes
import symmetry as symmetry_module
from abinitio import (merge_profile, _new_child, _parent_row, _profile, _task, _load_state, _save, now_iso, particle_range,
                      default_statistics, write_statistics, read_statistics, pooled_part_ssnr, average_sigma)
from refine3d import _num, _flag, _impose_helical, PLEASE_CREATE_PACKAGE_MESSAGE, CHILD_RECON, CHILD_MERGE
from stages import estimate_beamtilt as beamtilt_adapter, merge3d, reconstruct3d, refine_ctf as refine_ctf_adapter

STAGE = "refine_ctf"
CHILD_REFINE = "refinectf_refine_ctf"
CHILD_BEAMTILT = "refinectf_estimate_beamtilt"
MINIMUM_BEAM_TILT_SIGNIFICANCE_SCORE = 10.0  # defines.h

# RefineCTFPanel::SetDefaults(); the size-dependent ones follow the package.
DEFAULTS = {
    "refine_defocus": True, "refine_beam_tilt": True, "high_resolution_limit_a": 3.5,
    # the higher-order aberrations (RELION's three choices; the orders are RELION's defaults)
    "estimate_trefoil": False, "estimate_fourth_order": False, "odd_zernike_order": 3, "even_zernike_order": 4,
    # optics groups (web/CLAUDE.md, "Optics groups"): a group with fewer particles than these is not fitted on its own --
    # below the odd minimum it takes the whole data set's set, between the two its even set is the whole data set's
    "min_particles_per_group_odd": 500, "min_particles_per_group_even": 2000,
    "defocus_search_range_a": 500.0, "defocus_search_step_a": 20.0, "inner_mask_radius_a": 0.0,
    "score_to_weight_constant": 2.0, "adjust_score_for_defocus": True, "score_threshold": 0.0, "reconstruction_resolution_limit_a": 0.0,
    "autocrop_images": False, "apply_blurring": False, "smoothing_factor": 1.0,
    # cisTEM3's own: the real-space helical symmetrisation of the final maps (reconstruct3d / merge3d's last argument); only a helical package feels it.
    "impose_helical_symmetry": True,
    "use_mask": False, "auto_mask": True, "mask_edge_a": 10.0, "outside_mask_weight": 0.0, "low_pass_outside_mask": False, "mask_filter_resolution_a": 20.0,
}


def package_defaults(pkg):
    size = float(pkg["PARTICLE_SIZE"] or 150.0)
    return {"low_resolution_limit_a": round(min(size * 1.5, 300.0), 2), "mask_radius_a": round(size * 0.65, 2)}


def settings_from_params(params, pkg):
    s = {}
    for k, v in DEFAULTS.items():
        s[k] = _flag(params, k, v) if isinstance(v, bool) else _num(params, k, v)
    for k, v in package_defaults(pkg).items():
        s[k] = _num(params, k, v)
    if s["use_mask"]:
        s["auto_mask"] = False
    return s


def refinement_name(rid, refine_defocus, refine_beam_tilt, zernike=False):
    """RunRefinementJob()'s names, "Aberration" for the Zernike fit."""
    optics = "Aberration" if zernike else "Beam Tilt"
    if refine_defocus and (refine_beam_tilt or zernike):
        return "Defocus & {} Refinement #{}".format(optics, rid)
    if refine_defocus:
        return "Defocus Refinement #{}".format(rid)
    return "{} Refinement #{}".format(optics, rid)


# ---- the higher-order aberrations (web/CLAUDE.md, "Higher-order aberrations")

def uses_zernike(settings, input_sets):
    """Whether the odd aberrations are handled by the Zernike fit rather than cisTEM's beam-tilt grid search:
    asked for (trefoil or fourth-order estimation), or forced because the input refinement already carries an
    odd set -- the grid search on top of it would count the tilt twice."""
    return bool(settings["estimate_trefoil"] or settings["estimate_fourth_order"]
                or any(any(v != 0.0 for v in st.get("odd", [])) for st in input_sets or []))


def _default_set(sets):
    return _set_for_group(sets, 0)


def _set_for_group(sets, group):
    """The set a group's particles were refined with: the group's own line, else group 0's (AberrationTable::ForGroup's
    fallback), else nothing."""
    found = next((st for st in sets or [] if int(st.get("group", 0)) == int(group)), None)
    if found is None and int(group) != 0:
        found = next((st for st in sets or [] if int(st.get("group", 0)) == 0), None)
    return dict(found) if found else {"group": int(group), "odd": [], "even": [], "odd_order": 0, "even_order": 0}


def _sums_base(sums_path):
    return sums_path[:-4] if str(sums_path).lower().endswith(".mrc") else str(sums_path)


def group_sums_path(sums_path, group):
    """refine_ctf's per-group sums file beside the whole-data-set one: <base>_group_<g>.mrc."""
    return "{}_group_{}.mrc".format(_sums_base(sums_path), int(group))


def read_group_counts(sums_path):
    """<base>_groups.txt, "<group> <images>" per line, as {group: images}; {} when refine_ctf wrote none (no groups)."""
    path = _sums_base(sums_path) + "_groups.txt"
    counts = {}
    if os.path.isfile(path):
        for line in open(path):
            parts = line.split()
            if len(parts) >= 2:
                counts[int(parts[0])] = int(parts[1])
    return counts


def _weighted_rms(fit):
    """The residual of a fit as the RMS of the wrapped phase difference, weighted as the fit was (ring-normalised
    and by each ring's coherence), so rings beyond the data's resolution do not swamp it."""
    w = np.nan_to_num(fit["weight"])
    r = np.nan_to_num(aberrations._wrap(fit["measured_phase"] - fit["fitted_phase"]))
    return float(np.sqrt(np.sum(w * r * r) / np.sum(w))) if np.sum(w) > 0 else 0.0


def fit_refinement_aberrations(sums_path, box_size, pixel_size, cs_mm, voltage_kv, settings, input_sets, group=0, fit_even=True):
    """The fits from one of refine_ctf's sums files for a refinement, for one optics group (0 is the whole data
    set, the default set for every particle without a group line): the odd set is fitted whole (the sum
    measures the total odd phase, the projections carrying none) whenever beam tilt, trefoil or an existing odd
    set is in play, and trimmed to beam tilt and shift unless trefoil was asked for; the even residual is added
    to the even set the group's particles were refined with (its own line, else group 0's) when fourth-order
    estimation was asked for and `fit_even` allows it. Returns the new set, the beam tilt the odd set contains
    (mrad, A), the diagnostic half-plane maps and a summary. fit_all_groups() runs this per group."""
    s = settings
    current = _set_for_group(input_sets, group)
    odd_order, even_order = max(1, int(s["odd_zernike_order"])), max(0, int(s["even_zernike_order"]))
    rho, phi = aberrations.half_plane_frequencies(box_size, box_size, pixel_size)
    even_sums, S = aberrations.load_sums(sums_path)
    high, low = float(s["high_resolution_limit_a"]), float(s["low_resolution_limit_a"])
    new = {"group": int(group), "odd": [float(v) for v in current["odd"]], "even": [float(v) for v in current["even"]],
           "odd_order": int(current.get("odd_order") or 0), "even_order": int(current.get("even_order") or 0)}
    maps, summary = {}, {"high_resolution_limit_a": high, "low_resolution_limit_a": low}
    if s["refine_beam_tilt"] or s["estimate_trefoil"] or any(v != 0.0 for v in current["odd"]):
        fit = aberrations.fit_odd(S, rho, phi, odd_order, high, None, low)
        odd = np.array(fit["coefficients"], dtype=np.float64)
        if not s["estimate_trefoil"]:
            for i in range(odd.size):          # beam tilt and shift only: the order-1 and the m = +-1 order-3 terms
                m, n = aberrations.odd_index_to_mn(i)
                if n > 3 or abs(m) == 3:
                    odd[i] = 0.0
        new["odd"], new["odd_order"] = [float(v) for v in odd], odd_order
        maps["odd_measured"], maps["odd_fitted"] = fit["measured_phase"], fit["fitted_phase"]
        summary["odd_phase_rms"] = _weighted_rms(fit)
    if s["estimate_fourth_order"] and fit_even:
        fit = aberrations.fit_even(even_sums, rho, phi, even_order, high, low)
        residual = np.array(fit["coefficients"], dtype=np.float64)
        base = np.zeros(max(residual.size, len(current["even"])))
        base[:len(current["even"])] = current["even"]
        base[:residual.size] += residual
        new["even"], new["even_order"] = [float(v) for v in base], max(even_order, int(current.get("even_order") or 0))
        maps["even_measured"], maps["even_fitted"] = fit["measured_phase"], fit["fitted_phase"]
        summary["even_phase_rms"] = _weighted_rms(fit)
    tilt_x, tilt_y, shift_x, shift_y, rest = aberrations.beam_tilt_from_odd(new["odd"], cs_mm, voltage_kv) if new["odd"] else (0.0, 0.0, 0.0, 0.0, np.zeros(6))
    # the tilt is None with a Cs of 0 (aberrations.beam_tilt_from_odd): the coma terms stay in the set and `rest`
    summary.update({"beam_tilt_x_mrad": None if tilt_x is None else float(tilt_x), "beam_tilt_y_mrad": None if tilt_y is None else float(tilt_y),
                    "particle_shift_x_a": float(shift_x), "particle_shift_y_a": float(shift_y),
                    "trefoil": [float(rest[2]), float(rest[5])] if rest.size >= 6 else [0.0, 0.0],
                    "fourfold": [new["even"][4], new["even"][8]] if len(new["even"]) >= 9 else [0.0, 0.0],
                    "odd_order": new["odd_order"], "even_order": new["even_order"]})
    return {"set": new, "maps": maps, "summary": summary}


def fit_all_groups(sums_path, box_size, pixel_size, cs_mm, voltage_kv, settings, input_sets, group_counts=None, names=None):
    """The whole data set's fit (group 0) and one per optics group from refine_ctf's per-group sums files
    (group_sums_path; the counts from read_group_counts unless given). The minimum-size rule: a group with fewer
    particles than min_particles_per_group_odd gets no set of its own and so takes the whole data set's; one
    with fewer than min_particles_per_group_even gets its own odd set but the whole data set's even set.
    Returns the whole-data-set result's fields plus `sets` (group 0's set first, then each fitted group's, the
    aberration file's lines), `groups` (one summary per group: group, name, images, fitted, and the fit's
    numbers when fitted) and `group_maps` ({group: the half-plane maps})."""
    s = settings
    whole = fit_refinement_aberrations(sums_path, box_size, pixel_size, cs_mm, voltage_kv, s, input_sets, 0)
    counts = read_group_counts(sums_path) if group_counts is None else dict(group_counts)
    min_odd, min_even = int(s.get("min_particles_per_group_odd") or 0), int(s.get("min_particles_per_group_even") or 0)
    sets, groups, group_maps = [whole["set"]], [], {}
    for g in sorted(int(k) for k in counts):
        n = int(counts[g])
        if g == 0:
            continue
        path = group_sums_path(sums_path, g)
        entry = {"group": g, "name": (names or {}).get(g) or (names or {}).get(str(g)), "images": n, "fitted": False, "even_shared": False}
        if n < min_odd or not os.path.isfile(path):
            entry["reason"] = ("fewer than {} particles".format(min_odd) if n < min_odd else "no sums file") + ": uses the whole data set's set"
            groups.append(entry)
            continue
        r = fit_refinement_aberrations(path, box_size, pixel_size, cs_mm, voltage_kv, s, input_sets, g, fit_even=n >= min_even)
        if n < min_even and s["estimate_fourth_order"]:
            r["set"]["even"], r["set"]["even_order"] = list(whole["set"]["even"]), whole["set"]["even_order"]
            r["summary"]["fourfold"], r["summary"]["even_order"] = list(whole["summary"]["fourfold"]), whole["summary"]["even_order"]
            entry["even_shared"] = True
            entry["reason"] = "fewer than {} particles: the even set is the whole data set's".format(min_even)
        entry.update(r["summary"])
        entry["fitted"] = True
        sets.append(r["set"])
        groups.append(entry)
        group_maps[g] = r["maps"]
    return {"set": whole["set"], "maps": whole["maps"], "summary": whole["summary"], "sets": sets, "groups": groups, "group_maps": group_maps}


def defocus_histogram(changes, search_range, step):
    """RunRefinementJob()'s defocus_change_histogram: bins of `step` from
    -range to +range, each change counted at the nearest bin."""
    if step <= 0 or search_range <= 0:
        return [], []
    half = int(round(search_range / step))
    centres = [i * step for i in range(-half, half + 1)]
    counts = [0] * len(centres)
    for c in changes:
        i = int(round(c / step)) + half
        if 0 <= i < len(counts):
            counts[i] += 1
    return centres, counts


# ---------------------------------------------------------------------------
# Beam tilt significance (the numpy side of ProcessAllJobsFinished()'s
# ESTIMATE_BEAMTILT branch)
# ---------------------------------------------------------------------------

def wavelength_a(voltage_kv):
    """ReturnWavelenthInAngstroms()."""
    v = 1000.0 * voltage_kv
    return 12.2639 / math.sqrt(v + 0.97845e-6 * v * v)


def _wrap(phase):
    return (phase + math.pi) % (2.0 * math.pi) - math.pi


def phase_spectrum(image):
    """ComputeAmplitudeSpectrumFull2D(calculate_phases=true): the phase of
    every Fourier component of a real image on the centred full grid."""
    ft = np.fft.fftshift(np.fft.fft2(np.asarray(image, dtype=np.float64)))
    phase = np.angle(ft)
    phase[np.abs(ft) == 0.0] = 0.0
    return phase.astype(np.float32)


def cosine_ring_mask_2d(image, inner_radius, outer_radius, edge):
    """Image::CosineRingMask() for a centred real image: beyond the outer
    radius the edge-band average, inside the inner radius likewise, with
    cosine edges of width `edge` at both."""
    img = np.asarray(image, dtype=np.float32).copy()
    ny, nx = img.shape
    y, x = np.indices(img.shape)
    r2 = (x - nx // 2) ** 2 + (y - ny // 2) ** 2
    r = np.sqrt(r2)
    outer = max(outer_radius - edge / 2.0, 0.0)
    outer_plus = outer + edge
    inner = inner_radius + edge / 2.0
    inner_minus = inner - edge
    outer_band = (r >= outer) & (r <= outer_plus)
    inner_band = (r <= inner) & (r >= inner_minus)
    outer_avg = float(img[outer_band].mean()) if outer_band.any() else 0.0
    inner_avg = float(img[inner_band].mean()) if inner_band.any() else 0.0
    out = img.copy()
    e = (1.0 + np.cos(np.pi * (r - outer) / edge)) / 2.0
    out[outer_band] = img[outer_band] * e[outer_band] + (1.0 - e[outer_band]) * outer_avg
    out[r > outer_plus] = outer_avg
    if inner_radius > 0:
        ei = (1.0 + np.cos(np.pi * (inner - r) / edge)) / 2.0
        band = inner_band & ~outer_band & (r <= outer)
        out[band] = img[band] * ei[band] + (1.0 - ei[band]) * inner_avg
        out[r < inner_minus] = inner_avg
    return out


def beam_tilt_phase_image(shape, pixel_size, voltage_kv, cs_mm, beam_tilt_x, beam_tilt_y, shift_x_a, shift_y_a):
    """CalculateBeamTiltImage() followed by ComputeAmplitudeSpectrumFull2D(phases):
    the wrapped phase shift beam tilt (radians) and particle shift (A) give
    each Fourier component, on the centred full grid."""
    ny, nx = shape
    fy = (np.arange(ny) - ny // 2) / float(ny)
    fx = (np.arange(nx) - nx // 2) / float(nx)
    FX, FY = np.meshgrid(fx, fy)
    f2 = FX ** 2 + FY ** 2
    f = np.sqrt(f2)
    azimuth = np.arctan2(FY, FX)
    azimuth[(FX == 0) & (FY == 0)] = 0.0
    wl = wavelength_a(voltage_kv) / pixel_size                # in pixels
    cs = cs_mm * 1e7 / pixel_size                           # in pixels
    tilt = math.hypot(beam_tilt_x, beam_tilt_y)
    tilt_az = math.atan2(beam_tilt_y, beam_tilt_x) if tilt else 0.0
    shift = math.hypot(shift_x_a, shift_y_a) / pixel_size   # in pixels
    shift_az = math.atan2(shift_y_a, shift_x_a) if shift else 0.0
    phase = 2.0 * math.pi * cs * wl * wl * f2 * f * (tilt * np.cos(azimuth - tilt_az)) - 2.0 * math.pi * f * (shift * np.cos(azimuth - shift_az))
    return _wrap(phase).astype(np.float32)


def beam_tilt_significance(phase_difference_image, pixel_size, voltage_kv, cs_mm, beam_tilt_x, beam_tilt_y, shift_x_a, shift_y_a):
    """ReturnBeamTiltSignificanceScore(): how well the binarised measured
    phase pattern matches the binarised pattern the found tilt predicts.
    Returns (score, measured phase spectrum, predicted phase image)."""
    spectrum = phase_spectrum(phase_difference_image)
    ny, nx = spectrum.shape
    spectrum = cosine_ring_mask_2d(spectrum, 5.0, float(max(nx, ny)), 2.0)
    predicted = beam_tilt_phase_image(spectrum.shape, pixel_size, voltage_kv, cs_mm, beam_tilt_x, beam_tilt_y, shift_x_a, shift_y_a)
    buffer = (spectrum * float(nx) >= 0.00002).astype(np.float32)
    predicted_bin = (predicted >= 0.0).astype(np.float32)
    mask_radius = math.sqrt(float(buffer.mean()) * nx * ny / math.pi)
    y, x = np.indices(spectrum.shape)
    inside = (x - nx // 2) ** 2 + (y - ny // 2) ** 2 <= mask_radius ** 2
    diff = buffer - predicted_bin
    score = float((diff[inside] ** 2).mean()) if inside.any() else 0.0
    significance = 0.5 * math.pi * ((0.5 - score) * mask_radius) ** 2
    return significance, spectrum, predicted


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
    rid = params.get("input_refinement_id")
    if rid in (None, ""):
        rid = pkg["LAST_REFINEMENT_ID"]
    ref = refinements.refinement_row(conn, rid) if rid is not None else None
    if ref is None:
        raise ValueError("the input parameters (refinement {}) do not exist".format(rid))
    if ref["REFINEMENT_PACKAGE_ASSET_ID"] != pkg["REFINEMENT_PACKAGE_ASSET_ID"]:
        raise ValueError("refinement {} belongs to a different package".format(rid))
    if not _flag(params, "refine_defocus", True) and not _flag(params, "refine_beam_tilt", True):
        raise ValueError("Tick Refine Defocus Params., Refine Beam Tilt Parms., or both.")
    classes = max(1, int(pkg["NUMBER_OF_CLASSES"] or 1))
    refs = refinements.current_references(conn, pkg["REFINEMENT_PACKAGE_ASSET_ID"])
    reference_ids = []
    for k in range(1, classes + 1):
        vid = refs.get(k)
        if vid is None or vid < 0:
            raise ValueError("Class {} has no reference volume yet. Run Ab-Initio 3D, Auto Refine or Refine 3D first, or pick one on the Refine 3D panel.".format(k))
        vol = conn.execute("SELECT * FROM VOLUME_ASSETS WHERE VOLUME_ASSET_ID=?", (vid,)).fetchone()
        if vol is None or not vol["FILENAME"] or not os.path.isfile(vol["FILENAME"]):
            raise ValueError("the reference volume of class {} (asset {}) is missing".format(k, vid))
        if vol["X_SIZE"] != pkg["STACK_BOX_SIZE"] or abs(float(vol["PIXEL_SIZE"] or 0) - float(pkg["OUTPUT_PIXEL_SIZE"] or 0)) > 0.01:
            raise ValueError("Error: Reference volume has different dimensions / pixel size from the input stack.  This will currently not work.")
        reference_ids.append(vid)
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
    d = db.project_dir(project_id) / "Scratch" / "RefineCTF" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def phase_difference_dir(project_id):
    d = db.project_dir(project_id) / "Assets" / "PhaseDifferences"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _class_star(state, tag, k):
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
    reference_files = [conn.execute("SELECT FILENAME FROM VOLUME_ASSETS WHERE VOLUME_ASSET_ID=?", (vid,)).fetchone()["FILENAME"] for vid in reference_ids]
    class_rows = [refinements.load_table(conn, ref["REFINEMENT_ID"], k) for k in range(1, classes + 1)]
    first = class_rows[0][0] if class_rows and len(class_rows[0]) else None
    state = {
        "phase": None, "round": 0, "rounds": 1, "settings": s,
        "package_id": pkg["REFINEMENT_PACKAGE_ASSET_ID"], "package_name": pkg["NAME"], "stack_filename": pkg["STACK_FILENAME"],
        "pixel_size": float(pkg["OUTPUT_PIXEL_SIZE"] or 1.0), "box_size": int(pkg["STACK_BOX_SIZE"]), "invert_contrast": bool(pkg["STACK_HAS_WHITE_PROTEIN"]),
        "symmetry": pkg["SYMMETRY"] or "C1", "molecular_weight": float(pkg["MOLECULAR_WEIGHT"] or 300.0), "particle_size": float(pkg["PARTICLE_SIZE"] or 150.0),
        "voltage": float(first["voltage"]) if first is not None and first["voltage"] else 300.0,
        "cs": 2.7 if first is None or first["cs"] is None else float(first["cs"]),   # 0 is a real value (a Cs-corrected microscope): the tilt is then "not applicable"
        "number_of_particles": n, "number_of_classes": classes,
        "refinement_profile": profile["name"], "refinement_jobs": int(profile["total_jobs"]),
        "reconstruction_profile": recon_profile["name"], "reconstruction_jobs": int(recon_profile["total_jobs"]),
        "input_refinement_id": ref["REFINEMENT_ID"], "input_refinement_name": ref["NAME"], "output_refinement_id": None,
        "reference_files": reference_files, "reference_asset_ids": list(reference_ids), "original_reference_files": list(reference_files),
        "mask_file": mask["FILENAME"] if mask is not None else None, "mask_asset_id": mask["VOLUME_ASSET_ID"] if mask is not None else -1,
        "child_job_id": None, "child_task_count": 0, "child_done": 0, "history": [], "started_at": now_iso(), "scratch": str(scratch),
        "volume_ids": [], "beam_tilt": None, "defocus_histogram": None, "progress_files": [],
        "input_aberrations": refinements.load_aberrations(conn, ref["REFINEMENT_ID"]), "aberrations": None,
    }
    # the optics groups of the particles (their beam_tilt_group column): with any, the beam tilt goes through the
    # Zernike fit, which gives each group its own, where the grid search knows one tilt for the whole data set
    groups_of = class_rows[0]["beam_tilt_group"].astype(int) if class_rows and len(class_rows[0]) and "beam_tilt_group" in class_rows[0].dtype.names else np.zeros(0, dtype=int)
    values, counts = np.unique(groups_of[groups_of != 0], return_counts=True)
    names = {int(r["GROUP_ID"]): r["NAME"] for r in conn.execute("SELECT GROUP_ID, NAME FROM OPTICS_GROUPS")}
    state["optics_groups"] = [{"group": int(g), "images": int(c), "name": names.get(int(g))} for g, c in zip(values.tolist(), counts.tolist())]
    state["zernike"] = uses_zernike(s, state["input_aberrations"]) or bool(s["refine_beam_tilt"] and state["optics_groups"])
    _store_rows(state, "input", class_rows)
    with conn:
        conn.execute("UPDATE JOBS SET STATUS='running', STARTED_AT=?, PROGRESS=0 WHERE JOB_ID=?", (now_iso(), job_id))
    what = " and ".join(w for w, on in (("defocus", s["refine_defocus"]), ("beam tilt", s["refine_beam_tilt"] and not state["zernike"]),
                                        ("odd aberrations (Zernike order {})".format(int(s["odd_zernike_order"])), state["zernike"]),
                                        ("even aberrations (order {})".format(int(s["even_zernike_order"])), s["estimate_fourth_order"])) if on)
    _log(project_id, job_id, "Refine CTF ({}) of {!r} from {!r}: {} particles, {} class{}, {:.2f} Å limit, ±{:.0f} Å in {:.0f} Å steps; refinement profile {!r}, reconstruction profile {!r}".format(
        what, pkg["NAME"], ref["NAME"], n, classes, "" if classes == 1 else "es", s["high_resolution_limit_a"], s["defocus_search_range_a"], s["defocus_search_step_a"],
        profile["name"], recon_profile["name"]))
    if state["optics_groups"]:
        _log(project_id, job_id, "{} optics group{} ({} particles without one): each gets its own aberration fit; a group under {} particles takes the whole data set's, under {} its even set".format(
            len(state["optics_groups"]), "" if len(state["optics_groups"]) == 1 else "s", int((groups_of == 0).sum()),
            int(s["min_particles_per_group_odd"]), int(s["min_particles_per_group_even"])))
    _mask_then_refine(conn, project_id, job_id, state)
    _save(conn, job_id, state)
    return state


def _mask_then_refine(conn, project_id, job_id, state):
    """DoMasking(): the supplied mask or the auto-mask, then refine."""
    s = state["settings"]
    scratch = Path(state["scratch"])
    if s["use_mask"] and state.get("mask_file"):
        _log(project_id, job_id, "Masking reference reconstruction with selected mask")
        mask, _ps = volumes.read_mrc_volume(state["mask_file"])
        masked = []
        for k, ref in enumerate(state["reference_files"]):
            vol, _ps = volumes.read_mrc_volume(ref)
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
            volumes.write_mrc_volume(out, volumes.auto_mask(vol, state["pixel_size"], s["mask_radius_a"]), state["pixel_size"])
            masked.append(out)
        state["reference_files"] = masked
    _launch_refinement(conn, project_id, job_id, state)


def merged_input_rows(class_rows, reference_files):
    """SetupRefinementJob(): one row per particle from its highest-occupancy
    class, naming that class's reference. Returns (table, reference filename
    per row) -- the filename is a string column the binary writer takes
    separately (starfile.write_params(..., string_columns=...))."""
    tables = [refinements.as_table(c) for c in class_rows]
    if len(tables) == 1:
        return tables[0].copy(), [reference_files[0]] * len(tables[0])
    n = min(len(t) for t in tables)
    occ = np.stack([t["occupancy"][:n].astype(float) for t in tables], 0)
    best = np.argmax(occ, axis=0)
    merged = tables[0][:n].copy()
    for k in range(1, len(tables)):
        pick = best == k
        merged[pick] = tables[k][:n][pick]
    return merged, [reference_files[k] for k in best.tolist()]


def _launch_refinement(conn, project_id, job_id, state):
    """SetupRefinementJob() + RunRefinementJob()."""
    s = state["settings"]
    n = state["number_of_particles"]
    input_id = state["input_refinement_id"]
    rid = refinements.next_refinement_id(conn)
    state["output_refinement_id"] = rid
    scratch = Path(state["scratch"])
    class_rows = _load_rows(state, "input")
    merged, references = merged_input_rows(class_rows, state["reference_files"])
    star = str(scratch / "refine_ctf_input_star_{}.cistem".format(input_id))
    starfile.write_params(star, merged, string_columns={"reference_3d_filename": references})
    stats = [st for st in refinements.load_statistics(conn, input_id, 1) if 1 <= st["shell"] <= state["box_size"] // 2] \
        or default_statistics(state["molecular_weight"], state["pixel_size"], state["box_size"])
    stats_file = str(scratch / "input_stats_{}_1.txt".format(input_id))
    write_statistics(stats_file, stats, state["pixel_size"])
    phase_image = str(scratch / "phase_output.mrc")
    aberration_sums = str(scratch / "aberration_sums.mrc")
    aberration_file = refinements.write_aberration_file(conn, input_id, scratch / "aberrations_input.txt")
    jobs = max(1, min(n, state["refinement_jobs"]))
    tasks, progress_files = [], []
    for j in range(1, jobs + 1):
        first, last = particle_range(j, jobs, n)
        progress = str(scratch / "refine_ctf_progress_{}_{}.txt".format(rid, j))
        progress_files.append(progress)
        values = [state["stack_filename"], star, state["reference_files"][0], stats_file, True,
                  progress, "/dev/null", phase_image, "/dev/null", "/dev/null",
                  first, last, state["pixel_size"], state["molecular_weight"], s["inner_mask_radius_a"], s["mask_radius_a"],
                  s["low_resolution_limit_a"], s["high_resolution_limit_a"], s["defocus_search_range_a"], s["defocus_search_step_a"], 1.0,
                  s["refine_defocus"], bool(s["refine_beam_tilt"] or state.get("zernike")), True, state["invert_contrast"], False, not s["apply_blurring"], True,
                  j - 1, jobs, 1, aberration_file, aberration_sums]
        tasks.append(_task(refine_ctf_adapter, j - 1, j, values))
    parent = _parent_row(conn, job_id)
    child = _new_child(conn, job_id, CHILD_REFINE, "{} · refine_ctf".format(parent["NAME"]), parent)
    state.update({"phase": "refine", "child_job_id": child, "child_task_count": len(tasks), "child_done": 0,
                  "progress_files": progress_files, "phase_image": phase_image, "aberration_sums": aberration_sums, "refinement_jobs_this_round": jobs})
    _log(project_id, job_id, "{} ({} task{}) -> Refinement #{} — child job {}".format(
        "Refining defocus and calculating phase difference image" if s["refine_defocus"] and s["refine_beam_tilt"]
        else "Refining defocus" if s["refine_defocus"] else "Calculating phase difference image", len(tasks), "" if len(tasks) == 1 else "s", rid, child))
    _runtime.submit_child(project_id, child, refine_ctf_adapter, tasks, _profile(state["refinement_profile"]))


def _collect_defocus(state):
    """The refined defocus of every particle from the progress files -> the output rows."""
    s = state["settings"]
    results = {}
    for p in state.get("progress_files") or []:
        results.update(refine_ctf_adapter.read_progress(p))
    class_rows = _load_rows(state, "input")
    changes = []
    if s["refine_defocus"]:
        if not results:
            raise ValueError("refine_ctf reported no refined defocus values (were its intermediate results forwarded?)")
        got_positions = np.array(sorted(results), dtype=np.int64)
        got_df1 = np.array([results[p][0] for p in got_positions.tolist()], dtype=float)
        got_df2 = np.array([results[p][1] for p in got_positions.tolist()], dtype=float)
        for k, rows in enumerate(class_rows):
            positions = rows["position_in_stack"].astype(np.int64)
            idx = np.searchsorted(got_positions, positions)
            idx = np.clip(idx, 0, max(len(got_positions) - 1, 0))
            found = (len(got_positions) > 0) & (got_positions[idx] == positions) if len(got_positions) else np.zeros(len(positions), dtype=bool)
            if k == 0:
                changes.extend((got_df1[idx[found]] - rows["defocus_1"][found].astype(float)).tolist())
            rows["defocus_1"][found] = got_df1[idx[found]]
            rows["defocus_2"][found] = got_df2[idx[found]]
            if len(class_rows) == 1:
                got_logp = np.array([results[p][2] for p in got_positions.tolist()], dtype=float)
                got_score = np.array([results[p][3] for p in got_positions.tolist()], dtype=float)
                rows["logp"][found] = got_logp[idx[found]]
                rows["score"][found] = got_score[idx[found]]
    centres, counts = defocus_histogram(changes, s["defocus_search_range_a"], s["defocus_search_step_a"])
    state["defocus_histogram"] = {"centres": centres, "counts": counts, "n": len(changes)}
    _store_rows(state, "output", class_rows)
    for p in state.get("progress_files") or []:
        try:
            os.remove(p)
        except OSError:
            pass
    return len(results)


def _launch_beam_tilt(conn, project_id, job_id, state):
    """RunBeamTiltEstimationJob(): the search over the master's phase-difference image."""
    phase_image = state["phase_image"]
    if not os.path.isfile(phase_image):
        raise ValueError("refine_ctf's master did not write the phase-difference image {}".format(phase_image))
    jobs = max(1, state["refinement_jobs"])
    total = beamtilt_adapter.TOTAL_POSITIONS
    per = int(math.ceil(total / float(jobs)))
    tasks = []
    for j in range(jobs):
        first = j * per
        last = min((j + 1) * per - 1, total - 1)
        if first > last:
            break
        tasks.append(_task(beamtilt_adapter, j, j + 1, [phase_image, state["pixel_size"], state["voltage"], state["cs"], first, last]))
    parent = _parent_row(conn, job_id)
    child = _new_child(conn, job_id, CHILD_BEAMTILT, "{} · estimate_beamtilt".format(parent["NAME"]), parent)
    state.update({"phase": "beamtilt", "child_job_id": child, "child_task_count": len(tasks), "child_done": 0})
    _log(project_id, job_id, "Estimating beam tilt ({} task{} over {} positions) — child job {}".format(len(tasks), "" if len(tasks) == 1 else "s", total, child))
    _runtime.submit_child(project_id, child, beamtilt_adapter, tasks, _profile(state["refinement_profile"]))


def _apply_beam_tilt(conn, project_id, job_id, state, child_id):
    """The ESTIMATE_BEAMTILT branch of ProcessAllJobsFinished(): the best of
    the tasks' answers, the significance test, and the tilt written into
    every row."""
    best = None
    for row in conn.execute("SELECT RESULT_JSON FROM JOB_TASKS WHERE JOB_ID=? AND STATUS='ok'", (child_id,)).fetchall():
        try:
            data = json.loads(row["RESULT_JSON"] or "{}").get("data") or []
        except (TypeError, ValueError):
            continue
        if len(data) >= 5 and (best is None or data[0] < best[0]):
            best = data[:5]
    if best is None:
        raise ValueError("estimate_beamtilt returned no result")
    score, btx, bty, shx, shy = best
    image, _ps = volumes.read_mrc_volume(state["phase_image"])
    image = image[0]
    significance, spectrum, predicted = beam_tilt_significance(image, state["pixel_size"], state["voltage"], state["cs"], btx, bty, shx, shy)
    out_dir = phase_difference_dir(project_id)
    rid = state["output_refinement_id"]
    spectrum_file = str(out_dir / "refinement_{}_phase_difference.mrc".format(rid))
    tilt_file = str(out_dir / "refinement_{}_beam_tilt.mrc".format(rid))
    volumes.write_mrc_volume(spectrum_file, spectrum[None, :, :], state["pixel_size"])
    volumes.write_mrc_volume(tilt_file, predicted[None, :, :], state["pixel_size"])
    significant = significance > MINIMUM_BEAM_TILT_SIGNIFICANCE_SCORE
    if not significant:
        btx = bty = shx = shy = 0.0
    state["beam_tilt"] = {"beam_tilt_x_mrad": btx * 1000.0, "beam_tilt_y_mrad": bty * 1000.0, "particle_shift_x_a": shx, "particle_shift_y_a": shy,
                          "score": score, "significance": significance, "significant": significant,
                          "phase_difference_file": spectrum_file, "beam_tilt_file": tilt_file}
    _log(project_id, job_id, "Beam tilt {:.2f}, {:.2f} mrad; particle shift {:.2f}, {:.2f} Å; significance {:.2f}{}".format(
        best[1] * 1000.0, best[2] * 1000.0, best[3], best[4], significance, "" if significant else " — below {:.0f}, set to zero".format(MINIMUM_BEAM_TILT_SIGNIFICANCE_SCORE)))
    class_rows = _load_rows(state, "output")
    for rows in class_rows:
        rows["beam_tilt_x"], rows["beam_tilt_y"] = btx * 1000.0, bty * 1000.0
        rows["image_shift_x"], rows["image_shift_y"] = shx, shy
    _store_rows(state, "output", class_rows)


def _fit_aberrations(conn, project_id, job_id, state):
    """The Zernike fit in place of the beam-tilt grid search: the new set, the beam tilt it contains written to
    the live result (the particle rows' tilt and shift columns go to zero, since the set now carries them and
    the programs apply both), and the diagnostic phase maps under Assets/PhaseDifferences."""
    path = state["aberration_sums"]
    if not os.path.isfile(path):
        raise ValueError("refine_ctf did not write the aberration sums {}".format(path))
    names = {g["group"]: g["name"] for g in state.get("optics_groups") or []}
    result = fit_all_groups(path, state["box_size"], state["pixel_size"], state["cs"], state["voltage"], state["settings"], state["input_aberrations"], names=names)
    rid = state["output_refinement_id"]
    out_dir = phase_difference_dir(project_id)
    files = {}

    def write_maps(maps, prefix):
        out = {}
        for name, half in maps.items():
            full = aberrations.full_grid(np.nan_to_num(aberrations._wrap(half)), odd=name.startswith("odd"))     # wrapped: the fitted surface is shown modulo 2 pi
            out[name] = str(out_dir / "{}_{}.mrc".format(prefix, name))
            volumes.write_mrc_volume(out[name], full[None, :, :].astype(np.float32), state["pixel_size"])
        return out

    files = write_maps(result["maps"], "refinement_{}".format(rid))
    group_files = {str(g): write_maps(maps, "refinement_{}_group_{}".format(rid, g)) for g, maps in result["group_maps"].items()}
    summary = result["summary"]
    state["aberrations"] = {"sets": result["sets"], "files": files, "summary": summary, "groups": result["groups"], "group_files": group_files}
    for g in result["groups"]:
        if g["fitted"]:
            _log(project_id, job_id, "Group {}{}: {} particles, {}, shift {:.2f}, {:.2f} Å{}{}{}".format(
                g["group"], " ({})".format(g["name"]) if g.get("name") else "", g["images"],
                "beam tilt not applicable with a Cs of 0" if g["beam_tilt_x_mrad"] is None else "beam tilt {:.3f}, {:.3f} mrad".format(g["beam_tilt_x_mrad"], g["beam_tilt_y_mrad"]),
                g["particle_shift_x_a"], g["particle_shift_y_a"],
                "; trefoil {:.1f}, {:.1f}".format(*g["trefoil"]) if state["settings"]["estimate_trefoil"] else "",
                "; four-fold {:.1f}, {:.1f}".format(*g["fourfold"]) if state["settings"]["estimate_fourth_order"] and not g["even_shared"] else "",
                "; " + g["reason"] if g.get("reason") else ""))
        else:
            _log(project_id, job_id, "Group {}{}: {} particles, {}".format(g["group"], " ({})".format(g["name"]) if g.get("name") else "", g["images"], g["reason"]))
    state["beam_tilt"] = {"beam_tilt_x_mrad": summary["beam_tilt_x_mrad"], "beam_tilt_y_mrad": summary["beam_tilt_y_mrad"],
                          "particle_shift_x_a": summary["particle_shift_x_a"], "particle_shift_y_a": summary["particle_shift_y_a"],
                          "score": None, "significance": None, "significant": True, "method": "zernike",
                          "phase_difference_file": files.get("odd_measured"), "beam_tilt_file": files.get("odd_fitted")}
    _log(project_id, job_id, "Aberrations fitted from {:.0f} to {:.2f} Å: {}, shift {:.2f}, {:.2f} Å{}{}".format(
        summary["low_resolution_limit_a"], summary["high_resolution_limit_a"],
        "beam tilt not applicable with a Cs of 0 (its coma terms stay in the set)" if summary["beam_tilt_x_mrad"] is None
        else "beam tilt {:.3f}, {:.3f} mrad".format(summary["beam_tilt_x_mrad"], summary["beam_tilt_y_mrad"]),
        summary["particle_shift_x_a"], summary["particle_shift_y_a"],
        "; trefoil {:.1f}, {:.1f}".format(*summary["trefoil"]) if state["settings"]["estimate_trefoil"] else "",
        "; four-fold astigmatism {:.1f}, {:.1f}; even residual {:.3f} rad RMS".format(summary["fourfold"][0], summary["fourfold"][1], summary.get("even_phase_rms", 0.0))
        if state["settings"]["estimate_fourth_order"] else ""))
    class_rows = _load_rows(state, "output")
    for rows in class_rows:
        rows["beam_tilt_x"], rows["beam_tilt_y"] = 0.0, 0.0
        rows["image_shift_x"], rows["image_shift_y"] = 0.0, 0.0
    _store_rows(state, "output", class_rows)


def _output_aberration_file(conn, state):
    """The aberration file for the steps after the fit: the fitted set, else the input refinement's."""
    scratch = Path(state["scratch"])
    if state.get("aberrations"):
        return refinements.write_aberration_sets(state["aberrations"]["sets"], scratch / "aberrations_output.txt", "refinement {}".format(state["output_refinement_id"]))
    return refinements.write_aberration_file(conn, state["input_refinement_id"], scratch / "aberrations_input.txt")


def _launch_reconstruction(conn, project_id, job_id, state):
    """SetupReconstructionJob() + RunReconstructionJob()."""
    s = state["settings"]
    n, classes = state["number_of_particles"], state["number_of_classes"]
    rid = state["output_refinement_id"]
    scratch = Path(state["scratch"])
    class_rows = _load_rows(state, "output")
    written = []
    for k, rows in enumerate(class_rows):
        p = str(scratch / "beam_tilt_output_par_{}_{}.cistem".format(rid, k + 1))
        starfile.write_params(p, rows)
        written.append(p)
    jobs = max(1, min(n, state["reconstruction_jobs"]))
    aberration_file = _output_aberration_file(conn, state)
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
                      True, s["adjust_score_for_defocus"], state["invert_contrast"], False, s["autocrop_images"], False, False,
                      use_ref, True, True,
                      str(scratch / "dump_file_{}_{}_odd_{}.dump".format(rid, k, j)), str(scratch / "dump_file_{}_{}_even_{}.dump".format(rid, k, j)), 0, 1, _impose_helical(state), aberration_file]
            tasks.append(_task(reconstruct3d, index, k * 1000000 + j, values))
            index += 1
    parent = _parent_row(conn, job_id)
    child = _new_child(conn, job_id, CHILD_RECON, "{} · reconstruct3d".format(parent["NAME"]), parent)
    state.update({"phase": "recon", "child_job_id": child, "child_task_count": len(tasks), "child_done": 0, "number_of_dump_files": jobs})
    _log(project_id, job_id, "Calculating {} ({} task{}) — child job {}".format("reconstructions" if classes > 1 else "reconstruction", len(tasks), "" if len(tasks) == 1 else "s", child))
    _runtime.submit_child(project_id, child, reconstruct3d, tasks, _profile(state["reconstruction_profile"]))


def _launch_merge(conn, project_id, job_id, state):
    """SetupMerge3dJob() + RunMerge3dJob()."""
    s = state["settings"]
    classes = state["number_of_classes"]
    rid = state["output_refinement_id"]
    scratch = Path(state["scratch"])
    vol_dir = volumes.volume_dir(project_id)
    tasks, outputs, stats = [], [], []
    for k in range(classes):
        out = str(vol_dir / "volume_{}_{}.mrc".format(rid, k + 1))
        st = str(scratch / "volume_stats_{}_{}.txt".format(rid, k + 1))
        outputs.append(out)
        stats.append(st)
        tasks.append(_task(merge3d, k, k + 1, ["/dev/null", "/dev/null", out, st, state["molecular_weight"], s["inner_mask_radius_a"], s["mask_radius_a"],
                                                str(scratch / "dump_file_{}_{}_odd_.dump".format(rid, k)), str(scratch / "dump_file_{}_{}_even_.dump".format(rid, k)),
                                                k + 1, False, "", _required_count(state, "number_of_dump_files"), 1.0, 5.0, _impose_helical(state)]))
    parent = _parent_row(conn, job_id)
    child = _new_child(conn, job_id, CHILD_MERGE, "{} · merge3d".format(parent["NAME"]), parent)
    state.update({"phase": "merge", "child_job_id": child, "child_task_count": len(tasks), "child_done": 0,
                  "pending_volume_files": outputs, "pending_stats_files": stats})
    _log(project_id, job_id, "Merging and filtering {} — child job {}".format("reconstructions" if classes > 1 else "reconstruction", child))
    _runtime.submit_child(project_id, child, merge3d, tasks, merge_profile(_profile(state["reconstruction_profile"]), len(tasks)))


_PHASE_SPAN = {"refine": (0.0, 0.5), "beamtilt": (0.5, 0.65), "recon": (0.65, 0.9), "merge": (0.9, 1.0), "finished": (1.0, 1.0)}


def _progress_percent(state):
    frac = float(state.get("child_done", 0)) / float(max(state.get("child_task_count", 1), 1))
    lo, hi = _PHASE_SPAN.get(state.get("phase"), (0.0, 0.0))
    return max(0, min(100, int(100.0 * (lo + (hi - lo) * frac))))


def progress_info(state):
    if not state:
        return {}
    finished = [h["finished_at"] for h in state.get("history", []) if h.get("finished_at")]
    return {"task_count": 1, "tasks_done": len(finished), "first_task_finished_at": finished[0] if finished else None,
            "last_task_finished_at": finished[-1] if finished else None, "round": 0, "rounds": 1, "phase": state["phase"]}



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
                     daemon=True, name="refinectf-" + child_row["PARENT_JOB_ID"]).start()


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
                _advance(conn, project_id, parent_id, state, child_id)
            except Exception as exc:  # noqa: BLE001
                _finish(conn, project_id, parent_id, state, "failed", "could not continue: {}".format(exc))
        finally:
            conn.close()


def _advance(conn, project_id, parent_id, state, child_id):
    """ProcessAllJobsFinished()."""
    phase = state["phase"]
    s = state["settings"]
    if phase == "refine":
        got = _collect_defocus(state)
        if s["refine_defocus"]:
            h = state["defocus_histogram"]
            _log(project_id, parent_id, "Refined defocus of {} particles; {} changed by more than one step".format(
                got, sum(c for centre, c in zip(h["centres"], h["counts"]) if abs(centre) > s["defocus_search_step_a"] * 0.5)))
        if state.get("zernike"):
            progress_store.note(parent_id, 0, 0, "fitting the aberrations")
            _fit_aberrations(conn, project_id, parent_id, state)
            _launch_reconstruction(conn, project_id, parent_id, state)
        elif s["refine_beam_tilt"]:
            _launch_beam_tilt(conn, project_id, parent_id, state)
        else:
            _launch_reconstruction(conn, project_id, parent_id, state)
    elif phase == "beamtilt":
        progress_store.note(parent_id, 0, 0, "applying the beam tilt")
        _apply_beam_tilt(conn, project_id, parent_id, state, child_id)
        _launch_reconstruction(conn, project_id, parent_id, state)
    elif phase == "recon":
        _launch_merge(conn, project_id, parent_id, state)
    elif phase == "merge":
        progress_store.note(parent_id, 0, 0, "writing the refinement")
        _record(conn, project_id, parent_id, state)
        for p in Path(state["scratch"]).glob("dump_file_*.dump"):
            try:
                p.unlink()
            except OSError:
                pass
        state["phase"] = "finished"
        state["child_job_id"] = None
        _log(project_id, parent_id, "Refinement finished!")
        _finish(conn, project_id, parent_id, state, "completed", None)
        return
    else:
        raise ValueError("unexpected phase {!r}".format(phase))
    _save(conn, parent_id, state, _progress_percent(state))


def _record(conn, project_id, parent_id, state):
    """The MERGE branch of ProcessAllJobsFinished()."""
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
        recon_id = refinements.next_reconstruction_id(conn)
        vid = volumes.add_volume_asset(conn, "CTF Refine #{} - Class #{}".format(rid, k + 1), state["pending_volume_files"][k], state["pixel_size"],
                                       state["box_size"], state["box_size"], state["box_size"], reconstruction_job_id=recon_id)
        refinements.add_reconstruction_job(conn, recon_id, state["package_id"], rid, "", s["inner_mask_radius_a"], s["mask_radius_a"],
                                           s["reconstruction_resolution_limit_a"], s["score_to_weight_constant"], s["adjust_score_for_defocus"],
                                           s["autocrop_images"], False, s["apply_blurring"], s["smoothing_factor"], k + 1, vid)
        refinements.set_current_reference(conn, state["package_id"], k + 1, vid)
        volume_ids.append(vid)
        recon_ids.append(recon_id)
        est_res.append(refinements.estimated_resolution(stats[k], state["pixel_size"]) if stats[k] else None)
    state["reference_files"] = list(state["pending_volume_files"])
    volumes.prepare_orth_views(state["reference_files"], state["settings"]["mask_radius_a"], lambda t: _log(project_id, parent_id, t, "warning"))
    state["reference_asset_ids"] = volume_ids
    state["volume_ids"] = volume_ids
    for k in range(classes):
        if classes > 1:
            _log(project_id, parent_id, "Est. Res. Class {:2d} = {} Å ({:.2f} %)".format(k + 1, "{:.2f}".format(est_res[k]) if est_res[k] else "n/a", avgs[k]))
        else:
            _log(project_id, parent_id, "Est. Res. = {} Å".format("{:.2f}".format(est_res[k]) if est_res[k] else "n/a"))
    input_details = {d["CLASS_NUMBER"]: d for d in refinements.load_details(conn, state["input_refinement_id"])}
    details = []
    for k in range(classes):
        avg_occ = refinements.average_occupancy(class_rows[k])
        high = s["high_resolution_limit_a"] if s["refine_defocus"] else float(input_details.get(k + 1, {}).get("HIGH_RESOLUTION_LIMIT") or s["high_resolution_limit_a"])
        details.append({
            "REFERENCE_VOLUME_ASSET_ID": previous_refs[k], "LOW_RESOLUTION_LIMIT": s["low_resolution_limit_a"], "HIGH_RESOLUTION_LIMIT": high,
            "MASK_RADIUS": s["mask_radius_a"], "SIGNED_CC_RESOLUTION_LIMIT": 0.0, "GLOBAL_RESOLUTION_LIMIT": s["high_resolution_limit_a"],
            "GLOBAL_MASK_RADIUS": 0.0, "NUMBER_RESULTS_TO_REFINE": 0, "ANGULAR_SEARCH_STEP": 0.0, "SEARCH_RANGE_X": 0.0, "SEARCH_RANGE_Y": 0.0,
            "CLASSIFICATION_RESOLUTION_LIMIT": 0.0, "SHOULD_FOCUS_CLASSIFY": 0, "SPHERE_X_COORD": 0.0, "SPHERE_Y_COORD": 0.0, "SPHERE_Z_COORD": 0.0, "SPHERE_RADIUS": 0.0,
            "SHOULD_REFINE_CTF": 1 if s["refine_defocus"] else 0, "DEFOCUS_SEARCH_RANGE": s["defocus_search_range_a"], "DEFOCUS_SEARCH_STEP": s["defocus_search_step_a"],
            "AVERAGE_OCCUPANCY": avg_occ, "ESTIMATED_RESOLUTION": est_res[k] or 0.0, "RECONSTRUCTED_VOLUME_ASSET_ID": volume_ids[k], "RECONSTRUCTION_ID": recon_ids[k],
            "SHOULD_AUTOMASK": 1 if s["auto_mask"] else 0, "SHOULD_REFINE_INPUT_PARAMS": 0,
            "SHOULD_USE_SUPPLIED_MASK": 1 if s["use_mask"] else 0, "MASK_ASSET_ID": state.get("mask_asset_id", -1), "MASK_EDGE_WIDTH": s["mask_edge_a"],
            "OUTSIDE_MASK_WEIGHT": s["outside_mask_weight"], "SHOULD_LOWPASS_OUTSIDE_MASK": 1 if s["low_pass_outside_mask"] else 0,
            "MASK_FILTER_RESOLUTION": s["mask_filter_resolution_a"],
        })
    ref = {"refinement_id": rid, "refinement_package_asset_id": state["package_id"],
           "name": refinement_name(rid, s["refine_defocus"], s["refine_beam_tilt"], bool(state.get("zernike"))), "resolution_statistics_are_generated": False,
           "aberrations": state["aberrations"]["sets"] if state.get("aberrations") else None,
           "starting_refinement_id": state["input_refinement_id"], "number_of_particles": state["number_of_particles"], "number_of_classes": classes,
           "resolution_statistics_box_size": state["box_size"], "resolution_statistics_pixel_size": state["pixel_size"], "percent_used": 100.0,
           "job_id": parent_id}
    refinements.add_refinement(conn, ref, class_rows, stats, details, symmetry=state["symmetry"])
    state["history"].append({"round": 1, "refinement_id": rid, "label": "CTF refinement", "estimated_resolution": est_res,
                             "average_sigma": average_sigma(class_rows), "finished_at": now_iso()})
    _log(project_id, parent_id, "Refinement #{} written ({}); volume asset{} {}".format(rid, ref["name"], "" if classes == 1 else "s", ", ".join("#{}".format(v) for v in volume_ids)))


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
    metrics = {"refinement_id": state.get("output_refinement_id") if status == "completed" else None, "volume_ids": state.get("volume_ids", []),
               "beam_tilt": state.get("beam_tilt")}
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


def _live_histogram(state):
    """The defocus-change histogram: the recorded one once the refinement
    step is done, else read live from the progress files against the input."""
    if state.get("defocus_histogram"):
        return state["defocus_histogram"]
    s = state["settings"]
    if state.get("phase") != "refine" or not s.get("refine_defocus"):
        return None
    results = {}
    for p in state.get("progress_files") or []:
        results.update(refine_ctf_adapter.read_progress(p))
    if not results:
        return None
    try:
        rows = starfile.read_params(_class_star(state, "input", 0), as_table=True)
    except (OSError, ValueError):
        return None
    changes = [results[int(p)][0] - float(d) for p, d in zip(rows["position_in_stack"].tolist(), rows["defocus_1"].tolist()) if int(p) in results]
    centres, counts = defocus_histogram(changes, s["defocus_search_range_a"], s["defocus_search_step_a"])
    return {"centres": centres, "counts": counts, "n": len(changes)}


def live_result(conn, row):
    """The Jobs tab's Latest Result: the defocus-change histogram as it
    fills (DefocusHistorgramPlotPanel), the beam tilt once found, then the
    FSC and the reconstruction."""
    state = _load_state(conn, row["JOB_ID"])
    if not state:
        return None
    hist = _live_histogram(state)
    if not hist and not state.get("history") and not state.get("beam_tilt"):
        return None
    last = state["history"][-1] if state.get("history") else None
    rid = state.get("output_refinement_id")
    stats = refinements.load_statistics(conn, rid, 1) if last else []
    files = [p for p in (state.get("reference_files") or []) if p and os.path.isfile(p)]
    n_done = hist["n"] if hist else 0
    return {
        "kind": "refine_ctf", "task_index": (len(state.get("history", [])) * 1000) + (1 if state.get("phase") == "finished" else 0) + (10 if state.get("beam_tilt") else 0) + n_done,
        "label": last["label"] if last else "CTF refinement", "refinement_id": rid, "phase": state["phase"], "round": 0, "rounds": 1,
        "number_of_classes": state["number_of_classes"], "has_picture": bool(files) and bool(last), "current_volume": files[0] if files and last else None,
        "estimated_resolution": last.get("estimated_resolution") if last else None, "history": state.get("history", []),
        "defocus_histogram": hist, "beam_tilt": state.get("beam_tilt"), "aberrations": state.get("aberrations"),
        "refine_defocus": state["settings"]["refine_defocus"], "refine_beam_tilt": state["settings"]["refine_beam_tilt"],
        "fsc": [{"resolution": s["resolution"], "fsc": s["fsc"], "part_fsc": s["part_fsc"]} for s in stats if s["resolution"]],
        "pixel_size": state["pixel_size"], "volume_ids": state.get("volume_ids", []), "package_name": state.get("package_name"),
    }


def current_picture(conn, row, class_index=0):
    state = _load_state(conn, row["JOB_ID"])
    if not state or not state.get("history"):
        return None
    files = [p for p in (state.get("reference_files") or []) if p and os.path.isfile(p)]
    if not files:
        return None
    path = files[min(class_index, len(files) - 1)]
    png, meta = volumes.orthogonal_views_png_cached(path, state["settings"]["mask_radius_a"])
    return png, meta, path


def beam_tilt_picture(conn, row, which):
    """The phase-difference spectrum or the predicted beam-tilt phase image
    (Assets/PhaseDifferences), as a PNG through the image preview renderer."""
    state = _load_state(conn, row["JOB_ID"])
    bt = (state or {}).get("beam_tilt") or {}
    ab = (state or {}).get("aberrations") or {}
    files = ab.get("files") or {}
    m = GROUP_PICTURE_RE.match(which)
    if m:   # one optics group's map: group_<g>_<name>
        path = (ab.get("group_files") or {}).get(m.group(1), {}).get(m.group(2))
    else:
        path = files.get(which) if which in ABERRATION_PICTURES else bt.get("phase_difference_file" if which == "phase_difference" else "beam_tilt_file")
    return path if path and os.path.isfile(path) else None


ABERRATION_PICTURES = ("odd_measured", "odd_fitted", "even_measured", "even_fitted")
GROUP_PICTURE_RE = re.compile(r"^group_(\d+)_(odd_measured|odd_fitted|even_measured|even_fitted)$")


def group_pictures(state):
    """The per-group map names on disk, group_<g>_<name>, for ctf_refinement_result's `pictures`."""
    out = []
    for g, files in sorted(((ab := (state or {}).get("aberrations") or {}).get("group_files") or {}).items(), key=lambda kv: int(kv[0])):
        for name in ABERRATION_PICTURES:
            if files.get(name) and os.path.isfile(files[name]):
                out.append("group_{}_{}".format(g, name))
    return out


def ctf_refinement_result(conn, row):
    """What the Refine 3D results page shows under a refinement a Refine CTF job
    wrote -- cisTEM keeps these in the panel's live view only, this keeps them with
    the refinement: the recorded defocus-change histogram, the beam tilt with
    its significance, the Zernike fit's summary and sets, and which of the
    phase pictures (served by /jobs/<id>/refinectf/<which>.png) are on disk.
    None when the job's state is gone."""
    state = _load_state(conn, row["JOB_ID"])
    if not state:
        return None
    s = state["settings"]
    bt = dict(state.get("beam_tilt") or {})
    for k in ("phase_difference_file", "beam_tilt_file"):
        bt.pop(k, None)
    ab = state.get("aberrations")
    pictures = [w for w in ("phase_difference", "beam_tilt") + ABERRATION_PICTURES if beam_tilt_picture(conn, row, w)] + group_pictures(state)
    return {
        "job_id": row["JOB_ID"], "job_number": row["JOB_NUMBER"], "status": row["STATUS"],
        "refine_defocus": bool(s.get("refine_defocus")), "refine_beam_tilt": bool(s.get("refine_beam_tilt")), "zernike": bool(state.get("zernike")),
        "defocus_search_range_a": s.get("defocus_search_range_a"), "defocus_search_step_a": s.get("defocus_search_step_a"),
        "low_resolution_limit_a": s.get("low_resolution_limit_a"), "high_resolution_limit_a": s.get("high_resolution_limit_a"),
        "defocus_histogram": state.get("defocus_histogram"), "beam_tilt": bt or None,
        "aberrations": {"summary": ab.get("summary"), "sets": ab.get("sets"), "groups": ab.get("groups")} if ab else None,
        "optics_groups": state.get("optics_groups") or [],
        "pictures": pictures,
    }
