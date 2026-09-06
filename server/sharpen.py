"""Sharpen 3D: cisTEM's Sharpen3DPanel (src/gui/Sharpen3DPanel.cpp) behind
one synchronous request, the way the Find Particles preview is.

The panel's SharpenMapThread calls Image::SharpenMap() in the GUI process;
here the `sharpen_map` program is run directly (its interactive prompts fed
on stdin) on the volume asset, which gives the same SharpenMap() -- the
masking, spectral flattening, B-factors, optional figure-of-merit weighting
from the refinement's rec_SSNR, cosine cut-off, handedness flip and the
normalisation to 0.03. Two things the panel does around that call are done
here in numpy, since the program has no switch for them: the **auto-mask**
(ConvertToAutoMask at 10 A / 0.1, handed to the program as the 3D mask,
exactly the mask the thread builds) and **Correct Gridding Error**
(Image::CorrectSinc(outer mask radius), applied to the program's output).

The result is kept under the project's Scratch/Sharpen/ until the server
restarts: the page shows the Guinier plot (log amplitude of the masked
original and sharpened maps against spatial frequency, the sharpened curve
scaled onto the original's second point as OnSharpenThreadComplete() does)
and the central slices of both, and offers "Save Result" (the MRC itself,
as cisTEM's save dialog does) and "Import" as a new volume asset -- the
button cisTEM shows but never wired.
"""
import base64
import math
import os
import shutil
import subprocess
import threading
import uuid

import numpy as np

import db
import preview
import refinements
import volumes
from abinitio import write_statistics

# Sharpen3DPanel::ResetDefaults()
DEFAULTS = {
    "flatten_from_res_a": 12.0, "cutoff_res_a": 0.0, "pre_cutoff_bfactor": -90.0, "post_cutoff_bfactor": 0.0, "filter_edge_width_a": 20.0,
    "use_fom_weighting": True, "ssnr_scale_factor": 1.0, "inner_mask_radius_a": 0.0, "outer_mask_radius_a": 100.0,
    "auto_mask": True, "invert_handedness": False, "correct_gridding": True, "use_mask": False,
}
COSINE_EDGE_A = 10.0  # SharpenMap()'s cosine_edge
TIMEOUT_S = 900

_lock = threading.Lock()
_results = {}


def _num(params, key, default):
    v = params.get(key)
    if v is None or v == "":
        return float(default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _flag(params, key, default):
    v = params.get(key)
    if v is None or v == "":
        return bool(default)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def volume_context(conn, volume_id):
    """Sharpen3DPanel::OnVolumeComboBox(): what is known about a volume --
    the reconstruction that made it (its mask radii) and, when the
    refinement still points at this volume, the refinement and class whose
    statistics can weight the sharpening."""
    vol = conn.execute("SELECT * FROM VOLUME_ASSETS WHERE VOLUME_ASSET_ID=?", (int(volume_id),)).fetchone()
    if vol is None:
        raise LookupError("no such volume")
    ps = float(vol["PIXEL_SIZE"] or 1.0)
    out = {"volume_asset_id": vol["VOLUME_ASSET_ID"], "name": vol["NAME"], "pixel_size": ps, "x_size": vol["X_SIZE"],
           "inner_mask_radius_a": 0.0, "outer_mask_radius_a": ps * float(vol["X_SIZE"]) * 0.5,
           "refinement_id": None, "class_number": None, "has_statistics": False, "estimated_resolution": None}
    recon_id = vol["RECONSTRUCTION_JOB_ID"]
    if recon_id is not None and recon_id >= 0:
        recon = conn.execute("SELECT REFINEMENT_ID, CLASS_NUMBER, INNER_MASK_RADIUS, OUTER_MASK_RADIUS FROM RECONSTRUCTION_LIST WHERE RECONSTRUCTION_ID=?",
                             (int(recon_id),)).fetchone()
        if recon is not None:
            rid, k = recon["REFINEMENT_ID"], recon["CLASS_NUMBER"]
            details = refinements.load_details(conn, rid)
            mine = [d for d in details if d.get("CLASS_NUMBER") == k]
            if mine and mine[0].get("RECONSTRUCTED_VOLUME_ASSET_ID") == vol["VOLUME_ASSET_ID"]:
                stats = refinements.load_statistics(conn, rid, k)
                out.update({"inner_mask_radius_a": float(recon["INNER_MASK_RADIUS"] or 0.0), "outer_mask_radius_a": float(recon["OUTER_MASK_RADIUS"] or out["outer_mask_radius_a"]),
                            "refinement_id": rid, "class_number": k, "has_statistics": bool(stats),
                            "estimated_resolution": refinements.estimated_resolution(stats, ps) if stats else None})
    # the defaults the panel takes when this volume is picked
    d = dict(DEFAULTS)
    d["inner_mask_radius_a"] = out["inner_mask_radius_a"]
    d["outer_mask_radius_a"] = round(out["outer_mask_radius_a"], 2)
    if out["has_statistics"]:
        d["cutoff_res_a"] = round(out["estimated_resolution"], 2)
        d["use_fom_weighting"] = True
    else:
        d["cutoff_res_a"] = 3.5
        d["use_fom_weighting"] = False
    out["defaults"] = d
    return out, vol


# ---------------------------------------------------------------------------
# numpy pieces of the thread's work
# ---------------------------------------------------------------------------

def _sinc(x):
    return np.sinc(x / math.pi)


def correct_sinc(volume, mask_radius_px):
    """Image::CorrectSinc(wanted_mask_radius): divide the density (about
    the background level) by the squared product of the three axis sinc
    weights inside the mask radius, and by the weight at the radius outside
    it -- the real-space attenuation trilinear interpolation leaves towards
    the box edge."""
    v = np.asarray(volume, dtype=np.float32)
    nz, ny, nx = v.shape
    # ReturnAverageOfRealValues(0.45 * x, outside = true)
    r2 = volumes._radius_grid(v.shape)
    outside = v[r2 > (0.45 * nx) ** 2]
    average = float(outside.mean()) if outside.size else float(v.mean())
    if mask_radius_px <= 0:
        mask_radius_px = nx + ny + nz
    cz, cy, cx = volumes._centre(nz), volumes._centre(ny), volumes._centre(nx)
    z = (np.arange(nz) - cz).astype(np.float32)
    y = (np.arange(ny) - cy).astype(np.float32)
    x = (np.arange(nx) - cx).astype(np.float32)
    wz = _sinc(z * math.pi / nz)[:, None, None]
    wy = _sinc(y * math.pi / ny)[None, :, None]
    wx = _sinc(x * math.pi / nx)[None, None, :]
    weight = (wx * wy * wz) ** 2
    inside = r2 < mask_radius_px ** 2
    weight_outside = _sinc(mask_radius_px * math.pi / nx) ** 2
    out = np.where(inside, (v - average) / weight + average, (v - average) / weight_outside + average)
    return out.astype(np.float32)


def cosine_ring_mask(volume, inner_px, outer_px, edge_px):
    """Image::CosineRingMask(): 1 between the radii, a cosine of `edge_px`
    to zero outside the outer (and inside the inner) radius."""
    r = np.sqrt(volumes._radius_grid(volume.shape))
    w = np.ones(volume.shape, dtype=np.float32)
    if edge_px <= 0:
        edge_px = 1.0
    outer_lo = outer_px - edge_px / 2.0
    band = (r > outer_lo) & (r < outer_lo + edge_px)
    w[band] = 0.5 * (1.0 + np.cos(math.pi * (r[band] - outer_lo) / edge_px))
    w[r >= outer_lo + edge_px] = 0.0
    if inner_px > 0:
        inner_hi = inner_px + edge_px / 2.0
        band = (r < inner_hi) & (r > inner_hi - edge_px)
        w[band] *= 0.5 * (1.0 - np.cos(math.pi * (r[band] - (inner_hi - edge_px)) / edge_px))
        w[r <= inner_hi - edge_px] = 0.0
    return volume * w


def guinier_curve(volume, pixel_size):
    """The Guinier plot's curve: log of the rotationally averaged Fourier
    amplitude per shell of one Fourier pixel, out to Nyquist, against
    spatial frequency (1/A). Shell 0 is left out as cisTEM's plot does."""
    n = volume.shape[0]
    ft = np.fft.fftn(volume.astype(np.float32))
    power = np.abs(ft) ** 2
    freqs = [np.fft.fftfreq(s) for s in volume.shape]
    fz, fy, fx = np.meshgrid(*freqs, indexing="ij")
    radius = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2) * n  # in Fourier pixels
    shell = np.rint(radius).astype(np.int64)
    n_shells = n // 2 + 1
    keep = shell < n_shells
    sums = np.bincount(shell[keep], weights=power[keep], minlength=n_shells)
    counts = np.bincount(shell[keep], minlength=n_shells)
    amp = np.sqrt(np.divide(sums, np.maximum(counts, 1)))
    xs, ys = [], []
    for s in range(1, n_shells):
        if amp[s] <= 0:
            continue
        xs.append(s / (n * pixel_size))
        ys.append(float(math.log(amp[s])))
    return xs, ys


def _slices_png(volume, pixel_size, mask_radius_a):
    """CreateOrthogonalProjectionsImage(include_projections=false): the three
    central slices side by side."""
    canvas = volumes.orthogonal_views(volume, mask_radius_a / pixel_size if pixel_size and mask_radius_a else 0.0, include_projections=False)
    gray = (np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8)
    return "data:image/png;base64," + base64.b64encode(preview._encode_png(gray)).decode("ascii")


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def scratch_dir(project_id):
    d = db.project_dir(project_id) / "Scratch" / "Sharpen"
    d.mkdir(parents=True, exist_ok=True)
    return d


KEEP_RESULTS = 3  # per project: the maps a user might still save or import


def _prune(project_id, scratch):
    """Keep the newest few results' maps; everything else under
    Scratch/Sharpen -- older results, leftovers from before a restart -- goes."""
    mine = [rid for rid, r in _results.items() if r["project_id"] == project_id]
    for rid in mine[:-KEEP_RESULTS]:
        _results.pop(rid, None)
    keep = {os.path.basename(r["path"]) for rid, r in _results.items() if r["project_id"] == project_id}
    for p in scratch.iterdir():
        if p.name not in keep:
            try:
                p.unlink()
            except OSError:
                pass


def run(conn, project_id, volume_id, params, executable):
    """Sharpen3DPanel::OnRunButtonClick() + SharpenMapThread::Entry()."""
    ctx, vol = volume_context(conn, volume_id)
    if not vol["FILENAME"] or not os.path.isfile(vol["FILENAME"]):
        raise ValueError("the volume file {} is missing".format(vol["FILENAME"]))
    ps = ctx["pixel_size"]
    n = int(vol["X_SIZE"])
    s = {k: (_flag(params, k, v) if isinstance(v, bool) else _num(params, k, v)) for k, v in ctx["defaults"].items()}
    if s["use_mask"]:
        s["auto_mask"] = False
    if s["cutoff_res_a"] <= 0:
        raise ValueError("the resolution cut-off must be positive")
    if s["flatten_from_res_a"] <= 0:
        raise ValueError("the flattening resolution must be positive")
    if s["filter_edge_width_a"] <= 0:
        raise ValueError("the filter edge width must be positive")
    mask_file = ""
    mask_volume = None
    if s["use_mask"]:
        mid = params.get("mask_volume_id")
        if mid in (None, ""):
            raise ValueError("Pick a mask volume, or untick Supply a Mask.")
        mask = conn.execute("SELECT * FROM VOLUME_ASSETS WHERE VOLUME_ASSET_ID=?", (int(mid),)).fetchone()
        if mask is None or not mask["FILENAME"] or not os.path.isfile(mask["FILENAME"]):
            raise ValueError("the mask volume is missing")
        if mask["X_SIZE"] != vol["X_SIZE"]:
            raise ValueError("Volume and mask file have different dimensions")
        mask_file = mask["FILENAME"]
    use_stats = bool(s["use_fom_weighting"] and ctx["has_statistics"])
    if not _lock.acquire(timeout=1.0):
        raise RuntimeError("another sharpening is running; try again in a moment")
    try:
        rid = uuid.uuid4().hex[:10]
        scratch = scratch_dir(project_id)
        _prune(project_id, scratch)
        out_path = str(scratch / "sharpened_{}.mrc".format(rid))
        original, file_ps = volumes.read_mrc_volume(vol["FILENAME"])
        outer_px = s["outer_mask_radius_a"] / ps if s["outer_mask_radius_a"] > 0 else n / 2.0
        if s["auto_mask"] and not mask_file:
            # The thread's ConvertToAutoMask(pixel_size, outer_mask_radius, 10.0, 0.1), handed over as the 3D mask.
            mask_volume = volumes.convert_to_auto_mask(original, ps, outer_px * ps, filter_resolution_a=10.0, rebin_value=0.1)
            mask_file = str(scratch / "automask_{}.mrc".format(rid))
            volumes.write_mrc_volume(mask_file, mask_volume, ps)
        elif mask_file:
            mask_volume, _ = volumes.read_mrc_volume(mask_file)
        stats_file = ""
        if use_stats:
            stats = [st for st in refinements.load_statistics(conn, ctx["refinement_id"], ctx["class_number"]) if 1 <= st["shell"] <= n // 2]
            stats_file = str(scratch / "statistics_{}.txt".format(rid))
            write_statistics(stats_file, stats, ps)
        answers = [vol["FILENAME"], out_path, mask_file or "/dev/null", stats_file or "/dev/null", "Yes" if use_stats else "No",
                   "{:.6f}".format(ps), "{:.4f}".format(s["inner_mask_radius_a"]), "{:.4f}".format(s["outer_mask_radius_a"]),
                   "{:.4f}".format(s["pre_cutoff_bfactor"]), "{:.4f}".format(s["post_cutoff_bfactor"]), "{:.4f}".format(s["flatten_from_res_a"]),
                   "{:.4f}".format(s["cutoff_res_a"]), "{:.4f}".format(s["filter_edge_width_a"]), "{:.4f}".format(min(10.0, max(0.1, s["ssnr_scale_factor"]))),
                   "Yes" if mask_file else "No", "Yes" if s["invert_handedness"] else "No"]
        try:
            proc = subprocess.run([executable], input="\n".join(answers) + "\n", capture_output=True, text=True, cwd=str(scratch), timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise TimeoutError("sharpen_map did not finish within {} s".format(TIMEOUT_S))
        if proc.returncode != 0 or not os.path.isfile(out_path):
            tail = " | ".join((proc.stdout or "").strip().splitlines()[-3:]) or (proc.stderr or "")[-300:]
            raise RuntimeError("sharpen_map failed: {}".format(tail or "no output written"))
        sharpened, _ = volumes.read_mrc_volume(out_path)
        if s["correct_gridding"]:
            sharpened = correct_sinc(sharpened, outer_px)
            volumes.write_mrc_volume(out_path, sharpened, ps)
        # The Guinier curves, both maps masked as SharpenMap() masks them.
        edge_px = COSINE_EDGE_A / ps
        if mask_volume is not None:
            m_orig = volumes.apply_mask(original, mask_volume, edge_px, 0.0)
            m_sharp = volumes.apply_mask(sharpened, mask_volume, edge_px, 0.0)
        else:
            m_orig = cosine_ring_mask(original, s["inner_mask_radius_a"] / ps, outer_px, edge_px)
            m_sharp = cosine_ring_mask(sharpened, s["inner_mask_radius_a"] / ps, outer_px, edge_px)
        xs, ys_orig = guinier_curve(m_orig, ps)
        _xs, ys_sharp = guinier_curve(m_sharp, ps)
        if len(ys_orig) > 1 and len(ys_sharp) > 1 and ys_sharp[0] != 0:
            scale = ys_orig[0] / ys_sharp[0]  # OnSharpenThreadComplete(): the sharpened curve onto the original's first plotted point
            ys_sharp = [v * scale for v in ys_sharp]
        display_radius = s["outer_mask_radius_a"] if s["outer_mask_radius_a"] > 0 else 0.0
        result = {
            "result_id": rid, "volume_asset_id": vol["VOLUME_ASSET_ID"], "volume_name": vol["NAME"], "pixel_size": ps, "x_size": n,
            "settings": s, "used_statistics": use_stats, "used_mask": "supplied" if s["use_mask"] else ("auto" if s["auto_mask"] else "none"),
            "guinier": {"spatial_frequency": xs, "original": ys_orig, "sharpened": ys_sharp, "nyquist": 0.5 / ps},
            "original_png": _slices_png(original, ps, display_radius), "sharpened_png": _slices_png(sharpened, ps, display_radius),
            "elapsed_s": None,
        }
        _results[rid] = {"path": out_path, "volume_asset_id": vol["VOLUME_ASSET_ID"], "volume_name": vol["NAME"], "pixel_size": ps, "x_size": n,
                         "project_id": project_id, "settings": s}
        for p in (mask_file if s["auto_mask"] and not s["use_mask"] else "", stats_file):
            if p:
                try:
                    os.remove(p)
                except OSError:
                    pass
        return result
    finally:
        _lock.release()


def result(project_id, result_id):
    r = _results.get(result_id)
    if r is None or r["project_id"] != project_id or not os.path.isfile(r["path"]):
        return None
    return r


def import_result(conn, project_id, result_id, name=None):
    """The Import button: the sharpened map as a new volume asset under
    Assets/Volumes (no reconstruction record: it is not a reconstruction)."""
    r = result(project_id, result_id)
    if r is None:
        raise LookupError("no such sharpening result (results are kept until the server restarts)")
    vol_dir = volumes.volume_dir(project_id)
    nxt = conn.execute("SELECT COALESCE(MAX(VOLUME_ASSET_ID), 0) + 1 FROM VOLUME_ASSETS").fetchone()[0]
    dest = str(vol_dir / "sharpened_volume_{}_{}.mrc".format(r["volume_asset_id"], nxt))
    shutil.copyfile(r["path"], dest)
    name = (name or "").strip() or "{} (sharpened)".format(r["volume_name"])
    vid = volumes.add_volume_asset(conn, name, dest, r["pixel_size"], r["x_size"], r["x_size"], r["x_size"])
    return {"volume_asset_id": vid, "name": name, "filename": dest}
