"""3D volumes: MRC volume I/O, the few Image operations cisTEM's GUI does
on volumes between job steps, and the VOLUME_ASSETS / STARTUP_LIST tables.

The ab-initio panel (server/abinitio.py) does three things to a volume in
the GUI process rather than in a program: it **auto-masks** the current
reference before each refinement round (AutoMaskerThread ->
Image::ConvertToAutoMask), it **resamples** the final reconstruction to
the package's box size before registering it as a volume asset
(ResampleVolumeThread -> ForwardFFT / Resize / BackwardFFT), and it draws
the **orthogonal slices and projections** it shows while running
(OrthDrawerThread -> Image::CreateOrthogonalProjectionsImage). Those three
are ported here with numpy (and scipy's connected-component labelling for
the mask); the results follow cisTEM's arithmetic, including its FFT
normalisation (forward unnormalised, inverse divided by the *output* size,
which numpy's default also does).
"""
import math
import os
import struct

import numpy as np

import db
import preview

_MRC_DTYPES = {0: "i1", 1: "i2", 2: "f4", 6: "u2", 12: "f2"}


# ---------------------------------------------------------------------------
# MRC volumes
# ---------------------------------------------------------------------------

def read_mrc_header(path):
    with open(path, "rb") as fh:
        head = fh.read(1024)
    if len(head) < 1024:
        raise ValueError("{} is shorter than an MRC header".format(path))
    endian = ">" if head[212:214] == b"\x11\x11" else "<"
    nx, ny, nz, mode = struct.unpack_from(endian + "iiii", head, 0)
    mx = struct.unpack_from(endian + "i", head, 28)[0]
    cella_x = struct.unpack_from(endian + "f", head, 40)[0]
    nsymbt = struct.unpack_from(endian + "i", head, 92)[0]
    pixel_size = cella_x / mx if mx else 0.0
    return {"nx": nx, "ny": ny, "nz": nz, "mode": mode, "endian": endian, "nsymbt": nsymbt, "pixel_size": pixel_size}


def read_mrc_volume(path):
    """The whole file as float32 (z, y, x), rows in file order (cisTEM's y up)."""
    h = read_mrc_header(path)
    if h["mode"] not in _MRC_DTYPES:
        raise ValueError("unsupported MRC mode {} in {}".format(h["mode"], path))
    dtype = np.dtype(_MRC_DTYPES[h["mode"]]).newbyteorder(h["endian"])
    n = h["nx"] * h["ny"] * h["nz"]
    with open(path, "rb") as fh:
        fh.seek(1024 + max(h["nsymbt"], 0))
        data = np.frombuffer(fh.read(n * dtype.itemsize), dtype=dtype)
    if data.size != n:
        raise ValueError("{} ends early ({} of {} voxels)".format(path, data.size, n))
    return data.reshape(h["nz"], h["ny"], h["nx"]).astype(np.float32), h["pixel_size"]


def write_mrc_volume(path, volume, pixel_size):
    """Mode-2 MRC with the pixel size in the cell and the density statistics
    MRCFile::SetDensityStatistics() sets (min, max, mean, rms)."""
    arr = np.ascontiguousarray(volume, dtype="<f4")
    nz, ny, nx = arr.shape
    head = bytearray(1024)
    struct.pack_into("<iiii", head, 0, nx, ny, nz, 2)
    struct.pack_into("<iii", head, 28, nx, ny, nz)
    struct.pack_into("<fff", head, 40, nx * pixel_size, ny * pixel_size, nz * pixel_size)
    struct.pack_into("<fff", head, 52, 90.0, 90.0, 90.0)
    struct.pack_into("<iii", head, 64, 1, 2, 3)
    struct.pack_into("<fff", head, 76, float(arr.min()), float(arr.max()), float(arr.mean()))
    struct.pack_into("<ii", head, 88, 1, 0)                                # ispg = 1: a volume
    head[208:212] = b"MAP "
    head[212:216] = b"\x44\x44\x00\x00"
    struct.pack_into("<f", head, 216, float(arr.std()))
    struct.pack_into("<i", head, 220, 1)
    label = "cisTEM3 volume {}^3, {:.4f} A/px".format(nx, pixel_size).encode()[:80]
    head[224:224 + len(label)] = label
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(bytes(head))
        fh.write(arr.tobytes())


# ---------------------------------------------------------------------------
# Image operations
# ---------------------------------------------------------------------------

def _centre(n):
    # Image::physical_address_of_box_center: n/2 (integer) for both parities.
    return n // 2


def _radius_grid(shape):
    """Squared distance from the box centre, per voxel."""
    grids = np.meshgrid(*[(np.arange(n) - _centre(n)).astype(np.float32) ** 2 for n in shape], indexing="ij")
    return sum(grids)


def fourier_resize(volume, new_size):
    """ForwardFFT / Resize / BackwardFFT: crop (or zero-pad) the transform
    about the origin to `new_size` cubed and invert. Uses full (not
    half-plane) transforms so the crop is symmetric the way cisTEM's
    ClipInto of a complex image is."""
    volume = np.asarray(volume, dtype=np.float32)
    n = volume.shape[0]
    if new_size == n:
        return volume.copy()
    ft = np.fft.fftshift(np.fft.fftn(volume))
    c_old, c_new = _centre(n), _centre(new_size)
    out = np.zeros((new_size,) * 3, dtype=complex)
    lo = min(c_old, c_new)
    hi = min(n - c_old, new_size - c_new)
    src = tuple(slice(c_old - lo, c_old + hi) for _ in range(3))
    dst = tuple(slice(c_new - lo, c_new + hi) for _ in range(3))
    out[dst] = ft[src]
    return np.real(np.fft.ifftn(np.fft.ifftshift(out))).astype(np.float32)


def average_outside(volume, radius_px):
    """Image::ReturnAverageOfRealValues(radius, invert=true): the mean beyond the radius."""
    r2 = _radius_grid(volume.shape)
    outside = volume[r2 > radius_px ** 2]
    return float(outside.mean()) if outside.size else float(volume.mean())


def cosine_mask(volume, radius_px, edge_px=1.0, value=0.0):
    """Image::CosineMask(radius, edge, invert=false, force_mask_value=true, value):
    inside the radius (less half the edge) untouched, beyond radius plus
    edge set to `value`, a raised-cosine blend between."""
    inner = max(radius_px - edge_px * 0.5, 0.0)
    outer = inner + edge_px
    d = np.sqrt(_radius_grid(volume.shape))
    out = volume.astype(np.float32).copy()
    band = (d > inner) & (d < outer)
    w = 0.5 + 0.5 * np.cos(np.pi * (d[band] - inner) / edge_px)
    out[band] = out[band] * w + value * (1.0 - w)
    out[d >= outer] = value
    return out


def _average_of_max_n(volume, n_top, radius_px):
    """Image::ReturnAverageOfMaxN: the mean of the largest n voxels on the
    three central planes within a cube of the given half-width."""
    nz, ny, nx = volume.shape
    cz, cy, cx = _centre(nz), _centre(ny), _centre(nx)
    r = int(min(min(nx, ny, nz) / 2.0, radius_px)) if radius_px > 0 else int(min(nx, ny, nz) / 2.0)
    zs, ys, xs = np.ogrid[:nz, :ny, :nx]
    in_cube = (zs >= cz - r) & (zs < cz + r) & (ys >= cy - r) & (ys < cy + r) & (xs >= cx - r) & (xs < cx + r)
    on_plane = (xs == cx) | (ys == cy) | (zs == cz)
    vals = volume[in_cube & on_plane]
    if vals.size == 0:
        return float(volume.max())
    n_top = max(1, min(int(n_top), vals.size))
    return float(np.sort(vals)[-n_top:].mean())


def _largest_component(binary):
    from scipy import ndimage
    labels, n = ndimage.label(binary)
    if n <= 1:
        return binary.astype(np.float32)
    sizes = ndimage.sum(binary, labels, index=np.arange(1, n + 1))
    return (labels == (int(np.argmax(sizes)) + 1)).astype(np.float32)


def convert_to_auto_mask(volume, pixel_size, mask_radius_a, filter_resolution_a=7.0, rebin_value=0.1):
    """Image::ConvertToAutoMask(pixel_size, radius, 7.0, 0.1, auto_estimate=true):
    bin to the filter resolution, threshold at 5% of the way from the
    background level to the brightest density, keep the largest connected
    piece, smooth, resample back and rebinarise -> a 0/1 mask."""
    n = volume.shape[0]
    binning = filter_resolution_a / 2.0 / pixel_size
    binned = int(n / binning + 0.5)
    if binned % 2:
        binned += 1
    if binned > n:
        binned = n
    work = fourier_resize(volume, binned) if binned != n else volume.astype(np.float32).copy()
    r_b = mask_radius_a / binning / pixel_size if binned != n else mask_radius_a / pixel_size
    if binned == n:
        binning = 1.0
    original_average = average_outside(work, r_b)
    work = np.maximum(work, original_average)                     # SetMinimumValue
    average = average_outside(work, r_b)
    n_top = max(5, int(work.size * 0.000005))
    average_of_max = _average_of_max_n(work, n_top, r_b)
    threshold = average + (average_of_max - average) * 0.05
    work = cosine_mask(work, r_b, 1.0, value=-np.inf)
    binary = (work >= threshold)
    mask = _largest_component(binary)
    # GaussianLowPassFilter(binning / pixel_size / 16) in reciprocal pixels, then Resize back.
    sigma = binning / pixel_size / 16.0
    ft = np.fft.fftshift(np.fft.fftn(mask))
    freqs = [np.fft.fftshift(np.fft.fftfreq(binned)) for _ in range(3)]
    f2 = sum(np.meshgrid(*[f ** 2 for f in freqs], indexing="ij"))
    ft *= np.exp(-f2 / (2.0 * sigma ** 2))
    c_old, c_new = _centre(binned), _centre(n)
    out = np.zeros((n,) * 3, dtype=complex)
    lo, hi = min(c_old, c_new), min(binned - c_old, n - c_new)
    out[tuple(slice(c_new - lo, c_new + hi) for _ in range(3))] = ft[tuple(slice(c_old - lo, c_old + hi) for _ in range(3))]
    smooth = np.real(np.fft.ifftn(np.fft.ifftshift(out)))
    mx = float(smooth.max())
    if mx > 0:
        smooth = smooth / mx
    return (smooth >= rebin_value).astype(np.float32)


def auto_mask(volume, pixel_size, mask_radius_a):
    """AutoMaskerThread::Entry for one volume: everything the auto-mask
    excludes is set to the background level, nothing may fall below it,
    and a 1-pixel cosine edge at the mask radius takes the rest to zero."""
    r_px = mask_radius_a / pixel_size
    original_average = average_outside(volume, r_px)
    mask = convert_to_auto_mask(volume, pixel_size, mask_radius_a)
    out = np.where(mask == 0.0, np.float32(original_average), volume.astype(np.float32))
    out = np.maximum(out, original_average)
    return cosine_mask(out, r_px, 1.0, value=0.0)


def apply_mask(volume, mask, cosine_edge_px, weight_outside, low_pass_radius=0.0, filter_edge=0.0):
    """Image::ApplyMask(mask, edge, weight_outside, low_pass_radius, filter_edge)
    as Refine 3D's Multiply3DMaskerThread calls it: the mask is binarised
    (> 0), given a cosine edge of `cosine_edge_px` by convolution with a
    normalised cosine kernel, and the volume is kept inside it; outside it
    is replaced by the average density beyond 0.4 of the box, or -- with a
    weight and a low-pass radius (cycles/pixel) -- by that weight times a
    low-pass-filtered copy of the volume."""
    volume = np.asarray(volume, dtype=np.float32)
    n = volume.shape[0]
    binary = (np.asarray(mask) > 0.0).astype(np.float32)

    def cosine_blur(img):
        if cosine_edge_px <= 0.0:
            return img
        r = np.sqrt(_radius_grid(img.shape))
        kernel = np.where(r <= cosine_edge_px, (1.0 + np.cos(np.pi * r / cosine_edge_px)) / 2.0, 0.0).astype(np.float32)
        kernel /= kernel.sum()
        out = np.real(np.fft.ifftn(np.fft.fftn(img) * np.fft.fftn(np.fft.ifftshift(kernel))))
        out[np.abs(out) < 1e-3] = 0.0
        return out.astype(np.float32)

    soft = cosine_blur(binary)
    edge_value = average_outside(volume, 0.4 * n)
    if low_pass_radius > 0.0 and weight_outside > 0.0 and cosine_edge_px > 0.0:
        double = cosine_blur((soft > 0.1).astype(np.float32))
        blend = double * edge_value + (1.0 - double) * volume
        ft = np.fft.fftshift(np.fft.fftn(blend))
        freqs = [np.fft.fftshift(np.fft.fftfreq(m)) for m in volume.shape]
        f = np.sqrt(sum(np.meshgrid(*[q ** 2 for q in freqs], indexing="ij")))
        inner = max(low_pass_radius - filter_edge * 0.5, 0.0)
        w = np.clip((f - inner) / max(filter_edge, 1e-6), 0.0, 1.0)
        ft *= (0.5 + 0.5 * np.cos(np.pi * w))
        filtered = np.real(np.fft.ifftn(np.fft.ifftshift(ft))).astype(np.float32)
        return (soft * volume + weight_outside * (1.0 - soft) * filtered).astype(np.float32)
    return ((1.0 - soft) * edge_value + soft * volume).astype(np.float32)


def _circle_mask_with_edge_average(image, radius_px):
    """CircleMaskWithValue(radius, ReturnAverageOfRealValuesAtRadius(radius))."""
    h, w = image.shape
    ys, xs = np.ogrid[:h, :w]
    d2 = (ys - _centre(h)) ** 2 + (xs - _centre(w)) ** 2
    ring = np.abs(d2 - radius_px ** 2) < 4.0
    value = float(image[ring].mean()) if ring.any() else float(image.mean())
    out = image.copy()
    out[d2 > radius_px ** 2] = value
    return out


def orthogonal_views(volume, mask_radius_px=0.0, max_edge=160, include_projections=True):
    """Image::CreateOrthogonalProjectionsImage(): a 3 x 2 picture -- the
    three orthogonal projections on top, the three central slices below --
    each row normalised to its own range, each panel circle-masked at the
    mask radius. Returned in display order (first row on top). Panels are
    binned to at most `max_edge` px. Without `include_projections` (the
    Sharpen 3D panel's pictures) it is the single row of slices."""
    v = np.asarray(volume, dtype=np.float32)
    nz, ny, nx = v.shape
    cz, cy, cx = _centre(nz), _centre(ny), _centre(nx)
    slices = [v[cz, :, :], v[:, :, cx], v[:, cy, :]]
    projections = [v.sum(axis=0), v.sum(axis=2), v.sum(axis=1)]

    def bin2(img):
        f = max(1, -(-max(img.shape) // max_edge))
        if f == 1:
            return img
        h, w = img.shape[0] - img.shape[0] % f, img.shape[1] - img.shape[1] % f
        return img[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))

    slices = [bin2(s) for s in slices]
    projections = [bin2(p) for p in projections]
    scale = slices[0].shape[0] / float(nz)

    def normalise(group):
        lo = min(float(g.min()) for g in group)
        hi = max(float(g.max()) for g in group)
        rng = hi - lo if hi > lo else 1.0
        return [(g - lo) / rng for g in group]

    slices, projections = normalise(slices), normalise(projections)
    if mask_radius_px:
        r = mask_radius_px * scale
        slices = [_circle_mask_with_edge_average(s, r) for s in slices]
        projections = [_circle_mask_with_edge_average(p, r) for p in projections]
    th, tw = slices[0].shape
    if not include_projections:
        canvas = np.zeros((th, 3 * tw), dtype=np.float32)
        for i, sl in enumerate(slices):
            canvas[:, i * tw:(i + 1) * tw] = sl[:th, :tw]
        return canvas
    canvas = np.zeros((2 * th, 3 * tw), dtype=np.float32)
    for i, p in enumerate(projections):
        canvas[0:th, i * tw:(i + 1) * tw] = p[:th, :tw]
    for i, s in enumerate(slices):
        canvas[th:2 * th, i * tw:(i + 1) * tw] = s[:th, :tw]
    return canvas


ORTH_PANEL = 256   # each panel of the picture is drawn this size: a small box is scaled up to it, a large one binned down


def orthogonal_views_png(path, mask_radius_a=0.0, panel=ORTH_PANEL):
    """PNG of orthogonal_views() for a volume file, in cisTEM's display
    orientation (y up within each panel, projections above slices). Each
    panel is `panel` px: a box smaller than that is scaled up (bilinear)
    so a 108 px ab-initio map is not a postage stamp, a larger one is
    binned down as before; `scale` in the meta says by how much, and
    `upscaled` whether it was enlarged."""
    volume, pixel_size = read_mrc_volume(path)
    canvas = orthogonal_views(volume, mask_radius_a / pixel_size if pixel_size and mask_radius_a else 0.0, max_edge=panel)
    gray = (np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8)
    box = int(volume.shape[0])
    th = gray.shape[0] // 2
    upscaled = box < panel   # a box that only reached the panel size by resampling after binning is not "scaled up"
    if th != panel:
        from PIL import Image as _PILImage
        factor = panel / float(th)
        im = _PILImage.fromarray(gray).resize((int(round(gray.shape[1] * factor)), int(round(gray.shape[0] * factor))), _PILImage.BILINEAR)
        gray = np.asarray(im, dtype=np.uint8)
        th = gray.shape[0] // 2
    # Each panel's first row is its bottom (MRC order). _encode_png writes
    # bottom-up, which flips the whole canvas; flip each half so the rows
    # of panels stay in their places while the panels themselves turn.
    ordered = np.concatenate([gray[th:2 * th], gray[0:th]], axis=0)  # slices first -> end up below
    return preview._encode_png(ordered), {"width": gray.shape[1], "height": gray.shape[0], "pixel_size": pixel_size,
                                          "box": box, "panel": th, "scale": th / float(box), "upscaled": upscaled}


# ---------------------------------------------------------------------------
# Volume assets and startup runs
# ---------------------------------------------------------------------------

def volume_dir(project_id):
    d = db.project_dir(project_id) / "Assets" / "Volumes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def add_volume_asset(conn, name, filename, pixel_size, x_size, y_size, z_size, reconstruction_job_id=-1,
                     half_map_1="", half_map_2=""):
    """Database::AddNextVolumeAsset() plus membership of All Volumes."""
    with conn:
        vid = conn.execute("SELECT COALESCE(MAX(VOLUME_ASSET_ID), 0) + 1 FROM VOLUME_ASSETS").fetchone()[0]
        conn.execute("INSERT INTO VOLUME_ASSETS VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (vid, name, filename, reconstruction_job_id, pixel_size, x_size, y_size, z_size, half_map_1, half_map_2))
        conn.execute("INSERT OR IGNORE INTO VOLUME_GROUP_MEMBERS(GROUP_ID, VOLUME_ASSET_ID) VALUES (0, ?)", (vid,))
    return vid


def add_startup_job(conn, startup_id, package_id, name, settings, volume_ids, job_id=None):
    """Database::AddStartupJob(): the run's settings and its result volumes."""
    with conn:
        conn.execute("INSERT OR REPLACE INTO STARTUP_LIST(STARTUP_ID, REFINEMENT_PACKAGE_ASSET_ID, NAME, NUMBER_OF_STARTS, NUMBER_OF_CYCLES, "
                     "INITIAL_RES_LIMIT, FINAL_RES_LIMIT, AUTO_MASK, AUTO_PERCENT_USED, INITIAL_PERCENT_USED, FINAL_PERCENT_USED, MASK_RADIUS, "
                     "APPLY_LIKELIHOOD_BLURRING, SMOOTHING_FACTOR, JOB_ID) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (int(startup_id), int(package_id), name, int(settings["number_of_starts"]), int(settings["number_of_rounds"]),
                      float(settings["initial_resolution_limit"]), float(settings["final_resolution_limit"]),
                      1 if settings["auto_mask"] else 0, 1 if settings["auto_percent_used"] else 0,
                      float(settings["start_percent_used"]), float(settings["end_percent_used"]), float(settings["mask_radius"]),
                      1 if settings["apply_blurring"] else 0, float(settings["smoothing_factor"]), job_id))
        table = "STARTUP_RESULT_{}".format(int(startup_id))
        conn.execute("CREATE TABLE IF NOT EXISTS {}(CLASS_NUMBER INTEGER PRIMARY KEY, VOLUME_ASSET_ID INTEGER)".format(table))
        conn.execute("DELETE FROM {}".format(table))
        conn.executemany("INSERT INTO {} VALUES (?, ?)".format(table), [(k + 1, int(v)) for k, v in enumerate(volume_ids)])


def next_startup_id(conn):
    return conn.execute("SELECT COALESCE(MAX(STARTUP_ID), 0) + 1 FROM STARTUP_LIST").fetchone()[0]


def _table_exists(conn, name):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def list_startups(conn):
    out = []
    for r in conn.execute("SELECT s.*, rp.NAME AS PACKAGE_NAME, j.JOB_NUMBER FROM STARTUP_LIST s "
                          "LEFT JOIN REFINEMENT_PACKAGE_ASSETS rp ON rp.REFINEMENT_PACKAGE_ASSET_ID = s.REFINEMENT_PACKAGE_ASSET_ID "
                          "LEFT JOIN JOBS j ON j.JOB_ID = s.JOB_ID ORDER BY s.STARTUP_ID").fetchall():
        d = {k.lower(): r[k] for k in r.keys()}
        table = "STARTUP_RESULT_{}".format(r["STARTUP_ID"])
        d["volumes"] = []
        if _table_exists(conn, table):
            for k, vid in conn.execute("SELECT CLASS_NUMBER, VOLUME_ASSET_ID FROM {} ORDER BY CLASS_NUMBER".format(table)).fetchall():
                v = conn.execute("SELECT * FROM VOLUME_ASSETS WHERE VOLUME_ASSET_ID=?", (vid,)).fetchone()
                d["volumes"].append({"class_number": k, "volume_asset_id": vid, "name": v["NAME"] if v else None,
                                     "file_exists": bool(v and v["FILENAME"] and os.path.isfile(v["FILENAME"])), "filename": v["FILENAME"] if v else None})
        out.append(d)
    return out
