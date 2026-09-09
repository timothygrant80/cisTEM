"""The data behind the Display panel (cisTEM's DisplayPanel): one section of
an image file as numbers rather than a picture, so the browser can set the
grey range, zoom, invert, step through a stack and read pixel values itself.

`file_info()` reads the header: sizes, section count, pixel size, format.
`read_section()` returns one section (1-based; 0 is the sum of the first
MAX_SUM_SECTIONS sections, which is how a movie is first shown) as float32,
integer-binned so its longer edge fits `max_edge` -- the client draws that
and zooms it, since a 4k micrograph at 100% is not what anyone looks at
first, and a bin factor is reported so cursor positions map back to the
file's pixels. Rows stay in file order: an MRC's first row is the bottom of
the image (cisTEM's y up), and the client flips when drawing.
`global_range()` is what cisTEM's Global greys needs: the min and max over
every section (capped), read one section at a time.

MRC / MRCS (modes 0, 1, 2, 6, 12) and TIFF / BigTIFF through Pillow, as
preview.py; EER is not readable here for the same reasons it has no
preview.
"""
import os
import struct
from pathlib import Path

import numpy as np

import preview

DISPLAY_EXTENSIONS = {".mrc", ".mrcs", ".tif", ".tiff"}
MAX_SUM_SECTIONS = 200
MAX_GLOBAL_SECTIONS = 500


class DisplayError(Exception):
    pass


def _check(path):
    path = str(path or "").strip()
    if not path:
        raise DisplayError("a file path is required")
    if Path(path).suffix.lower() not in DISPLAY_EXTENSIONS:
        raise DisplayError("not a displayable file type ({})".format(Path(path).suffix or "no extension"))
    if not os.path.isfile(path):
        raise DisplayError("no such file: {}".format(path))
    return path


def _mrc_header(path):
    with open(path, "rb") as fh:
        head = fh.read(1024)
    if len(head) < 1024:
        raise DisplayError("file is shorter than an MRC header")
    endian = ">" if head[212:214] == b"\x11\x11" else "<"
    nx, ny, nz, mode = struct.unpack_from(endian + "iiii", head, 0)
    mx = struct.unpack_from(endian + "i", head, 28)[0]
    cella_x = struct.unpack_from(endian + "f", head, 40)[0]
    nsymbt = struct.unpack_from(endian + "i", head, 92)[0]
    if min(nx, ny, nz) <= 0:
        raise DisplayError("implausible MRC dimensions ({}x{}x{})".format(nx, ny, nz))
    if mode not in preview.MRC_MODES:
        raise DisplayError("unsupported MRC mode ({})".format(mode))
    return {"nx": nx, "ny": ny, "nz": nz, "mode": mode, "endian": endian, "nsymbt": max(nsymbt, 0),
            "pixel_size": (cella_x / mx) if mx else 0.0}


def _is_tiff(path):
    return Path(path).suffix.lower() in (".tif", ".tiff")


def file_info(path):
    path = _check(path)
    if _is_tiff(path):
        from PIL import Image
        with Image.open(path) as im:
            nz = getattr(im, "n_frames", 1)
            nx, ny = im.size
        return {"path": path, "format": "tiff", "nx": nx, "ny": ny, "nz": nz, "pixel_size": 0.0}
    h = _mrc_header(path)
    return {"path": path, "format": "mrc", "nx": h["nx"], "ny": h["ny"], "nz": h["nz"], "pixel_size": h["pixel_size"], "mode": h["mode"]}


def _read_mrc_section(path, h, section):
    dtype = np.dtype(preview.MRC_MODES[h["mode"]]).newbyteorder(h["endian"])
    n = h["nx"] * h["ny"]
    with open(path, "rb") as fh:
        fh.seek(1024 + h["nsymbt"] + (section - 1) * n * dtype.itemsize)
        raw = np.fromfile(fh, dtype=dtype, count=n)
    if raw.size < n:
        raise DisplayError("file ends before section {}".format(section))
    return raw.reshape(h["ny"], h["nx"]).astype(np.float32)


def _read_tiff_section(path, section):
    from PIL import Image
    with Image.open(path) as im:
        im.seek(section - 1)
        page = np.asarray(im, dtype=np.float32)
    if page.ndim == 3:
        page = page.mean(axis=2)
    return page


def _read(path, info, section):
    if info["format"] == "tiff":
        return _read_tiff_section(path, section)
    return _read_mrc_section(path, _mrc_header(path), section)


def bin_image(image, max_edge):
    """Integer-bin (mean) so the longer edge fits max_edge; the edge pixels
    that do not fill a bin are dropped. Returns (binned, factor)."""
    ny, nx = image.shape
    factor = max(1, -(-max(nx, ny) // int(max_edge))) if max_edge else 1
    if factor == 1:
        return image, 1
    by, bx = (ny // factor) * factor, (nx // factor) * factor
    binned = image[:by, :bx].reshape(by // factor, factor, bx // factor, factor).mean(axis=(1, 3), dtype=np.float32)
    return binned.astype(np.float32), factor


MAX_PAGE_SECTIONS = 400


def read_section(path, section=1, max_edge=1024, count=1):
    """`count` consecutive sections from `section` (or the sum, section 0)
    binned for display, concatenated, with the info the client needs --
    `count` is how many came back (fewer at the end of the stack), `width`
    and `height` the binned size of each. Returns (float32 array, info)."""
    info = file_info(path)
    section = int(section or 0)
    count = max(1, min(int(count or 1), MAX_PAGE_SECTIONS))
    if section < 0 or section > info["nz"]:
        raise DisplayError("section {} is not in the file's {} sections".format(section, info["nz"]))
    if section == 0:
        summed = min(info["nz"], MAX_SUM_SECTIONS)
        acc = None
        for s in range(1, summed + 1):
            page = _read(path, info, s)
            acc = page if acc is None else acc + page
        images = [acc]
        info["summed"] = summed
    else:
        images = [_read(path, info, s) for s in range(section, min(info["nz"], section + count - 1) + 1)]
    binned = [bin_image(im, max_edge) for im in images]
    factor = binned[0][1]
    stack = np.stack([b[0] for b in binned])
    finite = stack[np.isfinite(stack)]
    info.update({"section": section, "count": len(images), "bin": factor, "width": int(stack.shape[2]), "height": int(stack.shape[1]),
                 "min": float(finite.min()) if finite.size else 0.0, "max": float(finite.max()) if finite.size else 0.0,
                 "mean": float(finite.mean()) if finite.size else 0.0, "std": float(finite.std()) if finite.size else 0.0})
    return np.ascontiguousarray(stack, dtype="<f4"), info


def global_range(path, max_edge=1024, max_sections=MAX_GLOBAL_SECTIONS):
    """Min and max over every section (the first max_sections of a long
    stack), each binned as the display would be, for Global greys."""
    info = file_info(path)
    lo, hi = None, None
    n = min(info["nz"], max_sections)
    for s in range(1, n + 1):
        binned, _f = bin_image(_read(path, info, s), max_edge)
        finite = binned[np.isfinite(binned)]
        if not finite.size:
            continue
        a, b = float(finite.min()), float(finite.max())
        lo = a if lo is None else min(lo, a)
        hi = b if hi is None else max(hi, b)
    return {"min": lo if lo is not None else 0.0, "max": hi if hi is not None else 0.0, "sections": n, "nz": info["nz"]}
