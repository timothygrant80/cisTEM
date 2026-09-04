"""
Renders a viewable PNG preview of a movie asset.

Movies are summed rather than shown a frame at a time. A single frame of
counting-mode data is almost pure shot noise -- at the doses these are taken
at, one frame of the sample MRC here is visually indistinguishable from
static -- so the sum is the only thing worth looking at without alignment.

MRC/MRCS and TIFF (including BigTIFF) are supported. EER is not: it uses
Thermo's electron-event encoding, which is a decoder of its own, and a single
EER frame holds ~0.008 e/pixel so a useful image means decoding most of the
file. `can_preview()` is what the API and the Display button both use to tell
what's renderable.

MRC is read directly -- it's raw pixels after a fixed header. TIFF goes
through Pillow, which decodes LZW in C: the same movie takes ~7s that way
versus ~4min through a hand-written Python LZW decoder, and Pillow also
covers the TIFF variants (predictors, deflate, tiling, 16-bit) that such a
decoder would not. numpy does the summing and rescaling.

Note this is the *preview* path, used on demand. Header reading
(imageheaders.py) stays dependency-free because it runs on every import.

PNG is written with stdlib zlib -- it's a handful of CRC'd chunks.
"""

import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

# Longest edge of the rendered preview. Big enough to see particles, small
# enough that the PNG stays a few hundred KB.
MAX_PREVIEW_EDGE = 1024
# Contrast range, as percentiles. Cryo-EM frames have outliers (hot pixels,
# ice) that would otherwise flatten everything else to mid-grey.
CLIP_PERCENTILES = (0.5, 99.5)
# Bounds the worst case on a movie with an unusual number of frames. Typical
# movies (50-100 frames) are unaffected; when it does bite, the caller reports
# how many frames were actually summed rather than pretending it was all.
MAX_PREVIEW_FRAMES = 200

MRC_PREVIEW_EXTENSIONS = {".mrc", ".mrcs"}
TIFF_PREVIEW_EXTENSIONS = {".tif", ".tiff"}
PREVIEWABLE_EXTENSIONS = MRC_PREVIEW_EXTENSIONS | TIFF_PREVIEW_EXTENSIONS

# MRC mode -> numpy dtype. Complex modes (3, 4) aren't images to display.
MRC_MODES = {
    0: np.int8,
    1: np.int16,
    2: np.float32,
    6: np.uint16,
    12: np.float16,
}


class PreviewError(Exception):
    """The movie could not be rendered."""


def can_preview(path):
    """Whether this file's format is one we can render yet."""
    return Path(path).suffix.lower() in PREVIEWABLE_EXTENSIONS


def _sum_mrc(path, max_frames=None):
    """Sums the frames of an MRC stack into one float32 image.

    Frames are read one at a time so peak memory stays at roughly one frame
    plus the accumulator, rather than the whole file -- these run to hundreds
    of megabytes and up.
    """
    with open(path, "rb") as fh:
        head = fh.read(1024)
        if len(head) < 1024:
            raise PreviewError("file is shorter than an MRC header")
        endian = ">" if head[212:214] == b"\x11\x11" else "<"
        nx, ny, nz, mode = struct.unpack_from(endian + "iiii", head, 0)
        nsymbt = struct.unpack_from(endian + "i", head, 92)[0]
        if min(nx, ny, nz) <= 0:
            raise PreviewError("implausible MRC dimensions ({}x{}x{})".format(nx, ny, nz))
        if mode not in MRC_MODES:
            raise PreviewError("unsupported MRC mode ({})".format(mode))
        dtype = np.dtype(MRC_MODES[mode]).newbyteorder(endian)

        frames = nz if max_frames is None else min(nz, max_frames)
        accumulator = np.zeros(nx * ny, dtype=np.float32)
        fh.seek(1024 + max(nsymbt, 0))
        for _ in range(frames):
            raw = np.fromfile(fh, dtype=dtype, count=nx * ny)
            if raw.size < nx * ny:
                # Truncated file: keep what we managed to read rather than fail.
                break
            accumulator += raw.astype(np.float32)
    return accumulator.reshape(ny, nx), frames, nz


def _sum_tiff(path, max_frames=None):
    """Sums the pages of a TIFF/BigTIFF stack into one float32 image.

    Pillow handles the compression (LZW here) and the BigTIFF offsets; each
    page is added straight into the accumulator so only one decoded frame is
    held at a time.
    """
    with Image.open(path) as im:
        available = getattr(im, "n_frames", 1)
        frames = available if max_frames is None else min(available, max_frames)
        accumulator = None
        for index in range(frames):
            im.seek(index)
            page = np.asarray(im, dtype=np.float32)
            if page.ndim == 3:
                # Shouldn't happen for detector data, but average any channels
                # rather than failing outright.
                page = page.mean(axis=2)
            if accumulator is None:
                accumulator = np.zeros(page.shape, dtype=np.float32)
            elif page.shape != accumulator.shape:
                break  # ragged stack: keep what lined up
            accumulator += page
    if accumulator is None:
        raise PreviewError("TIFF contains no readable pages")
    return accumulator, frames, available


def _bin_image(image):
    """Integer-bins the image down so its longest edge fits MAX_PREVIEW_EDGE."""
    height, width = image.shape
    factor = max(1, -(-max(height, width) // MAX_PREVIEW_EDGE))
    if factor == 1:
        return image
    # Crop to a whole number of bins before reshaping.
    height -= height % factor
    width -= width % factor
    cropped = image[:height, :width]
    return cropped.reshape(height // factor, factor, width // factor, factor).mean(axis=(1, 3))


def _to_grayscale_bytes(image):
    """Percentile-clips and rescales to 8-bit."""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise PreviewError("image contains no finite values")
    low, high = np.percentile(finite, CLIP_PERCENTILES)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(finite.min()), float(finite.max())
        if high <= low:
            high = low + 1.0
    scaled = (np.clip(image, low, high) - low) / (high - low)
    return (scaled * 255.0).astype(np.uint8)


def _encode_png(gray):
    """Minimal 8-bit greyscale PNG. Each row is prefixed with filter byte 0."""
    height, width = gray.shape
    raw = b"".join(b"\x00" + gray[y].tobytes() for y in range(height))

    def chunk(kind, payload):
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def render_movie_preview(path, max_frames=None):
    """Returns (png_bytes, {width, height, frames_summed, frames_total}).

    Raises PreviewError for anything we can't render.
    """
    suffix = Path(path).suffix.lower()
    if suffix not in PREVIEWABLE_EXTENSIONS:
        raise PreviewError("no preview for {} files yet".format(suffix or "these"))
    if max_frames is None:
        max_frames = MAX_PREVIEW_FRAMES
    try:
        if suffix in MRC_PREVIEW_EXTENSIONS:
            summed, frames, total = _sum_mrc(path, max_frames=max_frames)
        else:
            summed, frames, total = _sum_tiff(path, max_frames=max_frames)
        binned = _bin_image(summed)
        gray = _to_grayscale_bytes(binned)
        png = _encode_png(gray)
    except PreviewError:
        raise
    except OSError as exc:
        raise PreviewError("could not read file: {}".format(exc))
    except Exception as exc:  # noqa: BLE001 - a bad file shouldn't 500
        raise PreviewError("could not render preview: {}".format(exc))
    return png, {
        "width": gray.shape[1],
        "height": gray.shape[0],
        "frames_summed": frames,
        "frames_total": total,
    }
