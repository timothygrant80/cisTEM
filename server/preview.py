"""
Renders a viewable PNG preview of a movie asset.

Movies are summed rather than shown a frame at a time. A single frame of
counting-mode data is almost pure shot noise -- at the doses these are taken
at, one frame of the sample MRC here is visually indistinguishable from
static -- so the sum is the only thing worth looking at without alignment.

Only MRC/MRCS is supported so far. TIFF needs an LZW decoder and EER needs
Thermo's electron-event decoder; both are real work, and `can_preview()` is
what the API and the Display button use to tell the difference.

PNG is written with stdlib zlib (it's a handful of CRC'd chunks), so Pillow
isn't needed. numpy is, though -- summing and rescaling frames pixel by pixel
in pure Python takes seconds per preview, versus milliseconds here.
"""

import struct
import zlib
from pathlib import Path

import numpy as np

# Longest edge of the rendered preview. Big enough to see particles, small
# enough that the PNG stays a few hundred KB.
MAX_PREVIEW_EDGE = 1024
# Contrast range, as percentiles. Cryo-EM frames have outliers (hot pixels,
# ice) that would otherwise flatten everything else to mid-grey.
CLIP_PERCENTILES = (0.5, 99.5)

PREVIEWABLE_EXTENSIONS = {".mrc", ".mrcs"}

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
    return accumulator.reshape(ny, nx), frames


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
    """Returns (png_bytes, {width, height, frames_summed}) for one movie.

    Raises PreviewError for anything we can't render.
    """
    if not can_preview(path):
        raise PreviewError(
            "no preview for {} files yet".format(Path(path).suffix.lower() or "these")
        )
    try:
        summed, frames = _sum_mrc(path, max_frames=max_frames)
        binned = _bin_image(summed)
        gray = _to_grayscale_bytes(binned)
        png = _encode_png(gray)
    except PreviewError:
        raise
    except OSError as exc:
        raise PreviewError("could not read file: {}".format(exc))
    except Exception as exc:  # noqa: BLE001 - a bad file shouldn't 500
        raise PreviewError("could not render preview: {}".format(exc))
    return png, {"width": gray.shape[1], "height": gray.shape[0], "frames_summed": frames}
