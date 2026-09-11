"""
Renders a viewable PNG preview of a movie asset.

Movies are summed rather than shown a frame at a time. A single frame of
counting-mode data is almost pure shot noise -- at the doses these are taken
at, one frame of the sample MRC here is visually indistinguishable from
static -- so the sum is the only thing worth looking at without alignment.

MRC/MRCS and TIFF (including BigTIFF) are supported. EER is deliberately
out of scope and there is no plan to add it: it uses Thermo's electron-event
encoding, which is a decoder of its own, and a single EER frame holds ~0.008
e/pixel, so a useful image would mean decoding most of the file. EER movies
still import and read their headers normally -- it's only the preview that
they don't get. `can_preview()` is what the API and the Display button both
use to tell what's renderable.

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


# Bumped whenever the rendering changes in a way a cached PNG would hide --
# the preview ETags include it, so browsers refetch after such a change.
# 2: rows written bottom-up, as cisTEM displays them.
RENDER_VERSION = 2


def can_preview(path):
    """Whether this file's format is one we render. EER deliberately isn't."""
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


def _bin_factor(shape):
    """The integer bin that brings the longest edge within MAX_PREVIEW_EDGE."""
    height, width = shape
    return max(1, -(-max(height, width) // MAX_PREVIEW_EDGE))


def taper_edges(image):
    """Image::TaperEdges() in 2D: along each axis, the mean of the first and
    last N/30 pixels of every line is taken, the two means' deviations from
    their average smoothed by a 3-point running average along the edge, and
    a ramp of that deviation (full at the edge, zero N/30 pixels in) is
    subtracted at each end -- so the picture wraps around without a step,
    which is what keeps the high-pass below from ringing off the borders.
    """
    out = np.array(image, dtype=np.float32, copy=True)
    for axis in (1, 0):   # x edges, then y edges, as the C++ loops dimension 1 then 2
        n = out.shape[axis]
        width = n // 30
        if width < 1 or n < 2 * width:
            continue
        lines = out if axis == 1 else out.T   # rows run along `axis`
        start = lines[:, :width].mean(axis=1)
        finish = lines[:, n - width:].mean(axis=1)
        mid = 0.5 * (start + finish)
        start, finish = start - mid, finish - mid
        # 3-point running average along the edge, edges of the edge averaged over what exists.
        pad_s = np.pad(start, 1, mode="edge"); pad_f = np.pad(finish, 1, mode="edge")
        counts = np.full(start.shape, 3.0); counts[0] = counts[-1] = 2.0
        sm_s = (pad_s[:-2] + pad_s[1:-1] + pad_s[2:] - np.where(np.arange(start.size) == 0, pad_s[0], 0) - np.where(np.arange(start.size) == start.size - 1, pad_s[-1], 0)) / counts
        sm_f = (pad_f[:-2] + pad_f[1:-1] + pad_f[2:] - np.where(np.arange(finish.size) == 0, pad_f[0], 0) - np.where(np.arange(finish.size) == finish.size - 1, pad_f[-1], 0)) / counts
        ramp = (width - np.arange(width)) / float(width)          # 1 at the edge pixel, 1/width at the inner end
        lines[:, :width] -= sm_s[:, None] * ramp[None, :]
        lines[:, n - width:] -= sm_f[:, None] * ramp[None, ::-1]
    return out


def _fourier_radius(shape):
    """|f| in cycles per pixel for numpy's rfft2 layout of a `shape` image."""
    height, width = shape
    fy = np.fft.fftfreq(height).reshape(-1, 1)
    fx = np.fft.rfftfreq(width).reshape(1, -1)
    return np.sqrt(fx * fx + fy * fy)


def high_pass_weight(shape):
    """PickingBitmapPanel's High-pass: CosineMask(r, 2r, invert) in Fourier
    space with r = 8 / width cycles per pixel -- so the mask radius proper is
    r - edge/2 = 0, and each component is scaled by 1 - (1 + cos(pi f / 2r)) / 2
    out to 2r, untouched beyond, the DC term removed. Takes out the density
    ramps across a micrograph that would otherwise set the grey range.
    """
    radius = 8.0 / float(shape[1])
    edge = 2.0 * radius
    f = _fourier_radius(shape)
    weight = np.ones(f.shape, dtype=np.float32)
    inside = f <= edge
    weight[inside] = 1.0 - (1.0 + np.cos(np.pi * f[inside] / edge)) / 2.0
    weight[f <= 0.0] = 0.0
    return weight


def filter_preview(image, pixel_size=None, lowpass_a=None, highpass=False):
    """UpdateImageInBitmap()'s filters on the binned picture, in its order:
    taper and high-pass first, then the Gaussian low-pass, one transform pair."""
    if not highpass and not lowpass_a:
        return image
    work = taper_edges(image) if highpass else np.asarray(image, dtype=np.float32)
    weight = np.ones((work.shape[0], work.shape[1] // 2 + 1), dtype=np.float32)
    if highpass:
        weight *= high_pass_weight(work.shape)
    if lowpass_a and pixel_size and pixel_size > 0 and lowpass_a > 0:
        sigma = (float(pixel_size) / float(lowpass_a)) * np.sqrt(2.0)
        f = _fourier_radius(work.shape)
        weight *= np.exp(-(f * f) / (2.0 * sigma * sigma))
    return np.fft.irfft2(np.fft.rfft2(work) * weight, s=work.shape).astype(np.float32)


def gaussian_low_pass(image, pixel_size, resolution_a):
    """PickingBitmapPanel::UpdateImageInBitmap()'s Low-pass: cisTEM's
    Image::GaussianLowPassFilter(radius * sqrt(2)) with radius = pixel size /
    resolution -- each Fourier component scaled by exp(-f^2 / (2 sigma^2)),
    f in cycles per pixel of the image given (the binned preview here, so
    the pixel size is the binned one). The mean is untouched.
    """
    if not pixel_size or pixel_size <= 0 or not resolution_a or resolution_a <= 0:
        return image
    sigma = (float(pixel_size) / float(resolution_a)) * np.sqrt(2.0)
    height, width = image.shape
    fy = np.fft.fftfreq(height).reshape(-1, 1)
    fx = np.fft.rfftfreq(width).reshape(1, -1)
    weight = np.exp(-(fx * fx + fy * fy) / (2.0 * sigma * sigma))
    spectrum = np.fft.rfft2(image.astype(np.float32))
    return np.fft.irfft2(spectrum * weight, s=image.shape).astype(np.float32)


def _bin_image(image):
    """Integer-bins the image down so its longest edge fits MAX_PREVIEW_EDGE."""
    height, width = image.shape
    factor = _bin_factor(image.shape)
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
    """Minimal 8-bit greyscale PNG. Each row is prefixed with filter byte 0.

    Rows are written bottom-up: an MRC's first row is the *bottom* of the
    image (y increases upward, as cisTEM draws it), while a PNG's first row
    is its top. Writing them in reverse shows the image the way cisTEM
    does -- and the way coordinates from its programs expect (a pick's y is
    measured from the bottom, a CTF diagnostic's fit sits lower-left)."""
    height, width = gray.shape
    raw = b"".join(b"\x00" + gray[y].tobytes() for y in range(height - 1, -1, -1))

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


def render_image_preview(path, lowpass_a=None, pixel_size=None, highpass=False):
    """Returns (png_bytes, {width, height}) for an already-averaged image.
    `lowpass_a`, with the file's `pixel_size`, low-pass filters the binned
    picture to that resolution and `highpass` removes its density ramps
    (the Find Particles panels' Low-pass and High-pass boxes).

    Same binning and contrast stretch as a movie preview, but reading only
    the first slice: an image asset is a single micrograph, and where the
    file does happen to hold more than one section (an .mrcs that was
    imported as an image), section 1 is the one the asset refers to --
    POSITION_IN_STACK is 1 for every imported image. Summing them the way a
    movie preview does would blur unrelated exposures together.
    """
    png, meta = render_movie_preview(path, max_frames=1, lowpass_a=lowpass_a, pixel_size=pixel_size, highpass=highpass)
    return png, {"width": meta["width"], "height": meta["height"]}


def render_movie_preview(path, max_frames=None, lowpass_a=None, pixel_size=None, highpass=False):
    """Returns (png_bytes, {width, height, frames_summed, frames_total}).

    Raises PreviewError for anything we can't render.
    """
    suffix = Path(path).suffix.lower()
    if suffix not in PREVIEWABLE_EXTENSIONS:
        raise PreviewError("previews aren't supported for {} files".format(suffix or "these"))
    if max_frames is None:
        max_frames = MAX_PREVIEW_FRAMES
    try:
        if suffix in MRC_PREVIEW_EXTENSIONS:
            summed, frames, total = _sum_mrc(path, max_frames=max_frames)
        else:
            summed, frames, total = _sum_tiff(path, max_frames=max_frames)
        binned = _bin_image(summed)
        if lowpass_a or highpass:
            binned = filter_preview(binned, (pixel_size or 0) * _bin_factor(summed.shape), lowpass_a, highpass)
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
