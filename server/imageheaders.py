"""
Reads image dimensions and frame counts out of movie file headers.

Only headers are parsed -- never pixel data -- so this stays cheap even for
multi-gigabyte movies. Pure stdlib on purpose: the one runtime dependency
this project has is Flask (see requirements.txt), and the formats below are
simple enough that pulling in Pillow/tifffile/mrcfile would cost more than
it saves. Pillow also wouldn't help with EER.

Supported:
  .mrc / .mrcs  MRC stacks -- NX, NY, NZ live in the fixed 1024-byte header
  .tif / .tiff  TIFF and BigTIFF -- dimensions from the first IFD, frame
                count from the length of the IFD chain
  .eer          EER is a BigTIFF container, so it parses as TIFF; what the
                numbers *mean* differs -- see read_movie_header()

Nothing here raises on a bad file: callers get HeaderError and decide. The
import route skips such files and reports them, mirroring the reference
dialog's "%s is not a valid image file, skipping".
"""

import struct
from pathlib import Path

# A movie dimension outside this range means we've misparsed (or the file is
# corrupt) -- better to report that than to store a nonsense number.
MAX_REASONABLE_DIMENSION = 262144
# Stops a malformed or hostile IFD chain from spinning forever. Well above any
# real movie: Falcon 4 EER runs are in the low thousands of raw frames.
MAX_IFD_CHAIN = 200000

MRC_EXTENSIONS = {".mrc", ".mrcs"}
TIFF_EXTENSIONS = {".tif", ".tiff"}
EER_EXTENSIONS = {".eer"}

TIFF_TAG_IMAGE_WIDTH = 256
TIFF_TAG_IMAGE_LENGTH = 257
TIFF_TYPE_SHORT = 3


class HeaderError(Exception):
    """The file could not be understood as a movie of its apparent format."""


def _sane_dimension(value):
    return isinstance(value, int) and 0 < value <= MAX_REASONABLE_DIMENSION


def _read_mrc(fh):
    """MRC's header is fixed-size and self-describing: NX/NY/NZ as int32 at
    bytes 0-11, with byte order given by the machine stamp at byte 212. NZ is
    the section count, i.e. the number of frames in a movie stack -- so unlike
    TIFF it costs nothing extra to know.
    """
    head = fh.read(1024)
    if len(head) < 1024:
        raise HeaderError("file is shorter than an MRC header")
    # 0x1111 big-endian, 0x4444 little-endian. Anything else (including older
    # files that left it blank) we treat as little-endian, which is universal
    # in practice.
    endian = ">" if head[212:214] == b"\x11\x11" else "<"
    nx, ny, nz = struct.unpack_from(endian + "iii", head, 0)
    if not (_sane_dimension(nx) and _sane_dimension(ny)):
        raise HeaderError("implausible MRC dimensions ({} x {})".format(nx, ny))
    if not _sane_dimension(nz):
        raise HeaderError("implausible MRC section count ({})".format(nz))
    return {"x_size": nx, "y_size": ny, "number_of_frames": nz}


def _read_tiff(fh, count_frames=True):
    """Walks the IFD chain. Dimensions come from the first IFD; the frame
    count is how many IFDs there are, which means touching every one -- the
    expensive part, and why callers can ask for count_frames=False (cf.
    skip_full_check_of_tiff_movies in the reference dialog).
    """
    header = fh.read(8)
    if len(header) < 8:
        raise HeaderError("file is too short to be a TIFF")
    if header[:2] == b"II":
        endian = "<"
    elif header[:2] == b"MM":
        endian = ">"
    else:
        raise HeaderError("not a TIFF (bad byte-order mark)")

    version = struct.unpack_from(endian + "H", header, 2)[0]
    if version == 42:
        is_big = False
        entry_size = 12
        next_ifd = struct.unpack_from(endian + "I", header, 4)[0]
    elif version == 43:
        # BigTIFF: 8-byte offsets, so movies can exceed 4GB.
        is_big = True
        entry_size = 20
        offset_size = struct.unpack_from(endian + "H", header, 4)[0]
        if offset_size != 8:
            raise HeaderError("unsupported BigTIFF offset size ({})".format(offset_size))
        raw = fh.read(8)
        if len(raw) < 8:
            raise HeaderError("truncated BigTIFF header")
        next_ifd = struct.unpack(endian + "Q", raw)[0]
    else:
        raise HeaderError("unsupported TIFF version ({})".format(version))

    offset_fmt = endian + ("Q" if is_big else "I")
    offset_bytes = 8 if is_big else 4
    count_fmt = endian + ("Q" if is_big else "H")
    count_bytes = 8 if is_big else 2

    width = height = None
    frames = 0
    seen_offsets = set()

    while next_ifd and frames < MAX_IFD_CHAIN:
        if next_ifd in seen_offsets:
            raise HeaderError("TIFF IFD chain loops back on itself")
        seen_offsets.add(next_ifd)

        fh.seek(next_ifd)
        raw = fh.read(count_bytes)
        if len(raw) < count_bytes:
            raise HeaderError("truncated TIFF IFD")
        entry_count = struct.unpack(count_fmt, raw)[0]

        if frames == 0:
            entries = fh.read(entry_size * entry_count)
            if len(entries) < entry_size * entry_count:
                raise HeaderError("truncated TIFF IFD entries")
            for i in range(entry_count):
                base = i * entry_size
                tag, field_type = struct.unpack_from(endian + "HH", entries, base)
                if tag not in (TIFF_TAG_IMAGE_WIDTH, TIFF_TAG_IMAGE_LENGTH):
                    continue
                # The value sits inline in the entry when it fits, which it
                # always does for these two scalar tags.
                if is_big:
                    value = struct.unpack_from(endian + "Q", entries, base + 12)[0]
                else:
                    value_fmt = endian + ("H" if field_type == TIFF_TYPE_SHORT else "I")
                    value = struct.unpack_from(value_fmt, entries, base + 8)[0]
                if tag == TIFF_TAG_IMAGE_WIDTH:
                    width = value
                else:
                    height = value
            if not (_sane_dimension(width) and _sane_dimension(height)):
                raise HeaderError("implausible TIFF dimensions ({} x {})".format(width, height))
            if not count_frames:
                return {"x_size": width, "y_size": height, "number_of_frames": None}
        else:
            fh.seek(entry_size * entry_count, 1)

        frames += 1
        raw = fh.read(offset_bytes)
        if len(raw) < offset_bytes:
            # A truncated trailing pointer still leaves the frames we counted.
            break
        next_ifd = struct.unpack(offset_fmt, raw)[0]

    return {"x_size": width, "y_size": height, "number_of_frames": frames}


def _to_rendered_eer(raw, super_res_factor, frames_per_image):
    """EER files describe the physical detector -- 4096x4096, and one IFD per
    raw detector frame -- but what gets processed is the *rendered* movie:
    super-sampled by the super-res factor, and with raw frames averaged in
    groups. Storing the rendered numbers is what keeps
    total dose = frames x dose-per-frame true, given the import dialog asks
    for pixel size "post EER sampling" and exposure "post EER avg.".

    Verified against the metadata Thermo embeds in these files (tag 65001):
    its numberOfFrames matches the IFD count exactly, and sensorImageWidth /
    sensorImageHeight match the TIFF dimensions.
    """
    factor = super_res_factor if super_res_factor and super_res_factor > 0 else 1
    group = frames_per_image if frames_per_image and frames_per_image > 0 else 1
    rendered = dict(raw)
    rendered["x_size"] = raw["x_size"] * factor
    rendered["y_size"] = raw["y_size"] * factor
    if raw["number_of_frames"] is not None:
        # A trailing partial group isn't a whole rendered frame, so it's dropped.
        rendered["number_of_frames"] = raw["number_of_frames"] // group
    return rendered


def read_image_header(path):
    """Returns {x_size, y_size} for one already-averaged image (micrograph).

    No frame count: an image asset is a single 2D micrograph, so there is
    nothing to count and never a reason to walk a TIFF's IFD chain. EER is
    rejected outright rather than parsed -- it is a raw movie container by
    construction, so an .eer file here is a mistake worth reporting, not a
    single image.

    Raises HeaderError if the file can't be read or doesn't parse.
    """
    suffix = Path(path).suffix.lower()
    if suffix in EER_EXTENSIONS:
        raise HeaderError("EER files are movies, not images")
    header = read_movie_header(path, count_frames=False)
    return {"x_size": header["x_size"], "y_size": header["y_size"]}


def read_movie_header(path, count_frames=True, eer_super_res_factor=1, eer_frames_per_image=1):
    """Returns {x_size, y_size, number_of_frames} for one movie file, picking
    the parser by extension. number_of_frames is None when it wasn't counted.

    For EER, the returned values are the rendered ones -- see
    _to_rendered_eer(); the EER arguments are ignored for other formats.

    Raises HeaderError if the file can't be read or doesn't parse.
    """
    suffix = Path(path).suffix.lower()
    if suffix not in MRC_EXTENSIONS and suffix not in TIFF_EXTENSIONS and suffix not in EER_EXTENSIONS:
        raise HeaderError("unsupported file type ({})".format(suffix or "no extension"))
    try:
        with open(path, "rb") as fh:
            if suffix in MRC_EXTENSIONS:
                # NZ is in the fixed header, so count_frames costs nothing here.
                return _read_mrc(fh)
            raw = _read_tiff(fh, count_frames=count_frames)
            if suffix in EER_EXTENSIONS:
                return _to_rendered_eer(raw, eer_super_res_factor, eer_frames_per_image)
            return raw
    except HeaderError:
        raise
    except OSError as exc:
        raise HeaderError("could not read file: {}".format(exc))
    except Exception as exc:  # noqa: BLE001 - a malformed file shouldn't 500
        raise HeaderError("could not parse header: {}".format(exc))
