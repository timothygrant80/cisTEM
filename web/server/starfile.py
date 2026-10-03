"""cisTEM star files (cisTEMParameters::WriteTocisTEMStarFile / cisTEMStarFileReader).

Every column cisTEM knows, with the label its reader matches on and the
printf format its writer uses, in the order the writer lays them out.
write_star() takes the subset of keys a particular file carries -- a
classification writes one set (Classification::WritecisTEMStarFile), a
refinement another (Refinement::WriteSingleClasscisTEMStarFile) -- and
read_star() returns whichever columns a file has, so the output star files
refine2d and refine3d write can be read back without knowing their exact
column set in advance. Unknown labels are skipped, missing keys are 0.

classification.py has its own smaller copy of this for the 2D case; this is
the general one the 3D code uses.
"""
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np

# key, label, printf format, python type
COLUMNS = (
    ("position_in_stack", "_cisTEMPositionInStack", "%8d", int),
    ("psi", "_cisTEMAnglePsi", "%7.2f", float),
    ("theta", "_cisTEMAngleTheta", "%7.2f", float),
    ("phi", "_cisTEMAnglePhi", "%7.2f", float),
    ("x_shift", "_cisTEMXShift", "%9.2f", float),
    ("y_shift", "_cisTEMYShift", "%9.2f", float),
    ("defocus_1", "_cisTEMDefocus1", "%8.1f", float),
    ("defocus_2", "_cisTEMDefocus2", "%8.1f", float),
    ("defocus_angle", "_cisTEMDefocusAngle", "%7.2f", float),
    ("phase_shift", "_cisTEMPhaseShift", "%7.2f", float),
    ("image_is_active", "_cisTEMImageActivity", "%5d", int),
    ("occupancy", "_cisTEMOccupancy", "%7.2f", float),
    ("logp", "_cisTEMLogP", "%9d", float),
    ("sigma", "_cisTEMSigma", "%10.4f", float),
    ("score", "_cisTEMScore", "%7.2f", float),
    ("score_change", "_cisTEMScoreChange", "%7.2f", float),
    ("pixel_size", "_cisTEMPixelSize", "%8.5f", float),
    ("voltage", "_cisTEMMicroscopeVoltagekV", "%7.2f", float),
    ("cs", "_cisTEMMicroscopeCsMM", "%7.2f", float),
    ("amplitude_contrast", "_cisTEMAmplitudeContrast", "%7.4f", float),
    ("beam_tilt_x", "_cisTEMBeamTiltX", "%7.3f", float),
    ("beam_tilt_y", "_cisTEMBeamTiltY", "%7.3f", float),
    ("image_shift_x", "_cisTEMImageShiftX", "%7.3f", float),
    ("image_shift_y", "_cisTEMImageShiftY", "%7.3f", float),
    ("best_2d_class", "_cisTEMBest2DClass", "%5d", int),
    ("beam_tilt_group", "_cisTEMBeamTiltGroup", "%5d", int),
    ("stack_filename", "_cisTEMStackFilename", "%50s", str),
    ("original_image_filename", "_cisTEMOriginalImageFilename", "%50s", str),
    ("reference_3d_filename", "_cisTEMReference3DFilename", "%50s", str),
    ("particle_group", "_cisTEMParticleGroup", "%8d", int),
    ("assigned_subset", "_cisTEMAssignedSubset", "%8d", int),
    ("pre_exposure", "_cisTEMPreExposure", "%7.2f", float),
    ("total_exposure", "_cisTEMTotalExposure", "%7.2f", float),
    ("original_x_position", "_cisTEMOriginalXPosition", "%8.2f", float),
    ("original_y_position", "_cisTEMOriginalYPosition", "%8.2f", float),
)
_BY_KEY = {c[0]: c for c in COLUMNS}
_LABEL_TO_KEY = {c[1]: c[0] for c in COLUMNS}
_ORDER = {c[0]: i for i, c in enumerate(COLUMNS)}

# Refinement::WriteSingleClasscisTEMStarFile()'s column set.
REFINEMENT_KEYS = ("position_in_stack", "image_is_active", "psi", "theta", "phi", "x_shift", "y_shift", "defocus_1", "defocus_2",
                   "defocus_angle", "phase_shift", "occupancy", "logp", "sigma", "score", "pixel_size", "voltage", "cs",
                   "amplitude_contrast", "beam_tilt_x", "beam_tilt_y", "image_shift_x", "image_shift_y", "assigned_subset")

# cisTEM's binary parameter file (cisTEMParameters::WriteTocisTEMBinaryFile /
# cisTEMStarFileReader::ReadBinaryFile, the ".cistem" extension): int32 column
# count, int32 line count, then per column its int64 bitmask identifier
# (cistem_parameters.h) and a uint8 fundamental_type (constants.h), then the
# records, one line after another, each column in header order as a 4-byte int,
# unsigned int or float, or an int32 length followed by that many bytes for a
# variable-length string. Columns are written in COLUMNS order, which is also
# the order the C++ writer uses. Reading and writing a million lines takes
# tens of milliseconds where the text form takes seconds, which is why the
# drivers exchange these with the programs and keep them in scratch.
_BITMASK = {
    "position_in_stack": 1, "image_is_active": 2, "psi": 4, "x_shift": 8, "y_shift": 16, "defocus_1": 32, "defocus_2": 64, "defocus_angle": 128,
    "phase_shift": 256, "occupancy": 512, "logp": 1024, "sigma": 2048, "score": 4096, "score_change": 8192, "pixel_size": 16384,
    "voltage": 32768, "cs": 65536, "amplitude_contrast": 131072, "beam_tilt_x": 262144, "beam_tilt_y": 524288, "image_shift_x": 1048576,
    "image_shift_y": 2097152, "theta": 4194304, "phi": 8388608, "stack_filename": 16777216, "original_image_filename": 33554432,
    "reference_3d_filename": 67108864, "best_2d_class": 134217728, "beam_tilt_group": 268435456, "particle_group": 536870912,
    "pre_exposure": 1073741824, "total_exposure": 2147483648, "assigned_subset": 4294967296, "original_x_position": 8589934592,
    "original_y_position": 17179869184,
}
_KEY_OF_BITMASK = {v: k for k, v in _BITMASK.items()}
# fundamental_type::Enum: none, text, integer, float, bool, long, double, char, variable_length, integer_unsigned
_TYPE_INTEGER, _TYPE_FLOAT, _TYPE_LONG, _TYPE_DOUBLE, _TYPE_VARIABLE, _TYPE_UNSIGNED = 2, 3, 5, 6, 8, 9
_BINARY_TYPE = {k: (_TYPE_UNSIGNED if k == "position_in_stack" else _TYPE_INTEGER if c[3] is int else _TYPE_VARIABLE if c[3] is str else _TYPE_FLOAT)
                for k, c in _BY_KEY.items()}
_NUMPY_OF_TYPE = {_TYPE_INTEGER: "<i4", _TYPE_UNSIGNED: "<u4", _TYPE_FLOAT: "<f4", _TYPE_LONG: "<i8", _TYPE_DOUBLE: "<f8"}
BINARY_EXTENSION = ".cistem"


def is_binary_path(path):
    return str(path).lower().endswith(BINARY_EXTENSION)


def table_dtype(keys):
    """The structured dtype of a parameter table with these columns, in cisTEM's column order (no string columns)."""
    keys = sorted(set(keys), key=lambda k: _ORDER[k])
    for k in keys:
        if _BY_KEY[k][3] is str:
            raise ValueError("string column {} has no fixed-width binary form".format(k))
    return np.dtype([(k, _NUMPY_OF_TYPE[_BINARY_TYPE[k]]) for k in keys])


def rows_to_table(rows, keys=REFINEMENT_KEYS):
    """Dict rows to a structured array with these columns (missing values 0)."""
    dt = table_dtype(keys)
    table = np.zeros(len(rows), dtype=dt)
    for k in dt.names:
        default = 0
        table[k] = [r.get(k, default) for r in rows]
    return table


def table_to_rows(table):
    """A structured array back to dict rows (ints as int, floats as float)."""
    names = table.dtype.names
    casts = [int if table.dtype[n].kind in "iu" else float for n in names]
    columns = [table[n].tolist() for n in names]
    return [{n: c(v) for n, c, v in zip(names, casts, values)} for values in zip(*columns)]


def write_cistem_binary(path, rows_or_table, keys=REFINEMENT_KEYS):
    """Write cisTEM's binary parameter file: dict rows or a structured array, the columns in cisTEM's order."""
    if isinstance(rows_or_table, np.ndarray):
        table = rows_or_table
        if set(table.dtype.names) != set(table_dtype(table.dtype.names).names) or list(table.dtype.names) != list(table_dtype(table.dtype.names).names):
            table = table[list(table_dtype(table.dtype.names).names)]   # cisTEM's column order
    else:
        table = rows_to_table(rows_or_table, keys)
    names = table.dtype.names
    header = [np.array([len(names), len(table)], dtype="<i4").tobytes()]
    for k in names:
        header.append(np.array([_BITMASK[k]], dtype="<i8").tobytes())
        header.append(bytes([_BINARY_TYPE[k]]))
    with open(path, "wb") as fh:
        fh.write(b"".join(header))
        fh.write(np.ascontiguousarray(table).tobytes())


def read_cistem_binary(path, as_table=False):
    """Read cisTEM's binary parameter file: dict rows (or the structured array with `as_table`).
    Unknown columns are kept under their bitmask number; string columns are read but not kept in a table."""
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) < 8:
        raise ValueError("{} is not a cisTEM binary parameter file".format(path))
    n_columns, n_lines = np.frombuffer(data[:8], dtype="<i4")
    offset = 8
    columns = []
    for _ in range(n_columns):
        bitmask = int(np.frombuffer(data[offset:offset + 8], dtype="<i8")[0])
        kind = data[offset + 8]
        offset += 9
        columns.append((_KEY_OF_BITMASK.get(bitmask, "column_{}".format(bitmask)), kind))
    if any(kind == _TYPE_VARIABLE for _, kind in columns):
        return _read_binary_with_strings(data, offset, n_lines, columns, as_table)
    dt = np.dtype([(k, _NUMPY_OF_TYPE[kind]) for k, kind in columns])
    table = np.frombuffer(data, dtype=dt, count=int(n_lines), offset=offset)
    return table.copy() if as_table else table_to_rows(table)


def _read_binary_with_strings(data, offset, n_lines, columns, as_table):
    """The slow path for files with variable-length string columns (never the refinement files)."""
    rows = []
    for _ in range(int(n_lines)):
        row = {}
        for k, kind in columns:
            if kind == _TYPE_VARIABLE:
                length = int(np.frombuffer(data[offset:offset + 4], dtype="<i4")[0]); offset += 4
                row[k] = data[offset:offset + length].decode("utf-8", "replace"); offset += length
            else:
                dtype = np.dtype(_NUMPY_OF_TYPE[kind])
                v = np.frombuffer(data[offset:offset + dtype.itemsize], dtype=dtype)[0]; offset += dtype.itemsize
                row[k] = int(v) if dtype.kind in "iu" else float(v)
        rows.append(row)
    if as_table:
        keys = [k for k, kind in columns if kind != _TYPE_VARIABLE]
        return rows_to_table(rows, keys)
    return rows


def read_params(path, as_table=False):
    """Rows (or a table) from a parameter file, binary or text by extension."""
    if is_binary_path(path):
        return read_cistem_binary(path, as_table=as_table)
    rows = read_star(path)
    if as_table:
        keys = [k for k in (rows[0].keys() if rows else REFINEMENT_KEYS) if _BY_KEY[k][3] is not str]
        return rows_to_table(rows, keys)
    return rows


def write_params(path, rows_or_table, keys=REFINEMENT_KEYS, comments=()):
    """Write a parameter file, binary or text by extension."""
    if is_binary_path(path):
        write_cistem_binary(path, rows_or_table, keys)
    else:
        rows = table_to_rows(rows_or_table) if isinstance(rows_or_table, np.ndarray) else rows_or_table
        write_star(path, rows, keys if not isinstance(rows_or_table, np.ndarray) else rows_or_table.dtype.names, comments)



def myroundint(v):
    v = float(v)
    return int(math.floor(v + 0.5)) if v >= 0 else -int(math.floor(-v + 0.5))


def write_star(path, rows, keys=REFINEMENT_KEYS, comments=()):
    """Rows (dicts) to a cisTEM star file with the given columns, in
    cisTEM's column order whatever order `keys` came in."""
    keys = sorted(set(keys), key=lambda k: _ORDER[k])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("# Written by cisTEM3 on {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        for c in comments:
            fh.write((c if c.startswith("#") else "# " + c) + "\n")
        fh.write(" \ndata_\n \nloop_\n")
        for n, k in enumerate(keys, 1):
            fh.write("{} #{}\n".format(_BY_KEY[k][1], n))
        for row in rows:
            fields = []
            for k in keys:
                _key, _label, fmt, cast = _BY_KEY[k]
                v = row.get(k, 0)
                if cast is str:
                    fields.append(fmt % ("'{}'".format(v)))
                elif fmt.endswith("d"):
                    fields.append(fmt % myroundint(v))
                else:
                    fields.append(fmt % float(v))
            fh.write(" ".join(fields) + " \n")
    return str(path)


def read_star(path):
    """Rows of a cisTEM star file as dicts keyed as in COLUMNS."""
    columns, rows = [], []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#") or s in ("data_", "loop_"):
                continue
            if s.startswith("_"):
                columns.append(_LABEL_TO_KEY.get(s.split()[0]))
                continue
            tokens = s.split()
            if len(tokens) < len(columns):
                continue
            row = {}
            for key, tok in zip(columns, tokens):
                if key is None:
                    continue
                cast = _BY_KEY[key][3]
                try:
                    if cast is str:
                        row[key] = tok.strip("'")
                    elif cast is int:
                        row[key] = int(float(tok))
                    else:
                        row[key] = float(tok)
                except ValueError:
                    row[key] = 0
            rows.append(row)
    return rows
