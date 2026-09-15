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
from datetime import datetime
from pathlib import Path

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
