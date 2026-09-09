"""2D classification: cisTEM's ClassificationManager (src/gui/MyRefine2DPanel.cpp)
as a server-side driver.

A 2D classification is not one program run but a cycle of them, and the job
runner runs exactly one program per job. So a user-visible `class2d` job
(the row on the Jobs tab) is a *parent* that drives hidden child jobs, one
program run each, the way the GUI's ClassificationManager fires one job
package after another:

    "New Classification"           starting from an earlier classification
    ---------------------          -----------------------------------
    STARTUP  refine2d x 1  -->     (its class averages are the input)
    round 1: REFINE refine2d x N -> MERGE merge2d x 1
    round 2: REFINE ...           -> MERGE ...
    ...      (number_of_rounds times)

STARTUP (RunInitialStartJob) makes the first class averages from randomly
chosen particles: one refine2d task on the whole stack with number_of_classes
> 0 and no input averages. Each REFINE round (RunRefinementJob) splits the
stack across N = min(run profile's total processes, particles) refine2d
tasks, each of which writes the sums it would add to every class to a dump
file rather than averages, and MERGE (RunMerge2dJob) is one merge2d task
that adds the dumps up into the round's class averages. Between rounds the
high-resolution limit ramps from the start value to the finish value and,
with Auto Percent Used, the fraction of particles used follows cisTEM's
schedule (300 per class at first, at least 30% in the middle, everything for
the last five rounds).

The children are ordinary JOBS rows -- STAGE "class2d_refine2d" or
"class2d_merge2d", PARENT_JOB_ID pointing at the parent -- so the runner,
the DbSink's logging and task recording, and the restart recovery all treat
them like any job; they are only hidden from GET /jobs. The parent keeps
where it has got to in JOBS.STATE_JSON (phase, round, the classification
being built, which child is running), which is what resume() reads after a
server restart. cisTEM streams every particle's result back over its
socket (SendRefineResult); here each refine2d task is instead given a real
output star file for its particle range, and the round's results are read
from those files when the merge finishes -- the same numbers, durable
across a server restart, and one less thing to reconcile.

What a round writes is what Database::AddClassification() writes: a
CLASSIFICATION_LIST row, a CLASSIFICATION_RESULT_<id> table of per-particle
results (psi, shifts, best class -- negative when the particle sat out that
round -- sigma, logP and the imaging parameters), the classification's id
appended to REFINEMENT_PACKAGE_CLASSIFICATIONS_LIST_<package>, and the class
averages themselves in Assets/ClassAverages/class_averages_<id>.mrc. The
startup run is recorded too, as "Random Start #<id>", exactly as cisTEM
does. Input star files and refine2d dumps live under <project>/Scratch/ and
are removed when the round they served is done.

The pure pieces -- the percent-used schedule, the resolution ramp, the
particle range split, star file writing and reading, the round statistics
-- take plain values and are what server/tests/test_classification.py
covers; the driver around them needs a runner and is exercised end to end
against the real binaries instead.
"""
import json
import math
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import db
import job_protocol as jp
import preview
import refinement_packages
from stages import merge2d, refine2d

STAGE = "class2d"
CHILD_REFINE = "class2d_refine2d"
CHILD_MERGE = "class2d_merge2d"
CHILD_STAGES = (CHILD_REFINE, CHILD_MERGE)

# MyRefine2DPanel's ResetDefaults()/SetDefaults() and the panel's designer defaults.
DEFAULTS = {
    "number_of_classes": 50,
    "number_of_rounds": 20,
    "low_resolution_limit_a": 300.0,
    "high_resolution_limit_start_a": 40.0,
    "high_resolution_limit_finish_a": 8.0,
    "mask_radius_a": 100.0,
    "angular_step_deg": 15.0,
    "max_search_range_a": 100.0,
    "smoothing_factor": 1.0,
    "exclude_blank_edges": False,
    "auto_percent_used": True,
    "percent_used": 100.0,
    "auto_mask": False,
    "auto_centre": True,  # cisTEM's SetDefaults() says No; Yes here at the author's request
}


def package_defaults(largest_dimension_a):
    """MyRefine2DPanel::SetDefaults()' size-derived prefills: the mask radius
    is 0.6 x the package's largest dimension, the X/Y search range 0.33 x."""
    size = float(largest_dimension_a or 0.0)
    return {"mask_radius_a": round(size * 0.6, 2), "max_search_range_a": round(size * 0.33, 2)}

PLEASE_CREATE_PACKAGE_MESSAGE = ("Please create a refinement package (in the assets panel) in order to perform a "
                                 "2D classification.")


# ---------------------------------------------------------------------------
# The pure pieces
# ---------------------------------------------------------------------------

def default_number_of_classes(particle_count):
    """MyRefine2DPanel::SetDefaults() for a new classification: about 300
    particles per class, rounded down to 50/40/30/20/10, never fewer than 5."""
    calculated = int(particle_count) // 300
    for step in (50, 40, 30, 20, 10):
        if calculated > step:
            return step
    return 5


def startup_percent_used(number_of_classes, number_of_particles):
    """RunInitialStartJob(): 300 particles per class, capped at 100%."""
    if number_of_particles <= 0:
        return 100.0
    return min(100.0, float(number_of_classes * 300) / float(number_of_particles) * 100.0)


def high_resolution_limit(round_index, rounds, start, finish):
    """RunRefinementJobPostStarFileWrite()'s ramp: from `start` on round 0
    to `finish`, reached at round 3/4 of the way through when there are at
    least four rounds (else on the last), and held there after."""
    if rounds <= 1:
        return float(finish)
    reach = rounds * 3 // 4 if rounds >= 4 else rounds
    if round_index >= reach:
        return float(finish)
    return float(start) + float(round_index) / float(reach - 1) * (float(finish) - float(start))


def auto_percent_used(round_index, rounds, number_of_classes, number_of_particles):
    """cisTEM's Auto Percent Used schedule for round `round_index` (0-based)
    of `rounds`: fewer than 10 rounds use everything; otherwise 300 per
    class for the first 5/10/15 rounds (<20, <30, >=30 rounds), then the
    same but at least 30%, and everything for the last five."""
    per_class = min(100.0, float(number_of_classes * 300) / float(max(number_of_particles, 1)) * 100.0)
    if rounds < 10:
        return 100.0
    early = 5 if rounds < 20 else (10 if rounds < 30 else 15)
    if round_index < early:
        return per_class
    if round_index < rounds - 5:
        return max(30.0, per_class)
    return 100.0


def particle_range(job_number, number_of_jobs, number_of_particles):
    """FirstLastParticleForJob(): 1-based inclusive range for job
    `job_number` (1-based) of `number_of_jobs`, the remainder spread over
    the first jobs."""
    per_job, remainder = divmod(int(number_of_particles), int(number_of_jobs))
    if job_number - 1 < remainder:
        first = (job_number - 1) * (per_job + 1) + 1
        last = first + per_job
    else:
        first = remainder * (per_job + 1) + (job_number - 1 - remainder) * per_job + 1
        last = first + per_job - 1
    return first, last


# The columns Classification::WritecisTEMStarFile() writes, in the order
# cisTEMParameters::WriteTocisTEMStarFile() lays columns out, with its
# printf formats. Keys are this module's row dict keys; the label is what
# cisTEM's star reader matches on.
STAR_COLUMNS = (
    ("position_in_stack", "_cisTEMPositionInStack", "%8d", int),
    ("psi", "_cisTEMAnglePsi", "%7.2f", float),
    ("x_shift", "_cisTEMXShift", "%9.2f", float),
    ("y_shift", "_cisTEMYShift", "%9.2f", float),
    ("defocus_1", "_cisTEMDefocus1", "%8.1f", float),
    ("defocus_2", "_cisTEMDefocus2", "%8.1f", float),
    ("defocus_angle", "_cisTEMDefocusAngle", "%7.2f", float),
    ("phase_shift", "_cisTEMPhaseShift", "%7.2f", float),
    ("logp", "_cisTEMLogP", "%9d", float),
    ("sigma", "_cisTEMSigma", "%10.4f", float),
    ("pixel_size", "_cisTEMPixelSize", "%8.5f", float),
    ("voltage", "_cisTEMMicroscopeVoltagekV", "%7.2f", float),
    ("cs", "_cisTEMMicroscopeCsMM", "%7.2f", float),
    ("amplitude_contrast", "_cisTEMAmplitudeContrast", "%7.4f", float),
    ("beam_tilt_x", "_cisTEMBeamTiltX", "%7.3f", float),
    ("beam_tilt_y", "_cisTEMBeamTiltY", "%7.3f", float),
    ("image_shift_x", "_cisTEMImageShiftX", "%7.3f", float),
    ("image_shift_y", "_cisTEMImageShiftY", "%7.3f", float),
    ("best_2d_class", "_cisTEMBest2DClass", "%5d", int),
)
_STAR_SHORT_HEADER = ("     POS     PSI       SHX       SHY      DF1      DF2  ANGAST  PSHIFT      LogP      SIGMA    PSIZE "
                      "   VOLT      Cs    AmpC  BTILTX  BTILTY  ISHFTX  ISHFTY 2DCLS")
_STAR_LABEL_TO_KEY = {label: key for key, label, _fmt, _cast in STAR_COLUMNS}
_STAR_CASTS = {key: cast for key, _label, _fmt, cast in STAR_COLUMNS}


def empty_result(position_in_stack):
    """ClassificationResult's constructor: what an as-yet-unclassified
    particle looks like (sigma 10, class 0)."""
    return {"position_in_stack": int(position_in_stack), "psi": 0.0, "x_shift": 0.0, "y_shift": 0.0,
            "defocus_1": 0.0, "defocus_2": 0.0, "defocus_angle": 0.0, "phase_shift": 0.0, "logp": 0.0, "sigma": 10.0,
            "pixel_size": 0.0, "voltage": 0.0, "cs": 0.0, "amplitude_contrast": 0.0, "beam_tilt_x": 0.0,
            "beam_tilt_y": 0.0, "image_shift_x": 0.0, "image_shift_y": 0.0, "best_2d_class": 0}


def write_star(path, rows, comments=()):
    """A cisTEM star file with STAR_COLUMNS, byte-for-byte in the shape
    WriteTocisTEMStarFile() produces (header comment, data_/loop_ blocks,
    labelled columns, a '#' line of short names, fixed-width rows)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("# Written by cisTEM3 on {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        for comment in comments:
            fh.write(("# " + comment if not comment.startswith("#") else comment) + "\n")
        fh.write(" \ndata_\n \nloop_\n")
        for n, (_key, label, _fmt, _cast) in enumerate(STAR_COLUMNS, 1):
            fh.write("{} #{}\n".format(label, n))
        fh.write("#" + _STAR_SHORT_HEADER[1:] + " \n")
        for row in rows:
            fields = []
            for key, _label, fmt, _cast in STAR_COLUMNS:
                v = row.get(key, 0)
                if key == "logp":
                    v = int(math.floor(float(v) + 0.5)) if v >= 0 else -int(math.floor(-float(v) + 0.5))  # myroundint
                elif fmt.endswith("d"):
                    v = int(v)
                fields.append(fmt % v)
            fh.write(" ".join(fields) + " \n")
    return str(path)


def read_star(path):
    """Rows of a cisTEM star file as dicts keyed like STAR_COLUMNS (columns
    this module doesn't know are skipped, missing ones default to 0)."""
    columns = []
    rows = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#") or s in ("data_", "loop_"):
                continue
            if s.startswith("_"):
                label = s.split()[0]
                columns.append(_STAR_LABEL_TO_KEY.get(label))
                continue
            tokens = s.split()
            if len(tokens) < len(columns):
                continue
            row = {}
            for key, tok in zip(columns, tokens):
                if key is None:
                    continue
                cast = _STAR_CASTS[key]
                try:
                    row[key] = cast(float(tok)) if cast is int else float(tok)
                except ValueError:
                    row[key] = 0
            rows.append(row)
    return rows


def round_statistics(output_rows, input_rows):
    """CycleRefinement()'s per-round numbers: mean logP and sigma over the
    particles that took part (best class > 0), and the percentage of those
    whose class changed from the input classification."""
    input_class = {r["position_in_stack"]: abs(int(r.get("best_2d_class", 0))) for r in input_rows}
    active = moved = 0
    sum_logp = sum_sigma = 0.0
    for r in output_rows:
        bc = int(r.get("best_2d_class", 0))
        if bc <= 0:
            continue
        active += 1
        sum_logp += float(r.get("logp", 0.0))
        sum_sigma += float(r.get("sigma", 0.0))
        if bc != input_class.get(r["position_in_stack"], 0):
            moved += 1
    if active == 0:
        return {"active_particles": 0, "average_logp": None, "average_sigma": None, "percent_moved": None}
    return {"active_particles": active, "average_logp": sum_logp / active, "average_sigma": sum_sigma / active,
            "percent_moved": 100.0 * moved / active}


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def _num(params, key, cast):
    v = params.get(key)
    if v is None or v == "":
        return cast(DEFAULTS[key])
    try:
        return cast(v)
    except (TypeError, ValueError):
        return cast(DEFAULTS[key])


def _flag(params, key):
    v = params.get(key)
    if v is None or v == "":
        return bool(DEFAULTS[key])
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def settings_from_params(params):
    """The panel's fields -> the active_* values BeginRefinementCycle() reads."""
    s = {
        "number_of_classes": max(1, _num(params, "number_of_classes", int)),
        "number_of_rounds": max(1, _num(params, "number_of_rounds", int)),
        "low_resolution_limit": _num(params, "low_resolution_limit_a", float),
        "high_resolution_limit_start": _num(params, "high_resolution_limit_start_a", float),
        "high_resolution_limit_finish": _num(params, "high_resolution_limit_finish_a", float),
        "mask_radius": _num(params, "mask_radius_a", float),
        "angular_step": _num(params, "angular_step_deg", float),
        "max_search_range": _num(params, "max_search_range_a", float),
        "smoothing_factor": _num(params, "smoothing_factor", float),
        "exclude_blank_edges": _flag(params, "exclude_blank_edges"),
        "auto_percent_used": _flag(params, "auto_percent_used"),
        "percent_used": min(100.0, max(0.0, _num(params, "percent_used", float))),
        "auto_mask": _flag(params, "auto_mask"),
        "auto_centre": _flag(params, "auto_centre"),
    }
    return s


# ---------------------------------------------------------------------------
# Project database
# ---------------------------------------------------------------------------

RESULT_COLUMNS = ("POSITION_IN_STACK", "PSI", "XSHIFT", "YSHIFT", "BEST_CLASS", "SIGMA", "LOGP", "PIXEL_SIZE", "VOLTAGE",
                  "CS", "AMPLITUDE_CONTRAST", "DEFOCUS_1", "DEFOCUS_2", "DEFOCUS_ANGLE", "PHASE_SHIFT", "BEAM_TILT_X",
                  "BEAM_TILT_Y", "IMAGE_SHIFT_X", "IMAGE_SHIFT_Y")
_RESULT_KEYS = ("position_in_stack", "psi", "x_shift", "y_shift", "best_2d_class", "sigma", "logp", "pixel_size", "voltage",
                "cs", "amplitude_contrast", "defocus_1", "defocus_2", "defocus_angle", "phase_shift", "beam_tilt_x",
                "beam_tilt_y", "image_shift_x", "image_shift_y")


def results_table(classification_id):
    return "CLASSIFICATION_RESULT_{}".format(int(classification_id))


def _table_exists(conn, name):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def package_row(conn, package_id):
    return conn.execute("SELECT * FROM REFINEMENT_PACKAGE_ASSETS WHERE REFINEMENT_PACKAGE_ASSET_ID=?", (int(package_id),)).fetchone()


def package_particles(conn, package_id):
    """The package's contained particles in stack order -- the imaging
    parameters every star file this module writes takes from the package,
    as Classification::WritecisTEMStarFile() does."""
    table = "REFINEMENT_PACKAGE_CONTAINED_PARTICLES_{}".format(int(package_id))
    if not _table_exists(conn, table):
        return []
    return conn.execute("SELECT * FROM {} ORDER BY POSITION_IN_STACK".format(table)).fetchall()


def initial_rows(particles):
    """RunInitialStartJob()'s output_classification before anything ran:
    every particle unclassified, with its imaging parameters from the package."""
    rows = []
    for p in particles:
        r = empty_result(p["POSITION_IN_STACK"])
        r.update({"defocus_1": p["DEFOCUS_1"] or 0.0, "defocus_2": p["DEFOCUS_2"] or 0.0,
                  "defocus_angle": p["DEFOCUS_ANGLE"] or 0.0, "phase_shift": p["PHASE_SHIFT"] or 0.0,
                  "pixel_size": p["PIXEL_SIZE"] or 0.0, "voltage": p["MICROSCOPE_VOLTAGE"] or 0.0,
                  "cs": p["SPHERICAL_ABERRATION"] or 0.0, "amplitude_contrast": p["AMPLITUDE_CONTRAST"] or 0.0})
        rows.append(r)
    return rows


def classification_rows(conn, classification_id, particles):
    """A stored classification as star rows: its per-particle results with
    the defocus values refreshed from the package, as WritecisTEMStarFile()
    does (so a later CTF refinement reaches the next round)."""
    table = results_table(classification_id)
    if not _table_exists(conn, table):
        raise ValueError("classification {} has no results table".format(classification_id))
    by_pos = {p["POSITION_IN_STACK"]: p for p in particles}
    rows = []
    for r in conn.execute("SELECT * FROM {} ORDER BY POSITION_IN_STACK".format(table)).fetchall():
        row = {k: (r[c] if r[c] is not None else 0) for k, c in zip(_RESULT_KEYS, RESULT_COLUMNS)}
        p = by_pos.get(r["POSITION_IN_STACK"])
        if p is not None:
            row.update({"defocus_1": p["DEFOCUS_1"] or 0.0, "defocus_2": p["DEFOCUS_2"] or 0.0,
                        "defocus_angle": p["DEFOCUS_ANGLE"] or 0.0, "phase_shift": p["PHASE_SHIFT"] or 0.0})
            # A particle that never took part has zeros here; refine2d
            # divides by the pixel size, so give it the package's values.
            if not row["pixel_size"]:
                row.update({"pixel_size": p["PIXEL_SIZE"] or 0.0, "voltage": p["MICROSCOPE_VOLTAGE"] or 0.0,
                            "cs": p["SPHERICAL_ABERRATION"] or 0.0, "amplitude_contrast": p["AMPLITUDE_CONTRAST"] or 0.0})
        rows.append(row)
    return rows


def add_classification(conn, cls, rows):
    """Database::AddClassification() plus the package's classification list:
    the CLASSIFICATION_LIST row, CLASSIFICATION_RESULT_<id>, and the entry
    in REFINEMENT_PACKAGE_CLASSIFICATIONS_LIST_<package>."""
    cid = int(cls["classification_id"])
    package_id = int(cls["refinement_package_asset_id"])
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO CLASSIFICATION_LIST(CLASSIFICATION_ID, REFINEMENT_PACKAGE_ASSET_ID, NAME, CLASS_AVERAGE_FILE, "
            "REFINEMENT_WAS_IMPORTED_OR_GENERATED, DATETIME_OF_RUN, STARTING_CLASSIFICATION_ID, NUMBER_OF_PARTICLES, NUMBER_OF_CLASSES, "
            "LOW_RESOLUTION_LIMIT, HIGH_RESOLUTION_LIMIT, MASK_RADIUS, ANGULAR_SEARCH_STEP, SEARCH_RANGE_X, SEARCH_RANGE_Y, "
            "SMOOTHING_FACTOR, EXCLUDE_BLANK_EDGES, AUTO_PERCENT_USED, PERCENT_USED, JOB_ID) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (cid, package_id, cls["name"], cls["class_average_file"], 1 if cls.get("was_imported_or_generated") else 0,
             int(cls.get("datetime_of_run") or db.now_epoch()), int(cls.get("starting_classification_id") if cls.get("starting_classification_id") is not None else -1),
             int(cls["number_of_particles"]), int(cls["number_of_classes"]), float(cls["low_resolution_limit"]),
             float(cls["high_resolution_limit"]), float(cls["mask_radius"]), float(cls["angular_search_step"]),
             float(cls["search_range_x"]), float(cls["search_range_y"]), float(cls["smoothing_factor"]),
             1 if cls["exclude_blank_edges"] else 0, 1 if cls["auto_percent_used"] else 0, float(cls["percent_used"]),
             cls.get("job_id")))
        table = results_table(cid)
        conn.execute("DROP TABLE IF EXISTS {}".format(table))
        conn.execute("CREATE TABLE {}(POSITION_IN_STACK INTEGER PRIMARY KEY, PSI REAL, XSHIFT REAL, YSHIFT REAL, BEST_CLASS INTEGER, "
                     "SIGMA REAL, LOGP REAL, PIXEL_SIZE REAL, VOLTAGE REAL, CS REAL, AMPLITUDE_CONTRAST REAL, DEFOCUS_1 REAL, DEFOCUS_2 REAL, "
                     "DEFOCUS_ANGLE REAL, PHASE_SHIFT REAL, BEAM_TILT_X REAL, BEAM_TILT_Y REAL, IMAGE_SHIFT_X REAL, IMAGE_SHIFT_Y REAL)".format(table))
        conn.executemany("INSERT OR REPLACE INTO {} VALUES ({})".format(table, ",".join("?" * len(RESULT_COLUMNS))),
                         [tuple(r.get(k, 0) for k in _RESULT_KEYS) for r in rows])
        list_table = "REFINEMENT_PACKAGE_CLASSIFICATIONS_LIST_{}".format(package_id)
        conn.execute("CREATE TABLE IF NOT EXISTS {}(CLASSIFICATION_NUMBER INTEGER PRIMARY KEY, CLASSIFICATION_ID INTEGER)".format(list_table))
        if conn.execute("SELECT 1 FROM {} WHERE CLASSIFICATION_ID=?".format(list_table), (cid,)).fetchone() is None:
            n = conn.execute("SELECT COALESCE(MAX(CLASSIFICATION_NUMBER), 0) + 1 FROM {}".format(list_table)).fetchone()[0]
            conn.execute("INSERT INTO {} VALUES (?, ?)".format(list_table), (n, cid))


def next_classification_id(conn):
    return conn.execute("SELECT COALESCE(MAX(CLASSIFICATION_ID), 0) + 1 FROM CLASSIFICATION_LIST").fetchone()[0]


def delete_classification(conn, classification_id, remove_file=True):
    cid = int(classification_id)
    row = conn.execute("SELECT * FROM CLASSIFICATION_LIST WHERE CLASSIFICATION_ID=?", (cid,)).fetchone()
    if row is None:
        return False
    with conn:
        conn.execute("DROP TABLE IF EXISTS {}".format(results_table(cid)))
        for (sid,) in conn.execute("SELECT SELECTION_ID FROM CLASSIFICATION_SELECTION_LIST WHERE CLASSIFICATION_ID=?", (cid,)).fetchall():
            conn.execute("DROP TABLE IF EXISTS {}".format(selection_table(sid)))
        conn.execute("DELETE FROM CLASSIFICATION_SELECTION_LIST WHERE CLASSIFICATION_ID=?", (cid,))
        list_table = "REFINEMENT_PACKAGE_CLASSIFICATIONS_LIST_{}".format(row["REFINEMENT_PACKAGE_ASSET_ID"])
        if _table_exists(conn, list_table):
            conn.execute("DELETE FROM {} WHERE CLASSIFICATION_ID=?".format(list_table), (cid,))
        conn.execute("DELETE FROM CLASSIFICATION_LIST WHERE CLASSIFICATION_ID=?", (cid,))
    if remove_file and row["CLASS_AVERAGE_FILE"] and os.path.isfile(row["CLASS_AVERAGE_FILE"]):
        try:
            os.remove(row["CLASS_AVERAGE_FILE"])
        except OSError:
            pass
    return True


_LIST_SELECT = ("SELECT c.*, rp.NAME AS PACKAGE_NAME, rp.STACK_FILENAME, rp.STACK_BOX_SIZE, rp.OUTPUT_PIXEL_SIZE, j.JOB_NUMBER "
                "FROM CLASSIFICATION_LIST c "
                "LEFT JOIN REFINEMENT_PACKAGE_ASSETS rp ON rp.REFINEMENT_PACKAGE_ASSET_ID = c.REFINEMENT_PACKAGE_ASSET_ID "
                "LEFT JOIN JOBS j ON j.JOB_ID = c.JOB_ID ")


def _classification_json(row):
    d = {k.lower(): row[k] for k in row.keys()}
    d["class_average_file_exists"] = bool(row["CLASS_AVERAGE_FILE"]) and os.path.isfile(row["CLASS_AVERAGE_FILE"])
    d["was_imported_or_generated"] = bool(row["REFINEMENT_WAS_IMPORTED_OR_GENERATED"])
    d["exclude_blank_edges"] = bool(row["EXCLUDE_BLANK_EDGES"])
    d["auto_percent_used"] = bool(row["AUTO_PERCENT_USED"])
    return d


def list_classifications(conn, package_id=None):
    sql = _LIST_SELECT
    args = ()
    if package_id is not None:
        sql += "WHERE c.REFINEMENT_PACKAGE_ASSET_ID = ? "
        args = (int(package_id),)
    sql += "ORDER BY c.CLASSIFICATION_ID"
    return [_classification_json(r) for r in conn.execute(sql, args).fetchall()]


def get_classification(conn, classification_id):
    """One classification with what Refine2DResultsPanel shows about it:
    how many particles each class holds (active members) and how many sat
    the round out."""
    row = conn.execute(_LIST_SELECT + "WHERE c.CLASSIFICATION_ID = ?", (int(classification_id),)).fetchone()
    if row is None:
        return None
    d = _classification_json(row)
    n_classes = int(row["NUMBER_OF_CLASSES"] or 0)
    counts = [0] * n_classes
    inactive = 0
    table = results_table(classification_id)
    if _table_exists(conn, table):
        for bc, n in conn.execute("SELECT BEST_CLASS, COUNT(*) FROM {} GROUP BY BEST_CLASS".format(table)).fetchall():
            bc = int(bc or 0)
            if bc > 0 and bc <= n_classes:
                counts[bc - 1] += n
            elif bc <= 0:
                inactive += n
    d["class_member_counts"] = counts
    d["inactive_particles"] = inactive
    d["active_particles"] = sum(counts)
    return d


def class_members(conn, classification_id, class_number, limit=None):
    """Positions in the stack of the particles whose best class is
    `class_number`, the ones that took part in the round first."""
    table = results_table(classification_id)
    if not _table_exists(conn, table):
        return [], 0
    k = int(class_number)
    total = conn.execute("SELECT COUNT(*) FROM {} WHERE ABS(BEST_CLASS) = ?".format(table), (k,)).fetchone()[0]
    sql = "SELECT POSITION_IN_STACK, BEST_CLASS, LOGP, SIGMA, PSI, XSHIFT, YSHIFT FROM {} WHERE ABS(BEST_CLASS) = ? ORDER BY (BEST_CLASS > 0) DESC, POSITION_IN_STACK".format(table)
    if limit:
        sql += " LIMIT {}".format(int(limit))
    rows = [{"position_in_stack": r[0], "active": r[1] > 0, "logp": r[2], "sigma": r[3], "psi": r[4], "x_shift": r[5], "y_shift": r[6]}
            for r in conn.execute(sql, (k,)).fetchall()]
    return rows, total


# ---------------------------------------------------------------------------
# Class selections (Refine2DResultsPanel's selection manager +
# Database::AddClassificationSelection): a named set of class averages of
# one classification, the raw material of a refinement package built from
# class averages.
# ---------------------------------------------------------------------------

def selection_table(selection_id):
    return "CLASSIFICATION_SELECTION_{}".format(int(selection_id))


def _selection_json(conn, row):
    d = {k.lower(): row[k] for k in row.keys()}
    d["name"] = row["SELECTION_NAME"]
    table = selection_table(row["SELECTION_ID"])
    d["classes"] = sorted(r[0] for r in conn.execute("SELECT CLASS_AVERAGE_NUMBER FROM {}".format(table)).fetchall()) if _table_exists(conn, table) else []
    d["particle_count"] = selection_particle_count(conn, row["CLASSIFICATION_ID"], d["classes"])
    return d


def selection_particle_count(conn, classification_id, classes):
    table = results_table(classification_id)
    if not classes or not _table_exists(conn, table):
        return 0
    return conn.execute("SELECT COUNT(*) FROM {} WHERE BEST_CLASS IN ({})".format(table, ",".join("?" * len(classes))),
                        [int(k) for k in classes]).fetchone()[0]


def list_selections(conn, classification_id=None, package_id=None):
    sql = "SELECT * FROM CLASSIFICATION_SELECTION_LIST"
    args, where = [], []
    if classification_id is not None:
        where.append("CLASSIFICATION_ID = ?"); args.append(int(classification_id))
    if package_id is not None:
        where.append("REFINEMENT_PACKAGE_ID = ?"); args.append(int(package_id))
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY SELECTION_ID"
    return [_selection_json(conn, r) for r in conn.execute(sql, args).fetchall()]


def get_selection(conn, selection_id):
    row = conn.execute("SELECT * FROM CLASSIFICATION_SELECTION_LIST WHERE SELECTION_ID=?", (int(selection_id),)).fetchone()
    return _selection_json(conn, row) if row is not None else None


def create_selection(conn, classification_id, name=None, classes=()):
    """OnAddButtonClick(): a "New Selection" on this classification."""
    cls = conn.execute("SELECT * FROM CLASSIFICATION_LIST WHERE CLASSIFICATION_ID=?", (int(classification_id),)).fetchone()
    if cls is None:
        raise ValueError("classification {} does not exist".format(classification_id))
    with conn:
        sid = conn.execute("SELECT COALESCE(MAX(SELECTION_ID), 0) + 1 FROM CLASSIFICATION_SELECTION_LIST").fetchone()[0]
        conn.execute("INSERT INTO CLASSIFICATION_SELECTION_LIST VALUES (?,?,?,?,?,?,?)",
                     (sid, (name or "").strip() or "New Selection", db.now_epoch(), cls["REFINEMENT_PACKAGE_ASSET_ID"],
                      cls["CLASSIFICATION_ID"], cls["NUMBER_OF_CLASSES"], 0))
        conn.execute("CREATE TABLE IF NOT EXISTS {}(CLASS_AVERAGE_NUMBER INTEGER PRIMARY KEY)".format(selection_table(sid)))
    if classes:
        set_selection_classes(conn, sid, classes)
    return sid


def set_selection_classes(conn, selection_id, classes):
    """The whole membership at once -- a click toggles one class, Clear
    empties it, Invert complements it; the page sends the result."""
    row = conn.execute("SELECT * FROM CLASSIFICATION_SELECTION_LIST WHERE SELECTION_ID=?", (int(selection_id),)).fetchone()
    if row is None:
        raise ValueError("selection {} does not exist".format(selection_id))
    n = int(row["NUMBER_OF_CLASSES"] or 0)
    wanted = sorted({int(k) for k in classes if 1 <= int(k) <= n})
    table = selection_table(selection_id)
    with conn:
        conn.execute("CREATE TABLE IF NOT EXISTS {}(CLASS_AVERAGE_NUMBER INTEGER PRIMARY KEY)".format(table))
        conn.execute("DELETE FROM {}".format(table))
        conn.executemany("INSERT INTO {} VALUES (?)".format(table), [(k,) for k in wanted])
        conn.execute("UPDATE CLASSIFICATION_SELECTION_LIST SET NUMBER_OF_SELECTIONS=? WHERE SELECTION_ID=?", (len(wanted), int(selection_id)))
    return wanted


def rename_selection(conn, selection_id, name):
    name = (name or "").strip()
    if not name:
        raise ValueError("a name is required")
    with conn:
        cur = conn.execute("UPDATE CLASSIFICATION_SELECTION_LIST SET SELECTION_NAME=? WHERE SELECTION_ID=?", (name, int(selection_id)))
    return cur.rowcount > 0


def delete_selection(conn, selection_id):
    with conn:
        cur = conn.execute("DELETE FROM CLASSIFICATION_SELECTION_LIST WHERE SELECTION_ID=?", (int(selection_id),))
        conn.execute("DROP TABLE IF EXISTS {}".format(selection_table(selection_id)))
    return cur.rowcount > 0


def selection_members(conn, selection_ids):
    """For the refinement package wizard: every particle in the selected
    classes of each selection -- (parent package id, position in stack,
    x/y shift of that classification) -- in cisTEM's order (selection by
    selection, class by class, stack order within a class). A particle whose
    best class is negative sat that round out and is not a member (cisTEM's
    Return2DClassMembers matches BEST_CLASS exactly)."""
    out = []
    for sid in selection_ids:
        sel = get_selection(conn, sid)
        if sel is None:
            raise ValueError("selection {} does not exist".format(sid))
        table = results_table(sel["classification_id"])
        if not sel["classes"] or not _table_exists(conn, table):
            continue
        rows = conn.execute("SELECT POSITION_IN_STACK, XSHIFT, YSHIFT, BEST_CLASS FROM {} WHERE BEST_CLASS IN ({}) ORDER BY BEST_CLASS, POSITION_IN_STACK".format(
            table, ",".join("?" * len(sel["classes"]))), sel["classes"]).fetchall()
        for r in rows:
            out.append({"package_id": sel["refinement_package_id"], "classification_id": sel["classification_id"], "selection_id": sel["selection_id"],
                        "position_in_stack": r[0], "x_shift": r[1] or 0.0, "y_shift": r[2] or 0.0, "best_class": r[3]})
    return out


# ---------------------------------------------------------------------------
# Pictures: a montage of class averages (ClassumDisplayPanel) or of class
# members cut from the stack (ParticleDisplayPanel)
# ---------------------------------------------------------------------------

MONTAGE_TILE = 128     # each image is binned down to at most this many pixels across
MONTAGE_GAP = 2


def _mrc_section_count(path):
    with open(path, "rb") as fh:
        head = fh.read(1024)
    if len(head) < 1024:
        raise preview.PreviewError("file is shorter than an MRC header")
    endian = ">" if head[212:214] == b"\x11\x11" else "<"
    import struct
    _nx, _ny, nz, _mode = struct.unpack_from(endian + "iiii", head, 0)
    return nz


def _tile(image):
    """One image binned to MONTAGE_TILE, contrast-stretched on its own
    (as a DisplayPanel does per image), flipped so its first row is on top."""
    h, w = image.shape
    factor = max(1, -(-max(h, w) // MONTAGE_TILE))
    if factor > 1:
        h -= h % factor
        w -= w % factor
        image = image[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
    finite = image[np.isfinite(image)]
    if finite.size == 0 or float(finite.std()) == 0.0:
        return np.full(image.shape, 128, dtype=np.uint8)
    low, high = np.percentile(finite, (0.5, 99.5))
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    scaled = (np.clip(image, low, high) - low) / (high - low)
    return (scaled * 255.0).astype(np.uint8)[::-1]


def render_montage(path, sections=None, columns=None, labels=True):
    """PNG of `sections` (1-based; all when None) of an MRC stack, tiled
    left to right, top to bottom, on a mid-grey ground. Returns (png, meta)
    with the grid geometry the page needs to map a click back to a tile."""
    if not os.path.isfile(path):
        raise preview.PreviewError("{} is missing".format(path))
    nz = _mrc_section_count(path)
    if sections is None:
        sections = list(range(1, nz + 1))
    sections = [s for s in sections if 1 <= s <= nz]
    if not sections:
        raise preview.PreviewError("nothing to show")
    tiles = [_tile(refinement_packages.read_mrc_section(path, s)) for s in sections]
    th, tw = tiles[0].shape
    n = len(tiles)
    cols = int(columns) if columns else max(1, min(n, int(math.ceil(math.sqrt(n * 1.6)))))
    rows = int(math.ceil(n / cols))
    W = cols * tw + (cols + 1) * MONTAGE_GAP
    H = rows * th + (rows + 1) * MONTAGE_GAP
    canvas = np.full((H, W), 64, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y = MONTAGE_GAP + r * (th + MONTAGE_GAP)
        x = MONTAGE_GAP + c * (tw + MONTAGE_GAP)
        canvas[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
    # _encode_png writes rows bottom-up (an MRC's first row is the bottom);
    # the canvas is already in display order, so hand it over reversed.
    png = preview._encode_png(canvas[::-1])
    meta = {"columns": cols, "rows": rows, "tile_width": tw, "tile_height": th, "gap": MONTAGE_GAP,
            "width": W, "height": H, "sections": sections}
    return png, meta


def montage_geometry(count, columns=None, box=None):
    """The same grid render_montage() would lay out, without rendering --
    for the page to draw a selection square over the picture."""
    n = int(count)
    cols = int(columns) if columns else max(1, min(n, int(math.ceil(math.sqrt(n * 1.6))))) if n else 1
    rows = int(math.ceil(n / cols)) if n else 0
    tile = min(MONTAGE_TILE, int(box)) if box else MONTAGE_TILE
    if box and int(box) > MONTAGE_TILE:
        factor = -(-int(box) // MONTAGE_TILE)
        tile = (int(box) - int(box) % factor) // factor
    return {"columns": cols, "rows": rows, "tile_width": tile, "tile_height": tile, "gap": MONTAGE_GAP,
            "width": cols * tile + (cols + 1) * MONTAGE_GAP, "height": rows * tile + (rows + 1) * MONTAGE_GAP}


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------

class Runtime:
    """What the driver needs from the server, handed over once at startup
    (configure()) so this module never imports cistem_server.

    submit_child(project_id, child_job_id, adapter, tasks, profile) -> None
        the child JOBS row exists; build the JobSpec, persist token/tasks, register with the sink, submit.
    cancel(job_id) -> bool          ask the runner to cancel a child.
    append_log(project_id, job_id, text, level="info")
    update_job(project_id, job_id, **fields)
    controller_available() -> bool
    """

    def __init__(self, submit_child, cancel, append_log, update_job):
        self.submit_child = submit_child
        self.cancel = cancel
        self.append_log = append_log
        self.update_job = update_job


_runtime = None
_lock = threading.Lock()   # one driver step at a time per process; steps are short


def configure(runtime):
    global _runtime
    _runtime = runtime


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def _load_state(conn, job_id):
    row = conn.execute("SELECT STATE_JSON FROM JOBS WHERE JOB_ID=?", (job_id,)).fetchone()
    if row is None or not row["STATE_JSON"]:
        return None
    return json.loads(row["STATE_JSON"])


def _save_state(conn, job_id, state):
    with conn:
        conn.execute("UPDATE JOBS SET STATE_JSON=? WHERE JOB_ID=?", (json.dumps(state), job_id))


def scratch_dir(project_id, job_id):
    d = db.project_dir(project_id) / "Scratch" / "class2d" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def class_average_dir(project_id):
    d = db.project_dir(project_id) / "Assets" / "ClassAverages"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _log(project_id, job_id, text, level="info"):
    _runtime.append_log(project_id, job_id, text, level=level)


def validate(conn, params):
    """What the panel's OnUpdateUI needs to have been true: a package with
    particles and a stack on disk, and -- when continuing -- a starting
    classification that belongs to it. Returns (package row, particles,
    starting classification row or None); raises ValueError."""
    package_id = params.get("refinement_package_id")
    if package_id in (None, ""):
        raise ValueError(PLEASE_CREATE_PACKAGE_MESSAGE)
    pkg = package_row(conn, package_id)
    if pkg is None:
        raise ValueError("refinement package {} does not exist".format(package_id))
    particles = package_particles(conn, package_id)
    if not particles:
        raise ValueError("refinement package {!r} contains no particles".format(pkg["NAME"]))
    if not pkg["STACK_FILENAME"] or not os.path.isfile(pkg["STACK_FILENAME"]):
        raise ValueError("the particle stack {} is missing".format(pkg["STACK_FILENAME"]))
    start = None
    start_id = params.get("starting_classification_id")
    if start_id not in (None, "", 0, "0", "new"):
        start = conn.execute("SELECT * FROM CLASSIFICATION_LIST WHERE CLASSIFICATION_ID=?", (int(start_id),)).fetchone()
        if start is None:
            raise ValueError("classification {} does not exist".format(start_id))
        if start["REFINEMENT_PACKAGE_ASSET_ID"] != pkg["REFINEMENT_PACKAGE_ASSET_ID"]:
            raise ValueError("classification {} belongs to a different refinement package".format(start_id))
        if not _table_exists(conn, results_table(start["CLASSIFICATION_ID"])):
            raise ValueError("classification {} has no per-particle results to start from".format(start_id))
        if not start["CLASS_AVERAGE_FILE"] or not os.path.isfile(start["CLASS_AVERAGE_FILE"]):
            raise ValueError("the class averages of classification {} are missing".format(start_id))
    return pkg, particles, start


def start(conn, project_id, job_id, params, profile):
    """BeginRefinementCycle(): called by the server once the parent JOBS row
    exists. Validates, records the plan in STATE_JSON and launches the
    first child. Raises ValueError with a message for the page."""
    pkg, particles, start_cls = validate(conn, params)
    if not profile or profile.get("total_jobs", 0) <= 0:
        raise ValueError("run profile {!r} has no run commands, so it can't launch anything".format((profile or {}).get("name")))
    s = settings_from_params(params)
    state = {
        "phase": None, "round": 0, "rounds": s["number_of_rounds"], "settings": s,
        "package_id": pkg["REFINEMENT_PACKAGE_ASSET_ID"], "package_name": pkg["NAME"],
        "stack_filename": pkg["STACK_FILENAME"], "pixel_size": float(pkg["OUTPUT_PIXEL_SIZE"] or particles[0]["PIXEL_SIZE"] or 1.0),
        "box_size": int(pkg["STACK_BOX_SIZE"] or 0), "invert_contrast": bool(pkg["STACK_HAS_WHITE_PROTEIN"]),
        "number_of_particles": len(particles), "profile_name": profile["name"], "profile_total_jobs": int(profile["total_jobs"]),
        "start_with_random": start_cls is None, "first_round_id": None, "input_classification_id": None,
        "min_percent_used": 0.0, "output": None, "child_job_id": None, "child_task_count": 0, "child_done": 0,
        "history": [], "startup_finished_at": None, "started_at": now_iso(),
    }
    if start_cls is not None:
        state["input_classification_id"] = start_cls["CLASSIFICATION_ID"]
        state["first_round_id"] = start_cls["CLASSIFICATION_ID"]
        state["min_percent_used"] = float(start_cls["PERCENT_USED"] or 0.0)
        # cisTEM takes the class count from the starting classification.
        s["number_of_classes"] = int(start_cls["NUMBER_OF_CLASSES"])
    with conn:
        conn.execute("UPDATE JOBS SET STATUS='running', STARTED_AT=?, PROGRESS=0 WHERE JOB_ID=?", (now_iso(), job_id))
    _log(project_id, job_id, "2D classification of {!r} ({} particles, {} classes, {} round{}, profile {!r})".format(
        pkg["NAME"], len(particles), s["number_of_classes"], s["number_of_rounds"], "" if s["number_of_rounds"] == 1 else "s", profile["name"]))
    if start_cls is None:
        _launch_startup(conn, project_id, job_id, state, particles)
    else:
        _launch_refine(conn, project_id, job_id, state, particles)
    _save_state(conn, job_id, state)
    return state


def _new_child(conn, project_id, parent_id, stage, name, parent_row):
    child_id = uuid.uuid4().hex[:10]
    with conn:
        conn.execute(
            "INSERT INTO JOBS(JOB_ID, STAGE, JOB_NUMBER, NAME, PARAMS_JSON, STATUS, PROGRESS, CREATED_AT, PARENT_JOB_ID) "
            "VALUES (?, ?, ?, ?, ?, 'queued', 0, ?, ?)",
            (child_id, stage, parent_row["JOB_NUMBER"], name, parent_row["PARAMS_JSON"], now_iso(), parent_id))
    return child_id


def _parent_row(conn, job_id):
    return conn.execute("SELECT * FROM JOBS WHERE JOB_ID=?", (job_id,)).fetchone()


def _profile(state):
    sys_conn = db.get_system_conn()
    try:
        return db.load_run_profile_by_name(sys_conn, state["profile_name"])
    finally:
        sys_conn.close()


def _refine2d_task(index, ref, values):
    A = jp.arg
    kinds = {"t": "text", "i": "int", "f": "float", "b": "bool"}
    args = [A(kinds[t], v) for t, v in zip(refine2d.ARGUMENT_TYPES, values)]
    assert len(args) == len(refine2d.ARGUMENT_NAMES)
    return {"index": index, "ref": ref, "args": args}


def _launch_startup(conn, project_id, job_id, state, particles):
    """RunInitialStartJob() + RunInitialStartJobPostStarFileWrite()."""
    s = state["settings"]
    cid = next_classification_id(conn)
    n = state["number_of_particles"]
    output = {
        "classification_id": cid, "refinement_package_asset_id": state["package_id"],
        "name": "Random Start #{}".format(cid),
        "class_average_file": str(class_average_dir(project_id) / "class_averages_{:04d}.mrc".format(cid)),
        "was_imported_or_generated": True, "datetime_of_run": db.now_epoch(), "starting_classification_id": -1,
        "number_of_particles": n, "number_of_classes": s["number_of_classes"],
        "low_resolution_limit": s["low_resolution_limit"], "high_resolution_limit": s["high_resolution_limit_start"],
        "mask_radius": s["mask_radius"], "angular_search_step": s["angular_step"],
        "search_range_x": s["max_search_range"], "search_range_y": s["max_search_range"],
        "smoothing_factor": s["smoothing_factor"], "exclude_blank_edges": s["exclude_blank_edges"],
        "auto_percent_used": s["auto_percent_used"], "percent_used": startup_percent_used(s["number_of_classes"], n),
        "job_id": job_id,
    }
    star = write_star(scratch_dir(project_id, job_id) / "classification_input_star_{}.star".format(cid), initial_rows(particles),
                      comments=["Input for Random Start #{}".format(cid)])
    values = [state["stack_filename"], star, "/dev/null", "/dev/null", output["class_average_file"],
              s["number_of_classes"], 1, n, output["percent_used"] / 100.0, state["pixel_size"], s["mask_radius"],
              output["low_resolution_limit"], output["high_resolution_limit"], s["angular_step"], s["max_search_range"],
              s["smoothing_factor"], 2, True, state["invert_contrast"], s["exclude_blank_edges"], False, "/dev/null",
              False, False, 1]
    tasks = [_refine2d_task(0, cid, values)]
    parent = _parent_row(conn, job_id)
    child_id = _new_child(conn, project_id, job_id, CHILD_REFINE, "{} · starting references".format(parent["NAME"]), parent)
    state.update({"phase": "startup", "output": output, "child_job_id": child_id, "child_task_count": 1, "child_done": 0})
    _log(project_id, job_id, "Creating initial references ({}, {:.0f}% of the particles) — child job {}".format(
        output["name"], output["percent_used"], child_id))
    _runtime.submit_child(project_id, child_id, refine2d, tasks, _profile(state))


def _launch_refine(conn, project_id, job_id, state, particles):
    """RunRefinementJob() + RunRefinementJobPostStarFileWrite()."""
    s = state["settings"]
    r = state["round"]
    rounds = state["rounds"]
    n = state["number_of_particles"]
    cid = next_classification_id(conn)
    input_id = state["input_classification_id"]
    input_cls = conn.execute("SELECT * FROM CLASSIFICATION_LIST WHERE CLASSIFICATION_ID=?", (input_id,)).fetchone()
    if input_cls is None:
        raise ValueError("input classification {} disappeared".format(input_id))
    n_classes = int(input_cls["NUMBER_OF_CLASSES"])
    high_res = high_resolution_limit(r, rounds, s["high_resolution_limit_start"], s["high_resolution_limit_finish"])
    if s["auto_percent_used"]:
        percent = auto_percent_used(r, rounds, n_classes, n)
    else:
        percent = s["percent_used"]
    percent = max(percent, float(state.get("min_percent_used") or 0.0))
    output = {
        "classification_id": cid, "refinement_package_asset_id": state["package_id"],
        "name": "Classification #{} (Start #{}, Round {})".format(cid, state["first_round_id"], r + 1),
        "class_average_file": str(class_average_dir(project_id) / "class_averages_{:04d}.mrc".format(cid)),
        "was_imported_or_generated": False, "datetime_of_run": db.now_epoch(), "starting_classification_id": input_id,
        "number_of_particles": n, "number_of_classes": n_classes,
        "low_resolution_limit": s["low_resolution_limit"], "high_resolution_limit": high_res,
        "mask_radius": s["mask_radius"], "angular_search_step": s["angular_step"],
        "search_range_x": s["max_search_range"], "search_range_y": s["max_search_range"],
        "smoothing_factor": s["smoothing_factor"], "exclude_blank_edges": s["exclude_blank_edges"],
        "auto_percent_used": s["auto_percent_used"], "percent_used": percent, "job_id": job_id,
    }
    scratch = scratch_dir(project_id, job_id)
    star = write_star(scratch / "classification_input_star_{}.star".format(cid), classification_rows(conn, input_id, particles),
                      comments=["Input for {}".format(output["name"])])
    number_of_jobs = max(1, min(int(state["profile_total_jobs"]), n))
    project_scratch = db.project_dir(project_id) / "Scratch"
    project_scratch.mkdir(parents=True, exist_ok=True)
    tasks = []
    for k in range(1, number_of_jobs + 1):
        first, last = particle_range(k, number_of_jobs, n)
        values = [state["stack_filename"], star, input_cls["CLASS_AVERAGE_FILE"],
                  str(scratch / "round_{}_{}.star".format(cid, k)), output["class_average_file"],
                  0, first, last, percent / 100.0, state["pixel_size"], s["mask_radius"],
                  output["low_resolution_limit"], high_res, s["angular_step"], s["max_search_range"],
                  s["smoothing_factor"], 2, True, state["invert_contrast"], s["exclude_blank_edges"], True,
                  str(project_scratch / "class_dump_file_{}_{}.dump".format(cid, k)),
                  s["auto_mask"], s["auto_centre"], 1]
        tasks.append(_refine2d_task(k - 1, k, values))
    parent = _parent_row(conn, job_id)
    child_id = _new_child(conn, project_id, job_id, CHILD_REFINE, "{} · round {} of {}".format(parent["NAME"], r + 1, rounds), parent)
    state.update({"phase": "refine", "output": output, "child_job_id": child_id, "child_task_count": number_of_jobs, "child_done": 0,
                  "number_of_dump_files": number_of_jobs})
    _log(project_id, job_id, "Running refinement round {} of {} ({} refine2d task{}) — child job {}".format(
        r + 1, rounds, number_of_jobs, "" if number_of_jobs == 1 else "s", child_id))
    _log(project_id, job_id, "High resolution limit: {:.1f} Å".format(high_res))
    if s["auto_percent_used"]:
        _log(project_id, job_id, "Using {:.0f}% of the particles ({} per class)".format(
            percent, int(round(n * percent / 100.0 / max(n_classes, 1)))))
    if s["auto_mask"]:
        _log(project_id, job_id, "Will automask reference class averages")
    _runtime.submit_child(project_id, child_id, refine2d, tasks, _profile(state))


def _launch_merge(conn, project_id, job_id, state):
    """RunMerge2dJob()."""
    output = state["output"]
    cid = output["classification_id"]
    n_dumps = int(state.get("number_of_dump_files") or state["child_task_count"])
    seed = str(db.project_dir(project_id) / "Scratch" / "class_dump_file_{}_.dump".format(cid))
    A = jp.arg
    tasks = [{"index": 0, "ref": cid, "args": [A("text", output["class_average_file"]), A("text", seed), A("int", n_dumps)]}]
    parent = _parent_row(conn, job_id)
    child_id = _new_child(conn, project_id, job_id, CHILD_MERGE, "{} · merge round {}".format(parent["NAME"], state["round"] + 1), parent)
    state.update({"phase": "merge", "child_job_id": child_id, "child_task_count": 1, "child_done": 0})
    _log(project_id, job_id, "Merging class averages ({} dump file{}) — child job {}".format(n_dumps, "" if n_dumps == 1 else "s", child_id))
    _runtime.submit_child(project_id, child_id, merge2d, tasks, _profile(state))


def _remove_scratch(project_id, job_id, state, dumps_only=False):
    cid = (state.get("output") or {}).get("classification_id")
    project_scratch = db.project_dir(project_id) / "Scratch"
    if cid is not None:
        for k in range(1, int(state.get("number_of_dump_files") or 0) + 1):
            p = project_scratch / "class_dump_file_{}_{}.dump".format(cid, k)
            if p.is_file():
                try:
                    p.unlink()
                except OSError:
                    pass
    if dumps_only:
        return
    d = project_scratch / "class2d" / job_id
    if d.is_dir():
        for p in d.iterdir():
            try:
                p.unlink()
            except OSError:
                pass
        try:
            d.rmdir()
        except OSError:
            pass


def _progress_percent(state):
    """Whole-run progress: the startup counts as one unit, each round as
    one, with a round nine-tenths refine and one-tenth merge."""
    had_startup = bool(state.get("start_with_random") or state.get("startup_finished_at"))
    total = float(state["rounds"]) + (1.0 if had_startup else 0.0)
    if total <= 0:
        return 0
    done = float(state["round"])
    if had_startup and state["phase"] != "startup":
        done += 1.0
    frac = float(state.get("child_done", 0)) / float(max(state.get("child_task_count", 1), 1))
    if state["phase"] in ("startup",):
        done += frac
    elif state["phase"] == "refine":
        done += 0.9 * frac
    elif state["phase"] == "merge":
        done += 0.9 + 0.1 * frac
    elif state["phase"] == "finished":
        done = total
    return max(0, min(100, int(100.0 * done / total)))


def progress_info(state):
    """What _row_to_job() reports for a running class2d parent so the page's
    time-remaining estimate works: rounds as tasks."""
    if not state:
        return {}
    total = int(state["rounds"]) + (1 if (state.get("start_with_random") or state.get("startup_finished_at")) else 0)
    finished_at = [h["finished_at"] for h in state.get("history", []) if h.get("finished_at")]
    if state.get("startup_finished_at"):
        finished_at.insert(0, state["startup_finished_at"])
    return {"task_count": total, "tasks_done": len(finished_at),
            "first_task_finished_at": finished_at[0] if finished_at else None,
            "last_task_finished_at": finished_at[-1] if finished_at else None,
            "round": state["round"], "rounds": state["rounds"], "phase": state["phase"]}


def child_progress(conn, parent_id, child_id, done_count, task_count):
    """DbSink.on_task_done for a child: moves the parent's progress bar."""
    state = _load_state(conn, parent_id)
    if not state or state.get("child_job_id") != child_id:
        return
    state["child_done"] = done_count
    state["child_task_count"] = task_count or state.get("child_task_count", 1)
    with conn:
        conn.execute("UPDATE JOBS SET STATE_JSON=?, PROGRESS=? WHERE JOB_ID=?", (json.dumps(state), _progress_percent(state), parent_id))


def child_finished(project_id, child_row, status, error=None):
    """DbSink: a child reached a terminal status. Runs the next step of the
    cycle on a worker thread (the sink is called from the runner's
    listener thread, which must not block on file I/O or a submit)."""
    threading.Thread(target=_child_finished, args=(project_id, child_row["JOB_ID"], child_row["PARENT_JOB_ID"], status, error),
                     daemon=True, name="class2d-" + child_row["PARENT_JOB_ID"]).start()


def _child_finished(project_id, child_id, parent_id, status, error):
    with _lock:
        conn = db.get_conn(project_id)
        try:
            state = _load_state(conn, parent_id)
            parent = _parent_row(conn, parent_id)
            if not state or parent is None:
                return
            if state.get("child_job_id") != child_id:
                return  # a stale child (already handled, or from before a resume)
            if parent["STATUS"] not in ("queued", "running"):
                return
            if parent["CANCEL_REQUESTED"] or status == "cancelled":
                _finish_parent(conn, project_id, parent_id, state, "cancelled", "cancelled during round {}".format(state["round"] + 1)
                               if state["phase"] != "startup" else "cancelled while creating the initial references")
                return
            if status != "completed":
                _finish_parent(conn, project_id, parent_id, state, "failed",
                               "{} run failed{}".format("refine2d" if state["phase"] != "merge" else "merge2d", ": " + error if error else ""))
                return
            try:
                _advance(conn, project_id, parent_id, state)
            except Exception as exc:  # noqa: BLE001
                _finish_parent(conn, project_id, parent_id, state, "failed", "could not continue the classification: {}".format(exc))
        finally:
            conn.close()


def _advance(conn, project_id, parent_id, state):
    """ProcessAllJobsFinished() + CycleRefinement()."""
    particles = package_particles(conn, state["package_id"])
    output = state["output"]
    if state["phase"] == "startup":
        if not os.path.isfile(output["class_average_file"]):
            raise ValueError("refine2d did not write {}".format(output["class_average_file"]))
        add_classification(conn, output, initial_rows(particles))
        state["startup_finished_at"] = now_iso()
        state["input_classification_id"] = output["classification_id"]
        state["first_round_id"] = output["classification_id"]
        state["start_with_random"] = False
        state["history"] = state.get("history", [])
        _log(project_id, parent_id, "Initial references written: {} ({})".format(output["name"], output["class_average_file"]))
        _launch_refine(conn, project_id, parent_id, state, particles)
    elif state["phase"] == "refine":
        _launch_merge(conn, project_id, parent_id, state)
    elif state["phase"] == "merge":
        if not os.path.isfile(output["class_average_file"]):
            raise ValueError("merge2d did not write {}".format(output["class_average_file"]))
        scratch = db.project_dir(project_id) / "Scratch" / "class2d" / parent_id
        cid = output["classification_id"]
        rows_by_pos = {}
        for k in range(1, int(state.get("number_of_dump_files") or 1) + 1):
            p = scratch / "round_{}_{}.star".format(cid, k)
            if not p.is_file():
                raise ValueError("refine2d task {} left no output star file ({})".format(k, p))
            for r in read_star(p):
                rows_by_pos[r["position_in_stack"]] = r
        # A particle no task wrote (there shouldn't be any) stays unclassified.
        rows = []
        for p in particles:
            r = rows_by_pos.get(p["POSITION_IN_STACK"])
            if r is None:
                r = empty_result(p["POSITION_IN_STACK"])
            rows.append(r)
        input_rows = classification_rows(conn, state["input_classification_id"], particles)
        stats = round_statistics(rows, input_rows)
        add_classification(conn, output, rows)
        _remove_scratch(project_id, parent_id, state, dumps_only=True)
        for p in scratch.glob("round_{}_*.star".format(cid)):
            try:
                p.unlink()
            except OSError:
                pass
        state["round"] += 1
        entry = dict(stats, round=state["round"], classification_id=cid, finished_at=now_iso(),
                     high_resolution_limit=output["high_resolution_limit"], percent_used=output["percent_used"])
        state.setdefault("history", []).append(entry)
        _log(project_id, parent_id, "Round {} of {} done: {} — {} particles active, mean logP {}, mean sigma {}, {} moved class".format(
            state["round"], state["rounds"], output["name"], stats["active_particles"],
            "{:.1f}".format(stats["average_logp"]) if stats["average_logp"] is not None else "n/a",
            "{:.3f}".format(stats["average_sigma"]) if stats["average_sigma"] is not None else "n/a",
            "{:.1f}%".format(stats["percent_moved"]) if stats["percent_moved"] is not None else "n/a"))
        if state["round"] < state["rounds"]:
            state["input_classification_id"] = cid
            _launch_refine(conn, project_id, parent_id, state, particles)
        else:
            state["phase"] = "finished"
            state["child_job_id"] = None
            _remove_scratch(project_id, parent_id, state)
            _finish_parent(conn, project_id, parent_id, state, "completed", None)
            return
    with conn:
        conn.execute("UPDATE JOBS SET STATE_JSON=?, PROGRESS=? WHERE JOB_ID=?", (json.dumps(state), _progress_percent(state), parent_id))


def _finish_parent(conn, project_id, parent_id, state, status, error):
    if status != "completed":
        _remove_scratch(project_id, parent_id, state)
    metrics = {"rounds_run": state.get("round", 0), "rounds_requested": state.get("rounds"),
               "classification_ids": [h["classification_id"] for h in state.get("history", [])],
               "final_classification_id": state["history"][-1]["classification_id"] if state.get("history") else None}
    cpu = conn.execute("SELECT COALESCE(SUM(json_extract(METRICS_JSON, '$.cpu_ms')), 0) FROM JOBS WHERE PARENT_JOB_ID=?", (parent_id,)).fetchone()[0]
    metrics["cpu_ms"] = cpu or 0
    with conn:
        conn.execute("UPDATE JOBS SET STATUS=?, ERROR=?, FINISHED_AT=?, PROGRESS=?, METRICS_JSON=?, STATE_JSON=? WHERE JOB_ID=?",
                     (status, error, now_iso(), 100 if status == "completed" else _progress_percent(state), json.dumps(metrics),
                      json.dumps(state), parent_id))
    if status == "completed":
        _log(project_id, parent_id, "All refinement cycles are finished! ({} classification{} written)".format(
            len(metrics["classification_ids"]), "" if len(metrics["classification_ids"]) == 1 else "s"))
    else:
        _log(project_id, parent_id, "job {}: {}".format(status, error), level="error")


def cancel(conn, project_id, parent_id):
    """The Jobs tab's Cancel on a class2d parent: flag it and stop whichever
    child is running; the parent finishes when the child reports back. A
    parent between children (or whose child is already gone) stops now."""
    state = _load_state(conn, parent_id)
    with conn:
        conn.execute("UPDATE JOBS SET CANCEL_REQUESTED=1 WHERE JOB_ID=?", (parent_id,))
    child_id = (state or {}).get("child_job_id")
    if child_id and _runtime.cancel(child_id):
        _log(project_id, parent_id, "cancel requested; stopping child job {}".format(child_id))
        return True
    if state is not None:
        _finish_parent(conn, project_id, parent_id, state, "cancelled", "cancelled")
    return False


def resume(project_id, parent_row):
    """After a server restart: the parent's children have been handed back
    to the runner by the usual recovery; if the current child had already
    finished (its status is terminal), carry on from it now."""
    conn = db.get_conn(project_id)
    try:
        state = _load_state(conn, parent_row["JOB_ID"])
        if not state:
            with conn:
                conn.execute("UPDATE JOBS SET STATUS='failed', ERROR='Server restarted before this classification recorded its plan', FINISHED_AT=? WHERE JOB_ID=?",
                             (now_iso(), parent_row["JOB_ID"]))
            return
        child_id = state.get("child_job_id")
        child = conn.execute("SELECT * FROM JOBS WHERE JOB_ID=?", (child_id,)).fetchone() if child_id else None
        if child is None:
            _finish_parent(conn, project_id, parent_row["JOB_ID"], state, "failed", "Server restarted and the running step could not be found")
            return
        _log(project_id, parent_row["JOB_ID"], "server restarted during {} (round {} of {}); waiting on child job {}".format(
            state["phase"], state["round"] + 1, state["rounds"], child_id))
        if child["STATUS"] in ("completed", "failed", "cancelled"):
            child_finished(project_id, child, child["STATUS"], child["ERROR"])
    finally:
        conn.close()


def live_result(conn, row):
    """What the Jobs tab's Latest Result panel draws for a running or
    finished class2d job: the newest class averages and the per-round
    statistics (MyRefine2DPanel's ResultDisplayPanel + PlotPanel)."""
    state = _load_state(conn, row["JOB_ID"])
    if not state:
        return None
    history = state.get("history", [])
    latest_id = history[-1]["classification_id"] if history else state.get("first_round_id")
    if latest_id is None:
        return None
    cls = conn.execute("SELECT * FROM CLASSIFICATION_LIST WHERE CLASSIFICATION_ID=?", (latest_id,)).fetchone()
    if cls is None:
        return None
    return {
        "kind": "class2d",
        "task_index": int(latest_id),
        "classification_id": int(latest_id),
        "name": cls["NAME"],
        "number_of_classes": cls["NUMBER_OF_CLASSES"],
        "class_average_file": cls["CLASS_AVERAGE_FILE"],
        "class_average_file_exists": bool(cls["CLASS_AVERAGE_FILE"]) and os.path.isfile(cls["CLASS_AVERAGE_FILE"]),
        "round": state["round"], "rounds": state["rounds"], "phase": state["phase"],
        "is_startup": not history,
        "package_name": state.get("package_name"),
        "history": history,
        "montage": montage_geometry(cls["NUMBER_OF_CLASSES"], box=state.get("box_size")),
    }
