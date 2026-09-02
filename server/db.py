"""
Per-project SQLite storage for the Cryo-EM Job Runner reference backend.

Each project is its own file at PROJECTS_ROOT/<project_id>/project.db. Table
names and columns are lifted verbatim from a real cisTEM project database
wherever this app stores the same kind of data (MASTER_SETTINGS, MOVIE_ASSETS,
MOVIE_ALIGNMENT_LIST, RUN_PROFILES, ...), so a project created here reads like
a real cisTEM project. Two deliberate deviations, noted inline below:

  - MOVIE_GROUP_MEMBERS is one junction table instead of cisTEM's per-group
    numbered tables (MOVIE_GROUP_<id>) -- that sharding exists in cisTEM for a
    desktop app accumulating thousands of groups over years; unnecessary at
    this app's scale.
  - JOBS / JOB_LOG_LINES have no cisTEM equivalent -- cisTEM tracks live job
    status via an in-memory socket manager, not its database, and only ever
    writes finished results. This app needs a durable "in progress" concept
    cisTEM's schema doesn't have.

No migration framework: SCHEMA_SQL is idempotent (CREATE TABLE IF NOT EXISTS)
and re-run on every connection open. This is a reference app with no deployed
user data -- if a column ever needs to change, add an ALTER TABLE guarded by
try/except sqlite3.OperationalError rather than reaching for a migration tool.
"""

import re
import shutil
import sqlite3
import time
import uuid
from pathlib import Path

PROJECTS_ROOT = Path(__file__).parent / "data" / "projects"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS MASTER_SETTINGS(
  NUMBER INTEGER PRIMARY KEY, PROJECT_DIRECTORY TEXT, PROJECT_NAME TEXT,
  CURRENT_VERSION INTEGER, TOTAL_CPU_HOURS REAL, TOTAL_JOBS_RUN INTEGER,
  CISTEM_VERSION_TEXT TEXT, CURRENT_WORKFLOW TEXT
);

CREATE TABLE IF NOT EXISTS MOVIE_ASSETS(
  MOVIE_ASSET_ID INTEGER PRIMARY KEY, NAME TEXT, FILENAME TEXT,
  POSITION_IN_STACK INTEGER, X_SIZE INTEGER, Y_SIZE INTEGER, NUMBER_OF_FRAMES INTEGER,
  VOLTAGE REAL, PIXEL_SIZE REAL, DOSE_PER_FRAME REAL, SPHERICAL_ABERRATION REAL,
  GAIN_FILENAME TEXT, DARK_FILENAME TEXT, OUTPUT_BINNING_FACTOR REAL,
  CORRECT_MAG_DISTORTION INTEGER, MAG_DISTORTION_ANGLE REAL,
  MAG_DISTORTION_MAJOR_SCALE REAL, MAG_DISTORTION_MINOR_SCALE REAL,
  PROTEIN_IS_WHITE INTEGER, EER_SUPER_RES_FACTOR INTEGER, EER_FRAMES_PER_IMAGE INTEGER
);

CREATE TABLE IF NOT EXISTS MOVIE_IMPORT_DEFAULTS(
  NUMBER INTEGER PRIMARY KEY, VOLTAGE REAL, SPHERICAL_ABERRATION REAL, PIXEL_SIZE REAL,
  EXPOSURE_PER_FRAME REAL, MOVIES_ARE_GAIN_CORRECTED INTEGER, GAIN_REFERENCE_FILENAME TEXT,
  MOVIES_ARE_DARK_CORRECTED INTEGER, DARK_REFERENCE_FILENAME TEXT, RESAMPLE_MOVIES INTEGER,
  DESIRED_PIXEL_SIZE REAL, CORRECT_MAG_DISTORTION INTEGER, MAG_DISTORTION_ANGLE REAL,
  MAG_DISTORTION_MAJOR_SCALE REAL, MAG_DISTORTION_MINOR_SCALE REAL, PROTEIN_IS_WHITE INTEGER,
  EER_SUPER_RES_FACTOR INTEGER, EER_FRAMES_PER_IMAGE INTEGER
);

CREATE TABLE IF NOT EXISTS MOVIE_GROUP_LIST(
  GROUP_ID INTEGER PRIMARY KEY, GROUP_NAME TEXT, LIST_ID INTEGER
);

-- Deviation from real cisTEM (see module docstring): one junction table
-- instead of a numbered table per group.
CREATE TABLE IF NOT EXISTS MOVIE_GROUP_MEMBERS(
  GROUP_ID INTEGER NOT NULL, MOVIE_ASSET_ID INTEGER NOT NULL,
  PRIMARY KEY(GROUP_ID, MOVIE_ASSET_ID)
);

CREATE TABLE IF NOT EXISTS IMAGE_ASSETS(
  IMAGE_ASSET_ID INTEGER PRIMARY KEY, NAME TEXT, FILENAME TEXT, POSITION_IN_STACK INTEGER,
  PARENT_MOVIE_ID INTEGER, ALIGNMENT_ID INTEGER, CTF_ESTIMATION_ID INTEGER,
  X_SIZE INTEGER, Y_SIZE INTEGER, PIXEL_SIZE REAL, VOLTAGE REAL,
  SPHERICAL_ABERRATION REAL, PROTEIN_IS_WHITE INTEGER
);

-- ALIGNMENT_JOB_ID is TEXT here (real cisTEM: INTEGER) to match this app's
-- existing job-id shape (uuid4().hex[:10]) instead of inventing a parallel
-- integer id scheme.
CREATE TABLE IF NOT EXISTS MOVIE_ALIGNMENT_LIST(
  ALIGNMENT_ID INTEGER PRIMARY KEY, DATETIME_OF_RUN INTEGER,
  ALIGNMENT_JOB_ID TEXT,
  MOVIE_ASSET_ID INTEGER, OUTPUT_FILE TEXT, VOLTAGE REAL, PIXEL_SIZE REAL,
  EXPOSURE_PER_FRAME REAL, PRE_EXPOSURE_AMOUNT REAL, MIN_SHIFT REAL, MAX_SHIFT REAL,
  SHOULD_DOSE_FILTER INTEGER, SHOULD_RESTORE_POWER INTEGER, TERMINATION_THRESHOLD REAL,
  MAX_ITERATIONS INTEGER, BFACTOR INTEGER, SHOULD_MASK_CENTRAL_CROSS INTEGER,
  HORIZONTAL_MASK INTEGER, VERTICAL_MASK INTEGER, SHOULD_INCLUDE_ALL_FRAMES_IN_SUM INTEGER,
  FIRST_FRAME_TO_SUM INTEGER, LAST_FRAME_TO_SUM INTEGER, FINAL_PIXEL_SIZE REAL
);

CREATE TABLE IF NOT EXISTS RUN_PROFILES(
  RUN_PROFILE_ID INTEGER PRIMARY KEY, PROFILE_NAME TEXT, MANAGER_RUN_COMMAND TEXT,
  GUI_ADDRESS TEXT, CONTROLLER_ADDRESS TEXT, COMMANDS_ID INTEGER
);

-- Phase-2 tables: schema created now for stability, not written to until
-- their stages get the same real-cisTEM-source parameter scrape Align
-- Movies already got.
CREATE TABLE IF NOT EXISTS ESTIMATED_CTF_PARAMETERS(
  CTF_ESTIMATION_ID INTEGER PRIMARY KEY, CTF_ESTIMATION_JOB_ID TEXT, DATETIME_OF_RUN INTEGER,
  IMAGE_ASSET_ID INTEGER, ESTIMATED_ON_MOVIE_FRAMES INTEGER, VOLTAGE REAL,
  SPHERICAL_ABERRATION REAL, PIXEL_SIZE REAL, AMPLITUDE_CONTRAST REAL, BOX_SIZE INTEGER,
  MIN_RESOLUTION REAL, MAX_RESOLUTION REAL, MIN_DEFOCUS REAL, MAX_DEFOCUS REAL,
  DEFOCUS_STEP REAL, RESTRAIN_ASTIGMATISM INTEGER, TOLERATED_ASTIGMATISM REAL,
  FIND_ADDITIONAL_PHASE_SHIFT INTEGER, MIN_PHASE_SHIFT REAL, MAX_PHASE_SHIFT REAL,
  PHASE_SHIFT_STEP REAL, DEFOCUS1 REAL, DEFOCUS2 REAL, DEFOCUS_ANGLE REAL,
  ADDITIONAL_PHASE_SHIFT REAL, SCORE REAL, DETECTED_RING_RESOLUTION REAL,
  DETECTED_ALIAS_RESOLUTION REAL, OUTPUT_DIAGNOSTIC_FILE TEXT, NUMBER_OF_FRAMES_AVERAGED INTEGER
);

CREATE TABLE IF NOT EXISTS PARTICLE_PICKING_LIST(
  PICKING_ID INTEGER PRIMARY KEY, DATETIME_OF_RUN INTEGER, PICKING_JOB_ID TEXT,
  PARENT_IMAGE_ASSET_ID INTEGER, PICKING_ALGORITHM INTEGER, CHARACTERISTIC_RADIUS REAL,
  MAXIMUM_RADIUS REAL, THRESHOLD_PEAK_HEIGHT REAL, HIGHEST_RESOLUTION_USED_IN_PICKING REAL,
  MIN_DIST_FROM_EDGES INTEGER, AVOID_HIGH_VARIANCE INTEGER, AVOID_HIGH_LOW_MEAN INTEGER,
  NUM_BACKGROUND_BOXES INTEGER, MANUAL_EDIT INTEGER
);

CREATE TABLE IF NOT EXISTS PARTICLE_POSITION_ASSETS(
  PARTICLE_POSITION_ASSET_ID INTEGER PRIMARY KEY, PARENT_IMAGE_ASSET_ID INTEGER,
  PICKING_ID INTEGER, PICK_JOB_ID TEXT, X_POSITION REAL, Y_POSITION REAL,
  PEAK_HEIGHT REAL, TEMPLATE_ASSET_ID INTEGER, TEMPLATE_PSI REAL, TEMPLATE_THETA REAL,
  TEMPLATE_PHI REAL
);

CREATE TABLE IF NOT EXISTS REFINEMENT_PACKAGE_ASSETS(
  REFINEMENT_PACKAGE_ASSET_ID INTEGER PRIMARY KEY, NAME TEXT, STACK_FILENAME TEXT,
  STACK_BOX_SIZE INTEGER, OUTPUT_PIXEL_SIZE REAL, SYMMETRY TEXT, MOLECULAR_WEIGHT REAL,
  PARTICLE_SIZE REAL, NUMBER_OF_CLASSES INTEGER, NUMBER_OF_REFINEMENTS INTEGER,
  LAST_REFINEMENT_ID INTEGER, STACK_HAS_WHITE_PROTEIN INTEGER
);

CREATE TABLE IF NOT EXISTS CLASSIFICATION_LIST(
  CLASSIFICATION_ID INTEGER PRIMARY KEY, REFINEMENT_PACKAGE_ASSET_ID INTEGER, NAME TEXT,
  CLASS_AVERAGE_FILE TEXT, REFINEMENT_WAS_IMPORTED_OR_GENERATED INTEGER, DATETIME_OF_RUN INTEGER,
  STARTING_CLASSIFICATION_ID INTEGER, NUMBER_OF_PARTICLES INTEGER, NUMBER_OF_CLASSES INTEGER,
  LOW_RESOLUTION_LIMIT REAL, HIGH_RESOLUTION_LIMIT REAL, MASK_RADIUS REAL,
  ANGULAR_SEARCH_STEP REAL, SEARCH_RANGE_X REAL, SEARCH_RANGE_Y REAL, SMOOTHING_FACTOR REAL,
  EXCLUDE_BLANK_EDGES INTEGER, AUTO_PERCENT_USED INTEGER, PERCENT_USED REAL
);

CREATE TABLE IF NOT EXISTS REFINEMENT_LIST(
  REFINEMENT_ID INTEGER PRIMARY KEY, REFINEMENT_PACKAGE_ASSET_ID INTEGER, NAME TEXT,
  RESOLUTION_STATISTICS_ARE_GENERATED INTEGER, DATETIME_OF_RUN INTEGER,
  STARTING_REFINEMENT_ID INTEGER, NUMBER_OF_PARTICLES INTEGER, NUMBER_OF_CLASSES INTEGER,
  RESOLUTION_STATISTICS_BOX_SIZE INTEGER, RESOLUTION_STATISTICS_PIXEL_SIZE REAL, PERCENT_USED REAL
);

-- Our own addition, no cisTEM equivalent -- see module docstring.
CREATE TABLE IF NOT EXISTS JOBS(
  JOB_ID TEXT PRIMARY KEY,
  STAGE TEXT NOT NULL, NAME TEXT, PARAMS_JSON TEXT,
  STATUS TEXT NOT NULL DEFAULT 'queued', PROGRESS INTEGER NOT NULL DEFAULT 0,
  CREATED_AT TEXT, STARTED_AT TEXT, FINISHED_AT TEXT, ERROR TEXT, METRICS_JSON TEXT,
  CANCEL_REQUESTED INTEGER NOT NULL DEFAULT 0,
  MOVIE_GROUP_ID INTEGER
);

CREATE TABLE IF NOT EXISTS JOB_LOG_LINES(
  JOB_ID TEXT NOT NULL, SEQ INTEGER NOT NULL, LINE TEXT,
  PRIMARY KEY(JOB_ID, SEQ)
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON JOBS(STATUS);
CREATE INDEX IF NOT EXISTS idx_movie_alignment_movie ON MOVIE_ALIGNMENT_LIST(MOVIE_ASSET_ID);
CREATE INDEX IF NOT EXISTS idx_image_assets_parent_movie ON IMAGE_ASSETS(PARENT_MOVIE_ID);
"""

# (profile_name, manager_run_command) -- matches the run-profile options
# job_runner.html has always hardcoded client-side (see <select id="runProfile">).
RUN_PROFILE_SEED = [
    "Local (single-threaded)",
    "Local (multi-threaded)",
    "Cluster (Slurm)",
]


def slugify(name):
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "project"


def project_dir(project_id):
    return PROJECTS_ROOT / project_id


def project_db_path(project_id):
    return project_dir(project_id) / "project.db"


def project_exists(project_id):
    return project_db_path(project_id).is_file()


def get_conn(project_id):
    """Open a fresh connection scoped to one project, schema guaranteed present."""
    path = project_db_path(project_id)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.executescript(SCHEMA_SQL)
    return conn


def create_project(name):
    name = (name or "").strip()
    if not name:
        raise ValueError("project name is required")

    project_id = "{}-{}".format(slugify(name), uuid.uuid4().hex[:6])
    pdir = project_dir(project_id)
    pdir.mkdir(parents=True, exist_ok=False)

    conn = get_conn(project_id)
    with conn:
        conn.execute(
            "INSERT INTO MASTER_SETTINGS(NUMBER, PROJECT_DIRECTORY, PROJECT_NAME, "
            "CURRENT_VERSION, TOTAL_CPU_HOURS, TOTAL_JOBS_RUN, CISTEM_VERSION_TEXT, "
            "CURRENT_WORKFLOW) VALUES (1, ?, ?, 1, 0, 0, ?, 'SINGLE_PARTICLE')",
            (str(pdir), name, "cryoem-job-runner web 0.1"),
        )
        conn.execute("INSERT INTO MOVIE_IMPORT_DEFAULTS(NUMBER) VALUES (1)")
        conn.execute(
            "INSERT INTO MOVIE_GROUP_LIST(GROUP_ID, GROUP_NAME, LIST_ID) "
            "VALUES (0, 'All Movies', 0)"
        )
        for profile_name in RUN_PROFILE_SEED:
            conn.execute(
                "INSERT INTO RUN_PROFILES(PROFILE_NAME, MANAGER_RUN_COMMAND) VALUES (?, NULL)",
                (profile_name,),
            )
    conn.close()
    return project_id


def get_project_summary(project_id):
    if not project_exists(project_id):
        return None
    conn = get_conn(project_id)
    row = conn.execute(
        "SELECT PROJECT_NAME, TOTAL_JOBS_RUN, CISTEM_VERSION_TEXT, CURRENT_WORKFLOW "
        "FROM MASTER_SETTINGS WHERE NUMBER=1"
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "id": project_id,
        "name": row["PROJECT_NAME"],
        "total_jobs_run": row["TOTAL_JOBS_RUN"] or 0,
        "cistem_version_text": row["CISTEM_VERSION_TEXT"],
        "current_workflow": row["CURRENT_WORKFLOW"],
    }


def list_projects():
    if not PROJECTS_ROOT.is_dir():
        return []
    out = []
    for entry in sorted(PROJECTS_ROOT.iterdir()):
        if not (entry / "project.db").is_file():
            continue
        try:
            summary = get_project_summary(entry.name)
        except sqlite3.DatabaseError:
            continue
        if summary is not None:
            out.append(summary)
    return out


def delete_project(project_id):
    pdir = project_dir(project_id)
    if pdir.is_dir():
        shutil.rmtree(pdir)


def now_epoch():
    return int(time.time())
