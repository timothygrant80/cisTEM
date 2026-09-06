"""
Per-project SQLite storage for the cisTEM3 reference backend.

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

import datetime
import os
import re
import shutil
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path

PROJECTS_ROOT = Path(__file__).parent / "data" / "projects"
# Run profiles describe the *machine* (how many processes, through which
# scheduler), not any one project, so they live in one system-wide file that
# administrators edit from the home page and every project reads. cisTEM
# keeps them per project; that meant setting the same thing up again for
# every project on the same box.
SYSTEM_DB_PATH = Path(__file__).parent / "data" / "system.db"

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

-- Columns match cisTEM's own CreateImageImportDefaultsTable() exactly: an
-- image import asks for voltage/Cs/pixel size/contrast and nothing else --
-- no dose, no gain/dark, no EER, since none of those describe an already
-- averaged micrograph. Cf. MOVIE_IMPORT_DEFAULTS above.
CREATE TABLE IF NOT EXISTS IMAGE_IMPORT_DEFAULTS(
  NUMBER INTEGER PRIMARY KEY, VOLTAGE REAL, SPHERICAL_ABERRATION REAL, PIXEL_SIZE REAL,
  PROTEIN_IS_WHITE INTEGER
);

CREATE TABLE IF NOT EXISTS IMAGE_GROUP_LIST(
  GROUP_ID INTEGER PRIMARY KEY, GROUP_NAME TEXT, LIST_ID INTEGER
);

-- Same deviation from real cisTEM as MOVIE_GROUP_MEMBERS (see module
-- docstring): one junction table instead of a numbered table per group.
CREATE TABLE IF NOT EXISTS IMAGE_GROUP_MEMBERS(
  GROUP_ID INTEGER NOT NULL, IMAGE_ASSET_ID INTEGER NOT NULL,
  PRIMARY KEY(GROUP_ID, IMAGE_ASSET_ID)
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

-- Particle position groups, the same shape as the movie and image groups
-- (cisTEM: PARTICLE_POSITION_GROUP_LIST + PARTICLE_POSITION_GROUP_<n>);
-- group 0 is All Particle Positions, seeded below.
CREATE TABLE IF NOT EXISTS PARTICLE_POSITION_GROUP_LIST(
  GROUP_ID INTEGER PRIMARY KEY, GROUP_NAME TEXT, LIST_ID INTEGER
);
CREATE TABLE IF NOT EXISTS PARTICLE_POSITION_GROUP_MEMBERS(
  GROUP_ID INTEGER NOT NULL, PARTICLE_POSITION_ASSET_ID INTEGER NOT NULL,
  PRIMARY KEY(GROUP_ID, PARTICLE_POSITION_ASSET_ID)
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

-- 3D volumes (MyVolumeAssetPanel): reconstructions, as cisTEM's VOLUME_ASSETS,
-- with the same groups machinery as the other asset kinds; group 0 is All
-- Volumes, seeded below. Ab-initio writes them; Refine 3D will read them.
CREATE TABLE IF NOT EXISTS VOLUME_ASSETS(
  VOLUME_ASSET_ID INTEGER PRIMARY KEY, NAME TEXT, FILENAME TEXT, RECONSTRUCTION_JOB_ID INTEGER,
  PIXEL_SIZE REAL, X_SIZE INTEGER, Y_SIZE INTEGER, Z_SIZE INTEGER,
  HALF_MAP1_FILENAME TEXT, HALF_MAP2_FILENAME TEXT
);
CREATE TABLE IF NOT EXISTS VOLUME_GROUP_LIST(
  GROUP_ID INTEGER PRIMARY KEY, GROUP_NAME TEXT, LIST_ID INTEGER
);
CREATE TABLE IF NOT EXISTS VOLUME_GROUP_MEMBERS(
  GROUP_ID INTEGER NOT NULL, VOLUME_ASSET_ID INTEGER NOT NULL,
  PRIMARY KEY(GROUP_ID, VOLUME_ASSET_ID)
);

-- Ab-initio 3D runs (Database::AddStartupJob): the settings of each run and,
-- in STARTUP_RESULT_<id>(CLASS_NUMBER, VOLUME_ASSET_ID), the volumes it made.
CREATE TABLE IF NOT EXISTS STARTUP_LIST(
  STARTUP_ID INTEGER PRIMARY KEY, REFINEMENT_PACKAGE_ASSET_ID INTEGER, NAME TEXT,
  NUMBER_OF_STARTS INTEGER, NUMBER_OF_CYCLES INTEGER, INITIAL_RES_LIMIT REAL, FINAL_RES_LIMIT REAL,
  AUTO_MASK INTEGER, AUTO_PERCENT_USED INTEGER, INITIAL_PERCENT_USED REAL, FINAL_PERCENT_USED REAL,
  MASK_RADIUS REAL, APPLY_LIKELIHOOD_BLURRING INTEGER, SMOOTHING_FACTOR REAL
);

-- Refine 3D's per-class reconstruction records (Database::AddReconstructionJob).
CREATE TABLE IF NOT EXISTS RECONSTRUCTION_LIST(
  RECONSTRUCTION_ID INTEGER PRIMARY KEY, REFINEMENT_PACKAGE_ID INTEGER, REFINEMENT_ID INTEGER, NAME TEXT,
  INNER_MASK_RADIUS REAL, OUTER_MASK_RADIUS REAL, RESOLUTION_LIMIT REAL, SCORE_WEIGHT_CONVERSION REAL,
  SHOULD_ADJUST_SCORES INTEGER, SHOULD_CROP_IMAGES INTEGER, SHOULD_SAVE_HALF_MAPS INTEGER, SHOULD_LIKELIHOOD_BLUR INTEGER,
  SMOOTHING_FACTOR REAL, CLASS_NUMBER INTEGER, VOLUME_ASSET_ID INTEGER
);

-- Refine2DResultsPanel's selection manager: a named set of class averages
-- of one classification (Database::AddClassificationSelection), the members
-- in CLASSIFICATION_SELECTION_<id>(CLASS_AVERAGE_NUMBER).
CREATE TABLE IF NOT EXISTS CLASSIFICATION_SELECTION_LIST(
  SELECTION_ID INTEGER PRIMARY KEY, SELECTION_NAME TEXT, CREATION_DATE INTEGER,
  REFINEMENT_PACKAGE_ID INTEGER, CLASSIFICATION_ID INTEGER, NUMBER_OF_CLASSES INTEGER,
  NUMBER_OF_SELECTIONS INTEGER
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
  JOB_NUMBER INTEGER,
  STATUS TEXT NOT NULL DEFAULT 'queued', PROGRESS INTEGER NOT NULL DEFAULT 0,
  CREATED_AT TEXT, STARTED_AT TEXT, FINISHED_AT TEXT, ERROR TEXT, METRICS_JSON TEXT,
  CANCEL_REQUESTED INTEGER NOT NULL DEFAULT 0,
  MOVIE_GROUP_ID INTEGER,
  JOB_TOKEN TEXT, CONTROLLER_SEQ INTEGER, TASKS_JSON TEXT
);

-- One row per unit of work the controller reported back (job_protocol
-- task_done), written as it arrives. This is the durable copy of what cisTEM
-- keeps in buffered_results[] until ProcessAllJobsFinished(): the stage
-- adapter turns these into MOVIE_ALIGNMENT_LIST etc. when the job finishes,
-- and after a server restart the runner rebuilds its done-set from here so a
-- resend of an already-recorded task is recognised as a duplicate.
CREATE TABLE IF NOT EXISTS JOB_TASKS(
  JOB_ID TEXT NOT NULL, TASK_INDEX INTEGER NOT NULL, REF TEXT,
  STATUS TEXT NOT NULL, CPU_MS INTEGER, ERROR TEXT, RESULT_JSON TEXT, FINISHED_AT TEXT,
  PRIMARY KEY(JOB_ID, TASK_INDEX)
);

CREATE TABLE IF NOT EXISTS JOB_LOG_LINES(
  JOB_ID TEXT NOT NULL, SEQ INTEGER NOT NULL, LINE TEXT,
  PRIMARY KEY(JOB_ID, SEQ)
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON JOBS(STATUS);
CREATE INDEX IF NOT EXISTS idx_movie_alignment_movie ON MOVIE_ALIGNMENT_LIST(MOVIE_ASSET_ID);
CREATE INDEX IF NOT EXISTS idx_ctf_estimate_image ON ESTIMATED_CTF_PARAMETERS(IMAGE_ASSET_ID);
CREATE INDEX IF NOT EXISTS idx_picking_image ON PARTICLE_PICKING_LIST(PARENT_IMAGE_ASSET_ID);
CREATE INDEX IF NOT EXISTS idx_position_image ON PARTICLE_POSITION_ASSETS(PARENT_IMAGE_ASSET_ID);
CREATE INDEX IF NOT EXISTS idx_image_assets_parent_movie ON IMAGE_ASSETS(PARENT_MOVIE_ID);
CREATE INDEX IF NOT EXISTS idx_image_group_members_asset ON IMAGE_GROUP_MEMBERS(IMAGE_ASSET_ID);
"""

# The run profiles every project starts with. Names are the three the Run
# Profile picker has always offered; the commands follow cisTEM's own defaults
# (RunProfileManager::AddDefaultLocalProfile and "Default Reconstruction"):
# the controller is launched locally with `$command`, and each run command is
# `$command` too, N copies with M threads each, 10 ms apart. `$command` is
# substituted by the launcher -- the server for the manager command, the
# controller for the run commands -- exactly as in guix_job_control.cpp.
#
# The Slurm profile is deliberately seeded with *no* run commands: a profile
# with zero total jobs can't be started (cisTEM's OnUpdateUI greys the start
# button), which is the honest state for a template nobody has adapted to
# their cluster yet. Editing profiles is still a known gap.
_CORES = os.cpu_count() or 4
RUN_PROFILE_SEED = [
    # (name, manager_command, [(command, copies, threads_per_copy, delay_ms), ...])
    # cores + 1 copies, as AddDefaultLocalProfile() does: the first process to
    # connect becomes the master and only dispatches, so this is `cores`
    # processes actually computing.
    ("Local (single-threaded)", "$command", [("$command", _CORES + 1, 1, 10)]),
    ("Local (multi-threaded)", "$command", [("$command", 2, max(1, _CORES // 2), 10)]),
    ("Cluster (Slurm)", "$command", []),
]

# cisTEM keeps each profile's commands in its own numbered table,
# RUN_PROFILE_COMMANDS_<RUN_PROFILE_ID>, and this follows suit -- unlike the
# group-membership sharding (see the module docstring), there are only ever a
# handful of profiles, so matching cisTEM costs nothing and a real cisTEM
# reads the result. Column names are verbatim, misspelling included.
_RUN_PROFILE_COMMANDS_SQL = (
    "CREATE TABLE IF NOT EXISTS RUN_PROFILE_COMMANDS_{}("
    "COMMANDS_NUMBER INTEGER PRIMARY KEY, COMMAND_STRING TEXT, NUMBER_OF_COPIES INTEGER, "
    "NUMBER_OF_THREADS_PER_COPY INTEGER, OVERRIDE_TOTAL_NUMBER_OF_COPIES INTEGER, "
    "OVERIDDEN_TOTAL_NUMBER_OF_COPIES INTEGER, DELAY_TIME_IN_MS INTEGER)"
)


def slugify(name):
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "project"


def project_dir(project_id):
    return PROJECTS_ROOT / project_id


def project_db_path(project_id):
    return project_dir(project_id) / "project.db"


def project_exists(project_id):
    return project_db_path(project_id).is_file()


# Columns added after the original schema shipped. No migration framework
# (see module docstring) -- each is a guarded ALTER TABLE, run unconditionally
# on every open. A no-op on databases that already have the column, whether
# that's because this ran before or because SCHEMA_SQL's CREATE TABLE already
# included it on a brand-new database.
_ALTER_STATEMENTS = [
    "ALTER TABLE MASTER_SETTINGS ADD COLUMN OWNER_USER_ID INTEGER",
    "ALTER TABLE MASTER_SETTINGS ADD COLUMN OWNER_USERNAME TEXT",
    # When the project was made (epoch seconds, like DATETIME_OF_RUN). cisTEM's
    # MASTER_SETTINGS has no such column; the project picker shows it.
    "ALTER TABLE MASTER_SETTINGS ADD COLUMN CREATION_DATE INTEGER",
    # The four columns cisTEM's WriteResultToDataBase() writes that the
    # copied schema predates (FindCTFPanel.cpp).
    "ALTER TABLE ESTIMATED_CTF_PARAMETERS ADD COLUMN LARGE_ASTIGMATISM_EXPECTED INTEGER",
    "ALTER TABLE ESTIMATED_CTF_PARAMETERS ADD COLUMN ICINESS REAL",
    "ALTER TABLE ESTIMATED_CTF_PARAMETERS ADD COLUMN TILT_ANGLE REAL",
    "ALTER TABLE ESTIMATED_CTF_PARAMETERS ADD COLUMN TILT_AXIS REAL",
    # Which pick job's positions an image currently carries (stages/find_particles.py):
    # cisTEM infers it from which PARTICLE_POSITION_ASSETS rows exist, which
    # says nothing for an image with no picks. Same idea as CTF_ESTIMATION_ID.
    "ALTER TABLE IMAGE_ASSETS ADD COLUMN ACTIVE_PICKING_ID INTEGER",
    "ALTER TABLE JOBS ADD COLUMN JOB_NUMBER INTEGER",
    # The job protocol's per-job state, so a controller can reconnect to a
    # restarted server (docs/job-protocol.md section 7.2): the token it must
    # present, the highest seq of its we processed, and the task list we sent
    # it -- kept verbatim so restore() needs nothing but this row.
    "ALTER TABLE JOBS ADD COLUMN JOB_TOKEN TEXT",
    "ALTER TABLE JOBS ADD COLUMN CONTROLLER_SEQ INTEGER",
    "ALTER TABLE JOBS ADD COLUMN TASKS_JSON TEXT",
    # 2D classification (server/classification.py): a user-visible class2d
    # job drives hidden child jobs (one refine2d/merge2d run each) that
    # point back at it, and keeps where it has got to in STATE_JSON so a
    # server restart can pick the cycle up again.
    "ALTER TABLE JOBS ADD COLUMN PARENT_JOB_ID TEXT",
    "ALTER TABLE JOBS ADD COLUMN STATE_JSON TEXT",
    # Which job a classification came from -- cisTEM has no job table to
    # point at; the same addition PARTICLE_PICKING_LIST.PICKING_JOB_ID is.
    "ALTER TABLE CLASSIFICATION_LIST ADD COLUMN JOB_ID TEXT",
    "ALTER TABLE STARTUP_LIST ADD COLUMN JOB_ID TEXT",
    "ALTER TABLE REFINEMENT_LIST ADD COLUMN JOB_ID TEXT",
]

# Rows every project must have, seeded here rather than in create_project()
# so that projects created before image assets existed get them too. All
# three are INSERT OR IGNORE / idempotent, so re-running them on every open
# is a no-op once they've landed.
#
# The last one backfills All Images membership: IMAGE_ASSETS rows predate
# IMAGE_GROUP_MEMBERS -- every completed Align Movies job has been writing
# them since long before images had groups -- so the master list has to
# adopt whatever it finds rather than assume every row was inserted through
# the image import route. Cheap at this scale (a project's images number in
# the thousands at most) and self-healing.
_SEED_STATEMENTS = [
    "INSERT OR IGNORE INTO PARTICLE_POSITION_GROUP_LIST(GROUP_ID, GROUP_NAME, LIST_ID) VALUES (0, 'All Particle Positions', 0)",
    "INSERT OR IGNORE INTO VOLUME_GROUP_LIST(GROUP_ID, GROUP_NAME, LIST_ID) VALUES (0, 'All Volumes', 0)",
    "INSERT OR IGNORE INTO IMAGE_GROUP_LIST(GROUP_ID, GROUP_NAME, LIST_ID) VALUES (0, 'All Images', 0)",
    "INSERT OR IGNORE INTO IMAGE_IMPORT_DEFAULTS(NUMBER) VALUES (1)",
    "INSERT OR IGNORE INTO IMAGE_GROUP_MEMBERS(GROUP_ID, IMAGE_ASSET_ID) "
    "SELECT 0, IMAGE_ASSET_ID FROM IMAGE_ASSETS",
    # Number the jobs that predate JOB_NUMBER, oldest first, so a project's
    # numbering is continuous rather than restarting at 1 alongside them.
    # Touches only NULL rows, so it stops being a no-op the moment it has
    # run once. Their NAME is left alone -- those were typed by hand, back
    # when the submit form asked for one.
    "UPDATE JOBS SET JOB_NUMBER = ("
    "  SELECT COUNT(*) FROM JOBS older WHERE older.CREATED_AT <= JOBS.CREATED_AT"
    ") WHERE JOB_NUMBER IS NULL",
]


def get_conn(project_id):
    """Open a fresh connection scoped to one project, schema guaranteed present."""
    path = project_db_path(project_id)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.executescript(SCHEMA_SQL)
    for stmt in _ALTER_STATEMENTS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass  # column already exists
    with conn:
        for stmt in _SEED_STATEMENTS:
            conn.execute(stmt)
        # Projects made before CREATION_DATE existed get a best guess once.
        if conn.execute("SELECT 1 FROM MASTER_SETTINGS WHERE NUMBER=1 AND CREATION_DATE IS NULL").fetchone():
            conn.execute("UPDATE MASTER_SETTINGS SET CREATION_DATE=? WHERE NUMBER=1", (_guess_creation_time(path.parent, conn),))
    return conn


def _guess_creation_time(pdir, conn):
    """For a project made before CREATION_DATE existed: the project
    directory's birth time if the filesystem records one (Python's os.stat
    doesn't expose it on Linux, GNU stat does), else the first job's
    creation time, else the directory's mtime. Not st_ctime -- that is the
    inode change time and moves with every write to the database."""
    try:
        birth = int(subprocess.run(["stat", "-c", "%W", str(pdir)], capture_output=True, text=True, timeout=5).stdout.strip() or 0)
        if birth > 0:
            return birth
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    first_job = conn.execute("SELECT MIN(CREATED_AT) FROM JOBS").fetchone()[0]
    if first_job:
        try:
            return int(datetime.datetime.fromisoformat(first_job).timestamp())
        except ValueError:
            pass
    return int(pdir.stat().st_mtime)


def _seed_run_profile_commands(conn):
    """Give every RUN_PROFILES row a manager command and a commands table,
    and fill the table with the seed defaults the first time -- so projects
    made before run profiles had commands get the same ones a new project
    does. Idempotent: only NULL manager commands and empty tables are
    touched, so a profile someone has edited is left alone."""
    defaults = {name: (manager, commands) for name, manager, commands in RUN_PROFILE_SEED}
    for row in conn.execute("SELECT RUN_PROFILE_ID, PROFILE_NAME, MANAGER_RUN_COMMAND FROM RUN_PROFILES").fetchall():
        pid = row["RUN_PROFILE_ID"]
        manager, commands = defaults.get(row["PROFILE_NAME"], ("$command", []))
        conn.execute(
            "UPDATE RUN_PROFILES SET MANAGER_RUN_COMMAND = COALESCE(MANAGER_RUN_COMMAND, ?), "
            "GUI_ADDRESS = COALESCE(GUI_ADDRESS, ''), CONTROLLER_ADDRESS = COALESCE(CONTROLLER_ADDRESS, ''), "
            "COMMANDS_ID = COALESCE(COMMANDS_ID, RUN_PROFILE_ID) WHERE RUN_PROFILE_ID = ?",
            (manager, pid),
        )
        conn.execute(_RUN_PROFILE_COMMANDS_SQL.format(pid))
        # Only tell an untouched (empty) table from a deliberately emptied one
        # by whether the manager command was still NULL -- i.e. this is the
        # first time this profile has been seen by code that knows commands.
        if row["MANAGER_RUN_COMMAND"] is None and commands:
            for number, (command, copies, threads, delay_ms) in enumerate(commands):
                conn.execute(
                    "INSERT OR IGNORE INTO RUN_PROFILE_COMMANDS_{}(COMMANDS_NUMBER, COMMAND_STRING, "
                    "NUMBER_OF_COPIES, NUMBER_OF_THREADS_PER_COPY, OVERRIDE_TOTAL_NUMBER_OF_COPIES, "
                    "OVERIDDEN_TOTAL_NUMBER_OF_COPIES, DELAY_TIME_IN_MS) VALUES (?, ?, ?, ?, 0, 0, ?)".format(pid),
                    (number, command, copies, threads, delay_ms),
                )


def load_run_profiles(conn):
    """Every run profile with its commands, in the shape the job protocol's
    `package.profile` wants (docs/job-protocol.md section 6.2) plus the ids
    the API needs. `total_jobs` is cisTEM's RunProfile::ReturnTotalJobs()."""
    profiles = []
    for row in conn.execute(
        "SELECT RUN_PROFILE_ID, PROFILE_NAME, MANAGER_RUN_COMMAND, GUI_ADDRESS, CONTROLLER_ADDRESS "
        "FROM RUN_PROFILES ORDER BY RUN_PROFILE_ID"
    ).fetchall():
        pid = row["RUN_PROFILE_ID"]
        conn.execute(_RUN_PROFILE_COMMANDS_SQL.format(pid))
        commands = [
            {
                "command": c["COMMAND_STRING"],
                "copies": c["NUMBER_OF_COPIES"] or 0,
                "threads_per_copy": c["NUMBER_OF_THREADS_PER_COPY"] or 1,
                "override_total_copies": bool(c["OVERRIDE_TOTAL_NUMBER_OF_COPIES"]),
                "overridden_total_copies": c["OVERIDDEN_TOTAL_NUMBER_OF_COPIES"] or 0,
                "delay_ms": c["DELAY_TIME_IN_MS"] or 0,
            }
            for c in conn.execute(
                "SELECT * FROM RUN_PROFILE_COMMANDS_{} ORDER BY COMMANDS_NUMBER".format(pid)
            ).fetchall()
        ]
        total_jobs = sum(
            c["overridden_total_copies"] if c["override_total_copies"] else c["copies"] for c in commands
        )
        profiles.append({
            "run_profile_id": pid,
            "name": row["PROFILE_NAME"],
            "manager_command": row["MANAGER_RUN_COMMAND"] or "$command",
            "gui_address": row["GUI_ADDRESS"] or "",
            "controller_address": row["CONTROLLER_ADDRESS"] or "",
            "run_commands": commands,
            "total_jobs": total_jobs,
        })
    return profiles


def load_run_profile_by_name(conn, name):
    for profile in load_run_profiles(conn):
        if profile["name"] == name:
            return profile
    return None


def load_run_profile(conn, run_profile_id):
    for profile in load_run_profiles(conn):
        if profile["run_profile_id"] == run_profile_id:
            return profile
    return None


# ---------------------------------------------------------------------------
# Editing run profiles -- MyRunProfilesPanel's Add / Rename / Remove /
# Duplicate and the Save of its commands panel, as database operations.
# Names are unique per project (case-insensitively) because jobs refer to
# their profile by name; cisTEM tolerates duplicates but has no such
# reference to keep straight.
# ---------------------------------------------------------------------------

class RunProfileError(ValueError):
    """A run-profile edit the API should refuse with a 400."""


def _validate_command_text(text, what):
    text = (text or "").strip()
    if "$command" not in text:
        # cisTEM's exact wording, from AddRunCommandDialog / CommandsSaveButtonClick.
        raise RunProfileError('Oops! - {} must contain "$command"'.format(what))
    return text


def _validate_run_commands(run_commands):
    clean = []
    for i, c in enumerate(run_commands or []):
        if not isinstance(c, dict):
            raise RunProfileError("run command {} is not an object".format(i))
        try:
            copies = int(c.get("copies", 1))
            threads = int(c.get("threads_per_copy", 1))
            overridden = int(c.get("overridden_total_copies", 0) or 0)
            delay = int(c.get("delay_ms", 10) or 0)
        except (TypeError, ValueError):
            raise RunProfileError("run command {} has a non-integer count".format(i))
        if copies < 1 or threads < 1 or overridden < 0 or delay < 0:
            raise RunProfileError("run command {} has a count out of range".format(i))
        clean.append({
            "command": _validate_command_text(c.get("command"), "Command"),
            "copies": copies,
            "threads_per_copy": threads,
            "override_total_copies": bool(c.get("override_total_copies", False)),
            "overridden_total_copies": overridden,
            "delay_ms": delay,
        })
    return clean


def _unique_profile_name(conn, wanted, exclude_id=None):
    """`wanted`, or `wanted (2)`, `wanted (3)`... -- the first not already
    taken by another profile."""
    taken = {
        r["PROFILE_NAME"].lower()
        for r in conn.execute("SELECT RUN_PROFILE_ID, PROFILE_NAME FROM RUN_PROFILES").fetchall()
        if r["RUN_PROFILE_ID"] != exclude_id
    }
    if wanted.lower() not in taken:
        return wanted
    n = 2
    while "{} ({})".format(wanted, n).lower() in taken:
        n += 1
    return "{} ({})".format(wanted, n)


def default_local_profile_spec():
    """What cisTEM's Add button creates (RunProfileManager::AddDefaultLocalProfile)."""
    return {
        "name": "Default Local",
        "manager_command": "$command",
        "gui_address": "",
        "controller_address": "",
        "run_commands": [{"command": "$command", "copies": _CORES + 1, "threads_per_copy": 1,
                          "override_total_copies": False, "overridden_total_copies": 0, "delay_ms": 10}],
    }


def _write_run_commands(conn, run_profile_id, commands):
    """Replace a profile's command table wholesale -- what
    Database::AddOrReplaceRunProfile does (DeleteTable + CreateTable)."""
    conn.execute("DROP TABLE IF EXISTS RUN_PROFILE_COMMANDS_{}".format(run_profile_id))
    conn.execute(_RUN_PROFILE_COMMANDS_SQL.format(run_profile_id))
    for number, c in enumerate(commands):
        conn.execute(
            "INSERT INTO RUN_PROFILE_COMMANDS_{}(COMMANDS_NUMBER, COMMAND_STRING, NUMBER_OF_COPIES, "
            "NUMBER_OF_THREADS_PER_COPY, OVERRIDE_TOTAL_NUMBER_OF_COPIES, OVERIDDEN_TOTAL_NUMBER_OF_COPIES, "
            "DELAY_TIME_IN_MS) VALUES (?, ?, ?, ?, ?, ?, ?)".format(run_profile_id),
            (number, c["command"], c["copies"], c["threads_per_copy"], 1 if c["override_total_copies"] else 0,
             c["overridden_total_copies"], c["delay_ms"]),
        )


def create_run_profile(conn, spec):
    """Add a profile from a spec shaped like load_run_profiles() output (any
    field may be missing). Returns the new id. The name is made unique the
    way cisTEM's Duplicate would want ("Copy of X", then "Copy of X (2)")."""
    name = (spec.get("name") or "New Profile").strip() or "New Profile"
    manager = _validate_command_text(spec.get("manager_command") or "$command", "Command")
    commands = _validate_run_commands(spec.get("run_commands") or [])
    with conn:
        name = _unique_profile_name(conn, name)
        cur = conn.execute(
            "INSERT INTO RUN_PROFILES(PROFILE_NAME, MANAGER_RUN_COMMAND, GUI_ADDRESS, CONTROLLER_ADDRESS) "
            "VALUES (?, ?, ?, ?)",
            (name, manager, (spec.get("gui_address") or "").strip(), (spec.get("controller_address") or "").strip()),
        )
        pid = cur.lastrowid
        conn.execute("UPDATE RUN_PROFILES SET COMMANDS_ID = RUN_PROFILE_ID WHERE RUN_PROFILE_ID = ?", (pid,))
        _write_run_commands(conn, pid, commands)
    return pid


def update_run_profile(conn, run_profile_id, fields):
    """Change any of name / manager_command / gui_address /
    controller_address / run_commands. Raises RunProfileError for an invalid
    value, KeyError for an unknown profile."""
    row = conn.execute("SELECT RUN_PROFILE_ID FROM RUN_PROFILES WHERE RUN_PROFILE_ID=?", (run_profile_id,)).fetchone()
    if row is None:
        raise KeyError(run_profile_id)
    sets, values = [], []
    if "name" in fields:
        name = (fields["name"] or "").strip()
        if not name:
            raise RunProfileError("a profile needs a name")
        if _unique_profile_name(conn, name, exclude_id=run_profile_id) != name:
            raise RunProfileError("another profile is already called {!r}".format(name))
        sets.append("PROFILE_NAME=?"); values.append(name)
    if "manager_command" in fields:
        sets.append("MANAGER_RUN_COMMAND=?"); values.append(_validate_command_text(fields["manager_command"], "Command"))
    if "gui_address" in fields:
        sets.append("GUI_ADDRESS=?"); values.append((fields["gui_address"] or "").strip())
    if "controller_address" in fields:
        sets.append("CONTROLLER_ADDRESS=?"); values.append((fields["controller_address"] or "").strip())
    commands = _validate_run_commands(fields["run_commands"]) if "run_commands" in fields else None
    with conn:
        if sets:
            conn.execute("UPDATE RUN_PROFILES SET {} WHERE RUN_PROFILE_ID=?".format(", ".join(sets)),
                         values + [run_profile_id])
        if commands is not None:
            _write_run_commands(conn, run_profile_id, commands)


def delete_run_profile(conn, run_profile_id):
    """Database::DeleteRunProfile: the row and its command table."""
    with conn:
        cur = conn.execute("DELETE FROM RUN_PROFILES WHERE RUN_PROFILE_ID=?", (run_profile_id,))
        conn.execute("DROP TABLE IF EXISTS RUN_PROFILE_COMMANDS_{}".format(run_profile_id))
    return cur.rowcount > 0


def create_project(name, owner_user_id, owner_username):
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
            "CURRENT_WORKFLOW, OWNER_USER_ID, OWNER_USERNAME, CREATION_DATE) "
            "VALUES (1, ?, ?, 1, 0, 0, ?, 'SINGLE_PARTICLE', ?, ?, ?)",
            (str(pdir), name, "cistem3 web 0.1", owner_user_id, owner_username, now_epoch()),
        )
        conn.execute("INSERT INTO MOVIE_IMPORT_DEFAULTS(NUMBER) VALUES (1)")
        conn.execute(
            "INSERT INTO MOVIE_GROUP_LIST(GROUP_ID, GROUP_NAME, LIST_ID) "
            "VALUES (0, 'All Movies', 0)"
        )
    conn.close()
    return project_id


_SYSTEM_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS RUN_PROFILES(
  RUN_PROFILE_ID INTEGER PRIMARY KEY, PROFILE_NAME TEXT, MANAGER_RUN_COMMAND TEXT,
  GUI_ADDRESS TEXT, CONTROLLER_ADDRESS TEXT, COMMANDS_ID INTEGER
);
"""


def get_system_conn():
    """The system-wide database: run profiles (see SYSTEM_DB_PATH). Seeded
    with the same three profiles a cisTEM project starts with the first
    time it is opened; after that, what administrators make of them. The
    run-profile helpers below take any connection with these tables, so
    they serve this file exactly as they used to serve a project's."""
    SYSTEM_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(SYSTEM_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.executescript(_SYSTEM_SCHEMA_SQL)
    with conn:
        if conn.execute("SELECT COUNT(*) FROM RUN_PROFILES").fetchone()[0] == 0:
            # MANAGER_RUN_COMMAND is left NULL on purpose: that is the marker
            # _seed_run_profile_commands() uses to know a profile has never
            # been given its commands.
            for profile_name, _manager, _commands in RUN_PROFILE_SEED:
                conn.execute("INSERT INTO RUN_PROFILES(PROFILE_NAME, MANAGER_RUN_COMMAND) VALUES (?, NULL)", (profile_name,))
        _seed_run_profile_commands(conn)
    return conn


def get_project_summary(project_id):
    if not project_exists(project_id):
        return None
    conn = get_conn(project_id)
    row = conn.execute(
        "SELECT PROJECT_NAME, TOTAL_JOBS_RUN, CISTEM_VERSION_TEXT, CURRENT_WORKFLOW, "
        "OWNER_USER_ID, OWNER_USERNAME, CREATION_DATE FROM MASTER_SETTINGS WHERE NUMBER=1"
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
        "owner_user_id": row["OWNER_USER_ID"],
        "owner_username": row["OWNER_USERNAME"],
        "creation_date": row["CREATION_DATE"],
    }


def get_project_owner(project_id):
    conn = get_conn(project_id)
    row = conn.execute("SELECT OWNER_USER_ID FROM MASTER_SETTINGS WHERE NUMBER=1").fetchone()
    conn.close()
    return row["OWNER_USER_ID"] if row else None


def list_projects(owner_user_id=None):
    """owner_user_id=None returns every project (admin view); a real id
    returns only that owner's projects, never NULL-owner (legacy/pre-auth)
    ones -- those stay admin-only visible by default."""
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
        if summary is None:
            continue
        if owner_user_id is not None and summary["owner_user_id"] != owner_user_id:
            continue
        out.append(summary)
    return out


def delete_project(project_id):
    pdir = project_dir(project_id)
    if pdir.is_dir():
        shutil.rmtree(pdir)


def now_epoch():
    return int(time.time())
