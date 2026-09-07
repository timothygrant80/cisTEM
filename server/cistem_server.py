"""
Reference job-queue API for the cisTEM3 page (cistem3.html).

This is a STARTING POINT, not a production job scheduler. It implements the
HTTP contract the front-end expects (see README.md), backed by one SQLite
file per project (see db.py) instead of a single global in-memory queue.
Everything a project owns -- imported movies, job history, job results --
lives in that project's own database, so opening a different project means
nothing more than pointing requests at a different file. Wire in your real
pipeline by editing STAGE_COMMANDS below -- everything else (routing, project
management, status tracking, logs, cancellation) already works.

Quick start
-----------
    pip install -r requirements.txt
    python cistem_server.py
    # serves on http://localhost:8000, API under /api

Then open cistem3.html in a browser and point "API base URL" at
http://localhost:8000/api

Security
--------
Bearer-token auth (see auth.py) -- every user has their own account and only
sees their own projects, except admins, who see and can access all of them.
CORS is wide open on origin (Access-Control-Allow-Origin: *) so the page can
be opened as a local file, but that's orthogonal to auth: nothing works
without a valid Authorization header regardless of where the request came
from. Only run this on a trusted network -- there's no rate limiting or
HTTPS enforcement (see README.md's Security section for what that implies).
"""

import glob as glob_module
import json
import os
import shlex
import shutil
import struct
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, g, jsonify, request, send_from_directory
from flask_cors import CORS

import abinitio
import auth
import classification
import db
import refine3d
import autorefine
import generate3d
import refinectf
import sharpen
import refinements
import job_runner
import refinement_packages
import stages
import imageheaders
import preview
import volumes

# Stages that are cycles of program runs rather than one: a user-visible
# parent job drives hidden children. Keyed by the parent's STAGE.
DRIVERS = {classification.STAGE: classification, abinitio.STAGE: abinitio, refine3d.STAGE: refine3d, autorefine.STAGE: autorefine,
           refinectf.STAGE: refinectf, generate3d.STAGE: generate3d}

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, allow_headers=["Content-Type", "Authorization"])

REPO_ROOT = Path(__file__).parent.parent


def _git(args):
    """Run a git command against this checkout; None on any failure (not a
    git repo, git not installed, etc.) rather than raising -- the home
    screen's version display just omits what it can't determine."""
    try:
        result = subprocess.run(
            ["git"] + args, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=3
        )
    except Exception:  # noqa: BLE001
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None

# Live-only state that can't be persisted: a running job's subprocess handle
# and a fast in-process cancel flag. Durable status/progress/log lives in
# each project's SQLite file (see db.py) so it survives across requests and
# is visible from any thread; this dict only exists to let cancel() reach
# the actual OS process and to avoid a DB round trip on every progress check.
_live_lock = threading.Lock()
_live = {}  # job_id -> {"proc": Popen|None, "cancel_requested": bool}


# ---------------------------------------------------------------------------
# Real execution: the job runner (docs/job-protocol.md)
#
# One JobRunner per server process listens for cistem_job_controller
# connections. A stage runs for real when it has an adapter in stages/ and
# the controller executable can be found; otherwise it falls back to the
# simulation below, exactly as before -- so a checkout with no cisTEM build
# still demos end to end. All settings come from the environment:
#
#   CISTEM_JOB_CONTROLLER  controller command, default "cistem_job_controller"
#                          (e.g. "python3 tools/fake_controller.py" to test
#                          the server side without any C++)
#   JOB_RUNNER_PORT        listening port, default 8010
#   JOB_RUNNER_BIND        bind address, default 0.0.0.0
#   JOB_RUNNER_HOSTS       comma-separated addresses the controller is told
#                          to dial; default: this machine's, loopback last
#   JOB_RUNNER_ENABLED     set to 0 to never start the listener
# ---------------------------------------------------------------------------

CONTROLLER_COMMAND = os.environ.get("CISTEM_JOB_CONTROLLER", "cistem_job_controller")
_job_runner = None


def _controller_available():
    """True if the first word of CONTROLLER_COMMAND resolves on PATH or is an
    existing file -- the same test the simulation fallback makes for stage
    binaries."""
    try:
        first = shlex.split(CONTROLLER_COMMAND)[0]
    except (ValueError, IndexError):
        return False
    return shutil.which(first) is not None or os.path.isfile(first)


class DbSink(job_runner.Sink):
    """Where the runner's callbacks land: the project database. The runner
    only knows job ids, so this keeps the job -> project map (registered by
    create_job / _recover_interrupted_jobs) and forgets an entry once the
    job reaches a terminal status."""

    def __init__(self):
        self._projects = {}
        self._progress = {}  # job id -> (adapter's on_task_progress or None, {task index: sent task})
        self._lock = threading.Lock()

    def register(self, job_id, project_id):
        with self._lock:
            self._projects[job_id] = project_id

    def _project(self, job_id):
        with self._lock:
            return self._projects.get(job_id)

    def _forget(self, job_id):
        with self._lock:
            self._projects.pop(job_id, None)
            self._progress.pop(job_id, None)

    def on_status(self, job_id, status, error=None):
        project_id = self._project(job_id)
        if project_id is None:
            return
        if status == job_runner.LAUNCHING:
            _update_job(project_id, job_id, STATUS="queued")
        elif status == job_runner.RUNNING:
            row = _fetch_job_row(project_id, job_id)
            fields = {"STATUS": "running"}
            if row is not None and not row["STARTED_AT"]:
                fields["STARTED_AT"] = now_iso()
            _update_job(project_id, job_id, **fields)
        elif status == job_runner.AWAITING_RECONNECT:
            pass  # still running as far as the API is concerned; the log says why
        else:
            fields = {"STATUS": status, "FINISHED_AT": now_iso()}
            if error:
                fields["ERROR"] = error
            if status == job_runner.COMPLETED:
                fields["PROGRESS"] = 100
            _update_job(project_id, job_id, **fields)
            append_log(project_id, job_id, "[{}] job {}{}".format(now_iso(), status, ": " + error if error else ""))
            self._forget(job_id)
            # A step of a multi-run stage: its parent decides what comes next.
            row = _fetch_job_row(project_id, job_id)
            if row is not None and row["PARENT_JOB_ID"]:
                driver = _driver_for_parent(project_id, row["PARENT_JOB_ID"])
                if driver is not None:
                    driver.child_finished(project_id, row, status, error)

    def on_log(self, job_id, text, level="info"):
        project_id = self._project(job_id)
        if project_id is None:
            return
        append_log(project_id, job_id, "[{}] {}{}".format(now_iso(), "ERROR: " if level == "error" else "", text))

    def on_workers(self, job_id, connected, expected):
        self.on_log(job_id, "{} / {} processes connected".format(connected, expected))

    def on_task_progress(self, job_id, task, ref, result_number, expected, result):
        """Intermediate results are only kept for adapters that ask
        (`on_task_progress` on the adapter -- refine_ctf, whose refined
        defocus values come no other way); the job's adapter and sent task
        list are looked up once and cached, since one job can send one of
        these per particle."""
        project_id = self._project(job_id)
        if project_id is None:
            return
        with self._lock:
            cached = self._progress.get(job_id)
        if cached is None:
            row = _fetch_job_row(project_id, job_id)
            adapter = stages.ADAPTERS.get(row["STAGE"]) if row is not None else None
            hook = getattr(adapter, "on_task_progress", None)
            tasks = {t["index"]: t for t in (json.loads(row["TASKS_JSON"]) if row is not None and row["TASKS_JSON"] else [])}
            cached = (hook, tasks)
            with self._lock:
                self._progress[job_id] = cached
        hook, tasks = cached
        if hook is None or task not in tasks:
            return
        try:
            hook(project_id, tasks[task], result)
        except Exception as exc:  # noqa: BLE001
            self.on_log(job_id, "could not record an intermediate result of task {}: {}".format(task, exc), level="error")

    def on_task_done(self, job_id, task, ref, status, result, error, cpu_ms, done_count, task_count):
        project_id = self._project(job_id)
        if project_id is None:
            return
        conn = db.get_conn(project_id)
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO JOB_TASKS(JOB_ID, TASK_INDEX, REF, STATUS, CPU_MS, ERROR, RESULT_JSON, "
                "FINISHED_AT) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (job_id, task, None if ref is None else str(ref), status, cpu_ms, error,
                 json.dumps(result) if result is not None else None, now_iso()),
            )
            conn.execute("UPDATE JOBS SET PROGRESS=? WHERE JOB_ID=?",
                         (int(done_count * 100 / task_count) if task_count else 0, job_id))
        parent = conn.execute("SELECT p.JOB_ID, p.STAGE FROM JOBS c JOIN JOBS p ON p.JOB_ID = c.PARENT_JOB_ID WHERE c.JOB_ID=?", (job_id,)).fetchone()
        if parent is not None and parent["STAGE"] in DRIVERS:
            DRIVERS[parent["STAGE"]].child_progress(conn, parent["JOB_ID"], job_id, done_count, task_count)
        conn.close()
        if status == "failed":
            self.on_log(job_id, "task {} failed: {}".format(task, error), level="error")

    def on_job_done(self, job_id, status, cpu_ms, tasks_ok, tasks_failed, error=None):
        project_id = self._project(job_id)
        if project_id is None:
            return
        self.on_log(job_id, "controller reports {}: {} ok, {} failed, {:.1f} CPU-hours".format(
            status, tasks_ok, tasks_failed, cpu_ms / 3600000.0))
        metrics = {"cpu_ms": cpu_ms, "tasks_ok": tasks_ok, "tasks_failed": tasks_failed}
        conn = db.get_conn(project_id)
        row = conn.execute("SELECT * FROM JOBS WHERE JOB_ID=?", (job_id,)).fetchone()
        adapter = stages.ADAPTERS.get(row["STAGE"]) if row is not None else None
        if adapter is not None and tasks_ok:
            job = _row_to_job(row)
            sent_tasks = json.loads(row["TASKS_JSON"]) if row["TASKS_JSON"] else []
            task_rows = conn.execute("SELECT * FROM JOB_TASKS WHERE JOB_ID=? ORDER BY TASK_INDEX", (job_id,)).fetchall()
            # The adapter logs on *this* connection: it holds the write
            # transaction, and a second connection would block on it.
            def log_here(text, level="info"):
                append_log(project_id, job_id, "[{}] {}{}".format(
                    now_iso(), "ERROR: " if level == "error" else "", text), conn=conn)

            try:
                summary = adapter.finalize(conn, project_id, job, sent_tasks, task_rows, log_here)
                metrics.update(summary)
                self.on_log(job_id, adapter.describe_summary(summary) if hasattr(adapter, "describe_summary")
                            else "wrote results to the project database: {}".format(summary))
            except Exception as exc:  # noqa: BLE001
                self.on_log(job_id, "could not write results to the project database: {}".format(exc), level="error")
        # cisTEM adds the controller's timing to the project's CPU-hours total.
        with conn:
            conn.execute("UPDATE MASTER_SETTINGS SET TOTAL_CPU_HOURS = COALESCE(TOTAL_CPU_HOURS, 0) + ?, "
                         "TOTAL_JOBS_RUN = COALESCE(TOTAL_JOBS_RUN, 0) + 1 WHERE NUMBER=1", (cpu_ms / 3600000.0,))
            conn.execute("UPDATE JOBS SET METRICS_JSON=? WHERE JOB_ID=?", (json.dumps(metrics), job_id))
        conn.close()

    def on_controller_seq(self, job_id, seq):
        project_id = self._project(job_id)
        if project_id is not None:
            _update_job(project_id, job_id, CONTROLLER_SEQ=seq)


_db_sink = DbSink()


def _driver_for_parent(project_id, parent_id):
    row = _fetch_job_row(project_id, parent_id)
    return DRIVERS.get(row["STAGE"]) if row is not None else None


def start_job_runner():
    """Start the listener unless disabled. A bind failure (port taken --
    e.g. a second server on the same machine) is logged and leaves the
    server running in simulation-only mode rather than refusing to start."""
    global _job_runner
    if os.environ.get("JOB_RUNNER_ENABLED", "1") in ("0", "false", "no"):
        print("job runner disabled (JOB_RUNNER_ENABLED=0); jobs will be simulated")
        return None
    hosts = os.environ.get("JOB_RUNNER_HOSTS")
    runner = job_runner.JobRunner(
        _db_sink,
        bind_host=os.environ.get("JOB_RUNNER_BIND", "0.0.0.0"),
        port=int(os.environ.get("JOB_RUNNER_PORT", "8010")),
        advertise_hosts=[h.strip() for h in hosts.split(",") if h.strip()] if hosts else None,
        controller_executable=CONTROLLER_COMMAND,
        server_info={"name": "cistem3-server", "version": _git(["describe", "--always", "--dirty"]) or "unknown"},
    )
    try:
        runner.start()
    except OSError as exc:
        print("job runner could not listen on port {}: {} -- jobs will be simulated".format(runner.port, exc))
        return None
    _job_runner = runner
    if not _controller_available():
        print("job runner listening, but '{}' is not on PATH -- jobs will be simulated until it is "
              "(set CISTEM_JOB_CONTROLLER)".format(CONTROLLER_COMMAND))
    return runner


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# TODO: point these at your real pipeline. Each entry is a template command
# (list of args, Python str.format placeholders filled from the job's
# `params`) run via subprocess. If the named binary isn't on PATH, the job
# runs in SIMULATION mode instead (sleeps, logs progress, fabricates
# plausible output) so you can exercise the page end-to-end before your
# pipeline is wired up.
# ---------------------------------------------------------------------------

STAGE_COMMANDS = {
    "motion_correction": {"binary": "MotionCor2", "command": None},
    "ctf_estimation": {"binary": "ctffind", "command": None},
    "particle_picking": {"binary": "relion_autopick", "command": None},
    "class2d": {"binary": "relion_refine", "command": None},
    "ab_initio_3d": {"binary": "refine3d", "command": None},
    "refine3d": {"binary": "relion_refine", "command": None},
    "auto_refine3d": {"binary": "refine3d", "command": None},
    "refine_ctf": {"binary": "refine_ctf", "command": None},
    "generate3d": {"binary": "reconstruct3d", "command": None},
}


# ---------------------------------------------------------------------------
# Job store helpers (SQLite-backed, one project's JOBS/JOB_LOG_LINES tables)
# ---------------------------------------------------------------------------

def _task_progress(conn, row):  # noqa: D401 -- see _task_progress_for
    """How far a runner-backed job has got, for the page's time-remaining
    estimate (cisTEM's JobTracker: seconds per task so far times tasks left):
    tasks finished out of tasks sent, and when the first and latest finished.
    Nothing for a simulated job -- it has no tasks, only a percentage."""
    if not row["TASKS_JSON"]:
        return {}
    task_count = len(json.loads(row["TASKS_JSON"]))
    done, first, latest = conn.execute(
        "SELECT COUNT(*), MIN(FINISHED_AT), MAX(FINISHED_AT) FROM JOB_TASKS WHERE JOB_ID=? AND STATUS IN ('ok','failed')",
        (row["JOB_ID"],)).fetchone()
    return {"task_count": task_count, "tasks_done": done, "first_task_finished_at": first, "last_task_finished_at": latest}


def _row_to_job(row, conn=None):
    """`conn`, when given, adds the task counts a running job's time-remaining
    estimate needs; the single-job routes don't bother."""
    return dict({
        "id": row["JOB_ID"],
        "stage": row["STAGE"],
        "number": row["JOB_NUMBER"],
        "name": row["NAME"],
        "params": json.loads(row["PARAMS_JSON"]) if row["PARAMS_JSON"] else {},
        "status": row["STATUS"],
        "progress": row["PROGRESS"],
        "created_at": row["CREATED_AT"],
        "started_at": row["STARTED_AT"],
        "finished_at": row["FINISHED_AT"],
        "error": row["ERROR"],
        "metrics": json.loads(row["METRICS_JSON"]) if row["METRICS_JSON"] else {},
        "cancel_requested": bool(row["CANCEL_REQUESTED"]),
        "actions": _job_actions(row),
    }, **(_task_progress_for(conn, row) if conn is not None and row["STATUS"] in ("queued", "running") else {}))


def _job_actions(row):
    """The buttons a running multi-run job offers besides Terminate (cisTEM's
    Take Current / Take Last Start, and Finish): from the driver's
    available_actions(state), [] for anything else."""
    driver = DRIVERS.get(row["STAGE"])
    if driver is None or row["STATUS"] not in ("queued", "running") or not row["STATE_JSON"] or not hasattr(driver, "available_actions"):
        return []
    try:
        return driver.available_actions(json.loads(row["STATE_JSON"]))
    except (ValueError, KeyError, TypeError):
        return []


def _task_progress_for(conn, row):
    if row["STAGE"] in DRIVERS:
        return DRIVERS[row["STAGE"]].progress_info(json.loads(row["STATE_JSON"]) if row["STATE_JSON"] else None)
    return _task_progress(conn, row)


def _fetch_job_row(project_id, job_id):
    conn = db.get_conn(project_id)
    row = conn.execute("SELECT * FROM JOBS WHERE JOB_ID=?", (job_id,)).fetchone()
    conn.close()
    return row


def _update_job(project_id, job_id, **fields):
    if not fields:
        return
    cols = ", ".join("{}=?".format(k) for k in fields)
    values = list(fields.values()) + [job_id]
    conn = db.get_conn(project_id)
    with conn:
        conn.execute("UPDATE JOBS SET {} WHERE JOB_ID=?".format(cols), values)
    conn.close()


def append_log(project_id, job_id, line, conn=None):
    """Append one line to the job log. Pass `conn` to write on a connection
    that already holds a transaction -- opening a second connection to the
    same file from inside one deadlocks on SQLite's write lock (and then
    fails after busy_timeout), which is exactly what a stage adapter's
    progress logging inside finalize() would otherwise do."""
    if conn is not None:
        _insert_log_line(conn, job_id, line)
        return
    conn = db.get_conn(project_id)
    with conn:
        _insert_log_line(conn, job_id, line)
    conn.close()


def _insert_log_line(conn, job_id, line):
    seq = conn.execute(
        "SELECT COALESCE(MAX(SEQ), -1) + 1 FROM JOB_LOG_LINES WHERE JOB_ID=?", (job_id,)
    ).fetchone()[0]
    conn.execute("INSERT INTO JOB_LOG_LINES(JOB_ID, SEQ, LINE) VALUES (?, ?, ?)", (job_id, seq, line))


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------

class JobCancelled(Exception):
    pass


def _check_cancelled(job_id):
    with _live_lock:
        info = _live.get(job_id)
    if info and info.get("cancel_requested"):
        raise JobCancelled()


def run_job(project_id, job_id):
    job = _row_to_job(_fetch_job_row(project_id, job_id))

    _update_job(project_id, job_id, STATUS="running", STARTED_AT=now_iso())
    append_log(project_id, job_id, "[{}] job started (stage: {})".format(now_iso(), job["stage"]))

    stage_cfg = STAGE_COMMANDS.get(job["stage"], {})
    binary = stage_cfg.get("binary")
    command_template = stage_cfg.get("command")
    have_binary = bool(binary) and shutil.which(binary) is not None

    try:
        if have_binary and command_template:
            _run_real(project_id, job, command_template)
        else:
            _run_simulated(project_id, job, binary)
    except JobCancelled:
        _update_job(project_id, job_id, STATUS="cancelled", FINISHED_AT=now_iso())
        append_log(project_id, job_id, "[{}] job cancelled".format(now_iso()))
        with _live_lock:
            _live.pop(job_id, None)
        return
    except Exception as exc:  # noqa: BLE001
        _update_job(project_id, job_id, STATUS="failed", ERROR=str(exc), FINISHED_AT=now_iso())
        append_log(project_id, job_id, "[{}] job failed: {}".format(now_iso(), exc))
        with _live_lock:
            _live.pop(job_id, None)
        return

    if job["stage"] == "motion_correction":
        try:
            _write_motion_correction_results(project_id, job)
        except Exception as exc:  # noqa: BLE001
            append_log(
                project_id, job_id,
                "[{}] warning: could not write results to project database: {}".format(now_iso(), exc),
            )

    _update_job(project_id, job_id, STATUS="completed", PROGRESS=100, FINISHED_AT=now_iso())
    append_log(project_id, job_id, "[{}] job completed".format(now_iso()))
    with _live_lock:
        _live.pop(job_id, None)


def _run_real(project_id, job, command_template):
    """Fill the command template from job params and run it, streaming
    stdout/stderr into the job's log."""
    params = dict(job["params"] or {})
    try:
        cmd = [str(part).format(**params) for part in command_template]
    except KeyError as exc:
        raise RuntimeError("missing parameter for command template: {}".format(exc))

    append_log(project_id, job["id"], "$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    with _live_lock:
        _live.setdefault(job["id"], {})["proc"] = proc

    for line in proc.stdout:
        append_log(project_id, job["id"], line.rstrip("\n"))
        _check_cancelled(job["id"])

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("exited with code {}".format(proc.returncode))


def _run_simulated(project_id, job, binary):
    """No real binary found -- walk the job through a plausible progress
    sequence so the front end has something to show."""
    note = (
        "'{}' not found on PATH -- simulating this job. ".format(binary)
        if binary
        else "no command configured for this stage -- simulating. "
    )
    append_log(project_id, job["id"], note + "Edit STAGE_COMMANDS in cistem_server.py to run the real thing.")
    steps = [10, 25, 45, 65, 85, 100]
    for pct in steps:
        _check_cancelled(job["id"])
        time.sleep(0.8)
        _update_job(project_id, job["id"], PROGRESS=pct)
        append_log(project_id, job["id"], "[{}] progress: {}%".format(now_iso(), pct))
    metrics = _fake_metrics(job["stage"])
    _update_job(project_id, job["id"], METRICS_JSON=json.dumps(metrics))


def _fake_metrics(stage):
    """Placeholder result numbers per stage, purely so the UI has something
    to render in simulation mode. Replace by parsing your real tool's
    output in _run_real() once STAGE_COMMANDS is wired up."""
    import random

    if stage == "motion_correction":
        return {"avg_motion_px": round(random.uniform(0.8, 2.4), 2)}
    if stage == "ctf_estimation":
        return {"avg_ctf_fit_a": round(random.uniform(2.8, 4.5), 2)}
    if stage == "particle_picking":
        return {"particles_picked": random.randint(20000, 120000)}
    if stage == "class2d":
        return {"classes_retained": random.randint(15, 40)}
    if stage == "refine3d":
        return {"resolution_a": round(random.uniform(2.2, 3.8), 2)}
    return {}


def _write_motion_correction_results(project_id, job):
    """On a completed Align Movies job, write one MOVIE_ALIGNMENT_LIST +
    IMAGE_ASSETS row per movie in the job's group -- the same finalization
    real cisTEM does in WriteResultToDataBase() once a job batch completes."""
    params = job["params"] or {}
    movie_group_id = params.get("movie_group_id")
    # cisTEM owns its project directory and writes aligned sums into
    # Assets/Images; so does this, rather than asking for a path on the form.
    output_dir = db.project_dir(project_id) / "Assets" / "Images"
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = db.get_conn(project_id)
    with conn:
        movies = conn.execute(
            "SELECT ma.* FROM MOVIE_ASSETS ma "
            "JOIN MOVIE_GROUP_MEMBERS gm ON gm.MOVIE_ASSET_ID = ma.MOVIE_ASSET_ID "
            "WHERE gm.GROUP_ID = ?",
            (movie_group_id,),
        ).fetchall()

        now = db.now_epoch()
        for movie in movies:
            binning = movie["OUTPUT_BINNING_FACTOR"] or 1.0
            final_pixel_size = (movie["PIXEL_SIZE"] or 0.0) * binning
            output_file = str(output_dir / "{}_aligned.mrc".format(movie["NAME"]))

            cur = conn.execute(
                "INSERT INTO MOVIE_ALIGNMENT_LIST("
                "DATETIME_OF_RUN, ALIGNMENT_JOB_ID, MOVIE_ASSET_ID, OUTPUT_FILE, VOLTAGE, "
                "PIXEL_SIZE, EXPOSURE_PER_FRAME, PRE_EXPOSURE_AMOUNT, MIN_SHIFT, MAX_SHIFT, "
                "SHOULD_DOSE_FILTER, SHOULD_RESTORE_POWER, TERMINATION_THRESHOLD, MAX_ITERATIONS, "
                "BFACTOR, SHOULD_MASK_CENTRAL_CROSS, HORIZONTAL_MASK, VERTICAL_MASK, "
                "SHOULD_INCLUDE_ALL_FRAMES_IN_SUM, FIRST_FRAME_TO_SUM, LAST_FRAME_TO_SUM, "
                "FINAL_PIXEL_SIZE) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    now, job["id"], movie["MOVIE_ASSET_ID"], output_file, movie["VOLTAGE"],
                    movie["PIXEL_SIZE"], movie["DOSE_PER_FRAME"], 0.0,
                    params.get("min_shift_a"), params.get("max_shift_a"),
                    1 if params.get("should_dose_filter") else 0,
                    1 if params.get("should_restore_power") else 0,
                    params.get("termination_threshold_a"), params.get("max_iterations"),
                    params.get("bfactor_a2"),
                    1 if params.get("mask_central_cross") else 0,
                    params.get("horizontal_mask_px"), params.get("vertical_mask_px"),
                    1 if params.get("include_all_frames") else 0,
                    params.get("first_frame"), params.get("last_frame"),
                    final_pixel_size,
                ),
            )
            alignment_id = cur.lastrowid

            cur = conn.execute(
                "INSERT INTO IMAGE_ASSETS("
                "NAME, FILENAME, POSITION_IN_STACK, PARENT_MOVIE_ID, ALIGNMENT_ID, "
                "X_SIZE, Y_SIZE, PIXEL_SIZE, VOLTAGE, SPHERICAL_ABERRATION, PROTEIN_IS_WHITE) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    movie["NAME"] + "_aligned", output_file, 1, movie["MOVIE_ASSET_ID"], alignment_id,
                    movie["X_SIZE"], movie["Y_SIZE"], final_pixel_size, movie["VOLTAGE"],
                    movie["SPHERICAL_ABERRATION"], movie["PROTEIN_IS_WHITE"],
                ),
            )
            # An aligned micrograph is an image asset like any other, so it
            # joins All Images here rather than waiting for db.py's backfill
            # to notice it on the next connection.
            conn.execute(
                "INSERT OR IGNORE INTO IMAGE_GROUP_MEMBERS(GROUP_ID, IMAGE_ASSET_ID) VALUES (0, ?)",
                (cur.lastrowid,),
            )

        conn.execute(
            "UPDATE MASTER_SETTINGS SET TOTAL_JOBS_RUN = COALESCE(TOTAL_JOBS_RUN, 0) + 1 WHERE NUMBER=1"
        )
    conn.close()


def _recover_interrupted_jobs():
    """Jobs still 'queued'/'running' when the server last stopped.

    A job that went through the job runner has a token and its task list in
    its row, so -- per docs/job-protocol.md section 7.2 -- it is handed back
    to the runner as *awaiting reconnect*: its controller, if still alive,
    dials in again and carries on. A simulated job has no such thing (its
    thread died with the process) and is marked failed as before.
    """
    if not db.PROJECTS_ROOT.is_dir():
        return
    for entry in db.PROJECTS_ROOT.iterdir():
        if not (entry / "project.db").is_file():
            continue
        project_id = entry.name
        conn = db.get_conn(project_id)
        rows = conn.execute("SELECT * FROM JOBS WHERE STATUS IN ('queued','running')").fetchall()
        parents = []
        for row in rows:
            job_id = row["JOB_ID"]
            adapter = stages.ADAPTERS.get(row["STAGE"])
            if row["STAGE"] in DRIVERS and _job_runner is not None and row["STATE_JSON"]:
                # Its children are restored by the branch below; it picks
                # up from whichever of them is (or has already) finished.
                parents.append(row)
                continue
            if _job_runner is not None and adapter is not None and row["JOB_TOKEN"] and row["TASKS_JSON"]:
                params = json.loads(row["PARAMS_JSON"]) if row["PARAMS_JSON"] else {}
                sys_conn = db.get_system_conn()
                profile = db.load_run_profile_by_name(sys_conn, params.get("run_profile")) or {
                    "name": params.get("run_profile") or "?", "manager_command": "$command",
                    "controller_address": "", "run_commands": [], "total_jobs": 0}
                spec = job_runner.JobSpec(
                    job_id, _package_job_info(project_id, row), adapter.PROGRAM, profile,
                    json.loads(row["TASKS_JSON"]), profile["manager_command"], token=row["JOB_TOKEN"],
                    forward_progress=getattr(adapter, "WANTS_TASK_PROGRESS", False),
                    controller_log=_controller_log_path(project_id, job_id))
                sys_conn.close()
                done = [r["TASK_INDEX"] for r in conn.execute(
                    "SELECT TASK_INDEX FROM JOB_TASKS WHERE JOB_ID=?", (job_id,)).fetchall()]
                _db_sink.register(job_id, project_id)
                _job_runner.restore(spec, row["CONTROLLER_SEQ"] or 0, done)
                continue
            with conn:
                conn.execute(
                    "UPDATE JOBS SET STATUS='failed', "
                    "ERROR='Server restarted while this job was in progress', FINISHED_AT=? WHERE JOB_ID=?",
                    (now_iso(), job_id),
                )
        conn.close()
        for row in parents:
            DRIVERS[row["STAGE"]].resume(project_id, row)


def _package_job_info(project_id, row):
    """package.job (docs/job-protocol.md section 6.2) from a JOBS row."""
    return {"id": row["JOB_ID"], "number": row["JOB_NUMBER"], "name": row["NAME"], "project": project_id}


def _controller_log_path(project_id, job_id):
    """Where a job's controller writes its stdout/stderr: alongside the
    project, so it survives the server and is findable afterwards."""
    return str(db.project_dir(project_id) / "Logs" / "{}_controller.log".format(job_id))


# ---------------------------------------------------------------------------
# Static frontend -- optional convenience so `python cistem_server.py` alone
# is enough to try the app: open http://localhost:8000/ instead of finding
# and double-clicking cistem3.html yourself. cistem3.html still works fine
# opened directly (file://) or served by anything else; this just adds one
# more way to reach it. Only this explicit allowlist of sibling files is
# served, never an arbitrary path under REPO_ROOT -- that would expose
# server/data/auth.db and the rest of the repo.
# ---------------------------------------------------------------------------

STATIC_FILES = {"cistem3.html", "config.js", "logo.png", "movie-alignment-example.png",
                "ctffind-definitions.png", "ctffind-diagnostic-image.png", "ctffind-example-1dfit.png",
                "class2d-example-1.png", "class2d-example-2.png", "class2d-example-3.png", "class2d-example-4.png",
                "abinitio-example.png", "refine3d-strategy.png"}


@app.route("/")
def index():
    return send_from_directory(REPO_ROOT, "cistem3.html")


@app.route("/<path:filename>")
def static_file(filename):
    if filename not in STATIC_FILES:
        abort(404)
    return send_from_directory(REPO_ROOT, filename)


# ---------------------------------------------------------------------------
@app.route("/api/users/<int:user_id>", methods=["DELETE"])
@auth.admin_required
def delete_user_route(user_id):
    """Remove an account. Refused for your own account and for the last
    admin. An account that owns projects can't simply vanish and leave them
    ownerless (only admins would see them): without `transfer_to` the
    request is answered 409 with the projects listed, and with
    `transfer_to` (another user's id) they are handed over first."""
    me = g.current_user
    target = auth.get_user_by_id(user_id)
    if target is None:
        return jsonify({"error": "user not found"}), 404
    if user_id == me["id"]:
        return jsonify({"error": "you can't delete your own account -- have another admin do it"}), 400
    if target["role"] == "admin" and auth.admin_count() <= 1:
        return jsonify({"error": "that is the only admin account; promote someone else first"}), 400
    owned = [p for p in db.list_projects() if p["owner_user_id"] == user_id]
    body = request.get_json(force=True, silent=True) or {}
    transfer_to = body.get("transfer_to")
    recipient = None
    if owned:
        if transfer_to in (None, ""):
            return jsonify({"error": "{} owns {} project{}; say who gets them (transfer_to) or delete them first".format(
                target["username"], len(owned), "" if len(owned) == 1 else "s"),
                "owned_project_count": len(owned), "projects": [{"id": p["id"], "name": p["name"]} for p in owned]}), 409
        try:
            recipient = auth.get_user_by_id(int(transfer_to))
        except (TypeError, ValueError):
            recipient = None
        if recipient is None or recipient["id"] == user_id:
            return jsonify({"error": "transfer_to must be another existing user"}), 400
        for p in owned:
            db.set_project_owner(p["id"], recipient["id"], recipient["username"])
    try:
        auth.delete_user(user_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify({"ok": True, "deleted": target["username"], "transferred": len(owned),
                    "transferred_to": recipient["username"] if recipient else None})


# Project routes
# ---------------------------------------------------------------------------

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "time": now_iso()})


@app.route("/api/version")
def version_route():
    commit = _git(["rev-parse", "--short", "HEAD"])
    if commit and _git(["status", "--porcelain"]):
        commit += "-dirty"
    return jsonify({
        "commit": commit,
        "commit_datetime": _git(["log", "-1", "--format=%ci"]),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
    })


MOVIE_EXTENSIONS = {".mrc", ".mrcs", ".tif", ".tiff", ".eer"}
# What each Import dialog's "Browse..." picker will show. Images never
# include .eer -- see IMAGE_IMPORT_EXTENSIONS, which the import route
# enforces on the resolved files regardless of what the picker offered.
BROWSABLE_EXTENSIONS = {"movie": MOVIE_EXTENSIONS, "image": {".mrc", ".mrcs", ".tif", ".tiff"}, "volume": {".mrc", ".mrcs"},
                        "text": {".txt", ".plt", ".dat", ".coords", ".box", ".csv"}}


@app.route("/api/browse")
@auth.login_required
def browse_filesystem():
    """Lists one directory on this server's own filesystem, for the "Browse..."
    picker on the Import Movies and Import Images dialogs. `types` (movie /
    image) picks which extensions are listed; anything else falls back to
    movies. Deliberately the server's filesystem, not the browser's: a
    hand-typed glob in that same field is already resolved against this
    machine (see import_movies()), so this is just a friendlier way to fill
    it in -- not a new capability an authenticated user didn't already have.
    """
    extensions = BROWSABLE_EXTENSIONS.get(request.args.get("types"), MOVIE_EXTENSIONS)
    raw_path = request.args.get("path") or str(Path.home())
    path = Path(raw_path).expanduser()
    if not path.is_dir():
        return jsonify({"error": "not a directory: {}".format(path)}), 400
    try:
        entries = list(path.iterdir())
    except OSError as exc:
        return jsonify({"error": str(exc)}), 400

    directories = []
    files = []
    for entry in entries:
        if entry.name.startswith("."):
            continue
        try:
            is_dir = entry.is_dir()
        except OSError:
            continue
        if is_dir:
            directories.append(entry.name)
        elif entry.suffix.lower() in extensions:
            files.append(entry.name)
    directories.sort(key=str.lower)
    files.sort(key=str.lower)

    resolved = path.resolve()
    parent = resolved.parent
    return jsonify({
        "path": str(resolved),
        "parent": str(parent) if parent != resolved else None,
        "directories": directories,
        "files": files,
    })


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.route("/api/auth/login", methods=["POST"])
def login_route():
    body = request.get_json(force=True, silent=True) or {}
    user = auth.authenticate(body.get("username"), body.get("password"))
    if user is None:
        return jsonify({"error": "invalid username or password"}), 401
    token = auth.create_session(user["id"])
    return jsonify({"token": token, "user": user})


@app.route("/api/auth/logout", methods=["POST"])
def logout_route():
    token = auth.get_bearer_token()
    if token:
        auth.revoke_session(token)
    return jsonify({"ok": True})


@app.route("/api/auth/me")
@auth.login_required
def me_route():
    return jsonify({"user": g.current_user})


@app.route("/api/auth/change-password", methods=["POST"])
@auth.login_required
def change_password_route():
    body = request.get_json(force=True, silent=True) or {}
    if not auth.verify_password(g.current_user["id"], body.get("current_password")):
        return jsonify({"error": "current password is incorrect"}), 400
    try:
        auth.set_password(g.current_user["id"], body.get("new_password"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    # Keep the session that made this request alive; sign the account out
    # everywhere else, same idea as "log out other devices" after a change.
    auth.revoke_user_sessions(g.current_user["id"], keep_token=auth.get_bearer_token())
    return jsonify({"ok": True})


@app.route("/api/users", methods=["GET"])
@auth.admin_required
def list_users_route():
    return jsonify({"users": auth.list_users()})


@app.route("/api/users", methods=["POST"])
@auth.admin_required
def create_user_route():
    body = request.get_json(force=True, silent=True) or {}
    try:
        user = auth.create_user(
            body.get("username"), body.get("password"), body.get("role"),
            display_name=body.get("display_name"), created_by_user_id=g.current_user["id"],
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(user), 201


@app.route("/api/users/<int:user_id>/reset-password", methods=["POST"])
@auth.admin_required
def reset_password_route(user_id):
    body = request.get_json(force=True, silent=True) or {}
    try:
        auth.set_password(user_id, body.get("new_password"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    # No session of the target user's to preserve -- an admin-driven reset
    # signs them out everywhere, forcing a fresh login with the new password.
    auth.revoke_user_sessions(user_id)
    return jsonify({"ok": True})


@app.route("/api/users/<int:user_id>", methods=["PATCH"])
@auth.admin_required
def update_user_route(user_id):
    """Change an account's role. Two refusals keep an admin from locking
    everyone out: you can't demote yourself, and you can't demote the last
    admin (the same thing, seen from the other side)."""
    body = request.get_json(force=True, silent=True) or {}
    if "role" not in body:
        return jsonify({"error": "nothing to change: body needs 'role'"}), 400
    target = auth.get_user_by_id(user_id)
    if target is None:
        return jsonify({"error": "user not found"}), 404
    if body["role"] != "admin" and target["role"] == "admin":
        if user_id == g.current_user["id"]:
            return jsonify({"error": "you can't remove your own admin role -- have another admin do it"}), 400
        if auth.admin_count() <= 1:
            return jsonify({"error": "that is the only admin account; promote someone else first"}), 400
    try:
        return jsonify(auth.set_role(user_id, body["role"]))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


# ---------------------------------------------------------------------------
# Project routes
# ---------------------------------------------------------------------------

@app.route("/api/projects", methods=["GET"])
@auth.login_required
def list_projects_route():
    owner_user_id = None if g.current_user["role"] == "admin" else g.current_user["id"]
    return jsonify({"projects": db.list_projects(owner_user_id=owner_user_id)})


@app.route("/api/projects", methods=["POST"])
@auth.login_required
def create_project_route():
    body = request.get_json(force=True, silent=True) or {}
    try:
        project_id = db.create_project(
            body.get("name"), g.current_user["id"], g.current_user["username"]
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(db.get_project_summary(project_id)), 201


@app.route("/api/projects/<project_id>")
@auth.project_access_required
def get_project_route(project_id):
    return jsonify(db.get_project_summary(project_id))


@app.route("/api/projects/<project_id>", methods=["DELETE"])
@auth.project_access_required
def delete_project_route(project_id):
    db.delete_project(project_id)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Asset routes (project-scoped)
#
# Movies and images are the same object with different metadata: a named file
# on disk, in one master group plus any number of user-made groups, listed /
# grouped / removed identically. So everything that doesn't depend on *which*
# metadata a kind carries -- grouping, membership, deletion, the
# already-imported check -- is written once against an AssetKind and shared,
# the way cisTEM hangs MyMovieAssetPanel and MyImageAssetPanel off a common
# MyAssetPanelParent. Only import and preview stay per-kind, because that is
# exactly where the two genuinely differ (frames, dose, gain/dark and EER are
# movie-only; parent movie and alignment are image-only).
# ---------------------------------------------------------------------------

class AssetKind:
    """Table and response-key names for one kind of asset."""

    def __init__(self, noun, asset_table, id_column, group_table, member_table, all_group_name):
        self.noun = noun  # "movie" / "image"
        self.plural = noun + "s"
        self.asset_table = asset_table
        self.id_column = id_column
        self.group_table = group_table
        self.member_table = member_table
        self.all_group_name = all_group_name

    # Response keys, kept in the shape the frontend already reads:
    # {"movies": [...]}, {"movie_groups": [...]}, {"movie_count": n}, and
    # {"movie_ids": [...]} on the way in.
    @property
    def list_key(self):
        return self.plural

    @property
    def groups_key(self):
        return self.noun + "_groups"

    @property
    def count_key(self):
        return self.noun + "_count"

    @property
    def ids_key(self):
        return self.noun + "_ids"


MOVIE_KIND = AssetKind("movie", "MOVIE_ASSETS", "MOVIE_ASSET_ID",
                       "MOVIE_GROUP_LIST", "MOVIE_GROUP_MEMBERS", "All Movies")
IMAGE_KIND = AssetKind("image", "IMAGE_ASSETS", "IMAGE_ASSET_ID",
                       "IMAGE_GROUP_LIST", "IMAGE_GROUP_MEMBERS", "All Images")
# Particle positions (cisTEM's MyParticlePositionAssetPanel): one asset per
# picked particle, written by Find Particles. Same group machinery.
POSITION_KIND = AssetKind("particle_position", "PARTICLE_POSITION_ASSETS", "PARTICLE_POSITION_ASSET_ID",
                          "PARTICLE_POSITION_GROUP_LIST", "PARTICLE_POSITION_GROUP_MEMBERS", "All Particle Positions")

# 3D volumes (cisTEM's MyVolumeAssetPanel): reconstructions written by
# Ab-Initio 3D (and, later, Refine 3D). Same group machinery.
VOLUME_KIND = AssetKind("volume", "VOLUME_ASSETS", "VOLUME_ASSET_ID",
                        "VOLUME_GROUP_LIST", "VOLUME_GROUP_MEMBERS", "All Volumes")

# Group 0 is the master list db.py seeds into every project -- every asset is
# a member of it, and the app treats it as the source of truth for what
# exists. Renaming, deleting or inverting it would leave the project without
# one, so the routes below refuse it for both kinds.
ALL_GROUP_ID = 0

# Images are 2D micrographs, so unlike movies they are never EER: an EER file
# is a raw movie container by construction. Cf. imageheaders.read_image_header().
IMAGE_IMPORT_EXTENSIONS = {".mrc", ".mrcs", ".tif", ".tiff"}
# MyVolumeImportDialog's file filter: "MRC files (*.mrc)|*.mrc;*.mrcs".
VOLUME_IMPORT_EXTENSIONS = {".mrc", ".mrcs"}


def _canonical_path(path):
    """One spelling per file, so the same file reached by a different route
    (relative vs absolute, a symlinked share, a trailing /./) is recognised as
    already imported rather than added twice."""
    try:
        return str(Path(path).expanduser().resolve())
    except OSError:
        return str(Path(path).expanduser())


def _imported_paths(conn, kind):
    """Canonical paths of every asset of this kind already in the project, for
    the already-an-asset check (cf. IsFileAnAsset() in the reference dialogs).
    Rows imported before paths were canonicalised are folded in too."""
    rows = conn.execute(
        "SELECT FILENAME FROM {} WHERE FILENAME IS NOT NULL".format(kind.asset_table)
    ).fetchall()
    return {_canonical_path(r["FILENAME"]) for r in rows}


def _partition_matches(conn, kind, input_glob, allowed_extensions=None):
    """Splits the glob's matches into ones not yet imported and ones already
    present, both in sorted canonical form. allowed_extensions, when given,
    drops matches of any other type -- the image import uses it so a
    directory-wide glob can't pull an EER movie in as a micrograph."""
    already = _imported_paths(conn, kind)
    matched = set()
    for path in glob_module.glob(input_glob):
        if not Path(path).is_file():
            continue
        if allowed_extensions is not None and Path(path).suffix.lower() not in allowed_extensions:
            continue
        matched.add(_canonical_path(path))
    return sorted(matched - already), sorted(matched & already)


def _check_import(project_id, kind, allowed_extensions=None):
    """Resolves a glob, checks any referenced files exist, and reports how many
    matches are already imported, so an Import dialog can keep its Import
    button disabled until there's something new to import. The reference
    dialogs can stat files and consult their own asset list locally; both live
    on the server here, so the dialog has to ask. Exposes no more of the
    filesystem than /api/browse already does.
    """
    body = request.get_json(force=True, silent=True) or {}
    input_glob = (body.get("glob") or "").strip()

    new_paths, duplicate_paths = [], []
    if input_glob:
        conn = db.get_conn(project_id)
        new_paths, duplicate_paths = _partition_matches(
            conn, kind, input_glob, allowed_extensions=allowed_extensions
        )
        conn.close()

    files = {}
    for raw in body.get("files") or []:
        path = (raw or "").strip()
        if path:
            files[path] = Path(path).expanduser().is_file()

    return jsonify({
        "glob_match_count": len(new_paths) + len(duplicate_paths),
        "new_count": len(new_paths),
        "already_imported_count": len(duplicate_paths),
        "glob_has_eer": any(p.lower().endswith(".eer") for p in new_paths + duplicate_paths),
        "files": files,
    })


def _list_assets(project_id, kind):
    group_id = request.args.get("group_id", type=int)
    conn = db.get_conn(project_id)
    if group_id is None:
        rows = conn.execute(
            "SELECT * FROM {t} ORDER BY {id}".format(t=kind.asset_table, id=kind.id_column)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT a.* FROM {t} a JOIN {m} m ON m.{id} = a.{id} "
            "WHERE m.GROUP_ID = ? ORDER BY a.{id}".format(
                t=kind.asset_table, m=kind.member_table, id=kind.id_column
            ),
            (group_id,),
        ).fetchall()
    conn.close()
    return jsonify({kind.list_key: [dict(r) for r in rows]})


def _list_groups(project_id, kind):
    conn = db.get_conn(project_id)
    rows = [dict(r) for r in conn.execute(
        "SELECT g.GROUP_ID as group_id, g.GROUP_NAME as group_name, "
        "COUNT(m.{id}) as {count} "
        "FROM {g} g LEFT JOIN {m} m ON m.GROUP_ID = g.GROUP_ID "
        "GROUP BY g.GROUP_ID ORDER BY g.GROUP_ID".format(
            id=kind.id_column, count=kind.count_key, g=kind.group_table, m=kind.member_table
        )
    ).fetchall()]
    if kind is IMAGE_KIND:
        # How many members have an active CTF estimate: Find Particles needs
        # all of them to (cisTEM's can_be_picked), and the Actions panel
        # says so before the job is refused.
        with_ctf = {r[0]: r[1] for r in conn.execute(
            "SELECT m.GROUP_ID, COUNT(*) FROM IMAGE_GROUP_MEMBERS m "
            "JOIN IMAGE_ASSETS ia ON ia.IMAGE_ASSET_ID = m.IMAGE_ASSET_ID "
            "JOIN ESTIMATED_CTF_PARAMETERS ce ON ce.CTF_ESTIMATION_ID = ia.CTF_ESTIMATION_ID GROUP BY m.GROUP_ID")}
        for r in rows:
            r["images_with_ctf"] = with_ctf.get(r["group_id"], 0)
    conn.close()
    return jsonify({kind.groups_key: rows})


def _create_group(project_id, kind):
    body = request.get_json(force=True, silent=True) or {}
    group_name = (body.get("group_name") or "").strip()
    if not group_name:
        return jsonify({"error": "group_name is required"}), 400

    conn = db.get_conn(project_id)
    existing = conn.execute(
        "SELECT GROUP_ID FROM {} WHERE LOWER(GROUP_NAME) = LOWER(?)".format(kind.group_table),
        (group_name,),
    ).fetchone()
    if existing:
        conn.close()
        return jsonify({"error": 'a group named "{}" already exists'.format(group_name)}), 400
    with conn:
        cur = conn.execute(
            "INSERT INTO {}(GROUP_NAME, LIST_ID) VALUES (?, 0)".format(kind.group_table),
            (group_name,),
        )
        group_id = cur.lastrowid
    conn.close()
    return jsonify({"group_id": group_id, "group_name": group_name}), 201


def _rename_group(project_id, kind, group_id):
    if group_id == ALL_GROUP_ID:
        return jsonify({"error": "the {} group cannot be renamed".format(kind.all_group_name)}), 400
    body = request.get_json(force=True, silent=True) or {}
    group_name = (body.get("group_name") or "").strip()
    if not group_name:
        return jsonify({"error": "group_name is required"}), 400

    conn = db.get_conn(project_id)
    row = conn.execute(
        "SELECT GROUP_ID FROM {} WHERE GROUP_ID = ?".format(kind.group_table), (group_id,)
    ).fetchone()
    if row is None:
        conn.close()
        return jsonify({"error": "no such group"}), 404
    clash = conn.execute(
        "SELECT GROUP_ID FROM {} WHERE LOWER(GROUP_NAME) = LOWER(?) AND GROUP_ID != ?".format(
            kind.group_table
        ),
        (group_name, group_id),
    ).fetchone()
    if clash:
        conn.close()
        return jsonify({"error": 'a group named "{}" already exists'.format(group_name)}), 400
    with conn:
        conn.execute(
            "UPDATE {} SET GROUP_NAME = ? WHERE GROUP_ID = ?".format(kind.group_table),
            (group_name, group_id),
        )
    conn.close()
    return jsonify({"group_id": group_id, "group_name": group_name})


def _delete_group(project_id, kind, group_id):
    # Drops the group and its memberships only -- the assets themselves stay
    # in the project (they're still in the master group), the same distinction
    # the per-asset Remove makes between a group and the master list.
    if group_id == ALL_GROUP_ID:
        return jsonify({"error": "the {} group cannot be deleted".format(kind.all_group_name)}), 400

    conn = db.get_conn(project_id)
    row = conn.execute(
        "SELECT GROUP_ID FROM {} WHERE GROUP_ID = ?".format(kind.group_table), (group_id,)
    ).fetchone()
    if row is None:
        conn.close()
        return jsonify({"error": "no such group"}), 404
    with conn:
        conn.execute("DELETE FROM {} WHERE GROUP_ID = ?".format(kind.member_table), (group_id,))
        conn.execute("DELETE FROM {} WHERE GROUP_ID = ?".format(kind.group_table), (group_id,))
    conn.close()
    return jsonify({"ok": True})


def _invert_group(project_id, kind, group_id):
    """Replaces the group's membership with its complement against the master
    group: afterwards it holds exactly the assets it didn't hold before.

    Refused for group 0 for the same reason rename and delete are -- it is the
    master list, and its complement is the empty set, which would read as
    "this project has nothing in it".
    """
    if group_id == ALL_GROUP_ID:
        return jsonify({"error": "the {} group cannot be inverted".format(kind.all_group_name)}), 400

    conn = db.get_conn(project_id)
    row = conn.execute(
        "SELECT GROUP_ID FROM {} WHERE GROUP_ID = ?".format(kind.group_table), (group_id,)
    ).fetchone()
    if row is None:
        conn.close()
        return jsonify({"error": "no such group"}), 404

    with conn:
        before = {
            r[kind.id_column] for r in conn.execute(
                "SELECT {id} FROM {m} WHERE GROUP_ID = ?".format(
                    id=kind.id_column, m=kind.member_table
                ),
                (group_id,),
            )
        }
        every = {
            r[kind.id_column] for r in conn.execute(
                "SELECT {id} FROM {t}".format(id=kind.id_column, t=kind.asset_table)
            )
        }
        after = every - before
        conn.execute("DELETE FROM {} WHERE GROUP_ID = ?".format(kind.member_table), (group_id,))
        conn.executemany(
            "INSERT INTO {m}(GROUP_ID, {id}) VALUES (?, ?)".format(
                m=kind.member_table, id=kind.id_column
            ),
            [(group_id, asset_id) for asset_id in sorted(after)],
        )
    conn.close()
    return jsonify({"was": len(before), "now": len(after)})


def _requested_asset_ids(kind):
    body = request.get_json(force=True, silent=True) or {}
    return body.get(kind.ids_key) or []


def _delete_assets(project_id, kind):
    # A straightforward delete -- doesn't cascade-clean any downstream results
    # (e.g. the MOVIE_ALIGNMENT_LIST/IMAGE_ASSETS rows a completed Align
    # Movies job may already have written for these movies, or the images
    # those rows point at), matching this app's existing scope: only Align
    # Movies is wired to real project data end-to-end, and nothing here
    # reconciles derived results either.
    asset_ids = _requested_asset_ids(kind)
    if not asset_ids:
        return jsonify({"error": "{} is required".format(kind.ids_key)}), 400

    conn = db.get_conn(project_id)
    with conn:
        placeholders = ",".join("?" * len(asset_ids))
        conn.execute(
            "DELETE FROM {m} WHERE {id} IN ({p})".format(
                m=kind.member_table, id=kind.id_column, p=placeholders
            ),
            asset_ids,
        )
        cur = conn.execute(
            "DELETE FROM {t} WHERE {id} IN ({p})".format(
                t=kind.asset_table, id=kind.id_column, p=placeholders
            ),
            asset_ids,
        )
        deleted = cur.rowcount
    conn.close()
    return jsonify({"deleted": deleted})


def _remove_from_group(project_id, kind, group_id):
    # Unlinks assets from this one group only -- they stay in the master group
    # (and any other group they're a member of), unlike the delete route.
    # "Remove" while viewing the master group means removing the asset from
    # the project entirely, since that's the master list; that's what the
    # delete route is for, so group 0 isn't handled here.
    if group_id == ALL_GROUP_ID:
        return jsonify({
            "error": "group 0 is {} -- use the delete route to remove {} from the project".format(
                kind.all_group_name, kind.plural
            )
        }), 400
    asset_ids = _requested_asset_ids(kind)
    if not asset_ids:
        return jsonify({"error": "{} is required".format(kind.ids_key)}), 400

    conn = db.get_conn(project_id)
    with conn:
        placeholders = ",".join("?" * len(asset_ids))
        cur = conn.execute(
            "DELETE FROM {m} WHERE GROUP_ID = ? AND {id} IN ({p})".format(
                m=kind.member_table, id=kind.id_column, p=placeholders
            ),
            [group_id] + asset_ids,
        )
        removed = cur.rowcount
    conn.close()
    return jsonify({"removed": removed})


def _rename_asset(project_id, kind, asset_id):
    """MyAssetPanelParent::RenameAsset(): the display name only -- the file
    keeps its name, and nothing else refers to an asset by name."""
    body = request.get_json(force=True, silent=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name is required"}), 400
    conn = db.get_conn(project_id)
    try:
        row = conn.execute("SELECT * FROM {t} WHERE {id}=?".format(t=kind.asset_table, id=kind.id_column), (asset_id,)).fetchone()
        if row is None:
            return jsonify({"error": "no such {}".format(kind.noun)}), 404
        with conn:
            conn.execute("UPDATE {t} SET NAME=? WHERE {id}=?".format(t=kind.asset_table, id=kind.id_column), (name, asset_id))
        row = conn.execute("SELECT * FROM {t} WHERE {id}=?".format(t=kind.asset_table, id=kind.id_column), (asset_id,)).fetchone()
        return jsonify(dict(row))
    finally:
        conn.close()


def _add_to_group(project_id, kind):
    body = request.get_json(force=True, silent=True) or {}
    asset_ids = body.get(kind.ids_key) or []
    group_name = (body.get("group_name") or "").strip()
    if not asset_ids:
        return jsonify({"error": "{} is required".format(kind.ids_key)}), 400
    if not group_name:
        return jsonify({"error": "group_name is required"}), 400

    conn = db.get_conn(project_id)
    with conn:
        row = conn.execute(
            "SELECT GROUP_ID FROM {} WHERE LOWER(GROUP_NAME) = LOWER(?)".format(kind.group_table),
            (group_name,),
        ).fetchone()
        if row:
            group_id = row["GROUP_ID"]
            created = False
        else:
            cur = conn.execute(
                "INSERT INTO {}(GROUP_NAME, LIST_ID) VALUES (?, 0)".format(kind.group_table),
                (group_name,),
            )
            group_id = cur.lastrowid
            created = True

        for asset_id in asset_ids:
            conn.execute(
                "INSERT OR IGNORE INTO {m}(GROUP_ID, {id}) VALUES (?, ?)".format(
                    m=kind.member_table, id=kind.id_column
                ),
                (group_id, asset_id),
            )
    conn.close()
    return jsonify({"group_id": group_id, "group_name": group_name, "created": created})


def _preview_response(project_id, kind, asset_id, render, extra_headers=None):
    """Shared plumbing for the Display button: look the asset up, check the
    file is there and renderable, and serve the PNG with an ETag on the
    file's mtime+size so reopening it is a 304.

    Rendering takes ~100ms for a 300MB stack, so it's done inline rather than
    as a background job -- but it's deterministic for a given file, so the
    response is cached and revalidated rather than re-rendered.
    """
    conn = db.get_conn(project_id)
    row = conn.execute(
        "SELECT NAME, FILENAME FROM {t} WHERE {id} = ?".format(
            t=kind.asset_table, id=kind.id_column
        ),
        (asset_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return jsonify({"error": "no such {}".format(kind.noun)}), 404

    path = row["FILENAME"]
    if not path or not Path(path).is_file():
        return jsonify({"error": "{} file is missing: {}".format(kind.noun, path)}), 404
    if not preview.can_preview(path):
        return jsonify({
            "error": "previews aren't supported for {} files".format(
                Path(path).suffix.lower() or "these"
            )
        }), 415

    stat = Path(path).stat()
    etag = '"r{}-{}-{}-{}-{}"'.format(preview.RENDER_VERSION, kind.noun, asset_id, int(stat.st_mtime), stat.st_size)
    if request.headers.get("If-None-Match") == etag:
        return "", 304

    try:
        png, meta = render(path)
    except preview.PreviewError as exc:
        return jsonify({"error": str(exc)}), 422

    response = app.response_class(png, mimetype="image/png")
    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = "private, no-cache"
    for header, value in (extra_headers or {}).items():
        response.headers[header] = str(meta[value])
    return response


# ---------------------------------------------------------------------------
# Movie routes
# ---------------------------------------------------------------------------

@app.route("/api/projects/<project_id>/movies/check-import", methods=["POST"])
@auth.project_access_required
def check_movie_import(project_id):
    return _check_import(project_id, MOVIE_KIND)


@app.route("/api/projects/<project_id>/movies", methods=["GET"])
@auth.project_access_required
def list_movies(project_id):
    return _list_assets(project_id, MOVIE_KIND)


@app.route("/api/projects/<project_id>/movies/<int:movie_id>/preview.png", methods=["GET"])
@auth.project_access_required
def movie_preview(project_id, movie_id):
    """The movie's frames, summed. A single frame of counting-mode data is
    essentially shot noise, so only the sum is worth looking at."""
    return _preview_response(
        project_id, MOVIE_KIND, movie_id, preview.render_movie_preview,
        extra_headers={
            "X-Preview-Frames-Summed": "frames_summed",
            "X-Preview-Frames-Total": "frames_total",
        },
    )


@app.route("/api/projects/<project_id>/movie-groups", methods=["GET"])
@auth.project_access_required
def list_movie_groups(project_id):
    return _list_groups(project_id, MOVIE_KIND)


@app.route("/api/projects/<project_id>/movies/import-defaults", methods=["GET"])
@auth.project_access_required
def get_movie_import_defaults(project_id):
    conn = db.get_conn(project_id)
    row = conn.execute("SELECT * FROM MOVIE_IMPORT_DEFAULTS WHERE NUMBER=1").fetchone()
    conn.close()
    return jsonify(dict(row) if row else {})


@app.route("/api/projects/<project_id>/movies/import", methods=["POST"])
@auth.project_access_required
def import_movies(project_id):
    body = request.get_json(force=True, silent=True) or {}

    input_glob = (body.get("input_glob") or "").strip()
    voltage_kv = body.get("voltage_kv")
    cs_mm = body.get("cs_mm")
    pixel_size_a = body.get("pixel_size_a")
    dose_per_frame = body.get("dose_per_frame")
    protein_is_white = bool(body.get("protein_is_white"))

    apply_gain = bool(body.get("apply_gain"))
    gain_ref = (body.get("gain_ref") or None) if apply_gain else None
    apply_dark = bool(body.get("apply_dark"))
    dark_ref = (body.get("dark_ref") or None) if apply_dark else None

    resample_movies = bool(body.get("resample_movies"))
    desired_pixel_size_a = body.get("desired_pixel_size_a")
    if resample_movies and desired_pixel_size_a and pixel_size_a:
        output_binning_factor = desired_pixel_size_a / pixel_size_a
    else:
        output_binning_factor = 1.0

    eer_frames_per_image = body.get("eer_frames_per_image")
    if eer_frames_per_image is not None:
        eer_frames_per_image = int(eer_frames_per_image)
    eer_super_res_factor = body.get("eer_super_res_factor")
    if eer_super_res_factor is not None:
        eer_super_res_factor = int(eer_super_res_factor)

    # Counting a TIFF/EER movie's frames means walking every IFD in the file;
    # this skips that and leaves the count NULL, keeping only the dimensions
    # from the first IFD. MRC is unaffected -- its section count sits in the
    # fixed header. Cf. skip_full_check_of_tiff_movies in the reference dialog.
    skip_full_check = bool(body.get("skip_full_check"))

    # Mirrors the reference dialog's CheckImportButtonStatus(), which keeps
    # its Import button disabled until the same conditions hold. The dialog
    # enforces these live so this should never fire, but a request can also
    # arrive from something other than that dialog.
    errors = []
    if not input_glob:
        errors.append("movie files path is required")
    for label, value in (
        ("voltage", voltage_kv),
        ("spherical aberration (Cs)", cs_mm),
        ("pixel size", pixel_size_a),
        ("dose per frame", dose_per_frame),
    ):
        if value is None or value == "":
            errors.append("{} is required".format(label))
    if apply_gain and not gain_ref:
        errors.append("gain reference is required when applying gain correction")
    if apply_dark and not dark_ref:
        errors.append("dark reference is required when applying dark correction")
    for label, path in (("gain reference", gain_ref), ("dark reference", dark_ref)):
        if path and not Path(path).expanduser().is_file():
            errors.append("{} not found: {}".format(label, path))
    if resample_movies:
        if not desired_pixel_size_a:
            errors.append("desired pixel size is required when resampling")
        elif pixel_size_a and desired_pixel_size_a <= pixel_size_a:
            errors.append("desired pixel size must be larger than the current pixel size")
    if errors:
        return jsonify({"error": "; ".join(errors)}), 400

    conn = db.get_conn(project_id)
    # Files already in this project are skipped rather than added twice --
    # the same call IsFileAnAsset() makes in the reference dialog.
    matched, already_imported = _partition_matches(conn, MOVIE_KIND, input_glob)
    if not matched and not already_imported:
        conn.close()
        return jsonify({"error": "no files match that path: {}".format(input_glob)}), 400
    if not matched:
        conn.close()
        return jsonify({
            "error": "all {} matching file{} already imported".format(
                len(already_imported), "" if len(already_imported) == 1 else "s are"
            )
        }), 400

    # Like the reference dialog's CheckForEERFiles(), decide from the actual
    # resolved files -- not just the client's glob-based guess -- whether
    # this is an EER import. Only store EER fields on the movie rows
    # themselves when it is (meaningless metadata otherwise); the defaults
    # row still carries the raw values forward regardless, same as the
    # reference dialog persisting whatever's in the (possibly disabled)
    # EER controls every time.
    is_eer_import = any(p.lower().endswith(".eer") for p in matched)
    asset_eer_frames_per_image = eer_frames_per_image if is_eer_import else None
    asset_eer_super_res_factor = eer_super_res_factor if is_eer_import else None

    # Imports land in the All Movies group (id 0) only -- organizing movies
    # into other groups is what Add To Group is for, after the fact.
    with conn:
        movie_ids = []
        failed = []
        for path in matched:
            name = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            # Dimensions and frame count come from the file's own header (see
            # imageheaders.py). A file that won't parse is skipped and
            # reported rather than failing the whole import, matching the
            # reference dialog's "%s is not a valid image file, skipping".
            try:
                header = imageheaders.read_movie_header(
                    path,
                    count_frames=not skip_full_check,
                    eer_super_res_factor=eer_super_res_factor,
                    eer_frames_per_image=eer_frames_per_image,
                )
            except imageheaders.HeaderError as exc:
                failed.append({"path": path, "reason": str(exc)})
                continue
            x_size = header["x_size"]
            y_size = header["y_size"]
            n_frames = header["number_of_frames"]
            # Same guard the reference dialog applies -- but only when the
            # count is trustworthy, i.e. we actually counted.
            if n_frames is not None and n_frames < 3:
                failed.append({
                    "path": path,
                    "reason": "contains fewer than 3 frames ({})".format(n_frames),
                })
                continue
            cur = conn.execute(
                "INSERT INTO MOVIE_ASSETS("
                "NAME, FILENAME, POSITION_IN_STACK, X_SIZE, Y_SIZE, NUMBER_OF_FRAMES, "
                "VOLTAGE, PIXEL_SIZE, DOSE_PER_FRAME, SPHERICAL_ABERRATION, GAIN_FILENAME, "
                "DARK_FILENAME, OUTPUT_BINNING_FACTOR, PROTEIN_IS_WHITE, EER_SUPER_RES_FACTOR, "
                "EER_FRAMES_PER_IMAGE) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    name, path, 1, x_size, y_size, n_frames,
                    voltage_kv, pixel_size_a, dose_per_frame, cs_mm, gain_ref, dark_ref,
                    output_binning_factor, int(protein_is_white),
                    asset_eer_super_res_factor, asset_eer_frames_per_image,
                ),
            )
            movie_id = cur.lastrowid
            movie_ids.append(movie_id)
            conn.execute(
                "INSERT INTO MOVIE_GROUP_MEMBERS(GROUP_ID, MOVIE_ASSET_ID) VALUES (0, ?)",
                (movie_id,),
            )

        conn.execute(
            "UPDATE MOVIE_IMPORT_DEFAULTS SET VOLTAGE=?, SPHERICAL_ABERRATION=?, PIXEL_SIZE=?, "
            "EXPOSURE_PER_FRAME=?, MOVIES_ARE_GAIN_CORRECTED=?, GAIN_REFERENCE_FILENAME=?, "
            "MOVIES_ARE_DARK_CORRECTED=?, DARK_REFERENCE_FILENAME=?, RESAMPLE_MOVIES=?, "
            "DESIRED_PIXEL_SIZE=?, PROTEIN_IS_WHITE=?, EER_SUPER_RES_FACTOR=?, "
            "EER_FRAMES_PER_IMAGE=? WHERE NUMBER=1",
            (
                voltage_kv, cs_mm, pixel_size_a, dose_per_frame,
                int(not apply_gain), gain_ref, int(not apply_dark), dark_ref,
                int(resample_movies), desired_pixel_size_a, int(protein_is_white),
                eer_super_res_factor, eer_frames_per_image,
            ),
        )
    conn.close()

    return jsonify({
        "movie_count": len(movie_ids),
        "skipped_count": len(already_imported),
        "failed": failed,
    }), 201


@app.route("/api/projects/<project_id>/movie-groups", methods=["POST"])
@auth.project_access_required
def create_movie_group(project_id):
    return _create_group(project_id, MOVIE_KIND)


@app.route("/api/projects/<project_id>/movie-groups/<int:group_id>", methods=["PATCH"])
@auth.project_access_required
def rename_movie_group(project_id, group_id):
    return _rename_group(project_id, MOVIE_KIND, group_id)


@app.route("/api/projects/<project_id>/movie-groups/<int:group_id>", methods=["DELETE"])
@auth.project_access_required
def delete_movie_group(project_id, group_id):
    return _delete_group(project_id, MOVIE_KIND, group_id)


@app.route("/api/projects/<project_id>/movie-groups/<int:group_id>/invert", methods=["POST"])
@auth.project_access_required
def invert_movie_group(project_id, group_id):
    return _invert_group(project_id, MOVIE_KIND, group_id)


@app.route("/api/projects/<project_id>/movies/<int:asset_id>", methods=["PATCH"])
@auth.project_access_required
def rename_movies(project_id, asset_id):
    return _rename_asset(project_id, MOVIE_KIND, asset_id)


@app.route("/api/projects/<project_id>/movies/delete", methods=["POST"])
@auth.project_access_required
def delete_movies(project_id):
    return _delete_assets(project_id, MOVIE_KIND)


@app.route("/api/projects/<project_id>/movie-groups/<int:group_id>/remove-movies", methods=["POST"])
@auth.project_access_required
def remove_movies_from_group(project_id, group_id):
    return _remove_from_group(project_id, MOVIE_KIND, group_id)


@app.route("/api/projects/<project_id>/movies/add-to-group", methods=["POST"])
@auth.project_access_required
def add_movies_to_group(project_id):
    return _add_to_group(project_id, MOVIE_KIND)


# ---------------------------------------------------------------------------
# Image routes
#
# An image asset is one already-averaged micrograph -- either produced by
# Align Movies (which writes IMAGE_ASSETS rows with a parent movie and an
# alignment id) or imported here from micrographs aligned elsewhere. Both
# kinds live in the same table and the same All Images group, which is why
# the import route below leaves PARENT_MOVIE_ID/ALIGNMENT_ID at -1 rather
# than NULL: that's the sentinel cisTEM's own AddNextImageAsset() uses for
# "imported, no parent in this project".
# ---------------------------------------------------------------------------

@app.route("/api/projects/<project_id>/images/check-import", methods=["POST"])
@auth.project_access_required
def check_image_import(project_id):
    return _check_import(project_id, IMAGE_KIND, allowed_extensions=IMAGE_IMPORT_EXTENSIONS)


@app.route("/api/projects/<project_id>/images", methods=["GET"])
@auth.project_access_required
def list_images(project_id):
    return _list_assets(project_id, IMAGE_KIND)


@app.route("/api/projects/<project_id>/images/<int:image_id>/preview.png", methods=["GET"])
@auth.project_access_required
def image_preview(project_id, image_id):
    """One micrograph, rendered as-is -- nothing to sum, unlike a movie."""
    return _preview_response(project_id, IMAGE_KIND, image_id, preview.render_image_preview)


@app.route("/api/projects/<project_id>/image-groups", methods=["GET"])
@auth.project_access_required
def list_image_groups(project_id):
    return _list_groups(project_id, IMAGE_KIND)


@app.route("/api/projects/<project_id>/images/import-defaults", methods=["GET"])
@auth.project_access_required
def get_image_import_defaults(project_id):
    conn = db.get_conn(project_id)
    row = conn.execute("SELECT * FROM IMAGE_IMPORT_DEFAULTS WHERE NUMBER=1").fetchone()
    conn.close()
    return jsonify(dict(row) if row else {})


@app.route("/api/projects/<project_id>/volumes/check-import", methods=["POST"])
@auth.project_access_required
def check_volume_import(project_id):
    """The volume counterpart of /images/check-import, plus `pixel_size_hint`
    and `box_sizes` read from the new files' MRC headers -- cisTEM's dialog
    leaves the pixel size blank for the user to type, but the header
    usually knows it."""
    response = _check_import(project_id, VOLUME_KIND, allowed_extensions=VOLUME_IMPORT_EXTENSIONS)
    body = request.get_json(force=True, silent=True) or {}
    input_glob = (body.get("glob") or "").strip()
    out = response.get_json()
    out["pixel_size_hint"] = None
    out["box_sizes"] = []
    if input_glob:
        conn = db.get_conn(project_id)
        try:
            new_paths, _dups = _partition_matches(conn, VOLUME_KIND, input_glob, allowed_extensions=VOLUME_IMPORT_EXTENSIONS)
        finally:
            conn.close()
        sizes = set()
        not_volumes = 0
        for path in new_paths:
            try:
                h = volumes.read_mrc_header(path)
            except (OSError, ValueError, struct.error):
                not_volumes += 1
                continue
            if h["nz"] <= 1 or h["nx"] <= 0 or h["ny"] <= 0:
                not_volumes += 1  # matched the glob but is no volume; import() would skip it
                continue
            if out["pixel_size_hint"] is None and h["pixel_size"] > 0:
                out["pixel_size_hint"] = round(h["pixel_size"], 4)
            sizes.add(h["nx"])
        out["box_sizes"] = sorted(sizes)
        out["not_volumes"] = not_volumes
        out["new_count"] = max(0, out["new_count"] - not_volumes)
    return jsonify(out)


@app.route("/api/projects/<project_id>/volumes/import", methods=["POST"])
@auth.project_access_required
def import_volumes(project_id):
    """MyVolumeImportDialog::ImportClick(): every matching MRC file not yet
    an asset becomes a volume asset named after the file, with the one pixel
    size typed in the dialog and its sizes from the header, no
    reconstruction (-1) and no half maps -- an imported volume comes from
    outside the project. A file that is not a 3D MRC volume is skipped and
    reported, as the dialog's error list does."""
    body = request.get_json(force=True, silent=True) or {}
    input_glob = (body.get("input_glob") or "").strip()
    pixel_size = body.get("pixel_size_a")
    if not input_glob:
        return jsonify({"error": "volume files path is required"}), 400
    try:
        pixel_size = float(pixel_size)
    except (TypeError, ValueError):
        return jsonify({"error": "pixel size is required"}), 400
    if pixel_size <= 0:
        return jsonify({"error": "pixel size must be positive"}), 400
    conn = db.get_conn(project_id)
    try:
        matched, already_imported = _partition_matches(conn, VOLUME_KIND, input_glob, allowed_extensions=VOLUME_IMPORT_EXTENSIONS)
        if not matched and not already_imported:
            return jsonify({"error": "no MRC files match that path: {}".format(input_glob)}), 400
        if not matched:
            return jsonify({"error": "all {} matching file{} already imported".format(len(already_imported), "" if len(already_imported) == 1 else "s are")}), 400
        imported, failed = [], []
        for path in matched:
            try:
                h = volumes.read_mrc_header(path)
            except (OSError, ValueError, struct.error) as exc:
                failed.append({"path": path, "reason": "not a valid MRC file ({})".format(exc)})
                continue
            if h["nz"] <= 1:
                failed.append({"path": path, "reason": "not a volume (one section)"})
                continue
            if h["mode"] not in (0, 1, 2, 6, 12):
                failed.append({"path": path, "reason": "unsupported MRC mode {}".format(h["mode"])})
                continue
            name = Path(path).stem
            vid = volumes.add_volume_asset(conn, name, path, pixel_size, h["nx"], h["ny"], h["nz"])
            imported.append({"volume_asset_id": vid, "name": name, "x_size": h["nx"], "y_size": h["ny"], "z_size": h["nz"],
                             "cubic": h["nx"] == h["ny"] == h["nz"]})
        if not imported and failed:
            return jsonify({"error": "none of the {} matching file{} could be imported: {}".format(
                len(failed), "" if len(failed) == 1 else "s", failed[0]["reason"]), "failed": failed}), 400
        return jsonify({"volume_count": len(imported), "skipped_count": len(already_imported), "failed": failed, "volumes": imported})
    finally:
        conn.close()


@app.route("/api/projects/<project_id>/images/import", methods=["POST"])
@auth.project_access_required
def import_images(project_id):
    """Imports micrographs as image assets.

    Deliberately much shorter than import_movies(): an image is already
    averaged, so there are no frames to count, no dose to record, no gain or
    dark reference to apply and no EER sampling to resolve -- exactly the
    difference between cisTEM's own MyImageImportDialog and
    MyMovieImportDialog, whose field set is voltage / Cs / pixel size /
    contrast and nothing more.
    """
    body = request.get_json(force=True, silent=True) or {}

    input_glob = (body.get("input_glob") or "").strip()
    voltage_kv = body.get("voltage_kv")
    cs_mm = body.get("cs_mm")
    pixel_size_a = body.get("pixel_size_a")
    protein_is_white = bool(body.get("protein_is_white"))

    # Mirrors CheckImportButtonStatus() in MyImageImportDialog.cpp: files
    # chosen, and voltage/pixel size/Cs all non-empty.
    errors = []
    if not input_glob:
        errors.append("image files path is required")
    for label, value in (
        ("voltage", voltage_kv),
        ("spherical aberration (Cs)", cs_mm),
        ("pixel size", pixel_size_a),
    ):
        if value is None or value == "":
            errors.append("{} is required".format(label))
    if errors:
        return jsonify({"error": "; ".join(errors)}), 400

    conn = db.get_conn(project_id)
    matched, already_imported = _partition_matches(
        conn, IMAGE_KIND, input_glob, allowed_extensions=IMAGE_IMPORT_EXTENSIONS
    )
    if not matched and not already_imported:
        conn.close()
        return jsonify({"error": "no files match that path: {}".format(input_glob)}), 400
    if not matched:
        conn.close()
        return jsonify({
            "error": "all {} matching file{} already imported".format(
                len(already_imported), "" if len(already_imported) == 1 else "s are"
            )
        }), 400

    with conn:
        image_ids = []
        failed = []
        for path in matched:
            name = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            try:
                header = imageheaders.read_image_header(path)
            except imageheaders.HeaderError as exc:
                failed.append({"path": path, "reason": str(exc)})
                continue
            cur = conn.execute(
                "INSERT INTO IMAGE_ASSETS("
                "NAME, FILENAME, POSITION_IN_STACK, PARENT_MOVIE_ID, ALIGNMENT_ID, "
                "CTF_ESTIMATION_ID, X_SIZE, Y_SIZE, PIXEL_SIZE, VOLTAGE, "
                "SPHERICAL_ABERRATION, PROTEIN_IS_WHITE) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    name, path, 1, -1, -1, -1,
                    header["x_size"], header["y_size"], pixel_size_a, voltage_kv,
                    cs_mm, int(protein_is_white),
                ),
            )
            image_id = cur.lastrowid
            image_ids.append(image_id)
            conn.execute(
                "INSERT INTO IMAGE_GROUP_MEMBERS(GROUP_ID, IMAGE_ASSET_ID) VALUES (0, ?)",
                (image_id,),
            )

        conn.execute(
            "UPDATE IMAGE_IMPORT_DEFAULTS SET VOLTAGE=?, SPHERICAL_ABERRATION=?, PIXEL_SIZE=?, "
            "PROTEIN_IS_WHITE=? WHERE NUMBER=1",
            (voltage_kv, cs_mm, pixel_size_a, int(protein_is_white)),
        )
    conn.close()

    return jsonify({
        "image_count": len(image_ids),
        "skipped_count": len(already_imported),
        "failed": failed,
    }), 201


@app.route("/api/projects/<project_id>/image-groups", methods=["POST"])
@auth.project_access_required
def create_image_group(project_id):
    return _create_group(project_id, IMAGE_KIND)


@app.route("/api/projects/<project_id>/image-groups/<int:group_id>", methods=["PATCH"])
@auth.project_access_required
def rename_image_group(project_id, group_id):
    return _rename_group(project_id, IMAGE_KIND, group_id)


@app.route("/api/projects/<project_id>/image-groups/<int:group_id>", methods=["DELETE"])
@auth.project_access_required
def delete_image_group(project_id, group_id):
    return _delete_group(project_id, IMAGE_KIND, group_id)


@app.route("/api/projects/<project_id>/image-groups/<int:group_id>/invert", methods=["POST"])
@auth.project_access_required
def invert_image_group(project_id, group_id):
    return _invert_group(project_id, IMAGE_KIND, group_id)


@app.route("/api/projects/<project_id>/images/<int:asset_id>", methods=["PATCH"])
@auth.project_access_required
def rename_images(project_id, asset_id):
    return _rename_asset(project_id, IMAGE_KIND, asset_id)


@app.route("/api/projects/<project_id>/images/delete", methods=["POST"])
@auth.project_access_required
def delete_images(project_id):
    return _delete_assets(project_id, IMAGE_KIND)


@app.route("/api/projects/<project_id>/image-groups/<int:group_id>/remove-images", methods=["POST"])
@auth.project_access_required
def remove_images_from_group(project_id, group_id):
    return _remove_from_group(project_id, IMAGE_KIND, group_id)


@app.route("/api/projects/<project_id>/images/add-to-group", methods=["POST"])
@auth.project_access_required
def add_images_to_group(project_id):
    return _add_to_group(project_id, IMAGE_KIND)


# ---------------------------------------------------------------------------
# Particle position assets -- the third AssetKind. Listing is bespoke (it
# joins the parent image's name and the pick job's number, and a project can
# hold hundreds of thousands, so it is capped); everything else is shared.
# ---------------------------------------------------------------------------

POSITION_LIST_LIMIT = 5000


@app.route("/api/projects/<project_id>/particle-positions", methods=["GET"])
@auth.project_access_required
def list_particle_positions(project_id):
    """Positions in a group (`group_id`, default all), oldest first, with the
    parent image's name and the pick job's number. `total` is the full
    count; at most POSITION_LIST_LIMIT rows come back (`truncated` says so)
    -- cisTEM's panel lists every position, but a browser table of 200k
    rows isn't a table anyone reads."""
    group_id = request.args.get("group_id", type=int)
    image_id = request.args.get("image_id", type=int)
    conn = db.get_conn(project_id)
    where, args = [], []
    if group_id is not None:
        where.append("pp.PARTICLE_POSITION_ASSET_ID IN (SELECT PARTICLE_POSITION_ASSET_ID FROM PARTICLE_POSITION_GROUP_MEMBERS WHERE GROUP_ID = ?)")
        args.append(group_id)
    if image_id is not None:
        where.append("pp.PARENT_IMAGE_ASSET_ID = ?")
        args.append(image_id)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    total = conn.execute("SELECT COUNT(*) FROM PARTICLE_POSITION_ASSETS pp" + clause, args).fetchone()[0]
    rows = conn.execute(
        "SELECT pp.*, ia.NAME AS IMAGE_NAME, j.JOB_NUMBER FROM PARTICLE_POSITION_ASSETS pp "
        "LEFT JOIN IMAGE_ASSETS ia ON ia.IMAGE_ASSET_ID = pp.PARENT_IMAGE_ASSET_ID "
        "LEFT JOIN JOBS j ON j.JOB_ID = pp.PICK_JOB_ID" + clause +
        " ORDER BY pp.PARTICLE_POSITION_ASSET_ID LIMIT ?", args + [POSITION_LIST_LIMIT]).fetchall()
    conn.close()
    return jsonify({"particle_positions": [dict(r) for r in rows], "total": total, "truncated": total > len(rows)})


def _import_particle_positions(conn, text):
    """MyParticlePositionAssetPanel::ImportAssetClick(): one position per
    line, `<image asset id or image filename> <x> <y>` in Angstroms, `#`
    comments and blank lines skipped. Extra columns are ignored with a
    warning; a line that doesn't parse is skipped and reported, as
    cisTEM's error dialog does. Imported positions carry no pick job
    (PICKING_ID -1, cisTEM's sentinel) and join All Particle Positions."""
    images = conn.execute("SELECT IMAGE_ASSET_ID, FILENAME, NAME FROM IMAGE_ASSETS").fetchall()
    by_id = {r["IMAGE_ASSET_ID"] for r in images}
    by_name = {}
    for r in images:
        for key in (r["FILENAME"], os.path.basename(r["FILENAME"] or ""), r["NAME"],
                    os.path.splitext(os.path.basename(r["FILENAME"] or ""))[0]):
            if key:
                by_name.setdefault(key, r["IMAGE_ASSET_ID"])
    imported, failed, warnings = 0, [], []
    with conn:
        for number, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                failed.append({"line": number, "reason": "contains fewer than 3 values"})
                continue
            if len(parts) > 3:
                warnings.append("line {} has more than 3 values; only the first 3 were used".format(number))
            try:
                image_id = int(parts[0])
                if image_id not in by_id:
                    failed.append({"line": number, "reason": "{} is not an existing image asset".format(image_id)})
                    continue
            except ValueError:
                image_id = by_name.get(parts[0])
                if image_id is None:
                    failed.append({"line": number, "reason": "column 1 is neither an image asset id nor the filename of one"})
                    continue
            try:
                x, y = float(parts[1]), float(parts[2])
            except ValueError:
                failed.append({"line": number, "reason": "columns 2 and 3 must be the X and Y positions in Angstroms"})
                continue
            cur = conn.execute(
                "INSERT INTO PARTICLE_POSITION_ASSETS(PARENT_IMAGE_ASSET_ID, PICKING_ID, PICK_JOB_ID, X_POSITION, Y_POSITION, "
                "PEAK_HEIGHT, TEMPLATE_ASSET_ID, TEMPLATE_PSI, TEMPLATE_THETA, TEMPLATE_PHI) VALUES (?, -1, NULL, ?, ?, NULL, 0, 0, 0, 0)",
                (image_id, x, y))
            conn.execute("INSERT OR IGNORE INTO PARTICLE_POSITION_GROUP_MEMBERS(GROUP_ID, PARTICLE_POSITION_ASSET_ID) VALUES (0, ?)", (cur.lastrowid,))
            imported += 1
    return {"imported": imported, "failed": failed, "warnings": warnings}


@app.route("/api/projects/<project_id>/particle-positions/import", methods=["POST"])
@auth.project_access_required
def import_particle_positions(project_id):
    """Body `{text}` (the file's contents, read in the browser) or `{path}`
    (a text file on this server, as the movie and image imports take) --
    lines of `<image asset id or filename> <x> <y>`, Angstroms.
    -> {imported, failed: [{line, reason}], warnings}."""
    body = request.get_json(force=True, silent=True) or {}
    text = body.get("text")
    if text is None and body.get("path"):
        path = Path(str(body["path"])).expanduser()
        if not path.is_file():
            return jsonify({"error": "no such file: {}".format(path)}), 400
        try:
            text = path.read_text(errors="replace")
        except OSError as exc:
            return jsonify({"error": str(exc)}), 400
    if text is None:
        return jsonify({"error": "text or path is required"}), 400
    conn = db.get_conn(project_id)
    try:
        result = _import_particle_positions(conn, text)
    finally:
        conn.close()
    if result["imported"] == 0 and result["failed"]:
        return jsonify(dict(result, error="no line could be imported")), 400
    return jsonify(result), 201


@app.route("/api/projects/<project_id>/particle-position-groups", methods=["GET"])
@auth.project_access_required
def list_position_groups(project_id):
    return _list_groups(project_id, POSITION_KIND)


@app.route("/api/projects/<project_id>/particle-position-groups", methods=["POST"])
@auth.project_access_required
def create_position_group(project_id):
    return _create_group(project_id, POSITION_KIND)


@app.route("/api/projects/<project_id>/particle-position-groups/<int:group_id>", methods=["PATCH"])
@auth.project_access_required
def rename_position_group(project_id, group_id):
    return _rename_group(project_id, POSITION_KIND, group_id)


@app.route("/api/projects/<project_id>/particle-position-groups/<int:group_id>", methods=["DELETE"])
@auth.project_access_required
def delete_position_group(project_id, group_id):
    return _delete_group(project_id, POSITION_KIND, group_id)


@app.route("/api/projects/<project_id>/particle-position-groups/<int:group_id>/invert", methods=["POST"])
@auth.project_access_required
def invert_position_group(project_id, group_id):
    return _invert_group(project_id, POSITION_KIND, group_id)


@app.route("/api/projects/<project_id>/particle-positions/delete", methods=["POST"])
@auth.project_access_required
def delete_particle_positions(project_id):
    return _delete_assets(project_id, POSITION_KIND)


@app.route("/api/projects/<project_id>/particle-position-groups/<int:group_id>/remove-particle-positions", methods=["POST"])
@auth.project_access_required
def remove_positions_from_group(project_id, group_id):
    return _remove_from_group(project_id, POSITION_KIND, group_id)


@app.route("/api/projects/<project_id>/particle-positions/add-to-group", methods=["POST"])
@auth.project_access_required
def add_positions_to_group(project_id):
    return _add_to_group(project_id, POSITION_KIND)


@app.route("/api/projects/<project_id>/particle-position-groups/from-image-group", methods=["POST"])
@auth.project_access_required
def position_group_from_image_group(project_id):
    """cisTEM's "New from parent": a particle position group holding every
    position whose parent image is in the given image group. Body
    {image_group_id, group_name}; an existing group of that name gains the
    positions, a new name creates one."""
    body = request.get_json(force=True, silent=True) or {}
    image_group_id = body.get("image_group_id")
    group_name = (body.get("group_name") or "").strip()
    if image_group_id is None or not group_name:
        return jsonify({"error": "image_group_id and group_name are required"}), 400
    conn = db.get_conn(project_id)
    with conn:
        ids = [r[0] for r in conn.execute(
            "SELECT PARTICLE_POSITION_ASSET_ID FROM PARTICLE_POSITION_ASSETS WHERE PARENT_IMAGE_ASSET_ID IN "
            "(SELECT IMAGE_ASSET_ID FROM IMAGE_GROUP_MEMBERS WHERE GROUP_ID = ?)", (int(image_group_id),))]
        row = conn.execute("SELECT GROUP_ID FROM PARTICLE_POSITION_GROUP_LIST WHERE LOWER(GROUP_NAME) = LOWER(?)", (group_name,)).fetchone()
        if row:
            gid = row["GROUP_ID"]
        else:
            gid = conn.execute("INSERT INTO PARTICLE_POSITION_GROUP_LIST(GROUP_NAME, LIST_ID) VALUES (?, 0)", (group_name,)).lastrowid
        conn.executemany("INSERT OR IGNORE INTO PARTICLE_POSITION_GROUP_MEMBERS(GROUP_ID, PARTICLE_POSITION_ASSET_ID) VALUES (?, ?)",
                         [(gid, i) for i in ids])
    conn.close()
    return jsonify({"group_id": gid, "group_name": group_name, "added": len(ids)}), 201


# ---------------------------------------------------------------------------
# Refinement packages (MyRefinementPackageAssetPanel + MyNewRefinementPackageWizard)
# ---------------------------------------------------------------------------

@app.route("/api/projects/<project_id>/refinement-packages", methods=["GET"])
@auth.project_access_required
def list_refinement_packages(project_id):
    conn = db.get_conn(project_id)
    out = refinement_packages.list_packages(conn)
    conn.close()
    return jsonify({"refinement_packages": out})


@app.route("/api/projects/<project_id>/refinement-packages/defaults", methods=["GET"])
@auth.project_access_required
def refinement_package_defaults(project_id):
    """What the wizard prefills for a particle group: the first particle's
    image pixel size and the box size cisTEM derives from the largest
    dimension (`?particle_group_id=&largest_dimension_a=`)."""
    group_id = request.args.get("particle_group_id", type=int)
    largest = request.args.get("largest_dimension_a", type=float) or 150.0
    selection_ids = [int(x) for x in request.args.get("selection_ids", "").split(",") if x.strip().isdigit()]
    if group_id is None and not selection_ids:
        return jsonify({"error": "particle_group_id or selection_ids is required"}), 400
    conn = db.get_conn(project_id)
    if selection_ids:
        # From class averages: the parent package's box and pixel size (BoxSizeWizardPage).
        try:
            out = refinement_packages.selection_defaults(conn, selection_ids)
        except ValueError as exc:
            conn.close()
            return jsonify({"error": str(exc)}), 400
    else:
        out = refinement_packages.group_defaults(conn, group_id, largest)
    out["next_name"] = "Refinement Package #{}".format(
        conn.execute("SELECT COALESCE(MAX(REFINEMENT_PACKAGE_ASSET_ID), 0) + 1 FROM REFINEMENT_PACKAGE_ASSETS").fetchone()[0])
    out["symmetries"] = list(refinement_packages.SYMMETRIES)
    conn.close()
    return jsonify(out)


@app.route("/api/projects/<project_id>/refinement-packages", methods=["POST"])
@auth.project_access_required
def create_refinement_package(project_id):
    """Body {particle_group_id, name, symmetry, molecular_weight_kda,
    largest_dimension_a, number_of_classes, box_size, output_pixel_size}:
    cuts the stack and writes the package and its "Random Parameters"
    refinement, as the wizard's Finish does. Synchronous -- a project of a
    few hundred thousand particles will take a while."""
    body = request.get_json(force=True, silent=True) or {}
    conn = db.get_conn(project_id)
    try:
        try:
            result = refinement_packages.create_package(conn, project_id, body)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except OSError as exc:
            return jsonify({"error": "could not read or write a file: {}".format(exc)}), 500
        return jsonify(result), 201
    finally:
        conn.close()


@app.route("/api/projects/<project_id>/refinement-packages/<int:package_id>", methods=["GET"])
@auth.project_access_required
def get_refinement_package(project_id, package_id):
    conn = db.get_conn(project_id)
    pkg = [p for p in refinement_packages.list_packages(conn) if p["refinement_package_asset_id"] == package_id]
    if not pkg:
        conn.close()
        return jsonify({"error": "no such refinement package"}), 404
    d = pkg[0]
    d["particles"], d["particle_total"] = refinement_packages.package_particles(conn, package_id)
    d["refinements"] = [dict(r) for r in conn.execute(
        "SELECT REFINEMENT_ID, NAME, DATETIME_OF_RUN, NUMBER_OF_PARTICLES, NUMBER_OF_CLASSES FROM REFINEMENT_LIST "
        "WHERE REFINEMENT_PACKAGE_ASSET_ID=? ORDER BY REFINEMENT_ID", (package_id,)).fetchall()]
    conn.close()
    return jsonify(d)


@app.route("/api/projects/<project_id>/refinement-packages/<int:package_id>", methods=["PATCH"])
@auth.project_access_required
def rename_refinement_package(project_id, package_id):
    body = request.get_json(force=True, silent=True) or {}
    conn = db.get_conn(project_id)
    try:
        try:
            ok = refinement_packages.rename_package(conn, package_id, body.get("name"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if not ok:
            return jsonify({"error": "no such refinement package"}), 404
        return jsonify({"renamed": package_id})
    finally:
        conn.close()


@app.route("/api/projects/<project_id>/refinement-packages/<int:package_id>", methods=["DELETE"])
@auth.project_access_required
def delete_refinement_package(project_id, package_id):
    conn = db.get_conn(project_id)
    ok = refinement_packages.delete_package(conn, package_id)
    conn.close()
    if not ok:
        return jsonify({"error": "no such refinement package"}), 404
    return jsonify({"deleted": package_id})


# ---------------------------------------------------------------------------
# 3D volumes (VOLUME_ASSETS) -- MyVolumeAssetPanel -- and the ab-initio runs
# (STARTUP_LIST) that make them.
# ---------------------------------------------------------------------------

@app.route("/api/projects/<project_id>/volumes", methods=["GET"])
@auth.project_access_required
def list_volumes(project_id):
    return _list_assets(project_id, VOLUME_KIND)


@app.route("/api/projects/<project_id>/volume-groups", methods=["GET"])
@auth.project_access_required
def list_volume_groups(project_id):
    return _list_groups(project_id, VOLUME_KIND)


@app.route("/api/projects/<project_id>/volume-groups", methods=["POST"])
@auth.project_access_required
def create_volume_group(project_id):
    return _create_group(project_id, VOLUME_KIND)


@app.route("/api/projects/<project_id>/volume-groups/<int:group_id>", methods=["PATCH"])
@auth.project_access_required
def rename_volume_group(project_id, group_id):
    return _rename_group(project_id, VOLUME_KIND, group_id)


@app.route("/api/projects/<project_id>/volume-groups/<int:group_id>", methods=["DELETE"])
@auth.project_access_required
def delete_volume_group(project_id, group_id):
    return _delete_group(project_id, VOLUME_KIND, group_id)


@app.route("/api/projects/<project_id>/volume-groups/<int:group_id>/invert", methods=["POST"])
@auth.project_access_required
def invert_volume_group(project_id, group_id):
    return _invert_group(project_id, VOLUME_KIND, group_id)


@app.route("/api/projects/<project_id>/volumes/<int:asset_id>", methods=["PATCH"])
@auth.project_access_required
def rename_volumes(project_id, asset_id):
    return _rename_asset(project_id, VOLUME_KIND, asset_id)


@app.route("/api/projects/<project_id>/volumes/delete", methods=["POST"])
@auth.project_access_required
def delete_volumes(project_id):
    return _delete_assets(project_id, VOLUME_KIND)


@app.route("/api/projects/<project_id>/volume-groups/<int:group_id>/remove-volumes", methods=["POST"])
@auth.project_access_required
def remove_volumes_from_group(project_id, group_id):
    return _remove_from_group(project_id, VOLUME_KIND, group_id)


@app.route("/api/projects/<project_id>/volumes/add-to-group", methods=["POST"])
@auth.project_access_required
def add_volumes_to_group(project_id):
    return _add_to_group(project_id, VOLUME_KIND)


@app.route("/api/projects/<project_id>/volumes/<int:volume_id>/preview.png", methods=["GET"])
@auth.project_access_required
def volume_preview(project_id, volume_id):
    """The Display button: three orthogonal projections over three central
    slices (Image::CreateOrthogonalProjectionsImage), the way the ab-initio
    panel shows a reconstruction."""
    conn = db.get_conn(project_id)
    row = conn.execute("SELECT * FROM VOLUME_ASSETS WHERE VOLUME_ASSET_ID=?", (volume_id,)).fetchone()
    conn.close()
    if row is None:
        return jsonify({"error": "no such volume"}), 404
    path = row["FILENAME"]
    if not path or not Path(path).is_file():
        return jsonify({"error": "volume file is missing: {}".format(path)}), 404
    stat = Path(path).stat()
    etag = '"r{}-orth-{}-{}-{}"'.format(preview.RENDER_VERSION, volume_id, int(stat.st_mtime), stat.st_size)
    if request.headers.get("If-None-Match") == etag:
        return "", 304
    try:
        png, _meta = volumes.orthogonal_views_png(path)
    except (ValueError, OSError) as exc:
        return jsonify({"error": str(exc)}), 422
    response = app.response_class(png, mimetype="image/png")
    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = "private, no-cache"
    return response


@app.route("/api/projects/<project_id>/startups", methods=["GET"])
@auth.project_access_required
def list_startups(project_id):
    """Every ab-initio run (STARTUP_LIST) with its settings and result volumes."""
    conn = db.get_conn(project_id)
    out = volumes.list_startups(conn)
    conn.close()
    return jsonify({"startups": out})


@app.route("/api/projects/<project_id>/abinitio/defaults", methods=["GET"])
@auth.project_access_required
def abinitio_defaults(project_id):
    """AbInitio3DPanel::SetDefaults() for a package: symmetry, mask radius
    (0.75 x largest dimension), search range (0.4 x), class count."""
    package_id = request.args.get("refinement_package_id", type=int)
    conn = db.get_conn(project_id)
    pkg = conn.execute("SELECT * FROM REFINEMENT_PACKAGE_ASSETS WHERE REFINEMENT_PACKAGE_ASSET_ID=?", (package_id,)).fetchone() if package_id is not None else None
    conn.close()
    if pkg is None:
        return jsonify({"error": "no such refinement package"}), 404
    out = abinitio.package_defaults(pkg)
    out["defaults"] = abinitio.DEFAULTS
    # The class selections a class-average run can start from, with what
    # BeginRefinementCycle() derives from each: the smallest class and the
    # number of averages it would make per class at the default 5 images.
    conn = db.get_conn(project_id)
    try:
        sels = classification.list_selections(conn, package_id=package_id)
        for sel in sels:
            counts = [classification.selection_particle_count(conn, sel["classification_id"], [k]) for k in sel["classes"]]
            sel["smallest_class"] = min(counts) if counts else 0
            cls = conn.execute("SELECT NAME FROM CLASSIFICATION_LIST WHERE CLASSIFICATION_ID=?", (sel["classification_id"],)).fetchone()
            sel["classification_name"] = cls["NAME"] if cls else None
        out["selections"] = [s for s in sels if s["classes"] and s["smallest_class"] > 0]
    finally:
        conn.close()
    return jsonify(out)


@app.route("/api/projects/<project_id>/jobs/<job_id>/abinitio/current.png", methods=["GET"])
@auth.project_access_required
def abinitio_current_picture(project_id, job_id):
    """Orthogonal views of a running (or finished) ab-initio or Refine 3D
    job's current reconstruction, for the Jobs tab's live view."""
    row = _fetch_job_row(project_id, job_id)
    driver = DRIVERS.get(row["STAGE"]) if row is not None else None
    if driver is None or not hasattr(driver, "current_picture"):
        return jsonify({"error": "not a 3D job"}), 404
    conn = db.get_conn(project_id)
    try:
        got = driver.current_picture(conn, row, request.args.get("class", default=0, type=int))
    finally:
        conn.close()
    if got is None:
        return jsonify({"error": "no reconstruction yet"}), 404
    png, _meta, path = got
    stat = Path(path).stat()
    response = app.response_class(png, mimetype="image/png")
    response.headers["ETag"] = '"r{}-abinitio-{}-{}-{}"'.format(preview.RENDER_VERSION, job_id, int(stat.st_mtime), stat.st_size)
    response.headers["Cache-Control"] = "private, no-cache"
    return response


# ---- 3D refinements (REFINEMENT_LIST) -- MyRefinementResultsPanel ----

@app.route("/api/projects/<project_id>/refinements", methods=["GET"])
@auth.project_access_required
def list_refinements(project_id):
    """Every refinement, or one package's (`?refinement_package_id=`), with
    per-class estimated resolution, occupancy and reconstructed volume."""
    conn = db.get_conn(project_id)
    out = refinements.list_refinements(conn, request.args.get("refinement_package_id", type=int))
    conn.close()
    return jsonify({"refinements": out})


@app.route("/api/projects/<project_id>/refinements/<int:refinement_id>", methods=["GET"])
@auth.project_access_required
def get_refinement(project_id, refinement_id):
    """One refinement with each class's FSC / SSNR curve and angular distribution."""
    conn = db.get_conn(project_id)
    d = refinements.get_refinement(conn, refinement_id)
    conn.close()
    if d is None:
        return jsonify({"error": "no such refinement"}), 404
    return jsonify(d)


@app.route("/api/projects/<project_id>/refine3d/defaults", methods=["GET"])
@auth.project_access_required
def refine3d_defaults(project_id):
    """MyRefine3DPanel::SetDefaults() for a package (`?refinement_package_id=`):
    the size-derived limits, the refinements that can be the input
    parameters, the current reference volume of each class, and the volumes
    a mask can be picked from."""
    package_id = request.args.get("refinement_package_id", type=int)
    conn = db.get_conn(project_id)
    try:
        pkg = conn.execute("SELECT * FROM REFINEMENT_PACKAGE_ASSETS WHERE REFINEMENT_PACKAGE_ASSET_ID=?", (package_id,)).fetchone() if package_id is not None else None
        if pkg is None:
            return jsonify({"error": "no such refinement package"}), 404
        out = refine3d.package_defaults(pkg)
        out["defaults"] = refine3d.DEFAULTS
        # The Generate 3D and Refine CTF panels share this route: their own
        # size-derived defaults ride along.
        out["particle_size"] = float(pkg["PARTICLE_SIZE"] or 150.0)
        out["generate3d"] = generate3d.package_defaults(pkg)
        out["refine_ctf"] = refinectf.package_defaults(pkg)
        out["refinements"] = [{"refinement_id": r["refinement_id"], "name": r["name"], "number_of_classes": r["number_of_classes"],
                               "datetime_of_run": r["datetime_of_run"], "starting_refinement_id": r["starting_refinement_id"]}
                              for r in refinements.list_refinements(conn, package_id)]
        out["last_refinement_id"] = pkg["LAST_REFINEMENT_ID"]
        refs = refinements.current_references(conn, package_id)
        out["references"] = []
        for k in range(1, int(pkg["NUMBER_OF_CLASSES"] or 1) + 1):
            vid = refs.get(k, -1)
            vol = conn.execute("SELECT NAME, FILENAME FROM VOLUME_ASSETS WHERE VOLUME_ASSET_ID=?", (vid,)).fetchone() if vid is not None and vid >= 0 else None
            out["references"].append({"class_number": k, "volume_asset_id": vid if vol else -1, "volume_name": vol["NAME"] if vol else None})
        out["volumes"] = [{"volume_asset_id": r["VOLUME_ASSET_ID"], "name": r["NAME"], "x_size": r["X_SIZE"], "pixel_size": r["PIXEL_SIZE"]}
                          for r in conn.execute("SELECT * FROM VOLUME_ASSETS ORDER BY VOLUME_ASSET_ID").fetchall()]
        return jsonify(out)
    finally:
        conn.close()


@app.route("/api/projects/<project_id>/auto-refine3d/defaults", methods=["GET"])
@auth.project_access_required
def auto_refine3d_defaults(project_id):
    """AutoRefine3DPanel::SetDefaults() for a package (`?refinement_package_id=`):
    the size-derived limits, the volumes the Starting Reference and mask
    pickers list, and the volume to preselect (the package's current
    reference for class 1, else the newest volume)."""
    package_id = request.args.get("refinement_package_id", type=int)
    conn = db.get_conn(project_id)
    try:
        pkg = conn.execute("SELECT * FROM REFINEMENT_PACKAGE_ASSETS WHERE REFINEMENT_PACKAGE_ASSET_ID=?", (package_id,)).fetchone() if package_id is not None else None
        if pkg is None:
            return jsonify({"error": "no such refinement package"}), 404
        out = autorefine.package_defaults(pkg)
        out["defaults"] = autorefine.DEFAULTS
        out["number_of_classes"] = int(pkg["NUMBER_OF_CLASSES"] or 1)
        out["volumes"] = [{"volume_asset_id": r["VOLUME_ASSET_ID"], "name": r["NAME"], "x_size": r["X_SIZE"], "pixel_size": r["PIXEL_SIZE"],
                           "fits": r["X_SIZE"] == pkg["STACK_BOX_SIZE"] and abs(float(r["PIXEL_SIZE"] or 0) - float(pkg["OUTPUT_PIXEL_SIZE"] or 0)) <= 0.01}
                          for r in conn.execute("SELECT * FROM VOLUME_ASSETS ORDER BY VOLUME_ASSET_ID").fetchall()]
        refs = refinements.current_references(conn, package_id)
        suggested = refs.get(1) if refs.get(1) is not None and refs.get(1) >= 0 else None
        if suggested is None or not any(v["volume_asset_id"] == suggested for v in out["volumes"]):
            fitting = [v for v in out["volumes"] if v["fits"]]
            suggested = fitting[-1]["volume_asset_id"] if fitting else None
        out["suggested_reference_id"] = suggested
        return jsonify(out)
    finally:
        conn.close()


@app.route("/api/projects/<project_id>/jobs/<job_id>/refinectf/<which>.png", methods=["GET"])
@auth.project_access_required
def refinectf_picture(project_id, job_id, which):
    """A Refine CTF job's beam-tilt pictures: the measured phase-difference
    spectrum (`phase_difference`) or the phase pattern the found tilt
    predicts (`beam_tilt`), from Assets/PhaseDifferences."""
    if which not in ("phase_difference", "beam_tilt"):
        return jsonify({"error": "unknown picture"}), 404
    row = _fetch_job_row(project_id, job_id)
    if row is None or row["STAGE"] != refinectf.STAGE:
        return jsonify({"error": "not a Refine CTF job"}), 404
    conn = db.get_conn(project_id)
    try:
        path = refinectf.beam_tilt_picture(conn, row, which)
    finally:
        conn.close()
    if path is None:
        return jsonify({"error": "no beam-tilt estimate yet"}), 404
    return _file_preview_response(path, "refinectf-{}-{}".format(which, job_id), which.replace("_", " "))


# ---- Sharpen 3D (Sharpen3DPanel) -- not a job: one synchronous run of sharpen_map ----

@app.route("/api/projects/<project_id>/sharpen/defaults", methods=["GET"])
@auth.project_access_required
def sharpen_defaults(project_id):
    """Sharpen3DPanel::OnVolumeComboBox() for a volume (`?volume_asset_id=`):
    the mask radii of the reconstruction that made it, whether its
    refinement's statistics are available for FOM weighting, the estimated
    resolution (the cut-off default), and the panel's defaults."""
    volume_id = request.args.get("volume_asset_id", type=int)
    if volume_id is None:
        return jsonify({"error": "volume_asset_id is required"}), 400
    conn = db.get_conn(project_id)
    try:
        try:
            ctx, _vol = sharpen.volume_context(conn, volume_id)
        except LookupError as exc:
            return jsonify({"error": str(exc)}), 404
        ctx["volumes"] = [{"volume_asset_id": r["VOLUME_ASSET_ID"], "name": r["NAME"], "x_size": r["X_SIZE"], "pixel_size": r["PIXEL_SIZE"]}
                          for r in conn.execute("SELECT * FROM VOLUME_ASSETS ORDER BY VOLUME_ASSET_ID").fetchall()]
        ctx["available"] = shutil.which("sharpen_map") is not None
        return jsonify(ctx)
    finally:
        conn.close()


@app.route("/api/projects/<project_id>/sharpen", methods=["POST"])
@auth.project_access_required
def sharpen_run(project_id):
    """Body `{volume_asset_id, params}` -> the sharpened map's Guinier
    curves and central slices, plus a `result_id` for Save / Import.
    Runs `sharpen_map` directly (`503` if it isn't on the server's PATH);
    writes nothing to the project until the result is imported."""
    body = request.get_json(force=True, silent=True) or {}
    volume_id = body.get("volume_asset_id")
    if volume_id in (None, ""):
        return jsonify({"error": "volume_asset_id is required"}), 400
    executable = shutil.which("sharpen_map")
    if not executable:
        return jsonify({"error": "sharpen_map is not on the server's PATH, so there is nothing to sharpen with"}), 503
    conn = db.get_conn(project_id)
    try:
        started = time.time()
        try:
            out = sharpen.run(conn, project_id, int(volume_id), body.get("params") or {}, executable)
        except LookupError as exc:
            return jsonify({"error": str(exc)}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except TimeoutError as exc:
            return jsonify({"error": str(exc)}), 504
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 502
        out["elapsed_s"] = round(time.time() - started, 2)
        return jsonify(out)
    finally:
        conn.close()


@app.route("/api/projects/<project_id>/sharpen/<result_id>/volume.mrc", methods=["GET"])
@auth.project_access_required
def sharpen_download(project_id, result_id):
    """Save Result: the sharpened map as an MRC file."""
    r = sharpen.result(project_id, result_id)
    if r is None:
        return jsonify({"error": "no such sharpening result (results are kept until the server restarts)"}), 404
    directory, name = os.path.split(r["path"])
    return send_from_directory(directory, name, as_attachment=True,
                               download_name="{}_sharpened.mrc".format("".join(c if c.isalnum() or c in "-_" else "_" for c in r["volume_name"])),
                               mimetype="application/octet-stream")


@app.route("/api/projects/<project_id>/sharpen/<result_id>/import", methods=["POST"])
@auth.project_access_required
def sharpen_import(project_id, result_id):
    """Import: the sharpened map becomes a volume asset (`{name}` optional)."""
    body = request.get_json(force=True, silent=True) or {}
    conn = db.get_conn(project_id)
    try:
        try:
            out = sharpen.import_result(conn, project_id, result_id, body.get("name"))
        except LookupError as exc:
            return jsonify({"error": str(exc)}), 404
        return jsonify(out), 201
    finally:
        conn.close()


@app.route("/api/projects/<project_id>/refinement-packages/<int:package_id>/references", methods=["PATCH"])
@auth.project_access_required
def set_package_reference(project_id, package_id):
    """MyRefine3DPanel's Active 3D References list: `{class_number, volume_asset_id}`
    (-1 for "generate from parameters") sets a class's current reference."""
    body = request.get_json(force=True, silent=True) or {}
    k, vid = body.get("class_number"), body.get("volume_asset_id")
    if k is None or vid is None:
        return jsonify({"error": "class_number and volume_asset_id are required"}), 400
    conn = db.get_conn(project_id)
    try:
        if int(vid) >= 0 and conn.execute("SELECT 1 FROM VOLUME_ASSETS WHERE VOLUME_ASSET_ID=?", (int(vid),)).fetchone() is None:
            return jsonify({"error": "no such volume"}), 404
        refinements.set_current_reference(conn, package_id, int(k), int(vid))
        return jsonify({"class_number": int(k), "volume_asset_id": int(vid)})
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 2D classifications (CLASSIFICATION_LIST) -- what Refine2DResultsPanel
# reads, plus the pictures it shows: the class averages as a montage and
# the members of one class cut from the package's stack.
# ---------------------------------------------------------------------------

@app.route("/api/projects/<project_id>/classifications", methods=["GET"])
@auth.project_access_required
def list_classifications(project_id):
    """Every classification, or those of one package (`?refinement_package_id=`),
    with its package's name and the job that made it."""
    package_id = request.args.get("refinement_package_id", type=int)
    conn = db.get_conn(project_id)
    out = classification.list_classifications(conn, package_id)
    conn.close()
    return jsonify({"classifications": out})


@app.route("/api/projects/<project_id>/class2d/defaults", methods=["GET"])
@auth.project_access_required
def class2d_defaults(project_id):
    """MyRefine2DPanel::SetDefaults() for the chosen package: the class
    count it would pick for a new classification, and, per earlier
    classification, the class count and high-resolution limit a run
    continuing from it inherits."""
    package_id = request.args.get("refinement_package_id", type=int)
    conn = db.get_conn(project_id)
    try:
        pkg = classification.package_row(conn, package_id) if package_id is not None else None
        if pkg is None:
            return jsonify({"error": "no such refinement package"}), 404
        particles = len(classification.package_particles(conn, package_id))
        return jsonify({
            "refinement_package_id": package_id,
            "particle_count": particles,
            "number_of_classes": classification.default_number_of_classes(particles),
            "defaults": classification.DEFAULTS,
            "classifications": [{"classification_id": c["classification_id"], "name": c["name"],
                                 "number_of_classes": c["number_of_classes"], "high_resolution_limit": c["high_resolution_limit"],
                                 "class_average_file_exists": c["class_average_file_exists"]}
                                for c in classification.list_classifications(conn, package_id)],
        })
    finally:
        conn.close()


@app.route("/api/projects/<project_id>/classifications/<int:classification_id>", methods=["GET"])
@auth.project_access_required
def get_classification(project_id, classification_id):
    conn = db.get_conn(project_id)
    d = classification.get_classification(conn, classification_id)
    conn.close()
    if d is None:
        return jsonify({"error": "no such classification"}), 404
    d["montage"] = classification.montage_geometry(d["number_of_classes"], box=d.get("stack_box_size"))
    return jsonify(d)


@app.route("/api/projects/<project_id>/classifications/<int:classification_id>", methods=["DELETE"])
@auth.project_access_required
def delete_classification(project_id, classification_id):
    conn = db.get_conn(project_id)
    ok = classification.delete_classification(conn, classification_id)
    conn.close()
    if not ok:
        return jsonify({"error": "no such classification"}), 404
    return jsonify({"deleted": classification_id})


def _montage_response(path, sections, etag_key, what):
    if not path or not Path(path).is_file():
        return jsonify({"error": "{} file is missing: {}".format(what, path)}), 404
    stat = Path(path).stat()
    etag = '"r{}-m{}-{}-{}-{}"'.format(preview.RENDER_VERSION, classification.MONTAGE_TILE, etag_key, int(stat.st_mtime), stat.st_size)
    if request.headers.get("If-None-Match") == etag:
        return "", 304
    try:
        png, _meta = classification.render_montage(path, sections)
    except (preview.PreviewError, ValueError, OSError) as exc:
        return jsonify({"error": str(exc)}), 422
    response = app.response_class(png, mimetype="image/png")
    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = "private, no-cache"
    return response


@app.route("/api/projects/<project_id>/classifications/<int:classification_id>/averages.png", methods=["GET"])
@auth.project_access_required
def classification_averages_png(project_id, classification_id):
    """The class averages tiled into one picture (ClassumDisplayPanel)."""
    conn = db.get_conn(project_id)
    row = conn.execute("SELECT CLASS_AVERAGE_FILE FROM CLASSIFICATION_LIST WHERE CLASSIFICATION_ID=?", (classification_id,)).fetchone()
    conn.close()
    if row is None:
        return jsonify({"error": "no such classification"}), 404
    return _montage_response(row["CLASS_AVERAGE_FILE"], None, "averages-{}".format(classification_id), "class averages")


@app.route("/api/projects/<project_id>/classifications/<int:classification_id>/class/<int:class_number>", methods=["GET"])
@auth.project_access_required
def classification_class_members(project_id, classification_id, class_number):
    """Which particles a class holds (Refine2DResultsPanel's Class Members
    list): the ones that took part in the round first, then those whose
    best class this was before they sat the round out."""
    limit = request.args.get("limit", default=CLASS_MEMBER_LIMIT, type=int)
    conn = db.get_conn(project_id)
    members, total = classification.class_members(conn, classification_id, class_number, limit)
    conn.close()
    return jsonify({"classification_id": classification_id, "class_number": class_number, "members": members, "total": total,
                    "montage": classification.montage_geometry(len(members))})


@app.route("/api/projects/<project_id>/classifications/<int:classification_id>/class/<int:class_number>/members.png", methods=["GET"])
@auth.project_access_required
def classification_class_members_png(project_id, classification_id, class_number):
    """The class's members cut from the package's particle stack, tiled
    (ParticleDisplayPanel)."""
    limit = request.args.get("limit", default=CLASS_MEMBER_LIMIT, type=int)
    conn = db.get_conn(project_id)
    row = conn.execute("SELECT rp.STACK_FILENAME FROM CLASSIFICATION_LIST c JOIN REFINEMENT_PACKAGE_ASSETS rp "
                       "ON rp.REFINEMENT_PACKAGE_ASSET_ID = c.REFINEMENT_PACKAGE_ASSET_ID WHERE c.CLASSIFICATION_ID=?",
                       (classification_id,)).fetchone()
    members, _total = classification.class_members(conn, classification_id, class_number, limit)
    conn.close()
    if row is None:
        return jsonify({"error": "no such classification"}), 404
    if not members:
        return jsonify({"error": "class {} has no members".format(class_number)}), 404
    return _montage_response(row["STACK_FILENAME"], [m["position_in_stack"] for m in members],
                             "members-{}-{}-{}".format(classification_id, class_number, limit), "particle stack")


CLASS_MEMBER_LIMIT = 100


# ---- class selections (Refine2DResultsPanel's selection manager) ----

@app.route("/api/projects/<project_id>/classification-selections", methods=["GET"])
@auth.project_access_required
def list_classification_selections(project_id):
    """Selections, all or one classification's (`?classification_id=`) or one
    package's (`?refinement_package_id=`), each with its class numbers and
    how many particles those classes hold."""
    conn = db.get_conn(project_id)
    out = classification.list_selections(conn, request.args.get("classification_id", type=int),
                                         request.args.get("refinement_package_id", type=int))
    conn.close()
    return jsonify({"selections": out})


@app.route("/api/projects/<project_id>/classification-selections", methods=["POST"])
@auth.project_access_required
def create_classification_selection(project_id):
    body = request.get_json(force=True, silent=True) or {}
    if body.get("classification_id") is None:
        return jsonify({"error": "classification_id is required"}), 400
    conn = db.get_conn(project_id)
    try:
        try:
            sid = classification.create_selection(conn, body["classification_id"], body.get("name"), body.get("classes") or [])
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(classification.get_selection(conn, sid)), 201
    finally:
        conn.close()


@app.route("/api/projects/<project_id>/classification-selections/<int:selection_id>", methods=["PATCH"])
@auth.project_access_required
def update_classification_selection(project_id, selection_id):
    """`{name}` renames; `{classes: [...]}` replaces the membership (a click
    toggles one class, Clear sends [], Invert sends the complement)."""
    body = request.get_json(force=True, silent=True) or {}
    conn = db.get_conn(project_id)
    try:
        if classification.get_selection(conn, selection_id) is None:
            return jsonify({"error": "no such selection"}), 404
        try:
            if "name" in body:
                classification.rename_selection(conn, selection_id, body["name"])
            if "classes" in body:
                classification.set_selection_classes(conn, selection_id, body["classes"] or [])
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(classification.get_selection(conn, selection_id))
    finally:
        conn.close()


@app.route("/api/projects/<project_id>/classification-selections/<int:selection_id>", methods=["DELETE"])
@auth.project_access_required
def delete_classification_selection(project_id, selection_id):
    conn = db.get_conn(project_id)
    ok = classification.delete_selection(conn, selection_id)
    conn.close()
    if not ok:
        return jsonify({"error": "no such selection"}), 404
    return jsonify({"deleted": selection_id})


# ---------------------------------------------------------------------------
# Results (project-scoped, read-only)
#
# What cisTEM's Results tab reads: for Align Movies, MOVIE_ALIGNMENT_LIST
# (MyMovieAlignResultsPanel's list and Job Details) and the per-alignment
# MOVIE_ALIGNMENT_PARAMETERS_<id> shift table (UnblurResultsPanel's drift
# plot), plus renders of the aligned sum and the amplitude spectrum unblur
# wrote beside it.
# ---------------------------------------------------------------------------

_ALIGNMENT_COLUMNS = (
    "ALIGNMENT_ID", "DATETIME_OF_RUN", "ALIGNMENT_JOB_ID", "MOVIE_ASSET_ID", "OUTPUT_FILE", "VOLTAGE",
    "PIXEL_SIZE", "EXPOSURE_PER_FRAME", "PRE_EXPOSURE_AMOUNT", "MIN_SHIFT", "MAX_SHIFT", "SHOULD_DOSE_FILTER",
    "SHOULD_RESTORE_POWER", "TERMINATION_THRESHOLD", "MAX_ITERATIONS", "BFACTOR", "SHOULD_MASK_CENTRAL_CROSS",
    "HORIZONTAL_MASK", "VERTICAL_MASK", "SHOULD_INCLUDE_ALL_FRAMES_IN_SUM", "FIRST_FRAME_TO_SUM",
    "LAST_FRAME_TO_SUM", "FINAL_PIXEL_SIZE",
)


def _spectrum_path(output_file):
    """unblur writes the amplitude spectrum to Spectra/ beside the sum, under
    the same name (stages/unblur.py builds both paths the same way)."""
    p = Path(output_file)
    return p.parent / "Spectra" / p.name


def _alignment_json(row, conn=None):
    d = {c.lower(): row[c] for c in _ALIGNMENT_COLUMNS}
    d["movie_name"] = row["MOVIE_NAME"]
    d["movie_filename"] = row["MOVIE_FILENAME"]
    # ia.IMAGE_ASSET_ID joins on ALIGNMENT_ID, so it is set only for the
    # alignment the movie's image asset currently points at: the active one.
    d["image_asset_id"] = row["IMAGE_ASSET_ID"]
    d["is_active"] = row["IMAGE_ASSET_ID"] is not None
    d["movie_image_asset_id"] = row["MOVIE_IMAGE_ASSET_ID"]
    d["job_number"] = row["JOB_NUMBER"]
    out = d["output_file"] or ""
    d["output_file_exists"] = bool(out) and Path(out).is_file()
    d["spectrum_file_exists"] = bool(out) and _spectrum_path(out).is_file()
    if conn is not None:
        try:
            d["frame_count"] = conn.execute(
                "SELECT COUNT(*) FROM MOVIE_ALIGNMENT_PARAMETERS_{}".format(row["ALIGNMENT_ID"])).fetchone()[0]
        except Exception:  # noqa: BLE001 -- a simulated job never wrote the table
            d["frame_count"] = None
    return d


_ALIGNMENT_SELECT = (
    "SELECT al.*, ma.NAME AS MOVIE_NAME, ma.FILENAME AS MOVIE_FILENAME, ia.IMAGE_ASSET_ID, j.JOB_NUMBER, "
    "(SELECT MIN(IMAGE_ASSET_ID) FROM IMAGE_ASSETS mi WHERE mi.PARENT_MOVIE_ID = al.MOVIE_ASSET_ID) AS MOVIE_IMAGE_ASSET_ID "
    "FROM MOVIE_ALIGNMENT_LIST al "
    "JOIN MOVIE_ASSETS ma ON ma.MOVIE_ASSET_ID = al.MOVIE_ASSET_ID "
    "LEFT JOIN IMAGE_ASSETS ia ON ia.ALIGNMENT_ID = al.ALIGNMENT_ID "
    "LEFT JOIN JOBS j ON j.JOB_ID = al.ALIGNMENT_JOB_ID "
)


@app.route("/api/projects/<project_id>/alignments", methods=["GET"])
@auth.project_access_required
def list_alignments(project_id):
    """Every movie alignment in the project, oldest first per movie, with
    the movie's name and the image asset the alignment produced. The page
    shows the latest per movie by default; earlier ones are history."""
    conn = db.get_conn(project_id)
    rows = conn.execute(_ALIGNMENT_SELECT + "ORDER BY al.MOVIE_ASSET_ID, al.ALIGNMENT_ID").fetchall()
    out = [_alignment_json(r, conn) for r in rows]
    conn.close()
    return jsonify({"alignments": out})


def _alignment_row(project_id, alignment_id):
    conn = db.get_conn(project_id)
    row = conn.execute(_ALIGNMENT_SELECT + "WHERE al.ALIGNMENT_ID = ?", (alignment_id,)).fetchone()
    return conn, row


@app.route("/api/projects/<project_id>/alignments/<int:alignment_id>", methods=["GET"])
@auth.project_access_required
def get_alignment(project_id, alignment_id):
    """One alignment with its per-frame shifts (in Å, as unblur reported
    them) -- the drift plot's data."""
    conn, row = _alignment_row(project_id, alignment_id)
    if row is None:
        conn.close()
        return jsonify({"error": "no such alignment"}), 404
    d = _alignment_json(row, conn)
    try:
        d["shifts"] = [
            {"frame": r["FRAME_NUMBER"], "x": r["X_SHIFT"], "y": r["Y_SHIFT"]}
            for r in conn.execute(
                "SELECT FRAME_NUMBER, X_SHIFT, Y_SHIFT FROM MOVIE_ALIGNMENT_PARAMETERS_{} ORDER BY FRAME_NUMBER".format(
                    alignment_id)).fetchall()
        ]
    except Exception:  # noqa: BLE001
        d["shifts"] = []
    conn.close()
    return jsonify(d)


@app.route("/api/projects/<project_id>/alignments/<int:alignment_id>/activate", methods=["POST"])
@auth.project_access_required
def activate_alignment(project_id, alignment_id):
    """Make this alignment the one the movie's image asset points at --
    cisTEM's checked cell in the Movie Alignment Results grid. Returns the
    alignment as GET does, now with is_active true."""
    conn = db.get_conn(project_id)
    try:
        try:
            image_asset_id = stages.ADAPTERS["motion_correction"].activate_alignment(conn, alignment_id)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 409
        if image_asset_id is None:
            return jsonify({"error": "no such alignment"}), 404
        row = conn.execute(_ALIGNMENT_SELECT + "WHERE al.ALIGNMENT_ID = ?", (alignment_id,)).fetchone()
        return jsonify(_alignment_json(row, conn))
    finally:
        conn.close()


@app.route("/api/projects/<project_id>/jobs/<job_id>/activate-results", methods=["POST"])
@app.route("/api/projects/<project_id>/jobs/<job_id>/activate-alignments", methods=["POST"])
@auth.project_access_required
def activate_job_results(project_id, job_id):
    """Make one job's results active for every asset it processed -- what a
    user wants after a re-run with better parameters, and what a finishing
    job does by itself. Dispatches on the job's stage adapter
    (activate_job_results); the old /activate-alignments path is kept as
    an alias. Assets whose file has gone are reported, not fatal."""
    row = _fetch_job_row(project_id, job_id)
    if row is None:
        return jsonify({"error": "not found"}), 404
    adapter = stages.ADAPTERS.get(row["STAGE"])
    if adapter is None or not hasattr(adapter, "activate_job_results"):
        return jsonify({"error": "this stage has no results to activate"}), 400
    conn = db.get_conn(project_id)
    try:
        total, failed = adapter.activate_job_results(conn, job_id)
        if total == 0:
            return jsonify({"error": "that job has no results"}), 404
        return jsonify({"activated": total - len(failed), "failed": failed})
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Results: Find CTF (MyFindCTFResultsPanel + ShowCTFResultsPanel)
# ---------------------------------------------------------------------------

_CTF_COLUMNS = (
    "CTF_ESTIMATION_ID", "CTF_ESTIMATION_JOB_ID", "DATETIME_OF_RUN", "IMAGE_ASSET_ID", "ESTIMATED_ON_MOVIE_FRAMES",
    "VOLTAGE", "SPHERICAL_ABERRATION", "PIXEL_SIZE", "AMPLITUDE_CONTRAST", "BOX_SIZE", "MIN_RESOLUTION",
    "MAX_RESOLUTION", "MIN_DEFOCUS", "MAX_DEFOCUS", "DEFOCUS_STEP", "RESTRAIN_ASTIGMATISM", "TOLERATED_ASTIGMATISM",
    "FIND_ADDITIONAL_PHASE_SHIFT", "MIN_PHASE_SHIFT", "MAX_PHASE_SHIFT", "PHASE_SHIFT_STEP", "DEFOCUS1", "DEFOCUS2",
    "DEFOCUS_ANGLE", "ADDITIONAL_PHASE_SHIFT", "SCORE", "DETECTED_RING_RESOLUTION", "DETECTED_ALIAS_RESOLUTION",
    "OUTPUT_DIAGNOSTIC_FILE", "NUMBER_OF_FRAMES_AVERAGED", "LARGE_ASTIGMATISM_EXPECTED", "ICINESS", "TILT_ANGLE", "TILT_AXIS",
)

_CTF_SELECT = (
    "SELECT ce.*, ia.NAME AS IMAGE_NAME, ia.FILENAME AS IMAGE_FILENAME, "
    "(ia.CTF_ESTIMATION_ID = ce.CTF_ESTIMATION_ID) AS IS_ACTIVE, j.JOB_NUMBER "
    "FROM ESTIMATED_CTF_PARAMETERS ce "
    "JOIN IMAGE_ASSETS ia ON ia.IMAGE_ASSET_ID = ce.IMAGE_ASSET_ID "
    "LEFT JOIN JOBS j ON j.JOB_ID = ce.CTF_ESTIMATION_JOB_ID "
)


def _ctf_json(row):
    d = {c.lower(): row[c] for c in _CTF_COLUMNS}
    d["image_name"] = row["IMAGE_NAME"]
    d["image_filename"] = row["IMAGE_FILENAME"]
    d["is_active"] = bool(row["IS_ACTIVE"])
    d["job_number"] = row["JOB_NUMBER"]
    out = d["output_diagnostic_file"] or ""
    d["diagnostic_file_exists"] = bool(out) and Path(out).is_file()
    return d


@app.route("/api/projects/<project_id>/ctf-estimates", methods=["GET"])
@auth.project_access_required
def list_ctf_estimates(project_id):
    """Every CTF estimate with its image's name, job number, and whether the
    image asset points at it (the active one). Read-only; feeds the Results
    tab's Find CTF grid."""
    conn = db.get_conn(project_id)
    rows = conn.execute(_CTF_SELECT + "ORDER BY ce.IMAGE_ASSET_ID, ce.CTF_ESTIMATION_ID").fetchall()
    conn.close()
    return jsonify({"ctf_estimates": [_ctf_json(r) for r in rows]})


@app.route("/api/projects/<project_id>/ctf-estimates/<int:estimate_id>", methods=["GET"])
@auth.project_access_required
def get_ctf_estimate(project_id, estimate_id):
    """One estimate plus `plot` -- the 1D curves from the _avrot.txt beside
    its diagnostic image (frequency in 1/A, smoothed spectrum, fit,
    quality), or null if that file is missing."""
    conn = db.get_conn(project_id)
    row = conn.execute(_CTF_SELECT + "WHERE ce.CTF_ESTIMATION_ID = ?", (estimate_id,)).fetchone()
    conn.close()
    if row is None:
        return jsonify({"error": "no such CTF estimate"}), 404
    d = _ctf_json(row)
    d["plot"] = stages.ADAPTERS["ctf_estimation"].read_avrot(d["output_diagnostic_file"]) if d["output_diagnostic_file"] else None
    return jsonify(d)


@app.route("/api/projects/<project_id>/ctf-estimates/<int:estimate_id>/activate", methods=["POST"])
@auth.project_access_required
def activate_ctf_estimate(project_id, estimate_id):
    """Point the image asset at this estimate -- cisTEM's checked cell in
    the CTF results grid. Returns the estimate as GET does, is_active true."""
    conn = db.get_conn(project_id)
    try:
        try:
            image_id = stages.ADAPTERS["ctf_estimation"].activate_estimate(conn, estimate_id)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 409
        if image_id is None:
            return jsonify({"error": "no such CTF estimate"}), 404
        row = conn.execute(_CTF_SELECT + "WHERE ce.CTF_ESTIMATION_ID = ?", (estimate_id,)).fetchone()
        return jsonify(_ctf_json(row))
    finally:
        conn.close()


@app.route("/api/projects/<project_id>/ctf-estimates/<int:estimate_id>/diagnostic.png", methods=["GET"])
@auth.project_access_required
def ctf_diagnostic_preview(project_id, estimate_id):
    conn = db.get_conn(project_id)
    row = conn.execute("SELECT OUTPUT_DIAGNOSTIC_FILE FROM ESTIMATED_CTF_PARAMETERS WHERE CTF_ESTIMATION_ID=?", (estimate_id,)).fetchone()
    conn.close()
    if row is None:
        return jsonify({"error": "no such CTF estimate"}), 404
    return _file_preview_response(row["OUTPUT_DIAGNOSTIC_FILE"], "ctf-diagnostic-{}".format(estimate_id), "diagnostic image")


# ---------------------------------------------------------------------------
# Results: Find Particles (MyPickingResultsPanel + PickingResultsDisplayPanel)
# ---------------------------------------------------------------------------

_PICK_COLUMNS = (
    "PICKING_ID", "DATETIME_OF_RUN", "PICKING_JOB_ID", "PARENT_IMAGE_ASSET_ID", "PICKING_ALGORITHM",
    "CHARACTERISTIC_RADIUS", "MAXIMUM_RADIUS", "THRESHOLD_PEAK_HEIGHT", "HIGHEST_RESOLUTION_USED_IN_PICKING",
    "MIN_DIST_FROM_EDGES", "AVOID_HIGH_VARIANCE", "AVOID_HIGH_LOW_MEAN", "NUM_BACKGROUND_BOXES", "MANUAL_EDIT",
)

_PICK_SELECT = (
    "SELECT pl.*, ia.NAME AS IMAGE_NAME, ia.FILENAME AS IMAGE_FILENAME, ia.X_SIZE, ia.Y_SIZE, ia.PIXEL_SIZE, "
    "(ia.ACTIVE_PICKING_ID = pl.PICKING_ID) AS IS_ACTIVE, j.JOB_NUMBER, "
    "ce.DEFOCUS1, ce.DEFOCUS2, ce.ICINESS "
    "FROM PARTICLE_PICKING_LIST pl "
    "JOIN IMAGE_ASSETS ia ON ia.IMAGE_ASSET_ID = pl.PARENT_IMAGE_ASSET_ID "
    "LEFT JOIN JOBS j ON j.JOB_ID = pl.PICKING_JOB_ID "
    "LEFT JOIN ESTIMATED_CTF_PARAMETERS ce ON ce.CTF_ESTIMATION_ID = ia.CTF_ESTIMATION_ID "
)


def _pick_json(row, conn):
    d = {c.lower(): row[c] for c in _PICK_COLUMNS}
    d["image_asset_id"] = row["PARENT_IMAGE_ASSET_ID"]
    d["image_name"] = row["IMAGE_NAME"]
    d["image_filename"] = row["IMAGE_FILENAME"]
    d["x_size"], d["y_size"], d["pixel_size"] = row["X_SIZE"], row["Y_SIZE"], row["PIXEL_SIZE"]
    d["is_active"] = bool(row["IS_ACTIVE"])
    d["job_number"] = row["JOB_NUMBER"]
    d["defocus1"], d["defocus2"], d["iciness"] = row["DEFOCUS1"], row["DEFOCUS2"], row["ICINESS"]
    d["image_file_exists"] = bool(row["IMAGE_FILENAME"]) and Path(row["IMAGE_FILENAME"]).is_file()
    table = stages.find_particles.results_table(row["PICKING_JOB_ID"])
    if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
        d["pick_count"] = conn.execute("SELECT COUNT(*) FROM {} WHERE PICKING_ID=?".format(table), (row["PICKING_ID"],)).fetchone()[0]
    else:
        d["pick_count"] = None
    return d


@app.route("/api/projects/<project_id>/picks", methods=["GET"])
@auth.project_access_required
def list_picks(project_id):
    """Every particle picking (one PARTICLE_PICKING_LIST row per image per
    job) with its image's name and size, job number, pick count and whether
    it is the image's active one. Read-only; feeds the Results tab's grid."""
    conn = db.get_conn(project_id)
    rows = conn.execute(_PICK_SELECT + "ORDER BY pl.PARENT_IMAGE_ASSET_ID, pl.PICKING_ID").fetchall()
    out = [_pick_json(r, conn) for r in rows]
    conn.close()
    return jsonify({"picks": out})


@app.route("/api/projects/<project_id>/picks/<int:picking_id>", methods=["GET"])
@auth.project_access_required
def get_pick(project_id, picking_id):
    """One picking plus `positions: [{x, y, peak_height}]` in Angstroms from
    the image origin (y up), as find_particles reports them."""
    conn = db.get_conn(project_id)
    row = conn.execute(_PICK_SELECT + "WHERE pl.PICKING_ID = ?", (picking_id,)).fetchone()
    if row is None:
        conn.close()
        return jsonify({"error": "no such picking"}), 404
    d = _pick_json(row, conn)
    d["positions"] = stages.find_particles.picks_for(conn, picking_id)
    conn.close()
    return jsonify(d)


@app.route("/api/projects/<project_id>/picks/<int:picking_id>/activate", methods=["POST"])
@auth.project_access_required
def activate_pick(project_id, picking_id):
    """These picks become the image's particle positions (cisTEM's checked
    cell in the picking results grid)."""
    conn = db.get_conn(project_id)
    try:
        try:
            image_id = stages.find_particles.activate_picking(conn, picking_id)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 409
        if image_id is None:
            return jsonify({"error": "no such picking"}), 404
        row = conn.execute(_PICK_SELECT + "WHERE pl.PICKING_ID = ?", (picking_id,)).fetchone()
        return jsonify(_pick_json(row, conn))
    finally:
        conn.close()


@app.route("/api/projects/<project_id>/preview/pick", methods=["POST"])
@auth.project_access_required
def preview_pick(project_id):
    """Run the particle picker on one image with the panel's current
    parameters and return the picks -- cisTEM's Preview / Auto preview.
    Not a job: nothing is written to the project, and the find_particles
    binary is run directly (it has to be on the server's PATH)."""
    body = request.get_json(force=True, silent=True) or {}
    image_id = body.get("image_asset_id")
    if image_id is None:
        return jsonify({"error": "image_asset_id is required"}), 400
    executable = shutil.which("find_particles")
    if not executable:
        return jsonify({"error": "find_particles is not on the server's PATH, so there is nothing to preview with"}), 503
    conn = db.get_conn(project_id)
    try:
        try:
            result = stages.find_particles.preview(conn, project_id, int(image_id), body.get("params") or {}, executable)
        except LookupError as exc:
            return jsonify({"error": str(exc)}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 409
        except TimeoutError as exc:
            return jsonify({"error": str(exc)}), 504
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 502
        return jsonify(result)
    finally:
        conn.close()


def _file_preview_response(path, etag_key, what):
    """_preview_response for a file that isn't an asset row: the aligned sum
    or spectrum an alignment points at."""
    if not path or not Path(path).is_file():
        return jsonify({"error": "{} file is missing: {}".format(what, path)}), 404
    if not preview.can_preview(path):
        return jsonify({"error": "previews aren't supported for {} files".format(Path(path).suffix.lower() or "these")}), 415
    stat = Path(path).stat()
    etag = '"r{}-{}-{}-{}"'.format(preview.RENDER_VERSION, etag_key, int(stat.st_mtime), stat.st_size)
    if request.headers.get("If-None-Match") == etag:
        return "", 304
    try:
        png, _meta = preview.render_image_preview(path)
    except preview.PreviewError as exc:
        return jsonify({"error": str(exc)}), 422
    response = app.response_class(png, mimetype="image/png")
    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = "private, no-cache"
    return response


@app.route("/api/projects/<project_id>/alignments/<int:alignment_id>/sum.png", methods=["GET"])
@auth.project_access_required
def alignment_sum_preview(project_id, alignment_id):
    conn, row = _alignment_row(project_id, alignment_id)
    conn.close()
    if row is None:
        return jsonify({"error": "no such alignment"}), 404
    return _file_preview_response(row["OUTPUT_FILE"], "alignment-sum-{}".format(alignment_id), "aligned sum")


@app.route("/api/projects/<project_id>/alignments/<int:alignment_id>/spectrum.png", methods=["GET"])
@auth.project_access_required
def alignment_spectrum_preview(project_id, alignment_id):
    conn, row = _alignment_row(project_id, alignment_id)
    conn.close()
    if row is None:
        return jsonify({"error": "no such alignment"}), 404
    return _file_preview_response(str(_spectrum_path(row["OUTPUT_FILE"] or "")), "alignment-spectrum-{}".format(alignment_id),
                                  "amplitude spectrum")


# ---------------------------------------------------------------------------
# Run profiles (project-scoped)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Run profiles (system-wide). Any logged-in user reads them -- the Run
# Profile picker on every job panel is filled from here -- and only an
# administrator edits them, from the home page. cisTEM keeps them per
# project (RunProfilesPanel under Settings); here they describe the machine
# the server runs on, which is the same for every project.
# ---------------------------------------------------------------------------

@app.route("/api/run-profiles", methods=["GET"])
@auth.login_required
def list_run_profiles():
    """Every run profile with its commands, for the Run Profile picker each
    job panel carries (cf. RunProfileComboBox in cisTEM's AlignMoviesPanel,
    filled from run_profiles_panel) and the home page's editor.

    `total_jobs` is what the start button gates on: a profile with no run
    commands (the seeded Slurm template) can't launch anything, and cisTEM's
    OnUpdateUI greys the button in that case rather than let the job fail.
    """
    conn = db.get_system_conn()
    profiles = db.load_run_profiles(conn)
    conn.close()
    return jsonify({"run_profiles": [_profile_json(p) for p in profiles]})


def _profile_json(p):
    return {
        "run_profile_id": p["run_profile_id"],
        "profile_name": p["name"],
        "manager_run_command": p["manager_command"],
        "gui_address": p["gui_address"],
        "controller_address": p["controller_address"],
        "run_commands": p["run_commands"],
        "total_jobs": p["total_jobs"],
    }


def _profile_spec_from_body(body):
    """The API's field names -> db's. Missing keys stay missing so PATCH
    can be partial."""
    spec = {}
    for api_key, db_key in (("profile_name", "name"), ("manager_run_command", "manager_command"),
                            ("gui_address", "gui_address"), ("controller_address", "controller_address"),
                            ("run_commands", "run_commands")):
        if api_key in body:
            spec[db_key] = body[api_key]
    return spec


@app.route("/api/run-profiles", methods=["POST"])
@auth.admin_required
def create_run_profile():
    """MyRunProfilesPanel's Add, Duplicate and Import in one route. An empty
    body adds cisTEM's "Default Local" profile; `copy_of` duplicates an
    existing one as "Copy of <name>"; a full profile in the body (the shape
    GET returns) imports it. Names are made unique."""
    body = request.get_json(force=True, silent=True) or {}
    conn = db.get_system_conn()
    try:
        if "copy_of" in body:
            source = db.load_run_profile(conn, int(body["copy_of"]))
            if source is None:
                return jsonify({"error": "run profile {} not found".format(body["copy_of"])}), 404
            spec = dict(source, name="Copy of " + source["name"])
        else:
            spec = _profile_spec_from_body(body)
            if not spec:
                spec = db.default_local_profile_spec()
        try:
            pid = db.create_run_profile(conn, spec)
        except db.RunProfileError as exc:
            return jsonify({"error": str(exc)}), 400
        profile = db.load_run_profile(conn, pid)
    finally:
        conn.close()
    return jsonify(_profile_json(profile)), 201


@app.route("/api/run-profiles/<int:run_profile_id>", methods=["PATCH"])
@auth.admin_required
def update_run_profile(run_profile_id):
    """Rename, or the commands panel's Save: any of profile_name,
    manager_run_command, gui_address, controller_address, run_commands."""
    body = request.get_json(force=True, silent=True) or {}
    conn = db.get_system_conn()
    try:
        try:
            db.update_run_profile(conn, run_profile_id, _profile_spec_from_body(body))
        except KeyError:
            return jsonify({"error": "run profile {} not found".format(run_profile_id)}), 404
        except db.RunProfileError as exc:
            return jsonify({"error": str(exc)}), 400
        profile = db.load_run_profile(conn, run_profile_id)
    finally:
        conn.close()
    return jsonify(_profile_json(profile))


@app.route("/api/run-profiles/<int:run_profile_id>", methods=["DELETE"])
@auth.admin_required
def delete_run_profile(run_profile_id):
    conn = db.get_system_conn()
    try:
        deleted = db.delete_run_profile(conn, run_profile_id)
    finally:
        conn.close()
    if not deleted:
        return jsonify({"error": "run profile {} not found".format(run_profile_id)}), 404
    return jsonify({"deleted": run_profile_id})


# ---------------------------------------------------------------------------
# Job routes (project-scoped)
# ---------------------------------------------------------------------------

@app.route("/api/projects/<project_id>/jobs", methods=["GET"])
@auth.project_access_required
def list_jobs(project_id):
    conn = db.get_conn(project_id)
    # The steps of a 2D classification are jobs to the runner but not to the
    # user: the parent row stands for the whole cycle.
    rows = conn.execute("SELECT * FROM JOBS WHERE PARENT_JOB_ID IS NULL ORDER BY CREATED_AT").fetchall()
    jobs = [_row_to_job(r, conn) for r in rows]
    conn.close()
    return jsonify({"jobs": jobs})


@app.route("/api/projects/<project_id>/jobs", methods=["POST"])
@auth.project_access_required
def create_job(project_id):
    body = request.get_json(force=True, silent=True) or {}
    stage = body.get("stage")
    if stage not in STAGE_COMMANDS:
        return jsonify({"error": "unknown stage '{}'".format(stage)}), 400

    params = body.get("params") or {}

    movie_group_id = None
    if stage == "motion_correction":
        movie_group_id = params.get("movie_group_id")
        if movie_group_id is None:
            return jsonify({"error": "movie_group_id is required"}), 400
        conn = db.get_conn(project_id)
        count = conn.execute(
            "SELECT COUNT(*) FROM MOVIE_GROUP_MEMBERS WHERE GROUP_ID=?", (movie_group_id,)
        ).fetchone()[0]
        conn.close()
        if count == 0:
            return jsonify({"error": "movie group has no movies"}), 400
    elif stage in ("ctf_estimation", "particle_picking"):
        # Find CTF and Find Particles consume images the way Align Movies
        # consumes movies: a group, whose members' metadata (voltage, Cs,
        # pixel size, parent movie, CTF estimate) is what the tasks are
        # built from.
        image_group_id = params.get("image_group_id")
        if image_group_id is None:
            return jsonify({"error": "image_group_id is required"}), 400
        conn = db.get_conn(project_id)
        count = conn.execute(
            "SELECT COUNT(*) FROM IMAGE_GROUP_MEMBERS WHERE GROUP_ID=?", (image_group_id,)
        ).fetchone()[0]
        # cisTEM's can_be_picked: every image in the group needs a CTF estimate first.
        missing_ctf = stages.find_particles.images_without_ctf(conn, image_group_id) if stage == "particle_picking" else []
        conn.close()
        if count == 0:
            return jsonify({"error": "image group has no images"}), 400
        if missing_ctf:
            return jsonify({"error": stages.find_particles.CTF_REQUIRED_MESSAGE}), 400
    elif stage in DRIVERS:
        conn = db.get_conn(project_id)
        try:
            DRIVERS[stage].validate(conn, params)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        finally:
            conn.close()

    job_id = uuid.uuid4().hex[:10]
    conn = db.get_conn(project_id)
    with conn:
        # The job's number and name are the server's to assign -- cisTEM
        # numbers jobs rather than asking for a name, and the Results table
        # already shows the stage in its own column, so "Job 3" is the whole
        # of what a name has to carry. MAX+1 rather than COUNT+1 so a number
        # is never reused if a job is ever deleted.
        job_number = conn.execute(
            "SELECT COALESCE(MAX(JOB_NUMBER), 0) + 1 FROM JOBS"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO JOBS(JOB_ID, STAGE, JOB_NUMBER, NAME, PARAMS_JSON, STATUS, PROGRESS, "
            "CREATED_AT, MOVIE_GROUP_ID) VALUES (?, ?, ?, ?, ?, 'queued', 0, ?, ?)",
            (job_id, stage, job_number, "Job {}".format(job_number), json.dumps(params),
             now_iso(), movie_group_id),
        )
    conn.close()

    adapter = stages.ADAPTERS.get(stage)
    if adapter is not None and _job_runner is not None and _controller_available():
        return _submit_to_runner(project_id, job_id, adapter, params)
    if stage in DRIVERS and _job_runner is not None and _controller_available():
        return _start_driver(DRIVERS[stage], project_id, job_id, params)
    if adapter is not None or stage in DRIVERS:
        # This stage *can* run for real; say exactly what is stopping it,
        # rather than the generic simulation note about a stage binary.
        if _job_runner is None:
            why = "the job runner is not listening (see the server's startup output)"
        else:
            why = "'{}' was not found on the server's PATH -- build it from the cisTEM tree " \
                  "(src/programs/cistem_job_controller) and install it next to unblur, or set " \
                  "CISTEM_JOB_CONTROLLER".format(CONTROLLER_COMMAND)
        append_log(project_id, job_id, "[{}] simulating: {}".format(now_iso(), why))

    with _live_lock:
        _live[job_id] = {"proc": None, "cancel_requested": False}

    thread = threading.Thread(target=run_job, args=(project_id, job_id), daemon=True)
    thread.start()

    return jsonify(_row_to_job(_fetch_job_row(project_id, job_id))), 201


def _submit_to_runner(project_id, job_id, adapter, params):
    """The real path: build the program's task list from the project's
    assets, persist what a restart would need, and hand the job to the
    runner, which launches the run profile's manager command."""
    conn = db.get_conn(project_id)
    try:
        sys_conn = db.get_system_conn()
        profile = db.load_run_profile_by_name(sys_conn, params.get("run_profile"))
        sys_conn.close()
        if profile is None:
            error = "unknown run profile {!r}".format(params.get("run_profile"))
        elif profile["total_jobs"] == 0:
            error = "run profile {!r} has no run commands, so it can't launch anything".format(profile["name"])
        else:
            error = None
        if error is None:
            try:
                tasks = adapter.build_tasks(conn, project_id, params)
            except ValueError as exc:
                error = str(exc)
        if error is not None:
            with conn:
                conn.execute("UPDATE JOBS SET STATUS='failed', ERROR=?, FINISHED_AT=? WHERE JOB_ID=?",
                             (error, now_iso(), job_id))
            return jsonify({"error": error}), 400

        row = conn.execute("SELECT * FROM JOBS WHERE JOB_ID=?", (job_id,)).fetchone()
        spec = job_runner.JobSpec(job_id, _package_job_info(project_id, row), adapter.PROGRAM, profile, tasks,
                                  profile["manager_command"], controller_log=_controller_log_path(project_id, job_id),
                                  forward_progress=getattr(adapter, "WANTS_TASK_PROGRESS", False))
        with conn:
            conn.execute("UPDATE JOBS SET JOB_TOKEN=?, TASKS_JSON=? WHERE JOB_ID=?",
                         (spec.token, json.dumps(tasks), job_id))
    finally:
        conn.close()

    append_log(project_id, job_id, "[{}] job created (stage: {}, {} task{}, profile: {})".format(
        now_iso(), row["STAGE"], len(tasks), "" if len(tasks) == 1 else "s", profile["name"]))
    _db_sink.register(job_id, project_id)
    _job_runner.submit(spec)
    return jsonify(_row_to_job(_fetch_job_row(project_id, job_id))), 201


def _submit_child_job(project_id, child_job_id, adapter, tasks, profile):
    """classification.Runtime.submit_child: the child's JOBS row exists;
    give it a token and its task list and hand it to the runner -- the tail
    of _submit_to_runner() without the validation, which the driver did."""
    conn = db.get_conn(project_id)
    try:
        row = conn.execute("SELECT * FROM JOBS WHERE JOB_ID=?", (child_job_id,)).fetchone()
        spec = job_runner.JobSpec(child_job_id, _package_job_info(project_id, row), adapter.PROGRAM, profile, tasks,
                                  profile["manager_command"], controller_log=_controller_log_path(project_id, child_job_id),
                                  forward_progress=getattr(adapter, "WANTS_TASK_PROGRESS", False))
        with conn:
            conn.execute("UPDATE JOBS SET JOB_TOKEN=?, TASKS_JSON=? WHERE JOB_ID=?",
                         (spec.token, json.dumps(tasks), child_job_id))
    finally:
        conn.close()
    append_log(project_id, child_job_id, "[{}] step of 2D classification {} created ({}, {} task{}, profile: {})".format(
        now_iso(), row["PARENT_JOB_ID"], adapter.PROGRAM["name"], len(tasks), "" if len(tasks) == 1 else "s", profile["name"]))
    _db_sink.register(child_job_id, project_id)
    _job_runner.submit(spec)


_driver_runtime = classification.Runtime(
    submit_child=_submit_child_job,
    cancel=lambda job_id: _job_runner is not None and _job_runner.cancel(job_id),
    append_log=lambda project_id, job_id, text, level="info": append_log(
        project_id, job_id, "[{}] {}{}".format(now_iso(), "ERROR: " if level == "error" else "", text)),
    update_job=_update_job,
)
classification.configure(_driver_runtime)
abinitio.configure(_driver_runtime)
refine3d.configure(_driver_runtime)
autorefine.configure(_driver_runtime)
refinectf.configure(_driver_runtime)
generate3d.configure(_driver_runtime)


def _start_driver(driver, project_id, job_id, params):
    """The multi-run path: no single task list to hand the runner -- the
    driver launches the first step and follows up from the sink's callbacks."""
    conn = db.get_conn(project_id)
    try:
        sys_conn = db.get_system_conn()
        profile = db.load_run_profile_by_name(sys_conn, params.get("run_profile"))
        sys_conn.close()
        error = None
        if profile is None:
            error = "unknown run profile {!r}".format(params.get("run_profile"))
        else:
            try:
                driver.start(conn, project_id, job_id, params, profile)
            except ValueError as exc:
                error = str(exc)
        if error is not None:
            with conn:
                conn.execute("UPDATE JOBS SET STATUS='failed', ERROR=?, FINISHED_AT=? WHERE JOB_ID=?",
                             (error, now_iso(), job_id))
            return jsonify({"error": error}), 400
    finally:
        conn.close()
    return jsonify(_row_to_job(_fetch_job_row(project_id, job_id))), 201


@app.route("/api/projects/<project_id>/jobs/<job_id>")
@auth.project_access_required
def get_job(project_id, job_id):
    row = _fetch_job_row(project_id, job_id)
    if row is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(_row_to_job(row))


@app.route("/api/projects/<project_id>/jobs/<job_id>/log")
@auth.project_access_required
def get_log(project_id, job_id):
    row = _fetch_job_row(project_id, job_id)
    if row is None:
        return jsonify({"error": "not found"}), 404
    conn = db.get_conn(project_id)
    lines = conn.execute(
        "SELECT LINE FROM JOB_LOG_LINES WHERE JOB_ID=? ORDER BY SEQ", (job_id,)
    ).fetchall()
    conn.close()
    return jsonify({"log": "\n".join(l["LINE"] for l in lines)})


def _latest_task(conn, job_id, task_index=None):
    if task_index is not None:
        return conn.execute("SELECT * FROM JOB_TASKS WHERE JOB_ID=? AND TASK_INDEX=?", (job_id, task_index)).fetchone()
    return conn.execute(
        "SELECT * FROM JOB_TASKS WHERE JOB_ID=? AND STATUS='ok' AND RESULT_JSON IS NOT NULL "
        "ORDER BY FINISHED_AT DESC, TASK_INDEX DESC LIMIT 1", (job_id,)).fetchone()


@app.route("/api/projects/<project_id>/jobs/<job_id>/latest-result")
@auth.project_access_required
def get_latest_result(project_id, job_id):
    """The most recently finished task's result, as the stage's adapter
    presents it -- what cisTEM's job panel draws while a job runs. The
    page polls this for the job selected on the Jobs tab and decides for
    itself how often to redraw; `task_index` lets it tell a new result
    from the one it is already showing. A job with no adapter (a simulated
    one) or no finished task yet returns `result: null` with the reason."""
    row = _fetch_job_row(project_id, job_id)
    if row is None:
        return jsonify({"error": "not found"}), 404
    adapter = stages.ADAPTERS.get(row["STAGE"])
    out = {"job_id": job_id, "status": row["STATUS"], "progress": row["PROGRESS"], "result": None}
    if row["STAGE"] in DRIVERS:
        driver = DRIVERS[row["STAGE"]]
        conn = db.get_conn(project_id)
        try:
            result = driver.live_result(conn, row)
            info = driver.progress_info(json.loads(row["STATE_JSON"]) if row["STATE_JSON"] else None)
        finally:
            conn.close()
        out["task_count"] = info.get("task_count")
        out["done_count"] = info.get("tasks_done")
        if result is None:
            out["reason"] = "The first result is still being computed." if row["STATUS"] in ("queued", "running") \
                else "This job recorded no results."
        else:
            out["result"] = result
        return jsonify(out)
    if adapter is None or not hasattr(adapter, "live_result"):
        out["reason"] = "This stage has no live results (it runs in simulation)."
        return jsonify(out)
    sent_tasks = json.loads(row["TASKS_JSON"]) if row["TASKS_JSON"] else []
    conn = db.get_conn(project_id)
    try:
        out["task_count"] = len(sent_tasks)
        out["done_count"] = conn.execute(
            "SELECT COUNT(*) FROM JOB_TASKS WHERE JOB_ID=? AND STATUS IN ('ok','failed')", (job_id,)).fetchone()[0]
        task_row = _latest_task(conn, job_id)
        task = next((t for t in sent_tasks if t["index"] == task_row["TASK_INDEX"]), None) if task_row else None
        if task is None:
            out["reason"] = "Nothing has finished yet." if row["STATUS"] in ("queued", "running") else "This job recorded no results."
            return jsonify(out)
        result = adapter.live_result(conn, task, task_row)
        if result is None:
            out["reason"] = "The latest finished task returned no usable result."
            return jsonify(out)
        result["task_index"] = task_row["TASK_INDEX"]
        result["finished_at"] = task_row["FINISHED_AT"]
        out["result"] = result
        return jsonify(out)
    finally:
        conn.close()


@app.route("/api/projects/<project_id>/jobs/<job_id>/tasks/<int:task_index>/<which>.png")
@auth.project_access_required
def task_result_preview(project_id, job_id, task_index, which):
    """PNG of one task's output picture -- `which` is a key of the adapter's
    live_result_files() (unblur: sum, spectrum; ctffind: diagnostic) --
    from the file names the task was sent with, so it is available as soon
    as the worker has written it, before the job finishes and the results
    routes know about it."""
    row = _fetch_job_row(project_id, job_id)
    if row is None:
        return jsonify({"error": "not found"}), 404
    adapter = stages.ADAPTERS.get(row["STAGE"])
    sent_tasks = json.loads(row["TASKS_JSON"]) if row["TASKS_JSON"] else []
    task = next((t for t in sent_tasks if t["index"] == task_index), None)
    if adapter is None or not hasattr(adapter, "live_result_files") or task is None:
        return jsonify({"error": "no such task"}), 404
    files = adapter.live_result_files(task)
    if which not in files:
        return jsonify({"error": "no such picture: {}".format(which)}), 404
    return _file_preview_response(files[which], "task-{}-{}-{}".format(which, job_id, task_index), which.replace("_", " "))


@app.route("/api/projects/<project_id>/jobs/<job_id>/actions/<name>", methods=["POST"])
@auth.project_access_required
def job_action(project_id, job_id, name):
    """One of the job's `actions` (see `Job`): ab-initio's Take Current /
    Take Last Start (stop and keep a reconstruction as the run's result),
    Auto Refine's and Refine 3D's Finish After This Round."""
    row = _fetch_job_row(project_id, job_id)
    if row is None:
        return jsonify({"error": "not found"}), 404
    driver = DRIVERS.get(row["STAGE"])
    if driver is None or not hasattr(driver, "perform_action"):
        return jsonify({"error": "this job has no actions"}), 400
    if row["STATUS"] not in ("queued", "running"):
        return jsonify({"error": "the job is {}".format(row["STATUS"])}), 409
    conn = db.get_conn(project_id)
    try:
        try:
            driver.perform_action(conn, project_id, job_id, name)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
    finally:
        conn.close()
    return jsonify(_row_to_job(_fetch_job_row(project_id, job_id)))


@app.route("/api/projects/<project_id>/jobs/<job_id>/cancel", methods=["POST"])
@auth.project_access_required
def cancel_job(project_id, job_id):
    row = _fetch_job_row(project_id, job_id)
    if row is None:
        return jsonify({"error": "not found"}), 404
    job = _row_to_job(row)
    if job["status"] not in ("queued", "running"):
        return jsonify(job)

    if row["STAGE"] in DRIVERS and row["STATE_JSON"]:
        conn = db.get_conn(project_id)
        try:
            DRIVERS[row["STAGE"]].cancel(conn, project_id, job_id)
        finally:
            conn.close()
        return jsonify(_row_to_job(_fetch_job_row(project_id, job_id)))

    _update_job(project_id, job_id, CANCEL_REQUESTED=1)
    if _job_runner is not None and _job_runner.cancel(job_id):
        append_log(project_id, job_id, "[{}] cancel requested; asking the controller to stop".format(now_iso()))
        return jsonify(_row_to_job(_fetch_job_row(project_id, job_id)))
    with _live_lock:
        info = _live.setdefault(job_id, {"proc": None})
        info["cancel_requested"] = True
        proc = info.get("proc")
    if proc is not None:
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001
            pass
    return jsonify(_row_to_job(_fetch_job_row(project_id, job_id)))


if __name__ == "__main__":
    auth.bootstrap_admin_if_needed()
    start_job_runner()
    _recover_interrupted_jobs()
    app.run(host="0.0.0.0", port=8000, threaded=True)
