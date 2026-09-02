"""
Reference job-queue API for the Cryo-EM Job Runner page (job_runner.html).

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
    python app.py
    # serves on http://localhost:8000, API under /api

Then open job_runner.html in a browser and point "API base URL" at
http://localhost:8000/api

Security
--------
No authentication. CORS is wide open (Access-Control-Allow-Origin: *) so the
page can be opened as a local file. Only run this on a trusted network --
add an auth check before exposing it any wider.
"""

import glob as glob_module
import json
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone

from flask import Flask, jsonify, request
from flask_cors import CORS

import db

app = Flask(__name__)
CORS(app)  # wide open by design -- see Security note above

SYNTHETIC_MOVIE_COUNT = 12

# Live-only state that can't be persisted: a running job's subprocess handle
# and a fast in-process cancel flag. Durable status/progress/log lives in
# each project's SQLite file (see db.py) so it survives across requests and
# is visible from any thread; this dict only exists to let cancel() reach
# the actual OS process and to avoid a DB round trip on every progress check.
_live_lock = threading.Lock()
_live = {}  # job_id -> {"proc": Popen|None, "cancel_requested": bool}


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
    "refine3d": {"binary": "relion_refine", "command": None},
}


# ---------------------------------------------------------------------------
# Job store helpers (SQLite-backed, one project's JOBS/JOB_LOG_LINES tables)
# ---------------------------------------------------------------------------

def _row_to_job(row):
    return {
        "id": row["JOB_ID"],
        "stage": row["STAGE"],
        "name": row["NAME"],
        "params": json.loads(row["PARAMS_JSON"]) if row["PARAMS_JSON"] else {},
        "status": row["STATUS"],
        "progress": row["PROGRESS"],
        "created_at": row["CREATED_AT"],
        "started_at": row["STARTED_AT"],
        "finished_at": row["FINISHED_AT"],
        "error": row["ERROR"],
        "metrics": json.loads(row["METRICS_JSON"]) if row["METRICS_JSON"] else {},
    }


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


def append_log(project_id, job_id, line):
    conn = db.get_conn(project_id)
    with conn:
        seq = conn.execute(
            "SELECT COALESCE(MAX(SEQ), -1) + 1 FROM JOB_LOG_LINES WHERE JOB_ID=?", (job_id,)
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO JOB_LOG_LINES(JOB_ID, SEQ, LINE) VALUES (?, ?, ?)", (job_id, seq, line)
        )
    conn.close()


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
    append_log(project_id, job["id"], note + "Edit STAGE_COMMANDS in app.py to run the real thing.")
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
    output_dir = (params.get("output_dir") or "").rstrip("/")

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
            output_file = (
                "{}/{}_aligned.mrc".format(output_dir, movie["NAME"]) if output_dir else ""
            )

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

            conn.execute(
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

        conn.execute(
            "UPDATE MASTER_SETTINGS SET TOTAL_JOBS_RUN = COALESCE(TOTAL_JOBS_RUN, 0) + 1 WHERE NUMBER=1"
        )
    conn.close()


def _recover_interrupted_jobs():
    """Any job still 'queued'/'running' when the server last stopped has no
    thread left to finish it -- mark it failed instead of leaving it stuck
    forever, since (unlike cisTEM's live socket-managed jobs) our execution
    threads don't survive a restart."""
    if not db.PROJECTS_ROOT.is_dir():
        return
    for entry in db.PROJECTS_ROOT.iterdir():
        if not (entry / "project.db").is_file():
            continue
        conn = db.get_conn(entry.name)
        with conn:
            conn.execute(
                "UPDATE JOBS SET STATUS='failed', "
                "ERROR='Server restarted while this job was in progress', FINISHED_AT=? "
                "WHERE STATUS IN ('queued','running')",
                (now_iso(),),
            )
        conn.close()


# ---------------------------------------------------------------------------
# Project routes
# ---------------------------------------------------------------------------

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "time": now_iso()})


def _project_guard(project_id):
    if not db.project_exists(project_id):
        return jsonify({"error": "project not found"}), 404
    return None


@app.route("/api/projects", methods=["GET"])
def list_projects_route():
    return jsonify({"projects": db.list_projects()})


@app.route("/api/projects", methods=["POST"])
def create_project_route():
    body = request.get_json(force=True, silent=True) or {}
    try:
        project_id = db.create_project(body.get("name"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(db.get_project_summary(project_id)), 201


@app.route("/api/projects/<project_id>")
def get_project_route(project_id):
    summary = db.get_project_summary(project_id)
    if summary is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(summary)


@app.route("/api/projects/<project_id>", methods=["DELETE"])
def delete_project_route(project_id):
    if not db.project_exists(project_id):
        return jsonify({"error": "not found"}), 404
    db.delete_project(project_id)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Movie import routes (project-scoped)
# ---------------------------------------------------------------------------

@app.route("/api/projects/<project_id>/movies", methods=["GET"])
def list_movies(project_id):
    err = _project_guard(project_id)
    if err:
        return err
    conn = db.get_conn(project_id)
    rows = conn.execute("SELECT * FROM MOVIE_ASSETS ORDER BY MOVIE_ASSET_ID").fetchall()
    conn.close()
    return jsonify({"movies": [dict(r) for r in rows]})


@app.route("/api/projects/<project_id>/movie-groups", methods=["GET"])
def list_movie_groups(project_id):
    err = _project_guard(project_id)
    if err:
        return err
    conn = db.get_conn(project_id)
    rows = conn.execute(
        "SELECT g.GROUP_ID as group_id, g.GROUP_NAME as group_name, "
        "COUNT(m.MOVIE_ASSET_ID) as movie_count "
        "FROM MOVIE_GROUP_LIST g LEFT JOIN MOVIE_GROUP_MEMBERS m ON m.GROUP_ID = g.GROUP_ID "
        "GROUP BY g.GROUP_ID ORDER BY g.GROUP_ID"
    ).fetchall()
    conn.close()
    return jsonify({"movie_groups": [dict(r) for r in rows]})


@app.route("/api/projects/<project_id>/movies/import-defaults", methods=["GET"])
def get_import_defaults(project_id):
    err = _project_guard(project_id)
    if err:
        return err
    conn = db.get_conn(project_id)
    row = conn.execute("SELECT * FROM MOVIE_IMPORT_DEFAULTS WHERE NUMBER=1").fetchone()
    conn.close()
    return jsonify(dict(row) if row else {})


@app.route("/api/projects/<project_id>/movies/import", methods=["POST"])
def import_movies(project_id):
    err = _project_guard(project_id)
    if err:
        return err
    body = request.get_json(force=True, silent=True) or {}

    input_glob = (body.get("input_glob") or "").strip()
    group_name = (body.get("group_name") or "").strip() or "Import {}".format(now_iso()[:19])
    voltage_kv = body.get("voltage_kv")
    cs_mm = body.get("cs_mm")
    pixel_size_a = body.get("pixel_size_a")
    dose_per_frame = body.get("dose_per_frame")
    gain_ref = body.get("gain_ref") or None
    dark_ref = body.get("dark_ref") or None

    matched = sorted(glob_module.glob(input_glob)) if input_glob else []
    synthetic = False
    if not matched:
        # No real files found (expected on a demo machine with no data on
        # hand) -- fabricate plausible movies, same spirit as _run_simulated().
        synthetic = True
        matched = ["sim_movie_{:04d}.tif".format(i + 1) for i in range(SYNTHETIC_MOVIE_COUNT)]

    conn = db.get_conn(project_id)
    with conn:
        cur = conn.execute(
            "INSERT INTO MOVIE_GROUP_LIST(GROUP_NAME, LIST_ID) VALUES (?, 0)", (group_name,)
        )
        group_id = cur.lastrowid

        movie_ids = []
        for path in matched:
            name = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            x_size = 4096 if synthetic else None
            y_size = 4096 if synthetic else None
            n_frames = 40 if synthetic else None
            cur = conn.execute(
                "INSERT INTO MOVIE_ASSETS("
                "NAME, FILENAME, POSITION_IN_STACK, X_SIZE, Y_SIZE, NUMBER_OF_FRAMES, "
                "VOLTAGE, PIXEL_SIZE, DOSE_PER_FRAME, SPHERICAL_ABERRATION, GAIN_FILENAME, "
                "DARK_FILENAME, OUTPUT_BINNING_FACTOR, PROTEIN_IS_WHITE) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    name, path, 1, x_size, y_size, n_frames,
                    voltage_kv, pixel_size_a, dose_per_frame, cs_mm, gain_ref, dark_ref,
                    1.0, 0,
                ),
            )
            movie_id = cur.lastrowid
            movie_ids.append(movie_id)
            conn.execute(
                "INSERT INTO MOVIE_GROUP_MEMBERS(GROUP_ID, MOVIE_ASSET_ID) VALUES (?, ?)",
                (group_id, movie_id),
            )
            conn.execute(
                "INSERT INTO MOVIE_GROUP_MEMBERS(GROUP_ID, MOVIE_ASSET_ID) VALUES (0, ?)",
                (movie_id,),
            )

        conn.execute(
            "UPDATE MOVIE_IMPORT_DEFAULTS SET VOLTAGE=?, SPHERICAL_ABERRATION=?, PIXEL_SIZE=?, "
            "EXPOSURE_PER_FRAME=?, GAIN_REFERENCE_FILENAME=?, DARK_REFERENCE_FILENAME=? WHERE NUMBER=1",
            (voltage_kv, cs_mm, pixel_size_a, dose_per_frame, gain_ref, dark_ref),
        )
    conn.close()

    return jsonify({
        "movie_group_id": group_id,
        "group_name": group_name,
        "synthetic": synthetic,
        "movie_count": len(movie_ids),
    }), 201


# ---------------------------------------------------------------------------
# Job routes (project-scoped)
# ---------------------------------------------------------------------------

@app.route("/api/projects/<project_id>/jobs", methods=["GET"])
def list_jobs(project_id):
    err = _project_guard(project_id)
    if err:
        return err
    conn = db.get_conn(project_id)
    rows = conn.execute("SELECT * FROM JOBS ORDER BY CREATED_AT").fetchall()
    conn.close()
    return jsonify({"jobs": [_row_to_job(r) for r in rows]})


@app.route("/api/projects/<project_id>/jobs", methods=["POST"])
def create_job(project_id):
    err = _project_guard(project_id)
    if err:
        return err
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

    job_id = uuid.uuid4().hex[:10]
    conn = db.get_conn(project_id)
    with conn:
        conn.execute(
            "INSERT INTO JOBS(JOB_ID, STAGE, NAME, PARAMS_JSON, STATUS, PROGRESS, CREATED_AT, "
            "MOVIE_GROUP_ID) VALUES (?, ?, ?, ?, 'queued', 0, ?, ?)",
            (job_id, stage, body.get("name") or job_id, json.dumps(params), now_iso(), movie_group_id),
        )
    conn.close()

    with _live_lock:
        _live[job_id] = {"proc": None, "cancel_requested": False}

    thread = threading.Thread(target=run_job, args=(project_id, job_id), daemon=True)
    thread.start()

    return jsonify(_row_to_job(_fetch_job_row(project_id, job_id))), 201


@app.route("/api/projects/<project_id>/jobs/<job_id>")
def get_job(project_id, job_id):
    err = _project_guard(project_id)
    if err:
        return err
    row = _fetch_job_row(project_id, job_id)
    if row is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(_row_to_job(row))


@app.route("/api/projects/<project_id>/jobs/<job_id>/log")
def get_log(project_id, job_id):
    err = _project_guard(project_id)
    if err:
        return err
    row = _fetch_job_row(project_id, job_id)
    if row is None:
        return jsonify({"error": "not found"}), 404
    conn = db.get_conn(project_id)
    lines = conn.execute(
        "SELECT LINE FROM JOB_LOG_LINES WHERE JOB_ID=? ORDER BY SEQ", (job_id,)
    ).fetchall()
    conn.close()
    return jsonify({"log": "\n".join(l["LINE"] for l in lines)})


@app.route("/api/projects/<project_id>/jobs/<job_id>/cancel", methods=["POST"])
def cancel_job(project_id, job_id):
    err = _project_guard(project_id)
    if err:
        return err
    row = _fetch_job_row(project_id, job_id)
    if row is None:
        return jsonify({"error": "not found"}), 404
    job = _row_to_job(row)
    if job["status"] not in ("queued", "running"):
        return jsonify(job)

    _update_job(project_id, job_id, CANCEL_REQUESTED=1)
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
    _recover_interrupted_jobs()
    app.run(host="0.0.0.0", port=8000, threaded=True)
