"""
Reference job-queue API for the Cryo-EM Job Runner page (job_runner.html).

This is a STARTING POINT, not a production job scheduler. It implements the
small HTTP contract the front-end expects (see README.md) using an in-memory
queue and a background thread per job. Wire in your real pipeline by editing
STAGE_COMMANDS below -- everything else (routing, status tracking, logs,
cancellation) already works.

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
add an auth check in `require_auth()` before exposing it any wider.
"""

import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # wide open by design -- see Security note above

# ---------------------------------------------------------------------------
# Job store (in-memory; swap for a real DB / your scheduler's own state if
# you outgrow this)
# ---------------------------------------------------------------------------

jobs = {}
jobs_lock = threading.Lock()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# TODO: point these at your real pipeline. Each entry is a template command
# (list of args, Python str.format placeholders filled from the job's
# `params`) run via subprocess. If the named binary isn't on PATH, the job
# runs in SIMULATION mode instead (sleeps, logs progress, fabricates
# plausible output) so you can exercise the page end-to-end before your
# pipeline is wired up.
#
# Example once you're ready:
#   "motion_correction": {
#       "binary": "MotionCor2",
#       "command": [
#           "MotionCor2", "-InTiff", "{input_glob}",
#           "-Gain", "{gain_ref}", "-PixSize", "{pixel_size_a}",
#           "-FmDose", "{dose_per_frame}", "-OutMrc", "{output_dir}/",
#       ],
#   },
# ---------------------------------------------------------------------------

STAGE_COMMANDS = {
    "motion_correction": {"binary": "MotionCor2", "command": None},
    "ctf_estimation": {"binary": "ctffind", "command": None},
    "particle_picking": {"binary": "relion_autopick", "command": None},
    "class2d": {"binary": "relion_refine", "command": None},
    "refine3d": {"binary": "relion_refine", "command": None},
}


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------

def append_log(job, line):
    with jobs_lock:
        job["log"].append(line)


def run_job(job_id):
    with jobs_lock:
        job = jobs[job_id]
        job["status"] = "running"
        job["started_at"] = now_iso()
    append_log(job, "[{}] job started (stage: {})".format(now_iso(), job["stage"]))

    stage_cfg = STAGE_COMMANDS.get(job["stage"], {})
    binary = stage_cfg.get("binary")
    command_template = stage_cfg.get("command")
    have_binary = bool(binary) and shutil.which(binary) is not None

    try:
        if have_binary and command_template:
            _run_real(job, command_template)
        else:
            _run_simulated(job, binary)
    except JobCancelled:
        with jobs_lock:
            job["status"] = "cancelled"
            job["finished_at"] = now_iso()
        append_log(job, "[{}] job cancelled".format(now_iso()))
        return
    except Exception as exc:  # noqa: BLE001
        with jobs_lock:
            job["status"] = "failed"
            job["error"] = str(exc)
            job["finished_at"] = now_iso()
        append_log(job, "[{}] job failed: {}".format(now_iso(), exc))
        return

    with jobs_lock:
        job["status"] = "completed"
        job["progress"] = 100
        job["finished_at"] = now_iso()
    append_log(job, "[{}] job completed".format(now_iso()))


class JobCancelled(Exception):
    pass


def _check_cancelled(job_id):
    with jobs_lock:
        if jobs[job_id]["_cancel_requested"]:
            raise JobCancelled()


def _run_real(job, command_template):
    """Fill the command template from job params and run it, streaming
    stdout/stderr into the job's log."""
    params = dict(job["params"] or {})
    try:
        cmd = [str(part).format(**params) for part in command_template]
    except KeyError as exc:
        raise RuntimeError("missing parameter for command template: {}".format(exc))

    append_log(job, "$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    with jobs_lock:
        job["_proc"] = proc

    for line in proc.stdout:
        append_log(job, line.rstrip("\n"))
        _check_cancelled(job["id"])

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("exited with code {}".format(proc.returncode))


def _run_simulated(job, binary):
    """No real binary found -- walk the job through a plausible progress
    sequence so the front end has something to show."""
    note = (
        "'{}' not found on PATH -- simulating this job. ".format(binary)
        if binary
        else "no command configured for this stage -- simulating. "
    )
    append_log(job, note + "Edit STAGE_COMMANDS in app.py to run the real thing.")
    steps = [10, 25, 45, 65, 85, 100]
    for pct in steps:
        _check_cancelled(job["id"])
        time.sleep(0.8)
        with jobs_lock:
            job["progress"] = pct
        append_log(job, "[{}] progress: {}%".format(now_iso(), pct))
    with jobs_lock:
        job["metrics"] = _fake_metrics(job["stage"])


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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "time": now_iso()})


@app.route("/api/jobs", methods=["GET"])
def list_jobs():
    with jobs_lock:
        out = [_public_job(j) for j in jobs.values()]
    return jsonify({"jobs": out})


@app.route("/api/jobs", methods=["POST"])
def create_job():
    body = request.get_json(force=True, silent=True) or {}
    stage = body.get("stage")
    if stage not in STAGE_COMMANDS:
        return jsonify({"error": "unknown stage '{}'".format(stage)}), 400

    job_id = uuid.uuid4().hex[:10]
    job = {
        "id": job_id,
        "stage": stage,
        "name": body.get("name") or job_id,
        "params": body.get("params") or {},
        "status": "queued",
        "progress": 0,
        "created_at": now_iso(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "metrics": {},
        "log": [],
        "_cancel_requested": False,
        "_proc": None,
    }
    with jobs_lock:
        jobs[job_id] = job

    thread = threading.Thread(target=run_job, args=(job_id,), daemon=True)
    thread.start()

    return jsonify(_public_job(job)), 201


@app.route("/api/jobs/<job_id>")
def get_job(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "not found"}), 404
        return jsonify(_public_job(job))


@app.route("/api/jobs/<job_id>/log")
def get_log(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "not found"}), 404
        return jsonify({"log": "\n".join(job["log"])})


@app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "not found"}), 404
        if job["status"] not in ("queued", "running"):
            return jsonify(_public_job(job))
        job["_cancel_requested"] = True
        proc = job.get("_proc")
    if proc is not None:
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001
            pass
    return jsonify(_public_job(job))


def _public_job(job):
    return {k: v for k, v in job.items() if not k.startswith("_") and k != "log"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
