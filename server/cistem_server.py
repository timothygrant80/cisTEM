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
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, g, jsonify, request, send_from_directory
from flask_cors import CORS

import auth
import db
import imageheaders
import preview

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
# Static frontend -- optional convenience so `python cistem_server.py` alone
# is enough to try the app: open http://localhost:8000/ instead of finding
# and double-clicking cistem3.html yourself. cistem3.html still works fine
# opened directly (file://) or served by anything else; this just adds one
# more way to reach it. Only this explicit allowlist of sibling files is
# served, never an arbitrary path under REPO_ROOT -- that would expose
# server/data/auth.db and the rest of the repo.
# ---------------------------------------------------------------------------

STATIC_FILES = {"cistem3.html", "config.js", "logo.png"}


@app.route("/")
def index():
    return send_from_directory(REPO_ROOT, "cistem3.html")


@app.route("/<path:filename>")
def static_file(filename):
    if filename not in STATIC_FILES:
        abort(404)
    return send_from_directory(REPO_ROOT, filename)


# ---------------------------------------------------------------------------
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
BROWSABLE_EXTENSIONS = {"movie": MOVIE_EXTENSIONS, "image": {".mrc", ".mrcs", ".tif", ".tiff"}}


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

# Group 0 is the master list db.py seeds into every project -- every asset is
# a member of it, and the app treats it as the source of truth for what
# exists. Renaming, deleting or inverting it would leave the project without
# one, so the routes below refuse it for both kinds.
ALL_GROUP_ID = 0

# Images are 2D micrographs, so unlike movies they are never EER: an EER file
# is a raw movie container by construction. Cf. imageheaders.read_image_header().
IMAGE_IMPORT_EXTENSIONS = {".mrc", ".mrcs", ".tif", ".tiff"}


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
    rows = conn.execute(
        "SELECT g.GROUP_ID as group_id, g.GROUP_NAME as group_name, "
        "COUNT(m.{id}) as {count} "
        "FROM {g} g LEFT JOIN {m} m ON m.GROUP_ID = g.GROUP_ID "
        "GROUP BY g.GROUP_ID ORDER BY g.GROUP_ID".format(
            id=kind.id_column, count=kind.count_key, g=kind.group_table, m=kind.member_table
        )
    ).fetchall()
    conn.close()
    return jsonify({kind.groups_key: [dict(r) for r in rows]})


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
    etag = '"{}-{}-{}-{}"'.format(kind.noun, asset_id, int(stat.st_mtime), stat.st_size)
    if request.headers.get("If-None-Match") == etag:
        return "", 304

    try:
        png, meta = render(path)
    except preview.PreviewError as exc:
        return jsonify({"error": str(exc)}), 422

    response = app.response_class(png, mimetype="image/png")
    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = "private, max-age=3600"
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
# Job routes (project-scoped)
# ---------------------------------------------------------------------------
# Run profiles (project-scoped)
# ---------------------------------------------------------------------------

@app.route("/api/projects/<project_id>/run-profiles", methods=["GET"])
@auth.project_access_required
def list_run_profiles(project_id):
    """The project's RUN_PROFILES rows, for the Run Profile picker each job
    panel carries (cf. RunProfileComboBox in cisTEM's AlignMoviesPanel, filled
    from run_profiles_panel). db.py seeds the same three profiles into every
    project; nothing edits them yet, so this is read-only for now.
    """
    conn = db.get_conn(project_id)
    rows = conn.execute(
        "SELECT RUN_PROFILE_ID as run_profile_id, PROFILE_NAME as profile_name, "
        "MANAGER_RUN_COMMAND as manager_run_command "
        "FROM RUN_PROFILES ORDER BY RUN_PROFILE_ID"
    ).fetchall()
    conn.close()
    return jsonify({"run_profiles": [dict(r) for r in rows]})


# ---------------------------------------------------------------------------

@app.route("/api/projects/<project_id>/jobs", methods=["GET"])
@auth.project_access_required
def list_jobs(project_id):
    conn = db.get_conn(project_id)
    rows = conn.execute("SELECT * FROM JOBS ORDER BY CREATED_AT").fetchall()
    conn.close()
    return jsonify({"jobs": [_row_to_job(r) for r in rows]})


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


@app.route("/api/projects/<project_id>/jobs/<job_id>/cancel", methods=["POST"])
@auth.project_access_required
def cancel_job(project_id, job_id):
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
    auth.bootstrap_admin_if_needed()
    _recover_interrupted_jobs()
    app.run(host="0.0.0.0", port=8000, threaded=True)
