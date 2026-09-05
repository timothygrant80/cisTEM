# cisTEM3

A local, project-based job-submission UI for a single-particle cryo-EM processing pipeline (Align Movies → Find CTF → Find Particles → 2D Classification → Refine 3D), styled after cisTEM's desktop interface.

Two pieces:

- **`cistem3.html`** — a standalone page you open in your own browser. On load it auto-connects to the API address in `config.js` (`http://localhost:8000/api` by default) — success shows a login screen, failure shows an error screen with a Retry button. Once logged in you create or open a **project**; once one's open it submits jobs and polls status against that same API, scoped to that project and to you. It makes no network calls anywhere else.
- **`config.js`** — the one file to edit if your API isn't at `localhost:8000`. A single `window.CRYOEM_CONFIG = { apiBase: "..." }`; reload the page after changing it.
- **`server/cistem_server.py`** + **`server/db.py`** + **`server/auth.py`** — a small reference Flask API implementing the contract the page expects, backed by one SQLite file per project (`server/data/projects/<id>/project.db`) plus a global `server/data/auth.db` for user accounts and sessions. Ships in "simulation mode" (fake progress + fake numbers) so you can try the whole flow before your real pipeline is wired in.

`reference/dashboard.html` is an earlier static mockup with richer diagnostic charts (throughput, defocus histogram, FSC curve) — a design reference, not wired to the job runner.

## Why a local file instead of a hosted link

The page needs to call an API on your own network (cryoSPARC, a Slurm cluster, a lab script) — a page hosted on claude.ai is sandboxed and can't reach arbitrary private servers. Running `cistem3.html` locally sidesteps that: it's an ordinary page, so it can call whatever your browser can reach.

## Quick start (try it with the reference server)

```bash
cd server
pip install -r requirements.txt
python cistem_server.py
```

This serves the API at `http://localhost:8000/api`, and also serves the page itself — open `http://localhost:8000/` in a browser and you're there, no separate step needed. On first run, since there are no users yet, it auto-creates an admin account and prints its password to the console (also written once to `server/data/admin_credentials.txt`) — copy that password before it scrolls away.

(You can still open `cistem3.html` directly instead — double-click it, or `open cistem3.html` — it connects to `http://localhost:8000/api` automatically either way; if your API is somewhere else, edit `config.js` first (see above) and reload.)

1. **Log in** as `admin` with the password from the console/credentials file.
2. **Manage Users** (admin-only panel on the home screen) — create a real account for yourself (and anyone else) with a role of `user` or `admin`. There's no self-registration; only an admin can create accounts.
3. **Create a project** — give it a name and hit Create. You're taken into the app, scoped to that project — and to you: other non-admin users won't see it.
4. **Assets tab → Import Movies** — pick a path/glob for movie files (there's a **Browse…** button that lists the API server's own filesystem) plus microscope metadata (voltage, Cs, pixel size, dose/frame). Import stays disabled until everything required is filled in and every path actually exists, so you'll need some real movie files on the machine running the server.
5. **Actions tab → Align Movies** — pick the movie group you just imported (metadata comes from the import, not retyped here), set an output directory, and Run. Since no real binaries are configured yet, it runs in simulation mode: progresses through a fake sequence and lands on `completed` with plausible placeholder numbers.
6. **Results tab** — watch the queue, open a job's log, cancel a running one.
7. **Close Project** (top right) returns you to the home screen — project data persists (it's a real SQLite file), so reopening it later shows the same movies and job history, even after restarting `cistem_server.py`. Your login persists too (a token in `localStorage`), so reloading the page skips straight back to the project picker.

If your browser blocks `fetch` from a `file://` page, open `http://localhost:8000/` instead (the reference server already serves the page — see above), or serve the folder yourself with `python -m http.server 8080` from this directory and open `http://localhost:8080/cistem3.html`.

## Wiring in your real pipeline

Open `server/cistem_server.py` and fill in `STAGE_COMMANDS` — one command template per stage, using `{param_key}` placeholders filled from the job's submitted parameters. For stages still using freeform typed params (`ctf_estimation`, `particle_picking`, `class2d`, `refine3d`) this is a direct 1:1 mapping, e.g.:

```python
"ctf_estimation": {
    "binary": "ctffind",
    "command": [
        "ctffind", "--in", "{input_glob}",
        "--voltage", "{voltage_kv}", "--cs", "{cs_mm}",
        "--box-size", "{box_size}",
    ],
},
```

`motion_correction` (Align Movies) is different: its job `params` carry a `movie_group_id`, not a file glob, because the movie files and their metadata now live in that project's `MOVIE_ASSETS`/`MOVIE_GROUP_MEMBERS` tables (set once at import time — see "Projects" below). `_run_real()`'s current template-fills-one-command model doesn't map cleanly onto "one job, N movies" batch execution; wiring a real `MotionCor2` (or similar) integration for this stage means resolving `params["movie_group_id"]` to its member movies (`server/db.py`'s `get_conn()` + a `MOVIE_GROUP_MEMBERS` join, same query `_write_motion_correction_results()` already uses) and looping a real command per movie, rather than a single template fill. This is real, but not yet built — flagged here rather than glossed over.

Any stage left as `"command": None`, or whose `binary` isn't found on `PATH`, keeps running in simulation mode — so you can wire stages up one at a time.

If you're driving **cryoSPARC** or **Slurm** instead of calling binaries directly, replace the body of `_run_real()` in `cistem_server.py` with calls to cryoSPARC's JSON API or `slurmrestd`, keeping the same job bookkeeping (status, progress, log, cancel) around it. The front end doesn't need to change either way — it only knows the HTTP contract below.

## Projects

Each project is a self-contained SQLite file (schema in `server/db.py`), modeled on a real cisTEM project database's table/column names wherever this app stores the same kind of data — `MASTER_SETTINGS`, `MOVIE_ASSETS`, `MOVIE_ALIGNMENT_LIST`, `RUN_PROFILES`, and so on. This means:

- Microscope/movie metadata is entered once at import time and referenced by jobs afterward, instead of retyped into every job form.
- A project survives restarting `cistem_server.py` — it's a file on disk, not an in-memory store.
- Switching projects (Close Project → open a different one) is just pointing subsequent requests at a different file; nothing bleeds between projects.

Only `motion_correction` is wired to real project data end-to-end right now. The other four stages are project-scoped (their job history lands in the right project) but still use typed/freeform params and don't read from or write to their own result tables yet (`ESTIMATED_CTF_PARAMETERS`, `PARTICLE_PICKING_LIST`, `CLASSIFICATION_LIST`, `REFINEMENT_LIST` — schema's there, just unused). See `CLAUDE.md` for the reasoning and what "wiring one up" would involve.

## Users and access control

Two roles: `user` (sees and manages only the projects they created) and `admin` (sees and can open/delete *every* project, and is the only role that can create new accounts). There's no public self-registration — an admin creates every account, from the home screen's Manage Users panel or `POST /users`.

Auth is a bearer token (`Authorization: Bearer <token>`), issued by `POST /auth/login` and stored client-side in `localStorage`, not a cookie — this keeps it compatible with the API's wide-open CORS (`Access-Control-Allow-Origin: *`, which can't be combined with cookie credentials) and means it works the same whether `cistem3.html` is opened as a file or served. Tokens are tracked server-side (`server/auth.py`'s `SESSIONS` table) so logout genuinely revokes them, unlike a stateless JWT. See `server/auth.py`'s module docstring for more (session expiry, the login-timing-attack guard, the bootstrap-admin process).

## The HTTP contract

`cistem3.html` only ever calls these, against whatever base URL you give it. Everything except `/health`, `/auth/login`, and `/auth/logout` requires a valid `Authorization: Bearer <token>` header (a `401` otherwise); everything under `/projects/:id/...` additionally requires that you own that project or are an admin (a `403` otherwise, `404` if the project doesn't exist at all):

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Connectivity check. Any 2xx JSON response counts. No auth. |
| `POST` | `/auth/login` | Body: `{username, password}` → `{token, user}`. No auth. |
| `POST` | `/auth/logout` | Revokes the presented token if any. Always `200`. |
| `GET` | `/auth/me` | `{user}` for the current token — used to silently re-validate a stored token on page load |
| `GET` | `/users` | Admin only. List users: `{ "users": [{id, username, role, display_name, created_at}, ...] }` — no password hashes |
| `POST` | `/users` | Admin only. Create a user. Body: `{username, password, role, display_name}` (`password` min. 8 characters, `role` is `user` or `admin`) |
| `GET` | `/projects` | List projects: `{ "projects": [{id, name, total_jobs_run, owner_username, ...}, ...] }` — your own, or all of them if you're an admin |
| `POST` | `/projects` | Create a project you own. Body: `{ "name" }` → returns its summary |
| `GET` | `/projects/:id` | One project's summary |
| `DELETE` | `/projects/:id` | Delete a project permanently (removes its `.db` file) |
| `GET` | `/projects/:id/movies` | List imported movies |
| `POST` | `/projects/:id/movie-groups/:gid/invert` | Invert a group against All Movies (self-reversing; `400` for All Movies) |
| `GET` | `/projects/:id/movies/:id/preview.png` | Summed-frame PNG preview (MRC and TIFF; EER has no preview, `415`) |
| `GET` | `/projects/:id/movie-groups` | List movie groups with member counts |
| `GET` | `/projects/:id/movies/import-defaults` | Last-used import form values |
| `POST` | `/projects/:id/movies/import` | Import movies into "All Movies". Body: `{input_glob, voltage_kv, cs_mm, pixel_size_a, dose_per_frame, protein_is_white, apply_gain, gain_ref, apply_dark, dark_ref, resample_movies, desired_pixel_size_a, eer_frames_per_image, eer_super_res_factor}`. `400`s unless the required fields are set, referenced paths exist, and the glob matches at least one file not already imported. Files already in the project are skipped. Dimensions and frame count are read from each file's header; `skip_full_check` skips the (expensive) TIFF/EER frame count. Returns `{movie_count, skipped_count, failed}`. |
| `POST` | `/projects/:id/movies/check-import` | Body: `{glob, files: [...]}` → `{glob_match_count, new_count, already_imported_count, glob_has_eer, files: {path: bool}}`, for the import dialog's live validation |
| `GET` | `/projects/:id/images` | List image assets (imported micrographs *and* Align Movies output) |
| `GET` | `/projects/:id/image-groups` | List image groups with member counts |
| `GET` | `/projects/:id/images/:id/preview.png` | PNG preview of the micrograph (MRC and TIFF). Nothing is summed — an image is one exposure |
| `GET` | `/projects/:id/images/import-defaults` | Last-used image import form values |
| `POST` | `/projects/:id/images/import` | Import micrographs into "All Images". Body: `{input_glob, voltage_kv, cs_mm, pixel_size_a, protein_is_white}` — no dose, gain/dark or EER fields, since an image is already averaged. `400`s unless voltage/Cs/pixel size are set and the glob matches at least one not-already-imported MRC/TIFF file (`.eer` never matches). Returns `{image_count, skipped_count, failed}`. |
| `POST` | `/projects/:id/images/check-import` | Same shape as the movie version, for the Import Images dialog's live validation |
| `POST` | `/projects/:id/image-groups/:gid/invert` | Invert a group against All Images (self-reversing; `400` for All Images) |
| `GET` | `/projects/:id/run-profiles` | `{ "run_profiles": [{run_profile_id, profile_name, manager_run_command, gui_address, controller_address, run_commands: [...], total_jobs}, ...] }` — fills the Run Profile picker and the Settings editor; `total_jobs == 0` greys the start button |
| `POST` | `/projects/:id/run-profiles` | Add a profile: `{}` for cisTEM's "Default Local", `{ "copy_of": id }` to duplicate, or a full profile (the `GET` shape) to import → `201` with the new profile |
| `PATCH` | `/projects/:id/run-profiles/:rid` | Change any of `profile_name`, `manager_run_command`, `gui_address`, `controller_address`, `run_commands` (each command: `{command, copies, threads_per_copy, override_total_copies, overridden_total_copies, delay_ms}`); `400` if a command lacks `$command` |
| `DELETE` | `/projects/:id/run-profiles/:rid` | Remove a profile and its commands |
| `GET` | `/projects/:id/jobs` | List jobs: `{ "jobs": [Job, ...] }` |
| `POST` | `/projects/:id/jobs` | Create a job. Body: `{ "stage", "params": {...} }` → returns the created `Job`. The server assigns the job's number and name (`Job 3`) — there's no name in the body, and `params` carries no output path either (see `Job` below). `400` if `params.run_profile` names a profile with no run commands |
| `GET` | `/projects/:id/jobs/:id/log` | `{ "log": "plain text, newline separated" }` |
| `POST` | `/projects/:id/jobs/:id/cancel` | Best-effort cancel → returns the updated `Job` |

Movies and images share one implementation on the server (`AssetKind` in `cistem_server.py`), so every group route exists for both kinds under the matching prefix and with the matching body key — `POST /projects/:id/{movies,images}/delete`, `POST /projects/:id/{movies,images}/add-to-group`, `POST|PATCH|DELETE /projects/:id/{movie,image}-groups[/:gid]`, and `POST /projects/:id/{movie,image}-groups/:gid/remove-{movies,images}`, taking `movie_ids` or `image_ids` respectively. Group `0` is the master list ("All Movies" / "All Images") and refuses rename, delete and invert.

`Job` shape:

```json
{
  "id": "a1b2c3d4e5",
  "stage": "motion_correction",
  "number": 3,
  "name": "Job 3",
  "params": { "movie_group_id": 1, "...": "...", "run_profile": "local_single" },
  "status": "queued | running | completed | failed | cancelled",
  "progress": 0,
  "created_at": "2026-09-01T14:03:00+00:00",
  "started_at": null,
  "finished_at": null,
  "error": null,
  "metrics": {}
}
```

Stages: `motion_correction` ("Align Movies"), `ctf_estimation` ("Find CTF"), `particle_picking` ("Find Particles"), `class2d` ("2D Classification"), `refine3d` ("Refine 3D").

`number` is per-project and assigned on creation (`MAX(JOB_NUMBER) + 1`), and `name` is just `Job <number>` — the submit form asks for neither, the same way cisTEM doesn't. Output paths are the server's too: a completed Align Movies job writes its aligned sums to `<project dir>/Assets/Images/<movie>_aligned.mrc`, mirroring cisTEM's own project layout, so no job parameter names a directory.

Point the page at any server that implements this contract — the reference Flask app is one option, not a requirement. A different backend just needs to keep the same project/job bookkeeping shape around it.

## Running real jobs

Stages with an adapter in `server/stages/` (today: Align Movies → `unblur`) run for real through the **job runner** when a controller executable can be found; everything else, and everything when it can't, runs in simulation exactly as before. The runner is the server's half of the protocol in `docs/job-protocol.md`: it listens for a per-job `cistem_job_controller` process, which the run profile's manager command launches and which in turn launches the cisTEM workers.

Settings, all environment variables:

| variable | default | meaning |
|---|---|---|
| `CISTEM_JOB_CONTROLLER` | `cistem_job_controller` | the controller command; the first word must be on `PATH` or the server falls back to simulation |
| `JOB_RUNNER_PORT` | `8010` | port the runner listens on for controllers |
| `JOB_RUNNER_BIND` | `0.0.0.0` | bind address |
| `JOB_RUNNER_HOSTS` | this machine's addresses, loopback last | comma-separated addresses the controller is told to dial, for NAT or multi-homed hosts |
| `JOB_RUNNER_ENABLED` | `1` | `0` never starts the listener |

To exercise the whole server side without a cisTEM build, point the controller setting at the Python stand-in, which speaks the protocol and fakes the workers:

```bash
CISTEM_JOB_CONTROLLER="python3 $PWD/tools/fake_controller.py" python server/cistem_server.py
```

The controller itself is `cistem_job_controller`, built from the cisTEM tree (`src/programs/cistem_job_controller/`, wired into both the autotools and CMake builds) — point `CISTEM_JOB_CONTROLLER` at it, and make sure the worker executables (`unblur`, …) are on the `PATH` of the server process, which the controller and its workers inherit. Each job's controller output (including the workers' stdout, since they inherit it) lands in `<project dir>/Logs/<job id>_controller.log`; per-task results are recorded in `JOB_TASKS` as they arrive and turned into `MOVIE_ALIGNMENT_LIST` rows, per-frame `MOVIE_ALIGNMENT_PARAMETERS_<id>` tables and image assets when the job finishes. A job that was running when the server stopped is **not** failed on restart any more: its controller reconnects within the reconnect window (10 minutes) and carries on.

Run profiles carry their commands (`RUN_PROFILE_COMMANDS_<id>`, cisTEM's own tables), seeded like cisTEM's defaults, and are edited on the Settings tab the way cisTEM's Run Profiles panel does it; the seeded Slurm profile has no commands and is refused at submit until you give it some there.

Tests: `python -m unittest discover -s server/tests` — codec tests plus runner integration tests that launch the fake controller for real.

## Security

Every request is authenticated (see "Users and access control" above) and projects are only visible to their owner or an admin. What's still explicitly *not* built, matching this app's "trusted lab network" scope: no rate limiting or lockout on login attempts, no HTTPS enforcement (bearer tokens over plain HTTP are sniffable on a hostile network — this only matters if the app leaves a trusted network), no password reset flow (a forgotten password needs an admin to recreate the account), and no user delete/edit endpoints (only create and list). If any of that matters for your deployment, put a reverse proxy in front that handles TLS and rate limiting, and treat the missing pieces above as the next things to build.

## Working on this in Claude Code

See `CLAUDE.md` for the project's architecture, design conventions, and known next steps — Claude Code reads it automatically at the start of a session.
