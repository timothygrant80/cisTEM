# Cryo-EM Job Runner

A local, project-based job-submission UI for a single-particle cryo-EM processing pipeline, styled after cisTEM's desktop GUI, plus a reference backend to develop against.

## Layout

- `job_runner.html` — the actual app. Single self-contained file (inline CSS + vanilla JS, no build step, no dependencies). Open it directly in a browser. On load it shows a home screen — connect to an API, log in, then create/open a project (closer to real cisTEM's own launcher) — before the Assets/Actions/Results/Settings shell; once a project is open, it talks to a pipeline API over plain `fetch()` at whatever base URL is entered in the Connection panel (persisted to `localStorage`), scoped to that project and authenticated as the logged-in user.
- `server/app.py` — reference Flask implementation of the API contract below. Runs jobs in **simulation mode** (fake progress + plausible fake numbers) for any stage whose binary isn't on `PATH`, so the full flow works before real tooling is wired in. Real commands go in `STAGE_COMMANDS` at the top of the file.
- `server/db.py` — per-project SQLite schema and helpers (`create_project`, `list_projects`, `get_conn`, ...). Table names/columns are lifted from a real cisTEM project database wherever this app stores the same kind of data, so a project created here reads like a real cisTEM project — see the module docstring for the couple of deliberate deviations.
- `server/auth.py` — user accounts, login, and access control: bearer-token sessions, `login_required`/`admin_required`/`project_access_required` route decorators, and the one-time bootstrap of an initial admin account. Its own SQLite file, separate from any project's (`server/data/auth.db`), since users/sessions are cross-project — see "Users and access control" below.
- `server/data/` — created at runtime: `auth.db` (users/sessions), `admin_credentials.txt` (written once, see below), and `projects/<project_id>/project.db` per project. Gitignored; nothing here is meant to be committed.
- `server/requirements.txt` — `flask` + `flask-cors` (sqlite3 and password hashing via `werkzeug.security` are both already available — Flask's own dependencies — no extra packages needed for auth).
- `reference/dashboard.html` — an earlier, read-only mockup (static "empty state" dashboard, published as a Claude Artifact) with richer diagnostic charts (throughput, defocus histogram, FSC curve) that `job_runner.html` doesn't have yet. Kept as a design reference, not wired to anything.
- `README.md` — setup instructions and the full API contract, written for a human reader.

## Why this shape

`job_runner.html` **cannot** be a Claude-hosted Artifact page: it needs to call an API on the user's own network (cryoSPARC, Slurm, a lab script), and Anthropic's Artifact sandbox blocks outbound calls to anything outside a small CDN allowlist. So this lives as a plain local file instead — no platform restrictions, but also no hosted link; it's opened directly or served with something like `python -m http.server`.

The backend is deliberately a *reference*, not a real scheduler: one thread per running job, no distributed job manager. Treat `server/app.py` as scaffolding to either extend directly or as the shape to replicate against a real cryoSPARC/Slurm-backed service. Project data itself (imported movies, job history, results) *is* durable — each project is its own SQLite file, survives a server restart, and any job still `queued`/`running` when the server last stopped is marked `failed` on the next startup rather than left stuck (there's no thread left to finish it).

## Users and access control

Every request except `/health`, `/auth/login`, and `/auth/logout` requires an `Authorization: Bearer <token>` header, checked against `server/auth.py`'s `SESSIONS` table (opaque tokens, not JWTs — server-tracked so logout actually revokes them). Two roles: `user` (sees and manages only their own projects) and `admin` (sees/accesses every project, and is the only role that can create new accounts via `POST /users` — there's no public self-registration). Ownership lives in each project's own `MASTER_SETTINGS` (`OWNER_USER_ID`/`OWNER_USERNAME`), not a central table — `db.py`'s `list_projects()` already opens every project's file to build the picker, so filtering by owner there is free and there's no second source of truth to drift.

On first run (`USERS` table empty), the server auto-creates an `admin` account with a random password, prints it to the console, and writes it once to `server/data/admin_credentials.txt` (`0600` permissions, gitignored). Log in as that account and create real accounts for your team from the home screen's Manage Users panel (admin-only). See `server/auth.py`'s module docstring for the full session/token design (bearer over cookies, timing-safe login, sliding expiry).

## Projects

Everything a project owns lives in its own SQLite file (`server/db.py`'s `SCHEMA_SQL`), not a shared global store — opening a different project just means pointing requests at a different file. This is what lets Assets metadata (imported movies' voltage, pixel size, dose/frame, gain reference) be entered once at import time instead of re-typed on every job form: `Align Movies` jobs pick a **movie group** imported on the Assets tab, rather than typing a glob + metadata directly.

**Scope note:** only `motion_correction` (Align Movies) is wired to real project data end-to-end (`MOVIE_ASSETS` → job → `MOVIE_ALIGNMENT_LIST` + `IMAGE_ASSETS`), because it's the only stage whose expert options were scraped from real cisTEM source (see below). `ctf_estimation`, `particle_picking`, `class2d`, and `refine3d` are still typed/freeform forms like before — they're project-scoped (their `JOBS` rows land in the right project) but don't yet read from or write to their corresponding `_LIST` tables (`ESTIMATED_CTF_PARAMETERS`, `PARTICLE_PICKING_LIST`, `CLASSIFICATION_LIST`, `REFINEMENT_LIST` — schema already created, just unused). Give them the same real-cisTEM-source scrape Align Movies got before wiring them the same way.

## The API contract (frontend ↔ backend)

Everything except `/health`, `/auth/login`, and `/auth/logout` requires auth (see "Users and access control" above); everything except `/health`/`/auth/*`/`/users` is additionally scoped under a project id, and enforces that the caller owns that project (or is an admin):

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/health` | none | Connectivity check — any 2xx JSON response counts |
| `POST` | `/auth/login` | none | Body `{username, password}` → `{token, user}` |
| `POST` | `/auth/logout` | best-effort | Revokes the presented token if any; always `200` |
| `GET` | `/auth/me` | any user | `{user}` — used to silently re-validate a stored token on load |
| `GET` | `/users` | admin | `{ "users": [{id, username, role, display_name, ...}, ...] }` |
| `POST` | `/users` | admin | Body `{username, password, role, display_name}` → creates a user |
| `GET` | `/projects` | any user | `{ "projects": [...] }` — own projects only, unless admin (then all, with `owner_username`) |
| `POST` | `/projects` | any user | Body `{ "name" }` → creates a project owned by the caller |
| `GET` | `/projects/:id` | owner or admin | One project's summary |
| `DELETE` | `/projects/:id` | owner or admin | Deletes a project (and its `.db` file) permanently |
| `GET` | `/projects/:id/movies` | owner or admin | Imported `MOVIE_ASSETS` rows |
| `GET` | `/projects/:id/movie-groups` | owner or admin | Movie groups + member counts (feeds the Align Movies picker) |
| `GET` | `/projects/:id/movies/import-defaults` | owner or admin | Last-used import form values |
| `POST` | `/projects/:id/movies/import` | owner or admin | Body `{input_glob, group_name, voltage_kv, cs_mm, pixel_size_a, dose_per_frame, gain_ref, dark_ref}` — imports matching files as movie assets into a new group (and the standing "All Movies" group). If the glob matches nothing, fabricates ~12 placeholder movies (same spirit as the job simulator) so the flow works without real data on hand. |
| `GET` | `/projects/:id/jobs` | owner or admin | `{ "jobs": [Job, ...] }` |
| `POST` | `/projects/:id/jobs` | owner or admin | Body `{ "stage", "name", "params": {...} }` → creates a `Job`. For `motion_correction`, `params` must include `movie_group_id` (not a glob) instead of the metadata fields now sourced from `MOVIE_ASSETS`. |
| `GET` | `/projects/:id/jobs/:id/log` | owner or admin | `{ "log": "plain text, newline separated" }` |
| `POST` | `/projects/:id/jobs/:id/cancel` | owner or admin | Best-effort cancel → returns the updated `Job` |

`Job`: `{ id, stage, name, params, status: queued|running|completed|failed|cancelled, progress, created_at, started_at, finished_at, error, metrics }`.

Stages (keys are fixed and referenced from both the frontend `STAGES` object and the backend `STAGE_COMMANDS` dict — keep them in sync if you rename anything):
`motion_correction`, `ctf_estimation`, `particle_picking`, `class2d`, `refine3d`.

Any server that implements this contract works with the front end unchanged — the reference Flask app is one option, not a requirement. If you swap in a different backend (e.g. a thin proxy in front of cryoSPARC's JSON API or `slurmrestd`), keep the same project/job bookkeeping shape around it.

## Design language (intentional, keep it consistent)

Restyled to evoke cisTEM's native wxWidgets desktop GUI, not a modern web-app look:

- Single light theme, no dark mode — cisTEM doesn't have one either, so this doesn't pretend to.
- System fonts only (`-apple-system`/`Segoe UI`/etc. + a monospace stack) — no web fonts, so the page has zero external network dependency and works on an offline lab machine.
- Home screen (connect → log in → project picker, plus an admin-only Manage Users panel) follows the same idiom as the rest of the app — bordered "groupbox" panels, no cards/shadows — rather than a separate visual system.
- Left icon rail navigation: **Assets / Actions / Results / Settings**, matching cisTEM's own top-level structure. Assets = movie import + imported movies/groups + job counts + per-stage inventory; Actions = the submit form (with a "Show Expert Options" disclosure, another cisTEM pattern); Results = the queue + log drawer; Settings = API connection + run profile.
- Stage labels use cisTEM's own vocabulary ("Align Movies", "Find CTF", "Find Particles") even though the internal stage keys stay the generic snake_case names the API contract uses.
- "Group box" panels: bordered boxes with the title cut into the top border (classic native-toolkit fieldset look), not cards with drop shadows. No border-radius to speak of, no elevation shadows anywhere — keep new UI flat and bordered, not shadowed.

If you add new stages or panels, follow this same idiom rather than reverting to generic modern-dashboard styling (rounded cards, shadows, a display webfont).

## Known gaps / natural next steps

- `STAGE_COMMANDS` in `server/app.py` has placeholder command templates commented out — wiring in real MotionCor2/CTFFIND/RELION binaries (or replacing `_run_real()` with cryoSPARC/`slurmrestd` API calls) is the main remaining work.
- No rate limiting or HTTPS enforcement on the API (explicit choice — trusted-network-only for now, same as before auth existed). Bearer tokens over plain HTTP are sniffable on a hostile network; this only matters if the app ever leaves a trusted lab network.
- No password reset or self-service password change — a forgotten password today has no recovery path short of an admin recreating the account. Smallest next step, if needed: an admin-driven password-set endpoint, not a full email-based reset flow (no email infra exists here to build that on).
- No user delete/edit/role-change endpoints — only create (`POST /users`) and list (`GET /users`) exist. Would reuse the exact `admin_required` pattern in `server/auth.py`.
- `ctf_estimation`, `particle_picking`, `class2d`, `refine3d` still use typed freeform params and don't read/write their project-database tables yet — see "Projects" above.
- The richer diagnostic charts in `reference/dashboard.html` (throughput over time, defocus distribution, FSC curve) aren't in `job_runner.html`'s Results tab yet — could be ported in once there's real per-job metric data to plot.
- `RUN_PROFILES` is seeded per project (matching real cisTEM columns) but not yet editable from the UI — the Settings tab's run-profile selector is still the original 3 hardcoded options, not read from the database.
