# Cryo-EM Job Runner

A local, project-based job-submission UI for a single-particle cryo-EM processing pipeline, styled after cisTEM's desktop GUI, plus a reference backend to develop against.

## Layout

- `job_runner.html` — the actual app. Single self-contained file (inline CSS + vanilla JS, no build step, no dependencies). Open it directly in a browser. On load it shows a home screen (create/open a project, closer to real cisTEM's own launcher) before the Assets/Actions/Results/Settings shell; once a project is open, it talks to a pipeline API over plain `fetch()` at whatever base URL is entered in the Connection panel (persisted to `localStorage`), scoped to that project.
- `server/app.py` — reference Flask implementation of the API contract below. Runs jobs in **simulation mode** (fake progress + plausible fake numbers) for any stage whose binary isn't on `PATH`, so the full flow works before real tooling is wired in. Real commands go in `STAGE_COMMANDS` at the top of the file.
- `server/db.py` — per-project SQLite schema and helpers (`create_project`, `list_projects`, `get_conn`, ...). Table names/columns are lifted from a real cisTEM project database wherever this app stores the same kind of data, so a project created here reads like a real cisTEM project — see the module docstring for the couple of deliberate deviations.
- `server/data/` — created at runtime, one directory + `project.db` per project (`server/data/projects/<project_id>/project.db`). Gitignored; nothing here is meant to be committed.
- `server/requirements.txt` — `flask` + `flask-cors` (sqlite3 is Python stdlib, no extra dependency).
- `reference/dashboard.html` — an earlier, read-only mockup (static "empty state" dashboard, published as a Claude Artifact) with richer diagnostic charts (throughput, defocus histogram, FSC curve) that `job_runner.html` doesn't have yet. Kept as a design reference, not wired to anything.
- `README.md` — setup instructions and the full API contract, written for a human reader.

## Why this shape

`job_runner.html` **cannot** be a Claude-hosted Artifact page: it needs to call an API on the user's own network (cryoSPARC, Slurm, a lab script), and Anthropic's Artifact sandbox blocks outbound calls to anything outside a small CDN allowlist. So this lives as a plain local file instead — no platform restrictions, but also no hosted link; it's opened directly or served with something like `python -m http.server`.

The backend is deliberately a *reference*, not a real scheduler: one thread per running job, no distributed job manager. Treat `server/app.py` as scaffolding to either extend directly or as the shape to replicate against a real cryoSPARC/Slurm-backed service. Project data itself (imported movies, job history, results) *is* durable — each project is its own SQLite file, survives a server restart, and any job still `queued`/`running` when the server last stopped is marked `failed` on the next startup rather than left stuck (there's no thread left to finish it).

## Projects

Everything a project owns lives in its own SQLite file (`server/db.py`'s `SCHEMA_SQL`), not a shared global store — opening a different project just means pointing requests at a different file. This is what lets Assets metadata (imported movies' voltage, pixel size, dose/frame, gain reference) be entered once at import time instead of re-typed on every job form: `Align Movies` jobs pick a **movie group** imported on the Assets tab, rather than typing a glob + metadata directly.

**Scope note:** only `motion_correction` (Align Movies) is wired to real project data end-to-end (`MOVIE_ASSETS` → job → `MOVIE_ALIGNMENT_LIST` + `IMAGE_ASSETS`), because it's the only stage whose expert options were scraped from real cisTEM source (see below). `ctf_estimation`, `particle_picking`, `class2d`, and `refine3d` are still typed/freeform forms like before — they're project-scoped (their `JOBS` rows land in the right project) but don't yet read from or write to their corresponding `_LIST` tables (`ESTIMATED_CTF_PARAMETERS`, `PARTICLE_PICKING_LIST`, `CLASSIFICATION_LIST`, `REFINEMENT_LIST` — schema already created, just unused). Give them the same real-cisTEM-source scrape Align Movies got before wiring them the same way.

## The API contract (frontend ↔ backend)

Everything except `/health` and `/projects` itself is scoped under a project id:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Connectivity check — any 2xx JSON response counts |
| `GET` | `/projects` | `{ "projects": [{id, name, total_jobs_run, ...}, ...] }` |
| `POST` | `/projects` | Body `{ "name" }` → creates a project, returns its summary |
| `GET` | `/projects/:id` | One project's summary |
| `DELETE` | `/projects/:id` | Deletes a project (and its `.db` file) permanently |
| `GET` | `/projects/:id/movies` | Imported `MOVIE_ASSETS` rows |
| `GET` | `/projects/:id/movie-groups` | Movie groups + member counts (feeds the Align Movies picker) |
| `GET` | `/projects/:id/movies/import-defaults` | Last-used import form values |
| `POST` | `/projects/:id/movies/import` | Body `{input_glob, group_name, voltage_kv, cs_mm, pixel_size_a, dose_per_frame, gain_ref, dark_ref}` — imports matching files as movie assets into a new group (and the standing "All Movies" group). If the glob matches nothing, fabricates ~12 placeholder movies (same spirit as the job simulator) so the flow works without real data on hand. |
| `GET` | `/projects/:id/jobs` | `{ "jobs": [Job, ...] }` |
| `POST` | `/projects/:id/jobs` | Body `{ "stage", "name", "params": {...} }` → creates a `Job`. For `motion_correction`, `params` must include `movie_group_id` (not a glob) instead of the metadata fields now sourced from `MOVIE_ASSETS`. |
| `GET` | `/projects/:id/jobs/:id/log` | `{ "log": "plain text, newline separated" }` |
| `POST` | `/projects/:id/jobs/:id/cancel` | Best-effort cancel → returns the updated `Job` |

`Job`: `{ id, stage, name, params, status: queued|running|completed|failed|cancelled, progress, created_at, started_at, finished_at, error, metrics }`.

Stages (keys are fixed and referenced from both the frontend `STAGES` object and the backend `STAGE_COMMANDS` dict — keep them in sync if you rename anything):
`motion_correction`, `ctf_estimation`, `particle_picking`, `class2d`, `refine3d`.

Any server that implements this contract works with the front end unchanged — the reference Flask app is one option, not a requirement. If you swap in a different backend (e.g. a thin proxy in front of cryoSPARC's JSON API or `slurmrestd`), keep the same project/job bookkeeping shape around it.

## Design language (intentional, keep it consistent)

Restyled to evoke cisTEM's native wxWidgets desktop GUI, not a modern web-app look:

- Single light theme, no dark mode — cisTEM doesn't have one either, so this doesn't pretend to.
- System fonts only (`-apple-system`/`Segoe UI`/etc. + a monospace stack) — no web fonts, so the page has zero external network dependency and works on an offline lab machine.
- Home screen (project picker) follows the same idiom as the rest of the app — bordered "groupbox" panels, no cards/shadows — rather than a separate visual system.
- Left icon rail navigation: **Assets / Actions / Results / Settings**, matching cisTEM's own top-level structure. Assets = movie import + imported movies/groups + job counts + per-stage inventory; Actions = the submit form (with a "Show Expert Options" disclosure, another cisTEM pattern); Results = the queue + log drawer; Settings = API connection + run profile.
- Stage labels use cisTEM's own vocabulary ("Align Movies", "Find CTF", "Find Particles") even though the internal stage keys stay the generic snake_case names the API contract uses.
- "Group box" panels: bordered boxes with the title cut into the top border (classic native-toolkit fieldset look), not cards with drop shadows. No border-radius to speak of, no elevation shadows anywhere — keep new UI flat and bordered, not shadowed.

If you add new stages or panels, follow this same idiom rather than reverting to generic modern-dashboard styling (rounded cards, shadows, a display webfont).

## Known gaps / natural next steps

- `STAGE_COMMANDS` in `server/app.py` has placeholder command templates commented out — wiring in real MotionCor2/CTFFIND/RELION binaries (or replacing `_run_real()` with cryoSPARC/`slurmrestd` API calls) is the main remaining work.
- No auth on the API (explicit choice — trusted-network-only for now). If this ever needs to leave a trusted network, add an `Authorization` header in `job_runner.html`'s `fetch` calls and check it server-side.
- `ctf_estimation`, `particle_picking`, `class2d`, `refine3d` still use typed freeform params and don't read/write their project-database tables yet — see "Projects" above.
- The richer diagnostic charts in `reference/dashboard.html` (throughput over time, defocus distribution, FSC curve) aren't in `job_runner.html`'s Results tab yet — could be ported in once there's real per-job metric data to plot.
- `RUN_PROFILES` is seeded per project (matching real cisTEM columns) but not yet editable from the UI — the Settings tab's run-profile selector is still the original 3 hardcoded options, not read from the database.
