# Cryo-EM Job Runner

A local job-submission UI for a single-particle cryo-EM processing pipeline, styled after cisTEM's desktop GUI, plus a reference backend to develop against.

## Layout

- `job_runner.html` — the actual app. Single self-contained file (inline CSS + vanilla JS, no build step, no dependencies). Open it directly in a browser. It talks to a pipeline API over plain `fetch()` at whatever base URL is entered in the Settings tab (persisted to `localStorage`).
- `server/app.py` — reference Flask implementation of the API contract below. Runs jobs in **simulation mode** (fake progress + plausible fake numbers) for any stage whose binary isn't on `PATH`, so the full flow works before real tooling is wired in. Real commands go in `STAGE_COMMANDS` at the top of the file.
- `server/requirements.txt` — `flask` + `flask-cors`.
- `reference/dashboard.html` — an earlier, read-only mockup (static "empty state" dashboard, published as a Claude Artifact) with richer diagnostic charts (throughput, defocus histogram, FSC curve) that `job_runner.html` doesn't have yet. Kept as a design reference, not wired to anything.
- `README.md` — setup instructions and the full API contract, written for a human reader.

## Why this shape

`job_runner.html` **cannot** be a Claude-hosted Artifact page: it needs to call an API on the user's own network (cryoSPARC, Slurm, a lab script), and Anthropic's Artifact sandbox blocks outbound calls to anything outside a small CDN allowlist. So this lives as a plain local file instead — no platform restrictions, but also no hosted link; it's opened directly or served with something like `python -m http.server`.

The backend is deliberately a *reference*, not a real scheduler: in-memory job store, one thread per job, no persistence across restarts. Treat `server/app.py` as scaffolding to either extend directly or as the shape to replicate against a real cryoSPARC/Slurm-backed service.

## The API contract (frontend ↔ backend)

`job_runner.html` only ever calls these, relative to the configured base URL:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Connectivity check — any 2xx JSON response counts |
| `GET` | `/jobs` | `{ "jobs": [Job, ...] }` |
| `POST` | `/jobs` | Body `{ "stage", "name", "params": {...} }` → creates a `Job` |
| `GET` | `/jobs/:id/log` | `{ "log": "plain text, newline separated" }` |
| `POST` | `/jobs/:id/cancel` | Best-effort cancel → returns the updated `Job` |

`Job`: `{ id, stage, name, params, status: queued|running|completed|failed|cancelled, progress, created_at, started_at, finished_at, error, metrics }`.

Stages (keys are fixed and referenced from both the frontend `STAGES` object and the backend `STAGE_COMMANDS` dict — keep them in sync if you rename anything):
`motion_correction`, `ctf_estimation`, `particle_picking`, `class2d`, `refine3d`.

Any server that implements this contract works with the front end unchanged — the reference Flask app is one option, not a requirement. If you swap in a different backend (e.g. a thin proxy in front of cryoSPARC's JSON API or `slurmrestd`), keep the same job bookkeeping shape around it.

## Design language (intentional, keep it consistent)

Restyled to evoke cisTEM's native wxWidgets desktop GUI, not a modern web-app look:

- Single light theme, no dark mode — cisTEM doesn't have one either, so this doesn't pretend to.
- System fonts only (`-apple-system`/`Segoe UI`/etc. + a monospace stack) — no web fonts, so the page has zero external network dependency and works on an offline lab machine.
- Left icon rail navigation: **Assets / Actions / Results / Settings**, matching cisTEM's own top-level structure. Assets = job counts + per-stage inventory; Actions = the submit form (with a "Show Expert Options" disclosure, another cisTEM pattern); Results = the queue + log drawer; Settings = API connection + run profile.
- Stage labels use cisTEM's own vocabulary ("Align Movies", "Find CTF", "Find Particles") even though the internal stage keys stay the generic snake_case names the API contract uses.
- "Group box" panels: bordered boxes with the title cut into the top border (classic native-toolkit fieldset look), not cards with drop shadows. No border-radius to speak of, no elevation shadows anywhere — keep new UI flat and bordered, not shadowed.

If you add new stages or panels, follow this same idiom rather than reverting to generic modern-dashboard styling (rounded cards, shadows, a display webfont).

## Known gaps / natural next steps

- `STAGE_COMMANDS` in `server/app.py` has placeholder command templates commented out — wiring in real MotionCor2/CTFFIND/RELION binaries (or replacing `_run_real()` with cryoSPARC/`slurmrestd` API calls) is the main remaining work.
- No auth on the API (explicit choice — trusted-network-only for now). If this ever needs to leave a trusted network, add an `Authorization` header in `job_runner.html`'s `fetch` calls and check it server-side.
- The richer diagnostic charts in `reference/dashboard.html` (throughput over time, defocus distribution, FSC curve) aren't in `job_runner.html`'s Results tab yet — could be ported in once there's real per-job metric data to plot.
- No persistence: restarting `server/app.py` drops the job history.
