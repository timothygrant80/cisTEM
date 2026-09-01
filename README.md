# Cryo-EM Job Runner

A local job-submission UI for a single-particle cryo-EM processing pipeline (Align Movies → Find CTF → Find Particles → 2D Classification → Refine 3D), styled after cisTEM's desktop interface.

Two pieces:

- **`job_runner.html`** — a standalone page you open in your own browser. It submits jobs and polls status against a pipeline API you point it at (Settings tab). It makes no network calls anywhere else.
- **`server/app.py`** — a small reference Flask API implementing the contract the page expects. Ships in "simulation mode" (fake progress + fake numbers) so you can try the whole flow before your real pipeline is wired in.

`reference/dashboard.html` is an earlier static mockup with richer diagnostic charts (throughput, defocus histogram, FSC curve) — a design reference, not wired to the job runner.

## Why a local file instead of a hosted link

The page needs to call an API on your own network (cryoSPARC, a Slurm cluster, a lab script) — a page hosted on claude.ai is sandboxed and can't reach arbitrary private servers. Running `job_runner.html` locally sidesteps that: it's an ordinary page, so it can call whatever your browser can reach.

## Quick start (try it with the reference server)

```bash
cd server
pip install -r requirements.txt
python app.py
```

This serves the API at `http://localhost:8000/api`. Then open `job_runner.html` (double-click it, or `open job_runner.html`). Go to **Settings**, enter `http://localhost:8000/api` as the API base URL, and hit **Connect**.

Submit a job from **Actions** — since no real binaries are configured yet, it runs in simulation mode: progresses through a fake sequence and lands on `completed` with plausible placeholder numbers, so you can confirm the queue, log drawer, and cancel button all behave before pointing it at anything real.

If your browser blocks `fetch` from a `file://` page, serve the folder instead: `python -m http.server 8080` from this directory, then open `http://localhost:8080/job_runner.html`.

## Wiring in your real pipeline

Open `server/app.py` and fill in `STAGE_COMMANDS` — one command template per stage, using `{param_key}` placeholders filled from the job's submitted parameters:

```python
"motion_correction": {
    "binary": "MotionCor2",
    "command": [
        "MotionCor2", "-InTiff", "{input_glob}",
        "-Gain", "{gain_ref}", "-PixSize", "{pixel_size_a}",
        "-FmDose", "{dose_per_frame}", "-OutMrc", "{output_dir}/",
    ],
},
```

Any stage left as `"command": None`, or whose `binary` isn't found on `PATH`, keeps running in simulation mode — so you can wire stages up one at a time.

If you're driving **cryoSPARC** or **Slurm** instead of calling binaries directly, replace the body of `_run_real()` in `app.py` with calls to cryoSPARC's JSON API or `slurmrestd`, keeping the same job bookkeeping (status, progress, log, cancel) around it. The front end doesn't need to change either way — it only knows the HTTP contract below.

## The HTTP contract

`job_runner.html` only ever calls these, against whatever base URL you give it:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Connectivity check. Any 2xx JSON response counts. |
| `GET` | `/jobs` | List jobs: `{ "jobs": [Job, ...] }` |
| `POST` | `/jobs` | Create a job. Body: `{ "stage", "name", "params": {...} }` → returns the created `Job` |
| `GET` | `/jobs/:id/log` | `{ "log": "plain text, newline separated" }` |
| `POST` | `/jobs/:id/cancel` | Best-effort cancel → returns the updated `Job` |

`Job` shape:

```json
{
  "id": "a1b2c3d4e5",
  "stage": "motion_correction",
  "name": "grid1_session_001",
  "params": { "...": "...", "run_profile": "local_single" },
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

Point the page at any server that implements this contract — the reference Flask app is one option, not a requirement.

## Security

The page sends no authentication header — an open network was the explicit choice for now. If you later expose this beyond a trusted lab network, add an `Authorization` header in `job_runner.html`'s `fetch` calls and check it in `app.py` (or put a reverse proxy in front that handles auth).

## Working on this in Claude Code

See `CLAUDE.md` for the project's architecture, design conventions, and known next steps — Claude Code reads it automatically at the start of a session.
