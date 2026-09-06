# cisTEM3

A local, project-based job-submission UI for a single-particle cryo-EM processing pipeline (Align Movies → Find CTF → Find Particles → 2D Classification → Ab-Initio 3D → Auto Refine / Refine 3D → Refine CTF → Generate 3D → Sharpen 3D), styled after cisTEM's desktop interface.

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

Open `server/cistem_server.py` and fill in `STAGE_COMMANDS` — one command template per stage, using `{param_key}` placeholders filled from the job's submitted parameters. For the stage still using freeform typed params (`refine3d`) this is a direct 1:1 mapping, e.g.:

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
| `PATCH` | `/users/:id` | Admin only. Body `{role}` (`admin` or `user`) → updated user; refuses to demote yourself or the last admin |
| `GET` | `/projects` | List projects: `{ "projects": [{id, name, total_jobs_run, owner_username, creation_date, ...}, ...] }` — your own, or all of them if you're an admin |
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
| `GET` | `/projects/:id/alignments` | `{ "alignments": [...] }` — every movie alignment with movie name, produced image asset id, job number, and whether its sum/spectrum files exist |
| `GET` | `/projects/:id/alignments/:aid` | One alignment plus `shifts: [{frame, x, y}]` in Å |
| `POST` | `/projects/:id/alignments/:aid/activate` | Make this alignment the movie's active one (its image asset now points at it) |
| `POST` | `/projects/:id/jobs/:id/activate-results` | Make one job's results active for every asset it processed → `{activated, failed}` (`/activate-alignments` is an alias) |
| `GET` | `/projects/:id/ctf-estimates` | `{ "ctf_estimates": [...] }` — every CTF estimate with image name, job number and `is_active` |
| `GET` | `/projects/:id/ctf-estimates/:cid` | One estimate plus `plot` (the 1D fit curves) |
| `POST` | `/projects/:id/ctf-estimates/:cid/activate` | Make this estimate the image's active one |
| `GET` | `/projects/:id/ctf-estimates/:cid/diagnostic.png` | PNG of ctffind's diagnostic image |
| `POST` | `/projects/:id/preview/pick` | Run the particle picker on one image with the given parameters, without a job → the picks |
| `GET` | `/projects/:id/particle-positions` | Particle position assets (`?group_id=`, `?image_id=`), with parent image name and pick job number; capped at 5000 rows |
| `GET`/`POST`/`PATCH`/`DELETE` | `/projects/:id/particle-position-groups[/:gid]`, `.../invert`, `.../remove-particle-positions`, `/particle-positions/delete`, `/particle-positions/add-to-group` | The same group routes movies and images have |
| `POST` | `/projects/:id/particle-positions/import` | Body `{text}` or `{path}`: lines of `<image id or filename> <x> <y>` in Å → `{imported, failed, warnings}` |
| `POST` | `/projects/:id/particle-position-groups/from-image-group` | A position group holding the positions of every image in an image group |
| `GET`/`POST` | `/projects/:id/refinement-packages` | List packages / create one from a particle position group (`particle_group_id`) or from class selections (`selection_ids`, `recentre`, `remove_duplicates`, `duplicate_threshold_a`) — cuts the stack, writes the package and its "Random Parameters" refinement |
| `GET` | `/projects/:id/refinement-packages/defaults` | The wizard's prefills for a group (`?particle_group_id=&largest_dimension_a=`) or for class selections (`?selection_ids=1,2` → the parent package's box and pixel size, particle count) |
| `GET`/`PATCH`/`DELETE` | `/projects/:id/refinement-packages/:pid` | Details with contained particles / rename / delete (tables and stack file) |
| `GET` | `/projects/:id/classifications` | Every 2D classification (`?refinement_package_id=` for one package's) with its package name, job number and whether its class averages file exists |
| `GET` | `/projects/:id/class2d/defaults` | `MyRefine2DPanel::SetDefaults()` for a package (`?refinement_package_id=`): the class-count default and the earlier classifications the starting-references picker offers |
| `GET`/`DELETE` | `/projects/:id/classifications/:cid` | One classification with per-class member counts and the montage geometry / delete it (results table and class averages file) |
| `GET` | `/projects/:id/classifications/:cid/averages.png` | The class averages tiled into one PNG |
| `GET` | `/projects/:id/classifications/:cid/class/:k` | The members of class k (`?limit=`, active ones first); `.../class/:k/members.png` tiles them from the particle stack |
| `GET` | `/projects/:id/volumes`, `/volume-groups` | 3D volume assets (`?group_id=`) and their groups; the usual group routes (`POST|PATCH|DELETE /volume-groups[/:gid]`, `/:gid/invert`, `/:gid/remove-volumes`, `POST /volumes/delete`, `/volumes/add-to-group`, keyed on `volume_ids`) |
| `GET` | `/projects/:id/volumes/:vid/preview.png` | Orthogonal projections (top) and central slices (bottom) of the volume |
| `GET` | `/projects/:id/startups` | Every ab-initio run (`STARTUP_LIST`) with its settings and result volumes |
| `GET` | `/projects/:id/abinitio/defaults` | `AbInitio3DPanel::SetDefaults()` for a package (`?refinement_package_id=`): symmetry, mask radius, search ranges, class count, and the class selections a class-average run can start from |
| `GET` | `/projects/:id/jobs/:jid/abinitio/current.png` | Orthogonal views of a running or finished ab-initio or Refine 3D job's current reconstruction (`?class=`) |
| `GET` | `/projects/:id/refinements` | Every 3D refinement (`?refinement_package_id=`) with per-class estimated resolution, occupancy and reconstructed volume |
| `GET` | `/projects/:id/refinements/:rid` | One refinement with each class's FSC / SSNR curve and angular distribution |
| `GET` | `/projects/:id/jobs/:jid/refinectf/<which>.png` | A Refine CTF job's beam-tilt pictures: `phase_difference` (the measured phase spectrum) or `beam_tilt` (the pattern the found tilt predicts) |
| `GET` | `/projects/:id/sharpen/defaults` | `Sharpen3DPanel::OnVolumeComboBox()` for a volume (`?volume_asset_id=`): the mask radii of the reconstruction that made it, whether its refinement's statistics are available (`has_statistics`, `estimated_resolution`), the panel's `defaults`, the volumes a mask can be, and `available` (whether `sharpen_map` is on the server's PATH) |
| `POST` | `/projects/:id/sharpen` | Body `{volume_asset_id, params}` (the Sharpen 3D panel's fields) → runs `sharpen_map` now and returns `result_id`, the Guinier curves (`guinier.spatial_frequency/original/sharpened`), the central slices of both maps as PNG data URIs, `used_statistics`, `used_mask`, `elapsed_s`. `503` if the binary is missing, `400` for bad parameters, `504` after 900 s. Writes nothing to the project. |
| `GET` | `/projects/:id/sharpen/:rid/volume.mrc` | Save Result: the sharpened map as an MRC download (results are kept until the server restarts) |
| `POST` | `/projects/:id/sharpen/:rid/import` | `{name}` (optional) → the sharpened map as a new volume asset (`201`) |
| `GET` | `/projects/:id/auto-refine3d/defaults` | `AutoRefine3DPanel::SetDefaults()` for a package (`?refinement_package_id=`): the size-derived limits, the volumes the Starting Reference / mask pickers list (with `fits`, whether each matches the package's box and pixel size) and `suggested_reference_id` |
| `GET` | `/projects/:id/refine3d/defaults` | `MyRefine3DPanel::SetDefaults()` for a package (`?refinement_package_id=`): limits, the refinements that can be the input parameters, each class's current reference, the volumes a mask can be; also `particle_size` and the `generate3d` / `refine_ctf` size-derived defaults those two panels share |
| `PATCH` | `/projects/:id/refinement-packages/:pid/references` | `{class_number, volume_asset_id}` sets a class's current reference volume (-1 = generate from parameters) |
| `GET`/`POST` | `/projects/:id/classification-selections` | Named selections of class averages (`?classification_id=` / `?refinement_package_id=`), each with `classes` and `particle_count` / create one `{classification_id, name, classes}` |
| `PATCH`/`DELETE` | `/projects/:id/classification-selections/:sid` | `{name}` renames, `{classes: [...]}` replaces the membership / delete |
| `GET` | `/projects/:id/picks` | `{ "picks": [...] }` — every particle picking with image name/size, job number, pick count and `is_active` |
| `GET` | `/projects/:id/picks/:pid` | One picking plus `positions` (Å from the image origin) |
| `POST` | `/projects/:id/picks/:pid/activate` | Make these picks the image's particle positions |
| `GET` | `/projects/:id/alignments/:aid/sum.png`, `/spectrum.png` | PNG renders of the aligned sum and its amplitude spectrum |
| `GET` | `/run-profiles` | The machine's run profiles (system-wide, any logged-in user) — feeds the Run Profile picker |
| `POST` / `PATCH` / `DELETE` | `/run-profiles[/:rid]` | Admin only: add (empty body), duplicate (`{copy_of}`), import (full profile), edit, remove |
| `GET` | `/projects/:id/jobs` | List jobs: `{ "jobs": [Job, ...] }` |
| `POST` | `/projects/:id/jobs` | Create a job. Body: `{ "stage", "params": {...} }` → returns the created `Job`. The server assigns the job's number and name (`Job 3`) — there's no name in the body, and `params` carries no output path either (see `Job` below). `400` if `params.run_profile` names a profile with no run commands |
| `GET` | `/projects/:id/jobs/:id/latest-result` | Newest finished task's result for the Jobs tab's live panel (`result: null` + `reason` when there is none yet) |
| `GET` | `/projects/:id/jobs/:id/tasks/:tidx/<name>.png` | PNG of one task's output picture (`sum`, `spectrum`, `diagnostic`), available mid-run |
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

Stages: `motion_correction` ("Align Movies"), `ctf_estimation` ("Find CTF"), `particle_picking` ("Find Particles"), `class2d` ("2D Classification"), `ab_initio_3d` ("Ab-Initio 3D"), `auto_refine3d` ("Auto Refine": `params` are `refinement_package_id`, `reference_volume_id`, `high_resolution_limit_a`, `reconstruction_run_profile` and the expert options; the run decides its own round count), `refine3d` ("Refine 3D"), `refine_ctf` ("Refine CTF": `params` are `refinement_package_id`, `input_refinement_id`, `refine_defocus`, `refine_beam_tilt`, `high_resolution_limit_a`, `reconstruction_run_profile` and the expert options), `generate3d` ("Generate 3D": `refinement_package_id`, `input_refinement_id` and the reconstruction options; the run profile is the reconstruction profile). The Actions bar also carries **Sharpen 3D**, which is not a job stage: its Run button calls `POST /sharpen` and shows the result in the panel.

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

The controller itself is `cistem_job_controller`, built from the cisTEM tree (`src/programs/cistem_job_controller/`, wired into both the autotools and CMake builds) — point `CISTEM_JOB_CONTROLLER` at it, and make sure the worker executables (`unblur`, …) are on the `PATH` of the server process, which the controller and its workers inherit. The controller gives up if no worker at all connects within `--worker-timeout` seconds of launching the run commands (default 300; once one worker is in, the rest may take as long as a scheduler needs), failing the job with the reason in its log — put `$command --worker-timeout 3600` in a profile's manager command if a cluster queue can legitimately take longer. Each job's controller output (including the workers' stdout, since they inherit it) lands in `<project dir>/Logs/<job id>_controller.log`; per-task results are recorded in `JOB_TASKS` as they arrive and turned into `MOVIE_ALIGNMENT_LIST` rows, per-frame `MOVIE_ALIGNMENT_PARAMETERS_<id>` tables and image assets when the job finishes. A job that was running when the server stopped is **not** failed on restart any more: its controller reconnects within the reconnect window (10 minutes) and carries on.

Run profiles carry their commands (`RUN_PROFILE_COMMANDS_<id>`, cisTEM's own tables), seeded like cisTEM's defaults, and are edited on the Settings tab the way cisTEM's Run Profiles panel does it; the seeded Slurm profile has no commands and is refused at submit until you give it some there.

Tests: `python -m unittest discover -s server/tests` — codec tests plus runner integration tests that launch the fake controller for real.

## Security

Every request is authenticated (see "Users and access control" above) and projects are only visible to their owner or an admin. What's still explicitly *not* built, matching this app's "trusted lab network" scope: no rate limiting or lockout on login attempts, no HTTPS enforcement (bearer tokens over plain HTTP are sniffable on a hostile network — this only matters if the app leaves a trusted network), no password reset flow (a forgotten password needs an admin to recreate the account), and no user delete/edit endpoints (only create and list). If any of that matters for your deployment, put a reverse proxy in front that handles TLS and rate limiting, and treat the missing pieces above as the next things to build.

## Working on this in Claude Code

See `CLAUDE.md` for the project's architecture, design conventions, and known next steps — Claude Code reads it automatically at the start of a session.
