"""`refine_ctf` as a step of a CTF refinement.

One task refines the defocus of a particle range (and sums the phase
differences for beam-tilt estimation) against the reference named per
particle in the star file (CTFRefinementManager::SetupRefinementJob, format
ttttbtttttiifffffffffbbbbbbbiii). Not a stage of its own: server/refinectf.py
submits these as hidden child jobs (STAGE "refinectf_refine_ctf").

Unlike refine3d, refine_ctf writes no output star file when it runs under a
controller: the refined defocus of each particle comes back as an
intermediate result (`task_progress`: position, defocus 1, defocus 2, logP,
score) and the phase-difference sums go to the *master*, which writes their
normalised sum to the phase-difference image path (argument 7) when the last
worker reports. So this adapter has the one thing the others lack: an
`on_task_progress()` hook, which appends each particle's result to the file
named by the task's output-star argument (unused by the program in that
mode), for the driver to read when the child finishes.
"""
import os

PROGRAM = {"name": "refine_ctf", "executable": "refine_ctf"}

ARGUMENT_TYPES = "ttttbtttttiifffffffffbbbbbbbiii"
ARGUMENT_NAMES = (
    "input_particle_images",
    "input_star_filename",
    "input_reconstruction",
    "input_reconstruction_statistics",
    "use_statistics",
    "output_star_filename",
    "output_shift_filename",
    "output_phase_difference_image",
    "output_beamtilt_image",
    "output_difference_image",
    "first_particle",
    "last_particle",
    "pixel_size",
    "molecular_mass_kDa",
    "inner_mask_radius",
    "outer_mask_radius",
    "low_resolution_limit",
    "high_resolution_limit",
    "defocus_search_range",
    "defocus_step",
    "padding",
    "ctf_refinement",
    "beamtilt_refinement",
    "normalize_particles",
    "invert_contrast",
    "exclude_blank_edges",
    "normalize_input_3d",
    "threshold_input_3d",
    "image_number_for_gui",
    "number_of_jobs_per_image_in_gui",
    "max_threads",
)
assert len(ARGUMENT_NAMES) == len(ARGUMENT_TYPES)

PROGRESS_ARGUMENT = ARGUMENT_NAMES.index("output_star_filename")

# package.forward_progress: the intermediate results *are* the results here.
WANTS_TASK_PROGRESS = True


def progress_path(sent_task):
    """Where a task's per-particle results are appended: the output-star
    argument, which the program leaves unwritten under a controller."""
    try:
        return sent_task["args"][PROGRESS_ARGUMENT]["value"]
    except (KeyError, IndexError, TypeError):
        return None


def on_task_progress(project_id, sent_task, result):
    """DbSink hook: one intermediate result (position_in_stack, defocus_1,
    defocus_2, logp, score) appended as a line."""
    path = progress_path(sent_task)
    data = (result or {}).get("data") if isinstance(result, dict) else None
    if not path or not data or len(data) < 5:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        fh.write("{:d} {:.4f} {:.4f} {:.6f} {:.6f}\n".format(int(round(data[0])), data[1], data[2], data[3], data[4]))


def read_progress(path):
    """The lines on_task_progress() wrote -> {position: (defocus_1, defocus_2, logp, score)}."""
    out = {}
    if not path or not os.path.isfile(path):
        return out
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                out[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
            except ValueError:
                continue
    return out


def finalize(conn, project_id, job, sent_tasks, task_rows, log):
    return {"refine_ctf_tasks_ok": sum(1 for r in task_rows if r["STATUS"] == "ok")}


def describe_summary(summary):
    n = summary.get("refine_ctf_tasks_ok", 0)
    return "{} refine_ctf task{} finished; the CTF refinement driver takes it from here".format(n, "" if n == 1 else "s")
