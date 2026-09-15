"""`refine2d`, as one leg of a 2D classification.

Not a stage of its own: a class2d job (server/classification.py) submits a
refine2d run per round -- one task per particle range, each dumping its
class sums for merge2d -- as a hidden child job whose STAGE is
"class2d_refine2d". This module is what the runner needs of an adapter for
such a child: the program to launch and a finalize() that has nothing to
write (the driver reads each task's output star file when the round is
merged). The 25-argument task list itself is built in classification.py,
next to the logic that decides the values, mirroring
ClassificationManager::RunRefinementJobPostStarFileWrite()'s AddJob call
(format "tttttiiiffffffffibbbbtbbi").
"""

PROGRAM = {"name": "refine2d", "executable": "refine2d"}

# refine2d's DoCalculation() argument order -- the contract with the binary.
ARGUMENT_NAMES = (
    "input_particle_images", "input_star_filename", "input_class_averages", "output_star_filename",
    "output_class_averages", "number_of_classes", "first_particle", "last_particle", "percent_used",
    "pixel_size", "mask_radius", "low_resolution_limit", "high_resolution_limit", "angular_step",
    "max_search_range", "smoothing_factor", "padding_factor", "normalize_particles", "invert_contrast",
    "exclude_blank_edges", "dump_arrays", "dump_file", "auto_mask", "auto_centre", "max_threads",
)
ARGUMENT_TYPES = "tttttiiiffffffffibbbbtbbi"


def finalize(conn, project_id, job, sent_tasks, task_rows, log):
    """Nothing to write here: the classification driver collects the round's
    per-particle results from the output star files once merge2d has run."""
    return {"refine2d_tasks_ok": sum(1 for r in task_rows if r["STATUS"] == "ok")}


def describe_summary(summary):
    n = summary.get("refine2d_tasks_ok", 0)
    return "{} refine2d task{} finished; the classification driver merges them next".format(n, "" if n == 1 else "s")
