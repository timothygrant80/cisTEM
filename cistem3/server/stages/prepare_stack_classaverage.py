"""`prepare_stack_classaverage` as the first step of an ab-initio 3D
reconstruction from 2D class averages.

For each selected class of a classification it builds `number_of_classes`
CTF-corrected averages of `images_per_class` randomly drawn members --
whitened, aligned by the classification's angles and shifts, resampled to
the final resolution's pixel size -- and those averages, not the particles,
are what the ab-initio cycle then refines (AbInitioManager::
SetupPrepareStackJob(), class-average branch, format ttttffbiiibbii).
Submitted by server/abinitio.py as a hidden child job (STAGE
"abinitio_prepare_stack_classaverage"); the argument list is built there.
"""

PROGRAM = {"name": "prepare_stack_classaverage", "executable": "prepare_stack_classaverage"}

ARGUMENT_TYPES = "ttttffbiiibbii"
ARGUMENT_NAMES = (
    "input_particle_images",
    "output_classaverage_images",
    "input_star_filename",
    "input_selection_file",
    "output_pixel_size",
    "mask_radius",
    "resample_box",
    "wanted_output_box_size",
    "number_of_classes",
    "images_per_class",
    "invert_contrast",
    "process_a_subset",
    "first_classaverage",
    "last_classaverage",
)
assert len(ARGUMENT_NAMES) == len(ARGUMENT_TYPES)


def finalize(conn, project_id, job, sent_tasks, task_rows, log):
    return {"prepare_stack_classaverage_tasks_ok": sum(1 for r in task_rows if r["STATUS"] == "ok")}


def describe_summary(summary):
    n = summary.get("prepare_stack_classaverage_tasks_ok", 0)
    return "{} class-average preparation task{} finished; the ab-initio driver takes it from here".format(n, "" if n == 1 else "s")
