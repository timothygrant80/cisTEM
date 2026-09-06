"""`prepare_stack` as a step of an ab-initio 3D reconstruction.

Preparing the particle stack for ab-initio 3D: whitened, resampled to the final resolution's pixel size (AbInitioManager::SetupPrepareStackJob, format tttffbibii).
Not a stage of its own: server/abinitio.py submits these as hidden child
jobs (STAGE "abinitio_prepare_stack") of an ab_initio_3d job; this module is what the
runner needs of an adapter -- the program, and a finalize() with nothing to
write, since the driver reads what it needs from the files the tasks leave.
The argument list is built in abinitio.py next to the values.
"""

PROGRAM = {"name": "prepare_stack", "executable": "prepare_stack"}

ARGUMENT_TYPES = "tttffbibii"
ARGUMENT_NAMES = (
    "input_particle_images",
    "input_star_file",
    "output_particle_images",
    "pixel_size",
    "mask_radius",
    "resample_box",
    "wanted_output_box_size",
    "process_a_subset",
    "first_particle",
    "last_particle",
)
assert len(ARGUMENT_NAMES) == len(ARGUMENT_TYPES)


def finalize(conn, project_id, job, sent_tasks, task_rows, log):
    return {"prepare_stack_tasks_ok": sum(1 for r in task_rows if r["STATUS"] == "ok")}


def describe_summary(summary):
    n = summary.get("prepare_stack_tasks_ok", 0)
    return "{} prepare_stack task{} finished; the ab-initio driver takes it from here".format(n, "" if n == 1 else "s")
