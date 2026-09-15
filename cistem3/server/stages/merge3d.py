"""`merge3d` as a step of an ab-initio 3D reconstruction.

The merge leg of a 3D round: merge3d adds the reconstruct3d dumps into the filtered reconstruction and its resolution statistics (AbInitioManager::SetupMerge3dJob, format ttttfffttibtiff).
Not a stage of its own: server/abinitio.py submits these as hidden child
jobs (STAGE "abinitio_merge3d") of an ab_initio_3d job; this module is what the
runner needs of an adapter -- the program, and a finalize() with nothing to
write, since the driver reads what it needs from the files the tasks leave.
The argument list is built in abinitio.py next to the values.
"""

PROGRAM = {"name": "merge3d", "executable": "merge3d"}

ARGUMENT_TYPES = "ttttfffttibtiff"
ARGUMENT_NAMES = (
    "output_reconstruction_1",
    "output_reconstruction_2",
    "output_reconstruction_filtered",
    "output_resolution_statistics",
    "molecular_mass_kDa",
    "inner_mask_radius",
    "outer_mask_radius",
    "dump_file_seed_1",
    "dump_file_seed_2",
    "class_number",
    "save_orthogonal_views_image",
    "orthogonal_views_filename",
    "number_of_dump_files",
    "wiener_nominator",
    "alignment_res",
)
assert len(ARGUMENT_NAMES) == len(ARGUMENT_TYPES)


def finalize(conn, project_id, job, sent_tasks, task_rows, log):
    return {"merge3d_tasks_ok": sum(1 for r in task_rows if r["STATUS"] == "ok")}


def describe_summary(summary):
    n = summary.get("merge3d_tasks_ok", 0)
    return "{} merge3d task{} finished; the ab-initio driver takes it from here".format(n, "" if n == 1 else "s")
