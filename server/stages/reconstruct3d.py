"""`reconstruct3d` as a step of an ab-initio 3D reconstruction.

The reconstruction leg of a 3D round: reconstruct3d on a particle range, dumping its half-set arrays for merge3d (AbInitioManager::SetupReconstructionJob, format ttttttttiiffffffffffbbbbbbbbbbttii).
Not a stage of its own: server/abinitio.py submits these as hidden child
jobs (STAGE "abinitio_reconstruct3d") of an ab_initio_3d job; this module is what the
runner needs of an adapter -- the program, and a finalize() with nothing to
write, since the driver reads what it needs from the files the tasks leave.
The argument list is built in abinitio.py next to the values.
"""

PROGRAM = {"name": "reconstruct3d", "executable": "reconstruct3d"}

ARGUMENT_TYPES = "ttttttttiiffffffffffbbbbbbbbbbttii"
ARGUMENT_NAMES = (
    "input_particle_stack",
    "input_star_filename",
    "input_reconstruction",
    "output_reconstruction_1",
    "output_reconstruction_2",
    "output_reconstruction_filtered",
    "output_resolution_statistics",
    "my_symmetry",
    "first_particle",
    "last_particle",
    "pixel_size",
    "molecular_mass_kDa",
    "inner_mask_radius",
    "outer_mask_radius",
    "resolution_limit_rec",
    "resolution_limit_ref",
    "score_weight_conversion",
    "score_threshold",
    "smoothing_factor",
    "padding",
    "normalize_particles",
    "adjust_scores",
    "invert_contrast",
    "exclude_blank_edges",
    "crop_images",
    "split_even_odd",
    "centre_mass",
    "use_input_reconstruction",
    "threshold_input_3d",
    "dump_arrays",
    "dump_file_1",
    "dump_file_2",
    "correct_ewald_sphere",
    "max_threads",
)
assert len(ARGUMENT_NAMES) == len(ARGUMENT_TYPES)


def finalize(conn, project_id, job, sent_tasks, task_rows, log):
    return {"reconstruct3d_tasks_ok": sum(1 for r in task_rows if r["STATUS"] == "ok")}


def describe_summary(summary):
    n = summary.get("reconstruct3d_tasks_ok", 0)
    return "{} reconstruct3d task{} finished; the ab-initio driver takes it from here".format(n, "" if n == 1 else "s")
