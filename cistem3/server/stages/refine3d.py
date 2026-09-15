"""`refine3d` as a step of an ab-initio 3D reconstruction.

One refinement round of a 3D refinement: refine3d on a particle range against the current reference (AbInitioManager::SetupRefinementJob, format ttttbttttiiffffffffffffifffffffffbbbbbbbbbbbbbbbibibb).
Not a stage of its own: server/abinitio.py submits these as hidden child
jobs (STAGE "abinitio_refine3d") of an ab_initio_3d job; this module is what the
runner needs of an adapter -- the program, and a finalize() with nothing to
write, since the driver reads what it needs from the files the tasks leave.
The argument list is built in abinitio.py next to the values.
"""

PROGRAM = {"name": "refine3d", "executable": "refine3d"}

ARGUMENT_TYPES = "ttttbttttiiffffffffffffifffffffffbbbbbbbbbbbbbbbibibb"
ARGUMENT_NAMES = (
    "input_particle_images",
    "input_star_filename",
    "input_reconstruction",
    "input_reconstruction_statistics",
    "use_statistics",
    "output_matching_projections",
    "output_star_filename",
    "output_shift_filename",
    "my_symmetry",
    "first_particle",
    "last_particle",
    "percent_used",
    "pixel_size",
    "molecular_mass_kDa",
    "inner_mask_radius",
    "outer_mask_radius",
    "low_resolution_limit",
    "high_resolution_limit",
    "signed_CC_limit",
    "classification_resolution_limit",
    "mask_radius_search",
    "high_resolution_limit_search",
    "angular_step",
    "best_parameters_to_keep",
    "max_search_x",
    "max_search_y",
    "mask_center_2d_x",
    "mask_center_2d_y",
    "mask_center_2d_z",
    "mask_radius_2d",
    "defocus_search_range",
    "defocus_step",
    "padding",
    "global_search",
    "local_refinement",
    "refine_psi",
    "refine_theta",
    "refine_phi",
    "refine_x_shift",
    "refine_y_shift",
    "calculate_matching_projections",
    "apply_2d_masking",
    "ctf_refinement",
    "normalize_particles",
    "invert_contrast",
    "exclude_blank_edges",
    "normalize_input_3d",
    "threshold_input_3d",
    "max_threads",
    "local_global_refine",
    "class_number",
    "ignore_input_parameters",
    "defocus_bias",
)
assert len(ARGUMENT_NAMES) == len(ARGUMENT_TYPES)


def finalize(conn, project_id, job, sent_tasks, task_rows, log):
    return {"refine3d_tasks_ok": sum(1 for r in task_rows if r["STATUS"] == "ok")}


def describe_summary(summary):
    n = summary.get("refine3d_tasks_ok", 0)
    return "{} refine3d task{} finished; the ab-initio driver takes it from here".format(n, "" if n == 1 else "s")
