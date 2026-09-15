"""Per-stage adapters between this server's job model and a cisTEM program.

An adapter knows two things about its program that nothing else in the server
does: how to turn a job's parameters and the project's assets into the
positional argument list the program's DoCalculation() reads
(`build_tasks`), and what its results mean once they come back
(`finalize`). Everything in between -- launching, the wire protocol,
progress, reconnection -- is generic and lives in job_runner.py.

Keys are the API's stage names (the same ones STAGE_COMMANDS uses); a stage
with no adapter here still runs, in simulation, exactly as before.
"""

from . import (ctffind, estimate_beamtilt, find_particles, merge2d, merge3d, prepare_stack, prepare_stack_classaverage, reconstruct3d, refine2d,
               refine3d, refine_ctf, unblur)

ADAPTERS = {
    "motion_correction": unblur,
    "ctf_estimation": ctffind,
    "particle_picking": find_particles,
    # The two legs of a 2D classification round. Not stages a user submits:
    # server/classification.py drives them as hidden child jobs of a class2d
    # job (their JOBS rows carry PARENT_JOB_ID), and they are here so the
    # runner can launch and, after a restart, restore them like any other.
    "class2d_refine2d": refine2d,
    "class2d_merge2d": merge2d,
    # The legs of an ab-initio 3D round, driven by server/abinitio.py the same way.
    "abinitio_prepare_stack": prepare_stack,
    "abinitio_prepare_stack_classaverage": prepare_stack_classaverage,
    "abinitio_refine3d": refine3d,
    "abinitio_reconstruct3d": reconstruct3d,
    "abinitio_merge3d": merge3d,
    # The legs of a CTF refinement (server/refinectf.py): refine_ctf streams
    # its per-particle results, which the sink hands to the adapter's
    # on_task_progress(); estimate_beamtilt searches the phase-difference image.
    "refinectf_refine_ctf": refine_ctf,
    "refinectf_estimate_beamtilt": estimate_beamtilt,
}
