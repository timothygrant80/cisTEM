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

from . import ctffind, find_particles, merge2d, refine2d, unblur

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
}
