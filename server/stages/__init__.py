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

from . import unblur

ADAPTERS = {
    "motion_correction": unblur,
}
