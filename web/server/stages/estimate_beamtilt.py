"""`estimate_beamtilt` as a step of a CTF refinement.

The beam-tilt search over the phase-difference image refine_ctf's master
wrote, split into position ranges of FindBeamTilt()'s grid
(CTFRefinementManager::RunBeamTiltEstimationJob, format tfffii). Each task
returns five floats -- score, beam tilt x and y (radians), particle shift x
and y (Angstroms) -- and the driver keeps the lowest score. Hidden child
jobs of a refine_ctf job (STAGE "refinectf_estimate_beamtilt").
"""

PROGRAM = {"name": "estimate_beamtilt", "executable": "estimate_beamtilt"}

ARGUMENT_TYPES = "tfffii"
ARGUMENT_NAMES = (
    "input_phase_difference_image",
    "pixel_size",
    "voltage_kV",
    "spherical_aberration_mm",
    "first_position_to_search",
    "last_position_to_search",
)
assert len(ARGUMENT_NAMES) == len(ARGUMENT_TYPES)

TOTAL_POSITIONS = 290880  # FindBeamTilt()'s grid, as RunBeamTiltEstimationJob() hard-codes it


def finalize(conn, project_id, job, sent_tasks, task_rows, log):
    return {"estimate_beamtilt_tasks_ok": sum(1 for r in task_rows if r["STATUS"] == "ok")}


def describe_summary(summary):
    n = summary.get("estimate_beamtilt_tasks_ok", 0)
    return "{} beam-tilt search task{} finished; the CTF refinement driver takes it from here".format(n, "" if n == 1 else "s")
