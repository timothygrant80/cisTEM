"""What a job's bookkeeping between steps has got to, for the Jobs tab.

A multi-run job (2D classification, ab-initio, the refinements) does work of
its own between its children -- reading a round's output star files, writing
the round's results to the project database, aligning symmetry -- during
which the job is "running" with no child to draw a bar from. A single-run
job's result write is the same kind of gap. Drivers and adapters note where
they are here, keyed by the job id the Jobs tab shows (the parent's), and the
job list carries it as `finishing`: {done, total, what, unit}. `total` may be
0 for a phase with nothing to count -- the page then shows the words only.

In memory only: a server restart loses it, and the page falls back to the
job's own counts, which is right -- the thread doing the work is gone too.
"""
import threading

_notes = {}
_lock = threading.Lock()


def note(job_id, done, total, what, unit=""):
    with _lock:
        _notes[job_id] = {"done": int(done or 0), "total": int(total or 0), "what": what, "unit": unit}


def get(job_id):
    with _lock:
        return _notes.get(job_id)


def clear(job_id):
    with _lock:
        _notes.pop(job_id, None)
