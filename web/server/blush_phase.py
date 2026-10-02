"""The two Blush phases a 3D driver runs, shared by Refine 3D, Auto Refine and
Ab-Initio 3D (see blush.py for the denoising itself):

  * `PRE` ("blush"), before a round: every class refines against the Blush
    companion of its reference. One that exists and is newer than the
    reference (written at the end of the round or run that made it, from the
    half maps) is used as it is; one that does not is made now from the
    filtered reference, which is all there is for a volume made without Blush.
  * `POST` ("blush_post"), after a round's merge: the new volume's companion
    is computed from the two half maps the merge kept (their average is the
    unfiltered reconstruction, RELION's input), cut at the round's FSC, and
    written where the next round or run finds it; the half maps are removed.
    A driver whose input choice is the filtered volume makes the companion
    from that instead and keeps no half maps.

Each phase runs on a thread of its own (minutes on a CPU), is recorded in
the job's state so resume() restarts it after a server restart, and reports
per-block progress through progress_store. The driver describes itself to
this module with a Host: the few functions every driver has under the same
names plus what differs between them (pixel size, mask radius, where a
reference's companion lives, which statistics give the round's FSC). The
machine's batch, process and thread settings come from blush.runtime_settings().
"""

import os
import threading
from pathlib import Path

import blush
import db
import progress_store

PRE = "blush"
POST = "blush_post"
PHASES = (PRE, POST)

_cancel = {}


class Host:
    """What a driver tells this module. `module` is the driver module itself,
    with `_job_lock`, `_load_state`, `_save`, `_finish`, `_parent_row`,
    `_log`, `_progress_percent`; `continue_pre` runs after the PRE phase
    (masking, then the refinement), `continue_post` after the POST phase (the
    cycle); the rest read the state."""

    def __init__(self, module, continue_pre, continue_post, pixel_size, mask_radius, statistics, companion_path, half_maps, unfiltered):
        self.module = module
        self.continue_pre = continue_pre
        self.continue_post = continue_post
        self.pixel_size = pixel_size
        self.mask_radius = mask_radius
        self.statistics = statistics
        self.companion_path = companion_path
        self.half_maps = half_maps
        self.unfiltered = unfiltered


def assets_companion(project_id, reference_file):
    """Where a volume asset's companion lives: Assets/Volumes/Blushed/<volume name>_blushed.mrc."""
    name = os.path.basename(str(reference_file))
    stem = name[:-4] if name.lower().endswith(".mrc") else name
    return str(Path(db.project_dir(project_id)) / "Assets" / "Volumes" / "Blushed" / (stem + "_blushed.mrc"))


def sibling_companion(reference_file):
    """A companion beside a scratch volume: <path>_blushed.mrc (ab-initio's per-round maps)."""
    path = str(reference_file)
    stem = path[:-4] if path.lower().endswith(".mrc") else path
    return stem + "_blushed.mrc"


def companion_is_current(companion, reference):
    """A companion counts only when it exists and is at least as new as the
    reference: a volume rewritten under the same name (ab-initio's rounds)
    must not pick up the previous round's companion."""
    try:
        return os.path.isfile(companion) and os.path.isfile(reference) and os.path.getmtime(companion) >= os.path.getmtime(reference) - 1.0
    except OSError:
        return False


def start(host, conn, project_id, job_id, state, phase):
    """Record the phase in the state and run it on a thread."""
    state.update({"phase": phase, "child_job_id": None, "child_task_count": 0, "child_done": 0})
    host.module._save(conn, job_id, state)
    _launch(host, project_id, job_id, phase)


def resume(host, project_id, job_id, phase):
    host.module._log(project_id, job_id, "server restarted during Blush; starting it again")
    _launch(host, project_id, job_id, phase)


def cancel(job_id):
    """Stop a running phase; True when one was running. The thread finishes the job as cancelled."""
    ev = _cancel.get(job_id)
    if ev is None:
        return False
    ev.set()
    return True


def _launch(host, project_id, job_id, phase):
    _cancel[job_id] = threading.Event()
    threading.Thread(target=_worker, args=(host, project_id, job_id, phase), daemon=True, name="blush-" + job_id).start()


def _worker(host, project_id, job_id, phase):
    m = host.module
    conn = db.get_conn(project_id)
    try:
        with m._job_lock(job_id):
            state = m._load_state(conn, job_id)
            parent = m._parent_row(conn, job_id)
            if not state or parent is None or state.get("phase") != phase or parent["STATUS"] not in ("queued", "running"):
                return
        classes = state["number_of_classes"]
        info = blush.availability()
        rt = blush.runtime_settings()
        layout = info["device"] if info["device"] != "cpu" else "{} process{} x {} threads".format(
            rt["processes"], "es" if rt["processes"] != 1 else "", rt["threads"] if rt["threads"] else "default")
        cancel_event = _cancel.get(job_id) or threading.Event()
        references = []
        for k in range(classes):
            ref = state["reference_files"][k]
            out = host.companion_path(project_id, state, ref)
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            label = "Blush: class {} of {}".format(k + 1, classes) if classes > 1 else "Blush"
            if phase == PRE:
                if companion_is_current(out, ref):
                    m._log(project_id, job_id, "Blush: class {} refines against {} (made when that volume was reconstructed)".format(k + 1, os.path.basename(out)))
                    references.append(out)
                    continue
                paths, filtered, note = [ref], True, "the filtered reference (no Blush companion for this volume yet)"
            else:
                halves = host.half_maps(state, k)
                if host.unfiltered(state) and halves and all(os.path.isfile(h) for h in halves):
                    paths, filtered, note = halves, False, "the half maps (unfiltered), cut at the FSC"
                else:
                    paths, filtered, note = [ref], True, "the filtered volume"

            def progress(done, total, _label=label):
                progress_store.note(job_id, done, total, _label, "blocks")
                return not cancel_event.is_set()

            progress_store.note(job_id, 0, 0, label + ": preparing ({})".format(layout))
            blush.denoise_file(paths, out, host.pixel_size(state), host.mask_radius(state), fsc_stats=host.statistics(conn, state, k), input_is_filtered=filtered,
                               batch_size=rt["batch_size"], threads=rt["threads"], processes=rt["processes"], progress=progress)
            m._log(project_id, job_id, "Blush: class {} denoised from {} -> {} ({}, batch {})".format(k + 1, note, os.path.basename(out), layout, rt["batch_size"]))
            references.append(out)
        if phase == POST:
            for k in range(classes):
                for h in host.half_maps(state, k) or []:
                    try:
                        os.remove(h)
                    except OSError:
                        pass
        with m._job_lock(job_id):
            state = m._load_state(conn, job_id)
            parent = m._parent_row(conn, job_id)
            if not state or parent is None or state.get("phase") != phase or parent["STATUS"] not in ("queued", "running"):
                return
            if parent["CANCEL_REQUESTED"] or cancel_event.is_set():
                m._finish(conn, project_id, job_id, state, "cancelled", "cancelled during Blush")
                return
            if phase == PRE:
                state["reference_files"] = references
                host.continue_pre(conn, project_id, job_id, state)
            else:
                state["pending_half_maps"] = []
                host.continue_post(conn, project_id, job_id, state)
            if state.get("phase") != "finished":
                m._save(conn, job_id, state, m._progress_percent(state))
    except blush.BlushCancelled:
        with m._job_lock(job_id):
            state = m._load_state(conn, job_id)
            if state and state.get("phase") == phase:
                m._finish(conn, project_id, job_id, state, "cancelled", "cancelled during Blush")
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).exception("Blush failed for job %s", job_id)
        with m._job_lock(job_id):
            state = m._load_state(conn, job_id)
            if state and state.get("phase") == phase:
                m._finish(conn, project_id, job_id, state, "failed", "Blush failed: {}".format(exc))
    finally:
        _cancel.pop(job_id, None)
        progress_store.clear(job_id)
        conn.close()
