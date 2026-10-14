"""Refine 3D writes every round's refinement but the last to the database on a thread while the next refine3d
runs: the next round reads only the scratch tables and the statistics file, the refinement appears whole once
the thread is done (its REFINEMENT_LIST row last), the history and the pending record are updated in place, a
server restart writes an interrupted record again, and a failed background write is redone in the foreground
before the next round's refinement."""
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import abinitio  # noqa: E402
import db  # noqa: E402
import package_io  # noqa: E402
import refine3d  # noqa: E402
import refinement_packages as rp  # noqa: E402
import refinements  # noqa: E402
import starfile  # noqa: E402
import volumes  # noqa: E402
from stages import refine3d as refine3d_adapter  # noqa: E402

N, BOX = 12, 16


class FakeRuntime:
    def __init__(self):
        self.calls = []
        self.log = []

    def submit_child(self, project_id, child_id, adapter, tasks, profile):
        self.calls.append({"child": child_id, "adapter": adapter, "tasks": tasks, "profile": profile})

    def append_log(self, project_id, job_id, text, level="info"):
        self.log.append(text)


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self._old = db.PROJECTS_ROOT, db.SYSTEM_DB_PATH, refine3d._runtime, refinements.add_refinement
        db.PROJECTS_ROOT = root / "projects"; db.PROJECTS_ROOT.mkdir()
        db.SYSTEM_DB_PATH = root / "system.db"
        self.runtime = FakeRuntime(); refine3d.configure(self.runtime)
        proj = db.create_project("P", 1, "u"); self.pid = proj["id"] if isinstance(proj, dict) else proj
        stack = str(root / "stack.mrc"); w = rp.MrcStackWriter(stack, BOX, 1.0)
        for i in range(N):
            w.append(np.random.default_rng(i).normal(size=(BOX, BOX)).astype(np.float32))
        w.close()
        star = str(root / "p.star")
        starfile.write_star(star, [{"position_in_stack": i + 1, "image_is_active": 1, "psi": 10.0 * i, "theta": 20.0, "phi": 30.0, "defocus_1": 15000.0, "defocus_2": 14800.0,
                                    "occupancy": 100.0, "sigma": 1.0, "score": 10.0, "pixel_size": 1.0, "voltage": 300.0, "cs": 2.7, "amplitude_contrast": 0.07, "assigned_subset": 1 + i % 2} for i in range(N)])
        conn = db.get_conn(self.pid)
        out = package_io.import_package(conn, self.pid, {"format": "cistem", "stack_path": stack, "metadata_path": star, "name": "S", "symmetry": "D2",
                                                          "molecular_weight_kda": 100, "largest_dimension_a": 12, "protein_is_white": False, "cs_mm": 2.7})
        self.pkg = out["refinement_package_asset_id"]
        self.rid = conn.execute("SELECT MAX(REFINEMENT_ID) FROM REFINEMENT_LIST WHERE REFINEMENT_PACKAGE_ASSET_ID=?", (self.pkg,)).fetchone()[0]
        self.vol = str(root / "ref.mrc"); volumes.write_mrc_volume(self.vol, np.zeros((BOX, BOX, BOX), np.float32), 1.0)
        vid = volumes.add_volume_asset(conn, "ref", self.vol, 1.0, BOX, BOX, BOX)
        refinements.set_current_reference(conn, self.pkg, 1, vid)
        self.job = "job12345678"
        with conn:
            conn.execute("INSERT INTO JOBS(JOB_ID, STAGE, JOB_NUMBER, NAME, PARAMS_JSON, STATUS, PROGRESS, CREATED_AT) VALUES (?, 'refine3d', 1, 'Job 1', '{}', 'queued', 0, 'now')", (self.job,))
        self.conn = conn
        sys_conn = db.get_system_conn()
        try:
            self.profile = db.load_run_profile_by_name(sys_conn, "Local (single-threaded)")
        finally:
            sys_conn.close()

    def tearDown(self):
        refine3d._wait_for_record(self.job)
        self.conn.close()
        db.PROJECTS_ROOT, db.SYSTEM_DB_PATH, refine3d._runtime, refinements.add_refinement = self._old
        self.tmp.cleanup()

    def _start(self, rounds):
        params = {"refinement_package_id": self.pkg, "input_refinement_id": self.rid, "run_profile": self.profile["name"], "number_of_rounds": rounds,
                  "refinement_type": "Local Refinement", "auto_mask": False, "use_mask": False, "use_blush": False, "percent_used": 100.0}
        return refine3d.start(self.conn, self.pid, self.job, params, self.profile)

    def _merged(self, state):
        """The state as the merge child leaves it: this round's volume and statistics written, as merge3d would."""
        rid = state["output_refinement_id"]
        vol = str(volumes.volume_dir(self.pid) / "volume_{}_1.mrc".format(rid))
        volumes.write_mrc_volume(vol, np.random.default_rng(rid).normal(size=(BOX, BOX, BOX)).astype(np.float32), 1.0)
        stats_path = str(Path(state["scratch"]) / "volume_stats_{}_1.txt".format(rid))
        stats = [{"shell": i, "resolution": (BOX / i if i else 0.0), "fsc": 1.0 - i / 10.0, "part_fsc": 1.0 - i / 10.0, "part_ssnr": 4.0, "rec_ssnr": 2.0} for i in range(BOX // 2 + 1)]
        abinitio.write_statistics(stats_path, stats, 1.0)
        state.update({"phase": "merge", "child_job_id": "merge-child", "pending_volume_files": [vol], "pending_stats_files": [stats_path], "pending_half_maps": [],
                      "number_of_dump_files": 1})
        return state

    def _rows(self, rid):
        return self.conn.execute("SELECT COUNT(*) FROM REFINEMENT_RESULT_{}_1".format(rid)).fetchone()[0]

    def test_a_round_that_is_not_the_last_is_written_while_the_next_runs(self):
        state = self._merged(self._start(rounds=2))
        rid = state["output_refinement_id"]
        gate = threading.Event()
        real = refinements.add_refinement

        def slow(*args, **kwargs):
            gate.wait(10)
            return real(*args, **kwargs)
        refinements.add_refinement = slow
        refine3d._advance(self.conn, self.pid, self.job, state)
        # the next round's refine3d is launched, from the scratch tables and the statistics file, with the refinement not yet in the database
        self.assertEqual(self.runtime.calls[-1]["adapter"], refine3d_adapter)
        self.assertEqual(state["round"], 1); self.assertEqual(state["input_refinement_id"], rid)
        self.assertIsNone(self.conn.execute("SELECT 1 FROM REFINEMENT_LIST WHERE REFINEMENT_ID=?", (rid,)).fetchone())
        saved = json.loads(self.conn.execute("SELECT STATE_JSON FROM JOBS WHERE JOB_ID=?", (self.job,)).fetchone()["STATE_JSON"])
        self.assertEqual(saved["pending_record"]["refinement_id"], rid); self.assertEqual(saved["history"], [])
        self.assertTrue(all(os.path.isfile(f) for f in saved["pending_record"]["files"]))
        with open(saved["input_stats_file"]) as fh:
            self.assertEqual(json.load(fh)["refinement_id"], rid)
        self.assertFalse(any("Refinement #{} written".format(rid) in t for t in self.runtime.log))
        stats_line = starfile.read_params(str(Path(state["scratch"]) / "input_par_{}_1.cistem".format(rid)), as_table=True)
        self.assertEqual(len(stats_line), N)
        gate.set()
        refine3d._wait_for_record(self.job)
        # now the refinement is whole: rows, statistics, D2 angular distribution, list row; the state updated in place
        self.assertEqual(self._rows(rid), N)
        self.assertEqual(self.conn.execute("SELECT NUMBER_OF_CLASSES FROM REFINEMENT_LIST WHERE REFINEMENT_ID=?", (rid,)).fetchone()[0], 1)
        self.assertAlmostEqual(sum(refinements.load_angular_distribution(self.conn, rid, 1)), 4 * N)
        self.assertTrue(refinements.load_statistics(self.conn, rid, 1))
        saved = json.loads(self.conn.execute("SELECT STATE_JSON FROM JOBS WHERE JOB_ID=?", (self.job,)).fetchone()["STATE_JSON"])
        self.assertNotIn("pending_record", saved)
        self.assertEqual([h["refinement_id"] for h in saved["history"]], [rid])
        self.assertEqual(saved["round"], 1)   # the in-place update kept the step's own save
        self.assertFalse(any(os.path.isfile(f) for f in state["pending_record"]["files"]))
        self.assertTrue(any("Refinement #{} written".format(rid) in t for t in self.runtime.log))

    def test_the_last_round_is_written_before_the_job_finishes(self):
        state = self._merged(self._start(rounds=1))
        rid = state["output_refinement_id"]
        refine3d._advance(self.conn, self.pid, self.job, state)
        self.assertEqual(state["phase"], "finished"); self.assertNotIn("pending_record", state)
        self.assertEqual(self._rows(rid), N)
        row = self.conn.execute("SELECT STATUS, METRICS_JSON FROM JOBS WHERE JOB_ID=?", (self.job,)).fetchone()
        self.assertEqual(row["STATUS"], "completed"); self.assertEqual(json.loads(row["METRICS_JSON"])["refinement_ids"], [rid])

    def test_a_restart_writes_an_interrupted_record_again(self):
        state = self._merged(self._start(rounds=2))
        rid = state["output_refinement_id"]
        refinements.add_refinement = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("server died"))
        refine3d._advance(self.conn, self.pid, self.job, state)
        refine3d._wait_for_record(self.job)
        self.assertIsNone(self.conn.execute("SELECT 1 FROM REFINEMENT_LIST WHERE REFINEMENT_ID=?", (rid,)).fetchone())
        refinements.add_refinement = self._old[3]
        refine3d.resume(self.pid, self.conn.execute("SELECT * FROM JOBS WHERE JOB_ID=?", (self.job,)).fetchone())
        refine3d._wait_for_record(self.job)
        self.assertEqual(self._rows(rid), N)
        saved = json.loads(self.conn.execute("SELECT STATE_JSON FROM JOBS WHERE JOB_ID=?", (self.job,)).fetchone()["STATE_JSON"])
        self.assertNotIn("pending_record", saved); self.assertEqual([h["refinement_id"] for h in saved["history"]], [rid])

    def test_a_failed_background_write_is_redone_before_the_next_record(self):
        state = self._merged(self._start(rounds=3))
        rid1 = state["output_refinement_id"]
        refinements.add_refinement = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full"))
        refine3d._advance(self.conn, self.pid, self.job, state)
        refine3d._wait_for_record(self.job)
        self.assertTrue(any("could not write Refinement #{}".format(rid1) in t for t in self.runtime.log))
        refinements.add_refinement = self._old[3]
        state = refine3d._load_state(self.conn, self.job)
        self._merged(state)
        rid2 = state["output_refinement_id"]
        refine3d._advance(self.conn, self.pid, self.job, state)    # round 2's record: round 1's written first, in the foreground
        self.assertEqual(self._rows(rid1), N)
        self.assertEqual(state["pending_record"]["refinement_id"], rid2)
        refine3d._wait_for_record(self.job)
        self.assertEqual(self._rows(rid2), N)
        saved = json.loads(self.conn.execute("SELECT STATE_JSON FROM JOBS WHERE JOB_ID=?", (self.job,)).fetchone()["STATE_JSON"])
        self.assertEqual([h["refinement_id"] for h in saved["history"]], [rid1, rid2])
        self.assertEqual(self.conn.execute("SELECT STARTING_REFINEMENT_ID FROM REFINEMENT_LIST WHERE REFINEMENT_ID=?", (rid2,)).fetchone()[0], rid1)


if __name__ == "__main__":
    unittest.main()
