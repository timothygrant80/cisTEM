"""Refine 3D's start, the path with reference volumes already set: the input refinement's rows must reach the
scratch tables before the first refine3d launch reads them (a job on the job server failed on 2026-10-09 with
"No such file or directory: .../refinement_output_class1.cistem" when they were stored only on the other path)."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import db  # noqa: E402
import package_io  # noqa: E402
import refine3d  # noqa: E402
import refinement_packages as rp  # noqa: E402
import refinements  # noqa: E402
import starfile  # noqa: E402
import volumes  # noqa: E402
from stages import refine3d as refine3d_adapter  # noqa: E402


class FakeRuntime:
    def __init__(self):
        self.calls = []

    def submit_child(self, project_id, child_id, adapter, tasks, profile):
        self.calls.append({"child": child_id, "adapter": adapter, "tasks": tasks, "profile": profile})

    def append_log(self, project_id, job_id, text, level="info"):
        self.log = getattr(self, "log", []) + [text]


class Refine3DStartTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self._old = db.PROJECTS_ROOT, db.SYSTEM_DB_PATH, refine3d._runtime
        db.PROJECTS_ROOT = root / "projects"; db.PROJECTS_ROOT.mkdir()
        db.SYSTEM_DB_PATH = root / "system.db"
        self.runtime = FakeRuntime(); refine3d.configure(self.runtime)
        proj = db.create_project("P", 1, "u"); self.pid = proj["id"] if isinstance(proj, dict) else proj
        n, box = 12, 16
        stack = str(root / "stack.mrc"); w = rp.MrcStackWriter(stack, box, 1.0)
        for i in range(n):
            w.append(np.random.default_rng(i).normal(size=(box, box)).astype(np.float32))
        w.close()
        star = str(root / "p.star")
        starfile.write_star(star, [{"position_in_stack": i + 1, "image_is_active": 1, "psi": 10.0 * i, "theta": 20.0, "phi": 30.0, "defocus_1": 15000.0, "defocus_2": 14800.0,
                                    "occupancy": 100.0, "sigma": 1.0, "score": 10.0, "pixel_size": 1.0, "voltage": 300.0, "cs": 2.7, "amplitude_contrast": 0.07, "assigned_subset": 1 + i % 2} for i in range(n)])
        conn = db.get_conn(self.pid)
        out = package_io.import_package(conn, self.pid, {"format": "cistem", "stack_path": stack, "metadata_path": star, "name": "S", "symmetry": "C1",
                                                          "molecular_weight_kda": 100, "largest_dimension_a": 12, "protein_is_white": False, "cs_mm": 2.7})
        self.pkg = out["refinement_package_asset_id"]
        self.rid = conn.execute("SELECT MAX(REFINEMENT_ID) FROM REFINEMENT_LIST WHERE REFINEMENT_PACKAGE_ASSET_ID=?", (self.pkg,)).fetchone()[0]
        vol = str(root / "ref.mrc"); volumes.write_mrc_volume(vol, np.zeros((box, box, box), np.float32), 1.0)
        vid = volumes.add_volume_asset(conn, "ref", vol, 1.0, box, box, box)
        refinements.set_current_reference(conn, self.pkg, 1, vid)
        self.job = "job12345678"
        with conn:
            conn.execute("INSERT INTO JOBS(JOB_ID, STAGE, JOB_NUMBER, NAME, PARAMS_JSON, STATUS, PROGRESS, CREATED_AT) VALUES (?, 'refine3d', 1, 'Job 1', '{}', 'queued', 0, 'now')", (self.job,))
        self.conn = conn

    def tearDown(self):
        self.conn.close()
        db.PROJECTS_ROOT, db.SYSTEM_DB_PATH, refine3d._runtime = self._old
        self.tmp.cleanup()

    def test_start_with_references_writes_the_scratch_tables_before_launching(self):
        sys_conn = db.get_system_conn()
        try:
            profile = db.load_run_profile_by_name(sys_conn, "Local (single-threaded)")
        finally:
            sys_conn.close()
        params = {"refinement_package_id": self.pkg, "input_refinement_id": self.rid, "run_profile": profile["name"], "number_of_rounds": 1,
                  "refinement_type": "Local Refinement", "auto_mask": False, "use_mask": False, "use_blush": False, "percent_used": 100.0}
        state = refine3d.start(self.conn, self.pid, self.job, params, profile)
        self.assertFalse(state["initial"])
        self.assertEqual(len(self.runtime.calls), 1)
        call = self.runtime.calls[0]
        self.assertIs(call["adapter"], refine3d_adapter)
        self.assertEqual(len(call["tasks"]), min(12, profile["total_jobs"]))
        scratch = Path(state["scratch"])
        stored = starfile.read_params(str(scratch / "refinement_output_class1.cistem"), as_table=True)
        self.assertEqual(len(stored), 12)
        written = starfile.read_params(str(scratch / "input_par_{}_1.cistem".format(self.rid)), as_table=True)
        self.assertEqual(written["psi"].tolist(), stored["psi"].tolist())
        row = self.conn.execute("SELECT STATUS, STATE_JSON FROM JOBS WHERE JOB_ID=?", (self.job,)).fetchone()
        self.assertEqual(row["STATUS"], "running")
        self.assertEqual(json.loads(row["STATE_JSON"])["phase"], "refine")


if __name__ == "__main__":
    unittest.main()
