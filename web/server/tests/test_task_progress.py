"""_task_progress(): a job whose tasks each cover a particle range reports
particles seen of particles expected (cisTEM's number_of_received_particle_results),
from the controller's count-only task_progress frames, finished tasks counting in
full; a job whose tasks are single results reports tasks only."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db  # noqa: E402
import job_protocol as jp  # noqa: E402
import cistem_server  # noqa: E402
from stages import refine2d, refine3d, ctffind  # noqa: E402


def _particle_task(adapter, index, first, last, percent_used=1.0, input_class_averages="averages.mrc"):
    values = {"first_particle": first, "last_particle": last, "percent_used": percent_used, "input_class_averages": input_class_averages}
    kinds = {"t": "text", "i": "int", "f": "float", "b": "bool"}
    args = []
    for name, t in zip(adapter.ARGUMENT_NAMES, adapter.ARGUMENT_TYPES):
        v = values.get(name, {"t": "x", "i": 1, "f": 1.0, "b": False}[t])
        args.append(jp.arg(kinds[t], v))
    return {"index": index, "ref": index, "args": args}


class TaskProgressTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._root = db.PROJECTS_ROOT
        db.PROJECTS_ROOT = Path(self.tmp)
        self.project = "t-" + os.path.basename(self.tmp)[-6:]
        (db.PROJECTS_ROOT / self.project).mkdir(parents=True)
        self.conn = db.get_conn(self.project)

    def tearDown(self):
        self.conn.close()
        db.PROJECTS_ROOT = self._root
        for job_id in ("r2d", "r2d-half", "r3d", "ctf"):
            cistem_server._forget_result_counts(job_id)

    def _job(self, job_id, stage, tasks):
        self.conn.execute("INSERT INTO JOBS(JOB_ID, STAGE, NAME, STATUS, CREATED_AT, TASKS_JSON) VALUES (?, ?, ?, 'running', ?, ?)",
                          (job_id, stage, job_id, cistem_server.now_iso(), json.dumps(tasks)))
        self.conn.commit()
        return self.conn.execute("SELECT * FROM JOBS WHERE JOB_ID=?", (job_id,)).fetchone()

    def _finish(self, job_id, index):
        self.conn.execute("INSERT INTO JOB_TASKS(JOB_ID, TASK_INDEX, STATUS, FINISHED_AT) VALUES (?, ?, 'ok', ?)",
                          (job_id, index, cistem_server.now_iso()))
        self.conn.commit()

    def test_refine2d_round_counts_every_particle_and_startup_the_used_share(self):
        # a refinement round: percent_used (a fraction on the wire) does not reduce what refine2d sends
        row = self._job("r2d", "class2d_refine2d", [_particle_task(refine2d, 0, 1, 100, 0.5), _particle_task(refine2d, 1, 101, 200, 0.5)])
        info = cistem_server._task_progress(self.conn, row)
        self.assertEqual((info["task_count"], info["tasks_done"]), (2, 0))
        self.assertEqual((info["results_expected"], info["results_seen"]), (200, 0))
        # the start-up round (no input averages) sends one result per particle it uses
        self.assertEqual(cistem_server._expected_results("class2d_refine2d", [_particle_task(refine2d, 0, 1, 1000, 0.25, "/dev/null")]), {0: 250})
        self.assertEqual(cistem_server._expected_results("class2d_refine2d", [_particle_task(refine2d, 0, 1, 1000, 25.0, "/dev/null")]), {0: 250})
        row = self._job("r2d-half", "class2d_refine2d", [_particle_task(refine2d, 0, 1, 100, 0.5), _particle_task(refine2d, 1, 101, 200, 0.5)])
        info = cistem_server._task_progress(self.conn, row)
        self.assertIsNone(info["first_result_at"])
        # counts arrive (the newest count wins, never a smaller resend), one task finishes
        cistem_server._note_result_count("r2d-half", 0, 20)
        cistem_server._note_result_count("r2d-half", 0, 17)
        cistem_server._note_result_count("r2d-half", 1, 5)
        self._finish("r2d-half", 1)
        info = cistem_server._task_progress(self.conn, row)
        self.assertEqual((info["results_expected"], info["results_seen"]), (200, 120))
        self.assertEqual(info["tasks_done"], 1)
        self.assertIsNotNone(info["first_result_at"])

    def test_refine3d_counts_every_particle_and_caps_at_the_range(self):
        row = self._job("r3d", "abinitio_refine3d", [_particle_task(refine3d, 0, 1, 40, 10.0), _particle_task(refine3d, 1, 41, 80, 10.0)])
        cistem_server._note_result_count("r3d", 0, 55)   # more than the range: a resent queue, say
        info = cistem_server._task_progress(self.conn, row)
        self.assertEqual((info["results_expected"], info["results_seen"]), (80, 40))

    def test_single_result_tasks_report_tasks_only(self):
        row = self._job("ctf", "ctf_estimation", [{"index": 0, "ref": 1, "args": []}, {"index": 1, "ref": 2, "args": []}])
        self._finish("ctf", 0)
        info = cistem_server._task_progress(self.conn, row)
        self.assertEqual((info["task_count"], info["tasks_done"]), (2, 1))
        self.assertNotIn("results_expected", info)
        self.assertIsNone(cistem_server._expected_results("ctf_estimation", [{"index": 0, "args": []}]))

    def test_count_only_frame_is_recorded_and_skips_the_adapter(self):
        sink = cistem_server.DbSink()
        sink.register("r3d", self.project)
        self._job("r3d", "abinitio_refine3d", [_particle_task(refine3d, 0, 1, 40)])
        sink.on_task_progress("r3d", 0, 0, 12, 0, None)
        self.assertEqual(cistem_server._result_counts("r3d")[0], {0: 12})
        sink._forget("r3d")
        self.assertEqual(cistem_server._result_counts("r3d"), ({}, None))


if __name__ == "__main__":
    unittest.main()
