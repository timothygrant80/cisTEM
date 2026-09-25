"""tools/write_job_results.py: a completed job whose results were never written
(the adapter's finalize failed) gets them written from the stored task
results; a second run refuses because the metrics now carry the summary."""
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import db  # noqa: E402
import cistem_server  # noqa: E402
from stages import ctffind  # noqa: E402

_spec = importlib.util.spec_from_file_location("write_job_results", os.path.join(os.path.dirname(HERE), "..", "tools", "write_job_results.py"))
write_job_results = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(write_job_results)


class WriteJobResultsToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._root = db.PROJECTS_ROOT
        db.PROJECTS_ROOT = Path(self.tmp)
        self.project = "t-" + os.path.basename(self.tmp)[-6:]
        (db.PROJECTS_ROOT / self.project).mkdir(parents=True)
        conn = db.get_conn(self.project)
        with conn:
            conn.execute("INSERT INTO IMAGE_ASSETS(IMAGE_ASSET_ID, NAME, FILENAME, PARENT_MOVIE_ID, X_SIZE, Y_SIZE, PIXEL_SIZE, VOLTAGE, "
                         "SPHERICAL_ABERRATION, PROTEIN_IS_WHITE) VALUES (7, 'img', '/none/img.mrc', -1, 100, 100, 1.2, 300.0, 2.7, 0)")
            conn.execute("INSERT INTO IMAGE_GROUP_LIST(GROUP_ID, GROUP_NAME, LIST_ID) VALUES (5, 'g', 0)")
            conn.execute("INSERT INTO IMAGE_GROUP_MEMBERS(GROUP_ID, IMAGE_ASSET_ID) VALUES (5, 7)")
        tasks = ctffind.build_tasks(conn, self.project, {"image_group_id": 5})
        data = [20015.0, 19800.0, 45.0, 0.0, 0.05, 4.2, 0.0, 0.14, 0.0, 0.0, None]
        with conn:
            conn.execute("INSERT INTO JOBS(JOB_ID, JOB_NUMBER, STAGE, STATUS, CREATED_AT, TASKS_JSON, METRICS_JSON) "
                         "VALUES ('job1', 1, 'ctf_estimation', 'completed', 0, ?, ?)",
                         (json.dumps(tasks), json.dumps({"cpu_ms": 10, "tasks_ok": 1, "tasks_failed": 0})))
            conn.execute("INSERT INTO JOB_TASKS(JOB_ID, TASK_INDEX, REF, STATUS, RESULT_JSON, FINISHED_AT) VALUES ('job1', 0, '7', 'ok', ?, 'now')",
                         (json.dumps({"kind": "floats", "data": data}),))
        conn.close()

    def tearDown(self):
        db.PROJECTS_ROOT = self._root

    def _run(self, *args):
        out = io.StringIO()
        with redirect_stdout(out):
            code = write_job_results.main(["write_job_results.py"] + list(args))
        return code, out.getvalue()

    def test_writes_once_then_refuses(self):
        code, out = self._run(self.project, "job1")
        self.assertEqual(code, 0, out)
        conn = db.get_conn(self.project)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM ESTIMATED_CTF_PARAMETERS").fetchone()[0], 1)
        metrics = json.loads(conn.execute("SELECT METRICS_JSON FROM JOBS WHERE JOB_ID='job1'").fetchone()[0])
        self.assertEqual(metrics["ctf_estimates_written"], 1)
        log = " ".join(r[0] for r in conn.execute("SELECT LINE FROM JOB_LOG_LINES WHERE JOB_ID='job1' ORDER BY SEQ"))
        self.assertIn("sample thickness", log)
        conn.close()
        code, out = self._run(self.project, "job1")
        self.assertEqual(code, 1)
        self.assertIn("already has its results written", out)

    def test_refuses_unknown_and_unfinished_jobs(self):
        self.assertEqual(self._run(self.project, "nope")[0], 1)
        self.assertEqual(self._run("no-such-project", "job1")[0], 1)
        conn = db.get_conn(self.project)
        with conn:
            conn.execute("UPDATE JOBS SET STATUS='running' WHERE JOB_ID='job1'")
        conn.close()
        code, out = self._run(self.project, "job1")
        self.assertEqual(code, 1)
        self.assertIn("not completed", out)


if __name__ == "__main__":
    unittest.main()
