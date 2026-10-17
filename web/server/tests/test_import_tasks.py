"""Movie and image imports as background tasks (POST ...?async=1), followed through GET /tasks/:id."""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import auth  # noqa: E402
import cistem_server  # noqa: E402
import db  # noqa: E402
import refinement_packages as rp  # noqa: E402


class ImportTaskTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._root, self._auth = db.PROJECTS_ROOT, auth.AUTH_DB_PATH
        db.PROJECTS_ROOT = Path(self.tmp) / "projects"; auth.AUTH_DB_PATH = Path(self.tmp) / "auth.db"
        self.project = "t-import-tasks"
        (db.PROJECTS_ROOT / self.project).mkdir(parents=True)
        db.get_conn(self.project).close()
        token = auth.create_session(auth.create_user("boss", "password123", "admin")["id"])
        self.client = cistem_server.app.test_client()
        self.headers = {"Authorization": "Bearer " + token}
        for i in range(3):
            w = rp.MrcStackWriter(os.path.join(self.tmp, "mov_%d.mrc" % i), 16, 1.0)
            for f in range(4):
                w.append(np.full((16, 16), float(f), dtype=np.float32))
            w.close()

    def tearDown(self):
        db.PROJECTS_ROOT, auth.AUTH_DB_PATH = self._root, self._auth

    def _wait(self, task_id):
        for _ in range(200):
            t = self.client.get("/api/projects/%s/tasks/%s" % (self.project, task_id), headers=self.headers).get_json()
            if t["state"] != "running":
                return t
            time.sleep(0.02)
        self.fail("task did not finish")

    def test_a_movie_import_runs_as_a_task_with_progress(self):
        body = {"input_glob": os.path.join(self.tmp, "mov_*.mrc"), "voltage_kv": 300, "cs_mm": 2.7, "pixel_size_a": 1.0, "dose_per_frame": 1.0}
        r = self.client.post("/api/projects/%s/movies/import?async=1" % self.project, json=body, headers=self.headers)
        self.assertEqual(r.status_code, 202, r.data)
        task_id = r.get_json()["task_id"]
        t = self._wait(task_id)
        self.assertEqual((t["kind"], t["state"], t["done"], t["total"]), ("movie_import", "done", 3, 3))
        self.assertEqual(t["name"], body["input_glob"])
        self.assertEqual(t["result"]["movie_count"], 3)
        n = db.get_conn(self.project).execute("SELECT COUNT(*) FROM MOVIE_ASSETS").fetchone()[0]
        self.assertEqual(n, 3)
        # a done task leaves no row behind, and the generic list is empty again
        self.assertEqual(self.client.get("/api/projects/%s/tasks" % self.project, headers=self.headers).get_json()["tasks"], [])
        # the package routes do not list imports
        self.assertEqual(self.client.get("/api/projects/%s/refinement-packages/tasks" % self.project, headers=self.headers).get_json()["tasks"], [])

    def test_a_refused_import_fails_the_task_with_the_reason(self):
        r = self.client.post("/api/projects/%s/images/import?async=1" % self.project, json={"input_glob": os.path.join(self.tmp, "nothing_*.mrc"), "voltage_kv": 300, "cs_mm": 2.7, "pixel_size_a": 1.0}, headers=self.headers)
        self.assertEqual(r.status_code, 202, r.data)
        t = self._wait(r.get_json()["task_id"])
        self.assertEqual((t["kind"], t["state"]), ("image_import", "failed"))
        self.assertIn("no files match", t["error"])
        # gone from memory (as after a restart) it is listed from the table, with its kind, until acknowledged
        time.sleep(0.1)
        with cistem_server._package_tasks_lock:
            cistem_server._package_tasks.pop(t["task_id"])
        listed = self.client.get("/api/projects/%s/tasks?kind=image_import" % self.project, headers=self.headers).get_json()["tasks"]
        self.assertEqual([(x["task_id"], x["kind"], x["state"]) for x in listed], [(t["task_id"], "image_import", "failed")])
        self.assertEqual(self.client.post("/api/projects/%s/tasks/%s/ack" % (self.project, t["task_id"]), headers=self.headers).status_code, 200)
        self.assertEqual(self.client.get("/api/projects/%s/tasks" % self.project, headers=self.headers).get_json()["tasks"], [])

    def test_the_synchronous_import_still_answers_directly(self):
        r = self.client.post("/api/projects/%s/images/import" % self.project, json={"input_glob": os.path.join(self.tmp, "mov_*.mrc"), "voltage_kv": 300, "cs_mm": 2.7, "pixel_size_a": 1.0}, headers=self.headers)
        self.assertEqual((r.status_code, r.get_json()["image_count"]), (201, 3))
        r = self.client.post("/api/projects/%s/images/import" % self.project, json={"input_glob": os.path.join(self.tmp, "mov_*.mrc"), "voltage_kv": 300, "cs_mm": 2.7, "pixel_size_a": 1.0}, headers=self.headers)
        self.assertEqual(r.status_code, 400)
        self.assertIn("already imported", r.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
