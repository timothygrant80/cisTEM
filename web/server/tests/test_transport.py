"""What goes over the wire: gzip for the page and JSON when the browser accepts
it (validators and 304s untouched), and a job list without each job's
parameters, which only GET /jobs/:id carries."""
import gzip
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import auth  # noqa: E402
import db  # noqa: E402
import cistem_server  # noqa: E402


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._root, self._auth = db.PROJECTS_ROOT, auth.AUTH_DB_PATH
        db.PROJECTS_ROOT = Path(self.tmp) / "projects"; auth.AUTH_DB_PATH = Path(self.tmp) / "auth.db"
        self.project = "t-wire"
        (db.PROJECTS_ROOT / self.project).mkdir(parents=True)
        token = auth.create_session(auth.create_user("boss", "password123", "admin")["id"])
        self.client = cistem_server.app.test_client()
        self.headers = {"Authorization": "Bearer " + token}
        conn = db.get_conn(self.project)
        with conn:
            for i in range(40):
                conn.execute("INSERT INTO JOBS(JOB_ID, STAGE, JOB_NUMBER, NAME, PARAMS_JSON, STATUS, CREATED_AT) VALUES (?, 'ctf_estimation', ?, ?, ?, 'completed', ?)",
                             ("job%d" % i, i, "Job %d" % i, '{"image_group_id": 5, "box_size": 512, "note": "%s"}' % ("x" * 200), "2026-09-26T0%d" % (i % 10)))
        conn.close()

    def tearDown(self):
        db.PROJECTS_ROOT, auth.AUTH_DB_PATH = self._root, self._auth

    def test_page_is_gzipped_for_a_browser_that_accepts_it(self):
        plain = self.client.get("/")
        gz = self.client.get("/", headers={"Accept-Encoding": "gzip, deflate"})
        self.assertIsNone(plain.headers.get("Content-Encoding"))
        self.assertEqual(gz.headers.get("Content-Encoding"), "gzip")
        self.assertIn("Accept-Encoding", gz.headers.get("Vary", ""))
        self.assertLess(int(gz.headers["Content-Length"]), int(plain.headers["Content-Length"]) // 3)
        self.assertEqual(gz.headers.get("ETag"), plain.headers.get("ETag"))
        again = self.client.get("/", headers={"Accept-Encoding": "gzip", "If-None-Match": plain.headers["ETag"]})
        self.assertEqual(again.status_code, 304)
        self.assertIsNone(again.headers.get("Content-Encoding"))
        png = self.client.get("/logo.png", headers={"Accept-Encoding": "gzip"})
        self.assertIsNone(png.headers.get("Content-Encoding"))

    def test_json_is_gzipped_when_large_and_left_alone_when_small(self):
        jobs = self.client.get("/api/projects/{}/jobs".format(self.project), headers=dict(self.headers, **{"Accept-Encoding": "gzip"}))
        self.assertEqual(jobs.status_code, 200)
        self.assertEqual(jobs.headers.get("Content-Encoding"), "gzip")
        self.assertEqual(len(json.loads(gzip.decompress(jobs.data))["jobs"]), 40)
        health = self.client.get("/api/health", headers={"Accept-Encoding": "gzip"})
        self.assertIsNone(health.headers.get("Content-Encoding"))

    def test_job_list_has_no_params_but_the_job_does(self):
        jobs = self.client.get("/api/projects/{}/jobs".format(self.project), headers=self.headers).get_json()["jobs"]
        self.assertNotIn("params", jobs[0])
        self.assertEqual(jobs[0]["name"], "Job 0")
        one = self.client.get("/api/projects/{}/jobs/job3".format(self.project), headers=self.headers).get_json()
        self.assertEqual(one["params"]["box_size"], 512)


if __name__ == "__main__":
    unittest.main()
