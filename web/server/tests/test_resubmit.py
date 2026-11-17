"""POST /jobs/<id>/resubmit: a failed or terminated job runs again as a new job with its exact parameters."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import auth  # noqa: E402
import cistem_server  # noqa: E402
import db  # noqa: E402

PARAMS = {"movie_group_id": 0, "min_shift_a": 2, "max_shift_a": 40, "bfactor_a2": 1500, "run_profile": "Local"}


class ResubmitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._root, self._auth, self._sys = db.PROJECTS_ROOT, auth.AUTH_DB_PATH, db.SYSTEM_DB_PATH
        db.PROJECTS_ROOT = Path(self.tmp) / "projects"; auth.AUTH_DB_PATH = Path(self.tmp) / "auth.db"; db.SYSTEM_DB_PATH = Path(self.tmp) / "system.db"
        self.project = "t-resubmit"
        (db.PROJECTS_ROOT / self.project).mkdir(parents=True)
        conn = db.get_conn(self.project)
        with conn:
            conn.execute("INSERT INTO MOVIE_ASSETS(MOVIE_ASSET_ID, NAME, FILENAME, X_SIZE, Y_SIZE, NUMBER_OF_FRAMES, PIXEL_SIZE, VOLTAGE, SPHERICAL_ABERRATION) VALUES (1, 'm', '/none.mrc', 10, 10, 3, 1.0, 300, 2.7)")
            conn.execute("INSERT OR IGNORE INTO MOVIE_GROUP_LIST(GROUP_ID, GROUP_NAME, LIST_ID) VALUES (0, 'All Movies', 0)")
            conn.execute("INSERT OR IGNORE INTO MOVIE_GROUP_MEMBERS(GROUP_ID, MOVIE_ASSET_ID) VALUES (0, 1)")
            conn.execute("INSERT INTO JOBS(JOB_ID, STAGE, JOB_NUMBER, NAME, PARAMS_JSON, STATUS, CREATED_AT) VALUES ('dead', 'motion_correction', 3, 'Job 3', ?, 'failed', 'c')", (json.dumps(PARAMS),))
            conn.execute("INSERT INTO JOBS(JOB_ID, STAGE, JOB_NUMBER, NAME, PARAMS_JSON, STATUS, CREATED_AT) VALUES ('busy', 'motion_correction', 4, 'Job 4', ?, 'running', 'c')", (json.dumps(PARAMS),))
            conn.execute("INSERT INTO JOBS(JOB_ID, STAGE, JOB_NUMBER, NAME, STATUS, CREATED_AT) VALUES ('old', 'motion_correction', 5, 'Job 5', 'failed', 'c')")
        conn.close()
        token = auth.create_session(auth.create_user("boss", "password123", "admin")["id"])
        self.client = cistem_server.app.test_client()
        self.headers = {"Authorization": "Bearer " + token}

    def tearDown(self):
        db.PROJECTS_ROOT, auth.AUTH_DB_PATH, db.SYSTEM_DB_PATH = self._root, self._auth, self._sys

    def _post(self, path):
        return self.client.post("/api/projects/{}{}".format(self.project, path), headers=self.headers)

    def test_a_failed_job_runs_again_as_a_new_job_with_the_same_parameters(self):
        r = self._post("/jobs/dead/resubmit")
        self.assertEqual(r.status_code, 201, r.get_data(as_text=True))
        job = r.get_json()
        self.assertNotEqual(job["id"], "dead")
        self.assertEqual(job["stage"], "motion_correction")
        self.assertEqual(job["number"], 6)
        conn = db.get_conn(self.project)
        row = conn.execute("SELECT PARAMS_JSON, STATUS FROM JOBS WHERE JOB_ID=?", (job["id"],)).fetchone()
        conn.close()
        self.assertEqual(json.loads(row["PARAMS_JSON"]), PARAMS)

    def test_a_running_job_an_old_row_and_an_unknown_id_are_refused(self):
        self.assertEqual(self._post("/jobs/busy/resubmit").status_code, 409)
        self.assertEqual(self._post("/jobs/old/resubmit").status_code, 400)
        self.assertEqual(self._post("/jobs/nope/resubmit").status_code, 404)


if __name__ == "__main__":
    unittest.main()
