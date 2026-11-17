"""Show Job Details: a single result's answer carries the job's stage and the inputs the Actions panel sent
(`job_stage`, `job_params`), and a 3D refinement's class carries its reconstruction job's row, so the page can
list every parameter -- the ones the result tables keep and the ones only the job's row keeps."""
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
import refinements  # noqa: E402
import volumes  # noqa: E402

PARAMS = {"focused_classification": True, "sphere_x_a": 12.5, "sphere_radius_a": 30, "use_blush": False, "run_profile": "Local"}


class JobDetailsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._root, self._auth = db.PROJECTS_ROOT, auth.AUTH_DB_PATH
        db.PROJECTS_ROOT = Path(self.tmp) / "projects"; auth.AUTH_DB_PATH = Path(self.tmp) / "auth.db"
        self.project = "t-details"
        (db.PROJECTS_ROOT / self.project).mkdir(parents=True)
        token = auth.create_session(auth.create_user("boss", "password123", "admin")["id"])
        self.client = cistem_server.app.test_client()
        self.headers = {"Authorization": "Bearer " + token}
        self.conn = db.get_conn(self.project)
        with self.conn as c:
            c.execute("INSERT INTO JOBS(JOB_ID, STAGE, JOB_NUMBER, NAME, PARAMS_JSON, STATUS, PROGRESS, CREATED_AT) VALUES ('j1', 'refine3d', 7, 'Job 7', ?, 'finished', 100, 'now')",
                      (json.dumps(PARAMS),))
            c.execute("INSERT INTO MOVIE_ASSETS(MOVIE_ASSET_ID, NAME, FILENAME) VALUES (1, 'm', '/none.mrc')")
            c.execute("INSERT INTO IMAGE_ASSETS(IMAGE_ASSET_ID, NAME, FILENAME, X_SIZE, Y_SIZE, PIXEL_SIZE, PARENT_MOVIE_ID) VALUES (1, 'img', '/none.mrc', 10, 10, 1.0, 1)")

    def tearDown(self):
        self.conn.close()
        db.PROJECTS_ROOT, auth.AUTH_DB_PATH = self._root, self._auth

    def _get(self, path):
        r = self.client.get("/api/projects/{}{}".format(self.project, path), headers=self.headers)
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        return r.get_json()

    def test_an_alignment_carries_its_jobs_inputs_and_none_without_a_job(self):
        with self.conn as c:
            c.execute("INSERT INTO MOVIE_ALIGNMENT_LIST(ALIGNMENT_ID, ALIGNMENT_JOB_ID, MOVIE_ASSET_ID, OUTPUT_FILE) VALUES (1, 'j1', 1, '/none_sum.mrc')")
            c.execute("INSERT INTO MOVIE_ALIGNMENT_LIST(ALIGNMENT_ID, ALIGNMENT_JOB_ID, MOVIE_ASSET_ID, OUTPUT_FILE) VALUES (2, 'gone', 1, '/none_sum.mrc')")
        d = self._get("/alignments/1")
        self.assertEqual(d["job_stage"], "refine3d")
        self.assertEqual(d["job_params"], PARAMS)
        d = self._get("/alignments/2")
        self.assertIsNone(d["job_stage"]); self.assertIsNone(d["job_params"])

    def test_a_ctf_estimate_a_picking_and_a_classification_carry_them_too(self):
        with self.conn as c:
            c.execute("INSERT INTO ESTIMATED_CTF_PARAMETERS(CTF_ESTIMATION_ID, CTF_ESTIMATION_JOB_ID, IMAGE_ASSET_ID, OUTPUT_DIAGNOSTIC_FILE) VALUES (1, 'j1', 1, '')")
            c.execute("INSERT INTO PARTICLE_PICKING_LIST(PICKING_ID, PICKING_JOB_ID, PARENT_IMAGE_ASSET_ID) VALUES (1, 'j1', 1)")
            c.execute("INSERT INTO CLASSIFICATION_LIST(CLASSIFICATION_ID, REFINEMENT_PACKAGE_ASSET_ID, NAME, NUMBER_OF_CLASSES, JOB_ID) VALUES (1, 1, 'c', 2, 'j1')")
        for path in ("/ctf-estimates/1", "/picks/1", "/classifications/1"):
            d = self._get(path)
            self.assertEqual(d["job_params"], PARAMS, path)
            self.assertEqual(d["job_stage"], "refine3d", path)

    def test_a_startup_carries_them(self):
        settings = {"number_of_starts": 2, "number_of_rounds": 40, "initial_resolution_limit": 20.0, "final_resolution_limit": 8.0, "auto_mask": True,
                    "auto_percent_used": True, "start_percent_used": 10.0, "end_percent_used": 10.0, "mask_radius": 90.0, "apply_blurring": False, "smoothing_factor": 1.0}
        volumes.add_startup_job(self.conn, 1, 1, "Ab-Initio #1", settings, [], job_id="j1")
        volumes.add_startup_job(self.conn, 2, 1, "Ab-Initio #2", settings, [], job_id=None)
        d = self._get("/startups")["startups"]
        self.assertEqual(d[0]["job_params"], PARAMS)
        self.assertIsNone(d[1]["job_params"])

    def test_a_refinements_class_carries_its_reconstruction_job(self):
        with self.conn as c:
            c.execute("INSERT INTO REFINEMENT_LIST(REFINEMENT_ID, REFINEMENT_PACKAGE_ASSET_ID, NAME, NUMBER_OF_CLASSES, NUMBER_OF_PARTICLES, JOB_ID) VALUES (3, 1, 'Refinement #3', 1, 10, 'j1')")
            c.execute("CREATE TABLE REFINEMENT_DETAILS_3({})".format(refinements.DETAIL_DDL))
            c.execute("INSERT INTO REFINEMENT_DETAILS_3(CLASS_NUMBER, SHOULD_FOCUS_CLASSIFY, SPHERE_X_COORD, SPHERE_RADIUS, RECONSTRUCTION_ID, RECONSTRUCTED_VOLUME_ASSET_ID) VALUES (1, 1, 12.5, 30.0, 5, -1)")
            c.execute("CREATE TABLE REFINEMENT_RESULT_3_1({})".format(refinements.RESULT_DDL))
        refinements.add_reconstruction_job(self.conn, 5, 1, 3, "", 0.0, 90.0, 0.0, 2.0, True, False, True, False, 1.0, 1, -1)
        d = self._get("/refinements/3")
        self.assertEqual(d["job_params"], PARAMS)
        c = d["classes"][0]
        self.assertEqual(c["should_focus_classify"], 1)
        self.assertEqual(c["sphere_x_coord"], 12.5)
        self.assertEqual(c["reconstruction"]["outer_mask_radius"], 90.0)
        self.assertEqual(c["reconstruction"]["should_adjust_scores"], 1)
        self.assertEqual(c["reconstruction"]["should_crop_images"], 0)
        # a refinement without a reconstruction (an import) has none
        with self.conn as c:
            c.execute("UPDATE REFINEMENT_DETAILS_3 SET RECONSTRUCTION_ID=-1")
        self.assertIsNone(self._get("/refinements/3")["classes"][0]["reconstruction"])


if __name__ == "__main__":
    unittest.main()
