"""GET /projects/:id/open: everything a project open shows, each part exactly
the corresponding route's answer, so the page makes one request instead of
fourteen and each loader takes its part."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import auth  # noqa: E402
import db  # noqa: E402
import cistem_server  # noqa: E402


class OpenProjectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._root, self._auth, self._sys = db.PROJECTS_ROOT, auth.AUTH_DB_PATH, db.SYSTEM_DB_PATH
        db.PROJECTS_ROOT = Path(self.tmp) / "projects"; auth.AUTH_DB_PATH = Path(self.tmp) / "auth.db"; db.SYSTEM_DB_PATH = Path(self.tmp) / "system.db"
        self.project = "t-open"
        (db.PROJECTS_ROOT / self.project).mkdir(parents=True)
        conn = db.get_conn(self.project)
        with conn:
            conn.execute("INSERT INTO MOVIE_ASSETS(MOVIE_ASSET_ID, NAME, FILENAME, X_SIZE, Y_SIZE, NUMBER_OF_FRAMES, PIXEL_SIZE, VOLTAGE, SPHERICAL_ABERRATION) VALUES (1, 'm', '/none.mrc', 10, 10, 3, 1.0, 300, 2.7)")
            conn.execute("INSERT OR IGNORE INTO MOVIE_GROUP_LIST(GROUP_ID, GROUP_NAME, LIST_ID) VALUES (0, 'All Movies', 0)")
            conn.execute("INSERT OR IGNORE INTO MOVIE_GROUP_MEMBERS(GROUP_ID, MOVIE_ASSET_ID) VALUES (0, 1)")
            conn.execute("INSERT INTO JOBS(JOB_ID, STAGE, JOB_NUMBER, NAME, STATUS, CREATED_AT) VALUES ('j1', 'ctf_estimation', 1, 'Job 1', 'completed', 'c')")
        conn.close()
        token = auth.create_session(auth.create_user("boss", "password123", "admin")["id"])
        self.client = cistem_server.app.test_client()
        self.headers = {"Authorization": "Bearer " + token}

    def tearDown(self):
        db.PROJECTS_ROOT, auth.AUTH_DB_PATH, db.SYSTEM_DB_PATH = self._root, self._auth, self._sys

    def _get(self, path):
        r = self.client.get("/api/projects/{}{}".format(self.project, path), headers=self.headers)
        self.assertEqual(r.status_code, 200, (path, r.data[:200]))
        return r.get_json()

    def test_every_part_matches_its_route(self):
        o = self._get("/open")
        expected = {
            "movies_import_defaults": "/movies/import-defaults", "movie_groups": "/movie-groups", "movies:0": "/movies?group_id=0",
            "images_import_defaults": "/images/import-defaults", "image_groups": "/image-groups", "images:0": "/images?group_id=0",
            "particle_position_groups": "/particle-position-groups", "particle_positions:0": "/particle-positions?group_id=0&offset=0&limit=5000",
            "refinement_packages": "/refinement-packages", "volume_groups": "/volume-groups", "volumes:0": "/volumes?group_id=0",
            "jobs": "/jobs", "package_tasks": "/refinement-packages/tasks",
        }
        self.assertEqual(set(o), set(expected) | {"run_profiles"})
        for key, path in expected.items():
            self.assertEqual(o[key], self._get(path), key)
        self.assertEqual(o["run_profiles"], self.client.get("/api/run-profiles", headers=self.headers).get_json())
        self.assertEqual(len(o["movies:0"]["movies"]), 1)
        self.assertEqual([j["name"] for j in o["jobs"]["jobs"]], ["Job 1"])

    def test_needs_the_token(self):
        self.assertEqual(self.client.get("/api/projects/{}/open".format(self.project)).status_code, 401)


if __name__ == "__main__":
    unittest.main()
