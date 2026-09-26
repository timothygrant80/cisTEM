"""Paged lists and whole-group bulk actions: GET /particle-positions and
GET /refinement-packages/:id/particles take offset/limit and report the
total; delete / remove-from-group / add-to-group accept all_in_group so the
page's Select All and Remove All act on every member, not the rows loaded."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import auth  # noqa: E402
import db  # noqa: E402
import cistem_server  # noqa: E402
import refinement_packages as rp  # noqa: E402


class PagedListTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._root, self._auth = db.PROJECTS_ROOT, auth.AUTH_DB_PATH
        db.PROJECTS_ROOT = Path(self.tmp) / "projects"; auth.AUTH_DB_PATH = Path(self.tmp) / "auth.db"
        self.project = "t-paged"
        (db.PROJECTS_ROOT / self.project).mkdir(parents=True)
        token = auth.create_session(auth.create_user("boss", "password123", "admin")["id"])
        self.client = cistem_server.app.test_client()
        self.headers = {"Authorization": "Bearer " + token}
        conn = db.get_conn(self.project)
        with conn as c:
            c.execute("INSERT INTO IMAGE_ASSETS(IMAGE_ASSET_ID, NAME, FILENAME, X_SIZE, Y_SIZE, PIXEL_SIZE, PARENT_MOVIE_ID) VALUES (1, 'img', '/none.mrc', 10, 10, 1.0, -1)")
            c.execute("INSERT OR IGNORE INTO PARTICLE_POSITION_GROUP_LIST(GROUP_ID, GROUP_NAME, LIST_ID) VALUES (0, 'All', 0)")
            c.execute("INSERT INTO PARTICLE_POSITION_GROUP_LIST(GROUP_ID, GROUP_NAME, LIST_ID) VALUES (5, 'some', 0)")
            for i in range(1, 13):
                c.execute("INSERT INTO PARTICLE_POSITION_ASSETS(PARTICLE_POSITION_ASSET_ID, PARENT_IMAGE_ASSET_ID, PICKING_ID, PICK_JOB_ID, X_POSITION, Y_POSITION, PEAK_HEIGHT, "
                          "TEMPLATE_ASSET_ID, TEMPLATE_PSI, TEMPLATE_THETA, TEMPLATE_PHI) VALUES (?,1,1,'j',?,?,1,-1,0,0,0)", (i, i * 10.0, i * 20.0))
                c.execute("INSERT INTO PARTICLE_POSITION_GROUP_MEMBERS(GROUP_ID, PARTICLE_POSITION_ASSET_ID) VALUES (0, ?)", (i,))
                if i <= 7:
                    c.execute("INSERT INTO PARTICLE_POSITION_GROUP_MEMBERS(GROUP_ID, PARTICLE_POSITION_ASSET_ID) VALUES (5, ?)", (i,))
        conn.close()

    def tearDown(self):
        db.PROJECTS_ROOT, auth.AUTH_DB_PATH = self._root, self._auth

    def _get(self, path):
        r = self.client.get("/api/projects/{}{}".format(self.project, path), headers=self.headers)
        self.assertEqual(r.status_code, 200, r.data)
        return r.get_json()

    def _post(self, path, body):
        return self.client.post("/api/projects/{}{}".format(self.project, path), json=body, headers=self.headers)

    def test_positions_page_through_a_group(self):
        first = self._get("/particle-positions?group_id=5&limit=3")
        self.assertEqual(([p["PARTICLE_POSITION_ASSET_ID"] for p in first["particle_positions"]], first["total"], first["offset"], first["truncated"]), ([1, 2, 3], 7, 0, True))
        last = self._get("/particle-positions?group_id=5&limit=3&offset=6")
        self.assertEqual(([p["PARTICLE_POSITION_ASSET_ID"] for p in last["particle_positions"]], last["truncated"]), ([7], False))
        beyond = self._get("/particle-positions?group_id=5&limit=3&offset=60")
        self.assertEqual((beyond["particle_positions"], beyond["total"]), ([], 7))
        capped = self._get("/particle-positions?group_id=0&limit=999999")
        self.assertEqual(capped["limit"], cistem_server.POSITION_LIST_LIMIT)

    def test_all_in_group_bulk_actions(self):
        r = self._post("/particle-positions/add-to-group", {"all_in_group": 5, "group_name": "copy"})
        self.assertEqual(r.status_code, 200, r.data)
        self.assertEqual(r.get_json()["added"], 7)
        copy_id = r.get_json()["group_id"]
        self.assertEqual(self._get("/particle-positions?group_id={}".format(copy_id))["total"], 7)
        r = self._post("/particle-position-groups/{}/remove-particle-positions".format(copy_id), {"all_in_group": copy_id})
        self.assertEqual((r.status_code, r.get_json()["removed"]), (200, 7))
        self.assertEqual(self._get("/particle-positions?group_id={}".format(copy_id))["total"], 0)
        self.assertEqual(self._get("/particle-positions?group_id=0")["total"], 12)
        r = self._post("/particle-positions/delete", {"all_in_group": 5})
        self.assertEqual((r.status_code, r.get_json()["deleted"]), (200, 7))
        self.assertEqual(self._get("/particle-positions?group_id=0")["total"], 5)
        r = self._post("/particle-positions/delete", {"particle_position_ids": [8, 9]})
        self.assertEqual((r.status_code, r.get_json()["deleted"]), (200, 2))
        r = self._post("/particle-positions/delete", {})
        self.assertEqual(r.status_code, 400)

    def test_package_particles_page(self):
        conn = db.get_conn(self.project)
        contained = [{"position_id": i, "image_id": 1, "position_in_stack": i, "x": 0.0, "y": 0.0, "pixel_size": 1.0, "defocus1": 1.0, "defocus2": 1.0,
                      "defocus_angle": 0.0, "phase_shift": 0.0, "cs": 2.7, "voltage": 300.0, "amplitude_contrast": 0.07, "subset": 1} for i in range(1, 10)]
        rp.insert_package(conn, 1, "pkg", "/none/stack.mrc", 32, 1.0, "C1", 100.0, 50.0, 1, contained, 1)
        rows = [(c["position_in_stack"], 0, 0, 0, 0, 0, 1, 1, 0, 0, 100, 0, 1, 0, 1, 1.0, 300, 2.7, 0.07, 0, 0, 0, 0, 1) for c in contained]
        rp.insert_initial_refinement(conn, 1, 1, "Random Parameters", [rows], 32, 1.0, 100.0)
        conn.close()
        page = self._get("/refinement-packages/1/particles?offset=4&limit=3")
        self.assertEqual(([p["position_in_stack"] for p in page["particles"]], page["total"]), ([5, 6, 7], 9))
        details = self._get("/refinement-packages/1")
        self.assertEqual((len(details["particles"]), details["particle_total"]), (9, 9))
        self.assertEqual(self.client.get("/api/projects/{}/refinement-packages/2/particles".format(self.project), headers=self.headers).status_code, 404)


if __name__ == "__main__":
    unittest.main()
