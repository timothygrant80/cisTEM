"""A package's stack, cut from picks, can be deleted from disk to save space and cut again byte for byte;
one whose boxes did not come straight from the images, or whose picks have changed since, cannot."""
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import auth  # noqa: E402
import cistem_server  # noqa: E402
import db  # noqa: E402
import refinement_packages as rp  # noqa: E402


def _write_mrc(path, image):
    ny, nx = image.shape
    head = bytearray(1024)
    struct.pack_into("<iiii", head, 0, nx, ny, 1, 2)
    struct.pack_into("<iii", head, 28, nx, ny, 1)
    struct.pack_into("<fff", head, 40, nx * 1.5, ny * 1.5, 1.5)
    struct.pack_into("<iii", head, 64, 1, 2, 3)
    head[208:212] = b"MAP "; head[212:216] = b"\x44\x44\x00\x00"
    with open(path, "wb") as fh:
        fh.write(bytes(head)); fh.write(image.astype("<f4").tobytes())


class StackRegenerationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._root, self._auth, self._sys = db.PROJECTS_ROOT, auth.AUTH_DB_PATH, db.SYSTEM_DB_PATH
        db.PROJECTS_ROOT = Path(self.tmp) / "projects"; auth.AUTH_DB_PATH = Path(self.tmp) / "auth.db"; db.SYSTEM_DB_PATH = Path(self.tmp) / "system.db"
        self.project = "t-stack"
        (db.PROJECTS_ROOT / self.project).mkdir(parents=True)
        self.conn = db.get_conn(self.project)
        rng = np.random.default_rng(5)
        with self.conn as c:
            for i in (1, 2):
                img = (rng.standard_normal((96, 128)) * 10 + 100).astype(np.float32)
                path = os.path.join(self.tmp, "img%d.mrc" % i); _write_mrc(path, img)
                c.execute("INSERT INTO IMAGE_ASSETS(IMAGE_ASSET_ID, NAME, FILENAME, X_SIZE, Y_SIZE, PIXEL_SIZE, VOLTAGE, SPHERICAL_ABERRATION, PARENT_MOVIE_ID, CTF_ESTIMATION_ID) "
                          "VALUES (?,?,?,128,96,1.5,300,2.7,-1,?)", (i, "img%d" % i, path, i))
                c.execute("INSERT INTO ESTIMATED_CTF_PARAMETERS(CTF_ESTIMATION_ID, IMAGE_ASSET_ID, VOLTAGE, SPHERICAL_ABERRATION, AMPLITUDE_CONTRAST, DEFOCUS1, DEFOCUS2, "
                          "DEFOCUS_ANGLE, ADDITIONAL_PHASE_SHIFT, PIXEL_SIZE) VALUES (?,?,300,2.7,0.07,15000,14000,10,0,1.5)", (i, i))
            c.execute("INSERT OR IGNORE INTO PARTICLE_POSITION_GROUP_LIST(GROUP_ID, GROUP_NAME, LIST_ID) VALUES (0, 'All', 0)")
            pid = 0
            for i, picks in {1: [(40, 30), (100, 60)], 2: [(3, 50), (120, 90), (64, 2)]}.items():
                for cx, cy in picks:
                    pid += 1
                    c.execute("INSERT INTO PARTICLE_POSITION_ASSETS(PARTICLE_POSITION_ASSET_ID, PARENT_IMAGE_ASSET_ID, PICKING_ID, PICK_JOB_ID, X_POSITION, Y_POSITION, PEAK_HEIGHT, "
                              "TEMPLATE_ASSET_ID, TEMPLATE_PSI, TEMPLATE_THETA, TEMPLATE_PHI) VALUES (?,?,1,'j',?,?,8,-1,0,0,0)", (pid, i, cx * 1.5, cy * 1.5))
                    c.execute("INSERT INTO PARTICLE_POSITION_GROUP_MEMBERS(GROUP_ID, PARTICLE_POSITION_ASSET_ID) VALUES (0, ?)", (pid,))
        self.res = rp.create_package(self.conn, self.project, {"particle_group_id": 0, "box_size": 32, "number_of_classes": 1})
        self.pid = self.res["refinement_package_asset_id"]
        token = auth.create_session(auth.create_user("boss", "password123", "admin")["id"])
        self.client = cistem_server.app.test_client()
        self.headers = {"Authorization": "Bearer " + token}

    def tearDown(self):
        self.conn.close()
        db.PROJECTS_ROOT, auth.AUTH_DB_PATH, db.SYSTEM_DB_PATH = self._root, self._auth, self._sys

    def test_a_stack_cut_from_picks_is_deleted_and_comes_back_byte_for_byte(self):
        path = self.res["stack_filename"]
        with open(path, "rb") as fh:
            before = fh.read()
        st = rp.stack_status(self.conn, self.pid)
        self.assertTrue(st["regenerable"], st["reason"]); self.assertTrue(st["on_disk"]); self.assertEqual(st["size"], len(before)); self.assertEqual(st["source"], "picks")
        self.assertEqual(rp.delete_stack(self.conn, self.pid), len(before))
        self.assertFalse(os.path.exists(path))
        st = rp.stack_status(self.conn, self.pid)
        self.assertFalse(st["on_disk"]); self.assertTrue(st["regenerable"])
        with self.assertRaises(ValueError):
            rp.delete_stack(self.conn, self.pid)   # already gone
        seen = []
        out = rp.regenerate_stack(self.conn, self.project, self.pid, progress=lambda d, t, m: seen.append((d, t)))
        self.assertEqual(out["particles"], 5)
        with open(path, "rb") as fh:
            self.assertEqual(fh.read(), before)
        self.assertFalse(os.path.exists(path + ".regenerating"))
        self.assertEqual(seen[-1], (5, 5))
        with self.assertRaises(ValueError):
            rp.regenerate_stack(self.conn, self.project, self.pid)   # already on disk
        # the package and its refinement are untouched throughout
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM REFINEMENT_LIST WHERE REFINEMENT_PACKAGE_ASSET_ID=?", (self.pid,)).fetchone()[0], 1)

    def test_a_package_from_before_the_source_column_counts_as_picks_by_its_first_refinement(self):
        with self.conn as c:
            c.execute("UPDATE REFINEMENT_PACKAGE_ASSETS SET STACK_SOURCE=NULL WHERE REFINEMENT_PACKAGE_ASSET_ID=?", (self.pid,))
        self.assertTrue(rp.stack_status(self.conn, self.pid)["regenerable"])
        with self.conn as c:
            c.execute("UPDATE REFINEMENT_LIST SET NAME='Parameters from Refinement #3' WHERE REFINEMENT_PACKAGE_ASSET_ID=?", (self.pid,))
        st = rp.stack_status(self.conn, self.pid)
        self.assertFalse(st["regenerable"]); self.assertIn("not recorded", st["reason"])

    def test_a_moved_or_deleted_pick_a_missing_image_a_shared_file_and_an_import_refuse(self):
        with self.conn as c:
            c.execute("UPDATE PARTICLE_POSITION_ASSETS SET X_POSITION=X_POSITION+3 WHERE PARTICLE_POSITION_ASSET_ID=2")
        st = rp.stack_status(self.conn, self.pid)
        self.assertFalse(st["regenerable"]); self.assertIn("1 of its 5 particles", st["reason"])
        with self.assertRaises(ValueError):
            rp.delete_stack(self.conn, self.pid)
        with self.conn as c:
            c.execute("UPDATE PARTICLE_POSITION_ASSETS SET X_POSITION=X_POSITION-3 WHERE PARTICLE_POSITION_ASSET_ID=2")
            c.execute("DELETE FROM PARTICLE_POSITION_ASSETS WHERE PARTICLE_POSITION_ASSET_ID=5")
        self.assertIn("1 of its 5 particles", rp.stack_status(self.conn, self.pid)["reason"])
        with self.conn as c:
            c.execute("INSERT INTO PARTICLE_POSITION_ASSETS(PARTICLE_POSITION_ASSET_ID, PARENT_IMAGE_ASSET_ID, PICKING_ID, PICK_JOB_ID, X_POSITION, Y_POSITION, PEAK_HEIGHT, "
                      "TEMPLATE_ASSET_ID, TEMPLATE_PSI, TEMPLATE_THETA, TEMPLATE_PHI) VALUES (5, 2, 1, 'j', 96.0, 3.0, 8, -1, 0, 0, 0)")
        self.assertTrue(rp.stack_status(self.conn, self.pid)["regenerable"])
        img2 = self.conn.execute("SELECT FILENAME FROM IMAGE_ASSETS WHERE IMAGE_ASSET_ID=2").fetchone()[0]
        os.rename(img2, img2 + ".away")
        self.assertIn("image 2 is missing", rp.stack_status(self.conn, self.pid)["reason"])
        os.rename(img2 + ".away", img2)
        with self.conn as c:
            c.execute("INSERT INTO REFINEMENT_PACKAGE_ASSETS(REFINEMENT_PACKAGE_ASSET_ID, NAME, STACK_FILENAME, STACK_BOX_SIZE, OUTPUT_PIXEL_SIZE, SYMMETRY, NUMBER_OF_CLASSES, STACK_SOURCE) "
                      "VALUES (9, 'over', ?, 32, 1.5, 'C1', 1, 'package')", (self.res["stack_filename"],))
        st = rp.stack_status(self.conn, self.pid)
        self.assertFalse(st["regenerable"]); self.assertIn("package 9 shares", st["reason"])
        self.assertIn("another package's particles", rp.stack_status(self.conn, 9)["reason"])
        with self.conn as c:
            c.execute("UPDATE REFINEMENT_PACKAGE_ASSETS SET STACK_SOURCE='import' WHERE REFINEMENT_PACKAGE_ASSET_ID=9")
        self.assertIn("imported", rp.stack_status(self.conn, 9)["reason"])

    def test_the_routes_refuse_while_a_job_reads_the_package(self):
        base = "/api/projects/{}/refinement-packages/{}/stack".format(self.project, self.pid)
        st = self.client.get(base, headers=self.headers).get_json()
        self.assertTrue(st["regenerable"]); self.assertEqual(st["in_use"], [])
        with self.conn as c:
            c.execute("INSERT INTO JOBS(JOB_ID, STAGE, JOB_NUMBER, NAME, PARAMS_JSON, STATUS, CREATED_AT) VALUES ('busy', 'refine3d', 4, 'Job 4', ?, 'running', 'c')",
                      (json.dumps({"refinement_package_id": str(self.pid)}),))
        self.assertEqual(self.client.get(base, headers=self.headers).get_json()["in_use"], [4])
        r = self.client.delete(base, headers=self.headers)
        self.assertEqual(r.status_code, 409); self.assertIn("job 4 is using", r.get_json()["error"])
        with self.conn as c:
            c.execute("UPDATE JOBS SET STATUS='completed' WHERE JOB_ID='busy'")
        r = self.client.delete(base, headers=self.headers)
        self.assertEqual(r.status_code, 200); self.assertGreater(r.get_json()["freed"], 1024)
        self.assertEqual(self.client.delete(base, headers=self.headers).status_code, 409)   # not on disk now
        r = self.client.post(base + "/regenerate", headers=self.headers)
        self.assertEqual(r.status_code, 202)
        task = r.get_json()["task_id"]
        for _ in range(200):
            t = self.client.get("/api/projects/{}/tasks/{}".format(self.project, task), headers=self.headers).get_json()
            if t["state"] != "running":
                break
            import time; time.sleep(0.05)
        self.assertEqual(t["state"], "done", t)
        self.assertEqual(t["result"]["particles"], 5)
        self.assertEqual(self.client.post(base + "/regenerate", headers=self.headers).status_code, 409)   # on disk again


if __name__ == "__main__":
    unittest.main()
