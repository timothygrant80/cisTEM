"""create_package(): the batched, read-ahead cutting gives the same stack as
cutting one particle at a time the way the wizard does (ClipInto with the
edge mean, ReplaceOutliersWithMean(6), zero-float-and-normalise), progress
reports both phases, and a project's running creations are listed for a
reloaded page to pick up."""
import json
import math
import os
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import auth  # noqa: E402
import db  # noqa: E402
import refinement_packages as rp  # noqa: E402


def _write_mrc(path, image):
    head = bytearray(1024)
    ny, nx = image.shape
    struct.pack_into("<iiii", head, 0, nx, ny, 1, 2); struct.pack_into("<iii", head, 28, nx, ny, 1)
    struct.pack_into("<fff", head, 40, nx, ny, 1); head[208:212] = b"MAP "; head[212:216] = b"\x44\x44\x00\x00"
    with open(path, "wb") as fh:
        fh.write(bytes(head)); fh.write(image.astype("<f4").tobytes())


def _reference_stack(image, picks, box):
    """The wizard's arithmetic, one particle at a time."""
    mean, sigma = float(image.mean()), float(image.std())
    image = np.where(np.abs(image - mean) > 6.0 * sigma, np.float32(mean), image)
    edge = float(np.concatenate([image[0, :], image[-1, :], image[1:-1, 0], image[1:-1, -1]]).mean())
    out = []
    for cx, cy in picks:
        half = box // 2
        r0, c0 = cy - half, cx - half
        b = np.full((box, box), edge, dtype=np.float32)
        ry0, ry1, rx0, rx1 = max(r0, 0), min(r0 + box, image.shape[0]), max(c0, 0), min(c0 + box, image.shape[1])
        b[ry0 - r0:ry1 - r0, rx0 - c0:rx1 - c0] = image[ry0:ry1, rx0:rx1]
        m, v = float(b.mean()), float(b.var())
        out.append((b - m) / math.sqrt(v) if v > 0 else b - m)
    return np.stack(out)


class CreatePackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._root = db.PROJECTS_ROOT
        db.PROJECTS_ROOT = Path(self.tmp) / "projects"
        self.project = "t-pkg"
        (db.PROJECTS_ROOT / self.project).mkdir(parents=True)
        self.conn = db.get_conn(self.project)
        rng = np.random.default_rng(3)
        self.images = {}
        with self.conn as c:
            for i in (1, 2):
                img = (rng.standard_normal((96, 128)) * 10 + 100).astype(np.float32)
                img[5, 7] = 5000.0   # an outlier the 6-sigma pass replaces
                path = os.path.join(self.tmp, "img%d.mrc" % i); _write_mrc(path, img); self.images[i] = img
                c.execute("INSERT INTO IMAGE_ASSETS(IMAGE_ASSET_ID, NAME, FILENAME, X_SIZE, Y_SIZE, PIXEL_SIZE, VOLTAGE, SPHERICAL_ABERRATION, PARENT_MOVIE_ID, CTF_ESTIMATION_ID) "
                          "VALUES (?,?,?,128,96,1.5,300,2.7,-1,?)", (i, "img%d" % i, path, i))
                c.execute("INSERT INTO ESTIMATED_CTF_PARAMETERS(CTF_ESTIMATION_ID, IMAGE_ASSET_ID, VOLTAGE, SPHERICAL_ABERRATION, AMPLITUDE_CONTRAST, DEFOCUS1, DEFOCUS2, "
                          "DEFOCUS_ANGLE, ADDITIONAL_PHASE_SHIFT, PIXEL_SIZE) VALUES (?,?,300,2.7,0.07,15000,14000,10,0,1.5)", (i, i))
            c.execute("INSERT OR IGNORE INTO PARTICLE_POSITION_GROUP_LIST(GROUP_ID, GROUP_NAME, LIST_ID) VALUES (0, 'All', 0)")
            # picks in pixels (x, y): interior ones and one hanging off each edge of image 2
            self.picks = {1: [(40, 30), (100, 60)], 2: [(3, 50), (120, 90), (64, 2)]}
            pid = 0
            for i, picks in self.picks.items():
                for cx, cy in picks:
                    pid += 1
                    c.execute("INSERT INTO PARTICLE_POSITION_ASSETS(PARTICLE_POSITION_ASSET_ID, PARENT_IMAGE_ASSET_ID, PICKING_ID, PICK_JOB_ID, X_POSITION, Y_POSITION, PEAK_HEIGHT, "
                              "TEMPLATE_ASSET_ID, TEMPLATE_PSI, TEMPLATE_THETA, TEMPLATE_PHI) VALUES (?,?,1,'j',?,?,8,-1,0,0,0)", (pid, i, cx * 1.5, cy * 1.5))
                    c.execute("INSERT INTO PARTICLE_POSITION_GROUP_MEMBERS(GROUP_ID, PARTICLE_POSITION_ASSET_ID) VALUES (0, ?)", (pid,))

    def tearDown(self):
        self.conn.close()
        db.PROJECTS_ROOT = self._root

    def test_batched_cutting_matches_the_wizard_arithmetic(self):
        seen = []
        res = rp.create_package(self.conn, self.project, {"particle_group_id": 0, "box_size": 32, "number_of_classes": 1},
                                progress=lambda d, t, m: seen.append((d, t, m)))
        self.assertEqual(res["particles"], 5)
        with open(res["stack_filename"], "rb") as fh:
            head = fh.read(1024)
            nx, ny, nz, mode = struct.unpack_from("<iiii", head, 0)
            stack = np.frombuffer(fh.read(), dtype="<f4").reshape(nz, ny, nx)
        self.assertEqual((nx, ny, nz, mode), (32, 32, 5, 2))
        expected = np.concatenate([_reference_stack(self.images[i], self.picks[i], 32) for i in (1, 2)])
        np.testing.assert_allclose(stack, expected, rtol=1e-4, atol=1e-4)
        dmin, dmax, dmean = struct.unpack_from("<fff", head, 76)
        self.assertAlmostEqual(dmin, float(expected.min()), places=3)
        self.assertAlmostEqual(dmax, float(expected.max()), places=3)
        self.assertAlmostEqual(dmean, float(expected.mean()), places=3)
        rows = self.conn.execute("SELECT POSITION_IN_STACK, PARENT_IMAGE_ASSET_ID, X_POSITION, DEFOCUS_1 FROM REFINEMENT_PACKAGE_CONTAINED_PARTICLES_{} ORDER BY POSITION_IN_STACK".format(
            res["refinement_package_asset_id"])).fetchall()
        self.assertEqual([tuple(r) for r in rows], [(1, 1, 60.0, 15000.0), (2, 1, 150.0, 15000.0), (3, 2, 4.5, 15000.0), (4, 2, 180.0, 15000.0), (5, 2, 96.0, 15000.0)])
        self.assertEqual(seen[-1], (5, 5, "Writing the package"))
        self.assertTrue(all(m.startswith("Cutting particles from image") for d, t, m in seen[:-1]))
        self.assertEqual(seen[-2][:2], (5, 5))

    def test_running_creations_are_listed_with_their_timing(self):
        import cistem_server
        auth_path = auth.AUTH_DB_PATH
        auth.AUTH_DB_PATH = Path(self.tmp) / "auth.db"
        try:
            token = auth.create_session(auth.create_user("boss", "password123", "admin")["id"])
            client = cistem_server.app.test_client()
            headers = {"Authorization": "Bearer " + token}
            with cistem_server._package_tasks_lock:
                cistem_server._package_tasks["aaa"] = {"state": "running", "done": 40, "total": 100, "message": "Cutting", "project_id": self.project, "started_at": time.time() - 8.0, "name": "Pkg"}
                cistem_server._package_tasks["bbb"] = {"state": "done", "done": 9, "total": 9, "message": "Done", "project_id": self.project, "started_at": time.time() - 90.0}
                cistem_server._package_tasks["ccc"] = {"state": "running", "done": 1, "total": 5, "message": "Cutting", "project_id": "other", "started_at": time.time()}
            try:
                r = client.get("/api/projects/{}/refinement-packages/tasks".format(self.project), headers=headers)
                self.assertEqual(r.status_code, 200, r.data)
                tasks = r.get_json()["tasks"]
                self.assertEqual([t["task_id"] for t in tasks], ["aaa"])
                self.assertEqual((tasks[0]["done"], tasks[0]["total"], tasks[0]["name"]), (40, 100, "Pkg"))
                self.assertGreaterEqual(tasks[0]["elapsed"], 8.0)
                one = client.get("/api/projects/{}/refinement-packages/tasks/aaa".format(self.project), headers=headers).get_json()
                self.assertEqual(one["task_id"], "aaa")
                self.assertGreaterEqual(one["elapsed"], 8.0)
            finally:
                with cistem_server._package_tasks_lock:
                    for k in ("aaa", "bbb", "ccc"):
                        cistem_server._package_tasks.pop(k, None)
        finally:
            auth.AUTH_DB_PATH = auth_path


if __name__ == "__main__":
    unittest.main()
