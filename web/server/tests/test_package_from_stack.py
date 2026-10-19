"""A package from 2D class selections whose particles have no parent image (an
imported stack): the particles are copied from the stack, cropped or padded to
the new box and moved by the classification's shift, with their CTF and
half-set carried over."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import classification as c2d  # noqa: E402
import db  # noqa: E402
import refinement_packages as rp  # noqa: E402

BOX, N = 24, 6


class FromStackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._root = db.PROJECTS_ROOT
        db.PROJECTS_ROOT = Path(self.tmp) / "projects"
        self.project = "t-stack"
        (db.PROJECTS_ROOT / self.project).mkdir(parents=True)
        self.conn = db.get_conn(self.project)
        # an imported stack: particle i is a ramp plus a bright pixel at (row 14, col 16), parent image -1
        self.stack = os.path.join(self.tmp, "imported.mrc")
        w = rp.MrcStackWriter(self.stack, BOX, 1.5)
        self.particles = []
        for i in range(N):
            img = (np.arange(BOX * BOX, dtype=np.float32).reshape(BOX, BOX) / 100.0 + i).astype(np.float32)
            img[14, 16] = 50.0 + i
            self.particles.append(img); w.append(img)
        w.close()
        contained = [{"position_id": i + 1, "image_id": -1, "position_in_stack": i + 1, "x": 0.0, "y": 0.0, "pixel_size": 1.5,
                      "defocus1": 15000.0 + i, "defocus2": 14000.0 + i, "defocus_angle": 30.0, "phase_shift": 0.0, "cs": 2.7, "voltage": 300.0,
                      "amplitude_contrast": 0.07, "subset": 2 if i % 3 == 0 else 1} for i in range(N)]
        rp.insert_package(self.conn, 1, "Imported", self.stack, BOX, 1.5, "C1", 120.0, 80.0, 1, contained, 1)
        rp.insert_initial_refinement(self.conn, 1, 1, "Random", [[(i + 1, 0, 0, 0, 0, 0, 15000.0 + i, 14000.0 + i, 30.0, 0.0, 100.0, 0, 1, 0, 1, 1.5, 300.0, 2.7, 0.07, 0, 0, 0, 0, 1) for i in range(N)]],
                                     BOX, 1.5, 120.0)
        # a classification: even particles in class 1 (shifted by +3 A in x = +2 px), odd in class 2
        cls = {"classification_id": 1, "refinement_package_asset_id": 1, "name": "C", "class_average_file": "/nonexistent.mrc", "number_of_particles": N,
               "number_of_classes": 2, "low_resolution_limit": 300, "high_resolution_limit": 8, "mask_radius": 100, "angular_search_step": 15,
               "search_range_x": 100, "search_range_y": 100, "smoothing_factor": 1, "exclude_blank_edges": False, "auto_percent_used": True, "percent_used": 100}
        rows = [dict(c2d.empty_result(i + 1), best_2d_class=1 + i % 2, x_shift=3.0 if i % 2 == 0 else 0.0, y_shift=0.0) for i in range(N)]
        c2d.add_classification(self.conn, cls, rows)
        self.selection = c2d.create_selection(self.conn, 1, "picked", [1])

    def tearDown(self):
        self.conn.close()
        db.PROJECTS_ROOT = self._root

    def _create(self, **extra):
        return rp.create_package(self.conn, self.project, dict({"selection_ids": [self.selection], "name": "FromStack", "output_pixel_size": 1.5}, **extra))

    def test_particles_are_copied_padded_and_shifted(self):
        out = self._create(box_size=32)
        self.assertEqual((out["particles"], out["copied"]), (3, 3))
        for n, i in enumerate((0, 2, 4), 1):
            got = rp.read_mrc_section(out["stack_filename"], n)
            self.assertEqual(got.shape, (32, 32))
            # padded by 4 on each side and moved by the shift: the pick moves by -3 A = -2 px, so the particle's centre
            # (src 12) is cut at column 12 - 2 = 10 -> the bright pixel at col 16 lands at 16 - 10 + 16 = 22, row 14 + 4 = 18
            self.assertEqual(float(got[18, 22]), 50.0 + i)
            edge = float(np.concatenate([self.particles[i][0], self.particles[i][-1], self.particles[i][1:-1, 0], self.particles[i][1:-1, -1]]).mean())
            self.assertAlmostEqual(float(got[0, 0]), edge, places=4)   # the padding is the particle's edge average
        rows = self.conn.execute("SELECT * FROM REFINEMENT_PACKAGE_CONTAINED_PARTICLES_{} ORDER BY POSITION_IN_STACK".format(out["refinement_package_asset_id"])).fetchall()
        self.assertEqual([r["PARENT_IMAGE_ASSET_ID"] for r in rows], [-1] * 3)
        self.assertEqual([r["DEFOCUS_1"] for r in rows], [15000.0, 15002.0, 15004.0])        # the source's CTF
        self.assertEqual([r["ASSIGNED_SUBSET"] for r in rows], [2, 1, 1])                   # and half-set
        self.assertEqual([r["X_POSITION"] for r in rows], [-3.0] * 3)                        # re-centred position
        self.assertTrue(all(r["ORIGINAL_PARTICLE_POSITION_ASSET_ID"] < 0 for r in rows))     # not position assets
        pkg = self.conn.execute("SELECT STACK_BOX_SIZE, OUTPUT_PIXEL_SIZE FROM REFINEMENT_PACKAGE_ASSETS WHERE REFINEMENT_PACKAGE_ASSET_ID=?", (out["refinement_package_asset_id"],)).fetchone()
        self.assertEqual((pkg[0], pkg[1]), (32, 1.5))

    def test_a_smaller_box_crops_without_recentring(self):
        out = self._create(box_size=16, recentre=False)
        got = rp.read_mrc_section(out["stack_filename"], 1)
        np.testing.assert_array_equal(got, self.particles[0][4:20, 4:20])

    def test_a_different_pixel_size_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            self._create(output_pixel_size=1.0)
        self.assertIn("pixel size", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
