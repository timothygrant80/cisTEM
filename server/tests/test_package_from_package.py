"""refinement_packages.create_package_from_package: a package over another's
particles with a refinement's parameters carried over."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db  # noqa: E402
import refinement_packages as rp  # noqa: E402
import refinements  # noqa: E402

BOX, N = 16, 12


class FromPackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._root = db.PROJECTS_ROOT
        db.PROJECTS_ROOT = Path(self.tmp)
        self.project = "t-" + os.path.basename(self.tmp)[-6:]
        (db.PROJECTS_ROOT / self.project).mkdir(parents=True)
        self.stack = os.path.join(self.tmp, "stack.mrc")
        w = rp.MrcStackWriter(self.stack, BOX, 1.25)
        for i in range(N):
            img = np.full((BOX, BOX), float(i + 1), dtype=np.float32)
            w.append(img)
        w.close()
        self.conn = db.get_conn(self.project)
        contained = [{"position_id": 100 + i, "image_id": 1 + i % 3, "position_in_stack": i + 1, "x": 10.0 * i, "y": 5.0 * i, "pixel_size": 1.25,
                      "defocus1": 15000.0 + i, "defocus2": 14000.0 + i, "defocus_angle": 30.0, "phase_shift": 0.0, "cs": 2.7, "voltage": 300.0,
                      "amplitude_contrast": 0.07, "subset": 1 + i % 2} for i in range(N)]
        rp.insert_package(self.conn, 1, "Source", self.stack, BOX, 1.25, "C1", 120.0, 80.0, 2, contained, 1)
        # a two-class refinement: even particles belong to class 1, odd to class 2, distinct angles per class
        rows = []
        for k in (1, 2):
            rows.append([(i + 1, 10.0 * k + i, 20.0 + i, 30.0 * k, 1.5, -2.5, 15000.0 + i, 14000.0 + i, 30.0, 0.0,
                          90.0 if (i % 2 == k - 1) else 10.0, -1234.0, 1.0, 12.5 + i, 1, 1.25, 300.0, 2.7, 0.07, 0.0, 0.0, 0.0, 0.0, 1 + i % 2) for i in range(N)])
        rp.insert_initial_refinement(self.conn, 1, 1, "Refined", rows, BOX, 1.25, 120.0)

    def tearDown(self):
        self.conn.close()
        db.PROJECTS_ROOT = self._root

    def test_all_particles_share_the_stack_and_copy_the_parameters(self):
        out = rp.create_package_from_package(self.conn, self.project, {"source_package_id": 1, "source_refinement_id": 1, "name": "Copy",
                                                                          "symmetry": "D2", "number_of_classes": 3, "class_sources": [[1], [2], [1, 2]]})
        self.assertTrue(out["shared_stack"]); self.assertEqual(out["stack_filename"], self.stack); self.assertEqual(out["particles"], N)
        pkg = self.conn.execute("SELECT * FROM REFINEMENT_PACKAGE_ASSETS WHERE REFINEMENT_PACKAGE_ASSET_ID=2").fetchone()
        self.assertEqual((pkg["SYMMETRY"], pkg["STACK_BOX_SIZE"], pkg["NUMBER_OF_CLASSES"], pkg["STACK_FILENAME"]), ("D2", BOX, 3, self.stack))
        c1 = refinements.load_rows(self.conn, out["refinement_id"], 1)
        c2 = refinements.load_rows(self.conn, out["refinement_id"], 2)
        c3 = refinements.load_rows(self.conn, out["refinement_id"], 3)
        self.assertEqual([r["psi"] for r in c1], [10.0 + i for i in range(N)])        # class 1 from source class 1
        self.assertEqual([r["psi"] for r in c2], [20.0 + i for i in range(N)])        # class 2 from source class 2
        # class 3 from the better-occupied of the two: class 1 for even particles, class 2 for odd
        self.assertEqual([r["psi"] for r in c3], [(10.0 if i % 2 == 0 else 20.0) + i for i in range(N)])
        # occupancies randomised across the three classes, each within |U| * 200 / 3
        self.assertTrue(all(0.0 <= r["occupancy"] <= 200.0 / 3 + 1e-6 for r in c1 + c2 + c3))
        self.assertTrue(len({round(r["occupancy"], 3) for r in c1}) > 1)
        # the angular distribution exists for the new symmetry
        self.assertTrue(refinements.load_angular_distribution(self.conn, out["refinement_id"], 1))
        # deleting the source keeps the stack the copy still points at; deleting the copy too removes it
        rp.delete_package(self.conn, 1)
        self.assertTrue(os.path.isfile(self.stack))
        rp.delete_package(self.conn, 2)
        self.assertFalse(os.path.isfile(self.stack))

    def test_a_class_subset_is_cut_into_its_own_stack(self):
        out = rp.create_package_from_package(self.conn, self.project, {"source_package_id": 1, "source_refinement_id": 1, "carry_over_classes": [2],
                                                                          "number_of_classes": 1})
        self.assertFalse(out["shared_stack"]); self.assertNotEqual(out["stack_filename"], self.stack)
        self.assertEqual(out["particles"], N // 2)            # the odd particles, whose best class is 2
        parts = self.conn.execute("SELECT POSITION_IN_STACK, ORIGINAL_PARTICLE_POSITION_ASSET_ID FROM REFINEMENT_PACKAGE_CONTAINED_PARTICLES_2 ORDER BY POSITION_IN_STACK").fetchall()
        self.assertEqual([p[0] for p in parts], list(range(1, N // 2 + 1)))
        self.assertEqual([p[1] for p in parts], [100 + i for i in range(N) if i % 2 == 1])
        # the copied slices are the odd ones (each source slice was filled with its number)
        self.assertEqual(float(rp.read_mrc_section(out["stack_filename"], 1).mean()), 2.0)
        self.assertEqual(float(rp.read_mrc_section(out["stack_filename"], 2).mean()), 4.0)
        rows = refinements.load_rows(self.conn, out["refinement_id"], 1)
        self.assertEqual([r["occupancy"] for r in rows], [100.0] * (N // 2))
        self.assertEqual([r["position_in_stack"] for r in rows], list(range(1, N // 2 + 1)))

    def test_refinement_must_belong_to_the_source(self):
        with self.assertRaises(ValueError):
            rp.create_package_from_package(self.conn, self.project, {"source_package_id": 1, "source_refinement_id": 99})


if __name__ == "__main__":
    unittest.main()
