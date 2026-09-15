"""Refinement package export and import (server/package_io.py) against a
throwaway project: a small package written through the same helpers the
wizard uses, exported in every format and imported back."""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import db  # noqa: E402
import package_io as pio  # noqa: E402
import refinement_packages as rp  # noqa: E402
import refinements  # noqa: E402
import starfile  # noqa: E402

BOX = 16
N = 5


class PackageIOTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._root = db.PROJECTS_ROOT
        db.PROJECTS_ROOT = Path(self.tmp)
        self.project = "t-" + os.path.basename(self.tmp)[-6:]
        (db.PROJECTS_ROOT / self.project).mkdir(parents=True)
        self.stack = os.path.join(self.tmp, "stack.mrc")
        w = rp.MrcStackWriter(self.stack, BOX, 1.25)
        rng = np.random.RandomState(0)
        for i in range(N):
            img = rng.normal(size=(BOX, BOX)).astype(np.float32)
            img[BOX // 2, BOX // 2] = -5.0 * (i + 1)   # a dark "particle" in the middle
            w.append(img)
        w.close()
        conn = db.get_conn(self.project)
        contained = [{"position_id": 100 + i, "image_id": -1, "position_in_stack": i + 1, "x": 0.0, "y": 0.0, "pixel_size": 1.25,
                      "defocus1": 15000.0 + i, "defocus2": 14000.0 + i, "defocus_angle": 30.0, "phase_shift": 0.0, "cs": 2.7, "voltage": 300.0,
                      "amplitude_contrast": 0.07, "subset": 1 + i % 2} for i in range(N)]
        rp.insert_package(conn, 1, "P", self.stack, BOX, 1.25, "D2", 120.0, 80.0, 1, contained, 1)
        rows = [(i + 1, 10.0 * i, 20.0 + i, -30.0 + i, 1.5, -2.5, 15000.0 + i, 14000.0 + i, 30.0, 0.0, 100.0, -1234.0, 1.0, 12.5 + i, 1,
                 1.25, 300.0, 2.7, 0.07, 0.0, 0.0, 0.0, 0.0, 1 + i % 2) for i in range(N)]
        rp.insert_initial_refinement(conn, 1, 1, "Random Parameters", [rows], BOX, 1.25, 120.0)
        conn.close()

    def tearDown(self):
        db.PROJECTS_ROOT = self._root
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _export(self, fmt, name):
        conn = db.get_conn(self.project)
        try:
            return pio.export_package(conn, 1, 1, 1, fmt, os.path.join(self.tmp, name + "_stack"), os.path.join(self.tmp, name + "_params"))
        finally:
            conn.close()

    def _import(self, **params):
        conn = db.get_conn(self.project)
        try:
            return pio.import_package(conn, self.project, params), conn
        except Exception:
            conn.close()
            raise

    def test_frealign_export_and_import(self):
        r = self._export("frealign", "fa")
        par, stack = r["files"]
        self.assertTrue(par.endswith(".par") and stack.endswith(".mrc"))
        with open(par) as fh:
            lines = fh.read().splitlines()
        self.assertTrue(lines[0].startswith("C           PSI   THETA"))
        data = [l for l in lines if not l.startswith("C")]
        self.assertEqual(len(data), N)
        first = data[0].split()
        self.assertEqual(first[0], "1")
        self.assertEqual((float(first[1]), float(first[2]), float(first[3])), (0.0, 20.0, -30.0))
        self.assertEqual(float(first[8]), 15000.0)
        self.assertEqual(len(first), 17)
        self.assertIn("Total particles included", lines[-1])
        self.assertEqual(os.path.getsize(stack), os.path.getsize(self.stack))
        # back in: Frealign carries no imaging parameters, so those come from the form
        res, conn = self._import(format="frealign", stack_path=stack, metadata_path=par, symmetry="D2", molecular_weight_kda=120,
                                 largest_dimension_a=80, pixel_size_a=1.25, voltage_kv=300, cs_mm=2.7, amplitude_contrast=0.07)
        try:
            self.assertEqual((res["particles"], res["box_size"], res["output_pixel_size"]), (N, BOX, 1.25))
            self.assertEqual(res["name"], "Refinement Package #2 (Frealign Import)")
            rows = refinements.load_rows(conn, res["refinement_id"], 1)
            self.assertEqual([r["theta"] for r in rows], [20.0 + i for i in range(N)])
            self.assertEqual(rows[0]["defocus_1"], 15000.0)
            self.assertEqual(rows[0]["pixel_size"], 1.25)
            self.assertEqual(refinements.refinement_row(conn, res["refinement_id"])["NAME"], "Imported Parameters")
            self.assertEqual(conn.execute("SELECT NUMBER_OF_REFINEMENTS FROM REFINEMENT_PACKAGE_ASSETS WHERE REFINEMENT_PACKAGE_ASSET_ID=2").fetchone()[0], 0)
            self.assertTrue(rp._table_exists(conn, "REFINEMENT_ANGULAR_DISTRIBUTION_{}_1".format(res["refinement_id"])))
        finally:
            conn.close()

    def test_cistem_star_round_trip(self):
        r = self._export("cistem", "cs")
        star, stack = r["files"]
        rows = starfile.read_star(star)
        self.assertEqual(len(rows), N)
        self.assertEqual(rows[2]["psi"], 20.0)
        self.assertEqual(rows[0]["assigned_subset"], 1)
        res, conn = self._import(format="cistem", stack_path=stack, metadata_path=star, symmetry="D2", molecular_weight_kda=120, largest_dimension_a=80, cs_mm=2.7)
        try:
            self.assertEqual(res["name"], "Refinement Package #2 (cisTEM Import)")
            back = refinements.load_rows(conn, res["refinement_id"], 1)
            for k in ("psi", "theta", "phi", "x_shift", "y_shift", "defocus_1", "defocus_2", "score", "assigned_subset", "voltage", "pixel_size"):
                self.assertEqual([b[k] for b in back], [r[k] for r in refinements.load_rows(conn, 1, 1)], k)
            contained = conn.execute("SELECT * FROM REFINEMENT_PACKAGE_CONTAINED_PARTICLES_2 ORDER BY POSITION_IN_STACK").fetchall()
            self.assertEqual([c["PARENT_IMAGE_ASSET_ID"] for c in contained], [-1] * N)
            self.assertEqual(contained[1]["ASSIGNED_SUBSET"], 2)
        finally:
            conn.close()

    def test_relion3_export_and_relion_import(self):
        r = self._export("relion3", "r3")
        self.assertEqual(os.path.basename(r["files"][0]), "r3_stack.mrcs")
        star = [f for f in r["files"] if f.endswith("r3_params.star")][0]
        with open(star) as fh:
            text = fh.read()
        self.assertIn("data_optics", text)
        self.assertIn("_rlnOriginXAngst #17", text)
        self.assertIn("_rlnRandomSubset #20", text)
        self.assertTrue(any(f.endswith("_corrected_micrographs.star") for f in r["files"]))
        # the stack: white protein now, sigma 1 from the pixels outside the particle
        img = rp.read_mrc_section(r["files"][0], 1)
        self.assertGreater(img[BOX // 2, BOX // 2], 0.0)
        # a particle line: phi theta psi then -shifts in angstroms
        lines = [l for l in text.splitlines() if l.startswith("unknown.mrc")]
        self.assertEqual(len(lines), N)
        tok = lines[0].split()
        self.assertEqual(tok[3], "000001@r3_stack.mrcs")
        self.assertEqual((float(tok[13]), float(tok[14]), float(tok[15])), (-30.0, 20.0, 0.0))
        self.assertEqual((float(tok[16]), float(tok[17])), (-1.5, 2.5))
        res, conn = self._import(format="relion", stack_path=r["files"][0], metadata_path=star, symmetry="C1", molecular_weight_kda=120,
                                 largest_dimension_a=80, pixel_size_a=1.25, voltage_kv=300, cs_mm=2.7, amplitude_contrast=0.07, protein_is_white=True)
        try:
            self.assertEqual(res["name"], "Refinement Package #2 (Relion Import)")
            back = refinements.load_rows(conn, res["refinement_id"], 1)
            self.assertEqual((back[0]["phi"], back[0]["theta"], back[0]["psi"]), (-30.0, 20.0, 0.0))
            self.assertEqual((back[0]["x_shift"], back[0]["y_shift"]), (1.5, -2.5))     # negated back, already in angstroms
            self.assertEqual((back[0]["occupancy"], back[0]["sigma"], back[0]["image_is_active"]), (100.0, 10.0, 1))
            self.assertEqual([b["assigned_subset"] for b in back], [1, 2, 1, 2, 1])
            self.assertEqual(conn.execute("SELECT STACK_HAS_WHITE_PROTEIN FROM REFINEMENT_PACKAGE_ASSETS WHERE REFINEMENT_PACKAGE_ASSET_ID=2").fetchone()[0], 1)
        finally:
            conn.close()

    def test_relion2_shifts_are_pixels(self):
        r = self._export("relion", "r2")
        star = r["files"][1]
        with open(star) as fh:
            text = fh.read()
        self.assertNotIn("data_optics", text)
        tok = [l for l in text.splitlines() if l.startswith("unknown.mrc")][0].split()
        self.assertEqual((float(tok[16]), float(tok[17])), (-1.5 / 1.25, 2.5 / 1.25))
        rows, in_angst = pio.read_relion_star(star)
        self.assertFalse(in_angst)
        self.assertEqual(rows[0]["x_shift"], -1.5 / 1.25)

    def test_import_checks(self):
        with self.assertRaises(ValueError):
            self._import(format="cistem", stack_path=self.stack, metadata_path="/nonexistent.star")
        star = os.path.join(self.tmp, "short.star")
        starfile.write_star(star, [{"position_in_stack": 1, "pixel_size": 1.25}])
        with self.assertRaises(ValueError) as cm:
            self._import(format="cistem", stack_path=self.stack, metadata_path=star)
        self.assertIn("different from the number of lines", str(cm.exception))
        with self.assertRaises(ValueError):
            self._import(format="frealign", stack_path=self.stack, metadata_path=star)   # pixel size missing
        # a bad class, refinement or output folder is refused
        conn = db.get_conn(self.project)
        try:
            with self.assertRaises(ValueError):
                pio.export_package(conn, 1, 1, 3, "frealign", os.path.join(self.tmp, "a"), os.path.join(self.tmp, "b"))
            with self.assertRaises(ValueError):
                pio.export_package(conn, 1, 99, 1, "frealign", os.path.join(self.tmp, "a"), os.path.join(self.tmp, "b"))
            with self.assertRaises(ValueError):
                pio.export_package(conn, 1, 1, 1, "frealign", os.path.join(self.tmp, "nodir", "a"), os.path.join(self.tmp, "b"))
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
