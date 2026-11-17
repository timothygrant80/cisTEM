"""The higher-order aberrations around the fit: the per-refinement sets in the project database, their
inheritance along the refinement chain, the aberration file the programs read, and the Refine CTF driver's
fit of a refinement from a synthetic sums file."""
import math
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import aberrations  # noqa: E402
import db  # noqa: E402
import refinement_packages as rp  # noqa: E402
import refinements  # noqa: E402
import refinectf  # noqa: E402
import volumes  # noqa: E402

BOX, N = 16, 6


def _rows(k_classes=1):
    rows = []
    for k in range(1, k_classes + 1):
        rows.append([(i + 1, 10.0 * k + i, 20.0 + i, 30.0 * k, 1.5, -2.5, 15000.0 + i, 14000.0 + i, 30.0, 0.0,
                      100.0, -1234.0, 1.0, 12.5 + i, 1, 1.25, 300.0, 2.7, 0.07, 0.0, 0.0, 0.0, 0.0, 1 + i % 2) for i in range(N)])
    return rows


class StorageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._root = db.PROJECTS_ROOT
        db.PROJECTS_ROOT = Path(self.tmp)
        self.project = "t-" + os.path.basename(self.tmp)[-6:]
        (db.PROJECTS_ROOT / self.project).mkdir(parents=True)
        self.stack = os.path.join(self.tmp, "stack.mrc")
        w = rp.MrcStackWriter(self.stack, BOX, 1.25)
        for i in range(N):
            w.append(np.full((BOX, BOX), float(i + 1), dtype=np.float32))
        w.close()
        self.conn = db.get_conn(self.project)
        contained = [{"position_id": 100 + i, "image_id": 1, "position_in_stack": i + 1, "x": 10.0 * i, "y": 5.0 * i, "pixel_size": 1.25,
                      "defocus1": 15000.0 + i, "defocus2": 14000.0 + i, "defocus_angle": 30.0, "phase_shift": 0.0, "cs": 2.7, "voltage": 300.0,
                      "amplitude_contrast": 0.07, "subset": 1 + i % 2} for i in range(N)]
        rp.insert_package(self.conn, 1, "Pkg", self.stack, BOX, 1.25, "C1", 120.0, 80.0, 1, contained, 1)
        rp.insert_initial_refinement(self.conn, 1, 1, "Random Parameters", _rows(), BOX, 1.25, 120.0)
        self.sets = [{"group": 0, "odd_order": 3, "even_order": 4, "odd": [0.1, -0.2, 30.0, 0.3, -0.4, 5.0], "even": [0.0] * 4 + [20.0, 0.0, 2.0, 0.0, -15.0]}]

    def tearDown(self):
        self.conn.close()
        db.PROJECTS_ROOT = self._root

    def _add(self, rid, starting, aberrations=None):
        ref = {"refinement_id": rid, "refinement_package_asset_id": 1, "name": "R{}".format(rid), "starting_refinement_id": starting,
               "number_of_particles": N, "number_of_classes": 1, "resolution_statistics_box_size": BOX, "resolution_statistics_pixel_size": 1.25,
               "percent_used": 100.0, "job_id": None}
        if aberrations is not None:
            ref["aberrations"] = aberrations
        class_rows = [refinements.load_table(self.conn, starting, 1)]
        refinements.add_refinement(self.conn, ref, class_rows, [[]], [{}], angular=False)

    def test_ctf_refinement_result_from_a_finished_job(self):
        """The Refine 3D results page's CTF section: the finished job's state, with
        the picture paths replaced by the list of pictures on disk."""
        import json
        pic = os.path.join(self.tmp, "refinement_2_phase_difference.mrc")
        volumes.write_mrc_volume(pic, np.zeros((1, 4, 4), np.float32), 1.25)
        state = {"settings": {"refine_defocus": True, "refine_beam_tilt": True, "defocus_search_range_a": 500.0, "defocus_search_step_a": 50.0,
                              "low_resolution_limit_a": 50.0, "high_resolution_limit_a": 3.5},
                 "zernike": True, "defocus_histogram": {"centres": [-50.0, 0.0, 50.0], "counts": [1, 2, 1], "n": 4},
                 "beam_tilt": {"beam_tilt_x_mrad": 0.1, "beam_tilt_y_mrad": -0.2, "particle_shift_x_a": 0.0, "particle_shift_y_a": 0.0, "method": "zernike",
                               "significant": True, "phase_difference_file": pic, "beam_tilt_file": os.path.join(self.tmp, "missing.mrc")},
                 "aberrations": {"summary": {"odd_phase_rms": 0.1}, "sets": self.sets, "files": {"odd_measured": pic, "odd_fitted": os.path.join(self.tmp, "missing.mrc")}}}
        with self.conn:
            self.conn.execute("INSERT INTO JOBS(JOB_ID, STAGE, JOB_NUMBER, NAME, STATUS, PROGRESS, CREATED_AT, STATE_JSON) VALUES ('j1', ?, 7, 'Job 7', 'completed', 100, 'now', ?)",
                              (refinectf.STAGE, json.dumps(state)))
        row = self.conn.execute("SELECT * FROM JOBS WHERE JOB_ID='j1'").fetchone()
        got = refinectf.ctf_refinement_result(self.conn, row)
        self.assertEqual(got["job_number"], 7)
        self.assertEqual(got["defocus_histogram"]["n"], 4)
        self.assertTrue(got["refine_defocus"] and got["zernike"])
        self.assertEqual(got["beam_tilt"]["beam_tilt_y_mrad"], -0.2)
        self.assertNotIn("phase_difference_file", got["beam_tilt"])
        self.assertEqual(got["pictures"], ["phase_difference", "odd_measured"])
        self.assertEqual(got["aberrations"]["sets"], self.sets)
        self.assertEqual(got["defocus_search_step_a"], 50.0)
        # a job whose state is gone gives nothing rather than failing the refinement
        with self.conn:
            self.conn.execute("UPDATE JOBS SET STATE_JSON=NULL WHERE JOB_ID='j1'")
        self.assertIsNone(refinectf.ctf_refinement_result(self.conn, self.conn.execute("SELECT * FROM JOBS WHERE JOB_ID='j1'").fetchone()))

    def test_save_load_and_json(self):
        self.assertEqual(refinements.load_aberrations(self.conn, 1), [])
        refinements.save_aberrations(self.conn, 1, self.sets)
        got = refinements.load_aberrations(self.conn, 1)
        self.assertEqual(got, self.sets)
        got_json = refinements.get_refinement(self.conn, 1)["aberrations"]
        self.assertEqual([{k: v for k, v in st.items() if k in self.sets[0]} for st in got_json], self.sets)
        self.assertIn("beam_tilt_x_mrad", got_json[0])     # the JSON adds the tilt the odd set contains
        refinements.save_aberrations(self.conn, 1, [])
        self.assertEqual(refinements.load_aberrations(self.conn, 1), [])

    def test_a_cs_of_zero_lists_the_tilt_as_null(self):
        # Apo2, 2026-11-16: the list and the refinement derived the tilt by dividing by the particles' Cs,
        # which was 0, and the Infinity that wrote left the Refine 3D results page on "Loading...".
        self._add(2, 1, self.sets)
        self.conn.execute("UPDATE REFINEMENT_RESULT_2_1 SET MICROSCOPE_CS=0.0")
        for d in (refinements.get_refinement(self.conn, 2), [r for r in refinements.list_refinements(self.conn, None) if r["refinement_id"] == 2][0]):
            st = d["aberrations"][0]
            self.assertIsNone(st["beam_tilt_x_mrad"])
            self.assertIsNone(st["beam_tilt_y_mrad"])
            self.assertAlmostEqual(st["particle_shift_x_a"], (-0.2 - 2.0 * -0.4) / (2.0 * math.pi))
            json.dumps(d, allow_nan=False)
        # and with a Cs the tilt is a number
        self.conn.execute("UPDATE REFINEMENT_RESULT_2_1 SET MICROSCOPE_CS=2.7")
        self.assertNotEqual(refinements.get_refinement(self.conn, 2)["aberrations"][0]["beam_tilt_x_mrad"], None)

    def test_a_refinement_inherits_its_input_sets_unless_given_new_ones(self):
        refinements.save_aberrations(self.conn, 1, self.sets)
        self._add(2, 1)
        self.assertEqual(refinements.load_aberrations(self.conn, 2), self.sets)
        new = [{"group": 0, "odd_order": 3, "even_order": 0, "odd": [0.0] * 6, "even": []}]
        self._add(3, 2, aberrations=new)
        self.assertEqual(refinements.load_aberrations(self.conn, 3), new)
        self._add(4, 3, aberrations=[])      # explicitly none: nothing inherited
        self.assertEqual(refinements.load_aberrations(self.conn, 4), [])
        self._add(5, 4)
        self.assertEqual(refinements.load_aberrations(self.conn, 5), [])

    def test_the_aberration_file(self):
        path = Path(self.tmp) / "aberrations.txt"
        self.assertEqual(refinements.write_aberration_file(self.conn, 1, path), "none")
        self.assertEqual(refinements.write_aberration_file(self.conn, -1, path), "none")
        self.assertFalse(path.exists())
        refinements.save_aberrations(self.conn, 1, self.sets)
        self.assertEqual(refinements.write_aberration_file(self.conn, 1, path), str(path))
        lines = [l for l in path.read_text().splitlines() if not l.startswith("#")]
        self.assertEqual(len(lines), 1)
        fields = lines[0].split()
        self.assertEqual(fields[:2], ["0", "6"])
        self.assertEqual([float(v) for v in fields[2:8]], self.sets[0]["odd"])
        self.assertEqual(fields[8], "9")
        self.assertEqual([float(v) for v in fields[9:]], self.sets[0]["even"])
        # an all-zero set counts as none
        self.assertEqual(refinements.write_aberration_sets([{"group": 0, "odd": [0.0] * 6, "even": []}], path), "none")

    def test_the_relion_export_writes_the_sets_in_the_optics_group(self):
        import package_io
        refinements.save_aberrations(self.conn, 1, self.sets)
        stack, star = os.path.join(self.tmp, "export.mrcs"), os.path.join(self.tmp, "export.star")
        package_io.export_package(self.conn, 1, 1, 1, "relion3", stack, star)
        text = Path(star).read_text()
        optics = text.split("data_particles")[0]
        self.assertIn("_rlnOddZernike", optics)
        self.assertIn("_rlnEvenZernike", optics)
        self.assertIn("[0.1,-0.2,30,0.3,-0.4,5]", optics)
        self.assertNotIn("_rlnBeamTiltX", optics)
        # without a set, the rows' beam tilt (zero here) is not written either
        refinements.save_aberrations(self.conn, 1, [])
        package_io.export_package(self.conn, 1, 1, 1, "relion3", stack, star)
        self.assertNotIn("Zernike", Path(star).read_text().split("data_particles")[0])

    def test_a_package_made_from_a_refinement_carries_its_sets(self):
        refinements.save_aberrations(self.conn, 1, self.sets)
        rp.create_package_from_package(self.conn, self.project, {"source_package_id": 1, "source_refinement_id": 1, "name": "Copy", "molecular_weight": 120.0,
                                                   "largest_dimension": 80.0, "symmetry": "C1", "number_of_classes": 1, "box_size": BOX, "output_pixel_size": 1.25})
        new_rid = self.conn.execute("SELECT MAX(REFINEMENT_ID) FROM REFINEMENT_LIST").fetchone()[0]
        self.assertGreater(new_rid, 1)
        self.assertEqual(refinements.load_aberrations(self.conn, new_rid), self.sets)


class DriverFitTest(unittest.TestCase):
    """fit_refinement_aberrations on a sums file built from a known set."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.box, self.pixel = 96, 1.2
        self.odd_truth = aberrations.odd_coefficients_for_beam_tilt(0.8, -0.3, 0.0, 0.0, 2.7, 300.0) + np.array([0, 0, 40.0, 0, 0, -30.0])
        self.even_truth = np.zeros(9)
        self.even_truth[4], self.even_truth[8] = 80.0, -60.0
        self.sums_path = os.path.join(self.tmp, "sums.mrc")
        self._write_sums(self.sums_path, self.odd_truth, self.even_truth, 5)
        self.settings = dict(refinectf.DEFAULTS, high_resolution_limit_a=3.0, low_resolution_limit_a=60.0)

    def _write_sums(self, path, odd_truth, even_truth, seed):
        """A sums file in refine_ctf's layout from a known set."""
        rng = np.random.default_rng(seed)
        rho, phi = aberrations.half_plane_frequencies(self.box, self.box, self.pixel)
        Zo, Ze = aberrations.basis(3, True, rho, phi), aberrations.basis(4, False, rho, phi)
        odd_phase, delta = np.tensordot(odd_truth, Zo, 1), np.tensordot(even_truth, Ze, 1)
        sums = np.zeros((7,) + rho.shape)
        for _ in range(40):
            gamma = math.pi * 0.02 * rng.uniform(8000, 20000) * rho ** 2 + rng.uniform(0, 0.3)
            a, b = -np.sin(gamma), np.cos(gamma)
            V = rng.uniform(0.5, 2.0, rho.shape)
            X = -np.sin(gamma - delta) * V + rng.normal(0, 0.2, rho.shape)
            sums[0] += a * a * V * V
            sums[1] += a * b * V * V
            sums[2] += b * b * V * V
            sums[3] += a * X * V
            sums[4] += b * X * V
        S = np.exp(-1j * (odd_phase + rng.normal(0, 0.05, rho.shape)))      # refine_ctf's sum: minus the aberration
        sums[5], sums[6] = S.real, S.imag
        volumes.write_mrc_volume(path, sums.astype(np.float32), self.pixel)

    def test_uses_zernike(self):
        self.assertFalse(refinectf.uses_zernike(self.settings, []))
        self.assertTrue(refinectf.uses_zernike(dict(self.settings, estimate_trefoil=True), []))
        self.assertTrue(refinectf.uses_zernike(dict(self.settings, estimate_fourth_order=True), []))
        self.assertTrue(refinectf.uses_zernike(self.settings, [{"group": 0, "odd": [0.0, 1.0], "even": []}]))
        self.assertFalse(refinectf.uses_zernike(self.settings, [{"group": 0, "odd": [0.0, 0.0], "even": [1.0]}]))

    def test_fit_recovers_tilt_trefoil_and_fourfold(self):
        s = dict(self.settings, estimate_trefoil=True, estimate_fourth_order=True)
        r = refinectf.fit_refinement_aberrations(self.sums_path, self.box, self.pixel, 2.7, 300.0, s, [])
        np.testing.assert_allclose(r["set"]["odd"], self.odd_truth, atol=1.0)
        self.assertAlmostEqual(r["summary"]["beam_tilt_x_mrad"], 0.8, delta=0.05)
        self.assertAlmostEqual(r["summary"]["beam_tilt_y_mrad"], -0.3, delta=0.05)
        np.testing.assert_allclose(r["summary"]["trefoil"], [40.0, -30.0], atol=1.0)
        np.testing.assert_allclose(r["summary"]["fourfold"], [80.0, -60.0], atol=8.0)
        self.assertEqual((r["set"]["odd_order"], r["set"]["even_order"]), (3, 4))
        self.assertEqual(set(r["maps"]), {"odd_measured", "odd_fitted", "even_measured", "even_fitted"})
        self.assertLess(r["summary"]["odd_phase_rms"], 0.1)

    def test_each_optics_group_gets_its_own_fit_under_the_minimum_size_rule(self):
        # group 1 (big): its own tilt and its own even set; group 2 (between the minimums): its own tilt, the
        # whole data set's even set; group 3 (tiny): no set of its own; group 4: counted but no sums file
        odd2 = aberrations.odd_coefficients_for_beam_tilt(-0.5, 0.6, 0.0, 0.0, 2.7, 300.0) + np.array([0, 0, -20.0, 0, 0, 25.0])
        even1 = np.zeros(9); even1[4], even1[8] = -50.0, 30.0
        self._write_sums(refinectf.group_sums_path(self.sums_path, 1), self.odd_truth, even1, 6)
        self._write_sums(refinectf.group_sums_path(self.sums_path, 2), odd2, self.even_truth, 7)
        self._write_sums(refinectf.group_sums_path(self.sums_path, 3), odd2, self.even_truth, 8)
        with open(os.path.join(self.tmp, "sums_groups.txt"), "w") as fh:
            fh.write("1 3000\n2 800\n3 100\n4 900\n")
        self.assertEqual(refinectf.read_group_counts(self.sums_path), {1: 3000, 2: 800, 3: 100, 4: 900})
        s = dict(self.settings, estimate_trefoil=True, estimate_fourth_order=True)
        r = refinectf.fit_all_groups(self.sums_path, self.box, self.pixel, 2.7, 300.0, s, [], names={1: "day one"})
        self.assertEqual([st["group"] for st in r["sets"]], [0, 1, 2])
        by_group = {g["group"]: g for g in r["groups"]}
        self.assertEqual(sorted(by_group), [1, 2, 3, 4])
        self.assertEqual((by_group[1]["fitted"], by_group[1]["name"], by_group[1]["images"], by_group[1]["even_shared"]), (True, "day one", 3000, False))
        self.assertAlmostEqual(by_group[1]["beam_tilt_x_mrad"], 0.8, delta=0.05)
        np.testing.assert_allclose(by_group[1]["fourfold"], [-50.0, 30.0], atol=8.0)
        self.assertEqual((by_group[2]["fitted"], by_group[2]["even_shared"]), (True, True))
        self.assertAlmostEqual(by_group[2]["beam_tilt_x_mrad"], -0.5, delta=0.05)
        self.assertAlmostEqual(by_group[2]["beam_tilt_y_mrad"], 0.6, delta=0.05)
        np.testing.assert_allclose(by_group[2]["trefoil"], [-20.0, 25.0], atol=1.0)
        self.assertEqual(r["sets"][2]["even"], r["sets"][0]["even"])                 # the whole data set's even set
        np.testing.assert_allclose(by_group[2]["fourfold"], r["summary"]["fourfold"])
        self.assertFalse(by_group[3]["fitted"]); self.assertIn("fewer than 500", by_group[3]["reason"])
        self.assertFalse(by_group[4]["fitted"]); self.assertIn("no sums file", by_group[4]["reason"])
        self.assertEqual(set(r["group_maps"]), {1, 2})
        # the aberration file carries the three lines
        path = refinements.write_aberration_sets(r["sets"], os.path.join(self.tmp, "ab.txt"))
        lines = [ln for ln in Path(path).read_text().splitlines() if not ln.startswith("#")]
        self.assertEqual([ln.split()[0] for ln in lines], ["0", "1", "2"])
        # without a groups file: the whole data set only, as before
        os.remove(os.path.join(self.tmp, "sums_groups.txt"))
        r0 = refinectf.fit_all_groups(self.sums_path, self.box, self.pixel, 2.7, 300.0, s, [])
        self.assertEqual([st["group"] for st in r0["sets"]], [0])
        self.assertEqual(r0["groups"], [])

    def test_beam_tilt_only_drops_the_trefoil_and_even_is_added_to_the_input(self):
        r = refinectf.fit_refinement_aberrations(self.sums_path, self.box, self.pixel, 2.7, 300.0, self.settings, [])
        self.assertEqual(r["set"]["odd"][2], 0.0)
        self.assertEqual(r["set"]["odd"][5], 0.0)
        self.assertEqual(r["set"]["even"], [])
        self.assertAlmostEqual(r["summary"]["beam_tilt_x_mrad"], 0.8, delta=0.1)
        self.assertNotIn("even_measured", r["maps"])
        # an input even set: the residual is added to it
        current = [{"group": 0, "odd_order": 0, "even_order": 4, "odd": [], "even": [0.0] * 4 + [10.0, 0.0, 0.0, 0.0, 0.0]}]
        s = dict(self.settings, estimate_fourth_order=True, refine_beam_tilt=False)
        r2 = refinectf.fit_refinement_aberrations(self.sums_path, self.box, self.pixel, 2.7, 300.0, s, current)
        self.assertAlmostEqual(r2["set"]["even"][4], 10.0 + r2["set"]["even"][4] - 10.0)    # shape only: 9 coefficients
        self.assertEqual(len(r2["set"]["even"]), 9)
        self.assertGreater(r2["set"]["even"][4], 80.0)      # 10 + a residual of about 80
        self.assertEqual(r2["set"]["odd"], [])


if __name__ == "__main__":
    unittest.main()
