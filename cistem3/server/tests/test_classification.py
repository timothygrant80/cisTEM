"""The pure pieces of server/classification.py: cisTEM's schedules, the
star file round trip, and the round statistics. The driver itself needs a
runner and real binaries; it is exercised end to end by hand."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import classification as c  # noqa: E402


class ScheduleTests(unittest.TestCase):
    def test_package_defaults_follow_set_defaults(self):
        d = c.package_defaults(150.0)
        self.assertEqual(d, {"mask_radius_a": 90.0, "max_search_range_a": 49.5})
        self.assertFalse(c.DEFAULTS["exclude_blank_edges"])   # ExcludeBlankEdgesNoRadio
        self.assertTrue(c.DEFAULTS["auto_centre"])            # on by request, unlike cisTEM

    def test_default_number_of_classes_follows_set_defaults(self):
        # MyRefine2DPanel::SetDefaults(): particles / 300, stepped down.
        self.assertEqual(c.default_number_of_classes(100), 5)
        self.assertEqual(c.default_number_of_classes(3100), 5)    # 10 per 300 is not *more than* 10
        self.assertEqual(c.default_number_of_classes(3400), 10)
        self.assertEqual(c.default_number_of_classes(6400), 20)
        self.assertEqual(c.default_number_of_classes(9400), 30)
        self.assertEqual(c.default_number_of_classes(12400), 40)
        self.assertEqual(c.default_number_of_classes(200000), 50)

    def test_startup_percent_used_is_300_per_class_capped(self):
        self.assertAlmostEqual(c.startup_percent_used(50, 100000), 15.0)
        self.assertEqual(c.startup_percent_used(50, 100), 100.0)

    def test_high_resolution_ramp(self):
        # 20 rounds: reaches the finish value at round 15 (3/4), then holds.
        limits = [c.high_resolution_limit(i, 20, 40.0, 8.0) for i in range(20)]
        self.assertEqual(limits[0], 40.0)
        self.assertAlmostEqual(limits[7], 24.0)
        self.assertEqual(limits[14], 8.0)
        self.assertTrue(all(l == 8.0 for l in limits[15:]))
        # Fewer than four rounds: the finish is reached on the last round.
        self.assertEqual([round(x, 2) for x in (c.high_resolution_limit(i, 3, 40.0, 8.0) for i in range(3))], [40.0, 24.0, 8.0])
        self.assertEqual(c.high_resolution_limit(0, 1, 40.0, 8.0), 8.0)

    def test_auto_percent_used_schedule(self):
        # < 10 rounds: everything.
        self.assertEqual(c.auto_percent_used(0, 5, 50, 100000), 100.0)
        # 15 rounds, 50 classes, 100k particles: 15% for five rounds, at
        # least 30% until the last five, then everything.
        sched = [c.auto_percent_used(i, 15, 50, 100000) for i in range(15)]
        self.assertEqual(sched[:5], [15.0] * 5)
        self.assertEqual(sched[5:10], [30.0] * 5)
        self.assertEqual(sched[10:], [100.0] * 5)
        # 20 rounds falls in cisTEM's "< 30" branch: ten early rounds.
        sched = [c.auto_percent_used(i, 20, 50, 100000) for i in range(20)]
        self.assertEqual(sched[:10], [15.0] * 10)
        self.assertEqual(sched[10:15], [30.0] * 5)
        self.assertEqual(sched[15:], [100.0] * 5)
        # Few particles: 300 per class is more than there are.
        self.assertEqual(c.auto_percent_used(0, 20, 5, 100), 100.0)
        # 30+ rounds: 15 early rounds.
        sched = [c.auto_percent_used(i, 30, 50, 100000) for i in range(30)]
        self.assertEqual(sched[14], 15.0)
        self.assertEqual(sched[15], 30.0)
        self.assertEqual(sched[25], 100.0)

    def test_particle_range_matches_first_last_particle_for_job(self):
        # 10 particles over 4 jobs: the remainder goes to the first jobs.
        self.assertEqual([c.particle_range(k, 4, 10) for k in range(1, 5)], [(1, 3), (4, 6), (7, 8), (9, 10)])
        self.assertEqual(c.particle_range(1, 1, 100), (1, 100))
        ranges = [c.particle_range(k, 7, 100) for k in range(1, 8)]
        self.assertEqual(ranges[0][0], 1)
        self.assertEqual(ranges[-1][1], 100)
        self.assertEqual(sum(b - a + 1 for a, b in ranges), 100)


class StarFileTests(unittest.TestCase):
    def test_round_trip_and_cistem_shape(self):
        rows = []
        for i in range(1, 4):
            r = c.empty_result(i)
            r.update(pixel_size=1.5, voltage=300.0, cs=2.7, amplitude_contrast=0.07, defocus_1=20000.5,
                     defocus_2=19000.4, logp=-1234.6, best_2d_class=3, psi=12.34, x_shift=-1.5)
            rows.append(r)
        d = tempfile.mkdtemp()
        path = c.write_star(os.path.join(d, "t.star"), rows, comments=["a comment"])
        text = open(path).read()
        # What cisTEMStarFileReader insists on, and the labelled columns.
        self.assertIn("\ndata_\n", text)
        self.assertIn("\nloop_\n", text)
        self.assertIn("_cisTEMPositionInStack #1\n", text)
        self.assertIn("_cisTEMBest2DClass #19\n", text)
        self.assertIn("# a comment\n", text)
        back = c.read_star(path)
        self.assertEqual(len(back), 3)
        self.assertEqual(back[0]["position_in_stack"], 1)
        self.assertEqual(back[0]["best_2d_class"], 3)
        self.assertAlmostEqual(back[0]["psi"], 12.34)
        self.assertAlmostEqual(back[0]["x_shift"], -1.5)
        self.assertAlmostEqual(back[0]["defocus_1"], 20000.5)
        self.assertAlmostEqual(back[0]["pixel_size"], 1.5)
        self.assertEqual(back[0]["logp"], -1235.0)  # myroundint, as cisTEM writes it
        self.assertEqual(back[0]["sigma"], 10.0)

    def test_reader_skips_unknown_columns(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "u.star")
        with open(path, "w") as fh:
            fh.write("data_\nloop_\n_cisTEMPositionInStack #1\n_cisTEMAngleTheta #2\n_cisTEMBest2DClass #3\n#  POS THETA CLS\n"
                     "   7  1.0   2\n   8  2.0  -2\n")
        rows = c.read_star(path)
        self.assertEqual([(r["position_in_stack"], r["best_2d_class"]) for r in rows], [(7, 2), (8, -2)])
        self.assertNotIn("theta", rows[0])


class SelectionTests(unittest.TestCase):
    """Selections and the class-average package source against a throwaway
    project database: cisTEM's selection tables, re-centring by the
    classification's shifts, and duplicate removal keeping the pick that
    moved least."""

    def setUp(self):
        import db
        import refinement_packages as rp
        self.db, self.rp = db, rp
        self.tmp = tempfile.mkdtemp()
        self._root = db.PROJECTS_ROOT
        db.PROJECTS_ROOT = __import__("pathlib").Path(self.tmp)
        self.project = "t-" + os.path.basename(self.tmp)[-6:]
        (db.PROJECTS_ROOT / self.project).mkdir(parents=True)
        conn = db.get_conn(self.project)
        with conn:
            conn.execute("INSERT INTO REFINEMENT_PACKAGE_ASSETS(REFINEMENT_PACKAGE_ASSET_ID, NAME, STACK_FILENAME, STACK_BOX_SIZE, OUTPUT_PIXEL_SIZE) VALUES (1, 'P', '/nonexistent.mrc', 64, 1.0)")
            conn.execute("CREATE TABLE REFINEMENT_PACKAGE_CONTAINED_PARTICLES_1(ORIGINAL_PARTICLE_POSITION_ASSET_ID INTEGER PRIMARY KEY, PARENT_IMAGE_ASSET_ID INTEGER, "
                         "POSITION_IN_STACK INTEGER, X_POSITION REAL, Y_POSITION REAL, PIXEL_SIZE REAL, DEFOCUS_1 REAL, DEFOCUS_2 REAL, DEFOCUS_ANGLE REAL, PHASE_SHIFT REAL, "
                         "SPHERICAL_ABERRATION REAL, MICROSCOPE_VOLTAGE REAL, AMPLITUDE_CONTRAST REAL, ASSIGNED_SUBSET INTEGER)")
            # three picks on image 7: two of them 5 A apart once re-centred, one far away
            for pid, pos, x, y in ((10, 1, 100.0, 100.0), (11, 2, 130.0, 100.0), (12, 3, 500.0, 500.0)):
                conn.execute("INSERT INTO REFINEMENT_PACKAGE_CONTAINED_PARTICLES_1 VALUES (?,7,?,?,?,1.0,1,1,0,0,2.7,300,0.07,1)", (pid, pos, x, y))
            conn.execute("INSERT INTO IMAGE_ASSETS(IMAGE_ASSET_ID, NAME, FILENAME, PIXEL_SIZE, VOLTAGE, SPHERICAL_ABERRATION, PROTEIN_IS_WHITE) VALUES (7, 'img', '/nonexistent.mrc', 1.0, 300, 2.7, 0)")
        cls = {"classification_id": 1, "refinement_package_asset_id": 1, "name": "C", "class_average_file": "/nonexistent.mrc", "number_of_particles": 3,
               "number_of_classes": 2, "low_resolution_limit": 300, "high_resolution_limit": 8, "mask_radius": 100, "angular_search_step": 15,
               "search_range_x": 100, "search_range_y": 100, "smoothing_factor": 1, "exclude_blank_edges": True, "auto_percent_used": True, "percent_used": 100}
        rows = [dict(c.empty_result(1), best_2d_class=1, x_shift=0.0, y_shift=0.0),
                dict(c.empty_result(2), best_2d_class=1, x_shift=25.0, y_shift=0.0),   # moved a lot, lands 5 A from particle 1
                dict(c.empty_result(3), best_2d_class=2, x_shift=1.0, y_shift=-1.0)]
        c.add_classification(conn, cls, rows)
        conn.close()

    def tearDown(self):
        self.db.PROJECTS_ROOT = self._root
        __import__("shutil").rmtree(self.tmp, ignore_errors=True)

    def test_selection_crud_and_members(self):
        conn = self.db.get_conn(self.project)
        sid = c.create_selection(conn, 1, "sel", [1])
        sel = c.get_selection(conn, sid)
        self.assertEqual((sel["name"], sel["classes"], sel["particle_count"], sel["number_of_classes"]), ("sel", [1], 2, 2))
        c.set_selection_classes(conn, sid, [1, 2, 99])   # 99 is not a class
        self.assertEqual(c.get_selection(conn, sid)["classes"], [1, 2])
        members = c.selection_members(conn, [sid])
        self.assertEqual([m["position_in_stack"] for m in members], [1, 2, 3])
        self.assertTrue(c.rename_selection(conn, sid, "renamed"))
        self.assertTrue(c.delete_selection(conn, sid))
        self.assertEqual(c.list_selections(conn, 1), [])
        conn.close()

    def test_class_selection_source_recentres_and_removes_duplicates(self):
        conn = self.db.get_conn(self.project)
        sid = c.create_selection(conn, 1, "sel", [1, 2])
        rows, info = self.rp._particles_of_selections(conn, {"selection_ids": [sid], "recentre": True, "remove_duplicates": True, "duplicate_threshold_a": 20.0})
        # Particle 2 (moved 25 A) is the duplicate of particle 1 and goes; particle 3 is re-centred.
        self.assertEqual(info["duplicates_removed"], 1)
        self.assertEqual(sorted(r["PARTICLE_POSITION_ASSET_ID"] for r in rows), [10, 12])
        p3 = [r for r in rows if r["PARTICLE_POSITION_ASSET_ID"] == 12][0]
        self.assertEqual((p3["X_POSITION"], p3["Y_POSITION"]), (499.0, 501.0))
        # Without re-centring nothing moves and nothing is removed.
        rows, info = self.rp._particles_of_selections(conn, {"selection_ids": [sid], "recentre": False, "remove_duplicates": True})
        self.assertEqual((len(rows), info["duplicates_removed"]), (3, 0))
        self.assertEqual([r["X_POSITION"] for r in rows], [100.0, 130.0, 500.0])
        conn.close()

    def test_selection_defaults_inherit_the_parent_package(self):
        conn = self.db.get_conn(self.project)
        sid = c.create_selection(conn, 1, "sel", [1, 2])
        d = self.rp.selection_defaults(conn, [sid])
        pkg = conn.execute("SELECT * FROM REFINEMENT_PACKAGE_ASSETS WHERE REFINEMENT_PACKAGE_ASSET_ID=1").fetchone()
        self.assertEqual((d["symmetry"], d["molecular_weight_kda"], d["largest_dimension_a"], d["number_of_classes"], d["box_size"], d["pixel_size"]),
                         (pkg["SYMMETRY"], pkg["MOLECULAR_WEIGHT"], pkg["PARTICLE_SIZE"], pkg["NUMBER_OF_CLASSES"], pkg["STACK_BOX_SIZE"], pkg["OUTPUT_PIXEL_SIZE"]))
        self.assertEqual(d["parent_package_ids"], [1])
        conn.close()


class StatisticsTests(unittest.TestCase):
    def test_round_statistics(self):
        inputs = [dict(c.empty_result(i), best_2d_class=1) for i in range(1, 5)]
        outputs = [dict(c.empty_result(1), best_2d_class=1, logp=-10.0, sigma=1.0),
                   dict(c.empty_result(2), best_2d_class=2, logp=-20.0, sigma=3.0),
                   dict(c.empty_result(3), best_2d_class=-1, logp=-99.0, sigma=9.0),   # sat the round out
                   dict(c.empty_result(4), best_2d_class=0)]                            # never classified
        st = c.round_statistics(outputs, inputs)
        self.assertEqual(st["active_particles"], 2)
        self.assertAlmostEqual(st["average_logp"], -15.0)
        self.assertAlmostEqual(st["average_sigma"], 2.0)
        self.assertAlmostEqual(st["percent_moved"], 50.0)
        self.assertIsNone(c.round_statistics([outputs[3]], inputs)["average_logp"])

    def test_montage_geometry_is_consistent(self):
        g = c.montage_geometry(50, box=162)
        self.assertEqual(g["columns"] * g["rows"] >= 50, True)
        self.assertEqual(g["width"], g["columns"] * g["tile_width"] + (g["columns"] + 1) * g["gap"])
        self.assertLessEqual(g["tile_width"], c.MONTAGE_TILE)
        self.assertEqual(c.montage_geometry(0)["rows"], 0)


if __name__ == "__main__":
    unittest.main()
