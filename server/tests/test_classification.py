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
