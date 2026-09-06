"""The pure pieces of server/abinitio.py and server/volumes.py: cisTEM's
ab-initio schedules, the star and statistics files, occupancy updates, and
the volume operations the GUI does between job steps (Fourier resampling,
auto-masking, orthogonal views). The driver itself needs a runner and the
real binaries and is exercised end to end by hand."""
import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import abinitio as ab  # noqa: E402
import starfile  # noqa: E402
import volumes as V  # noqa: E402


class ScheduleTests(unittest.TestCase):
    def test_asymmetric_units(self):
        self.assertEqual([ab.asymmetric_units(s) for s in ("C1", "C4", "D2", "D7", "T", "O", "I", "I2")], [1, 4, 4, 14, 12, 24, 60, 60])

    def test_percent_used_plan(self):
        # 100k particles, one class, D2: 2500 / 10000 asymmetric units.
        plan = ab.percent_used_plan(100000, 1, "D2", True, 10, 10)
        self.assertAlmostEqual(plan["start"], 2.5)
        self.assertAlmostEqual(plan["end"], 10.0)
        self.assertAlmostEqual(plan["sym_start"], 0.625)
        self.assertAlmostEqual(plan["sym_end"], 2.5)
        # Few particles: capped at everything.
        self.assertEqual(ab.percent_used_plan(100, 1, "C1", True, 10, 10)["start"], 100.0)
        # Manual: the user's numbers, and the symmetry variant pinned to the start value.
        self.assertEqual(ab.percent_used_plan(100000, 1, "C1", False, 7.5, 30.0), {"start": 7.5, "end": 30.0, "sym_start": 7.5, "sym_end": 7.5})

    def test_resolution_ramp_resets_to_start_early_on(self):
        plan = {"start": 10, "end": 10, "sym_start": 10, "sym_end": 10}
        hr = [ab.round_schedule(i, 40, 20.0, 8.0, plan, False)["high_res"] for i in range(40)]
        self.assertEqual(hr[0], 20.0)
        self.assertEqual(hr[4], 20.0)            # every 4th round (40/10) and the one after
        self.assertEqual(hr[5], 20.0)
        self.assertAlmostEqual(hr[6], 20.0 + (8.0 - 20.0) * 6 / 39.0)
        self.assertEqual(hr[28], 20.0 + (8.0 - 20.0) * 28 / 39.0)  # past 65%: no more resets
        self.assertEqual(hr[39], 8.0)
        self.assertAlmostEqual(ab.round_schedule(3, 40, 20.0, 8.0, plan, False)["next_high_res"], 20.0 + (8.0 - 20.0) * 4 / 39.0)
        # A short run must not divide by zero in the reset rule.
        self.assertEqual(ab.round_schedule(0, 3, 20.0, 8.0, plan, False)["high_res"], 20.0)

    def test_percent_follows_symmetry_plan(self):
        plan = {"start": 2.5, "end": 10.0, "sym_start": 0.625, "sym_end": 2.5}
        self.assertAlmostEqual(ab.round_schedule(39, 40, 20, 8, plan, False)["percent_used"], 10.0)
        self.assertAlmostEqual(ab.round_schedule(39, 40, 20, 8, plan, True)["percent_used"], 2.5)

    def test_wiener_and_signed_cc(self):
        self.assertEqual(ab.wiener_nominator(0, 40, 0), 500.0)
        self.assertAlmostEqual(ab.wiener_nominator(20, 40, 0), 105.0)
        self.assertEqual(ab.wiener_nominator(39, 40, 0), max(10.0, 200.0 - 190.0 * 39 / 40.0))
        self.assertEqual(ab.wiener_nominator(0, 40, 1), 10.0)
        self.assertEqual([ab.signed_cc_limit(i, 4) for i in range(4)], [15.0, 0.0, 15.0, 0.0])
        self.assertAlmostEqual(ab.angular_step(20.0), math.degrees(40.0 / 75.0))

    def test_class_averages_per_class(self):
        # 43 members / 5 per average = 8, but at least 2500 averages over 2 classes.
        self.assertEqual(ab.class_averages_per_class(43, 5, 2), 1250)
        self.assertEqual(ab.class_averages_per_class(100000, 5, 2), 10000)   # capped at 20000 in all
        self.assertEqual(ab.class_averages_per_class(30000, 5, 4), 5000)
        self.assertEqual(ab.classaverage_job_ranges(2, 8), [(0, 0), (1, 1)])
        self.assertEqual(ab.classaverage_job_ranges(5, 2), [(0, 2), (3, 4)])

    def test_prepare_stack_jobs_and_box(self):
        self.assertEqual(ab.prepare_stack_jobs(100, 8), 1)
        self.assertEqual(ab.prepare_stack_jobs(7821, 8), 8)
        self.assertEqual(ab.prepare_stack_jobs(450, 8), 4)
        self.assertEqual(ab.binned_box_size(162, 8.0 / 2.0 / 1.5), 64)


class RefinementBookkeepingTests(unittest.TestCase):
    def test_update_occupancies_two_classes(self):
        a = [{"occupancy": 50.0, "logp": -10.0}, {"occupancy": 50.0, "logp": -100.0}]
        b = [{"occupancy": 50.0, "logp": -10.0}, {"occupancy": 50.0, "logp": -50.0}]
        ab.update_occupancies([a, b])
        self.assertAlmostEqual(a[0]["occupancy"], 50.0)
        self.assertAlmostEqual(b[0]["occupancy"], 50.0)
        self.assertAlmostEqual(a[1]["occupancy"], 0.0)      # 50 logP behind: out of the 10-unit window
        self.assertAlmostEqual(b[1]["occupancy"], 100.0)

    def test_average_sigma_is_occupancy_weighted_over_active(self):
        rows = [[{"image_is_active": 1, "occupancy": 100.0, "sigma": 2.0}, {"image_is_active": -1, "occupancy": 100.0, "sigma": 99.0},
                 {"image_is_active": 1, "occupancy": 50.0, "sigma": 4.0}]]
        self.assertAlmostEqual(ab.average_sigma(rows), (2.0 + 2.0) / 1.5)

    def test_statistics_round_trip_and_cap(self):
        stats = ab.default_statistics(300.0, 1.5, 64)
        self.assertEqual(stats[0]["shell"], 1)
        self.assertEqual(len(stats), 64 // 2)  # shells 1..number_of_bins-1
        d = tempfile.mkdtemp()
        p = ab.write_statistics(os.path.join(d, "s.txt"), stats, 1.5)
        back = ab.read_statistics(p)
        self.assertEqual(len(back), len(stats))
        self.assertAlmostEqual(back[3]["resolution"], stats[3]["resolution"], places=3)
        self.assertAlmostEqual(back[3]["part_ssnr"], stats[3]["part_ssnr"], places=2)
        inflated = [dict(s, part_ssnr=s["part_ssnr"] * 10) for s in back]
        capped = ab.cap_part_ssnr(inflated, stats)
        self.assertTrue(all(c["part_ssnr"] <= s["part_ssnr"] + 1e-6 for c, s in zip(capped, stats)))

    def test_refinement_star_round_trip(self):
        rows = [{"position_in_stack": i, "image_is_active": 1 if i % 2 else -1, "psi": 10.0 * i, "theta": 20.0, "phi": 30.0, "x_shift": -1.25,
                 "y_shift": 0.5, "defocus_1": 20000.0, "defocus_2": 19000.0, "defocus_angle": 45.0, "phase_shift": 0.0, "occupancy": 100.0,
                 "logp": -1234.4, "sigma": 1.0, "score": 12.5, "pixel_size": 1.5, "voltage": 300.0, "cs": 2.7, "amplitude_contrast": 0.07,
                 "assigned_subset": 1 + i % 2} for i in range(1, 4)]
        d = tempfile.mkdtemp()
        p = starfile.write_star(os.path.join(d, "r.star"), rows)
        text = open(p).read()
        self.assertIn("_cisTEMImageActivity", text)
        self.assertIn("_cisTEMAssignedSubset", text)
        back = starfile.read_star(p)
        self.assertEqual([r["image_is_active"] for r in back], [1, -1, 1])
        self.assertAlmostEqual(back[2]["psi"], 30.0)
        self.assertEqual(back[0]["logp"], -1234.0)
        self.assertEqual([r["assigned_subset"] for r in back], [2, 1, 2])


class Refine3DTests(unittest.TestCase):
    def test_package_defaults_follow_the_particle_size(self):
        import refine3d
        d = refine3d.package_defaults({"PARTICLE_SIZE": 100.0})
        self.assertEqual(d["mask_radius_a"], 65.0)
        self.assertEqual(d["global_mask_radius_a"], 80.0)
        self.assertEqual(d["search_range_x_a"], 15.0)
        self.assertEqual(d["low_resolution_limit_a"], 150.0)
        self.assertAlmostEqual(d["angular_step_deg"], math.degrees(60.0 / 65.0), places=2)
        self.assertEqual(refine3d.package_defaults({"PARTICLE_SIZE": 300.0})["low_resolution_limit_a"], 300.0)

    def test_settings_global_flag_and_bools(self):
        import refine3d
        s = refine3d.settings_from_params({"refinement_type": "Global Search", "refine_ctf": "true", "number_of_rounds": "3"}, {"PARTICLE_SIZE": 100.0})
        self.assertTrue(s["global"])
        self.assertTrue(s["refine_ctf"])
        self.assertEqual(s["number_of_rounds"], 3)
        self.assertFalse(refine3d.settings_from_params({}, {"PARTICLE_SIZE": 100.0})["global"])

    def test_estimated_resolution_and_angular_histogram(self):
        import refinements
        stats = [{"shell": i, "resolution": 100.0 / i, "fsc": 1.0 - i * 0.1, "part_fsc": 1.0 - i * 0.1} for i in range(1, 11)]
        # FSC drops below 0.143 at shell 9 (0.1): midway between 100/8 and 100/9.
        self.assertAlmostEqual(refinements.estimated_resolution(stats, 1.0), (100.0 / 8 + 100.0 / 9) / 2)
        self.assertEqual(refinements.estimated_resolution([{"shell": 1, "resolution": 50.0, "fsc": 1.0, "part_fsc": 1.0}], 1.5), 3.0)  # never better than Nyquist
        rows = [{"theta": 0.0, "phi": 0.0, "image_is_active": 1}, {"theta": 170.0, "phi": 10.0, "image_is_active": 1}, {"theta": 45.0, "phi": 90.0, "image_is_active": -1}]
        hist = refinements.angular_histogram(rows)
        self.assertEqual(len(hist), 18 * 72)
        self.assertEqual(sum(hist), 2)          # the inactive particle is not counted
        self.assertEqual(hist[0] + hist[18 * 38], 2)  # theta 170 folds to 10 deg with phi + 180 (bin 38)

    def test_symmetry_matrices(self):
        import symmetry
        for symbol, n in (("C1", 1), ("C4", 4), ("D2", 4), ("D7", 14), ("T", 12), ("T2", 12), ("O", 24), ("I", 60), ("I2", 60)):
            mats = symmetry.matrices(symbol)
            self.assertEqual(len(mats), n, symbol)
            for m in mats:  # every one a proper rotation
                mt = tuple(zip(*m))
                p = symmetry.matmul(m, mt)
                for i in range(3):
                    for j in range(3):
                        self.assertAlmostEqual(p[i][j], 1.0 if i == j else 0.0, places=4)
        with self.assertRaises(ValueError):
            symmetry.matrices("X3")
        with self.assertRaises(ValueError):
            symmetry.matrices("C")
        # The C2 mate of a view is the same tilt, phi + 180.
        views = symmetry.symmetry_related_views(30.0, 40.0, 50.0, "C2")
        self.assertAlmostEqual(views[0][0], 40.0)
        self.assertAlmostEqual(views[0][1], 30.0)
        self.assertAlmostEqual(views[1][0], 40.0)
        self.assertAlmostEqual(views[1][1] % 360.0, 210.0)

    def test_angular_histogram_expands_symmetry(self):
        import refinements
        rows = [{"theta": 0.0, "phi": 0.0, "psi": 0.0, "image_is_active": 1}]
        self.assertEqual(sum(refinements.angular_histogram(rows, 1, "C4")), 4)
        self.assertEqual(refinements.angular_histogram(rows, 1, "D2")[0], 4)  # the pole stays at the pole under D2
        rows = [{"theta": 40.0, "phi": 32.0, "psi": 50.0, "image_is_active": 1}]
        c1, d2, ico = (refinements.angular_histogram(rows, 1, s) for s in ("C1", "D2", "I"))
        self.assertEqual((sum(c1), sum(d2), sum(ico)), (1, 4, 60))
        self.assertEqual([i for i, v in enumerate(c1) if v], [18 * 6 + 4])  # phi 32 -> bin 6; theta 40 -> bin 4 (equal-area bins: 38.9-43.8 deg)
        self.assertTrue(all(d2[b] for b in (18 * 6 + 4, 18 * 42 + 4)))       # its C2 mate at phi 212
        self.assertEqual(sum(refinements.angular_histogram(rows, 1, "nonsense")), 1)  # unknown symbols fall back to C1
        # Several classes: a particle counts only for its highest-occupancy class.
        cls1 = [{"position_in_stack": 1, "theta": 10.0, "phi": 0.0, "psi": 0.0, "occupancy": 80.0, "image_is_active": 1},
                {"position_in_stack": 2, "theta": 10.0, "phi": 0.0, "psi": 0.0, "occupancy": 20.0, "image_is_active": 1}]
        cls2 = [{"position_in_stack": 1, "theta": 50.0, "phi": 0.0, "psi": 0.0, "occupancy": 20.0, "image_is_active": 1},
                {"position_in_stack": 2, "theta": 50.0, "phi": 0.0, "psi": 0.0, "occupancy": 80.0, "image_is_active": 1}]
        self.assertEqual(refinements.best_class_per_particle([cls1, cls2]), {1: 1, 2: 2})
        self.assertEqual(sum(refinements.angular_histogram([cls1, cls2], 1, "C1")), 1)
        self.assertEqual(sum(refinements.angular_histogram([cls1, cls2], 2, "C3")), 3)

    def test_apply_mask_keeps_inside_and_replaces_outside(self):
        n = 32
        z, y, x = np.indices((n, n, n))
        vol = np.ones((n, n, n), dtype=np.float32) * 5.0
        vol[(z - 16) ** 2 + (y - 16) ** 2 + (x - 16) ** 2 > 12 ** 2] = 1.0
        mask = ((z - 16) ** 2 + (y - 16) ** 2 + (x - 16) ** 2 <= 6 ** 2).astype(np.float32)
        out = V.apply_mask(vol, mask, 2.0, 0.0)
        self.assertAlmostEqual(float(out[16, 16, 16]), 5.0, places=3)
        self.assertAlmostEqual(float(out[0, 0, 0]), 1.0, places=3)   # the average beyond 0.4 of the box
        kept = V.apply_mask(vol, mask, 2.0, 0.5, 0.25, 0.05)
        self.assertGreater(float(kept[16, 16, 30]), 0.0)
        self.assertLess(float(kept[16, 16, 30]), 5.0)


class VolumeTests(unittest.TestCase):
    def setUp(self):
        n = 48
        z, y, x = np.indices((n, n, n))
        c = n // 2
        self.vol = np.exp(-(((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2) / (2 * 5.0 ** 2))).astype(np.float32)
        self.vol += np.random.default_rng(1).normal(0, 0.01, self.vol.shape).astype(np.float32)
        self.vol[3, 3, 3] = 5.0   # a bright speck away from the particle

    def test_mrc_round_trip(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "v.mrc")
        V.write_mrc_volume(p, self.vol, 1.25)
        back, ps = V.read_mrc_volume(p)
        self.assertEqual(back.shape, self.vol.shape)
        self.assertAlmostEqual(ps, 1.25, places=5)
        self.assertTrue(np.allclose(back, self.vol))
        self.assertEqual(V.read_mrc_header(p)["nz"], 48)

    def test_fourier_resize_keeps_the_object_centred(self):
        vol = self.vol.copy()
        vol[3, 3, 3] = 0.0   # the speck would outshine the blob once binned
        small = V.fourier_resize(vol, 32)
        self.assertEqual(small.shape, (32, 32, 32))
        self.assertEqual(np.unravel_index(np.argmax(small), small.shape), (16, 16, 16))
        big = V.fourier_resize(vol, 64)
        self.assertEqual(np.unravel_index(np.argmax(big), big.shape), (32, 32, 32))
        self.assertTrue(np.allclose(V.fourier_resize(vol, 48), vol))

    def test_auto_mask_keeps_the_particle_and_drops_the_speck(self):
        masked = V.auto_mask(self.vol, 1.5, 20.0)
        self.assertGreater(masked[24, 24, 24], 0.9)
        self.assertEqual(masked[3, 3, 3], 0.0)          # beyond the mask radius
        self.assertEqual(masked[0, 0, 0], 0.0)
        mask = V.convert_to_auto_mask(self.vol, 1.5, 20.0)
        self.assertEqual(mask[24, 24, 24], 1.0)
        self.assertEqual(mask[3, 3, 3], 0.0)

    def test_orthogonal_views_layout(self):
        canvas = V.orthogonal_views(self.vol, 20.0 / 1.5)
        self.assertEqual(canvas.shape, (2 * 48, 3 * 48))
        # The particle sits at the centre of every panel.
        for row in (0, 1):
            for col in range(3):
                panel = canvas[row * 48:(row + 1) * 48, col * 48:(col + 1) * 48]
                self.assertEqual(np.unravel_index(np.argmax(panel), panel.shape), (24, 24))
        self.assertTrue(0.0 <= canvas.min() and canvas.max() <= 1.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()


class AutoRefineTests(unittest.TestCase):
    """AutoRefinementManager's schedules (BeginRefinementCycle / CycleRefinement /
    SetupRefinementJob), as pure functions."""

    def test_defaults_and_settings(self):
        import autorefine
        d = autorefine.package_defaults({"PARTICLE_SIZE": 100.0})
        self.assertEqual((d["mask_radius_a"], d["global_mask_radius_a"], d["search_range_x_a"], d["low_resolution_limit_a"]), (65.0, 80.0, 15.0, 150.0))
        s = autorefine.settings_from_params({}, {"PARTICLE_SIZE": 100.0})
        self.assertEqual(s["high_resolution_limit_a"], 20.0)
        self.assertTrue(s["auto_mask"] and s["autocenter"] and not s["autocrop_images"])
        s = autorefine.settings_from_params({"use_mask": "true", "auto_mask": "true"}, {"PARTICLE_SIZE": 100.0})
        self.assertFalse(s["auto_mask"])   # a supplied mask switches auto-masking (and centring) off
        self.assertFalse(s["autocenter"])

    def test_asymmetric_units_and_start_percent(self):
        import autorefine
        self.assertEqual([autorefine.asymmetric_units(x) for x in ("C1", "C4", "D2", "D7", "T", "O", "I", "I2", "junk")], [1, 4, 4, 14, 12, 24, 60, 60, 1])
        # 8000 * exp(75 / 20^2) = 9651 asymmetric units wanted at 20 A.
        wanted = int(round(8000.0 * math.exp(75.0 / 400.0)))
        self.assertAlmostEqual(autorefine.percent_for_resolution(20.0, 100000, "C1", 1), wanted / 1000.0)
        self.assertAlmostEqual(autorefine.percent_for_resolution(20.0, 100000, "D2", 1), wanted / 4000.0)
        self.assertAlmostEqual(autorefine.percent_for_resolution(20.0, 100000, "C1", 2), 2 * wanted / 1000.0)
        self.assertEqual(autorefine.percent_for_resolution(20.0, 100, "C1", 1), 100.0)  # capped

    def test_resolution_statistics_helpers(self):
        import autorefine
        # 10 shells, resolution 100/i; FSC drops below 0.5 between shells 5 and 6, below 0.143 between 8 and 9.
        stats = [{"shell": i, "resolution": 100.0 / i, "fsc": 1.0 - i * 0.1, "part_fsc": 1.0 - i * 0.1} for i in range(1, 11)]
        self.assertAlmostEqual(autorefine.resolution_at(stats, 0.5, 1.0), (100.0 / 5 + 100.0 / 6) / 2)
        self.assertAlmostEqual(autorefine.resolution_at(stats, 0.143, 1.0), (100.0 / 8 + 100.0 / 9) / 2)
        self.assertEqual(autorefine.resolution_at([{"shell": 1, "resolution": 50.0, "fsc": 1.0, "part_fsc": 1.0}], 0.5, 1.5), 3.0)
        # NShellsAfter: first shell finer than 30 A is shell 4 (25 A), +2 -> shell 6 (16.67 A).
        self.assertAlmostEqual(autorefine.resolution_n_shells_after(stats, 30.0, 2, 1.0), 100.0 / 6)
        self.assertEqual(autorefine.resolution_n_shells_after(stats, 5.0, 2, 1.0), 0.0)       # nothing finer than 5 A
        self.assertEqual(autorefine.resolution_n_shells_after(stats, 30.0, 20, 1.0), 2.0)     # runs off the end -> Nyquist
        # NShellsBefore: last shell coarser than 12 A is index 7 (12.5 A), -2 -> index 5 (16.67 A).
        self.assertAlmostEqual(autorefine.resolution_n_shells_before(stats, 12.0, 2), 100.0 / 6)
        self.assertEqual(autorefine.resolution_n_shells_before(stats, 12.0, 10), 0.0)
        # The next limit never gets coarser than the current one.
        self.assertLessEqual(autorefine.next_class_limit(stats, 30.0, 100, 1.0, 40.0), 30.0)
        self.assertLessEqual(autorefine.next_class_limit(stats, 8.0, 100, 1.0, 40.0), 8.0)

    def test_choose_global(self):
        import autorefine, random
        rng = random.Random(1)
        # Never aligned globally -> global search, whatever else is true.
        self.assertEqual(autorefine.choose_global(3, 10.0, 8.0, 100.0, 0, 5, True, False, rng), 0)
        # Aligned globally this round, or the final round -> local.
        self.assertEqual(autorefine.choose_global(3, 10.0, 8.0, 20.0, 1, 0, True, False, rng), 1)
        self.assertEqual(autorefine.choose_global(3, 10.0, 8.0, 20.0, 1, 4, True, True, rng), 1)
        # Already globally aligned at 5 A or better -> no more global searches.
        self.assertTrue(all(autorefine.choose_global(5, 4.0, 4.0, 4.5, 2, 3, True, False, rng) == 1 for _ in range(50)))
        # High resolution, reference from all particles, last global at low res -> global again.
        self.assertEqual(autorefine.choose_global(4, 4.5, 7.0, 20.0, 1, 3, True, False, rng), 0)

    def test_should_stop(self):
        import autorefine
        stable = [20.0, 15.0, 12.0, 11.0, 11.0, 11.0]   # the last two rounds did not beat the one before them
        self.assertTrue(autorefine.should_stop(stable, 100.0, 0.0, 1))
        self.assertFalse(autorefine.should_stop(stable, 90.0, 0.0, 1))          # not all particles in use
        self.assertFalse(autorefine.should_stop(stable[2:], 100.0, 0.0, 1))     # fewer than 5 rounds
        self.assertFalse(autorefine.should_stop([20.0, 15.0, 12.0, 11.0, 11.0], 100.0, 0.0, 1))  # 11 still beat 12
        self.assertFalse(autorefine.should_stop([20.0, 15.0, 12.0, 11.0, 11.0, 10.5], 100.0, 0.0, 1))  # still improving
        self.assertFalse(autorefine.should_stop(stable, 100.0, 2.0, 2))         # occupancies still moving
        self.assertFalse(autorefine.should_stop(stable, 100.0, 0.0, 2))         # several classes need 10 rounds
        self.assertTrue(autorefine.should_stop(stable * 2, 100.0, 0.5, 2))

    def test_update_occupancies_flag(self):
        cls1 = [{"logp": 0.0, "occupancy": 90.0}]
        cls2 = [{"logp": 0.0, "occupancy": 10.0}]
        ab.update_occupancies([cls1, cls2], use_old_occupancies=False)
        self.assertAlmostEqual(cls1[0]["occupancy"], 50.0)   # equal logP, equal prior -> 50/50
        cls1 = [{"logp": 0.0, "occupancy": 90.0}]
        cls2 = [{"logp": 0.0, "occupancy": 10.0}]
        ab.update_occupancies([cls1, cls2])
        self.assertAlmostEqual(cls1[0]["occupancy"], 90.0)   # the old averages weigh in
