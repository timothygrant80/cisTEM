import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import aberrations  # noqa: E402


class BasisTest(unittest.TestCase):
    def test_relion_ordering(self):
        self.assertEqual(aberrations.number_of_odd_coefficients(3), 6)
        self.assertEqual(aberrations.number_of_odd_coefficients(5), 12)
        self.assertEqual(aberrations.number_of_even_coefficients(4), 9)
        self.assertEqual(aberrations.number_of_even_coefficients(6), 16)
        self.assertEqual([aberrations.odd_index_to_mn(i) for i in range(6)],
                         [(-1, 1), (1, 1), (-3, 3), (-1, 3), (1, 3), (3, 3)])
        self.assertEqual([aberrations.even_index_to_mn(i) for i in range(9)],
                         [(0, 0), (-2, 2), (0, 2), (2, 2), (-4, 4), (-2, 4), (0, 4), (2, 4), (4, 4)])

    def test_radial_polynomials(self):
        rho = 0.37
        self.assertAlmostEqual(float(aberrations.radial_polynomial(1, 1, rho)), rho, places=12)
        self.assertAlmostEqual(float(aberrations.radial_polynomial(0, 2, rho)), 2 * rho ** 2 - 1, places=12)
        self.assertAlmostEqual(float(aberrations.radial_polynomial(1, 3, rho)), 3 * rho ** 3 - 2 * rho, places=12)
        self.assertAlmostEqual(float(aberrations.radial_polynomial(0, 4, rho)), 6 * rho ** 4 - 6 * rho ** 2 + 1, places=12)
        self.assertAlmostEqual(float(aberrations.radial_polynomial(2, 4, rho)), 4 * rho ** 4 - 3 * rho ** 2, places=12)
        self.assertEqual(float(aberrations.radial_polynomial(1, 2, rho)), 0.0)
        self.assertAlmostEqual(float(aberrations.zernike(-3, 3, rho, 0.4)), rho ** 3 * math.sin(1.2), places=12)

    def test_half_plane_frequencies(self):
        rho, phi = aberrations.half_plane_frequencies(8, 8, 2.0)
        self.assertEqual(rho.shape, (8, 5))
        self.assertAlmostEqual(rho[0, 4], 4 / 16.0)
        self.assertAlmostEqual(rho[4, 0], 4 / 16.0)          # row ny/2 is -ny/2, as in cisTEM
        self.assertAlmostEqual(phi[7, 0], -math.pi / 2)      # row ny-1 is fy = -1
        self.assertEqual(phi[0, 0], 0.0)

    def test_beam_tilt_round_trip(self):
        c = aberrations.odd_coefficients_for_beam_tilt(0.3, -0.2, 1.5, -0.7, 2.7, 300.0)
        self.assertEqual(c.size, 6)
        self.assertEqual(c[2], 0.0)
        self.assertEqual(c[5], 0.0)
        tilt_x, tilt_y, shift_x, shift_y, rest = aberrations.beam_tilt_from_odd(c + np.array([0, 0, 4.0, 0, 0, -3.0]), 2.7, 300.0)
        self.assertAlmostEqual(tilt_x, 0.3, places=9)
        self.assertAlmostEqual(tilt_y, -0.2, places=9)
        self.assertAlmostEqual(shift_x, 1.5, places=9)
        self.assertAlmostEqual(shift_y, -0.7, places=9)
        np.testing.assert_allclose(rest, [0, 0, 4.0, 0, 0, -3.0])
        # the tilt term is RELION's: Z3x = -2 pi Cs[A] lambda^2 t_x / 3
        scale = 2 * math.pi * 2.7e7 * aberrations.wavelength_a(300.0) ** 2
        self.assertAlmostEqual(c[4], -scale * 0.3e-3 / 3.0)

    def test_a_cs_of_zero_gives_no_tilt_and_keeps_the_coma_terms(self):
        # A Cs-corrected microscope: the tilt's phase term goes through Cs, so no finite tilt stands
        # for the order-3 m = +-1 coefficients; they stay in the set, the shift still comes out.
        c = np.array([2.0 * math.pi * -0.7, 2.0 * math.pi * 1.5, 4.0, 0.25, -0.5, -3.0])
        tilt_x, tilt_y, shift_x, shift_y, rest = aberrations.beam_tilt_from_odd(c, 0.0, 300.0)
        self.assertIsNone(tilt_x)
        self.assertIsNone(tilt_y)
        self.assertAlmostEqual(shift_x, 1.5 - 2.0 * -0.5 / (2.0 * math.pi), places=9)
        self.assertAlmostEqual(shift_y, -0.7 - 2.0 * 0.25 / (2.0 * math.pi), places=9)
        np.testing.assert_allclose(rest, [0, 0, 4.0, 0.25, -0.5, -3.0])
        np.testing.assert_allclose(aberrations.odd_coefficients_for_beam_tilt(0.3, -0.2, 0.0, 0.0, 0.0, 300.0), np.zeros(6))

    def test_full_grid_symmetry(self):
        rho, phi = aberrations.half_plane_frequencies(16, 16, 1.0)
        odd = aberrations.zernike(1, 3, rho, phi)                     # antisymmetric in frequency
        even = aberrations.zernike(2, 4, rho, phi)
        g_odd = aberrations.full_grid(odd, True)
        g_even = aberrations.full_grid(even, False)
        self.assertEqual(g_odd.shape, (16, 16))
        # (fx, fy) at [8 + fy, 8 + fx]; the mirror pixel is (-fx, -fy)
        for fx, fy in [(3, 2), (5, -4), (1, 7), (6, 0)]:
            self.assertAlmostEqual(g_odd[8 + fy, 8 + fx], -g_odd[8 - fy, 8 - fx])
            self.assertAlmostEqual(g_even[8 + fy, 8 + fx], g_even[8 - fy, 8 - fx])
        self.assertAlmostEqual(g_odd[8 + 2, 8 + 3], odd[2, 3])
        self.assertAlmostEqual(g_even[8 - 4, 8 + 5], even[16 - 4, 5])


class FitTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)
        self.rho, self.phi = aberrations.half_plane_frequencies(96, 96, 1.2)

    def test_odd_fit_recovers_the_coefficients(self):
        # a physical set: 0.9 / -0.4 mrad of tilt with a 0.5 A shift, and trefoil
        truth = aberrations.odd_coefficients_for_beam_tilt(0.9, -0.4, 0.5, -0.3, 2.7, 300.0) + np.array([0, 0, 30.0, 0, 0, -25.0])
        Z = aberrations.basis(3, True, self.rho, self.phi)
        weight = self.rng.uniform(0.2, 1.0, self.rho.shape)
        # refine_ctf's sum carries minus the aberration's phase (projection x conj(image))
        S = weight * np.exp(-1j * (np.tensordot(truth, Z, 1) + self.rng.normal(0, 0.02, self.rho.shape)))
        fit = aberrations.fit_odd(S, self.rho, self.phi, order=3, resolution_limit_a=3.0)
        # the orders 1 and 3 are correlated on a small disc, so the noise moves the coefficients more than the phase
        np.testing.assert_allclose(fit["coefficients"], truth, atol=0.1)
        inside = (self.rho > 0) & (self.rho <= 1 / 3.0)
        phase_error = (fit["fitted_phase"] - np.tensordot(truth, Z, 1))[inside]
        self.assertLess(np.sqrt(np.mean(phase_error ** 2)), 0.01)
        # a set already accounted for is removed before fitting and added back
        current = np.array([0.0, 0.0, 30.0, 0.0, 0.0, 0.0])      # part of the truth already accounted for
        fit2 = aberrations.fit_odd(S, self.rho, self.phi, order=3, resolution_limit_a=3.0, current=current)
        # the ridge acts on what is fitted (the residual), so the two agree to its small bias only
        np.testing.assert_allclose(fit2["coefficients"], fit["coefficients"], atol=0.05)
        np.testing.assert_allclose(fit2["residual"], fit2["coefficients"] - current, atol=1e-9)
        self.assertTrue(np.isnan(fit["measured_phase"][0, 0]))
        # the phase outside the resolution limit is not used
        outside = self.rho > 1 / 3.0
        self.assertTrue(np.all(np.isnan(fit["fitted_phase"][outside])))

    def test_odd_fit_handles_a_wrapped_phase(self):
        # a large beam tilt: the phase wraps well inside the limit, the Newton polish must still find it
        truth = aberrations.odd_coefficients_for_beam_tilt(1.5, -1.0, 0.0, 0.0, 2.7, 300.0)
        Z = aberrations.basis(3, True, self.rho, self.phi)
        S = np.exp(-1j * np.tensordot(truth, Z, 1))
        self.assertGreater(np.nanmax(np.abs(np.tensordot(truth, Z, 1)[self.rho <= 1 / 2.5])), math.pi)
        fit = aberrations.fit_odd(S, self.rho, self.phi, order=3, resolution_limit_a=2.5)
        np.testing.assert_allclose(fit["coefficients"], truth, atol=0.05)      # the ridge costs a few hundredths
        inside = (self.rho > 0) & (self.rho <= 1 / 2.5)
        self.assertLess(np.abs(fit["fitted_phase"][inside] - np.tensordot(truth, Z, 1)[inside]).max(), 0.01)

    def test_even_fit_recovers_the_coefficients(self):
        truth = np.zeros(9)
        truth[4], truth[6], truth[8] = 20.0, 5.0, -15.0
        Z = aberrations.basis(4, False, self.rho, self.phi)
        delta = np.tensordot(truth, Z, 1)
        # particles with random CTFs: X = -sin(gamma - delta) V + noise, V of random magnitude
        sums = np.zeros((5,) + self.rho.shape)
        for _ in range(60):
            gamma = math.pi * 0.02 * self.rng.uniform(8000, 20000) * self.rho ** 2 * 1.2 ** 0 + self.rng.uniform(0, 0.3)
            a, b = -np.sin(gamma), np.cos(gamma)
            V = self.rng.uniform(0.5, 2.0, self.rho.shape)
            X = -np.sin(gamma - delta) * V + self.rng.normal(0, 0.3, self.rho.shape)
            sums[0] += a * a * V * V
            sums[1] += a * b * V * V
            sums[2] += b * b * V * V
            sums[3] += a * X * V
            sums[4] += b * X * V
        fit = aberrations.fit_even(sums, self.rho, self.phi, order=4, resolution_limit_a=2.5)
        np.testing.assert_allclose(fit["coefficients"][1:], truth[1:], atol=0.3)
        # the constant is only defined modulo 2 pi: here the true phase at DC is 5 rad
        self.assertAlmostEqual(math.remainder(fit["coefficients"][0] - truth[0], 2 * math.pi), 0.0, places=1)
        inside = self.rho <= 1 / 2.5
        residual = (fit["measured_phase"] - delta)[inside & (fit["weight"] > np.nanpercentile(fit["weight"][inside], 50))]
        self.assertLess(np.nanmedian(np.abs(residual)), 0.1)

    def test_per_pixel_even_phase_is_exact_without_noise(self):
        delta_true = np.array([[0.4, -1.3], [2.9, 0.0]])
        gammas = [0.3, 1.1, 2.0, -0.7]
        sums = np.zeros((5, 2, 2))
        for g in gammas:
            a, b = -math.sin(g), math.cos(g)
            X = -np.sin(g - delta_true)
            sums[0] += a * a
            sums[1] += a * b
            sums[2] += b * b
            sums[3] += a * X
            sums[4] += b * X
        delta, curvature = aberrations.per_pixel_even_phase(*sums)
        np.testing.assert_allclose(delta, delta_true, atol=1e-6)
        self.assertTrue(np.all(curvature > 0))


if __name__ == "__main__":
    unittest.main()
