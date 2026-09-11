"""preview.py: the Find Particles panels' Low-pass filter."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import preview  # noqa: E402


class GaussianLowPassTests(unittest.TestCase):
    def test_constant_image_is_unchanged(self):
        image = np.full((64, 48), 3.5, dtype=np.float32)
        out = preview.gaussian_low_pass(image, pixel_size=1.0, resolution_a=20.0)
        self.assertEqual(out.shape, image.shape)
        np.testing.assert_allclose(out, image, atol=1e-5)

    def test_mean_is_kept_and_high_frequencies_are_removed(self):
        rng = np.random.default_rng(1)
        image = rng.normal(10.0, 1.0, size=(128, 128)).astype(np.float32)
        out = preview.gaussian_low_pass(image, pixel_size=1.0, resolution_a=20.0)
        self.assertAlmostEqual(float(out.mean()), float(image.mean()), places=4)
        # White noise filtered to 20 A at 1 A/px keeps only the lowest ~5% of
        # frequencies, so its variance collapses.
        self.assertLess(float(out.std()), 0.3 * float(image.std()))
        # The Nyquist checkerboard is gone entirely.
        checker = np.indices((64, 64)).sum(axis=0) % 2 * 2.0 - 1.0
        flat = preview.gaussian_low_pass(checker.astype(np.float32), pixel_size=1.0, resolution_a=10.0)
        self.assertLess(float(np.abs(flat).max()), 1e-3)

    def test_weight_follows_cistem(self):
        # Image::GaussianLowPassFilter(sigma): exp(-f^2 / (2 sigma^2)) with
        # sigma = (pixel size / resolution) * sqrt(2) -- so a cosine at the
        # filter resolution keeps exp(-1/4) of its amplitude.
        n = 256
        x = np.arange(n, dtype=np.float32)
        wave = np.cos(2 * np.pi * x / 16.0)              # period 16 px = 16 A at 1 A/px
        image = np.tile(wave, (32, 1))
        out = preview.gaussian_low_pass(image, pixel_size=1.0, resolution_a=16.0)
        self.assertAlmostEqual(float(out[0].max()), float(np.exp(-0.25)), places=3)

    def test_no_pixel_size_leaves_the_image_alone(self):
        image = np.random.default_rng(2).normal(size=(16, 16)).astype(np.float32)
        self.assertIs(preview.gaussian_low_pass(image, pixel_size=None, resolution_a=20.0), image)


class HighPassTests(unittest.TestCase):
    def test_taper_edges_removes_the_step_between_opposite_edges(self):
        # A ramp across x wraps with a big step; after tapering the two edges meet.
        x = np.linspace(0.0, 10.0, 300, dtype=np.float32)
        image = np.tile(x, (90, 1))
        out = preview.taper_edges(image)
        self.assertEqual(out.shape, image.shape)
        step_before = abs(float(image[:, 0].mean() - image[:, -1].mean()))
        step_after = abs(float(out[:, 0].mean() - out[:, -1].mean()))
        self.assertLess(step_after, 0.15 * step_before)
        # The middle of the picture is untouched (the taper is N/30 wide).
        np.testing.assert_allclose(out[:, 20:280], image[:, 20:280])

    def test_high_pass_weight_follows_cistem(self):
        # CosineMask(r, 2r, invert) with r = 8 / width: DC gone, half weight at f = r, full from 2r.
        w = preview.high_pass_weight((256, 512))
        r = 8.0 / 512
        self.assertEqual(float(w[0, 0]), 0.0)
        fx = np.fft.rfftfreq(512)
        at_r = int(np.argmin(np.abs(fx - r)))
        self.assertAlmostEqual(float(w[0, at_r]), 0.5, places=2)
        self.assertEqual(float(w[0, int(np.argmin(np.abs(fx - 3 * r)))]), 1.0)

    def test_filter_preview_removes_a_ramp_and_keeps_detail(self):
        rng = np.random.default_rng(3)
        yy, xx = np.mgrid[0:240, 0:300]
        ramp = (xx / 300.0) * 50.0 + (yy / 240.0) * 30.0
        detail = rng.normal(0.0, 1.0, size=ramp.shape)
        image = (ramp + detail).astype(np.float32)
        out = preview.filter_preview(image, pixel_size=1.0, lowpass_a=None, highpass=True)
        # The ramp spanned 80 grey levels (std 16.8); away from the N/30 taper
        # strips what is left is the noise at about its own scale, mean zero.
        interior = out[20:220, 20:280]
        self.assertLess(float(interior.std()), 1.3)
        self.assertGreater(float(interior.std()), 0.8)
        self.assertAlmostEqual(float(out.mean()), 0.0, places=3)
        # The noise itself passes almost unchanged.
        flat = preview.filter_preview(detail.astype(np.float32), pixel_size=1.0, lowpass_a=None, highpass=True)
        self.assertAlmostEqual(float(flat.std()), float(detail.std()), delta=0.02)
        # Nothing asked for: the image comes back as it was.
        self.assertIs(preview.filter_preview(image), image)


if __name__ == "__main__":
    unittest.main()
