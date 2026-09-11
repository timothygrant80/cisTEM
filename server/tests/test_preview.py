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


if __name__ == "__main__":
    unittest.main()
