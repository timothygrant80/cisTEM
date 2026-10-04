"""The rendered orthogonal views kept beside a volume and in memory (volumes.orthogonal_views_png_cached)."""
import os
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import volumes  # noqa: E402


class OrthViewsCacheTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.d.name, "vol.mrc")
        rng = np.random.default_rng(1)
        volumes.write_mrc_volume(self.path, rng.normal(size=(32, 32, 32)).astype(np.float32), 1.5)
        volumes._ORTH_CACHE.clear()
        self.render = volumes.orthogonal_views_png
        self.calls = 0

        def counting(path, mask_radius_a=0.0, panel=volumes.ORTH_PANEL):
            out = self.render(path, mask_radius_a, panel)
            self.calls += 1   # renders that succeeded
            return out
        volumes.orthogonal_views_png = counting

    def tearDown(self):
        volumes.orthogonal_views_png = self.render
        volumes._ORTH_CACHE.clear()
        self.d.cleanup()

    def test_rendered_once_then_served_from_memory_and_disk(self):
        png, meta = volumes.orthogonal_views_png_cached(self.path)
        self.assertEqual(self.calls, 1)
        side = volumes.orth_views_file(self.path)
        self.assertTrue(os.path.isfile(side))
        with open(side, "rb") as fh:
            self.assertEqual(fh.read(), png)
        direct_png, direct_meta = self.render(self.path)
        self.assertEqual(direct_png, png)
        self.assertEqual(meta, direct_meta)   # the header-only meta equals the renderer's
        # memory
        png2, _ = volumes.orthogonal_views_png_cached(self.path)
        self.assertEqual(self.calls, 1)
        self.assertEqual(png2, png)
        # disk, after the server restarts
        volumes._ORTH_CACHE.clear()
        png3, meta3 = volumes.orthogonal_views_png_cached(self.path)
        self.assertEqual(self.calls, 1)
        self.assertEqual(png3, png)
        self.assertEqual(meta3, direct_meta)

    def test_a_rewritten_volume_is_rendered_again(self):
        volumes.orthogonal_views_png_cached(self.path)
        self.assertEqual(self.calls, 1)
        side = volumes.orth_views_file(self.path)
        later = time.time() + 5
        os.utime(self.path, (later, later))   # the map rewritten under the same name (ab-initio's rounds)
        volumes.orthogonal_views_png_cached(self.path)
        self.assertEqual(self.calls, 2)
        self.assertGreaterEqual(os.path.getmtime(side), later - 1.0)

    def test_mask_radius_has_its_own_picture(self):
        volumes.orthogonal_views_png_cached(self.path)
        volumes.orthogonal_views_png_cached(self.path, 20.0)
        self.assertEqual(self.calls, 2)
        self.assertNotEqual(volumes.orth_views_file(self.path), volumes.orth_views_file(self.path, 20.0))
        self.assertEqual(sorted(volumes.orth_views_companions(self.path)),
                         sorted([volumes.orth_views_file(self.path), volumes.orth_views_file(self.path, 20.0)]))

    def test_a_corrupt_picture_on_disk_is_replaced(self):
        side = volumes.orth_views_file(self.path)
        with open(side, "wb") as fh:
            fh.write(b"not a png")
        later = time.time() + 5
        os.utime(side, (later, later))
        png, _ = volumes.orthogonal_views_png_cached(self.path)
        self.assertEqual(self.calls, 1)
        self.assertTrue(png.startswith(b"\x89PNG"))
        with open(side, "rb") as fh:
            self.assertEqual(fh.read(), png)

    def test_prepare_skips_missing_and_broken_volumes(self):
        notes = []
        broken = os.path.join(self.d.name, "broken.mrc")
        with open(broken, "wb") as fh:
            fh.write(b"\0" * 2048)
        volumes.prepare_orth_views([self.path, None, os.path.join(self.d.name, "absent.mrc"), broken], 0.0, notes.append)
        self.assertEqual(self.calls, 1)
        self.assertTrue(os.path.isfile(volumes.orth_views_file(self.path)))
        self.assertEqual(len(notes), 1)
        self.assertIn("broken.mrc", notes[0])


if __name__ == "__main__":
    unittest.main()
