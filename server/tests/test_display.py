"""server/display.py: sections of MRC stacks and TIFFs for the Display panel."""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import display  # noqa: E402
import refinement_packages as rp  # noqa: E402


class DisplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.stack = os.path.join(self.tmp, "stack.mrcs")
        w = rp.MrcStackWriter(self.stack, 40, 2.0)
        for i in range(3):
            w.append(np.full((40, 40), float(i + 1), dtype=np.float32))
        w.close()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_info_and_sections(self):
        info = display.file_info(self.stack)
        self.assertEqual((info["nx"], info["ny"], info["nz"], info["pixel_size"], info["format"]), (40, 40, 3, 2.0, "mrc"))
        data, info = display.read_section(self.stack, 2)
        self.assertEqual((data.shape, data.dtype.str, info["bin"], info["section"]), ((40, 40), "<f4", 1, 2))
        self.assertEqual((info["min"], info["max"], info["mean"]), (2.0, 2.0, 2.0))
        # section 0 is the sum
        data, info = display.read_section(self.stack, 0)
        self.assertEqual((float(data[0, 0]), info["summed"]), (6.0, 3))
        with self.assertRaises(display.DisplayError):
            display.read_section(self.stack, 4)

    def test_binning(self):
        img = np.arange(100 * 60, dtype=np.float32).reshape(60, 100)
        binned, factor = display.bin_image(img, 30)
        self.assertEqual((factor, binned.shape), (4, (15, 25)))
        self.assertAlmostEqual(float(binned[0, 0]), float(img[:4, :4].mean()), places=3)
        data, info = display.read_section(self.stack, 1, max_edge=16)
        self.assertEqual((info["bin"], info["width"], info["height"]), (3, 13, 13))

    def test_global_range(self):
        r = display.global_range(self.stack)
        self.assertEqual((r["min"], r["max"], r["sections"]), (1.0, 3.0, 3))

    def test_tiff(self):
        from PIL import Image
        path = os.path.join(self.tmp, "img.tif")
        frames = [Image.fromarray(np.full((8, 12), v, dtype=np.uint16)) for v in (5, 9)]
        frames[0].save(path, save_all=True, append_images=frames[1:])
        info = display.file_info(path)
        self.assertEqual((info["nx"], info["ny"], info["nz"], info["format"]), (12, 8, 2, "tiff"))
        data, info = display.read_section(path, 2)
        self.assertEqual((data.shape, float(data[0, 0])), ((8, 12), 9.0))

    def test_rejects_other_files(self):
        with self.assertRaises(display.DisplayError):
            display.file_info(os.path.join(self.tmp, "nothing.mrc"))
        with self.assertRaises(display.DisplayError):
            display.file_info(os.path.join(self.tmp, "stack.eer"))


if __name__ == "__main__":
    unittest.main()
