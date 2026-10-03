"""cisTEM's binary parameter files (starfile.read_cistem_binary / write_cistem_binary),
checked against a file the C++ wrote: cisTEMParameters::WriteTocisTEMBinaryFile on this
branch, three particles with the 24 refinement columns (FIXTURE_HEX, 512 bytes), and the
text star the same object wrote beside it (FIXTURE_STAR)."""

import binascii
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import starfile  # noqa: E402

FIXTURE_HEX = (
    "180000000300000001000000000000000904000000000000000300004000000000000300008000000000000308000000"
    "000000000310000000000000000320000000000000000340000000000000000380000000000000000300010000000000"
    "000302000000000000000200020000000000000300040000000000000300080000000000000300100000000000000300"
    "400000000000000300800000000000000300000100000000000300000200000000000300000400000000000300000800"
    "000000000300001000000000000300002000000000000300000000010000000201000000000028410000a2410000f141"
    "000000800000000000606a46004067460000044200000000010000000000c842004a23c6000074410000a6410ad7833f"
    "00009643cdcc2c40295c8f3d0000000000000080000000000000008001000000020000000000a8410000224200007142"
    "0000c0bf0000204000646a460044674600000442cdcccc3dffffffff0000c642004e23c6000082410000ae410ad7833f"
    "00009643cdcc2c40295c8f3d6f12833a6f1203bb0000003f000080be02000000030000000000fc410000734200c0b442"
    "000040c00000a04000686a460048674600000442cdcc4c3e010000000000c442005223c600008a410000b6410ad7833f"
    "00009643cdcc2c40295c8f3d6f12033b6f1283bb0000803f000000bf01000000"
)
FIXTURE_STAR = """# Written by cisTEM Version undefined on 2026-10-03 15:32:04
 
data_
 
loop_
_cisTEMPositionInStack #1
_cisTEMAnglePsi #2
_cisTEMAngleTheta #3
_cisTEMAnglePhi #4
_cisTEMXShift #5
_cisTEMYShift #6
_cisTEMDefocus1 #7
_cisTEMDefocus2 #8
_cisTEMDefocusAngle #9
_cisTEMPhaseShift #10
_cisTEMImageActivity #11
_cisTEMOccupancy #12
_cisTEMLogP #13
_cisTEMSigma #14
_cisTEMScore #15
_cisTEMPixelSize #16
_cisTEMMicroscopeVoltagekV #17
_cisTEMMicroscopeCsMM #18
_cisTEMAmplitudeContrast #19
_cisTEMBeamTiltX #20
_cisTEMBeamTiltY #21
_cisTEMImageShiftX #22
_cisTEMImageShiftY #23
_cisTEMAssignedSubset #24
#    POS     PSI   THETA     PHI       SHX       SHY      DF1      DF2  ANGAST  PSHIFT  STAT     OCC      LogP      SIGMA   SCORE    PSIZE    VOLT      Cs    AmpC  BTILTX  BTILTY  ISHFTX  ISHFTY  SUBSET 
       1   10.50   20.25   30.12     -0.00      0.00  15000.0  14800.0   33.00    0.00     1  100.00    -10451    15.2500   20.75  1.03000  300.00    2.70  0.0700   0.000  -0.000   0.000  -0.000        1 
       2   21.00   40.50   60.25     -1.50      2.50  15001.0  14801.0   33.00    0.10    -1   99.00    -10452    16.2500   21.75  1.03000  300.00    2.70  0.0700   0.001  -0.002   0.500  -0.250        2 
       3   31.50   60.75   90.38     -3.00      5.00  15002.0  14802.0   33.00    0.20     1   98.00    -10453    17.2500   22.75  1.03000  300.00    2.70  0.0700   0.002  -0.004   1.000  -0.500        1 
"""


class BinaryParameterFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.binary = os.path.join(self.tmp, "fixture.cistem")
        with open(self.binary, "wb") as fh:
            fh.write(binascii.unhexlify("".join(FIXTURE_HEX)))
        self.text = os.path.join(self.tmp, "fixture.star")
        with open(self.text, "w") as fh:
            fh.write(FIXTURE_STAR)

    def test_reads_what_the_cpp_wrote(self):
        rows = starfile.read_cistem_binary(self.binary)
        text = starfile.read_star(self.text)
        self.assertEqual(len(rows), 3)
        self.assertEqual(sorted(rows[0]), sorted(text[0]))
        self.assertEqual([r["position_in_stack"] for r in rows], [1, 2, 3])
        self.assertEqual([r["image_is_active"] for r in rows], [1, -1, 1])
        self.assertEqual([r["assigned_subset"] for r in rows], [1, 2, 1])
        self.assertAlmostEqual(rows[1]["psi"], 21.0, places=5)
        self.assertAlmostEqual(rows[2]["logp"], -10452.5, places=3)
        self.assertAlmostEqual(rows[1]["beam_tilt_y"], -0.002, places=6)
        for r, t in zip(rows, text):   # the text file prints to limited precision; the binary agrees within it
            for k in r:
                self.assertAlmostEqual(float(r[k]), float(t[k]), delta=0.01 * max(1.0, abs(float(t[k]))), msg=k)

    def test_writes_byte_identical_to_the_cpp(self):
        rows = starfile.read_cistem_binary(self.binary)
        out = os.path.join(self.tmp, "rewritten.cistem")
        starfile.write_cistem_binary(out, rows)
        with open(self.binary, "rb") as a, open(out, "rb") as b:
            self.assertEqual(a.read(), b.read())
        # From a table too, whatever column order the table came in.
        table = starfile.read_cistem_binary(self.binary, as_table=True)
        shuffled = table[list(reversed(table.dtype.names))]
        starfile.write_cistem_binary(out, shuffled)
        with open(self.binary, "rb") as a, open(out, "rb") as b:
            self.assertEqual(a.read(), b.read())

    def test_tables_and_dispatch(self):
        table = starfile.read_params(self.binary, as_table=True)
        self.assertIsInstance(table, np.ndarray)
        self.assertEqual(table.dtype["position_in_stack"], np.dtype("<u4"))
        self.assertEqual(table.dtype["image_is_active"], np.dtype("<i4"))
        self.assertEqual(table.dtype["psi"], np.dtype("<f4"))
        np.testing.assert_allclose(table["theta"], [20.25, 40.5, 60.75])
        rows = starfile.table_to_rows(table)
        self.assertEqual(rows, starfile.read_params(self.binary))
        back = starfile.rows_to_table(rows)
        self.assertEqual(back.dtype, table.dtype)
        self.assertTrue((back == table).all())
        # Text and binary by extension, both ways.
        out_star = os.path.join(self.tmp, "o.star"); out_bin = os.path.join(self.tmp, "o.cistem")
        starfile.write_params(out_star, table)
        starfile.write_params(out_bin, rows)
        self.assertEqual(starfile.read_params(out_bin), rows)
        again = starfile.read_params(out_star)
        self.assertEqual([r["position_in_stack"] for r in again], [1, 2, 3])
        self.assertAlmostEqual(again[1]["sigma"], 16.25, places=3)

    def test_string_columns_refused_in_tables_but_read(self):
        with self.assertRaises(ValueError):
            starfile.table_dtype(("position_in_stack", "stack_filename"))
        # A binary file with a string column: header + one record, by hand.
        path = os.path.join(self.tmp, "s.cistem")
        with open(path, "wb") as fh:
            fh.write(np.array([2, 1], dtype="<i4").tobytes())
            fh.write(np.array([1], dtype="<i8").tobytes() + bytes([9]))           # position_in_stack, unsigned
            fh.write(np.array([16777216], dtype="<i8").tobytes() + bytes([8]))    # stack_filename, variable length
            fh.write(np.array([7], dtype="<u4").tobytes())
            fh.write(np.array([5], dtype="<i4").tobytes() + b"a.mrc")
        rows = starfile.read_cistem_binary(path)
        self.assertEqual(rows, [{"position_in_stack": 7, "stack_filename": "a.mrc"}])


if __name__ == "__main__":
    unittest.main()


class FieldSubsetTableTests(unittest.TestCase):
    def test_a_field_subset_view_is_written_packed_and_in_column_order(self):
        """table[["a", "b"]] keeps the parent's itemsize and offsets; the file must carry
        packed records in cisTEM's column order, with the columns matched by name."""
        import tempfile
        import numpy as np
        keys = ("position_in_stack", "psi", "best_2d_class", "sigma")
        t = np.zeros(10, dtype=starfile.table_dtype(keys))
        t["position_in_stack"] = np.arange(1, 11)
        t["psi"] = np.arange(10) * 1.5
        t["best_2d_class"] = 7
        t["sigma"] = 2.0
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "v.cistem")
            starfile.write_cistem_binary(path, t[["position_in_stack", "best_2d_class", "psi"]], ("position_in_stack", "best_2d_class", "psi"))
            back = starfile.read_cistem_binary(path, as_table=True)
        self.assertEqual(list(back.dtype.names), ["position_in_stack", "psi", "best_2d_class"])
        self.assertEqual(back["psi"].tolist(), t["psi"].tolist())
        self.assertEqual(back["best_2d_class"].tolist(), [7] * 10)
        self.assertEqual(back["position_in_stack"].tolist(), list(range(1, 11)))


class ContentSniffTests(unittest.TestCase):
    def test_a_text_star_under_a_cistem_name_is_read_as_text(self):
        import tempfile
        rows = [{"position_in_stack": 1, "psi": 10.0, "sigma": 2.0}, {"position_in_stack": 2, "psi": 20.0, "sigma": 3.0}]
        with tempfile.TemporaryDirectory() as d:
            text_as_cistem = os.path.join(d, "out.cistem")
            starfile.write_star(text_as_cistem, rows, ("position_in_stack", "psi", "sigma"))   # an old program's output
            self.assertFalse(starfile.is_binary_file(text_as_cistem))
            with self.assertLogs("starfile", level="WARNING"):
                back = starfile.read_params(text_as_cistem, as_table=True)
            self.assertEqual(back["psi"].tolist(), [10.0, 20.0])
            binary = os.path.join(d, "b.cistem")
            starfile.write_params(binary, rows, ("position_in_stack", "psi", "sigma"))
            self.assertTrue(starfile.is_binary_file(binary))
            self.assertEqual(starfile.read_params(binary, as_table=True)["sigma"].tolist(), [2.0, 3.0])
