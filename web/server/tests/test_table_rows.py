"""The drivers' per-round bookkeeping on parameter tables (numpy structured arrays), the form the
drivers hold a class's particles in since 2026-10-03. A 3D classification failed at its second round
with "'numpy.void' object has no attribute 'get'" where the record step still treated a row as a dict."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import autorefine  # noqa: E402
import refinements  # noqa: E402
import starfile  # noqa: E402


def _table(n, **columns):
    t = np.zeros(n, dtype=starfile.table_dtype(refinements.RESULT_KEYS))
    t["position_in_stack"] = np.arange(1, n + 1)
    t["image_is_active"] = 1
    t["occupancy"] = 100.0
    for k, v in columns.items():
        t[k] = v
    return t


class AverageOccupancyTests(unittest.TestCase):
    def test_over_active_particles_of_a_table(self):
        t = _table(4, occupancy=[50.0, 30.0, 10.0, 90.0], image_is_active=[1, 0, -1, 1])
        self.assertAlmostEqual(refinements.average_occupancy(t), (50.0 + 30.0 + 90.0) / 3.0, places=4)

    def test_no_active_particles_is_zero(self):
        self.assertEqual(refinements.average_occupancy(_table(2, image_is_active=-1)), 0.0)

    def test_dict_rows_still_work(self):
        rows = [{"position_in_stack": 1, "occupancy": 40.0}, {"position_in_stack": 2, "occupancy": 60.0, "image_is_active": -1}]
        self.assertAlmostEqual(refinements.average_occupancy(rows), 40.0, places=5)   # a row without the flag counts as active

    def test_a_table_row_is_not_a_dict(self):
        # the failure's shape: iterating a table gives numpy.void rows, which have no .get()
        row = _table(1)[0]
        self.assertFalse(hasattr(row, "get"))
        self.assertEqual(float(row["occupancy"]), 100.0)


class TrackingUpdateTests(unittest.TestCase):
    def test_global_searches_are_counted_and_the_rest_age(self):
        tracking = {"globals": np.zeros(5, dtype=np.int64), "since_global": np.array([3, 3, 3, 3, 3], dtype=np.int64),
                    "last_global_res": np.full(5, 20.0)}
        inp = _table(5, image_is_active=[0, 1, 0, 1, 0])
        out = _table(5, image_is_active=[1, 1, 1, 1, 0])   # particles 1 and 3 went through a global search
        autorefine.update_tracking(tracking, inp, out, 9.5)
        self.assertEqual(tracking["globals"].tolist(), [1, 0, 1, 0, 0])
        self.assertEqual(tracking["since_global"].tolist(), [0, 4, 0, 4, 4])
        self.assertEqual(tracking["last_global_res"].tolist(), [9.5, 20.0, 9.5, 20.0, 20.0])

    def test_shorter_tables_leave_the_rest_ageing(self):
        tracking = {"globals": np.zeros(3, dtype=np.int64), "since_global": np.zeros(3, dtype=np.int64), "last_global_res": np.zeros(3)}
        autorefine.update_tracking(tracking, _table(2, image_is_active=0), _table(2, image_is_active=1), 8.0)
        self.assertEqual(tracking["globals"].tolist(), [1, 1, 0])
        self.assertEqual(tracking["since_global"].tolist(), [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
