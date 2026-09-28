"""Auto Refine's reconstruct3d input: particles refine3d skipped this round are
ranked below every refined one, so the score-percentage threshold picks among
this round's refined particles rather than across stale scores from earlier
rounds at other resolution limits."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import autorefine  # noqa: E402


class RowsForReconstructionTests(unittest.TestCase):
    def test_inactive_rows_rank_below_the_refined_minimum_and_the_stored_rows_keep_theirs(self):
        rows = [{"position_in_stack": 1, "image_is_active": 1, "score": 12.0}, {"position_in_stack": 2, "image_is_active": -1, "score": 30.0},
                {"position_in_stack": 3, "image_is_active": 1, "score": 9.5}, {"position_in_stack": 4, "image_is_active": -1, "score": 25.0}]
        out = autorefine._rows_for_reconstruction(rows)
        self.assertEqual([r["score"] for r in out], [12.0, 8.5, 9.5, 8.5])
        self.assertEqual([r["score"] for r in rows], [12.0, 30.0, 9.5, 25.0])   # the originals untouched

    def test_all_inactive_or_all_active_are_left_sensible(self):
        self.assertEqual([r["score"] for r in autorefine._rows_for_reconstruction([{"image_is_active": -1, "score": 5.0}])], [-1.0])
        self.assertEqual([r["score"] for r in autorefine._rows_for_reconstruction([{"image_is_active": 1, "score": 5.0}])], [5.0])


if __name__ == "__main__":
    unittest.main()
