"""Ids handed out ahead of their rows stay out of reach of a second job on the
same project until the row is written -- two 3D jobs used to take the same
refinement id and overwrite each other's volumes and results."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db  # noqa: E402
import refinements  # noqa: E402
import refinement_packages as rp  # noqa: E402


class IdReservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._root = db.PROJECTS_ROOT
        db.PROJECTS_ROOT = Path(self.tmp)
        (db.PROJECTS_ROOT / "p").mkdir()
        self.conn = db.get_conn("p")
        with refinements._ID_LOCK:
            refinements._RESERVED.clear()

    def tearDown(self):
        self.conn.close()
        db.PROJECTS_ROOT = self._root
        with refinements._ID_LOCK:
            refinements._RESERVED.clear()

    def test_two_jobs_get_different_refinement_ids_before_either_writes(self):
        a = refinements.next_refinement_id(self.conn)
        b = refinements.next_refinement_id(self.conn)
        self.assertEqual((a, b), (1, 2))
        refinements.release_id("refinement", a)          # a's row written (or a gave up)
        self.assertEqual(refinements.next_refinement_id(self.conn), 3)   # still above b, which is out and unwritten
        with self.conn:
            self.conn.execute("INSERT INTO REFINEMENT_LIST(REFINEMENT_ID, REFINEMENT_PACKAGE_ASSET_ID, NAME, RESOLUTION_STATISTICS_ARE_GENERATED, DATETIME_OF_RUN, "
                              "STARTING_REFINEMENT_ID, NUMBER_OF_PARTICLES, NUMBER_OF_CLASSES, RESOLUTION_STATISTICS_BOX_SIZE, RESOLUTION_STATISTICS_PIXEL_SIZE, PERCENT_USED) "
                              "VALUES (5, 1, 'x', 1, 0, -1, 1, 1, 32, 1.0, 100.0)")
        self.assertEqual(refinements.next_refinement_id(self.conn), 6)   # above the table too

    def test_reconstruction_and_package_ids_the_same_way(self):
        self.assertEqual((refinements.next_reconstruction_id(self.conn), refinements.next_reconstruction_id(self.conn)), (1, 2))
        self.assertEqual((rp.next_package_id(self.conn), rp.next_package_id(self.conn)), (1, 2))


if __name__ == "__main__":
    unittest.main()
