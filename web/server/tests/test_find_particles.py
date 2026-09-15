"""stages/find_particles.py: the manual-edit save (replace_picks)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db  # noqa: E402
from stages import find_particles as fp  # noqa: E402


class ReplacePicksTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._root = db.PROJECTS_ROOT
        db.PROJECTS_ROOT = Path(self.tmp)
        self.project = "t-" + os.path.basename(self.tmp)[-6:]
        (db.PROJECTS_ROOT / self.project).mkdir(parents=True)
        self.conn = db.get_conn(self.project)
        c = self.conn
        with c:
            c.execute("INSERT INTO IMAGE_ASSETS(IMAGE_ASSET_ID, NAME, FILENAME, X_SIZE, Y_SIZE, PIXEL_SIZE) VALUES (7, 'img', '/none.mrc', 100, 100, 1.0)")
            c.execute("INSERT INTO JOBS(JOB_ID, JOB_NUMBER, STAGE, STATUS, CREATED_AT) VALUES ('job1', 1, 'particle_picking', 'completed', 0)")
            c.execute("INSERT INTO PARTICLE_PICKING_LIST(PICKING_ID, DATETIME_OF_RUN, PICKING_JOB_ID, PARENT_IMAGE_ASSET_ID, PICKING_ALGORITHM, "
                      "CHARACTERISTIC_RADIUS, MAXIMUM_RADIUS, THRESHOLD_PEAK_HEIGHT, HIGHEST_RESOLUTION_USED_IN_PICKING, MIN_DIST_FROM_EDGES, "
                      "AVOID_HIGH_VARIANCE, AVOID_HIGH_LOW_MEAN, NUM_BACKGROUND_BOXES, MANUAL_EDIT) VALUES (3, 0, 'job1', 7, 0, 80, 120, 6, 15, 81, 1, 0, 40, 0)")
            c.execute(fp._RESULTS_TABLE_SQL.format(fp.results_table("job1")))
            c.executemany("INSERT INTO {}(PICKING_ID, PARENT_IMAGE_ASSET_ID, X_POSITION, Y_POSITION, PEAK_HEIGHT, TEMPLATE_ASSET_ID, TEMPLATE_PSI, "
                          "TEMPLATE_THETA, TEMPLATE_PHI) VALUES (3, 7, ?, ?, ?, -1, 0, 0, 0)".format(fp.results_table("job1")),
                          [(100.0, 200.0, 8.0), (300.0, 400.0, 7.0), (500.0, 600.0, 6.5)])
            fp._replace_active_picks(c, 7, 3, "job1")

    def tearDown(self):
        self.conn.close()
        db.PROJECTS_ROOT = self._root

    def positions(self):
        return [(p["x"], p["y"], p["peak_height"]) for p in fp.picks_for(self.conn, 3)]

    def test_edit_replaces_results_marks_manual_and_rebuilds_active_assets(self):
        self.assertEqual(len(self.positions()), 3)
        image_id = fp.replace_picks(self.conn, 3, [{"x": 100.0, "y": 200.0, "peak_height": 8.0}, {"x": 900.0, "y": 950.0}])
        self.assertEqual(image_id, 7)
        # The removed pick is gone, the added one has peak height 0 (cisTEM's default for a hand pick).
        self.assertEqual(self.positions(), [(100.0, 200.0, 8.0), (900.0, 950.0, 0.0)])
        self.assertEqual(self.conn.execute("SELECT MANUAL_EDIT FROM PARTICLE_PICKING_LIST WHERE PICKING_ID=3").fetchone()[0], 1)
        assets = self.conn.execute("SELECT X_POSITION, Y_POSITION FROM PARTICLE_POSITION_ASSETS WHERE PARENT_IMAGE_ASSET_ID=7 ORDER BY X_POSITION").fetchall()
        self.assertEqual([tuple(r) for r in assets], [(100.0, 200.0), (900.0, 950.0)])
        # Every asset is in All Particle Positions.
        n = self.conn.execute("SELECT COUNT(*) FROM PARTICLE_POSITION_GROUP_MEMBERS m JOIN PARTICLE_POSITION_ASSETS a "
                              "ON a.PARTICLE_POSITION_ASSET_ID = m.PARTICLE_POSITION_ASSET_ID WHERE m.GROUP_ID = 0 AND a.PARENT_IMAGE_ASSET_ID = 7").fetchone()[0]
        self.assertEqual(n, 2)

    def test_edit_of_an_inactive_picking_leaves_the_assets_alone(self):
        with self.conn:
            self.conn.execute("UPDATE IMAGE_ASSETS SET ACTIVE_PICKING_ID = 99 WHERE IMAGE_ASSET_ID = 7")
        fp.replace_picks(self.conn, 3, [])
        self.assertEqual(self.positions(), [])
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM PARTICLE_POSITION_ASSETS WHERE PARENT_IMAGE_ASSET_ID=7").fetchone()[0], 3)

    def test_unknown_picking(self):
        self.assertIsNone(fp.replace_picks(self.conn, 42, []))


if __name__ == "__main__":
    unittest.main()
