"""db.get_conn(): opening a project database must not need the write lock
once the database is up to date, or every open would queue behind whatever
is writing (the crash at the end of a Find CTF job: "database is locked")."""
import os
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db  # noqa: E402


class OpenWithoutWritingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._root = db.PROJECTS_ROOT
        db.PROJECTS_ROOT = Path(self.tmp)
        self.project = "t-" + os.path.basename(self.tmp)[-6:]
        (db.PROJECTS_ROOT / self.project).mkdir(parents=True)

    def tearDown(self):
        db.PROJECTS_ROOT = self._root

    def test_first_open_seeds_later_opens_do_not_write(self):
        first = db.get_conn(self.project)
        self.assertEqual(first.execute("SELECT GROUP_NAME FROM IMAGE_GROUP_LIST WHERE GROUP_ID=0").fetchone()[0], "All Images")
        self.assertIn("SAMPLE_THICKNESS", {r[1] for r in first.execute("PRAGMA table_info(ESTIMATED_CTF_PARAMETERS)")})
        # Another connection holds the write lock for a long transaction...
        writer = sqlite3.connect(str(db.project_db_path(self.project)), check_same_thread=False)
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("INSERT INTO IMAGE_ASSETS(IMAGE_ASSET_ID, NAME) VALUES (1, 'x')")
        # ...and opening the project must still come back at once and be usable for reading.
        started = time.time()
        second = db.get_conn(self.project)
        self.assertLess(time.time() - started, 2.0)
        self.assertEqual(second.execute("SELECT COUNT(*) FROM IMAGE_GROUP_LIST").fetchone()[0], 1)
        writer.rollback()
        writer.close()
        first.close()
        second.close()

    def test_recreated_database_is_prepared_again(self):
        conn = db.get_conn(self.project)
        conn.close()
        os.remove(db.project_db_path(self.project))
        for suffix in ("-wal", "-shm"):
            try:
                os.remove(str(db.project_db_path(self.project)) + suffix)
            except FileNotFoundError:
                pass
        conn = db.get_conn(self.project)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM IMAGE_GROUP_LIST WHERE GROUP_ID=0").fetchone()[0], 1)
        conn.close()

    def test_missing_members_are_seeded_on_first_open(self):
        conn = db.get_conn(self.project)
        with conn:
            conn.execute("INSERT INTO IMAGE_ASSETS(IMAGE_ASSET_ID, NAME) VALUES (5, 'y')")
        conn.close()
        db._prepared_databases.clear()  # as a new server process would see it
        conn = db.get_conn(self.project)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM IMAGE_GROUP_MEMBERS WHERE GROUP_ID=0 AND IMAGE_ASSET_ID=5").fetchone()[0], 1)
        conn.close()


if __name__ == "__main__":
    unittest.main()
