"""append_log(): concurrent writers to one job's log must never collide on
SEQ (the runner and a driver thread log the same job within milliseconds)."""
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db  # noqa: E402
import cistem_server  # noqa: E402


class AppendLogConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._root = db.PROJECTS_ROOT
        db.PROJECTS_ROOT = Path(self.tmp)
        self.project = "t-" + os.path.basename(self.tmp)[-6:]
        (db.PROJECTS_ROOT / self.project).mkdir(parents=True)
        db.get_conn(self.project).close()

    def tearDown(self):
        db.PROJECTS_ROOT = self._root

    def test_two_threads_interleaving_lines_keep_every_line(self):
        errors = []

        def writer(tag):
            try:
                for i in range(150):
                    cistem_server.append_log(self.project, "job1", "{} {}".format(tag, i))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(t,)) for t in ("runner", "driver", "flask")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        conn = db.get_conn(self.project)
        n, distinct = conn.execute("SELECT COUNT(*), COUNT(DISTINCT SEQ) FROM JOB_LOG_LINES WHERE JOB_ID='job1'").fetchone()
        conn.close()
        self.assertEqual((n, distinct), (450, 450))


if __name__ == "__main__":
    unittest.main()
