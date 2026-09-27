"""progress_store: what a job's bookkeeping between steps has got to, and how
the job list carries it."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import progress_store  # noqa: E402
import cistem_server  # noqa: E402
import classification  # noqa: E402


class ProgressStoreTests(unittest.TestCase):
    def tearDown(self):
        progress_store.clear("j")

    def test_note_get_clear(self):
        self.assertIsNone(progress_store.get("j"))
        progress_store.note("j", 3, 10, "reading the round's results", "files")
        self.assertEqual(progress_store.get("j"), {"done": 3, "total": 10, "what": "reading the round's results", "unit": "files"})
        progress_store.note("j", 0, 0, "aligning symmetry")
        self.assertEqual(progress_store.get("j")["total"], 0)
        self.assertEqual(cistem_server._finishing("j")["what"], "aligning symmetry")
        progress_store.clear("j")
        self.assertIsNone(progress_store.get("j"))

    def test_a_driver_step_clears_its_note_however_it_ends(self):
        progress_store.note("j", 0, 0, "left over")
        # an unknown project: _child_finished fails to open it; the wrapper clears the note all the same
        with self.assertRaises(Exception):
            classification._child_finished_cleared("no-such-project-" + os.urandom(3).hex(), "child", "j", "completed", None)
        self.assertIsNone(progress_store.get("j"))


if __name__ == "__main__":
    unittest.main()
