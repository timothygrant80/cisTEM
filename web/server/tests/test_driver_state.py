"""Driver state under concurrency and damage: a child's progress is written
into STATE_JSON in place rather than as a whole stale copy; a merge step
refuses to guess a count the launch step did not record; a round whose tasks
did not return every particle fails instead of keeping last round's rows."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db  # noqa: E402
import abinitio  # noqa: E402
import classification  # noqa: E402
import starfile  # noqa: E402


class ChildProgressTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._root = db.PROJECTS_ROOT
        db.PROJECTS_ROOT = Path(self.tmp)
        (db.PROJECTS_ROOT / "p").mkdir()
        self.conn = db.get_conn("p")

    def tearDown(self):
        self.conn.close()
        db.PROJECTS_ROOT = self._root

    def _parent(self, job_id, state):
        with self.conn:
            self.conn.execute("INSERT INTO JOBS(JOB_ID, STAGE, NAME, STATUS, CREATED_AT, STATE_JSON) VALUES (?, 'ab_initio_3d', 'j', 'running', 'c', ?)",
                              (job_id, json.dumps(state)))

    def _state(self, job_id):
        return json.loads(self.conn.execute("SELECT STATE_JSON FROM JOBS WHERE JOB_ID=?", (job_id,)).fetchone()[0])

    def test_progress_touches_only_its_two_counters(self):
        state = {"child_job_id": "c1", "starts": 2, "rounds": 10, "start": 0, "round": 3, "initial": False, "phase": "refine",
                 "pending_output_stars": ["a.star", "b.star"], "refinement_jobs_this_round": 2, "number_of_dump_files": 2}
        self._parent("p1", state)
        abinitio.child_progress(self.conn, "p1", "c1", 1, 2)
        after = self._state("p1")
        self.assertEqual((after["child_done"], after["child_task_count"]), (1, 2))
        for key in ("pending_output_stars", "refinement_jobs_this_round", "number_of_dump_files", "child_job_id"):
            self.assertEqual(after[key], state[key], key)
        self.assertGreater(self.conn.execute("SELECT PROGRESS FROM JOBS WHERE JOB_ID='p1'").fetchone()[0], 0)
        abinitio.child_progress(self.conn, "p1", "stale-child", 2, 2)   # not the running child: nothing changes
        self.assertEqual(self._state("p1")["child_done"], 1)

    def test_classification_progress_the_same_way(self):
        state = {"child_job_id": "c1", "rounds": 4, "round": 1, "phase": "refine", "start_with_random": True, "history": [{"round": 1}], "child_task_count": 3}
        self._parent("p2", state)
        classification.child_progress(self.conn, "p2", "c1", 2, None)   # no task count given: the recorded one stays
        after = self._state("p2")
        self.assertEqual((after["child_done"], after["child_task_count"], after["history"]), (2, 3, [{"round": 1}]))


class MergeGuardTests(unittest.TestCase):
    def test_a_missing_count_is_refused_not_guessed(self):
        with self.assertRaises(ValueError):
            abinitio._required_count({"refinement_jobs_this_round": None}, "refinement_jobs_this_round")
        self.assertEqual(abinitio._required_count({"number_of_dump_files": 8}, "number_of_dump_files"), 8)

    def test_a_round_that_did_not_return_every_particle_fails(self):
        scratch = tempfile.mkdtemp()
        rows = [{"position_in_stack": i, "psi": 0.0, "theta": 0.0, "phi": 0.0, "x_shift": 0.0, "y_shift": 0.0, "occupancy": 100.0, "logp": 0.0, "sigma": 1.0,
                 "score": 0.0, "image_is_active": 1, "pixel_size": 1.0} for i in (1, 2, 3)]
        state = {"scratch": scratch, "number_of_classes": 1, "refinement_jobs_this_round": 1}
        abinitio._store_rows(state, "input", [rows])
        out = os.path.join(scratch, "out.star")
        starfile.write_star(out, rows[:2])   # the task wrote two of the three
        state["pending_output_stars"] = [out]
        with self.assertRaises(ValueError) as ctx:
            abinitio._merge_output_stars(state)
        self.assertIn("returned 2 of 3 particles", str(ctx.exception))
        starfile.write_star(out, rows)       # all three: fine
        self.assertEqual(len(abinitio._merge_output_stars(state)[0]), 3)


if __name__ == "__main__":
    unittest.main()
