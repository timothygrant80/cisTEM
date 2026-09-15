"""db-level tests for run profile editing, on a scratch system database.

    python -m unittest discover -s server/tests
"""

import os
import pathlib
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import db  # noqa: E402


class RunProfileEditingTests(unittest.TestCase):
    def setUp(self):
        self._root = tempfile.mkdtemp(prefix="rp-test-")
        self._saved_path = db.SYSTEM_DB_PATH
        db.SYSTEM_DB_PATH = pathlib.Path(self._root) / "system.db"
        self.conn = db.get_system_conn()

    def tearDown(self):
        self.conn.close()
        db.SYSTEM_DB_PATH = self._saved_path
        shutil.rmtree(self._root, ignore_errors=True)

    def names(self):
        return [p["name"] for p in db.load_run_profiles(self.conn)]

    def test_seed_shape(self):
        profiles = db.load_run_profiles(self.conn)
        self.assertEqual(self.names(), ["Local (single-threaded)", "Local (multi-threaded)", "Cluster (Slurm)"])
        single = profiles[0]
        self.assertEqual(single["run_commands"][0]["copies"], db._CORES + 1)  # master + `cores` workers
        self.assertEqual(single["total_jobs"], db._CORES + 1)
        self.assertEqual(profiles[2]["total_jobs"], 0)

    def test_add_default_local_and_uniqueness(self):
        a = db.create_run_profile(self.conn, db.default_local_profile_spec())
        b = db.create_run_profile(self.conn, db.default_local_profile_spec())
        got = {p["run_profile_id"]: p["name"] for p in db.load_run_profiles(self.conn)}
        self.assertEqual(got[a], "Default Local")
        self.assertEqual(got[b], "Default Local (2)")
        prof = db.load_run_profile(self.conn, a)
        self.assertEqual(prof["manager_command"], "$command")
        self.assertEqual(len(prof["run_commands"]), 1)
        self.assertEqual(prof["run_commands"][0]["delay_ms"], 10)

    def test_duplicate_shape(self):
        src = db.load_run_profiles(self.conn)[0]
        new = db.create_run_profile(self.conn, dict(src, name="Copy of " + src["name"]))
        copy = db.load_run_profile(self.conn, new)
        self.assertEqual(copy["name"], "Copy of Local (single-threaded)")
        self.assertEqual(copy["run_commands"], src["run_commands"])
        self.assertNotEqual(copy["run_profile_id"], src["run_profile_id"])

    def test_update_fields_and_commands(self):
        pid = db.load_run_profiles(self.conn)[2]["run_profile_id"]  # the empty Slurm one
        db.update_run_profile(self.conn, pid, {
            "manager_command": "ssh head-node $command",
            "controller_address": "10.0.0.5",
            "gui_address": "192.168.1.9",
            "run_commands": [
                {"command": "sbatch --wrap=\"$command\"", "copies": 8, "threads_per_copy": 2,
                 "override_total_copies": True, "overridden_total_copies": 16, "delay_ms": 250},
                {"command": "$command", "copies": 1},
            ],
        })
        p = db.load_run_profile(self.conn, pid)
        self.assertEqual(p["manager_command"], "ssh head-node $command")
        self.assertEqual(p["controller_address"], "10.0.0.5")
        self.assertEqual(p["gui_address"], "192.168.1.9")
        self.assertEqual(len(p["run_commands"]), 2)
        self.assertEqual(p["run_commands"][0]["overridden_total_copies"], 16)
        self.assertEqual(p["run_commands"][1]["threads_per_copy"], 1)  # default filled in
        self.assertEqual(p["total_jobs"], 16 + 1)  # override counts, plain copies count

    def test_rename_rules(self):
        ids = [p["run_profile_id"] for p in db.load_run_profiles(self.conn)]
        db.update_run_profile(self.conn, ids[0], {"name": "Workstation"})
        self.assertEqual(self.names()[0], "Workstation")
        with self.assertRaises(db.RunProfileError):
            db.update_run_profile(self.conn, ids[1], {"name": "workstation"})  # case-insensitive clash
        with self.assertRaises(db.RunProfileError):
            db.update_run_profile(self.conn, ids[1], {"name": "   "})
        db.update_run_profile(self.conn, ids[0], {"name": "Workstation"})  # renaming to itself is fine

    def test_command_validation(self):
        pid = db.load_run_profiles(self.conn)[0]["run_profile_id"]
        with self.assertRaises(db.RunProfileError):
            db.update_run_profile(self.conn, pid, {"manager_command": "ssh node unblur"})
        with self.assertRaises(db.RunProfileError):
            db.update_run_profile(self.conn, pid, {"run_commands": [{"command": "run it", "copies": 1}]})
        with self.assertRaises(db.RunProfileError):
            db.update_run_profile(self.conn, pid, {"run_commands": [{"command": "$command", "copies": 0}]})
        with self.assertRaises(db.RunProfileError):
            db.update_run_profile(self.conn, pid, {"run_commands": [{"command": "$command", "copies": "many"}]})
        # nothing above should have changed the profile
        self.assertEqual(db.load_run_profile(self.conn, pid)["manager_command"], "$command")

    def test_delete_drops_table_and_unknown_ids(self):
        pid = db.load_run_profiles(self.conn)[1]["run_profile_id"]
        self.assertTrue(db.delete_run_profile(self.conn, pid))
        self.assertNotIn("Local (multi-threaded)", self.names())
        tables = [r[0] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE name='RUN_PROFILE_COMMANDS_{}'".format(pid))]
        self.assertEqual(tables, [])
        self.assertFalse(db.delete_run_profile(self.conn, pid))
        with self.assertRaises(KeyError):
            db.update_run_profile(self.conn, pid, {"name": "x"})
        # a reopen must not resurrect it
        self.conn.close()
        self.conn = db.get_system_conn()
        self.assertEqual(len(self.names()), 2)


if __name__ == "__main__":
    unittest.main()
