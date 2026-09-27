"""Package creation tasks outlive the server's memory: a task's row in the
project's PACKAGE_TASKS table lets a restart mid-creation be reported as a
failure with a reason (rather than a task id nobody knows), the page reads
failed ones from the list and acknowledges them, and a finished task's row
goes when it finishes."""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import auth  # noqa: E402
import db  # noqa: E402
import cistem_server  # noqa: E402


class PackageTaskTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._root, self._auth = db.PROJECTS_ROOT, auth.AUTH_DB_PATH
        db.PROJECTS_ROOT = Path(self.tmp) / "projects"; auth.AUTH_DB_PATH = Path(self.tmp) / "auth.db"
        self.project = "t-tasks"
        (db.PROJECTS_ROOT / self.project).mkdir(parents=True)
        db.get_conn(self.project).close()
        token = auth.create_session(auth.create_user("boss", "password123", "admin")["id"])
        self.client = cistem_server.app.test_client()
        self.headers = {"Authorization": "Bearer " + token}

    def tearDown(self):
        db.PROJECTS_ROOT, auth.AUTH_DB_PATH = self._root, self._auth

    def _rows(self):
        conn = db.get_conn(self.project)
        try:
            return [dict(r) for r in conn.execute("SELECT * FROM PACKAGE_TASKS ORDER BY STARTED_AT").fetchall()]
        finally:
            conn.close()

    def _wait_terminal(self, task_id):
        for _ in range(100):
            with cistem_server._package_tasks_lock:
                state = cistem_server._package_tasks[task_id]["state"]
            if state != "running":
                return state
            time.sleep(0.05)
        self.fail("task never finished")

    def test_a_failed_creation_stays_until_acknowledged(self):
        r = self.client.post("/api/projects/{}/refinement-packages?async=1".format(self.project), json={"name": "Bad"}, headers=self.headers)
        self.assertEqual(r.status_code, 202, r.data)
        task_id = r.get_json()["task_id"]
        self.assertEqual(self._wait_terminal(task_id), "failed")   # no group, no selection: a ValueError
        time.sleep(0.1)
        rows = self._rows()
        self.assertEqual([(x["TASK_ID"], x["STATE"], x["NAME"]) for x in rows], [(task_id, "failed", "Bad")])
        # gone from memory (as after a restart): the list and the single GET still know it
        with cistem_server._package_tasks_lock:
            cistem_server._package_tasks.pop(task_id)
        listed = self.client.get("/api/projects/{}/refinement-packages/tasks".format(self.project), headers=self.headers).get_json()["tasks"]
        self.assertEqual([(t["task_id"], t["state"], t["from_before_restart"]) for t in listed], [(task_id, "failed", True)])
        self.assertIn("required", listed[0]["error"])
        one = self.client.get("/api/projects/{}/refinement-packages/tasks/{}".format(self.project, task_id), headers=self.headers)
        self.assertEqual((one.status_code, one.get_json()["state"]), (200, "failed"))
        ack = self.client.post("/api/projects/{}/refinement-packages/tasks/{}/ack".format(self.project, task_id), headers=self.headers)
        self.assertEqual(ack.status_code, 200)
        self.assertEqual(self._rows(), [])
        self.assertEqual(self.client.get("/api/projects/{}/refinement-packages/tasks/{}".format(self.project, task_id), headers=self.headers).status_code, 404)

    def test_a_restart_fails_a_running_task_with_the_reason(self):
        conn = db.get_conn(self.project)
        with conn:
            cistem_server._package_task_row_write(conn, "abc123", {"state": "running", "done": 40, "total": 100, "message": "Cutting", "name": "Pkg", "started_at": time.time() - 30})
        conn.close()
        cistem_server._recover_package_tasks()
        rows = self._rows()
        self.assertEqual((rows[0]["STATE"], rows[0]["ERROR"]), ("failed", cistem_server.RESTART_ERROR))
        listed = self.client.get("/api/projects/{}/refinement-packages/tasks".format(self.project), headers=self.headers).get_json()["tasks"]
        self.assertEqual((listed[0]["task_id"], listed[0]["name"], listed[0]["done"]), ("abc123", "Pkg", 40))
        self.assertIn("restarted", listed[0]["error"])


if __name__ == "__main__":
    unittest.main()
