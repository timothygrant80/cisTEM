"""User management through the API: DELETE /users/:id and its rules.

Runs the Flask app's test client against a throwaway auth database and a
throwaway projects directory, so nothing in server/data is touched.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import auth  # noqa: E402
import db  # noqa: E402


class DeleteUserTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self._old_auth_path, self._old_projects_root = auth.AUTH_DB_PATH, db.PROJECTS_ROOT
        auth.AUTH_DB_PATH = root / "auth.db"
        db.PROJECTS_ROOT = root / "projects"
        db.PROJECTS_ROOT.mkdir()
        import cistem_server  # noqa: E402  (imports Flask app; no server started)
        self.app = cistem_server.app
        self.client = self.app.test_client()
        self.admin = auth.create_user("boss", "password123", "admin")
        self.other_admin = auth.create_user("deputy", "password123", "admin")
        self.user = auth.create_user("alice", "password123", "user")
        self.admin_token = auth.create_session(self.admin["id"])
        self.user_token = auth.create_session(self.user["id"])

    def tearDown(self):
        auth.AUTH_DB_PATH, db.PROJECTS_ROOT = self._old_auth_path, self._old_projects_root
        self._tmp.cleanup()

    def _delete(self, user_id, token=None, body=None):
        return self.client.delete("/api/users/{}".format(user_id), json=body or {},
                                  headers={"Authorization": "Bearer " + (token or self.admin_token)})

    def test_admin_only(self):
        r = self._delete(self.other_admin["id"], token=self.user_token)
        self.assertEqual(r.status_code, 403)

    def test_cannot_delete_self(self):
        r = self._delete(self.admin["id"])
        self.assertEqual(r.status_code, 400)
        self.assertIn("own account", r.get_json()["error"])
        self.assertIsNotNone(auth.get_user_by_id(self.admin["id"]))
        # Another admin can be deleted while one remains.
        self.assertEqual(self._delete(self.other_admin["id"]).status_code, 200)
        self.assertEqual(auth.admin_count(), 1)

    def test_delete_revokes_sessions(self):
        r = self._delete(self.user["id"])
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["deleted"], "alice")
        self.assertIsNone(auth.get_user_by_id(self.user["id"]))
        self.assertIsNone(auth.validate_token(self.user_token))

    def test_owned_projects_must_be_transferred(self):
        p = db.create_project("Alice's grid", self.user["id"], "alice")
        r = self._delete(self.user["id"])
        self.assertEqual(r.status_code, 409)
        body = r.get_json()
        self.assertEqual(body["owned_project_count"], 1)
        self.assertEqual(body["projects"][0]["name"], "Alice's grid")
        self.assertIsNotNone(auth.get_user_by_id(self.user["id"]))  # nothing happened
        # transfer_to must be a different, existing user
        self.assertEqual(self._delete(self.user["id"], body={"transfer_to": self.user["id"]}).status_code, 400)
        self.assertEqual(self._delete(self.user["id"], body={"transfer_to": 9999}).status_code, 400)
        r = self._delete(self.user["id"], body={"transfer_to": self.other_admin["id"]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["transferred"], 1)
        self.assertEqual(r.get_json()["transferred_to"], "deputy")
        summary = db.get_project_summary(p if isinstance(p, str) else p["id"])
        self.assertEqual((summary["owner_user_id"], summary["owner_username"]), (self.other_admin["id"], "deputy"))
        self.assertIsNone(auth.get_user_by_id(self.user["id"]))


if __name__ == "__main__":
    unittest.main()
