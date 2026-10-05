"""GET /browse: the Browse picker's listing, and a typed path that names a file."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import auth  # noqa: E402
import db  # noqa: E402


class BrowseTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self._old = auth.AUTH_DB_PATH, db.PROJECTS_ROOT
        auth.AUTH_DB_PATH = root / "auth.db"
        db.PROJECTS_ROOT = root / "projects"
        db.PROJECTS_ROOT.mkdir()
        import cistem_server  # noqa: E402
        self.client = cistem_server.app.test_client()
        user = auth.create_user("alice", "password123", "user")
        self.headers = {"Authorization": "Bearer " + auth.create_session(user["id"])}
        self.data = root / "data"
        (self.data / "sub").mkdir(parents=True)
        for name in ("a.mrc", "b.tif", "notes.txt", ".hidden.mrc"):
            (self.data / name).write_bytes(b"x")

    def tearDown(self):
        auth.AUTH_DB_PATH, db.PROJECTS_ROOT = self._old
        self._tmp.cleanup()

    def _browse(self, **args):
        return self.client.get("/api/browse", query_string=args, headers=self.headers)

    def test_lists_a_folder_by_type(self):
        d = self._browse(path=str(self.data), types="movie").get_json()
        self.assertEqual(d["path"], str(self.data.resolve()))
        self.assertEqual(d["directories"], ["sub"])
        self.assertEqual(d["files"], ["a.mrc", "b.tif"])
        self.assertNotIn("file", d)
        self.assertEqual(d["parent"], str(self.data.resolve().parent))

    def test_a_typed_file_path_names_the_file(self):
        d = self._browse(path=str(self.data / "a.mrc"), types="movie").get_json()
        self.assertEqual(d["path"], str(self.data.resolve()))
        self.assertEqual(d["file"], "a.mrc")
        self.assertEqual(d["files"], ["a.mrc", "b.tif"])

    def test_a_missing_path_is_refused(self):
        r = self._browse(path=str(self.data / "nowhere"))
        self.assertEqual(r.status_code, 400)
        self.assertIn("not a directory", r.get_json()["error"])

    @unittest.skipIf(os.geteuid() == 0, "root can read anything")
    def test_a_file_inside_an_unreadable_folder_can_still_be_chosen(self):
        locked = self.data / "locked"
        locked.mkdir()
        (locked / "c.mrc").write_bytes(b"x")
        locked.chmod(0o311)   # traversable, not listable
        try:
            r = self._browse(path=str(locked), types="movie")
            self.assertEqual(r.status_code, 400)
            d = self._browse(path=str(locked / "c.mrc"), types="movie").get_json()
            self.assertEqual(d["file"], "c.mrc")
            self.assertEqual(d["path"], str(locked.resolve()))
            self.assertEqual(d["files"], [])
        finally:
            locked.chmod(0o755)


if __name__ == "__main__":
    unittest.main()
