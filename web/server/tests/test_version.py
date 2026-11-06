"""The version the home page shows: from git in a checkout, from the file the
install writes beside the server otherwise."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import cistem_server  # noqa: E402

WRITE_VERSION = HERE.parent.parent / "tools" / "write_version.py"


class VersionInfoTests(unittest.TestCase):
    def test_a_checkout_answers_from_git(self):
        def fake_git(args):
            if args[:2] == ["rev-parse", "--short"]: return "abc1234"
            if args[:2] == ["rev-parse", "--abbrev-ref"]: return "no-wx"
            if args[0] == "log": return "2026-11-06 12:00:00 -0600"
            return None   # a clean tree: `status --porcelain` prints nothing
        with mock.patch.object(cistem_server, "_git", side_effect=fake_git):
            info = cistem_server.version_info()
        self.assertEqual(info["commit"], "abc1234")
        self.assertEqual(info["branch"], "no-wx")
        self.assertEqual(info["source"], "git")

    def test_an_install_answers_from_the_file_the_install_wrote(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "version.json"
            path.write_text(json.dumps({"commit": "7c76956", "commit_datetime": "2026-11-06 12:50:00 -0600", "branch": "no-wx", "written_at": "x"}))
            with mock.patch.object(cistem_server, "_git", return_value=None), mock.patch.object(cistem_server, "VERSION_FILE", path):
                info = cistem_server.version_info()
        self.assertEqual(info["commit"], "7c76956")
        self.assertEqual(info["commit_datetime"], "2026-11-06 12:50:00 -0600")
        self.assertEqual(info["branch"], "no-wx")
        self.assertEqual(info["source"], "file")

    def test_neither_leaves_the_fields_empty_rather_than_failing(self):
        with mock.patch.object(cistem_server, "_git", return_value=None), mock.patch.object(cistem_server, "VERSION_FILE", pathlib.Path("/nonexistent/version.json")):
            info = cistem_server.version_info()
        self.assertIsNone(info["commit"])
        self.assertEqual(info["source"], "none")

    def test_the_route_serves_it(self):
        client = cistem_server.app.test_client()
        r = client.get("/api/version")
        self.assertEqual(r.status_code, 200)
        self.assertIn("commit", r.get_json())


class WriteVersionTests(unittest.TestCase):
    def test_writes_this_checkouts_version(self):
        if not (HERE.parent.parent.parent / ".git").exists():
            self.skipTest("not a git checkout")
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "deep" / "version.json"
            r = subprocess.run([sys.executable, str(WRITE_VERSION), str(out)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            info = json.loads(out.read_text())
        self.assertTrue(info["commit"])
        self.assertTrue(info["branch"])
        self.assertTrue(info["commit_datetime"])

    def test_a_directory_that_is_not_a_checkout_still_writes_and_exits_0(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "version.json"
            r = subprocess.run([sys.executable, str(WRITE_VERSION), "--repo", tmp, str(out)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            info = json.loads(out.read_text())
        self.assertIsNone(info["commit"])
        self.assertIn("not a git checkout", r.stderr)


if __name__ == "__main__":
    unittest.main()
