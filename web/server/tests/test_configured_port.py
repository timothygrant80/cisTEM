"""configured_port(): CISTEM_PORT, else the `port: N` line of config.js
(comments ignored), else 8000."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cistem_server  # noqa: E402


class ConfiguredPortTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._root = cistem_server.REPO_ROOT
        cistem_server.REPO_ROOT = self.tmp
        self._env = os.environ.pop("CISTEM_PORT", None)

    def tearDown(self):
        cistem_server.REPO_ROOT = self._root
        if self._env is not None:
            os.environ["CISTEM_PORT"] = self._env

    def _config(self, text):
        (self.tmp / "config.js").write_text(text)

    def test_shipped_config_js_says_8000(self):
        cistem_server.REPO_ROOT = self._root
        self.assertEqual(cistem_server.configured_port(), 8000)

    def test_port_line_is_read_and_comments_are_ignored(self):
        self._config("// port: 1111 in a comment\nwindow.CRYOEM_CONFIG = {\n  port: 9123,\n  // apiBase: \"http://x:2222/api\"\n};\n")
        self.assertEqual(cistem_server.configured_port(), 9123)

    def test_missing_file_or_line_falls_back_to_8000(self):
        self.assertEqual(cistem_server.configured_port(), 8000)
        self._config("window.CRYOEM_CONFIG = {};\n")
        self.assertEqual(cistem_server.configured_port(), 8000)

    def test_environment_wins(self):
        self._config("window.CRYOEM_CONFIG = { port: 9123 };\n")
        os.environ["CISTEM_PORT"] = "7777"
        self.assertEqual(cistem_server.configured_port(), 7777)


if __name__ == "__main__":
    unittest.main()
