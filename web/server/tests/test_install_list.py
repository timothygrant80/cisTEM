"""Every file under web/ that the server or page needs must be in Makefile.am's
install list -- `make install` copies from that list, not the directory, and
a new module left off it is a ModuleNotFoundError on the installed copy."""
import os
import re
import unittest

WEB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(WEB)
SKIP_DIRS = {"tests", "__pycache__", "data", ".claude"}
SKIP_FILES = {"CLAUDE.md", ".gitignore"}


class InstallListTests(unittest.TestCase):
    def test_every_web_file_is_installed(self):
        listed = set(re.findall(r"web/[^\s\\]+", open(os.path.join(ROOT, "Makefile.am")).read()))
        missing = []
        for dirpath, dirnames, filenames in os.walk(WEB):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                if name in SKIP_FILES or name.endswith((".pyc", ".log", ".db", ".txt.tmp")):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, name), ROOT)
                if rel not in listed:
                    missing.append(rel)
        self.assertEqual(missing, [], "add to nobase_dist_pkgdata_DATA in Makefile.am: " + ", ".join(missing))


if __name__ == "__main__":
    unittest.main()
