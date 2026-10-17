"""The movie and image imports read every file's header before opening their
transaction, so the project's write lock is not held while files on a slow
filesystem are read (minutes for TIFF movies on NFS; every other writer then
failed with "database is locked")."""
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import auth  # noqa: E402
import cistem_server  # noqa: E402
import db  # noqa: E402
import imageheaders  # noqa: E402
import refinement_packages as rp  # noqa: E402


class ImportLockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._root, self._auth = db.PROJECTS_ROOT, auth.AUTH_DB_PATH
        db.PROJECTS_ROOT = Path(self.tmp) / "projects"; auth.AUTH_DB_PATH = Path(self.tmp) / "auth.db"
        self.project = "t-lock"
        (db.PROJECTS_ROOT / self.project).mkdir(parents=True)
        token = auth.create_session(auth.create_user("boss", "password123", "admin")["id"])
        self.client = cistem_server.app.test_client()
        self.headers = {"Authorization": "Bearer " + token}
        db.get_conn(self.project).close()
        for i in range(2):
            w = rp.MrcStackWriter(os.path.join(self.tmp, "mov_%d.mrc" % i), 16, 1.0)
            for f in range(4):
                w.append(np.full((16, 16), float(f), dtype=np.float32))
            w.close()
        self.other_writes = []

    def tearDown(self):
        db.PROJECTS_ROOT, auth.AUTH_DB_PATH = self._root, self._auth

    def _write_from_elsewhere(self):
        # What a second request would do meanwhile: a write on its own connection, waiting at most 0.3 s.
        other = sqlite3.connect(str(db.project_db_path(self.project)))
        other.execute("PRAGMA busy_timeout = 300")
        try:
            with other:
                other.execute("UPDATE MOVIE_IMPORT_DEFAULTS SET VOLTAGE=200 WHERE NUMBER=1")
            self.other_writes.append("ok")
        except sqlite3.OperationalError as exc:
            self.other_writes.append(str(exc))
        finally:
            other.close()

    def test_movie_headers_are_read_outside_the_transaction(self):
        real = imageheaders.read_movie_header

        def slow_header(path, **kw):
            self._write_from_elsewhere()
            return real(path, **kw)
        imageheaders.read_movie_header = slow_header
        try:
            r = self.client.post("/api/projects/%s/movies/import" % self.project, headers=self.headers,
                                 json={"input_glob": os.path.join(self.tmp, "mov_*.mrc"), "voltage_kv": 300, "cs_mm": 2.7, "pixel_size_a": 1.0, "dose_per_frame": 1.0})
        finally:
            imageheaders.read_movie_header = real
        self.assertEqual(r.status_code, 201, r.get_data(as_text=True))
        self.assertEqual(r.get_json()["movie_count"], 2)
        self.assertEqual(self.other_writes, ["ok", "ok"])

    def test_image_headers_are_read_outside_the_transaction(self):
        real = imageheaders.read_image_header

        def slow_header(path):
            self._write_from_elsewhere()
            return real(path)
        imageheaders.read_image_header = slow_header
        try:
            r = self.client.post("/api/projects/%s/images/import" % self.project, headers=self.headers,
                                 json={"input_glob": os.path.join(self.tmp, "mov_*.mrc"), "voltage_kv": 300, "cs_mm": 2.7, "pixel_size_a": 1.0})
        finally:
            imageheaders.read_image_header = real
        self.assertEqual(r.status_code, 201, r.get_data(as_text=True))
        self.assertEqual(r.get_json()["image_count"], 2)
        self.assertEqual(self.other_writes, ["ok", "ok"])


if __name__ == "__main__":
    unittest.main()
