"""optics_groups.py: the EPU XML / mdoc sidecars, the four groupings, and the grouping written to the images."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db  # noqa: E402
import optics_groups as og  # noqa: E402

EPU_XML = """<?xml version="1.0" encoding="utf-8"?>
<MicroscopeImage xmlns:i="http://www.w3.org/2001/XMLSchema-instance" xmlns="http://schemas.datacontract.org/2004/07/Fei.SharedObjects">
  <microscopeData>
    <acquisition><acquisitionDateTime>2024-03-05T14:23:11.1234567+01:00</acquisitionDateTime></acquisition>
    <optics>
      <BeamShift xmlns:a="http://schemas.datacontract.org/2004/07/Fei.Types"><a:_x>{x}</a:_x><a:_y>{y}</a:_y></BeamShift>
    </optics>
  </microscopeData>
</MicroscopeImage>
"""

MDOC = """PixelSpacing = 0.83
[ZValue = 0]
TiltAngle = 0
StagePosition = 123.4 -56.7
ImageShift = {x} {y}
DateTime = 05-Mar-24  14:23:11
"""


class SidecarTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_epu_xml_beside_a_fractions_movie(self):
        Path(self.tmp, "FoilHole_1_Data_2_3_20240305_142311.xml").write_text(EPU_XML.format(x=0.0123, y=-0.045))
        movie = os.path.join(self.tmp, "FoilHole_1_Data_2_3_20240305_142311_fractions.tiff")
        Path(movie).write_bytes(b"x")
        meta = og.read_sidecar(movie)
        self.assertAlmostEqual(meta["beam_shift_x"], 0.0123)
        self.assertAlmostEqual(meta["beam_shift_y"], -0.045)
        self.assertEqual(meta["acquisition_time"], 1709644991)   # 14:23:11 at +01:00 is 13:23:11 UTC
        self.assertEqual(og.metadata_for_movie(movie)["acquisition_time"], meta["acquisition_time"])

    def test_the_epu_time_is_utc(self):
        Path(self.tmp, "m.xml").write_text(EPU_XML.format(x=1, y=2))
        import datetime
        want = int(datetime.datetime(2024, 3, 5, 13, 23, 11, tzinfo=datetime.timezone.utc).timestamp())
        self.assertEqual(og.read_epu_xml(os.path.join(self.tmp, "m.xml"))["acquisition_time"], want)

    def test_mdoc_beside_a_movie(self):
        movie = os.path.join(self.tmp, "grid3_0007.tif")
        Path(movie).write_bytes(b"x")
        Path(movie + ".mdoc").write_text(MDOC.format(x=-0.25, y=1.5))
        meta = og.read_sidecar(movie)
        self.assertEqual((meta["beam_shift_x"], meta["beam_shift_y"]), (-0.25, 1.5))
        import datetime
        self.assertEqual(meta["acquisition_time"], int(datetime.datetime(2024, 3, 5, 14, 23, 11, tzinfo=datetime.timezone.utc).timestamp()))

    def test_no_sidecar_gives_the_mtime_only(self):
        movie = os.path.join(self.tmp, "plain.mrc")
        Path(movie).write_bytes(b"x")
        self.assertEqual(og.read_sidecar(movie), {})
        meta = og.metadata_for_movie(movie)
        self.assertIsNone(meta["beam_shift_x"])
        self.assertEqual(meta["acquisition_time"], int(os.stat(movie).st_mtime))

    def test_a_non_epu_xml_is_ignored(self):
        Path(self.tmp, "m.xml").write_text("<root><other>1</other></root>")
        self.assertEqual(og.read_epu_xml(os.path.join(self.tmp, "m.xml")), {})


class GroupingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._root = db.PROJECTS_ROOT
        db.PROJECTS_ROOT = Path(self.tmp) / "projects"
        self.project = "t-og"
        (db.PROJECTS_ROOT / self.project).mkdir(parents=True)
        self.conn = db.get_conn(self.project)
        rng = np.random.default_rng(1)
        # 40 images from 40 movies: four beam-shift positions (10 each), two days, EPU-style names
        centres = [(0.0, 0.0), (0.03, 0.0), (0.0, 0.03), (0.03, 0.03)]
        day0 = 1709600000
        with self.conn as c:
            for i in range(40):
                pos = i % 4
                bx, by = centres[pos][0] + rng.normal(0, 0.0005), centres[pos][1] + rng.normal(0, 0.0005)
                t = day0 + (0 if i < 20 else 86400 * 2) + i * 60
                name = "FoilHole_{}_Data_{}_{}_20240305_1423{:02d}".format(100 + i // 4, 1000 + pos, 2000 + pos, i % 60)
                c.execute("INSERT INTO MOVIE_ASSETS(MOVIE_ASSET_ID, NAME, FILENAME, BEAM_SHIFT_X, BEAM_SHIFT_Y, ACQUISITION_TIME) VALUES (?,?,?,?,?,?)",
                          (i + 1, name, "/data/" + name + "_fractions.tiff", bx, by, t))
                c.execute("INSERT INTO IMAGE_ASSETS(IMAGE_ASSET_ID, NAME, FILENAME, PARENT_MOVIE_ID, BEAM_SHIFT_X, BEAM_SHIFT_Y, ACQUISITION_TIME) VALUES (?,?,?,?,?,?,?)",
                          (i + 1, name + "_aligned", "/data/" + name + "_aligned.mrc", i + 1, bx, by, t))
                c.execute("INSERT INTO IMAGE_GROUP_MEMBERS(GROUP_ID, IMAGE_ASSET_ID) VALUES (0, ?)", (i + 1,))
            c.execute("INSERT INTO IMAGE_GROUP_LIST(GROUP_ID, GROUP_NAME, LIST_ID) VALUES (1, 'first ten', 0)")
            c.execute("INSERT INTO IMAGE_GROUP_LIST(GROUP_ID, GROUP_NAME, LIST_ID) VALUES (2, 'next five', 0)")
            for i in range(10):
                c.execute("INSERT INTO IMAGE_GROUP_MEMBERS(GROUP_ID, IMAGE_ASSET_ID) VALUES (1, ?)", (i + 1,))
            for i in range(10, 15):
                c.execute("INSERT INTO IMAGE_GROUP_MEMBERS(GROUP_ID, IMAGE_ASSET_ID) VALUES (2, ?)", (i + 1,))

    def tearDown(self):
        self.conn.close()
        db.PROJECTS_ROOT = self._root

    def test_from_image_groups(self):
        p = og.preview(self.conn, {"method": "image_groups", "group_ids": [1, 2]})
        self.assertEqual([(g["group"], g["name"], g["images"]) for g in p["groups"]], [(1, "first ten", 10), (2, "next five", 5)])
        self.assertEqual(p["unassigned"], 25)

    def test_from_the_filename_preset(self):
        p = og.preview(self.conn, {"method": "filename", "preset": "epu_position"})
        self.assertEqual(len(p["groups"]), 4)
        self.assertEqual(sorted(g["images"] for g in p["groups"]), [10, 10, 10, 10])
        self.assertEqual(p["groups"][0]["name"], "1000_2000")
        with self.assertRaises(ValueError):
            og.preview(self.conn, {"method": "filename", "pattern": "nocapture"})

    def test_by_day_and_by_gap(self):
        p = og.preview(self.conn, {"method": "time", "mode": "day"})
        self.assertEqual([(g["name"], g["images"]) for g in p["groups"]], [("2024-03-05", 20), ("2024-03-07", 20)])
        p = og.preview(self.conn, {"method": "time", "mode": "gap", "gap_hours": 1.0})
        self.assertEqual([g["images"] for g in p["groups"]], [20, 20])
        p = og.preview(self.conn, {"method": "time", "mode": "gap", "gap_hours": 100.0})
        self.assertEqual([g["images"] for g in p["groups"]], [40])

    def test_by_beam_shift_finds_the_four_positions(self):
        p = og.preview(self.conn, {"method": "beam_shift"})
        self.assertEqual(len(p["groups"]), 4)
        self.assertEqual([g["images"] for g in p["groups"]], [10, 10, 10, 10])
        self.assertEqual(len(p["points"]), 40)
        # the numbering follows the centres, so the group of each position is the same whichever way k-means started
        by_image = {int(k): v for k, v in p["assignment"].items()}
        self.assertEqual(len({by_image[i + 1] for i in range(0, 40, 4)}), 1)
        p3 = og.preview(self.conn, {"method": "beam_shift", "k": 2})
        self.assertEqual(len(p3["groups"]), 2)

    def test_apply_writes_images_movies_and_names_and_clear_undoes_it(self):
        r = og.apply(self.conn, {"method": "beam_shift"})
        self.assertEqual(len(r["groups"]), 4)
        cur = og.current(self.conn)
        self.assertEqual([g["images"] for g in cur["groups"]], [10, 10, 10, 10])
        self.assertEqual(cur["groups"][0]["source"], "beam_shift")
        self.assertEqual((cur["unassigned"], cur["image_count"], cur["images_with_beam_shift"]), (0, 40, 40))
        img = self.conn.execute("SELECT OPTICS_GROUP FROM IMAGE_ASSETS WHERE IMAGE_ASSET_ID=5").fetchone()[0]
        mov = self.conn.execute("SELECT OPTICS_GROUP FROM MOVIE_ASSETS WHERE MOVIE_ASSET_ID=5").fetchone()[0]
        self.assertEqual(img, mov)
        self.assertGreater(img, 0)
        # a second grouping over a scope, added rather than replacing: numbered after the existing four
        r2 = og.apply(self.conn, {"method": "image_groups", "group_ids": [2], "image_group_id": 2, "replace": False})
        self.assertEqual([g["group"] for g in r2["groups"]], [5])
        self.assertEqual(len(og.current(self.conn)["groups"]), 5)
        # and replacing starts over
        og.apply(self.conn, {"method": "time", "mode": "day"})
        self.assertEqual([g["name"] for g in og.current(self.conn)["groups"]], ["2024-03-05", "2024-03-07"])

    def test_read_metadata_fills_images_from_their_movies(self):
        with self.conn as c:
            c.execute("UPDATE IMAGE_ASSETS SET BEAM_SHIFT_X=NULL, BEAM_SHIFT_Y=NULL, ACQUISITION_TIME=NULL")
        r = og.read_metadata(self.conn)
        self.assertEqual(r["images_read"], 40)
        self.assertEqual(og.current(self.conn)["images_with_beam_shift"], 40)


class ImportTests(unittest.TestCase):
    """The movie import records the sidecar's beam shift and time."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._root = db.PROJECTS_ROOT
        db.PROJECTS_ROOT = Path(self.tmp) / "projects"
        self.project = "t-og-import"
        (db.PROJECTS_ROOT / self.project).mkdir(parents=True)
        db.get_conn(self.project).close()

    def tearDown(self):
        db.PROJECTS_ROOT = self._root

    def test_the_movie_import_reads_the_epu_xml(self):
        import cistem_server
        import refinement_packages as rp
        for i in range(2):
            w = rp.MrcStackWriter(os.path.join(self.tmp, "FoilHole_%d_Data_1_2_20240305_142311_fractions.mrc" % i), 16, 1.0)
            for f in range(4):
                w.append(np.full((16, 16), float(f), dtype=np.float32))
            w.close()
        Path(self.tmp, "FoilHole_0_Data_1_2_20240305_142311.xml").write_text(EPU_XML.format(x=0.5, y=-0.25))
        r = cistem_server._import_movies_work(self.project, {"input_glob": os.path.join(self.tmp, "FoilHole_*_fractions.mrc"), "voltage_kv": 300, "cs_mm": 2.7,
                                                             "pixel_size_a": 1.0, "dose_per_frame": 1.0})
        self.assertEqual(r["movie_count"], 2)
        rows = db.get_conn(self.project).execute("SELECT NAME, BEAM_SHIFT_X, BEAM_SHIFT_Y, ACQUISITION_TIME, OPTICS_GROUP FROM MOVIE_ASSETS ORDER BY MOVIE_ASSET_ID").fetchall()
        self.assertEqual((rows[0]["BEAM_SHIFT_X"], rows[0]["BEAM_SHIFT_Y"], rows[0]["OPTICS_GROUP"]), (0.5, -0.25, 0))
        self.assertIsNone(rows[1]["BEAM_SHIFT_X"])              # no sidecar: the mtime stands for the time
        self.assertIsNotNone(rows[1]["ACQUISITION_TIME"])


class KMeansTests(unittest.TestCase):
    def test_suggest_k(self):
        rng = np.random.default_rng(3)
        pts = np.concatenate([rng.normal(c, 0.01, (25, 2)) for c in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 2), (3, 1), (2, 0)]])
        self.assertEqual(og.suggest_k(pts), 7)
        self.assertEqual(og.suggest_k(np.zeros((10, 2))), 1)
        self.assertEqual(og.suggest_k(np.array([[0, 0], [1, 1]])), 1)


if __name__ == "__main__":
    unittest.main()
