"""stages/ctffind.py: the argument contract with ctffind's DoCalculation() and
the thickness columns finalize() writes."""
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db  # noqa: E402
from stages import ctffind  # noqa: E402

# CtffindApp::DoInteractiveUserInput()'s ManualSetArguments() format string (src/programs/ctffind/ctffind.cpp);
# 's' and 't' are both text.
CTFFIND_FORMAT = "tbitffffifffffbfbfffbffbbsbsbfffbfffbiiibbbbfffbb".replace("s", "t")
KIND_CODE = {"text": "t", "bool": "b", "int": "i", "float": "f"}


class CtffindStageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._root = db.PROJECTS_ROOT
        db.PROJECTS_ROOT = Path(self.tmp)
        self.project = "t-" + os.path.basename(self.tmp)[-6:]
        (db.PROJECTS_ROOT / self.project).mkdir(parents=True)
        self.conn = db.get_conn(self.project)
        with self.conn:
            self.conn.execute("INSERT INTO IMAGE_ASSETS(IMAGE_ASSET_ID, NAME, FILENAME, PARENT_MOVIE_ID, X_SIZE, Y_SIZE, PIXEL_SIZE, VOLTAGE, "
                              "SPHERICAL_ABERRATION, PROTEIN_IS_WHITE) VALUES (7, 'img', '/none/img.mrc', -1, 100, 100, 1.2, 300.0, 2.7, 0)")
            self.conn.execute("INSERT INTO IMAGE_GROUP_LIST(GROUP_ID, GROUP_NAME, LIST_ID) VALUES (5, 'g', 0)")
            self.conn.execute("INSERT INTO IMAGE_GROUP_MEMBERS(GROUP_ID, IMAGE_ASSET_ID) VALUES (5, 7)")
            self.conn.execute("INSERT INTO JOBS(JOB_ID, JOB_NUMBER, STAGE, STATUS, CREATED_AT) VALUES ('job1', 1, 'ctf_estimation', 'completed', 0)")

    def tearDown(self):
        self.conn.close()
        db.PROJECTS_ROOT = self._root

    def test_argument_list_matches_ctffind(self):
        tasks = ctffind.build_tasks(self.conn, self.project, {"image_group_id": 5})
        self.assertEqual(len(tasks), 1)
        args = tasks[0]["args"]
        self.assertEqual("".join(KIND_CODE[a["type"]] for a in args), CTFFIND_FORMAT)
        values = [a["value"] for a in args]
        # Thickness fitting off by default, with the program's own defaults for its options.
        self.assertEqual(values[41:49], [False, True, True, 30.0, 3.0, 1.4, False, False])
        self.assertIs(values[23], True)  # resample if the pixel is too small

    def test_thickness_options_and_result_are_stored(self):
        params = {"image_group_id": 5, "fit_nodes": True, "fit_nodes_1d": False, "fit_nodes_low_res_a": 25, "fit_nodes_high_res_a": 4,
                  "fit_nodes_downweight": True, "resample_if_pixel_too_small": False, "target_pixel_size_a": 1.6, "search_tilt": True}
        tasks = ctffind.build_tasks(self.conn, self.project, params)
        values = [a["value"] for a in tasks[0]["args"]]
        self.assertEqual(values[41:49], [True, False, True, 25.0, 4.0, 1.6, False, True])
        self.assertIs(values[23], False)
        self.assertIs(values[36], True)
        # ctffind's eleven floats: defocus 1, 2, angle, phase shift, score, fit res, alias res, iciness, tilt angle, tilt axis, thickness.
        data = [20015.0, 19800.0, 45.0, 0.0, 0.05, 4.2, 0.0, 0.14, 1.0, 2.0, 812.5]
        row = {"TASK_INDEX": 0, "STATUS": "ok", "REF": 7, "RESULT_JSON": json.dumps({"kind": "floats", "data": data})}
        summary = ctffind.finalize(self.conn, self.project, {"id": "job1"}, tasks, [row], lambda *a, **k: None)
        self.assertEqual(summary["ctf_estimates_written"], 1)
        r = self.conn.execute("SELECT * FROM ESTIMATED_CTF_PARAMETERS").fetchone()
        self.assertEqual(r["SAMPLE_THICKNESS"], 812.5)
        self.assertEqual((r["FIT_NODES"], r["FIT_NODES_1D"], r["FIT_NODES_2D"]), (1, 0, 1))
        self.assertEqual((r["FIT_NODES_LOW_LIMIT"], r["FIT_NODES_HIGH_LIMIT"]), (25.0, 4.0))
        self.assertEqual((r["FIT_NODES_ROUNDED_SQUARE"], r["FIT_NODES_DOWNWEIGHT_NODES"]), (0, 1))
        self.assertEqual((r["RESAMPLE_IF_NESCESSARY"], r["TARGET_PIXEL_SIZE"], r["DETERMINE_TILT"]), (0, 1.6, 1))
        self.assertEqual((r["TILT_ANGLE"], r["TILT_AXIS"], r["ICINESS"]), (1.0, 2.0, 0.14))
        self.assertEqual(self.conn.execute("SELECT CTF_ESTIMATION_ID FROM IMAGE_ASSETS WHERE IMAGE_ASSET_ID=7").fetchone()[0], r["CTF_ESTIMATION_ID"])
        live = ctffind.live_result(self.conn, tasks[0], row)
        self.assertEqual((live["sample_thickness"], live["fit_nodes"]), (812.5, True))

    def test_a_nan_result_is_stored_as_zero_and_logged_instead_of_failing_the_job(self):
        # The controller writes a NaN or infinite float as JSON null; float(None) used to
        # raise inside finalize()'s transaction and lose every image's result.
        tasks = ctffind.build_tasks(self.conn, self.project, {"image_group_id": 5})
        data = [20015.0, 19800.0, 45.0, 0.0, 0.05, 4.2, 0.0, 0.14, 0.0, 0.0, None]
        row = {"TASK_INDEX": 0, "STATUS": "ok", "REF": 7, "RESULT_JSON": json.dumps({"kind": "floats", "data": data})}
        logged = []
        summary = ctffind.finalize(self.conn, self.project, {"id": "job1"}, tasks, [row], lambda msg, **k: logged.append((msg, k.get("level"))))
        self.assertEqual(summary["ctf_estimates_written"], 1)
        r = self.conn.execute("SELECT DEFOCUS1, SAMPLE_THICKNESS FROM ESTIMATED_CTF_PARAMETERS").fetchone()
        self.assertEqual((r["DEFOCUS1"], r["SAMPLE_THICKNESS"]), (20015.0, 0.0))
        self.assertEqual(len(logged), 1)
        self.assertIn("sample thickness", logged[0][0])
        self.assertEqual(logged[0][1], "error")
        self.assertEqual(ctffind.live_result(self.conn, tasks[0], row)["sample_thickness"], 0.0)

    def test_program_format_string_in_source(self):
        # Guard against the two drifting apart silently: the format string above must be the one in ctffind.cpp.
        src = Path(__file__).resolve().parents[3] / "src" / "programs" / "ctffind" / "ctffind.cpp"
        if not src.is_file():
            self.skipTest("cisTEM sources not alongside")
        m = re.search(r'ManualSetArguments\("([a-z]+)"', src.read_text())
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1).replace("s", "t"), CTFFIND_FORMAT)


if __name__ == "__main__":
    unittest.main()
