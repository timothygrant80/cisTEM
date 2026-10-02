"""blush.py: the numpy helpers against RELION's formulas, the pipeline with an
identity network (what comes out must be the masked, trailed input), and the
driver's settings. The network itself is exercised only when torch and the
weights are present (test_real_model_runs), since neither is a test dependency."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import blush  # noqa: E402

try:
    import torch  # noqa: E402
except ImportError:  # pragma: no cover - torch is optional
    torch = None


class HelperTests(unittest.TestCase):
    def test_block_starts_cover_the_span_and_end_at_the_edge(self):
        self.assertEqual(blush.block_starts(128, 64, 20), [0, 20, 40, 60, 64])
        self.assertEqual(blush.block_starts(64, 64, 20), [0])
        self.assertEqual(blush.block_starts(84, 64, 20), [0, 20])

    def test_weight_box_is_symmetric_positive_and_zero_margin_clipped(self):
        w = blush.make_weight_box(64, 10)
        self.assertEqual(w.shape, (64, 64, 64))
        self.assertTrue((w > 0).all())
        self.assertAlmostEqual(float(w[32, 32, 32]), 1.0, places=2)       # an even grid has no exact centre voxel
        self.assertAlmostEqual(float(w[0, 32, 32]), 1e-6, places=9)      # the margin
        self.assertTrue(np.allclose(w, w[::-1], atol=1e-6))

    def test_radial_mask_edges(self):
        m = blush.radial_mask(32, 10.0, edge_width=4.0)
        self.assertAlmostEqual(float(m[16, 16, 16]), 1.0)
        self.assertEqual(float(m[16, 16, 31]), 0.0)
        self.assertTrue(0.0 < float(m[16, 16, 16 + 10]) < 1.0)
        hard = blush.radial_mask(32, 10.0)
        self.assertEqual(set(np.unique(hard).tolist()), {0.0, 1.0})

    def test_fft_roundtrip_and_resampling_scale(self):
        rng = np.random.default_rng(0)
        v = rng.standard_normal((32, 32, 32)).astype(np.float32)
        self.assertTrue(np.allclose(blush.ifft(blush.fft(v)), v, atol=1e-5))
        up, voxel = blush.resample_fourier(v, 2.0, 1.5)
        self.assertEqual(up.shape, (42, 42, 42))               # 32 * 2.0 / 1.5 = 42.7 -> the even size nearest 1.5 A
        self.assertAlmostEqual(voxel, 2.0 * 32 / 42)
        # A smooth volume keeps its values through the resampling (density scaling, not total mass).
        z, y, x = np.indices((32, 32, 32)) - 16
        blob = np.exp(-((x * x + y * y + z * z) / 40.0)).astype(np.float32)
        up, _ = blush.resample_fourier(blob, 2.0, 1.5)
        self.assertAlmostEqual(float(up.max()), float(blob.max()), places=2)
        back, _ = blush.resample_fourier(up, 2.0 * 32 / 42, 2.0)
        self.assertEqual(back.shape, (32, 32, 32))
        self.assertTrue(np.allclose(back, blob, atol=1e-2))

    def test_crossover_grid_is_one_below_and_zero_above(self):
        g = blush.crossover_grid(10, 64, 3)
        shells = blush.fourier_shells(g.shape)
        self.assertTrue((g[shells < 9] == 1.0).all())
        self.assertTrue((g[shells > 11] == 0.0).all())

    def test_fsc_crossing_index_matches_relion(self):
        fsc = np.ones(33); fsc[20:] = 0.1
        self.assertEqual(blush.fsc_crossing_index(fsc), 18)        # RELION: argmax(fsc < t) - 1, then - 1
        self.assertEqual(blush.fsc_crossing_index(np.ones(33)), 31)   # never crosses: RELION's last shell, less one
        rows = [{"shell": i, "part_fsc": 1.0 if i < 12 else 0.05} for i in range(1, 33)]
        by_shell = blush.fsc_by_shell(rows, 64)
        self.assertEqual(by_shell.size, 33)
        self.assertEqual(float(by_shell[0]), 1.0)
        self.assertEqual(blush.fsc_crossing_index(by_shell), 10)

    def test_availability_reports_missing_weights(self):
        old = os.environ.get("CISTEM_BLUSH_WEIGHTS")
        os.environ["CISTEM_BLUSH_WEIGHTS"] = "/nonexistent/blush.ckpt"
        try:
            info = blush.availability(refresh=True)
            if info["torch"]:
                self.assertFalse(info["available"])
                self.assertIn("weights", info["reason"])
            else:
                self.assertIn("PyTorch", info["reason"])
        finally:
            if old is None:
                os.environ.pop("CISTEM_BLUSH_WEIGHTS")
            else:
                os.environ["CISTEM_BLUSH_WEIGHTS"] = old
            blush.availability(refresh=True)


@unittest.skipIf(torch is None, "torch not installed")
class PipelineTests(unittest.TestCase):
    class Identity(torch.nn.Module if torch else object):
        def forward(self, grid, local_std):
            return grid, torch.zeros_like(grid)

    def _blob(self, n=48):
        z, y, x = np.indices((n, n, n)) - n // 2
        r2 = x * x + y * y + z * z
        return np.exp(-(r2 / 30.0)).astype(np.float32), np.sqrt(r2)

    def test_identity_network_returns_the_masked_input(self):
        vol, r = self._blob()
        out = blush.denoise(vol, 2.0, mask_radius_a=40.0, fsc=None, model=self.Identity(), device="cpu")
        self.assertEqual(out.shape, vol.shape)
        self.assertTrue(np.allclose(out[r < 8], vol[r < 8], atol=2e-3))   # inside: unchanged
        self.assertTrue((np.abs(out[r > 26]) < 1e-3).all())              # beyond the mask's edge (22.5 + 2.5 voxels): nothing

    def test_spectral_trailing_removes_high_frequencies(self):
        vol, r = self._blob()
        rng = np.random.default_rng(1)
        noisy = vol + 0.3 * rng.standard_normal(vol.shape).astype(np.float32)
        fsc = np.ones(25); fsc[6:] = 0.0            # the data end at shell 6 of 24
        out = blush.denoise(noisy, 2.0, mask_radius_a=40.0, fsc=fsc, model=self.Identity(), device="cpu")
        df = np.abs(blush.fft(out))
        shells = blush.fourier_shells(df.shape)
        self.assertLess(df[shells > 8].max(), 1e-3 * df.max())
        plain = blush.denoise(noisy, 2.0, mask_radius_a=40.0, fsc=None, input_is_filtered=True, model=self.Identity(), device="cpu")
        self.assertGreater(np.abs(blush.fft(plain))[shells > 8].max(), 1e-2 * df.max())   # no cut without an FSC on a filtered input

    def test_progress_is_reported_and_can_cancel(self):
        vol, _ = self._blob(64)
        seen = []
        blush.denoise(vol, 2.0, mask_radius_a=60.0, model=self.Identity(), device="cpu", batch_size=4, progress=lambda d, t: seen.append((d, t)))
        self.assertTrue(seen and seen[-1][0] == seen[-1][1] > 0)
        with self.assertRaises(blush.BlushCancelled):
            blush.denoise(vol, 2.0, mask_radius_a=60.0, model=self.Identity(), device="cpu", progress=lambda d, t: False)

    @unittest.skipUnless(blush.availability()["available"], "Blush weights not installed")
    def test_process_pool_matches_single_process(self):
        vol, _ = self._blob(48)
        rng = np.random.default_rng(4)
        noisy = vol + 0.5 * rng.standard_normal(vol.shape).astype(np.float32)
        single = blush.denoise(noisy, 2.0, mask_radius_a=40.0, fsc=None, batch_size=2)
        seen = []
        pooled = blush.denoise(noisy, 2.0, mask_radius_a=40.0, fsc=None, batch_size=2, processes=2, threads=2, progress=lambda d, t: seen.append((d, t)))
        self.assertLess(float(np.abs(single - pooled).max()), 1e-4 * float(np.abs(single).max()))
        self.assertTrue(seen and seen[-1][0] == seen[-1][1] > 0)
        with self.assertRaises(blush.BlushCancelled):
            blush.denoise(noisy, 2.0, mask_radius_a=40.0, fsc=None, processes=2, threads=1, progress=lambda d, t: False)

    @unittest.skipUnless(blush.availability()["available"], "Blush weights not installed")
    def test_real_model_runs(self):
        vol, r = self._blob(48)
        rng = np.random.default_rng(2)
        noisy = vol + 0.5 * rng.standard_normal(vol.shape).astype(np.float32)
        out = blush.denoise(noisy, 2.0, mask_radius_a=40.0, fsc=None, batch_size=2)
        self.assertEqual(out.shape, vol.shape)
        self.assertTrue(np.isfinite(out).all())
        self.assertLess(out[(r > 14) & (r < 22)].std(), noisy[(r > 14) & (r < 22)].std())   # quieter solvent


class CompanionDeleteTests(unittest.TestCase):
    """Deleting a volume asset removes its Blush companion, and leaves the volume file as before."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        import auth
        import db
        import cistem_server
        import refine3d
        self.tmp = tempfile.mkdtemp()
        self._root, self._auth, self._sys = db.PROJECTS_ROOT, auth.AUTH_DB_PATH, db.SYSTEM_DB_PATH
        db.PROJECTS_ROOT = Path(self.tmp) / "projects"; auth.AUTH_DB_PATH = Path(self.tmp) / "auth.db"; db.SYSTEM_DB_PATH = Path(self.tmp) / "system.db"
        self.project = "t-blush"
        vol_dir = db.PROJECTS_ROOT / self.project / "Assets" / "Volumes"
        (vol_dir / "Blushed").mkdir(parents=True)
        self.volume = vol_dir / "volume_7_1.mrc"
        self.volume.write_bytes(b"x")
        self.companion = Path(refine3d.blushed_file(self.project, str(self.volume)))
        self.companion.write_bytes(b"y")
        self.other = Path(refine3d.blushed_file(self.project, str(vol_dir / "volume_7_2.mrc")))
        self.other.write_bytes(b"z")
        conn = db.get_conn(self.project)
        with conn:
            conn.execute("INSERT INTO VOLUME_ASSETS(VOLUME_ASSET_ID, NAME, FILENAME, PIXEL_SIZE, X_SIZE, Y_SIZE, Z_SIZE, RECONSTRUCTION_JOB_ID) VALUES (1, 'v', ?, 1.0, 8, 8, 8, -1)", (str(self.volume),))
        conn.close()
        token = auth.create_session(auth.create_user("boss", "password123", "admin")["id"])
        self.client = cistem_server.app.test_client()
        self.headers = {"Authorization": "Bearer " + token}

    def tearDown(self):
        import auth
        import db
        db.PROJECTS_ROOT, auth.AUTH_DB_PATH, db.SYSTEM_DB_PATH = self._root, self._auth, self._sys

    def test_delete_removes_the_companion_only(self):
        self.assertTrue(self.companion.is_file())
        r = self.client.post("/api/projects/{}/volumes/delete".format(self.project), json={"volume_ids": [1]}, headers=self.headers)
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        self.assertEqual(r.get_json()["deleted"], 1)
        self.assertFalse(self.companion.exists())
        self.assertTrue(self.volume.is_file())      # the asset's own file stays, as for every asset kind
        self.assertTrue(self.other.is_file())       # another volume's companion is untouched

    def test_companion_name(self):
        import refine3d
        self.assertTrue(refine3d.blushed_file("p", "/a/b/startup_volume_3_2.mrc").endswith("/Assets/Volumes/Blushed/startup_volume_3_2_blushed.mrc"))


class DriverSettingsTests(unittest.TestCase):
    def test_blush_settings_parse(self):
        import refine3d
        pkg = {"PARTICLE_SIZE": 150.0, "OUTPUT_PIXEL_SIZE": 1.0}
        s = refine3d.settings_from_params({"use_blush": True, "blush_input": "Filtered reference", "blush_batch_size": "0", "blush_processes": "8", "blush_threads": "16"}, pkg)
        self.assertTrue(s["use_blush"])
        self.assertFalse(s["blush_unfiltered"])
        self.assertEqual(s["blush_batch_size"], 1)
        self.assertEqual((s["blush_processes"], s["blush_threads"]), (8, 16))
        s = refine3d.settings_from_params({}, pkg)
        self.assertFalse(s["use_blush"])
        self.assertTrue(s["blush_unfiltered"])


if __name__ == "__main__":
    unittest.main()
