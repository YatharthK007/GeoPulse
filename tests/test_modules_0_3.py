"""Unit tests for Modules 0-3.  Run: python -m unittest discover -s tests -v"""
import os
import sys
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.compression import roi_codec as rc  # noqa: E402
from src.degradation import simulator as sim  # noqa: E402
from src.ingestion import radiometric as rad  # noqa: E402
from src.restoration import engine as eng  # noqa: E402
from src.utils import metrics as mt  # noqa: E402

S = os.path.join(ROOT, "data", "samples")


def load(name="urban_earthquake"):
    return cv2.imread(os.path.join(S, name + ".png"))


class TestMetrics(unittest.TestCase):
    def test_psnr_ssim_epi(self):
        a = load()
        self.assertEqual(mt.psnr(a, a), float("inf"))
        self.assertAlmostEqual(mt.ssim(a, a), 1.0, places=5)
        self.assertAlmostEqual(mt.epi(a, a), 1.0, places=5)
        b = cv2.GaussianBlur(a, (0, 0), 3)
        self.assertLess(mt.psnr(a, b), 40)
        self.assertLess(mt.epi(a, b), 0.9)

    def test_miou(self):
        g = np.array([[0, 0, 1, 1], [255, 2, 2, 2]])
        self.assertEqual(mt.miou(g, g, 3), 1.0)
        p = np.array([[0, 1, 1, 1], [0, 2, 2, 2]])
        self.assertTrue(0 < mt.miou(p, g, 3) < 1)

    def test_bd_rate(self):
        r = np.array([100, 200, 400, 800.0])
        q = np.array([30, 33, 36, 39.0])
        self.assertAlmostEqual(mt.bd_rate(r, q, r * 0.7, q), -30.0, places=3)


class TestRadiometric(unittest.TestCase):
    def test_metadata_exif(self):
        m = rad.extract_metadata(os.path.join(S, "flooded_highway.png"))
        self.assertEqual(m["metadata_source"], "exif")
        self.assertAlmostEqual(m["GPSLatitude"], 29.758, places=2)
        self.assertAlmostEqual(m["GPSLongitude"], -95.366, places=2)
        self.assertAlmostEqual(m["RelativeAltitude"], 110.0, places=0)
        self.assertAlmostEqual(m["GimbalPitchDegree"], -75.0)
        self.assertIn("geo_transform", m)

    def test_metadata_fallback(self):
        p = os.path.join(ROOT, "outputs", "_stripped.png")
        cv2.imwrite(p, load())
        m = rad.extract_metadata(p)
        self.assertEqual(m["metadata_source"], "synthetic_telemetry")
        self.assertEqual(m, rad.extract_metadata(p))  # deterministic
        os.remove(p)

    def test_raw_tonemap_colors_ndwi(self):
        img = load("hurricane_coast")
        r = rad.simulate_raw_and_tonemap(img)
        self.assertEqual(r["raw12"].dtype, np.uint16)
        self.assertLessEqual(int(r["raw12"].max()), 4095)
        self.assertEqual(r["sdr"].shape, img.shape)
        self.assertGreater(r["dynamic_range_stops"], 3)
        cs = rad.convert_color_spaces(img)
        self.assertEqual(set(cs), {"YCbCr", "LAB", "LUV"})
        nir = cv2.imread(os.path.join(S, "hurricane_coast_nir.png"), 0)
        mask = cv2.imread(os.path.join(S, "hurricane_coast_mask.png"), 0)
        ndwi = rad.compute_ndwi(img[..., 1], nir)
        self.assertGreater(ndwi[mask == 6].mean(), ndwi[mask == 7].mean() + 0.3)


class TestSimulator(unittest.TestCase):
    def test_psf_and_determinism(self):
        psf = sim.make_motion_psf(15, 35)
        self.assertAlmostEqual(psf.sum(), 1.0)
        img = load()
        a = sim.degrade_pipeline(img, "severe")["degraded"]
        b = sim.degrade_pipeline(img, "severe")["degraded"]
        self.assertTrue(np.array_equal(a, b))

    def test_presets_monotonic(self):
        img = load()
        p = [mt.psnr(img, sim.degrade_pipeline(img, s)["degraded"]) for s in ("mild", "moderate", "severe")]
        self.assertGreater(p[0], p[1])
        self.assertGreater(p[1], p[2])


class TestRestoration(unittest.TestCase):
    def test_blur_estimation(self):
        gray = cv2.cvtColor(load(), cv2.COLOR_BGR2GRAY)
        for L, A in [(15, 35), (21, 120), (25, 10)]:
            bl, _ = sim.apply_motion_blur(gray, L, A)
            e = eng.estimate_blur_fft(bl)
            d = abs(e["est_angle"] - A) % 180
            self.assertLess(min(d, 180 - d), 8, f"angle {A} -> {e['est_angle']}")
            self.assertLess(abs(e["est_length"] - L), 4, f"len {L} -> {e['est_length']}")
            self.assertEqual(e["spectrum_vis"].ndim, 3)

    def test_wiener_improves(self):
        img = load()
        bl, psf = sim.apply_motion_blur(img, 15, 35)
        rec = eng.wiener_deblur_color(bl, psf, 0.005)
        self.assertGreater(mt.psnr(img, rec), mt.psnr(img, bl) + 1.0)

    def test_dehaze(self):
        img = load("flooded_highway")
        hz = sim.apply_atmospheric_haze(img, 1.2, 0.85)
        out = eng.dehaze_guided_filter(hz)
        self.assertGreater(mt.psnr(img, out), mt.psnr(img, hz))

    def test_denoisers(self):
        img = load("landslide_village")
        noisy = sim.apply_sensor_noise(img, 0.01, 8.0)
        base = mt.psnr(img, noisy)
        for m in eng.DENOISERS:
            out = eng.denoise_suite(noisy, m)
            self.assertEqual(out.shape, img.shape)
            if m != "adaptive_median":  # impulse-noise filter; checked below
                self.assertGreater(mt.psnr(img, out), base, m)
        imp = img.copy()
        rng = np.random.default_rng(0)
        m = rng.random(img.shape[:2])
        imp[m < 0.03] = 0
        imp[m > 0.97] = 255
        self.assertGreater(mt.psnr(img, eng.denoise_suite(imp, "adaptive_median")), mt.psnr(img, imp) + 5)

    def test_cleanup_and_auto(self):
        img = load()
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 15])
        jp = cv2.imdecode(buf, 1)
        self.assertEqual(eng.cleanup_compression_artifacts(jp).shape, img.shape)
        d = sim.degrade_pipeline(img, "moderate")
        r = eng.restore_auto(d["degraded"])
        self.assertEqual(r["restored"].shape, img.shape)


class TestCompression(unittest.TestCase):
    def setUp(self):
        self.img = load("hurricane_coast")
        mask = cv2.imread(os.path.join(S, "hurricane_coast_mask.png"), 0)
        self.mask = mask
        self.roi = rc.generate_roi_importance_map(self.img, rc.mask_to_probs(mask))

    def test_roi_map(self):
        self.assertEqual(self.roi.shape, self.img.shape[:2])
        self.assertTrue(0 <= self.roi.min() and self.roi.max() <= 1)
        fg = np.isin(self.mask, rc.FG_CLASSES)
        self.assertGreater(self.roi[fg].mean(), self.roi[~fg].mean())

    def test_codecs_hit_bitrate(self):
        for codec in ("jpeg2000", "webp"):
            r = rc.encode_roi_adaptive(self.img, self.roi, 0.5, codec)
            self.assertLess(abs(r["actual_bpp"] - 0.5) / 0.5, 0.25, codec)
            self.assertEqual(r["reconstructed_img"].shape, self.img.shape)
            self.assertEqual(r["bit_allocation_heatmap"].shape, self.roi.shape)
        u = rc.encode_uniform(self.img, 0.5, "jpeg")
        self.assertLess(abs(u["actual_bpp"] - 0.5) / 0.5, 0.35)

    def test_roi_beats_uniform_on_foreground(self):
        roi = rc.encode_roi_adaptive(self.img, self.roi, 0.5, "jpeg2000")
        uni = rc.encode_uniform(self.img, 0.5, "jpeg2000")
        self.assertGreater(rc.fg_psnr(self.img, roi["reconstructed_img"], self.roi),
                           rc.fg_psnr(self.img, uni["reconstructed_img"], self.roi))

    def test_arithmetic_coder(self):
        rng = np.random.default_rng(0)
        x = np.round(rng.laplace(0, 2, 5000)).astype(int)
        r = rc.arithmetic_coding_demo(x)
        self.assertTrue(r["lossless_roundtrip"])
        self.assertLess(r["achieved_bps"], r["entropy_bps"] * 1.02 + 0.02)
        df = rc.dwt_subband_arithmetic_benchmark(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY))
        self.assertTrue(df["lossless_roundtrip"].all())

    def test_sweep(self):
        df = rc.rate_distortion_sweep(self.img, self.roi, bpps=(0.3, 0.6))
        self.assertGreaterEqual(len(df), 10)
        self.assertIn("fg_psnr", df)


if __name__ == "__main__":
    unittest.main()
