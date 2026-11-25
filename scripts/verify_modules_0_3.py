"""Quick verification that Modules 0-3 and the demo assets run cleanly.  python scripts/verify_modules_0_3.py"""
import os
import sys
import time
import traceback

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
S, SAT, OUT = (os.path.join(ROOT, *p) for p in (("data", "samples"), ("data", "satellite_pairs"), ("outputs", "benchmarks")))
os.makedirs(OUT, exist_ok=True)
results = []


def check(name):
    def deco(fn):
        t = time.time()
        try:
            info = fn()
            results.append((name, "PASS", f"{time.time() - t:.1f}s", info or ""))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            results.append((name, "FAIL", f"{time.time() - t:.1f}s", repr(e)))
        return fn
    return deco


SCENES = ["hurricane_coast", "urban_earthquake", "flooded_highway", "landslide_village"]


@check("imports")
def _():
    import numpy, scipy, cv2, skimage, pywt, PIL, exifread, pandas, plotly, streamlit, torch, torchvision  # noqa
    return f"torch {torch.__version__} cuda={torch.cuda.is_available()} cv2 {cv2.__version__} ximgproc={hasattr(cv2, 'ximgproc')}"


@check("assets")
def _():
    for s in SCENES:
        im = cv2.imread(f"{S}/{s}.png")
        assert im.shape == (512, 768, 3), s
        assert cv2.imread(f"{S}/{s}_nir.png", 0).shape == (512, 768)
        m = cv2.imread(f"{S}/{s}_mask.png", 0)
        assert set(np.unique(m)) <= set(range(10)) | {255}
        assert cv2.imread(f"{SAT}/{s}_sat.png") is not None
    cap = cv2.VideoCapture(f"{S}/uav_flyover.mp4")
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 60 and abs(cap.get(cv2.CAP_PROP_FPS) - 20) < 0.5
    return "4 scenes + sat tiles + 60-frame video"


from src.compression import roi_codec as rc  # noqa: E402
from src.degradation import simulator as sim  # noqa: E402
from src.ingestion import radiometric as rad  # noqa: E402
from src.restoration import engine as eng  # noqa: E402
from src.utils import metrics as mt  # noqa: E402

img = cv2.imread(f"{S}/hurricane_coast.png")
mask = cv2.imread(f"{S}/hurricane_coast_mask.png", 0)


@check("module0_ingestion")
def _():
    m = rad.extract_metadata(f"{S}/hurricane_coast.png")
    r = rad.simulate_raw_and_tonemap(img)
    return f"{m['metadata_source']} lat={m['GPSLatitude']:.4f} alt={m['RelativeAltitude']}m DR={r['dynamic_range_stops']:.1f} stops"


@check("module1_simulator")
def _():
    rows = []
    for sev in ("mild", "moderate", "severe"):
        d = sim.degrade_pipeline(img, sev)["degraded"]
        rows.append(f"{sev}={mt.psnr(img, d):.1f}dB")
    return " ".join(rows)


@check("module2_restoration")
def _():
    d = sim.degrade_pipeline(img, "moderate")
    est = eng.estimate_blur_fft(cv2.cvtColor(d["blurred"], cv2.COLOR_BGR2GRAY))
    rows = []
    noisy = sim.apply_sensor_noise(img, 0.01, 8.0)
    for m in eng.DENOISERS:
        t = time.time()
        o = eng.denoise_suite(noisy, m)
        rows.append({"method": m, "psnr": mt.psnr(img, o), "ssim": mt.ssim(img, o), "epi": mt.epi(img, o),
                     "ms": (time.time() - t) * 1000})
    pd.DataFrame(rows).round(3).to_csv(f"{OUT}/verify_denoisers.csv", index=False)
    full = eng.restore_auto(d["degraded"])["restored"]
    return f"blur est angle={est['est_angle']:.0f} (true 35) len={est['est_length']} (true 15); full-pipeline {mt.psnr(img, d['degraded']):.1f}->{mt.psnr(img, full):.1f}dB"


@check("module3_compression")
def _():
    roi = rc.generate_roi_importance_map(img, rc.mask_to_probs(mask))
    df = rc.rate_distortion_sweep(img, roi)
    df.round(3).to_csv(f"{OUT}/verify_rd_sweep.csv", index=False)
    ac = rc.dwt_subband_arithmetic_benchmark(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    ac.round(3).to_csv(f"{OUT}/verify_arithmetic.csv", index=False)
    assert ac["lossless_roundtrip"].all()
    p = df[df.target_bpp == 0.5].set_index("method")["fg_psnr"]
    return f"RD sweep rows={len(df)}; fg-PSNR@0.5bpp jp2 uniform={p['jpeg2000_uniform']:.1f} roi={p['jpeg2000_roi']:.1f}"


print("\n" + "=" * 70)
for r in results:
    print(f"[{r[1]}] {r[0]:<22} {r[2]:>6}  {r[3]}")
sys.exit(0 if all(r[1] == "PASS" for r in results) else 1)
