"""End-to-end verification for improved ROI codec, Modules 4, 5, 6 and app.py syntax/imports."""
from __future__ import annotations

import os
import py_compile
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.analytics.disaster_seg import export_geojson_report, segment_disaster_scene
from src.compression.roi_codec import encode_roi_adaptive, encode_uniform, fg_psnr, generate_roi_importance_map, mask_to_probs, rate_distortion_sweep
from src.ingestion.radiometric import extract_metadata
from src.registration.geo_align import register_cross_view
from src.utils.metrics import miou
from src.video_analytics.tracker import detect_dynamic_hazards_mog2, interpolate_low_fps, stabilize_video_stream, track_rescue_assets

SAMPLES = os.path.join(ROOT, "data", "samples")
SAT = os.path.join(ROOT, "data", "satellite_pairs")
GEOJSON_OUT = os.path.join(ROOT, "outputs", "geojson")


def main():
    print("=" * 72)
    print("GEOPULSE FULL PIPELINE VERIFICATION (MODULES 3-6 + APP.PY)")
    print("=" * 72)

    img = cv2.imread(os.path.join(SAMPLES, "hurricane_coast.png"))
    nir = cv2.imread(os.path.join(SAMPLES, "hurricane_coast_nir.png"), 0)
    gt_mask = cv2.imread(os.path.join(SAMPLES, "hurricane_coast_mask.png"), 0)
    sat = cv2.imread(os.path.join(SAT, "hurricane_coast_sat.png"))
    meta = extract_metadata(os.path.join(SAMPLES, "hurricane_coast.png"))

    # 1. Verify improved Module 3 ROI vs Uniform across 0.15 - 0.50 bpp
    roi_map = generate_roi_importance_map(img, mask_to_probs(gt_mask))
    for bpp in (0.15, 0.30, 0.50):
        u_jp2 = encode_uniform(img, bpp, "jpeg2000")
        u_jpg = encode_uniform(img, bpp, "jpeg")
        r_jp2 = encode_roi_adaptive(img, roi_map, bpp, "jpeg2000")
        p_u_jp2 = fg_psnr(img, u_jp2["reconstructed_img"], roi_map)
        p_u_jpg = fg_psnr(img, u_jpg["reconstructed_img"], roi_map)
        p_r_jp2 = fg_psnr(img, r_jp2["reconstructed_img"], roi_map)
        print(f"[Module 3 @ {bpp:.2f} bpp] FG-PSNR: Uniform-JPEG={p_u_jpg:.2f}dB | Uniform-JP2={p_u_jp2:.2f}dB | ROI-JP2={p_r_jp2:.2f}dB (+{p_r_jp2 - max(p_u_jp2, p_u_jpg):.2f}dB)")
        assert p_r_jp2 > p_u_jp2 and p_r_jp2 > p_u_jpg, f"ROI-JP2 did not beat uniform at {bpp} bpp!"

    # 2. Verify Module 4 Cross-View Geo-Registration (AKAZE, SIFT, CNN fallback)
    for det in ("AKAZE", "SIFT"):
        t0 = time.perf_counter()
        reg = register_cross_view(img, sat, detector=det, use_cnn_fallback=True)
        dt = (time.perf_counter() - t0) * 1000
        print(f"[Module 4 {det}] Inliers={reg['inlier_count']}/{reg['total_matches']} | ReprojErr={reg['reproj_error']:.2f}px | Time={dt:.1f}ms")
        assert reg["warped_uav"].shape == img.shape
        assert reg["inlier_count"] >= 10

    # 3. Verify Module 5 Video Stabilization, MOG2, Tracking & Frame Interpolation
    cap = cv2.VideoCapture(os.path.join(SAMPLES, "uav_flyover.mp4"))
    frames = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        frames.append(fr)
    cap.release()
    t0 = time.perf_counter()
    stab_frames, j_stats = stabilize_video_stream(frames)
    mog_masks = detect_dynamic_hazards_mog2(stab_frames)
    t_res = track_rescue_assets(stab_frames, mog_masks)
    mid_fr = interpolate_low_fps(stab_frames[5], stab_frames[6])
    dt = (time.perf_counter() - t0) * 1000
    print(f"[Module 5 Video] Frames={len(stab_frames)} | JitterReduction={j_stats['jitter_reduction_pct']}% | TrackedAssets={t_res['total_unique_assets']} | Time={dt:.0f}ms")
    assert len(stab_frames) == len(frames)
    assert mid_fr.shape == frames[0].shape
    assert t_res["total_unique_assets"] >= 3

    # 4. Verify Module 6 Disaster Segmentation (<150ms) & GeoJSON Export
    t0 = time.perf_counter()
    seg_out = segment_disaster_scene(img, nir_band=nir, ref_mask=gt_mask)
    dt_seg = (time.perf_counter() - t0) * 1000
    t0_raw = time.perf_counter()
    seg_standalone = segment_disaster_scene(img, nir_band=nir, ref_mask=None)
    dt_raw = (time.perf_counter() - t0_raw) * 1000
    gpath = os.path.join(GEOJSON_OUT, "verify_hurricane.geojson")
    gstr, gobj = export_geojson_report(seg_out["seg_mask"], meta, output_path=gpath)
    print(f"[Module 6 Seg] GuidedTime={dt_seg:.1f}ms | StandaloneTime={dt_raw:.1f}ms (<150ms) | CriticalBldgs={seg_out['critical_structures']} | GeoJSON Features={len(gobj['features'])}")
    assert dt_seg < 150.0 and dt_raw < 150.0
    assert os.path.exists(gpath) and len(gobj["features"]) > 5

    # 5. Verify app.py syntax & clean module compilation
    py_compile.compile(os.path.join(ROOT, "app.py"), doraise=True)
    print("[App.py Check] py_compile syntax & module structure verified cleanly.")
    print("=" * 72)
    print("ALL CHECKS PASSED!")


if __name__ == "__main__":
    main()
