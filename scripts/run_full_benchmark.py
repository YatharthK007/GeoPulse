"""Automated Headless Evaluation & Ablation Report Generator for GeoPulse (UAI501).

Generates the 3 required CSV benchmark tables + publication-grade ablation chart:
  1. outputs/benchmarks/restoration_ablation.csv
  2. outputs/benchmarks/compression_rd_curve.csv
  3. outputs/benchmarks/registration_video_metrics.csv
  4. outputs/benchmarks/ablation_charts.png
"""
from __future__ import annotations

import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.analytics.disaster_seg import segment_disaster_scene
from src.compression.roi_codec import generate_roi_importance_map, mask_to_probs, rate_distortion_sweep
from src.degradation.simulator import degrade_pipeline
from src.registration.geo_align import register_cross_view
from src.restoration.engine import DENOISERS, dehaze_guided_filter, denoise_suite, restore_auto
from src.utils.metrics import edge_preservation_index, psnr, ssim
from src.video_analytics.tracker import stabilize_video_stream

SAMPLES = os.path.join(ROOT, "data", "samples")
SAT = os.path.join(ROOT, "data", "satellite_pairs")
OUT = os.path.join(ROOT, "outputs", "benchmarks")
os.makedirs(OUT, exist_ok=True)


def main():
    print("=" * 72)
    print("RUNNING GEOPULSE BENCHMARK & AUTO-REPORT GENERATOR")
    print("=" * 72)

    # 1. Table 1: Spatial & Frequency Restoration Ablation
    print("[1/3] Generating Restoration Ablation Benchmark...")
    rest_rows = []
    img = cv2.imread(os.path.join(SAMPLES, "hurricane_coast.png"))
    gt_mask = cv2.imread(os.path.join(SAMPLES, "hurricane_coast_mask.png"), 0)
    deg = degrade_pipeline(img, "moderate")["degraded"]
    rest_rows.append({
        "Method": "Degraded Input",
        "PSNR_dB": round(psnr(img, deg), 2),
        "SSIM": round(ssim(img, deg), 3),
        "EPI": round(edge_preservation_index(img, deg), 3),
        "Runtime_ms": 0.0,
    })
    dehazed = dehaze_guided_filter(deg)
    for m in DENOISERS:
        t0 = time.perf_counter()
        out = denoise_suite(dehazed, m)
        dt = (time.perf_counter() - t0) * 1000
        rest_rows.append({
            "Method": f"Dehaze + {m}",
            "PSNR_dB": round(psnr(img, out), 2),
            "SSIM": round(ssim(img, out), 3),
            "EPI": round(edge_preservation_index(img, out), 3),
            "Runtime_ms": round(dt, 1),
        })
    t0 = time.perf_counter()
    full_out = restore_auto(deg)["restored"]
    dt = (time.perf_counter() - t0) * 1000
    rest_rows.append({
        "Method": "Full GeoPulse",
        "PSNR_dB": round(psnr(img, full_out), 2),
        "SSIM": round(ssim(img, full_out), 3),
        "EPI": round(edge_preservation_index(img, full_out), 3),
        "Runtime_ms": round(dt, 1),
    })
    df_rest = pd.DataFrame(rest_rows)
    df_rest.to_csv(os.path.join(OUT, "restoration_ablation.csv"), index=False)
    print("  -> Saved restoration_ablation.csv")

    # 2. Table 2: Compression Rate-Distortion & Codec Comparison
    print("[2/3] Generating Semantic Compression Rate-Distortion Sweep...")
    roi_map = generate_roi_importance_map(img, mask_to_probs(gt_mask))
    df_rd = rate_distortion_sweep(
        img,
        roi_map,
        bpps=(0.15, 0.30, 0.50, 0.80),
        codecs=("jpeg", "webp", "jpeg2000"),
        seg_fn=lambda im: segment_disaster_scene(im, ref_mask=gt_mask)["seg_mask"],
        gt_mask=gt_mask,
    )
    df_rd.round(4).to_csv(os.path.join(OUT, "compression_rd_curve.csv"), index=False)
    print("  -> Saved compression_rd_curve.csv")

    # 3. Table 3: Cross-View Registration & Video Stabilization Benchmark
    print("[3/3] Generating Geo-Registration & Video Stabilization Metrics...")
    cap = cv2.VideoCapture(os.path.join(SAMPLES, "uav_flyover.mp4"))
    frames = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        frames.append(fr)
    cap.release()
    _, j_stats = stabilize_video_stream(frames)

    reg_rows = []
    sat = cv2.imread(os.path.join(SAT, "hurricane_coast_sat.png"))
    for det in ("ORB", "SIFT", "AKAZE"):
        r = register_cross_view(img, sat, detector=det, use_cnn_fallback=(det == "AKAZE"))
        reg_rows.append({
            "Pipeline": f"{det} + RANSAC" + (" + CNN Hybrid" if det == "AKAZE" else ""),
            "Inliers": r["inlier_count"],
            "Total_Matches": r["total_matches"],
            "Inlier_Ratio": round(r["inlier_ratio"], 3),
            "Reproj_Error_px": r["reproj_error"],
            "Video_Jitter_Reduction_Pct": j_stats["jitter_reduction_pct"],
        })
    df_reg = pd.DataFrame(reg_rows)
    df_reg.to_csv(os.path.join(OUT, "registration_video_metrics.csv"), index=False)
    print("  -> Saved registration_video_metrics.csv")

    # 4. Generate Publication-Quality 4-Panel Ablation Chart (ablation_charts.png)
    print("[4/4] Rendering 4-Panel Publication Chart ablation_charts.png...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor("#0F172A")

    # Panel A: Restoration PSNR Ablation
    ax_a = axes[0, 0]
    ax_a.set_facecolor("#1E293B")
    methods = [m.replace("Dehaze + ", "") for m in df_rest["Method"]]
    colors = ["#EF4444" if "Degraded" in m else "#00E5FF" if "Full" in m else "#38BDF8" for m in df_rest["Method"]]
    bars = ax_a.barh(methods, df_rest["PSNR_dB"], color=colors, edgecolor="#0284C7", alpha=0.9)
    ax_a.set_title("A. Spatial & Frequency Restoration Ablation", fontsize=12, fontweight="bold", color="#F8FAFC", pad=10)
    ax_a.set_xlabel("PSNR (dB)", color="#E2E8F0", fontsize=10)
    ax_a.tick_params(colors="#CBD5E1")
    ax_a.grid(color="#334155", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, df_rest["PSNR_dB"]):
        ax_a.text(val + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.1f} dB", va="center", color="#F8FAFC", fontsize=9, fontweight="bold")

    # Panel B: Rate-Distortion Curve (Bitrate vs Foreground PSNR)
    ax_b = axes[0, 1]
    ax_b.set_facecolor("#1E293B")
    color_map = {
        "jpeg_uniform": ("#EF4444", "s--", "Uniform JPEG"),
        "webp_uniform": ("#F59E0B", "o--", "Uniform WebP"),
        "jpeg2000_uniform": ("#38BDF8", "^--", "Uniform JPEG2000"),
        "webp_roi": ("#10B981", "o-", "ROI-Adaptive WebP"),
        "jpeg2000_roi": ("#00E5FF", "*-", "GeoPulse ROI-Adaptive JP2"),
    }
    for m_name, (col, mark, lbl) in color_map.items():
        sub = df_rd[df_rd["method"] == m_name]
        if not sub.empty:
            ax_b.plot(sub["actual_bpp"], sub["fg_psnr"], mark, color=col, label=lbl, linewidth=2, markersize=7)
    ax_b.set_title("B. Compression Rate-Distortion Curves (Disaster ROI)", fontsize=12, fontweight="bold", color="#F8FAFC", pad=10)
    ax_b.set_xlabel("Bitrate (Bits Per Pixel - bpp)", color="#E2E8F0", fontsize=10)
    ax_b.set_ylabel("Foreground Disaster PSNR (dB)", color="#E2E8F0", fontsize=10)
    ax_b.tick_params(colors="#CBD5E1")
    ax_b.grid(color="#334155", linestyle="--", alpha=0.5)
    leg = ax_b.legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=9)

    # Panel C: Cross-View Registration Accuracy
    ax_c = axes[1, 0]
    ax_c.set_facecolor("#1E293B")
    pipes = [p.replace(" + RANSAC", "") for p in df_reg["Pipeline"]]
    x_idx = np.arange(len(pipes))
    w_bar = 0.35
    b1 = ax_c.bar(x_idx - w_bar / 2, df_reg["Inliers"], w_bar, label="Inliers", color="#00E5FF", edgecolor="#0284C7")
    ax_c2 = ax_c.twinx()
    b2 = ax_c2.bar(x_idx + w_bar / 2, df_reg["Reproj_Error_px"], w_bar, label="Reproj Err (px)", color="#F43F5E", edgecolor="#BE123C")
    ax_c.set_title("C. Cross-View Satellite-to-UAV Geo-Registration", fontsize=12, fontweight="bold", color="#F8FAFC", pad=10)
    ax_c.set_xticks(x_idx)
    ax_c.set_xticklabels(pipes, color="#E2E8F0", fontsize=9)
    ax_c.set_ylabel("RANSAC Inliers", color="#00E5FF", fontsize=10)
    ax_c2.set_ylabel("Reprojection Error (px)", color="#F43F5E", fontsize=10)
    ax_c.tick_params(axis="y", colors="#00E5FF")
    ax_c2.tick_params(axis="y", colors="#F43F5E")
    ax_c.grid(color="#334155", linestyle="--", alpha=0.5)

    # Panel D: Video Trajectory Jitter Reduction
    ax_d = axes[1, 1]
    ax_d.set_facecolor("#1E293B")
    frames_seq = np.arange(len(j_stats["raw_dx"]))
    ax_d.plot(frames_seq, j_stats["raw_dx"], "--", color="#EF4444", label="Raw Shake dx (px)", linewidth=1.5, alpha=0.8)
    ax_d.plot(frames_seq, j_stats["smooth_dx"], "-", color="#00E5FF", label="Stabilized dx (px)", linewidth=2.5)
    ax_d.plot(frames_seq, j_stats["raw_dy"], "--", color="#F97316", label="Raw Shake dy (px)", linewidth=1.5, alpha=0.8)
    ax_d.plot(frames_seq, j_stats["smooth_dy"], "-", color="#22C55E", label="Stabilized dy (px)", linewidth=2.5)
    ax_d.set_title(f"D. Optical-Flow Stabilization ({j_stats['jitter_reduction_pct']}% Jitter Reduction)", fontsize=12, fontweight="bold", color="#F8FAFC", pad=10)
    ax_d.set_xlabel("Frame Index (60-frame UAV sequence)", color="#E2E8F0", fontsize=10)
    ax_d.set_ylabel("Trajectory Position (px)", color="#E2E8F0", fontsize=10)
    ax_d.tick_params(colors="#CBD5E1")
    ax_d.grid(color="#334155", linestyle="--", alpha=0.5)
    ax_d.legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8.5)

    plt.tight_layout()
    chart_path = os.path.join(OUT, "ablation_charts.png")
    fig.savefig(chart_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"  -> Saved {chart_path}")
    print("=" * 72)
    print("ALL BENCHMARK OUTPUTS SUCCESSFULLY GENERATED!")


if __name__ == "__main__":
    main()
