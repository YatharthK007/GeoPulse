"""Image-quality and segmentation metrics: PSNR, SSIM, EPI, mIoU, BD-Rate."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def _f(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def psnr(ref: np.ndarray, test: np.ndarray, data_range: float = 255.0) -> float:
    """Peak signal-to-noise ratio in dB (inf for identical images)."""
    mse = float(np.mean((_f(ref) - _f(test)) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def ssim(ref: np.ndarray, test: np.ndarray, data_range: float = 255.0) -> float:
    """Mean structural similarity (multichannel aware)."""
    ref, test = np.asarray(ref), np.asarray(test)
    ch = -1 if ref.ndim == 3 else None
    return float(structural_similarity(ref, test, data_range=data_range, channel_axis=ch))


def _gray(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 3:
        a = cv2.cvtColor(a.astype(np.uint8) if a.dtype != np.uint8 else a, cv2.COLOR_BGR2GRAY)
    return a.astype(np.float64)


def edge_preservation_index(ref: np.ndarray, test: np.ndarray) -> float:
    """EPI: correlation of Laplacian high-pass images (1.0 = edges perfectly kept)."""
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    hr = cv2.filter2D(_gray(ref), -1, k)
    ht = cv2.filter2D(_gray(test), -1, k)
    hr -= hr.mean()
    ht -= ht.mean()
    den = np.sqrt(np.sum(hr ** 2) * np.sum(ht ** 2))
    return float(np.sum(hr * ht) / den) if den > 0 else 0.0


epi = edge_preservation_index


def miou(pred: np.ndarray, gt: np.ndarray, num_classes: int = 10,
         ignore_index: int = 255, return_per_class: bool = False):
    """Mean intersection-over-union over classes present in pred or gt."""
    pred, gt = np.asarray(pred), np.asarray(gt)
    valid = gt != ignore_index
    ious = np.full(num_classes, np.nan)
    for c in range(num_classes):
        p, g = (pred == c) & valid, (gt == c) & valid
        union = np.logical_or(p, g).sum()
        if union > 0:
            ious[c] = np.logical_and(p, g).sum() / union
    m = float(np.nanmean(ious)) if np.any(~np.isnan(ious)) else 0.0
    return (m, ious) if return_per_class else m


def bd_rate(rate_ref, psnr_ref, rate_test, psnr_test) -> float:
    """Bjontegaard delta-rate (%) of test vs reference; negative = test saves bits.

    Uses cubic polynomial fit of log10(rate) as function of quality (>=4 points advised).
    """
    r1, r2 = np.log10(np.asarray(rate_ref, float)), np.log10(np.asarray(rate_test, float))
    q1, q2 = np.asarray(psnr_ref, float), np.asarray(psnr_test, float)
    deg = min(3, len(q1) - 1, len(q2) - 1)
    p1, p2 = np.polyfit(q1, r1, deg), np.polyfit(q2, r2, deg)
    lo, hi = max(q1.min(), q2.min()), min(q1.max(), q2.max())
    if hi <= lo:
        return float("nan")
    i1, i2 = np.polyint(p1), np.polyint(p2)
    avg1 = (np.polyval(i1, hi) - np.polyval(i1, lo)) / (hi - lo)
    avg2 = (np.polyval(i2, hi) - np.polyval(i2, lo)) / (hi - lo)
    return float((10 ** (avg2 - avg1) - 1) * 100.0)
