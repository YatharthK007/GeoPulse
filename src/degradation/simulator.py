"""Module 1: deterministic physics-based degradation simulator."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

PRESETS = {
    "mild":     dict(length=7,  angle=20.0, beta=0.4, airlight=0.85, shot_scale=0.002, read_sigma=2.0, gamma=1.0),
    "moderate": dict(length=15, angle=35.0, beta=0.8, airlight=0.85, shot_scale=0.004, read_sigma=4.0, gamma=1.2),
    "severe":   dict(length=25, angle=35.0, beta=1.2, airlight=0.85, shot_scale=0.008, read_sigma=8.0, gamma=1.5),
}


def make_motion_psf(length: float = 15, angle: float = 35.0) -> np.ndarray:
    """Normalised linear-motion PSF. angle: degrees, counter-clockwise as seen on screen
    (0 = horizontal). Built by bilinear splatting so non-integer lengths are exact."""
    half = int(np.ceil(length / 2.0)) + 1
    size = 2 * half + 1
    k = np.zeros((size, size), np.float64)
    th = np.deg2rad(angle)
    dx, dy = np.cos(th), -np.sin(th)
    n = max(int(length * 8), 8)
    for t in np.linspace(-length / 2.0, length / 2.0, n):
        x, y = half + t * dx, half + t * dy
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        k[y0, x0] += (1 - fx) * (1 - fy)
        k[y0, x0 + 1] += fx * (1 - fy)
        k[y0 + 1, x0] += (1 - fx) * fy
        k[y0 + 1, x0 + 1] += fx * fy
    return (k / k.sum()).astype(np.float64)


def apply_motion_blur(img: np.ndarray, length: int = 15, angle: float = 35.0) -> Tuple[np.ndarray, np.ndarray]:
    psf = make_motion_psf(length, angle)
    out = cv2.filter2D(img.astype(np.float32), -1, psf.astype(np.float32), borderType=cv2.BORDER_REFLECT)
    return np.clip(out, 0, 255).astype(np.uint8), psf


def _depth_map(h: int, w: int, seed: int = 7) -> np.ndarray:
    """Smooth synthetic depth/elevation gradient in [0,1] (far at top)."""
    rng = np.random.default_rng(seed)
    yy = np.linspace(1.0, 0.0, h)[:, None] * np.ones((1, w))
    xx = np.linspace(0, 1, w)[None, :] * np.ones((h, 1))
    noise = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), max(h, w) / 8.0)
    noise = (noise - noise.min()) / (np.ptp(noise) + 1e-9)
    d = 0.6 * yy + 0.15 * xx + 0.25 * noise
    return np.clip((d - d.min()) / (np.ptp(d) + 1e-9), 0, 1)


def apply_atmospheric_haze(img: np.ndarray, beta: float = 1.2, airlight: float = 0.85) -> np.ndarray:
    """Koschmieder model I = J t + A (1 - t), t = exp(-beta d)."""
    h, w = img.shape[:2]
    t = np.exp(-beta * _depth_map(h, w))[..., None]
    j = img.astype(np.float64) / 255.0
    out = j * t + airlight * (1 - t)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def apply_sensor_noise(img: np.ndarray, shot_scale: float = 0.02, read_sigma: float = 8.0, seed: int = 0) -> np.ndarray:
    """Signal-dependent Poisson shot noise + additive Gaussian read noise (deterministic)."""
    rng = np.random.default_rng(seed)
    x = img.astype(np.float64) / 255.0
    shot = rng.poisson(x / shot_scale) * shot_scale
    out = shot * 255.0 + rng.normal(0.0, read_sigma, img.shape)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def degrade_pipeline(img: np.ndarray, severity: str = "severe", seed: int = 0) -> dict:
    """Blur -> haze -> low-light gamma drop -> sensor noise, using a named preset."""
    if severity not in PRESETS:
        raise ValueError(f"severity must be one of {list(PRESETS)}")
    p = PRESETS[severity]
    blurred, psf = apply_motion_blur(img, p["length"], p["angle"])
    hazy = apply_atmospheric_haze(blurred, p["beta"], p["airlight"])
    dark = (255.0 * (hazy.astype(np.float64) / 255.0) ** p["gamma"]).astype(np.uint8)
    noisy = apply_sensor_noise(dark, p["shot_scale"], p["read_sigma"], seed)
    return {"clean": img, "degraded": noisy, "blurred": blurred, "hazy": hazy, "psf": psf,
            "params": dict(p), "severity": severity}
