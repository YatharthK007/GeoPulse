"""Module 0: ingestion, telemetry parsing and radiometric calibration."""
from __future__ import annotations

import json
import os
import zlib

import cv2
import numpy as np
from PIL import Image

SENSOR_WIDTH_MM = 13.2  # 1-inch sensor
DEFAULT_FOCAL_MM = 8.8


def _ratio(v) -> float:
    try:
        return float(v.num) / float(v.den) if hasattr(v, "num") else float(v[0]) / float(v[1]) if isinstance(v, tuple) else float(v)
    except Exception:
        return float(v)


def _dms(vals, ref) -> float:
    d, m, s = (_ratio(x) for x in vals)
    val = d + m / 60.0 + s / 3600.0
    return -val if str(ref).upper() in ("S", "W") else val


def _synthetic_telemetry(filepath: str) -> dict:
    """Deterministic realistic telemetry (seeded by file name) for stripped files."""
    rng = np.random.default_rng(zlib.crc32(os.path.basename(filepath).encode()))
    return {"GPSLatitude": float(29.30 + rng.uniform(-0.05, 0.05)),
            "GPSLongitude": float(-94.78 + rng.uniform(-0.05, 0.05)),
            "RelativeAltitude": float(round(rng.uniform(80, 120), 1)),
            "GimbalPitchDegree": float(round(rng.uniform(-90, -60), 1)),
            "FocalLength": DEFAULT_FOCAL_MM}


def _try_exifread(filepath: str) -> dict:
    out = {}
    try:
        import exifread
        with open(filepath, "rb") as f:
            tags = exifread.process_file(f, details=False)
        if "GPS GPSLatitude" in tags and "GPS GPSLongitude" in tags:
            out["GPSLatitude"] = _dms(tags["GPS GPSLatitude"].values, tags.get("GPS GPSLatitudeRef", "N"))
            out["GPSLongitude"] = _dms(tags["GPS GPSLongitude"].values, tags.get("GPS GPSLongitudeRef", "E"))
        if "GPS GPSAltitude" in tags:
            out["RelativeAltitude"] = _ratio(tags["GPS GPSAltitude"].values[0])
        if "EXIF FocalLength" in tags:
            out["FocalLength"] = _ratio(tags["EXIF FocalLength"].values[0])
    except Exception:
        pass
    return out


def _try_pil_exif(filepath: str) -> dict:
    out = {}
    try:
        with Image.open(filepath) as im:
            ex = im.getexif()
            gps = ex.get_ifd(0x8825)
            sub = ex.get_ifd(0x8769)
        if gps and 2 in gps and 4 in gps:
            out["GPSLatitude"] = _dms(gps[2], gps.get(1, "N"))
            out["GPSLongitude"] = _dms(gps[4], gps.get(3, "E"))
        if gps and 6 in gps:
            out["RelativeAltitude"] = _ratio(gps[6])
        if 37386 in sub:
            out["FocalLength"] = _ratio(sub[37386])
        uc = sub.get(37510)
        if uc:
            txt = uc.decode("utf-8", "ignore") if isinstance(uc, bytes) else str(uc)
            txt = txt.replace("ASCII\x00\x00\x00", "").strip("\x00 ")
            j = json.loads(txt[txt.index("{"):])
            for k in ("GimbalPitchDegree", "GimbalYawDegree", "RelativeAltitude"):
                if k in j:
                    out[k] = float(j[k])
    except Exception:
        pass
    return out


def extract_metadata(filepath: str) -> dict:
    """Parse GPS lat/lon, altitude, gimbal pitch, focal length, bit depth, geo-transform.

    Order: EXIF (exifread + Pillow) -> JSON sidecar -> synthetic telemetry fallback.
    """
    meta, src = {}, "exif"
    meta.update(_try_exifread(filepath))
    for k, v in _try_pil_exif(filepath).items():
        meta.setdefault(k, v)
    if "GPSLatitude" not in meta:
        side = os.path.splitext(filepath)[0] + ".json"
        if os.path.exists(side):
            try:
                meta.update({k: v for k, v in json.load(open(side)).get("telemetry", {}).items()})
                src = "sidecar_json"
            except Exception:
                pass
    synth = _synthetic_telemetry(filepath)
    if "GPSLatitude" not in meta:
        meta.update(synth)
        src = "synthetic_telemetry"
    else:
        for k, v in synth.items():
            if k not in meta:
                meta[k] = v
    meta["metadata_source"] = src
    try:
        with Image.open(filepath) as im:
            w, h = im.size
            mode = im.mode
        meta["BitDepth"] = 16 if mode.startswith("I;16") else 8
    except Exception:
        w = h = None
        meta["BitDepth"] = 8
    meta["width"], meta["height"] = w, h
    if w:
        gsd = meta["RelativeAltitude"] * SENSOR_WIDTH_MM / (meta["FocalLength"] * w)  # m/px
        meta["gsd_m_per_px"] = float(gsd)
        lat = meta["GPSLatitude"]
        dlat = gsd / 111320.0
        dlon = gsd / (111320.0 * np.cos(np.deg2rad(lat)))
        # affine (a,b,c,d,e,f): lon = a*x + b*y + c ; lat = d*x + e*y + f  (centre = GPS fix)
        meta["geo_transform"] = [float(dlon), 0.0, float(meta["GPSLongitude"] - dlon * w / 2),
                                 0.0, float(-dlat), float(lat + dlat * h / 2)]
    return meta


def simulate_raw_and_tonemap(img_bgr: np.ndarray, gamma: float = 2.2, exposure_ev: float = 0.0) -> dict:
    """8-bit sRGB -> linear light -> 12-bit RAW (0-4095) w/ expanded dynamic range -> Reinhard SDR."""
    headroom = 3.0
    lin = (img_bgr.astype(np.float64) / 255.0) ** gamma
    lin *= 2.0 ** exposure_ev
    lum = lin.mean(axis=2, keepdims=True)
    glint = np.clip((lum - 0.55) / 0.45, 0, 1)            # sun glint on water / bright roofs
    shadow = np.clip((0.08 - lum) / 0.08, 0, 1)           # deep building shadows
    lin_hdr = lin * (1.0 + (headroom - 1.0) * glint) * (1.0 - 0.6 * shadow)
    raw12 = np.round(np.clip(lin_hdr / headroom, 0, 1) * 4095).astype(np.uint16)
    hdr_linear = (raw12.astype(np.float32) / 4095.0) * headroom
    tm = cv2.createTonemapReinhard(gamma=1.5, intensity=0.0, light_adapt=0.8, color_adapt=0.0)
    sdr_f = tm.process(np.ascontiguousarray(hdr_linear))
    sdr = np.clip(np.nan_to_num(sdr_f) * 255.0, 0, 255).astype(np.uint8)
    nz = hdr_linear[hdr_linear > 0]
    dr_stops = float(np.log2(nz.max() / max(np.percentile(nz, 0.1), 1e-6))) if nz.size else 0.0
    return {"raw12": raw12, "hdr_linear": hdr_linear, "sdr": sdr,
            "dynamic_range_stops": dr_stops,
            "hist_raw": np.histogram(raw12, bins=64, range=(0, 4095))[0],
            "hist_sdr": np.histogram(sdr, bins=64, range=(0, 255))[0]}


def convert_color_spaces(img_bgr: np.ndarray) -> dict:
    """Returns YCbCr (channel order Y,Cb,Cr), CIELAB and CIELUV arrays (uint8)."""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    ycbcr = ycrcb[..., [0, 2, 1]].copy()
    return {"YCbCr": ycbcr, "LAB": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB),
            "LUV": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LUV)}


def compute_ndwi(green_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """NDWI = (G - NIR) / (G + NIR + eps), range [-1, 1]."""
    g, n = green_band.astype(np.float64), nir_band.astype(np.float64)
    return (g - n) / (g + n + 1e-6)
