"""Generate synthetic aerial disaster scenes, NIR bands, GT masks, satellite refs and a UAV video.

Run:  python scripts/generate_demo_assets.py
Mask classes: 0 Bldg-NoDamage 1 Bldg-Minor 2 Bldg-Major 3 Bldg-Destroyed 4 Road-Clear
              5 Road-Blocked 6 Water 7 Tree 8 Vehicle 9 Pool, 255 = background/ground.
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import piexif
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SAMPLES = os.path.join(ROOT, "data", "samples")
SAT = os.path.join(ROOT, "data", "satellite_pairs")
W, H = 768, 512

TELEMETRY = {  # lat, lon, rel. altitude (m), gimbal pitch (deg), focal (mm)
    "hurricane_coast":   (29.2870, -94.8120, 95.0, -88.0, 8.8),
    "urban_earthquake":  (37.5850, 36.9370, 82.0, -90.0, 8.8),
    "flooded_highway":   (29.7580, -95.3660, 110.0, -75.0, 8.8),
    "landslide_village": (11.6050, 76.1310, 120.0, -80.0, 8.8),
}


# ------------------------------------------------------------------ helpers
def fractal(h, w, rng, octaves=((64, 1.0), (16, 0.5), (4, 0.25))):
    out = np.zeros((h, w), np.float32)
    for s, a in octaves:
        n = rng.random((h // s + 2, w // s + 2)).astype(np.float32)
        out += a * cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)[:h, :w]
    out -= out.min()
    return out / (out.max() + 1e-6)


def blobfield(h, w, rng, scale=90):
    n = rng.random((h // 8 + 2, w // 8 + 2)).astype(np.float32)
    n = cv2.GaussianBlur(cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC), (0, 0), scale / 8)
    n -= n.min()
    return n / (n.max() + 1e-6)


class Canvas:
    def __init__(self, h, w, rng):
        self.h, self.w, self.rng = h, w, rng
        self.img = np.zeros((h, w, 3), np.float32)       # BGR 0-255
        self.nir = np.full((h, w), 0.4, np.float32)
        self.mask = np.full((h, w), 255, np.uint8)
        self.occ = np.zeros((h, w), np.uint8)             # occupancy for placement

    def poly(self, pts, color, cls=None, nir=None, jitter=8):
        p = np.round(np.asarray(pts, np.float32)).astype(np.int32)
        m = np.zeros((self.h, self.w), np.uint8)
        cv2.fillPoly(m, [p], 1)
        self.fill(m.astype(bool), color, cls, nir, jitter)

    def fill(self, m, color, cls=None, nir=None, jitter=8):
        tex = (self.rng.random((self.h, self.w)).astype(np.float32) - 0.5) * 2 * jitter
        for c in range(3):
            self.img[..., c][m] = color[c] + tex[m]
        if cls is not None:
            self.mask[m] = cls
        if nir is not None:
            self.nir[m] = nir

    def blend(self, m, color, alpha):
        a = (np.asarray(m, np.float32) * alpha)[..., None]
        self.img = self.img * (1 - a) + np.array(color, np.float32) * a


def rot_rect(cx, cy, w, h, ang):
    c, s = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
    pts = np.array([[-w, -h], [w, -h], [w, h], [-w, h]], np.float32) / 2
    return np.stack([cx + pts[:, 0] * c - pts[:, 1] * s, cy + pts[:, 0] * s + pts[:, 1] * c], 1)


def road_mask(cv, pts, width):
    m = np.zeros((cv.h, cv.w), np.uint8)
    cv2.polylines(m, [np.round(np.asarray(pts)).astype(np.int32)], False, 1, int(width), cv2.LINE_8)
    return m.astype(bool)


def draw_road(cv, pts, width, color=(85, 85, 90), lanes=True):
    m = road_mask(cv, pts, width)
    cv.fill(m, color, 4, 0.30, 6)
    cv.occ[cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0] = 1
    if lanes:  # dashed centre line
        pts = np.asarray(pts, np.float32)
        for a, b in zip(pts[:-1], pts[1:]):
            L = np.linalg.norm(b - a)
            for t in np.arange(0, L, 22):
                p0, p1 = a + (b - a) * t / L, a + (b - a) * min(t + 11, L) / L
                cv2.line(cv.img, tuple(map(int, p0)), tuple(map(int, p1)), (210, 215, 220), 1, cv2.LINE_AA)
    return m


def road_point(pts, t):
    pts = np.asarray(pts, np.float32)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    d = t * seg.sum()
    for i, s in enumerate(seg):
        if d <= s:
            v = (pts[i + 1] - pts[i]) / s
            return pts[i] + v * d, np.rad2deg(np.arctan2(v[1], v[0]))
        d -= s
    return pts[-1], 0.0


def rubble(cv, cx, cy, r, cls, n=60, tint=None):
    for _ in range(n):
        a, d = cv.rng.uniform(0, 2 * np.pi), r * np.sqrt(cv.rng.random())
        x, y = cx + d * np.cos(a), cy + d * np.sin(a) * 0.8
        s = cv.rng.uniform(2, 6)
        col = tint or [(70, 75, 85), (60, 90, 140), (150, 155, 160), (40, 45, 55), (90, 110, 170)][cv.rng.integers(5)]
        cv.poly(rot_rect(x, y, s * 1.6, s, cv.rng.uniform(0, 180)), col, cls, 0.35, 12)


def crack(cv, pts, color=(25, 25, 30), th=1):
    cv2.polylines(cv.img, [np.round(pts).astype(np.int32)], False, color, th, cv2.LINE_AA)


def jag(cv, p0, p1, n=7, amp=5):
    ts = np.linspace(0, 1, n)[:, None]
    pts = p0 * (1 - ts) + p1 * ts
    pts[1:-1] += cv.rng.normal(0, amp, (n - 2, 2))
    return pts


def draw_building(cv, b, damaged=True):
    cx, cy, bw, bh, ang, col, lvl = b
    if not damaged:
        lvl = 0
    cs, sn = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
    sh = rot_rect(cx + 6, cy + 5, bw, bh, ang)  # shadow
    m = np.zeros((cv.h, cv.w), np.uint8)
    cv2.fillPoly(m, [np.round(sh).astype(np.int32)], 1)
    cv.blend(cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2), (10, 10, 12), 0.4)
    rect = rot_rect(cx, cy, bw, bh, ang)
    if lvl == 3:
        cv.poly(rect * 0.0 + rect, (95, 100, 105), 3, 0.35, 18)
        rubble(cv, cx, cy, max(bw, bh) * 0.62, 3, n=int(bw * bh / 22))
        return
    cv.poly(rect, col, lvl, 0.45, 5)
    # roof ridge + HVAC for realism
    a, b2 = rot_rect(cx, cy, bw, 0, ang)[:2]
    cv2.line(cv.img, tuple(map(int, a)), tuple(map(int, b2)), tuple(int(c * 0.75) for c in col), 1, cv2.LINE_AA)
    if lvl >= 1:
        for _ in range(2 if lvl == 1 else 3):
            p0 = np.array([cx + cv.rng.uniform(-bw / 2, bw / 2), cy + cv.rng.uniform(-bh / 2, bh / 2)])
            crack(cv, jag(cv, p0, p0 + cv.rng.normal(0, bw / 3, 2)))
    if lvl == 2:  # partial roof collapse
        sub = rot_rect(cx + cs * bw * 0.2, cy + sn * bw * 0.2, bw * 0.55, bh * 0.7, ang)
        cv.poly(sub, (35, 38, 45), 2, 0.3, 14)
        rubble(cv, cx + cs * bw * 0.2, cy + sn * bw * 0.2, bw * 0.3, 2, n=int(bw * bh / 60))


def draw_tree(cv, x, y, r, nir=0.85):
    m = np.zeros((cv.h, cv.w), np.uint8)
    cv2.circle(m, (int(x), int(y)), int(r), 1, -1)
    base = np.array([35 + cv.rng.integers(-8, 8), 95 + cv.rng.integers(-20, 25), 40 + cv.rng.integers(-8, 8)])
    cv.fill(m.astype(bool), base, 7, nir, 14)
    cv2.circle(cv.img, (int(x - r * 0.3), int(y - r * 0.3)), max(int(r * 0.5), 1),
               tuple(int(v) for v in np.clip(base * 1.3, 0, 255)), -1, cv2.LINE_AA)


def draw_vehicle(cv, x, y, ang, col=None):
    col = col or [(40, 40, 200), (200, 200, 200), (30, 30, 30), (200, 120, 30), (40, 170, 220)][cv.rng.integers(5)]
    cv.poly(rot_rect(x, y, 13, 6, ang), col, 8, 0.5, 3)
    cv.poly(rot_rect(x, y, 5, 4.5, ang), (60, 50, 40), 8, 0.5, 2)


def place_buildings(cv, n, region, size=(24, 52), levels=(0, 0, 1, 2, 3), margin=6, angle=None, tries=400):
    out = []
    pal = [(70, 95, 190), (160, 165, 170), (190, 190, 195), (90, 120, 200), (135, 140, 150), (175, 130, 90), (110, 150, 210)]
    for _ in range(tries):
        if len(out) >= n:
            break
        cx, cy = cv.rng.uniform(region[0], region[2]), cv.rng.uniform(region[1], region[3])
        bw, bh = cv.rng.uniform(*size), cv.rng.uniform(size[0], size[1] * 0.8)
        ang = angle if angle is not None else cv.rng.choice([0, 0, 90, cv.rng.uniform(0, 30)])
        rect = rot_rect(cx, cy, bw + margin, bh + margin, ang)
        t = np.zeros((cv.h, cv.w), np.uint8)
        cv2.fillPoly(t, [np.round(rect).astype(np.int32)], 1)
        if (t & cv.occ).any() or rect.min() < 0 or rect[:, 0].max() >= cv.w or rect[:, 1].max() >= cv.h:
            continue
        cv.occ |= t
        out.append((cx, cy, bw, bh, ang, pal[cv.rng.integers(len(pal))], int(cv.rng.choice(levels))))
    return out


def finish(cv, tint=(1.0, 1.0, 1.0), contrast=1.0, blur=0.6):
    img = cv.img * np.array(tint, np.float32)
    img = (img - 128) * contrast + 128
    img = cv2.GaussianBlur(img, (0, 0), blur)
    img += cv.rng.normal(0, 1.5, img.shape)
    nir = cv2.GaussianBlur(cv.nir, (0, 0), blur) + cv.rng.normal(0, 0.01, cv.nir.shape)
    return np.clip(img, 0, 255).astype(np.uint8), np.clip(nir * 255, 0, 255).astype(np.uint8), cv.mask.copy()


def ground(cv, base, var=25, mud=None):
    f = fractal(cv.h, cv.w, cv.rng)
    for c in range(3):
        cv.img[..., c] = base[c] + (f - 0.5) * var * 2 + (cv.rng.random((cv.h, cv.w)) - 0.5) * 8
    cv.nir[:] = 0.45 + 0.15 * (f - 0.5)


def flood(cv, field_thr, color=(120, 105, 70), region=None, pools=False):
    """Paint flood-water where blob field exceeds a threshold (over roads, under buildings)."""
    st = cv.rng.bit_generator.state
    f = blobfield(cv.h, cv.w, cv.rng, 110)
    if region is not None:
        f = f + region
    m = f > field_thr
    ripple = fractal(cv.h, cv.w, cv.rng, ((24, 1.0), (6, 0.6)))
    cv.fill(m, color, 6, 0.06, 6)
    for c in range(3):
        cv.img[..., c][m] += (ripple[m] - 0.5) * 22
    glint = (ripple > 0.85) & m
    cv.img[glint] += 40
    cv.occ[m.astype(np.uint8) > 0] = 1
    cv.rng.bit_generator.state = st
    return m


def sea(cv, y0, amp=12):
    xs = np.arange(cv.w)
    edge = y0 + amp * np.sin(xs / 55.0) + amp * 0.5 * np.sin(xs / 17.0)
    yy = np.arange(cv.h)[:, None]
    m = yy > edge[None, :]
    wave = fractal(cv.h, cv.w, cv.rng, ((32, 1.0), (8, 0.7)))
    depth = np.clip((yy - edge[None, :]) / (cv.h - y0), 0, 1)
    for c, (a, b) in enumerate(zip((150, 130, 70), (110, 75, 25))):
        cv.img[..., c][m] = (a * (1 - depth) + b * depth)[m] + (wave[m] - 0.5) * 30
    cv.mask[m] = 6
    cv.nir[m] = 0.04
    foam = (np.abs(yy - edge[None, :]) < 3)
    cv.img[foam & m] += 80
    cv.occ[(yy > edge[None, :] - 8)] = 1
    return m


# ------------------------------------------------------------------ scenes
def build_scene(name: str, seed: int, disaster: bool = True, h: int = H, w: int = W, return_layout=False):
    # layout random stream is independent of `disaster`, so pre/post tiles share structure
    rng = np.random.default_rng(seed)
    cv = Canvas(h, w, rng)
    layout = {}
    if name == "hurricane_coast":
        ground(cv, (95, 140, 120), 18)
        sand = np.arange(h)[:, None] > h * 0.62
        cv.img[sand.repeat(w, 1)] = cv.img[sand.repeat(w, 1)] * 0.4 + np.array([150, 190, 215]) * 0.6
        layout["road"] = [(0, h * 0.50), (w * 0.35, h * 0.54), (w * 0.7, h * 0.50), (w, h * 0.55)]
        draw_road(cv, layout["road"], 16)
        draw_road(cv, [(w * 0.45, 0), (w * 0.43, h * 0.54), (w * 0.47, h * 0.7)], 12)
        if disaster:
            flood(cv, 0.98, (125, 110, 75), region=(np.arange(h)[:, None] / h - 0.4).clip(0) * 1.2)
        sea(cv, h * 0.76)
        bs = place_buildings(cv, 20, (20, 20, w - 20, h * 0.68), levels=(0, 1, 2, 3, 3, 2))
        trees = [(rng.uniform(0, w), rng.uniform(0, h * 0.7), rng.uniform(6, 11)) for _ in range(30)]
        tint = (0.93, 0.95, 0.95) if disaster else (1.02, 1.02, 1.0)
    elif name == "urban_earthquake":
        ground(cv, (120, 130, 135), 14)
        layout["hroads"] = [h * 0.22, h * 0.52, h * 0.82]
        layout["vroads"] = [w * 0.15, w * 0.40, w * 0.65, w * 0.90]
        rd = []
        for y in layout["hroads"]:
            rd.append([(0, y), (w, y + rng.uniform(-6, 6))])
        for x in layout["vroads"]:
            rd.append([(x, 0), (x + rng.uniform(-6, 6), h)])
        for p in rd:
            draw_road(cv, p, 14)
        layout["road"] = rd
        bs = []
        for x0, x1 in zip([0] + layout["vroads"], layout["vroads"] + [w]):
            for y0, y1 in zip([0] + layout["hroads"], layout["hroads"] + [h]):
                bs += place_buildings(cv, 4, (x0 + 22, y0 + 22, x1 - 22, y1 - 22), size=(20, 40), angle=0,
                                      levels=(0, 1, 2, 3, 3, 2), margin=5, tries=60)
        trees = [(rng.uniform(0, w), rng.uniform(0, h), rng.uniform(4, 7)) for _ in range(14)]
        tint = (0.97, 0.97, 0.97)
    elif name == "flooded_highway":
        ground(cv, (85, 135, 95), 20)
        layout["road"] = [(0, h * 0.28), (w * 0.4, h * 0.45), (w * 0.75, h * 0.62), (w, h * 0.78)]
        draw_road(cv, layout["road"], 34, (80, 80, 85))
        for off in (-8, 8):
            cv2.polylines(cv.img, [np.round(np.array(layout["road"]) + [0, off]).astype(np.int32)],
                          False, (220, 220, 225), 1, cv2.LINE_AA)
        side = [(rng.uniform(0, w), rng.uniform(0, h)) for _ in range(0)]
        if disaster:
            reg = np.exp(-((np.arange(h)[:, None] - h * 0.55) ** 2) / (2 * (h * 0.28) ** 2)) * 0.5
            flood(cv, 1.0, (130, 115, 80), region=reg)
        bs = place_buildings(cv, 16, (20, 20, w - 20, h - 20), size=(20, 40), levels=(0, 0, 1, 2, 3), margin=10)
        pools = place_buildings(cv, 3, (20, 20, w - 20, h - 20), size=(10, 14), levels=(0,), margin=4)
        for p in pools:
            cv.poly(rot_rect(p[0], p[1], p[2], p[3] * 0.7, 0), (190, 170, 40), 9, 0.08, 3)
        trees = [(rng.uniform(0, w), rng.uniform(0, h), rng.uniform(6, 10)) for _ in range(26)]
        tint = (0.96, 0.98, 0.98)
    elif name == "landslide_village":
        ground(cv, (70, 125, 90), 22)
        hill = (np.sin(np.linspace(0, 6, w))[None, :] + np.cos(np.linspace(0, 4, h))[:, None]) * 10
        cv.img += hill[..., None]
        layout["road"] = [(0, h * 0.78), (w * 0.3, h * 0.7), (w * 0.55, h * 0.62), (w * 0.8, h * 0.4), (w, h * 0.25)]
        draw_road(cv, layout["road"], 14, (95, 95, 100))
        scar = np.array([(w * 0.28, 0), (w * 0.45, 0), (w * 0.72, h * 0.55), (w * 0.62, h * 0.85), (w * 0.52, h * 0.6)], np.float32)
        scar[:, 0] += 0
        bs = place_buildings(cv, 18, (20, h * 0.35, w - 20, h - 25), size=(18, 34), levels=(0, 0, 1, 2), margin=8, tries=300)
        if disaster:  # buildings inside scar are destroyed
            sm = np.zeros((h, w), np.uint8)
            cv2.fillPoly(sm, [np.round(scar).astype(np.int32)], 1)
            bs = [b if not sm[int(b[1]), int(b[0])] else b[:6] + (3,) for b in bs]
        trees = [(rng.uniform(0, w), rng.uniform(0, h), rng.uniform(5, 9)) for _ in range(60)]
        layout["scar"] = scar
        tint = (0.97, 1.0, 0.98)
        if disaster:
            sm = np.zeros((h, w), np.float32)
            cv2.fillPoly(sm, [np.round(scar).astype(np.int32)], 1)
            sm = cv2.GaussianBlur(sm, (0, 0), 6)
            mud = fractal(h, w, np.random.default_rng(5), ((16, 1.0), (4, 0.5)))
            mudc = np.stack([55 + 30 * mud, 85 + 35 * mud, 120 + 40 * mud], -1)
            cv.img = cv.img * (1 - sm[..., None]) + mudc * sm[..., None]
            cv.nir = cv.nir * (1 - sm) + 0.3 * sm
    else:
        raise ValueError(name)

    for b in bs:
        draw_building(cv, b, damaged=disaster)
    for (x, y, r) in trees:
        if not cv.occ[int(min(y, h - 1)), int(min(x, w - 1))] and cv.mask[int(min(y, h - 1)), int(min(x, w - 1))] in (255,):
            draw_tree(cv, x, y, r)
    # blocked road sections with rubble + vehicles
    roads = layout["road"] if isinstance(layout["road"][0][0], (tuple, list, np.ndarray)) else [layout["road"]]
    if disaster:
        n_block = {"hurricane_coast": 3, "urban_earthquake": 5, "flooded_highway": 1, "landslide_village": 2}[name]
        for _ in range(n_block):
            rp = roads[rng.integers(len(roads))]
            p, _a = road_point(rp, rng.uniform(0.15, 0.85))
            rubble(cv, p[0], p[1], 15, 5, n=55)
        if name == "landslide_village":  # scar covers road
            p, _a = road_point(layout["road"], 0.68)
            rubble(cv, p[0], p[1], 20, 5, n=90, tint=(60, 90, 130))
        if name == "urban_earthquake":  # road cracks
            for _ in range(8):
                rp = roads[rng.integers(len(roads))]
                p, _a = road_point(rp, rng.uniform(0.05, 0.95))
                crack(cv, jag(cv, p, p + rng.normal(0, 20, 2), 8, 3), (20, 20, 25), 1)
    rng = np.random.default_rng(seed + 77)
    nveh = {"hurricane_coast": 5, "urban_earthquake": 7, "flooded_highway": 12, "landslide_village": 4}[name]
    for i in range(nveh):
        rp = roads[rng.integers(len(roads))]
        p, a = road_point(rp, rng.uniform(0.05, 0.95))
        p = p + np.array([-np.sin(np.deg2rad(a)), np.cos(np.deg2rad(a))]) * rng.choice([-4, 4])
        draw_vehicle(cv, p[0], p[1], a)
    if name == "hurricane_coast":
        for _ in range(3):  # boats in the sea
            draw_vehicle(cv, rng.uniform(80, w - 80), rng.uniform(h * 0.85, h * 0.95), rng.uniform(-20, 20), (230, 230, 235))
    if name == "urban_earthquake" and disaster:
        pass
    rgb_out = finish(cv, tint=tint, contrast=1.05 if disaster else 1.0)
    return (*rgb_out, layout) if return_layout else rgb_out


def save_scene(path_png, bgr, nir, mask, tel, extra=None):
    lat, lon, alt, pitch, focal = tel

    def dms(v):
        v = abs(v)
        d, m = int(v), int((abs(v) - int(v)) * 60)
        s = (v - d - m / 60) * 3600
        return ((d, 1), (m, 1), (int(s * 1000), 1000))
    exif = {"0th": {piexif.ImageIFD.Make: b"GeoPulseSim", piexif.ImageIFD.Model: b"SynthUAV-1in"},
            "Exif": {piexif.ExifIFD.FocalLength: (int(focal * 10), 10),
                     piexif.ExifIFD.UserComment: b"ASCII\x00\x00\x00" + json.dumps(
                         {"GimbalPitchDegree": pitch, "GimbalYawDegree": 12.5, "RelativeAltitude": alt}).encode()},
            "GPS": {piexif.GPSIFD.GPSLatitudeRef: b"N" if lat >= 0 else b"S", piexif.GPSIFD.GPSLatitude: dms(lat),
                    piexif.GPSIFD.GPSLongitudeRef: b"E" if lon >= 0 else b"W", piexif.GPSIFD.GPSLongitude: dms(lon),
                    piexif.GPSIFD.GPSAltitude: (int(alt * 10), 10), piexif.GPSIFD.GPSAltitudeRef: 0},
            "1st": {}, "thumbnail": None}
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(path_png, exif=piexif.dump(exif))
    base = os.path.splitext(path_png)[0]
    cv2.imwrite(base + "_nir.png", nir)
    cv2.imwrite(base + "_mask.png", mask)
    side = {"telemetry": {"GPSLatitude": lat, "GPSLongitude": lon, "RelativeAltitude": alt,
                          "GimbalPitchDegree": pitch, "FocalLength": focal}, "classes": [
        "Building-No-Damage", "Building-Minor", "Building-Major", "Building-Destroyed", "Road-Clear",
        "Road-Blocked", "Water", "Tree", "Vehicle", "Pool"], "background_value": 255}
    if extra:
        side.update(extra)
    json.dump(side, open(base + ".json", "w"), indent=1)


def make_satellite_tile(name, seed, idx):
    """Pre-disaster nadir reference: intact layout, rotated/scaled, softer, colour-shifted."""
    bgr, nir, _ = build_scene(name, seed, disaster=False)
    rng = np.random.default_rng(100 + idx)
    ang, sc = float(rng.uniform(-9, 9)), float(rng.uniform(0.88, 1.12))
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, sc)
    sat = cv2.warpAffine(bgr, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    sat = cv2.GaussianBlur(sat, (0, 0), 1.1)
    sat = np.clip(sat.astype(np.float32) * np.array([1.04, 1.0, 0.95]) + 6, 0, 255).astype(np.uint8)
    sat_nir = cv2.warpAffine(nir, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    p = os.path.join(SAT, f"{name}_sat.png")
    cv2.imwrite(p, sat)
    cv2.imwrite(os.path.join(SAT, f"{name}_sat_nir.png"), sat_nir)
    # H maps UAV(post) -> satellite(pre) coordinates
    A = np.vstack([M, [0, 0, 1]])
    json.dump({"rotation_deg": ang, "scale": sc, "H_uav_to_sat": A.tolist()},
              open(os.path.join(SAT, f"{name}_sat.json"), "w"), indent=1)
    return ang, sc


# ------------------------------------------------------------------ video
def make_video(path, seed=11, n=60, fps=20):
    from src.degradation.simulator import apply_atmospheric_haze, apply_sensor_noise
    BW, BH = 1000, 700
    bgr, nir, mask, layout = build_scene("hurricane_coast", 0, True, BH, BW, return_layout=True)
    bg = bgr.copy()
    rng = np.random.default_rng(seed)
    road = layout["road"]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    jitter = []
    walk = np.zeros(3)
    vehicles = [(rng.uniform(0, 1), rng.choice([-1, 1]) * rng.uniform(0.004, 0.009), rng.integers(5)) for _ in range(5)]
    boats = [(rng.uniform(120, 500), BH * rng.uniform(0.88, 0.96), rng.uniform(1.5, 3.0) * rng.choice([-1, 1])) for _ in range(3)]
    cols = [(40, 40, 200), (230, 230, 230), (30, 30, 30), (200, 120, 30), (40, 170, 220)]
    for f in range(n):
        fr = bg.copy()
        for (t0, sp, ci) in vehicles:
            t = (t0 + sp * f) % 1.0
            p, a = road_point(road, t)
            if sp < 0:
                a += 180
            pts = rot_rect(p[0], p[1], 14, 7, a)
            cv2.fillPoly(fr, [np.round(pts).astype(np.int32)], cols[ci], cv2.LINE_AA)
        for (x0, y, sp) in boats:
            x = (x0 + sp * f * 3) % (BW - 60) + 30
            cv2.fillPoly(fr, [np.round(np.array([[x - 12, y - 4], [x + 12, y - 4], [x + 16, y], [x + 12, y + 4], [x - 12, y + 4]])).astype(np.int32)],
                         (235, 235, 240), cv2.LINE_AA)
            cv2.line(fr, (int(x - 24 * np.sign(sp)), int(y)), (int(x - 12 * np.sign(sp)), int(y)), (255, 255, 255), 1, cv2.LINE_AA)  # wake
        walk += rng.normal(0, [3.0, 3.0, 0.25]) - 0.15 * walk * np.array([1, 1, 4])
        dx, dy, dth = walk + np.array([12 * np.sin(f / 6), 8 * np.cos(f / 5), 0.8 * np.sin(f / 7)])
        jitter.append([float(dx), float(dy), float(dth)])
        M = cv2.getRotationMatrix2D((BW / 2, BH / 2), dth, 1.0)
        M[0, 2] += dx + (W - BW) / 2
        M[1, 2] += dy + (H - BH) / 2
        crop = cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if f % 4 == 0 or True:  # rotor-vibration blur along jitter direction
            L = max(int(np.hypot(*(np.array(jitter[-1][:2]) - (jitter[-2][:2] if len(jitter) > 1 else [0, 0])))), 1)
            k = np.zeros((L | 1, L | 1), np.float32)
            k[L // 2, :] = 1.0 / k.shape[1]
            crop = cv2.filter2D(crop, -1, k)
        crop = apply_atmospheric_haze(crop, 0.9, 0.82)
        crop = apply_sensor_noise(crop, 0.006, 4.0, seed=f)
        vw.write(crop)
    vw.release()
    json.dump({"jitter_dx_dy_dtheta": jitter, "fps": fps, "frames": n}, open(os.path.splitext(path)[0] + "_jitter.json", "w"))


def main():
    os.makedirs(SAMPLES, exist_ok=True)
    os.makedirs(SAT, exist_ok=True)
    for i, name in enumerate(TELEMETRY):
        bgr, nir, mask = build_scene(name, seed=i + 1, disaster=True)
        save_scene(os.path.join(SAMPLES, f"{name}.png"), bgr, nir, mask, TELEMETRY[name])
        ang, sc = make_satellite_tile(name, i + 1, i)
        cls = {int(c): int((mask == c).sum()) for c in np.unique(mask)}
        print(f"{name}: pixels/class={cls} | sat rot={ang:.1f} scale={sc:.2f}")
    make_video(os.path.join(SAMPLES, "uav_flyover.mp4"))
    cap = cv2.VideoCapture(os.path.join(SAMPLES, "uav_flyover.mp4"))
    print("video frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps:", cap.get(cv2.CAP_PROP_FPS))


if __name__ == "__main__":
    main()
