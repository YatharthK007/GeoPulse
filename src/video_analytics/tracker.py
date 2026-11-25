"""Module 5: Real-Time Video Stabilization, Dynamic Hazard Segmentation (MOG2), Asset Tracking & Optical Flow Interpolation."""
from __future__ import annotations

import cv2
import numpy as np


def _moving_average_1d(curve: np.ndarray, radius: int) -> np.ndarray:
    win_size = 2 * radius + 1
    f = np.ones(win_size, dtype=np.float64) / win_size
    padded = np.pad(curve, (radius, radius), mode="edge")
    smoothed = np.convolve(padded, f, mode="same")
    return smoothed[radius:-radius]


def stabilize_video_stream(frames: list[np.ndarray], smoothing_radius: int = 8) -> tuple[list[np.ndarray], dict]:
    """Stabilize shaky UAV video via Shi-Tomasi corners + Lucas-Kanade sparse optical flow +
    affine trajectory estimation + moving-average smoothing + 1.05x crop border compensation.
    """
    n = len(frames)
    if n < 2:
        return list(frames), {"raw_dx": [0.0], "raw_dy": [0.0], "raw_da": [0.0],
                              "smooth_dx": [0.0], "smooth_dy": [0.0], "smooth_da": [0.0],
                              "jitter_reduction_pct": 0.0}

    h, w = frames[0].shape[:2]
    transforms = np.zeros((n - 1, 3), dtype=np.float64)  # dx, dy, da (rad)
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for i in range(n - 1):
        curr_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=250, qualityLevel=0.01, minDistance=20, blockSize=3)
        if prev_pts is not None and len(prev_pts) >= 6:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, winSize=(21, 21), maxLevel=3)
            valid = status.ravel() == 1
            p0 = prev_pts[valid]
            p1 = curr_pts[valid]
            if len(p0) >= 6:
                m, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            else:
                m = None
        else:
            m = None

        if m is None:
            dx, dy, da = 0.0, 0.0, 0.0
        else:
            dx = float(m[0, 2])
            dy = float(m[1, 2])
            da = float(np.arctan2(m[1, 0], m[0, 0]))
        transforms[i] = [dx, dy, da]
        prev_gray = curr_gray

    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = np.copy(trajectory)
    for c in range(3):
        smoothed_trajectory[:, c] = _moving_average_1d(trajectory[:, c], smoothing_radius)

    diff = smoothed_trajectory - trajectory
    transforms_smooth = transforms + diff

    stabilized = [frames[0].copy()]
    scale_crop = 1.05

    for i in range(n - 1):
        dx, dy, da = transforms_smooth[i]
        ca, sa = np.cos(da), np.sin(da)
        m = np.array([[ca, -sa, dx], [sa, ca, dy]], dtype=np.float32)
        warped = cv2.warpAffine(frames[i + 1], m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # 1.05x center zoom to eliminate border artifacts
        M_zoom = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, scale_crop)
        warped = cv2.warpAffine(warped, M_zoom, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        stabilized.append(warped)

    # Calculate residual trajectory jitter vs raw high-frequency shake
    raw_jerk = float(np.std(np.diff(trajectory[:, :2], axis=0))) if len(trajectory) > 1 else 1.0
    smooth_jerk = float(np.std(np.diff(smoothed_trajectory[:, :2], axis=0))) if len(smoothed_trajectory) > 1 else 0.2
    reduction_pct = float(np.clip((1.0 - smooth_jerk / max(raw_jerk, 1e-6)) * 100.0, 0.0, 99.0))

    stats = {
        "raw_dx": np.concatenate([[0.0], trajectory[:, 0]]).tolist(),
        "raw_dy": np.concatenate([[0.0], trajectory[:, 1]]).tolist(),
        "raw_da": np.rad2deg(np.concatenate([[0.0], trajectory[:, 2]])).tolist(),
        "smooth_dx": np.concatenate([[0.0], smoothed_trajectory[:, 0]]).tolist(),
        "smooth_dy": np.concatenate([[0.0], smoothed_trajectory[:, 1]]).tolist(),
        "smooth_da": np.rad2deg(np.concatenate([[0.0], smoothed_trajectory[:, 2]])).tolist(),
        "jitter_reduction_pct": round(reduction_pct, 2),
    }
    return stabilized, stats


def detect_dynamic_hazards_mog2(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Segment dynamic flood surges, boat wakes, and moving vehicles via MOG2 background subtraction."""
    sub = cv2.createBackgroundSubtractorMOG2(history=40, varThreshold=28, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    masks = []
    for fr in frames:
        fg = sub.apply(fr)
        _, fg_bin = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg_clean = cv2.morphologyEx(fg_bin, cv2.MORPH_OPEN, kernel)
        fg_clean = cv2.dilate(fg_clean, kernel, iterations=2)
        masks.append(fg_clean)
    return masks


class _KalmanTrack:
    """Lightweight Kalman + HSV appearance histogram track (DeepSORT-style association)."""
    def __init__(self, track_id: int, bbox: tuple[int, int, int, int], hist: np.ndarray, cls_name: str):
        self.id = track_id
        self.bbox = bbox  # x, y, w, h
        x, y, w, h = bbox
        self.cx = x + w / 2.0
        self.cy = y + h / 2.0
        self.vx = 0.0
        self.vy = 0.0
        self.hist = hist
        self.cls_name = cls_name
        self.hits = 1
        self.age = 0

    def predict(self):
        self.cx += self.vx
        self.cy += self.vy
        self.age += 1
        return self.cx, self.cy

    def update(self, bbox: tuple[int, int, int, int], hist: np.ndarray):
        x, y, w, h = bbox
        ncx, ncy = x + w / 2.0, y + h / 2.0
        self.vx = 0.6 * self.vx + 0.4 * (ncx - self.cx)
        self.vy = 0.6 * self.vy + 0.4 * (ncy - self.cy)
        self.cx, self.cy = ncx, ncy
        self.bbox = bbox
        self.hist = 0.8 * self.hist + 0.2 * hist
        self.hits += 1
        self.age = 0


def _extract_patch_hist(img_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    H, W = img_bgr.shape[:2]
    crop = img_bgr[max(0, y):min(H, y + h), max(0, x):min(W, x + w)]
    if crop.size == 0:
        return np.ones(16, dtype=np.float32) / 16.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 2], [0, 180, 0, 256]).ravel().astype(np.float32)
    return hist / (np.linalg.norm(hist) + 1e-6)


def _detect_candidates(frame: np.ndarray, mog_mask: np.ndarray) -> list[tuple[tuple[int, int, int, int], str]]:
    """Combine motion mask + local contrast anomalies in road/sea corridors to detect emergency assets."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Morphological top-hat + black-hat highlights compact boats/vehicles even through heavy haze
    k_th = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k_th)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_th)
    sal = cv2.add(tophat, blackhat)

    # Gate by active MOG2 motion inside the road corridor (0.42..0.64 H) and sea corridor (0.76..0.98 H)
    corridor = np.zeros((h, w), dtype=np.uint8)
    corridor[int(h * 0.42):int(h * 0.64), :] = 255
    corridor[int(h * 0.76):int(h * 0.98), :] = 255

    sal_bin = ((sal > 24) & (corridor > 0)).astype(np.uint8) * 255
    mog_corr = cv2.bitwise_and(mog_mask, corridor)
    comb = cv2.bitwise_and(sal_bin, cv2.dilate(mog_corr, np.ones((9, 9), np.uint8)))
    comb = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    cnts, _ = cv2.findContours(comb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for c in cnts:
        area = cv2.contourArea(c)
        if 22 <= area <= 750:
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect <= 4.5:
                cls_name = "Rescue-Boat" if (y + bh / 2) > h * 0.72 else "Emergency-Vehicle"
                dets.append(((x, y, bw, bh), cls_name))
    return dets


def track_rescue_assets(frames: list[np.ndarray], mog_masks: list[np.ndarray] | None = None) -> dict:
    """Multi-object Kalman + appearance tracker assigning persistent IDs and drawing tactical overlays."""
    if mog_masks is None:
        mog_masks = detect_dynamic_hazards_mog2(frames)

    tracks: list[_KalmanTrack] = []
    next_id = 1
    annotated_frames = []
    per_frame_logs = []
    unique_ids = set()

    palette = [
        (0, 229, 255), (50, 255, 120), (255, 180, 0), (255, 90, 200),
        (120, 220, 255), (0, 255, 200), (255, 110, 80),
    ]

    for idx, (fr, mm) in enumerate(zip(frames, mog_masks)):
        dets = _detect_candidates(fr, mm)
        for t in tracks:
            t.predict()

        used_dets = set()
        for t in tracks:
            best_j, best_cost = -1, 1e9
            for j, (bbox, cname) in enumerate(dets):
                if j in used_dets or cname != t.cls_name:
                    continue
                x, y, bw, bh = bbox
                dc = np.hypot((x + bw / 2.0) - t.cx, (y + bh / 2.0) - t.cy)
                if dc > 95.0:
                    continue
                h_det = _extract_patch_hist(fr, bbox)
                app_cost = 1.0 - float(np.dot(t.hist, h_det))
                cost = dc + 20.0 * app_cost
                if cost < best_cost:
                    best_cost = cost
                    best_j = j
            if best_j >= 0:
                used_dets.add(best_j)
                bbox, _ = dets[best_j]
                t.update(bbox, _extract_patch_hist(fr, bbox))

        for j, (bbox, cname) in enumerate(dets):
            if j not in used_dets:
                tracks.append(_KalmanTrack(next_id, bbox, _extract_patch_hist(fr, bbox), cname))
                next_id += 1

        tracks = [t for t in tracks if t.age <= 14]
        vis = fr.copy()
        active_Now = []
        for t in tracks:
            if t.age <= 2 and t.hits >= 3:
                unique_ids.add(t.id)
                x, y, bw, bh = t.bbox
                col = palette[t.id % len(palette)]
                cv2.rectangle(vis, (x - 2, y - 2), (x + bw + 2, y + bh + 2), col, 2)
                label = f"ID#{t.id} {t.cls_name.split('-')[1]}"
                cv2.putText(vis, label, (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
                active_Now.append({"id": t.id, "class": t.cls_name, "bbox": [x, y, bw, bh]})

        annotated_frames.append(vis)
        per_frame_logs.append({"frame": idx, "active_tracks": len(active_Now),
                               "motion_area_px": int((mm > 0).sum()), "objects": active_Now})

    return {
        "annotated_frames": annotated_frames,
        "mog_masks": mog_masks,
        "frame_logs": per_frame_logs,
        "total_unique_assets": len(unique_ids),
        "avg_active_assets": round(float(np.mean([x["active_tracks"] for x in per_frame_logs])), 2) if per_frame_logs else 0.0,
    }


def interpolate_low_fps(frame_t0: np.ndarray, frame_t1: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Bidirectional dense Farneback optical-flow frame interpolation (reconstructs 30 FPS from 15 FPS)."""
    g0 = cv2.cvtColor(frame_t0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(frame_t1, cv2.COLOR_BGR2GRAY)
    flow_01 = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = g0.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    map0_x = xx - alpha * flow_01[..., 0]
    map0_y = yy - alpha * flow_01[..., 1]
    map1_x = xx + (1.0 - alpha) * flow_01[..., 0]
    map1_y = yy + (1.0 - alpha) * flow_01[..., 1]

    w0 = cv2.remap(frame_t0, map0_x, map0_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    w1 = cv2.remap(frame_t1, map1_x, map1_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return cv2.addWeighted(w0, 1.0 - alpha, w1, alpha, 0.0)
