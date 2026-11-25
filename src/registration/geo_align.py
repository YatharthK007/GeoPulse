"""Module 4: Cross-View Satellite-to-UAV Geo-Registration (AKAZE/SIFT + RANSAC + CNN Hybrid Fallback)."""
from __future__ import annotations

import cv2
import numpy as np

_CNN_MODEL = None


def _get_cnn_extractor():
    """Lazy-load lightweight MobileNetV3-Small intermediate feature extractor if available."""
    global _CNN_MODEL
    if _CNN_MODEL is not None:
        return _CNN_MODEL
    try:
        import torch
        import torchvision.models as models
        # Use architecture features (pretrained if cached locally, else deterministic random projection)
        try:
            m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).features[:4]
        except Exception:
            torch.manual_seed(42)
            m = models.mobilenet_v3_small(weights=None).features[:4]
        m.eval()
        _CNN_MODEL = m
    except Exception:
        _CNN_MODEL = False
    return _CNN_MODEL


def _cnn_embedding_keypoints(uav_img: np.ndarray, sat_img: np.ndarray, grid: int = 20):
    """Dense spatial CNN feature cosine-similarity matching fallback when severe rubble breaks local gradients."""
    import torch
    import torch.nn.functional as F

    m = _get_cnn_extractor()
    h, w = sat_img.shape[:2]
    if not m:
        return [], [], []
    with torch.no_grad():
        def _feat(im):
            t = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            f = m(t)  # (1, C, Hf, Wf)
            return F.normalize(f[0], dim=0)

        fu = _feat(uav_img)
        fs = _feat(sat_img)
        c, hf, wf = fu.shape
        sy, sx = h / hf, w / wf

        kps_u, kps_s, matches = [], [], []
        fu_flat = fu.view(c, -1)  # (C, N)
        fs_flat = fs.view(c, -1)  # (C, N)
        sim = torch.mm(fu_flat.t(), fs_flat)  # (N_u, N_s)
        vals, idx = torch.topk(sim, k=2, dim=1)

        for i in range(fu_flat.shape[1]):
            uy, ux = divmod(i, wf)
            kps_u.append(cv2.KeyPoint(float((ux + 0.5) * sx), float((uy + 0.5) * sy), 12.0))
        for j in range(fs_flat.shape[1]):
            vy, vx = divmod(j, wf)
            kps_s.append(cv2.KeyPoint(float((vx + 0.5) * sx), float((vy + 0.5) * sy), 12.0))

        for i in range(fu_flat.shape[1]):
            best_sim = float(vals[i, 0])
            second_sim = float(vals[i, 1])
            if best_sim > 0.72 and (best_sim - second_sim) > 0.015:
                j = int(idx[i, 0])
                matches.append(cv2.DMatch(i, j, float(1.0 - best_sim)))
    return kps_u, kps_s, matches


def register_cross_view(
    uav_img: np.ndarray,
    sat_img: np.ndarray,
    detector: str = "AKAZE",
    use_cnn_fallback: bool = True,
    ratio_thresh: float = 0.78,
    ransac_thresh: float = 5.0,
) -> dict:
    """Register post-disaster UAV frame onto pre-disaster satellite reference tile.

    Supports AKAZE, SIFT, or ORB feature detection + BFMatcher Lowe's ratio test + RANSAC homography,
    with MobileNetV3-Small CNN deep feature fallback when inliers < 12 (or when hybrid mode is enabled).
    """
    h, w = sat_img.shape[:2]
    if uav_img.shape[:2] != (h, w):
        uav_img = cv2.resize(uav_img, (w, h), interpolation=cv2.INTER_AREA)

    g_uav = cv2.cvtColor(uav_img, cv2.COLOR_BGR2GRAY)
    g_sat = cv2.cvtColor(sat_img, cv2.COLOR_BGR2GRAY)

    # CLAHE normalization helps cross-sensor (UAV vs satellite) illumination invariant matching
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_uav_eq = clahe.apply(g_uav)
    g_sat_eq = clahe.apply(g_sat)

    det_name = detector.upper()
    if det_name == "SIFT":
        det = cv2.SIFT_create(nfeatures=1500)
        norm = cv2.NORM_L2
    elif det_name == "ORB":
        det = cv2.ORB_create(nfeatures=1500)
        norm = cv2.NORM_HAMMING
    else:
        det = cv2.AKAZE_create(threshold=0.0005)
        norm = cv2.NORM_HAMMING

    kp_u, des_u = det.detectAndCompute(g_uav_eq, None)
    kp_s, des_s = det.detectAndCompute(g_sat_eq, None)

    good = []
    if des_u is not None and des_s is not None and len(kp_u) >= 4 and len(kp_s) >= 4:
        bf = cv2.BFMatcher(norm, crossCheck=False)
        raw_matches = bf.knnMatch(des_u, des_s, k=2)
        for pair in raw_matches:
            if len(pair) == 2 and pair[0].distance < ratio_thresh * pair[1].distance:
                good.append(pair[0])

    H_mat = None
    inlier_mask = None
    inlier_count = 0
    used_cnn = False

    if len(good) >= 4:
        pts_u = np.float32([kp_u[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_s = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H_mat, inlier_mask = cv2.findHomography(pts_u, pts_s, cv2.RANSAC, ransac_thresh)
        if inlier_mask is not None:
            inlier_count = int(inlier_mask.sum())

    # Hybrid CNN Deep Feature Fallback or Fusion
    if use_cnn_fallback and inlier_count < 12:
        ckp_u, ckp_s, cmatches = _cnn_embedding_keypoints(uav_img, sat_img)
        if len(cmatches) >= 6:
            offset_u, offset_s = len(kp_u), len(kp_s)
            kp_u = list(kp_u) + ckp_u
            kp_s = list(kp_s) + ckp_s
            for cm in cmatches:
                good.append(cv2.DMatch(cm.queryIdx + offset_u, cm.trainIdx + offset_s, cm.distance))
            pts_u = np.float32([kp_u[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            pts_s = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H_mat, inlier_mask = cv2.findHomography(pts_u, pts_s, cv2.RANSAC, ransac_thresh * 1.5)
            if inlier_mask is not None:
                inlier_count = int(inlier_mask.sum())
                used_cnn = True

    if H_mat is None:
        H_mat = np.eye(3, dtype=np.float64)
        inlier_mask = np.zeros((len(good), 1), dtype=np.uint8)
        inlier_count = 0
        reproj_err = 99.9
    else:
        pts_u = np.float32([kp_u[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_s = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(pts_u, H_mat)
        dists = np.linalg.norm(proj - pts_s, axis=2).ravel()
        inl = inlier_mask.ravel().astype(bool)
        reproj_err = float(np.mean(dists[inl])) if inl.any() else float(np.mean(dists))

    warped_uav = cv2.warpPerspective(uav_img, H_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Structural change heatmap (|Gray_post - Gray_pre| weighted by gradient difference)
    g_warp = cv2.cvtColor(warped_uav, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = np.abs(g_warp - g_sat.astype(np.float32))
    diff = cv2.GaussianBlur(diff, (0, 0), 2.5)
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    change_heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_INFERNO)

    # Draw green inlier correspondence lines and red outliers
    vis_matches = good[:120]
    vis_mask = inlier_mask.ravel().tolist()[:120] if inlier_mask is not None else None
    match_vis = cv2.drawMatches(
        uav_img,
        kp_u,
        sat_img,
        kp_s,
        vis_matches,
        None,
        matchColor=(0, 255, 100),
        singlePointColor=(255, 80, 80),
        matchesMask=vis_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    return {
        "warped_uav": warped_uav,
        "match_vis": match_vis,
        "homography": H_mat,
        "total_matches": len(good),
        "inlier_count": inlier_count,
        "inlier_ratio": float(inlier_count / max(len(good), 1)),
        "reproj_error": round(reproj_err, 3),
        "change_heatmap": change_heatmap,
        "change_score_map": (diff_norm.astype(np.float32) / 255.0),
        "used_cnn_hybrid": used_cnn,
        "detector": det_name,
    }
