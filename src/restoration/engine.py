"""Module 2: spatial & frequency-domain restoration engine."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle, denoise_wavelet
from skimage.transform import radon

from src.degradation.simulator import make_motion_psf

DENOISERS = ("adaptive_median", "bilateral", "nlm", "anisotropic", "tv", "wavelet")


# ---------------------------------------------------------------- blur estimation
def _patch_spectrum(gray: np.ndarray, n: int = 256, max_patches: int = 12) -> np.ndarray:
    """Hann-windowed, patch-averaged log-magnitude spectrum (fftshifted)."""
    h, w = gray.shape
    if h < n or w < n:
        gray = cv2.resize(gray, (max(w, n), max(h, n)))
        h, w = gray.shape
    win = np.outer(np.hanning(n), np.hanning(n))
    ys = np.linspace(0, h - n, 4).astype(int)
    xs = np.linspace(0, w - n, 4).astype(int)
    acc, cnt = np.zeros((n, n)), 0
    for y in ys:
        for x in xs:
            p = gray[y:y + n, x:x + n]
            p = (p - p.mean()) * win
            acc += np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(p))))
            cnt += 1
            if cnt >= max_patches:
                break
        if cnt >= max_patches:
            break
    return acc / cnt


def estimate_blur_fft(gray_y: np.ndarray) -> dict:
    """Estimate linear motion-blur angle / length from the log-magnitude 2D-FFT spectrum.

    Radon projections of the detrended spectrum find the dominant fringe orientation
    (fringes are perpendicular to the blur direction); the first spectral null along the
    blur direction gives L = 1 / f_null.
    """
    g = gray_y.astype(np.float64)
    n = 256
    S = _patch_spectrum(g, n)
    # detrend radial 1/f falloff
    yy, xx = np.indices(S.shape)
    r = np.hypot(yy - n // 2, xx - n // 2).astype(int)
    radial = np.bincount(r.ravel(), S.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    R = S - radial[r]
    R = cv2.GaussianBlur(R, (0, 0), 1.0)
    rr = np.hypot(yy - n // 2, xx - n // 2)
    R[(rr < 6) | (rr > 110)] = 0
    # fringe-orientation search via Radon (variance of projections)
    thetas = np.arange(0, 180, 1.0)
    sino = radon(R, theta=thetas, circle=True)
    score = sino.var(axis=0)
    fringe_dir_idx = float(thetas[int(np.argmax(score))])
    # skimage radon: projection angle t integrates along lines whose direction is t+90 deg.
    # Peak variance => lines of constant spectrum (fringes) run along (t+90); blur is perpendicular => t.
    angle = fringe_dir_idx % 180.0
    # spectrum x axis = +u, y axis (image rows) = +v, rows go down; convert to counter-clockwise
    est_angle = (angle if angle <= 90 else angle - 180)
    est_angle = float(est_angle % 180.0)
    # length: profile along blur direction in frequency plane
    th = np.deg2rad(est_angle)
    ts = np.arange(0.5, 0.5 * n * 0.85, 0.25)  # bins
    px = n // 2 + ts * np.cos(th) * 1.0
    py = n // 2 - ts * np.sin(th) * 1.0
    Rd = S - radial[r]  # detrended, un-zeroed spectrum for the profile
    prof = cv2.remap(Rd.astype(np.float32), px.astype(np.float32).reshape(1, -1),
                     py.astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR).ravel()
    prof = cv2.GaussianBlur(prof.reshape(1, -1), (0, 0), 3.0).ravel()
    fcyc = ts / n  # cycles/pixel
    from scipy.signal import find_peaks
    mins, _ = find_peaks(-prof, prominence=0.12 * (prof.max() - prof.min() + 1e-9))
    mins = [m for m in mins if fcyc[m] > 1.0 / 80.0]
    est_len = int(round(np.clip(1.0 / fcyc[mins[0]], 3, 80))) if mins else 1
    vis = cv2.normalize(R - R.min(), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis = cv2.applyColorMap(cv2.normalize(S, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
    return {"spectrum_vis": vis, "est_angle": est_angle, "est_length": est_len}


# ---------------------------------------------------------------- deblurring
def wiener_deblur_freq(channel: np.ndarray, psf: np.ndarray, nsr: float = 0.015) -> np.ndarray:
    """Frequency-domain Wiener: X = H* / (|H|^2 + NSR) * Y. Works on one float/uint8 channel."""
    ch = channel.astype(np.float64)
    ph, pw = psf.shape
    pad = max(ph, pw)
    chp = np.pad(ch, pad, mode="reflect")
    H, W = chp.shape
    k = np.zeros((H, W))
    k[:ph, :pw] = psf
    k = np.roll(k, (-(ph // 2), -(pw // 2)), axis=(0, 1))  # == ifftshift centring
    Hf = np.fft.fft2(k)
    G = np.conj(Hf) / (np.abs(Hf) ** 2 + nsr)
    x = np.real(np.fft.ifft2(G * np.fft.fft2(chp)))
    return x[pad:-pad, pad:-pad]


def wiener_deblur_color(img_bgr: np.ndarray, psf: np.ndarray, nsr: float = 0.015) -> np.ndarray:
    """Deblur luminance (Y of YCrCb) only, preserving chroma."""
    ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float64)
    ycc[..., 0] = wiener_deblur_freq(ycc[..., 0], psf, nsr)
    out = np.clip(ycc, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)


# ---------------------------------------------------------------- dehazing
def _box(a, r):
    return cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Gray-guide guided filter; uses cv2.ximgproc if present, else box-filter implementation."""
    guide, src = guide.astype(np.float32), src.astype(np.float32)
    if hasattr(cv2, "ximgproc"):
        return cv2.ximgproc.guidedFilter(guide, src, radius, eps)
    mi, mp = _box(guide, radius), _box(src, radius)
    cov = _box(guide * src, radius) - mi * mp
    var = _box(guide * guide, radius) - mi * mi
    a = cov / (var + eps)
    b = mp - a * mi
    return _box(a, radius) * guide + _box(b, radius)


def dehaze_guided_filter(img_bgr: np.ndarray, omega: float = 0.92, radius: int = 40,
                         eps: float = 1e-3, patch: int = 15) -> np.ndarray:
    """Dark Channel Prior + guided-filter-refined transmission."""
    I = img_bgr.astype(np.float32) / 255.0
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    dark = cv2.erode(I.min(axis=2), ker)
    n = max(int(dark.size * 0.001), 1)
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    cand = I.reshape(-1, 3)[idx]
    A = cand[np.argmax(cand.sum(axis=1))]
    A = np.maximum(A, 0.3)
    t_raw = 1.0 - omega * cv2.erode((I / A).min(axis=2), ker)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t = np.clip(guided_filter(gray, t_raw, radius, eps), 0.1, 1.0)[..., None]
    J = (I - A) / np.maximum(t, 0.1) + A
    return np.clip(J * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- denoisers
def _adaptive_median_ch(ch: np.ndarray, smax: int = 7) -> np.ndarray:
    out = ch.copy()
    done = np.zeros(ch.shape, bool)
    for w in range(3, smax + 1, 2):
        k = np.ones((w, w), np.uint8)
        med = cv2.medianBlur(ch, w)
        mn, mx = cv2.erode(ch, k), cv2.dilate(ch, k)
        a_ok = (med.astype(int) > mn) & (med.astype(int) < mx)
        b_ok = (ch.astype(int) > mn) & (ch.astype(int) < mx)
        res = np.where(b_ok, ch, med)
        sel = a_ok & ~done
        out[sel] = res[sel]
        done |= a_ok
        if w == smax:
            out[~done] = med[~done]
    return out


def perona_malik(img: np.ndarray, iterations: int = 15, kappa: float = 25.0, lam: float = 0.2) -> np.ndarray:
    """Perona-Malik diffusion with c(g) = exp(-(g/kappa)^2), per channel."""
    u = img.astype(np.float32)
    for _ in range(iterations):
        p = np.pad(u, ((1, 1), (1, 1), (0, 0)) if u.ndim == 3 else ((1, 1), (1, 1)), mode="edge")
        c = p[1:-1, 1:-1]
        dn, ds = p[:-2, 1:-1] - c, p[2:, 1:-1] - c
        de, dw = p[1:-1, 2:] - c, p[1:-1, :-2] - c
        f = lambda d: np.exp(-(d / kappa) ** 2)
        u = u + lam * (f(dn) * dn + f(ds) * ds + f(de) * de + f(dw) * dw)
    return np.clip(u + 0.5, 0, 255).astype(np.uint8)


def denoise_suite(img_bgr: np.ndarray, method: str = "anisotropic") -> np.ndarray:
    if method == "adaptive_median":
        return np.dstack([_adaptive_median_ch(np.ascontiguousarray(img_bgr[..., c])) for c in range(3)])
    if method == "bilateral":
        return cv2.bilateralFilter(img_bgr, 9, 40, 9)
    if method == "nlm":
        return cv2.fastNlMeansDenoisingColored(img_bgr, None, 8, 8, 7, 21)
    if method in ("anisotropic", "perona_malik"):
        return perona_malik(img_bgr)
    f = img_bgr.astype(np.float64) / 255.0
    if method == "tv":
        out = denoise_tv_chambolle(f, weight=0.1, channel_axis=-1)
    elif method == "wavelet":
        out = denoise_wavelet(f, wavelet="db4", method="BayesShrink", mode="soft",
                              wavelet_levels=3, channel_axis=-1, rescale_sigma=True)
    else:
        raise ValueError(f"unknown method {method!r}; choose from {DENOISERS}")
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def cleanup_compression_artifacts(img_bgr: np.ndarray, block: int = 8) -> np.ndarray:
    """Rolling-guidance / bilateral smoothing confined to 8x8 DCT block-boundary bands."""
    if hasattr(cv2, "ximgproc"):
        smooth = cv2.ximgproc.rollingGuidanceFilter(img_bgr, d=-1, sigmaColor=25, sigmaSpace=3, numOfIter=3)
    else:
        smooth = img_bgr
        for _ in range(3):
            smooth = cv2.bilateralFilter(smooth, 5, 25, 3)
    h, w = img_bgr.shape[:2]
    m = np.zeros((h, w), np.float32)
    m[:, (np.arange(w) % block == 0) | (np.arange(w) % block == block - 1)] = 1
    m[(np.arange(h) % block == 0) | (np.arange(h) % block == block - 1), :] = 1
    m = cv2.GaussianBlur(m, (0, 0), 0.7)[..., None]
    return np.clip(m * smooth + (1 - m) * img_bgr, 0, 255).astype(np.uint8)


def restore_auto(img_bgr: np.ndarray, psf: np.ndarray | None = None, nsr: float = 0.02,
                 denoiser: str = "anisotropic") -> dict:
    """Full GeoPulse pipeline: dehaze -> FFT-estimated Wiener deblur -> denoise."""
    out = dehaze_guided_filter(img_bgr)
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    est = estimate_blur_fft(gray)
    if psf is None:
        psf = make_motion_psf(est["est_length"], est["est_angle"])
    out = denoise_suite(out, denoiser)
    out = wiener_deblur_color(out, psf, nsr)
    return {"restored": out, "psf": psf, "estimate": est}
