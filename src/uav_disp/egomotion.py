"""Dense background ego-motion field (beyond the reference papers - the UAV adaptation).

~40 static background tiles act as Chen-style "virtual accelerometers"; each is
tracked with the phase estimator, a robust affine field d(x) = A (x - xbar) + t is
fitted per frame and evaluated AT the cable target. Also: two independent
rolling-shutter readout-time estimators and a Lanczos fractional delay.
Coordinates: absolute full-frame px unless stated.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage, signal

from .phase_disp import PhaseAccumulator, coarse_cuts, cut_patches
from .steerable import PyramidSpec, build_filters, decompose_band, padded_fft
from .track_zncc import zncc_map


# ---------------------------------------------------------------------------
# tile selection (frame 0 + one probe frame)

def red_mask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[..., i].astype(np.int32) for i in range(3))
    return (r > 90) & (r > 1.6 * g) & (r > 1.6 * b)


def texture_score(tile: np.ndarray, spec: PyramidSpec = PyramidSpec(2, 2, True)) -> float:
    """Sum of A^2 over the two finest scales (Chen's amplitude = edge strength)."""
    F = build_filters(tile.shape, spec)
    X = padded_fft(tile.astype(np.float32), F)
    return float(sum(np.sum(np.abs(decompose_band(X, F, b)) ** 2) for b in range(F.n_bands)))


def farthest_point_subset(xy: np.ndarray, n: int, start: int = 0) -> np.ndarray:
    """Greedy farthest-point selection -> indices into xy."""
    chosen = [start]
    d = np.linalg.norm(xy - xy[start], axis=1)
    while len(chosen) < min(n, len(xy)):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(xy - xy[i], axis=1))
    return np.array(chosen)


def select_tiles(gray0: np.ndarray, probe: np.ndarray, rgb0: np.ndarray, cable_center: Sequence[float],
                 tile: int = 128, n_tiles: int = 40, zncc_min: float = 0.9, cable_dilate_px: int = 40,
                 target_exclude_px: int = 150, search_rad: int = 24) -> np.ndarray:
    """(K, 2) tile centres (absolute px) on static, textured background."""
    H, W = gray0.shape
    bad = ndimage.maximum_filter(red_mask(rgb0), size=2 * cable_dilate_px + 1)
    cands, scores = [], []
    for cy in range(tile // 2 + search_rad, H - tile // 2 - search_rad, tile):
        for cx in range(tile // 2 + search_rad, W - tile // 2 - search_rad, tile):
            y0, x0 = cy - tile // 2, cx - tile // 2
            if bad[y0:y0 + tile, x0:x0 + tile].any():
                continue
            if abs(cx - cable_center[0]) < target_exclude_px and abs(cy - cable_center[1]) < target_exclude_px:
                continue
            t0 = gray0[cy - 32:cy + 32, cx - 32:cx + 32]
            s = probe[cy - 32 - search_rad:cy + 32 + search_rad, cx - 32 - search_rad:cx + 32 + search_rad]
            if zncc_map(s, t0).max() < zncc_min:
                continue
            cands.append((cx, cy))
            scores.append(texture_score(gray0[y0:y0 + tile, x0:x0 + tile]))
    if not cands:
        raise RuntimeError("no static textured tiles found")
    cands, scores = np.array(cands, float), np.array(scores)
    order = np.argsort(scores)[::-1][: max(3 * n_tiles, n_tiles)]
    pick = farthest_point_subset(cands[order], n_tiles, start=0)
    return cands[order][pick]


# ---------------------------------------------------------------------------
# per-tile phase tracks

class TileTracker:
    """Phase accumulators for K tiles, cut piecewise-constantly along a coarse global shift."""

    def __init__(self, frame0: np.ndarray, tile_xy: np.ndarray, coarse_global: np.ndarray,
                 spec: PyramidSpec = PyramidSpec(), tile: int = 128, step_px: float = 1.0, k: int = 51,
                 device: str | None = None):
        self.tile_xy = np.asarray(tile_xy, float)
        self.tile = tile
        cuts, _ = coarse_cuts(np.asarray(coarse_global, float), step_px, k)
        self.cuts = cuts                                     # (T, 2) int global shift
        self.accs = [PhaseAccumulator(cut_patches(frame0, np.rint(xy).astype(int) + cuts[0], tile), spec,
                                      per_scale=False, device=device) for xy in self.tile_xy]
        self.t = 1

    def push(self, frame: np.ndarray) -> None:
        c = self.cuts[self.t]
        for acc, xy in zip(self.accs, self.tile_xy):
            acc.push(cut_patches(frame, np.rint(xy).astype(int) + c, self.tile))
        self.t += 1

    def result(self):
        """disp (T, K, 2) absolute tile displacement vs frame 0, sigma (T, K, 2), flags (T, K)."""
        fits = [a.result() for a in self.accs]
        T = len(fits[0].uv)
        cuts = self.cuts[:T] - self.cuts[0]
        disp = np.stack([cuts + f.uv for f in fits], axis=1)
        sigma = np.stack([f.sigma for f in fits], axis=1)
        flags = np.stack([f.flags for f in fits], axis=1)
        return disp, sigma, flags


def staticness_screen(disp: np.ndarray, k_med: float = 3.0) -> np.ndarray:
    """Keep tiles whose residual after removing the per-frame median translation is not an outlier."""
    resid = disp - np.median(disp, axis=1, keepdims=True)
    s = resid.std(axis=0).max(axis=1)
    return s <= k_med * np.median(s)


# ---------------------------------------------------------------------------
# robust affine field

def _design(xy: np.ndarray, centroid: np.ndarray, model: str) -> np.ndarray:
    K = len(xy)
    if model == "translation":
        A = np.zeros((2 * K, 2))
        A[0::2, 0] = 1
        A[1::2, 1] = 1
        return A
    d = xy - centroid
    A = np.zeros((2 * K, 6))
    A[0::2, 0], A[0::2, 1], A[0::2, 4] = d[:, 0], d[:, 1], 1
    A[1::2, 2], A[1::2, 3], A[1::2, 5] = d[:, 0], d[:, 1], 1
    return A


def affine_fit(tile_xy: np.ndarray, d_t: np.ndarray, sigma_t: np.ndarray | None = None,
               huber_k: float = 1.345, iters: int = 3, model: str = "affine"):
    """Weighted Huber-IRLS fit of d = A (x - xbar) + t. Returns (params, resid (K,2), weights (K,))."""
    xy = np.asarray(tile_xy, float)
    centroid = xy.mean(axis=0)
    A = _design(xy, centroid, model)
    y = np.asarray(d_t, float).ravel()
    w0 = np.ones(len(y)) if sigma_t is None else 1.0 / np.maximum(np.asarray(sigma_t, float).ravel(), 1e-6) ** 2
    w = w0.copy()
    for _ in range(iters + 1):
        sw = np.sqrt(w)
        p, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        r = y - A @ p
        scale = 1.4826 * np.median(np.abs(r * np.sqrt(w0))) + 1e-12
        z = np.abs(r * np.sqrt(w0)) / scale
        w = w0 * np.where(z <= huber_k, 1.0, huber_k / z)
    params = p if model != "translation" else np.array([0, 0, 0, 0, p[0], p[1]])
    return params, r.reshape(-1, 2), w.reshape(-1, 2).mean(axis=1) / (w0.reshape(-1, 2).mean(axis=1) + 1e-30)


def evaluate_field(params: np.ndarray, xy: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    d = np.atleast_2d(np.asarray(xy, float) - centroid)
    A = params[:4].reshape(2, 2)
    return d @ A.T + params[4:6]


def ego_at_target(tile_xy: np.ndarray, disp: np.ndarray, sigma: np.ndarray, target_xy: Sequence[float],
                  model: str = "affine", near_px: float | None = None, huber_k: float = 1.345,
                  tau: float | None = None, fps: float = 50.0):
    """(T, 2) ego-motion at the target; optional rolling-shutter re-timing of each tile (tau s/row)."""
    xy = np.asarray(tile_xy, float)
    keep = np.ones(len(xy), bool) if near_px is None else (np.linalg.norm(xy - target_xy, axis=1) <= near_px)
    if keep.sum() < (3 if model == "affine" else 1):
        keep[:] = True
    xy, disp, sigma = xy[keep], disp[:, keep], sigma[:, keep]
    if tau:
        disp = disp.copy()
        for k in range(len(xy)):
            adv = tau * (target_xy[1] - xy[k, 1]) * fps        # target row is read later by this many samples
            for ax in range(2):
                disp[:, k, ax] = fractional_delay(disp[:, k, ax], -adv)
    centroid = xy.mean(axis=0)
    out = np.empty((len(disp), 2))
    params = np.empty((len(disp), 6))
    for t in range(len(disp)):
        p, _, _ = affine_fit(xy, disp[t], sigma[t], huber_k, model=model)
        params[t] = p
        out[t] = evaluate_field(p, np.asarray(target_xy, float), centroid)[0]
    return out, params


# ---------------------------------------------------------------------------
# rolling shutter

def _xspec_phase_slope(x: np.ndarray, ref: np.ndarray, fps: float, band: tuple[float, float], nperseg: int = 512):
    """Slope of cross-spectrum phase vs frequency (s), weighted by coherence, over `band`."""
    f, pxy = signal.csd(x, ref, fs=fps, nperseg=nperseg)
    _, coh = signal.coherence(x, ref, fs=fps, nperseg=nperseg)
    m = (f >= band[0]) & (f <= band[1]) & (coh > 0.3)
    if m.sum() < 3:
        return np.nan, 0.0
    ph = -np.angle(pxy[m])          # scipy csd = conj(X) Y -> phase of Y relative to X; we want X relative to ref
    w = coh[m] * np.abs(pxy[m]) / np.abs(pxy[m]).max()   # coherence alone is 1 everywhere for noise-free data
    A = np.stack([2 * np.pi * f[m], np.ones(m.sum())], axis=1)
    p, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], ph * np.sqrt(w), rcond=None)
    return p[0], float(w.sum())


def estimate_tau_from_tiles(disp: np.ndarray, tile_xy: np.ndarray, fps: float = 50.0,
                            band: tuple[float, float] = (3.0, 14.0)) -> tuple[float, float]:
    """E-A: rigid UAV jitter reaches row y at t + tau*y. Cross-spectral phase slope vs delta-row -> tau."""
    ref = int(np.argmin(np.abs(tile_xy[:, 1] - np.median(tile_xy[:, 1]))))
    ax = int(np.argmax(disp.std(axis=(0, 1))))
    r = disp[:, ref, ax] - disp[:, ref, ax].mean()
    drow, slope, wts = [], [], []
    for k in range(len(tile_xy)):
        if k == ref:
            continue
        s, w = _xspec_phase_slope(disp[:, k, ax] - disp[:, k, ax].mean(), r, fps, band)
        if np.isfinite(s) and w > 0:
            drow.append(tile_xy[k, 1] - tile_xy[ref, 1]); slope.append(s); wts.append(w)
    if len(drow) < 3:
        return np.nan, 0.0
    drow, slope, wts = map(np.array, (drow, slope, wts))
    tau = float(np.sum(wts * drow * slope) / np.sum(wts * drow * drow))     # slope = tau * drow
    ss_res = np.sum(wts * (slope - tau * drow) ** 2)
    ss_tot = np.sum(wts * (slope - np.average(slope, weights=wts)) ** 2) + 1e-30
    return tau, float(max(0.0, 1 - ss_res / ss_tot))


def estimate_tau_from_cable(point_tracks: np.ndarray, rows: np.ndarray, mode_freqs=(3.9, 7.5, 11.7),
                            fps: float = 50.0, nperseg: int = 1024) -> tuple[float, float]:
    """E-B: along the cable a standing mode has spatial phase 0 or pi; a phase ramp vs row whose
    slope grows with frequency is rolling shutter. point_tracks (T, P) scalar motion, rows (P,)."""
    P = point_tracks.shape[1]
    ref = P // 2
    slopes, freqs = [], []
    for f0 in mode_freqs:
        ph = []
        for p in range(P):
            f, pxy = signal.csd(point_tracks[:, p], point_tracks[:, ref], fs=fps, nperseg=nperseg)
            i = int(np.argmin(np.abs(f - f0)))
            a = -np.angle(pxy[i - 1:i + 2].sum())      # scipy csd convention (see _xspec_phase_slope)
            a = (a + np.pi / 2) % np.pi - np.pi / 2          # remove pi flips of the mode shape
            ph.append(a)
        A = np.stack([rows - rows[ref], np.ones(P)], axis=1)
        s, *_ = np.linalg.lstsq(A, np.array(ph), rcond=None)
        slopes.append(s[0]); freqs.append(2 * np.pi * f0)
    slopes, freqs = np.array(slopes), np.array(freqs)
    tau = float(np.sum(freqs * slopes) / np.sum(freqs * freqs))
    ss_res = np.sum((slopes - tau * freqs) ** 2)
    ss_tot = np.sum((slopes - slopes.mean()) ** 2) + 1e-30
    return tau, float(max(0.0, 1 - ss_res / ss_tot))


def fractional_delay(x: np.ndarray, delay_samples: float, taps: int = 8) -> np.ndarray:
    """y[n] = x[n - delay] via a Lanczos-windowed sinc (a = taps); edges replicated."""
    d = float(delay_samples)
    n_int = int(np.floor(d))
    frac = d - n_int
    k = np.arange(-taps, taps + 1)
    arg = k - frac
    h = np.sinc(arg) * np.sinc(arg / taps) * (np.abs(arg) < taps)
    h /= h.sum()
    xp = np.pad(x, taps + abs(n_int) + 1, mode="edge")
    y = np.convolve(xp, h, mode="same")                 # y[n] = sum_k h[k] x[n - k]
    y = np.roll(y, n_int)
    m = taps + abs(n_int) + 1
    return y[m:m + len(x)]


def ego_at_target_from_file(ego, target_xy_abs: Sequence[float], T: int) -> np.ndarray:
    """Ego-motion at the target (px, relative to frame 0) from results/egomotion.npz, using the
    model and tau recorded there. Returned in the same (relative) units the ref track uses."""
    keep = ego["keep"].astype(bool)
    model = str(ego["model"])
    near = float(ego["near_px"]) if "near_px" in ego and np.isfinite(float(ego["near_px"])) else None
    tau = float(ego["tau"]) if "tau" in ego and np.isfinite(float(ego["tau"])) and float(ego["tau"]) != 0 else None
    out, _ = ego_at_target(ego["tile_xy"][keep], ego["disp"][:T, keep], ego["sigma"][:T, keep],
                           target_xy_abs, model=model, near_px=near, tau=tau)
    return out
