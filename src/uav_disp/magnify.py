"""Phase-based video motion magnification (Wadhwa 2013 / Yang 2024) + QC metrics.

Visualisation and quality-control only - NOT a measurement path (Yang Eqs. 12-14
are deliberately not implemented; magnification cannot add information).
Pipeline: luma -> stabilise -> complex steerable pyramid -> temporal band-pass of
the unwrapped phase difference vs frame 0 -> amplitude-weighted spatial phase
denoise -> x alpha -> optimized 1D Row GDGIF on the finest scales -> reconstruct.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy import fft as sfft
from scipy import ndimage, signal

from . import gdgif
from .steerable import PyramidSpec, build_filters, padded_fft
from .synth import fourier_shift
from .video_io import ffmpeg_exe

LUMA = np.array([0.299, 0.587, 0.114])


def to_luma(frame: np.ndarray) -> np.ndarray:
    f = np.asarray(frame)
    return (f @ LUMA).astype(np.float32) if f.ndim == 3 else f.astype(np.float32)


def magnify_sequence(frames: Iterable[np.ndarray], band_hz: tuple[float, float] | None, alpha: float,
                     fps: float = 50.0, spec: PyramidSpec = PyramidSpec(n_scales=4, n_orients=4),
                     gdgif_on: bool = True, gdgif_scales: Sequence[int] = (0, 1),
                     stabilise_xy: np.ndarray | None = None, phase_sigma_px: float = 2.0,
                     gdgif_h: int = 16, gdgif_mu: float = 0.022) -> np.ndarray:
    """-> (T, H, W) uint8 magnified luma. band_hz=None magnifies the raw phase difference vs frame 0."""
    stack = []
    for t, f in enumerate(frames):
        g = to_luma(f)
        if stabilise_xy is not None:
            g = fourier_shift(g.astype(np.float64), -stabilise_xy[t, 0], -stabilise_xy[t, 1]).astype(np.float32)
        stack.append(g)
    lum = np.stack(stack)
    T, H, W = lum.shape
    F = build_filters((H, W), spec)
    Hp, Wp = F.padded
    py_, px_ = F.pad
    X = sfft.fft2(np.pad(lum, ((0, 0), (py_, py_), (px_, px_)), mode="reflect"),
                  axes=(-2, -1), workers=-1).astype(np.complex64)      # (T, Hp, Wp), batched

    # residuals: analysis x synthesis (batched over frames, all cores)
    out = np.real(sfft.ifft2(X * (F.hi ** 2 + F.lo ** 2)[None], axes=(-2, -1), workers=-1)).astype(np.float32)
    sos = None
    if band_hz is not None:
        lo, hi = band_hz
        sos = signal.butter(2, [lo, hi], "bp", fs=fps, output="sos")

    for s in range(F.n_scales):
        print(f"    magnify band {band_hz} alpha {alpha:g} gdgif={gdgif_on}: scale {s + 1}/{F.n_scales} ({T} frames)", flush=True)
        scale_img = np.zeros((T, Hp, Wp), np.float32)
        for b in np.flatnonzero(F.scale_of == s):
            S = sfft.ifft2(X * F.bands[b][None], axes=(-2, -1), workers=-1).astype(np.complex64)
            A0 = np.abs(S[0])
            dphi = np.angle(S * np.conj(S[0])[None])
            # temporal ops with time as the fast (last) axis: unwrap/sosfiltfilt along axis 0 of a
            # C-ordered (T, H, W) stack is stride-hostile and was ~10x slower
            d = np.ascontiguousarray(np.moveaxis(dphi, 0, -1), dtype=np.float64)   # (H, W, T)
            d = np.unwrap(d, axis=-1)
            if sos is not None:
                d = signal.sosfiltfilt(sos, d, axis=-1)
            dphi = np.ascontiguousarray(np.moveaxis(d, -1, 0), dtype=np.float32)
            del d
            if phase_sigma_px > 0:
                den = ndimage.gaussian_filter(A0, phase_sigma_px) + 1e-6
                dphi = ndimage.gaussian_filter(A0[None] * dphi, phase_sigma_px, axes=(1, 2)) / den[None]
            S = S * np.exp(1j * alpha * dphi)
            scale_img += 2.0 * np.real(sfft.ifft2(sfft.fft2(S, axes=(-2, -1), workers=-1) * F.bands[b][None],
                                                  axes=(-2, -1), workers=-1)).astype(np.float32)
        if gdgif_on and s in gdgif_scales:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor() as ex:          # row_filter is per-image (global stats), so keep per frame
                for t, r in enumerate(ex.map(lambda img: gdgif.row_filter(img, h=gdgif_h, mu=gdgif_mu), scale_img)):
                    scale_img[t] = r
        out += scale_img
    py, px = F.pad
    return np.clip(np.rint(out[:, py:py + H, px:px + W]), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# I/O and QC

def write_video(frames: np.ndarray, path: str | Path, fps: float = 50.0, crf: int = 18) -> None:
    T, H, W = frames.shape[:3]
    cmd = [ffmpeg_exe(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()


def space_time_slice(frames: np.ndarray, x0: int, y_range: tuple[int, int]) -> np.ndarray:
    """(y1 - y0, T) slice through column x0 (Yang Figs. 3-6)."""
    return np.asarray(frames)[:, y_range[0]:y_range[1], x0].T


def ssim_psnr(a: np.ndarray, b: np.ndarray, L: float = 255.0) -> tuple[float, float]:
    """Mean SSIM (7x7 Gaussian sigma 1.5, K1=.01, K2=.03) and PSNR (dB) for two Y-channel images."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    g = lambda x: ndimage.gaussian_filter(x, 1.5, truncate=2.0)
    mu_a, mu_b = g(a), g(b)
    va, vb, vab = g(a * a) - mu_a ** 2, g(b * b) - mu_b ** 2, g(a * b) - mu_a * mu_b
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    ssim = ((2 * mu_a * mu_b + c1) * (2 * vab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    mse = np.mean((a - b) ** 2)
    psnr = np.inf if mse == 0 else 10 * np.log10(L ** 2 / mse)
    return float(ssim.mean()), float(psnr)


def canny_edges(img: np.ndarray, sigma: float = 1.5, low: float = 0.1, high: float = 0.3) -> np.ndarray:
    """scipy-only Canny: Gaussian -> Sobel -> non-max suppression -> hysteresis. Thresholds relative to max."""
    g = ndimage.gaussian_filter(np.asarray(img, np.float64), sigma)
    gx, gy = ndimage.sobel(g, axis=1), ndimage.sobel(g, axis=0)
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx)
    q = np.rint(ang / (np.pi / 4)).astype(int) % 4
    nms = np.zeros_like(mag, bool)
    shifts = {0: ((0, 1), (0, -1)), 1: ((1, 1), (-1, -1)), 2: ((1, 0), (-1, 0)), 3: ((1, -1), (-1, 1))}
    for k, (s1, s2) in shifts.items():
        m1 = np.roll(mag, s1, axis=(0, 1))
        m2 = np.roll(mag, s2, axis=(0, 1))
        nms |= (q == k) & (mag >= m1) & (mag >= m2)
    mx = mag.max() + 1e-12
    strong = nms & (mag >= high * mx)
    weak = nms & (mag >= low * mx)
    lab, n = ndimage.label(weak, structure=np.ones((3, 3)))
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[strong])] = True
    keep[0] = False
    return keep[lab]


def intensity_trace(frames: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Yang Eq. (12)-style mean intensity in an ROI (x, y, w, h) - a VISUAL trace, not a displacement."""
    x, y, w, h = roi
    return np.asarray(frames)[:, y:y + h, x:x + w].mean(axis=(1, 2))
