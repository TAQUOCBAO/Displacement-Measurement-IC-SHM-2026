"""Turn raw pixel tracks into a physical displacement signal.

Steps: ego-motion compensation (cable minus stationary reference), projection
onto the in-plane normal of the cable axis, px -> mm scaling, and a band-pass
around the real motion band (config.HIGHPASS_HZ..LOWPASS_HZ). The high-pass
removes the residual parallax drift that translation-only compensation cannot
cancel (the two targets sit at slightly different depths); the low-pass removes
tracker noise above the highest VIV mode.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from . import config


def cable_axis_angle(rgb_frame: np.ndarray, center_xy: tuple[float, float], half: int = 220) -> float:
    """Cable axis angle in image coords (rad), from PCA of the red-cable mask near the target."""
    x0, y0 = int(center_xy[0]), int(center_xy[1])
    roi = rgb_frame[max(0, y0 - half) : y0 + half, max(0, x0 - half) : x0 + half].astype(np.float32)
    r, g, b = roi[..., 0], roi[..., 1], roi[..., 2]
    mask = (r > 90) & (r > 1.6 * g) & (r > 1.6 * b)
    ys, xs = np.nonzero(mask)
    if len(xs) < 500:
        raise ValueError("red cable mask too small for axis estimation")
    pts = np.stack([xs, ys], axis=1).astype(float)
    pts -= pts.mean(axis=0)
    cov = pts.T @ pts / len(pts)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, np.argmax(evals)]
    return float(np.arctan2(axis[1], axis[0]))


def displacement_mm(
    cable_xy: np.ndarray,
    ref_xy: np.ndarray,
    square_px: float,
    axis_angle_rad: float,
    highpass_hz: float | None = config.HIGHPASS_HZ,
    lowpass_hz: float | None = config.LOWPASS_HZ,
    fps: float = config.FPS,
) -> np.ndarray:
    """Zero-mean displacement (mm) along the in-plane normal to the cable axis."""
    rel = cable_xy - ref_xy
    rel = rel - rel.mean(axis=0)
    # in-plane normal; image y points down, so this is the "up-slope" direction
    normal = np.array([np.sin(axis_angle_rad), -np.cos(axis_angle_rad)])
    proj = rel @ normal
    mm = proj * (config.SQUARE_MM / square_px)
    if highpass_hz is not None:
        sos = signal.butter(4, highpass_hz, "hp", fs=fps, output="sos")
        mm = signal.sosfiltfilt(sos, mm)
    if lowpass_hz is not None:
        sos = signal.butter(6, lowpass_hz, "lp", fs=fps, output="sos")
        mm = signal.sosfiltfilt(sos, mm)
    return mm - mm.mean()
