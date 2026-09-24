"""Classical baseline: ZNCC template matching with sub-pixel quadratic peak fit.

Templates are cut once from frame 0 (no template update -> no drift error
accumulation). The search window follows the previous-frame peak, which absorbs
slow UAV drift; the window radius comfortably covers per-frame motion.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def zncc_map(search: np.ndarray, template: np.ndarray) -> np.ndarray:
    t = template.astype(np.float64)
    t = t - t.mean()
    tn = np.sqrt((t * t).sum())
    wins = sliding_window_view(search.astype(np.float64), t.shape)
    n = t.size
    s1 = wins.sum(axis=(2, 3))
    s2 = (wins * wins).sum(axis=(2, 3))
    cross = np.einsum("ijkl,kl->ij", wins, t)
    var = s2 - s1 * s1 / n
    var = np.maximum(var, 1e-12)
    return cross / (np.sqrt(var) * tn)


def subpixel_peak(corr: np.ndarray) -> tuple[float, float]:
    """Integer argmax refined by a 2D quadratic LSQ fit on the 5x5 neighborhood."""
    iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
    r = 2
    y0, y1 = max(0, iy - r), min(corr.shape[0], iy + r + 1)
    x0, x1 = max(0, ix - r), min(corr.shape[1], ix + r + 1)
    patch = corr[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    xx, yy = (xx - ix).ravel().astype(float), (yy - iy).ravel().astype(float)
    A = np.stack([np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy], axis=1)
    coef, *_ = np.linalg.lstsq(A, patch.ravel(), rcond=None)
    _, b, c, d, e, f = coef
    H = np.array([[2 * d, e], [e, 2 * f]])
    if np.linalg.det(H) == 0 or not np.all(np.linalg.eigvalsh(H) < 0):
        return float(ix), float(iy)
    dx, dy = np.linalg.solve(H, [-b, -c])
    dx, dy = np.clip([dx, dy], -1.0, 1.0)
    return float(ix + dx), float(iy + dy)


def track(
    frames: Iterable[np.ndarray],
    init_centers: dict[str, tuple[float, float]],
    template_half: int = 32,
    search_rad: int = 24,
) -> dict[str, np.ndarray]:
    """Track each named point through the frame stream.

    Returns {name: (n_frames, 2) float array of (x, y) positions} in the frame
    (crop) coordinate system, plus {name}_peak with the ZNCC peak values.
    """
    names = list(init_centers)
    templates: dict[str, np.ndarray] = {}
    tracks = {n: [] for n in names}
    peaks = {n: [] for n in names}
    prev = {n: np.array(init_centers[n], dtype=float) for n in names}

    for i, frame in enumerate(frames):
        for n in names:
            cx, cy = prev[n]
            if i == 0:
                icx, icy = int(round(cx)), int(round(cy))
                templates[n] = frame[
                    icy - template_half : icy + template_half,
                    icx - template_half : icx + template_half,
                ].copy()
                # position of the template center at frame 0 defines the origin
                prev[n] = np.array([icx, icy], dtype=float)
                tracks[n].append(prev[n].copy())
                peaks[n].append(1.0)
                continue
            icx, icy = int(round(cx)), int(round(cy))
            x0 = icx - template_half - search_rad
            y0 = icy - template_half - search_rad
            search = frame[y0 : icy + template_half + search_rad, x0 : icx + template_half + search_rad]
            corr = zncc_map(search, templates[n])
            px, py = subpixel_peak(corr)
            pos = np.array([x0 + px + template_half, y0 + py + template_half])
            tracks[n].append(pos)
            peaks[n].append(float(corr.max()))
            prev[n] = pos

    out = {n: np.array(tracks[n]) for n in names}
    out.update({f"{n}_peak": np.array(peaks[n]) for n in names})
    return out
