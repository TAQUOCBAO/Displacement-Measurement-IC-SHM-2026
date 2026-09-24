"""Yang & Jiang 2024 - optimized 1D row-gradient-domain guided image filter.

Implements Eqs. (6)-(11) of the paper with a *row* window (length h) instead of
a square one: the point is to isolate the row-aligned stripe component of
high-frequency sub-bands. Fully vectorised with cumulative-sum box means.

Symbols -> variables: xi_{u,16} mu_u | xi_{X,16} mu_X | xi_{u*X,16} mu_uX |
sigma^2_{u,16} var16 | sigma_{u,1} sd1 | chi | gamma_rk gamma | eta |
xi_{chi,inf} chi_inf | Gamma_u Gamma | a_rk a | b_rk b | a_bar, b_bar | H.
"""

from __future__ import annotations

import numpy as np

H_DEFAULT = 16
MU_DEFAULT = 0.022


def _row_box_mean(a: np.ndarray, h: int) -> np.ndarray:
    """Mean over a centred row window of length h (even h -> [i-h//2, i+h//2-1]),
    reflect padding along rows. O(HW) via cumulative sums."""
    a = np.asarray(a, dtype=np.float64)
    left = h // 2
    right = h - 1 - left
    p = np.pad(a, ((0, 0), (left, right)), mode="reflect")
    c = np.cumsum(p, axis=1)
    c = np.concatenate([np.zeros((a.shape[0], 1)), c], axis=1)
    return (c[:, h:] - c[:, :-h]) / h


def edge_weights(u: np.ndarray, h: int = H_DEFAULT, eps: float | None = None):
    """(chi, gamma, Gamma) of Eqs. (9)-(10) for a guide u already scaled to [0, 1]."""
    u = np.asarray(u, dtype=np.float64)
    mu_u = _row_box_mean(u, h)
    var16 = np.maximum(_row_box_mean(u * u, h) - mu_u * mu_u, 0.0)
    mu3 = _row_box_mean(u, 3)
    var3 = np.maximum(_row_box_mean(u * u, 3) - mu3 * mu3, 0.0)
    chi = np.sqrt(var3) * np.sqrt(var16)
    if eps is None:
        eps = 1e-6 * (chi.max() - chi.min() + 1e-12)
    chi_inf = chi.mean()
    eta = 4.0 / (chi_inf - chi.min() + 1e-12)
    gamma = 1.0 - 1.0 / (1.0 + np.exp(np.clip(eta * (chi - chi_inf), -50.0, 50.0)))
    Gamma = (chi_inf + eps) / (chi + eps)          # closed form of (1/N) sum_k (chi_k+eps)/(chi_i+eps)
    return chi, gamma, Gamma


def row_filter(X: np.ndarray, guide: np.ndarray | None = None, h: int = H_DEFAULT,
               mu: float = MU_DEFAULT, eps: float | None = None) -> np.ndarray:
    """Filter X (H, W) with guide u (default: X itself). Returns float32 (H, W)."""
    X = np.asarray(X, dtype=np.float64)
    u = X if guide is None else np.asarray(guide, dtype=np.float64)
    if X.shape != u.shape or X.ndim != 2:
        raise ValueError("X and guide must be 2-D arrays of the same shape")
    lo, hi = u.min(), u.max()
    rng = hi - lo
    if rng <= 0:
        return X.astype(np.float32)
    un = (u - lo) / rng
    Xn = (X - lo) / rng

    mu_u = _row_box_mean(un, h)
    mu_X = _row_box_mean(Xn, h)
    mu_uX = _row_box_mean(un * Xn, h)
    var16 = np.maximum(_row_box_mean(un * un, h) - mu_u * mu_u, 0.0)
    _, gamma, Gamma = edge_weights(un, h, eps)

    reg = mu / Gamma
    a = (mu_uX - mu_u * mu_X + reg * gamma) / (var16 + reg)
    b = mu_X - a * mu_u
    a_bar = _row_box_mean(a, h)
    b_bar = _row_box_mean(b, h)
    Hn = a_bar * un + b_bar
    return (Hn * rng + lo).astype(np.float32)


def reference_loop(X: np.ndarray, h: int = H_DEFAULT, mu: float = MU_DEFAULT) -> np.ndarray:
    """Slow per-pixel reference of row_filter (self-guided) for tests only."""
    X = np.asarray(X, dtype=np.float64)
    lo, hi = X.min(), X.max()
    un = (X - lo) / (hi - lo)
    Hh, W = un.shape
    left = h // 2
    right = h - 1 - left

    def win(row, i, half_l, half_r):
        idx = np.arange(i - half_l, i + half_r + 1)
        idx = np.abs(idx)                       # reflect
        idx = np.where(idx >= W, 2 * (W - 1) - idx, idx)
        return row[idx]

    chi = np.zeros_like(un)
    for r in range(Hh):
        for i in range(W):
            w16, w3 = win(un[r], i, left, right), win(un[r], i, 1, 1)
            chi[r, i] = w3.std() * w16.std()
    eps = 1e-6 * (chi.max() - chi.min() + 1e-12)
    chi_inf, eta = chi.mean(), 4.0 / (chi.mean() - chi.min() + 1e-12)
    gamma = 1 - 1 / (1 + np.exp(np.clip(eta * (chi - chi_inf), -50, 50)))
    Gamma = np.zeros_like(chi)
    for r in range(Hh):
        for i in range(W):
            Gamma[r, i] = np.mean((chi + eps) / (chi[r, i] + eps))
    a = np.zeros_like(un)
    b = np.zeros_like(un)
    for r in range(Hh):
        for i in range(W):
            w = win(un[r], i, left, right)
            reg = mu / Gamma[r, i]
            a[r, i] = (np.mean(w * w) - w.mean() ** 2 + reg * gamma[r, i]) / (w.var() + reg)
            b[r, i] = w.mean() - a[r, i] * w.mean()
    out = np.zeros_like(un)
    for r in range(Hh):
        for i in range(W):
            out[r, i] = win(a[r], i, left, right).mean() * un[r, i] + win(b[r], i, left, right).mean()
    return out * (hi - lo) + lo
