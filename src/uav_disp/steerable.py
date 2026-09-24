"""Complex steerable pyramid (frequency domain) + Chen's G2/H2 quadrature pair.

Both filter banks are expressed as frequency-domain masks on a reflect-padded
patch, so one code path (`decompose_band`, `band_gradient`) serves both:

* pyramid  - Simoncelli/Freeman radial raised-cosine bands x one-sided
             cos^(K-1) angular masks (Wadhwa 2013, Yang 2024 Eq. 1).  Real masks;
             one-sided support makes the inverse FFT analytic (amplitude + phase).
* g2h2     - Freeman & Adelson 9-tap G2/H2 pair as reproduced in Chen 2015
             App. A (taps corrected, see G_F1 below). Complex masks G + iH.

Sub-bands are complex64; amplitude/phase are meaningful only because the
support is one-sided (pyramid) or the pair is in quadrature (g2h2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy import fft as sfft

# ---------------------------------------------------------------------------
# Chen 2015 Table A1 (Freeman & Adelson 1991), sign-verified against the
# analytic profiles 0.9213(2x^2-1)e^{-x^2} / e^{-x^2} / x(-2.205+0.978x^2)e^{-x^2}
# sampled at x = k*0.67. The printed G_f2 tap +3 (0.0480) is a typo -> 0.0176.
G_F1 = np.array([0.0094, 0.1148, 0.3964, -0.0601, -0.9213, -0.0601, 0.3964, 0.1148, 0.0094])
G_F2 = np.array([0.0008, 0.0176, 0.1660, 0.6383, 1.0000, 0.6383, 0.1660, 0.0176, 0.0008])
H_F1 = np.array([-0.0098, -0.0618, 0.0998, 0.7551, 0.0000, -0.7551, -0.0998, 0.0618, 0.0098])
H_F2 = G_F2


@dataclass(frozen=True)
class PyramidSpec:
    n_scales: int = 4          # band-pass levels (excluding hi/lo residuals)
    n_orients: int = 2         # K orientations, theta_k = k*pi/K
    half_octave: bool = True   # radial spacing 2^-0.5 (True) or 2^-1
    twidth: float = 1.0        # raised-cosine transition width, octaves
    pad: int | None = None     # reflect pad per side; None -> side // 4
    bank: str = "pyramid"      # "pyramid" | "g2h2"

    def pad_for(self, shape: tuple[int, int]) -> tuple[int, int]:
        if self.pad is not None:
            return (self.pad, self.pad)
        return (shape[0] // 4, shape[1] // 4)


@dataclass
class Filters:
    shape: tuple[int, int]
    padded: tuple[int, int]
    pad: tuple[int, int]
    hi: np.ndarray                 # (Hp, Wp) float32
    lo: np.ndarray                 # (Hp, Wp) float32
    bands: list[np.ndarray]        # (Hp, Wp) float32 (pyramid) or complex64 (g2h2)
    scale_of: np.ndarray           # (n_bands,) int, 0 = finest
    orient_of: np.ndarray          # (n_bands,) int
    omega_x: np.ndarray            # (Hp, Wp) float32, rad/px
    omega_y: np.ndarray
    spec: PyramidSpec
    bw_frac: np.ndarray = field(default=None)  # (n_bands,) fraction of independent samples after filtering

    @property
    def n_bands(self) -> int:
        return len(self.bands)

    @property
    def n_scales(self) -> int:
        return int(self.scale_of.max()) + 1 if self.n_bands else 0


# ---------------------------------------------------------------------------
# construction

def _frequency_grid(hp: int, wp: int):
    wx = (2 * np.pi * np.fft.fftfreq(wp))[None, :].astype(np.float32)
    wy = (2 * np.pi * np.fft.fftfreq(hp))[:, None].astype(np.float32)
    omega_x = np.broadcast_to(wx, (hp, wp)).copy()
    omega_y = np.broadcast_to(wy, (hp, wp)).copy()
    return omega_x, omega_y


def _radial_pair(log_r: np.ndarray, c: float, twidth: float):
    """(hi, lo) raised-cosine masks with transition on [c - twidth, c] (log2 units, 0 = Nyquist)."""
    x = np.clip((log_r - c) / twidth, -1.0, 0.0) + 1.0
    return np.sin(0.5 * np.pi * x), np.cos(0.5 * np.pi * x)


def _pyramid_filters(shape: tuple[int, int], spec: PyramidSpec) -> Filters:
    py, px = spec.pad_for(shape)
    hp, wp = shape[0] + 2 * py, shape[1] + 2 * px
    omega_x, omega_y = _frequency_grid(hp, wp)
    r = np.hypot(omega_x, omega_y)
    with np.errstate(divide="ignore"):
        log_r = np.log2(r / np.pi)
    log_r[0, 0] = -1e6
    theta = np.arctan2(omega_y, omega_x)

    hi, cum = _radial_pair(log_r, 0.0, spec.twidth)
    delta = 0.5 if spec.half_octave else 1.0
    K = spec.n_orients

    # angular masks; normalisation makes sum_k Theta_k_full^2 == 1 (steerability)
    thetas = [k * np.pi / K for k in range(K)]
    full = [np.abs(np.cos(theta - t)) ** (K - 1) for t in thetas]
    norm = np.sqrt(sum(f * f for f in full))
    norm[norm == 0] = 1.0
    onesided = []
    for t, f in zip(thetas, full):
        d = np.angle(np.exp(1j * (theta - t)))
        onesided.append(np.where(np.abs(d) < np.pi / 2, f / norm, 0.0))

    bands, scale_of, orient_of = [], [], []
    for s in range(spec.n_scales):
        h_s, l_s = _radial_pair(log_r, -(s + 1) * delta, spec.twidth)
        radial = h_s * cum
        cum = cum * l_s
        for k in range(K):
            b = (radial * onesided[k]).astype(np.float32)
            b[0, 0] = 0.0
            bands.append(b)
            scale_of.append(s)
            orient_of.append(k)
    lo = cum.astype(np.float32)
    lo[0, 0] = 1.0
    F = Filters(shape, (hp, wp), (py, px), hi.astype(np.float32), lo, bands,
                np.array(scale_of), np.array(orient_of), omega_x, omega_y, spec)
    F.bw_frac = np.array([_bandwidth_fraction(b) for b in bands])
    return F


def _bandwidth_fraction(mask: np.ndarray) -> float:
    """Fraction of independent samples that white noise keeps after this filter."""
    p2 = np.abs(mask) ** 2
    return float(p2.sum() ** 2 / (mask.size * (p2 * p2).sum() + 1e-30))


def g2h2_bank() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{'0': (G2^0, H2^0), 'pi/2': (G2^pi/2, H2^pi/2)} as 9x9 kernels, k[y, x]."""
    _check_g2h2_taps()
    f1 = G_F1 - G_F1.mean()
    g0 = np.outer(G_F2, f1)       # rows: y-profile f2, cols: x-profile f1
    h0 = np.outer(H_F2, H_F1)
    g90 = np.outer(f1, G_F2)
    h90 = np.outer(H_F1, H_F2)
    return {"0": (g0, h0), "pi/2": (g90, h90)}


def _check_g2h2_taps() -> None:
    f1 = G_F1 - G_F1.mean()
    assert abs(G_F1.sum()) < 1e-3, "G_f1 DC response too large"
    assert abs(H_F1.sum()) < 1e-12, "H_f1 must have zero DC"
    assert np.allclose(G_F1, G_F1[::-1]) and np.allclose(G_F2, G_F2[::-1]), "G taps must be symmetric"
    assert np.allclose(H_F1, -H_F1[::-1]), "H_f1 must be antisymmetric"
    n = 512
    F1, H1 = np.fft.rfft(f1, n), np.fft.rfft(H_F1, n)
    f = np.fft.rfftfreq(n)
    m = (f > 0.1) & (f < 0.4)
    ratio = np.abs(H1[m]) / np.abs(F1[m])
    lag = np.angle(H1[m] * np.conj(F1[m]))
    assert np.all(np.abs(ratio - 1) < 0.15), f"G2/H2 not in quadrature (magnitude): {ratio.min():.3f}..{ratio.max():.3f}"
    assert np.all(np.abs(np.abs(lag) - np.pi / 2) < 0.05), f"G2/H2 phase lag not pi/2: {lag.min():.3f}..{lag.max():.3f}"


def _kernel_mask(kernel: np.ndarray, hp: int, wp: int) -> np.ndarray:
    """Frequency mask of a small centred kernel so that ifft(X * mask) == convolve(img, kernel)."""
    kh, kw = kernel.shape
    emb = np.zeros((hp, wp), np.float64)
    emb[:kh, :kw] = kernel
    emb = np.roll(emb, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return np.fft.fft2(emb)


def _g2h2_filters(shape: tuple[int, int], spec: PyramidSpec) -> Filters:
    py, px = spec.pad_for(shape)
    hp, wp = shape[0] + 2 * py, shape[1] + 2 * px
    omega_x, omega_y = _frequency_grid(hp, wp)
    bank = g2h2_bank()
    bands = []
    for key in ("0", "pi/2"):
        g, h = bank[key]
        bands.append((_kernel_mask(g, hp, wp) + 1j * _kernel_mask(h, hp, wp)).astype(np.complex64))
    zeros = np.zeros((hp, wp), np.float32)
    F = Filters(shape, (hp, wp), (py, px), zeros, zeros.copy(), bands,
                np.zeros(2, int), np.arange(2), omega_x, omega_y, spec)
    F.bw_frac = np.array([_bandwidth_fraction(b) for b in bands])
    return F


@lru_cache(maxsize=16)
def _build_cached(shape: tuple[int, int], spec: PyramidSpec) -> Filters:
    if spec.bank == "pyramid":
        return _pyramid_filters(shape, spec)
    if spec.bank == "g2h2":
        return _g2h2_filters(shape, spec)
    raise ValueError(f"unknown bank {spec.bank!r}")


def build_filters(shape: tuple[int, int], spec: PyramidSpec = PyramidSpec()) -> Filters:
    return _build_cached((int(shape[0]), int(shape[1])), spec)


# ---------------------------------------------------------------------------
# transforms

def _pad(img: np.ndarray, F: Filters) -> np.ndarray:
    py, px = F.pad
    return np.pad(img, ((py, py), (px, px)), mode="reflect")


def _crop(arr: np.ndarray, F: Filters) -> np.ndarray:
    py, px = F.pad
    h, w = F.shape
    return arr[py:py + h, px:px + w]


def padded_fft(img: np.ndarray, F: Filters) -> np.ndarray:
    """reflect-pad by F.pad then fft2 -> (Hp, Wp) complex64."""
    x = np.asarray(img, dtype=np.float32)
    if x.shape != F.shape:
        raise ValueError(f"image shape {x.shape} != filter shape {F.shape}")
    return sfft.fft2(_pad(x, F), workers=-1).astype(np.complex64)


def decompose_band(X: np.ndarray, F: Filters, b: int, crop: bool = True) -> np.ndarray:
    """One complex sub-band from a padded spectrum."""
    out = sfft.ifft2(X * F.bands[b], workers=-1).astype(np.complex64)
    return _crop(out, F) if crop else out


def band_gradient(X: np.ndarray, F: Filters, b: int, crop: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Analytic spatial derivatives (d/dx, d/dy) of sub-band b."""
    Y = X * F.bands[b]
    gx = sfft.ifft2(Y * (1j * F.omega_x), workers=-1).astype(np.complex64)
    gy = sfft.ifft2(Y * (1j * F.omega_y), workers=-1).astype(np.complex64)
    return (_crop(gx, F), _crop(gy, F)) if crop else (gx, gy)


def decompose(img: np.ndarray, F: Filters, crop: bool = True) -> list[np.ndarray]:
    """[hi (float32), band_0..band_{n-1} (complex64), lo (float32)]."""
    X = padded_fft(img, F)
    hi = np.real(sfft.ifft2(X * F.hi, workers=-1)).astype(np.float32)
    lo = np.real(sfft.ifft2(X * F.lo, workers=-1)).astype(np.float32)
    bands = [decompose_band(X, F, b, crop=False) for b in range(F.n_bands)]
    parts = [hi, *bands, lo]
    return [_crop(p, F) for p in parts] if crop else parts


def reconstruct(parts: Sequence[np.ndarray], F: Filters) -> np.ndarray:
    """Inverse of decompose. Padded parts reconstruct exactly; cropped parts are
    reflect-padded first (exact only away from the patch boundary)."""
    if F.spec.bank != "pyramid":
        raise ValueError("reconstruct is only defined for the tight-frame pyramid bank")
    hi, bands, lo = parts[0], parts[1:-1], parts[-1]
    padded = hi.shape == F.padded
    P = (lambda a: a) if padded else (lambda a: _pad(a, F))
    Y = sfft.fft2(P(hi)) * F.hi + sfft.fft2(P(lo)) * F.lo
    for b, band in enumerate(bands):
        Y = Y + 2.0 * sfft.fft2(P(band)) * F.bands[b]     # x2: one-sided bands carry half the energy
    out = np.real(sfft.ifft2(Y)).astype(np.float32)
    return out if padded else _crop(out, F)


def frame_identity(F: Filters) -> np.ndarray:
    """hi^2 + lo^2 + sum_b (|B_b(w)|^2 + |B_b(-w)|^2); must be 1 everywhere for the pyramid."""
    tot = F.hi.astype(np.float64) ** 2 + F.lo.astype(np.float64) ** 2
    for b in F.bands:
        p = np.abs(b.astype(np.complex128)) ** 2
        tot += p + np.roll(p[::-1, ::-1], (1, 1), axis=(0, 1))
    return tot


def g2h2_decompose(img: np.ndarray, spec: PyramidSpec | None = None) -> list[np.ndarray]:
    """[S_0, S_pi/2] complex64 via the G2/H2 pair (reflect boundary)."""
    spec = spec or PyramidSpec(bank="g2h2")
    F = build_filters(img.shape, spec)
    X = padded_fft(img, F)
    return [decompose_band(X, F, b) for b in range(F.n_bands)]
