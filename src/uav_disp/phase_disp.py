"""Local-phase displacement estimator (Chen et al. 2015 s2.3, generalised).

Per sub-band S_t = A e^{i phi}, with frame 0 as fixed reference:
    dphi_t  = arg(S_t conj(S_0))                   (wrap-safe temporal phase difference)
    phi_x/y = Im(conj(S_0) dS_0/dx,y) / |S_0|^2    (spatial phase gradient, reference frame only)
    (phi_x, phi_y, dphi_t) . (u, v, 1) = 0         (Chen Eq. 3, phase constancy)
Solved per frame as amplitude-weighted least squares over ALL pixels and sub-bands:
    M (u, v)^T = b,  M = sum w [phi_x^2, phi_x phi_y; ., phi_y^2],  b = -sum w [phi_x, phi_y] dphi_t
with w = A_0^2 * mask (inverse-variance weight = Chen's amplitude-weighted average).
Sign: content moving by +d gives dphi = -phi_x d, hence the minus sign in b
(tests/test_phase_disp.py::test_sign_convention_matches_fourier_shift is the authority).

Per-frame uncertainty from the weighted residual (Cov = s^2 M^-1, corrected for
the spatial correlation the band-pass filters induce), cond(M) and wrap flags.
Everything is accumulated one sub-band at a time - no (T, H, W) complex stacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import ndimage

from .detect import Board
from .steerable import Filters, PyramidSpec, band_gradient, build_filters, decompose_band, padded_fft

FLAG_COND = 1
FLAG_WRAP = 2
FLAG_NAN = 4
WRAP_RAD = 2.5      # |dphi| above this at the finest scale counts as wrap risk


@dataclass
class PhaseFit:
    uv: np.ndarray                     # (T, 2) residual displacement px (u = x, v = y); frame 0 == 0
    sigma: np.ndarray                  # (T, 2) 1-sigma px
    n_eff: np.ndarray                  # (T,)
    cond: np.ndarray                   # (T,)
    flags: np.ndarray                  # (T,) uint8 bitmask FLAG_*
    resid_rms: np.ndarray              # (T,) weighted rms phase residual, rad
    per_scale_uv: np.ndarray | None    # (T, n_scales, 2)
    per_scale_sigma: np.ndarray | None


# ---------------------------------------------------------------------------
# weights and geometry helpers

def gaussian_taper(h: int, w: int, rel_sigma: float = 0.3) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w]
    r2 = (xx - (w - 1) / 2) ** 2 + (yy - (h - 1) / 2) ** 2
    return np.exp(-0.5 * r2 / (rel_sigma * w) ** 2).astype(np.float32)


def board_mask(board: Board, h: int, w: int, patch_origin: Sequence[float], margin_px: float = 6.0) -> np.ndarray:
    """1 inside the rotated 2x2 board square (side 2*square_px + 2*margin), else 0."""
    cx, cy = board.center[0] - patch_origin[0], board.center[1] - patch_origin[1]
    ang = np.deg2rad(board.angle_deg)
    yy, xx = np.mgrid[:h, :w]
    dx, dy = xx - cx, yy - cy
    u = dx * np.cos(ang) + dy * np.sin(ang)
    v = -dx * np.sin(ang) + dy * np.cos(ang)
    half = board.square_px + margin_px
    return ((np.abs(u) <= half) & (np.abs(v) <= half)).astype(np.float32)


def block_weight(origin_xy: Sequence[int], h: int, w: int, period: int = 16, width: int = 1,
                 w_block: float = 0.25) -> np.ndarray:
    """Down-weight pixels within +-width of the absolute `period`-px macroblock grid."""
    ox, oy = int(origin_xy[0]), int(origin_xy[1])
    xs = (np.arange(w) + ox) % period
    ys = (np.arange(h) + oy) % period
    near = lambda a: (a <= width) | (a >= period - width)
    m = near(ys)[:, None] | near(xs)[None, :]
    return np.where(m, w_block, 1.0).astype(np.float32)


def coarse_cuts(track: np.ndarray, step_px: float = 1.0, k: int = 51) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-constant integer cut positions from a noisy coarse track.

    The track is median-filtered (k frames) and the integer cut is held until the
    smoothed track drifts >= step_px (per axis) from the current cut.
    Returns ((T, 2) int cuts, indices of frames where the cut changed)."""
    t = np.asarray(track, dtype=float)
    k = int(k) | 1
    sm = np.stack([ndimage.median_filter(t[:, i], size=k, mode="nearest") for i in range(2)], axis=1)
    cuts = np.empty_like(t, dtype=int)
    cur = np.rint(sm[0]).astype(int)
    changes = []
    for i in range(len(t)):
        moved = np.abs(sm[i] - cur) >= step_px
        if moved.any():
            cur = np.where(moved, np.rint(sm[i]).astype(int), cur)
            changes.append(i)
        cuts[i] = cur
    return cuts, np.array(changes, dtype=int)


def cut_patches(frame: np.ndarray, cut_xy: Sequence[int], size: int) -> np.ndarray:
    cx, cy = int(cut_xy[0]), int(cut_xy[1])
    h = size // 2
    y0, x0 = cy - h, cx - h
    if y0 < 0 or x0 < 0 or y0 + size > frame.shape[0] or x0 + size > frame.shape[1]:
        raise ValueError(f"patch at ({cx},{cy}) size {size} leaves the frame {frame.shape}")
    return frame[y0:y0 + size, x0:x0 + size].astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# estimator

class PhaseAccumulator:
    """Feed frames one at a time; frame 0 is the reference given at construction."""

    def __init__(self, ref_patch: np.ndarray, spec: PyramidSpec = PyramidSpec(),
                 weight_mask: np.ndarray | None = None, cond_max: float = 50.0,
                 per_scale: bool = True, scales: Sequence[int] | None = None,
                 chen_mode: bool = False, block_origin: Sequence[int] | None = None,
                 block_kw: dict | None = None, n_refine: int = 1, device: str | None = None):
        ref = np.asarray(ref_patch, dtype=np.float32)
        self.shape = ref.shape
        self.F: Filters = build_filters(ref.shape, spec)
        self.spec = spec
        self.cond_max = cond_max
        self.per_scale = per_scale
        self.chen_mode = chen_mode
        self.block_kw = block_kw or {}
        self.n_refine = int(n_refine)   # Gauss-Newton passes: re-solve after shifting the reference by the estimate
        self.device = device            # None -> numpy/scipy; "cuda"/"cpu" -> batched torch.fft backend (same math)
        self.mask = gaussian_taper(*ref.shape) if weight_mask is None else np.asarray(weight_mask, np.float32)
        keep = set(range(self.F.n_scales)) if scales is None else set(int(s) for s in scales)
        self.bands = [b for b in range(self.F.n_bands) if int(self.F.scale_of[b]) in keep]
        if not self.bands:
            raise ValueError("no sub-bands selected")
        self.finest_scale = min(int(self.F.scale_of[b]) for b in self.bands)

        X0 = padded_fft(ref, self.F)
        self.X0 = X0
        self.S0, self.phi_x, self.phi_y, self.w_base = [], [], [], []
        for b in self.bands:
            S0 = decompose_band(X0, self.F, b)
            gx, gy = band_gradient(X0, self.F, b)
            A2 = (S0.real ** 2 + S0.imag ** 2).astype(np.float32)
            good = A2 > (1e-3 * np.sqrt(A2.max())) ** 2
            inv = np.where(good, 1.0 / np.maximum(A2, 1e-30), 0.0).astype(np.float32)
            self.S0.append(S0)
            self.phi_x.append((np.imag(np.conj(S0) * gx) * inv).astype(np.float32))
            self.phi_y.append((np.imag(np.conj(S0) * gy) * inv).astype(np.float32))
            self.w_base.append((A2 * self.mask * good).astype(np.float32))
        self._block_origin = None
        self.set_block_origin(block_origin)

        self.uv, self.sigma, self.n_eff, self.cond, self.flags, self.resid = [], [], [], [], [], []
        self.ps_uv, self.ps_sigma = [], []
        self._gpu = _TorchBackend(self) if device else None
        self.push(ref)      # frame 0: exactly zero by construction

    # -- weights --------------------------------------------------------------
    def set_block_origin(self, origin: Sequence[int] | None) -> None:
        """(Re)compute weights and normal matrices; cheap, call only when the cut changes."""
        if origin is not None:
            origin = (int(origin[0]), int(origin[1]))
        if origin == self._block_origin and hasattr(self, "M"):
            return
        self._block_origin = origin
        bw = 1.0 if origin is None else block_weight(origin, *self.shape, **self.block_kw)
        self.w = [wb * bw for wb in self.w_base]
        self.M = np.zeros((2, 2))
        self.M_scale = {}
        self.sum_w = 0.0
        self.sum_w2 = 0.0
        self.sum_w_bw = 0.0
        for i, b in enumerate(self.bands):
            w, px, py = self.w[i], self.phi_x[i], self.phi_y[i]
            m = np.array([[np.sum(w * px * px), np.sum(w * px * py)],
                          [0.0, np.sum(w * py * py)]], dtype=float)
            m[1, 0] = m[0, 1]
            self.M += m
            s = int(self.F.scale_of[b])
            self.M_scale[s] = self.M_scale.get(s, np.zeros((2, 2))) + m
            sw = float(w.sum())
            self.sum_w += sw
            self.sum_w2 += float(np.sum(w * w))
            self.sum_w_bw += sw * float(self.F.bw_frac[b])
        self.Minv = np.linalg.pinv(self.M)
        self.Minv_scale = {s: np.linalg.pinv(m) for s, m in self.M_scale.items()}
        self.n_eff_val = self.sum_w ** 2 / max(self.sum_w2, 1e-30)
        if getattr(self, "_gpu", None) is not None:
            self._gpu.w = self._gpu.torch.stack([self._gpu.torch.as_tensor(w) for w in self.w]).to(self._gpu.dev)
        self.cond_val = float(np.linalg.cond(self.M))
        # Spatial-correlation correction of Cov: band-pass noise is correlated over ~1/bw_frac
        # pixels, but the A^2 weights sit on edges where it is much less so; empirically the
        # square root of the full-band factor calibrates (tests/test_phase_disp.py::test_sigma_calibration).
        self.kappa = float(np.sqrt(self.sum_w / max(self.sum_w_bw, 1e-30)))

    # -- per frame ------------------------------------------------------------
    def _dphis(self, X: np.ndarray, uv: np.ndarray | None) -> list[np.ndarray]:
        """Wrap-safe phase differences vs the reference (optionally pre-shifted by uv)."""
        out = []
        for i, b in enumerate(self.bands):
            S = decompose_band(X, self.F, b)
            if uv is None:
                S0 = self.S0[i]
            else:   # reference content moved by +uv: multiply its spectrum by e^{-i w.uv}
                ph = np.exp(-1j * (self.F.omega_x * uv[0] + self.F.omega_y * uv[1])).astype(np.complex64)
                S0 = decompose_band(self.X0 * ph, self.F, b)
            out.append(np.angle(S * np.conj(S0)).astype(np.float32))
        return out

    def _solve(self, dphis: list[np.ndarray]) -> tuple[np.ndarray, dict]:
        b_vec = np.zeros(2)
        b_scale: dict[int, np.ndarray] = {}
        for i, b in enumerate(self.bands):
            w, d = self.w[i], dphis[i]
            contrib = -np.array([np.sum(w * self.phi_x[i] * d), np.sum(w * self.phi_y[i] * d)])
            b_vec += contrib
            s = int(self.F.scale_of[b])
            b_scale[s] = b_scale.get(s, np.zeros(2)) + contrib
        uv = self._solve_chen(dphis) if self.chen_mode else self.Minv @ b_vec
        return uv, b_scale

    def push(self, patch: np.ndarray, block_origin: Sequence[int] | None = None,
             init_uv: Sequence[float] | None = None) -> None:
        """init_uv: linearisation point (px) from a coarse tracker; the reference is Fourier-shifted
        by it so the phase difference only carries the coarse tracker's error (no wrap)."""
        if block_origin is not None:
            self.set_block_origin(block_origin)
        if self._gpu is not None:
            return self._gpu.push(patch, init_uv)
        X = padded_fft(np.asarray(patch, np.float32), self.F)

        uv = np.zeros(2) if init_uv is None else np.asarray(init_uv, float)
        dphis = self._dphis(X, None if init_uv is None else uv)
        wrap_w = sum(float(np.sum(self.w[i] * (np.abs(dphis[i]) > WRAP_RAD)))
                     for i, b in enumerate(self.bands) if int(self.F.scale_of[b]) == self.finest_scale)
        duv, b_scale = self._solve(dphis)
        uv = uv + duv
        ps_uv = {s: (uv - duv) + self.Minv_scale[s] @ bs for s, bs in b_scale.items()}
        for _ in range(self.n_refine):
            if not np.all(np.isfinite(uv)):
                break
            dphis = self._dphis(X, uv)
            duv, db_scale = self._solve(dphis)
            uv = uv + duv
            ps_uv = {s: ps_uv[s] + self.Minv_scale[s] @ (db_scale[s] + self.M_scale[s] @ (uv - duv - ps_uv[s]))
                     for s in ps_uv}   # per-scale: same linearisation point (uv), own residual
        swr2 = 0.0
        for i in range(len(self.bands)):    # dphis are relative to the last linearisation point uv - duv
            r = dphis[i] + self.phi_x[i] * duv[0] + self.phi_y[i] * duv[1]
            swr2 += float(np.sum(self.w[i] * r * r))
        self._record(uv, swr2, wrap_w, ps_uv)

    def _record(self, uv: np.ndarray, swr2: float, wrap_w: float, ps_uv: dict) -> None:
        n_eff = self.n_eff_val
        s2 = swr2 / max(n_eff - 2.0, 1.0)
        cov = s2 * self.kappa * self.Minv
        sigma = np.sqrt(np.maximum(np.diag(cov), 0.0))

        flags = 0
        if self.cond_val > self.cond_max:
            flags |= FLAG_COND
        finest_w = sum(float(self.w[i].sum()) for i, b in enumerate(self.bands)
                       if int(self.F.scale_of[b]) == self.finest_scale)
        if finest_w > 0 and wrap_w / finest_w > 0.05:
            flags |= FLAG_WRAP
        if not np.all(np.isfinite(uv)):
            flags |= FLAG_NAN

        self.uv.append(uv)
        self.sigma.append(sigma)
        self.n_eff.append(n_eff)
        self.cond.append(self.cond_val)
        self.flags.append(flags)
        self.resid.append(np.sqrt(swr2 / max(self.sum_w, 1e-30)))
        if self.per_scale:
            arr_uv = np.full((self.F.n_scales, 2), np.nan)
            arr_sg = np.full((self.F.n_scales, 2), np.nan)
            for s in ps_uv:
                arr_uv[s] = ps_uv[s]
                arr_sg[s] = np.sqrt(np.maximum(np.diag(s2 * self.kappa * self.Minv_scale[s]), 0.0))
            self.ps_uv.append(arr_uv)
            self.ps_sigma.append(arr_sg)

    def _solve_chen(self, dphis: list[np.ndarray]) -> np.ndarray:
        """Chen Eq. (4)/(5): orientation 0 -> u from phi_x only; orientation 1 -> v from phi_y only."""
        num = np.zeros(2)
        den = np.zeros(2)
        for i, b in enumerate(self.bands):
            k = int(self.F.orient_of[b])
            if k == 0:
                num[0] += -np.sum(self.w[i] * self.phi_x[i] * dphis[i])
                den[0] += np.sum(self.w[i] * self.phi_x[i] ** 2)
            elif k == 1:
                num[1] += -np.sum(self.w[i] * self.phi_y[i] * dphis[i])
                den[1] += np.sum(self.w[i] * self.phi_y[i] ** 2)
        return num / np.maximum(den, 1e-30)

    def result(self) -> PhaseFit:
        return PhaseFit(
            uv=np.array(self.uv), sigma=np.array(self.sigma), n_eff=np.array(self.n_eff),
            cond=np.array(self.cond), flags=np.array(self.flags, dtype=np.uint8),
            resid_rms=np.array(self.resid),
            per_scale_uv=np.array(self.ps_uv) if self.per_scale else None,
            per_scale_sigma=np.array(self.ps_sigma) if self.per_scale else None,
        )


class _TorchBackend:
    """Batched torch.fft implementation of PhaseAccumulator.push (all bands in one call)."""

    def __init__(self, acc: PhaseAccumulator):
        import torch
        self.torch = torch
        self.acc = acc
        dev = torch.device(acc.device)
        self.dev = dev
        F = acc.F
        c64 = torch.complex64
        self.B = torch.stack([torch.as_tensor(np.asarray(F.bands[b], np.complex64)) for b in acc.bands]).to(dev)   # (nb, Hp, Wp)
        self.wx = torch.as_tensor(F.omega_x).to(dev)
        self.wy = torch.as_tensor(F.omega_y).to(dev)
        self.X0 = torch.as_tensor(acc.X0).to(dev)
        self.S0 = torch.stack([torch.as_tensor(s) for s in acc.S0]).to(dev)
        self.phi_x = torch.stack([torch.as_tensor(p) for p in acc.phi_x]).to(dev)
        self.phi_y = torch.stack([torch.as_tensor(p) for p in acc.phi_y]).to(dev)
        self.w = torch.stack([torch.as_tensor(w) for w in acc.w]).to(dev)
        self.py, self.px = F.pad
        self.h, self.wd = F.shape
        self.scale_idx = [torch.as_tensor(np.flatnonzero(np.array([int(F.scale_of[b]) for b in acc.bands]) == s)).to(dev)
                          for s in range(F.n_scales)]
        self.finest = torch.as_tensor(np.flatnonzero(np.array([int(F.scale_of[b]) for b in acc.bands]) == acc.finest_scale)).to(dev)
        self.pad_mode = "reflect"

    def _crop(self, a):
        return a[..., self.py:self.py + self.h, self.px:self.px + self.wd]

    def _bands(self, X):
        return self._crop(self.torch.fft.ifft2(X[None] * self.B))

    def _dphis(self, Xb, uv):
        if uv is None:
            S0 = self.S0
        else:
            ph = self.torch.exp(-1j * (self.wx * float(uv[0]) + self.wy * float(uv[1]))).to(self.torch.complex64)
            S0 = self._bands(self.X0 * ph)
        return self.torch.angle(Xb * self.torch.conj(S0))

    def _solve(self, d):
        t = self.torch
        bx = -(self.w * self.phi_x * d).sum(dim=(1, 2))      # (nb,)
        by = -(self.w * self.phi_y * d).sum(dim=(1, 2))
        b_scale = {s: np.array([bx[i].sum().item(), by[i].sum().item()]) for s, i in enumerate(self.scale_idx) if len(i)}
        b_vec = np.array([bx.sum().item(), by.sum().item()])
        acc = self.acc
        if acc.chen_mode:
            k = np.array([int(acc.F.orient_of[b]) for b in acc.bands])
            den_x = (self.w * self.phi_x ** 2).sum(dim=(1, 2)).cpu().numpy()
            den_y = (self.w * self.phi_y ** 2).sum(dim=(1, 2)).cpu().numpy()
            bxn, byn = bx.cpu().numpy(), by.cpu().numpy()
            uv = np.array([bxn[k == 0].sum() / max(den_x[k == 0].sum(), 1e-30),
                           byn[k == 1].sum() / max(den_y[k == 1].sum(), 1e-30)])
        else:
            uv = acc.Minv @ b_vec
        return uv, b_scale

    def push(self, patch, init_uv):
        t = self.torch
        acc = self.acc
        x = t.as_tensor(np.asarray(patch, np.float32)).to(self.dev)
        xp = t.nn.functional.pad(x[None, None], (self.px, self.px, self.py, self.py), mode=self.pad_mode)[0, 0]
        X = t.fft.fft2(xp).to(t.complex64)
        Xb = self._bands(X)
        uv = np.zeros(2) if init_uv is None else np.asarray(init_uv, float)
        d = self._dphis(Xb, None if init_uv is None else uv)
        wrap_w = (self.w[self.finest] * (d[self.finest].abs() > WRAP_RAD)).sum().item()
        duv, b_scale = self._solve(d)
        uv = uv + duv
        ps_uv = {s: (uv - duv) + acc.Minv_scale[s] @ bs for s, bs in b_scale.items()}
        for _ in range(acc.n_refine):
            if not np.all(np.isfinite(uv)):
                break
            d = self._dphis(Xb, uv)
            duv, db_scale = self._solve(d)
            uv = uv + duv
            ps_uv = {s: ps_uv[s] + acc.Minv_scale[s] @ (db_scale[s] + acc.M_scale[s] @ (uv - duv - ps_uv[s])) for s in ps_uv}
        r = d + self.phi_x * float(duv[0]) + self.phi_y * float(duv[1])
        swr2 = (self.w * r * r).sum().item()
        acc._record(uv, swr2, wrap_w, ps_uv)


def solve_patch(patches: np.ndarray, spec: PyramidSpec = PyramidSpec(),
                weight_mask: np.ndarray | None = None, block_grid_origin: np.ndarray | None = None,
                per_scale: bool = True, chen_mode: bool = False, cond_max: float = 50.0,
                scales: Sequence[int] | None = None, n_refine: int = 1, device: str | None = None) -> PhaseFit:
    """Batch interface: patches (T, H, W) float32, frame 0 = reference. Implemented as
    a loop of PhaseAccumulator.push so batch == incremental by construction."""
    patches = np.asarray(patches, dtype=np.float32)
    if patches.ndim != 3:
        raise ValueError("patches must be (T, H, W)")
    origin0 = None if block_grid_origin is None else block_grid_origin[0]
    acc = PhaseAccumulator(patches[0], spec, weight_mask, cond_max, per_scale, scales, chen_mode, origin0,
                           n_refine=n_refine, device=device)
    for t in range(1, len(patches)):
        acc.push(patches[t], None if block_grid_origin is None else block_grid_origin[t])
    return acc.result()
