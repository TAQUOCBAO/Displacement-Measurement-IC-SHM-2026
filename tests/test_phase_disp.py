import time

import numpy as np
import pytest

from conftest import shifted_stack
from uav_disp.phase_disp import (FLAG_WRAP, PhaseAccumulator, block_weight, coarse_cuts, gaussian_taper,
                                 solve_patch)
from uav_disp.steerable import PyramidSpec

SPEC = PyramidSpec()
MASK = np.zeros((192, 192), np.float32)
MASK[44:148, 44:148] = 1.0     # board + a little surround; keeps reflect-padded borders out


def test_sign_convention_matches_fourier_shift(board_source):
    st = shifted_stack(board_source, np.array([[0, 0], [0.3, 0], [0, -0.7]]))
    fit = solve_patch(st, SPEC, weight_mask=MASK)
    assert np.allclose(fit.uv[1], [0.3, 0.0], atol=1e-3)
    assert np.allclose(fit.uv[2], [0.0, -0.7], atol=1e-3)
    assert np.allclose(fit.uv[0], 0.0)


def test_pure_shift_recovery_noise_free(board_source, rng):
    shifts = np.vstack([[0, 0], rng.uniform(-1, 1, (19, 2))])
    fit = solve_patch(shifted_stack(board_source, shifts), SPEC, weight_mask=MASK)
    assert np.sqrt(np.mean((fit.uv - shifts) ** 2, axis=0)).max() < 1e-3
    assert not np.any(fit.flags)


def test_chen_mode_recovers_shift(board_source, rng):
    shifts = np.vstack([[0, 0], rng.uniform(-1, 1, (10, 2))])
    fit = solve_patch(shifted_stack(board_source, shifts), SPEC, weight_mask=MASK, chen_mode=True)
    # Chen's per-orientation Eq. (4)/(5) approximation: ~10x worse than the joint Eq. (3) solve here
    assert np.sqrt(np.mean((fit.uv - shifts) ** 2, axis=0)).max() < 2e-2


def test_g2h2_bank_recovers_shift(board_source, rng):
    shifts = np.vstack([[0, 0], rng.uniform(-0.8, 0.8, (10, 2))])
    fit = solve_patch(shifted_stack(board_source, shifts), PyramidSpec(bank="g2h2"), weight_mask=MASK)
    assert np.sqrt(np.mean((fit.uv - shifts) ** 2, axis=0)).max() < 5e-3


def test_per_scale_agreement(board_source, rng):
    shifts = np.vstack([[0, 0], rng.uniform(-1, 1, (10, 2))])
    fit = solve_patch(shifted_stack(board_source, shifts), SPEC, weight_mask=MASK)
    for s in range(SPEC.n_scales):
        err = np.sqrt(np.mean((fit.per_scale_uv[:, s] - shifts) ** 2, axis=0)).max()
        assert err < 5e-3, (s, err)


def test_sigma_calibration(board_source):
    errs, sigs = [], []
    for seed in range(60):
        st = shifted_stack(board_source, np.array([[0, 0], [0.4, -0.2]]), noise=2.0, seed=seed)
        fit = solve_patch(st, SPEC, weight_mask=MASK, per_scale=False)
        errs.append(fit.uv[1] - [0.4, -0.2])
        sigs.append(fit.sigma[1])
    emp = np.std(np.array(errs), axis=0)
    pred = np.mean(np.array(sigs), axis=0)
    ratio = emp / pred
    assert np.all((ratio > 0.7) & (ratio < 1.4)), (emp, pred)


def test_wrap_limit_flagged(board_source):
    st = shifted_stack(board_source, np.array([[0, 0], [2.5, 0]]))
    fit = solve_patch(st, SPEC, weight_mask=MASK, n_refine=0)
    assert fit.flags[1] & FLAG_WRAP
    fit2 = solve_patch(st, SPEC, weight_mask=MASK, scales=[2, 3])
    assert abs(fit2.uv[1, 0] - 2.5) < 0.02 and abs(fit2.uv[1, 1]) < 0.02
    # with a coarse prior (linearisation point) all scales are usable and no wrap flag is raised
    acc = PhaseAccumulator(st[0], SPEC, weight_mask=MASK)
    acc.push(st[1], init_uv=(2.3, 0.15))
    f3 = acc.result()
    assert not (f3.flags[1] & FLAG_WRAP) and abs(f3.uv[1, 0] - 2.5) < 2e-3 and abs(f3.uv[1, 1]) < 2e-3


def test_weight_mask_zero_outside_board(board_patch):
    h, w = board_patch.shape
    m = np.zeros((h, w), np.float32)
    m[60:132, 60:132] = 1
    acc = PhaseAccumulator(board_patch, SPEC, weight_mask=m, per_scale=False)
    assert 0 < acc.n_eff_val < 4 * 72 * 72     # bounded by (n_bands-weighted) board pixels


def test_block_weight_geometry():
    w = block_weight((1600 + 37, 600 + 52), 40, 40)
    xs = (np.arange(40) + 1637) % 16
    ys = (np.arange(40) + 652) % 16
    near = lambda a: (a <= 1) | (a >= 15)
    exp = np.where(near(ys)[:, None] | near(xs)[None, :], 0.25, 1.0)
    assert np.array_equal(w, exp.astype(np.float32))


def test_coarse_cuts_piecewise_constant(rng):
    t = np.arange(1000)
    track = np.stack([10 + 0.01 * t + rng.normal(0, 0.3, 1000), 20 + rng.normal(0, 0.3, 1000)], axis=1)
    cuts, ch = coarse_cuts(track, 1.0, 51)
    assert cuts.dtype.kind == "i"
    assert np.all(np.diff(cuts[:, 1]) == 0)                       # no drift in y -> never re-cut
    assert 8 <= len(ch) <= 11                                      # ~10 px of x drift -> ~10 re-cuts
    assert np.all(np.abs(cuts[:, 0] - (10 + 0.01 * t)) < 1.6)


def test_incremental_equals_batch(board_source, rng):
    shifts = np.vstack([[0, 0], rng.uniform(-1, 1, (5, 2))])
    st = shifted_stack(board_source, shifts)
    batch = solve_patch(st, SPEC, weight_mask=MASK)
    acc = PhaseAccumulator(st[0], SPEC, weight_mask=MASK)
    for p in st[1:]:
        acc.push(p)
    inc = acc.result()
    assert np.allclose(batch.uv, inc.uv, atol=1e-9) and np.allclose(batch.sigma, inc.sigma, atol=1e-9)


@pytest.mark.slow
def test_runtime_budget(board_patch):
    """Plan §9 budget: < 60 s per target for the full video on the CUDA backend; the numpy
    path is allowed 150 s (measured ~120 s on 8 cores; 8 bands x 2 GN passes of 256^2 ffts)."""
    device, budget = None, 150.0
    try:
        import torch
        if torch.cuda.is_available():
            device, budget = "cuda", 60.0
    except ImportError:
        pass
    acc = PhaseAccumulator(board_patch, SPEC, device=device)
    n = 300
    acc.push(board_patch)  # warm-up (cuda init / plan caching) excluded from the timing
    t = time.perf_counter()
    for _ in range(n):
        acc.push(board_patch)
    per = (time.perf_counter() - t) / n
    assert per * 3033 < budget, f"{per*3033:.1f} s projected for the full video ({device or 'numpy'}, budget {budget:.0f} s)"
