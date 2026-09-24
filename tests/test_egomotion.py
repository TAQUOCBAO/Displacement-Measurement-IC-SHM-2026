import numpy as np
import pytest

from conftest import ROOT
from uav_disp.egomotion import (affine_fit, estimate_tau_from_tiles, evaluate_field, fractional_delay,
                                staticness_screen)


def test_affine_fit_recovers_synthetic_field(rng):
    xy = rng.uniform(200, 3600, (40, 2))
    A = np.array([[1e-4, -3e-4], [2e-4, 5e-5]])
    t = np.array([0.31, -0.12])
    c = xy.mean(axis=0)
    d = (xy - c) @ A.T + t + rng.normal(0, 0.01, (40, 2))
    d[:3] += 0.5
    p, r, w = affine_fit(xy, d, np.full((40, 2), 0.01))
    assert np.allclose(p[:4], A.ravel(), atol=5e-6)     # noise 0.01 px over a ~3400 px baseline
    assert np.allclose(p[4:], t, atol=5e-3)
    assert w[:3].max() < 0.5 < w[3:].min()
    assert np.allclose(evaluate_field(p, c, c)[0], p[4:])


def test_translation_model(rng):
    xy = rng.uniform(0, 1000, (10, 2))
    d = np.tile([0.2, -0.4], (10, 1)) + rng.normal(0, 0.005, (10, 2))
    p, _, _ = affine_fit(xy, d, model="translation")
    assert np.allclose(p[4:], [0.2, -0.4], atol=5e-3) and np.all(p[:4] == 0)


def test_fractional_delay_sinusoid():
    t = np.arange(600) / 50.0
    x = np.sin(2 * np.pi * 7.5 * t)
    y = fractional_delay(x, 0.37)
    ref = np.sin(2 * np.pi * 7.5 * (t - 0.37 / 50.0))
    assert np.abs(y - ref)[20:-20].max() < 2e-3
    y2 = fractional_delay(x, -2.6)
    ref2 = np.sin(2 * np.pi * 7.5 * (t + 2.6 / 50.0))
    assert np.abs(y2 - ref2)[20:-20].max() < 2e-3


def test_tau_from_tiles_synthetic(rng):
    fps, T = 50.0, 3000
    tau = 15e-3 / 2160
    tile_xy = np.stack([rng.uniform(0, 3800, 30), rng.uniform(0, 2160, 30)], axis=1)
    t = np.arange(T) / fps
    jitter = sum(rng.uniform(0.2, 0.5) * np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f in (4.1, 6.3, 9.7, 12.5))
    disp = np.zeros((T, 30, 2))
    for k in range(30):
        disp[:, k, 0] = fractional_delay(jitter, -tau * tile_xy[k, 1] * fps) + rng.normal(0, 0.01, T)
    est, r2 = estimate_tau_from_tiles(disp, tile_xy, fps)
    assert abs(est - tau) / tau < 0.1 and r2 > 0.5


def test_staticness_screen(rng):
    disp = rng.normal(0, 0.01, (500, 12, 2)) + rng.normal(0, 1, (500, 1, 2))
    disp[:, 5] += np.sin(np.arange(500) / 7)[:, None]
    keep = staticness_screen(disp)
    assert not keep[5] and keep.sum() == 11
