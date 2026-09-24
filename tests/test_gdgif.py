import time

import numpy as np
from scipy import ndimage

from uav_disp.gdgif import _row_box_mean, edge_weights, reference_loop, row_filter


def test_constant_image_identity():
    X = np.full((16, 64), 3.0, np.float32)
    assert np.allclose(row_filter(X), X, atol=1e-6)


def test_gamma_Gamma_ranges(rng):
    u = rng.random((32, 128))
    chi, gamma, Gamma = edge_weights(u)
    assert np.all((gamma > 0) & (gamma < 1))
    assert np.all(Gamma > 0)
    i = np.unravel_index(np.argmin(np.abs(chi - chi.mean())), chi.shape)
    assert abs(Gamma[i] - 1) < 0.05


def test_row_box_mean_matches_uniform_filter(rng):
    a = rng.random((8, 50))
    for h in (3, 16):
        ref = ndimage.uniform_filter1d(a, h, axis=1, mode="mirror")   # numpy 'reflect' == scipy 'mirror'
        assert np.allclose(_row_box_mean(a, h), ref, atol=1e-9)


def test_vectorised_equals_reference_loop(rng):
    X = rng.random((8, 40))
    X[:, 20:] += 1.0
    assert np.allclose(row_filter(X), reference_loop(X), atol=1e-6)


def test_stripe_suppression_edge_preservation():
    H, W = 64, 256
    yy, xx = np.mgrid[:H, :W]
    edge = (xx >= 128).astype(float)
    # Yang's target artifact: ringing / double-edge oscillation ALONG the row (x), period 4 px
    stripe = 0.1 * np.sin(2 * np.pi * xx / 4)
    img = edge + stripe
    out = row_filter(img.astype(np.float32))

    def stripe_energy(a):
        f = np.fft.fft2(a - a.mean())
        p = np.abs(f) ** 2
        ky = np.fft.fftfreq(H)
        kx = np.fft.fftfreq(W)
        m = (np.abs(ky[:, None]) < 1e-9) & (np.abs(np.abs(kx[None, :]) - 0.25) < 1e-9)
        return p[m].sum()

    assert 10 * np.log10(stripe_energy(img) / stripe_energy(out)) >= 8.0    # measured 9.6 dB at h=16, mu=0.022
    step_in = img[:, 136:152].mean() - img[:, 104:120].mean()     # edge contrast across the step
    step_out = out[:, 136:152].mean() - out[:, 104:120].mean()
    assert abs(step_out - step_in) / step_in < 0.05


def test_runtime(rng):
    X = rng.random((192, 192)).astype(np.float32)
    row_filter(X)
    t = time.perf_counter(); row_filter(X); dt = time.perf_counter() - t
    assert dt < 0.05
    X = rng.random((1216, 1344)).astype(np.float32)
    t = time.perf_counter(); row_filter(X); dt = time.perf_counter() - t
    assert dt < 1.0
