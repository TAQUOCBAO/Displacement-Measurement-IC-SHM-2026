import numpy as np
import pytest

from uav_disp import synth
from uav_disp.steerable import (G_F1, G_F2, H_F1, PyramidSpec, band_gradient, build_filters, decompose,
                                decompose_band, frame_identity, g2h2_bank, g2h2_decompose, padded_fft,
                                reconstruct)

SPECS = [PyramidSpec(4, 2, True), PyramidSpec(4, 4, True), PyramidSpec(3, 2, False)]


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("shape", [(192, 192), (128, 160)])
def test_tight_frame(spec, shape):
    F = build_filters(shape, spec)
    assert np.allclose(frame_identity(F), 1.0, atol=1e-6)


def test_perfect_reconstruction_padded(rng):
    F = build_filters((192, 192), PyramidSpec())
    img = rng.normal(0, 30, (192, 192)).astype(np.float32)
    parts = decompose(img, F, crop=False)
    rec = reconstruct(parts, F)
    ref = np.pad(img, F.pad, mode="reflect")
    assert np.max(np.abs(rec - ref)) < 1e-3       # float32 FFT round-trip


def test_reconstruction_real_patch(board_patch):
    F = build_filters(board_patch.shape, PyramidSpec())
    rec = reconstruct(decompose(board_patch, F), F)
    m = 24
    err = np.abs(rec - board_patch)[m:-m, m:-m].max()
    assert err < 3e-3 * np.ptp(board_patch)   # re-padding the long-support lo band; padded round-trip is exact


def test_one_sided_analytic():
    x = np.arange(192)[None, :] * np.ones((192, 1))
    img = np.cos(61 * np.pi / 192 * (x + 0.5)).astype(np.float32)   # reflection-seamless plane wave, ~1 rad/px
    F = build_filters(img.shape, PyramidSpec(4, 2, True))
    X = padded_fft(img, F)
    # pick the band (orientation 0) whose radial mask is largest at r = 0.35 rad/px... scan all
    amps = []
    for b in range(F.n_bands):
        S = decompose_band(X, F, b)
        amps.append(np.abs(S)[32:-32, 32:-32])
    e0 = [a.mean() for b, a in enumerate(amps) if F.orient_of[b] == 0]
    e1 = [a.mean() for b, a in enumerate(amps) if F.orient_of[b] == 1]
    b_best = int(np.argmax(e0))
    a = [amps[b] for b in range(F.n_bands) if F.orient_of[b] == 0][b_best]
    assert a.std() / a.mean() < 2e-3                 # analytic signal: constant amplitude (float32 FFT + leakage)
    assert max(e1) < 1e-3 * max(e0)                  # pure x-variation: no energy in the y band


def test_band_gradient_is_exact_on_plane_wave():
    k = 61 * np.pi / 192
    x = np.arange(192)[None, :] * np.ones((192, 1))
    img = np.cos(k * (x + 0.5)).astype(np.float32)
    F = build_filters(img.shape, PyramidSpec(4, 2, True))
    X = padded_fft(img, F)
    b = int(np.argmax([np.abs(decompose_band(X, F, i))[48:-48, 48:-48].mean() for i in range(F.n_bands)]))
    S = decompose_band(X, F, b)
    gx, gy = band_gradient(X, F, b)
    m = slice(32, -32)
    assert np.abs(gx - 1j * k * S)[m, m].max() < 2e-3 * np.abs(S)[m, m].max()
    assert np.abs(gy)[m, m].max() < 2e-3 * np.abs(S)[m, m].max()


def test_g2h2_taps():
    assert np.allclose(G_F1, G_F1[::-1]) and np.allclose(G_F2, G_F2[::-1])
    assert np.allclose(H_F1, -H_F1[::-1])
    x = np.arange(-4, 5) * 0.67
    assert np.allclose(0.9213 * (2 * x ** 2 - 1) * np.exp(-x ** 2), G_F1, atol=6e-5)   # sign pattern + + + - - - + + +
    assert np.allclose(np.exp(-x ** 2), G_F2, atol=6e-5)
    bank = g2h2_bank()   # runs the quadrature/DC asserts
    assert set(bank) == {"0", "pi/2"} and bank["0"][0].shape == (9, 9)


def test_g2h2_phase_shift_law(board_source):
    p = synth.PAD
    a = board_source[p:-p, p:-p].astype(np.float32)
    b = synth.fourier_shift(board_source, 0.25, 0.0)[p:-p, p:-p].astype(np.float32)
    S0, S1 = g2h2_decompose(a)[0], g2h2_decompose(b)[0]
    F = build_filters(a.shape, PyramidSpec(bank="g2h2"))
    X0 = padded_fft(a, F)
    gx, _ = band_gradient(X0, F, 0)
    phi_x = np.imag(np.conj(S0) * gx) / np.abs(S0) ** 2
    dphi = np.angle(S1 * np.conj(S0))
    m = (np.abs(S0) > 0.3 * np.abs(S0).max()) & (np.abs(phi_x) > 0.3)
    m[:16] = m[-16:] = False
    m[:, :16] = m[:, -16:] = False
    est = -np.median(dphi[m] / phi_x[m])
    assert abs(est - 0.25) < 0.05 * 0.25
