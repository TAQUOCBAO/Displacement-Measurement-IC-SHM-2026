import numpy as np
import pytest

from conftest import ROOT, shifted_stack
from uav_disp import synth
from uav_disp.magnify import canny_edges, magnify_sequence, ssim_psnr, write_video
from uav_disp.phase_disp import solve_patch
from uav_disp.steerable import PyramidSpec


def _blob(h, w, cx, cy, s=1.5):
    yy, xx = np.mgrid[:h, :w]
    return 200 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))


def _centroid(img, cx, cy, r=6):
    sub = img[cy - r:cy + r + 1, cx - r:cx + r + 1].astype(float)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (sub * xx).sum() / sub.sum() + cx, (sub * yy).sum() / sub.sum() + cy


def test_two_impulse_multidirectional():
    # feature A moves +1 in y, feature B moves -1 in x; a global-FFT magnifier cannot separate them
    f0 = _blob(48, 48, 14, 14) + _blob(48, 48, 34, 34)
    f1 = _blob(48, 48, 14, 15) + _blob(48, 48, 33, 34)
    # 6 scales so the (un-magnified) lo residual holds little of the blob energy: 2.4 of the ideal 3 px
    out = magnify_sequence([f0, f1], band_hz=None, alpha=2.0, spec=PyramidSpec(6, 4, True, pad=16),
                           gdgif_on=False, phase_sigma_px=0)
    ax0, ay0 = _centroid(out[0], 14, 14)
    ax1, ay1 = _centroid(out[1], 14, 15)
    bx0, by0 = _centroid(out[0], 34, 34)
    bx1, by1 = _centroid(out[1], 33, 34)
    dA, dB = (ax1 - ax0, ay1 - ay0), (bx1 - bx0, by1 - by0)
    assert dA[1] > 1.8 and abs(dA[0]) < 0.25 * dA[1], dA        # ~3 px along +y only
    assert dB[0] < -1.8 and abs(dB[1]) < 0.25 * abs(dB[0]), dB  # ~3 px along -x only


def test_alpha_zero_is_identity(board_patch):
    st = np.stack([board_patch] * 4)
    out = magnify_sequence(st, band_hz=None, alpha=0.0, gdgif_on=False, phase_sigma_px=0)
    assert np.max(np.abs(out.astype(int) - np.rint(board_patch).astype(int))) <= 1


def test_magnified_amplitude_scales_linearly(board_source):
    fps, n, amp, f0, alpha = 50.0, 120, 0.05, 4.0, 20.0
    traj = synth.sinusoid_traj(n, amp, f0, fps, np.array([1.0, 0.0]))
    st = shifted_stack(board_source, traj)
    out = magnify_sequence(st, band_hz=(3.5, 4.5), alpha=alpha, fps=fps, gdgif_on=False,
                           spec=PyramidSpec(6, 4, True))
    m = np.zeros((192, 192), np.float32)
    m[44:148, 44:148] = 1
    fit = solve_patch(out.astype(np.float32), PyramidSpec(), weight_mask=m, per_scale=False)
    u = fit.uv[10:-10, 0]
    t = np.arange(n)[10:-10] / fps
    A = np.stack([np.sin(2 * np.pi * f0 * t), np.cos(2 * np.pi * f0 * t)], axis=1)
    c, *_ = np.linalg.lstsq(A, u - u.mean(), rcond=None)
    meas = np.hypot(*c)
    # the lo residual is not magnified, so PVMM under-shoots (1+alpha)*amp by the lo-band energy share
    assert 0.8 < meas / (amp * (alpha + 1)) < 1.1, meas


def test_ssim_identity_and_noise(rng, board_patch):
    a = np.rint(board_patch).astype(np.uint8)
    s, p = ssim_psnr(a, a)
    assert s == pytest.approx(1.0) and p == np.inf
    b = np.clip(a + rng.normal(0, 5, a.shape), 0, 255)
    _, p = ssim_psnr(a, b)
    assert abs(p - 10 * np.log10(255 ** 2 / 25)) < 0.6


def test_canny_on_step_edge():
    img = np.zeros((32, 64))
    img[:, 32:] = 100
    e = canny_edges(img)
    cols = np.flatnonzero(e[8:-8].any(axis=0))
    assert len(cols) <= 2 and np.all(np.abs(cols - 31.5) < 1.5)


@pytest.mark.slow
def test_write_video_roundtrip(tmp_path, board_patch):
    from uav_disp.video_io import stream_frames
    frames = np.stack([np.rint(board_patch).astype(np.uint8)] * 20)
    p = tmp_path / "t.mp4"
    write_video(frames, p)
    back = np.stack(list(stream_frames(p, crop=(0, 0, 192, 192), gray=True, count=20)))
    assert np.mean(np.abs(back.astype(float) - frames)) < 3
