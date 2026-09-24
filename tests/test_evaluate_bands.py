import numpy as np

from uav_disp import config, evaluate, postprocess


def test_band_scores_synthetic(rng):
    fps, n = 50.0, 3000
    t = np.arange(n) / fps
    tones = {1.5: 0.3, 7.5: 1.0, 18.0: 0.2}
    full = sum(a * np.sin(2 * np.pi * f * t) for f, a in tones.items())
    noise = rng.normal(0, 0.01, n)
    ref = full + noise
    vis = full - tones[18.0] * np.sin(2 * np.pi * 18.0 * t) + rng.normal(0, 0.01, n)   # 18 Hz tone missing
    bs = evaluate.band_scores(vis, ref, fps=fps)
    assert bs["coherence"][0] > 0.95 and bs["coherence"][1] > 0.95
    assert bs["coherence"][2] < 0.3
    assert abs(bs["rmse"][2] - tones[18.0] / np.sqrt(2)) / (tones[18.0] / np.sqrt(2)) < 0.05
    assert bs["rmse"][1] < 0.02


def test_displacement_mm_full_band_passthrough(rng):
    cable = rng.normal(0, 1, (500, 2)).cumsum(axis=0)
    ref = np.zeros_like(cable)
    ang = 0.4
    mm = postprocess.displacement_mm(cable, ref, 31.0, ang, highpass_hz=None, lowpass_hz=None)
    normal = np.array([np.sin(ang), -np.cos(ang)])
    exp = (cable - cable.mean(axis=0)) @ normal * (config.SQUARE_MM / 31.0)
    assert np.allclose(mm, exp - exp.mean(), atol=1e-9)
