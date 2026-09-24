import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_disp import config as cfg, synth  # noqa: E402

FRAME0 = ROOT / "results" / "frame0_gray.npy"
PATCH = 192


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs data/Video.MP4 and ffmpeg")
    config.addinivalue_line("markers", "torch: needs torch")


def pytest_collection_modifyitems(config, items):
    try:
        from uav_disp.video_io import ffmpeg_exe
        ffmpeg_exe()
        no_ffmpeg = False
    except Exception:
        no_ffmpeg = True
    no_video = no_ffmpeg or not (ROOT / "data" / "Video.MP4").exists()
    try:
        import torch  # noqa: F401
        no_torch = False
    except Exception:
        no_torch = True
    for it in items:
        if "slow" in it.keywords and no_video:
            it.add_marker(pytest.mark.skip(reason="needs Video.MP4 + ffmpeg"))
        if "torch" in it.keywords and no_torch:
            it.add_marker(pytest.mark.skip(reason="needs torch"))


@pytest.fixture(scope="session")
def frame0() -> np.ndarray:
    return np.load(FRAME0)


@pytest.fixture(scope="session")
def board_source(frame0) -> np.ndarray:
    """(PATCH+2*PAD)^2 float64 source centred on the cable board, for synth.fourier_shift."""
    cx, cy = cfg.CABLE_APPROX
    h = PATCH // 2 + synth.PAD
    return frame0[cy - h:cy + h, cx - h:cx + h].astype(np.float64)


@pytest.fixture(scope="session")
def board_patch(board_source) -> np.ndarray:
    p = synth.PAD
    return board_source[p:-p, p:-p].astype(np.float32)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def shifted_stack(source: np.ndarray, shifts: np.ndarray, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """(T, H, W) float32 stack; frame t is source shifted by shifts[t] (content moves by +shift)."""
    r = np.random.default_rng(seed)
    p = synth.PAD
    out = []
    for dx, dy in shifts:
        f = synth.fourier_shift(source, dx, dy)[p:-p, p:-p]
        if noise:
            f = f + r.normal(0, noise, f.shape)
        out.append(f.astype(np.float32))
    return np.stack(out)
