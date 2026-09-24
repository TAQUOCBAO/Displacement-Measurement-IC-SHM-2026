"""Regression: ffmpeg trim used to deliver frame `start` twice (and one frame past `count`)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_disp import config  # noqa: E402
from uav_disp.video_io import VIDEO_PATH, stream_frames  # noqa: E402


@pytest.mark.slow
def test_stream_start_count_exact():
    if not VIDEO_PATH.exists():
        pytest.skip("needs data/Video.MP4")
    ref = list(stream_frames(crop=config.ROI, gray=True, count=4))          # frames 0..3
    got = list(stream_frames(crop=config.ROI, gray=True, start=1, count=3))  # must be frames 1..3
    assert len(got) == 3
    for g, r in zip(got, ref[1:]):
        assert np.array_equal(g, r)
    assert not np.array_equal(got[0], got[1])
