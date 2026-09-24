"""Does stream_frames(start=1) really start at video frame 1? Compare against the ROI cache."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uav_disp import config
from uav_disp.video_io import stream_frames, cached_roi_gray
cache = cached_roi_gray(config.ROI)
s = list(stream_frames(crop=config.ROI, gray=True, start=1, count=3))
print("frames delivered for start=1,count=3:", len(s))
for j, f in enumerate(s):
    d = [float(np.abs(f.astype(int) - cache[k].astype(int)).mean()) for k in range(0, 5)]
    print(f"  delivered[{j}] mean|diff| vs cache frames 0..4: {np.round(d, 2)}  -> matches frame {int(np.argmin(d))}")
s0 = list(stream_frames(crop=config.ROI, gray=True, count=3))
print("start=0,count=3 delivered:", len(s0), "match:", [int(np.argmin([float(np.abs(f.astype(int) - cache[k].astype(int)).mean()) for k in range(5)])) for f in s0])
