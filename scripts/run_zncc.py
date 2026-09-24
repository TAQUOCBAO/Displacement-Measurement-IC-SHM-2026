"""Run the ZNCC baseline tracker over the UAV video.

Usage: python scripts/run_zncc.py [n_frames]   (default: all)
Writes results/tracks_zncc[_slice].npz with cable/ref center tracks in
ROI-local coordinates plus board geometry from frame 0.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uav_disp import config
from uav_disp.detect import find_board
from uav_disp.track_zncc import track
from uav_disp.video_io import stream_frames


def main() -> None:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else None
    frames = stream_frames(crop=config.ROI, count=count, gray=True)
    first = next(frames)
    cable_b = find_board(first, config.CABLE_APPROX, config.APPROX_SQUARE_PX)
    ref_b = find_board(first, config.REF_APPROX, config.APPROX_SQUARE_PX)

    def chain():
        yield first
        yield from frames

    res = track(chain(), {"cable": tuple(cable_b.center), "ref": tuple(ref_b.center)})
    res["cable_square_px"] = np.array(cable_b.square_px)
    res["cable_angle_deg"] = np.array(cable_b.angle_deg)
    res["ref_square_px"] = np.array(ref_b.square_px)

    suffix = "_slice" if count else ""
    path = f"results/tracks_zncc{suffix}.npz"
    np.savez(path, **res)
    print(f"tracked {len(res['cable'])} frames -> {path} | min ZNCC peaks: "
          f"cable {res['cable_peak'][1:].min():.3f}, ref {res['ref_peak'][1:].min():.3f}")


if __name__ == "__main__":
    main()
