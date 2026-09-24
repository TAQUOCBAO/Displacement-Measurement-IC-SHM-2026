"""Run CoTracker3 online tracking of both targets over the UAV video.

Usage: python scripts/run_cotracker.py [n_frames]   (default: all)
Writes results/tracks_cotracker[_slice].npz with per-target point tracks in
ROI-local coordinates, plus board geometry from frame 0.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uav_disp import config
from uav_disp.detect import find_board
from uav_disp.track_cotracker import CROP_H, CROP_W, crop_origin, load_model, pick_device, track_target
from uav_disp.video_io import stream_frames


ZOOM = 2  # track in a 2x lanczos-upscaled crop: CoTracker noise is ~constant in
          # model pixels, so zooming halves it in real pixels
N_SUPPORT = 48  # extra background query points; they stabilize the joint
                # transformer but are excluded from the output tracks


def main() -> None:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else None
    device = pick_device()
    print("device:", device)
    model = load_model(device)

    gray0 = next(stream_frames(crop=config.ROI, count=1, gray=True))
    boards = {
        "cable": find_board(gray0, config.CABLE_APPROX, config.APPROX_SQUARE_PX),
        "ref": find_board(gray0, config.REF_APPROX, config.APPROX_SQUARE_PX),
    }

    roi_x, roi_y, roi_w, roi_h = config.ROI
    src_w, src_h = CROP_W // ZOOM, CROP_H // ZOOM
    out: dict[str, np.ndarray] = {}
    for name, board in boards.items():
        ox = int(round(board.center[0] - src_w / 2))
        oy = int(round(board.center[1] - src_h / 2))
        target_q = (board.query_points() - np.array([ox, oy])) * ZOOM
        n_target = len(target_q)
        gx, gy = np.meshgrid(np.linspace(30, CROP_W - 30, 8), np.linspace(30, CROP_H - 30, 6))
        grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
        keep = np.linalg.norm(grid - target_q.mean(axis=0), axis=1) > ZOOM * board.square_px * 2.5
        queries = np.vstack([target_q, grid[keep][:N_SUPPORT]])
        # stream this target's own crop, upscaled to the model's native 512x384
        frames = stream_frames(
            crop=(roi_x + ox, roi_y + oy, src_w, src_h),
            count=count, gray=False, scale_to=(CROP_W, CROP_H),
        )
        t0 = time.time()
        tracks, vis = track_target(frames, queries, model, device)
        tracks, vis = tracks[:, :n_target], vis[:, :n_target]  # drop support points
        print(f"{name}: {tracks.shape[0]} frames, {tracks.shape[1]} pts, "
              f"{time.time() - t0:.0f}s, vis rate {vis.mean():.3f}")
        out[name] = tracks / ZOOM + np.array([ox, oy])  # back to ROI-local coords
        out[f"{name}_vis"] = vis

    out["cable_square_px"] = np.array(boards["cable"].square_px)
    out["cable_angle_deg"] = np.array(boards["cable"].angle_deg)
    out["ref_square_px"] = np.array(boards["ref"].square_px)

    suffix = "_slice" if count else ""
    path = f"results/tracks_cotracker{suffix}.npz"
    np.savez(path, **out)
    print("saved", path)


if __name__ == "__main__":
    main()
