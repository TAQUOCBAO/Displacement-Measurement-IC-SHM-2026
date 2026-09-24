"""Train the scene-adaptive refinement net and refine the CoTracker coarse tracks.

Usage: python scripts/run_refined.py
Needs results/tracks_cotracker.npz (coarse tracks + frame-0 query positions).
Writes results/tracks_refined.npz (same layout as tracks_cotracker.npz) and
models/refine_net.pt. Score with: python scripts/run_pipeline.py refined
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_disp import config, refine_net
from uav_disp.refine_net import PATCH, SRC, RefineNet, normalize, refine_frame, train
from uav_disp.track_cotracker import pick_device
from uav_disp.video_io import stream_frames

TRAIN_FRAMES = [0, 750, 1500, 2250]


def grab_gray(idx: int) -> np.ndarray:
    return next(stream_frames(crop=config.ROI, start=idx, count=1, gray=True))


def cut(gray: np.ndarray, center_xy: np.ndarray, side: int) -> np.ndarray:
    h = side // 2
    x, y = np.rint(center_xy).astype(int)
    return gray[y - h : y + h, x - h : x + h].astype(np.float64)


def main():
    import os
    device = os.environ.get("UAV_DEVICE") or pick_device()
    ct = np.load(ROOT / "results/tracks_cotracker.npz")
    coarse = {"cable": ct["cable"], "ref": ct["ref"]}
    q = {n: coarse[n][0] for n in coarse}          # frame-0 sub-pixel queries
    q_frac = {n: q[n] - np.rint(q[n]) for n in q}

    print("collecting training patches from frames", TRAIN_FRAMES)
    sources = []
    for idx in TRAIN_FRAMES:
        g = grab_gray(idx)
        for n in coarse:
            for p in coarse[n][idx]:
                sources.append(cut(g, p, SRC))
    print(f"training refinement net on {len(sources)} source patches ({device})")
    model, val_rmse = train(sources, device)
    (ROOT / "models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), ROOT / "models/refine_net.pt")

    gray0 = grab_gray(0)
    templates = {
        n: normalize(torch.from_numpy(np.stack([cut(gray0, p, PATCH) for p in q[n]])))
        for n in coarse
    }

    n_frames = len(ct["cable"])
    out = {n: np.empty_like(coarse[n]) for n in coarse}
    model.eval()
    t0 = time.time()
    for i, g in enumerate(stream_frames(crop=config.ROI, gray=True)):
        for n in coarse:
            out[n][i] = refine_frame(model, device, g, templates[n], coarse[n][i], q_frac[n])
        if i % 500 == 0:
            print(f"  frame {i}/{n_frames} ({time.time() - t0:.0f}s)")

    np.savez(ROOT / "results/tracks_refined.npz",
             cable=out["cable"], ref=out["ref"],
             cable_square_px=ct["cable_square_px"],
             cable_angle_deg=ct["cable_angle_deg"],
             ref_square_px=ct["ref_square_px"],
             val_rmse_px=np.array(val_rmse))
    print("wrote results/tracks_refined.npz")


if __name__ == "__main__":
    main()
