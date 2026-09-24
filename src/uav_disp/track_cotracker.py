"""Foundation-model tracker: CoTracker3 (online/streaming mode) via torch.hub.

Each target is tracked in its own 512x384 crop, matching the model's internal
resolution (interp_shape) so no information is lost to resizing. The online
predictor consumes overlapping 16-frame windows with stride 8 and returns the
full track history after the last window.
"""

from __future__ import annotations

import os

import numpy as np
import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

CROP_W, CROP_H = 512, 384  # CoTracker3 interp_shape is (384, 512)


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(device: str) -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    return model.to(device).eval()


@torch.no_grad()
def track_target(
    frames,  # iterable of RGB uint8 (H, W, 3) crops of size (CROP_H, CROP_W)
    query_points_xy: np.ndarray,  # (N, 2) in crop coords, all queried at t=0
    model: torch.nn.Module,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (tracks (T, N, 2) float, visibility (T, N) bool)."""
    step = model.step  # 8
    window = 2 * step
    queries = torch.tensor(
        [[0.0, float(x), float(y)] for x, y in query_points_xy],
        dtype=torch.float32, device=device,
    )[None]

    buf: list[torch.Tensor] = []
    n_frames = 0
    is_first = True
    pred_tracks = pred_vis = None

    def to_tensor(frame: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(frame.copy()).permute(2, 0, 1).float()

    def process(chunk: list[torch.Tensor], first: bool):
        nonlocal pred_tracks, pred_vis
        video_chunk = torch.stack(chunk)[None].to(device)
        out = model(video_chunk=video_chunk, is_first_step=first, queries=queries if first else None)
        if out is not None and out[0] is not None:
            pred_tracks, pred_vis = out[0], out[1]

    for frame in frames:
        t = to_tensor(frame)
        n_frames += 1
        if is_first:
            # first call only registers the queries; feed it the first window
            buf.append(t)
            if len(buf) == window:
                process(buf, True)
                is_first = False
                process(buf, False)
                buf = buf[step:]
            continue
        buf.append(t)
        if len(buf) == window:
            process(buf, False)
            buf = buf[step:]

    if len(buf) > step or pred_tracks is None or pred_tracks.shape[1] < n_frames:
        # flush the tail: pad with repeats of the last frame to a full window
        while len(buf) < window:
            buf.append(buf[-1])
        if is_first:
            process(buf, True)
        process(buf, False)

    tracks = pred_tracks[0].cpu().numpy()[:n_frames]
    vis = pred_vis[0].cpu().numpy()[:n_frames].astype(bool)
    return tracks, vis


def crop_origin(center_xy: tuple[float, float], frame_wh: tuple[int, int]) -> tuple[int, int]:
    """Top-left of a CROP_WxCROP_H window centered on the target, clamped to the frame."""
    x = int(round(center_xy[0] - CROP_W / 2))
    y = int(round(center_xy[1] - CROP_H / 2))
    x = max(0, min(x, frame_wh[0] - CROP_W))
    y = max(0, min(y, frame_wh[1] - CROP_H))
    return x, y
