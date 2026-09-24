"""Synthetic sub-pixel benchmark: real imagery, exact known motion.

A single real frame crop is translated by a known sub-pixel trajectory via
Fourier phase shifting, giving ground truth that is exact by construction.
The source crop is padded and center-cropped after shifting so FFT wraparound
never enters the evaluated region, and small iid Gaussian noise is added per
frame so trackers cannot lock onto a repeated noise realization.
"""

from __future__ import annotations

import numpy as np

PAD = 8  # px margin discarded after shifting; must exceed max |shift|


def fourier_shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate a (H, W) or (H, W, C) float image by (dx, dy) via FFT phase."""
    h, w = img.shape[:2]
    fx = np.fft.fftfreq(w)[None, :]
    fy = np.fft.fftfreq(h)[:, None]
    phase = np.exp(-2j * np.pi * (fx * dx + fy * dy))
    if img.ndim == 2:
        return np.real(np.fft.ifft2(np.fft.fft2(img) * phase))
    return np.stack(
        [np.real(np.fft.ifft2(np.fft.fft2(img[..., c]) * phase)) for c in range(img.shape[2])],
        axis=-1,
    )


def sinusoid_traj(n: int, amp_px: float, freq_hz: float, fps: float,
                  direction: np.ndarray) -> np.ndarray:
    """(n, 2) trajectory: amp * sin(2 pi f t) along a unit direction vector."""
    t = np.arange(n) / fps
    s = amp_px * np.sin(2 * np.pi * freq_hz * t)
    return s[:, None] * direction[None, :]


def make_sequence(
    source: np.ndarray,            # (H+2*PAD, W+2*PAD[, 3]) uint8 source crop
    traj_xy: np.ndarray,           # (n, 2) shifts in px, |shift| < PAD
    noise_sigma: float = 1.5,
    seed: int = 0,
):
    """Yield n uint8 frames of size (H, W[, 3]), frame i shifted by traj_xy[i]."""
    assert np.abs(traj_xy).max() < PAD, "trajectory exceeds wraparound margin"
    rng = np.random.default_rng(seed)
    src = source.astype(np.float64)
    for dx, dy in traj_xy:
        f = fourier_shift(src, dx, dy)
        f = f[PAD:-PAD, PAD:-PAD]
        f = f + rng.normal(0.0, noise_sigma, f.shape)
        yield np.clip(f, 0, 255).astype(np.uint8)


def h264_roundtrip(frames, fps: float, tmp_path, bits_per_px: float = 0.4):
    """Encode the sequence with libx264 at the real video's ~bpp and decode back.

    Codec artifacts are the dominant real-world degradation for sub-pixel
    tracking (they are block-aligned and do NOT translate with content); the
    clean Fourier-shift benchmark alone is a lower bound.
    """
    import subprocess

    from .video_io import ffmpeg_exe, stream_frames

    frames = list(frames)
    h, w = frames[0].shape[:2]
    bitrate = int(w * h * fps * bits_per_px)
    cmd = [ffmpeg_exe(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-b:v", str(bitrate), "-pix_fmt", "yuv420p", str(tmp_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    proc.wait()
    yield from stream_frames(tmp_path, crop=(0, 0, w, h), count=len(frames))


def upscale(frame: np.ndarray, zoom: float) -> np.ndarray:
    """Bicubic upscale via torch (matches the pipeline's interpolated-input regime)."""
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(frame.astype(np.float32))
    t = t[None, None] if frame.ndim == 2 else t.permute(2, 0, 1)[None]
    out_h, out_w = int(round(frame.shape[0] * zoom)), int(round(frame.shape[1] * zoom))
    up = F.interpolate(t, size=(out_h, out_w), mode="bicubic", align_corners=False)
    up = up.clamp(0, 255)[0]
    up = up[0] if frame.ndim == 2 else up.permute(1, 2, 0)
    return up.numpy().astype(np.uint8)
