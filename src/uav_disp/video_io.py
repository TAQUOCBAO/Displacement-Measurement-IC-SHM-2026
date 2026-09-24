"""Frame streaming from the UAV video via an ffmpeg rawvideo pipe (no cv2).

The 4K video is 1.2 GB / 3033 frames, so frames are decoded once and streamed.
A single crop rectangle covering all regions of interest keeps one decode pass;
callers slice sub-windows out of the yielded array in numpy.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    """ffmpeg binary: $FFMPEG_BIN, else imageio-ffmpeg's static build (has libx264), else PATH.

    The cluster's `module load ffmpeg/4.2.2` build lacks libx264, which the H.264
    round-trip benchmark and the magnification writer need."""
    if os.environ.get("FFMPEG_BIN"):
        return os.environ["FFMPEG_BIN"]
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("no ffmpeg found: pip install imageio-ffmpeg, or module load ffmpeg, or set FFMPEG_BIN")
    return exe

VIDEO_PATH = Path(__file__).resolve().parents[2] / "data" / "Video.MP4"
FPS = 50.0
FRAME_W, FRAME_H = 3840, 2160
N_FRAMES = 3033


def stream_frames(
    path: str | Path = VIDEO_PATH,
    crop: tuple[int, int, int, int] | None = None,  # (x, y, w, h)
    gray: bool = False,
    start: int = 0,
    count: int | None = None,
    scale_to: tuple[int, int] | None = None,  # lanczos-upscale output to (w, h)
) -> Iterator[np.ndarray]:
    """Yield frames as uint8 arrays, (h, w) if gray else (h, w, 3) RGB."""
    vf = []
    if start or count is not None:
        # trim + setpts and passthrough vsync: without them ffmpeg duplicates the first trimmed frame
        # (frame `start` is delivered twice), shifting every later frame by one
        end = "" if count is None else f":end_frame={start + count}"
        vf.append(f"trim=start_frame={start}{end},setpts=PTS-STARTPTS")
    if crop is not None:
        x, y, w, h = crop
        vf.append(f"crop={w}:{h}:{x}:{y}")
    else:
        w, h = FRAME_W, FRAME_H
    if scale_to is not None:
        w, h = scale_to
        vf.append(f"scale={w}:{h}:flags=lanczos")
    pix_fmt = "gray" if gray else "rgb24"
    frame_bytes = w * h * (1 if gray else 3)

    cmd = [ffmpeg_exe(), "-v", "error", "-i", str(path)]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-vsync", "0", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]      # passthrough: no frame dup/drop

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 4)
    n = 0
    try:
        while count is None or n < count:            # never deliver more than `count` frames
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            n += 1
            frame = np.frombuffer(buf, dtype=np.uint8)
            yield frame.reshape((h, w) if gray else (h, w, 3))
    finally:
        proc.stdout.close()
        proc.wait()


# ---------------------------------------------------------------------------
# one-time lossless cache of the grayscale ROI (4.9 GB), memmapped afterwards

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def cached_roi_gray(crop: tuple[int, int, int, int], path: str | Path = VIDEO_PATH,
                    n_frames: int = N_FRAMES, rebuild: bool = False) -> np.ndarray:
    """(T, h, w) uint8 memmap of the gray crop; built once from the ffmpeg stream.

    Every script that needs the ROI reads this instead of re-decoding the 1.2 GB
    video. The file is the exact rawvideo output of `stream_frames(crop, gray=True)`.
    """
    x, y, w, h = crop
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / f"roi_gray_{x}_{y}_{w}_{h}.npy"
    if f.exists() and not rebuild:
        return np.load(f, mmap_mode="r")
    tmp = f.with_suffix(".building.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n_frames, h, w))
    n = 0
    for n, frame in enumerate(stream_frames(path, crop=crop, gray=True, count=n_frames)):
        arr[n] = frame
    n += 1
    arr.flush()
    del arr
    if n != n_frames:  # shorter video than expected: truncate
        a = np.load(tmp, mmap_mode="r")[:n]
        np.save(f, np.ascontiguousarray(a))
        tmp.unlink()
    else:
        tmp.rename(f)
    return np.load(f, mmap_mode="r")


def roi_frames(crop: tuple[int, int, int, int], count: int | None = None, use_cache: bool = True):
    """Iterate gray ROI frames, from the cache if present, else straight from the video."""
    if use_cache:
        f = CACHE_DIR / f"roi_gray_{crop[0]}_{crop[1]}_{crop[2]}_{crop[3]}.npy"
        if f.exists():
            a = np.load(f, mmap_mode="r")
            yield from (a[i] for i in range(len(a) if count is None else min(count, len(a))))
            return
    yield from stream_frames(crop=crop, gray=True, count=count)

# ---------------------------------------------------------------------------
# one-time lossless cache of the full-frame gray video (25 GB), memmapped afterwards.
# ffmpeg decodes this 4K stream at only ~5 fps here, so the dense ego-motion pass (which
# needs the whole frame) is decode-bound; the cache is built once with parallel chunk
# decoders and then read at disk speed.

FULL_CACHE = CACHE_DIR / "full_gray.npy"


def _decode_chunk(args) -> int:
    path, tmp, lo, hi, n_frames = args
    arr = np.load(tmp, mmap_mode="r+")
    n = 0
    for n, frame in enumerate(stream_frames(path, gray=True, start=lo, count=hi - lo)):
        if lo + n >= hi:
            break
        arr[lo + n] = frame
    arr.flush()
    return n


def cached_full_gray(path: str | Path = VIDEO_PATH, n_frames: int = N_FRAMES, workers: int = 4,
                     rebuild: bool = False) -> np.ndarray:
    """(T, 2160, 3840) uint8 memmap of the gray full frames; built once with `workers` parallel decoders."""
    from multiprocessing import Pool

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if FULL_CACHE.exists() and not rebuild:
        return np.load(FULL_CACHE, mmap_mode="r")
    tmp = FULL_CACHE.with_suffix(".building.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n_frames, FRAME_H, FRAME_W))
    del arr
    edges = np.linspace(0, n_frames, workers + 1).astype(int)
    jobs = [(str(path), str(tmp), int(edges[i]), int(edges[i + 1]), n_frames) for i in range(workers)]
    with Pool(workers) as pool:
        pool.map(_decode_chunk, jobs)
    tmp.rename(FULL_CACHE)
    return np.load(FULL_CACHE, mmap_mode="r")


def full_frames(start: int = 0, count: int | None = None, use_cache: bool = True):
    """Iterate gray full frames [start, start+count), from the cache if present, else from the video."""
    if use_cache and FULL_CACHE.exists():
        a = np.load(FULL_CACHE, mmap_mode="r")
        end = len(a) if count is None else min(start + count, len(a))
        yield from (a[i] for i in range(start, end))
        return
    for n, frame in enumerate(stream_frames(gray=True, start=start, count=count)):
        if count is not None and n >= count:   # ffmpeg trim may emit one frame past end_frame
            break
        yield frame
