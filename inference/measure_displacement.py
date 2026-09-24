"""Measure the in-plane displacement of a checkerboard target from a hovering-UAV video.

One command, video in -> 50 Hz CSV out (time_s, displacement_mm), using the local-phase tracker
of the repo (src/uav_disp) with a stationary same-depth reference target for ego-motion cancellation.

    python inference/measure_displacement.py data/Video.MP4 --out displacement.csv
    python inference/measure_displacement.py my_video.mp4 --roi 1600,600,1344,1216 --cable 300,290 --ref 907,738 \
        --square-px 30 --square-mm 27.5 --band 3-14 --device cuda --out out.csv

Defaults reproduce the competition scene. For a new scene give: --roi (crop x,y,w,h in the full
frame containing both boards), --cable / --ref (approximate board centres, ROI-local px), --square-px
(approximate checker square size), --square-mm (physical square size), and either --axis-deg (cable
axis in the image, degrees) or leave it to be measured from the red-cable mask (this scene only).
Processing: two passes over the video (ZNCC template matching for the coarse track, then the phase
tracker linearised at it). ffmpeg is needed (imageio-ffmpeg wheel or on PATH).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_disp import config, phase_disp, postprocess  # noqa: E402
from uav_disp.detect import find_board  # noqa: E402
from uav_disp.steerable import PyramidSpec  # noqa: E402
from uav_disp.track_zncc import track as zncc_track  # noqa: E402
from uav_disp.video_io import stream_frames  # noqa: E402


def ints(s: str) -> tuple[int, ...]:
    return tuple(int(v) for v in s.split(","))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", default="displacement.csv")
    ap.add_argument("--roi", type=ints, default=config.ROI, help="x,y,w,h crop containing both boards")
    ap.add_argument("--cable", type=ints, default=config.CABLE_APPROX, help="approx. cable-board centre, ROI px")
    ap.add_argument("--ref", type=ints, default=config.REF_APPROX, help="approx. reference-board centre, ROI px")
    ap.add_argument("--square-px", type=float, default=config.APPROX_SQUARE_PX)
    ap.add_argument("--square-mm", type=float, default=config.SQUARE_MM)
    ap.add_argument("--fps", type=float, default=config.FPS)
    ap.add_argument("--band", default="3-14", help="'lo-hi' Hz band-pass of the output, or 'full'")
    ap.add_argument("--axis-deg", type=float, default=None, help="cable axis angle in the image (deg); "
                    "default: measured from the red-cable mask around the target")
    ap.add_argument("--n-frames", type=int, default=None)
    ap.add_argument("--device", default=None, help="'cuda' for the batched torch backend (needs torch)")
    a = ap.parse_args()
    t0 = time.perf_counter()

    # ---- pass 1: detect the boards in frame 0, ZNCC coarse tracks
    frames = stream_frames(a.video, crop=a.roi, gray=True, count=a.n_frames)
    first = next(frames)
    boards = {"cable": find_board(first, a.cable, a.square_px), "ref": find_board(first, a.ref, a.square_px)}
    for k, b in boards.items():
        print(f"{k} board: centre ({b.center[0]:.1f}, {b.center[1]:.1f}) px, square {b.square_px:.2f} px, "
              f"rotation {b.angle_deg:.1f} deg", flush=True)

    def chain():
        yield first
        yield from frames
    zn = zncc_track(chain(), {k: tuple(b.center) for k, b in boards.items()})
    T = len(zn["cable"])
    print(f"ZNCC pass: {T} frames in {time.perf_counter() - t0:.0f} s", flush=True)

    # ---- pass 2: local-phase tracker linearised at the ZNCC track (piecewise-constant integer cuts)
    size, half = config.PHASE_PATCH, config.PHASE_PATCH // 2
    spec = PyramidSpec(**config.PHASE_SPEC, pad=config.PHASE_PAD)
    cuts = {k: phase_disp.coarse_cuts(zn[k], config.PHASE_COARSE_STEP, config.PHASE_COARSE_K)[0] for k in boards}
    accs = {}
    for k, b in boards.items():
        c0 = cuts[k][0]
        origin = (c0[0] - half, c0[1] - half)
        m = phase_disp.board_mask(b, size, size, origin, margin_px=6.0) * phase_disp.gaussian_taper(size, size, 0.3)
        accs[k] = phase_disp.PhaseAccumulator(phase_disp.cut_patches(first, c0, size), spec, weight_mask=m,
                                              cond_max=config.PHASE_COND_MAX, n_refine=1, device=a.device)
    for i, frame in enumerate(stream_frames(a.video, crop=a.roi, gray=True, start=1, count=T - 1), start=1):
        if i >= T:
            break
        for k in boards:
            c = cuts[k][i]
            accs[k].push(phase_disp.cut_patches(frame, c, size), init_uv=zn[k][i] - c)
        if i % 500 == 0:
            print(f"  phase pass: frame {i}/{T}  {time.perf_counter() - t0:.0f} s", flush=True)
    tracks = {}
    for k in boards:
        fit = accs[k].result()
        n = len(fit.uv)
        tracks[k] = cuts[k][:n] + fit.uv
        print(f"{k}: predicted sigma {np.mean(fit.sigma, axis=0).round(4)} px, wrap-risk frames "
              f"{np.mean(fit.flags & 2 > 0) * 100:.1f} %", flush=True)

    # ---- ego-motion cancellation, projection on the cable normal, mm scaling, band-pass
    if a.axis_deg is None:
        rgb0 = next(stream_frames(a.video, crop=a.roi, count=1))
        angle = postprocess.cable_axis_angle(rgb0, tuple(boards["cable"].center))
    else:
        angle = np.deg2rad(a.axis_deg)
    hp, lp = (None, None) if a.band == "full" else (float(a.band.split("-")[0]), float(a.band.split("-")[1]))
    n = min(len(tracks["cable"]), len(tracks["ref"]))
    mm = postprocess.displacement_mm(tracks["cable"][:n], tracks["ref"][:n], boards["cable"].square_px, angle,
                                     highpass_hz=hp, lowpass_hz=lp, fps=a.fps)
    mm = mm * (a.square_mm / config.SQUARE_MM)          # displacement_mm assumes config.SQUARE_MM per square
    t = np.arange(len(mm)) / a.fps
    np.savetxt(a.out, np.column_stack([t, mm]), delimiter=",", header="time_s,displacement_mm", comments="", fmt="%.6f")
    print(f"axis {np.degrees(angle) % 180:.2f} deg | scale {a.square_mm / boards['cable'].square_px:.4f} mm/px | "
          f"band {a.band} Hz | std {mm.std():.4f} mm | {len(mm)} samples -> {a.out} | {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
