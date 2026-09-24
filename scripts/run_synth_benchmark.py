"""Synthetic sub-pixel benchmark experiments.

Usage: python scripts/run_synth_benchmark.py E1|E2|E3|E4 [--compress]

--compress passes every sequence through an H.264 encode/decode round-trip at
the real video's ~0.4 bits/px before tracking (results saved with a C suffix,
e.g. E1C.csv). The clean variant is the tracker's lower-bound floor; the
compressed variant is the deployment-realistic one.

E1: zoom sweep x {static, 0.5 px sinusoid} x {cotracker, zncc}  (zoom law + noise floor)
E2: amplitude sweep at zoom 2                                    (linearity)
E3: cotracker query-count / support-grid ablation at zoom 2
E4: phase-based estimator (Chen/Yang) vs zncc: amplitude sweep x zoom {1, 2}   (gate G0)

Ground truth is exact: frames are Fourier-shifted copies of one real frame.
"Zoom Z" means the tracker sees a 512/Z-wide source region rendered at 512x384,
i.e. Z-times magnification in model pixels. All errors are reported in original
video pixels (and mm via the 0.884 mm/px scale).

Writes results/synth/<exp>.csv and per-run traces in results/synth/<exp>.npz.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_disp import config, synth
from uav_disp.detect import find_board
from uav_disp.track_zncc import track as zncc_track
from uav_disp.video_io import stream_frames

N_FRAMES = 250
SIN_HZ = 3.0
NOISE_SIGMA = 1.5
MM_PER_PX = config.SQUARE_MM / 31.09  # measured cable-target scale, for reporting


def get_scene():
    """One real RGB ROI frame, the reference-board geometry, and the motion direction."""
    rgb0 = next(stream_frames(crop=config.ROI, count=1))
    gray0 = (rgb0 @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
    board = find_board(gray0, config.REF_APPROX, config.APPROX_SQUARE_PX)
    angle = float(np.load(ROOT / "results/eval_zncc.npz")["angle_rad"])
    normal = np.array([np.sin(angle), -np.cos(angle)])
    return rgb0, board, normal


def run_one(rgb0, board, normal, zoom: float, amp_px: float, tracker: str,
            n_query: int = 9, support: bool = True, seed: int = 0,
            compress: bool = False):
    """Returns (rmse_px, est, gt) for one synthetic sequence."""
    sw, sh = int(round(512 / zoom)) // 2 * 2, int(round(384 / zoom)) // 2 * 2
    cx, cy = board.center
    ox, oy = int(round(cx - sw / 2)), int(round(cy - sh / 2))
    p = synth.PAD
    source = rgb0[oy - p : oy + sh + p, ox - p : ox + sw + p]

    traj = synth.sinusoid_traj(N_FRAMES, amp_px, SIN_HZ, config.FPS, normal)
    gt = traj @ normal  # signed scalar displacement, exact
    eff_zoom = 512 / sw

    def frames_native():
        gen = synth.make_sequence(source, traj, NOISE_SIGMA, seed)
        if compress:
            tmp = ROOT / "results/synth/_roundtrip.mp4"
            yield from synth.h264_roundtrip(gen, config.FPS, tmp)
        else:
            yield from gen

    if tracker == "phase":
        from uav_disp import phase_disp
        from uav_disp.steerable import PyramidSpec
        size = config.PHASE_PATCH
        gen = frames_native()
        stack = []
        for f in gen:
            g = (f @ np.array([0.299, 0.587, 0.114])).astype(np.float32)
            if eff_zoom != 1:
                g = synth.upscale(g, eff_zoom).astype(np.float32)
            c = (int(round((cx - ox) * eff_zoom)), int(round((cy - oy) * eff_zoom)))
            stack.append(phase_disp.cut_patches(g, c, size))
        stack = np.stack(stack)
        m = phase_disp.gaussian_taper(size, size, 0.3)
        sq = board.square_px * eff_zoom
        yy, xx = np.mgrid[:size, :size]
        m *= ((np.abs(xx - size / 2) <= sq + 6) & (np.abs(yy - size / 2) <= sq + 6)).astype(np.float32)
        fit = phase_disp.solve_patch(stack, PyramidSpec(**config.PHASE_SPEC), weight_mask=m, per_scale=False)
        est_xy = fit.uv / eff_zoom
        run_one.last_sigma = float(np.mean(fit.sigma) / eff_zoom)
    elif tracker == "zncc":
        def frames_gray():
            for f in frames_native():
                g = (f @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
                yield g if eff_zoom == 1 else synth.upscale(g, eff_zoom)
        init = ((cx - ox) * eff_zoom, (cy - oy) * eff_zoom)
        th = min(int(32 * eff_zoom), 64)
        res = zncc_track(frames_gray(), {"t": init}, template_half=th, search_rad=16)
        est_xy = res["t"] / eff_zoom
    else:
        import torch
        from uav_disp.track_cotracker import load_model, pick_device, track_target
        device = pick_device()
        if not hasattr(run_one, "_model"):
            run_one._model = load_model(device)
        model = run_one._model

        def frames_up():
            for f in frames_native():
                yield synth.upscale(f, eff_zoom)[:384, :512]

        q = (board.query_points()[:n_query] - np.array([ox, oy])) * eff_zoom
        if support:
            gx, gy = np.meshgrid(np.linspace(30, 482, 8), np.linspace(30, 354, 6))
            grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
            keep = np.linalg.norm(grid - q.mean(axis=0), axis=1) > eff_zoom * board.square_px * 2.5
            q_all = np.vstack([q, grid[keep][:48]])
        else:
            q_all = q
        tracks, _ = track_target(frames_up(), q_all, model, device)
        est_xy = tracks[:, :n_query].mean(axis=1) / eff_zoom

    est = (est_xy - est_xy[0]) @ normal
    est, gt_c = est - est.mean(), gt - gt.mean()
    rmse = float(np.sqrt(np.mean((est - gt_c) ** 2)))
    return rmse, est, gt_c


def main():
    exp = sys.argv[1]
    compress = "--compress" in sys.argv
    tag = exp + ("C" if compress else "")
    out_dir = ROOT / "results/synth"
    out_dir.mkdir(exist_ok=True)
    rgb0, board, normal = get_scene()

    rows, traces = [], {}
    if exp == "E1":
        for zoom in [1, 1.5, 2, 3, 4]:
            for amp in [0.0, 0.5]:
                for tracker in ["zncc", "cotracker"]:
                    rmse, est, gt = run_one(rgb0, board, normal, zoom, amp, tracker,
                                            compress=compress)
                    rows.append((tracker, zoom, amp, rmse))
                    traces[f"{tracker}_z{zoom}_a{amp}"] = est
                    print(f"{tracker:9s} zoom {zoom:<3} amp {amp:<4} -> "
                          f"RMSE {rmse:.4f} px ({rmse * MM_PER_PX:.4f} mm)", flush=True)
        hdr = "tracker,zoom,amp_px,rmse_px"
    elif exp == "E2":
        for amp in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
            for tracker in ["zncc", "cotracker"]:
                rmse, est, gt = run_one(rgb0, board, normal, 2, amp, tracker,
                                        compress=compress)
                rows.append((tracker, 2, amp, rmse))
                traces[f"{tracker}_a{amp}"] = est
                print(f"{tracker:9s} amp {amp:<5} -> RMSE {rmse:.4f} px", flush=True)
        hdr = "tracker,zoom,amp_px,rmse_px"
    elif exp == "E3":
        for n_q in [1, 5, 9]:
            for support in [False, True]:
                rmse, est, gt = run_one(rgb0, board, normal, 2, 0.5, "cotracker",
                                        n_query=n_q, support=support, compress=compress)
                rows.append((n_q, support, rmse))
                print(f"queries {n_q} support {support} -> RMSE {rmse:.4f} px", flush=True)
        hdr = "n_query,support,rmse_px"
    elif exp == "E4":
        for zoom in [1, 2]:
            for amp in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
                for tracker in ["phase", "zncc"]:
                    rmse, est, gt = run_one(rgb0, board, normal, zoom, amp, tracker, compress=compress)
                    sig = getattr(run_one, "last_sigma", float("nan")) if tracker == "phase" else float("nan")
                    rows.append((tracker, zoom, amp, rmse, sig))
                    traces[f"{tracker}_z{zoom}_a{amp}"] = est
                    print(f"{tracker:9s} zoom {zoom:<3} amp {amp:<4} -> RMSE {rmse:.4f} px "
                          f"({rmse * MM_PER_PX:.4f} mm) sigma {sig:.4f} px", flush=True)
        hdr = "tracker,zoom,amp_px,rmse_px,mean_sigma_px"
    else:
        raise SystemExit(f"unknown experiment {exp}")

    np.savez(out_dir / f"{tag}.npz", **traces)
    with open(out_dir / f"{tag}.csv", "w") as f:
        f.write(hdr + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print("wrote", out_dir / f"{tag}.csv")


if __name__ == "__main__":
    main()
