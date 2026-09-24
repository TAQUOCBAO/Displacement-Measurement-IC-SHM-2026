"""Phase-based displacement branch (Chen 2015 / Yang 2024) on the UAV video.

Usage: python scripts/run_phase.py [--ego single|dense] [--n-frames N]
                                   [--mask board|board+surround] [--weights A2|A2+block|A2+gdgif]
                                   [--cut piecewise|perframe] [--scales 0,1,2,3] [--orients 2|4]
                                   [--chen] [--decimate 1|2|4] [--no-refine] [--no-prior] [--pad P] [--tag NAME] [--device cuda]

Coarse prior: results/tracks_zncc[_slice].npz (integer cuts, piecewise-constant).
Writes results/tracks_phase[_dense][_slice][_<tag>].npz with the run_pipeline contract
(cable, ref, cable_square_px) plus per-frame diagnostics (sigma, n_eff, cond, flags, cuts).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_disp import config, gdgif, phase_disp  # noqa: E402
from uav_disp.detect import find_board  # noqa: E402
from uav_disp.steerable import PyramidSpec  # noqa: E402
from uav_disp.video_io import roi_frames  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ego", default="single", choices=["single", "dense"])
    ap.add_argument("--n-frames", type=int, default=None)
    ap.add_argument("--mask", default="board", choices=["board", "board+surround"])
    ap.add_argument("--weights", default="A2", choices=["A2", "A2+block", "A2+gdgif"])
    ap.add_argument("--cut", default="piecewise", choices=["piecewise", "perframe"])
    ap.add_argument("--scales", default=None)
    ap.add_argument("--orients", type=int, default=config.PHASE_SPEC["n_orients"])
    ap.add_argument("--chen", action="store_true")
    ap.add_argument("--decimate", type=int, default=config.PHASE_DECIMATE)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--no-prior", action="store_true", help="do not linearise at the ZNCC estimate (wrap-limited)")
    ap.add_argument("--pad", type=int, default=config.PHASE_PAD)
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default=None, help="None: numpy/scipy; 'cuda' or 'cpu': batched torch.fft backend")
    return ap.parse_args()


def decimate2d(img: np.ndarray, d: int) -> np.ndarray:
    """Chen-style spatial downsampling (box average) before filtering."""
    if d == 1:
        return img
    h, w = (img.shape[0] // d) * d, (img.shape[1] // d) * d
    return img[:h, :w].reshape(h // d, d, w // d, d).mean(axis=(1, 3)).astype(np.float32)


def main() -> None:
    a = parse()
    suffix = "_slice" if a.n_frames else ""
    zn = np.load(ROOT / f"results/tracks_zncc{suffix}.npz")
    T = len(zn["cable"]) if a.n_frames is None else min(a.n_frames, len(zn["cable"]))
    size = config.PHASE_PATCH
    half = size // 2
    scales = None if a.scales is None else [int(s) for s in a.scales.split(",")]
    spec = PyramidSpec(n_scales=config.PHASE_SPEC["n_scales"], n_orients=a.orients,
                       half_octave=config.PHASE_SPEC["half_octave"], twidth=config.PHASE_SPEC["twidth"], pad=a.pad)

    cuts, changes = {}, {}
    for name in ("cable", "ref"):
        tr = zn[name][:T]
        if a.cut == "piecewise":
            cuts[name], changes[name] = phase_disp.coarse_cuts(tr, config.PHASE_COARSE_STEP, config.PHASE_COARSE_K)
        else:
            cuts[name] = np.rint(tr).astype(int)
            changes[name] = np.flatnonzero(np.any(np.diff(cuts[name], axis=0) != 0, axis=1)) + 1

    frames = roi_frames(config.ROI, count=T)
    first = np.asarray(next(frames))
    boards = {"cable": find_board(first, config.CABLE_APPROX, config.APPROX_SQUARE_PX),
              "ref": find_board(first, config.REF_APPROX, config.APPROX_SQUARE_PX)}

    accs = {}
    t0 = time.perf_counter()
    d = a.decimate
    for name in ("cable", "ref"):
        c0 = cuts[name][0]
        origin = (c0[0] - half, c0[1] - half)
        patch = phase_disp.cut_patches(first, c0, size)
        m = phase_disp.board_mask(boards[name], size, size, origin, margin_px=6.0 if a.mask == "board" else 40.0)
        m = m * phase_disp.gaussian_taper(size, size, 0.3)
        if a.weights == "A2+gdgif":
            _, _, Gamma = gdgif.edge_weights((patch - patch.min()) / np.ptp(patch), config.GDGIF_H)
            m = m * (Gamma / Gamma.max()).astype(np.float32)
        block = (config.ROI[0] + origin[0], config.ROI[1] + origin[1]) if a.weights == "A2+block" else None
        accs[name] = phase_disp.PhaseAccumulator(
            decimate2d(patch, d), spec, weight_mask=decimate2d(m, d), cond_max=config.PHASE_COND_MAX,
            scales=scales, chen_mode=a.chen, block_origin=None if block is None else (block[0] // d, block[1] // d),
            block_kw=dict(w_block=config.PHASE_BLOCK_W, period=max(16 // d, 1)), n_refine=0 if a.no_refine else 1,
            device=a.device)

    for i, frame in enumerate(frames, start=1):
        if i >= T:
            break
        frame = np.asarray(frame)
        for name in ("cable", "ref"):
            c = cuts[name][i]
            patch = decimate2d(phase_disp.cut_patches(frame, c, size), d)
            block = None
            if a.weights == "A2+block":
                block = ((config.ROI[0] + c[0] - half) // d, (config.ROI[1] + c[1] - half) // d)
            init = None if a.no_prior else (zn[name][i] - c) / d      # raw ZNCC estimate as linearisation point
            accs[name].push(patch, block, init_uv=init)
        if i % 500 == 0:
            print(f"  frame {i}/{T}  {time.perf_counter() - t0:.1f} s", flush=True)
    runtime = time.perf_counter() - t0

    out = {"cable_square_px": np.array(boards["cable"].square_px), "ref_square_px": np.array(boards["ref"].square_px),
           "cable_angle_deg": np.array(boards["cable"].angle_deg), "runtime_s": np.array(runtime),
           "spec": json.dumps(spec.__dict__), "args": json.dumps(vars(a))}
    for name in ("cable", "ref"):
        fit = accs[name].result()
        n = len(fit.uv)
        out[name] = cuts[name][:n] + fit.uv * d
        out[f"uv_{name}"] = fit.uv * d
        out[f"sigma_{name}"] = fit.sigma * d
        out[f"n_eff_{name}"] = fit.n_eff
        out[f"cond_{name}"] = fit.cond
        out[f"flags_{name}"] = fit.flags
        out[f"resid_{name}"] = fit.resid_rms
        out[f"cuts_{name}"] = cuts[name][:n]
        out[f"cut_changes_{name}"] = changes[name]
        if fit.per_scale_uv is not None:
            out[f"per_scale_uv_{name}"] = fit.per_scale_uv * d
        mm = config.SQUARE_MM / boards[name].square_px
        print(f"[{name}] frames {n} | mean sigma {fit.sigma.mean(axis=0).round(4)} px = "
              f"{(fit.sigma.mean(axis=0) * mm * d).round(4)} mm | cond {fit.cond[0]:.1f} | "
              f"frac cond>{config.PHASE_COND_MAX:g}: {np.mean(fit.flags & 1 > 0):.3f} | "
              f"frac wrap-risk: {np.mean(fit.flags & 2 > 0):.3f} | median N_eff {np.median(fit.n_eff):.0f} | "
              f"cut changes {len(changes[name])}")

    if a.ego == "dense":
        ego = np.load(ROOT / "results/egomotion.npz")
        from uav_disp import egomotion  # noqa: E402
        out["ref"] = egomotion.ego_at_target_from_file(ego, out["cable"][0] + np.array(config.ROI[:2]), T)
    tag = f"_{a.tag}" if a.tag else ""
    path = ROOT / f"results/tracks_phase{'_dense' if a.ego == 'dense' else ''}{suffix}{tag}.npz"
    np.savez(path, **out)
    print(f"runtime {runtime:.1f} s -> {path}")


if __name__ == "__main__":
    main()
