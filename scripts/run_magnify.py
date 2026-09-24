"""PVMM + GDGIF magnification clips and the Yang/Chen validation figures (visualisation / QC only).

Usage: python scripts/run_magnify.py [--frames 0:1000] [--crop 512] [--band 0]
Writes results/magnified_<lo>-<hi>_a<alpha>[_gdgif].mp4 and results/fig_magnify_{slice,ssim,canny}.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_disp import config, magnify  # noqa: E402
from uav_disp.steerable import PyramidSpec  # noqa: E402
from uav_disp.video_io import roi_frames  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="0:1000")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--band", type=int, default=None, help="index into config.MAGNIFY_BANDS (default: all)")
    a = ap.parse_args()
    f0, f1 = (int(v) for v in a.frames.split(":"))
    zn = np.load(ROOT / "results/tracks_zncc.npz")
    cx, cy = np.rint(zn["cable"][0]).astype(int)
    h = a.crop // 2
    ego = zn["ref"][f0:f1] - zn["ref"][0]
    frames = np.stack([np.asarray(fr)[cy - h:cy + h, cx - h:cx + h] for i, fr in enumerate(roi_frames(config.ROI, count=f1)) if i >= f0])
    bands = config.MAGNIFY_BANDS if a.band is None else [config.MAGNIFY_BANDS[a.band]]
    spec = PyramidSpec(n_scales=5, n_orients=4, half_octave=True, pad=64)
    stab = np.stack([magnify.fourier_shift(fr.astype(np.float64), -e[0], -e[1]) for fr, e in zip(frames, ego)])
    stab = np.clip(np.rint(stab), 0, 255).astype(np.uint8)
    for lo, hi, alpha in bands:
        outs = {}
        for g in (False, True):
            out = magnify.magnify_sequence(frames, (lo, hi), alpha, config.FPS, spec, gdgif_on=g, stabilise_xy=ego,
                                           gdgif_h=config.GDGIF_H, gdgif_mu=config.GDGIF_MU)
            tag = f"{lo:g}-{hi:g}_a{alpha:g}{'_gdgif' if g else ''}"
            magnify.write_video(out, ROOT / f"results/magnified_{tag}.mp4", config.FPS)
            outs[g] = out
            print("wrote", f"magnified_{tag}.mp4")
        # figures (Yang Figs. 3-6, 10, 11-12 protocol)
        x0 = h
        fig, ax = plt.subplots(3, 1, figsize=(12, 8))
        for k, (name, arr) in enumerate([("input (stabilised)", stab), ("PVMM", outs[False]), ("PVMM + GDGIF", outs[True])]):
            ax[k].imshow(magnify.space_time_slice(arr, x0, (h - 80, h + 80)), cmap="gray", aspect="auto")
            ax[k].set(title=f"{name}: space-time slice at x={x0}, band {lo:g}-{hi:g} Hz, alpha {alpha:g}")
        fig.tight_layout(); fig.savefig(ROOT / f"results/fig_magnify_slice_{lo:g}-{hi:g}.png", dpi=120)
        ss = {g: np.array([magnify.ssim_psnr(stab[t], outs[g][t]) for t in range(len(stab))]) for g in outs}
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        for g, lab in [(False, "PVMM"), (True, "PVMM + GDGIF")]:
            ax[0].plot(ss[g][:, 0], lw=0.8, label=f"{lab}: mean SSIM {ss[g][:, 0].mean():.4f}")
            ax[1].plot(ss[g][:, 1], lw=0.8, label=f"{lab}: mean PSNR {ss[g][:, 1].mean():.2f} dB")
        ax[0].set(ylabel="SSIM"); ax[1].set(ylabel="PSNR (dB)", xlabel="frame")
        ax[0].legend(); ax[1].legend()
        fig.tight_layout(); fig.savefig(ROOT / f"results/fig_magnify_ssim_{lo:g}-{hi:g}.png", dpi=120)
        print(f"  SSIM/PSNR  PVMM {ss[False].mean(axis=0).round(4)}  +GDGIF {ss[True].mean(axis=0).round(4)}")
        # Canny at the peak-deflection frame (from the ZNCC cable track)
        rel = (zn["cable"][f0:f1] - zn["ref"][f0:f1]); rel -= rel.mean(axis=0)
        tpk = int(np.argmax(np.abs(rel[:, 1])))
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for k, (name, arr) in enumerate([("input", stab), ("PVMM", outs[False]), ("PVMM + GDGIF", outs[True])]):
            ax[k].imshow(magnify.canny_edges(arr[tpk][h - 100:h + 100, h - 100:h + 100]), cmap="gray")
            ax[k].set(title=f"Canny, frame {f0 + tpk}: {name}")
        fig.tight_layout(); fig.savefig(ROOT / f"results/fig_magnify_canny_{lo:g}-{hi:g}.png", dpi=120)


if __name__ == "__main__":
    main()
