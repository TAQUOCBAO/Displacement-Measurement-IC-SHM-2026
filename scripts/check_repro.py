"""Consistency checks after the fresh-environment run (STAGE=repro).

1. results/tracks_phase_slice.npz (300 frames, numpy backend, slice-only ZNCC prior) vs the committed
   full-video tracks (CUDA backend): the two differ only by the linearisation point at the slice edges
   and float32 round-off -> max |d| < 0.02 px and RMS < 0.005 px (tracker sigma is ~0.007 px).
2. inference/measure_displacement.py CSV vs the displacement computed from the slice tracks with the
   same band-pass: RMS difference < 0.01 mm.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uav_disp import config, postprocess  # noqa: E402

full = np.load(ROOT / "results/tracks_phase.npz")
sl = np.load(ROOT / "results/tracks_phase_slice.npz")
n = len(sl["cable"])
d = sl["cable"] - full["cable"][:n]
dmax, drms = float(np.abs(d).max()), float(np.sqrt(np.mean(d ** 2)))
print(f"[1] phase slice vs full-video tracks: {n} frames, max |d| = {dmax:.4f} px, RMS = {drms:.4f} px")
ok1 = dmax < 0.02 and drms < 0.005

csv = np.loadtxt(ROOT / "results/repro_slice.csv", delimiter=",", skiprows=1)
angle = float(np.load(ROOT / "results/eval_zncc.npz")["angle_rad"])
mm = postprocess.displacement_mm(sl["cable"], sl["ref"], float(sl["cable_square_px"]), angle,
                                 highpass_hz=3.0, lowpass_hz=14.0)
m = min(len(mm), len(csv))
diff = float(np.sqrt(np.mean((csv[:m, 1] - mm[:m]) ** 2)))
print(f"[2] inference CSV vs slice-track displacement: {m} rows, dt = {np.diff(csv[:, 0]).mean():.4f} s, "
      f"std {csv[:, 1].std():.4f} vs {mm.std():.4f} mm, RMS difference = {diff:.4f} mm")
ok2 = len(csv) == n and abs(np.diff(csv[:, 0]).mean() - 1 / config.FPS) < 1e-9 and np.all(np.isfinite(csv[:, 1])) and diff < 0.01
print("REPRO CHECK:", "PASS" if (ok1 and ok2) else "FAIL")
sys.exit(0 if (ok1 and ok2) else 1)
