"""Real-data ablations against the LDS reference + error budget.

Usage: python scripts/run_ablations.py
Needs results/tracks_zncc.npz and results/tracks_cotracker.npz.
Writes results/ablations.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_disp import config, evaluate, lds, postprocess

MM_PER_PX = None  # set from tracks


def similarity_stabilize(cable_mean: np.ndarray, ref_pts: np.ndarray) -> np.ndarray:
    """Per-frame Umeyama similarity fit ref_pts[0] -> ref_pts[t]; returns the
    cable point mapped back through the inverse transform (removes UAV
    translation + roll + scale breathing, not just translation)."""
    p0 = ref_pts[0]
    mu0 = p0.mean(axis=0)
    q0 = p0 - mu0
    out = np.empty_like(cable_mean)
    for t in range(len(cable_mean)):
        pt = ref_pts[t]
        mut = pt.mean(axis=0)
        qt = pt - mut
        H = qt.T @ q0
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(U @ Vt))
        D = np.diag([1.0, d])
        R = U @ D @ Vt                      # rotation frame0 -> frame t
        s = np.trace(np.diag(S) @ D) / (q0 * q0).sum()
        # invert: x0 = R^T (xt - mut)/s + mu0
        out[t] = R.T @ (cable_mean[t] - mut) / s + mu0
    return out


def score(mm: np.ndarray, lds50: np.ndarray) -> tuple[float, float]:
    a = evaluate.align_and_score(mm, lds50)
    return a.rmse, a.corr


def main():
    zn = np.load(ROOT / "results/tracks_zncc.npz")
    ct = np.load(ROOT / "results/tracks_cotracker.npz")
    ev = np.load(ROOT / "results/eval_zncc.npz")
    angle = float(ev["angle_rad"])
    sq = float(zn["cable_square_px"])
    global MM_PER_PX
    MM_PER_PX = config.SQUARE_MM / sq

    lds_raw = lds.load(ROOT / "data/lds.npy")
    lds50 = lds.decimate_to(lds_raw, config.FPS)

    lines = ["# Ablations vs LDS reference (full 60 s)", ""]

    # --- A. ego-motion compensation ---
    lines += ["## A. Ego-motion compensation (band-pass 3-14 Hz applied in all rows)", "",
              "| tracker | compensation | RMSE (mm) | corr |", "|---|---|---|---|"]
    zn_c, zn_r = zn["cable"], zn["ref"]
    ct_c, ct_r = ct["cable"].mean(axis=1), ct["ref"].mean(axis=1)
    cases = [
        ("zncc", "none", zn_c, np.zeros_like(zn_c)),
        ("zncc", "translation", zn_c, zn_r),
        ("cotracker", "none", ct_c, np.zeros_like(ct_c)),
        ("cotracker", "translation", ct_c, ct_r),
    ]
    for tr, name, c, r in cases:
        rmse, corr = score(postprocess.displacement_mm(c, r, sq, angle), lds50)
        lines.append(f"| {tr} | {name} | {rmse:.4f} | {corr:.4f} |")
    stab = similarity_stabilize(ct_c, ct["ref"])
    rmse, corr = score(postprocess.displacement_mm(stab, np.zeros_like(stab), sq, angle), lds50)
    lines.append(f"| cotracker | similarity (9-pt) | {rmse:.4f} | {corr:.4f} |")

    # --- B. filtering (translation compensation) ---
    lines += ["", "## B. Band-pass ablation (translation compensation)", "",
              "| tracker | filter | RMSE (mm) | corr |", "|---|---|---|---|"]
    for tr, c, r in [("zncc", zn_c, zn_r), ("cotracker", ct_c, ct_r)]:
        for fname, hp, lp in [("none", None, None), ("HP 3 Hz only", 3.0, None),
                              ("LP 14 Hz only", None, 14.0), ("band 3-14 Hz", 3.0, 14.0)]:
            mm = postprocess.displacement_mm(c, r, sq, angle, highpass_hz=hp, lowpass_hz=lp)
            rmse, corr = score(mm, lds50)
            lines.append(f"| {tr} | {fname} | {rmse:.4f} | {corr:.4f} |")

    # --- C. fusion weight sweep ---
    mm_zn = postprocess.displacement_mm(zn_c, zn_r, sq, angle)
    mm_ct = postprocess.displacement_mm(ct_c, ct_r, sq, angle)
    n = min(len(mm_zn), len(mm_ct))
    lines += ["", "## C. Fusion weight (w * zncc + (1-w) * cotracker)", "",
              "| w | RMSE (mm) |", "|---|---|"]
    best = (9, 0)
    for w in np.arange(0, 1.001, 0.1):
        rmse, _ = score(w * mm_zn[:n] + (1 - w) * mm_ct[:n], lds50)
        best = min(best, (rmse, w))
        lines.append(f"| {w:.1f} | {rmse:.4f} |")
    lines.append(f"\nBest: w = {best[1]:.1f} -> {best[0]:.4f} mm")

    # --- D. error budget ---
    # LDS self-noise: flat PSD floor of the raw 10 kHz record between 20-40 Hz
    # (above the structural band, below any decimation effects)
    f, p = signal.welch(lds_raw, fs=config.LDS_FS, nperseg=1 << 16)
    floor = np.median(p[(f > 20) & (f < 40)])
    band = config.LOWPASS_HZ - config.HIGHPASS_HZ
    lds_noise = float(np.sqrt(floor * band))
    err_corr = np.nan
    e1_path = ROOT / "results/synth/E1.csv"
    vision_floor = None
    if e1_path.exists():
        rows = [r.split(",") for r in e1_path.read_text().strip().splitlines()[1:]]
        static = {(r[0], float(r[1])): float(r[3]) for r in rows if float(r[2]) == 0.0}
        vision_floor = {k: v * MM_PER_PX for k, v in static.items()}
    ez = np.load(ROOT / "results/eval_zncc.npz")
    ec = np.load(ROOT / "results/eval_cotracker.npz")
    m = min(len(ez["vision"]), len(ec["vision"]))
    err_corr = float(np.corrcoef(ez["vision"][:m] - ez["lds"][:m],
                                 ec["vision"][:m] - ec["lds"][:m])[0, 1])
    lines += ["", "## D. Error budget", "",
              f"- LDS self-noise in the 3-14 Hz scoring band (from its 20-40 Hz PSD floor): "
              f"**{lds_noise:.4f} mm RMS** (irreducible for any vision method)",
              f"- Inter-tracker error correlation: **{err_corr:.3f}** "
              f"(shared floor: LDS noise, rolling shutter, reference-target micro-vibration)"]
    if vision_floor:
        lines.append("- Synthetic static noise floors (mm, from E1): " +
                     ", ".join(f"{t} z{z:g}: {v:.4f}" for (t, z), v in sorted(vision_floor.items())))

    out = ROOT / "results/ablations.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
