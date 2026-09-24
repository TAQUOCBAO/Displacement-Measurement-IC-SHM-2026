"""Phase-branch ablations vs the LDS reference -> appends '## E. Phase branch' to results/ablations.md.

Usage: python scripts/run_phase_ablations.py [--quick]   (--quick: 1000-frame slice, for smoke tests)
Each row runs scripts/run_phase.py with one option changed and scores full-band and 3-14 Hz.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from uav_disp import config, evaluate, lds  # noqa: E402
from run_pipeline import signal_from_tracks, sigma_mm  # noqa: E402

ROWS = [
    ("baseline (A2, board, piecewise, LSQ, 4 scales, 2 orients)", []),
    ("scales finest only", ["--scales", "0"]),
    ("scales 0,1", ["--scales", "0,1"]),
    ("scales 0,1,2", ["--scales", "0,1,2"]),
    ("orientations 4", ["--orients", "4"]),
    ("weights A2+block", ["--weights", "A2+block"]),
    ("weights A2+GDGIF-Gamma", ["--weights", "A2+gdgif"]),
    ("mask board+surround", ["--mask", "board+surround"]),
    ("cut per-frame", ["--cut", "perframe"]),
    ("estimator Chen Eq.4/5", ["--chen"]),
    ("decimate 2 (Chen-style)", ["--decimate", "2"]),
    ("decimate 4 (Chen-style)", ["--decimate", "4"]),
    ("no Gauss-Newton refinement", ["--no-refine"]),
    ("no ZNCC prior (wrap-limited)", ["--no-prior"]),
]


def score(path: Path, lds50: np.ndarray, angle: float) -> dict:
    out = {}
    for name, hp, lp in [("full", None, None), ("3-14", 3.0, 14.0)]:
        mm, _, d = signal_from_tracks(path, angle, hp=hp, lp=lp)
        a = evaluate.align_and_score(mm, lds50)
        bs = evaluate.band_scores(a.vision, a.lds)
        s = sigma_mm(d, angle)
        out[name] = dict(rmse=a.rmse, corr=a.corr, coh=bs["coherence"], sigma=float(np.mean(s)) if s is not None else np.nan,
                         runtime=float(d["runtime_s"]), wrap=float(np.mean(d["flags_cable"] & 2 > 0)))
    return out


def main() -> None:
    quick = "--quick" in sys.argv
    n = ["--n-frames", "1000"] if quick else []
    import os
    if os.environ.get("PHASE_DEVICE"):
        n += ["--device", os.environ["PHASE_DEVICE"]]
    lds50 = lds.decimate_to(lds.load(ROOT / "data/lds.npy"), config.FPS)
    angle = float(np.load(ROOT / "results/eval_zncc.npz")["angle_rad"])
    lines = ["", "## E. Phase branch (Chen 2015 / Yang 2024) — full-band and 3–14 Hz vs LDS" + (" [1000-frame slice]" if quick else ""), "",
             "| variant | RMSE full (mm) | RMSE 3–14 (mm) | corr 3–14 | coh 0–3 / 3–14 / 14–25 | mean σ (mm) | RMSE/σ | wrap-risk | runtime (s) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, (name, args) in enumerate(ROWS):
        tag = f"abl{i}"
        subprocess.run([sys.executable, str(ROOT / "scripts/run_phase.py"), "--tag", tag, *args, *n], check=True)
        path = ROOT / f"results/tracks_phase{'_slice' if quick else ''}_{tag}.npz"
        r = score(path, lds50, angle)
        f, b = r["full"], r["3-14"]
        coh = "/".join(f"{c:.2f}" for c in f["coh"])
        lines.append(f"| {name} | {f['rmse']:.4f} | {b['rmse']:.4f} | {b['corr']:.4f} | {coh} | {b['sigma']:.4f} | "
                     f"{b['rmse'] / b['sigma']:.1f} | {b['wrap']:.3f} | {b['runtime']:.0f} |")
        print(lines[-1], flush=True)
    # cut-change audit on the baseline: residual jumps at cut instants vs elsewhere
    d = np.load(ROOT / f"results/tracks_phase{'_slice' if quick else ''}_abl0.npz")
    mm, _, _ = signal_from_tracks(ROOT / f"results/tracks_phase{'_slice' if quick else ''}_abl0.npz", angle, hp=3.0, lp=14.0)
    a = evaluate.align_and_score(mm, lds50)
    res = a.vision - a.lds
    jumps = np.abs(np.diff(res))
    ch = d["cut_changes_cable"] - max(a.lag_samples, 0)
    ch = ch[(ch > 0) & (ch < len(jumps))]
    lines.append("")
    other = np.ones(len(jumps), bool)
    other[ch] = False
    at_cuts = jumps[ch].mean() if len(ch) else float("nan")
    lines.append(f"Cut-change audit (baseline): mean |Δresidual| at the {len(ch)} cut-change instants "
                 f"{at_cuts:.4f} mm vs {jumps[other].mean():.4f} mm elsewhere (should be indistinguishable).")
    with open(ROOT / "results/ablations.md", "a") as fh:
        fh.write("\n".join(lines) + "\n")
    print("appended §E to results/ablations.md")


if __name__ == "__main__":
    main()
