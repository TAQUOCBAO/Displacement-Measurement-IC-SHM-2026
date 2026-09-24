"""Package the competition deliverable: validated 50 Hz CSV + code + README with citations.

Usage: python scripts/make_submission.py <method_key> [--out submission]
    <method_key> selects results/submission_<method_key>.csv, e.g. phase_3-14 or phase_dense_3-14.
Checks the CSV contract of the task (columns time_s, displacement_mm; 50 Hz; finite; ~60 s),
copies it to <out>/displacement.csv together with the code needed for the reproducibility
check (src/, scripts/, tests/, job.pbs, requirements.txt, report.md, evaluation figure) and
writes <out>/README.md (method, reproduce commands, external resources) + MANIFEST.txt.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FPS = 50.0

README = """# Project 1 — UAV-based vision displacement measurement: submission

`displacement.csv` — columns `time_s`, `displacement_mm`, sampled at 50 Hz ({n} samples,
{dur:.2f} s), in-plane displacement of the cable target perpendicular to the cable axis,
mean-removed, time base aligned to the LDS record start (video/LDS clock offset estimated by
cross-correlation, see `report.md` §2.4). Method key: `{key}`.

## Method (see `report.md` for the full description and results)

Local-phase tracker on a complex steerable pyramid (Chen et al. 2015 / Yang & Jiang 2024):
phase-constancy constraint solved by amplitude-weighted least squares over 4 scales × 2
orientations with a Gauss–Newton refinement, linearised at a ZNCC template-matching estimate;
UAV ego-motion cancelled with the stationary reference target ({ego}); projection onto the
cable-axis normal; 27.5 mm-per-square scaling; 3–14 Hz band-pass around the VIV modes.

## Reproduce

```bash
pip install -r requirements.txt          # numpy, scipy, pandas, matplotlib, imageio-ffmpeg (+ torch for --device cuda)
python scripts/run_zncc.py               # ZNCC prior            -> results/tracks_zncc.npz
python scripts/run_phase.py [--device cuda]   # phase tracker    -> results/tracks_phase.npz
python scripts/run_pipeline.py phase --band 3-14              -> results/submission_phase_3-14.csv
python scripts/make_submission.py phase_3-14                   # this package
```
On a PBS cluster: `qsub -q gpus -l select=1:ncpus=8:ngpus=1:mem=64gb -v STAGE=phase_gpu job.pbs`.
Data expected at `data/Video.MP4` and `data/LDS data.xlsx` (not included). Unit tests: `python -m pytest tests`.

## External resources used

- Chen, J.G., Wadhwa, N., Cha, Y.-J., Durand, F., Freeman, W.T., Buyukozturk, O. (2015).
  Modal identification of simple structures with high-speed video using motion magnification.
  *J. Sound Vib.* 345, 58–71 — local-phase displacement estimator (Eqs. 1–5).
- Yang, Y., Jiang, S. (2024). Phase-based video motion magnification with an optimized 1D row
  guided dynamic gradient image filter. *Mech. Syst. Signal Process.* 215 — steerable-pyramid PVMM
  and Row-GDGIF (used for visual QC only).
- Simoncelli, E.P., Freeman, W.T. (1995) / Wadhwa, N. et al. (2013) — complex steerable pyramid
  (re-implemented from scratch in `src/uav_disp/steerable.py`).
- Karaev, N. et al. (2024). CoTracker3 — https://github.com/facebookresearch/co-tracker, checkpoint
  `scaled_online.pth` via torch.hub (baseline tracker, not part of the submitted signal).
- Software: FFmpeg (via `imageio-ffmpeg`), NumPy, SciPy, pandas, Matplotlib, PyTorch (batched-FFT backend).
- No public dataset, private data or pretrained model enters the submitted displacement; only the
  provided video and LDS files were used.

## Contents

See `MANIFEST.txt` (paths + SHA-256).
"""


def validate(csv: Path) -> np.ndarray:
    with open(csv) as fh:
        header = fh.readline().strip()
    assert header == "time_s,displacement_mm", f"bad header {header!r}"
    d = np.loadtxt(csv, delimiter=",", skiprows=1)
    assert d.ndim == 2 and d.shape[1] == 2, f"bad shape {d.shape}"
    t, x = d[:, 0], d[:, 1]
    dt = np.diff(t)
    assert np.allclose(dt, 1 / FPS, atol=1e-6), f"not uniformly 50 Hz: dt in [{dt.min():.6f}, {dt.max():.6f}]"
    assert abs(t[0]) < 1e-9, f"time does not start at 0 ({t[0]})"
    assert np.all(np.isfinite(x)), "non-finite displacement values"
    assert 55 <= t[-1] <= 61, f"unexpected duration {t[-1]:.2f} s"
    assert abs(x.mean()) < 0.05, f"signal not mean-removed (mean {x.mean():.4f} mm)"
    assert 0.1 < x.std() < 5, f"implausible amplitude (std {x.std():.3f} mm)"
    return d


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("key", help="method key, e.g. phase_3-14")
    ap.add_argument("--out", default="submission")
    a = ap.parse_args()
    src = ROOT / f"results/submission_{a.key}.csv"
    d = validate(src)
    print(f"{src.name}: {len(d)} samples, {d[-1, 0]:.2f} s, std {d[:, 1].std():.3f} mm  [OK]")

    out = ROOT / a.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    shutil.copy(src, out / "displacement.csv")
    for rel in ["src", "scripts", "tests"]:
        shutil.copytree(ROOT / rel, out / rel, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.md"))
    for rel in ["job.pbs", "requirements.txt", "report.md", "README.md", "results/ablations.md",
                f"results/eval_{a.key}.png", "reference_paper/IMPLEMENTATION_PLAN.md"]:
        p = ROOT / rel
        if p.exists():
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, out / rel)
        else:
            print(f"  (missing, skipped) {rel}")
    ego = "dense background tile field" if "dense" in a.key else "reference-target track subtraction"
    (out / "README.md").write_text(README.format(n=len(d), dur=d[-1, 0], key=a.key, ego=ego))
    files = sorted(p for p in out.rglob("*") if p.is_file())
    (out / "MANIFEST.txt").write_text("\n".join(f"{sha(p)}  {p.relative_to(out)}" for p in files) + "\n")
    print(f"wrote {out}/ ({len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
