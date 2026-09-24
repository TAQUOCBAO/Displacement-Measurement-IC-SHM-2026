# Results tables

All numbers against the 10 kHz laser displacement sensor (LDS), decimated to 50 Hz, mean-removed,
aligned by cross-correlation (0.20 s clock offset), 3000 overlapping samples. Source files:
`results/eval_<method>[_band].npz`, `results/ablations.md`, `results/egomotion.npz`.

## Trackers (3–14 Hz band, shared pipeline)

| tracker | RMSE (mm) | RMSE (px) | r | tracker σ (mm) | runtime |
|---|---|---|---|---|---|
| CoTracker3 online, 2× zoom, 9 points + support grid | 0.1145 | 0.129 | 0.964 | — | ~7 min (MPS) |
| ZNCC template matching (quadratic sub-pixel peak) | 0.0847 | 0.096 | 0.975 | — | ~2 min CPU |
| fused 0.8 ZNCC + 0.2 CoTracker3 | 0.0829 | 0.094 | 0.976 | — | |
| CoTracker3 + scene-adaptive refinement CNN | 0.0764 | 0.086 | 0.979 | — | minutes to train |
| **local phase (submitted)** | **0.0746** | **0.084** | **0.980** | 0.0085 | 29 s GPU / 5 min CPU |

Scale: 27.5 mm per checker square = 31.09 px → 0.885 mm/px.

## Local phase, per band

| output band | RMSE full (mm) | 0–3 Hz RMSE / coh. | 3–14 Hz RMSE / coh. | 14–25 Hz RMSE / coh. | LDS power in band |
|---|---|---|---|---|---|
| 0–25 Hz (unfiltered) | 0.2907 | 0.279 / 0.19 | 0.070 / 0.98 | 0.033 / 0.12 | 0.3 % / 99.6 % / 0.05 % |
| 3–14 Hz (submission) | 0.0746 | 0.022 / — | 0.068 / 0.975 | 0.010 / — | |

Band-edge sweep (RMSE, mm): 1–25 Hz 0.0901 · 2–25 Hz 0.0853 · 1–20 Hz 0.0873 · 2–14 Hz 0.0770 · **3–14 Hz 0.0746**.

## Ego-motion reference (local phase, 3–14 Hz)

| reference | RMSE 3–14 (mm) | r | coherence 3–14 | RMSE full (mm) | RMSE 14–25 (mm) |
|---|---|---|---|---|---|
| tripod board at the target's depth (1 track) | **0.0746** | 0.980 | 0.975 | 0.291 | 0.010 |
| dense background field, 40 tiles, Huber affine | 0.2613 | 0.807 | 0.861 | 1.051 | 0.088 |

Parallax go/no-go: field at the tripod board vs the board's own track, 3–14 Hz RMS = 0.313 px (affine and translation); gate 0.02 px → fail.
Rolling shutter: τ from the tiles 8.7 µs/row (r² 0.16), from six cable points −7.6 µs/row (r² 0.07) → inconsistent, not applied.
(An earlier run of this stage, job 759618, reported 0.193 px / 0.286 mm and a consistent τ = 12.7 µs/row (r² 0.98); it was invalidated by an ffmpeg frame-duplication bug in `stream_frames(start>0)`, fixed in `video_io.py`, and re-run from the frame cache.)

## Synthetic benchmark (300 frames, sinusoid, real frame-0 board), RMSE in model px

| amplitude (px) | phase, clean | ZNCC, clean | phase, H.264 | ZNCC, H.264 |
|---|---|---|---|---|
| 0.05 | 0.0005 | 0.0078 | 0.0088 | 0.0134 |
| 0.10 | 0.0005 | 0.0150 | 0.0100 | 0.0208 |
| 0.25 | 0.0006 | 0.0360 | 0.0125 | 0.0412 |
| 0.50 | 0.0006 | 0.0629 | 0.0149 | 0.0687 |
| 1.00 | 0.0006 | 0.0393 | 0.0130 | 0.0393 |
| 2.00 | 0.0010 | 0.0529 | 0.0110 | 0.0546 |

Native resolution (zoom ×1); the ×2 rows are in `results/synth/E4*.csv`.

## Ablations

Full table (14 variants, cut-change audit) in [`../results/ablations.md`](../results/ablations.md) §E; ego-motion §F; magnification QC §G.
