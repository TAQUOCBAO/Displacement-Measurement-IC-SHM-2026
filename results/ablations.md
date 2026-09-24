# Ablations vs LDS reference (full 60 s)

## A. Ego-motion compensation (band-pass 3-14 Hz applied in all rows)

| tracker | compensation | RMSE (mm) | corr |
|---|---|---|---|
| zncc | none | 0.2490 | 0.8274 |
| zncc | translation | 0.0847 | 0.9746 |
| cotracker | none | 0.2452 | 0.7908 |
| cotracker | translation | 0.1145 | 0.9636 |
| cotracker | similarity (9-pt) | 0.5723 | 0.4637 |

## B. Band-pass ablation (translation compensation)

| tracker | filter | RMSE (mm) | corr |
|---|---|---|---|
| zncc | none | 0.2988 | 0.7890 |
| zncc | HP 3 Hz only | 0.0989 | 0.9674 |
| zncc | LP 14 Hz only | 0.2944 | 0.7834 |
| zncc | band 3-14 Hz | 0.0847 | 0.9746 |
| cotracker | none | 0.3110 | 0.7167 |
| cotracker | HP 3 Hz only | 0.1168 | 0.9562 |
| cotracker | LP 14 Hz only | 0.3101 | 0.7088 |
| cotracker | band 3-14 Hz | 0.1145 | 0.9636 |

## C. Fusion weight (w * zncc + (1-w) * cotracker)

| w | RMSE (mm) |
|---|---|
| 0.0 | 0.1145 |
| 0.1 | 0.1081 |
| 0.2 | 0.1022 |
| 0.3 | 0.0969 |
| 0.4 | 0.0923 |
| 0.5 | 0.0885 |
| 0.6 | 0.0856 |
| 0.7 | 0.0837 |
| 0.8 | 0.0829 |
| 0.9 | 0.0833 |
| 1.0 | 0.0847 |

Best: w = 0.8 -> 0.0829 mm

## D. Error budget

- LDS self-noise in the 3-14 Hz scoring band (from its 20-40 Hz PSD floor): **0.0018 mm RMS** (irreducible for any vision method)
- Inter-tracker error correlation: **0.567** (shared floor: LDS noise, rolling shutter, reference-target micro-vibration)

## E. Phase branch (Chen 2015 / Yang 2024) — full-band and 3–14 Hz vs LDS

| variant | RMSE full (mm) | RMSE 3–14 (mm) | corr 3–14 | coh 0–3 / 3–14 / 14–25 | mean σ (mm) | RMSE/σ | wrap-risk | runtime (s) |
|---|---|---|---|---|---|---|---|---|
| baseline (A2, board, piecewise, LSQ, 4 scales, 2 orients) | 0.2907 | 0.0746 | 0.9802 | 0.19/0.98/0.12 | 0.0085 | 8.7 | 0.000 | 19 |
| scales finest only | 0.2875 | 0.0758 | 0.9796 | 0.19/0.97/0.12 | 0.0146 | 5.2 | 0.000 | 12 |
| scales 0,1 | 0.2891 | 0.0749 | 0.9800 | 0.19/0.98/0.12 | 0.0113 | 6.6 | 0.000 | 13 |
| scales 0,1,2 | 0.2904 | 0.0745 | 0.9802 | 0.19/0.98/0.12 | 0.0097 | 7.7 | 0.000 | 14 |
| orientations 4 | 0.2905 | 0.0743 | 0.9803 | 0.19/0.98/0.12 | 0.0073 | 10.2 | 0.000 | 15 |
| weights A2+block | 0.2900 | 0.0746 | 0.9802 | 0.19/0.98/0.12 | 0.0096 | 7.8 | 0.000 | 16 |
| weights A2+GDGIF-Gamma | 0.2928 | 0.0758 | 0.9795 | 0.19/0.97/0.13 | 0.0151 | 5.0 | 0.000 | 15 |
| mask board+surround | 0.2924 | 0.0821 | 0.9776 | 0.18/0.97/0.12 | 0.0153 | 5.4 | 0.092 | 15 |
| cut per-frame | 0.2908 | 0.0743 | 0.9803 | 0.19/0.98/0.13 | 0.0080 | 9.2 | 0.000 | 15 |
| estimator Chen Eq.4/5 | 0.2918 | 0.0747 | 0.9801 | 0.19/0.98/0.12 | 0.0085 | 8.8 | 0.000 | 18 |
| decimate 2 (Chen-style) | 0.2927 | 0.0751 | 0.9799 | 0.19/0.98/0.13 | 0.0148 | 5.1 | 0.000 | 17 |
| decimate 4 (Chen-style) | 0.2891 | 0.0842 | 0.9749 | 0.16/0.97/0.11 | 0.0491 | 1.7 | 0.000 | 16 |
| no Gauss-Newton refinement | 0.2907 | 0.0746 | 0.9802 | 0.19/0.98/0.12 | 0.0086 | 8.7 | 0.000 | 11 |
| no ZNCC prior (wrap-limited) | 0.3715 | 0.1617 | 0.9044 | 0.15/0.93/0.11 | 0.0119 | 13.6 | 0.435 | 13 |

Cut-change audit (baseline): mean |Δresidual| at the 291 cut-change instants 0.0676 mm vs 0.0659 mm elsewhere (should be indistinguishable).

## F. Ego-motion reference (job 759618) — phase tracker, 3–14 Hz and full band vs LDS

| reference for ego-motion | RMSE 3–14 (mm) | corr 3–14 | coh 3–14 | RMSE full (mm) | RMSE 14–25 (mm) |
|---|---|---|---|---|---|
| tripod marker (same depth as target, 1 track) | **0.0746** | 0.980 | 0.975 | 0.291 | 0.010 |
| dense background field (40 tiles, affine) | 0.2613 | 0.807 | 0.861 | 1.051 | 0.088 |

G2 parallax test: field-at-marker vs marker 0.313 px RMS in 3–14 Hz (gate 0.02) — the background is
not at the target's depth, so its field carries UAV-translation parallax the marker does not. G3: τ from
the tiles 8.7 µs/row (r² 0.16) vs −7.6 µs/row from the cable points (r² 0.07): inconsistent, not applied.
Numbers from job 759638 (frame cache). Job 759618 (0.193 px, 0.286 mm, τ 12.7 µs/row r² 0.98) was
invalidated by ffmpeg duplicating the first trimmed frame in `stream_frames(start=1)` (fixed).

## G. Phase-based motion magnification (visual QC only; jobs 759614 / 759624)

Crop 512² around the cable target, stabilised with the marker track, complex steerable pyramid
(5 scales), zero-phase Butterworth band-pass of the unwrapped phase, amplitude-weighted phase denoise,
Row-GDGIF (h=16, μ=0.022) on the two finest scales. Outputs `results/magnified_<band>_a<α>[_gdgif].mp4`,
`results/fig_magnify_{slice,ssim,canny}_<band>.png`.

| band (Hz) | α | frames | SSIM / PSNR vs stabilised input, PVMM | PVMM + GDGIF |
|---|---|---|---|---|
| 3.7–4.1 (mode 1) | 20 | 1000 | see `fig_magnify_ssim_3.7-4.1.png` | — |
| 11.5–11.9 (mode 3) | 50 | 500 | 0.790 / 24.6 dB | 0.796 / 24.7 dB |

GDGIF raises SSIM/PSNR marginally (+0.006 / +0.08 dB), the direction Yang & Jiang report; at α=50 the
space-time slice shows the 11.7 Hz oscillation clearly but with ringing around the board edges — the
clips are used only to confirm the mode is a rigid in-plane oscillation of the board, never for
measurement. Runtime after batching the FFT/temporal ops: ~15 min per band on 16 cores (500 frames).
