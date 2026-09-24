# UAV-Based Vision Displacement Measurement of a Stay Cable under Vortex-Induced Vibration

**Competition Project 1 — Project Report**
*Author: [name] — [affiliation / team]*

## 1. Overview

We estimate the in-plane displacement time history (perpendicular to the cable
axis) of the 55×55 mm checkerboard target mounted on the flexible stay cable,
from the 60 s DJI Mavic 3 Pro hover video (3840×2160 @ 50 fps), and evaluate it
against the 10 kHz laser displacement sensor (LDS) reference. The central
challenge is precision: the vibration amplitude is only ±1.3 mm, which at the
measured imaging scale of 0.884 mm/px corresponds to **±1.5 pixels** — every
component of the pipeline must preserve sub-pixel accuracy while simultaneously
rejecting UAV platform motion that is an order of magnitude larger than the
signal.

Three independent trackers are run over a shared pipeline:

1. a **local-phase tracker** — the phase of a complex steerable pyramid, after
   Chen et al. [2] and the phase-based magnification literature [3], solved as
   an amplitude-weighted least-squares problem with a Gauss–Newton refinement
   and linearised at a classical template-matching prior;
2. a **foundation-model tracker** — CoTracker3 [1] in online (streaming) mode,
   adapted for sub-pixel structural measurement; and
3. a **classical ZNCC template-matching tracker** with sub-pixel refinement,
   serving as baseline, fusion partner and coarse prior for (1).

The local-phase tracker achieves **RMSE = 0.075 mm (0.084 px) with correlation
0.980** against the LDS reference over the full 60 s record in the 3–14 Hz
structural band, with a per-frame uncertainty estimate of 0.006 mm.

## 2. Method

### 2.1 Frame acquisition and target detection

Frames are streamed directly from the H.264 video through an ffmpeg rawvideo
pipe (no intermediate files, no full-video memory load). A single 1344×1216
crop of the 4K frame covers both regions of interest: the cable target and a
**stationary reference checkerboard** mounted on a tripod near the same depth.

Both 2×2 checkerboards are localized automatically in frame 0: local
thresholding isolates their two black squares, connected-component analysis
yields the square centroids, and from these the board center, checker square
size in pixels, and board rotation follow directly. The known 27.5 mm checker
square measured 31.09 px, giving the scale factor 0.884 mm/px. No manual
annotation is used.

### 2.2 Tracking

**CoTracker3 (foundation model).** We use the official `cotracker3_online`
predictor via torch.hub. Two adaptations proved essential for sub-pixel
structural measurement:

- *Zoomed input.* CoTracker3's localization noise is approximately constant in
  model pixels (~0.15–0.2 px) irrespective of scene content. We therefore track
  inside a 256×192 crop upscaled 2× (Lanczos) to the model's native 512×384
  resolution, halving the noise expressed in real pixels. A 3× zoom degraded
  accuracy again — the model loses surrounding context — so 2× is the optimum
  for this scene.
- *Query design.* Nine well-textured board points are queried per target
  (center X-junction, the four square centroids, and the four edge
  T-junctions), plus a 48-point background support grid that stabilizes the
  joint transformer; support tracks are discarded. The per-frame rigid mean of
  the nine points forms the target track. All points remain visible in all
  3033 frames.

**Local-phase tracker (submitted).** Following Chen et al. [2], displacement
is read from the *local phase* of quadrature band-pass filters, which is linear
in sub-pixel translation and insensitive to illumination. Instead of the
9-tap G2/H2 pair (Table A1 of [2], whose taps we verified and use as a fast
path), the 192×192 board patch is decomposed by a complex steerable pyramid
(4 half-octave scales × 2 orientations, one-sided angular masks, tight frame,
reflect padding) [3]. For every band the wrap-safe phase change against the
frame-0 reference, Δφ = arg(S_t·S̄_0), enters the phase-constancy constraint
φ_x u + φ_y v + Δφ = 0 (Eq. 3 of [2]); weighting by the squared band amplitude
and the board mask gives 2×2 normal equations that pool ≈10⁴ effective
measurements per frame, followed by one Gauss–Newton pass in which the
reference spectrum is Fourier-shifted by the current estimate (removing the
0.01 px linearisation bias at 1 px shifts). Because the finest scale wraps at
±2.5 px while the target moves ±1.5 px on top of ±18 px platform drift, the
patch is re-cut on an integer, piecewise-constant grid derived from the
median-filtered ZNCC track (keeping the H.264 macroblock phase fixed) and the
phase is linearised at ZNCC's fractional estimate: the phase then only carries
ZNCC's ~0.1 px error, and no frame is wrap-flagged. Each frame also reports a
predicted σ (from the weighted residual, corrected for the bands' spatial
correlation), N_eff and cond(M) as self-diagnostics. All FFTs are batched on
the GPU (PyTorch), so the full 3033-frame video takes 29 s.

On a synthetic benchmark (checkerboard scene, sinusoidal motion, 300 frames)
the estimator is at 0.0005 px noise-free for 0.05–2 px amplitudes and at
0.009–0.015 px after H.264 round-trip at native resolution, versus
0.013–0.069 px for ZNCC; upscaling helps ZNCC but hurts the phase estimate, so
the phase tracker runs at native resolution.

**ZNCC baseline.** A 64×64 template fixed at frame 0 (no template update, so no
drift accumulation) is matched by zero-normalized cross-correlation in a search
window that follows the previous-frame peak; the correlation peak is refined by
a 2-D quadratic least-squares fit on its 5×5 neighborhood. The minimum
correlation over the full video is 0.977, confirming uninterrupted lock.

### 2.3 UAV motion compensation and projection

Platform motion (jitter and drift, ~±18 px over the record) is cancelled by
subtracting the reference-target track from the cable-target track per frame.
Because the two targets sit at slightly different depths, translation-only
compensation leaves a small residual parallax component concentrated below
1 Hz; the LDS spectrum carries <0.2 % of its power below 2 Hz, so a 4th-order
3 Hz high-pass removes this artifact without touching real motion. A 6th-order
14 Hz low-pass removes tracker noise above the highest significant structural
mode (the response is narrowband, with vortex-induced modes at 3.9, 7.5, and
11.7 Hz). Both filters are applied zero-phase.

The compensated pixel displacement is projected onto the in-plane normal of
the cable axis. The axis direction is measured from the image itself by PCA of
the red-cable color mask near the target (26.54° — consistent with the stated
26.95° physical inclination), then converted to mm with the detected scale.

### 2.4 Fusion and temporal alignment

The two trackers' error signals are only 57 % correlated, so a weighted average
(0.8 ZNCC + 0.2 CoTracker3, approximating inverse-variance weighting) reduces
the independent noise component below either tracker alone.

With the local-phase tracker available, fusion is no longer used for the
submission: its in-band error is below the fused result, and its predicted
noise (0.006 mm) is far below the residual, which is therefore dominated by
the ego-motion term common to all trackers rather than by independent tracker
noise that averaging could reduce.

The sub-second LDS/video clock offset noted in the data description is
estimated by cross-correlation over ±1 s, giving +0.20 s (video leading);
sub-sample refinement of the lag yielded no further improvement, indicating
alignment is not a residual error source. The submitted CSV
(`results/submission_phase_3-14.csv`, columns `time_s`, `displacement_mm`, 50 Hz) is
expressed on the LDS time base with the mean removed, per the data notes.

## 3. Results

Evaluation against the LDS reference decimated to 50 Hz (anti-aliased), over
the full overlapping 60 s window:

| Method | RMSE (mm) | RMSE (px) | Correlation |
|---|---|---|---|
| CoTracker3 (2× zoom, 9 queries + support grid) | 0.114 | 0.129 | 0.964 |
| ZNCC baseline | 0.085 | 0.096 | 0.975 |
| Fused ZNCC + CoTracker3 | 0.083 | 0.094 | 0.976 |
| CoTracker3 + scene-adaptive refinement net | 0.076 | 0.086 | 0.979 |
| **Local-phase tracker (3–14 Hz, submitted)** | **0.075** | **0.084** | **0.980** |
| Local-phase tracker, full band 0–25 Hz | 0.291 | 0.329 | 0.794 |

Per-band decomposition of the submitted signal (RMSE / magnitude-squared
coherence with the LDS): 0–3 Hz 0.022 mm / 0.17, 3–14 Hz 0.068 mm / 0.975,
14–25 Hz 0.010 mm / 0.12. The LDS carries 99.6 % of its power in 3–14 Hz and
only 0.34 % below 3 Hz; without the high-pass the vision signal shows a
0.28 mm low-frequency error with coherence 0.19 — the parallax between the
cable and reference targets, not cable motion. Above 14 Hz the LDS has 0.05 %
of its power and the vision signal only noise (0.033 mm), so the low-pass is
kept in all cases. Moving the high-pass edge to 2 Hz or 1 Hz costs +0.002 and
+0.015 mm respectively.

The predicted tracker uncertainty is 0.006 mm, ten times below the in-band
residual; the residual is therefore a systematic term shared by all trackers
(ego-motion/parallax leakage into the band, rolling-shutter timing between the
two targets), which is the object of the dense background ego-motion stage
currently being evaluated (Section 5).

The LDS signal std is 0.377 mm. Diagnostics (`results/eval_fused.png`): the
overlay tracks the LDS beat-by-beat through all amplitude-modulation envelopes;
the residual is white with no drift or transients; the vision and LDS spectra
coincide at all three structural peaks. The high correlation between the two
trackers' errors suggests the remaining residual is dominated by sources
external to tracking — LDS sensor noise (visible as a broadband floor in its
spectrum), rolling-shutter effects, and micro-vibration of the reference
tripod — rather than by tracker precision.

## 4. Reproducibility

All code is self-contained Python (numpy/scipy/PyTorch; video I/O via ffmpeg;
no OpenCV dependency); the phase branch is numpy/scipy-only on CPU and is
covered by a 41-test pytest suite (`tests/`, synthetic Fourier-shift, tight
frame, quadrature, σ-calibration and sign-convention checks). Model weights
for CoTracker3 are the public online checkpoint fetched via torch.hub. Full
reproduction of the submitted result:

```bash
pip install -r requirements.txt      # includes imageio-ffmpeg (static ffmpeg)
python scripts/run_zncc.py           # ~2 min CPU  (coarse prior)
python scripts/run_phase.py --device cuda      # 29 s GPU (~5 min on 8 CPU cores)
python scripts/run_pipeline.py phase --band 3-14   # alignment, RMSE, plots, submission CSV
```

On a PBS cluster the same steps are `qsub -v STAGE=phase_gpu job.pbs`. No
private or unreleased data, models, or software are used. The full
specification from which the phase branch was implemented, including its
acceptance gates and measured results, is `reference_paper/IMPLEMENTATION_PLAN.md`.

## 5. Dense ego-motion and rolling shutter: a negative result

We also implemented a dense background ego-motion field (40 static, textured
128 px tiles across the full 4K frame, tracked with the same phase estimator,
Huber-robust affine fit evaluated at the target; `egomotion.py`,
`scripts/run_egomotion.py`) and a rolling-shutter line-delay estimator. The
dense field fails its parallax go/no-go: evaluated at the tripod marker it
disagrees with the marker's own motion by 0.31 px RMS in 3–14 Hz, and used as
the ego-motion reference it raises the in-band RMSE from 0.075 to 0.261 mm
(full band 1.05 mm). The explanation is depth: the marker stands at the same
≈ 2 m range as the cable target, whereas the background is several times
farther, so the drone's translational jitter produces different image shifts
on the two (Δx = f·Δt/Z — 0.3 px at 2 m corresponds to only ≈ 0.2 mm of drone
motion). A same-depth reference cancels this parallax exactly; a distant
background cannot, however densely it is sampled. We therefore keep the
single same-depth reference and report the dense field only as a diagnostic.
The rolling-shutter estimates from the tiles and from six points along the
cable do not agree (8.7 vs −7.6 µs/row, r² 0.16 / 0.07), so no line-delay
correction is applied. The remaining in-band residual (0.068 mm, ten times the
tracker's predicted σ) is thus the parallax between the *two* targets'
slightly different depths plus reference-marker micro-vibration, which no
single-camera method can separate without a second same-depth reference.
Phase-based motion magnification with the Row-GDGIF of Yang & Jiang [4]
(`magnify.py`) is used only for visual verification of the mode shapes, not
for measurement: clips of the 3.9 Hz and 11.7 Hz modes (α = 20 / 50) confirm a
rigid in-plane oscillation of the board, and the Row-GDGIF improves SSIM/PSNR
against the stabilised input only marginally (0.790 → 0.796, 24.6 → 24.7 dB at
11.7 Hz; `results/ablations.md` §G).

## References

[1] Karaev, N., Makarov, I., Wang, J., Neverova, N., Vedaldi, A., Rupprecht, C.
"CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real
Videos." arXiv:2410.11831, 2024. https://github.com/facebookresearch/co-tracker

[2] Chen, J.G., Wadhwa, N., Cha, Y.-J., Durand, F., Freeman, W.T., Buyukozturk,
O. "Modal identification of simple structures with high-speed video using
motion magnification." Journal of Sound and Vibration 345 (2015) 58–71.

[3] Wadhwa, N., Rubinstein, M., Durand, F., Freeman, W.T. "Phase-based video
motion processing." ACM Trans. Graph. 32(4), 2013 (complex steerable pyramid
after Simoncelli & Freeman, 1995).

[4] Yang, Y., Jiang, S. "Phase-based video motion magnification with an
optimized 1D row guided dynamic gradient image filter." Mechanical Systems and
Signal Processing 215 (2024).

[5] FFmpeg (video decoding, static build via imageio-ffmpeg), NumPy/SciPy
(signal processing), PyTorch (model runtime and batched FFTs) — standard
open-source software, versions pinned in `requirements.txt`.
