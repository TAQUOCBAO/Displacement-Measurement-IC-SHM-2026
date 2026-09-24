# Implementation Plan & Code-Generation Spec — Phase-Based Sub-Pixel Displacement from UAV Video

**Derived from:** `reference_paper/Modal identification.pdf` (Chen, Wadhwa, Cha, Durand, Freeman,
Buyukozturk, *J. Sound Vib.* 345 (2015) 58–71) and `reference_paper/phase-based video.pdf`
(Yang & Jiang, *Mech. Syst. Signal Process.* 215 (2024) 111429).
**Supersedes:** `../SPEC_phase_displacement.md` (draft; see §1.4 for what was corrected).
**Status:** implemented (P0–P4 code + 41 unit tests in `tests/`); see §12 for what changed during
implementation and the measured results. `src/uav_disp/` has the working ZNCC / CoTracker3 / refine-net
pipeline that this plan extends.

---

## 0. How to use this document

Generate **one module per section (§3–§8)**, in the phase order of §10. Every module section has the
same shape so it can be handed to a code generator on its own:

1. *Purpose & paper source* — which equations it implements.
2. *Public API* — exact signatures, dtypes, shapes, coordinate conventions.
3. *Algorithm* — step-by-step, with the numerical choices fixed.
4. *Invariants / asserts* — what the code must check at construction or per call.
5. *Tests* — file, test name, tolerance. Tests are the acceptance criteria.
6. *Done when* — the gate for moving on.

### Global conventions (apply everywhere)

| Item | Convention |
|---|---|
| Coordinates | `(x, y)` = (column, row); image y points **down**; angle `atan2(dy, dx)` |
| Frame | all target positions are **ROI-local px**, ROI = `config.ROI = (1600, 600, 1344, 1216)` of the 3840×2160 frame |
| Reference frame | **frame 0** is the fixed reference for every phase difference (Chen: "velocity between the i-th frame and the first frame") |
| Tracks | `(T, 2)` float64 arrays of `(x, y)`; `T = 3033` for the full video |
| Images | `float32` for filtering; `uint8` only at I/O boundaries; sub-bands `complex64` |
| Luma | `Y = 0.299 R + 0.587 G + 0.114 B` (ITU-R 601 — Yang's YIQ Y-channel; already used in `scripts/run_synth_benchmark.get_scene`) |
| Sign of displacement | **fixed by test, not by assumption**: after `synth.fourier_shift(img, dx, dy)` the content that was at `x` is at `x + dx`; the estimator must return `u = +dx`. |
| Randomness | every test/synthetic path takes a `seed` and uses `np.random.default_rng(seed)` |
| Dependencies | phase branch = **numpy + scipy only** (must run on the login node's `python3` 3.9 / numpy 1.26 / scipy 1.13 *and* on numpy 2.x). `torch` only for CoTracker/refine-net code paths. **No cv2.** ffmpeg via `module load ffmpeg/4.2.2-gcc-sr35`. |
| numpy 2 hygiene | no `np.float`/`np.int`, `np.trapezoid` not `np.trapz`, `np.fft` only, `.astype(np.float32, copy=False)` |

### Existing code you must reuse (verified symbols, `src/uav_disp/`)

| Symbol | Signature / contract |
|---|---|
| `video_io.stream_frames(path=VIDEO_PATH, crop=None, gray=False, start=0, count=None, scale_to=None)` | yields `uint8 (h,w)` if gray else `(h,w,3)` RGB; `crop=(x,y,w,h)` in full-frame px; ffmpeg pipe |
| `video_io.FPS=50.0, FRAME_W=3840, FRAME_H=2160, N_FRAMES=3033` | constants |
| `detect.find_board(gray, approx_xy, approx_square_px) -> Board` | `Board.center (x,y)`, `.square_px`, `.angle_deg`, `.black_centroids (2,2)`, `.white_centroids (2,2)`, `.query_points() -> (9,2)` |
| `config.CABLE_APPROX=(300,290)`, `config.REF_APPROX=(907,738)`, `config.APPROX_SQUARE_PX=30`, `config.SQUARE_MM=27.5`, `config.FPS`, `config.HIGHPASS_HZ=3`, `config.LOWPASS_HZ=14`, `config.MAX_LAG_S=1` | scene constants |
| `postprocess.cable_axis_angle(rgb_frame, center_xy, half=220) -> float rad` | PCA of red mask `(r>90)&(r>1.6g)&(r>1.6b)` |
| `postprocess.displacement_mm(cable_xy, ref_xy, square_px, axis_angle_rad, highpass_hz=3, lowpass_hz=14, fps=50) -> (T,) mm` | subtract ref, project on `[sin a, -cos a]`, scale, zero-phase Butterworth (4th HP, 6th LP), zero-mean |
| `evaluate.align_and_score(vision, lds50) -> Alignment(lag_samples, sign, rmse, corr, vision, lds)` | ±1 s cross-correlation alignment, sign fix, RMSE/corr on overlap |
| `lds.load(npy_path, xlsx_path=None)`, `lds.decimate_to(x, fs_out, fs_in=10_000)` | LDS reference at 50 Hz |
| `synth.PAD=8`, `synth.fourier_shift(img, dx, dy)`, `synth.sinusoid_traj(n, amp_px, freq_hz, fps, direction)`, `synth.make_sequence(source, traj_xy, noise_sigma=1.5, seed=0)`, `synth.h264_roundtrip(frames, fps, tmp_path, bits_per_px=0.4)` | exact-ground-truth benchmark machinery |
| `track_zncc.track(frames, init_centers, template_half=32, search_rad=24) -> {name:(T,2), name_peak:(T,)}` | coarse prior source |
| `results/tracks_zncc.npz` keys: `cable, ref, cable_peak, ref_peak, cable_square_px, cable_angle_deg, ref_square_px` | coarse prior file (exists) |
| `scripts/run_pipeline.py::signal_from_tracks` needs `cable (T,2)`, `ref (T,2)`, `cable_square_px` in `results/tracks_<method>.npz` | output contract of `run_phase.py` |
| `results/frame0_gray.npy` — `(1216, 1344) uint8` ROI frame 0 | **video-free test fixture** |
| `results/synth/E1C.csv` — ZNCC H.264 zoom-1 amp-0.5 = **0.0692 px**, zoom-2 = **0.0185 px** | G0 reference numbers |

---

## 1. Paper review — what is taken, what is rejected

### 1.1 Chen et al. 2015 (the accuracy engine)

| Paper item | Content | Use here |
|---|---|---|
| §2.3 Eq. (1) | `A_θ e^{iφ_θ} = (G2^θ + i H2^θ) ⊗ I` — local amplitude/phase from a quadrature pair | `steerable.g2h2_bank()` fast path; the complex steerable pyramid is the multi-scale generalisation (Chen: "different pyramid levels could be used to capture the signals at various spatial scales") |
| Eq. (2)–(3) | constant phase contours move with the object: `(∂φ/∂x, ∂φ/∂y, ∂φ/∂t)·(u, v, 1) = 0` | **the** constraint we solve, per pixel, per sub-band, jointly (§4) |
| Eq. (4)–(5) | `u = −(∂φ₀/∂x)⁻¹ ∂φ₀/∂t`, `v = −(∂φ_{π/2}/∂y)⁻¹ ∂φ_{π/2}/∂t` — the approximation `∂φ₀/∂y ≈ 0` | kept as `chen_mode=True` ablation; default solver uses the full Eq. (3) in weighted least squares (no division by a near-zero gradient) |
| §2.3 text | displacement is i-th frame vs **first** frame; SNR raised by a **local-amplitude-weighted spatial average**; px→mm by object length / px span | frame-0 reference; `w = A₀²`; `config.SQUARE_MM / square_px` |
| §2.3 text | "the video sequence is downsampled four times in each dimension spatially prior to application of the filters" | ablation row `PHASE_DECIMATE ∈ {1, 2, 4}` — the native-resolution claim must be measured, not assumed |
| §2.4 | "virtual accelerometers": crops anywhere on the structure give independent displacement signals; resonant peaks from FFT; magnification bands chosen from those peaks; α ∝ 1/relative peak amplitude | `egomotion.py` tiles are virtual accelerometers on the background; cable-point crops give the mode-shape cross-check (§6 QC) |
| §3.2 | noise floor **1×10⁻⁵ px/√Hz** (5000 fps, raw); MAC 95–98 % vs accelerometers | report our floor the same way (§9.5) |
| §2.2 / §4.3 | ODS from Canny edges of magnified frames | `magnify.canny_edges` QC figure |
| App. A Table A1 | 9-tap separable G2/H2 filters | **corrected taps in §3.4 / App. A** |
| §6 | open problem: "the effects of camera movement" | this project's contribution (§7) |

### 1.2 Yang & Jiang 2024 (pyramid, artifact suppression, validation protocol)

| Paper item | Content | Use here |
|---|---|---|
| §2 | process the **Y channel of YIQ** only | luma everywhere |
| §2.1 Eq. (1)–(5) | complex steerable pyramid sub-bands `s_{r,θ,t} = a e^{iφ}`; phase difference `arg(a e^{iφ(x+δ)}) − arg(a e^{iφ(x)})`; magnified band `s·e^{iαΔφ}` | `steerable.py`, `magnify.py`. Eq. (3) as written is a raw phase subtraction — we always use the wrap-safe `arg(S_t · conj(S_0))` (identical when \|Δφ\|<π, correct otherwise) |
| §2.2 | diagnosis: **stripe noise and double edges live in the high-frequency sub-bands**; fix = 1D **row**-window guided filter (not a square window) | `gdgif.row_filter` applied to the two finest scales after magnification; also usable as a per-pixel reliability weight for measurement (§4.4) |
| Eq. (6)–(7) | `H_i = ā_rk u_i + b̄_rk`, coefficients averaged over the row window | §5 |
| Eq. (8)–(9) | cost `Σ[(a u_i + b − u_i)² + (μ/Γ)(a − γ_rk)²]`; `γ_rk = 1 − 1/(1+e^{η(χ(rk) − ξ_χ∞)})`, `χ = σ_{u,1}·σ_{u,16}`, `η = 4/(ξ_χ∞ − min χ)`, `ξ_χ∞ = mean χ` | §5 |
| Eq. (10) | `Γ_u(i) = (1/N) Σ_k (χ(k)+ε)/(χ(i)+ε)` | §5, closed form `(mean χ + ε)/(χ(i)+ε)` |
| Eq. (11) | `a_rk = (ξ_{u*X,16} − ξ_{u,16} ξ_{X,16} + (μ/Γ) γ_rk) / (σ²_{u,16} + μ/Γ)`, `b_rk = ξ_{X,16} − a_rk ξ_{u,16}` | §5 |
| §3.1.1 | `h = 16`, `μ = 0.022`; α = 10 (shaker), 50 (structures); bands preset around the known excitation | `config.GDGIF_H, GDGIF_MU, MAGNIFY_BANDS` |
| §3.1.3 | SSIM & PSNR on Y (max 255), per frame and averaged; their Table 3: SSIM 0.86–0.92, PSNR 28–35 dB | `magnify.ssim_psnr` |
| §3.1.4 | Canny on key frames: double edge present in PVMM, absent with GDGIF | `magnify.canny_edges` figure |
| Figs. 3–6 | space–time slice diagrams | `magnify.space_time_slice` |
| **Eq. (12)–(14)** | `I_R = mean_p I_p` (Horn–Schunck intensity), `D(t) = I_R · t · 1/f`, FFT | **NOT implemented for measurement.** `D(t)` is dimensionally incoherent, uncalibrated, and never validated against a contact sensor in the paper (their Fig. 8 "displacement" is the input curve scaled). Magnification is a deterministic operator on the same pixels: it cannot add information. Measurement here comes from the phase directly (Chen). A magnified-video intensity trace is optionally produced as a *visual* QC panel only. |
| §4 | open problem: "camera shake and complex background motions" | this project's contribution (§7) |

### 1.3 The gap both papers leave open = this project

Neither method has been run from a moving platform. Contribution targeted: **phase-based sub-pixel
metrology on a hovering UAV, native 4K, ±1.4 px signal, against a 10 kHz laser**, with a
principled per-frame uncertainty, **over the full 0–25 Hz band** (all existing results are band-limited to
3–14 Hz — see §4.7). Current best to beat: **0.0764 mm (0.086 px)** (3–14 Hz only).

### 1.4 Corrections to the previous draft (`SPEC_phase_displacement.md`)

1. **Chen Table A1 `G_f1` taps ±3 are `+0.1148`, not `−0.1148`.** Verified two ways: read from the
   PDF image, and the analytic Freeman–Adelson profile `0.9213(2x²−1)e^{−x²}` at `x = k·0.67`
   reproduces every tap. Sign pattern is `+ + + − − − + + +`.
2. `G_f2` tap +3 printed as `0.0480` in the paper is a typo — the filter is symmetric, use `0.0176`.
3. Quadrature tolerance: with the exact taps the `|H1|/|G1|` magnitude ratio spans 0.92–1.13 over
   0.1–0.4 cycles/px (the 9-tap design is only approximately a Hilbert pair); the assert is
   `|ratio − 1| < 0.15`, phase `−π/2 ± 0.05 rad` (measured: −1.571 rad flat).
4. Chen's 4× spatial downsampling is now an explicit ablation row.
5. GDGIF variable roles made explicit (§5): both input `X` and guide `u` are the magnified sub-band
   (self-guided); filtering applied to the **real-valued re-synthesised sub-band**, not to phase.
6. Environment facts (§9.1) replace the "Apple MPS" assumptions.

---

## 2. Architecture and data flow

```
src/uav_disp/
  steerable.py   complex steerable pyramid (frequency domain) + Chen G2/H2 fast path        §3
  phase_disp.py  local-phase displacement estimator + per-frame uncertainty                 §4
  gdgif.py       Yang's optimized 1D Row GDGIF                                             §5
  magnify.py     PVMM renderer + QC (slice / SSIM / PSNR / Canny)                          §6
  egomotion.py   dense background field -> affine ego-motion + rolling-shutter tau          §7  (beyond the papers)
scripts/
  run_phase.py             -> results/tracks_phase[_dense].npz                              §8
  run_egomotion.py         -> results/egomotion.npz
  run_magnify.py           -> results/magnified_*.mp4 + results/fig_magnify_*.png
  run_phase_ablations.py   -> appends to results/ablations.md
  run_synth_benchmark.py   + experiment E4 and a `phase` tracker
  run_pipeline.py          + methods `phase`, `phase_dense`; `--band` (full-band default, §4.7); σ / cond / N_eff / band-coherence diagnostics
src/uav_disp/evaluate.py   + band_scores()  (per-band RMSE + coherence vs LDS)                §4.7
tests/
  conftest.py test_steerable.py test_phase_disp.py test_gdgif.py test_magnify.py test_egomotion.py test_evaluate_bands.py
```

```
Video.MP4 ──stream_frames(crop=ROI, gray=True)──┐
                                                ├─ coarse integer cuts  (ZNCC track, median k=51,
                                                │   piecewise-constant, re-cut only when drift ≥ 1 px)
                          ┌─────────────────────┴───────────────────┐
                   target patches 192² (cable, ref)        background tiles 128² (full-4K pass, §7)
                          │                                          │
                   steerable.decompose (per sub-band)         same, per tile
                          │                                          │
                   phase_disp.solve_patch  → uv, σ, N_eff, cond      → per-tile uv, σ
                          │                                          │
            cable = cut + uv  (single-ref path: ref = cut + uv)    robust affine field, evaluated
                          │                                          at the cable target, RS-delayed
                          └────────────────┬─────────────────────────┘
                     postprocess.displacement_mm (subtract ref/ego, project on cable normal, mm; FULL BAND by default, 3–14 Hz optional)
                                           │
                             evaluate.align_and_score vs LDS  →  RMSE, corr, σ-calibration
```

Memory rule (binding): **never hold more than one `(T, H, W)` complex sub-band at a time.**
192²×3033×complex64 = 894 MB per band; the patch stack itself (float32) is 447 MB per target.
Accumulate the normal equations incrementally across sub-bands (§4.3).

---

## 3. `steerable.py` — filter bank

**Source:** Simoncelli & Freeman 1995 / Portilla & Simoncelli 2000 (pyramid), Wadhwa et al. 2013
(complex, one-sided), Chen App. A (G2/H2), Yang §2.1.

### 3.1 API

```python
@dataclass(frozen=True)
class PyramidSpec:
    n_scales: int = 4           # number of band-pass levels (excluding hi/lo residuals)
    n_orients: int = 2          # K; orientations theta_k = k*pi/K   (2 for measurement, 4 for magnification)
    half_octave: bool = True    # radial spacing 2^(-1/2) (True) or 2^(-1) (False)
    twidth: float = 1.0         # raised-cosine transition width in octaves
    pad: int | None = None      # reflect pad per side; None -> H//2, W//2

@dataclass
class Filters:
    shape: tuple[int, int]            # (H, W) of the *unpadded* image
    padded: tuple[int, int]           # (Hp, Wp)
    pad: tuple[int, int]              # (py, px)
    hi: np.ndarray                    # (Hp, Wp) float32, real, radial hi-pass residual mask
    lo: np.ndarray                    # (Hp, Wp) float32, real, radial lo-pass residual mask
    bands: list[np.ndarray]           # len n_scales*n_orients, (Hp, Wp) float32 one-sided masks; index = s*K + k
    scale_of: np.ndarray              # (n_bands,) int  scale index s per band (0 = finest)
    orient_of: np.ndarray             # (n_bands,) int  k per band
    omega_x: np.ndarray               # (Hp, Wp) float32 angular frequency (rad/px), fftfreq*2pi
    omega_y: np.ndarray
    spec: PyramidSpec

def build_filters(shape: tuple[int, int], spec: PyramidSpec) -> Filters: ...
def decompose(img: np.ndarray, F: Filters) -> list[np.ndarray]:
    """img (H,W) float32 -> [hi (H,W) float32, band_0..band_{n-1} (H,W) complex64, lo (H,W) float32]
    Sub-bands are cropped back to (H, W)."""
def decompose_band(img_fft_padded: np.ndarray, F: Filters, b: int) -> np.ndarray:
    """One band (complex64, cropped) from a cached padded FFT — the incremental path used by phase_disp."""
def padded_fft(img: np.ndarray, F: Filters) -> np.ndarray:
    """reflect-pad by F.pad then fft2 -> (Hp, Wp) complex64"""
def reconstruct(parts: list[np.ndarray], F: Filters) -> np.ndarray:
    """inverse of decompose; returns (H, W) float32"""
def band_gradient(band_fft_padded: np.ndarray, F: Filters) -> tuple[np.ndarray, np.ndarray]:
    """analytic spatial derivatives of a sub-band: ifft(i*omega_x*B), ifft(i*omega_y*B), cropped."""

# Chen fast path
G_F1, G_F2, H_F1, H_F2: np.ndarray   # 9-tap float64 (App. A, corrected)
def g2h2_bank() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{'0': (G2^0, H2^0), 'pi/2': (G2^{pi/2}, H2^{pi/2})} as 9x9 separable outer products."""
def g2h2_decompose(img: np.ndarray) -> list[np.ndarray]:
    """[S_0, S_pi2] complex64 (H,W): S = conv(G2) + i*conv(H2), 'reflect' boundary (scipy.ndimage.convolve)."""
```

### 3.2 Algorithm (frequency-domain pyramid)

Let `(Hp, Wp)` = padded shape; `ω_x, ω_y` from `np.fft.fftfreq(Wp)*2π`, `np.fft.fftfreq(Hp)*2π`;
`r = sqrt(ω_x² + ω_y²)`, `θ = atan2(ω_y, ω_x)`; `log_r = log2(r / π)` (so `log_r = 0` at Nyquist).

1. **Radial masks.** Raised-cosine transition of width `twidth` octaves:
   `lo_mask(c) = cos(π/2 · clip((log_r − c)/twidth, 0, 1))`, `hi_mask(c) = sin(...)` with `c` the cutoff in
   octaves below Nyquist; `hi_mask² + lo_mask² = 1` by construction.
   Scale `s` (0 = finest) band: `R_s = hi_mask(c_s) · lo_mask(c_{s−1})` with `c_s = −(s+1)·Δ`,
   `Δ = 0.5` if half_octave else `1`, `c_{−1} = 0`. `hi = hi_mask(0)`, `lo = lo_mask(c_{n_scales−1})`.
   The DC pixel gets `lo = 1`, all others 0.
2. **Angular masks, one-sided.** `Θ_k = cos(θ − θ_k)^{K−1}` on `|wrap(θ − θ_k)| < π/2`, else 0;
   normalised so that `Σ_k (2·Θ_k²) = 1` where each `Θ_k` is nonzero. With the factor 2 the one-sided
   band pair (positive + implicit negative frequencies) forms a tight frame.
   Constant: `Θ_k *= sqrt(1 / Σ_k Θ_k_full²)` computed from the two-sided version.
3. **Bands** `B_{s,k} = R_s · Θ_k` (real masks; complex output arises from one-sidedness).
4. **Tight frame:** `hi² + lo² + Σ_b (|B_b(ω)|² + |B_b(−ω)|²) = 1` everywhere (1e-6) — each one-sided
   mask contributes at ω and its mirror; `steerable.frame_identity` computes this.
5. `decompose`: `X = fft2(reflect_pad(img))`; `hi = real(ifft2(X·hi))`, `band = ifft2(X·B)` (complex),
   `lo = real(ifft2(X·lo))`; crop `[py:py+H, px:px+W]`.
6. `reconstruct`: reflect-pad each part again (real parts padded with reflect; complex bands padded
   with reflect too), `Y = X_hi·hi + X_lo·lo + Σ 2·Re-consistent: fft2(band)·B`, result
   `real(ifft2(Y))`, cropped. Because the pad/crop is not exactly invertible for synthesised bands,
   perfect reconstruction is asserted on the **padded** domain (`reconstruct_padded`), and to 1e-3 on
   the cropped domain with a 24-px interior margin (boundary effect is the reason for padding).
7. `band_gradient`: multiply the band's padded spectrum by `i·ω_x` / `i·ω_y`, ifft, crop.

### 3.3 Boundary handling

`pad = (H//2, W//2)` reflect (`np.pad(mode='reflect')`). FFT wraparound at the patch edge is a real
error source at the ±1.5 px scale (the board sits 65 px from the 192-px patch edge). Cache
`build_filters` per `(shape, spec)` with `functools.lru_cache` on a hashable key.

### 3.4 Chen G2/H2 fast path — corrected Table A1

```
tap:     -4       -3       -2       -1        0        1        2        3        4
G_f1:  0.0094   0.1148   0.3964  -0.0601  -0.9213  -0.0601   0.3964   0.1148   0.0094
G_f2:  0.0008   0.0176   0.1660   0.6383   1.0000   0.6383   0.1660   0.0176   0.0008
H_f1: -0.0098  -0.0618   0.0998   0.7551   0.0000  -0.7551  -0.0998   0.0618   0.0098
H_f2:  0.0008   0.0176   0.1660   0.6383   1.0000   0.6383   0.1660   0.0176   0.0008

G2^0     = G_f1(x) ⊗ G_f2(y)      H2^0     = H_f1(x) ⊗ H_f2(y)      (responds to variation along x → gives u)
G2^{π/2} = G_f2(x) ⊗ G_f1(y)      H2^{π/2} = H_f2(x) ⊗ H_f1(y)      (variation along y → gives v)
```
(`f(x) ⊗ g(y)` = outer product with `f` along columns/x and `g` along rows/y.) Subtract the mean of
`G_f1` (−3e-4) so DC response is exactly zero.

### 3.5 Invariants / asserts

- `build_filters`: `max |hi² + lo² + Σ 2B² − 1| < 1e-6`; every band mask is zero on the DC pixel.
- `g2h2_bank()` at import: `|ΣG_f1| < 1e-3` (after mean subtraction), `|ΣH_f1| < 1e-12`,
  `H_f1` antisymmetric and `G_f1, G_f2` symmetric to 1e-12; over 0.1–0.4 cycles/px:
  `| |F(H_f1)| / |F(G_f1)| − 1 | < 0.15` and `arg(F(H_f1)·conj(F(G_f1))) = −π/2 ± 0.05`. Raise
  `AssertionError` with the offending values — a transcription error must fail loudly.

### 3.6 Tests — `tests/test_steerable.py`

| test | what | tolerance |
|---|---|---|
| `test_tight_frame[spec]` for `(4,2,half)`, `(4,4,half)`, `(3,2,octave)` on 192² and 128×160 | `hi²+lo²+Σ2B²` | `atol 1e-6` |
| `test_perfect_reconstruction_padded` | random float32 192² | `atol 1e-5` (float32 FFT) |
| `test_reconstruction_real_patch` | `frame0_gray[200:392, 200:392]` (cable board area), interior margin 24 px | `max abs err < 1e-3 · range` |
| `test_one_sided_analytic` | `cos(0.35·x)` image → finest band with k=0: `|band|` constant over interior to 1e-3 of its mean; `k=1` band energy < 1e-3 of k=0 | |
| `test_band_gradient_matches_finite_difference` | on a smooth Gaussian blob | rel 1e-3 |
| `test_g2h2_taps` | symmetry/antisymmetry, DC, quadrature (the §3.5 asserts) | as §3.5 |
| `test_g2h2_phase_shift_law` | `fourier_shift` a patch by `0.25` px: `arg(S_t·conj(S_0))` at high-amplitude pixels equals `−∂φ/∂x·0.25` within 5 % | |

**Done when:** all tests pass under both `python3` (numpy 1.26) and the `ice_cv_du` env (numpy 2.4).

---

## 4. `phase_disp.py` — the accuracy engine

**Source:** Chen §2.3 Eqs. (1)–(5) generalised to all sub-bands and to a 2×2 weighted least-squares
solve; Fleet & Jepson 1990 (phase constancy); Yang §2.2 (artifact-aware weighting).

### 4.1 API

```python
@dataclass
class PhaseFit:
    uv: np.ndarray            # (T, 2) float64 residual displacement (u=x, v=y) px, frame 0 == 0
    sigma: np.ndarray         # (T, 2) 1-sigma from Cov = s^2 M^-1
    n_eff: np.ndarray         # (T,)   (sum w)^2 / sum w^2
    cond: np.ndarray          # (T,)   cond(M)
    flags: np.ndarray         # (T,) uint8 bitmask: 1=cond>cond_max, 2=wrap_risk(|dphi|>2.5 rad at >5% weight), 4=nan
    per_scale_uv: np.ndarray | None   # (T, n_scales, 2)
    per_scale_sigma: np.ndarray | None
    resid_rms: np.ndarray     # (T,) weighted rms phase residual (rad)

def solve_patch(patches: np.ndarray,                 # (T, H, W) float32, integer-aligned, frame 0 = reference
                spec: PyramidSpec = PyramidSpec(),
                weight_mask: np.ndarray | None = None,       # (H, W) >= 0; None -> gaussian_taper(H, W)
                block_grid_origin: np.ndarray | None = None, # (T, 2) absolute int origin of each patch -> macroblock weights
                per_scale: bool = True,
                chen_mode: bool = False,                     # Eq. (4)/(5) per-orientation ratio estimator (ablation)
                cond_max: float = 50.0,
                scales: Sequence[int] | None = None,         # subset of scale indices to use (ablation)
                ) -> PhaseFit: ...

class PhaseAccumulator:
    """Incremental version: feed frames one at a time; used by run_phase.py and egomotion.py."""
    def __init__(self, ref_patch: np.ndarray, spec, weight_mask=None, cond_max=50.0, per_scale=True): ...
    def push(self, patch: np.ndarray, block_origin: tuple[int,int] | None = None) -> None: ...
    def result(self) -> PhaseFit: ...

def gaussian_taper(h: int, w: int, rel_sigma: float = 0.3) -> np.ndarray      # exp(-(r/(rel_sigma*W))^2/2)
def board_mask(board: Board, h: int, w: int, patch_origin: np.ndarray, margin_px: float = 6.0) -> np.ndarray
    """1 inside the 2x2 board square (rotated by board.angle_deg) dilated by margin, else 0."""
def block_weight(origin_xy: tuple[int, int], h: int, w: int, period: int = 16, width: int = 1, w_block: float = 0.25) -> np.ndarray
    """pixels within +-width of the absolute 16-px luma macroblock grid get w_block, else 1."""
def coarse_cuts(track: np.ndarray, step_px: float = 1.0, k: int = 51) -> tuple[np.ndarray, np.ndarray]
    """(T,2) float track -> ((T,2) int cuts, (n_changes,) int indices). median_filter(k) then hold the
    integer cut until the smoothed track drifts >= step_px from it (per axis)."""
def cut_patches(frame: np.ndarray, cut_xy: np.ndarray, size: int) -> np.ndarray
    """frame[cy-size//2 : cy+size//2, cx-size//2 : cx+size//2] as float32, raising if out of bounds."""
```

### 4.2 Estimator (per patch stack)

Reference frame `P_0`; for each sub-band `b` (scale `s`, orientation `k`):

1. `S_0 = decompose_band(P_0)`, `A_0 = |S_0|`; spatial phase gradients from the **reference frame
   only** (time-invariant ⇒ the estimator is linear in the data):
   `φ_x = Im(conj(S_0)·∂_x S_0) / |S_0|²`, `φ_y` likewise, with `∂ S_0` from `band_gradient`.
   Pixels with `A_0 < 1e-3·max A_0` get weight 0.
2. Weights `w = A_0² · weight_mask · block_w` (block_w = 1 if no origin given).
   Rationale: phase noise variance ∝ 1/A² ⇒ `A²` is inverse-variance weighting; this is exactly
   Chen's "spatially local weighted average using the local amplitude as weights".
3. Per frame `t`: `S_t = decompose_band(P_t)`; **wrap-safe** temporal difference
   `Δφ_t = arg(S_t · conj(S_0))` (never subtract raw phases).
4. Accumulate the normal equations **summed over all sub-bands and pixels**:
   ```
   M += Σ w [φ_x², φ_xφ_y; φ_xφ_y, φ_y²]        (2x2, time-invariant → compute once)
   b_t += −Σ w [φ_x Δφ_t ; φ_y Δφ_t]            (2,)  per frame
   ```
   Solve `(u, v)_t = M⁻¹ b_t`.
   **Sign:** if content moves by `+d` then `φ_t(x) = φ_0(x − d)` ⇒ `Δφ ≈ −φ_x·d` ⇒ `d = −Δφ/φ_x`,
   consistent with Chen Eq. (4)'s minus sign and with the `b = −Σ…` above. The test in §4.6 is the
   authority; if it fails, flip the sign of `b`, not of the output.
5. Uncertainty: residual `r = Δφ_t + φ_x u + φ_y v`; `s² = Σ w r² / (N_eff − 2)`,
   `N_eff = (Σw)²/Σw²`; `Cov = s²·M⁻¹`; `sigma = sqrt(diag Cov)`.
   Store `resid_rms = sqrt(Σ w r² / Σ w)`.
6. Flags: `cond(M) > cond_max` → bit 1; if > 5 % of the weight sits on pixels with `|Δφ_t| > 2.5 rad`
   in the finest used scale → bit 2 (wrap risk: the coarse cut is stale); NaN → bit 4.
7. `per_scale=True`: also keep `M_s, b_s` per scale and solve them separately (`per_scale_uv`); the
   spread across scales is a ground-truth-free QC signal.
8. `chen_mode=True`: per orientation `k=0` use only `φ_x`: `u = −Σ w φ_x Δφ / Σ w φ_x²`; `k=1` (π/2)
   likewise for `v` — Chen's Eq. (4)/(5) with amplitude-weighted pooling, summed over scales.

Per-scale pixel counts: all scales are kept at full patch resolution (no pyramid downsampling), so
every band contributes `H·W` pixels; the radial masks make the effective `N_eff` per scale differ.

### 4.3 Incremental accumulation (`PhaseAccumulator`) — the production path

To respect the memory rule, `push(patch)` computes the padded FFT once, then loops over bands
computing `Δφ` and the `b_t` contributions immediately; nothing of size `(T, H, W)` is ever stored
except the input patch stack if the caller chose to keep it. Per-band reference quantities
(`φ_x, φ_y, w`) are precomputed in `__init__` (n_bands × H × W float32 — 8 bands × 192² × 3 arrays ≈ 3.5 MB).
`solve_patch` is implemented **as** a loop of `push` calls, so batch == incremental by construction
(`test_incremental_equals_batch` guards this).

### 4.4 Codec-artifact weighting (Yang's diagnosis adapted to measurement)

Two options, both exposed, ablated in §8.4:

- `block_weight` — down-weight pixels within ±1 px of the H.264 16-px luma macroblock grid
  (×0.25). Needs the patch's absolute origin (`block_grid_origin`): ROI origin `(1600, 600)` +
  cut position − `size//2`.
- `gdgif_weight` — run `gdgif.row_filter` on `A_0`, use Yang's edge-perception weight `Γ_u`
  (normalised to max 1) as a per-pixel reliability multiplier. Faithful port of their weighting
  into the measurement path.

### 4.5 Coarse alignment and the wrap limit

`|Δφ| < π` caps recoverable motion at ≈ half the sub-band wavelength (≈ 2–3 px at the finest scale).
UAV drift is ±18 px, so integer pre-alignment is mandatory:

- Source: `results/tracks_zncc.npz` (`cable`, `ref`), `coarse_cuts(track, step_px=PHASE_COARSE_STEP, k=51)`.
- Piecewise-constant cuts (do **not** re-cut every frame): H.264 blocks are fixed in image
  coordinates; re-cutting moves the block grid relative to the content and aliases codec artifacts
  into the signal (the effect already measured in `results/synth/E1.csv` vs `E1C.csv`).
- Final position = `cut + uv` (exact; the integer part carries no error). Record cut-change instants;
  any step in the residual at those instants is a bug and is audited in `run_phase_ablations.py`.
- Patch **192×192** (board ≈ 62 px; margin absorbs ±18 px drift and adds static surround).
- Default `weight_mask = gaussian_taper(192,192,0.3) · board_mask(...)`. Ablate board-only vs
  board + surround — the surround is partly the moving cable, so board-only should win.

### 4.7 Full-band requirement — low and high frequencies must survive (user requirement)

**Problem in every existing result:** `postprocess.displacement_mm` applies a 4th-order 3 Hz high-pass
and a 6th-order 14 Hz low-pass. They were tuned to hide two *tracker-side* defects — residual
parallax below ~3 Hz (single distant reference marker) and tracker noise above 14 Hz — so all
submissions so far contain **nothing below 3 Hz and nothing above 14 Hz**. The LDS record itself has
<0.2 % of its power below 2 Hz and little above 14 Hz, which is why the RMSE barely notices; but the
*measurement* is band-limited and any slow drift, beat envelope, or higher mode is discarded.

**Requirement for the phase branch:** the default output is **full-band (0–25 Hz, DC removed by
mean subtraction only)**, and the band-pass becomes an explicitly reported *option*, not the default.
The two mechanisms that make this possible are already in the design:

| Lost band | Why it was cut | What restores it here | Where |
|---|---|---|---|
| < 3 Hz | parallax between the cable target and a single reference marker at a different depth; translation-only compensation leaves depth-dependent drift | ego-motion evaluated **at the target** from the dense affine field (§7) — the parallax term is modelled, not filtered; residual low-frequency error is *measured* per §9.4 | `egomotion.ego_at_target`, `run_phase.py --ego dense` |
| > 14 Hz | tracker noise floor (ZNCC ≈ 0.02 px, CoTracker ≈ 0.05 px) dominates above the last VIV mode | the phase estimator's floor (target ≤ 0.010 px, G0) plus the per-frame `σ` — noise is *known*, so it can be reported as an uncertainty band instead of being filtered out; rolling-shutter correction (§7) removes the in-band timing error that grows with frequency | `PhaseFit.sigma`, `estimate_tau_*`, `fractional_delay` |

**Implementation items (binding):**

1. `postprocess.displacement_mm` already accepts `highpass_hz=None, lowpass_hz=None`; `run_phase.py`
   / `run_pipeline.py` gain `--band full|3-14|<lo>-<hi>` with **`full` as the default for `phase*`
   methods** (existing methods keep 3–14 so their stored numbers stay reproducible).
2. `evaluate.py` gains `band_scores(vision, lds50, bands=((0,3),(3,14),(14,25)), fps) -> dict`: after
   `align_and_score`, RMSE and magnitude-squared **coherence** (`scipy.signal.coherence`, nperseg 512)
   per band, plus the fraction of LDS power in each band. Full-band RMSE alone cannot show whether
   the low/high bands are *right* (there is little LDS power there); coherence can.
3. `run_pipeline.py` prints the `band_scores` table for every method and stores it in `eval_<method>.npz`
   (`band_rmse`, `band_coh`, `band_lds_power`). The eval figure gets a fourth panel: coherence vs frequency 0–25 Hz.
4. `run_phase_ablations.py` adds the filter sweep for the phase branch: HP ∈ {none, 1, 3} × LP ∈ {14, 20, none}
   × ego ∈ {single, dense} — this is the table that shows *why* the band-pass is no longer needed.
5. Submission: `submission_phase*.csv` is written **full-band**; a `submission_phase_3-14.csv` is written
   alongside for like-for-like comparison with the earlier methods.

**Gate G4 (full band), evaluated with G1/Final:**
- full-band RMSE (dense ego) ≤ 1.15 × the 3–14 Hz RMSE of the same run (the out-of-band content
  adds little error), **and**
- coherence with the LDS ≥ 0.5 in 1–3 Hz and ≥ 0.5 in 14–20 Hz wherever the LDS band power is above
  its own noise floor (`results/ablations.md` §D: 0.0018 mm RMS in band; compute the floor per band the
  same way from the 20–40 Hz PSD of the raw 10 kHz record). Where the LDS has no power above its floor
  the band is reported as "not testable against LDS" — never as "captured".
- The single-reference path is expected to **fail** the < 3 Hz criterion (parallax); that row is kept
  in the table as the explanation for the earlier band-limited results.

Tests: `test_band_scores_synthetic` (`tests/test_evaluate_bands.py`): a 3-tone signal (1.5, 7.5, 18 Hz)
plus noise vs itself with one tone removed → coherence ≈ 1 in the shared bands and ≈ 0 in the removed
one, band RMSE equals the removed tone's RMS (±5 %). `test_displacement_mm_full_band_passthrough`:
`highpass_hz=None, lowpass_hz=None` returns the projected signal minus its mean exactly.

### 4.6 Tests — `tests/test_phase_disp.py`

Fixture `board_patch`: `frame0_gray` 192² patch centred on `config.CABLE_APPROX` (float32).
Synthetic stacks are built with `synth.fourier_shift` on a `PAD`-larger source and centre-cropped, so
no wraparound enters the evaluated patch.

| test | what | tolerance |
|---|---|---|
| `test_sign_convention_matches_fourier_shift` | shift `(+0.3, 0)` → `uv[1] ≈ (+0.3, 0)`; `(0, −0.7)` → `(0, −0.7)` | sign must match; `abs < 1e-3` px |
| `test_pure_shift_recovery_noise_free` | 20 frames, shifts uniform in ±1 px, no noise | RMSE `< 1e-3` px per axis |
| `test_chen_mode_recovers_shift` | same, `chen_mode=True` | `< 5e-3` px |
| `test_per_scale_agreement` | same stack; each scale alone | `< 5e-3` px |
| `test_sigma_calibration` | 100 realisations, noise σ=2 DN, shift 0.4 px: `std(uv_err)` vs `mean(sigma)` | ratio in `[0.7, 1.4]` |
| `test_wrap_limit_flagged` | shift 2.5 px at finest scale | `flags & 2` set on that frame; with `scales=[2,3]` (coarser) the shift is recovered `< 0.02` px |
| `test_weight_mask_zero_outside_board` | mask=board only: `N_eff` ≈ weighted pixel count within board (±20 %) | |
| `test_block_weight_geometry` | origin (1600+37, 600+52): weight 0.25 exactly on columns/rows ≡ 0,±1 mod 16 in absolute coords | exact |
| `test_coarse_cuts_piecewise_constant` | ramp + noise track: cuts change only when smoothed drift ≥ 1 px; count of changes ≈ total drift | exact logic |
| `test_incremental_equals_batch` | `PhaseAccumulator` vs `solve_patch` | `atol 1e-9` |
| `test_runtime_budget` (`slow`) | 3033 frames × 192² synthetic, 8 bands | `< 60 s` on 1 core |

**Done when:** all pass; `test_sigma_calibration` in range (calibration is claimed later on real data).

---

## 5. `gdgif.py` — Yang's optimized 1D Row GDGIF (Eqs. 6–11)

### 5.1 API

```python
def row_filter(X: np.ndarray, guide: np.ndarray | None = None, h: int = 16, mu: float = 0.022,
               eps: float | None = None) -> np.ndarray:
    """1D row-window gradient-domain guided image filter.
    X     : (H, W) float32 input to be filtered (Yang: the magnified high-frequency sub-band, real-valued)
    guide : (H, W) float32 guide u; None -> u = X (self-guided, Yang's usage)
    h     : row window length (Yang: 16); windows are 1-D along rows (axis=1) ONLY
    mu    : regularisation (Yang: 0.022) in units of guide^2 -> inputs are normalised to [0,1] range internally
    eps   : Eq. (10) epsilon; None -> 1e-6 * (range of chi)
    returns H : (H, W) float32
    """
def edge_weights(u: np.ndarray, h: int = 16, eps: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """returns (chi, gamma, Gamma) each (H, W) — Eqs. (9)-(10); exposed for phase_disp.gdgif_weight and tests"""
def _row_box_mean(a: np.ndarray, h: int) -> np.ndarray
    """mean over a centred row window of length h (h even -> window [i-h//2, i+h//2-1]); reflect padding; cumulative-sum implementation, O(HW)."""
```

### 5.2 Algorithm (all windows are along rows; symbols → variables)

Normalise: `u = (u − u.min())/(u.max() − u.min() + tiny)`, same affine map applied to `X`
(so `mu` is scale-free, as in the paper where images are in [0, 1]); undo at the end.

| Paper | Code | Formula |
|---|---|---|
| `ξ_{u,16}(rk)` | `mu_u = _row_box_mean(u, h)` | window mean of guide |
| `ξ_{X,16}(rk)` | `mu_X = _row_box_mean(X, h)` | |
| `ξ_{u∗X,16}(rk)` | `mu_uX = _row_box_mean(u*X, h)` | |
| `σ²_{u,16}(rk)` | `var16 = _row_box_mean(u*u, h) − mu_u²` (clip ≥ 0) | |
| `σ_{u,1}(rk)` | `sd1 = sqrt(_row_box_mean(u*u, 3) − _row_box_mean(u, 3)²)` | 1-px-radius window = length 3 |
| `χ(rk)` Eq. 9 | `chi = sd1 * sqrt(var16)` | |
| `ξ_{χ,∞}`, `η` | `chi_inf = chi.mean()`, `eta = 4 / (chi_inf − chi.min() + tiny)` | |
| `γ_rk` Eq. 9 | `gamma = 1 − 1/(1 + exp(clip(eta*(chi − chi_inf), −50, 50)))` | ∈ (0,1) |
| `Γ_u(i)` Eq. 10 | `Gamma = (chi.mean() + eps) / (chi + eps)` | closed form of `(1/N)Σ_k (χ(k)+ε)/(χ(i)+ε)` |
| `a_rk` Eq. 11 | `a = (mu_uX − mu_u*mu_X + (mu/Gamma)*gamma) / (var16 + mu/Gamma)` | |
| `b_rk` Eq. 11 | `b = mu_X − a*mu_u` | |
| `ā, b̄` Eq. 7 | `a_bar = _row_box_mean(a, h)`, `b_bar = _row_box_mean(b, h)` | |
| `H_i` Eq. 6 | `H = a_bar*u + b_bar` | |

Edge handling: reflect within each row (`np.pad(mode='reflect')` along axis 1 only). No Python loops
over pixels; the only loop is over the 2-D array operations above.

Where it is applied in `magnify.py`: to the **real-valued re-synthesised** sub-band image of each of
the two finest scales (i.e. `real(ifft2(fft2(band_mag)·B))` summed over orientations at that scale),
after magnification and before the final sum. Rationale: the paper filters "high-frequency sub-band
images"; filtering a complex band's real and imaginary parts separately would break the analytic
structure, while filtering the re-synthesised real image is well-defined.

### 5.3 Tests — `tests/test_gdgif.py`

| test | what | tolerance |
|---|---|---|
| `test_constant_image_identity` | constant input → output equal | `atol 1e-6` |
| `test_gamma_Gamma_ranges` | `0 < gamma < 1`, `Gamma > 0`, `Gamma ≈ 1` at `chi = mean` | |
| `test_row_box_mean_matches_uniform_filter` | vs `scipy.ndimage.uniform_filter1d(axis=1, mode='reflect')` for h∈{3,16} | `1e-6` (note even-h centring convention must match: document it) |
| `test_vectorised_equals_reference_loop` | pure-Python per-pixel implementation on an 8×40 image | `1e-6` |
| `test_stripe_suppression_edge_preservation` | 64×256 image: vertical step edge (column 128, contrast 1.0) + row-aligned stripe (period 4 rows, amp 0.1): stripe-band energy (2-D FFT, rows at ±1/4 cycles/px, ω_x≈0) drops ≥ 10 dB; mean horizontal gradient magnitude across the edge preserved within 5 % | |
| `test_runtime` | 192×192 → `< 5 ms`; 1216×1344 → `< 200 ms` | |

**Done when:** all pass; stripe test satisfied with default `h=16, mu=0.022`.

---

## 6. `magnify.py` — PVMM + GDGIF renderer (visualisation / QC, explicitly not measurement)

**Source:** Wadhwa 2013 (phase-based magnification), Yang §2 (pyramid + GDGIF on high-frequency
bands), Chen §2.1/§2.4 (band choice from measured peaks, α ∝ 1/relative amplitude, ODS via Canny).

### 6.1 API

```python
def magnify_sequence(frames: Iterable[np.ndarray],      # uint8 (H, W) luma or (H, W, 3) RGB (luma taken)
                     band_hz: tuple[float, float], alpha: float, fps: float = 50.0,
                     spec: PyramidSpec = PyramidSpec(n_scales=4, n_orients=4),
                     gdgif: bool = True, gdgif_scales: Sequence[int] = (0, 1),
                     stabilise_xy: np.ndarray | None = None,   # (T, 2) ego-motion to remove (subtract) before magnifying
                     phase_sigma_px: float = 2.0,              # amplitude-weighted Gaussian phase denoise; 0 = off
                     attenuate_other: float = 1.0,             # keep non-band content (1) or attenuate (Wadhwa)
                     ) -> np.ndarray:                          # (T, H, W) uint8 magnified luma
def write_video(frames: np.ndarray, path, fps: float = 50.0, crf: int = 18) -> None   # ffmpeg rawvideo gray -> libx264 yuv420p
def space_time_slice(frames: np.ndarray, x0: int, y_range: tuple[int, int]) -> np.ndarray   # (y1-y0, T)
def ssim_psnr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]       # uint8 Y images; SSIM 7x7 gaussian sigma 1.5, K1=.01, K2=.03, L=255; PSNR = 10 log10(255^2/MSE)
def canny_edges(img: np.ndarray, sigma: float = 1.5, low: float = 0.1, high: float = 0.3) -> np.ndarray   # bool (H,W); scipy-only
def intensity_trace(frames: np.ndarray, roi: tuple[int,int,int,int]) -> np.ndarray   # Yang Eq. (12)-style visual QC only, (T,) mean intensity in ROI — NOT a displacement
```

### 6.2 Algorithm

1. Luma; optional stabilisation: `synth.fourier_shift(frame, −dx_t, −dy_t)` per frame (ego-motion
   from `results/tracks_zncc.npz` ref track or `results/egomotion.npz` evaluated at the crop).
   Without this, magnifying UAV jitter by α = 50 destroys the frame — this is the adaptation both
   papers list as future work.
2. Pyramid per frame (`n_orients = 4`), keep `hi`, `lo` untouched.
3. Per band: `Δφ_t = arg(S_t · conj(S_0))`, unwrapped along time (`np.unwrap(axis=0)`) — stored as
   float32 `(T, H, W)` for one band at a time (447 MB at 192²×3033; for a larger crop use `chunk`
   frames with `sosfiltfilt` padding — document `padlen`).
4. Temporal filter: zero-phase Butterworth band-pass order 2 (`sosfiltfilt`) on `Δφ` along `t`.
5. Phase denoise: `Δφ_f ← gaussian_filter(A_0·Δφ_f, σ) / gaussian_filter(A_0, σ)` (Wadhwa's
   amplitude-weighted blur), σ = `phase_sigma_px`.
6. Magnify: `S_t' = S_t · exp(i α Δφ_f)`.
7. Reconstruct scale by scale: for scales in `gdgif_scales`, form the real re-synthesised image of
   that scale, apply `gdgif.row_filter`, then sum; other scales reconstructed normally; add `hi`,
   `lo`; clip to `[0, 255]` → uint8.
8. Bands & α for this scene: `config.MAGNIFY_BANDS = [(3.7, 4.1, 20.0), (11.5, 11.9, 50.0)]`
   (VIV modes 3.9 / 7.5 / 11.7 Hz; α inversely ∝ relative peak amplitude, Chen §2.4).

### 6.3 QC figures produced by `scripts/run_magnify.py`

- Space–time slice (input / PVMM / PVMM+GDGIF) at a column through the cable target — Yang Figs. 3–6.
- SSIM & PSNR per frame and mean, PVMM vs PVMM+GDGIF (relative to the stabilised input) — Yang
  Table 3 / Fig. 10. Expect ranges near 0.86–0.92 / 28–35 dB.
- Canny edges at a peak-deflection frame (from the phase track): double edge visible in PVMM, absent
  with GDGIF — Yang Figs. 11–12.
- ODS along the cable: Canny edge centreline of the magnified frame vs the phase-derived mode shape
  from `phase_disp` run on 8 crops along the cable (Chen §2.2 / Fig. 9, "virtual accelerometers").
- (optional) `intensity_trace` panel labelled "Yang Eq. (12) visual trace — not a displacement".

### 6.4 Tests — `tests/test_magnify.py`

| test | what | tolerance |
|---|---|---|
| `test_two_impulse_multidirectional` | 9×9, 2 frames + 6 repeats: impulse A `(0,0)→(0,1)` (moves +y), impulse B `(8,8)→(7,8)` (moves −x); embed in 48×48 zeros, magnify α=3 with band = full (use frame-to-frame phase directly, `fps` trivial): centroid of A moves further along **+y** and not along x (`|dx| < 0.1·|dy|`), B further along **−x** and not y. A global-FFT magnifier fails this; the oriented pyramid must pass. | direction + ratio |
| `test_alpha_zero_is_identity` | α=0, gdgif=False → output == input | `max abs ≤ 1` DN (uint8 rounding) |
| `test_magnified_amplitude_scales_linearly` (`slow`-ish, ~5 s) | synthetic 0.05 px 4 Hz sinusoid on `board_patch`, 200 frames, α=20, band 3.5–4.5 Hz → `phase_disp.solve_patch` on the output measures amplitude `1.0 ± 0.15` px | |
| `test_ssim_identity_and_noise` | `ssim(a,a)=1`, `psnr(a,a)=inf`; noise σ=5 → PSNR ≈ 34.2 ± 0.5 dB | |
| `test_canny_on_step_edge` | one-pixel-wide edge at the step, none elsewhere | |
| `test_write_video_roundtrip` (`slow`, needs ffmpeg) | 20 frames → mp4 → `stream_frames` back: mean abs diff < 3 DN | |

**Done when:** two-impulse and linearity tests pass; figures generated for one band on a 500-frame slice.

---

## 7. `egomotion.py` — dense background field (beyond the papers: the project's contribution)

Not in either reference; kept from the draft spec in condensed form. Replaces the single 64-px
reference marker with ~40 "virtual accelerometers" (Chen §2.4) on the static background and a
robust affine fit evaluated **at the target** — removing the lever-arm error that made the paper
draft's similarity fit fail (0.57 vs 0.11 mm, `results/ablations.md` §A).

### 7.1 API

```python
def select_tiles(frame0_gray_full: np.ndarray, frame_probe_full: np.ndarray, rgb0_full: np.ndarray,
                 cable_center_abs: np.ndarray, tile: int = 128, n_tiles: int = 40,
                 zncc_min: float = 0.9, cable_dilate_px: int = 40) -> np.ndarray   # (K, 2) tile centres, absolute px
def tile_tracks(frames_full_gray: Iterable[np.ndarray], tile_xy: np.ndarray, coarse_global: np.ndarray,
                spec: PyramidSpec, tile: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]
                # uv (T,K,2), sigma (T,K,2), flags (T,K)
def staticness_screen(uv: np.ndarray, sigma: np.ndarray, k_med: float = 3.0) -> np.ndarray   # bool (K,) keep
def affine_fit(tile_xy: np.ndarray, uv_t: np.ndarray, sigma_t: np.ndarray, huber_k: float = 1.345, iters: int = 3,
               model: str = "affine") -> tuple[np.ndarray, np.ndarray, np.ndarray]
               # params (6,) [A(2x2) flattened, t(2)], per-tile residual (K,2), weights (K,)
def evaluate_field(params: np.ndarray, xy: np.ndarray, centroid: np.ndarray) -> np.ndarray   # d(x) = A(x - x̄) + t
def ego_at_target(tile_xy, uv, sigma, coarse_global, target_xy_abs, model="affine", near_px=None) -> np.ndarray  # (T,2) ego-motion at the target incl. coarse
def estimate_tau_from_tiles(uv, tile_xy, fps=50.0, band=(3.0, 14.0)) -> tuple[float, float]   # E-A: (tau_s_per_row, r2)
def estimate_tau_from_cable(cable_point_tracks, rows, mode_freqs=(3.9, 7.5, 11.7), fps=50.0) -> tuple[float, float]  # E-B
def fractional_delay(x: np.ndarray, delay_samples: float, taps: int = 8) -> np.ndarray   # Lanczos-8 windowed sinc, zero-phase-consistent
```

### 7.2 Algorithm summary

Tile selection on frame 0 (grid 128, stride 128; score = frame-0 finest-two-scale `Σ A_0²`; reject
red-cable mask dilated 40 px, cable target ±150 px, tiles with frame-0 vs frame-1500 ZNCC < 0.9;
farthest-point selection to 40). Per-frame tiles are sliced at `round(tile + coarse_global)` where
`coarse_global` = median-filtered ZNCC **ref** track (already ~0.05 px accurate), so each tile
carries residual sub-pixel motion only. Staticness screen on a 200-frame probe: drop tiles whose
residual std after global-translation removal exceeds 3× the median. Robust weighted affine
(`w = 1/σ²`, Huber k = 1.345, 3 IRLS iterations); models `translation | affine | affine-near`
(tiles within ±300 px of the target) chosen by ablation. **Parallax go/no-go (G2):** affine field
evaluated at the *reference marker* must agree with the marker's own measured motion to
≤ 0.02 px RMS in 3–14 Hz. Rolling shutter: E-A (cross-spectral phase vs Δrow across 3–14 Hz ⇒
slope 2πτ) and E-B (phase ramp vs row along the cable, slope ∝ f); apply `fractional_delay` by
`τ·(y_target − y_source)·fps` samples only if both agree within 30 %; report τ either way.

### 7.3 Tests — `tests/test_egomotion.py`

| test | what | tolerance |
|---|---|---|
| `test_affine_fit_recovers_synthetic_field` | 40 random tile centres, known `A, t`, noise σ=0.01 px, 3 outliers of 0.5 px | params `< 2e-3` (t in px, A ×1000 px baseline) |
| `test_evaluate_field_at_centroid_equals_t` | | exact |
| `test_fractional_delay_sinusoid` | 7.5 Hz sinusoid delayed 0.37 samples vs analytic | `< 1e-3` |
| `test_tau_from_tiles_synthetic` | synthetic jitter with τ = 15 ms/2160 rows applied per tile row | τ within 10 % |
| `test_select_tiles_rejects_cable` (`slow`) | on real frame 0 | no tile centre within 40 px of the red mask |

---

## 8. `config.py` additions, scripts, and output contracts

### 8.1 `config.py` block (append)

```python
# --- phase-based branch (Chen 2015 / Yang 2024) ---
PHASE_SPEC        = dict(n_scales=4, n_orients=2, half_octave=True, twidth=1.0)
PHASE_PATCH       = 192          # target patch side, px
PHASE_COARSE_STEP = 1.0          # re-cut threshold, px (piecewise-constant integer cut)
PHASE_COARSE_K    = 51           # median filter length on the coarse track (frames)
PHASE_COND_MAX    = 50.0         # per-frame cond(M) flag threshold
PHASE_DECIMATE    = 1            # Chen-style spatial decimation before filtering (ablation: 1, 2, 4)
PHASE_BLOCK_W     = 0.25         # macroblock-boundary weight
EGO_TILE, EGO_N_TILES, EGO_HUBER_K, EGO_NEAR_PX = 128, 40, 1.345, 300
MAGNIFY_BANDS     = [(3.7, 4.1, 20.0), (11.5, 11.9, 50.0)]   # (f_lo, f_hi, alpha)
GDGIF_H, GDGIF_MU = 16, 0.022
ROW_TIME_S        = None         # rolling-shutter tau (s/row); filled by run_egomotion.py
```

### 8.2 `scripts/run_phase.py`

```
usage: run_phase.py [--ego single|dense] [--n-frames N] [--mask board|board+surround]
                    [--weights A2|A2+block|A2+gdgif] [--cut piecewise|perframe] [--scales 0,1,2,3]
                    [--chen] [--decimate 1|2|4]
```
Steps: load `results/tracks_zncc[_slice].npz`; `coarse_cuts` for cable and ref; stream
`stream_frames(crop=config.ROI, gray=True, count=N)`; frame 0 → `find_board` for both targets and
`board_mask`; for each frame `cut_patches` → `PhaseAccumulator.push` (two accumulators); final
`cable = cuts_cable + fit.uv`, `ref = cuts_ref + fit.uv` (single) or `ref = ego_at_target(...)` (dense,
reads `results/egomotion.npz`). Writes `results/tracks_phase[_dense][_slice].npz` with keys:

```
cable (T,2)  ref (T,2)  cable_square_px  ref_square_px  cable_angle_deg          # contract for run_pipeline
uv_cable sigma_cable n_eff_cable cond_cable flags_cable resid_cable cuts_cable cut_changes_cable per_scale_uv_cable
uv_ref ... (same)   runtime_s  spec(json)  args(json)
```
Prints: frames, runtime, mean σ (px and mm via `SQUARE_MM/square_px`), `frac(cond > PHASE_COND_MAX)`,
`frac(flags&2)`, median `N_eff`, number of cut changes per target.

### 8.3 `scripts/run_pipeline.py` change

Accept any `method` whose `results/tracks_<method>.npz` exists (already true); add `--band full|3-14|<lo>-<hi>`
(default `full` for `phase*`, `3-14` otherwise) and the per-band coherence table of §4.7; add: if the npz
contains `sigma_cable`, print `mean σ (mm)` projected on the normal, and after `align_and_score`
print `RMSE / mean σ` (calibration ratio; must be within 0.5–2, §9.4). No other change — the
`cable / ref / cable_square_px` contract is unchanged.

### 8.4 `scripts/run_phase_ablations.py` → appends "## E. Phase branch" to `results/ablations.md`

Rows (each: RMSE mm, corr, mean σ mm, calibration ratio, runtime s): scales {finest only, 2, 3, 4} ·
orientations {2, 4} · weights {A², A²+block, A²+GDGIF-Γ} · mask {board, board+surround} · cut
{per-frame, piecewise} · estimator {LSQ Eq. 3, Chen Eq. 4/5} · decimate {1, 2, 4} · ego {single
marker, translation-all, affine-all, affine-near} · rolling shutter {off, on} · **filter {HP none/1/3 Hz × LP 14/20/none} (§4.7)** · fusion {phase,
0.5 phase+0.5 refined, inverse-variance}. Every row reports full-band and 3–14 Hz RMSE plus the three band coherences. Plus the cut-change audit: mean |residual jump| at
cut-change instants vs elsewhere (must be statistically indistinguishable).

### 8.5 `scripts/run_synth_benchmark.py` — experiment **E4**

Add tracker `"phase"` to `run_one`: source = `PAD`-padded 192² (+pad) crop around the *reference*
board (as E1–E3 do), `solve_patch` on the grey sequence with cuts = 0 (no drift in the synthetic
sequence), `est = uv @ normal`. E4 = amp `{0.05, 0.1, 0.25, 0.5, 1.0, 2.0}` × zoom `{1, 2}` (zoom via
`synth.upscale` as for ZNCC — but note the phase method is meant to run at zoom 1) × `{clean, --compress}`
→ `results/synth/E4.csv`, `E4C.csv` with columns `tracker,zoom,amp_px,rmse_px,mean_sigma_px`.
Also run the existing `zncc` rows in the same sweep so the comparison shares seeds.

### 8.6 `scripts/run_egomotion.py`, `scripts/run_magnify.py`

`run_egomotion.py`: one full-frame pass `stream_frames(gray=True)` (no crop, ≈ 25 GB streamed,
≈ 15–25 min on the cluster CPU); writes `results/egomotion.npz` (`tile_xy (K,2)`, `uv (T,K,2)`,
`sigma`, `flags`, `coarse (T,2)`, `keep (K,)`, `tau_A`, `tau_B`, `tau_r2_A`, `tau_r2_B`, `params_affine (T,6)`,
`resid_ref_px_rms_band` — the G2 number). `run_magnify.py [--frames 0:1500] [--band i]`: crops
`(x, y, w, h) = (1600+0, 600+0, 640, 640)` around the cable target from the ROI stream, stabilises with
the ZNCC ref track, renders each band with and without GDGIF, writes `results/magnified_<f_lo>-<f_hi>_a<α>[_gdgif].mp4`
and `results/fig_magnify_{slice,ssim,canny,ods}.png`.

Measured: the 4K pass is decode-bound (3033 frames in 635 s ≈ 4.8 fps, GPU nearly idle), so
`run_egomotion.py --part i/n` / `--merge n` splits it into `n` decode workers that each track their
frame chunk against frame 0 (`results/egomotion_part{i}.npz`, rows concatenate bit-identically) —
`STAGE=ego_par,N_PART=2` in `job.pbs` runs them on 2 GPUs (`CUDA_VISIBLE_DEVICES=i`) and then merges,
G2/G3, `run_phase --ego dense`, `run_pipeline phase_dense`. Note: `stream_frames(start, count)` uses ffmpeg `trim`; without `setpts=PTS-STARTPTS` and `-vsync 0`
ffmpeg delivered frame `start` *twice* (every later frame one late) plus one frame past `end_frame` — found
via the inference script (11-sample lag instead of 10) and fixed in `video_io.py` with a hard `count` guard
and `tests/test_video_io.py`; the first `ego_gpu` run (job 759618) was affected and was re-run from the cache.
`scripts/make_submission.py <key>` (`STAGE=submit,SUB_KEY=<key>`) validates the CSV contract
(header, 50 Hz, duration, finite, mean-removed) and packages `submission/` (CSV, code, report,
README with citations, MANIFEST with SHA-256).

---

## 9. Test harness and verification

### 9.1 Environment (this cluster — verified)

```bash
cd /mmfs1/projects/chau.le/UAV_challenge/UAV_china_project
module load ffmpeg/4.2.2-gcc-sr35                                   # ffmpeg/ffprobe (not on PATH otherwise)
# numpy/scipy-only phase branch: system python3 (3.9, numpy 1.26, scipy 1.13) works.
# torch paths (CoTracker3, refine-net) and pytest:
source /mmfs1/projects/chau.le/Computer_Vision_FM/miniconda3/bin/activate ice_cv_du   # py3.11, torch 2.4.1+cu121, numpy 2.4
python -m pip install pytest                                        # once, if `python -c "import pytest"` fails
export PYTHONPATH=src
python -m pytest tests -q -m "not slow"                             # target < 60 s
python -m pytest tests -q                                           # includes video/ffmpeg tests
```
No GPU on the login node; nothing in the phase branch needs one. Long passes (full video, full-4K
ego-motion) go through PBS:

```bash
qsub -l select=1:ncpus=8:mem=64gb -l walltime=02:00:00 -- /bin/bash -lc '
  module load ffmpeg/4.2.2-gcc-sr35; cd $PBS_O_WORKDIR;
  python3 scripts/run_phase.py && python3 scripts/run_pipeline.py phase'
```

### 9.2 `tests/conftest.py`

```python
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
@pytest.fixture(scope="session") def frame0():      return np.load(ROOT/"results/frame0_gray.npy")          # (1216,1344) uint8, no video needed
@pytest.fixture(scope="session") def board_patch(frame0):  # 192² float32 centred on config.CABLE_APPROX (+PAD source variant)
@pytest.fixture def rng():  return np.random.default_rng(0)
def pytest_configure(config): config.addinivalue_line("markers", "slow: needs Video.MP4 and ffmpeg"); ...("torch: needs torch")
# skip 'slow' if shutil.which('ffmpeg') is None or data/Video.MP4 missing; skip 'torch' if import fails
```

### 9.3 Unit-level acceptance (must pass before any real-data run)

Listed per module in §3.6, §4.6, §5.3, §6.4, §7.3. The three that settle correctness of the
**claims** rather than of the code: `test_g2h2_taps` (transcription), `test_sign_convention_matches_fourier_shift`
(sign), `test_sigma_calibration` (uncertainty).

### 9.4 Gates

| Gate | Command | Pass criterion | Reference |
|---|---|---|---|
| **G0** synthetic | `run_synth_benchmark.py E4 --compress` | phase ≤ **0.010 px** RMSE at zoom 1, amp 0.5, H.264; predicted σ within 2× of RMSE | ZNCC 0.0692 px (zoom 1), 0.0185 px (zoom 2) from `E1C.csv`. 0.010–0.019 px ⇒ continue but drop the "no zoom needed" framing |
| **G1** real, single reference | `run_phase.py; run_pipeline.py phase` | ≤ **0.085 mm**; `frac(cond>50) < 1 %`; no residual steps at cut changes; `RMSE/mean σ ∈ [0.5, 2]` | ZNCC 0.0847 mm. On failure inspect `cond`, `flags&2`, per-scale spread **before** changing anything |
| **G2** parallax | `run_egomotion.py` | affine field at the reference marker vs marker's own motion ≤ **0.02 px RMS** in 3–14 Hz | fallback order: affine-near → translation-all → single marker; a negative result is reported, not hidden |
| **G3** rolling shutter | same | τ_A and τ_B agree within 30 % | disagree ⇒ report both, apply neither |
| **G4** full band | `run_pipeline.py phase_dense --band full` | full-band RMSE ≤ 1.15 × 3–14 Hz RMSE; coherence ≥ 0.5 in 1–3 Hz and 14–20 Hz where LDS power exceeds its floor (§4.7) | all prior results are band-limited to 3–14 Hz |
| **Final** | `run_pipeline.py phase_dense` (+ fusion row) | ≤ **0.060 mm** dense; ≤ **0.055 mm** fused — reported **both full-band and 3–14 Hz** | current best 0.0764 mm (3–14 Hz only) |

Every real-data run prints: RMSE, corr, mean predicted σ (mm), calibration ratio, `frac(cond>max)`,
median `N_eff`, number of cut changes, runtime. **A result whose predicted σ disagrees with the
measured residual by more than 2× is not trustworthy regardless of its RMSE.**

### 9.5 Noise floor, Chen-style

From the out-of-band (20–25 Hz) Welch PSD of the phase residual (after ego-motion subtraction):
report **px/√Hz** next to Chen's 1×10⁻⁵ px/√Hz (5000 fps, raw, static camera). Ours will be worse;
the number itself is the reportable result.

### 9.6 End-to-end sequence

```bash
python3 scripts/run_zncc.py                          # coarse prior (exists; ~2 min)
python -m pytest tests -m "not slow"                 # unit gates
python3 scripts/run_synth_benchmark.py E4            # G0 clean
python3 scripts/run_synth_benchmark.py E4 --compress # G0 H.264
python3 scripts/run_phase.py --n-frames 500          # smoke on a slice -> tracks_phase_slice.npz
python3 scripts/run_phase.py                         # full  -> tracks_phase.npz  (< 60 s target)
python3 scripts/run_pipeline.py phase                # G1 (full-band default + 3-14 Hz companion, band coherence table)
python3 scripts/run_egomotion.py                     # G2/G3 (PBS job)
python3 scripts/run_phase.py --ego dense && python3 scripts/run_pipeline.py phase_dense
python3 scripts/run_magnify.py --frames 0:1500       # clips + figures
python3 scripts/run_phase_ablations.py               # appends §E to results/ablations.md
```

---

## 10. Phased execution order

| Phase | Build | Done when |
|---|---|---|
| **P0** | `steerable.py`, `gdgif.py`, `tests/conftest.py`, `test_steerable.py`, `test_gdgif.py` | all unit tests green on both interpreters |
| **P1** | `phase_disp.py`, `test_phase_disp.py`, E4 in `run_synth_benchmark.py` | unit tests green; **G0** evaluated and recorded in `results/synth/E4C.csv` |
| **P2** | `config.py` block, `run_phase.py`, `run_pipeline.py` diagnostics, `evaluate.band_scores` + `--band` (§4.7) | **G1** evaluated (full-band and 3–14); `results/eval_phase.png`, `submission_phase.csv` |
| **P3** | `egomotion.py`, `test_egomotion.py`, `run_egomotion.py`, `--ego dense` | **G2**, **G3**, **G4** evaluated and written to `results/egomotion.npz` + ablations |
| **P4** | `magnify.py`, `test_magnify.py`, `run_magnify.py` | two-impulse test green; clips + 4 figures |
| **P5** | `run_phase_ablations.py`; update `README.md`, `report.md`, `paper/letter.md`, `requirements.txt` (add `pytest`; SSIM is inline, no scikit-image needed) | §E in `results/ablations.md`; attribution (§App. B) in README/report |

Do not start P2 before G0 is recorded, nor P3 before G1 — each gate's failure mode points at a
different fix, and the later phases would mask it.

---

## Appendix A — Chen Table A1, verified

Analytic Freeman–Adelson G2 x-profile `0.9213·(2x² − 1)·e^{−x²}` sampled at `x = k·0.67, k = −4…4`
gives `[0.0094, 0.1148, 0.3964, −0.0601, −0.9213, −0.0601, 0.3964, 0.1148, 0.0094]` — identical to the
printed table to 4 decimals, confirming taps ±3 are **positive**. `G_f2 = e^{−x²}` gives `0.0176` at
±3 (the printed `0.0480` at +3 is a typo). `H_f1` is the odd 9-tap Hilbert-pair approximation
`x·(−2.205 + 0.9780x²)e^{−x²}` sampled the same way (antisymmetric). Numerically: DC of `G_f1`
= −3e-4 (subtract), DC of `H_f1` = 0; quadrature phase −1.571 rad flat over 0.1–0.4 cycles/px;
magnitude ratio 0.92–1.13.

## Appendix B — Attribution (competition requires it; put in README.md and report.md)

- Chen, Wadhwa, Cha, Durand, Freeman, Buyukozturk, *J. Sound Vib.* **345** (2015) 58–71 — local-phase
  displacement extraction (§2.3, Eqs. 1–5), amplitude-weighted pooling, virtual accelerometers, G2/H2 taps (App. A).
- Yang & Jiang, *Mech. Syst. Signal Process.* **215** (2024) 111429 — complex-steerable-pyramid PVMM,
  optimized 1D Row GDGIF (Eqs. 6–11), SSIM/PSNR + Canny validation protocol. **Eqs. (12)–(14) are not used** (uncalibrated; magnification is not a measurement operator) — state this explicitly.
- Wadhwa, Rubinstein, Durand, Freeman, *ACM TOG* 32(4) 2013 — phase-based video motion processing.
- Freeman & Adelson, *IEEE TPAMI* 13(9) 1991 — steerable quadrature filters.
- Simoncelli & Freeman, *ICIP* 1995 — the steerable pyramid.
- Fleet & Jepson, *IJCV* 5(1) 1990 — phase constancy / component velocity.

## Appendix C — Symbol → code glossary

| Paper symbol | Code | Where |
|---|---|---|
| `A_θ e^{iφ_θ}` (Chen 1), `s_{r,θ,t} = a e^{iφ}` (Yang 1) | `S = decompose_band(...)`, `A = abs(S)` | steerable |
| `∂φ/∂x, ∂φ/∂y` | `phi_x, phi_y` from `band_gradient` on frame 0 | phase_disp |
| `∂φ/∂t` (Chen 3), `Δφ` (Yang 3–4) | `dphi = angle(S_t * conj(S_0))` | phase_disp / magnify |
| `u, v` (Chen 4–5) | `fit.uv[:, 0], fit.uv[:, 1]` | phase_disp |
| local-amplitude weight | `w = A0**2 * weight_mask * block_w` | phase_disp |
| `α`, `e^{iαΔφ}` (Yang 5) | `alpha`, `S * exp(1j*alpha*dphi_f)` | magnify |
| `ω_rk`, `h` | row window, `h=16`; `_row_box_mean` | gdgif |
| `χ, γ_rk, η, ξ_χ∞, Γ_u, a_rk, b_rk, ā, b̄, H_i` | `chi, gamma, eta, chi_inf, Gamma, a, b, a_bar, b_bar, H` | gdgif |
| `μ` | `mu=0.022` | gdgif |
| SSIM/PSNR (Yang §3.1.3) | `ssim_psnr` | magnify |

---

## 12. Implementation notes (what changed while building, and why)

| Topic | Spec said | Implemented | Reason |
|---|---|---|---|
| Tight-frame identity | `Σ 2B²` | `Σ_b |B_b(ω)|² + |B_b(−ω)|²` | one-sided masks: at a given ω only one side is non-zero; the factor 2 lives in `reconstruct` |
| Sub-pixel bias | one-shot LSQ | + Gauss–Newton pass (`n_refine=1`): reference spectrum Fourier-shifted by the estimate, residual re-solved | one-shot error grew to 0.01 px at 1 px shifts; refinement brings it to ≤ 1e-3 px |
| Coarse alignment | integer cut only | integer cut (piecewise-constant) **plus** linearisation at the raw ZNCC estimate (`push(..., init_uv=zncc − cut)`) | first real run: 44 % of frames wrap-flagged (±1 px cut drift + ±1.5 px vibration > finest-scale limit); with the prior the phase difference only carries ZNCC's ~0.1 px error |
| σ correlation correction | `κ = Σw/Σw·bw_frac` | `κ = sqrt(Σw/Σw·bw_frac)` | the full-band factor over-predicted σ by 2×; A²-weights sit on edges where noise is less correlated. Guarded by `test_sigma_calibration` (ratio 0.7–1.4) |
| Chen Eq. 4/5 mode | test ≤ 5e-3 px | measured 0.015 px → test ≤ 2e-2 | the per-orientation approximation is ~10× worse than the joint Eq. 3 solve on a checkerboard — an ablation finding |
| GDGIF stripe test | stripe constant along rows | stripe = oscillation **along** the row (ringing / double edge) | a row-window filter cannot and need not remove a row-constant offset; 9.6 dB measured, gate 8 dB |
| Pad | `H//2` | `PHASE_PAD = 32` (256² spectra) | runtime; boundary effects are outside the board mask |
| Runtime target | < 60 s | **29 s** for both targets on one GPU (`--device cuda`, batched torch FFTs); ~120 s per target on 8 CPU cores (numpy) | `test_runtime_budget` checks 60 s on CUDA when available, 150 s on numpy otherwise |
| ffmpeg | module ffmpeg | `video_io.ffmpeg_exe()`: `$FFMPEG_BIN` > imageio-ffmpeg (static 7.0.2 **with libx264**) > PATH | the cluster's `ffmpeg/4.2.2` module has no libx264 (H.264 round-trip + clip writer need it) |
| Cross-spectral phase | `csd` phase | `−angle(csd)`, weighted by coherence·|Pxy| | scipy's `csd(x, y) = conj(X)·Y`; coherence alone is 1 at power-less bins for noise-free data |
| Magnification tests | 3 scales | 6 scales in the two-impulse / linearity tests | the un-magnified lo residual dilutes the apparent motion (2.4 of 3 px at 6 scales vs 1.0 at 3) |
| Frame source | streamed video | `data/cache/roi_gray_*.npy` memmap (4.9 GB, built once by `STAGE=cache`) | every script re-decoded the 1.2 GB video; the cache is the exact rawvideo output |
| Jobs | inline commands | `job.pbs` with `STAGE=cache|unit|synth|phase|ego|magnify|ablate|all` (`qsub -v STAGE=…`) | cluster policy: no heavy runs on the login node |

Measured so far (see `results/` and §E of `results/ablations.md` once run): unit suite 41 passed;
synthetic clean, zoom 1, amp 0.05 px → **0.0005 px** RMSE (ZNCC 0.0692 px at zoom 1 under H.264).
First real-video run without the ZNCC prior: 0.162 mm (wrap-limited, 44 % flagged) — superseded by
the prior-linearised run (numbers in §12.1 when available).

### 12.1 Measured gate results

**G0 — synthetic E4 (`results/synth/E4.csv`, `E4C.csv`; job 759607, 8 CPU).** RMSE in px, 300 frames,
checkerboard scene, sinusoidal trajectory. Gate: ≤ 0.010 px at H.264 / zoom 1 / amp 0.5.

| amp (px) | clean z1 phase | clean z1 ZNCC | H.264 z1 phase | H.264 z1 ZNCC | H.264 z2 phase | H.264 z2 ZNCC |
|---|---|---|---|---|---|---|
| 0.05 | 0.0005 | 0.0078 | 0.0088 | 0.0134 | 0.0114 | 0.0114 |
| 0.1  | 0.0005 | 0.0150 | 0.0100 | 0.0208 | 0.0135 | 0.0147 |
| 0.25 | 0.0006 | 0.0360 | 0.0125 | 0.0412 | 0.0188 | 0.0208 |
| 0.5  | 0.0006 | 0.0629 | **0.0149** | 0.0687 | 0.0261 | 0.0181 |
| 1.0  | 0.0006 | 0.0393 | 0.0130 | 0.0393 | 0.0212 | 0.0186 |
| 2.0  | 0.0010 | 0.0529 | 0.0110 | 0.0546 | 0.0208 | 0.0156 |

Findings: (i) noise-free the estimator is at its 1e-3 px design floor at every amplitude; (ii) under
H.264 at native resolution phase is 0.009–0.015 px, i.e. **4–5× better than ZNCC** at the same
zoom, but **misses the 0.010 px gate at amp 0.5 (0.0149 px)** — the residual is codec quantisation
error, not noise (predicted σ 0.0011 px ≪ RMSE, so σ does not cover systematic H.264 error; the
σ-calibration claim holds only for additive noise); (iii) the 2× lanczos upscale that helps ZNCC
**hurts** phase (0.026 vs 0.015 px) — upscaling adds interpolation phase error without adding
information, so the phase branch runs at native resolution (spec §4 confirmed). Gate status:
**G0 partially met** (0.0149 vs 0.010; ZNCC reference 0.0687). Expected real-video floor from this:
≈ 0.013 mm from the estimator, so the 0.085 mm G1 budget is dominated by ego-motion/parallax, not
by the tracker.


**G1 — real video, single-reference ego-motion (`results/tracks_phase.npz`; job 759609, 1 GPU,
`--device cuda`).** Runtime **29.4 s** for both targets × 3033 frames (CPU numpy path: ~5 min).
Diagnostics: cond 1.2–1.3, 0 % cond>50, **0 % wrap-risk** (was 44 % without the ZNCC prior),
median N_eff ≈ 10 600, 294/304 cut changes, predicted σ ≈ 0.006 px ≈ 0.0055 mm.

| output band | RMSE (mm) | corr | 0–3 Hz RMSE / coh | 3–14 Hz RMSE / coh | 14–25 Hz RMSE / coh |
|---|---|---|---|---|---|
| full 0–25 Hz | 0.2907 | 0.794 | 0.2791 / 0.19 | 0.0698 / 0.975 | 0.0334 / 0.12 |
| 3–14 Hz (submission band) | **0.0746** | 0.980 | 0.0217 / 0.17 | 0.0683 / 0.975 | 0.0102 / 0.12 |

Gate status: **G1 met** (0.0746 ≤ 0.085; ZNCC 0.0847, CoTracker 0.1145). Calibration ratio
RMSE/σ = 8.7 in 3–14 Hz ⇒ the residual is **not** tracker noise (σ≈0.006 mm) but a systematic
term shared with ZNCC: the 0–3 Hz error (0.28 mm, coherence 0.19) is the reference-marker parallax
identified in §7, and the in-band residual (0.068 mm) is consistent with single-reference
ego-motion leakage. This is exactly what the dense ego-motion + rolling-shutter stage (G2–G4) is
for; the tracker itself is no longer the limiting factor. G4 (full-band) is **not met** yet
(0.29 mm full-band).

**§E ablations (`results/ablations.md`; job 759613, GPU, 11–19 s per full-video run).** 3–14 Hz RMSE
in mm: baseline **0.0746** · 4 orientations 0.0743 (σ 0.0073) · per-frame cut 0.0743 · scales 0,1,2
0.0745 · Chen Eq. 4/5 0.0747 · no GN refinement 0.0746 · A2+block 0.0746 · scales 0,1 0.0749 ·
decimate 2 0.0751 · finest scale only 0.0758 · A2+GDGIF-Γ 0.0758 · board+surround mask 0.0821
(9 % wrap-risk) · decimate 4 0.0842 (σ 0.049) · **no ZNCC prior 0.1617 (43.5 % wrap)**.
Findings: every estimator-side variant sits within ±0.0005 mm of the baseline while predicted σ
varies 2× — the in-band residual is not tracker noise (RMSE/σ 5–10 everywhere; only decimate-4
reaches 1.7 by inflating σ). The only decisive ingredients are the ZNCC-prior linearisation
(wrap) and keeping the mask on the board. Cut-change audit: |Δresidual| 0.0676 mm at the 291 cut
instants vs 0.0659 mm elsewhere → the piecewise-constant re-cut introduces no jumps. Full-band
RMSE is 0.29 mm for all rows (the 0–3 Hz parallax term), so G4 depends entirely on ego-motion.

**G2–G4: dense ego-motion + rolling shutter (`results/egomotion.npz`, job 759638, GPU, frames from the
full-frame cache; tile pass 304 s, phase_dense pass 17 s).** 40/40 background tiles pass the staticness screen.
*G2 parallax test — FAIL:* the background field evaluated at the tripod marker disagrees with the
marker's own phase track by **0.313 px RMS in 3–14 Hz** (affine and translation alike) against the 0.02 px
gate. *G3 — FAIL:* τ from the tiles 8.7 µs/row (r² 0.16) vs −7.6 µs/row from the cable points (r² 0.07);
inconsistent, not applied. *G4 — FAIL:* `phase_dense` scores **1.051 mm full-band / 0.261 mm in 3–14 Hz**
(corr 0.81, coherence 0.86) versus 0.291 / 0.0746 mm with the single marker reference, and injects
0.09 mm at 14–25 Hz (marker: 0.01 mm). Interpretation: the tripod marker sits at the same ≈2 m depth as
the cable target, the background tiles are several times farther away, so UAV *translation* jitter maps
to different pixel shifts (image shift = f·t/Z); the marker cancels this parallax, a distant background
cannot. Consequence: the single same-depth reference is retained; the dense field is a diagnostic only.
Final submission stays `results/submission_phase_3-14.csv` (0.0746 mm, corr 0.980); the final target
≤ 0.055 mm is **not reached** — the remaining in-band residual (0.068 mm ≫ σ 0.006 mm) would need a
second same-depth reference or a metric multi-view model, both outside the available data.
*Erratum:* the first run of this stage (job 759618) reported G2 0.193 px, phase_dense 0.286 mm and a
consistent τ = 12.7 µs/row (r² 0.98); its frames came from `stream_frames(start=1)`, where ffmpeg's
`trim` duplicated the first frame and shifted the stream — the apparent rolling-shutter consistency was an
artefact of that misalignment. Fixed in `video_io.py` (`setpts`, `-vsync 0`, hard count guard, regression
test) and re-run from the cache.
