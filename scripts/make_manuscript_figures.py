"""Manuscript figures (journal style, one message per figure) -> figures/fig<N>_<name>.{pdf,png}.

Usage: python scripts/make_manuscript_figures.py            (run as a PBS job: STAGE=figs)
Reads results/*.npz, results/synth/E4*.csv, results/ablations.md, results/egomotion.npz,
data/cache/full_gray.npy (frame 0), results/magnified_*.mp4; writes figures/ and figures/README.md.

Storyline
  1 scene            the measurement problem: two same-depth targets, in-plane normal direction
  2 phase principle  local phase is linear in sub-pixel displacement
  3 synthetic        codec sets the floor; phase 5x below ZNCC at native resolution
  4 time history     0.075 mm agreement with the LDS over 60 s
  5 spectra          all cable motion lies in 3-14 Hz; sub-3 Hz vision content is parallax
  6 trackers         local phase is the best of five trackers on this scene
  7 ego-motion       a far background carries depth parallax; the same-depth marker cancels it
  8 magnification    the 11.7 Hz mode is a rigid in-plane oscillation of the board (visual QC)
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

from uav_disp import config, egomotion, steerable, synth  # noqa: E402
from uav_disp.video_io import FULL_CACHE, stream_frames  # noqa: E402

# ---------------------------------------------------------------------------
# style: colour-blind-safe (Okabe-Ito), single column 3.5 in, double 7.2 in, 300 dpi
C = dict(lds="#000000", phase="#0072B2", zncc="#D55E00", cot="#E69F00", fused="#CC79A7",
         refined="#009E73", dense="#56B4E9", grey="#8C8C8C", light="#DDDDDD", band="#F0E442")
SINGLE, DOUBLE = 3.5, 7.2
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "lines.linewidth": 0.9, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.dpi": 300, "pdf.fonttype": 42,
})
FPS = config.FPS
MM_PER_PX = config.SQUARE_MM / float(np.load(ROOT / "results/tracks_phase.npz")["cable_square_px"])
CAPTIONS: list[tuple[str, str, str]] = []


def label(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, f"({s})", transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom", ha="left")


def save(fig, name, message, caption):
    for ext in ("png",):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    CAPTIONS.append((name, message, caption))
    print("wrote", name, flush=True)


def bandpass(x, lo=3.0, hi=14.0, fs=FPS, order=4):
    sos = signal.butter(order, [lo, hi], "bp", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x - np.mean(x, axis=0), axis=0)


# ---------------------------------------------------------------------------
# data

def load_scene():
    zn = np.load(ROOT / "results/tracks_zncc.npz")
    roi = np.array(config.ROI[:2])
    full0 = np.load(FULL_CACHE, mmap_mode="r")[0] if FULL_CACHE.exists() else \
        (next(stream_frames(count=1)) @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
    ev = np.load(ROOT / "results/eval_phase_3-14.npz")
    ang = float(ev["angle_rad"])
    return dict(zn=zn, roi=roi, full0=np.asarray(full0), cable=zn["cable"][0] + roi, ref=zn["ref"][0] + roi,
                axis=np.array([np.cos(ang), np.sin(ang)]), normal=np.array([np.sin(ang), -np.cos(ang)]),
                roi_gray0=np.load(ROOT / "results/frame0_gray.npy"))


# ---------------------------------------------------------------------------
# Fig 1 — scene geometry

def fig1_scene(S):
    ego = np.load(ROOT / "results/egomotion.npz")
    fig = plt.figure(figsize=(DOUBLE, 3.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax.imshow(S["full0"], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    x, y, w, h = config.ROI
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=C["phase"], lw=0.8, ls="--"))
    ax.text(x + 15, y - 25, "ROI (1344 × 1216 px)", color=C["phase"], fontsize=7)
    for t in ego["tile_xy"]:
        ax.add_patch(Rectangle(t - 64, 128, 128, fill=False, ec=C["cot"], lw=0.5, alpha=0.9))
    ax.plot([], [], color=C["cot"], lw=0.8, label="background tiles (Fig. 7)")
    for p, name, col in [(S["cable"], "cable target", C["zncc"]), (S["ref"], "reference (tripod)", C["refined"])]:
        ax.add_patch(Rectangle(p - 96, 192, 192, fill=False, ec=col, lw=1.2))
        ax.text(p[0] - 96, p[1] - 110, name, color=col, fontsize=7.5, fontweight="bold")
    ax.set(xlim=(0, S["full0"].shape[1]), ylim=(S["full0"].shape[0], 0))
    ax.axis("off")
    ax.legend(loc="lower left", fontsize=6.5, labelcolor="white", facecolor="none")
    label(ax, "a", x=0.0)

    ax = fig.add_subplot(gs[1])
    c = S["cable"] - S["roi"]
    hp = 110
    patch = S["roi_gray0"][int(c[1]) - hp:int(c[1]) + hp, int(c[0]) - hp:int(c[0]) + hp]
    ax.imshow(patch, cmap="gray", vmin=0, vmax=255, extent=(-hp, hp, hp, -hp), interpolation="nearest")
    a, n = S["axis"] * 80, S["normal"] * 60
    ax.annotate("", xy=a, xytext=-a, arrowprops=dict(arrowstyle="<->", color=C["zncc"], lw=1.4))
    ax.annotate("", xy=n, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=C["phase"], lw=1.8))
    ax.text(a[0] + 4, a[1] + 14, "cable axis 26.5°", color=C["zncc"], fontsize=7, ha="right", va="top")
    ax.text(-hp + 4, -hp + 6, "measured direction ⊥ axis\n= LDS line of sight", color=C["phase"], fontsize=7, va="top")
    ax.text(-hp + 4, hp - 6, f"55 × 55 mm target; {1 / MM_PER_PX * config.SQUARE_MM:.1f} px per 27.5 mm square", fontsize=6.5, color="white")
    ax.set(xticks=[], yticks=[])
    label(ax, "b", x=0.0)
    save(fig, "fig1_scene",
         "The measurement problem: a vibrating checkerboard and a stationary tripod checkerboard at the same ≈2 m depth; only the in-plane component perpendicular to the cable axis is measured.",
         "Frame 0 of the UAV video (3840 × 2160 px, 50 fps). (a) Region of interest, the two 55 mm checkerboard targets and the 40 background tiles used for the dense ego-motion field of Fig. 7. "
         "(b) The cable target: the cable axis (26.5° in the image, 26.95° physical) and the measured direction perpendicular to it, which coincides with the laser displacement sensor (LDS) line of sight.")


# ---------------------------------------------------------------------------
# Fig 2 — phase principle

def fig2_phase_principle(S):
    c = S["cable"] - S["roi"]
    P = config.PHASE_PATCH // 2
    patch = S["roi_gray0"][int(c[1]) - P:int(c[1]) + P, int(c[0]) - P:int(c[0]) + P].astype(np.float32)
    spec = steerable.PyramidSpec(**config.PHASE_SPEC, pad=config.PHASE_PAD)
    F = steerable.build_filters(patch.shape, spec)
    X0 = steerable.padded_fft(patch, F)
    b = 0  # finest scale, horizontal orientation (phi_x != 0)
    S0 = steerable.decompose_band(X0, F, b)
    shifts = np.array([-1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0])
    est = []
    gx, _ = steerable.band_gradient(X0, F, b)
    w = np.abs(S0) ** 2
    px = np.imag(np.conj(S0) * gx) / (w + 1e-9)          # d(phase)/dx, as in phase_disp
    m = np.zeros_like(w, bool)
    m[24:-24, 24:-24] = True
    for dx in shifts:
        St = steerable.decompose_band(steerable.padded_fft(synth.fourier_shift(patch, dx, 0.0), F), F, b)
        dphi = np.angle(St * np.conj(S0))
        # 1-D phase constancy: phi_x * u + dphi = 0  ->  u = -sum(w phi_x dphi) / sum(w phi_x^2)
        est.append(-np.sum((w * px * dphi)[m]) / np.sum((w * px * px)[m]))
    est = np.array(est)
    St = steerable.decompose_band(steerable.padded_fft(synth.fourier_shift(patch, 0.3, 0.0), F), F, b)
    dphi = np.angle(St * np.conj(S0))

    fig, axs = plt.subplots(1, 4, figsize=(DOUBLE, 2.3), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.35], wspace=0.55))
    axs[0].imshow(patch, cmap="gray", vmin=0, vmax=255); axs[0].set_xlabel("board patch\n192 × 192 px")
    axs[1].imshow(np.abs(S0), cmap="magma"); axs[1].set_xlabel("|S₀| of the finest band\n(= LSQ weight)")
    axs[2].imshow(dphi, cmap="RdBu_r", vmin=-1.2, vmax=1.2)
    axs[2].set_xlabel("Δφ for a 0.3 px shift\n(blue −1.2 … red +1.2 rad)")
    for a in axs[:3]:
        a.set(xticks=[], yticks=[])
    ax = axs[3]
    ax.plot(shifts, shifts, color=C["grey"], lw=0.8, ls="--", label="identity")
    ax.plot(shifts, est, "o", color=C["phase"], ms=3.5, label="phase estimate")
    err = est - shifts
    ax.text(0.04, 0.96, f"max |error| = {np.abs(err).max() * 1000:.1f} × 10⁻³ px", transform=ax.transAxes, va="top", fontsize=7)
    ax.set(xlabel="applied shift u (px)", xticks=[-1, -0.5, 0, 0.5, 1], title="estimated shift u (px)")
    ax.legend(loc="lower right")
    for a, s in zip(axs, "abcd"):
        label(a, s, x=0.0 if s != "d" else -0.32, y=1.02)
    save(fig, "fig2_phase_principle",
         "Local phase is a linear, sub-pixel ruler: the phase difference of one steerable-pyramid band is proportional to the shift, and one weighted least-squares solve recovers it to 10⁻³ px.",
         "Principle of the local-phase tracker. (a) The 192 × 192 px board patch at frame 0. (b) Amplitude of the finest-scale, horizontally oriented band of its complex steerable pyramid; amplitude squared is the weight in the least-squares solve. "
         "(c) Phase difference Δφ = arg(S_t · conj S_0) after a synthetic 0.3 px Fourier shift of the patch: uniform over the checker edges, i.e. Δφ = −φ_x u. (d) Shift recovered from the phase-constancy constraint versus the applied shift for ±1 px (noise-free).")


# ---------------------------------------------------------------------------
# Fig 3 — synthetic benchmark

def fig3_synthetic():
    def read(p):
        with open(p) as f:
            return [dict(r, zoom=int(r["zoom"]), amp_px=float(r["amp_px"]), rmse_px=float(r["rmse_px"])) for r in csv.DictReader(f)]
    clean, comp = read(ROOT / "results/synth/E4.csv"), read(ROOT / "results/synth/E4C.csv")
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, zoom in zip(axs, (1, 2)):
        for rows, ls, tag in [(clean, "--", "clean"), (comp, "-", "H.264")]:
            for tr, col, name in [("phase", C["phase"], "local phase"), ("zncc", C["zncc"], "ZNCC")]:
                r = sorted([x for x in rows if x["tracker"] == tr and x["zoom"] == zoom], key=lambda x: x["amp_px"])
                if r:
                    ax.plot([x["amp_px"] for x in r], [x["rmse_px"] for x in r], ls=ls, marker="o", ms=3, color=col,
                            label=f"{name}, {tag}")
        ax.axhline(0.01, color=C["grey"], lw=0.6, ls=":"); ax.text(2.1, 0.0105, "0.01 px", fontsize=6.5, color=C["grey"], ha="right")
        ax.set(xscale="log", yscale="log", xlabel="motion amplitude (px)", title=f"zoom ×{zoom}" + (" (native)" if zoom == 1 else " (lanczos upscale)"))
        ax.grid(True, which="major", color=C["light"], lw=0.5)
    axs[0].set_ylabel("RMSE (px, model pixels)")
    axs[0].legend(loc="upper left", ncol=1)
    label(axs[0], "a"); label(axs[1], "b", x=-0.04)
    save(fig, "fig3_synthetic_benchmark",
         "On synthetic sequences the codec, not the estimator, sets the floor: local phase is 0.001 px clean and 0.009–0.015 px under H.264 — 3–6× below ZNCC — and gains nothing from upscaling.",
         "Synthetic benchmark (300 frames, sinusoidal motion of the real frame-0 board, ±8 px reflect padding). RMSE in model pixels versus motion amplitude for the local-phase tracker and ZNCC template matching, "
         "on noise-free (dashed) and H.264-compressed (solid) sequences, (a) at native resolution and (b) after 2× Lanczos upscaling. Upscaling helps ZNCC but hurts phase, so the phase tracker runs at native resolution.")


# ---------------------------------------------------------------------------
# Fig 4 — time history

def fig4_time_history():
    ev = np.load(ROOT / "results/eval_phase_3-14.npz")
    v, l = ev["vision"], ev["lds"]
    t = np.arange(len(v)) / FPS
    fig = plt.figure(figsize=(DOUBLE, 3.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1], width_ratios=[2.2, 1], hspace=0.45, wspace=0.18)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t, l, color=C["lds"], lw=0.6, label="LDS (10 kHz → 50 Hz)")
    ax.plot(t, v, color=C["phase"], lw=0.6, alpha=0.85, label="UAV video, local phase")
    ax.set(xlabel="time (s)", ylabel="displacement (mm)", xlim=(0, 60))
    ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.9, edgecolor="none")
    ax.set_title(f"RMSE {float(ev['rmse']):.4f} mm   ·   r = {np.corrcoef(v, l)[0, 1]:.3f}   ·   LDS std {l.std():.3f} mm", loc="left", fontsize=8)
    ax.add_patch(Rectangle((30, ax.get_ylim()[0]), 1.5, np.diff(ax.get_ylim())[0], fc=C["band"], alpha=0.35, ec="none"))
    label(ax, "a", x=-0.06)
    ax = fig.add_subplot(gs[1, 0])
    m = (t >= 30) & (t <= 31.5)
    ax.plot(t[m], l[m], color=C["lds"], lw=1.0, marker=".", ms=2.5)
    ax.plot(t[m], v[m], color=C["phase"], lw=1.0, marker=".", ms=2.5, alpha=0.85)
    ax.set(xlabel="time (s)", ylabel="displacement (mm)", xlim=(30, 31.5), title="1.5 s detail (shaded in a)")
    label(ax, "b", x=-0.2)
    ax = fig.add_subplot(gs[1, 1])
    r = v - l
    ax.hist(r, bins=60, color=C["phase"], alpha=0.8)
    ax.axvline(0, color=C["grey"], lw=0.6)
    ax.set(xlabel="residual (mm)", ylabel="count", title=f"residual: σ = {r.std():.4f} mm")
    label(ax, "c", x=-0.3)
    save(fig, "fig4_time_history",
         "The video-based displacement reproduces the laser reference over the full 60 s to 0.075 mm RMSE (r = 0.98) at 50 Hz, after ego-motion cancellation and a 3–14 Hz band-pass.",
         "Displacement of the cable target perpendicular to the cable axis. (a) Full record: LDS reference decimated to 50 Hz (black) and the local-phase result from the UAV video (blue), both mean-removed, aligned by cross-correlation (0.20 s clock offset). "
         "(b) Detail of the shaded 1.5 s window. (c) Histogram of the residual.")


# ---------------------------------------------------------------------------
# Fig 5 — spectra and coherence

def fig5_spectra():
    evf = np.load(ROOT / "results/eval_phase.npz")          # full-band output
    v, l = evf["vision"], evf["lds"]
    f, pl = signal.welch(l, fs=FPS, nperseg=1024)
    _, pv = signal.welch(v, fs=FPS, nperseg=1024)
    _, coh = signal.coherence(v, l, fs=FPS, nperseg=512)
    fig, axs = plt.subplots(2, 1, figsize=(SINGLE, 4.1), sharex=True, gridspec_kw=dict(hspace=0.3))
    for ax in axs:
        ax.axvspan(3, 14, color=C["band"], alpha=0.35, lw=0)
    ax = axs[0]
    ax.semilogy(f, pl, color=C["lds"], label="LDS")
    ax.semilogy(f, pv, color=C["phase"], alpha=0.85, label="video (unfiltered)")
    for fm, name in [(3.9, "mode 1"), (7.5, "mode 2"), (11.7, "mode 3")]:
        ax.annotate(name, xy=(fm, pl[np.argmin(np.abs(f - fm))]), xytext=(fm + 0.6, pl[np.argmin(np.abs(f - fm))] * 2), fontsize=6.5)
    ax.set(ylabel="PSD (mm²/Hz)", ylim=(1e-8, 1), xlim=(0, 25))
    ax.legend(loc="upper right")
    bp = evf["band_lds_power"]
    ax.text(0.02, 0.05, f"LDS power: {bp[0] * 100:.1f} % below 3 Hz · {bp[1] * 100:.1f} % in 3–14 Hz · {bp[2] * 100:.2f} % above",
            transform=ax.transAxes, fontsize=6.5)
    label(ax, "a", x=-0.2)
    ax = axs[1]
    ax.plot(f[:0] , [], color=C["phase"])
    fc = np.linspace(0, FPS / 2, len(coh))
    ax.plot(fc, coh, color=C["phase"])
    ax.set(xlabel="frequency (Hz)", ylabel="coherence video–LDS", ylim=(0, 1.02), xlim=(0, 25))
    br, bc = evf["band_rmse"], evf["band_coh"]
    for (lo, hi), r, c in zip(config.SCORE_BANDS, br, bc):
        ax.text((lo + hi) / 2, 1.04, f"RMSE {r:.3f} mm\ncoh. {c:.2f}", ha="center", va="bottom", fontsize=6.3,
                transform=ax.get_xaxis_transform())
    label(ax, "b", x=-0.2)
    save(fig, "fig5_spectra_coherence",
         "All of the cable's motion (99.6 % of LDS power) lies in 3–14 Hz where video and laser are coherent (0.98); the video's extra content below 3 Hz is incoherent with the laser — it is residual UAV parallax, not vibration — which justifies the 3–14 Hz band.",
         "Spectral view of the unfiltered video result against the LDS. (a) Welch power spectral densities; the three vortex-induced-vibration modes and the 3–14 Hz analysis band (shaded). "
         "(b) Magnitude-squared coherence between video and LDS. Band-wise RMSE and coherence are printed.")


# ---------------------------------------------------------------------------
# Fig 6 — tracker comparison

def fig6_trackers():
    rows = [("CoTracker3 (2× zoom, 9 pts)", "cotracker", C["cot"]), ("ZNCC template matching", "zncc", C["zncc"]),
            ("fused ZNCC + CoTracker3", "fused", C["fused"]), ("CoTracker3 + refinement net", "refined", C["refined"]),
            ("local phase (this work)", "phase_3-14", C["phase"])]
    vals = []
    for name, key, col in rows:
        ev = np.load(ROOT / f"results/eval_{key}.npz")
        vals.append((name, float(ev["rmse"]), float(np.corrcoef(ev["vision"], ev["lds"])[0, 1]), col))
    fig, ax = plt.subplots(figsize=(SINGLE, 2.3))
    y = np.arange(len(vals))[::-1]
    for yi, (name, r, c, col) in zip(y, vals):
        ax.barh(yi, r, color=col, height=0.62)
        ax.text(r + 0.002, yi, f"{r:.4f} mm  (r = {c:.3f})", va="center", fontsize=7)
    ax.set(yticks=y, yticklabels=[v[0] for v in vals], xlabel="RMSE vs LDS, 3–14 Hz (mm)", xlim=(0, 0.16))
    ax.axvline(0.002, color=C["grey"], lw=0.6, ls=":")
    ax.text(0.004, y[0] + 0.55, "LDS noise ≈ 0.002 mm", fontsize=6, color=C["grey"], va="bottom")
    ax.tick_params(axis="y", length=0)
    save(fig, "fig6_tracker_comparison",
         "Of five trackers on the same pipeline, the local-phase tracker is the most accurate (0.0746 mm, r = 0.980), ahead of a scene-adapted refinement CNN and classical ZNCC.",
         "RMSE against the LDS in the 3–14 Hz band for the five trackers implemented on the shared pipeline (same detection, ego-motion cancellation, projection and band-pass). Correlation coefficients in parentheses. "
         "The dotted line is the LDS's own in-band noise.")


# ---------------------------------------------------------------------------
# Fig 7 — ego-motion: same-depth marker vs far background

def fig7_egomotion(S):
    ego = np.load(ROOT / "results/egomotion.npz")
    keep = ego["keep"].astype(bool)
    est, _ = egomotion.ego_at_target(ego["tile_xy"][keep], ego["disp"][:, keep], ego["sigma"][:, keep], S["ref"], "affine", None)
    n = min(len(est), len(ego["marker_track"]))
    e_n = bandpass(est[:n] @ S["normal"]) * MM_PER_PX
    m_n = bandpass(ego["marker_track"][:n] @ S["normal"]) * MM_PER_PX
    t = np.arange(n) / FPS
    g2 = json.loads(str(ego["g2"]))
    r_marker = float(np.load(ROOT / "results/eval_phase_3-14.npz")["rmse"])
    r_dense = float(np.load(ROOT / "results/eval_phase_dense_3-14.npz")["rmse"])

    fig = plt.figure(figsize=(DOUBLE, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.1, 1], width_ratios=[1.45, 1], hspace=0.3, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(S["full0"], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    tiles = ego["tile_xy"]
    rms = np.sqrt(np.mean(bandpass(ego["disp"][:n] @ S["normal"]) ** 2, axis=0))
    sc = ax.scatter(tiles[:, 0], tiles[:, 1], c=rms * MM_PER_PX, cmap="viridis", s=38, marker="s", ec="white", lw=0.4)
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01, shrink=0.7); cb.set_label("tile motion, 3–14 Hz RMS (mm)", fontsize=7); cb.ax.tick_params(labelsize=6)
    for p, name, col in [(S["cable"], "target", C["zncc"]), (S["ref"], "marker", C["refined"])]:
        ax.plot(*p, "o", mfc="none", mec=col, ms=9, mew=1.4); ax.text(p[0] + 70, p[1] + 10, name, color=col, fontsize=7.5, fontweight="bold")
    ax.axis("off"); label(ax, "a", x=0.0)

    ax = fig.add_subplot(gs[1, 0])
    m = (t >= 20) & (t <= 22)
    ax.plot(t[m], m_n[m], color=C["refined"], label="marker's own motion (same depth as target)")
    ax.plot(t[m], e_n[m], color=C["dense"], label="background field evaluated at the marker")
    ax.set(xlabel="time (s)", ylabel="ego-motion, 3–14 Hz (mm)", xlim=(20, 22))
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1, fontsize=6.2, borderaxespad=0)
    ax.text(0.02, 0.05, f"disagreement {g2['affine']:.3f} px RMS = {g2['affine'] * MM_PER_PX:.3f} mm (gate 0.02 px)", transform=ax.transAxes, fontsize=6.5)
    label(ax, "b", y=1.2)

    ax = fig.add_subplot(gs[1, 1])
    bars = [("same-depth\nmarker", r_marker, C["refined"]), ("dense far\nbackground", r_dense, C["dense"])]
    for i, (nm, r, col) in enumerate(bars):
        ax.bar(i, r, color=col, width=0.6); ax.text(i, r + 0.006, f"{r:.3f} mm", ha="center", fontsize=7)
    ax.set(xticks=[0, 1], xticklabels=[b[0] for b in bars], ylabel="RMSE vs LDS, 3–14 Hz (mm)", ylim=(0, 0.34), title="ego-motion reference")
    label(ax, "c", x=-0.35)
    save(fig, "fig7_egomotion_parallax",
         "A dense 40-tile background field is not a valid ego-motion reference for a 2 m target: it disagrees with the same-depth marker by 0.31 px in band (depth parallax of the drone's translational jitter) and triples the error; the same-depth reference cancels this parallax exactly.",
         "Ego-motion reference: same-depth marker versus far background. (a) The 40 static background tiles tracked with the phase estimator over the full 4K frame, coloured by their 3–14 Hz motion; target and tripod marker circled. "
         "(b) Two seconds of the Huber-robust affine background field evaluated at the marker, against the marker's own phase track (both projected on the measured direction, 3–14 Hz). "
         "(c) Resulting RMSE against the LDS when either is used as the ego-motion reference.")


# ---------------------------------------------------------------------------
# Fig 8 — magnification QC

def fig8_magnification(S):
    clip = str(ROOT / "results/magnified_11.5-11.9_a50_gdgif.mp4")
    mag = np.stack(list(stream_frames(clip, crop=(0, 0, 512, 512), gray=True, count=250)))
    H = mag.shape[1] // 2
    c = S["cable"] - S["roi"]
    from uav_disp.video_io import roi_frames
    inp = np.stack([np.asarray(f)[int(c[1]) - H:int(c[1]) + H, int(c[0]) - H:int(c[0]) + H] for f in roi_frames(config.ROI, count=250)])
    fig, axs = plt.subplots(2, 1, figsize=(DOUBLE, 3.2), sharex=True, gridspec_kw=dict(hspace=0.25))
    for ax, arr, title in [(axs[0], inp, "input video"), (axs[1], mag, "phase-magnified, 11.5–11.9 Hz, α = 50 (+ Row-GDGIF)")]:
        sl = arr[:, H - 80:H + 80, H].T
        ax.imshow(sl, cmap="gray", aspect="auto", extent=(0, arr.shape[0] / FPS, 160, 0), interpolation="nearest")
        ax.set(ylabel="y (px)", title=title)
    axs[1].set_xlabel("time (s)")
    label(axs[0], "a", x=-0.06); label(axs[1], "b", x=-0.06)
    save(fig, "fig8_magnification_qc",
         "Phase-based magnification of the 11.7 Hz band turns an invisible 0.02 px oscillation into a visible, rigid up-and-down motion of the whole board — a visual check that the third mode is genuine in-plane cable motion, not tracker artefact.",
         "Visual verification by phase-based motion magnification (Yang & Jiang's pipeline with the Row-GDGIF). Space-time slices through the board centre column for (a) the stabilised input crop and (b) the same crop magnified 50× in the 11.5–11.9 Hz band (first 5 s). "
         "Magnification is used only for verification, never for measurement.")


# ---------------------------------------------------------------------------

def write_readme():
    lines = ["# Manuscript figures", "",
             "Generated by `python scripts/make_manuscript_figures.py` (PBS stage `figs`). Each figure carries one message; "
             "read in order they tell the story: problem → principle → synthetic validation → real-data result → why 3–14 Hz → "
             "tracker comparison → ego-motion parallax → visual verification.", "",
             "| # | file | message |", "|---|---|---|"]
    lines.append("| 0 | `fig0_setup_competition.png` | Wind-tunnel setup — figure from the competition brief (HIT WTWF), not generated here |")
    for i, (name, msg, _) in enumerate(CAPTIONS, 1):
        lines.append(f"| {i} | `{name}.png` | {msg} |")
    lines += ["", "## Captions", ""]
    for i, (name, _, cap) in enumerate(CAPTIONS, 1):
        lines += [f"**Fig. {i}** (`{name}`). {cap}", ""]
    (OUT / "README.md").write_text("\n".join(lines))
    print("wrote figures/README.md")


def main():
    S = load_scene()
    for fn in (lambda: fig1_scene(S), lambda: fig2_phase_principle(S), fig3_synthetic, fig4_time_history, fig5_spectra,
               fig6_trackers, lambda: fig7_egomotion(S), lambda: fig8_magnification(S)):
        try:
            fn()
        except Exception as e:      # keep going: one failed figure must not lose the rest
            import traceback
            traceback.print_exc()
            print(f"FAILED: {e}", flush=True)
    write_readme()


if __name__ == "__main__":
    main()
