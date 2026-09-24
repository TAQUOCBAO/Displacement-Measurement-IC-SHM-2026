"""End-to-end: tracks -> displacement -> LDS comparison -> submission CSV + plots.

Usage: python scripts/run_pipeline.py [zncc|cotracker|fused|phase|phase_dense|<any tracks_X>]
                                      [--band full|3-14|<lo>-<hi>]   (default: zncc)

Expects results/tracks_<method>.npz from the tracking scripts ("fused" needs
both and combines the two displacement signals with config.FUSION_W_ZNCC).
--band: output band. Default "full" (0-25 Hz, mean removed) for phase* methods,
"3-14" for the legacy trackers so their stored numbers stay reproducible.
Writes results/submission_<method>[_<band>].csv (time_s, displacement_mm @ 50 Hz,
time base aligned to the LDS record start), eval_<method>[_<band>].{png,npz}, and
prints the per-band RMSE / coherence table (results are not judged in 3-14 Hz only).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_disp import config, evaluate, lds, postprocess
from uav_disp.video_io import stream_frames


def parse_band(arg: str | None, method: str) -> tuple[float | None, float | None]:
    if arg is None:
        arg = "full" if method.startswith("phase") else "3-14"
    if arg == "full":
        return None, None
    lo, hi = arg.split("-")
    return (float(lo) if float(lo) > 0 else None), (float(hi) if float(hi) < config.FPS / 2 else None)


def cable_angle(cable0: np.ndarray) -> float:
    ev = ROOT / "results/eval_zncc.npz"
    if ev.exists():
        return float(np.load(ev)["angle_rad"])
    rgb0 = next(stream_frames(crop=config.ROI, count=1))
    return postprocess.cable_axis_angle(rgb0, cable0)


def signal_from_tracks(tracks_path: Path, angle: float | None = None,
                       hp: float | None = config.HIGHPASS_HZ, lp: float | None = config.LOWPASS_HZ):
    d = np.load(tracks_path)
    cable, ref = d["cable"], d["ref"]
    if cable.ndim == 3:  # cotracker: (T, N, 2) point sets -> rigid mean
        cable, ref = cable.mean(axis=1), ref.mean(axis=1)
    if angle is None:
        angle = cable_angle(cable[0])
    mm = postprocess.displacement_mm(cable, ref, float(d["cable_square_px"]), angle, highpass_hz=hp, lowpass_hz=lp)
    return mm, angle, d


def sigma_mm(d, angle: float) -> np.ndarray | None:
    """Per-frame predicted 1-sigma of the projected displacement (mm), if the tracks carry it."""
    if "sigma_cable" not in d:
        return None
    normal = np.abs([np.sin(angle), -np.cos(angle)])
    s2 = (d["sigma_cable"] ** 2) @ normal ** 2
    if "sigma_ref" in d and d["sigma_ref"].shape == d["sigma_cable"].shape:
        s2 = s2 + (d["sigma_ref"] ** 2) @ normal ** 2
    return np.sqrt(s2) * config.SQUARE_MM / float(d["cable_square_px"])


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    method = args[0] if args else "zncc"
    band_arg = sys.argv[sys.argv.index("--band") + 1] if "--band" in sys.argv else None
    hp, lp = parse_band(band_arg, method)
    band_tag = "" if band_arg is None else f"_{band_arg}"
    sig = None
    if method == "fused":
        mm_zn, angle, _ = signal_from_tracks(ROOT / "results/tracks_zncc.npz", hp=hp, lp=lp)
        mm_ct, _, _ = signal_from_tracks(ROOT / "results/tracks_cotracker.npz", angle, hp=hp, lp=lp)
        n = min(len(mm_zn), len(mm_ct))
        w = config.FUSION_W_ZNCC
        mm = w * mm_zn[:n] + (1 - w) * mm_ct[:n]
    else:
        mm, angle, d = signal_from_tracks(ROOT / f"results/tracks_{method}.npz", hp=hp, lp=lp)
        sig = sigma_mm(d, angle)
    method_out = method + band_tag
    band_str = "full (0-25 Hz)" if hp is None and lp is None else f"{hp or 0:g}-{lp or config.FPS / 2:g} Hz"

    lds_raw = lds.load(ROOT / "data/lds.npy", ROOT / "data/LDS data.xlsx")
    lds50 = lds.decimate_to(lds_raw, config.FPS)
    a = evaluate.align_and_score(mm, lds50)
    print(f"[{method}] cable axis: {np.degrees(angle) % 180:.2f} deg | "
          f"lag: {a.lag_samples} samples ({a.lag_samples / config.FPS:.2f} s) | sign: {a.sign:+.0f}")
    print(f"[{method}] band {band_str} | RMSE: {a.rmse:.4f} mm | corr: {a.corr:.4f} | "
          f"LDS std: {lds50.std():.4f} mm | n: {len(a.vision)}")
    bs = evaluate.band_scores(a.vision, a.lds)
    print(evaluate.format_band_table(bs))
    if sig is not None:
        lag = a.lag_samples
        s_seg = sig[max(lag, 0):max(lag, 0) + len(a.vision)]
        ms = float(np.mean(s_seg))
        print(f"[{method}] mean predicted sigma: {ms:.4f} mm | calibration RMSE/sigma: {a.rmse / ms:.2f} "
              f"(trustworthy if within 0.5-2)")
        if "flags_cable" in d:
            fl = d["flags_cable"]
            print(f"[{method}] frac cond>{config.PHASE_COND_MAX:g}: {np.mean(fl & 1 > 0):.3f} | frac wrap-risk: "
                  f"{np.mean(fl & 2 > 0):.3f} | median N_eff: {np.median(d['n_eff_cable']):.0f} | "
                  f"cut changes: {len(d['cut_changes_cable'])} | runtime: {float(d['runtime_s']):.1f} s")

    # submission on the LDS time base
    t = np.arange(len(a.vision)) / config.FPS
    out_csv = ROOT / f"results/submission_{method_out}.csv"
    np.savetxt(out_csv, np.column_stack([t, a.vision]), delimiter=",",
               header="time_s,displacement_mm", comments="", fmt="%.6f")

    fig, axes = plt.subplots(4, 1, figsize=(12, 13))
    axes[0].plot(t, a.lds, lw=0.7, label="LDS")
    axes[0].plot(t, a.vision, lw=0.7, label=f"vision ({method})", alpha=0.8)
    axes[0].set(xlabel="time (s)", ylabel="displacement (mm)",
                title=f"{method} [{band_str}]: RMSE {a.rmse:.4f} mm, corr {a.corr:.4f}")
    axes[0].legend()
    axes[1].plot(t, a.vision - a.lds, lw=0.7, color="crimson")
    axes[1].set(xlabel="time (s)", ylabel="error (mm)", title="residual")
    for x, lab in [(a.lds, "LDS"), (a.vision, "vision")]:
        f, p = signal.welch(x, fs=config.FPS, nperseg=1024)
        axes[2].semilogy(f, p, lw=0.9, label=lab)
    axes[2].set(xlabel="frequency (Hz)", ylabel="PSD (mm$^2$/Hz)", title="spectra")
    axes[2].legend()
    axes[3].plot(bs["f"], bs["coh_f"], lw=0.9)
    for lo, hi in config.SCORE_BANDS:
        axes[3].axvspan(lo, hi, alpha=0.06)
    axes[3].set(xlabel="frequency (Hz)", ylabel="coherence", ylim=(0, 1.02),
                title="vision-LDS coherence: " + ", ".join(f"{lo:g}-{hi:g} Hz {c:.2f}" for (lo, hi), c in zip(bs["bands"], bs["coherence"])))
    fig.tight_layout()
    fig.savefig(ROOT / f"results/eval_{method_out}.png", dpi=130)
    print("wrote", out_csv.name, f"and eval_{method_out}.png")

    np.savez(ROOT / f"results/eval_{method_out}.npz", vision=a.vision, lds=a.lds,
             rmse=a.rmse, lag=a.lag_samples, sign=a.sign, angle_rad=angle, band=band_str,
             band_rmse=np.array(bs["rmse"]), band_coh=np.array(bs["coherence"]),
             band_lds_power=np.array(bs["lds_power_frac"]),
             mean_sigma_mm=np.nan if sig is None else ms)


if __name__ == "__main__":
    main()
