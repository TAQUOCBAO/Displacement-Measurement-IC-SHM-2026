"""Paper figures from benchmark CSVs and real-data evaluations.

Usage: python scripts/make_figures.py
Reads results/synth/E1.csv, E1C.csv, E2C.csv and results/eval_*.npz.
Writes paper/fig1_zoomlaw.(png|pdf), fig2_linearity, fig3_validation.
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

C_COTRACKER = "#2E63D6"  # validated categorical pair (dataviz six checks)
C_ZNCC = "#C85200"
INK, MUTED = "#222222", "#777777"
MM_PER_PX = 27.5 / 31.09

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "grid.color": "#E4E4E0", "grid.linewidth": 0.6,
    "figure.facecolor": "white", "savefig.dpi": 200,
})


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fig1_zoomlaw():
    clean = read_csv(ROOT / "results/synth/E1.csv")
    comp = read_csv(ROOT / "results/synth/E1C.csv")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for rows, ls, tag in [(comp, "-", "H.264"), (clean, "--", "clean")]:
        for tracker, color in [("cotracker", C_COTRACKER), ("zncc", C_ZNCC)]:
            pts = [(float(r["zoom"]), float(r["rmse_px"])) for r in rows
                   if r["tracker"] == tracker and float(r["amp_px"]) == 0.5]
            z, e = zip(*sorted(pts))
            ax.plot(z, e, ls, color=color, lw=1.8, marker="o", ms=4.5,
                    mfc="white", mew=1.4, mec=color)
            if tag == "H.264":
                name = "CoTracker3" if tracker == "cotracker" else "ZNCC"
                ax.annotate(name, (z[-1], e[-1]), xytext=(6, 0),
                            textcoords="offset points", color=color, fontsize=9,
                            va="center")
    ax.set(xlabel="magnification Z (model px per video px)",
           ylabel="RMSE (video px)", yscale="log")
    ax.grid(True, axis="y")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color=INK, ls="-", label="H.264 round-trip (realistic)"),
        Line2D([], [], color=INK, ls="--", label="clean (lower bound)"),
    ], frameon=False, fontsize=8, loc="lower left")
    sec = ax.secondary_yaxis("right", functions=(lambda x: x * MM_PER_PX,
                                                 lambda x: x / MM_PER_PX))
    sec.set_ylabel("RMSE (mm)", color=MUTED)
    sec.tick_params(colors=MUTED)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"paper/fig1_zoomlaw.{ext}")
    plt.close(fig)


def fig2_linearity():
    rows = read_csv(ROOT / "results/synth/E2C.csv")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for tracker, color in [("cotracker", C_COTRACKER), ("zncc", C_ZNCC)]:
        pts = [(float(r["amp_px"]), float(r["rmse_px"])) for r in rows
               if r["tracker"] == tracker]
        a, e = zip(*sorted(pts))
        name = "CoTracker3" if tracker == "cotracker" else "ZNCC"
        ax.plot(a, e, "-", color=color, lw=1.8, marker="o", ms=4.5,
                mfc="white", mew=1.4, mec=color)
        ax.annotate(name, (a[-1], e[-1]), xytext=(6, 0),
                    textcoords="offset points", color=color, fontsize=9, va="center")
    ax.set(xlabel="vibration amplitude (px)", ylabel="RMSE (px)",
           xscale="log", yscale="log")
    ax.grid(True, which="both", axis="both")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"paper/fig2_linearity.{ext}")
    plt.close(fig)


def fig3_validation():
    ev = np.load(ROOT / "results/eval_refined.npz")
    v, r = ev["vision"], ev["lds"]
    t = np.arange(len(v)) / 50.0
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6),
                             gridspec_kw={"width_ratios": [1.7, 1]})
    w = slice(1500, 1750)  # 5 s window for readability
    axes[0].plot(t[w], r[w], color=INK, lw=1.1, label="LDS reference")
    axes[0].plot(t[w], v[w], color=C_COTRACKER, lw=1.1, alpha=0.85,
                 label="vision (refined)")
    axes[0].set(xlabel="time (s)", ylabel="displacement (mm)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right", ncols=2)
    axes[0].grid(True, axis="y")
    for x, color, lab in [(r, INK, "LDS"), (v, C_COTRACKER, "vision")]:
        f, p = signal.welch(x, fs=50, nperseg=1024)
        axes[1].semilogy(f, p, color=color, lw=1.1, label=lab)
    axes[1].set(xlabel="frequency (Hz)", ylabel="PSD (mm$^2$/Hz)", xlim=(0, 20))
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(True, axis="y")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"paper/fig3_validation.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    done = []
    for fn in [fig1_zoomlaw, fig2_linearity, fig3_validation]:
        try:
            fn()
            done.append(fn.__name__)
        except FileNotFoundError as e:
            print(f"skip {fn.__name__}: missing {e.filename}")
    print("wrote:", ", ".join(done))
