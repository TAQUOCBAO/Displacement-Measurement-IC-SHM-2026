"""Alignment and scoring of the vision signal against the LDS reference.

The LDS/video clocks are offset by an unknown sub-second delay (data notes), so
the lag is estimated by cross-correlation over +/- MAX_LAG_S. The correlation
sign also settles the projection-direction sign convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from . import config


@dataclass
class Alignment:
    lag_samples: int      # vision leads LDS by this many 50 Hz samples (can be negative)
    sign: float           # +1/-1 applied to the vision signal
    rmse: float
    corr: float           # peak normalized correlation
    vision: np.ndarray    # trimmed, sign-fixed, zero-mean vision segment
    lds: np.ndarray       # matching LDS segment


def align_and_score(vision: np.ndarray, lds50: np.ndarray, fps: float = config.FPS) -> Alignment:
    v = vision - vision.mean()
    r = lds50 - lds50.mean()
    max_lag = int(config.MAX_LAG_S * fps)

    full = signal.correlate(v, r, mode="full")
    lags = signal.correlation_lags(len(v), len(r), mode="full")
    window = np.abs(lags) <= max_lag
    idx = np.argmax(np.abs(full[window]))
    lag = int(lags[window][idx])
    sign = float(np.sign(full[window][idx]))

    if lag >= 0:  # vision[lag:] aligns with lds[0:]
        n = min(len(v) - lag, len(r))
        v_seg, r_seg = v[lag : lag + n], r[:n]
    else:
        n = min(len(v), len(r) + lag)
        v_seg, r_seg = v[:n], r[-lag : -lag + n]

    v_seg = sign * v_seg
    v_seg = v_seg - v_seg.mean()
    r_seg = r_seg - r_seg.mean()
    rmse = float(np.sqrt(np.mean((v_seg - r_seg) ** 2)))
    corr = float(np.dot(v_seg, r_seg) / (np.linalg.norm(v_seg) * np.linalg.norm(r_seg)))
    return Alignment(lag, sign, rmse, corr, v_seg, r_seg)


def band_scores(vision: np.ndarray, lds: np.ndarray, bands=config.SCORE_BANDS,
                fps: float = config.FPS, nperseg: int = 512) -> dict:
    """Per-band RMSE / coherence / LDS power fraction for two ALIGNED equal-length signals.

    Full-band RMSE cannot show whether the low (<3 Hz) and high (>14 Hz) bands are
    right, because the LDS has little power there; magnitude-squared coherence can.
    """
    v = vision - vision.mean()
    r = lds - lds.mean()
    f, coh = signal.coherence(v, r, fs=fps, nperseg=nperseg)
    fw, p_lds = signal.welch(r, fs=fps, nperseg=nperseg)
    _, p_err = signal.welch(v - r, fs=fps, nperseg=nperseg)
    df = fw[1] - fw[0]
    out = {"bands": [], "rmse": [], "coherence": [], "lds_power_frac": [], "err_rms": []}
    total = p_lds.sum() * df
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        mw = (fw >= lo) & (fw < hi)
        out["bands"].append((lo, hi))
        out["coherence"].append(float(np.average(coh[m], weights=p_lds[mw] + 1e-30)))
        out["lds_power_frac"].append(float(p_lds[mw].sum() * df / total))
        out["err_rms"].append(float(np.sqrt(p_err[mw].sum() * df)))
        sos = None
        if lo <= 0 and hi >= fps / 2:
            vb, rb = v, r
        else:
            if lo <= 0:
                sos = signal.butter(4, hi, "lp", fs=fps, output="sos")
            elif hi >= fps / 2:
                sos = signal.butter(4, lo, "hp", fs=fps, output="sos")
            else:
                sos = signal.butter(4, [lo, hi], "bp", fs=fps, output="sos")
            vb, rb = signal.sosfiltfilt(sos, v), signal.sosfiltfilt(sos, r)
        out["rmse"].append(float(np.sqrt(np.mean((vb - rb) ** 2))))
    out["f"], out["coh_f"] = f, coh
    return out


def format_band_table(bs: dict) -> str:
    lines = ["| band (Hz) | RMSE (mm) | coherence | LDS power |", "|---|---|---|---|"]
    for (lo, hi), rm, c, p in zip(bs["bands"], bs["rmse"], bs["coherence"], bs["lds_power_frac"]):
        lines.append(f"| {lo:g}-{hi:g} | {rm:.4f} | {c:.3f} | {100 * p:.2f} % |")
    return "\n".join(lines)
