"""Dense background ego-motion field over the full 4K frame (one decode pass).

Usage: python scripts/run_egomotion.py [--n-frames N] [--tiles K] [--device cuda]
       python scripts/run_egomotion.py --part i/n [...]     # worker: frames of chunk i of n -> results/egomotion_part{i}.npz
       python scripts/run_egomotion.py --merge n            # concatenate the n chunks, then G2/G3 as in the single-process run
The 4K pass is decode-bound (~5 fps) unless data/cache/full_gray.npy exists (STAGE=cache4k,
25 GB, built once); without the cache the chunked form gives ~n× speed-up with n workers
(each on its own GPU, see STAGE=ego_par in job.pbs); every chunk tracks against frame 0, so the
concatenation is bit-identical to the single-process run.
Needs results/tracks_zncc.npz (coarse global shift = reference-marker track).
Writes results/egomotion.npz: tile_xy (K,2) abs px, disp (T,K,2), sigma, flags, keep,
coarse (T,2), model (chosen by the G2 parallax test), tau (rolling shutter, s/row) + diagnostics,
and cable_tracks (T,P) scalar tracks along the cable for the E-B estimator.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_disp import config, egomotion, phase_disp  # noqa: E402
from uav_disp.steerable import PyramidSpec  # noqa: E402
from uav_disp.video_io import full_frames, stream_frames  # noqa: E402

PROBE = 1500
N_CABLE_PTS = 8
CABLE_STEP_PX = 70


def band_rms(x: np.ndarray, lo=config.HIGHPASS_HZ, hi=config.LOWPASS_HZ, fps=config.FPS) -> float:
    sos = signal.butter(4, [lo, hi], "bp", fs=fps, output="sos")
    return float(np.sqrt(np.mean(signal.sosfiltfilt(sos, x - x.mean(), axis=0) ** 2)))


def chunk_bounds(T: int, i: int, n: int) -> tuple[int, int]:
    """Frame range [lo, hi) of chunk i of n over frames 1..T-1 (frame 0 is the reference)."""
    edges = np.linspace(1, T, n + 1).astype(int)
    return int(edges[i]), int(edges[i + 1])


def track_frames(a, zn, T: int, lo: int, hi: int, t0: float) -> dict:
    """Decode frames [lo, hi) and run the tile / cable-point / marker phase accumulators against frame 0.
    Returns per-frame arrays for frames lo..hi-1 (plus frame 0 when lo == 1) and the static geometry."""
    roi = np.array(config.ROI[:2], float)
    coarse = zn["ref"][:T] - zn["ref"][0]                       # global shift, px, rel. frame 0
    cable_abs = zn["cable"][0] + roi
    ref_abs = zn["ref"][0] + roi

    rgb0 = next(stream_frames(count=1))
    gray0 = (rgb0 @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
    probe = next(full_frames(start=PROBE, count=1))
    tile_xy = egomotion.select_tiles(gray0, probe, rgb0, cable_abs, config.EGO_TILE, a.tiles)
    print(f"selected {len(tile_xy)} tiles in {time.perf_counter() - t0:.0f} s")

    # "virtual accelerometers" along the cable (E-B): points at +-k*CABLE_STEP along the axis
    ev = np.load(ROOT / "results/eval_zncc.npz")
    ang = float(ev["angle_rad"])
    axis = np.array([np.cos(ang), np.sin(ang)])
    offs = (np.arange(N_CABLE_PTS) - (N_CABLE_PTS - 1) / 2) * CABLE_STEP_PX
    cable_pts = cable_abs + offs[:, None] * axis[None, :]
    cable_pts = cable_pts[np.abs(offs) > config.PHASE_PATCH / 2]          # skip the board itself
    cuts_c, _ = phase_disp.coarse_cuts(zn["cable"][:T], config.PHASE_COARSE_STEP, config.PHASE_COARSE_K)
    cuts_r, _ = phase_disp.coarse_cuts(zn["ref"][:T], config.PHASE_COARSE_STEP, config.PHASE_COARSE_K)

    spec = PyramidSpec(**config.PHASE_SPEC, pad=config.PHASE_PAD)
    tracker = egomotion.TileTracker(gray0, tile_xy, coarse, spec, config.EGO_TILE, device=a.device)
    tracker.t = lo                                                # chunk start (cuts index)
    c0 = cuts_c[0] + roi.astype(int)
    caccs = [phase_disp.PhaseAccumulator(phase_disp.cut_patches(gray0, np.rint(p - (zn["cable"][0] + roi) + c0).astype(int), 64),
                                         spec, per_scale=False, device=a.device) for p in cable_pts]
    marker = phase_disp.PhaseAccumulator(phase_disp.cut_patches(gray0, np.rint(ref_abs).astype(int), 128), spec, per_scale=False, device=a.device)

    for i, frame in enumerate(full_frames(start=lo, count=hi - lo), start=lo):
        tracker.push(frame)
        cc = cuts_c[i] + roi.astype(int)
        for acc, p in zip(caccs, cable_pts):
            acc.push(phase_disp.cut_patches(frame, np.rint(p - (zn["cable"][0] + roi) + cc).astype(int), 64),
                     init_uv=zn["cable"][i] - cuts_c[i])
        cr = cuts_r[i] + roi.astype(int)
        marker.push(phase_disp.cut_patches(frame, cr, 128), init_uv=zn["ref"][i] - cuts_r[i])
        if i % 500 == 0:
            print(f"  frame {i}/{T}  {time.perf_counter() - t0:.0f} s", flush=True)

    # accumulator rows: [frame 0, lo, ..., hi-1]; drop the frame-0 row unless this chunk starts at 1
    fits = [acc.result() for acc in tracker.accs]
    idx = np.r_[0, np.arange(lo, hi)]
    tile_cuts = tracker.cuts[idx] - tracker.cuts[0]
    disp = np.stack([tile_cuts + f.uv for f in fits], axis=1)
    sigma = np.stack([f.sigma for f in fits], axis=1)
    flags = np.stack([f.flags for f in fits], axis=1)
    normal = np.array([np.sin(ang), -np.cos(ang)])
    cable_tracks = np.stack([(cuts_c[idx] - cuts_c[0] + np.array(acc.uv)) @ normal for acc in caccs], axis=1)
    marker_track = cuts_r[idx] - cuts_r[0] + np.array(marker.uv)
    s = slice(0, None) if lo == 1 else slice(1, None)
    return dict(frames=idx[s], disp=disp[s], sigma=sigma[s], flags=flags[s], cable_tracks=cable_tracks[s],
                marker_track=marker_track[s], tile_xy=tile_xy, cable_pts=cable_pts, cable_rows=cable_pts[:, 1],
                normal=normal, coarse=coarse, ref_abs=ref_abs)


def finalize(d: dict, t0: float) -> None:
    """Staticness screen, G2 parallax test, G3 rolling-shutter estimate -> results/egomotion.npz."""
    tile_xy, disp, sigma, flags = d["tile_xy"], d["disp"], d["sigma"], d["flags"]
    cable_tracks, marker_track, normal, ref_abs = d["cable_tracks"], d["marker_track"], d["normal"], d["ref_abs"]
    keep = egomotion.staticness_screen(disp)
    print(f"tiles kept by staticness screen: {keep.sum()}/{len(keep)}")

    # G2 parallax test: field evaluated at the reference marker vs the marker's own motion (3-14 Hz RMS)
    results = {}
    for model, near in [("translation", None), ("affine", None), ("affine", config.EGO_NEAR_PX)]:
        est, _ = egomotion.ego_at_target(tile_xy[keep], disp[:, keep], sigma[:, keep], ref_abs, model, near)
        n = min(len(est), len(marker_track))
        err = est[:n] - marker_track[:n]
        key = f"{model}{'-near' if near else ''}"
        results[key] = band_rms(err @ normal)
        print(f"  G2 {key:16s}: field-at-marker vs marker, 3-14 Hz RMS {results[key]:.4f} px")
    best = min(results, key=results.get)
    model, near = best.split("-")[0], (config.EGO_NEAR_PX if best.endswith("near") else np.nan)
    print(f"  chosen model: {best}  ({'PASS' if results[best] <= 0.02 else 'FAIL'} G2 <= 0.02 px)")

    tau_a, r2_a = egomotion.estimate_tau_from_tiles(disp[:, keep], tile_xy[keep], config.FPS)
    rows = d["cable_rows"]
    tau_b, r2_b = egomotion.estimate_tau_from_cable(cable_tracks, rows, fps=config.FPS)
    agree = np.isfinite(tau_a) and np.isfinite(tau_b) and abs(tau_a - tau_b) <= 0.3 * max(abs(tau_a), abs(tau_b))
    tau = 0.5 * (tau_a + tau_b) if agree else 0.0
    print(f"  rolling shutter: tau_A {tau_a * 1e6:.2f} us/row (r2 {r2_a:.2f}), tau_B {tau_b * 1e6:.2f} us/row (r2 {r2_b:.2f})"
          f" -> readout {tau_a * 2160 * 1e3:.1f} / {tau_b * 2160 * 1e3:.1f} ms; {'AGREE, applied' if agree else 'DISAGREE, not applied'} (G3)")

    np.savez(ROOT / "results/egomotion.npz", tile_xy=tile_xy, disp=disp, sigma=sigma, flags=flags, keep=keep,
             coarse=d["coarse"], model=model, near_px=near, tau=tau, tau_a=tau_a, tau_b=tau_b, r2_a=r2_a, r2_b=r2_b,
             g2=json.dumps(results), marker_track=marker_track, cable_tracks=cable_tracks, cable_rows=rows,
             runtime_s=time.perf_counter() - t0)
    print(f"wrote results/egomotion.npz in {time.perf_counter() - t0:.0f} s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=None)
    ap.add_argument("--tiles", type=int, default=config.EGO_N_TILES)
    ap.add_argument("--device", default=None, help="torch device for the tile phase tracker (e.g. cuda)")
    ap.add_argument("--part", default=None, help="i/n: track only chunk i of n frames -> results/egomotion_part{i}.npz")
    ap.add_argument("--merge", type=int, default=None, help="n: merge results/egomotion_part{0..n-1}.npz and finalize")
    a = ap.parse_args()
    zn = np.load(ROOT / "results/tracks_zncc.npz")
    T = len(zn["ref"]) if a.n_frames is None else min(a.n_frames, len(zn["ref"]))
    t0 = time.perf_counter()

    if a.merge is not None:
        parts = [dict(np.load(ROOT / f"results/egomotion_part{i}.npz")) for i in range(a.merge)]
        d = dict(parts[0])
        for k in ("frames", "disp", "sigma", "flags", "cable_tracks", "marker_track"):
            d[k] = np.concatenate([p[k] for p in parts], axis=0)
        assert np.array_equal(d["frames"], np.arange(T)), f"chunks do not tile frames 0..{T - 1}"
        finalize(d, t0)
        return

    if a.part is not None:
        i, n = (int(v) for v in a.part.split("/"))
        lo, hi = chunk_bounds(T, i, n)
        print(f"chunk {i}/{n}: frames [{lo}, {hi})")
        d = track_frames(a, zn, T, lo, hi, t0)
        np.savez(ROOT / f"results/egomotion_part{i}.npz", **d)
        print(f"wrote results/egomotion_part{i}.npz in {time.perf_counter() - t0:.0f} s")
        return

    d = track_frames(a, zn, T, 1, T, t0)
    finalize(d, t0)


if __name__ == "__main__":
    main()
