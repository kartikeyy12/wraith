"""
Sanity-check plot for one logged session.

    python csi_visualize.py --csv ../../data/raw/<session>.csv

Panels: LLTF amplitude heatmap (valid subcarriers x time), turbulence series (Hampel filtered),
and the per-window turbulence std (the main motion feature).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dsp"))
from csi_features import (FeatureConfig, amplitude_valid, hampel, resample_uniform,  # noqa: E402
                          session_windows, turbulence)
from csi_io import load_session  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fs", type=float, default=50.0)
    ap.add_argument("--win-s", type=float, default=5.0)
    ap.add_argument("--step-s", type=float, default=2.5)
    a = ap.parse_args()
    cfg = FeatureConfig(fs=a.fs, win_s=a.win_s, step_s=a.step_s)
    csv = Path(a.csv)
    out = Path(a.out) if a.out else csv.with_name(csv.stem + "_check.png")

    s = load_session(csv)
    g, Ag, ok = resample_uniform(s.t, amplitude_valid(s, cfg), cfg.fs, cfg.max_gap_s)
    turb = hampel(turbulence(Ag, cfg.turb_mode), cfg.hampel_win, cfg.hampel_thr)
    X, ts, names = session_windows(s, cfg)

    fig, ax = plt.subplots(3, 1, figsize=(12, 9), height_ratios=[2, 1, 1], sharex=True)
    im = ax[0].imshow(Ag.T, aspect="auto", origin="lower", cmap="viridis", extent=[g[0], g[-1], 0, Ag.shape[1]])
    fig.colorbar(im, ax=ax[0], label="amplitude")
    ax[0].set_ylabel("valid subcarrier")
    ax[0].set_title(f"{s.name}  label={s.label}  activity={s.activity}  day={s.day}  packets={len(s.t)}")
    ax[1].plot(g, turb, lw=0.8)
    ax[1].set_ylabel("turbulence")
    if len(X):
        ax[2].plot(ts + cfg.win_s / 2, X[:, names.index("turb_std")], marker="o")
    ax[2].set_ylabel("window turb_std")
    ax[2].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"saved {out}   windows={len(X)}   dropped packets={s.n_dropped}")
    if len(X):
        v = X[:, names.index("turb_std")]
        print(f"turb_std: mean={v.mean():.4f} min={v.min():.4f} max={v.max():.4f}")


if __name__ == "__main__":
    main()
