"""
Quick sanity-check visualization for a labeled CSI CSV session (as produced by csi_logger.py).
Plots:
  1. Amplitude heatmap: subcarrier index (y) vs packet index (x)
  2. NBVI over time, computed via sliding windows, to eyeball motion/stillness separation

Usage:
    python3 csi_visualize.py --csv ../../data/raw/zoneA_1234567890.csv --out zoneA_check.png
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dsp"))
from csi_features import nbvi, sliding_windows, trim_subcarriers


def load_session(csv_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    amplitudes = np.stack([np.array(json.loads(row)) for row in df["amplitude_json"]])
    return amplitudes, df


def plot_session(amplitudes: np.ndarray, df: pd.DataFrame, out_path: Path, window_size: int = 1000, step: int = 500):
    label = df["label"].iloc[0] if "label" in df.columns else "unknown"
    n_packets = amplitudes.shape[0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

    im = axes[0].imshow(amplitudes.T, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title(f"CSI Amplitude Heatmap — label='{label}' — {n_packets} packets")
    axes[0].set_xlabel("Packet index")
    axes[0].set_ylabel("Subcarrier index")
    fig.colorbar(im, ax=axes[0], label="Amplitude")

    packet_list = [trim_subcarriers(a) for a in amplitudes]
    if n_packets >= window_size:
        nbvi_vals, centers = [], []
        for start, window in sliding_windows(packet_list, window_size=window_size, step=step):
            nbvi_vals.append(nbvi(window))
            centers.append(start + window_size // 2)
        axes[1].plot(centers, nbvi_vals, marker="o")
        axes[1].set_title(f"NBVI over time (window={window_size}, step={step})")
        axes[1].set_xlabel("Packet index")
        axes[1].set_ylabel("NBVI")
    else:
        axes[1].text(0.5, 0.5, f"Not enough packets for one window\n(have {n_packets}, need {window_size})",
                      ha="center", va="center", transform=axes[1].transAxes)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved plot to {out_path}")
    print(f"Session summary: label='{label}', n_packets={n_packets}, n_subcarriers={amplitudes.shape[1]}")
    if n_packets >= window_size:
        print(f"NBVI: mean={np.mean(nbvi_vals):.4f}, min={np.min(nbvi_vals):.4f}, max={np.max(nbvi_vals):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None, help="Output PNG path. Defaults to <csv_stem>_check.png")
    ap.add_argument("--window-size", type=int, default=1000)
    ap.add_argument("--step", type=int, default=500)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out) if args.out else csv_path.with_name(csv_path.stem + "_check.png")

    amplitudes, df = load_session(csv_path)
    plot_session(amplitudes, df, out_path, window_size=args.window_size, step=args.step)


if __name__ == "__main__":
    main()
