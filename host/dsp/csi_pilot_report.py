"""
Pilot report: what is the firmware actually sending? Run it on a ~60 s capture before freezing the
firmware configuration or collecting any dataset.

    python csi_logger.py --port COM5 --label empty --activity baseline --duration 60 --firmware default-ht40
    python csi_pilot_report.py --csv ../../data/raw/<that file>.csv

Prints: achieved packet rate and loss, packet-type mix (len / sig_mode / bandwidth / stbc / secondary
channel), which LLTF buffer positions carry signal, amplitude (gain) stability; saves a 3-panel PNG.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csi_features import HI_POS, LO_POS  # noqa: E402
from csi_io import read_meta, unwrap_device_time  # noqa: E402

TYPE_COLS = ["len", "sig_mode", "bandwidth", "stbc", "secondary_channel"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    csv = Path(a.csv)
    out = Path(a.out) if a.out else csv.with_name(csv.stem + "_pilot.png")
    df = pd.read_csv(csv, dtype={"raw_iq": str})
    meta = read_meta(csv)
    n = len(df)
    warn = []

    t = unwrap_device_time(df["device_timestamp"].to_numpy())
    dur = float(t[-1])
    rate = (n - 1) / dur if dur > 0 else float("nan")
    d = np.diff(df["seq"].to_numpy())
    lost = int(((d - 1)[(d > 1) & (d < 5000)]).sum())
    loss = 100.0 * lost / max(n + lost, 1)
    gaps = np.diff(t) * 1000
    print(f"session: {meta.get('session')}  firmware tag: {meta.get('firmware')}")
    print(f"packets: {n}   device duration: {dur:.1f} s   achieved rate: {rate:.1f} pkt/s (firmware nominal: 100)")
    print(f"estimated loss: {loss:.2f}%   inter-arrival ms: p50={np.percentile(gaps, 50):.1f} "
          f"p95={np.percentile(gaps, 95):.1f} max={gaps.max():.1f}")
    print(f"rssi: mean={df['rssi'].mean():.1f} std={df['rssi'].std():.2f}   noise floor: {df['noise_floor'].mode()[0]}")
    if rate < 80:
        warn.append(f"achieved rate {rate:.0f}/s is well below the nominal 100/s")
    if loss > 5:
        warn.append(f"estimated packet loss {loss:.1f}% is high (move the nodes closer or change channel)")

    print("\nfield value counts (top 5):")
    for c in TYPE_COLS + ["channel", "rate", "mcs"]:
        print(f"  {c:18s} {df[c].value_counts(dropna=False).head(5).to_dict()}")
    combos = df.groupby(TYPE_COLS, dropna=False).size().sort_values(ascending=False)
    print("\npacket-type combinations (len, sig_mode, bandwidth, stbc, secondary_channel):")
    for k, v in combos.head(6).items():
        print(f"  {100 * v / n:5.1f}%  {k}")
    if (combos / n > 0.02).sum() > 1:
        warn.append("more than one packet type above 2%: other traffic or mixed frame formats are being captured")
    top_len = int(df["len"].mode()[0])
    if top_len < 128:
        warn.append(f"most common len is {top_len} (< 128): no full LLTF block, features cannot be computed")

    sub = df[df["len"] == top_len].head(3000)
    arr = np.stack([np.array(s.split(), dtype=np.int16) for s in sub["raw_iq"]])[:, :top_len].astype(float)
    amp = np.abs(arr[:, 1::2] + 1j * arr[:, 0::2])
    mean_amp = amp.mean(0)
    lltf = mean_amp[:64]
    nz = lltf[lltf > 0]
    active = lltf > 0.2 * np.median(nz) if len(nz) else np.zeros(64, bool)
    expect = np.zeros(64, bool)
    expect[LO_POS] = expect[HI_POS] = True
    extra, silent = np.flatnonzero(active & ~expect), np.flatnonzero(~active & expect)
    print(f"\nLLTF block (first 64 complex values) for len={top_len}: {int(active.sum())} active positions "
          f"(expected 52 at 1..26 and 38..63)")
    print(f"  active but unexpected: {extra.tolist()}   expected but silent: {silent.tolist()}")
    if len(extra) > 4 or len(silent) > 4:
        warn.append("LLTF layout differs from the documented one: re-check the block interpretation before trusting features")
    lvl = amp[:, :64][:, expect].mean(1)
    print(f"mean LLTF amplitude level: {lvl.mean():.1f}, per-packet CV (gain/AGC jitter + noise): {lvl.std() / lvl.mean():.4f}")
    print(f"blocks in this packet type: {len(mean_amp) / 64:.2f} x 64 complex values")

    print("\nWARNINGS:" if warn else "\nNo warnings.")
    for w in warn:
        print("  -", w)

    fig, ax = plt.subplots(3, 1, figsize=(11, 10))
    ax[0].plot(mean_amp)
    for b in range(64, len(mean_amp), 64):
        ax[0].axvline(b, color="k", ls=":", lw=0.8)
    ax[0].set_title(f"mean amplitude per buffer position, len={top_len} (dotted lines every 64 = block boundaries)")
    ax[1].hist(np.clip(gaps, 0, 60), bins=80)
    ax[1].set_title("packet inter-arrival time (ms, clipped at 60)")
    ax[2].plot(t[:len(lvl)] if len(lvl) == len(t) else np.arange(len(lvl)) / max(rate, 1), lvl)
    ax[2].set_title("per-packet mean LLTF amplitude (should be flat in an empty room)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\nplot saved: {out}")


if __name__ == "__main__":
    main()
