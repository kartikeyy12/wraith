"""
Synthetic CSI generator for TESTING THE PIPELINE ONLY.

Produces firmware-format CSI_DATA lines and feeds them through the real logger, so the whole
chain (parser -> logger -> CSV/meta -> loader -> features -> evaluation) is exercised without hardware.
Nothing here is evidence about real-world accuracy: the "physics" is a toy model with known structure
(smooth static fingerprint per zone, motion modulation, per-day multiplicative drift).

    python synth_csi.py --out ../../data/synthetic --days 2026-01-01,2026-01-02,2026-01-03
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "logger"))
from csi_logger import run_logging_session  # noqa: E402

VALID = np.concatenate([np.arange(1, 27), np.arange(38, 64)])


def _smooth(rng, n=64, k=6):
    z = rng.normal(size=n) + 1j * rng.normal(size=n)
    return np.convolve(z, np.ones(k) / k, mode="same")


class MockSource:
    """Yields pre-built lines, then reports `exhausted` (the logger stops instead of waiting)."""
    def __init__(self, lines):
        self._it = iter(lines)
        self.exhausted = False

    def readline(self):
        try:
            return next(self._it)
        except StopIteration:
            self.exhausted = True
            return ""

    def close(self):
        pass


def _pack(h, rng):
    """Complex 64-vector (zero outside VALID) -> 128 ints [imag, real, ...]."""
    out = np.zeros(128)
    out[0::2] = np.round(h.imag)
    out[1::2] = np.round(h.real)
    return out.astype(int)


def make_lines(rng, h_day, zone_vec, activity, n_pkts, rate_hz, drop=0.02, noise=0.6, static_amp=0.04):
    t = np.arange(n_pkts) / rate_hz
    amp = np.zeros(n_pkts)
    if activity == "still":
        amp = static_amp + 0.01 * np.sin(2 * np.pi * 0.27 * t + rng.uniform(0, 6.28))
    elif activity == "moving":
        f1, f2 = rng.uniform(0.5, 2.5, 2)
        walk = np.cumsum(rng.normal(0, 0.08, n_pkts))
        amp = static_amp + 0.12 * np.abs(np.sin(2 * np.pi * f1 * t + walk)) + 0.05 * np.abs(np.sin(2 * np.pi * f2 * t))
    present = activity in ("still", "moving")
    ts = (2 ** 32 - 3_000_000) + np.cumsum(np.full(n_pkts, 1e6 / rate_hz) + rng.normal(0, 800, n_pkts))
    lines, seq = [], 0
    for i in range(n_pkts):
        seq += 1
        if rng.random() < drop:
            continue
        h = h_day * (1 + (amp[i] * zone_vec if present else 0))
        h = h * (1 + 0.01 * rng.normal())
        h = h + noise * (rng.normal(size=64) + 1j * rng.normal(size=64))
        h = np.where(np.isin(np.arange(64), VALID), h, 0)
        vals = np.concatenate([_pack(h, rng), _pack(h * (1 + 0.02 * rng.normal()), rng)])
        meta = f"{seq},1a:00:00:00:00:00,-40,11,1,7,1,1,1,0,0,0,0,-90,0,11,2,{int(ts[i]) % 2**32},0,66,0,256,0"
        lines.append(f'CSI_DATA,{meta},"[{",".join(map(str, vals))}]"')
    return lines


def generate(out_dir, days, seed=0, duration_s=60, rate_hz=50, drift=0.15, room="roomS", zones=("zoneA", "zoneB"),
             sessions_per_zone=2, static_amp=0.04):
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    h0 = 40 * (1 + 0.4 * _smooth(rng))
    zvec = {}
    for z in zones:
        v = _smooth(rng)
        zvec[z] = v / np.sqrt((np.abs(v) ** 2).mean())
    n = int(duration_s * rate_hz)
    for day in days:
        h_day = h0 * (1 + drift * _smooth(rng))
        plan = [("empty", "baseline", 1), ("empty", "empty", 1)]
        plan += [(z, a, sessions_per_zone) for z in zones for a in ("moving", "still")]
        for label, activity, reps in plan:
            for r in range(reps):
                lines = make_lines(rng, h_day, zvec.get(label, zvec[zones[0]]), activity, n, rate_hz,
                                   static_amp=static_amp)
                run_logging_session(MockSource(lines), label, activity, 3600, out_dir,
                                    meta_extra={"day": day, "room": room, "subject": "P1", "firmware": "synthetic"},
                                    session_name=f"{day}_{label}_{activity}_{r}")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", default="2026-01-01,2026-01-02,2026-01-03")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--drift", type=float, default=0.15)
    a = ap.parse_args()
    generate(a.out, a.days.split(","), a.seed, a.duration, drift=a.drift)
