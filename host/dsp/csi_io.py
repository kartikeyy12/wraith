"""
Loading logged sessions.

Only the LLTF block is used for features. The ESP-IDF docs list LLTF as the first block
of the CSI buffer for every packet type: 64 complex values (128 numbers, imag first),
buffer position p <-> subcarrier index p (p < 32) or p - 64 (p >= 32). Of these, positions
1..26 and 38..63 carry signal (802.11a/g: +-26 subcarriers); position 0 is DC and 27..37
are guard. Later blocks (HT-LTF, STBC-HT-LTF) vary with bandwidth/packet type and are
ignored here until the pilot capture shows which are present.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LLTF_VALUES = 128  # 64 subcarriers x (imag, real)


def meta_path_for(csv_path) -> Path:
    p = Path(csv_path)
    return p.with_name(p.stem + ".meta.json")


def read_meta(csv_path) -> dict:
    mp = meta_path_for(csv_path)
    if mp.exists():
        return json.loads(mp.read_text())
    return {"session": Path(csv_path).stem, "label": None, "activity": "unknown", "day": None}


def unwrap_device_time(ts_us: np.ndarray) -> np.ndarray:
    """32-bit microsecond device clock -> seconds, tolerant of wraparound."""
    ts = np.asarray(ts_us, dtype=np.int64)
    d = np.diff(ts) % (2 ** 32)
    d = np.where(d > 2 ** 31, d - 2 ** 32, d)
    return np.concatenate([[0.0], np.cumsum(d)]) / 1e6


@dataclass
class Session:
    name: str
    path: Path
    meta: dict
    t: np.ndarray            # (N,) seconds, strictly increasing
    lltf: np.ndarray         # (N, 64) complex
    df: pd.DataFrame = field(repr=False)   # per-packet metadata for the kept rows
    n_rows: int = 0
    n_dropped: int = 0

    @property
    def label(self):
        return self.meta.get("label")

    @property
    def activity(self):
        return self.meta.get("activity")

    @property
    def day(self):
        return self.meta.get("day")


def parse_raw_row(s: str, n=None):
    parts = s.split(None, n) if n else s.split()
    if n:
        parts = parts[:n]
    return np.array(parts, dtype=np.int16)


def load_session(csv_path, max_rows=None) -> Session:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, dtype={"raw_iq": str}, nrows=max_rows)
    rows, keep = [], []
    for i, s in enumerate(df["raw_iq"].tolist()):
        if isinstance(s, str):
            parts = s.split(None, LLTF_VALUES)[:LLTF_VALUES]
            if len(parts) == LLTF_VALUES:
                rows.append(np.array(parts, dtype=np.int16))
                keep.append(i)
    n_rows = len(df)
    if not rows:
        raise ValueError(f"{csv_path.name}: no packets with a full LLTF block")
    arr = np.stack(rows).astype(np.float64)
    lltf = arr[:, 1::2] + 1j * arr[:, 0::2]
    sub = df.iloc[keep].reset_index(drop=True)
    t = unwrap_device_time(sub["device_timestamp"].to_numpy())
    inc = np.concatenate([[True], np.diff(t) > 0])   # drop out-of-order/duplicate timestamps
    lltf, sub, t = lltf[inc], sub[inc].reset_index(drop=True), t[inc]
    meta = read_meta(csv_path)
    return Session(name=meta.get("session", csv_path.stem), path=csv_path, meta=meta,
                   t=t, lltf=lltf, df=sub.drop(columns=["raw_iq"]),
                   n_rows=n_rows, n_dropped=n_rows - len(t))
