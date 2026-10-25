"""
CSI packet parser for ESP32-S3 esp-csi output.
Parses the CSV line format Espressif's csi_recv example prints over serial,
extracts per-subcarrier amplitude and phase, and keeps the raw int8 I/Q values.
"""
import re
import numpy as np

CSI_FIELDS = [
    "type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth",
    "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi",
    "noise_floor", "ampdu_cnt", "channel", "secondary_channel",
    "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word", "data"
]

def parse_csi_line(line: str) -> dict | None:
    """
    Parse one raw CSI_DATA line into a dict of metadata + amplitude/phase/raw arrays.
    Returns None if the line isn't a valid CSI_DATA line.
    """
    line = line.strip()
    if not line.startswith("CSI_DATA"):
        return None

    match = re.match(r'^(.*?),"?\[(.*?)\]"?\s*$', line)
    if not match:
        return None

    meta_str, data_str = match.groups()
    meta_values = meta_str.split(",")

    if len(meta_values) != len(CSI_FIELDS) - 1:
        return {"error": f"field count mismatch: got {len(meta_values)}, expected {len(CSI_FIELDS)-1}"}

    record = dict(zip(CSI_FIELDS[:-1], meta_values))

    raw = [int(x) for x in data_str.split(",") if x.strip() != ""]
    if len(raw) % 2 != 0:
        raw = raw[:-1]

    imag = np.array(raw[0::2], dtype=np.float64)
    real = np.array(raw[1::2], dtype=np.float64)

    record["n_subcarriers"] = len(real)
    record["amplitude"] = np.sqrt(real**2 + imag**2)
    record["phase"] = np.arctan2(imag, real)
    record["raw"] = raw  # lossless [imag, real, imag, real, ...] ints
    record["rssi"] = int(record["rssi"])
    record["local_timestamp"] = int(record["local_timestamp"])

    return record
