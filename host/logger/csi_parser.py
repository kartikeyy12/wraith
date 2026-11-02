"""
Parser for the CSV lines printed by Espressif's esp-csi `csi_recv` example.

Supports the ESP32-S3 layout (24 metadata fields + data) and the shorter
ESP32-C5/C6/C61 layout (14 metadata fields + data). Pure Python (no numpy) so the
logger runs on a bare Windows install with only pyserial.

The data array is [imag0, real0, imag1, real1, ...] (imaginary first). It is kept
raw: which block of it is LLTF / HT-LTF depends on packet type and bandwidth, so
interpretation happens later, in host/dsp/csi_io.py.
"""
import re

LAYOUT_S3 = [
    "type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth",
    "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi",
    "noise_floor", "ampdu_cnt", "channel", "secondary_channel",
    "local_timestamp", "ant", "sig_len", "rx_format", "len", "first_word",
]
LAYOUT_C6 = [
    "type", "id", "mac", "rssi", "rate", "noise_floor", "fft_gain", "agc_gain",
    "channel", "local_timestamp", "sig_len", "rx_format", "len", "first_word",
]
LAYOUTS = {len(LAYOUT_S3): LAYOUT_S3, len(LAYOUT_C6): LAYOUT_C6}

_STR_FIELDS = {"type", "mac"}
_LINE_RE = re.compile(r'^(.*?),"?\[(.*?)\]"?\s*$')
COMMON_KEYS = (
    "rssi", "noise_floor", "rate", "sig_mode", "mcs", "bandwidth", "stbc",
    "secondary_channel", "channel", "local_timestamp", "len", "first_word",
)


def parse_csi_line(line: str):
    """Return a dict for a valid CSI_DATA line, None for any other line,
    or {"error": ...} for a CSI_DATA line that cannot be parsed."""
    line = line.strip()
    if not line.startswith("CSI_DATA"):
        return None
    m = _LINE_RE.match(line)
    if not m:
        return None
    meta_str, data_str = m.groups()
    parts = meta_str.split(",")
    layout = LAYOUTS.get(len(parts))
    if layout is None:
        return {"error": f"unknown metadata field count {len(parts)}"}
    rec = {}
    try:
        for name, val in zip(layout, parts):
            rec[name] = val if name in _STR_FIELDS else int(val)
        raw = [int(x) for x in data_str.split(",") if x.strip() != ""]
    except ValueError as e:
        return {"error": f"bad number: {e}"}
    if len(raw) % 2:
        raw = raw[:-1]
    rec["seq"] = rec["id"]
    rec["raw"] = raw
    for k in COMMON_KEYS:
        rec.setdefault(k, None)
    return rec
