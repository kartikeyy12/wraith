"""
Labeled CSI logger. One run = one session = one CSV + one sidecar .meta.json.

Example (Windows):
    python csi_logger.py --port COM5 --label zoneA --activity moving --duration 600 \
        --room roomK --subject P1 --firmware default-ht40 --notes "door closed"

Labels:      empty | zoneA | zoneB | ... (any string)
Activities:  baseline (empty room, used only to calibrate that day)
             empty    (empty room, a test/training session)
             moving   (person moving naturally inside the zone)
             still    (person standing/sitting still inside the zone)

Packet columns store the raw I/Q values untouched (lossless). Everything needed to
interpret them (len, sig_mode, bandwidth, stbc, secondary_channel, channel) is stored
per packet, and the session sidecar records packet loss and achieved rate.
"""
import argparse
import csv
import datetime
import json
import sys
import time
from pathlib import Path

from csi_parser import parse_csi_line

CSV_COLUMNS = [
    "host_timestamp", "device_timestamp", "seq", "rssi", "noise_floor", "rate",
    "sig_mode", "mcs", "bandwidth", "stbc", "secondary_channel", "channel",
    "len", "first_word", "label", "raw_iq",
]
ACTIVITIES = ("baseline", "moving", "still", "empty")
TOOL_VERSION = "csi_logger v2"


class SerialLineSource:
    """Thin pyserial wrapper so the logger can also be driven by a mock source."""
    def __init__(self, port: str, baud: int = 921600, timeout: float = 1.0):
        import serial
        self._ser = serial.Serial(port, baud, timeout=timeout)

    def readline(self) -> str:
        return self._ser.readline().decode("utf-8", errors="ignore")

    def close(self):
        self._ser.close()


def _write_meta(path: Path, meta: dict):
    path.write_text(json.dumps(meta, indent=2))


def run_logging_session(line_source, label, activity, duration_s, outdir,
                        meta_extra=None, session_name=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now()
    name = session_name or f"{label}_{activity}_{now.strftime('%Y%m%d_%H%M%S')}"
    csv_path = outdir / f"{name}.csv"
    meta_path = outdir / f"{name}.meta.json"

    meta = {
        "session": name, "label": label, "activity": activity,
        "day": now.date().isoformat(), "start_local": now.isoformat(timespec="seconds"),
        "start_epoch": time.time(), "duration_requested_s": duration_s,
        "status": "running", "tool": TOOL_VERSION,
    }
    meta.update({k: v for k, v in (meta_extra or {}).items() if v is not None})
    _write_meta(meta_path, meta)

    n_ok = n_skipped = n_malformed = 0
    lost = resets = 0
    last_seq = prev_ts = None
    dev_elapsed_us = 0
    warned = False
    status = "complete"
    start = time.monotonic()
    last_report = start

    try:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(CSV_COLUMNS)
            while True:
                now_m = time.monotonic()
                if now_m - start >= duration_s or getattr(line_source, "exhausted", False):
                    break
                if n_ok == 0 and not warned and now_m - start > 10:
                    print("WARNING: no CSI packets in the first 10s. Check the COM port, that the TX board is "
                          "powered, and that the RX board is plugged into its UART port.", file=sys.stderr)
                    warned = True

                line = line_source.readline()
                if not line:
                    continue
                rec = parse_csi_line(line)
                if rec is None:
                    n_skipped += 1
                    continue
                if "error" in rec:
                    n_malformed += 1
                    continue

                w.writerow([
                    f"{time.time():.6f}", rec["local_timestamp"], rec["seq"], rec["rssi"],
                    rec["noise_floor"], rec["rate"], rec["sig_mode"], rec["mcs"], rec["bandwidth"],
                    rec["stbc"], rec["secondary_channel"], rec["channel"], rec["len"],
                    rec["first_word"], label, " ".join(map(str, rec["raw"])),
                ])
                n_ok += 1

                s = rec["seq"]
                if last_seq is not None:
                    d = s - last_seq
                    if 1 < d < 5000:
                        lost += d - 1
                    elif d <= 0 or d >= 5000:
                        resets += 1
                last_seq = s
                ts = rec["local_timestamp"]
                if ts is not None and prev_ts is not None:
                    dev_elapsed_us += (ts - prev_ts) % (2 ** 32)
                prev_ts = ts

                if n_ok % 1000 == 0:
                    f.flush()
                if time.monotonic() - last_report >= 5:
                    last_report = time.monotonic()
                    el = last_report - start
                    print(f"  [{el:6.1f}s] {n_ok} packets ({n_ok / el:5.1f}/s), lost~{lost}, "
                          f"skipped={n_skipped}, malformed={n_malformed}", file=sys.stderr)
    except KeyboardInterrupt:
        status = "interrupted"
        print("\nStopped early with Ctrl+C, keeping what was logged.", file=sys.stderr)
    finally:
        elapsed = time.monotonic() - start
        meta.update({
            "status": status, "elapsed_s": round(elapsed, 3), "packets_logged": n_ok,
            "lines_skipped": n_skipped, "lines_malformed": n_malformed,
            "packets_lost_est": lost, "seq_resets": resets,
            "packet_loss_pct_est": round(100.0 * lost / max(n_ok + lost, 1), 3),
            "rate_hz_host": round(n_ok / max(elapsed, 1e-9), 2),
            "rate_hz_device": round(n_ok / (dev_elapsed_us / 1e6), 2) if dev_elapsed_us else None,
        })
        _write_meta(meta_path, meta)

    print(f"Session {status}: {n_ok} packets -> {csv_path} "
          f"(lost~{lost}, skipped={n_skipped}, malformed={n_malformed})")
    return csv_path, meta


def main():
    ap = argparse.ArgumentParser(description="Log one labeled CSI session.")
    ap.add_argument("--port", required=True, help="COM5 (Windows) or /dev/ttyUSB0 (Linux)")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--label", required=True, help="ground truth: empty, zoneA, zoneB, ...")
    ap.add_argument("--activity", required=True, choices=ACTIVITIES)
    ap.add_argument("--duration", type=float, required=True, help="seconds")
    ap.add_argument("--outdir", default="../../data/raw")
    ap.add_argument("--room", help="room id, e.g. roomK")
    ap.add_argument("--subject", help="anonymous person id, e.g. P1 (no names)")
    ap.add_argument("--operator", help="who ran the session")
    ap.add_argument("--setup", help="node placement note, e.g. 'RX on desk by wall, TX shelf 3m'")
    ap.add_argument("--firmware", help="firmware variant tag, e.g. default-ht40")
    ap.add_argument("--notes", help="free text: door open, furniture moved, ...")
    ap.add_argument("--day", help="override session day (YYYY-MM-DD)")
    a = ap.parse_args()

    extra = {"room": a.room, "subject": a.subject, "operator": a.operator, "setup": a.setup,
             "firmware": a.firmware, "notes": a.notes, "day": a.day}
    print(f"Opening {a.port} @ {a.baud}: label={a.label} activity={a.activity} for {a.duration}s")
    src = SerialLineSource(a.port, a.baud)
    try:
        run_logging_session(src, a.label, a.activity, a.duration, a.outdir, meta_extra=extra)
    finally:
        src.close()


if __name__ == "__main__":
    main()
