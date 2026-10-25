"""
CSI serial logger with ground-truth labeling.
Run one session per zone: connects to the RX node's serial port, reads CSI_DATA
lines for a fixed duration, tags every packet with the given zone label, writes
a labeled CSV to disk. Stores the raw int8 I/Q values (lossless, ~7x smaller than
amplitude/phase JSON); amplitude and phase are recomputed when loading.

Usage:
    python csi_logger.py --port COM5 --label zoneA --duration 600 --outdir ../../data/raw
"""
import argparse
import csv
import sys
import time
from pathlib import Path

from csi_parser import parse_csi_line


class SerialLineSource:
    """Wraps pyserial so csi_logger doesn't need to know about real vs mock sources."""
    def __init__(self, port: str, baud: int = 921600, timeout: float = 1.0):
        import serial
        self._ser = serial.Serial(port, baud, timeout=timeout)

    def readline(self) -> str:
        raw = self._ser.readline()
        return raw.decode("utf-8", errors="ignore")

    def close(self):
        self._ser.close()


def run_logging_session(line_source, label: str, duration_s: float, outdir: Path, session_name: str | None = None):
    outdir.mkdir(parents=True, exist_ok=True)
    session_name = session_name or f"{label}_{int(time.time())}"
    outpath = outdir / f"{session_name}.csv"

    n_ok, n_skipped, n_malformed = 0, 0, 0
    warned = False
    start = time.monotonic()

    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["host_timestamp", "device_timestamp", "label", "rssi", "mac", "n_values", "raw_iq"])
        try:
            while time.monotonic() - start < duration_s:
                if n_ok == 0 and not warned and time.monotonic() - start > 10:
                    print("WARNING: no CSI packets in the first 10s. Check the COM port, that the TX board is powered, "
                          "and that the RX board is plugged into its UART port.", file=sys.stderr)
                    warned = True

                line = line_source.readline()
                if not line:
                    continue

                record = parse_csi_line(line)
                if record is None:
                    n_skipped += 1
                    continue
                if "error" in record:
                    n_malformed += 1
                    continue

                writer.writerow([
                    time.time(),
                    record["local_timestamp"],
                    label,
                    record["rssi"],
                    record["mac"],
                    len(record["raw"]),
                    " ".join(map(str, record["raw"])),
                ])
                n_ok += 1

                if n_ok % 500 == 0:
                    elapsed = time.monotonic() - start
                    print(f"  [{elapsed:6.1f}s] {n_ok} packets logged (skipped={n_skipped}, malformed={n_malformed})", file=sys.stderr)
        except KeyboardInterrupt:
            print("\nStopped early with Ctrl+C, keeping what was logged.", file=sys.stderr)

    print(f"Session done: {n_ok} packets written to {outpath} (skipped={n_skipped}, malformed={n_malformed})")
    return outpath, n_ok, n_skipped, n_malformed


def main():
    ap = argparse.ArgumentParser(description="Log labeled CSI data for one zone session.")
    ap.add_argument("--port", required=True, help="Serial port, e.g. COM5 (Windows) or /dev/ttyUSB0 (Linux)")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--label", required=True, help="Ground-truth zone label for this session, e.g. zoneA")
    ap.add_argument("--duration", type=float, required=True, help="Session length in seconds")
    ap.add_argument("--outdir", default="../../data/raw")
    args = ap.parse_args()

    print(f"Opening {args.port} @ {args.baud} baud, logging label='{args.label}' for {args.duration}s...")
    src = SerialLineSource(args.port, args.baud)
    try:
        run_logging_session(src, args.label, args.duration, Path(args.outdir))
    finally:
        src.close()


if __name__ == "__main__":
    main()
