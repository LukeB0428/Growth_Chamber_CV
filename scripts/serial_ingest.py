"""
serial_ingest.py — live Arduino CTRL serial stream → Supabase control_telemetry.

Long-running Pi-side service. Reads the 5s CTRL feed emitted by
EE496_FYP_CO2Control_1.ino, parses each line, and upserts in small batches.
Reconnects automatically on USB hiccup/reboot. The SD card remains the
authoritative backup; sd_reconcile.py backfills anything this feed drops.

Env:  SUPABASE_URL, SUPABASE_KEY, [ARDUINO_PORT], [EXPERIMENT_ID]
Run:  python scripts/serial_ingest.py --source co2_controller_enriched
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import serial  # pyserial
except ImportError:
    sys.exit("pyserial not installed: pip install pyserial")

from observations_db import get_client, parse_ctrl_line, upsert_control

FLUSH_SECONDS = 60  # also flush a partial batch this often, so slow feeds don't stall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=os.environ.get("ARDUINO_PORT", "/dev/ttyACM0"))
    ap.add_argument("--baud", type=int, default=9600)  # matches firmware Serial.begin
    ap.add_argument("--source", default="co2_controller_enriched")
    ap.add_argument("--experiment-id",
                    default=os.environ.get("EXPERIMENT_ID", "ee496_arabidopsis_round2"))
    ap.add_argument("--batch", type=int, default=12,
                    help="rows to buffer before upsert (~1/min at 5s cadence)")
    args = ap.parse_args()

    client = get_client()
    buf = []
    last_flush = time.monotonic()

    def flush():
        nonlocal buf, last_flush
        if buf:
            n = upsert_control(client, buf)
            print(f"[ingest] upserted {n} rows (latest {buf[-1]['rtc_timestamp']})", flush=True)
            buf = []
        last_flush = time.monotonic()

    while True:  # outer reconnect loop
        try:
            with serial.Serial(args.port, args.baud, timeout=10) as ser:
                print(f"[ingest] connected {args.port} @ {args.baud}", flush=True)
                while True:
                    raw = ser.readline().decode("utf-8", "replace")
                    row = parse_ctrl_line(raw) if raw else None
                    if row is not None:
                        row["experiment_id"] = args.experiment_id
                        row["source"] = args.source
                        buf.append(row)
                    if len(buf) >= args.batch or (
                        buf and time.monotonic() - last_flush >= FLUSH_SECONDS
                    ):
                        flush()
        except serial.SerialException as e:
            print(f"[ingest] serial error: {e}; retrying in 5s", file=sys.stderr, flush=True)
            flush()  # don't lose what we've buffered
            time.sleep(5)
        except KeyboardInterrupt:
            flush()
            print("[ingest] stopped", flush=True)
            return


if __name__ == "__main__":
    main()
