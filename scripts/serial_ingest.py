"""
serial_ingest.py — live Arduino serial stream -> Supabase.

Handles both device roles via --role:
  controller  : enriched CO2 unit.  Parses 'CTRL,...' -> control_telemetry (wide).
  env_logger  : control-chamber unit. Parses 'ENV,...'  -> observations (long).

Bind --port to a STABLE udev symlink (e.g. /dev/ttyACM-enriched), NEVER the raw
enumeration order: two near-identical Arduinos can swap ttyACM0/1 on reboot and
silently mislabel chambers — corrupting the exact comparison the experiment rests
on. See deploy/99-growthchamber-arduino.rules.

Env:  SUPABASE_URL, SUPABASE_KEY, [EXPERIMENT_ID]
Run:  python scripts/serial_ingest.py --role controller --port /dev/ttyACM-enriched
      python scripts/serial_ingest.py --role env_logger --port /dev/ttyACM-control
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

from observations_db import (get_client, parse_ctrl_line, parse_env_line,
                             upsert_control, upsert_observations)

FLUSH_SECONDS = 60  # flush a partial batch this often so a slow feed doesn't stall

# Per-role wiring: which parser, which upsert/table, and whether the parser returns
# a list of rows (env_logger emits one observation row per metric).
ROLES = {
    "controller": {
        "parse": parse_ctrl_line, "upsert": upsert_control,
        "many": False, "default_source": "co2_controller_enriched",
    },
    "env_logger": {
        "parse": parse_env_line, "upsert": upsert_observations,
        "many": True, "default_source": "env_logger_control",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=list(ROLES))
    ap.add_argument("--port", default=os.environ.get("ARDUINO_PORT", "/dev/ttyACM0"))
    ap.add_argument("--baud", type=int, default=9600)  # matches firmware Serial.begin
    ap.add_argument("--source", help="override the source label (defaults per role)")
    ap.add_argument("--experiment-id",
                    default=os.environ.get("EXPERIMENT_ID", "ee496_arabidopsis_round2"))
    ap.add_argument("--batch", type=int, default=12,
                    help="rows to buffer before upsert (~1/min at 5s cadence)")
    args = ap.parse_args()

    role = ROLES[args.role]
    source = args.source or role["default_source"]
    parse, upsert, many = role["parse"], role["upsert"], role["many"]

    client = get_client()
    buf = []
    last_flush = time.monotonic()

    def flush():
        nonlocal buf, last_flush
        if buf:
            n = upsert(client, buf)
            print(f"[ingest:{args.role}] upserted {n} rows", flush=True)
            buf = []
        last_flush = time.monotonic()

    def add(parsed):
        if parsed is None:
            return
        for row in (parsed if many else [parsed]):
            row["experiment_id"] = args.experiment_id
            row["source"] = source
            buf.append(row)

    while True:  # outer reconnect loop
        try:
            with serial.Serial(args.port, args.baud, timeout=10) as ser:
                print(f"[ingest:{args.role}] connected {args.port} @ {args.baud}", flush=True)
                while True:
                    raw = ser.readline().decode("utf-8", "replace")
                    if raw:
                        add(parse(raw))
                    if len(buf) >= args.batch or (
                        buf and time.monotonic() - last_flush >= FLUSH_SECONDS
                    ):
                        flush()
        except serial.SerialException as e:
            print(f"[ingest:{args.role}] serial error: {e}; retrying in 5s",
                  file=sys.stderr, flush=True)
            flush()  # don't lose what we've buffered
            time.sleep(5)
        except KeyboardInterrupt:
            flush()
            print(f"[ingest:{args.role}] stopped", flush=True)
            return


if __name__ == "__main__":
    main()
