"""
sd_reconcile.py — backfill control_telemetry from the Arduino SD datalog (hourly).

The live serial feed drops records on USB hiccups/reboots. The SD card keeps an
authoritative hourly log. This script parses datalog.txt and upserts each record;
because the upsert key is (source, rtc_timestamp) and reconciliation is idempotent,
re-running is safe and only fills genuine gaps. Rows are tagged cadence='sd_hourly'
so the agent can prefer the 5s serial feed where both exist.

NOTE: SD logging is HOURLY only — this restores coverage, not 5s resolution.

Run (point --logfile at the mounted SD or a scp'd copy):
  python scripts/sd_reconcile.py --logfile /mnt/sd/datalog.txt --source co2_controller_enriched
  python scripts/sd_reconcile.py --logfile /mnt/sd/datalog.txt --source env_logger_control --role logging_only
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from observations_db import get_client, parse_sd_line, upsert_control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logfile", required=True, help="path to SD datalog.txt (mounted or copied)")
    ap.add_argument("--source", default="co2_controller_enriched")
    ap.add_argument("--experiment-id", default=os.environ.get("EXPERIMENT_ID", "arabidopsis_co2"))
    ap.add_argument("--setpoint", type=float, default=1100.0,
                    help="manifest setpoint (SD log omits it)")
    ap.add_argument("--role", default="dosing", choices=["dosing", "logging_only"])
    args = ap.parse_args()

    client = get_client()
    rows = []
    for line in Path(args.logfile).read_text(errors="replace").splitlines():
        row = parse_sd_line(line, setpoint=args.setpoint, role=args.role)
        if row is None:
            continue
        row["experiment_id"] = args.experiment_id
        row["source"] = args.source
        rows.append(row)

    n = upsert_control(client, rows)
    print(f"[reconcile] upserted {n} hourly rows from {args.logfile}")


if __name__ == "__main__":
    main()
