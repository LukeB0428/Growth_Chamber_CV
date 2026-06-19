"""
verify_ingest.py — integration checks for the live data path (run ON the Pi after
rows start landing). This is the four-point validation, not a smoke test:

  1. Row counts per (source, cadence)        — both chambers present? both rates?
  2. Duplicate (source, rtc_timestamp)        — dedup/upsert holding? (want 0)
  3. Latest rtc_offset_sec per source         — drift guard alive + sane (~0)?
  4. Melt-view metrics per chamber            — observations_unified joined right?

Env: SUPABASE_URL, SUPABASE_KEY
Run: python scripts/verify_ingest.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from observations_db import get_client


def main():
    c = get_client()

    # 1. counts per (source, cadence)
    ct = c.table("control_telemetry").select("source,cadence").execute().data
    print("1) control_telemetry by (source,cadence):",
          dict(Counter((r["source"], r["cadence"]) for r in ct)))

    # 2. duplicate key check (the whole reason for the upsert)
    keys = [(r["source"], r["rtc_timestamp"])
            for r in c.table("control_telemetry").select("source,rtc_timestamp").execute().data]
    dups = sum(1 for _, n in Counter(keys).items() if n > 1)
    print(f"2) duplicate (source,rtc_timestamp): {dups}  (want 0)")

    # 3. latest rtc_offset_sec per source — drift guard
    off = (c.table("observations_unified")
           .select("source,value,timestamp")
           .eq("metric_name", "rtc_offset_sec")
           .order("timestamp", desc=True).limit(50).execute().data)
    latest = {}
    for r in off:
        latest.setdefault(r["source"], r)
    print("3) latest rtc_offset_sec per source:")
    for s, r in latest.items():
        flag = "" if abs(r["value"]) < 120 else "  <-- DRIFT / RTC RESET?"
        print(f"     {s}: {r['value']:.1f}s @ {r['timestamp']}{flag}")

    # 4. melt-view: distinct metrics per chamber
    mv = c.table("observations_unified").select("chamber,metric_name").limit(5000).execute().data
    by_ch = {}
    for r in mv:
        by_ch.setdefault(r["chamber"], set()).add(r["metric_name"])
    print("4) observations_unified metrics per chamber:")
    for ch, mset in by_ch.items():
        print(f"     {ch or '(none)'}: {sorted(mset)}")


if __name__ == "__main__":
    main()
