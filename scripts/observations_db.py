"""
observations_db.py — Supabase client + parsers for the chamber monitoring data layer.

Shared by:
  - serial_ingest.py  (live 5s CTRL feed from the Arduino)
  - sd_reconcile.py    (hourly backfill from the SD datalog.txt)

Both write into the `control_telemetry` wide table, upserting on
(source, rtc_timestamp) so a tick is ONE logical row whether it arrived live or
via SD reconciliation. Reconciliation is additive (hourly timestamps rarely
collide with 5s ones); the `cadence` column ('serial_5s' vs 'sd_hourly') lets the
agent prefer the live feed when both cover the same window.

Canonical clock = the Arduino RTC. We also record rtc_offset_sec
(host_local - rtc_local) so the agent can detect the watchdog/boot RTC-reset
flagged in the manifest (rtc_reset_risk). Note: the RTC holds LOCAL wall-clock;
we label it UTC for a stable monotonic axis and compute the offset in local frame,
which is ~0 in normal operation and jumps hard on a reset.
"""
import os
import re
from datetime import datetime, timezone

try:
    from supabase import create_client
except ImportError:  # allow import for offline parser unit-testing
    create_client = None

# Fixed field order of the firmware CTRL serial line.
# See Arduino/EE496_FYP_CO2Control_1.ino — keep in lockstep with the firmware.
CTRL_FIELDS = [
    "unixtime", "setpoint", "co2", "co2_1min", "co2_5min", "co2_24h",
    "duty", "duty_5min", "proj_co2", "proj_dev", "control_state",
    "temp", "pressure", "humidity", "gas",
]

# Fixed field order of the firmware ENV serial line (control-chamber env logger).
# See Arduino/EnvironmentalLogger_EE496_v2_1.ino.
ENV_FIELDS = [
    "unixtime", "co2", "co2_5min", "co2_24h",
    "temp", "pressure", "humidity", "gas",
]


def get_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("Set SUPABASE_URL and SUPABASE_KEY in the environment.")
    if create_client is None:
        raise RuntimeError("supabase-py not installed: pip install supabase")
    return create_client(url, key)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _rtc_fields(unixtime):
    """Return (rtc_timestamp_iso, rtc_offset_sec) for a firmware RTC unixtime."""
    rtc_dt = datetime.fromtimestamp(unixtime, tz=timezone.utc)  # RTC wall-clock as UTC
    rtc_local_naive = rtc_dt.replace(tzinfo=None)
    offset = (datetime.now() - rtc_local_naive).total_seconds()  # ~0 normally
    return rtc_dt.isoformat(), offset


def parse_ctrl_line(line):
    """Parse a 'CTRL,...' serial line → partial control_telemetry row (no
    experiment_id/source — the caller adds those). Returns None if malformed."""
    line = line.strip()
    if not line.startswith("CTRL,"):
        return None
    parts = line.split(",")
    if len(parts) != len(CTRL_FIELDS) + 1:
        return None
    v = dict(zip(CTRL_FIELDS, parts[1:]))
    try:
        unixtime = int(float(v["unixtime"]))
    except (TypeError, ValueError):
        return None

    rtc_iso, offset = _rtc_fields(unixtime)
    setpoint = _f(v["setpoint"])
    co2 = _f(v["co2"])
    return {
        "rtc_timestamp":   rtc_iso,
        "rtc_offset_sec":  offset,
        "setpoint_ppm":    setpoint,
        "measured_co2_ppm": co2,
        "co2_1min_ppm":    _f(v["co2_1min"]),
        "co2_5min_ppm":    _f(v["co2_5min"]),
        "co2_24h_ppm":     _f(v["co2_24h"]),
        "error_ppm":       (setpoint - co2) if (setpoint is not None and co2 is not None) else None,
        "proj_co2_ppm":    _f(v["proj_co2"]),
        "proj_dev":        _f(v["proj_dev"]),
        "duty_cycle":      _f(v["duty"]),
        "duty_5min":       _f(v["duty_5min"]),
        "control_state":   v["control_state"] or None,
        "temp_c":          _f(v["temp"]),
        "pressure_mbar":   _f(v["pressure"]),
        "humidity_pct":    _f(v["humidity"]),
        "gas_kohm":        _f(v["gas"]),
        "cadence":         "serial_5s",
    }


def _grab(text, key):
    """Extract a numeric value for an exact 'key=NUMBER' token (unit suffix ignored)."""
    m = re.search(re.escape(key) + r"=([-\d.]+)", text)
    return float(m.group(1)) if m else None


def parse_sd_line(line, setpoint=1100.0, role="dosing"):
    """Parse one hourly SD datalog.txt record → partial control_telemetry row.
    `setpoint` (not in the SD log) comes from the manifest constant; `role`
    selects how control_state is derived. Returns None if not a log record."""
    if "TimeStamp=" not in line:
        return None
    ts = _grab(line, "TimeStamp")
    if ts is None:
        return None
    unixtime = int(ts)
    rtc_iso, offset = _rtc_fields(unixtime)
    co2 = _grab(line, "CO2")

    if role == "logging_only":
        state = "logging_only"
    else:
        hr = datetime.fromtimestamp(unixtime, tz=timezone.utc).hour
        if hr < 6 or hr > 19:
            state = "overnight_closed"
        elif (co2 or 0) > 1800:
            state = "safety_cutoff"
        else:
            state = "dosing"

    return {
        "rtc_timestamp":   rtc_iso,
        "rtc_offset_sec":  offset,
        "setpoint_ppm":    setpoint,
        "measured_co2_ppm": co2,
        "co2_5min_ppm":    _grab(line, "CO2_5min"),
        "co2_24h_ppm":     _grab(line, "CO2_24h_avg"),
        "error_ppm":       (setpoint - co2) if co2 is not None else None,
        "duty_cycle":      _grab(line, "Duty"),
        "duty_5min":       _grab(line, "Duty_5min"),
        "control_state":   state,
        "temp_c":          _grab(line, "Temperature"),
        "pressure_mbar":   _grab(line, "Pressure"),
        "humidity_pct":    _grab(line, "Humidity"),
        "gas_kohm":        _grab(line, "Gas"),
        "cadence":         "sd_hourly",
    }


def upsert_control(client, rows):
    """Upsert rows into control_telemetry on (source, rtc_timestamp). Idempotent."""
    if not rows:
        return 0
    client.table("control_telemetry").upsert(
        rows, on_conflict="source,rtc_timestamp"
    ).execute()
    return len(rows)


def parse_env_line(line, chamber="control"):
    """Parse an 'ENV,...' serial line -> LIST of long-table observation rows (one per
    metric). The control-chamber logger has no controller fields, so it routes to
    `observations`, not control_telemetry. Metric names align with the melt-view so
    observations_unified yields apples-to-apples cross-chamber rows (the CO2/temp
    calibration caveat lives in the manifest, not here). Returns None if malformed;
    the caller adds experiment_id + source."""
    line = line.strip()
    if not line.startswith("ENV,"):
        return None
    parts = line.split(",")
    if len(parts) != len(ENV_FIELDS) + 1:
        return None
    v = dict(zip(ENV_FIELDS, parts[1:]))
    try:
        unixtime = int(float(v["unixtime"]))
    except (TypeError, ValueError):
        return None

    rtc_iso, offset = _rtc_fields(unixtime)
    metrics = [
        ("measured_co2_ppm", _f(v["co2"]),      "ppm"),
        ("co2_5min_ppm",     _f(v["co2_5min"]), "ppm"),
        ("temp_c",           _f(v["temp"]),     "celsius"),
        ("pressure_mbar",    _f(v["pressure"]), "mbar"),
        ("humidity_pct",     _f(v["humidity"]), "pct"),
        ("gas_kohm",         _f(v["gas"]),      "kohm"),
        ("rtc_offset_sec",   offset,            "seconds"),  # drift guard for this logger
    ]
    rows = []
    for name, val, units in metrics:
        if val is None:
            continue
        rows.append({
            "timestamp":   rtc_iso,
            "metric_name": name,
            "value":       val,
            "units":       units,
            "chamber":     chamber,
            "pot_label":   "",
        })
    return rows


def upsert_observations(client, rows):
    """Upsert long-table observation rows. Idempotent on the full natural key."""
    if not rows:
        return 0
    client.table("observations").upsert(
        rows, on_conflict="experiment_id,timestamp,metric_name,source,chamber,pot_label"
    ).execute()
    return len(rows)
