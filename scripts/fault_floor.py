"""
fault_floor.py — deterministic fault floor for the chamber monitor. NO LLM, NO tokens.

Runs on the Pi (cron, every fault_floor.poll_seconds). Evaluates the manifest's
controller thresholds against recent control_telemetry and maintains an alert
lifecycle in the `alerts` table (open on fire, resolve on clear). This is the
never-miss safety net — it catches CO2 faults in minutes, independent of the agent,
the API, and the agent's schedule. Also appends to a local logfile, so there is a
record even if the Supabase write fails.

Rules (numeric params come from the manifest, so behaviour is portable):
  dosing_undershoot : in the dosing window, measured_co2 < setpoint - tolerance, sustained
  duty_pinned_high  : duty_cycle pinned at the max, sustained, while dosing
  sensor_dropout    : measured_co2 <= 0 or temp_c <= sentinel (instantaneous)
  feed_silent       : newest tick older than stale_after_min, or no rows at all
                      (dead sensor / crashed ingest — the total-silence fault)

evaluate_rules() is a PURE function (rows in, findings out) so it unit-tests offline.

Env: SUPABASE_URL, SUPABASE_KEY
Run: python scripts/fault_floor.py            # single shot (for cron)
     python scripts/fault_floor.py --loop      # standalone loop
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml

_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = _ROOT / "config" / "manifests" / "arabidopsis.yaml"
LOCAL_LOG = _ROOT / "results" / "fault_floor.log"


def load_manifest(path=MANIFEST):
    with open(path) as f:
        return yaml.safe_load(f)


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_ts(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def build_params(m):
    enr = m["controllers"]["enriched"]
    ff = m.get("fault_floor", {})
    return {
        "setpoint":      float(enr["co2_setpoint_ppm"]),
        "tolerance":     float(enr["co2_tolerance_ppm"]),
        "duty_max":      float(enr["duty_cycle_range"][1]),
        "window":        enr["dosing_window_local"],
        "sustain_min":   float(ff.get("sustain_min", 15)),
        "sensor_sentinel": float(ff.get("sensor_sentinel", -7000)),
        "fire_frac":     float(ff.get("fire_frac", 0.9)),
        "min_coverage":  float(ff.get("min_coverage_frac", 0.6)),
        "period_s":      float(m.get("cadence", {}).get("serial_live_sec", 5)),
        "lookback_min":  float(ff.get("lookback_min", 30)),
        "poll_seconds":  float(ff.get("poll_seconds", 300)),
        "stale_after_min": float(ff.get("stale_after_min", 10)),
    }


def evaluate_rules(rows, params, now, now_local=None):
    """Pure evaluator. rows: list of control_telemetry dicts. Returns
    (findings, meta) where findings is one dict per rule with a 'firing' bool —
    non-firing rules are included so the caller can resolve cleared alerts.

    `now` is real UTC (used for the sustain window). `now_local` is host-local
    wall clock (naive); it is the correct basis for the freshness check because
    rtc_timestamp is host-local time cosmetically tagged +00:00 (see
    observations_db._rtc_fields), so comparing it to real UTC would be off by
    the host's UTC offset. Defaults to `now` stripped of tz if not supplied."""
    if now_local is None:
        now_local = now.replace(tzinfo=None)
    rows = sorted(rows, key=lambda r: r["rtc_timestamp"])
    sustain_start = now - timedelta(minutes=params["sustain_min"])
    win = [r for r in rows if _parse_ts(r["rtc_timestamp"]) >= sustain_start]
    expected = max(1, int(params["sustain_min"] * 60 / params["period_s"]))

    setpoint, tol = params["setpoint"], params["tolerance"]
    duty_max, fire_frac, min_cov = params["duty_max"], params["fire_frac"], params["min_coverage"]

    dosing = [r for r in win if r.get("control_state") == "dosing"]
    have_dosing = bool(dosing) and (len(dosing) / expected) >= min_cov

    findings = []

    # 1) dosing_undershoot — only over valid (co2 > 0) ticks, so a dropped sensor
    #    doesn't masquerade as undershoot (sensor_dropout owns that case below).
    valid_dosing = [r for r in dosing if (_num(r.get("measured_co2_ppm")) or 0) > 0]
    if have_dosing and valid_dosing:
        under = [r for r in valid_dosing if _num(r["measured_co2_ppm"]) < setpoint - tol]
        frac = len(under) / len(valid_dosing)
        firing = frac >= fire_frac
        worst = min((_num(r["measured_co2_ppm"]) for r in under), default=None)
        findings.append({
            "rule_id": "dosing_undershoot", "firing": firing, "severity": "high", "value": worst,
            "detail": (f"{frac*100:.0f}% of dosing ticks below {setpoint-tol:.0f} ppm over "
                       f"{params['sustain_min']:.0f} min (min {worst:.0f})") if firing else "",
        })
    else:
        findings.append({"rule_id": "dosing_undershoot", "firing": False,
                         "severity": "high", "value": None, "detail": ""})

    # 2) duty_pinned_high
    if have_dosing:
        pinned = [r for r in dosing if (_num(r.get("duty_cycle")) is not None
                                        and _num(r["duty_cycle"]) >= duty_max - 1)]
        frac = len(pinned) / len(dosing)
        firing = frac >= fire_frac
        findings.append({
            "rule_id": "duty_pinned_high", "firing": firing, "severity": "high", "value": duty_max,
            "detail": (f"duty pinned at {duty_max:.0f} for {frac*100:.0f}% of dosing ticks over "
                       f"{params['sustain_min']:.0f} min") if firing else "",
        })
    else:
        findings.append({"rule_id": "duty_pinned_high", "firing": False,
                         "severity": "high", "value": None, "detail": ""})

    # 3) sensor_dropout — instantaneous on the most recent ticks
    recent = rows[-3:]
    sentinel = params["sensor_sentinel"]
    bad = [r for r in recent
           if (_num(r.get("measured_co2_ppm")) is not None and _num(r["measured_co2_ppm"]) <= 0)
           or (_num(r.get("temp_c")) is not None and _num(r["temp_c"]) <= sentinel)]
    findings.append({
        "rule_id": "sensor_dropout", "firing": bool(bad), "severity": "high", "value": None,
        "detail": "sensor returned a dropout sentinel on recent ticks" if bad else "",
    })

    # 4) feed_silent — data freshness. None of the rules above fire on TOTAL
    #    silence: an empty window makes have_dosing False, and sensor_dropout
    #    only inspects rows[-3:], which are stale (or absent) when the feed has
    #    died. A dead sensor / crashed ingest is exactly the fault the never-miss
    #    floor most needs to catch, so this rule owns it. Age is measured in the
    #    host-local basis (see docstring) to avoid a UTC-offset error.
    stale_after = params["stale_after_min"]
    if rows:
        latest_naive = _parse_ts(rows[-1]["rtc_timestamp"]).replace(tzinfo=None)
        age_min = (now_local - latest_naive).total_seconds() / 60.0
    else:
        age_min = float("inf")
    silent = age_min > stale_after
    if not rows:
        silent_detail = f"no telemetry in the {params['lookback_min']:.0f}-min lookback window"
    elif silent:
        silent_detail = f"no telemetry for {age_min:.0f} min (stale after {stale_after:.0f} min)"
    else:
        silent_detail = ""
    findings.append({
        "rule_id": "feed_silent", "firing": silent, "severity": "high",
        "value": None if age_min == float("inf") else round(age_min, 1),
        "detail": silent_detail,
    })

    # Precedence (one root cause → one alert):
    #  - a silent feed makes EVERY other assessment meaningless → suppress all;
    #  - otherwise a firing sensor_dropout makes CO2/duty assessment meaningless.
    by_id = {f["rule_id"]: f for f in findings}
    if by_id["feed_silent"]["firing"]:
        for f in findings:
            if f["rule_id"] != "feed_silent" and f["firing"]:
                f["firing"] = False
                f["detail"] = "suppressed: feed silent"
    elif by_id["sensor_dropout"]["firing"]:
        for f in findings:
            if f["rule_id"] not in ("sensor_dropout", "feed_silent") and f["firing"]:
                f["firing"] = False
                f["detail"] = "suppressed: sensor dropout active"

    meta = {"n_rows": len(rows), "n_window": len(win), "expected_window": expected,
            "latest_age_min": None if age_min == float("inf") else round(age_min, 1)}
    return findings, meta


def sync_alerts(client, experiment_id, source, findings, now):
    """Open an alert when a rule starts firing; resolve it when the rule clears."""
    open_rows = (client.table("alerts").select("id,rule_id")
                 .eq("experiment_id", experiment_id).eq("source", source)
                 .eq("status", "open").execute().data)
    open_by = {r["rule_id"]: r["id"] for r in open_rows}
    opened, resolved = [], []
    for f in findings:
        rid = f["rule_id"]
        if f["firing"] and rid not in open_by:
            client.table("alerts").insert({
                "experiment_id": experiment_id, "source": source, "rule_id": rid,
                "severity": f["severity"], "status": "open",
                "opened_at": now.isoformat(), "detail": f["detail"], "value": f["value"],
            }).execute()
            opened.append(rid)
        elif (not f["firing"]) and rid in open_by:
            client.table("alerts").update({
                "status": "resolved", "resolved_at": now.isoformat(),
            }).eq("id", open_by[rid]).execute()
            resolved.append(rid)
    return opened, resolved


def _log_local(now, source, firing, opened, resolved, meta, error=None):
    LOCAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = (f"{now.isoformat()} {source} rows={meta.get('n_rows','?')} "
            f"firing={firing} opened={opened} resolved={resolved}"
            + (f" ERROR={error}" if error else ""))
    with open(LOCAL_LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def run_once(client, m, experiment_id, source):
    params = build_params(m)
    now = datetime.now(timezone.utc)
    now_local = datetime.now()  # host-local wall clock — basis for the freshness check
    cutoff = (now - timedelta(minutes=params["lookback_min"])).isoformat()
    try:
        rows = (client.table("control_telemetry")
                .select("rtc_timestamp,measured_co2_ppm,duty_cycle,control_state,temp_c,setpoint_ppm")
                .eq("source", source).gte("rtc_timestamp", cutoff)
                .order("rtc_timestamp").limit(5000).execute().data)
        findings, meta = evaluate_rules(rows, params, now, now_local)
        opened, resolved = sync_alerts(client, experiment_id, source, findings, now)
        firing = [f["rule_id"] for f in findings if f["firing"]]
        _log_local(now, source, firing, opened, resolved, meta)
        return firing
    except Exception as e:  # never let the floor die silently — record locally
        _log_local(now, source, [], [], [], {}, error=repr(e))
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="co2_controller_enriched")
    ap.add_argument("--experiment-id", default=os.environ.get("EXPERIMENT_ID", "arabidopsis_co2"))
    ap.add_argument("--loop", action="store_true", help="run continuously instead of single-shot")
    args = ap.parse_args()

    from observations_db import get_client
    client = get_client()
    m = load_manifest()

    if not args.loop:
        run_once(client, m, args.experiment_id, args.source)
        return
    poll = build_params(m)["poll_seconds"]
    while True:
        run_once(client, m, args.experiment_id, args.source)
        time.sleep(poll)


if __name__ == "__main__":
    main()
