"""
aggregate.py — deterministic pre-aggregation for the scheduled agent. NO LLM.

This is the token lever: it collapses thousands of 5s ticks into a compact summary
dict (~30 numbers) that the agent reasons over, instead of feeding it raw rows.
`summarize()` and `decide()` are PURE so they unit-test offline.

`decide()` is also the gate: if it returns 'nominal', the runner skips the LLM
entirely and writes a templated report (zero tokens).
"""
from datetime import datetime, timedelta, timezone


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def _maxabs(xs):
    xs = [abs(x) for x in xs if x is not None]
    return round(max(xs), 1) if xs else None


def _metric_vals(env_rows, name):
    return [_num(r.get("value")) for r in env_rows if r.get("metric_name") == name]


def summarize(ct_rows, env_rows, open_alerts, params, window_hours, now):
    """Pure. ct_rows = enriched control_telemetry; env_rows = control-chamber
    observations (long); open_alerts = current open alert rows. Returns a compact,
    JSON-serializable aggregate."""
    setpoint, tol, duty_max = params["setpoint"], params["tolerance"], params["duty_max"]
    period_s = params["period_s"]

    co2v = [c for c in (_num(r.get("measured_co2_ppm")) for r in ct_rows) if c is not None]
    dosing = [r for r in ct_rows if r.get("control_state") == "dosing"]
    dosing_co2 = [c for c in (_num(r.get("measured_co2_ppm")) for r in dosing) if (c or 0) > 0]
    in_band = [c for c in dosing_co2 if abs(c - setpoint) <= tol]
    dutyv = [d for d in (_num(r.get("duty_cycle")) for r in ct_rows) if d is not None]
    pinned = [d for d in dutyv if d >= duty_max - 1]
    err = [e for e in (_num(r.get("error_ppm")) for r in ct_rows) if e is not None]

    expected = max(1, int(window_hours * 3600 / period_s))
    enr = {
        "co2_mean": _mean(co2v), "co2_min": round(min(co2v), 1) if co2v else None,
        "co2_max": round(max(co2v), 1) if co2v else None,
        "pct_in_band": round(len(in_band) / len(dosing_co2), 3) if dosing_co2 else None,
        "pct_time_dosing": round(len(dosing) / len(ct_rows), 3) if ct_rows else None,
        "duty_mean": _mean(dutyv),
        "pct_duty_pinned": round(len(pinned) / len(dutyv), 3) if dutyv else None,
        "abs_error_mean": _mean([abs(e) for e in err]),
        "temp_mean": _mean([_num(r.get("temp_c")) for r in ct_rows]),
        "humidity_mean": _mean([_num(r.get("humidity_pct")) for r in ct_rows]),
        "rtc_offset_max_abs": _maxabs([_num(r.get("rtc_offset_sec")) for r in ct_rows]),
    }
    ctrl = {
        "co2_mean": _mean(_metric_vals(env_rows, "measured_co2_ppm")),
        "temp_mean": _mean(_metric_vals(env_rows, "temp_c")),
        "humidity_mean": _mean(_metric_vals(env_rows, "humidity_pct")),
        "rtc_offset_max_abs": _maxabs(_metric_vals(env_rows, "rtc_offset_sec")),
    }
    return {
        "window_hours": window_hours,
        "generated_at": now.isoformat(),
        "setpoint": setpoint, "tolerance": tol,
        "coverage": {
            "enriched_ticks": len(ct_rows), "expected": expected,
            "frac": round(len(ct_rows) / expected, 3),
            "control_present": bool(env_rows),
        },
        "enriched": enr,
        "control": ctrl,
        "open_alerts": [
            {"rule_id": a.get("rule_id"), "severity": a.get("severity"),
             "opened_at": a.get("opened_at"), "detail": a.get("detail")}
            for a in open_alerts
        ],
    }


def decide(agg):
    """Gate + status. Returns (status, flags). 'nominal' => runner skips the LLM."""
    alerts = agg["open_alerts"]
    flags = []
    if agg["coverage"]["frac"] < 0.5:
        flags.append("low_coverage")
    if not agg["coverage"]["control_present"]:
        flags.append("control_chamber_missing")
    pib = agg["enriched"]["pct_in_band"]
    if pib is not None and pib < 0.8:
        flags.append("co2_out_of_band")
    for ch in ("enriched", "control"):
        off = agg[ch]["rtc_offset_max_abs"]
        if off is not None and off > 120:
            flags.append(f"{ch}_rtc_drift")

    if any(a.get("severity") == "high" for a in alerts):
        return "alert", flags
    if alerts or flags:
        return "attention", flags
    return "nominal", flags


def fetch_inputs(client, params, window_hours, experiment_id,
                 controller_source="co2_controller_enriched",
                 env_source="env_logger_control", cap=20000):
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(hours=window_hours)).isoformat()
    ct = (client.table("control_telemetry")
          .select("rtc_timestamp,measured_co2_ppm,duty_cycle,control_state,error_ppm,"
                  "temp_c,humidity_pct,rtc_offset_sec")
          .eq("source", controller_source).gte("rtc_timestamp", cutoff)
          .order("rtc_timestamp", desc=True).limit(cap).execute().data)
    env = (client.table("observations")
           .select("metric_name,value")
           .eq("source", env_source).gte("timestamp", cutoff)
           .in_("metric_name", ["measured_co2_ppm", "temp_c", "humidity_pct", "rtc_offset_sec"])
           .limit(cap).execute().data)
    alerts = (client.table("alerts").select("rule_id,severity,opened_at,detail")
              .eq("experiment_id", experiment_id).eq("status", "open").execute().data)
    return ct, env, alerts, now


def build_aggregate(client, params, window_hours, experiment_id, **kw):
    ct, env, alerts, now = fetch_inputs(client, params, window_hours, experiment_id, **kw)
    return summarize(ct, env, alerts, params, window_hours, now)
