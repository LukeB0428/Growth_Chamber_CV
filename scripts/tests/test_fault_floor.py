"""
Tests for fault_floor.evaluate_rules — the deterministic never-miss safety net.

Focus: the feed_silent (data-freshness) rule and rule precedence. evaluate_rules
is a pure function, so these run offline with no DB / no Supabase.

Runs two ways:
    scripts/.venv/Scripts/python scripts/tests/test_fault_floor.py   # standalone
    pytest scripts/tests/test_fault_floor.py                          # under CI
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fault_floor import evaluate_rules

# A minimal params dict (mirrors build_params output) so tests don't need a manifest.
PARAMS = {
    "setpoint": 1000.0, "tolerance": 100.0, "duty_max": 100.0,
    "sustain_min": 15.0, "sensor_sentinel": -7000.0, "fire_frac": 0.9,
    "min_coverage": 0.6, "period_s": 5.0, "lookback_min": 30.0,
    "stale_after_min": 10.0,
}

# Fixed reference instants (no Date.now(); tests must be deterministic).
NOW_UTC = datetime(2026, 7, 2, 13, 30, 0, tzinfo=timezone.utc)
# Host is Irish summer time (UTC+1): local wall clock is one hour ahead of UTC,
# and rtc_timestamp is stored in that local basis tagged +00:00.
NOW_LOCAL = datetime(2026, 7, 2, 14, 30, 0)


def _row(local_dt, co2=1000.0, duty=50.0, state="dosing", temp=22.0):
    """One control_telemetry row. local_dt is host-local wall clock; it is
    serialised tagged +00:00 exactly as observations_db._rtc_fields does."""
    return {
        "rtc_timestamp": local_dt.replace(tzinfo=timezone.utc).isoformat(),
        "measured_co2_ppm": co2, "duty_cycle": duty,
        "control_state": state, "temp_c": temp, "setpoint_ppm": 1000.0,
    }


def _find(findings, rule_id):
    return next(f for f in findings if f["rule_id"] == rule_id)


def test_feed_silent_fires_on_no_rows():
    findings, meta = evaluate_rules([], PARAMS, NOW_UTC, NOW_LOCAL)
    assert _find(findings, "feed_silent")["firing"] is True
    assert meta["latest_age_min"] is None
    # Nothing else should fire when there's no data.
    assert all(not f["firing"] for f in findings if f["rule_id"] != "feed_silent")


def test_feed_silent_fires_on_stale_feed():
    # Newest tick is 20 min old in the LOCAL basis (> stale_after_min=10).
    rows = [_row(NOW_LOCAL - timedelta(minutes=25)),
            _row(NOW_LOCAL - timedelta(minutes=20))]
    findings, meta = evaluate_rules(rows, PARAMS, NOW_UTC, NOW_LOCAL)
    assert _find(findings, "feed_silent")["firing"] is True
    assert meta["latest_age_min"] == 20.0


def test_fresh_feed_does_not_fire_silent():
    # A tick 1 min old must be treated as fresh. This only works if age is
    # measured in the local basis: against NOW_UTC the same tick looks ~59 min
    # in the FUTURE, so a UTC-based check would misbehave.
    rows = [_row(NOW_LOCAL - timedelta(minutes=1))]
    findings, _ = evaluate_rules(rows, PARAMS, NOW_UTC, NOW_LOCAL)
    assert _find(findings, "feed_silent")["firing"] is False


def test_feed_silent_takes_precedence():
    # Feed is stale AND the last ticks show a sensor dropout (co2<=0). Only
    # feed_silent should survive — one root cause, one alert.
    rows = [_row(NOW_LOCAL - timedelta(minutes=22), co2=-7999.0),
            _row(NOW_LOCAL - timedelta(minutes=21), co2=-7999.0),
            _row(NOW_LOCAL - timedelta(minutes=20), co2=-7999.0)]
    findings, _ = evaluate_rules(rows, PARAMS, NOW_UTC, NOW_LOCAL)
    assert _find(findings, "feed_silent")["firing"] is True
    assert _find(findings, "sensor_dropout")["firing"] is False
    assert "suppressed: feed silent" in _find(findings, "sensor_dropout")["detail"]


def test_sensor_dropout_still_fires_when_feed_fresh():
    # Fresh feed but the sensor is returning the failure sentinel → dropout fires.
    rows = [_row(NOW_LOCAL - timedelta(minutes=2), co2=-7999.0),
            _row(NOW_LOCAL - timedelta(minutes=1), co2=-7999.0),
            _row(NOW_LOCAL, co2=-7999.0)]
    findings, _ = evaluate_rules(rows, PARAMS, NOW_UTC, NOW_LOCAL)
    assert _find(findings, "feed_silent")["firing"] is False
    assert _find(findings, "sensor_dropout")["firing"] is True


def _run_standalone():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_standalone() else 0)
