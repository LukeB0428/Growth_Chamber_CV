"""
test_feasibility_probe.py — tests for the occlusion-probe decision logic.

    python -m brassica_pods.test_feasibility_probe
"""

from __future__ import annotations

from brassica_pods import feasibility_probe as fp


def _close(a, b, tol=1e-2):
    return abs(a - b) <= tol


def test_verdict_boundaries():
    assert fp.verdict(0.10, 0.50) == "GO"        # exactly on GO thresholds
    assert fp.verdict(0.05, 0.70) == "GO"
    assert fp.verdict(0.12, 0.60) == "MARGINAL"  # CV in 0.10–0.20
    assert fp.verdict(0.20, 0.60) == "MARGINAL"  # CV == 0.20 not yet NO-GO
    assert fp.verdict(0.21, 0.60) == "NO-GO"     # CV > 0.20
    assert fp.verdict(0.05, 0.35) == "NO-GO"     # f_mean < 0.40
    assert fp.verdict(0.08, 0.45) == "MARGINAL"  # good CV but f_mean 0.40–0.50


def test_consistent_fraction_is_go():
    # f == 0.7 for all plants -> ~zero CV -> GO
    recs = [{"plant_id": f"p{i}", "visible": 0.70 * (100 + i * 5), "true": 100 + i * 5}
            for i in range(8)]
    s = fp.analyze_probe(recs)
    assert s["verdict"] == "GO"
    assert _close(s["f_mean"], 0.70)
    assert s["cv_f"] < 0.02


def test_variable_fraction_is_nogo():
    # f swings 0.4–0.9 -> high CV -> NO-GO
    fs = [0.4, 0.9, 0.5, 0.85, 0.45, 0.8]
    recs = [{"plant_id": f"p{i}", "visible": f * 100, "true": 100} for i, f in enumerate(fs)]
    s = fp.analyze_probe(recs)
    assert s["verdict"] == "NO-GO"
    assert s["cv_f"] > 0.20


def test_optimistic_mape_equals_cv():
    fs = [0.6, 0.66, 0.72, 0.69]
    recs = [{"plant_id": f"p{i}", "visible": f * 100, "true": 100} for i, f in enumerate(fs)]
    s = fp.analyze_probe(recs)
    assert _close(s["optimistic_mape_pct"], s["cv_f"] * 100)


def test_low_fmean_is_nogo_even_if_consistent():
    # consistently only 30% visible -> NO-GO regardless of low CV
    recs = [{"plant_id": f"p{i}", "visible": 0.30 * (100 + i), "true": 100 + i} for i in range(6)]
    s = fp.analyze_probe(recs)
    assert s["verdict"] == "NO-GO" and s["f_mean"] < 0.40


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1; print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    import sys
    sys.exit(1 if _main() else 0)
