"""
Tests for statistical_comparison — the pot-level aggregation, Cliff's delta, and
Benjamini-Hochberg correction that replaced the pseudoreplicated per-pot-day
Mann-Whitney + Cohen's d.

Runs standalone or under pytest (see test_fault_floor.py header).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from statistical_comparison import (aggregate_per_pot, cliffs_delta,
                                     cliffs_magnitude, apply_fdr, run_stats,
                                     MIN_POTS)


def _df(n_pots=8, days=10):
    """Synthetic pot-day frame: enriched pots run higher than control pots."""
    rows = []
    for chamber, base in [("enriched", 1.0), ("control", 0.0)]:
        for p in range(n_pots):
            for d in range(days):
                rows.append({
                    "chamber": chamber,
                    "pot_label": f"{chamber}_{p}",
                    "date": pd.Timestamp("2026-04-09") + pd.Timedelta(days=d),
                    "metric": base + p * 0.01 + d * 0.001,
                })
    return pd.DataFrame(rows)


def test_aggregate_collapses_to_one_value_per_pot():
    df = _df(n_pots=8, days=10)
    e, c = aggregate_per_pot(df, "metric")
    # 8 pots per chamber, regardless of the 10 days each — this is the fix.
    assert len(e) == 8 and len(c) == 8


def test_cliffs_delta_extremes_and_ties():
    assert cliffs_delta([4, 5, 6], [1, 2, 3]) == 1.0    # a all greater
    assert cliffs_delta([1, 2, 3], [4, 5, 6]) == -1.0   # a all smaller
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0    # symmetric
    assert cliffs_magnitude(1.0) == "large"
    assert cliffs_magnitude(0.05) == "negligible"


def test_apply_fdr_adjusts_and_flags():
    results = [
        {"metric": "a", "p_value": 0.001},
        {"metric": "b", "p_value": 0.04},
        {"metric": "c", "p_value": 0.8},
    ]
    for r in results:  # apply_fdr expects these keys present
        r.setdefault("p_value_adj", np.nan)
        r.setdefault("significant", False)
    apply_fdr(results, alpha=0.05)
    # Adjusted p >= raw p, and BH is monotone non-decreasing in rank order.
    assert results[0]["p_value_adj"] >= 0.001
    assert results[2]["p_value_adj"] >= results[1]["p_value_adj"]
    assert results[0]["significant"] is True
    assert results[2]["significant"] is False


def test_apply_fdr_controls_family_error():
    # A family where 0.03 and 0.04 are individually < 0.05 but sit among larger
    # p-values: BH must inflate them above 0.05 so only the smallest survives.
    pvals = [0.001, 0.03, 0.04, 0.06, 0.2, 0.5, 0.7, 0.9]
    results = [{"metric": str(i), "p_value": p, "p_value_adj": np.nan,
                "significant": False} for i, p in enumerate(pvals)]
    apply_fdr(results, alpha=0.05)
    assert results[0]["significant"] is True, "p=0.001 should survive FDR"
    assert results[1]["significant"] is False, "raw-sig p=0.03 must not survive FDR here"
    assert results[2]["significant"] is False, "raw-sig p=0.04 must not survive FDR here"


def test_run_stats_requires_min_pots():
    # Fewer than MIN_POTS pots per chamber → no test run (guards underpowered n).
    df = _df(n_pots=MIN_POTS - 1, days=10)
    r = run_stats(df, "metric")
    assert r["effect_size"] == "insufficient data"
    assert np.isnan(r["p_value"])


def test_run_stats_detects_separation():
    df = _df(n_pots=8, days=10)  # enriched cleanly above control
    r = run_stats(df, "metric")
    apply_fdr([r])
    assert r["n_enriched"] == 8 and r["n_control"] == 8
    assert r["cliffs_delta"] == 1.0


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
