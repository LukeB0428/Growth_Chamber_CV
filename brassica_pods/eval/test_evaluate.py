"""
test_evaluate.py — unit tests for the eval-hardening metric functions.

Pure-metric tests (no model / no weights):
    python -m brassica_pods.eval.test_evaluate
Proves count metrics, dense/sparse split, the TasselNet gate boundaries, the
count report, and size metrics — so the harness is correct before weights land.
"""

from __future__ import annotations

from brassica_pods.eval import evaluate as ev


def _close(a, b, tol=1e-2):
    return abs(a - b) <= tol


def test_count_metrics_known_values():
    m = ev.count_metrics([10, 20, 30], [12, 18, 33])
    assert m["n"] == 3
    assert _close(m["mae"], 2.333)
    assert _close(m["mape_pct"], 12.29)
    assert _close(m["bias"], -1.0)
    assert _close(m["rmse"], 2.380)
    assert _close(m["r2"], 0.9273, 1e-3)


def test_count_metrics_perfect():
    m = ev.count_metrics([5, 6, 7], [5, 6, 7])
    assert m["mae"] == 0 and m["mape_pct"] == 0 and m["r2"] == 1.0


def test_split_dense_sparse():
    recs = [{"gt_count": 5}, {"gt_count": 25}, {"gt_count": 40}, {"gt_count": 24}]
    dense, sparse = ev.split_dense_sparse(recs, dense_threshold=25)
    assert len(dense) == 2 and len(sparse) == 2


def test_tasselnet_gate_boundaries():
    assert ev.tasselnet_gate(10.0)["decision"] == "skip"     # ≤10
    assert ev.tasselnet_gate(9.9)["decision"] == "skip"
    assert ev.tasselnet_gate(12.0)["decision"] == "optional"  # 10–15
    assert ev.tasselnet_gate(15.0)["decision"] == "optional"  # not >15
    assert ev.tasselnet_gate(15.01)["decision"] == "build"    # >15
    assert ev.tasselnet_gate(40.0)["decision"] == "build"


def test_count_report_splits_and_gates():
    # sparse predicted well; dense predicted poorly (~25% off) -> gate "build"
    recs = [
        {"pred_count": 5, "gt_count": 5}, {"pred_count": 11, "gt_count": 10},
        {"pred_count": 22, "gt_count": 30}, {"pred_count": 21, "gt_count": 28},
    ]
    rep = ev.count_report(recs, dense_threshold=25)
    assert rep["overall"]["n"] == 4
    assert rep["dense"]["n"] == 2 and rep["sparse"]["n"] == 2
    assert rep["tasselnet_gate"]["decision"] == "build"
    # gate must read the DENSE subset, not the overall
    assert rep["tasselnet_gate"]["dense_mape_pct"] == rep["dense"]["mape_pct"]


def test_count_report_no_dense():
    recs = [{"pred_count": 5, "gt_count": 5}, {"pred_count": 9, "gt_count": 10}]
    rep = ev.count_report(recs, dense_threshold=25)
    assert rep["dense"]["n"] == 0
    assert rep["tasselnet_gate"]["decision"] == "unknown"


def test_size_metrics_handles_missing():
    m = ev.size_metrics([60.0, None, 55.0], [58.0, 50.0, None])
    assert m["n"] == 1 and _close(m["mae_mm"], 2.0)
    assert ev.size_metrics([None], [None])["n"] == 0


def test_evaluate_counts_backcompat():
    m = ev.evaluate_counts([1, 2, 3], [1, 2, 4])
    assert set(m) == {"mae", "mape_pct", "n"} and m["n"] == 3


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
