"""
evaluate.py — count + size accuracy evaluation (hardened, v2 §B).

Reports, ALWAYS split DENSE vs SPARSE (never hide the hard case in an average):
  - Count: MAE, % error (MAPE), bias, RMSE, R² vs manual count.
  - Mask quality: mask mAP on the SYNTHETIC val set (the real images have no
    masks — only count/size GT — so AP is measured where masks exist).
  - Size: predicted vs GT per-image mean pod length (needs a px→mm scale).
  - Per-image predictions + errors logged so active learning can target the
    worst cases.

The metric functions are pure + unit-tested now (test_evaluate.py); the model
calls (analyze / ultralytics val) are lazy so this imports without weights.

TASSELNET GATE (set 2026-06-18, decided before measuring — do not revise after):
read the instance model's count %error on the DENSE subset only:
  ≤10%  → skip TasselNet (instance seg sufficient)
  >15%  → build the TasselNet count-regression head
  10–15% → optional (only if count-only on dense images, size not needed)

CLI (once weights exist):
    python -m brassica_pods.eval.evaluate \
        --weights brassica_pods/weights/pods_best.pt \
        --images <real_images_dir> \
        --gt brassica_pods/data/eval_ground_truth.csv \
        --out brassica_pods/data/eval --dense-threshold 25
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# Default: an image is "dense" if its GT pod count ≥ this.
# ⚠️ LOAD-BEARING PROXY: count is NOT density. 30 well-spaced pods are not
# "dense/occluded"; 20 tightly clustered ones are — and this rule buckets them
# the WRONG way round. The TasselNet gate rides entirely on the dense-subset
# %err, so before trusting the gate on real images, EYEBALL whether the ≥25
# bucket actually corresponds to visually crowded shots and refine the threshold
# (or move to an overlap/area criterion) if not. Fine as a v1 default only.
DEFAULT_DENSE_THRESHOLD = 25


# ─────────────────────────────────────────────────────────────────────────────
# Pure metric functions (unit-tested without a model)
# ─────────────────────────────────────────────────────────────────────────────
def count_metrics(pred_counts, true_counts) -> dict:
    """MAE, MAPE %, bias, RMSE, R² between predicted and manual pod counts."""
    pred = np.asarray(pred_counts, dtype=float)
    true = np.asarray(true_counts, dtype=float)
    if pred.shape != true.shape or pred.size == 0:
        raise ValueError("pred_counts and true_counts must be equal, non-empty")
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    nz = true != 0
    mape = float(np.mean(np.abs(err[nz]) / true[nz]) * 100) if nz.any() else float("nan")
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    return {"n": int(pred.size), "mae": round(mae, 3), "mape_pct": round(mape, 2),
            "bias": round(bias, 3), "rmse": round(rmse, 3), "r2": round(r2, 4)}


# Back-compat alias (older name).
def evaluate_counts(pred_counts, true_counts) -> dict:
    m = count_metrics(pred_counts, true_counts)
    return {"mae": m["mae"], "mape_pct": m["mape_pct"], "n": m["n"]}


def split_dense_sparse(records, dense_threshold=DEFAULT_DENSE_THRESHOLD,
                       key="gt_count"):
    """Partition records into (dense, sparse) by GT count threshold."""
    dense = [r for r in records if r[key] >= dense_threshold]
    sparse = [r for r in records if r[key] < dense_threshold]
    return dense, sparse


def tasselnet_gate(dense_mape_pct: float) -> dict:
    """Apply the locked gate to the DENSE-subset % error."""
    if dense_mape_pct <= 10:
        decision, why = "skip", "instance seg sufficient on dense subset"
    elif dense_mape_pct > 15:
        decision, why = "build", "dense-subset error too high; add count-regression head"
    else:
        decision, why = "optional", "build only if count-only on dense images, size not needed"
    return {"dense_mape_pct": round(dense_mape_pct, 2), "decision": decision, "why": why}


def count_report(records, dense_threshold=DEFAULT_DENSE_THRESHOLD) -> dict:
    """Overall + dense + sparse count metrics, plus the TasselNet gate read."""
    dense, sparse = split_dense_sparse(records, dense_threshold)
    out = {"dense_threshold": dense_threshold,
           "overall": count_metrics([r["pred_count"] for r in records],
                                    [r["gt_count"] for r in records])}
    out["dense"] = (count_metrics([r["pred_count"] for r in dense],
                                  [r["gt_count"] for r in dense]) if dense
                    else {"n": 0})
    out["sparse"] = (count_metrics([r["pred_count"] for r in sparse],
                                   [r["gt_count"] for r in sparse]) if sparse
                     else {"n": 0})
    if dense:
        out["tasselnet_gate"] = tasselnet_gate(out["dense"]["mape_pct"])
    else:
        out["tasselnet_gate"] = {"decision": "unknown",
                                 "why": "no dense images in this set"}
    return out


def size_metrics(pred_mm, true_mm) -> dict:
    """Per-image mean pod length agreement (mm). Drops pairs missing a value."""
    pairs = [(p, t) for p, t in zip(pred_mm, true_mm)
             if p is not None and t is not None]
    if not pairs:
        return {"n": 0, "note": "no paired sizes (px→mm scale required)"}
    p = np.array([a for a, _ in pairs]); t = np.array([b for _, b in pairs])
    return {"n": len(pairs), "mae_mm": round(float(np.mean(np.abs(p - t))), 2),
            "bias_mm": round(float(np.mean(p - t)), 2)}


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration (lazy model imports — only needed when weights exist)
# ─────────────────────────────────────────────────────────────────────────────
def _load_gt(gt_csv) -> dict:
    rows = {}
    with open(gt_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["image"]] = {
                "gt_count": int(r["gt_count"]),
                "gt_len_mm": float(r["mean_pod_length_mm"]) if r.get("mean_pod_length_mm") else None,
            }
    return rows


def run_count_eval(weights, images_dir, gt_csv, out_dir,
                   dense_threshold=DEFAULT_DENSE_THRESHOLD, scale=None,
                   conf=0.25) -> dict:
    """Run the model over the real images, build per-image records, write
    per-image CSV + summary JSON + scatter PNG, return the count report."""
    from brassica_pods.infer.analyze import analyze   # lazy (needs weights)

    gt = _load_gt(gt_csv)
    images_dir = Path(images_dir)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, g in gt.items():
        ip = images_dir / name
        if not ip.is_file():
            continue
        res = analyze(str(ip), scale=scale, weights=weights, conf=conf)
        lens = [pp.get("length_mm") for pp in res["per_pod"] if pp.get("length_mm")]
        records.append({
            "image": name, "gt_count": g["gt_count"], "pred_count": res["count"],
            "abs_err": abs(res["count"] - g["gt_count"]),
            "gt_len_mm": g["gt_len_mm"],
            "pred_mean_len_mm": round(float(np.mean(lens)), 2) if lens else None,
        })
    if not records:
        raise RuntimeError("No images matched between GT and the images dir.")

    # Per-image log (worst-first for active-learning targeting).
    records.sort(key=lambda r: r["abs_err"], reverse=True)
    with open(out_dir / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader(); w.writerows(records)

    report = count_report(records, dense_threshold)
    report["size"] = size_metrics([r["pred_mean_len_mm"] for r in records],
                                  [r["gt_len_mm"] for r in records])
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))
    _scatter(records, out_dir / "count_scatter.png", dense_threshold)
    return report


def _scatter(records, path, dense_threshold):
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except Exception:
        return
    gt = np.array([r["gt_count"] for r in records])
    pr = np.array([r["pred_count"] for r in records])
    dense = gt >= dense_threshold
    plt.figure(figsize=(5, 5))
    plt.scatter(gt[~dense], pr[~dense], s=14, alpha=0.6, label="sparse")
    plt.scatter(gt[dense], pr[dense], s=14, alpha=0.6, label="dense", color="crimson")
    lim = [0, max(int(pr.max()), int(gt.max())) + 2]
    plt.plot(lim, lim, "k--", lw=1); plt.xlim(lim); plt.ylim(lim)
    plt.xlabel("manual count (GT)"); plt.ylabel("predicted count")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()


def mask_map(weights, data_yaml, imgsz=1536) -> dict:
    """Mask mAP on the synthetic val set (the only set with GT masks)."""
    from ultralytics import YOLO   # lazy
    m = YOLO(str(weights)).val(data=str(data_yaml), imgsz=imgsz, verbose=False)
    seg = m.seg
    return {"mask_mAP50": round(float(seg.map50), 4),
            "mask_mAP50_95": round(float(seg.map), 4)}


def _print_report(report: dict):
    print(json.dumps(report, indent=2))
    g = report.get("tasselnet_gate", {})
    print(f"\nTASSELNET GATE → {g.get('decision', '?').upper()} "
          f"(dense %err={report.get('dense', {}).get('mape_pct', '?')}): {g.get('why', '')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Hardened count/size/mask evaluation")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default="brassica_pods/data/eval")
    ap.add_argument("--dense-threshold", type=int, default=DEFAULT_DENSE_THRESHOLD)
    ap.add_argument("--data-yaml", default=None, help="pods.yaml for mask mAP")
    a = ap.parse_args()

    report = run_count_eval(a.weights, a.images, a.gt, a.out, a.dense_threshold)
    if a.data_yaml:
        report["mask"] = mask_map(a.weights, a.data_yaml)
        Path(a.out, "summary.json").write_text(json.dumps(report, indent=2))
    _print_report(report)
