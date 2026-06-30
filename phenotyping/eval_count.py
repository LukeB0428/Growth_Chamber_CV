"""
eval_count.py — count + detection-F1 evaluation for the shared engine.

Runs phenotyping.object_count.detect_and_count over a YOLO-format test split and
reports the count metrics (DiC / |DiC| / % agreement / MSE) plus detection
Precision/Recall/F1 at IoU 0.5. Reusable: GWHD cereal spikes now, barley spikes
on the same model later.

Run (Colab, after training a detector):
  python -m phenotyping.eval_count \
    --images gwhd_yolo/images/val --labels gwhd_yolo/labels/val \
    --weights runs/detect/train/weights/best.pt --imgsz 1024 --out results/gwhd_eval.csv
"""
import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phenotyping.object_count import detect_and_count


def read_yolo(txt, W, H):
    """YOLO label file -> list of (x1,y1,x2,y2) pixel boxes."""
    boxes = []
    if not os.path.isfile(txt):
        return boxes
    for line in open(txt):
        p = line.split()
        if len(p) >= 5:
            _, cx, cy, w, h = (float(x) for x in p[:5])
            boxes.append(((cx - w / 2) * W, (cy - h / 2) * H,
                          (cx + w / 2) * W, (cy + h / 2) * H))
    return boxes


def masks_to_boxes(masks):
    out = []
    for m in masks:
        ys, xs = np.where(m)
        if len(xs):
            out.append((float(xs.min()), float(ys.min()),
                        float(xs.max()), float(ys.max())))
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(gt, pred, thr=0.5):
    """Greedy IoU matching -> (tp, fp, fn)."""
    used, tp = set(), 0
    for g in gt:
        best, bj = thr, -1
        for j, p in enumerate(pred):
            if j in used:
                continue
            v = iou(g, p)
            if v >= best:
                best, bj = v, j
        if bj >= 0:
            used.add(bj)
            tp += 1
    return tp, len(pred) - len(used), len(gt) - tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--dedup", action="store_true",
                    help="apply the engine's mask-dedup (off by default — a YOLO "
                         "detector already NMS-dedups; mask-dedup over-merges dense spikes)")
    ap.add_argument("--out", default="results/count_eval.csv")
    args = ap.parse_args()

    imgs = sorted(glob.glob(os.path.join(args.images, "*.*")))
    if not imgs:
        sys.exit(f"No images in {args.images}")

    rows, TP, FP, FN = [], 0, 0, 0
    for ip in imgs:
        img = cv2.imread(ip)
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = read_yolo(os.path.join(args.labels, Path(ip).stem + ".txt"), W, H)
        res = detect_and_count(img, weights=args.weights, conf=args.conf,
                               imgsz=args.imgsz, dedup=args.dedup, min_area_frac=0.0)
        tp, fp, fn = match(gt, masks_to_boxes(res["masks"]), args.iou)
        TP += tp; FP += fp; FN += fn
        dic = res["count"] - len(gt)
        rows.append({"image": Path(ip).name, "gt": len(gt), "pred": res["count"],
                     "dic": dic, "tp": tp, "fp": fp, "fn": fn})
        print(f"{Path(ip).name}: GT={len(gt)} pred={res['count']} DiC={dic:+d}", flush=True)

    if not rows:
        sys.exit("No images/labels matched.")
    dics = np.array([r["dic"] for r in rows], float)
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0

    print("\n==== count + detection metrics ====")
    print(f"images          : {len(rows)}")
    print(f"DiC  (bias)     : {dics.mean():+.3f}")
    print(f"|DiC| (abs err) : {np.abs(dics).mean():.3f}")
    print(f"% agreement     : {100 * np.mean(dics == 0):.1f}%")
    print(f"MSE             : {np.mean(dics ** 2):.3f}")
    print(f"detection P/R/F1@{args.iou} : {P:.3f} / {R:.3f} / {F1:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"per-image -> {args.out}")


if __name__ == "__main__":
    main()
