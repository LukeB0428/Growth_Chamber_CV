"""
eval_cvppp.py — benchmark the leaf counter against the CVPPP Leaf Segmentation set.

Reports the STANDARD CVPPP metrics so you get a citable number for the methods
paper instead of a self-graded MAE:
  - DiC          mean signed Difference in Count (shows over/under-counting bias)
  - |DiC|        mean absolute count error (the headline counting metric)
  - % agreement  fraction of images with an exact count match
  - MSE          mean squared count error
  - SBD          Symmetric Best Dice (instance-segmentation quality; --sbd)

CVPPP layout (per subset folder, e.g. A1 = Arabidopsis, the closest to your setup):
    plantNNN_rgb.png     RGB image
    plantNNN_label.png   per-leaf instance label (each leaf a unique colour, bg black)
    plantNNN_fg.png      foreground mask (optional; use with --use-fg)

Get the data: register at the Plant Phenotyping Datasets / CVPPP LSC site and
download the training set. A1 is the Arabidopsis subset.

Run (GPU strongly recommended — Colab):
    pip install sam2 torch
    python scripts/eval_cvppp.py --cvppp /path/to/A1 --device cuda --sbd
    # Isolate the COUNTING algorithm from your HSV segmentation's domain gap:
    python scripts/eval_cvppp.py --cvppp /path/to/A1 --device cuda --use-fg

Report both: --use-fg answers "how good is the counter?", default answers
"how good is the end-to-end pipeline on out-of-domain images?".
"""
import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import leaf_count as lc


def gt_instances(label_img):
    """Return (count, [boolean masks]) from a CVPPP label image (unique colour per leaf)."""
    flat = label_img.reshape(-1, label_img.shape[-1])
    colours = set(map(tuple, flat))
    colours.discard((0, 0, 0))
    masks = [np.all(label_img == np.array(c, label_img.dtype), axis=-1) for c in colours]
    return len(masks), masks


def dice(a, b):
    s = a.sum() + b.sum()
    return (2.0 * np.logical_and(a, b).sum() / s) if s > 0 else 0.0


def _best_dice(A, B):
    return sum(max((dice(a, b) for b in B), default=0.0) for a in A) / len(A) if A else 0.0


def sbd(pred, gt):
    """Symmetric Best Dice (CVPPP definition): min of the two best-match directions."""
    if not pred or not gt:
        return 0.0
    return min(_best_dice(pred, gt), _best_dice(gt, pred))


def predict_masks(image, mask, method):
    if method == "watershed":
        return lc._watershed_count(mask)
    return lc._try_sam2(image, mask)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cvppp", required=True, help="path to a CVPPP subset folder (e.g. A1)")
    ap.add_argument("--method", default="sam2", choices=["sam2", "watershed"])
    ap.add_argument("--device", default="cpu", help="cpu | cuda (Colab GPU)")
    ap.add_argument("--use-fg", action="store_true",
                    help="use CVPPP's *_fg.png as the mask (isolates counting from your HSV segmentation)")
    ap.add_argument("--sbd", action="store_true", help="also compute Symmetric Best Dice (slower)")
    ap.add_argument("--limit", type=int, default=0, help="cap number of images (0 = all)")
    ap.add_argument("--out", default="results/cvppp_eval.csv")
    args = ap.parse_args()

    lc.SAM2_DEVICE = args.device

    rgb_files = sorted(glob.glob(os.path.join(args.cvppp, "*_rgb.png")))
    if args.limit:
        rgb_files = rgb_files[:args.limit]
    if not rgb_files:
        sys.exit(f"No *_rgb.png in {args.cvppp}")

    rows = []
    for i, rgb_path in enumerate(rgb_files, 1):
        label_path = rgb_path.replace("_rgb.png", "_label.png")
        if not os.path.isfile(label_path):
            continue
        image = cv2.imread(rgb_path)
        label = cv2.imread(label_path)
        gt_count, gt_masks = gt_instances(label)

        if args.use_fg:
            fg = cv2.imread(rgb_path.replace("_rgb.png", "_fg.png"), cv2.IMREAD_GRAYSCALE)
            mask = ((fg > 0).astype(np.uint8) * 255) if fg is not None else lc._green_mask_hsv(image)
        else:
            mask = lc._green_mask_hsv(image)

        try:
            pred_masks = predict_masks(image, mask, args.method)
        except Exception as e:
            print(f"  {os.path.basename(rgb_path)}: predict error {e}")
            pred_masks = []

        dic = len(pred_masks) - gt_count
        row = {"image": os.path.basename(rgb_path), "gt": gt_count, "pred": len(pred_masks), "dic": dic}
        if args.sbd:
            row["sbd"] = round(sbd(pred_masks, gt_masks), 4)
        rows.append(row)
        msg = f"[{i}/{len(rgb_files)}] {row['image']}: GT={gt_count} pred={len(pred_masks)} DiC={dic:+d}"
        print(msg + (f" SBD={row['sbd']:.3f}" if args.sbd else ""), flush=True)

    if not rows:
        sys.exit("No labelled images evaluated.")

    dics = np.array([r["dic"] for r in rows], float)
    print("\n==== CVPPP metrics ====")
    print(f"images          : {len(dics)}")
    print(f"DiC  (bias)     : {dics.mean():+.3f}  (std {dics.std():.3f})")
    print(f"|DiC| (abs err) : {np.abs(dics).mean():.3f}  (std {np.abs(dics).std():.3f})")
    print(f"% agreement     : {100.0 * np.mean(dics == 0):.1f}%")
    print(f"MSE             : {np.mean(dics ** 2):.3f}")
    if args.sbd:
        sbds = np.array([r["sbd"] for r in rows], float)
        print(f"SBD (seg)       : {sbds.mean():.3f}  (std {sbds.std():.3f})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nper-image results -> {args.out}")


if __name__ == "__main__":
    main()
