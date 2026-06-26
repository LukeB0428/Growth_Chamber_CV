"""
validate_komatsuna.py — validate the live-plant (vegetative) phenotyping pipeline
on real Brassica (KOMATSUNA = Brassica rapa, overhead RGB-D, leaf-instance labels).

Proves the existing canopy/greenness/leaf-count pipeline — configured for brassica
via config/species/brassica.json — works on a REAL brassica, against ground truth:
  - Canopy/plant segmentation: our HSV green mask vs GT plant mask  -> IoU / Dice
  - Leaf count:                our leaf counter vs GT leaf count     -> MAE
  - Greenness:                computed + reported (no GT — sanity only)

KOMATSUNA is B. rapa (Diarmuid's is B. napus) — a same-genus PROXY. The vegetative
rosette traits (canopy, leaves, greenness) generalise across brassicas; treat the
numbers as "the pipeline works on a real brassica", not deployment accuracy.

Reads the DatasetNinja tar directly (no extraction needed).

Usage:
    python validate_komatsuna.py --tar C:/Users/LukeB/Downloads/komatsuna-DatasetNinja.tar \
        --subset rgb-d --limit 25 --leaf-method watershed
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import tarfile
import zlib

import cv2
import numpy as np

from config import SPECIES_CONFIG_DIR, RESULTS_DIR

OUT_DIR = os.path.join(str(RESULTS_DIR), "komatsuna_validation")


# ── Supervisely bitmap decode ────────────────────────────────────────────────
def _decode_bitmap(s: str) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(s))
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, 3].astype(bool)
    return img.astype(bool)


def _gt_from_ann(ann: dict) -> tuple[int, np.ndarray]:
    """Return (leaf_count, plant_mask) from a KOMATSUNA Supervisely annotation."""
    H, W = ann["size"]["height"], ann["size"]["width"]
    plant = np.zeros((H, W), bool)
    n = 0
    for o in ann.get("objects", []):
        if o.get("classTitle") != "leaf":
            continue
        bm = o["bitmap"]
        m = _decode_bitmap(bm["data"])
        ox, oy = bm["origin"]
        h, w = m.shape
        plant[oy:oy + h, ox:ox + w] |= m
        n += 1
    return n, plant


# ── Brassica green mask (faithful to the pipeline's segmentation) ─────────────
def _make_green_mask_fn(seg: dict):
    lo = np.array(seg["hsv_lower"], np.uint8)
    hi = np.array(seg["hsv_upper"], np.uint8)
    k = int(seg.get("morph_kernel_size", 7))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def green_mask(bgr):
        m = cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), lo, hi)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, kern)
    return green_mask


def _iou_dice(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    p, g = pred > 0, gt > 0
    inter = int(np.logical_and(p, g).sum())
    pa, ga = int(p.sum()), int(g.sum())
    union = pa + ga - inter
    iou = inter / union if union else 0.0
    dice = 2 * inter / (pa + ga) if (pa + ga) else 0.0
    return iou, dice


def main():
    ap = argparse.ArgumentParser(description="Validate brassica vegetative pipeline on KOMATSUNA")
    ap.add_argument("--tar", default="C:/Users/LukeB/Downloads/komatsuna-DatasetNinja.tar")
    ap.add_argument("--subset", default="rgb-d", choices=["rgb-d", "multi-view"])
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--leaf-method", default="watershed",
                    choices=["watershed", "sam2", "none"])
    ap.add_argument("--previews", type=int, default=5)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(str(SPECIES_CONFIG_DIR), "brassica.json")))
    green_mask = _make_green_mask_fn(cfg["segmentation"])

    # Configure greenness + leaf modules for brassica.
    try:
        from species_config import load_and_apply
        load_and_apply("brassica")
    except Exception as e:
        print(f"(species_config not applied: {e})")
    from greenness_metrics import compute_greenness_metrics
    if a.leaf_method == "watershed":
        from leaf_count import _watershed_count
    elif a.leaf_method == "sam2":
        from leaf_count import count_leaves

    t = tarfile.open(a.tar)
    names = set(t.getnames())
    img_names = sorted(n for n in names
                       if n.startswith(a.subset + "/img/") and n.endswith(".png"))

    # Pre-scan leaf counts, then pick `limit` images SPREAD across the maturity
    # range (seedling -> mature) so the metrics aren't biased to one stage.
    cands = []
    for ip in img_names:
        an = f"{a.subset}/ann/{ip.split('/')[-1]}.json"
        if an not in names:
            continue
        n = sum(1 for o in json.load(t.extractfile(an)).get("objects", [])
                if o.get("classTitle") == "leaf")
        if n > 0:
            cands.append((ip, an, n))
    cands.sort(key=lambda c: c[2])
    if len(cands) > a.limit:
        idx = np.linspace(0, len(cands) - 1, a.limit).round().astype(int)
        cands = [cands[i] for i in idx]
    print(f"selected {len(cands)} images spanning {cands[0][2]}-{cands[-1][2]} leaves")

    os.makedirs(OUT_DIR, exist_ok=True)
    ious, dices, leaf_err = [], [], []
    rows, n_done, n_prev = [], 0, 0

    for ip, ann_name, _ in cands:
        ann = json.load(t.extractfile(ann_name))
        gt_n, gt_mask = _gt_from_ann(ann)
        if gt_n == 0:
            continue
        bgr = cv2.imdecode(np.frombuffer(t.extractfile(ip).read(), np.uint8),
                           cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != gt_mask.shape:
            continue

        gm = green_mask(bgr)
        iou, dice = _iou_dice(gm, gt_mask)
        ious.append(iou); dices.append(dice)

        if a.leaf_method == "watershed":
            pred_leaves = len(_watershed_count(gm))
        elif a.leaf_method == "sam2":
            pred_leaves, _ = count_leaves(gm, bgr, save_vis=False)
        else:
            pred_leaves = None
        if pred_leaves is not None:
            leaf_err.append(abs(pred_leaves - gt_n))

        g = compute_greenness_metrics(gm, bgr)
        rows.append({"image": ip.split("/")[-1], "gt_leaves": gt_n,
                     "pred_leaves": pred_leaves, "iou": round(iou, 3),
                     "dice": round(dice, 3),
                     "greenness_score": g.get("greenness_score")})
        n_done += 1

        if n_prev < a.previews:
            vis = np.hstack([bgr,
                             cv2.cvtColor((gt_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                             cv2.cvtColor(gm, cv2.COLOR_GRAY2BGR)])
            cv2.imwrite(os.path.join(OUT_DIR, f"prev_{n_prev}_{ip.split('/')[-1]}"), vis)
            n_prev += 1

    # ── Report ───────────────────────────────────────────────────────────────
    print(f"\nKOMATSUNA validation ({a.subset}, n={n_done}, leaf={a.leaf_method})")
    print(f"  Canopy segmentation: mean IoU={np.mean(ious):.3f}  Dice={np.mean(dices):.3f}")
    if leaf_err:
        gtm = np.mean([r['gt_leaves'] for r in rows])
        print(f"  Leaf count: MAE={np.mean(leaf_err):.2f}  (GT mean {gtm:.1f} leaves)")
    gs = [r['greenness_score'] for r in rows if r['greenness_score'] is not None]
    if gs:
        print(f"  Greenness score: mean={np.mean(gs):.1f}  range=[{min(gs):.0f},{max(gs):.0f}]")
    import csv
    with open(os.path.join(OUT_DIR, "per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"  per-image + {n_prev} previews -> {OUT_DIR}")


if __name__ == "__main__":
    main()
