"""
leaf_count.py — Leaf counting and germination detection
EE496 | Luke Buckley | Maynooth University

Two segmentation approaches in priority order:
  1. SAM2 zero-shot instance segmentation (sam2.1_hiera_tiny)
     — best accuracy, no training data required
     — ~30-60s per image on CPU (acceptable for daily pipeline)
     — install: pip install sam2
     — model checkpoint auto-downloaded on first run to scripts/sam2_weights/
  2. Watershed fallback
     — used automatically if SAM2 is not installed or raises an error
     — faster but less accurate on overlapping leaves (MAE ~7.4 on CVPPP)

Usage:
    python leaf_count.py --image path/to/image.jpg --chamber enriched
    python leaf_count.py --validate --cvppp path/to/A1
"""

import cv2
import numpy as np
import os
import json
import argparse
import csv
from datetime import datetime
from scipy import ndimage
from scipy.ndimage import label as scipy_label
from config import RESULTS_DIR, LEAF_VIS_DIR, METRICS_CSV, SAM2_WEIGHTS_DIR

# ── Constants ────────────────────────────────────────────────────────────────
RESULTS_DIR  = str(RESULTS_DIR)
VIS_DIR      = str(LEAF_VIS_DIR)
METRICS_CSV  = str(METRICS_CSV)
SAM2_WEIGHTS = str(SAM2_WEIGHTS_DIR)

# SAM2 leaf filtering thresholds
MIN_LEAF_AREA_PX   = 300    # ignore masks smaller than this (noise)
MAX_LEAF_AREA_FRAC = 0.40   # ignore masks covering >40% of image (background leakage)
MIN_GREEN_FRAC     = 0.35   # mask must be ≥35% green pixels to count as a leaf

SAM2_DEVICE = "cpu"         # override to "cuda" for GPU (e.g. eval_cvppp.py on Colab)

# Watershed constants (fallback)
WS_DIST_THRESHOLD  = 0.4
WS_MIN_REGION_SIZE = 500

# Colour palette for visualisation (up to 20 leaves)
LEAF_COLOURS = [
    (220, 60,  60),  (60,  180, 75),  (60,  130, 200), (245, 130, 48),
    (145, 30,  180), (70,  240, 240), (240, 50,  230), (210, 245, 60),
    (250, 190, 212), (0,   128, 128), (220, 190, 255), (170, 110, 40),
    (255, 250, 200), (128, 0,   0),   (170, 255, 195), (128, 128, 0),
    (255, 216, 177), (0,   0,   117), (128, 128, 128), (255, 255, 255),
]


# ── Green mask helper ─────────────────────────────────────────────────────────
def _green_mask_hsv(image):
    hsv   = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, np.array([25, 40, 40]), np.array([90, 255, 255]))
    k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)


def _green_fraction(image, mask_bool):
    """Fraction of pixels inside a SAM2 mask that are green."""
    green = _green_mask_hsv(image)
    region_pixels = np.sum(mask_bool)
    if region_pixels == 0:
        return 0.0
    return np.sum(green[mask_bool] > 0) / region_pixels


# ─────────────────────────────────────────────────────────────────────────────
# SAM2 approach — point-prompted
# ─────────────────────────────────────────────────────────────────────────────
def _find_leaf_prompts(green_mask):
    """
    Find one foreground point per leaf using distance-transform local maxima.
    The peak of the distance transform inside each blob is the most interior
    point — an ideal SAM2 prompt for that leaf.
    Returns list of (x, y) pixel coordinates.
    """
    from scipy.ndimage import maximum_filter
    import scipy.ndimage as ndi

    binary = (green_mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return []

    # Neighbourhood size scales with typical leaf radius
    neigh  = max(15, int(dist.max() * 0.5))
    loc_max = maximum_filter(dist, size=neigh)
    peaks   = (dist == loc_max) & (dist > 0.25 * dist.max())

    labeled, n = ndi.label(peaks)
    if n == 0:
        return []

    points = []
    for cy, cx in ndi.center_of_mass(peaks, labeled, range(1, n + 1)):
        ix, iy = int(round(cx)), int(round(cy))
        if 0 <= iy < green_mask.shape[0] and 0 <= ix < green_mask.shape[1]:
            if green_mask[iy, ix] > 0:
                points.append((ix, iy))
    return points


def _try_sam2(image, green_mask):
    """
    SAM2 point-prompted leaf segmentation.
    One foreground point prompt per leaf centre (distance-transform peak)
    is passed to SAM2ImagePredictor. This is more accurate than automatic
    grid sampling for dense, overlapping rosette leaves.
    """
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import torch
    except ImportError:
        raise ImportError("SAM2 not installed")

    prompts = _find_leaf_prompts(green_mask)
    if not prompts:
        raise ValueError("No leaf prompts found in green mask")

    os.makedirs(SAM2_WEIGHTS, exist_ok=True)
    cfg        = "configs/sam2.1/sam2.1_hiera_t.yaml"
    checkpoint = os.path.join(SAM2_WEIGHTS, "sam2.1_hiera_tiny.pt")

    if not os.path.isfile(checkpoint):
        print("Downloading SAM2 tiny checkpoint (~150 MB)...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
            checkpoint)
        print("Download complete.")

    device    = SAM2_DEVICE
    sam2      = build_sam2(cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    h, w       = image.shape[:2]
    total_px   = h * w
    leaf_masks = []

    print(f"  SAM2 prompted: {len(prompts)} candidate leaf points")

    with torch.inference_mode():
        for (px, py) in prompts:
            try:
                masks, scores, _ = predictor.predict(
                    point_coords     = np.array([[px, py]], dtype=np.float32),
                    point_labels     = np.array([1],        dtype=np.int32),
                    multimask_output = True,
                )
                # Convert to numpy if torch tensors returned
                if hasattr(masks, 'numpy'):
                    masks  = masks.numpy()
                if hasattr(scores, 'numpy'):
                    scores = scores.numpy()

                best = masks[int(np.argmax(scores))].astype(bool)
                area = int(np.sum(best))

                if area < MIN_LEAF_AREA_PX:
                    continue
                if area / total_px > MAX_LEAF_AREA_FRAC:
                    continue
                if _green_fraction(image, best) < MIN_GREEN_FRAC:
                    continue
                # Skip if mask is mostly outside the valid canopy region
                if np.sum(best & (green_mask > 0)) / area < 0.30:
                    continue
                # Deduplicate: skip if >50% of this mask overlaps an existing leaf
                if any(np.sum(best & ex) / area > 0.5 for ex in leaf_masks):
                    continue

                leaf_masks.append(best)

            except Exception as e:
                print(f"  SAM2 predict error at ({px},{py}): {e}")
                continue

    return leaf_masks, "sam2-prompted"


# ─────────────────────────────────────────────────────────────────────────────
# Watershed fallback
# ─────────────────────────────────────────────────────────────────────────────
def _watershed_count(green_mask):
    """
    Original watershed leaf counter. Returns list of boolean region masks.
    MAE ~7.4 on mature CVPPP plants; better on early-stage rosettes.
    """
    binary = (green_mask > 0).astype(np.uint8)
    dist   = ndimage.distance_transform_edt(binary)

    threshold   = WS_DIST_THRESHOLD * dist.max() if dist.max() > 0 else 0
    foreground  = (dist > threshold).astype(np.uint8)
    labeled, _  = scipy_label(foreground)

    markers     = np.zeros_like(binary, dtype=np.int32)
    for lbl in range(1, labeled.max() + 1):
        if np.sum(labeled == lbl) >= 20:
            markers[labeled == lbl] = lbl

    dist_3ch    = np.stack([dist] * 3, axis=-1).astype(np.uint8)
    markers     = cv2.watershed(dist_3ch, markers)

    leaf_masks  = []
    for lbl in range(1, markers.max() + 1):
        region = (markers == lbl) & (green_mask > 0)
        if np.sum(region) >= WS_MIN_REGION_SIZE:
            leaf_masks.append(region)

    return leaf_masks


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────
def _save_visualisation(image, leaf_masks, method, chamber_id, image_path):
    os.makedirs(VIS_DIR, exist_ok=True)
    vis    = image.copy()
    overlay = np.zeros_like(image)

    for i, mask in enumerate(leaf_masks):
        colour = LEAF_COLOURS[i % len(LEAF_COLOURS)]
        overlay[mask] = colour

    vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

    # Label each leaf 1 to N sequentially (independent of SAM2 internal mask indices)
    for label_num, mask in enumerate(leaf_masks, start=1):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        text   = str(label_num)
        # scale font to leaf area so labels fit neatly on small and large leaves
        font_scale = max(0.5, min(1.0, np.sum(mask) / 15000))
        thickness  = 2 if font_scale > 0.7 else 1
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        # dark outline then white fill for readability on any background
        cv2.putText(vis, text, (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), thickness + 1)
        cv2.putText(vis, text, (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    # Header bar
    bar = np.zeros((40, vis.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, f"Leaves: {len(leaf_masks)}  |  Method: {method}  |  Chamber: {chamber_id}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    vis = np.vstack([bar, vis])

    date_str  = datetime.now().strftime("%Y-%m-%d")
    out_path  = os.path.join(VIS_DIR, f"{date_str}_{chamber_id}_leaves.jpg")
    cv2.imwrite(out_path, vis)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Public API — called by analyse_image.py
# ─────────────────────────────────────────────────────────────────────────────
def count_leaves(green_mask, image, save_vis=True, chamber_id="enriched", image_path=""):
    """
    Count individual leaves using SAM2 (preferred) or watershed (fallback).

    Returns:
        leaf_count (int), method_used (str)
    """
    leaf_masks = []
    method     = "watershed"  # default

    # ── Try SAM2 ──────────────────────────────────────────────────────────────
    try:
        print("Attempting SAM2 leaf segmentation (CPU — may take 30–60s)...")
        leaf_masks, method = _try_sam2(image, green_mask)
        print(f"SAM2 complete — {len(leaf_masks)} leaves detected.")
    except ImportError:
        print("SAM2 not installed — using watershed fallback.")
        leaf_masks = _watershed_count(green_mask)
        method     = "watershed"
    except Exception as e:
        print(f"SAM2 failed ({e}) — using watershed fallback.")
        leaf_masks = _watershed_count(green_mask)
        method     = "watershed"

    leaf_count = len(leaf_masks)

    if save_vis and leaf_count > 0:
        _save_visualisation(image, leaf_masks, method, chamber_id, image_path)

    return leaf_count, method


# ─────────────────────────────────────────────────────────────────────────────
# Germination detection — unchanged from previous version
# ─────────────────────────────────────────────────────────────────────────────
def check_germination(chamber_id, current_canopy_cover, csv_path=METRICS_CSV,
                      threshold=0.5):
    """
    Returns (is_germination_day, germination_date_str).
    Germination is the first day canopy cover exceeds threshold %.
    """
    if not os.path.isfile(csv_path):
        if current_canopy_cover > threshold:
            return True, datetime.now().strftime("%Y-%m-%d")
        return False, None

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('chamber') != chamber_id:
                continue
            try:
                if float(row.get('germination_flag', 0)) == 1:
                    return False, None   # already flagged
            except ValueError:
                pass

    if current_canopy_cover > threshold:
        return True, datetime.now().strftime("%Y-%m-%d")
    return False, None


# ─────────────────────────────────────────────────────────────────────────────
# CVPPP validation — standalone
# ─────────────────────────────────────────────────────────────────────────────
def _validate_cvppp(cvppp_path):
    import glob
    rgb_files = sorted(glob.glob(os.path.join(cvppp_path, "*_rgb.png")))
    if not rgb_files:
        print(f"No *_rgb.png files found in {cvppp_path}")
        return

    errors_sam2 = []
    errors_ws   = []

    for rgb_path in rgb_files[:20]:
        gt_path = rgb_path.replace("_rgb.png", "_label.png")
        if not os.path.isfile(gt_path):
            continue

        image      = cv2.imread(rgb_path)
        gt         = cv2.imread(gt_path)
        green_mask = _green_mask_hsv(image)

        # Ground truth: count unique non-black colours
        gt_flat  = gt.reshape(-1, 3)
        unique   = set(map(tuple, gt_flat))
        gt_count = len([c for c in unique if c != (0, 0, 0)])

        # Watershed
        ws_masks  = _watershed_count(green_mask)
        ws_count  = len(ws_masks)

        # SAM2
        try:
            sam2_masks, _ = _try_sam2(image, green_mask)
            sam2_count    = len(sam2_masks)
            errors_sam2.append(abs(sam2_count - gt_count))
        except Exception:
            sam2_count = None

        errors_ws.append(abs(ws_count - gt_count))
        sam2_str = str(sam2_count) if sam2_count is not None else "N/A"
        print(f"{os.path.basename(rgb_path)}: GT={gt_count}  WS={ws_count}  SAM2={sam2_str}")

    print(f"\nWatershed  MAE: {np.mean(errors_ws):.2f}")
    if errors_sam2:
        print(f"SAM2       MAE: {np.mean(errors_sam2):.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leaf counter with SAM2 + watershed fallback.")
    parser.add_argument("--image",    default=None, help="Path to chamber image")
    parser.add_argument("--chamber",  default="enriched")
    parser.add_argument("--method",   default="hsv", choices=["hsv", "model"])
    parser.add_argument("--validate", action="store_true", help="Run CVPPP validation")
    parser.add_argument("--cvppp",    default=None, help="Path to CVPPP A1 folder")
    args = parser.parse_args()

    if args.validate:
        if not args.cvppp:
            print("--cvppp path required for validation.")
        else:
            _validate_cvppp(args.cvppp)

    elif args.image:
        image      = cv2.imread(args.image)
        if image is None:
            print(f"Could not load image: {args.image}")
        else:
            if args.method == "model":
                from predict import get_model_mask
                green_mask = get_model_mask(image)
            else:
                green_mask = _green_mask_hsv(image)

            count, method = count_leaves(
                green_mask, image,
                save_vis=True,
                chamber_id=args.chamber,
                image_path=args.image
            )
            print(f"Leaf count: {count}  (method: {method})")
    else:
        parser.print_help()
