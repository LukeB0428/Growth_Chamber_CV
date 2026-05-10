"""
auto_calibrate.py — Automatic pot detection for Growth Chamber CV pipeline
EE496 | Luke Buckley | Maynooth University

Detects the 8 pot circles in a chamber image automatically using local
Hough Circle Transform — searches within a small ROI around each known
pot position, so false positives from elsewhere in the image are ignored.

Two modes:
  1. Auto (silent) — all 8 pots detected within ROI_RADIUS of prior positions
     → saves calibration JSON and returns
  2. Partial — some pots not found in their ROI
     → those pots keep their prior positions, others are updated

Replaces the need to run calibrate_pots.py before each analysis session.
Called automatically by analyse_chamber.py before each per-pot loop.

Usage (standalone, for testing):
    python auto_calibrate.py --image images/enriched/2026-04-12_enriched.jpg --chamber enriched
    python auto_calibrate.py --image images/control/2026-04-12_control.jpg --chamber control
    python auto_calibrate.py --image ... --chamber enriched --debug  # saves debug image
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from config import CALIB_DIR

# ── Tuning parameters ─────────────────────────────────────────────────────────

# ROI size around each prior pot centre to search for circle (in original image px)
# Should be larger than the maximum expected pot movement between sessions
ROI_RADIUS = 200

# Hough parameters (applied within each ROI patch)
HOUGH_DP         = 1.2
HOUGH_PARAM1     = 50    # Canny upper threshold
HOUGH_PARAM2     = 18    # accumulator threshold — higher = fewer false positives
HOUGH_MIN_DIST   = 60    # min distance between centres within ROI

# Radius search range — ratio of prior radius to search within
RADIUS_TOLERANCE = 0.20  # accept ±20% of prior radius

# Maximum allowed displacement from prior centre (px).
# Pots are fixed in a tray — reject any circle further than this.
MAX_DISPLACEMENT = 50

# Only update calibration if this many pots are confidently detected.
# If fewer are found, keep ALL prior positions unchanged.
MIN_CONFIDENT_POTS = 7

N_POTS = 8


# ── Preprocessing ──────────────────────────────────────────────────────────────

def _make_grey(image_bgr):
    """
    Remove green plant pixels, convert to grayscale, apply blur.
    Returns a grayscale image ready for Hough detection.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([25, 40, 40]), np.array([90, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    green_mask = cv2.dilate(green_mask, kernel)

    # Replace green pixels with median background colour
    median_colour = np.median(image_bgr[green_mask == 0], axis=0).astype(np.uint8)
    clean = image_bgr.copy()
    clean[green_mask > 0] = median_colour

    grey = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (9, 9), 2)
    return grey


# ── Local ROI search ───────────────────────────────────────────────────────────

def _search_roi(grey, cx, cy, prior_r, img_h, img_w):
    """
    Search for a circle near (cx, cy) with expected radius prior_r.
    Crops a ROI_RADIUS patch around the prior centre, runs Hough within it.

    Returns (new_x, new_y, new_r) in original image coords if found, else None.
    """
    # Compute ROI bounds (clamped to image edges)
    x1 = max(0, cx - ROI_RADIUS)
    y1 = max(0, cy - ROI_RADIUS)
    x2 = min(img_w, cx + ROI_RADIUS)
    y2 = min(img_h, cy + ROI_RADIUS)

    roi = grey[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    min_r = max(5, int(prior_r * (1 - RADIUS_TOLERANCE)))
    max_r = int(prior_r * (1 + RADIUS_TOLERANCE))

    circles = cv2.HoughCircles(
        roi,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)

    # Pick the circle closest to the prior centre within the ROI
    roi_cx = cx - x1  # prior centre in ROI coords
    roi_cy = cy - y1

    best = None
    best_dist = float("inf")
    for (rx, ry, rr) in circles:
        dist = np.sqrt((rx - roi_cx) ** 2 + (ry - roi_cy) ** 2)
        if dist < best_dist:
            best_dist = dist
            best = (int(rx + x1), int(ry + y1), int(rr))  # native int for JSON

    # Reject if too far from prior — likely a false positive from wet soil / shadows
    if best is not None and best_dist > MAX_DISPLACEMENT:
        return None

    return best


# ── Calibration I/O ───────────────────────────────────────────────────────────

def _load_prior(chamber):
    path = Path(CALIB_DIR) / f"{chamber}_calibration.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _prior_pots_in_original(calib, orig_w, orig_h):
    """Scale prior pot coords from stored image size to original (1920×1080)."""
    stored_w, stored_h = calib["image_size"]
    sx = orig_w / stored_w
    sy = orig_h / stored_h
    result = []
    for p in calib["pots"]:
        result.append({
            "label": p["label"],
            "x":     int(p["x"] * sx),
            "y":     int(p["y"] * sy),
            "r":     int(p["r"] * ((sx + sy) / 2)),
        })
    return result


def _save_calibration(chamber, pots, orig_w, orig_h):
    calib = {
        "chamber":    chamber,
        "image_size": [orig_w, orig_h],
        "pots":       pots,
    }
    path = Path(CALIB_DIR) / f"{chamber}_calibration.json"
    tmp  = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(calib, f, indent=2)
    tmp.replace(path)  # atomic on same filesystem — never leaves a partial file
    return path


def _save_debug_image(image_bgr, pots, found_flags, chamber):
    vis = image_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for p, found in zip(pots, found_flags):
        colour = (0, 255, 0) if found else (0, 165, 255)  # green=detected, orange=prior
        cv2.circle(vis, (p["x"], p["y"]), p["r"], colour, 3)
        cv2.putText(vis, p["label"], (p["x"] - 40, p["y"] + 8), font, 0.7, (255, 255, 255), 2)
    # Legend
    cv2.putText(vis, "Green = auto-detected", (20, 40), font, 0.8, (0, 255, 0), 2)
    cv2.putText(vis, "Orange = prior (not updated)", (20, 75), font, 0.8, (0, 165, 255), 2)
    preview_path = Path(CALIB_DIR) / f"{chamber}_auto_calibration_debug.jpg"
    cv2.imwrite(str(preview_path), vis)
    print(f"  Debug image saved to {preview_path}")


# ── Main entry point ───────────────────────────────────────────────────────────

def auto_calibrate(image_bgr, chamber, debug=False, verbose=True):
    """
    Attempt automatic pot detection using local ROI Hough search.

    Args:
        image_bgr : full-resolution BGR image
        chamber   : 'enriched' or 'control'
        debug     : save annotated debug image
        verbose   : print status messages

    Returns:
        pots   : list of 8 pot dicts (label, x, y, r) in original image coords
        method : 'auto' (all updated) | 'partial' (some fell back) | 'no_prior'
    """
    orig_h, orig_w = image_bgr.shape[:2]
    prior = _load_prior(chamber)

    if prior is None:
        if verbose:
            print("  [auto_calibrate] No prior calibration — run calibrate_pots.py first")
        return None, "no_prior"

    prior_pots = _prior_pots_in_original(prior, orig_w, orig_h)
    grey = _make_grey(image_bgr)

    updated_pots = []
    found_flags  = []
    n_updated    = 0
    n_fallback   = 0

    for pp in prior_pots:
        result = _search_roi(grey, pp["x"], pp["y"], pp["r"], orig_h, orig_w)

        if result is not None:
            nx, ny, _ = result
            updated_pots.append({"label": pp["label"], "x": nx, "y": ny, "r": pp["r"]})  # radius locked to prior
            found_flags.append(True)
            n_updated += 1
        else:
            # Keep prior position (already in original coords)
            updated_pots.append({"label": pp["label"], "x": pp["x"], "y": pp["y"], "r": pp["r"]})
            found_flags.append(False)
            n_fallback += 1

    method = "auto" if n_fallback == 0 else "partial"

    if n_updated < MIN_CONFIDENT_POTS:
        # Not enough confident detections — keep ALL prior positions to avoid drift
        if verbose:
            print(f"  [auto_calibrate] Only {n_updated}/{N_POTS} pots detected "
                  f"(need {MIN_CONFIDENT_POTS}) — keeping all prior positions unchanged")
        updated_pots = [{"label": pp["label"], "x": pp["x"], "y": pp["y"], "r": pp["r"]}
                        for pp in prior_pots]
        found_flags  = [False] * N_POTS
        method = "prior"
    elif verbose:
        if n_fallback == 0:
            print(f"  [auto_calibrate] All {N_POTS} pots detected — calibration updated")
        else:
            print(f"  [auto_calibrate] {n_updated}/{N_POTS} pots detected, "
                  f"{n_fallback} kept prior position")

    _save_calibration(chamber, updated_pots, orig_w, orig_h)

    if debug:
        _save_debug_image(image_bgr, updated_pots, found_flags, chamber)

    return updated_pots, method


# ── Standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic pot detection for Growth Chamber CV")
    parser.add_argument("--image",   required=True)
    parser.add_argument("--chamber", required=True, choices=["enriched", "control"])
    parser.add_argument("--debug",   action="store_true")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: could not load image from {args.image}")
        exit(1)

    print(f"\n=== Auto Calibration — {args.chamber} ===")
    pots, method = auto_calibrate(image, args.chamber, debug=args.debug, verbose=True)

    if pots:
        print(f"\nResult ({method}):")
        for p in pots:
            print(f"  {p['label']}: centre=({p['x']}, {p['y']}), radius={p['r']}px")
    else:
        print("\nAuto-calibration failed — run calibrate_pots.py manually.")
