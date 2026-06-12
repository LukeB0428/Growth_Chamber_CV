"""
bolting_detection.py — Bolting Detection for Growth Chamber Project
EE496 | Luke Buckley | Maynooth University

Detects the onset of bolting — the transition from vegetative rosette growth
to reproductive growth — in top-down daily images of Arabidopsis thaliana.

Bolting is characterised by:
  1. Rapid increase in rosette diameter relative to canopy cover
     (plant grows upward/outward faster than it fills the tray)
  2. Emergence of a central elongated structure (the flower stalk)
     visible as a high-aspect-ratio contour at the centre of the rosette
  3. A sustained drop in canopy greenness (VARI) as the stalk tissue
     is less green than leaf tissue

Method:
  - Rule-based detector using metrics already computed by analyse_image.py
  - Reads the CSV to detect multi-day trends (diameter/cover ratio, VARI trend)
  - Checks current image for elongated central structure using contour analysis
  - Flags bolting when 2 or more signals agree

This approach requires no training data and runs instantly on CPU.
It is most reliable from week 2 of the trial onwards when enough daily
data points exist to compute meaningful trends.

Usage (standalone):
    python bolting_detection.py --image path/to/image.jpg --chamber enriched

Usage (as module, called from analyse_image.py):
    from bolting_detection import check_bolting
    bolting_flag, bolting_signals = check_bolting(image, green_mask, chamber_id)
"""

import cv2
import numpy as np
import csv
import os
import argparse
from datetime import datetime
from config import METRICS_CSV, BOLTING_VIS_DIR


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

RESULTS_PATH    = str(METRICS_CSV)
BOLTING_VIS_DIR = str(BOLTING_VIS_DIR)

# Minimum days of data before trend analysis is reliable
MIN_DAYS_FOR_TREND = 5

# Rolling median window for smoothing daily metric noise before trend analysis
SMOOTHING_WINDOW = 3

# Bolting signal thresholds
DIAMETER_COVER_RATIO_THRESHOLD = 0.85   # diameter(px) / (canopy_cover * image_area)
                                          # rises sharply at bolting as stalk extends
ELONGATION_ASPECT_RATIO        = 2.5     # central contour aspect ratio for stalk detection
VARI_DROP_THRESHOLD            = 0.03    # VARI drop over 3 days indicating less green tissue
DEPTH_HEIGHT_THRESHOLD_MM      = 40.0   # canopy_height_max_mm above soil baseline — bolt stem


def configure(cfg):
    """Apply species config to this module's bolting detection thresholds."""
    global DIAMETER_COVER_RATIO_THRESHOLD, ELONGATION_ASPECT_RATIO
    global VARI_DROP_THRESHOLD, DEPTH_HEIGHT_THRESHOLD_MM
    global MIN_DAYS_FOR_TREND, SMOOTHING_WINDOW
    b = cfg.get('bolting', {})
    DIAMETER_COVER_RATIO_THRESHOLD = b.get('diameter_cover_ratio_threshold', DIAMETER_COVER_RATIO_THRESHOLD)
    ELONGATION_ASPECT_RATIO        = b.get('elongation_aspect_ratio',        ELONGATION_ASPECT_RATIO)
    VARI_DROP_THRESHOLD            = b.get('vari_drop_threshold',            VARI_DROP_THRESHOLD)
    DEPTH_HEIGHT_THRESHOLD_MM      = b.get('depth_height_threshold_mm',      DEPTH_HEIGHT_THRESHOLD_MM)
    MIN_DAYS_FOR_TREND             = b.get('min_days_for_trend',             MIN_DAYS_FOR_TREND)
    SMOOTHING_WINDOW               = b.get('smoothing_window',               SMOOTHING_WINDOW)


# ─────────────────────────────────────────────
# SMOOTHING HELPER
# ─────────────────────────────────────────────

def _rolling_median(values, window=SMOOTHING_WINDOW):
    """
    Apply a rolling median to a list of floats.
    Each output value is the median of up to `window` preceding values
    (including itself). Reduces single-day noise without shifting trends.
    """
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(float(np.median(values[start:i + 1])))
    return smoothed


def _trend_slope(values):
    """
    Returns the linear regression slope of a sequence of values.
    Positive = rising, negative = falling. More robust than strict monotonic check.
    """
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    if len(x) < 2:
        return 0.0
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


# ─────────────────────────────────────────────
# SIGNAL 1: DIAMETER / COVER RATIO TREND
# ─────────────────────────────────────────────

def check_diameter_cover_trend(chamber_id, csv_path=RESULTS_PATH):
    """
    Checks if the ratio of rosette diameter to canopy cover is rising rapidly.

    During vegetative growth, diameter and canopy cover increase together.
    At bolting, the stalk extends the diameter rapidly while canopy cover
    may plateau or drop. A rising diameter/cover ratio over 3+ days is
    a strong bolting signal.

    Returns:
        signal : bool   — True if bolting signal detected
        detail : str    — explanation of the signal value
    """
    if not os.path.isfile(csv_path):
        return False, "No CSV data yet"

    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('chamber') != chamber_id:
                continue
            try:
                cc   = float(row['canopy_cover_%'])
                diam = float(row['rosette_diameter_px'])
                if cc > 0 and diam > 0:
                    rows.append({'canopy_cover': cc, 'diameter': diam,
                                 'timestamp': row['timestamp']})
            except (ValueError, KeyError):
                pass

    if len(rows) < MIN_DAYS_FOR_TREND:
        return False, f"Only {len(rows)} data points — need {MIN_DAYS_FOR_TREND} for trend"

    recent = rows[-MIN_DAYS_FOR_TREND:]
    raw_ratios      = [r['diameter'] / r['canopy_cover'] for r in recent]
    smoothed_ratios = _rolling_median(raw_ratios)

    slope  = _trend_slope(smoothed_ratios)
    latest = smoothed_ratios[-1]

    if slope > 0 and latest > DIAMETER_COVER_RATIO_THRESHOLD:
        return True, (f"Diameter/cover ratio rising (slope={slope:.3f}, "
                      f"smoothed={[round(v,2) for v in smoothed_ratios]})")
    return False, f"Diameter/cover ratio stable: {latest:.2f} (slope={slope:.3f})"


# ─────────────────────────────────────────────
# SIGNAL 2: CENTRAL ELONGATED STRUCTURE
# ─────────────────────────────────────────────

def check_central_elongation(green_mask, image):
    """
    Checks for an elongated central structure (flower stalk) in the rosette.

    Method:
      1. Find the centroid of the canopy mask
      2. Examine a small central region around the centroid
      3. Fit an ellipse to the largest contour in that region
      4. If the ellipse has a high aspect ratio (long and thin), flag as stalk

    Returns:
        signal : bool  — True if elongated central structure detected
        detail : str   — explanation
    """
    if np.sum(green_mask > 0) == 0:
        return False, "No canopy detected"

    # Find centroid
    moments = cv2.moments(green_mask)
    if moments["m00"] == 0:
        return False, "Could not compute centroid"

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    h, w = green_mask.shape

    # Crop a central region (20% of image size around centroid)
    margin = int(min(h, w) * 0.1)
    x1 = max(0, cx - margin)
    x2 = min(w, cx + margin)
    y1 = max(0, cy - margin)
    y2 = min(h, cy + margin)

    central_region = green_mask[y1:y2, x1:x2]

    if central_region.size == 0 or np.sum(central_region > 0) < 50:
        return False, "Central region too small"

    # Find contours in central region
    contours, _ = cv2.findContours(central_region, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, "No contours in central region"

    # Fit ellipse to largest contour in central region
    largest = max(contours, key=cv2.contourArea)

    if len(largest) < 5:
        return False, "Not enough points for ellipse fitting"

    try:
        ellipse    = cv2.fitEllipse(largest)
        axes       = ellipse[1]  # (minor_axis, major_axis)
        minor, major = min(axes), max(axes)

        if minor == 0:
            return False, "Degenerate ellipse"

        aspect_ratio = major / minor

        if aspect_ratio > ELONGATION_ASPECT_RATIO:
            return True, f"Central elongated structure detected (aspect ratio: {aspect_ratio:.2f})"
        return False, f"Central region not elongated (aspect ratio: {aspect_ratio:.2f})"

    except cv2.error:
        return False, "Ellipse fitting failed"


# ─────────────────────────────────────────────
# SIGNAL 3: VARI TREND DROP
# ─────────────────────────────────────────────

def check_vari_trend(chamber_id, csv_path=RESULTS_PATH):
    """
    Checks for a sustained drop in VARI over recent days.

    As the flower stalk emerges it is less green than leaf tissue,
    pulling down the mean VARI value of the canopy. A consistent
    drop over MIN_DAYS_FOR_TREND days is a supporting bolting signal.

    Returns:
        signal : bool — True if VARI is consistently dropping
        detail : str  — explanation
    """
    if not os.path.isfile(csv_path):
        return False, "No CSV data yet"

    vari_values = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('chamber') != chamber_id:
                continue
            try:
                vari_values.append(float(row['vari_mean']))
            except (ValueError, KeyError):
                pass

    if len(vari_values) < MIN_DAYS_FOR_TREND:
        return False, f"Only {len(vari_values)} VARI values — need {MIN_DAYS_FOR_TREND}"

    recent          = vari_values[-MIN_DAYS_FOR_TREND:]
    smoothed_vari   = _rolling_median(recent)
    slope           = _trend_slope(smoothed_vari)
    drop            = smoothed_vari[0] - smoothed_vari[-1]

    if slope < 0 and drop > VARI_DROP_THRESHOLD:
        return True, (f"VARI dropping (slope={slope:.4f}, drop={drop:.4f}, "
                      f"smoothed={[round(v,4) for v in smoothed_vari]})")
    return False, (f"VARI stable (slope={slope:.4f}, "
                   f"smoothed={[round(v,4) for v in smoothed_vari]})")


# ─────────────────────────────────────────────
# SIGNAL 4: DEPTH HEIGHT SPIKE
# ─────────────────────────────────────────────

def check_depth_height(canopy_height_max_mm):
    """
    Checks if the tallest point of the canopy exceeds DEPTH_HEIGHT_THRESHOLD_MM
    above the soil baseline. A bolting stem shoots upward rapidly and will
    produce a clear height spike in the depth map that rosette leaves do not.

    Args:
        canopy_height_max_mm : float or None — from compute_depth_metrics

    Returns:
        signal : bool — True if height spike detected
        detail : str  — explanation
    """
    if canopy_height_max_mm is None:
        return False, "No depth data available"
    if canopy_height_max_mm > DEPTH_HEIGHT_THRESHOLD_MM:
        return True, f"Canopy height spike: {canopy_height_max_mm:.1f} mm > {DEPTH_HEIGHT_THRESHOLD_MM} mm threshold"
    return False, f"Canopy height normal: {canopy_height_max_mm:.1f} mm"


# ─────────────────────────────────────────────
# SAVE BOLTING VISUALISATION
# ─────────────────────────────────────────────

def save_bolting_visualisation(image, green_mask, signals_fired, chamber_id):
    """
    Saves an annotated image when bolting is detected, showing the
    canopy mask overlay and which signals triggered.
    """
    os.makedirs(BOLTING_VIS_DIR, exist_ok=True)

    vis = image.copy().astype(float) / 255.0

    # Green overlay on canopy
    canopy = green_mask > 0
    vis[canopy, 0] = vis[canopy, 0] * 0.3
    vis[canopy, 1] = vis[canopy, 1] * 0.3 + 0.5
    vis[canopy, 2] = vis[canopy, 2] * 0.3
    vis = (vis.clip(0, 1) * 255).astype(np.uint8)

    # Add bolting alert text
    cv2.putText(vis, "*** BOLTING DETECTED ***",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for i, signal in enumerate(signals_fired):
        cv2.putText(vis, f"- {signal}",
                    (10, 60 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    date_str  = datetime.now().strftime("%Y-%m-%d")
    save_path = os.path.join(BOLTING_VIS_DIR, f"{date_str}_{chamber_id}_bolting.jpg")
    cv2.imwrite(save_path, vis)
    print(f"Bolting visualisation saved to {save_path}")


# ─────────────────────────────────────────────
# MAIN BOLTING CHECK FUNCTION
# ─────────────────────────────────────────────

def check_bolting(image, green_mask, chamber_id, csv_path=RESULTS_PATH,
                  canopy_height_max_mm=None):
    """
    Runs all four bolting signals and flags bolting if 2 or more agree.

    Args:
        image                : BGR image as loaded by OpenCV
        green_mask           : binary canopy mask (0 or 255)
        chamber_id           : 'enriched' or 'control'
        csv_path             : path to metrics CSV for trend analysis
        canopy_height_max_mm : optional float from depth metrics — enables Signal 4

    Returns:
        bolting_flag    : int  — 1 if bolting detected, 0 otherwise
        bolting_date    : str  — today's date if bolting detected, else None
        signals_summary : str  — summary of which signals fired
    """
    s1, d1 = check_diameter_cover_trend(chamber_id, csv_path)
    s2, d2 = check_central_elongation(green_mask, image)
    s3, d3 = check_vari_trend(chamber_id, csv_path)
    s4, d4 = check_depth_height(canopy_height_max_mm)

    signals_fired = []
    if s1: signals_fired.append("DiamCover")
    if s2: signals_fired.append("Elongation")
    if s3: signals_fired.append("VARIdrop")
    if s4: signals_fired.append("DepthSpike")

    n_signals = len(signals_fired)

    print(f"  Bolting signals: DiamCover={'Y' if s1 else 'N'} | "
          f"Elongation={'Y' if s2 else 'N'} | "
          f"VARIdrop={'Y' if s3 else 'N'} | "
          f"DepthSpike={'Y' if s4 else 'N'}  ({n_signals}/4 fired)")

    # Require VARIdrop (sustained greenness decline) or Elongation (stalk structure)
    # as one of the firing signals. DiamCover + DepthSpike alone fires too readily
    # on noisy depth data in plants that haven't developed a visible stalk.
    if n_signals >= 2 and (s2 or s3):
        today = datetime.now().strftime("%Y-%m-%d")
        print(f"  *** BOLTING DETECTED — {chamber_id} — {today} ***")
        save_bolting_visualisation(image, green_mask, signals_fired, chamber_id)
        return 1, today, "+".join(signals_fired)

    return 0, None, "+".join(signals_fired) if signals_fired else "none"


# ─────────────────────────────────────────────
# STANDALONE ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bolting detection for Arabidopsis trial.")
    parser.add_argument("--image",   required=True, help="Path to today's image")
    parser.add_argument("--chamber", default="enriched")
    parser.add_argument("--method",  default="hsv", choices=["hsv", "model"])
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load {args.image}")
        exit(1)

    # Get mask
    if args.method == "model":
        from predict import get_model_mask
        green_mask = get_model_mask(image)
    else:
        hsv        = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, np.array([25, 40, 40]), np.array([90, 255, 255]))
        kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    print(f"\n=== Bolting Detection — {args.chamber} ===")
    flag, date, summary = check_bolting(image, green_mask, args.chamber)

    if flag:
        print(f"\nBolting onset recorded: {date} | Signals: {summary}")
    else:
        print(f"\nNo bolting detected today. Signals fired: {summary if summary != 'none' else 'none'}")
