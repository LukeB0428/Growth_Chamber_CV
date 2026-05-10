"""
image_quality.py — Image quality gate for Growth Chamber CV pipeline
EE496 | Luke Buckley | Maynooth University

Checks a captured image for two failure modes before analysis runs:
  1. Blur     — Laplacian variance below threshold (camera shake, condensation)
  2. Brightness — mean pixel value too dark or too bright (lighting failure)

If either check fails the image is flagged. analyse_chamber.py logs the
failure and skips analysis so bad data never enters the CSV.

A quality log is appended to results/quality_log.csv for review.

Usage (standalone):
    python image_quality.py --image images/enriched/2026-04-12_enriched.jpg

Usage (as module):
    from image_quality import check_image_quality
    ok, report = check_image_quality(image_bgr, image_path, chamber_id)
    if not ok:
        print(report)
"""

import cv2
import numpy as np
import csv
import os
from datetime import datetime
from pathlib import Path
from config import RESULTS_DIR

# ── Thresholds ────────────────────────────────────────────────────────────────

# Laplacian variance — measures edge sharpness across the image.
# Computed on a 640×360 downsample for speed.
# Well-focused greenhouse images typically score 200–800+.
# Below 80 indicates significant blur (condensation, camera shake, dirty lens).
BLUR_THRESHOLD = 80

# Mean pixel brightness (0–255 grayscale).
# Below 30  = image too dark (lights off, exposure failure)
# Above 230 = image overexposed (direct sunlight glare, LED wash)
BRIGHTNESS_MIN = 30
BRIGHTNESS_MAX = 230

# Output log
QUALITY_LOG = RESULTS_DIR / "quality_log.csv"
QUALITY_LOG_FIELDS = ["timestamp", "chamber", "image_path",
                      "blur_score", "brightness", "blur_ok", "brightness_ok", "passed"]


# ── Core checks ───────────────────────────────────────────────────────────────

def _blur_score(image_bgr):
    """
    Laplacian variance on a downsampled grayscale image.
    Higher = sharper. Returns float.
    """
    small = cv2.resize(image_bgr, (640, 360), interpolation=cv2.INTER_AREA)
    grey  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def _brightness(image_bgr):
    """Mean pixel brightness (grayscale). Returns float 0–255."""
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(grey))


# ── Logging ───────────────────────────────────────────────────────────────────

def _log_result(chamber, image_path, blur, brightness, blur_ok, brightness_ok):
    passed = blur_ok and brightness_ok
    os.makedirs(RESULTS_DIR, exist_ok=True)
    write_header = not Path(QUALITY_LOG).exists()
    with open(QUALITY_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=QUALITY_LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "chamber":      chamber,
            "image_path":   str(image_path),
            "blur_score":   round(blur, 2),
            "brightness":   round(brightness, 2),
            "blur_ok":      blur_ok,
            "brightness_ok": brightness_ok,
            "passed":       passed,
        })
    return passed


# ── Main function ─────────────────────────────────────────────────────────────

def check_image_quality(image_bgr, image_path, chamber_id):
    """
    Run blur and brightness checks on a loaded image.

    Args:
        image_bgr  : BGR image array (as loaded by cv2.imread)
        image_path : path to the image file (for logging)
        chamber_id : 'enriched' or 'control'

    Returns:
        passed : bool — True if image is acceptable for analysis
        report : str  — human-readable summary of results
    """
    blur       = _blur_score(image_bgr)
    brightness = _brightness(image_bgr)

    blur_ok       = blur >= BLUR_THRESHOLD
    brightness_ok = BRIGHTNESS_MIN <= brightness <= BRIGHTNESS_MAX

    passed = _log_result(chamber_id, image_path, blur, brightness, blur_ok, brightness_ok)

    lines = [f"  [quality] Blur: {blur:.1f} ({'OK' if blur_ok else 'FAIL — image too blurry'})"
             f"  |  Brightness: {brightness:.1f} ({'OK' if brightness_ok else 'FAIL — lighting issue'})"]

    if not blur_ok:
        lines.append(f"  [quality] WARNING: blur score {blur:.1f} < {BLUR_THRESHOLD} "
                     f"— check for condensation or camera movement")
    if brightness < BRIGHTNESS_MIN:
        lines.append(f"  [quality] WARNING: image too dark ({brightness:.1f}) "
                     f"— check chamber lighting")
    if brightness > BRIGHTNESS_MAX:
        lines.append(f"  [quality] WARNING: image overexposed ({brightness:.1f}) "
                     f"— check for direct sunlight")
    if passed:
        lines.append(f"  [quality] PASSED — image accepted for analysis")
    else:
        lines.append(f"  [quality] FAILED — analysis skipped, result logged to quality_log.csv")

    return passed, "\n".join(lines)


# ── Standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check image quality before analysis")
    parser.add_argument("--image",   required=True)
    parser.add_argument("--chamber", default="enriched")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: could not load {args.image}")
        exit(1)

    passed, report = check_image_quality(image, args.image, args.chamber)
    print(f"\n=== Image Quality Check — {args.chamber} ===")
    print(report)
    exit(0 if passed else 1)
