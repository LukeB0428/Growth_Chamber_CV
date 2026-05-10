"""
health_metrics.py — Plant Health and Stress Indicators for Growth Chamber Project
EE496 | Luke Buckley | Maynooth University

Computes five plant health metrics from a top-down RGB image:

  1. Chlorosis %      — percentage of canopy pixels showing yellowing
  2. Necrosis %       — percentage of canopy pixels showing browning/death
  3. Leaf curl score  — shape-based stress indicator (0=healthy, 1=stressed)
  4. Symmetry score   — radial symmetry of the rosette (0=asymmetric, 1=symmetric)
  5. LAI              — Leaf Area Index (stubbed, requires depth data from Stage 4)

Saves a visualisation image showing chlorotic and necrotic regions highlighted
on the original image.

Usage (standalone):
    python health_metrics.py --image path/to/image.jpg --chamber enriched

Usage (as module, called from analyse_image.py):
    from health_metrics import compute_health_metrics
    health = compute_health_metrics(image_bgr, green_mask, chamber_id, image_path)
"""

import cv2
import numpy as np
import argparse
import os
from datetime import datetime
from config import HEALTH_VIS_DIR


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

HEALTH_VIS_DIR = str(HEALTH_VIS_DIR)


# ─────────────────────────────────────────────
# 1. CHLOROSIS DETECTION
# ─────────────────────────────────────────────

def detect_chlorosis(image, green_mask):
    """
    Detects yellowing of leaf tissue (chlorosis) within the canopy mask.

    Chlorotic tissue has shifted from healthy green towards yellow-green.
    In HSV space, yellow-green occupies hue range 15-25 degrees — the band
    just below the healthy green range (25-90). We only look within the
    canopy mask to avoid detecting yellow soil or pot edges.

    Returns:
        chlorosis_pct : float — percentage of canopy pixels classified as chlorotic
        chlorosis_mask: np.uint8 array — binary mask of chlorotic pixels
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Yellow-green hue range (just below healthy green)
    lower_yellow = np.array([15, 40, 60])
    upper_yellow = np.array([25, 255, 255])
    yellow_mask  = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Restrict to canopy pixels only
    canopy_pixels    = green_mask > 0
    chlorosis_mask   = (yellow_mask > 0) & canopy_pixels

    canopy_count     = np.sum(canopy_pixels)
    chlorosis_count  = np.sum(chlorosis_mask)

    if canopy_count > 0:
        chlorosis_pct = (chlorosis_count / canopy_count) * 100
    else:
        chlorosis_pct = 0.0

    return round(chlorosis_pct, 4), chlorosis_mask.astype(np.uint8) * 255


# ─────────────────────────────────────────────
# 2. NECROSIS DETECTION
# ─────────────────────────────────────────────

def detect_necrosis(image, green_mask):
    """
    Detects browning or death of leaf tissue (necrosis) within the canopy mask.

    Necrotic tissue appears brown in RGB images. In HSV space, brown occupies
    a low-saturation warm hue range (5-20 degrees, moderate saturation).
    We look within a dilated version of the canopy mask to catch brown tissue
    at leaf edges that may have been excluded from the green mask itself.

    Returns:
        necrosis_pct : float — percentage of canopy pixels classified as necrotic
        necrosis_mask: np.uint8 array — binary mask of necrotic pixels
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Brown hue range
    lower_brown = np.array([5, 40, 30])
    upper_brown = np.array([20, 180, 180])
    brown_mask  = cv2.inRange(hsv, lower_brown, upper_brown)

    # Dilate the canopy mask slightly to catch necrotic edges
    kernel         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated_canopy = cv2.dilate(green_mask, kernel)

    canopy_pixels  = dilated_canopy > 0
    necrosis_mask  = (brown_mask > 0) & canopy_pixels

    # Percentage relative to original (undilated) canopy size
    canopy_count   = np.sum(green_mask > 0)
    necrosis_count = np.sum(necrosis_mask)

    if canopy_count > 0:
        necrosis_pct = min((necrosis_count / canopy_count) * 100, 100.0)
    else:
        necrosis_pct = 0.0

    return round(necrosis_pct, 4), necrosis_mask.astype(np.uint8) * 255


# ─────────────────────────────────────────────
# 3. LEAF CURL SCORE
# ─────────────────────────────────────────────

def compute_leaf_curl(green_mask):
    """
    Estimates leaf curl and wilting from the shape of leaf contours.

    Healthy flat leaves have a high solidity (contour area / convex hull area).
    Curled or wilted leaves have irregular shapes with concavities, giving
    lower solidity. We compute the mean solidity across all significant
    contours in the mask.

    Score interpretation:
        Close to 1.0 = healthy flat leaves
        Below 0.75   = possible curl or wilting
        Below 0.60   = significant stress indicated

    Returns:
        curl_score: float — mean solidity across leaf contours (0 to 1)
    """
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0

    solidities = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:  # Skip tiny noise contours
            continue
        hull     = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidities.append(area / hull_area)

    if not solidities:
        return 0.0

    return round(float(np.mean(solidities)), 4)


# ─────────────────────────────────────────────
# 4. SYMMETRY SCORE
# ─────────────────────────────────────────────

def compute_symmetry(green_mask):
    """
    Measures the radial symmetry of the rosette.

    Method: find the centroid of the canopy mask, divide the mask into
    four quadrants around the centroid, and compare the canopy area in
    opposing quadrant pairs. A perfectly symmetric rosette has equal
    area in all four quadrants.

    Score interpretation:
        1.0 = perfectly symmetric
        0.0 = completely asymmetric
        Below 0.7 = notable asymmetry, possible stress or directional growth

    Returns:
        symmetry_score: float (0 to 1)
    """
    if np.sum(green_mask > 0) == 0:
        return 0.0

    # Find centroid of canopy mask
    moments = cv2.moments(green_mask)
    if moments["m00"] == 0:
        return 0.0

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    h, w = green_mask.shape

    # Split into four quadrants around centroid
    q1 = green_mask[:cy, :cx]           # top-left
    q2 = green_mask[:cy, cx:]           # top-right
    q3 = green_mask[cy:, :cx]           # bottom-left
    q4 = green_mask[cy:, cx:]           # bottom-right

    a1, a2 = np.sum(q1 > 0), np.sum(q2 > 0)
    a3, a4 = np.sum(q3 > 0), np.sum(q4 > 0)

    # Compare opposing pairs (top-left vs bottom-right, top-right vs bottom-left)
    def pair_symmetry(x, y):
        if x + y == 0:
            return 1.0
        return 1.0 - abs(x - y) / (x + y)

    sym_diag1 = pair_symmetry(a1, a4)
    sym_diag2 = pair_symmetry(a2, a3)

    return round(float((sym_diag1 + sym_diag2) / 2), 4)


# ─────────────────────────────────────────────
# 5. LEAF AREA INDEX (STUB)
# ─────────────────────────────────────────────

def compute_lai(green_mask, depth_map=None):
    """
    Leaf Area Index using Beer-Lambert approximation with optional depth correction (Stage 4).

    Without depth:
        LAI = -ln(1 - canopy_fraction) / k
        k = 0.5 (standard extinction coefficient for planophile canopy)
        canopy_fraction clamped to 0.999 to avoid ln(0).

    With depth (uint16 640x400 array in mm from OAK-D Lite):
        height_factor = clamp(mean_canopy_height_m / 0.05, 0, 1)
        LAI_depth = LAI * (1 + height_factor)
        0.05m reference = typical mature Arabidopsis rosette height.
        Correction saturates at LAI * 2 for very tall canopies.

    Args:
        green_mask : binary uint8 canopy mask (1920x1080)
        depth_map  : uint16 depth array (640x400, mm), or None

    Returns:
        lai: float — estimated leaf area index, rounded to 4 dp
    """
    k = 0.5
    canopy_fraction = min(float(np.sum(green_mask > 0)) / green_mask.size, 0.999)
    lai_bl = -np.log(1.0 - canopy_fraction) / k

    if depth_map is None:
        return round(float(lai_bl), 4)

    # Depth-based layering correction
    mask_small = cv2.resize(green_mask, (640, 400), interpolation=cv2.INTER_NEAREST)
    plant_mask      = mask_small > 0
    valid_mask      = depth_map > 0
    non_plant_valid = (~plant_mask) & valid_mask
    plant_valid     = plant_mask & valid_mask

    if np.sum(non_plant_valid) < 100 or np.sum(plant_valid) < 10:
        return round(float(lai_bl), 4)

    soil_baseline_mm = float(np.median(depth_map[non_plant_valid].astype(np.float32)))
    mean_plant_mm    = float(np.median(depth_map[plant_valid].astype(np.float32)))
    mean_height_m    = (soil_baseline_mm - mean_plant_mm) / 1000.0
    height_factor    = float(np.clip(mean_height_m / 0.05, 0.0, 1.0))

    return round(float(lai_bl * (1.0 + height_factor)), 4)


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def save_health_visualisation(image, green_mask, chlorosis_mask, necrosis_mask,
                               chamber_id, image_path):
    """
    Saves an annotated image showing chlorotic (yellow) and necrotic (red)
    regions overlaid on the original image, alongside the canopy mask.
    """
    os.makedirs(HEALTH_VIS_DIR, exist_ok=True)

    vis = image.copy().astype(float) / 255.0

    # Highlight healthy canopy in green
    canopy = (green_mask > 0) & (chlorosis_mask == 0) & (necrosis_mask == 0)
    vis[canopy, 0] = vis[canopy, 0] * 0.3
    vis[canopy, 1] = vis[canopy, 1] * 0.3 + 0.5
    vis[canopy, 2] = vis[canopy, 2] * 0.3

    # Highlight chlorosis in yellow
    chlor = chlorosis_mask > 0
    vis[chlor, 0] = vis[chlor, 0] * 0.3 + 0.7   # boost red
    vis[chlor, 1] = vis[chlor, 1] * 0.3 + 0.7   # boost green
    vis[chlor, 2] = vis[chlor, 2] * 0.3          # reduce blue

    # Highlight necrosis in red
    necr = necrosis_mask > 0
    vis[necr, 0] = vis[necr, 0] * 0.3 + 0.7     # boost red
    vis[necr, 1] = vis[necr, 1] * 0.3            # reduce green
    vis[necr, 2] = vis[necr, 2] * 0.3            # reduce blue

    vis = (vis.clip(0, 1) * 255).astype(np.uint8)

    # Add legend text
    font       = cv2.FONT_HERSHEY_SIMPLEX
    vis = cv2.putText(vis, "Green = Healthy", (10, 25),  font, 0.6, (0, 200, 0),   2)
    vis = cv2.putText(vis, "Yellow = Chlorosis", (10, 50), font, 0.6, (0, 200, 200), 2)
    vis = cv2.putText(vis, "Red = Necrosis",   (10, 75), font, 0.6, (0, 0, 200),   2)

    # Save with date and chamber in filename
    date_str  = datetime.now().strftime("%Y-%m-%d")
    base_name = f"{date_str}_{chamber_id}_health.jpg"
    save_path = os.path.join(HEALTH_VIS_DIR, base_name)
    cv2.imwrite(save_path, vis)
    print(f"Health visualisation saved to {save_path}")

    return save_path


# ─────────────────────────────────────────────
# MAIN FUNCTION — called from analyse_image.py
# ─────────────────────────────────────────────

def compute_health_metrics(image_bgr, green_mask, chamber_id, image_path, depth_map=None):
    """
    Runs all health metric computations on a single image.

    Args:
        image_bgr  : BGR image as loaded by OpenCV
        green_mask : binary canopy mask (0 or 255) from HSV or model
        chamber_id : 'enriched' or 'control'
        image_path : original image path (used for naming visualisation)
        depth_map  : uint16 depth array (640x400, mm) from OAK-D Lite, or None (Stage 4)

    Returns:
        dict with keys: chlorosis_pct, necrosis_pct, curl_score,
                        symmetry_score, lai
    """
    chlorosis_pct,  chlorosis_mask  = detect_chlorosis(image_bgr, green_mask)
    necrosis_pct,   necrosis_mask   = detect_necrosis(image_bgr, green_mask)
    curl_score                       = compute_leaf_curl(green_mask)
    symmetry_score                   = compute_symmetry(green_mask)
    lai                              = compute_lai(green_mask, depth_map)

    save_health_visualisation(
        image_bgr, green_mask, chlorosis_mask, necrosis_mask,
        chamber_id, image_path
    )

    print(f"  Chlorosis: {chlorosis_pct:.2f}% | Necrosis: {necrosis_pct:.2f}% | "
          f"Curl: {curl_score:.4f} | Symmetry: {symmetry_score:.4f} | LAI: {lai:.4f}")

    return {
        "chlorosis_pct":  chlorosis_pct,
        "necrosis_pct":   necrosis_pct,
        "curl_score":     curl_score,
        "symmetry_score": symmetry_score,
        "lai":            lai,
    }


# ─────────────────────────────────────────────
# STANDALONE ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute plant health metrics from a chamber image.")
    parser.add_argument("--image",   required=True, help="Path to the image file")
    parser.add_argument("--chamber", default="enriched", help="Chamber ID")
    parser.add_argument("--method",  default="hsv", choices=["hsv", "model"],
                        help="Segmentation method for canopy mask")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image from {args.image}")
        exit(1)

    # Get canopy mask using chosen method
    if args.method == "model":
        from predict import get_model_mask
        green_mask = get_model_mask(image)
    else:
        hsv         = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        green_mask  = cv2.inRange(hsv, lower_green, upper_green)
        kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        green_mask  = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    print(f"\n=== Health Metrics — {args.chamber} ({args.method}) ===")
    metrics = compute_health_metrics(image, green_mask, args.chamber, args.image)

    print("\nSummary:")
    for key, val in metrics.items():
        print(f"  {key}: {val}")
