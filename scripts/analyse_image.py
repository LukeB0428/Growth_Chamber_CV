"""
analyse_image.py — Whole-Chamber Image Analysis for Growth Chamber CV
EE496 | Luke Buckley | Maynooth University

Runs the complete analysis pipeline on a single top-down chamber image,
computing canopy cover, vegetation indices, rosette geometry, depth metrics,
health indicators, leaf count, bolting detection, and greenness scores.
All results are appended as a new row in results/metrics.csv.

This script operates on the whole chamber frame. For per-pot analysis
(the primary daily workflow) use analyse_chamber.py instead.

Method:
  1. Build green canopy mask via HSV thresholding or U-Net model
  2. Compute vegetation indices: ExG, VARI, NGRDI (+ 13 distribution stats each)
  3. Fit bounding circle to detect rosette diameter
  4. Load 16-bit depth PNG and compute canopy height, volume, soil baseline
  5. Run health sub-pipeline: chlorosis, necrosis, curl, symmetry, LAI
  6. Count leaves (SAM2 primary, watershed fallback)
  7. Run bolting detection (4-signal rule-based detector)
  8. Compute greenness / colour metrics: GCC, CIE Lab, hue, saturation
  9. Compute composite health score (0–100)
 10. Append all metrics to results/metrics.csv

Usage:
    python analyse_image.py --image path/to/image.jpg --chamber enriched
    python analyse_image.py --image path/to/image.jpg --chamber control --method model
    python analyse_image.py --image path/to/image.jpg --no-health --no-leaves
"""

import cv2
import numpy as np
import csv
import os
import math
import argparse
from datetime import datetime
from scipy import stats as scipy_stats
from greenness_metrics import compute_greenness_metrics
from health_score import compute_health_score
from config import METRICS_CSV

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_PATH = str(METRICS_CSV)


def get_previous_canopy_cover(chamber_id):
    if not os.path.isfile(RESULTS_PATH):
        return None
    last_value = None
    with open(RESULTS_PATH, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['chamber'] == chamber_id:
                try:
                    last_value = float(row['canopy_cover_%'])
                except (ValueError, KeyError):
                    pass
    return last_value


_HSV_LOWER       = np.array([25, 40, 40])
_HSV_UPPER       = np.array([90, 255, 255])
_MORPH_KERNEL_SZ = 7


def configure(cfg):
    """Apply species config to this module's segmentation constants."""
    global _HSV_LOWER, _HSV_UPPER, _MORPH_KERNEL_SZ
    seg = cfg.get('segmentation', {})
    if 'hsv_lower' in seg:
        _HSV_LOWER = np.array(seg['hsv_lower'], dtype=np.uint8)
    if 'hsv_upper' in seg:
        _HSV_UPPER = np.array(seg['hsv_upper'], dtype=np.uint8)
    _MORPH_KERNEL_SZ = seg.get('morph_kernel_size', _MORPH_KERNEL_SZ)


def get_green_mask_hsv(image):
    """Classical HSV thresholding (Stage 2). Thresholds loaded from species config."""
    hsv        = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, _HSV_LOWER, _HSV_UPPER)
    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_MORPH_KERNEL_SZ, _MORPH_KERNEL_SZ))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    return green_mask


def get_green_mask_model(image):
    """U-Net model segmentation (Stage 7)."""
    from predict import get_model_mask
    return get_model_mask(image)


def load_depth_map(image_path):
    """
    Load the 16-bit depth PNG corresponding to an RGB image (Stage 4).
    Depth file is named YYYY-MM-DD_{chamber}_depth.png in the same directory.
    Values are in mm; 0 indicates invalid/no reading.
    Returns a uint16 numpy array (400, 640), or None if file not found.
    """
    base, _ = os.path.splitext(image_path)
    depth_path = base + "_depth.png"
    if not os.path.isfile(depth_path):
        print(f"  Depth file not found: {depth_path} — skipping depth metrics")
        return None
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # MUST use UNCHANGED for 16-bit
    if depth is None:
        print(f"  Warning: Could not read depth file: {depth_path}")
        return None
    return depth


def compute_depth_metrics(green_mask, depth_map, pot_mask=None):
    """
    Compute canopy height and volume from OAK-D Lite depth map (Stage 4).

    green_mask  : binary uint8 mask at RGB resolution (1920x1080)
    depth_map   : uint16 array at depth resolution (640x400), values in mm
                  Pixels with depth == 0 are invalid and excluded.
    pot_mask    : optional uint8 mask at RGB resolution; when supplied, soil
                  baseline is restricted to non-plant pixels within the pot circle,
                  preventing the bench surface outside pots from skewing the estimate.

    Method:
      - Resize green_mask to depth resolution with INTER_NEAREST (preserves binary)
      - Soil baseline = median depth of valid non-plant pixels (within pot if mask given)
      - Plant height  = soil_baseline - plant_depth  (smaller depth = closer = taller)
      - Volume        = sum(pixel_heights) * pixel_area_mm2 / 1000 -> cm3
        Pixel area ~2.34 mm2 at ~1m (OAK-D Lite ~73 HFOV, 640 horizontal pixels)

    Returns dict with keys:
        soil_baseline_mm, canopy_height_mean_mm, canopy_height_max_mm, canopy_volume_cm3
    All values float rounded to 2 dp, or None if insufficient data.
    """
    PIXEL_AREA_MM2 = 2.34  # mm² per depth pixel at ~1m camera-to-canopy distance

    # Suppress stereo false-match artifacts on flat leaf texture.
    # Isolated false-match pixels are spatially small; a 5×5 median kills them
    # while preserving real structure (bolting stalk = connected region).
    depth_f = cv2.medianBlur(depth_map.astype(np.float32), 5)
    depth_f[depth_map == 0] = 0.0   # restore invalid pixels (0 = no stereo data)
    depth_map = depth_f

    mask_small  = cv2.resize(green_mask, (640, 400), interpolation=cv2.INTER_NEAREST)
    plant_mask  = mask_small > 0
    valid_mask  = depth_map > 0
    plant_valid = plant_mask & valid_mask

    if pot_mask is not None:
        pot_mask_small  = cv2.resize(pot_mask, (640, 400), interpolation=cv2.INTER_NEAREST) > 0
        non_plant_valid = pot_mask_small & (~plant_mask) & valid_mask
    else:
        non_plant_valid = (~plant_mask) & valid_mask

    if np.sum(non_plant_valid) < 100:
        print("  Warning: insufficient soil pixels for depth baseline estimate")
        return {
            "soil_baseline_mm":      None,
            "canopy_height_mean_mm": None,
            "canopy_height_max_mm":  None,
            "canopy_volume_cm3":     None,
        }

    soil_baseline_mm = float(np.median(depth_map[non_plant_valid].astype(np.float32)))

    if np.sum(plant_valid) == 0:
        return {
            "soil_baseline_mm":      round(soil_baseline_mm, 2),
            "canopy_height_mean_mm": None,
            "canopy_height_max_mm":  None,
            "canopy_volume_cm3":     None,
        }

    plant_depths = depth_map[plant_valid].astype(np.float32)

    # Keep only plant pixels within a physically plausible height range above the soil.
    # Arabidopsis max height (bolting) ≈ 300mm. Pixels outside [baseline-300, baseline]
    # are either stereo artifacts (too far) or chamber objects above the pots (too close).
    MAX_PLANT_HEIGHT_MM = 300.0
    plant_above_soil = plant_depths[
        (plant_depths < soil_baseline_mm) &
        (plant_depths >= soil_baseline_mm - MAX_PLANT_HEIGHT_MM)
    ]
    if len(plant_above_soil) < 10:
        print("  Warning: too few valid plant depth pixels above soil baseline — "
              "depth may be unreliable for this image")
        return {
            "soil_baseline_mm":      round(soil_baseline_mm, 2),
            "canopy_height_mean_mm": None,
            "canopy_height_max_mm":  None,
            "canopy_volume_cm3":     None,
        }

    # Smaller depth value = closer to camera = taller plant
    canopy_height_mean_mm = round(float(soil_baseline_mm - np.median(plant_above_soil)), 2)
    # Use 10th percentile instead of min to avoid single-pixel stereo spikes
    canopy_height_max_mm  = round(float(soil_baseline_mm - np.percentile(plant_above_soil, 10)), 2)
    pixel_heights         = np.clip(soil_baseline_mm - plant_above_soil, 0, None)
    canopy_volume_cm3     = round(float(np.sum(pixel_heights) * PIXEL_AREA_MM2 / 1000), 2)

    return {
        "soil_baseline_mm":      round(soil_baseline_mm, 2),
        "canopy_height_mean_mm": canopy_height_mean_mm,
        "canopy_height_max_mm":  canopy_height_max_mm,
        "canopy_volume_cm3":     canopy_volume_cm3,
    }


def compute_index_stats(values):
    """
    Compute 13 descriptive statistics for a 1D array of index values.
    Matches the feature set used in Jakunskas et al. (2025) for LI-600 regression.

    Returns a dict with keys: mean, median, mode, std, variance, min, max,
    range, skewness, kurtosis, q1, q3, iqr
    """
    if len(values) == 0:
        return {k: 0.0 for k in [
            'mean','median','mode','std','variance',
            'min','max','range','skewness','kurtosis','q1','q3','iqr'
        ]}
    q1  = float(np.percentile(values, 25))
    q3  = float(np.percentile(values, 75))
    return {
        'mean':     round(float(np.mean(values)),    6),
        'median':   round(float(np.median(values)),  6),
        'mode':     round(float(scipy_stats.mode(np.round(values, 3), keepdims=True).mode[0]), 6),
        'std':      round(float(np.std(values)),     6),
        'variance': round(float(np.var(values)),     6),
        'min':      round(float(np.min(values)),     6),
        'max':      round(float(np.max(values)),     6),
        'range':    round(float(np.max(values) - np.min(values)), 6),
        'skewness': round(float(scipy_stats.skew(values)),   6),
        'kurtosis': round(float(scipy_stats.kurtosis(values)), 6),
        'q1':       round(q1, 6),
        'q3':       round(q3, 6),
        'iqr':      round(q3 - q1, 6),
    }


def analyse_image(image_path, chamber_id, method="hsv",
                  run_health=True, run_leaf_count=True, run_bolting=True):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    img_float = image.astype(float)
    B = img_float[:, :, 0]
    G = img_float[:, :, 1]
    R = img_float[:, :, 2]

    # ── Green mask ────────────────────────────────────────────────────────────
    if method == "model":
        green_mask   = get_green_mask_model(image)
        method_label = "model"
    else:
        green_mask   = get_green_mask_hsv(image)
        method_label = "hsv"

    # ── Depth map ─────────────────────────────────────────────────────────────
    depth_map = load_depth_map(image_path)

    # ── Canopy cover ──────────────────────────────────────────────────────────
    canopy_cover = (np.sum(green_mask > 0) / green_mask.size) * 100
    green_pixels = green_mask > 0

    # ── Vegetation indices ────────────────────────────────────────────────────
    ExG      = 2 * G - R - B
    exg_mean = np.mean(ExG)

    # --- VARI ---
    denom_vari                   = G + R - B
    denom_vari[denom_vari == 0]  = 1e-10
    VARI      = (G - R) / denom_vari
    vari_mean = np.mean(VARI[green_pixels]) if np.sum(green_pixels) > 0 else 0.0

    # --- NGRDI (Normalised Green-Red Difference Index) ---
    # Formula: (G - R) / (G + R)
    # Strongest single RGB predictor of LI-600 stomatal conductance
    # (Jakunskas et al. 2025, R² = 0.42 for gsw in greenhouse experiment)
    denom_ngrdi                      = G + R
    denom_ngrdi[denom_ngrdi == 0]    = 1e-10
    NGRDI      = (G - R) / denom_ngrdi
    ngrdi_mean = np.mean(NGRDI[green_pixels]) if np.sum(green_pixels) > 0 else 0.0

    # --- Distribution statistics over canopy pixels ---
    # Computed for NGRDI and VARI — 13 stats each
    # Used as feature set for LI-600 regression (Stage 13)
    if np.sum(green_pixels) > 0:
        ngrdi_stats = compute_index_stats(NGRDI[green_pixels].flatten())
        vari_stats  = compute_index_stats(VARI[green_pixels].flatten())
    else:
        ngrdi_stats = compute_index_stats(np.array([]))
        vari_stats  = compute_index_stats(np.array([]))

    # ── Rosette diameter ──────────────────────────────────────────────────────
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rosette_diameter_px = 0.0
    rosette_area_px     = 0.0
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rosette_area_px = cv2.contourArea(largest_contour)
        if rosette_area_px > 0:
            (cx, cy), radius    = cv2.minEnclosingCircle(largest_contour)
            rosette_diameter_px = round(2 * radius, 2)

    # ── Relative growth rate ──────────────────────────────────────────────────
    previous_cover = get_previous_canopy_cover(chamber_id)
    if previous_cover is not None and previous_cover > 0 and canopy_cover > 0:
        rgr = round(math.log(canopy_cover) - math.log(previous_cover), 6)
    else:
        rgr = None

    # ── Depth metrics ─────────────────────────────────────────────────────────
    canopy_height_mean_mm = canopy_height_max_mm = canopy_volume_cm3 = soil_baseline_mm = None
    if depth_map is not None:
        try:
            dm                    = compute_depth_metrics(green_mask, depth_map)
            soil_baseline_mm      = dm["soil_baseline_mm"]
            canopy_height_mean_mm = dm["canopy_height_mean_mm"]
            canopy_height_max_mm  = dm["canopy_height_max_mm"]
            canopy_volume_cm3     = dm["canopy_volume_cm3"]
        except Exception as e:
            print(f"Warning: depth metrics failed — {e}")

    # ── Health metrics ────────────────────────────────────────────────────────
    chlorosis_pct = necrosis_pct = curl_score = symmetry_score = lai = None
    if run_health:
        try:
            from health_metrics import compute_health_metrics
            health         = compute_health_metrics(image, green_mask, chamber_id, image_path, depth_map)
            chlorosis_pct  = health["chlorosis_pct"]
            necrosis_pct   = health["necrosis_pct"]
            curl_score     = health["curl_score"]
            symmetry_score = health["symmetry_score"]
            lai            = health["lai"]
        except Exception as e:
            print(f"Warning: health metrics failed — {e}")

    # ── Leaf count + germination ──────────────────────────────────────────────
    leaf_count       = None
    germination_flag = 0
    germination_date = None
    if run_leaf_count:
        try:
            from leaf_count import count_leaves, check_germination
            leaf_count, _ = count_leaves(
                green_mask, image,
                save_vis=True,
                chamber_id=chamber_id,
                image_path=image_path
            )
            is_germ, germ_date = check_germination(
                chamber_id, canopy_cover, RESULTS_PATH
            )
            if is_germ:
                germination_flag = 1
                germination_date = germ_date
        except Exception as e:
            print(f"Warning: leaf count failed — {e}")

    # ── Bolting detection ─────────────────────────────────────────────────────
    bolting_flag    = 0
    bolting_date    = None
    bolting_signals = None
    if run_bolting:
        try:
            from bolting_detection import check_bolting
            bolting_flag, bolting_date, bolting_signals = check_bolting(
                image, green_mask, chamber_id, RESULTS_PATH
            )
        except Exception as e:
            print(f"Warning: bolting detection failed — {e}")

    # --- Greenness / Colour Metrics (Stage 15) ---
    # Computes 12 absolute colour metrics over canopy pixels:
    #   Raw: mean_hue, mean_saturation, mean_value, mean_r, mean_g, mean_b
    #   GCC: gcc (Green Chromatic Coordinate)
    #   CIE Lab: lab_L, lab_a (primary greenness axis), lab_b
    #   Composite: greenness_score (0-100), green_shade (named label)
    greenness = compute_greenness_metrics(green_mask, image)

    # --- Composite Health Score (Stage 15) ---
    # Weighted combination of 6 metrics → 0-100 score + named label
    # Weights: necrosis 25%, chlorosis 20%, NGRDI 20%, curl 15%,
    #          symmetry 10%, canopy cover 10%
    health = compute_health_score(
        chlorosis_pct    = chlorosis_pct    if chlorosis_pct    is not None else 0.0,
        necrosis_pct     = necrosis_pct     if necrosis_pct     is not None else 0.0,
        curl_score       = curl_score       if curl_score       is not None else 0.5,
        symmetry_score   = symmetry_score   if symmetry_score   is not None else 0.5,
        ngrdi_mean       = ngrdi_mean,
        canopy_cover_pct = canopy_cover,
    )

    # ── Save to CSV ───────────────────────────────────────────────────────────
    file_exists = os.path.isfile(RESULTS_PATH)
    with open(RESULTS_PATH, 'a', newline='') as csvfile:
        fieldnames = [
            'timestamp', 'chamber', 'method',
            'canopy_cover_%', 'exg_mean', 'vari_mean', 'ngrdi_mean',
            'rosette_diameter_px', 'rosette_area_px', 'rgr',
            'chlorosis_pct', 'necrosis_pct', 'curl_score',
            'symmetry_score', 'lai',
            'leaf_count', 'germination_flag', 'germination_date',
            'bolting_flag', 'bolting_date', 'bolting_signals',
            # NGRDI distribution stats (13)
            'ngrdi_mean_stat', 'ngrdi_median', 'ngrdi_mode', 'ngrdi_std',
            'ngrdi_variance', 'ngrdi_min', 'ngrdi_max', 'ngrdi_range',
            'ngrdi_skewness', 'ngrdi_kurtosis', 'ngrdi_q1', 'ngrdi_q3', 'ngrdi_iqr',
            # VARI distribution stats (13)
            'vari_mean_stat', 'vari_median', 'vari_mode', 'vari_std',
            'vari_variance', 'vari_min', 'vari_max', 'vari_range',
            'vari_skewness', 'vari_kurtosis', 'vari_q1', 'vari_q3', 'vari_iqr',
            # Stage 4 depth metrics (4)
            'canopy_height_mean_mm', 'canopy_height_max_mm',
            'canopy_volume_cm3', 'soil_baseline_mm',
            # Stage 15 greenness / colour metrics (12)
            'mean_hue', 'mean_saturation', 'mean_value',
            'mean_r', 'mean_g', 'mean_b',
            'gcc', 'lab_L', 'lab_a', 'lab_b',
            'greenness_score', 'green_shade',
            # Stage 15 composite health score (2)
            'health_score', 'health_label',
            'image_file',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        def fmt(v):
            return v if v is not None else ''

        writer.writerow({
            'timestamp':           datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'chamber':             chamber_id,
            'method':              method_label,
            'canopy_cover_%':      round(canopy_cover, 2),
            'exg_mean':            round(exg_mean, 2),
            'vari_mean':           round(vari_mean, 4),
            'ngrdi_mean':          round(ngrdi_mean, 4),
            'rosette_diameter_px': rosette_diameter_px,
            'rosette_area_px':     round(rosette_area_px, 2),
            'rgr':                 fmt(rgr),
            'chlorosis_pct':       fmt(chlorosis_pct),
            'necrosis_pct':        fmt(necrosis_pct),
            'curl_score':          fmt(curl_score),
            'symmetry_score':      fmt(symmetry_score),
            'lai':                 fmt(lai),
            'leaf_count':          fmt(leaf_count),
            'germination_flag':    germination_flag,
            'germination_date':    fmt(germination_date),
            'bolting_flag':        bolting_flag,
            'bolting_date':        fmt(bolting_date),
            'bolting_signals':     fmt(bolting_signals),
            # NGRDI stats
            'ngrdi_mean_stat':  ngrdi_stats['mean'],
            'ngrdi_median':     ngrdi_stats['median'],
            'ngrdi_mode':       ngrdi_stats['mode'],
            'ngrdi_std':        ngrdi_stats['std'],
            'ngrdi_variance':   ngrdi_stats['variance'],
            'ngrdi_min':        ngrdi_stats['min'],
            'ngrdi_max':        ngrdi_stats['max'],
            'ngrdi_range':      ngrdi_stats['range'],
            'ngrdi_skewness':   ngrdi_stats['skewness'],
            'ngrdi_kurtosis':   ngrdi_stats['kurtosis'],
            'ngrdi_q1':         ngrdi_stats['q1'],
            'ngrdi_q3':         ngrdi_stats['q3'],
            'ngrdi_iqr':        ngrdi_stats['iqr'],
            # VARI stats
            'vari_mean_stat':  vari_stats['mean'],
            'vari_median':     vari_stats['median'],
            'vari_mode':       vari_stats['mode'],
            'vari_std':        vari_stats['std'],
            'vari_variance':   vari_stats['variance'],
            'vari_min':        vari_stats['min'],
            'vari_max':        vari_stats['max'],
            'vari_range':      vari_stats['range'],
            'vari_skewness':   vari_stats['skewness'],
            'vari_kurtosis':   vari_stats['kurtosis'],
            'vari_q1':         vari_stats['q1'],
            'vari_q3':         vari_stats['q3'],
            'vari_iqr':        vari_stats['iqr'],
            # Stage 4 depth metrics
            'canopy_height_mean_mm': fmt(canopy_height_mean_mm),
            'canopy_height_max_mm':  fmt(canopy_height_max_mm),
            'canopy_volume_cm3':     fmt(canopy_volume_cm3),
            'soil_baseline_mm':      fmt(soil_baseline_mm),
            # Stage 15 greenness / colour metrics
            'mean_hue':        fmt(greenness['mean_hue']),
            'mean_saturation': fmt(greenness['mean_saturation']),
            'mean_value':      fmt(greenness['mean_value']),
            'mean_r':          fmt(greenness['mean_r']),
            'mean_g':          fmt(greenness['mean_g']),
            'mean_b':          fmt(greenness['mean_b']),
            'gcc':             fmt(greenness['gcc']),
            'lab_L':           fmt(greenness['lab_L']),
            'lab_a':           fmt(greenness['lab_a']),
            'lab_b':           fmt(greenness['lab_b']),
            'greenness_score': fmt(greenness['greenness_score']),
            'green_shade':     fmt(greenness['green_shade']),
            # Stage 15 composite health score
            'health_score':    health['health_score'],
            'health_label':    health['health_label'],
            'image_file':      os.path.basename(image_path)
        })

    print(f"[{method_label}] Chamber {chamber_id} | "
          f"Canopy: {canopy_cover:.2f}% | ExG: {exg_mean:.2f} | "
          f"VARI: {vari_mean:.4f} | NGRDI: {ngrdi_mean:.4f} | "
          f"Diameter: {rosette_diameter_px:.1f}px | "
          f"RGR: {f'{rgr:.6f}' if rgr is not None else 'N/A'} | "
          f"Leaves: {leaf_count if leaf_count is not None else 'N/A'} | "
          f"Bolting: {'YES' if bolting_flag else 'no'} | "
          f"Shade: {greenness['green_shade']} | Score: {greenness['greenness_score']} | "
          f"a*: {greenness['lab_a']} | "
          f"Health: {health['health_score']} ({health['health_label']})")
    if depth_map is not None:
        print(f"  Depth: height={canopy_height_mean_mm}mm max={canopy_height_max_mm}mm "
              f"vol={canopy_volume_cm3}cm3 baseline={soil_baseline_mm}mm")

    return canopy_cover, exg_mean, vari_mean, ngrdi_mean, rosette_diameter_px, rgr


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse a growth chamber image and log metrics.")
    parser.add_argument("--image",       default=None)
    parser.add_argument("--chamber",     default="enriched")
    parser.add_argument("--csv",         default=None)
    parser.add_argument("--method",      default="hsv", choices=["hsv", "model"])
    parser.add_argument("--no-health",   action="store_true", help="Skip health metrics")
    parser.add_argument("--no-leaves",   action="store_true", help="Skip leaf counting")
    parser.add_argument("--no-bolting",  action="store_true", help="Skip bolting detection")
    args = parser.parse_args()

    if args.csv:
        RESULTS_PATH = args.csv

    analyse_image(
        args.image,
        chamber_id=args.chamber,
        method=args.method,
        run_health=not args.no_health,
        run_leaf_count=not args.no_leaves,
        run_bolting=not args.no_bolting
    )
