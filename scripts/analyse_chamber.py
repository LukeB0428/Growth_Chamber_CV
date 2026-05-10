"""
analyse_chamber.py — Per-Pot Analysis Orchestrator for Growth Chamber Project
EE496 | Luke Buckley | Maynooth University

Orchestrates the full analysis pipeline for a single chamber capture:
  1. Whole-chamber analysis via analyse_image.py  → results/metrics.csv
  2. Per-pot analysis using calibration JSON       → results/pot_metrics.csv

Per-pot analysis uses circular masks derived from calibrate_pots.py output
to isolate each pot region in the full 1920x1080 image, then runs the full
metric pipeline on each pot independently.

Produces one CSV row per pot per day in pot_metrics.csv (51 columns).

Graceful degradation: if calibration JSON is missing, whole-chamber analysis
still runs and a warning is printed. Per-pot analysis is silently skipped until
calibrate_pots.py has been run.

Usage (standalone):
    python analyse_chamber.py --image images/enriched/2026-03-04_enriched.jpg --chamber enriched

Usage (called by scheduler_final.py — drop-in replacement for analyse_image.py):
    python analyse_chamber.py --image <path> --chamber enriched|control
                              [--method hsv|model] [--no-health] [--no-leaves] [--no-bolting]
"""

import cv2
import numpy as np
import csv
import os
import json
import math
import argparse
from datetime import datetime
from scipy import stats as scipy_stats
from config import CALIB_DIR, POT_METRICS_CSV, METRICS_CSV


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

CALIB_DIR     = str(CALIB_DIR)
POT_METRICS   = str(POT_METRICS_CSV)
WHOLE_METRICS = str(METRICS_CSV)

# Plants below this canopy cover % are too small for reliable health/leaf/bolting metrics.
# These modules are skipped when canopy is below this threshold to avoid false positives.
MIN_CANOPY_PCT       = 12.0   # threshold for health metrics
MIN_LEAF_COUNT_PCT   = 20.0   # higher threshold for SAM2 leaf counting


# ─────────────────────────────────────────────
# CSV SCHEMA — 67 columns
# ─────────────────────────────────────────────

POT_FIELDNAMES = [
    'timestamp', 'chamber', 'pot_label', 'method',
    'canopy_cover_%', 'exg_mean', 'vari_mean', 'ngrdi_mean',
    'rosette_diameter_px', 'rosette_area_px', 'rgr',
    'chlorosis_pct', 'necrosis_pct', 'curl_score', 'symmetry_score', 'lai',
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
    'canopy_height_mean_mm', 'canopy_height_max_mm', 'canopy_volume_cm3', 'soil_baseline_mm',
    # Stage 15 greenness / colour metrics (12)
    'mean_hue', 'mean_saturation', 'mean_value',
    'mean_r', 'mean_g', 'mean_b',
    'gcc', 'lab_L', 'lab_a', 'lab_b',
    'greenness_score', 'green_shade',
    # Stage 15 composite health score (2)
    'health_score', 'health_label',
    'plant_status',
    'image_file',
]


# ─────────────────────────────────────────────
# CALIBRATION HELPERS
# ─────────────────────────────────────────────

def load_calibration(chamber_id):
    """
    Load the calibration JSON produced by calibrate_pots.py.

    Expected JSON structure:
        { "chamber": str, "image_size": [w, h],
          "hive": {"x": int, "y": int, "r": int},
          "pots": [{"label": str, "x": int, "y": int, "r": int}, ...] }

    Returns the parsed dict, or None if the file does not exist.
    """
    path = os.path.join(CALIB_DIR, f"{chamber_id}_calibration.json")
    if not os.path.isfile(path):
        print(f"  Calibration not found: {path} — skipping per-pot analysis")
        return None
    with open(path) as f:
        return json.load(f)


def make_pot_mask(image_shape, pot):
    """
    Create a binary uint8 circular mask for a single pot.

    Args:
        image_shape : (h, w) or (h, w, c) — from image.shape
        pot         : dict with keys x, y, r (pixel coords in original image)

    Returns:
        uint8 mask, same spatial size as image, 255 inside circle, 0 outside
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.circle(mask, (pot['x'], pot['y']), pot['r'], 255, -1)
    return mask


# ─────────────────────────────────────────────
# PER-POT CSV HELPERS
# ─────────────────────────────────────────────

def get_plant_status(chamber_id, pot_label, current_cover):
    """
    Assess plant status based on recent canopy cover history.

    Rules:
      - 'dead'      : cover < 0.5% for 3+ consecutive days
      - 'warning'   : cover < 1.5% for 2+ consecutive days (but not dead)
      - 'declining' : cover dropped > 30% relative to 3-day average
      - 'healthy'   : otherwise

    Returns one of: 'healthy', 'warning', 'declining', 'dead'
    """
    if not os.path.isfile(POT_METRICS):
        return 'healthy'

    history = []
    with open(POT_METRICS, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('chamber') == chamber_id and row.get('pot_label') == pot_label:
                try:
                    history.append(float(row['canopy_cover_%']))
                except (ValueError, KeyError):
                    pass

    # Add today's value
    all_vals = history + [current_cover]

    if len(all_vals) >= 3:
        last3 = all_vals[-3:]
        if all(v < 0.5 for v in last3):
            return 'dead'
        if all(v < 1.5 for v in last3[-2:]):
            return 'warning'
        avg3 = sum(last3[:-1]) / len(last3[:-1])
        if avg3 > 1.0 and current_cover < avg3 * 0.7:
            return 'declining'
    elif len(all_vals) >= 2:
        if all(v < 0.5 for v in all_vals):
            return 'dead'

    return 'healthy'


def get_previous_pot_cover(chamber_id, pot_label):
    """
    Return the most recent canopy_cover_% for a specific pot from pot_metrics.csv.
    Used to compute per-pot Relative Growth Rate (RGR).
    Returns None if no previous data exists.
    """
    if not os.path.isfile(POT_METRICS):
        return None
    last_value = None
    with open(POT_METRICS, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('chamber') == chamber_id and row.get('pot_label') == pot_label:
                try:
                    last_value = float(row['canopy_cover_%'])
                except (ValueError, KeyError):
                    pass
    return last_value


def write_pot_row(metrics, pot_label, chamber_id, method_label, image_path, override_date=None):
    """
    Append one per-pot metrics row to pot_metrics.csv.

    Args:
        metrics      : dict returned by analyse_pot()
        pot_label    : e.g. 'P1', 'P2', ...
        chamber_id   : 'enriched' or 'control'
        method_label : 'hsv' or 'model'
        image_path   : original chamber image path (for image_file column)
    """
    file_exists = os.path.isfile(POT_METRICS)

    def fmt(v):
        return v if v is not None else ''

    _empty_stats = {k: 0.0 for k in [
        'mean', 'median', 'mode', 'std', 'variance',
        'min', 'max', 'range', 'skewness', 'kurtosis', 'q1', 'q3', 'iqr'
    ]}
    ns = metrics.get('ngrdi_stats', _empty_stats)
    vs = metrics.get('vari_stats',  _empty_stats)

    with open(POT_METRICS, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=POT_FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'timestamp':           (override_date + ' 12:00:00') if override_date else datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'chamber':             chamber_id,
            'pot_label':           pot_label,
            'method':              method_label,
            'canopy_cover_%':      round(metrics['canopy_cover'], 2),
            'exg_mean':            round(metrics['exg_mean'], 2),
            'vari_mean':           round(metrics['vari_mean'], 4),
            'ngrdi_mean':          round(metrics['ngrdi_mean'], 4),
            'rosette_diameter_px': metrics['rosette_diameter_px'],
            'rosette_area_px':     round(metrics['rosette_area_px'], 2),
            'rgr':                 fmt(metrics.get('rgr')),
            'chlorosis_pct':       fmt(metrics.get('chlorosis_pct')),
            'necrosis_pct':        fmt(metrics.get('necrosis_pct')),
            'curl_score':          fmt(metrics.get('curl_score')),
            'symmetry_score':      fmt(metrics.get('symmetry_score')),
            'lai':                 fmt(metrics.get('lai')),
            'leaf_count':          fmt(metrics.get('leaf_count')),
            'germination_flag':    metrics.get('germination_flag', 0),
            'germination_date':    fmt(metrics.get('germination_date')),
            'bolting_flag':        metrics.get('bolting_flag', 0),
            'bolting_date':        fmt(metrics.get('bolting_date')),
            'bolting_signals':     fmt(metrics.get('bolting_signals')),
            # NGRDI stats
            'ngrdi_mean_stat':  ns['mean'],
            'ngrdi_median':     ns['median'],
            'ngrdi_mode':       ns['mode'],
            'ngrdi_std':        ns['std'],
            'ngrdi_variance':   ns['variance'],
            'ngrdi_min':        ns['min'],
            'ngrdi_max':        ns['max'],
            'ngrdi_range':      ns['range'],
            'ngrdi_skewness':   ns['skewness'],
            'ngrdi_kurtosis':   ns['kurtosis'],
            'ngrdi_q1':         ns['q1'],
            'ngrdi_q3':         ns['q3'],
            'ngrdi_iqr':        ns['iqr'],
            # VARI stats
            'vari_mean_stat':  vs['mean'],
            'vari_median':     vs['median'],
            'vari_mode':       vs['mode'],
            'vari_std':        vs['std'],
            'vari_variance':   vs['variance'],
            'vari_min':        vs['min'],
            'vari_max':        vs['max'],
            'vari_range':      vs['range'],
            'vari_skewness':   vs['skewness'],
            'vari_kurtosis':   vs['kurtosis'],
            'vari_q1':         vs['q1'],
            'vari_q3':         vs['q3'],
            'vari_iqr':        vs['iqr'],
            # Depth
            'canopy_height_mean_mm': fmt(metrics.get('canopy_height_mean_mm')),
            'canopy_height_max_mm':  fmt(metrics.get('canopy_height_max_mm')),
            'canopy_volume_cm3':     fmt(metrics.get('canopy_volume_cm3')),
            'soil_baseline_mm':      fmt(metrics.get('soil_baseline_mm')),
            # Stage 15 greenness / colour metrics
            'mean_hue':        fmt(metrics.get('mean_hue')),
            'mean_saturation': fmt(metrics.get('mean_saturation')),
            'mean_value':      fmt(metrics.get('mean_value')),
            'mean_r':          fmt(metrics.get('mean_r')),
            'mean_g':          fmt(metrics.get('mean_g')),
            'mean_b':          fmt(metrics.get('mean_b')),
            'gcc':             fmt(metrics.get('gcc')),
            'lab_L':           fmt(metrics.get('lab_L')),
            'lab_a':           fmt(metrics.get('lab_a')),
            'lab_b':           fmt(metrics.get('lab_b')),
            'greenness_score': fmt(metrics.get('greenness_score')),
            'green_shade':     fmt(metrics.get('green_shade')),
            # Stage 15 composite health score
            'health_score':    fmt(metrics.get('health_score')),
            'health_label':    fmt(metrics.get('health_label')),
            'plant_status':    fmt(metrics.get('plant_status', 'healthy')),
            'image_file':      os.path.basename(image_path),
        })


# ─────────────────────────────────────────────
# PER-POT ANALYSIS
# ─────────────────────────────────────────────

def analyse_pot(image, depth_map, pot, chamber_id, image_path, method="hsv",
                run_health=True, run_leaf_count=True, run_bolting=True):
    """
    Run the full metric pipeline on a single pot.

    Approach: apply a circular mask at the pot's (x, y, r) to the full
    1920x1080 image, zero out all pixels outside the pot, then pass to the
    existing analysis functions unchanged. This avoids any coordinate
    transform complexity while reusing all existing metric code.

    Canopy cover is computed as % of pot circle area (not total image area),
    giving a meaningful per-pot density metric.

    Args:
        image      : full BGR chamber image (1920x1080)
        depth_map  : uint16 depth array (640x400, mm) or None
        pot        : dict with keys label, x, y, r
        chamber_id : 'enriched' or 'control'
        image_path : original image file path (for health vis naming)
        method     : 'hsv' or 'model'

    Returns:
        dict of computed metrics (keys correspond to POT_FIELDNAMES minus metadata)
    """
    from analyse_image import (
        get_green_mask_hsv, get_green_mask_model,
        compute_index_stats, compute_depth_metrics,
    )

    pot_label = pot['label']

    # ── 1. Pot mask & masked image ─────────────────────────────────────────────
    pot_mask  = make_pot_mask(image.shape, pot)
    pot_image = image.copy()
    pot_image[pot_mask == 0] = 0  # zero out everything outside pot circle

    # ── 2. Green mask restricted to pot ───────────────────────────────────────
    if method == "model":
        green_mask = get_green_mask_model(pot_image)
    else:
        green_mask = get_green_mask_hsv(pot_image)
    # Restrict to pot circle (guards against any green bleeding from zeroed edges)
    green_mask = cv2.bitwise_and(green_mask, pot_mask)

    # ── 3. Canopy cover as % of pot area ──────────────────────────────────────
    pot_area_px  = float(np.sum(pot_mask > 0))
    canopy_cover = (np.sum(green_mask > 0) / pot_area_px * 100) if pot_area_px > 0 else 0.0
    green_pixels = green_mask > 0

    # Early-stage guards — bolting always runs regardless of canopy size.
    if canopy_cover < MIN_CANOPY_PCT:
        print(f"  [early stage — canopy {canopy_cover:.2f}% < {MIN_CANOPY_PCT}% — skipping health]")
        run_health = False
    if canopy_cover < MIN_LEAF_COUNT_PCT:
        print(f"  [early stage — canopy {canopy_cover:.2f}% < {MIN_LEAF_COUNT_PCT}% — skipping leaf count]")
        run_leaf_count = False

    # ── 4. Spectral indices ────────────────────────────────────────────────────
    img_float = pot_image.astype(float)
    B = img_float[:, :, 0]
    G = img_float[:, :, 1]
    R = img_float[:, :, 2]

    ExG      = 2 * G - R - B
    # ExG mean over all pot pixels (includes soil) — consistent with whole-chamber method
    exg_mean = float(np.mean(ExG[pot_mask > 0])) if pot_area_px > 0 else 0.0

    denom_vari                  = G + R - B
    denom_vari[denom_vari == 0] = 1e-10
    VARI      = (G - R) / denom_vari
    vari_mean = float(np.mean(VARI[green_pixels])) if np.sum(green_pixels) > 0 else 0.0

    denom_ngrdi                   = G + R
    denom_ngrdi[denom_ngrdi == 0] = 1e-10
    NGRDI      = (G - R) / denom_ngrdi
    ngrdi_mean = float(np.mean(NGRDI[green_pixels])) if np.sum(green_pixels) > 0 else 0.0

    if np.sum(green_pixels) > 0:
        ngrdi_stats = compute_index_stats(NGRDI[green_pixels].flatten())
        vari_stats  = compute_index_stats(VARI[green_pixels].flatten())
    else:
        ngrdi_stats = compute_index_stats(np.array([]))
        vari_stats  = compute_index_stats(np.array([]))

    # ── 5. Rosette diameter ────────────────────────────────────────────────────
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rosette_diameter_px = 0.0
    rosette_area_px     = 0.0
    if contours:
        largest         = max(contours, key=cv2.contourArea)
        rosette_area_px = cv2.contourArea(largest)
        if rosette_area_px > 0:
            _, radius           = cv2.minEnclosingCircle(largest)
            rosette_diameter_px = round(2 * radius, 2)

    # ── 6. RGR (reads from pot_metrics.csv) ───────────────────────────────────
    prev_cover = get_previous_pot_cover(chamber_id, pot_label)
    if prev_cover is not None and prev_cover > 0 and canopy_cover > 0:
        rgr = round(math.log(canopy_cover) - math.log(prev_cover), 6)
    else:
        rgr = None

    # ── 7. Depth metrics ───────────────────────────────────────────────────────
    # green_mask is already restricted to pot circle, so depth analysis is pot-scoped.
    # Soil baseline is computed from non-plant pixels in the full depth frame
    # (valid since the chamber floor is at a consistent depth across all pots).
    canopy_height_mean_mm = canopy_height_max_mm = canopy_volume_cm3 = soil_baseline_mm = None
    if depth_map is not None:
        try:
            dm                    = compute_depth_metrics(green_mask, depth_map)
            soil_baseline_mm      = dm['soil_baseline_mm']
            canopy_height_mean_mm = dm['canopy_height_mean_mm']
            canopy_height_max_mm  = dm['canopy_height_max_mm']
            canopy_volume_cm3     = dm['canopy_volume_cm3']
        except Exception as e:
            print(f"    Warning: depth metrics failed for {pot_label} — {e}")

    # ── 8. Health metrics ──────────────────────────────────────────────────────
    # Note: save_health_visualisation inside compute_health_metrics will overwrite
    # per-pot saves; the last pot's vis is retained. Acceptable limitation.
    chlorosis_pct = necrosis_pct = curl_score = symmetry_score = lai = None
    if run_health:
        try:
            from health_metrics import compute_health_metrics
            health         = compute_health_metrics(
                pot_image, green_mask, chamber_id, image_path, depth_map
            )
            chlorosis_pct  = health['chlorosis_pct']
            necrosis_pct   = health['necrosis_pct']
            curl_score     = health['curl_score']
            symmetry_score = health['symmetry_score']
            lai            = health['lai']
        except Exception as e:
            print(f"    Warning: health metrics failed for {pot_label} — {e}")

    # ── 9. Leaf count + germination ────────────────────────────────────────────
    leaf_count       = None
    germination_flag = 0
    germination_date = None
    if run_leaf_count:
        try:
            from leaf_count import count_leaves, check_germination
            leaf_count, _ = count_leaves(
                green_mask, pot_image,
                save_vis=False,  # skip per-pot leaf visualisation
                chamber_id=chamber_id, image_path=image_path
            )
            is_germ, germ_date = check_germination(chamber_id, canopy_cover, POT_METRICS)
            if is_germ:
                germination_flag = 1
                germination_date = germ_date
        except Exception as e:
            print(f"    Warning: leaf count failed for {pot_label} — {e}")

    # ── 10. Bolting detection ──────────────────────────────────────────────────
    bolting_flag    = 0
    bolting_date    = None
    bolting_signals = None
    if run_bolting:
        try:
            from bolting_detection import check_bolting
            bolting_flag, bolting_date, bolting_signals = check_bolting(
                pot_image, green_mask, chamber_id, POT_METRICS,
                canopy_height_max_mm=canopy_height_max_mm,
            )
        except Exception as e:
            print(f"    Warning: bolting detection failed for {pot_label} — {e}")

    # ── 11. Greenness / colour metrics + composite health score (Stage 15) ─────
    greenness_data = {
        'mean_hue': None, 'mean_saturation': None, 'mean_value': None,
        'mean_r': None, 'mean_g': None, 'mean_b': None,
        'gcc': None, 'lab_L': None, 'lab_a': None, 'lab_b': None,
        'greenness_score': None, 'green_shade': None,
    }
    health_score_val   = None
    health_label_val   = None
    try:
        from greenness_metrics import compute_greenness_metrics
        greenness_data = compute_greenness_metrics(green_mask, pot_image)
    except Exception as e:
        print(f"    Warning: greenness metrics failed for {pot_label} — {e}")
    try:
        from health_score import compute_health_score
        hs = compute_health_score(
            chlorosis_pct    = chlorosis_pct    if chlorosis_pct    is not None else 0.0,
            necrosis_pct     = necrosis_pct     if necrosis_pct     is not None else 0.0,
            curl_score       = curl_score       if curl_score       is not None else 0.5,
            symmetry_score   = symmetry_score   if symmetry_score   is not None else 0.5,
            ngrdi_mean       = ngrdi_mean,
            canopy_cover_pct = canopy_cover,
        )
        health_score_val = hs['health_score']
        health_label_val = hs['health_label']
    except Exception as e:
        print(f"    Warning: health score failed for {pot_label} — {e}")

    plant_status = get_plant_status(chamber_id, pot_label, canopy_cover)
    status_str = {"dead": "  *** DEAD/NO GROWTH ***", "warning": "  ! WARNING: very low growth",
                  "declining": "  ! WARNING: declining", "healthy": ""}.get(plant_status, "")
    print(f"  [{method}] {pot_label} | Canopy: {canopy_cover:.2f}% | "
          f"NGRDI: {ngrdi_mean:.4f} | Diameter: {rosette_diameter_px:.1f}px | "
          f"Leaves: {leaf_count if leaf_count is not None else 'N/A'} | "
          f"Bolting: {'YES' if bolting_flag else 'no'}{status_str}")

    return {
        'canopy_cover':          canopy_cover,
        'exg_mean':              exg_mean,
        'vari_mean':             vari_mean,
        'ngrdi_mean':            ngrdi_mean,
        'ngrdi_stats':           ngrdi_stats,
        'vari_stats':            vari_stats,
        'rosette_diameter_px':   rosette_diameter_px,
        'rosette_area_px':       rosette_area_px,
        'rgr':                   rgr,
        'chlorosis_pct':         chlorosis_pct,
        'necrosis_pct':          necrosis_pct,
        'curl_score':            curl_score,
        'symmetry_score':        symmetry_score,
        'lai':                   lai,
        'leaf_count':            leaf_count,
        'germination_flag':      germination_flag,
        'germination_date':      germination_date,
        'bolting_flag':          bolting_flag,
        'bolting_date':          bolting_date,
        'bolting_signals':       bolting_signals,
        'canopy_height_mean_mm': canopy_height_mean_mm,
        'canopy_height_max_mm':  canopy_height_max_mm,
        'canopy_volume_cm3':     canopy_volume_cm3,
        'soil_baseline_mm':      soil_baseline_mm,
        **greenness_data,
        'health_score':          health_score_val,
        'health_label':          health_label_val,
        'plant_status':          plant_status,
    }


# ─────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────

def analyse_chamber(image_path, chamber_id, method="hsv",
                    run_health=True, run_leaf_count=True, run_bolting=True,
                    run_whole=False, skip_auto_calibrate=False, override_date=None):
    """
    Main orchestrator for a single chamber capture.

    1. Optionally runs whole-chamber analysis via analyse_image() → results/metrics.csv
    2. Loads calibration JSON for this chamber
    3. If calibration found: runs per-pot pipeline for each pot → results/pot_metrics.csv
    4. If calibration missing: prints a warning and returns

    Args:
        image_path : path to the chamber RGB JPEG
        chamber_id : 'enriched' or 'control'
        method     : 'hsv' or 'model'
        run_health, run_leaf_count, run_bolting : toggle optional sub-modules
        run_whole  : if True, also run whole-chamber analysis → metrics.csv (default False)
    """
    from analyse_image import analyse_image, load_depth_map

    # ── 1. Whole-chamber analysis (optional) ─────────────────────────────────
    if run_whole:
        print(f"\n=== Whole-Chamber Analysis ({chamber_id}) ===")
        analyse_image(image_path, chamber_id, method, run_health, run_leaf_count, run_bolting)

    # ── 2. Load full image + depth ────────────────────────────────────────────
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not reload image from {image_path}")
        return
    depth_map = load_depth_map(image_path)

    # ── 2b. Image quality gate ────────────────────────────────────────────────
    from image_quality import check_image_quality
    quality_ok, quality_report = check_image_quality(image, image_path, chamber_id)
    print(quality_report)
    if not quality_ok:
        return

    # ── 3. Auto-calibrate (update pot positions from image) ──────────────────
    if not skip_auto_calibrate:
        try:
            from auto_calibrate import auto_calibrate
            auto_calibrate(image, chamber_id, verbose=True)
        except Exception as e:
            print(f"  [auto_calibrate] Warning: {e} — using existing calibration")

    # ── 4. Load calibration ───────────────────────────────────────────────────
    calib = load_calibration(chamber_id)
    if calib is None:
        return  # calibration not yet run — skip per-pot analysis

    # ── 5. Per-pot loop ───────────────────────────────────────────────────────
    pots = calib['pots']
    print(f"\n=== Per-Pot Analysis ({chamber_id}) — {len(pots)} pots ===")

    for pot in pots:
        print(f"\n--- Pot {pot['label']} ---")
        try:
            metrics = analyse_pot(
                image, depth_map, pot, chamber_id, image_path,
                method, run_health, run_leaf_count, run_bolting
            )
            write_pot_row(metrics, pot['label'], chamber_id, method, image_path, override_date)
        except Exception as e:
            print(f"  Error analysing pot {pot['label']}: {e}")
            continue

    print(f"\n[Done] {len(pots)} pots logged to {POT_METRICS}")


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Per-pot analysis orchestrator for growth chamber. "
            "Drop-in replacement for analyse_image.py in the scheduler."
        )
    )
    parser.add_argument("--image",      required=True,
                        help="Path to the chamber RGB image")
    parser.add_argument("--chamber",    default="enriched",
                        choices=["enriched", "control"])
    parser.add_argument("--method",     default="hsv",
                        choices=["hsv", "model"])
    parser.add_argument("--csv",        default=None,
                        help="Override whole-chamber metrics CSV path")
    parser.add_argument("--no-health",  action="store_true")
    parser.add_argument("--no-leaves",  action="store_true")
    parser.add_argument("--no-bolting",         action="store_true")
    parser.add_argument("--no-auto-calibrate",  action="store_true",
                        help="Skip auto-calibration step (use existing calibration JSON as-is)")
    parser.add_argument("--whole",      action="store_true",
                        help="Also run whole-chamber analysis → metrics.csv (default: per-pot only)")
    parser.add_argument("--date",       default=None,
                        help="Override timestamp date (YYYY-MM-DD) — use when re-running past images")
    args = parser.parse_args()

    # Allow --csv override to propagate to analyse_image
    if args.csv:
        import analyse_image
        analyse_image.RESULTS_PATH = args.csv

    analyse_chamber(
        args.image,
        chamber_id          = args.chamber,
        method              = args.method,
        run_health          = not args.no_health,
        run_leaf_count      = not args.no_leaves,
        run_bolting         = not args.no_bolting,
        run_whole           = args.whole,
        skip_auto_calibrate = args.no_auto_calibrate,
        override_date       = args.date,
    )
