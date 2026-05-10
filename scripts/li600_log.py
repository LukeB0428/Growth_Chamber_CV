"""
li600_log.py -- Ground truth data entry for LI-600 and SPAD readings

Workflow per pot:
  1. Loads the latest chamber image, crops to the pot circle, detects leaves
     via watershed segmentation, and saves an annotated image showing which
     leaves to measure (bright green = target, grey = skip).
  2. Opens the annotated image automatically in Windows Photo Viewer.
  3. Prompts for LI-600 + SPAD readings leaf by leaf, with:
     - 30-second stabilisation countdown + beep when ready
     - Previous session value shown as reference while typing
  4. Derives phi_psii = (Fm' - Fs) / Fm' automatically.
  5. Averages 3 SPAD readings per leaf.
  6. Appends all rows to results/ground_truth.csv keyed by
     date / chamber / pot_label / leaf_id.

CONFIGURATION:
    Edit the ACTIVE_METRICS list to match your specific LI-600 unit.
    Comment out any metrics that are not available.

Usage:
    python li600_log.py --chamber enriched
    python li600_log.py --chamber control  --leaves 3
    python li600_log.py --chamber enriched --date 2026-04-10
    python li600_log.py --chamber enriched --no-timer   # skip countdown
    python li600_log.py --chamber enriched --no-image   # skip annotation
"""

import csv
import os
import sys
import json
import argparse
import time as _time
from datetime import datetime
from pathlib import Path
from config import RESULTS_DIR, CALIB_DIR, IMAGES_DIR, LI600_ANNOT_DIR, GROUND_TRUTH_CSV

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: opencv-python not found — image annotation disabled.")

try:
    import winsound
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False   # non-Windows or missing


# ── Paths ─────────────────────────────────────────────────────────────────────
ANNOT_DIR        = LI600_ANNOT_DIR

# ── HSV green thresholds (mirrors analyse_image.py) ───────────────────────────
if HAS_CV2:
    HSV_LOWER = np.array([25,  40,  40])
    HSV_UPPER = np.array([90, 255, 255])

# ── Pot labels ────────────────────────────────────────────────────────────────
POT_LABELS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]

# ── CONFIGURE METRICS HERE ────────────────────────────────────────────────────
ACTIVE_METRICS = [
    # LI-600 Porometer
    ("gsw",      "Stomatal conductance",      "mol/m²/s",   "gsw — primary regression target"),
    ("vpleaf",   "Leaf vapor pressure",       "kPa",        "VPleaf"),
    ("vpdleaf",  "Vapor pressure deficit",    "kPa",        "VPDleaf"),
    ("h2oleaf",  "Leaf H2O mole fraction",    "mmol/mol",   "H2Oleaf"),
    # LI-600 Fluorometer (comment out if not available)
    ("fs",       "Steady-state fluorescence", "rel. units", "Fs — baseline"),
    ("fm_prime", "Maximum fluorescence",      "rel. units", "Fm' — saturating pulse"),
    # phi_psii is derived automatically — do not add here
    # SPAD meter (3 readings averaged automatically)
    ("spad",     "SPAD chlorophyll index",    "SPAD units", "Mean of 3 readings"),
]

# ── CSV schema ────────────────────────────────────────────────────────────────
FIXED_COLS  = ["date", "chamber", "pot_label", "leaf_id", "leaf_area_px", "leaf_notes"]
METRIC_COLS = [m[0] for m in ACTIVE_METRICS] + ["phi_psii"]
ALL_COLS    = FIXED_COLS + METRIC_COLS


# ── Image helpers ─────────────────────────────────────────────────────────────

def load_calibration(chamber):
    path = CALIB_DIR / f"{chamber}_calibration.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_latest_image(chamber):
    chamber_dir = IMAGES_DIR / chamber
    if not chamber_dir.exists():
        return None
    candidates = sorted(
        [f for f in chamber_dir.glob("*.jpg")
         if "depth" not in f.name and "snapshot" not in f.name],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def crop_pot(img, pot_info, padding=20):
    """Crop a circular pot region from a full chamber image."""
    x = int(pot_info["x"])
    y = int(pot_info["y"])
    r = int(pot_info["r"])
    x1 = max(0, x - r - padding)
    y1 = max(0, y - r - padding)
    x2 = min(img.shape[1], x + r + padding)
    y2 = min(img.shape[0], y + r + padding)
    crop = img[y1:y2, x1:x2].copy()
    # Zero pixels outside the circle
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cx_local = x - x1
    cy_local = y - y1
    cv2.circle(mask, (cx_local, cy_local), r, 255, -1)
    crop[mask == 0] = 0
    return crop


def detect_leaves(pot_img, min_area=300):
    """
    Watershed leaf segmentation on a cropped pot image.
    Returns list of dicts: {leaf_id, area, centroid, contour}
    sorted by area descending (1 = oldest/largest, N = youngest/smallest).
    """
    hsv   = cv2.cvtColor(pot_img, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean  = cv2.morphologyEx(green, cv2.MORPH_OPEN,  kernel, iterations=2)
    clean  = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    if clean.sum() == 0:
        return []

    dist = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
    _, fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    fg     = fg.astype(np.uint8)
    bg     = cv2.dilate(clean, kernel, iterations=3)
    unkn   = cv2.subtract(bg, fg)

    n_labels, markers = cv2.connectedComponents(fg)
    markers = markers + 1
    markers[unkn == 255] = 0

    ws_img = pot_img.copy()
    ws_img[pot_img.sum(axis=2) == 0] = [100, 100, 100]
    cv2.watershed(ws_img, markers)

    leaves = []
    for lbl in range(2, n_labels + 1):
        lmask   = (markers == lbl).astype(np.uint8)
        cnts, _ = cv2.findContours(lmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt  = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        leaves.append({"area": area, "centroid": (cx, cy), "contour": cnt})

    leaves.sort(key=lambda l: l["area"], reverse=True)
    for i, leaf in enumerate(leaves):
        leaf["leaf_id"] = i + 1

    return leaves


def pick_targets(leaves, n_targets):
    """
    Select n_targets leaves to measure.
    Always includes leaf 1 (oldest) and leaf N (youngest above min threshold).
    Fills in evenly spaced leaves for n_targets > 2.
    """
    if not leaves:
        return []
    n = len(leaves)
    if n_targets >= n:
        return [l["leaf_id"] for l in leaves]
    if n_targets == 1:
        # Youngest fully expanded (skip the very last if n > 1 — may still be unfurling)
        return [leaves[max(0, n - 2)]["leaf_id"]]
    if n_targets == 2:
        return [leaves[0]["leaf_id"], leaves[-1]["leaf_id"]]
    # n_targets >= 3: evenly spaced indices
    indices = [round(i * (n - 1) / (n_targets - 1)) for i in range(n_targets)]
    return [leaves[i]["leaf_id"] for i in sorted(set(indices))]


def annotate_and_save(pot_img, leaves, target_ids, save_path):
    """
    Draw numbered leaf labels. Green = measure, grey = skip.
    Saves the annotated image and opens it in the default viewer.
    """
    ann = pot_img.copy()

    for leaf in leaves:
        cx, cy  = leaf["centroid"]
        lid     = leaf["leaf_id"]
        target  = lid in target_ids
        color   = (0, 230, 60) if target else (160, 160, 160)
        thick   = 2            if target else 1

        cv2.drawContours(ann, [leaf["contour"]], -1, color, thick)
        r_circle = 16
        cv2.circle(ann, (cx, cy), r_circle, color, -1)
        cv2.circle(ann, (cx, cy), r_circle, (0, 0, 0), 1)
        text = str(lid)
        text_x = cx - 5 * len(text)
        cv2.putText(ann, text, (text_x, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Legend
    y_leg = 22
    cv2.putText(ann, "GREEN = measure  |  GREY = skip",
                (8, y_leg), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 60), 1)
    y_leg += 18
    for tid in sorted(target_ids):
        age = "oldest" if tid == min(target_ids) else "youngest" if tid == max(target_ids) else "middle"
        cv2.putText(ann, f"Leaf {tid}: {age}",
                    (8, y_leg), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 60), 1)
        y_leg += 16

    os.makedirs(save_path.parent, exist_ok=True)
    cv2.imwrite(str(save_path), ann)

    # Open in default Windows image viewer
    try:
        os.startfile(str(save_path))
    except Exception:
        pass

    return save_path


# ── Input helpers ─────────────────────────────────────────────────────────────

def countdown(seconds=30):
    """30-second countdown with beep at end."""
    print()
    for i in range(seconds, 0, -1):
        print(f"\r  ⏱  Stabilising... {i:2d}s  ", end="", flush=True)
        _time.sleep(1)
    print("\r  ✓  Ready — read the display now.      ")
    if HAS_SOUND:
        for _ in range(2):
            winsound.Beep(1000, 250)
            _time.sleep(0.1)


def get_previous_value(chamber, pot_label, leaf_id, metric):
    """Return the most recent logged value for this pot/leaf/metric."""
    if not GROUND_TRUTH_CSV.exists():
        return None
    try:
        rows = []
        with open(GROUND_TRUTH_CSV, newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("chamber")   == chamber   and
                    row.get("pot_label") == pot_label and
                    row.get("leaf_id")   == leaf_id   and
                    row.get(metric, "")  != ""):
                    rows.append(row)
        if not rows:
            return None
        rows.sort(key=lambda r: r["date"], reverse=True)
        return float(rows[0][metric])
    except Exception:
        return None


def prompt_float(label, unit, description, prev=None, required=False):
    prev_str = f"  prev: {prev:.4g}" if prev is not None else ""
    hint     = f"({description}{prev_str})"
    while True:
        raw = input(f"  {label} [{unit}]  {hint}: ").strip()
        if raw == "" and not required:
            return None
        if raw == "" and required:
            print("  Required — enter a number.")
            continue
        try:
            return float(raw)
        except ValueError:
            print("  Invalid — enter a number or press Enter to skip.")


def prompt_spad(prev=None):
    """Prompt for 3 SPAD readings and return the mean."""
    prev_str = f"  (prev mean: {prev:.1f})" if prev is not None else ""
    print(f"  SPAD chlorophyll index [SPAD units]{prev_str} — 3 readings:")
    readings = []
    for i in range(1, 4):
        while True:
            raw = input(f"    Reading {i}/3 (Enter to skip): ").strip()
            if raw == "":
                break
            try:
                readings.append(float(raw))
                break
            except ValueError:
                print("    Invalid — enter a number.")
    if not readings:
        return None
    mean_val = round(sum(readings) / len(readings), 2)
    print(f"    → SPAD mean: {mean_val:.1f}  ({len(readings)} reading{'s' if len(readings)>1 else ''})")
    return mean_val


# ── CSV ───────────────────────────────────────────────────────────────────────

def ensure_csv_header():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not GROUND_TRUTH_CSV.exists():
        with open(GROUND_TRUTH_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=ALL_COLS).writeheader()
        print(f"Created {GROUND_TRUTH_CSV}")


# ── Main session ──────────────────────────────────────────────────────────────

def log_readings(chamber, date_str, n_targets, use_timer, use_image):
    ensure_csv_header()

    calib      = load_calibration(chamber)  if (use_image and HAS_CV2) else None
    latest_img = get_latest_image(chamber)  if (use_image and HAS_CV2) else None

    if use_image and HAS_CV2:
        if calib is None:
            print("  Note: no calibration JSON found — image annotation disabled.")
        if latest_img is None:
            print("  Note: no captured images found — image annotation disabled.")

    full_img = None
    if calib and latest_img:
        full_img = cv2.imread(str(latest_img))

    print()
    print("=" * 65)
    print(f"  LI-600 Logger — {chamber.upper()} chamber")
    print(f"  Date  : {date_str}")
    print(f"  Leaves: {n_targets} per pot  |  Timer: {'ON' if use_timer else 'OFF'}")
    print(f"  Image : {latest_img.name if latest_img else 'none'}")
    print("=" * 65)
    print(f"  Metrics: {', '.join(m[0] for m in ACTIVE_METRICS)}")
    print(f"  Derived: phi_psii = (Fm' − Fs) / Fm'")
    print("  Press Enter to skip any reading.")
    print("  Measure youngest fully expanded leaf (lying flat, not curling).")
    print("=" * 65)

    all_rows = []

    for pot_label in POT_LABELS:
        pot_idx  = POT_LABELS.index(pot_label)
        pot_info = calib["pots"][pot_idx] if calib else None

        print(f"\n{'─'*65}")
        print(f"  POT {pot_label}  ({pot_idx+1} of {len(POT_LABELS)})")
        print(f"{'─'*65}")

        # ── Leaf detection + annotation ───────────────────────────────────────
        leaves     = []
        target_ids = []

        if full_img is not None and pot_info is not None:
            try:
                pot_crop = crop_pot(full_img, pot_info)
                leaves   = detect_leaves(pot_crop)
                if leaves:
                    target_ids = pick_targets(leaves, n_targets)
                    annot_path = (ANNOT_DIR /
                                  f"{date_str}_{chamber}_{pot_label}_leaves.jpg")
                    annotate_and_save(pot_crop, leaves, target_ids, annot_path)
                    print(f"  Detected {len(leaves)} leaves — measuring {len(target_ids)}: "
                          f"{', '.join('leaf ' + str(t) for t in target_ids)}")
                    print(f"  Annotation: {annot_path.name}  (opening...)")
                else:
                    print("  No leaves detected — check the image or skip this pot.")
            except Exception as e:
                print(f"  Annotation failed: {e}")

        # Fallback: no detection — prompt for n_targets leaves by index
        if not target_ids:
            target_ids = list(range(1, n_targets + 1))
            print(f"  Measuring {n_targets} leaf{'s' if n_targets>1 else ''} (no auto-detection).")

        input("  Press Enter when you have found the target leaf(s)...")

        # ── Prompt per target leaf ────────────────────────────────────────────
        for leaf_id in target_ids:
            leaf_area = next(
                (l["area"] for l in leaves if l["leaf_id"] == leaf_id), ""
            )

            age_label = ""
            if target_ids:
                if leaf_id == min(target_ids):
                    age_label = " (oldest)"
                elif leaf_id == max(target_ids):
                    age_label = " (youngest)"
                else:
                    age_label = " (middle)"

            print(f"\n  ── Leaf {leaf_id}{age_label} ──")
            if leaf_area:
                print(f"     Area: {int(leaf_area)} px")

            leaf_notes = input("  Leaf notes (optional): ").strip()

            row = {
                "date":        date_str,
                "chamber":     chamber,
                "pot_label":   pot_label,
                "leaf_id":     f"leaf_{leaf_id}",
                "leaf_area_px": int(leaf_area) if leaf_area != "" else "",
                "leaf_notes":  leaf_notes,
            }

            for col_name, display_name, unit, description in ACTIVE_METRICS:
                if col_name == "spad":
                    prev = get_previous_value(chamber, pot_label, f"leaf_{leaf_id}", "spad")
                    value = prompt_spad(prev=prev)
                else:
                    # Show timer prompt before first porometer reading
                    if col_name == "gsw" and use_timer:
                        print("  Clamp the LI-600 onto the leaf now.")
                        countdown(30)
                    prev  = get_previous_value(chamber, pot_label, f"leaf_{leaf_id}", col_name)
                    value = prompt_float(display_name, unit, description, prev=prev)

                row[col_name] = value if value is not None else ""

            # Derive phi_psii
            fs_v  = row.get("fs",       "")
            fm_v  = row.get("fm_prime", "")
            if fs_v != "" and fm_v != "" and float(fm_v) > 0:
                phi = (float(fm_v) - float(fs_v)) / float(fm_v)
                row["phi_psii"] = round(phi, 4)
                print(f"  → ΦPSII = {row['phi_psii']:.4f}")
            else:
                row["phi_psii"] = ""

            filled = [k for k in METRIC_COLS if row.get(k) != ""]
            print(f"  Logged {len(filled)}/{len(METRIC_COLS)} metrics.")
            all_rows.append(row)

    # ── Write to CSV ──────────────────────────────────────────────────────────
    with open(GROUND_TRUTH_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLS)
        writer.writerows(all_rows)

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"  Saved {len(all_rows)} leaf readings → {GROUND_TRUTH_CSV.name}")
    print("=" * 65)
    summary_cols = ["gsw", "phi_psii", "spad"]
    print(f"  {'Pot':<5} {'Leaf':<8} " +
          "  ".join(f"{c:>10}" for c in summary_cols))
    print(f"  {'─'*5} {'─'*8} " +
          "  ".join(f"{'─'*10}" for _ in summary_cols))
    for row in all_rows:
        vals = "  ".join(
            f"{str(row.get(c, ''))[:10]:>10}" for c in summary_cols
        )
        print(f"  {row['pot_label']:<5} {row['leaf_id']:<8} {vals}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Log LI-600 and SPAD readings for a growth chamber."
    )
    parser.add_argument("--chamber",  required=True, choices=["enriched", "control"])
    parser.add_argument("--date",     default=None,
                        help="Date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--leaves",   type=int, default=2,
                        help="Leaves to measure per pot (default: 2 = oldest + youngest).")
    parser.add_argument("--no-timer", action="store_true",
                        help="Skip the 30-second stabilisation countdown.")
    parser.add_argument("--no-image", action="store_true",
                        help="Skip leaf annotation images.")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")

    log_readings(
        chamber   = args.chamber,
        date_str  = date_str,
        n_targets = args.leaves,
        use_timer = not args.no_timer,
        use_image = not args.no_image,
    )
