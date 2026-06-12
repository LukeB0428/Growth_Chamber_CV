"""
repair_csv.py — Pads old CSV rows to match the current schema
EE496 | Luke Buckley | Maynooth University

Repairs both results/metrics.csv (whole-chamber) and results/pot_metrics.csv
(per-pot). Any row written by an older version of the pipeline that is missing
columns will have those columns padded with empty strings.

A timestamped backup is created before any changes are written.

Usage:
    python repair_csv.py
"""

import csv
import os
import shutil
from datetime import datetime
from config import METRICS_CSV, POT_METRICS_CSV

# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

# Whole-chamber metrics (results/metrics.csv)
WHOLE_FIELDNAMES = [
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
    # Depth metrics (4)
    'canopy_height_mean_mm', 'canopy_height_max_mm', 'canopy_volume_cm3', 'soil_baseline_mm',
    # Greenness / colour metrics (12)
    'mean_hue', 'mean_saturation', 'mean_value',
    'mean_r', 'mean_g', 'mean_b',
    'gcc', 'lab_L', 'lab_a', 'lab_b',
    'greenness_score', 'green_shade',
    # Composite health score (2)
    'health_score', 'health_label',
    'image_file',
]

# Per-pot metrics (results/pot_metrics.csv) — must stay in sync with POT_FIELDNAMES
# in analyse_chamber.py
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
    # Depth metrics (4)
    'canopy_height_mean_mm', 'canopy_height_max_mm', 'canopy_volume_cm3', 'soil_baseline_mm',
    # Greenness / colour metrics (12)
    'mean_hue', 'mean_saturation', 'mean_value',
    'mean_r', 'mean_g', 'mean_b',
    'gcc', 'lab_L', 'lab_a', 'lab_b',
    'greenness_score', 'green_shade',
    # Composite health score (2)
    'health_score', 'health_label',
    'plant_status',
    # Developmental stage (3)
    'developmental_stage', 'developmental_stage_bbch', 'developmental_stage_conf',
    'image_file',
]


# ─────────────────────────────────────────────
# REPAIR FUNCTION
# ─────────────────────────────────────────────

def repair(csv_path, fieldnames, label):
    csv_path = str(csv_path)

    if not os.path.isfile(csv_path):
        print(f"[{label}] Not found — nothing to repair")
        return

    backup = csv_path.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    shutil.copy(csv_path, backup)
    print(f"[{label}] Backup saved: {backup}")

    with open(csv_path, 'r', newline='') as f:
        content = f.read()

    lines = [l for l in content.strip().splitlines() if l.strip()]
    if not lines:
        print(f"[{label}] CSV is empty — nothing to repair")
        return

    header = None
    for line in lines:
        if line.startswith('timestamp'):
            header = line.split(',')
            break

    if header is None:
        print(f"[{label}] Could not find header row — aborting")
        return

    data_lines = [l for l in lines if not l.startswith('timestamp')]
    print(f"[{label}] Header: {len(header)} columns -> target: {len(fieldnames)} columns")
    print(f"[{label}] Data rows: {len(data_lines)}")

    new_cols = [c for c in fieldnames if c not in header]
    if new_cols:
        print(f"[{label}] New columns being added: {new_cols}")

    repaired = []
    for line in data_lines:
        values = line.split(',')
        row = {header[i].strip(): values[i].strip() if i < len(values) else ''
               for i in range(len(header))}
        repaired.append({col: row.get(col, '') for col in fieldnames})

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(repaired)

    print(f"[{label}] Repaired {len(repaired)} rows -- {len(fieldnames)} columns each\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    repair(METRICS_CSV,     WHOLE_FIELDNAMES, "metrics.csv")
    repair(POT_METRICS_CSV, POT_FIELDNAMES,   "pot_metrics.csv")
