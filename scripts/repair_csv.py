"""
repair_csv.py — Fixes metrics.csv column inconsistencies
EE496 | Luke Buckley | Maynooth University

Run this once to repair metrics.csv rows written by older versions of
analyse_image.py that had fewer columns. Pads missing columns with empty
values so all rows match the current 66-column schema.

Usage:
    python repair_csv.py
"""

import csv
import os
import shutil
from datetime import datetime
from config import METRICS_CSV

METRICS_CSV = str(METRICS_CSV)

FIELDNAMES = [
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
    'canopy_height_mean_mm', 'canopy_height_max_mm', 'canopy_volume_cm3', 'soil_baseline_mm',
    # Stage 15 greenness / colour metrics (12)
    'mean_hue', 'mean_saturation', 'mean_value',
    'mean_r', 'mean_g', 'mean_b',
    'gcc', 'lab_L', 'lab_a', 'lab_b',
    'greenness_score', 'green_shade',
    # Stage 15 composite health score (2)
    'health_score', 'health_label',
    'image_file',
]

if not os.path.isfile(METRICS_CSV):
    print(f"No CSV found at {METRICS_CSV}")
    exit(0)

# Backup first
backup_path = METRICS_CSV.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
shutil.copy(METRICS_CSV, backup_path)
print(f"Backup saved to {backup_path}")

# Read all rows tolerantly
rows = []
with open(METRICS_CSV, 'r', newline='') as f:
    content = f.read()

lines = [l for l in content.strip().splitlines() if l.strip()]
if not lines:
    print("CSV is empty")
    exit(0)

# Find header line
header = None
data_lines = []
for line in lines:
    if line.startswith('timestamp'):
        header = line.split(',')
        break

for line in lines:
    if not line.startswith('timestamp'):
        data_lines.append(line)

if header is None:
    print("Could not find header row")
    exit(0)

print(f"Found header with {len(header)} columns")
print(f"Found {len(data_lines)} data rows")

# Rewrite with full schema
repaired_rows = []
for line in data_lines:
    values = line.split(',')
    row = {}
    for i, col in enumerate(header):
        row[col.strip()] = values[i].strip() if i < len(values) else ''
    # Fill in any missing columns from new schema
    full_row = {col: row.get(col, '') for col in FIELDNAMES}
    repaired_rows.append(full_row)

with open(METRICS_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(repaired_rows)

print(f"Repaired {len(repaired_rows)} rows — all now have {len(FIELDNAMES)} columns")
print(f"CSV saved to {METRICS_CSV}")
