"""
reset_csv.py — Resets metrics.csv and pot_metrics.csv with the correct column schemas.
Run this before starting the trial or after adding new metrics to analyse_image.py.
A timestamped backup is created automatically before any changes are made.

Functions:
    reset_metrics_csv()  — resets results/metrics.csv     (66 columns, whole-chamber)
    reset_pot_csv()      — resets results/pot_metrics.csv (71 columns, per-pot)
"""

import csv
import os
import shutil
from datetime import datetime
from config import METRICS_CSV, POT_METRICS_CSV

POT_CSV = str(POT_METRICS_CSV)
METRICS_CSV = str(METRICS_CSV)

# 66-column whole-chamber schema (matches analyse_image.py)
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

# 71-column per-pot schema (matches analyse_chamber.py) — adds pot_label
POT_FIELDNAMES = [
    'timestamp', 'chamber', 'pot_label', 'method',
    'canopy_cover_%', 'exg_mean', 'vari_mean', 'ngrdi_mean',
    'rosette_diameter_px', 'rosette_area_px', 'rgr',
    'chlorosis_pct', 'necrosis_pct', 'curl_score', 'symmetry_score', 'lai',
    'leaf_count', 'germination_flag', 'germination_date',
    'bolting_flag', 'bolting_date', 'bolting_signals',
    'ngrdi_mean_stat', 'ngrdi_median', 'ngrdi_mode', 'ngrdi_std',
    'ngrdi_variance', 'ngrdi_min', 'ngrdi_max', 'ngrdi_range',
    'ngrdi_skewness', 'ngrdi_kurtosis', 'ngrdi_q1', 'ngrdi_q3', 'ngrdi_iqr',
    'vari_mean_stat', 'vari_median', 'vari_mode', 'vari_std',
    'vari_variance', 'vari_min', 'vari_max', 'vari_range',
    'vari_skewness', 'vari_kurtosis', 'vari_q1', 'vari_q3', 'vari_iqr',
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


def reset_metrics_csv():
    """Reset results/metrics.csv with the current 66-column schema."""
    if os.path.isfile(METRICS_CSV):
        backup = METRICS_CSV.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        shutil.copy(METRICS_CSV, backup)
        print(f"Backup saved to {backup}")
    with open(METRICS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
    print(f"metrics.csv reset with {len(FIELDNAMES)} columns — ready for trial data.")


def reset_pot_csv():
    """Reset results/pot_metrics.csv with the current 71-column schema."""
    if os.path.isfile(POT_CSV):
        backup = POT_CSV.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        shutil.copy(POT_CSV, backup)
        print(f"Backup saved to {backup}")
    with open(POT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=POT_FIELDNAMES)
        writer.writeheader()
    print(f"pot_metrics.csv reset with {len(POT_FIELDNAMES)} columns — ready for trial data.")


if __name__ == "__main__":
    reset_metrics_csv()
    reset_pot_csv()
