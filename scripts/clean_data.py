"""
clean_data.py — One-shot data cleaning for Growth Chamber CV pipeline
EE496 | Luke Buckley | Maynooth University

Removes bad data from pot_metrics.csv without altering the analysis scripts.

Actions:
  1. Remove rows dated after CUT_DATE (bad post-recalibration data)
  2. Deduplicate rows by (date, chamber, pot_label) — keeps last occurrence
  3. Null out depth/height columns — stereo depth unreliable throughout trial
  4. Null colorimetric/greenness columns on known bad-lighting dates

Usage:
    python clean_data.py [--dry-run] [--cut-date YYYY-MM-DD]
"""

import pandas as pd
import shutil
import argparse
from datetime import datetime
from pathlib import Path
import sys
import os

# Add scripts dir to path so config imports work when called from project root
sys.path.insert(0, os.path.dirname(__file__))
from config import POT_METRICS_CSV

CUT_DATE_DEFAULT = "2026-06-06"   # keep data through this date inclusive

HEIGHT_COLS = [
    "canopy_height_mean_mm",
    "canopy_height_max_mm",
    "canopy_volume_cm3",
    "soil_baseline_mm",
]

# Dates where greenhouse overhead lighting caused colorimetric artefacts.
# Structural metrics (canopy cover, leaf count, diameter) are unaffected.
BAD_LIGHTING_DATES = [
    "2026-04-28",
    "2026-05-05",
    "2026-05-12",
    "2026-05-25",
    "2026-06-03",
]

LIGHTING_ARTIFACT_COLS = [
    "exg_mean",
    "vari_mean", "vari_mean_stat", "vari_median", "vari_mode", "vari_std",
    "vari_variance", "vari_min", "vari_max", "vari_range",
    "vari_skewness", "vari_kurtosis", "vari_q1", "vari_q3", "vari_iqr",
    "ngrdi_mean", "ngrdi_mean_stat", "ngrdi_median", "ngrdi_mode", "ngrdi_std",
    "ngrdi_variance", "ngrdi_min", "ngrdi_max", "ngrdi_range",
    "ngrdi_skewness", "ngrdi_kurtosis", "ngrdi_q1", "ngrdi_q3", "ngrdi_iqr",
    "gcc",
    "greenness_score",
    "green_shade",
    "lab_L", "lab_a", "lab_b",
]


def clean(cut_date_str: str, dry_run: bool = False):
    csv_path = Path(POT_METRICS_CSV)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        return

    cut_date = datetime.strptime(cut_date_str, "%Y-%m-%d").date()

    df = pd.read_csv(csv_path, on_bad_lines="warn", engine="python")
    original_rows = len(df)
    print(f"Loaded {original_rows} rows from {csv_path.name}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["_date"] = df["timestamp"].dt.date

    # 1. Remove rows after cut date
    after_cut = df["_date"] > cut_date
    n_cut = after_cut.sum()
    if n_cut:
        removed_dates = sorted(df.loc[after_cut, "_date"].unique())
        print(f"\nRemoving {n_cut} rows after {cut_date_str}:")
        for d in removed_dates:
            cnt = (df["_date"] == d).sum()
            print(f"  {d}: {cnt} rows")
    else:
        print(f"\nNo rows after {cut_date_str} -- nothing to trim.")
    df = df[~after_cut]

    # 2. Deduplicate
    before_dedup = len(df)
    df = df.sort_values("timestamp").drop_duplicates(
        subset=["_date", "chamber", "pot_label"], keep="last"
    )
    n_dedup = before_dedup - len(df)
    if n_dedup:
        print(f"\nRemoved {n_dedup} duplicate rows (kept last per date/chamber/pot).")
    else:
        print("\nNo duplicates found.")

    # 3. Null height columns
    nulled = []
    for col in HEIGHT_COLS:
        if col in df.columns:
            df[col] = None
            nulled.append(col)
    if nulled:
        print(f"\nNulled height/depth columns: {', '.join(nulled)}")

    # 4. Null colorimetric columns on known bad-lighting dates
    bad_dates = pd.to_datetime(BAD_LIGHTING_DATES).date
    lighting_mask = df["_date"].isin(bad_dates)
    n_lighting = lighting_mask.sum()
    if n_lighting:
        cols_to_null = [c for c in LIGHTING_ARTIFACT_COLS if c in df.columns]
        df.loc[lighting_mask, cols_to_null] = None
        print(f"\nNulled colorimetric columns on {len(BAD_LIGHTING_DATES)} bad-lighting dates "
              f"({n_lighting} rows affected): {', '.join(BAD_LIGHTING_DATES)}")
    else:
        print("\nNo bad-lighting date rows found in this CSV.")

    df = df.drop(columns=["_date"])

    final_rows = len(df)
    print(f"\nResult: {original_rows} -> {final_rows} rows "
          f"({original_rows - final_rows} removed total)")

    if dry_run:
        print("\nDry run -- no changes written.")
        return

    # Backup before overwriting
    backup = csv_path.with_suffix(".csv.bak")
    shutil.copy2(csv_path, backup)
    print(f"Backup saved to {backup.name}")

    df.to_csv(csv_path, index=False)
    print(f"Cleaned CSV written to {csv_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean pot_metrics.csv.")
    parser.add_argument(
        "--cut-date",
        default=CUT_DATE_DEFAULT,
        help=f"Remove rows strictly after this date (default: {CUT_DATE_DEFAULT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be removed without writing anything.",
    )
    args = parser.parse_args()
    clean(args.cut_date, dry_run=args.dry_run)
