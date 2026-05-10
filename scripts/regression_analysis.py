"""
regression_analysis.py — CV metric vs LI-600 ground truth regression
EE496 | Luke Buckley | Maynooth University

Merges pot_metrics.csv with ground_truth.csv, computes Pearson and Spearman
correlations between CV-derived metrics and physiological ground truth
(gsw, phi_psii, spad), then produces:
  - results/regression_results.csv   — full correlation table
  - results/plots/correlation_heatmap.png
  - results/plots/regression_scatter_<metric>.png  (one per GT metric)

Usage:
    python regression_analysis.py
    python regression_analysis.py --chamber enriched   # single chamber
    python regression_analysis.py --min-date 2026-05-01
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from config import RESULTS_DIR, POT_METRICS_CSV, GROUND_TRUTH_CSV

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Output paths ──────────────────────────────────────────────────────────────
PLOTS_DIR          = Path(RESULTS_DIR) / "plots"
REGRESSION_CSV     = Path(RESULTS_DIR) / "regression_results.csv"

# ── CV metrics to correlate against ground truth ──────────────────────────────
CV_METRICS = [
    "canopy_cover_%",
    "exg_mean",
    "vari_mean",
    "ngrdi_mean",
    "rosette_diameter_px",
    "rosette_area_px",
    "chlorosis_pct",
    "necrosis_pct",
    "symmetry_score",
    "lai",
    "leaf_count",
    "canopy_height_mean_mm",
    "canopy_height_max_mm",
    "canopy_volume_cm3",
    "gcc",
    "health_score",
    "mean_saturation",
    "lab_a",
    "lab_b",
]

# ── Ground truth metrics ───────────────────────────────────────────────────────
GT_METRICS = ["gsw", "phi_psii", "spad"]


# ── Data loading ──────────────────────────────────────────────────────────────

def _pot_number(pot_label: str) -> str:
    """Extract pot number from label. 'Control_Pot3' or 'Co2_Pot3' → 'P3'."""
    for part in pot_label.replace("_", " ").split():
        if part.isdigit():
            return f"P{part}"
    return ""


def load_cv_metrics(chamber_filter=None, min_date=None):
    df = pd.read_csv(POT_METRICS_CSV, on_bad_lines="skip")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["pot_num"] = df["pot_label"].apply(_pot_number)

    # Keep only numeric CV metrics + join keys
    keep = ["date", "chamber", "pot_label", "pot_num"] + [
        c for c in CV_METRICS if c in df.columns
    ]
    df = df[keep].copy()

    # Average across multiple runs on the same day
    group_cols = ["date", "chamber", "pot_label", "pot_num"]
    numeric_cols = [c for c in df.columns if c not in group_cols]
    df = df.groupby(group_cols, as_index=False)[numeric_cols].mean(numeric_only=True)

    if chamber_filter:
        df = df[df["chamber"] == chamber_filter]
    if min_date:
        df = df[df["date"] >= min_date]

    return df


def load_ground_truth(chamber_filter=None, min_date=None):
    df = pd.read_csv(GROUND_TRUTH_CSV, on_bad_lines="skip")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)

    # Handle legacy plant_id column (test data) vs pot_label (li600_log.py)
    if "pot_label" not in df.columns and "plant_id" in df.columns:
        df = df.rename(columns={"plant_id": "pot_label"})

    # Derive phi_psii if not present but fs/fm_prime are
    if "phi_psii" not in df.columns and "fs" in df.columns and "fm_prime" in df.columns:
        df["phi_psii"] = (
            pd.to_numeric(df["fm_prime"], errors="coerce") -
            pd.to_numeric(df["fs"], errors="coerce")
        ) / pd.to_numeric(df["fm_prime"], errors="coerce")

    available_gt = [c for c in GT_METRICS if c in df.columns]
    group_cols = [c for c in ["date", "chamber", "pot_label"] if c in df.columns]
    keep = group_cols + available_gt
    df = df[[c for c in keep if c in df.columns]].copy()

    for col in available_gt:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if group_cols and available_gt:
        df = df.groupby(group_cols, as_index=False)[available_gt].mean(numeric_only=True)

    if chamber_filter and "chamber" in df.columns:
        df = df[df["chamber"] == chamber_filter]
    if min_date and "date" in df.columns:
        df = df[df["date"] >= min_date]

    return df


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_data(cv_df, gt_df):
    """
    Join on date + chamber + pot number.
    CV uses pot_num (P1–P8); GT uses pot_label (P1–P8 directly from li600_log).
    """
    gt = gt_df.copy()
    if "pot_label" in gt.columns:
        gt = gt.rename(columns={"pot_label": "pot_num"})

    on_cols = [c for c in ["date", "chamber", "pot_num"] if c in cv_df.columns and c in gt.columns]
    if not on_cols:
        return pd.DataFrame()

    merged = pd.merge(cv_df, gt, on=on_cols, how="inner")
    return merged


# ── Correlation ───────────────────────────────────────────────────────────────

def compute_correlations(df, available_gt):
    records = []
    for gt_col in available_gt:
        y = pd.to_numeric(df[gt_col], errors="coerce")
        for cv_col in CV_METRICS:
            if cv_col not in df.columns:
                continue
            x = pd.to_numeric(df[cv_col], errors="coerce")
            mask = x.notna() & y.notna()
            n = mask.sum()
            if n < 4:
                continue
            xv, yv = x[mask].values, y[mask].values
            pr, pp = stats.pearsonr(xv, yv)
            sr, sp = stats.spearmanr(xv, yv)
            slope, intercept, _, _, _ = stats.linregress(xv, yv)
            records.append({
                "gt_metric":        gt_col,
                "cv_metric":        cv_col,
                "n":                int(n),
                "pearson_r":        round(pr, 4),
                "pearson_p":        round(pp, 4),
                "spearman_r":       round(sr, 4),
                "spearman_p":       round(sp, 4),
                "r_squared":        round(pr ** 2, 4),
                "slope":            round(slope, 6),
                "intercept":        round(intercept, 6),
            })

    return pd.DataFrame(records).sort_values(
        ["gt_metric", "r_squared"], ascending=[True, False]
    )


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(corr_df, available_gt):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    pivot = corr_df.pivot(index="cv_metric", columns="gt_metric", values="pearson_r")
    pivot = pivot.reindex(columns=available_gt)

    fig, ax = plt.subplots(figsize=(3 + len(available_gt) * 1.5, max(6, len(pivot) * 0.5)))
    fig.patch.set_facecolor("#0a150a")
    ax.set_facecolor("#0a150a")

    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Pearson r")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, color="white", fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, color="white", fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="black" if abs(val) > 0.5 else "white", fontsize=8)

    ax.set_title("CV Metrics vs Ground Truth — Pearson Correlation",
                 color="white", fontsize=13, pad=12)
    plt.tight_layout()
    out = PLOTS_DIR / "correlation_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved {out}")


def plot_scatter(df, corr_df, gt_col, top_n=6):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    subset = corr_df[corr_df["gt_metric"] == gt_col].head(top_n)
    if subset.empty:
        return

    n_cols = min(3, len(subset))
    n_rows = (len(subset) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4.5 * n_rows))
    fig.patch.set_facecolor("#0a150a")
    axes = np.array(axes).flatten()

    y = pd.to_numeric(df[gt_col], errors="coerce")
    chambers = df["chamber"].values if "chamber" in df.columns else None
    colours = {"enriched": "#4CAF50", "control": "#9C27B0"}

    for idx, (_, row) in enumerate(subset.iterrows()):
        ax = axes[idx]
        ax.set_facecolor("#111a11")
        cv_col = row["cv_metric"]
        x = pd.to_numeric(df[cv_col], errors="coerce")
        mask = x.notna() & y.notna()

        if chambers is not None:
            for ch, colour in colours.items():
                m = mask & (df["chamber"] == ch)
                ax.scatter(x[m], y[m], c=colour, alpha=0.75, s=40, label=ch)
        else:
            ax.scatter(x[mask], y[mask], c="#4CAF50", alpha=0.75, s=40)

        # Regression line
        xv, yv = x[mask].values, y[mask].values
        if len(xv) >= 2:
            xs = np.linspace(xv.min(), xv.max(), 100)
            ys = row["slope"] * xs + row["intercept"]
            ax.plot(xs, ys, color="#FF9800", linewidth=1.5)

        r2_str = f"R²={row['r_squared']:.3f}  p={row['pearson_p']:.3f}"
        ax.set_title(f"{cv_col}", color="white", fontsize=9)
        ax.set_xlabel(cv_col, color="#aaaaaa", fontsize=8)
        ax.set_ylabel(gt_col, color="#aaaaaa", fontsize=8)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.text(0.05, 0.95, r2_str, transform=ax.transAxes,
                color="#dddddd", fontsize=7.5, va="top")
        if chambers is not None:
            ax.legend(fontsize=7, facecolor="#1a2a1a", labelcolor="white")

    # Hide unused axes
    for idx in range(len(subset), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Top CV predictors of {gt_col}", color="white", fontsize=13, y=1.01)
    plt.tight_layout()
    out = PLOTS_DIR / f"regression_scatter_{gt_col}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(chamber_filter=None, min_date=None):
    print("\n=== Regression Analysis ===")

    cv_df = load_cv_metrics(chamber_filter, min_date)
    gt_df = load_ground_truth(chamber_filter, min_date)

    print(f"  CV metric rows:      {len(cv_df)}")
    print(f"  Ground truth rows:   {len(gt_df)}")

    merged = merge_data(cv_df, gt_df)
    print(f"  Merged rows:         {len(merged)}")

    if merged.empty:
        print("\nNo matching rows — check that ground truth dates and pot labels align with pot_metrics.csv.")
        print("Ground truth pot labels should be P1–P8 (as logged by li600_log.py).")
        return

    available_gt = [c for c in GT_METRICS if c in merged.columns and merged[c].notna().sum() >= 4]
    if not available_gt:
        print("\nNot enough ground truth data yet for regression (need ≥4 data points per metric).")
        return

    print(f"  GT metrics available: {available_gt}")
    print(f"  Chambers:            {merged['chamber'].unique().tolist()}")
    print(f"  Date range:          {merged['date'].min()} → {merged['date'].max()}")

    corr_df = compute_correlations(merged, available_gt)

    # Save CSV
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(REGRESSION_CSV, index=False)
    print(f"\n  Results saved to {REGRESSION_CSV}")

    # Print top correlations
    print("\n── Top correlations (|Pearson r| ≥ 0.5) ──────────────────────────")
    top = corr_df[corr_df["pearson_r"].abs() >= 0.5].head(20)
    if top.empty:
        print("  None yet — more data points needed.")
    else:
        print(f"  {'GT metric':<12} {'CV metric':<30} {'r':>7} {'R²':>7} {'p':>8}")
        print(f"  {'-'*12} {'-'*30} {'-'*7} {'-'*7} {'-'*8}")
        for _, row in top.iterrows():
            print(f"  {row['gt_metric']:<12} {row['cv_metric']:<30} "
                  f"{row['pearson_r']:>7.3f} {row['r_squared']:>7.3f} {row['pearson_p']:>8.4f}")

    # Plots
    print("\n── Generating plots ───────────────────────────────────────────────")
    plot_correlation_heatmap(corr_df, available_gt)
    for gt_col in available_gt:
        plot_scatter(merged, corr_df, gt_col)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CV vs LI-600 regression analysis")
    parser.add_argument("--chamber",  choices=["enriched", "control"], default=None)
    parser.add_argument("--min-date", default=None, help="Only use data from this date onward (YYYY-MM-DD)")
    args = parser.parse_args()
    run(chamber_filter=args.chamber, min_date=args.min_date)
