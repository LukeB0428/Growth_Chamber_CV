"""
visualise_pots.py — Per-Pot Growth Dashboard for Growth Chamber CV
EE496 | Luke Buckley | Maynooth University

Reads pot_metrics.csv and produces a set of plots for comparing individual
pot performance within and between chambers (enriched vs control).

Plots produced:
  01_pots_canopy_cover.png   — 8-pot grid, canopy cover % over time
  02_pots_ngrdi.png          — 8-pot grid, NGRDI over time
  03_pots_health_score.png   — 8-pot grid, composite health score over time
  04_pots_rgr.png            — 8-pot grid, relative growth rate over time
  05_pots_heatmap.png        — spatial heatmap of canopy cover on latest day
  06_pots_summary.png        — enriched vs control mean ± std across all pots

Each 8-pot grid has one subplot per pot label (P1–P8), with enriched and
control lines plotted together so the CO2 treatment effect is visible per pot.

Usage:
    python visualise_pots.py
    python visualise_pots.py --chamber enriched   # single chamber only
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import os
import argparse
from config import POT_METRICS_CSV, PLOTS_DIR

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH    = str(POT_METRICS_CSV)
OUT_DIR     = str(PLOTS_DIR)
POT_LABELS  = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
CHAMBERS    = ["enriched", "control"]
COLORS      = {"enriched": "#2D6A4F", "control": "#74C69D"}
GRID_ROWS   = 2
GRID_COLS   = 4

os.makedirs(OUT_DIR, exist_ok=True)


# ── Load data ─────────────────────────────────────────────────────────────────

def load_pot_data(chamber_filter=None):
    if not os.path.isfile(CSV_PATH):
        print(f"pot_metrics.csv not found at {CSV_PATH}")
        exit(0)

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        print("pot_metrics.csv is empty — run analyse_chamber.py first.")
        exit(0)

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    numeric_cols = [
        'canopy_cover_%', 'exg_mean', 'vari_mean', 'ngrdi_mean',
        'rosette_diameter_px', 'rosette_area_px', 'rgr',
        'chlorosis_pct', 'necrosis_pct', 'curl_score', 'symmetry_score',
        'lai', 'leaf_count', 'health_score',
        'canopy_height_mean_mm', 'canopy_height_max_mm', 'canopy_volume_cm3',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if chamber_filter:
        df = df[df['chamber'] == chamber_filter]

    print(f"Loaded {len(df)} rows from pot_metrics.csv")
    print(f"Chambers: {df['chamber'].unique().tolist()}")
    print(f"Pots:     {sorted(df['pot_label'].unique().tolist())}")
    print(f"Dates:    {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print()
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_plot(fig, filename):
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


def format_xaxis(ax, df):
    n_days = (df['timestamp'].max() - df['timestamp'].min()).days
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    if n_days <= 21:
        ax.xaxis.set_major_locator(mdates.DayLocator())
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)


def pot_grid_figure(df, metric, title, ylabel, filename,
                    hline=None, ylim=None):
    """
    Create a 2×4 grid of subplots, one per pot label.
    Each subplot shows enriched and control lines for that pot over time.
    """
    chambers_present = [c for c in CHAMBERS if c in df['chamber'].values]
    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS,
                             figsize=(16, 7), sharey=False)
    axes = axes.flatten()

    for idx, pot_label in enumerate(POT_LABELS):
        ax = axes[idx]
        has_data = False

        for chamber in chambers_present:
            subset = df[
                (df['chamber'] == chamber) &
                (df['pot_label'] == pot_label)
            ][['timestamp', metric]].dropna()

            if subset.empty:
                continue

            ax.plot(
                subset['timestamp'], subset[metric],
                marker='o', linewidth=1.8, markersize=4,
                color=COLORS[chamber], label=chamber.capitalize(),
            )
            has_data = True

        ax.set_title(pot_label, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.25)

        if hline is not None:
            ax.axhline(hline, color='grey', linewidth=0.8, linestyle='--')
        if ylim:
            ax.set_ylim(ylim)
        if has_data:
            format_xaxis(ax, df)

    # Shared y-label on left column
    for row in range(GRID_ROWS):
        axes[row * GRID_COLS].set_ylabel(ylabel, fontsize=8)

    # Single legend at bottom
    handles = [
        mpatches.Patch(color=COLORS[c], label=c.capitalize())
        for c in chambers_present
    ]
    fig.legend(handles=handles, loc='lower center', ncol=len(chambers_present),
               fontsize=10, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_plot(fig, filename)


# ── Plot 1: Canopy Cover grid ──────────────────────────────────────────────────

def plot_canopy_cover(df):
    pot_grid_figure(
        df, metric='canopy_cover_%',
        title='Per-Pot Canopy Cover (%) Over Time',
        ylabel='Canopy Cover (%)',
        filename='07_pots_canopy_cover.png',
    )


# ── Plot 2: NGRDI grid ────────────────────────────────────────────────────────

def plot_ngrdi(df):
    pot_grid_figure(
        df, metric='ngrdi_mean',
        title='Per-Pot NGRDI Over Time',
        ylabel='Mean NGRDI',
        filename='08_pots_ngrdi.png',
    )


# ── Plot 3: Health score grid ─────────────────────────────────────────────────

def plot_health_score(df):
    if 'health_score' not in df.columns or df['health_score'].dropna().empty:
        print("Skipping health score plot — no data yet.")
        return
    pot_grid_figure(
        df, metric='health_score',
        title='Per-Pot Composite Health Score (0–100) Over Time',
        ylabel='Health Score',
        filename='09_pots_health_score.png',
        ylim=(0, 100),
    )


# ── Plot 4: RGR grid ──────────────────────────────────────────────────────────

def plot_rgr(df):
    if 'rgr' not in df.columns or df['rgr'].dropna().empty:
        print("Skipping RGR plot — no data yet (needs at least 2 days).")
        return
    pot_grid_figure(
        df, metric='rgr',
        title='Per-Pot Relative Growth Rate (RGR) Over Time',
        ylabel='RGR (per day)',
        filename='10_pots_rgr.png',
        hline=0,
    )


# ── Plot 5: Spatial heatmap (latest day) ──────────────────────────────────────

def plot_heatmap(df):
    """
    Show canopy cover as a colour grid matching the physical pot layout.
    Uses the most recent day's data for each chamber.
    Pot layout mirrors the hive arrangement: 2 rows × 4 cols.
    """
    chambers_present = [c for c in CHAMBERS if c in df['chamber'].values]
    n = len(chambers_present)
    if n == 0:
        return

    latest_date = df['timestamp'].dt.date.max()
    latest_df   = df[df['timestamp'].dt.date == latest_date]

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, chamber in zip(axes, chambers_present):
        grid = np.full((GRID_ROWS, GRID_COLS), np.nan)

        for idx, pot_label in enumerate(POT_LABELS):
            row = idx // GRID_COLS
            col = idx  % GRID_COLS
            subset = latest_df[
                (latest_df['chamber']   == chamber) &
                (latest_df['pot_label'] == pot_label)
            ]['canopy_cover_%'].dropna()
            if not subset.empty:
                grid[row, col] = subset.iloc[-1]

        im = ax.imshow(grid, cmap='YlGn', vmin=0, vmax=100,
                       aspect='auto', interpolation='nearest')
        plt.colorbar(im, ax=ax, label='Canopy Cover (%)')

        # Label each cell with pot name and value
        for idx, pot_label in enumerate(POT_LABELS):
            row = idx // GRID_COLS
            col = idx  % GRID_COLS
            val = grid[row, col]
            cell_text = f"{pot_label}\n{val:.1f}%" if not np.isnan(val) else pot_label
            ax.text(col, row, cell_text, ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='black' if (np.isnan(val) or val < 60) else 'white')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{chamber.capitalize()} — {latest_date}", fontsize=11, fontweight='bold')

    fig.suptitle('Spatial Canopy Cover Heatmap (Latest Day)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_plot(fig, '11_pots_heatmap.png')


# ── Plot 6: Chamber summary — mean ± std across all pots ──────────────────────

def plot_summary(df):
    """
    Aggregate all pots per chamber per day (mean ± std) for key metrics.
    Shows the chamber-level signal with within-chamber variability as shading.
    """
    metrics = [
        ('canopy_cover_%',  'Canopy Cover (%)',    None),
        ('ngrdi_mean',      'Mean NGRDI',           None),
        ('health_score',    'Health Score (0–100)', (0, 100)),
        ('leaf_count',      'Leaf Count',           None),
    ]
    # Keep only metrics with actual data
    metrics = [(m, l, yl) for m, l, yl in metrics
               if m in df.columns and df[m].dropna().shape[0] > 0]

    if not metrics:
        print("Skipping summary plot — no numeric data yet.")
        return

    chambers_present = [c for c in CHAMBERS if c in df['chamber'].values]
    fig, axes = plt.subplots(len(metrics), 1,
                              figsize=(12, 4 * len(metrics)), sharex=False)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric, label, ylim) in zip(axes, metrics):
        for chamber in chambers_present:
            sub = df[df['chamber'] == chamber].copy()
            # Group by date (ignore time)
            sub['date'] = sub['timestamp'].dt.date
            grouped = sub.groupby('date')[metric].agg(['mean', 'std']).reset_index()
            grouped['date'] = pd.to_datetime(grouped['date'])
            grouped = grouped.dropna(subset=['mean'])
            if grouped.empty:
                continue

            ax.plot(grouped['date'], grouped['mean'],
                    marker='o', linewidth=2, markersize=5,
                    color=COLORS[chamber], label=chamber.capitalize())
            ax.fill_between(
                grouped['date'],
                grouped['mean'] - grouped['std'].fillna(0),
                grouped['mean'] + grouped['std'].fillna(0),
                alpha=0.15, color=COLORS[chamber],
            )

        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        if ylim:
            ax.set_ylim(ylim)

        n_days = (df['timestamp'].max() - df['timestamp'].min()).days
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        if n_days <= 21:
            ax.xaxis.set_major_locator(mdates.DayLocator())
        else:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

    axes[-1].set_xlabel('Date')
    fig.suptitle('Chamber Summary — Mean ± Std Across All Pots',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_plot(fig, '12_pots_summary.png')


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-pot growth dashboard for Growth Chamber CV."
    )
    parser.add_argument("--chamber", choices=["enriched", "control"],
                        help="Plot a single chamber only (default: both)")
    args = parser.parse_args()

    df = load_pot_data(chamber_filter=args.chamber)

    plot_canopy_cover(df)
    plot_ngrdi(df)
    plot_health_score(df)
    plot_rgr(df)
    plot_heatmap(df)
    plot_summary(df)

    print(f"\nAll plots saved to: {OUT_DIR}")
