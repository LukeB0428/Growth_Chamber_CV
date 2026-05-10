"""
visualise_results.py — Whole-Chamber Growth Metric Plots for Growth Chamber CV
EE496 | Luke Buckley | Maynooth University

Reads results/metrics.csv and saves six publication-quality time-series plots
comparing enriched vs control chambers across the trial period.

Plots produced (saved to results/plots/):
  01_canopy_cover.png      — Canopy Cover % over time
  02_exg.png               — Excess Green Index (ExG)
  03_vari.png              — VARI vegetation index
  04_rosette_diameter.png  — Rosette diameter in pixels
  05_rgr.png               — Relative Growth Rate per day
  06_summary.png           — All metrics stacked in one figure

Usage:
    python visualise_results.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from config import METRICS_CSV, PLOTS_DIR

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH  = str(METRICS_CSV)
OUT_DIR   = str(PLOTS_DIR)
CHAMBERS  = ["enriched", "control"]
COLORS    = {"enriched": "#2D6A4F", "control": "#74C69D"}

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load and clean data ───────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Coerce all metric columns to numeric, replacing bad values with NaN
metric_cols = ['canopy_cover_%', 'exg_mean', 'vari_mean',
               'rosette_diameter_px', 'rosette_area_px', 'rgr']
for col in metric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows where VARI is clearly corrupted (early bug, values like -183654)
if 'vari_mean' in df.columns:
    df.loc[df['vari_mean'] < -10, 'vari_mean'] = float('nan')

print(f"Loaded {len(df)} rows from CSV.")
print(f"Chambers found: {df['chamber'].unique().tolist()}")
print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
print()


def plot_metric(ax, df, metric, chamber, color, label):
    """Plot a single metric for a single chamber on a given axis."""
    subset = df[df['chamber'] == chamber][['timestamp', metric]].dropna()
    if subset.empty:
        return
    ax.plot(subset['timestamp'], subset[metric],
            marker='o', linewidth=2, markersize=5,
            color=color, label=label)


def save_plot(fig, filename):
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


def format_xaxis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


# ── Plot 1: Canopy Cover % ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for chamber in CHAMBERS:
    if chamber in df['chamber'].values:
        plot_metric(ax, df, 'canopy_cover_%', chamber,
                    COLORS[chamber], chamber.capitalize())
ax.set_title('Canopy Cover % Over Time', fontsize=14, fontweight='bold')
ax.set_ylabel('Canopy Cover (%)')
ax.set_xlabel('Date')
ax.legend()
ax.grid(True, alpha=0.3)
format_xaxis(ax)
fig.tight_layout()
save_plot(fig, '01_canopy_cover.png')


# ── Plot 2: ExG (Excess Green Index) ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for chamber in CHAMBERS:
    if chamber in df['chamber'].values:
        plot_metric(ax, df, 'exg_mean', chamber,
                    COLORS[chamber], chamber.capitalize())
ax.set_title('Excess Green Index (ExG) Over Time', fontsize=14, fontweight='bold')
ax.set_ylabel('Mean ExG')
ax.set_xlabel('Date')
ax.legend()
ax.grid(True, alpha=0.3)
format_xaxis(ax)
fig.tight_layout()
save_plot(fig, '02_exg.png')


# ── Plot 3: VARI ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for chamber in CHAMBERS:
    if chamber in df['chamber'].values:
        plot_metric(ax, df, 'vari_mean', chamber,
                    COLORS[chamber], chamber.capitalize())
ax.set_title('VARI (Vegetation Index) Over Time', fontsize=14, fontweight='bold')
ax.set_ylabel('Mean VARI')
ax.set_xlabel('Date')
ax.legend()
ax.grid(True, alpha=0.3)
format_xaxis(ax)
fig.tight_layout()
save_plot(fig, '03_vari.png')


# ── Plot 4: Rosette Diameter ──────────────────────────────────────────────────
if 'rosette_diameter_px' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    for chamber in CHAMBERS:
        if chamber in df['chamber'].values:
            plot_metric(ax, df, 'rosette_diameter_px', chamber,
                        COLORS[chamber], chamber.capitalize())
    ax.set_title('Rosette Diameter Over Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('Diameter (px)')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True, alpha=0.3)
    format_xaxis(ax)
    fig.tight_layout()
    save_plot(fig, '04_rosette_diameter.png')


# ── Plot 5: Relative Growth Rate ─────────────────────────────────────────────
if 'rgr' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    for chamber in CHAMBERS:
        if chamber in df['chamber'].values:
            plot_metric(ax, df, 'rgr', chamber,
                        COLORS[chamber], chamber.capitalize())
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_title('Relative Growth Rate (RGR) Over Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('RGR (per day)')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True, alpha=0.3)
    format_xaxis(ax)
    fig.tight_layout()
    save_plot(fig, '05_rgr.png')


# ── Plot 6: All metrics in one summary figure ─────────────────────────────────
available = [c for c in ['canopy_cover_%', 'exg_mean', 'vari_mean',
                          'rosette_diameter_px', 'rgr'] if c in df.columns]
titles = {
    'canopy_cover_%':     'Canopy Cover (%)',
    'exg_mean':           'ExG',
    'vari_mean':          'VARI',
    'rosette_diameter_px':'Rosette Diameter (px)',
    'rgr':                'RGR'
}

fig, axes = plt.subplots(len(available), 1,
                          figsize=(12, 4 * len(available)), sharex=True)
if len(available) == 1:
    axes = [axes]

for ax, metric in zip(axes, available):
    for chamber in CHAMBERS:
        if chamber in df['chamber'].values:
            plot_metric(ax, df, metric, chamber,
                        COLORS[chamber], chamber.capitalize())
    ax.set_ylabel(titles[metric])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if metric == 'rgr':
        ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')

axes[-1].set_xlabel('Date')
format_xaxis(axes[-1])
fig.suptitle('Growth Chamber CV — All Metrics Summary',
             fontsize=15, fontweight='bold', y=1.01)
fig.tight_layout()
save_plot(fig, '06_summary.png')

print("\nAll plots saved to:", OUT_DIR)
