"""
statistical_comparison.py — Enriched vs Control Statistical Comparison
EE496 | Luke Buckley | Maynooth University

Compares CV metrics between enriched (elevated CO2) and control (ambient CO2)
chambers using daily mean values per chamber.

Statistical tests:
  - Mann-Whitney U test (non-parametric, appropriate for small samples)
  - Effect size: Cohen's d

Outputs (saved to results/plots/):
  - growth_curves.png         — canopy cover over time with 95% CI shading
  - vegetation_indices.png    — NGRDI, VARI, EXG over time
  - rgr_comparison.png        — relative growth rate comparison
  - stats_summary.csv         — statistical test results table

Usage:
    python statistical_comparison.py
    python statistical_comparison.py --exclude-dead   # exclude dead/warning pots
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats as scipy_stats
from pathlib import Path

warnings.filterwarnings('ignore')

from config import POT_METRICS_CSV, RESULTS_DIR

PLOTS_DIR = RESULTS_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Plot style ─────────────────────────────────────────────────────────────────
ENRICHED_COLOUR = '#4CAF50'   # green
CONTROL_COLOUR  = '#9C27B0'   # purple
ALPHA_FILL      = 0.15
LINEWIDTH       = 2.0
FIGSIZE         = (10, 5)

plt.rcParams.update({
    'figure.facecolor': '#0a150a',
    'axes.facecolor':   '#0a150a',
    'axes.edgecolor':   '#4CAF50',
    'axes.labelcolor':  '#cccccc',
    'xtick.color':      '#cccccc',
    'ytick.color':      '#cccccc',
    'text.color':       '#cccccc',
    'grid.color':       '#1a2e1a',
    'grid.linestyle':   '--',
    'legend.facecolor': '#0a150a',
    'legend.edgecolor': '#4CAF50',
})


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(exclude_dead=False):
    df = pd.read_csv(POT_METRICS_CSV, on_bad_lines='skip')
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['date'] = df['timestamp'].dt.date

    # Coerce numeric columns
    skip_cols = {'timestamp', 'date', 'chamber', 'pot_label', 'image_file',
                 'image_path', 'method', 'plant_status', 'health_label',
                 'germination_date', 'bolting_date', 'bolting_signals', 'green_shade'}
    for c in df.columns:
        if c not in skip_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    if exclude_dead and 'plant_status' in df.columns:
        before = len(df)
        df = df[~df['plant_status'].isin(['dead', 'warning'])]
        print(f"  Excluded {before - len(df)} rows with dead/warning status")

    return df


def daily_stats(df, metric):
    """Return per-chamber daily mean ± SEM."""
    grouped = (df.groupby(['date', 'chamber'])[metric]
                 .agg(['mean', 'std', 'count'])
                 .reset_index())
    grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])
    grouped['ci95'] = grouped['sem'] * 1.96
    grouped['date'] = pd.to_datetime(grouped['date'])
    return grouped


# ── Statistical tests ──────────────────────────────────────────────────────────

def cohens_d(a, b):
    """Compute Cohen's d effect size."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_std = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 +
                          (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def run_stats(df, metric):
    """Run Mann-Whitney U test between enriched and control for a metric."""
    enriched = df[df['chamber'] == 'enriched'][metric].dropna()
    control  = df[df['chamber'] == 'control'][metric].dropna()

    if len(enriched) < 3 or len(control) < 3:
        return {'metric': metric, 'n_enriched': len(enriched), 'n_control': len(control),
                'mean_enriched': np.nan, 'mean_control': np.nan,
                'U_stat': np.nan, 'p_value': np.nan, 'significant': False,
                'cohens_d': np.nan, 'effect_size': 'insufficient data'}

    u_stat, p_val = scipy_stats.mannwhitneyu(enriched, control, alternative='two-sided')
    d = cohens_d(enriched.values, control.values)

    if abs(d) < 0.2:
        effect = 'negligible'
    elif abs(d) < 0.5:
        effect = 'small'
    elif abs(d) < 0.8:
        effect = 'medium'
    else:
        effect = 'large'

    return {
        'metric':        metric,
        'n_enriched':    len(enriched),
        'n_control':     len(control),
        'mean_enriched': round(float(enriched.mean()), 4),
        'mean_control':  round(float(control.mean()), 4),
        'U_stat':        round(float(u_stat), 2),
        'p_value':       round(float(p_val), 4),
        'significant':   p_val < 0.05,
        'cohens_d':      round(float(d), 3),
        'effect_size':   effect,
    }


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _format_ax(ax, title, ylabel, xlabel='Date'):
    ax.set_title(title, color='#4CAF50', pad=10)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')


def plot_metric_over_time(df, metric, title, ylabel, filename, stats_row=None):
    """Plot daily mean ± 95% CI for enriched vs control."""
    gdf = daily_stats(df, metric)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for chamber, colour, label in [
        ('enriched', ENRICHED_COLOUR, 'Enriched (elevated CO₂)'),
        ('control',  CONTROL_COLOUR,  'Control (ambient CO₂)'),
    ]:
        sub = gdf[gdf['chamber'] == chamber].sort_values('date')
        if sub.empty:
            continue
        ax.plot(sub['date'], sub['mean'], color=colour, lw=LINEWIDTH, label=label, marker='o', ms=4)
        ax.fill_between(sub['date'],
                        sub['mean'] - sub['ci95'],
                        sub['mean'] + sub['ci95'],
                        color=colour, alpha=ALPHA_FILL)

    # Annotate with stats if provided
    if stats_row is not None:
        p = stats_row['p_value']
        d = stats_row['cohens_d']
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        ax.text(0.02, 0.97,
                f"Mann-Whitney U  p={p:.3f} {sig}\nCohen's d={d:.2f} ({stats_row['effect_size']})",
                transform=ax.transAxes, va='top', fontsize=8,
                color='#cccccc',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a2e1a', edgecolor='#4CAF50', alpha=0.8))

    _format_ax(ax, title, ylabel)
    ax.legend(loc='upper left')
    fig.tight_layout()
    out = PLOTS_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_vegetation_indices(df, stats_dict):
    """3-panel figure: NGRDI, VARI, EXG."""
    metrics = [
        ('ngrdi_mean', 'NGRDI',    'NGRDI (mean)'),
        ('vari_mean',  'VARI',     'VARI (mean)'),
        ('exg_mean',   'ExG',      'ExG (mean)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (metric, label, ylabel) in zip(axes, metrics):
        gdf = daily_stats(df, metric)
        for chamber, colour, clabel in [
            ('enriched', ENRICHED_COLOUR, 'Enriched'),
            ('control',  CONTROL_COLOUR,  'Control'),
        ]:
            sub = gdf[gdf['chamber'] == chamber].sort_values('date')
            if sub.empty:
                continue
            ax.plot(sub['date'], sub['mean'], color=colour, lw=LINEWIDTH, label=clabel, marker='o', ms=4)
            ax.fill_between(sub['date'],
                            sub['mean'] - sub['ci95'],
                            sub['mean'] + sub['ci95'],
                            color=colour, alpha=ALPHA_FILL)

        sr = stats_dict.get(metric)
        if sr:
            p   = sr['p_value']
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            ax.set_title(f"{label}  (p={p:.3f} {sig})", color='#4CAF50', pad=8)
        else:
            ax.set_title(label, color='#4CAF50', pad=8)

        ax.set_ylabel(ylabel)
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        ax.legend(fontsize=8)

    fig.suptitle('Vegetation Indices — Enriched vs Control', color='#4CAF50', fontsize=13)
    fig.tight_layout()
    out = PLOTS_DIR / 'vegetation_indices.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_boxplot_comparison(df, metrics, filename, title):
    """Side-by-side boxplots for multiple metrics."""
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (metric, label) in zip(axes, metrics):
        enriched = df[df['chamber'] == 'enriched'][metric].dropna()
        control  = df[df['chamber'] == 'control'][metric].dropna()

        bp = ax.boxplot(
            [enriched, control],
            labels=['Enriched', 'Control'],
            patch_artist=True,
            medianprops=dict(color='white', lw=2),
            whiskerprops=dict(color='#cccccc'),
            capprops=dict(color='#cccccc'),
            flierprops=dict(marker='o', color='#cccccc', ms=4),
        )
        bp['boxes'][0].set_facecolor(ENRICHED_COLOUR)
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor(CONTROL_COLOUR)
        bp['boxes'][1].set_alpha(0.7)

        ax.set_title(label, color='#4CAF50')
        ax.grid(True, alpha=0.4, axis='y')

    fig.suptitle(title, color='#4CAF50', fontsize=13)
    fig.tight_layout()
    out = PLOTS_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(exclude_dead=False, start_date=None, end_date=None):
    print(f"\n=== Statistical Comparison — Enriched vs Control ===")
    if exclude_dead:
        print("  Mode: excluding dead/warning pots")

    df = load_data(exclude_dead=exclude_dead)

    # Date filtering
    if start_date:
        df = df[pd.to_datetime(df['date']) >= pd.to_datetime(start_date)]
        print(f"  Filter: from {start_date}")
    if end_date:
        df = df[pd.to_datetime(df['date']) <= pd.to_datetime(end_date)]
        print(f"  Filter: to {end_date}")

    print(f"  Loaded {len(df)} rows | {df['date'].nunique()} days | "
          f"enriched: {(df['chamber']=='enriched').sum()} rows, "
          f"control: {(df['chamber']=='control').sum()} rows")

    # Metrics to test
    test_metrics = [
        'canopy_cover_%',
        'ngrdi_mean',
        'vari_mean',
        'exg_mean',
        'rgr',
        'health_score',
        'greenness_score',
        'gcc',
    ]

    # Run stats
    results = []
    stats_dict = {}
    for m in test_metrics:
        if m in df.columns and df[m].notna().sum() > 5:
            r = run_stats(df, m)
            results.append(r)
            stats_dict[m] = r

    # Save summary CSV — always write headers so dashboard never reads an empty file
    COLS = ['metric', 'n_enriched', 'n_control', 'mean_enriched', 'mean_control',
            'U_stat', 'p_value', 'significant', 'cohens_d', 'effect_size']
    stats_df = pd.DataFrame(results, columns=COLS) if results else pd.DataFrame(columns=COLS)
    out_csv = RESULTS_DIR / 'stats_summary.csv'
    stats_df.to_csv(out_csv, index=False)
    print(f"\n  Stats summary saved to {out_csv}")
    if not stats_df.empty:
        print()
        print(stats_df[['metric', 'mean_enriched', 'mean_control', 'p_value', 'significant', 'cohens_d', 'effect_size']].to_string(index=False))
    else:
        print("  No data in selected date range — empty results written.")

    # Plot growth curves
    print("\n  Generating plots...")
    plot_metric_over_time(
        df, 'canopy_cover_%',
        'Canopy Cover Over Time — Enriched vs Control',
        'Canopy Cover (%)',
        'growth_curves.png',
        stats_row=stats_dict.get('canopy_cover_%'),
    )

    plot_vegetation_indices(df, stats_dict)

    if df['rgr'].notna().sum() > 10:
        plot_metric_over_time(
            df, 'rgr',
            'Relative Growth Rate — Enriched vs Control',
            'RGR (day⁻¹)',
            'rgr_comparison.png',
            stats_row=stats_dict.get('rgr'),
        )

    plot_boxplot_comparison(
        df,
        [('canopy_cover_%', 'Canopy Cover (%)'),
         ('ngrdi_mean',     'NGRDI'),
         ('exg_mean',       'ExG')],
        'metric_boxplots.png',
        'Metric Distribution — Enriched vs Control',
    )

    if 'health_score' in df.columns and df['health_score'].notna().sum() > 10:
        plot_metric_over_time(
            df, 'health_score',
            'Health Score Over Time — Enriched vs Control',
            'Health Score (0–100)',
            'health_score_comparison.png',
            stats_row=stats_dict.get('health_score'),
        )

    print(f"\nDone. All figures saved to {PLOTS_DIR}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exclude-dead', action='store_true',
                        help='Exclude pots flagged as dead or warning from analysis')
    parser.add_argument('--start-date', default=None, help='Filter from date (YYYY-MM-DD)')
    parser.add_argument('--end-date',   default=None, help='Filter to date (YYYY-MM-DD)')
    args = parser.parse_args()
    main(exclude_dead=args.exclude_dead, start_date=args.start_date, end_date=args.end_date)
