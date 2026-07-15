"""
statistical_comparison.py — Enriched vs Control Statistical Comparison
EE496 | Luke Buckley | Maynooth University

Compares CV metrics between enriched (elevated CO2) and control (ambient CO2)
chambers. The experimental UNIT is the pot: pot-day rows are collapsed to one
mean per pot (8 pots per chamber) before any test is run. This removes the
day-to-day temporal autocorrelation that would otherwise treat ~47 repeated
measures of the same 8 pots as independent samples — which inflates n and
badly understates p-values (pseudoreplication).

Statistical tests:
  - Mann-Whitney U test (non-parametric), pot-level, two-sided
  - Effect size: Cliff's delta (rank-based — consistent with Mann-Whitney;
    replaces the parametric Cohen's d used previously)
  - Multiple metrics are corrected with Benjamini-Hochberg (FDR). The
    `significant` flag reflects the FDR-ADJUSTED p-value, not the raw one.
  - Reported both overall and per developmental stage.

CAVEAT — design-level pseudoreplication (read before quoting a p-value):
  CO2 is applied at the CHAMBER level and there is only ONE chamber per
  treatment, so the 8 pots are sub-samples, not true independent replicates of
  the CO2 effect. Pot-level tests describe THIS enriched-vs-control chamber
  pair; formal causal inference to "elevated CO2 causes X" would require
  chamber-level replication. The per-stage windows below are DESCRIPTIVE
  developmental phases (chosen from the observed greenness curve), so per-stage
  results are EXPLORATORY, not confirmatory.

Outputs (saved to results/):
  - stats_summary.csv         — overall pot-level test results (FDR-adjusted)
  - stats_by_stage.csv        — same tests within each developmental stage
Outputs (saved to results/plots/):
  - growth_curves.png         — canopy cover over time with 95% CI shading
  - vegetation_indices.png    — NGRDI, VARI, EXG over time
  - rgr_comparison.png        — relative growth rate comparison

Usage:
    python statistical_comparison.py
    python statistical_comparison.py --exclude-dead   # exclude dead/warning pots
"""

import argparse
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats as scipy_stats
from scipy.stats import false_discovery_control
from pathlib import Path

warnings.filterwarnings('ignore')

# Windows consoles default to cp1252 and choke on the unicode used below
# (≤, —, δ). Force UTF-8 so console + captured-subprocess output never crash.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

from config import POT_METRICS_CSV, RESULTS_DIR

PLOTS_DIR = RESULTS_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Analysis parameters ──────────────────────────────────────────────────────
# Pot is the unit of replication → at most 8 values per chamber. Mann-Whitney U
# on 8-vs-8 can reach p<0.05; on 4-vs-4 the smallest attainable two-sided p is
# ~0.028, and on 3-vs-3 it is ~0.10 (can NEVER be significant). Require at least
# this many pots per chamber so a test is not run when it cannot be meaningful.
MIN_POTS = 5

# Descriptive developmental windows from the project README (greenness phases).
# These are POST-HOC / exploratory boundaries — see the caveat in the docstring.
STAGES = [
    ('Stage 1', '2026-04-09', '2026-04-20'),
    ('Stage 2', '2026-04-21', '2026-05-12'),
    ('Stage 3', '2026-05-13', '2026-06-01'),
]

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

RESULT_COLS = ['metric', 'n_enriched', 'n_control', 'mean_enriched', 'mean_control',
               'U_stat', 'p_value', 'p_value_adj', 'significant',
               'cliffs_delta', 'effect_size']


def aggregate_per_pot(df, metric, start=None, end=None):
    """Collapse pot-day rows to ONE mean per pot (the experimental unit),
    optionally within a [start, end] date window. Returns (enriched, control)
    arrays of per-pot means — this is what makes the test valid (removes the
    pot-day temporal autocorrelation that inflates n)."""
    sub = df
    if start is not None:
        sub = sub[pd.to_datetime(sub['date']) >= pd.to_datetime(start)]
    if end is not None:
        sub = sub[pd.to_datetime(sub['date']) <= pd.to_datetime(end)]

    per_pot = (sub.dropna(subset=[metric])
                  .groupby(['chamber', 'pot_label'])[metric]
                  .mean())
    chambers = per_pot.index.get_level_values(0)
    enriched = per_pot[chambers == 'enriched'].to_numpy()
    control  = per_pot[chambers == 'control'].to_numpy()
    return enriched, control


def cliffs_delta(a, b):
    """Cliff's delta — rank-based effect size that matches Mann-Whitney U.
    Range [-1, 1]: fraction of (a>b) pairs minus fraction of (a<b) pairs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 1 or len(b) < 1:
        return np.nan
    return float(np.sign(a[:, None] - b[None, :]).mean())


def cliffs_magnitude(delta):
    """Romano et al. (2006) thresholds for |Cliff's delta|."""
    if delta is None or np.isnan(delta):
        return 'insufficient data'
    ad = abs(delta)
    if ad < 0.147:
        return 'negligible'
    if ad < 0.33:
        return 'small'
    if ad < 0.474:
        return 'medium'
    return 'large'


def run_stats(df, metric, start=None, end=None):
    """Mann-Whitney U between enriched and control on PER-POT aggregated values.
    `significant` is filled in later by apply_fdr() from the adjusted p-value."""
    enriched, control = aggregate_per_pot(df, metric, start, end)
    n_e, n_c = len(enriched), len(control)

    row = {'metric': metric, 'n_enriched': n_e, 'n_control': n_c,
           'mean_enriched': np.nan, 'mean_control': np.nan,
           'U_stat': np.nan, 'p_value': np.nan, 'p_value_adj': np.nan,
           'significant': False, 'cliffs_delta': np.nan,
           'effect_size': 'insufficient data'}

    if n_e < MIN_POTS or n_c < MIN_POTS:
        return row

    u_stat, p_val = scipy_stats.mannwhitneyu(enriched, control, alternative='two-sided')
    delta = cliffs_delta(enriched, control)
    row.update({
        'mean_enriched': round(float(np.mean(enriched)), 4),
        'mean_control':  round(float(np.mean(control)), 4),
        'U_stat':        round(float(u_stat), 2),
        'p_value':       round(float(p_val), 4),
        'cliffs_delta':  round(float(delta), 3),
        'effect_size':   cliffs_magnitude(delta),
    })
    return row


def apply_fdr(results, alpha=0.05):
    """Benjamini-Hochberg across the metric family. Sets p_value_adj and the
    `significant` flag from the ADJUSTED p. Metrics with no p (insufficient
    pots) are excluded from the correction and stay non-significant."""
    idx = [i for i, r in enumerate(results) if r['p_value'] is not None and not np.isnan(r['p_value'])]
    if idx:
        pv  = np.array([results[i]['p_value'] for i in idx], dtype=float)
        adj = false_discovery_control(pv, method='bh')
        for j, i in enumerate(idx):
            results[i]['p_value_adj'] = round(float(adj[j]), 4)
            results[i]['significant'] = bool(adj[j] < alpha)
    return results


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

    # Annotate with the overall pot-level stat if provided
    if stats_row is not None and not pd.isna(stats_row.get('p_value_adj')):
        p = stats_row['p_value_adj']
        d = stats_row['cliffs_delta']
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        ax.text(0.02, 0.97,
                f"Mann-Whitney U (per pot)  p(FDR)={p:.3f} {sig}\n"
                f"Cliff's δ={d:.2f} ({stats_row['effect_size']})",
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
        if sr and not pd.isna(sr.get('p_value_adj')):
            p   = sr['p_value_adj']
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            ax.set_title(f"{label}  (p(FDR)={p:.3f} {sig})", color='#4CAF50', pad=8)
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

    def run_family(sub_start=None, sub_end=None):
        """Run every metric over one window, then FDR-correct the family."""
        rows = []
        for m in test_metrics:
            if m in df.columns and df[m].notna().sum() > 5:
                rows.append(run_stats(df, m, sub_start, sub_end))
        return apply_fdr(rows)

    # ── Overall test (pot-level, FDR-corrected) ────────────────────────────────
    results = run_family()
    stats_dict = {r['metric']: r for r in results}

    stats_df = (pd.DataFrame(results, columns=RESULT_COLS)
                if results else pd.DataFrame(columns=RESULT_COLS))
    out_csv = RESULTS_DIR / 'stats_summary.csv'
    stats_df.to_csv(out_csv, index=False)
    print(f"\n  Overall pot-level stats saved to {out_csv}")
    print(f"  Unit of replication: pot (n≤8 per chamber). p-values are "
          f"Benjamini-Hochberg FDR-adjusted; 'significant' uses the adjusted p.")
    if not stats_df.empty:
        print()
        print(stats_df[['metric', 'n_enriched', 'n_control', 'mean_enriched',
                        'mean_control', 'p_value', 'p_value_adj', 'significant',
                        'cliffs_delta', 'effect_size']].to_string(index=False))
    else:
        print("  No data in selected date range — empty results written.")

    # ── Per-stage exploratory tests ────────────────────────────────────────────
    stage_rows = []
    for name, s, e in STAGES:
        fam = run_family(s, e)
        for r in fam:
            stage_rows.append({'stage': name, **r})
    stage_df = (pd.DataFrame(stage_rows, columns=['stage'] + RESULT_COLS)
                if stage_rows else pd.DataFrame(columns=['stage'] + RESULT_COLS))
    stage_csv = RESULTS_DIR / 'stats_by_stage.csv'
    stage_df.to_csv(stage_csv, index=False)
    print(f"\n  Per-stage (exploratory) stats saved to {stage_csv}")
    if not stage_df.empty:
        print(stage_df[['stage', 'metric', 'mean_enriched', 'mean_control',
                        'p_value', 'p_value_adj', 'significant',
                        'cliffs_delta']].to_string(index=False))

    # ── Honest caveat, printed every run ───────────────────────────────────────
    print("\n  " + "-" * 70)
    print("  CAVEAT: CO2 is applied at chamber level with ONE chamber per")
    print("  treatment, so pots are sub-samples, not true replicates of the CO2")
    print("  effect. These tests describe THIS chamber pair. Per-stage windows")
    print("  are post-hoc/descriptive → treat per-stage results as exploratory.")
    print("  " + "-" * 70)

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
