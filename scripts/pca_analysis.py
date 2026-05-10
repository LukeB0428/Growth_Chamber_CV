"""
pca_analysis.py — PCA of CV metrics, enriched vs control
EE496 | Luke Buckley | Maynooth University

Runs Principal Component Analysis on per-pot CV metrics to visualise
whether elevated CO2 (enriched) and control plants separate in metric space.

Outputs (saved to results/plots/):
  - pca_scatter.png       — PC1 vs PC2 scatter coloured by chamber
  - pca_loadings.png      — top feature contributions to PC1 and PC2
  - pca_variance.png      — explained variance per component (scree plot)
  - pca_trajectory.png    — mean PC1/PC2 per chamber over time

Usage:
    python pca_analysis.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

warnings.filterwarnings('ignore')

from config import POT_METRICS_CSV, RESULTS_DIR

PLOTS_DIR = RESULTS_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ENRICHED_COLOUR = '#4CAF50'
CONTROL_COLOUR  = '#9C27B0'

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

# Metrics to include — meaningful, non-redundant
PCA_METRICS = [
    'canopy_cover_%',
    'ngrdi_mean',
    'vari_mean',
    'exg_mean',
    'rosette_diameter_px',
    'rgr',
    'mean_hue',
    'mean_saturation',
    'gcc',
    'lab_a',       # green-red axis in Lab space
    'greenness_score',
    'health_score',
]


def load_data():
    df = pd.read_csv(POT_METRICS_CSV, on_bad_lines='skip')
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['date'] = df['timestamp'].dt.date

    skip = {'timestamp', 'date', 'chamber', 'pot_label', 'image_file',
            'image_path', 'method', 'plant_status', 'health_label',
            'germination_date', 'bolting_date', 'bolting_signals', 'green_shade'}
    for c in df.columns:
        if c not in skip:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    return df


def run_pca(df):
    # Keep only rows with enough data
    available = [m for m in PCA_METRICS if m in df.columns]
    sub = df[['chamber', 'date', 'pot_label'] + available].dropna(subset=available)

    print(f"  PCA input: {len(sub)} rows, {len(available)} features")

    X = sub[available].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(len(available), 6))
    X_pca = pca.fit_transform(X_scaled)

    result = sub[['chamber', 'date', 'pot_label']].copy().reset_index(drop=True)
    for i in range(X_pca.shape[1]):
        result[f'PC{i+1}'] = X_pca[:, i]

    return result, pca, available, scaler


def plot_scatter(result, pca):
    fig, ax = plt.subplots(figsize=(9, 6))

    for chamber, colour, label in [
        ('enriched', ENRICHED_COLOUR, 'Enriched (elevated CO₂)'),
        ('control',  CONTROL_COLOUR,  'Control (ambient CO₂)'),
    ]:
        sub = result[result['chamber'] == chamber]
        ax.scatter(sub['PC1'], sub['PC2'], c=colour, label=label,
                   alpha=0.6, s=40, edgecolors='none')

    # Confidence ellipses
    for chamber, colour in [('enriched', ENRICHED_COLOUR), ('control', CONTROL_COLOUR)]:
        sub = result[result['chamber'] == chamber][['PC1', 'PC2']].values
        if len(sub) < 3:
            continue
        mean = sub.mean(axis=0)
        cov  = np.cov(sub.T)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * 1.96 * np.sqrt(vals)
        from matplotlib.patches import Ellipse
        ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      edgecolor=colour, facecolor='none', lw=1.5, linestyle='--', alpha=0.7)
        ax.add_patch(ell)

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f'PC1 ({var1:.1f}% variance explained)')
    ax.set_ylabel(f'PC2 ({var2:.1f}% variance explained)')
    ax.set_title('PCA — Enriched vs Control Separation', color='#4CAF50', pad=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = PLOTS_DIR / 'pca_scatter.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_loadings(pca, feature_names):
    loadings = pca.components_[:2]  # PC1 and PC2
    n = len(feature_names)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, pc_idx, title in zip(axes, [0, 1], ['PC1 Loadings', 'PC2 Loadings']):
        loads = loadings[pc_idx]
        order = np.argsort(np.abs(loads))[::-1]
        names = [feature_names[i] for i in order]
        vals  = [loads[i] for i in order]
        colours = [ENRICHED_COLOUR if v > 0 else CONTROL_COLOUR for v in vals]
        bars = ax.barh(range(n), vals, color=colours, alpha=0.8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(names, fontsize=9)
        ax.axvline(0, color='#cccccc', lw=0.8)
        ax.set_title(title, color='#4CAF50')
        ax.set_xlabel('Loading coefficient')
        ax.grid(True, alpha=0.3, axis='x')

    fig.suptitle('PCA Feature Contributions', color='#4CAF50', fontsize=13)
    fig.tight_layout()
    out = PLOTS_DIR / 'pca_loadings.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_scree(pca):
    var = pca.explained_variance_ratio_ * 100
    cumvar = np.cumsum(var)
    components = range(1, len(var) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(components, var, color=ENRICHED_COLOUR, alpha=0.7, label='Individual')
    ax.plot(components, cumvar, color=CONTROL_COLOUR, marker='o', lw=2, label='Cumulative')
    ax.axhline(80, color='#cccccc', lw=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('Scree Plot — Explained Variance', color='#4CAF50', pad=10)
    ax.set_xticks(list(components))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = PLOTS_DIR / 'pca_variance.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_trajectory(result):
    """Mean PC1 per chamber over time — shows divergence trend."""
    result = result.copy()
    result['date'] = pd.to_datetime(result['date'])
    daily = result.groupby(['date', 'chamber'])[['PC1', 'PC2']].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    for chamber, colour, label in [
        ('enriched', ENRICHED_COLOUR, 'Enriched (elevated CO₂)'),
        ('control',  CONTROL_COLOUR,  'Control (ambient CO₂)'),
    ]:
        sub = daily[daily['chamber'] == chamber].sort_values('date')
        ax.plot(sub['date'], sub['PC1'], color=colour, lw=2, marker='o', ms=5, label=label)

    ax.set_xlabel('Date')
    ax.set_ylabel('Mean PC1 Score')
    ax.set_title('PC1 Trajectory Over Time — Enriched vs Control', color='#4CAF50', pad=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = PLOTS_DIR / 'pca_trajectory.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def main(start_date=None, end_date=None):
    print("\n=== PCA Analysis — Enriched vs Control ===")

    df = load_data()

    if start_date:
        df = df[pd.to_datetime(df['date']) >= pd.to_datetime(start_date)]
        print(f"  Filter: from {start_date}")
    if end_date:
        df = df[pd.to_datetime(df['date']) <= pd.to_datetime(end_date)]
        print(f"  Filter: to {end_date}")

    result, pca, features, scaler = run_pca(df)

    print(f"\n  Explained variance per component:")
    for i, v in enumerate(pca.explained_variance_ratio_):
        print(f"    PC{i+1}: {v*100:.1f}%")

    print("\n  Generating plots...")
    plot_scatter(result, pca)
    plot_loadings(pca, features)
    plot_scree(pca)
    plot_trajectory(result)

    print(f"\nDone. All figures saved to {PLOTS_DIR}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', default=None, help='Filter from date (YYYY-MM-DD)')
    parser.add_argument('--end-date',   default=None, help='Filter to date (YYYY-MM-DD)')
    args = parser.parse_args()
    main(start_date=args.start_date, end_date=args.end_date)
