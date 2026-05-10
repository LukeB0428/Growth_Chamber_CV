"""
dashboard.py — Growth Chamber CV — Streamlit Dashboard
EE496 | Luke Buckley | Maynooth University

Local web dashboard for monitoring the Arabidopsis growth chamber trial.
Reads from results/metrics.csv, results/pot_metrics.csv, and
results/ground_truth.csv produced by the analysis pipeline.

Usage:
    scripts\.venv\Scripts\streamlit run scripts/dashboard.py

Then open http://localhost:8501 in your browser.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import subprocess
import sys
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from config import (BASE_DIR, METRICS_CSV, POT_METRICS_CSV, GROUND_TRUTH_CSV,
                    IMAGES_DIR, PYTHON_BIN, ANALYSE_SCRIPT, CALIB_DIR)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ── Paths ─────────────────────────────────────────────────────────────────────
POT_CSV     = POT_METRICS_CSV
GT_CSV      = GROUND_TRUTH_CSV
PYTHON      = PYTHON_BIN
ANALYSE     = ANALYSE_SCRIPT

CHAMBERS   = ["enriched", "control"]
POT_LABELS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
COLORS     = {"enriched": "#4CAF50", "control": "#9C27B0"}

PLOTLY_TEMPLATE = "plotly_dark"
CHART_BG        = "rgba(13,34,13,0.0)"   # transparent so theme bg shows


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Growth Chamber CV",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — tighten metric cards, style the header, fix table colours
st.markdown("""
<style>
    /* Header bar */
    .main-header {
        background: linear-gradient(90deg, #1a3a1a 0%, #0d260d 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        color: #e8f5e9;
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .main-header p {
        color: #81C784;
        margin: 0.2rem 0 0 0;
        font-size: 0.85rem;
    }
    /* Metric cards */
    [data-testid="metric-container"] {
        background: #132213;
        border: 1px solid #2e4a2e;
        border-radius: 8px;
        padding: 0.8rem 1rem;
    }
    [data-testid="metric-container"] label {
        color: #81C784 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e8f5e9 !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
    }
    /* Section divider */
    .section-title {
        color: #4CAF50;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 1px solid #2e4a2e;
        padding-bottom: 0.4rem;
        margin: 1.2rem 0 0.8rem 0;
    }
    /* Status pill */
    .status-ok   { background:#1b3d1b; color:#4CAF50; padding:2px 10px;
                   border-radius:12px; font-size:0.8rem; font-weight:600; }
    .status-warn { background:#3d2e0a; color:#FFC107; padding:2px 10px;
                   border-radius:12px; font-size:0.8rem; font-weight:600; }
    .status-none { background:#2a1a1a; color:#EF5350; padding:2px 10px;
                   border-radius:12px; font-size:0.8rem; font-weight:600; }
    /* Log output box */
    .log-box {
        background: #060f06;
        border: 1px solid #2e4a2e;
        border-radius: 6px;
        padding: 1rem;
        font-family: monospace;
        font-size: 0.8rem;
        color: #a5d6a7;
        max-height: 400px;
        overflow-y: auto;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loaders (cached) ─────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_metrics():
    df = pd.DataFrame()
    if METRICS_CSV.exists():
        try:
            df = pd.read_csv(METRICS_CSV, on_bad_lines='skip')
        except Exception:
            df = pd.read_csv(METRICS_CSV, engine='python', on_bad_lines='skip')
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            numeric = [c for c in df.columns if c not in ('timestamp', 'chamber', 'image_path', 'method')]
            for c in numeric:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            return df

    # metrics.csv is empty or missing — aggregate pot_metrics.csv per chamber per day
    pot_df = load_pot_metrics()
    if pot_df.empty:
        return pd.DataFrame()
    skip = {'timestamp', 'chamber', 'pot_label', 'image_file', 'image_path', 'method',
            'germination_date', 'bolting_date', 'bolting_signals', 'green_shade', 'health_label',
            'plant_status'}
    numeric = [c for c in pot_df.columns if c not in skip]
    pot_df['_date'] = pot_df['timestamp'].dt.date
    agg = pot_df.groupby(['_date', 'chamber'])[numeric].mean().reset_index()
    agg['timestamp'] = pd.to_datetime(agg['_date'])
    agg.drop(columns=['_date'], inplace=True)
    return agg


@st.cache_data(ttl=60)
def load_pot_metrics():
    if not POT_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(POT_CSV, on_bad_lines='skip')
    except Exception:
        df = pd.read_csv(POT_CSV, engine='python', on_bad_lines='skip')
    if df.empty:
        return df
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    numeric = [c for c in df.columns if c not in ('timestamp', 'chamber', 'pot_label', 'image_path', 'method', 'plant_status', 'health_label', 'image_file')]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


@st.cache_data(ttl=60)
def load_ground_truth():
    if not GT_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(GT_CSV)
    df['date'] = pd.to_datetime(df['date'])
    return df


def invalidate_cache():
    load_metrics.clear()
    load_pot_metrics.clear()
    load_ground_truth.clear()


# ── Chart helpers ─────────────────────────────────────────────────────────────

def apply_chart_style(fig, title="", height=380):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        title=dict(text=title, font=dict(size=14, color="#e8f5e9")),
        height=height,
        margin=dict(l=50, r=20, t=45, b=50),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2e4a2e", borderwidth=1),
        xaxis=dict(gridcolor="#1e3a1e", zerolinecolor="#2e4a2e"),
        yaxis=dict(gridcolor="#1e3a1e", zerolinecolor="#2e4a2e"),
        font=dict(color="#c8e6c9"),
    )
    return fig


def metric_line_chart(df, metric, title, yaxis_label, date_range=None):
    if df.empty or metric not in df.columns:
        return None
    if date_range:
        df = df[(df['timestamp'].dt.date >= date_range[0]) &
                (df['timestamp'].dt.date <= date_range[1])]
    fig = go.Figure()
    for chamber in CHAMBERS:
        sub = df[df['chamber'] == chamber][['timestamp', metric]].dropna()
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub['timestamp'], y=sub[metric],
            mode='lines+markers',
            name=chamber.capitalize(),
            line=dict(color=COLORS[chamber], width=2),
            marker=dict(size=6),
        ))
    fig.update_yaxes(title_text=yaxis_label)
    return apply_chart_style(fig, title)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 0.5rem 0 1rem 0;'>
        <span style='font-size:2.5rem;'>🌱</span>
        <h2 style='color:#4CAF50; margin:0.2rem 0 0 0; font-size:1.1rem;'>Growth Chamber CV</h2>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Overview", "Growth Trends", "Per-Pot Dashboard", "Ground Truth", "Log Readings", "Run Analysis", "Live View", "Timelapse", "Statistics", "Metrics"],
        label_visibility="collapsed",
    )

    st.divider()

    # Trial info
    df_side = load_metrics()
    if not df_side.empty:
        first_day = df_side['timestamp'].min().date()
        last_day  = df_side['timestamp'].max().date()
        n_days    = (last_day - first_day).days + 1
        st.markdown(f"**Trial start:** {first_day}")
        st.markdown(f"**Last capture:** {last_day}")
        st.markdown(f"**Days running:** {n_days}")
    else:
        st.markdown("*No trial data yet*")

    st.divider()
    if st.button("Refresh Data", use_container_width=True):
        invalidate_cache()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "Overview":
    st.markdown("""
    <div class='main-header'>
        <h1>Overview</h1>
        <p>Trial status and daily metrics summary</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_metrics()

    if df.empty:
        st.info("No data yet — run an analysis first using the **Run Analysis** page.")
        st.stop()

    # Controls row
    ctrl1, ctrl2 = st.columns([2, 1])
    with ctrl1:
        available_dates = sorted(df['timestamp'].dt.date.unique(), reverse=True)
        selected_date = st.selectbox(
            "View metrics for date:",
            available_dates,
            format_func=lambda d: d.strftime("%A, %d %B %Y") + (" (latest)" if d == available_dates[0] else ""),
        )
    with ctrl2:
        selected_chambers = st.multiselect(
            "Chambers",
            CHAMBERS,
            default=CHAMBERS,
            key="overview_chambers",
        )
    if not selected_chambers:
        st.warning("Select at least one chamber.")
        st.stop()

    day_df = df[
        (df['timestamp'].dt.date == selected_date) &
        (df['chamber'].isin(selected_chambers))
    ]

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Canopy Cover</div>", unsafe_allow_html=True)
    n_ch  = len(selected_chambers)
    cols  = st.columns(n_ch * 2)

    for i, chamber in enumerate(selected_chambers):
        row = day_df[day_df['chamber'] == chamber]
        if row.empty:
            cols[i].metric(f"{chamber.capitalize()} Canopy", "—")
        else:
            val = row['canopy_cover_%'].values[0]
            prev_dates = [d for d in available_dates if d < selected_date]
            delta_str  = None
            if prev_dates:
                prev_df = df[(df['timestamp'].dt.date == prev_dates[0]) &
                             (df['chamber'] == chamber)]
                if not prev_df.empty and not pd.isna(prev_df['canopy_cover_%'].values[0]):
                    delta = val - prev_df['canopy_cover_%'].values[0]
                    delta_str = f"{delta:+.1f}% vs prev day"
            cols[i].metric(
                label=f"{chamber.capitalize()} Canopy",
                value=f"{val:.1f}%" if not pd.isna(val) else "—",
                delta=delta_str,
            )

    for i, chamber in enumerate(selected_chambers):
        row = day_df[day_df['chamber'] == chamber]
        val = row['ngrdi_mean'].values[0] if not row.empty and 'ngrdi_mean' in row.columns else None
        cols[i + n_ch].metric(
            label=f"{chamber.capitalize()} NGRDI",
            value=f"{val:.3f}" if val is not None and not pd.isna(val) else "—",
        )

    # ── Health + depth cards ──────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Plant Health & 3D Metrics</div>", unsafe_allow_html=True)
    cols2 = st.columns(4)

    health_metrics = [
        ('health_score',         'Health Score',     '{:.0f}/100'),
        ('lai',                  'LAI',              '{:.2f}'),
        ('canopy_height_mean_mm','Canopy Height',    '{:.0f} mm'),
        ('leaf_count',           'Leaf Count',       '{:.0f}'),
    ]

    for i, (col_name, label, fmt) in enumerate(health_metrics):
        vals = []
        for chamber in selected_chambers:
            row = day_df[day_df['chamber'] == chamber]
            if not row.empty and col_name in row.columns:
                v = row[col_name].values[0]
                vals.append(f"{chamber[:3].upper()}: {fmt.format(v)}" if not pd.isna(v) else f"{chamber[:3].upper()}: —")
            else:
                vals.append(f"{chamber[:3].upper()}: —")
        cols2[i].metric(label=label, value=vals[0], delta=vals[1] if len(vals) > 1 else None)

    # ── Bolting status ────────────────────────────────────────────────────────
    if 'bolting_flag' in day_df.columns:
        st.markdown("<div class='section-title'>Bolting Status</div>", unsafe_allow_html=True)
        bolt_cols = st.columns(len(selected_chambers))
        for col, chamber in zip(bolt_cols, selected_chambers):
            row = day_df[day_df['chamber'] == chamber]
            if not row.empty:
                flag = row['bolting_flag'].values[0]
                date = row['bolting_date'].values[0] if 'bolting_date' in row.columns else None
                signals = row['bolting_signals'].values[0] if 'bolting_signals' in row.columns else None
                if pd.notna(flag) and int(flag) == 1:
                    date_str = f" — detected {date}" if pd.notna(date) else ""
                    sig_str  = f"Signals: {signals}" if pd.notna(signals) else ""
                    col.error(f"🌱 BOLTING — {chamber.capitalize()}{date_str}\n{sig_str}")
                else:
                    col.success(f"No bolting — {chamber.capitalize()}")

    # ── Full metrics table ────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Full Metrics Table</div>", unsafe_allow_html=True)

    display_cols = [c for c in [
        'chamber', 'canopy_cover_%', 'exg_mean', 'vari_mean', 'ngrdi_mean',
        'lai', 'leaf_count', 'health_score',
        'canopy_height_mean_mm', 'canopy_volume_cm3',
        'chlorosis_pct', 'necrosis_pct', 'rgr',
        'bolting_flag', 'bolting_date', 'bolting_signals',
    ] if c in day_df.columns]

    st.dataframe(
        day_df[display_cols].set_index('chamber').round(3),
        use_container_width=True,
    )

    # ── Mini trend (last 7 days canopy cover) ─────────────────────────────────
    st.markdown("<div class='section-title'>Last 7 Days — Canopy Cover</div>", unsafe_allow_html=True)
    cutoff = selected_date - timedelta(days=7)
    recent = df[
        (df['timestamp'].dt.date >= cutoff) &
        (df['chamber'].isin(selected_chambers))
    ]
    fig = metric_line_chart(recent, 'canopy_cover_%', '', 'Canopy Cover (%)')
    if fig:
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: GROWTH TRENDS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Growth Trends":
    st.markdown("""
    <div class='main-header'>
        <h1>Growth Trends</h1>
        <p>Time-series comparison of enriched vs control chambers</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_metrics()

    if df.empty:
        st.info("No data yet — run an analysis first.")
        st.stop()

    # Controls
    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        date_range = st.date_input(
            "Date range",
            value=(df['timestamp'].min().date(), df['timestamp'].max().date()),
            min_value=df['timestamp'].min().date(),
            max_value=df['timestamp'].max().date(),
        )
        if len(date_range) != 2:
            st.stop()

    with col_ctrl2:
        chambers_sel = st.multiselect(
            "Chambers",
            CHAMBERS,
            default=CHAMBERS,
        )

    if not chambers_sel:
        st.warning("Select at least one chamber.")
        st.stop()

    df_filt = df[
        (df['timestamp'].dt.date >= date_range[0]) &
        (df['timestamp'].dt.date <= date_range[1]) &
        (df['chamber'].isin(chambers_sel))
    ]

    TREND_METRICS = {
        'canopy_cover_%':         'Canopy Cover (%)',
        'exg_mean':               'Excess Green Index (ExG)',
        'vari_mean':              'VARI',
        'ngrdi_mean':             'NGRDI',
        'lai':                    'Leaf Area Index (LAI)',
        'leaf_count':             'Leaf Count',
        'rgr':                    'Relative Growth Rate (RGR)',
        'health_score':           'Health Score (0–100)',
        'chlorosis_pct':          'Chlorosis %',
        'necrosis_pct':           'Necrosis %',
        'canopy_height_mean_mm':  'Mean Canopy Height (mm)',
        'canopy_volume_cm3':      'Canopy Volume (cm³)',
    }

    available_metrics = {k: v for k, v in TREND_METRICS.items() if k in df.columns}

    # ── Plot selected metric ──────────────────────────────────────────────────
    selected_metric = st.selectbox(
        "Metric",
        list(available_metrics.keys()),
        format_func=lambda k: available_metrics[k],
    )

    fig = go.Figure()
    for chamber in chambers_sel:
        sub = df_filt[df_filt['chamber'] == chamber][['timestamp', selected_metric]].dropna()
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub['timestamp'], y=sub[selected_metric],
            mode='lines+markers',
            name=chamber.capitalize(),
            line=dict(color=COLORS[chamber], width=2.5),
            marker=dict(size=7),
            hovertemplate=f"<b>{chamber}</b><br>%{{x|%d %b %Y}}<br>{available_metrics[selected_metric]}: %{{y:.3f}}<extra></extra>",
        ))

    fig.update_yaxes(title_text=available_metrics[selected_metric])
    apply_chart_style(fig, title=available_metrics[selected_metric], height=440)
    st.plotly_chart(fig, use_container_width=True)

    # ── Summary grid (4 key metrics) ─────────────────────────────────────────
    st.markdown("<div class='section-title'>Summary Grid</div>", unsafe_allow_html=True)

    grid_metrics = [
        ('canopy_cover_%', 'Canopy Cover (%)'),
        ('ngrdi_mean',     'NGRDI'),
        ('health_score',   'Health Score'),
        ('leaf_count',     'Leaf Count'),
    ]
    grid_metrics = [(m, l) for m, l in grid_metrics if m in df.columns]

    if grid_metrics:
        fig2 = make_subplots(
            rows=2, cols=2,
            subplot_titles=[l for _, l in grid_metrics],
            vertical_spacing=0.14,
            horizontal_spacing=0.1,
        )
        positions = [(1,1),(1,2),(2,1),(2,2)]

        for (metric, label), (row, col) in zip(grid_metrics, positions):
            for chamber in chambers_sel:
                sub = df_filt[df_filt['chamber'] == chamber][['timestamp', metric]].dropna()
                if sub.empty:
                    continue
                fig2.add_trace(
                    go.Scatter(
                        x=sub['timestamp'], y=sub[metric],
                        mode='lines+markers',
                        name=chamber.capitalize(),
                        line=dict(color=COLORS[chamber], width=2),
                        marker=dict(size=5),
                        showlegend=(row == 1 and col == 1),
                    ),
                    row=row, col=col,
                )

        fig2.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_BG,
            height=550,
            margin=dict(l=50, r=20, t=50, b=50),
            font=dict(color="#c8e6c9"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        fig2.update_xaxes(gridcolor="#1e3a1e")
        fig2.update_yaxes(gridcolor="#1e3a1e")
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PER-POT DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Per-Pot Dashboard":
    st.markdown("""
    <div class='main-header'>
        <h1>Per-Pot Dashboard</h1>
        <p>Individual pot metrics, spatial heatmap, and time-series trends</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_pot_metrics()

    if df.empty:
        st.info("No per-pot data yet. Per-pot analysis requires a calibration JSON — run `calibrate_pots.py` after physical setup.")
        st.stop()

    # Derive actual pot labels and chambers from the data (not hardcoded)
    all_pot_labels   = sorted(df['pot_label'].unique())
    all_chambers     = sorted(df['chamber'].unique())
    available_dates  = sorted(df['timestamp'].dt.date.unique(), reverse=True)

    pot_metric_opts = {k: v for k, v in {
        'canopy_cover_%':       'Canopy Cover (%)',
        'ngrdi_mean':           'NGRDI',
        'exg_mean':             'ExG',
        'vari_mean':            'VARI',
        'health_score':         'Health Score',
        'leaf_count':           'Leaf Count',
        'lai':                  'LAI',
        'rgr':                  'RGR',
        'chlorosis_pct':        'Chlorosis (%)',
        'necrosis_pct':         'Necrosis (%)',
        'rosette_diameter_px':  'Rosette Diameter (px)',
        'canopy_height_mean_mm':'Canopy Height Mean (mm)',
        'canopy_height_max_mm': 'Canopy Height Max (mm)',
        'canopy_volume_cm3':    'Canopy Volume (cm³)',
        'soil_baseline_mm':     'Soil Baseline (mm)',
        'bolting_flag':         'Bolting Flag (0/1)',
    }.items() if k in df.columns}

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl1, ctrl2 = st.columns([1, 1])
    with ctrl1:
        pot_chambers_sel = st.multiselect(
            "Chambers", all_chambers, default=all_chambers, key="pot_chambers",
        )
        if not pot_chambers_sel:
            st.warning("Select at least one chamber.")
            st.stop()
    with ctrl2:
        sel_date = st.selectbox(
            "Heatmap date",
            available_dates,
            format_func=lambda d: d.strftime("%d %b %Y") + (" (latest)" if d == available_dates[0] else ""),
        )

    # ── Plant status warnings ─────────────────────────────────────────────────
    if 'plant_status' in df.columns:
        latest_df = (df[df['plant_status'].notna() & (df['plant_status'] != '')]
                     .sort_values('timestamp')
                     .groupby(['chamber', 'pot_label'], as_index=False)
                     .last())
        dead    = latest_df[latest_df['plant_status'] == 'dead']
        warning = latest_df[latest_df['plant_status'].isin(['warning', 'declining'])]
        if not dead.empty:
            dead_list = ", ".join(f"{r.chamber}/{r.pot_label}" for _, r in dead.iterrows())
            st.error(f"**Dead / No Growth:** {dead_list} — canopy cover has been near zero for 3+ days")
        if not warning.empty:
            warn_list = ", ".join(f"{r.chamber}/{r.pot_label}" for _, r in warning.iterrows())
            st.warning(f"**Low Growth Warning:** {warn_list} — canopy cover is very low or declining")

    # ── Section 1: Spatial heatmap ────────────────────────────────────────────
    st.markdown("<div class='section-title'>Spatial Heatmap — Pot Layout</div>", unsafe_allow_html=True)

    heatmap_metric = st.selectbox(
        "Heatmap metric",
        list(pot_metric_opts.keys()),
        format_func=lambda k: pot_metric_opts[k],
        key="heatmap_metric",
    )

    day_df           = df[df['timestamp'].dt.date == sel_date]
    chambers_present = [c for c in pot_chambers_sel if c in day_df['chamber'].values]
    GRID_R, GRID_C   = 2, 4

    hmap_cols = st.columns(max(len(chambers_present), 1))
    for hcol, chamber in zip(hmap_cols, chambers_present):
        # Use pot labels present for this chamber on this date, sorted
        ch_pots = sorted(day_df[day_df['chamber'] == chamber]['pot_label'].unique())
        # Pad to 8 slots
        grid = np.full((GRID_R, GRID_C), np.nan)
        text = [["" for _ in range(GRID_C)] for _ in range(GRID_R)]

        for idx in range(GRID_R * GRID_C):
            r, c = idx // GRID_C, idx % GRID_C
            if idx < len(ch_pots):
                pot = ch_pots[idx]
                sub = day_df[(day_df['chamber'] == chamber) &
                             (day_df['pot_label'] == pot)][heatmap_metric].dropna()
                if not sub.empty:
                    v = sub.iloc[-1]
                    grid[r][c] = v
                    text[r][c] = f"<b>{pot}</b><br>{v:.2f}"
                else:
                    text[r][c] = f"<b>{pot}</b><br>—"
            else:
                text[r][c] = "—"

        grid = grid[::-1]
        text = text[::-1]

        if np.all(np.isnan(grid)):
            hcol.markdown(f"**{chamber.capitalize()}**")
            hcol.info(f"No data for **{pot_metric_opts[heatmap_metric]}** on {sel_date}.")
        else:
            fig = go.Figure(go.Heatmap(
                z=grid, text=text, texttemplate="%{text}",
                textfont=dict(size=11, color="white"),
                colorscale="YlGn", showscale=True,
                colorbar=dict(
                    title=dict(text=pot_metric_opts[heatmap_metric], font=dict(color="#c8e6c9")),
                    tickfont=dict(color="#c8e6c9"),
                ),
                hovertemplate="%{text}<extra></extra>",
            ))
            fig.update_layout(
                title=dict(text=chamber.capitalize(), font=dict(color="#e8f5e9", size=14)),
                template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                height=250, margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False),
            )
            hcol.plotly_chart(fig, use_container_width=True)

    # ── Section 2: Individual pot detail ─────────────────────────────────────
    st.markdown("<div class='section-title'>Individual Pot Detail</div>", unsafe_allow_html=True)

    det_col1, det_col2 = st.columns([1, 1])
    with det_col1:
        sel_chamber_detail = st.selectbox("Chamber", pot_chambers_sel, key="det_chamber")
    with det_col2:
        pots_in_chamber = sorted(df[df['chamber'] == sel_chamber_detail]['pot_label'].unique())
        sel_pot = st.selectbox("Pot", pots_in_chamber, key="det_pot")

    pot_df = df[(df['chamber'] == sel_chamber_detail) & (df['pot_label'] == sel_pot)].sort_values('timestamp')

    if pot_df.empty:
        st.info(f"No data for {sel_pot} in {sel_chamber_detail}.")
    else:
        # Latest values as metric cards
        latest = pot_df.iloc[-1]
        card_metrics = [
            ('canopy_cover_%', 'Canopy Cover', '{:.2f}%'),
            ('ngrdi_mean',     'NGRDI',        '{:.4f}'),
            ('health_score',   'Health Score', '{:.1f}/100'),
            ('leaf_count',     'Leaf Count',   '{:.0f}'),
            ('lai',            'LAI',          '{:.4f}'),
            ('rgr',            'RGR',          '{:.4f}'),
        ]
        card_cols = st.columns(len(card_metrics))
        for i, (col_key, label, fmt) in enumerate(card_metrics):
            if col_key in pot_df.columns:
                val = latest.get(col_key)
                card_cols[i].metric(label, fmt.format(val) if pd.notna(val) else "—")

        # Multi-metric trend chart for this pot
        trend_metrics_sel = st.multiselect(
            "Metrics to plot",
            list(pot_metric_opts.keys()),
            default=[k for k in ['canopy_cover_%', 'ngrdi_mean'] if k in pot_df.columns],
            format_func=lambda k: pot_metric_opts[k],
            key="det_trend_metrics",
        )
        if trend_metrics_sel:
            fig_det = make_subplots(
                rows=len(trend_metrics_sel), cols=1,
                subplot_titles=[pot_metric_opts[m] for m in trend_metrics_sel],
                shared_xaxes=True,
                vertical_spacing=0.1,
            )
            for i, metric in enumerate(trend_metrics_sel):
                sub = pot_df[['timestamp', metric]].dropna()
                if not sub.empty:
                    fig_det.add_trace(
                        go.Scatter(
                            x=sub['timestamp'], y=sub[metric],
                            mode='lines+markers',
                            name=pot_metric_opts[metric],
                            line=dict(color=COLORS.get(sel_chamber_detail, '#4CAF50'), width=2),
                            marker=dict(size=6),
                            showlegend=False,
                        ),
                        row=i + 1, col=1,
                    )
            fig_det.update_layout(
                template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                height=220 * len(trend_metrics_sel),
                margin=dict(l=60, r=20, t=40, b=40),
                font=dict(color="#c8e6c9", size=11),
            )
            fig_det.update_xaxes(gridcolor="#1e3a1e")
            fig_det.update_yaxes(gridcolor="#1e3a1e")
            st.plotly_chart(fig_det, use_container_width=True)

        # Full data table for this pot
        with st.expander("Full data table"):
            display_cols = ['timestamp'] + [c for c in pot_metric_opts.keys() if c in pot_df.columns]
            st.dataframe(
                pot_df[display_cols].sort_values('timestamp', ascending=False).reset_index(drop=True),
                use_container_width=True,
            )

    # ── Section 3: Compare all pots — one metric, all pots as lines ──────────
    st.markdown("<div class='section-title'>Compare All Pots</div>", unsafe_allow_html=True)

    cmp_col1, cmp_col2 = st.columns([1, 1])
    with cmp_col1:
        cmp_chamber = st.selectbox("Chamber", pot_chambers_sel, key="cmp_chamber")
    with cmp_col2:
        cmp_metric = st.selectbox(
            "Metric",
            list(pot_metric_opts.keys()),
            format_func=lambda k: pot_metric_opts[k],
            key="cmp_metric",
        )

    cmp_pots = sorted(df[df['chamber'] == cmp_chamber]['pot_label'].unique())
    fig_cmp = go.Figure()
    palette = ['#4CAF50','#81C784','#AED581','#DCE775','#FFD54F','#FFB74D','#FF8A65','#A1887F']
    for i, pot in enumerate(cmp_pots):
        sub = df[(df['chamber'] == cmp_chamber) & (df['pot_label'] == pot)][['timestamp', cmp_metric]].dropna()
        if sub.empty:
            continue
        fig_cmp.add_trace(go.Scatter(
            x=sub['timestamp'], y=sub[cmp_metric],
            mode='lines+markers', name=pot,
            line=dict(color=palette[i % len(palette)], width=2),
            marker=dict(size=5),
        ))
    fig_cmp.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        height=380,
        margin=dict(l=60, r=20, t=20, b=40),
        font=dict(color="#c8e6c9", size=11),
        yaxis_title=pot_metric_opts[cmp_metric],
        xaxis=dict(gridcolor="#1e3a1e"),
        yaxis=dict(gridcolor="#1e3a1e"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="v"),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: GROUND TRUTH
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Ground Truth":
    st.markdown("""
    <div class='main-header'>
        <h1>Ground Truth</h1>
        <p>LI-600 stomatal conductance, chlorophyll fluorescence and SPAD readings</p>
    </div>
    """, unsafe_allow_html=True)

    gt = load_ground_truth()

    if gt.empty:
        st.info("No ground truth data yet. Use `li600_log.py` to log LI-600 and SPAD readings.")
        st.stop()

    # Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        date_range_gt = st.date_input(
            "Date range",
            value=(gt['date'].min().date(), gt['date'].max().date()),
            key="gt_dates",
        )
    with f2:
        ch_sel = st.multiselect("Chamber", CHAMBERS, default=CHAMBERS, key="gt_ch")
    with f3:
        search = st.text_input("Search plant ID", "")

    gt_filt = gt.copy()
    if len(date_range_gt) == 2:
        gt_filt = gt_filt[
            (gt_filt['date'].dt.date >= date_range_gt[0]) &
            (gt_filt['date'].dt.date <= date_range_gt[1])
        ]
    if ch_sel:
        gt_filt = gt_filt[gt_filt['chamber'].isin(ch_sel)]
    if search:
        gt_filt = gt_filt[gt_filt['plant_id'].str.contains(search, case=False, na=False)]

    st.markdown(f"**{len(gt_filt)} readings** matching filters")
    st.dataframe(gt_filt.sort_values('date', ascending=False), use_container_width=True)

    # ── Trend charts for the three key metrics ────────────────────────────────
    LI_METRICS = [
        ('gsw',      'Stomatal Conductance (gsw, mol/m²/s)'),
        ('fs',       'Steady-state Fluorescence (Fs)'),
        ('spad',     'SPAD Chlorophyll Index'),
    ]
    available_li = [(m, l) for m, l in LI_METRICS if m in gt.columns]

    if available_li and not gt_filt.empty:
        st.markdown("<div class='section-title'>Trends Over Time</div>", unsafe_allow_html=True)
        for metric, label in available_li:
            fig = go.Figure()
            for chamber in ch_sel:
                sub = gt_filt[gt_filt['chamber'] == chamber][['date', metric]].dropna()
                if sub.empty:
                    continue
                sub_grouped = sub.groupby('date')[metric].mean().reset_index()
                fig.add_trace(go.Scatter(
                    x=sub_grouped['date'], y=sub_grouped[metric],
                    mode='lines+markers',
                    name=chamber.capitalize(),
                    line=dict(color=COLORS[chamber], width=2),
                    marker=dict(size=7),
                    hovertemplate=f"<b>{chamber}</b><br>%{{x|%d %b}}<br>{label}: %{{y:.4f}}<extra></extra>",
                ))
            apply_chart_style(fig, title=f"{label} — Chamber Mean per Day", height=320)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RUN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Run Analysis":
    st.markdown("""
    <div class='main-header'>
        <h1>Run Analysis</h1>
        <p>Trigger the analysis pipeline on a captured image</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("<div class='section-title'>Configuration</div>", unsafe_allow_html=True)

        chamber = st.selectbox("Chamber", CHAMBERS)

        # Auto-detect available images for selected chamber
        chamber_dir = IMAGES_DIR / chamber
        images = []
        if chamber_dir.exists():
            images = sorted(
                [f for f in chamber_dir.glob("*.jpg") if "depth" not in f.name],
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )

        if images:
            img_options = {f.name: f for f in images[:20]}   # last 20
            selected_img_name = st.selectbox(
                "Image",
                list(img_options.keys()),
                help="Sorted by most recent first",
            )
            image_path = img_options[selected_img_name]
        else:
            st.warning(f"No images found in images/{chamber}/")
            image_path = None
            selected_img_name = None

        st.markdown("**Options:**")
        skip_leaves  = st.checkbox("Skip leaf counting (faster, ~5s vs ~90s)", value=False)
        skip_bolting = st.checkbox("Skip bolting detection", value=False)

    with col_r:
        st.markdown("<div class='section-title'>Image Preview</div>", unsafe_allow_html=True)
        if image_path and image_path.exists():
            st.image(str(image_path), caption=selected_img_name, use_container_width=True)
        else:
            st.markdown("*No image selected*")

    st.divider()

    # ── Capture + Analyse button ──────────────────────────────────────────────
    st.markdown("<div class='section-title'>Capture & Analyse</div>", unsafe_allow_html=True)
    st.markdown("Capture a fresh image from the camera and immediately run analysis.")

    capture_script = BASE_DIR / "scripts" / "capture_image.py"

    if st.button("📷  Capture & Analyse Now", type="primary", use_container_width=False):
        output_placeholder = st.empty()
        output_lines = []

        def stream_cmd(cmd):
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in proc.stdout:
                    output_lines.append(line.rstrip())
                    output_placeholder.markdown(
                        f"<div class='log-box'>{'<br>'.join(output_lines[-40:])}</div>",
                        unsafe_allow_html=True,
                    )
                proc.wait()
                return proc.returncode
            except Exception as e:
                st.error(f"Failed to run command: {e}")
                return 1

        # Step 1: Capture
        with st.spinner("Capturing image..."):
            capture_cmd = [str(PYTHON), str(capture_script), "--chamber", chamber]
            rc = stream_cmd(capture_cmd)

        if rc != 0:
            st.error("Capture failed. Check camera connection.")
            st.stop()

        # Find the newly captured image
        today = datetime.now().strftime("%Y-%m-%d")
        new_image = IMAGES_DIR / chamber / f"{today}_{chamber}.jpg"

        if new_image.exists():
            st.image(str(new_image), caption=f"Captured: {new_image.name}", use_container_width=True)
        else:
            st.warning("Image captured but could not display preview.")

        # Step 2: Analyse
        with st.spinner("Running analysis..."):
            analyse_cmd = [
                str(PYTHON), str(ANALYSE),
                "--image",   str(new_image),
                "--chamber", chamber,
            ]
            if skip_leaves:
                analyse_cmd.append("--no-leaves")
            if skip_bolting:
                analyse_cmd.append("--no-bolting")
            rc = stream_cmd(analyse_cmd)

        if rc == 0:
            st.success("Capture and analysis complete!")
            invalidate_cache()
            st.success("Done!")
        else:
            st.error("Analysis failed. Check the log above.")

    st.divider()

    # ── Run button (existing image) ───────────────────────────────────────────
    st.markdown("<div class='section-title'>Analyse Existing Image</div>", unsafe_allow_html=True)

    if image_path is None:
        st.button("Run Analysis", disabled=True)
    else:
        if st.button("▶  Run Analysis", type="primary", use_container_width=False):
            cmd = [
                str(PYTHON), str(ANALYSE),
                "--image",   str(image_path),
                "--chamber", chamber,
            ]
            if skip_leaves:
                cmd.append("--no-leaves")
            if skip_bolting:
                cmd.append("--no-bolting")

            st.markdown(f"**Running:** `{' '.join(cmd)}`")

            output_placeholder = st.empty()
            output_lines = []

            with st.spinner("Analysis running..."):
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    for line in proc.stdout:
                        output_lines.append(line.rstrip())
                        output_placeholder.markdown(
                            f"<div class='log-box'>{'<br>'.join(output_lines[-40:])}</div>",
                            unsafe_allow_html=True,
                        )
                    proc.wait()
                    rc = proc.returncode
                except Exception as e:
                    st.error(f"Failed to start analysis: {e}")
                    st.stop()

            if rc == 0:
                st.success("Analysis complete. Refreshing data...")
                invalidate_cache()
                st.success("Done!")
            else:
                st.error(f"Analysis failed (exit code {rc}). Check the log above for details.")

    # ── Recent analysis log ───────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Recent Results</div>", unsafe_allow_html=True)
    df_recent = load_metrics()
    if not df_recent.empty:
        recent_cols = [c for c in [
            'timestamp', 'chamber', 'canopy_cover_%', 'ngrdi_mean',
            'leaf_count', 'health_score', 'lai',
        ] if c in df_recent.columns]
        st.dataframe(
            df_recent[recent_cols]
              .sort_values('timestamp', ascending=False)
              .head(10)
              .round(3),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LOG READINGS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Log Readings":
    st.markdown("""
    <div class='main-header'>
        <h1>Log Readings</h1>
        <p>Enter LI-600 and SPAD measurements — leaf by leaf, pot by pot</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Leaf detection helpers (self-contained) ───────────────────────────────

    _HSV_LO = np.array([25,  40,  40])
    _HSV_HI = np.array([90, 255, 255])

    def _load_calib(chamber):
        p = CALIB_DIR / f"{chamber}_calibration.json"
        if not p.exists():
            return None
        with open(p) as f:
            return json.load(f)

    def _get_latest_img(chamber):
        d = IMAGES_DIR / chamber
        if not d.exists():
            return None
        cands = sorted(
            [f for f in d.glob("*.jpg")
             if "depth" not in f.name and "snapshot" not in f.name],
            key=lambda f: f.stat().st_mtime, reverse=True,
        )
        return cands[0] if cands else None

    def _crop_pot(img, pot_info, padding=20):
        x, y, r = int(pot_info["x"]), int(pot_info["y"]), int(pot_info["r"])
        x1, y1  = max(0, x-r-padding), max(0, y-r-padding)
        x2, y2  = min(img.shape[1], x+r+padding), min(img.shape[0], y+r+padding)
        crop    = img[y1:y2, x1:x2].copy()
        mask    = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x-x1, y-y1), r, 255, -1)
        crop[mask == 0] = 0
        return crop

    def _detect_leaves(pot_img, min_area=300):
        hsv   = cv2.cvtColor(pot_img, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, _HSV_LO, _HSV_HI)
        k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean = cv2.morphologyEx(green, cv2.MORPH_OPEN,  k, iterations=2)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k, iterations=2)
        if clean.sum() == 0:
            return []
        dist  = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
        _, fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
        fg    = fg.astype(np.uint8)
        bg    = cv2.dilate(clean, k, iterations=3)
        unkn  = cv2.subtract(bg, fg)
        n_lbl, markers = cv2.connectedComponents(fg)
        markers = markers + 1
        markers[unkn == 255] = 0
        ws = pot_img.copy()
        ws[pot_img.sum(axis=2) == 0] = [100, 100, 100]
        cv2.watershed(ws, markers)
        leaves = []
        for lbl in range(2, n_lbl + 1):
            lm      = (markers == lbl).astype(np.uint8)
            cnts, _ = cv2.findContours(lm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt  = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            leaves.append({"area": area, "centroid": (cx, cy), "contour": cnt})
        leaves.sort(key=lambda l: l["area"], reverse=True)
        for i, l in enumerate(leaves):
            l["leaf_id"] = i + 1
        return leaves

    def _pick_targets(leaves, n):
        if not leaves:
            return list(range(1, n + 1))
        total = len(leaves)
        if n >= total:
            return [l["leaf_id"] for l in leaves]
        if n == 1:
            return [leaves[max(0, total - 2)]["leaf_id"]]
        if n == 2:
            return [leaves[0]["leaf_id"], leaves[-1]["leaf_id"]]
        idxs = [round(i * (total - 1) / (n - 1)) for i in range(n)]
        return [leaves[i]["leaf_id"] for i in sorted(set(idxs))]

    def _annotate_img(pot_img, leaves, target_ids):
        ann = pot_img.copy()
        for leaf in leaves:
            cx, cy = leaf["centroid"]
            lid    = leaf["leaf_id"]
            tgt    = lid in target_ids
            col    = (0, 230, 60) if tgt else (160, 160, 160)
            cv2.drawContours(ann, [leaf["contour"]], -1, col, 2 if tgt else 1)
            cv2.circle(ann, (cx, cy), 16, col, -1)
            cv2.circle(ann, (cx, cy), 16, (0, 0, 0), 1)
            t = str(lid)
            cv2.putText(ann, t, (cx - 5 * len(t), cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(ann, "GREEN = measure   GREY = skip",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 60), 1)
        return cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)

    def _prev_val(chamber, pot_label, leaf_id, metric):
        if not GROUND_TRUTH_CSV.exists():
            return None
        try:
            rows = []
            with open(GROUND_TRUTH_CSV, newline="") as f:
                for row in csv.DictReader(f):
                    if (row.get("chamber")   == chamber and
                        row.get("pot_label") == pot_label and
                        row.get("leaf_id")   == leaf_id and
                        row.get(metric, "")  != ""):
                        rows.append(row)
            if not rows:
                return None
            rows.sort(key=lambda r: r["date"], reverse=True)
            return float(rows[0][metric])
        except Exception:
            return None

    # ── Metric definitions ────────────────────────────────────────────────────

    LR_METRICS = [
        ("gsw",      "Stomatal Conductance", "mol/m²/s", "%.4f"),
        ("vpleaf",   "Leaf Vapor Pressure",  "kPa",      "%.3f"),
        ("vpdleaf",  "Vapor Pressure Deficit","kPa",     "%.3f"),
        ("h2oleaf",  "Leaf H₂O Fraction",    "mmol/mol", "%.2f"),
        ("fs",       "Fluorescence Fs",      "rel.",     "%.1f"),
        ("fm_prime", "Fluorescence Fm'",     "rel.",     "%.1f"),
    ]

    GT_ALL_COLS = ["date", "chamber", "pot_label", "leaf_id", "leaf_area_px",
                   "leaf_notes", "gsw", "vpleaf", "vpdleaf", "h2oleaf",
                   "fs", "fm_prime", "spad", "phi_psii"]

    # ── Session state ─────────────────────────────────────────────────────────

    def _init_state():
        st.session_state.setdefault("lr_active",   False)
        st.session_state.setdefault("lr_chamber",  "enriched")
        st.session_state.setdefault("lr_date",     datetime.now().date())
        st.session_state.setdefault("lr_n_leaves", 2)
        st.session_state.setdefault("lr_pot_idx",  0)
        st.session_state.setdefault("lr_rows",     [])
        st.session_state.setdefault("lr_annots",   {})

    _init_state()

    # ── Setup screen ──────────────────────────────────────────────────────────

    if not st.session_state.lr_active:
        st.markdown("<div class='section-title'>New Session</div>",
                    unsafe_allow_html=True)

        s1, s2, s3 = st.columns(3)
        with s1:
            st.session_state.lr_chamber = st.selectbox("Chamber", CHAMBERS)
        with s2:
            st.session_state.lr_date = st.date_input(
                "Date", value=datetime.now().date())
        with s3:
            st.session_state.lr_n_leaves = st.selectbox(
                "Leaves per pot", [1, 2, 3], index=1,
                help="1 = youngest only  |  2 = oldest + youngest  |  3 = oldest, middle, youngest",
            )

        n_lv   = st.session_state.lr_n_leaves
        est    = len(POT_LABELS) * n_lv * 3
        st.info(
            f"Estimated session time: ~{est} minutes  "
            f"({len(POT_LABELS)} pots × {n_lv} leaf{'s' if n_lv > 1 else ''} × ~3 min each)"
        )

        if not HAS_CV2:
            st.warning("opencv-python not found — leaf annotation images will be unavailable.")

        if st.button("▶  Start Session", type="primary"):
            st.session_state.lr_pot_idx = 0
            st.session_state.lr_rows    = []
            st.session_state.lr_annots  = {}

            chamber  = st.session_state.lr_chamber
            n_leaves = st.session_state.lr_n_leaves
            calib    = _load_calib(chamber)    if HAS_CV2 else None
            img_path = _get_latest_img(chamber) if HAS_CV2 else None
            full_img = cv2.imread(str(img_path)) if (img_path and HAS_CV2) else None

            with st.spinner("Detecting leaves and generating annotations..."):
                for pot_label in POT_LABELS:
                    entry = {"img": None, "leaves": [], "tgts": list(range(1, n_leaves + 1))}
                    if full_img is not None and calib:
                        try:
                            pot_info = calib["pots"][POT_LABELS.index(pot_label)]
                            crop     = _crop_pot(full_img, pot_info)
                            leaves   = _detect_leaves(crop)
                            tgts     = _pick_targets(leaves, n_leaves)
                            entry    = {
                                "img":    _annotate_img(crop, leaves, tgts),
                                "leaves": leaves,
                                "tgts":   tgts,
                            }
                        except Exception:
                            pass
                    st.session_state.lr_annots[pot_label] = entry

            st.session_state.lr_active = True
            st.rerun()

    # ── Active session ────────────────────────────────────────────────────────

    else:
        pot_idx   = st.session_state.lr_pot_idx
        pot_label = POT_LABELS[pot_idx]
        chamber   = st.session_state.lr_chamber
        date_str  = str(st.session_state.lr_date)
        pot_data  = st.session_state.lr_annots.get(pot_label, {})
        leaves    = pot_data.get("leaves", [])
        tgts      = pot_data.get("tgts",   list(range(1, st.session_state.lr_n_leaves + 1)))
        is_last   = (pot_idx == len(POT_LABELS) - 1)

        # Progress bar
        st.progress(pot_idx / len(POT_LABELS),
                    text=f"Pot {pot_idx + 1} of {len(POT_LABELS)} — {pot_label}")

        st.markdown(
            f"<div class='section-title'>{chamber.capitalize()} — Pot {pot_label}</div>",
            unsafe_allow_html=True,
        )

        img_col, form_col = st.columns([1, 2])

        with img_col:
            ann_img = pot_data.get("img")
            if ann_img is not None:
                st.image(ann_img, caption=f"{pot_label} — green = measure",
                         use_container_width=True)
                if leaves:
                    st.caption(
                        f"{len(leaves)} leaves detected — "
                        f"measuring {len(tgts)}: {', '.join('leaf ' + str(t) for t in tgts)}"
                    )
            else:
                st.info("No annotation available — enter values below.")

        with form_col:
            with st.form(key=f"lr_form_{pot_label}_{pot_idx}"):
                collected = {}

                for leaf_id in tgts:
                    leaf_area = next(
                        (int(l["area"]) for l in leaves if l["leaf_id"] == leaf_id), None
                    )
                    age = ("oldest"   if leaf_id == min(tgts) else
                           "youngest" if leaf_id == max(tgts) else "middle")

                    area_str = f"  |  area: {leaf_area} px" if leaf_area else ""
                    st.markdown(f"**Leaf {leaf_id} — {age}**{area_str}")

                    leaf_key = f"leaf_{leaf_id}"
                    vals     = {}

                    # Porometer: 4 columns
                    p1, p2, p3, p4 = st.columns(4)
                    for col_w, (key, label, unit, fmt) in zip(
                            [p1, p2, p3, p4], LR_METRICS[:4]):
                        prev = _prev_val(chamber, pot_label, leaf_key, key)
                        help_txt = f"prev: {prev:{fmt[1:]}}" if prev is not None else unit
                        vals[key] = col_w.number_input(
                            label, value=None, format=fmt,
                            help=help_txt, key=f"{pot_label}_{leaf_id}_{key}",
                        )

                    # Fluorescence: 2 columns + derived ΦPSII
                    f1, f2, f3 = st.columns(3)
                    for col_w, (key, label, unit, fmt) in zip(
                            [f1, f2], LR_METRICS[4:]):
                        prev = _prev_val(chamber, pot_label, leaf_key, key)
                        help_txt = f"prev: {prev:.1f}" if prev is not None else unit
                        vals[key] = col_w.number_input(
                            label, value=None, format=fmt,
                            help=help_txt, key=f"{pot_label}_{leaf_id}_{key}",
                        )
                    fs_v, fm_v = vals.get("fs"), vals.get("fm_prime")
                    if fs_v is not None and fm_v is not None and fm_v > 0:
                        phi = round((fm_v - fs_v) / fm_v, 4)
                        f3.metric("ΦPSII", f"{phi:.4f}")
                        vals["phi_psii"] = phi
                    else:
                        vals["phi_psii"] = None

                    # SPAD: 3 readings
                    st.markdown("**SPAD** — enter up to 3 readings (averaged automatically)")
                    sc1, sc2, sc3 = st.columns(3)
                    prev_spad = _prev_val(chamber, pot_label, leaf_key, "spad")
                    help_spad = f"prev mean: {prev_spad:.1f}" if prev_spad else "SPAD units"
                    sv1 = sc1.number_input("Reading 1", value=None, format="%.1f",
                                           help=help_spad, key=f"{pot_label}_{leaf_id}_s1")
                    sv2 = sc2.number_input("Reading 2", value=None, format="%.1f",
                                           key=f"{pot_label}_{leaf_id}_s2")
                    sv3 = sc3.number_input("Reading 3", value=None, format="%.1f",
                                           key=f"{pot_label}_{leaf_id}_s3")
                    spad_readings = [v for v in [sv1, sv2, sv3] if v is not None]
                    if spad_readings:
                        vals["spad"] = round(sum(spad_readings) / len(spad_readings), 2)
                        st.caption(f"→ SPAD mean: {vals['spad']:.1f}  "
                                   f"({len(spad_readings)} reading{'s' if len(spad_readings)>1 else ''})")
                    else:
                        vals["spad"] = None

                    vals["leaf_notes"] = st.text_input(
                        "Leaf notes", placeholder="e.g. 4th from outside, slight curl",
                        key=f"{pot_label}_{leaf_id}_notes",
                    )

                    collected[leaf_id] = vals
                    if leaf_id != tgts[-1]:
                        st.divider()

                btn_txt   = "Save & Finish ✓" if is_last else f"Save & Next → {POT_LABELS[pot_idx+1] if not is_last else ''}"
                submitted = st.form_submit_button(btn_txt, type="primary",
                                                  use_container_width=True)

            if submitted:
                for leaf_id, vals in collected.items():
                    leaf_area = next(
                        (int(l["area"]) for l in leaves if l["leaf_id"] == leaf_id), ""
                    )
                    row = {
                        "date":        date_str,
                        "chamber":     chamber,
                        "pot_label":   pot_label,
                        "leaf_id":     f"leaf_{leaf_id}",
                        "leaf_area_px": leaf_area,
                        "leaf_notes":  vals.get("leaf_notes", ""),
                    }
                    for key, _, _, _ in LR_METRICS:
                        v = vals.get(key)
                        row[key] = v if v is not None else ""
                    row["spad"]     = vals.get("spad") or ""
                    row["phi_psii"] = vals.get("phi_psii") or ""
                    st.session_state.lr_rows.append(row)

                if is_last:
                    write_header = not GROUND_TRUTH_CSV.exists()
                    os.makedirs(RESULTS_DIR, exist_ok=True)
                    with open(GROUND_TRUTH_CSV, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=GT_ALL_COLS,
                                                extrasaction="ignore")
                        if write_header:
                            writer.writeheader()
                        writer.writerows(st.session_state.lr_rows)
                    n_saved = len(st.session_state.lr_rows)
                    st.session_state.lr_active = False
                    st.session_state.lr_rows   = []
                    invalidate_cache()
                    st.success(f"Session complete — {n_saved} leaf readings saved.")
                    st.success("Done!")
                    st.rerun()
                else:
                    st.session_state.lr_pot_idx += 1
                    st.rerun()

        # Skip / abandon row
        sk_col, ab_col = st.columns([1, 3])
        with sk_col:
            if st.button("Skip pot"):
                if is_last:
                    write_header = not GROUND_TRUTH_CSV.exists()
                    if st.session_state.lr_rows:
                        os.makedirs(RESULTS_DIR, exist_ok=True)
                        with open(GROUND_TRUTH_CSV, "a", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=GT_ALL_COLS,
                                                    extrasaction="ignore")
                            if write_header:
                                writer.writeheader()
                            writer.writerows(st.session_state.lr_rows)
                    st.session_state.lr_active = False
                    st.session_state.lr_rows   = []
                    invalidate_cache()
                else:
                    st.session_state.lr_pot_idx += 1
                st.rerun()
        with ab_col:
            if st.button("✕  Abandon session — discard all unsaved data"):
                st.session_state.lr_active  = False
                st.session_state.lr_rows    = []
                st.session_state.lr_pot_idx = 0
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE VIEW
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Live View":
    st.markdown("""
    <div class='main-header'>
        <h1>Live View</h1>
        <p>Latest captured image from each chamber — auto-refreshes every 30 seconds</p>
    </div>
    """, unsafe_allow_html=True)

    def get_latest_image(chamber):
        """Return (rgb_path, depth_preview_path, mtime) for the most recent capture."""
        chamber_dir = IMAGES_DIR / chamber
        if not chamber_dir.exists():
            return None, None, None
        candidates = sorted(
            [f for f in chamber_dir.glob("*.jpg")
             if "depth" not in f.name and "snapshot" not in f.name],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return None, None, None
        rgb   = candidates[0]
        depth = rgb.parent / rgb.name.replace(".jpg", "_depth_preview.jpg")
        mtime = datetime.fromtimestamp(rgb.stat().st_mtime)
        return rgb, depth if depth.exists() else None, mtime

    # Controls
    ctrl_l, ctrl_m, ctrl_r = st.columns([2, 2, 1])
    with ctrl_l:
        show_depth = st.toggle("Show depth preview alongside RGB", value=False)
    with ctrl_m:
        show_mask = st.toggle("Show green mask overlay", value=False)
    with ctrl_r:
        auto_refresh = st.toggle("Auto-refresh (30s)", value=False)

    st.divider()

    # ── Per-chamber panels ────────────────────────────────────────────────────
    cam_cols = st.columns(2)

    for col, chamber in zip(cam_cols, CHAMBERS):
        rgb_path, depth_path, mtime = get_latest_image(chamber)

        with col:
            st.markdown(
                f"<div class='section-title'>{chamber.capitalize()} Chamber</div>",
                unsafe_allow_html=True,
            )

            if rgb_path is None:
                st.warning(f"No images found in `images/{chamber}/`")
                continue

            # Timestamp + staleness warning
            age_minutes = (datetime.now() - mtime).total_seconds() / 60
            age_str     = mtime.strftime("%d %b %Y — %H:%M:%S")
            if age_minutes < 90:
                st.success(f"Captured: {age_str}")
            elif age_minutes < 1440:
                st.warning(f"Captured: {age_str}  ({int(age_minutes/60)}h ago)")
            else:
                st.error(f"Captured: {age_str}  ({int(age_minutes/1440)}d ago — stale)")

            # Build image panels
            panels = [("RGB", str(rgb_path), rgb_path.name)]
            if show_depth:
                if depth_path:
                    panels.append(("Depth", str(depth_path), "Depth preview (closer = brighter)"))
                else:
                    panels.append(("Depth", None, "No depth preview available"))
            if show_mask:
                import cv2 as _cv2
                import numpy as _np
                _img = _cv2.imread(str(rgb_path))
                if _img is not None:
                    _hsv  = _cv2.cvtColor(_img, _cv2.COLOR_BGR2HSV)
                    _mask = _cv2.inRange(_hsv, _np.array([25, 40, 40]), _np.array([90, 255, 255]))
                    _overlay = _img.copy()
                    _overlay[_mask > 0] = [0, 220, 0]
                    _blended = _cv2.addWeighted(_img, 0.4, _overlay, 0.6, 0)
                    _blended_rgb = _cv2.cvtColor(_blended, _cv2.COLOR_BGR2RGB)
                    panels.append(("Mask", _blended_rgb, "Green mask overlay"))

            img_cols = st.columns(len(panels))
            for (label, src, caption), col in zip(panels, img_cols):
                with col:
                    if src is None:
                        st.caption(caption)
                    elif isinstance(src, str):
                        st.image(src, caption=caption, use_container_width=True)
                    else:
                        st.image(src, caption=caption, use_container_width=True)

            # Quick metrics for this image date
            df_lv = load_metrics()
            if not df_lv.empty:
                img_date = mtime.date()
                row = df_lv[
                    (df_lv['timestamp'].dt.date == img_date) &
                    (df_lv['chamber'] == chamber)
                ]
                if not row.empty:
                    m1, m2, m3 = st.columns(3)
                    cc  = row['canopy_cover_%'].values[0]
                    ng  = row['ngrdi_mean'].values[0]  if 'ngrdi_mean'   in row.columns else None
                    hs  = row['health_score'].values[0] if 'health_score' in row.columns else None
                    m1.metric("Canopy Cover", f"{cc:.1f}%"  if not pd.isna(cc) else "—")
                    m2.metric("NGRDI",        f"{ng:.3f}"   if ng  is not None and not pd.isna(ng)  else "—")
                    m3.metric("Health Score", f"{hs:.0f}/100" if hs is not None and not pd.isna(hs) else "—")

    st.divider()

    # ── Capture buttons ───────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Capture New Images</div>", unsafe_allow_html=True)
    st.caption("Triggers `capture_image.py` — camera must be connected and powered on.")

    cap_cols = st.columns(3)
    capture_script = BASE_DIR / "scripts" / "capture_image.py"

    for btn_col, chamber in zip(cap_cols[:2], CHAMBERS):
        with btn_col:
            if st.button(f"Capture — {chamber.capitalize()}", use_container_width=True):
                cmd = [str(PYTHON), str(capture_script), "--chamber", chamber]
                with st.spinner(f"Capturing from {chamber} camera..."):
                    try:
                        result = subprocess.run(
                            cmd, capture_output=True, text=True, timeout=60
                        )
                        if result.returncode == 0:
                            st.success(f"{chamber.capitalize()} image captured.")
                            invalidate_cache()
                            st.rerun()
                        else:
                            st.error(f"Capture failed:\n{result.stderr.strip()}")
                    except subprocess.TimeoutExpired:
                        st.error("Camera timed out after 60s — is it connected?")
                    except Exception as e:
                        st.error(f"Error: {e}")

    with cap_cols[2]:
        if st.button("Capture — Both Chambers", use_container_width=True):
            for chamber in CHAMBERS:
                cmd = [str(PYTHON), str(capture_script), "--chamber", chamber]
                with st.spinner(f"Capturing {chamber}..."):
                    try:
                        result = subprocess.run(
                            cmd, capture_output=True, text=True, timeout=60
                        )
                        if result.returncode == 0:
                            st.success(f"{chamber.capitalize()} done.")
                        else:
                            st.error(f"{chamber.capitalize()} failed: {result.stderr.strip()}")
                    except subprocess.TimeoutExpired:
                        st.error(f"{chamber.capitalize()} timed out.")
                    except Exception as e:
                        st.error(f"{chamber.capitalize()} error: {e}")
            invalidate_cache()
            st.rerun()

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    if auto_refresh:
        import time as _time
        refresh_placeholder = st.empty()
        for i in range(30, 0, -1):
            refresh_placeholder.caption(f"Auto-refreshing in {i}s...")
            _time.sleep(1)
        refresh_placeholder.empty()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Statistics":
    st.markdown("""
    <div class='main-header'>
        <h1>Statistical Comparison</h1>
        <p>Enriched (elevated CO₂) vs Control (ambient CO₂) — Mann-Whitney U tests, effect sizes, and growth curves</p>
    </div>
    """, unsafe_allow_html=True)

    RESULTS_DIR = METRICS_CSV.parent
    STATS_CSV   = RESULTS_DIR / "stats_summary.csv"
    PLOTS_DIR   = RESULTS_DIR / "plots"

    # ── Week selector ─────────────────────────────────────────────────────────
    PLANTING_DATE = pd.Timestamp("2026-03-16")
    pot_df_stats  = load_pot_metrics()
    if not pot_df_stats.empty:
        first_day = pot_df_stats['timestamp'].min().normalize()
        last_day  = pot_df_stats['timestamp'].max().normalize()
    else:
        first_day = PLANTING_DATE
        last_day  = pd.Timestamp.now().normalize()

    # Build week options (no "All time" entry — empty selection = all time)
    # Start weeks from first actual data date, not planting date
    week_options = {}
    current = first_day
    week_num = 1
    while current <= last_day:
        week_end = current + pd.Timedelta(days=6)
        label = f"Week {week_num}  ({current.strftime('%d %b')} – {week_end.strftime('%d %b')})"
        week_options[label] = (current.strftime('%Y-%m-%d'), week_end.strftime('%Y-%m-%d'))
        current  = week_end + pd.Timedelta(days=1)
        week_num += 1

    selected_weeks = st.multiselect(
        "Date range — select week(s) or leave empty for all time",
        options=list(week_options.keys()),
        default=[],
        help="Select one or more weeks to filter the analysis. Leave empty to use all data. Weeks counted from planting date (16 Mar 2026)."
    )

    # Resolve selected weeks to a date range
    if not selected_weeks:
        start_date_filter = None
        end_date_filter   = None
        date_label = "All time"
    else:
        dates = [week_options[w] for w in selected_weeks]
        start_date_filter = min(d[0] for d in dates)
        end_date_filter   = max(d[1] for d in dates)
        date_label = f"{start_date_filter} to {end_date_filter}"

    st.caption(f"Analysis period: **{date_label}**")
    st.divider()

    # ── Run / refresh buttons ─────────────────────────────────────────────────
    col_btn1, col_btn2, col_btn3, _ = st.columns([1, 1, 1, 2])
    with col_btn1:
        exclude_dead = st.toggle("Exclude dead/warning pots", value=False)
    with col_btn2:
        run_stats_btn = st.button("Run Statistical Analysis", type="primary")
    with col_btn3:
        run_pca_btn = st.button("Run PCA", type="secondary")

    if run_pca_btn:
        with st.spinner("Running PCA..."):
            cmd = [sys.executable, str(BASE_DIR / "scripts" / "pca_analysis.py")]
            if start_date_filter:
                cmd += ["--start-date", start_date_filter, "--end-date", end_date_filter]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE_DIR))
        if result.returncode == 0:
            st.success("PCA complete.")
            st.cache_data.clear()
        else:
            st.error(f"Error: {result.stderr[-500:] if result.stderr else 'unknown'}")

    if run_stats_btn:
        with st.spinner("Running statistical comparison..."):
            cmd = [sys.executable, str(BASE_DIR / "scripts" / "statistical_comparison.py")]
            if exclude_dead:
                cmd.append("--exclude-dead")
            if start_date_filter:
                cmd += ["--start-date", start_date_filter, "--end-date", end_date_filter]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE_DIR))
        if result.returncode == 0:
            st.success("Analysis complete.")
            st.cache_data.clear()
        else:
            st.error(f"Error: {result.stderr[-500:] if result.stderr else 'unknown'}")

    st.divider()

    # ── Stats summary table ───────────────────────────────────────────────────
    if STATS_CSV.exists():
        try:
            stats_df = pd.read_csv(STATS_CSV)
            if stats_df.empty or 'metric' not in stats_df.columns:
                raise ValueError("empty")
        except Exception:
            stats_df = None

        if stats_df is not None:
            st.markdown("<div class='section-title'>Statistical Test Results</div>", unsafe_allow_html=True)

        # Format for display
            def sig_stars(p):
                if pd.isna(p): return ''
                if p < 0.001: return '***'
                if p < 0.01:  return '**'
                if p < 0.05:  return '*'
                return 'ns'

            stats_df['significance'] = stats_df['p_value'].apply(sig_stars)
            stats_df['significant']  = stats_df['significant'].map({True: '✓', False: '✗'})

            display_cols = ['metric', 'mean_enriched', 'mean_control', 'p_value',
                            'significance', 'cohens_d', 'effect_size']
            st.dataframe(
                stats_df[display_cols].rename(columns={
                    'metric': 'Metric', 'mean_enriched': 'Mean Enriched',
                    'mean_control': 'Mean Control', 'p_value': 'p-value',
                    'significance': 'Sig.', 'cohens_d': "Cohen's d",
                    'effect_size': 'Effect Size',
                }),
                use_container_width=True, hide_index=True,
            )
            st.caption("Significance: *** p<0.001 | ** p<0.01 | * p<0.05 | ns = not significant. Mann-Whitney U test (two-sided).")
        else:
            st.info("No statistics yet — click 'Run Statistical Analysis' above.")
    else:
        st.info("No statistics yet — click 'Run Statistical Analysis' above.")

    st.divider()

    # ── Figures ───────────────────────────────────────────────────────────────
    figures = [
        ("growth_curves.png",         "Canopy Cover Over Time"),
        ("vegetation_indices.png",     "Vegetation Indices (NGRDI, VARI, ExG)"),
        ("rgr_comparison.png",         "Relative Growth Rate"),
        ("metric_boxplots.png",        "Metric Distributions"),
        ("health_score_comparison.png","Health Score Over Time"),
        ("pca_scatter.png",            "PCA — Chamber Separation"),
        ("pca_loadings.png",           "PCA — Feature Contributions"),
        ("pca_variance.png",           "PCA — Explained Variance (Scree Plot)"),
        ("pca_trajectory.png",         "PCA — PC1 Trajectory Over Time"),
    ]

    for fname, title in figures:
        fpath = PLOTS_DIR / fname
        if fpath.exists():
            st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
            st.image(str(fpath), use_container_width=True)
            with open(fpath, "rb") as f:
                st.download_button(
                    label=f"Download {title}",
                    data=f,
                    file_name=fname,
                    mime="image/png",
                    key=f"dl_{fname}",
                )
            st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TIMELAPSE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Timelapse":
    st.markdown("""
    <div class='main-header'>
        <h1>Timelapse</h1>
        <p>Daily growth timelapse compiled from all captured images</p>
    </div>
    """, unsafe_allow_html=True)

    TIMELAPSE_DIR = BASE_DIR / "timelapse"

    def get_timelapse(chamber):
        path = TIMELAPSE_DIR / f"{chamber}_timelapse.mp4"
        return path if path.exists() else None

    col1, col2 = st.columns(2)

    for col, chamber in zip([col1, col2], CHAMBERS):
        with col:
            st.markdown(
                f"<div class='section-title'>{chamber.capitalize()} Chamber</div>",
                unsafe_allow_html=True,
            )
            mp4 = get_timelapse(chamber)
            if mp4:
                with open(mp4, 'rb') as _f:
                    st.video(_f.read())
                with open(mp4, "rb") as f:
                    st.download_button(
                        label=f"Download {chamber} timelapse",
                        data=f,
                        file_name=mp4.name,
                        mime="video/mp4",
                    )
            else:
                st.info(f"No timelapse found for {chamber} chamber. Run `python scripts/timelapse.py --chamber {chamber} --timelapse` to generate.")

    st.divider()
    st.markdown("<div class='section-title'>Regenerate Timelapse</div>", unsafe_allow_html=True)
    st.caption("Rebuilds the timelapse from all images in the images folder. Run this after new captures to update.")

    regen_col1, regen_col2 = st.columns(2)
    for col, chamber in zip([regen_col1, regen_col2], CHAMBERS):
        with col:
            if st.button(f"Regenerate {chamber.capitalize()}", key=f"regen_{chamber}"):
                with st.spinner(f"Building {chamber} timelapse..."):
                    import subprocess
                    result = subprocess.run(
                        [sys.executable, str(BASE_DIR / "scripts" / "timelapse.py"),
                         "--chamber", chamber, "--timelapse"],
                        capture_output=True, text=True, cwd=str(BASE_DIR)
                    )
                if result.returncode == 0:
                    st.success("Done!")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(f"Failed: {result.stderr[-500:] if result.stderr else 'unknown error'}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: METRICS GLOSSARY
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Metrics":
    st.markdown("""
    <div class='main-header'>
        <h1>Metrics Glossary</h1>
        <p>Explanation of every metric calculated by the analysis pipeline</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    All metrics are calculated daily per chamber by `analyse_chamber.py` and written
    to `results/metrics.csv` (whole chamber) and `results/pot_metrics.csv` (per pot).
    """)

    # ── Vegetation Indices ────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Vegetation Indices</div>", unsafe_allow_html=True)

    veg_data = {
        "Metric": [
            "canopy_cover_%",
            "exg_mean / exg_std",
            "vari_mean / vari_std",
            "ngrdi_mean / ngrdi_std",
        ],
        "Full Name": [
            "Canopy Cover",
            "Excess Green Index",
            "Visible Atmospherically Resistant Index",
            "Normalised Green-Red Difference Index",
        ],
        "Formula": [
            "Green pixels / total area × 100",
            "2G − R − B  (normalised)",
            "(G − R) / (G + R − B)",
            "(G − R) / (G + R)",
        ],
        "What it tells you": [
            "How much of the pot/chamber is covered by plant material. Primary growth indicator.",
            "Overall greenness. Sensitive to biomass but affected by lighting changes.",
            "Greenness normalised against blue channel — more robust to illumination variation than ExG.",
            "Best RGB predictor of stomatal conductance (R²≈0.42). Distinguishes healthy green tissue from yellowing.",
        ],
    }
    st.dataframe(pd.DataFrame(veg_data), use_container_width=True, hide_index=True)

    # ── Growth Metrics ────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Growth Metrics</div>", unsafe_allow_html=True)

    growth_data = {
        "Metric": [
            "rosette_diameter_px",
            "rosette_area_px",
            "rgr",
            "leaf_count",
            "lai",
        ],
        "Full Name": [
            "Rosette Diameter",
            "Rosette Area",
            "Relative Growth Rate",
            "Leaf Count",
            "Leaf Area Index",
        ],
        "Formula / Method": [
            "Bounding circle diameter of green mask (pixels)",
            "Total green pixel count",
            "ln(cover_today / cover_yesterday)",
            "SAM2 instance segmentation (watershed fallback)",
            "−ln(1 − canopy_fraction) / 0.5  (Beer-Lambert, k=0.5)",
        ],
        "What it tells you": [
            "Physical spread of the rosette across the image.",
            "Raw size of the canopy in pixels.",
            "Day-on-day proportional growth speed. Positive = growing, negative = shrinking.",
            "Number of individual leaves detected. Useful for developmental stage tracking.",
            "Modelled leaf area per unit ground area. Corrected with canopy height when depth data is available.",
        ],
    }
    st.dataframe(pd.DataFrame(growth_data), use_container_width=True, hide_index=True)

    # ── Health Metrics ────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Health Metrics</div>", unsafe_allow_html=True)

    health_data = {
        "Metric": [
            "chlorosis_pct",
            "necrosis_pct",
            "curl_score",
            "symmetry_score",
            "health_score",
        ],
        "Full Name": [
            "Chlorosis %",
            "Necrosis %",
            "Leaf Curl Score",
            "Rosette Symmetry",
            "Composite Health Score",
        ],
        "Method": [
            "% of green pixels with yellowing hue (HSV H: 15–30°)",
            "% of green pixels with brown/dead tissue (HSV H: 5–15°, low saturation)",
            "Ratio of perimeter² to area — high values indicate curling or irregular edges",
            "Comparison of left/right and top/bottom halves of the green mask",
            "Weighted combination: 40% chlorosis, 30% necrosis, 15% curl, 15% symmetry — scaled 0–100",
        ],
        "What it tells you": [
            "Early nutrient deficiency or stress. Yellowing before visible wilting.",
            "Late-stage stress or disease. Brown/dead tissue.",
            "Physical leaf deformation — heat, drought, or disease stress.",
            "Whether the rosette is growing evenly. Asymmetry can indicate localised stress.",
            "Single summary score. 100 = fully healthy, 0 = severe stress across all indicators.",
        ],
    }
    st.dataframe(pd.DataFrame(health_data), use_container_width=True, hide_index=True)

    # ── Depth / 3D Metrics ────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Depth & 3D Metrics</div>", unsafe_allow_html=True)

    depth_data = {
        "Metric": [
            "soil_baseline_mm",
            "canopy_height_mean_mm",
            "canopy_height_max_mm",
            "canopy_volume_cm3",
        ],
        "Full Name": [
            "Soil Baseline Depth",
            "Mean Canopy Height",
            "Max Canopy Height",
            "Canopy Volume",
        ],
        "Method": [
            "Median depth of non-plant valid pixels from OAK-D Lite stereo depth map (16-bit PNG, mm)",
            "Mean of (soil_baseline − plant_pixel_depth) for valid plant pixels",
            "5th percentile of plant pixel heights (robust to stereo spikes)",
            "Sum of per-pixel heights × pixel area (2.34 mm² at ~1m) ÷ 1000",
        ],
        "What it tells you": [
            "Reference surface. Used to calculate plant height relative to soil.",
            "Average vertical extent of the canopy above soil. Increases as plants grow upward.",
            "Tallest point of the canopy — indicates bolting stem height once plants flower.",
            "3D estimate of plant biomass volume. More informative than 2D area alone.",
        ],
    }
    st.dataframe(pd.DataFrame(depth_data), use_container_width=True, hide_index=True)

    # ── Ground Truth Metrics ──────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Ground Truth (LI-600 & SPAD)</div>", unsafe_allow_html=True)

    gt_data = {
        "Metric": ["gsw", "fs / fm_prime", "spad"],
        "Full Name": [
            "Stomatal Conductance",
            "Chlorophyll Fluorescence (Fs / Fm')",
            "SPAD Chlorophyll Index",
        ],
        "Instrument": ["LI-600 Porometer", "LI-600 Fluorometer module", "SPAD meter (separate)"],
        "Units": ["mol H₂O m⁻² s⁻¹", "Relative units", "SPAD units (0–99)"],
        "What it tells you": [
            "How open the stomata are. CO₂-enriched plants typically show reduced gsw as they photosynthesize more efficiently with less gas exchange needed.",
            "Fs = steady-state fluorescence, Fm' = maximum. Used to derive ΦPSII = (Fm'−Fs)/Fm' — operating efficiency of Photosystem II. Detects stress before visible symptoms.",
            "Leaf chlorophyll content proxy via transmittance at 650 nm vs 940 nm. Correlates with NGRDI from the camera.",
        ],
    }
    st.dataframe(pd.DataFrame(gt_data), use_container_width=True, hide_index=True)

    st.info(
        "**Regression target:** NGRDI is the strongest RGB predictor of gsw (R²≈0.42, Jakunskas et al. 2025). "
        "The machine learning model will use NGRDI, VARI, ExG, LAI, chlorosis_pct, and canopy height "
        "to predict gsw, ΦPSII, and SPAD from images alone."
    )
