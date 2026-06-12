"""
timelapse.py — Timelapse Video and Results Visualisation for Growth Chamber Project
EE496 | Luke Buckley | Maynooth University

Stage 11 — Timelapse and Visualisation

Two main functions:

1. TIMELAPSE VIDEO
   Compiles all daily images from a chamber into an MP4 timelapse video.
   Images are sorted by date, resized to a consistent resolution, and
   annotated with the date and key metrics for that day.

2. RESULTS PLOTS
   Reads metrics.csv and generates publication-quality trend plots for
   all metrics over the trial period, comparing enriched vs control chambers.

Usage:
    # Generate timelapse for both chambers
    python timelapse.py --timelapse

    # Generate all results plots
    python timelapse.py --plots

    # Generate both
    python timelapse.py --timelapse --plots
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob
import argparse
from datetime import datetime
from config import BASE_DIR, IMAGES_DIR, RESULTS_DIR, METRICS_CSV, PLOTS_DIR


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

IMAGES_DIR    = str(IMAGES_DIR)
RESULTS_DIR   = str(RESULTS_DIR)
TIMELAPSE_DIR = str(BASE_DIR / "timelapse")
METRICS_CSV   = str(METRICS_CSV)
PLOTS_DIR     = str(PLOTS_DIR)

CHAMBERS      = ["enriched", "control"]
CHAMBER_COLOURS = {"enriched": "#2ecc71", "control": "#9C27B0"}

# Timelapse settings
TIMELAPSE_FPS        = 2     # Frames per second — 0.5 seconds per day
TIMELAPSE_RESOLUTION = (1280, 960)  # Width x Height of output video


# ─────────────────────────────────────────────
# 1. TIMELAPSE VIDEO
# ─────────────────────────────────────────────

def build_timelapse(chamber_id, metrics_df=None):
    """
    Compiles daily images from a chamber into an annotated MP4 timelapse.

    Images must follow the naming convention: YYYY-MM-DD_chamber.jpg
    (as set by scheduler.py). They are sorted by date automatically.

    Each frame is annotated with:
      - Date
      - Day number in trial
      - Canopy cover % and leaf count (if available in CSV)

    Args:
        chamber_id : 'enriched' or 'control'
        metrics_df : optional DataFrame from metrics.csv for annotations
    """
    chamber_dir = os.path.join(IMAGES_DIR, chamber_id)
    if not os.path.isdir(chamber_dir):
        print(f"No image folder found for {chamber_id} at {chamber_dir}")
        return

    # Find all dated images
    image_files = sorted(glob.glob(os.path.join(chamber_dir, f"*_{chamber_id}.jpg")))
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(chamber_dir, "*.jpg")))
    if not image_files:
        print(f"No images found in {chamber_dir}")
        return

    print(f"Building timelapse for {chamber_id}: {len(image_files)} frames")

    os.makedirs(TIMELAPSE_DIR, exist_ok=True)
    output_path = os.path.join(TIMELAPSE_DIR, f"{chamber_id}_timelapse.mp4")

    # Set up video writer — try H.264 first, fall back to mp4v
    for codec in ("avc1", "mp4v", "XVID"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, TIMELAPSE_FPS, TIMELAPSE_RESOLUTION)
        if writer.isOpened():
            print(f"  Using codec: {codec}")
            break
    else:
        print("  ERROR: could not open VideoWriter with any codec")
        return

    for day_num, img_path in enumerate(image_files, start=1):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  Skipping unreadable image: {os.path.basename(img_path)}")
            continue

        # Resize to consistent resolution
        frame = cv2.resize(frame, TIMELAPSE_RESOLUTION)

        # Extract date from filename (YYYY-MM-DD_chamber.jpg)
        basename = os.path.basename(img_path)
        try:
            date_str = basename[:10]  # First 10 chars = YYYY-MM-DD
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            date_display = date_obj.strftime("%d %b %Y")
        except ValueError:
            date_display = basename

        # Get metrics for this date/chamber from CSV if available
        canopy_str = ""
        leaf_str   = ""
        if metrics_df is not None and not metrics_df.empty:
            day_rows = metrics_df[
                (metrics_df['chamber'] == chamber_id) &
                (metrics_df['timestamp'].str.startswith(date_str))
            ]
            if not day_rows.empty:
                row = day_rows.iloc[-1]
                try:
                    canopy_str = f"Canopy: {float(row['canopy_cover_%']):.1f}%"
                except:
                    pass
                try:
                    lc = row.get('leaf_count', '')
                    if lc and str(lc).strip():
                        leaf_str = f"Leaves: {int(float(lc))}"
                except:
                    pass

        # Draw semi-transparent black bar at bottom for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, TIMELAPSE_RESOLUTION[1]-90),
                      (TIMELAPSE_RESOLUTION[0], TIMELAPSE_RESOLUTION[1]),
                      (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Annotate frame
        font   = cv2.FONT_HERSHEY_SIMPLEX
        colour = (255, 255, 255)

        cv2.putText(frame, f"Day {day_num}  |  {date_display}",
                    (15, TIMELAPSE_RESOLUTION[1]-60), font, 0.7, colour, 2)
        cv2.putText(frame, f"{chamber_id.upper()}  {canopy_str}  {leaf_str}".strip(),
                    (15, TIMELAPSE_RESOLUTION[1]-25), font, 0.65, colour, 1)

        writer.write(frame)

    writer.release()

    # Re-encode to H.264/yuv420p so browsers can play it inline
    import subprocess
    tmp_path = output_path + '.tmp.mp4'
    try:
        os.rename(output_path, tmp_path)
        r = subprocess.run(
            ['ffmpeg', '-y', '-i', tmp_path,
             '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
             '-movflags', '+faststart', output_path],
            capture_output=True
        )
        if r.returncode == 0:
            os.remove(tmp_path)
            print(f"Timelapse saved to {output_path} (H.264)")
        else:
            os.rename(tmp_path, output_path)
            print(f"Timelapse saved to {output_path} (ffmpeg unavailable, may not play in browser)")
    except FileNotFoundError:
        if os.path.exists(tmp_path):
            os.rename(tmp_path, output_path)
        print(f"Timelapse saved to {output_path} (ffmpeg not found, may not play in browser)")


# ─────────────────────────────────────────────
# 2. RESULTS PLOTS
# ─────────────────────────────────────────────

def load_metrics():
    """Load and preprocess metrics CSV, tolerating mixed column counts from
    rows written by earlier versions of analyse_image.py."""
    if not os.path.isfile(METRICS_CSV):
        print(f"No metrics CSV found at {METRICS_CSV}")
        return None

    # on_bad_lines='warn' skips malformed rows rather than crashing
    # engine='python' is more tolerant of inconsistent column counts
    df = pd.read_csv(METRICS_CSV, on_bad_lines='warn', engine='python')
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values('timestamp')

    # Keep only HSV rows for consistency (or model if that's what was used)
    # If both methods present, default to hsv
    if 'method' in df.columns and df['method'].nunique() > 1:
        df = df[df['method'] == 'hsv']

    print(f"Loaded {len(df)} rows from metrics.csv")
    print(f"Chambers: {df['chamber'].unique().tolist()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def plot_metric(ax, df, metric_col, ylabel, title, chambers=CHAMBERS):
    """Plot a single metric over time for both chambers on a given axes."""
    for chamber in chambers:
        chamber_df = df[df['chamber'] == chamber].copy()
        if chamber_df.empty or metric_col not in chamber_df.columns:
            continue
        chamber_df = chamber_df.dropna(subset=[metric_col])
        if chamber_df.empty:
            continue

        ax.plot(chamber_df['timestamp'], chamber_df[metric_col],
                label=chamber.capitalize(),
                color=CHAMBER_COLOURS.get(chamber, 'grey'),
                marker='o', markersize=4, linewidth=2)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def generate_plots(df):
    """Generate all results plots from the metrics DataFrame."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── Plot 1: Core growth metrics (2x2) ──────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Growth Metrics — Enriched vs Control", fontsize=13, fontweight='bold')

    plot_metric(axes[0,0], df, 'canopy_cover_%',     'Canopy Cover (%)',    'Canopy Cover')
    plot_metric(axes[0,1], df, 'exg_mean',            'ExG',                 'Excess Green Index (ExG)')
    plot_metric(axes[1,0], df, 'vari_mean',            'VARI',                'Vegetation Index (VARI)')
    plot_metric(axes[1,1], df, 'rosette_diameter_px', 'Diameter (px)',        'Rosette Diameter')

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "growth_metrics.png")
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # ── Plot 2: Growth rate ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle("Relative Growth Rate — Enriched vs Control", fontsize=13, fontweight='bold')
    plot_metric(ax, df, 'rgr', 'RGR (day⁻¹)', 'Relative Growth Rate')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "growth_rate.png")
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # ── Plot 3: Health metrics (2x3) ───────────────────────────────────
    health_cols = ['chlorosis_pct', 'necrosis_pct', 'curl_score',
                   'symmetry_score', 'lai', 'leaf_count']
    health_labels = ['Chlorosis (%)', 'Necrosis (%)', 'Curl Score',
                     'Symmetry Score', 'LAI', 'Leaf Count']
    health_titles = ['Chlorosis', 'Necrosis', 'Leaf Curl Score',
                     'Rosette Symmetry', 'Leaf Area Index', 'Leaf Count']

    available = [(c, l, t) for c, l, t in zip(health_cols, health_labels, health_titles)
                 if c in df.columns and df[c].notna().any()]

    if available:
        n  = len(available)
        nc = 3
        nr = (n + nc - 1) // nc
        fig, axes = plt.subplots(nr, nc, figsize=(14, nr * 4))
        fig.suptitle("Plant Health Metrics — Enriched vs Control",
                     fontsize=13, fontweight='bold')
        axes = np.array(axes).flatten()

        for i, (col, label, title) in enumerate(available):
            plot_metric(axes[i], df, col, label, title)

        # Hide unused subplots
        for j in range(len(available), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "health_metrics.png")
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

    # ── Plot 4: Bolting timeline ────────────────────────────────────────
    if 'bolting_flag' in df.columns and df['bolting_flag'].sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 3))
        fig.suptitle("Bolting Detection Timeline", fontsize=13, fontweight='bold')

        for chamber in CHAMBERS:
            chamber_df = df[df['chamber'] == chamber]
            bolting    = chamber_df[chamber_df['bolting_flag'] == 1]
            if not bolting.empty:
                for _, row in bolting.iterrows():
                    ax.axvline(row['timestamp'],
                               color=CHAMBER_COLOURS.get(chamber, 'grey'),
                               linewidth=2, linestyle='--',
                               label=f"{chamber.capitalize()} bolting")

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yticks([])
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "bolting_timeline.png")
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

    # ── Plot 5: Combined summary figure ────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    fig.suptitle("Growth Chamber Trial — Full Summary\nEnriched vs Control",
                 fontsize=14, fontweight='bold')

    plot_metric(axes[0,0], df, 'canopy_cover_%',  'Canopy Cover (%)',  'Canopy Cover')
    plot_metric(axes[0,1], df, 'rosette_diameter_px', 'Diameter (px)', 'Rosette Diameter')
    plot_metric(axes[1,0], df, 'vari_mean',        'VARI',              'Vegetation Index')
    plot_metric(axes[1,1], df, 'rgr',              'RGR (day⁻¹)',       'Relative Growth Rate')
    plot_metric(axes[2,0], df, 'leaf_count',       'Leaf Count',        'Leaf Count')
    plot_metric(axes[2,1], df, 'symmetry_score',   'Symmetry',          'Rosette Symmetry')

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "summary_figure.png")
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    print(f"\nAll plots saved to {PLOTS_DIR}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Timelapse and results visualisation.")
    parser.add_argument("--timelapse", action="store_true",
                        help="Build timelapse videos for both chambers")
    parser.add_argument("--plots",     action="store_true",
                        help="Generate results plots from metrics.csv")
    parser.add_argument("--chamber",   default=None,
                        help="Limit timelapse to one chamber (enriched or control)")
    args = parser.parse_args()

    if not args.timelapse and not args.plots:
        print("Specify --timelapse, --plots, or both.")
        print("Example: python timelapse.py --timelapse --plots")
        exit(0)

    df = load_metrics() if args.plots or args.timelapse else None

    if args.timelapse:
        chambers = [args.chamber] if args.chamber else CHAMBERS
        for chamber in chambers:
            build_timelapse(chamber, metrics_df=df)

    if args.plots:
        if df is not None:
            generate_plots(df)
        else:
            print("No metrics data available for plotting.")
