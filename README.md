# Growth Chamber CV

Automated plant phenotyping pipeline for *Arabidopsis thaliana* under elevated CO₂.

Two OAK-D Lite stereo cameras capture daily overhead RGB and depth images of sealed mini-chambers — one enriched (1100ppm) and one control. A Raspberry Pi runs the pipeline autonomously at noon every day and stores results to CSV.

---

## Hardware

| Component | Detail |
|-----------|--------|
| Cameras | OAK-D Lite stereo depth (×2) |
| Compute | Raspberry Pi 4 (autonomous capture + analysis) |
| Chambers | 2 sealed mini-chambers, 8 pots each |
| CO₂ control | Arduino + solenoid valve + CO₂ sensor |
| Environmental logging | Arduino + DHT22 (temp/humidity/CO₂ to SD card) |
| Remote access | Tailscale VPN over eduroam |

---

## Pipeline Overview

```
capture_image.py
      │
      ▼
image_quality.py  ──── quality gate (blur + brightness)
      │
      ▼
auto_calibrate.py ──── updates pot ROI positions (Hough circles)
      │
      ▼
analyse_chamber.py ─── per-pot loop
      │
      ├── greenness_metrics.py   (NGRDI, VARI, ExG, GCC, CIE Lab)
      ├── health_metrics.py      (chlorosis, necrosis, curl, symmetry, LAI)
      ├── health_score.py        (composite 0–100 score)
      ├── bolting_detection.py   (4-signal rule-based onset detection)
      ├── leaf_count.py          (SAM2 primary, watershed fallback)
      └── pot_metrics.csv
```

Daily schedule (12:00 PM, both chambers):  `scheduler_final.py` → backup to Google Drive

---

## Results

| Metric | Value |
|--------|-------|
| Experiment duration | April 9 – June 1, 2026 (47 imaging days) |
| Pot-day observations | 728 |
| Canopy cover peak | Enriched 10.9% vs Control 8.0% (+37%) |
| Bolting onset (enriched) | Mean day 29.1 (range 26–33) |
| Bolting onset (control) | Mean day 39.4 (range 30–52) |
| NGRDI advantage (Stage 3) | Enriched +0.022 higher from day 34 onward |

**Three-stage greenness pattern:**
- **Stage 1 (Apr 9–20):** Control initially greener (NGRDI −0.016)
- **Stage 2 (Apr 21–May 12):** Enriched overtakes (+0.016)
- **Stage 3 (May 13–Jun 1):** Enriched clearly greener (+0.022 NGRDI, +0.037 VARI)

---

## Metrics Recorded (per pot per day)

**Canopy geometry:** canopy cover %, rosette diameter (px), rosette area (px), relative growth rate

**Vegetation indices:** NGRDI (13 statistics: mean, median, std, IQR, skewness, kurtosis, Q1, Q3, min, max, range, mode, variance), VARI (13 statistics), ExG mean

**Colour:** mean hue, saturation, value, R, G, B, GCC, CIE L\*a\*b\*, greenness score (0–100), shade label

**Depth:** canopy height mean/max (mm), canopy volume (cm³), soil baseline (mm)

**Health:** chlorosis %, necrosis %, curl score, symmetry score, LAI, health score (0–100), health label

**Phenology:** germination flag/date, bolting flag/date/signals, leaf count, plant status

---

## Scripts

| Script | Purpose |
|--------|---------|
| `capture_image.py` | OAK-D Lite RGB + depth capture |
| `analyse_chamber.py` | Per-pot orchestrator → pot_metrics.csv |
| `analyse_image.py` | Whole-chamber analysis → metrics.csv |
| `auto_calibrate.py` | Automatic pot ROI detection (Hough circles) |
| `calibrate_pots.py` | Interactive one-time pot calibration GUI |
| `image_quality.py` | Blur + brightness quality gate |
| `greenness_metrics.py` | Vegetation index computation |
| `health_metrics.py` | Chlorosis, necrosis, curl, symmetry, LAI |
| `health_score.py` | Composite health score (0–100) |
| `bolting_detection.py` | 4-signal bolting onset detection |
| `leaf_count.py` | SAM2 + watershed leaf counter |
| `predict.py` | U-Net/ResNet34 segmentation inference |
| `train.py` | U-Net training on CVPPP A1 dataset |
| `dashboard.py` | Streamlit monitoring dashboard |
| `scheduler_final.py` | Daily pipeline orchestrator + Google Drive backup |
| `li600_log.py` | Manual LI-600 porometer/fluorometer data entry |
| `regression_analysis.py` | CV vs physiological metric correlation analysis |
| `statistical_comparison.py` | Mann-Whitney U + Cohen's d between chambers |
| `pca_analysis.py` | PCA phenotype separation visualisation |
| `timelapse.py` | MP4 timelapse from daily images |
| `live_view.py` | Real-time camera preview with depth/green overlay |
| `visualise_results.py` | Whole-chamber time-series plots |
| `visualise_pots.py` | Per-pot grid and heatmap plots |
| `repair_csv.py` | Pads old CSV rows to current schema |
| `reset_csv.py` | Resets CSVs to empty schema with headers |
| `config.py` | Cross-platform path configuration |

---

## File Structure

```
Growth_Chamber_cv/
├── images/
│   ├── enriched/          # YYYY-MM-DD_enriched.jpg + _depth.png
│   └── control/           # YYYY-MM-DD_control.jpg + _depth.png
├── results/
│   ├── metrics.csv        # Whole-chamber metrics (66 columns)
│   ├── pot_metrics.csv    # Per-pot metrics (51 columns)
│   └── ground_truth.csv   # LI-600 physiological readings
├── calibration/
│   ├── enriched_calibration.json
│   └── control_calibration.json
├── scripts/
│   └── *.py
├── arduino/
│   ├── co2_controller/    # CO₂ solenoid control firmware
│   └── env_logger/        # Temp/humidity/CO₂ logger firmware
└── .streamlit/
    └── config.toml        # Green/black dashboard theme
```

---

## Setup

### Raspberry Pi (autonomous capture)

```bash
git clone <repo>
cd Growth_Chamber_cv
python3 -m venv scripts/.venv
source scripts/.venv/bin/activate
pip install -r requirements.txt
```

Schedule daily capture (cron):
```
0 12 * * * /home/pi/Growth_Chamber_cv/scripts/.venv/bin/python /home/pi/Growth_Chamber_cv/scripts/scheduler_final.py
```

### Dashboard

```bash
# Local
scripts\.venv\Scripts\streamlit run scripts\dashboard.py

# Network accessible
scripts\.venv\Scripts\streamlit run scripts\dashboard.py --server.address 0.0.0.0
```

---

## Segmentation Model

U-Net with ResNet34 encoder trained on the [CVPPP 2014 A1 dataset](https://www.plant-phenotyping.org/CVPPP2014-dataset). Used as the high-accuracy segmentation backend when `--method model` is specified; HSV thresholding is the default fast method.

Leaf counting: SAM2 zero-shot instance segmentation (primary), watershed (fallback). Mean absolute error on validation set: **3.94 leaves**.

---

## Ground Truth Validation

LI-600 porometer/fluorometer measurements collected weekly per pot:
- **gsw** — stomatal conductance (mol m⁻² s⁻¹)
- **phi_psii** — PSII operating efficiency, derived as (Fm′ − Fs) / Fm′
- **spad** — relative chlorophyll content (mean of 3 readings)

Regression analysis against CV metrics via `regression_analysis.py`.

---

*EE496 Final Year Project — Maynooth University, 2025–2026*
