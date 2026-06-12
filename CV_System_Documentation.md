# Growth Chamber CV — Computer Vision System Documentation

**Project:** EE496 Final Year Project, Maynooth University, 2025–2026
**Author:** Luke Buckley
**System:** Automated plant phenotyping pipeline for *Arabidopsis thaliana* under elevated CO₂

---

## 1. System Overview

This system automates the daily monitoring of two sealed mini plant growth chambers using computer vision. A Raspberry Pi connected to two OAK-D Lite stereo depth cameras captures overhead RGB images and depth maps of both chambers at noon every day. A Python pipeline then extracts quantitative growth metrics for each of the 8 plant pots per chamber — without any manual intervention.

**Why it is useful:** Traditional plant phenotyping requires a researcher to be present, manually measure plants with rulers or spad meters, and record data by hand. This system replaces that with continuous automated measurement, capturing subtle day-to-day changes that would be missed by weekly manual checks. It produces a timestamped CSV record of over 50 metrics per pot per day, enabling statistical analysis of how elevated CO₂ affects plant growth, physiology, and reproductive timing.

### Experimental Setup

| Chamber | CO₂ Level | Pots |
|---------|-----------|------|
| Enriched | 800–1200 ppm (elevated) | 8 × *Arabidopsis thaliana* |
| Control | ~420 ppm (ambient) | 8 × *Arabidopsis thaliana* |

Cameras are mounted overhead looking straight down at ~1 m height. Each camera captures:
- **RGB image:** 1920×1080 px JPEG
- **Depth map:** 640×400 px 16-bit PNG (stereo disparity, values in mm)

---

## 2. Pipeline Architecture

The analysis pipeline runs in this order for each chamber daily:

```
1. capture_image.py       → Raw images saved to disk
2. image_quality.py       → Quality gate (rejects blurry or dark images)
3. auto_calibrate.py      → Updates pot circle positions (Hough detection)
4. analyse_chamber.py     → Per-pot loop:
   a. Crop pot ROI from full image
   b. greenness_metrics.py   → Vegetation indices
   c. health_metrics.py      → Stress indicators
   d. health_score.py        → Composite health score
   e. bolting_detection.py   → Reproductive phase detection
   f. leaf_count.py          → Individual leaf counting
   g. Append row to pot_metrics.csv
5. scheduler_final.py     → Backs up CSVs to Google Drive
```

All results are stored in `results/pot_metrics.csv` — one row per pot per day.

---

## 3. Script Reference

---

### 3.1 `capture_image.py`

**What it does:**
Connects to a specific OAK-D Lite stereo camera (identified by serial number) and captures one RGB frame and one depth frame simultaneously. Saves the RGB image as a JPEG (1920×1080) and the depth map as a 16-bit PNG (640×400) with a matched preview JPEG.

**Key inputs:**
- `--chamber enriched` or `--chamber control`
- Camera serial numbers mapped in `camera_map.json`

**Key outputs:**
- `images/{chamber}/YYYY-MM-DD_{chamber}.jpg`
- `images/{chamber}/YYYY-MM-DD_{chamber}_depth.png`
- `images/{chamber}/YYYY-MM-DD_{chamber}_depth_preview.jpg`

**Why it is useful:**
Provides a consistent, automated daily image at the same time each day from a fixed camera position. The 16-bit depth PNG preserves full millimetre-resolution stereo data needed for canopy height calculations. Without this, images would need to be taken manually and filed by hand.

---

### 3.2 `analyse_chamber.py`

**What it does:**
The main per-pot orchestrator. Loads the chamber's calibration JSON (8 pot circle positions), crops each pot from the full image as a circular ROI, and runs the complete metric pipeline on each pot independently. Tracks plant status across days (healthy / warning / declining / dead) and writes one CSV row per pot.

**Key inputs:**
- Full RGB image path
- Chamber ID
- `calibration/{chamber}_calibration.json` (8 circle centres + radii)
- Optional depth map

**Key outputs:**
- `results/pot_metrics.csv` — 51 metrics per pot per day

**Why it is useful:**
By treating each pot independently, the system can track individual plants rather than averaging over the whole chamber. This is essential for detecting events like a single plant dying or bolting earlier than others, and allows per-pot correlation with ground truth physiological measurements.

---

### 3.3 `analyse_image.py`

**What it does:**
Runs the same metric pipeline on a whole-chamber image rather than individual pots. Useful for whole-chamber trends and as a development/testing tool. Computes the same vegetation indices, depth metrics, health indicators, bolting detection, and leaf count for the entire chamber view.

**Key inputs:**
- RGB image path
- Chamber ID
- Optional depth PNG
- `--method hsv` (fast, default) or `--method model` (U-Net segmentation)

**Key outputs:**
- `results/metrics.csv` — 66 columns per chamber per day

**Why it is useful:**
Provides a quick whole-chamber health check and is used during development to test metric changes before deploying them to the per-pot pipeline.

---

### 3.4 `auto_calibrate.py`

**What it does:**
Automatically detects the positions and sizes of the 8 pot circles in a new image using the Hough Circle Transform. Rather than searching the whole image (which produces false positives on leaves and shadows), it searches within a small region of interest around each pot's known prior position. If fewer than 7 pots are detected confidently, it falls back to the previous calibration.

**Key inputs:**
- RGB image
- Chamber ID
- Prior `calibration/{chamber}_calibration.json`

**Key outputs:**
- Updated `calibration/{chamber}_calibration.json`
- Optional debug image showing detected circles

**How it works technically:**
1. Removes green pixels (plants) from the image to avoid false circle detections on leaf edges
2. Converts to grayscale
3. For each of the 8 prior pot positions, searches a small ROI with Hough Circle Transform
4. Accepts the best circle within tolerance; keeps the prior position if no good circle is found
5. Saves updated positions atomically (writes to temp file first, then renames)

**Why it is useful:**
Eliminates the need to manually re-run calibration after minor camera shifts or vibrations. The system stays accurate over weeks of autonomous operation without any human intervention.

---

### 3.5 `calibrate_pots.py`

**What it does:**
An interactive GUI tool for the one-time setup of pot positions in a new chamber or after a significant camera move. The user clicks on each of the 8 pot centres, adjusts the radius using the scroll wheel, and optionally renames each pot label. Saves the result as a calibration JSON.

**Key inputs:**
- RGB image path
- Chamber ID

**Key outputs:**
- `calibration/{chamber}_calibration.json`
- Preview image showing the 8 labelled circles

**Why it is useful:**
Only needs to be run once when setting up a new chamber or after physically moving a camera. After that, `auto_calibrate.py` handles daily drift automatically.

**Note:** Requires a physical display — cannot run headlessly over SSH.

---

### 3.6 `image_quality.py`

**What it does:**
A pre-analysis quality gate that checks two things before any metrics are computed:
1. **Blur:** Computes the Laplacian variance of the image. A low score means the image is blurry (out of focus or camera shake).
2. **Brightness:** Checks mean pixel brightness. Very dark or overexposed images are rejected.

Images that fail either check are logged but not analysed, preventing bad data from entering the CSV.

**Key inputs:**
- BGR image array
- Image file path
- Chamber ID

**Key outputs:**
- `(passed: bool, report: str)` — returned to the calling script
- `results/quality_log.csv` — timestamped log of all checks

**Why it is useful:**
Prevents corrupted or poorly-lit images from silently producing bad metrics that would skew trends. Common failure modes include the Pi capturing at a moment of low light, or a camera slightly shifting focus.

---

### 3.7 `config.py`

**What it does:**
Defines all file paths for the project as Python `Path` objects. A single source of truth for where images, CSVs, calibration files, and model weights are stored. Uses `pathlib` for cross-platform compatibility so the same code runs on both Windows (development laptop) and Linux (Raspberry Pi).

**Key exports:**
- `BASE_DIR` — project root
- `IMAGES_DIR`, `RESULTS_DIR`, `CALIB_DIR`
- `METRICS_CSV`, `POT_METRICS_CSV`, `GROUND_TRUTH_CSV`
- `MODEL_PATH` — best_model.pth location

**Why it is useful:**
Every other script imports paths from here rather than hardcoding them. If the project is moved to a new machine, only this file needs updating.

---

### 3.8 `greenness_metrics.py`

**What it does:**
Computes 12 colour and vegetation index metrics from the green-masked canopy pixels of a pot image. Metrics are calibrated for *Arabidopsis thaliana* leaf colour under typical greenhouse lighting.

**Metrics computed:**

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| NGRDI | (G − R) / (G + R) | Normalised green-red difference; plant vigour |
| VARI | (G − R) / (G + R − B) | Visible atmospherically resistant index; chlorophyll proxy |
| ExG | 2G − R − B | Excess green; simple canopy detection |
| GCC | G / (R + G + B) | Green chromatic coordinate; illumination-invariant |
| CIE L* | — | Lightness |
| CIE a* | — | Green–red axis (negative = green, positive = red) |
| CIE b* | — | Blue–yellow axis |
| Greenness score | Weighted composite | 0–100 normalised health indicator |
| Green shade label | Threshold | Named label: yellow-green / light-green / mid-green / dark-green |

For NGRDI and VARI, 13 descriptive statistics are computed (mean, median, mode, std, variance, min, max, range, IQR, Q1, Q3, skewness, kurtosis).

**Why it is useful:**
Vegetation indices are more informative than raw RGB values because they are partially illumination-invariant and correlate with chlorophyll content and plant stress. NGRDI and VARI have been used in precision agriculture to detect water stress, nitrogen deficiency, and disease before they are visible to the naked eye.

---

### 3.9 `health_metrics.py`

**What it does:**
Computes five indicators of plant stress by analysing the colour and shape of canopy pixels:

1. **Chlorosis %** — Proportion of canopy pixels showing yellow-green colouration (characteristic of nitrogen deficiency or iron deficiency)
2. **Necrosis %** — Proportion showing brown/dead tissue
3. **Leaf curl score** — Ratio of canopy area to convex hull area (solidity). Healthy flat rosettes have high solidity; stressed curling leaves reduce it
4. **Symmetry score** — Radial symmetry of the canopy around its centroid. Healthy rosettes are rotationally symmetric; asymmetry indicates mechanical damage or uneven growth
5. **LAI (Leaf Area Index)** — Beer-Lambert approximation of canopy leaf area density, corrected using depth map data when available

Also saves an annotated health visualisation image showing healthy (green), chlorotic (yellow), and necrotic (red) regions.

**Why it is useful:**
These metrics can detect plant stress days before visible wilting, giving early warning of problems such as overwatering, nutrient deficiency, or disease. In the context of the CO₂ experiment, they allow comparison of plant stress levels between the enriched and control chambers.

---

### 3.10 `health_score.py`

**What it does:**
Combines six plant metrics into a single composite health score from 0 to 100, using a weighted sum. The six inputs are: chlorosis %, necrosis %, curl score, symmetry score, NGRDI mean, and canopy cover %. Stress metrics (chlorosis, necrosis) are inverted so that higher values always mean healthier.

**Score labels:**
- 75–100: Healthy
- 55–74: Mild stress
- 35–54: Moderate stress
- 0–34: Severe stress

**Why it is useful:**
Reduces the complexity of 50+ individual metrics to a single at-a-glance number. Useful for quickly identifying pots that need attention and for tracking overall chamber health over time on the dashboard.

---

### 3.11 `bolting_detection.py`

**What it does:**
Detects when an *Arabidopsis* plant transitions from vegetative (rosette) growth to reproductive (bolting) growth — the point at which the central inflorescence stem emerges. Uses four independent signals computed from daily image history:

| Signal | Method | What it detects |
|--------|--------|-----------------|
| DiamCover | Diameter/cover ratio trend over 5 days | Rosette stops expanding relative to cover area |
| Elongation | Ellipse aspect ratio of canopy | Elongated central structure of the emerging bolt |
| VARIdrop | VARI trend over 5 days | Greenness drops as reproductive tissue replaces leaves |
| DepthSpike | canopy_height_max_mm threshold | Sudden height increase from the bolt stalk |

**Firing rule:** Bolting is flagged when **2 or more signals fire**, with **VARIdrop or Elongation mandatory** (to prevent false positives from noise in geometry or depth).

When bolting is detected, the date is recorded in `bolting_date` and a visualisation image is saved.

**Why it is useful:**
Bolting marks the end of the vegetative phase and is a key phenological transition in plant biology. Elevated CO₂ is expected to accelerate this transition. Detecting exact onset day for each pot allows precise comparison of reproductive timing between chambers and correlation with physiological data.

---

### 3.12 `leaf_count.py`

**What it does:**
Counts the individual leaves of each plant using two methods:

1. **SAM2 (primary):** Meta's Segment Anything Model 2, used zero-shot. SAM2 generates instance segmentation masks for the canopy, which are filtered and counted. Checkpoint is auto-downloaded on first run.
2. **Watershed (fallback):** Classical morphological watershed segmentation on the distance transform of the green mask. Used if SAM2 is unavailable or fails.

Also detects **germination** — the first day canopy cover exceeds 0.5% after a period of zero coverage — and records the germination date.

**Accuracy:** Mean absolute error of 3.94 leaves on the CVPPP A1 validation set.

**Why it is useful:**
Leaf count is a standard plant phenotyping metric. It correlates with developmental stage and overall plant size. The automated count replaces tedious manual counting and enables tracking of leaf emergence rate over time.

---

### 3.13 `predict.py`

**What it does:**
Loads the trained U-Net segmentation model (ResNet34 encoder) and runs inference on a single BGR image to produce a binary plant/background mask. The model is cached in memory after the first call so it only loads once per session.

**Key inputs:**
- BGR image (any resolution)

**Key outputs:**
- Binary mask (same dimensions as input, values 0 or 255)

**Why it is useful:**
Provides a more accurate plant segmentation than HSV thresholding alone, especially for plants with unusual colouration (bolting, chlorosis) or in variable lighting. Used when `--method model` is specified in the analysis pipeline.

---

### 3.14 `train.py`

**What it does:**
Trains the U-Net segmentation model from scratch using the CVPPP 2014 A1 *Arabidopsis* dataset. The training loop monitors validation loss and saves the best-performing checkpoint as `best_model.pth`. Generates training curve plots (loss and IoU) at the end.

**Architecture:** U-Net with ResNet34 encoder (pre-trained on ImageNet), binary cross-entropy + Dice loss.

**Dataset:** CVPPP 2014 Plant Phenotyping Challenge, Dataset A1 — overhead RGB images of *Arabidopsis* rosettes with pixel-level plant/background labels.

**Why it is useful:**
Provides the deep learning backbone for high-accuracy segmentation when HSV thresholding fails. The CVPPP dataset is specifically *Arabidopsis thaliana* images, making it directly applicable to this experiment.

---

### 3.15 `dashboard.py`

**What it does:**
A Streamlit web dashboard that provides real-time monitoring of both growth chambers. Auto-reloads data every 60 seconds. Accessible from any device on the network via the Pi's Tailscale IP address.

**Dashboard pages:**

| Page | Contents |
|------|---------|
| Overview | Live chamber comparison, latest health scores, pot status cards |
| Growth Trends | Time-series plots for canopy cover, vegetation indices, RGR |
| Per-Pot Dashboard | Individual pot time series, heatmaps, bolting timeline |
| Ground Truth | LI-600 physiological data vs CV metric correlation |
| Run Analysis | Manual trigger to re-capture or re-analyse an image |
| Live View | Real-time camera feed with depth and green mask overlays |
| Statistics | PCA, Mann-Whitney U tests, Cohen's d effect sizes |

**Why it is useful:**
Allows monitoring of the experiment from anywhere (phone, laptop) without needing to SSH into the Pi or open CSV files. The visual interface makes it easy to spot problems — a sudden canopy drop, a health score decline, or a bolting event — at a glance.

---

### 3.16 `scheduler_final.py`

**What it does:**
The daily pipeline orchestrator. Triggered by a cron job at 12:00 PM on the Raspberry Pi. For each chamber in sequence: captures a new image, runs the full analysis pipeline, and backs up the CSVs and images to Google Drive. If the camera capture fails, it falls back to re-analysing the most recent available image.

**Why it is useful:**
Makes the entire system fully autonomous. Once set up, no human intervention is needed for daily data collection. The Google Drive backup ensures data is not lost if the Pi's SD card fails.

---

### 3.17 `li600_log.py`

**What it does:**
A guided command-line tool for entering LI-600 porometer/fluorometer and SPAD meter readings after a manual measurement session in the lab. For each pot, it displays an annotated image of the pot (with detected leaves numbered), then prompts for readings leaf-by-leaf with a 30-second stabilisation countdown. Automatically derives `phi_psii = (Fm' − Fs) / Fm'` from the fluorometer values. Averages 3 SPAD readings per pot. Saves all data to `ground_truth.csv`.

**Metrics logged:**
- **gsw** — stomatal conductance (mol H₂O m⁻² s⁻¹)
- **phi_psii** — PSII operating efficiency (dimensionless, 0–1)
- **spad** — relative chlorophyll content

**Why it is useful:**
Provides physiological ground truth to validate the computer vision metrics. For example, NGRDI can be correlated with SPAD (chlorophyll proxy), and canopy cover can be correlated with stomatal conductance. This validation is essential for demonstrating that the CV metrics are biologically meaningful.

---

### 3.18 `regression_analysis.py`

**What it does:**
Merges `pot_metrics.csv` with `ground_truth.csv` (matched by chamber, pot label, and date) and computes pairwise correlations between all CV metrics and the three physiological ground truth metrics (gsw, phi_psii, spad). Reports Pearson and Spearman correlation coefficients. Generates a correlation heatmap and scatter plots for the strongest relationships.

**Why it is useful:**
Quantifies how well the non-destructive CV metrics (such as NGRDI, canopy cover, health score) predict actual plant physiology measured by the LI-600 instrument. Strong correlations validate the usefulness of the imaging pipeline; weak correlations identify which CV metrics are less informative.

---

### 3.19 `statistical_comparison.py`

**What it does:**
Compares enriched and control chambers using rigorous statistical testing. For each metric, computes daily means across all live pots per chamber, then runs a Mann-Whitney U test (non-parametric, appropriate for non-normally distributed data) and Cohen's d effect size. Produces publication-quality plots of growth curves with 95% confidence interval shading.

**Why it is useful:**
Provides the statistical evidence needed to make claims such as "enriched chamber plants grew significantly larger" in a scientific paper or presentation. The Mann-Whitney test is used rather than a t-test because phenotyping data is often non-normal and the sample size per chamber is small (8 pots).

---

### 3.20 `pca_analysis.py`

**What it does:**
Runs Principal Component Analysis on 12 CV metrics to reduce dimensionality and visualise how well the imaging data separates enriched and control plants. Produces four output plots: scatter plot (PC1 vs PC2, coloured by chamber), feature loadings (which metrics contribute to each PC), scree plot (variance explained), and trajectory plot (how the chambers move through PCA space over time).

**Why it is useful:**
PCA is a standard technique for multivariate phenotyping data. It shows whether enriched and control plants are phenotypically distinct as a whole — rather than examining each metric in isolation — and identifies which combination of metrics drives the difference.

---

### 3.21 `timelapse.py`

**What it does:**
Compiles all daily images for a given chamber into an MP4 video at 1 frame per second (each second = one day of growth). Each frame is annotated with the date and key metrics from that day. Also generates 6 publication-quality metric plots comparing both chambers over the full experiment duration.

**Why it is useful:**
Provides an intuitive visual summary of the entire experiment for presentations and reports. The timelapse makes subtle growth differences between chambers immediately apparent to an audience.

---

### 3.22 `live_view.py`

**What it does:**
Opens a real-time preview window from the OAK-D Lite camera. Displays the RGB feed with optional overlays: depth colourmap (false-colour distance), green mask (detected plant pixels), and calibration pot circles. Pressing S saves a snapshot; pressing Q quits.

**Why it is useful:**
Used during physical setup to verify camera alignment, check that all 8 pot circles are visible, and confirm that the depth sensor is functioning correctly. Also used to verify the green mask is correctly segmenting plants and not picking up soil or shadows.

---

### 3.23 `visualise_results.py`

**What it does:**
Reads `metrics.csv` and generates 6 time-series plots comparing enriched and control chamber-level metrics over the full experiment: canopy cover %, ExG, VARI, rosette diameter, relative growth rate, and a multi-metric summary. Saves all plots as high-resolution PNGs.

---

### 3.24 `visualise_pots.py`

**What it does:**
Reads `pot_metrics.csv` and generates per-pot visualisations: a 2×4 grid plot for each metric showing each pot's time series individually; a spatial heatmap showing metric values across all 8 pot positions for the most recent day; and a chamber summary plot with mean ± standard deviation shading.

---

### 3.25 `repair_csv.py`

**What it does:**
One-time utility to fix `metrics.csv` rows written by older versions of the analysis scripts that had fewer columns. Pads missing columns with empty values and rewrites the file with the current 66-column schema. Creates a timestamped backup before modifying anything.

---

### 3.26 `reset_csv.py`

**What it does:**
Resets `metrics.csv` and `pot_metrics.csv` to empty files with only the correct column headers. Creates timestamped backups first. Used when starting a new experiment or after a misconfigured analysis has written corrupt data.

---

## 4. Data Files

### `results/pot_metrics.csv`

One row per pot per imaging day. 51 columns including:
- Timestamp, chamber, pot_label, method
- Canopy geometry: canopy_cover_%, rosette_diameter_px, rosette_area_px, rgr
- Vegetation indices: ngrdi_mean + 12 stats, vari_mean + 12 stats, exg_mean
- Colour: mean_hue, mean_saturation, mean_value, mean_r, mean_g, mean_b, gcc, lab_L, lab_a, lab_b
- Greenness: greenness_score, green_shade
- Depth: canopy_height_mean_mm, canopy_height_max_mm, canopy_volume_cm3, soil_baseline_mm
- Health: chlorosis_pct, necrosis_pct, curl_score, symmetry_score, lai, health_score, health_label
- Phenology: germination_flag, germination_date, bolting_flag, bolting_date, bolting_signals, leaf_count
- Status: plant_status, image_file

### `results/ground_truth.csv`

One row per pot per measurement session. Columns: date, chamber, pot_label, gsw, vpleaf, vpdleaf, h2oleaf, fs, fm_prime, spad, phi_psii.

### `calibration/{chamber}_calibration.json`

List of 8 objects, each with: label (e.g. "Control_Pot1"), x, y (centre pixels), r (radius pixels).

---

## 5. Depth Measurement Notes

- Depth maps are 16-bit PNG, values in millimetres from camera
- Pixel area at ~1 m height: approximately 2.34 mm² (OAK-D Lite ~73° HFOV, 640 px width)
- Minimum reliable stereo range: ~200 mm (camera must be > 20 cm above soil)
- Depth metrics are unreliable for early-stage plants with < 5% canopy cover due to stereo artefacts on sparse surfaces
- A 5×5 median filter is applied to suppress stereo false-match speckle noise before height calculations
- `MAX_PLANT_HEIGHT_MM = 300 mm` — pixels above this threshold are treated as noise and excluded

---

## 6. Key Results (April 9 – June 1, 2026)

- **47 imaging days**, 728 pot-day observations
- **Canopy cover:** Enriched peaked at 10.9% vs control 8.0% at week 5 (+37% relative)
- **Three-stage greenness pattern confirmed:**
  - Stage 1 (Apr 9–20): Control initially greener (NGRDI −0.016)
  - Stage 2 (Apr 21–May 12): Enriched overtakes (+0.016)
  - Stage 3 (May 13–Jun 1): Enriched clearly greener (+0.022 NGRDI, +0.037 VARI)
- **Bolting — enriched:** 7/7 surviving pots, mean onset day 29.1 (range 26–33, ~May 5–12)
- **Bolting — control:** Mean onset day 39.4 (range 30–52), ~10 days later and less synchronous
- **Plant survival:** 1 loss per chamber (enriched Pot3 died day 15; control Pot1 damaged day 43)

---

*Generated from Growth_Chamber_cv pipeline — Maynooth University EE496*
