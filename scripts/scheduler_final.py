"""
scheduler_final.py — Daily CV Pipeline Scheduler for Growth Chamber CV
EE496 | Luke Buckley | Maynooth University

Runs at 12:00 PM each day. For each chamber (enriched, control):
  1. Captures RGB + depth image from OAK-D Lite via capture_image.py
  2. Runs analyse_image.py on the captured image
  3. Appends results to results/metrics.csv
  4. Copies metrics.csv and today's images to Google Drive sync folder

If camera capture fails, falls back to the most recent existing image in
images/{chamber}/ so a camera hiccup does not lose a whole day's analysis.

Usage:
    python scheduler.py
    python scheduler.py --now     # run immediately for testing
"""

import schedule
import time
import subprocess
import shutil
import logging
import sys
import argparse
from pathlib import Path
from datetime import datetime
from config import (PYTHON_BIN, CAPTURE_SCRIPT, ANALYSE_SCRIPT,
                    IMAGES_DIR, RESULTS_DIR, METRICS_CSV, POT_METRICS_CSV, SCHEDULER_LOG)

# ── CONFIG ────────────────────────────────────────────────────────────────────
CHAMBERS     = ["enriched", "control"]

PYTHON         = PYTHON_BIN
GDRIVE_BACKUP_FOLDER = Path(r"G:\My Drive\Growth_Chamber_Backup")

POT_METRICS = POT_METRICS_CSV
STATUS_LOG  = SCHEDULER_LOG

CAPTURE_TIMEOUT  = 60    # seconds — camera warmup + frame grab
ANALYSIS_TIMEOUT = 300   # seconds — SAM2 ~60s + other pipeline stages

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(STATUS_LOG, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def capture_image(chamber, today_str):
    """Call capture_image.py to grab RGB + depth from OAK-D Lite.
    Returns Path to saved RGB image, or None on failure."""
    cmd = [str(PYTHON), str(CAPTURE_SCRIPT), "--chamber", chamber]
    log.info(f"[{chamber}] Capturing image from OAK-D Lite...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=CAPTURE_TIMEOUT)
        if result.returncode == 0:
            expected = IMAGES_DIR / chamber / f"{today_str}_{chamber}.jpg"
            if expected.exists():
                log.info(f"[{chamber}] Image captured: {expected.name}")
                return expected
            log.error(f"[{chamber}] Capture ran but image not found at {expected}")
            return None
        log.error(f"[{chamber}] Capture failed (exit {result.returncode}): {result.stderr.strip()}")
        return None
    except subprocess.TimeoutExpired:
        log.error(f"[{chamber}] Camera capture timed out after {CAPTURE_TIMEOUT}s.")
        return None
    except Exception as e:
        log.error(f"[{chamber}] Unexpected error during capture: {e}")
        return None


def find_fallback_image(chamber):
    """Return most recent image in images/{chamber}/ as a fallback."""
    chamber_dir = IMAGES_DIR / chamber
    if not chamber_dir.exists():
        return None
    candidates = sorted(
        [f for f in chamber_dir.glob("*")
         if f.suffix.lower() in (".jpg", ".jpeg", ".png") and "depth" not in f.name],
        key=lambda f: f.stat().st_mtime, reverse=True,
    )
    return candidates[0] if candidates else None


def run_analysis(image_path, chamber):
    """Call analyse_image.py on image_path. Returns True on success."""
    cmd = [
        str(PYTHON), str(ANALYSE_SCRIPT),
        "--image",            str(image_path),
        "--chamber",          chamber,
        "--csv",              str(METRICS_CSV),
        "--no-auto-calibrate",
    ]
    log.info(f"[{chamber}] Running analysis on {image_path.name}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=ANALYSIS_TIMEOUT)
        if result.returncode == 0:
            log.info(f"[{chamber}] Analysis complete. {result.stdout.strip()}")
            return True
        log.error(f"[{chamber}] Analysis failed (exit {result.returncode}): {result.stderr.strip()}")
        return False
    except subprocess.TimeoutExpired:
        log.error(f"[{chamber}] Analysis timed out after {ANALYSIS_TIMEOUT}s.")
        return False
    except Exception as e:
        log.error(f"[{chamber}] Unexpected error: {e}")
        return False


def backup_to_gdrive(today_str, images):
    """Copy metrics.csv and today's images to Google Drive."""
    if not GDRIVE_BACKUP_FOLDER.exists():
        log.warning(f"Google Drive folder not found: {GDRIVE_BACKUP_FOLDER}. Skipping backup.")
        return False
    backup_dir = GDRIVE_BACKUP_FOLDER / today_str
    backup_dir.mkdir(parents=True, exist_ok=True)
    success = True
    try:
        shutil.copy2(METRICS_CSV, backup_dir / "metrics.csv")
        log.info("Backed up metrics.csv")
    except Exception as e:
        log.error(f"Failed to back up metrics.csv: {e}")
        success = False
    if POT_METRICS.exists():
        try:
            shutil.copy2(POT_METRICS, backup_dir / "pot_metrics.csv")
            log.info("Backed up pot_metrics.csv")
        except Exception as e:
            log.error(f"Failed to back up pot_metrics.csv: {e}")
            success = False
    for img in images:
        try:
            shutil.copy2(img, backup_dir / img.name)
            depth = img.parent / img.name.replace(".jpg", "_depth.png")
            if depth.exists():
                shutil.copy2(depth, backup_dir / depth.name)
            log.info(f"Backed up {img.name}")
        except Exception as e:
            log.error(f"Failed to back up {img.name}: {e}")
            success = False
    return success


# ── MAIN DAILY JOB ────────────────────────────────────────────────────────────

def daily_run():
    today_str       = datetime.now().strftime("%Y-%m-%d")
    images_captured = []
    any_failure     = False

    log.info("=" * 60)
    log.info(f"DAILY RUN START -- {today_str}")
    log.info("=" * 60)

    for chamber in CHAMBERS:
        log.info(f"--- Chamber: {chamber} ---")

        # Step 1: capture from camera
        image = capture_image(chamber, today_str)

        # Step 2: fallback if capture failed
        if image is None:
            log.warning(f"[{chamber}] Camera failed -- trying fallback image.")
            image = find_fallback_image(chamber)
            if image is None:
                log.error(f"[{chamber}] No image available -- skipping.")
                any_failure = True
                continue
            log.warning(f"[{chamber}] Using fallback: {image.name}")

        images_captured.append(image)

        # Step 3: run full analysis pipeline
        if not run_analysis(image, chamber):
            any_failure = True

    # Step 4: back up to Google Drive
    backup_ok = False
    if images_captured or METRICS_CSV.exists():
        backup_ok = backup_to_gdrive(today_str, images_captured)
    else:
        log.warning("No images and no CSV -- skipping backup.")

    status = "COMPLETED WITH ERRORS" if any_failure else "OK"
    log.info(f"DAILY RUN {status} -- backup {'OK' if backup_ok else 'FAILED/SKIPPED'}")
    log.info("=" * 60)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Growth Chamber daily scheduler.")
    parser.add_argument("--now", action="store_true",
                        help="Run the daily job immediately (for testing)")
    args = parser.parse_args()

    log.info("Scheduler started. Daily run at 12:00 PM.")
    log.info(f"Project root : {RESULTS_DIR.parent}")
    log.info(f"Chambers     : {', '.join(CHAMBERS)}")
    log.info(f"Drive backup : {GDRIVE_BACKUP_FOLDER}")
    log.info("Press Ctrl+C to stop.")

    if args.now:
        log.info("--now flag -- running immediately.")
        daily_run()

    schedule.every().day.at("12:00").do(daily_run)

    while True:
        schedule.run_pending()
        time.sleep(30)
