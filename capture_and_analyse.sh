#!/bin/bash
# capture_and_analyse.sh — Daily capture and analysis for both chambers
# EE496 | Luke Buckley | Maynooth University
#
# Runs automatically via cron on Raspberry Pi at noon daily.
# Captures images from both OAK-D Lite cameras, runs analysis pipeline.
#
# Cron entry (run at 12:00 PM daily):
#   0 12 * * * /home/pi/Growth_Chamber_cv/capture_and_analyse.sh >> /home/pi/Growth_Chamber_cv/results/scheduler_log.txt 2>&1
#
# Manual run:
#   bash capture_and_analyse.sh

set -e  # exit on any error

PROJECT="/home/pi/Growth_Chamber_cv"
PYTHON="$PROJECT/scripts/.venv/bin/python"
DATE=$(date +%Y-%m-%d)

echo ""
echo "========================================="
echo "  Growth Chamber — Daily Run — $DATE"
echo "========================================="

# ── Enriched chamber ──────────────────────────────────────────────────────────
echo ""
echo "[1/4] Capturing enriched chamber..."
$PYTHON "$PROJECT/scripts/capture_image.py" --chamber enriched

echo "[2/4] Analysing enriched chamber..."
$PYTHON "$PROJECT/scripts/analyse_chamber.py" \
    --image "$PROJECT/images/enriched/${DATE}_enriched.jpg" \
    --chamber enriched

# ── Control chamber ───────────────────────────────────────────────────────────
echo ""
echo "[3/4] Capturing control chamber..."
$PYTHON "$PROJECT/scripts/capture_image.py" --chamber control

echo "[4/4] Analysing control chamber..."
$PYTHON "$PROJECT/scripts/analyse_chamber.py" \
    --image "$PROJECT/images/control/${DATE}_control.jpg" \
    --chamber control

echo ""
echo "Done — $DATE"
echo "========================================="
