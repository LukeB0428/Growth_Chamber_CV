#!/usr/bin/env bash
# Mechanical Pi setup for the monitoring data layer. Run from the repo on the Pi.
# Does NOT enable services or fill secrets/serials — see deploy/BRINGUP.md for the
# ordered runbook (schema, creds, udev serials, firmware flash come first).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/scripts/.venv"

echo "[1/4] installing python deps into $VENV"
"$VENV/bin/pip" install -r "$ROOT/scripts/requirements_monitoring.txt"

echo "[2/4] installing systemd units"
sudo cp "$ROOT/deploy/serial_ingest.service" "$ROOT/deploy/env_ingest.service" /etc/systemd/system/

echo "[3/4] installing udev rules (FILL IN BOARD SERIALS FIRST — BRINGUP.md step 3)"
sudo cp "$ROOT/deploy/99-growthchamber-arduino.rules" /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger || true
sudo systemctl daemon-reload

echo "[4/4] done. NEXT (manual, see deploy/BRINGUP.md):"
echo "  - create .env.monitoring (creds)"
echo "  - run supabase/schema.sql in Supabase"
echo "  - confirm /dev/ttyACM-enriched and /dev/ttyACM-control resolve"
echo "  - sudo systemctl enable --now serial_ingest env_ingest"
echo "  - scripts/.venv/bin/python scripts/verify_ingest.py"
