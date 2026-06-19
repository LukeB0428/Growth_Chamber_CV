# Tuesday bring-up runbook — monitoring data layer

Ordered; each step has a check. Budget ~30–45 min. Stop and fix if a check fails —
later steps depend on earlier ones. Full detail per command is in `deploy/README.md`.

## 0. Pull the branch
```bash
cd /home/pi/Growth_Chamber_cv && git fetch && git checkout monitoring-agent && git pull
```

## 1. Apply the Supabase schema
Run `supabase/schema.sql` in the Supabase SQL editor.
**Check:** tables `control_telemetry`, `observations`, `alerts`, `agent_reports` and view
`observations_unified` exist.

## 2. Credentials
Create `/home/pi/Growth_Chamber_cv/.env.monitoring` (`chmod 600`) with `SUPABASE_URL`,
`SUPABASE_KEY`, `EXPERIMENT_ID`, `ANTHROPIC_API_KEY` (see README).

## 3. Pin the two Arduinos — CRITICAL, before any rows
Plug ONE board at a time and read its serial:
```bash
udevadm info -a -n /dev/ttyACM0 | grep '{serial}' | head -1
```
Fill both serials into `deploy/99-growthchamber-arduino.rules`, then:
```bash
bash deploy/setup_pi.sh
```
**Check:** `ls -l /dev/ttyACM-enriched /dev/ttyACM-control` — both resolve. (If they don't,
the ingest will mislabel chambers — do not proceed.)

## 4. Flash firmware (BOTH boards) with one-time RTC recovery
For each `.ino`: uncomment the one-time forced `rtc.adjust(...)` in `setup()`, flash, confirm
the time is correct on the serial monitor, then re-comment and re-flash.
**Check:** serial monitor shows `CTRL,<recent-unixtime>,...` (enriched) and `ENV,...` (control).

## 5. Start ingestion
```bash
sudo systemctl enable --now serial_ingest env_ingest
journalctl -u serial_ingest -u env_ingest -f
```
**Check:** "upserted N rows" appears for both.

## 6. VALIDATE — the gate before trusting anything
```bash
scripts/.venv/bin/python scripts/verify_ingest.py
```
**Want:** both sources present, **0 duplicates**, `rtc_offset_sec` ~0 on both, melt-view metrics
for both chambers. If a chamber's data resembles the other's → udev swap; redo step 3.

## 7. Floor + agent schedule
Dry-run the agent first (no tokens): `scripts/.venv/bin/python scripts/agent_runner.py --mode check --dry-run`.
Then add the cron lines from `deploy/README.md` (floor `*/5`, check `8,20`, report weekly).
**Check:** `results/fault_floor.log` gets a line each run.

## 8. Dashboard
Add `EnvironmentFile=/home/pi/Growth_Chamber_cv/.env.monitoring` to the streamlit systemd unit
(drop-in), `systemctl restart`. Open the **Live Monitoring** page.
**Check:** the Logger Health strip shows both loggers recent + RTC offset OK.

## 9. (optional) Q&A smoke test
```bash
scripts/.venv/bin/python scripts/agent_qa.py --question "Is the enriched chamber holding setpoint today?"
```

## After data is flowing
Add the hourly `sd_reconcile.py` cron(s) (README) once the SD path is known, and run a
deliberate overlap to confirm dedup holds (validation point 2).
