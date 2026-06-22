# Monitoring-agent data layer — Pi deployment

Stage 1: get both chambers' Arduino telemetry into the Supabase normalized layer.
Nothing here talks to the LLM — this is the pipeline the agent (and the dashboard)
read from.

## Device → table routing
| Device (chamber)              | Serial line | Table                | Source label            |
|-------------------------------|-------------|----------------------|-------------------------|
| CO2 controller (enriched)     | `CTRL,...`  | `control_telemetry`  | `co2_controller_enriched` |
| Environmental logger (control)| `ENV,...`   | `observations`       | `env_logger_control`    |

The enriched unit's environmental fields surface through the `observations_unified`
melt-view, so cross-chamber humidity/CO2/temp comparisons read from that single view.

## Components
- `supabase/schema.sql` — run once against the Supabase project (SQL editor).
- `scripts/serial_ingest.py --role {controller|env_logger}` — live feed (one service per device).
- `scripts/sd_reconcile.py` — hourly SD `datalog.txt` backfill (cron).
- `scripts/verify_ingest.py` — the four-point live-path validation.
- `scripts/observations_db.py` — shared client + parsers.
- `scripts/fault_floor.py` — deterministic fault floor (no LLM) → `alerts` table.
- `scripts/aggregate.py` — pre-aggregation + the gate (`summarize`/`decide`).
- `scripts/agent_runner.py` — scheduled agent (`--mode check|report`), gated.

## Wiring two near-identical Arduinos (do this FIRST)
Two Mega boards on the Pi can swap `ttyACM0`/`ttyACM1` on reboot. If they swap, you
log enriched data as control with no error. Pin them by USB serial number:
```bash
sudo cp deploy/99-growthchamber-arduino.rules /etc/udev/rules.d/   # fill in serials first
sudo udevadm control --reload-rules && sudo udevadm trigger
ls -l /dev/ttyACM-enriched /dev/ttyACM-control                     # confirm both resolve
```
**Power:** the Arduinos draw little, but the two OAK-D Lites are marginal on host USB
power. If OAK-D disconnects appear after adding the second logger, move everything to
a **powered USB hub** — fixes the port budget and the brownout at once.

## Env (`/home/pi/Growth_Chamber_cv/.env.monitoring`, chmod 600)
```
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_KEY=<service-role-or-anon-key>
EXPERIMENT_ID=arabidopsis_co2
ANTHROPIC_API_KEY=sk-ant-...        # only needed by agent_runner.py (not ingestion)
```

## Install
```bash
scripts/.venv/bin/pip install -r scripts/requirements_monitoring.txt
sudo cp deploy/serial_ingest.service deploy/env_ingest.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now serial_ingest env_ingest
journalctl -u serial_ingest -u env_ingest -f

# Hourly SD reconciliation (crontab -e) — one line per board:
17 * * * * /home/pi/Growth_Chamber_cv/scripts/.venv/bin/python \
  /home/pi/Growth_Chamber_cv/scripts/sd_reconcile.py \
  --logfile /mnt/sd-enriched/datalog.txt --source co2_controller_enriched >> /tmp/reconcile.log 2>&1
```

## Four-point validation (run after rows land — this is the gate before stage 3)
```bash
scripts/.venv/bin/python scripts/verify_ingest.py
```
1. **Uniqueness** — `observations` key includes `metric_name` (schema), so co-timestamped
   metrics don't collide. `verify_ingest` reports duplicate control keys (want 0).
2. **Dedup under load** — let the live feed run, then `sd_reconcile` a deliberately
   overlapping window; confirm 0 duplicates and RTC is the surviving timestamp.
3. **RTC recovery + offset** — confirm the firmware RTC guard took on each board and
   `rtc_offset_sec` is logged and ~0 (drift/reset shows as a large offset).
4. **Melt-view** — `observations_unified` returns the joined control tuple and both
   chambers' metrics.

## Agent layer (stage 3) — two tiers, token-frugal
The deterministic floor runs continuously (no tokens); the LLM agent runs on a sparse
schedule and skips the LLM entirely on a clean check.
```bash
# crontab -e   (PYBIN = scripts/.venv/bin/python, ROOT = /home/pi/Growth_Chamber_cv)
*/5  * * * *  cd ROOT && PYBIN scripts/fault_floor.py            >> /tmp/floor.log 2>&1
0  8,20 * * *  cd ROOT && PYBIN scripts/agent_runner.py --mode check   >> /tmp/agent.log 2>&1
30   6 * * 0  cd ROOT && PYBIN scripts/agent_runner.py --mode report  >> /tmp/agent.log 2>&1
```
- **Floor** (`*/5`): evaluates controller thresholds, opens/resolves `alerts`. Never-miss,
  independent of the agent/API/network; also logs to `results/fault_floor.log`.
- **check** (08:00 / 20:00): a clean run writes a templated "nominal" report and makes
  **no LLM call**. Only flags/alerts cost tokens (Haiku, ~450-token prompt).
- **report** (weekly): one Sonnet narrative call.
- Dry-run any time without spending tokens: `agent_runner.py --mode check --dry-run`.
- `agent_reports.tokens_in/out` track spend; the dashboard's Agent Reports panel reads it.

## First-flash gotcha
The RTC guard only self-corrects a board that lost power. For a board already holding
a wrong date, do the one-time forced `rtc.adjust(...)` (commented line in `setup()`),
confirm the time on the serial line, then re-comment and re-flash. Applies to BOTH
`.ino` files.

## Current limits
- Cadence: serial = 5s, SD backfill = hourly. The `cadence` column disambiguates —
  don't treat the two as a uniform sample rate.
- CO2/temperature are NOT cross-calibrated between chambers (control CO2 carries a
  -269 ppm firmware offset; BME680s were never co-calibrated). The manifest encodes
  this; comparative claims must respect it.
