# Monitoring-agent data layer — Pi deployment

Stage 1 of the chamber monitoring agent: get Arduino controller telemetry into the
Supabase normalized layer. Nothing here talks to the LLM — this is the pipeline the
agent reads from.

## Components
- `supabase/schema.sql` — run once against the Supabase project (SQL editor).
- `scripts/serial_ingest.py` — live 5s `CTRL` feed → `control_telemetry` (systemd service).
- `scripts/sd_reconcile.py` — hourly SD `datalog.txt` backfill (cron).
- `scripts/observations_db.py` — shared client + parsers.

## Env (`/home/pi/Growth_Chamber_cv/.env.monitoring`, chmod 600)
```
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_KEY=<service-role-or-anon-key>
ARDUINO_PORT=/dev/ttyACM0
EXPERIMENT_ID=ee496_arabidopsis_round2
```

## Install
```bash
scripts/.venv/bin/pip install -r scripts/requirements_monitoring.txt
sudo cp deploy/serial_ingest.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now serial_ingest
journalctl -u serial_ingest -f          # watch it ingest

# Hourly SD reconciliation (crontab -e):
17 * * * * /home/pi/Growth_Chamber_cv/scripts/.venv/bin/python \
  /home/pi/Growth_Chamber_cv/scripts/sd_reconcile.py \
  --logfile /mnt/sd/datalog.txt --source co2_controller_enriched >> /tmp/reconcile.log 2>&1
```

## Notes / current limits
- **Only the CO2 controller streams serial.** The control-chamber Environmental
  Logger firmware was NOT patched, so it has no live feed — reconcile it from SD with
  `--source env_logger_control --role logging_only` until it gets the same patch.
- **Firmware first-flash:** the RTC guard only self-corrects a board that lost power.
  For a board already holding a wrong date, do the one-time forced `rtc.adjust(...)`
  (see the commented line in `setup()`), confirm the time on the CTRL line, then re-flash.
- **Cadence:** serial = 5s, SD backfill = hourly. The `cadence` column disambiguates;
  don't treat the two as a uniform sample rate.
