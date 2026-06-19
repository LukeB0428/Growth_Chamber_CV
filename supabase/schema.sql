-- ============================================================================
-- Growth Chamber Monitoring Agent — normalized data layer (Supabase / Postgres)
-- EE496 | Luke Buckley | Maynooth University
--
-- Two physical tables + one view:
--   control_telemetry   WIDE  — the tightly-coupled controller tuple, one row per
--                               tick. Per-station "adapter"; controller monitoring
--                               reads this directly (error + duty at the same row,
--                               no self-join).
--   observations        LONG  — portable, station-varying metrics (CV/phenotype +
--                               anything new). This is where portability lives.
--   observations_unified VIEW — the agent's SINGLE read interface. Melts the wide
--                               control table into the long shape so the agent code
--                               never needs to know a station's internal schema.
--
-- Canonical clock = the Arduino RTC (rtc_timestamp). ingest_timestamp + rtc_offset_sec
-- exist only to DETECT clock problems (see rtc_reset_risk in the manifest), never to
-- replace the RTC as the time axis.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- WIDE: controller telemetry (one row per control tick)
-- ----------------------------------------------------------------------------
create table if not exists control_telemetry (
  id               bigint generated always as identity primary key,
  experiment_id    text not null,
  source           text not null,          -- 'co2_controller_enriched' | 'env_logger_control'
  rtc_timestamp    timestamptz not null,   -- canonical clock (RTC unixtime)
  ingest_timestamp timestamptz not null default now(),
  rtc_offset_sec   double precision,       -- host_now - rtc_timestamp; drift guard

  -- control loop
  setpoint_ppm     double precision,       -- 1100 (firmware `treatment`)
  measured_co2_ppm double precision,
  co2_1min_ppm     double precision,
  co2_5min_ppm     double precision,
  co2_24h_ppm      double precision,
  error_ppm        double precision,       -- setpoint - measured (computed at ingest)
  proj_co2_ppm     double precision,       -- 15s-ahead projection
  proj_dev         double precision,       -- (setpoint - proj_co2)/setpoint
  duty_cycle       double precision,       -- 0..500
  duty_5min        double precision,
  control_state    text,                   -- dosing | overnight_closed | safety_cutoff | logging_only

  -- environment (BME680 on the same unit)
  temp_c           double precision,
  pressure_mbar    double precision,
  humidity_pct     double precision,
  gas_kohm         double precision,

  cadence          text not null default 'serial_5s', -- serial_5s | sd_hourly

  -- upsert / dedup key: same RTC tick from the same source is one logical record,
  -- whether it arrived live (serial) or via SD reconciliation backfill.
  unique (source, rtc_timestamp)
);

create index if not exists idx_ctrl_time   on control_telemetry (rtc_timestamp);
create index if not exists idx_ctrl_source on control_telemetry (source, rtc_timestamp);

-- ----------------------------------------------------------------------------
-- LONG: portable observations (CV/phenotype + future metrics)
-- ----------------------------------------------------------------------------
create table if not exists observations (
  id            bigint generated always as identity primary key,
  experiment_id text not null,
  timestamp     timestamptz not null,
  metric_name   text not null,
  value         double precision,
  units         text,
  source        text not null,             -- 'cv_pipeline' | 'co2_controller_enriched' | 'env_logger_control'
  chamber       text not null default '',   -- enriched | control | '' — NOT NULL so PostgREST upsert
  pot_label     text not null default '',   -- P1..P8 | ''            can target a plain unique constraint

  -- Natural key includes metric_name: many metrics share a (timestamp, source).
  -- Plain columns (no coalesce expr) so on_conflict works via supabase-py.
  unique (experiment_id, timestamp, metric_name, source, chamber, pot_label)
);

create index if not exists idx_obs_time   on observations (timestamp);
create index if not exists idx_obs_metric on observations (metric_name, timestamp);
create index if not exists idx_obs_scope  on observations (chamber, pot_label, metric_name);

-- ----------------------------------------------------------------------------
-- VIEW: the agent's single uniform interface
--   observations  +  control_telemetry melted into (metric_name, value, units)
-- ----------------------------------------------------------------------------
create or replace view observations_unified as
  select experiment_id, timestamp, metric_name, value, units, source, chamber, pot_label
  from observations
  union all
  select
    c.experiment_id,
    c.rtc_timestamp                       as timestamp,
    m.metric_name,
    m.value,
    m.units,
    c.source,
    case when c.source like '%enriched%' then 'enriched' else 'control' end as chamber,
    ''::text                              as pot_label
  from control_telemetry c
  cross join lateral (values
      ('setpoint_ppm',     c.setpoint_ppm,     'ppm'),
      ('measured_co2_ppm', c.measured_co2_ppm, 'ppm'),
      ('co2_5min_ppm',     c.co2_5min_ppm,     'ppm'),
      ('error_ppm',        c.error_ppm,        'ppm'),
      ('duty_cycle',       c.duty_cycle,       'unitless_0_500'),
      ('temp_c',           c.temp_c,           'celsius'),
      ('humidity_pct',     c.humidity_pct,     'pct'),
      ('pressure_mbar',    c.pressure_mbar,    'mbar'),
      ('gas_kohm',         c.gas_kohm,         'kohm'),
      ('rtc_offset_sec',   c.rtc_offset_sec,   'seconds')
  ) as m(metric_name, value, units)
  where m.value is not null;

-- ----------------------------------------------------------------------------
-- Agent layer: alerts (deterministic floor) + agent_reports (scheduled LLM runs)
-- ----------------------------------------------------------------------------
create table if not exists alerts (
  id            bigint generated always as identity primary key,
  experiment_id text not null,
  source        text not null,
  rule_id       text not null,
  severity      text not null,
  status        text not null default 'open',   -- open | resolved
  opened_at     timestamptz not null default now(),
  resolved_at   timestamptz,
  detail        text,
  value         double precision
);
-- At most one OPEN alert per (experiment, source, rule); resolved history is unbounded.
create unique index if not exists uniq_open_alert
  on alerts (experiment_id, source, rule_id) where status = 'open';
create index if not exists idx_alerts_open on alerts (status, opened_at);

create table if not exists agent_reports (
  id            bigint generated always as identity primary key,
  experiment_id text not null,
  created_at    timestamptz not null default now(),
  kind          text not null,            -- check | report
  status        text,                     -- nominal | attention | alert
  summary       text,
  detail        jsonb,                    -- the aggregate + any structured findings
  used_llm      boolean not null default false,
  tokens_in     integer,
  tokens_out    integer
);
create index if not exists idx_reports_recent on agent_reports (created_at desc);
