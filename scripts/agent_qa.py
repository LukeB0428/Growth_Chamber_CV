"""
agent_qa.py — interactive plain-English Q&A over the chamber data layer (stage 5).

This is where the agent's tool-use loop earns its place (unlike the one-shot
scheduled runners): the model decides which deterministic tools to call to ground
its answer. The LLM orchestrates; the tools do the querying and the math.

Tools (all deterministic, no LLM): get_latest, summarize_metric, compare_chambers
(the ONLY path to a chamber comparison — enforces the stats_policy), list_open_alerts,
recent_reports.

Env: SUPABASE_URL, SUPABASE_KEY, ANTHROPIC_API_KEY
Run: python scripts/agent_qa.py --question "Is the enriched chamber holding setpoint today?"
     python scripts/agent_qa.py                      # interactive (stdin)
     python scripts/agent_qa.py --list-tools         # print tool schemas, no API
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import telemetry_stats

MODEL = "claude-sonnet-4-6"

SYSTEM = (
    "You answer questions about a CO2 plant-growth-chamber experiment (enriched vs "
    "control) using ONLY the provided tools — never invent numbers. To compare the two "
    "chambers you MUST call compare_chambers and report its p-value and effect size; "
    "never assert a difference without it. Respect the calibration caveats: inter-chamber "
    "CO2 and temperature are NOT cross-calibrated (control CO2 has a -269ppm offset); "
    "humidity is the reliable cross-chamber signal. Be concise and factual."
)

METRICS = ["measured_co2_ppm", "duty_cycle", "error_ppm", "temp_c", "humidity_pct", "rtc_offset_sec"]

TOOLS = [
    {"name": "get_latest", "description": "Most recent value of a metric for a chamber.",
     "input_schema": {"type": "object", "properties": {
         "metric": {"type": "string", "enum": METRICS},
         "chamber": {"type": "string", "enum": ["enriched", "control"]}},
         "required": ["metric", "chamber"]}},
    {"name": "summarize_metric", "description": "Mean/min/max/n of a metric for a chamber over the last N hours.",
     "input_schema": {"type": "object", "properties": {
         "metric": {"type": "string", "enum": METRICS},
         "chamber": {"type": "string", "enum": ["enriched", "control"]},
         "hours": {"type": "number"}}, "required": ["metric", "chamber", "hours"]}},
    {"name": "compare_chambers", "description": "Mann-Whitney U + Cohen's d comparing enriched vs control for a metric over the last N hours. The ONLY valid way to compare chambers.",
     "input_schema": {"type": "object", "properties": {
         "metric": {"type": "string", "enum": METRICS},
         "hours": {"type": "number"}}, "required": ["metric", "hours"]}},
    {"name": "list_open_alerts", "description": "Current open fault alerts from the deterministic floor.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "recent_reports", "description": "The N most recent scheduled agent reports.",
     "input_schema": {"type": "object", "properties": {"n": {"type": "integer"}}}},
]


def _series(client, metric, chamber, hours, cap=20000):
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    rows = (client.table("observations_unified").select("value")
            .eq("metric_name", metric).eq("chamber", chamber)
            .gte("timestamp", cutoff).limit(cap).execute().data)
    return [r["value"] for r in rows if r.get("value") is not None]


def dispatch(name, args, client, experiment_id):
    if name == "get_latest":
        rows = (client.table("observations_unified").select("timestamp,value")
                .eq("metric_name", args["metric"]).eq("chamber", args["chamber"])
                .order("timestamp", desc=True).limit(1).execute().data)
        return rows[0] if rows else {"error": "no data"}
    if name == "summarize_metric":
        vals = _series(client, args["metric"], args["chamber"], args["hours"])
        if not vals:
            return {"n": 0}
        import numpy as np
        return {"n": len(vals), "mean": round(float(np.mean(vals)), 3),
                "min": round(float(min(vals)), 3), "max": round(float(max(vals)), 3)}
    if name == "compare_chambers":
        e = _series(client, args["metric"], "enriched", args["hours"])
        c = _series(client, args["metric"], "control", args["hours"])
        return telemetry_stats.compare(e, c)
    if name == "list_open_alerts":
        return (client.table("alerts").select("rule_id,severity,opened_at,detail")
                .eq("experiment_id", experiment_id).eq("status", "open").execute().data)
    if name == "recent_reports":
        n = int(args.get("n", 3))
        return (client.table("agent_reports").select("created_at,kind,status,summary")
                .order("created_at", desc=True).limit(n).execute().data)
    return {"error": f"unknown tool {name}"}


def answer(question, client, experiment_id, model=MODEL, max_iters=6):
    import anthropic
    anthro = anthropic.Anthropic()
    messages = [{"role": "user", "content": question}]
    tin = tout = 0
    for _ in range(max_iters):
        resp = anthro.messages.create(model=model, max_tokens=700, system=SYSTEM,
                                      tools=TOOLS, messages=messages)
        tin += resp.usage.input_tokens
        tout += resp.usage.output_tokens
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            return text, (tin, tout)
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for b in resp.content:
            if getattr(b, "type", None) == "tool_use":
                out = dispatch(b.name, b.input, client, experiment_id)
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": json.dumps(out, default=str)})
        messages.append({"role": "user", "content": results})
    return "(stopped: too many tool iterations)", (tin, tout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question")
    ap.add_argument("--experiment-id", default=os.environ.get("EXPERIMENT_ID", "arabidopsis_co2"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--list-tools", action="store_true", help="print tool schemas and exit (no API)")
    args = ap.parse_args()

    if args.list_tools:
        print(json.dumps(TOOLS, indent=2))
        return

    from observations_db import get_client
    client = get_client()

    if args.question:
        text, (tin, tout) = answer(args.question, client, args.experiment_id, args.model)
        print(f"{text}\n\n[{tin}->{tout} tokens]")
        return

    print("Ask about the chambers (Ctrl-D / empty line to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q:
            break
        text, (tin, tout) = answer(q, client, args.experiment_id, args.model)
        print(f"\n{text}\n[{tin}->{tout} tokens]")


if __name__ == "__main__":
    main()
