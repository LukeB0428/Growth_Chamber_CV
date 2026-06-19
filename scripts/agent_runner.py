"""
agent_runner.py — the scheduled monitoring agent. Token-frugal by design.

Modes (cron-driven):
  check   twice daily — GATED: if decide() says 'nominal', writes a templated
          report and makes NO LLM call (zero tokens). Only anomalies cost tokens.
  report  weekly — always writes a short narrative (one LLM call), calling the
          deterministic stats elsewhere rather than doing math itself.

The agent reasons over the compact aggregate from aggregate.py — never raw rows —
which is the main cost control. A cheap model triages; a stronger model narrates.

Env: SUPABASE_URL, SUPABASE_KEY, ANTHROPIC_API_KEY
Run: python scripts/agent_runner.py --mode check
     python scripts/agent_runner.py --mode report
     python scripts/agent_runner.py --mode check --dry-run   # print prompt, no LLM, no write
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fault_floor import load_manifest, build_params
from aggregate import build_aggregate, decide

# Cheap model triages the twice-daily check; stronger model writes the weekly narrative.
DEFAULT_MODEL = {"check": "claude-haiku-4-5-20251001", "report": "claude-sonnet-4-6"}
MAX_TOKENS = {"check": 300, "report": 600}
DEFAULT_WINDOW_HOURS = {"check": 12, "report": 168}

SYSTEM = (
    "You are a monitoring assistant for a CO2 plant-growth-chamber experiment. "
    "Be terse and factual. Never assert a statistical effect between chambers without a "
    "computed test. Respect the calibration caveats: inter-chamber CO2 and temperature "
    "are NOT cross-calibrated (control CO2 has a -269ppm offset); humidity is the reliable "
    "cross-chamber signal. Distinguish real faults from benign events."
)


def nominal_summary(agg):
    e = agg["enriched"]
    pib = f"{e['pct_in_band']*100:.0f}%" if e["pct_in_band"] is not None else "n/a"
    return (f"All nominal over the last {agg['window_hours']}h: CO2 {e['co2_mean']} ppm "
            f"({pib} in band), duty {e['duty_mean']}, both loggers reporting, "
            f"RTC offsets OK. No open alerts.")


def build_prompt(agg, manifest, mode, flags):
    caveats = [c["rule"] for c in manifest.get("caveats", [])]
    benign = [f"{b['id']}: {b['description']}" for b in manifest.get("benign_patterns", [])]
    task = (
        "Triage the flags/alerts below: are they real faults or benign (see benign_patterns)? "
        "Give a 2-3 sentence status a researcher can act on."
        if mode == "check" else
        "Write a short weekly status (one paragraph): controller performance vs setpoint, "
        "environment, and any anomalies. Hedge any enriched-vs-control comparison unless a "
        "test is cited."
    )
    return (
        f"TASK: {task}\n\n"
        f"FLAGS: {flags}\n\n"
        f"AGGREGATE:\n{json.dumps(agg, indent=2)}\n\n"
        f"CAVEATS:\n- " + "\n- ".join(caveats) + "\n\n"
        f"BENIGN PATTERNS:\n- " + "\n- ".join(benign)
    )


def call_llm(prompt, model, max_tokens):
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic SDK not installed: pip install anthropic")
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def write_report(client, experiment_id, kind, status, summary, detail, used_llm, tin=None, tout=None):
    client.table("agent_reports").insert({
        "experiment_id": experiment_id, "kind": kind, "status": status,
        "summary": summary, "detail": detail, "used_llm": used_llm,
        "tokens_in": tin, "tokens_out": tout,
    }).execute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["check", "report"])
    ap.add_argument("--window-hours", type=int)
    ap.add_argument("--model")
    ap.add_argument("--experiment-id", default="ee496_arabidopsis_round2")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the aggregate + decide the gate, print the prompt, but do not call the LLM or write")
    args = ap.parse_args()

    mode = args.mode
    window_hours = args.window_hours or DEFAULT_WINDOW_HOURS[mode]
    model = args.model or DEFAULT_MODEL[mode]

    from observations_db import get_client
    client = get_client()
    m = load_manifest()
    params = build_params(m)

    agg = build_aggregate(client, params, window_hours, args.experiment_id)
    status, flags = decide(agg)

    # GATE: a clean twice-daily check costs zero tokens.
    if mode == "check" and status == "nominal":
        summary = nominal_summary(agg)
        if args.dry_run:
            print("[gate] nominal -> no LLM\n" + summary)
            return
        write_report(client, args.experiment_id, mode, status, summary, agg, used_llm=False)
        print(f"[{mode}] nominal (no LLM): {summary}")
        return

    prompt = build_prompt(agg, m, mode, flags)
    if args.dry_run:
        print(f"[gate] status={status} flags={flags} -> would call {model}\n\n{prompt}")
        return

    text, tin, tout = call_llm(prompt, model, MAX_TOKENS[mode])
    write_report(client, args.experiment_id, mode, status, text, agg, used_llm=True, tin=tin, tout=tout)
    print(f"[{mode}] status={status} ({tin}->{tout} tok): {text}")


if __name__ == "__main__":
    main()
