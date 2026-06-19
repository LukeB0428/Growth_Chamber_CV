"""
telemetry_stats.py — deterministic comparative stats on chamber telemetry. NO LLM.

Keeps the LLM out of the math (the manifest's stats_policy): the Q&A agent calls
compare(); it never computes a p-value itself. Mann-Whitney U + Cohen's d, matching
the tests already used elsewhere in the project.

CAVEAT: 5s telemetry ticks are autocorrelated, so treating them as independent
samples makes p-values optimistic. This is a monitoring convenience, not a clean
experimental test — the agent's prompt hedges chamber comparisons accordingly, and
inter-chamber CO2/temperature carry their own calibration caveat.
"""
import numpy as np

try:
    from scipy.stats import mannwhitneyu
except ImportError:
    mannwhitneyu = None


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((na - 1) * sa ** 2 + (nb - 1) * sb ** 2) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def compare(enriched, control, alpha=0.05):
    """Two-sided Mann-Whitney U + Cohen's d. Returns a JSON-serializable verdict."""
    e = [x for x in enriched if x is not None]
    c = [x for x in control if x is not None]
    if len(e) < 3 or len(c) < 3:
        return {"ok": False, "reason": "insufficient data", "n_enriched": len(e), "n_control": len(c)}
    if mannwhitneyu is None:
        return {"ok": False, "reason": "scipy not installed"}
    u, p = mannwhitneyu(e, c, alternative="two-sided")
    d = cohens_d(e, c)
    return {
        "ok": True, "test": "mann_whitney_u", "n_enriched": len(e), "n_control": len(c),
        "mean_enriched": round(float(np.mean(e)), 3), "mean_control": round(float(np.mean(c)), 3),
        "u": float(u), "p_value": float(p), "significant": bool(p < alpha),
        "cohens_d": round(d, 3) if d is not None else None, "alpha": alpha,
        "note": "ticks are autocorrelated; p-value is optimistic — monitoring aid, not a clean test",
    }
