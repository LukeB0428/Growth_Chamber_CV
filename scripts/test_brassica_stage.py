"""
test_brassica_stage.py — phenology test for the Brassica BBCH stage tracker.

Feeds a synthetic full-lifecycle daily series through BrassicaStageDetector and
checks it walks the ladder in order without reverting (monotonic BBCH), and that
the real flowering signal drives flowering -> pod_fill.

    python test_brassica_stage.py
"""
import json
import os

from config import SPECIES_CONFIG_DIR
from developmental_stage import BrassicaStageDetector

# (canopy_cover_%, bolting_flag, flower_area_pct) per day — a stylised lifecycle.
SERIES = [
    (0.0, 0, 0.0), (0.2, 0, 0.0),          # dormant
    (0.6, 0, 0.0),                          # germination
    (2.0, 0, 0.0), (8.0, 0, 0.0),           # seedling
    (13.0, 0, 0.0), (20.0, 0, 0.0),         # vegetative
    (25.0, 1, 0.0), (28.0, 1, 0.0),         # bolting
    (30.0, 1, 5.0), (31.0, 1, 8.0), (30.0, 1, 6.0),   # flowering
    (30.0, 1, 0.3), (30.0, 1, 0.2), (30.0, 1, 0.0),   # flowers fading -> pod_fill
    (29.0, 1, 0.0), (29.0, 1, 0.0),         # pod_fill (cover steady)
    (15.0, 1, 0.0), (14.0, 1, 0.0), (13.0, 1, 0.0),   # canopy declining
    (12.0, 1, 0.0), (11.0, 1, 0.0),         # -> senescence (5 sustained low days)
]

LADDER = ["dormant", "germination", "seedling", "vegetative", "bolting",
          "flowering", "pod_fill", "senescence"]


def run():
    cfg = json.load(open(os.path.join(str(SPECIES_CONFIG_DIR), "brassica.json")))
    det = BrassicaStageDetector(cfg)
    history, stages, bbchs = [], [], []
    for cover, bolt, flower in SERIES:
        cur = {"canopy_cover_%": cover, "bolting_flag": bolt, "flower_area_pct": flower}
        s = det.detect(cur, list(history))
        stages.append(s.stage_name); bbchs.append(s.bbch_code)
        history.append(cur)

    failed = 0
    def check(cond, msg):
        nonlocal failed
        print(("PASS  " if cond else "FAIL  ") + msg)
        failed += 0 if cond else 1

    # All 8 stages appear, each first in ladder order.
    first = {}
    for i, s in enumerate(stages):
        first.setdefault(s, i)
    order = [first[s] for s in LADDER if s in first]
    check(set(first) == set(LADDER), f"all 8 stages reached: {sorted(set(stages))}")
    check(order == sorted(order), "stages first-appear in BBCH order")
    # Monotonic BBCH (phenology never reverts).
    check(all(b2 >= b1 for b1, b2 in zip(bbchs, bbchs[1:])), "BBCH non-decreasing (no reversion)")
    # Real flower signal drove flowering; its decline drove pod_fill.
    check("flowering" in stages and "pod_fill" in stages
          and first["pod_fill"] > first["flowering"], "flowering -> pod_fill via flower signal")
    check(stages[-1] == "senescence", "ends in senescence after canopy drop")

    print("\ntimeline:", " -> ".join(dict.fromkeys(stages)))
    print(f"{len(SERIES)-failed if failed==0 else ''}".strip(),
          "ALL PASS" if failed == 0 else f"{failed} FAILED")
    return failed


if __name__ == "__main__":
    import sys
    sys.exit(1 if run() else 0)
