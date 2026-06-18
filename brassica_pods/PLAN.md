# Brassica Pod Counter — Plan

Executive plan. Detail lives in [ROADMAP.md](ROADMAP.md) (technical),
[PROBE.md](PROBE.md) (feasibility test), [RIG.md](RIG.md) (hardware),
[CAPTURE_SCHEMA.md](CAPTURE_SCHEMA.md) (capture spec). This is the decision layer.

## 1. Goal
A **lab tool for Diarmuid** (not the enriched/control trial): an imaging station
that returns the **final silique (pod) count per plant** for *Brassica napus*,
within **~10–15%**, to remove the manual labour of stripping pods off by hand and
counting them. Ideal: count pods **on the intact living plant** (potted, carried
to a controlled station one at a time, rotated, imaged, returned). Size/greenness
are wanted too if they come off the same masks.

## 2. The honest feasibility split
The project is **one safe part bolted to one unproven part**:

- **SAFE — spread-and-count:** cut plant → spread pods on black → software counts.
  ~±5–10%, proven, model training now. **But** it is destructive and still needs
  someone to strip the pods by hand — so it only automates the *counting*, not the
  *handling* Diarmuid wants gone.
- **UNPROVEN — intact-plant station:** count pods on the standing potted plant.
  This is what actually solves his problem, but it is **physics-limited by interior
  occlusion** (pods hidden behind pods) and is **currently unvalidated**. Whether
  ±10–15% is even possible depends on one unmeasured number.

> Everything built so far is sound scaffolding around the unproven part. None of
> it has reduced the core risk. The next real step is not code — it is the probe.

## 3. The plan — feasibility-first, gated
```
Phase 0  Software foundation .......................... DONE (see §4)
Phase 1  FEASIBILITY PROBE (PROBE.md) ................. THE GATE — do before any hardware
            ├─ GO        -> Phase 2A (build the station)
            ├─ MARGINAL  -> Phase 2B (estimate-only, decide with Diarmuid)
            └─ NO-GO     -> Phase 2C (spread-and-count, semi-automated)
Phase 2A Station: rig + fine-tune detector + 3D dedup + calibrate to spread truth
Phase 2B Estimate product: same rig, reported with honest +/-20-30% error bars
Phase 2C Spread bench: automate counting only; human still spreads (partial win)
Phase 3  Validate vs destructive manual counts; report MAE/%err, R2/RMSE
Phase 4  (later) Task-2 chlorophyll — capture schema already protects for it
```

**The gate is non-negotiable:** do not build the station rig, and do not promise
Diarmuid non-destructive ±10–15%, until the probe returns **GO**. The probe is a
phone + tally counter + ~8 plants + an afternoon, for ~€0. It measures the visible
pod fraction `f` and its consistency `CV(f)`; since best-possible error ≈ `CV(f)`,
it tells you the hard answer before you spend a cent on hardware.

## 4. Status (what exists today)
Branch `brassica-pods-module`, all tested:
- **Synthetic-data pipeline** — deepcanola-style generator; 1000-image set built.
- **Trainer** — self-contained Colab YOLO11-seg notebook (running now).
- **Analyzer** — `analyze(image)` → count + per-pod size + greenness.
- **Eval harness** — count MAE/%err, dense/sparse split, TasselNet gate read (8/8).
- **Capture gate + loop** — ArUco/QR/ColorChecker/sharpness/exposure, per-mode
  depth-aware scale, sample-ID + manifest, ChArUco calibration; mock-tested (25/25).
- **Feasibility probe** — protocol + analysis + verdict (5/5).
- **Docs** — ROADMAP, RIG, PROBE, CAPTURE_SCHEMA.
Not built (gated): station detector (needs real images), the rig, live capture I/O.

## 5. Top risks (condensed)
1. **Interior occlusion** may cap intact-plant accuracy below ±15% — *the* gating
   risk; the probe measures it.
2. **OAK-D Lite depth** is coarse and weak on thin, repetitive siliques — the 3D
   de-dup may not work (COLMAP sub-test checks this).
3. **Station detector unbuilt** — needs real potted-plant images + labelling that
   can't start until plants + rig exist.
4. **Validation needs the very labour we're removing** — destructive hand counts on
   a calibration subset are unavoidable.
5. **Resourcing vs ambition** — solo build, weak sensor, vs a research-grade problem.

## 6. Immediate next steps
1. **When `pods_best.pt` lands:** run the eval on the deepcanola novel set (no
   labelling) → first real numbers + TasselNet decision.
2. **When podding plants exist:** run the **feasibility probe** → GO/MARGINAL/NO-GO.
3. Get the **Diarmuid answers** below — several gate the whole approach.

---

## 7. Questions for Diarmuid

### A. Feasibility-critical (these decide whether the product is even possible)
1. **How big and bushy is a plant at podding?** Rough height, and is it an open
   structure or a dense self-occluding bush? (This is the single biggest unknown —
   it drives whether intact-plant counting can hit ±10–15% at all.)
2. **What growth stage do you count at** — green immature pods, or dry/mature? (Dry
   pods are browner, more brittle, may shatter; changes appearance and handling.)
3. Are plants grown **under standardised conditions** (so their architecture is
   reasonably consistent plant-to-plant)? (Consistency is what makes calibration
   work.)

### B. The requirement
4. **Is ±10–15% the real target?** What absolute count error is actually *useful*
   for your work — would ±20% still be valuable, or do you need tighter?
5. Do you need a true **absolute count**, or would a **consistent relative index**
   (good for ranking/comparing plants) be enough?
6. Besides count, do you want **pod size** (length/width) and/or **greenness** per
   plant?

### C. Workflow & throughput
7. **What is the current manual process**, and how long does it take per plant?
   (Sets the baseline the station must beat to be worth it.)
8. **How many plants per run / per season?** (Sets throughput + labelling budget.)
9. Is **carrying each potted plant to a station** (and back) an acceptable workflow,
   or does it need to happen at the plant?
10. **Throughput target** — plants/day? (Decides motorised vs manual turntable.)

### D. Calibration & validation
11. How many plants can be **destructively hand-counted** for ground truth /
    calibration? (We need a meaningful sample spanning the size range.)
12. Is cutting + stripping a **subset** of plants acceptable for that purpose?

### E. Hardware & practical
13. **Pot size and weight** at podding? (Turntable load + platform size.)
14. Any **budget** for the rig (lights, turntable, ColorChecker)?
15. **Cultivar(s)** of B. napus? (Affects pod morphology + how well the model
    transfers.)
16. Any preference/constraint on **backdrop** or imaging location/lighting?

### F. Task 2 (chlorophyll) — only to protect the capture schema now
17. Will the chlorophyll work need **per-organ** (pod/stem/leaf) sampling, and what
    is the **vial / spectrophotometer sample-ID** convention so our QR scheme
    matches your wet-lab workflow from day one?

> The ones that change *what we build* most: **A1 (bushiness)**, **B4/B5
> (accuracy & absolute-vs-relative)**, and **A2 (maturity stage)**. If Diarmuid can
> answer only a few, prioritise those.
