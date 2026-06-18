# Brassica Pod Counting — Roadmap

From the current prototype (a YOLO-seg model trained on excised pods on black)
to the goal: **absolute silique counts per plant, within ~10–15%**, for
*Brassica napus* (Diarmuid, trial Q1 2027).

## Decisions locked (2026-06-17, refined 2026-06-18)
- **Species:** Brassica napus.
- **Deliverable:** absolute pod count per plant, within ~10–15%.
- **Why (clarified 2026-06-18):** Diarmuid wants the **final count per plant**
  with **lab throughput / no manual labour** — today his team strips pods off by
  hand and counts them. "Non-destructive" here means **don't make anyone remove
  the pods**, NOT "track the plant over time." He wants an imaging station that
  counts pods **while still attached to the living plant**.
- Size (length/width) and greenness also wanted; both come cheaply off the masks.

## Product (clarified 2026-06-18)
**An imaging station the lab feeds POTTED LIVE PLANTS through, one at a time.**
Each potted plant is carried from its chamber to the station — a controlled
booth with a plain backdrop, even lighting, and a **turntable** — rotated for
multi-view RGB + OAK-D depth, counted, and returned to the chamber. Fully
**non-destructive** (no cutting, no stripping): it images the intact living
plant in its pot.

Key insight: moving the *potted* plant to the station gives the **controlled
imaging environment** (clean background, even light, free rotation) WITHOUT
cutting — the plant is never in the cluttered chamber scene and never harvested
for the count. This is strictly easier for the software than true in-chamber
in-situ, and strictly less destructive than cut-and-spread.

> Terminology: the capture mode `in_situ_multiview` now means **potted plant on
> the station turntable** (controlled environment), NOT imaging in the chamber.
> The gate behaviour (multi-view + depth required) is unchanged; only the
> framing is. (A future rename to `station_multiview` is optional — flagged.)

## The core problem
A mature B. napus plant is a dense 3D bush of siliques. From any single 2D view
a large, **variable** fraction of pods are occluded — you cannot count what you
cannot see, and the hidden fraction differs plant-to-plant, so a single in-situ
image cannot give absolute counts. The literature splits into two accurate
camps: **spread the pods flat and image them** (~95%+, e.g. Rapepod/deepcanola)
or **multi-view 3D reconstruction** (~97%, e.g. PST, KAN-GLNet). "Absolute within
10–15%" therefore rules out single-view in-situ.

## Why we still build both
The **station (potted plant)** is the product. **Spread-and-count** is the
**answer key**: the only way the station's count becomes trustworthy is by
checking it against the true count of the *same* plants — and the cleanest true
count is to (on a SUBSET only) strip + spread + count those plants. So spread
is no longer the deliverable; it is the calibration + validation ground truth,
plus a fallback if the station can't reach target. We build it once, on a few
plants, and reuse the truth to calibrate the station.

---

## The Station — potted live plant, multi-view (THE PRODUCT)
Non-destructive count of the intact living plant, on the controlled station.

1. **Station rig:** booth with plain backdrop + even lighting; potted plant on a
   **turntable**, **8–12 angles**, RGB + OAK-D **stereo depth** per angle. One
   plant at a time, carried from the chamber and returned.
2. **Per-view detection** with the YOLO-seg model, fine-tuned on real
   potted-plant station images via active learning (label hardest failures).
3. **3D dedup:** back-project each per-view detection using depth + turntable
   angle to a 3D position; cluster across views so each physical pod is counted
   once (avoids full dense reconstruction).
4. **Calibrate to truth:** regress the station's multi-view count against the
   spread-and-count truth on the *same* plants → correct the residual occlusion
   undercount. We do not trust the raw visible count; we correct it against the
   answer key. (Report R²/RMSE — see the risk section.)

## Spread-and-count — calibration answer-key + fallback (NOT the product)
Run on a SUBSET of plants to generate ground truth and validate the station.
This is deepcanola's exact setup, which is why the model training **now applies
directly** here.

A. **Spread protocol:** strip the subset plants' siliques, spread on black cloth,
   image top-down with the px→mm fiducial + ColorChecker in frame.
B. **Fine-tune** the synthetic-trained model on a small set of *our* spread-pod
   images (our camera/lighting).
C. **Manual count** the same subset by hand → the gold ground truth that both
   validates the spread reading AND calibrates the station (step 4 above).
Spread pods are the easy case; ~10–15% is very achievable here, so it is a
trustworthy yardstick — and a fallback if the station underperforms.

## The Station (multi-view) — Core Risk & Framing

**Interior-occlusion ceiling.** In a bushy B. napus canopy, interior and lower
pods can be hidden from every exterior viewpoint. No number of turntable angles
guarantees they are imaged, so any image-based count of the intact plant carries
a residual structural undercount. More angles reduce but cannot eliminate this.
**A potted LIVE plant is slightly worse than a cut plant here:** you can rotate
the pot but you cannot fan the branches apart or lift the lowest pods clear of
the pot rim without damaging a living plant — so a few interior/low pods stay
hidden. Compounded by cross-view deduplication error (matching the same pod
across views), which worsens with the OAK-D's coarse depth on thin siliques, the
station must be assumed to carry an inherent negative bias.

**Reframe the station count as a CALIBRATED ESTIMATE, not a raw absolute count.**
Do not treat the station as "absolute count from images." Treat it as a
non-destructive estimate calibrated against spread-and-count ground truth — the
same Branch-B pattern as the chlorophyll workstream. Build a regression:
multi-view visible-pod count -> true total, fit and validated against
spread-and-count on the SAME plants, reported with R2/RMSE. This makes the
station number defensible and publishable rather than silently low.

Spread-and-count remains the **most accurate absolute method** — but it is now
the **calibration answer-key + fallback**, not the product. The product is the
station; spread exists to make the station's number trustworthy (and to fall
back on if the station can't reach ~10–15%).

---

## v2 additions (2026-06-18)
Deltas from Build Plan v2. They sharpen the **2D counting engine** and protect the
**Task-2 (chlorophyll) future**; they do **not** change the absolute-count strategy
above — the in-situ absolute number still hinges on Track 2's calibrated multi-view.

### Counting engine — instance seg vs count regression
Two complementary outputs; pick the count engine on measured dense-case error.

| Approach | Output | Strength | Handles | Use when |
|---|---|---|---|---|
| YOLO-seg / deepcanola (instance) | count **+ per-pod masks + size** | size metrics, per-pod data | 2D overlap (moderately) | objects separable; need length/width/area |
| TasselNetV2+ (count regression) | **count only** | robust to **2D image-plane overlap** | dense touching/crossing pods *in view* | dense clusters where you only need "how many" |

**Scope correction (important):** TasselNetV2+ handles **2D image-plane OVERLAP**,
**not 3D OCCLUSION**. It regresses from *visible* density, so it cannot recover
pods hidden behind the canopy — it undercounts hidden pods exactly like instance
seg. Therefore scope it as the counting engine for **Track 1 (spread-and-count,
all pods visible → pure 2D overlap)** and for **Track 2's per-view step** only. It
is **not** a general occlusion fix and does **not** substitute for multi-view 3D
dedup. (Bonus: density/dot labels come free from our synthetic pod centroids, so
the head-to-head trains on data we already generate.)
Repo: https://github.com/poppinace/tasselnetv2plus

**TasselNet gate — set now, decided before measuring.** After eval, read the
instance model's count error on the **DENSE subset only**:
- **≤10%** dense-subset count error → **skip TasselNet** (instance seg sufficient).
- **>15%** → **build** the TasselNet count-regression head.
- **10–15%** → optional; build only if count is the sole deliverable on dense
  images and size is not needed.
Do **not** revise these bounds after seeing the number.

### Evaluation hardening
- Count **MAE and % error** vs manual count; for the instance route also **mask AP**
  and size validation (caliper subset; px→mm scale from the in-frame fiducial).
- **Report DENSE/OVERLAPPING vs SPARSE subsets separately, everywhere.** An average
  hides the priority failure case. Split by GT pod density.
- Log per-image predictions + errors so active learning targets the worst cases.
- **PhenoBench** (https://github.com/PRBonn/phenobench): use **only** as a
  metrics-reporting *convention* (PQ / IoU / panoptic quality). It is sugar-beet /
  weed field data with **no pods** — it is **not** an external validation set for
  our siliques. Be precise about this wherever it is cited (thesis/claims).

### In-situ feasibility probe — run in this order (gates the whole Track 2 rig)
1. **First, smartphone → COLMAP photogrammetry as the optimistic UPPER BOUND.** If
   COLMAP (higher effective resolution) cannot resolve individual pods on a bushy
   plant, the lower-res OAK-D stereo depth won't either — **stop, do not build the
   depth rig.**
2. **Only if COLMAP shows pods are resolvable**, proceed to the OAK-D multi-view +
   stereo-depth station rig (The Station, step 1) — potted plant on the turntable.

### Task-2 bridge — capture now, model later
- **Now (irreversible if missed):** every capture includes a **ColorChecker + scale
  fiducial + a QR/barcode sample-ID** in frame; chlorophyll normalised **per area
  (µg/cm²)** so it pairs with mask area. See [CAPTURE_SCHEMA.md](CAPTURE_SCHEMA.md).
- **NOT NOW:** the multi-class organ-seg model ({pod, stem, leaf}). Stems/leaves
  need their own cut-outs + masks (real data-sourcing — mine AgML / PRBonn leaf
  data). Keep the backbone *able* to go multi-class, but **don't build it** until
  Task 2 starts. Method ref:
  https://github.com/PRBonn/leaf-plant-instance-segmentation

### Utility — AgML
Source of pretrained ag models + labelled datasets to fine-tune from (and a likely
source of stem/leaf data for the future organ-seg work). Mine it, don't depend on
it. https://github.com/Project-AgML/AgML

---

## Full 3D (not now)
Dense reconstruction (NeRF/SfM → 3D instance segmentation, KAN-GLNet style) is the
accuracy ceiling but is research-grade and high-risk for the timeline. Only revisit
if calibrated multi-view cannot reach 10–15%.

---

## Domain-gap engineering (applies to both tracks)
The current model has only seen excised pods on black. To work on real material:
1. **Realistic synthetic backgrounds** — reuse the same pod cut-outs, composite
   onto real rig/greenhouse backgrounds via the existing
   [synth/generate_dataset.py](synth/generate_dataset.py). Big win, zero new
   labelling. (Track 1 needs this less — its background *is* black cloth.)
2. **Real labelled data + fine-tune** — SAM2-assisted labelling (reuse
   scripts/sam2_weights), even 30–100 images moves the needle.
3. **Active-learning loop** — run model, label worst failures, retrain, repeat.

## Integration with the existing pipeline
- **Phenological gating:** pods exist only post-flowering. `scripts/developmental_stage.py`
  + brassica.json already define `pod_fill` (BBCH 71) — the counter activates only
  in that window.
- **Per-pot** analysis like the existing rosette pipeline.
- **Outputs** flow through `analyze()` → pot_metrics.csv → dashboard, alongside
  current metrics. `analyze(..., greenness=True)` already wired.

## Validation (non-negotiable)
Manual counts on a calibration subset (strip + spread + hand-count) are mandatory
to defend any absolute-count claim and to fit the **station** calibration
regression. Plan how many plants can be sacrificed for ground truth — note these
ARE destroyed (the rest, imaged on the station, are not).

---

## Status & immediate next steps
- [x] Phase 0 scaffold; synthetic pipeline validated on real B. napus
- [x] Self-contained Colab trainer ([colab/train_pods_colab.ipynb](colab/train_pods_colab.ipynb))
- [x] **Capture schema SPEC locked** ([CAPTURE_SCHEMA.md](CAPTURE_SCHEMA.md)) — ColorChecker +
  fiducial + QR sample-ID required in every frame
- [~] **Capture enforcement** — gate ([capture/capture_profile.py](capture/capture_profile.py), 20/20)
  + loop ([capture/capture_session.py](capture/capture_session.py), 5/5 via MockFrameSource) BUILT + tested.
  Remaining for LIVE: wire `OakDFrameSource` to depthai v3 + operator-test at the rig.
  **⚠️ The irreversible-capture risk is retired only when enforcement is live AND rig-tested — NOT when the gate/loop pass in unit tests.** The camera I/O is the one unverified piece.
- [ ] **One-time ChArUco distortion calibration** — intrinsics + distortion coeffs per camera/zoom/setup; every production frame undistorted before measurement
- [ ] **Confirm + measure marker sizes** — gate refuses to start a session until
  `MARKER_SIZE_MM` holds caliper-measured printed sizes and `MARKER_SIZE_CONFIRMED=True`
- **RULE — manifest mode:** pilot captures may use `warn` (passes but surfaces unknown
  samples loudly); the **real data-collection run MUST use `strict`**. Every frame records
  `manifest_mode`, so warn-in-production is auditable and must not become the default.
- [ ] **1. Finish current Colab run** → `pods_best.pt`, confirm machinery end-to-end
- [~] **2. Eval hardening** — code built + unit-tested ([eval/evaluate.py](eval/evaluate.py),
  8/8): count MAE/%err/bias/RMSE/R², **dense/sparse split**, mask mAP (synthetic val),
  size check, per-image worst-first log, **TasselNet-gate read**. Runs on `pods_best.pt` arrival.
- [ ] **3. Spread-and-count protocol** (calibration answer-key + fallback) — on a SUBSET, + manual counts
- [ ] **4. Station feasibility probe** — smartphone→COLMAP upper-bound on a potted plant *before* any depth-rig build
- [ ] **5. Station rig** (only if probe passes): potted plant on turntable, multi-view RGB+depth → 3D dedup → calibration regression vs spread truth
- [ ] Realistic-background synthetic + real potted-plant labelling + active learning
- [ ] Spread-and-count ground-truth campaign on the calibration subset
- [ ] TasselNet count-regression head — **only if dense-subset gate fires (>15%, or 10–15% & count-only)**
- [ ] (Deferred to Task 2) multi-class organ-seg {pod, stem, leaf}

## Resolved (2026-06-18)
- "Non-destructive" = **no manual pod stripping/counting** (labour/throughput),
  NOT longitudinal. Only the **final count per plant** is needed.
- Workflow = **potted live plant carried to the station one at a time**, rotated,
  imaged, returned. Not cut, not imaged in the cluttered chamber.

## Open questions for Diarmuid
1. **How big / bushy are the plants at podding?** This is make-or-break for the
   station — a compact plant is imageable; a tall self-occluding bush may not hit
   ~10–15% from any number of turntable angles. (Drives the feasibility probe.)
2. **Throughput target** (plants/day)? Decides motorised vs manual turntable.
3. **How many plants for the spread-count calibration subset?** These ARE
   destroyed; sets calibration N.
4. Pot/plant size + turntable load — can the potted plant physically sit and
   rotate on the station? Lowest pods clear of the pot rim?
5. Cultivar within B. napus? (transfer + label quality)
6. Task 2 sample-tracking: confirm the QR/vial/spectrophotometer ID convention so
   the capture schema matches the wet-lab workflow.

Data: deepcanola, CC-BY-4.0, Zenodo 10.5281/zenodo.13903900 — cite in any output.
