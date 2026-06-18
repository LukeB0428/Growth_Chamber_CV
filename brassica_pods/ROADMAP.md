# Brassica Pod Counting — Roadmap

From the current prototype (a YOLO-seg model trained on excised pods on black)
to the goal: **absolute silique counts per plant, within ~10–15%**, for
*Brassica napus* (Diarmuid, trial Q1 2027).

## Decisions locked (2026-06-17)
- **Species:** Brassica napus.
- **Deliverable:** absolute pod count (not just treatment comparison).
- **Accuracy target:** within ~10–15%.
- **Method:** **BOTH** — destructive endpoint count for the accurate absolute
  number, AND a non-destructive in-situ estimate tracked over time, validated
  against the endpoint truth.
- Size (length/width) and greenness also wanted; both come cheaply off the masks.

## The core problem
A mature B. napus plant is a dense 3D bush of siliques. From any single 2D view
a large, **variable** fraction of pods are occluded — you cannot count what you
cannot see, and the hidden fraction differs plant-to-plant, so a single in-situ
image cannot give absolute counts. The literature splits into two accurate
camps: **spread the pods flat and image them** (~95%+, e.g. Rapepod/deepcanola)
or **multi-view 3D reconstruction** (~97%, e.g. PST, KAN-GLNet). "Absolute within
10–15%" therefore rules out single-view in-situ.

## Why "both" is the right call
The destructive endpoint count is not only a deliverable — it is **the ground
truth the in-situ method needs anyway**. We build them together and use the
truth twice: once as the reported absolute number, once to calibrate the in-situ
estimate. Destructive validation was always required; "both" just leverages it.

---

## Track 1 — Endpoint absolute count (committed deliverable, ~10–15% high-confidence)
The reliable backbone. This is essentially deepcanola's exact setup, which is why
the model we are training **now applies directly**.

1. **Spread-and-count protocol** at final harvest: cut plant → spread siliques on
   black cloth → image top-down with the px→mm fiducial + ColorChecker in frame.
2. **Fine-tune** the synthetic-trained YOLO-seg model on a small set of *our*
   spread-pod images (our camera/lighting) — tightens transfer.
3. **Validate** against fully manual counts on N plants → report count MAE / %err
   and per-pod length vs caliper. Spread pods are the easy case; 10–15% is very
   achievable (likely better).

Track 1 alone satisfies the absolute-count requirement.

## Track 2 — In-situ longitudinal estimate (higher-upside R&D, de-risked by Track 1)
Non-destructive count on the standing plant, tracked over the podding window.

4. **Multi-view rig:** turntable, 8–12 angles, RGB + OAK-D **stereo depth**, at
   the podding timepoint. (Depth is an asset deepcanola did not have.)
5. **Per-view detection** with the same YOLO-seg model, fine-tuned on real
   standing-plant images via active learning (label hardest failures, retrain).
6. **3D dedup:** back-project each per-view detection using depth + known
   rotation angle to a 3D position; cluster across views so each physical pod is
   counted once (avoids full dense reconstruction).
7. **Calibrate to truth:** regress the multi-view count against the Track-1
   spread-and-count on the *same* plants → learn the correction for residual
   occlusion. This calibration is what makes 10–15% in-situ plausible — we do not
   trust the raw visible count, we correct it against known truth.

## Track 2 (in-situ multi-view) — Core Risk & Framing

**Interior-occlusion ceiling.** In a bushy B. napus canopy, interior and lower
pods can be hidden from every exterior viewpoint. No number of orbit angles
guarantees they are imaged, so any image-based in-situ count carries a residual
structural undercount. More cameras reduce but cannot eliminate this. Compounded
by cross-view deduplication error (matching the same pod across views), which
worsens with the OAK-D's coarse depth on thin siliques, Track 2 in-situ must be
assumed to carry an inherent negative bias.

**Reframe in-situ as a CALIBRATED ESTIMATE, not an absolute count.** Do not treat
Track 2 as "absolute count from images." Treat it as a non-destructive estimate
calibrated against Track 1 spread-and-count as ground truth — the same Branch-B
pattern as the chlorophyll workstream. Build a regression: multi-view visible-pod
count -> true total, fit and validated against spread-and-count on the SAME
plants, reported with R2/RMSE. This makes the in-situ number defensible and
publishable rather than silently low.

Track 1 spread-and-count remains the absolute-count backbone and now ALSO serves
as the calibration ground truth for Track 2.

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
   stereo-depth capture rig (Track 2, step 4).

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
Destructive manual counts on a calibration subset are mandatory to defend any
absolute-count claim and to fit the Track-2 calibration. Plan how many plants can
be sacrificed for ground truth.

---

## Status & immediate next steps
- [x] Phase 0 scaffold; synthetic pipeline validated on real B. napus
- [x] Self-contained Colab trainer ([colab/train_pods_colab.ipynb](colab/train_pods_colab.ipynb))
- [x] **Capture schema SPEC locked** ([CAPTURE_SCHEMA.md](CAPTURE_SCHEMA.md)) — ColorChecker +
  fiducial + QR sample-ID required in every frame
- [ ] **Capture enforcement LIVE** ([capture/capture_profile.py](capture/capture_profile.py)) — gate
  detects + refuses non-compliant frames, tested at the rig.
  **⚠️ The irreversible-capture risk is retired only when enforcement is live AND tested — NOT when the spec is locked.** Spec on paper does not stop a non-compliant capture; the gate does.
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
- [ ] **3. Track 1 spread-and-count capture protocol** (the absolute-count backbone + Track-2 calibration truth)
- [ ] **4. In-situ feasibility probe** — smartphone→COLMAP upper-bound *before* any depth-rig build
- [ ] **5. Track 2 multi-view rig** (only if probe passes): turntable + depth→3D dedup + calibration regression
- [ ] Realistic-background synthetic + real labelling + active learning
- [ ] Destructive ground-truth campaign + calibration
- [ ] TasselNet count-regression head — **only if dense-subset gate fires (>15%, or 10–15% & count-only)**
- [ ] (Deferred to Task 2) multi-class organ-seg {pod, stem, leaf}

## Open questions for Diarmuid
1. Capture pose / can plants go on a turntable at podding? (Track 2 rig)
2. How many plants can be destructively counted for ground truth? (sets calibration N)
3. Is the in-situ longitudinal estimate needed *during* podding, or only the
   endpoint number? (sets Track 2 priority)
4. One plant per pot imaged whole, vs the current dense multi-pot layout?
5. Cultivar within B. napus? (transfer + label quality)
6. Task 2 sample-tracking: confirm the QR/vial/spectrophotometer ID convention so
   the capture schema matches the wet-lab workflow.

Data: deepcanola, CC-BY-4.0, Zenodo 10.5281/zenodo.13903900 — cite in any output.
