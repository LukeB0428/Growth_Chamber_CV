# barley — *Hordeum vulgare* phenotyping

Barley is a tall, self-occluding monocot — the opposite of the flat Arabidopsis
rosette. Its CV does **not** reuse the rosette pipeline; it reuses the shared
**detect-and-count engine** (`phenotyping/`), the same one brassica pods use.

## Approach
- **Spike/ear count** (headline yield trait) — `phenotyping.object_count` detection
  path, a YOLO detector **warm-started on GWHD** (Global Wheat Head Detection) and
  fine-tuned on barley once Grace's trial data exists. Same engine + metric harness
  (`phenotyping/eval_count.py`) as brassica pods and the CVPPP leaf benchmark.
- **Greenness / senescence, canopy cover** — reuse `scripts/greenness_metrics` via
  `config/species/barley.json`.
- **Growth stage** — Zadoks/BBCH ladder in `barley.json` + `BarleyStageDetector`
  (`scripts/developmental_stage.py`); confidence capped ≤ 0.6 until calibrated.
- **Plant height, tiller count** — Phase 2, side view (see the `imaging` block in
  `barley.json`).
- **3D structure / biomass** — later (3D Gaussian Splatting; Masters track).

## Why no code lives here yet
Counting runs on the shared engine; staging on the existing detector; everything
else is config in `barley.json`. There is **no barley data yet** (Grace's trial is
Q4 2026), so building barley-specific pipeline code now would be speculative. This
module is intentionally just config + the data-collection spec until then.

## Status
- [x] `config/species/barley.json` — v0.1 warm-start (literature thresholds, Zadoks
      ladder, spike `object_count` scaffold, multi-view `imaging` note)
- [x] Shared engine + GWHD harness (`phenotyping/`) — barley reuses, no new counting code
- [ ] GWHD warm-start trained (Colab) → `phenotyping/weights/barley_spike.pt`
- [ ] **Grace's Q4 2026 trial data** → fine-tune + calibrate (gates everything)
- [ ] Multi-view (top + side) capture rig

## Build sequence (when barley data lands)
1. Train the GWHD warm-start — `phenotyping/datasets/gwhd_to_yolo.py` → `yolo detect train`.
2. Fine-tune it on barley spikes; validate with `phenotyping/eval_count.py`.
3. Calibrate `barley.json` thresholds on the first ~2 weeks (the calibration window).

**Collecting that trial data correctly is the dependency — see
[GROUND_TRUTH.md](GROUND_TRUTH.md) before the trial starts.**
