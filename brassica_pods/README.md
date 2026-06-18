# brassica_pods — Seed-Pod (Silique) Segmentation & Counting

RGB instance-segmentation module for Brassica siliques. Given an image it
returns per-pod masks, a total pod count, and per-pod size (length, width, area
in mm). **Task 1** of the Brassica imaging station; chlorophyll estimation
(Task 2) is a separate later workstream and out of scope here.

> This is a **standalone module**. It does **not** modify the thesis-critical
> Arabidopsis pipeline in [`scripts/`](../scripts/). Shared resources (SAM2
> checkpoint, project paths) are reused *by reference* via
> [`common.py`](common.py) — `scripts/` is never edited.

**Planning docs:** [PLAN.md](PLAN.md) (decisions + questions for Diarmuid) ·
[ROADMAP.md](ROADMAP.md) (technical) · [PROBE.md](PROBE.md) (feasibility gate) ·
[RIG.md](RIG.md) (hardware) · [CAPTURE_SCHEMA.md](CAPTURE_SCHEMA.md) (capture spec).
**Start at PLAN.md** — and run the feasibility probe before building any rig.

## Approach
- **Model:** Ultralytics **YOLO-seg** (YOLO11-seg / YOLOv8-seg) — fastest path to
  a working counter, CPU-runnable for inference, proven on this crop.
- **Data:** small hand-labelled seed set (SAM2-assisted) amplified by
  **deepcanola-style synthetic compositing** (paste cut-out pods + masks onto
  backgrounds). See [`synth/generate_dataset.py`](synth/generate_dataset.py).
- **Training:** GPU/Colab (local torch is **CPU-only**, confirmed Phase 0).
  Inference runs locally on CPU. OAK-D VPU deployment is deferred.

## Reference work (read, do not vendor)
- **deepcanola** (Atkins et al., GPL-3.0) — synthetic data + active learning.
  We reimplement the *methodology*; we do not copy the GPL code.
- **Rapepod_yolov8** — YOLOv8 + Mask R-CNN worked example for rapeseed pods,
  incl. per-pod size metrics.

## Layout
```
brassica_pods/
├── common.py            # paths + PixelScale (px->mm); bridge to scripts/
├── data/                # raw/, labels/, pod_cutouts/, pod_masks/,
│                        #   backgrounds/, synthetic/, dataset/ (YOLO-seg split)
├── capture/             # Brassica capture profile (Phase 1; reuses scripts/capture_image.py)
├── synth/               # deepcanola-style synthetic dataset generator  ✅ functional
├── train/              # pods.yaml + YOLO-seg trainer (placeholder — Phase 4)
├── infer/analyze.py     # analyze(image) -> {count, masks, per_pod}  ✅ phenotype extraction done
├── eval/                # count + size accuracy (placeholder — Phase 6)
└── weights/             # trained checkpoints (pods_best.pt)
```

## Inference contract
```python
from brassica_pods.common import PixelScale
from brassica_pods.infer.analyze import analyze

scale = PixelScale.from_fiducial(fiducial_px=412, fiducial_mm=50)  # ruler in frame
result = analyze("path/to/pods.jpg", scale=scale,
                 weights="brassica_pods/weights/pods_best.pt", greenness=True)
# result = {count, masks:[HxW bool], per_pod:[{length_mm,width_mm,area_mm2,...}],
#           method, greenness:{ngrdi/gcc/lab/greenness_score/...}}
```
`analyze()` needs trained weights; the **size extraction** (`infer.analyze.pod_size`)
works from masks alone and is the stable contract today.

## Generate a synthetic dataset
```bash
scripts/.venv/Scripts/python -m brassica_pods.synth.generate_dataset \
  --cutouts brassica_pods/data/pod_cutouts \
  --masks   brassica_pods/data/pod_masks \
  --backgrounds brassica_pods/data/backgrounds \
  --out     brassica_pods/data/synthetic \
  --num-images 500 --min-pods 8 --max-pods 60 --seed 42
```
RGBA cut-outs use their alpha as the mask (`--masks` optional); BGR cut-outs
need a sibling `<name>_mask.png`.

## px → mm scale (mandatory)
Every capture session **must** include a fiducial (ruler / ArUco tag) of known
length, or size metrics are invalid (plan §7). Build a `PixelScale` from it and
persist it next to the image; pass it into `analyze()`.

## Status — Phase 0 complete
- [x] Scaffold + venv + Ultralytics (8.4.70); CUDA = **False** (CPU)
- [x] Reference repos inspected
- [x] Synthetic generator (functional)
- [x] `analyze()` interface + phenotype extraction
- [ ] **Phase 1: capture protocol** ← next, pending approval + Diarmuid's answers
- [ ] Phase 2 labelling · Phase 3 baseline · Phase 4 train · Phase 5 phenotype · Phase 6 eval

## Requirements (answered by Diarmuid, 2026-06-17)
1. **Species: _Brassica napus_** (oilseed rape) → deepcanola weights apply
   directly as warm-start/baseline.
2. **Whole plant in-situ** (not excised/dried pods on a backdrop) → the **harder
   regime**: dense overlap/occlusion is the priority case for labelling + eval.
   deepcanola (whole-plant *B. napus*) is the closest reference; the black-cloth
   >90% paper setup does **not** apply.
3. **Count is the requirement.** Size metrics (length/width/area) kept because
   they come free off the masks. **Greenness metrics also wanted** — wired via
   `analyze(..., greenness=True)`, reusing `scripts/greenness_metrics.py`.
4. **Sample count unknown** → lean on synthetic compositing to amplify a small
   hand-labelled seed set.
