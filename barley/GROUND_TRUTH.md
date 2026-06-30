# Barley trial — ground-truth collection protocol

What to record in Grace's Q4 2026 barley trial so the data trains/validates the CV
models **the day it arrives**. Without this, the images are unusable for ML. Light
enough to do routinely, strict enough to be machine-usable. Agree it with Grace
**before the trial starts** — it can't be added retroactively.

## 1. Imaging (every capture)
- **Two views per plant/pot:** top-down (nadir) + side. Same geometry every day.
- **Fixed working distance + lighting** (the imaging-station principle).
- **A fiducial of known size in every frame** (ruler / ArUco tag). Without a px→mm
  scale, all size/height metrics are meaningless.
- **Consistent naming:** `YYYY-MM-DD_{potID}_{view}.jpg`  (view = `top` | `side`).
- Daily, same time of day.

## 2. Per-plant metadata (record once)
- **Variety + row-type** (2-row vs 6-row — changes spike appearance and the fine-tune).
- Sowing date, pot/plant ID, treatment (CO₂ / temp / light condition).

## 3. Manual measurements (the labels)
Record ~2–3×/week; spike traits from heading onward:

| Trait | How | Feeds |
|---|---|---|
| **Spike count** | manual count per plant (the headline label) | spike-detector fine-tune + eval |
| **Zadoks score** | decimal code per plant (00–92) | growth-stage detector calibration |
| **Plant height** | ruler, tallest point (mm) | side-view height calibration |
| **Tiller count** | count tillers (Zadoks 21–29) | tiller-count calibration |
| **Senescence / SPAD** | visual % senesced or SPAD (optional) | greenness/senescence bounds |

## 4. For the spike-detector fine-tune specifically
- ~50–150 top-down images **bounding-box labelled** for spikes (post-heading) is
  enough to fine-tune the GWHD warm-start. Hand-count everything; box-label a subset.
- **Cover the range:** sparse → dense canopies, and the heading→ripening colour shift
  (green spike → golden). Label only one appearance and the model fails on the rest.

## 5. The calibration window
Treat the **first ~2 weeks** as calibration: extract real HSV/greenness bounds,
height/cover ladders, and Zadoks timing from those images + manual scores, then lock
`config/species/barley.json`. Everything in that file today is a literature-
approximate placeholder.

## Why each item matters (the failure if skipped)
- No fiducial → no mm → height/size metrics useless.
- No manual spike counts → nothing to fine-tune or validate the counter against.
- No Zadoks scores → the stage detector stays at ≤ 0.6 placeholder confidence forever.
- Single-appearance labels → the detector breaks on the colour shift it never saw.
- No two-view capture → no height/tillering, and no 3D on-ramp later.
