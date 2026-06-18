# Capture Schema — LOCK BEFORE ANY REAL CAPTURES

**Why this is urgent and irreversible.** Metrics derived from an image can only
be as good as what was in the frame when the shutter fired. If a capture is taken
without a scale fiducial, all size/area/greenness numbers from it are
**permanently invalid** — you cannot add the scale back later. If it is taken
without a sample ID, you cannot link it to destructive ground truth or (later) to
a chlorophyll vial. **Every real capture from day one MUST satisfy this spec.**
Re-shooting is rarely possible (plants change daily; harvest is one-shot).

This spec covers Task 1 (pod count/size) AND pre-positions Task 2 (per-organ
chlorophyll) so Task-1 captures never block Task 2. The Task-2 *models* are out of
scope now; the Task-2 *capture requirements* are in scope now.

---

## 1. Mandatory in-frame elements (every frame, every mode)

| Element | Purpose | Recommended |
|---|---|---|
| **Scale fiducial** | px→mm; without it size/area/greenness are invalid | **ArUco marker(s)** of known physical size (auto-detected → auto px/mm). Fallback: a rigid ruler. |
| **ColorChecker** | colour normalisation → reliable greenness now, chlorophyll later | X-Rite ColorChecker Classic/Passport, fixed position, in shade-free light |
| **Sample ID** | self-identifying image; links image → ground truth → (Task 2) vial | **QR or DataMatrix** encoding the `sample_id` (format below), printed, in frame |

### Sample-ID format (LOCKED 2026-06-18)
`BATCH-PLANT-ORGAN-ORGANNUMBER`, uppercase, hyphen-separated — regex
`^[A-Z0-9]{2,8}-\d{3}-(POD|STEM|LEAF|PLANT)-\d{3}$`:

| Field | Meaning | Example |
|---|---|---|
| `BATCH` | experiment/batch code, `[A-Z0-9]` 2–8 | `EXP1` |
| `PLANT` | zero-padded 3-digit plant number (001–999) | `001` |
| `ORGAN` | exactly one of `POD STEM LEAF PLANT` | `POD` |
| `ORGANNUMBER` | zero-padded 3-digit replicate of that organ | `001` |

Semantic rules (enforced after the regex): **`ORGAN==PLANT` ⇒ `ORGANNUMBER==000`**
(whole-plant / pod-count image); **`POD/STEM/LEAF` ⇒ `ORGANNUMBER≥001`**. Lowercase
entry is tolerated and canonicalised to uppercase. Examples:
`EXP1-001-PLANT-000` (Task-1 pod-count image), `EXP1-001-POD-001`,
`EXP1-012-STEM-003`. **Not in the ID:** variety, treatment, dates — those live in
the manifest; capture date/time is auto-recorded in the sidecar.

Why ArUco over a ruler: a marker of known side length gives **automatic,
per-frame** px/mm via corner detection (`cv2.aruco`), with no manual pixel
picking and no drift if working distance changes. Production frames carry a
**single ArUco marker** (fixed dictionary **DICT_5X5_100**) for per-frame
scale/pose — *not* the full board (the board is calibration-only, below).

> Build a `PixelScale` from the detected marker:
> `PixelScale.from_fiducial(marker_side_px, marker_side_mm, source="ArUco id=N")`
> (see [common.py](common.py)). Persist it in the sidecar (below).

### Marker sizes (per mode) + confirmation guard
Physical marker side length differs by mode (in-situ uses a larger marker so it
resolves across the turntable working distance):

| Mode | `MARKER_SIZE_MM` (default/placeholder) |
|---|---|
| `spread` | 30 mm |
| `daily_growth` | 30 mm |
| `in_situ_multiview` | 70 mm |

These are **placeholders**. A wrong marker size is a **silent, irreversible
systematic error in every measurement** — so the gate **refuses to start a
capture session** (`start_capture_session()` → `SessionNotReady`) while
`MARKER_SIZE_CONFIRMED = False`. Clear it only after the print rules below.

**Print rules (do exactly):**
1. Print at **100% scale** — disable "fit to page" / "shrink to fit".
2. Use **matte, rigid** stock (no gloss, no curl).
3. Keep the **white quiet-zone border** around the marker.
4. **Measure the actual printed square with calipers** — printers lie.
5. Enter THAT measured value into `MARKER_SIZE_MM`, then set
   `MARKER_SIZE_CONFIRMED = True`.

### One-time ChArUco distortion calibration (per camera / zoom / setup)
Lens distortion biases length/area — worst near frame edges — so it must be
removed before any measurement.

- **Board:** `cv2.aruco` ChArUco, **5×7 squares**, `squareLength 30 mm`,
  `markerLength 23 mm`, dictionary **DICT_5X5_100** (matches the production
  marker dict). Printable on A4 at 100%. (`charuco_board()` in
  [capture_profile.py](capture/capture_profile.py).)
- **Capture 15–25 shots** per camera/zoom/setup, varied angles + distances, with
  the board filling **different regions including the frame edges** (where
  distortion is worst).
- **Process:** detect ChArUco corners → `cv2.calibrateCamera` → store camera
  matrix + distortion coeffs to `calibration/intrinsics_<camera>.npz`
  (`calibrate_charuco()` + `save_calibration()`).
- **Accept only if mean reprojection error < 1.0 px** (target **< 0.5**). Higher →
  recapture with more varied poses. (`result["accepted"]`.)
- **Recalibrate triggers:** new camera · any zoom/focus change · any rig
  reconfiguration.
- **Every production frame is undistorted** with the stored intrinsics
  (`undistort()`) **before** ArUco scale, QR, ColorChecker, or any measurement.
- Production frames carry only a **single ArUco marker** (not the board) for
  per-frame scale/pose.

The sidecar records `calibration:{intrinsics_ref, distortion_ref, undistorted:true}`.

> **Dependency:** ChArUco detection works in base opencv ≥ 4.7; the ColorChecker
> (`cv2.mcc`) check needs **opencv-contrib-python**. Installing contrib covers
> both — use it at the rig (see [requirements.txt](requirements.txt)).

---

## 2. Per-frame sidecar (write next to every image)

`<image_stem>.json` alongside `<image_stem>.jpg`. Base metadata plus the blocks
populated by the enforcement gate ([capture/capture_profile.py](capture/capture_profile.py)):

```json
{
  "image": "2027-01-15_enriched_P3_spread.jpg",
  "timestamp": "2027-01-15T12:00:00",
  "capture_mode": "spread | daily_growth | in_situ_multiview",
  "sample_id": "EXP1-001-PLANT-000",
  "species": "brassica_napus",
  "cultivar": "<cultivar>",
  "chamber": "enriched",
  "pot_label": "P3",
  "view_angle_deg": 0,
  "exposure_us": 29985, "iso": 210, "wb_kelvin": 4742,
  "operator": "LB",

  "aruco": {"dictionary": "DICT_5X5_100", "marker_id": 23, "marker_size_mm": 30,
            "detected": true, "pose": {"distance_mm": 412.0, "rvec": [], "tvec": []}},
  "pixel_scale": {"mode": "spread", "type": "scalar", "value_or_ref": 8.24},
  "qr": {"decoded": "EXP1-001-PLANT-000", "valid": true, "canonical": "EXP1-001-PLANT-000",
         "batch": "EXP1", "plant": "001", "organ": "PLANT", "organ_number": "000",
         "plant_id": "EXP1-001", "manifest_known": true},
  "colorchecker": {"detected": true, "patches_found": 24},
  "quality": {"laplacian_var": 312.5, "exposure_ok": true},
  "calibration": {"intrinsics_ref": "calibration/brassica_intrinsics.json",
                  "distortion_ref": "calibration/brassica_intrinsics.json",
                  "undistorted": true},
  "compliance": {"passed": true, "reasons": []}
}
```

- `capture_mode` selects which pipeline path consumes the image, and which
  `pixel_scale.type` is required (see §3 / the gate).
- `view_angle_deg` is the turntable angle (in-situ multi-view only); used by the
  Track-2 3D dedup. `0` otherwise.
- `pixel_scale` is written from the detected ArUco (scalar) or, for
  `in_situ_multiview`, a depth-aware per-pixel map — never hard-coded.
- A frame is kept **only if `compliance.passed == true`**; rejected frames go to
  quarantine with their `reasons` (stable failure codes, see the gate).

**Filename convention:** `YYYY-MM-DD_{chamber}_{pot}_{mode}[_aNNN].jpg`
(`_aNNN` = view angle for multi-view). Matches the existing
`images/{chamber}/YYYY-MM-DD_{chamber}.jpg` style.

---

## 3. Capture modes

### 3a. `spread` — Track 1 endpoint spread-and-count (the absolute-count backbone)
- At final harvest: detach siliques, spread on **matte black cloth**, minimal
  overlap (this is the easy 2D case the current model targets).
- Top-down, fixed working distance, **locked exposure/WB**, ColorChecker + ArUco +
  QR in frame.
- One `sample_id` per plant → links directly to the manual count of that plant.

### 3b. `in_situ_multiview` — Track 2 (only after the COLMAP feasibility probe passes)
- Plant on a **turntable**; capture **8–12 views** at fixed angle steps, RGB +
  OAK-D depth at each. Same `sample_id`, incrementing `view_angle_deg`.
- ColorChecker + ArUco + QR visible across the orbit (or at a fixed reference pose).
- Feeds per-view detection → depth+angle 3D dedup → **calibrated** against the
  same plant's `spread` count.

### 3c. `daily_growth` — existing longitudinal RGB+depth
- The current daily chamber capture, now also carrying ArUco + ColorChecker + QR
  so any frame can yield valid size/greenness and be sample-linked.

---

## 4. Standardisation (per session)
- **Lock exposure + white balance** (reuse `scripts/capture_image.py` exposure
  locking) so colour is comparable across days — required for greenness and
  chlorophyll.
- Fixed working distance + lighting per mode; diffuse, shade-free illumination on
  the ColorChecker.
- Re-detect ArUco **every frame** (don't assume a constant px/mm).

---

## 4b. Plant manifest — [data/manifest.csv](data/manifest.csv)
Plant-level metadata, keyed by `BATCH-PLANT` (e.g. `EXP1-001`). One row per
plant; **every organ ID inherits its plant's metadata via this prefix** (so the
ID stays minimal — variety/treatment/dates are NOT in the ID). The gate's QR
check looks the prefix up:
- `manifest_mode="strict"` → an ID whose `BATCH-PLANT` is not in the manifest is
  **rejected** with `UNKNOWN_SAMPLE`.
- `manifest_mode="warn"` → frame **passes** but the unknown sample is surfaced
  **loud** in the operator's live pass/fail view (not just a log) and recorded
  (`qr.manifest_known=false`, `qr.manifest_mode="warn"`).
- no manifest provided → the check is skipped.

**RULE:** pilot/shake-down captures may use `"warn"`; the **real data-collection
run MUST use `"strict"`**. Warn must not silently become the production default —
every frame records `manifest_mode`, so this is auditable after the fact.

Columns: `plant_id, batch, plant_number, variety, treatment, sown_date, notes`.
Example: `EXP1-001,EXP1,001,Aviron,control,2026-03-10,first cohort`.

---

## 5. Task-2 (chlorophyll) sample-tracking — schema to honour now
The QR `sample_id` is the join key across a chain that must reconcile later:

```
image (mask → organ area in cm²)
   └─ sample_id ─┬─ extraction vial (fresh weight, leaf/organ area)
                 ├─ spectrophotometer reading (µg chlorophyll)
                 └─ → chlorophyll normalised PER AREA (µg/cm²)
```

- Normalise chlorophyll **per area (µg/cm²)** so it pairs directly with mask area
  (mask px × (mm/px)² → cm², enabled by the ArUco scale).
- Keep a `sample_tracking.csv`: `sample_id, image, vial_id, fresh_weight_g,
  area_cm2, chlorophyll_ug, chlorophyll_ug_cm2, timestamp`.
- Getting this join key in-frame **today** is the whole point — it is what makes
  the future calibration study possible without re-capturing.

---

## 6. Checklist (print and tape to the rig)
- [ ] ArUco marker(s) of known size in frame, unoccluded
- [ ] ColorChecker in frame, evenly lit
- [ ] QR/DataMatrix `sample_id` in frame and legible
- [ ] Exposure + WB locked for the session
- [ ] Sidecar JSON written (mode, sample_id, px_per_mm, angle)
- [ ] Filename follows convention
- [ ] (multi-view) all angles share one `sample_id`, correct `view_angle_deg`

**If any box is unchecked, the capture is not usable. Do not proceed.**

---

*Enforcement: a future `capture/capture_profile.py` will auto-detect the ArUco,
decode the QR, write the sidecar, and refuse to save a frame that fails the
checklist. Until then, follow this manually.*
