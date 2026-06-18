# Imaging Rig — Build Guide

Two rigs. Build the **station** (the product) and a simple **spread bench** (the
calibration answer-key). Both feed the same software via the capture gate; both
must satisfy [CAPTURE_SCHEMA.md](CAPTURE_SCHEMA.md) (ArUco + ColorChecker + QR in
every frame, locked exposure, ChArUco-calibrated camera).

> Sizing depends on one unknown: **how big/bushy the plants are at podding.**
> Confirm that with Diarmuid before buying — it sets the station's working
> distance, backdrop size, and turntable load. Numbers below assume a ~40–70 cm
> potted plant.

---

## Rig 1 — The Station (PRODUCT): potted live plant, multi-view

A controlled booth the lab feeds potted plants through one at a time. Plant on a
turntable, camera fixed, rotate for 8–12 views (RGB + depth), count, return plant
to chamber. Non-destructive.

```
   plain matte backdrop  ───────────────┐
                                         │
     [ OAK-D Lite ] ── horizontal ──►   │   ← camera fixed, ~60–80 cm from plant,
        on tripod / rail                 │     centred on plant mid-height
                                         │
        diffuse light \      / diffuse   │   ← 2 LED panels at ~45°, even, no glare
                       \    /            │
                   ( potted plant )      │
                   ════ turntable ════   │   ← plant rotates; camera + lights still
                   ─────── bench ────────┘
        ArUco + ColorChecker + QR in frame (on the turntable platform / a fixed tab)
```

**Components + rough cost (EUR):**
| Part | Note | ~Cost |
|---|---|---|
| OAK-D Lite | you have it; use RGB **and** stereo depth here | – |
| Turntable | lazy-susan bearing + **NEMA-17 stepper + Arduino** for repeatable angles (you already use Arduino), or manual with marked detents | 30–120 |
| Backdrop | matte, uniform, **contrasting** colour (mid-grey or blue reads well against green/brown pods); cloth or board, wrinkle-free | 15–40 |
| 2× LED panel lights | bi-colour, dimmable, **with diffusers**; even + shadow-free | 80–200 |
| Camera mount | tripod or a rigid rail/arm; must not move between plants | 30–80 |
| ColorChecker | Calibrite/X-Rite (24-patch); cheaper card OK | 30–120 |
| ArUco marker (~70 mm) + QR labels | matte rigid print; **caliper-measure the printed size** | <10 |
| ChArUco board (A4) | one-time calibration | <5 |
| Booth / enclosure (optional) | blocks stray light → repeatable | 0–150 |

**Build steps:**
1. Fix the camera horizontal at plant mid-height, **~60–80 cm** away (OAK-D RGB
   ~69° HFOV / ~55° VFOV → ~70 cm fits a ~60 cm plant). **Lock the position** —
   px↔mm and depth depend on it.
2. Hang the backdrop behind the turntable, wrinkle-free, filling the frame behind
   the plant.
3. Place the two lights at ~45° each side; tune for **even, glare-free** light
   (check the ColorChecker has no hotspots).
4. Mount the turntable so the pot sits centred; mark or program **8–12 equal
   angles** (e.g. every 30–45°).
5. Put the ArUco + ColorChecker + a QR holder **in frame at a fixed spot** (on the
   turntable platform edge, so they rotate with the plant, or on a fixed tab —
   either works as long as the ArUco is visible each shot).
6. **Lock exposure + white balance** (reuse `scripts/capture_image.py` locking).
7. **One-time:** ChArUco calibration (`calibrate_charuco`), reproj < 1.0 px.
8. Per plant: load pot → for each angle, capture RGB+depth, same `sample_id`
   (`EXP1-NNN-PLANT-000`), incrementing `view_angle_deg`. Gate
   (`station_multiview` mode) rejects any non-compliant frame.

**Why this works:** controlled background + even light + free rotation + depth is
exactly what the software needs to fight occlusion — none of which you get from a
plant left in the chamber.

---

## Rig 2 — Spread bench (CALIBRATION answer-key + fallback)

Top-down copy-stand over matte black cloth. Used on a SUBSET of plants only:
strip the siliques, spread them, image, then hand-count the same plants. Produces
the ground truth that calibrates the station.

```
        [ OAK-D Lite, lens down ]   ← fixed height ~40–55 cm (RGB only; pods flat)
              |  |
   light \    |  |    / light       ← diffuse, no glare on the black
   ┌───────────────────────┐
   │ ▦ ColorChecker  ▣ ArUco│        ← matte BLACK cloth, level bench
   │   pods spread, minimal │
   │   overlap   ▤ QR       │
   └───────────────────────┘
```
Same fiducial/ColorChecker/QR rules; mode `spread` (scalar scale, no depth
needed). This is the easy case — ~10–15% is very achievable, so it's a
trustworthy yardstick. Reuses most of the station's lighting + markers.

---

## Shared / mandatory (both rigs)
- **Every frame:** ArUco (caliper-measured size) + ColorChecker + QR sample-ID.
- **Lock exposure + WB** per session.
- **ChArUco-calibrate** each camera/zoom/setup; undistort before measuring.
- Confirm `MARKER_SIZE_MM` and set `MARKER_SIZE_CONFIRMED=True` or the gate won't
  start a session.
- Install **opencv-contrib-python** at the rig (ColorChecker check).
- The gate ([capture/capture_session.py](capture/capture_session.py)) keeps only
  compliant frames; rejects go to quarantine.

## Before buying — confirm with Diarmuid
1. **Plant size/bushiness at podding** → station working distance, backdrop size,
   whether ~10–15% is even reachable (the feasibility probe).
2. **Pot size + weight** → turntable load + platform diameter.
3. **Throughput (plants/day)** → motorised stepper turntable vs manual detents.
4. **Backdrop colour** → pick for best contrast against mature (yellow/brown) pods
   on a green-ish plant; mid-grey or blue are safe defaults.

## Build order
1. **Smartphone→COLMAP feasibility probe** on one real potted plant FIRST — if
   pods can't be resolved, do not build the station depth rig.
2. Spread bench (cheap, unblocks calibration + uses the current model directly).
3. Station rig (only if the probe passes).
