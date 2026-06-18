# Feasibility Probe — can a station count pods on an intact plant to ±10–15%?

**Run this BEFORE building the station rig or promising the deliverable.** It is
cheap (a phone, a tally counter, a few plants, a few hours) and answers the one
question the whole product rides on. No rig, no model, no code-to-train needed.

## The core idea
The station can't see interior pods — a fraction are hidden from every external
view. We don't escape that by seeing them; we escape it by **calibration**:
estimate `true ≈ visible / f`, where `f` = the visible fraction. That works only
if **`f` is consistent across plants**.

Crucial maths: with a perfect detector and a single global calibration constant
`f_mean`, each plant's count error is exactly `|f − f_mean| / f_mean`. So the
**best-possible mean %error ≈ CV(f)** — the coefficient of variation of the
visible fraction. A *real* detector adds error on top. Therefore:

> **CV(f) is an optimistic FLOOR on achievable count error.** If CV(f) is already
> above your target, the target is impossible — no detector, sensor, or number of
> views can fix it. This is what the probe measures.

(With Diarmuid's lab-tool use case — no enriched/control comparison — we only need
`f` stable across his *normally-grown* plants, which is far more achievable than
across deliberately different treatments.)

## Materials
- **≥ 8 representative podding plants** (more is better; spanning the size/bushiness
  range the lab actually grows — this is the population `f` must be stable over).
- A phone (or the OAK-D), even lighting, a way to rotate the plant / walk around it.
- Tally counter, gloves, bags, calipers optional.

## Protocol — per plant
**Step 1 — Visible count (non-destructive, the generous ceiling):**
Rotate the plant / walk fully around it under good light. Count every silique you
can see **WITHOUT touching or parting any branches** — only what is externally
visible. Tally → `V_human`. (A human with free movement + stereo vision sees *more*
than a fixed camera rig, so this is an upper bound on what any station could see.)

**Step 2 — Photo-visible count (optional, more realistic):**
Fix the plant, take **8–12 photos** at even angles at the planned working distance.
Count distinct pods visible **only in those photos** → `V_photo`. This approximates
the real fixed multi-view rig (always ≤ `V_human`). If you do this, base the
decision on `V_photo` (closer to reality).

**Step 3 — True count (destructive):**
Strip every pod and count the true total → `T`. **Time this step** — it is the
manual method the lab wants replaced, so it doubles as the throughput baseline.

## Step 4 — Resolvability sub-test (one plant, once)
Take **30–50 overlapping photos** around one plant; run **COLMAP** (or any
photogrammetry). Look at the reconstruction: are individual siliques resolved as
separate structures, or do they blur into mush?
- **Resolved** → multi-view 3D de-dup is viable; depth-based counting has a chance.
- **Mush** → high-res photogrammetry already fails, so the OAK-D's coarse stereo
  depth definitely will. **Drop the 3D-dedup plan**; fall back to 2D per-view
  counting + calibration only, and don't budget for depth.

## Decision rule — SET NOW, do not move after seeing the data
Compute `f = visible / T` per plant (use `V_photo` if collected, else `V_human`),
then `f_mean`, `std(f)`, `CV(f) = std/mean`. (`brassica_pods/feasibility_probe.py`
does this.)

| Condition | Verdict | Meaning |
|---|---|---|
| `CV(f) ≤ 0.10` **and** `f_mean ≥ 0.5` | **GO** | ±10–15% plausible with a good detector — build the station |
| `0.10 < CV(f) ≤ 0.20` | **MARGINAL** | ±15% unlikely; realistically ±20–30%. Decide if useful as an *estimate* |
| `CV(f) > 0.20` **or** `f_mean < 0.4` | **NO-GO** | Non-destructive absolute count not achievable; fall back to spread-and-count (semi-automated) or estimate-only |

Two hard truths baked in:
- **Even a perfect detector cannot beat `CV(f)`.** The real detector makes it worse,
  so treat the GO threshold as needing headroom (hence ≤0.10, not ≤0.15).
- **Low `f_mean` is dangerous independently:** if most pods are hidden, small
  errors in the visible count get amplified by `1/f`, and detection on a packed
  canopy is itself unreliable.

## What the probe also gives you (free)
- **Throughput baseline** — the timed manual strip+count, i.e. what the station
  must beat to be worth it.
- **Pods-per-plant range** — sets how hard the per-view detector has to work.
- **A real `f` distribution** — the actual calibration curve seed if you proceed.

## Why this is the right next step
Everything built so far is scaffolding around an untested physical assumption.
This probe converts "we hope intact-plant counting works" into a measured number
with a pre-committed decision. If it says NO-GO, you've saved months of rig-building
for ~€0. If GO, you proceed with evidence, not hope.
