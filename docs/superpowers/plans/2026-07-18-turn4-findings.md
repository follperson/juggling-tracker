# Turn 4: Precision/Recall Quality Push — Findings

**Status: mixed-mode turn.** Workstreams 1–4 were TDD code changes (committed earlier this turn:
`a067233`, `d25629c`, `0653c56`, `99eaa48`, `aaf735c`, `0288b2a`, `e333cd7`, `f06ef8b`, `db3b5e3`;
185 tests); this document covers the whole turn including the closing exploratory ops (hard-negative
mining, corpus v4, v4 training, scorecard) which changed no code. Nothing was tuned to flatter
numbers; regressions are reported as measured.

Trigger: user field report — missing throws and "pupil" false positives on the new outdoor
videos.

## 1. Diagnosis: where real flights die (link_max_dt + apex attribution)

Frame-level attribution on the two field-report holdouts (v3 detections, defaults), with two
methodology corrections discovered mid-analysis (loose-config candidate diffs produce chimera
false-misses AND miss real losses that only a direct default-arc census finds):

| Video | Real flights | Found | Miss mechanism |
|---|---|---|---|
| af1.mov (in-domain) | 60 | 59 (98.3%) | 1× EM boundary artifact (needs min_points AND min_duration relaxed together; n=1, deferred) |
| PXL_20260716_191702756 @1280 (outdoor holdout) | 12 | 7 (58.3%) | 3× `link_max_dt` too tight (0.12 s ceiling vs measured 3–4-frame detection gaps of 0.125–0.166 s), 2× apex guard on genuinely shallow warm-up tosses |

Key ranking evidence: single-gate binary search recovered all 3 linker-gate misses by relaxing
only `link_max_dt`; the apex-guard pair have apex_gap −0.141/−0.086 (real low tosses, guard
working as designed). `resid_tol` was explicitly ruled out — the 2 af1 "misses" it appears to fix
are chimera artifacts of the diff method, and loosening it is the same knob implicated in the
Meschke over-count. Full details: `.superpowers/sdd/turn4-diagnosis.md`.

## 2. Recall fixes (commits `0653c56`, `99eaa48`)

1. `link_max_dt` 0.12 → 0.18 (linker tolerates three-frame detection gaps).
2. Apex guard only judges *witnessed* apexes (`apex_t` inside `[t_start, t_end]`); extrapolated
   apexes no longer rejected on fitted-parabola height.

Re-attribution after the fixes: outdoor holdout **7/12 → 9/12 found (75%)**. The 10th (t=5.74–6.20)
now extracts but its gap-bridged fit yields a witnessed apex_gap of 0.0042 — just under the 0.02
threshold — a boundary case neither fix addresses by design. af1 unaffected (59/60, same single
miss). Full details: `.superpowers/sdd/turn4-recall-report.md`.

## 3. VFR timestamps + overlay dot filtering (commits `a067233`, `d25629c`)

- `VideoReader.frames()` prefers per-frame decode PTS (`CAP_PROP_POS_MSEC`) over `idx/fps`, with
  epoch subtraction and a permanent fallback to `idx/fps` on non-monotonic/zero PTS — phone VFR
  footage's container fps is only an average.
- `render_overlay(dots="verified")` (new default): only arc-assigned detections get drawn — the
  same physics gate that excludes junk from counts now excludes it from the visual, so rejected
  detections (e.g. the reported "pupil dots") no longer alarm the user. `--dots all|none`
  preserve the old behavior / disable dots.

## 4. The validator bake-off: three lanes, two shipped, two refused

Three candidate run-validators were prototyped against a shared battery (af1, outdoor PXL,
ss3_id_016, ss441_id_013, sim seed sweep) before any implementation:

| Lane | Verdict | Why |
|---|---|---|
| Drift-cohort gate (clause A: dx dominance + monotonic cx + no alternation) | **SHIPPED** (`f06ef8b`, `events/validate.py`) | Fires only on the pinned slow-drift junk shape across the entire battery; kills the known junk-cohort gap (test flipped to assert rejection) with zero real-run casualties |
| Clause B ("freg" frequency-regularity) | **REFUSED** | Tuned against the old, bogus ss3 oracle target (23 — see §5); with the corrected reference it rejects real runs |
| Periodicity threshold gate | **REFUSED** | Same corrupted-reference problem; quality *fixes* shipped instead (`e333cd7`: short-window "can't judge" ≠ 0.0, arc-span scoring, adaptive lag band) with **no gating** — quality is scored and stored only |
| Split-stitch (lane 3 stage B) | **SHIPPED** (`db3b5e3`, final pass in `extract_arcs`) | Repairs gap-split flights (2 genuine af1 repairs, 59 → 57 arcs with all thrown windows still covered); crossing-balls guard verified from a second angle |

## 5. THE ORACLE CORRECTION: ss3 reference 23 → 52 (commit `0288b2a`)

The turn-3 headline problem — "v3 over-counts ss3_id_016: 39 catches vs oracle 23" — was an
artifact of the oracle itself. The old oracle script flattened all per-ball ground-truth
trajectories into one detection cloud before `analyze_detections`; a dense, perfectly-labeled
multi-ball cloud (balls tracked even while held) poisons the shared linker/EM and under-extracts
real flights.

`meschke_import.oracle_events` fixes this: `extract_arcs` runs **per ball trajectory** (no
cross-ball ambiguity exists in per-ball GT), arc unions feed the same shared `_events_from_arcs`
tail the real pipeline uses. Corrected references:

- **ss3_id_016: 52 catches / 5 runs** (was 23 — the "over-count" of 39 was actually an
  under-count; the problem dissolved into recall)
- **ss441_id_013: 136 catches / 1 run** (unchanged; its framing never triggered the merged-cloud
  interference)

Caveat (unchanged from the validator report): the corrected references are the corrected
*mechanism's* output, not frame-audited human ground truth.

## 6. Hard-negative mining (commit `aaf735c` + this turn's ops)

`export_hard_negatives`: frames where **every** detection is arc-unassigned (≥2 of them) become
zero-box COCO negative images; frames with any assigned detection are skipped as ambiguous.
`juggletrack label --negatives DIR` wires it into the labeling pass.

### Mining results (this turn)

Motion-detector pass (`label --detector motion --stride 1 --negatives`) over the 6 own new clips
+ af1 + af2:

| Source | Negative images |
|---|---|
| PXL_20260716_182734164 | 25 |
| PXL_20260716_182847912 | 36 |
| PXL_20260716_182922676 | 0 |
| PXL_20260716_183035587 | 28 |
| PXL_20260716_191622045 | 0 |
| PXL_20260716_191702756 (holdout; yielded nothing — see caveats) | 0 |
| af1 | 1 |
| af2 | 0 |

v3-detector pass (the pupil FPs come from v3, not motion): `analyze --model v3 --stride 1
--imgsz 1280 --save-intermediates` then `extract_arcs` + `export_hard_negatives` on the saved
jsonl, on PXL_191622045 + PXL_183035587 (NOT the holdout):

| Source | Candidates | Skipped ambiguous | Exported |
|---|---|---|---|
| PXL_20260716_191622045 | 156 | 321 | 40 (cap) |
| PXL_20260716_183035587 | 428 | 216 | 40 (cap) |

**Total: 170 negative images** across 6 non-empty source dirs (90 motion + 80 v3-based). Zero
frame overlap between the motion and v3 sets on the shared video.

### FINDING: the "pupil FP" diagnosis was wrong, and some negatives are contaminated

Visual audit of PXL_191622045's arc-unassigned clusters (actual frame crops, not coordinates):

1. The dominant cluster (y≈0.60–0.67, ~434 dets) is the juggler's **hands holding real balls**
   at hip height — correctly unassigned (no parabola in-hand), but wrongly eligible as "pure
   junk" negatives.
2. The face-height cluster (y≈0.40, ~125 dets) is a **real ball passing directly in front of the
   mouth/eye** mid-flight (frames 108, 630 confirmed by crop) — the visible mechanism behind the
   "pupil FP" report, but it is real-ball content the linker failed to stitch, not an appearance
   hallucination.

Consequence: **19/40 (47.5%) of PXL_191622045's exported negatives match a held-ball position
heuristic** (frame 108 — confirmed real ball at the mouth — is among the exported negatives).
PXL_183035587's negatives audited clean (0/40; its junk is framed wall pictures, confirmed by
crop). This is a pre-existing limitation of `export_hard_negatives` ("arc-unassigned" ≉ "not a
ball"), flagged for a follow-up fix, not patched in this measurement-only closing pass.

## 7. Corpus v4 + training

`datasets/flywheel-v4` = flywheel-v3's exact 134 sources (reconstruction verified to reproduce
v3's totals byte-for-count: 12,772 images / 44,281 boxes) + the 6 negative dirs. Negatives are
**1.31%** of corpus images (170/12,942), far under the ~10% cap — no subsampling.
`assemble_dataset(val_fraction=0.15, seed=0)`:

| Split | Sources | Images | Boxes |
|---|---|---|---|
| train | 119 | 11,774 | 40,180 |
| val | 21 | 1,168 | 4,101 |

Note: 2 of the 6 negative dirs (PXL_182734164, af1) landed in **val** under the source-level
seed-0 permutation — accepted, consistent with the existing source-level split convention, but it
means only 4 negative dirs (129 images) actually trained.

Training: `juggletrack train --model yolo11n.pt --epochs 40 --imgsz 640 --device mps --name
flywheel-v4`. **Interruption note (reproducibility):** the run was killed at epoch 11 when the
launching session ended and was resumed from `last.pt` (epoch 12 → 40) by the controller; zero
epochs lost; the resumed leg ran markedly slower (29 epochs / 11.1 h, vs ~5 min/epoch on the
first leg — overnight machine contention). Run dir:
`runs/detect/runs/finetune/flywheel-v4/` (worktree).

Val metrics — **not comparable to v3's 0.820 mAP50** (v4's val split contains negative images
and a different source permutation): best mAP50 epoch 7 (P .862 / R .594 / mAP50 .671); best
fitness epoch 38 (P .892 / R .605 / mAP50 .655 / mAP50-95 .488) — ultralytics selects `best.pt`
by fitness, so the shipped weights are epoch 38's. Copied to `models/juggletrack-v4/best.pt`.

## 8. The v4 scorecard (holdouts; the true comparison)

### 8a. Pupil-FP precision probe — PXL_191622045 @1280, stride 1

| Metric | v3 | v4 |
|---|---|---|
| Total detections | 1,744 | 1,435 |
| Post-static-filter survivors | 1,470 | 1,335 |
| Arcs extracted | 20 | **25** |
| Arc-unassigned (junk) | 1,108 | 630 |
| **Junk rate** | **63.5%** | **43.9%** |

Junk rate down ~20 pp while arcs went **up** — v4 wastes fewer detections on non-arc content and
extracts more physics-consistent trajectories on the mined video. Face-frame persistence: v4
still detects at the face point on both audit frames (108: conf .414; 630: conf .051) — and per
§6 that is **correct** behavior, since those detections are a real ball in front of the face.

### 8b. Coverage, v3 → v4 (stride 2; cov ≥1 / ≥2 / ≥3 %)

| Holdout row | v3 | v4 | Direction |
|---|---|---|---|
| af1 @640 | 58.2 / 30.8 / 8.2 | 76.5 / 41.3 / 9.5 | **up strongly** |
| af2 @640 | 80.4 / 62.5 / 43.7 | 73.9 / 56.2 / 33.0 | down moderately |
| PXL_191702756 @640 | 39.0 / 22.6 / 13.0 | 27.7 / 11.3 / 6.2 | **down sharply** |
| PXL_191702756 @1280 | 48.0 / 23.7 / 11.3 | 40.1 / 13.0 / 4.0 | **down sharply** |
| Meschke ss3 @640 | 100 / 100 / 100 | 100 / 100 / 100 | flat (median conf 0.91 both) |
| Kinetics -awp8ZYxm04 @640 | 22.3 / 17.5 / 10.2 | 32.5 / 27.1 / 20.5 | up |

The outdoor-holdout regression is material and is the honest cost of this negative batch. It is
*consistent with* the §6 contamination (the contaminated negatives come from the same backyard
environment as the holdout; a model taught "ball near body in this scene = background" would lose
exactly this recall), but causality is unproven without an ablation retrain.

### 8c. Events vs corrected per-ball oracle (stride 1 @640)

| Video | Corrected oracle | v3 (before) | v4 (after) |
|---|---|---|---|
| ss3_id_016 | **52** catches / 5 runs | 39 / 9 runs | 72 / 8 runs |
| ss441_id_013 | **136** catches / 1 run | 137 / 1 run | 134 / 1 run |

ss441 stays within ±2 of reference across both models. ss3 flips from −13 (under) to +20
(over): v4's recall gain on dense 3-ball footage (80 arcs vs v3's 56) pushes the event layer past
the reference — the over-count problem the corrected oracle "dissolved" at v3's operating point
re-emerges at v4's. The event-layer dedup/validator work (bake-off lanes that were refused on the
old reference) is worth revisiting against the corrected one.

### 8d. Outdoor holdout events — PXL_191702756 @1280, stride 1

| | v3 (current HEAD) | v4 |
|---|---|---|
| Detections | 368 | 227 |
| Runs | 2 (8.11–10.48, 11.01–12.13) | 1 (8.35–10.47) |
| Catches | **9** | **6** |

Regression, matching 8b. **Baseline correction:** the turn plan quoted v3's outdoor baseline as
"1 run / 7 catches" — that was the stale pre-recall-fix turn-3 number. v3 at current HEAD
(re-measured this session) gives 2 runs / 9 catches, agreeing with §2's 9/12 re-attribution; that
is the honest "before" column.

## 9. Honest caveats

- **The corrected oracle is still not frame-audited.** 52/136 are the per-ball mechanism's
  output on human-labeled trajectories, not human-counted catches. A human af1 catch count has
  been requested from the user and remains outstanding.
- **Negative contamination is real and quantified** (19/40 on one source, §6): fixing
  `export_hard_negatives` and re-mining before v5 is prerequisite to trusting negative-driven
  precision gains. The v4 outdoor recall drop may be partly or wholly this batch's fault.
- **v4 is a net regression for the field-report use case** (outdoor events 9 → 6) despite the
  precision win on the mined video. v3 remains the better outdoor-events model at current HEAD;
  do not promote v4 to the default weights without the ablation retrain.
- **Training interruption** (§7): killed at epoch 11, resumed from `last.pt` — believed benign
  (ultralytics resume is state-complete), but the run is not a single uninterrupted trajectory.
- The brief's motion-mining video list included the outdoor holdout (PXL_191702756); it was run
  as listed but produced 0 negative images, so no holdout-derived pixels entered training. The
  v3-based mining step deliberately substituted non-holdout videos per the brief's own
  correction.
- Meschke duplicate-label warnings during training (same as turn 3): YOLO drops exact-duplicate
  boxes; harmless, still unaddressed at the exporter level.

## 10. Next steps

1. **Resume the realtime plan** (deferred at turn-4 start) — the offline quality loop has hit
   diminishing returns until the items below land.
2. Fix `export_hard_negatives` contamination (follow-up task already spawned): exclude frames
   whose "junk" is plausibly a held/near-body real ball; then re-mine and ablation-retrain
   (v4b: same corpus, clean negatives) to attribute the outdoor recall drop.
3. Revisit the event-layer over-count on dense footage (ss3: 72 vs 52 at v4's recall) against
   the *corrected* oracle — the refused bake-off lanes were refused for tuning against 23, not
   for being wrong ideas.
4. Frame-audit a sample of ss3_id_016 catches to convert the corrected oracle from "better
   mechanism" to "verified reference"; collect the requested human af1 count.
5. The 78 moving-camera harvest clips remain deferred (need appearance-detector labeling).
