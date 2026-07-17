# Plan 3 Task 8 (Turn 3): Flywheel Turn 3 — Full-Corpus Run — Findings

**Status: exploratory pipeline run, not TDD.** Turn 3's deliverable was the first v3 detector
trained on the full multi-domain corpus plus the first TRUE end-to-end event-accuracy numbers
(catch counts vs Meschke ground truth). All tooling already existed (commits `575f996`, `3ab0b7d`,
`73b0b0f`); this turn is orchestration, measurement, and honest reporting. Nothing was tuned to
flatter numbers.

## Holdout battery (fixed before any labeling; never trained on, in any form)

| Holdout | Role |
|---|---|
| `af1.mov` | turn-1 holdout (original environment) |
| `PXL_20260716_191702756.mp4` | new-env holdout (turn 2) |
| Meschke `ss3_id_016` | event-accuracy GT holdout (3-ball cascade) |
| Meschke `ss441_id_013` | event-accuracy GT holdout (441; chosen — matches the turn-2 probe's sample dir) |
| Kinetics `-awp8ZYxm04`, `1klCnboHVQg`, `1oucU1rJvQU` | first 3 STATIC-triaged kinetics clips alphabetically |
| External `4mjaOWVIiLI.webm` | first STATIC-triaged external CC clip alphabetically |

## Corpus census

### Triage extension (step 0)

`outputs/turn3/triage.json` originally only globbed `.mp4` (180 entries). Extended it via direct
`juggletrack.data.triage.estimate_camera_motion` calls over the 52 missed `.webm/.mkv/.ogv` files
(41 in `data/raw/external`, 11 in `data/raw/commons-archive`), same 0.004 threshold. Final counts:

| Source dir | Total | Static | Moving |
|---|---|---|---|
| kinetics-jb | 143 | 88 | 55 |
| external | 49 | 38 | 11 |
| commons-archive | 20 | 8 | 12 |

**Moving-camera clips: skipped this turn** (78 clips across the three dirs) — MotionDetector
requires a static camera; these need an appearance-detector labeling pass with a model that works
on them, which is exactly what v3 now potentially is. Deferred to turn 4.

### Step 1 — bulk motion-labeling of the static harvest

130 static-triaged clips (85 kinetics, 37 external, 8 commons; the 4 in-harvest holdouts
excluded) labeled with `juggletrack label --detector motion --stride 1`, 4-way parallel. Zero
crashes. Wall time: kinetics median 1 s/clip, external median 9 s (max 64 s), commons median 17 s
(max 55 s) — the triage+busy-frame-guard pipeline made bulk labeling cheap.

**Only 16/130 clips cleared the ≥50-box gate** (2,992 images / 9,639 boxes kept):

| Clip (`outputs/turn3/labels/<name>`) | Images | Boxes |
|---|---|---|
| external-wfol2Ss1QWc | 1,996 | 7,471 |
| external-k6HT3g0pPHg | 399 | 720 |
| external-JyXZkjPbrRg | 92 | 324 |
| kinetics-jb-eBp6_EgnJjU | 89 | 188 |
| kinetics-jb-ocUj-Fy06w4 | 44 | 123 |
| external-b8thWrXTNgk | 34 | 113 |
| kinetics-jb-xam_opcO1rc | 20 | 100 |
| kinetics-jb-jxvRaInsdzw | 51 | 82 |
| external-mtV9NwVug2k | 28 | 75 |
| kinetics-jb-KSK4BSUy8TE | 46 | 72 |
| external-AzfxfSaU828 | 38 | 67 |
| external-90BfjKa4tb0 | 50 | 66 |
| commons-archive-commons_circus_jugglers_pope_voa | 16 | 64 |
| kinetics-jb-GwFpcPCOdbM | 20 | 62 |
| kinetics-jb-gTO_wwlGAN4 | 31 | 58 |
| kinetics-jb-D0mleCwCndg | 38 | 54 |

114 clips excluded (<50 boxes; most yield 0). The dominant failure mode on kinetics: ~10-second
clips leave little room after MOG2's 10-frame warmup, and frames busy with subject motion trip the
busy-frame guard; short + static-triaged does not imply motion-labelable. The exclusion list is in
the step-1 log (`.superpowers/sdd/turn3-report.md` running notes; per-clip logs in the session
scratchpad).

### Step 2 — bulk Meschke export

All 159 CSV↔video pairs resolved automatically (id-number fallback caught both known filename
quirks: `ss(6,6)(2,6x)_id_107.csv` ↔ `ss(66)(26x)_id_107.MP4` and the doubled-extension
`trick_5bShowerMultiplex_id_886.MP4.MP4`). 2 holdouts excluded → 157 export attempts:

- **107 exported ok** (first pass `frame_stride=5, max_frames=60, seed=0`: 6,388 images / 28,981
  boxes).
- **42 failed on non-integer coordinates**: `_read_csv_rows` casts with `int(v)` and these CSVs
  contain decimal pixel values (e.g. `191.1111111111111` — resampled/interpolated tracks in part
  of the corpus). A real importer limitation the 3-video probe never hit. NOT patched this turn
  (orchestration-only scope); flagged for a one-line `int(float(v))`-style fix + tests in turn 4.
- **8 failed on row-count/frame-count mismatch >2** (differences of 8 up to 755 frames) — the
  importer's own misalignment guard doing its job; these pairs are genuinely unreliable.

**Dominance check tripped:** at max_frames=60 Meschke was 53.8% of all boxes. Re-exported at the
brief's prescribed fallback **max_frames=40** → 107 ok / 4,267 images / **19,380 boxes**
(`outputs/turn3/labels-mf40/meschke-<stem>`). Meschke share fell to **43.8% — still above the
~40% guideline**, but this is the specified single fallback; documented rather than tuned further.

### Step 3 — ROI-crop pass (`crop_coco_source`, crop=640, margin=0.35, seed=0)

25 video-derived sources cropped → `outputs/turn3/cropped/`: 4 turn-1 appearance-labeled dirs
(af2, pxl, yt-3ball, yt-5easy), 5 turn-2c calibrated-motion dirs (the PXL clips), 16 turn-3
harvest dirs. Meschke and roboflow-canonical passed through uncropped (per plan: Meschke's balls
are already 0.065 of frame width; Roboflow boxes are already tight).

Box-count-weighted median relative ball size (box w / crop side) after cropping:

| Source group | Boxes | Median rel size | Notes |
|---|---|---|---|
| turn1-appearance | 4,247 | **0.066** | af2 0.066, pxl 0.092, yt-3ball 0.062, yt-5easy 0.065 |
| t2cal-motion (PXL) | 10,664 | **0.025** | was 0.008–0.010 pre-crop — 3× improvement |
| turn3-harvest | 8,655 | **0.036** | wfol2Ss1QWc 0.040 dominates; low-res clips crop to ≤360px sides |
| meschke (uncropped) | 19,380 | **0.037–0.065** | 0.065 of width; 0.0368 median vs long side |
| roboflow (uncropped) | 1,335 | 0.055 / 0.043 | universe / own |

The scale band is now ~0.02–0.07 across every group — the consistent-scale goal of the crop pass
is met (turn 2c's core problem was 0.008-rel balls, i.e. ~6 px at train resolution). 1,127 boxes
(2.5%) were dropped out-of-bounds by the crop's max-coverage fallback, mostly wfol2Ss1QWc's
wide-spread frames (957).

### Step 4 — datasets/flywheel-v3

`assemble_dataset(134 sources, val_fraction=0.15, seed=0)`: 25 cropped + 107 meschke-mf40 + 2
roboflow-canonical. Split is by-source (never by frame). **Train: 11,954 images / 40,746 boxes.
Val: 818 images / 3,535 boxes** (3,235 instances after ultralytics dedup) across 20 val sources —
18 meschke + external-90BfjKa4tb0 + kinetics-jb-jxvRaInsdzw. Box-share by family:

| Family | Images | Boxes | Share |
|---|---|---|---|
| meschke (mf40) | 4,267 | 19,380 | **43.8%** |
| t2-motion-cal | 2,241 | 10,664 | 24.1% |
| turn3-harvest | 2,992 | 8,655 | 19.5% |
| turn1-appearance | 2,724 | 4,247 | 9.6% |
| roboflow-canonical | 548 | 1,335 | 3.0% |
| **Total** | **12,772** | **44,281** | |

## Training v3

`juggletrack train datasets/flywheel-v3/data.yaml --model yolo11n.pt --epochs 40 --imgsz 640
--device mps` (base yolo11n, AdamW lr0=0.002 auto, batch 16 — the turn-1/2c recipe). ~3.4 h wall
(748 batches/epoch at ~2.3 it/s — the brief's ~90 min estimate assumed a v2-sized corpus; v3's is
3× larger). **Reproducibility note:** the run was accidentally killed at epoch 26 by the session
controller and resumed from `last.pt` (ultralytics resume); zero epochs lost, but the run is a
2-process resume rather than one continuous process. Run dir (worktree):
`runs/detect/runs/finetune/flywheel-v3/`.

**Best-by-fitness epoch 39: P 0.897 / R 0.724 / mAP50 0.820 / mAP50-95 0.612** (final best.pt
val print: P 0.906 / R 0.719 / mAP50 0.820 / mAP50-95 0.612; 818 images / 3,235 instances).
Weights copied to `/Users/andrew.follmann/personal-projects/juggling/models/juggletrack-v3/best.pt`.

This is the first meaningful val metric of the flywheel: v2b/v2c val (mAP50 0.039 / 0.024) was
structurally doomed — 100% motion-labeled portrait footage with ~6 px balls at train resolution.
v3's val is mixed-domain and scale-normalized, and the curve behaves like a real learning curve
(monotone-ish rise to a 0.82 plateau). Composition caveat: val is 90% Meschke sources by count,
so the 0.82 mostly certifies "learns Meschke-domain balls"; the cross-domain story is what the
holdout batteries below measure.

## Eval battery A — coverage (stride 2, conf 0.05)

cov≥1/≥2/≥3 = fraction of sampled frames with at least 1/2/3 detections. Post = after
`filter_static_detections`.

### af1.mov — turn-1 holdout (600 sampled frames)

| Model | ≥1 | ≥2 | ≥3 | dets (post-filter) |
|---|---|---|---|---|
| stock | 64.0% | 44.2% | 18.3% | 785 (785) |
| **ft-v1** | **99.7%** | **95.8%** | **92.3%** | 3,510 (1,400) |
| v3 @640 | 58.2% | 30.8% | 8.2% | 603 (603) |
| v3 @1280 | 95.5% | 84.2% | 67.2% | 2,627 (1,483) |

### PXL_20260716_191702756 — new-env holdout (177 sampled frames)

| Model | ≥1 | ≥2 | ≥3 | dets (post-filter) |
|---|---|---|---|---|
| stock | 4.0% | 0.0% | 0.0% | 7 (7) |
| ft-v1 | 74.6% | 53.1% | 29.4% | 328 (328) |
| v3 @640 | 39.0% | 22.6% | 13.0% | 160 (160) |
| v3 @1280 | 48.0% | 23.7% | 11.3% | 191 (191) |

### Meschke holdouts (3,075 / 2,970 sampled frames)

| Video | Model | ≥1 | ≥2 | ≥3 |
|---|---|---|---|---|
| ss3_id_016 | stock | 99.7% | 72.6% | 18.8% |
| ss3_id_016 | ft-v1 | 100% | 74.6% | 29.3% |
| ss3_id_016 | **v3** | **100%** | **100%** | **100%** |
| ss441_id_013 | stock | 97.9% | 70.6% | 9.1% |
| ss441_id_013 | ft-v1 | 99.7% | 58.3% | 25.1% |
| ss441_id_013 | **v3** | **100%** | **100%** | **98.2%** |

v3's median confidence on both Meschke holdouts is **0.91** (stock 0.28/0.35, ft-v1 0.20/0.20) —
these are confident, per-ball detections, not threshold-straddling noise.

### Kinetics + external holdouts

| Video | Model | ≥1 | ≥2 | ≥3 |
|---|---|---|---|---|
| kinetics--awp8ZYxm04 | stock | 0.0% | 0.0% | 0.0% |
| kinetics--awp8ZYxm04 | ft-v1 | 49.4% | 36.1% | 26.5% |
| kinetics--awp8ZYxm04 | v3 | 22.3% | 17.5% | 10.2% |
| kinetics-1klCnboHVQg | stock | 9.2% | 0.5% | 0.0% |
| kinetics-1klCnboHVQg | ft-v1 | 71.4% | 32.1% | 15.8% |
| kinetics-1klCnboHVQg | v3 | 6.6% | 0.0% | 0.0% |
| kinetics-1oucU1rJvQU | stock | 0.0% | 0.0% | 0.0% |
| kinetics-1oucU1rJvQU | ft-v1 | 13.2% | 0.0% | 0.0% |
| kinetics-1oucU1rJvQU | v3 | 1.5% | 0.0% | 0.0% |
| external-4mjaOWVIiLI | stock | 30.7% | 6.5% | 1.1% |
| external-4mjaOWVIiLI | ft-v1 | 100% | 99.4% | 97.8% |
| external-4mjaOWVIiLI | v3 | 35.0% | 13.3% | 4.8% |

**Honest reading of battery A.** v3 is a domain-shifted model, not a strictly better one: it
*dominates* on Meschke-like footage (100/100/100 at 0.91 median conf — the training corpus'
largest family) and improves sharply on af1 when inference resolution matches its training scale
band (v3@1280 95.5/84.2/67.2 vs @640 58.2/30.8/8.2 — v3 learned small balls, and af1's balls at
640 sit at the band's edge). But ft-v1 still wins on af1@640, on the new-env holdout (74.6% vs
48.0% even after v3's @1280 boost), and on all four harvest holdouts. A caveat cuts against ft-v1
in that comparison, though: ft-v1's high coverage rows carry known static-FP inflation (60% of its
af1 detections are static-filtered; its median conf on the harvest rows is 0.06–0.11), while v3's
Meschke coverage is confident and filter-clean. Also note the harvest-holdout rows measure clips
whose *training representation* is 16 clips totaling ~9.6k boxes — the harvest family is present
but thin and heterogeneous (YouTube compilations, mixed content), so weak transfer there is
consistent with, not contradictory to, the corpus composition.

## Eval battery B — TRUE event accuracy (first ever)

### Oracle reference (GT trajectories → `analyze_detections`, pure defaults)

The gravity floor (ay 0.05) + apex guard (0.02) fixes at HEAD `73b0b0f` make the oracle work at
pure DEFAULTS (turn 2's probe needed manual g_range surgery):

| Holdout | GT points | Arcs | Runs | Catches | Drops |
|---|---|---|---|---|---|
| ss3_id_016 | 18,450 (3 balls) | 45 | 8 | 23 | 0 |
| ss441_id_013 | 17,820 (3 balls) | 139 | 1 | 136 | 0 |

**Oracle caveat (stated plainly): the reference is arc-derived, not human-verified catch counts.**
These runs/catches are what the event core produces from *perfect* detections; scoring a
prediction against them measures the detector's contribution to the pipeline, not absolute
event-truth. No drops exist in either oracle output, so drop P/R below is vacuous (1.00 with 0
TP by convention).

### Headline table (scored with `evaluate_session` against oracle-derived VideoLabels)

All predictions: full pipeline `analyze` at stride 1, imgsz 640, conf 0.05 (v3's 640-coverage on
both holdouts is 100% ≥1, so no 1280 re-run was triggered per the brief's <50% rule).

| Video | Model | Pred runs / oracle | Run IoU≥0.9 | Catch ±1 | Total catches (pred/oracle) | Drop P/R |
|---|---|---|---|---|---|---|
| ss3_id_016 | ft-v1 | 1 / 8 | 0% | 0% | 15 / 23 | — (no GT drops) |
| ss3_id_016 | **v3** | **9 / 8** | 12% | 12% | **39 / 23** | — (no GT drops) |
| ss441_id_013 | ft-v1 | 1 / 1 | 0% | 0% | 20 / 136 | — (no GT drops) |
| ss441_id_013 | **v3** | **1 / 1** | 0% | **100%** | **137 / 136** | — (no GT drops) |

Per-run detail:

- **ss3_id_016 / ft-v1** (156 arcs → 1 run): a burst of spurious arcs in the first ~2 s and
  nothing sustained after — detections exist everywhere (coverage ~100% ≥1) but are too
  noisy/mislocalized for arc chains. Matched run IoU 0.14, catch error 12.
- **ss3_id_016 / v3** (18,516 dets → 56 arcs → 9 runs, 39 catches): real run structure across the
  whole 205 s clip for the first time. 5 of 8 oracle runs matched (IoU 0.97/0.79/0.41/0.39/0.28);
  the best-matched run has catch error 0. But v3 over-counts: 39 total catches vs the oracle's 23,
  3 oracle runs unmatched, 4 predicted runs spurious or fragmented. The run-level catch-±1 metric
  (12%) is honest about this: one run in nine agrees.
- **ss441_id_013 / v3** (19,008 dets → 139 arcs → 1 run, 137 catches): near-oracle-identical arc
  structure — the oracle also finds exactly 139 arcs and 1 run, and total catches differ by ONE
  (137 vs 136). Run-boundary IoU is only 0.35, but inspection shows both runs contain the *same
  139 arcs*; the start/end times are extrapolated hand-line crossings of the first/last arc fits,
  which diverge between the two (oracle 0.59–23.40 s vs pred −1.84–64.21 s, both nonsensical as
  literal spans for a 198 s continuous pattern). The boundary metric here measures a run-boundary
  definition quirk in `segment_runs`, not detector quality — flagged as a next-step.

**Takeaway:** on the domain the corpus is rich in, the full pipeline's catch counting went from
useless (ft-v1: 15/23 and 20/136 with no run structure) to within ±1 on one holdout and
structurally right but over-counting ~70% on the other. That over-count (39 vs 23 on the
3-ball-cascade clip) is the new headline problem: v3's confident detections produce more
gravity-plausible arcs than the GT trajectories do (56 vs 45), suggesting box-center jitter or
duplicate detections are being read as extra flight segments.

## Eval battery C — own-footage events (v3, stride 1, overlays)

| | af1 ft-v1 (turn 1) | af1 ft-v2c (turn 2c) | af1 v3 | 191702756 ft-v1 | 191702756 ft-v2c | 191702756 v3 @640 | 191702756 v3 @1280 |
|---|---|---|---|---|---|---|---|
| Detections | 7,025 | 2,400 | 1,188 | 328 | 106 | 335 | 368 |
| Arcs | 58 | 62 | 59 | 1 | 2 | 10 | 10 |
| Runs | 3 | 3 | 3 | 0 | 0 | **1** | **1** |
| Catches | 55 | 50 | 50 | 0 | 0 | **4** | **7** |
| Drops | 0 | 2 | 1 | 0 | 0 | 0 | 0 |

- **af1 / v3 @640:** runs 1.86–4.18 s (5 catches), 8.89–9.46 s (15), 23.31–24.16 s (30); 1 drop at
  t=36.04 s (floor_descent + periodicity_collapse — same event and signals as prior models' drop
  detections, without ft-v2c's double-fire). Same 3-run structure as every model since stock; catch
  total 50 ties ft-v2c, still under ft-v1's 55. Notably v3 does this from 1,188 detections vs
  ft-v1's 7,025 — a much sparser but cleaner detection stream.
- **191702756 / v3:** **the first event-level output any appearance model has produced on the
  new-env holdout** (ft-v1, ft-v2b, ft-v2c all had 0 runs there, at any imgsz). @640: 1 run
  8.32–9.85 s, 4 catches. @1280: 1 run 8.34–12.12 s, 7 catches — consistent with the coverage
  table (@1280 adds detections). No drops recovered. No human GT exists for this clip, so these
  counts are unverified — but 0 → nonzero is the structural change.

Overlays: `/Users/andrew.follmann/personal-projects/juggling/outputs/turn3/analyze-v3/{af1,PXL_20260716_191702756,PXL_20260716_191702756-1280}/overlay.mp4`.

## Caveats

1. **Oracle-as-reference is arc-derived** — the event core's own output on perfect input, not
   human catch counts. Catch-count errors vs the oracle isolate detector-induced degradation;
   they do not certify absolute counting accuracy.
2. **Motion labels remain entirely unaudited** (now ~29k motion-derived boxes across t2cal +
   harvest). The arc gate verifies position, not extent.
3. **Meschke importer coverage**: 50/157 sources failed (42 float-coordinate CSVs, 8 frame-count
   mismatches) — the corpus in v3 is the *integer-labeled, aligned* 2/3 of Meschke. Fixing the
   float parse would add ~40 sources next turn.
4. **Meschke box size is a constant prior** (0.065×width square), not per-video measured.
5. **NC/ND-licensed commons clips are flagged internal-only** — commons_circus_jugglers_pope_voa
   is the only commons source in v3; check its license tag before any model release.
6. **Meschke = 43.8% of training boxes** — above the ~40% guideline even after the max_frames=40
   fallback. Domain balance is Meschke-heavy; val (90% Meschke) inherits this.
7. **Training was interrupted/resumed once** (epoch 26, controller kill, ultralytics resume from
   last.pt) — bitwise reproducibility of the run is compromised, metrics lineage is not.
8. **Moving-camera clips (78) untouched** this turn.

## Next steps

1. **Catch over-counting on dense detections** (ss3_id_016: 39 vs 23) — v3's clean detections
   yield MORE arcs than GT trajectories; investigate duplicate-detection merging / arc
   fragmentation before the next accuracy round.
2. **Run-boundary extrapolation quirk** in `segment_runs` — both oracle and prediction produce
   physically nonsensical start/end times on a continuous 198 s pattern (0.59–23.40 s covering 139
   arcs); boundary IoU is currently not a trustworthy metric on long continuous runs.
3. Fix `_read_csv_rows` float-coordinate parse (+ tests) → +42 Meschke sources.
4. Appearance-labeling pass (v3 as the labeler) over the 78 moving-camera clips and the 114
   below-gate static clips — v3 can now bootstrap where motion labeling couldn't; also consider
   per-video inference-imgsz selection (af1's @1280 coverage jump says one global imgsz is
   leaving accuracy on the table).
5. Human audit slice of the motion-label corpus (still open from turn 2).
6. Human event labels for af1/af2 (true accuracy vs human, not arc-derived oracle).
7. Meschke per-video ball-size estimation (replace the 0.065 constant).

## Artifacts (not committed)

- Extended triage: `/Users/andrew.follmann/personal-projects/juggling/outputs/turn3/triage.json`
- Harvest labels: `/Users/andrew.follmann/personal-projects/juggling/outputs/turn3/labels/<source>-<stem>/`
- Meschke exports: `/Users/andrew.follmann/personal-projects/juggling/outputs/turn3/labels-mf40/meschke-<stem>/` (max_frames=40, used) and `labels/meschke-<stem>/` (max_frames=60, superseded)
- Cropped sources: `/Users/andrew.follmann/personal-projects/juggling/outputs/turn3/cropped/`
- Dataset: `/Users/andrew.follmann/personal-projects/juggling/datasets/flywheel-v3/`
- Weights: `/Users/andrew.follmann/personal-projects/juggling/models/juggletrack-v3/best.pt` (run dir in worktree `runs/detect/runs/finetune/flywheel-v3/`)
- Oracle outputs: `/Users/andrew.follmann/personal-projects/juggling/outputs/turn3/oracle/`
- Coverage detections: `/Users/andrew.follmann/personal-projects/juggling/outputs/turn3/coverage/`
- Analyses: `/Users/andrew.follmann/personal-projects/juggling/outputs/turn3/analyze-ftv1/` and `analyze-v3/` (per-video `eval_report_vs_oracle.json` inside each Meschke analysis dir)
- Execution notes: `.superpowers/sdd/turn3-report.md` (worktree)
