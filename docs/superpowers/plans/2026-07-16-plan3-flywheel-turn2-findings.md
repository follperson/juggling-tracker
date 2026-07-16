# Plan 3 Task 8 (Turn 2): Flywheel Turn 2 — Findings

**Status: exploratory, not TDD.** Turn 2's question was: how does ft-v1 do on genuinely new footage, and how much does a retrained model improve on both holdouts? The answer turned out to be a three-layer diagnosis story — the new footage broke ft-v1 badly enough that most of the turn went to finding out *why* and building a cold-start path around it (motion bootstrap), before any retraining could use the new data at all. Numbers are reported as measured; nothing was tuned to look better.

## New footage inventory (and duplicate check)

Six new clips recorded 2026-07-16 in `/Users/andrew.follmann/personal-projects/juggling/data/raw/20260716/`, all portrait 1080×1920 phone footage in a NEW outdoor environment (foliage background, dark balls, dark shirt — turn 1's corpus was indoor):

| Clip (PXL_20260716_…) | Frames (container) | fps | Duration | Session batch |
|---|---|---|---|---|
| 182734164 | 1,500 | 23.98 | 55.3 s | 1 |
| 182847912 | 506 | 30 | 16.8 s | 1 |
| 182922676 | 683 | 30 | 24.1 s | 1 |
| 183035587 | 1,225 | 23.98 | 42.0 s | 1 |
| 191622045 | 753 | 24 | 31.3 s | 2 |
| **191702756 (NEW-ENV HOLDOUT)** | 353 | 24.08 | 14.7 s | 2 |

- **Duplicate check:** the same directory also contains `PXL_20251226_195820315.mp4` (114,752,863 bytes, dated Dec 26 2025). The brief asked for a `cmp -s` against `data/raw/PXL_20251226_195820315.mp4`, but that original path no longer exists — this is now the only raw copy of that filename on disk. Filename + byte-for-byte size + date match the turn-1 training video ("PXL…315.mp4", 150 images / 254 boxes, train split), so it was treated as the same file relocated, **excluded from the new set** (training-leakage risk). The "six new clips" are exactly the `PXL_20260716_*` files.
- `191702756` was designated the **new-environment holdout** before any labeling: it never contributes labels to any training corpus, in any form.

## ft-v1 on the new footage: coverage held up unevenly, events collapsed to zero

`juggletrack analyze --model ft-v1 --stride 2` on all six clips: **0 runs, 0 drops, ≤1 arc on every clip.** If the user recorded deliberate drops, none were recovered — no event data of any kind came through. Coverage (same flags, stock vs ft-v1):

| Clip | stock ≥1 / ≥2 / ≥3 | ft-v1 ≥1 / ≥2 / ≥3 |
|---|---|---|
| 182734164 | 41.6% / 5.3% / 0.0% | 97.2% / 56.9% / 38.5% |
| 182847912 | 0.0% / 0.0% / 0.0% | 83.4% / 73.1% / 55.3% |
| 182922676 | 0.6% / 0.0% / 0.0% | 39.8% / 13.5% / 5.3% |
| 183035587 | 0.8% / 0.0% / 0.0% | 7.2% / 3.6% / 1.8% |
| 191622045 | 9.0% / 0.3% / 0.0% | 99.2% / 96.6% / 76.9% |
| 191702756 (holdout) | 4.0% / 0.0% / 0.0% | 74.6% / 53.1% / 29.4% |

Stock YOLO11n is nearly blind here (0–9% ≥1 on five of six clips — far below its 44–57% ≥2 on turn-1's environment), confirming this is genuinely out-of-domain footage. ft-v1's coverage looks respectable on some clips — but the coverage numbers on this environment are **inflated by static false positives** (see next section), and frame inspection confirmed the true-positive story is much worse: the dark balls against foliage and a dark shirt are frequently missed outright.

## Three-layer diagnosis: why zero events

1. **First hypothesis (refuted): linker distance.** The initial FN split on 182734164/191622045 showed detector-side coverage was fine-to-excellent (56.9% / 96.6% of sampled frames had ≥2 detections) yet 100% of detections failed arc assignment — so the extractor looked like the culprit, and the fixed `link_max_dist=0.08` (tuned on turn-1 footage) was the named suspect. The sweep below tested that exhaustively: **zero arcs at every linker setting from 0.08 to 0.8.** Link distance was not the cause.
2. **Second layer (fixed): persistent static false positives.** Manual pipeline stepping showed the linker *was* forming large fragments — out of near-static high-confidence false positives (e.g. one point at (0.97, 0.72) present in 711 of 182734164's 1,510 detections, spanning ~24 s with fitted ay≈0.000 — a background object the detector consistently misreads as a ball). These static clusters win the greedy nearest-neighbor competition and poison extraction. Fix: `filter_static_detections` pre-filter in `extract_arcs` (commit `fe5ba27`) — bins detections into 0.03-normalized cells and drops cells continuously occupied >1.5 s (run-chaining, so periodically-revisited real-ball cells survive). Removes 85.3% / 61.3% of detections on the two diagnosis clips, exactly the flagged clusters.
3. **Third layer (remaining): genuinely low true-positive recall.** Even after static filtering, both clips still extract 0 arcs — the surviving real-ball detections are too sparse/inconsistent to satisfy min-points/duration/gravity gates. Frame inspection confirmed: dark balls against foliage and a dark shirt are simply hard for ft-v1 (trained on indoor footage). Much of ft-v1's raw coverage on 182734164/191622045 was the static FPs themselves. This is a detector-recall problem no extractor tuning can fix — it needs training data from this environment, which is a chicken-and-egg problem when the auto-labeler needs arcs to accept labels.

## Linker sweep

**Context.** Turn 2's FN diagnosis (`.superpowers/sdd/turn2-report.md` step 3) found `extract_arcs` returns **zero arcs** on both `PXL_20260716_182734164` and `PXL_20260716_191622045` despite good-to-excellent detector coverage (≥2 dets on 56.9% / 96.6% of sampled frames respectively), and hypothesized the fixed `link_max_dist=0.08` — tuned on turn-1's environment — is too tight for this closer-framed portrait footage. Commit 1 of this task plumbed `link_max_dt`/`link_max_dist`/`em_iters` through `AnalyzeConfig` and the CLI so that hypothesis could actually be tested per-video. This section is the test.

**Method.** For each saved `detections.jsonl`, ran `extract_arcs(dets, link_max_dist=d, link_max_dt=t)` for `d ∈ {0.08, 0.12, 0.16, 0.20, 0.25}` × `t ∈ {0.12, 0.25}` (all other params at `extract_arcs` defaults: `resid_tol=0.02`, `min_points=6`, `min_duration=0.15`, `g_range=(0.5, 8.0)`, `em_iters=5`). For each combo, `assign_detections(dets, arcs)` (default `resid_tol=0.02`) counts total assigned detections.

### Sweep table

**PXL_20260716_182734164** (1,510 detections, 750 sampled frames, stride 2):

| link_max_dist | link_max_dt | arcs | assigned dets | median ay | throws | catches |
|---|---|---|---|---|---|---|
| 0.08 | 0.12 | 0 | 0 | — | 0 | 0 |
| 0.08 | 0.25 | 0 | 0 | — | 0 | 0 |
| 0.12 | 0.12 | 0 | 0 | — | 0 | 0 |
| 0.12 | 0.25 | 0 | 0 | — | 0 | 0 |
| 0.16 | 0.12 | 0 | 0 | — | 0 | 0 |
| 0.16 | 0.25 | 0 | 0 | — | 0 | 0 |
| 0.20 | 0.12 | 0 | 0 | — | 0 | 0 |
| 0.20 | 0.25 | 0 | 0 | — | 0 | 0 |
| 0.25 | 0.12 | 0 | 0 | — | 0 | 0 |
| 0.25 | 0.25 | 0 | 0 | — | 0 | 0 |

**PXL_20260716_191622045** (1,412 detections, 377 sampled frames, stride 2):

| link_max_dist | link_max_dt | arcs | assigned dets | median ay | throws | catches |
|---|---|---|---|---|---|---|
| 0.08 | 0.12 | 0 | 0 | — | 0 | 0 |
| 0.08 | 0.25 | 0 | 0 | — | 0 | 0 |
| 0.12 | 0.12 | 0 | 0 | — | 0 | 0 |
| 0.12 | 0.25 | 0 | 0 | — | 0 | 0 |
| 0.16 | 0.12 | 0 | 0 | — | 0 | 0 |
| 0.16 | 0.25 | 0 | 0 | — | 0 | 0 |
| 0.20 | 0.12 | 0 | 0 | — | 0 | 0 |
| 0.20 | 0.25 | 0 | 0 | — | 0 | 0 |
| 0.25 | 0.12 | 0 | 0 | — | 0 | 0 |
| 0.25 | 0.25 | 0 | 0 | — | 0 | 0 |

**Zero arcs at every point in the requested grid, on both videos.** No plateau to describe — the numbers never move off zero, so there is no "median ay" or event-derivation count to report for a "chosen setting" from this grid.

**Extended check (beyond the requested grid, for root-cause context).** Pushed `link_max_dist` further to see whether *any* width recovers substantial arcs: at `dist ∈ {0.3, 0.4, 0.5}` (both `dt` values) — still 0 arcs on both videos. At `dist=0.6, dt=0.12`: 1 arc on video 1. At `dist=0.4, dt=0.25`: 1 arc on video 2. At `dist=0.8, dt=0.12`: still just 1 arc on video 1. So even widening the linker by 10x over the default never recovers more than a single arc on either video — nowhere near "substantial."

**Root cause is not (solely) linking distance.** Manually stepping through `_link_fragments` → `_split_ballistic` → initial `fit_arc` on video 1 shows the linker *does* form large fragments even at the default `link_max_dist=0.08` (sizes up to 324, 299, 277 points) — contradicting the "never links" phrasing of the original hypothesis. The problem is what they link into: these large fragments are near-static high-confidence false positives (fitted `ay ≈ 0.000`, spanning 20–24 seconds) — almost certainly a background object the detector consistently mistakes for a ball. Widening `link_max_dist`/`link_max_dt` makes this worse, not better: at wider settings the same kind of static-point fragment grows even larger (409, 701 points), because the greedy nearest-neighbor linker has more competing open fragments alive at once and a real ball's next sample can get absorbed into the nearest *static* fragment's prediction instead of its own true (fast-moving) fragment. A handful of genuine gravity-plausible seeds (`ay` in `[0.25, 4.0]`) do appear in the initial-fit stage at wider settings (up to 7 simultaneously, e.g. `ay=3.86, n=4`; `ay=0.30, n=14`), but they are short (4–14 points, sub-1s) and get eliminated by `_prune`/`_gravity_prune` in the EM loop — never more than one survives to the end, and even that one doesn't survive a second EM iteration.

**Recommended setting.** Because no in-grid setting changes the outcome on either video, there is no evidence-based "smallest setting that recovers substantial arcs" to recommend *for these two specific videos*. For the general-purpose knob (informed by the synthetic downsampling scenario in `tests/test_analyze.py::test_config_plumbs_linker_knobs`, where widening `link_max_dist` to 0.2 with `link_max_dt=0.35` did recover a fully-linked cascade that the 0.08 default missed), a moderate default of **`link_max_dist=0.12`, `link_max_dt=0.2`** is a reasonable starting point for footage with wider per-sample displacement than turn-1's environment — enough headroom to link real flight without the extreme over-linking risk seen at 0.4+. But this pair of turn-2 videos needs a different fix before the linker distance matters at all: filtering (or down-weighting) the dominant static/background false-positive cluster before `_link_fragments` runs, so it stops winning the greedy nearest-neighbor competition against genuine ball flight. That was already the extractor-side direction flagged in `turn2-report.md` step 3; this sweep sharpens it from "adaptive linking distance" to specifically "background-point filtering," since linking distance alone — checked exhaustively from 0.08 to 0.8 — does not fix it.

## Motion bootstrap: breaking the chicken-and-egg

The auto-labeler only accepts detections that lie on extracted arcs; with ft-v1's true recall too low to form arcs on this footage, the appearance-based path yields (nearly) zero labels — the first YOLO-label pass produced 0–10 boxes per clip, all below the ≥50-box gate. Commit `aa5fb72` added a cold-start alternative: `juggletrack label --detector motion` (MOG2 background subtraction, static camera required, stride 1). Motion detection needs no appearance model at all, and the same arc-verification gate then filters its (noisy, ~70–130/frame) blobs down to parabolic-flight boxes.

Field results on the two diagnosis clips (from the motion agent's verification, `.superpowers/sdd/turn2-motion-report.md`):

| Clip | Motion boxes | Images | Review frames | Arcs | Throws | Catches | Median ay |
|---|---|---|---|---|---|---|---|
| 182734164 | 3,648 | 745 | 1,455 | 64 | 52 | 34 | 1.246 |
| 191622045 | 2,094 | 424 | 743 | 38 | 27 | 11 | 0.984 |

Both clips went from 0 arcs (ft-v1) to dozens of arcs with gravity-plausible median curvature — the cold-start path works. Wall-clock caveat: motion labeling is far slower than YOLO labeling on cluttered clips (MOG2 emits ~100+ blobs/frame on foliage; `extract_arcs`'s EM/merge passes grind on tens of thousands of points — the 16.8 s clip 182847912 took the longest of any clip this turn).

Motion-labeling the remaining three labelable clips (191702756 stays held out, never labeled):

| Clip | Images | Boxes | Review frames | Wall | ≥50-box gate |
|---|---|---|---|---|---|
| 182847912 | 194 | 698 | 492 | 27.3 min | pass |
| 182922676 | 353 | 1,769 | 673 | 3.6 min | pass |
| 183035587 | 525 | 2,597 | 1,099 | 4.9 min | pass |

Contrast with the YOLO-label pass on the same clips (0 / 9 / 0 boxes — all failed the gate): the motion path recovered **10,806 boxes across five clips** where the appearance path recovered effectively zero. Label dirs: `/Users/andrew.follmann/personal-projects/juggling/outputs/flywheel-t2/labels-motion/<stem>/`.

## Corpus v2b

`assemble_dataset(sources=[af2, pxl, yt-3ball, yt-5easy, 182734164, 182847912, 182922676, 183035587, 191622045], out_dir=datasets/flywheel-v2b, val_fraction=0.2, seed=0)` — nine sources, whole-video split, `n_val = max(1, round(0.2×9)) = 2`; seed 0 selected **182734164 + 182847912 (both new-env clips) as the val sources**.

| Split | Sources | Images | Boxes |
|---|---|---|---|
| train | af2, pxl, yt-3ball, yt-5easy, 182922676, 183035587, 191622045 | 4,026 | 10,708 |
| val | 182734164, 182847912 | 939 | 4,346 |

(Arithmetic: 420+150+42+2,112+353+525+424 = 4,026 images; 692+254+83+3,219+1,769+2,597+2,094 = 10,708 boxes; 745+194 = 939; 3,648+698 = 4,346.) Two structural shifts vs turn 1: yt-3ball (turn-1's val) is now in train, and the val split is 100% new-environment, motion-labeled footage — so checkpoint selection now optimizes for the new domain. 60.3% of training boxes (6,460 = 1,769+2,597+2,094 of 10,708) and 100% of val boxes are motion-derived labels, whose geometry differs from appearance boxes (see caveat 1).

An earlier corpus-v2 attempt (before the motion bootstrap existed) had NO qualifying new-video labels, i.e. it was **identical to corpus v1** — that dataset (`datasets/flywheel-v2`) became the schedule-control experiment below.

## Training: ft-v2-control and ft-v2b

Rationale for shorter schedules: turn-1's 40-epoch run peaked at epoch 10 and declined afterward, so turn 2 planned 15–20-epoch runs. Both trained from BASE `yolo11n.pt` (never from ft-v1, to avoid compounding turn-1 overfit), `--imgsz 640 --device mps`, ultralytics 8.4.96, batch 16, AdamW lr0=0.002 (auto), seed 0.

**ft-v2-control** (= corpus v1 data, 15 epochs — isolates the schedule variable): "15 epochs completed in 0.310 hours" (~18.6 min). Best-by-fitness epoch: **1**. Final best.pt val (42-image yt-3ball split, same as turn 1): P 0.727 / R 0.217 / **mAP50 0.418** / mAP50-95 0.264 — well below ft-v1's 0.507. Weights: `/Users/andrew.follmann/personal-projects/juggling/models/juggletrack-ft-v2-control/best.pt`.

**ft-v2b** (= corpus v2b, 20 epochs): "20 epochs completed in 0.709 hours" (~42.5 min). Best-by-fitness epoch: **14**. Best.pt val (939-image new-env motion-labeled split): P 0.229 / R 0.065 / **mAP50 0.039** / mAP50-95 0.029. The val curve is extremely noisy and never exceeds mAP50 0.042 at any epoch (range 0.0004–0.042) — the model essentially never learns to detect on the new-env val clips to the motion labels' satisfaction. Weights: `/Users/andrew.follmann/personal-projects/juggling/models/juggletrack-ft-v2b/best.pt`; per-epoch metrics in the worktree at `runs/detect/runs/finetune/flywheel-v2b/results.csv` (control: `.../flywheel-v2/results.csv`).

**Read together with turn 1, the control is the most useful datum of the three trainings:** same data as ft-v1, 15 epochs instead of 40 → mAP50 drops 0.507→0.418 and field coverage collapses (see table below). The 40-epoch schedule (with best-checkpoint selection doing the early stopping) mattered more than anticipated; "turn-1 peaked at epoch 10" did NOT mean "15 epochs from a fresh seed reproduces epoch-10 quality" — a fresh run's own best checkpoint landed at epoch 1 (control) and 14 (v2b) with much weaker weights.

## Three-way+ coverage (fresh detection, `--stride 2 --conf 0.05 --imgsz 640`)

Post-filter = detections surviving `filter_static_detections` (the static-clutter gate from commit `fe5ba27`), reported for the holdout rows so static false positives can't inflate the comparison.

**af1.mov — TURN-1 HOLDOUT (original environment), 600 sampled frames:**

| Model | Detections | ≥1 | ≥2 | ≥3 | Median conf | Post-filter dets |
|---|---|---|---|---|---|---|
| stock | 785 | 64.0% | 44.2% | 18.3% | 0.164 | 785 (−0.0%) |
| **ft-v1** | 3,510 | **99.7%** | **95.8%** | **92.3%** | 0.127 | 1,400 (−60.1%) |
| ft-v2-control | 984 | 69.3% | 57.5% | 34.3% | 0.181 | 984 (−0.0%) |
| ft-v2b | 1,078 | 70.3% | 55.8% | 34.2% | 0.212 | 1,078 (−0.0%) |

**PXL_20260716_191702756 — NEW-ENV HOLDOUT (never labeled anywhere), 177 sampled frames:**

| Model | Detections | ≥1 | ≥2 | ≥3 | Median conf | Post-filter dets |
|---|---|---|---|---|---|---|
| stock | 7 | 4.0% | 0.0% | 0.0% | 0.207 | 7 (−0.0%) |
| **ft-v1** | 328 | **74.6%** | **53.1%** | **29.4%** | 0.077 | 328 (−0.0%) |
| ft-v2-control | 15 | 7.9% | 0.6% | 0.0% | 0.094 | 15 (−0.0%) |
| ft-v2b | 91 | 31.1% | 7.9% | 4.0% | 0.111 | 91 (−0.0%) |

**af2.mp4 (train source for all ft models), 536 sampled frames:**

| Model | Detections | ≥1 | ≥2 | ≥3 |
|---|---|---|---|---|
| stock | 982 | 75.6% | 56.9% | 33.2% |
| ft-v1 | 2,375 | 98.9% | 92.5% | 86.9% |
| ft-v2b | 1,318 | 89.0% | 75.2% | 45.1% |

**PXL_20260716_182734164 (motion-labeled; VAL source for ft-v2b — seed 0 put it in val, so it never shaped v2b gradients), 750 sampled frames:**

| Model | Detections | ≥1 | ≥2 | ≥3 |
|---|---|---|---|---|
| stock | 352 | 41.6% | 5.3% | 0.0% |
| ft-v1 | 1,510 | 97.2% | 56.9% | 38.5% |
| ft-v2b | 1,043 | 87.7% | 48.3% | 2.9% |

**Honest reading.**
- **ft-v1 wins every row.** Neither new model comes close on either holdout. Turn 2 did not produce a better detector; it produced a labeling path (motion bootstrap) plus strong evidence about what went wrong.
- **The post-filter column needs care in BOTH directions.** On the new-env holdout, no model's detections are static-flagged — so ft-v1's 53.1% ≥2 there is not a static-FP artifact (the static-FP inflation lives in 182734164/191622045, which are not in this table's holdout rows). On af1, ft-v1 loses 60.1% to the filter while the weak models lose 0% — but inspection of turn-1 overlays suggests much of that 60% is *real held/idle balls* (a ball resting in a hand or on the floor between runs occupies one cell continuously for >1.5 s, exactly the filter's static signature). The removal% on af1 correlates with recall on stationary balls, not with false-positive rate. Post-filter counts are a clutter-robustness signal on cluttered footage, not a precision measurement.
- **Schedule dominates data.** ft-v2-control (old data) and ft-v2b (old data + 10.8k new-env motion boxes) land within ~2 pp of each other on af1 — both ~40 pp below ft-v1. The added new-env data did help where expected (new-env holdout: 0.6%→7.9% ≥2 control→v2b; 182734164: v2b at 48.3% ≥2 approaches ft-v1's 56.9% despite that clip being val-only for v2b) but the recipe regression swamps everything.

## Event-level results on the holdouts (ft-v2b, stride 1, overlays rendered)

| | af1 stock (Plan 2) | af1 ft-v1 (turn 1) | af1 ft-v2b | 191702756 ft-v1 (stride 2) | 191702756 ft-v2b |
|---|---|---|---|---|---|
| Detections | 1,529 | 7,025 | 2,144 | 328 | 189 |
| Arcs | 60 | 58 | 60 | 1 | 7 |
| Runs | 3 | 3 | 3 | 0 | 0 |
| Total catches | 39 | 55 | 42 | 0 | 0 |
| Drops | 2 | 0 | 1 | 0 | 0 |

- **af1 / ft-v2b:** runs 2.15–3.76 s (2 catches), 8.54–10.73 s (17), 23.25–24.18 s (23); 1 drop at t=36.09 s (signals: floor_descent + periodicity_collapse). Same run structure as prior models; catch totals sit between stock and ft-v1, consistent with its in-between detection density.
- **191702756 / ft-v2b:** 189 detections, 7 arcs — but 0 runs, 0 drops. Still no event-level output on the new-environment holdout under any model this turn (ft-v1 also produced 0 runs there). **No drop data was recovered from any of the six new clips by any appearance model** — if the user recorded deliberate drops, the only pipeline that currently sees them is the motion-detector path (182734164 motion analysis found 64 arcs / 34 catches; drop analysis of motion output wasn't part of this turn's labeling runs).
- Overlays for eyeballing: `/Users/andrew.follmann/personal-projects/juggling/outputs/flywheel-t2/analyze-ftv2b/af1/overlay.mp4` and `/Users/andrew.follmann/personal-projects/juggling/outputs/flywheel-t2/analyze-ftv2b/PXL_20260716_191702756/overlay.mp4`.

## Caveats

1. **ft-v2b's val metric and field coverage disagree about *how bad* it is, and both may mislead.** Val mAP50 0.039 says "useless"; field coverage says "weaker than ft-v1 but far from useless" (87.7% ≥1 on 182734164). Likely reconciliation: motion-derived val boxes (blob geometry, includes blur streaks) punish an appearance model's tight boxes at IoU≥0.5 even when the ball is found. Motion-label geometry vs appearance-label geometry is an unquantified mismatch running through all v2b numbers.
2. **Motion labels are entirely unaudited.** 10,806 boxes passed the arc gate, but no human looked at them. The gate verifies parabolic *position*; it does not verify box *extent* (blur streaks pass), nor does it catch arc-shaped non-ball motion (birds, swinging foliage).
3. **The schedule/recipe confound was only partially controlled.** ft-v2-control isolates schedule-on-old-data (15 vs 40 epochs: large regression), but no 40-epoch run on corpus v2b exists — so "would the new data help under the v1 recipe?" is unanswered. That is the single most informative next training run.
4. **VFR / frame-count metadata.** These phone clips are variable-frame-rate; `CAP_PROP_FRAME_COUNT` for 191622045 reads 753 (24.0 fps, 31.3 s), which ffprobe corroborates (`nb_frames=753`, duration×fps ≈ 753), but the motion agent flagged the count as suspicious relative to expectations. Timestamps derived as `frame_idx/fps` on VFR footage carry small errors that feed arc fitting; coverage denominators for this clip depend on that count. Unresolved; worth a decode-and-count pass if this clip's numbers are ever load-bearing.
5. **The duplicate-check was indirect.** The original `data/raw/PXL_20251226_195820315.mp4` no longer exists on disk, so identity was established by filename + exact byte size + date against turn-1 records, not `cmp -s`. The file was excluded from the new set either way.
6. **stock-af1 analyze runs now differ from Plan 2 baselines** (4 runs / 1 drop at stride 2 here vs 3 runs / 2 drops at stride 1 in Plan 2) — stride and the new static pre-filter both moved event outputs; cross-turn event comparisons should hold stride constant (the event table above uses stride-1 numbers for af1).

## Next steps

1. **Retrain corpus v2b with the v1 recipe** (40 epochs from base, best-checkpoint selection) — the schedule-dominance finding predicts this recovers most of the gap; if it also lifts the new-env holdout above ft-v1's 53.1% ≥2, the flywheel's data-diversity thesis survives turn 2.
2. **Audit motion-label geometry** (sample ~100 boxes/clip): measure box-tightness vs appearance labels; consider shrinking motion boxes toward blob centroids or re-deriving extent from the appearance model's own high-confidence detections on the same frames (label distillation).
3. **Mixed val split** (one old-env + one new-env source) so checkpoint selection balances domains instead of optimizing one.
4. **Drop analysis on motion-path output** for the new clips — the user may have recorded deliberate drops; the motion pipeline is currently the only one that can see them (64/38 arcs on the two diagnosis clips).
5. **Extractor throughput** on dense motion input (27 min for a 17 s clip is the current worst case) — `_merge_pass` is O(arcs²) with a Python-level `fit_arc` per candidate union; vectorizing or pre-bucketing by time would make motion labeling routine.
6. Turn 1's carry-forward items that remain open: review-UI triage of flagged frames (motion labeling added 4,462 more: 1,455+743+492+673+1,099), human event labels for af1/af2, external diverse data.

## Artifacts (not committed)

- Motion label corpora: `/Users/andrew.follmann/personal-projects/juggling/outputs/flywheel-t2/labels-motion/{182734164,182847912,182922676,183035587,191622045}/`
- YOLO-label attempt (all below gate): `/Users/andrew.follmann/personal-projects/juggling/outputs/flywheel-t2/labels/<stem>/`
- ft-v1 analyze outputs on new clips: `/Users/andrew.follmann/personal-projects/juggling/outputs/flywheel-t2/analyze-ftv1/<stem>/`
- ft-v2b holdout analyses + overlays: `/Users/andrew.follmann/personal-projects/juggling/outputs/flywheel-t2/analyze-ftv2b/{af1,PXL_20260716_191702756}/`
- Datasets: `/Users/andrew.follmann/personal-projects/juggling/datasets/flywheel-v2/` (control) and `/Users/andrew.follmann/personal-projects/juggling/datasets/flywheel-v2b/`
- Weights: `/Users/andrew.follmann/personal-projects/juggling/models/juggletrack-ft-v2-control/best.pt`, `/Users/andrew.follmann/personal-projects/juggling/models/juggletrack-ft-v2b/best.pt`
- Execution notes: `.superpowers/sdd/turn2-report.md`, `.superpowers/sdd/turn2-staticfilter-report.md`, `.superpowers/sdd/turn2-motion-report.md` (worktree)
