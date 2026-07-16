# Plan 3 Task 7: First Flywheel Turn — Findings

**Status: exploratory, not TDD.** One full turn of the data flywheel: auto-label own footage with the arc-verification gate → assemble a whole-video-split dataset (af1.mov held out entirely) → fine-tune YOLO11n → measure detection coverage before/after. Numbers below are reported as measured; nothing was tuned to look better. One real bug was found and fixed mid-measurement (see Coverage section).

## External harvest outcome

The best-effort Creative Commons harvest (`uvx yt-dlp "ytsearch30:3 ball juggling cascade"` with `license~='(?i)creative commons' & duration<240`, time-boxed 10 min) checked all 30 search results and **0 of 30 passed the license filter** — every candidate is Standard YouTube License. The run took 13 s (metadata-only checks), exit 0. No external videos entered the corpus; the only artifact is the playlist metadata JSON at `/Users/andrew.follmann/personal-projects/juggling/data/raw/external/3 ball juggling cascade.info.json`. This is the anticipated "legitimate zero" outcome; diverse external data remains the top need for turn 2 (see Next turn).

## Corpus

Auto-labeling used `juggletrack label` (arc-verified selection: a detection becomes a label only if it lies on an extracted parabolic arc; frames with unverified detections go to the review manifest). Plan 2's saved `detections.jsonl` were replayed for af1/af2/pxl/yt-3ball; yt-5easy had no saved jsonl and ran fresh detection (`yolo11n.pt --stride 2`, 3 m 33 s wall).

| Video | Detection source | Images | Boxes | Review frames | In training set? |
|---|---|---|---|---|---|
| af2.mp4 | Plan 2 jsonl (stride 1) | 420 | 692 | 689 | yes (train) |
| af1.mov | Plan 2 jsonl (stride 1) | 604 | 815 | 534 | **NO — HELD OUT** |
| PXL_20251226_195820315.mp4 | Plan 2 jsonl (stride 1) | 150 | 254 | 366 | yes (train) |
| yt-3ball tutorial | Plan 2 jsonl (stride 2) | 42 | 83 | 3,493 | yes (val) |
| yt-5easy tutorial | fresh yolo11n, stride 2 | 2,112 | 3,219 | 11,704 | yes (train) |
| **Total** | | **3,328** | **5,063** | **16,786** | |

- **Exclusions:** none — every video cleared the ≥50-box sanity gate (minimum: yt-3ball at 83). No external videos existed to spot-check.
- Label dirs: `/Users/andrew.follmann/personal-projects/juggling/outputs/plan3-flywheel/labels/<stem>/` (images + COCO `annotations.json` + `review_manifest.json`).
- Note the review-frame counts: 16,786 flagged frames corpus-wide vs 3,328 accepted images. The arc gate rejects far more frames than it accepts, especially on the two tutorials (yt-5easy: 11,704 flagged vs 2,112 accepted). None of the flagged frames were human-reviewed this turn — the review UI is deferred (see Next turn) — so auto-label precision is unaudited.

## Training

Dataset assembled with `assemble_dataset(sources=[af2, pxl, yt-3ball, yt-5easy], val_fraction=0.2, seed=0)` — af1's labels were never passed in. Whole-video split (spec §6): with 4 sources, `n_val = max(1, round(0.2×4)) = 1`, and seed 0 selected **yt-3ball as the sole val source**.

| Split | Sources | Images | Boxes |
|---|---|---|---|
| train | af2, pxl, yt-5easy | 2,682 | 4,165 |
| val | yt-3ball | 42 | 83 |

(Arithmetic check: 420+150+2,112 = 2,682 images; 692+254+3,219 = 4,165 boxes.)

- **Config:** `juggletrack train datasets/flywheel-v1/data.yaml --model yolo11n.pt --epochs 40 --imgsz 640 --device mps` → ultralytics 8.4.96, torch 2.13.0, Apple M4 Pro (MPS), batch 16, AdamW lr0=0.002 (auto), deterministic seed 0. 168 batches/epoch.
- **Wall time:** launched 12:46:21, "40 epochs completed in 0.832 hours" (~49.9 min; results.csv cumulative time 2,995.4 s agrees). ~55 min total including setup and final validation.
- **Final val metrics (best.pt on the 42-image yt-3ball val split):** P 0.534, R 0.663, **mAP50 0.507**, mAP50-95 0.287.
- **Overfitting signal, reported honestly:** the best epoch was **10** (val mAP50 0.5072). Val mAP50 then declined steadily to 0.259 by epoch 40 — the model kept improving on the three train sources while degrading on the one (very different, 480p wide-shot) val source. best.pt is the epoch-10 checkpoint; the last 30 epochs were wasted or worse. A 10–15-epoch schedule (or early stopping patience tuned down from 100) is indicated for this dataset size.
- Weights: `/Users/andrew.follmann/personal-projects/juggling/models/juggletrack-ft-v1/best.pt` (copied from `runs/detect/runs/finetune/flywheel-v1/weights/best.pt` in the worktree — note ultralytics nested the run dir under `runs/detect/`). Full per-epoch log: `runs/detect/runs/finetune/flywheel-v1/results.csv`.
- **Val-split caveat:** a 42-image single-source val set is statistically thin, and that source (yt-3ball) is the corpus's most atypical video. Val mAP here is a weak model-selection signal, not a generalization estimate.

## Coverage before/after

**Bug found during measurement (and fixed):** the first fine-tuned coverage runs returned **0 detections on every video**. Root cause: `YOLODetector` hardcoded `classes=(32,)` (COCO "sports ball"), which filtered out everything from the fine-tuned single-class model (class 0, "ball"). Fixed in commit `732ee20` (`fix: resolve ball class ids from model names, not hardcoded COCO 32`): the class filter now resolves from the loaded model's own names. All numbers below are post-fix. This is exactly the kind of integration gap a first flywheel turn exists to surface.

Both conditions ran fresh detection with identical settings: `--conf 0.05 --imgsz 640 --stride 2`. Stock = `yolo11n.pt`; fine-tuned = `models/juggletrack-ft-v1/best.pt`. Stock stride-2 numbers reproduce Plan 2's published values almost exactly (yt-3ball identically, since its Plan 2 baseline was also stride 2; the stride-1 → stride-2 shift moves the others ≤1.7 pp).

| Video | Model | Detections | ≥1 | ≥2 | ≥3 | Median conf | Wall |
|---|---|---|---|---|---|---|---|
| **af1.mov (HELD OUT)** | stock | 785 / 600 frames | 64.0% | 44.2% | 18.3% | 0.164 | 14 s |
| **af1.mov (HELD OUT)** | fine-tuned | 3,510 / 600 | **99.7%** | **95.8%** | **92.3%** | 0.127 | 13 s |
| af2.mp4 | stock | 982 / 536 | 75.6% | 56.9% | 33.2% | 0.227 | 13 s |
| af2.mp4 | fine-tuned | 2,375 / 536 | 98.9% | 92.5% | 86.9% | 0.167 | 13 s |
| PXL…315.mp4 | stock | 503 / 263 | 81.7% | 49.4% | 31.2% | 0.286 | 7 s |
| PXL…315.mp4 | fine-tuned | 1,332 / 263 | 99.2% | 92.8% | 81.4% | 0.145 | 7 s |
| yt-3ball (= val source) | stock | 7,502 / 7,236 | 48.6% | 26.1% | 13.8% | 0.202 | 118 s |
| yt-3ball (= val source) | fine-tuned | 34,423 / 7,236 | 98.8% | 93.9% | 81.4% | 0.112 | 115 s |

**Coverage ≥2 deltas:** af1 (held out) +51.6 pp, af2 +35.6 pp, PXL +43.4 pp, yt-3ball +67.8 pp. No usable external video existed to measure (harvest yielded zero). Throughput is symmetric (same architecture): wall times match within a few seconds.

**Read this with the right caveat:** coverage counts *any* detection above conf 0.05 — it is a recall-flavored proxy with no precision component. The fine-tuned model's median confidence is *lower* than stock's on every video (0.112–0.167 vs 0.164–0.286), and detection volume is 2.4–4.6× higher, so some of the jump could be low-confidence false positives. The overlays (paths below) are the precision eyeball-check. That said, the ≥2/≥3 thresholds require multiple simultaneous boxes on the same frame — harder to satisfy by noise alone — and the ≥3 jumps (18.3%→92.3% on held-out af1) are the strongest evidence the model is genuinely seeing the balls it used to miss.

## Event-level deltas (af1 + af2, full `analyze` with fine-tuned weights)

Baselines read directly from Plan 2 artifacts (`outputs/plan2-validation/{af1,af2}/analysis.json`, stride 1, stock). Fine-tuned runs: stride 1, same conf/imgsz, `--save-intermediates`, outputs under `/Users/andrew.follmann/personal-projects/juggling/outputs/plan3-flywheel/analyze-ft/<stem>/`.

| | af1 stock (Plan 2) | af1 fine-tuned | af2 stock (Plan 2) | af2 fine-tuned |
|---|---|---|---|---|
| Detections | 1,529 | 7,025 | 2,000 | 4,740 |
| Arcs | 60 | 58 | 48 | 57 |
| Runs | 3 | 3 | 3 | 2 |
| Total catches | 39 | 55 | 29 | 53 |
| Drops | 2 | 0 | 2 | 0 |
| End reasons | stop×3 | stop×3 | stop×3 | stop×2 |

Run-level detail:

- **af1 ft:** 1.76–3.82 s (3 catches), 8.48–18.18 s (24), 23.53–24.32 s (28). The middle run now spans 9.7 s vs the baseline's 1.0 s fragment (8.52–9.53 s) — denser detections let the run survive gaps that previously broke it.
- **af2 ft:** 6.43–7.81 s (9), 15.35–25.01 s (44). The baseline's runs 2+3 (16.55–20.86 s, 27.64–28.75 s) are restructured: run 2 extends much longer; no run covers the baseline's third segment as a separate run.

**Honest reading:** catches +16 (af1) and +24 (af2), drops 2→0 on both. There are **no human ground-truth event labels for these videos**, so these deltas demonstrate that better detection substantially changes event output — they do not prove the new counts are *correct*. In particular, the two baseline drops per video and the fine-tuned zero can't be adjudicated without watching the footage. The event-accuracy regression gate (Plan 3 Task 1) runs on simulation; real-video event validation needs labeled ground truth, which is a review-UI deliverable.

## Caveats

1. **On-domain, not generalization.** af1 was excluded from training, but it shares the room, lighting, and balls with af2 (train source). Its +51.6 pp jump measures on-domain improvement. The vdrumsta lesson stands: training on own-footage auto-labels risks environment overfitting, and the epoch-10-peak/epoch-40-decline val curve is consistent with exactly that. No truly out-of-domain test video exists in this corpus.
2. **yt-3ball is not held out either.** It contributed the val split, which selected the best checkpoint — its +67.8 pp is partially "seen" through model selection (and 42 of its frames' labels shaped no gradients but did pick the epoch).
3. **Auto-label precision is unaudited.** 16,786 review-flagged frames were never human-checked; the 5,063 accepted boxes rest entirely on the parabola gate. Label noise in training data is plausible, especially from the stride-2 tutorial footage.
4. **Stride-2 label sparsity for tutorials.** yt-5easy (fresh) and yt-3ball (replayed jsonl) were detected at stride 2, so their labels sample every other frame, and yt-5easy contributed 79% of training images — the dataset skews heavily toward one long tutorial's visual style.
5. **Coverage is not precision.** See Coverage section: lower median confidence + 2.4–4.6× detection volume means some coverage gain may be false positives. Eyeball the overlays.
6. **Event deltas are unvalidated.** No ground-truth run/catch/drop labels exist for af1/af2; "more catches, fewer drops" is a change report, not an accuracy claim.
7. **Class-filter bug.** The pre-fix fine-tuned model scored 0.0% across the board (commit `732ee20` fixed it). Any future custom-class model would have hit the same wall; regression tests added in `tests/test_detect.py`.

## Next turn (what the second flywheel turn needs most)

1. **Diverse environments** — the single biggest gap. Own-footage-only training peaked at epoch 10 then overfit. Options, in order: Roboflow juggling datasets (needs the user's API key — not present in this environment), Kinetics juggling clips, and the Meschke dataset (requires a permission email to the author). All were named out-of-scope for this turn.
2. **Review-UI pass on flagged frames.** 16,786 frames await triage; even a sampled audit (say 200 frames) would put a precision number on the auto-labels and correct the worst noise before retraining.
3. **Human event labels for af1/af2** (runs/catches/drops with timestamps) so event-level deltas become accuracy measurements — the `juggletrack eval` harness already consumes them.
4. **Shorter schedule / earlier stopping** — 10–15 epochs, or patience ≪ 100, given the epoch-10 peak; and a multi-source val split once more sources exist (1-source val is both thin and atypical).
5. **Re-run yt-5easy coverage with ft weights** (skipped this turn: not in the brief's measurement list; its stride-2 stock baseline exists in Plan 2's addendum at 89.3% ≥2, already near ceiling and an outlier).

## Artifacts for eyeballing (not committed)

- Fine-tuned overlays: `/Users/andrew.follmann/personal-projects/juggling/outputs/plan3-flywheel/analyze-ft/af1/overlay.mp4`, `/Users/andrew.follmann/personal-projects/juggling/outputs/plan3-flywheel/analyze-ft/af2/overlay.mp4` (compare against Plan 2's `outputs/plan2-validation/{af1,af2}/overlay.mp4`).
- Label corpora: `/Users/andrew.follmann/personal-projects/juggling/outputs/plan3-flywheel/labels/{af1,af2,pxl,yt-3ball,yt-5easy}/`.
- Dataset: `/Users/andrew.follmann/personal-projects/juggling/datasets/flywheel-v1/` (`data.yaml`, images/labels per split).
- Weights + training curves: `/Users/andrew.follmann/personal-projects/juggling/models/juggletrack-ft-v1/best.pt`; per-epoch metrics in the worktree at `runs/detect/runs/finetune/flywheel-v1/results.csv`.
