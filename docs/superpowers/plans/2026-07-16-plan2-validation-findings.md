# Plan 2 Task 9: Real-Video Validation Findings (stock YOLO baseline)

**Status: exploratory, not TDD.** This document reports what happened when the finished offline pipeline (`juggletrack analyze`) was run against the user's own footage for the first time, using stock COCO-pretrained YOLO11n at conf 0.05 — no fine-tuning, no event-core tuning. Low numbers below are the expected, useful result: they are the datum that motivates Plan 3's labeling + fine-tuning work. Nothing in the event core was changed to make these numbers look better.

## Setup

- **Weights:** `/Users/andrew.follmann/personal-projects/juggling/yolo11n.pt` (stock COCO YOLO11n, "sports ball" class 32) — not fine-tuned.
- **Detector config:** `conf=0.05` (CLI default), `imgsz=640` (default; `960` for the sensitivity run), `device=None` → ultralytics auto-selected MPS.
- **Machine:** Apple M4 Pro, macOS 26.5.2, Python 3.11 (venv), 14 CPU cores.
- **Stride:** `1` for the three short/close videos (af1.mov, af2.mp4, PXL...). For the two long YouTube tutorials, the brief's pre-run planning heuristic (~0.2–0.3s/frame) projected 60 and 102 minutes of processing respectively for their 14,471 and 24,470 frames — both over the ~15-minute budget — so both ran with `--stride 2`. In practice, actual measured throughput on this machine turned out to be roughly 10x faster than that heuristic (~0.02–0.03s/frame observed on the short videos), so stride 2 was more conservative than strictly necessary in hindsight. It was still the correct call given the information available *before* running, so it stands; it's noted here for honesty about the coverage numbers below being computed over half the frames for those two videos.
- **Command shape** (per video):
  ```
  uv run juggletrack analyze <video> --out <outdir> --model yolo11n.pt --save-intermediates [--stride 2] [--imgsz 960]
  ```
- **Outputs:** `/Users/andrew.follmann/personal-projects/juggling/outputs/plan2-validation/<name>/` (not committed; gitignored).
- **Failures:** 2 of the 6 `analyze` invocations run today crashed with an unhandled `LinAlgError` (see Qualitative failure modes #5) rather than completing. Per the brief, these are recorded and the rest of the run continued.

## Per-video table

| Video | Duration (frames@fps) | Resolution | Stride | Wall time | Detections | Coverage ≥1 / ≥2 / ≥3 | Median conf | Median w | Arcs | Runs | Catches | Drops | End reasons |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| af1.mov | 40.0s (1199f@30.00) | 1620×1080 | 1 | 29.6s | 1529 | 64.0% / 42.5% / 17.3% | 0.162 | 0.032 | 60 | 3 | 39 | 2 | stop×3 |
| af2.mp4 | 35.6s (1071f@30.05) | 1080×1920 | 1 | 33.3s | 2000 | 75.8% / 56.6% / 34.6% | 0.219 | 0.045 | 48 | 3 | 29 | 2 | stop×3 |
| PXL_20251226_195820315.mp4 | 17.5s (525f@30.05) | 1080×1920 | 1 | 18.9s | 1002 | 82.1% / 48.8% / 30.3% | 0.289 | 0.066 | 17 | 2 | 3 | 2 | stop×2 |
| YTDown 3-Ball tutorial | 482.8s (14471f@29.97) | 854×480 | 2 | 2m 16.5s | 7502 | 48.6% / 26.1% / 13.8% | 0.202 | 0.033 | 35 | 2 | 9 | 0 | stop×2 |
| YTDown 5-Easy tutorial | 816.5s (24470f@29.97) | 854×480 | 2 | **CRASHED** at 3m 19s (post-detection) | — | — | — | — | — | — | — | — |

**Totals across the 4 completed videos:** 10 runs, 80 catches, 6 drops, 160 arcs, 11,033 raw detections. All 10 completed runs ended with `end_reason="stop"` — none hit `video_end` or an explicit drop-terminus classification in this batch.

Coverage ≥2 balls, the primary "is this usable" signal: **af2.mp4 56.6%** (best) → PXL 48.8% → af1.mov 42.5% → YTDown 3-Ball 26.1% (worst, and it's also the lowest-resolution, most-distant framing).

## Sensitivity: imgsz 640 vs 960

The brief calls for re-running the single best-covered video (af2.mp4, 56.6% coverage ≥2) at `--imgsz 960`.

- **af2.mp4 at imgsz 960 crashed** with the identical `LinAlgError` described below — same failure as the yt-5easy video, on an entirely different, previously-clean video. No coverage data is available for af2 at 960.
- To still get a working before/after comparison, the second-best-covered video (PXL, 48.8% coverage ≥2) was substituted:

| | imgsz 640 | imgsz 960 | Δ |
|---|---|---|---|
| Detections | 1002 | 979 | −2.3% |
| Coverage ≥1 | 82.1% | 81.0% | −1.1pp |
| Coverage ≥2 | 48.8% | 51.4% | +2.6pp |
| Coverage ≥3 | 30.3% | 28.6% | −1.7pp |
| Median confidence | 0.289 | 0.317 | +0.028 |
| Median box width (norm.) | 0.066 | 0.059 | −0.007 |
| Catches | 3 | 5 | +2 |
| Wall time | 18.9s | 23.4s | +24% |

**Reading:** the deltas are noise-level in both directions — no coverage threshold moved by more than ~3 percentage points, and the direction isn't even consistent across ≥1/≥2/≥3. Confidence rose marginally. This is a clean negative result: **doubling the input resolution area (640→960) does not meaningfully change how often the stock detector finds the ball.** Combined with af2's crash under the same treatment, this sensitivity check ended up costing more than it returned, but the direction of the (small, real) result is informative for Plan 3 — see Implications below.

## Qualitative failure modes

1. **Coverage ranges from poor to mediocre, never good.** The best video (af2.mp4, closely filmed, high-res, vertical) tops out at 56.6% of frames having ≥2 simultaneous ball detections — meaning arcs have real gaps even in the best case. The worst production video (YTDown 3-Ball tutorial, filmed at a distance, 480p) manages only 26.1%.
2. **Confidence is low and marginal across the board.** Medians cluster at 0.16–0.32 — well below the "confident detection" range — meaning most kept detections are right at the edge of the 0.05 threshold and would appear/disappear with small threshold changes. This is consistent with a model that's guessing "sports ball"-shaped blobs rather than recognizing a juggling ball specifically.
3. **Arc fragmentation is sometimes the tighter bottleneck, not raw coverage.** PXL had *higher* ≥1 coverage than af2 (82.1% vs 75.8%) but only 17 arcs and 3 catches, versus af2's 48 arcs and 29 catches. Isolated single-frame detections that don't chain into multi-point trajectories don't produce usable arcs — coverage ≥1 alone overstates usability; coverage ≥2/≥3 and arc count are the more honest signals.
4. **Every completed run ended `stop`, none `video_end` or drop-terminated.** A mild positive signal for the catch/end-reason classification logic itself (Plan 1/2's carry-forward fix): every observed sequence's last arc was witnessed crossing the hand line within tolerance, not an extrapolation artifact. This is orthogonal to detection quality and suggests the event core's ending logic is behaving as designed on real (if sparse) data.
5. **Critical robustness gap: the event core crashes on dense/messy real detections.** Two of six `analyze` runs today died with an unhandled `LinAlgError: SVD did not converge in Linear Least Squares`, raised from the weighted `np.polyfit` call in `src/juggletrack/arcs/fit.py:24` (`fit_arc`), invoked from the EM refit step in `arcs/extract.py`'s `_em_assign_refit`. It hit two unrelated videos — the longest tutorial (yt-5easy, most total detections) and the *best-performing* video re-run at higher resolution (af2 at imgsz 960, also more detections than at 640) — which points to detection **density**, not any one video's content, as the trigger (most likely a candidate point cluster with near-duplicate timestamps producing a zero-variance column in the weighted design matrix, seen as `RuntimeWarning: invalid value encountered in divide` immediately before each crash). This is not a "few detections" finding, it's an uncaught exception that kills the whole run partway through, after all detection work is done and before any output is written (no `analysis.json`/`detections.jsonl`/overlay for either failed run).

## Implications for Plan 3

- **Fine-tuning need: confirmed, and significant.** Even the single best, closely-filmed video only reaches 56.6% coverage ≥2; three of five videos are at or below 49%. The stock model frequently fails to see thrown juggling balls at all. This is exactly the result Plan 3 exists to fix.
- **Resolution is not the lever.** The imgsz 640→960 sensitivity check produced no meaningful coverage change (see above). Don't spend Plan 3 budget on "just detect at higher resolution" — the model doesn't recognize juggling-ball-shaped objects well at any input size tested; that's a training-data problem, not a spatial-resolution problem.
- **ROI cropping is a plausible cheaper lever, worth testing.** Coverage was reliably higher on the closely-framed, higher-resolution personal videos (af1, af2, PXL — all vertical 1080×1920 or 1620×1080, presumably filmed close to the juggler) than on the wide, distant, 480p tutorial video (yt-3ball, worst coverage among completed runs). Framing distance and resolution are confounded in this dataset, but the pattern is consistent with "a small apparent ball size hurts a nano detector" — cropping to the juggler's region before running detection could be a cheap complement to fine-tuning, worth a quick experiment before committing to a full labeling pass.
- **Fix the arc-fitting crash before scaling up eval or fine-tuning work.** A fine-tuned detector will produce *more* detections per frame, which — per the density-triggered pattern observed today — makes the `LinAlgError` crash *more* likely to recur, not less. Any Plan 3 eval run against a denser/better model risks silently dying partway through unless `arcs/fit.py`'s `fit_arc` (or its caller in `arcs/extract.py`) is hardened against degenerate point sets (e.g., catch `LinAlgError` and drop the candidate arc, or guard against near-duplicate `dt` values before calling `polyfit`). This is a bug independent of fine-tuning and should be prioritized ahead of or alongside Plan 3's first real eval run.
- **Infrastructure is ready.** Every completed run exercised `--save-intermediates`, `--stride`, and `--imgsz` correctly; the eval harness (Task 7) and CLI (Task 8) need no changes to work with a fine-tuned model's weights once Plan 3 produces them.

## Overlay videos (for human eyeballing)

- `/Users/andrew.follmann/personal-projects/juggling/outputs/plan2-validation/af1/overlay.mp4`
- `/Users/andrew.follmann/personal-projects/juggling/outputs/plan2-validation/af2/overlay.mp4`
- `/Users/andrew.follmann/personal-projects/juggling/outputs/plan2-validation/pxl/overlay.mp4`
- `/Users/andrew.follmann/personal-projects/juggling/outputs/plan2-validation/yt-3ball/overlay.mp4`
- `/Users/andrew.follmann/personal-projects/juggling/outputs/plan2-validation/pxl-imgsz960/overlay.mp4` (sensitivity re-run)
- yt-5easy and af2-imgsz960 have no overlay — both crashed before overlay rendering.

Each directory also has `detections.jsonl` (raw per-frame boxes) and `analysis.json` (the full `SessionResult`) for anyone who wants to re-derive different stats without re-running detection.
