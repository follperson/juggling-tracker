# Accuracy foundations implementation plan

**Goal:** Make training splits, model selection, and accuracy measurements trustworthy before changing the tracking algorithm.

**Architecture:** Keep the detector → physics → events separation. Fix reproducible failures first, reuse the existing evaluator, and compare saved detections without requiring a GPU or video decoder. Persistent tracking is a separate experiment whose acceptance depends on independently reviewed footage.

**Execution:** Sequential changes in this checkout on `codex/accuracy-foundations`. Preserve the user's existing legacy changes. Add regression tests before behavior changes. Run focused tests after each task, then the complete standard suite and lint. No model training, downloads, or legacy migration is required for the initial batch.

## Evidence and success criteria

- Baseline standard suite: 272 passed, 3 detector tests deselected; lint passes.
- Rebuilding four source images with different split seeds left six files, including two images in both splits.
- `live` defaults to a developer-specific absolute model path; other detector commands default to stock YOLO.
- Current saved-detection replay of `ss3_id_016`: offline 9 runs / 41 catches / 0 drops; direct realtime 9 / 210 / 7. This isolates downstream divergence, not detector quality.
- Generated Meschke reference runs report 157 catches in 4.4 seconds and 42 catches in 3.2 seconds. These are algorithm-derived references, not independent event ground truth.
- The recorded 22-video score at the shipped configuration is 9/26 (34.6%) within one catch when unmatched reference runs remain in the denominator. These videos were also used for tuning.

## 1. Protect training/validation separation

Files: `src/juggletrack/data/dataset.py`, `tests/test_dataset.py`.

- [x] Reproduce reuse of a nonempty output directory in a regression test.
- [x] Require an empty output directory; refuse before writing anything. Preserve previous data rather than silently overwriting it.
- [x] Reject colliding source directory names, which otherwise overwrite image identities or leak the same source across splits.
- [x] Validate split inputs and ensure that a multi-source dataset retains both training and validation sources.
- [x] Verify existing assembly tests plus the new failure cases.

Acceptance: no stale-file leakage, no silent name collisions, and existing datasets remain intact after a rejected rebuild.

## 2. Use one portable inference model default

Files: `src/juggletrack/detect/weights.py`, `src/juggletrack/cli.py`, `tests/test_weights.py`, `tests/test_cli.py`, `README.md`.

- [x] Test explicit model selection, environment override, local champion weights, and missing weights.
- [x] Resolve inference weights in this order: `--model`, `JUGGLETRACK_MODEL`, `models/juggletrack-v3/best.pt` relative to the working directory.
- [x] If the default is absent, explain how to supply weights or explicitly request stock `yolo11n.pt`. Do not silently change models.
- [x] Apply to analyze/live/coverage/YOLO labeling; replay and motion labeling must remain independent of model availability. Keep training base weights and the person-veto model explicitly separate.
- [x] Verify portable invocation and update setup instructions.

Acceptance: the same input commands select the same detector; no machine-specific paths in production defaults.

## 3. Make event benchmarks portable and independent

Files: `src/juggletrack/eval/benchmark.py`, `src/juggletrack/cli.py`, `tests/test_benchmark.py`, `docs/benchmarking.md`.

- [x] Define a versioned manifest with per-video saved detections and independently reviewed event labels, resolved relative to the manifest.
- [x] Reuse `evaluate_session`; aggregate over all labeled runs, including unmatched runs. Report unmatched predictions and drop counts explicitly.
- [x] Store the full analysis configuration, package/schema version, and input hashes with results.
- [x] Reject duplicate video identities, invalid labels, and manifests that identify generated references as reviewed labels.
- [x] Add an optional frame-by-frame realtime replay using manifest FPS/frame count; compare its totals to offline results without video decoding.
- [x] Test manifest portability, missing-run penalties, replay, provenance, and CLI output using small deterministic fixtures.

Acceptance: a fresh checkout can reproduce the same score from an explicit manifest; no hidden paths or algorithm-generated oracle is required. Human review of real event labels remains necessary before this becomes a release gate.

## 4. Simplify development and documentation

Files: `.github/workflows/tests.yml`, `tests/test_detect_integration.py`, `README.md`, `todo.md`, `src/juggletrack/analyze.py`.

- [x] Add standard tests/lint to CI using the committed dependency lockfile.
- [x] Replace machine-specific detector smoke-test paths with repository-relative paths/environment configuration.
- [x] Move the long parameter-tuning narrative out of `AnalyzeConfig`; retain concise meanings, current defaults, limitations, and a link to committed evidence.
- [x] Replace stale README accuracy claims and the two-item TODO with the ordered roadmap.
- [x] Run the full suite and lint and review the final diff.

Acceptance: the default development path is portable and current; experimental history remains available without overwhelming implementation code.

## 5. Improve detector accuracy with measured experiments

Proceed after split protection. Event counts and detection counts are not detector precision/recall.

- [x] Add `detection-eval` and pure center metrics with one-to-one pixel matching, empty-frame penalties, error-frame output, confidence filtering, and CSV/detection hashes.
- [x] Measure three existing development clips plus saved 1280 detections; record the confidence/resolution trade-offs in `docs/detection-findings-2026-09-07.md`. These are development diagnostics, not holdout validation.

- [ ] Build a manually checked frame benchmark with ball boxes or trusted ball centers, including empty frames. Split by recording/person/location; reserve an untouched test group.
- [ ] Measure one-to-one object matches, missed balls, duplicate/background detections, small-ball recall, and latency. Report by lighting, ball size, blur, occlusion, apex, hand region, and crossings.
- [ ] Compare the current v3 detector at 640 with a tighter person/flight-region crop and higher input resolution on identical frames. Measure costs and detection metrics before interpreting event-count changes.
- [ ] Add reviewed positives from the dominant miss categories and reviewed negatives from the dominant false positives. Do not label a real occluded/held ball as background because arc extraction rejected it.
- [ ] Fine-tune one candidate at a time; accept only gains on untouched data with no unacceptable latency or false-positive regression. Compare confidence/NMS settings on the development split, not the final test set.

## 6. Repair reference semantics and persistent live identity

These are algorithm changes, not safe constant retunes.

- [ ] Independently annotate complete run boundaries/catch totals/drop times on representative short, long, interrupted, and high-throw videos. Keep Meschke trajectory-derived events labeled as diagnostic references.
- [ ] Decide whether a run ends at the first confirmed miss or includes recovery; align counted events and reported intervals with that definition. Test continued catches after a suspected miss before changing `segment_runs`.
- [ ] Prototype persistent flight/event identities across analysis windows. Preserve physics fitting; use one-to-one association of observations to active flights and explicit event confirmation rather than timestamp-only dedup.
- [ ] Separate run liveness from event confirmation delay; test short real gaps versus temporary occlusions.
- [ ] Compare baseline and candidate on the same saved-detection manifests, including long noisy sessions and true drops. Require independent count accuracy, bounded state growth, and measured latency; offline parity alone is insufficient.
- [ ] Choose the supported UI path: migrate useful legacy screens to the tested `juggletrack` API before adding leaderboard/streaming features. Do not mix current uncommitted legacy work into these fixes.

## Execution record

This section is updated as tests and changes are completed. Unchecked later experiments are not claims of implementation or accuracy improvements.


### Completed foundation batch

- Dataset protection: 10 new failure/regression cases; existing data is preserved.
- Inference defaults: shared resolver for CLI and direct YOLO use; explicit stock weights remain available. No detector thresholds or trained weights changed.
- Event benchmark: relative manifests, fixed denominator, provenance/configuration hashes, reviewed-label declaration, CFR realtime replay, invalid-label/config rejection. Configuration typos and non-finite numeric values now fail explicitly.
- Development: CI workflow added (not executed on GitHub in this session); local paths removed from smoke-test fixtures; 130+ lines of historical tuning commentary moved out of `AnalyzeConfig`, with defaults verified unchanged.
- Detection measurement: new center evaluator and confidence sweep demonstrate that one difficult video's recall collapses under a global confidence increase. Findings are recorded; no inferred training labels were created.
- Independent code review found two issues in the new evaluator commands: report/input path collisions and silent exclusion of out-of-video detections. Both were reproduced and fixed, with regression cases for all input types, symlink aliases (including a symlinked manifest), invalid indices, and legitimate one-frame trimming.

### Remaining work and constraints

Independent real-video event labels, a truly untouched frame holdout, reviewed error-frame labeling, new detector training, run semantics, persistent live tracking, and legacy UI migration remain unchecked above. The benchmark's `human_reviewed` flag is a declaration, not proof of review. A generated-reference result has not been relabeled as independent truth. The source changes are on `codex/accuracy-foundations`; existing legacy edits are preserved.

### Verification

- Full standard suite: **320 passed, 3 detector tests deselected** (96.50 seconds).
- A final symlinked-manifest regression was added after that suite started. The current benchmark suite including that case passes: **17 passed**. The overall test inventory is now 321 standard cases.
- `ruff check src tests`: passed.
- `uv lock --check --offline --cache-dir /tmp/juggletrack-review-uv-cache`: passed; 72 packages resolved.
- Scoped `git diff --check`: passed for this batch's source/tests/docs. Existing legacy changes were excluded from this check and not modified.
- Workflow YAML parses; GitHub-hosted CI has not been run or pushed in this session.
- `AnalyzeConfig` defaults serialize identically to the pre-change baseline. No new model weights were trained or detector settings promoted.
