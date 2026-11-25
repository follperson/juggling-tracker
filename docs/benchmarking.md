# Measuring accuracy

Measure detector quality and event-count quality separately. Three detections
in a frame can be three balls, three boxes on one ball, or background clutter.
Likewise, correct detections can still produce incorrect counts when live
analysis repeatedly reconstructs the same event.

## Event benchmark

Create a manifest alongside saved detection files and independently reviewed
labels. Paths are relative to the manifest, so the directory can move between
machines. Every `video` identity must be unique and match its label file.

```json
{
  "schema_version": "1",
  "split": "development",
  "clips": [
    {
      "video": "practice-01",
      "detections": "practice-01/detections.jsonl",
      "labels": "practice-01/labels.json",
      "label_source": "human_reviewed",
      "fps": 30,
      "frame_count": 1800,
      "detector": {"model": "juggletrack-v3", "imgsz": 640, "conf": 0.05}
    }
  ]
}
```

The label file uses the existing `VideoLabels` format:

```json
{
  "video": "practice-01",
  "runs": [{"start_t": 1.0, "end_t": 6.0, "catches": 12}],
  "drops": []
}
```

These are format examples, not reviewed footage or measured results. Inspect
the real video and count events before marking labels `human_reviewed`.
`oracle_events` derives events from Meschke trajectories using the same
downstream algorithm as predictions. It is useful diagnostically, but it is
not an independent event oracle. The manifest's provenance flag records your
assertion of review; software cannot verify that a person reviewed a video.

```bash
uv run juggletrack benchmark data/benchmarks/development/manifest.json \
  --out outputs/benchmark-baseline.json
uv run juggletrack benchmark data/benchmarks/development/manifest.json \
  --realtime --out outputs/benchmark-live.json
```

Use `--config analysis-config.json` to supply any `AnalyzeConfig` fields;
unspecified fields retain shipped defaults. Reports include the complete
configuration, package version, source-code hash, manifest hash, and hashes of
each detection/label file. Record the detector model hash and inference
settings in the manifest's `detector` object when creating detections.
Report destinations must differ from every input, including symlink or hardlink
aliases. A report cannot replace its manifest, configuration, labels or detections.

Aggregate event scores use **all labeled runs** as the denominator. An
unmatched labeled run is a failure. Extra predicted runs and drop TP/FP/FN are
reported explicitly. Metrics with no applicable observations are `null` in the
aggregate and in each clip's offline scores, not a claimed perfect score.

Realtime mode feeds every frame, including empty frames, and flushes the tail.
It feeds the clock the detections were recorded with: a frame with detections
uses their `t`, an empty frame between two detected frames is interpolated
linearly, and empty frames before the first or after the last detected frame
step by `1 / fps`. On detected frames this is exactly the clock that
`live VIDEO --detections FILE` feeds for the same recording; on empty frames it
agrees with the decoder's timestamps to within rounding. Substituting
`frame_idx / fps` would differ by rounding errors that change which detections
the sliding window keeps. This decoder-free mode requires constant-rate
timestamps starting at zero; inconsistent timestamps, and detections in one
frame with different `t`, are rejected. For variable-rate footage, use the
existing `live VIDEO --detections FILE --no-display` path, which reads the
video's presentation timestamps. Realtime reports compare total catches to
labels and offline totals; they do not claim per-run IoU because the realtime
state does not expose completed run intervals.

Keep `development` and `holdout` manifests separate. Once a holdout influences
parameter selection it has become development data. Historical result JSONs
without source/configuration hashes should remain labeled historical.

## Event-core parity

A refactor or speedup of the event core should leave every result unchanged.
`scripts/event_core_parity.py` checks that on real footage. It runs
`analyze_detections` on every saved `detections.jsonl` under `outputs/` and
records a SHA-256 digest of each `SessionResult`. With `--realtime`, it also
replays the matching sessions frame by frame through `RealtimeAnalyzer` and
hashes every emitted state except the timing field. The replay feeds the clock
that live mode recorded: a frame with detections gets their timestamp, and an
empty frame gets the linear interpolation between its detected neighbours.
Frames before the first detection step back from it at the median frame
interval. `RealtimeAnalyzer` reacts to clock differences as small as one
floating-point step, so a replay on a reconstructed constant-rate clock can
report different counts. Take one snapshot before the change and one after it,
on the same machine:

```bash
uv run --no-sync python scripts/event_core_parity.py snapshot before.json \
  --realtime meschke-val/ --realtime-frames 900
# apply the change
uv run --no-sync python scripts/event_core_parity.py snapshot after.json \
  --realtime meschke-val/ --realtime-frames 900
uv run --no-sync python scripts/event_core_parity.py compare before.json after.json
```

`compare` prints `PARITY OK` and the speed ratio, or names each session whose
digest or input changed and exits nonzero. Each snapshot records the source
hash of the package it imported and the git commit of the checkout that holds
it, so a baseline can come from an older checkout on `PYTHONPATH`. When both
snapshots record the same source hash, `compare` says so: matching digests then
show only that the run repeats. Digests depend on the platform's floating point and numpy
build. Never commit them as golden values or compare snapshots from different
machines. Realtime digests also depend on the harness version: snapshots taken
before the harness used the recorded clock do not match current ones, so take
both snapshots with the same copy of the script. The saved detections are
local, gitignored data, so CI does not run this check. A change that is meant
to alter results fails `compare` by design. In that case, review the
per-session run, catch and drop counts that each snapshot records instead.

## Detector center benchmark

For Meschke footage with saved detections, run:

```bash
uv run juggletrack detection-eval data/raw/meschke/videos/ss3_id_110.MP4 \
  outputs/meschke-val/ss3_id_110/detections.jsonl \
  data/raw/meschke/csv/ss3_id_110.csv \
  --tolerance-px 10 --conf 0.05 --out outputs/centers-005.json
```

Repeat with `--conf 0.1` or `0.25` to filter the same saved predictions. This
cannot recover predictions below the confidence threshold used when saving
them. Match `--stride` to the detector's sampling stride. The scorer includes
empty labeled frames and uses maximum-cardinality one-to-one center matching,
then minimizes distance among valid matches. Coordinates are scaled separately
by video width and height before measuring pixel distance.

An unmatched prediction within the tolerance of a labeled ball is a duplicate
candidate; `background_fp` includes both background detections and detections
whose localization error exceeds the tolerance. These categories require visual
review before treating them as training labels. The report includes per-frame
errors and input hashes. Tolerance is explicit: these are center-localization
precision/recall metrics, not box mAP. A CSV/video frame-count difference of up
to two frames is truncated to the common span and reported; larger differences
are rejected. Prediction indices outside the actual video are rejected before
any permitted common-span trim. Report destinations cannot replace the video,
detections or center annotations.

The [September 7 diagnostic findings](detection-findings-2026-09-07.md) record
the first comparison on existing development footage.

## Reviewed frames from practice recordings

Use `reviewed-detection-eval` with a local source video, saved `Detection` rows in JSONL, and a sparse frame review. It works without model weights. Preselect the frames and match radius, then review the raw video with detection overlays hidden. Use the image as displayed after rotation. Coordinates run from zero to one across its width and height. Use zero-based decode indices and seconds on the source frame clock relative to the first frame. Record each visible ball in the frame. Mark uncertain contact or an unresolvable candidate as `ambiguous`. Omit a frame if a full review of that frame is impossible.

```json
{
  "schema_version": 1,
  "review_method": "model_assisted",
  "source_sha256": "<replace with the SHA-256 of the source video>",
  "width": 1080,
  "height": 1920,
  "frames": [
    {"frame_idx": 0, "t": 0.0, "centers": [
      {"x": 0.40, "y": 0.30, "state": "airborne"},
      {"x": 0.61, "y": 0.66, "state": "held"}
    ]},
    {"frame_idx": 8, "t": 0.267, "centers": []}
  ]
}
```

Replace the dimensions, indices, timestamps and centers with your observations. Set `review_method` to `human` for human-reviewed positions, `model_assisted` for model-assisted visual judgments, or `synthetic` for a constructed fixture. The report preserves this declaration. Keep reviewer notes, contact brackets and unresolved alternatives alongside the input. Compute the source digest with `shasum -a 256 practice.mp4` or `sha256sum practice.mp4`.

```bash
uv run juggletrack reviewed-detection-eval practice.mp4 detections.jsonl centers.json \
  --tolerance-px 12 --out reports/practice-centers.json
uv run pytest -q tests/test_reviewed_detection_cli.py
```

The test builds its own synthetic recording and exercises the command. For a real comparison, set the pixel radius in advance from the displayed ball size and expected labeling uncertainty. Record detector weights, confidence, stride and inference settings with each saved prediction file. Include only frames the experiment intended to process. An explicit empty label list can incur false positives. A reviewed ball without a prediction incurs a miss. Unlisted frames contribute nothing to the scores.

The command checks the source file digest, displayed dimensions and reviewed timestamps against the decoder, including frames containing zero predictions. Detections on a reviewed frame must match its timestamp within one millisecond. It records hashes of the source, labels, detections and scoring implementation. A saved file without an inference manifest still lacks proof of which source and weights generated it.

Matching first maximizes the number of definite ball matches within the radius, then minimizes their total distance. A remaining prediction can match an ambiguous center once and be excluded from judged counts. Both definite airborne and held centers contribute to generic ball precision and recall. The separate airborne recall answers how often the detector located an airborne ball. It does not measure whether the detector can classify flight state. Missing denominators appear as `null`. Each reviewed frame includes counts for inspecting errors back on the recording. [Experiment and decision](research-plan.md#first-controlled-experiment)

## Detector improvement sequence

1. **Create an independent frame benchmark.** Start with roughly 200–300
   manually checked frames from several recordings, including empty frames,
   small/distant balls, apexes, hand regions, motion blur, crossings, and
   difficult lighting. Split by recording/person/location, never adjacent
   frames. This is an initial diagnostic sample, not a claimed sufficient
   dataset size for generalization.
2. **Count individual errors.** Match predictions to labeled balls one-to-one.
   Report precision, recall, duplicate boxes, background false positives,
   localization error, and latency. Break out small-ball/occlusion/lighting
   slices. Meschke's trusted centers can support center-based localization
   metrics without its algorithm-derived event labels; box-IoU metrics need
   independently checked boxes.
3. **Run a resolution/crop comparison before training.** Compare the same v3
   weights at 640 versus 960/1280 and a crop covering the complete person and
   flight region. Preserve coordinate mapping to the original frame. A crop
   can give small balls more pixels at lower cost than scaling the full frame,
   but must not cut off high throws. These are hypotheses to measure, not
   automatic default changes.
4. **Review targeted training examples.** Add positives from the largest miss
   categories and negatives from actual background false positives. Review
   auto-labels, especially around hands, occlusions and out-of-frame throws;
   absence of a fitted arc does not mean absence of a ball.
5. **Tune one candidate at a time.** Compare confidence/NMS settings and then
   fine-tuned weights on the development split. Increasing confidence can
   reduce false positives while losing real balls. More aggressive NMS can
   suppress duplicates while merging real crossing balls. Judge both sides.
6. **Accept only independently measured gains.** Keep a fixed latency budget
   and report regressions as well as improvements on untouched recordings.
   Run the event benchmark too; better detector recall previously increased
   catch over-count on `ss3_id_016` rather than fixing it.

This sequence follows Ultralytics' emphasis on representative, consistently
annotated data and controlled testing. See their
[data collection guidance](https://docs.ultralytics.com/guides/data-collection-and-annotation/),
[model testing guidance](https://docs.ultralytics.com/guides/model-testing/), and
[inference settings](https://docs.ultralytics.com/modes/predict/) for the
current `imgsz`, `conf`, and `iou` controls. No detector improvement is claimed
until the above comparisons are measured.
