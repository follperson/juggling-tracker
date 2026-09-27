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
aggregate, not a claimed perfect score.

Realtime mode feeds every frame, including empty frames, and flushes the tail.
This decoder-free mode requires constant-rate timestamps starting at zero;
inconsistent timestamps are rejected. For variable-rate footage, use the
existing `live VIDEO --detections FILE --no-display` path, which reads the
video's presentation timestamps. Realtime reports compare total catches to
labels and offline totals; they do not claim per-run IoU because the realtime
state does not expose completed run intervals.

Keep `development` and `holdout` manifests separate. Once a holdout influences
parameter selection it has become development data. Historical result JSONs
without source/configuration hashes should remain labeled historical.

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
