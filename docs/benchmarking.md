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
not an independent event oracle. For a reference that does not use the
extractor, see the [flight reference check](#flight-reference-check). The
manifest's provenance flag records your
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
agrees with the decoder's timestamps to within rounding. The realtime analyzer
compares times against its clock with a 1 µs tolerance, so a clock that differs
by less than that does not change its output, as long as the detections keep
their own timestamps. On the 22 saved `meschke-val` sessions, `frame_idx / fps`
at each video's own rate gives the same state on every frame as the recorded
clock. A clock that differs by more can still change the emitted states. Two
examples are `frame_idx / 30` for 29.75 fps footage, and a Matroska remux, whose
timestamps are rounded to the millisecond. When a detector stamps its
detections with the clock, as `live` does without `--detections`, noise in that
clock reaches arc extraction, which this tolerance does not cover. This
decoder-free mode requires constant-rate
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

## Flight reference check

`scripts/flight_score.py` compares the offline catch count with the flights
that the Meschke ball labels show. `juggletrack.eval.flight_reference` reads
those flights off each labelled ball's track. It fits no parabola and imports
nothing from the arc extractor or the event stages, so an extractor loss shows
up as a missed flight instead of cancelling out.

A flight is a labelled ball that rises from a low point to an apex and falls
back, where each swing is at least 10% of the clip's pattern height. The ball
lands when it comes back down to hand level, which is a quarter of the rise
above the low point. A flight counts when the labels show all of it and the
ball does not go on to the floor.

```bash
uv run --no-sync python scripts/flight_score.py flight-score.json
uv run --no-sync python scripts/flight_score.py controls.json \
  --only ss3_id_987 --controls
```

The first command scores every `<clip>/detections.jsonl` under
`outputs/meschke-val/post-hardening-2b` that has labels in
`data/raw/meschke/csv` and a video in `data/raw/meschke/videos`. Use `--root`,
`--csv-dir` and `--video-dir` for other locations. The video supplies fps and
frame size, and the report records its hash and frame count. The script prints one row per clip and writes a JSON report
with the hashes of the detections, labels and video, the source hash, and the
times of every missed flight and unmatched catch. It refuses a report path
that would replace an input, and two runs on the same inputs write identical
bytes.

Each row has these columns.

- `flights`: the reference flights that count.
- `catches`: the catches in the runs that `analyze_detections` kept.
- `matched`: the flights that got a catch. Matching is one to one. A catch can
  match a flight when it comes within 0.2 of that flight's own duration of the
  landing, and within 0.3 of the pattern width of the landing position. Both
  bounds follow the footage, so slow motion and real time are held to the same
  standard.
- `duplicates`: the catches on a flight that already has one. A flight that
  the labels do not show whole still lands once, so a second catch on it is a
  duplicate too.
- `unmatched`: the catches near no flight.
- `ignored`: the catches outside the labelled span, and the one catch on a
  flight that the labels do not show whole. These catches count neither for
  nor against.
- `missed`: the flights with no catch.
- `count_error`: `matched + duplicates + unmatched - flights`.
- `offset_s`: the median of catch time minus landing time over the matched
  pairs, in seconds. It is positive when the pipeline counts catches after the
  reference landings. A shift of the catch clock moves it long before it turns
  matched catches into unmatched ones. The totals leave it blank.
- `explained`: the flights with at least one extracted arc that is in flight
  at the apex time and passes within 0.1 of the pattern height of the apex.
  This column judges the extractor alone. `repeated` counts the flights with
  more than one such arc.

The flight count must not depend on the 10% threshold. The script repeats the
count at 5%, 15%, 20% and 30%. When the five counts differ by more than 3%, the
row says `FLAGGED` and stays out of the `TRUSTED` totals. Patterns that mix
high throws with low passes are flagged, and so are clips where a held ball
drifts in the hand.

`--controls` adds three seeded faults per clip, to show that the scorer can
see each kind of error. It deletes every tenth catch, doubles every tenth
catch, and moves every catch 2 s later. The faults must appear as `missed`,
`duplicates` and `unmatched`.

On 2026-11-22 the default root gave 1,485 flights and 1,041 catches over 22
clips, with 1,016 flights matched. Six clips were trusted, with 501 flights
and 344 matched. On `ss3_id_016` the labels show 177 flights, the extractor
explains 53 of them, and the pipeline counts 41 catches.

On 2026-11-23 the extractor seeds each flight from its apex and the catch
witness margin grows with the flight (GitHub issue #7, see
`docs/parameter-tuning.md`). The same root then gives 1,301 catches with
1,270 flights matched, 1 duplicate and 13 unmatched catches, against 2 and 8
before. The trusted clips match 498 of 501 flights with 1 unmatched catch, as
before. On `ss3_id_016` the extractor explains all 177 flights and the
pipeline counts 178 catches in one run, matching all 177 flights. The other
catch falls on a flight the labels do not show whole. A catch on
`ss8040_id_070` at 37.03 s is the same catch as before, moved 0.03 s, and now
scores as unmatched instead of duplicate. Five unmatched catches are new, all
on flagged clips, and the labelled tracks show that each is a real catch the
reference cannot match:

- `ss531_id_988` at 44.48 s and `ss531_id_989` at 43.67 s and 44.87 s: the
  ball comes to rest in the hand at the end of the clip just above the
  reference hand level, so the reference finds no landing.
- `ss531_id_989` at 9.23 s: the ball is caught high, near y 0.57, and carried
  down. The reference puts the landing 0.5 s later, where the carried ball
  passes hand level.
- `ss531_id_989` at 42.85 s: a real catch counted 0.49 s late, outside the
  0.37 s tolerance, because EM extends the arc over the ball resting in the
  hand.

The change also costs one matched flight on the flagged `ss531_id_014` (26 to
25). The same day's parity run over every saved session is summarised in
`docs/parameter-tuning.md`.

Limits:

- The check counts flights, not human-reviewed catches. Release-grade accuracy
  claims still need human-reviewed labels.
- A flagged clip has no dependable flight count. Sixteen of the 22 clips are
  flagged, most of them multi-height patterns.
- The Meschke clips were also used for tuning, so they are development data
  and not a holdout.
- The scorer sees only the time and x position of a catch. A catch that is
  late by about one landing interval of the same hand looks like a catch of
  the next flight. A 2 s shift is that close on `ss3_id_016` and on the `ss42`
  clips, so the shift control leaves most catches matched there. `offset_s`
  still shows it: on `ss3_id_016` it moves from +0.234 s to -0.045 s.
- The pipeline's hand line and the reference hand level are different heights,
  and the 0.2 time bound absorbs the difference. A change that moves catch
  times by less than that bound leaves the counts alone and shows only in
  `offset_s`.
- A one-frame label glitch can split a flight in two. On the flagged
  `ss423_id_088`, ball 2 is labelled at another ball's position for one frame
  at 1.40 s, and the reference counts one extra flight. A three-sample median
  over the heights removes such glitches, but it also removes real 1-zips on
  the `ss441` clips, so the reference does not use one.
- The saved detections must be on the `frame_idx / fps` clock that the labels
  use. The script stops when they are not.
- The check scores offline analysis only. It needs per-ball labels, which only
  the Meschke clips have.

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
