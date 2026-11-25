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

### Timing catches against visible contact

A run can have the right total even when its predicted catches occur after the
balls have settled in the hands. To measure that error, add `catch_windows` to
the video label file for intervals you reviewed from start to finish. For each
successful reception, bracket the first contact on the source video clock.
Use the following frames to check that the hand retained the incoming ball.
A ball that brushes the fingers and falls supplies a negative example.

```json
{
  "video": "practice-01",
  "runs": [{"start_t": 1.0, "end_t": 6.0, "catches": 12}],
  "drops": [],
  "catch_windows": [
    {"start_t": 2.0, "end_t": 3.0, "catches": [
      {"earliest_t": 2.20, "latest_t": 2.24},
      {"earliest_t": 2.68, "latest_t": 2.72}
    ]},
    {"start_t": 4.0, "end_t": 4.5, "catches": []}
  ]
}
```

The second window records half a second of reviewed footage with zero catches.
Leave ambiguous or unreviewed footage outside the windows. A window includes
its start and excludes its end. Both endpoints of each contact bracket count.

`benchmark` reports each offline match with its arc ID and predicted time,
each extra prediction, and each missed bracket in `offline.catch_timing`.
Each prediction and bracket can match at most once. Simultaneous catches can
have overlapping brackets. The score compares timestamps directly on the source
clock. It cannot establish ball identity or measure live display latency.
The summary pools precision and recall over the reviewed windows. Use the
ordinary run and drop labels for the whole-recording scores above.

Scoring uses four constants from `src/juggletrack/eval/metrics.py`, and
`juggletrack eval` applies the same rules. A predicted run and a labeled run can
match only when their temporal IoU is above `RUN_MATCH_MIN_IOU` (0.1). Runs pair
greedily from the highest IoU down, each run at most once, and equal IoU goes to
the earlier start time. A matched run passes the boundary check at IoU of at least
`RUN_IOU_PASS` (0.9), and passes the catch check when its catch count is within
`CATCH_ERROR_PASS` (1) of the label. A predicted drop matches a labeled drop
within `DROP_TOL_S` (1.0 s). Drops pair one to one, with the most matches first
and the smallest total time error second. Reports do not record these constants,
so changing one changes what earlier reports mean.

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

`--realtime-frames 900` stops each replay after its first 900 frames, about
30 seconds of video, and finalizes the analyzer there. Over the sessions this
command selects, that replays about a quarter of the recorded frames, and the
`final` counts it records are the state at the cap, not the session totals. A
change that only shows up later in a session passes the capped check. Before
merging a change to `RealtimeAnalyzer`, take one pair of snapshots with
`--realtime-frames 0`, which replays every frame. `snapshot` exits with an
error and writes no file when it analyses no session, or when a `--realtime`
value ends up replaying no session, whether because it matches nothing under
`--root` or because `--max-detections` skipped every match or every match was
empty. It checks the match before the run starts, so a typo fails at once.

`compare` prints `PARITY OK` and the speed ratio, or names each session whose
digest or input changed and exits nonzero. When the two snapshots share no
session it compares nothing, says so and exits 2. Each snapshot records the source
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

### Tracing missed catches with dense ball positions

A separate normal-speed development recording had 17 visually reviewed receptions
in a six-second window. The position review covered all 240 consecutive frames
in that window and its one-second buffers. A second review marked the reception
times from the video. Both reviews were model-assisted and used the source frame
timestamps. The event comparison matched each prediction to at most one contact
bracket, extended by two source frames on each side.

The previous extractor matched seven receptions using the saved detections.
Four missed flights had only two to five usable ball positions, one well-observed
flight disappeared during extraction, three descents continued into hand motion,
and two fitted arcs ended before the landing was witnessed. Replaying only the
reviewed airborne positions recovered 15 timed matches. This diagnostic gives
the tracker information it would have to infer in ordinary use.

Two changes recovered three receptions on the original detections. The extractor
can restore an observed apex after the other fits converge when its points are
unclaimed and its curvature agrees with the other flights. A floor-bound track
can end at the hand when consecutive observations show a lateral reversal and
slower descent than the incoming flight predicts. The revised pipeline matched
ten of the 17 receptions, with zero unmatched reports in the window, and removed
one false drop. Sparse coverage and ambiguous contact still limit this recording.
The private video and its labels are development material. The public regression
tests use generated trajectories in `tests/test_enclosed_flights.py` and
`tests/test_hand_carry.py`.

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

Use `reviewed-detection-eval` with a local source video, saved `Detection` rows in JSONL, and a sparse frame review. It works without model weights. Preselect the frames and match radius, then review the raw video with detection overlays hidden. Use the image as displayed after rotation. Coordinates run from zero to one across its width and height. Use zero-based decode indices and seconds on the source frame clock relative to the first frame. Record each visible ball in the frame. Use `visible` for a definite ball whose airborne or held state is unclear. Use `ambiguous` when even its identity as a ball is uncertain. Omit a frame if a full review of that frame is impossible.

```json
{
  "schema_version": 1,
  "review_method": "model_assisted",
  "source_sha256": "<replace with the SHA-256 of the source video>",
  "width": 1080,
  "height": 1920,
  "frames": [
    {"frame_idx": 0, "t": 0.0, "centers": [
      {"x": 0.40, "y": 0.30, "state": "airborne",
       "visible_box": {"left": 0.39, "top": 0.29, "right": 0.41, "bottom": 0.31}},
      {"x": 0.61, "y": 0.66, "state": "held"}
    ]},
    {"frame_idx": 8, "t": 0.267, "centers": []}
  ]
}
```

The optional `visible_box` records the visible outline in normalized source coordinates. Omit it where blur, cropping or a hand makes the boundary unreliable. A definite ball can still contribute a center in that case.

With measured extents the JSON report includes `box_localization`. Each annotated target carries its visible width and height in source pixels. The scorer reuses the one-to-one center assignment and compares the matched prediction's visible rectangle, clipped to the displayed image, against the reviewed box. `center_error_px` is null for a missed center. `iou` and `area_ratio` are also null when a matched prediction supplies only a point. `reviewed`, `center_matched` and `with_predicted_extent` give the three denominators. `at_least_double_area` counts predicted boxes covering at least twice the reference area. `iou_at_least_0_3` counts center-assigned pairs meeting that overlap. These counts let you trace a poorly sized box separately from a missing ball. A catch comparison requires replaying the detections through the event core.

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
