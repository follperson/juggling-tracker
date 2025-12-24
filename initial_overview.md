JuggleCount — Scaffolding Docs (v0)
Goal

Build an offline pipeline that:

Accepts arbitrary videos (may include non-juggling content),

Automatically finds juggling segments (start/end),

Counts throws within each juggling segment for 3-ball cascade,

Produces run lengths (streaks of consecutive throws separated by gaps/drops),

Supports semi-automated labeling to improve models over time,

Outputs machine-readable JSON + optional debug overlay video.

Non-goals (v0)

Real-time on-phone inference (future phase)

General juggling props / >3 balls / clubs / rings

Definitions

Throw: an event when a ball reaches a local minimum near the hand zone and then moves upward (velocity sign change).

Juggling segment: contiguous time interval in the video where 3-ball cascade is occurring.

Run: a contiguous series of throws where inter-throw gaps stay below a threshold; breaks imply drop/stop/segment end.

Expected Outputs
Per-video

analysis.json:

video_metadata: fps, resolution, duration

segments: list of juggling segments with:

start_time, end_time

total_throws

runs: list of {start_time, end_time, throw_count}

throw_times: list of timestamps (seconds) for all throws in the segment

quality: confidence metrics (detection coverage, track stability)

debug_overlay.mp4 (optional):

ball detections + track IDs + trajectory tails

throw counter + throw markers

segment start/end markers

System Architecture
Modules

ingest/

Load video, extract metadata, sample frames

segmenter/

Detect juggling intervals (start/end)

detector/

Per-frame ball detection (baseline generic → later fine-tuned)

tracker/

Track balls across frames, output trajectories

events/

Throw-event extraction from trajectories

runs/

Convert throws into runs + summary metrics

labeling/

Semi-automated labeling UI + dataset format + active learning hooks

viz/

Debug overlay video generation

cli/

jugglecount analyze <video> ... and jugglecount label ...

Data Contracts

Define stable JSON schemas for:

SegmentCandidate

DetectionsPerFrame

Tracks

ThrowEvents

RunSummary

Use pydantic models to enforce schema + versioning.

Approach Plan (v0 → v1)
Stage A: Baseline segmenter (heuristics)

A quick segmenter that doesn’t require training:

Compute per-frame “ball-likeness activity”:

number of detected balls per frame

track continuity over window

vertical motion magnitude / parabolic arc consistency

Sliding window classify as juggling if:

~3 balls detected consistently OR 2–3 with strong track continuity

periodic up/down motion exists across tracks

Post-process with hysteresis:

require min_on_duration (e.g., 1.5s) to start a segment

require min_off_duration (e.g., 1.0s) to end

Stage B: Baseline throw counter (trajectories → local minima)

Smooth y(t) per track

Local minima + velocity sign flip

Hand-zone gating: minima in bottom ~40% of frame

Debounce per ball: ignore events too close together

Merge events across balls and compute runs

Stage C: Labeling + dataset

Store clips and frame samples for:

segmentation labels (start/end)

ball bounding boxes (optional, for model improvement)

throw timestamps (optional, for evaluation)

Semi-automated:

run baseline → generate candidate segments + throw markers

human corrects via UI (accept/split/trim segments; quick “add/remove throw”)

Use labeled data to train:

better segmenter (light classifier over features or small temporal model)

better detector (fine-tune on juggling balls)

CLI Design
Analyze

jugglecount analyze video.mp4 --out outdir --debug-overlay --save-intermediates

Outputs:

outdir/analysis.json

outdir/debug_overlay.mp4 (optional)

outdir/intermediates/ (optional): detections, tracks, feature timeseries

Label

jugglecount label video.mp4 --pred outdir/analysis.json

Launch labeling UI to:

confirm/adjust segments

optionally confirm throw markers in a segment

export labels/*.json + dataset_manifest.json

Semi-automated labeling UX (minimum viable)

Start simple (keyboard-driven):

Timeline scrubber

Show video with overlays

Segment list with:

accept/reject

trim start/end

split segment at current time

Throw review mode:

show throw timestamps as ticks

add/remove at current time

quick jump between predicted throws

Implementation options:

Streamlit app (fast)

Lightweight web UI (FastAPI + simple frontend)

Desktop (PyQt) if needed (probably overkill for v0)

Evaluation

Define 3 metrics:

Segmentation IoU with labeled segments (temporal overlap)

Throw count error per segment: abs(pred - true)

Event F1: match throws within ±N frames tolerance (e.g., ±2 frames)

Also log:

detection coverage: % frames with >=2 balls detected

track fragmentation rate

confidence per segment

Storage Layout
jugglecount/
  src/jugglecount/
  data/
    raw_videos/
    clips/                # auto-extracted candidate segments
    labels/
    datasets/
  outputs/
    <video_id>/
      analysis.json
      debug_overlay.mp4
      intermediates/

Task List for Implementation (Markdown)
0. Repo + tooling

 Initialize repo with:

src/ layout, pyproject.toml

ruff, black, mypy (optional), pytest

pre-commit hooks

 Add CI workflow: lint + unit tests

 Create README.md with quickstart + examples

1. Video ingest + utilities

 Implement VideoReader:

load fps, frame count, resolution, duration

iterate frames with frame index + timestamp

 Add frame sampling helpers:

fixed stride sampling (e.g., every N frames)

clip extraction: extract_clip(video, t_start, t_end)

 Unit tests for timestamp conversions and clip extraction boundaries

2. Baseline ball detector wrapper

 Add detector interface:

detect(frame) -> List[Detection(xyxy, score, cls)]

 Implement baseline using a pre-trained model (pluggable):

keep model loading isolated (lazy init)

configurable confidence threshold

 Add non-max suppression handling if needed

 Save per-frame detections to intermediates/detections.jsonl

3. Tracking

 Define Track data model:

id, list of (t, x, y, bbox, score)

 Implement tracker wrapper (ByteTrack/SORT-like):

input: detections per frame

output: tracks

 Add basic track smoothing:

interpolate small gaps

drop tracks shorter than X frames

 Save tracks to intermediates/tracks.json

4. Feature extraction for segmentation

 Compute per-frame / per-window features:

num detections

num active tracks

mean track velocity magnitude

periodicity proxy (autocorrelation peak on y-velocity per window)

“parabolic-ness” score (fit quadratic to short arcs, low residual)

 Serialize feature timeseries to intermediates/features.parquet (or json)

5. Baseline segmenter (heuristic + hysteresis)

 Implement segment_video(features) -> List[Segment(start, end, confidence)]

windowed score + thresholds

hysteresis (min_on, min_off)

merge nearby segments (gap < 0.5s)

 Add segment confidence metrics:

coverage ratio, track stability, periodicity strength

 Unit tests with synthetic feature sequences

6. Throw-event detection

 Implement extract_throw_events(tracks, segment) -> List[ThrowEvent(time, track_id, confidence)]

smooth y(t)

local minima + velocity sign flip

hand-zone gating (bottom % of frame)

debounce per track

 Merge events across tracks into time-ordered stream

 Add tolerances for fps (frame-based and time-based thresholds)

7. Runs computation

 Implement compute_runs(throw_times) -> List[Run]

estimate median inter-throw interval

break threshold: max(0.8, 3*median_delta)

output run start/end/count

 Add run stats: longest run, mean run length

8. Debug overlay video

 Implement overlay renderer:

draw bboxes + track IDs

draw trajectory tails (last N points)

display counters (segment index, throws, current run length)

flash markers at throw events

 Export debug_overlay.mp4 with same fps

9. JSON output schema + persistence

 Define pydantic models for:

VideoMetadata, SegmentResult, ThrowEvent, Run

 Write analysis.json writer + version field

 Ensure deterministic ordering of events/runs

10. Labeling: segment correction UI (MVP)

 Build Streamlit labeling app:

load video + predicted segments

timeline scrubber and segment table

actions: accept/reject, trim start/end, split at timestamp

 Save labels:

labels/<video_id>_segments.json

 Add export of corrected clips into data/clips/

11. Labeling: throw correction UI (optional v1)

 Within a selected segment:

show predicted throw ticks

add/remove throw at current timestamp

 Save:

labels/<video_id>_throws.json

 Add “quick review” mode: jump throw-to-throw

12. Dataset + training hooks (v1+)

 Dataset manifest:

list of videos, segments, optional throw labels, detector bboxes

 Training stubs:

segmenter classifier from features (sklearn baseline)

detector fine-tuning data export (COCO-style)

 Active learning loop:

select low-confidence segments for labeling first

13. Documentation

 Write docs/architecture.md explaining pipeline + data contracts

 Write docs/labeling.md explaining labeling workflow

 Write docs/evaluation.md with metrics + how to run evaluation

Notes / Implementation Tips

Keep everything configurable via a config.yaml:

detector thresholds, hand-zone %, debounce, segment hysteresis, run gap logic

Save intermediates early so you can iterate on logic without rerunning detection.

For cascade videos, false positives usually come from:

reflections, bright spots, background clutter → solved by fine-tuning detector

tracking ID switches → solved by tuning tracker and smoothing