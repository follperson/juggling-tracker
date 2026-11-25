# juggletrack

Physics-first juggling tracker: detects juggling balls in video, verifies them
against ballistic physics, and counts throws, catches, runs, and drops — offline
on recorded video or live from a webcam.

**How it works:** a fine-tuned YOLO detector proposes ball positions; everything
downstream is physics. Detections only count when they lie on a fitted parabolic
arc (gravity is the precision filter), catches/throws come from arcs crossing an
estimated hand line, runs chain periodic arcs, and drops need agreeing signals.
Catches are counted per-arc, not per-tracked-ID, so identity switches don't
corrupt counts.

## Setup

```bash
uv sync --locked
```

Inference commands use `--model`, then `JUGGLETRACK_MODEL`, then
`models/juggletrack-v3/best.pt` relative to the working directory. This is the
fine-tuned yolo11n v3 detector used in the recorded benchmarks.

Weights are not included in Git and are not distributed. Copy a v3 checkpoint
to that location, point `--model` at it, or set
`JUGGLETRACK_MODEL=/absolute/path/to/best.pt`. To experiment with the
stock detector, explicitly pass `--model yolo11n.pt` (Ultralytics may download
it). Missing default weights produce an actionable error. Saved-detection
replay and motion labeling do not need ball-detector weights.

## Usage

Live webcam demo (press `q` to quit; prints the session summary):

```bash
uv run juggletrack live 0 --model models/juggletrack-v3/best.pt --display
```

Analyze a recorded video (writes `analysis.json` + debug overlay):

```bash
uv run juggletrack analyze path/to/video.mp4 --model models/juggletrack-v3/best.pt
```

All commands (`--help` on each for options):

| Command | Purpose |
|---|---|
| `analyze` | Offline pipeline: detect → arcs → events → runs/drops, with debug overlay (`--dots all` shows raw detections) |
| `live` | Realtime tracker: webcam index or video file, HUD overlay, `--detections` replays saved detections |
| `eval` | Score an `analysis.json` against hand labels (catch error, run IoU, drop P/R) |
| `benchmark` | Evaluate a manifest of saved detections and reviewed labels; optional realtime replay, with configuration/input hashes |
| `label` | Arc-verified auto-labeling → COCO dataset (`--detector motion` for cold-start; `--negatives` mines hard negatives) |
| `coverage` | Per-frame detection-count stats for a video (detector health check) |
| `detection-eval` | Compare saved detections with Meschke ball centers: precision, recall, duplicate candidates, and localization error |
| `train` | Fine-tune YOLO on an assembled dataset |
| `export` | Export weights to CoreML (blocked by torch/coremltools version skew as of 2026-08; see bench findings) |

## Outputs

All coordinates are normalized to the frame, with `y` growing downward. Times
are seconds from the start of the video. `src/juggletrack/types.py` is the
schema of record.

| File | Written by | Contents |
|---|---|---|
| `analysis.json` | `analyze` | `schema_version`, `runs`, `drops`, `arcs`, `hand_line_y`, and `meta` |
| `overlay.mp4` | `analyze` | The input video with the hand line, arc tails, drop markers, and a counter |
| `detections.jsonl` | `analyze --save-intermediates` | One detection per line: `frame_idx`, `t`, `x`, `y`, `w`, `h`, `confidence` |
| `live_session.json` | `live --out DIR` | The final live state, including the run, catch, and drop counters |

Each run in `analysis.json` has `start_t`, `end_t`, `catches`, `throws`,
`end_reason` (`drop`, `stop`, or `video_end`), `period_s`, and the ids of its
arcs. A run that was already under way when the video started has a negative
`start_t`: the time of its first throw, extrapolated from the fitted arc. Pass
a saved `detections.jsonl` back with `--detections` to re-run the event core
without the detector.

## Accuracy envelope (measured)

- The recorded 22-video offline comparison at shipped defaults scores 9/26
  reference runs (34.6%) within one catch, including unmatched runs as failures.
  Those videos were used for parameter tuning, and the trajectory-derived
  reference events have known boundary errors. This is a diagnostic result,
  not a verified holdout accuracy claim.
- Realtime throughput was previously measured at 32–47 fps on Apple Silicon
  (MPS). Accuracy on long sessions remains unresolved: a direct live replay of
  saved `ss3_id_016` detections (`juggletrack live --detections`) counts 210
  catches and 8 drops versus offline's 41 catches and 0 drops. Both report 9
  runs; the generated reference reports 5 runs and 52 catches. Offline parity
  alone is not sufficient.
- Scope: 3-ball patterns. The detector is trained on juggling balls — other
  thrown objects won't detect reliably.

## Development

```bash
uv run pytest -q          # excludes detector tests requiring weights/footage
uv run ruff check src tests scripts
```

GitHub Actions runs the standard suite and lint. Run optional detector checks
with `uv run pytest -m detector`; set `JUGGLETRACK_TEST_VIDEO` to override the
local smoke-test video. These checks can download stock weights and train a
small smoke-test model.

A weekly job runs `uv audit --locked` against `uv.lock`. To clear a finding,
upgrade only the named package with `uv lock --upgrade-package <name>`, then
rerun both test tiers.

Dataset assembly requires an empty output directory and unique source names.
Use a new versioned directory for each build. Split by independent recording;
different crops/exports of the same recording must stay together. Single-source
assembly exists only for smoke tests: its validation path reuses training
images and must not be used to report accuracy.

[Benchmarking and detector experiments](docs/benchmarking.md) describe the
measurement workflow, and [the tuning record](docs/parameter-tuning.md) explains
the duplicate-suppression defaults. Open work is tracked in
[GitHub issues](https://github.com/follperson/juggling-tracker/issues).
`AGENTS.md` holds the working rules for agents. Historical designs and findings
remain in `docs/superpowers/`.
The supported package is `src/juggletrack`; the separate `legacy/` prototype
has its own dependencies and is outside the standard test suite.

## Data credits

Ground-truth validation uses the Stephen Meschke Juggling Data Set
(https://sites.google.com/view/jugglingdataset). The benchmark commands expect
its videos under `data/raw/meschke/videos` and its label files under
`data/raw/meschke/csv`. Training corpora include
Creative Commons–licensed videos (per-directory `MANIFEST.json` files record
sources and licenses).
