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
uv sync
```

Ball detector weights: `models/juggletrack-v3/best.pt` (fine-tuned yolo11n;
champion model for diverse footage). Commands default to stock `yolo11n.pt`
unless `--model` is passed — pass the v3 weights for real use.

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
| `label` | Arc-verified auto-labeling → COCO dataset (`--detector motion` for cold-start; `--negatives` mines hard negatives) |
| `coverage` | Per-frame detection-count stats for a video (detector health check) |
| `train` | Fine-tune YOLO on an assembled dataset |
| `export` | Export weights to CoreML (blocked by torch/coremltools version skew as of 2026-08; see bench findings) |

## Accuracy envelope (measured)

- Offline catch counting: 137 counted vs 136 ground truth on the held-out
  136-catch validation run. Broader per-video oracle validation lives in
  `docs/superpowers/plans/` findings documents.
- Realtime: 32–47 fps on Apple Silicon (MPS), run segmentation matches offline
  exactly on all benchmarks; a known catch over-count remains on runs longer
  than the 8s analysis window (documented, with root cause, in
  `docs/superpowers/plans/2026-07-19-plan4-bench-findings.md`).
- Scope: 3-ball patterns. The detector is trained on juggling balls — other
  thrown objects won't detect reliably.

## Development

```bash
uv run pytest -q          # 244 tests
uv run ruff check src tests
```

Design docs, plans, and measured findings: `docs/superpowers/`. The legacy
prototype this replaced is preserved under `legacy/`.

## Data credits

Ground-truth validation uses the Stephen Meschke Juggling Data Set
(https://sites.google.com/view/jugglingdataset). Training corpora include
Creative Commons–licensed videos (per-directory `MANIFEST.json` files record
sources and licenses).
