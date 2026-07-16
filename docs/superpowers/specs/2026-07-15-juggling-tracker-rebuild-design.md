# Juggling Tracker Rebuild — Design (v1: Desktop Prove-Out)

**Date:** 2026-07-15
**Status:** Approved by Andrew (sections 1–3 reviewed interactively)
**Supersedes:** the `jugglecount` prototype (moves to `legacy/`)

## 1. Goal

Rebuild the juggling tracker from scratch as **`juggletrack`**: detect juggling balls in
video, track them across frames, and determine juggling **runs** (start/end via **drops**)
with per-run **catch counts**. v1 proves the algorithm on desktop (macOS, Apple Silicon);
phase 2 ports it to a phone (Android — the user's videos are Pixel captures).

### v1 deliverables ("proven" means all three)

1. **Run/drop detection accuracy** on a labeled validation set: run boundaries at temporal
   IoU ≥0.9 on ≥80% of runs, and catch count within ±1 on ≥90% of runs.
2. **Realtime Mac webcam demo**: live overlay (balls, arcs, current-run catch count) at
   ≥15fps floor, 30fps goal.
3. **Offline video-analysis CLI**: any video file in → versioned `analysis.json` + annotated
   debug overlay video out.

### Scope and non-goals (v1)

- **3-ball cascade only.** Other patterns/props later; ball count is an assumption, not an output.
- **Detector must generalize to arbitrary videos** (any balls, lighting, background) — via a
  fine-tuned neural detector, not per-session color calibration.
- Non-goals: session history/stats DB, leaderboards, voice control, siteswap inference,
  phone app itself, >3 balls, clubs/rings.

## 2. Research grounding (July 2026)

Full research output: 5-angle parallel web survey; all 10 load-bearing claims verified
against primary sources.

- **[Hawkeye](https://github.com/jkboyce/hawkeye)** (MIT, abandoned 2020, by Jack Boyce):
  the only mature prior art. Blueprint: cheap detection → group into **parabolic arcs**
  (EM-refined weighted least squares, after Ribnick et al.) → throws/catches derived
  analytically from arc endpoints → runs from temporal arc overlap. Self-calibrates scale
  from fitted gravity: `cm_per_pixel = 980.7 / (g_px · fps²)`.
- **Cozens & Godsill, IEEE FUSION 2024** ([10706333](https://ieeexplore.ieee.org/document/10706333)):
  bimodal state-space tracker — two motion modes (airborne=ballistic, caught=in-hand) —
  plus rhythm/beat tracking for error correction. Code unreleased; implement from paper.
- **Sports-ball literature** (TrackNet family, [WASB BMVC 2023](https://github.com/nttcom/WASB-SBDT),
  TTNet, BlurBall): multi-frame heatmap regression is the consensus for tiny fast blurred
  balls. Juggling balls at webcam distance are 15–40px (larger than broadcast balls), so a
  fine-tuned bbox nano-detector is expected to suffice; heatmap is the documented fallback.
- **Tracking** ([arXiv 2509.18451](https://arxiv.org/abs/2509.18451)): all off-the-shelf
  Kalman trackers (ByteTrack/OC-SORT/BoT-SORT/StrongSORT) inadequate for tiny fast balls;
  appearance ReID useless on identical balls. Physics-informed Kalman + Hungarian is
  validated best practice. Portable tricks: ByteTrack low-confidence second association,
  OC-SORT observation-centric re-update, direction-consistency assignment cost.
- **Drop detection: no published prior art.** Genuine differentiator; designed from first
  principles here; highest empirical risk; gets dedicated eval clips.
- **Market**: 2025 wave of iOS counter apps (JuggleVision, Plapp Juggling Counter — the
  latter pose-only, no ball tracking) with no published tech. Niche for robust drop-based
  run detection is open.

## 3. Architecture

Fresh package; old prototype moves to `legacy/` untouched (reference until parity, then
deleted). Python 3.12, managed with `uv`.

```
src/juggletrack/
  types.py            # Data contracts: Detection, TrackPoint, Arc, ThrowEvent,
                      # CatchEvent, Run, Drop, SessionResult (pydantic, versioned)
  detect/             # BallDetector protocol; YOLO impl; person-ROI cropper;
                      # heatmap impl slot (fallback)
  track/              # Online two-mode Kalman (airborne|held) + Hungarian
  arcs/               # Parabola fitting: online incremental + offline global EM
  events/             # Arc endpoints → catches/throws; runs; drops; periodicity
  calib/              # Hand-line estimation (pose wrists), gravity→scale calibration
  pipeline/
    offline.py        # Batch: video → analysis.json + debug overlay (+ intermediates)
    realtime.py       # Streaming engine: same stages, ring buffer, half-arc latency
  apps/
    cli.py            # juggletrack analyze | live | label | eval | train  (typer)
    live_demo.py      # OpenCV-window realtime overlay
  data/               # Downloaders (Roboflow/Kinetics/yt-dlp), auto-labeler, COCO I/O
  eval/               # Metrics + eval harness
```

### Core decisions

- **`Arc` is the central abstraction** — a fitted parabola `(coefficients, t_start, t_end,
  quality)` with derived sub-frame throw/catch endpoints. Events, runs, drops, counting,
  and auto-labeling all compute from arcs. Catches are counted **per arc, not per ball
  identity**, so track-ID switches cannot corrupt counts.
- **Two pipelines, one algorithm.** Offline and realtime share `types/detect/events/calib`;
  they differ only in association: offline fits arcs globally (EM over the detection cloud,
  ignores track IDs); realtime builds arcs incrementally from the online tracker with
  ~half-arc (~0.3s) confirmation latency.
- **No DB in v1.** Outputs are versioned JSON files. Runtime state and persisted results
  are separate types (fixes the prototype's SQLModel split-brain).
- **Phone-portability rule:** detector stays a small pure-CNN with static input shape;
  training data stays in framework-neutral COCO JSON (survives a later switch to
  Apache-licensed models or to LiteRT/TFLite for Android).

### Salvaged from the prototype

Gravity-Kalman core (`legacy` tracker.py), parabolic gap interpolation idea, `find_peaks`
event logic (kept as cross-check), debug-overlay concept, pipeline-with-intermediates
pattern. **Dropped:** streamlit-webrtc, voice processor, SQLModel coupling, DB layer.

## 4. Algorithm core

### Detection

- Fine-tuned single-class ("ball") **YOLO11n @ 640×640 static** (YOLO26n when tooling
  matures). AGPL acceptable for a personal prove-out; COCO-format data keeps an
  Apache-model retrain (D-FINE / DEIMv2 / RF-DETR) open if this ever ships commercially.
- **Person-ROI cropping**: detect inside a square crop around the juggler (from pose;
  fallback: previous frame's ball-cluster centroid). 2–3× effective resolution for free.
- **Blur handling**: label blurred balls as the full visible streak (BlurBall convention);
  directional motion-blur augmentation; mine hard examples from tracker-interpolated frames.
- **Escalation triggers** (explicit, in order): (1) stack 3 grayscale frames as input
  channels of the same detector; (2) TrackNet/WASB-style multi-frame heatmap model
  (~1.5M params, 288×512, 3-frame input) — only if fine-tuned recall still fails at
  crossings/peak blur.

### Tracking (online path)

Owned, ~250 lines. State `[x, y, vx, vy]`, **two modes**:
- *Airborne*: gravity as control input; tight Mahalanobis gate (flight is exactly
  constant-acceleration).
- *Held*: wrist-anchored / high process noise; loose gate. Transitions at catch/throw.

Upgrades over the prototype: ByteTrack second association pass on low-confidence
detections; OC-SORT observation-centric re-update after occlusion gaps; velocity-direction
consistency term in the Hungarian cost.

### Arc extraction

- Online: sliding-window parabola fit per track; segment becomes an `Arc` when residuals
  stay under threshold ≥5 points; closes on residual blow-up (catch) or track end.
- Offline: Hawkeye-style **EM over the whole detection cloud** — E-step weights detection↔arc
  affiliation, M-step re-fits weighted least-squares parabolas; merge/prune passes.

### Events, hand line, calibration

- Throw = arc start crossing the hand line upward; catch = arc end crossing it downward.
  Sub-frame timestamps solved analytically from parabola coefficients.
- **Hand line** = rolling median wrist height from MediaPipe **Pose-lite** (wrists only),
  run every 2–3 frames, interpolated. Pose is assistive, not load-bearing; fallback hand
  line derives from arc-endpoint statistics (Hawkeye proves this suffices).
- Fitted gravity → real-world scale (pattern height in cm as a free metric).

### Runs

- Primary (Hawkeye rule): arcs overlapping in time chain into a run; gap > ~1.3× measured
  throw period ends it.
- Validator: rolling **autocorrelation of airborne-ball-count**; sustained peak at
  lag ≈ throw period ⇒ run active. Rejects false runs (walking, pickups); doubles as a
  per-run quality score.

### Drops (first-principles; highest-risk component)

Three corroborating signals; ≥2 required:
1. *Trajectory*: descending arc continues below hand line into a floor band; no wrist
   within proximity radius at arc end (held balls decelerate at hand height and co-move
   with a wrist).
2. *Bounce signature*: velocity sign flip with amplitude loss near the floor.
3. *Pattern-level*: run periodicity collapses within ~1 period. Drop vs deliberate stop:
   a stop ends with all balls at hand height near wrists; a drop leaves one ball low and
   far from both.

Juggling volume (x-range, hand line, apex band, floor band) is calibrated online from the
current run's arc statistics — no hardcoded pixels.

## 5. Data & training loop

Three tiers, all stored/exported as COCO JSON with a per-source manifest:

- **Tier A — labeled seed (~1.4k images):** Roboflow CC0 `juggling/juggling-balls`
  (400 imgs); Andrew's existing 68-image Roboflow workspace;
  [Meschke Juggling Data Set](https://sites.google.com/view/jugglingdataset/home)
  (222,150 tracked frames, 967,902 ball x,y CSVs → fixed-size boxes; also free trajectory
  ground truth for event-logic validation). Email Meschke for permission per his site.
- **Tier B — own footage:** existing Pixel videos in `data/raw/` + new recordings across
  varied rooms/lighting/ball colors (single-environment training overfits — vdrumsta lesson).
- **Tier C — diversity, unlabeled:** Kinetics-700 "juggling balls" (~700 clips via
  CVDF/FiftyOne); CC-BY YouTube via `yt-dlp` with license match-filters and
  `--write-info-json` attribution manifests.

**Semi-automated labeling flywheel:** offline pipeline over Tier B/C → detections that
survive **arc-verification become auto-labels** (false positives rarely fit gravity) →
human review only for low-confidence / physics-orphaned frames (Roboflow UI) → retrain →
repeat. Secondary tool: SAM2 video propagation (click each ball once per clip) for footage
the detector can't bootstrap. Expected total: a few thousand labeled frames.

**Licensing hygiene:** personal research use is fine for all tiers; anything redistributable
restricts to CC0 + CC-BY(+attribution) + Meschke-with-permission; Kinetics/UCF101 stay local.

## 6. Evaluation

**Hold out entire videos, never frames.** Metrics in priority order:

1. **Catch-count error per run**: `|pred − true|`; target ≤1 on ≥90% of runs.
2. **Run boundaries**: temporal IoU vs labels, target IoU ≥0.9 on ≥80% of runs;
   **drop precision/recall** on a dedicated drop-heavy clip set (deliberate drops +
   confusers: pauses, pickups, walk-aways), provisional target ≥0.8 each (provisional
   because drop detection has no prior art baseline).
3. **Event F1**: catches within ±2 frames (Meschke trajectories provide this cheaply).
4. Detection PR and realtime fps as diagnostics.

Never-trained-on generalization set: UCF101 `JugglingBalls` clips + Wikimedia Commons
juggling videos.

## 7. Testing

- **Synthetic-trajectory unit tests**: programmatically generate perfect 3-ball-cascade
  ballistics (known throws/catches/drops), add noise/dropout/ID-switches; assert arc
  fitter, event, run, and drop logic recover ground truth. The event core is fully
  testable with zero video files.
- **Golden-file regression tests**: 3–5 short real clips with hand-verified counts;
  CI asserts outputs within tolerance.
- **Property test**: catch count invariant under track-ID permutation.

## 8. Delivery

- **CLI** (`typer`): `analyze` (JSON + overlay), `live` (webcam demo), `label`
  (auto-label + export), `eval` (metrics), `train` (fine-tune).
- **Realtime demo**: plain OpenCV capture/render loop; detector exported to **CoreML**
  (MPS fallback). Overlay: balls, arcs, current-run catch count, run history, fps.
- **Error handling**: stages emit typed results and persist intermediates
  (`detections.jsonl`, `arcs.json`) for independent re-runs; pose failure degrades to
  arc-statistics hand line with a warning; single pydantic-validated YAML config.

## 9. Build order

1. `types.py` + synthetic-trajectory generator + event core, with unit tests (no ML).
2. Offline pipeline with stock COCO YOLO on own Pixel videos → first end-to-end
   `analysis.json` + debug overlay.
3. Eval harness + hand-labeled ground truth for own videos.
4. Data tiers + labeling flywheel + first fine-tune → measure the accuracy jump.
5. Online two-mode tracker + realtime engine → Mac webcam demo.
6. Drop-detection hardening on dedicated drop clips.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Drop detection unproven anywhere | Three corroborating signals; dedicated eval clips; build last, after event core is trusted |
| Motion blur kills detection recall at 30fps | ROI crop + blur labeling/augmentation; staged escalation to 3-frame stacking, then heatmap model |
| Fine-tuned detector overfits to own environment | Tier C diversity; whole-video holdouts; UCF101/Wikimedia never-trained-on set |
| AGPL (ultralytics) if project ever ships | Training data in COCO JSON; Apache retrain path (D-FINE/DEIMv2) documented |
| MediaPipe Pose perf on Apple Silicon unbenchmarked | Day-one spike; pose is assistive-only with a working fallback |
| Realtime fps on Mac | CoreML export (~90fps-class for nano models on M-series per vendor benchmarks); frame skipping; ROI mode |

## 11. Decision log (from design Q&A)

- v1 target: **desktop prove-out** (phone = phase 2; Android per Pixel footage).
- Scope: **3-ball cascade only**.
- Detection: **generalize to arbitrary video** via fine-tuned neural detector.
- v1 outputs: run/drop accuracy + realtime Mac demo + offline CLI (session stats deferred).
- Labeling: **semi-automated flywheel** with human correction.
- Approach: **A — physics-first hybrid** (over heatmap-centerpiece and salvage-and-fix).
