# Plan 4 Task 5: Field Bench + Offline-Parity Validation — Findings

**Status: exploratory, not TDD.** This document reports what happened running the finished
realtime pipeline (`juggletrack live`, `juggletrack export`) against real footage and a real
CoreML export attempt, and measures the two numbers the spec cares about most: live fps against
the ≥15fps floor / 30fps goal, and live-vs-offline event parity. Nothing in the realtime engine
was changed to make these numbers look better; a genuine, material parity gap was found and is
reported as measured (see §3).

**Machine/setup:** Apple M4 Pro, macOS, Python 3.12.13, `uv`-managed venv. Weights:
`/Users/andrew.follmann/personal-projects/juggling/models/juggletrack-v3/best.pt` throughout
(the fine-tuned v3 detector from Plan 3/turn 3-4). Detector config: `conf=0.05`, `imgsz=640`
(CLI defaults), `device=mps` for all fresh-detection runs. All raw artifacts referenced below
live under `/Users/andrew.follmann/personal-projects/juggling/outputs/plan4-bench/` (not
committed; recomputable from the commands in each section).

## 1. CoreML export + single-frame detector bench

`uv run juggletrack export models/juggletrack-v3/best.pt` **failed** on this machine, in two
stages:

1. First attempt: `coremltools` wasn't installed at all, and ultralytics' internal
   auto-pip-install fell over because the `uv`-managed venv has no `pip` module
   (`No module named pip`). Installed via `uv pip install "coremltools>=9.0"` (succeeded,
   `coremltools==9.0`) and retried.
2. Second attempt (coremltools present) still failed, now with a real conversion error:
   `TypeError: only 0-dimensional arrays can be converted to Python scalars`, raised inside
   coremltools' torch→MIL op conversion (the `int`/`_cast` op, node
   `model/10/m/0/attn/476`). Root cause is **version skew**: torch is 2.13.0 on this machine,
   and coremltools' own startup warning says *"Torch 2.13.0 has not been tested with
   coremltools ... Torch 2.7.0 is the most recent version that has been tested"* — compounded by
   numpy 2.5.1 (ultralytics' own CoreML exporter carries a comment that numpy ≥2.4.x breaks
   coremltools). Full log:
   `outputs/plan4-bench/detector-bench/coreml_export_attempt.log`. Fails fast (~3.4s), not a
   hang. **No `.mlpackage` was produced** — no product path to report.

Per the brief, this known-risk outcome is itself the finding: **on this machine's current
dependency stack, CoreML export through the ultralytics wrapper does not work**, independent of
juggletrack's own code. Continuing MPS-only below.

**Detector bench** (200 frames of af2.mp4, first 10 excluded as warmup, `YOLODetector` v3
weights, `imgsz=640`, `conf=0.05`; script + raw JSON:
`outputs/plan4-bench/detector-bench/mps_bench.json`):

| Backend | p50 ms/frame | p95 ms/frame | mean ms/frame | min / max |
|---|---|---|---|---|
| MPS (.pt) | 12.01 | 19.42 | 12.59 | 8.11 / 23.58 |
| CoreML (.mlpackage) | — | — | — | export failed, see above |

12ms/frame median implies ~83fps of detector throughput alone — well inside the fps budget below
once analysis/drawing overhead is added.

## 2. Live file-mode fps (the demo rehearsal)

Fresh detection, MPS, `--no-display`, one run per video (`juggletrack live <video> --model
models/juggletrack-v3/best.pt --no-display --out .../live/<stem>`):

| Video | Duration | Runs | Catches | Drops | fps | vs ≥15 floor | vs 30 goal |
|---|---|---|---|---|---|---|---|
| af2.mp4 | 35.6s | 4 | 39 | 4 | 36 | clears (2.4×) | clears |
| af1.mov | 40.0s | 4 | 53 | 1 | 44 | clears (2.9×) | clears |
| ss3_id_016.MP4 | 205.0s | 46 | 306 | 42 | 32 | clears (2.1×) | clears |
| outdoor holdout (PXL_20260716_191702756.mp4) | 14.7s | 1 | 9 | 0 | 47 | clears (3.1×) | clears |

**Every video clears both the ≥15fps floor and the 30fps goal by a comfortable margin**,
including the small-resolution but event-dense ss3_id_016 clip (46 runs / 306 catches worth of
analyzer churn) and the 1080×1920 phone-resolution outdoor holdout. fps is not the bottleneck for
this demo on this machine. (Run/catch/drop counts in this table are the fresh-detector numbers;
see §3 for why several of these numbers do not match offline `analyze` on the same clip.)

## 3. Parity table — and a material finding

The brief's exact-parity method: run `juggletrack analyze --save-intermediates` fresh (v3,
stride 1, imgsz 640) to get `detections.jsonl` + `analysis.json`, then replay that **same**
`detections.jsonl` through `juggletrack live --detections ... ` — eliminating the detector as a
variable so any difference is purely the online vs. offline algorithm.

| Video | Offline runs/catches/drops | Live runs/catches/drops | Same input? | Live fps |
|---|---|---|---|---|
| af2.mp4 | 2 / 20 / 1 | 4 / 39 / 4 | **yes** (replay) | 36 (fresh) / 79 (replay) |
| af1.mov | 3 / 51¹ / 1 | 4 / 53 / 1 | no (both fresh, separately run) | 44 |
| outdoor holdout | 1 / 6 / 0 | 1 / 9 / 0 | no (both fresh, separately run) | 47 |
| ss3_id_016.MP4 | **9 / 39 / 0** | **46 / 306 / 42** | **yes** (replay) | 32 (fresh) / 84 (replay) |

¹ af1's offline run 3 (23.30–24.16s, 0.86s span) reports 30 catches — almost certainly the same
"dense, clean v3 detections mint a burst of spurious short arcs" failure mode turn-3's findings
already documented for this model, not a new bug; flagged, not investigated further here
(out of scope for this task).

**Both same-input replays (af2, ss3) reproduced their fresh-detector Step-2 numbers exactly**
(af2: 4/39/4 both times; ss3: 46/306/42 both times). This confirms YOLO/MPS inference is
effectively deterministic on this machine, and — critically — that **the live/offline gap is not
detector noise. It is the online algorithm itself**, even though both paths share the same
`analyze_detections` code (the architecture's stated parity-by-construction claim).

**Investigated root cause** (`RealtimeAnalyzer.feed()` replayed frame-by-frame outside the CLI,
logging every `runs_completed`/`catches_total`/`drops_total` transition with its timestamp):
offline af2's run 2 spans **17.37s–26.69s (9.32s)** — longer than `RealtimeConfig.window_s =
8.0s`. The realtime engine's sliding window can never hold this run's full history at once, so as
its tail slides past the 8s cut, `detect_drops`'s `periodicity_collapse` signal (which needs
~0.75s of *no continuation* computed from arcs local to the run, inside whatever window is
currently live) never sees what the offline pass sees over the whole video: offline registers a
drop at t=30.18s and stops; the online engine instead keeps finding "live" arc activity in each
successive 8s window all the way to the true end of the video (35.6s), manufacturing **two extra
completed runs** and **~19 extra catches** in the same stretch offline calls one run + a drop.
This is **not** simply "a run still open at EOF that `finalize()` should have closed" — by the
time the divergence starts, `run_active` is already `False` well before EOF. It is a genuine
segmentation divergence for any run whose duration exceeds `window_s`, which is common for real,
steady cascades. ss3_id_016 makes this unmistakable at scale: its offline runs go up to ~14.4s
long (117.48–131.89s), and the realtime engine turns 9 real runs into 46, inventing **42 phantom
drops where the offline pass — run on the identical detections — found zero.**

The two videos without the exact replay (af1, outdoor holdout) still diverge in the same
direction (live over-counts runs/catches) but by a smaller margin, consistent with their shorter
individual run durations relative to `window_s`.

## 4. Freeze-horizon / analysis-cycle latency

Instrumented via `RealtimeState.last_analysis_ms`, replaying detections directly through
`RealtimeAnalyzer.feed()` (bypassing frame decode/draw) to isolate `analyze_detections()` cost
per cycle (cadence: every 3 frames, i.e. every ~100ms at ~30fps source):

| Video | Cycles | Mean | p50 | p95 | Max |
|---|---|---|---|---|---|
| af2.mp4 | 1069 | 27.11ms | 29.57ms | 40.08ms | 43.64ms |
| ss3_id_016.MP4 | 6148 | 35.09ms | 35.68ms | 36.98ms | 42.10ms |

Analysis compute (27–43ms) sits comfortably inside the ~100ms cadence budget — compute is not
the constraint here (matches §1's detector headroom). The user-facing event latency is
dominated by the **freeze horizon**, not analysis time: catches/throws are confirmed
`freeze_s=1.5s` after they occur (~1.5–1.6s total), drops after `drop_freeze_s=2.5s`
(~2.5–2.6s total) — inherent to the design's stability/latency trade, and a separate concern
from the run/catch-count parity bug in §3.

## 5. Webcam smoke (best-effort)

`uv run juggletrack live 0 --max-frames 300 --out .../webcam-smoke` failed exactly as expected
in this headless agent context:

```
OpenCV: not authorized to capture video (status 0), requesting...
OpenCV: camera failed to properly initialize!
ValueError: could not open source: 0
```

No attempt was made to fight macOS camera permissions (out of scope; agent sandboxes don't hold
camera entitlements). Full log: `outputs/plan4-bench/webcam-smoke/attempt.log`.

**User-facing demo command** (run on the user's own Mac, with the terminal/IDE granted camera
access in System Settings → Privacy & Security → Camera):

```
uv run juggletrack live 0 --model /Users/andrew.follmann/personal-projects/juggling/models/juggletrack-v3/best.pt --display
```

Press `q` to quit; drop `--max-frames` for an open-ended session; add `--out <dir>` to save
`live_session.json` at the end.

## 6. Caveats

- Parity numbers in §3 are per-session totals (`runs_completed`, `catches_total`,
  `drops_total`), not per-event alignment — a large gap could in principle hide compensating
  errors, but here the direction is consistent and one-sided (live always over-counts), and the
  root-cause trace in §3 ties the gap to a specific, understood mechanism rather than random
  noise.
- af1's spurious 30-catch run (footnote ¹) and ss3's run-1 `start_t = -1.02s` (a parabola-fit
  extrapolation artifact placing a run's start before the video begins) are both pre-existing
  offline-pipeline quirks unrelated to Plan 4; noted for transparency, not investigated further
  here.
- CoreML export failure is a dependency-version-skew problem (torch 2.13.0 vs. coremltools'
  tested ceiling of 2.7.0), not a juggletrack code defect; it may resolve itself on a future
  coremltools/ultralytics release, or with a pinned older torch in a dedicated export
  environment — neither was attempted here (out of scope for this bench task).
- All fps numbers are single-run, single-machine measurements (no repeated trials / variance
  bars) — adequate for a floor/goal comparison at this margin (2.1–3.1× headroom), not for
  fine-grained regression tracking.

## 7. Next steps — Kalman contingency verdict

**On this evidence, the Kalman contingency should be escalated from "documented fallback" to
"seriously reconsidered."** The architecture header's parity-by-construction argument
("same code path" as the offline analyzer) is compute-true but not output-true: sharing
`analyze_detections` guarantees the *analysis step* behaves identically given identical input,
but the realtime engine's input is a truncated, sliding 8-second buffer, not the whole session
the offline pass sees. Any run — or any drop signal needing lookback/lookahead — that spans more
than `window_s` diverges, and steady real-world juggling sessions routinely run longer than 8
seconds (this bench's own af2 run and all of ss3_id_016's 9 runs qualify). The failure mode is
not marginal: 4 vs 2 runs on a 36-second clip, 46 vs 9 runs and **42 phantom drops** on a
3.4-minute clip.

Two candidate fixes, in rough order of effort:

1. **Cheap mitigation, not a fix:** grow `window_s` well past typical run length (e.g. 20–30s).
   §4 showed only ~30–40% of the ~100ms cadence budget is used at `window_s=8`, so there's
   headroom before fps suffers, but this only pushes the failure threshold out — any run longer
   than the new window still diverges, and per-cycle cost grows with buffer size (untested here
   how far it scales).
2. **The documented contingency:** an actual online tracker (Kalman or otherwise) that maintains
   persistent per-ball/per-run state across the whole session instead of periodically
   re-deriving run/arc structure from a bounded window. This directly targets the observed
   mechanism (state loss at the window boundary), not just its symptom.

Given the fps floor/goal are cleared with wide margin (§2) and compute is not the constraint
(§1, §4), the case for the sliding-window shortcut was speed; the measured cost is **correctness**
on exactly the sessions a live demo is meant to showcase. This bench's recommendation: don't ship
the current window_s=8s default as "parity with offline" without a caveat, and treat the Kalman
(or persistent-state) rewrite as the next priority ahead of further CoreML/perf work.

## 8. Addendum — two in-architecture fix waves, and where the leak actually lives

Two fix attempts were run against §3's failure before accepting the contingency verdict. Together
they closed two of the three divergence mechanisms exactly and isolated the third to a stage
neither architecture can reach. All numbers below are the same-input ss3_id_016 replay
(offline reference **9 runs / 39 catches / 0 drops**).

**Wave 1 — left-edge window guard (committed, `dd01055`).** The freeze horizon guarded the
buffer's *right* edge but nothing guarded the *left*: arcs whose history was truncated by the
trailing edge of the 8s buffer look like catchless run-enders, manufacturing phantom drops and
churning run segmentation. The fix discards events/drops derived from arcs starting within
`edge_pad=0.5s` of the buffer's oldest sample, adds sticky-open liveness for `run_active`
(a recent arc keeps an open run open without re-clearing `min_arcs`), and smooths the hand line
with an EMA (α=0.2; inter-cycle swings >0.1 in `hand_line_y` were measured before it).
Result: **46/306/42 → 9/209/32 — runs now match offline exactly.** A new pinned test
(`test_parity_very_long_stream`, a 27s three-run sim spanning >3× `window_s`) holds runs and
drops exact against offline with a +4 catch allowance.

**Wave 2 — persistent arc registry + global symbolic replay (built, measured, reverted; not
committed).** Attribution from wave 1 showed the remaining catch leak was duplicate
confirmation: every ~100ms re-analysis re-derives arcs from scratch, refit event times jitter
past `event_match_tol=0.15s`, and one physical catch was observed confirmed 7 times across the
~60 cycles that see it. The candidate fix gave arcs persistent identity — matched across cycles
by apex signature (|Δapex_t|<0.12s, |Δapex_x|<0.08), confirmed once when fully interior to the
trustworthy region — and re-ran the cheap symbolic stages (events → runs → drops) over the
*entire* confirmed-arc history each cycle (~+1ms/cycle at 200 arcs; cost is not a constraint).
Result: **1/191/0 — drops now match offline exactly and duplicate confirmation is closed**, but
catches barely moved (209 → 191) and global replay over the denser registry bridged offline's
run gaps (9 → 1 runs). Neither wave's output dominates the other.

**Where the leak actually lives: windowed extraction itself.** The registry exposed the real
mechanism. On identical detections, whole-video `extract_arcs` produces 56 arcs on ss3; the
windowed path confirmed **194** — with ~85 of the excess in one t=16–69s stretch, and these are
*well-witnessed* fits (n_points 62–91), not noise the registry could gate out. Growing the
window doesn't converge to the offline answer: sweeping `window_s` 8→60s over that stretch moves
its catch count 83→51 against offline's 18 — asymptotically wrong. The reason is that
extraction's junk suppression is built on **global statistics**: static-cell occupancy computed
over the whole video and gravity-cohort pruning against the median of all arcs. A bounded window
re-derives both from local context, so dense in-hand/near-hand detections that whole-video
statistics would kill instead mint extra physically-plausible arcs — and no downstream registry,
dedup, or gating can repair a wrong arc decomposition (four such mitigations were measured; none
moved catches materially).

**Sharpened verdict.** The contingency is no longer optional hardening. Closing the residual gap
requires replacing windowed re-extraction with persistent online arc identity — the documented
Kalman/per-ball-tracker contingency — or making extraction's global statistics streaming-native
(incremental static-cell occupancy and gravity-cohort state maintained across the whole session).
Both are new scope beyond this plan. What ships today is honest about its envelope: exact parity
on short-to-moderate streams (the pinned sim holds runs/drops exact), exact run segmentation plus
a documented catch/drop over-count on multi-minute dense footage, and 2–3× the required fps.
