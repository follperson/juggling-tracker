# Juggletrack Plan 4: Realtime Engine + Live Demo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `juggletrack live` shows a juggler their performance in real time — balls, arcs, current-run catch count, drops, fps — from a webcam or file, with event outputs that provably match the offline analyzer.

**Architecture:** Sliding-window re-analysis (CONSCIOUS SPEC DEVIATION, decided by the controller with rationale): spec §3 prescribed a bespoke online two-mode Kalman tracker, designed when windowed re-analysis was presumed too slow; turn-4 profiling measured full `extract_arcs` on ~8s windows at ~0.01s, so the realtime engine instead re-runs the proven offline core (`analyze_detections`) over a rolling detection buffer every few frames and confirms events monotonically behind a freeze horizon. Parity with offline analysis is by construction (same code path), the entire tested event core is reused, and the two-mode Kalman remains a documented contingency if phone-class profiling later demands it. New pieces: shared drawing helpers (`pipeline/draw.py`), the `RealtimeAnalyzer` (`pipeline/realtime.py`, cv2-free), a live capture loop (`pipeline/live.py`), CoreML export, and the `live`/`export` CLI commands.

**Tech Stack:** Python 3.12, `uv`, existing juggletrack core, cv2 (capture/draw only), ultralytics (lazy; CoreML export), pydantic v2, pytest.

## Global Constraints

- Python 3.12; everything via `uv`. The event core stays cv2/torch-free; `pipeline/realtime.py` must be importable without cv2 (it consumes `Detection`s, not frames). cv2 allowed in `video/`, `pipeline/draw.py`, `pipeline/live.py`, `pipeline/overlay.py`. ultralytics only via lazy imports.
- Realtime target (spec §1): ≥15fps floor, 30fps goal on Apple Silicon; the detector dominates the budget — the analyzer layer must stay <5ms per re-analysis on 8s windows (assert in tests with a loose 50ms CI-safe bound, report measured numbers).
- Event semantics: NO retractions — once the live engine reports an event/count it never un-reports it (monotonicity is a tested property). Freeze-horizon confirmation: events become final only once they are `freeze_s` behind the stream head.
- Parity contract (tested): feeding a finite stream through `RealtimeAnalyzer` then `finalize()` yields catches within ±1, and drops/runs counts equal, vs `analyze_detections` on the same detections.
- Existing 190 tests (+2 deselected) stay green; never loosen an existing test. Weights-dependent tests marked `@pytest.mark.detector`.
- Conventional commits + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; PATH-SCOPED commits (`git commit -m ... -- <files>`); verify `git rev-parse --show-toplevel` is the worktree before git ops; `git show --stat HEAD` after each commit.
- Real assets: v3 weights `/Users/andrew.follmann/personal-projects/juggling/models/juggletrack-v3/best.pt`; videos in `/Users/andrew.follmann/personal-projects/juggling/data/raw/`.

## File Structure

```
src/juggletrack/pipeline/draw.py      # Task 1: shared drawing (dots, arc tails, HUD text)
src/juggletrack/pipeline/overlay.py   # Task 1: refactored to use draw.py (behavior-identical)
src/juggletrack/pipeline/realtime.py  # Task 2: RealtimeConfig, RealtimeState, RealtimeAnalyzer
src/juggletrack/pipeline/live.py      # Task 3: run_live capture loop + fps meter
src/juggletrack/train/export.py       # Task 4: export_coreml wrapper
src/juggletrack/cli.py                # Task 4: `live` + `export` commands
tests/test_draw.py                    # Task 1
tests/test_realtime.py                # Task 2
tests/test_live.py                    # Task 3
tests/test_export.py                  # Task 4 (stub-unit + detector-marked)
tests/test_cli.py                     # Task 4 (extend)
docs/superpowers/plans/2026-07-19-plan4-bench-findings.md  # Task 5 output
```

---

### Task 1: Shared drawing helpers (`pipeline/draw.py`) + overlay refactor

**Files:**
- Create: `src/juggletrack/pipeline/draw.py`
- Modify: `src/juggletrack/pipeline/overlay.py` (use the helpers; behavior-identical)
- Test: `tests/test_draw.py`

**Interfaces:**
- Consumes: `Arc` (`x_at/y_at`), `Detection`.
- Produces (consumed by Tasks 3 and by overlay.py):
  - `draw_hand_line(frame, hand_line_y: float) -> None`
  - `draw_detections(frame, dets: list[Detection]) -> None` (4px white circles)
  - `draw_arc_tails(frame, arcs: list[Arc], t: float, *, tail_s: float = 0.4) -> None` (green polylines, sample 1/60s, only arcs with `t_start <= t <= t_end + 0.1`)
  - `draw_hud(frame, text: str) -> None` (white 0.6-scale text at (10, 24))
  - `draw_drop_marker(frame, x: float, y: float) -> None` (red tilted cross, 24px, 3px thick)
  All mutate the BGR frame in place; colors module-level constants moved from overlay.py.

- [ ] **Step 1: Write the failing tests**

`tests/test_draw.py`:

```python
import numpy as np

from juggletrack.pipeline.draw import (
    draw_arc_tails,
    draw_detections,
    draw_drop_marker,
    draw_hand_line,
    draw_hud,
)
from juggletrack.types import Arc, Detection


def blank():
    return np.zeros((240, 320, 3), dtype=np.uint8)


def test_each_helper_draws_something():
    arc = Arc(id=0, t_start=0.0, t_end=1.1, ay=1.0, by=-1.1, cy=0.65,
              bx=0.1, cx=0.4, n_points=30, rmse=0.005)
    det = Detection(frame_idx=0, t=0.5, x=0.5, y=0.5)
    for fn, args in [
        (draw_hand_line, (0.65,)),
        (draw_detections, ([det],)),
        (draw_arc_tails, ([arc], 0.6)),
        (draw_hud, ("run 1  catches 5",)),
        (draw_drop_marker, (0.5, 0.8)),
    ]:
        frame = blank()
        fn(frame, *args)
        assert frame.sum() > 0, f"{fn.__name__} drew nothing"


def test_arc_tails_skip_inactive_arcs():
    arc = Arc(id=0, t_start=5.0, t_end=6.1, ay=1.0, by=-1.1, cy=0.65,
              bx=0.1, cx=0.4, n_points=30, rmse=0.005)
    frame = blank()
    draw_arc_tails(frame, [arc], t=1.0)  # long before the arc
    assert frame.sum() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_draw.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'juggletrack.pipeline.draw'`.

- [ ] **Step 3: Implement `draw.py` by EXTRACTING the drawing code from `overlay.py`**

Move (not copy) the color constants and the per-frame drawing arithmetic from `render_overlay`'s loop body into module functions in `src/juggletrack/pipeline/draw.py`:

```python
"""Shared frame-drawing helpers for the offline overlay and the live HUD."""
from __future__ import annotations

import numpy as np

from juggletrack.types import Arc, Detection

GREEN = (0, 200, 0)
YELLOW = (0, 220, 220)
RED = (0, 0, 255)
WHITE = (240, 240, 240)


def draw_hand_line(frame: np.ndarray, hand_line_y: float) -> None:
    import cv2

    h, w = frame.shape[:2]
    y = int(hand_line_y * h)
    cv2.line(frame, (0, y), (w, y), YELLOW, 1)


def draw_detections(frame: np.ndarray, dets: list[Detection]) -> None:
    import cv2

    h, w = frame.shape[:2]
    for d in dets:
        cv2.circle(frame, (int(d.x * w), int(d.y * h)), 4, WHITE, 1)


def draw_arc_tails(frame: np.ndarray, arcs: list[Arc], t: float, *, tail_s: float = 0.4) -> None:
    import cv2

    h, w = frame.shape[:2]
    for arc in arcs:
        if not (arc.t_start <= t <= arc.t_end + 0.1):
            continue
        t0 = max(arc.t_start, t - tail_s)
        ts = np.arange(t0, min(t, arc.t_end) + 1e-9, 1 / 60)
        pts = np.array([[int(arc.x_at(tt) * w), int(arc.y_at(tt) * h)] for tt in ts])
        if len(pts) >= 2:
            cv2.polylines(frame, [pts], False, GREEN, 2)


def draw_hud(frame: np.ndarray, text: str) -> None:
    import cv2

    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)


def draw_drop_marker(frame: np.ndarray, x: float, y: float) -> None:
    import cv2

    h, w = frame.shape[:2]
    cv2.drawMarker(frame, (int(x * w), int(y * h)), RED, cv2.MARKER_TILTED_CROSS, 24, 3)
```

(cv2 imported inside functions so `realtime.py` never transitively pulls it via type imports; note it in the module docstring.) Then rewrite `render_overlay`'s loop body to call these helpers — same order, same values; the HUD string construction and drop-window/active-run logic stay in overlay.py. Behavior-identical: all existing overlay tests must pass unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_draw.py tests/test_overlay.py -v && uv run pytest -q && uv run ruff check src tests`
Expected: draw tests pass; overlay tests UNCHANGED and green; full suite 192 passed, 2 deselected.

- [ ] **Step 5: Commit (path-scoped)**

```bash
git add src/juggletrack/pipeline/draw.py src/juggletrack/pipeline/overlay.py tests/test_draw.py
git commit -m "refactor: extract shared frame-drawing helpers for live HUD reuse

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" -- src/juggletrack/pipeline/draw.py src/juggletrack/pipeline/overlay.py tests/test_draw.py
```

---

### Task 2: `RealtimeAnalyzer` (`pipeline/realtime.py`) — the heart of the plan

**Files:**
- Create: `src/juggletrack/pipeline/realtime.py`
- Test: `tests/test_realtime.py`

**Interfaces:**
- Consumes: `analyze_detections(dets, config) -> SessionResult`, `AnalyzeConfig`, `Detection`, `Arc`.
- Produces (consumed by Tasks 3–4):
  - `RealtimeConfig(window_s=8.0, cadence_frames=3, freeze_s=1.5, drop_freeze_s=2.5, event_match_tol=0.15, analyze=AnalyzeConfig())`
  - `RealtimeState(t, catches_total, throws_total, drops_total, runs_completed, run_active, catches_current_run, hand_line_y, window_arcs: list[Arc], last_analysis_ms)`
  - `RealtimeAnalyzer(config=None)` with `feed(dets: list[Detection], t: float) -> RealtimeState` (call once per frame, in time order; `dets` may be empty) and `finalize() -> RealtimeState` (confirms everything pending; idempotent).

**Design (implement exactly):** keep a rolling buffer of detections trimmed to `window_s`; every `cadence_frames` feeds (and on `finalize`), run `analyze_detections` on the buffer. From each window result: confirm throws/catches whose `t <= now - freeze_s` and drops whose `t <= now - drop_freeze_s`, deduplicating against already-confirmed events of the same type within `event_match_tol` seconds (windows overlap, so the same physical event reappears — nearest-confirmed check, not exact match). Run bookkeeping avoids cross-window run identity entirely: a window run is "live" if its `end_t > now - freeze_s`; if any live run exists and no current run is open, open one (record its start); if none exists and a run is open, close it (`runs_completed += 1`); `catches_current_run` = confirmed catches with `t >= current run start`. `finalize()` = one last analysis with the freeze horizon pushed past the end (confirm all), then close any open run. Monotonicity is structural: confirmed lists only append.

- [ ] **Step 1: Write the failing tests**

`tests/test_realtime.py`:

```python
import sys
import subprocess
from collections import defaultdict

import pytest

from juggletrack.analyze import analyze_detections
from juggletrack.pipeline.realtime import RealtimeAnalyzer, RealtimeConfig
from juggletrack.sim import simulate_cascade


def frames_of(dets):
    by = defaultdict(list)
    for d in dets:
        by[d.frame_idx].append(d)
    return [(idx / 30.0, by.get(idx, [])) for idx in range(max(by) + 1)]


def stream(r, analyzer):
    states = []
    for t, dets in frames_of(r.detections):
        states.append(analyzer.feed(dets, t))
    states.append(analyzer.finalize())
    return states


def test_parity_short_clean_stream():
    r = simulate_cascade(n_throws=12, fps=30.0, noise=0.003, dropout=0.1, seed=1)
    offline = analyze_detections(r.detections)
    final = stream(r, RealtimeAnalyzer())[-1]
    assert final.catches_total == sum(run.catches for run in offline.runs)
    assert final.runs_completed == len(offline.runs)
    assert final.drops_total == len(offline.drops)


def test_parity_long_stream_exercises_trimming():
    r = simulate_cascade(n_throws=24, fps=30.0, noise=0.003, dropout=0.1, seed=2)
    assert r.run_end > 8.0, "fixture must outlast the analysis window"
    offline = analyze_detections(r.detections)
    final = stream(r, RealtimeAnalyzer())[-1]
    off_catches = sum(run.catches for run in offline.runs)
    assert abs(final.catches_total - off_catches) <= 1
    assert final.runs_completed == len(offline.runs)
    assert final.drops_total == len(offline.drops)


def test_parity_drop_stream():
    r = simulate_cascade(n_throws=20, fps=30.0, drop_at_throw=8, seed=1)
    offline = analyze_detections(r.detections)
    final = stream(r, RealtimeAnalyzer())[-1]
    assert final.drops_total == len(offline.drops) == 1
    off_catches = sum(run.catches for run in offline.runs)
    assert abs(final.catches_total - off_catches) <= 1


def test_counters_are_monotonic():
    r = simulate_cascade(n_throws=20, fps=30.0, drop_at_throw=8, seed=1)
    states = stream(r, RealtimeAnalyzer())
    for field in ("catches_total", "throws_total", "drops_total", "runs_completed"):
        vals = [getattr(s, field) for s in states]
        assert vals == sorted(vals), f"{field} regressed: not monotonic"


def test_confirmation_latency_bounded():
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    offline = analyze_detections(r.detections)
    from juggletrack.events.catches import derive_events

    _, catches = derive_events(offline.arcs, offline.hand_line_y)
    cfg = RealtimeConfig()
    analyzer = RealtimeAnalyzer(cfg)
    confirm_t: list[float] = []
    seen = 0
    for t, dets in frames_of(r.detections):
        s = analyzer.feed(dets, t)
        while seen < s.catches_total:
            confirm_t.append(t)
            seen += 1
    bound = cfg.freeze_s + cfg.cadence_frames / 30.0 + 0.25
    for ct, ev in zip(confirm_t, sorted(c.t for c in catches)[: len(confirm_t)]):
        assert ct - ev <= bound, f"catch at {ev:.2f} confirmed {ct - ev:.2f}s late"


def test_analysis_stays_fast_and_reports_timing():
    r = simulate_cascade(n_throws=24, fps=30.0, seed=3)
    states = stream(r, RealtimeAnalyzer())
    timings = [s.last_analysis_ms for s in states if s.last_analysis_ms > 0]
    assert timings, "no analysis timings recorded"
    mean_ms = sum(timings) / len(timings)
    assert mean_ms < 50, f"mean re-analysis {mean_ms:.1f}ms exceeds loose CI bound"


def test_realtime_module_is_cv2_free():
    code = ("import sys; import juggletrack.pipeline.realtime; "
            "sys.exit(1 if 'cv2' in sys.modules else 0)")
    proc = subprocess.run([sys.executable, "-c", code], cwd="src")
    assert proc.returncode == 0, "importing realtime pulled in cv2"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_realtime.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'juggletrack.pipeline.realtime'`.

- [ ] **Step 3: Implement `src/juggletrack/pipeline/realtime.py`**

```python
"""Realtime event engine: sliding-window re-analysis with freeze-horizon
confirmation.

CONSCIOUS SPEC DEVIATION (documented in the plan header): spec §3 called for
a bespoke online two-mode Kalman tracker; measured extraction cost (~0.01s
per 8s window) makes re-running the proven offline core affordable at frame
rate, giving offline parity by construction. The Kalman design remains the
contingency if phone-class profiling demands it.

This module must stay cv2-free: it consumes Detections, not frames.
"""
from __future__ import annotations

import time

from pydantic import BaseModel, Field

from juggletrack.analyze import AnalyzeConfig, analyze_detections
from juggletrack.events.catches import derive_events
from juggletrack.types import Arc, Detection


class RealtimeConfig(BaseModel):
    window_s: float = 8.0
    cadence_frames: int = 3
    freeze_s: float = 1.5
    drop_freeze_s: float = 2.5
    event_match_tol: float = 0.15
    analyze: AnalyzeConfig = Field(default_factory=AnalyzeConfig)


class RealtimeState(BaseModel):
    t: float
    catches_total: int = 0
    throws_total: int = 0
    drops_total: int = 0
    runs_completed: int = 0
    run_active: bool = False
    catches_current_run: int = 0
    hand_line_y: float = 0.0
    window_arcs: list[Arc] = Field(default_factory=list)
    last_analysis_ms: float = 0.0


class RealtimeAnalyzer:
    def __init__(self, config: RealtimeConfig | None = None):
        self.cfg = config or RealtimeConfig()
        self._buffer: list[Detection] = []
        self._frames_since_analysis = 0
        self._now = 0.0
        self._confirmed: dict[str, list[float]] = {"throw": [], "catch": [], "drop": []}
        self._runs_completed = 0
        self._run_start: float | None = None
        self._finalized = False
        self._last_state = RealtimeState(t=0.0)

    def feed(self, dets: list[Detection], t: float) -> RealtimeState:
        self._now = t
        self._buffer.extend(dets)
        cut = t - self.cfg.window_s
        if self._buffer and self._buffer[0].t < cut:
            self._buffer = [d for d in self._buffer if d.t >= cut]
        self._frames_since_analysis += 1
        if self._frames_since_analysis >= self.cfg.cadence_frames:
            self._frames_since_analysis = 0
            self._analyze(freeze=self.cfg.freeze_s, drop_freeze=self.cfg.drop_freeze_s)
        return self._state()

    def finalize(self) -> RealtimeState:
        if not self._finalized:
            self._analyze(freeze=-1.0, drop_freeze=-1.0)  # horizon past the end
            if self._run_start is not None:
                self._runs_completed += 1
                self._run_start = None
            self._finalized = True
        return self._state()

    def _analyze(self, *, freeze: float, drop_freeze: float) -> None:
        t0 = time.perf_counter()
        session = analyze_detections(list(self._buffer), self.cfg.analyze)
        throws, catches = derive_events(session.arcs, session.hand_line_y)

        self._confirm("throw", [e.t for e in throws], self._now - freeze)
        self._confirm("catch", [e.t for e in catches], self._now - freeze)
        self._confirm("drop", [d.t for d in session.drops], self._now - drop_freeze)

        live = any(r.end_t > self._now - freeze for r in session.runs)
        if live and self._run_start is None:
            starts = [r.start_t for r in session.runs if r.end_t > self._now - freeze]
            self._run_start = min(starts)
        elif not live and self._run_start is not None:
            self._runs_completed += 1
            self._run_start = None

        self._last_state = RealtimeState(
            t=self._now,
            catches_total=len(self._confirmed["catch"]),
            throws_total=len(self._confirmed["throw"]),
            drops_total=len(self._confirmed["drop"]),
            runs_completed=self._runs_completed,
            run_active=self._run_start is not None,
            catches_current_run=sum(
                1 for ct in self._confirmed["catch"]
                if self._run_start is not None and ct >= self._run_start
            ),
            hand_line_y=session.hand_line_y,
            window_arcs=session.arcs,
            last_analysis_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def _confirm(self, kind: str, times: list[float], horizon: float) -> None:
        known = self._confirmed[kind]
        for et in sorted(times):
            if et > horizon:
                continue
            if any(abs(et - k) <= self.cfg.event_match_tol for k in known):
                continue
            known.append(et)

    def _state(self) -> RealtimeState:
        return self._last_state.model_copy(update={"t": self._now})
```

- [ ] **Step 4: Run tests, tune honestly**

Run: `uv run pytest tests/test_realtime.py -v && uv run pytest -q && uv run ruff check src tests`
Expected: 7 new tests pass; full suite 199 passed, 2 deselected. Likely trouble spots and the honest responses: (a) long-stream parity off by >1 — inspect whether trimming cuts an unconfirmed event (raise `window_s` relation check: `freeze_s`+longest flight must fit; investigate, don't just widen tolerance); (b) run_active flapping on noisy streams — the `min(starts)` re-open may double-count runs; if `runs_completed` overshoots offline, dedup by requiring `now - last_close > 0.5s` before reopening counts as a NEW run (document what you shipped); (c) `_confirm`'s tol-based dedup can merge two real catches <0.15s apart — the sim's min inter-catch spacing is ~0.45s/3 balls ≈ 0.15s boundary: verify on the fixtures and report the closest real pair you observed.

- [ ] **Step 5: Commit (path-scoped)**

```bash
git add src/juggletrack/pipeline/realtime.py tests/test_realtime.py
git commit -m "feat: realtime engine via sliding-window re-analysis with freeze-horizon confirmation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" -- src/juggletrack/pipeline/realtime.py tests/test_realtime.py
```

---

### Task 3: Live capture loop (`pipeline/live.py`)

**Files:**
- Create: `src/juggletrack/pipeline/live.py`
- Test: `tests/test_live.py`

**Interfaces:**
- Consumes: `BallDetector`, `RealtimeAnalyzer`/`RealtimeConfig`/`RealtimeState`, draw helpers (Task 1), `VideoReader` conventions (but live uses cv2.VideoCapture directly — webcams aren't seekable files).
- Produces (consumed by Task 4):
  - `run_live(source: int | str, detector: BallDetector, *, config: RealtimeConfig | None = None, display: bool = True, max_frames: int | None = None, on_state: Callable[[RealtimeState, np.ndarray], None] | None = None) -> RealtimeState` — opens cv2.VideoCapture(source) (int = webcam index, str = file path; raise ValueError if unopened); per frame: PTS-or-index timestamp (same health rule as VideoReader — reuse by instantiating the reader when source is a str path; for webcams use `time.monotonic()` deltas from the first frame); detect → analyzer.feed → draw HUD (hand line, verified dots = detections assigned to `state.window_arcs` via `assign_detections`, arc tails, HUD text `run N | catches M | drops D | fps F`) → optional `cv2.imshow` when display (with `q` to quit) → `on_state` callback (testing hook). Maintains an EMA processing-fps meter shown in the HUD and returned via the final state's... fps is loop-level, not RealtimeState — return `(final_state, fps)`? Keep the interface: `run_live(...) -> tuple[RealtimeState, float]` (final state after `finalize()`, mean processing fps).

- [ ] **Step 1: Write the failing tests**

`tests/test_live.py`:

```python
import numpy as np
import pytest

from juggletrack.analyze import analyze_detections
from juggletrack.detect.fake import FakeDetector
from juggletrack.pipeline.live import run_live
from juggletrack.sim import simulate_cascade
from tests.helpers import write_test_video


@pytest.fixture()
def sim_video(tmp_path):
    sim = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    n_frames = max(d.frame_idx for d in sim.detections) + 1
    video = tmp_path / "live.mp4"
    write_test_video(video, n_frames=n_frames, fps=30.0)
    return sim, video


def test_run_live_file_source_headless_parity(sim_video):
    sim, video = sim_video
    final, fps = run_live(str(video), FakeDetector(sim.detections), display=False)
    offline = analyze_detections(sim.detections)
    assert final.catches_total == sum(r.catches for r in offline.runs)
    assert final.runs_completed == len(offline.runs)
    assert fps > 0


def test_run_live_max_frames_stops_early(sim_video):
    sim, video = sim_video
    states = []
    final, _ = run_live(
        str(video), FakeDetector(sim.detections), display=False,
        max_frames=30, on_state=lambda s, f: states.append(s),
    )
    assert len(states) == 30
    assert final.t <= 30 / 30.0 + 0.05


def test_run_live_on_state_receives_annotated_frames(sim_video):
    sim, video = sim_video
    captured = []
    run_live(str(video), FakeDetector(sim.detections), display=False,
             max_frames=40, on_state=lambda s, f: captured.append(f.copy()))
    assert captured and all(isinstance(f, np.ndarray) for f in captured)
    # at least one frame after warmup must carry HUD pixels (non-black rows at top)
    assert any(f[:30].sum() > 0 for f in captured[10:])


def test_run_live_bad_source_raises():
    from juggletrack.detect.fake import FakeDetector

    with pytest.raises(ValueError):
        run_live("/nonexistent/video.mp4", FakeDetector([]), display=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_live.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'juggletrack.pipeline.live'`.

- [ ] **Step 3: Implement `src/juggletrack/pipeline/live.py`**

```python
"""Live capture loop: webcam or file -> detector -> RealtimeAnalyzer -> HUD."""
from __future__ import annotations

import time
from collections.abc import Callable

import cv2
import numpy as np

from juggletrack.arcs.extract import assign_detections
from juggletrack.detect import BallDetector
from juggletrack.pipeline.draw import (
    draw_arc_tails,
    draw_detections,
    draw_hand_line,
    draw_hud,
)
from juggletrack.pipeline.realtime import RealtimeAnalyzer, RealtimeConfig, RealtimeState


def run_live(
    source: int | str,
    detector: BallDetector,
    *,
    config: RealtimeConfig | None = None,
    display: bool = True,
    max_frames: int | None = None,
    on_state: Callable[[RealtimeState, np.ndarray], None] | None = None,
) -> tuple[RealtimeState, float]:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"could not open source: {source!r}")
    fps_meta = cap.get(cv2.CAP_PROP_FPS) or 30.0
    is_file = isinstance(source, str)

    analyzer = RealtimeAnalyzer(config)
    idx = 0
    t_wall0 = time.monotonic()
    proc_fps = 0.0
    try:
        while True:
            if max_frames is not None and idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if is_file:
                msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                t = msec / 1000.0 if msec > 0 else idx / fps_meta
            else:
                t = time.monotonic() - t_wall0

            dets = detector.detect(frame, idx, t)
            state = analyzer.feed(dets, t)

            draw_hand_line(frame, state.hand_line_y)
            if dets:
                verified = assign_detections(dets, state.window_arcs)
                draw_detections(frame, [d for d, a in zip(dets, verified) if a != -1])
            draw_arc_tails(frame, state.window_arcs, t)
            elapsed = time.monotonic() - t_wall0
            proc_fps = (idx + 1) / elapsed if elapsed > 0 else 0.0
            run_txt = f"run {state.runs_completed + (1 if state.run_active else 0)}"
            draw_hud(frame, f"{run_txt}  catches {state.catches_current_run}"
                            f"  drops {state.drops_total}  fps {proc_fps:.0f}")

            if on_state is not None:
                on_state(state, frame)
            if display:
                cv2.imshow("juggletrack live", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            idx += 1
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
    return analyzer.finalize(), proc_fps
```

Note the file-source timestamp rule mirrors VideoReader's PTS-preference but simplified (no seam rebase — webcams use the wall clock and files usually have healthy PTS; comment this and reference reader.py for the full treatment).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_live.py -v && uv run pytest -q && uv run ruff check src tests`
Expected: 4 passed; full suite 203 passed, 2 deselected.

- [ ] **Step 5: Commit (path-scoped)**

```bash
git add src/juggletrack/pipeline/live.py tests/test_live.py
git commit -m "feat: live capture loop with realtime HUD

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" -- src/juggletrack/pipeline/live.py tests/test_live.py
```

---

### Task 4: CLI `live` + `export` commands, CoreML wrapper

**Files:**
- Create: `src/juggletrack/train/export.py`
- Modify: `src/juggletrack/cli.py`
- Test: `tests/test_export.py`, `tests/test_cli.py` (extend)

**Interfaces:**
- Produces:
  - `export_coreml(weights: str | Path, *, imgsz: int = 640, half: bool = True) -> Path` in `train/export.py` — lazy ultralytics; `YOLO(weights).export(format="coreml", imgsz=imgsz, half=half, nms=True)`; returns the `.mlpackage` path; `FileNotFoundError` if the export product is missing.
  - CLI `juggletrack export WEIGHTS [--imgsz 640] [--no-half]` — prints the output path.
  - CLI `juggletrack live [SOURCE] [--model PATH] [--detections JSONL] [--conf 0.05] [--imgsz 640] [--device STR] [--no-display] [--max-frames N] [--out DIR]` — SOURCE default "0" (webcam 0; numeric strings become int webcam indices, everything else is a file path); `--detections` replays a saved jsonl through `FakeDetector` (file sources only — testing/re-analysis mode, mirrors `analyze`); `--out` writes `live_session.json` (final `RealtimeState.model_dump_json`) into DIR. Echo the final summary line: `N runs, M catches, D drops @ F fps`.

- [ ] **Step 1: Write the failing tests**

`tests/test_export.py` (stub-unit via the fake-ultralytics pattern from tests/test_train.py, + one detector-marked real export):

```python
import sys
import types
from pathlib import Path

import pytest


def install_fake_ultralytics(monkeypatch, tmp_path, *, create_product=True):
    calls = {}

    class FakeYOLO:
        def __init__(self, weights):
            calls["weights"] = weights

        def export(self, **kw):
            calls["export_kwargs"] = kw
            product = tmp_path / "best.mlpackage"
            if create_product:
                product.mkdir()
            return str(product)

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    return calls


def test_export_coreml_plumbing(tmp_path, monkeypatch):
    calls = install_fake_ultralytics(monkeypatch, tmp_path)
    from juggletrack.train.export import export_coreml

    out = export_coreml(tmp_path / "best.pt", imgsz=640, half=True)
    assert out.exists() and out.suffix == ".mlpackage"
    kw = calls["export_kwargs"]
    assert kw["format"] == "coreml" and kw["imgsz"] == 640
    assert kw["half"] is True and kw["nms"] is True


def test_export_coreml_missing_product_raises(tmp_path, monkeypatch):
    install_fake_ultralytics(monkeypatch, tmp_path, create_product=False)
    from juggletrack.train.export import export_coreml

    with pytest.raises(FileNotFoundError):
        export_coreml(tmp_path / "best.pt")


@pytest.mark.detector
def test_export_coreml_real():
    from juggletrack.train.export import export_coreml

    weights = Path("/Users/andrew.follmann/personal-projects/juggling/models/juggletrack-v3/best.pt")
    if not weights.exists():
        pytest.skip("v3 weights not on this machine")
    out = export_coreml(weights)
    assert out.exists()
```

Append to `tests/test_cli.py`:

```python
def test_live_command_file_replay(workspace):
    from juggletrack.cli import app

    sim, video, dets, tmp = workspace
    out = tmp / "live_out"
    result = runner.invoke(app, [
        "live", str(video), "--detections", str(dets),
        "--no-display", "--out", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert "catches" in result.output and "fps" in result.output
    saved = json.loads((out / "live_session.json").read_text())
    assert saved["catches_total"] == 12
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_export.py tests/test_cli.py::test_live_command_file_replay -v`
Expected: FAIL — no `juggletrack.train.export`, and the CLI has no `live` command (exit code 2).

- [ ] **Step 3: Implement**

`src/juggletrack/train/export.py`:

```python
"""CoreML export for the ball detector (Mac realtime path; phone later uses
LiteRT — spec notes the Android target). ultralytics imported lazily."""
from __future__ import annotations

from pathlib import Path


def export_coreml(weights: str | Path, *, imgsz: int = 640, half: bool = True) -> Path:
    from ultralytics import YOLO  # lazy

    product = Path(YOLO(str(weights)).export(
        format="coreml", imgsz=imgsz, half=half, nms=True
    ))
    if not product.exists():
        raise FileNotFoundError(f"CoreML export missing: {product}")
    return product
```

Add to `src/juggletrack/cli.py` (following the existing lazy-import command pattern):

```python
@app.command()
def export(
    weights: Path = typer.Argument(..., exists=True, dir_okay=False),
    imgsz: int = typer.Option(640),
    half: bool = typer.Option(True, "--half/--no-half"),
) -> None:
    """Export detector weights to CoreML (.mlpackage) for the Mac live path."""
    from juggletrack.train.export import export_coreml

    typer.echo(f"exported: {export_coreml(weights, imgsz=imgsz, half=half)}")


@app.command()
def live(
    source: str = typer.Argument("0", help="Webcam index (digits) or video file path"),
    model: str = typer.Option(
        "/Users/andrew.follmann/personal-projects/juggling/models/juggletrack-v3/best.pt",
        help="Detector weights (.pt or .mlpackage)",
    ),
    detections: Path | None = typer.Option(
        None, exists=True, dir_okay=False,
        help="Replay saved detections.jsonl (file sources only; no detector run)",
    ),
    conf: float = typer.Option(0.05),
    imgsz: int = typer.Option(640),
    device: str | None = typer.Option(None),
    display: bool = typer.Option(True, "--display/--no-display"),
    max_frames: int | None = typer.Option(None),
    out: Path | None = typer.Option(None, help="Write live_session.json here"),
) -> None:
    """Live juggling tracker: webcam or file, realtime HUD."""
    from juggletrack.pipeline.live import run_live
    from juggletrack.pipeline.realtime import RealtimeConfig

    src: int | str = int(source) if source.isdigit() else source
    if detections is not None:
        from juggletrack.detect.fake import FakeDetector
        from juggletrack.pipeline.offline import load_detections_jsonl

        detector = FakeDetector(load_detections_jsonl(detections))
    else:
        from juggletrack.detect.yolo import YOLODetector

        detector = YOLODetector(model_path=model, conf=conf, imgsz=imgsz, device=device)

    final, fps = run_live(
        src, detector, config=RealtimeConfig(),
        display=display, max_frames=max_frames,
    )
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "live_session.json").write_text(final.model_dump_json(indent=2))
    typer.echo(
        f"{final.runs_completed} runs, {final.catches_total} catches, "
        f"{final.drops_total} drops @ {fps:.0f} fps"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_export.py tests/test_cli.py -v && uv run pytest -q && uv run ruff check src tests`
Expected: full suite ~207 passed, 3 deselected (new detector-marked export test joins the deselected set). Sanity: `uv run juggletrack live --help`.

- [ ] **Step 5: Commit (path-scoped)**

```bash
git add src/juggletrack/train/export.py src/juggletrack/cli.py tests/test_export.py tests/test_cli.py
git commit -m "feat: juggletrack live and export commands

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" -- src/juggletrack/train/export.py src/juggletrack/cli.py tests/test_export.py tests/test_cli.py
```

---

### Task 5: Field bench + offline-parity validation (exploratory — NOT TDD)

**Files:**
- Create: `docs/superpowers/plans/2026-07-19-plan4-bench-findings.md` (committed)
- Output (NOT committed): `/Users/andrew.follmann/personal-projects/juggling/outputs/plan4-bench/…`

Run the live pipeline against reality and measure the two numbers the spec demands: fps and live-vs-offline event parity.

- [ ] **Step 1: CoreML export + single-frame detector bench**

`uv run juggletrack export /Users/andrew.follmann/personal-projects/juggling/models/juggletrack-v3/best.pt` — record wall time and product path. Then a short python bench: 200 frames from af2.mp4 through (a) YOLODetector(.pt, device=mps), (b) YOLODetector(.mlpackage) — report ms/frame p50/p95 for each (first 10 frames excluded — warmup). If the .mlpackage load or predict fails through the ultralytics wrapper, record the error and continue with MPS only (a known-risk area; the finding matters either way).

- [ ] **Step 2: Live file-mode runs (the demo rehearsal)**

For af2.mp4, af1.mov, one Meschke clip (ss3_id_016), and the outdoor holdout: `uv run juggletrack live <video> --model <v3 .pt> --no-display --out /Users/andrew.follmann/personal-projects/juggling/outputs/plan4-bench/<stem>` (fresh detection, MPS; repeat af2 with the .mlpackage if step 1 succeeded). Record per video: fps, runs/catches/drops from live_session.json.

- [ ] **Step 3: Parity table**

For each video from step 2, compare live counts vs `juggletrack analyze` on the same detections... detection runs differ between invocations only by nondeterminism (should be none — same weights/frames); to be exact, run live with `--detections` replay of a saved analyze run's jsonl for af2 + ss3, giving a true same-input parity check: live-replay counts vs offline analysis.json counts. Table: video × {offline catches/runs/drops, live catches/runs/drops, fps}. Parity deviations beyond ±1 catch or any run/drop mismatch get investigated (freeze-horizon interaction with the video's ending is the likely suspect — a run still open at EOF must be closed by finalize, verify).

- [ ] **Step 4: The real thing (best-effort)**

Attempt a ~10s webcam smoke: `uv run juggletrack live 0 --max-frames 300 --out .../webcam-smoke` — in this headless agent context the webcam will likely be unavailable or permission-blocked; if so record the exact error and state that the user runs this command themselves for the true demo (give them the exact command with display on). Do not fight macOS camera permissions.

- [ ] **Step 5: Findings + commit**

`docs/superpowers/plans/2026-07-19-plan4-bench-findings.md`: detector bench table (MPS vs CoreML), live fps per video vs the ≥15fps floor / 30 goal, parity table, freeze-horizon latency observed (from RealtimeState timing if instrumented in step 3's replay), webcam smoke outcome + the user-facing demo command, caveats, next steps (Kalman contingency verdict: needed or not on this evidence). Commit doc only (path-scoped): `docs: plan 4 bench — live fps and offline parity` + trailer.

---

## Plan Self-Review Notes (already applied)

- **Spec coverage:** spec §1 realtime Mac demo = Tasks 3–5; §3 realtime engine = Task 2 (with the documented architecture deviation and Kalman contingency); §4 CoreML-on-Mac preference = Tasks 4–5 (with MPS fallback measured); §8 `live` CLI = Task 4. The two-mode Kalman tracker is deliberately NOT implemented — deviation is stated in the header, justified by measurement, and Task 5's findings must render a verdict on whether the contingency is needed.
- **Type consistency:** `RealtimeState` fields used by Task 3's HUD match Task 2's model; `run_live` returns `tuple[RealtimeState, float]` and Task 4's CLI destructures accordingly; `FakeDetector`/`load_detections_jsonl` signatures match the existing codebase; draw helpers' signatures match between Tasks 1 and 3.
- **Known risks, stated where they bite:** Task 2 step 4 lists the three likely failure modes with honest responses; Task 4's CoreML export through ultralytics may fail on this machine (Task 5 step 1 explicitly continues with MPS and records it); webcam access in an agent context is expected to fail (Task 5 step 4 hands the command to the user rather than fighting permissions).
- **Placeholder scan:** clean — complete code in every code step; Task 5 is exploratory with exact commands.


