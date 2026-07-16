# Juggletrack Plan 1: Event Core on Synthetic Data — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and prove the entire juggling event core (parabolic arc extraction → catches/throws → runs → drops) against a synthetic 3-ball-cascade simulator, with zero ML dependencies.

**Architecture:** Pure-logic layer of the `juggletrack` package per the approved spec (`docs/superpowers/specs/2026-07-15-juggling-tracker-rebuild-design.md`). Detections (from any source) flow through: global arc extraction (greedy fragment linking + EM refinement) → hand-line estimation from arc statistics → sub-frame throw/catch events from parabola coefficients → run chaining via arc overlap + periodicity validation → three-signal drop detection. A deterministic cascade simulator provides ground truth for every test.

**Tech Stack:** Python 3.12, `uv`, pydantic v2, numpy, scipy, pytest. No torch/opencv/ultralytics in this plan.

**Plan sequence context:** This is Plan 1 of 5. Plan 2 (offline video pipeline) consumes this plan's `extract_arcs`/`analyze_detections` API with real detections. Nothing here may import cv2 or torch.

## Global Constraints

- Python 3.12; all package management via `uv` (`uv sync`, `uv run pytest`, `uv add`). Never pip/poetry.
- New code lives in `src/juggletrack/`; the old prototype moves to `legacy/` untouched and is never imported.
- Coordinates are normalized to [0,1]; **y increases downward** (screen convention). Gravity is positive.
- Times are seconds (float). Frame indices are ints. Sub-frame event times are legal and expected.
- All schemas carry `SCHEMA_VERSION = "1.0"` and serialize via pydantic v2 (`model_dump_json`).
- All randomness is seeded (`numpy.random.default_rng(seed)`); every test is deterministic.
- Core modules (`types`, `sim`, `arcs`, `events`) may depend only on numpy, scipy, pydantic.
- Commit after every green test cycle. Conventional-commit messages, ending with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## File Structure

```
pyproject.toml                        # rewritten for juggletrack (uv, py3.12)
legacy/jugglecount/                   # old src/jugglecount moved here verbatim
legacy/tests/                         # old tests moved here
legacy/pyproject.toml                 # old pyproject preserved for reference
src/juggletrack/__init__.py
src/juggletrack/types.py              # Task 2: all data contracts
src/juggletrack/sim.py                # Task 3: synthetic cascade generator
src/juggletrack/arcs/__init__.py
src/juggletrack/arcs/fit.py           # Task 4: weighted parabola fit + Arc quality
src/juggletrack/arcs/extract.py       # Task 5: fragments → seeds → EM → arcs
src/juggletrack/events/__init__.py
src/juggletrack/events/handline.py    # Task 6: hand line from arc statistics
src/juggletrack/events/catches.py     # Task 6: throw/catch events from arcs
src/juggletrack/events/runs.py        # Task 7: run chaining + period estimation
src/juggletrack/events/periodicity.py # Task 7: airborne-count autocorrelation
src/juggletrack/events/drops.py       # Task 8: three-signal drop detector
src/juggletrack/analyze.py            # Task 9: detections → SessionResult
tests/test_types.py
tests/test_sim.py
tests/test_fit.py
tests/test_extract.py
tests/test_handline.py
tests/test_catches.py
tests/test_runs.py
tests/test_periodicity.py
tests/test_drops.py
tests/test_analyze.py
```

---

### Task 1: Scaffolding — retire the prototype, stand up the new package

**Files:**
- Create: `pyproject.toml` (replace), `src/juggletrack/__init__.py`, `tests/test_types.py` (smoke only)
- Move: `src/jugglecount/` → `legacy/jugglecount/`, `tests/test_live_processor.py` → `legacy/tests/test_live_processor.py`, old `pyproject.toml` → `legacy/pyproject.toml`

**Interfaces:**
- Produces: importable `juggletrack` package; `uv run pytest` works.

- [ ] **Step 1: Move the prototype to legacy/**

```bash
mkdir -p legacy/tests
git mv src/jugglecount legacy/jugglecount
git mv tests/test_live_processor.py legacy/tests/test_live_processor.py
git mv pyproject.toml legacy/pyproject.toml
```

- [ ] **Step 2: Write the new pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "juggletrack"
version = "0.1.0"
description = "Juggling run/catch/drop tracker: physics-first video analysis"
requires-python = ">=3.12"
dependencies = [
    "numpy>=2.0",
    "scipy>=1.14",
    "pydantic>=2.7",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.5",
]

[tool.hatch.build.targets.wheel]
packages = ["src/juggletrack"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

- [ ] **Step 3: Create the package and a smoke test**

`src/juggletrack/__init__.py`:

```python
"""juggletrack: physics-first juggling run/catch/drop tracking."""

SCHEMA_VERSION = "1.0"
```

`tests/test_types.py`:

```python
from juggletrack import SCHEMA_VERSION


def test_package_imports():
    assert SCHEMA_VERSION == "1.0"
```

- [ ] **Step 4: Sync and run the smoke test**

Run: `uv sync && uv run pytest tests/test_types.py -v`
Expected: PASS (1 passed). If `uv sync` complains about the missing lockfile for legacy poetry files, delete `poetry.lock`: `git rm poetry.lock` (it belongs to the retired prototype).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: retire jugglecount prototype to legacy/, scaffold juggletrack package

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Data contracts (`types.py`)

**Files:**
- Create: `src/juggletrack/types.py`
- Test: `tests/test_types.py` (extend)

**Interfaces:**
- Produces (consumed by every later task):
  - `Detection(frame_idx: int, t: float, x: float, y: float, w: float = 0.0, h: float = 0.0, confidence: float = 1.0)`
  - `Arc(id, t_start, t_end, ay, by, cy, bx, cx, n_points, rmse)` with methods `y_at(t)`, `x_at(t)`, `vy_at(t)`, `apex_t()`, `apex_y()`
  - `ThrowEvent(t, x, arc_id)`, `CatchEvent(t, x, arc_id)`
  - `DropEvent(t, x, arc_id: int | None, signals: list[str])`
  - `Run(start_t, end_t, catches, throws, arc_ids, end_reason, period_s, quality)`
  - `SessionResult(schema_version, runs, drops, arcs, hand_line_y, meta)`

- [ ] **Step 1: Write failing tests for Arc math and serialization**

Append to `tests/test_types.py`:

```python
import json

import pytest

from juggletrack.types import Arc, Detection, Run, SessionResult


def make_arc() -> Arc:
    # y(dt) = 1.0*dt^2 - 1.1*dt + 0.65  (down-positive: starts at hand, rises, returns)
    # vy(dt) = 2*dt - 1.1  -> apex at dt = 0.55, flight ends at dt = 1.1
    return Arc(
        id=1, t_start=10.0, t_end=11.1,
        ay=1.0, by=-1.1, cy=0.65, bx=0.16, cx=0.41,
        n_points=30, rmse=0.004,
    )


def test_arc_y_endpoints():
    arc = make_arc()
    assert arc.y_at(10.0) == pytest.approx(0.65)
    assert arc.y_at(11.1) == pytest.approx(0.65)  # symmetric flight returns to launch height


def test_arc_apex():
    arc = make_arc()
    assert arc.apex_t() == pytest.approx(10.55)
    # apex height above hand: v0^2/(2g) with v0=1.1, g=2*ay=2.0 -> 0.3025
    assert arc.apex_y() == pytest.approx(0.65 - 0.3025)
    assert arc.apex_y() < arc.y_at(10.0)  # apex is above (smaller y) than endpoints


def test_arc_velocity_sign():
    arc = make_arc()
    assert arc.vy_at(10.0) < 0  # rising (y shrinking) at throw
    assert arc.vy_at(11.1) > 0  # falling at catch


def test_arc_x_linear():
    arc = make_arc()
    assert arc.x_at(10.0) == pytest.approx(0.41)
    assert arc.x_at(11.1) == pytest.approx(0.41 + 0.16 * 1.1)


def test_session_result_json_roundtrip():
    arc = make_arc()
    run = Run(start_t=10.0, end_t=15.0, catches=9, throws=10, arc_ids=[1],
              end_reason="drop", period_s=0.45, quality=0.8)
    sr = SessionResult(runs=[run], drops=[], arcs=[arc], hand_line_y=0.65)
    blob = sr.model_dump_json()
    back = SessionResult.model_validate(json.loads(blob))
    assert back == sr
    assert back.schema_version == "1.0"


def test_detection_defaults():
    d = Detection(frame_idx=3, t=0.1, x=0.5, y=0.6)
    assert d.confidence == 1.0 and d.w == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_types.py -v`
Expected: FAIL with `ImportError: cannot import name 'Arc' from 'juggletrack.types'` (module doesn't exist yet).

- [ ] **Step 3: Implement `src/juggletrack/types.py`**

```python
"""Data contracts for juggletrack. Pure pydantic models — no cv2/torch imports.

Conventions: normalized [0,1] coordinates, y increases DOWNWARD, times in seconds.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from juggletrack import SCHEMA_VERSION


class Detection(BaseModel):
    """One ball detection in one frame (source-agnostic: detector, simulator, or label)."""

    frame_idx: int
    t: float
    x: float
    y: float
    w: float = 0.0
    h: float = 0.0
    confidence: float = 1.0


class Arc(BaseModel):
    """A ballistic flight segment.

    y(t) = ay*dt^2 + by*dt + cy,  x(t) = bx*dt + cx,  where dt = t - t_start.
    ay ≈ g/2 > 0 in down-positive normalized units.
    """

    id: int
    t_start: float
    t_end: float
    ay: float
    by: float
    cy: float
    bx: float
    cx: float
    n_points: int
    rmse: float

    def y_at(self, t: float) -> float:
        dt = t - self.t_start
        return self.ay * dt * dt + self.by * dt + self.cy

    def x_at(self, t: float) -> float:
        return self.bx * (t - self.t_start) + self.cx

    def vy_at(self, t: float) -> float:
        return 2.0 * self.ay * (t - self.t_start) + self.by

    def apex_t(self) -> float:
        """Time of the arc's highest point (vertex of the parabola)."""
        if self.ay == 0.0:
            return self.t_start
        return self.t_start - self.by / (2.0 * self.ay)

    def apex_y(self) -> float:
        return self.y_at(self.apex_t())

    def duration(self) -> float:
        return self.t_end - self.t_start


class ThrowEvent(BaseModel):
    t: float
    x: float
    arc_id: int


class CatchEvent(BaseModel):
    t: float
    x: float
    arc_id: int


class DropEvent(BaseModel):
    t: float
    x: float
    arc_id: int | None
    signals: list[str] = Field(default_factory=list)


class Run(BaseModel):
    start_t: float
    end_t: float
    catches: int
    throws: int
    arc_ids: list[int] = Field(default_factory=list)
    end_reason: Literal["drop", "stop", "video_end"] = "stop"
    period_s: float | None = None
    quality: float = 0.0


class SessionResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    runs: list[Run] = Field(default_factory=list)
    drops: list[DropEvent] = Field(default_factory=list)
    arcs: list[Arc] = Field(default_factory=list)
    hand_line_y: float = 0.0
    meta: dict[str, float | int | str] = Field(default_factory=dict)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_types.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/juggletrack/types.py tests/test_types.py
git commit -m "feat: core data contracts (Detection, Arc, events, Run, SessionResult)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Synthetic cascade simulator (`sim.py`)

**Files:**
- Create: `src/juggletrack/sim.py`
- Test: `tests/test_sim.py`

**Interfaces:**
- Consumes: `Detection` from Task 2.
- Produces (the ground-truth oracle for every later task):
  - `CascadeParams(n_balls=3, period_s=0.45, dwell_s=0.25, hand_y=0.65, hand_sep=0.18, center_x=0.5, g=2.0, floor_y=0.92, restitution=0.35)` with derived properties `flight_s` (= `n_balls*period_s - dwell_s`) and `v0` (= `g*flight_s/2`)
  - `SimResult(detections, throw_times, catch_times, missed_catch_t, drop_t, run_start, run_end, params)`
  - `simulate_cascade(n_throws=20, fps=30.0, params=None, noise=0.0, dropout=0.0, drop_at_throw=None, include_held=False, false_positives_per_frame=0.0, seed=0) -> SimResult`

**Physics being simulated** (so the implementer can verify by hand): throws happen every `period_s`, alternating hands (even index = left). A thrown ball is airborne for `flight_s`, following `y(dt) = hand_y - v0*dt + 0.5*g*dt²` (down-positive, so it rises then falls, returning to `hand_y` at `dt = flight_s`) and moving linearly in x from throw hand to the opposite hand. If `drop_at_throw=k`, flight k is never caught: its parabola continues to `floor_y`, one bounce with `restitution`, then the ball rests on the floor; throws scheduled after the missed catch time are cancelled (the juggler stops).

- [ ] **Step 1: Write failing tests**

`tests/test_sim.py`:

```python
import numpy as np
import pytest

from juggletrack.sim import CascadeParams, simulate_cascade


def test_derived_params():
    p = CascadeParams()
    assert p.flight_s == pytest.approx(3 * 0.45 - 0.25)  # 1.10
    assert p.v0 == pytest.approx(2.0 * 1.10 / 2.0)       # 1.10


def test_clean_run_ground_truth():
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    assert len(r.throw_times) == 12
    assert len(r.catch_times) == 12
    p = r.params
    for th, ca in zip(r.throw_times, r.catch_times):
        assert ca == pytest.approx(th + p.flight_s)
    assert r.throw_times[1] - r.throw_times[0] == pytest.approx(p.period_s)
    assert r.drop_t is None and r.missed_catch_t is None
    assert r.run_start == pytest.approx(r.throw_times[0])
    assert r.run_end == pytest.approx(r.catch_times[-1])


def test_detections_geometry():
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    p = r.params
    assert len(r.detections) > 0
    xs = np.array([d.x for d in r.detections])
    ys = np.array([d.y for d in r.detections])
    assert xs.min() >= 0.0 and xs.max() <= 1.0
    assert ys.min() >= 0.0 and ys.max() <= 1.0
    # airborne balls never go below hand level in a clean run
    assert ys.max() <= p.hand_y + 0.02
    # apex reached: hand_y - v0^2/(2g) = 0.65 - 0.3025
    assert ys.min() == pytest.approx(p.hand_y - p.v0**2 / (2 * p.g), abs=0.02)


def test_airborne_ball_count_bounded():
    r = simulate_cascade(n_throws=20, fps=30.0, seed=1)
    from collections import Counter

    per_frame = Counter(d.frame_idx for d in r.detections)
    assert max(per_frame.values()) <= 3  # never more than 3 balls
    # cascade with flight 1.1s / period 0.45s keeps 2-3 airborne mid-run
    mid = [c for f, c in per_frame.items() if 3.0 < f / 30.0 < 6.0]
    assert min(mid) >= 2


def test_drop_injection():
    r = simulate_cascade(n_throws=20, fps=30.0, drop_at_throw=8, seed=1)
    assert r.missed_catch_t == pytest.approx(r.throw_times[8] + r.params.flight_s)
    assert r.drop_t is not None and r.drop_t > r.missed_catch_t
    # juggler stops: fewer throws than requested
    assert len(r.throw_times) < 20
    # dropped ball produces floor-region detections
    floor_dets = [d for d in r.detections if d.y > r.params.hand_y + 0.1]
    assert len(floor_dets) > 0
    assert r.run_end == pytest.approx(r.missed_catch_t)


def test_noise_dropout_determinism():
    a = simulate_cascade(n_throws=10, noise=0.005, dropout=0.2, seed=7)
    b = simulate_cascade(n_throws=10, noise=0.005, dropout=0.2, seed=7)
    assert a.detections == b.detections  # same seed → identical
    clean = simulate_cascade(n_throws=10, seed=7)
    assert len(a.detections) < len(clean.detections)  # dropout removed some


def test_false_positives():
    r = simulate_cascade(n_throws=10, false_positives_per_frame=0.5, seed=3)
    clean = simulate_cascade(n_throws=10, seed=3)
    assert len(r.detections) > len(clean.detections)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sim.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'juggletrack.sim'`.

- [ ] **Step 3: Implement `src/juggletrack/sim.py`**

```python
"""Deterministic 3-ball-cascade simulator: ground truth for the event core.

Down-positive normalized coordinates. A flight is y(dt) = hand_y - v0*dt + g*dt²/2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from juggletrack.types import Detection


class CascadeParams(BaseModel):
    n_balls: int = 3
    period_s: float = 0.45
    dwell_s: float = 0.25
    hand_y: float = 0.65
    hand_sep: float = 0.18
    center_x: float = 0.5
    g: float = 2.0
    floor_y: float = 0.92
    restitution: float = 0.35

    @property
    def flight_s(self) -> float:
        return self.n_balls * self.period_s - self.dwell_s

    @property
    def v0(self) -> float:
        return self.g * self.flight_s / 2.0

    def hand_x(self, hand: int) -> float:
        """hand 0 = left, 1 = right."""
        side = -1.0 if hand == 0 else 1.0
        return self.center_x + side * self.hand_sep / 2.0


class SimResult(BaseModel):
    detections: list[Detection]
    throw_times: list[float]
    catch_times: list[float]
    missed_catch_t: float | None
    drop_t: float | None
    run_start: float
    run_end: float
    params: CascadeParams


@dataclass
class _Flight:
    t0: float
    x0: float
    x1: float
    dur: float
    caught: bool

    def pos(self, t: float, p: CascadeParams) -> tuple[float, float]:
        dt = t - self.t0
        y = p.hand_y - p.v0 * dt + 0.5 * p.g * dt * dt
        x = self.x0 + (self.x1 - self.x0) * dt / self.dur
        return x, y


def simulate_cascade(
    n_throws: int = 20,
    fps: float = 30.0,
    params: CascadeParams | None = None,
    noise: float = 0.0,
    dropout: float = 0.0,
    drop_at_throw: int | None = None,
    include_held: bool = False,
    false_positives_per_frame: float = 0.0,
    seed: int = 0,
) -> SimResult:
    p = params or CascadeParams()
    rng = np.random.default_rng(seed)
    t_first = 0.5  # lead-in before first throw

    # --- schedule flights -------------------------------------------------
    missed_catch_t: float | None = None
    if drop_at_throw is not None:
        missed_catch_t = t_first + drop_at_throw * p.period_s + p.flight_s

    flights: list[_Flight] = []
    throw_times: list[float] = []
    catch_times: list[float] = []
    for i in range(n_throws):
        t0 = t_first + i * p.period_s
        if missed_catch_t is not None and t0 > missed_catch_t:
            break  # juggler noticed the drop and stopped throwing
        hand = i % 2
        caught = drop_at_throw is None or i != drop_at_throw
        flights.append(_Flight(t0=t0, x0=p.hand_x(hand), x1=p.hand_x(1 - hand),
                               dur=p.flight_s, caught=caught))
        throw_times.append(t0)
        if caught:
            catch_times.append(t0 + p.flight_s)

    # --- dropped-ball extension: fall to floor, one bounce, rest ----------
    drop_t: float | None = None
    drop_segments: list[tuple[float, float, _Flight]] = []  # (t_from, t_to, flight-like)
    rest_from: float | None = None
    rest_x: float | None = None
    if drop_at_throw is not None:
        f = flights[drop_at_throw]
        # solve hand_y - v0*dt + g*dt²/2 = floor_y for dt > flight_s
        disc = p.v0**2 + 2.0 * p.g * (p.floor_y - p.hand_y)
        dt_floor = (p.v0 + math.sqrt(disc)) / p.g
        drop_t = f.t0 + dt_floor
        vx = (f.x1 - f.x0) / f.dur
        x_floor = f.x0 + vx * dt_floor
        drop_segments.append((f.t0 + f.dur, drop_t, f))  # continuation below hand line
        # bounce: rises from floor with v_b, returns after 2*v_b/g
        vy_floor = -p.v0 + p.g * dt_floor            # downward speed at impact
        v_b = p.restitution * vy_floor
        bounce_dur = 2.0 * v_b / p.g
        bounce = _Flight(t0=drop_t, x0=x_floor, x1=x_floor + vx * bounce_dur * 0.3,
                         dur=bounce_dur, caught=False)
        rest_from = drop_t + bounce_dur
        rest_x = bounce.x1
        drop_segments.append((drop_t, rest_from, _BounceSeg(bounce, v_b, p)))

    run_start = throw_times[0]
    run_end = missed_catch_t if missed_catch_t is not None else catch_times[-1]
    t_max = (drop_t + 1.0 if drop_t is not None else run_end) + 0.5

    # --- render detections frame by frame ----------------------------------
    detections: list[Detection] = []
    n_frames = int(t_max * fps) + 1
    for f_idx in range(n_frames):
        t = f_idx / fps
        positions: list[tuple[float, float]] = []
        for fl in flights:
            if fl.t0 <= t <= fl.t0 + fl.dur:
                positions.append(fl.pos(t, p))
        for t_from, t_to, seg in drop_segments:
            if t_from < t <= t_to:
                positions.append(seg.pos(t, p))
        if rest_from is not None and t > rest_from:
            positions.append((rest_x, p.floor_y))
        if include_held:
            for hand in (0, 1):
                if _hand_holds_ball(t, hand, flights, p):
                    positions.append((p.hand_x(hand), p.hand_y))
        n_fp = rng.poisson(false_positives_per_frame)
        for _ in range(n_fp):
            positions.append((float(rng.uniform()), float(rng.uniform())))

        for x, y in positions:
            if dropout > 0.0 and rng.random() < dropout:
                continue
            if noise > 0.0:
                x += float(rng.normal(0.0, noise))
                y += float(rng.normal(0.0, noise))
            x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
            detections.append(Detection(frame_idx=f_idx, t=t, x=x, y=y))

    return SimResult(
        detections=detections, throw_times=throw_times, catch_times=catch_times,
        missed_catch_t=missed_catch_t, drop_t=drop_t,
        run_start=run_start, run_end=run_end, params=p,
    )


class _BounceSeg:
    """Bounce parabola starting at the floor with upward speed v_b."""

    def __init__(self, fl: _Flight, v_b: float, p: CascadeParams):
        self.fl, self.v_b, self.p = fl, v_b, p

    def pos(self, t: float, p: CascadeParams) -> tuple[float, float]:
        dt = t - self.fl.t0
        y = p.floor_y - self.v_b * dt + 0.5 * p.g * dt * dt
        x = self.fl.x0 + (self.fl.x1 - self.fl.x0) * dt / self.fl.dur
        return x, min(y, p.floor_y)


def _hand_holds_ball(t: float, hand: int, flights: list[_Flight], p: CascadeParams) -> bool:
    """A hand holds a ball between catching one flight and throwing the next."""
    last_catch = None
    next_throw = None
    for fl in flights:
        thrown_from = 0 if fl.x0 < p.center_x else 1
        caught_by = 1 - thrown_from
        if fl.caught and caught_by == hand and fl.t0 + fl.dur <= t:
            last_catch = max(last_catch or -1.0, fl.t0 + fl.dur)
        if thrown_from == hand and fl.t0 >= t:
            next_throw = min(next_throw or 1e9, fl.t0)
    return last_catch is not None and (next_throw is None or t < next_throw)
```

Note the `drop_segments` list holds objects exposing `.pos(t, p)` — `_Flight` for the below-hand-line continuation and `_BounceSeg` for the bounce. The continuation segment reuses the original `_Flight` unchanged (its parabola remains valid past `dur`); only the time window `(t_from, t_to]` differs.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sim.py -v`
Expected: PASS (7 passed). If `test_detections_geometry` fails on the apex assertion, check the frame rate actually samples near the apex (at 30fps and flight 1.1s it does).

- [ ] **Step 5: Commit**

```bash
git add src/juggletrack/sim.py tests/test_sim.py
git commit -m "feat: deterministic 3-ball cascade simulator with drop injection

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Parabola fitting (`arcs/fit.py`)

**Files:**
- Create: `src/juggletrack/arcs/__init__.py` (empty), `src/juggletrack/arcs/fit.py`
- Test: `tests/test_fit.py`

**Interfaces:**
- Consumes: `Arc`, `Detection` from Task 2.
- Produces (consumed by Task 5's extractor and Plan 2's offline pipeline):
  - `points_array(dets: list[Detection]) -> np.ndarray` — shape (N, 4) columns `[t, x, y, confidence]`, sorted by t
  - `fit_arc(arr: np.ndarray, arc_id: int = -1) -> Arc` — weighted least-squares; requires N ≥ 3, raises `ValueError` otherwise
  - `y_residuals(arc: Arc, arr: np.ndarray) -> np.ndarray` — absolute y residuals per point

- [ ] **Step 1: Write failing tests**

`tests/test_fit.py`:

```python
import numpy as np
import pytest

from juggletrack.arcs.fit import fit_arc, points_array, y_residuals
from juggletrack.sim import CascadeParams, simulate_cascade


def flight_points(noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Sample one clean flight from the simulator's first throw."""
    r = simulate_cascade(n_throws=1, fps=60.0, noise=noise, seed=seed)
    return points_array(r.detections)


def test_fit_recovers_gravity():
    arr = flight_points()
    arc = fit_arc(arr, arc_id=7)
    p = CascadeParams()
    assert arc.id == 7
    assert arc.ay == pytest.approx(p.g / 2.0, rel=1e-3)   # ay = g/2
    assert arc.rmse < 1e-6
    assert arc.n_points == len(arr)


def test_fit_recovers_apex_and_endpoints():
    arr = flight_points()
    arc = fit_arc(arr)
    p = CascadeParams()
    assert arc.apex_y() == pytest.approx(p.hand_y - p.v0**2 / (2 * p.g), abs=1e-3)
    assert arc.y_at(arc.t_start) == pytest.approx(p.hand_y, abs=0.01)


def test_fit_with_noise_rmse_tracks_noise():
    arr = flight_points(noise=0.005, seed=2)
    arc = fit_arc(arr)
    assert 0.001 < arc.rmse < 0.015
    p = CascadeParams()
    assert arc.ay == pytest.approx(p.g / 2.0, rel=0.15)


def test_confidence_weighting_downweights_outlier():
    arr = flight_points()
    outlier = arr[len(arr) // 2].copy()
    outlier[2] += 0.3          # push y far off the parabola
    outlier[3] = 0.01          # ...but with near-zero confidence
    arr_w = np.vstack([arr, outlier])
    arc = fit_arc(arr_w)
    assert arc.ay == pytest.approx(CascadeParams().g / 2.0, rel=0.02)


def test_residuals_flag_the_outlier():
    arr = flight_points()
    outlier = arr[10].copy()
    outlier[2] += 0.2
    arr2 = np.vstack([arr, outlier])
    arc = fit_arc(arr)  # fit without the outlier
    res = y_residuals(arc, arr2)
    assert res[-1] > 0.15 and res[:-1].max() < 0.01


def test_fit_requires_three_points():
    with pytest.raises(ValueError):
        fit_arc(np.array([[0.0, 0.5, 0.5, 1.0], [0.1, 0.5, 0.5, 1.0]]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fit.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'juggletrack.arcs'`.

- [ ] **Step 3: Implement `src/juggletrack/arcs/fit.py`** (and empty `src/juggletrack/arcs/__init__.py`)

```python
"""Weighted least-squares parabola fitting for ballistic arcs."""
from __future__ import annotations

import numpy as np

from juggletrack.types import Arc, Detection


def points_array(dets: list[Detection]) -> np.ndarray:
    """(N, 4) array of [t, x, y, confidence], sorted by t."""
    arr = np.array([[d.t, d.x, d.y, d.confidence] for d in dets], dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 4)
    return arr[np.argsort(arr[:, 0])]


def fit_arc(arr: np.ndarray, arc_id: int = -1) -> Arc:
    if len(arr) < 3:
        raise ValueError(f"fit_arc needs >= 3 points, got {len(arr)}")
    arr = arr[np.argsort(arr[:, 0])]
    t0 = arr[0, 0]
    dt = arr[:, 0] - t0
    w = arr[:, 3]
    ay, by, cy = np.polyfit(dt, arr[:, 2], 2, w=w)
    bx, cx = np.polyfit(dt, arr[:, 1], 1, w=w)
    resid = (ay * dt * dt + by * dt + cy) - arr[:, 2]
    rmse = float(np.sqrt(np.average(resid**2, weights=w)))
    return Arc(
        id=arc_id, t_start=float(t0), t_end=float(arr[-1, 0]),
        ay=float(ay), by=float(by), cy=float(cy), bx=float(bx), cx=float(cx),
        n_points=len(arr), rmse=rmse,
    )


def y_residuals(arc: Arc, arr: np.ndarray) -> np.ndarray:
    dt = arr[:, 0] - arc.t_start
    return np.abs(arc.ay * dt * dt + arc.by * dt + arc.cy - arr[:, 2])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_fit.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/juggletrack/arcs/ tests/test_fit.py
git commit -m "feat: weighted parabola fitting with residual diagnostics

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Global arc extraction (`arcs/extract.py`)

**Files:**
- Create: `src/juggletrack/arcs/extract.py`
- Test: `tests/test_extract.py`

**Interfaces:**
- Consumes: `points_array`, `fit_arc`, `y_residuals` from Task 4; `Detection`, `Arc` from Task 2.
- Produces (consumed by Tasks 6–9 and by Plan 2 with real detections):
  - `extract_arcs(dets: list[Detection], *, g_range=(0.5, 8.0), link_max_dt=0.12, link_max_dist=0.08, resid_tol=0.02, min_points=6, min_duration=0.15, em_iters=3) -> list[Arc]` — arcs sorted by `t_start`, ids re-numbered 0..N-1.

**Algorithm** (Hawkeye-style, hard-assignment EM): (1) greedy nearest-neighbor linking of detections into fragments using constant-velocity prediction; (2) split fragments into ballistic pieces wherever the incremental parabola fit's rmse exceeds `resid_tol`; (3) EM iterations — E: re-assign every detection to the best arc covering its time (±0.15s margin) if its residual < `2*resid_tol`; M: refit arcs from assignments; then merge time-adjacent arcs whose union still fits one parabola, and prune arcs that are too short, too brief, non-gravitational (`ay` outside `[g_range[0]/2, g_range[1]/2]`), or poor fits. Held balls (stationary → `ay≈0`) and floor-resting balls prune out automatically; a dropped ball's below-hand-line continuation stays part of its flight arc (same parabola); a bounce becomes its own small arc (same gravity — deliberately kept for drop signal 2).

- [ ] **Step 1: Write failing tests**

`tests/test_extract.py`:

```python
import numpy as np
import pytest

from juggletrack.arcs.extract import extract_arcs
from juggletrack.sim import simulate_cascade


def match_arcs_to_flights(arcs, result, tol):
    """Return the arcs that match ground-truth (throw, catch) windows 1:1."""
    matched = []
    for th in result.throw_times:
        ca = th + result.params.flight_s
        hits = [a for a in arcs if abs(a.t_start - th) < tol and abs(a.t_end - ca) < tol]
        matched.append((th, hits))
    return matched


def test_clean_run_yields_one_arc_per_throw():
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    arcs = extract_arcs(r.detections)
    assert len(arcs) == 12
    for th, hits in match_arcs_to_flights(arcs, r, tol=0.06):
        assert len(hits) == 1, f"throw at {th} matched {len(hits)} arcs"
    p = r.params
    for a in arcs:
        assert a.ay == pytest.approx(p.g / 2.0, rel=0.05)
    assert arcs == sorted(arcs, key=lambda a: a.t_start)
    assert [a.id for a in arcs] == list(range(12))


def test_noise_and_dropout_still_recovers_all_arcs():
    r = simulate_cascade(n_throws=12, fps=30.0, noise=0.004, dropout=0.15, seed=2)
    arcs = extract_arcs(r.detections)
    assert len(arcs) == 12
    for th, hits in match_arcs_to_flights(arcs, r, tol=0.10):
        assert len(hits) == 1


def test_false_positives_do_not_create_arcs():
    r = simulate_cascade(n_throws=12, fps=30.0, false_positives_per_frame=0.5, seed=3)
    arcs = extract_arcs(r.detections)
    assert len(arcs) == 12


def test_shuffle_invariance():
    """Property from the spec: results independent of detection ordering/identity."""
    r = simulate_cascade(n_throws=10, fps=30.0, noise=0.002, seed=4)
    arcs_a = extract_arcs(r.detections)
    rng = np.random.default_rng(0)
    shuffled = list(r.detections)
    rng.shuffle(shuffled)
    arcs_b = extract_arcs(shuffled)
    assert arcs_a == arcs_b


def test_drop_scenario_arc_reaches_floor():
    r = simulate_cascade(n_throws=20, fps=30.0, drop_at_throw=8, seed=1)
    arcs = extract_arcs(r.detections)
    p = r.params
    floor_arcs = [a for a in arcs if a.y_at(a.t_end) > p.hand_y + 0.15]
    # the dropped flight continues to the floor; the bounce may add one more
    assert 1 <= len(floor_arcs) <= 2
    main = min(floor_arcs, key=lambda a: a.t_start)
    assert main.t_end == pytest.approx(r.drop_t, abs=0.08)


def test_held_balls_produce_no_arcs():
    r = simulate_cascade(n_throws=8, fps=30.0, include_held=True, seed=5)
    arcs = extract_arcs(r.detections)
    assert len(arcs) == 8  # held-ball (stationary) detections must not become arcs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_extract.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'juggletrack.arcs.extract'`.

- [ ] **Step 3: Implement `src/juggletrack/arcs/extract.py`**

```python
"""Global arc extraction: detections -> ballistic arcs (hard-assignment EM).

Identity-free by design: works on the unordered detection cloud, so track-ID
switches upstream cannot corrupt results (spec section 4).
"""
from __future__ import annotations

import math

import numpy as np

from juggletrack.arcs.fit import fit_arc, points_array, y_residuals
from juggletrack.types import Arc, Detection

_EM_TIME_MARGIN = 0.15  # arcs may claim points this far beyond their current span


def extract_arcs(
    dets: list[Detection],
    *,
    g_range: tuple[float, float] = (0.5, 8.0),
    link_max_dt: float = 0.12,
    link_max_dist: float = 0.08,
    resid_tol: float = 0.02,
    min_points: int = 6,
    min_duration: float = 0.15,
    em_iters: int = 3,
) -> list[Arc]:
    arr = points_array(dets)
    if len(arr) < min_points:
        return []

    fragments = _link_fragments(arr, link_max_dt, link_max_dist)
    seeds: list[list[int]] = []
    for frag in fragments:
        seeds.extend(_split_ballistic(arr, frag, resid_tol))

    arcs = [fit_arc(arr[idxs]) for idxs in seeds if len(idxs) >= 4]

    for _ in range(em_iters):
        arcs = _em_assign_refit(arr, arcs, resid_tol)
        arcs = _merge_pass(arr, arcs, resid_tol)
        arcs = _prune(arcs, g_range, resid_tol, min_points, min_duration)

    arcs.sort(key=lambda a: a.t_start)
    return [a.model_copy(update={"id": i}) for i, a in enumerate(arcs)]


def _link_fragments(arr: np.ndarray, max_dt: float, max_dist: float) -> list[list[int]]:
    """Greedy constant-velocity linker: detection cloud -> candidate fragments."""
    open_frags: list[dict] = []
    done: list[list[int]] = []
    for i in range(len(arr)):
        t, x, y = arr[i, 0], arr[i, 1], arr[i, 2]
        still = []
        for fr in open_frags:
            if t - fr["t"] > max_dt:
                done.append(fr["idxs"])
            else:
                still.append(fr)
        open_frags = still
        best, best_d = None, max_dist
        for fr in open_frags:
            dt = t - fr["t"]
            if dt <= 0:
                continue
            d = math.hypot(x - (fr["x"] + fr["vx"] * dt), y - (fr["y"] + fr["vy"] * dt))
            if d < best_d:
                best, best_d = fr, d
        if best is None:
            open_frags.append({"idxs": [i], "t": t, "x": x, "y": y, "vx": 0.0, "vy": 0.0})
        else:
            dt = t - best["t"]
            best["vx"], best["vy"] = (x - best["x"]) / dt, (y - best["y"]) / dt
            best["t"], best["x"], best["y"] = t, x, y
            best["idxs"].append(i)
    done.extend(fr["idxs"] for fr in open_frags)
    return [f for f in done if len(f) >= 4]


def _split_ballistic(arr: np.ndarray, idxs: list[int], resid_tol: float) -> list[list[int]]:
    """Split a fragment wherever one parabola stops explaining it."""
    pieces: list[list[int]] = []
    cur: list[int] = []
    for i in idxs:
        cur.append(i)
        if len(cur) >= 4 and fit_arc(arr[cur]).rmse > resid_tol:
            pieces.append(cur[:-1])
            cur = [i]
    if len(cur) >= 4:
        pieces.append(cur)
    return [p for p in pieces if len(p) >= 4]


def _em_assign_refit(arr: np.ndarray, arcs: list[Arc], resid_tol: float) -> list[Arc]:
    if not arcs:
        return []
    t = arr[:, 0]
    best_res = np.full(len(arr), np.inf)
    best_arc = np.full(len(arr), -1, dtype=int)
    for k, arc in enumerate(arcs):
        in_span = (t >= arc.t_start - _EM_TIME_MARGIN) & (t <= arc.t_end + _EM_TIME_MARGIN)
        res = np.where(in_span, y_residuals(arc, arr), np.inf)
        better = res < best_res
        best_res[better] = res[better]
        best_arc[better] = k
    best_arc[best_res > 2 * resid_tol] = -1

    out: list[Arc] = []
    for k in range(len(arcs)):
        member = np.where(best_arc == k)[0]
        if len(member) >= 3:
            out.append(fit_arc(arr[member]))
    return out


def _merge_pass(arr: np.ndarray, arcs: list[Arc], resid_tol: float) -> list[Arc]:
    arcs = sorted(arcs, key=lambda a: a.t_start)
    t = arr[:, 0]
    merged: list[Arc] = []
    i = 0
    while i < len(arcs):
        a = arcs[i]
        if i + 1 < len(arcs):
            b = arcs[i + 1]
            if b.t_start - a.t_end < 0.2:
                sel = ((t >= a.t_start) & (t <= a.t_end)) | ((t >= b.t_start) & (t <= b.t_end))
                pts = arr[sel]
                keep_a = y_residuals(a, pts) < 2 * resid_tol
                keep_b = y_residuals(b, pts) < 2 * resid_tol
                pts = pts[keep_a | keep_b]
                if len(pts) >= 3:
                    union = fit_arc(pts)
                    if union.rmse <= resid_tol:
                        merged.append(union)
                        i += 2
                        continue
        merged.append(a)
        i += 1
    return merged


def _prune(
    arcs: list[Arc], g_range: tuple[float, float], resid_tol: float,
    min_points: int, min_duration: float,
) -> list[Arc]:
    lo, hi = g_range[0] / 2.0, g_range[1] / 2.0
    return [
        a for a in arcs
        if a.n_points >= min_points
        and a.duration() >= min_duration
        and lo <= a.ay <= hi
        and a.rmse <= resid_tol
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_extract.py -v`
Expected: PASS (6 passed). Tuning notes if not: `test_noise_and_dropout` failures usually mean `resid_tol` too tight for `noise=0.004` (3σ ≈ 0.012 < 0.02, should pass) or the EM margin dropping points at flight edges — check that arcs' `n_points` ≈ 30 for 1.1s flights at 30fps before touching thresholds. `test_shuffle_invariance` failure means somewhere iteration order leaks into results — `points_array` sorting plus deterministic tie-breaks must make it exact.

- [ ] **Step 5: Commit**

```bash
git add src/juggletrack/arcs/extract.py tests/test_extract.py
git commit -m "feat: identity-free global arc extraction (link, split, EM, merge, prune)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Hand line + throw/catch events (`events/handline.py`, `events/catches.py`)

**Files:**
- Create: `src/juggletrack/events/__init__.py` (empty), `src/juggletrack/events/handline.py`, `src/juggletrack/events/catches.py`
- Test: `tests/test_handline.py`, `tests/test_catches.py`

**Interfaces:**
- Consumes: `Arc`, `ThrowEvent`, `CatchEvent` from Task 2; `extract_arcs` from Task 5 (in tests).
- Produces (consumed by Tasks 7–9):
  - `estimate_hand_line(arcs: list[Arc]) -> float` — robust arc-endpoint median (spec: pose-free fallback is the v1 default; pose arrives in Plan 2)
  - `derive_events(arcs: list[Arc], hand_line: float, *, min_apex_above=0.05, floor_margin=0.10) -> tuple[list[ThrowEvent], list[CatchEvent]]` — sub-frame times from parabola roots; an arc only produces a `CatchEvent` if it terminates near the hand line (a dropped ball's arc, ending near the floor, produces a throw but **no catch**); arcs whose apex never rises `min_apex_above` above the hand line (bounces, noise) produce no events at all.

- [ ] **Step 1: Write failing tests**

`tests/test_handline.py`:

```python
import pytest

from juggletrack.arcs.extract import extract_arcs
from juggletrack.events.handline import estimate_hand_line
from juggletrack.sim import simulate_cascade


def test_hand_line_clean():
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    hl = estimate_hand_line(extract_arcs(r.detections))
    assert hl == pytest.approx(r.params.hand_y, abs=0.02)


def test_hand_line_robust_to_drop():
    r = simulate_cascade(n_throws=20, fps=30.0, drop_at_throw=8, seed=1)
    hl = estimate_hand_line(extract_arcs(r.detections))
    assert hl == pytest.approx(r.params.hand_y, abs=0.03)


def test_hand_line_empty_arcs():
    assert estimate_hand_line([]) == 0.0
```

`tests/test_catches.py`:

```python
import numpy as np
import pytest

from juggletrack.arcs.extract import extract_arcs
from juggletrack.events.catches import derive_events
from juggletrack.events.handline import estimate_hand_line
from juggletrack.sim import simulate_cascade


def pipeline(r):
    arcs = extract_arcs(r.detections)
    hl = estimate_hand_line(arcs)
    throws, catches = derive_events(arcs, hl)
    return arcs, hl, throws, catches


def test_clean_run_event_counts_and_subframe_accuracy():
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    _, _, throws, catches = pipeline(r)
    assert len(throws) == 12 and len(catches) == 12
    throw_err = np.abs(np.array([e.t for e in throws]) - np.array(r.throw_times))
    catch_err = np.abs(np.array([e.t for e in catches]) - np.array(r.catch_times))
    # sub-frame: better than half a frame at 30fps
    assert throw_err.max() < 0.017 and catch_err.max() < 0.017


def test_events_sorted_and_linked_to_arcs():
    r = simulate_cascade(n_throws=10, fps=30.0, seed=2)
    arcs, _, throws, catches = pipeline(r)
    assert [e.t for e in throws] == sorted(e.t for e in throws)
    arc_ids = {a.id for a in arcs}
    assert all(e.arc_id in arc_ids for e in throws + catches)


def test_dropped_ball_has_throw_but_no_catch():
    r = simulate_cascade(n_throws=20, fps=30.0, drop_at_throw=8, seed=1)
    _, _, throws, catches = pipeline(r)
    assert len(throws) == len(r.throw_times)
    assert len(catches) == len(r.catch_times)  # exactly the successful ones
    # and specifically: no catch event near the missed catch time
    assert all(abs(c.t - r.missed_catch_t) > 0.1 for c in catches)


def test_noisy_run_events_within_tolerance():
    r = simulate_cascade(n_throws=12, fps=30.0, noise=0.004, dropout=0.15, seed=2)
    _, _, throws, catches = pipeline(r)
    assert len(throws) == 12 and len(catches) == 12
    catch_err = np.abs(np.array([e.t for e in catches]) - np.array(r.catch_times))
    assert catch_err.max() < 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_handline.py tests/test_catches.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'juggletrack.events'`.

- [ ] **Step 3: Implement**

`src/juggletrack/events/handline.py`:

```python
"""Hand-line estimation from arc statistics (pose-free fallback, spec §4)."""
from __future__ import annotations

import numpy as np

from juggletrack.types import Arc


def estimate_hand_line(arcs: list[Arc]) -> float:
    """Robust median of arc endpoint heights.

    Throws and catches happen near hand height, so arc endpoints cluster there.
    A dropped ball contributes one floor-level endpoint; the median absorbs it.
    """
    if not arcs:
        return 0.0
    ys = [a.y_at(a.t_start) for a in arcs] + [a.y_at(a.t_end) for a in arcs]
    return float(np.median(ys))
```

`src/juggletrack/events/catches.py`:

```python
"""Sub-frame throw/catch events from arc coefficients (spec §4).

throw = rising crossing of the hand line (smaller quadratic root);
catch = falling crossing (larger root), only if the arc actually terminates
near the hand line — an arc that keeps descending toward the floor is a drop
candidate and yields no catch.
"""
from __future__ import annotations

import math

from juggletrack.types import Arc, CatchEvent, ThrowEvent


def derive_events(
    arcs: list[Arc],
    hand_line: float,
    *,
    min_apex_above: float = 0.05,
    floor_margin: float = 0.10,
) -> tuple[list[ThrowEvent], list[CatchEvent]]:
    throws: list[ThrowEvent] = []
    catches: list[CatchEvent] = []
    for arc in arcs:
        if arc.ay <= 0:
            continue
        if arc.apex_y() > hand_line - min_apex_above:
            continue  # never rose meaningfully above the hands: bounce or noise

        disc = arc.by**2 - 4.0 * arc.ay * (arc.cy - hand_line)
        if disc >= 0:
            root = math.sqrt(disc)
            dt_throw = (-arc.by - root) / (2.0 * arc.ay)
            dt_catch = (-arc.by + root) / (2.0 * arc.ay)
        else:  # fitted arc sits entirely above the hand line: fall back to span
            dt_throw, dt_catch = 0.0, arc.t_end - arc.t_start

        t_throw = arc.t_start + dt_throw
        throws.append(ThrowEvent(t=t_throw, x=arc.x_at(t_throw), arc_id=arc.id))

        if arc.y_at(arc.t_end) <= hand_line + floor_margin:
            t_catch = arc.t_start + dt_catch
            catches.append(CatchEvent(t=t_catch, x=arc.x_at(t_catch), arc_id=arc.id))

    throws.sort(key=lambda e: e.t)
    catches.sort(key=lambda e: e.t)
    return throws, catches
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_handline.py tests/test_catches.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add src/juggletrack/events/ tests/test_handline.py tests/test_catches.py
git commit -m "feat: hand-line estimation and sub-frame throw/catch events from arcs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Runs — period, chaining, periodicity (`events/runs.py`, `events/periodicity.py`)

**Files:**
- Modify: `src/juggletrack/events/catches.py` (extract a public crossing helper)
- Create: `src/juggletrack/events/periodicity.py`, `src/juggletrack/events/runs.py`
- Test: `tests/test_periodicity.py`, `tests/test_runs.py`

**Interfaces:**
- Consumes: Tasks 2, 5, 6 outputs.
- Produces (consumed by Tasks 8–9):
  - `hand_line_crossings(arc: Arc, hand_line: float) -> tuple[float, float] | None` (in `catches.py`) — absolute (rising_t, falling_t); falls back to `(t_start, t_end)` when the fitted arc never meets the hand line; `None` if `ay <= 0`
  - `periodicity_score(arcs, t_from, t_to, *, dt=1/30, lag_range=(0.2, 1.2)) -> tuple[float, float | None]` — (max normalized autocorrelation of the airborne-count signal, best lag in seconds); `(0.0, None)` for degenerate signals
  - `estimate_period(throws: list[ThrowEvent], default=0.5) -> float`
  - `segment_runs(arcs, throws, catches, hand_line, *, gap_factor=1.3, min_arcs=3, default_period=0.5) -> list[Run]` — `end_reason` left as `"stop"`; Task 8 upgrades to `"drop"`, Task 9 to `"video_end"`. Run `end_t` = latest falling hand-line crossing among its arcs, so a dropped throw's *scheduled* (missed) catch time correctly ends the run.

- [ ] **Step 1: Refactor `catches.py` — extract the crossing helper (existing tests stay green)**

In `src/juggletrack/events/catches.py`, add:

```python
def hand_line_crossings(arc: Arc, hand_line: float) -> tuple[float, float] | None:
    """Absolute times where the arc crosses the hand line (rising, falling).

    Falls back to (t_start, t_end) when the fitted arc never reaches the line.
    """
    if arc.ay <= 0:
        return None
    disc = arc.by**2 - 4.0 * arc.ay * (arc.cy - hand_line)
    if disc < 0:
        return arc.t_start, arc.t_end
    root = math.sqrt(disc)
    dt_rise = (-arc.by - root) / (2.0 * arc.ay)
    dt_fall = (-arc.by + root) / (2.0 * arc.ay)
    return arc.t_start + dt_rise, arc.t_start + dt_fall
```

and rewrite the body of `derive_events` to use it:

```python
    for arc in arcs:
        crossings = hand_line_crossings(arc, hand_line)
        if crossings is None:
            continue
        if arc.apex_y() > hand_line - min_apex_above:
            continue
        t_throw, t_catch = crossings
        throws.append(ThrowEvent(t=t_throw, x=arc.x_at(t_throw), arc_id=arc.id))
        if arc.y_at(arc.t_end) <= hand_line + floor_margin:
            catches.append(CatchEvent(t=t_catch, x=arc.x_at(t_catch), arc_id=arc.id))
```

Run: `uv run pytest tests/test_catches.py -v` — Expected: PASS (still 4 passed). Commit:

```bash
git add src/juggletrack/events/catches.py
git commit -m "refactor: extract hand_line_crossings helper

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 2: Write failing tests**

`tests/test_periodicity.py`:

```python
import pytest

from juggletrack.arcs.extract import extract_arcs
from juggletrack.events.periodicity import periodicity_score
from juggletrack.sim import simulate_cascade


def test_cascade_is_periodic_at_throw_period():
    r = simulate_cascade(n_throws=16, fps=30.0, seed=1)
    arcs = extract_arcs(r.detections)
    score, lag = periodicity_score(arcs, r.run_start, r.run_end)
    assert score > 0.5
    assert lag == pytest.approx(r.params.period_s, abs=0.07)


def test_no_arcs_scores_zero():
    score, lag = periodicity_score([], 0.0, 5.0)
    assert score == 0.0 and lag is None


def test_signal_dies_after_run_end():
    r = simulate_cascade(n_throws=16, fps=30.0, seed=1)
    arcs = extract_arcs(r.detections)
    score, _ = periodicity_score(arcs, r.run_end + 0.2, r.run_end + 3.0)
    assert score == 0.0  # nothing airborne after the run
```

`tests/test_runs.py`:

```python
import pytest

from juggletrack.arcs.extract import extract_arcs
from juggletrack.events.catches import derive_events
from juggletrack.events.handline import estimate_hand_line
from juggletrack.events.runs import estimate_period, segment_runs
from juggletrack.sim import simulate_cascade
from juggletrack.types import Detection


def pipeline(dets):
    arcs = extract_arcs(dets)
    hl = estimate_hand_line(arcs)
    throws, catches = derive_events(arcs, hl)
    return arcs, hl, throws, catches, segment_runs(arcs, throws, catches, hl)


def shift(dets, dt_s, dframes):
    return [
        Detection(frame_idx=d.frame_idx + dframes, t=d.t + dt_s, x=d.x, y=d.y,
                  w=d.w, h=d.h, confidence=d.confidence)
        for d in dets
    ]


def test_single_clean_run():
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    _, _, throws, _, runs = pipeline(r.detections)
    assert len(runs) == 1
    run = runs[0]
    assert run.throws == 12 and run.catches == 12
    assert run.start_t == pytest.approx(r.run_start, abs=0.05)
    assert run.end_t == pytest.approx(r.run_end, abs=0.05)
    assert run.period_s == pytest.approx(r.params.period_s, abs=0.03)
    assert run.quality > 0.5
    assert run.end_reason == "stop"
    assert estimate_period(throws) == pytest.approx(0.45, abs=0.02)


def test_two_separated_runs():
    a = simulate_cascade(n_throws=10, fps=30.0, seed=1)
    b = simulate_cascade(n_throws=8, fps=30.0, seed=2)
    dets = a.detections + shift(b.detections, 12.0, 360)
    _, _, _, _, runs = pipeline(dets)
    assert len(runs) == 2
    assert runs[0].throws == 10 and runs[1].throws == 8
    assert runs[1].start_t == pytest.approx(12.0 + b.run_start, abs=0.05)


def test_drop_run_ends_at_missed_catch():
    r = simulate_cascade(n_throws=20, fps=30.0, drop_at_throw=8, seed=1)
    _, _, _, _, runs = pipeline(r.detections)
    assert len(runs) == 1
    assert runs[0].end_t == pytest.approx(r.missed_catch_t, abs=0.1)
    assert runs[0].catches == len(r.catch_times)


def test_min_arcs_filters_stray_tosses():
    r = simulate_cascade(n_throws=2, fps=30.0, seed=3)
    _, _, _, _, runs = pipeline(r.detections)
    assert runs == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_periodicity.py tests/test_runs.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement**

`src/juggletrack/events/periodicity.py`:

```python
"""Airborne-count periodicity: run validator + quality score (spec §4)."""
from __future__ import annotations

import numpy as np

from juggletrack.types import Arc


def airborne_count(arcs: list[Arc], t_grid: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(t_grid))
    for a in arcs:
        counts += (t_grid >= a.t_start) & (t_grid <= a.t_end)
    return counts


def periodicity_score(
    arcs: list[Arc],
    t_from: float,
    t_to: float,
    *,
    dt: float = 1.0 / 30.0,
    lag_range: tuple[float, float] = (0.2, 1.2),
) -> tuple[float, float | None]:
    if t_to - t_from < 2 * lag_range[1]:
        # window too short to see two periods at the largest lag: score what we can
        t_to = t_from + 2 * lag_range[1]
    grid = np.arange(t_from, t_to, dt)
    if len(grid) < 8:
        return 0.0, None
    s = airborne_count(arcs, grid)
    s = s - s.mean()
    var = float(np.dot(s, s))
    if var < 1e-9:
        return 0.0, None
    full = np.correlate(s, s, mode="full")[len(s) - 1 :] / var  # lag 0..n-1, normalized
    lo, hi = int(lag_range[0] / dt), min(int(lag_range[1] / dt) + 1, len(full))
    if hi <= lo:
        return 0.0, None
    window = full[lo:hi]
    k = int(np.argmax(window))
    return float(window[k]), float((lo + k) * dt)
```

`src/juggletrack/events/runs.py`:

```python
"""Run segmentation: chain event-bearing arcs, score with periodicity (spec §4)."""
from __future__ import annotations

import numpy as np

from juggletrack.events.catches import hand_line_crossings
from juggletrack.events.periodicity import periodicity_score
from juggletrack.types import Arc, CatchEvent, Run, ThrowEvent


def estimate_period(throws: list[ThrowEvent], default: float = 0.5) -> float:
    if len(throws) < 3:
        return default
    ts = sorted(e.t for e in throws)
    return float(np.median(np.diff(ts)))


def segment_runs(
    arcs: list[Arc],
    throws: list[ThrowEvent],
    catches: list[CatchEvent],
    hand_line: float,
    *,
    gap_factor: float = 1.3,
    min_arcs: int = 3,
    default_period: float = 0.5,
) -> list[Run]:
    throw_arc_ids = {e.arc_id for e in throws}
    juggling_arcs = sorted((a for a in arcs if a.id in throw_arc_ids), key=lambda a: a.t_start)
    if not juggling_arcs:
        return []
    period = estimate_period(throws, default_period)

    def arc_end(a: Arc) -> float:
        crossings = hand_line_crossings(a, hand_line)
        return crossings[1] if crossings else a.t_end

    groups: list[list[Arc]] = [[juggling_arcs[0]]]
    cur_end = arc_end(juggling_arcs[0])
    for a in juggling_arcs[1:]:
        if a.t_start > cur_end + gap_factor * period:
            groups.append([a])
            cur_end = arc_end(a)
        else:
            groups[-1].append(a)
            cur_end = max(cur_end, arc_end(a))

    runs: list[Run] = []
    for group in groups:
        if len(group) < min_arcs:
            continue
        ids = {a.id for a in group}
        run_throws = sorted(e.t for e in throws if e.arc_id in ids)
        run_catches = [e for e in catches if e.arc_id in ids]
        start_t = run_throws[0]
        end_t = max(arc_end(a) for a in group)
        p = float(np.median(np.diff(run_throws))) if len(run_throws) >= 3 else None
        quality, _ = periodicity_score(group, start_t, end_t)
        runs.append(Run(
            start_t=start_t, end_t=end_t,
            catches=len(run_catches), throws=len(run_throws),
            arc_ids=sorted(ids), end_reason="stop", period_s=p, quality=quality,
        ))
    return runs
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_periodicity.py tests/test_runs.py -v`
Expected: PASS (8 passed). If `test_cascade_is_periodic_at_throw_period` finds the lag at a harmonic (0.9 instead of 0.45), tighten `lag_range` — but with `lag_range=(0.2, 1.2)` the true period's autocorrelation peak dominates for this signal; investigate before tuning.

- [ ] **Step 6: Commit**

```bash
git add src/juggletrack/events/ tests/test_periodicity.py tests/test_runs.py
git commit -m "feat: run segmentation via arc chaining with periodicity quality score

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Drop detection (`events/drops.py`)

**Files:**
- Create: `src/juggletrack/events/drops.py`
- Test: `tests/test_drops.py`

**Interfaces:**
- Consumes: Tasks 2, 5, 6, 7 outputs (`hand_line_crossings`, `Run`, `Arc`).
- Produces (consumed by Task 9):
  - `detect_drops(arcs, runs, hand_line, *, floor_margin=0.12, bounce_window=0.6, bounce_x_tol=0.10, collapse_factor=1.5) -> tuple[list[DropEvent], list[Run]]` — returns drop events (each listing which of the three spec signals fired; ≥2 required) and runs with `end_reason` upgraded to `"drop"` where a drop ends them.

**The three signals (spec §4):** `floor_descent` — arc terminates well below the hand line still moving downward; `bounce` — a low-apex arc starts just after and near where the candidate ended; `periodicity_collapse` — no other juggling arc is airborne within `collapse_factor × period` after the candidate's missed-catch time.

- [ ] **Step 1: Write failing tests**

`tests/test_drops.py`:

```python
import pytest

from juggletrack.arcs.extract import extract_arcs
from juggletrack.events.catches import derive_events
from juggletrack.events.drops import detect_drops
from juggletrack.events.handline import estimate_hand_line
from juggletrack.events.runs import segment_runs
from juggletrack.sim import simulate_cascade


def pipeline(dets):
    arcs = extract_arcs(dets)
    hl = estimate_hand_line(arcs)
    throws, catches = derive_events(arcs, hl)
    runs = segment_runs(arcs, throws, catches, hl)
    drops, runs = detect_drops(arcs, runs, hl)
    return arcs, hl, drops, runs


def test_drop_detected_with_signals():
    r = simulate_cascade(n_throws=20, fps=30.0, drop_at_throw=8, seed=1)
    _, _, drops, runs = pipeline(r.detections)
    assert len(drops) == 1
    d = drops[0]
    assert len(d.signals) >= 2
    assert "floor_descent" in d.signals
    assert d.t == pytest.approx(r.missed_catch_t, abs=0.1)
    assert runs[0].end_reason == "drop"


def test_clean_stop_is_not_a_drop():
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    _, _, drops, runs = pipeline(r.detections)
    assert drops == []
    assert runs[0].end_reason == "stop"


def test_noisy_clean_run_no_false_drops():
    r = simulate_cascade(n_throws=14, fps=30.0, noise=0.004, dropout=0.15, seed=6)
    _, _, drops, runs = pipeline(r.detections)
    assert drops == []
    assert runs and runs[0].end_reason == "stop"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_drops.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'juggletrack.events.drops'`.

- [ ] **Step 3: Implement `src/juggletrack/events/drops.py`**

```python
"""Three-signal drop detection (spec §4 — no prior art; first-principles design)."""
from __future__ import annotations

from juggletrack.events.catches import hand_line_crossings
from juggletrack.types import Arc, DropEvent, Run


def detect_drops(
    arcs: list[Arc],
    runs: list[Run],
    hand_line: float,
    *,
    floor_margin: float = 0.12,
    bounce_window: float = 0.6,
    bounce_x_tol: float = 0.10,
    collapse_factor: float = 1.5,
) -> tuple[list[DropEvent], list[Run]]:
    drops: list[DropEvent] = []
    by_id = {a.id: a for a in arcs}

    for run in runs:
        period = run.period_s or 0.5
        run_arcs = [by_id[i] for i in run.arc_ids if i in by_id]
        for cand in run_arcs:
            end_y = cand.y_at(cand.t_end)
            if end_y <= hand_line + floor_margin:
                continue  # ended near the hands: caught, not dropped

            signals = []
            if cand.vy_at(cand.t_end) > 0:
                signals.append("floor_descent")

            for b in arcs:
                if b.id == cand.id:
                    continue
                starts_after = 0.0 < b.t_start - cand.t_end < bounce_window
                near_x = abs(b.x_at(b.t_start) - cand.x_at(cand.t_end)) < bounce_x_tol
                stays_low = b.apex_y() > hand_line
                if starts_after and near_x and stays_low:
                    signals.append("bounce")
                    break

            crossings = hand_line_crossings(cand, hand_line)
            miss_t = crossings[1] if crossings else cand.t_end
            others_airborne = any(
                a.id != cand.id and a.t_start <= miss_t + collapse_factor * period
                and a.t_end > miss_t + 0.5 * period
                for a in run_arcs
            )
            if not others_airborne:
                signals.append("periodicity_collapse")

            if len(signals) >= 2:
                drops.append(DropEvent(
                    t=miss_t, x=cand.x_at(miss_t), arc_id=cand.id, signals=signals,
                ))

    drops.sort(key=lambda d: d.t)
    updated: list[Run] = []
    for run in runs:
        period = run.period_s or 0.5
        ends_in_drop = any(abs(d.t - run.end_t) <= period for d in drops)
        updated.append(run.model_copy(update={"end_reason": "drop"}) if ends_in_drop else run)
    return drops, updated
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_drops.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/juggletrack/events/drops.py tests/test_drops.py
git commit -m "feat: three-signal drop detection with run end_reason upgrade

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: End-to-end assembly (`analyze.py`)

**Files:**
- Create: `src/juggletrack/analyze.py`
- Test: `tests/test_analyze.py`

**Interfaces:**
- Consumes: everything above.
- Produces (Plan 2's offline pipeline calls exactly this with real detections):
  - `AnalyzeConfig` — pydantic model bundling the tuning knobs: `g_range=(0.5, 8.0)`, `resid_tol=0.02`, `min_points=6`, `min_duration=0.15`, `gap_factor=1.3`, `min_arcs=3`, `video_end_margin=1.0` (in periods)
  - `analyze_detections(dets: list[Detection], config: AnalyzeConfig | None = None) -> SessionResult`

- [ ] **Step 1: Write failing tests**

`tests/test_analyze.py`:

```python
import json

import numpy as np
import pytest

from juggletrack.analyze import analyze_detections
from juggletrack.sim import simulate_cascade
from juggletrack.types import SessionResult


def test_end_to_end_clean_run():
    r = simulate_cascade(n_throws=12, fps=30.0, noise=0.003, dropout=0.1, seed=1)
    sr = analyze_detections(r.detections)
    assert len(sr.runs) == 1
    assert sr.runs[0].catches == 12
    assert sr.runs[0].end_reason == "stop"
    assert sr.hand_line_y == pytest.approx(r.params.hand_y, abs=0.03)
    back = SessionResult.model_validate(json.loads(sr.model_dump_json()))
    assert back == sr


def test_end_to_end_drop_run():
    r = simulate_cascade(n_throws=20, fps=30.0, drop_at_throw=8, seed=1)
    sr = analyze_detections(r.detections)
    assert len(sr.runs) == 1
    assert sr.runs[0].end_reason == "drop"
    assert len(sr.drops) == 1
    assert sr.drops[0].t == pytest.approx(r.missed_catch_t, abs=0.1)


def test_video_end_run():
    r = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    cutoff = r.catch_times[8] + 0.05
    truncated = [d for d in r.detections if d.t <= cutoff]
    sr = analyze_detections(truncated)
    assert sr.runs and sr.runs[-1].end_reason == "video_end"


def test_shuffle_invariance_end_to_end():
    """Spec property test: catch counts invariant under detection reordering."""
    r = simulate_cascade(n_throws=12, fps=30.0, noise=0.002, seed=4)
    sr_a = analyze_detections(r.detections)
    shuffled = list(r.detections)
    np.random.default_rng(0).shuffle(shuffled)
    sr_b = analyze_detections(shuffled)
    assert sr_a == sr_b


def test_empty_input():
    sr = analyze_detections([])
    assert sr.runs == [] and sr.arcs == [] and sr.drops == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_analyze.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'juggletrack.analyze'`.

- [ ] **Step 3: Implement `src/juggletrack/analyze.py`**

```python
"""Detections -> SessionResult: the full event core in one call (spec §3 data flow)."""
from __future__ import annotations

from pydantic import BaseModel

from juggletrack.arcs.extract import extract_arcs
from juggletrack.events.catches import derive_events
from juggletrack.events.drops import detect_drops
from juggletrack.events.handline import estimate_hand_line
from juggletrack.events.runs import segment_runs
from juggletrack.types import Detection, SessionResult


class AnalyzeConfig(BaseModel):
    g_range: tuple[float, float] = (0.5, 8.0)
    resid_tol: float = 0.02
    min_points: int = 6
    min_duration: float = 0.15
    gap_factor: float = 1.3
    min_arcs: int = 3
    video_end_margin: float = 1.0  # in throw periods


def analyze_detections(
    dets: list[Detection], config: AnalyzeConfig | None = None
) -> SessionResult:
    cfg = config or AnalyzeConfig()
    if not dets:
        return SessionResult()

    arcs = extract_arcs(
        dets, g_range=cfg.g_range, resid_tol=cfg.resid_tol,
        min_points=cfg.min_points, min_duration=cfg.min_duration,
    )
    hand_line = estimate_hand_line(arcs)
    throws, catches = derive_events(arcs, hand_line)
    runs = segment_runs(
        arcs, throws, catches, hand_line,
        gap_factor=cfg.gap_factor, min_arcs=cfg.min_arcs,
    )
    drops, runs = detect_drops(arcs, runs, hand_line)

    t_last = max(d.t for d in dets)
    final: list = []
    for run in runs:
        period = run.period_s or 0.5
        if run.end_reason == "stop" and t_last - run.end_t < cfg.video_end_margin * period:
            run = run.model_copy(update={"end_reason": "video_end"})
        final.append(run)

    return SessionResult(
        runs=final, drops=drops, arcs=arcs, hand_line_y=hand_line,
        meta={"n_detections": len(dets), "t_last": t_last},
    )
```

- [ ] **Step 4: Run full suite**

Run: `uv run pytest -v`
Expected: ALL PASS (≈40 tests). Also run `uv run ruff check src tests` — expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/juggletrack/analyze.py tests/test_analyze.py
git commit -m "feat: end-to-end analyze_detections with video_end classification

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Plan Self-Review Notes (already applied)

- **Spec coverage:** spec §4 arc extraction (Task 5), sub-frame events + hand line (Task 6), runs + periodicity validator (Task 7), three-signal drops + stop-vs-drop (Task 8), §7 synthetic-trajectory tests (Task 3 + every task), §7 property test (Tasks 5 & 9). Deliberately out of scope for Plan 1 (later plans): real detectors, pose wrists, ROI, CLI, overlay, eval harness, data tiers.
- **Type consistency:** `hand_line_crossings` introduced in Task 7 and used by Tasks 7–8; `derive_events` signature identical in Tasks 6–9; `Run.end_reason` literals match `types.py` exactly (`"drop" | "stop" | "video_end"`).
- **Known tuning risk:** thresholds (`resid_tol`, gap factors, margins) are physics-motivated defaults; if a test fails on a threshold, the expected values in tests are ground truth — fix the algorithm/threshold, never loosen a test beyond the tolerances written here without flagging it.



