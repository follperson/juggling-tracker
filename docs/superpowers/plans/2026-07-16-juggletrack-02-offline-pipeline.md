# Juggletrack Plan 2: Offline Pipeline on Real Video — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `juggletrack analyze video.mp4` works end-to-end on real footage — video → stock-YOLO ball detections → the Plan 1 event core → `analysis.json` + debug overlay video — plus an eval harness that scores results against hand-written labels.

**Architecture:** Thin real-world adapters around the proven event core. `video/` wraps cv2 capture; `detect/` defines a `BallDetector` protocol with a stock-COCO YOLO implementation (lazy ultralytics import) and a `FakeDetector` for tests; `pipeline/` runs detection over a video, persists intermediates (`detections.jsonl`), calls `analyze_detections`, and renders the debug overlay; `eval/` compares a `SessionResult` against labeled ground truth (run boundary IoU, catch-count error, drop precision/recall — spec §6); `cli.py` (typer) exposes `analyze` and `eval`. The event core (`types/sim/arcs/events/analyze`) is untouched except one carry-forward fix in `arcs/fit.py`.

**Tech Stack:** Python 3.12, `uv`, opencv-python, ultralytics (YOLO11n stock COCO weights), typer, pydantic v2, numpy, pytest.

**Plan sequence context:** Plan 2 of 5. Plan 1 (merged) built the event core on synthetic data. Plan 3 will add the labeling flywheel + fine-tuned detector; Plan 4 realtime; Plan 5 drop hardening. Carry-forward item from Plan 1's final review implemented here as Task 1 (fit.py rmse weighting) because real confidence-weighted detections arrive in this plan.

## Global Constraints

- Python 3.12; all package management via `uv` (`uv sync`, `uv run pytest`, `uv add`). Never pip/poetry.
- The event-core modules (`types.py`, `sim.py`, `arcs/`, `events/`, `analyze.py`) must remain importable WITHOUT cv2/ultralytics/typer installed-or-imported. cv2 imports are confined to `video/` and `pipeline/`; ultralytics is imported lazily inside `YOLODetector.__init__` only; typer only in `cli.py`.
- `Detection` coordinates stay normalized [0,1] with **y increasing downward**; `x, y` are box centers. Frames are BGR `np.ndarray` (cv2 convention, shape `(H, W, 3)`, dtype uint8).
- COCO class id for "sports ball" is **32** (0-indexed, 80-class list).
- Tests needing model weights/downloads are marked `@pytest.mark.detector` and deselected by default via pytest `addopts = "-m 'not detector'"`.
- The existing 56 tests must stay green; never loosen an existing test.
- All randomness seeded; every non-`detector` test deterministic and network-free.
- Conventional commits ending with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Real videos for the validation task live in the MAIN checkout: `/Users/andrew.follmann/personal-projects/juggling/data/raw/` (this worktree has no copy — use absolute paths). Stock weights may exist at `/Users/andrew.follmann/personal-projects/juggling/yolo11n.pt`.

## File Structure

```
pyproject.toml                          # + opencv-python, ultralytics, typer deps; [project.scripts]; pytest markers
src/juggletrack/arcs/fit.py             # Task 1: rmse weight fix (w -> w**2)
src/juggletrack/video/__init__.py       # empty
src/juggletrack/video/reader.py         # Task 2: VideoInfo + VideoReader (cv2)
src/juggletrack/detect/__init__.py      # Task 3: BallDetector protocol
src/juggletrack/detect/fake.py          # Task 3: FakeDetector (per-frame lookup)
src/juggletrack/detect/yolo.py          # Task 4: YOLODetector (lazy ultralytics) + pure conversion fn
src/juggletrack/pipeline/__init__.py    # empty
src/juggletrack/pipeline/offline.py     # Task 5: detect_video, jsonl io, analyze_video
src/juggletrack/pipeline/overlay.py     # Task 6: render_overlay
src/juggletrack/eval/__init__.py        # empty
src/juggletrack/eval/labels.py          # Task 7: LabeledRun, VideoLabels
src/juggletrack/eval/metrics.py         # Task 7: temporal_iou, evaluate_session, EvalReport
src/juggletrack/cli.py                  # Task 8: typer app (analyze, eval)
tests/helpers.py                        # Task 2: write_test_video helper
tests/test_fit.py                       # Task 1: extend
tests/test_video_reader.py              # Task 2
tests/test_detect.py                    # Task 3 + Task 4 (pure parts)
tests/test_detect_integration.py        # Task 4 (marked 'detector')
tests/test_pipeline_offline.py          # Task 5
tests/test_overlay.py                   # Task 6
tests/test_eval.py                      # Task 7
tests/test_cli.py                       # Task 8
docs/superpowers/plans/2026-07-16-plan2-validation-findings.md  # Task 9 output
```

---

### Task 1: Carry-forward — make `fit_arc`'s rmse consistent with the weighted objective

**Files:**
- Modify: `src/juggletrack/arcs/fit.py`
- Test: `tests/test_fit.py` (extend)

**Interfaces:**
- Consumes/Produces: `fit_arc(arr, arc_id=-1) -> Arc` — signature unchanged; only the `rmse` computation changes.

**Why:** `np.polyfit(..., w=w)` minimizes `Σ(w_i·resid_i)²`, but `rmse` currently averages `resid²` with weights `w` (not `w²`). Inert while all confidences are 1.0 (synthetic data); wrong the moment real confidence-weighted detections arrive — a low-confidence outlier would inflate `rmse` ~10× more than the fit objective says it should, and every `resid_tol` comparison in `extract.py` consumes this rmse.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fit.py`:

```python
def test_rmse_downweights_outlier_consistently_with_fit():
    """rmse must reflect the fit's own objective: weights enter squared.

    A near-zero-confidence outlier barely moves the fit (already tested);
    it must also barely move rmse. With linear weights the outlier's
    contribution is ~w*res^2 (=> rmse ~5.5e-3 here); with the correct w^2
    weighting it is ~w^2*res^2 (=> rmse ~5.5e-4).
    """
    arr = flight_points()
    outlier = arr[len(arr) // 2].copy()
    outlier[2] += 0.3
    outlier[3] = 0.01
    arc = fit_arc(np.vstack([arr, outlier]))
    assert arc.rmse < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_fit.py::test_rmse_downweights_outlier_consistently_with_fit -v`
Expected: FAIL — `arc.rmse` ≈ 5.5e-3, assertion `< 1e-3` fails.

- [ ] **Step 3: Fix the rmse computation**

In `src/juggletrack/arcs/fit.py`, change the rmse line:

```python
    resid = (ay * dt * dt + by * dt + cy) - arr[:, 2]
    # polyfit minimizes sum((w*resid)^2), so the consistent diagnostic
    # averages resid^2 with weights w^2 (uniform-confidence data unaffected).
    rmse = float(np.sqrt(np.average(resid**2, weights=w**2)))
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -v`
Expected: 57 passed (56 + 1 new). Uniform-weight tests are unaffected because `w**2 == w` when all weights are 1.0.

- [ ] **Step 5: Commit**

```bash
git add src/juggletrack/arcs/fit.py tests/test_fit.py
git commit -m "fix: rmse weighting consistent with polyfit objective (w^2)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Video reader (`video/reader.py`) + synthetic-video test helper

**Files:**
- Modify: `pyproject.toml` (add `opencv-python>=4.10` to `[project].dependencies`)
- Create: `src/juggletrack/video/__init__.py` (empty), `src/juggletrack/video/reader.py`, `tests/helpers.py`
- Test: `tests/test_video_reader.py`

**Interfaces:**
- Produces (consumed by Tasks 4–6, 8, 9):
  - `VideoInfo(path: str, fps: float, width: int, height: int, frame_count: int, duration: float)` (pydantic)
  - `VideoReader(path: str | Path)` — raises `FileNotFoundError` for missing path, `ValueError` if cv2 cannot open; attribute `info: VideoInfo`; method `frames() -> Iterator[tuple[int, float, np.ndarray]]` yielding `(frame_idx, t_seconds, bgr_frame)`; `release()`; context-manager support (`__enter__`/`__exit__` call release).
  - Test helper `write_test_video(path, n_frames=60, fps=30.0, size=(320, 240)) -> list[tuple[float, float]]` — writes an mp4 of a white circle moving on black, returns the circle's normalized (x, y) center per frame.

- [ ] **Step 1: Add the dependency**

Run: `uv add "opencv-python>=4.10"`
Expected: `pyproject.toml` gains the dep; `uv.lock` updated; `uv run python -c "import cv2; print(cv2.__version__)"` prints ≥ 4.10.

- [ ] **Step 2: Write the test helper**

`tests/helpers.py`:

```python
"""Shared test utilities for video-facing tests."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_test_video(
    path: str | Path,
    n_frames: int = 60,
    fps: float = 30.0,
    size: tuple[int, int] = (320, 240),
) -> list[tuple[float, float]]:
    """Write an mp4 of a white circle orbiting on black.

    Returns the circle's normalized (x, y) center for each frame.
    """
    w, h = size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open (mp4v codec missing?)")
    centers: list[tuple[float, float]] = []
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        phase = 2 * np.pi * i / n_frames
        cx = 0.5 + 0.3 * np.cos(phase)
        cy = 0.5 + 0.3 * np.sin(phase)
        cv2.circle(frame, (int(cx * w), int(cy * h)), 10, (255, 255, 255), -1)
        writer.write(frame)
        centers.append((cx, cy))
    writer.release()
    return centers
```

- [ ] **Step 3: Write the failing tests**

`tests/test_video_reader.py`:

```python
import numpy as np
import pytest

from juggletrack.video.reader import VideoReader
from tests.helpers import write_test_video


@pytest.fixture()
def video_path(tmp_path):
    path = tmp_path / "orbit.mp4"
    write_test_video(path, n_frames=60, fps=30.0, size=(320, 240))
    return path


def test_info_fields(video_path):
    with VideoReader(video_path) as reader:
        info = reader.info
        assert info.fps == pytest.approx(30.0, rel=0.01)
        assert (info.width, info.height) == (320, 240)
        assert info.frame_count == 60
        assert info.duration == pytest.approx(2.0, rel=0.02)


def test_frames_iteration(video_path):
    with VideoReader(video_path) as reader:
        frames = list(reader.frames())
    assert len(frames) == 60
    idx0, t0, frame0 = frames[0]
    assert idx0 == 0 and t0 == 0.0
    assert frame0.shape == (240, 320, 3) and frame0.dtype == np.uint8
    idx9, t9, _ = frames[9]
    assert idx9 == 9 and t9 == pytest.approx(9 / 30.0)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        VideoReader("/nonexistent/nope.mp4")


def test_unreadable_file_raises(tmp_path):
    bogus = tmp_path / "not_a_video.mp4"
    bogus.write_bytes(b"this is not an mp4")
    with pytest.raises(ValueError):
        VideoReader(bogus)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_video_reader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'juggletrack.video'`.

- [ ] **Step 5: Implement `src/juggletrack/video/reader.py`** (and empty `src/juggletrack/video/__init__.py`)

```python
"""cv2-backed video reading. The only module (with pipeline/) allowed to import cv2.

Phone videos carry rotation metadata; OpenCV >= 4.5 applies it automatically
(CAP_PROP_ORIENTATION_AUTO defaults on), so width/height are post-rotation.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel


class VideoInfo(BaseModel):
    path: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration: float


class VideoReader:
    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"video not found: {path}")
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise ValueError(f"cv2 could not open video: {path}")
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise ValueError(f"video has no readable frames/fps: {path}")
        self.info = VideoInfo(
            path=str(path),
            fps=fps,
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=frame_count,
            duration=frame_count / fps,
        )

    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        """Yield (frame_idx, timestamp_seconds, BGR frame) from the start."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield idx, idx / self.info.fps, frame
            idx += 1

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_video_reader.py -v && uv run pytest -q`
Expected: 4 passed; full suite 61 passed. (If `test_info_fields` fails on `frame_count`, the mp4v codec on this machine wrote a different count — investigate `write_test_video`, do not loosen the test.)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/juggletrack/video/ tests/helpers.py tests/test_video_reader.py
git commit -m "feat: cv2 video reader with synthetic-video test helper

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Detector protocol + FakeDetector

**Files:**
- Create: `src/juggletrack/detect/__init__.py`, `src/juggletrack/detect/fake.py`
- Test: `tests/test_detect.py`

**Interfaces:**
- Produces (consumed by Tasks 4, 5, 8):
  - `BallDetector` (Protocol, in `detect/__init__.py`): `detect(self, frame: np.ndarray, frame_idx: int, t: float) -> list[Detection]`
  - `FakeDetector(detections: list[Detection])` — returns the pre-supplied detections whose `frame_idx` matches, ignoring the frame pixels. Lets every pipeline test run simulator ground truth through the real pipeline machinery with zero ML.

- [ ] **Step 1: Write the failing tests**

`tests/test_detect.py`:

```python
import numpy as np

from juggletrack.detect import BallDetector
from juggletrack.detect.fake import FakeDetector
from juggletrack.sim import simulate_cascade


def test_fake_detector_returns_per_frame_detections():
    r = simulate_cascade(n_throws=5, fps=30.0, seed=1)
    det = FakeDetector(r.detections)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    expected_frames = {d.frame_idx for d in r.detections}
    some_frame = min(expected_frames)
    got = det.detect(frame, some_frame, some_frame / 30.0)
    assert got == [d for d in r.detections if d.frame_idx == some_frame]


def test_fake_detector_empty_for_unknown_frame():
    det = FakeDetector([])
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    assert det.detect(frame, 999, 33.3) == []


def test_fake_detector_satisfies_protocol():
    det: BallDetector = FakeDetector([])
    assert callable(det.detect)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_detect.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'juggletrack.detect'`.

- [ ] **Step 3: Implement**

`src/juggletrack/detect/__init__.py`:

```python
"""Ball detection interfaces. Implementations must NOT import cv2/torch at module load."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from juggletrack.types import Detection


@runtime_checkable
class BallDetector(Protocol):
    def detect(self, frame: np.ndarray, frame_idx: int, t: float) -> list[Detection]:
        """Return ball detections for one BGR frame (normalized coordinates)."""
        ...
```

`src/juggletrack/detect/fake.py`:

```python
"""Deterministic detector double: replays pre-supplied detections by frame index."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from juggletrack.types import Detection


class FakeDetector:
    def __init__(self, detections: list[Detection]):
        self._by_frame: dict[int, list[Detection]] = defaultdict(list)
        for d in detections:
            self._by_frame[d.frame_idx].append(d)

    def detect(self, frame: np.ndarray, frame_idx: int, t: float) -> list[Detection]:
        return list(self._by_frame.get(frame_idx, []))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_detect.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/juggletrack/detect/ tests/test_detect.py
git commit -m "feat: BallDetector protocol and FakeDetector test double

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: YOLODetector (stock COCO, lazy ultralytics)

**Files:**
- Modify: `pyproject.toml` (add `ultralytics>=8.3` dep; add pytest markers config)
- Create: `src/juggletrack/detect/yolo.py`
- Test: `tests/test_detect.py` (extend, pure part), `tests/test_detect_integration.py` (marked `detector`)

**Interfaces:**
- Produces (consumed by Tasks 5, 8, 9):
  - `detections_from_xywhn(xywhn: np.ndarray, confs: np.ndarray, frame_idx: int, t: float) -> list[Detection]` — pure conversion from ultralytics' normalized-center boxes; unit-testable without ultralytics.
  - `YOLODetector(model_path: str = "yolo11n.pt", conf: float = 0.05, imgsz: int = 640, classes: tuple[int, ...] = (32,), device: str | None = None)` implementing `BallDetector`. Class 32 = COCO "sports ball". `from ultralytics import YOLO` happens inside `__init__` (module import stays light).

- [ ] **Step 1: Add dependency and pytest markers**

Run: `uv add "ultralytics>=8.3"`
Then in `pyproject.toml`, replace the `[tool.pytest.ini_options]` section with:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-m 'not detector'"
markers = [
    "detector: needs YOLO weights (network download on first run); run with -m detector",
]
```

Run: `uv run pytest -q` — Expected: existing count unchanged (61 passed), proving addopts breaks nothing.

- [ ] **Step 2: Write the failing pure-conversion tests**

Append to `tests/test_detect.py`:

```python
def test_detections_from_xywhn_maps_fields():
    import numpy as np

    from juggletrack.detect.yolo import detections_from_xywhn

    xywhn = np.array([[0.5, 0.6, 0.05, 0.08], [0.2, 0.3, 0.04, 0.04]])
    confs = np.array([0.9, 0.15])
    dets = detections_from_xywhn(xywhn, confs, frame_idx=7, t=7 / 30.0)
    assert len(dets) == 2
    d = dets[0]
    assert (d.frame_idx, d.t) == (7, 7 / 30.0)
    assert (d.x, d.y, d.w, d.h) == (0.5, 0.6, 0.05, 0.08)
    assert d.confidence == 0.9


def test_detections_from_xywhn_empty():
    import numpy as np

    from juggletrack.detect.yolo import detections_from_xywhn

    assert detections_from_xywhn(np.zeros((0, 4)), np.zeros(0), 0, 0.0) == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_detect.py -v`
Expected: 2 new FAIL with `ModuleNotFoundError: No module named 'juggletrack.detect.yolo'`; 3 prior pass.

- [ ] **Step 4: Implement `src/juggletrack/detect/yolo.py`**

```python
"""Stock-COCO YOLO ball detector. ultralytics is imported lazily in __init__
so `import juggletrack` (and the whole event core) stays torch-free.
"""
from __future__ import annotations

import numpy as np

from juggletrack.types import Detection

SPORTS_BALL_CLASS = 32  # COCO 80-class index


def detections_from_xywhn(
    xywhn: np.ndarray, confs: np.ndarray, frame_idx: int, t: float
) -> list[Detection]:
    """Convert ultralytics normalized-center boxes to Detections."""
    return [
        Detection(
            frame_idx=frame_idx, t=t,
            x=float(cx), y=float(cy), w=float(w), h=float(h),
            confidence=float(c),
        )
        for (cx, cy, w, h), c in zip(xywhn, confs)
    ]


class YOLODetector:
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf: float = 0.05,
        imgsz: int = 640,
        classes: tuple[int, ...] = (SPORTS_BALL_CLASS,),
        device: str | None = None,
    ):
        from ultralytics import YOLO  # lazy: torch loads only when a real detector is built

        self._model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.classes = classes
        self.device = device

    def detect(self, frame: np.ndarray, frame_idx: int, t: float) -> list[Detection]:
        result = self._model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=list(self.classes),
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        return detections_from_xywhn(
            boxes.xywhn.cpu().numpy(), boxes.conf.cpu().numpy(), frame_idx, t
        )
```

- [ ] **Step 5: Write the integration test (marked, deselected by default)**

`tests/test_detect_integration.py`:

```python
"""Integration tests that exercise real YOLO weights. Deselected by default
(pytest addopts -m 'not detector'); run with: uv run pytest -m detector -v
"""
from pathlib import Path

import pytest

pytestmark = pytest.mark.detector

REAL_VIDEO = Path("/Users/andrew.follmann/personal-projects/juggling/data/raw/af2.mp4")
LOCAL_WEIGHTS = Path("/Users/andrew.follmann/personal-projects/juggling/yolo11n.pt")


def test_yolo_detects_balls_in_real_juggling_video():
    if not REAL_VIDEO.exists():
        pytest.skip("real juggling video not available on this machine")
    from juggletrack.detect.yolo import YOLODetector
    from juggletrack.video.reader import VideoReader

    model = str(LOCAL_WEIGHTS) if LOCAL_WEIGHTS.exists() else "yolo11n.pt"
    det = YOLODetector(model_path=model)
    found = []
    with VideoReader(REAL_VIDEO) as reader:
        for idx, t, frame in reader.frames():
            if idx >= 90:  # first 3 seconds
                break
            found.extend(det.detect(frame, idx, t))
    assert found, "stock YOLO found zero sports balls in 90 frames of juggling"
    for d in found:
        assert 0.0 <= d.x <= 1.0 and 0.0 <= d.y <= 1.0
        assert 0.0 < d.w <= 1.0 and 0.0 < d.h <= 1.0
        assert 0.0 < d.confidence <= 1.0
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_detect.py -v` — Expected: 5 passed.
Run: `uv run pytest -q` — Expected: 63 passed (integration file auto-deselected).
Run: `uv run pytest -m detector -v` — Expected: 1 passed (or skipped if the video is missing). This downloads yolo11n.pt on first run if no local weights; report the detection count you saw.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/juggletrack/detect/yolo.py tests/test_detect.py tests/test_detect_integration.py
git commit -m "feat: stock-COCO YOLO ball detector with lazy ultralytics import

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Offline pipeline — detect, persist, analyze (`pipeline/offline.py`)

**Files:**
- Create: `src/juggletrack/pipeline/__init__.py` (empty), `src/juggletrack/pipeline/offline.py`
- Test: `tests/test_pipeline_offline.py`

**Interfaces:**
- Consumes: `VideoReader`, `BallDetector`, `analyze_detections(dets, config) -> SessionResult`, `AnalyzeConfig`, `Detection`.
- Produces (consumed by Tasks 6, 8, 9):
  - `detect_video(reader: VideoReader, detector: BallDetector, *, stride: int = 1) -> list[Detection]` — runs the detector over every `stride`-th frame, concatenated in frame order.
  - `save_detections_jsonl(dets: list[Detection], path: str | Path) -> None` / `load_detections_jsonl(path) -> list[Detection]` — one `Detection.model_dump_json()` per line.
  - `analyze_video(video_path: str | Path, detector: BallDetector, *, config: AnalyzeConfig | None = None, out_dir: str | Path | None = None, save_intermediates: bool = False, stride: int = 1) -> SessionResult` — full offline run; enriches `SessionResult.meta` with `video_path, fps, width, height, frame_count, duration_s, stride`; when `out_dir` is given, writes `out_dir/analysis.json` (and `out_dir/detections.jsonl` when `save_intermediates`), creating the directory.

- [ ] **Step 1: Write the failing tests**

`tests/test_pipeline_offline.py`:

```python
import json

import pytest

from juggletrack.analyze import analyze_detections
from juggletrack.detect.fake import FakeDetector
from juggletrack.pipeline.offline import (
    analyze_video,
    detect_video,
    load_detections_jsonl,
    save_detections_jsonl,
)
from juggletrack.sim import simulate_cascade
from juggletrack.types import SessionResult
from juggletrack.video.reader import VideoReader
from tests.helpers import write_test_video


@pytest.fixture()
def sim():
    return simulate_cascade(n_throws=12, fps=30.0, noise=0.003, dropout=0.1, seed=1)


@pytest.fixture()
def video_path(tmp_path, sim):
    n_frames = max(d.frame_idx for d in sim.detections) + 1
    path = tmp_path / "cascade.mp4"
    write_test_video(path, n_frames=n_frames, fps=30.0)
    return path


def test_jsonl_roundtrip(tmp_path, sim):
    path = tmp_path / "dets.jsonl"
    save_detections_jsonl(sim.detections, path)
    assert load_detections_jsonl(path) == sim.detections


def test_detect_video_replays_fake_detections(video_path, sim):
    with VideoReader(video_path) as reader:
        dets = detect_video(reader, FakeDetector(sim.detections))
    assert dets == sorted(sim.detections, key=lambda d: (d.frame_idx,))


def test_detect_video_stride_skips_frames(video_path, sim):
    with VideoReader(video_path) as reader:
        dets = detect_video(reader, FakeDetector(sim.detections), stride=2)
    assert dets and all(d.frame_idx % 2 == 0 for d in dets)


def test_analyze_video_end_to_end(tmp_path, video_path, sim):
    out = tmp_path / "out"
    sr = analyze_video(
        video_path, FakeDetector(sim.detections),
        out_dir=out, save_intermediates=True,
    )
    # same event results as feeding detections straight to the core
    direct = analyze_detections(sim.detections)
    assert [r.catches for r in sr.runs] == [r.catches for r in direct.runs]
    assert sr.runs[0].catches == 12
    assert sr.drops == direct.drops
    # meta enrichment
    assert sr.meta["fps"] == pytest.approx(30.0, rel=0.01)
    assert sr.meta["video_path"] == str(video_path)
    assert sr.meta["stride"] == 1
    # artifacts on disk
    written = SessionResult.model_validate(
        json.loads((out / "analysis.json").read_text())
    )
    assert written == sr
    assert load_detections_jsonl(out / "detections.jsonl") == sorted(
        sim.detections, key=lambda d: (d.frame_idx,)
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline_offline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'juggletrack.pipeline'`.

- [ ] **Step 3: Implement `src/juggletrack/pipeline/offline.py`** (and empty `src/juggletrack/pipeline/__init__.py`)

```python
"""Offline video pipeline: video -> detections -> event core -> artifacts.

Intermediates are persisted (spec: stages re-runnable independently) —
`detections.jsonl` lets you re-tune the event core without re-detecting.
"""
from __future__ import annotations

from pathlib import Path

from juggletrack.analyze import AnalyzeConfig, analyze_detections
from juggletrack.detect import BallDetector
from juggletrack.types import Detection, SessionResult
from juggletrack.video.reader import VideoReader


def detect_video(
    reader: VideoReader, detector: BallDetector, *, stride: int = 1
) -> list[Detection]:
    dets: list[Detection] = []
    for idx, t, frame in reader.frames():
        if idx % stride:
            continue
        dets.extend(detector.detect(frame, idx, t))
    return dets


def save_detections_jsonl(dets: list[Detection], path: str | Path) -> None:
    with open(path, "w") as f:
        for d in dets:
            f.write(d.model_dump_json() + "\n")


def load_detections_jsonl(path: str | Path) -> list[Detection]:
    with open(path) as f:
        return [Detection.model_validate_json(line) for line in f if line.strip()]


def analyze_video(
    video_path: str | Path,
    detector: BallDetector,
    *,
    config: AnalyzeConfig | None = None,
    out_dir: str | Path | None = None,
    save_intermediates: bool = False,
    stride: int = 1,
) -> SessionResult:
    with VideoReader(video_path) as reader:
        info = reader.info
        dets = detect_video(reader, detector, stride=stride)

    session = analyze_detections(dets, config)
    session.meta.update(
        video_path=info.path, fps=info.fps, width=info.width, height=info.height,
        frame_count=info.frame_count, duration_s=info.duration, stride=stride,
    )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "analysis.json").write_text(session.model_dump_json(indent=2))
        if save_intermediates:
            save_detections_jsonl(dets, out / "detections.jsonl")
    return session
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline_offline.py -v && uv run pytest -q`
Expected: 4 passed; full suite 67 passed. (If `test_analyze_video_end_to_end` finds `written != sr`: pydantic float serialization is stable through json round-trips for these models — investigate rather than weaken the equality.)

- [ ] **Step 5: Commit**

```bash
git add src/juggletrack/pipeline/ tests/test_pipeline_offline.py
git commit -m "feat: offline pipeline with jsonl intermediates and analysis.json output

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Debug overlay renderer (`pipeline/overlay.py`)

**Files:**
- Create: `src/juggletrack/pipeline/overlay.py`
- Test: `tests/test_overlay.py`

**Interfaces:**
- Consumes: `VideoReader`, `SessionResult`, `Arc` methods (`y_at/x_at`), `derive_events` (re-derives catch/throw times for the HUD), `Detection`.
- Produces (consumed by Task 8):
  - `render_overlay(video_path: str | Path, session: SessionResult, out_path: str | Path, *, detections: list[Detection] | None = None, tail_s: float = 0.4) -> None` — writes an mp4 with: horizontal hand line; detection dots (if provided); arc trajectory tails (last `tail_s` seconds of each active arc, sampled every 1/60 s); HUD text (run number, catches-so-far in the active run, drop count); a red X at each drop location for 0.5 s after it.

**What "catches so far" means:** re-derive events with `derive_events(session.arcs, session.hand_line_y)` and count catches with `t <= current frame t` belonging to the active run's `arc_ids`.

- [ ] **Step 1: Write the failing tests**

`tests/test_overlay.py`:

```python
import pytest

from juggletrack.analyze import analyze_detections
from juggletrack.pipeline.overlay import render_overlay
from juggletrack.sim import simulate_cascade
from juggletrack.video.reader import VideoReader
from tests.helpers import write_test_video


@pytest.fixture()
def setup(tmp_path):
    sim = simulate_cascade(n_throws=10, fps=30.0, drop_at_throw=6, seed=1)
    n_frames = max(d.frame_idx for d in sim.detections) + 1
    video = tmp_path / "in.mp4"
    write_test_video(video, n_frames=n_frames, fps=30.0)
    session = analyze_detections(sim.detections)
    return sim, video, session, n_frames


def test_overlay_writes_decodable_video(tmp_path, setup):
    sim, video, session, n_frames = setup
    out = tmp_path / "overlay.mp4"
    render_overlay(video, session, out, detections=sim.detections)
    assert out.exists() and out.stat().st_size > 0
    with VideoReader(out) as reader:
        assert reader.info.frame_count == n_frames
        assert (reader.info.width, reader.info.height) == (320, 240)
        # decoding must actually work, not just headers
        assert sum(1 for _ in reader.frames()) == n_frames


def test_overlay_draws_on_frames(tmp_path, setup):
    """Overlay frames must differ from the input (something was drawn)."""
    import numpy as np

    sim, video, session, n_frames = setup
    out = tmp_path / "overlay.mp4"
    render_overlay(video, session, out, detections=sim.detections)
    with VideoReader(video) as a, VideoReader(out) as b:
        fa = next(iter(a.frames()))[2]
        fb = next(iter(b.frames()))[2]
    assert np.abs(fa.astype(int) - fb.astype(int)).sum() > 0


def test_overlay_without_detections(tmp_path, setup):
    _, video, session, _ = setup
    out = tmp_path / "overlay2.mp4"
    render_overlay(video, session, out)
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_overlay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'juggletrack.pipeline.overlay'`.

- [ ] **Step 3: Implement `src/juggletrack/pipeline/overlay.py`**

```python
"""Debug overlay: draw what the event core concluded on top of the source video."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from juggletrack.events.catches import derive_events
from juggletrack.types import Detection, SessionResult
from juggletrack.video.reader import VideoReader

_GREEN = (0, 200, 0)
_YELLOW = (0, 220, 220)
_RED = (0, 0, 255)
_WHITE = (240, 240, 240)


def render_overlay(
    video_path: str | Path,
    session: SessionResult,
    out_path: str | Path,
    *,
    detections: list[Detection] | None = None,
    tail_s: float = 0.4,
) -> None:
    dets_by_frame: dict[int, list[Detection]] = defaultdict(list)
    for d in detections or []:
        dets_by_frame[d.frame_idx].append(d)

    _, catches = derive_events(session.arcs, session.hand_line_y)
    catch_times_by_run = [
        sorted(c.t for c in catches if c.arc_id in set(run.arc_ids))
        for run in session.runs
    ]

    with VideoReader(video_path) as reader:
        w, h = reader.info.width, reader.info.height
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), reader.info.fps, (w, h)
        )
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter failed to open: {out_path}")

        hand_y_px = int(session.hand_line_y * h)
        for idx, t, frame in reader.frames():
            cv2.line(frame, (0, hand_y_px), (w, hand_y_px), _YELLOW, 1)

            for d in dets_by_frame.get(idx, []):
                cv2.circle(frame, (int(d.x * w), int(d.y * h)), 4, _WHITE, 1)

            for arc in session.arcs:
                if not (arc.t_start <= t <= arc.t_end + 0.1):
                    continue
                t0 = max(arc.t_start, t - tail_s)
                ts = np.arange(t0, min(t, arc.t_end) + 1e-9, 1 / 60)
                pts = np.array(
                    [[int(arc.x_at(tt) * w), int(arc.y_at(tt) * h)] for tt in ts]
                )
                if len(pts) >= 2:
                    cv2.polylines(frame, [pts], False, _GREEN, 2)

            active = next(
                (i for i, r in enumerate(session.runs) if r.start_t - 0.5 <= t <= r.end_t + 0.5),
                None,
            )
            drops_so_far = sum(1 for d in session.drops if d.t <= t)
            if active is not None:
                caught = sum(1 for ct in catch_times_by_run[active] if ct <= t)
                hud = f"run {active + 1}/{len(session.runs)}  catches {caught}  drops {drops_so_far}"
            else:
                hud = f"runs {len(session.runs)}  drops {drops_so_far}"
            cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 2)

            for drop in session.drops:
                if drop.t <= t <= drop.t + 0.5:
                    x, y = int(drop.x * w), int(drop.y * h)
                    cv2.drawMarker(frame, (x, y), _RED, cv2.MARKER_TILTED_CROSS, 24, 3)

            writer.write(frame)
        writer.release()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_overlay.py -v && uv run pytest -q`
Expected: 3 passed; full suite 70 passed.

- [ ] **Step 5: Commit**

```bash
git add src/juggletrack/pipeline/overlay.py tests/test_overlay.py
git commit -m "feat: debug overlay renderer (hand line, arcs, HUD, drop markers)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Eval harness — labels schema + metrics (`eval/`)

**Files:**
- Create: `src/juggletrack/eval/__init__.py` (empty), `src/juggletrack/eval/labels.py`, `src/juggletrack/eval/metrics.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `SessionResult`, `Run`.
- Produces (consumed by Task 8):
  - `LabeledRun(start_t: float, end_t: float, catches: int, end_reason: str = "stop")`, `VideoLabels(video: str, runs: list[LabeledRun] = [], drops: list[float] = [])` (pydantic; a labels file is one `VideoLabels` JSON document)
  - `temporal_iou(a_start, a_end, b_start, b_end) -> float`
  - `RunMatch(pred_idx: int, label_idx: int, iou: float, catch_error: int)`
  - `EvalReport(video, n_labeled_runs, n_pred_runs, matches: list[RunMatch], unmatched_labeled: list[int], unmatched_pred: list[int], frac_runs_iou90: float, frac_catch_within_1: float, drop_tp: int, drop_fp: int, drop_fn: int, drop_precision: float, drop_recall: float)`
  - `evaluate_session(session: SessionResult, labels: VideoLabels, *, drop_tol_s: float = 1.0, min_match_iou: float = 0.1) -> EvalReport`

**Metric semantics (spec §6, made precise):** greedy run matching — enumerate all (pred, label) pairs with `temporal_iou > min_match_iou`, sort by IoU descending, take pairs whose pred AND label are both still unmatched. `frac_runs_iou90` = matched pairs with IoU ≥ 0.9 ÷ **n_labeled_runs** (unmatched labeled runs count as failures). `frac_catch_within_1` = matched pairs with `|pred.catches − label.catches| ≤ 1` ÷ n_labeled_runs. Both are 1.0 when there are no labeled runs. Drops: greedy nearest-first matching of predicted drop times to labeled drop times within `drop_tol_s`; precision = tp/(tp+fp) (1.0 when no predictions), recall = tp/(tp+fn) (1.0 when no labeled drops).

- [ ] **Step 1: Write the failing tests**

`tests/test_eval.py`:

```python
import pytest

from juggletrack.eval.labels import LabeledRun, VideoLabels
from juggletrack.eval.metrics import evaluate_session, temporal_iou
from juggletrack.types import DropEvent, Run, SessionResult


def run(start, end, catches, reason="stop"):
    return Run(start_t=start, end_t=end, catches=catches, throws=catches,
               end_reason=reason, period_s=0.45, quality=0.8)


def session(runs=(), drops=()):
    return SessionResult(
        runs=list(runs),
        drops=[DropEvent(t=t, x=0.5, arc_id=None) for t in drops],
    )


def test_temporal_iou():
    assert temporal_iou(0, 10, 0, 10) == 1.0
    assert temporal_iou(0, 10, 5, 15) == pytest.approx(5 / 15)
    assert temporal_iou(0, 1, 2, 3) == 0.0


def test_perfect_match():
    sr = session(runs=[run(1.0, 6.0, 12)], drops=[6.0])
    labels = VideoLabels(video="v.mp4", runs=[LabeledRun(start_t=1.0, end_t=6.0, catches=12)],
                         drops=[6.2])
    rep = evaluate_session(sr, labels)
    assert rep.frac_runs_iou90 == 1.0
    assert rep.frac_catch_within_1 == 1.0
    assert rep.matches[0].catch_error == 0
    assert (rep.drop_tp, rep.drop_fp, rep.drop_fn) == (1, 0, 0)
    assert rep.drop_precision == 1.0 and rep.drop_recall == 1.0


def test_catch_error_and_boundary_miss():
    sr = session(runs=[run(1.0, 5.0, 9)])  # boundary short, catches off by 3
    labels = VideoLabels(video="v.mp4",
                         runs=[LabeledRun(start_t=1.0, end_t=6.0, catches=12)])
    rep = evaluate_session(sr, labels)
    assert rep.matches[0].catch_error == 3
    assert rep.matches[0].iou == pytest.approx(4 / 5)
    assert rep.frac_runs_iou90 == 0.0
    assert rep.frac_catch_within_1 == 0.0


def test_unmatched_runs_count_as_failures():
    sr = session(runs=[run(1.0, 6.0, 12), run(20.0, 22.0, 4)])  # 2nd is spurious
    labels = VideoLabels(video="v.mp4", runs=[
        LabeledRun(start_t=1.0, end_t=6.0, catches=12),
        LabeledRun(start_t=10.0, end_t=15.0, catches=10),  # missed entirely
    ])
    rep = evaluate_session(sr, labels)
    assert len(rep.matches) == 1
    assert rep.unmatched_labeled == [1]
    assert rep.unmatched_pred == [1]
    assert rep.frac_runs_iou90 == 0.5
    assert rep.frac_catch_within_1 == 0.5


def test_drop_precision_recall():
    sr = session(runs=[run(1, 6, 12)], drops=[6.0, 30.0])  # one true, one spurious
    labels = VideoLabels(video="v.mp4",
                         runs=[LabeledRun(start_t=1, end_t=6, catches=12)],
                         drops=[6.3, 12.0])  # 12.0 missed
    rep = evaluate_session(sr, labels)
    assert (rep.drop_tp, rep.drop_fp, rep.drop_fn) == (1, 1, 1)
    assert rep.drop_precision == 0.5 and rep.drop_recall == 0.5


def test_empty_everything():
    rep = evaluate_session(session(), VideoLabels(video="v.mp4"))
    assert rep.frac_runs_iou90 == 1.0 and rep.frac_catch_within_1 == 1.0
    assert rep.drop_precision == 1.0 and rep.drop_recall == 1.0


def test_labels_json_roundtrip(tmp_path):
    labels = VideoLabels(video="v.mp4",
                         runs=[LabeledRun(start_t=1, end_t=6, catches=12)], drops=[6.0])
    p = tmp_path / "labels.json"
    p.write_text(labels.model_dump_json(indent=2))
    assert VideoLabels.model_validate_json(p.read_text()) == labels
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'juggletrack.eval'`.

- [ ] **Step 3: Implement**

`src/juggletrack/eval/labels.py`:

```python
"""Hand-written ground-truth labels for a video (one JSON document per video)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class LabeledRun(BaseModel):
    start_t: float
    end_t: float
    catches: int
    end_reason: str = "stop"


class VideoLabels(BaseModel):
    video: str
    runs: list[LabeledRun] = Field(default_factory=list)
    drops: list[float] = Field(default_factory=list)
```

`src/juggletrack/eval/metrics.py`:

```python
"""Spec §6 metrics: run-boundary IoU, catch-count error, drop precision/recall."""
from __future__ import annotations

from pydantic import BaseModel, Field

from juggletrack.eval.labels import VideoLabels
from juggletrack.types import SessionResult


def temporal_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


class RunMatch(BaseModel):
    pred_idx: int
    label_idx: int
    iou: float
    catch_error: int


class EvalReport(BaseModel):
    video: str
    n_labeled_runs: int
    n_pred_runs: int
    matches: list[RunMatch] = Field(default_factory=list)
    unmatched_labeled: list[int] = Field(default_factory=list)
    unmatched_pred: list[int] = Field(default_factory=list)
    frac_runs_iou90: float
    frac_catch_within_1: float
    drop_tp: int
    drop_fp: int
    drop_fn: int
    drop_precision: float
    drop_recall: float


def evaluate_session(
    session: SessionResult,
    labels: VideoLabels,
    *,
    drop_tol_s: float = 1.0,
    min_match_iou: float = 0.1,
) -> EvalReport:
    pairs = sorted(
        (
            (temporal_iou(p.start_t, p.end_t, l.start_t, l.end_t), pi, li)
            for pi, p in enumerate(session.runs)
            for li, l in enumerate(labels.runs)
        ),
        key=lambda x: -x[0],
    )
    matches: list[RunMatch] = []
    used_p: set[int] = set()
    used_l: set[int] = set()
    for iou, pi, li in pairs:
        if iou <= min_match_iou or pi in used_p or li in used_l:
            continue
        used_p.add(pi)
        used_l.add(li)
        matches.append(RunMatch(
            pred_idx=pi, label_idx=li, iou=iou,
            catch_error=abs(session.runs[pi].catches - labels.runs[li].catches),
        ))

    n_l = len(labels.runs)
    frac_iou = sum(1 for m in matches if m.iou >= 0.9) / n_l if n_l else 1.0
    frac_catch = sum(1 for m in matches if m.catch_error <= 1) / n_l if n_l else 1.0

    pred_drops = sorted(d.t for d in session.drops)
    label_drops = sorted(labels.drops)
    drop_pairs = sorted(
        (
            (abs(pt - lt), pi, li)
            for pi, pt in enumerate(pred_drops)
            for li, lt in enumerate(label_drops)
        ),
        key=lambda x: x[0],
    )
    dp_used: set[int] = set()
    dl_used: set[int] = set()
    tp = 0
    for dist, pi, li in drop_pairs:
        if dist > drop_tol_s or pi in dp_used or li in dl_used:
            continue
        dp_used.add(pi)
        dl_used.add(li)
        tp += 1
    fp = len(pred_drops) - tp
    fn = len(label_drops) - tp

    return EvalReport(
        video=labels.video,
        n_labeled_runs=n_l,
        n_pred_runs=len(session.runs),
        matches=matches,
        unmatched_labeled=[i for i in range(n_l) if i not in used_l],
        unmatched_pred=[i for i in range(len(session.runs)) if i not in used_p],
        frac_runs_iou90=frac_iou,
        frac_catch_within_1=frac_catch,
        drop_tp=tp, drop_fp=fp, drop_fn=fn,
        drop_precision=tp / (tp + fp) if (tp + fp) else 1.0,
        drop_recall=tp / (tp + fn) if (tp + fn) else 1.0,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_eval.py -v && uv run pytest -q`
Expected: 7 passed; full suite 77 passed.

- [ ] **Step 5: Commit**

```bash
git add src/juggletrack/eval/ tests/test_eval.py
git commit -m "feat: eval harness — labels schema and spec metrics

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: CLI (`cli.py`) — `juggletrack analyze` and `juggletrack eval`

**Files:**
- Modify: `pyproject.toml` (add `typer>=0.12` dep and `[project.scripts]`)
- Create: `src/juggletrack/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `analyze_video`, `load_detections_jsonl`, `FakeDetector`, `render_overlay`, `evaluate_session`, `VideoLabels`, `SessionResult`.
- Produces: console script `juggletrack` with:
  - `analyze VIDEO [--out DIR] [--overlay/--no-overlay] [--save-intermediates] [--detections JSONL] [--model PATH] [--conf 0.05] [--imgsz 640] [--stride 1] [--device STR]` — default `--out` is `outputs/<video stem>`. With `--detections`, a `FakeDetector` replays the JSONL (no ultralytics import — this is also the re-analysis workflow for tuning the event core on saved intermediates). Prints a run/catch/drop summary; writes `analysis.json`, `overlay.mp4` (unless `--no-overlay`), `detections.jsonl` (when `--save-intermediates`).
  - `eval ANALYSIS_JSON LABELS_JSON [--out FILE]` — prints the metric summary; writes `EvalReport` JSON when `--out` given. Exit code 0.

- [ ] **Step 1: Add dependency and script entry**

Run: `uv add "typer>=0.12"`
Then add to `pyproject.toml`:

```toml
[project.scripts]
juggletrack = "juggletrack.cli:app"
```

- [ ] **Step 2: Write the failing tests**

`tests/test_cli.py`:

```python
import json

import pytest
from typer.testing import CliRunner

from juggletrack.pipeline.offline import save_detections_jsonl
from juggletrack.sim import simulate_cascade
from juggletrack.types import SessionResult
from tests.helpers import write_test_video

runner = CliRunner()


@pytest.fixture()
def workspace(tmp_path):
    sim = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    n_frames = max(d.frame_idx for d in sim.detections) + 1
    video = tmp_path / "cascade.mp4"
    write_test_video(video, n_frames=n_frames, fps=30.0)
    dets = tmp_path / "dets.jsonl"
    save_detections_jsonl(sim.detections, dets)
    return sim, video, dets, tmp_path


def test_analyze_with_saved_detections(workspace):
    from juggletrack.cli import app

    sim, video, dets, tmp = workspace
    out = tmp / "out"
    result = runner.invoke(app, [
        "analyze", str(video), "--out", str(out), "--detections", str(dets),
    ])
    assert result.exit_code == 0, result.output
    sr = SessionResult.model_validate(json.loads((out / "analysis.json").read_text()))
    assert sr.runs and sr.runs[0].catches == 12
    assert (out / "overlay.mp4").exists()
    assert "catches" in result.output


def test_analyze_no_overlay(workspace):
    from juggletrack.cli import app

    _, video, dets, tmp = workspace
    out = tmp / "out2"
    result = runner.invoke(app, [
        "analyze", str(video), "--out", str(out),
        "--detections", str(dets), "--no-overlay",
    ])
    assert result.exit_code == 0, result.output
    assert (out / "analysis.json").exists()
    assert not (out / "overlay.mp4").exists()


def test_eval_command(workspace, tmp_path):
    from juggletrack.cli import app

    sim, video, dets, tmp = workspace
    out = tmp / "out3"
    runner.invoke(app, [
        "analyze", str(video), "--out", str(out),
        "--detections", str(dets), "--no-overlay",
    ])
    labels = {
        "video": str(video),
        "runs": [{"start_t": sim.run_start, "end_t": sim.run_end, "catches": 12}],
        "drops": [],
    }
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels))
    report_path = tmp_path / "eval.json"
    result = runner.invoke(app, [
        "eval", str(out / "analysis.json"), str(labels_path),
        "--out", str(report_path),
    ])
    assert result.exit_code == 0, result.output
    report = json.loads(report_path.read_text())
    assert report["frac_catch_within_1"] == 1.0
    assert "catch" in result.output.lower()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'juggletrack.cli'`.

- [ ] **Step 4: Implement `src/juggletrack/cli.py`**

```python
"""juggletrack CLI: analyze videos, evaluate against labels."""
from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def analyze(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path | None = typer.Option(None, help="Output dir (default outputs/<stem>)"),
    overlay: bool = typer.Option(True, help="Render debug overlay video"),
    save_intermediates: bool = typer.Option(False, help="Save detections.jsonl"),
    detections: Path | None = typer.Option(
        None, exists=True, dir_okay=False,
        help="Replay saved detections.jsonl instead of running the detector",
    ),
    model: str = typer.Option("yolo11n.pt", help="YOLO weights path or name"),
    conf: float = typer.Option(0.05, help="Detector confidence threshold"),
    imgsz: int = typer.Option(640, help="Detector input size"),
    stride: int = typer.Option(1, help="Detect every Nth frame"),
    device: str | None = typer.Option(None, help="Torch device (mps/cpu/cuda)"),
) -> None:
    from juggletrack.pipeline.offline import analyze_video, load_detections_jsonl

    out_dir = out or Path("outputs") / video.stem
    if detections is not None:
        from juggletrack.detect.fake import FakeDetector

        detector = FakeDetector(load_detections_jsonl(detections))
    else:
        from juggletrack.detect.yolo import YOLODetector

        detector = YOLODetector(
            model_path=model, conf=conf, imgsz=imgsz, device=device
        )

    session = analyze_video(
        video, detector,
        out_dir=out_dir, save_intermediates=save_intermediates, stride=stride,
    )

    if overlay:
        from juggletrack.pipeline.overlay import render_overlay

        dets_for_overlay = None
        jsonl = out_dir / "detections.jsonl"
        if detections is not None:
            dets_for_overlay = load_detections_jsonl(detections)
        elif jsonl.exists():
            dets_for_overlay = load_detections_jsonl(jsonl)
        render_overlay(video, session, out_dir / "overlay.mp4",
                       detections=dets_for_overlay)

    for i, run in enumerate(session.runs):
        typer.echo(
            f"run {i + 1}: {run.start_t:.2f}-{run.end_t:.2f}s  "
            f"catches {run.catches}  end={run.end_reason}"
        )
    typer.echo(
        f"{len(session.runs)} run(s), {len(session.drops)} drop(s); "
        f"results in {out_dir}"
    )


@app.command()
def eval(
    analysis_json: Path = typer.Argument(..., exists=True, dir_okay=False),
    labels_json: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path | None = typer.Option(None, help="Write EvalReport JSON here"),
) -> None:
    from juggletrack.eval.labels import VideoLabels
    from juggletrack.eval.metrics import evaluate_session
    from juggletrack.types import SessionResult

    session = SessionResult.model_validate(json.loads(analysis_json.read_text()))
    labels = VideoLabels.model_validate_json(labels_json.read_text())
    report = evaluate_session(session, labels)

    typer.echo(f"video: {report.video}")
    typer.echo(f"runs: {report.n_pred_runs} predicted / {report.n_labeled_runs} labeled")
    typer.echo(f"run boundary IoU>=0.9: {report.frac_runs_iou90:.0%}")
    typer.echo(f"catch count within +/-1: {report.frac_catch_within_1:.0%}")
    typer.echo(
        f"drops: precision {report.drop_precision:.2f} "
        f"recall {report.drop_recall:.2f} "
        f"(tp={report.drop_tp} fp={report.drop_fp} fn={report.drop_fn})"
    )
    if out is not None:
        out.write_text(report.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v && uv run pytest -q && uv run ruff check src tests`
Expected: 3 passed; full suite 80 passed; ruff clean. Also sanity: `uv run juggletrack --help` lists both commands.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/juggletrack/cli.py tests/test_cli.py
git commit -m "feat: juggletrack CLI (analyze, eval)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Real-video validation run (exploratory — NOT TDD)

**Files:**
- Create: `docs/superpowers/plans/2026-07-16-plan2-validation-findings.md` (committed)
- Output (NOT committed; `outputs/` is gitignored): `/Users/andrew.follmann/personal-projects/juggling/outputs/plan2-validation/<video-stem>/…`

**Interfaces:**
- Consumes: the finished CLI.

This task runs the pipeline against reality for the first time and documents what happens. There is no pass/fail test; the deliverable is an honest findings report. Expectation-setting: stock COCO YOLO at conf 0.05 may detect juggling balls poorly — that is *the datum this task exists to measure* (it motivates Plan 3's fine-tuning). Do not tune the event core to make numbers look better.

- [ ] **Step 1: Confirm weights and run the five videos**

Weights: use `/Users/andrew.follmann/personal-projects/juggling/yolo11n.pt` if present, else the name `yolo11n.pt` (auto-downloads). For each video in `/Users/andrew.follmann/personal-projects/juggling/data/raw/` (`af1.mov`, `af2.mp4`, `PXL_20251226_195820315.mp4`, and the two `YTDown…480p.mp4` tutorials):

```bash
uv run juggletrack analyze /Users/andrew.follmann/personal-projects/juggling/data/raw/<video> \
  --out /Users/andrew.follmann/personal-projects/juggling/outputs/plan2-validation/<stem> \
  --model /Users/andrew.follmann/personal-projects/juggling/yolo11n.pt \
  --save-intermediates
```

Record wall-clock time per video (`time …`). If a video errors, record the error and continue with the rest.

- [ ] **Step 2: Compute detection-coverage stats per video**

For each `detections.jsonl`, compute with a short `uv run python -c` script: total detections; fraction of frames with ≥1, ≥2, ≥3 detections; median confidence; median normalized box size (`w`). From each `analysis.json`: number of arcs, runs (with catches/end_reason each), drops, hand_line_y.

- [ ] **Step 3: Try one sensitivity variation**

Re-run the single best-covered video with `--imgsz 960` (higher input resolution, same weights) using `--out …/<stem>-imgsz960`, and record how coverage changes. (This probes whether resolution or training data is the binding constraint — a key Plan 3 design input.)

- [ ] **Step 4: Write and commit the findings report**

`docs/superpowers/plans/2026-07-16-plan2-validation-findings.md` with sections: **Setup** (weights, conf, imgsz, machine); **Per-video table** (duration, wall time, detections, coverage ≥2 balls, arcs, runs, catches, drops); **Sensitivity** (imgsz 640 vs 960); **Qualitative failure modes** (from the numbers — e.g. "coverage 4% on af1.mov: stock model rarely sees the balls" or "coverage fine but arcs fragment"); **Implications for Plan 3** (what the data says about fine-tuning need, ROI cropping, resolution). Reference overlay paths for the user to eyeball. Be blunt; low numbers are the expected, useful result.

```bash
git add docs/superpowers/plans/2026-07-16-plan2-validation-findings.md
git commit -m "docs: plan 2 validation findings on real videos (stock YOLO baseline)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Plan Self-Review Notes (already applied)

- **Spec coverage:** build-order step 2 (offline pipeline with stock YOLO on own videos → `analysis.json` + debug overlay) = Tasks 2–6, 8, 9; step 3 (eval harness) = Task 7; spec §8 CLI `analyze`/`eval` = Task 8 (`label`/`train` are Plan 3, `live` is Plan 4); Plan 1 carry-forward rmse fix = Task 1. Ground-truth *labels* for the user's videos require a human watching them — the harness (Task 7) and the CLI (Task 8) are deliverable now; hand-labeling happens with the user after Task 9's overlays exist.
- **Type consistency:** `Detection` fields match Plan 1's `types.py` exactly; `analyze_video` passes `AnalyzeConfig | None` straight through to `analyze_detections`; `FakeDetector` is reused by the CLI's `--detections` path so tests and production share one code path; `SessionResult.meta` accepts `dict[str, float | int | str]` — all meta values written in Task 5 are str/int/float.
- **Known risk:** `tests/helpers.py` imports depend on `tests/` being importable as a package — pytest's rootdir conftest handling covers `from tests.helpers import …` only if `tests/__init__.py` exists or rootdir is on `sys.path`; if the import fails at execution time, add an empty `tests/__init__.py` (do not restructure the helper).
- **Codec risk:** mp4v via cv2 is bundled with opencv-python wheels on macOS; if `write_test_video` fails to open the writer, the helper raises immediately with a clear message rather than producing empty videos.



