# Juggletrack Plan 3: Data Flywheel + First Fine-Tune — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the semi-automated labeling flywheel once end-to-end — arc-verified auto-labels from real footage → COCO dataset → fine-tuned YOLO11n — and measure the detection-coverage jump against Plan 2's stock baseline (af2 56.6% / PXL 48.8% / af1 42.5% / yt-3ball 26.1% coverage ≥2 balls).

**Architecture:** The offline pipeline's physics layer becomes the labeler: detections that survive arc extraction are high-precision "ball" labels (false positives rarely fit gravity — spec §5). Two extractor-quality upgrades land first because label precision depends on them (x-residuals in EM assignment; gravity self-consistency pruning). New `data/` package handles label selection, COCO export, and dataset assembly with whole-video train/val splits; new `train/` package wraps ultralytics fine-tuning; the CLI gains `label`, `coverage`, and `train`. The final task runs the flywheel on real footage and documents before/after coverage honestly.

**Tech Stack:** Python 3.12, `uv`, existing juggletrack packages, ultralytics (fine-tune), pyyaml, cv2, yt-dlp (via `uvx`, best-effort external harvest).

**Plan sequence context:** Plan 3 of 5. Plans 1–2 (merged) built the event core and the offline pipeline. Plan 4 = realtime engine; Plan 5 = drop hardening. Carry-forwards from Plan 1's final review implemented here: x in EM assignment cost, gravity self-consistency re-prune, seed-sweep accuracy regression test (baseline measured 80% vs the 90% spec target — Task 1 exists to close that gap).

## Global Constraints

- Python 3.12; everything via `uv`. cv2 only in `video/`, `pipeline/`, and (new) `data/`; ultralytics imported lazily inside functions only (`detect/yolo.py`, `train/finetune.py`); the event core stays torch/cv2-free.
- Canonical label format is **COCO JSON** (spec: framework-neutral); YOLO-txt layout is derived, never authoritative. COCO category: `{"id": 1, "name": "ball"}`; YOLO class index 0.
- Train/val splits are **by source video**, never by frame (spec §6: hold out entire videos).
- Training artifacts, downloaded videos, datasets, and weights are NEVER committed: `runs/`, `models/`, `datasets/` added to `.gitignore` (Task 6); external downloads land in the MAIN checkout under `/Users/andrew.follmann/personal-projects/juggling/data/raw/external/` with attribution manifests.
- Existing 86 tests (+1 deselected) stay green; never loosen an existing test. New model-touching tests are marked `@pytest.mark.detector` (deselected by default).
- All randomness seeded; non-`detector` tests deterministic and network-free.
- Conventional commits ending with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Real assets (main checkout, absolute paths): videos `/Users/andrew.follmann/personal-projects/juggling/data/raw/`, stock weights `/Users/andrew.follmann/personal-projects/juggling/yolo11n.pt`, Plan 2 saved detections under `/Users/andrew.follmann/personal-projects/juggling/outputs/plan2-validation/<stem>/detections.jsonl` (af1, af2, pxl, yt-3ball have them; yt-5easy and af2-imgsz960 do not).
- `ROBOFLOW_API_KEY` is NOT set in this environment: external Roboflow seed data is out of scope for this plan (documented in Task 7's findings as future work; do not block on it).

## File Structure

```
src/juggletrack/arcs/fit.py          # Task 1: + x_residuals
src/juggletrack/arcs/extract.py      # Task 1: EM x-cost, _gravity_prune; Task 2: assign_detections
src/juggletrack/analyze.py           # Task 1: (no change; seed-sweep test exercises it)
src/juggletrack/data/__init__.py     # Task 3: empty (new package)
src/juggletrack/data/autolabel.py    # Task 3: select_autolabels, export_video_labels (COCO)
src/juggletrack/data/dataset.py      # Task 4: assemble_dataset (COCO dirs → YOLO layout + data.yaml)
src/juggletrack/train/__init__.py    # Task 6: empty
src/juggletrack/train/finetune.py    # Task 6: train_detector (lazy ultralytics)
src/juggletrack/cli.py               # Task 5: label + coverage; Task 6: train
.gitignore                           # Task 6: runs/, models/, datasets/
tests/test_extract.py                # Task 1 (extend): sweep + gravity tests
tests/test_assign.py                 # Task 2
tests/test_autolabel.py              # Task 3
tests/test_dataset.py                # Task 4
tests/test_cli.py                    # Task 5 (extend)
tests/test_train.py                  # Task 6 (stubbed unit + marked integration)
docs/superpowers/plans/2026-07-16-plan3-flywheel-findings.md  # Task 7 output
```

---

### Task 1: Extractor label-precision hardening (x-residuals in EM, gravity self-consistency, seed-sweep gate)

**Files:**
- Modify: `src/juggletrack/arcs/fit.py` (add `x_residuals`), `src/juggletrack/arcs/extract.py` (`_em_assign_refit` cost, new `_gravity_prune`, wire into loop)
- Test: `tests/test_extract.py` (extend)

**Interfaces:**
- Consumes: existing `y_residuals(arc, arr)`, `Arc`, extraction internals.
- Produces (consumed by Task 2 and everything downstream):
  - `x_residuals(arc: Arc, arr: np.ndarray) -> np.ndarray` in `fit.py` — absolute x residuals against the arc's linear x model, same shape contract as `y_residuals`.
  - EM assignment residual becomes `max(y_residual, x_residual)` (a point must fit BOTH models); threshold unchanged (`< 2*resid_tol`).
  - `_gravity_prune(arcs: list[Arc], band: float = 0.3) -> list[Arc]` — with ≥4 arcs, drop arcs whose `ay` falls outside `[0.7*median_ay, 1.3*median_ay]`; fewer than 4 arcs → unchanged. Runs each EM iteration after `_prune`.

**Why:** Plan 1's final review demonstrated (a) cross-ball chimera arcs fitting `ay≈2.2` vs true `1.0` survive the wide static `g_range=(0.5, 8.0)` prune and mint phantom catches, and (b) EM assignment ignoring x steals crossing-ball points. Measured effect: only 16/20 seeds within ±1 catch at noise=0.004/dropout=0.15 vs the spec §1 target of ≥90%. Both fixes attack root causes; the seed-sweep test locks the target in.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_extract.py`:

```python
def test_x_residuals_flag_cross_ball_points():
    from juggletrack.arcs.fit import fit_arc, x_residuals

    r = simulate_cascade(n_throws=1, fps=60.0, seed=0)
    arr = points_array(r.detections)
    arc = fit_arc(arr)
    # a point on the arc's y-parabola but at a wrong x (another ball's position)
    t_mid = (arc.t_start + arc.t_end) / 2
    impostor = np.array([[t_mid, arc.x_at(t_mid) + 0.2, arc.y_at(t_mid), 1.0]])
    res = x_residuals(arc, impostor)
    assert res[0] == pytest.approx(0.2, abs=1e-6)
    assert x_residuals(arc, arr).max() < 0.01  # true points fit x tightly


def test_gravity_prune_kills_chimera_curvature():
    from juggletrack.arcs.extract import _gravity_prune
    from juggletrack.types import Arc

    def arc_with_ay(i, ay):
        return Arc(id=i, t_start=float(i), t_end=float(i) + 1.0, ay=ay, by=-1.1,
                   cy=0.65, bx=0.1, cx=0.4, n_points=20, rmse=0.005)

    arcs = [arc_with_ay(i, 1.0 + 0.03 * i) for i in range(5)] + [arc_with_ay(9, 2.2)]
    kept = _gravity_prune(arcs)
    assert {a.id for a in kept} == {0, 1, 2, 3, 4}
    # fewer than 4 arcs: untouched even with an outlier
    few = [arc_with_ay(0, 1.0), arc_with_ay(1, 2.2)]
    assert _gravity_prune(few) == few


def test_catch_accuracy_seed_sweep():
    """Spec §1 target: catch count within ±1 on >=90% of runs.

    Plan 1's final review measured 16/20 at this noise regime; the EM x-cost
    and gravity pruning exist to close the gap. This is the regression gate.
    """
    from juggletrack.analyze import analyze_detections

    ok = 0
    failures = []
    for seed in range(20):
        r = simulate_cascade(n_throws=12, fps=30.0, noise=0.004, dropout=0.15, seed=seed)
        sr = analyze_detections(r.detections)
        total = sum(run.catches for run in sr.runs)
        if abs(total - 12) <= 1:
            ok += 1
        else:
            failures.append((seed, total, len(sr.runs)))
    assert ok >= 18, f"catch accuracy {ok}/20 below 90% target; failures: {failures}"
```

- [ ] **Step 2: Run tests to verify failure status**

Run: `uv run pytest tests/test_extract.py -v -k "x_residuals or gravity or sweep"`
Expected: `test_x_residuals...` FAILS with ImportError (`x_residuals` doesn't exist); `test_gravity_prune...` FAILS with ImportError; `test_catch_accuracy_seed_sweep` FAILS the `>= 18` assertion (baseline ~16/20) — record the exact pre-fix count in your report.

- [ ] **Step 3: Implement**

In `src/juggletrack/arcs/fit.py`, after `y_residuals`:

```python
def x_residuals(arc: Arc, arr: np.ndarray) -> np.ndarray:
    dt = arr[:, 0] - arc.t_start
    return np.abs(arc.bx * dt + arc.cx - arr[:, 1])
```

In `src/juggletrack/arcs/extract.py`:

1. Import `x_residuals` alongside `y_residuals`.
2. In `_em_assign_refit`, replace the residual line:

```python
        res_both = np.maximum(y_residuals(arc, arr), x_residuals(arc, arr))
        res = np.where(in_span, res_both, np.inf)
```

3. Add after `_prune`:

```python
def _gravity_prune(arcs: list[Arc], band: float = 0.3) -> list[Arc]:
    """Self-consistency: all real arcs share one gravity, so curvature outliers
    (cross-ball chimeras fit ay far above the cohort) are spurious. Needs a
    quorum of >=4 arcs so the median is trustworthy."""
    if len(arcs) < 4:
        return arcs
    med = float(np.median([a.ay for a in arcs]))
    lo, hi = (1.0 - band) * med, (1.0 + band) * med
    return [a for a in arcs if lo <= a.ay <= hi]
```

4. In `extract_arcs`'s EM loop, after the `_prune` line:

```python
        arcs = _gravity_prune(arcs)
```

- [ ] **Step 4: Run the full suite and record the sweep number**

Run: `uv run pytest tests/test_extract.py -v && uv run pytest -q`
Expected: new tests pass, sweep reaches ≥18/20; full suite 89 passed, 1 deselected — existing tests must not regress. If an existing extraction/analyze test breaks, the new cost/prune is too aggressive for that scenario — root-cause before touching anything (likely suspects: `_gravity_prune` band too tight for short-arc cohorts, or x-residual threshold clipping legitimate points at flight edges). If the sweep lands at 17/20 after faithful implementation, diagnose each failing seed (which stage: extraction arc count? run segmentation? drop misfire?) and report DONE_WITH_CONCERNS with the per-seed table — do NOT weaken the assertion.

- [ ] **Step 5: Commit**

```bash
git add src/juggletrack/arcs/ tests/test_extract.py
git commit -m "feat: x-consistent EM assignment and gravity self-consistency pruning

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Detection→arc assignment API (`assign_detections`)

**Files:**
- Modify: `src/juggletrack/arcs/extract.py` (add public function)
- Test: `tests/test_assign.py`

**Interfaces:**
- Consumes: `Arc`, `Detection`.
- Produces (consumed by Task 3's label selection):
  - `assign_detections(dets: list[Detection], arcs: list[Arc], *, resid_tol: float = 0.02, time_margin: float = 0.02) -> list[int]` — for each detection IN INPUT ORDER, the id of the best-fitting arc whose time span (±margin) covers it and whose `max(y_res, x_res) < 2*resid_tol`; else `-1`.

- [ ] **Step 1: Write the failing tests**

`tests/test_assign.py`:

```python
import pytest

from juggletrack.arcs.extract import assign_detections, extract_arcs
from juggletrack.sim import simulate_cascade
from juggletrack.types import Detection


def test_flight_detections_get_assigned():
    r = simulate_cascade(n_throws=10, fps=30.0, noise=0.003, seed=1)
    arcs = extract_arcs(r.detections)
    labels = assign_detections(r.detections, arcs)
    assert len(labels) == len(r.detections)
    frac_assigned = sum(1 for a in labels if a != -1) / len(labels)
    assert frac_assigned >= 0.9
    arc_ids = {a.id for a in arcs}
    assert all(a in arc_ids for a in labels if a != -1)


def test_far_points_get_minus_one():
    r = simulate_cascade(n_throws=10, fps=30.0, seed=1)
    arcs = extract_arcs(r.detections)
    t_mid = (r.run_start + r.run_end) / 2
    junk = [Detection(frame_idx=999, t=t_mid, x=0.5, y=0.02),   # far above any apex
            Detection(frame_idx=999, t=t_mid, x=0.98, y=0.65)]  # far right of pattern
    labels = assign_detections(junk, arcs)
    assert labels == [-1, -1]


def test_input_order_preserved():
    r = simulate_cascade(n_throws=6, fps=30.0, seed=2)
    arcs = extract_arcs(r.detections)
    fwd = assign_detections(r.detections, arcs)
    rev = assign_detections(list(reversed(r.detections)), arcs)
    assert rev == list(reversed(fwd))


def test_empty_inputs():
    assert assign_detections([], []) == []
    r = simulate_cascade(n_throws=6, fps=30.0, seed=2)
    assert assign_detections(r.detections, []) == [-1] * len(r.detections)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_assign.py -v`
Expected: FAIL with `ImportError: cannot import name 'assign_detections'`.

- [ ] **Step 3: Implement** — append to `src/juggletrack/arcs/extract.py`:

```python
def assign_detections(
    dets: list[Detection],
    arcs: list[Arc],
    *,
    resid_tol: float = 0.02,
    time_margin: float = 0.02,
) -> list[int]:
    """Best-fitting arc id per detection (input order), or -1 if none fits.

    Same acceptance rule as EM assignment (max of y/x residuals under
    2*resid_tol), so 'assigned' means 'would have survived extraction'.
    This is the auto-labeler's precision gate (spec §5).
    """
    out: list[int] = []
    for d in dets:
        best_id, best_res = -1, 2.0 * resid_tol
        for arc in arcs:
            if not (arc.t_start - time_margin <= d.t <= arc.t_end + time_margin):
                continue
            dt = d.t - arc.t_start
            ry = abs(arc.ay * dt * dt + arc.by * dt + arc.cy - d.y)
            rx = abs(arc.bx * dt + arc.cx - d.x)
            res = max(ry, rx)
            if res < best_res:
                best_id, best_res = arc.id, res
        out.append(best_id)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_assign.py -v && uv run pytest -q`
Expected: 4 passed; full suite 93 passed, 1 deselected.

- [ ] **Step 5: Commit**

```bash
git add src/juggletrack/arcs/extract.py tests/test_assign.py
git commit -m "feat: public detection-to-arc assignment API for auto-labeling

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Auto-label selection + COCO export (`data/autolabel.py`)

**Files:**
- Create: `src/juggletrack/data/__init__.py` (empty), `src/juggletrack/data/autolabel.py`
- Test: `tests/test_autolabel.py`

**Interfaces:**
- Consumes: `assign_detections` (Task 2), `extract_arcs`, `Detection`, `Arc`, `VideoReader`.
- Produces (consumed by Tasks 4–5, 7):
  - `select_autolabels(dets: list[Detection], arcs: list[Arc], *, resid_tol=0.02, default_box=0.04) -> tuple[list[Detection], list[int]]` — (arc-verified detections with `w/h` defaulted to `default_box` when 0, list of frame_idx values that contain UNverified detections → the human-review queue).
  - `export_video_labels(video_path, labels: list[Detection], out_dir, *, review_frames: list[int] | None = None, jpeg_quality: int = 90) -> dict` — writes `out_dir/images/frame_NNNNNN.jpg` for every frame with ≥1 label, `out_dir/annotations.json` (COCO: `images`, `annotations` with PIXEL `[x_min, y_min, w, h]` bboxes, `categories=[{"id": 1, "name": "ball"}]`), `out_dir/review_manifest.json` (`{"video": ..., "review_frames": [...]}`); returns stats dict `{"n_images": int, "n_boxes": int, "n_review_frames": int}`.

**COCO details the implementer must honor:** image ids sequential from 1 in frame order; annotation ids sequential from 1; `file_name` is the bare `frame_%06d.jpg` (COCO convention: relative to the images dir); bbox clamped to image bounds; every annotation has `"iscrowd": 0` and `"area": w*h` (pixels); `images[i]` carries `width`/`height`.

- [ ] **Step 1: Write the failing tests**

`tests/test_autolabel.py`:

```python
import json

import pytest

from juggletrack.arcs.extract import extract_arcs
from juggletrack.data.autolabel import export_video_labels, select_autolabels
from juggletrack.sim import simulate_cascade
from juggletrack.types import Detection
from tests.helpers import write_test_video


@pytest.fixture()
def sim_with_junk():
    r = simulate_cascade(n_throws=8, fps=30.0, noise=0.002, seed=3)
    junk = [Detection(frame_idx=10, t=10 / 30.0, x=0.5, y=0.02),
            Detection(frame_idx=40, t=40 / 30.0, x=0.97, y=0.6)]
    return r, r.detections + junk


def test_select_autolabels_filters_unverified(sim_with_junk):
    r, dets = sim_with_junk
    arcs = extract_arcs(dets)
    labels, review = select_autolabels(dets, arcs)
    assert 0 < len(labels) < len(dets)
    assert all(lb.w == pytest.approx(0.04) and lb.h == pytest.approx(0.04) for lb in labels)
    # the junk frames must land in the review queue
    assert 10 in review and 40 in review


def test_export_writes_coco_and_frames(tmp_path, sim_with_junk):
    r, dets = sim_with_junk
    arcs = extract_arcs(dets)
    labels, review = select_autolabels(dets, arcs)
    n_frames = max(d.frame_idx for d in dets) + 1
    video = tmp_path / "v.mp4"
    write_test_video(video, n_frames=n_frames, fps=30.0, size=(320, 240))

    out = tmp_path / "labels_out"
    stats = export_video_labels(video, labels, out, review_frames=review)

    coco = json.loads((out / "annotations.json").read_text())
    assert coco["categories"] == [{"id": 1, "name": "ball"}]
    assert stats["n_images"] == len(coco["images"])
    assert stats["n_boxes"] == len(coco["annotations"]) == len(labels)
    labeled_frames = {d.frame_idx for d in labels}
    assert stats["n_images"] == len(labeled_frames)
    # every referenced image file exists with correct size metadata
    for im in coco["images"]:
        assert (out / "images" / im["file_name"]).exists()
        assert (im["width"], im["height"]) == (320, 240)
    # bbox sanity: pixel coords inside the image, area consistent
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        assert 0 <= x <= 320 and 0 <= y <= 240 and w > 0 and h > 0
        assert x + w <= 320 + 1e-6 and y + h <= 240 + 1e-6
        assert ann["area"] == pytest.approx(w * h)
        assert ann["iscrowd"] == 0 and ann["category_id"] == 1
    manifest = json.loads((out / "review_manifest.json").read_text())
    assert manifest["review_frames"] == sorted(set(review))


def test_export_bbox_matches_normalized_center(tmp_path):
    lb = Detection(frame_idx=0, t=0.0, x=0.5, y=0.5, w=0.1, h=0.2)
    video = tmp_path / "v.mp4"
    write_test_video(video, n_frames=1, fps=30.0, size=(320, 240))
    out = tmp_path / "one"
    export_video_labels(video, [lb], out)
    coco = json.loads((out / "annotations.json").read_text())
    assert coco["annotations"][0]["bbox"] == pytest.approx([144.0, 96.0, 32.0, 48.0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_autolabel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'juggletrack.data'`.

- [ ] **Step 3: Implement `src/juggletrack/data/autolabel.py`** (and empty `src/juggletrack/data/__init__.py`)

```python
"""Arc-verified auto-labeling: physics-surviving detections become COCO labels.

Precision comes from the parabola gate (spec §5): a detection only becomes a
label if it lies on an extracted arc. Frames holding detections that did NOT
verify go to the review manifest for a human pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from juggletrack.arcs.extract import assign_detections
from juggletrack.types import Arc, Detection

BALL_CATEGORY = {"id": 1, "name": "ball"}


def select_autolabels(
    dets: list[Detection],
    arcs: list[Arc],
    *,
    resid_tol: float = 0.02,
    default_box: float = 0.04,
) -> tuple[list[Detection], list[int]]:
    assignment = assign_detections(dets, arcs, resid_tol=resid_tol)
    labels: list[Detection] = []
    review_frames: set[int] = set()
    for d, arc_id in zip(dets, assignment):
        if arc_id == -1:
            review_frames.add(d.frame_idx)
            continue
        labels.append(d.model_copy(update={
            "w": d.w if d.w > 0 else default_box,
            "h": d.h if d.h > 0 else default_box,
        }))
    return labels, sorted(review_frames)


def export_video_labels(
    video_path: str | Path,
    labels: list[Detection],
    out_dir: str | Path,
    *,
    review_frames: list[int] | None = None,
    jpeg_quality: int = 90,
) -> dict:
    from juggletrack.video.reader import VideoReader

    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)

    by_frame: dict[int, list[Detection]] = {}
    for lb in labels:
        by_frame.setdefault(lb.frame_idx, []).append(lb)

    images: list[dict] = []
    annotations: list[dict] = []
    with VideoReader(video_path) as reader:
        w_px, h_px = reader.info.width, reader.info.height
        for idx, _t, frame in reader.frames():
            frame_labels = by_frame.get(idx)
            if not frame_labels:
                continue
            file_name = f"frame_{idx:06d}.jpg"
            cv2.imwrite(
                str(out / "images" / file_name), frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )
            image_id = len(images) + 1
            images.append({"id": image_id, "file_name": file_name,
                           "width": w_px, "height": h_px})
            for lb in frame_labels:
                bw, bh = lb.w * w_px, lb.h * h_px
                x_min = min(max(lb.x * w_px - bw / 2, 0.0), w_px - 1.0)
                y_min = min(max(lb.y * h_px - bh / 2, 0.0), h_px - 1.0)
                bw = min(bw, w_px - x_min)
                bh = min(bh, h_px - y_min)
                annotations.append({
                    "id": len(annotations) + 1, "image_id": image_id,
                    "category_id": BALL_CATEGORY["id"],
                    "bbox": [x_min, y_min, bw, bh],
                    "area": bw * bh, "iscrowd": 0,
                })

    (out / "annotations.json").write_text(json.dumps({
        "images": images, "annotations": annotations,
        "categories": [BALL_CATEGORY],
    }, indent=2))
    (out / "review_manifest.json").write_text(json.dumps({
        "video": str(video_path),
        "review_frames": sorted(set(review_frames or [])),
    }, indent=2))
    return {"n_images": len(images), "n_boxes": len(annotations),
            "n_review_frames": len(set(review_frames or []))}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_autolabel.py -v && uv run pytest -q`
Expected: 3 passed; full suite 96 passed, 1 deselected.

- [ ] **Step 5: Commit**

```bash
git add src/juggletrack/data/ tests/test_autolabel.py
git commit -m "feat: arc-verified auto-labeling with COCO export and review manifest

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Dataset assembly (`data/dataset.py`) — COCO sources → ultralytics layout with whole-video splits

**Files:**
- Modify: `pyproject.toml` (add `pyyaml>=6.0`)
- Create: `src/juggletrack/data/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: COCO dirs produced by Task 3 (`images/` + `annotations.json`).
- Produces (consumed by Tasks 6–7):
  - `assemble_dataset(source_dirs: list[Path], out_dir: Path, *, val_fraction: float = 0.2, seed: int = 0) -> dict` — copies images into `out_dir/images/{train,val}/<source-name>_<file>`, writes YOLO labels (`class 0, cx cy w h` normalized, one line per box) into `out_dir/labels/{train,val}/…`, writes `out_dir/data.yaml`; **split is by source dir** (≥1 val source when ≥2 sources; 0 val sources when only 1 — data.yaml's `val:` then points at train as a degenerate fallback, flagged in the returned stats). Returns `{"train_sources": [...], "val_sources": [...], "n_train_images": int, "n_val_images": int, "n_train_boxes": int, "n_val_boxes": int}`.

- [ ] **Step 1: Add the dependency**

Run: `uv add "pyyaml>=6.0"`

- [ ] **Step 2: Write the failing tests**

`tests/test_dataset.py`:

```python
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from juggletrack.data.dataset import assemble_dataset


def make_coco_source(root: Path, name: str, n_images: int, boxes_per_image: int = 1):
    """Minimal valid COCO source dir with black 64x48 jpgs."""
    import cv2

    src = root / name
    (src / "images").mkdir(parents=True)
    images, annotations = [], []
    for i in range(n_images):
        fn = f"frame_{i:06d}.jpg"
        cv2.imwrite(str(src / "images" / fn), np.zeros((48, 64, 3), dtype=np.uint8))
        images.append({"id": i + 1, "file_name": fn, "width": 64, "height": 48})
        for b in range(boxes_per_image):
            annotations.append({
                "id": len(annotations) + 1, "image_id": i + 1, "category_id": 1,
                "bbox": [16.0 + b, 12.0, 8.0, 6.0], "area": 48.0, "iscrowd": 0,
            })
    (src / "annotations.json").write_text(json.dumps({
        "images": images, "annotations": annotations,
        "categories": [{"id": 1, "name": "ball"}],
    }))
    return src


def test_assemble_splits_by_source(tmp_path):
    sources = [make_coco_source(tmp_path, f"vid{i}", n_images=3) for i in range(4)]
    out = tmp_path / "ds"
    stats = assemble_dataset(sources, out, val_fraction=0.25, seed=0)
    assert len(stats["val_sources"]) == 1
    assert len(stats["train_sources"]) == 3
    assert stats["n_train_images"] == 9 and stats["n_val_images"] == 3
    # no source appears in both splits
    assert not set(stats["train_sources"]) & set(stats["val_sources"])
    # every train image has a matching label file
    for img in (out / "images" / "train").iterdir():
        assert (out / "labels" / "train" / (img.stem + ".txt")).exists()


def test_yolo_label_contents(tmp_path):
    src = make_coco_source(tmp_path, "only", n_images=1)
    out = tmp_path / "ds"
    assemble_dataset([src], out, seed=0)
    txts = list((out / "labels" / "train").glob("*.txt"))
    assert len(txts) == 1
    parts = txts[0].read_text().split()
    # bbox [16, 12, 8, 6] in 64x48 -> cx=20/64, cy=15/48, w=8/64, h=6/48
    assert parts[0] == "0"
    assert [float(p) for p in parts[1:]] == pytest.approx([0.3125, 0.3125, 0.125, 0.125])


def test_data_yaml(tmp_path):
    sources = [make_coco_source(tmp_path, f"v{i}", 2) for i in range(2)]
    out = tmp_path / "ds"
    assemble_dataset(sources, out, val_fraction=0.5, seed=1)
    cfg = yaml.safe_load((out / "data.yaml").read_text())
    assert cfg["names"] == {0: "ball"}
    assert cfg["path"] == str(out.resolve())
    assert cfg["train"] == "images/train" and cfg["val"] == "images/val"


def test_single_source_degenerate_val(tmp_path):
    src = make_coco_source(tmp_path, "solo", 2)
    out = tmp_path / "ds"
    stats = assemble_dataset([src], out, seed=0)
    assert stats["val_sources"] == []
    cfg = yaml.safe_load((out / "data.yaml").read_text())
    assert cfg["val"] == "images/train"  # degenerate fallback, flagged in stats


def test_deterministic_split(tmp_path):
    sources = [make_coco_source(tmp_path, f"d{i}", 1) for i in range(5)]
    a = assemble_dataset(sources, tmp_path / "dsA", val_fraction=0.4, seed=7)
    b = assemble_dataset(sources, tmp_path / "dsB", val_fraction=0.4, seed=7)
    assert a["val_sources"] == b["val_sources"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'juggletrack.data.dataset'`.

- [ ] **Step 4: Implement `src/juggletrack/data/dataset.py`**

```python
"""Assemble ultralytics-ready datasets from COCO source dirs.

COCO stays canonical (spec: framework-neutral); this derives the YOLO layout.
Split is BY SOURCE VIDEO — never by frame (spec §6).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import yaml


def assemble_dataset(
    source_dirs: list[Path],
    out_dir: Path,
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> dict:
    sources = [Path(s) for s in source_dirs]
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(sources)))
    n_val = max(1, round(val_fraction * len(sources))) if len(sources) >= 2 else 0
    val_idx = set(order[:n_val])

    out = Path(out_dir)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train_sources": [], "val_sources": [],
             "n_train_images": 0, "n_val_images": 0,
             "n_train_boxes": 0, "n_val_boxes": 0}

    for i, src in enumerate(sources):
        split = "val" if i in val_idx else "train"
        stats[f"{split}_sources"].append(src.name)
        coco = json.loads((src / "annotations.json").read_text())
        anns_by_image: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)
        for im in coco["images"]:
            stem = f"{src.name}_{Path(im['file_name']).stem}"
            shutil.copyfile(src / "images" / im["file_name"],
                            out / "images" / split / f"{stem}.jpg")
            lines = []
            for ann in anns_by_image.get(im["id"], []):
                x, y, w, h = ann["bbox"]
                cx, cy = (x + w / 2) / im["width"], (y + h / 2) / im["height"]
                lines.append(
                    f"0 {cx:.6f} {cy:.6f} {w / im['width']:.6f} {h / im['height']:.6f}"
                )
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            stats[f"n_{split}_images"] += 1
            stats[f"n_{split}_boxes"] += len(lines)

    (out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val" if stats["n_val_images"] else "images/train",
        "names": {0: "ball"},
    }, sort_keys=False))
    return stats
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_dataset.py -v && uv run pytest -q`
Expected: 5 passed; full suite 101 passed, 1 deselected. (If `test_yolo_label_contents` disagrees on cx: bbox x=16, w=8 → cx=(16+4)/64=0.3125 — the test values are correct; fix the code, not the test.)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/juggletrack/data/dataset.py tests/test_dataset.py
git commit -m "feat: COCO-to-YOLO dataset assembly with whole-video splits

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: CLI `label` + `coverage` commands

**Files:**
- Modify: `src/juggletrack/cli.py`
- Test: `tests/test_cli.py` (extend)

**Interfaces:**
- Consumes: Tasks 2–3 outputs, existing pipeline/CLI plumbing.
- Produces (used heavily by Task 7):
  - `juggletrack label VIDEO [--out DIR] [--detections JSONL] [--model PATH] [--conf 0.05] [--imgsz 640] [--stride 1] [--device STR]` — detect (or replay saved jsonl), extract arcs, select arc-verified labels, export COCO to `--out` (default `outputs/labels/<stem>`); prints `N images, M boxes, K review frames -> DIR`.
  - `juggletrack coverage VIDEO [--detections JSONL] [--model ...] [--conf ...] [--imgsz ...] [--stride 1] [--device ...]` — prints total detections, coverage ≥1/≥2/≥3 over `ceil(frame_count/stride)` sampled frames, median confidence, median normalized box width. With `--detections`, pass the SAME `--stride` the jsonl was produced with (it sets the denominator).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
def test_label_command_with_saved_detections(workspace):
    from juggletrack.cli import app

    sim, video, dets, tmp = workspace
    out = tmp / "labels_out"
    result = runner.invoke(app, [
        "label", str(video), "--out", str(out), "--detections", str(dets),
    ])
    assert result.exit_code == 0, result.output
    assert (out / "annotations.json").exists()
    assert (out / "review_manifest.json").exists()
    assert any((out / "images").iterdir())
    assert "boxes" in result.output and "review" in result.output


def test_coverage_command(workspace):
    from juggletrack.cli import app

    sim, video, dets, tmp = workspace
    result = runner.invoke(app, [
        "coverage", str(video), "--detections", str(dets),
    ])
    assert result.exit_code == 0, result.output
    assert f"detections {len(sim.detections)}" in result.output
    assert ">=2:" in result.output and ">=3:" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v -k "label or coverage"`
Expected: FAIL — typer exits code 2 (no such command).

- [ ] **Step 3: Implement** — add to `src/juggletrack/cli.py`:

```python
def _get_detections(video, detections, model, conf, imgsz, stride, device):
    from juggletrack.pipeline.offline import detect_video, load_detections_jsonl
    from juggletrack.video.reader import VideoReader

    if detections is not None:
        return load_detections_jsonl(detections)
    from juggletrack.detect.yolo import YOLODetector

    det = YOLODetector(model_path=model, conf=conf, imgsz=imgsz, device=device)
    with VideoReader(video) as reader:
        return detect_video(reader, det, stride=stride)


@app.command()
def label(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path | None = typer.Option(None, help="Output dir (default outputs/labels/<stem>)"),
    detections: Path | None = typer.Option(
        None, exists=True, dir_okay=False,
        help="Replay saved detections.jsonl instead of running the detector",
    ),
    model: str = typer.Option("yolo11n.pt"),
    conf: float = typer.Option(0.05),
    imgsz: int = typer.Option(640),
    stride: int = typer.Option(1, min=1),
    device: str | None = typer.Option(None),
) -> None:
    """Auto-label a video: arc-verified detections become COCO 'ball' boxes."""
    from juggletrack.arcs.extract import extract_arcs
    from juggletrack.data.autolabel import export_video_labels, select_autolabels

    out_dir = out or Path("outputs") / "labels" / video.stem
    dets = _get_detections(video, detections, model, conf, imgsz, stride, device)
    arcs = extract_arcs(dets)
    labels, review = select_autolabels(dets, arcs)
    stats = export_video_labels(video, labels, out_dir, review_frames=review)
    typer.echo(
        f"{stats['n_images']} images, {stats['n_boxes']} boxes, "
        f"{stats['n_review_frames']} review frames -> {out_dir}"
    )


@app.command()
def coverage(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    detections: Path | None = typer.Option(
        None, exists=True, dir_okay=False,
        help="Score saved detections.jsonl (pass the SAME --stride it was made with)",
    ),
    model: str = typer.Option("yolo11n.pt"),
    conf: float = typer.Option(0.05),
    imgsz: int = typer.Option(640),
    stride: int = typer.Option(1, min=1),
    device: str | None = typer.Option(None),
) -> None:
    """Detection-coverage stats: the before/after fine-tuning metric."""
    import math
    import statistics
    from collections import Counter

    from juggletrack.video.reader import VideoReader

    dets = _get_detections(video, detections, model, conf, imgsz, stride, device)
    with VideoReader(video) as reader:
        n_sampled = math.ceil(reader.info.frame_count / stride)
    per_frame = Counter(d.frame_idx for d in dets)
    cov = {
        k: sum(1 for v in per_frame.values() if v >= k) / n_sampled if n_sampled else 0.0
        for k in (1, 2, 3)
    }
    typer.echo(f"detections {len(dets)} over {n_sampled} sampled frames")
    typer.echo(f"coverage >=1: {cov[1]:.1%}  >=2: {cov[2]:.1%}  >=3: {cov[3]:.1%}")
    if dets:
        typer.echo(
            f"median conf {statistics.median(d.confidence for d in dets):.3f}  "
            f"median w {statistics.median(d.w for d in dets):.4f}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v && uv run pytest -q && uv run ruff check src tests`
Expected: 5 CLI tests pass; full suite 103 passed, 1 deselected; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/juggletrack/cli.py tests/test_cli.py
git commit -m "feat: CLI label and coverage commands

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: `juggletrack train` (`train/finetune.py`)

**Files:**
- Modify: `src/juggletrack/cli.py`, `.gitignore`
- Create: `src/juggletrack/train/__init__.py` (empty), `src/juggletrack/train/finetune.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `data.yaml` from Task 4.
- Produces (used by Task 7):
  - `train_detector(data_yaml: str | Path, *, base_model="yolo11n.pt", epochs=40, imgsz=640, device=None, project="runs/finetune", name="juggletrack") -> Path` — runs ultralytics fine-tuning (lazy import), returns the `best.pt` path; raises `FileNotFoundError` if training completed without producing it.
  - CLI: `juggletrack train DATA_YAML [--model yolo11n.pt] [--epochs 40] [--imgsz 640] [--device STR] [--project runs/finetune] [--name juggletrack]`.

- [ ] **Step 1: Add gitignore entries**

Append to `.gitignore` (create the lines only if absent):

```
runs/
models/
datasets/
```

- [ ] **Step 2: Write the failing tests**

`tests/test_train.py`:

```python
import sys
import types
from pathlib import Path

import pytest


def install_fake_ultralytics(monkeypatch, tmp_path, *, create_best=True):
    calls = {}

    class FakeYOLO:
        def __init__(self, base_model):
            calls["base_model"] = base_model

        def train(self, **kw):
            calls["train_kwargs"] = kw
            save_dir = tmp_path / kw["project"] / kw["name"]
            if create_best:
                (save_dir / "weights").mkdir(parents=True, exist_ok=True)
                (save_dir / "weights" / "best.pt").write_bytes(b"fake-weights")
            else:
                save_dir.mkdir(parents=True, exist_ok=True)
            return types.SimpleNamespace(save_dir=str(save_dir))

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    return calls


def test_train_detector_plumbing(tmp_path, monkeypatch):
    calls = install_fake_ultralytics(monkeypatch, tmp_path)
    from juggletrack.train.finetune import train_detector

    best = train_detector(
        tmp_path / "data.yaml", base_model="yolo11n.pt", epochs=3, imgsz=320,
        project=str(tmp_path / "runs"), name="tst",
    )
    assert best.exists() and best.name == "best.pt"
    kw = calls["train_kwargs"]
    assert calls["base_model"] == "yolo11n.pt"
    assert kw["epochs"] == 3 and kw["imgsz"] == 320
    assert kw["data"].endswith("data.yaml")
    assert kw["exist_ok"] is True and kw["plots"] is False


def test_train_detector_missing_best_raises(tmp_path, monkeypatch):
    install_fake_ultralytics(monkeypatch, tmp_path, create_best=False)
    from juggletrack.train.finetune import train_detector

    with pytest.raises(FileNotFoundError):
        train_detector(tmp_path / "data.yaml", project=str(tmp_path / "runs"), name="x")


def test_cli_train_command(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    calls = install_fake_ultralytics(monkeypatch, tmp_path)
    from juggletrack.cli import app

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names:\n  0: ball\n")
    result = CliRunner().invoke(app, [
        "train", str(data_yaml), "--epochs", "2",
        "--project", str(tmp_path / "runs"), "--name", "clitest",
    ])
    assert result.exit_code == 0, result.output
    assert "best.pt" in result.output
    assert calls["train_kwargs"]["epochs"] == 2


@pytest.mark.detector
def test_train_one_epoch_real(tmp_path):
    """Real ultralytics smoke: 1 epoch on a tiny synthetic dataset."""
    from juggletrack.data.dataset import assemble_dataset
    from juggletrack.train.finetune import train_detector
    from tests.test_dataset import make_coco_source

    sources = [make_coco_source(tmp_path, f"s{i}", n_images=4) for i in range(2)]
    ds = tmp_path / "ds"
    assemble_dataset(sources, ds, val_fraction=0.5, seed=0)
    best = train_detector(ds / "data.yaml", epochs=1, imgsz=64,
                          project=str(tmp_path / "runs"), name="smoke")
    assert best.exists()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_train.py -v`
Expected: 3 collected (the marked one deselected), all FAIL with `ModuleNotFoundError: No module named 'juggletrack.train'`.

- [ ] **Step 4: Implement**

`src/juggletrack/train/finetune.py` (and empty `src/juggletrack/train/__init__.py`):

```python
"""Fine-tune the ball detector. ultralytics imported lazily (torch stays out
of module import); training data comes from data/dataset.py's data.yaml.
"""
from __future__ import annotations

from pathlib import Path


def train_detector(
    data_yaml: str | Path,
    *,
    base_model: str = "yolo11n.pt",
    epochs: int = 40,
    imgsz: int = 640,
    device: str | None = None,
    project: str | Path = "runs/finetune",
    name: str = "juggletrack",
) -> Path:
    from ultralytics import YOLO  # lazy

    model = YOLO(base_model)
    results = model.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, device=device,
        project=str(project), name=name, exist_ok=True, plots=False,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"training finished but best.pt missing: {best}")
    return best
```

Add to `src/juggletrack/cli.py`:

```python
@app.command()
def train(
    data_yaml: Path = typer.Argument(..., exists=True, dir_okay=False),
    model: str = typer.Option("yolo11n.pt", help="Base weights to fine-tune"),
    epochs: int = typer.Option(40, min=1),
    imgsz: int = typer.Option(640),
    device: str | None = typer.Option(None),
    project: Path = typer.Option(Path("runs/finetune")),
    name: str = typer.Option("juggletrack"),
) -> None:
    """Fine-tune the ball detector on an assembled dataset."""
    from juggletrack.train.finetune import train_detector

    best = train_detector(
        data_yaml, base_model=model, epochs=epochs, imgsz=imgsz,
        device=device, project=project, name=name,
    )
    typer.echo(f"best weights: {best}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_train.py -v && uv run pytest -q && uv run ruff check src tests`
Expected: 3 passed (1 deselected marked test); full suite 106 passed, 2 deselected; ruff clean. Optionally run the marked smoke once (`uv run pytest tests/test_train.py -m detector -v`) and report its wall time — do not block on it if the tiny-image training errors inside ultralytics internals; report the error instead.

- [ ] **Step 6: Commit**

```bash
git add .gitignore src/juggletrack/train/ src/juggletrack/cli.py tests/test_train.py
git commit -m "feat: detector fine-tuning wrapper and CLI train command

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: First flywheel turn (exploratory — NOT TDD)

**Files:**
- Create: `docs/superpowers/plans/2026-07-16-plan3-flywheel-findings.md` (committed)
- Output (NOT committed): labels under `/Users/andrew.follmann/personal-projects/juggling/outputs/plan3-flywheel/labels/<stem>/`, dataset under `/Users/andrew.follmann/personal-projects/juggling/datasets/flywheel-v1/`, weights copied to `/Users/andrew.follmann/personal-projects/juggling/models/juggletrack-ft-v1/best.pt`, external videos under `/Users/andrew.follmann/personal-projects/juggling/data/raw/external/`.

This task runs the flywheel once and documents the coverage jump honestly. Expectation-setting: training only on own-footage auto-labels risks environment overfitting (the vdrumsta lesson from research); af1.mov is therefore excluded from training entirely and used as the held-out test video — but note in the findings that af1 shares the room/balls with af2, so this measures on-domain improvement, not true generalization. Do not tune numbers to look better.

- [ ] **Step 1: Best-effort CC external harvest (time-box ~10 minutes)**

```bash
uvx yt-dlp "ytsearch30:3 ball juggling cascade" \
  --match-filters "license~='(?i)creative commons' & duration<240" \
  --max-downloads 5 -f "bv*[height<=720]+ba/b[height<=720]" \
  --write-info-json \
  -o "/Users/andrew.follmann/personal-projects/juggling/data/raw/external/%(id)s.%(ext)s"
```

Record how many videos passed the license filter (0 is a legitimate result — most YouTube videos are Standard License; note it and continue). Spot-check any downloads actually contain toss juggling by auto-labeling them (next step) — a video yielding ~0 arcs is not juggling; drop it from the corpus and say so.

- [ ] **Step 2: Auto-label the corpus**

For each own video, reuse Plan 2's saved detections where they exist (af1, af2, pxl, yt-3ball; NOTE yt-3ball's jsonl was made with `--stride 2`):

```bash
uv run juggletrack label /Users/andrew.follmann/personal-projects/juggling/data/raw/af2.mp4 \
  --detections /Users/andrew.follmann/personal-projects/juggling/outputs/plan2-validation/af2/detections.jsonl \
  --out /Users/andrew.follmann/personal-projects/juggling/outputs/plan3-flywheel/labels/af2
```

(likewise af1, pxl, yt-3ball). For yt-5easy (no saved jsonl) and each usable external: fresh `juggletrack label … --model /Users/andrew.follmann/personal-projects/juggling/yolo11n.pt --stride 2`. Record per-video: images, boxes, review frames. Sanity-gate: if a video yields < 50 boxes, exclude it from training and note why.

- [ ] **Step 3: Assemble the dataset (af1 held out entirely)**

Write a short `uv run python -c` (or heredoc script) calling `assemble_dataset` with ALL label dirs EXCEPT af1's, `out_dir=/Users/andrew.follmann/personal-projects/juggling/datasets/flywheel-v1`, `val_fraction=0.2, seed=0`. Record the returned stats (sources per split, image/box counts) for the findings table. af1's labels are never in this dataset — it is the held-out test video.

- [ ] **Step 4: Train**

```bash
cd /Users/andrew.follmann/personal-projects/juggling/.claude/worktrees/juggling-tracker-app-8661ea
uv run juggletrack train /Users/andrew.follmann/personal-projects/juggling/datasets/flywheel-v1/data.yaml \
  --model /Users/andrew.follmann/personal-projects/juggling/yolo11n.pt \
  --epochs 40 --imgsz 640 --device mps --project runs/finetune --name flywheel-v1
```

Run in the background and poll (training on a few thousand images @ M4 Pro MPS is expected in the 15–45 min range). Record: wall time, final epoch metrics ultralytics prints (mAP50 on the val split), dataset size. Copy the resulting `best.pt` to `/Users/andrew.follmann/personal-projects/juggling/models/juggletrack-ft-v1/best.pt`. If training crashes, capture the error, halve `--imgsz` or epochs only to *diagnose*, and report honestly — a failed first turn is a finding, not something to hide.

- [ ] **Step 5: Measure before/after**

For each of: **af1 (held out)**, af2, pxl, yt-3ball, one usable external (if any) — run `juggletrack coverage` twice with fresh detection (same `--conf 0.05 --imgsz 640 --stride 2` both times for wall-time symmetry):

```bash
uv run juggletrack coverage <video> --model /Users/andrew.follmann/personal-projects/juggling/yolo11n.pt --stride 2
uv run juggletrack coverage <video> --model /Users/andrew.follmann/personal-projects/juggling/models/juggletrack-ft-v1/best.pt --stride 2
```

Then re-run full analysis on af1 + af2 with the fine-tuned weights (`juggletrack analyze … --model …ft-v1/best.pt --save-intermediates --out /Users/andrew.follmann/personal-projects/juggling/outputs/plan3-flywheel/analyze-ft/<stem>`) and note run/catch/drop deltas vs Plan 2's numbers (af2: 1 run / 29 catches / 1 drop at 640 baseline — read the actual baseline from `/Users/andrew.follmann/personal-projects/juggling/outputs/plan2-validation/af2/analysis.json`).

- [ ] **Step 6: Write and commit the findings**

`docs/superpowers/plans/2026-07-16-plan3-flywheel-findings.md`, sections: **Corpus** (per-video label counts, review-frame counts, exclusions, external-harvest outcome); **Training** (dataset stats, config, wall time, val mAP50); **Coverage before/after** (table: video × stock/fine-tuned × ≥1/≥2/≥3, with af1 clearly marked HELD OUT); **Event-level deltas** (af1/af2 analyze results vs Plan 2 baseline); **Caveats** (on-domain vs generalization: af1 shares environment with af2; auto-label precision not human-audited — cite review-frame counts; stride-2 label sparsity for tutorials); **Next turn** (what data the second flywheel turn needs most: diverse environments — Kinetics/Roboflow with user's API key, Meschke email; review-UI pass on flagged frames). Reference overlay paths for the user to eyeball.

```bash
git add docs/superpowers/plans/2026-07-16-plan3-flywheel-findings.md
git commit -m "docs: plan 3 first flywheel turn — coverage before/after fine-tune

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Plan Self-Review Notes (already applied)

- **Spec coverage:** spec §5 semi-automated flywheel = Tasks 2–5 + 7 (auto-label → human-review manifest → retrain loop; the review UI itself is deferred to the next turn and named in Task 7's findings); §5 "COCO JSON canonical" = Tasks 3–4; §6 whole-video splits = Task 4; §8 `label`/`train` CLI = Tasks 5–6; spec §1 accuracy target promoted to a regression gate = Task 1; Plan 1 carry-forwards (x in EM, gravity self-consistency, seed sweep) = Task 1. Tier A external seed data (Roboflow/Meschke) is explicitly out of scope: no API key in the environment and Meschke requires a permission email — both named in Task 7's "Next turn" section rather than silently dropped.
- **Type consistency:** `x_residuals` (Task 1) is used by `assign_detections` inline math (Task 2 — deliberately re-derived per point, no array detour); `select_autolabels` consumes `assign_detections` with matching `resid_tol` defaults; `export_video_labels` output shape is exactly what `make_coco_source`/`assemble_dataset` (Task 4) consume; CLI `label`/`coverage`/`train` signatures match the functions they call; `Detection.model_copy` is valid pydantic v2.
- **Known risks, stated where they bite:** Task 1's 18/20 sweep gate may require iteration beyond the two named fixes (escalation path written into Step 4); Task 6's real-training smoke may hit ultralytics internals on 64px images (explicitly allowed to report-and-continue); Task 7's external harvest may legitimately yield zero CC videos (explicitly a valid outcome); training wall time is bounded by running in background with polling.
- **Placeholder scan:** clean — every code step carries complete code; Task 7 is exploratory by design with exact commands.



