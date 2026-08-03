# Juggletrack 05 — Validation Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the two in-scope accuracy defects exposed by the 22-video Meschke oracle validation (duplicate-box arc minting; run-span truncation at unwitnessed misses), fix the oracle CSV parser gap, and re-measure the full validation suite from saved detections.

**Architecture:** Two small, evidence-targeted changes to existing stages — a per-frame detection pre-clustering pure function applied at the top of `analyze_detections` (offline and realtime share it by construction, since `RealtimeAnalyzer` delegates to `analyze_detections`), and a semantics correction in `segment_runs` so run spans truncate only at *floor-bound* misses (real drops), not unwitnessed ones. No new subsystems.

**Tech stack:** existing (Python 3.12, pydantic models, numpy, pytest).

## Evidence base (measured 2026-08-03, `outputs/meschke-val/`)

22 three-ball vanilla-siteswap Meschke videos, per-ball GT oracles. Spec §6.1
(per-run |catch Δ| ≤1 on ≥90% of runs) measured at **27.8%**; §6.2 (run IoU ≥0.9
on ≥80%) at **22.2%**. Three distinct mechanisms:

1. **Duplicate-box arc minting (over-count)** — `ss3_id_086` frame 194: 25 boxes
   forming exactly 3 spatial clusters (the 3 balls), duplicates 0.01–0.03 apart on
   w≈0.065 boxes (below YOLO NMS IoU threshold). Each duplicate subset fits its own
   well-witnessed parallel arc: 76 arcs / 57 predicted catches vs oracle 24.
   Same signature: ss441_id_089 (+40 catches, 19 phantom drops), ss423_id_088 (+29),
   ss3_id_110 (+12, median 4 max 11 dets/frame). Under-counting videos have CLEAN
   density (ss3_id_016: median 3/frame) — clustering must not change them.
2. **Run-span truncation at unwitnessed misses** — `segment_runs` truncates `end_t`
   at the first uncaught arc but keeps counting the whole group's catches:
   ss42_id_011 reports one run −0.5..38.7s holding 93 catches whose member arcs span
   0..200s. Downstream consumers of the span (run IoU, realtime liveness via
   `r.end_t`, drop context) all inherit the error. ss3_id_016's over-segmentation
   (9 runs vs oracle 5) and part of its live churn share this root.
3. **High-pattern apex fragmentation (OUT OF SCOPE for this plan)** — ss50505_id_093:
   high "5" throws fragment at apex (detector loses the ball), ascent halves carry
   throws-without-catches (135 throws / 6 catches / 35 phantom drops). Affects
   505/531/51 patterns; v1 scope is the 3-ball cascade. Documented as a scope
   boundary in Task 4's findings doc; candidate future fix is extending split-stitch
   across out-of-frame gaps.

Also: 2 of 22 oracle CSVs (`ss531_id_988/989`) hold float coordinates and crash
`_read_csv_rows` (`int('200.888...')` ValueError) — oracle-side parser gap.

## Global Constraints

- Any count change on the 22-video suite must be justified against the oracle in the
  Task 4 findings doc — per-video, both directions (improvements and regressions).
- Videos already at |total catch Δ| ≤ 2 (ss441_id_013, ss42_id_010/011, ss3_id_079,
  ss3_id_987, ss50505_id_012) must stay within ±2 of oracle after every task.
- No existing test may be loosened without a measured justification in its docstring.
- Path-scoped commits; full suite + `ruff check src tests` green at every commit.
- All validation reruns use the SAVED `outputs/meschke-val/<stem>/detections.jsonl`
  (never re-detect — determinism and speed).

---

### Task 1: Meschke CSV float-coordinate parser fix

**Files:**
- Modify: `src/juggletrack/data/meschke_import.py` (`_read_csv_rows`, line 63)
- Test: `tests/test_meschke_import.py`

**Interfaces:** unchanged (`load_meschke_trajectories`, `oracle_events`).

- [ ] **Step 1: failing test** — copy an existing meschke CSV fixture pattern in
  `tests/test_meschke_import.py`; write a fixture whose data rows contain float
  strings (`"200.8888888888889"`), assert `load_meschke_trajectories` parses and
  rounds to nearest int pixel:

```python
def test_float_coordinate_rows_parse(tmp_path):
    csv = tmp_path / "gt.csv"
    csv.write_text(
        "frame,ball1_x,ball1_y\n"
        "0,200.8888888888889,101.2\n"
        "1,203.1111111111111,98.7\n"
    )
    # adapt column layout to the real fixture format used by existing tests
    trajs = load_meschke_trajectories(csv, width=480, height=848)
    assert trajs  # parses without ValueError; coordinates rounded to int
```

  (Match the real header/column convention from existing tests — the assertion that
  matters is: float strings parse, values equal `int(round(float(v)))`.)
- [ ] **Step 2: run** `uv run pytest tests/test_meschke_import.py -q` → FAIL (ValueError).
- [ ] **Step 3: fix** — in `_read_csv_rows`: `int(v)` → `int(round(float(v)))`.
- [ ] **Step 4: run** the file's tests → PASS; then verify against the real data:
  `oracle_events('data/raw/meschke/csv/ss531_id_988.csv', fps=..., width=..., height=...)`
  computes without error (probe fps/width/height via `VideoReader`; record counts in
  the task report for Task 4's table).
- [ ] **Step 5: commit** `fix: meschke CSV parser accepts float coordinates`.

### Task 2: per-frame duplicate-box clustering

**Files:**
- Create: `src/juggletrack/detect/cluster.py`
- Modify: `src/juggletrack/analyze.py` (`AnalyzeConfig` + `analyze_detections`)
- Test: `tests/test_cluster.py`, additions to `tests/test_realtime.py`

**Interfaces:**
- Produces: `cluster_detections(dets: list[Detection], *, merge_dist: float) -> list[Detection]`
  — pure, order-independent per frame, no cross-frame state.
- `AnalyzeConfig.cluster_merge_dist: float = 0.03` (0.0 disables; must validate ≥ 0).
- `analyze_detections` applies clustering FIRST (before `filter_static_detections`) so
  every downstream stage — offline and the realtime window path — sees clustered
  detections.

**Algorithm (greedy, confidence-first):** per frame: sort detections by confidence
descending; for each detection, if its center is within `merge_dist` (euclidean, in
normalized units) of an already-kept cluster's center, absorb it into that cluster,
else it starts a new cluster. Cluster output: confidence-weighted mean `x`/`y`/`w`/`h`,
`confidence = max` of members, `frame_idx`/`t` preserved.

**Default justification (record in docstring):** measured duplicate offsets are
0.01–0.03 on w≈0.065 boxes (ss3_id_086 frame 194); distinct cascade balls approach
closer than 0.03 only in brief crossing instants, where losing one detection point
per ball to a merge is absorbed by the EM assigner (verified by the sim test below).

- [ ] **Step 1: failing unit tests** in `tests/test_cluster.py`:

```python
from juggletrack.detect.cluster import cluster_detections
from juggletrack.types import Detection

def _d(x, y, conf, frame=0, t=0.0):
    return Detection(frame_idx=frame, t=t, x=x, y=y, w=0.06, h=0.06, confidence=conf)

def test_near_duplicates_merge_to_strongest_center():
    dets = [_d(0.360, 0.770, 0.30), _d(0.363, 0.755, 0.12), _d(0.357, 0.780, 0.08)]
    out = cluster_detections(dets, merge_dist=0.03)
    assert len(out) == 1
    assert out[0].confidence == 0.30
    assert abs(out[0].x - 0.360) < 0.01  # confidence-weighted, dominated by strongest

def test_distinct_balls_survive():
    dets = [_d(0.36, 0.77, 0.3), _d(0.42, 0.44, 0.2), _d(0.73, 0.53, 0.3)]
    assert len(cluster_detections(dets, merge_dist=0.03)) == 3

def test_frames_are_independent():
    dets = [_d(0.36, 0.77, 0.3, frame=0), _d(0.36, 0.77, 0.3, frame=1, t=0.033)]
    assert len(cluster_detections(dets, merge_dist=0.03)) == 2

def test_zero_merge_dist_is_identity():
    dets = [_d(0.360, 0.770, 0.3), _d(0.361, 0.770, 0.2)]
    assert cluster_detections(dets, merge_dist=0.0) == dets
```

- [ ] **Step 2:** run → FAIL (module missing). Implement `cluster.py` per the algorithm.
- [ ] **Step 3: failing integration test** (the load-bearing one) in `tests/test_cluster.py`:

```python
def test_duplicate_injection_does_not_inflate_catches():
    """Duplicate-box storms mint parallel arcs and inflate catches (measured:
    ss3_id_086 oracle 24 -> 57 pre-fix). Injecting 2-3 jittered clones of every
    sim detection must leave the catch count at the clean baseline."""
    import numpy as np
    from juggletrack.analyze import AnalyzeConfig, analyze_detections
    from juggletrack.sim import simulate_cascade
    sim = simulate_cascade(n_throws=12, fps=30.0, seed=3)
    rng = np.random.default_rng(7)
    clones = []
    for d in sim.detections:
        for _ in range(int(rng.integers(2, 4))):
            clones.append(d.model_copy(update={
                "x": d.x + float(rng.uniform(-0.02, 0.02)),
                "y": d.y + float(rng.uniform(-0.02, 0.02)),
                "confidence": max(0.05, d.confidence * float(rng.uniform(0.3, 0.9))),
            }))
    clean = analyze_detections(sim.detections, config=AnalyzeConfig())
    dirty = analyze_detections(sim.detections + clones, config=AnalyzeConfig())
    clean_c = sum(r.catches for r in clean.runs)
    dirty_c = sum(r.catches for r in dirty.runs)
    assert abs(dirty_c - clean_c) <= 1
```

  Confirm RED with `cluster_merge_dist=0.0` hardwired (or pre-integration) — record
  the inflated count; then wire `AnalyzeConfig.cluster_merge_dist=0.03` +
  `analyze_detections` integration → GREEN.
- [ ] **Step 4: realtime inheritance test** in `tests/test_realtime.py`: feed the same
  duplicate-injected stream through `RealtimeAnalyzer` (pattern of existing parity
  tests); assert final `catches_total` within ±1 of the clean live baseline.
- [ ] **Step 5: config validation** — `cluster_merge_dist >= 0` in the existing
  `AnalyzeConfig` validation (add validator if none exists), one RED/GREEN test.
- [ ] **Step 6: field spot-check (not committed):** re-run offline analyze from saved
  detections for ss3_id_086, ss3_id_110, ss441_id_089, ss3_id_079, ss441_id_013
  (`uv run juggletrack analyze <video> --out <scratch> --detections
  outputs/meschke-val/<stem>/detections.jsonl --no-overlay`): expect 57→~24, 21→~9,
  59→~19, 6→6, 137→137±1. If a gate video regresses, tune `merge_dist` with the
  measured justification — do not ship a value that breaks the clean-video constraint.
- [ ] **Step 7: full suite + ruff; commit** `feat: per-frame duplicate-box clustering
  kills parallel-arc over-count`.

### Task 3: run spans truncate only at floor-bound misses

**Files:**
- Modify: `src/juggletrack/events/drops.py` (extract predicate), `src/juggletrack/events/runs.py`
- Test: `tests/test_runs.py`

**Interfaces:**
- Produces: `is_floor_bound(arc: Arc, hand_line: float, *, floor_margin: float = FLOOR_MARGIN) -> bool`
  — extracted from `detect_drops`'s floor-descent test (drops.py:37) into a shared
  helper (in `events/drops.py`, imported by `runs.py`); `detect_drops` must use the
  same helper (single source of truth). Read `detect_drops` carefully for the y-axis
  convention before extracting — behavior of `detect_drops` must be byte-identical
  (existing drop tests pin it).
- `segment_runs` change: `first_miss` becomes the first arc that is uncaught AND
  `is_floor_bound(...)`. A merely-unwitnessed uncaught arc (extraction miss,
  occlusion) no longer truncates `end_t`; the else-branch (`max(arc_end(a) for a in
  group)`) applies. Update the long comment to the new semantics.

- [ ] **Step 1: failing test** in `tests/test_runs.py`:

```python
def test_unwitnessed_miss_does_not_truncate_run_span():
    """Plan-3 carry-forward, measured at scale (ss42_id_011): one unwitnessed
    catch mid-run truncated end_t to 38.7s while the run's 95 arcs (and its
    93 counted catches) span 0..200s. An uncaught arc that is NOT floor-bound
    must not truncate the span."""
    sim = simulate_cascade(n_throws=12, fps=30.0, seed=1)
    sr = analyze_detections(sim.detections, config=AnalyzeConfig())
    assert len(sr.runs) == 1
    full = sr.runs[0]
    # remove all detections of one mid-run flight so its catch is unwitnessed
    # (pick a frame window straddling one arc's descent; reuse the dropout
    # pattern from tests/test_analyze.py occlusion tests)
    ...  # build `gapped` detections
    sr2 = analyze_detections(gapped, config=AnalyzeConfig())
    assert len(sr2.runs) == 1
    assert sr2.runs[0].end_t > 0.9 * full.end_t   # span survives the miss
```

  (Adapt the gap construction from the existing occlusion tests; the binding
  assertions are: single run, span not truncated to the miss point.)
- [ ] **Step 2:** RED, then implement the predicate extraction + `segment_runs` change.
- [ ] **Step 3: floor-bound truncation still works** — add the positive case: end one
  sim run with `drop_at_throw` (existing sim support); assert the run's `end_t` stays
  at the drop's crossing (existing drop tests may already pin this — if so, cite them
  in the task report instead of duplicating).
- [ ] **Step 4:** full suite (existing drop/run/realtime tests green unchanged unless a
  pinned value legitimately shifts — any shift needs its own justification in the
  docstring per Global Constraints) + ruff.
- [ ] **Step 5: commit** `fix: run spans truncate only at floor-bound misses`.

### Task 4: field re-validation + findings doc

**Files:**
- Create: `docs/superpowers/plans/2026-08-03-meschke-validation-findings.md`
- (Uses `outputs/meschke-val/` scripts; nothing else committed.)

- [ ] **Step 1:** re-run offline analyze from saved detections for ALL 22 videos
  (write a small rerun script beside `outputs/meschke-val/run_validation.py` that
  skips detection and reuses each `<stem>/detections.jsonl`, including the two
  parser-fixed oracles), and re-run the realtime replays the same way.
- [ ] **Step 2:** re-run `outputs/meschke-val/aggregate.py`; capture before/after for
  every video and both spec §6 metrics.
- [ ] **Step 3: gates** —
  - ss3_id_086 total catches within 24±4; ss3_id_110 within 9±3; ss441_id_089 within
    19±5 AND drops ≤ 2; ss423_id_088 within 15±5.
  - ss42_id_011 matched-run IoU ≥ 0.9; ss42_id_010 IoU ≥ 0.9.
  - Clean-video constraint (Global Constraints list) holds.
  - Spec §6.1 fraction ≥ 55% overall, AND ≥ 80% restricted to cascade-and-low
    patterns (ss3/ss423/ss42/ss441 families) — high patterns are the documented
    scope boundary. If a gate fails, iterate within Tasks 2/3 designs; if the design
    can't reach it, STOP and report BLOCKED with per-video attribution.
- [ ] **Step 4: findings doc** — tables (before/after per video, oracle reference),
  the three mechanisms with evidence, the high-pattern scope boundary (candidate
  future fix: split-stitch across out-of-frame gaps), realtime envelope after the
  span fix (ss3_id_016 live replay especially — `end_t` feeds liveness), and the
  Meschke citation ("Stephen Meschke - Juggling Data Set -
  https://sites.google.com/view/jugglingdataset").
- [ ] **Step 5: commit** `docs: meschke 22-video validation findings — before/after hardening`.

## Self-review notes

- Task 2's integration point (top of `analyze_detections`) is what makes realtime
  inherit the fix without touching `realtime.py` — Step 4 pins that inheritance.
- Task 3 deliberately does NOT split groups at floor-bound misses into multiple runs
  (drop detection + downstream run segmentation already handle run boundaries);
  it only fixes the span/count inconsistency. Re-visit splitting only if Task 4's
  IoU gates fail.
- The known under-count mechanisms (detector recall on ss3_id_016; high-pattern
  fragmentation) are explicitly out of scope — Task 4 documents rather than fixes.
