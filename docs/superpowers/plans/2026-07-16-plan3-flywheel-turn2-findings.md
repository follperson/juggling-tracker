# Plan 3 Task 8 (Turn 2): Flywheel Turn 2 — Findings

## Linker sweep

**Context.** Turn 2's FN diagnosis (`.superpowers/sdd/turn2-report.md` step 3) found `extract_arcs` returns **zero arcs** on both `PXL_20260716_182734164` and `PXL_20260716_191622045` despite good-to-excellent detector coverage (≥2 dets on 56.9% / 96.6% of sampled frames respectively), and hypothesized the fixed `link_max_dist=0.08` — tuned on turn-1's environment — is too tight for this closer-framed portrait footage. Commit 1 of this task plumbed `link_max_dt`/`link_max_dist`/`em_iters` through `AnalyzeConfig` and the CLI so that hypothesis could actually be tested per-video. This section is the test.

**Method.** For each saved `detections.jsonl`, ran `extract_arcs(dets, link_max_dist=d, link_max_dt=t)` for `d ∈ {0.08, 0.12, 0.16, 0.20, 0.25}` × `t ∈ {0.12, 0.25}` (all other params at `extract_arcs` defaults: `resid_tol=0.02`, `min_points=6`, `min_duration=0.15`, `g_range=(0.5, 8.0)`, `em_iters=5`). For each combo, `assign_detections(dets, arcs)` (default `resid_tol=0.02`) counts total assigned detections.

### Sweep table

**PXL_20260716_182734164** (1,510 detections, 750 sampled frames, stride 2):

| link_max_dist | link_max_dt | arcs | assigned dets | median ay | throws | catches |
|---|---|---|---|---|---|---|
| 0.08 | 0.12 | 0 | 0 | — | 0 | 0 |
| 0.08 | 0.25 | 0 | 0 | — | 0 | 0 |
| 0.12 | 0.12 | 0 | 0 | — | 0 | 0 |
| 0.12 | 0.25 | 0 | 0 | — | 0 | 0 |
| 0.16 | 0.12 | 0 | 0 | — | 0 | 0 |
| 0.16 | 0.25 | 0 | 0 | — | 0 | 0 |
| 0.20 | 0.12 | 0 | 0 | — | 0 | 0 |
| 0.20 | 0.25 | 0 | 0 | — | 0 | 0 |
| 0.25 | 0.12 | 0 | 0 | — | 0 | 0 |
| 0.25 | 0.25 | 0 | 0 | — | 0 | 0 |

**PXL_20260716_191622045** (1,412 detections, 377 sampled frames, stride 2):

| link_max_dist | link_max_dt | arcs | assigned dets | median ay | throws | catches |
|---|---|---|---|---|---|---|
| 0.08 | 0.12 | 0 | 0 | — | 0 | 0 |
| 0.08 | 0.25 | 0 | 0 | — | 0 | 0 |
| 0.12 | 0.12 | 0 | 0 | — | 0 | 0 |
| 0.12 | 0.25 | 0 | 0 | — | 0 | 0 |
| 0.16 | 0.12 | 0 | 0 | — | 0 | 0 |
| 0.16 | 0.25 | 0 | 0 | — | 0 | 0 |
| 0.20 | 0.12 | 0 | 0 | — | 0 | 0 |
| 0.20 | 0.25 | 0 | 0 | — | 0 | 0 |
| 0.25 | 0.12 | 0 | 0 | — | 0 | 0 |
| 0.25 | 0.25 | 0 | 0 | — | 0 | 0 |

**Zero arcs at every point in the requested grid, on both videos.** No plateau to describe — the numbers never move off zero, so there is no "median ay" or event-derivation count to report for a "chosen setting" from this grid.

**Extended check (beyond the requested grid, for root-cause context).** Pushed `link_max_dist` further to see whether *any* width recovers substantial arcs: at `dist ∈ {0.3, 0.4, 0.5}` (both `dt` values) — still 0 arcs on both videos. At `dist=0.6, dt=0.12`: 1 arc on video 1. At `dist=0.4, dt=0.25`: 1 arc on video 2. At `dist=0.8, dt=0.12`: still just 1 arc on video 1. So even widening the linker by 10x over the default never recovers more than a single arc on either video — nowhere near "substantial."

**Root cause is not (solely) linking distance.** Manually stepping through `_link_fragments` → `_split_ballistic` → initial `fit_arc` on video 1 shows the linker *does* form large fragments even at the default `link_max_dist=0.08` (sizes up to 324, 299, 277 points) — contradicting the "never links" phrasing of the original hypothesis. The problem is what they link into: these large fragments are near-static high-confidence false positives (fitted `ay ≈ 0.000`, spanning 20–24 seconds) — almost certainly a background object the detector consistently mistakes for a ball. Widening `link_max_dist`/`link_max_dt` makes this worse, not better: at wider settings the same kind of static-point fragment grows even larger (409, 701 points), because the greedy nearest-neighbor linker has more competing open fragments alive at once and a real ball's next sample can get absorbed into the nearest *static* fragment's prediction instead of its own true (fast-moving) fragment. A handful of genuine gravity-plausible seeds (`ay` in `[0.25, 4.0]`) do appear in the initial-fit stage at wider settings (up to 7 simultaneously, e.g. `ay=3.86, n=4`; `ay=0.30, n=14`), but they are short (4–14 points, sub-1s) and get eliminated by `_prune`/`_gravity_prune` in the EM loop — never more than one survives to the end, and even that one doesn't survive a second EM iteration.

**Recommended setting.** Because no in-grid setting changes the outcome on either video, there is no evidence-based "smallest setting that recovers substantial arcs" to recommend *for these two specific videos*. For the general-purpose knob (informed by the synthetic downsampling scenario in `tests/test_analyze.py::test_config_plumbs_linker_knobs`, where widening `link_max_dist` to 0.2 with `link_max_dt=0.35` did recover a fully-linked cascade that the 0.08 default missed), a moderate default of **`link_max_dist=0.12`, `link_max_dt=0.2`** is a reasonable starting point for footage with wider per-sample displacement than turn-1's environment — enough headroom to link real flight without the extreme over-linking risk seen at 0.4+. But this pair of turn-2 videos needs a different fix before the linker distance matters at all: filtering (or down-weighting) the dominant static/background false-positive cluster before `_link_fragments` runs, so it stops winning the greedy nearest-neighbor competition against genuine ball flight. That was already the extractor-side direction flagged in `turn2-report.md` step 3; this sweep sharpens it from "adaptive linking distance" to specifically "background-point filtering," since linking distance alone — checked exhaustively from 0.08 to 0.8 — does not fix it.
