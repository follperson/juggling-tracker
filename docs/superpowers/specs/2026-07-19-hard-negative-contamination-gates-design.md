# Hard-negative contamination gates — design

**Date:** 2026-07-19 · **Follow-up to:** turn-4 findings §6 (`2026-07-18-turn4-findings.md`)

## Problem

`export_hard_negatives` exports zero-box negative training images from frames where every
detection is arc-unassigned. But "arc-unassigned" ≉ "not a ball", and two leak classes survive
even the turn-4 activity-window (`pad`) guard:

1. **Real balls at rest** — held in-hand during idle stretches more than `pad` from any arc
   (PXL_20260716_191622045 frame 87, t≈2.9 s vs first arc at 5.2 s), and balls resting on the
   floor for most of the video (clusters at y≈0.96, spanning 82–100 % of the video).
2. **Unstitched flights** — a real ball crossing in front of the juggler's face (frames
   108/630) where the linker never produced an arc, so no activity window exists to exclude it.

Both classes are still present in the post-`pad`-fix re-mine (`negatives-v3-fixed/191622045`
contains frames 87 and 630). These negatives poisoned v4 (outdoor recall regressed; v4 was not
promoted; v3 remains champion).

## Falsified designs (dry-run on turn-4 saved detections)

Dry-runs against `outputs/turn4/analyze-v3-negmine/*/detections.jsonl` — the known-clean video
(183035587, junk = framed wall pictures) versus the known-contaminated one (191622045, "junk"
= held/face/floor real balls) — killed three simpler designs before implementation:

- **Persistence/dwell alone.** Real-ball clusters have long time spans too: face-height
  clusters span 74–78 % of the video (the ball recrosses the same spot), floor-resting balls
  82–100 %. Density overlaps as well (0.44–0.55 real vs 0.34–0.65 junk). No separation.
- **Arc-space proximity.** Genuine junk sits as close as d=0.025 to arc trajectories (an arc
  flies right past a wall picture), while real held/floor balls sit 0.12–0.45 away (arcs
  truncate before reaching the hands, so the pipeline literally cannot see where hands are).
  Inverted/overlapping margins.
- **Ball-count certificate** ("junk must co-occur with ≥3 assigned detections"). Inverted:
  dense detections fire multiple boxes per ball, so held-ball frames reach 8 assigned
  detections, while the clean video has only 2 frames with ≥3 assigned. Certifies the
  contamination, starves the junk.

## Design: layered vetoes, one per named failure mode

A frame must pass **all** gates to export. Every gate fails safe (rejects yield, never admits
contamination). Order (cheap/pure first):

0. **No-arcs guard.** Zero extracted arcs → zero negatives (`n_images == 0`). "Arc-rejected"
   is meaningless without arcs, and a no-arc juggling video is exactly the
   maximum-contamination case (extraction failed everywhere). Stat: `n_skipped_no_arcs`
   (counts every detection-bearing frame).
1. **All-unassigned** (existing). Any assigned detection → `n_skipped_ambiguous`. (`min_junk`
   stays the final uncounted filter, as today, applied after gate 4 and before gate 5 so the
   person model never runs on frames that cannot export anyway.)
2. **Activity window** (existing, `pad=1.0`). Frame time within `pad` of any arc's
   `[t_start, t_end]` → `n_skipped_active`.
3. **Static-persistence gate** (new) — rejects *moving* unassigned objects (unstitched flight
   fragments). Cluster idle-frame unassigned detections by greedy nearest-centroid within
   `persist_radius` (0.04 ≈ one default ball-width). A cluster is *trusted junk* iff
   (a) its idle membership has ≥ `min_persist_count` (8) detections, and (b) the time span of
   **all** unassigned detections within `persist_radius` of its centroid (any frame, so a wall
   picture's during-run firings count as evidence) covers ≥ `min_persist_frac` (0.5) of the
   video's overall detection span. A frame with any detection outside every trusted cluster →
   `n_skipped_transient`.
4. **Floor-band veto** (new) — rejects floor-resting balls, which are static and persistent
   (indistinguishable from junk by gates 0–3). Any detection with y > `floor_band_y` (0.85) →
   `n_skipped_floor`. Sacrifices genuine floor-clutter junk; camera angles where the floor
   appears higher in frame are a documented residual risk.
5. **Person-region veto** (new) — rejects held balls and face/body-crossing balls. If person
   boxes are available for a frame, any detection inside a person box expanded by
   `person_margin` (0.02) → `n_skipped_person`. Frames with no detected person pass (held/face
   balls require a person by definition; empty-scene junk is the safest yield there is).
   Person boxes come from the stock COCO model (`yolo11n.pt`, class 0 = person), already a
   repo dependency via ultralytics — run only on frames that survived gates 0–4, so the cost
   is a few dozen single-frame inferences per video.

Gates 3–5 divide the evidence cleanly: 3 rejects what *moves*, 4–5 reject what is *static but
on the person or floor*. The wall-picture junk that motivated mining passes everything
(measured: spans 0.94–0.97, above floor band, outside person).

## API

- `export_hard_negatives(...)` gains keyword params: `persist_radius=0.04`,
  `min_persist_frac=0.5`, `min_persist_count=8`, `floor_band_y=0.85`,
  `person_boxes: dict[int, list[tuple[float, float, float, float]]] | None = None`
  (normalized xyxy per frame; injectable for tests), `person_model: str | None = None`
  (when set and `person_boxes` is None, boxes are computed on gate-surviving candidate frames
  via the helper below).
- New helper `detect_person_boxes(video_path, frame_indices, *, model="yolo11n.pt", conf=0.25)
  -> dict[int, list[tuple]]` in `autolabel.py` (lazy ultralytics import).
- Stats dict gains `n_skipped_transient`, `n_skipped_floor`, `n_skipped_person`,
  `n_skipped_no_arcs`.
- `negatives_manifest.json` gains `trusted_clusters` (centroid, n, t-span) for future audits.
- CLI `label --negatives`: echo new stats; new `--neg-person-model` option (default
  `yolo11n.pt`; `none` disables).

## Testing (TDD, tests/test_autolabel.py)

Update the `_expected_candidates` reference helper to mirror the new gate precedence. New
sim-based fixtures/tests (no model inference — person boxes injected as dicts):

- Idle-hold leak repro (frame-87 shape): synthetic idle segment with junk + two held-ball
  detections at hand positions, outside all pad windows; with injected person boxes the frames
  are rejected (`n_skipped_person`), and without person boxes they are still rejected by gate 3
  only if transient — the test asserts the person gate specifically.
- Unstitched-flight repro (frame-108/630 shape): 5-point ballistic ghost (below `extract_arcs`'
  `min_points=6`) during the idle tail → rejected as transient.
- Floor-resting ball: persistent static detections at y≈0.95 during idle → rejected by
  floor band despite being trusted-persistent.
- No-arcs guard: junk-only detections, no arcs → zero images.
- Yield preservation: persistent junk fixture still mines its idle tail (existing tests keep
  passing with new stats keys).

## Re-audit plan

Re-mine both v3-pass videos from saved `detections.jsonl` with all gates + real person boxes →
`outputs/turn4/negatives-v5/`; re-mine the motion-pass sources similarly. Compare against
`negatives/`, `negatives-v3/`, `negatives-v3-fixed/` manifests; visually verify samples of
kept/rejected frames; report per-dir contamination verdicts and which dirs to quarantine
before any v4b/v5 retrain.

## Residual risks (documented, accepted)

- Statue-still juggler holding balls at one spot for most of the video with the person
  detector failing on them: passes gates 0–4. Mitigated by person veto in practice.
- Camera angles with floor above y=0.85 (param exposed).
- Person-detector misses reduce gate 5 to a no-op on those frames (fail-open for that gate;
  gates 3–4 still apply).
- A ball momentarily *colocated* with a trusted junk cluster's own footprint (within
  `persist_radius / 2` of recurring junk detections) is geometrically indistinguishable
  from that junk and passes gate 3. Adversarial review demonstrated the wider variant —
  a slow drifter merely *near* a trusted cluster inheriting its trust via greedy
  membership — which is now closed by per-detection tight-neighborhood evidence
  (`_trusted_static_clusters`); only exact colocation remains.
- The library-level defaults run without gate 5 (`person_boxes=None, person_model=None`);
  the CLI defaults it on. Any script mining training data via the library API must pass
  `person_model=` explicitly (docstring carries a WARNING).

## Deviation from plan (recorded post-implementation)

The `_expected_candidates` test helper mirrors gates 0–2 only; gates 3–5 are pinned by
dedicated fixtures/tests with exact stat counts instead of a full parallel implementation
(which would just duplicate the algorithm and dilute the reference value).
