# Turn-4 hard-negative re-audit — findings

**Date:** 2026-07-19 · **Follow-up to:** turn-4 findings §6 + the contamination-gates fix
(spec: `../specs/2026-07-19-hard-negative-contamination-gates-design.md`, commits `93475e6`
+ `e82e172` + `9a03c26`).

All turn-4 mined negative dirs were re-mined with the fixed `export_hard_negatives`
(all gates on, person veto via stock `yolo11n.pt`, including the post-review
per-detection staticness tightening and the post-audit presence-dilation hardening), and
the surviving outputs were visually audited frame-by-frame (zoom-verified) for real-ball
contamination. The audit loop was load-bearing twice: an adversarial code review found a
drift-through-cluster leak, and the final exhaustive audit found one contaminated frame
that then drove the person-gate redesign (below).

## Re-mine results (old → new)

v3-detector pass (replayed from `analyze-v3-negmine/*/detections.jsonl`, same as turn 4):

| Source | Old (`negatives-v3`) | New (`negatives-v5`) | Gate rejections (frames) | Visual audit |
|---|---|---|---|---|
| PXL_20260716_191622045 | 40 (≥19 real-ball) | **0** | 38 transient, 23 floor, 24 person | n/a (empty) |
| PXL_20260716_183035587 | 40 | **19** | 195 transient, 69 person | clean (19/19, exhaustive) |

Motion-detector pass (fresh `label --detector motion --stride 1`, new gates):

| Source | Old (`negatives/`) | New (`negatives-v5-motion`) | Gate rejections | Visual audit |
|---|---|---|---|---|
| PXL_20260716_182734164 | 25 | 0 | 22 transient | n/a (empty) |
| PXL_20260716_182847912 | 36 | 0 | 35 transient | n/a (empty) |
| PXL_20260716_183035587 | 28 | 1 | 7 transient | clean |
| af1 | 1 | **0** | 1 no-arcs | n/a (empty) |

Key confirmations:

- The three field-confirmed contaminated frames on 191622045 are all rejected, each by a
  gate designed for its class: frame 87 (two balls held at the hip) → person/persistence
  gates; frames 108/630 (ball crossing mouth/eye, unstitched flight) → transient/person.
  The video now yields zero negatives — correct, since its arc-unassigned content is
  overwhelmingly real balls (held at hands, face crossings, floor-resting).
- af1's single old negative came from a video where motion-pass arc extraction found **no
  arcs at all** — the new no-arcs guard refuses to mine such videos entirely.
- An adversarial review of the first implementation demonstrated a live leak (a
  slow-moving ball inheriting a trusted cluster's status by drifting through its
  neighborhood — greedy membership). The per-detection tightening (`e82e172`) closes it,
  and on real footage it also rejected a walking pet that the looser gate had trusted on
  182847912, plus the flickery MOG2 junk on both 1827* videos (their blobs wobble more
  than the tight evidence neighborhood). Those two videos drop to zero yield — fail-safe
  yield loss, not contamination.
- The final exhaustive audit (all 20 then-surviving frames, with source-video neighbor
  frames to resolve ambiguity) caught ONE contaminated frame the gates had passed:
  183035587 frame 221, a motion-blurred person entering carrying the balls in hand. The
  balls produced no detection (blind spot for every detection-space gate) and the person
  detector scored the blurred figure 0.246 — under threshold — while nailing neighbor
  frames at 0.57–0.86. Fix (`9a03c26`): gate 5 vetoes on person PRESENCE within ±5 frames
  of a candidate (a person cannot teleport) at conf 0.15. Frame 221 is now rejected; the
  surviving 19 frames match the exhaustive audit's clean list exactly.
- The kept junk is what mining was built for: framed wall pictures and fixed bathroom
  fixtures, zoom-verified ball-free.

## Contamination verdicts on the OLD dirs

- `negatives-v3/PXL_20260716_191622045` and `negatives-v3-fixed/191622045`: **contaminated**
  (the -fixed re-mine still contains frames 87 and 630; the activity-window pad cannot see
  held balls in long idle stretches or flights that never produced an arc). Quarantine.
- `negatives/af1`: untrustworthy (no-arc video). Quarantine.
- `negatives/PXL_20260716_182847912`: at least some frames contain a moving pet the old
  rule could not distinguish (not a ball, so not strictly contamination, but untrusted
  motion content). Prefer the v5 dirs.
- All other old dirs: no ball contamination found, but they were mined with none of the
  new gates — prefer the v5 dirs wholesale rather than auditing old frames individually.

## Action for the next retrain (v4b/v5)

Point the negatives portion of the corpus at `outputs/turn4/negatives-v5/` +
`outputs/turn4/negatives-v5-motion/` (20 verified-clean images) and drop every
`outputs/turn4/negatives/` and `negatives-v3*/` source. Negative count falls 170 → 20;
that is the price of verified purity, and the v4 postmortem says contaminated negatives
cost more recall than clean-but-few negatives can buy back. If more negative yield is
wanted later, the exposed knobs (`persist_radius`, `min_persist_frac`, `floor_band_y`)
can be revisited — but only with a fresh visual audit of whatever extra frames they admit,
never by tuning until a target count appears.
