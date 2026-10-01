# Drift-cohort gate: real-footage findings

Date: 2026-09-30. Scope: `events/validate.is_drift_cohort`, the run-level gate
that deletes a whole run when its arcs look like one drifting junk object.

## Summary

- The gate deleted three real juggling runs on footage outside the turn-4
  battery. Over the same footage it caught real junk once.
- The fix adds a fourth necessary condition. The cohort must sweep, which means
  each arc's x-range lies wholly outside the hull of all earlier arcs.
- Across all 113 cached real sessions, 110 are byte-identical, 3 gained a run,
  none lost a run, and none of the 22 Meschke ground-truth sessions changed.

## What the gate rejected

The battery is every `detections.jsonl` cached under `outputs/`: 113 sessions,
202 runs that reach the gate. Before the fix the gate rejected 4 runs. Each
verdict comes from inspecting the video frames with the fitted arcs drawn on.

| Session | Detector | Verdict by eye | Arc x-ranges, in time order | Sweeps |
|---|---|---|---|---|
| `plan2-validation/af2` | stock COCO yolo11n | Real juggling | [0.569, 0.674] [0.584, 0.656] [0.610, 0.766] | No |
| `plan2-validation/pxl` | stock COCO yolo11n | Real juggling | [0.503, 0.574] [0.512, 0.526] [0.554, 0.818] | No |
| `labels-motion/191622045` | motion | Real juggling | [0.418, 0.673] [0.433, 0.546] [0.252, 0.509] | No |
| `labels-motion/182734164` | motion | Junk, a person entering frame | [0.440, 0.446] [0.608, 0.639] [0.939, 0.989] | Yes |

The synthetic fixture `test_slow_drift_junk_cohort_known_gap` also sweeps, with
ranges [0.15, 0.33] [0.37, 0.55] [0.59, 0.77].

The mechanism is detector fragmentation. A weak detector recovers about three
throws from a real cascade. If those throws share a direction, the cohort is
unidirectional and never alternates. A 3-point Pearson correlation then gives
`mono` above 0.9 easily. Junk traces new ground. Juggling keeps revisiting one
pattern, so its x-ranges overlap.

On the shipped v3 detector the gate fired on none of the cached footage.
On the 22 Meschke ground-truth videos it judged 56 runs and rejected 0, and on
the ground-truth oracle path it judged 26 runs and rejected 0. Long real runs
still sat close to it. `ss42_id_011` is a 95-arc run with 93 true catches,
`dom2 = 1.0`, `alt2 = 0.0` and `mono = 0.61`. The sweep condition makes such
runs structurally safe, because one overlapping arc keeps the whole run.

## Alternatives rejected

- **A gravity floor on the cohort.** Real runs have median `ay` from 0.058 to
  1.442, with a 10th percentile of 0.095. The junk fixture has `ay = 0.1`.
  Commit `73b0b0f` removed an absolute floor because it zeroed out ground-truth
  footage.
- **A minimum arc count.** The pinned junk fixture has 3 arcs.
- **A minimum fraction of meaningful arcs.** Real `ss42` runs have every arc
  meaningful, the same as the junk.

## Verification

- 13 new tests pin real cohorts, real junk, and the sweep semantics, and 2
  strict-xfail tests pin the first two known gaps below. All 8 sweep mutants
  are killed, each by the case built for it. The mutants are
  any-step, first-arc-only, last-arc-only, previous-arc-only, rightward-only,
  touching-as-disjoint, always-true and always-false.
- Before and after snapshots of all 113 sessions differ only in the three real
  sessions above. Each gained the deleted run. `af2` went from 21 to 24
  catches, `pxl` from 2 to 5, and `191622045` from 0 to 1.
- Independent reviewers re-cut each real cohort's arc windows by up to 4
  frames. The new gate rejected them 0% of the time. The old gate rejected
  them 50 to 80% of the time.

## Known gaps

- **A drifting juggler with sparse detection still sweeps.** If the detector
  recovers one ball and the juggler walks or the camera pans faster than the
  throws move sideways, each arc lands on new ground. Reviewers placed the
  threshold near `hand_sep / (2 * period + flight)`, about 0.09 frame-widths
  per second in simulator geometry. The old gate deleted these runs too.
  Pinned by `test_panning_juggler_with_one_ball_detected_is_not_flagged`.
- **Junk escapes when its pieces overlap in x.** Examples are two body parts
  of one walker, abutting segments of one continuous track, and a piece
  tracked long enough to reach back over an earlier one. An arc's x-range
  spans only the frames it was tracked. Pinned by
  `test_junk_with_two_overlapping_pieces_is_flagged`.
- **The boundary has no tolerance.** Reviewers disagreed on the direction. One
  would loosen it to catch abutting junk. Another would tighten it to protect
  juggling near the edge. No real cohort sits within 0.02 of it, so neither
  change shipped.
- **The one real junk catch is fragile.** Under the same 4-frame re-cut, both
  gates catch `182734164` about 74% of the time, because a 74 ms difference in
  start order sets its monotonicity. This predates the fix.
- **`alt2` is implied by `dom2`.** At `DOM_T = 0.99`, every meaningful arc
  shares a sign for any run under 200 meaningful arcs, so `alt2` is always 0
  or undefined when `dom2` passes. This predates the fix.

## Open question: keep the gate at all?

Deleting the gate was measured on the same 113 sessions. It differs from the
sweep fix on exactly one session, `labels-motion/182734164`, where it adds the
junk run and 1 phantom catch. Every other session is identical. That session
is cold-start footage from the motion detector, and the shipped `label`
command does not route through this gate. On the shipped v3 detector the gate
has not fired on any cached footage, before or after the fix.

After this fix the gate is close to inert on all observed footage. Keeping it
costs about 40 lines and leaves the drifting-juggler gap above. Deleting it
removes the only defense against the one junk shape seen on real footage. That
is a product call about phantom catches against missed runs, so it stays open.
