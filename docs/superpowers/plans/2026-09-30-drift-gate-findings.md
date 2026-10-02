# Drift-cohort gate: real-footage findings

Date: 2026-09-30. Scope: `events/validate.is_drift_cohort`, the run-level gate
that deletes a whole run when its arcs look like one drifting junk object.

## Summary

- In offline analysis, the gate deleted three real juggling runs on footage
  outside the turn-4 battery. Over the same footage it caught real junk once.
- The fix adds a fourth necessary condition. The cohort must sweep, which means
  each arc's x-range lies wholly outside the hull of all earlier arcs.
- In offline analysis of all 113 cached real sessions, 110 are byte-identical,
  3 gained a run, none lost a run, and none of the 22 Meschke ground-truth
  sessions changed.
- Live mode moves more, because the old gate fired often on 8 s windows of real
  juggling. See [Live mode](#live-mode).
- On the ground-truth oracle path, 1 of 159 Meschke sessions changes. It gains
  a real run.

## What the gate rejected

The battery is every `detections.jsonl` cached under `outputs/`, run through
offline analysis: 113 sessions, 202 runs that reach the gate. Before the fix
the gate rejected 4 runs. Each verdict comes from inspecting the video frames
with the fitted arcs drawn on.

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

In offline analysis on the shipped v3 detector, the gate fired on none of the
cached footage. On the 22 Meschke ground-truth videos it judged 56 runs and
rejected 0, and on those videos' oracle path it judged 26 runs and rejected 0.
Long real runs still sat close to it. `ss42_id_011` is a 95-arc run with 93
true catches, `dom2 = 1.0`, `alt2 = 0.0` and `mono = 0.61`. The sweep condition
makes such runs structurally safe, because one overlapping arc keeps the whole
run. Live mode is different. See [Live mode](#live-mode).

## Alternatives rejected

- **A gravity floor on the cohort.** Real runs have median `ay` from 0.058 to
  1.442, with a 10th percentile of 0.095. The junk fixture has `ay = 0.1`.
  Commit `73b0b0f` lowered the absolute floor from `ay` 0.25 to 0.05 because
  the old floor zeroed out ground-truth footage.
- **A minimum arc count.** The pinned junk fixture has 3 arcs.
- **A minimum fraction of meaningful arcs.** Real `ss42` runs have every arc
  meaningful, the same as the junk.

## Verification

- 13 new tests pin real cohorts, real junk, and the sweep semantics, and 2
  strict-xfail tests pin the first two known gaps below. All 8 sweep mutants
  are killed, each by the case built for it. The mutants are
  any-step, first-arc-only, last-arc-only, previous-arc-only, rightward-only,
  touching-as-disjoint, always-true and always-false.
- Before and after offline snapshots of all 113 sessions differ only in the
  three real sessions above. Each gained the deleted run. `af2` went from 21
  to 24 catches, `pxl` from 2 to 5, and `191622045` from 0 to 1.
- Independent reviewers re-cut each real cohort's arc windows by up to 4
  frames. The new gate rejected them 0% of the time. The old gate rejected
  them 50 to 80% of the time.

## Live mode

Everything above is offline analysis. Live mode re-analyzes a sliding 8 s
window every 3 frames, and the gate judges every window's runs. The fix moves
more there.

**Method.** Each saved `detections.jsonl` is replayed frame by frame through
`RealtimeAnalyzer`. Each frame gets its recorded timestamp: the saved `t` on
frames with detections, and linear interpolation between them on empty
frames. This is the clock that `juggletrack live VIDEO --detections FILE`
feeds. The old gate is commit `324ad61` and the new gate is commit `d832cc7`.
The battery is the 22 `outputs/meschke-val/ss*` sessions at full length, plus
`meschke-val/post-hardening/ss3_id_016-1280` and `turn4/analyze-v4/ss3_id_016`.
That is 24 sessions.

**What moved.** 13 sessions confirm identical events. 6 differ only by catch
re-timings of at most 0.09 s. Runs are unchanged everywhere. 5 sessions change
counts:

| Session | Old [runs, catches, drops] | New | What moved |
|---|---|---|---|
| `ss42_id_010` | [1, 85, 0] | [1, 89, 0] | +3 real catches at 81.38, 83.53 and 143.31 s. +1 duplicate at 5.04 s. |
| `ss8040_id_070` | [1, 29, 2] | [1, 33, 2] | +3 real catches at 51.45, 55.13 and 56.79 s. +1 duplicate at 13.42 s. |
| `ss90501_id_145` | [3, 15, 1] | [3, 16, 1] | +1 duplicate at 18.53 s. |
| `ss3_id_016` | [9, 210, 7] | [9, 210, 8] | +1 false drop at 62.60 s. |
| `turn4/analyze-v4/ss3_id_016` | [8, 201, 5] | [8, 202, 5] | +1 duplicate at 109.83 s. |

Each real catch lands within 0.25 s of a ground-truth arc end on the oracle
path. Each duplicate sits 0.155 to 0.208 s from another confirmed catch. Where
the oracle has an arc there, at 5.04 s and 13.42 s, only one ball lands. The
false drop is on ball 1 of `ss3_id_016`. The labels show that ball bottoming
out at y 0.80 near 63.2 s and being thrown again.

**Net effect.** +6 real catches, +4 duplicate catches and +1 false drop.

**Mechanism.** The old gate fired in 975 analysis cycles across 16 of the 24
sessions, up to 357 cycles in `ss42_id_010`. Each firing deleted a window's
run of 3 to 5 arcs, and that cycle's catches with it. The windows inspected
are real juggling. An 8 s window often holds a few throws from one hand that
share a direction, and their x-ranges overlap. The new gate keeps them.
`test_live_window_of_one_hand_throws_is_not_flagged` pins one such window
from `ss42_id_010`.

The duplicates and the false drop are existing live behaviour that the deleted
windows happened to hide. In `ss3_id_016` the old gate fired from feed time
64.97 s. The drop at 62.60 s is first confirmed at 65.17 s, once the 2.5 s
drop freeze has passed.

The new gate still fires in 3 cycles of `ss90501_id_145`, at feed times 33.87
to 38.37 s. The old gate fired there too. The labels show real juggling, and
the oracle path recovers the same three arcs. They drift right with x-ranges
[0.512, 0.544] [0.546, 0.552] [0.600, 0.606], so they sweep. This is the
drifting-juggler gap below, seen on real footage.

**Clock caveat.** Live output is sensitive at the 1-ulp clock level. On
`ss3_id_016`, feeding `frame_idx / fps` instead of the recorded clock moves
each frame's time by at most 2.8e-14 s. With identical code, that changes the
old gate's result from [9, 210, 7] to [8, 206, 8], and the new gate's from
[9, 210, 8] to [8, 207, 9]. Only recorded-clock numbers are live numbers.
`benchmark --realtime` feeds the recorded clock for this reason.

**Open follow-ups.**

- Live mode confirms re-fits of one catch as two catches when they land more
  than `event_match_tol` (0.15 s) apart. This accounts for 4 of the 11 changes
  above.
- Live mode confirms a false drop in `ss3_id_016` once the gate stops deleting
  its window. The cause is not isolated yet. Reviewers suspect windows that
  see only one side of the ball's path.

## Oracle path

`oracle_events` was run on all 159 Meschke CSVs, with each video's real fps
and frame size, under both gates. One session changes. `trick_4up3up_id_885`
goes from 1 run and 4 catches to 2 runs and 7 catches. The regained run spans
4.82 to 9.39 s with 3 catches. The oracle is built from ground-truth ball
trajectories, so the regained run is real juggling. The other 158 sessions,
including the 22 validation videos, are byte-identical.

## Known gaps

- **A drifting juggler with sparse detection still sweeps.** If the detector
  recovers one ball and the juggler walks or the camera pans faster than the
  throws move sideways, each arc lands on new ground. Reviewers placed the
  threshold near `hand_sep / (2 * period + flight)`, about 0.09 frame-widths
  per second in simulator geometry. The old gate deleted these runs too.
  Pinned by `test_panning_juggler_with_one_ball_detected_is_not_flagged`.
- **The sweep test ignores direction.** `_sweeps` asks only that each arc
  clear the hull of the earlier arcs. It does not ask that the cohort march
  the way its arcs fly. Three arcs that each fly right but start further left
  each time, with x-ranges [0.75, 0.93] [0.53, 0.71] [0.31, 0.49], still
  sweep and are still deleted. The junk seen so far does not move like this. A
real juggler
  drifting slowly against the direction of the detected throws can. No test
  pins this.
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

Deleting the gate was measured in offline analysis on the same 113 sessions.
It differs from the sweep fix on exactly one session, `labels-motion/182734164`,
where it adds the junk run and 1 phantom catch. Every other session is
identical. That session is cold-start footage from the motion detector, and
the shipped `label` command does not route through this gate. In offline
analysis on the shipped v3 detector, the gate has not fired on any cached
footage, before or after the fix. Live mode was not part of this measurement.

After this fix the gate is close to inert in offline analysis of all observed
footage. In live replay it still fires on 1 of the 24 sessions measured above,
in windows the old gate also deleted. Keeping it costs about 40 lines and
leaves the drifting-juggler gap above. Deleting it removes the only defense
against the one junk shape seen on real footage. That is a product call about
phantom catches against missed runs, so it stays open.
