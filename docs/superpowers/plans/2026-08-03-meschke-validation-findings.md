# Plan 5 Task 4: Meschke 22-Video Validation Findings — Before/After Hardening

**Status: exploratory measurement, not TDD.** This document re-runs the full 22-video
Meschke oracle validation from **saved detections** (no re-detection) through the
Plan-5-hardened pipeline (`claude/juggletrack-05-hardening`, commits `6f3db2e`
CSV parser fix, `c4b8d03`/`46136b2`/`9f9d303` duplicate-box clustering, `fce5e29`/
`2ff89d4` floor-bound-only run-span truncation) and compares it to the pre-fix
baseline recorded in `docs/superpowers/plans/2026-08-03-juggletrack-05-validation-hardening.md`.
Nothing in the event core was changed to produce these numbers — this is
measurement and attribution, per the plan's own scoping ("Task 4 documents rather
than fixes").

**Data set:** Stephen Meschke — *Juggling Data Set* —
https://sites.google.com/view/jugglingdataset. 22 three-ball vanilla-siteswap
clips with per-frame ball-center ground truth, from which
`juggletrack.data.meschke_import.oracle_events` derives per-ball run/catch/drop
events (the oracle used throughout).

**Method:** `outputs/meschke-val/rerun_validation.py` (companion to
`run_validation.py`) reads each `<stem>/detections.jsonl` already saved under
`outputs/meschke-val/` (main checkout), and for every video shells out to
`uv run juggletrack analyze --detections ...` / `... live --detections ...`
with `cwd` set to **this worktree**, so the hardened `src/juggletrack` code
runs while video/csv/model paths (untracked, main-checkout-only) resolve
correctly. Oracle counts are recomputed with the worktree's parser fix.
Outputs land in `outputs/meschke-val/post-hardening/` — the pre-fix artifacts
in `outputs/meschke-val/<stem>/` are untouched. `outputs/meschke-val/rerun_aggregate.py`
mirrors `aggregate.py`'s per-run greedy-IoU matching and spec-§6 metrics,
pointed at the new directory. All 22 videos completed with no errors on this
run (the pre-fix batch had 2 errors: `ss531_id_988/989`, the parser crash).

## 1. Per-video table (`runs/catches/drops`)

| video | oracle | offline BEFORE | offline AFTER | live BEFORE | live AFTER |
|---|---|---|---|---|---|
| ss3_id_016 | 5/52/0 | 9/39/0 | 9/39/0 | 9/207/5 | 9/213/6 |
| ss3_id_079 | 1/6/0 | 1/6/0 | 1/6/0 | 1/6/0 | 1/6/0 |
| ss3_id_086 | 1/24/0 | 1/57/1 | 1/25/0 | 1/48/8 | 1/26/0 |
| ss3_id_110 | 1/9/0 | 1/21/0 | 1/17/0 | 1/21/0 | 1/18/0 |
| ss3_id_987 | 1/20/0 | 1/21/0 | 1/22/0 | 1/22/0 | 1/22/0 |
| ss423_id_007 | 1/73/0 | 1/82/0 | 1/79/0 | 3/83/5 | 1/82/3 |
| ss423_id_017 | 1/130/0 | 1/121/0 | 1/120/0 | 1/143/8 | 1/138/13 |
| ss423_id_088 | 1/15/0 | 1/44/6 | 1/18/0 | 1/35/14 | 1/20/0 |
| ss42_id_010 | 1/66/0 | 1/65/0 | 1/68/0 | 1/86/0 | 1/85/0 |
| ss42_id_011 | 1/93/0 | 1/93/0 | 1/93/0 | 1/94/0 | 1/94/0 |
| ss441_id_013 | 1/136/0 | 1/137/0 | **1/136/0** | 1/136/0 | 1/141/0 |
| ss441_id_089 | 1/19/0 | 1/59/19 | **1/19/0** | 1/48/21 | 1/19/1 |
| ss50505_id_012 | 1/157/0 | 1/155/0 | 1/151/0 | 1/148/2 | 1/147/2 |
| ss50505_id_093 | 1/42/0 | 1/6/35 | **1/38/0** | 1/35/112 | 2/29/9 |
| ss51_id_163 | 1/58/0 | 1/41/0 | 1/51/0 | 1/50/0 | 2/58/0 |
| ss531_id_005 | 1/63/0 | 1/40/0 | **2/16/0** | 1/64/0 | 2/81/1 |
| ss531_id_014 | 1/29/0 | 1/25/0 | 1/29/0 | 2/36/2 | 1/41/0 |
| ss531_id_988 | 1/21/0 | n/a (oracle crash) | 1/17/0 | n/a | 1/26/0 |
| ss531_id_989 | 1/19/0 | n/a (oracle crash) | **2/15/0** | n/a | 1/37/0 |
| ss60_id_151 | 1/26/0 | 1/29/0 | 1/29/0 | 1/32/0 | 2/33/0 |
| ss8040_id_070 | 1/33/0 | 1/36/0 | 1/33/0 | 1/23/0 | 1/26/1 |
| ss90501_id_145 | 1/5/0 | 1/11/0 | 1/9/0 | 3/17/1 | 1/23/4 |

Bold = exact oracle match. `n/a` = pre-fix `oracle_events` crashed on float
coordinates (`ss531_id_988/989`); their offline/live BEFORE numbers are not
"missing data", they never ran the live step either — `run_validation.py`
raised on step 2 before reaching step 3.

**Totals (22 videos, oracle uniformly recomputed with the parser fix):**
oracle 1096 catches; offline BEFORE 1128 (**+2.9%**); offline AFTER 1030
(**−6.0%**); live BEFORE 1334 (n=20, 988/989 excluded); live AFTER 1365 (n=22).

The *officially recorded* pre-fix headline (20/22 scorable videos, exactly as
measured in the hardening plan's evidence section) was §6.1 **27.8%**, §6.2
**22.2%**. See §2.

## 2. Spec §6 metrics, before → after

Spec (`docs/superpowers/specs/2026-07-15-juggling-tracker-rebuild-design.md` §6):
§6.1 catch-count error per run, target ≤1 on ≥90% of runs. §6.2 run-boundary
IoU, target ≥0.9 on ≥80% of runs.

| scope | §6.1 (≤1 catch Δ) | §6.2 (IoU ≥0.9) | matched runs |
|---|---|---|---|
| BEFORE, 20/22 videos (official baseline) | 27.8% | 22.2% | 18 |
| AFTER, 22/22 videos | **37.5%** | **50.0%** | 24 |
| AFTER, cascade-low family only (ss3/ss423/ss42/ss441) | 46.7% | 40.0% | 15 |
| AFTER, high-pattern family only (ss50505/ss51/ss531/ss60/ss8040/ss90501) | 22.2% | 66.7% | 9 |

Both headline metrics improved substantially (§6.1 +9.7pp, §6.2 +27.8pp), but
neither clears the Step-3 gate thresholds — see §3. The IoU story is more
positive than the catch-count story: run-span truncation was IoU's dominant
error source (ss42_id_011 alone went from IoU≈0.05, `-0.5..38.7s` vs the real
`0..200s`, to IoU=1.000), so §6.2 moved the most. §6.1 moved less because the
duplicate-box clustering fix fully resolved 3 of its worst offenders
(`ss3_id_086`, `ss441_id_089`, partially `ss423_id_088`) while a comparable
number of *new or residual* catch-count misses remain (§5, §6).

## 3. Step-3 gate results

| gate | measured | target | result |
|---|---|---|---|
| ss3_id_086 total catches | 25 | 24±4 (20–28) | **PASS** |
| ss3_id_110 total catches | 17 | 9±3 (6–12) | **FAIL** |
| ss441_id_089 total catches / drops | 19 / 0 | 19±5 (14–24) AND ≤2 | **PASS** |
| ss423_id_088 total catches | 18 | 15±5 (10–20) | **PASS** |
| ss42_id_011 matched-run IoU | 1.000 | ≥0.9 | **PASS** |
| ss42_id_010 matched-run IoU | 1.000 | ≥0.9 | **PASS** |
| Clean-video constraint (±2 of oracle, all tasks) | see below | ±2 | **FAIL** (`ss50505_id_012`) |
| §6.1 ≥55% overall | 37.5% | ≥55% | **FAIL** |
| §6.1 ≥80% cascade-low family | 46.7% | ≥80% | **FAIL** |

**Clean-video constraint detail** — the six videos already at `\|Δ\|≤2` pre-fix
must stay there after every task: `ss441_id_013` Δ0, `ss42_id_010` Δ+2,
`ss42_id_011` Δ0, `ss3_id_079` Δ0, `ss3_id_987` Δ+2 all hold. `ss50505_id_012`
does not: offline 151 vs oracle 157, **Δ−6**, a regression from its pre-fix
Δ−2. Attribution in §6.

**Verdict: 5 of 9 gates PASS, 4 FAIL.** Per the plan's own escalation rule
("if the design can't reach it, STOP and report BLOCKED with per-video
attribution"), this is a **BLOCKED** result on the §6.1/clean-video/id_110
gates specifically — reported here with full attribution rather than iterated
on, since Task 4 is scoped to measurement, and Tasks 2/3 are already merged
and reviewed on this branch.

## 4. Mechanism 1 — duplicate-box clustering (Task 2, `c4b8d03`/`46136b2`/`9f9d303`)

**What it fixed.** A detector firing 2–3 overlapping boxes per ball in one
frame (`ss3_id_086` frame 194: 25 boxes forming exactly 3 spatial clusters)
let `extract_arcs`' EM assigner mint parallel "ghost" arcs, inflating catch
counts. `cluster_detections` (strict-lower-confidence, confidence-weighted,
per-frame, order-independent; default `merge_dist=0.023`) collapses these
before extraction ever sees them.

**Confirmed on re-validation:**
- `ss3_id_086`: 57 → 25 catches (oracle 24) — resolved.
- `ss441_id_089`: 59 catches/19 phantom drops → 19 catches/0 drops (oracle
  19, exact) — resolved.
- `ss423_id_088`: 44/6 drops → 18/0 drops (oracle 15) — much closer, not
  exact (see §6).
- Clean-density videos unaffected: `ss3_id_079` (6→6), `ss441_id_013`
  (137→136, essentially unchanged).

**Not fully closed:** `ss3_id_110` stays at 17 (oracle 9) — see §6.

## 5. Mechanism 2 — run-span truncation at unwitnessed misses (Task 3, `fce5e29`/`2ff89d4`)

**What it fixed.** `segment_runs` used to truncate a run's `end_t` at the
first *uncaught* arc, witnessed or not, while still counting the whole
group's catches. `is_floor_bound` (extracted from `detect_drops`, byte-
identical there) now gates that truncation to genuinely floor-bound misses
only; an extraction miss or occlusion no longer ends the reported span.

**Confirmed on re-validation:**
- `ss42_id_011`: run IoU went from **0.19** (`-0.5..38.7s` reported span vs
  the oracle's real `-0.5..201.3s`, 93 catches both sides) to **1.000** (exact
  span match). This is the single largest §6.2 mover in the whole set.
- `ss42_id_010`: IoU also lands at 1.000.
- `ss441_id_013`: 137 → 136 (oracle 136, exact) once clustering's small
  duplicate-driven noise is also accounted for.

**Unexpected beneficial side effect, `ss50505_id_093` (high pattern, nominally
out of scope — §7):** offline drops went from **35 → 0**, catches 6 → 38
(oracle 42). Apex-fragmented high throws were producing uncaught, non-floor-
bound arc endings that the old truncation logic misread as run-ending misses,
which then fed spurious drop counts; the corrected predicate stops
misclassifying them. Not a claimed fix for the fragmentation mechanism itself
(38 vs 42 still misses some throws), but a real, measured improvement that
happened to fall out of the span-semantics fix.

**Realtime inheritance verified live, not just structurally:** `end_t` feeds
the realtime engine's liveness check (`r.end_t` in `pipeline/realtime.py`).
Field replays (Task 3b report) confirm the fix rescues one specific
overlapping-cascade shape from the run-close debounce's job but does *not*
make the debounce redundant elsewhere (`RUN_CLOSE_DEBOUNCE_S` disabled on
`ss3_id_016` still flaps `runs_completed` 9→70; `af2` 2→3, catches/drops
unchanged in both — the debounce stays load-bearing, kept as-is).

## 6. New finding — clustering's cost on high-density/high-pattern videos

Task 2's own field spot-checks (`.superpowers/sdd/task-2-report.md`) only
covered the 5 cascade-family videos named in the plan brief. Task 4's full
22-video re-run surfaces a real, previously-undocumented **cost** of the
default `merge_dist=0.023` on three high-pattern videos, isolated by
re-running `analyze_detections` on the same saved detections at
`cluster_merge_dist=0.0` vs the shipped `0.023` (both on the current,
Task-3-inclusive worktree code, so the fix contributes zero in every ablation
below — each `merge_dist=0.0` result exactly reproduces its pre-hardening
baseline):

| video | merge=0.0 (= pre-fix exactly) | merge=0.023 (shipped) | oracle |
|---|---|---|---|
| ss50505_id_012 | 1 run / 155 catches | 1 run / **151** catches | 157 |
| ss531_id_005 | 1 run / 40 catches | **2 runs / 16** catches | 63 |
| ss531_id_989 | 1 run / 18 catches | **2 runs / 15** catches | 19 |
| ss531_id_988 | 1 run / 22 catches | 1 run / 17 catches | 21 |

`ss531_id_005` is the worst regression in the entire 22-video set
(catch Δ −47, the top row of the aggregate's worst-offenders list) and is
squarely a clustering artifact, not a span-fix artifact: at `merge_dist=0.0`
this video already matches its pre-fix baseline exactly (1/40/0), and
clustering alone turns it into a 2-run, 16-catch result. The mechanism is the
same cross-contamination Task 2's own docstring already names as an accepted
residual at genuine crossings ("a duplicate clone... can land closer to the
OTHER real ball's anchor than to its own true source") — on `ss531`'s denser,
faster pattern this removes enough real detection points that an arc which
used to bridge a gap now ends early, `is_floor_bound` reads that early ending
as a real miss, and `segment_runs`' temporal-gap grouping (unrelated to
either fix — see §7) splits what should be one run into two.

Not every high-pattern video is hurt: `ss51_id_163` (41→51, moving *toward*
oracle 58) and `ss90501_id_145` (11→9, moving toward oracle 5) both improve
with clustering on. The effect is mixed, not uniformly harmful — but the
`ss531` family and `ss50505_id_012` are real, measured regressions that
violate the plan's own clean-video constraint (§3) and were not caught by
Task 2's narrower spot-check scope. Flagged here per Global Constraints
("any count change... must be justified... both directions"); not fixed in
this task per its own scope (measurement/documentation only, single
path-scoped commit).

## 7. High-pattern scope boundary (unchanged, documented per plan)

High "5"-throws fragment at apex (detector loses the ball at the top of a
higher, slower arc), producing throws without matching catches. This plan
explicitly scopes to the 3-ball cascade family (ss3/ss423/ss42/ss441); the
`ss50505`/`ss51`/`ss531`/`ss60`/`ss8040`/`ss90501` families are documented,
not fixed. §6's `ss50505_id_093` result shows the span-truncation fix
*helps* around the edges of this boundary (eliminating spurious drops from
misclassified fragment endings) without closing it (still 38 vs 42 oracle
catches) — and §6 also shows the *opposite* can happen (clustering actively
hurting `ss531`). Net: the scope boundary holds, but it is not uniform in
direction. **Candidate future fix (unchanged from the plan):** extend
split-stitch across out-of-frame gaps so an apex-lost ball's ascent and
descent halves reassemble into one arc instead of two throws-without-catches.

## 8. Realtime envelope after the span fix

Live inherits every offline fix through `analyze_detections`
(`RealtimeAnalyzer` delegates to it — Task 2's integration point), confirmed
by the per-video table: `ss441_id_089` live drops 21→1, `ss423_id_088` live
drops 14→0, matching their offline improvements.

**Biggest remaining live deltas (live − offline, AFTER hardening):**

| video | offline | live | Δ |
|---|---|---|---|
| ss3_id_016 | 39 | 213 | **+174** |
| ss531_id_989 | 15 | 37 | +22 |
| ss423_id_017 | 120 | 138 | +18 |
| ss42_id_010 | 68 | 85 | +17 |
| ss90501_id_145 | 9 | 23 | +14 |
| ss531_id_014 | 29 | 41 | +12 |
| ss531_id_988 | 17 | 26 | +9 |
| ss50505_id_093 | 38 | 29 | −9 |
| ss8040_id_070 | 33 | 26 | −7 |

`ss3_id_016` dominates by an order of magnitude. This is *not* new: the
span fix leaves its run count and offline catch count byte-identical to
pre-fix (9 runs/39 catches both before and after — clean-density video,
clustering is a no-op, span fix doesn't change its grouping either). The
live gap's root cause was already isolated in the Plan 4 bench addendum
(`docs/superpowers/plans/2026-07-19-plan4-bench-findings.md` §8) and
re-confirmed REFUTED-for-cheap-fixes by the streaming-global-stats spike
(progress.md, task #39): windowed re-extraction re-derives junk-suppression
statistics from local context instead of the whole-video statistics offline
uses, minting extra physically-plausible arcs on dense multi-minute footage
that no downstream dedup/gating can repair. Plan 5 did not touch this path
(out of scope by design); the debounce (§5) keeps `runs_completed` from also
flapping, but the catch/drop over-count on this video is unchanged and
belongs to the pending Kalman/persistent-tracker contingency (task #39).

## 9. Extension — id_016 at imgsz 1280 through the hardened pipeline

Detections re-run at `imgsz=1280` (vs the default 640) were saved separately
(`ss3_id_016` detector recall investigation) and pushed through the same
hardened `analyze`/`live --detections` path. Oracle: 5 runs / 52 catches / 0
drops.

| | offline | live |
|---|---|---|
| pre-hardening, imgsz 640 (baseline) | 9 runs / 39 catches / 0 drops | 9 / 207 / 5 |
| pre-hardening, imgsz 1280 (unhardened, recall-recovery experiment) | 1 truncated run (`-1.0..34.3s` of 205s) / **134 catches** | not run |
| hardened, imgsz 640 | 9 / **39** / 0 (unchanged — clean density, no-op) | 9 / 213 / 6 |
| hardened, imgsz 1280 | 9 / **88** / 0 | 3 / **195** / 7 |
| oracle | 5 / 52 / 0 | — |

Detection density at 1280 (mean 3.31/frame, max 11/frame) is modestly higher
than at 640 (mean 3.01/frame, max 6/frame) but both are "clean" by the
clustering mechanism's own standard (median 3/frame either way; coverage
≥3 is 100% at both resolutions) — clustering is not the lever here.

**Verdict: hardened-1280 does NOT land near the oracle, and the id_016 story
is not "detector-input quality + hardening suffices."** Hardening fixed
exactly the truncation-count mismatch the extension hoped to isolate: the
span now covers the video's real `-1.0..204.5s` extent instead of
truncating to 34.3s while still claiming 134 catches. But once span-correct,
the *decomposition* is 9 runs against the oracle's 5 (the same
over-segmentation already present at 640, confirmed independent of both
Plan-5 fixes — see below), and the catch total overshoots to 88 (+69% vs
oracle) rather than undershooting to 39 (−25%) as at 640: better detector
recall trades an under-count problem for an over-count problem via the same
run-segmentation mechanism, it does not resolve it. Live is worse in
absolute terms at 1280 (195 catches, +122% over its own offline, +275% over
oracle) than the offline number alone would suggest, though the live/offline
*ratio* is smaller than at 640 (2.2x vs 5.5x) — consistent with, not
contradicting, the windowed-extraction root cause in §8 (more real
detections partially crowd out the spurious local-statistics arcs, but don't
eliminate them).

**Over-segmentation is not a Plan-5 artifact.** Comparing the oracle's 5 run
spans to the hardened-1280 offline's 9 (e.g. oracle `17.06–97.46s`, one
continuous 19-catch run, splits into three predicted runs ending at 49.18s,
83.68s, and continuing into a fourth) shows genuine multi-second coverage
gaps inside what the human annotator counted as one uninterrupted juggling
run — `segment_runs`' temporal-gap grouping (upstream of both the clustering
and floor-bound fixes) starts a new run at each gap regardless of `merge_dist`
or the floor-bound predicate. This afflicts id_016 at 640 identically (9 runs,
unchanged by either fix, per §1's table) — the imgsz experiment does not
create a new problem, it just fails to fix an old one. **What remains of the
live gap at 640 (and, worse, proportionally at 1280) is the same windowed
global-statistics mechanism from §8** — this extension is further evidence
that the Kalman/persistent-arc-identity contingency (task #39) is not made
unnecessary by higher-resolution detection input; if anything, richer input
without a matching extraction-side fix moves the failure mode from
under-count to over-count without closing the gap to oracle in either
direction.

## 10. Suite / lint state

No source or test files were modified for this task (measurement only). Full
suite `262 passed, 3 deselected`; `ruff check src tests`: `All checks passed!`
— confirmed on the worktree HEAD (`2ff89d4`) both before and after this
validation run.

## Concerns

- Four Step-3 gates fail (§3): `ss3_id_110` (residual duplicate-cluster
  mechanism, `.superpowers/sdd/task-2-report.md`'s own documented concern —
  larger, more variable duplicate offsets than `ss3_id_086`/`ss441_id_089`
  don't fit one global `merge_dist`), the clean-video constraint
  (`ss50505_id_012`, newly attributed to clustering in §6), and both §6.1
  fractions. Per the plan's escalation rule this is a BLOCKED result on those
  axes, reported with attribution rather than iterated on in this task.
- §6 is a genuinely new finding (not anticipated in the original Task 2
  brief's scope) with a real, currently unmitigated regression on
  `ss531_id_005` (Δ−47, the worst single-video result in the set) and
  `ss531_id_989` (2-way run split). Any future work tuning `cluster_merge_dist`
  needs to re-check this family, not just the cascade family the original
  spot-check covered.
- The imgsz-1280 extension (§9) is a one-video, one-resolution data point;
  it establishes direction (over-segmentation and windowed-extraction
  over-count dominate once truncation is fixed) but is not a sweep.

## Addendum: after parallel-arc dedup (Task 2b)

**Status: exploratory measurement, not TDD** (consistent with the rest of
this doc). Task 2b (`9dc1fc5`, `feat: parallel-arc dedup — trajectory-level
duplicate removal protects real crossings`) retired box-level clustering to
a no-op default (`cluster_merge_dist: 0.023 → 0.0`) and added arc-level
trajectory dedup (`arc_dedup_overlap_frac=0.75`, `arc_dedup_traj_tol=0.15`)
as the mechanism that discriminates real duplicate boxes from genuine ball
crossings — see `.superpowers/sdd/task-2b-report.md` for the full tuning
history (two failed rounds against the sim suite before landing on raising
`overlap_frac` instead of capping `traj_tol`). This addendum re-runs the
same saved-detection machinery (`outputs/meschke-val/rerun_validation_2b.py`
+ `rerun_aggregate_2b.py`) at this new HEAD, writing to
`outputs/meschke-val/post-hardening-2b/`; the Tasks-1-3 artifacts in
`post-hardening/` are untouched.

### Final per-video table (`runs/catches/drops`)

| video | oracle | offline (2b) | live (2b) |
|---|---|---|---|
| ss3_id_016 | 5/52/0 | 9/39/0 | 9/207/5 |
| ss3_id_079 | 1/6/0 | 1/6/0 | 1/6/0 |
| ss3_id_086 | 1/24/0 | 1/27/0 | 1/28/0 |
| ss3_id_110 | 1/9/0 | 1/21/0 | 1/21/0 |
| ss3_id_987 | 1/20/0 | 1/21/0 | 1/22/0 |
| ss423_id_007 | 1/73/0 | 1/82/0 | 3/83/5 |
| ss423_id_017 | 1/130/0 | 1/121/0 | 1/143/8 |
| ss423_id_088 | 1/15/0 | 1/17/0 | 1/17/1 |
| ss42_id_010 | 1/66/0 | 1/65/0 | 1/86/0 |
| ss42_id_011 | 1/93/0 | 1/93/0 | 1/94/0 |
| ss441_id_013 | 1/136/0 | 1/137/0 | 1/136/0 |
| ss441_id_089 | 1/19/0 | 1/17/1 | 1/17/1 |
| ss50505_id_012 | 1/157/0 | 1/155/0 | 1/148/2 |
| ss50505_id_093 | 1/42/0 | **1/3/2** | 1/31/37 |
| ss51_id_163 | 1/58/0 | 1/40/0 | 1/47/0 |
| ss531_id_005 | 1/63/0 | 1/40/0 | 1/64/0 |
| ss531_id_014 | 1/29/0 | 1/25/0 | 2/36/2 |
| ss531_id_988 | 1/21/0 | 1/22/0 | 1/43/1 |
| ss531_id_989 | 1/19/0 | 1/18/0 | 3/29/1 |
| ss60_id_151 | 1/26/0 | 1/29/0 | 1/32/0 |
| ss8040_id_070 | 1/33/0 | 1/36/0 | 1/23/0 |
| ss90501_id_145 | 1/5/0 | 1/11/0 | 3/15/1 |

### Spec §6 metrics — three states

| state | §6.1 (≤1 catch Δ) | §6.2 (IoU ≥0.9) | matched runs | total offline catches (Δ vs oracle 1096) |
|---|---|---|---|---|
| pre-Plan-5 (official, 20/22 scorable) | 27.8% | 22.2% | 18 | 1088 on 1056-scope oracle (+3.0%) |
| post-Tasks-1-3 (22/22) | 37.5% | 50.0% | 24 | 1030 (−6.0%) |
| **post-Task-2b (22/22)** | **28.0%** | **48.0%** | 25 | **1025 (−6.5%)** |

Cascade-low family (ss3/ss423/ss42/ss441), §6.1 only: 46.7% (post-1-3) →
**31.2%** (post-2b, n=16). High-pattern family §6.1: 22.2% → 22.2%
(unchanged); §6.2: 66.7% → 77.8% (improved).

**This is not a uniform win.** Task 2b's own scope was 9 specific field
gates (task-2b-report.md §4), and it hits 8/9 of those. But the broader
spec-§6 aggregate across all 22 videos' matched runs is *worse* than the
post-Tasks-1-3 state on both metrics, and worse than the pre-Plan-5 baseline
on §6.1 specifically. Two things drive this, both measured directly by
diffing the full per-match delta lists:
1. **`ss50505_id_093` collapsed further** (§ below) — a single video
   contributing a matched-run delta of 39, the largest in the set.
2. **Several previously-exact/near-exact cascade videos drifted away from
   zero** as a side effect of the retune, even while none of them are
   Task 2b's own required gates: `ss3_id_086` delta 1→3 (loosened gate is
   `≤30`; 27 still passes it), `ss531_id_014` delta 0→4, `ss441_id_089`
   delta 0→2 (drops 0→1), `ss8040_id_070` delta 0→3. None individually
   large, but the ≤1 binary threshold is unforgiving of small drift.

Offsetting this, several previously badly-wrong videos improved
substantially in absolute terms even though the binary metric doesn't
credit it: `ss531_id_005` 52→23 catch-delta, `ss531_id_988`/`989` roughly
4-10→1 each. The **clean-video constraint is now fully resolved**:
`ss50505_id_012` (the post-Tasks-1-3 regression flagged in §6/§3 above) is
back to Δ−2 (155 vs oracle 157), and all six originally-clean videos
(`ss441_id_013`, `ss42_id_010/011`, `ss3_id_079/987`, `ss50505_id_012`) sit
within ±2 again.

### Step-3 gates, updated

| gate | post-1-3 | post-2b | target | result |
|---|---|---|---|---|
| ss3_id_086 | 25 | 27 | 24±4 | PASS |
| ss3_id_110 | 17 | **21** | 9±3 | **FAIL, worse** |
| ss441_id_089 catches/drops | 19/0 | 17/1 | 19±5 & ≤2 | PASS |
| ss423_id_088 | 18 | 17 | 15±5 | PASS |
| ss42_id_011 IoU | 1.000 | 1.000 | ≥0.9 | PASS |
| ss42_id_010 IoU | 1.000 | 1.000 | ≥0.9 | PASS |
| Clean-video constraint | FAIL (id_012 Δ−6) | **PASS** (all six ±2) | ±2 | **RESOLVED** |
| §6.1 ≥55% overall | 37.5% FAIL | 28.0% FAIL | ≥55% | FAIL, worse |
| §6.1 ≥80% cascade-low | 46.7% FAIL | 31.2% FAIL | ≥80% | FAIL, worse |

**6/9 PASS (up from 5/9)** — the clean-video constraint is fixed, but
`ss3_id_110` moved further from its target (17→21) and both §6.1 fractions
regressed in absolute terms even though neither newly passes nor newly
fails relative to their gate thresholds.

### ss3_id_110 — structural frontier (from task-2b-report.md §5)

Confirmed by direct measurement, not just asserted: `ss3_id_110`'s own
duplicate-storm arc pairs have mean positional separation **≥0.23**, which
is *above* `ss531_id_989`'s genuine-crossing floor of 0.164. Any single
global `merge_dist`/`traj_tol` that collapses `ss3_id_110`'s duplicates down
to its ≤18 gate necessarily also collapses `ss531_id_989`'s real crossings
(breaking it below 17) or pushes `ss50505_id_012` outside 153–161 — verified
by two exhaustive sweeps (0.001 and 0.0005 step, 0.000–0.023). This is the
same "larger, more variable duplicate offsets" residual flagged in
`task-2-report.md` and the main body of this doc (§6) — now confirmed
structural (no achievable single-knob value closes it) rather than merely
untuned. Closing it needs a mechanism beyond both box-level clustering and
arc-trajectory dedup: per-video/adaptive thresholding, or a discriminator
richer than mean positional separation.

### ss50505_id_093 — lost its accidental clustering benefit

This high-pattern video (out of both Task 2's and Task 2b's gate scope) is
the clearest illustration of §6/§7's "mixed, not uniform" finding, now
resolved in the negative direction. Its improvement in the main body of this
doc (drops 35→0, catches 6→38 vs oracle 42, §5) turns out to have depended on
**box-level clustering's pre-extraction cleanup**, not the span-truncation
fix credited there: its raw detections are messy enough (135 post-extraction
arcs) that retiring `cluster_merge_dist` to 0.0 removes that cleanup, and
arc-level dedup cannot recover the same result after the fact (dedup only
ever discards one of two already-extracted, already-damaged arcs — it can't
repair points that never got merged before extraction ran). Result: **3
catches / 2 drops** (down from 38/0), the single worst matched-run delta in
the 22-video set. Documented as a known collateral cost in
`task-2b-report.md` §3/§8, not fixed there (reopening `cluster_merge_dist`
would reopen the `ss531`/`ss50505_id_012` regressions Task 2b's gates are
scored on) — flagged here for the same reason. Candidate future direction
(per both reports): adaptive, per-video density-based `merge_dist` selection
that runs box-clustering only where the raw detection stream needs it,
instead of one global default serving both regimes.

### Live envelope, updated

The live/offline gap structure is essentially unchanged by Task 2b:
`ss3_id_016` still dominates by an order of magnitude (offline 39, live 207,
**+168** — the windowed-extraction global-statistics mechanism from §8 is
untouched by either box- or arc-level dedup, both of which run upstream of
the realtime/offline divergence). The next-largest deltas shuffle slightly
(`ss42_id_010` +21, `ss531_id_988` +21, `ss423_id_017` +22, `ss531_id_989`
+11 across a new 3-way run split) but stay in the same rough band as
post-Tasks-1-3. `ss50505_id_093`'s live number (31/37 drops) is no longer
comparable to its former self — it inherited the offline collapse from the
retuned defaults rather than a live-specific regression.

