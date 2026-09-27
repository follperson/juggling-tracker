# Duplicate suppression: tuning record

This records the reasoning behind the current duplicate-box and parallel-arc
thresholds. These settings were tuned on the same 22 Meschke videos used to
compare them; they are not independent holdout results. The current default
remains `cluster_merge_dist=0.012`, with arc dedup overlap `0.75` and trajectory
tolerance `0.15`.

The [committed validation findings](superpowers/plans/2026-08-03-meschke-validation-findings.md)
contain the final per-video results and corrected denominators. References below
to `.superpowers/` describe historical local notes; use the committed findings
for portable evidence.

## Historical parameter rationale

Per-frame duplicate-box clustering radius (see detect/cluster.py's
module docstring for the full measured justification): a detector
firing 2-3 overlapping boxes for one physical ball mints parallel
"ghost" arcs downstream that inflate catch counts. Applied BEFORE
filter_static_detections/extract_arcs so every downstream stage --
offline and the realtime window path alike -- sees clustered
detections. 0.0 disables clustering entirely (identity passthrough,
see cluster_detections).

0.012, not the earlier 0.023 (Plan 5 task 2) or task 2b's initial 0.0:
box-level clustering at 0.023 could not tell a duplicate echo from a
genuine crossing on REAL footage (real crossing balls almost always
differ in confidence, so the strict-lower-confidence guard doesn't
protect them the way it protects the sim's exactly-tied detections)
-- turning merge_dist up far enough to collapse duplicate storms
regressed real crossings: ss531_id_005 40->16 catches (oracle 63),
ss531_id_989 18->15 (oracle 19), ss50505_id_012 155->151 (oracle 157).
Once arc_dedup_traj_tol below took over most of the duplicate-storm
job at the ARC level, task 2b's own field-gate sweep found 0.0
(clustering fully retired) passed 8/9 required gates -- but a
coordinator-directed full 22-video re-aggregation (task-2b-report.md
§9) showed 0.0 costs spec §6.1 (per-run catch accuracy, the primary
metric) versus the old 0.023 state: 37.5%->28.0% overall,
46.7%->31.2% on the cascade-low family. A combo sweep of
cluster_merge_dist in {0.010, 0.012, 0.015} WITH arc dedup active
(0.75/0.15, unchanged) across all 22 videos, offline only, produced:

  merge_dist | 9-gate | clean(5) | cascade-low §6.1 | overall §6.1 | catch Δ
  0.0        | 8/9 (fails ss3_id_110)         | 5/5 | 31.2% | 28.0% | -71
  0.010      | 5/9                            | 5/5 | 43.8% | 26.9% | -84
  0.012      | 8/9 (fails ss531_id_989: 16<17) | 5/5 | 53.3% | 39.1% | -55
  0.015      | 7/9 (fails ss531_id_989, ss50505_id_012) | 4/5 | 40.0% | 29.2% | -60

ADJUDICATION (controller, task-2b-report.md §10, recorded verbatim):
"the ss531_id_989 >=17 gate was task-dispatch scaffolding, not a
plan-binding constraint -- the plan's Global Constraints clean-video
list (id_013, id_010/011, id_079, id_987, id_012) passes 5/5 at
0.012, and the spec's primary metric §6.1 governs: 0.012 more than
doubles cascade-low §6.1 (31.2->53.3%) and lifts overall §6.1 to
39.1% (best of all four states) at the documented cost of id_989
landing at 16 vs oracle 19 (delta 3, was delta 1)." Shipped value is
therefore 0.012, not 0.0 or 0.023.
[Editorial notes, final review, not part of the verbatim quote above:
(1) the plan's clean-video list actually names SIX videos
(id_013, id_010, id_011, id_079, id_987, id_012 -- id_010/011 are two
entries, not one) and the eval script's "clean(5)" checker above only
tested five of them (ss50505_id_012 is verified separately by its own
153-161 gate in the same script); at 0.012 id_012 lands at 156 vs
oracle 157 (|delta|=1), so the true result is 6/6, stronger than the
quoted 5/5, not weaker. (2) "more than doubles" is 53.3%/31.2% =
1.7x, not >=2x -- doubling would need >=62.4%; in matched-run counts
it is 8/15 vs 5/16. Both corrections leave the adjudication's
conclusion unchanged; see the findings doc PS for the full
k/n-annotated progression.]

OVERFITTING CAVEAT (controller-directed, record verbatim): this
constant is now tuned against the 22-video suite itself -- there is
no held-out set at this granularity. Generalization is deferred to
the spec's never-trained-on holdout set; treat 0.012 as measured on
its own training data, not as validated out-of-sample.

MARGIN CAVEAT (final review, quantified): the 0.012-vs-neighbor
margins on this suite are 2-4 individual matched-run flips (0.012 vs
0.0: +4/-2, McNemar exact p~=0.69; vs 0.010: +3/-1, p~=0.63; vs
0.015: +3/-1) out of 23-26 matched runs -- statistically
indistinguishable, and overall §6.1 is non-monotonic across
{0.010, 0.012, 0.015} (26.9%->39.1%->29.2%), an isolated spike at the
adopted value rather than a monotonic trend. Read 0.012 as the best
TESTED point on this suite, not a robust optimum, on top of (not
instead of) the overfitting caveat above.

KNOWN COST (measured, unchanged by this retune): ss50505_id_093
(high-pattern family, not a required gate) relied on box-level
clustering's PRE-extraction cleanup at the OLD 0.023 -- 0.012 is
still far short of that (raw detection stream is messy enough, 135
post-extraction arcs, that a small merge_dist and arc-level dedup
together do not recover 0.023's 38 catches, oracle 42). The sweep
table and adjudication quote above are the committed record of this
history; see .superpowers/sdd/task-2b-report.md §§3,8,9,10 (untracked
local file -- not resolvable from a fresh clone) only for additional
per-round trade-off detail not already inlined here.
`cluster_merge_dist: float = 0.012`
Arc-level parallel-arc dedup (Plan 5 task 2b, arcs/extract.py's
dedup_parallel_arcs): applied AFTER extract_arcs, BEFORE the
event-derivation tail, offline and realtime alike (same integration
point cluster_merge_dist used). Field motivation: box-level
clustering can't tell a duplicate echo from a genuine crossing on
real footage; full arc trajectories can (see dedup_parallel_arcs's
own docstring) -- a duplicate storm's arcs trace nearly the same
parabola over their WHOLE shared window, while a genuine crossing's
arcs only agree briefly.

traj_tol=0.15, overlap_frac=0.75 (both widened from an initial
0.02/0.5 guess): the field-video evidence alone (ss3_id_086/
ss441_id_089's duplicate pairs top out at mean-diff ~0.097;
ss531_id_989's genuine crossing floor is 0.164 -- the SMALLEST
genuine-crossing pair measured, from a single video; not a population
floor, and the base rate of tighter real crossings elsewhere is
unmeasured) suggested traj_tol could go as high as ~0.10-0.16 with
overlap_frac=0.5. That FAILED the full test suite: two existing sim
fixtures (test_low_gravity_framing_recovers_events's low-g cascade,
whose alternating-hand throws overlap and trace mean-diff as low as
0.093 at overlap_frac=0.5; and
test_left_edge_guard_prevents_phantom_catches_from_truncated_refit's
noisy fixture, mean-diff 0.029 at overlap_frac=0.70) are genuinely
DIFFERENT arcs that a traj_tol wide enough for the field videos would
incorrectly collapse. Raising overlap_frac to 0.75 excludes both
conflicting sim pairs outright (their own overlap_frac, 0.59 and 0.70,
sits below the new threshold) regardless of traj_tol, which reopens
room to raise traj_tol to 0.15 -- verified safe up to 0.18 against the
full test suite, but 0.18 would exceed the measured 0.164 field
crossing floor above, so 0.15 keeps a (thin, n=1-measurement) margin
over it rather than being a suite-safe value with headroom to spare;
do not raise traj_tol toward 0.18 without re-measuring crossing floors
on more footage. 0.15 still collapses the bulk of ss3_id_086/
ss441_id_089's duplicate pairs (most of which sit above 0.75
overlap_frac; see .superpowers/sdd/task-2b-report.md, an untracked
local file, for the measured pair distributions on both sides of this
trade-off -- not resolvable from a fresh clone, but the governing
values are inlined above and in test_extract.py's
test_dedup_parallel_arcs_keeps_both_at_shipped_crossing_floor, which
pins this exact floor against live AnalyzeConfig defaults).

ss3_id_110 is a documented, unresolved exception: its own duplicate-
storm arc pairs measure mean-diff >=0.23 (ABOVE ss531_id_989's
genuine-crossing floor of 0.164), so no traj_tol can collapse them
without also unsafely collapsing real crossings elsewhere -- and
exhaustive merge_dist sweeps (0.001 and 0.0005 steps, 0.000-0.023)
confirm no merge_dist recovers it without breaking ss531_id_989 or
ss50505_id_012 instead. BLOCKED on this one video; the committed
findings doc's "ss3_id_110 -- structural frontier" section
(docs/superpowers/plans/2026-08-03-meschke-validation-findings.md)
carries this same conclusion. .superpowers/sdd/task-2b-report.md §5
has the full per-sweep measurements but is an untracked local file,
not resolvable from a fresh clone.
`arc_dedup_overlap_frac: float = ARC_DEDUP_OVERLAP_FRAC`
`arc_dedup_traj_tol: float = ARC_DEDUP_TRAJ_TOL`

