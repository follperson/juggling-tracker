# Turn 4 — Missed-Throw Attribution Diagnosis

Exploratory diagnostic (no repo code changes). Goal: for the two field-report holdouts
(af1.mov, PXL_20260716_191702756), figure out exactly which pipeline stage loses each
real ball flight that never becomes a `ThrowEvent`.

## Inputs used

- af1: `outputs/turn3/analyze-v3/af1/detections.jsonl` (v3 weights, imgsz 640, stride 1 — already present, not re-run).
- Outdoor holdout: `outputs/turn3/analyze-v3/PXL_20260716_191702756-1280/detections.jsonl` (v3 weights, imgsz **1280**, stride 1 — already present from turn 3's 1280 probe; no re-run needed, confirmed better operating point per turn-2c/turn-3 notes: 368 detections vs 335 at 640).
- Default pipeline config = `AnalyzeConfig()` defaults = `extract_arcs` defaults (`min_points=6, min_duration=0.15, link_max_dt=0.12, link_max_dist=0.08, resid_tol=0.02, g_range=(0.1,8)`), `derive_events(min_apex_above=0.02, floor_margin=0.10)`.
- Loose candidate-inventory config: `min_points=4, min_duration=0.08, link_max_dt=0.25, link_max_dist=0.16, resid_tol=0.03, g_range=(0.05,8)`.

## Method (and a correction to it, found mid-analysis)

The prescribed method builds a "ground-truth-ish" candidate-flight inventory with a
LOOSE `extract_arcs` config and diffs it against what the DEFAULT config extracts and
throws. **This diff has two failure modes of its own**, both discovered by the
eyeball step and confirmed analytically before being trusted:

1. **Over-merge chimeras (false-positive "misses").** In dense multi-ball regions,
   loose's wider `resid_tol`/`link_max_dt` can bridge across a real catch→re-throw
   transition (or a same-time crossing of two different balls) and fit one bogus
   parabola across it. That bogus "candidate" then has no matching default arc —
   looking like a miss — when in fact **default already extracted and threw both
   real flights on either side of it.** Detected by: candidate overlaps ≥2
   *already-thrown* default arcs. Confirmed by eyeball on af1 (see below).
2. **Over-merge-then-prune blind spot (false-negative — misses invisible to the diff).**
   The same over-merging can instead produce a fused arc that's *bad enough* to get
   rejected by `_prune`/`_gravity_prune` entirely, so the loose inventory has
   **nothing** in that time span — even though the default config, with its tighter
   gates, correctly extracts two separate, real, short arcs there. These misses are
   invisible to the candidate-vs-default diff and only surface via a **direct census
   of every default arc's throw status** (arc extracted but `derive_events` never
   emitted a throw for it). Confirmed by eyeball on the outdoor clip's t=2.7–3.6s
   region.

Both corrections were folded into the final attribution (script:
`turn4_final.py` in the session scratchpad), plus the required binary-search gate
relaxation and the eyeball pass on 3 candidates (2 frames each, cropped to the
expected position, read visually).

## Per-video attribution table

**af1.mov** (in-domain, own footage, 30fps, 1188 filtered detections, hand_line_y=0.664)

| Category | Count | % of 60 real flights |
|---|---|---|
| Found (default extracts + throws) | 59 | 98.3% |
| **Genuine miss** | **1** | **1.7%** |
| Diff-method false positives (chimera artifacts, NOT real misses) | 2 | — (excluded from denominator) |

The 1 genuine af1 miss: **t=16.80–16.97s** (observed span 0.17s; the true full flight
is longer — see below). All 59 *extracted* default arcs got a throw (0% event-layer
loss on af1) — every loss on this video is extraction-side.

**PXL_20260716_191702756 (outdoor holdout, @1280, 368 filtered detections, hand_line_y=0.356)**

| Category | Count | % of 12 real flights |
|---|---|---|
| Found (default extracts + throws) | 7 | 58.3% |
| **Genuine miss — extraction (linker gate)** | 3 | 25.0% |
| **Genuine miss — event layer (apex guard)** | 2 | 16.7% |
| Diff-method false positives | 0 | — |

Total real flights on the outdoor holdout = 12 (10 surfaced by the loose diff + 2
more found only by the direct default-arc census). **Overall miss rate 5/12 = 41.7%**
— a 25x higher loss rate than af1's 1.7%, which lines up with the field report calling
out outdoor clips specifically.

## Gate / predicate ranking (evidence)

| Rank | Mechanism | Videos hit | Count | Evidence |
|---|---|---|---|---|
| **1** | **`link_max_dt` too tight (0.12s)** | PXL (outdoor) | 3/5 misses | Single-gate binary search: relaxing *only* `link_max_dt` (→0.25) recovers all 3 (t=5.74-6.20, 8.03-8.40, 8.78-9.07). Root cause confirmed at the frame level: the raw detection stream around each miss has a **3–4 consecutive dropped-frame gap** (0.125s / 0.166s / 0.166s — note 0.166s ≈ 4 frames at PXL's 24.04fps), exceeding the 0.12s budget. af1 (30fps, in-domain) never showed a gap this large in its one genuine-miss window (max local gap 0.033s = 1 frame). This is a **detector-recall problem that surfaces as a linker-gate failure**: outdoor/domain-shifted footage doesn't drop so many points that `min_points` fails, it clusters the drops into multi-frame gaps that `link_max_dt` can't bridge. |
| **2** | **Apex guard (`min_apex_above=0.02`) on arcs the loose diff never saw** | PXL | 2/5 misses | Direct default-arc census (not the loose diff — see method correction #2): 2 default arcs at t=2.74-3.16 and t=3.20-3.58 are cleanly extracted (12 and 13 points, rmse 0.001–0.003) but fail the apex guard by a wide margin (`apex_gap` = hand_line − apex_y = **−0.141** and **−0.086**, vs. the required `> +0.02`) — these look like genuinely shallow warm-up tosses at the very start of the clip, not measurement artifacts. |
| 3 | Boundary/EM-iteration sensitivity (min_points **and** min_duration jointly) | af1 | 1/1 miss | The 1 genuine af1 miss recovers *only* when `min_points` (6→4) **and** `min_duration` (0.15→0.08) are both relaxed together — no single gate alone recovers it (verified: link_max_dt, link_max_dist, resid_tol, min_points, min_duration, g_range each tried alone, none work). The surviving fragment sits exactly on the default boundary (6 points, 0.167s duration — both individually clear the default thresholds), so this reads as an EM-iteration ordering artifact (a marginal seed pruned mid-iteration before its membership count restabilizes) rather than one clean gate. Lowest priority: n=1, single video. |
| — (ruled out) | `resid_tol` (0.02→0.03) | af1 | flagged 2, both **false positives** | Binary search initially flagged 2 af1 "misses" as resid_tol-recoverable. Eyeballing + arc-census cross-check showed both are chimera artifacts (loosened resid_tol lets the fitter bridge across two already-thrown real arcs). **Loosening resid_tol would not recover any real throw here** and would likely *worsen* the known Meschke ss3_id_016 over-counting problem (39 vs 23) — same knob, opposite failure mode. Do not tune this in response to this diagnostic. |

## Low-throw vs. high-throw loss rate

Per the brief's threshold (flight duration < 0.4s = "low"):

| | Low throws (n) | Low miss rate | High throws (n) | High miss rate |
|---|---|---|---|---|
| af1 | 3 | 33.3% (1/3) | 57 | 0.0% (0/57) |
| PXL (outdoor) | 8 | 37.5% (3/8) | 4 | 50.0% (2/4) |
| **Combined** | **11** | **36.4% (4/11)** | **61** | **3.3% (2/61)** |

Combined, low throws are lost at **~11x** the rate of high throws — but the PXL
per-video breakdown shows the mechanism isn't "apex guard specifically targets short
throws": on the outdoor clip, high-duration throws actually miss at a *higher* raw
rate (50%, n=4) than low ones (37.5%, n=8) — both duration bands get hit by the same
`link_max_dt` gap problem there. The combined number is dominated by af1's 57
uniformly-easy high-duration throws diluting the denominator. **Caveat on "duration":**
the one genuine af1 miss is labeled "low" (0.17s observed) only because the flight
got truncated by the same gap that causes the miss — the fitted apex time
(`apex_t=17.054`) falls *outside* the observed window (16.80–16.97), meaning the true
full flight is a normal ~0.5s throw with a normal apex height (apex_gap ≈ +0.23,
nowhere near the guard). Observed-span duration is not a reliable proxy for "was this
throw physically low" whenever the flight got cut short by a gap — a duration-vs-apex
distinction future analyses should keep separate.

## Sanity-visual (3 candidates, 2 frames each, eyeballed)

1. **af1 t=16.80–16.97 (genuine miss).** Frames 504 (t=16.80) and 508 (t=16.94),
   cropped to the predicted position. Both show a real green ball — frame 504 shows
   *two* balls simultaneously (the descending/catching ball + the newly-thrown
   ascending one), frame 508 shows the ascending ball alone near the face, confirming
   a real flight in progress. **Confirmed real miss**, and the 2-ball proximity in
   frame 504 is exactly the mechanism that starves the linker of clean points for the
   new throw right as the other ball is being caught nearby.
2. **af1 t=12.77–13.27 (flagged, later ruled a diff-artifact).** Frames 385 (t=12.84)
   and 397 (t=13.24). Both frames show **two real balls in flight simultaneously**
   (af1 is a multi-ball cascade) at different frame positions — confirming this is a
   dense-crossing region, consistent with the resid_tol-relaxed re-extraction showing
   default already independently throws the arc before (ends 12.94) and after (starts
   13.20) this window; the "candidate" here is the loose config's own chimera, not a
   third real flight.
3. **PXL t=8.78–9.07 (genuine miss, outdoor, single ball).** Frames 211 (t=8.78) and
   213 (t=8.86), cropped to the predicted position. Both show a clear, real dark ball
   against open sky — this window has **zero** overlapping default arc of any kind
   (default produces literally nothing there). **Confirmed real miss**, the cleanest
   case of the linker-gate loss mechanism (#1 above).

Frame crops (session scratchpad, not committed):
`c1_af1_f504_t16802.png`, `c1_af1_f508_t16935.png`, `c2_af1_f385_t12835.png`,
`c2_af1_f397_t13235.png`, `c3_pxl_f211_t8776.png`, `c3_pxl_f213_t8859.png`.

## Surprise / unifying observation

The same `resid_tol`/merge machinery that would need to loosen to "recover" af1's two
falsely-flagged misses is the **same knob already implicated in the over-counting
problem** on Meschke ss3_id_016 (39 vs 23 catches) noted in the turn-3/brief context.
This diagnostic independently reproduces the mechanism on af1's own dense 2-ball
regions: loosening the split/merge tolerance trades false-negatives (fixed, but they
weren't real misses anyway) for false-positives (real risk, already documented
elsewhere). **Recall and precision on dense multi-ball footage are in tension through
this one parameter** — a periodicity/self-consistency validator (already flagged as
unbuilt spec §4 work in `progress.md`) is the principled fix for *both* problems at
once, rather than hand-tuning `resid_tol` in either direction.

A second, smaller surprise: two of PXL's five losses (apex-guard, rank #2) were
**invisible to the prescribed loose-candidate-diff method entirely** — they only
appeared once every default arc's throw status was directly audited. Any future
recall diagnostic on this pipeline should default to a **direct default-arc census**
first, and treat the loose-candidate diff as a secondary/extraction-only check, not
the other way around — the loose config's own merge+prune step is not a reliable
superset of what default finds.

## Recommended fixes, ranked by expected recall gain

1. **Widen `link_max_dt` from 0.12s to ~0.17–0.20s.** Directly recovers 3/5 (60%) of
   the outdoor holdout's losses — the single highest-value, single-parameter change
   found in this diagnostic. Evidence: single-gate binary search + measured gap sizes
   (0.125s/0.166s/0.166s, i.e. 3–4 dropped frames at PXL's 24.04fps) sitting just
   above the current 0.12s ceiling. Expected to generalize to any footage with
   similar per-frame detection-confidence variance (domain-shifted/outdoor/lower-light
   clips), not just this one video. **Verify against af1 and the Meschke
   holdouts before landing** — af1's dense 2-ball crossings mean a looser dt budget
   could increase chimera risk there too (rank-3 miss's own fix needs
   `link_max_dist`/`min_points`/`min_duration` together, not `link_max_dt`, so this
   change is not expected to regress af1, but should be checked empirically, not
   assumed).

2. **Make the apex guard "witnessed-apex aware" before touching `min_apex_above`.**
   Currently `derive_events` evaluates `apex_y()` from the fitted parabola even when
   `apex_t` falls outside the arc's own observed `[t_start, t_end]` span (extrapolated
   apex) — the same "witnessed" concept `CATCH_EXTRAPOLATION_MARGIN` already applies
   to catches is missing on the throw/apex side. This wouldn't change the 2 genuinely-shallow
   PXL misses (their apex_gap is large and negative — real low tosses, guard working
   as intended) but would prevent near-boundary false rejects on gap-truncated arcs
   (the af1 rank-3 miss's own apex was extrapolated 0.19s past its last real point).
   Lower quantified yield than #1 today, but compounds with #1: widening `link_max_dt`
   will itself convert more extrapolated-apex arcs into witnessed ones, so do #1 first
   and re-measure before deciding whether #2 is still needed.

3. **Do not loosen `resid_tol` in response to this diagnostic.** Explicitly
   ruled out (see ranking table and the "surprise" section) — the 2 af1 misses it
   would appear to fix are diff-method false positives, not real throws, and
   loosening it risks worsening the known over-count problem. Flagged here as a
   fix to actively avoid, since a naive reading of the raw binary-search output
   (before the eyeball step) would have recommended it.

4. *(Lowest priority, n=1, deferred)* af1's one genuine miss needs `min_points`
   and `min_duration` relaxed together, or better, a periodicity-aware seed-retention
   rule so a marginal seed isn't pruned mid-EM-iteration before its membership count
   stabilizes. Not worth a standalone change for one flight; revisit if the
   periodicity/self-consistency validator (surprise section) gets built, since that's
   the more principled fix for this exact "should this marginal boundary-case arc
   survive" question.
