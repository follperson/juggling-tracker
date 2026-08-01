"""Realtime event engine: sliding-window re-analysis with freeze-horizon
confirmation.

CONSCIOUS SPEC DEVIATION (documented in the plan header): spec §3 called for
a bespoke online two-mode Kalman tracker; measured extraction cost (~0.01s
per 8s window) makes re-running the proven offline core affordable at frame
rate. This buys MEASURED, not by-construction, offline parity: the sliding
window re-derives events/runs/drops from scratch every cadence tick, and
that windowed re-derivation is a real, quantified source of divergence from
a single whole-video offline pass (see docs/superpowers/plans/
2026-07-19-plan4-bench-findings.md §7-8 for the measured envelope: agreement
is close on short/typical clips and degrades as a run's own length
approaches window_s). The Kalman design remains the contingency if that
gap, rather than raw compute cost, forces the issue.

This module must stay cv2-free: it consumes Detections, not frames.
"""
from __future__ import annotations

import time

from pydantic import BaseModel, Field, model_validator

from juggletrack.analyze import AnalyzeConfig, _events_from_arcs, analyze_detections
from juggletrack.events.catches import derive_events
from juggletrack.types import Arc, Detection


class RealtimeConfig(BaseModel):
    window_s: float = 8.0
    cadence_frames: int = 3
    freeze_s: float = 1.5
    drop_freeze_s: float = 2.5
    event_match_tol: float = 0.15
    # Left-edge guard (symmetric with the freeze horizon's right-edge guard):
    # an arc whose t_start falls within `edge_pad` of the window's trailing
    # cut (`now - window_s`) had its early history trimmed by the sliding
    # buffer before this analysis cycle ever ran -- its fitted coefficients
    # (and anything derived from it: crossing times, "first missed catch",
    # floor-descent signals) reflect a tail-only fit, not the ball's real
    # trajectory. See _analyze's left-edge guard for the mechanism this
    # fixes (docs/superpowers/plans/2026-07-19-plan4-bench-findings.md §3,
    # .superpowers/sdd/task-5b-report.md).
    edge_pad: float = 0.5
    # Hand-line EMA smoothing factor. MEASURED (not assumed) before adding:
    # a raw per-cycle hand-line re-estimate can swing >0.1 in normalized y
    # between consecutive 0.1s cadence ticks on real, arc-sparse footage
    # (trace evidence in task-5b-report.md), which shifts every
    # hand-line-relative computation (crossings, catch/drop witnessing)
    # enough to manufacture duplicate event confirmations. Smoothing alone
    # measured a ~14% reduction in duplicate catches on the ss3_id_016
    # field-gate replay -- real but partial; see the report for the
    # remaining, larger, out-of-scope mechanism (local-window vs
    # whole-video arc-segmentation instability) this does not reach.
    hand_line_ema_alpha: float = 0.2
    analyze: AnalyzeConfig = Field(default_factory=AnalyzeConfig)

    # E4: nothing above enforced the invariant the left-edge guard's own
    # soundness argument depends on (see _analyze's comment) -- that a real
    # event is always confirmable (t <= now - freeze) while its arc is still
    # non-truncated (t_start >= now - window_s + edge_pad). Violate it (e.g.
    # window_s too small for freeze_s/drop_freeze_s) and every event's arc is
    # truncated before its confirmation horizon ever arrives: events are
    # silently discarded every cycle, with no error or warning -- reviewer's
    # repro P2 measured RealtimeConfig(window_s=2.5) on a 20-throw/1-drop
    # fixture silently dropping 90% of catches and the drop entirely.
    # 1.5s is a stated, not derived, flight-time allowance: comfortably above
    # a real arc's flight+freeze lag (~2.6s measured in _analyze's left-edge
    # comment includes freeze already, so this is the remaining slack on top
    # of freeze/drop_freeze) and below window_s's own default margin (8.0 -
    # 0.5 - 2.5 = 5.0 >> 1.5).
    @model_validator(mode="after")
    def _validate_envelope(self) -> "RealtimeConfig":
        if self.window_s <= 0:
            raise ValueError(f"window_s must be > 0 (got {self.window_s})")
        if self.cadence_frames < 1:
            raise ValueError(f"cadence_frames must be >= 1 (got {self.cadence_frames})")
        if self.freeze_s < 0:
            raise ValueError(f"freeze_s must be >= 0 (got {self.freeze_s})")
        if self.drop_freeze_s < 0:
            raise ValueError(f"drop_freeze_s must be >= 0 (got {self.drop_freeze_s})")
        if not (0.0 < self.hand_line_ema_alpha <= 1.0):
            raise ValueError(
                f"hand_line_ema_alpha must be in (0, 1] (got {self.hand_line_ema_alpha})"
            )
        flight_allowance = 1.5
        required = max(self.freeze_s, self.drop_freeze_s) + flight_allowance
        if self.window_s - self.edge_pad <= required:
            raise ValueError(
                f"window_s - edge_pad ({self.window_s - self.edge_pad:.2f}) must exceed "
                f"max(freeze_s, drop_freeze_s) + {flight_allowance} ({required:.2f}) -- "
                "otherwise every event's arc is truncated by the left-edge guard before "
                "its own confirmation horizon arrives, silently discarding events every "
                "cycle (see RealtimeConfig's model_validator docstring / finding E4)"
            )
        return self


class RealtimeState(BaseModel):
    t: float
    catches_total: int = 0
    throws_total: int = 0
    drops_total: int = 0
    runs_completed: int = 0
    run_active: bool = False
    catches_current_run: int = 0
    hand_line_y: float = 0.0
    window_arcs: list[Arc] = Field(default_factory=list)
    last_analysis_ms: float = 0.0


class RealtimeAnalyzer:
    # Run-close debounce: how long a run must look "not live" (see _analyze)
    # before the close is actually committed to runs_completed. Re-running
    # analyze_detections on a shifting, noisy window can momentarily lose the
    # tail arc's link (a dropout run, a re-extraction gap right at the
    # freeze horizon) and make a genuinely ongoing run look finished for one
    # or two analysis cycles before the next window relinks it -- without a
    # debounce this flaps runs_completed +1 too high (measured: ~6% of
    # noisy/dropout streams; matches the brief's predicted failure mode).
    # 0.5s: comparable to segment_runs's own same-run merge gap
    # (gap_factor * period ~= 1.3 * 0.45 = 0.585s for the sim's default
    # cascade), so a real second run (a genuine pause) still counts as new.
    # Deliberately does not increment-then-decrement runs_completed to undo a
    # false close: RealtimeState.runs_completed is exposed every frame and
    # must stay monotonic (test_counters_are_monotonic), so the close itself
    # is deferred rather than committed-and-rolled-back.
    #
    # THE TRADE, STATED PLAINLY: this fixes the flicker-overshoot direction
    # (test_counters_are_monotonic and the parity tests would otherwise
    # occasionally see runs_completed tick up one extra) but it KNOWINGLY
    # merges two REAL runs when the silence between them is inside roughly
    # freeze_s's own horizon (~0.6-1.5s band, empirically) -- see
    # test_debounce_merges_narrow_gap_runs_known_tradeoff, which pins that
    # exact under-count as current behavior, and
    # test_two_runs_with_wide_gap_counted_separately, which pins the
    # (correct) count once the gap is comfortably outside that band. This
    # happens because `freeze_s` is doing double duty here: it is both the
    # event-confirmation lag (how long before a throw/catch is trusted) and
    # the run-liveness horizon (`live = ... r.end_t > self._now - freeze` in
    # _analyze) that this debounce is measured against. A short real gap and
    # a same-run dropout look identical to the debounce because both drain
    # from the same clock. Revisit this constant (and likely split it in
    # two) once run-liveness has its own horizon decoupled from
    # event-confirmation lag -- until then, flipping either pinning test's
    # assertion is a deliberate, visible decision, not a drive-by fix.
    RUN_CLOSE_DEBOUNCE_S = 0.5

    def __init__(self, config: RealtimeConfig | None = None):
        self.cfg = config or RealtimeConfig()
        self._buffer: list[Detection] = []
        self._frames_since_analysis = 0
        self._now = 0.0
        self._confirmed: dict[str, list[float]] = {"throw": [], "catch": [], "drop": []}
        self._runs_completed = 0
        self._run_start: float | None = None
        self._pending_close_t: float | None = None
        self._finalized = False
        self._last_state = RealtimeState(t=0.0)
        self._hand_line_ema: float | None = None

    def feed(self, dets: list[Detection], t: float) -> RealtimeState:
        # E5: feed() after finalize() used to silently resume analysis with
        # no way to ever flush the resumed tail again (finalize() is
        # idempotent -- a second call is a no-op once _finalized is set),
        # permanently losing events inside the final freeze window and any
        # run re-opened after the "final" state. Make the contract explicit
        # rather than leaving it an undefined, silently-lossy resume.
        if self._finalized:
            raise RuntimeError(
                "feed() called after finalize(): RealtimeAnalyzer cannot resume "
                "a finalized session (construct a new RealtimeAnalyzer instead)"
            )
        self._now = t
        self._buffer.extend(dets)
        cut = t - self.cfg.window_s
        # E7: filter unconditionally. The old `if buffer[0].t < cut` early
        # exit assumed the buffer stays t-sorted (true for well-behaved
        # frame-by-frame feeding, but not guaranteed -- a single
        # out-of-order/anomalous detection at index 0 with a large t
        # defeats the proxy and defers eviction of everything behind it
        # indefinitely, growing the buffer -- and therefore every window's
        # re-analysis cost -- unboundedly).
        self._buffer = [d for d in self._buffer if d.t >= cut]
        self._frames_since_analysis += 1
        if self._frames_since_analysis >= self.cfg.cadence_frames:
            self._frames_since_analysis = 0
            self._analyze(freeze=self.cfg.freeze_s, drop_freeze=self.cfg.drop_freeze_s)
        return self._state()

    def finalize(self) -> RealtimeState:
        if not self._finalized:
            self._analyze(freeze=-1.0, drop_freeze=-1.0)  # horizon past the end
            if self._run_start is not None:
                # Force-close bypasses the debounce in _analyze (finalize
                # confirms everything pending, unconditionally) -- so it must
                # also patch _last_state itself: _analyze already built this
                # frame's snapshot with the run still open (the debounce
                # hadn't committed yet), and _state() below only refreshes
                # `t`, not the fields this closing actually changes.
                self._runs_completed += 1
                self._run_start = None
                self._pending_close_t = None
                self._last_state = self._last_state.model_copy(update={
                    "runs_completed": self._runs_completed,
                    "run_active": False,
                    "catches_current_run": 0,
                })
            self._finalized = True
        return self._state()

    def _analyze(self, *, freeze: float, drop_freeze: float) -> None:
        t0 = time.perf_counter()
        session = analyze_detections(list(self._buffer), self.cfg.analyze)

        # Left-edge guard: any arc whose t_start is within edge_pad of the
        # window's trailing cut (`left_edge = now - window_s`, matching
        # feed()'s own eviction rule -- NOT `buffer[0].t`, which stays
        # pinned at the stream's own start for the whole first window_s
        # seconds and would falsely flag every early-session arc as
        # "truncated" before feed() has ever evicted anything) had its early
        # history trimmed by the sliding buffer before this cycle's
        # extraction ever ran. Its fitted coefficients (and anything derived
        # from it: crossing times, floor-descent signals) reflect a
        # tail-only fit, not the ball's real trajectory. window_s (8s) is
        # far longer than a real arc's flight+freeze lag (~2.6s), so a
        # genuine event is always confirmed from a still-full,
        # non-truncated fit several cycles before its arc could ever reach
        # this edge -- discarding events/drops sourced from a truncated arc
        # therefore never loses a real confirmation, only a corrupted
        # re-derivation of one that already landed.
        #
        # Deliberately NOT implemented as "drop the arc and re-run
        # segment_runs/detect_drops on what's left": measured (see
        # .superpowers/sdd/task-5b-report.md) that this footage can have as
        # few as 2-5 arcs in a full 8s window (a slow real cascade, not this
        # module's fast synthetic-test cadence), so removing even one arc
        # routinely drops the group below segment_runs' min_arcs=3 gate and
        # wipes out run recognition entirely -- worse than the bug it was
        # meant to fix. Filtering the DERIVED events/drops list is enough:
        # it can only ever remove a confirmation, never fabricate one.
        left_edge = self._now - self.cfg.window_s
        truncated_ids = {a.id for a in session.arcs if a.t_start < left_edge + self.cfg.edge_pad}

        # Hand-line EMA (see RealtimeConfig.hand_line_ema_alpha's docstring
        # for the measured jitter this smooths).
        if session.arcs:
            raw_hand = session.hand_line_y
            self._hand_line_ema = (
                raw_hand if self._hand_line_ema is None
                else self.cfg.hand_line_ema_alpha * raw_hand
                + (1.0 - self.cfg.hand_line_ema_alpha) * self._hand_line_ema
            )
        hand_line = self._hand_line_ema if self._hand_line_ema is not None else session.hand_line_y

        # E2: ONE hand line drives EVERY derivation this cycle, not just
        # derive_events. Before this fix, session.runs/session.drops came
        # from analyze_detections' RAW per-window estimate (analyze.py's
        # _events_from_arcs, called with no override) while throws/catches
        # below used the EMA line -- so an arc landing in the straddle band
        # between the two lines could be a witnessed catch per one line AND
        # an independent drop candidate per the other, an invariant offline
        # can never violate (it only ever has one line). Re-running the
        # post-extraction tail on the EMA line is cheap (no re-extraction;
        # see _events_from_arcs's docstring for why this is the one shared
        # code path rather than a realtime-only reimplementation) -- the
        # mild cost of computing it twice per cycle (once raw inside
        # analyze_detections above, once more here on the EMA line) is
        # deliberately accepted to keep arc extraction single-sourced.
        ema_session = (
            _events_from_arcs(session.arcs, session.meta["t_last"], self.cfg.analyze,
                               hand_line_override=hand_line)
            if session.arcs else session
        )

        throws, catches = derive_events(session.arcs, hand_line)
        # E1: gate the sticky-open liveness signal (below) to arcs that
        # actually produced a derived throw -- i.e. apex cleared the hand
        # line -- computed BEFORE the truncation filter so it reflects
        # every event-bearing arc in the window, truncated or not.
        event_bearing_ids = {e.arc_id for e in throws}
        run_arc_ids = {aid for run in ema_session.runs for aid in run.arc_ids}

        throws = [e for e in throws if e.arc_id not in truncated_ids]
        # E3: live must only confirm catches offline would also count.
        # Offline's parity reference (sum of Run.catches, runs.py:96) counts
        # only catches whose arc belongs to a segmented, min_arcs-gated run
        # -- but live used to confirm EVERY derived catch regardless of run
        # membership. Sub-min_arcs activity (warm-up tosses, an isolated
        # throw-catch between runs) is real detector output that
        # contributes nothing offline, so confirming it live silently
        # inflated catches_total relative to the parity contract (measured:
        # a clean 2-throw stream gives offline 0 run-catches vs live's old
        # catches_total=2). `run_arc_ids` reflects the CURRENT window's
        # segment_runs groups (any run recognized in this window, live or
        # not), matching offline's inclusion of every surviving run's
        # catches, not just the currently-open one.
        catches = [
            e for e in catches
            if e.arc_id not in truncated_ids and e.arc_id in run_arc_ids
        ]
        # A drop with arc_id=None is never emitted by detect_drops today,
        # but the type allows it (DropEvent.arc_id: int | None) -- treat
        # "no arc to check" as "not truncated" rather than crashing.
        drops = [
            d for d in ema_session.drops
            if d.arc_id is None or d.arc_id not in truncated_ids
        ]

        self._confirm("throw", [e.t for e in throws], self._now - freeze)
        self._confirm("catch", [e.t for e in catches], self._now - freeze)
        self._confirm("drop", [d.t for d in drops], self._now - drop_freeze)

        # Run liveness: the naive `end_t > now - freeze` check is exactly
        # right for deciding whether to OPEN a new tracked run (it's
        # segment_runs' own, min_arcs-gated notion of "a real run exists").
        # But measured evidence (task-5b-report.md) shows this same check
        # can flap false on an ALREADY-open run purely from re-extraction
        # noise at low arc density -- segment_runs re-derives the whole
        # arc/run graph from scratch every cadence tick, and at 2-5 arcs per
        # window (slow real cascades), losing or regrouping even one arc
        # between consecutive ticks can swing "one clearly live run" to "no
        # runs at all", with no left-edge truncation involved. Once a run is
        # already open, treat recent EVENT-BEARING arc activity (E1: an arc
        # that produced a derived throw -- apex cleared the hand line --
        # NOT any arc in the window at all) as proof the pattern is still
        # going, so a single from-scratch-re-extraction blip can't force a
        # false close -- only genuine silence (no throw-bearing arc
        # activity anywhere near `now`) still ends it. Before E1, ANY arc
        # counted, including sub-hand-line junk (a dropped ball bouncing on
        # the floor) that can never become a throw/catch/run -- a
        # domain-native source of continuous "activity" that bridged
        # arbitrarily long inter-run silences and merged two real runs into
        # one (reviewer's repro: a bounce chain filling a 3.0s gap between
        # two runs collapsed live's count from 2 to 1; offline unaffected).
        #
        # E6: liveness always uses cfg.freeze_s, a config-derived constant,
        # NOT the `freeze` parameter this call received -- finalize() passes
        # freeze=-1.0 for IMMEDIATE event confirmation (see _confirm's
        # horizon use just above), but reusing that same -1 here would make
        # `now - freeze == now + 1.0`, an unsatisfiable bound, for BOTH
        # `strict_live` and `recent_activity`. That silently prevented a run
        # from ever opening if its completing arcs only ever appeared in the
        # analyzer's very last window (never seen live during an earlier
        # periodic feed() cycle) -- the whole run would vanish from
        # runs_completed instead of being recognized and immediately
        # force-closed by finalize()'s own explicit close logic below.
        liveness_freeze = self.cfg.freeze_s
        strict_live = any(r.end_t > self._now - liveness_freeze for r in ema_session.runs)
        if self._run_start is None:
            live = strict_live
        else:
            recent_activity = any(
                a.t_end > self._now - liveness_freeze
                for a in session.arcs if a.id in event_bearing_ids
            )
            live = strict_live or recent_activity

        if live:
            # Live again: cancel any pending close outright. run_start is
            # left untouched when one was already open (the debounce below
            # never cleared it), so a flap-and-recover never mints a second
            # run identity or resets catches_current_run.
            self._pending_close_t = None
            if self._run_start is None:
                starts = [
                    r.start_t for r in ema_session.runs
                    if r.end_t > self._now - liveness_freeze
                ]
                self._run_start = min(starts)
        elif self._run_start is not None:
            if self._pending_close_t is None:
                self._pending_close_t = self._now
            elif self._now - self._pending_close_t > self.RUN_CLOSE_DEBOUNCE_S:
                self._runs_completed += 1
                self._run_start = None
                self._pending_close_t = None

        self._last_state = RealtimeState(
            t=self._now,
            catches_total=len(self._confirmed["catch"]),
            throws_total=len(self._confirmed["throw"]),
            drops_total=len(self._confirmed["drop"]),
            runs_completed=self._runs_completed,
            run_active=self._run_start is not None,
            catches_current_run=sum(
                1 for ct in self._confirmed["catch"]
                if self._run_start is not None and ct >= self._run_start
            ),
            hand_line_y=hand_line,
            window_arcs=session.arcs,
            last_analysis_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def _confirm(self, kind: str, times: list[float], horizon: float) -> None:
        known = self._confirmed[kind]
        # E8: dedup only against events confirmed in PRIOR cycles (`prior`,
        # snapshotted before this call's loop starts) -- NOT against entries
        # this same call appends as it goes. `times` is the FULL set of
        # currently-derived event times for this window (re-derived from
        # scratch every cycle, so it naturally re-includes already-known
        # events alongside any new ones); one derive_events()/detect_drops()
        # pass never emits two synthetic times for the same physical event
        # (each is tied to a distinct arc id), so two same-cycle times within
        # event_match_tol of EACH OTHER are genuinely distinct events, not a
        # dedup collision. Checking against `known` as it mutated in-place
        # used to treat the second of two real, closely-spaced events as a
        # "duplicate" of the first (added moments earlier in the same call)
        # and silently drop it.
        prior = list(known)
        for et in sorted(times):
            if et > horizon:
                continue
            if any(abs(et - k) <= self.cfg.event_match_tol for k in prior):
                continue
            known.append(et)

    def _state(self) -> RealtimeState:
        return self._last_state.model_copy(update={"t": self._now})
