"""Realtime event engine: sliding-window re-analysis with freeze-horizon
confirmation.

CONSCIOUS SPEC DEVIATION (documented in the plan header): spec §3 called for
a bespoke online two-mode Kalman tracker; measured extraction cost (~0.01s
per 8s window) makes re-running the proven offline core affordable at frame
rate, giving offline parity by construction. The Kalman design remains the
contingency if phone-class profiling demands it.

This module must stay cv2-free: it consumes Detections, not frames.
"""
from __future__ import annotations

import time

from pydantic import BaseModel, Field

from juggletrack.analyze import AnalyzeConfig, analyze_detections
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
        self._now = t
        self._buffer.extend(dets)
        cut = t - self.cfg.window_s
        if self._buffer and self._buffer[0].t < cut:
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

        throws, catches = derive_events(session.arcs, hand_line)
        throws = [e for e in throws if e.arc_id not in truncated_ids]
        catches = [e for e in catches if e.arc_id not in truncated_ids]
        # A drop with arc_id=None is never emitted by detect_drops today,
        # but the type allows it (DropEvent.arc_id: int | None) -- treat
        # "no arc to check" as "not truncated" rather than crashing.
        drops = [d for d in session.drops if d.arc_id is None or d.arc_id not in truncated_ids]

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
        # already open, treat ANY recent arc activity (any arc at all, not
        # gated by min_arcs or first-missed-catch grouping) as proof the
        # pattern is still going, so a single from-scratch-re-extraction
        # blip can't force a false close -- only genuine silence (no arc
        # activity anywhere near `now`) still ends it.
        strict_live = any(r.end_t > self._now - freeze for r in session.runs)
        if self._run_start is None:
            live = strict_live
        else:
            recent_activity = any(a.t_end > self._now - freeze for a in session.arcs)
            live = strict_live or recent_activity

        if live:
            # Live again: cancel any pending close outright. run_start is
            # left untouched when one was already open (the debounce below
            # never cleared it), so a flap-and-recover never mints a second
            # run identity or resets catches_current_run.
            self._pending_close_t = None
            if self._run_start is None:
                starts = [r.start_t for r in session.runs if r.end_t > self._now - freeze]
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
        for et in sorted(times):
            if et > horizon:
                continue
            if any(abs(et - k) <= self.cfg.event_match_tol for k in known):
                continue
            known.append(et)

    def _state(self) -> RealtimeState:
        return self._last_state.model_copy(update={"t": self._now})
