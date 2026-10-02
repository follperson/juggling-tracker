"""Prove an event-core change preserves output, and measure its speed.

Snapshot every saved ``detections.jsonl`` before and after a change on the
same machine, then compare. Offline mode hashes the full ``SessionResult`` of
``analyze_detections``. Realtime mode replays selected sessions frame by
frame through ``RealtimeAnalyzer`` and hashes every emitted state.

    uv run --no-sync python scripts/event_core_parity.py snapshot before.json
    # ...apply the change...
    uv run --no-sync python scripts/event_core_parity.py snapshot after.json
    uv run --no-sync python scripts/event_core_parity.py compare before.json after.json

Digests depend on the platform's floating point and numpy build. Compare two
snapshots from one machine; never commit them as golden values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import juggletrack
from juggletrack.analyze import analyze_detections
from juggletrack.eval.benchmark import _implementation_sha256
from juggletrack.pipeline.offline import load_detections_jsonl
from juggletrack.pipeline.realtime import RealtimeAnalyzer
from juggletrack.types import Detection


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _offline(dets: list[Detection]) -> dict:
    t0 = time.perf_counter()
    result = analyze_detections(dets)
    seconds = time.perf_counter() - t0
    return {
        "digest": _sha256(result.model_dump_json().encode()),
        "seconds": seconds,
        "arcs": len(result.arcs),
        "runs": len(result.runs),
        "catches": sum(r.catches for r in result.runs),
        "drops": len(result.drops),
    }


def _realtime(dets: list[Detection], max_frames: int) -> dict:
    by_frame: dict[int, list[Detection]] = {}
    for d in dets:
        by_frame.setdefault(d.frame_idx, []).append(d)
    last = max(by_frame)
    fps = last / by_frame[last][0].t if last else 30.0
    n_frames = last + 1 if max_frames <= 0 else min(last + 1, max_frames)

    analyzer = RealtimeAnalyzer()
    digest = hashlib.sha256()
    feed_s = 0.0
    cycle_ms: list[float] = []
    for idx in range(n_frames):
        t0 = time.perf_counter()
        state = analyzer.feed(by_frame.get(idx, []), idx / fps)
        feed_s += time.perf_counter() - t0
        if state.last_analysis_ms and (not cycle_ms or state.last_analysis_ms != cycle_ms[-1]):
            cycle_ms.append(state.last_analysis_ms)
        digest.update(state.model_dump_json(exclude={"last_analysis_ms"}).encode())
    final = analyzer.finalize()
    digest.update(final.model_dump_json(exclude={"last_analysis_ms"}).encode())
    return {
        "digest": digest.hexdigest(),
        "frames": n_frames,
        "feed_seconds": feed_s,
        "cycles": len(cycle_ms),
        "analysis_ms_p50": statistics.median(cycle_ms) if cycle_ms else 0.0,
        "analysis_ms_p95": float(np.percentile(cycle_ms, 95)) if cycle_ms else 0.0,
        "final": [final.runs_completed, final.catches_total, final.drops_total],
    }


def snapshot(args: argparse.Namespace) -> int:
    root = Path(args.root)
    paths = sorted(root.rglob("detections.jsonl"))
    if args.only:
        paths = [p for p in paths if any(s in str(p) for s in args.only)]
    if not paths:
        print(f"no detections.jsonl under {root}", file=sys.stderr)
        return 2

    # Stamp the code identity before analysing: the run can take minutes, and
    # the checkout may move on while it does.
    meta = {
        "package": str(Path(juggletrack.__file__).parent),
        "implementation_sha256": _implementation_sha256(),
        "git_head": _git_head(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "root": str(root.resolve()),
        "realtime_frames": args.realtime_frames,
    }
    offline: dict[str, dict] = {}
    realtime: dict[str, dict] = {}
    for path in paths:
        rel = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        dets = load_detections_jsonl(path)
        if args.max_detections and len(dets) > args.max_detections:
            print(f"skip  n={len(dets):7d} {rel}", flush=True)
            continue
        with warnings.catch_warnings():
            # RankWarning from degenerate fits is part of today's behaviour, not
            # a parity signal; silence it so the report stays readable.
            warnings.simplefilter("ignore")
            row = {"input_sha256": _sha256(raw), "n": len(dets), **_offline(dets)}
            offline[rel] = row
            print(f"{row['seconds']:8.3f}s n={len(dets):7d} arcs={row['arcs']:5d} {rel}", flush=True)
            if dets and any(s in rel for s in args.realtime):
                rt = _realtime(dets, args.realtime_frames)
                realtime[rel] = rt
                print(
                    f"  realtime frames={rt['frames']} feed={rt['feed_seconds']:.2f}s "
                    f"cycle p50={rt['analysis_ms_p50']:.1f}ms p95={rt['analysis_ms_p95']:.1f}ms "
                    f"final runs/catches/drops={rt['final']}",
                    flush=True,
                )

    Path(args.out).write_text(json.dumps(
        {"meta": meta, "offline": offline, "realtime": realtime}, indent=1,
    ))
    total = sum(r["seconds"] for r in offline.values())
    print(f"offline sessions={len(offline)} analyze_total={total:.2f}s -> {args.out}")
    return 0


def _compare_section(name: str, base: dict, new: dict, time_key: str) -> int:
    missing = sorted(set(base) ^ set(new))
    for rel in missing:
        print(f"{name}: present in only one snapshot: {rel}")
    common = sorted(set(base) & set(new))
    bad = 0
    for rel in common:
        b, n = base[rel], new[rel]
        if b.get("input_sha256") != n.get("input_sha256"):
            print(f"{name}: INPUT CHANGED {rel}")
            bad += 1
        elif b["digest"] != n["digest"]:
            summary = {k: (b[k], n[k]) for k in ("arcs", "runs", "catches", "drops", "final")
                       if k in b and b[k] != n[k]}
            print(f"{name}: MISMATCH {rel} {summary}")
            bad += 1
    if common:
        tb = sum(base[r][time_key] for r in common)
        tn = sum(new[r][time_key] for r in common)
        ratios = [base[r][time_key] / new[r][time_key] for r in common if new[r][time_key] > 0]
        print(
            f"{name}: sessions={len(common)} mismatches={bad} missing={len(missing)} "
            f"{time_key} {tb:.2f}s -> {tn:.2f}s ({tb / tn:.2f}x total, "
            f"median {statistics.median(ratios):.2f}x)"
        )
    return bad + len(missing)


def compare(args: argparse.Namespace) -> int:
    base = json.loads(Path(args.base).read_text())
    new = json.loads(Path(args.new).read_text())
    print("implementation " + " -> ".join(
        s["meta"].get("implementation_sha256", "unrecorded")[:12] for s in (base, new)))
    for key in ("python", "numpy", "platform"):
        if base["meta"][key] != new["meta"][key]:
            print(f"warning: {key} differs ({base['meta'][key]} vs {new['meta'][key]}); "
                  "digests are only comparable on one machine and environment")
    failures = _compare_section("offline", base["offline"], new["offline"], "seconds")
    if base["realtime"] or new["realtime"]:
        failures += _compare_section("realtime", base["realtime"], new["realtime"], "feed_seconds")
        for rel in sorted(set(base["realtime"]) & set(new["realtime"])):
            b, n = base["realtime"][rel], new["realtime"][rel]
            print(f"realtime: {rel} cycle p50 {b['analysis_ms_p50']:.1f} -> "
                  f"{n['analysis_ms_p50']:.1f}ms, p95 {b['analysis_ms_p95']:.1f} -> "
                  f"{n['analysis_ms_p95']:.1f}ms")
    print("PARITY OK" if failures == 0 else f"PARITY FAILED ({failures})")
    return 0 if failures == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot", help="hash and time every saved session")
    snap.add_argument("out")
    snap.add_argument("--root", default="outputs")
    snap.add_argument("--only", action="append", default=[],
                      help="keep paths containing this substring (repeatable)")
    snap.add_argument("--max-detections", type=int, default=0,
                      help="skip sessions with more detections than this (0 keeps all)")
    snap.add_argument("--realtime", action="append", default=[],
                      help="also replay paths containing this substring through RealtimeAnalyzer")
    snap.add_argument("--realtime-frames", type=int, default=0,
                      help="cap each realtime replay at this many frames (0 replays all)")
    snap.set_defaults(func=snapshot)
    cmp = sub.add_parser("compare", help="fail on any digest difference; report speed")
    cmp.add_argument("base")
    cmp.add_argument("new")
    cmp.set_defaults(func=compare)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
