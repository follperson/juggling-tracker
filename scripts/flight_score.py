"""Score offline catches against flights read off the Meschke ball labels.

``oracle_events`` builds its reference with the arc extractor it is meant to
judge. This check uses ``juggletrack.eval.flight_reference`` instead, which
reads flights off the labelled per-ball tracks with no parabola fit. For each
``<clip>/detections.jsonl`` under the root it runs ``analyze_detections``,
matches the catches the pipeline counted to the reference flights one to one,
and counts the flights that an extracted arc explains. The video supplies fps
and frame size only.

    uv run --no-sync python scripts/flight_score.py flight-score.json
    uv run --no-sync python scripts/flight_score.py controls.json \\
        --only ss3_id_987 --controls

``--controls`` also scores three seeded faults per clip: every tenth catch
deleted, every tenth catch doubled, and every catch moved 2 s later. They
must show up as missed flights, duplicates and unmatched catches.

A clip whose flight count moves with the detection threshold is flagged, and
its numbers are kept out of the trusted totals. Two runs on the same inputs
write byte-identical reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy

import juggletrack
from juggletrack.analyze import AnalyzeConfig, analyze_detections
from juggletrack.data.meschke_import import load_meschke_trajectories
from juggletrack.eval.benchmark import _implementation_sha256
from juggletrack.eval.flight_reference import (
    FlightReference,
    flight_reference,
    score_arcs,
    score_catches,
)
from juggletrack.events.catches import derive_events
from juggletrack.pipeline.offline import load_detections_jsonl
from juggletrack.types import Arc, CatchEvent, SessionResult
from juggletrack.video.reader import VideoReader

DEFAULT_ROOT = "outputs/meschke-val/post-hardening-2b"
COUNTS = ("flights", "catches", "matched", "duplicates", "unmatched", "ignored", "missed",
          "count_error")
ARC_COUNTS = ("arcs", "explained", "repeated")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _overwrites_input(out: Path, inputs: list[Path]) -> bool:
    if "detections.jsonl" in (out.name, out.resolve().name):
        return True
    # samefile also catches symlink and hardlink aliases.
    return out.exists() and any(out.samefile(p) for p in inputs if p.exists())


def _counted_catches(result: SessionResult) -> list[CatchEvent]:
    """The catch events behind the pipeline's count: those on arcs of a kept run."""
    in_a_run = {arc_id for run in result.runs for arc_id in run.arc_ids}
    _, catches = derive_events(result.arcs, result.hand_line_y)
    counted = [c for c in catches if c.arc_id in in_a_run]
    if len(counted) != sum(run.catches for run in result.runs):
        raise RuntimeError(
            f"rebuilt {len(counted)} catch events but the runs count "
            f"{sum(run.catches for run in result.runs)}; this script no longer "
            "matches how analyze_detections counts catches"
        )
    return counted


def _seeded_faults(catches: list[CatchEvent]) -> dict[str, list[CatchEvent]]:
    return {
        "delete_10pct": [c for i, c in enumerate(catches) if i % 10],
        "duplicate_10pct": sorted(catches + catches[::10], key=lambda c: c.t),
        "shift_2s": [c.model_copy(update={"t": c.t + 2.0}) for c in catches],
    }


def _score(reference: FlightReference, catches: list[CatchEvent]) -> dict:
    score = score_catches(reference, catches)
    judged = list(zip(catches, score.verdicts))
    return {
        **score.model_dump(exclude={"verdicts", "missed"}),
        "missed": len(score.missed),
        "missed_apex_t": [round(reference.flights[k].apex_t, 3) for k in score.missed],
        "duplicate_t": [round(c.t, 3) for c, v in judged if v == "duplicate"],
        "unmatched_t": [round(c.t, 3) for c, v in judged if v == "unmatched"],
    }


def _explained(reference: FlightReference, arcs: list[Arc]) -> dict:
    score = score_arcs(reference, arcs)
    return {
        "arcs": score.arcs, "explained": score.explained, "repeated": score.repeated,
        "unexplained_apex_t": [round(reference.flights[k].apex_t, 3) for k in score.unexplained],
    }


def _line(name: str, trusted: str, row: dict) -> str:
    def cells(keys: tuple[str, ...]) -> str:
        return " ".join(f"{row[k]:>{max(7, len(k))}}" for k in keys)

    # A median offset does not add up across clips, so totals leave it blank.
    offset = row.get("offset_s", "")
    if isinstance(offset, float):
        offset = f"{offset:+.3f}"
    elif offset is None:
        offset = "-"
    arcs = f" | {cells(ARC_COUNTS)}" if "arcs" in row else ""
    return f"{name:30s} {trusted:7s} {cells(COUNTS)} {offset:>8}{arcs}".rstrip()


def _total(rows: list[dict]) -> dict:
    keys = [k for k in (*COUNTS, *ARC_COUNTS) if all(k in r for r in rows)]
    return {"clips": len(rows), **{k: sum(r[k] for r in rows) for k in keys}}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("out", help="JSON report to write")
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="directory of <clip>/detections.jsonl (default: %(default)s)")
    parser.add_argument("--csv-dir", default="data/raw/meschke/csv")
    parser.add_argument("--video-dir", default="data/raw/meschke/videos")
    parser.add_argument("--only", action="append", default=[],
                        help="keep clips whose name contains this substring (repeatable)")
    parser.add_argument("--controls", action="store_true",
                        help="also score three seeded faults per clip")
    args = parser.parse_args()

    root, out = Path(args.root), Path(args.out)
    saved = {p.parent.name: p for p in sorted(root.glob("*/detections.jsonl"))}
    labels = {clip: Path(args.csv_dir) / f"{clip}.csv" for clip in saved}
    videos = {clip: Path(args.video_dir) / f"{clip}.MP4" for clip in saved}
    # Check every input under the root, not just the --only selection: a
    # report must never replace recorded detections, labels or footage.
    if _overwrites_input(out, [*saved.values(), *labels.values(), *videos.values()]):
        print(f"refusing to write {out}: it would overwrite an input", file=sys.stderr)
        return 2

    selected = [c for c in saved if not args.only or any(s in c for s in args.only)]
    clips = [c for c in selected if labels[c].exists() and videos[c].exists()]
    for clip in sorted(set(selected) - set(clips)):
        print(f"skip  {clip}: no Meschke labels or video", file=sys.stderr)
    if not clips:
        print(f"no clip with detections under {root} and Meschke labels", file=sys.stderr)
        return 2

    config = AnalyzeConfig()
    report: dict = {
        "meta": {
            "package": str(Path(juggletrack.__file__).parent),
            "implementation_sha256": _implementation_sha256(),
            "script_sha256": _sha256(Path(__file__)),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "root": str(root.resolve()),
            "csv_dir": str(Path(args.csv_dir).resolve()),
            "video_dir": str(Path(args.video_dir).resolve()),
            "analyze_config": config.model_dump(mode="json"),
        },
        "clips": {},
    }
    print(_line("clip", "trusted", {k: k for k in (*COUNTS, "offset_s", *ARC_COUNTS)}))

    for clip in clips:
        with VideoReader(videos[clip]) as reader:
            info = reader.info
        dets = load_detections_jsonl(saved[clip])
        off_clock = max((abs(d.t - d.frame_idx / info.fps) for d in dets), default=0.0)
        if off_clock > 0.5 / info.fps:
            print(f"{clip}: detections are up to {off_clock:.3f} s off the frame_idx / fps "
                  "clock that the labels use, so catch times cannot be compared",
                  file=sys.stderr)
            return 2
        reference = flight_reference(load_meschke_trajectories(
            labels[clip], fps=info.fps, width=info.width, height=info.height,
        ))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.exceptions.RankWarning)
            result = analyze_detections(dets, config)
        catches = _counted_catches(result)

        row = {
            "detections_sha256": _sha256(saved[clip]),
            "labels_sha256": _sha256(labels[clip]),
            "fps": info.fps, "width": info.width, "height": info.height,
            "labelled_s": list(reference.span),
            "trusted": reference.trusted,
            "sweep": [list(pair) for pair in reference.sweep],
            "flights_found": len(reference.flights),
            "median_flight_s": reference.median_flight_s,
            **_score(reference, catches),
            **_explained(reference, result.arcs),
        }
        print(_line(clip, "yes" if reference.trusted else "FLAGGED", row), flush=True)
        if args.controls:
            row["controls"] = {
                name: _score(reference, seeded)
                for name, seeded in _seeded_faults(catches).items()
            }
            for name, control in row["controls"].items():
                print(_line(f"  {name}", "", control), flush=True)
        report["clips"][clip] = row

    rows = list(report["clips"].values())
    report["totals"] = {
        "trusted": _total([r for r in rows if r["trusted"]]),
        "all": _total(rows),
    }
    for name, label in (("trusted", "TRUSTED"), ("all", "ALL WITH FLAGGED")):
        total = report["totals"][name]
        print(_line(f"{label} ({total['clips']} clips)", "", total))
    if args.controls:
        report["totals"]["controls"] = {
            name: _total([r["controls"][name] for r in rows])
            for name in next(iter(rows))["controls"]
        }
        for name, total in report["totals"]["controls"].items():
            print(_line(f"  {name} (all clips)", "", total))

    out.write_text(json.dumps(report, indent=1) + "\n")
    flagged = sum(not r["trusted"] for r in rows)
    print(f"{len(rows)} clips, {flagged} flagged -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
