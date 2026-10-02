"""Reproducible saved-detection benchmarks against independently reviewed labels."""
from __future__ import annotations

import bisect
import hashlib
import math
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from juggletrack import SCHEMA_VERSION
from juggletrack.analyze import AnalyzeConfig, analyze_detections
from juggletrack.eval.labels import VideoLabels
from juggletrack.eval.metrics import EvalReport, evaluate_session
from juggletrack.pipeline.offline import load_detections_jsonl
from juggletrack.pipeline.realtime import RealtimeAnalyzer, RealtimeConfig
from juggletrack.types import Detection


class BenchmarkClip(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    video: str = Field(min_length=1)
    detections: Path
    labels: Path
    label_source: Literal["human_reviewed"]
    fps: float = Field(gt=0)
    frame_count: int = Field(gt=0)
    detector: dict[str, str | float | int] = Field(default_factory=dict)


class BenchmarkManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"]
    split: Literal["development", "holdout"]
    clips: list[BenchmarkClip] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_videos(self) -> "BenchmarkManifest":
        if len({c.video for c in self.clips}) != len(self.clips):
            raise ValueError("benchmark video identities must be unique")
        return self


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _implementation_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _validate_detections(dets: list[Detection], clip: BenchmarkClip) -> None:
    for d in dets:
        values = (d.t, d.x, d.y, d.w, d.h, d.confidence)
        if (not all(math.isfinite(v) for v in values)
                or not 0 <= d.frame_idx < clip.frame_count
                or not 0 <= d.t < clip.frame_count / clip.fps
                or not 0 <= d.confidence <= 1):
            raise ValueError(f"{clip.video}: invalid detection at frame {d.frame_idx}")


def _recorded_clock(frame_times: dict[int, float], frame_count: int, fps: float) -> list[float]:
    """Each frame's time on the clock the detections were recorded with.

    The realtime window evicts on ``d.t >= now - window_s``, so a clock that
    differs from the recorded one by a rounding error can change results.
    Empty frames between detected frames are interpolated; frames before the
    first or after the last detected frame step by ``1 / fps`` from it.
    """
    known = sorted(frame_times)
    clock = []
    for idx in range(frame_count):
        pos = bisect.bisect_left(known, idx)
        if pos < len(known) and known[pos] == idx:
            clock.append(frame_times[idx])
        elif 0 < pos < len(known):
            i0, i1 = known[pos - 1], known[pos]
            t0, t1 = frame_times[i0], frame_times[i1]
            clock.append(t0 + (t1 - t0) * (idx - i0) / (i1 - i0))
        elif pos:
            clock.append(frame_times[known[-1]] + (idx - known[-1]) / fps)
        elif known:
            clock.append(frame_times[known[0]] - (known[0] - idx) / fps)
        else:
            clock.append(idx / fps)
    return clock


def _replay(dets: list[Detection], clip: BenchmarkClip, cfg: RealtimeConfig) -> dict:
    """Replay CFR detections including empty frames; never substitute clocks silently."""
    by_frame: dict[int, list[Detection]] = {}
    for d in dets:
        if not math.isclose(d.t, d.frame_idx / clip.fps, rel_tol=0, abs_tol=1e-5):
            raise ValueError(
                f"{clip.video}: realtime benchmark requires constant-rate timestamps "
                "starting at zero; use the live video replay for variable-rate footage"
            )
        frame = by_frame.setdefault(d.frame_idx, [])
        if frame and frame[0].t != d.t:
            raise ValueError(f"{clip.video}: detections in frame {d.frame_idx} disagree on t")
        frame.append(d)
    clock = _recorded_clock({i: ds[0].t for i, ds in by_frame.items()},
                            clip.frame_count, clip.fps)
    analyzer = RealtimeAnalyzer(cfg)
    for idx, t in enumerate(clock):
        analyzer.feed(by_frame.get(idx, []), t)
    final = analyzer.finalize()
    return {"runs": final.runs_completed, "catches": final.catches_total,
            "drops": final.drops_total}


def _summarize(reports: list[EvalReport]) -> dict:
    n = sum(r.n_labeled_runs for r in reports)
    catch_ok = sum(m.catch_error <= 1 for r in reports for m in r.matches)
    iou_ok = sum(m.iou >= 0.9 for r in reports for m in r.matches)
    tp = sum(r.drop_tp for r in reports)
    fp = sum(r.drop_fp for r in reports)
    fn = sum(r.drop_fn for r in reports)
    return {
        "n_videos": len(reports), "n_labeled_runs": n,
        "n_pred_runs": sum(r.n_pred_runs for r in reports),
        "unmatched_labeled": sum(len(r.unmatched_labeled) for r in reports),
        "unmatched_pred": sum(len(r.unmatched_pred) for r in reports),
        "frac_catch_within_1": catch_ok / n if n else None,
        "frac_runs_iou90": iou_ok / n if n else None,
        "drop_tp": tp, "drop_fp": fp, "drop_fn": fn,
        "drop_precision": tp / (tp + fp) if tp + fp else None,
        "drop_recall": tp / (tp + fn) if tp + fn else None,
    }


def run_benchmark(
    manifest_path: str | Path, *, config: AnalyzeConfig | None = None,
    realtime: bool = False,
) -> dict:
    """Score all clips, preserving missing runs in the aggregate denominator.

    ``human_reviewed`` records the manifest author's assertion of label
    provenance. It cannot establish that human review actually happened.
    Generated oracle events belong in diagnostic analyses, not this benchmark.
    """
    path = Path(manifest_path).resolve()
    manifest = BenchmarkManifest.model_validate_json(path.read_text())
    cfg = config or AnalyzeConfig()
    live_cfg = RealtimeConfig(analyze=cfg) if realtime else None
    reports: list[EvalReport] = []
    clips: list[dict] = []
    seen_inputs: set[Path] = set()
    for clip in manifest.clips:
        det_path = (path.parent / clip.detections).resolve()
        label_path = (path.parent / clip.labels).resolve()
        if det_path in seen_inputs:
            raise ValueError(f"{clip.video}: duplicate detection input in benchmark")
        seen_inputs.add(det_path)
        labels = VideoLabels.model_validate_json(label_path.read_text())
        if labels.video != clip.video:
            raise ValueError(f"{clip.video}: labels identify a different video: {labels.video}")
        duration = clip.frame_count / clip.fps
        if any(r.start_t < 0 for r in labels.runs):
            raise ValueError(f"{clip.video}: labeled run starts before the video")
        if any(r.end_t > duration for r in labels.runs):
            raise ValueError(f"{clip.video}: labeled run ends after the video")
        if any(not 0 <= t <= duration for t in labels.drops):
            raise ValueError(f"{clip.video}: labeled drop lies outside the video")
        dets = load_detections_jsonl(det_path)
        _validate_detections(dets, clip)
        session = analyze_detections(dets, cfg)
        report = evaluate_session(session, labels)
        reports.append(report)
        result = {
            "video": clip.video, "label_source": clip.label_source,
            "detector": clip.detector, "fps": clip.fps, "frame_count": clip.frame_count,
            "detections_sha256": _sha256(det_path), "labels_sha256": _sha256(label_path),
            "offline": report.model_dump(mode="json"),
            "offline_catches": sum(r.catches for r in session.runs),
            "labeled_catches": sum(r.catches for r in labels.runs),
        }
        if live_cfg is not None:
            live = _replay(dets, clip, live_cfg)
            live["catch_error"] = abs(live["catches"] - result["labeled_catches"])
            live["catch_delta_offline"] = live["catches"] - result["offline_catches"]
            result["realtime"] = live
        clips.append(result)

    return {
        "schema_version": "1", "session_schema_version": SCHEMA_VERSION,
        "package_version": version("juggletrack"),
        "implementation_sha256": _implementation_sha256(),
        "manifest_sha256": _sha256(path), "split": manifest.split,
        "analyze_config": cfg.model_dump(mode="json"),
        "realtime_config": live_cfg.model_dump(mode="json") if live_cfg else None,
        "summary": _summarize(reports), "clips": clips,
    }
