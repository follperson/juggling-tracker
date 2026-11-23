"""Offline video pipeline: video -> detections -> event core -> artifacts.

Intermediates are persisted (spec: stages re-runnable independently) —
`detections.jsonl` lets you re-tune the event core without re-detecting.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

from pydantic import ValidationError

from juggletrack.analyze import AnalyzeConfig, analyze_detections
from juggletrack.detect import BallDetector
from juggletrack.types import Detection, SessionResult
from juggletrack.video.reader import VideoInfo, VideoReader


def detect_frames(
    reader: VideoReader, detector: BallDetector, stride: int
) -> tuple[list[Detection], int]:
    """Return the detections and the number of frames decoded."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    dets: list[Detection] = []
    decoded = 0
    for idx, t, frame in reader.frames():
        decoded += 1
        if idx % stride:
            continue
        dets.extend(detector.detect(frame, idx, t))
    return dets, decoded


def detect_video(
    reader: VideoReader, detector: BallDetector, *, stride: int = 1
) -> list[Detection]:
    return detect_frames(reader, detector, stride)[0]


def save_detections_jsonl(dets: list[Detection], path: str | Path) -> None:
    with open(path, "w") as f:
        for d in dets:
            f.write(d.model_dump_json() + "\n")


_FINITE_FIELDS = ("t", "x", "y", "w", "h", "confidence")


def load_detections_jsonl(path: str | Path) -> list[Detection]:
    """Raise ValueError naming the file and line of a row that does not parse,
    or whose time, position, size or confidence is NaN or infinite."""
    dets: list[Detection] = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                d = Detection.model_validate_json(line)
            except ValidationError as exc:
                err = exc.errors()[0]
                field = ".".join(map(str, err["loc"]))
                raise ValueError(
                    f"{path}:{lineno}: {field + ': ' if field else ''}{err['msg']}"
                ) from exc
            bad = next((name for name in _FINITE_FIELDS if not math.isfinite(getattr(d, name))),
                       None)
            if bad is not None:
                raise ValueError(f"{path}:{lineno}: {bad} is not finite ({getattr(d, bad)})")
            dets.append(d)
    return dets


def check_frames_in_video(dets: list[Detection], frame_count: int) -> None:
    """Raise ValueError when a detection's frame lies outside [0, frame_count).

    This catches detections from a longer video, not every mismatched pair.
    """
    bad = next((d.frame_idx for d in dets if not 0 <= d.frame_idx < frame_count), None)
    if bad is not None:
        raise ValueError(
            f"detection at frame {bad} is outside the video's {frame_count} frames; "
            "were these detections made from this video?"
        )


class OfflineRun(NamedTuple):
    session: SessionResult
    detections: list[Detection]
    info: VideoInfo


def run_offline(
    video_path: str | Path,
    source: BallDetector | list[Detection],
    *,
    config: AnalyzeConfig | None = None,
    out_dir: str | Path | None = None,
    save_intermediates: bool = False,
    stride: int = 1,
) -> OfflineRun:
    """Detect (or take saved detections), analyse, and write artifacts.

    Saved detections are taken as they are: only the video's header is read
    and no frame is decoded. A detector run records ``frames_decoded`` in
    meta beside the header's ``frame_count``.
    """
    frames_decoded = None
    with VideoReader(video_path) as reader:
        info = reader.info
        if isinstance(source, list):
            dets = sorted(
                (d for d in source if d.frame_idx % stride == 0), key=lambda d: d.frame_idx
            )
        else:
            dets, frames_decoded = detect_frames(reader, source, stride)

    session = analyze_detections(dets, config)
    session.meta.update(
        video_path=info.path, fps=info.fps, width=info.width, height=info.height,
        frame_count=info.frame_count, duration_s=info.duration, stride=stride,
    )
    if frames_decoded is not None:
        session.meta["frames_decoded"] = frames_decoded

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "analysis.json").write_text(session.model_dump_json(indent=2))
        if save_intermediates:
            save_detections_jsonl(dets, out / "detections.jsonl")
    return OfflineRun(session, dets, info)


def analyze_video(
    video_path: str | Path,
    detector: BallDetector,
    *,
    config: AnalyzeConfig | None = None,
    out_dir: str | Path | None = None,
    save_intermediates: bool = False,
    stride: int = 1,
) -> SessionResult:
    return run_offline(
        video_path, detector, config=config, out_dir=out_dir,
        save_intermediates=save_intermediates, stride=stride,
    ).session
