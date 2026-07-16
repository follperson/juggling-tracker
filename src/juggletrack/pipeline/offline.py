"""Offline video pipeline: video -> detections -> event core -> artifacts.

Intermediates are persisted (spec: stages re-runnable independently) —
`detections.jsonl` lets you re-tune the event core without re-detecting.
"""
from __future__ import annotations

from pathlib import Path

from juggletrack.analyze import AnalyzeConfig, analyze_detections
from juggletrack.detect import BallDetector
from juggletrack.types import Detection, SessionResult
from juggletrack.video.reader import VideoReader


def detect_video(
    reader: VideoReader, detector: BallDetector, *, stride: int = 1
) -> list[Detection]:
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    dets: list[Detection] = []
    for idx, t, frame in reader.frames():
        if idx % stride:
            continue
        dets.extend(detector.detect(frame, idx, t))
    return dets


def save_detections_jsonl(dets: list[Detection], path: str | Path) -> None:
    with open(path, "w") as f:
        for d in dets:
            f.write(d.model_dump_json() + "\n")


def load_detections_jsonl(path: str | Path) -> list[Detection]:
    with open(path) as f:
        return [Detection.model_validate_json(line) for line in f if line.strip()]


def analyze_video(
    video_path: str | Path,
    detector: BallDetector,
    *,
    config: AnalyzeConfig | None = None,
    out_dir: str | Path | None = None,
    save_intermediates: bool = False,
    stride: int = 1,
) -> SessionResult:
    with VideoReader(video_path) as reader:
        info = reader.info
        dets = detect_video(reader, detector, stride=stride)

    session = analyze_detections(dets, config)
    session.meta.update(
        video_path=info.path, fps=info.fps, width=info.width, height=info.height,
        frame_count=info.frame_count, duration_s=info.duration, stride=stride,
    )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "analysis.json").write_text(session.model_dump_json(indent=2))
        if save_intermediates:
            save_detections_jsonl(dets, out / "detections.jsonl")
    return session
