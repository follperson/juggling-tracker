"""juggletrack CLI: analyze videos, evaluate against labels."""
from __future__ import annotations

import enum
import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


class DetectorBackend(str, enum.Enum):
    yolo = "yolo"
    motion = "motion"


class DotsMode(str, enum.Enum):
    verified = "verified"
    all = "all"
    none = "none"


@app.command()
def analyze(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path | None = typer.Option(None, help="Output dir (default outputs/<stem>)"),
    overlay: bool = typer.Option(True, help="Render debug overlay video"),
    save_intermediates: bool = typer.Option(False, help="Save detections.jsonl"),
    detections: Path | None = typer.Option(
        None, exists=True, dir_okay=False,
        help="Replay saved detections.jsonl instead of running the detector",
    ),
    model: str = typer.Option("yolo11n.pt", help="YOLO weights path or name"),
    conf: float = typer.Option(0.05, help="Detector confidence threshold"),
    imgsz: int = typer.Option(640, help="Detector input size"),
    stride: int = typer.Option(1, min=1, help="Detect every Nth frame"),
    device: str | None = typer.Option(None, help="Torch device (mps/cpu/cuda)"),
    link_max_dist: float = typer.Option(
        0.08, help="Linker max normalized distance per step (extract_arcs knob)"
    ),
    dots: DotsMode = typer.Option(
        DotsMode.verified,
        help="Overlay dot filtering: 'verified' (default, arc-assigned "
        "detections only), 'all' (every raw detection), or 'none'",
    ),
) -> None:
    from juggletrack.analyze import AnalyzeConfig
    from juggletrack.pipeline.offline import analyze_video, load_detections_jsonl

    out_dir = out or Path("outputs") / video.stem
    if detections is not None:
        from juggletrack.detect.fake import FakeDetector

        detector = FakeDetector(load_detections_jsonl(detections))
    else:
        from juggletrack.detect.yolo import YOLODetector

        detector = YOLODetector(
            model_path=model, conf=conf, imgsz=imgsz, device=device
        )

    session = analyze_video(
        video, detector,
        out_dir=out_dir, save_intermediates=save_intermediates, stride=stride,
        config=AnalyzeConfig(link_max_dist=link_max_dist),
    )

    if overlay:
        from juggletrack.pipeline.overlay import render_overlay

        dets_for_overlay = None
        jsonl = out_dir / "detections.jsonl"
        if detections is not None:
            dets_for_overlay = load_detections_jsonl(detections)
        elif jsonl.exists():
            dets_for_overlay = load_detections_jsonl(jsonl)
        render_overlay(video, session, out_dir / "overlay.mp4",
                       detections=dets_for_overlay, dots=dots.value)

    for i, run in enumerate(session.runs):
        typer.echo(
            f"run {i + 1}: {run.start_t:.2f}-{run.end_t:.2f}s  "
            f"catches {run.catches}  end={run.end_reason}"
        )
    typer.echo(
        f"{len(session.runs)} run(s), {len(session.drops)} drop(s); "
        f"results in {out_dir}"
    )


@app.command()
def eval(
    analysis_json: Path = typer.Argument(..., exists=True, dir_okay=False),
    labels_json: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path | None = typer.Option(None, help="Write EvalReport JSON here"),
) -> None:
    from juggletrack.eval.labels import VideoLabels
    from juggletrack.eval.metrics import evaluate_session
    from juggletrack.types import SessionResult

    session = SessionResult.model_validate(json.loads(analysis_json.read_text()))
    labels = VideoLabels.model_validate_json(labels_json.read_text())
    report = evaluate_session(session, labels)

    typer.echo(f"video: {report.video}")
    typer.echo(f"runs: {report.n_pred_runs} predicted / {report.n_labeled_runs} labeled")
    typer.echo(f"run boundary IoU>=0.9: {report.frac_runs_iou90:.0%}")
    typer.echo(f"catch count within +/-1: {report.frac_catch_within_1:.0%}")
    typer.echo(
        f"drops: precision {report.drop_precision:.2f} "
        f"recall {report.drop_recall:.2f} "
        f"(tp={report.drop_tp} fp={report.drop_fp} fn={report.drop_fn})"
    )
    if out is not None:
        out.write_text(report.model_dump_json(indent=2))


def _get_detections(video, detections, model, conf, imgsz, stride, device):
    from juggletrack.pipeline.offline import detect_video, load_detections_jsonl
    from juggletrack.video.reader import VideoReader

    if detections is not None:
        return load_detections_jsonl(detections)
    from juggletrack.detect.yolo import YOLODetector

    det = YOLODetector(model_path=model, conf=conf, imgsz=imgsz, device=device)
    with VideoReader(video) as reader:
        return detect_video(reader, det, stride=stride)


@app.command()
def label(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path | None = typer.Option(None, help="Output dir (default outputs/labels/<stem>)"),
    detections: Path | None = typer.Option(
        None, exists=True, dir_okay=False,
        help="Replay saved detections.jsonl instead of running the detector",
    ),
    detector: DetectorBackend = typer.Option(
        DetectorBackend.yolo,
        help="Detector backend. 'motion' (MOG2 background subtraction) is a "
        "cold-start bootstrap for footage where the appearance detector fails "
        "(static camera required); it IGNORES --model/--conf/--imgsz. Motion "
        "labels also get their box sizes physics-calibrated (see "
        "calibrate_label_boxes) before export, since raw motion-blob boxes "
        "underestimate ball size and vary with speed.",
    ),
    model: str = typer.Option("yolo11n.pt"),
    conf: float = typer.Option(0.05),
    imgsz: int = typer.Option(640),
    stride: int = typer.Option(1, min=1),
    device: str | None = typer.Option(None),
    link_max_dist: float = typer.Option(
        0.08, help="Linker max normalized distance per step (extract_arcs knob)"
    ),
    negatives: Path | None = typer.Option(
        None,
        help="Also mine arc-rejected junk detections into zero-box hard-negative "
        "images (COCO, same dets/arcs, no extra detection pass) at this dir",
    ),
    neg_person_model: str = typer.Option(
        "yolo11n.pt",
        help="Stock COCO model for the hard-negative person-region veto "
        "(rejects candidate frames whose 'junk' sits on the juggler -- held "
        "balls and unstitched face/body crossings; turn-4 contamination "
        "postmortem). Runs only on the few frames that survive the pure "
        "gates. 'none' disables the veto (NOT recommended for training data).",
    ),
) -> None:
    """Auto-label a video: arc-verified detections become COCO 'ball' boxes."""
    from juggletrack.arcs.extract import extract_arcs
    from juggletrack.data.autolabel import (
        calibrate_label_boxes,
        export_hard_negatives,
        export_video_labels,
        select_autolabels,
    )

    out_dir = out or Path("outputs") / "labels" / video.stem
    if detector == DetectorBackend.motion:
        if stride > 1:
            typer.echo(
                "warning: --detector motion degrades at stride>1 (the "
                "background model expects consecutive frames); recommend "
                "--stride 1"
            )
        if detections is not None:
            from juggletrack.pipeline.offline import load_detections_jsonl

            dets = load_detections_jsonl(detections)
        else:
            from juggletrack.detect.motion import MotionDetector
            from juggletrack.pipeline.offline import detect_video
            from juggletrack.video.reader import VideoReader

            with VideoReader(video) as reader:
                dets = detect_video(reader, MotionDetector(), stride=stride)
    else:
        dets = _get_detections(video, detections, model, conf, imgsz, stride, device)
    arcs = extract_arcs(dets, link_max_dist=link_max_dist)
    labels, review = select_autolabels(dets, arcs)
    if detector == DetectorBackend.motion:
        labels = calibrate_label_boxes(labels, arcs)
    stats = export_video_labels(video, labels, out_dir, review_frames=review)
    typer.echo(
        f"{stats['n_images']} images, {stats['n_boxes']} boxes, "
        f"{stats['n_review_frames']} review frames -> {out_dir}"
    )
    if negatives is not None:
        neg_stats = export_hard_negatives(
            video, dets, arcs, negatives,
            person_model=None if neg_person_model.lower() == "none" else neg_person_model,
        )
        typer.echo(
            f"negatives: {neg_stats['n_images']} images "
            f"({neg_stats['n_candidate_frames']} candidates, "
            f"{neg_stats['n_skipped_ambiguous']} skipped ambiguous, "
            f"{neg_stats['n_skipped_active']} skipped active, "
            f"{neg_stats['n_skipped_transient']} skipped transient, "
            f"{neg_stats['n_skipped_floor']} skipped floor, "
            f"{neg_stats['n_skipped_person']} skipped person"
            + (f", {neg_stats['n_skipped_no_arcs']} skipped no-arcs"
               if neg_stats["n_skipped_no_arcs"] else "")
            + f") -> {negatives}"
        )


@app.command()
def coverage(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    detections: Path | None = typer.Option(
        None, exists=True, dir_okay=False,
        help="Score saved detections.jsonl (pass the SAME --stride it was made with)",
    ),
    model: str = typer.Option("yolo11n.pt"),
    conf: float = typer.Option(0.05),
    imgsz: int = typer.Option(640),
    stride: int = typer.Option(1, min=1),
    device: str | None = typer.Option(None),
) -> None:
    """Detection-coverage stats: the before/after fine-tuning metric."""
    import math
    import statistics
    from collections import Counter

    from juggletrack.video.reader import VideoReader

    dets = _get_detections(video, detections, model, conf, imgsz, stride, device)
    with VideoReader(video) as reader:
        n_sampled = math.ceil(reader.info.frame_count / stride)
    per_frame = Counter(d.frame_idx for d in dets)
    cov = {
        k: sum(1 for v in per_frame.values() if v >= k) / n_sampled if n_sampled else 0.0
        for k in (1, 2, 3)
    }
    typer.echo(f"detections {len(dets)} over {n_sampled} sampled frames")
    typer.echo(f"coverage >=1: {cov[1]:.1%}  >=2: {cov[2]:.1%}  >=3: {cov[3]:.1%}")
    if dets:
        typer.echo(
            f"median conf {statistics.median(d.confidence for d in dets):.3f}  "
            f"median w {statistics.median(d.w for d in dets):.4f}"
        )


@app.command()
def train(
    data_yaml: Path = typer.Argument(..., exists=True, dir_okay=False),
    model: str = typer.Option("yolo11n.pt", help="Base weights to fine-tune"),
    epochs: int = typer.Option(40, min=1),
    imgsz: int = typer.Option(640),
    device: str | None = typer.Option(None),
    project: Path = typer.Option(Path("runs/finetune")),
    name: str = typer.Option("juggletrack"),
) -> None:
    """Fine-tune the ball detector on an assembled dataset."""
    from juggletrack.train.finetune import train_detector

    best = train_detector(
        data_yaml, base_model=model, epochs=epochs, imgsz=imgsz,
        device=device, project=project, name=name,
    )
    typer.echo(f"best weights: {best}")


if __name__ == "__main__":
    app()
