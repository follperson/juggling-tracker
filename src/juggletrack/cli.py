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


def _resolve_model(model: str | None) -> str:
    from juggletrack.detect.weights import resolve_model

    try:
        return resolve_model(model)
    except FileNotFoundError as exc:
        # Only the default can be missing: explicit names go to Ultralytics.
        raise typer.BadParameter(str(exc)) from exc


def _check_report_output(out: Path, inputs: list[Path]) -> None:
    if out.is_dir():
        raise ValueError("report output must be a file")
    for source in inputs:
        if out.resolve() == source.resolve() or (
            out.exists() and source.exists() and out.samefile(source)
        ):
            raise ValueError(f"report output must not overwrite an input: {source}")


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
    model: str | None = typer.Option(None, help="Weights (default: env or local v3 model)"),
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
            model_path=_resolve_model(model), conf=conf, imgsz=imgsz, device=device
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


@app.command()
def benchmark(
    manifest: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path = typer.Option(..., help="Write benchmark report JSON here"),
    config: Path | None = typer.Option(None, exists=True, dir_okay=False,
                                     help="AnalyzeConfig JSON (otherwise shipped defaults)"),
    realtime: bool = typer.Option(False, help="Also replay constant-rate detections frame by frame"),
) -> None:
    """Benchmark saved detections against independently reviewed event labels."""
    from juggletrack.analyze import AnalyzeConfig
    from juggletrack.eval.benchmark import BenchmarkManifest, run_benchmark

    try:
        manifest = manifest.resolve()
        spec = BenchmarkManifest.model_validate_json(manifest.read_text())
        inputs = [manifest] + ([config] if config else [])
        inputs += [manifest.parent / path for c in spec.clips for path in (c.detections, c.labels)]
        _check_report_output(out, inputs)
        cfg = AnalyzeConfig.model_validate_json(config.read_text()) if config else None
        report = run_benchmark(manifest, config=cfg, realtime=realtime)
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False))
    summary = report["summary"]
    score = summary["frac_catch_within_1"]
    formatted = f"{score:.1%}" if score is not None else "n/a (no labeled runs)"
    typer.echo(
        f"{summary['n_videos']} videos, {summary['n_labeled_runs']} labeled runs; "
        f"catch count within +/-1: {formatted}; "
        f"missed runs {summary['unmatched_labeled']}, extra runs {summary['unmatched_pred']} "
        f"-> {out}"
    )


def _get_detections(video, detections, model, conf, imgsz, stride, device):
    from juggletrack.pipeline.offline import detect_video, load_detections_jsonl
    from juggletrack.video.reader import VideoReader

    if detections is not None:
        return load_detections_jsonl(detections)
    from juggletrack.detect.yolo import YOLODetector

    det = YOLODetector(model_path=_resolve_model(model), conf=conf, imgsz=imgsz, device=device)
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
    model: str | None = typer.Option(None, help="Weights (default: env or local v3 model)"),
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
    model: str | None = typer.Option(None, help="Weights (default: env or local v3 model)"),
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
def detection_eval(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    detections: Path = typer.Argument(..., exists=True, dir_okay=False),
    centers_csv: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path = typer.Option(..., help="Write center-metric report JSON here"),
    tolerance_px: float = typer.Option(10.0, min=0.001, help="Center matching distance in pixels"),
    stride: int = typer.Option(1, min=1, help="Use the same frame stride as the detector"),
    conf: float = typer.Option(0.0, min=0, max=1, help="Filter saved predictions by confidence"),
) -> None:
    """Score detections against Meschke ball centers; no event oracle or model needed."""
    import csv
    import hashlib

    from juggletrack.data.meschke_import import CITATION, load_meschke_trajectories
    from juggletrack.eval.detection_metrics import evaluate_ball_centers
    from juggletrack.pipeline.offline import load_detections_jsonl
    from juggletrack.video.reader import VideoReader

    try:
        _check_report_output(out, [video, detections, centers_csv])
        with VideoReader(video) as reader:
            info = reader.info
        with centers_csv.open(newline="") as stream:
            rows = csv.reader(stream)
            next(rows, None)
            n_rows = sum(bool(row) for row in rows)
        if abs(n_rows - info.frame_count) > 2:
            raise ValueError("ground-truth row count and video frame count differ by more than 2")
        frame_count = min(n_rows, info.frame_count)
        targets = load_meschke_trajectories(
            centers_csv, fps=info.fps, width=info.width, height=info.height,
        )
        dets = load_detections_jsonl(detections)
        if any(not 0 <= d.frame_idx < info.frame_count for d in dets):
            raise ValueError("prediction frame index is outside the video frame range")
        report = evaluate_ball_centers(
            [d for d in dets if d.frame_idx < frame_count],
            [d for trajectory in targets for d in trajectory if d.frame_idx < frame_count],
            width=info.width, height=info.height, frame_count=frame_count,
            tolerance_px=tolerance_px, stride=stride, min_conf=conf,
        )
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    report.update(video=str(video), citation=CITATION,
                  frames_trimmed=abs(n_rows - info.frame_count))
    for name, path in (("detections", detections), ("centers", centers_csv)):
        with path.open("rb") as stream:
            report[f"{name}_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False))
    precision = f"{report['precision']:.1%}" if report["precision"] is not None else "n/a"
    recall = f"{report['recall']:.1%}" if report["recall"] is not None else "n/a"
    typer.echo(f"center precision {precision}, recall {recall}; "
               f"duplicate candidates {report['duplicate_fp']}, "
               f"other false positives {report['background_fp']} -> {out}")


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


@app.command()
def export(
    weights: Path = typer.Argument(..., exists=True, dir_okay=False),
    imgsz: int = typer.Option(640),
    half: bool = typer.Option(True, "--half/--no-half"),
) -> None:
    """Export detector weights to CoreML (.mlpackage) for the Mac live path."""
    from juggletrack.train.export import export_coreml

    typer.echo(f"exported: {export_coreml(weights, imgsz=imgsz, half=half)}")


@app.command()
def live(
    source: str = typer.Argument("0", help="Webcam index (digits) or video file path"),
    model: str | None = typer.Option(
        None, help="Weights (default: env or local v3 model)",
    ),
    detections: Path | None = typer.Option(
        None, exists=True, dir_okay=False,
        help="Replay saved detections.jsonl (file sources only; no detector run)",
    ),
    conf: float = typer.Option(0.05),
    imgsz: int = typer.Option(640),
    device: str | None = typer.Option(None),
    display: bool = typer.Option(True, "--display/--no-display"),
    max_frames: int | None = typer.Option(None),
    out: Path | None = typer.Option(None, help="Write live_session.json here"),
) -> None:
    """Live juggling tracker: webcam or file, realtime HUD."""
    from juggletrack.pipeline.live import run_live
    from juggletrack.pipeline.realtime import RealtimeConfig

    # S4: str.isdigit() is true for various non-ASCII digit characters
    # (e.g. superscripts like '²', circled digits like '①', full-width
    # digits like '１'), some of which int() then rejects (ValueError) and
    # some of which int() silently converts (e.g. '３' -> 3) -- neither is
    # the intended "webcam index" parse for a CLI argument. Require ASCII
    # digits first so `source` only ever takes the int() branch for what a
    # user actually typed as an ordinary webcam index.
    src: int | str = int(source) if source.isascii() and source.isdigit() else source
    # S7: --detections replays a saved detections.jsonl keyed by frame_idx
    # against file-source frame timestamps (see FakeDetector); a webcam
    # index has no frames to replay against.
    if detections is not None and isinstance(src, int):
        raise typer.BadParameter("--detections requires a file source")
    if detections is not None:
        from juggletrack.detect.fake import FakeDetector
        from juggletrack.pipeline.offline import load_detections_jsonl

        detector = FakeDetector(load_detections_jsonl(detections))
    else:
        from juggletrack.detect.yolo import YOLODetector

        detector = YOLODetector(
            model_path=_resolve_model(model), conf=conf, imgsz=imgsz, device=device,
        )

    final, fps = run_live(
        src, detector, config=RealtimeConfig(),
        display=display, max_frames=max_frames,
    )
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "live_session.json").write_text(final.model_dump_json(indent=2))
    typer.echo(
        f"{final.runs_completed} runs, {final.catches_total} catches, "
        f"{final.drops_total} drops @ {fps:.0f} fps"
    )


if __name__ == "__main__":
    app()
