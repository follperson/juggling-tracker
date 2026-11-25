"""juggletrack CLI: analyze videos, evaluate against labels."""
from __future__ import annotations

import enum
import json
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from juggletrack.types import Detection

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _print_version(value: bool) -> None:
    if value:
        from importlib.metadata import version

        typer.echo(f"juggletrack {version('juggletrack')}")
        raise typer.Exit()


@app.callback()
def main(
    _version: bool = typer.Option(
        False, "--version", callback=_print_version, is_eager=True,
        help="Print the installed version and exit",
    ),
) -> None:
    pass


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
        raise typer.BadParameter(str(exc)) from exc


def _is_local_path(source: str) -> bool:
    """False for what OpenCV opens without a file of that name: a URL or an image-sequence pattern."""
    return "://" not in source and "%" not in source


def _load_replay(detections: Path, video: Path | str | None) -> list[Detection]:
    """Load saved detections. With a local video, also check they fit inside it."""
    from juggletrack.pipeline.offline import check_frames_in_video, load_detections_jsonl
    from juggletrack.video.reader import VideoReader

    frame_count = None
    if video is not None:
        with VideoReader(video) as reader:
            frame_count = reader.info.frame_count
    try:
        dets = load_detections_jsonl(detections)
        if frame_count is not None:
            check_frames_in_video(dets, frame_count)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--detections") from exc
    return dets


def _warn_if_short(video: Path, decoded: int, frame_count: int) -> None:
    if decoded < 0.9 * frame_count:
        typer.echo(
            f"warning: decoded {decoded} of the {frame_count} frames the header of {video} "
            "reports; the file may be cut short or its header may be stale",
            err=True,
        )


def _detect(video: Path, detector, stride: int) -> tuple[list[Detection], int]:
    """Return the detections and the number of frames decoded, warning on a short read."""
    from juggletrack.pipeline.offline import detect_frames
    from juggletrack.video.reader import VideoReader

    with VideoReader(video) as reader:
        dets, decoded = detect_frames(reader, detector, stride)
        frame_count = reader.info.frame_count
    _warn_if_short(video, decoded, frame_count)
    return dets, decoded


def _yolo_detector(model, conf, imgsz, device):
    from juggletrack.detect.yolo import YOLODetector

    return YOLODetector(model_path=_resolve_model(model), conf=conf, imgsz=imgsz, device=device)


def _check_out_dir(out: Path) -> Path:
    """Fail unless --out is, or could be created as, a writable directory. Create nothing."""
    # lexists and isdir never raise: a dangling link is there as far as mkdir
    # is concerned, and a parent that may not be searched is not writable.
    existing = next(p for p in (out, *out.parents) if os.path.lexists(p))
    if not os.path.isdir(existing):
        raise typer.BadParameter(f"cannot write {out}: {existing} is not a directory",
                                 param_hint="--out")
    if not os.access(existing, os.W_OK | os.X_OK):
        raise typer.BadParameter(f"cannot write {out}: {existing} is not writable",
                                 param_hint="--out")
    return out


def _check_report_output(out: Path, inputs: list[Path]) -> None:
    if out.is_dir():
        raise ValueError("report output must be a file")
    for source in inputs:
        if out.resolve() == source.resolve() or (
            out.exists() and source.exists() and out.samefile(source)
        ):
            raise ValueError(f"report output must not overwrite an input: {source}")


@contextmanager
def _atomic_report(out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as tmp:
        path = Path(tmp) / "report"
        yield path
        path.replace(out)


@app.command()
def analyze(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path | None = typer.Option(None, help="Output directory (default outputs/<stem>)"),
    overlay: bool = typer.Option(True, help="Render debug overlay video"),
    save_intermediates: bool = typer.Option(False, help="Save detections.jsonl"),
    detections: Path | None = typer.Option(
        None, exists=True, dir_okay=False,
        help="Replay saved detections.jsonl instead of running the detector",
    ),
    model: str | None = typer.Option(None, help="Weights (default: env or local v3 model)"),
    conf: float = typer.Option(0.05, min=0, max=1, help="Detector confidence threshold"),
    imgsz: int = typer.Option(640, min=32, help="Detector input size"),
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
    """Find juggling runs, catches and drops in a video; write analysis.json and an overlay."""
    from pydantic import ValidationError

    from juggletrack.analyze import AnalyzeConfig
    from juggletrack.pipeline.offline import run_offline

    try:
        config = AnalyzeConfig(link_max_dist=link_max_dist)
    except ValidationError as exc:
        raise typer.BadParameter(exc.errors()[0]["msg"], param_hint="--link-max-dist") from exc
    out_dir = _check_out_dir(out or Path("outputs") / video.stem)
    if detections is not None:
        source = _load_replay(detections, video)
    else:
        from juggletrack.detect.yolo import YOLODetector

        source = YOLODetector(
            model_path=_resolve_model(model), conf=conf, imgsz=imgsz, device=device
        )

    session, analysed, info = run_offline(
        video, source,
        out_dir=out_dir, save_intermediates=save_intermediates, stride=stride,
        config=config,
    )
    if "frames_decoded" in session.meta:
        _warn_if_short(video, session.meta["frames_decoded"], info.frame_count)

    if overlay:
        from juggletrack.pipeline.overlay import render_overlay

        render_overlay(video, session, out_dir / "overlay.mp4",
                       detections=analysed, dots=dots.value)

    for i, run in enumerate(session.runs):
        # A negative start_t is a throw extrapolated to before the first frame.
        span = f"{max(run.start_t, 0.0):.2f}s to {run.end_t:.2f}s"
        if run.start_t < 0:
            span += " (began before the clip)"
        typer.echo(f"run {i + 1}: {span}  catches {run.catches}  end={run.end_reason}")
    typer.echo(
        f"total: {sum(r.catches for r in session.runs)} catches, "
        f"{sum(r.throws for r in session.runs)} throws"
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
    """Score a saved analysis.json against hand-labeled runs and drops."""
    from juggletrack.eval.labels import VideoLabels
    from juggletrack.eval.metrics import (
        CATCH_ERROR_PASS,
        RUN_IOU_PASS,
        aggregate_scores,
        evaluate_session,
    )
    from juggletrack.types import SessionResult

    try:
        if out is not None:
            _check_report_output(out, [analysis_json, labels_json])
        document = json.loads(analysis_json.read_text())
        keys = document if isinstance(document, dict) else {}
        missing = [key for key in ("schema_version", "runs", "arcs") if key not in keys]
        if missing:
            raise ValueError(
                f"{analysis_json} has no {', '.join(missing)} key; expected an analysis.json "
                "written by juggletrack analyze"
            )
        session = SessionResult.model_validate(document)
        labels = VideoLabels.model_validate_json(labels_json.read_text())
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    report = evaluate_session(session, labels)
    scores = aggregate_scores([report])

    def fmt(key: str, spec: str) -> str:
        return "n/a" if scores[key] is None else format(scores[key], spec)

    typer.echo(f"video: {report.video}")
    typer.echo(f"runs: {report.n_pred_runs} predicted / {report.n_labeled_runs} labeled")
    typer.echo(f"run boundary IoU>={RUN_IOU_PASS}: {fmt('frac_runs_iou90', '.0%')}")
    typer.echo(f"catch count within +/-{CATCH_ERROR_PASS}: {fmt('frac_catch_within_1', '.0%')}")
    typer.echo(
        f"drops: precision {fmt('drop_precision', '.2f')} "
        f"recall {fmt('drop_recall', '.2f')} "
        f"(tp={report.drop_tp} fp={report.drop_fp} fn={report.drop_fn})"
    )
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
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
    from juggletrack.eval.metrics import CATCH_ERROR_PASS

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
        f"catch count within +/-{CATCH_ERROR_PASS}: {formatted}; "
        f"missed runs {summary['unmatched_labeled']}, extra runs {summary['unmatched_pred']} "
        f"-> {out}"
    )


def _get_detections(video, detections, model, conf, imgsz, stride, device):
    if detections is not None:
        return _load_replay(detections, video)
    return _detect(video, _yolo_detector(model, conf, imgsz, device), stride)[0]


@app.command()
def label(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path | None = typer.Option(
        None, help="Output directory (default outputs/labels/<stem>)",
    ),
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
    conf: float = typer.Option(0.05, min=0, max=1),
    imgsz: int = typer.Option(640, min=32),
    stride: int = typer.Option(1, min=1),
    device: str | None = typer.Option(None),
    link_max_dist: float = typer.Option(
        0.08, help="Linker max normalized distance per step (extract_arcs knob)"
    ),
    negatives: Path | None = typer.Option(
        None,
        help="Also mine arc-rejected junk detections into zero-box hard-negative "
        "images (COCO, same detections, no extra detection pass) at this dir; "
        "a frame that got a label is never a negative",
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
    from juggletrack.analyze import AnalyzeConfig
    from juggletrack.arcs.extract import dedup_parallel_arcs, extract_arcs
    from juggletrack.data.autolabel import (
        calibrate_label_boxes,
        export_hard_negatives,
        export_video_labels,
        select_autolabels,
    )
    from juggletrack.detect.cluster import cluster_detections

    out_dir = _check_out_dir(out or Path("outputs") / "labels" / video.stem)
    if detector == DetectorBackend.motion:
        if stride > 1:
            typer.echo(
                "warning: --detector motion degrades at stride>1 (the "
                "background model expects consecutive frames); recommend "
                "--stride 1"
            )
        if detections is not None:
            dets = _load_replay(detections, video)
        else:
            from juggletrack.detect.motion import MotionDetector

            dets = _detect(video, MotionDetector(), stride)[0]
    else:
        dets = _get_detections(video, detections, model, conf, imgsz, stride, device)
    cfg = AnalyzeConfig()
    label_dets = cluster_detections(dets, merge_dist=cfg.cluster_merge_dist)
    label_arcs = dedup_parallel_arcs(
        extract_arcs(label_dets, link_max_dist=link_max_dist),
        overlap_frac=cfg.arc_dedup_overlap_frac, traj_tol=cfg.arc_dedup_traj_tol,
    )
    labels, review = select_autolabels(label_dets, label_arcs)
    if detector == DetectorBackend.motion:
        labels = calibrate_label_boxes(labels, label_arcs)
    stats = export_video_labels(video, labels, out_dir, review_frames=review)
    typer.echo(
        f"{stats['n_images']} images, {stats['n_boxes']} boxes, "
        f"{stats['n_review_frames']} review frames -> {out_dir}"
    )
    if negatives is not None:
        neg_stats = export_hard_negatives(
            video, dets, extract_arcs(dets, link_max_dist=link_max_dist), negatives,
            person_model=None if neg_person_model.lower() == "none" else neg_person_model,
            labelled_frames={lb.frame_idx for lb in labels},
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
    conf: float = typer.Option(0.05, min=0, max=1),
    imgsz: int = typer.Option(640, min=32),
    stride: int = typer.Option(1, min=1),
    device: str | None = typer.Option(None),
) -> None:
    """Detection-coverage stats: the before/after fine-tuning metric."""
    import math
    import statistics
    from collections import Counter

    from juggletrack.video.reader import VideoReader

    if detections is not None:
        dets = _load_replay(detections, video)
        with VideoReader(video) as reader:
            n_frames = reader.info.frame_count
    else:
        dets, n_frames = _detect(video, _yolo_detector(model, conf, imgsz, device), stride)
    off_grid = next((d.frame_idx for d in dets if d.frame_idx % stride), None)
    if off_grid is not None:
        raise typer.BadParameter(
            f"detection at frame {off_grid} is not on the --stride {stride} grid; "
            "pass the stride the detections were made with",
            param_hint="--stride",
        )
    n_sampled = math.ceil(n_frames / stride)
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
def reviewed_detection_eval(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    detections: Path = typer.Argument(..., exists=True, dir_okay=False),
    labels: Path = typer.Argument(..., exists=True, dir_okay=False),
    tolerance_px: float = typer.Option(..., min=0.001, help="Match radius in displayed pixels"),
    out: Path = typer.Option(..., help="Write the reviewed-frame report here"),
    conf: float = typer.Option(0.0, min=0, max=1, help="Filter saved predictions by confidence"),
) -> None:
    """Score ball centers and optional visible box extents on reviewed frames."""
    from juggletrack.eval.benchmark import _implementation_sha256, _sha256
    from juggletrack.eval.reviewed_centers import (
        ReviewedCenters,
        evaluate_reviewed_centers,
        same_frame_time,
    )
    from juggletrack.pipeline.offline import load_detections_jsonl
    from juggletrack.video.reader import VideoReader

    try:
        _check_report_output(out, [video, detections, labels])
        review = ReviewedCenters.model_validate_json(labels.read_bytes())
        if _sha256(video) != review.source_sha256:
            raise ValueError("source video SHA-256 differs from the reviewed recording")
        dets = load_detections_jsonl(detections)
        remaining = {frame.frame_idx: frame.t for frame in review.frames}
        with VideoReader(video) as reader:
            if (reader.info.width, reader.info.height) != (review.width, review.height):
                raise ValueError("reviewed dimensions differ from the displayed video dimensions")
            if any(d.frame_idx >= reader.info.frame_count for d in dets):
                raise ValueError("detection frame index is outside the video")
            for idx, t, _ in reader.frames():
                if idx in remaining:
                    if not same_frame_time(remaining.pop(idx), t):
                        raise ValueError(f"review timestamp differs from source at frame {idx}")
                    if not remaining:
                        break
        if remaining:
            raise ValueError(f"video ended before reviewed frame {min(remaining)}")
        report = evaluate_reviewed_centers(dets, review, tolerance_px=tolerance_px, min_conf=conf)
        report.update(labels_sha256=_sha256(labels), detections_sha256=_sha256(detections),
                      implementation_sha256=_implementation_sha256())
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if "visible" in report["states"] or "box_localization" in report:
        typer.echo(f"Definite centers matched {report['tp']}/{report['tp'] + report['fn']}")
    if "box_localization" in report:
        boxes = report["box_localization"]
        typer.echo(f"Measured boxes with at least twice the labeled area "
                   f"{boxes['at_least_double_area']}/{boxes['with_predicted_extent']}")
    airborne = report["states"]["airborne"]
    typer.echo(f"Airborne centers matched {airborne['matched']}/{airborne['n']} "
               f"on {report['n_reviewed_frames']} reviewed frames -> {out}")


@app.command()
def train(
    data_yaml: Path = typer.Argument(..., exists=True, dir_okay=False),
    model: str = typer.Option("yolo11n.pt", help="Base weights to fine-tune"),
    epochs: int = typer.Option(40, min=1),
    imgsz: int = typer.Option(640),
    device: str | None = typer.Option(None),
    project: Path = typer.Option(
        Path("runs/finetune"), help="Run directory parent, relative to the current directory",
    ),
    name: str = typer.Option(
        "juggletrack", help="Run name; an existing run gets a numbered suffix, never overwritten",
    ),
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
    conf: float = typer.Option(0.05, min=0, max=1),
    imgsz: int = typer.Option(640, min=32),
    device: str | None = typer.Option(None),
    display: bool = typer.Option(True, "--display/--no-display"),
    max_frames: int | None = typer.Option(None, min=1),
    out: Path | None = typer.Option(None, help="Directory to write live_session.json into"),
) -> None:
    """Live juggling tracker: webcam or file, realtime HUD."""
    from juggletrack.pipeline.live import SourceOpenError, run_live
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
    local = isinstance(src, str) and _is_local_path(src)
    if detections is not None and local and not Path(src).exists():
        raise typer.BadParameter(f"file {src} does not exist", param_hint="source")
    if out is not None:
        _check_out_dir(out)
    if detections is not None:
        from juggletrack.detect.fake import FakeDetector

        detector = FakeDetector(_load_replay(detections, src if local else None))
    else:
        from juggletrack.detect.yolo import YOLODetector

        detector = YOLODetector(
            model_path=_resolve_model(model), conf=conf, imgsz=imgsz, device=device,
        )

    try:
        final, fps = run_live(
            src, detector, config=RealtimeConfig(),
            display=display, max_frames=max_frames,
        )
    except SourceOpenError as exc:
        if isinstance(src, int):
            message = f"{exc}; check the camera index and that this terminal may use the camera"
        elif local and not Path(src).exists():
            message = f"could not open source {src}: the file does not exist"
        else:
            message = f"could not open source {src}"
        raise typer.BadParameter(message, param_hint="source") from exc
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "live_session.json").write_text(final.model_dump_json(indent=2))
    typer.echo(
        f"{final.runs_completed} runs, {final.catches_total} catches, "
        f"{final.drops_total} drops @ {fps:.0f} fps"
    )



balls = typer.Typer(no_args_is_help=True, help="Review moving candidates for registered ball colors.")
app.add_typer(balls, name="balls")


def _pick_circle(image) -> tuple[int, int, int]:
    import sys

    import cv2

    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY")
                                                or os.environ.get("WAYLAND_DISPLAY")):
        raise ValueError("an interactive display is required for --pick; use --circle X Y R")
    h, w = image.shape[:2]
    scale = min(1.0, 1100/w, 750/h)
    preview = cv2.resize(image, (max(1, round(w*scale)), max(1, round(h*scale))))
    title = "Box one whole ball, then press Enter"
    try:
        x, y, bw, bh = cv2.selectROI(title, preview, showCrosshair=True, fromCenter=False)
    finally:
        cv2.destroyWindow(title)
    if not bw or not bh:
        raise ValueError("ball selection cancelled")
    sx, sy = w / preview.shape[1], h / preview.shape[0]
    return round((x+bw/2)*sx), round((y+bh/2)*sy), round(min(bw*sx, bh*sy)/2)


@balls.command("register")
def register_color(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    frame: int = typer.Option(..., min=0, help="Zero-based frame showing the whole ball"),
    circle: tuple[int, int, int] | None = typer.Option(
        None, help="Center X Y and full radius in upright source pixels",
    ),
    pick: bool = typer.Option(False, "--pick", help="Drag a box around one ball on screen"),
    out: Path = typer.Option(..., help="Write the reusable color profile JSON"),
) -> None:
    """Sample one uniformly colored ball at a known size."""
    import cv2

    from juggletrack.detect.registered import register_ball
    from juggletrack.video.reader import VideoReader

    if pick == bool(circle):
        raise typer.BadParameter("choose exactly one of --pick and --circle X Y R")
    try:
        _check_report_output(out, [video])
        with VideoReader(video) as reader:
            image = next((img for i, _, img in reader.frames() if i == frame), None)
        if image is None:
            raise ValueError(f"could not decode frame {frame}")
        profile = register_ball(image, *(_pick_circle(image) if pick else circle))
        with _atomic_report(out) as path:
            path.write_text(profile.model_dump_json(indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, cv2.error) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Saved ball profile to {out}")


@balls.command("propose")
def propose_colors(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    profile: list[Path] = typer.Option(
        ..., exists=True, dir_okay=False, help="Repeat for each ball color",
    ),
    out: Path = typer.Option(..., help="Write candidate detections as JSONL"),
) -> None:
    """Export offline candidates for visual review with analyze --detections."""
    import cv2

    from juggletrack.detect.registered import BallProfile, registered_proposals
    from juggletrack.pipeline.offline import save_detections_jsonl
    from juggletrack.video.reader import VideoReader

    try:
        _check_report_output(out, [video, *profile])
        profiles = []
        for source in profile:
            try:
                profiles.append(BallProfile.model_validate_json(source.read_bytes()))
            except ValueError as exc:
                raise ValueError(f"invalid ball profile {source}: {exc}") from exc
        with VideoReader(video) as reader, _atomic_report(out) as path:
            decoded = 0

            def frames():
                nonlocal decoded
                for row in reader.frames():
                    decoded += 1
                    yield row

            save_detections_jsonl(registered_proposals(frames(), profiles), path)
            if not decoded:
                raise ValueError("could not decode any frames")
            _warn_if_short(video, decoded, reader.info.frame_count)
            empty = path.stat().st_size == 0
    except (OSError, ValueError, cv2.error) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if empty:
        typer.echo("No moving candidates matched the registered profiles.")
    typer.echo(f"Saved candidates from {decoded} frames to {out}")


if __name__ == "__main__":
    app()
