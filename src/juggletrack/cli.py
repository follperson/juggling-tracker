"""juggletrack CLI: analyze videos, evaluate against labels."""
from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


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
    stride: int = typer.Option(1, help="Detect every Nth frame"),
    device: str | None = typer.Option(None, help="Torch device (mps/cpu/cuda)"),
) -> None:
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
                       detections=dets_for_overlay)

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


if __name__ == "__main__":
    app()
