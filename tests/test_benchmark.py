import json

import pytest
from typer.testing import CliRunner

from juggletrack.pipeline.offline import save_detections_jsonl
from juggletrack.sim import simulate_cascade


@pytest.fixture
def benchmark_manifest(tmp_path):
    sim = simulate_cascade(n_throws=8, fps=30.0, seed=1)
    save_detections_jsonl(sim.detections, tmp_path / "detections.jsonl")
    (tmp_path / "labels.json").write_text(json.dumps({
        "video": "cascade", "runs": [
            {"start_t": sim.run_start, "end_t": sim.run_end, "catches": 8},
            {"start_t": 20, "end_t": 25, "catches": 10},  # deliberately missed run
        ], "drops": [],
    }))
    manifest = tmp_path / "benchmark.json"
    manifest.write_text(json.dumps({
        "schema_version": "1", "split": "development", "clips": [{
            "video": "cascade", "detections": "detections.jsonl", "labels": "labels.json",
            "label_source": "human_reviewed", "fps": 30,
            "frame_count": 900,  # includes the second labeled run and idle tail
        }],
    }))
    return manifest


def test_benchmark_resolves_paths_and_penalizes_missing_runs(benchmark_manifest, monkeypatch):
    from juggletrack.eval.benchmark import run_benchmark

    monkeypatch.chdir(benchmark_manifest.parent.parent)
    result = run_benchmark(benchmark_manifest)
    summary = result["summary"]
    assert summary["n_labeled_runs"] == 2
    assert summary["n_pred_runs"] == 1
    assert summary["unmatched_labeled"] == 1
    assert summary["frac_catch_within_1"] == 0.5
    assert summary["frac_runs_iou90"] == 0.5
    assert result["split"] == "development"
    assert len(result["manifest_sha256"]) == 64
    assert len(result["implementation_sha256"]) == 64
    assert len(result["clips"][0]["detections_sha256"]) == 64


def test_benchmark_records_configuration_and_changed_inputs(benchmark_manifest):
    from juggletrack.analyze import AnalyzeConfig
    from juggletrack.eval.benchmark import run_benchmark

    first = run_benchmark(benchmark_manifest, config=AnalyzeConfig(min_arcs=100))
    assert first["analyze_config"]["min_arcs"] == 100
    assert first["summary"]["frac_catch_within_1"] == 0
    path = benchmark_manifest.parent / "labels.json"
    labels = json.loads(path.read_text())
    labels["runs"][0]["catches"] = 6
    path.write_text(json.dumps(labels))
    second = run_benchmark(benchmark_manifest)
    assert first["clips"][0]["labels_sha256"] != second["clips"][0]["labels_sha256"]


@pytest.mark.parametrize("change", ["duplicate", "generated", "invalid_span", "nan"])
def test_benchmark_rejects_invalid_references(benchmark_manifest, change):
    from juggletrack.eval.benchmark import run_benchmark

    manifest = json.loads(benchmark_manifest.read_text())
    if change == "duplicate":
        manifest["clips"].append(manifest["clips"][0])
    elif change == "generated":
        manifest["clips"][0]["label_source"] = "oracle_events"
    else:
        path = benchmark_manifest.parent / "labels.json"
        labels = json.loads(path.read_text())
        labels["runs"][0]["end_t"] = -10 if change == "invalid_span" else float("nan")
        path.write_text(json.dumps(labels))
    benchmark_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        run_benchmark(benchmark_manifest)


def _edit_labels(manifest, edit):
    path = manifest.parent / "labels.json"
    labels = json.loads(path.read_text())
    edit(labels)
    path.write_text(json.dumps(labels))


def _edit_first_detection(manifest, **fields):
    path = manifest.parent / "detections.jsonl"
    lines = path.read_text().splitlines()
    lines[0] = json.dumps({**json.loads(lines[0]), **fields})
    path.write_text("\n".join(lines))


def _reuse_detections_for_second_video(manifest):
    spec = json.loads(manifest.read_text())
    (manifest.parent / "copy-labels.json").write_text(
        json.dumps({"video": "cascade-copy", "runs": [], "drops": []}))
    spec["clips"].append({**spec["clips"][0], "video": "cascade-copy",
                          "labels": "copy-labels.json"})
    manifest.write_text(json.dumps(spec))


# The manifest's clip is 900 frames at 30 fps: 30 s long.
@pytest.mark.parametrize(("corrupt", "message"), [
    (_reuse_detections_for_second_video, "duplicate detection input"),
    (lambda m: _edit_labels(m, lambda lab: lab.update(video="other")),
     "labels identify a different video"),
    (lambda m: _edit_labels(m, lambda lab: lab["runs"][1].update(end_t=30.5)),
     "labeled run ends after the video"),
    (lambda m: _edit_labels(m, lambda lab: lab["runs"][0].update(start_t=-0.5)),
     "labeled run starts before the video"),
    (lambda m: _edit_labels(m, lambda lab: lab.update(drops=[30.5])),
     "labeled drop lies outside the video"),
    (lambda m: _edit_labels(m, lambda lab: lab.update(drops=[-0.5])),
     "labeled drop lies outside the video"),
    (lambda m: _edit_first_detection(m, frame_idx=900), "invalid detection at frame 900"),
    (lambda m: _edit_first_detection(m, confidence=1.5), "invalid detection"),
    (lambda m: _edit_first_detection(m, t=31.0), "invalid detection"),
], ids=["duplicate_detections", "video_mismatch", "run_past_end", "negative_start",
        "drop_past_end", "negative_drop", "frame_past_end", "confidence_above_one",
        "time_past_end"])
def test_benchmark_rejects_inputs_inconsistent_with_the_clip(benchmark_manifest, corrupt, message):
    from juggletrack.eval.benchmark import run_benchmark

    corrupt(benchmark_manifest)
    with pytest.raises(ValueError, match=message):
        run_benchmark(benchmark_manifest)


def test_benchmark_accepts_events_at_the_clip_boundaries(benchmark_manifest):
    from juggletrack.eval.benchmark import run_benchmark

    def edit(labels):
        labels["runs"][1]["end_t"] = 30.0
        labels["drops"] = [0.0, 30.0]

    _edit_labels(benchmark_manifest, edit)
    assert run_benchmark(benchmark_manifest)["summary"]["drop_fn"] == 2


def test_benchmark_realtime_replays_without_video_or_weights(benchmark_manifest):
    from juggletrack.eval.benchmark import run_benchmark

    result = run_benchmark(benchmark_manifest, realtime=True)
    live = result["clips"][0]["realtime"]
    assert live["catches"] == 8
    assert live["runs"] == 1
    assert live["drops"] == 0
    assert live["catch_error"] == 10  # includes the independently labeled missed run
    assert live["catch_delta_offline"] == 0


def test_benchmark_cli_writes_report(benchmark_manifest, tmp_path):
    from juggletrack.cli import app

    out = tmp_path / "report.json"
    result = CliRunner().invoke(app, ["benchmark", str(benchmark_manifest), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["summary"]["n_labeled_runs"] == 2
    assert "50.0%" in result.output


@pytest.mark.parametrize("config", [{"cluster_merge_distance": .1}, {"resid_tol": float("nan")}])
def test_benchmark_rejects_mistyped_or_nonfinite_config(benchmark_manifest, tmp_path, config):
    from juggletrack.cli import app

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config))
    out = tmp_path / "report.json"
    result = CliRunner().invoke(app, [
        "benchmark", str(benchmark_manifest), "--config", str(cfg), "--out", str(out),
    ])
    assert result.exit_code == 2, result.output
    assert not out.exists()


def test_realtime_benchmark_rejects_inconsistent_timestamps(benchmark_manifest):
    from juggletrack.eval.benchmark import run_benchmark

    dets = benchmark_manifest.parent / "detections.jsonl"
    lines = dets.read_text().splitlines()
    first = json.loads(lines[0])
    first["t"] += .01
    lines[0] = json.dumps(first)
    dets.write_text("\n".join(lines))
    with pytest.raises(ValueError, match="constant-rate timestamps"):
        run_benchmark(benchmark_manifest, realtime=True)


@pytest.mark.parametrize("destination", ["benchmark.json", "labels.json", "detections.jsonl",
                                         "config.json", "label-link.json"])
def test_benchmark_cannot_overwrite_inputs(benchmark_manifest, destination):
    from juggletrack.cli import app

    root = benchmark_manifest.parent
    config = root / "config.json"
    config.write_text("{}")
    (root / "label-link.json").symlink_to(root / "labels.json")
    out = root / destination
    before = out.read_bytes()
    result = CliRunner().invoke(app, [
        "benchmark", str(benchmark_manifest), "--config", str(config), "--out", str(out),
    ])
    assert result.exit_code == 2, result.output
    assert out.read_bytes() == before


def test_symlinked_manifest_protects_its_real_relative_inputs(benchmark_manifest):
    from juggletrack.cli import app

    alias_dir = benchmark_manifest.parent / "alias"
    alias_dir.mkdir()
    alias = alias_dir / "manifest.json"
    alias.symlink_to(benchmark_manifest)
    labels = benchmark_manifest.parent / "labels.json"
    before = labels.read_bytes()
    result = CliRunner().invoke(app, ["benchmark", str(alias), "--out", str(labels)])
    assert result.exit_code == 2, result.output
    assert labels.read_bytes() == before
