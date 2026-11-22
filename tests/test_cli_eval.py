import json

import pytest
from typer.testing import CliRunner

from juggletrack.cli import app
from juggletrack.types import Run, SessionResult

runner = CliRunner()


@pytest.fixture()
def inputs(tmp_path):
    session = SessionResult(runs=[Run(start_t=1.0, end_t=6.0, catches=12, throws=12,
                                      end_reason="stop", period_s=0.45, quality=0.8)])
    analysis = tmp_path / "analysis.json"
    analysis.write_text(session.model_dump_json())
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({
        "video": "v.mp4", "runs": [{"start_t": 1.0, "end_t": 6.0, "catches": 12}], "drops": [],
    }))
    return analysis, labels


def test_eval_creates_missing_report_directories(inputs, tmp_path):
    analysis, labels = inputs
    report = tmp_path / "nested" / "dir" / "report.json"
    result = runner.invoke(app, ["eval", str(analysis), str(labels), "--out", str(report)])
    assert result.exit_code == 0, result.output
    assert json.loads(report.read_text())["frac_catch_within_1"] == 1.0


@pytest.mark.parametrize("which", [0, 1], ids=["analysis", "labels"])
def test_eval_refuses_to_overwrite_an_input(inputs, which):
    before = [p.read_bytes() for p in inputs]
    result = runner.invoke(app, ["eval", *map(str, inputs), "--out", str(inputs[which])])
    assert result.exit_code == 2
    assert "must not overwrite an input" in result.output
    assert [p.read_bytes() for p in inputs] == before


def test_eval_prints_na_for_scores_with_an_empty_denominator(inputs):
    analysis, labels = inputs
    labels.write_text(json.dumps({"video": "idle.mp4", "runs": [], "drops": []}))
    result = runner.invoke(app, ["eval", str(analysis), str(labels)])
    assert result.exit_code == 0, result.output
    assert "runs: 1 predicted / 0 labeled" in result.output
    assert "run boundary IoU>=0.9: n/a" in result.output
    assert "catch count within +/-1: n/a" in result.output
    assert "drops: precision n/a recall n/a (tp=0 fp=0 fn=0)" in result.output


@pytest.mark.parametrize("document", [
    {},
    {"t": 30.0, "catches_total": 12, "drops_total": 1, "runs_completed": 1, "run_active": False},
], ids=["empty_object", "live_session"])
def test_eval_rejects_analysis_without_schema_version(inputs, document):
    analysis, labels = inputs
    analysis.write_text(json.dumps(document))
    result = runner.invoke(app, ["eval", str(analysis), str(labels)])
    assert result.exit_code == 2
    assert "schema_version" in result.output
    assert "runs:" not in result.output


@pytest.mark.parametrize("document", [
    {"schema_version": "1", "split": "development", "clips": []},
    {"schema_version": "1", "split": "development", "clips": [], "summary": {}},
], ids=["benchmark_manifest", "benchmark_report"])
def test_eval_rejects_versioned_json_that_is_not_an_analysis(inputs, document):
    analysis, labels = inputs
    analysis.write_text(json.dumps(document))
    result = runner.invoke(app, ["eval", str(analysis), str(labels)])
    assert result.exit_code == 2, result.output
    assert "has no runs, arcs key" in " ".join(result.output.replace("│", " ").split())
    assert "predicted" not in result.output


@pytest.mark.parametrize("text", ["null", "5"], ids=["null", "number"])
def test_eval_rejects_analysis_that_is_not_an_object(inputs, text):
    analysis, labels = inputs
    analysis.write_text(text)
    result = runner.invoke(app, ["eval", str(analysis), str(labels)])
    assert result.exit_code == 2, result.output
    assert "has no schema_version, runs, arcs key" in " ".join(
        result.output.replace("│", " ").split()
    )


def test_eval_has_a_help_summary():
    result = runner.invoke(app, ["--help"])
    line = next(row for row in result.output.splitlines() if " eval " in row)
    assert "Score" in line
