from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def clean_model_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("JUGGLETRACK_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)


def test_explicit_model_overrides_environment(monkeypatch):
    from juggletrack.detect.weights import resolve_model

    monkeypatch.setenv("JUGGLETRACK_MODEL", "environment.pt")
    assert resolve_model("yolo11n.pt") == "yolo11n.pt"


def test_environment_model_expands_home(monkeypatch):
    from juggletrack.detect.weights import resolve_model

    monkeypatch.setenv("JUGGLETRACK_MODEL", "~/weights/best.pt")
    assert resolve_model() == str(Path.home() / "weights/best.pt")


def test_local_champion_is_selected():
    from juggletrack.detect.weights import resolve_model

    weights = Path("models/juggletrack-v3/best.pt")
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"fixture")
    assert Path(resolve_model()).resolve() == weights.resolve()


def test_missing_default_explains_explicit_model_selection():
    from juggletrack.detect.weights import resolve_model

    with pytest.raises(FileNotFoundError, match="--model"):
        resolve_model()
