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


def write_local_champion() -> Path:
    weights = Path("models/juggletrack-v3/best.pt")
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"fixture")
    return weights


def test_local_champion_is_selected():
    from juggletrack.detect.weights import resolve_model

    weights = write_local_champion()
    assert Path(resolve_model()).resolve() == weights.resolve()


def test_environment_model_overrides_local_champion(monkeypatch):
    from juggletrack.detect.weights import resolve_model

    write_local_champion()
    monkeypatch.setenv("JUGGLETRACK_MODEL", "environment.pt")
    assert resolve_model() == "environment.pt"


def test_missing_default_says_where_it_looked_and_how_to_fix(tmp_path):
    from juggletrack.detect.weights import resolve_model

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_model()
    message = str(excinfo.value)
    assert str(tmp_path.resolve() / "models/juggletrack-v3/best.pt") in message
    assert f"current directory {tmp_path.resolve()}" in message
    for remedy in ("--model", "JUGGLETRACK_MODEL", "repository root"):
        assert remedy in message
