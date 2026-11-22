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


def test_environment_model_expands_home(monkeypatch, tmp_path):
    from juggletrack.detect.weights import resolve_model

    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights/best.pt").write_bytes(b"fixture")
    monkeypatch.setenv("JUGGLETRACK_MODEL", "~/weights/best.pt")
    assert resolve_model() == str(tmp_path / "weights/best.pt")


def test_missing_explicit_weights_path_names_the_option():
    from juggletrack.detect.weights import resolve_model

    with pytest.raises(FileNotFoundError, match="--model"):
        resolve_model("models/juggletrack-v9/best.pt")


def test_missing_environment_weights_path_names_the_variable(monkeypatch):
    from juggletrack.detect.weights import resolve_model

    monkeypatch.setenv("JUGGLETRACK_MODEL", "/nonexistent/best.pt")
    with pytest.raises(FileNotFoundError, match="JUGGLETRACK_MODEL"):
        resolve_model()


def test_exported_package_directory_is_accepted():
    from juggletrack.detect.weights import resolve_model

    Path("export/best.mlpackage").mkdir(parents=True)
    assert resolve_model("export/best.mlpackage") == "export/best.mlpackage"


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
