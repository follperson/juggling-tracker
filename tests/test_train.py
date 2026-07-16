import sys
import types

import pytest


def install_fake_ultralytics(monkeypatch, tmp_path, *, create_best=True):
    calls = {}

    class FakeYOLO:
        def __init__(self, base_model):
            calls["base_model"] = base_model

        def train(self, **kw):
            calls["train_kwargs"] = kw
            save_dir = tmp_path / kw["project"] / kw["name"]
            if create_best:
                (save_dir / "weights").mkdir(parents=True, exist_ok=True)
                (save_dir / "weights" / "best.pt").write_bytes(b"fake-weights")
            else:
                save_dir.mkdir(parents=True, exist_ok=True)
            return types.SimpleNamespace(save_dir=str(save_dir))

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    return calls


def test_train_detector_plumbing(tmp_path, monkeypatch):
    calls = install_fake_ultralytics(monkeypatch, tmp_path)
    from juggletrack.train.finetune import train_detector

    best = train_detector(
        tmp_path / "data.yaml", base_model="yolo11n.pt", epochs=3, imgsz=320,
        project=str(tmp_path / "runs"), name="tst",
    )
    assert best.exists() and best.name == "best.pt"
    kw = calls["train_kwargs"]
    assert calls["base_model"] == "yolo11n.pt"
    assert kw["epochs"] == 3 and kw["imgsz"] == 320
    assert kw["data"].endswith("data.yaml")
    assert kw["exist_ok"] is True and kw["plots"] is False


def test_train_detector_missing_best_raises(tmp_path, monkeypatch):
    install_fake_ultralytics(monkeypatch, tmp_path, create_best=False)
    from juggletrack.train.finetune import train_detector

    with pytest.raises(FileNotFoundError):
        train_detector(tmp_path / "data.yaml", project=str(tmp_path / "runs"), name="x")


def test_cli_train_command(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    calls = install_fake_ultralytics(monkeypatch, tmp_path)
    from juggletrack.cli import app

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names:\n  0: ball\n")
    result = CliRunner().invoke(app, [
        "train", str(data_yaml), "--epochs", "2",
        "--project", str(tmp_path / "runs"), "--name", "clitest",
    ])
    assert result.exit_code == 0, result.output
    assert "best.pt" in result.output
    assert calls["train_kwargs"]["epochs"] == 2


@pytest.mark.detector
def test_train_one_epoch_real(tmp_path):
    """Real ultralytics smoke: 1 epoch on a tiny synthetic dataset."""
    from juggletrack.data.dataset import assemble_dataset
    from juggletrack.train.finetune import train_detector
    from tests.test_dataset import make_coco_source

    sources = [make_coco_source(tmp_path, f"s{i}", n_images=4) for i in range(2)]
    ds = tmp_path / "ds"
    assemble_dataset(sources, ds, val_fraction=0.5, seed=0)
    best = train_detector(ds / "data.yaml", epochs=1, imgsz=64,
                          project=str(tmp_path / "runs"), name="smoke")
    assert best.exists()
