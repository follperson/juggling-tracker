import sys
import types
from pathlib import Path

import pytest


def install_fake_ultralytics(monkeypatch, tmp_path, *, create_product=True):
    calls = {}

    class FakeYOLO:
        def __init__(self, weights):
            calls["weights"] = weights

        def export(self, **kw):
            calls["export_kwargs"] = kw
            product = tmp_path / "best.mlpackage"
            if create_product:
                product.mkdir()
            return str(product)

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    return calls


def test_export_coreml_plumbing(tmp_path, monkeypatch):
    calls = install_fake_ultralytics(monkeypatch, tmp_path)
    from juggletrack.train.export import export_coreml

    out = export_coreml(tmp_path / "best.pt", imgsz=640, half=True)
    assert out.exists() and out.suffix == ".mlpackage"
    kw = calls["export_kwargs"]
    assert kw["format"] == "coreml" and kw["imgsz"] == 640
    assert kw["half"] is True and kw["nms"] is True


def test_export_coreml_missing_product_raises(tmp_path, monkeypatch):
    install_fake_ultralytics(monkeypatch, tmp_path, create_product=False)
    from juggletrack.train.export import export_coreml

    with pytest.raises(FileNotFoundError):
        export_coreml(tmp_path / "best.pt")


@pytest.mark.detector
def test_export_coreml_real():
    from juggletrack.train.export import export_coreml

    weights = Path("/Users/andrew.follmann/personal-projects/juggling/models/juggletrack-v3/best.pt")
    if not weights.exists():
        pytest.skip("v3 weights not on this machine")
    out = export_coreml(weights)
    assert out.exists()
