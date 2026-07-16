"""Fine-tune the ball detector. ultralytics imported lazily (torch stays out
of module import); training data comes from data/dataset.py's data.yaml.
"""
from __future__ import annotations

from pathlib import Path


def train_detector(
    data_yaml: str | Path,
    *,
    base_model: str = "yolo11n.pt",
    epochs: int = 40,
    imgsz: int = 640,
    device: str | None = None,
    project: str | Path = "runs/finetune",
    name: str = "juggletrack",
) -> Path:
    from ultralytics import YOLO  # lazy

    model = YOLO(base_model)
    results = model.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, device=device,
        project=str(project), name=name, exist_ok=True, plots=False,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"training finished but best.pt missing: {best}")
    return best
