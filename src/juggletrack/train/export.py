"""CoreML export for the ball detector (Mac realtime path; phone later uses
LiteRT — spec notes the Android target). ultralytics imported lazily."""
from __future__ import annotations

from pathlib import Path


def export_coreml(weights: str | Path, *, imgsz: int = 640, half: bool = True) -> Path:
    from ultralytics import YOLO  # lazy

    product = Path(YOLO(str(weights)).export(
        format="coreml", imgsz=imgsz, half=half, nms=True
    ))
    if not product.exists():
        raise FileNotFoundError(f"CoreML export missing: {product}")
    return product
