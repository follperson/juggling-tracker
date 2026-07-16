"""YOLO ball detector (stock COCO or fine-tuned). ultralytics is imported
lazily in __init__ so `import juggletrack` (and the whole event core) stays
torch-free.
"""
from __future__ import annotations

import numpy as np

from juggletrack.types import Detection

SPORTS_BALL_CLASS = 32  # COCO 80-class index
BALL_CLASS_NAMES = frozenset({"ball", "sports ball"})


def resolve_ball_classes(names: dict[int, str]) -> tuple[int, ...] | None:
    """Class-id filter for ball-like classes in a loaded model.

    Stock COCO models keep only 'sports ball' (32); fine-tuned single-class
    models match their 'ball' class. None (no filter) when no name matches —
    a custom ball model may name its class anything, and filtering a
    dedicated detector to an id it doesn't have silences it entirely.
    """
    matched = tuple(sorted(
        i for i, n in names.items() if str(n).lower() in BALL_CLASS_NAMES
    ))
    return matched or None


def detections_from_xywhn(
    xywhn: np.ndarray, confs: np.ndarray, frame_idx: int, t: float
) -> list[Detection]:
    """Convert ultralytics normalized-center boxes to Detections."""
    return [
        Detection(
            frame_idx=frame_idx, t=t,
            x=float(cx), y=float(cy), w=float(w), h=float(h),
            confidence=float(c),
        )
        for (cx, cy, w, h), c in zip(xywhn, confs)
    ]


class YOLODetector:
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf: float = 0.05,
        imgsz: int = 640,
        classes: tuple[int, ...] | None = None,
        device: str | None = None,
    ):
        from ultralytics import YOLO  # lazy: torch loads only when a real detector is built

        self._model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        # None = resolve from the model's own class names, so fine-tuned
        # single-class 'ball' models aren't silently filtered to COCO id 32.
        self.classes = (
            classes if classes is not None
            else resolve_ball_classes(self._model.names or {})
        )
        self.device = device

    def detect(self, frame: np.ndarray, frame_idx: int, t: float) -> list[Detection]:
        result = self._model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=list(self.classes) if self.classes is not None else None,
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        return detections_from_xywhn(
            boxes.xywhn.cpu().numpy(), boxes.conf.cpu().numpy(), frame_idx, t
        )
