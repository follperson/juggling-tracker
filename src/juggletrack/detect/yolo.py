"""Stock-COCO YOLO ball detector. ultralytics is imported lazily in __init__
so `import juggletrack` (and the whole event core) stays torch-free.
"""
from __future__ import annotations

import numpy as np

from juggletrack.types import Detection

SPORTS_BALL_CLASS = 32  # COCO 80-class index


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
        classes: tuple[int, ...] = (SPORTS_BALL_CLASS,),
        device: str | None = None,
    ):
        from ultralytics import YOLO  # lazy: torch loads only when a real detector is built

        self._model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.classes = classes
        self.device = device

    def detect(self, frame: np.ndarray, frame_idx: int, t: float) -> list[Detection]:
        result = self._model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=list(self.classes),
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        return detections_from_xywhn(
            boxes.xywhn.cpu().numpy(), boxes.conf.cpu().numpy(), frame_idx, t
        )
