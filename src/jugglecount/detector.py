from abc import ABC, abstractmethod
from typing import List, Tuple
from pydantic import BaseModel
import numpy as np

class Detection(BaseModel):
    bbox: Tuple[float, float, float, float]  # xyxy
    confidence: float
    class_id: int

class BallDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        pass

class YOLOBallDetector(BallDetector):
    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.25):
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        # Juggling balls are typically 'sports ball' in COCO which is class ID 32
        self.target_class_ids = [32] 

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id in self.target_class_ids:
                    xyxy = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    detections.append(Detection(
                        bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                        confidence=conf,
                        class_id=cls_id
                    ))
        return detections
