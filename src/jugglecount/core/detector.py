from abc import ABC, abstractmethod
from typing import List, Tuple
from pydantic import BaseModel
import numpy as np
import math
import cv2
import torch

class Detection(BaseModel):
    bbox: Tuple[float, float, float, float]  # xyxy
    confidence: float
    class_id: int

class BallDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        pass

class YOLOBallDetector(BallDetector):
    def __init__(self, model_name: str = "yolov8s.pt", confidence_threshold: float = 0.1, imgsz: int = 640):
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        # Juggling balls are typically 'sports ball' in COCO which is class ID 32
        self.target_class_ids = [32] 
        self.load_model(model_name) 

    def load_model(self, model_name: str):
        """"Reload the YOLO model."""
        from ultralytics import YOLO
        import logging
        logger = logging.getLogger(__name__)
        self.model_name = model_name
        
        # Select device: MPS (Mac) > CUDA (NVIDIA) > CPU
        self.device = "cpu"
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
            
        logger.info(f"Loading YOLO model {model_name} on {self.device}...")
        self.model = YOLO(model_name)
        if not model_name.endswith(".mlpackage"):
            self.model.to(self.device) # Explicitly move to device

    def set_imgsz(self, size: int):
        self.imgsz = size

    def set_confidence(self, conf: float):
        self.confidence_threshold = conf

    def update_target_classes(self, class_ids: List[int]):
        """Update the list of allowed class IDs."""
        self.target_class_ids = class_ids

    @property
    def class_names(self):
        """Return the class names from the model."""
        return self.model.names 



    def detect(self, frame: np.ndarray, allowed_classes: List[int] = None) -> List[Detection]:
        results = self.model(frame, conf=self.confidence_threshold, imgsz=self.imgsz, verbose=False)
        detections = []
        
        # Determine filtering set
        filter_ids = allowed_classes if allowed_classes is not None else self.target_class_ids
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if filter_ids is None or cls_id in filter_ids:
                    xyxy = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    detections.append(Detection(
                        bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                        confidence=conf,
                        class_id=cls_id
                    ))
        return detections

    def detect_from_rois(self, frame: np.ndarray, rois: List[Tuple[float, float, float, float]]) -> List[Detection]:
        """
        Run detection on crops defined by ROIs using Mosaic Inference.
        rois: List of (x_center_norm, y_center_norm, w_norm, h_norm)
        """
        detections = []
        h, w = frame.shape[:2]
        
        crops = []
        roi_pixel_coords = [] # (x1, y1) for each crop to map back
        
        for (xc, yc, rw, rh) in rois:
            # Convert to pixels
            cx_px = int(xc * w)
            cy_px = int(yc * h)
            w_px = int(rw * w)
            h_px = int(rh * h)
            
            x1 = max(0, cx_px - w_px // 2)
            y1 = max(0, cy_px - h_px // 2)
            x2 = min(w, x1 + w_px)
            y2 = min(h, y1 + h_px)
            
            # Ensure valid crop
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop = frame[y1:y2, x1:x2]
            crops.append(crop)
            roi_pixel_coords.append((x1, y1))
            
        if not crops:
            return []
            
        # Mosaic Construction
        num_crops = len(crops)
        grid_n = math.ceil(math.sqrt(num_crops))
        
        # Target cell size (try to keep resolution high but fit in model input)
        if hasattr(self, 'imgsz') and self.imgsz:
            target_dim = self.imgsz
        else:
             target_dim = 640
             
        cell_size = target_dim // grid_n
        mosaic_w = grid_n * cell_size
        mosaic_h = grid_n * cell_size
        
        # Create canvas (fill with gray or zeros)
        canvas = np.full((mosaic_h, mosaic_w, 3), 114, dtype=np.uint8)
        
        # ... (Comments omitted for brevity) ...
        
        for i, crop in enumerate(crops):
            row = i // grid_n
            col = i % grid_n
            
            x_offset = col * cell_size
            y_offset = row * cell_size
            
            # Resize crop to cell_size
            resized_crop = cv2.resize(crop, (cell_size, cell_size))
            canvas[y_offset:y_offset+cell_size, x_offset:x_offset+cell_size] = resized_crop
            
        # Run inference on single mosaic
        # imgsz should match canvas, or be close.
        # For CoreML, we must pass the model's native size (self.imgsz)
        results = self.model(canvas, conf=self.confidence_threshold, imgsz=target_dim, verbose=False, stream=False)
        
        # Process results
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id in self.target_class_ids:
                    # Box in mosaic coordinates
                    mx1, my1, mx2, my2 = box.xyxy[0].tolist()
                    mconf = box.conf[0].item()
                    
                    # Determine which cell this belongs to
                    # Center of box
                    mcx = (mx1 + mx2) / 2
                    mcy = (my1 + my2) / 2
                    
                    col = int(mcx // cell_size)
                    row = int(mcy // cell_size)
                    idx = row * grid_n + col
                    
                    if idx >= len(crops):
                        continue # Ghost detection in empty cell?
                        
                    # Map back to crop coordinates
                    # Local coordinates in cell
                    cell_x1 = mx1 - (col * cell_size)
                    cell_y1 = my1 - (row * cell_size)
                    cell_x2 = mx2 - (col * cell_size)
                    cell_y2 = my2 - (row * cell_size)
                    
                    # Normalize to [0,1] in cell
                    norm_x1 = cell_x1 / cell_size
                    norm_y1 = cell_y1 / cell_size
                    norm_x2 = cell_x2 / cell_size
                    norm_y2 = cell_y2 / cell_size
                    
                    # Map to global crop coordinates
                    crop_orig = crops[idx]
                    ch, cw = crop_orig.shape[:2]
                    
                    crop_x1 = norm_x1 * cw
                    crop_y1 = norm_y1 * ch
                    crop_x2 = norm_x2 * cw
                    crop_y2 = norm_y2 * ch
                    
                    # Map to global frame coordinates
                    offset_x, offset_y = roi_pixel_coords[idx]
                    
                    global_x1 = crop_x1 + offset_x
                    global_y1 = crop_y1 + offset_y
                    global_x2 = crop_x2 + offset_x
                    global_y2 = crop_y2 + offset_y
                    
                    detections.append(Detection(
                        bbox=(global_x1, global_y1, global_x2, global_y2),
                        confidence=mconf,
                        class_id=cls_id
                    ))
                    
        return detections
