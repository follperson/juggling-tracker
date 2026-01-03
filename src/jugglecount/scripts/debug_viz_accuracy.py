import cv2
import numpy as np
import os
from jugglecount.core.signature_detector import SignatureBallDetector
from jugglecount.core.detector import YOLOBallDetector

def debug_viz():
    video_path = "data/raw/af1.mov"
    if not os.path.exists(video_path):
        return

    # Compare Baseline vs Proposed
    configs = [
        ("Baseline", 180, 256, 0.0),
        ("Proposed", 30, 32, 1.0)
    ]
    
    cap = cv2.VideoCapture(video_path)
    
    # Use YOLO to find the object to track
    print("Finding initial object with YOLO...")
    yolo = YOLOBallDetector(model_name="yolov8n.pt") 
    
    frame0 = None
    bbox = None
    
    # Scan first 150 frames
    for _ in range(150):
        ret, frame = cap.read()
        if not ret: break
        
        dets = yolo.detect(frame)
        if dets:
            best_det = max(dets, key=lambda x: x.confidence)
            bbox = best_det.bbox
            frame0 = frame
            print(f"Tracking object found at {bbox}")
            break
            
    if frame0 is None:
        print("No object found to track in first 30 frames.")
        return
    
    detectors = []
    for name, hb, sb, sig in configs:
        d = SignatureBallDetector(h_bins=hb, s_bins=sb, sigma=sig)
        d.register_object(frame0, bbox, "ball")
        d.register_background(frame0)
        detectors.append((name, d))
        
    # Process next 100 frames and save video
    h, w = frame0.shape[:2]
    out_path = "outputs/debug_viz_accuracy.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    
    print("Generating video...")
    for i in range(100):
        ret, frame = cap.read()
        if not ret: break
        
        vis = frame.copy()
        
        for name, det in detectors:
            detections = det.detect(frame)
            color = (0, 0, 255) if name == "Baseline" else (0, 255, 0)
            
            for d in detections:
                x1, y1, x2, y2 = map(int, d.bbox)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
        writer.write(vis)
        
    writer.release()
    cap.release()
    print(f"Saved debug video to {out_path}")

if __name__ == "__main__":
    debug_viz()
