import cv2
import numpy as np
import os
from jugglecount.core.signature_detector import SignatureBallDetector
from jugglecount.core.detector import YOLOBallDetector, Detection

def debug_tracking_viz():
    video_path = "data/raw/af1.mov"
    if not os.path.exists(video_path):
        return

    # Use the "Green" configuration (High Recall, Low Precision)
    # H=30, S=32, Sigma=1.0, Threshold=0.7, Mask=30%
    # We expect this to be noisy on full frame, but clean with ROI
    detector = SignatureBallDetector(h_bins=30, s_bins=32, sigma=1.0, confidence_threshold=0.7)
    
    cap = cv2.VideoCapture(video_path)
    
    # Initialization
    print("Finding initial object with YOLO...")
    yolo = YOLOBallDetector(model_name="yolov8n.pt") 
    
    frame0 = None
    bbox = None
    
    # Scan first 150 frames
    frames_processed = 0
    while frames_processed < 150:
        ret, frame = cap.read()
        if not ret: break
        frames_processed += 1
        
        dets = yolo.detect(frame)
        if dets:
            best_det = max(dets, key=lambda x: x.confidence)
            bbox = best_det.bbox
            frame0 = frame
            print(f"Tracking object found at {bbox} on frame {frames_processed}")
            break
            
    if frame0 is None:
        print("No object found.")
        return

    # Register
    detector.register_object(frame0, bbox, "ball")
    detector.register_background(frame0)
    
    # Setup video writer
    h, w = frame0.shape[:2]
    out_path = "outputs/debug_viz_tracking.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    
    # Simple Tracker State
    # (x, y, vx, vy)
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    state = {'x': cx, 'y': cy, 'vx': 0, 'vy': 0}
    
    # ROI Size (Normalized)
    roi_size = 150 # pixels
    
    print("Processing tracking loop...")
    for i in range(200): # Process 200 frames
        ret, frame = cap.read()
        if not ret: break
        
        vis = frame.copy()
        
        # 1. Full Frame Detection (Red - Baseline Behavior)
        # We run this just to visualize what we are filtering OUT
        ff_dets = detector.detect(frame)
        for d in ff_dets:
            x1, y1, x2, y2 = map(int, d.bbox)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1) # Red for raw detections
            
        # 2. ROI Tracking (Green - Proposed Behavior)
        # Predict
        pred_x = state['x'] + state['vx']
        pred_y = state['y'] + state['vy'] + 1.0 # gravity?
        
        # Define ROI (x_center_norm, y_center_norm, w_norm, h_norm)
        # roi_size needs to be large enough to catch the ball
        rw = roi_size / w
        rh = roi_size / h
        pcx_norm = pred_x / w
        pcy_norm = pred_y / h
        
        rois = [(pcx_norm, pcy_norm, rw, rh)]
        
        # Visualize ROI
        rx1 = int((pcx_norm - rw/2) * w)
        ry1 = int((pcy_norm - rh/2) * h)
        rx2 = int((pcx_norm + rw/2) * w)
        ry2 = int((pcy_norm + rh/2) * h)
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2) # Cyan ROI
        
        roi_dets = detector.detect_from_rois(frame, rois)
        
        # Update State
        if roi_dets:
            # Pick closest to center of ROI
            best_dist = float('inf')
            best_det = None
            for d in roi_dets:
                 dcx = (d.bbox[0] + d.bbox[2]) / 2
                 dcy = (d.bbox[1] + d.bbox[3]) / 2
                 dist = (dcx - pred_x)**2 + (dcy - pred_y)**2
                 if dist < best_dist:
                     best_dist = dist
                     best_det = d
            
            if best_det:
                bx1, by1, bx2, by2 = map(int, best_det.bbox)
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 0), 3) # Green for Tracked
                
                new_cx = (best_det.bbox[0] + best_det.bbox[2]) / 2
                new_cy = (best_det.bbox[1] + best_det.bbox[3]) / 2
                
                # Update velocity (simple EWMA)
                alpha = 0.5
                state['vx'] = alpha * (new_cx - state['x']) + (1-alpha) * state['vx']
                state['vy'] = alpha * (new_cy - state['y']) + (1-alpha) * state['vy']
                state['x'] = new_cx
                state['y'] = new_cy
                
        else:
            # No detection, coast
            state['x'] = pred_x
            state['y'] = pred_y
            
        writer.write(vis)

    writer.release()
    cap.release()
    print(f"Saved tracking debug video to {out_path}")

if __name__ == "__main__":
    debug_tracking_viz()
