import cv2
import numpy as np
import os
import csv
from jugglecount.core.signature_detector import SignatureBallDetector
from jugglecount.core.detector import YOLOBallDetector, Detection
from jugglecount.core.tracker import SimpleTracker

def debug_tracking_viz(video_path: str):
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
    
    # Derive output name from input video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_path = f"outputs/debug_viz_{video_name}.mp4"

    # "Green" Configuration (High Recall)
    detector = SignatureBallDetector(h_bins=180, s_bins=256, sigma=0.0, confidence_threshold=0.7)
    
    cap = cv2.VideoCapture(video_path)
    
    # Initialization
    print("Finding initial object with YOLO...", flush=True)
    yolo = YOLOBallDetector(model_name="yolov8n.pt") 
    
    frame0 = None
    bbox = None
    
    frames_processed = 0
    start_offset = 45 # 1.5 seconds at 30fps
    
    while frames_processed < 150:
        ret, frame = cap.read()
        if not ret: break
        frames_processed += 1
        
        if frames_processed < start_offset:
            continue
            
        dets = yolo.detect(frame)
        if dets:
            best_det = max(dets, key=lambda x: x.confidence)
            bbox = best_det.bbox
            frame0 = frame
            print(f"Tracking object found at {bbox} on frame {frames_processed}", flush=True)
            break
            
    if frame0 is None:
        print("No object found.", flush=True)
        return

    # Register
    detector.register_object(frame0, bbox, "ball")
    detector.register_background(frame0)
    
    h, w = frame0.shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    
    # Init Real Tracker
    tracker = SimpleTracker(max_distance=150, max_inactive_frames=10)
    
    # Initialize tracker with the first detection from YOLO
    initial_det = Detection(bbox=bbox, confidence=1.0, class_id=0)
    tracker.update([initial_det], 0, 0.0, w, h)
    
    current_time = 0.0
    frame_idx = 0
    
    # ROI Size (Normalized)
    roi_size_px = 250 
    
    stats_path = "outputs/tracking_stats.csv"
    print("Processing tracking loop...", flush=True)
    
    with open(stats_path, 'w', newline='') as csvfile:
        fieldnames = ['frame', 'track_id', 'area', 'aspect_ratio', 'confidence', 'x', 'y']
        writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer_csv.writeheader()

        try:
            for i in range(300): # Process 300 frames
                ret, frame = cap.read()
                if not ret: break
                current_time += 1/30.0
                frame_idx += 1
                
                if i % 10 == 0:
                    print(f"Processed frame {i} | Tracks: {len(tracker.tracks)}", flush=True)
                    
                vis = frame.copy()
                
                # 1. Determine Scan Mode
                # ONLY do full scan if we have NO active tracks (recovery mode)
                # NO periodic scans - they pick up too much background noise
                do_full_scan = not tracker.tracks
                
                roi_dets = []
                
                if not do_full_scan:
                    # ROI Tracking (Green)
                    predictions = tracker.predict_next_locations(current_time)
                    
                    rois = []
                    for tid, (px, py, pw, ph) in predictions.items():
                        # Dynamic ROI size: 4x the object size
                        roi_w_norm = pw * 4.0
                        roi_h_norm = ph * 4.0
                        
                        rois.append((px, py, roi_w_norm, roi_h_norm))
                        
                        # Viz Prediction Center
                        cx_px = int(px * w)
                        cy_px = int(py * h)
                        cv2.circle(vis, (cx_px, cy_px), 3, (255, 255, 0), -1) 
                        
                        # Viz ROI
                        rx1 = int((px - roi_w_norm/2) * w)
                        ry1 = int((py - roi_h_norm/2) * h)
                        rx2 = int((px + roi_w_norm/2) * w)
                        ry2 = int((py + roi_h_norm/2) * h)
                        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1) 
                        
                    if rois:
                         roi_dets = detector.detect_from_rois(frame, rois)
                    else:
                         # No predictions means no tracks - recover
                         do_full_scan = True

                # Fallback / Full Scan - ONLY when all tracks are lost
                if do_full_scan:
                    roi_dets = detector.detect(frame)
                
                # Filter Detections (Hand/Noise Rejection)
                # Stricter area filter for full-frame scans (background has many small blobs)
                min_area = 2500 if do_full_scan else 800
                filtered_roi_dets = []
                for d in roi_dets:
                    bx1, by1, bx2, by2 = map(int, d.bbox)
                    area = (bx2-bx1)*(by2-by1)
                    aspect = (bx2-bx1)/(by2-by1) if (by2-by1) > 0 else 0
                    
                    if area > min_area and 0.8 < aspect < 1.2:
                        filtered_roi_dets.append(d)
                
                # Viz ROI Detections
                for d in roi_dets: # Viz ALL (to see ignored ones in red?)
                    bx1, by1, bx2, by2 = map(int, d.bbox)
                    area = (bx2-bx1)*(by2-by1)
                    aspect = (bx2-bx1)/(by2-by1) if (by2-by1) > 0 else 0
                    
                    label = f"A:{area} AR:{aspect:.2f}"
                    is_valid = area > 800 and 0.8 < aspect < 1.2
                    color = (0, 255, 0) if is_valid else (0, 0, 255) # Green if valid, Red if rejected
                    
                    cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 2) 
                    cv2.putText(vis, label, (bx1, by1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    # Log raw detections
                    writer_csv.writerow({
                        'frame': i,
                        'track_id': -1, 
                        'area': area,
                        'aspect_ratio': round(aspect, 2),
                        'confidence': round(d.confidence, 2),
                        'x': (bx1+bx2)//2,
                        'y': (by1+by2)//2
                    })

                # print(f"F{i}: Update", flush=True)
                tracker.update(filtered_roi_dets, frame_idx, current_time, w, h)
                
                # Viz Active Tracks
                for t in tracker.tracks:
                     if tracker.inactive_counts[t.id] < 5:
                          if t.points:
                              lp = t.points[-1]
                              ltx, lty = int(lp.pos_x * w), int(lp.pos_y * h)
                              cv2.putText(vis, f"ID:{t.id}", (ltx, lty+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                writer.write(vis)

        finally:
            writer.release()
            cap.release()
            print(f"Saved tracking debug video + stats to {out_path}", flush=True)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        debug_tracking_viz(sys.argv[1])
    else:
        # Default: run on both videos
        debug_tracking_viz("data/raw/af1.mov")
        debug_tracking_viz("data/raw/af2.mp4")
