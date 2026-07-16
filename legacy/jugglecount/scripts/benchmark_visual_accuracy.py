import os
import time
import cv2
import numpy as np
import pandas as pd
from jugglecount.core.signature_detector import SignatureBallDetector
from jugglecount.core.detector import Detection, YOLOBallDetector

def get_test_signature(frames: list[np.ndarray]) -> tuple:
    """
    Scan frames to find the first ball to use as a signature reference.
    Returns (bbox, frame_index) or (bbox, 0) if fallback
    """
    # Initialize YOLO just for this one-shot detection
    yolo = YOLOBallDetector(model_name="yolov8n.pt", imgsz=640)
    
    print("Scanning first 150 frames for object to track...")
    for i, frame in enumerate(frames):
        if i > 150: break
        
        detections = yolo.detect(frame)
        if detections:
            best_det = max(detections, key=lambda x: x.confidence)
            print(f"  -> Found object in frame {i}: {best_det.bbox}")
            return best_det.bbox, i

    print("Warning: No ball found in start of video for signature initialization!")
    h, w = frames[0].shape[:2]
    cx, cy = w // 2, h // 2
    return (cx - 30, cy - 30, cx + 30, cy + 30), 0

def run_accuracy_benchmark():
    video_path = "data/raw/af1.mov"
    if not os.path.exists(video_path):
        print(f"Video {video_path} not found.")
        return

    # Configurations to test
    # (h_bins, s_bins, sigma)
    configs = [
        (180, 256, 0), # Baseline (Original)
        (90, 128, 0.5),
        (45, 64, 1.0),
        (30, 32, 1.0), # Proposed Default
        (16, 16, 2.0)  # Extreme smoothing
    ]
    
    results = []

    # Read first 100 frames
    print("Pre-loading frames...")
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} frames.")

    if not frames:
        return
        
    ref_bbox, ref_idx = get_test_signature(frames)
    ref_frame = frames[ref_idx]
    
    output_dir = "outputs/benchmark_viz_accuracy"
    os.makedirs(output_dir, exist_ok=True)

    for (h_bins, s_bins, sigma) in configs:
        print(f"Testing Config: H={h_bins}, S={s_bins}, Sigma={sigma}...")
        
        # Use default scale=1.0 for accuracy test
        detector = SignatureBallDetector(
            scale=1.0, 
            h_bins=h_bins, 
            s_bins=s_bins, 
            sigma=sigma
        )
        detector.register_object(ref_frame, ref_bbox, "ball")
        detector.register_background(ref_frame)
        
        start_time = time.time()
        detections_count = 0
        
        # Process frames starting from where we found the ball? 
        # Or all frames to see if it picks it up?
        # Let's process all frames to be consistent with FPS calc
        for frame in frames:
            dets = detector.detect(frame)
            detections_count += len(dets)
            
        elapsed = time.time() - start_time
        fps = len(frames) / elapsed
        
        print(f"  -> Detections: {detections_count}, FPS: {fps:.1f}")
        
        results.append({
            "h_bins": h_bins,
            "s_bins": s_bins,
            "sigma": sigma,
            "detections": detections_count,
            "fps": fps
        })

    df = pd.DataFrame(results)
    print("\nAccuracy Benchmark Results:")
    print(df.to_markdown())
    df.to_csv(os.path.join(output_dir, "accuracy_results.csv"), index=False)

if __name__ == "__main__":
    run_accuracy_benchmark()
