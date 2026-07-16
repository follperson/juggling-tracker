import os
import time
import cv2
import numpy as np
import pandas as pd
from jugglecount.core.signature_detector import SignatureBallDetector
from jugglecount.core.detector import Detection

def get_test_signature(frame: np.ndarray) -> tuple:
    """
    Extract a dummy signature from the center of the frame for testing.
    Returns (bbox, frame)
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    bbox = (cx - 30, cy - 30, cx + 30, cy + 30)
    return bbox

def run_benchmark():
    video_path = "data/raw/af1.mov"
    if not os.path.exists(video_path):
        print(f"Video {video_path} not found.")
        return

    scales = [1.0, 0.5, 0.25]
    results = []

    # Read first 100 frames into memory to avoid I/O bottlenecks during bench
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

    # Get a signature source from the first frame
    if not frames:
        return
        
    # We'll use a dummy signature registration on the first frame
    ref_frame = frames[0]
    ref_bbox = get_test_signature(ref_frame)
    
    output_dir = "outputs/benchmark_viz_sig"
    os.makedirs(output_dir, exist_ok=True)

    for scale in scales:
        print(f"Benchmarking Scale: {scale}...")
        
        detector = SignatureBallDetector(scale=scale)
        # Register a signature
        detector.register_object(ref_frame, ref_bbox, "ball")
        # Register background
        detector.register_background(ref_frame)
        
        start_time = time.time()
        
        detections_count = 0
        for frame in frames:
            # Simulate real-time usage (no mask for raw speed test, or add simple mog2 if needed)
            # For pure detector speed, we process raw frames. 
            # In app, we use mask, which adds cost, but detector is the bottleneck usually.
            dets = detector.detect(frame)
            detections_count += len(dets)
            
        elapsed = time.time() - start_time
        fps = len(frames) / elapsed
        
        print(f"  -> FPS: {fps:.1f}, Detections: {detections_count}")
        
        results.append({
            "scale": scale,
            "fps": fps,
            "total_time": elapsed,
            "detections": detections_count
        })

    df = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(df.to_markdown())
    df.to_csv(os.path.join(output_dir, "results.csv"), index=False)

if __name__ == "__main__":
    run_benchmark()
