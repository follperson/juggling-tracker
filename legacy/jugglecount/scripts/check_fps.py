import time
from jugglecount.core.pipeline import run_analysis

def check_fps():
    video = "data/raw/af1.mov"
    model = "yolo11x.mlpackage"
    res = 640
    
    print(f"Checking FPS for {model} @ {res}px...")
    start = time.time()
    
    analysis = run_analysis(
        video_path=video,
        output_dir="outputs/fps_test",
        test=False,
        enable_roi=True,
        process_interval=2,
        detector_model=model,
        detector_imgsz=res
    )
    
    elapsed = time.time() - start
    # Frame count is total frames.
    # Processing FPS = Total Frames / Total Wall Time
    # (Even though we skip frames, we processed the *video* at this rate)
    fps = analysis.video_metadata.frame_count / elapsed 
    
    print(f"\nRESULTS:")
    print(f"Total Time: {elapsed:.2f}s")
    print(f"Video Frames: {analysis.video_metadata.frame_count}")
    print(f"Processing Speed: {fps:.2f} FPS")

if __name__ == "__main__":
    check_fps()
