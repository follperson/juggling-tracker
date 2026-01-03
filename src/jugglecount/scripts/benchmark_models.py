import os
import time
import pandas as pd
from jugglecount.core.pipeline import run_analysis

def run_benchmark():
    videos = [
        "data/raw/af1.mov", 
        "data/raw/af2.mp4"
    ]
    
    resolutions = [320, 480, 640, 960, 1280]
    model_sizes = ["n", "s", "m", "l", "x"] # suffix for yolov8
    
    results = []
    
    output_dir = "outputs/benchmark_results"
    os.makedirs(output_dir, exist_ok=True)

    for video in videos:
        if not os.path.exists(video):
            print(f"Skipping {video}, not found.")
            continue
            
        print(f"Benchmarking {video}...")
        
        for size in model_sizes:
            model_name = f"yolov8{size}.pt"
            
            for res in resolutions:
                print(f"  Testing {model_name} @ {res}px...")
                
                # Run Analysis
                start_time = time.time()
                try:
                    # We use a unique output dir for each run to avoid collisions/caching issues if any
                    run_output = os.path.join(output_dir, f"{os.path.basename(video)}_{size}_{res}")
                    
                    # Assuming run_analysis returns the FullAnalysis object
                    analysis = run_analysis(
                        video_path=video,
                        output_dir=run_output,
                        test=False,
                        user_id="benchmark",
                        enable_roi=True, # Keep ROI optimizations on as that's the production config
                        process_interval=2,
                        detector_model=model_name,
                        detector_imgsz=res
                    )
                    
                    elapsed = time.time() - start_time
                    fps = analysis.video_metadata.frame_count / elapsed # Rough FPS
                    
                    # Compute total throws from runs (more accurate than raw EventType.THROW)
                    total_throws = sum(run.throw_count for run in analysis.runs)
                    max_streak = max([run.throw_count for run in analysis.runs]) if analysis.runs else 0
                    
                    # Add to results
                    results.append({
                        "video": os.path.basename(video),
                        "model": model_name,
                        "resolution": res,
                        "fps": round(fps, 1),
                        "total_time": round(elapsed, 2),
                        "total_throws": total_throws,
                        "max_streak": max_streak,
                        "run_count": len(analysis.runs)
                    })
                    
                    print(f"    -> FPS: {fps:.1f}, Throws: {total_throws}")
                    
                except Exception as e:
                    print(f"    -> FAILED: {e}")
                    results.append({
                        "video": os.path.basename(video),
                        "model": model_name,
                        "resolution": res,
                        "error": str(e)
                    })

    # Save Results
    df = pd.DataFrame(results)
    csv_path = "outputs/model_benchmark.csv"
    df.to_csv(csv_path, index=False)
    
    print("\nBenchmark Complete!")
    print(df.to_markdown())

if __name__ == "__main__":
    run_benchmark()
