import sys
import os
from loguru import logger
from jugglecount.ingest import VideoReader
from jugglecount.detector import YOLOBallDetector
from jugglecount.tracker import SimpleTracker
from jugglecount.features import compute_features
from jugglecount.segmenter import segment_video
from jugglecount.events import extract_throw_events
from jugglecount.runs import compute_runs
from jugglecount.schema import FullAnalysis, VideoMetadata, SegmentAnalysis, EventType
from jugglecount.viz import create_debug_overlay
from jugglecount.processing import interpolate_tracks
import cv2
import json

def run_analysis(video_path: str, output_dir: str):
    logger.info(f"Starting analysis on {video_path}")
    reader = VideoReader(video_path)
    
    # Use larger imgsz for high-res videos to detect small balls
    # YOLO requires imgsz to be a multiple of 32
    imgsz = 640
    if reader.height >= 1080:
        imgsz = 1088
    elif reader.height >= 720:
        imgsz = 736
        
    logger.info(f"Using detector imgsz: {imgsz}")
    detector = YOLOBallDetector(imgsz=imgsz)
    tracker = SimpleTracker()

    for i, timestamp, frame in reader:
        # if i >= 500:
        #     break
        detections = detector.detect(frame)
        tracker.update(detections, i, timestamp, reader.width, reader.height)
        
        if i % 100 == 0:
            logger.info(f"Processed frame {i}/{reader.frame_count}")

    # Interpolate tracks to fill gaps from motion blur
    tracks = interpolate_tracks(tracker.tracks)
    
    features = compute_features(tracks, reader.frame_count, reader.fps)
    segments = segment_video(features)
    
    segment_results = []
    all_events = []
    
    for seg in segments:
        events = extract_throw_events(tracks, seg)
        all_events.extend(events)
        
        # Filter throws and peaks for counts
        throws = [e for e in events if e.event_type == EventType.THROW]
        peaks = [e for e in events if e.event_type == EventType.PEAK]
        
        runs = compute_runs(throws) # Still compute runs based on throws for now
        
        segment_results.append(SegmentAnalysis(
            segment=seg,
            throws=events,
            runs=runs,
            total_throws=len(throws),
            total_peaks=len(peaks)
        ))

    analysis = FullAnalysis(
        video_metadata=VideoMetadata(
            fps=round(reader.fps, 4),
            width=reader.width,
            height=reader.height,
            frame_count=reader.frame_count,
            duration=round(reader.duration, 4)
        ),
        segments=segment_results,
        tracks=tracks
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "analysis.json"), "w") as f:
        f.write(analysis.model_dump_json(indent=2))

    logger.info(f"Generating debug overlay...")
    create_debug_overlay(video_path, os.path.join(output_dir, "debug_overlay.mp4"), tracks, all_events, segments)

    reader.release()
    logger.info(f"Analysis complete. Results in {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_video.py <video_path> <output_dir>")
        sys.exit(1)
        
    run_analysis(sys.argv[1], sys.argv[2])
