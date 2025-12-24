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
from jugglecount.schema import FullAnalysis, VideoMetadata, SegmentAnalysis
from jugglecount.viz import create_debug_overlay
import cv2
import json

def run_analysis(video_path: str, output_dir: str):
    logger.info(f"Starting analysis on {video_path}")
    reader = VideoReader(video_path)
    detector = YOLOBallDetector()
    tracker = SimpleTracker()

    for i, timestamp, frame in reader:
        if i >= 500:
            break
        detections = detector.detect(frame)
        tracker.update(detections, i, timestamp)
        
        if i % 100 == 0:
            logger.info(f"Processed frame {i}/{reader.frame_count}")

    features = compute_features(tracker.tracks, reader.frame_count, reader.fps)
    segments = segment_video(features)
    
    segment_results = []
    all_events = []
    
    for seg in segments:
        events = extract_throw_events(tracker.tracks, seg)
        runs = compute_runs(events)
        all_events.extend(events)
        
        segment_results.append(SegmentAnalysis(
            segment=seg,
            throws=events,
            runs=runs,
            total_throws=len(events)
        ))

    analysis = FullAnalysis(
        video_metadata=VideoMetadata(
            fps=reader.fps,
            width=reader.width,
            height=reader.height,
            frame_count=reader.frame_count,
            duration=reader.duration
        ),
        segments=segment_results,
        tracks=tracker.tracks
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "analysis.json"), "w") as f:
        f.write(analysis.model_dump_json(indent=2))

    logger.info(f"Generating debug overlay...")
    create_debug_overlay(video_path, os.path.join(output_dir, "debug_overlay.mp4"), tracker.tracks, all_events, segments, max_frames=500)

    reader.release()
    logger.info(f"Analysis complete. Results in {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_video.py <video_path> <output_dir>")
        sys.exit(1)
        
    run_analysis(sys.argv[1], sys.argv[2])
