import os
from loguru import logger
from jugglecount.core.ingest import VideoReader
from jugglecount.core.detector import YOLOBallDetector
from jugglecount.core.tracker import SimpleTracker
from jugglecount.core.features import compute_features
from jugglecount.core.segmenter import segment_video
from jugglecount.core.events import extract_throw_events
from jugglecount.core.runs import compute_runs
from jugglecount.db.schema import FullAnalysis, VideoMetadata, SegmentAnalysis, EventType
from jugglecount.core.viz import create_debug_overlay
from jugglecount.core.processing import interpolate_tracks
from jugglecount.db.db import init_db, upsert_user, save_session_results
import uuid
from datetime import datetime

def run_analysis(video_path: str, output_dir: str, test: bool = False, user_id: str = "default_user", enable_roi: bool = True, process_interval: int = 2, detector_model: str = "yolo11x.mlpackage", detector_imgsz: int = 640):
    if test: 
        user_id = "test_user"
    logger.info(f"Starting analysis on {video_path} (ROI: {enable_roi}, Interval: {process_interval}, Model: {detector_model})")
    reader = VideoReader(video_path)
    
    # Defaults to 640 if not specified, which we found is optimal for 'Large' model accuracy/speed balance
    if detector_imgsz is None:
        imgsz = 640
    else:
        imgsz = detector_imgsz
        
    logger.info(f"Using detector imgsz: {imgsz}")
    detector = YOLOBallDetector(model_name=detector_model, imgsz=imgsz)
    tracker = SimpleTracker()

    roi_size_norm = (256.0 / reader.width, 256.0 / reader.height) # Normalized ROI size (approx 256x256)
    full_scan_interval = 30 # Run full detection every 30 frames
    # process_interval = 2 # Process every 2nd frame (effective input 15fps) - Configured via arg

    for i, timestamp, frame in reader:
        if test and i >= 500:
            break
            
        # Frame Skipping
        if i % process_interval != 0:
            continue
            
        use_rois = False
        rois = []
        detections = []
        
        # STRICT ROI LOGIC
        if enable_roi:
            # Determine if we should do a full scan
            # 1. Periodic Interval OR
            # 2. No tracks (Recovery)
            do_full_scan = (i % full_scan_interval == 0) or (not tracker.tracks)
            
            if not do_full_scan:
                # Predict next locations
                predictions = tracker.predict_next_locations(timestamp)
                if predictions:
                    use_rois = True
                    for track_id, (pred_x, pred_y, pred_w, pred_h) in predictions.items():
                        # Dynamic ROI 4.0x
                        roi_w = pred_w * 4.0
                        roi_h = pred_h * 4.0
                        rois.append((pred_x, pred_y, roi_w, roi_h))
            
            if use_rois:
                detections = detector.detect_from_rois(frame, rois)
                # Fallback logic: If we expected something but found nothing, Full Scan
                if len(detections) == 0:
                     detections = detector.detect(frame)
            else:
                # If we didn't use ROIs (either periodic scan or no predictions), Full Scan
                detections = detector.detect(frame)
        else:
            # PURE FULL FRAME MODE - Zero overhead
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
        
        runs = compute_runs(peaks) # Use peaks for run analysis as requested
        
        segment_results.append(SegmentAnalysis(
            segment=seg,
            throws=events,
            runs=runs,
            total_throws=len(throws),
            total_peaks=len(peaks)
        ))

    # Initialize DB and ensure user exists
    init_db()
    upsert_user(user_id=user_id, name=user_id, email=user_id)

    # Create session-level performance data
    total_duration = round(reader.duration, 4)
    session_uuid = str(uuid.uuid4())
    
    total_throws = sum([seg.total_throws for seg in segment_results])
    total_peaks = sum([seg.total_peaks for seg in segment_results])
    
    # Collect all runs across segments
    all_runs = []
    for seg in segment_results:
        all_runs.extend(seg.runs)
    
    metrics_data = {
        "total_throws": total_throws,
        "total_peaks": total_peaks,
        "duration_seconds": total_duration,
        "average_confidence": sum(e.confidence for e in all_events) / len(all_events) if all_events else 0.0,
        "timestamp": datetime.utcnow().timestamp()
    }


    # Persist to database
    save_session_results(
        user_id=user_id,
        session_uuid=session_uuid,
        start_time=0.0,
        end_time=total_duration,
        metrics_data=metrics_data, # Ignored by db, kept for generic structure if needed later
        throws=all_events,
        runs=all_runs
    )

    analysis = FullAnalysis(
        video_metadata=VideoMetadata(
            fps=round(reader.fps, 4),
            width=reader.width,
            height=reader.height,
            frame_count=reader.frame_count,
            duration=total_duration
        ),
        segments=segment_results,
        tracks=tracks,
        runs=all_runs
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "analysis.json"), "w") as f:
        f.write(analysis.model_dump_json(indent=2))

    logger.info(f"Generating debug overlay...")
    create_debug_overlay(video_path, os.path.join(output_dir, "debug_overlay.mp4"), tracks, all_events, segments, runs=all_runs)
    
    logger.info(f"Analysis complete. Results in {output_dir}")
    return analysis

    reader.release()
    logger.info(f"Analysis complete. Results in {output_dir}")
