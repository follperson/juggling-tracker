import cv2
import numpy as np
from typing import List
import statistics
from jugglecount.db.schema import Track, ThrowEvent, Segment, EventType, Run

def create_debug_overlay(
    video_path: str, 
    output_path: str, 
    tracks: List[Track], 
    events: List[ThrowEvent], 
    segments: List[Segment], 
    runs: List[Run] = [],
    max_frames: int = None
):
    from .ingest import VideoReader
    reader = VideoReader(video_path)
    
    # Use avc1 for better browser compatibility (Streamlit)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, reader.fps, (reader.width, reader.height))
    
    # Pre-map events to frames for easy lookup
    event_frames = {}
    for ev in events:
        f_idx = int(ev.timestamp * reader.fps)
        event_frames[f_idx] = event_frames.get(f_idx, []) + [ev]

    for i, timestamp, frame in reader:
        if max_frames is not None and i >= max_frames:
            break
            
        # --- Metrics Calculation ---
        
        # 1. Session Stats
        session_throws = [e for e in events if e.event_type == EventType.THROW and e.timestamp <= timestamp]
        total_session_throws = len(session_throws)
        
        # 2. Run Stats (Completed)
        completed_runs = [r for r in runs if r.end_time < timestamp]
        run_counts = [r.throw_count for r in completed_runs]
        mean_throws = statistics.mean(run_counts) if run_counts else 0.0
        median_throws = statistics.median(run_counts) if run_counts else 0.0
        
        # 3. Current Run Stats
        current_run_idx = -1
        current_run_throws = 0
        
        for idx, r in enumerate(runs):
            # Check if we are "in" a run (from start to end)
            if r.start_time <= timestamp <= r.end_time:
                current_run_idx = idx + 1 # 1-based index
                # Count throws in this run so far
                current_run_throws = len([
                    e for e in session_throws 
                    if r.start_time <= e.timestamp <= timestamp
                ])
                break
        
        # --- Drawing Overlay ---

        # Draw metrics box background
        cv2.rectangle(frame, (10, 10), (350, 160), (0, 0, 0), -1)
        
        # Draw counts
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Session Throws: {total_session_throws}", (20, 40), font, 0.7, (255, 255, 255), 2)
        
        if current_run_idx != -1:
            cv2.putText(frame, f"Run #{current_run_idx}: {current_run_throws} throws", (20, 70), font, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Waiting for run...", (20, 70), font, 0.7, (150, 150, 150), 2)
            
        # Draw stats
        cv2.putText(frame, f"Mean per Run: {mean_throws:.1f}", (20, 110), font, 0.6, (200, 200, 255), 1)
        cv2.putText(frame, f"Median per Run: {median_throws:.1f}", (20, 135), font, 0.6, (200, 200, 255), 1)

        # Draw tracks
        for track in tracks:
            # Only draw points up to current frame
            points = [p for p in track.points if p.frame_idx <= i]
            if not points:
                continue
            
            # Draw tail
            tail = points[-20:] # Last 20 frames
            if len(tail) > 1:
                for j in range(len(tail) - 1):
                    p1 = tail[j]
                    p2 = tail[j+1]
                    c1 = (int(p1.pos_x * reader.width), int(p1.pos_y * reader.height))
                    c2 = (int(p2.pos_x * reader.width), int(p2.pos_y * reader.height))
                    cv2.line(frame, c1, c2, (0, 255, 0), 2)
            
            # Draw current centroid if active
            last_p = points[-1]
            if last_p.frame_idx == i:
                cx = int(last_p.pos_x * reader.width)
                cy = int(last_p.pos_y * reader.height)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                cv2.putText(frame, f"ID:{track.id}", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Draw throw/peak events
        for event in events:
            # Display event label for a short duration after the event timestamp
            if event.timestamp <= timestamp <= event.timestamp + 0.5:
                color = (0, 255, 0) if event.event_type == EventType.THROW else (255, 0, 255) # Green for THROW, Magenta for PEAK
                label = "THROW" if event.event_type == EventType.THROW else "PEAK"
                y_pos = reader.height - 50 if event.event_type == EventType.THROW else 50 # THROW at bottom, PEAK at top
                
                cv2.putText(frame, f"{label} ID:{event.track_id}", (50, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Check if in a segment
        in_segment = any(s.start_time <= timestamp <= s.end_time for s in segments)
        if in_segment:
             cv2.putText(frame, "JUGGLING", (reader.width - 200, 50), font, 1, (0, 255, 255), 2)

        out.write(frame)
        
    reader.release()
    out.release()

