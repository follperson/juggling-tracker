import cv2
import numpy as np
from typing import List
from .schema import Track, ThrowEvent, Segment, EventType

def create_debug_overlay(video_path: str, output_path: str, tracks: List[Track], events: List[ThrowEvent], segments: List[Segment], max_frames: int = None):
    from .ingest import VideoReader
    reader = VideoReader(video_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, reader.fps, (reader.width, reader.height))
    
    # Pre-map events to frames for easy lookup
    event_frames = {}
    for ev in events:
        f_idx = int(ev.timestamp * reader.fps)
        event_frames[f_idx] = event_frames.get(f_idx, []) + [ev]

    for i, timestamp, frame in reader:
        if max_frames is not None and i >= max_frames:
            break
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
                    p1 = tail[j].pos
                    p2 = tail[j+1].pos
                    c1 = (int(p1[0] * reader.width), int(p1[1] * reader.height))
                    c2 = (int(p2[0] * reader.width), int(p2[1] * reader.height))
                    cv2.line(frame, c1, c2, (0, 255, 0), 2)
            
            # Draw current centroid if active
            last_p = points[-1]
            if last_p.frame_idx == i:
                cx = int(last_p.pos[0] * reader.width)
                cy = int(last_p.pos[1] * reader.height)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                cv2.putText(frame, f"ID:{track.id}", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Draw throw/peak events
        for event in events: # Changed from all_events to events
            # Display event label for a short duration after the event timestamp
            if event.timestamp <= timestamp <= event.timestamp + 0.5:
                # Map position if we want to draw it at the event location
                # For now just draw markers at the top/bottom
                color = (0, 255, 0) if event.event_type == EventType.THROW else (255, 0, 255) # Green for THROW, Magenta for PEAK
                label = "THROW" if event.event_type == EventType.THROW else "PEAK"
                y_pos = reader.height - 50 if event.event_type == EventType.THROW else 50 # THROW at bottom, PEAK at top
                
                cv2.putText(frame, f"{label} ID:{event.track_id}", (50, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Draw info
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Check if in a segment
        in_segment = any(s.start_time <= timestamp <= s.end_time for s in segments)
        if in_segment:
            cv2.putText(frame, "JUGGLING", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        out.write(frame)
        
    reader.release()
    out.release()
