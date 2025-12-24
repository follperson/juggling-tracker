import cv2
import numpy as np
from typing import List
from .tracker import Track
from .events import ThrowEvent
from .segmenter import Segment

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
                    p1 = tail[j].bbox
                    p2 = tail[j+1].bbox
                    c1 = (int((p1[0]+p1[2])/2), int((p1[1]+p1[3])/2))
                    c2 = (int((p2[0]+p2[2])/2), int((p2[1]+p2[3])/2))
                    cv2.line(frame, c1, c2, (0, 255, 0), 2)
            
            # Draw current bbox if active
            last_p = points[-1]
            if last_p.frame_idx == i:
                bbox = last_p.bbox
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                cv2.putText(frame, f"ID:{track.id}", (int(bbox[0]), int(bbox[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Draw events (flash)
        if i in event_frames:
            cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)
            cv2.putText(frame, "THROW!", (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw info
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Check if in a segment
        in_segment = any(s.start_time <= timestamp <= s.end_time for s in segments)
        if in_segment:
            cv2.putText(frame, "JUGGLING", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        out.write(frame)
        
    reader.release()
    out.release()
