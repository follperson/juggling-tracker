from typing import List
import numpy as np
from .schema import Track, TrackPoint

def interpolate_tracks(tracks: List[Track], max_gap_frames: int = 10) -> List[Track]:
    """
    Fills in gaps in tracks by linear interpolation.
    """
    processed_tracks = []
    
    for track in tracks:
        if len(track.points) < 2:
            processed_tracks.append(track)
            continue
            
        new_points = []
        for i in range(len(track.points) - 1):
            p1 = track.points[i]
            p2 = track.points[i+1]
            new_points.append(p1)
            
            gap = p2.frame_idx - p1.frame_idx
            if 1 < gap <= max_gap_frames:
                # Interpolate
                for j in range(1, gap):
                    alpha = j / gap
                    interp_f_idx = p1.frame_idx + j
                    interp_timestamp = p1.timestamp + (p2.timestamp - p1.timestamp) * alpha
                    interp_x = p1.pos[0] + (p2.pos[0] - p1.pos[0]) * alpha
                    interp_y = p1.pos[1] + (p2.pos[1] - p1.pos[1]) * alpha
                    interp_conf = p1.confidence + (p2.confidence - p1.confidence) * alpha
                    
                    new_points.append(TrackPoint(
                        frame_idx=interp_f_idx,
                        timestamp=round(interp_timestamp, 4),
                        pos=(round(interp_x, 4), round(interp_y, 4)),
                        confidence=round(interp_conf, 4)
                    ))
        
        new_points.append(track.points[-1])
        track.points = new_points
        processed_tracks.append(track)
        
    return processed_tracks
