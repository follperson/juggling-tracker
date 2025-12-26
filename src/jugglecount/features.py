from typing import List, Dict
import numpy as np
from .schema import Track, FrameFeatures

def compute_features(tracks: List[Track], frame_count: int, fps: float) -> List[FrameFeatures]:
    features_list = []
    
    # Pre-calculate velocities per track
    track_velocities: Dict[int, Dict[int, float]] = {}
    for track in tracks:
        track_velocities[track.id] = {}
        for i in range(1, len(track.points)):
            p1 = track.points[i-1]
            p2 = track.points[i]
            dt = p2.timestamp - p1.timestamp
            if dt > 0:
                y1 = p1.pos[1]
                y2 = p2.pos[1]
                vy = (y2 - y1) / dt
                track_velocities[track.id][p2.frame_idx] = vy

    for f_idx in range(frame_count):
        timestamp = f_idx / fps
        active_tracks_at_frame = []
        for track in tracks:
            # Check if track has a point at this frame
            for p in track.points:
                if p.frame_idx == f_idx:
                    active_tracks_at_frame.append(track)
                    break
        
        num_detections = len(active_tracks_at_frame)
        
        # Compute velocities for active tracks at this frame
        current_vys = []
        for track in active_tracks_at_frame:
            if f_idx in track_velocities[track.id]:
                current_vys.append(track_velocities[track.id][f_idx])
        
        mean_v_y = np.mean(current_vys) if current_vys else 0.0
        std_v_y = np.std(current_vys) if current_vys else 0.0
        
        features_list.append(FrameFeatures(
            frame_idx=f_idx,
            timestamp=timestamp,
            num_detections=num_detections,
            num_active_tracks=len(active_tracks_at_frame),
            mean_velocity_y=float(mean_v_y),
            std_velocity_y=float(std_v_y)
        ))
        
    return features_list
