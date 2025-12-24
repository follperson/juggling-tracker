from typing import List, Tuple, Dict, Optional
import numpy as np
from .detector import Detection
from .schema import Track, TrackPoint

class SimpleTracker:
    """A very basic distance-based tracker for the baseline."""
    def __init__(self, max_distance: float = 100.0, max_inactive_frames: int = 5):
        self.max_distance = max_distance
        self.max_inactive_frames = max_inactive_frames
        self.tracks: List[Track] = []
        self.next_id = 0
        self.inactive_counts: Dict[int, int] = {}

    def update(self, detections: List[Detection], frame_idx: int, timestamp: float):
        current_active_ids = []
        
        # Calculate centroids for current detections
        det_centroids = []
        for det in detections:
            x_mid = (det.bbox[0] + det.bbox[2]) / 2
            y_mid = (det.bbox[1] + det.bbox[3]) / 2
            det_centroids.append((x_mid, y_mid))

        # Match with existing tracks
        matched_detections = [False] * len(detections)
        active_tracks = [t for t in self.tracks if self.inactive_counts.get(t.id, 0) < self.max_inactive_frames]

        for track in active_tracks:
            last_p = track.last_point
            last_x = (last_p.bbox[0] + last_p.bbox[2]) / 2
            last_y = (last_p.bbox[1] + last_p.bbox[3]) / 2
            
            best_dist = float('inf')
            best_det_idx = -1
            
            for i, (dx, dy) in enumerate(det_centroids):
                if matched_detections[i]:
                    continue
                dist = np.sqrt((dx - last_x)**2 + (dy - last_y)**2)
                if dist < best_dist and dist < self.max_distance:
                    best_dist = dist
                    best_det_idx = i
            
            if best_det_idx != -1:
                track.points.append(TrackPoint(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    bbox=detections[best_det_idx].bbox,
                    confidence=detections[best_det_idx].confidence
                ))
                matched_detections[best_det_idx] = True
                self.inactive_counts[track.id] = 0
            else:
                self.inactive_counts[track.id] = self.inactive_counts.get(track.id, 0) + 1

        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if not matched_detections[i]:
                new_track = Track(
                    id=self.next_id,
                    points=[TrackPoint(
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        bbox=det.bbox,
                        confidence=det.confidence
                    )]
                )
                self.tracks.append(new_track)
                self.inactive_counts[self.next_id] = 0
                self.next_id += 1
