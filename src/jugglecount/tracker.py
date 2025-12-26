from typing import List, Tuple, Dict, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment
from .detector import Detection
from .schema import Track, TrackPoint

class SimpleTracker:
    """A robust tracker using Hungarian Algorithm and Trajectory Prediction."""
    def __init__(self, max_distance: float = 120.0, max_inactive_frames: int = 15):
        self.max_distance = max_distance
        self.max_inactive_frames = max_inactive_frames
        self.tracks: List[Track] = []
        self.next_id = 0
        self.inactive_counts: Dict[int, int] = {}
        # Stores velocity as (dx, dy) in pixels per second
        self.track_velocities: Dict[int, Tuple[float, float]] = {}

    def _get_velocity(self, track: Track) -> Tuple[float, float]:
        if len(track.points) < 2:
            return (0.0, 0.0)
        p_last = track.points[-1]
        p_prev = track.points[-2]
        dt = p_last.timestamp - p_prev.timestamp
        if dt <= 0:
            return (0.0, 0.0)
        
        # Calculate velocity in normalized coordinates (for consistency)
        dx = (p_last.pos[0] - p_prev.pos[0]) / dt
        dy = (p_last.pos[1] - p_prev.pos[1]) / dt
        return (dx, dy)

    def update(self, detections: List[Detection], frame_idx: int, timestamp: float, width: int, height: int):
        # 1. Prepare detections (centroids in pixels)
        det_centroids = []
        for det in detections:
            x_mid = (det.bbox[0] + det.bbox[2]) / 2
            y_mid = (det.bbox[1] + det.bbox[3]) / 2
            det_centroids.append((x_mid, y_mid))

        # 2. Get active tracks and their predicted positions
        active_track_indices = [i for i, t in enumerate(self.tracks) 
                                if self.inactive_counts.get(t.id, 0) < self.max_inactive_frames]
        
        if not active_track_indices or not detections:
            self._handle_unmatched(detections, active_track_indices, [], frame_idx, timestamp, width, height)
            return

        # 3. Build cost matrix (distance between predicted track position and detection)
        cost_matrix = np.zeros((len(active_track_indices), len(detections)))
        for i, t_idx in enumerate(active_track_indices):
            track = self.tracks[t_idx]
            last_p = track.last_point
            
            # Prediction: last_pos + velocity * dt
            vel = self._get_velocity(track)
            dt = timestamp - last_p.timestamp
            
            pred_x_norm = last_p.pos[0] + vel[0] * dt
            pred_y_norm = last_p.pos[1] + vel[1] * dt
            
            # Clip prediction to frame (optional)
            pred_x = pred_x_norm * width
            pred_y = pred_y_norm * height
            
            for j, (dx, dy) in enumerate(det_centroids):
                dist = np.sqrt((dx - pred_x)**2 + (dy - pred_y)**2)
                cost_matrix[i, j] = dist

        # 4. Hungarian Algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 5. Filter matches by max_distance and update
        matches = []
        assigned_tracks = set()
        assigned_detections = set()
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.max_distance:
                matches.append((active_track_indices[r], c))
                assigned_tracks.add(active_track_indices[r])
                assigned_detections.add(c)

        # 6. Finalize updates
        self._handle_unmatched(detections, active_track_indices, matches, frame_idx, timestamp, width, height)

    def _handle_unmatched(self, detections, active_track_indices, matches, frame_idx, timestamp, width, height):
        matched_track_ids = set()
        matched_det_indices = set()
        
        # Update matched tracks
        for t_idx, d_idx in matches:
            track = self.tracks[t_idx]
            det = detections[d_idx]
            cx = (det.bbox[0] + det.bbox[2]) / 2
            cy = (det.bbox[1] + det.bbox[3]) / 2
            
            track.points.append(TrackPoint(
                frame_idx=frame_idx,
                timestamp=round(timestamp, 4),
                pos=(round(cx / width, 4), round(cy / height, 4)),
                confidence=round(det.confidence, 4)
            ))
            self.inactive_counts[track.id] = 0
            matched_track_ids.add(t_idx)
            matched_det_indices.add(d_idx)

        # Increment inactive counts for unmatched tracks
        for t_idx in active_track_indices:
            if t_idx not in matched_track_ids:
                self.inactive_counts[self.tracks[t_idx].id] += 1

        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_det_indices:
                cx = (det.bbox[0] + det.bbox[2]) / 2
                cy = (det.bbox[1] + det.bbox[3]) / 2
                new_track = Track(
                    id=self.next_id,
                    points=[TrackPoint(
                        frame_idx=frame_idx,
                        timestamp=round(timestamp, 4),
                        pos=(round(cx / width, 4), round(cy / height, 4)),
                        confidence=round(det.confidence, 4)
                    )]
                )
                self.tracks.append(new_track)
                self.inactive_counts[self.next_id] = 0
                self.next_id += 1
