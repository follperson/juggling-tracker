from typing import List, Tuple, Dict, Optional
import numpy as np
from enum import Enum
from scipy.optimize import linear_sum_assignment
from sqlmodel import SQLModel, Field
from .detector import Detection
from jugglecount.db.schema import TrackPoint

class Track(SQLModel):
    id: int = Field(primary_key=True)
    points: List[TrackPoint] = Field(default_factory=list)
    
    # Runtime only fields (not persisted to DB yet unless we update schema)
    # We can store them here but they won't be saved if SQLModel ignores extra fields.
    # Ideally should be in tracker runtime state if not changing schema.
    # But for simplicity let's rely on Python object persistence in memory.
    
    @property
    def last_point(self) -> TrackPoint:
        return self.points[-1]

class KalmanFilter:
    """
    A simple linear Kalman Filter for 2D tracking (Constant Velocity Model).
    State: [x, y, dx, dy]
    Supports optional gravity for parabolic Y-axis motion.
    """
    def __init__(self, initial_state: np.ndarray, initial_covariance: float = 1.0, 
                 process_noise: float = 0.01, measurement_noise: float = 0.1,
                 gravity: float = 0.0):
        # State vector [x, y, vx, vy]
        self.state = initial_state.astype(float)
        
        # Gravity constant (normalized units/frame^2) - positive = downward
        self.gravity = gravity
        
        # Covariance matrix P
        self.P = np.eye(4) * initial_covariance
        
        # Process Noise Q ( Uncertainty in the model )
        self.Q = np.eye(4) * process_noise
        # Allow more flexibility in velocity
        self.Q[2:4, 2:4] *= 10.0 
        
        # Measurement Noise R
        self.R = np.eye(2) * measurement_noise
        
        # Measurement Matrix H (we observe x, y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

    def predict(self, dt: float):
        # State Transition Matrix F
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # x = Fx
        self.state = F @ self.state
        
        # Apply gravity to vertical velocity (vy += g * dt)
        # Positive gravity = downward in screen coordinates (y increases downward)
        if self.gravity != 0:
            self.state[3] += self.gravity * dt
        
        # P = FPF' + Q
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement: np.ndarray):
        # y = z - Hx (Innovation)
        z = measurement
        y = z - (self.H @ self.state)
        
        # S = HPH' + R (Innovation Covariance)
        S = self.H @ self.P @ self.H.T + self.R
        
        # K = PH'S^-1 (Kalman Gain)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # x = x + Ky
        self.state = self.state + (K @ y)
        
        # P = (I - KH)P
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P


class TrackState(Enum):
    TENTATIVE = 0
    CONFIRMED = 1
    DELETED = 2

class SimpleTracker:
    """A robust tracker using Kalman Filter and Trajectory Prediction."""
    def __init__(self, max_distance: float = 120.0, max_inactive_frames: int = 10, min_hits: int = 3,
                 gravity: float = 0.001, enable_interpolation: bool = True):
        self.max_distance = max_distance
        self.max_inactive_frames = max_inactive_frames
        self.min_hits = min_hits
        self.gravity = gravity  # Normalized gravity (1/frame^2), tunable
        self.enable_interpolation = enable_interpolation
        
        self.tracks: List[Track] = []
        self.next_id = 0
        
        # Runtime state for filters and lifecycle
        self.filters: Dict[int, KalmanFilter] = {}
        self.track_states: Dict[int, TrackState] = {}
        self.inactive_counts: Dict[int, int] = {}
        self.hit_counts: Dict[int, int] = {}
        
        # Track effective size (width, height) in normalized coords
        self.track_sizes: Dict[int, Tuple[float, float]] = {}
        
        # Debug info: store last frame's predictions for visualization
        self.last_predictions: Dict[int, Tuple[float, float, float, float]] = {}
        self.last_velocities: Dict[int, Tuple[float, float]] = {}  # vx, vy per track

    def _init_track(self, detection: Detection, frame_idx: int, timestamp: float, width: int, height: int):
        # Bbox: x1, y1, x2, y2 (pixels)
        x1, y1, x2, y2 = detection.bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w_px = x2 - x1
        h_px = y2 - y1
        
        # Normalize
        norm_x = cx / width
        norm_y = cy / height
        norm_w = w_px / width
        norm_h = h_px / height
        
        # Create Track
        new_track = Track(
            id=self.next_id,
            points=[TrackPoint(
                frame_idx=frame_idx,
                timestamp=round(timestamp, 4),
                pos_x=round(norm_x, 4),
                pos_y=round(norm_y, 4),
                confidence=round(detection.confidence, 4)
            )]
        )
        
        # Init KF state [x, y, 0, 0] with gravity-aware model
        kf = KalmanFilter(
            initial_state=np.array([norm_x, norm_y, 0.0, 0.0]),
            measurement_noise=0.001,  # Tune this: lower = trust detection more
            gravity=self.gravity
        )
        
        self.tracks.append(new_track)
        self.filters[self.next_id] = kf
        self.track_states[self.next_id] = TrackState.TENTATIVE
        self.inactive_counts[self.next_id] = 0
        self.hit_counts[self.next_id] = 1
        self.track_sizes[self.next_id] = (norm_w, norm_h)
        
        if self.min_hits <= 1:
             self.track_states[self.next_id] = TrackState.CONFIRMED

        self.next_id += 1

    def predict_next_locations(self, next_timestamp: float) -> Dict[int, Tuple[float, float, float, float]]:
        """
        Predicts the normalized (x, y, w, h) location for each active track.
        """
        predictions = {}
        active_track_indices = [i for i, t in enumerate(self.tracks) 
                                if self.track_states[t.id] != TrackState.DELETED]
        
        for idx in active_track_indices:
            track = self.tracks[idx]
            kf = self.filters[track.id]
            
            # Identify dt from last POINT to next_timestamp
            last_p = track.last_point
            dt = next_timestamp - last_p.timestamp
            
            # Predict pos
            pred_x = kf.state[0] + kf.state[2] * dt
            pred_y = kf.state[1] + kf.state[3] * dt
            
            # Clip
            pred_x = max(0.0, min(1.0, pred_x))
            pred_y = max(0.0, min(1.0, pred_y))
            
            # Get estimated size
            w_norm, h_norm = self.track_sizes.get(track.id, (0.05, 0.05))
            
            predictions[track.id] = (pred_x, pred_y, w_norm, h_norm)
        return predictions

    def update(self, detections: List[Detection], frame_idx: int, timestamp: float, width: int, height: int):
        # 1. Prepare detections (centroids in normalized coords)
        # Also store size for update
        det_centroids = []
        det_sizes = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            w_px = x2 - x1
            h_px = y2 - y1
            
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            
            det_centroids.append(np.array([cx / width, cy / height]))
            det_sizes.append((w_px / width, h_px / height))

        # 2. Predict step for all active tracks
        active_track_indices = [i for i, t in enumerate(self.tracks) 
                                if self.track_states[t.id] != TrackState.DELETED]
        
        for idx in active_track_indices:
            track = self.tracks[idx]
            kf = self.filters[track.id]
            
            # dt
            last_p = track.last_point
            dt = timestamp - last_p.timestamp
            
            kf.predict(dt if dt > 0 else 1/30.0)

        # 3. Build Cost Matrix
        cost_matrix = np.zeros((len(active_track_indices), len(detections)))
        
        for i, t_idx in enumerate(active_track_indices):
            track = self.tracks[t_idx]
            kf = self.filters[track.id]
            
            # Predicted pos from KF
            pred_x, pred_y = kf.state[0], kf.state[1]
            
            pred_px_x = pred_x * width
            pred_px_y = pred_y * height
            
            for j, det_norm in enumerate(det_centroids):
                det_px_x = det_norm[0] * width
                det_px_y = det_norm[1] * height
                
                dist = np.sqrt((pred_px_x - det_px_x)**2 + (pred_px_y - det_px_y)**2)
                cost_matrix[i, j] = dist

        # 4. Hungarian Algorithm
        if len(active_track_indices) > 0 and len(detections) > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            row_ind, col_ind = [], []

        # 5. Assignments
        matched_track_indices = set()
        matched_det_indices = set()
        
        # EMA alpha for size update (e.g. 0.1 means slow adaptation, 0.5 fast)
        alpha_size = 0.3
        
        # Clear debug info for this frame
        self.last_predictions.clear()
        self.last_velocities.clear()
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.max_distance:
                t_idx = active_track_indices[r]
                track = self.tracks[t_idx]
                det = detections[c]
                kf = self.filters[track.id]
                
                # Store debug info before update
                w_norm, h_norm = self.track_sizes.get(track.id, (0.05, 0.05))
                self.last_predictions[track.id] = (kf.state[0], kf.state[1], w_norm, h_norm)
                self.last_velocities[track.id] = (kf.state[2], kf.state[3])
                
                # Update KF
                kf.update(det_centroids[c])
                
                # Update Size (EMA)
                curr_w, curr_h = self.track_sizes.get(track.id, (0.05, 0.05))
                meas_w, meas_h = det_sizes[c]
                
                new_w = curr_w * (1 - alpha_size) + meas_w * alpha_size
                new_h = curr_h * (1 - alpha_size) + meas_h * alpha_size
                self.track_sizes[track.id] = (new_w, new_h)
                
                # Update Track Record
                track.points.append(TrackPoint(
                    frame_idx=frame_idx,
                    timestamp=round(timestamp, 4),
                    pos_x=round(det_centroids[c][0], 4),
                    pos_y=round(det_centroids[c][1], 4),
                    confidence=round(det.confidence, 4),
                    is_interpolated=False
                ))
                
                # Update Counts
                self.inactive_counts[track.id] = 0
                self.hit_counts[track.id] += 1
                
                # Promote to Confirmed
                if self.track_states[track.id] == TrackState.TENTATIVE and self.hit_counts[track.id] >= self.min_hits:
                    self.track_states[track.id] = TrackState.CONFIRMED
                
                matched_track_indices.add(t_idx)
                matched_det_indices.add(c)
                
        # 6. Handle Unmatched Tracks - Generate synthetic interpolated points
        for t_idx in active_track_indices:
            if t_idx not in matched_track_indices:
                track = self.tracks[t_idx]
                kf = self.filters[track.id]
                
                # Store debug info for unmatched tracks too
                w_norm, h_norm = self.track_sizes.get(track.id, (0.05, 0.05))
                self.last_predictions[track.id] = (kf.state[0], kf.state[1], w_norm, h_norm)
                self.last_velocities[track.id] = (kf.state[2], kf.state[3])
                
                self.inactive_counts[track.id] += 1
                
                # Generate synthetic point if interpolation is enabled
                # The ball can't disappear - use Kalman prediction as synthetic detection
                if self.enable_interpolation and self.track_states[track.id] == TrackState.CONFIRMED:
                    # Use predicted position from Kalman filter (already computed in predict step)
                    pred_x = max(0.0, min(1.0, kf.state[0]))
                    pred_y = max(0.0, min(1.0, kf.state[1]))
                    
                    # Add synthetic point with low confidence to indicate interpolation
                    synthetic_conf = 0.1 * (1.0 - self.inactive_counts[track.id] / self.max_inactive_frames)
                    
                    track.points.append(TrackPoint(
                        frame_idx=frame_idx,
                        timestamp=round(timestamp, 4),
                        pos_x=round(pred_x, 4),
                        pos_y=round(pred_y, 4),
                        confidence=round(max(0.01, synthetic_conf), 4),
                        is_interpolated=True
                    ))
                
                # Check for Deletion
                if self.inactive_counts[track.id] >= self.max_inactive_frames:
                    self.track_states[track.id] = TrackState.DELETED
                    
        # 7. Handle New Detections
        for i, det in enumerate(detections):
            if i not in matched_det_indices:
                self._init_track(det, frame_idx, timestamp, width, height)

    def get_active_tracks(self) -> List[Track]:
        """Returns only CONFIRMED, active tracks."""
        return [t for t in self.tracks if self.track_states.get(t.id) == TrackState.CONFIRMED]
    
    def get_debug_info(self) -> Dict:
        """Returns debug information for visualization."""
        return {
            'predictions': self.last_predictions.copy(),
            'velocities': self.last_velocities.copy(),
            'track_states': {tid: state.name for tid, state in self.track_states.items()},
            'inactive_counts': self.inactive_counts.copy(),
            'hit_counts': self.hit_counts.copy(),
            'track_sizes': self.track_sizes.copy()
        }
