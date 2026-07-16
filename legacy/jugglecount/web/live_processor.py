import av
import cv2
import numpy as np
import time
from typing import List, Tuple, Dict
from streamlit_webrtc import VideoProcessorBase
from jugglecount.core.detector import YOLOBallDetector, Detection
from jugglecount.core.tracker import SimpleTracker, TrackState
from jugglecount.core.signature_detector import SignatureBallDetector
# Import shared context type (using string forward ref or direct import)
from jugglecount.web.voice_processor import JugglingSharedContext
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)

class JugglingProcessor(VideoProcessorBase):
    def __init__(self, shared_context: JugglingSharedContext = None):
        self.shared_context = shared_context
        # Initialize detectors
        self.detector_yolo = YOLOBallDetector(imgsz=480)
        # Robust signature detector with edge verification to reject foreheads/smooth surfaces
        self.detector_signature = SignatureBallDetector(
            confidence_threshold=0.6, h_bins=36, sigma=1.0, 
            min_area=400, min_circularity=0.65,
            require_edges=True, min_edge_ratio=0.3
        )
        self.active_detector_type = "YOLO" 
        
        # Background Subtractor (MOG2)
        # Tuning: Increase varThreshold to 50 (was 16) to reduce noise
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        
        # Tracker with physics-based interpolation (gravity-aware)
        self.tracker = SimpleTracker(max_distance=100, max_inactive_frames=10, gravity=0.002, enable_interpolation=True)
        self.full_scan_interval = 30 # Frames
        
        self.throw_count = 0
        self.peak_count = 0
        self.track_data = {} 
        self.track_last_peak_time = {} 
        
        self.frame_count = 0
        self.start_time = time.time()
        
        # FPS tracking for debug display
        self.fps_window: List[float] = []
        self.last_frame_time = time.time()
        
        # Detection debug info for current frame
        self.last_raw_detections: List[Detection] = []
        self.last_filtered_detections: List[Detection] = []
        self.last_scan_mode = "FULL"  # "FULL" or "ROI"
        
        # Registration State
        self.registration_mode = False
        self.allowed_class_ids = [32] 
        self.last_detected_object = None 
        self.pending_registration_label = None 
        self.pending_bg_registration = False # New Flag
        
        # Debug State
        self.debug_mode = False 
        
        logger.info("JugglingProcessor initialized")

    @property
    def detector(self):
        if self.active_detector_type == "Signature":
            return self.detector_signature
        return self.detector_yolo

    def set_detector_type(self, type_name: str):
        if type_name in ["YOLO", "Signature"]:
            self.active_detector_type = type_name

    def set_registration_mode(self, enabled: bool):
        self.registration_mode = enabled
        self.last_detected_object = None
        
    def set_debug_mode(self, enabled: bool):
        self.debug_mode = enabled

    def set_background_threshold(self, threshold: float):
        self.bg_subtractor.setVarThreshold(threshold)

    def set_signature_confidence(self, confidence: float):
        self.detector_signature.confidence_threshold = confidence

    def set_yolo_model(self, model_name: str):
        """Switch YOLO model (e.g., yolov8n.pt, yolov8m.pt)."""
        if self.active_detector_type == "YOLO":
            self.detector_yolo.load_model(model_name)

    def set_yolo_imgsz(self, size: int):
        """Update YOLO inference resolution."""
        if self.active_detector_type == "YOLO":
            self.detector_yolo.set_imgsz(size)

    def set_yolo_confidence(self, conf: float):
        """Update YOLO confidence threshold."""
        if self.active_detector_type == "YOLO":
            self.detector_yolo.set_confidence(conf)

    def add_class_id(self, class_id: int):
        if class_id not in self.allowed_class_ids:
            self.allowed_class_ids.append(class_id)
            self.detector_yolo.update_target_classes(self.allowed_class_ids)

    def set_tracker_gravity(self, gravity: float):
        """Set the gravity constant for physics-based tracking (normalized units/frame^2)."""
        self.tracker.gravity = gravity
        # Update existing Kalman filters
        for kf in self.tracker.filters.values():
            kf.gravity = gravity
    
    def set_tracker_interpolation(self, enabled: bool):
        """Enable/disable synthetic point generation during detection gaps."""
        self.tracker.enable_interpolation = enabled

    def trigger_signature_registration(self, label: str):
        self.pending_registration_label = label
        
    def trigger_background_registration(self):
        """Trigger background color registration on next frame."""
        self.pending_bg_registration = True

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            
            # 0. Mirroring
            img = cv2.flip(img, 1)
            
            # Handle Background Registration Trigger
            if self.pending_bg_registration and self.active_detector_type == "Signature":
                 self.detector_signature.register_background(img)
                 self.pending_bg_registration = False
                 # Visual Flash or Text?
                 cv2.putText(img, "BACKGROUND REGISTERED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                 return av.VideoFrame.from_ndarray(img, format="bgr24")

            height, width = img.shape[:2]
            
            # Robust time handling
            try:
                current_time = frame.time
            except Exception:
                current_time = None
            
            if current_time is None:
                current_time = time.time() - self.start_time
                
            # 0.5. Background Subtraction
            fg_mask = self.bg_subtractor.apply(img)
            
            # Clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            # Helper: Check Voice Commands
            if self.shared_context:
                if self.shared_context.check_and_clear_ball_trigger():
                    self.trigger_signature_registration("Ball (Voice)")
                    self.registration_mode = True # Force reticle ON if off
                    
                if self.shared_context.check_and_clear_background_trigger():
                    self.trigger_background_registration()
                    
                if self.shared_context.check_and_clear_reset_trigger():
                    self.detector_signature.signatures.clear()
                    self.detector_signature.class_id_map.clear()
            
            
            # Note: Debug overlay is now applied after detection/tracking (see visualization section)
            
            if self.registration_mode:
                # Draw Reticle
                cx, cy = width // 2, height // 2
                rx1, ry1 = cx - 50, cy - 50
                rx2, ry2 = cx + 50, cy + 50
                
                # Registration Logic based on Type
                if self.active_detector_type == "YOLO":
                    detections = self.detector_yolo.detect(img, allowed_classes=None)
                    best_det = None
                    min_dist = float("inf")
                    
                    for det in detections:
                        dcx = (det.bbox[0] + det.bbox[2]) / 2
                        dcy = (det.bbox[1] + det.bbox[3]) / 2
                        
                        if (rx1 < dcx < rx2) and (ry1 < dcy < ry2):
                             dist = (dcx - cx)**2 + (dcy - cy)**2
                             if dist < min_dist:
                                 min_dist = dist
                                 best_det = det
                                 
                    if best_det:
                        class_name = self.detector_yolo.class_names[best_det.class_id]
                        self.last_detected_object = (class_name, best_det.class_id)
                        cv2.putText(img, f"Found: {class_name}", (cx - 60, cy + 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                        cv2.rectangle(img, (int(best_det.bbox[0]), int(best_det.bbox[1])), (int(best_det.bbox[2]), int(best_det.bbox[3])), (0, 255, 0), 3)
                
                elif self.active_detector_type == "Signature":
                    if self.pending_registration_label:
                        self.detector_signature.register_object(img, (rx1, ry1, rx2, ry2), self.pending_registration_label)
                        # Get sample count for feedback
                        reg_info = self.detector_signature.get_registration_info(self.pending_registration_label)
                        sample_count = reg_info['sample_count'] if reg_info else 1
                        self.pending_registration_label = None 
                        cv2.putText(img, f"REGISTERED (Sample {sample_count})", (cx - 100, cy + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(img, "Register at different distances!", (cx - 120, cy + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)
                    else:
                        # Show current signature status
                        sig_count = len(self.detector_signature.signatures)
                        if sig_count > 0:
                            cv2.putText(img, f"Signatures: {sig_count}", (cx - 50, cy + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cv2.putText(img, "Hold object here + say 'register ball'", (cx - 150, cy + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # Draw Reticle Overlay
                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
                cv2.line(img, (cx, cy - 60), (cx, cy + 60), (0, 255, 255), 2)
                cv2.line(img, (cx - 60, cy), (cx + 60, cy), (0, 255, 255), 2)
                
                return av.VideoFrame.from_ndarray(img, format="bgr24")


            # 1. Detection
            detections = []
            raw_detections_for_debug = []
            
            # Determine Scan Mode
            # Force full scan periodically OR if tracker has lost all tracks (to recover)
            do_full_scan = (self.frame_count % self.full_scan_interval == 0) or (not self.tracker.tracks)
            
            if self.active_detector_type == "YOLO":
                # YOLO always full scan for now (fast enough)
                do_full_scan = True
            
            use_rois = False
            rois_for_debug = []  # Store ROIs for debug visualization
            
            # ROI Tracking (Signature only usually)
            if not do_full_scan and self.active_detector_type == "Signature":
                predictions = self.tracker.predict_next_locations(current_time)
                
                if predictions:
                    # Construct ROIs from predictions
                    rois = []
                    
                    for tid, (px, py, pw, ph) in predictions.items():
                        # Dynamic ROI: 4.0x object size
                        roi_w_norm = pw * 4.0
                        roi_h_norm = ph * 4.0
                        rois.append((px, py, roi_w_norm, roi_h_norm))
                    
                    if rois:
                        use_rois = True
                        rois_for_debug = rois.copy()
                        self.last_scan_mode = "ROI"
                        
                        # Run on ROIs
                        raw_detections = self.detector_signature.detect_from_rois(img, rois)
                        raw_detections_for_debug = raw_detections.copy()
                        
                        # Filter Detections (Hand/Noise Rejection)
                        detections = []
                        for d in raw_detections:
                            bx1, by1, bx2, by2 = map(int, d.bbox)
                            area = (bx2-bx1)*(by2-by1)
                            aspect = (bx2-bx1)/(by2-by1) if (by2-by1) > 0 else 0
                            
                            # Hand/Noise Rejection:
                            if area > 800 and 0.8 < aspect < 1.2:
                                detections.append(d)
            
            # Fallback / Full Scan 
            if do_full_scan or (use_rois and len(detections) == 0):
                 self.last_scan_mode = "FULL"
                 if self.active_detector_type == "YOLO":
                      detections = self.detector.detect(img, allowed_classes=self.allowed_class_ids)
                      raw_detections_for_debug = detections.copy()
                 else:
                      detections = self.detector_signature.detect(img, mask=fg_mask)
                      raw_detections_for_debug = detections.copy()
            
            # Store debug info
            self.last_raw_detections = raw_detections_for_debug
            self.last_filtered_detections = detections
                      
            # Merge logic implies if we did ROI, we accept it. 
            # If we missed, we might want to full scan? 
            # Let's keep it simple: If ROI yielded detections, good. 
            # If not, and we have tracks (expected balls), maybe we lost them. 
            # But we also need to find NEW balls. 
            # Hybrid: Run ROI. Run Full Scan on *remainder*? Expense.
            
            # Revised Plan based on User Request "Only consider objects within trajectory":
            # This implies strict filtering. 
            # If we have tracks, we should PREFER ROI hits.
            # But we still need to initialize. 
            
            # Let's stick to: If tracks exist, use ROIs. 
            # PERIODICALLY (e.g every 5 frames) run full scan to recover/find new.
            
            if use_rois and self.active_detector_type == "Signature":
                # If we are in "Tracking Mode", we might skip full scan to save noise
                # But we need to find new balls. 
                # Let's run full scan every 5 frames to find new stuff.
                if self.frame_count % 5 == 0:
                     full_dets = self.detector_signature.detect(img, mask=fg_mask)
                     # Merge? Or strict filter?
                     # If we just append, we get noise every 5 frames.
                     # We should only add full_dets that are NOT near existing tracks?
                     # This is getting complex.
                     # Simple approach: ROI detection is high priority. 
                     detections.extend(full_dets) 
                     # Tracker Handle duplicates? SimpleTracker might double count?
                     # SimpleTracker handles matching.
                     pass
            
            # 2. Tracking
            self.tracker.update(detections, self.frame_count, current_time, width, height)
            self.frame_count += 1
            
            # 3. Process Tracks
            active_tracks = [
                t for t in self.tracker.tracks 
                if self.tracker.inactive_counts.get(t.id, 0) < self.tracker.max_inactive_frames
            ]
            
            for track in active_tracks:
                if track.id not in self.track_data:
                    self.track_data[track.id] = []
                    self.track_last_peak_time[track.id] = 0.0
                    
                if track.points:
                    latest = track.points[-1]
                    self.track_data[track.id].append((latest.timestamp, latest.pos_y))
                    if len(self.track_data[track.id]) > 20:
                        self.track_data[track.id].pop(0)
                        
                    history = self.track_data[track.id]
                    if len(history) >= 5:
                        y_vals = np.array([-p[1] for p in history])
                        t_vals = np.array([p[0] for p in history])
                        
                        indices, _ = find_peaks(y_vals, prominence=0.05)
                        
                        for idx in indices:
                            peak_time = t_vals[idx]
                            last_peak = self.track_last_peak_time[track.id]
                            
                            if peak_time > last_peak:
                                self.peak_count += 1
                                self.track_last_peak_time[track.id] = peak_time
                                
            # 4. Visualization
            
            # Update FPS tracking
            current_frame_time = time.time()
            frame_dt = current_frame_time - self.last_frame_time
            self.last_frame_time = current_frame_time
            if frame_dt > 0:
                self.fps_window.append(1.0 / frame_dt)
                if len(self.fps_window) > 30:
                    self.fps_window.pop(0)
            
            # Get tracker debug info
            debug_info = self.tracker.get_debug_info()
            
            if self.debug_mode:
                # === COMPREHENSIVE DEBUG OVERLAY ===
                
                # 4.1 Draw ROI search regions (semi-transparent)
                overlay = img.copy()
                predictions = debug_info.get('predictions', {})
                for tid, (px, py, pw, ph) in predictions.items():
                    # ROI zone (4x object size, matching detection logic)
                    roi_w = int(pw * 4.0 * width)
                    roi_h = int(ph * 4.0 * height)
                    roi_cx = int(px * width)
                    roi_cy = int(py * height)
                    roi_x1 = max(0, roi_cx - roi_w // 2)
                    roi_y1 = max(0, roi_cy - roi_h // 2)
                    roi_x2 = min(width, roi_x1 + roi_w)
                    roi_y2 = min(height, roi_y1 + roi_h)
                    
                    # Color by track state
                    state = debug_info.get('track_states', {}).get(tid, 'DELETED')
                    if state == 'CONFIRMED':
                        color = (0, 255, 0)  # Green
                    elif state == 'TENTATIVE':
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (128, 128, 128)  # Gray
                    
                    cv2.rectangle(overlay, (roi_x1, roi_y1), (roi_x2, roi_y2), color, 2)
                
                # Blend overlay onto image
                cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
                
                # 4.2 Draw Kalman predictions with velocity arrows
                for tid, (px, py, pw, ph) in predictions.items():
                    pred_cx = int(px * width)
                    pred_cy = int(py * height)
                    
                    # Draw predicted position (hollow circle with crosshair)
                    cv2.circle(img, (pred_cx, pred_cy), 12, (255, 0, 255), 1)  # Magenta prediction
                    cv2.line(img, (pred_cx - 8, pred_cy), (pred_cx + 8, pred_cy), (255, 0, 255), 1)
                    cv2.line(img, (pred_cx, pred_cy - 8), (pred_cx, pred_cy + 8), (255, 0, 255), 1)
                    
                    # Draw velocity arrow
                    velocities = debug_info.get('velocities', {})
                    if tid in velocities:
                        vx, vy = velocities[tid]
                        # Scale velocity for visualization (adjust multiplier as needed)
                        arrow_scale = 500
                        arrow_end_x = int(pred_cx + vx * arrow_scale)
                        arrow_end_y = int(pred_cy + vy * arrow_scale)
                        cv2.arrowedLine(img, (pred_cx, pred_cy), (arrow_end_x, arrow_end_y), (255, 0, 255), 2, tipLength=0.3)
                
                # 4.3 Draw raw detections (blue - before filtering)
                for det in self.last_raw_detections:
                    x1, y1, x2, y2 = map(int, det.bbox)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 0), 1)  # Blue
                    cv2.putText(img, f"{det.confidence:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 0), 1)
                
                # 4.4 Draw filtered detections (green - matched to tracks)
                for det in self.last_filtered_detections:
                    x1, y1, x2, y2 = map(int, det.bbox)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green
                
                # 4.5 Draw track info panels
                for track in active_tracks:
                    if track.points:
                        pt = track.points[-1]
                        cx, cy = int(pt.pos_x * width), int(pt.pos_y * height)
                        
                        # Track state info
                        state = debug_info.get('track_states', {}).get(track.id, 'UNK')
                        hits = debug_info.get('hit_counts', {}).get(track.id, 0)
                        inactive = debug_info.get('inactive_counts', {}).get(track.id, 0)
                        
                        # State abbreviation
                        state_abbr = 'C' if state == 'CONFIRMED' else ('T' if state == 'TENTATIVE' else 'D')
                        
                        # Color based on interpolation
                        if pt.is_interpolated:
                            circle_color = (0, 165, 255)  # Orange for interpolated
                            label = f"{track.id}({state_abbr}) I"
                        else:
                            circle_color = (0, 255, 0)  # Green for real detection
                            label = f"{track.id}({state_abbr})"
                        
                        cv2.circle(img, (cx, cy), 20, circle_color, 2)
                        cv2.putText(img, label, (cx - 25, cy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, circle_color, 1)
                        cv2.putText(img, f"H:{hits} I:{inactive}", (cx - 25, cy + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                        
                        # Draw trajectory with interpolation distinction
                        points = track.points[-15:]
                        for i in range(1, len(points)):
                            p1 = points[i-1]
                            p2 = points[i]
                            pt1 = (int(p1.pos_x * width), int(p1.pos_y * height))
                            pt2 = (int(p2.pos_x * width), int(p2.pos_y * height))
                            
                            # Different color for interpolated segments
                            if p2.is_interpolated:
                                line_color = (0, 165, 255)  # Orange dashed-ish
                                cv2.line(img, pt1, pt2, line_color, 1)
                            else:
                                line_color = (0, 255, 255)  # Yellow for real
                                cv2.line(img, pt1, pt2, line_color, 2)
                
                # 4.6 Draw metrics panel (top-left)
                avg_fps = sum(self.fps_window) / len(self.fps_window) if self.fps_window else 0
                metrics = [
                    f"FPS: {avg_fps:.1f}",
                    f"Mode: {self.last_scan_mode}",
                    f"Tracks: {len(active_tracks)}",
                    f"Dets: {len(self.last_filtered_detections)}/{len(self.last_raw_detections)}",
                    f"Peaks: {self.peak_count}"
                ]
                
                # Draw semi-transparent background for metrics
                panel_h = 25 * len(metrics) + 10
                cv2.rectangle(img, (10, 10), (180, panel_h), (0, 0, 0), -1)
                cv2.rectangle(img, (10, 10), (180, panel_h), (100, 100, 100), 1)
                
                for i, text in enumerate(metrics):
                    cv2.putText(img, text, (15, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                    
            else:
                # === NORMAL VISUALIZATION (non-debug) ===
                for track in active_tracks:
                    if track.points:
                        pt = track.points[-1]
                        cx, cy = pt.pos_x * width, pt.pos_y * height
                        r = 20
                        
                        # Different color for interpolated points
                        if pt.is_interpolated:
                            color = (0, 165, 255)  # Orange
                        else:
                            color = (0, 255, 0)  # Green
                            
                        cv2.circle(img, (int(cx), int(cy)), r, color, 2)
                        cv2.putText(img, str(track.id), (int(cx), int(cy)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        points = track.points[-15:]
                        for i in range(1, len(points)):
                            p1 = points[i-1]
                            p2 = points[i]
                            pt1 = (int(p1.pos_x * width), int(p1.pos_y * height))
                            pt2 = (int(p2.pos_x * width), int(p2.pos_y * height))
                            line_color = (0, 165, 255) if p2.is_interpolated else (0, 255, 255)
                            cv2.line(img, pt1, pt2, line_color, 2)

                cv2.putText(img, f"Peaks: {self.peak_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            logger.error(f"Error in recv: {e}", exc_info=True)
            return frame
