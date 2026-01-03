from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
import threading
from .detector import BallDetector, Detection


class ObjectSignature:
    """Stores color and texture signatures for an object."""
    def __init__(self, label: str, class_id: int):
        self.label = label
        self.class_id = class_id
        self.color_samples: List[np.ndarray] = []  # Hue histograms from multiple registrations
        self.texture_samples: List[np.ndarray] = []  # LBP histograms
        self.merged_color_hist: Optional[np.ndarray] = None
        self.merged_texture_hist: Optional[np.ndarray] = None
        
    def add_sample(self, color_hist: np.ndarray, texture_hist: np.ndarray):
        """Add a new registration sample."""
        self.color_samples.append(color_hist.copy())
        self.texture_samples.append(texture_hist.copy())
        self._recompute_merged()
    
    def _recompute_merged(self):
        """Merge all samples into final signatures."""
        if self.color_samples:
            # Average all color samples
            stacked = np.stack(self.color_samples, axis=0)
            self.merged_color_hist = np.mean(stacked, axis=0).astype(np.float32)
            cv2.normalize(self.merged_color_hist, self.merged_color_hist, 0, 255, cv2.NORM_MINMAX)
            
        if self.texture_samples:
            # Average all texture samples
            stacked = np.stack(self.texture_samples, axis=0)
            self.merged_texture_hist = np.mean(stacked, axis=0).astype(np.float32)
            # Already normalized during computation
    
    @property
    def sample_count(self) -> int:
        return len(self.color_samples)


class SignatureBallDetector(BallDetector):
    """
    Robust ball detector using color and texture signatures.
    
    Improvements over basic histogram matching:
    - CLAHE normalization for lighting invariance
    - Hue-only histogram (more lighting stable than H+S)
    - Multi-sample registration for different distances/lighting
    - LBP texture features for scale invariance
    """
    
    def __init__(self, confidence_threshold: float = 0.6, scale: float = 1.0, 
                 h_bins: int = 36, sigma: float = 1.0,
                 color_weight: float = 0.6, texture_weight: float = 0.4,
                 min_area: int = 400, min_circularity: float = 0.65,
                 require_edges: bool = True, min_edge_ratio: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self.min_area = min_area
        self.min_circularity = min_circularity
        self.require_edges = require_edges  # Require visible circular edge
        self.min_edge_ratio = min_edge_ratio  # Min ratio of edge pixels around perimeter
        self.scale = scale
        self.h_bins = h_bins  # Hue bins (0-180 degrees)
        self.sigma = sigma
        self.color_weight = color_weight
        self.texture_weight = texture_weight
        
        # Signatures storage
        self.signatures: Dict[str, ObjectSignature] = {}
        self.class_id_map: Dict[int, str] = {}
        self.next_class_id = 100
        
        # Background reference
        self.background_hist: Optional[np.ndarray] = None
        
        # CLAHE for lighting normalization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # LBP parameters
        self.lbp_radius = 2
        self.lbp_n_points = 8 * self.lbp_radius
        
        self.lock = threading.Lock()
    
    def _has_circular_edge(self, gray_img: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        """
        Check if candidate region has visible circular edges.
        Returns True if edges form a roughly circular boundary.
        This rejects smooth blobs (foreheads) that lack defined edges.
        """
        # Extract ROI with padding
        pad = max(5, int(max(w, h) * 0.2))
        y1 = max(0, y - pad)
        y2 = min(gray_img.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(gray_img.shape[1], x + w + pad)
        
        roi = gray_img[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        
        # Apply Canny edge detection
        # Use adaptive thresholds based on ROI statistics
        median_val = np.median(roi)
        lower = int(max(0, 0.5 * median_val))
        upper = int(min(255, 1.5 * median_val))
        edges = cv2.Canny(roi, lower, upper)
        
        # Create circular mask for the expected ball location
        center_x = (x - x1) + w // 2
        center_y = (y - y1) + h // 2
        radius = max(w, h) // 2
        
        # Create annular mask (ring around expected ball perimeter)
        outer_mask = np.zeros_like(edges)
        inner_mask = np.zeros_like(edges)
        cv2.circle(outer_mask, (center_x, center_y), int(radius * 1.2), 255, -1)
        cv2.circle(inner_mask, (center_x, center_y), int(radius * 0.8), 255, -1)
        ring_mask = cv2.bitwise_and(outer_mask, cv2.bitwise_not(inner_mask))
        
        # Count edge pixels in the ring
        edge_in_ring = cv2.bitwise_and(edges, ring_mask)
        edge_count = np.count_nonzero(edge_in_ring)
        
        # Expected perimeter of the circle
        expected_perimeter = 2 * np.pi * radius
        
        # Check if enough edges are present
        edge_ratio = edge_count / (expected_perimeter + 1e-6)
        
        return edge_ratio >= self.min_edge_ratio

    @property
    def class_names(self) -> Dict[int, str]:
        with self.lock:
            return self.class_id_map.copy()

    def get_signatures(self) -> Dict[str, np.ndarray]:
        """Returns merged color histograms for compatibility."""
        with self.lock:
            return {label: sig.merged_color_hist for label, sig in self.signatures.items() 
                    if sig.merged_color_hist is not None}
    
    def _normalize_lighting(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Apply CLAHE to V channel for lighting normalization."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = self.clahe.apply(hsv[:, :, 2])
        return hsv
    
    def _compute_hue_hist(self, img_hsv: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute Hue-only histogram (lighting invariant)."""
        # Filter out low-saturation pixels (they have unreliable hue)
        if mask is not None:
            sat_mask = (img_hsv[:, :, 1] > 30).astype(np.uint8) * 255
            combined_mask = cv2.bitwise_and(mask, sat_mask)
        else:
            combined_mask = (img_hsv[:, :, 1] > 30).astype(np.uint8) * 255
        
        hist = cv2.calcHist([img_hsv], [0], combined_mask, [self.h_bins], [0, 180])
        
        # Smooth histogram
        if self.sigma > 0:
            ksize = int(2 * round(3 * self.sigma) + 1)
            if ksize > 1:
                hist = cv2.GaussianBlur(hist, (ksize, 1), self.sigma)
        
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        return hist.flatten()
    
    def _compute_lbp_hist(self, gray_img: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute Local Binary Pattern histogram for texture fingerprinting."""
        # Simple LBP implementation without skimage dependency
        # For each pixel, compare with neighbors in a circle
        
        h, w = gray_img.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        
        # Use a simple 3x3 LBP for speed
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = gray_img[i, j]
                code = 0
                # 8 neighbors clockwise from top-left
                neighbors = [
                    gray_img[i-1, j-1], gray_img[i-1, j], gray_img[i-1, j+1],
                    gray_img[i, j+1], gray_img[i+1, j+1], gray_img[i+1, j],
                    gray_img[i+1, j-1], gray_img[i, j-1]
                ]
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                lbp[i, j] = code
        
        # Compute histogram
        if mask is not None:
            pixels = lbp[mask > 0]
        else:
            pixels = lbp.ravel()
        
        hist, _ = np.histogram(pixels, bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-6)
        return hist

    def register_background(self, frame: np.ndarray):
        """Registers the global background color distribution."""
        hsv = self._normalize_lighting(frame)
        hist = self._compute_hue_hist(hsv)
        with self.lock:
            self.background_hist = hist

    def register_object(self, frame: np.ndarray, bbox: Tuple[float, float, float, float], label: str) -> int:
        """
        Registers an object signature from an ROI.
        Can be called multiple times for the same label to build a robust signature.
        bbox: xyxy (pixels)
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        
        # Clip
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return -1
            
        roi = frame[y1:y2, x1:x2]
        
        # Normalize lighting
        hsv_roi = self._normalize_lighting(roi)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Wider elliptical mask (60% radius instead of 30%)
        center = (roi.shape[1] // 2, roi.shape[0] // 2)
        axes = (int(roi.shape[1] * 0.6), int(roi.shape[0] * 0.6))
        mask_roi = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
        cv2.ellipse(mask_roi, center, axes, 0, 0, 360, 255, -1)
        
        # Compute histograms
        color_hist = self._compute_hue_hist(hsv_roi, mask_roi)
        texture_hist = self._compute_lbp_hist(gray_roi, mask_roi)
        
        with self.lock:
            if label not in self.signatures:
                cid = self.next_class_id
                self.signatures[label] = ObjectSignature(label, cid)
                self.class_id_map[cid] = label
                self.next_class_id += 1
            else:
                cid = self.signatures[label].class_id
            
            self.signatures[label].add_sample(color_hist, texture_hist)
            sample_count = self.signatures[label].sample_count
            
        return cid

    def detect(self, frame: np.ndarray, allowed_classes: List[int] = None, mask: np.ndarray = None) -> List[Detection]:
        with self.lock:
            if not self.signatures:
                return []
            signatures_snapshot = {k: v for k, v in self.signatures.items()}
            class_map_snapshot = self.class_id_map.copy()
            bg_hist_snapshot = self.background_hist.copy() if self.background_hist is not None else None

        # Scaling
        if self.scale != 1.0:
            processing_frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
            if mask is not None:
                processing_mask = cv2.resize(mask, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
            else:
                processing_mask = None
        else:
            processing_frame = frame
            processing_mask = mask

        # Normalize lighting
        hsv_frame = self._normalize_lighting(processing_frame)
        detections = []
        
        for label, sig in signatures_snapshot.items():
            if sig.merged_color_hist is None:
                continue
                
            cid = sig.class_id
            
            if allowed_classes is not None and cid not in allowed_classes:
                continue
            
            # Use Hue-only histogram for backprojection
            target_hist = sig.merged_color_hist.reshape(-1, 1)
            
            # Apply background subtraction if available
            if bg_hist_snapshot is not None:
                bg = bg_hist_snapshot.reshape(-1, 1)
                ratio_hist = target_hist / (target_hist + bg + 1e-5)
                cv2.normalize(ratio_hist, ratio_hist, 0, 255, cv2.NORM_MINMAX)
                target_hist = ratio_hist.astype(np.float32)

            # Backprojection (Hue only)
            prob_map = cv2.calcBackProject([hsv_frame], [0], target_hist, [0, 180], 1)
            
            # Apply motion mask if provided
            if processing_mask is not None:
                prob_map = cv2.bitwise_and(prob_map, prob_map, mask=processing_mask)
            
            # Cleaning
            ksize = 7 if self.scale >= 0.5 else 5
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            cv2.filter2D(prob_map, -1, kernel, prob_map)
            
            # More permissive thresholding
            thresh_val = int(self.confidence_threshold * 255)
            _, mask_thresh = cv2.threshold(prob_map, thresh_val, 255, cv2.THRESH_BINARY)
            
            # Morphological cleanup
            mask_thresh = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            mask_thresh = cv2.morphologyEx(mask_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(mask_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                
                # Stricter area threshold to reject noise
                scaled_min_area = self.min_area * self.scale * self.scale
                if area < scaled_min_area:
                    continue
                    
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Balls are round - tighter aspect ratio
                if not (0.7 < aspect_ratio < 1.4):
                    continue
                
                # Circularity check - balls should be circular
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Stricter circularity for ball detection
                if circularity < self.min_circularity:
                    continue
                
                # Edge verification - reject smooth blobs without circular edges
                if self.require_edges:
                    gray_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)
                    if not self._has_circular_edge(gray_frame, x, y, w, h):
                        continue  # Reject - no visible circular edge (likely forehead/smooth surface)

                # Compute confidence based on prob_map intensity in the region
                region_mask = np.zeros_like(prob_map)
                cv2.drawContours(region_mask, [cnt], -1, 255, -1)
                mean_prob = cv2.mean(prob_map, mask=region_mask)[0] / 255.0
                
                # Scale bbox back up
                if self.scale != 1.0:
                    inv_scale = 1.0 / self.scale
                    bbox = (
                        float(x * inv_scale), 
                        float(y * inv_scale), 
                        float((x + w) * inv_scale), 
                        float((y + h) * inv_scale)
                    )
                else:
                    bbox = (float(x), float(y), float(x+w), float(y+h))

                detections.append(Detection(
                    bbox=bbox,
                    confidence=mean_prob,
                    class_id=cid
                ))
                    
        return detections

    def detect_from_rois(self, frame: np.ndarray, rois: List[Tuple[float, float, float, float]]) -> List[Detection]:
        """
        Run detection on crops defined by ROIs.
        rois: List of (x_center_norm, y_center_norm, w_norm, h_norm)
        """
        all_detections = []
        h, w = frame.shape[:2]

        for (xc, yc, rw, rh) in rois:
            cx_px = int(xc * w)
            cy_px = int(yc * h)
            w_px = int(rw * w)
            h_px = int(rh * h)
            
            x1 = max(0, cx_px - w_px // 2)
            y1 = max(0, cy_px - h_px // 2)
            x2 = min(w, x1 + w_px)
            y2 = min(h, y1 + h_px)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop = frame[y1:y2, x1:x2]
            crop_detections = self.detect(crop)
            
            for det in crop_detections:
                lx1, ly1, lx2, ly2 = det.bbox
                det.bbox = (lx1 + x1, ly1 + y1, lx2 + x1, ly2 + y1)
                all_detections.append(det)
                
        return all_detections
    
    def get_registration_info(self, label: str) -> Optional[Dict]:
        """Get info about a registered signature."""
        with self.lock:
            if label in self.signatures:
                sig = self.signatures[label]
                return {
                    'label': label,
                    'class_id': sig.class_id,
                    'sample_count': sig.sample_count,
                    'has_color': sig.merged_color_hist is not None,
                    'has_texture': sig.merged_texture_hist is not None
                }
        return None
