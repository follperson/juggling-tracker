from typing import List, Tuple
import numpy as np
from scipy.signal import find_peaks
from .schema import Track, Segment, ThrowEvent, EventType

def extract_throw_events(tracks: List[Track], segment: Segment) -> List[ThrowEvent]:
    """
    Detects throw events (local maxima in y) and peaks (local minima in y).
    Uses scipy.signal.find_peaks for robustness against slow transitions and plateaus.
    """
    events = []
    
    for track in tracks:
        # Filter points within the segment
        segment_points = [p for p in track.points if segment.start_time <= p.timestamp <= segment.end_time]
        if len(segment_points) < 5:
            continue
            
        timestamps = np.array([p.timestamp for p in segment_points])
        y_coords = np.array([p.pos[1] for p in segment_points])
        
        # 1. Detect THROWS (local maxima in y - bottom of arc)
        # Prominence ensures we don't catch tiny jitters
        throw_indices, _ = find_peaks(y_coords, prominence=0.02)
        for idx in throw_indices:
            events.append(ThrowEvent(
                timestamp=round(float(timestamps[idx]), 4),
                track_id=track.id,
                confidence=0.9,
                event_type=EventType.THROW
            ))
            
        # 2. Detect PEAKS (local minima in y - top of arc)
        # We invert y to find minima as peaks
        peak_indices, _ = find_peaks(-y_coords, prominence=0.01)
        for idx in peak_indices:
            events.append(ThrowEvent(
                timestamp=round(float(timestamps[idx]), 4),
                track_id=track.id,
                confidence=0.9,
                event_type=EventType.PEAK
            ))

        # 3. Handle off-frame peaks
        # Check if the track begins or ends near the top edge
        # Check first point
        if segment_points[0].pos[1] < 0.15: # Slightly more generous threshold
            # If no peak was found very close to the start
            if not any(e.event_type == EventType.PEAK and abs(e.timestamp - timestamps[0]) < 0.4 for e in events if e.track_id == track.id):
                events.append(ThrowEvent(
                    timestamp=round(float(timestamps[0]), 4),
                    track_id=track.id,
                    confidence=0.7,
                    event_type=EventType.PEAK
                ))

        # Check last point
        if segment_points[-1].pos[1] < 0.15:
            # If no peak was found very close to the end
            if not any(e.event_type == EventType.PEAK and abs(e.timestamp - timestamps[-1]) < 0.4 for e in events if e.track_id == track.id):
                events.append(ThrowEvent(
                    timestamp=round(float(timestamps[-1]), 4),
                    track_id=track.id,
                    confidence=0.7,
                    event_type=EventType.PEAK
                ))
                    
    # Sort events by time
    events.sort(key=lambda x: x.timestamp)
    return events
