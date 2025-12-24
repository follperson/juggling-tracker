from typing import List, Tuple
import numpy as np
from .schema import Track, Segment, ThrowEvent

def extract_throw_events(tracks: List[Track], segment: Segment, hand_zone_threshold: float = 0.6) -> List[ThrowEvent]:
    """
    Detects throw events by finding local maxima in y (bottom of the arc).
    Image coordinates: y increases downwards.
    """
    events = []
    
    for track in tracks:
        # Filter points within the segment
        segment_points = [p for p in track.points if segment.start_time <= p.timestamp <= segment.end_time]
        if len(segment_points) < 10:
            continue
            
        timestamps = [p.timestamp for p in segment_points]
        y_coords = [(p.bbox[1] + p.bbox[3]) / 2 for p in segment_points]
        
        # Smooth y coordinates
        y_smooth = np.convolve(y_coords, np.ones(5)/5, mode='same')
        
        # Find local maxima in y (lowest point in image, highest value)
        for i in range(2, len(y_smooth) - 2):
            if y_smooth[i] > y_smooth[i-1] and y_smooth[i] > y_smooth[i+1]:
                # Check if it's in the hand zone (bottom part of the frame)
                # We need frame height to know the absolute scale, or use relative
                # For now let's assume y is normalized 0-1, but OpenCV usually gives pixels.
                # Let's just use the local maximum as a proxy for throw/catch event.
                
                # Debounce: avoid multiple events too close
                if not events or (timestamps[i] - events[-1].timestamp > 0.2):
                    events.append(ThrowEvent(
                        timestamp=timestamps[i],
                        track_id=track.id,
                        confidence=0.9
                    ))
                    
    # Sort events by time
    events.sort(key=lambda x: x.timestamp)
    return events
