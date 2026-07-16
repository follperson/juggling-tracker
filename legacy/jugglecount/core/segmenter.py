from typing import List
from jugglecount.db.schema import Segment, FrameFeatures

def segment_video(features: List[FrameFeatures], min_on_duration: float = 1.0, min_off_duration: float = 0.5) -> List[Segment]:
    """
    Very basic segmenter that looks for frames with >= 2 active tracks.
    Uses hysteresis to avoid flickering.
    """
    segments = []
    is_juggling = False
    segment_start = 0.0
    
    # Simple smoothing/hysteresis
    # 1. Identify "candidate" juggling frames
    candidate_frames = [f.num_active_tracks >= 2 for f in features]
    
    # 2. Find contiguous blocks
    i = 0
    while i < len(candidate_frames):
        if candidate_frames[i]:
            start_idx = i
            # Look for the end of the segment, allowing for small gaps
            gap_count = 0
            max_gap_frames = int(min_off_duration * (len(features) / features[-1].timestamp)) if features[-1].timestamp > 0 else 5
            
            while i < len(candidate_frames):
                if candidate_frames[i]:
                    gap_count = 0
                else:
                    gap_count += 1
                    if gap_count > max_gap_frames:
                        break
                i += 1
            
            end_idx = i - gap_count
            duration = features[end_idx-1].timestamp - features[start_idx].timestamp
            
            if duration >= min_on_duration:
                segments.append(Segment(
                    start_time=round(features[start_idx].timestamp, 4),
                    end_time=round(features[end_idx-1].timestamp, 4),
                    confidence=0.8 # Placeholder
                ))
        else:
            i += 1
            
    return segments
