from typing import List, Tuple
from pydantic import BaseModel

class TrackPoint(BaseModel):
    frame_idx: int
    timestamp: float
    bbox: Tuple[float, float, float, float]
    confidence: float

class Track(BaseModel):
    id: int
    points: List[TrackPoint]

    @property
    def last_point(self) -> TrackPoint:
        return self.points[-1]

class FrameFeatures(BaseModel):
    frame_idx: int
    timestamp: float
    num_detections: int
    num_active_tracks: int
    mean_velocity_y: float
    std_velocity_y: float

class Segment(BaseModel):
    start_time: float
    end_time: float
    confidence: float

class ThrowEvent(BaseModel):
    timestamp: float
    track_id: int
    confidence: float

class Run(BaseModel):
    start_time: float
    end_time: float
    throw_count: int

class VideoMetadata(BaseModel):
    fps: float
    width: int
    height: int
    frame_count: int
    duration: float

class SegmentAnalysis(BaseModel):
    segment: Segment
    throws: List[ThrowEvent]
    runs: List[Run]
    total_throws: int

class FullAnalysis(BaseModel):
    video_metadata: VideoMetadata
    segments: List[SegmentAnalysis]
    tracks: List[Track] = []
