from typing import List, Tuple
from enum import Enum
from pydantic import BaseModel, Field, validator, conint, confloat


class EventType(str, Enum):
    THROW = "throw"
    PEAK = "peak"

class TrackPoint(BaseModel):
    frame_idx: conint(ge=0) = Field(..., description="Frame index, non-negative")
    timestamp: confloat(ge=0) = Field(..., description="Timestamp in seconds, non-negative")
    pos: Tuple[confloat(ge=0, le=1), confloat(ge=0, le=1)] = Field(..., description="Normalized [x, y] coordinates in range [0, 1]")
    confidence: confloat(ge=0, le=1) = Field(..., description="Detection confidence between 0 and 1")


class Track(BaseModel):
    id: conint(ge=0) = Field(..., description="Track identifier, non-negative")
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
    start_time: confloat(ge=0) = Field(...)
    end_time: confloat(ge=0) = Field(...)
    confidence: confloat(ge=0, le=1) = Field(...)

    @validator("end_time")
    def end_after_start(cls, v, values):
        if "start_time" in values and v < values["start_time"]:
            raise ValueError("end_time must be >= start_time")
        return v

class ThrowEvent(BaseModel):
    timestamp: confloat(ge=0) = Field(...)
    track_id: conint(ge=0) = Field(...)
    confidence: confloat(ge=0, le=1) = Field(...)
    event_type: EventType = EventType.THROW

class Run(BaseModel):
    start_time: confloat(ge=0) = Field(...)
    end_time: confloat(ge=0) = Field(...)
    throw_count: conint(ge=0) = Field(...)

    @validator("end_time")
    def end_after_start(cls, v, values):
        if "start_time" in values and v < values["start_time"]:
            raise ValueError("end_time must be >= start_time")
        return v

class VideoMetadata(BaseModel):
    fps: confloat(gt=0) = Field(...)
    width: conint(gt=0) = Field(...)
    height: conint(gt=0) = Field(...)
    frame_count: conint(gt=0) = Field(...)
    duration: confloat(gt=0) = Field(...)

class SegmentAnalysis(BaseModel):
    segment: Segment
    throws: List[ThrowEvent]
    runs: List[Run]
    total_throws: conint(ge=0) = Field(...)
    total_peaks: conint(ge=0) = Field(...)

class FullAnalysis(BaseModel):
    video_metadata: VideoMetadata
    segments: List[SegmentAnalysis]
    tracks: List[Track] = Field(default_factory=list)
