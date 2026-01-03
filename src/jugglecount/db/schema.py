from __future__ import annotations
from typing import List, Tuple, Optional
from enum import Enum
from pydantic import validator, conint, confloat
from sqlmodel import SQLModel, Field, Relationship, Session, create_engine
from sqlalchemy.orm import relationship


class EventType(str, Enum):
    THROW = "throw"
    PEAK = "peak"

class VideoMetadata(SQLModel):
    fps: float
    width: int
    height: int
    frame_count: int
    duration: float

class TrackPoint(SQLModel):
    frame_idx: int
    timestamp: float
    pos_x: float
    pos_y: float
    confidence: float
    is_interpolated: bool = False  # True for synthetic/interpolated points

class Track(SQLModel):
    id: int = Field(primary_key=True)
    points: List[TrackPoint] = Field(default_factory=list)

    @property
    def last_point(self) -> TrackPoint:
        return self.points[-1]

class FrameFeatures(SQLModel):
    frame_idx: int
    timestamp: float
    num_detections: int
    num_active_tracks: int
    mean_velocity_y: float
    std_velocity_y: float

class Segment(SQLModel):
    start_time: float
    end_time: float
    confidence: float

# --- Database Tables ---

class User(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    user_id: str = Field(primary_key=True)
    name: str
    email: str
    
    # Relationships
    sessions: List["JugglingSession"] = Relationship(
        back_populates="user",
        sa_relationship=relationship("jugglecount.db.schema.JugglingSession", back_populates="user")
    )

class JugglingSession(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    session_uuid: str = Field(index=True, unique=True)
    user_id: str = Field(foreign_key="user.user_id")
    start_time: float
    end_time: float
    
    # Relationships
    user: User = Relationship(back_populates="sessions")
    throws: List["ThrowEvent"] = Relationship(
        back_populates="session",
        sa_relationship=relationship("jugglecount.db.schema.ThrowEvent", back_populates="session")
    )
    runs: List["Run"] = Relationship(
        back_populates="session", 
        sa_relationship=relationship("jugglecount.db.schema.Run", back_populates="session")
    )

class ThrowEvent(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    session_pk: Optional[int] = Field(default=None, foreign_key="jugglingsession.id")
    timestamp: float
    track_id: int
    confidence: float
    event_type: EventType = EventType.THROW
    
    session: JugglingSession = Relationship(back_populates="throws")

class Run(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    session_pk: Optional[int] = Field(default=None, foreign_key="jugglingsession.id")
    start_time: float
    end_time: float
    throw_count: int
    
    session: JugglingSession = Relationship(back_populates="runs")

# --- Analysis Models ---

class SegmentAnalysis(SQLModel):
    segment: Segment
    throws: List[ThrowEvent]
    runs: List[Run]
    total_throws: int
    total_peaks: int

class FullAnalysis(SQLModel):
    video_metadata: VideoMetadata
    segments: List[SegmentAnalysis]
    tracks: List[Track]
    runs: List[Run]

# Resolve forward references
User.update_forward_refs()
JugglingSession.update_forward_refs()
ThrowEvent.update_forward_refs()
Run.update_forward_refs()
