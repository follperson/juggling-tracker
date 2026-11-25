"""Data contracts for juggletrack. Pure pydantic models — no cv2/torch imports.

Conventions: normalized [0,1] coordinates, y increases DOWNWARD, times in seconds.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, model_serializer

from juggletrack import SCHEMA_VERSION


class Detection(BaseModel):
    """One ball detection in one frame (source-agnostic: detector, simulator, or label)."""

    frame_idx: int
    t: float
    x: float
    y: float
    w: float = 0.0
    h: float = 0.0
    confidence: float = 1.0


class Arc(BaseModel):
    """A ballistic flight segment.

    y(t) = ay*dt^2 + by*dt + cy,  x(t) = bx*dt + cx,  where dt = t - t_start.
    ay ≈ g/2 > 0 in down-positive normalized units.
    """

    id: int
    t_start: float
    t_end: float
    ay: float
    by: float
    cy: float
    bx: float
    cx: float
    n_points: int
    rmse: float
    # Pre-separation fitted endpoint heights retain observed hand depth.
    # Present only when a descending hand carry was separated from flight.
    carry_fit_endpoints_y: tuple[float, float] | None = None

    @model_serializer(mode="wrap")
    def _serialize_arc(self, handler: SerializerFunctionWrapHandler) -> dict:
        """Keep the existing wire representation for an ordinary flight."""
        data = handler(self)
        if self.carry_fit_endpoints_y is None:
            data.pop("carry_fit_endpoints_y", None)
        return data

    def peer_hand_heights(self) -> tuple[float, float]:
        return self.carry_fit_endpoints_y or (self.cy, self.y_at(self.t_end))

    def y_at(self, t: float) -> float:
        dt = t - self.t_start
        return self.ay * dt * dt + self.by * dt + self.cy

    def x_at(self, t: float) -> float:
        return self.bx * (t - self.t_start) + self.cx

    def vy_at(self, t: float) -> float:
        return 2.0 * self.ay * (t - self.t_start) + self.by

    def apex_t(self) -> float:
        """Time of the arc's highest point (vertex of the parabola)."""
        if self.ay == 0.0:
            return self.t_start
        return self.t_start - self.by / (2.0 * self.ay)

    def apex_y(self) -> float:
        return self.y_at(self.apex_t())

    def duration(self) -> float:
        return self.t_end - self.t_start


class ThrowEvent(BaseModel):
    t: float
    x: float
    arc_id: int


class CatchEvent(BaseModel):
    t: float
    x: float
    arc_id: int


class DropEvent(BaseModel):
    t: float
    x: float
    arc_id: int | None
    signals: list[str] = Field(default_factory=list)


class Run(BaseModel):
    start_t: float
    end_t: float
    catches: int
    throws: int
    arc_ids: list[int] = Field(default_factory=list)
    end_reason: Literal["drop", "stop", "video_end"] = "stop"
    period_s: float | None = None
    # Airborne-count periodicity (events.periodicity.periodicity_score) over
    # this run's own arc span, judged against a lag band derived from
    # period_s. 0.0 is overloaded: it means EITHER "judged, and no
    # periodicity found" OR "unjudgeable" (the arc span was too short to
    # search any lag in the band) -- periodicity_score distinguishes those
    # (returns None for the latter) but this field can't, so
    # events.runs.segment_runs collapses "unjudgeable" to 0.0.
    quality: float = 0.0


class SessionResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    runs: list[Run] = Field(default_factory=list)
    drops: list[DropEvent] = Field(default_factory=list)
    arcs: list[Arc] = Field(default_factory=list)
    hand_line_y: float = 0.0
    meta: dict[str, float | int | str] = Field(default_factory=dict)
