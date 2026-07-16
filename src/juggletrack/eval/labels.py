"""Hand-written ground-truth labels for a video (one JSON document per video)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class LabeledRun(BaseModel):
    start_t: float
    end_t: float
    catches: int
    end_reason: str = "stop"


class VideoLabels(BaseModel):
    video: str
    runs: list[LabeledRun] = Field(default_factory=list)
    drops: list[float] = Field(default_factory=list)
