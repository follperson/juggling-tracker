"""Hand-written ground-truth labels for a video (one JSON document per video)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LabeledRun(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    start_t: float
    end_t: float
    catches: int = Field(ge=0)
    end_reason: str = "stop"

    @model_validator(mode="after")
    def ordered_span(self) -> "LabeledRun":
        if self.end_t <= self.start_t:
            raise ValueError("labeled run end_t must exceed start_t")
        return self


class VideoLabels(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    video: str
    runs: list[LabeledRun] = Field(default_factory=list)
    drops: list[float] = Field(default_factory=list)
