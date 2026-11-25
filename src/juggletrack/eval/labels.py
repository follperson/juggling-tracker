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


class CatchBracket(BaseModel):
    """Inclusive uncertainty bounds for one observed receiving contact."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    earliest_t: float = Field(ge=0)
    latest_t: float = Field(ge=0)

    @model_validator(mode="after")
    def ordered_bounds(self) -> "CatchBracket":
        if self.latest_t < self.earliest_t:
            raise ValueError("catch latest_t must not precede earliest_t")
        return self


class ReviewedCatchWindow(BaseModel):
    """Exhaustively reviewed [start_t, end_t) coverage, including negatives."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    start_t: float = Field(ge=0)
    end_t: float = Field(gt=0)
    catches: list[CatchBracket]

    @model_validator(mode="after")
    def contained_contacts(self) -> "ReviewedCatchWindow":
        if self.end_t <= self.start_t:
            raise ValueError("reviewed catch window end_t must exceed start_t")
        if any(c.earliest_t < self.start_t or c.latest_t >= self.end_t
               for c in self.catches):
            raise ValueError("catch bracket must lie inside its reviewed [start_t, end_t) window")
        return self


def validate_catch_coverage(windows: list[ReviewedCatchWindow]) -> None:
    ordered = sorted(windows, key=lambda window: window.start_t)
    if any(a.end_t > b.start_t for a, b in zip(ordered, ordered[1:])):
        raise ValueError("reviewed catch windows must not overlap")


class VideoLabels(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    video: str
    runs: list[LabeledRun] = Field(default_factory=list)
    drops: list[float] = Field(default_factory=list)
    catch_windows: list[ReviewedCatchWindow] | None = None

    @model_validator(mode="after")
    def disjoint_catch_coverage(self) -> "VideoLabels":
        validate_catch_coverage(self.catch_windows or [])
        return self
