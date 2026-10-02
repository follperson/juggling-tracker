"""Portable model selection shared by inference commands and the detector."""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MODEL = Path("models/juggletrack-v3/best.pt")


def resolve_model(model: str | None = None) -> str:
    """Prefer explicit weights, then the environment, then the local champion.

    Explicit names such as ``yolo11n.pt`` are passed to Ultralytics, which
    can download them. An absent default never silently selects stock weights.
    """
    selected = model or os.environ.get("JUGGLETRACK_MODEL")
    if selected:
        return os.path.expanduser(selected)
    if DEFAULT_MODEL.is_file():
        return str(DEFAULT_MODEL)
    raise FileNotFoundError(
        f"No model selected and default weights not found: looked for "
        f"{DEFAULT_MODEL.resolve()} ({DEFAULT_MODEL} relative to the current "
        f"directory {Path.cwd()}). Pass --model /path/to/best.pt, set "
        "JUGGLETRACK_MODEL, or run from the repository root. "
        "For a stock-detector experiment, explicitly pass --model yolo11n.pt."
    )
