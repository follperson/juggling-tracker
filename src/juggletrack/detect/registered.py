"""Conservative offline color proposals; these are not counted flight events.

Enrollment assumes a single chromatic prop, entirely visible in a circular
selection. Keep framing and illumination similar when reusing a profile.
OpenCV is loaded only when enrollment or proposal generation is requested.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable, Iterator, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from juggletrack.types import Detection


class BallProfile(BaseModel):
    """Portable registration, with radius relative to the shorter image side."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True, frozen=True)
    schema_version: Literal[1] = 1
    hue: float = Field(ge=0, lt=180)
    hue_halfwidth: float = Field(default=12.0, gt=0, le=20)
    saturation_min: float = Field(ge=35, le=255)
    value_min: float = Field(ge=25, le=255)
    radius_fraction: float = Field(gt=0, lt=0.5)

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value):
        if type(value) is not int:
            raise ValueError("schema_version must be integer 1")
        return value


def _frame(frame: np.ndarray) -> None:
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim != 3 or frame.shape[2] != 3
            or min(frame.shape[:2]) < 8):
        raise ValueError("frame must be a nonempty uint8 BGR image, at least 8 by 8")


def _hue_distance(a, b):
    distance = np.abs(a - b)
    return np.minimum(distance, 180 - distance)


def register_ball(frame: np.ndarray, x: int, y: int, radius: int) -> BallProfile:
    """Register a FULL ball radius, sampling its inner half to avoid background.

    Reject achromatic, dark, mixed-color and poorly isolated selections rather
    than silently constructing an unrestricted detector. No image data is saved.
    """
    _frame(frame)
    h, w = frame.shape[:2]
    if (any(type(v) is not int for v in (x, y, radius)) or radius < 4
            or x - radius < 0 or y - radius < 0
            or x + radius >= w or y + radius >= h):
        raise ValueError("selection must be an integer full-ball circle inside the image, radius >=4")
    import cv2

    patch = frame[y-radius:y+radius+1, x-radius:x+radius+1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
    pixels = hsv[xx*xx + yy*yy <= (radius/2)**2].astype(float)
    if np.percentile(pixels[:, 1], 20) < 60 or np.percentile(pixels[:, 2], 20) < 40:
        raise ValueError("unsupported achromatic, dark or mixed/background enrollment")
    # Circular mean handles a red prop spanning the OpenCV hue wrap at 0/179.
    angle = pixels[:, 0] * (2*math.pi/180)
    hue = float(np.angle(np.mean(np.exp(1j*angle))) * 180/(2*math.pi) % 180)
    if np.mean(_hue_distance(pixels[:, 0], hue) <= 12) < .85:
        raise ValueError("ambiguous or mixed-color enrollment: select one uniform chromatic surface")
    return BallProfile(
        hue=hue, saturation_min=max(35., float(np.percentile(pixels[:, 1], 10))*.55),
        value_min=max(25., float(np.percentile(pixels[:, 2], 10))*.5),
        radius_fraction=float(radius/min(h, w)),
    )


@dataclass
class _Candidate:
    detection: Detection
    radius: float
    moving: bool
    profile_id: int
    confirmed: bool = False


def _components(frame, previous, profiles, index, t):
    import cv2

    height, width = frame.shape[:2]
    scale = min(1., 960/max(height, width))
    small = cv2.resize(frame, None, fx=scale, fy=scale)
    h, w = small.shape[:2]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    diff = np.zeros_like(gray) if previous is None else cv2.absdiff(gray, previous)
    candidates = []
    kernel = np.ones((3, 3), np.uint8)
    for profile_id, profile in enumerate(profiles):
        radius = profile.radius_fraction * min(h, w)
        mask = ((_hue_distance(hsv[:, :, 0].astype(float), profile.hue)
                 <= profile.hue_halfwidth) & (hsv[:, :, 1] >= profile.saturation_min)
                & (hsv[:, :, 2] >= profile.value_min)).astype(np.uint8)*255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        _, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
        for label, (x, y, bw, bh, area) in enumerate(stats[1:], 1):
            if (not .25*math.pi*radius**2 <= area <= 4*math.pi*radius**2
                    or min(bw, bh) < 3 or max(bw, bh) > 4*radius
                    or max(bw/bh, bh/bw) > 3.2 or area/(bw*bh) < .25):
                continue
            x0, x1 = max(0, x-bw//2), min(w, x+bw+bw//2)
            y0, y1 = max(0, y-bh//2), min(h, y+bh+bh//2)
            ring_count = np.count_nonzero(mask[y0:y1, x0:x1]) - np.count_nonzero(
                mask[y:y+bh, x:x+bw])
            ring_fraction = ring_count/max(1, (x1-x0)*(y1-y0)-bw*bh)
            if ring_fraction > .5 or area/(bw*bh) < 1.6*max(.01, ring_fraction):
                continue
            cx, cy = centers[label]
            detection = Detection(frame_idx=index, t=t, x=float(cx/w), y=float(cy/h),
                                  w=float(bw/w), h=float(bh/h), confidence=1.)
            region = np.s_[y:y+bh, x:x+bw]
            moving = bool(np.mean(diff[region][labels[region] == label] > 18) >= .06)
            candidates.append(_Candidate(detection, radius/scale, moving, profile_id))
    return candidates, gray


def _duplicate(a, b):
    dx, dy = abs(a.x-b.x), abs(a.y-b.y)
    overlap = max(0., min(a.x+a.w/2, b.x+b.w/2)-max(a.x-a.w/2, b.x-b.w/2)) * max(
        0., min(a.y+a.h/2, b.y+b.h/2)-max(a.y-a.h/2, b.y-b.h/2))
    union = a.w*a.h+b.w*b.h-overlap
    return (union > 0 and overlap/union >= .7
            and dx <= .2*min(a.w, b.w) and dy <= .2*min(a.h, b.h))


def _confirm(window, width, height):
    def point(candidate):
        d = candidate.detection
        return np.array([d.x*width, d.y*height])

    def neighbors(candidate, choices):
        return [c for c in choices
                if candidate.profile_id == c.profile_id and
                np.linalg.norm(point(c)-point(candidate)) <= 4*min(c.radius, candidate.radius)]

    before, middle, after = window
    for center in middle:
        left, right = neighbors(center, before), neighbors(center, after)
        if len(left) != 1 or len(right) != 1:
            continue
        a, c = left[0], right[0]
        if len(neighbors(a, middle)) != 1 or len(neighbors(c, middle)) != 1:
            continue
        da, db, dc = a.detection, center.detection, c.detection
        if not (db.frame_idx == da.frame_idx+1 and dc.frame_idx == db.frame_idx+1):
            continue
        # Use actual PTS, including irregular intervals; never infer elapsed time
        # from the frame rate or integer index.
        alpha = (db.t-da.t)/(dc.t-da.t)
        residual = np.linalg.norm(point(center)-((1-alpha)*point(a)+alpha*point(c)))
        radius = min(a.radius, center.radius, c.radius)
        if (residual <= (8/15)*radius
                and np.linalg.norm(point(c)-point(a)) >= .4*radius
                and sum(v.moving for v in (a, center, c)) >= 2):
            a.confirmed = center.confirmed = c.confirmed = True


def _confirmed_detections(candidates):
    # Each profile first establishes motion at its own size. Deduplicate only
    # confirmed regions so an overlapping profile cannot veto another's witness.
    detections = sorted((c.detection for c in candidates if c.confirmed),
                        key=lambda d: (d.x, d.y, d.w, d.h))
    emitted = []
    for detection in detections:
        if not any(_duplicate(detection, old) for old in emitted):
            emitted.append(detection)
            yield detection


def registered_proposals(
    frames: Iterable[tuple[int, float, np.ndarray]], profiles: list[BallProfile],
) -> Iterator[Detection]:
    """Yield chronological, deduplicated offline proposals with two-frame lookahead.

    At most three candidate frames and one resized grayscale image are retained.
    A component needs a coherent, unambiguous three-frame motion witness. This
    rejects static props and repeated frames; it does not classify flight or catch.
    Confidence is a constant proposal flag, not a calibrated probability.
    """
    if not isinstance(profiles, list) or not profiles or any(
            not isinstance(p, BallProfile) for p in profiles):
        raise ValueError("profiles must be a nonempty list of validated BallProfile objects")
    profiles = list(dict.fromkeys(profiles))
    previous = None
    window = deque()
    shape = None
    last_index, last_t = -1, -math.inf
    for index, t, frame in frames:
        _frame(frame)
        if (type(index) is not int or index < 0 or index <= last_index
                or isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, float))
                or not math.isfinite(t) or t < 0 or t <= last_t):
            raise ValueError("frames require increasing nonnegative integer indices and finite PTS")
        if shape is not None and frame.shape != shape:
            raise ValueError("frame geometry must remain fixed within a recording")
        shape = frame.shape
        last_index, last_t = index, float(t)
        current, previous = _components(frame, previous, profiles, index, float(t))
        window.append(current)
        if len(window) == 3:
            _confirm(window, shape[1], shape[0])
            yield from _confirmed_detections(window.popleft())
    for remaining in window:
        yield from _confirmed_detections(remaining)
