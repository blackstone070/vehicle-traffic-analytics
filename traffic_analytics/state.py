"""Per-track speed smoothing and count-line crossing logic."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot


Point = tuple[float, float]


def cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def signed_line_side(point: Point, line_a: Point, line_b: Point) -> float:
    """Signed cross product distance from the oriented infinite line."""
    return cross((line_b[0] - line_a[0], line_b[1] - line_a[1]),
                 (point[0] - line_a[0], point[1] - line_a[1]))


def segment_intersection_parameter(p0: Point, p1: Point, q0: Point, q1: Point) -> tuple[float, float] | None:
    """Return the segment parameters (t, u) at an intersection, if any."""
    r = (p1[0] - p0[0], p1[1] - p0[1])
    s = (q1[0] - q0[0], q1[1] - q0[1])
    denominator = cross(r, s)
    if abs(denominator) < 1e-9:
        return None
    delta = (q0[0] - p0[0], q0[1] - p0[1])
    t = cross(delta, s) / denominator
    u = cross(delta, r) / denominator
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return t, u
    return None


@dataclass
class TrackState:
    track_id: int
    vehicle_class: str
    smoothing_window: int = 8
    previous_bev: Point | None = None
    previous_frame: int | None = None
    previous_image_anchor: tuple[int, int] | None = None
    trail: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=32))
    speed_samples: deque[float] = field(init=False)
    crossed_count_line: bool = False
    counted_direction: str | None = None
    last_nonzero_line_side: int | None = None
    last_seen_frame: int = 0

    def __post_init__(self) -> None:
        self.speed_samples = deque(maxlen=self.smoothing_window)

    @property
    def average_speed_kmh(self) -> float | None:
        if not self.speed_samples:
            return None
        return sum(self.speed_samples) / len(self.speed_samples)

    def observe(
        self,
        image_anchor: tuple[int, int],
        bev_point: Point | None,
        frame_index: int,
        fps: float,
        max_speed_kmh: float,
        max_speed_gap_frames: int,
    ) -> None:
        """Update this track's speed history and image-space trail."""
        self.last_seen_frame = frame_index
        self.previous_image_anchor = image_anchor
        self.trail.append(image_anchor)

        if bev_point is None:
            self.previous_bev = None
            self.previous_frame = frame_index
            self.last_nonzero_line_side = None
            return None

        if self.previous_bev is not None and self.previous_frame is not None:
            frame_delta = frame_index - self.previous_frame
            if 0 < frame_delta <= max_speed_gap_frames:
                dt = frame_delta / fps
                speed_kmh = hypot(bev_point[0] - self.previous_bev[0],
                                  bev_point[1] - self.previous_bev[1]) / dt * 3.6
                if 0.0 <= speed_kmh <= max_speed_kmh:
                    self.speed_samples.append(speed_kmh)

        self.previous_bev = bev_point
        self.previous_frame = frame_index
        return None

    def check_count_line(self, previous_bev: Point | None, current_bev: Point,
                         line_a: Point, line_b: Point) -> str | None:
        """Count one finite-line crossing per ID and return its signed direction."""
        if self.crossed_count_line:
            return None
        side_value = signed_line_side(current_bev, line_a, line_b)
        current_side = 1 if side_value > 1e-6 else (-1 if side_value < -1e-6 else 0)
        if current_side == 0:
            return None
        direction = None
        if (self.last_nonzero_line_side is not None
                and current_side != self.last_nonzero_line_side
                and previous_bev is not None
                and segment_intersection_parameter(previous_bev, current_bev, line_a, line_b) is not None):
            direction = ("negative_to_positive" if self.last_nonzero_line_side < 0
                         else "positive_to_negative")
            self.crossed_count_line = True
            self.counted_direction = direction
        self.last_nonzero_line_side = current_side
        return direction

