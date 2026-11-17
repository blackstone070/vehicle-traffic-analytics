"""OpenCV drawing helpers for the annotated video and live dashboard."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from .state import TrackState


def track_color(track_id: int) -> tuple[int, int, int]:
    """Produce a stable, reasonably bright BGR color for an ID."""
    hue = (track_id * 47) % 180
    hsv = np.uint8([[[hue, 210, 245]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def draw_track(frame: np.ndarray, box: Sequence[float], state: TrackState, speed_unit: str = "kmh") -> None:
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    color = track_color(state.track_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    for index in range(1, len(state.trail)):
        thickness = max(1, int(3 * index / len(state.trail)))
        cv2.line(frame, state.trail[index - 1], state.trail[index], color, thickness, cv2.LINE_AA)
    if state.trail:
        cv2.circle(frame, state.trail[-1], 4, color, -1, cv2.LINE_AA)

    speed_value = state.average_speed_kmh
    if speed_value is not None and speed_unit == "mph":
        speed_value *= 0.621371
    unit_label = "mph" if speed_unit == "mph" else "km/h"
    speed = "speed n/a" if speed_value is None else f"{speed_value:.1f} {unit_label}"
    label = f"{state.vehicle_class} | ID {state.track_id} | {speed}"
    text_y = max(18, y1 - 8)
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    cv2.rectangle(frame, (x1, text_y - text_height - baseline),
                  (x1 + text_width + 6, text_y + baseline), color, -1)
    cv2.putText(frame, label, (x1 + 3, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (12, 12, 12), 2, cv2.LINE_AA)


def draw_count_line(frame: np.ndarray, endpoints: tuple[tuple[int, int], tuple[int, int]]) -> None:
    cv2.line(frame, endpoints[0], endpoints[1], (0, 220, 255), 3, cv2.LINE_AA)
    cv2.circle(frame, endpoints[0], 5, (0, 220, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, endpoints[1], 5, (0, 220, 255), -1, cv2.LINE_AA)
    midpoint = ((endpoints[0][0] + endpoints[1][0]) // 2,
                (endpoints[0][1] + endpoints[1][1]) // 2)
    cv2.putText(frame, "COUNT LINE", (midpoint[0] + 8, midpoint[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    total_count: int,
    class_counts: Mapping[str, int],
    direction_counts: Mapping[str, int],
    active_speeds: Sequence[float],
    processing_fps: float,
    frame_index: int,
    speed_unit: str = "kmh",
) -> None:
    average_speed = sum(active_speeds) / len(active_speeds) if active_speeds else None
    unit_label = "mph" if speed_unit == "mph" else "km/h"
    speed_text = "n/a" if average_speed is None else f"{average_speed:.1f} {unit_label}"
    classes_text = "  ".join(f"{name}: {count}" for name, count in sorted(class_counts.items()))
    directions_text = "  ".join(f"{name}: {count}" for name, count in sorted(direction_counts.items()))
    lines = [
        f"COUNTED: {total_count}    ACTIVE AVG SPEED: {speed_text}    PROCESS FPS: {processing_fps:.1f}",
        f"BY CLASS  {classes_text or 'no crossings yet'}",
        f"DIRECTION  {directions_text or 'no crossings yet'}    FRAME: {frame_index}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.56
    line_height = 25
    panel_width = min(frame.shape[1] - 20, max(cv2.getTextSize(line, font, font_scale, 1)[0][0] for line in lines) + 28)
    panel_height = line_height * len(lines) + 14
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (15, 20, 26), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (20, 34 + index * line_height), font, font_scale,
                    (240, 245, 248), 1, cv2.LINE_AA)

