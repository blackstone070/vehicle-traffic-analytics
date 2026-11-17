"""Interactively calibrate four road-plane points for metric BEV coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "Road Plane Calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Click a known road rectangle to create homography_config.json."
    )
    parser.add_argument("video", help="Input traffic video; its first frame is shown for calibration.")
    parser.add_argument("--width-m", type=float, required=True,
                        help="Real-world width of the rectangle across the road, in meters.")
    parser.add_argument("--length-m", type=float, required=True,
                        help="Real-world length of the rectangle along the road, in meters.")
    parser.add_argument("--output", default="homography_config.json",
                        help="Calibration JSON output path (default: homography_config.json).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.width_m <= 0 or args.length_m <= 0:
        raise SystemExit("--width-m and --length-m must both be greater than zero")

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise SystemExit(f"Could not read the first frame from: {args.video}")

    points: list[tuple[int, int]] = []
    frame_height, frame_width = frame.shape[:2]
    display_scale = min(1.0, 1280.0 / frame_width, 720.0 / frame_height)
    display_size = (max(1, round(frame_width * display_scale)),
                    max(1, round(frame_height * display_scale)))

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((round(x / display_scale), round(y / display_scale)))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    try:
        while True:
            display = cv2.resize(frame, display_size) if display_scale < 1.0 else frame.copy()
            instructions = [
                "Click in order: near-left, near-right, far-right, far-left",
                f"Road rectangle: width {args.width_m:g} m x length {args.length_m:g} m",
                "Enter: save when 4 points are selected | R: reset | right-click: undo | Esc: cancel",
            ]
            for index, message in enumerate(instructions):
                cv2.putText(display, message, (14, 28 + 27 * index), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (20, 20, 20), 4, cv2.LINE_AA)
                cv2.putText(display, message, (14, 28 + 27 * index), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 255, 255), 1, cv2.LINE_AA)

            for index, point in enumerate(points):
                shown_point = (round(point[0] * display_scale), round(point[1] * display_scale))
                cv2.circle(display, shown_point, 7, (0, 220, 255), -1, cv2.LINE_AA)
                cv2.putText(display, str(index + 1), (shown_point[0] + 8, shown_point[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
            if len(points) >= 2:
                for index in range(len(points) - 1):
                    p0 = tuple(round(value * display_scale) for value in points[index])
                    p1 = tuple(round(value * display_scale) for value in points[index + 1])
                    cv2.line(display, p0, p1, (0, 220, 255), 2, cv2.LINE_AA)
            if len(points) == 4:
                p0 = tuple(round(value * display_scale) for value in points[3])
                p1 = tuple(round(value * display_scale) for value in points[0])
                cv2.line(display, p0, p1, (0, 220, 255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                print("Calibration cancelled.")
                return 1
            if key in (ord("r"), ord("R")):
                points.clear()
            if key in (10, 13) and len(points) == 4:
                break
    finally:
        cv2.destroyAllWindows()

    image_points = np.asarray(points, dtype=np.float32)
    if abs(cv2.contourArea(image_points)) < 25.0:
        raise SystemExit("Selected points are degenerate; click a larger road-plane quadrilateral")
    if not cv2.isContourConvex(image_points.astype(np.int32)):
        raise SystemExit("Selected points are not a convex quadrilateral; reset and click corners in the shown order")

    # The first two points are the near edge; the last two are the far edge.
    bev_points = np.asarray([
        [0.0, 0.0],
        [args.width_m, 0.0],
        [args.width_m, args.length_m],
        [0.0, args.length_m],
    ], dtype=np.float32)
    image_to_bev = cv2.getPerspectiveTransform(image_points, bev_points)
    if not np.isfinite(image_to_bev).all() or abs(float(np.linalg.det(image_to_bev))) < 1e-12:
        raise SystemExit("Could not create a stable homography from these points")
    bev_to_image = np.linalg.inv(image_to_bev)

    frame_height, frame_width = frame.shape[:2]
    payload = {
        "coordinate_convention": "x=near-left to near-right in meters; y=near edge to far edge in meters",
        "image_points_order": ["near_left", "near_right", "far_right", "far_left"],
        "image_points_px": image_points.tolist(),
        "bev_points_meters": bev_points.tolist(),
        "homography_image_to_bev": image_to_bev.tolist(),
        "homography_bev_to_image": bev_to_image.tolist(),
        "width_m": float(args.width_m),
        "length_m": float(args.length_m),
        "frame_size_px": [int(frame_width), int(frame_height)],
        "count_line_bev_meters": [
            [0.0, float(args.length_m) / 2.0],
            [float(args.width_m), float(args.length_m) / 2.0],
        ],
        "count_direction_convention": {
            "negative_to_positive": "toward_far",
            "positive_to_negative": "toward_near",
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
    print(f"Saved calibration to {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

