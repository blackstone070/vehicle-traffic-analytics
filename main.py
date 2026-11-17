"""Vehicle detection, tracking, road-plane speed estimation, and flow counting."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from traffic_analytics.drawing import draw_count_line, draw_hud, draw_track
from traffic_analytics.geometry import Calibration
from traffic_analytics.state import TrackState


VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
KMH_TO_MPH = 0.621371


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track road vehicles, estimate ground-plane speed, and count line crossings."
    )
    parser.add_argument("source", help="Video file, stream URL, or numeric camera index.")
    parser.add_argument("--calibration", default="homography_config.json",
                        help="JSON created by calibrate.py (default: homography_config.json).")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics YOLO weights (default: yolo11n.pt).")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config (default: bytetrack.yaml).")
    parser.add_argument("--output-video", default="results/annotated.mp4", help="Annotated video output path.")
    parser.add_argument("--csv", default="results/traffic_log.csv", help="Count-line crossing CSV output path.")
    parser.add_argument("--summary-json", default="results/traffic_summary.json", help="Aggregate traffic summary JSON path.")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override video FPS metadata; required when the source does not report FPS reliably.")
    parser.add_argument("--conf", type=float, default=0.15, help="Detector confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--device", default=None, help="Ultralytics device, such as cpu, 0, or 0,1.")
    parser.add_argument("--smooth-window", type=int, default=8, help="Per-track moving-average window in observations.")
    parser.add_argument("--max-speed-kmh", type=float, default=220.0,
                        help="Reject implausible instantaneous speed samples above this value.")
    parser.add_argument("--max-speed-gap-frames", type=int, default=15,
                        help="Skip speed updates after longer observation gaps, such as long occlusions.")
    parser.add_argument("--track-memory-frames", type=int, default=120,
                        help="Forget local display/speed state after this many missed frames.")
    parser.add_argument("--unit", choices=("kmh", "mph"), default="kmh", help="Speed unit in labels and CSV.")
    parser.add_argument("--display", action="store_true", help="Show a live preview; press Q to stop early.")
    return parser.parse_args()


def model_class_names(names: Any) -> dict[str, int]:
    items = names.items() if isinstance(names, dict) else enumerate(names)
    return {str(name).lower(): int(class_id) for class_id, name in items}


def format_timestamp(frame_index: int, fps: float) -> str:
    milliseconds = round(frame_index / fps * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def to_output_speed(speed_kmh: float | None, unit: str) -> float | None:
    if speed_kmh is None:
        return None
    return speed_kmh if unit == "kmh" else speed_kmh * KMH_TO_MPH


def run() -> int:
    args = parse_args()
    if args.smooth_window < 1:
        raise SystemExit("--smooth-window must be at least 1")
    if args.max_speed_gap_frames < 1 or args.track_memory_frames < 1:
        raise SystemExit("--max-speed-gap-frames and --track-memory-frames must be positive")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        raise SystemExit("--fps must be greater than zero")
    if args.max_speed_kmh <= 0 or args.imgsz <= 0:
        raise SystemExit("--max-speed-kmh and --imgsz must be greater than zero")

    calibration = Calibration.load(args.calibration)
    source: str | int = int(args.source) if args.source.isdecimal() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video or stream source: {args.source}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = float(args.fps) if args.fps is not None else source_fps
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise SystemExit("The source did not report a usable FPS. Supply --fps with the native frame rate.")

    ok, first_frame = capture.read()
    if not ok or first_frame is None:
        capture.release()
        raise SystemExit(f"Could not read a frame from: {args.source}")
    frame_height, frame_width = first_frame.shape[:2]
    if (frame_width, frame_height) != (calibration.frame_width, calibration.frame_height):
        capture.release()
        raise SystemExit(
            "Input resolution does not match the calibration: "
            f"video is {frame_width}x{frame_height}, calibration is "
            f"{calibration.frame_width}x{calibration.frame_height}. Recalibrate at the input resolution."
        )

    try:
        from ultralytics import YOLO
    except ImportError as error:
        capture.release()
        raise SystemExit("Ultralytics is not installed. Run `python -m pip install -r requirements.txt`.") from error

    print(f"Loading model {args.model} ...")
    model = YOLO(args.model)
    class_ids_by_name = model_class_names(model.names)
    selected_class_ids = [class_ids_by_name[name] for name in sorted(VEHICLE_CLASSES) if name in class_ids_by_name]
    if not selected_class_ids:
        capture.release()
        raise SystemExit("The selected model has no COCO-style car, motorcycle, bus, or truck classes.")
    resolved_names = {class_id: name for name, class_id in class_ids_by_name.items()}

    output_video = Path(args.output_video)
    csv_path = Path(args.csv)
    summary_path = Path(args.summary_json)
    for path in (output_video, csv_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    if not output_video.suffix:
        output_video = output_video.with_suffix(".mp4")
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
    )
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"Could not create output video: {output_video}")

    count_line = tuple(tuple(float(value) for value in point) for point in calibration.count_line)
    count_line_pixels = calibration.image_count_line()
    tracks: dict[int, TrackState] = {}
    class_counts: Counter[str] = Counter()
    direction_counts: Counter[str] = Counter()
    crossing_speeds: list[float] = []
    frame_index = 0
    processing_fps = 0.0
    inference_kwargs: dict[str, Any] = {
        "persist": True,
        "tracker": args.tracker,
        "classes": selected_class_ids,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device is not None:
        inference_kwargs["device"] = args.device

    print(f"Processing at {fps:.3f} source FPS. Output: {output_video}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=("timestamp", "track_id", "vehicle_class", "avg_speed", "direction"),
            )
            csv_writer.writeheader()

            pending_frame = first_frame
            while pending_frame is not None:
                frame_started = time.perf_counter()
                frame = pending_frame
                pending_frame = None

                results = model.track(frame, **inference_kwargs)
                result = results[0] if results else None
                boxes = result.boxes if result is not None else None
                if boxes is not None and boxes.id is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    track_ids = boxes.id.int().cpu().numpy()
                    class_indexes = boxes.cls.int().cpu().numpy()

                    for box, raw_track_id, class_id in zip(xyxy, track_ids, class_indexes):
                        track_id = int(raw_track_id)
                        vehicle_class = resolved_names.get(int(class_id), str(int(class_id)))
                        x1, y1, x2, y2 = (float(value) for value in box)
                        anchor = (int(round((x1 + x2) / 2.0)), int(round(y2)))
                        anchor_float = (float(anchor[0]), float(anchor[1]))
                        bev_point = None
                        if calibration.is_inside_image_region(anchor_float):
                            projected = calibration.project(anchor_float)
                            if projected is not None:
                                x_m, y_m = projected
                                if -0.25 <= x_m <= calibration.width_m + 0.25 and -0.25 <= y_m <= calibration.length_m + 0.25:
                                    bev_point = projected

                        state = tracks.get(track_id)
                        if state is None:
                            state = TrackState(track_id=track_id, vehicle_class=vehicle_class,
                                               smoothing_window=args.smooth_window)
                            tracks[track_id] = state
                        else:
                            state.vehicle_class = vehicle_class

                        previous_bev = state.previous_bev
                        direction = None
                        if bev_point is not None:
                            direction = state.check_count_line(previous_bev, bev_point,
                                                               count_line[0], count_line[1])
                        state.observe(
                            image_anchor=anchor,
                            bev_point=bev_point,
                            frame_index=frame_index,
                            fps=fps,
                            max_speed_kmh=args.max_speed_kmh,
                            max_speed_gap_frames=args.max_speed_gap_frames,
                        )
                        if direction is not None:
                            class_counts[vehicle_class] += 1
                            direction_counts[direction] += 1
                            output_speed = to_output_speed(state.average_speed_kmh, args.unit)
                            if output_speed is not None:
                                crossing_speeds.append(output_speed)
                            csv_writer.writerow({
                                "timestamp": format_timestamp(frame_index, fps),
                                "track_id": track_id,
                                "vehicle_class": vehicle_class,
                                "avg_speed": "" if output_speed is None else f"{output_speed:.2f}",
                                "direction": direction,
                            })
                            csv_file.flush()

                        draw_track(frame, box, state, args.unit)

                stale_ids = [track_id for track_id, state in tracks.items()
                             if frame_index - state.last_seen_frame > args.track_memory_frames]
                for track_id in stale_ids:
                    del tracks[track_id]

                active_speeds = [
                    speed for state in tracks.values()
                    if frame_index - state.last_seen_frame <= int(fps * 2)
                    for speed in [to_output_speed(state.average_speed_kmh, args.unit)]
                    if speed is not None
                ]
                elapsed = max(time.perf_counter() - frame_started, 1e-6)
                instantaneous_fps = 1.0 / elapsed
                processing_fps = instantaneous_fps if processing_fps == 0 else 0.9 * processing_fps + 0.1 * instantaneous_fps
                draw_count_line(frame, count_line_pixels)
                draw_hud(frame, sum(class_counts.values()), class_counts, direction_counts,
                         active_speeds, processing_fps, frame_index, args.unit)
                writer.write(frame)
                frame_index += 1

                if args.display:
                    cv2.imshow("Vehicle Traffic Analytics", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                        break

                ok, next_frame = capture.read()
                if ok:
                    pending_frame = next_frame

    finally:
        capture.release()
        writer.release()
        if args.display:
            cv2.destroyAllWindows()

    summary = {
        "source": args.source,
        "processed_frames": frame_index,
        "source_fps": fps,
        "speed_unit": "mph" if args.unit == "mph" else "km/h",
        "total_counted_crossings": int(sum(class_counts.values())),
        "counts_by_class": dict(sorted(class_counts.items())),
        "counts_by_direction": dict(sorted(direction_counts.items())),
        "average_crossing_speed": None if not crossing_speeds else round(sum(crossing_speeds) / len(crossing_speeds), 2),
        "vehicles_with_speed_at_crossing": len(crossing_speeds),
        "count_line_bev_meters": calibration.count_line.tolist(),
        "outputs": {"annotated_video": str(output_video), "traffic_log_csv": str(csv_path)},
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
        summary_file.write("\n")

    print(f"Processed {frame_index} frames; counted {sum(class_counts.values())} line crossings.")
    print(f"CSV: {csv_path.resolve()}")
    print(f"Annotated video: {output_video.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

