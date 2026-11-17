"""Streamlit Community Cloud interface for calibration and traffic analysis."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates


PROJECT_DIR = Path(__file__).resolve().parent
VEHICLE_POINT_ORDER = ("near-left", "near-right", "far-right", "far-left")
MAX_VIDEO_BYTES = 200 * 1024 * 1024
MAX_VIDEO_SECONDS = 120


def make_calibration(points: list[tuple[float, float]], width_m: float, length_m: float,
                     frame_width: int, frame_height: int) -> dict[str, Any]:
    image_points = np.asarray(points, dtype=np.float32)
    if image_points.shape != (4, 2):
        raise ValueError("Select exactly four road-plane points.")
    if abs(cv2.contourArea(image_points)) < 25 or not cv2.isContourConvex(image_points.astype(np.int32)):
        raise ValueError("The four points must form a non-degenerate convex quadrilateral in the shown order.")

    bev_points = np.asarray([[0, 0], [width_m, 0], [width_m, length_m], [0, length_m]], dtype=np.float32)
    image_to_bev = cv2.getPerspectiveTransform(image_points, bev_points)
    if not np.isfinite(image_to_bev).all() or abs(float(np.linalg.det(image_to_bev))) < 1e-12:
        raise ValueError("Could not form a stable homography from those points.")
    return {
        "coordinate_convention": "x=near-left to near-right in meters; y=near edge to far edge in meters",
        "image_points_order": ["near_left", "near_right", "far_right", "far_left"],
        "image_points_px": image_points.tolist(),
        "bev_points_meters": bev_points.tolist(),
        "homography_image_to_bev": image_to_bev.tolist(),
        "homography_bev_to_image": np.linalg.inv(image_to_bev).tolist(),
        "width_m": float(width_m),
        "length_m": float(length_m),
        "frame_size_px": [int(frame_width), int(frame_height)],
        "count_line_bev_meters": [[0.0, float(length_m) / 2], [float(width_m), float(length_m) / 2]],
        "count_direction_convention": {
            "negative_to_positive": "toward_far",
            "positive_to_negative": "toward_near",
        },
    }


def show_saved_results() -> None:
    video_bytes = st.session_state.get("result_video")
    if not video_bytes:
        return
    st.success("Analysis complete.")
    st.video(video_bytes)
    st.download_button("Download annotated video", video_bytes, "annotated.mp4", "video/mp4")
    st.download_button("Download traffic CSV", st.session_state["result_csv"], "traffic_log.csv", "text/csv")
    st.download_button("Download summary JSON", st.session_state["result_summary"],
                       "traffic_summary.json", "application/json")
    st.json(json.loads(st.session_state["result_summary"].decode("utf-8")))


def render_app() -> None:
    st.set_page_config(page_title="Vehicle Traffic Analytics", page_icon="🚘", layout="wide")
    st.title("🚘 Vehicle Traffic Analytics")
    st.caption("YOLO + ByteTrack tracking, metric road-plane speed, and directional crossing counts.")
    st.info("Uploaded footage is processed temporarily by this app. Avoid uploading sensitive video. "
            "The public demo runs on CPU and is limited to 120 seconds per clip.")

    uploaded_video = st.file_uploader("Upload a traffic video", type=["mp4", "mov", "avi", "mkv", "webm"])
    if uploaded_video is None:
        st.markdown("Upload a fixed-camera clip to calibrate the road rectangle and run analysis.")
        return

    video_bytes = uploaded_video.getvalue()
    if len(video_bytes) > MAX_VIDEO_BYTES:
        st.error("This demo accepts video files up to 200 MB.")
        return
    video_id = hashlib.sha256(video_bytes).hexdigest()
    if st.session_state.get("video_id") != video_id:
        st.session_state["video_id"] = video_id
        st.session_state["calibration_points"] = []
        st.session_state["last_click_time"] = None
        st.session_state.pop("result_video", None)

    width_col, length_col = st.columns(2)
    with width_col:
        width_m = st.number_input("Measured road rectangle width (meters)", min_value=0.1, value=7.0, step=0.1)
    with length_col:
        length_m = st.number_input("Measured road rectangle length (meters)", min_value=0.1, value=30.0, step=0.5)

    suffix = Path(uploaded_video.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as input_file:
        input_file.write(video_bytes)
        input_path = Path(input_file.name)

    try:
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            st.error("OpenCV could not read this video. Try MP4 encoded with H.264 or MPEG-4.")
            return
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, first_frame = capture.read()
        capture.release()
        if not ok or first_frame is None:
            st.error("Could not read the first video frame.")
            return
        frame_height, frame_width = first_frame.shape[:2]
        duration = frame_count / fps if fps > 0 and frame_count > 0 else None
        if duration is not None and duration > MAX_VIDEO_SECONDS:
            st.error(f"This public CPU demo accepts clips up to {MAX_VIDEO_SECONDS} seconds; this clip is {duration:.1f}s.")
            return
        fps_override = st.number_input("FPS override (0 uses video metadata)", min_value=0.0,
                                       max_value=120.0, value=0.0, step=0.1)
        if fps <= 0 and fps_override <= 0:
            st.error("This video has no usable FPS metadata. Enter the clip's native FPS above.")
            return

        st.subheader("Calibrate the road rectangle")
        st.write("Click the road surface corners in order: near-left → near-right → far-right → far-left.")
        frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        image_width = min(960, frame_width)
        preview_frame = frame_rgb.copy()
        for point_index, point in enumerate(st.session_state["calibration_points"]):
            shown_point = (round(point[0] * image_width / frame_width),
                           round(point[1] * image_width / frame_width))
            cv2.circle(preview_frame, shown_point, 7, (255, 205, 40), -1, cv2.LINE_AA)
            cv2.putText(preview_frame, str(point_index + 1),
                        (shown_point[0] + 9, shown_point[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 205, 40), 2, cv2.LINE_AA)
            if point_index > 0:
                previous = st.session_state["calibration_points"][point_index - 1]
                previous_shown = (round(previous[0] * image_width / frame_width),
                                  round(previous[1] * image_width / frame_width))
                cv2.line(preview_frame, previous_shown, shown_point, (255, 205, 40), 2, cv2.LINE_AA)
        if len(st.session_state["calibration_points"]) == 4:
            first = st.session_state["calibration_points"][0]
            first_shown = (round(first[0] * image_width / frame_width),
                           round(first[1] * image_width / frame_width))
            last = st.session_state["calibration_points"][3]
            last_shown = (round(last[0] * image_width / frame_width),
                          round(last[1] * image_width / frame_width))
            cv2.line(preview_frame, last_shown, first_shown, (255, 205, 40), 2, cv2.LINE_AA)
        click = streamlit_image_coordinates(
            preview_frame, width=image_width, key=f"calibration-{video_id[:12]}", cursor="crosshair"
        )
        if isinstance(click, dict) and click.get("x") is not None and click.get("unix_time") != st.session_state.get("last_click_time"):
            display_width = float(click.get("width", image_width))
            display_height = float(click.get("height", round(frame_height * image_width / frame_width)))
            px = float(click["x"]) * frame_width / max(display_width, 1.0)
            py = float(click["y"]) * frame_height / max(display_height, 1.0)
            if len(st.session_state["calibration_points"]) < 4:
                st.session_state["calibration_points"].append((round(px, 2), round(py, 2)))
                st.session_state["last_click_time"] = click.get("unix_time")

        points = st.session_state["calibration_points"]
        if st.button("Undo last point", disabled=not points):
            points.pop()
            st.session_state["calibration_points"] = points
        if st.button("Reset calibration points", disabled=not points):
            st.session_state["calibration_points"] = []
            points = []

        if points:
            st.write({f"{index + 1}. {VEHICLE_POINT_ORDER[index]}": point for index, point in enumerate(points)})
        if len(points) == 4:
            try:
                calibration = make_calibration(points, width_m, length_m, frame_width, frame_height)
                config_bytes = json.dumps(calibration, indent=2).encode("utf-8")
                st.download_button("Download homography_config.json", config_bytes,
                                   "homography_config.json", "application/json")
            except ValueError as error:
                st.error(str(error))
                calibration = None
        else:
            calibration = None
            st.caption(f"{len(points)} of 4 points selected.")

        st.subheader("Run analysis")
        unit = st.selectbox("Speed unit", ["kmh", "mph"], format_func=lambda value: "km/h" if value == "kmh" else "mph")
        smooth_window = st.slider("Speed smoothing window (observations)", 5, 10, 8)
        run_clicked = st.button("Analyze video", type="primary", disabled=calibration is None)
        if run_clicked and calibration is not None:
            with tempfile.TemporaryDirectory(prefix="traffic-results-") as output_dir:
                output_root = Path(output_dir)
                calibration_path = output_root / "homography_config.json"
                calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
                old_argv = sys.argv
                sys.argv = [
                    str(PROJECT_DIR / "main.py"), str(input_path),
                    "--calibration", str(calibration_path),
                    "--output-video", str(output_root / "annotated.mp4"),
                    "--csv", str(output_root / "traffic_log.csv"),
                    "--summary-json", str(output_root / "traffic_summary.json"),
                    "--unit", unit,
                    "--smooth-window", str(smooth_window),
                    "--device", "cpu",
                ]
                if fps_override > 0:
                    sys.argv.extend(["--fps", str(fps_override)])
                try:
                    with st.spinner("Running detection, tracking, and speed estimation…"):
                        sys.path.insert(0, str(PROJECT_DIR)) if str(PROJECT_DIR) not in sys.path else None
                        import main as pipeline
                        pipeline.run()
                    st.session_state["result_video"] = (output_root / "annotated.mp4").read_bytes()
                    st.session_state["result_csv"] = (output_root / "traffic_log.csv").read_bytes()
                    st.session_state["result_summary"] = (output_root / "traffic_summary.json").read_bytes()
                except (Exception, SystemExit) as error:
                    st.error(f"Analysis failed: {error}")
                finally:
                    sys.argv = old_argv
        show_saved_results()
    finally:
        input_path.unlink(missing_ok=True)


if __name__ == "__main__":
    render_app()

