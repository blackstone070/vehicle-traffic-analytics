"""Loading and applying a road-plane homography."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class Calibration:
    """Image-to-ground calibration and the calibrated road quadrilateral."""

    image_to_bev: np.ndarray
    bev_to_image: np.ndarray
    image_points: np.ndarray
    width_m: float
    length_m: float
    frame_width: int
    frame_height: int
    count_line: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            data: dict[str, Any] = json.load(file)

        matrix = np.asarray(data["homography_image_to_bev"], dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"{config_path}: homography matrix must be 3x3")
        if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
            raise ValueError(f"{config_path}: homography matrix is singular or non-finite")
        inverse_data = data.get("homography_bev_to_image")
        inverse = np.asarray(inverse_data, dtype=np.float64) if inverse_data is not None else np.linalg.inv(matrix)
        image_points = np.asarray(data["image_points_px"], dtype=np.float32)
        width_m = float(data["width_m"])
        length_m = float(data["length_m"])
        frame_size = data["frame_size_px"]
        count_line = np.asarray(data["count_line_bev_meters"], dtype=np.float32)

        if inverse.shape != (3, 3):
            raise ValueError(f"{config_path}: homography matrices must be 3x3")
        if image_points.shape != (4, 2):
            raise ValueError(f"{config_path}: image_points_px must contain four (x, y) points")
        if count_line.shape != (2, 2):
            raise ValueError(f"{config_path}: count_line_bev_meters must contain two (x, y) points")
        if not np.isfinite(inverse).all():
            raise ValueError(f"{config_path}: inverse homography matrix is non-finite")
        if width_m <= 0 or length_m <= 0:
            raise ValueError(f"{config_path}: width_m and length_m must be positive")
        if np.linalg.norm(count_line[1] - count_line[0]) < 1e-6:
            raise ValueError(f"{config_path}: count line endpoints must be distinct")

        return cls(
            image_to_bev=matrix,
            bev_to_image=inverse,
            image_points=image_points,
            width_m=width_m,
            length_m=length_m,
            frame_width=int(frame_size[0]),
            frame_height=int(frame_size[1]),
            count_line=count_line,
        )

    def project(self, image_point: tuple[float, float]) -> tuple[float, float] | None:
        """Project one pixel point to meters, returning None for invalid results."""
        source = np.asarray([[[image_point[0], image_point[1]]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(source, self.image_to_bev.astype(np.float64))[0, 0]
        if not np.isfinite(mapped).all():
            return None
        return float(mapped[0]), float(mapped[1])

    def is_inside_image_region(self, point: tuple[float, float]) -> bool:
        """True when a point is inside the polygon used to create the homography."""
        return cv2.pointPolygonTest(self.image_points, point, False) >= 0

    def image_count_line(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Map the configured ground-plane count line back into image pixels."""
        bev = np.asarray([self.count_line], dtype=np.float32)
        image = cv2.perspectiveTransform(bev, self.bev_to_image.astype(np.float64))[0]
        return (tuple(int(value) for value in np.rint(image[0])),
                tuple(int(value) for value in np.rint(image[1])))

