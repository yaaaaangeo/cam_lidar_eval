"""
input/camera.py

Camera model + camera data loader, per the Input Loader Spec (v0.1) in
evaluation_metric_spec.md.

Responsibility: PARSE ONLY. This module turns raw camera config + files into
a standardized CameraModel + list of Frame objects. It does not validate
calibration correctness (that's input/extrinsic.py's verify_extrinsic) and
does not compute any evaluation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import glob
import os

import numpy as np
import cv2

from geometry.projection import intrinsics_matrix, plumb_bob_dist_coeffs, fisheye_dist_coeffs


SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def as_matrix(self) -> np.ndarray:
        return intrinsics_matrix(self.fx, self.fy, self.cx, self.cy)


@dataclass
class CameraDistortion:
    model: Literal["plumb_bob", "fisheye_equidistant", "none"]
    coeffs: dict = field(default_factory=dict)

    def as_array(self) -> Optional[np.ndarray]:
        if self.model == "none":
            return None
        if self.model == "plumb_bob":
            return plumb_bob_dist_coeffs(self.coeffs)
        if self.model == "fisheye_equidistant":
            return fisheye_dist_coeffs(self.coeffs)
        raise ValueError(f"Unknown distortion model: {self.model!r}")


@dataclass
class CameraSource:
    kind: Literal["image_dir", "video", "rosbag", "ros_topic"]
    path: str
    topic: Optional[str] = None
    timestamp_source: Literal["filename", "embedded", "topic_header"] = "filename"


@dataclass
class CameraModel:
    width: int
    height: int
    model: Literal["pinhole", "fisheye"]
    intrinsics: CameraIntrinsics
    distortion: CameraDistortion
    source: CameraSource

    # floor(Z) Term 3 -- optional, see quality/noise_floor.py
    edge_localization_floor_px: Optional[float] = None

    def K(self) -> np.ndarray:
        return self.intrinsics.as_matrix()

    def dist_coeffs(self) -> Optional[np.ndarray]:
        return self.distortion.as_array()

    def projection_model_name(self) -> str:
        """Maps the high-level 'pinhole'/'fisheye' model field to the
        projection function selector used by geometry/projection.py."""
        return self.model


@dataclass
class CameraFrame:
    timestamp: float
    path: Optional[str] = None
    image: Optional[np.ndarray] = None  # lazily loaded if only `path` is set

    def load(self) -> np.ndarray:
        """Return the image array, loading from disk on first access if
        needed. Cached on the frame object after first load."""
        if self.image is not None:
            return self.image
        if self.path is None:
            raise ValueError("CameraFrame has neither `image` nor `path` set.")
        img = cv2.imread(self.path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Failed to read image at {self.path}")
        self.image = img
        return img


class CameraLoadWarning(RuntimeWarning):
    pass


@dataclass
class CameraLoadResult:
    camera: CameraModel
    frames: list[CameraFrame]
    warnings: list[str] = field(default_factory=list)


def _timestamp_from_filename(path: str) -> float:
    """
    Extract a timestamp from a filename stem. Supports:
      - pure numeric stems (e.g. '1699999999.123456.png' or '000123.png')
      - falls back to file index order (returned as float) with a caller-side
        warning if the stem isn't numeric -- handled by the caller so it can
        aggregate a single warning instead of one per file.
    """
    stem = Path(path).stem
    try:
        return float(stem)
    except ValueError:
        return float("nan")


def load_camera_from_image_dir(
    path: str,
    width: int,
    height: int,
    model: Literal["pinhole", "fisheye"],
    intrinsics: CameraIntrinsics,
    distortion: CameraDistortion,
    timestamp_source: Literal["filename", "embedded"] = "filename",
    edge_localization_floor_px: Optional[float] = None,
    lazy: bool = True,
) -> CameraLoadResult:
    """
    Load a CameraModel + sorted list of CameraFrame from a directory of
    image files.

    timestamp_source:
      - 'filename': parse timestamp from the numeric filename stem. If
        filenames aren't numeric, falls back to sequential indices
        (0, 1, 2, ...) and records a warning -- downstream sync (dataset.py)
        will then be unable to do real timestamp matching, which the
        warning makes explicit rather than failing silently.
      - 'embedded': not implemented in this pass (would require per-format
        metadata extraction, e.g. EXIF); raises NotImplementedError.
    """
    warnings: list[str] = []

    files = sorted(
        f for f in glob.glob(os.path.join(path, "*"))
        if os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No supported image files found in {path!r} "
                                 f"(looked for {SUPPORTED_IMAGE_EXTENSIONS})")

    if timestamp_source == "embedded":
        raise NotImplementedError(
            "timestamp_source='embedded' is not implemented yet; use 'filename'."
        )
    if timestamp_source != "filename":
        raise ValueError(f"Unsupported timestamp_source for image_dir: {timestamp_source!r}")

    raw_timestamps = [_timestamp_from_filename(f) for f in files]
    if any(np.isnan(t) for t in raw_timestamps):
        warnings.append(
            "One or more image filenames were not numeric; falling back to "
            "sequential integer timestamps (0, 1, 2, ...). Timestamp-based "
            "sync with LiDAR frames will not reflect real capture time."
        )
        raw_timestamps = [float(i) for i in range(len(files))]

    frames = [
        CameraFrame(timestamp=ts, path=f, image=None)
        for ts, f in zip(raw_timestamps, files)
    ]

    if not lazy:
        for fr in frames:
            fr.load()

    source = CameraSource(kind="image_dir", path=path, timestamp_source=timestamp_source)
    camera = CameraModel(
        width=width, height=height, model=model,
        intrinsics=intrinsics, distortion=distortion, source=source,
        edge_localization_floor_px=edge_localization_floor_px,
    )

    return CameraLoadResult(camera=camera, frames=frames, warnings=warnings)


def load_camera_from_video(*args, **kwargs) -> CameraLoadResult:
    raise NotImplementedError(
        "Video source loading is not implemented in this pass. "
        "Use load_camera_from_image_dir with pre-extracted frames, "
        "or extend this function (cv2.VideoCapture) as a follow-up."
    )


def load_camera_from_rosbag(*args, **kwargs) -> CameraLoadResult:
    raise NotImplementedError(
        "rosbag/ros_topic camera loading requires ROS message deserialization "
        "dependencies (rosbag2_py / rclpy) not included in this environment. "
        "Implement as a follow-up when ROS tooling is available."
    )
