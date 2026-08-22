"""
app/cli.py

Command-line entry point for the Cam-LiDAR Calibration Evaluation tool.
Ties together everything built so far:

  input/          -> load camera, LiDAR, extrinsic, sync into a dataset
  evaluation/      -> M0 sanity gate, M2/M3/M4 metrics
  quality/         -> sensor-relative floor(Z), 0-100 scoring, category aggregation
  visualization/   -> overlay / trajectory / histogram PNGs
  report/          -> JSON + HTML report assembly and serialization

Two ways to run it:

  1. Against real data, via a YAML config:
       python -m app.cli --config path/to/config.yaml --output-dir out/

  2. Against a built-in synthetic scene, for a quick smoke test with no
     data of your own:
       python -m app.cli --demo --output-dir out/
       python -m app.cli --demo --scenario drift --output-dir out/

Config YAML schema (mirrors the Input Loader Spec in
evaluation_metric_spec.md):

  camera:
    source: image_dir                 # optional, default: image_dir. Or: rosbag
    image_dir: path/to/images         # required if source == image_dir
    # rosbag_path: path/to/bag        # required if source == rosbag (rosbag1
    #                                  # .bag file or rosbag2 directory)
    # topic: /camera/image_raw        # optional if source == rosbag and the bag
    #                                  # has exactly one Image/CompressedImage topic
    width: 1920
    height: 1080
    model: pinhole            # or fisheye
    intrinsics: {fx: ..., fy: ..., cx: ..., cy: ...}
    distortion: {model: plumb_bob, coeffs: {k1: ..., k2: ..., p1: ..., p2: ...}}
    edge_localization_floor_px: 0.5   # optional

  lidar:
    source: pcd_dir                   # optional, default: pcd_dir. Or: rosbag
    pcd_dir: path/to/pointclouds      # required if source == pcd_dir (.pcd or .ply)
    # rosbag_path: path/to/bag        # required if source == rosbag
    # topic: /lidar/points            # optional if source == rosbag and the bag
    #                                  # has exactly one PointCloud2 topic
    sensor_spec:
      horizontal_resolution_deg: 0.2   # optional (see fallback rules in spec)
      vertical_resolution_deg: 0.2     # optional
      channels: 32                    # optional, used if vertical_resolution_deg absent
      vertical_fov_deg: 40.0          # optional, used with channels
      range_accuracy_m: 0.02          # optional
      min_range_m: 0.1
      max_range_m: 200.0

  extrinsic:
    parent: lidar                     # or camera
    child: camera                     # or lidar
    translation: [0.1, -0.05, 0.2]
    rotation: [0.0, 0.0, 0.0]         # meaning depends on rotation_format
    rotation_format: rpy_deg          # rpy_deg | rpy_rad | quaternion | matrix3x3 | matrix4x4
    unit: m                           # m | cm | mm

  evaluation:                         # all optional, shown with defaults
    sync_max_time_diff_ms: 50.0
    n_blocks: 4
    min_frames_per_block: 30
    min_frames_m4: 30
    depth_jump_threshold_m: 0.3
    edge_radius_px: 3.0
    frame_index: null                 # which frame M2's headline number comes from;
                                       # null -> the temporally-middle frame
    weights: null                     # e.g. {geometry: 0.5, generalization: 0.25, stability: 0.25}

rosbag source note: reading rosbag1 (.bag) or rosbag2 (directory) files
requires the optional `rosbags` (+ `rosbags-image` for camera) dependency:
    pip install "cam-lidar-eval[rosbag]"
This is a pure-Python bag reader -- no ROS/rclpy installation is needed.
Live `ros_topic` (subscribing to a running ROS node) is NOT supported --
that needs an active ROS2 middleware/DDS connection, which is a
categorically different thing from reading an already-recorded bag file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Optional

import numpy as np
import yaml
import jsonschema

from input.camera import CameraIntrinsics, CameraDistortion, load_camera_from_image_dir, load_camera_from_rosbag
from input.lidar import LidarSensorSpec, load_lidar_from_pcd_dir, load_lidar_from_rosbag
from input.extrinsic import ExtrinsicRaw, load_extrinsic, verify_extrinsic
from input.dataset import SyncConfig, build_dataset, EvaluationDataset, SyncedFrame, SyncStats
from input.validation import validate_input, InputValidationError, ValidationStatus

from evaluation.sanity_gate import run_sanity_gate
from quality.noise_floor import resolve_floor_inputs
from evaluation.edge_alignment import evaluate_edge_alignment
from evaluation.holdout_consistency import evaluate_holdout_consistency
from evaluation.multiframe_consistency import evaluate_multiframe_consistency
from evaluation.plane_consistency import evaluate_plane_consistency
from evaluation.perturbation import evaluate_perturbation_sensitivity
from visualization.sensitivity_plot import render_sensitivity_from_result
from evaluation.temporal_drift import evaluate_temporal_drift

from quality.quality_score import compute_quality_score

from visualization.overlay import render_overlay_from_result, encode_png
from visualization.projection_overlay import render_projection_overlay_from_frame
from visualization.range_image import render_range_image_from_points
from visualization.deskew_comparison import render_deskew_comparison_png
from visualization.uncertainty_plot import render_uncertainty_plot_from_result
from visualization.spatial_analysis_plot import render_spatial_analysis_from_result
from evaluation.dynamic_filter import classify_points_by_motion_consistency, dynamic_point_mask, compare_with_without_dynamic_filtering
from visualization.dynamic_filter_overlay import render_dynamic_filter_overlay_from_frame
from motion.deskew import deskew_points_constant_velocity, compare_before_after
from visualization.trajectory import render_m4_trajectory_png
from visualization.histogram import render_error_histogram_png
from visualization.colorized_pointcloud import render_colorized_pointcloud_from_frame
from visualization.error_heatmap import render_error_heatmap_from_result
from visualization.camera_frustum import render_camera_frustum_from_dataset
from visualization.bev_dual_panel import render_bev_dual_panel_from_result
from visualization.interactive_viewer import build_interactive_scene_from_dataset
from visualization.sequence import render_sequence_gif

from report.builder import build_report
from report.json import write_json_report
from report.html import write_html_report
from report.diff import compute_report_diff, render_diff_console
from report.markdown import render_github_comment


# ---------------------------------------------------------------------------
# Config loading (real-data path)
# ---------------------------------------------------------------------------

class ConfigSchemaError(ValueError):
    """Raised when a --config YAML is missing required keys or has an
    invalid structure. Carries a human-readable, actionable message (every
    problem found, collected in one pass, plus a pointer to where the
    schema is documented) instead of letting a raw KeyError/TypeError from
    deep inside config parsing surface to the user."""


_REQUIRED_TOP_LEVEL_KEYS = ("camera", "lidar", "extrinsic")
_REQUIRED_EXTRINSIC_KEYS = ("parent", "child", "rotation", "rotation_format")
_VALID_ROTATION_FORMATS = ("rpy_deg", "rpy_rad", "quaternion", "matrix3x3", "matrix4x4")
_VALID_EXTRINSIC_ROLES = ("lidar", "camera")
_VALID_CAMERA_SOURCES = ("image_dir", "rosbag")
_VALID_LIDAR_SOURCES = ("pcd_dir", "rosbag")

_SCHEMA_POINTER = (
    "See the full config schema in `python -m app.cli --help`'s epilog, "
    "app/cli.py's module docstring, or evaluation_metric_spec.md's "
    "Input Loader Spec."
)

# Declarative JSON Schema (Draft 7) for --config, replacing what used to
# be ~80 lines of hand-written if/isinstance/for-key-in checks. The
# required-key lists, enum values, and "image_dir required unless
# source=rosbag" conditional logic all now live in this one data
# structure instead of scattered imperative control flow -- adding a
# new required key or enum value going forward is a one-line schema
# edit, not a new if-block to remember to write correctly.
#
# camera.source / lidar.source select which loader is used
# ("image_dir"/"rosbag" and "pcd_dir"/"rosbag" respectively, each
# defaulting to the directory-based source if omitted for backward
# compatibility with configs written before rosbag support existed).
# The if/then/else blocks below encode exactly that: jsonschema's `if`
# only matches when `source` is present AND equals "rosbag" (`required:
# [source]` inside `if` is what makes a missing source fall through to
# `else` rather than vacuously matching `if`), so an absent source
# correctly requires image_dir/pcd_dir, matching the loaders' own
# defaulting behavior.
_CONFIG_JSON_SCHEMA: dict = {
    "type": "object",
    "required": list(_REQUIRED_TOP_LEVEL_KEYS),
    "properties": {
        "camera": {
            "type": "object",
            "required": ["width", "height", "intrinsics"],
            "properties": {
                "source": {"enum": list(_VALID_CAMERA_SOURCES)},
                "intrinsics": {
                    "type": "object",
                    "required": ["fx", "fy", "cx", "cy"],
                },
            },
            "allOf": [{
                "if": {"required": ["source"], "properties": {"source": {"const": "rosbag"}}},
                "then": {"required": ["rosbag_path"]},
                "else": {"required": ["image_dir"]},
            }],
        },
        "lidar": {
            "type": "object",
            "properties": {
                "source": {"enum": list(_VALID_LIDAR_SOURCES)},
            },
            "allOf": [{
                "if": {"required": ["source"], "properties": {"source": {"const": "rosbag"}}},
                "then": {"required": ["rosbag_path"]},
                "else": {"required": ["pcd_dir"]},
            }],
        },
        "extrinsic": {
            "type": "object",
            "required": list(_REQUIRED_EXTRINSIC_KEYS),
            "properties": {
                "rotation_format": {"enum": list(_VALID_ROTATION_FORMATS)},
                "parent": {"enum": list(_VALID_EXTRINSIC_ROLES)},
                "child": {"enum": list(_VALID_EXTRINSIC_ROLES)},
            },
        },
    },
}

_MISSING_REQUIRED_PROPERTY_RE = re.compile(r"^'(.+?)' is a required property$")


def _format_schema_error(error: jsonschema.exceptions.ValidationError) -> str:
    """
    Convert one jsonschema ValidationError into the same human-readable
    "problem" string format the original hand-written checks produced
    (e.g. "missing top-level key 'camera'", "missing 'camera.intrinsics'",
    "'extrinsic.rotation_format' is 'rpy_degrees', must be one of (...)"),
    so switching validation engines doesn't change what a --config user
    sees on the command line.
    """
    path = [str(p) for p in error.path]

    if error.validator == "required":
        m = _MISSING_REQUIRED_PROPERTY_RE.match(error.message)
        missing_key = m.group(1) if m else "?"
        if not path:
            return f"missing top-level key '{missing_key}'"
        return f"missing '{'.'.join([*path, missing_key])}'"

    if error.validator == "enum":
        dotted = ".".join(path) or "<root>"
        return f"'{dotted}' is {error.instance!r}, must be one of {tuple(error.validator_value)}"

    if error.validator == "type" and error.validator_value == "object":
        dotted = ".".join(path)
        return f"'{dotted}' must be a mapping" if dotted else (
            f"did not parse into a YAML mapping (got {type(error.instance).__name__} instead)"
        )

    # Fallback for any other jsonschema validator keyword (e.g. a future
    # schema addition this function hasn't been taught a custom phrasing
    # for yet) -- still actionable, just not hand-tuned wording.
    dotted = ".".join(path) or "<root>"
    return f"'{dotted}': {error.message}"


def _validate_config_schema(cfg, config_path: str) -> None:
    """
    Check a parsed --config YAML against _CONFIG_JSON_SCHEMA, collecting
    every problem found in a single pass (missing keys, wrong types,
    invalid enum values) instead of stopping at the first one. Raises
    ConfigSchemaError with a bullet list naming every problem and a
    pointer back to the schema, so a first-time --config user isn't left
    staring at a bare "'camera'" KeyError with no idea what it means or
    where to look.

    This intentionally checks presence/shape only (required keys exist,
    are the right container type, enum-like fields have an allowed value)
    -- it does not validate values deeply (e.g. it doesn't check that
    image_dir actually exists on disk, or that intrinsics are physically
    plausible). Those are left to the loaders themselves, which already
    raise their own specific, readable errors when they run.
    """
    if not isinstance(cfg, dict):
        raise ConfigSchemaError(
            f"'{config_path}' did not parse into a YAML mapping (got "
            f"{type(cfg).__name__} instead). Expected top-level keys: "
            f"{', '.join(_REQUIRED_TOP_LEVEL_KEYS)}. {_SCHEMA_POINTER}"
        )

    validator = jsonschema.Draft7Validator(_CONFIG_JSON_SCHEMA)
    problems = [_format_schema_error(e) for e in validator.iter_errors(cfg)]

    if problems:
        bullet_list = "\n".join(f"  - {p}" for p in problems)
        raise ConfigSchemaError(
            f"'{config_path}' does not match the expected schema:\n"
            f"{bullet_list}\n{_SCHEMA_POINTER}"
        )


def validate_config_only(config_path: str) -> None:
    """
    Parse and schema-check a --config YAML WITHOUT building any loaders
    -- doesn't touch image_dir/pcd_dir/rosbag files on disk at all, just
    the YAML structure itself (required keys present, right container
    types, enum fields valid). This is what app.cli's --validate-config
    flag calls: a fast "is this config well-formed" check for CI or a
    pre-commit hook, without paying for image/point-cloud I/O or running
    the actual evaluation pipeline.

    Raises FileNotFoundError if config_path doesn't exist, yaml.YAMLError
    if it isn't valid YAML, or ConfigSchemaError (see
    _validate_config_schema) if it's valid YAML but doesn't match the
    expected schema. Returns normally (no return value) if valid.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _validate_config_schema(cfg, config_path)


def load_dataset_from_config(config_path: str) -> tuple[EvaluationDataset, dict, list[str]]:
    """Parse a YAML config per the schema in this module's docstring and
    build a synced EvaluationDataset. Returns (dataset, evaluation_config,
    warnings) -- warnings collects everything the loaders themselves
    surfaced (missing sensor specs, non-numeric filenames, sync drops,
    extrinsic sanity issues), so the caller can fold them into the report
    rather than losing them.

    Raises ConfigSchemaError (with a full list of problems and a pointer to
    the schema) if required keys are missing or malformed, before any
    loader runs -- so a config typo fails fast with an actionable message
    rather than surfacing as a bare KeyError from wherever it happened to
    be first dereferenced."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _validate_config_schema(cfg, config_path)

    warnings: list[str] = []

    cam_cfg = cfg["camera"]
    intrinsics = CameraIntrinsics(**cam_cfg["intrinsics"])
    dist_cfg = cam_cfg.get("distortion", {"model": "none"})
    distortion = CameraDistortion(model=dist_cfg.get("model", "none"), coeffs=dist_cfg.get("coeffs", {}))
    cam_source = cam_cfg.get("source", "image_dir")
    if cam_source == "rosbag":
        cam_result = load_camera_from_rosbag(
            cam_cfg["rosbag_path"], width=cam_cfg["width"], height=cam_cfg["height"],
            model=cam_cfg.get("model", "pinhole"), intrinsics=intrinsics, distortion=distortion,
            topic=cam_cfg.get("topic"),
            edge_localization_floor_px=cam_cfg.get("edge_localization_floor_px"),
        )
    else:
        cam_result = load_camera_from_image_dir(
            cam_cfg["image_dir"], width=cam_cfg["width"], height=cam_cfg["height"],
            model=cam_cfg.get("model", "pinhole"), intrinsics=intrinsics, distortion=distortion,
            timestamp_source=cam_cfg.get("timestamp_source", "filename"),
            edge_localization_floor_px=cam_cfg.get("edge_localization_floor_px"),
        )
    warnings.extend(f"[camera] {w}" for w in cam_result.warnings)

    lidar_cfg = cfg["lidar"]
    sensor_spec = LidarSensorSpec(**lidar_cfg.get("sensor_spec", {}))
    lidar_source = lidar_cfg.get("source", "pcd_dir")
    if lidar_source == "rosbag":
        lidar_result = load_lidar_from_rosbag(
            lidar_cfg["rosbag_path"], sensor_spec, topic=lidar_cfg.get("topic"),
        )
    else:
        lidar_result = load_lidar_from_pcd_dir(lidar_cfg["pcd_dir"], sensor_spec)
    warnings.extend(f"[lidar] {w}" for w in lidar_result.warnings)

    # STEP 1 -- Input Validation. Runs BEFORE extrinsic/sync/any evaluation
    # metric: a broken input (non-monotonic timestamps, NaN points, an
    # empty point cloud, a degenerate intrinsics matrix, ...) must surface
    # as INPUT INVALID with concrete reasons, never as "Calibration BAD".
    # raise_on_invalid=True short-circuits the pipeline here via
    # InputValidationError, which main() catches and reports distinctly
    # from a calibration-quality failure.
    validation_report = validate_input(
        cam_result.camera, cam_result.frames,
        lidar_result.lidar, lidar_result.frames,
        raise_on_invalid=True,
    )
    if validation_report.status != ValidationStatus.VALID:
        warnings.extend(f"[input_validation] {r}" for r in validation_report.reasons())

    ext_cfg = cfg["extrinsic"]
    ext_raw = ExtrinsicRaw(
        parent=ext_cfg["parent"], child=ext_cfg["child"],
        translation=tuple(ext_cfg.get("translation", (0.0, 0.0, 0.0))),
        rotation=ext_cfg["rotation"], rotation_format=ext_cfg["rotation_format"],
        unit=ext_cfg.get("unit", "m"),
    )
    extrinsic = load_extrinsic(ext_raw)
    sanity = verify_extrinsic(extrinsic)
    if not sanity.all_passed:
        for item in sanity.failed_items():
            warnings.append(f"[extrinsic] sanity check '{item.name}' failed: {item.detail}")

    eval_cfg = cfg.get("evaluation", {}) or {}
    sync_config = SyncConfig(max_time_diff_ms=eval_cfg.get("sync_max_time_diff_ms", 50.0))

    dataset = build_dataset(
        cam_result.camera, cam_result.frames, lidar_result.lidar, lidar_result.frames,
        extrinsic, sync_config,
    )
    dataset.input_validation = validation_report.to_dict()
    # NOTE: dataset.warnings (sync-related: majority-drop, offset) is
    # intentionally NOT folded into the `warnings` list returned here --
    # report/builder.py's build_report() already reads dataset.warnings
    # directly from the EvaluationDataset object it's given. Adding them
    # here too would double them up in the final report (they'd arrive via
    # both this function's return value -> main()'s extra_warnings, AND
    # via build_report's own `list(dataset.warnings)`).

    return dataset, eval_cfg, warnings


# ---------------------------------------------------------------------------
# Built-in demo dataset (no external data required)
# ---------------------------------------------------------------------------

def _build_demo_dataset(scenario: str, num_frames: int = 40) -> EvaluationDataset:
    """
    Self-contained synthetic scene for --demo mode: a depth step (two flat
    surfaces at different distances) built by back-projecting from a dense
    pixel grid, so it lines up exactly with a drawn image edge -- see
    evaluation_metric_spec.md's test methodology notes for why this
    construction (rather than a naive world-space grid) avoids perspective
    row-shift artifacts.

    scenario:
      "good"  -> T_CL is a perfect match for every frame (illustrates a
                 healthy calibration)
      "drift" -> the second half of frames have their true alignment
                 drift away from the evaluated T_CL (illustrates what a
                 degrading calibration looks like in the report)
    """
    import cv2
    from input.camera import CameraModel, CameraSource, CameraFrame
    from input.lidar import LidarModel, LidarSource, LidarFrame
    from input.extrinsic import ExtrinsicModel

    width, height = 640, 480
    fx = fy = 500.0
    cx, cy = 320.0, 240.0
    z_near, z_far = 5.0, 10.0

    camera = CameraModel(
        width=width, height=height, model="pinhole",
        intrinsics=CameraIntrinsics(fx, fy, cx, cy),
        distortion=CameraDistortion(model="none"),
        source=CameraSource(kind="image_dir", path="<demo>"),
    )

    image = np.zeros((height, width), dtype=np.uint8)
    image[:, int(cx):] = 255
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    u_vals = np.linspace(0, width - 1, 220)
    v_vals = np.linspace(0, height - 1, 140)
    uu, vv = np.meshgrid(u_vals, v_vals)
    uu, vv = uu.ravel(), vv.ravel()
    zz = np.where(uu < cx, z_near, z_far)
    xx = (uu - cx) * zz / fx
    yy = (vv - cy) * zz / fy
    base_points = np.stack([xx, yy, zz], axis=1)

    lidar_spec = LidarSensorSpec(horizontal_resolution_deg=0.05, vertical_resolution_deg=0.05,
                                  range_accuracy_m=0.02, min_range_m=0.1, max_range_m=200.0)
    lidar = LidarModel(source=LidarSource(kind="pcd_dir", path="<demo>"), sensor_spec=lidar_spec)

    ext_raw = ExtrinsicRaw(parent="lidar", child="camera", translation=(0, 0, 0),
                            rotation=(0, 0, 0), rotation_format="rpy_deg")
    extrinsic = ExtrinsicModel(T_CL=np.eye(4), parent="lidar", child="camera", raw=ext_raw)

    if scenario == "drift":
        drifts = [0.0] * (num_frames // 2) + [0.08] * (num_frames - num_frames // 2)
    elif scenario == "good":
        drifts = [0.0] * num_frames
    else:
        raise ValueError(f"Unknown demo scenario: {scenario!r} (expected 'good' or 'drift')")

    frames = []
    for i, drift in enumerate(drifts):
        pts = base_points.copy()
        pts[:, 0] -= drift
        frames.append(SyncedFrame(
            index=i, timestamp=float(i),
            camera_frame=CameraFrame(timestamp=float(i), image=image),
            lidar_frame=LidarFrame(timestamp=float(i), points=pts),
            time_diff_ms=0.0,
        ))

    return EvaluationDataset(
        camera=camera, lidar=lidar, extrinsic=extrinsic,
        sync_config=SyncConfig(), frames=frames,
        # Synthetic frames are perfectly time-aligned by construction (each
        # SyncedFrame above is built directly, bypassing build_dataset's
        # real sync pass) -- fill in SyncStats to match so the console/
        # report Synchronization section still shows something meaningful
        # (GOOD, 0ms offset) instead of silently omitting the section.
        sync_stats=SyncStats(
            num_camera_frames=len(frames), num_lidar_frames=len(frames),
            num_matched=len(frames), num_camera_dropped=0, num_lidar_dropped=0,
            mean_time_diff_ms=0.0, max_time_diff_ms=0.0,
            estimated_offset_ms=0.0, offset_std_ms=0.0, drop_ratio=0.0,
            classification="GOOD",
        ),
    )


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def _parse_weights(spec: Optional[str]) -> Optional[dict]:
    """Parse '--weights geometry=0.5,generalization=0.25,stability=0.25'."""
    if not spec:
        return None
    weights = {}
    for part in spec.split(","):
        key, _, value = part.partition("=")
        weights[key.strip()] = float(value)
    return weights


def _parse_vec3(spec: Optional[str], flag_name: str) -> Optional[np.ndarray]:
    """Parse '--deskew-linear-velocity 3.0,0.0,0.5' into a (3,) array.
    flag_name is used only to make the error message point at the actual
    flag that was malformed, since both --deskew-linear-velocity and
    --deskew-angular-velocity share this parser."""
    if not spec:
        return None
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{flag_name} expects 3 comma-separated numbers (x,y,z), got {spec!r}")
    try:
        return np.array([float(p) for p in parts], dtype=np.float64)
    except ValueError as e:
        raise ValueError(f"{flag_name} expects 3 numbers (x,y,z), got {spec!r}: {e}") from e


def run_pipeline(
    dataset: EvaluationDataset,
    output_dir: str,
    n_blocks: int = 4,
    min_frames_per_block: int = 30,
    min_frames_m4: int = 30,
    depth_jump_threshold_m: float = 0.3,
    edge_radius_px: float = 3.0,
    frame_index: Optional[int] = None,
    weights: Optional[dict] = None,
    no_visuals: bool = False,
    json_only: bool = False,
    advanced: bool = False,
    extra_warnings: Optional[list[str]] = None,
    interactive_max_points: Optional[int] = None,
    sequence_gif: bool = False,
    sequence_max_frames: int = 16,
    deskew_linear_velocity_mps: Optional[np.ndarray] = None,
    deskew_angular_velocity_rps: Optional[np.ndarray] = None,
    deskew_scan_period_s: float = 0.1,
    deskew_reference_time_s: Optional[float] = None,
    deskew_clockwise: bool = False,
    dynamic_filter: bool = False,
    dynamic_filter_window: int = 5,
    dynamic_filter_range_std_threshold_m: float = 0.3,
    dynamic_filter_min_frames_present: int = 3,
) -> dict:
    """Run M0/M2/M3/M4 + quality scoring + visuals + report writing for a
    built EvaluationDataset. Returns the report dict (already written to
    disk in output_dir -- report.json always, report.html unless
    json_only=True).

    interactive_max_points: overrides the interactive 3D viewer's default
    point budget (see visualization.interactive_viewer). None keeps that
    module's own default.

    sequence_gif: if True (and visuals aren't skipped), also renders an
    animated GIF of the M2 overlay sampled across up to
    sequence_max_frames frames in the sequence (see
    visualization.sequence) -- opt-in and comparatively expensive, since
    each sampled frame re-runs the full M2 pipeline.

    advanced: if True, also runs the Phase-5 diagnostics (Plane Consistency,
    Perturbation Sensitivity, Temporal Drift) and attaches them to the
    report's 'advanced' section. These never affect quality_score -- they're
    supplementary, not part of the MVP scored set -- and cost noticeably
    more time (perturbation alone re-runs M2 roughly 24 times), so they're
    opt-in rather than the default.

    deskew_linear_velocity_mps / deskew_angular_velocity_rps: STEP5 --
    if either is given (not None), the headline frame's LiDAR points are
    ALSO run through motion.deskew (constant-velocity model) purely as a
    diagnostic: a 'motion_deskew' report section + before/after
    visualization are added showing the correction this velocity would
    produce. This is DIAGNOSTIC ONLY -- M0/M2/M3/M4 always score the
    original, undeskewed points; deskewing here never changes
    quality_score. Needs an external velocity source (IMU, wheel
    odometry, ...) this tool has no way to measure on its own -- omitted
    (None, the default) unless the caller explicitly supplies one.
    Whichever of the two is left None defaults to zero (e.g. giving only
    linear velocity assumes no rotation).

    dynamic_filter: STEP8 -- if True, classifies the headline frame's
    LiDAR edge points as static/dynamic/unknown via multi-frame motion
    consistency (evaluation.dynamic_filter.classify_points_by_motion_
    consistency, using a window of dynamic_filter_window frames on each
    side of the headline frame from this dataset) and adds a
    'dynamic_filter' report section comparing M2's "overall" (unfiltered,
    today's default) result against a "static only" (dynamic points
    removed) result, plus a dynamic_contamination_ratio. Like deskewing,
    this is DIAGNOSTIC ONLY -- M0/M2/M3/M4's own scored results always
    use the unfiltered point set; dynamic_filter never changes
    quality_score. IMPORTANT: the motion-consistency method assumes the
    platform is approximately STATIONARY across the frame window (see
    evaluation/dynamic_filter.py's module docstring) -- on a moving
    platform this will misclassify the whole static scene as "dynamic"
    and the comparison will not be meaningful. Off by default since it
    needs that assumption to hold and costs extra time (builds a range
    image per frame in the window).

    json_only: if True, skip both visual generation (overlay/trajectory/
    histogram PNGs, same as no_visuals) AND HTML report assembly/writing
    entirely -- only report.json is produced. For CI gates (--fail-on-bad/
    --fail-on-partial) that only ever read the JSON, this avoids spending
    time building and writing an HTML file nobody opens. Implies
    no_visuals (visuals would just be discarded unused if HTML isn't
    written), so passing no_visuals=False alongside json_only=True has no
    effect -- visuals are skipped either way."""
    if not dataset.frames:
        raise ValueError("Dataset has no synced frames; nothing to evaluate.")

    skip_visuals = no_visuals or json_only

    lidar_spec = dataset.lidar.sensor_spec
    edge_kwargs = {"depth_jump_threshold_m": depth_jump_threshold_m, "edge_radius_px": edge_radius_px}

    headline_idx = frame_index if frame_index is not None else len(dataset.frames) // 2
    headline_idx = max(0, min(headline_idx, len(dataset.frames) - 1))
    sf = dataset.frames[headline_idx]
    image = sf.camera_frame.load()
    dataset.camera.verify_image_shape(image)
    points = sf.lidar_frame.load()

    m0 = run_sanity_gate(points, T_CL=dataset.extrinsic.T_CL, camera=dataset.camera, lidar_spec=lidar_spec)
    m2 = evaluate_edge_alignment(image=image, points_lidar=points, T_CL=dataset.extrinsic.T_CL,
                                  camera=dataset.camera, lidar_spec=lidar_spec, **edge_kwargs)
    m3 = evaluate_holdout_consistency(dataset, lidar_spec=lidar_spec, n_blocks=n_blocks,
                                       min_frames_per_block=min_frames_per_block,
                                       edge_alignment_kwargs=edge_kwargs)
    m4 = evaluate_multiframe_consistency(dataset, lidar_spec=lidar_spec, min_frames=min_frames_m4,
                                          edge_alignment_kwargs=edge_kwargs)
    quality = compute_quality_score(m2, m3, m4, weights=weights)

    plane_result = perturbation_result = temporal_drift_result = None
    if advanced:
        plane_result = evaluate_plane_consistency(image=image, points_lidar=points, T_CL=dataset.extrinsic.T_CL,
                                                    camera=dataset.camera, lidar_spec=lidar_spec)
        perturbation_result = evaluate_perturbation_sensitivity(
            image=image, points_lidar=points, T_CL=dataset.extrinsic.T_CL, camera=dataset.camera,
            lidar_spec=lidar_spec, edge_alignment_kwargs=edge_kwargs,
        )
        temporal_drift_result = evaluate_temporal_drift(m4)

    deskew_compare: Optional[dict] = None
    deskew_result = None
    if deskew_linear_velocity_mps is not None or deskew_angular_velocity_rps is not None:
        lin_v = deskew_linear_velocity_mps if deskew_linear_velocity_mps is not None else np.zeros(3)
        ang_v = deskew_angular_velocity_rps if deskew_angular_velocity_rps is not None else np.zeros(3)
        ref_t = deskew_reference_time_s if deskew_reference_time_s is not None else deskew_scan_period_s / 2.0
        deskew_result = deskew_points_constant_velocity(
            points, scan_period_s=deskew_scan_period_s,
            linear_velocity_mps=lin_v, angular_velocity_rps=ang_v,
            reference_time_s=ref_t, clockwise=deskew_clockwise,
        )
        deskew_compare = compare_before_after(points, deskew_result)

    dynamic_filter_comparison = None
    dynamic_filter_motion_result = None
    if dynamic_filter:
        lo = max(0, headline_idx - dynamic_filter_window)
        hi = min(len(dataset.frames), headline_idx + dynamic_filter_window + 1)
        window_points = [dataset.frames[i].lidar_frame.load() for i in range(lo, hi) if i != headline_idx]
        dynamic_filter_motion_result = classify_points_by_motion_consistency(
            window_points,
            num_rings=lidar_spec.channels or 32,
            vertical_fov_deg=lidar_spec.vertical_fov_deg,
            range_std_threshold_m=dynamic_filter_range_std_threshold_m,
            min_frames_present=min(dynamic_filter_min_frames_present, max(1, len(window_points))),
        )
        d_mask = dynamic_point_mask(points, dynamic_filter_motion_result, vertical_fov_deg=lidar_spec.vertical_fov_deg)
        dynamic_filter_comparison = compare_with_without_dynamic_filtering(
            image=image, points_lidar=points, T_CL=dataset.extrinsic.T_CL,
            camera=dataset.camera, lidar_spec=lidar_spec, dynamic_mask=d_mask, **edge_kwargs,
        )

    visuals = {}
    if not skip_visuals:
        proj_overlay = render_projection_overlay_from_frame(image, points, dataset.extrinsic.T_CL, dataset.camera)
        if proj_overlay is not None and proj_overlay.num_valid_points > 0:
            visuals["projection_overlay_png"] = encode_png(proj_overlay.image)
        range_image_png = render_range_image_from_points(
            points, num_rings=lidar_spec.channels or 32,
            vertical_fov_deg=lidar_spec.vertical_fov_deg,
            depth_jump_threshold_m=depth_jump_threshold_m,
        )
        if range_image_png is not None:
            visuals["range_image_png"] = range_image_png
        floor_inputs = resolve_floor_inputs(
            fx_px=dataset.camera.intrinsics.fx, T_CL=dataset.extrinsic.T_CL, lidar_spec=lidar_spec,
            edge_localization_floor_px=dataset.camera.edge_localization_floor_px,
        )
        uncertainty_png = render_uncertainty_plot_from_result(m2, floor_inputs)
        if uncertainty_png is not None:
            visuals["uncertainty_plot_png"] = uncertainty_png
        spatial_png = render_spatial_analysis_from_result(m2, dataset.camera.width, dataset.camera.height)
        if spatial_png is not None:
            visuals["spatial_analysis_png"] = spatial_png
        if perturbation_result is not None:
            sensitivity_png = render_sensitivity_from_result(perturbation_result)
            if sensitivity_png is not None:
                visuals["sensitivity_png"] = sensitivity_png
        if deskew_result is not None:
            deskew_png = render_deskew_comparison_png(points, deskew_result)
            if deskew_png is not None:
                visuals["deskew_comparison_png"] = deskew_png
        if dynamic_filter_motion_result is not None:
            dynamic_overlay = render_dynamic_filter_overlay_from_frame(
                image, points, dataset.extrinsic.T_CL, dataset.camera,
                motion_result=dynamic_filter_motion_result,
                vertical_fov_deg=lidar_spec.vertical_fov_deg,
            )
            if dynamic_overlay is not None and dynamic_overlay.num_valid_points > 0:
                visuals["dynamic_filter_overlay_png"] = encode_png(dynamic_overlay.image)
        overlay_img = render_overlay_from_result(image, m2)
        if overlay_img is not None:
            visuals["overlay_png"] = encode_png(overlay_img)
        if m2.edge_point_errors_px is not None:
            hist_png = render_error_histogram_png(m2.edge_point_errors_px, m2.floor_px)
            if hist_png is not None:
                visuals["histogram_png"] = hist_png
        traj_png = render_m4_trajectory_png(m4)
        if traj_png is not None:
            visuals["trajectory_png"] = traj_png
        colorized_png = render_colorized_pointcloud_from_frame(
            image, points, dataset.extrinsic.T_CL, dataset.camera,
        )
        if colorized_png is not None:
            visuals["colorized_pointcloud_png"] = colorized_png
        heatmap_img = render_error_heatmap_from_result(image, m2)
        if heatmap_img is not None:
            visuals["error_heatmap_png"] = encode_png(heatmap_img)
        frustum_png = render_camera_frustum_from_dataset(dataset, frame_index=headline_idx)
        if frustum_png is not None:
            visuals["camera_frustum_png"] = frustum_png
        bev_png = render_bev_dual_panel_from_result(image, points, dataset.extrinsic.T_CL, dataset.camera, m2,
                                                      edge_kwargs=edge_kwargs)
        if bev_png is not None:
            visuals["bev_dual_panel_png"] = bev_png
        interactive_kwargs = {} if interactive_max_points is None else {"colorize_max_points": interactive_max_points}
        visuals["interactive_scene"] = build_interactive_scene_from_dataset(
            dataset, frame_index=headline_idx, **interactive_kwargs,
        )
        if sequence_gif:
            gif_bytes = render_sequence_gif(
                dataset, lidar_spec, edge_kwargs=edge_kwargs, max_frames=sequence_max_frames,
            )
            if gif_bytes is not None:
                visuals["sequence_gif"] = gif_bytes

    report = build_report(
        dataset, m2, m3, m4, quality, m0_result=m0.to_dict(),
        n_blocks=n_blocks, min_frames_m4=min_frames_m4, extra_warnings=extra_warnings,
        plane_result=plane_result, perturbation_result=perturbation_result,
        temporal_drift_result=temporal_drift_result, deskew_compare=deskew_compare,
        dynamic_filter_comparison=dynamic_filter_comparison,
    )

    os.makedirs(output_dir, exist_ok=True)
    write_json_report(report, os.path.join(output_dir, "report.json"))
    if not json_only:
        write_html_report(report, os.path.join(output_dir, "report.html"), visuals=visuals)

    return report


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_console_summary(report: dict, output_dir: str, json_only: bool = False) -> None:
    q = report["quality_score"]
    m0 = report["m0_sanity_gate"]
    width = 58
    sep = "-" * width

    def row(label: str, value: str) -> str:
        return f" {label:<24}: {value}"

    print(sep)
    print(" Cam-LiDAR Calibration Quality")
    print(sep)
    input_validation = report.get("input_validation")
    if input_validation is not None and input_validation["status"] != "INPUT_VALID":
        print(row("Input Validation", input_validation["status"]))
        for reason in input_validation["reasons"]:
            print(f"   - {reason}")
        print(sep)
    sync = report.get("synchronization")
    if sync is not None:
        print(" Synchronization")
        print(row("  Matched frames", f'{sync["num_matched"]} / {sync["num_camera_frames"]}'))
        mean_dt = sync["estimated_offset_ms"]
        print(row("  Mean \u0394t", f'{mean_dt:+.1f} ms' if mean_dt is not None else "N/A"))
        offset_std = sync["offset_std_ms"]
        print(row("  Offset std", f'{offset_std:.1f} ms' if offset_std is not None else "N/A"))
        drop_ratio = sync["drop_ratio"]
        print(row("  Drop ratio", f'{drop_ratio:.1%}' if drop_ratio is not None else "N/A"))
        print(row("  Status", sync["classification"]))
        print(sep)
    if m0 is not None:
        print(row("M0 Sanity Gate", "PASS" if m0["passed"] else "FAIL"))
    for cat in q["categories"]:
        label = f'{cat["name"].capitalize()} ({cat["metric"]})'
        score_str = f'{cat["score"]:.1f} / 100' if cat["score"] is not None else "N/A"
        print(row(label, f'{score_str:<14} [{cat["classification"]}]'))
    print(sep)
    overall_str = f'{q["overall_score"]:.1f} / 100' if q["overall_score"] is not None else "N/A"
    num_valid = q["num_valid_categories"]
    num_total = len(q["categories"])
    partial_suffix = f" ({num_valid}/{num_total} categories)" if num_valid < num_total else ""
    print(row("OVERALL QUALITY", f'{overall_str:<14} [{q["overall_classification"]}]{partial_suffix}'))
    print(sep)

    advanced = report.get("advanced")
    if advanced:
        plane = advanced.get("plane_consistency")
        pert = advanced.get("perturbation")
        drift = advanced.get("temporal_drift")
        print(" Advanced Diagnostics")
        if plane:
            print(row("  Plane Consistency", plane["classification"]))
        if pert:
            print(row("  Perturbation", pert["classification"]))
        if drift:
            print(row("  Temporal Drift", drift["classification"]))
        print(sep)

    n_warnings = len(report.get("warnings", []))
    if n_warnings:
        print(f" {n_warnings} warning(s) -- see report for details.")
    if json_only:
        print(f" Report written to: {os.path.join(output_dir, 'report.json')}")
    else:
        print(f" Report written to: {os.path.join(output_dir, 'report.html')}")
        print(f"                    {os.path.join(output_dir, 'report.json')}")
    print(sep)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

_CONFIG_SCHEMA_EPILOG = """\
--config YAML schema (mirrors the Input Loader Spec in evaluation_metric_spec.md):

  camera:
    source: image_dir                 # optional, default: image_dir. Or: rosbag
    image_dir: path/to/images         # required if source == image_dir
    # rosbag_path: path/to/bag        # required if source == rosbag
    # topic: /camera/image_raw        # optional if bag has exactly one image topic
    width: 1920
    height: 1080
    model: pinhole            # or fisheye
    intrinsics: {fx: ..., fy: ..., cx: ..., cy: ...}
    distortion: {model: plumb_bob, coeffs: {k1: ..., k2: ..., p1: ..., p2: ...}}
    edge_localization_floor_px: 0.5   # optional

  lidar:
    source: pcd_dir                   # optional, default: pcd_dir. Or: rosbag
    pcd_dir: path/to/pointclouds      # required if source == pcd_dir
    # rosbag_path: path/to/bag        # required if source == rosbag
    # topic: /lidar/points            # optional if bag has exactly one PointCloud2 topic
    sensor_spec:
      horizontal_resolution_deg: 0.2   # optional (see fallback rules in spec)
      vertical_resolution_deg: 0.2     # optional
      channels: 32                    # optional, used if vertical_resolution_deg absent
      vertical_fov_deg: 40.0          # optional, used with channels
      range_accuracy_m: 0.02          # optional
      min_range_m: 0.1
      max_range_m: 200.0

  extrinsic:
    parent: lidar                     # or camera
    child: camera                     # or lidar
    translation: [0.1, -0.05, 0.2]
    rotation: [0.0, 0.0, 0.0]         # meaning depends on rotation_format
    rotation_format: rpy_deg          # rpy_deg | rpy_rad | quaternion | matrix3x3 | matrix4x4
    unit: m                           # m | cm | mm

  evaluation:                         # all optional, shown with defaults
    sync_max_time_diff_ms: 50.0
    n_blocks: 4
    min_frames_per_block: 30
    min_frames_m4: 30
    depth_jump_threshold_m: 0.3
    edge_radius_px: 3.0
    frame_index: null                 # which frame M2's headline number comes from;
                                       # null -> the temporally-middle frame
    weights: null                     # e.g. {geometry: 0.5, generalization: 0.25, stability: 0.25}

rosbag source: requires `pip install "cam-lidar-eval[rosbag]"` (a pure-Python
rosbag1/rosbag2 reader -- no ROS/rclpy installation needed). Live `ros_topic`
sources are NOT supported (that needs an active ROS2 middleware connection).

Only camera/lidar/extrinsic's non-comment keys above are required; everything
marked "optional" (including the whole `evaluation:` block) may be omitted.
"""


def _get_version() -> str:
    """Read the installed package version (from pyproject.toml's
    [project].version, via installed package metadata) so --version can't
    drift out of sync with a hardcoded string here. Falls back to
    "unknown" when running from a source checkout that was never
    `pip install`-ed (importlib.metadata needs installed package metadata,
    not just the source files, to resolve a version)."""
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version("cam-lidar-eval")
        except PackageNotFoundError:
            return "unknown (not installed -- run `pip install -e .`)"
    except ImportError:  # pragma: no cover -- importlib.metadata is stdlib since 3.8
        return "unknown"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cam-lidar-eval",
        description="GT-free quality evaluation for an EXISTING camera-LiDAR extrinsic calibration.",
        epilog=_CONFIG_SCHEMA_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_get_version()}")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=str, help="Path to a YAML config (see --help's epilog below for the schema).")
    source.add_argument("--demo", action="store_true", help="Run against a built-in synthetic scene (no data required).")

    parser.add_argument("--validate-config", action="store_true",
                         help="Only check --config's YAML against the expected schema and exit -- no data "
                              "loading, no evaluation. Prints 'OK' and exits 0 if valid; prints every problem "
                              "found and exits 1 otherwise. Requires --config (not --demo).")

    parser.add_argument("--scenario", choices=["good", "drift"], default="good",
                         help="Demo scenario (only used with --demo). Default: good.")
    parser.add_argument("--demo-frames", type=int, default=40, help="Number of frames in the demo dataset.")

    parser.add_argument("--output-dir", type=str, default="./cam_lidar_eval_report",
                         help="Directory to write report.json / report.html into.")

    parser.add_argument("--n-blocks", type=int, default=4, help="M3: number of contiguous time blocks.")
    parser.add_argument("--min-frames-per-block", type=int, default=30, help="M3: minimum frames per block.")
    parser.add_argument("--min-frames-m4", type=int, default=30, help="M4: minimum total frames required.")
    parser.add_argument("--depth-jump-threshold-m", type=float, default=0.3,
                         help="M2: depth discontinuity threshold (meters) for LiDAR edge-point extraction.")
    parser.add_argument("--edge-radius-px", type=float, default=3.0,
                         help="M2: pixel radius for neighbor-based edge-point detection.")
    parser.add_argument("--frame-index", type=int, default=None,
                         help="Which synced frame M2's headline result comes from. Default: the temporally-middle frame.")
    parser.add_argument("--weights", type=str, default=None,
                         help="Override category weights, e.g. 'geometry=0.5,generalization=0.25,stability=0.25'.")
    parser.add_argument("--no-visuals", action="store_true", help="Skip generating overlay/trajectory/histogram images.")
    parser.add_argument("--json-only", action="store_true",
                         help="Skip generating report.html entirely (and its visuals, same as --no-visuals) -- "
                              "only report.json is written. For CI gates (--fail-on-bad/--fail-on-partial) that "
                              "only ever read the JSON, this saves the time spent building an HTML file nobody opens.")
    parser.add_argument("--advanced", action="store_true",
                         help="Also run Phase-5 advanced diagnostics (Plane Consistency, Perturbation "
                              "Sensitivity, Temporal Drift). Slower -- perturbation alone re-runs M2 "
                              "roughly 24 times. Never affects the quality score.")
    parser.add_argument("--fail-on-bad", action="store_true",
                         help="Exit with a non-zero status if overall quality is BAD or FAIL (useful in CI).")
    parser.add_argument("--fail-on-partial", action="store_true",
                         help="Exit with a non-zero status if any category (M2/M3/M4) FAILed outright and "
                              "was excluded from Overall Quality -- even if the surviving categories scored "
                              "well enough that overall quality itself is GOOD/WARNING. Use this when a "
                              "calibration must be evaluated on all three axes to be trusted, not just the "
                              "ones that happened to be measurable (useful in CI).")

    parser.add_argument("--compare-to", type=str, default=None,
                         help="Path to a previous run's report.json. If given, the console summary (or "
                              "--format github-comment output) includes a delta vs this run for the overall "
                              "score and each category, and --fail-on-regression becomes available.")
    parser.add_argument("--fail-on-regression", action="store_true",
                         help="Exit with status 4 if --compare-to shows a regression: any category (or the "
                              "overall score) got a strictly worse classification, or held the same "
                              "classification but scored lower than the previous run. Requires --compare-to.")
    parser.add_argument("--format", choices=["console", "github-comment"], default="console",
                         help="Output format for the terminal summary. 'github-comment' prints GitHub-flavored "
                              "markdown (score table, emoji status, --compare-to delta if given) instead of "
                              "the plain-text console summary -- pipe it straight into `gh pr comment "
                              "--body-file -` or similar. report.json/report.html are written exactly the "
                              "same either way.")
    parser.add_argument("--interactive-max-points", type=int, default=None,
                         help="Override the interactive 3D viewer's point budget (default: 6000). Higher "
                              "shows more detail but makes report.html larger.")
    parser.add_argument("--sequence-gif", action="store_true",
                         help="Also render an animated GIF of the M2 overlay sampled across the sequence "
                              "(see --sequence-max-frames), embedded in report.html. Opt-in: each sampled "
                              "frame re-runs the full M2 pipeline, so this is slower than the default visuals.")
    parser.add_argument("--sequence-max-frames", type=int, default=16,
                         help="Max frames sampled for --sequence-gif (evenly spaced across the sequence, "
                              "always including the first and last frame). Default: 16.")
    parser.add_argument("--deskew-linear-velocity", type=str, default=None,
                         help="STEP5: platform linear velocity in m/s during the LiDAR scan, as 'vx,vy,vz' "
                              "in the LiDAR body frame. If given (with or without --deskew-angular-velocity), "
                              "runs a diagnostic-only motion deskew on the headline frame and adds a "
                              "'Motion Deskew' report section + before/after visualization. Never affects "
                              "M0/M2/M3/M4 scoring -- this tool has no independent way to measure platform "
                              "velocity, so it's only ever computed from what you explicitly supply here.")
    parser.add_argument("--deskew-angular-velocity", type=str, default=None,
                         help="STEP5: platform angular velocity in rad/s during the LiDAR scan, as 'wx,wy,wz' "
                              "in the LiDAR body frame. See --deskew-linear-velocity.")
    parser.add_argument("--deskew-scan-period-s", type=float, default=0.1,
                         help="STEP5: LiDAR scan period in seconds (e.g. 0.1 for a 10Hz spinning LiDAR). "
                              "Only used if --deskew-linear-velocity or --deskew-angular-velocity is given. "
                              "Default: 0.1.")
    parser.add_argument("--deskew-reference-time-s", type=float, default=None,
                         help="STEP5: which instant within the scan (seconds since scan start, in "
                              "[0, --deskew-scan-period-s]) all points are corrected to -- e.g. the camera's "
                              "own capture instant within the LiDAR scan window, if known. Default: the scan "
                              "midpoint (--deskew-scan-period-s / 2).")
    parser.add_argument("--deskew-clockwise", action="store_true",
                         help="STEP5: treat the LiDAR's azimuth sweep as clockwise instead of the default "
                              "counter-clockwise, when approximating per-point scan time from azimuth angle "
                              "(only matters if your point cloud has no explicit per-point timestamp field, "
                              "which is the common case for this tool's current loaders).")
    parser.add_argument("--dynamic-filter", action="store_true",
                         help="STEP8: classify the headline frame's LiDAR edge points as static/dynamic/"
                              "unknown via multi-frame motion consistency, and add a 'Dynamic Object "
                              "Filtering' report section comparing M2 with vs without those points. "
                              "Diagnostic only -- never changes quality_score. IMPORTANT: assumes the "
                              "platform is approximately stationary across the frame window (see "
                              "evaluation/dynamic_filter.py) -- on a moving platform this will "
                              "misclassify the whole static scene as 'dynamic' and won't be meaningful.")
    parser.add_argument("--dynamic-filter-window", type=int, default=5,
                         help="STEP8: number of frames on EACH SIDE of the headline frame used to build "
                              "the motion-consistency classification (so up to 2x this many frames total, "
                              "excluding the headline frame itself). Default: 5.")
    parser.add_argument("--dynamic-filter-range-std-threshold-m", type=float, default=0.3,
                         help="STEP8: a range-image cell whose range value's standard deviation across "
                              "the frame window exceeds this (in meters) is classified DYNAMIC. Default: 0.3.")
    parser.add_argument("--dynamic-filter-min-frames-present", type=int, default=3,
                         help="STEP8: a cell needs data in at least this many frames of the window to be "
                              "classified static/dynamic at all; otherwise it's UNKNOWN. Default: 3.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.validate_config:
        if args.demo:
            print("error: --validate-config requires --config, not --demo", file=sys.stderr)
            return 1
        try:
            validate_config_only(args.config)
        except (FileNotFoundError, ConfigSchemaError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        except yaml.YAMLError as e:
            print(f"error: failed to parse YAML: {e}", file=sys.stderr)
            return 1
        print(f"OK: {args.config} is a valid config.")
        return 0

    if args.fail_on_regression and not args.compare_to:
        print("error: --fail-on-regression requires --compare-to", file=sys.stderr)
        return 1

    extra_warnings: list[str] = []

    try:
        if args.demo:
            dataset = _build_demo_dataset(args.scenario, num_frames=args.demo_frames)
            # The production default min_frames_per_block=30 (per spec) assumes
            # real datasets with hundreds+ of frames. The demo's small frame
            # count can't satisfy that with n_blocks=4, so unless the person
            # explicitly overrode it, use a demo-appropriate default instead
            # of silently producing an M3 FAIL that has nothing to do with
            # calibration quality.
            if args.min_frames_per_block == parser.get_default("min_frames_per_block"):
                args.min_frames_per_block = max(3, args.demo_frames // args.n_blocks)
            if args.min_frames_m4 == parser.get_default("min_frames_m4"):
                args.min_frames_m4 = min(args.min_frames_m4, args.demo_frames)
        else:
            dataset, eval_cfg, cfg_warnings = load_dataset_from_config(args.config)
            extra_warnings.extend(cfg_warnings)
            # config values fill in defaults for anything not passed on the CLI
            args.n_blocks = eval_cfg.get("n_blocks", args.n_blocks)
            args.min_frames_per_block = eval_cfg.get("min_frames_per_block", args.min_frames_per_block)
            args.min_frames_m4 = eval_cfg.get("min_frames_m4", args.min_frames_m4)
            args.depth_jump_threshold_m = eval_cfg.get("depth_jump_threshold_m", args.depth_jump_threshold_m)
            args.edge_radius_px = eval_cfg.get("edge_radius_px", args.edge_radius_px)
            args.frame_index = eval_cfg.get("frame_index", args.frame_index)
            if args.weights is None and eval_cfg.get("weights"):
                args.weights = ",".join(f"{k}={v}" for k, v in eval_cfg["weights"].items())
    except InputValidationError as e:
        # STEP 1 -- Input Validation failed. Deliberately NOT phrased as
        # "failed to load dataset" or any calibration-quality language:
        # this is a data problem (bad timestamps, NaN points, degenerate
        # intrinsics, ...), and must read as one, not as "Calibration BAD".
        print(f"error: {e}", file=sys.stderr)
        return 5
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"error: failed to load dataset: {e}", file=sys.stderr)
        return 1

    weights = _parse_weights(args.weights)

    try:
        deskew_linear_velocity_mps = _parse_vec3(args.deskew_linear_velocity, "--deskew-linear-velocity")
        deskew_angular_velocity_rps = _parse_vec3(args.deskew_angular_velocity, "--deskew-angular-velocity")
        report = run_pipeline(
            dataset, args.output_dir,
            n_blocks=args.n_blocks, min_frames_per_block=args.min_frames_per_block,
            min_frames_m4=args.min_frames_m4, depth_jump_threshold_m=args.depth_jump_threshold_m,
            edge_radius_px=args.edge_radius_px, frame_index=args.frame_index,
            weights=weights, no_visuals=args.no_visuals, json_only=args.json_only, advanced=args.advanced,
            extra_warnings=extra_warnings,
            interactive_max_points=args.interactive_max_points,
            sequence_gif=args.sequence_gif, sequence_max_frames=args.sequence_max_frames,
            deskew_linear_velocity_mps=deskew_linear_velocity_mps,
            deskew_angular_velocity_rps=deskew_angular_velocity_rps,
            deskew_scan_period_s=args.deskew_scan_period_s,
            deskew_reference_time_s=args.deskew_reference_time_s,
            deskew_clockwise=args.deskew_clockwise,
            dynamic_filter=args.dynamic_filter,
            dynamic_filter_window=args.dynamic_filter_window,
            dynamic_filter_range_std_threshold_m=args.dynamic_filter_range_std_threshold_m,
            dynamic_filter_min_frames_present=args.dynamic_filter_min_frames_present,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    diff = None
    if args.compare_to:
        try:
            with open(args.compare_to, "r", encoding="utf-8") as f:
                old_report = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"error: failed to load --compare-to report: {e}", file=sys.stderr)
            return 1
        diff = compute_report_diff(old_report, report)

    if args.format == "github-comment":
        print(render_github_comment(report, diff=diff))
    else:
        print_console_summary(report, args.output_dir, json_only=args.json_only)
        if diff is not None:
            print(render_diff_console(diff))

    q = report["quality_score"]
    if args.fail_on_bad and q["overall_classification"] in ("BAD", "FAIL"):
        return 2
    if args.fail_on_partial and q["num_valid_categories"] < len(q["categories"]):
        return 3
    if args.fail_on_regression and diff is not None and diff["any_regressed"]:
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
