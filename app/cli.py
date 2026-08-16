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
import os
import sys
from typing import Optional

import numpy as np
import yaml

from input.camera import CameraIntrinsics, CameraDistortion, load_camera_from_image_dir, load_camera_from_rosbag
from input.lidar import LidarSensorSpec, load_lidar_from_pcd_dir, load_lidar_from_rosbag
from input.extrinsic import ExtrinsicRaw, load_extrinsic, verify_extrinsic
from input.dataset import SyncConfig, build_dataset, EvaluationDataset, SyncedFrame

from evaluation.sanity_gate import run_sanity_gate
from evaluation.edge_alignment import evaluate_edge_alignment
from evaluation.holdout_consistency import evaluate_holdout_consistency
from evaluation.multiframe_consistency import evaluate_multiframe_consistency
from evaluation.plane_consistency import evaluate_plane_consistency
from evaluation.perturbation import evaluate_perturbation_sensitivity
from evaluation.temporal_drift import evaluate_temporal_drift

from quality.quality_score import compute_quality_score

from visualization.overlay import render_overlay_from_result, encode_png
from visualization.trajectory import render_m4_trajectory_png
from visualization.histogram import render_error_histogram_png
from visualization.colorized_pointcloud import render_colorized_pointcloud_from_frame
from visualization.error_heatmap import render_error_heatmap_from_result
from visualization.camera_frustum import render_camera_frustum_from_dataset
from visualization.bev_dual_panel import render_bev_dual_panel_from_result
from visualization.interactive_viewer import build_interactive_scene_from_dataset

from report.builder import build_report
from report.json import write_json_report
from report.html import write_html_report


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
_REQUIRED_CAMERA_KEYS = ("width", "height", "intrinsics")
_REQUIRED_CAMERA_INTRINSICS_KEYS = ("fx", "fy", "cx", "cy")
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


def _validate_config_schema(cfg, config_path: str) -> None:
    """
    Check a parsed --config YAML against the schema documented in this
    module's docstring, collecting every problem found in a single pass
    (missing keys, wrong types, invalid enum values) instead of stopping at
    the first one. Raises ConfigSchemaError with a bullet list naming every
    problem and a pointer back to the schema, so a first-time --config user
    isn't left staring at a bare "'camera'" KeyError with no idea what it
    means or where to look.

    This intentionally checks presence/shape only (required keys exist,
    are the right container type, enum-like fields have an allowed value)
    -- it does not validate values deeply (e.g. it doesn't check that
    image_dir actually exists on disk, or that intrinsics are physically
    plausible). Those are left to the loaders themselves, which already
    raise their own specific, readable errors when they run.

    camera.source / lidar.source select which loader is used
    ("image_dir"/"rosbag" and "pcd_dir"/"rosbag" respectively, each
    defaulting to the directory-based source if omitted for backward
    compatibility with configs written before rosbag support existed).
    Each source kind has its own required path key, checked conditionally
    below rather than unconditionally requiring both.
    """
    problems: list[str] = []

    if not isinstance(cfg, dict):
        raise ConfigSchemaError(
            f"'{config_path}' did not parse into a YAML mapping (got "
            f"{type(cfg).__name__} instead). Expected top-level keys: "
            f"{', '.join(_REQUIRED_TOP_LEVEL_KEYS)}. {_SCHEMA_POINTER}"
        )

    for key in _REQUIRED_TOP_LEVEL_KEYS:
        if key not in cfg:
            problems.append(f"missing top-level key '{key}'")

    cam_cfg = cfg.get("camera")
    if cam_cfg is not None:
        if not isinstance(cam_cfg, dict):
            problems.append("'camera' must be a mapping (e.g. `camera: {image_dir: ..., ...}`)")
        else:
            for key in _REQUIRED_CAMERA_KEYS:
                if key not in cam_cfg:
                    problems.append(f"missing 'camera.{key}'")
            cam_source = cam_cfg.get("source", "image_dir")
            if cam_source not in _VALID_CAMERA_SOURCES:
                problems.append(
                    f"'camera.source' is {cam_source!r}, must be one of {_VALID_CAMERA_SOURCES}"
                )
            elif cam_source == "image_dir" and "image_dir" not in cam_cfg:
                problems.append("missing 'camera.image_dir' (required when camera.source is 'image_dir')")
            elif cam_source == "rosbag" and "rosbag_path" not in cam_cfg:
                problems.append("missing 'camera.rosbag_path' (required when camera.source is 'rosbag')")
            if "intrinsics" in cam_cfg:
                intrinsics_cfg = cam_cfg["intrinsics"]
                if not isinstance(intrinsics_cfg, dict):
                    problems.append("'camera.intrinsics' must be a mapping with fx/fy/cx/cy")
                else:
                    for key in _REQUIRED_CAMERA_INTRINSICS_KEYS:
                        if key not in intrinsics_cfg:
                            problems.append(f"missing 'camera.intrinsics.{key}'")

    lidar_cfg = cfg.get("lidar")
    if lidar_cfg is not None:
        if not isinstance(lidar_cfg, dict):
            problems.append("'lidar' must be a mapping (e.g. `lidar: {pcd_dir: ...}`)")
        else:
            lidar_source = lidar_cfg.get("source", "pcd_dir")
            if lidar_source not in _VALID_LIDAR_SOURCES:
                problems.append(
                    f"'lidar.source' is {lidar_source!r}, must be one of {_VALID_LIDAR_SOURCES}"
                )
            elif lidar_source == "pcd_dir" and "pcd_dir" not in lidar_cfg:
                problems.append("missing 'lidar.pcd_dir' (required when lidar.source is 'pcd_dir')")
            elif lidar_source == "rosbag" and "rosbag_path" not in lidar_cfg:
                problems.append("missing 'lidar.rosbag_path' (required when lidar.source is 'rosbag')")

    ext_cfg = cfg.get("extrinsic")
    if ext_cfg is not None:
        if not isinstance(ext_cfg, dict):
            problems.append("'extrinsic' must be a mapping")
        else:
            for key in _REQUIRED_EXTRINSIC_KEYS:
                if key not in ext_cfg:
                    problems.append(f"missing 'extrinsic.{key}'")
            if "rotation_format" in ext_cfg and ext_cfg["rotation_format"] not in _VALID_ROTATION_FORMATS:
                problems.append(
                    f"'extrinsic.rotation_format' is {ext_cfg['rotation_format']!r}, "
                    f"must be one of {_VALID_ROTATION_FORMATS}"
                )
            for role_key in ("parent", "child"):
                if role_key in ext_cfg and ext_cfg[role_key] not in _VALID_EXTRINSIC_ROLES:
                    problems.append(
                        f"'extrinsic.{role_key}' is {ext_cfg[role_key]!r}, "
                        f"must be one of {_VALID_EXTRINSIC_ROLES}"
                    )

    if problems:
        bullet_list = "\n".join(f"  - {p}" for p in problems)
        raise ConfigSchemaError(
            f"'{config_path}' does not match the expected schema:\n"
            f"{bullet_list}\n{_SCHEMA_POINTER}"
        )


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
    warnings.extend(dataset.warnings)

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

    return EvaluationDataset(camera=camera, lidar=lidar, extrinsic=extrinsic,
                              sync_config=SyncConfig(), frames=frames)


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
) -> dict:
    """Run M0/M2/M3/M4 + quality scoring + visuals + report writing for a
    built EvaluationDataset. Returns the report dict (already written to
    disk in output_dir -- report.json always, report.html unless
    json_only=True).

    advanced: if True, also runs the Phase-5 diagnostics (Plane Consistency,
    Perturbation Sensitivity, Temporal Drift) and attaches them to the
    report's 'advanced' section. These never affect quality_score -- they're
    supplementary, not part of the MVP scored set -- and cost noticeably
    more time (perturbation alone re-runs M2 roughly 24 times), so they're
    opt-in rather than the default.

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

    visuals = {}
    if not skip_visuals:
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
        visuals["camera_frustum_png"] = render_camera_frustum_from_dataset(dataset, frame_index=headline_idx)
        bev_png = render_bev_dual_panel_from_result(image, points, dataset.extrinsic.T_CL, dataset.camera, m2,
                                                      edge_kwargs=edge_kwargs)
        if bev_png is not None:
            visuals["bev_dual_panel_png"] = bev_png
        visuals["interactive_scene"] = build_interactive_scene_from_dataset(dataset, frame_index=headline_idx)

    report = build_report(
        dataset, m2, m3, m4, quality, m0_result=m0.to_dict(),
        n_blocks=n_blocks, min_frames_m4=min_frames_m4, extra_warnings=extra_warnings,
        plane_result=plane_result, perturbation_result=perturbation_result,
        temporal_drift_result=temporal_drift_result,
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
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

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
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"error: failed to load dataset: {e}", file=sys.stderr)
        return 1

    weights = _parse_weights(args.weights)

    try:
        report = run_pipeline(
            dataset, args.output_dir,
            n_blocks=args.n_blocks, min_frames_per_block=args.min_frames_per_block,
            min_frames_m4=args.min_frames_m4, depth_jump_threshold_m=args.depth_jump_threshold_m,
            edge_radius_px=args.edge_radius_px, frame_index=args.frame_index,
            weights=weights, no_visuals=args.no_visuals, json_only=args.json_only, advanced=args.advanced,
            extra_warnings=extra_warnings,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print_console_summary(report, args.output_dir, json_only=args.json_only)

    q = report["quality_score"]
    if args.fail_on_bad and q["overall_classification"] in ("BAD", "FAIL"):
        return 2
    if args.fail_on_partial and q["num_valid_categories"] < len(q["categories"]):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
