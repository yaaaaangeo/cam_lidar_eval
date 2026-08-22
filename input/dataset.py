"""
input/dataset.py

Ties camera + lidar + extrinsic together into a single EvaluationDataset,
handling timestamp synchronization and providing the contiguous time-block
split needed by M3 (Hold-out Consistency), per the Input Loader Spec (v0.1).

STEP 2 -- Timestamp Synchronization (see evaluation_metric_spec.md's STEP 2):
replaces plain "closest lidar frame, full-array argmin" nearest-neighbor
matching with:
  1. candidate window: a camera frame may only match a lidar frame within
     +/- max_time_diff_ms (SyncConfig.max_time_diff_ms).
  2. monotonic matching: matched (camera_index, lidar_index) pairs are
     enforced to be strictly ordered -- if camera frame i matches lidar
     frame j, camera frame i+1 can only match lidar frame j' >= j. This is
     what a plain "argmin over all unused lidar frames" approach does NOT
     guarantee, and is what actually earns the name "synchronization"
     rather than "coincidentally usually-right matching".
  3. timestamp offset estimation: Δt = camera_clock - lidar_clock is
     estimated via a coarse, window-free median bootstrap pass (see
     _coarse_offset_estimate), then the real candidate-window search is
     re-centered using that estimate -- recovering matches that a
     systematic (but otherwise fixable) clock offset would otherwise push
     outside the raw window, without widening the window itself (which
     would let in more jitter, not just correct the offset). The offset
     figure actually reported is then re-measured from the pairs that
     passed the real window, not the coarse bootstrap value itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from input.camera import CameraModel, CameraFrame
from input.lidar import LidarModel, LidarFrame
from input.extrinsic import ExtrinsicModel


@dataclass
class SyncConfig:
    max_time_diff_ms: float = 50.0
    drop_unmatched: bool = True


@dataclass
class SyncedFrame:
    index: int
    timestamp: float
    camera_frame: CameraFrame
    lidar_frame: LidarFrame
    time_diff_ms: float


# Sync quality classification thresholds (mirrors the GOOD/WARNING/BAD/FAIL
# vocabulary used throughout evaluation/* and quality/quality_score.py).
# Kept relative to the caller's own max_time_diff_ms / camera frame count
# rather than fixed absolute constants, consistent with this project's
# "thresholds are sensor/config-relative, not hardcoded absolutes" design
# principle (see floor(Z) in quality/noise_floor.py).
DROP_RATIO_GOOD_MAX = 0.05       # <=5% of camera frames dropped
DROP_RATIO_WARNING_MAX = 0.20    # <=20% dropped
OFFSET_STD_GOOD_FRACTION = 0.2   # offset jitter <=20% of the tolerance window
OFFSET_STD_WARNING_FRACTION = 0.5  # <=50% of the tolerance window

_SYNC_SEVERITY = {"GOOD": 0, "WARNING": 1, "BAD": 2, "FAIL": 3}


def _worse_classification(a: str, b: str) -> str:
    return a if _SYNC_SEVERITY[a] >= _SYNC_SEVERITY[b] else b


def classify_sync(
    num_matched: int,
    num_camera_frames: int,
    offset_std_ms: float,
    max_time_diff_ms: float,
) -> str:
    """
    GOOD/WARNING/BAD/FAIL classification for a synchronization result.

    FAIL: nothing matched at all (the two streams could not be
    synchronized -- see the "no pairs matched" warning in build_dataset).

    Otherwise, the worse of two independent judgements:
      - drop_ratio: how much of the camera stream had no usable lidar
        match at all.
      - offset_std_ms relative to max_time_diff_ms: how much residual
        jitter remains in matched pairs' time differences AFTER the
        clock-offset correction -- a high value means either matching is
        unstable (irregular frame rates / dropped messages) or the two
        clocks aren't actually offset by a constant amount (e.g. clock
        drift), neither of which a constant Δt estimate can fix.
    """
    if num_matched == 0:
        return "FAIL"

    drop_ratio = 1.0 - num_matched / max(num_camera_frames, 1)
    if drop_ratio <= DROP_RATIO_GOOD_MAX:
        drop_class = "GOOD"
    elif drop_ratio <= DROP_RATIO_WARNING_MAX:
        drop_class = "WARNING"
    else:
        drop_class = "BAD"

    if max_time_diff_ms > 0:
        std_ratio = offset_std_ms / max_time_diff_ms
    else:
        std_ratio = 0.0
    if std_ratio <= OFFSET_STD_GOOD_FRACTION:
        std_class = "GOOD"
    elif std_ratio <= OFFSET_STD_WARNING_FRACTION:
        std_class = "WARNING"
    else:
        std_class = "BAD"

    return _worse_classification(drop_class, std_class)


@dataclass
class SyncStats:
    num_camera_frames: int
    num_lidar_frames: int
    num_matched: int
    num_camera_dropped: int
    num_lidar_dropped: int
    mean_time_diff_ms: float   # mean |camera_ts - lidar_ts| across matched pairs
    max_time_diff_ms: float    # max |camera_ts - lidar_ts| across matched pairs (observed, not the config threshold)

    # STEP 2 additions -- see classify_sync's docstring.
    estimated_offset_ms: float = float("nan")  # mean SIGNED (camera - lidar) clock offset, i.e. Δt
    offset_std_ms: float = float("nan")        # std of the signed per-pair offset (residual jitter after Δt correction)
    drop_ratio: float = float("nan")           # 1 - num_matched / num_camera_frames
    classification: str = "FAIL"               # GOOD | WARNING | BAD | FAIL

    def to_dict(self) -> dict:
        def _safe(x):
            xf = float(x)
            return xf if np.isfinite(xf) else None

        return {
            "num_camera_frames": self.num_camera_frames,
            "num_lidar_frames": self.num_lidar_frames,
            "num_matched": self.num_matched,
            "num_camera_dropped": self.num_camera_dropped,
            "num_lidar_dropped": self.num_lidar_dropped,
            "mean_time_diff_ms": _safe(self.mean_time_diff_ms),
            "max_time_diff_ms": _safe(self.max_time_diff_ms),
            "estimated_offset_ms": _safe(self.estimated_offset_ms),
            "offset_std_ms": _safe(self.offset_std_ms),
            "drop_ratio": _safe(self.drop_ratio),
            "classification": self.classification,
        }


@dataclass
class EvaluationDataset:
    camera: CameraModel
    lidar: LidarModel
    extrinsic: ExtrinsicModel
    sync_config: SyncConfig
    frames: list[SyncedFrame] = field(default_factory=list)
    sync_stats: Optional[SyncStats] = None
    warnings: list[str] = field(default_factory=list)
    # STEP 1 -- Input Validation (input/validation.py) result, as a plain
    # dict (ValidationReport.to_dict()). None means validation wasn't run
    # for this dataset (e.g. the synthetic --demo path, which is valid by
    # construction). Kept as an already-serialized dict rather than a
    # ValidationReport object so report/builder.py can pass it straight
    # through without an input/validation.py import.
    input_validation: Optional[dict] = None

    def time_blocks(self, n: int) -> list[list[SyncedFrame]]:
        """
        Split self.frames into n contiguous, roughly-equal time blocks, in
        frame order (frames are already time-sorted by construction).
        Used by M3 (Hold-out Consistency). Per spec: contiguous blocks only
        -- no random shuffling, to preserve temporal structure.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if not self.frames:
            return [[] for _ in range(n)]

        total = len(self.frames)
        # np.array_split gives contiguous, near-equal-size chunks (some may
        # differ by 1 frame when total isn't divisible by n) -- exactly the
        # "자동 균등 분할" behavior specified.
        indices = np.array_split(np.arange(total), n)
        return [[self.frames[i] for i in block] for block in indices]


def _two_pointer_match(
    cam_ts_sorted: np.ndarray,
    lid_ts_sorted: np.ndarray,
    max_diff_s: float,
    offset_s: float = 0.0,
) -> list[tuple[int, int, float]]:
    """
    Single matching pass over TWO ALREADY TIME-SORTED sequences, producing
    (camera_index, lidar_index, signed_diff_seconds) triples where
    signed_diff = camera_ts[camera_index] - lidar_ts[lidar_index] (the raw,
    uncorrected difference -- always the true Δt for that pair, regardless
    of what `offset_s` was used to find the candidate).

    This is STEP 2's 2-2 (candidate window) + 2-3 (monotonic matching)
    combined into one pass: a monotonically-advancing lidar pointer `j`
    means a matched lidar index can never go backwards as the camera index
    increases, which a per-camera-frame "argmin over ALL unused lidar
    frames" search does not guarantee. `offset_s` shifts the search target
    (camera_ts - offset_s) without widening the window itself -- see
    _monotonic_windowed_sync's docstring for why that matters.

    A camera frame with no lidar frame within the window is simply
    skipped (dropped) without advancing `j`, so it doesn't block a later,
    better-matching camera frame from claiming that same lidar frame.
    """
    n_lidar = lid_ts_sorted.shape[0]
    matches: list[tuple[int, int, float]] = []
    j = 0
    for i in range(cam_ts_sorted.shape[0]):
        if j >= n_lidar:
            break  # no lidar frames left at all; every remaining camera frame drops
        target = cam_ts_sorted[i] - offset_s
        while j + 1 < n_lidar and abs(lid_ts_sorted[j + 1] - target) <= abs(lid_ts_sorted[j] - target):
            j += 1
        if abs(lid_ts_sorted[j] - target) <= max_diff_s:
            actual_diff = float(cam_ts_sorted[i] - lid_ts_sorted[j])
            matches.append((i, j, actual_diff))
            j += 1  # this lidar frame is claimed; never reused, preserving monotonicity
        # else: no lidar frame close enough to this camera frame -> dropped;
        # j is NOT advanced, since it may still be the best match for the
        # next camera frame.
    return matches


def _coarse_offset_estimate(cam_ts_sorted: np.ndarray, lid_ts_sorted: np.ndarray) -> float:
    """
    Bootstrap estimate of Δt = camera_clock - lidar_clock, used ONLY to
    pick a starting offset for the real (windowed, monotonic) matching
    pass below -- NOT used to produce actual matched pairs itself.

    Deliberately ignores max_time_diff_ms entirely: if the true clock
    offset is itself larger than the configured tolerance window (a very
    real scenario -- e.g. a sensor with a known fixed publish latency),
    a windowed first pass would match nothing and there would be no data
    to estimate an offset FROM. So this pass finds each camera frame's
    single nearest lidar frame by raw absolute time (via searchsorted,
    since both arrays are already time-sorted) with no distance limit,
    then takes the MEDIAN of the signed differences -- median rather than
    mean specifically so that camera frames with no true nearby lidar
    frame at all (which will report some arbitrary nearest neighbor,
    however far away) don't drag a mean estimate off toward nonsense;
    a modest minority of such outliers has no effect on the median as
    long as most camera frames DO have a genuinely close lidar frame.
    """
    if cam_ts_sorted.size == 0 or lid_ts_sorted.size == 0:
        return 0.0

    insert_idx = np.searchsorted(lid_ts_sorted, cam_ts_sorted)
    right_idx = np.clip(insert_idx, 0, lid_ts_sorted.size - 1)
    left_idx = np.clip(insert_idx - 1, 0, lid_ts_sorted.size - 1)
    right_diff = np.abs(lid_ts_sorted[right_idx] - cam_ts_sorted)
    left_diff = np.abs(lid_ts_sorted[left_idx] - cam_ts_sorted)
    nearest_idx = np.where(left_diff <= right_diff, left_idx, right_idx)

    signed_diffs = cam_ts_sorted - lid_ts_sorted[nearest_idx]
    return float(np.median(signed_diffs))


def _monotonic_windowed_sync(
    camera_frames: list[CameraFrame],
    lidar_frames: list[LidarFrame],
    max_time_diff_ms: float,
) -> tuple[list[SyncedFrame], SyncStats]:
    """
    STEP 2 timestamp synchronization: candidate window + monotonic
    matching + timestamp offset estimation (see this module's docstring).

    Pipeline:
      1. _coarse_offset_estimate: a window-free, median-based bootstrap
         estimate of Δt = camera_clock - lidar_clock, from EVERY camera
         frame's single nearest lidar frame (not yet subject to
         max_time_diff_ms or monotonicity -- see that function's
         docstring for why a window here would be self-defeating when the
         true offset exceeds it).
      2. _two_pointer_match using that estimate to re-center the candidate
         window (2-2) and a monotonically-advancing lidar pointer (2-3) --
         this pass produces the actual matched pairs, and IS bound by
         max_time_diff_ms on each pair's raw (uncorrected) difference.
      3. The final estimated_offset_ms / offset_std_ms reported are
         recomputed from step 2's ACCEPTED matches (mean/std of their raw
         signed differences) -- a precise re-measurement using only pairs
         that passed the real tolerance check, not the coarse bootstrap
         value itself.

    If step 2 finds nothing using the coarse-corrected window, falls back
    to an uncorrected (offset=0) pass, in case the coarse estimate was
    itself misled by a pathological timestamp pattern -- this never makes
    things worse, only recovers a plain nearest-neighbor-in-window result.
    """
    cam_ts = np.array([f.timestamp for f in camera_frames], dtype=float)
    lid_ts = np.array([f.timestamp for f in lidar_frames], dtype=float)
    max_diff_s = max_time_diff_ms / 1000.0

    # Sort a working copy by timestamp for the two-pointer scan; frames
    # aren't guaranteed to already be in that order for every caller (e.g.
    # a hand-built list in a test), even though the loaders themselves
    # now sort by parsed timestamp (see input/camera.py, input/lidar.py).
    cam_order = np.argsort(cam_ts, kind="stable")
    lid_order = np.argsort(lid_ts, kind="stable")
    cam_ts_sorted = cam_ts[cam_order]
    lid_ts_sorted = lid_ts[lid_order]

    coarse_offset_s = _coarse_offset_estimate(cam_ts_sorted, lid_ts_sorted)
    final_matches = _two_pointer_match(cam_ts_sorted, lid_ts_sorted, max_diff_s, offset_s=coarse_offset_s)
    if not final_matches and coarse_offset_s != 0.0:
        final_matches = _two_pointer_match(cam_ts_sorted, lid_ts_sorted, max_diff_s, offset_s=0.0)

    if final_matches:
        final_diffs = np.array([d for _, _, d in final_matches])
        estimated_offset_s = float(np.mean(final_diffs))
        offset_std_s = float(np.std(final_diffs))  # population std; still informative as "0 when n==1"
    else:
        estimated_offset_s = float("nan")
        offset_std_s = float("nan")

    matches: list[SyncedFrame] = []
    for i_sorted, j_sorted, diff in final_matches:
        cam_idx = int(cam_order[i_sorted])
        lid_idx = int(lid_order[j_sorted])
        matches.append(SyncedFrame(
            index=len(matches),
            timestamp=float(cam_ts[cam_idx]),
            camera_frame=camera_frames[cam_idx],
            lidar_frame=lidar_frames[lid_idx],
            time_diff_ms=abs(diff) * 1000.0,
        ))

    # Re-sort by timestamp (should already be in this order since i_sorted
    # is monotonic in the two-pointer scan, but re-assert + fix up `index`
    # for clarity/determinism, matching the previous implementation).
    matches.sort(key=lambda m: m.timestamp)
    for i, m in enumerate(matches):
        m.index = i

    num_matched = len(matches)
    time_diffs_ms = [m.time_diff_ms for m in matches]
    estimated_offset_ms = estimated_offset_s * 1000.0
    offset_std_ms = offset_std_s * 1000.0
    drop_ratio = 1.0 - num_matched / max(len(camera_frames), 1)

    stats = SyncStats(
        num_camera_frames=len(camera_frames),
        num_lidar_frames=len(lidar_frames),
        num_matched=num_matched,
        num_camera_dropped=len(camera_frames) - num_matched,
        num_lidar_dropped=len(lidar_frames) - num_matched,
        mean_time_diff_ms=float(np.mean(time_diffs_ms)) if time_diffs_ms else float("nan"),
        max_time_diff_ms=float(np.max(time_diffs_ms)) if time_diffs_ms else float("nan"),
        estimated_offset_ms=estimated_offset_ms,
        offset_std_ms=offset_std_ms,
        drop_ratio=drop_ratio,
        classification=classify_sync(num_matched, len(camera_frames), offset_std_ms, max_time_diff_ms),
    )
    return matches, stats


def build_dataset(
    camera: CameraModel,
    camera_frames: list[CameraFrame],
    lidar: LidarModel,
    lidar_frames: list[LidarFrame],
    extrinsic: ExtrinsicModel,
    sync_config: Optional[SyncConfig] = None,
) -> EvaluationDataset:
    """
    Construct an EvaluationDataset by time-synchronizing camera and lidar
    frame sequences (STEP 2 -- see this module's docstring and
    _monotonic_windowed_sync). Drops unmatched frames (per
    SyncConfig.drop_unmatched; the only currently-implemented behavior is
    drop=True) and always reports how many were dropped -- per spec,
    dropped-frame counts are a data-quality signal and must never be
    silently discarded.
    """
    sync_config = sync_config or SyncConfig()
    if not sync_config.drop_unmatched:
        raise NotImplementedError(
            "drop_unmatched=False (e.g. interpolation-based sync) is not "
            "implemented in this pass."
        )

    matches, stats = _monotonic_windowed_sync(
        camera_frames, lidar_frames, sync_config.max_time_diff_ms
    )

    warnings: list[str] = []
    if stats.num_matched == 0:
        warnings.append(
            "No camera-lidar frame pairs matched within max_time_diff_ms="
            f"{sync_config.max_time_diff_ms}. Check that both sensors' "
            "timestamps are in the same clock/epoch, or increase max_time_diff_ms."
        )
    else:
        if stats.drop_ratio > 0.5:
            warnings.append(
                f"Over half of camera frames ({stats.num_camera_dropped}/"
                f"{stats.num_camera_frames}) had no matching lidar frame within "
                f"{sync_config.max_time_diff_ms}ms. Sync quality is poor; "
                f"evaluation results may not be representative."
            )
        if abs(stats.estimated_offset_ms) > sync_config.max_time_diff_ms * 0.5:
            warnings.append(
                f"Estimated camera-lidar clock offset ({stats.estimated_offset_ms:+.1f}ms) "
                f"is more than half of max_time_diff_ms ({sync_config.max_time_diff_ms}ms). "
                f"Consider a fixed per-topic latency/offset correction upstream, or "
                f"increasing max_time_diff_ms if this offset is expected."
            )

    return EvaluationDataset(
        camera=camera,
        lidar=lidar,
        extrinsic=extrinsic,
        sync_config=sync_config,
        frames=matches,
        sync_stats=stats,
        warnings=warnings,
    )
