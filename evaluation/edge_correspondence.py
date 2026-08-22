"""
evaluation/edge_correspondence.py

STEP 6 -- M2 Edge Correspondence improvement (see evaluation_metric_spec.md's
STEP 6, "이제 M2를 제대로 고친다").

The ORIGINAL M2 matching (evaluation.edge_alignment's compute_distance_transform
+ sample_bilinear) answers a purely geometric question: "what's the pixel
distance from this LiDAR edge point to the NEAREST image edge, whatever it
is?" That's cheap and works when there's only one plausible edge nearby, but
says nothing about whether the nearest edge is actually the SAME physical
structure the LiDAR point is on the boundary of -- two edges 4px apart with
completely different orientations (e.g. a window frame's vertical edge next
to a shadow's near-horizontal edge) are indistinguishable to a pure
nearest-distance search, and it will happily "match" to whichever happens to
be a pixel closer, physically correct or not.

This module replaces that single nearest-distance lookup with:
  1. candidate search (2-2): image edge pixels within GROWING radii
     (5px, then 10px, then 15px -- stop growing as soon as a radius produces
     at least one candidate that also passes step 2's orientation filter).
  2. orientation agreement (2-3): each LiDAR edge point's own local boundary
     tangent (estimated from neighboring LiDAR edge points, see
     estimate_lidar_edge_orientations) is compared against each candidate
     image edge pixel's own tangent (from the image gradient, see
     compute_edge_orientation_map). Candidates whose orientation disagrees
     by more than max_orientation_diff_deg are DISCARDED, not just down-
     weighted -- matching the spec's explicit framing ("82° vs 86° -> good
     candidate. 82° vs 170° -> discard.").
  3. gradient strength (2-3, cont.): among orientation-surviving candidates,
     the one with the STRONGEST image gradient wins (ties broken by
     distance) -- preferring a bold, well-defined structural edge over a
     faint one that happens to be marginally closer.
  4. local consistency (2-4): after every point has a tentative match, a
     point's match is checked against its NEIGHBORING LiDAR edge points'
     matches -- if this point's displacement (matched pixel - LiDAR pixel)
     points in a wildly different direction than its neighbors' displacements,
     the match is almost certainly a coincidence (e.g. a lone noisy point
     snapping to unrelated texture) rather than genuine structure, and is
     discarded.

A point that ends up with NO surviving candidate (failed candidate search,
orientation, or consistency) is NOT silently dropped from the error
statistics -- that would let a badly-matched scene hide its worst points by
having them simply "not count". Instead it's penalized at
max(radii_px) -- "we searched out to our largest radius and found nothing
plausible, so this point is worse than anything we were willing to search"
-- a principled floor, not an arbitrary constant, and one that never
REWARDS failing to find a proper correspondence (which reusing the old
nearest-any-edge distance as a fallback would risk doing, since a nearby
but orientation-rejected edge could be closer than a genuinely correct but
farther one).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2
from scipy.spatial import cKDTree

from evaluation.edge_alignment import extract_image_edges


# ---------------------------------------------------------------------------
# Step 1: image edge orientation + strength map
# ---------------------------------------------------------------------------

@dataclass
class EdgeOrientationMap:
    edge_mask: np.ndarray        # (H, W) uint8, 0/255 -- same as extract_image_edges
    orientation_deg: np.ndarray  # (H, W) float, edge TANGENT direction, mod 180 (undirected)
    strength: np.ndarray         # (H, W) float, gradient magnitude (unnormalized)


def compute_edge_orientation_map(
    image: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    sobel_ksize: int = 3,
    edge_mask: Optional[np.ndarray] = None,
) -> EdgeOrientationMap:
    """
    Canny edge mask (candidate pixel pool) + per-pixel Sobel gradient
    orientation/strength (evaluated everywhere, not just on edge pixels --
    candidates are still restricted to edge_mask pixels by the caller, but
    computing orientation/strength on the full image avoids a second,
    separate restricted pass).

    edge_mask: if the caller already ran extract_image_edges on this same
    image (e.g. evaluate_edge_alignment does, to check for "no edges at
    all" before matching), pass it in here to skip a redundant second
    Canny pass over the same image.

    orientation_deg is the edge TANGENT (the direction the edge line
    itself runs), not the gradient direction (which points ACROSS the
    edge, i.e. the tangent + 90 degrees) -- comparing tangents is what
    "does this LiDAR boundary run the same way as this image edge" means.
    Taken mod 180 throughout since a line's orientation is undirected (a
    boundary running "up-right" and one running "down-left" at the same
    angle are the same edge orientation).
    """
    if edge_mask is None:
        edge_mask = extract_image_edges(image, canny_low, canny_high)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray_f = gray.astype(np.float64)
    gx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    gy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=sobel_ksize)

    strength = np.hypot(gx, gy)
    # gradient direction (across the edge) -> tangent is +90 degrees;
    # mod 180 collapses the undirected ambiguity.
    gradient_deg = np.degrees(np.arctan2(gy, gx))
    orientation_deg = np.mod(gradient_deg + 90.0, 180.0)

    return EdgeOrientationMap(edge_mask=edge_mask, orientation_deg=orientation_deg, strength=strength)


# ---------------------------------------------------------------------------
# Step 2: LiDAR edge point local orientation (from neighboring LiDAR edge
# points' own 2D layout, via local PCA)
# ---------------------------------------------------------------------------

def estimate_lidar_edge_orientations(edge_pixels: np.ndarray, k: int = 6) -> np.ndarray:
    """
    Per-LiDAR-edge-point local boundary tangent, estimated from its k
    nearest OTHER LiDAR edge points' pixel positions via PCA: the
    principal (largest-variance) direction of a small local neighborhood
    of points lying along a boundary IS that boundary's tangent direction,
    a standard local-line-fitting technique.

    Returns degrees, mod 180 (undirected, matching
    compute_edge_orientation_map's convention so the two are directly
    comparable). Points with fewer than 2 usable neighbors (including
    itself) get NaN -- there's no meaningful local direction to estimate
    from a single point, and NaN orientation differences always fail the
    orientation filter in find_correspondences (comparisons against NaN
    are False), which is the correct, honest behavior: "we don't know
    this point's orientation" should never let it pass as a match.
    """
    n = edge_pixels.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.full(1, np.nan)

    tree = cKDTree(edge_pixels)
    k_query = min(k + 1, n)  # +1 since the point itself is always its own nearest neighbor
    _, neighbor_idx = tree.query(edge_pixels, k=k_query)
    if k_query == 1:
        return np.full(n, np.nan)

    orientations = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        neighbors = edge_pixels[neighbor_idx[i]]  # includes point i itself
        centered = neighbors - neighbors.mean(axis=0)
        # principal direction via SVD of the centered neighborhood --
        # equivalent to PCA's leading eigenvector of the covariance matrix,
        # numerically steadier for small point counts than eig(cov).
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        principal = vt[0]
        angle = np.degrees(np.arctan2(principal[1], principal[0]))
        orientations[i] = np.mod(angle, 180.0)
    return orientations


# ---------------------------------------------------------------------------
# Step 3-4: candidate search + orientation filter + strength selection,
# then local consistency filtering
# ---------------------------------------------------------------------------

def _angular_diff_mod180(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """Smallest difference between two mod-180 (undirected) angles, in
    [0, 90] degrees."""
    diff = np.abs(a_deg - b_deg) % 180.0
    return np.minimum(diff, 180.0 - diff)


@dataclass
class CorrespondenceResult:
    """Per-LiDAR-edge-point correspondence outcome (arrays are all length
    N, aligned with the input edge_pixels order)."""
    matched: np.ndarray             # (N,) bool -- found a candidate surviving orientation + consistency
    matched_pixels: np.ndarray      # (N, 2) -- matched candidate's pixel coords; NaN row if not matched
    distance_px: np.ndarray         # (N,) -- the reported per-point error: real distance if matched,
                                     #         max(radii_px) penalty if not
    orientation_diff_deg: np.ndarray  # (N,) -- NaN if not matched
    strength: np.ndarray            # (N,) -- candidate's gradient strength; NaN if not matched
    num_candidates_considered: np.ndarray  # (N,) int -- orientation-surviving candidates at the radius that succeeded (0 if never succeeded)
    rejection_reason: list          # (N,) -- None if matched, else "no_candidate" | "consistency"


def find_correspondences(
    lidar_pixels: np.ndarray,
    lidar_orientations_deg: np.ndarray,
    orientation_map: EdgeOrientationMap,
    radii_px: tuple[float, ...] = (5.0, 10.0, 15.0),
    max_orientation_diff_deg: float = 30.0,
) -> CorrespondenceResult:
    """
    STEP 6's 2-2 (candidate search) + 2-3 (orientation + strength): for
    each LiDAR edge point, search image edge pixels at growing radii,
    keep only orientation-agreeing candidates, and pick the strongest
    (ties broken by distance) among them. Does NOT yet apply local
    consistency (2-4) -- see apply_local_consistency_filter, which
    operates on this function's output.

    Vectorized per radius (batched KD-tree query + flatten-to-pairs +
    sort-based groupby-argmax, the same "flatten every candidate pair
    into one array, reduce in bulk" approach evaluation.edge_alignment's
    extract_lidar_edge_points already uses for its own neighbor search)
    rather than looping point-by-point in Python -- with edge point
    counts in the thousands on real (non-synthetic-test) point clouds
    and this function running once per frame across M2/M3/M4, a
    per-point Python loop here would scale badly. Only points still
    unmatched after a smaller radius are re-queried at the next, larger
    radius, so points that succeed early don't get needlessly re-searched.
    """
    n = lidar_pixels.shape[0]
    matched = np.zeros(n, dtype=bool)
    matched_pixels = np.full((n, 2), np.nan)
    max_radius = max(radii_px)
    distance_px = np.full(n, max_radius, dtype=np.float64)
    orientation_diff_deg = np.full(n, np.nan)
    strength = np.full(n, np.nan)
    num_candidates = np.zeros(n, dtype=np.int64)
    rejection_reason: list = ["no_candidate"] * n

    edge_rows, edge_cols = np.nonzero(orientation_map.edge_mask)
    if edge_rows.size == 0 or n == 0:
        return CorrespondenceResult(matched, matched_pixels, distance_px, orientation_diff_deg,
                                     strength, num_candidates, rejection_reason)

    edge_coords_xy = np.stack([edge_cols, edge_rows], axis=1).astype(np.float64)  # (M, 2) as (u, v)
    tree = cKDTree(edge_coords_xy)

    # Points with unknown (NaN) local orientation can never pass the
    # orientation filter (NaN comparisons are always False) -- excluding
    # them up front avoids wasting a query on them at every radius.
    remaining_idx = np.nonzero(np.isfinite(lidar_orientations_deg))[0]

    for radius in sorted(radii_px):
        if remaining_idx.size == 0:
            break

        query_points = lidar_pixels[remaining_idx]
        neighbor_lists = tree.query_ball_point(query_points, r=radius)

        lengths = np.array([len(lst) for lst in neighbor_lists], dtype=np.int64)
        if lengths.sum() == 0:
            continue  # nobody found any candidate at all at this radius -- try the next one

        point_local_repeat = np.repeat(np.arange(remaining_idx.size), lengths)
        cand_flat = np.concatenate([np.asarray(lst, dtype=np.intp) for lst in neighbor_lists if lst])
        point_global_idx = remaining_idx[point_local_repeat]

        cand_xy = edge_coords_xy[cand_flat]
        cand_rows = cand_xy[:, 1].astype(np.intp)
        cand_cols = cand_xy[:, 0].astype(np.intp)
        cand_orient = orientation_map.orientation_deg[cand_rows, cand_cols]
        cand_strength = orientation_map.strength[cand_rows, cand_cols]

        lidar_orient_expanded = lidar_orientations_deg[point_global_idx]
        diff = _angular_diff_mod180(lidar_orient_expanded, cand_orient)
        valid = diff <= max_orientation_diff_deg
        if not valid.any():
            continue

        pg = point_global_idx[valid]
        cxy = cand_xy[valid]
        cs = cand_strength[valid]
        cdiff = diff[valid]
        dist = np.linalg.norm(cxy - lidar_pixels[pg], axis=1)

        # Vectorized "pick the best candidate per point" (groupby-argmax
        # without a Python loop): sort so each point's own candidates are
        # contiguous and already ordered strongest-first / closest-first,
        # then take the first row of each group. np.lexsort's LAST key is
        # the primary sort key, so (dist, -cs, pg) groups by point index
        # first, then strongest gradient, then closest distance as the
        # final tiebreaker.
        order = np.lexsort((dist, -cs, pg))
        pg_sorted = pg[order]
        winners_pg, first_idx, counts = np.unique(pg_sorted, return_index=True, return_counts=True)
        winner_rows = order[first_idx]

        matched[winners_pg] = True
        matched_pixels[winners_pg] = cxy[winner_rows]
        distance_px[winners_pg] = dist[winner_rows]
        orientation_diff_deg[winners_pg] = cdiff[winner_rows]
        strength[winners_pg] = cs[winner_rows]
        num_candidates[winners_pg] = counts
        for idx in winners_pg:
            rejection_reason[idx] = None

        remaining_idx = np.setdiff1d(remaining_idx, winners_pg, assume_unique=True)

    return CorrespondenceResult(matched, matched_pixels, distance_px, orientation_diff_deg,
                                 strength, num_candidates, rejection_reason)


def apply_local_consistency_filter(
    lidar_pixels: np.ndarray,
    result: CorrespondenceResult,
    k: int = 5,
    max_angle_diff_deg: float = 45.0,
    max_magnitude_ratio: float = 3.0,
    min_magnitude_diff_px: float = 3.0,
    min_displacement_for_angle_check_px: float = 1.5,
    penalty_distance_px: Optional[float] = None,
) -> CorrespondenceResult:
    """
    STEP 6's 2-4: demote a match to "unmatched" (with the same max(radii)
    penalty find_correspondences uses) if its displacement vector
    (matched_pixel - lidar_pixel) disagrees sharply with its k nearest
    MATCHED neighbors' own displacement vectors -- a lone point whose
    "correspondence" points in a completely different direction (or is
    wildly larger/smaller in magnitude) than everything around it is much
    more likely a coincidental snap to unrelated texture than genuine
    structure, even if it individually passed the orientation filter.

    Only evaluated among points that are ALREADY matched (unmatched points
    have nothing to compare) and only against OTHER matched points as
    neighbors (comparing against an already-rejected neighbor's
    nonexistent displacement would be meaningless). Points with fewer
    than 2 matched neighbors within range are left as-is (nothing to
    check consistency against) rather than being penalized for something
    outside their control.

    The magnitude check requires BOTH max_magnitude_ratio AND
    min_magnitude_diff_px to be exceeded before demoting -- a pure ratio
    is unstable near zero (e.g. a 0.9px vs 3.9px displacement pair is a
    ~4.3x ratio, which would trip a ratio-only threshold, but the
    absolute 3px difference is well within ordinary pixel-grid
    discretization noise, not a sign of a spurious match). Genuinely
    inconsistent matches (e.g. 2px vs 20px) still clear both bars.

    The ANGLE check is only applied when this point's own displacement is
    at least min_displacement_for_angle_check_px -- a near-zero
    displacement vector's DIRECTION is dominated by sub-pixel matching
    noise, not signal (e.g. [0.07, 0.03] vs [-0.05, 0.09] can differ by
    90+ degrees despite both meaning "this point already matches almost
    exactly", which is the BEST possible outcome, not an inconsistency to
    flag). Without this gate, a well-calibrated scene with genuinely tiny
    (sub-pixel) errors would have most of its points spuriously demoted
    on essentially meaningless angle noise -- discovered via exactly that
    failure mode during development (see evaluation_metric_spec.md's
    perturbation-sensitivity test suite, which specifically checks that
    the TRUE calibration is a local error minimum; it caught this).

    penalty_distance_px: the distance value assigned to any point demoted
    here -- should be the SAME max(radii_px) find_correspondences used,
    passed through explicitly (match_lidar_edges_to_image does this) so a
    demoted point's penalty is correct even in the edge case where EVERY
    point matched successfully (so result.distance_px never happens to
    contain the true ceiling value to fall back on). If not given,
    falls back to max(result.distance_px), which is only exact when at
    least one point in `result` is already unmatched.

    Vectorized across all matched points at once (single batched KD-tree
    query + a single np.nanmedian call over the whole (M, k) neighbor
    array) rather than looping point-by-point with a separate
    np.median() call per point -- see find_correspondences' docstring
    for why a per-point Python loop doesn't scale to real point counts.

    Returns a NEW CorrespondenceResult (does not mutate the input).
    """
    n = lidar_pixels.shape[0]
    matched = result.matched.copy()
    matched_pixels = result.matched_pixels.copy()
    distance_px = result.distance_px.copy()
    orientation_diff_deg = result.orientation_diff_deg.copy()
    strength = result.strength.copy()
    num_candidates = result.num_candidates_considered.copy()
    rejection_reason = list(result.rejection_reason)

    if penalty_distance_px is None:
        penalty_distance_px = float(distance_px.max()) if distance_px.size else 0.0

    matched_idx = np.nonzero(matched)[0]
    m = matched_idx.size
    if m < 3:
        # not enough matched points anywhere to form a meaningful "local
        # neighborhood" comparison -- leave everything as find_correspondences
        # left it.
        return CorrespondenceResult(matched, matched_pixels, distance_px, orientation_diff_deg,
                                     strength, num_candidates, rejection_reason)

    matched_pixels_only = lidar_pixels[matched_idx]
    displacements = result.matched_pixels[matched_idx] - matched_pixels_only  # (M, 2)

    tree = cKDTree(matched_pixels_only)
    k_query = min(k + 1, m)
    _, neighbor_local_idx = tree.query(matched_pixels_only, k=k_query)
    if neighbor_local_idx.ndim == 1:
        neighbor_local_idx = neighbor_local_idx[:, None]  # k_query == 1 edge case

    self_row = np.arange(m)[:, None]
    is_self = neighbor_local_idx == self_row

    neighbor_disps = displacements[neighbor_local_idx]  # (M, k_query, 2)
    neighbor_mags = np.linalg.norm(neighbor_disps, axis=2)  # (M, k_query)
    invalid = is_self | (neighbor_mags <= 1e-9)

    valid_neighbor_count = (~invalid).sum(axis=1)  # (M,)
    enough_neighbors = valid_neighbor_count >= 2

    masked_disps = np.where(invalid[:, :, None], np.nan, neighbor_disps)
    with np.errstate(invalid="ignore"):
        import warnings as _warnings
        with _warnings.catch_warnings():
            # rows with zero valid neighbors produce an all-NaN slice,
            # which nanmedian warns about -- those rows are already
            # excluded via enough_neighbors below, so the warning is
            # noise, not a signal of an actual problem.
            _warnings.simplefilter("ignore", category=RuntimeWarning)
            median_disp = np.nanmedian(masked_disps, axis=1)  # (M, 2), NaN row if no valid neighbors

    own_mag = np.linalg.norm(displacements, axis=1)  # (M,)
    median_mag = np.linalg.norm(median_disp, axis=1)  # (M,)

    with np.errstate(invalid="ignore", divide="ignore"):
        dot = np.einsum("ij,ij->i", displacements, median_disp)
        cos_angle = np.clip(dot / (own_mag * median_mag), -1.0, 1.0)
        angle_diff_deg = np.degrees(np.arccos(cos_angle))
        mag_ratio = np.maximum(own_mag / median_mag, median_mag / own_mag)

    usable = enough_neighbors & (own_mag > 1e-9) & (median_mag > 1e-9)
    magnitude_diff = np.abs(own_mag - median_mag)
    demote_angle = usable & (own_mag >= min_displacement_for_angle_check_px) & (angle_diff_deg > max_angle_diff_deg)
    demote_magnitude = usable & (mag_ratio > max_magnitude_ratio) & (magnitude_diff > min_magnitude_diff_px)
    demote_local = demote_angle | demote_magnitude

    demote_global = matched_idx[demote_local]
    if demote_global.size > 0:
        matched[demote_global] = False
        matched_pixels[demote_global] = np.nan
        distance_px[demote_global] = penalty_distance_px
        orientation_diff_deg[demote_global] = np.nan
        strength[demote_global] = np.nan
        num_candidates[demote_global] = 0
        for idx in demote_global:
            rejection_reason[idx] = "consistency"

    return CorrespondenceResult(matched, matched_pixels, distance_px, orientation_diff_deg,
                                 strength, num_candidates, rejection_reason)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def match_lidar_edges_to_image(
    lidar_pixels: np.ndarray,
    image: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    radii_px: tuple[float, ...] = (5.0, 10.0, 15.0),
    max_orientation_diff_deg: float = 30.0,
    k_orientation: int = 6,
    k_consistency: int = 5,
    max_consistency_angle_deg: float = 45.0,
    max_consistency_magnitude_ratio: float = 3.0,
    min_consistency_magnitude_diff_px: float = 3.0,
    min_consistency_displacement_for_angle_check_px: float = 1.5,
    edge_mask: Optional[np.ndarray] = None,
) -> CorrespondenceResult:
    """
    Full STEP 6 pipeline: image edge orientation/strength map -> LiDAR
    edge point local orientation -> candidate search + orientation +
    strength selection -> local consistency filtering. This is what
    evaluation.edge_alignment.evaluate_edge_alignment calls in place of
    the old compute_distance_transform + sample_bilinear step.

    edge_mask: passed straight through to compute_edge_orientation_map to
    skip a redundant Canny pass if the caller already has one for this
    image (evaluate_edge_alignment does).
    """
    orientation_map = compute_edge_orientation_map(image, canny_low, canny_high, edge_mask=edge_mask)
    lidar_orientations = estimate_lidar_edge_orientations(lidar_pixels, k=k_orientation)
    raw = find_correspondences(
        lidar_pixels, lidar_orientations, orientation_map,
        radii_px=radii_px, max_orientation_diff_deg=max_orientation_diff_deg,
    )
    return apply_local_consistency_filter(
        lidar_pixels, raw, k=k_consistency,
        max_angle_diff_deg=max_consistency_angle_deg,
        max_magnitude_ratio=max_consistency_magnitude_ratio,
        min_magnitude_diff_px=min_consistency_magnitude_diff_px,
        min_displacement_for_angle_check_px=min_consistency_displacement_for_angle_check_px,
        penalty_distance_px=max(radii_px),
    )
