# Copyright (c) 2025 The University of Washington
#
# This file is part of rapidtools.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# rapidtools. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Barbaros Cetiner
#
# Last updated:
# 10-03-2026

"""
From sightings to objects: tracking, triangulation, merging and ray voting.

A street survey sees every parked car, pole or hydrant in many consecutive
frames. Each sighting gives a precise bearing from a known camera position
and only a rough range, so the right way to turn sightings into objects is
the one every multi-view system uses:

1. **Track** detections from frame to frame within a sequence by bearing
   continuity (:func:`track_sequence`). Association happens in bearing space,
   where the data is accurate, not in ground space, where it is not.
2. **Triangulate** each track robustly (:func:`triangulate_track`). Rays
   that cannot be intersected consistently belong to a moving vehicle and
   are dropped; tracks without parallax keep a single-view estimate with an
   honest, elongated uncertainty.
3. **Merge** the resulting estimates with a Mahalanobis gate
   (:func:`merge_estimates`), so a car seen from two passes or split tracks
   becomes one object while two cars seen in the same frame never do.

:func:`vote_rays` is an independent baseline: every ray deposits votes into
a ground grid and objects are the peaks. It needs no association at all and
is useful to check the tracking result against.

All geometry runs in a local metre frame from
:func:`~rapidtools.processing.street_localization.local_projection`.

Example:
    >>> from rapidtools.processing.street_tracking import (
    ...     discover_objects,
    ... )
    >>> estimates = discover_objects(observations, max_range_m=30.0)  # doctest: +SKIP
    >>> estimates[0].localization, estimates[0].n_images  # doctest: +SKIP
    ('triangulated', 6)
"""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rapidtools.core import Observation

from .street_localization import (
    DEFAULT_MAX_RANGE_M,
    MIN_POLYGON_AREA_FRACTION,
    MIN_TRIANGULATION_SEPARATION_DEG,
    local_projection,
    localize,
    polygon_angular_size,
)

logger = logging.getLogger(__name__)

#: One-sigma bearing error of a sighting (camera heading plus mask edge).
BEARING_SIGMA_DEG = 0.75
#: Chi-square threshold (2 degrees of freedom, 99 %) for the merge gate.
CHI2_2D_99 = 9.21
#: Fraction of a track's rays that must agree for it to count as static.
STATIC_INLIER_FRACTION = 0.6
#: How closely an outline must match the full height of an object at its
#: ground-contact range before a range disagreement is blamed on motion
#: rather than on something hiding the object's lower part.
MIN_VISIBLE_FRACTION = 0.9
#: Ray pairs tried per track when searching for the static consensus.
RANSAC_MAX_PAIRS = 48

Project = Callable[[float, float], tuple[float, float]]
Unproject = Callable[[float, float], tuple[float, float]]


# --------------------------------------------------------------- records
@dataclass
class ObjectEstimate:
    """
    One candidate object: a position with uncertainty and its sightings.

    Attributes:
        x: East coordinate in the local metre frame.
        y: North coordinate in the local metre frame.
        cov: 2 x 2 position covariance in square metres.
        members: The sightings that produced the estimate.
        localization: ``'triangulated'`` (rays intersected) or
            ``'single_view'`` (ground-contact ranges only).
        rms_m: Root-mean-square ray residual of a triangulation, else ``None``.
        parallax_deg: Largest bearing separation among the rays used.
    """

    x: float
    y: float
    cov: np.ndarray
    members: list[Observation]
    localization: str
    rms_m: float | None = None
    parallax_deg: float = 0.0

    @property
    def image_ids(self) -> set[str]:
        """Frames that saw the object."""
        return {o.image_id for o in self.members}

    @property
    def sequence_ids(self) -> set[str]:
        """Sequences that saw the object."""
        return {o.sequence_id for o in self.members if o.sequence_id}

    @property
    def n_images(self) -> int:
        """Number of distinct frames."""
        return len(self.image_ids)

    @property
    def sigma_m(self) -> float:
        """Largest one-sigma extent of the position uncertainty in metres."""
        return float(math.sqrt(max(np.linalg.eigvalsh(self.cov).max(), 0.0)))


@dataclass
class _Track:
    members: list[Observation] = field(default_factory=list)
    last_index: int = 0
    estimate: tuple[float, float] | None = None
    triangulated: bool = False
    rays: dict[str, tuple[float, float, float]] = field(default_factory=dict)


# --------------------------------------------------------------- helpers
def _angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings in degrees."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _bearing_to(cx: float, cy: float, x: float, y: float) -> float:
    return math.degrees(math.atan2(x - cx, y - cy)) % 360.0


def _unit(bearing_deg: float) -> np.ndarray:
    b = math.radians(bearing_deg)
    return np.array([math.sin(b), math.cos(b)])


def single_view_covariance(
    bearing_deg: float, range_m: float, bearing_sigma_deg: float = BEARING_SIGMA_DEG
) -> np.ndarray:
    """
    Covariance of a ground-contact position estimate.

    The error is small across the ray (bearing noise times range) and large
    along it (the ground contact and the camera height are uncertain), so
    the ellipse is elongated along the line of sight.

    Example:
        >>> cov = single_view_covariance(90.0, 10.0)
        >>> bool(cov[0, 0] > cov[1, 1])    # looking east: long axis is east-west
        True
    """
    across = range_m * math.tan(math.radians(bearing_sigma_deg)) + 0.3
    along = max(0.25 * range_m, 1.0)
    d = _unit(bearing_deg)
    n = np.array([d[1], -d[0]])
    return along**2 * np.outer(d, d) + across**2 * np.outer(n, n)


def _ray_arrays(
    rays: Sequence[tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera positions, unit directions and unit normals as arrays."""
    arr = np.asarray(rays, dtype=float).reshape(-1, 3)
    cams = arr[:, :2]
    b = np.radians(arr[:, 2])
    dirs = np.column_stack([np.sin(b), np.cos(b)])
    normals = np.column_stack([dirs[:, 1], -dirs[:, 0]])
    return cams, dirs, normals


def _solve_intersection(
    cams: np.ndarray, normals: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Solve the weighted normal equations of a ray intersection."""
    # sum_i w_i n_i n_i^T  and  sum_i w_i n_i n_i^T c_i, vectorised:
    wn = normals * weights[:, None]
    a = wn.T @ normals
    b = np.einsum('ij,ik,ik->j', wn, normals, cams)
    try:
        point = np.linalg.solve(a, b)
        cov = np.linalg.inv(a)
    except np.linalg.LinAlgError:
        return None
    return point, cov


def _weighted_intersection(
    rays: Sequence[tuple[float, float, float]],
    sigma_rad: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Weighted least-squares ray intersection: ``(point, covariance)``."""
    cams, _, normals = _ray_arrays(rays)
    weights = np.ones(len(cams))
    fit = None
    for _ in range(3):
        fit = _solve_intersection(cams, normals, weights)
        if fit is None:
            return None
        dist = np.linalg.norm(fit[0] - cams, axis=1)
        weights = 1.0 / np.maximum(sigma_rad * dist, 0.05) ** 2
    return fit


def _residuals(
    point: np.ndarray, cams: np.ndarray, dirs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Perpendicular and along-ray distances of ``point`` from array rays."""
    rel = point[None, :] - cams
    along = np.einsum('ij,ij->i', rel, dirs)
    perp = np.linalg.norm(rel - along[:, None] * dirs, axis=1)
    return perp, along


def _ray_residuals(
    point: np.ndarray, rays: Sequence[tuple[float, float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    """Perpendicular distance and along-ray distance of ``point`` per ray."""
    cams, dirs, _ = _ray_arrays(rays)
    return _residuals(point, cams, dirs)


def _lon_lat(obs: Observation) -> tuple[float, float]:
    """Ground position of an observation that has one."""
    if obs.lon is None or obs.lat is None:
        raise ValueError(f'Observation {obs.image_id!r} has no ground position.')
    return obs.lon, obs.lat


def _max_separation(bearings: Iterable[float]) -> float:
    degs = list(bearings)
    if len(degs) < 2:
        return 0.0
    return max(_angle_diff(a, b) for i, a in enumerate(degs) for b in degs[i + 1 :])


# --------------------------------------------------------------- tracking
def track_sequence(
    observations: Sequence[Observation],
    project: Project,
    max_gap_frames: int = 2,
    base_gate_deg: float = 6.0,
    gate_m: float = 2.5,
    gate_fraction: float = 0.10,
    loose_gate_deg: float = 25.0,
) -> list[list[Observation]]:
    """
    Link the sightings of one sequence into tracks, one per object.

    Frames are visited in capture order. For every live track the bearing
    the object should have from the new camera position is predicted (from
    the track's current position estimate, or its last bearing when it has
    none yet) and detections are assigned to tracks by the Hungarian
    algorithm on the normalised bearing difference plus a mild size term.
    Two detections in one frame can never join the same track, and a track
    not seen for ``max_gap_frames`` frames is closed.

    Args:
        observations: Sightings of one class in one sequence with ``bearing``
            set (see :func:`~rapidtools.processing.street_localization.localize`).
            Those with a ``range_m`` seed the position estimate.
        project: Local metre projection shared by the caller.
        max_gap_frames: Frames a track may miss before it is closed.
        base_gate_deg: Minimum angular gate for a track with a position.
        gate_m: Ground-distance tolerance behind the angular gate.
        gate_fraction: Extra tolerance as a fraction of the distance.
        loose_gate_deg: Gate for tracks that have no position estimate yet.

    Returns:
        list[list[Observation]]: The tracks, each in capture order.

    Example:
        >>> from rapidtools.processing.street_localization import local_projection
        >>> project, _ = local_projection(-117.4, 47.7)
        >>> tracks = track_sequence(sightings, project)  # doctest: +SKIP
        >>> [len(t) for t in tracks]  # doctest: +SKIP
        [7, 6, 1]
    """
    from scipy.optimize import linear_sum_assignment

    frames: dict[str, list[Observation]] = defaultdict(list)
    order: dict[str, tuple[str, str]] = {}
    for obs in observations:
        if obs.bearing is None:
            continue
        frames[obs.image_id].append(obs)
        order[obs.image_id] = (obs.captured_at or '', obs.image_id)
    frame_ids = sorted(frames, key=lambda i: order[i])

    live: list[_Track] = []
    done: list[_Track] = []
    for index, frame_id in enumerate(frame_ids):
        dets = frames[frame_id]
        cam = project(dets[0].camera_lon, dets[0].camera_lat)
        still_live = []
        for track in live:
            if index - track.last_index > max_gap_frames:
                done.append(track)
            else:
                still_live.append(track)
        live = still_live

        if live:
            cost = np.full((len(live), len(dets)), 1e6)
            for i, track in enumerate(live):
                last = track.members[-1]
                if track.estimate is not None:
                    dist = math.hypot(
                        track.estimate[0] - cam[0], track.estimate[1] - cam[1]
                    )
                    predicted = _bearing_to(cam[0], cam[1], *track.estimate)
                    tolerance = gate_m + gate_fraction * dist
                    gate = max(
                        base_gate_deg,
                        math.degrees(math.atan2(tolerance, max(dist, 0.5))),
                    )
                    if not track.triangulated:
                        # A single-view position can be metres off along the
                        # ray, so do not trust its predicted bearing too much:
                        gate = max(gate, loose_gate_deg)
                else:
                    predicted = float(last.bearing or 0.0)
                    gate = loose_gate_deg
                for j, det in enumerate(dets):
                    if det.label != last.label:
                        continue
                    diff = _angle_diff(float(det.bearing or 0.0), predicted) / gate
                    if diff > 1.0:
                        continue
                    size = abs(math.log(max(det.area, 1e-9) / max(last.area, 1e-9)))
                    cost[i, j] = diff + 0.25 * min(size, 2.0)
            rows, cols = linear_sum_assignment(cost)
            assigned = set()
            for i, j in zip(rows, cols, strict=True):
                if cost[i, j] >= 1e6:
                    continue
                _extend(live[i], dets[j], index, cam, project)
                assigned.add(j)
        else:
            assigned = set()
        for j, det in enumerate(dets):
            if j in assigned:
                continue
            track = _Track(last_index=index)
            _extend(track, det, index, cam, project)
            live.append(track)
    done.extend(live)
    return [t.members for t in done]


def _extend(
    track: _Track,
    det: Observation,
    index: int,
    cam: tuple[float, float],
    project: Project,
) -> None:
    track.members.append(det)
    track.last_index = index
    track.rays.setdefault(det.image_id, (cam[0], cam[1], float(det.bearing or 0.0)))
    # Position estimate: a quick intersection once two rays are far enough
    # apart, otherwise the mean of the single-view ground positions.
    rays = list(track.rays.values())
    if len(rays) >= 2 and _max_separation(b for _, _, b in rays) >= 3.0:
        fit = _weighted_intersection(rays, math.radians(BEARING_SIGMA_DEG))
        if fit is not None:
            perp, along = _ray_residuals(fit[0], rays)
            if np.all(along > 0):
                track.estimate = (float(fit[0][0]), float(fit[0][1]))
                track.triangulated = True
                return
    located = [project(*_lon_lat(o)) for o in track.members if o.has_location]
    if located:
        xs, ys = zip(*located, strict=True)
        track.estimate = (sum(xs) / len(xs), sum(ys) / len(ys))
        track.triangulated = False


# --------------------------------------------------------------- triangulation
def triangulate_track(
    members: Sequence[Observation],
    project: Project,
    max_range_m: float = DEFAULT_MAX_RANGE_M,
    bearing_sigma_deg: float = BEARING_SIGMA_DEG,
    min_parallax_deg: float = MIN_TRIANGULATION_SEPARATION_DEG,
    inlier_fraction: float = STATIC_INLIER_FRACTION,
    range_ratio_bounds: tuple[float, float] = (0.6, 1.6),
    min_object_width_m: float = 1.0,
    object_height_m: float = 1.5,
    camera_height_m: float = 2.4,
) -> ObjectEstimate | None:
    """
    Position a track from its rays, or ``None`` when it is not a static object.

    A thin wrapper over :func:`classify_track` that drops the reason.
    """
    return classify_track(
        members,
        project,
        max_range_m=max_range_m,
        bearing_sigma_deg=bearing_sigma_deg,
        min_parallax_deg=min_parallax_deg,
        inlier_fraction=inlier_fraction,
        range_ratio_bounds=range_ratio_bounds,
        min_object_width_m=min_object_width_m,
        object_height_m=object_height_m,
        camera_height_m=camera_height_m,
    )[0]


def classify_track(
    members: Sequence[Observation],
    project: Project,
    max_range_m: float = DEFAULT_MAX_RANGE_M,
    bearing_sigma_deg: float = BEARING_SIGMA_DEG,
    min_parallax_deg: float = MIN_TRIANGULATION_SEPARATION_DEG,
    inlier_fraction: float = STATIC_INLIER_FRACTION,
    range_ratio_bounds: tuple[float, float] = (0.6, 1.6),
    min_object_width_m: float = 1.0,
    object_height_m: float = 1.5,
    camera_height_m: float = 2.4,
    frame_sightings: Mapping[str, Sequence[Observation]] | None = None,
) -> tuple[ObjectEstimate | None, str]:
    """
    Position a track from its rays, or decide why it is not a static object.

    Rays from distinct cameras are intersected with a RANSAC search over ray
    pairs followed by a weighted least-squares refinement over the inliers.
    A track whose rays largely agree is ``'triangulated'``; a track with too
    little parallax falls back to its single-view ground positions
    (``'single_view'``); a track with several rays that do not agree was a
    moving object and yields ``None``.

    Args:
        members: The track's sightings (``bearing`` set on each).
        project: Local metre projection.
        max_range_m: Reject solutions farther than this from any camera.
        bearing_sigma_deg: One-sigma bearing error used for weights and the
            inlier tolerance.
        min_parallax_deg: Smallest bearing separation that counts as
            parallax.
        inlier_fraction: Share of rays that must agree for a static object.
        range_ratio_bounds: Accepted band for the median ratio between the
            sightings' ground-contact ranges and their distances to the
            intersection. A target moving at constant speed makes the rays
            meet at a phantom point whose distances do not match those
            ranges, so this is the check that catches such vehicles. Ranges
            that are too *long* can also come from occlusion (a fence hides
            the wheels, so the visible bottom edge sits above the ground
            contact); the track is only called moving when enough of the
            object is visible for that explanation to fail.
        min_object_width_m: Objects narrower than this, judged from the
            outline's angular width at the estimated distance, are
            fragments or false detections and are dropped.
        object_height_m: Typical height of the class, used to decide how
            much of an object should be visible at a given distance.
        camera_height_m: Camera height above the ground.
        frame_sightings: Every sighting of the class per image id. When
            another detection sits directly below the track's outline in a
            frame (a car at the kerb in front of one in a driveway), the
            raised ground contact is explained by occlusion and the track is
            kept.

    Returns:
        tuple[ObjectEstimate | None, str]: The estimate (``None`` when the
        track is rejected) and a reason: ``'ok'``, ``'moving'``,
        ``'fragment'`` or ``'no_geometry'``.
    """
    sigma_rad = math.radians(bearing_sigma_deg)
    rays_by_image: dict[str, tuple[float, float, float]] = {}
    for o in members:
        if o.bearing is None:
            continue
        cx, cy = project(o.camera_lon, o.camera_lat)
        rays_by_image.setdefault(o.image_id, (cx, cy, float(o.bearing)))
    rays = list(rays_by_image.values())
    located = [o for o in members if o.has_location and o.range_m is not None]

    def physical_width(obs: Observation, distance: float) -> float:
        width_deg, _ = polygon_angular_size(obs)
        return 2.0 * distance * math.tan(math.radians(width_deg) / 2.0)

    def single_view() -> tuple[ObjectEstimate | None, str]:
        if not located:
            return None, 'no_geometry'
        widths = [physical_width(o, float(o.range_m or 0.0)) for o in located]
        if float(np.median(widths)) < min_object_width_m:
            return None, 'fragment'
        info = np.zeros((2, 2))
        vec = np.zeros(2)
        for o in located:
            p = np.array(project(*_lon_lat(o)))
            cov = single_view_covariance(
                float(o.bearing or 0.0), float(o.range_m or 0.0), bearing_sigma_deg
            )
            inv = np.linalg.inv(cov)
            info += inv
            vec += inv @ p
        cov = np.linalg.inv(info)
        point = cov @ vec
        return (
            ObjectEstimate(
                float(point[0]),
                float(point[1]),
                cov,
                list(members),
                'single_view',
                parallax_deg=_max_separation(b for _, _, b in rays),
            ),
            'ok',
        )

    if len(rays) < 2 or _max_separation(b for _, _, b in rays) < min_parallax_deg:
        return single_view()

    # RANSAC over ray pairs with enough separation. Every pair is tried for
    # short tracks; long tracks sample a fixed number of well-separated pairs
    # (a 150-frame track has 11,000 pairs, and a handful already finds the
    # consensus), which keeps the cost linear in the track length.
    n = len(rays)
    cams, dirs, normals = _ray_arrays(rays)
    bearings = np.asarray([b for _, _, b in rays], dtype=float)
    pairs = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if _angle_diff(float(bearings[i]), float(bearings[j])) >= min_parallax_deg
    ]
    if len(pairs) > RANSAC_MAX_PAIRS:
        pairs = random.Random(n).sample(pairs, RANSAC_MAX_PAIRS)
    best: tuple[int, list[int]] = (0, [])
    unit = np.ones(2)
    for i, j in pairs:
        fit = _solve_intersection(cams[[i, j]], normals[[i, j]], unit)
        if fit is None:
            continue
        perp, along = _residuals(fit[0], cams, dirs)
        dist = np.linalg.norm(fit[0] - cams, axis=1)
        tol = 2.5 * sigma_rad * dist + 0.3
        # Rays from far-away cameras still vote; the range bound is applied
        # to the final position, not to individual rays.
        mask = (perp <= tol) & (along > 0.0)
        count = int(mask.sum())
        if count > best[0]:
            best = (count, [int(k) for k in np.flatnonzero(mask)])
            if count == n:
                break
    count, inliers = best
    if count < 2 or _max_separation(rays[k][2] for k in inliers) < min_parallax_deg:
        return single_view()
    if count / n < inlier_fraction:
        if n >= 3:
            return None, 'moving'  # rays disagree: the object moved
        return single_view()

    fit = _weighted_intersection([rays[k] for k in inliers], sigma_rad)
    if fit is None:
        return single_view()
    point, cov = fit
    perp, along = _ray_residuals(point, [rays[k] for k in inliers])
    if np.any(along <= 0) or float(np.min(along)) > max_range_m:
        # Behind a camera, or farther than max_range_m from every camera
        # that saw it: not a position worth reporting.
        return single_view()
    rms = float(math.sqrt(float(np.mean(perp**2))))

    inlier_rays = {rays[k] for k in inliers}
    inlier_images = {
        image for image, ray in rays_by_image.items() if ray in inlier_rays
    }
    seen = [o for o in members if o.image_id in inlier_images and o.bearing is not None]
    distances = {}
    for o in seen:
        cx, cy = project(o.camera_lon, o.camera_lat)
        distances[o.image_id] = math.hypot(point[0] - cx, point[1] - cy)

    # Absolute ranges from the ground contact must agree with the distances
    # to the intersection (see ``range_ratio_bounds``):
    ratios = [
        o.range_m / distances[o.image_id]
        for o in seen
        if o.range_m is not None and distances[o.image_id] > 0
    ]
    if len(ratios) >= 2 and n >= 3:
        median = float(np.median(ratios))
        if median < range_ratio_bounds[0]:
            return None, 'moving'  # appears closer than the rays allow
        if median > range_ratio_bounds[1] and not _occlusion_explains(
            seen, camera_height_m, object_height_m, frame_sightings
        ):
            return None, 'moving'

    # Physical size: an outline that would be narrower than a real object at
    # this distance is a fragment or a false detection.
    widths = [physical_width(o, distances[o.image_id]) for o in seen]
    if widths and float(np.median(widths)) < min_object_width_m:
        return None, 'fragment'

    # Guard against over-confident covariances from near-parallel rays:
    cov = cov + np.eye(2) * 0.05**2
    return (
        ObjectEstimate(
            float(point[0]),
            float(point[1]),
            cov,
            list(members),
            'triangulated',
            rms_m=rms,
            parallax_deg=_max_separation(rays[k][2] for k in inliers),
        ),
        'ok',
    )


def _occlusion_explains(
    seen: Sequence[Observation],
    camera_height_m: float,
    object_height_m: float,
    frame_sightings: Mapping[str, Sequence[Observation]] | None,
) -> bool:
    """
    Can a hidden lower part explain ground-contact ranges that read too long?

    Two signs say yes. Another detection of the class sits directly below
    the outline in most frames (the occluder itself), or the outline is much
    shorter than a full object of ``object_height_m`` would be at the range
    its bottom edge implies (the bottom edge is not the ground contact). A
    moving vehicle in the open shows neither: nothing stands in front of it
    and its outline has the full height of an object at that range.
    """
    occluded_frames = 0
    visible = []
    for o in seen:
        x0, y0, x1, y1 = o.bbox
        others = (frame_sightings or {}).get(o.image_id, ())
        for p in others:
            if p is o:
                continue
            px0, py0, px1, py1 = p.bbox
            overlap = min(x1, px1) - max(x0, px0)
            if overlap <= 0.3 * max(x1 - x0, 1e-9):
                continue
            # The other outline starts no lower than our bottom edge (a
            # little tolerance) and reaches further down the image:
            if py0 <= y1 + 0.15 * (y1 - y0) and py1 > y1:
                occluded_frames += 1
                break
        if o.range_m is not None and o.range_m > 0:
            expected = math.degrees(
                math.atan2(camera_height_m, o.range_m)
                - math.atan2(camera_height_m - object_height_m, o.range_m)
            )
            if expected > 0:
                _, height_deg = polygon_angular_size(o)
                visible.append(height_deg / expected)
    if seen and occluded_frames / len(seen) >= 0.5:
        return True
    return bool(visible) and float(np.median(visible)) < MIN_VISIBLE_FRACTION


# --------------------------------------------------------------- merging
def merge_estimates(
    estimates: Sequence[ObjectEstimate],
    max_distance_m: float = 6.0,
    chi2: float = CHI2_2D_99,
    floor_m: float = 0.5,
) -> list[ObjectEstimate]:
    """
    Fuse estimates of the same object; keep different objects apart.

    Candidates are visited from the most to the least certain. A candidate
    joins an existing object when it lies within ``max_distance_m`` of it,
    passes the Mahalanobis gate under the combined covariance, and shares no
    frame with it (two detections in one frame are two objects). Positions
    are fused by information weighting, so a precise triangulation is not
    dragged around by a vague single view. ``floor_m`` is added to both
    covariances for the gate only: an object's apparent centre wanders by
    that much with the viewpoint, which a triangulation covariance does not
    know about.

    Example:
        >>> import numpy as np
        >>> a = ObjectEstimate(0.0, 0.0, np.eye(2) * 0.1, [], 'triangulated')
        >>> b = ObjectEstimate(0.4, 0.2, np.eye(2) * 0.5, [], 'single_view')
        >>> c = ObjectEstimate(5.0, 0.0, np.eye(2) * 0.1, [], 'triangulated')
        >>> [round(e.x, 2) for e in merge_estimates([a, b, c])]
        [0.07, 5.0]
    """
    ordered = sorted(estimates, key=lambda e: float(np.trace(e.cov)))
    merged: list[ObjectEstimate] = []
    frames: list[set[str]] = []
    cell = max(max_distance_m, 1.0)
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for est in ordered:
        gx, gy = int(math.floor(est.x / cell)), int(math.floor(est.y / cell))
        target = None
        best_d2 = chi2
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for k in grid.get((gx + dx, gy + dy), ()):
                    other = merged[k]
                    if math.hypot(other.x - est.x, other.y - est.y) > max_distance_m:
                        continue
                    if frames[k] & est.image_ids:
                        continue
                    delta = np.array([est.x - other.x, est.y - other.y])
                    gate_cov = other.cov + est.cov + np.eye(2) * (2 * floor_m**2)
                    try:
                        d2 = float(delta @ np.linalg.solve(gate_cov, delta))
                    except np.linalg.LinAlgError:
                        continue
                    if d2 < best_d2:
                        best_d2, target = d2, k
        if target is None:
            merged.append(
                ObjectEstimate(
                    est.x,
                    est.y,
                    est.cov.copy(),
                    list(est.members),
                    est.localization,
                    est.rms_m,
                    est.parallax_deg,
                )
            )
            frames.append(set(est.image_ids))
            grid[(gx, gy)].append(len(merged) - 1)
            continue
        other = merged[target]
        inv_a = np.linalg.inv(other.cov)
        inv_b = np.linalg.inv(est.cov)
        cov = np.linalg.inv(inv_a + inv_b)
        point = cov @ (
            inv_a @ np.array([other.x, other.y]) + inv_b @ np.array([est.x, est.y])
        )
        other.x, other.y, other.cov = float(point[0]), float(point[1]), cov
        other.members.extend(est.members)
        if est.localization == 'triangulated' and other.localization != 'triangulated':
            other.localization = 'triangulated'
        if est.rms_m is not None:
            other.rms_m = (
                est.rms_m if other.rms_m is None else max(other.rms_m, est.rms_m)
            )
        other.parallax_deg = max(other.parallax_deg, est.parallax_deg)
        frames[target] |= est.image_ids
    return merged


# --------------------------------------------------------------- voting
def vote_rays(
    observations: Sequence[Observation],
    project: Project,
    min_range_m: float = 2.0,
    max_range_m: float = 30.0,
    cell_m: float = 0.5,
    object_size_m: float = 4.5,
    min_score: float = 1.5,
) -> list[ObjectEstimate]:
    """
    Locate objects as peaks of accumulated ray votes on a ground grid.

    Every sighting spreads one unit of weight along its ray between
    ``min_range_m`` and ``max_range_m``, concentrated around its
    ground-contact range when it has one. Objects are local maxima of the
    smoothed grid, separated by at least ``object_size_m``; each sighting is
    then attributed to the nearest peak along its ray.

    Args:
        observations: Sightings with ``bearing`` set.
        project: Local metre projection.
        min_range_m: Nearest range a vote can land at.
        max_range_m: Farthest range a vote can land at.
        cell_m: Grid resolution.
        object_size_m: Non-maximum-suppression window.
        min_score: Smallest peak that counts as an object; about the number
            of sightings that must agree.

    Returns:
        list[ObjectEstimate]: One estimate per peak, ``localization``
        ``'voted'``.
    """
    rays = [
        (project(o.camera_lon, o.camera_lat), float(o.bearing), o)
        for o in observations
        if o.bearing is not None
    ]
    if not rays:
        return []
    xs = [c[0] for c, _, _ in rays]
    ys = [c[1] for c, _, _ in rays]
    x0, y0 = min(xs) - max_range_m, min(ys) - max_range_m
    nx = int(math.ceil((max(xs) + max_range_m - x0) / cell_m)) + 1
    ny = int(math.ceil((max(ys) + max_range_m - y0) / cell_m)) + 1
    grid = np.zeros((ny, nx))
    step = cell_m / 2.0
    samples = np.arange(min_range_m, max_range_m + step, step)
    for (cx, cy), bearing, obs in rays:
        d = _unit(bearing)
        if obs.range_m is not None:
            sigma = max(0.25 * obs.range_m, 1.0)
            w = np.exp(-0.5 * ((samples - obs.range_m) / sigma) ** 2)
        else:
            w = 1.0 / np.sqrt(samples)
        w = w / w.sum()
        px = cx + samples * d[0]
        py = cy + samples * d[1]
        ix = np.clip(((px - x0) / cell_m).astype(int), 0, nx - 1)
        iy = np.clip(((py - y0) / cell_m).astype(int), 0, ny - 1)
        np.add.at(grid, (iy, ix), w)
    # 3 x 3 box smoothing:
    padded = np.pad(grid, 1)
    smooth = np.zeros_like(grid)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            smooth += padded[1 + dy : ny + 1 + dy, 1 + dx : nx + 1 + dx]
    radius = max(int(round(object_size_m / cell_m / 2.0)), 1)
    peaks: list[tuple[float, int, int]] = []
    order = np.argsort(smooth, axis=None)[::-1]
    taken = np.zeros_like(smooth, dtype=bool)
    for flat in order:
        iy, ix = divmod(int(flat), nx)
        score = float(smooth[iy, ix])
        if score < min_score:
            break
        if taken[iy, ix]:
            continue
        peaks.append((score, iy, ix))
        taken[
            max(iy - radius, 0) : iy + radius + 1, max(ix - radius, 0) : ix + radius + 1
        ] = True
    estimates: list[ObjectEstimate] = []
    members: list[list[Observation]] = [[] for _ in peaks]
    centres = [
        (x0 + (ix + 0.5) * cell_m, y0 + (iy + 0.5) * cell_m) for _, iy, ix in peaks
    ]
    for (cx, cy), bearing, obs in rays:
        d = _unit(bearing)
        best, best_score = None, None
        for k, (px, py) in enumerate(centres):
            rel = np.array([px - cx, py - cy])
            along = float(rel @ d)
            if along < min_range_m or along > max_range_m:
                continue
            perp = float(np.hypot(*(rel - along * d)))
            if perp > object_size_m / 2.0:
                continue
            penalty = perp
            if obs.range_m is not None:
                penalty += abs(along - obs.range_m) * 0.25
            if best_score is None or penalty < best_score:
                best, best_score = k, penalty
        if best is not None:
            members[best].append(obs)
    for k, (px, py) in enumerate(centres):
        if not members[k]:
            continue
        estimates.append(
            ObjectEstimate(
                px,
                py,
                np.eye(2) * (object_size_m / 4.0) ** 2,
                members[k],
                'voted',
                rms_m=None,
                parallax_deg=_max_separation(
                    float(o.bearing) for o in members[k] if o.bearing is not None
                ),
            )
        )
    return estimates


# --------------------------------------------------------------- driver
def discover_objects(
    observations: Sequence[Observation],
    method: str = 'tracks',
    camera_height_m: float = 2.4,
    min_range_m: float = 2.0,
    max_range_m: float = DEFAULT_MAX_RANGE_M,
    min_area_fraction: float = MIN_POLYGON_AREA_FRACTION,
    bearing_sigma_deg: float = BEARING_SIGMA_DEG,
    min_parallax_deg: float = MIN_TRIANGULATION_SEPARATION_DEG,
    max_merge_m: float = 6.0,
    merge_floor_m: float = 0.5,
    track_gap_frames: int = 2,
    min_path_distance_m: float = 1.5,
    min_object_width_m: float = 1.0,
    object_height_m: float = 1.5,
    object_size_m: float = 4.5,
    vote_cell_m: float = 0.5,
    min_vote_score: float = 1.5,
) -> tuple[list[ObjectEstimate], dict[str, int], Unproject]:
    """
    Turn the sightings of one class into object estimates.

    Args:
        observations: Sightings of one class (any number of sequences).
        method: ``'tracks'`` for track-then-triangulate with covariance
            merging, ``'voting'`` for the ray-voting baseline.
        camera_height_m: Camera height above the ground, for single-view
            ranges.
        min_range_m: Closest plausible object.
        max_range_m: Farthest triangulated position reported.
        min_area_fraction: Outline area below which a detection is decoding
            noise; the default only removes garbage.
        bearing_sigma_deg: One-sigma bearing error.
        min_parallax_deg: Smallest bearing separation that counts as
            parallax.
        max_merge_m: Farthest two estimates can be and still be one object.
        merge_floor_m: Viewpoint-dependent centre wander added to the merge
            gate.
        track_gap_frames: Frames a track may miss before it is closed.
        min_path_distance_m: Objects closer than this to the line the camera
            drove are discarded: the survey vehicle passed through that spot,
            so whatever was seen there was moving (or is the vehicle itself).
        min_object_width_m: Objects narrower than this at their estimated
            distance are fragments or false detections.
        object_height_m: Typical height of the class, for the
            occlusion-aware moving check.
        object_size_m: Object footprint used by the voting baseline.
        vote_cell_m: Grid cell of the voting baseline.
        min_vote_score: Peak threshold of the voting baseline.

    Returns:
        tuple: ``(estimates, counts, unproject)`` where ``counts`` reports
        ``'sightings'``, ``'with_bearing'``, ``'tracks'``, ``'moving'``,
        ``'fragment'``, ``'on_path'`` and ``'objects'``, and ``unproject``
        converts local metres back to ``(lon, lat)``.
    """
    if method not in ('tracks', 'voting'):
        raise ValueError(f"method must be 'tracks' or 'voting', got {method!r}.")
    counts = {
        'sightings': len(observations),
        'with_bearing': 0,
        'tracks': 0,
        'moving': 0,
        'fragment': 0,
        'on_path': 0,
        'objects': 0,
    }
    usable: list[Observation] = []
    for obs in observations:
        # Sets bearing/elevation (and a range when the ground contact is
        # plausible); a sighting without a range still carries a ray.
        localize(
            obs,
            camera_height_m=camera_height_m,
            min_range_m=min_range_m,
            max_range_m=max_range_m,
            min_area_fraction=min_area_fraction,
        )
        if obs.bearing is not None:
            usable.append(obs)
    counts['with_bearing'] = len(usable)
    if not usable:
        return [], counts, lambda x, y: (0.0, 0.0)
    project, unproject = local_projection(usable[0].camera_lon, usable[0].camera_lat)

    if method == 'voting':
        estimates = vote_rays(
            usable,
            project,
            min_range_m=min_range_m,
            max_range_m=max_range_m,
            cell_m=vote_cell_m,
            object_size_m=object_size_m,
            min_score=min_vote_score,
        )
        counts['objects'] = len(estimates)
        return estimates, counts, unproject

    by_sequence: dict[str, list[Observation]] = defaultdict(list)
    for obs in usable:
        by_sequence[obs.sequence_id or '__none__'].append(obs)
    frame_sightings: dict[str, list[Observation]] = defaultdict(list)
    for obs in usable:
        frame_sightings[obs.image_id].append(obs)
    estimates = []
    for members in by_sequence.values():
        for track in track_sequence(members, project, max_gap_frames=track_gap_frames):
            counts['tracks'] += 1
            est, reason = classify_track(
                track,
                project,
                max_range_m=max_range_m,
                bearing_sigma_deg=bearing_sigma_deg,
                min_parallax_deg=min_parallax_deg,
                min_object_width_m=min_object_width_m,
                object_height_m=object_height_m,
                camera_height_m=camera_height_m,
                frame_sightings=frame_sightings,
            )
            if est is None:
                if reason in ('moving', 'fragment'):
                    counts[reason] += 1
                continue
            estimates.append(est)
    merged = merge_estimates(
        estimates, max_distance_m=max_merge_m, floor_m=merge_floor_m
    )
    if min_path_distance_m > 0:
        paths = camera_paths(usable, project)
        kept = []
        for est in merged:
            if distance_to_paths(est.x, est.y, paths) < min_path_distance_m:
                counts['on_path'] += 1
                continue
            kept.append(est)
        merged = kept
    counts['objects'] = len(merged)
    return merged, counts, unproject


def camera_paths(
    observations: Sequence[Observation], project: Project
) -> list[list[tuple[float, float]]]:
    """Camera positions per sequence in capture order, in local metres."""
    seen: dict[str, dict[str, tuple[str, tuple[float, float]]]] = defaultdict(dict)
    for o in observations:
        seen[o.sequence_id or '__none__'].setdefault(
            o.image_id, (o.captured_at or '', project(o.camera_lon, o.camera_lat))
        )
    paths = []
    for frames in seen.values():
        ordered = sorted(frames.items(), key=lambda kv: (kv[1][0], kv[0]))
        paths.append([xy for _, (_, xy) in ordered])
    return paths


def distance_to_paths(
    x: float, y: float, paths: Sequence[Sequence[tuple[float, float]]]
) -> float:
    """Shortest distance from a point to any camera path polyline."""
    best = math.inf
    for path in paths:
        if len(path) == 1:
            best = min(best, math.hypot(x - path[0][0], y - path[0][1]))
            continue
        for (ax, ay), (bx, by) in zip(path, path[1:], strict=False):
            vx, vy = bx - ax, by - ay
            length2 = vx * vx + vy * vy
            t = (
                0.0
                if length2 == 0
                else max(0.0, min(1.0, ((x - ax) * vx + (y - ay) * vy) / length2))
            )
            best = min(best, math.hypot(x - (ax + t * vx), y - (ay + t * vy)))
    return best


def estimate_attributes(est: ObjectEstimate) -> dict[str, Any]:
    """Attribute dictionary describing an estimate's quality."""
    out: dict[str, Any] = {
        'localization': est.localization,
        'n_images': est.n_images,
        'position_sigma_m': round(est.sigma_m, 2),
        'parallax_deg': round(est.parallax_deg, 1),
    }
    if est.rms_m is not None:
        out['position_rms_m'] = round(est.rms_m, 2)
    return out
