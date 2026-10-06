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
        rays: ``(camera_x, camera_y, bearing_deg)`` per frame, the geometry
            that is reliable regardless of range; used to associate
            fragments and duplicates by their lines of sight.
        size_m: Physical width of the object as seen (median over frames),
            or ``None`` when unknown.
    """

    x: float
    y: float
    cov: np.ndarray
    members: list[Observation]
    localization: str
    rms_m: float | None = None
    parallax_deg: float = 0.0
    rays: list[tuple[float, float, float]] = field(default_factory=list)
    size_m: float | None = None

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


# --------------------------------------------------------------- frame cleanup
def suppress_duplicate_sightings(
    observations: Sequence[Observation],
    overlap_fraction: float = 0.6,
) -> tuple[list[Observation], int]:
    """
    Drop a detector's duplicate outlines within each frame.

    Mapillary sometimes returns one object twice: a second outline nested in
    the first (a wheel, a window, the part of a car visible past a tree), or
    the two halves of a panorama's seam. Each would start a separate track
    that could never merge (two detections in one frame are normally two
    objects), so they are resolved here:

    * an outline whose box lies mostly (``overlap_fraction``) inside another
      of the same class is dropped, keeping the larger one;
    * on a panorama, an outline touching the left edge and one touching the
      right edge at the same height are one object cut by the seam: the
      smaller piece is dropped.

    Adjacent pieces are deliberately *not* joined: two cars parked side by
    side, or one behind another, look exactly like a split outline, and are
    far more common than one.

    Returns:
        tuple[list[Observation], int]: The cleaned sightings and how many
        were removed.
    """
    by_frame: dict[str, list[Observation]] = defaultdict(list)
    for o in observations:
        by_frame[o.image_id].append(o)
    kept: list[Observation] = []
    removed = 0
    for frame in by_frame.values():
        if len(frame) < 2:
            kept.extend(frame)
            continue
        frame = sorted(frame, key=lambda o: -o.area)
        alive: list[Observation | None] = list(frame)
        boxes = [o.bbox for o in frame]
        for i, big in enumerate(frame):
            if alive[i] is None:
                continue
            for j in range(i + 1, len(frame)):
                small = alive[j]
                if small is None or small.label != big.label:
                    continue
                ax0, ay0, ax1, ay1 = boxes[i]
                bx0, by0, bx1, by1 = boxes[j]
                inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(
                    0.0, min(ay1, by1) - max(ay0, by0)
                )
                area_small = max((bx1 - bx0) * (by1 - by0), 1e-12)
                if inter / area_small >= overlap_fraction or (
                    big.is_pano and _seam_halves(boxes[i], boxes[j])
                ):
                    alive[j] = None
                    removed += 1
        kept.extend(o for o in alive if o is not None)
    return kept, removed


def _seam_halves(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
    edge: float = 0.01,
) -> bool:
    """Two boxes on opposite edges of a panorama at the same height."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    on_edges = (ax0 <= edge and bx1 >= 1.0 - edge) or (
        bx0 <= edge and ax1 >= 1.0 - edge
    )
    if not on_edges:
        return False
    overlap = min(ay1, by1) - max(ay0, by0)
    return overlap >= 0.5 * min(ay1 - ay0, by1 - by0)


# --------------------------------------------------------------- tracking
def track_sequence(
    observations: Sequence[Observation],
    project: Project,
    max_gap_frames: int = 4,
    base_gate_deg: float = 6.0,
    gate_m: float = 2.5,
    gate_fraction: float = 0.10,
    loose_gate_deg: float = 45.0,
    min_object_width_m: float = 1.0,
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
        loose_gate_deg: Widest gate ever used, for a close object whose
            bearing swings quickly between frames.
        min_object_width_m: Narrowest real object of this class. With the
            outline's angular width it bounds how close the object can be,
            hence how far its bearing can move while the camera advances;
            a track without a reliable position is gated by that bound
            instead of ``loose_gate_deg``.

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
                # How far can a static object's bearing have moved? At most
                # the angle subtended by the camera's displacement at the
                # closest distance an object this narrow could be.
                last_cam = track.rays.get(last.image_id, cam)
                step = math.hypot(cam[0] - last_cam[0], cam[1] - last_cam[1])
                width_deg, _ = polygon_angular_size(last)
                nearest = min_object_width_m / math.tan(
                    math.radians(min(max(width_deg, 0.05), 89.0))
                )
                motion_gate = math.degrees(math.atan2(step + gate_m, max(nearest, 0.5)))
                motion_gate = min(loose_gate_deg, max(base_gate_deg, motion_gate))
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
                        gate = max(gate, motion_gate)
                else:
                    predicted = float(last.bearing or 0.0)
                    gate = motion_gate
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
        # The nearest frames carry the best ground-contact ranges; frames far
        # down the road would only drag the estimate along the ray.
        nearest = min(float(o.range_m or 0.0) for o in located)
        close = [o for o in located if float(o.range_m or 0.0) <= 1.5 * nearest + 1.0]
        info = np.zeros((2, 2))
        vec = np.zeros(2)
        for o in close:
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
                rays=list(rays),
                size_m=float(np.median(widths)),
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
            rays=[rays[k] for k in inliers],
            size_m=float(np.median(widths)) if widths else None,
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
def merge_by_rays(
    estimates: Sequence[ObjectEstimate],
    bearing_sigma_deg: float = BEARING_SIGMA_DEG,
    search_radius_m: float = 40.0,
    object_length_m: float = 5.0,
    weak_range_m: float = 25.0,
    weak_along_fraction: float = 0.3,
) -> list[ObjectEstimate]:
    """
    Merge estimates whose lines of sight point at the same object.

    Positions of single-view fragments can be metres off along the ray, so
    they are not compared by position at all: a fragment joins a
    triangulated object when its bearings all pass within tolerance of that
    object's position and in front of its cameras. The same back-check is
    applied to *weak* triangulations (see :func:`is_weak_estimate`: never
    seen closer than ``weak_range_m``, large uncertainty or little
    parallax): a car triangulated from 40 m down the street, with a tree
    hiding half of it, lands metres along its line of sight from where the
    close pass put it, so instead of comparing positions its rays are
    checked against the well-located objects they point at, and it joins
    the nearest such object within ``weak_along_fraction`` of its range
    along the ray. The strong object's position is kept. Two solid
    triangulated estimates merge when each one's rays pass through the
    other's position and they lie within half an object length, which
    reunites one car seen from two sides while keeping cars parked
    alongside each other apart.
    Estimates that share a frame are never merged (two detections in one
    image are two objects).

    Args:
        estimates: Candidates with ``rays`` set.
        bearing_sigma_deg: One-sigma bearing error.
        search_radius_m: Only pairs this close are examined.
        object_length_m: Typical length of the class; the tolerance uses
            the larger of this and the measured size.
        weak_range_m: Nearest sighting beyond which a triangulation is
            treated like a fragment.
        weak_along_fraction: How far along its line of sight, as a fraction
            of its nearest range (at least 4 m), a weak estimate may be from
            the object it joins.

    Returns:
        list[ObjectEstimate]: The merged candidates.
    """
    sigma_rad = math.radians(bearing_sigma_deg)
    weak = [is_weak_estimate(e, weak_range_m=weak_range_m) for e in estimates]
    order = sorted(
        range(len(estimates)),
        key=lambda i: (
            estimates[i].localization != 'triangulated',
            weak[i],
            float(np.trace(estimates[i].cov)),
        ),
    )
    merged: list[ObjectEstimate] = []
    merged_weak: list[bool] = []
    frames: list[set[str]] = []
    cell = search_radius_m
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)

    def passes(
        rays: Sequence[tuple[float, float, float]], target: ObjectEstimate, size: float
    ) -> bool:
        if not rays:
            return False
        cams, dirs, _ = _ray_arrays(rays)
        perp, along = _residuals(np.array([target.x, target.y]), cams, dirs)
        dist = np.linalg.norm(np.array([target.x, target.y]) - cams, axis=1)
        tol = 2.5 * sigma_rad * dist + 0.3 + 0.5 * size
        ok = (perp <= tol) & (along > 0.0)
        return float(np.mean(ok)) >= 0.7

    for i in order:
        est = estimates[i]
        est_weak = weak[i]
        gx, gy = int(math.floor(est.x / cell)), int(math.floor(est.y / cell))
        # Every candidate is scored and the closest match wins. Taking the
        # first rule that fires would let a car join its neighbour 3 m away
        # before reaching its own twin from another pass 0.7 m away, after
        # which the shared-frame rule can no longer keep the neighbours apart.
        target = None
        attached = False  # joined by its rays: keep the target's position
        best = (math.inf, math.inf)  # (rule rank, distance)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for k in grid.get((gx + dx, gy + dy), ()):
                    other = merged[k]
                    d = math.hypot(other.x - est.x, other.y - est.y)
                    if (
                        est.localization == 'triangulated'
                        and other.localization == 'triangulated'
                        and d < MIN_SEPARATION_M
                    ):
                        if (0, d) < best:  # one spot: one object
                            target, attached, best = k, False, (0, d)
                        continue
                    if frames[k] & est.image_ids:
                        continue
                    size = max(object_length_m, est.size_m or 0.0, other.size_m or 0.0)
                    if d > max(search_radius_m, size):
                        continue
                    if other.localization != 'triangulated':
                        continue
                    if est.localization == 'triangulated' and (
                        not est_weak or merged_weak[k]
                    ):
                        # Two solid estimates (or two weak ones from two far
                        # passes): each must look at the other's position and
                        # they must be within half an object length. Rays pass
                        # within metres of a neighbour parked alongside, so
                        # the distance, not the rays, keeps those apart.
                        if d > max(MIN_SEPARATION_M, 0.5 * size):
                            continue
                        if (
                            passes(est.rays, other, size)
                            and passes(other.rays, est, size)
                            and (1, d) < best
                        ):
                            target, attached, best = k, False, (1, d)
                    elif passes(est.rays, other, size):
                        if est.localization == 'triangulated':
                            # A weak triangulation may sit metres along its
                            # line of sight from the object; bound that, and
                            # prefer the nearest candidate (nose-to-tail cars).
                            nearest = nearest_range_m(est) or 0.0
                            if d > max(4.0, weak_along_fraction * nearest):
                                continue
                        if (2, d) < best:
                            target, attached, best = k, True, (2, d)
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
                    list(est.rays),
                    est.size_m,
                )
            )
            frames.append(set(est.image_ids))
            merged_weak.append(est_weak)
            grid[(gx, gy)].append(len(merged) - 1)
            continue
        other = merged[target]
        if est.localization == 'triangulated' and not attached:
            inv_a = np.linalg.inv(other.cov)
            inv_b = np.linalg.inv(est.cov)
            cov = np.linalg.inv(inv_a + inv_b)
            point = cov @ (
                inv_a @ np.array([other.x, other.y]) + inv_b @ np.array([est.x, est.y])
            )
            other.x, other.y, other.cov = float(point[0]), float(point[1]), cov
            if est.rms_m is not None:
                other.rms_m = (
                    est.rms_m if other.rms_m is None else max(other.rms_m, est.rms_m)
                )
        other.members.extend(est.members)
        other.rays.extend(est.rays)
        other.parallax_deg = max(other.parallax_deg, est.parallax_deg)
        if est.size_m is not None:
            other.size_m = max(other.size_m or 0.0, est.size_m)
        frames[target] |= est.image_ids
    return merged


#: Two estimates closer than this are one object whatever else is known:
#: two vehicles cannot occupy the same spot.
MIN_SEPARATION_M = 1.5
#: A weak object never seen up close, this near one that was, is a far
#: sighting of it or of its neighbour rather than a vehicle of its own.
FAR_NEIGHBOUR_M = 15.0


def _points_at_a_neighbour(
    est: ObjectEstimate,
    close_xy: Sequence[tuple[float, float]],
    project: Project,
    sigma_rad: float,
    object_size_m: float,
    share: float = 0.7,
) -> bool:
    """Whether most of ``est``'s closest rays pass through a close object nearby."""
    near = [
        (x, y)
        for x, y in close_xy
        if math.hypot(est.x - x, est.y - y) <= FAR_NEIGHBOUR_M
    ]
    if not near:
        return False
    rays = sorted(
        (o for o in est.members if o.bearing is not None and o.range_m is not None),
        key=lambda o: float(o.range_m or 0.0),
    )[:5]
    if not rays:
        return False
    for x, y in near:
        hits = 0
        for o in rays:
            geometry = _sighting_geometry(o, (x, y), project)
            if geometry is None:
                continue
            dist, along, across, _ = geometry
            if (
                along > 0
                and across <= 2.5 * sigma_rad * dist + 0.3 + 0.5 * object_size_m
            ):
                hits += 1
        if hits >= share * len(rays):
            return True
    return False


# ----------------------------------------------------------------- pieces
def _sighting_geometry(
    obs: Observation, target: tuple[float, float], project: Project
) -> tuple[float, float, float, np.ndarray] | None:
    """Distance, along- and across-ray offsets of ``target`` from a sighting's ray."""
    if obs.bearing is None or obs.camera_lon is None or obs.camera_lat is None:
        return None
    cam = np.array(project(obs.camera_lon, obs.camera_lat))
    direction = _unit(float(obs.bearing))
    offset = np.array(target) - cam
    along = float(offset @ direction)
    across = abs(float(offset[0] * direction[1] - offset[1] * direction[0]))
    return (
        float(np.linalg.norm(offset)),
        along,
        across,
        cam + direction * (obs.range_m or 0.0),
    )


def sighting_shows(
    obs: Observation,
    target: tuple[float, float],
    project: Project,
    bearing_sigma_deg: float = BEARING_SIGMA_DEG,
    size_m: float = 5.0,
    range_ratio_bounds: tuple[float, float] = (0.6, 1.6),
) -> bool:
    """
    Whether a sighting can be of an object at ``target``.

    The ray must pass within a quarter of the object's size of the target
    (plus bearing noise) with the target in front of the camera, and the
    ground-contact range, when there is one, must agree with the distance
    to the target within ``range_ratio_bounds``. The lateral tolerance is
    deliberately tighter than :func:`merge_by_rays` uses: a car parked
    alongside is 2.5 m across and must fail this test, while a piece of
    the object itself (half a car past a pole, its rear or front) is within
    about 1 m of the centre.

    Args:
        obs: The sighting (``bearing`` set).
        target: ``(x, y)`` in the local metre frame.
        project: Local metre projection.
        bearing_sigma_deg: One-sigma bearing error.
        size_m: Object size the lateral tolerance scales with.
        range_ratio_bounds: Accepted band for range over distance.

    Returns:
        bool: ``True`` when the sighting is consistent with the target.

    Example:
        >>> from rapidtools.core import Observation
        >>> from rapidtools.processing.street_localization import local_projection
        >>> project, _ = local_projection(-117.4, 47.7)
        >>> box = [(0.5, 0.5), (0.52, 0.5), (0.52, 0.55)]
        >>> o = Observation('i', 'car', box, -117.4, 47.7, 0.0, bearing=90.0)
        >>> o.range_m = 10.0
        >>> sighting_shows(o, (10.0, 0.5), project)
        True
        >>> sighting_shows(o, (10.0, 3.0), project)  # 3 m across: the car alongside
        False
    """
    geometry = _sighting_geometry(obs, target, project)
    if geometry is None:
        return False
    dist, along, across, _ = geometry
    sigma_rad = math.radians(bearing_sigma_deg)
    if along <= 0 or across > 2.5 * sigma_rad * dist + 0.3 + 0.25 * size_m:
        return False
    if obs.range_m is None or dist <= 0:
        return True
    return range_ratio_bounds[0] <= obs.range_m / dist <= range_ratio_bounds[1]


def merge_pieces(
    estimates: Sequence[ObjectEstimate],
    project: Project,
    bearing_sigma_deg: float = BEARING_SIGMA_DEG,
    object_length_m: float = 5.0,
    search_radius_m: float = 15.0,
    judge_range_m: float = 20.0,
    closest_k: int = 5,
    on_fraction: float = 0.6,
    evidence_fraction: float = 0.25,
    size_ratio: float = 1.3,
    solid_distance_m: float = 2.5,
    solid_parallax_deg: float = 45.0,
    solid_sigma_m: float = 0.25,
    far_fraction: float = 0.3,
    far_distance_m: float = 4.0,
) -> list[ObjectEstimate]:
    """
    Fold objects that are really pieces of a neighbour into that neighbour.

    Where several vehicles line up along the camera's line of sight (a
    driveway seen end-on, cars down the street) the tracker can slide from
    one to the next, and a detector can cut one car in two past a pole or
    into a front and a rear. The result is a second object a few metres
    from a well-located one, made of sightings that show the same vehicle.
    The two share frames, which the other merge stages take as proof of two
    objects, so they are judged here on what their sightings show instead.

    A less precise object ``B`` joins a more precise triangulated ``A``
    when:

    * most of ``B``'s rays pass near ``A`` (as :func:`merge_by_rays` tests),
    * ``B``'s ``closest_k`` sightings within ``judge_range_m``, the ones that
      define what ``B`` is, mostly (``on_fraction``) show ``A``: the ray
      passes within a quarter object size of ``A`` and the ground-contact
      range lands within ``max(2.5 m, 35 %)`` of ``A``'s position. This is
      decisive on its own, whatever ``B``'s own triangulation says: a
      low-parallax track lands metres from where its sightings point,
    * no more than ``evidence_fraction`` of those frames contradict it,
      where a contradiction is a frame in which ``A``'s own sighting shows
      ``A`` while ``B``'s does not (both sightings showing ``A`` are two
      pieces of one vehicle; ``A``'s sighting not showing ``A`` was the
      tracker astray and proves nothing). One such frame in which ``A`` was
      seen within ``judge_range_m`` settles it on its own: the vehicle is
      right there, large and unmistakable, and ``B``'s outline is
      something else,
    * ``B`` is not larger than ``size_ratio`` times ``A`` (or the class
      size): a motorhome behind a pickup lines up and fails only on size,
    * and, when ``B`` is solid (parallax over ``solid_parallax_deg`` and
      uncertainty under ``solid_sigma_m``) so that its own position is
      trustworthy, it lies within ``solid_distance_m`` of ``A``.

    A ``B`` never seen within ``judge_range_m`` has no dependable position
    of its own whatever its parallax: ground-contact ranges read from far
    away are off by a third, so its triangulation lands metres along the
    line of sight from the vehicle. Its closest sightings, wherever they
    are, define it: the ray test above runs over them rather than the whole
    track, whose far slivers may point anywhere. When their ground points
    do not land on ``A`` but their ranges fit the ``range_ratio_bounds``
    band, ``B`` still joins if its triangulation lies within
    ``far_fraction`` of its nearest range (at least ``far_distance_m``) of
    ``A``. It may only join an ``A`` that was itself seen within
    ``judge_range_m``: the close pass is the dependable one and keeps the
    position.

    ``A`` keeps its position; it gains the sightings of ``B`` that show it.

    Args:
        estimates: Merged candidates with ``members`` set.
        project: Local metre projection.
        bearing_sigma_deg: One-sigma bearing error.
        object_length_m: Typical size of the class.
        search_radius_m: Only pairs this close are examined.
        judge_range_m: Sightings farther than this carry ranges too noisy
            to judge and are ignored.
        closest_k: How many of ``B``'s closest sightings are judged.
        on_fraction: Share of them that must show ``A``.
        evidence_fraction: Share of contradicting frames that blocks the
            merge.
        size_ratio: Largest size of ``B`` relative to ``A`` (at least the
            class size) still taken for one object.
        solid_distance_m: Farthest a solid ``B`` may lie from ``A``.
        solid_parallax_deg: Parallax from which ``B`` counts as solid.
        solid_sigma_m: Uncertainty below which ``B`` counts as solid.
        far_fraction: How far, as a fraction of its nearest range, a ``B``
            never seen within ``judge_range_m`` may lie from ``A``.
        far_distance_m: Least distance allowed for such a ``B``.

    Returns:
        list[ObjectEstimate]: The remaining objects, in input order.

    Example:
        >>> merged = merge_pieces(estimates, project)  # doctest: +SKIP
    """
    sigma_rad = math.radians(bearing_sigma_deg)
    n = len(estimates)
    alive = [True] * n
    members = [list(e.members) for e in estimates]
    order = sorted(range(n), key=lambda i: -float(np.trace(estimates[i].cov)))
    cell = max(search_radius_m, 1.0)
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, est in enumerate(estimates):
        grid[(int(math.floor(est.x / cell)), int(math.floor(est.y / cell)))].append(i)

    def by_image(index: int) -> dict[str, Observation]:
        out: dict[str, Observation] = {}
        for o in members[index]:
            out.setdefault(o.image_id, o)
        return out

    for b in order:  # least precise first
        est = estimates[b]
        if not alive[b]:
            continue
        rays = [
            o for o in members[b] if o.bearing is not None and o.camera_lon is not None
        ]
        if not rays:
            continue
        ranged = sorted(
            (o for o in rays if o.range_m is not None and o.range_m > 0),
            key=lambda o: float(o.range_m or 0.0),
        )
        judged = [o for o in ranged if float(o.range_m or 0.0) <= judge_range_m]
        judged = judged[:closest_k]
        # One sighting just inside judge_range_m is no basis for the strict
        # close rules: a track with fewer than two is judged as far-only.
        far_only = len(judged) < 2
        if far_only:
            judged = ranged[:closest_k]
        if not judged:
            continue
        b_solid = (
            est.localization == 'triangulated'
            and est.parallax_deg >= solid_parallax_deg
            and est.sigma_m <= solid_sigma_m
        )
        b_size = est.size_m
        gx, gy = int(math.floor(est.x / cell)), int(math.floor(est.y / cell))
        target = None
        # Rank: share of judged sightings showing A, then how well A's distance
        # from the judged cameras matches their ground ranges (the tie-break
        # that tells the vehicle in front from the one behind along the same
        # line of sight, which the triangulated distance cannot for a far-only
        # B), then the triangulated distance.
        best = (-1.0, -math.inf, -math.inf)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for a in grid.get((gx + dx, gy + dy), ()):
                    other = estimates[a]
                    if (
                        a == b
                        or not alive[a]
                        or other.localization != 'triangulated'
                        or float(np.trace(other.cov)) > float(np.trace(est.cov))
                    ):
                        continue
                    d = math.hypot(other.x - est.x, other.y - est.y)
                    # A far-only B's own position is not to be trusted whatever
                    # its parallax, so the solid-distance bound is for close ones.
                    if d > search_radius_m or (
                        b_solid and not far_only and d > solid_distance_m
                    ):
                        continue
                    if (
                        far_only
                        and (nearest_range_m(other) or math.inf) > judge_range_m
                    ):
                        continue  # neither was seen up close: nothing to trust
                    # The measured size bounds how much larger B may be; the
                    # lateral tolerances use the class size, since outlines
                    # that merged with a neighbour inflate the measurement.
                    size = max(object_length_m, other.size_m or 0.0)
                    if b_size is not None and b_size > size_ratio * size + 0.5:
                        continue
                    point = (other.x, other.y)
                    # A far-only B is defined by its closest sightings; the far
                    # slivers of a blended track may point anywhere.
                    pointing = judged if far_only else rays
                    hits = 0
                    for o in pointing:
                        geometry = _sighting_geometry(o, point, project)
                        if geometry is None:
                            continue
                        dist, along, across, _ = geometry
                        if (
                            along > 0
                            and across
                            <= 2.5 * sigma_rad * dist + 0.3 + 0.5 * object_length_m
                        ):
                            hits += 1
                    if hits < 0.7 * len(pointing):
                        continue
                    # A piece may sit a quarter of the object from its centre;
                    # the end of a long vehicle cut by a tree, seen only from
                    # afar, half of it (a neighbour parked alongside would have
                    # close sightings of its own and not be far-only).
                    lateral = (0.5 if far_only else 0.25) * object_length_m
                    b_size_tol = object_length_m * (2.0 if far_only else 1.0)
                    on_a = 0  # ground point lands on A
                    in_band = 0  # ray on A, range within the band
                    log_ratios: list[float] = []
                    for o in judged:
                        geometry = _sighting_geometry(o, point, project)
                        if geometry is None:
                            continue
                        log_ratios.append(
                            abs(
                                math.log(
                                    float(o.range_m or 0.0) / max(geometry[0], 1e-6)
                                )
                            )
                        )
                        dist, along, across, foot = geometry
                        if (
                            along <= 0
                            or across > 2.5 * sigma_rad * dist + 0.3 + lateral
                        ):
                            continue
                        ratio = float(o.range_m or 0.0) / max(dist, 1e-6)
                        if 0.6 <= ratio <= 1.6:
                            in_band += 1
                        if math.hypot(foot[0] - point[0], foot[1] - point[1]) <= max(
                            2.5, 0.35 * float(o.range_m or 0.0)
                        ):
                            on_a += 1
                    share = on_a / len(judged)
                    if share < on_fraction:
                        if not far_only:
                            continue
                        # Ranges read from far away are off by a third, so for
                        # a far-only B the band suffices when its triangulation
                        # also lies within far_fraction of its range of A.
                        nearest = float(judged[0].range_m or 0.0)
                        share = in_band / len(judged)
                        if share < on_fraction or d > max(
                            far_distance_m, far_fraction * nearest
                        ):
                            continue
                    a_frames = by_image(a)
                    contradictions = 0
                    plain = False  # A seen up close while B's outline is elsewhere
                    for o in judged:
                        own = a_frames.get(o.image_id)
                        if own is None:
                            continue
                        if sighting_shows(
                            own, point, project, bearing_sigma_deg, object_length_m
                        ) and not sighting_shows(
                            o, point, project, bearing_sigma_deg, b_size_tol
                        ):
                            contradictions += 1
                            if (own.range_m or math.inf) <= judge_range_m:
                                plain = True
                    if plain or contradictions > evidence_fraction * len(judged):
                        continue
                    fit = -float(np.median(log_ratios)) if log_ratios else -math.inf
                    key = (share, fit, -d) if far_only else (share, -d, fit)
                    if key > best:
                        best, target = key, a
        if target is None:
            continue
        other = estimates[target]
        point = (other.x, other.y)
        size = object_length_m * (2.0 if far_only else 1.0)
        taken = [
            o
            for o in members[b]
            if o.image_id not in other.image_ids
            and sighting_shows(o, point, project, bearing_sigma_deg, size)
        ]
        members[target].extend(taken)
        other.members = members[target]
        other.parallax_deg = max(other.parallax_deg, est.parallax_deg)
        alive[b] = False
        logger.debug(
            f'Object at ({est.x:.1f}, {est.y:.1f}) is a piece of the one at '
            f'({other.x:.1f}, {other.y:.1f}): {len(taken)} sightings moved.'
        )
    kept = []
    for i, est in enumerate(estimates):
        if alive[i]:
            est.members = members[i]
            kept.append(est)
    return kept


# ------------------------------------------------------------ abeam first
def travel_headings(
    observations: Sequence[Observation], project: Project, min_step_m: float = 0.5
) -> dict[str, float]:
    """
    Direction of travel of the camera at each frame, per sequence.

    The heading is taken from the camera positions of the neighbouring
    frames in capture order (the previous and the next, or whichever
    exists), which is the direction the survey vehicle drove, independent
    of where the camera happened to point. Frames whose neighbours are
    closer than ``min_step_m`` (the vehicle was stopped) fall back to the
    camera's compass angle.

    Args:
        observations: Sightings with camera positions and sequence ids.
        project: Local metre projection.
        min_step_m: Shortest camera displacement that defines a direction.

    Returns:
        dict[str, float]: Travel heading in degrees clockwise from north,
        per image id.

    Example:
        >>> headings = travel_headings(sightings, project)  # doctest: +SKIP
        >>> round(headings['img-7'])  # doctest: +SKIP
        90
    """
    frames: dict[str, dict[str, tuple[float, float, float, str]]] = defaultdict(dict)
    for o in observations:
        if o.camera_lon is None or o.camera_lat is None:
            continue
        seq = o.sequence_id or '__none__'
        if o.image_id not in frames[seq]:
            x, y = project(o.camera_lon, o.camera_lat)
            frames[seq][o.image_id] = (
                x,
                y,
                float(o.compass_angle or 0.0),
                o.captured_at or '',
            )
    headings: dict[str, float] = {}
    for seq_frames in frames.values():
        ordered = sorted(seq_frames.items(), key=lambda kv: (kv[1][3], kv[0]))
        for i, (image_id, (x, y, compass, _)) in enumerate(ordered):
            before = ordered[i - 1][1] if i > 0 else None
            after = ordered[i + 1][1] if i + 1 < len(ordered) else None
            x0, y0 = (before[0], before[1]) if before else (x, y)
            x1, y1 = (after[0], after[1]) if after else (x, y)
            if math.hypot(x1 - x0, y1 - y0) >= min_step_m:
                headings[image_id] = math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360.0
            else:
                headings[image_id] = compass
    return headings


def split_by_view(
    observations: Sequence[Observation],
    headings: Mapping[str, float],
    window_deg: float = 60.0,
) -> tuple[list[Observation], list[Observation]]:
    """
    Separate the sightings made while passing an object from the rest.

    A sighting is *passing* when its bearing lies within ``window_deg`` of
    abeam, that is between ``90 - window_deg`` and ``90 + window_deg``
    degrees off the direction of travel on either side. Sightings in the
    cones ahead and behind are *approach* sightings: their rays are nearly
    parallel to the road, so they cannot place an object, and they are
    where tracks slide between vehicles that line up down the street.

    Args:
        observations: Sightings with ``bearing`` set.
        headings: Travel heading per image id (see :func:`travel_headings`).
        window_deg: Half-width of the passing window about abeam.

    Returns:
        tuple[list[Observation], list[Observation]]: Passing and approach
        sightings. A sighting whose frame has no heading counts as passing.

    Example:
        >>> from rapidtools.core import Observation
        >>> box = [(0.5, 0.5), (0.52, 0.5), (0.52, 0.55)]
        >>> abeam = Observation('a', 'car', box, 0.0, 0.0, 0.0, bearing=80.0)
        >>> ahead = Observation('b', 'car', box, 0.0, 0.0, 0.0, bearing=5.0)
        >>> p, q = split_by_view([abeam, ahead], {'a': 0.0, 'b': 0.0})
        >>> [o.image_id for o in p], [o.image_id for o in q]
        (['a'], ['b'])
    """
    passing: list[Observation] = []
    approach: list[Observation] = []
    for o in observations:
        heading = headings.get(o.image_id)
        if heading is None or o.bearing is None:
            passing.append(o)
            continue
        rel = abs((float(o.bearing) - heading + 180.0) % 360.0 - 180.0)
        if 90.0 - window_deg <= rel <= 90.0 + window_deg:
            passing.append(o)
        else:
            approach.append(o)
    return passing, approach


def attach_approach_sightings(
    estimates: Sequence[ObjectEstimate],
    approach: Sequence[Observation],
    project: Project,
    bearing_sigma_deg: float = BEARING_SIGMA_DEG,
    object_length_m: float = 5.0,
    range_ratio_bounds: tuple[float, float] = (0.6, 1.6),
) -> int:
    """
    Give placed objects the approach sightings whose rays point at them.

    An approach sighting joins the triangulated object its ray passes
    within half an object of, in front of the camera, whose distance agrees
    with the sighting's ground-contact range within ``range_ratio_bounds``
    when there is one. Among several such objects, lined up down the road,
    the one whose distance best fits the range wins. Positions are not
    touched: the sighting only adds to the object's record, so its crops
    and frame count are complete. Sightings that point at nothing placed
    are left out.

    Args:
        estimates: Placed objects; their ``members`` lists are extended.
        approach: Sightings from the cones ahead and behind.
        project: Local metre projection.
        bearing_sigma_deg: One-sigma bearing error.
        object_length_m: Typical size of the class.
        range_ratio_bounds: Accepted band for range over distance.

    Returns:
        int: Number of sightings attached.

    Example:
        >>> n = attach_approach_sightings(objects, approach, project)  # doctest: +SKIP
    """
    targets = [e for e in estimates if e.localization == 'triangulated']
    if not targets or not approach:
        return 0
    xy = np.array([[e.x, e.y] for e in targets], dtype=float)
    sigma_rad = math.radians(bearing_sigma_deg)
    attached = 0
    for o in approach:
        if o.bearing is None or o.camera_lon is None or o.camera_lat is None:
            continue
        cam = np.array(project(o.camera_lon, o.camera_lat))
        direction = _unit(float(o.bearing))
        offset = xy - cam
        along = offset @ direction
        across = np.abs(offset[:, 0] * direction[1] - offset[:, 1] * direction[0])
        dist = np.linalg.norm(offset, axis=1)
        ok = (along > 0) & (
            across <= 2.5 * sigma_rad * dist + 0.3 + 0.5 * object_length_m
        )
        if o.range_m is not None:
            ratio = float(o.range_m) / np.maximum(dist, 1e-6)
            ok &= (ratio >= range_ratio_bounds[0]) & (ratio <= range_ratio_bounds[1])
            score = np.abs(np.log(np.maximum(ratio, 1e-6)))
        else:
            score = dist
        candidates = np.flatnonzero(ok)
        if len(candidates) == 0:
            continue
        best = candidates[int(np.argmin(score[candidates]))]
        targets[best].members.append(o)
        attached += 1
    return attached


# --------------------------------------------------------------- witnesses
@dataclass
class CameraFrame:
    """
    Where one survey frame was taken and what it could see.

    One record per frame, including frames with no detection of the class:
    those are the ones that can say an object is *not* somewhere.

    Attributes:
        image_id: Mapillary image ID.
        lon, lat: Camera position.
        compass_angle: Camera heading in degrees clockwise from north.
        is_pano: Whether the frame is a 360-degree panorama.
        captured_at: ISO timestamp (the date part identifies the survey day).
        sequence_id: Mapillary sequence.
        focal_norm: Focal length over the larger image side (OpenSfM), for
            the horizontal field of view of a perspective camera.
    """

    image_id: str
    lon: float
    lat: float
    compass_angle: float
    is_pano: bool = True
    captured_at: str | None = None
    sequence_id: str | None = None
    focal_norm: float | None = None

    @property
    def day(self) -> str | None:
        """The survey day (``YYYY-MM-DD``), if the capture time is known."""
        return self.captured_at[:10] if self.captured_at else None

    @property
    def horizontal_fov_deg(self) -> float:
        """Horizontal field of view; 360 for a panorama."""
        if self.is_pano:
            return 360.0
        focal = self.focal_norm if self.focal_norm else 0.85
        return 2.0 * math.degrees(math.atan(0.5 / focal))

    def sees(self, bearing: float, margin_deg: float = 5.0) -> bool:
        """Whether a ground point at ``bearing`` lies inside the frame."""
        if self.is_pano:
            return True
        return _angle_diff(bearing, self.compass_angle) <= (
            self.horizontal_fov_deg / 2.0 - margin_deg
        )

    @classmethod
    def from_observation(cls, obs: Observation) -> CameraFrame:
        """The frame record of a sighting's camera."""
        params = (obs.extra or {}).get('camera_parameters')
        focal = float(params[0]) if params and params[0] else None
        return cls(
            obs.image_id,
            obs.camera_lon,
            obs.camera_lat,
            float(obs.compass_angle or 0.0),
            obs.is_pano,
            obs.captured_at,
            obs.sequence_id,
            focal,
        )


def nearest_range_m(est: ObjectEstimate) -> float | None:
    """The closest ground-contact range among an estimate's sightings."""
    ranges = [o.range_m for o in est.members if o.range_m is not None]
    return min(ranges) if ranges else None


def is_weak_estimate(
    est: ObjectEstimate,
    weak_range_m: float = 25.0,
    weak_sigma_m: float = 1.0,
    weak_parallax_deg: float = 15.0,
) -> bool:
    """
    Whether an estimate's position should not be trusted on its own.

    Single views are weak by definition. A triangulation is weak when it
    was never seen closer than ``weak_range_m`` (the rays are then nearly
    parallel and a bearing bias from partial occlusion moves the
    intersection metres along the line of sight, far more than its
    covariance admits), when its reported uncertainty exceeds
    ``weak_sigma_m``, or when its parallax is below ``weak_parallax_deg``.
    """
    if est.localization != 'triangulated':
        return True
    nearest = nearest_range_m(est)
    if nearest is None or nearest > weak_range_m:
        return True
    return est.sigma_m > weak_sigma_m or est.parallax_deg < weak_parallax_deg


def prune_unwitnessed(
    estimates: Sequence[ObjectEstimate],
    frames: Sequence[CameraFrame],
    frame_sightings: Mapping[str, Sequence[Observation]],
    project: Project,
    witness_radius_m: float = 12.0,
    min_witnesses: int = 2,
    object_size_m: float = 4.5,
    min_distance_m: float = 1.5,
    bearing_margin_deg: float = 3.0,
    weak_only: bool = True,
) -> tuple[list[ObjectEstimate], int]:
    """
    Drop estimates that nearby frames should have seen but did not.

    An object's position is otherwise built only from the frames in which
    the detector fired. The frames in which it did not fire are evidence
    too: if a camera passed within ``witness_radius_m`` of the estimated
    position on the same survey day, with that position in its field of
    view, and that frame holds no detection of the class anywhere near the
    predicted bearing, the object is not there. A far single-view estimate
    is typically metres off along its ray, and this is what catches it when
    the survey happened to drive past the spot it was placed on.

    A frame counts as a positive witness when any detection of the class
    lies within the object's angular size of the predicted bearing (an
    object parked in front of it counts too, since it could hide the
    estimate). Votes are tallied per survey day, and an
    estimate is dropped only when, on every day that has witnesses, at
    least ``min_witnesses`` negative ones exist and they outnumber the
    positive ones; a vehicle seen up close on one day is reported even if
    it had gone by a later survey day. The
    frames the estimate was built from count as positive witnesses when
    they lie within the radius, so an object seen up close in one pass
    survives a later pass that day in which it had gone. Frames from other
    days (the vehicle may have left) and frames closer than
    ``min_distance_m`` are not witnesses.

    Args:
        estimates: Candidates after merging.
        frames: Every frame of the survey, including those with no sighting.
        frame_sightings: Sightings of the class per image ID.
        project: Local metre projection shared by the caller.
        witness_radius_m: Farthest a camera can be and still count; at this
            distance a vehicle is unmissable for the detector.
        min_witnesses: Negative frames required.
        object_size_m: Object footprint, for the bearing tolerance.
        min_distance_m: Closest a witness may be (nearer ones are on the path).
        bearing_margin_deg: Extra bearing tolerance for pose error.
        weak_only: Judge only estimates whose position is not trustworthy
            on its own (see :func:`is_weak_estimate`). A car triangulated
            from a close pass is real whatever a later pass shows: driveway
            cars are routinely hidden from the next pass by a hedge or a
            car in front, and the detector itself misses some.

    Returns:
        tuple[list[ObjectEstimate], int]: The kept estimates and the number
        dropped.
    """
    if not frames or witness_radius_m <= 0:
        return list(estimates), 0
    cams = np.array([project(f.lon, f.lat) for f in frames], dtype=float)
    cell = witness_radius_m
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (x, y) in enumerate(cams):
        grid[(int(math.floor(x / cell)), int(math.floor(y / cell)))].append(i)
    kept: list[ObjectEstimate] = []
    dropped = 0
    for est in estimates:
        if weak_only and not is_weak_estimate(est):
            kept.append(est)
            continue
        days = {o.captured_at[:10] for o in est.members if o.captured_at}
        own = est.image_ids
        votes: dict[str | None, list[int]] = defaultdict(lambda: [0, 0])
        gx, gy = int(math.floor(est.x / cell)), int(math.floor(est.y / cell))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for i in grid.get((gx + dx, gy + dy), ()):
                    frame = frames[i]
                    if days and frame.day not in days:
                        continue
                    cx, cy = cams[i]
                    dist = math.hypot(est.x - cx, est.y - cy)
                    if dist > witness_radius_m or dist < min_distance_m:
                        continue
                    bearing = _bearing_to(cx, cy, est.x, est.y)
                    if not frame.sees(bearing):
                        continue
                    tol = (
                        math.degrees(math.atan2(object_size_m / 2.0 + 1.0, dist))
                        + bearing_margin_deg
                    )
                    # The frames the estimate was built from saw it by
                    # definition; they vote for it, so a car seen from 3 m in
                    # one pass is not vetoed by a later pass after it left.
                    seen = frame.image_id in own or any(
                        o.bearing is not None
                        and _angle_diff(float(o.bearing), bearing) <= tol
                        for o in frame_sightings.get(frame.image_id, ())
                    )
                    votes[frame.day][0 if seen else 1] += 1
        # Tally per survey day: a vehicle seen up close on one day is real
        # even if it had gone by the time of a later survey day.
        contradicted = bool(votes) and all(
            neg >= min_witnesses and neg > pos for pos, neg in votes.values()
        )
        if contradicted:
            dropped += 1
        else:
            kept.append(est)
    return kept, dropped


def merge_estimates(
    estimates: Sequence[ObjectEstimate],
    max_distance_m: float = 6.0,
    chi2: float = CHI2_2D_99,
    floor_m: float = 0.5,
    min_separation_m: float = MIN_SEPARATION_M,
    max_solid_separation_m: float = 2.5,
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
    know about. Two solid triangulations (see :func:`is_weak_estimate`) are
    never fused when more than ``max_solid_separation_m`` apart: a precise
    position from one pass and another from a second pass that disagree by
    more than a car's width are two cars parked next to each other, one of
    them hidden in each pass, not one car (the halves of a long vehicle are
    reunited earlier by :func:`merge_by_rays`).

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
    solid: list[bool] = []
    cell = max(max_distance_m, 1.0)
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for est in ordered:
        gx, gy = int(math.floor(est.x / cell)), int(math.floor(est.y / cell))
        est_solid = not is_weak_estimate(est)
        target = None
        best_d2 = chi2
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for k in grid.get((gx + dx, gy + dy), ()):
                    other = merged[k]
                    separation = math.hypot(other.x - est.x, other.y - est.y)
                    if separation > max_distance_m:
                        continue
                    if separation < min_separation_m:
                        target, best_d2 = k, -1.0  # physically the same spot
                        break
                    if frames[k] & est.image_ids:
                        continue
                    if est_solid and solid[k] and separation > max_solid_separation_m:
                        continue
                    delta = np.array([est.x - other.x, est.y - other.y])
                    # A long vehicle's apparent centre wanders along its body
                    # with the viewpoint, so the floor grows with its size:
                    floor = max(
                        floor_m, 0.2 * max(est.size_m or 0.0, other.size_m or 0.0)
                    )
                    gate_cov = other.cov + est.cov + np.eye(2) * (2 * floor**2)
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
                    list(est.rays),
                    est.size_m,
                )
            )
            frames.append(set(est.image_ids))
            solid.append(est_solid)
            grid[(gx, gy)].append(len(merged) - 1)
            continue
        other = merged[target]
        other.rays.extend(est.rays)
        if est.size_m is not None:
            other.size_m = max(other.size_m or 0.0, est.size_m)
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
        solid[target] = solid[target] or est_solid
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
    merge_pieces_of_neighbours: bool = True,
    close_range_m: float = 20.0,
    far_min_parallax_deg: float = 45.0,
    far_max_sigma_m: float = 0.5,
    min_sighting_deg: float = 6.0,
    report_single_views: bool = False,
    abeam_window_deg: float = 0.0,
    min_path_distance_m: float = 1.5,
    min_object_width_m: float = 1.0,
    object_height_m: float = 1.5,
    object_size_m: float = 4.5,
    vote_cell_m: float = 0.5,
    min_vote_score: float = 1.5,
    max_single_view_range_m: float = 30.0,
    frames: Sequence[CameraFrame] | None = None,
    witness_radius_m: float = 12.0,
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
        merge_pieces_of_neighbours: After the position merge, fold objects
            whose closest sightings show a better-located neighbour into
            that neighbour (see :func:`merge_pieces`): the second track a
            detector's split outline or a tracker sliding between vehicles
            lined up along the line of sight leaves behind.
        close_range_m: Range within which an object must have been seen to
            be trusted. Ground-contact ranges read from farther away are
            off by a third and the tracker hops between vehicles lined up
            down the road, so an object never seen within this range whose
            closest rays point at an object within 15 m that was seen up
            close is a far sighting of that object and is dropped, unless
            it is well triangulated regardless: parallax of at least
            ``far_min_parallax_deg`` and uncertainty of at most
            ``far_max_sigma_m``. One whose rays point elsewhere, a car in a
            driveway behind the kerb, stays. Also the range the piece
            merge judges sightings by. ``0`` keeps every object.
        far_min_parallax_deg: Parallax that lets a far-only object through.
        far_max_sigma_m: Uncertainty that lets a far-only object through.
        min_sighting_deg: An object must have been seen at least once with
            an outline this wide or tall, in degrees of the panorama (6
            degrees is 34 pixels on a 2048-pixel thumbnail and 225 on the
            original). What was never more than a speck cannot be
            identified by anyone and is not reported; a drive-by survey
            sees anything worth counting properly in some frame. The specks
            still take part in tracking, where they lend parallax. ``0``
            keeps every object.
        report_single_views: Keep objects placed from ground-contact ranges
            alone, without parallax. Their positions are guesses and, on
            the surveys tried, none was worth keeping; off by default.
        abeam_window_deg: Objects are built only from the sightings made
            while passing them, within this many degrees of abeam on either
            side of the direction of travel (see :func:`split_by_view`).
            A drive-by survey passes everything worth counting, and the
            passing frames carry the widest parallax and the largest
            outlines; the frames looking ahead or behind down the road
            have rays nearly parallel to it, cannot place anything, and are
            where tracks slide between vehicles that line up. Those
            sightings are attached afterwards to the objects their rays
            point at (:func:`attach_approach_sightings`), so records stay
            complete, but they never create an object. Off (``0``) by
            default: on the Spokane survey a 75-degree window lost 17 of
            382 verified vehicles, because the track classification (the
            moving-vehicle, parallax and size tests) was tuned on tracks
            that include the approach, and a pass alone trips them; with
            the far-sighting gates in place it removed few duplicates in
            return. Kept as an option for when the classifier is reworked
            for passes alone.
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
        max_single_view_range_m: A single-view position whose nearest
            sighting is farther than this is not reported: at that distance
            the ground-contact range is a guess, not a measurement. Such
            far fragments still take part in merging first, so a far view of
            a well-located object joins it rather than being lost. ``0``
            disables the cut.
        frames: Every frame of the survey as :class:`CameraFrame` records,
            including frames with no sighting of the class. When given,
            estimates that nearby same-day frames should have seen but did
            not are dropped (see :func:`prune_unwitnessed`).
        witness_radius_m: Radius of that check; ``0`` disables it.

    Returns:
        tuple: ``(estimates, counts, unproject)`` where ``counts`` reports
        ``'sightings'``, ``'with_bearing'``, ``'duplicates'`` (detector
        outlines resolved within a frame), ``'tracks'``, ``'moving'``,
        ``'fragment'``, ``'on_path'``, ``'far_single_view'``,
        ``'unwitnessed'`` and ``'objects'``, and ``unproject`` converts
        local metres back to ``(lon, lat)``.
    """
    if method not in ('tracks', 'voting'):
        raise ValueError(f"method must be 'tracks' or 'voting', got {method!r}.")
    counts = {
        'sightings': len(observations),
        'with_bearing': 0,
        'tracks': 0,
        'duplicates': 0,
        'moving': 0,
        'fragment': 0,
        'on_path': 0,
        'pieces': 0,
        'far_only': 0,
        'indiscernible': 0,
        'single_view': 0,
        'approach': 0,
        'attached': 0,
        'far_single_view': 0,
        'unwitnessed': 0,
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
    usable, duplicates = suppress_duplicate_sightings(usable)
    counts['duplicates'] = duplicates
    project, unproject = local_projection(usable[0].camera_lon, usable[0].camera_lat)
    frame_sightings: dict[str, list[Observation]] = defaultdict(list)
    for obs in usable:
        frame_sightings[obs.image_id].append(obs)

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

    approach: list[Observation] = []
    tracked = usable
    if abeam_window_deg > 0:
        headings = travel_headings(usable, project)
        tracked, approach = split_by_view(usable, headings, abeam_window_deg)
        counts['approach'] = len(approach)
    by_sequence: dict[str, list[Observation]] = defaultdict(list)
    for obs in tracked:
        by_sequence[obs.sequence_id or '__none__'].append(obs)
    estimates = []
    for members in by_sequence.values():
        for track in track_sequence(
            members,
            project,
            max_gap_frames=track_gap_frames,
            min_object_width_m=min_object_width_m,
        ):
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
    estimates = merge_by_rays(
        estimates, bearing_sigma_deg=bearing_sigma_deg, object_length_m=object_size_m
    )
    merged = merge_estimates(
        estimates, max_distance_m=max_merge_m, floor_m=merge_floor_m
    )
    if merge_pieces_of_neighbours:
        before = len(merged)
        merged = merge_pieces(
            merged,
            project,
            bearing_sigma_deg=bearing_sigma_deg,
            object_length_m=object_size_m,
            judge_range_m=close_range_m if close_range_m > 0 else 20.0,
        )
        counts['pieces'] = before - len(merged)
    if close_range_m > 0:
        # A weak object never seen within close_range_m whose closest rays
        # point at an object within FAR_NEIGHBOUR_M that was, is a far
        # sighting of that object: the close pass already counted it. One
        # pointing elsewhere, a car in a driveway behind the kerb, stays.
        close_xy = [
            (e.x, e.y)
            for e in merged
            if (nearest_range_m(e) or math.inf) <= close_range_m
        ]
        kept = []
        for est in merged:
            nearest = nearest_range_m(est)
            far_weak = (nearest is None or nearest > close_range_m) and not (
                est.localization == 'triangulated'
                and est.parallax_deg >= far_min_parallax_deg
                and est.sigma_m <= far_max_sigma_m
            )
            if far_weak and _points_at_a_neighbour(
                est, close_xy, project, math.radians(bearing_sigma_deg), object_size_m
            ):
                counts['far_only'] += 1
                continue
            kept.append(est)
        merged = kept
    if min_path_distance_m > 0:
        paths = camera_paths(usable, project)
        kept = []
        for est in merged:
            if distance_to_paths(est.x, est.y, paths) < min_path_distance_m:
                counts['on_path'] += 1
                continue
            kept.append(est)
        merged = kept
    if max_single_view_range_m > 0:
        kept = []
        for est in merged:
            if est.localization == 'single_view':
                ranges = [o.range_m for o in est.members if o.range_m is not None]
                if not ranges or min(ranges) > max_single_view_range_m:
                    counts['far_single_view'] += 1
                    continue
            kept.append(est)
        merged = kept
    if not report_single_views:
        kept = [est for est in merged if est.localization != 'single_view']
        counts['single_view'] = len(merged) - len(kept)
        merged = kept
    if min_sighting_deg > 0:
        kept = []
        for est in merged:
            largest = max(
                (max(polygon_angular_size(o)) for o in est.members), default=0.0
            )
            if largest < min_sighting_deg:
                counts['indiscernible'] += 1
                continue
            kept.append(est)
        merged = kept
    if frames and witness_radius_m > 0:
        merged, unwitnessed = prune_unwitnessed(
            merged,
            frames,
            frame_sightings,
            project,
            witness_radius_m=witness_radius_m,
            object_size_m=object_size_m,
            min_distance_m=max(min_path_distance_m, 1.0),
        )
        counts['unwitnessed'] = unwitnessed
    if approach:
        counts['attached'] = attach_approach_sightings(
            merged,
            approach,
            project,
            bearing_sigma_deg=bearing_sigma_deg,
            object_length_m=object_size_m,
        )
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
