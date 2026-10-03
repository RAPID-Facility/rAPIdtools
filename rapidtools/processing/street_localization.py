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
# Contributors:
# Barbaros Cetiner
#
# Last updated:
# 09-29-2026

"""
Geometry for placing street-level detections on the ground.

A detection in a street-level image is a polygon in normalised image
coordinates. Knowing where the camera stood and looked, the polygon's centre
column gives the bearing to the object and its lowest row gives the elevation
of the ground contact, from which the camera height yields a range. One view
gives a rough point; several views of the same object from different camera
positions give a much better one by intersecting the bearings. This module
provides those steps plus the two filters every street survey needs: dropping
the collection vehicle itself (it sits at the same place in every frame of a
sequence) and thinning frames that are only a metre or two apart.

Example:
    >>> from rapidtools.core import Observation
    >>> from rapidtools.processing.street_localization import localize
    >>> obs = Observation(
    ...     image_id='1', label='object--vehicle--car',
    ...     polygon=[(0.70, 0.55), (0.76, 0.55), (0.76, 0.62), (0.70, 0.62)],
    ...     camera_lon=-117.41, camera_lat=47.66, compass_angle=0.0,
    ... )
    >>> localize(obs, camera_height_m=2.4)
    True
    >>> round(obs.bearing), round(obs.range_m, 1)
    (83, 6.1)
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from rapidtools.core import Observation

EARTH_RADIUS_M = 6_371_000.0

# A detection whose lowest edge is less than this far below the horizon has
# no usable ground contact (the range would be hundreds of metres or worse):
MIN_DEPRESSION_DEG = 1.0

# Bearings closer than this cannot be intersected reliably:
MIN_TRIANGULATION_SEPARATION_DEG = 8.0


# --------------------------------------------------------------- basics
def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Great-circle distance between two WGS84 points in metres.

    Example:
        >>> round(haversine_m(0.0, 0.0, 0.0, 0.001))
        111
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlam = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def destination_point(
    lon: float, lat: float, bearing_deg: float, distance_m: float
) -> tuple[float, float]:
    """
    Point reached from ``(lon, lat)`` travelling ``distance_m`` along a bearing.

    Example:
        >>> lon, lat = destination_point(0.0, 0.0, 0.0, 111_195)
        >>> round(lon, 6), round(lat, 3)
        (0.0, 1.0)
    """
    delta = distance_m / EARTH_RADIUS_M
    theta = math.radians(bearing_deg)
    phi1, lam1 = math.radians(lat), math.radians(lon)
    phi2 = math.asin(
        math.sin(phi1) * math.cos(delta)
        + math.cos(phi1) * math.sin(delta) * math.cos(theta)
    )
    lam2 = lam1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2),
    )
    return math.degrees(lam2), math.degrees(phi2)


def local_projection(
    lon0: float, lat0: float
) -> tuple[
    Callable[[float, float], tuple[float, float]],
    Callable[[float, float], tuple[float, float]],
]:
    """
    Return ``(project, unproject)`` for a local east/north metre frame.

    Uses the equirectangular approximation centred on ``(lon0, lat0)``, which
    is accurate to well under a metre over the few kilometres a survey covers.

    Example:
        >>> project, unproject = local_projection(-117.4, 47.66)
        >>> x, y = project(-117.4, 47.661)
        >>> round(y), round(x)
        (111, 0)
        >>> [round(v, 4) for v in unproject(x, y)]
        [-117.4, 47.661]
    """
    cos_lat = math.cos(math.radians(lat0))

    def project(lon: float, lat: float) -> tuple[float, float]:
        dlon = (lon - lon0 + 180.0) % 360.0 - 180.0
        return (
            math.radians(dlon) * cos_lat * EARTH_RADIUS_M,
            math.radians(lat - lat0) * EARTH_RADIUS_M,
        )

    def unproject(x: float, y: float) -> tuple[float, float]:
        return (
            lon0 + math.degrees(x / (EARTH_RADIUS_M * cos_lat)),
            lat0 + math.degrees(y / EARTH_RADIUS_M),
        )

    return project, unproject


def bbox_iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """
    Intersection over union of two ``(min_x, min_y, max_x, max_y)`` boxes.

    Example:
        >>> bbox_iou((0, 0, 1, 1), (0.5, 0, 1.5, 1))
        0.3333333333333333
    """
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------- single view
def view_angles(
    polygon: Sequence[tuple[float, float]],
    compass_angle: float,
    is_pano: bool = True,
    focal_norm: float | None = None,
    aspect: float = 2.0,
) -> tuple[float, float]:
    """
    Bearing to a detection and elevation of its ground contact.

    Args:
        polygon: Detection outline in normalised image coordinates (``y``
            down).
        compass_angle: Heading of the image centre column, degrees clockwise
            from north.
        is_pano: Equirectangular panorama (360 x 180 degrees). For panoramas
            the horizontal centre is a circular mean so objects straddling
            the left/right seam are handled.
        focal_norm: Focal length divided by image width, for perspective
            images. Defaults to ``0.85`` when unknown.
        aspect: ``width / height`` of a perspective image.

    Returns:
        tuple[float, float]: ``(bearing_deg, elevation_deg)``. The bearing is
        absolute (0-360); the elevation is negative below the horizon.

    Example:
        >>> bearing, elevation = view_angles(
        ...     [(0.7, 0.5), (0.8, 0.5), (0.8, 0.6), (0.7, 0.6)], compass_angle=10.0
        ... )
        >>> round(bearing), round(elevation)
        (100, -18)
    """
    xs = [p[0] for p in polygon]
    y_bottom = max(p[1] for p in polygon)
    if is_pano:
        angles = [2 * math.pi * x for x in xs]
        cx = (
            math.atan2(sum(map(math.sin, angles)), sum(map(math.cos, angles)))
            / (2 * math.pi)
        ) % 1.0
        rel_bearing = (cx - 0.5) * 360.0
        elevation = (0.5 - y_bottom) * 180.0
    else:
        f = focal_norm if focal_norm else 0.85
        cx = sum(xs) / len(xs)
        rel_bearing = math.degrees(math.atan((cx - 0.5) / f))
        elevation = -math.degrees(math.atan((y_bottom - 0.5) / aspect / f))
    return (compass_angle + rel_bearing) % 360.0, elevation


def ground_range(elevation_deg: float, camera_height_m: float) -> float | None:
    """
    Horizontal distance to a ground contact seen ``elevation_deg`` below level.

    Returns ``None`` when the contact is at or above the horizon (within
    :data:`MIN_DEPRESSION_DEG`), where no range can be inferred.

    Example:
        >>> round(ground_range(-45.0, 2.4), 2)
        2.4
        >>> ground_range(0.5, 2.4) is None
        True
    """
    if elevation_deg > -MIN_DEPRESSION_DEG:
        return None
    return camera_height_m / math.tan(math.radians(-elevation_deg))


def localize(
    obs: Observation,
    camera_height_m: float = 2.4,
    min_range_m: float = 2.0,
    max_range_m: float = 30.0,
    min_area_fraction: float = 0.0004,
) -> bool:
    """
    Fill in the single-view ground position of an observation.

    Sets ``bearing``, ``elevation``, ``range_m``, ``lon`` and ``lat`` on
    ``obs`` and returns ``True`` when the geometry is plausible. Detections
    that are too small, have no ground contact, or fall outside the range
    window are left without a position and ``False`` is returned.

    Args:
        obs: The observation to update in place.
        camera_height_m: Height of the camera above the ground.
        min_range_m: Closest plausible object (the ego vehicle and the road
            surface sit inside this radius).
        max_range_m: Farthest object worth keeping; ranges beyond a few tens
            of metres are dominated by height and slope errors.
        min_area_fraction: Minimum polygon area as a fraction of the image.
    """
    if obs.area < min_area_fraction:
        return False
    focal_norm = None
    params = obs.extra.get('camera_parameters')
    if params:
        focal_norm = float(params[0])
    aspect = 2.0
    if obs.image_width and obs.image_height:
        aspect = obs.image_width / obs.image_height
    bearing, elevation = view_angles(
        obs.polygon, obs.compass_angle, obs.is_pano, focal_norm, aspect
    )
    obs.bearing, obs.elevation = bearing, elevation
    rng = ground_range(elevation, camera_height_m)
    if rng is None or not (min_range_m <= rng <= max_range_m):
        return False
    obs.range_m = rng
    obs.lon, obs.lat = destination_point(obs.camera_lon, obs.camera_lat, bearing, rng)
    return True


# --------------------------------------------------------------- multi view
def intersect_bearings(
    rays: Sequence[tuple[float, float, float]],
    project: Callable[[float, float], tuple[float, float]],
    unproject: Callable[[float, float], tuple[float, float]],
    max_range_m: float = 60.0,
    min_separation_deg: float = MIN_TRIANGULATION_SEPARATION_DEG,
) -> tuple[float, float, float] | None:
    """
    Least-squares intersection of bearings from several camera positions.

    Args:
        rays: ``(camera_lon, camera_lat, bearing_deg)`` per observation.
        project: Local metre projection from :func:`local_projection`.
        unproject: Its inverse.
        max_range_m: Reject solutions farther than this from any camera.
        min_separation_deg: Require at least one pair of bearings this far
            apart; near-parallel rays give an ill-conditioned intersection.

    Returns:
        tuple[float, float, float] | None: ``(lon, lat, rms_residual_m)`` or
        ``None`` when the rays cannot be intersected (fewer than two distinct
        cameras, near-parallel bearings, or a solution behind a camera).

    Example:
        >>> project, unproject = local_projection(0.0, 0.0)
        >>> # Two cameras 20 m apart both looking at a point 10 m north of the
        >>> # midpoint between them:
        >>> lon_a, lat_a = unproject(-10.0, 0.0)
        >>> lon_b, lat_b = unproject(10.0, 0.0)
        >>> lon, lat, rms = intersect_bearings(
        ...     [(lon_a, lat_a, 45.0), (lon_b, lat_b, 315.0)], project, unproject
        ... )
        >>> [round(v) for v in project(lon, lat)], round(rms, 3)
        ([0, 10], 0.0)
    """
    cams = [project(lon, lat) for lon, lat, _ in rays]
    bearings = [math.radians(b) for _, _, b in rays]
    # Need at least two cameras that are not (nearly) the same position:
    distinct = {(round(x, 1), round(y, 1)) for x, y in cams}
    if len(distinct) < 2:
        return None
    degs = [b for _, _, b in rays]
    spread = max(
        min(abs(a - b) % 360.0, 360.0 - abs(a - b) % 360.0) for a in degs for b in degs
    )
    if spread < min_separation_deg:
        return None
    # Each ray: point p_i, unit direction d_i = (sin b, cos b) in east/north.
    # Minimise sum of squared perpendicular distances: solve (sum (I - d d^T)) x
    # = sum (I - d d^T) p.
    a_sum = np.zeros((2, 2))
    b_sum = np.zeros(2)
    for (x, y), b in zip(cams, bearings, strict=True):
        d = np.array([math.sin(b), math.cos(b)])
        m = np.eye(2) - np.outer(d, d)
        a_sum += m
        b_sum += m @ np.array([x, y])
    try:
        point = np.linalg.solve(a_sum, b_sum)
    except np.linalg.LinAlgError:
        return None
    residuals = []
    for (x, y), b in zip(cams, bearings, strict=True):
        d = np.array([math.sin(b), math.cos(b)])
        rel = point - np.array([x, y])
        along = float(rel @ d)
        if along <= 0 or along > max_range_m:
            return None  # behind the camera or implausibly far
        residuals.append(float(np.hypot(*(rel - along * d))))
    lon, lat = unproject(float(point[0]), float(point[1]))
    rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    return lon, lat, rms


# --------------------------------------------------------------- filters
def estimate_ego_mask(
    observations: Iterable[Observation],
    min_recurrence: float = 0.5,
    iou_threshold: float = 0.5,
    min_frames: int = 3,
) -> set[int]:
    """
    Find detections of the collection vehicle itself.

    A parked car drifts across the frame as the camera drives past; the
    vehicle carrying the camera does not. Within each sequence, a detection
    whose box recurs (IoU above ``iou_threshold``) in at least
    ``min_recurrence`` of the frames is flagged as ego. Sequences with fewer
    than ``min_frames`` frames are left alone, since recurrence cannot be
    judged.

    Args:
        observations: Observations of one label, in any order.
        min_recurrence: Fraction of a sequence's frames the box must recur in.
        iou_threshold: Box overlap that counts as "the same place".
        min_frames: Minimum frames in a sequence before flagging anything.

    Returns:
        set[int]: Indices into ``observations`` (in iteration order) that
        belong to the ego vehicle.

    Example:
        >>> from rapidtools.core import Observation
        >>> ego = [(0.40, 0.80), (0.60, 0.80), (0.60, 1.00), (0.40, 1.00)]
        >>> frames = [
        ...     Observation(str(i), 'car', ego, 0.0, 0.0, 0.0, sequence_id='s')
        ...     for i in range(4)
        ... ]
        >>> box = [(0.1, 0.5), (0.2, 0.5), (0.2, 0.6), (0.1, 0.6)]
        >>> parked = Observation('9', 'car', box, 0.0, 0.0, 0.0, sequence_id='s')
        >>> sorted(estimate_ego_mask(frames + [parked]))
        [0, 1, 2, 3]
    """
    obs_list = list(observations)
    by_sequence: dict[str, list[int]] = defaultdict(list)
    for i, obs in enumerate(obs_list):
        by_sequence[obs.sequence_id or '__none__'].append(i)

    ego: set[int] = set()
    for indices in by_sequence.values():
        frames = {obs_list[i].image_id for i in indices}
        if len(frames) < min_frames:
            continue
        # The survey vehicle sits at the same place in every frame, so its
        # boxes are near-identical. Group boxes that round to the same
        # position first and compare groups, not sightings: a sequence of
        # thousands of frames then costs one comparison per distinct box
        # instead of one per pair, which made city-wide runs quadratic.
        quantum = 0.01
        groups: dict[tuple[int, int, int, int], dict] = {}
        for i in indices:
            box = obs_list[i].bbox
            key = (
                int(round(box[0] / quantum)),
                int(round(box[1] / quantum)),
                int(round(box[2] / quantum)),
                int(round(box[3] / quantum)),
            )
            group = groups.get(key)
            if group is None:
                group = groups[key] = {
                    'box': list(box),
                    'n': 0,
                    'frames': set(),
                    'members': [],
                }
            n = group['n']
            group['box'] = [(group['box'][k] * n + box[k]) / (n + 1) for k in range(4)]
            group['n'] = n + 1
            group['frames'].add(obs_list[i].image_id)
            group['members'].append(i)
        # Bucket the group boxes by their centre so each is compared with few others:
        cell = 0.05
        buckets: dict[tuple[int, int], list[tuple[int, int, int, int]]] = defaultdict(
            list
        )
        for key, group in groups.items():
            b = group['box']
            buckets[
                (int((b[0] + b[2]) / 2 / cell), int((b[1] + b[3]) / 2 / cell))
            ].append(key)
        for group in groups.values():
            b = group['box']
            cx, cy = int((b[0] + b[2]) / 2 / cell), int((b[1] + b[3]) / 2 / cell)
            matching_frames: set[str] = set()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for other in buckets.get((cx + dx, cy + dy), ()):
                        if bbox_iou(b, groups[other]['box']) >= iou_threshold:
                            matching_frames |= groups[other]['frames']
            if len(matching_frames) / len(frames) >= min_recurrence:
                ego.update(group['members'])
    return ego


def simplify_polygon(
    polygon: Sequence[tuple[float, float]], tolerance: float
) -> list[tuple[float, float]]:
    """
    Reduce a detection outline to the vertices that matter at ``tolerance``.

    Mapillary segmentation outlines trace every pixel step of a mask, so a
    single car can carry hundreds of vertices. Only the bounding box, the
    bottom edge and the horizontal centre feed the localisation, and a crop
    outline needs no sub-pixel detail, so the ring is simplified with
    Douglas-Peucker before it is stored. The bounding box of the result is
    within ``tolerance`` of the original's.

    Args:
        polygon: Exterior ring in normalised image coordinates.
        tolerance: Largest allowed deviation, in normalised units (``0.002``
            is about four pixels of a 2048-wide image). ``0`` disables the
            simplification.

    Returns:
        list[tuple[float, float]]: The simplified ring (at least three
        vertices), or the input unchanged when simplification fails.

    Example:
        >>> ring = [(0.1, 0.1), (0.2, 0.1001), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)]
        >>> simplify_polygon(ring, 0.002)
        [(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)]
    """
    points = [(float(x), float(y)) for x, y in polygon]
    if tolerance <= 0 or len(points) <= 4:
        return points
    try:
        from shapely.geometry import Polygon

        simplified = Polygon(points).simplify(tolerance, preserve_topology=True)
        if simplified.is_empty or simplified.geom_type != 'Polygon':
            return points
        ring = [(float(x), float(y)) for x, y in simplified.exterior.coords[:-1]]
    except Exception:  # noqa: BLE001 - provider outlines can be degenerate
        return points
    return ring if len(ring) >= 3 else points


def thin_frames(
    frames: Sequence[tuple[str, float, float, str | None, str | None]],
    spacing_m: float,
) -> set[str]:
    """
    Keep one frame per ``spacing_m`` metres along each sequence.

    Args:
        frames: ``(image_id, lon, lat, captured_at, sequence_id)`` tuples.
            Frames are ordered by ``captured_at`` within a sequence.
        spacing_m: Minimum distance between kept frames; ``0`` keeps all.

    Returns:
        set[str]: The image IDs to keep.

    Example:
        >>> frames = [('a', 0.0, 0.0, '1', 's'), ('b', 0.00003, 0.0, '2', 's'),
        ...           ('c', 0.0002, 0.0, '3', 's')]
        >>> sorted(thin_frames(frames, spacing_m=10))
        ['a', 'c']
    """
    if spacing_m <= 0:
        return {f[0] for f in frames}
    by_sequence: dict[str, list] = defaultdict(list)
    for frame in frames:
        by_sequence[frame[4] or frame[0]].append(frame)
    keep: set[str] = set()
    for seq in by_sequence.values():
        seq.sort(key=lambda f: (f[3] or '', f[0]))
        last = None
        for image_id, lon, lat, _, _ in seq:
            if last is None or haversine_m(last[0], last[1], lon, lat) >= spacing_m:
                keep.add(image_id)
                last = (lon, lat)
    return keep


def _ground_position(obs: Observation) -> tuple[float, float]:
    """Return ``(lon, lat)`` of an observation that :func:`localize` accepted."""
    if obs.lon is None or obs.lat is None:
        raise ValueError(f'Observation {obs.image_id!r} has no ground position.')
    return obs.lon, obs.lat


def cluster_observations(
    observations: Sequence[Observation], radius_m: float
) -> list[list[Observation]]:
    """
    Group located observations that are within ``radius_m`` of each other.

    Greedy single-linkage clustering seeded from the closest views first:
    each observation joins the first existing cluster whose running centre
    lies within the radius, otherwise it starts a new one.

    Args:
        observations: Observations with ``lon``/``lat`` set (others are
            skipped).
        radius_m: Two views closer than this are the same object.

    Returns:
        list[list[Observation]]: One list of observations per object.

    Example:
        >>> from rapidtools.core import Observation
        >>> def obs(i, lon, rng):
        ...     o = Observation(str(i), 'car', [(0, 0), (1, 0), (1, 1)], 0.0, 0.0, 0.0)
        ...     o.lon, o.lat, o.range_m = lon, 0.0, rng
        ...     return o
        >>> views = [obs(1, 0.0, 5), obs(2, 0.00001, 6), obs(3, 0.001, 5)]
        >>> groups = cluster_observations(views, 4.0)
        >>> [len(g) for g in groups]
        [2, 1]
    """
    located = [o for o in observations if o.has_location]
    if not located:
        return []
    project, _ = local_projection(*_ground_position(located[0]))
    grid: dict[tuple[int, int], list[dict]] = defaultdict(list)
    clusters: list[dict] = []
    for obs in sorted(
        located, key=lambda o: o.range_m if o.range_m is not None else 1e9
    ):
        x, y = project(*_ground_position(obs))
        gx, gy = int(math.floor(x / radius_m)), int(math.floor(y / radius_m))
        target = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for cluster in grid.get((gx + dx, gy + dy), ()):
                    if math.hypot(cluster['x'] - x, cluster['y'] - y) <= radius_m:
                        target = cluster
                        break
                if target:
                    break
            if target:
                break
        if target is None:
            target = {'x': x, 'y': y, 'members': []}
            clusters.append(target)
            grid[(gx, gy)].append(target)
        target['members'].append(obs)
        n = len(target['members'])
        target['x'] += (x - target['x']) / n
        target['y'] += (y - target['y']) / n
    return [c['members'] for c in clusters]
