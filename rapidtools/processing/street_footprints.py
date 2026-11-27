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
#
# Last updated:
# 11-26-2026

"""
Sightings of buildings whose footprints are already known.

Where a footprint layer exists, a street survey does not have to discover
the buildings: it only has to say which outline in which image shows which
footprint. That is a matter of projection. Every footprint is raised to a
box and projected into each frame near it; nearer boxes hide farther ones;
what is left is the part of the image where the building should appear.
An outline that lies there is a sighting of that building.

This needs neither a point cloud nor a decision about which outlines
belong together: two masks that a segmentation model draws on the wings of
one house fall on the same footprint, and two neighbors joined by a fence
in a detector's building surface fall on two. It works with either kind of
outline:

- **One outline per object** (``extra['instance']`` set, as a segmentation
  model gives): an outline is a sighting of the footprint that holds most
  of it.
- **A surface outline** (Mapillary's building class): the part of the
  surface inside a footprint's region is its sighting, when it fills
  enough of the region.

Camera poses come from the images' own coordinates and rotation; the
ground is taken flat, ``camera_height_m`` below each camera, and the boxes
are tall, so that neither a height attribute nor the terrain has to be
right. A footprint a few meters off its true place is still matched, with
a looser fit.

Example:
    >>> from rapidtools.processing.street_footprints import footprint_sightings
    >>> views, unclaimed, counts = footprint_sightings(  # doctest: +SKIP
    ...     outlines, footprints)
    >>> views[0].n_expected, views[0].n_seen, len(views[0].members)  # doctest: +SKIP
    (86, 72, 12)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree
from shapely import affinity
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from rapidtools.core import Observation

from .street_localization import local_projection, rotation_matrix
from .street_structures import project_points, spread_views

logger = logging.getLogger(__name__)

Project = Callable[[float, float], tuple[float, float]]
Unproject = Callable[[float, float], tuple[float, float]]


@dataclass
class FootprintViews:
    """
    What a survey saw of one known footprint.

    Attributes:
        members: The sightings kept: one per frame, the outline of the
            building in that frame.
        n_expected: Frames in which the building should be visible: within
            range, and not hidden behind a nearer footprint.
        n_seen: Frames among those in which an outline was found on it. A
            building expected in many frames and seen in none is hidden
            (by trees, say) or no longer there.
    """

    members: list[Observation] = field(default_factory=list)
    n_expected: int = 0
    n_seen: int = 0

    @property
    def n_images(self) -> int:
        """Number of distinct frames among the sightings kept."""
        return len({o.image_id for o in self.members})


def footprint_region(
    footprint: Polygon,
    rotation: np.ndarray,
    camera: np.ndarray,
    base_m: float,
    top_m: float,
) -> tuple[Polygon | None, float]:
    """
    The part of a panorama a footprint raised to a box projects onto.

    Args:
        footprint: Footprint in the local meter frame.
        rotation: ``3 x 3`` world-to-camera rotation matrix.
        camera: Camera center (east, north, altitude).
        base_m: Altitude of the foot of the box.
        top_m: Altitude of its top.

    Returns:
        tuple[Polygon | None, float]: The convex outline of the box in
        normalized image coordinates and the distance from the camera to
        the nearest corner. The outline of a box across the left/right
        seam of the panorama runs past 1 on the right. ``None`` when the
        box projects onto a line.

    Example:
        >>> import numpy as np
        >>> house = Polygon([(-5, 10), (5, 10), (5, 20), (-5, 20)])
        >>> level = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])  # looking north
        >>> region, nearest = footprint_region(house, level, np.zeros(3), -2.0, 6.0)
        >>> round(region.centroid.x, 2), round(nearest, 1)
        (0.5, 10.2)
    """
    corners = np.asarray(footprint.exterior.coords[:-1], dtype=float)
    edge: list[np.ndarray] = []
    for a, b in zip(corners, np.roll(corners, -1, axis=0), strict=True):
        # a long wall keeps its shape when points along it are projected too
        n = max(2, int(np.hypot(*(b - a)) // 2.0) + 1)
        edge.append(a + (b - a) * np.linspace(0.0, 1.0, n, endpoint=False)[:, None])
    xy = np.vstack(edge)
    world = np.vstack(
        [np.c_[xy, np.full(len(xy), base_m)], np.c_[xy, np.full(len(xy), top_m)]]
    )
    u, v, dist = project_points(world - camera, rotation, np.zeros(3))
    if u.max() - u.min() > 0.5:
        u = np.where(u < 0.5, u + 1.0, u)
    ring = np.c_[u, v]
    try:
        hull = Polygon(ring[ConvexHull(ring).vertices])
    except QhullError:
        return None, float(dist.min())
    return (hull if hull.area > 0 else None), float(dist.min())


def footprint_sightings(
    observations: Sequence[Observation],
    footprints: Sequence[Polygon],
    *,
    camera_height_m: float = 2.4,
    max_range_m: float = 40.0,
    box_height_m: float = 9.0,
    min_share: float = 0.5,
    min_cover: float = 0.2,
    min_visible: float = 0.2,
    max_views: int = 12,
    project: Project | None = None,
) -> tuple[list[FootprintViews], list[int], dict[str, int]]:
    """
    Say which outlines show which known footprints.

    See the module docstring for the method.

    Args:
        observations: Outlines of the class in panoramas with a
            ``computed_rotation`` in ``extra``. Outlines of an image that
            all carry ``extra['instance']`` are one outline per object;
            otherwise they are taken for a surface.
        footprints: Footprint polygons in WGS84 longitude and latitude.
        camera_height_m: Camera height above the ground.
        max_range_m: Farthest footprint a frame is asked about.
        box_height_m: Height the footprints are raised to. Generous on
            purpose: a box taller than the building only makes the match
            depend on where the building stands along the street.
        min_share: Share of a per-object outline that must lie inside a
            footprint's region to be its sighting.
        min_cover: Share of a footprint's region a surface outline must
            fill to be its sighting. The region is that of a tall box, so a
            one-story house fills only part of it: on a 1 km test area 0.2
            found 323 of 324 houses facing the street and 0.3 only 298,
            with hardly more sightings that a per-object mask disputes.
        min_visible: Share of a footprint's box that must stay clear of
            nearer boxes for the frame to be asked about it.
        max_views: Sightings kept per footprint, spread over the
            directions it was seen from; ``0`` keeps one per frame.
        project: Projection to local meters; made from the sightings when
            omitted.

    Returns:
        tuple[list[FootprintViews], list[int], dict[str, int]]: For every
        footprint what was seen of it, the positions in ``observations`` of
        the per-object outlines that lie on no footprint (buildings the
        layer does not have, or objects too far to tell), and counters
        describing the run.

    Example:
        >>> views, unclaimed, counts = footprint_sightings(  # doctest: +SKIP
        ...     outlines, footprints, max_views=3)
        >>> counts['footprints_seen'], counts['footprints']  # doctest: +SKIP
        (326, 428)
    """
    counts: dict[str, int] = defaultdict(int)
    counts['sightings'] = len(observations)
    counts['footprints'] = len(footprints)
    views = [FootprintViews() for _ in footprints]
    if project is None:
        lon0 = float(np.mean([o.camera_lon for o in observations] or [0.0]))
        lat0 = float(np.mean([o.camera_lat for o in observations] or [0.0]))
        project, _ = local_projection(lon0, lat0)
    local = [
        _valid(Polygon([project(x, y) for x, y in f.exterior.coords]))
        for f in footprints
    ]

    by_image: dict[str, list[int]] = defaultdict(list)
    for number, obs in enumerate(observations):
        if (obs.extra or {}).get('computed_rotation') is None or not obs.is_pano:
            counts['unusable'] += 1
            continue
        by_image[obs.image_id].append(number)
    counts['frames'] = len(by_image)
    if not by_image or not local:
        counts['footprints_seen'] = 0
        return views, [], dict(counts)

    centers = np.array([[f.centroid.x, f.centroid.y] for f in local])
    reach = max(math.sqrt(f.area) for f in local) + max_range_m
    tree = cKDTree(centers)
    found: list[list[tuple[float, Observation]]] = [[] for _ in footprints]
    unclaimed: list[int] = []
    for image_id, numbers in by_image.items():
        base = observations[numbers[0]]
        x, y = project(base.camera_lon, base.camera_lat)
        camera = np.array([x, y, camera_height_m])
        rotation = rotation_matrix(
            np.asarray(base.extra['computed_rotation'], dtype=float)
        )
        # nearer boxes hide farther ones
        boxes = []
        for j in tree.query_ball_point(camera[:2], reach):
            if local[j].distance(Point(x, y)) > max_range_m:
                continue
            region, nearest = footprint_region(
                local[j], rotation, camera, -1.0, box_height_m
            )
            if region is not None:
                boxes.append((nearest, j, region))
        boxes.sort(key=lambda box: box[0])
        hidden: BaseGeometry | None = None
        regions: list[tuple[int, BaseGeometry]] = []
        for _, j, region in boxes:
            clear = region if hidden is None else region.difference(hidden)
            if not clear.is_empty and clear.area >= min_visible * region.area:
                regions.append((j, clear))
                views[j].n_expected += 1
            hidden = region if hidden is None else hidden.union(region)
        shapes = [_valid(Polygon(observations[n].polygon)) for n in numbers]
        per_object = all('instance' in (observations[n].extra or {}) for n in numbers)
        seen: dict[int, list[BaseGeometry]] = defaultdict(list)
        if per_object:
            masks: dict[Any, list[int]] = defaultdict(list)
            for k, n in enumerate(numbers):
                masks[observations[n].extra['instance']].append(k)
            for parts in masks.values():
                mask = unary_union([shapes[k] for k in parts])
                best, share = -1, 0.0
                if not mask.is_empty and mask.area > 0:
                    for j, clear in regions:
                        inside = _overlap(mask, clear) / mask.area
                        if inside > share:
                            best, share = j, inside
                if share >= min_share:
                    seen[best].append(mask)
                else:
                    unclaimed.extend(numbers[k] for k in parts)
        elif shapes:
            surface = unary_union(shapes)
            for j, clear in regions:
                if _overlap(surface, clear) >= min_cover * clear.area:
                    seen[j].append(_within(surface, clear))
        for j, parts in seen.items():
            area = unary_union(parts)
            if not isinstance(area, Polygon):
                area = area.convex_hull  # wings outlined apart, or a mask in parts
            if not isinstance(area, Polygon) or area.is_empty:
                continue
            views[j].n_seen += 1
            offset = centers[j] - camera[:2]
            obs = Observation(
                image_id=image_id,
                label=base.label,
                polygon=[
                    (min(max(float(px), 0.0), 1.0), min(max(float(py), 0.0), 1.0))
                    for px, py in area.exterior.coords[:-1]
                ],
                camera_lon=base.camera_lon,
                camera_lat=base.camera_lat,
                compass_angle=base.compass_angle,
                is_pano=base.is_pano,
                sequence_id=base.sequence_id,
                captured_at=base.captured_at,
                confidence=base.confidence,
                source=base.source,
                image_width=base.image_width,
                image_height=base.image_height,
                extra={k: v for k, v in base.extra.items() if k != 'instance'},
            )
            obs.bearing = round(
                math.degrees(math.atan2(offset[0], offset[1])) % 360.0, 2
            )
            obs.range_m = round(float(local[j].distance(Point(x, y))), 2)
            obs.lon, obs.lat = footprints[j].centroid.x, footprints[j].centroid.y
            found[j].append((float(obs.range_m), obs))
    for view, sightings in zip(views, found, strict=True):
        view.members = spread_views(sightings, max_views)
    counts['footprints_expected'] = sum(v.n_expected > 0 for v in views)
    counts['footprints_seen'] = sum(v.n_seen > 0 for v in views)
    counts['unclaimed'] = len(unclaimed)
    return views, sorted(unclaimed), dict(counts)


def _valid(shape: Polygon) -> BaseGeometry:
    """The shape, repaired when its outline crosses itself."""
    return shape if shape.is_valid else shape.buffer(0)


def _overlap(outline: BaseGeometry, region: BaseGeometry) -> float:
    """Area an outline shares with a region that may run past the panorama's seam."""
    shared = float(outline.intersection(region).area)
    if region.bounds[2] > 1.0:
        shared += float(affinity.translate(outline, 1.0).intersection(region).area)
    return shared


def _within(outline: BaseGeometry, region: BaseGeometry) -> BaseGeometry:
    """The part of an outline inside a region, brought back left of the seam."""
    inside = outline.intersection(region)
    if region.bounds[2] > 1.0:
        past = affinity.translate(outline, 1.0).intersection(region)
        inside = unary_union([inside, affinity.translate(past, -1.0)])
    return inside
