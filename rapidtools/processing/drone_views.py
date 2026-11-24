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
# 11-12-2026

"""
Choosing the drone photos that show the most of each asset.

A drone flight photographs every building dozens of times, mostly badly:
clipped by the frame, hidden behind a neighbour, seen end-on, or the same
side over and over. Sending all of them to a vision-language model is slow
and dilutes the evidence; sending the nearest one misses the sides it does
not show. This module picks, for every asset, the few photos that together
show as much of its surface as the flight captured.

The method:

1. **Surface model.** The footprint is extruded to a prism and its surface
   is sampled into patches: wall patches along every edge at a few heights,
   and roof patches (:func:`surface_patches`).
2. **Visibility.** For every photo and every patch, a quality between 0 and
   1 says how well the photo shows the patch: 0 when the patch is outside
   the frame, faces away, or is hidden behind the asset itself or a
   neighbouring prism (rays are cast against the extruded footprints around
   it); otherwise the product of an incidence term (how squarely the patch
   is seen) and a resolution term (pixels per metre against a target)
   (:func:`view_quality`).
3. **Selection.** The view set maximises ``sum over patches of the best
   quality any chosen photo gives it``. That objective is monotone
   submodular, so adding photos greedily is guaranteed to reach at least
   63 % of the best possible set of the same size, and in practice lands
   within a few percent. A photo that repeats what is already covered gains
   nothing and is never picked, so near-duplicate frames drop out without a
   separate rule (:func:`select_views`).
4. **Reporting.** What remains unseen is reported per asset (coverage of the
   roof and of the walls facing north, east, south and west) and as wall
   segments with a suggested camera position, so a pilot knows where to fly
   next (:meth:`DroneViewSelector.coverage_gaps`).

:class:`DroneImageExtractor` wraps this as a pipeline step that crops the
chosen photos around each asset and attaches the crops, so
:class:`~rapidtools.processing.AssetAnalyzer` works unchanged downstream.

Example:
    >>> from rapidtools.data_sources import read_drone_images
    >>> from rapidtools.processing import DroneImageExtractor
    >>> extractor = DroneImageExtractor(
    ...     read_drone_images('flight_01/'), save_directory='crops', max_views=4
    ... )  # doctest: +SKIP
    >>> buildings = extractor(buildings)  # doctest: +SKIP
    >>> buildings[0].attributes['drone_view_coverage']  # doctest: +SKIP
    0.83
"""

from __future__ import annotations

import logging
import math
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from shapely import contains_xy
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.strtree import STRtree
from tqdm import tqdm

from rapidtools.core import ImageAsset, PhysicalAsset, PhysicalAssetCollection
from rapidtools.core.cancellation import raise_if_cancelled
from rapidtools.data_sources.drone_images import DroneCamera, read_drone_images

from . import outlines
from .step import Stage
from .street_localization import local_projection

logger = logging.getLogger(__name__)

#: Compass names of the four wall-facing sectors, clockwise from north.
SIDES = ('north', 'east', 'south', 'west')
#: Attribute keys tried, in order, for an asset's height in metres.
HEIGHT_KEYS = ('height_m', 'height', 'building_height', 'building:height')
#: Attribute keys tried for a storey count when no height is recorded.
LEVEL_KEYS = ('building:levels', 'levels', 'num_stories', 'stories')
#: Height of one storey, used with :data:`LEVEL_KEYS`.
STOREY_HEIGHT_M = 3.0
#: Patch points sit this far off the surface so a ray to a patch does not
#: graze the wall or roof the patch lies on.
SURFACE_OFFSET_M = 0.03

GroundElevation = float | str | Callable[[float, float], float]


# ----------------------------------------------------------------- records
@dataclass
class SurfacePatches:
    """
    Sample points on the surface of an extruded footprint.

    Attributes:
        points: ``P x 3`` patch centres in local east/north/up metres.
        normals: ``P x 3`` outward unit normals.
        weights: Importance of each patch: its area in square metres, times
            the roof weight for roof patches.
        wall: Index of the wall (footprint edge) each patch lies on, ``-1``
            for roof patches.
        walls: ``W x 4`` wall end points ``(x0, y0, x1, y1)``.
        wall_azimuth: Compass direction each wall faces, in degrees.
        polygon: The footprint in local metres.
        base: Elevation of the wall bases.
        top: Elevation of the roof.

    Example:
        >>> from shapely.geometry import box
        >>> patches = surface_patches(box(0, 0, 10, 8), base=0.0, height=6.0)
        >>> sorted(float(a) for a in patches.wall_azimuth)
        [0.0, 90.0, 180.0, 270.0]
        >>> bool(patches.is_roof.any()), round(float(patches.weights.sum()))
        (True, 296)
    """

    points: np.ndarray
    normals: np.ndarray
    weights: np.ndarray
    wall: np.ndarray
    walls: np.ndarray
    wall_azimuth: np.ndarray
    polygon: Polygon
    base: float
    top: float

    @property
    def is_roof(self) -> np.ndarray:
        """Boolean mask of the roof patches."""
        return self.wall < 0

    def side_of_patch(self) -> np.ndarray:
        """Sector index into :data:`SIDES` per patch, ``-1`` for the roof."""
        sides = np.full(len(self.wall), -1, dtype=int)
        mask = ~self.is_roof
        azimuth = self.wall_azimuth[self.wall[mask]]
        sides[mask] = (np.floor((azimuth + 45.0) / 90.0).astype(int)) % 4
        return sides


@dataclass
class SelectedView:
    """
    One photo chosen for an asset.

    Attributes:
        camera: The photo.
        gain: Share of the asset's surface this photo added to what the
            photos chosen before it already showed (0 to 1).
        quality: Share of the surface this photo shows on its own.
        azimuth_deg: Compass direction from the asset to the camera.
        elevation_deg: Angle of the camera above the asset's horizon.
        distance_m: Distance from the camera to the asset.
        shows: What the photo contributes: ``'roof'`` and wall sectors.
        frame_fraction: Share of the asset inside the frame.
        bbox: Pixel box ``(left, top, right, bottom)`` of the projected
            prism, unclipped.
        outline: Pixel outline of the projected prism (its convex hull).
    """

    camera: DroneCamera
    gain: float
    quality: float
    azimuth_deg: float
    elevation_deg: float
    distance_m: float
    shows: list[str]
    frame_fraction: float
    bbox: tuple[float, float, float, float]
    outline: list[tuple[float, float]]


@dataclass
class AssetViewSet:
    """
    The photos chosen for one asset and what they leave unseen.

    Attributes:
        asset_id: The asset.
        views: Chosen photos, most informative first.
        coverage: Quality-weighted share of the surface the set shows.
        seen_fraction: Share of the surface seen at all (quality above the
            selector's ``seen_threshold``).
        coverage_by_side: Coverage of the ``'roof'`` and of the walls facing
            ``'north'``, ``'east'``, ``'south'`` and ``'west'``; a side the
            footprint has no wall on is absent.
        unseen_walls: Walls no chosen photo shows, each a dict with
            ``wall`` (index), ``facing_deg``, ``length_m``, ``coverage``,
            ``line`` (two ``(lon, lat)`` points) and ``viewpoint`` (a
            suggested camera position: ``lon``, ``lat``, ``height_m``,
            ``yaw_deg``, ``pitch_deg``).
        n_candidates: Photos that showed any of the asset.
        height_m: Prism height used.
        ground_elevation_m: Base elevation used.
    """

    asset_id: str
    views: list[SelectedView] = field(default_factory=list)
    coverage: float = 0.0
    seen_fraction: float = 0.0
    coverage_by_side: dict[str, float] = field(default_factory=dict)
    unseen_walls: list[dict[str, Any]] = field(default_factory=list)
    n_candidates: int = 0
    height_m: float = 0.0
    ground_elevation_m: float = 0.0


# ---------------------------------------------------------- surface model
def footprint_polygon(
    geometry: BaseGeometry,
    project: Callable[[float, float], tuple[float, float]],
    point_radius_m: float = 1.0,
    simplify_m: float = 0.3,
) -> Polygon | None:
    """
    Footprint of an asset as one polygon in local metres.

    Polygons keep their outer ring (courtyards are ignored), multi-part
    geometries use the convex hull of their parts, and points and lines are
    buffered by ``point_radius_m`` so poles and similar assets get a slim
    prism.

    Args:
        geometry: Asset geometry in longitude/latitude.
        project: Local projection.
        point_radius_m: Buffer for points and lines.
        simplify_m: Douglas-Peucker tolerance; removes sliver edges.

    Returns:
        shapely.geometry.Polygon | None: Counter-clockwise polygon, or
        ``None`` for an empty geometry.

    Example:
        >>> from shapely.geometry import Point
        >>> from rapidtools.processing.street_localization import local_projection
        >>> project, _ = local_projection(0.0, 0.0)
        >>> round(footprint_polygon(Point(0, 0), project).area, 1)
        3.1
    """
    if geometry is None or geometry.is_empty:
        return None

    def to_local(geom: BaseGeometry) -> list[tuple[float, float]]:
        return [project(x, y) for x, y, *_ in geom.coords]

    if geometry.geom_type == 'Polygon':
        local = Polygon(to_local(geometry.exterior))
    elif geometry.geom_type == 'MultiPolygon':
        parts = [Polygon(to_local(g.exterior)) for g in geometry.geoms]
        local = max(parts, key=lambda g: g.area) if len(parts) == 1 else None
        if local is None:
            local = MultiPoint(
                [p for g in parts for p in g.exterior.coords]
            ).convex_hull
    elif geometry.geom_type == 'Point':
        local = Point(*project(geometry.x, geometry.y)).buffer(point_radius_m, 8)
    elif geometry.geom_type == 'LineString':
        local = LineString(to_local(geometry)).buffer(point_radius_m, 2)
    else:
        hull = geometry.convex_hull
        if hull.geom_type != 'Polygon':
            centre = hull.centroid
            return Point(*project(centre.x, centre.y)).buffer(point_radius_m, 8)
        local = Polygon(to_local(hull.exterior))
    if not local.is_valid:
        local = local.buffer(0)
        if local.geom_type != 'Polygon':
            local = local.convex_hull
    if local.is_empty or local.geom_type != 'Polygon':
        return None
    if simplify_m > 0 and geometry.geom_type in ('Polygon', 'MultiPolygon'):
        simpler = local.simplify(simplify_m)
        if simpler.geom_type == 'Polygon' and not simpler.is_empty:
            local = simpler
    return orient(Polygon(local.exterior), 1.0)


def surface_patches(
    polygon: Polygon,
    base: float,
    height: float,
    wall_spacing_m: float = 2.0,
    wall_levels: int = 2,
    roof_spacing_m: float = 2.5,
    roof_weight: float = 1.0,
) -> SurfacePatches:
    """
    Sample the walls and roof of an extruded footprint.

    Args:
        polygon: Footprint in local metres.
        base: Ground elevation at the asset.
        height: Height of the prism.
        wall_spacing_m: Distance between wall patches along an edge.
        wall_levels: Rows of patches up each wall.
        roof_spacing_m: Grid spacing of the roof patches.
        roof_weight: Importance of roof area relative to wall area. ``0``
            leaves the roof out of the objective (facades only); values
            above ``1`` favour views of the roof.

    Returns:
        SurfacePatches: Patch centres, normals and weights.

    Example:
        >>> from shapely.geometry import box
        >>> patches = surface_patches(box(0, 0, 4, 4), 0.0, 3.0, wall_levels=1)
        >>> len(patches.points), int(patches.is_roof.sum())
        (12, 4)
    """
    polygon = orient(Polygon(polygon.exterior), 1.0)
    height = max(float(height), 0.1)
    ring = np.asarray(polygon.exterior.coords)[:-1, :2]
    starts, ends = ring, np.roll(ring, -1, axis=0)
    points, normals, weights, wall_ids, azimuths = [], [], [], [], []
    levels = max(1, int(wall_levels))
    for index, (a, b) in enumerate(zip(starts, ends, strict=True)):
        edge = b - a
        length = float(np.hypot(*edge))
        if length < 1e-6:
            azimuths.append(0.0)
            continue
        normal = np.array([edge[1], -edge[0], 0.0]) / length  # outward (CCW ring)
        azimuths.append(math.degrees(math.atan2(normal[0], normal[1])) % 360.0)
        count = max(1, int(round(length / wall_spacing_m)))
        for i in range(count):
            xy = a + edge * (i + 0.5) / count
            for level in range(levels):
                z = base + height * (level + 0.5) / levels
                points.append([xy[0], xy[1], z])
                normals.append(normal)
                weights.append(length / count * height / levels)
                wall_ids.append(index)
    if roof_weight > 0:
        minx, miny, maxx, maxy = polygon.bounds
        nx = max(1, int(round((maxx - minx) / roof_spacing_m)))
        ny = max(1, int(round((maxy - miny) / roof_spacing_m)))
        gx = minx + (np.arange(nx) + 0.5) * (maxx - minx) / nx
        gy = miny + (np.arange(ny) + 0.5) * (maxy - miny) / ny
        mx, my = (m.ravel() for m in np.meshgrid(gx, gy))
        inside = contains_xy(polygon, mx, my)
        if not inside.any():
            centre = polygon.representative_point()
            mx, my, inside = (
                np.array([centre.x]),
                np.array([centre.y]),
                np.array([True]),
            )
        share = polygon.area / int(inside.sum()) * roof_weight
        for x, y in zip(mx[inside], my[inside], strict=True):
            points.append([x, y, base + height])
            normals.append([0.0, 0.0, 1.0])
            weights.append(share)
            wall_ids.append(-1)
    return SurfacePatches(
        points=np.asarray(points, dtype=float).reshape(-1, 3),
        normals=np.asarray(normals, dtype=float).reshape(-1, 3),
        weights=np.asarray(weights, dtype=float),
        wall=np.asarray(wall_ids, dtype=int),
        walls=np.column_stack([starts, ends]),
        wall_azimuth=np.asarray(azimuths, dtype=float),
        polygon=polygon,
        base=float(base),
        top=float(base) + height,
    )


# --------------------------------------------------------------- visibility
def blocked_rays(
    origins: np.ndarray,
    targets: np.ndarray,
    edges: np.ndarray,
    tops: np.ndarray,
    chunk: int = 2_000_000,
) -> np.ndarray:
    """
    Which rays are blocked by extruded footprints.

    A ray from a camera to a surface point is blocked when, in plan view, it
    crosses a footprint edge at a place where the ray is still below the top
    of that footprint: it would have to pass through the wall. Rays are
    straight lines, so their height at each crossing is known, and testing
    every edge crossing is exact for prisms.

    Args:
        origins: ``K x 3`` ray starts (cameras).
        targets: ``K x 3`` ray ends (surface points).
        edges: ``E x 4`` footprint edges ``(x0, y0, x1, y1)``.
        tops: ``E`` elevations of the prism each edge belongs to.
        chunk: Ray-edge pairs tested at a time (bounds memory).

    Returns:
        numpy.ndarray: Boolean mask of length ``K``.

    Example:
        >>> import numpy as np
        >>> wall = np.array([[5.0, -5.0, 5.0, 5.0]])  # a wall 4 m tall at x = 5
        >>> low = blocked_rays([[0, 0, 1.0]], [[10, 0, 1.0]], wall, [4.0])
        >>> high = blocked_rays([[0, 0, 30.0]], [[10, 0, 1.0]], wall, [4.0])
        >>> bool(low[0]), bool(high[0])
        (True, False)
    """
    origins = np.asarray(origins, dtype=float).reshape(-1, 3)
    targets = np.asarray(targets, dtype=float).reshape(-1, 3)
    edges = np.asarray(edges, dtype=float).reshape(-1, 4)
    tops = np.asarray(tops, dtype=float).reshape(-1)
    blocked = np.zeros(len(origins), dtype=bool)
    if not len(origins) or not len(edges):
        return blocked
    edge_start = edges[:, :2]
    edge_dir = edges[:, 2:] - edges[:, :2]
    step = max(1, chunk // len(edges))
    for begin in range(0, len(origins), step):
        o = origins[begin : begin + step]
        r = targets[begin : begin + step] - o
        # 2-D cross products, rays (K, 1) against edges (1, E):
        denom = (
            r[:, None, 0] * edge_dir[None, :, 1] - r[:, None, 1] * edge_dir[None, :, 0]
        )
        dx = edge_start[None, :, 0] - o[:, None, 0]
        dy = edge_start[None, :, 1] - o[:, None, 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (dx * edge_dir[None, :, 1] - dy * edge_dir[None, :, 0]) / denom
            s = (dx * r[:, None, 1] - dy * r[:, None, 0]) / denom
        crossing = (np.abs(denom) > 1e-12) & (t > 1e-6) & (t < 1 - 1e-9)
        crossing &= (s >= 0.0) & (s <= 1.0)
        height = o[:, None, 2] + t * r[:, None, 2]
        crossing &= height < tops[None, :] - 0.01
        blocked[begin : begin + step] = crossing.any(axis=1)
    return blocked


def view_quality(
    positions: np.ndarray,
    rotations: np.ndarray,
    focal_px: np.ndarray,
    sizes: np.ndarray,
    centres: np.ndarray,
    patches: SurfacePatches,
    occluder_edges: np.ndarray | None = None,
    occluder_tops: np.ndarray | None = None,
    target_px_per_m: float = 15.0,
    full_incidence_deg: float = 45.0,
    max_incidence_deg: float = 85.0,
    edge_margin: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """
    How well each camera shows each surface patch.

    Args:
        positions: ``C x 3`` camera centres in local metres.
        rotations: ``C x 3 x 3`` world-to-camera rotations.
        focal_px: ``C`` focal lengths in pixels.
        sizes: ``C x 2`` image sizes ``(width, height)``.
        centres: ``C x 2`` principal points.
        patches: Surface samples of one asset.
        occluder_edges: ``E x 4`` footprint edges of everything that can
            hide the asset, its own included.
        occluder_tops: ``E`` top elevations of those prisms.
        target_px_per_m: Resolution at which a patch counts as fully
            resolved; coarser views score proportionally less.
        full_incidence_deg: Patches seen within this angle of their normal
            get full credit; credit then falls with the cosine.
        max_incidence_deg: Patches seen more obliquely than this do not
            count as seen.
        edge_margin: Width of the border strip, as a fraction of the shorter
            image side, in which a patch counts as outside the frame.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: ``C x P`` qualities in
        ``[0, 1]`` and the ``C x P`` boolean mask of patches inside the
        frame (whether or not they face the camera).

    Example:
        >>> import numpy as np
        >>> from shapely.geometry import box
        >>> patches = surface_patches(box(-5, -5, 5, 5), 0.0, 6.0)
        >>> quality, _ = view_quality(
        ...     np.array([[0.0, 0.0, 60.0]]),
        ...     np.array([[[1.0, 0, 0], [0, -1, 0], [0, 0, -1]]]),
        ...     np.array([1500.0]), np.array([[2000, 1500]]), np.array([[1000, 750]]),
        ...     patches,
        ... )
        >>> bool(quality[0, patches.is_roof].min() > 0.9)  # nadir sees the roof
        True
        >>> float(quality[0, ~patches.is_roof].max())  # and no wall
        0.0
    """
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    points = patches.points + SURFACE_OFFSET_M * patches.normals
    rays = points[None, :, :] - positions[:, None, :]  # (C, P, 3)
    distance = np.linalg.norm(rays, axis=2)
    cam = np.einsum('cij,cpj->cpi', rotations, rays)
    depth = cam[:, :, 2]
    safe = np.where(depth > 1e-6, depth, 1.0)
    px = centres[:, None, 0] + focal_px[:, None] * cam[:, :, 0] / safe
    py = centres[:, None, 1] + focal_px[:, None] * cam[:, :, 1] / safe
    margin = edge_margin * sizes.min(axis=1)[:, None]
    in_frame = (
        (depth > 0.1)
        & (px >= margin)
        & (px <= sizes[:, None, 0] - margin)
        & (py >= margin)
        & (py <= sizes[:, None, 1] - margin)
    )
    cosine = -np.einsum('cpj,pj->cp', rays, patches.normals) / np.maximum(
        distance, 1e-9
    )
    visible = in_frame & (cosine > math.cos(math.radians(max_incidence_deg)))
    if occluder_edges is not None and len(occluder_edges) and visible.any():
        cams, pts = np.nonzero(visible)
        hidden = blocked_rays(
            positions[cams], points[pts], occluder_edges, occluder_tops
        )
        visible[cams[hidden], pts[hidden]] = False
    incidence = np.clip(cosine / math.cos(math.radians(full_incidence_deg)), 0.0, 1.0)
    resolution = np.clip(focal_px[:, None] / safe / target_px_per_m, 0.0, 1.0)
    return np.where(visible, incidence * resolution, 0.0), in_frame


# ---------------------------------------------------------------- selection
def select_views(
    quality: np.ndarray,
    weights: np.ndarray,
    max_views: int = 4,
    min_gain: float = 0.03,
    directions: np.ndarray | None = None,
    diversity: float = 0.15,
) -> tuple[list[int], list[float], np.ndarray]:
    """
    Greedily pick the views that together show the most.

    Maximises ``sum_p weights[p] * max_i quality[i, p]`` over the chosen
    views ``i``. Each round adds the view with the largest gain over what
    is already covered and stops when ``max_views`` are chosen or the best
    remaining gain is under ``min_gain`` of the whole surface.

    Args:
        quality: ``C x P`` view-by-patch qualities.
        weights: ``P`` patch weights.
        max_views: Largest set to return.
        min_gain: Smallest share of the surface a further view must add.
        directions: Optional ``C x 3`` unit vectors from the asset to each
            camera. When given, a candidate's gain is scaled by up to
            ``1 + diversity`` according to its angular distance from the
            views already chosen, which breaks near-ties in favour of a new
            vantage point.
        diversity: Strength of that preference; ``0`` disables it.

    Returns:
        tuple[list[int], list[float], numpy.ndarray]: Indices of the chosen
        views in pick order, the share of the surface each one added, and
        the best quality reached per patch.

    Example:
        >>> import numpy as np
        >>> quality = np.array([[1.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> chosen, gains, best = select_views(quality, np.ones(3), max_views=3)
        >>> chosen, [round(g, 2) for g in gains]  # the duplicate view is skipped
        ([0, 2], [0.33, 0.33])
    """
    quality = np.asarray(quality, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    best = np.zeros(quality.shape[1] if quality.ndim == 2 else 0)
    chosen: list[int] = []
    gains: list[float] = []
    if total <= 0 or not quality.size:
        return chosen, gains, best
    available = np.ones(len(quality), dtype=bool)
    for _ in range(max(0, max_views)):
        gain = (np.maximum(quality - best[None, :], 0.0) * weights[None, :]).sum(axis=1)
        gain[~available] = -1.0
        score = gain
        if directions is not None and chosen and diversity > 0:
            cosines = np.clip(directions @ directions[chosen].T, -1.0, 1.0)
            separation = np.degrees(np.arccos(cosines)).min(axis=1) / 180.0
            score = gain * (1.0 + diversity * separation)
        pick = int(np.argmax(score))
        if gain[pick] / total < min_gain:
            break
        chosen.append(pick)
        gains.append(float(gain[pick] / total))
        best = np.maximum(best, quality[pick])
        available[pick] = False
    return chosen, gains, best


def _corners(polygon: Polygon, base: float, top: float) -> np.ndarray:
    """Footprint vertices at the base and at the top of a prism."""
    ring = np.asarray(polygon.exterior.coords)[:-1, :2]
    low = np.column_stack([ring, np.full(len(ring), base)])
    high = np.column_stack([ring, np.full(len(ring), top)])
    return np.vstack([low, high])


def estimate_image_shift(
    expected: np.ndarray, observed: np.ndarray, max_shift: int
) -> tuple[int, int, float, float]:
    """
    Translation that best lays an expected mask over an observed one.

    Used to correct a photo's pose error in the image: ``expected`` is where
    the footprints project with the recorded pose, ``observed`` is where a
    segmentation model actually sees buildings. Every shift up to
    ``max_shift`` pixels is scored by the intersection over union of the two
    masks (one FFT correlation), and the best is returned.

    Args:
        expected: Boolean ``H x W`` mask.
        observed: Boolean mask of the same shape.
        max_shift: Largest shift considered, in pixels, along each axis.

    Returns:
        tuple[int, int, float, float]: ``(dx, dy)`` to add to the expected
        positions, the overlap score at that shift, and the score with no
        shift.

    Example:
        >>> import numpy as np
        >>> expected = np.zeros((40, 60), dtype=bool)
        >>> expected[10:20, 10:25] = True
        >>> observed = np.roll(expected, (3, -5), axis=(0, 1))
        >>> dx, dy, best, before = estimate_image_shift(expected, observed, 8)
        >>> (dx, dy), round(best, 2), best > before
        ((-5, 3), 1.0, True)
    """
    from scipy.signal import fftconvolve

    expected = np.asarray(expected, dtype=float)
    observed = np.asarray(observed, dtype=float)
    area = expected.sum() + observed.sum()
    if expected.sum() == 0 or observed.sum() == 0:
        return 0, 0, 0.0, 0.0
    overlap = fftconvolve(observed, expected[::-1, ::-1], mode='full')
    height, width = expected.shape
    reach = int(max(0, max_shift))
    window = overlap[
        height - 1 - reach : height + reach, width - 1 - reach : width + reach
    ]
    score = window / np.maximum(area - window, 1e-9)
    row, column = np.unravel_index(int(np.argmax(score)), score.shape)
    return (
        int(column) - reach,
        int(row) - reach,
        float(score[row, column]),
        float(score[reach, reach]),
    )


def _numeric(value: Any) -> float | None:
    """A positive number out of ``6``, ``'6.5'`` or ``'6.5 m'``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip().split(' ')[0].rstrip('m')
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


class DroneViewSelector:
    """
    Pick, for every asset, the drone photos that show the most of it.

    Args:
        cameras (Sequence[DroneCamera]):
            The flight, from
            :func:`~rapidtools.data_sources.read_drone_images`.
        max_views (int):
            Most photos kept per asset. Defaults to 4.
        min_gain (float):
            A further photo is kept only if it adds at least this share of
            the asset's surface. Defaults to 0.03.
        default_height_m (float):
            Prism height for assets without a height attribute. Defaults to
            6.0 (a two-storey house).
        height_attribute (str | None):
            Attribute holding the asset height in metres; when ``None`` the
            usual keys (``height_m``, ``height``, storey counts) are tried.
        ground_elevation (float | str | Callable):
            Elevation of the ground under the assets, in the datum of the
            camera elevations: one number for flat terrain (default ``0``,
            the take-off level when cameras use relative altitude), the
            name of an asset attribute, or a function ``(lon, lat) ->
            elevation`` such as a DEM lookup.
        roof_weight (float):
            Importance of roof area relative to wall area. Defaults to 1.0;
            ``0`` selects for facades only.
        target_px_per_m (float):
            Resolution at which a surface counts as fully resolved.
            Defaults to 15.
        full_incidence_deg (float):
            Surfaces seen within this angle of their normal get full
            credit. Defaults to 45.
        max_incidence_deg (float):
            Surfaces seen more obliquely than this do not count. Defaults
            to 85.
        clip_penalty (float):
            Exponent applied to the share of the asset inside the frame: a
            photo that cuts the asset off is worth that much less on every
            patch. ``0`` disables the penalty. Defaults to 1.0.
        diversity (float):
            Preference for new vantage points among near-equal candidates.
            Defaults to 0.15.
        seen_threshold (float):
            Quality above which a patch counts as seen in the coverage
            report. Defaults to 0.25.
        max_range_m (float):
            Photos farther than this from an asset are not considered.
            Defaults to 400.
        max_candidates (int):
            Photos per asset that get the full occlusion test, the best by
            unoccluded quality within each viewing direction. Defaults to
            160.
        occluder_radius_m (float):
            Neighbours within this distance of an asset can hide it.
            Defaults to 80.
        obstacles (PhysicalAssetCollection | None):
            Further things that block the view but are not analysed (tree
            canopies, buildings outside the collection), as footprints with
            a height attribute.
        wall_spacing_m (float):
            Distance between wall patches. Defaults to 2.0.
        wall_levels (int):
            Rows of patches up each wall. Defaults to 2.
        roof_spacing_m (float):
            Grid spacing of roof patches. Defaults to 2.5.
        point_radius_m (float):
            Footprint radius given to point assets. Defaults to 1.0.
        cancel_event (threading.Event | None):
            Cooperative cancellation flag checked between assets.

    Example:
        >>> selector = DroneViewSelector(cameras, max_views=3)  # doctest: +SKIP
        >>> view_sets = selector.select(buildings)  # doctest: +SKIP
        >>> view_sets['b12'].coverage_by_side  # doctest: +SKIP
        {'roof': 0.97, 'north': 0.0, 'east': 0.81, 'south': 0.9, 'west': 0.74}
    """

    def __init__(
        self,
        cameras: Sequence[DroneCamera],
        max_views: int = 4,
        min_gain: float = 0.03,
        default_height_m: float = 6.0,
        height_attribute: str | None = None,
        ground_elevation: GroundElevation = 0.0,
        roof_weight: float = 1.0,
        target_px_per_m: float = 15.0,
        full_incidence_deg: float = 45.0,
        max_incidence_deg: float = 85.0,
        clip_penalty: float = 1.0,
        diversity: float = 0.15,
        seen_threshold: float = 0.25,
        max_range_m: float = 400.0,
        max_candidates: int = 160,
        occluder_radius_m: float = 80.0,
        obstacles: PhysicalAssetCollection | None = None,
        wall_spacing_m: float = 2.0,
        wall_levels: int = 2,
        roof_spacing_m: float = 2.5,
        point_radius_m: float = 1.0,
        cancel_event: threading.Event | None = None,
    ) -> None:
        if max_views < 1:
            raise ValueError('max_views must be at least 1.')
        if roof_weight < 0:
            raise ValueError('roof_weight cannot be negative.')
        self.cameras = list(cameras)
        self.max_views = max_views
        self.min_gain = min_gain
        self.default_height_m = default_height_m
        self.height_attribute = height_attribute
        self.ground_elevation = ground_elevation
        self.roof_weight = roof_weight
        self.target_px_per_m = target_px_per_m
        self.full_incidence_deg = full_incidence_deg
        self.max_incidence_deg = max_incidence_deg
        self.clip_penalty = clip_penalty
        self.diversity = diversity
        self.seen_threshold = seen_threshold
        self.max_range_m = max_range_m
        self.max_candidates = max_candidates
        self.occluder_radius_m = occluder_radius_m
        self.obstacles = obstacles
        self.wall_spacing_m = wall_spacing_m
        self.wall_levels = wall_levels
        self.roof_spacing_m = roof_spacing_m
        self.point_radius_m = point_radius_m
        self.cancel_event = cancel_event
        self._project: Callable[[float, float], tuple[float, float]] | None = None
        self._prisms: dict[str, tuple[Polygon, float, float]] = {}
        self._unproject: Callable[[float, float], tuple[float, float]] | None = None

    # ------------------------------------------------------------- inputs
    def asset_height(self, asset: PhysicalAsset) -> float:
        """Prism height of an asset in metres (attribute or default)."""
        attributes = asset.attributes
        keys = (self.height_attribute,) if self.height_attribute else HEIGHT_KEYS
        for key in keys:
            value = _numeric(attributes.get(key))
            if value is not None:
                return value
        if not self.height_attribute:
            for key in LEVEL_KEYS:
                value = _numeric(attributes.get(key))
                if value is not None:
                    return value * STOREY_HEIGHT_M
        return self.default_height_m

    def asset_ground(self, asset: PhysicalAsset) -> float:
        """Ground elevation under an asset."""
        source = self.ground_elevation
        if callable(source):
            centre = asset.geometry.centroid
            return float(source(centre.x, centre.y))
        if isinstance(source, str):
            value = asset.attributes.get(source)
            try:
                return float(value)
            except (TypeError, ValueError):
                logger.warning(
                    f"Asset {asset.id} has no usable '{source}'; ground taken as 0."
                )
                return 0.0
        return float(source)

    def _camera_arrays(self, project) -> dict[str, np.ndarray]:
        cams = self.cameras
        return {
            'positions': np.array([c.position(project) for c in cams]).reshape(-1, 3),
            'rotations': np.array([c.rotation for c in cams]).reshape(-1, 3, 3),
            'focal': np.array([c.focal_px for c in cams], dtype=float),
            'sizes': np.array([[c.width, c.height] for c in cams], dtype=float).reshape(
                -1, 2
            ),
            'centres': np.array([[c.cx, c.cy] for c in cams], dtype=float).reshape(
                -1, 2
            ),
        }

    # ---------------------------------------------------------- selection
    def select(
        self,
        assets: Iterable[PhysicalAsset],
        allowed: Mapping[str, Iterable[str]] | None = None,
    ) -> dict[str, AssetViewSet]:
        """
        Choose the view set of every asset.

        Args:
            assets: The assets (a collection or any iterable of them).
                Their footprints also act as occluders for one another.
            allowed: Optional restriction of the photos considered per
                asset: asset ID to the image IDs that may be chosen for it.
                Assets not listed keep the whole flight. Useful for
                comparing against another way of picking photos, or for
                limiting an asset to one flight of several.

        Returns:
            dict[str, AssetViewSet]: One entry per asset with a usable
            geometry, keyed by asset ID. Assets no photo shows get an
            empty view set with zero coverage.

        Example:
            >>> view_sets = selector.select(buildings)  # doctest: +SKIP
            >>> [v.camera.image_id for v in view_sets['b12'].views]  # doctest: +SKIP
            ['DJI_0101', 'DJI_0178', 'DJI_0033']
        """
        assets = list(assets)
        results: dict[str, AssetViewSet] = {}
        if not assets:
            return results
        located = [a.geometry.centroid for a in assets if not a.geometry.is_empty]
        if not located:
            return results
        centre = MultiPoint(located).centroid
        project, unproject = local_projection(centre.x, centre.y)
        self._project, self._unproject = project, unproject
        prisms: list[tuple[PhysicalAsset, Polygon, float, float]] = []
        for asset in assets:
            polygon = footprint_polygon(asset.geometry, project, self.point_radius_m)
            if polygon is None:
                logger.warning(f'Asset {asset.id} has no usable footprint; skipped.')
                continue
            prisms.append(
                (asset, polygon, self.asset_ground(asset), self.asset_height(asset))
            )
        self._prisms = {
            asset.id: (polygon, base, height) for asset, polygon, base, height in prisms
        }
        # Everything that can hide an asset: the assets themselves plus any
        # extra obstacles, as edge lists with the top elevation of each prism.
        blockers = [(polygon, base + height) for _, polygon, base, height in prisms]
        for obstacle in self.obstacles or ():
            polygon = footprint_polygon(obstacle.geometry, project, self.point_radius_m)
            if polygon is not None:
                top = self.asset_ground(obstacle) + self.asset_height(obstacle)
                blockers.append((polygon, top))
        tree = STRtree([polygon for polygon, _ in blockers])
        edge_lists = []
        for polygon, _ in blockers:
            ring = np.asarray(polygon.exterior.coords)[:, :2]
            edge_lists.append(np.column_stack([ring[:-1], ring[1:]]))
        if not self.cameras:
            for asset, _, base, height in prisms:
                results[asset.id] = AssetViewSet(
                    asset.id, height_m=height, ground_elevation_m=base
                )
            return results
        arrays = self._camera_arrays(project)
        camera_tree = cKDTree(arrays['positions'][:, :2])
        for asset, polygon, base, height in tqdm(prisms, desc='Selecting drone views'):
            raise_if_cancelled(self.cancel_event, 'drone view selection')
            near = tree.query(polygon.buffer(self.occluder_radius_m))
            edges = np.concatenate([edge_lists[i] for i in near])
            tops = np.concatenate(
                [np.full(len(edge_lists[i]), blockers[i][1]) for i in near]
            )
            permitted = None
            if allowed is not None and asset.id in allowed:
                permitted = set(allowed[asset.id])
            results[asset.id] = self._select_asset(
                asset,
                polygon,
                base,
                height,
                arrays,
                camera_tree,
                edges,
                tops,
                permitted,
            )
        return results

    def _select_asset(
        self,
        asset,
        polygon,
        base,
        height,
        arrays,
        camera_tree,
        edges,
        tops,
        permitted=None,
    ) -> AssetViewSet:
        patches = surface_patches(
            polygon,
            base,
            height,
            self.wall_spacing_m,
            self.wall_levels,
            self.roof_spacing_m,
            self.roof_weight,
        )
        result = AssetViewSet(asset.id, height_m=height, ground_elevation_m=base)
        centroid = np.array([polygon.centroid.x, polygon.centroid.y, base + height / 2])
        candidates = np.asarray(
            camera_tree.query_ball_point(centroid[:2], self.max_range_m), dtype=int
        )
        if permitted is not None and candidates.size:
            keep = [self.cameras[int(i)].image_id in permitted for i in candidates]
            candidates = candidates[np.asarray(keep, dtype=bool)]
        total = float(patches.weights.sum())
        if candidates.size:
            # Frustum cull on one point: drop photos whose frame is nowhere
            # near the asset before any per-patch work.
            candidates.sort()
            relative = centroid[None, :] - arrays['positions'][candidates]
            cam = np.einsum('cij,cj->ci', arrays['rotations'][candidates], relative)
            depth = np.maximum(cam[:, 2], 1e-6)
            focal = arrays['focal'][candidates]
            reach = focal * (patches.polygon.length / 2 + height) / depth
            half = arrays['sizes'][candidates] / 2 + reach[:, None]
            offset = np.abs(focal[:, None] * cam[:, :2] / depth[:, None])
            offset += np.abs(
                arrays['centres'][candidates] - arrays['sizes'][candidates] / 2
            )
            candidates = candidates[(cam[:, 2] > 0.1) & (offset <= half).all(axis=1)]
        if candidates.size and total > 0:
            kwargs = {
                'target_px_per_m': self.target_px_per_m,
                'full_incidence_deg': self.full_incidence_deg,
                'max_incidence_deg': self.max_incidence_deg,
            }

            def take(index):
                return (
                    arrays['positions'][index],
                    arrays['rotations'][index],
                    arrays['focal'][index],
                    arrays['sizes'][index],
                    arrays['centres'][index],
                )

            # Cheap pass without occlusion, to find the photos worth the
            # ray casting: the best few from every viewing direction.
            rough, in_frame = view_quality(*take(candidates), patches, **kwargs)
            score = rough @ patches.weights
            keep = score > 0
            candidates, rough, in_frame, score = (
                candidates[keep],
                rough[keep],
                in_frame[keep],
                score[keep],
            )
            result.n_candidates = int(len(candidates))
        if not candidates.size or total <= 0:
            self._report(result, patches, np.zeros(len(patches.weights)))
            return result
        offsets = arrays['positions'][candidates] - centroid
        distance = np.linalg.norm(offsets, axis=1)
        directions = offsets / np.maximum(distance, 1e-9)[:, None]
        if len(candidates) > self.max_candidates:
            azimuth = np.degrees(np.arctan2(offsets[:, 0], offsets[:, 1])) % 360.0
            elevation = np.degrees(np.arcsin(np.clip(directions[:, 2], -1, 1)))
            sector = (azimuth // 30).astype(int) * 3 + np.digitize(
                elevation, [35.0, 65.0]
            )
            per_sector = max(2, self.max_candidates // 36)
            order = np.argsort(-score)
            counts: dict[int, int] = defaultdict(int)
            chosen_rows = []
            for row in order:
                if counts[int(sector[row])] < per_sector:
                    counts[int(sector[row])] += 1
                    chosen_rows.append(row)
            for row in order:  # fill up with the best of the rest
                if len(chosen_rows) >= self.max_candidates:
                    break
                if row not in chosen_rows:
                    chosen_rows.append(row)
            rows = np.sort(np.asarray(chosen_rows[: self.max_candidates]))
            candidates, in_frame = candidates[rows], in_frame[rows]
            offsets, distance, directions = (
                offsets[rows],
                distance[rows],
                directions[rows],
            )
        quality, _ = view_quality(*take(candidates), patches, edges, tops, **kwargs)
        frame_fraction = (in_frame * patches.weights[None, :]).sum(axis=1) / total
        if self.clip_penalty > 0:
            quality = quality * (frame_fraction**self.clip_penalty)[:, None]
        chosen, gains, best = select_views(
            quality,
            patches.weights,
            self.max_views,
            self.min_gain,
            directions,
            self.diversity,
        )
        covered = np.zeros(len(patches.weights))
        sides = patches.side_of_patch()
        for row, gain in zip(chosen, gains, strict=True):
            camera = self.cameras[int(candidates[row])]
            added = np.maximum(quality[row] - covered, 0.0) * patches.weights
            covered = np.maximum(covered, quality[row])
            shows = []
            for name, mask in [('roof', sides < 0)] + [
                (side, sides == i) for i, side in enumerate(SIDES)
            ]:
                weight = float(patches.weights[mask].sum())
                if weight > 0 and added[mask].sum() / weight >= 0.15:
                    shows.append(name)
            corners = self._prism_corners(patches)
            pixels, depth = camera.project_points(corners, self._project)
            pixels = pixels[depth > 0.1]
            if len(pixels) >= 3:
                hull = MultiPoint([tuple(p) for p in pixels]).convex_hull
                outline = (
                    [(float(x), float(y)) for x, y in hull.exterior.coords]
                    if hull.geom_type == 'Polygon'
                    else [(float(x), float(y)) for x, y in pixels]
                )
                bbox = tuple(float(v) for v in hull.bounds)
            else:
                outline, bbox = [], (0.0, 0.0, 0.0, 0.0)
            result.views.append(
                SelectedView(
                    camera=camera,
                    gain=round(gain, 4),
                    quality=round(float(quality[row] @ patches.weights / total), 4),
                    azimuth_deg=round(
                        float(
                            np.degrees(np.arctan2(offsets[row, 0], offsets[row, 1]))
                            % 360.0
                        ),
                        1,
                    ),
                    elevation_deg=round(
                        float(
                            np.degrees(np.arcsin(np.clip(directions[row, 2], -1, 1)))
                        ),
                        1,
                    ),
                    distance_m=round(float(distance[row]), 1),
                    shows=shows,
                    frame_fraction=round(float(frame_fraction[row]), 3),
                    bbox=bbox,
                    outline=outline,
                )
            )
        self._report(result, patches, best)
        return result

    @staticmethod
    def _prism_corners(patches: SurfacePatches) -> np.ndarray:
        return _corners(patches.polygon, patches.base, patches.top)

    def project_prisms(
        self, camera: DroneCamera
    ) -> dict[str, list[tuple[float, float]]]:
        """
        Outline of every asset of the last :meth:`select` call in one photo.

        Args:
            camera: The photo.

        Returns:
            dict[str, list[tuple[float, float]]]: Asset ID to the pixel
            outline (convex hull) of its projected prism, for the assets in
            front of the camera whose outline touches the frame.

        Example:
            >>> outlines = selector.project_prisms(cameras[0])  # doctest: +SKIP
            >>> sorted(outlines)[:2]  # doctest: +SKIP
            ['b12', 'b13']
        """
        projected: dict[str, list[tuple[float, float]]] = {}
        if self._project is None:
            return projected
        frame = (0.0, 0.0, float(camera.width), float(camera.height))
        for asset_id, (polygon, base, height) in self._prisms.items():
            pixels, depth = camera.project_points(
                _corners(polygon, base, base + height), self._project
            )
            if (depth <= 0.1).any():
                continue
            hull = MultiPoint([tuple(p) for p in pixels]).convex_hull
            if hull.geom_type != 'Polygon':
                continue
            left, top, right, bottom = hull.bounds
            if (
                right < frame[0]
                or left > frame[2]
                or bottom < frame[1]
                or top > frame[3]
            ):
                continue
            projected[asset_id] = [
                (float(x), float(y)) for x, y in hull.exterior.coords
            ]
        return projected

    def _report(
        self, result: AssetViewSet, patches: SurfacePatches, best: np.ndarray
    ) -> None:
        """Fill the coverage figures and the list of unseen walls."""
        weights = patches.weights
        total = float(weights.sum())
        if total <= 0:
            return
        result.coverage = round(float(best @ weights / total), 4)
        result.seen_fraction = round(
            float(weights[best >= self.seen_threshold].sum() / total), 4
        )
        sides = patches.side_of_patch()
        for name, mask in [('roof', sides < 0)] + [
            (side, sides == i) for i, side in enumerate(SIDES)
        ]:
            weight = float(weights[mask].sum())
            if weight > 0:
                result.coverage_by_side[name] = round(
                    float(best[mask] @ weights[mask] / weight), 3
                )
        unproject = self._unproject
        for index, (x0, y0, x1, y1) in enumerate(patches.walls):
            mask = patches.wall == index
            weight = float(weights[mask].sum())
            if weight <= 0:
                continue
            coverage = float(best[mask] @ weights[mask] / weight)
            length = float(math.hypot(x1 - x0, y1 - y0))
            if coverage >= self.seen_threshold or length < 1.0:
                continue
            facing = float(patches.wall_azimuth[index])
            # Where a camera would see this wall well: back along its normal,
            # far enough to frame it, looking down at 45 degrees.
            reach = max(15.0, length)
            mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
            view_x = mid_x + reach * math.sin(math.radians(facing))
            view_y = mid_y + reach * math.cos(math.radians(facing))
            lon, lat = unproject(view_x, view_y)
            result.unseen_walls.append(
                {
                    'wall': index,
                    'facing_deg': round(facing, 1) % 360.0,
                    'side': SIDES[int(math.floor((facing + 45.0) / 90.0)) % 4],
                    'length_m': round(length, 1),
                    'coverage': round(coverage, 3),
                    'line': [list(unproject(x0, y0)), list(unproject(x1, y1))],
                    'viewpoint': {
                        'lon': lon,
                        'lat': lat,
                        'height_m': round(patches.top - patches.base + reach, 1),
                        'yaw_deg': round((facing + 180.0) % 360.0, 1),
                        'pitch_deg': -45.0,
                    },
                }
            )

    # ------------------------------------------------------------ products
    @staticmethod
    def coverage_gaps(view_sets: dict[str, AssetViewSet]) -> PhysicalAssetCollection:
        """
        The walls no chosen photo shows, as a collection of line assets.

        Each line carries the asset it belongs to, the direction the wall
        faces, its length, and a suggested camera position
        (``fly_to_lon``, ``fly_to_lat``, ``fly_height_m`` above the ground
        at the asset, ``fly_yaw_deg``, ``fly_pitch_deg``) from which the
        wall would be seen squarely. Write it with ``to_geojson`` and load
        it in a flight planner or GIS to see where the survey has holes.

        Args:
            view_sets: Output of :meth:`select`.

        Returns:
            PhysicalAssetCollection: One ``LineString`` asset per unseen
            wall.

        Example:
            >>> gaps = DroneViewSelector.coverage_gaps(view_sets)  # doctest: +SKIP
            >>> gaps.to_geojson('coverage_gaps.geojson')  # doctest: +SKIP
        """
        gaps = PhysicalAssetCollection()
        for asset_id, view_set in view_sets.items():
            for wall in view_set.unseen_walls:
                spot = wall['viewpoint']
                gaps.add(
                    PhysicalAsset(
                        id=f'{asset_id}_wall{wall["wall"]}',
                        geometry=LineString(wall['line']),
                        attributes={
                            'asset_type': 'coverage_gap',
                            'asset_id': asset_id,
                            'side': wall['side'],
                            'facing_deg': wall['facing_deg'],
                            'length_m': wall['length_m'],
                            'coverage': wall['coverage'],
                            'fly_to_lon': spot['lon'],
                            'fly_to_lat': spot['lat'],
                            'fly_height_m': spot['height_m'],
                            'fly_yaw_deg': spot['yaw_deg'],
                            'fly_pitch_deg': spot['pitch_deg'],
                        },
                    )
                )
        return gaps


# ------------------------------------------------------------ pipeline step
def _safe_name(text: str) -> str:
    """File-name-safe version of an identifier."""
    return ''.join(c if c.isalnum() or c in '-_.' else '_' for c in str(text))


class DroneImageExtractor:
    """
    Attach to every asset the drone photos that show the most of it.

    A pipeline step for perspective (oblique or nadir) drone photographs.
    For each asset it selects a small set of photos that together cover as
    much of the asset's walls and roof as the flight captured (see
    :class:`DroneViewSelector`), crops each chosen photo around the asset
    and attaches the crops as :class:`~rapidtools.core.ImageAsset` objects,
    most informative first. Analysis steps downstream then see a compact
    multi-view record of every asset.

    The step also writes what the flight did and did not capture to each
    asset: ``drone_view_coverage`` (quality-weighted share of the surface
    shown), ``drone_seen_fraction``, ``drone_coverage_roof`` and
    ``drone_coverage_north`` / ``_east`` / ``_south`` / ``_west``,
    ``drone_unseen_sides`` and ``n_drone_views``.

    Args:
        images (Sequence[DroneCamera] | str | Path):
            The flight: cameras from
            :func:`~rapidtools.data_sources.read_drone_images`, or a folder
            of photos to read with its default settings.
        save_directory (str | Path):
            Where the crops are written.
        max_views (int):
            Most photos kept per asset. Defaults to 4.
        crop_buffer (float | str):
            Margin around the projected asset: a percentage of its longest
            side (``'20%'``, the default) or pixels.
        position_error_m (float):
            Expected error of the camera positions. The crop grows by the
            pixel shift such an error causes at the asset's distance, so a
            pose that is a little off still frames the asset. Defaults to
            2.0 (consumer GPS).
        angle_error_deg (float):
            Expected error of the gimbal angles, handled the same way.
            Defaults to 1.0.
        min_crop_px (int):
            Crops are grown to at least this many pixels on their shorter
            side. Defaults to 256.
        max_crop_px (int | None):
            Crops whose longer side exceeds this are scaled down to it
            before saving. Vision-language models spend time and memory in
            proportion to the pixels they are given, and a building does
            not need thousands of them. ``None`` (default) keeps the
            photo's resolution.
        overlay_asset_outline (bool):
            Mark the asset on each crop so a model knows which building is
            meant. Defaults to ``True``.
        outline_shape (str):
            ``'corners'`` (default: brackets around the projected asset,
            leaving it uncovered), ``'bbox'``, ``'rotated_bbox'``,
            ``'convex_hull'`` or ``'geometry'``.
        outline_width (int | float | str):
            Stroke width in pixels or percent of the shorter crop side.
            Defaults to ``'0.8%'``.
        outline_color (str | tuple[int, int, int]):
            PIL colour of the mark. Defaults to ``'red'``.
        image_prefix (str):
            Prefix of crop file names and image IDs. Defaults to
            ``'drone'``.
        attribute_prefix (str):
            Prefix of the attributes written to assets. Defaults to
            ``'drone'``.
        max_workers (int):
            Photos opened concurrently. Defaults to 4.
        cancel_event (threading.Event | None):
            Cooperative cancellation flag.
        refine_poses (bool):
            Correct each photo's pose error in the image before cropping:
            SAM 3 segments the buildings in every photo a view was chosen
            from, and the projected footprints are shifted to where they
            overlap those buildings best (see :func:`estimate_image_shift`).
            The crop and the mark then sit on the building even when the
            GPS or the compass was a few metres or degrees off. Needs a GPU
            for reasonable speed and enough standing structures in view to
            lock on to; photos where the match is weak keep the recorded
            pose. Defaults to ``False``.
        refine_prompt (str):
            What SAM 3 is asked to segment. Defaults to ``'building'``.
        refine_model (Any | None):
            A loaded :class:`~rapidtools.models.SAM3Inference` (or anything
            with its ``run_inference``); loaded on demand otherwise.
        refine_min_overlap (float):
            Smallest overlap score (intersection over union of projected
            footprints and segmented buildings) at which a shift is
            trusted. Defaults to 0.2.
        selector (DroneViewSelector | None):
            A configured selector to use instead of building one.
        **selector_options:
            Any other :class:`DroneViewSelector` argument, for example
            ``default_height_m``, ``ground_elevation``, ``roof_weight``,
            ``min_gain`` or ``obstacles``.

    Example:
        >>> from rapidtools.core import PhysicalAssetCollection
        >>> from rapidtools.processing import DroneImageExtractor
        >>> buildings = PhysicalAssetCollection.from_geojson(
        ...     'footprints.geojson'
        ... )  # doctest: +SKIP
        >>> extractor = DroneImageExtractor(
        ...     'flight_01/', save_directory='drone_crops', max_views=4,
        ...     default_height_m=6.0,
        ... )  # doctest: +SKIP
        >>> buildings = extractor(buildings)  # doctest: +SKIP
        >>> extractor.coverage_gaps().to_geojson('gaps.geojson')  # doctest: +SKIP
    """

    stage = Stage.EXTRACT_IMAGERY

    def __init__(
        self,
        images: Sequence[DroneCamera] | str | Path,
        save_directory: str | Path,
        max_views: int = 4,
        crop_buffer: float | str = '20%',
        position_error_m: float = 2.0,
        angle_error_deg: float = 1.0,
        min_crop_px: int = 256,
        max_crop_px: int | None = None,
        overlay_asset_outline: bool = True,
        outline_shape: str = 'corners',
        outline_width: int | float | str = '0.8%',
        outline_color: str | tuple[int, int, int] = 'red',
        image_prefix: str = 'drone',
        attribute_prefix: str = 'drone',
        max_workers: int = 4,
        cancel_event: threading.Event | None = None,
        refine_poses: bool = False,
        refine_prompt: str = 'building',
        refine_model: Any | None = None,
        refine_min_overlap: float = 0.2,
        selector: DroneViewSelector | None = None,
        **selector_options: Any,
    ) -> None:
        if outline_shape not in outlines.OUTLINE_SHAPES:
            raise ValueError(
                f'outline_shape must be one of {outlines.OUTLINE_SHAPES}, '
                f'got {outline_shape!r}.'
            )
        if isinstance(images, (str, Path)):
            cameras = read_drone_images(images)
        else:
            cameras = list(images)
        usable = [c for c in cameras if c.path is not None]
        if len(usable) < len(cameras):
            logger.warning(
                f'{len(cameras) - len(usable)} cameras have no image file and '
                'are ignored.'
            )
        self.cameras = usable
        self.save_directory = Path(save_directory).resolve()
        self._crop_buffer_spec = outlines.parse_outline_buffer(crop_buffer)
        self.position_error_m = position_error_m
        self.angle_error_deg = angle_error_deg
        self.min_crop_px = min_crop_px
        self.max_crop_px = max_crop_px
        self.overlay_asset_outline = overlay_asset_outline
        self.outline_shape = outline_shape
        self._outline_width_spec = outlines.parse_outline_width(outline_width)
        outlines.validate_outline_color(outline_color)
        self.outline_color = outline_color
        self.image_prefix = image_prefix
        self.attribute_prefix = attribute_prefix
        self.max_workers = max(1, max_workers)
        self.cancel_event = cancel_event
        self.refine_poses = refine_poses
        self.refine_prompt = refine_prompt
        self.refine_model = refine_model
        self.refine_min_overlap = refine_min_overlap
        #: Image shift applied to each photo by the last refinement.
        self.pose_shifts: dict[str, tuple[int, int]] = {}
        if selector is not None and selector_options:
            raise ValueError('Pass selector options or a selector, not both.')
        self.selector = selector or DroneViewSelector(
            self.cameras,
            max_views=max_views,
            cancel_event=cancel_event,
            **selector_options,
        )
        #: View sets of the last run, keyed by asset ID.
        self.view_sets: dict[str, AssetViewSet] = {}

    # ---------------------------------------------------------------- crops
    def crop_box(self, view: SelectedView) -> tuple[int, int, int, int]:
        """
        Pixel box cropped out of a chosen photo.

        The projected asset plus the ``crop_buffer`` margin plus the shift
        the expected pose error causes at the asset's distance, grown to
        ``min_crop_px`` and clipped to the photo.

        Example:
            >>> box = extractor.crop_box(view)  # doctest: +SKIP
        """
        camera = view.camera
        left, top, right, bottom = view.bbox
        extent = max(right - left, bottom - top, 1.0)
        margin = outlines.resolve_buffer_px(self._crop_buffer_spec, extent, 0.0)
        margin += camera.focal_px * (
            self.position_error_m / max(view.distance_m, 1.0)
            + math.tan(math.radians(self.angle_error_deg))
        )
        left, right = left - margin, right + margin
        top, bottom = top - margin, bottom + margin
        if right - left < self.min_crop_px:
            grow = (self.min_crop_px - (right - left)) / 2
            left, right = left - grow, right + grow
        if bottom - top < self.min_crop_px:
            grow = (self.min_crop_px - (bottom - top)) / 2
            top, bottom = top - grow, bottom + grow
        return (
            max(0, int(math.floor(left))),
            max(0, int(math.floor(top))),
            min(camera.width, int(math.ceil(right))),
            min(camera.height, int(math.ceil(bottom))),
        )

    def _crop_and_save(
        self, image: Image.Image, asset: PhysicalAsset, view: SelectedView, rank: int
    ) -> ImageAsset | None:
        box = self.crop_box(view)
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            return None
        crop = image.crop(box).convert('RGB')
        scale = 1.0
        if self.max_crop_px and max(crop.size) > self.max_crop_px:
            scale = self.max_crop_px / max(crop.size)
            crop = crop.resize(
                (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                Image.Resampling.LANCZOS,
            )
        outline = [
            ((x - box[0]) * scale, (y - box[1]) * scale) for x, y in view.outline
        ]
        if self.overlay_asset_outline and len(outline) >= 3:
            outlines.draw_outline(
                crop,
                outline,
                is_closed=True,
                shape=self.outline_shape,
                width=self._outline_width_spec,
                color=self.outline_color,
            )
        camera = view.camera
        name = (
            f'{self.image_prefix}_{_safe_name(asset.id)}_{_safe_name(camera.image_id)}'
        )
        path = self.save_directory / f'{name}.jpg'
        crop.save(path, quality=92)
        return ImageAsset(
            id=f'{asset.id}_{self.image_prefix}_{camera.image_id}',
            path=path,
            properties={
                'source': 'drone',
                'image_id': camera.image_id,
                'source_image': str(camera.path),
                'captured_at': camera.captured_at,
                'view_rank': rank,
                'view_azimuth_deg': view.azimuth_deg,
                'view_elevation_deg': view.elevation_deg,
                'distance_m': view.distance_m,
                'shows': list(view.shows),
                'view_quality': view.quality,
                'coverage_gain': view.gain,
                'frame_fraction': view.frame_fraction,
                'camera_yaw_deg': round(camera.yaw_deg, 2),
                'camera_pitch_deg': round(camera.pitch_deg, 2),
                'longitude': camera.lon,
                'latitude': camera.lat,
                'crop_box': list(box),
                'crop_scale': round(scale, 4),
                'pose_shift_px': list(self.pose_shifts.get(camera.image_id, (0, 0))),
                'source_image_size': [camera.width, camera.height],
                'outline': [[round(x, 1), round(y, 1)] for x, y in outline],
            },
        )

    def _process_image(
        self, camera: DroneCamera, jobs: list[tuple[PhysicalAsset, SelectedView, int]]
    ) -> dict[tuple[str, str], ImageAsset]:
        raise_if_cancelled(self.cancel_event, 'drone image extraction')
        crops: dict[tuple[str, str], ImageAsset] = {}
        with Image.open(camera.path) as image:
            image.load()
            for asset, view, rank in jobs:
                crop = self._crop_and_save(image, asset, view, rank)
                if crop is not None:
                    crops[(asset.id, camera.image_id)] = crop
        return crops

    def _refine(self, jobs: dict[str, list], cameras: dict[str, DroneCamera]) -> None:
        """Shift the projected outlines of each photo onto the buildings seen."""
        if self.refine_model is None:
            from rapidtools.models import SAM3Inference

            self.refine_model = SAM3Inference()
        self.pose_shifts = {}
        for image_id in tqdm(sorted(jobs), desc='Refining poses'):
            raise_if_cancelled(self.cancel_event, 'drone pose refinement')
            camera = cameras[image_id]
            output = self.refine_model.run_inference(
                [str(camera.path)], prompt=self.refine_prompt
            )
            masks = output.masks[0] if output is not None and output.masks else None
            if masks is None or not len(masks):
                continue
            observed = np.asarray(masks).reshape(-1, *np.asarray(masks).shape[-2:])
            observed = observed.any(axis=0)
            scale = max(1, int(round(max(observed.shape) / 512)))
            observed = observed[::scale, ::scale]
            factor = observed.shape[1] / camera.width
            canvas = Image.new('1', (observed.shape[1], observed.shape[0]), 0)
            draw = ImageDraw.Draw(canvas)
            for outline in self.selector.project_prisms(camera).values():
                draw.polygon([(x * factor, y * factor) for x, y in outline], fill=1)
            distance = min(view.distance_m for _, view, _ in jobs[image_id])
            reach = (
                3.0
                * camera.focal_px
                * (
                    self.position_error_m / max(distance, 1.0)
                    + math.tan(math.radians(self.angle_error_deg))
                )
            )
            dx, dy, score, before = estimate_image_shift(
                np.asarray(canvas, dtype=bool), observed, int(math.ceil(reach * factor))
            )
            # A shift must be a clear improvement: with a good pose the box
            # model and the real roofs never overlap perfectly, and chasing
            # the last few percent only adds jitter.
            if score < self.refine_min_overlap or score < 1.1 * before:
                continue
            shift = (int(round(dx / factor)), int(round(dy / factor)))
            self.pose_shifts[image_id] = shift
            for _, view, _ in jobs[image_id]:
                left, top, right, bottom = view.bbox
                view.bbox = (
                    left + shift[0],
                    top + shift[1],
                    right + shift[0],
                    bottom + shift[1],
                )
                view.outline = [(x + shift[0], y + shift[1]) for x, y in view.outline]
        logger.info(
            f'Pose refinement shifted {len(self.pose_shifts)} of {len(jobs)} photos.'
        )

    def coverage_gaps(self) -> PhysicalAssetCollection:
        """
        Walls the last run found no photo of (see
        :meth:`DroneViewSelector.coverage_gaps`).

        Example:
            >>> extractor.coverage_gaps().to_geojson('gaps.geojson')  # doctest: +SKIP
        """
        return DroneViewSelector.coverage_gaps(self.view_sets)

    def __call__(
        self, asset_collection: PhysicalAssetCollection
    ) -> PhysicalAssetCollection:
        """
        Select views, crop them and attach the crops to the assets.

        Args:
            asset_collection (PhysicalAssetCollection):
                Assets with footprint (or point) geometries.

        Returns:
            PhysicalAssetCollection: The same collection, with crops and
            coverage attributes added. Assets no photo shows get the
            attributes (zero coverage) and no images.
        """
        assets = list(asset_collection)
        if not assets:
            return asset_collection
        if not self.cameras:
            logger.warning('No drone photos with a usable pose; nothing extracted.')
            return asset_collection
        self.view_sets = self.selector.select(assets)
        jobs: dict[str, list[tuple[PhysicalAsset, SelectedView, int]]] = defaultdict(
            list
        )
        by_id = {}
        for asset in assets:
            view_set = self.view_sets.get(asset.id)
            if view_set is None:
                continue
            for rank, view in enumerate(view_set.views, start=1):
                jobs[view.camera.image_id].append((asset, view, rank))
                by_id[view.camera.image_id] = view.camera
        if self.refine_poses and jobs:
            self._refine(jobs, by_id)
        self.save_directory.mkdir(parents=True, exist_ok=True)
        logger.info(
            f'Cropping {sum(len(j) for j in jobs.values())} views of '
            f'{len(assets)} assets from {len(jobs)} of {len(self.cameras)} photos...'
        )
        crops: dict[tuple[str, str], ImageAsset] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._process_image, by_id[image_id], image_jobs): image_id
                for image_id, image_jobs in jobs.items()
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc='Cropping drone photos'
            ):
                try:
                    crops.update(future.result())
                except OSError as exc:
                    logger.error(f'Failed to crop photo {futures[future]}: {exc}')
        prefix = self.attribute_prefix
        for asset in assets:
            view_set = self.view_sets.get(asset.id)
            if view_set is None:
                continue
            attached = 0
            for view in view_set.views:
                crop = crops.get((asset.id, view.camera.image_id))
                if crop is not None:
                    asset.add_image_assets(crop)
                    attached += 1
            sides = view_set.coverage_by_side
            asset.attributes.update(
                {
                    f'n_{prefix}_views': attached,
                    f'{prefix}_view_coverage': view_set.coverage,
                    f'{prefix}_seen_fraction': view_set.seen_fraction,
                    f'{prefix}_unseen_sides': sorted(
                        {wall['side'] for wall in view_set.unseen_walls}
                    ),
                    f'{prefix}_candidate_photos': view_set.n_candidates,
                }
            )
            for side in ('roof', *SIDES):
                if side in sides:
                    asset.attributes[f'{prefix}_coverage_{side}'] = sides[side]
        logger.info(f'Saved {len(crops)} crops to {self.save_directory}.')
        return asset_collection
