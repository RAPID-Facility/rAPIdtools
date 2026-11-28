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
# 11-27-2026

"""
Building-like outlines for segmentation masks.

A footprint traced from a segmentation mask follows the pixels: it is
stair-stepped, its corners are rounded, and trees or shadows along a wall
leave dents and bulges. Buildings are mostly made of straight walls that
meet at right angles, so this module redraws each footprint that way:

1. The outline is smoothed and its dominant wall direction estimated (the
   building's own walls, its minimum rotated rectangle and, for a
   collection, the direction shared by its neighbors are all tried).
2. The footprint is rasterized on a grid rotated to that direction, cleaned
   with square morphology (so details narrower than about a meter go) and
   traced back, which gives walls that are parallel or perpendicular.
3. Steps shorter than about a meter are folded into the walls on either
   side, and a footprint that is nearly a rectangle becomes one.

A result that no longer overlaps the input well (intersection over union
below ``min_iou``), as for round or strongly angled buildings, falls back to
a plain simplification of the outline. All functions here work in a planar
CRS in meters; :class:`FootprintRegularizer` handles the projection for
WGS84 asset collections.

Example:
    >>> from shapely.geometry import Polygon
    >>> from rapidtools.processing.postprocessing.footprints import (
    ...     regularize_footprint,
    ... )
    >>> jagged = Polygon([(0, 0), (10.2, 0.3), (10, 6.1), (5.1, 6.0),
    ...                   (5, 8.9), (0.2, 9.1)])
    >>> clean = regularize_footprint(jagged)
    >>> len(clean.exterior.coords) - 1  # an L-shape: six walls
    6
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pyproj
from rasterio import features as rio_features
from rasterio.transform import from_origin
from scipy import ndimage
from shapely import affinity
from shapely.geometry import MultiPolygon, Point, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.validation import make_valid

from rapidtools.core import PhysicalAssetCollection
from rapidtools.processing.step import Stage

logger = logging.getLogger(__name__)

HALF_PI = math.pi / 2.0

# A regularized outline must keep at least this intersection over union with
# the smoothed input, or the plain simplification is returned instead:
DEFAULT_MIN_IOU = 0.82
# A footprint whose oriented rectangle overlaps it at least this well becomes
# that rectangle:
DEFAULT_RECTANGLE_IOU = 0.92
# Small buildings (sheds, garages) take the rectangle at a lower overlap:
SMALL_BUILDING_AREA_M2 = 40.0
SMALL_RECTANGLE_IOU = 0.85
# Outlines at least this circular (4 pi area / perimeter^2: 1 for a circle,
# 0.95 for an octagon, 0.79 for a square) are round buildings and are only
# simplified:
ROUND_CIRCULARITY = 0.92
# Interior rings smaller than this are filled (trees, vents, shadows):
DEFAULT_HOLE_MIN_AREA_M2 = 12.0
# Footprints smaller than this are dropped by the collection helpers:
DEFAULT_MIN_AREA_M2 = 4.0
# Neighbors within this radius share their wall direction (see
# regularize_footprints):
DEFAULT_NEIGHBOR_RADIUS_M = 40.0
# Supersampling of the rotated grid when estimating cell coverage:
_SUPERSAMPLE = 4


# --------------------------------------------------------------- helpers
def largest_polygon(geometry: BaseGeometry | None) -> Polygon | None:
    """
    Return the largest polygon of a geometry, or ``None`` if it has none.

    Example:
        >>> from shapely.geometry import MultiPolygon, box
        >>> largest_polygon(MultiPolygon([box(0, 0, 1, 1), box(5, 5, 8, 8)])).area
        9.0
    """
    if geometry is None or geometry.is_empty:
        return None
    if geometry.geom_type == 'Polygon':
        return geometry  # type: ignore[return-value]
    polygons: list[Polygon] = []
    for part in getattr(geometry, 'geoms', []):
        if part.geom_type == 'Polygon' and not part.is_empty:
            polygons.append(part)
        elif part.geom_type == 'MultiPolygon':
            polygons.extend(p for p in part.geoms if not p.is_empty)
    return max(polygons, key=lambda p: p.area) if polygons else None


def _valid_polygon(geometry: BaseGeometry | None) -> Polygon | None:
    """Repair a geometry and keep its largest polygon."""
    if geometry is None or geometry.is_empty:
        return None
    if not geometry.is_valid:
        geometry = make_valid(geometry)
    return largest_polygon(geometry)


def iou(a: BaseGeometry, b: BaseGeometry) -> float:
    """
    Intersection over union of two geometries (``0`` when both are empty).

    Example:
        >>> from shapely.geometry import box
        >>> iou(box(0, 0, 2, 1), box(1, 0, 3, 1))
        0.3333333333333333
    """
    inter = a.intersection(b).area
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def local_metric_transformers(
    lon: float, lat: float
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """
    Transformers between WGS84 and a transverse Mercator plane centered on a point.

    The projection is conformal, so right angles on the ground stay right
    angles in the plane, and over a few kilometers its scale error is below
    one part in a million.

    Args:
        lon: Longitude of the projection center.
        lat: Latitude of the projection center.

    Returns:
        tuple: ``(to_local, to_wgs84)`` functions for
        :func:`shapely.ops.transform`; local units are meters.

    Example:
        >>> to_local, to_wgs84 = local_metric_transformers(-118.14, 34.18)
        >>> [round(v, 6) for v in to_local(-118.14, 34.18)]
        [0.0, 0.0]
    """
    local = pyproj.CRS.from_proj4(
        f'+proj=tmerc +lat_0={lat} +lon_0={lon} +k=1 +x_0=0 +y_0=0 '
        '+datum=WGS84 +units=m +no_defs'
    )
    wgs84 = pyproj.CRS.from_epsg(4326)
    to_local = pyproj.Transformer.from_crs(wgs84, local, always_xy=True).transform
    to_wgs84 = pyproj.Transformer.from_crs(local, wgs84, always_xy=True).transform
    return to_local, to_wgs84


def _axis_deviation(angle: float, theta: float) -> float:
    """Angle between a line direction and the nearer of ``theta``, ``theta + 90``."""
    d = (angle - theta) % HALF_PI
    return min(d, HALF_PI - d)


def _douglas_peucker_ring(points: np.ndarray, tolerance: float) -> list[int]:
    """
    Douglas-Peucker simplification of a closed ring, returning kept indices.

    The ring is split at the vertex farthest from the centroid and the
    vertex farthest from that one, two points that are almost always
    corners, so the simplification does not depend on where the ring
    happens to start.
    """
    n = len(points)
    if n <= 3:
        return list(range(n))
    center = points.mean(axis=0)
    start = int(np.argmax(np.hypot(*(points - center).T)))
    far = int(np.argmax(np.hypot(*(points - points[start]).T)))
    order = np.roll(np.arange(n), -start)
    split_at = int(np.nonzero(order == far)[0][0])
    seq = np.vstack([points[order], points[order[:1]]])
    keep = {0, split_at}
    stack = [(0, split_at), (split_at, n)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        a, b = seq[i], seq[j]
        ab = b - a
        length = math.hypot(ab[0], ab[1])
        inner = seq[i + 1 : j]
        if length == 0:
            dist = np.hypot(*(inner - a).T)
        else:
            dist = (
                np.abs(ab[0] * (inner[:, 1] - a[1]) - ab[1] * (inner[:, 0] - a[0]))
                / length
            )
        k = int(np.argmax(dist))
        if dist[k] > tolerance:
            m = i + 1 + k
            keep.add(m)
            stack.extend(((i, m), (m, j)))
    return sorted(int(order[p]) for p in keep if p < n)


def dominant_orientation(
    polygon: Polygon, tolerance: float = 0.5
) -> tuple[float, float]:
    """
    Estimate the main wall direction of a footprint.

    Edges of the Douglas-Peucker simplified outline vote for their direction
    modulo 90 degrees, weighted by length (a circular mean of four times
    each angle, so a wall and the wall perpendicular to it agree). Edges far
    from the result are then left out and the mean recomputed, which keeps
    a diagonal porch or a rounded corner from tilting the estimate.

    Args:
        polygon: Footprint in a planar CRS.
        tolerance: Simplification tolerance in CRS units; it removes pixel
            stair-steps, which would otherwise pull the result toward the
            image axes.

    Returns:
        tuple[float, float]: The direction in radians in ``[0, pi/2)`` and
        its strength in ``[0, 1]`` (1 when every edge is parallel or
        perpendicular to it, near 0 for a round shape).

    Example:
        >>> from shapely import affinity
        >>> from shapely.geometry import box
        >>> rotated = affinity.rotate(box(0, 0, 20, 10), 30, origin=(0, 0))
        >>> theta, strength = dominant_orientation(rotated)
        >>> round(math.degrees(theta)), round(strength, 2)
        (30, 1.0)
    """
    ring = np.asarray(polygon.exterior.coords, dtype=float)[:-1]
    if len(ring) < 3:
        return 0.0, 0.0
    kept = ring[_douglas_peucker_ring(ring, tolerance)]
    closed = np.vstack([kept, kept[:1]])
    deltas = np.diff(closed, axis=0)
    lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    mask = lengths > 0
    deltas, lengths = deltas[mask], lengths[mask]
    if len(lengths) == 0:
        return 0.0, 0.0
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    weights = lengths.copy()
    theta = 0.0
    for _ in range(3):
        s = float((weights * np.sin(4 * angles)).sum())
        c = float((weights * np.cos(4 * angles)).sum())
        if s == 0 and c == 0:
            break
        theta = (math.atan2(s, c) / 4.0) % HALF_PI
        deviation = np.array([_axis_deviation(a, theta) for a in angles])
        weights = np.where(deviation <= math.radians(15), lengths, 0.0)
        if weights.sum() == 0:
            weights = lengths
    s = float((lengths * np.sin(4 * angles)).sum())
    c = float((lengths * np.cos(4 * angles)).sum())
    return theta, math.hypot(s, c) / float(lengths.sum())


def _rectangle_orientation(polygon: Polygon) -> float:
    """Direction of the minimum rotated rectangle's first side, modulo 90 degrees."""
    rectangle = polygon.minimum_rotated_rectangle
    if rectangle.geom_type != 'Polygon':
        return 0.0
    coords = np.asarray(rectangle.exterior.coords)
    d = coords[1] - coords[0]
    return math.atan2(d[1], d[0]) % HALF_PI


def oriented_rectangle(polygon: Polygon, theta: float) -> Polygon:
    """
    The smallest rectangle with sides along ``theta`` that contains a polygon.

    Example:
        >>> from shapely.geometry import box
        >>> oriented_rectangle(box(0, 0, 4, 2), 0.0).area
        8.0
    """
    center = polygon.centroid
    rotated = affinity.rotate(polygon, -theta, origin=center, use_radians=True)
    minx, miny, maxx, maxy = rotated.bounds
    rectangle = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
    return affinity.rotate(rectangle, theta, origin=center, use_radians=True)


# ------------------------------------------------ rectilinear ring cleanup
def _rectilinear_edges(coords: Sequence[Sequence[float]]) -> list[dict[str, Any]]:
    """
    Describe an axis-aligned ring as a list of walls.

    Each wall is ``{'cls': 0 or 1, 'off': float, 'len': float}``: class 0 is
    horizontal (the line ``y = off``), class 1 vertical (``x = off``).
    """
    pts = np.asarray(coords, dtype=float)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    corners = []
    n = len(pts)
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > 1e-12:
            corners.append(b)
    edges = []
    m = len(corners)
    for i in range(m):
        p0, p1 = corners[i], corners[(i + 1) % m]
        if abs(p1[1] - p0[1]) < 1e-9:
            edges.append({'cls': 0, 'off': float(p0[1]), 'len': abs(p1[0] - p0[0])})
        else:
            edges.append({'cls': 1, 'off': float(p0[0]), 'len': abs(p1[1] - p0[1])})
    return edges


def _fold_parallel_neighbors(edges: list[dict[str, Any]]) -> None:
    """Merge consecutive walls of the same class into one, in place."""
    changed = True
    while changed and len(edges) > 4:
        changed = False
        for i in range(len(edges)):
            j = (i + 1) % len(edges)
            a, b = edges[i], edges[j]
            if a['cls'] == b['cls']:
                total = a['len'] + b['len']
                if total > 0:
                    a['off'] = (a['off'] * a['len'] + b['off'] * b['len']) / total
                a['len'] = total
                del edges[j]
                changed = True
                break


def _update_lengths(edges: list[dict[str, Any]]) -> None:
    """Recompute wall lengths from the offsets of the walls on either side."""
    m = len(edges)
    for i in range(m):
        prev, cur, nxt = edges[i - 1], edges[i], edges[(i + 1) % m]
        if prev['cls'] != cur['cls'] and nxt['cls'] != cur['cls']:
            cur['len'] = abs(nxt['off'] - prev['off'])


def _remove_short_steps(
    edges: list[dict[str, Any]], min_step: float
) -> list[tuple[float, float]]:
    """
    Fold steps shorter than ``min_step`` into the walls on either side.

    A short wall between two parallel walls is a step; removing it puts the
    two walls on one line, at their length-weighted mean offset. Steps are
    removed shortest first. Returns the vertices of the cleaned ring.
    """
    guard = 0
    while len(edges) > 4 and guard < 10_000:
        guard += 1
        n = len(edges)
        shortest = None
        for k in range(n):
            if edges[k]['len'] >= min_step:
                continue
            if edges[k - 1]['cls'] == edges[(k + 1) % n]['cls'] and (
                shortest is None or edges[k]['len'] < edges[shortest]['len']
            ):
                shortest = k
        if shortest is None:
            break
        before, after = (shortest - 1) % n, (shortest + 1) % n
        a, b = edges[before], edges[after]
        total = a['len'] + b['len']
        if total > 0:
            a['off'] = (a['off'] * a['len'] + b['off'] * b['len']) / total
        a['len'] = total
        for index in sorted({shortest, after}, reverse=True):
            del edges[index]
        _fold_parallel_neighbors(edges)
        _update_lengths(edges)
    vertices = []
    m = len(edges)
    for i in range(m):
        a, b = edges[i], edges[(i + 1) % m]
        if a['cls'] == b['cls']:
            continue
        if a['cls'] == 0:
            vertices.append((b['off'], a['off']))
        else:
            vertices.append((a['off'], b['off']))
    return vertices


def rectilinear_fit(
    polygon: Polygon,
    theta: float,
    cell: float = 0.25,
    min_feature: float = 1.0,
    min_step: float = 0.8,
    hole_min_area: float = DEFAULT_HOLE_MIN_AREA_M2,
) -> Polygon | None:
    """
    Redraw a footprint with walls parallel or perpendicular to ``theta``.

    The polygon is rotated by ``-theta``, rasterized on a grid of ``cell``
    (a cell is set when at least half of it is covered), closed and opened
    with a square of side ``min_feature``, traced back into an axis-aligned
    outline, cleared of steps shorter than ``min_step`` and rotated back.

    Args:
        polygon: Footprint in a planar CRS (meters).
        theta: Wall direction in radians.
        cell: Grid cell size.
        min_feature: Bulges and dents narrower than this disappear.
        min_step: Offsets in a wall shorter than this are straightened.
        hole_min_area: Interior rings smaller than this are filled.

    Returns:
        Polygon | None: The rectilinear footprint, or ``None`` if nothing is
        left after the cleanup.

    Example:
        >>> from shapely.geometry import box
        >>> fit = rectilinear_fit(box(0, 0, 10, 6).buffer(0.3), 0.0)
        >>> len(fit.exterior.coords) - 1
        4
    """
    center = polygon.centroid
    rotated = affinity.rotate(polygon, -theta, origin=center, use_radians=True)
    minx, miny, maxx, maxy = rotated.bounds
    pad = 2.0 * min_feature + 2.0 * cell
    width = int(math.ceil((maxx - minx + 2 * pad) / cell))
    height = int(math.ceil((maxy - miny + 2 * pad) / cell))
    x0, y0 = minx - pad, maxy + pad
    fine = rio_features.rasterize(
        [(rotated, 1)],
        out_shape=(height * _SUPERSAMPLE, width * _SUPERSAMPLE),
        transform=from_origin(x0, y0, cell / _SUPERSAMPLE, cell / _SUPERSAMPLE),
        dtype=np.uint8,
    )
    coverage = fine.reshape(height, _SUPERSAMPLE, width, _SUPERSAMPLE).mean(axis=(1, 3))
    grid = coverage >= 0.5
    side = max(1, int(round(min_feature / cell)))
    square = np.ones((side, side), dtype=bool)
    grid = ndimage.binary_closing(grid, structure=square)
    grid = ndimage.binary_opening(grid, structure=square)
    labels, count = ndimage.label(grid)
    if count == 0:
        return None
    sizes = ndimage.sum(grid, labels, range(1, count + 1))
    grid = labels == (1 + int(np.argmax(sizes)))
    holes = ndimage.binary_fill_holes(grid) & ~grid
    hole_labels, hole_count = ndimage.label(holes)
    if hole_count:
        hole_areas = ndimage.sum(holes, hole_labels, range(1, hole_count + 1))
        for index, cells in enumerate(hole_areas):
            if cells * cell * cell < hole_min_area:
                grid[hole_labels == index + 1] = True
    traced = [
        shape(geom)
        for geom, value in rio_features.shapes(
            grid.astype(np.uint8),
            mask=grid,
            transform=from_origin(x0, y0, cell, cell),
        )
        if value == 1
    ]
    if not traced:
        return None
    outline = max(traced, key=lambda p: p.area)
    exterior = _remove_short_steps(
        _rectilinear_edges(outline.exterior.coords), min_step
    )
    if len(exterior) < 4:
        return None
    interiors = []
    for ring in outline.interiors:
        cleaned = _remove_short_steps(_rectilinear_edges(ring.coords), min_step)
        if len(cleaned) >= 4:
            interiors.append(cleaned)
    result = _valid_polygon(Polygon(exterior, interiors))
    if result is None:
        return None
    return affinity.rotate(result, theta, origin=center, use_radians=True)


# ------------------------------------------------------------ public API
def _scale_parameters(area: float, min_feature: float | None) -> dict[str, float]:
    """Size-dependent tolerances (meters) for a footprint of ``area`` m^2."""
    size = math.sqrt(area)
    feature = (
        float(np.clip(0.08 * size, 0.6, 1.5)) if min_feature is None else min_feature
    )
    return {
        'tolerance': float(np.clip(0.035 * size, 0.25, 0.8)),
        'cell': max(float(np.clip(0.02 * size, 0.15, 0.35)), size / 400.0),
        'min_feature': feature,
        'min_step': float(np.clip(0.07 * size, 0.6, 1.5))
        if min_feature is None
        else 0.9 * feature,
    }


def regularize_footprint(
    polygon: BaseGeometry,
    orientation: float | None = None,
    orientation_hints: Sequence[float] = (),
    *,
    min_feature: float | None = None,
    rectangle_iou: float = DEFAULT_RECTANGLE_IOU,
    min_iou: float = DEFAULT_MIN_IOU,
    hole_min_area: float = DEFAULT_HOLE_MIN_AREA_M2,
    return_info: bool = False,
) -> Any:
    """
    Turn a mask-traced footprint into a building-like outline.

    Args:
        polygon: The footprint in a planar CRS in meters (``Polygon``; for
            a ``MultiPolygon`` the largest part is used).
        orientation: Wall direction in radians to use instead of estimating
            one.
        orientation_hints: Further directions to try, such as the one shared
            by neighboring buildings. The direction whose outline overlaps
            the footprint best wins, with ties going to the hints.
        min_feature: Width in meters below which bulges, dents and steps are
            removed. Defaults to a value that grows with the building, from
            0.6 m for a shed to 1.5 m for large buildings.
        rectangle_iou: A footprint whose oriented rectangle overlaps it at
            least this well (intersection over union) becomes the rectangle.
        min_iou: The regularized outline must overlap the smoothed footprint
            at least this well; otherwise a simplified outline is returned.
        hole_min_area: Interior rings smaller than this (m^2) are filled.
        return_info: Also return a dictionary with ``'method'``
            (``'rectangle'``, ``'rectilinear'`` or ``'simplified'``),
            ``'orientation'`` (radians) and ``'iou'``.

    Returns:
        Polygon | None: The regularized footprint (``None`` for empty or
        degenerate input), or ``(polygon, info)`` with ``return_info=True``.

    Example:
        >>> from shapely import affinity
        >>> from shapely.geometry import box
        >>> wobbly = affinity.rotate(box(0, 0, 12, 8), 25).buffer(0.4)
        >>> clean = regularize_footprint(wobbly)
        >>> len(clean.exterior.coords) - 1
        4
    """
    info: dict[str, Any] = {'method': None, 'orientation': None, 'iou': 0.0}

    def done(result: Polygon | None) -> Any:
        return (result, info) if return_info else result

    footprint = _valid_polygon(polygon)
    if footprint is None or footprint.area <= 0:
        return done(None)
    area = footprint.area
    params = _scale_parameters(area, min_feature)
    tolerance = params['tolerance']

    # Fill small holes and smooth away pixel noise (close, then open):
    holes = [
        ring for ring in footprint.interiors if Polygon(ring).area >= hole_min_area
    ]
    footprint = Polygon(footprint.exterior, holes)
    radius = 0.5 * tolerance
    smoothed = largest_polygon(
        footprint.buffer(radius).buffer(-2 * radius).buffer(radius)
    )
    if smoothed is None or smoothed.area < 0.25 * area:
        smoothed = footprint

    if orientation is not None:
        candidates = [float(orientation) % HALF_PI]
    else:
        # Round buildings (tanks, silos, gazebos) have no wall direction;
        # forcing square walls on them would draw a staircase:
        outline = _valid_polygon(smoothed.simplify(tolerance, preserve_topology=True))
        if outline is not None and outline.length > 0:
            circularity = 4 * math.pi * outline.area / outline.length**2
            if circularity >= ROUND_CIRCULARITY:
                info.update(method='simplified', iou=iou(outline, smoothed))
                return done(outline)
        own, _strength = dominant_orientation(smoothed, tolerance)
        candidates = [float(t) % HALF_PI for t in orientation_hints]
        candidates += [own, _rectangle_orientation(smoothed)]

    best: tuple[float, Polygon, float] | None = None
    tried: list[float] = []
    for theta in candidates:
        if any(_axis_deviation(theta, t) < math.radians(0.5) for t in tried):
            continue
        tried.append(theta)
        fit = rectilinear_fit(
            smoothed,
            theta,
            cell=params['cell'],
            min_feature=params['min_feature'],
            min_step=params['min_step'],
            hole_min_area=hole_min_area,
        )
        if fit is None:
            continue
        score = iou(fit, smoothed)
        if best is None or score > best[0] + 0.005:
            best = (score, fit, theta)

    if best is not None:
        score, fit, theta = best
        info['orientation'] = theta
        rectangle = oriented_rectangle(smoothed, theta)
        rect_score = iou(rectangle, smoothed)
        if rect_score >= rectangle_iou or (
            area < SMALL_BUILDING_AREA_M2 and rect_score >= SMALL_RECTANGLE_IOU
        ):
            info.update(method='rectangle', iou=rect_score)
            return done(rectangle)
        if score >= min_iou:
            info.update(method='rectilinear', iou=score)
            return done(fit)

    simplified = _valid_polygon(smoothed.simplify(tolerance, preserve_topology=True))
    if simplified is None:
        return done(None)
    info.update(method='simplified', iou=iou(simplified, smoothed))
    return done(simplified)


def _neighbor_orientations(
    polygons: Sequence[Polygon | None], radius: float
) -> list[float | None]:
    """
    The wall direction shared by each footprint's neighbors.

    For every footprint, the directions of the other footprints whose
    centroid lies within ``radius`` are averaged (four-times-angle circular
    mean weighted by area and orientation strength). ``None`` where a
    footprint has no neighbors that agree on a direction.
    """
    centroids: list[Point | None] = []
    own: list[tuple[float, float] | None] = []
    for poly in polygons:
        if poly is None or poly.is_empty:
            own.append(None)
            centroids.append(None)
            continue
        theta, strength = dominant_orientation(
            poly, _scale_parameters(poly.area, None)['tolerance']
        )
        own.append((theta, strength * poly.area))
        centroids.append(poly.centroid)
    points = [c for c in centroids if c is not None]
    index_of = [i for i, c in enumerate(centroids) if c is not None]
    if len(points) < 2:
        return [None] * len(polygons)
    tree = STRtree(points)
    result: list[float | None] = [None] * len(polygons)
    for position, i in enumerate(index_of):
        hits = tree.query(points[position].buffer(radius))
        s = c = total = 0.0
        for hit in hits:
            j = index_of[int(hit)]
            entry = own[j]
            if j == i or entry is None:
                continue
            theta, weight = entry
            s += weight * math.sin(4 * theta)
            c += weight * math.cos(4 * theta)
            total += weight
        if total > 0 and math.hypot(s, c) / total >= 0.5:
            result[i] = (math.atan2(s, c) / 4.0) % HALF_PI
    return result


def resolve_overlaps(
    polygons: Sequence[Polygon | None],
    priorities: Sequence[float] | None = None,
    min_area: float = DEFAULT_MIN_AREA_M2,
) -> list[Polygon | None]:
    """
    Remove the overlap between neighboring footprints.

    Footprints are visited from the highest priority (by default the
    largest area) down; each one loses whatever it shares with footprints
    already kept. Slivers thinner than 0.3 m are trimmed from what remains,
    the largest piece is kept, and footprints left smaller than
    ``min_area`` become ``None``.

    Args:
        polygons: Footprints in a planar CRS in meters (``None`` entries are
            passed through).
        priorities: Larger values keep contested area. Defaults to the area.
        min_area: Smallest footprint kept, in m^2.

    Returns:
        list[Polygon | None]: Non-overlapping footprints, in input order.

    Example:
        >>> from shapely.geometry import box
        >>> a, b = resolve_overlaps([box(0, 0, 10, 10), box(9, 0, 15, 10)])
        >>> b.bounds
        (10.0, 0.0, 15.0, 10.0)
    """
    result: list[Polygon | None] = list(polygons)
    valid = [i for i, p in enumerate(result) if p is not None and not p.is_empty]
    if len(valid) < 2:
        return result
    weights = (
        [result[i].area for i in valid]  # type: ignore[union-attr]
        if priorities is None
        else [priorities[i] for i in valid]
    )
    order = [valid[k] for k in np.argsort(-np.asarray(weights), kind='stable')]
    kept_geoms: list[Polygon] = []
    tree: STRtree | None = None
    kept_at_build = 0
    for i in order:
        poly = result[i]
        assert poly is not None
        if kept_geoms:
            if tree is None or kept_at_build != len(kept_geoms):
                tree = STRtree(kept_geoms)
                kept_at_build = len(kept_geoms)
            hits = [kept_geoms[int(h)] for h in tree.query(poly)]
            overlapping = [g for g in hits if g.intersection(poly).area > 1e-6]
            if overlapping:
                remainder = poly.difference(unary_union(overlapping))
                # Trim slivers thinner than 0.3 m; mitred joins keep the
                # square corners square, and the intersection guarantees
                # the opening cannot grow back into a neighbor:
                opened = remainder.buffer(-0.15, join_style='mitre').buffer(
                    0.15, join_style='mitre'
                )
                poly = largest_polygon(  # type: ignore[assignment]
                    _valid_polygon(opened.intersection(remainder))
                )
                # The intersection can leave a vertex in the middle of a
                # straight wall, off the line only by float noise (seen on
                # Linux builds of GEOS); a micrometer simplify drops it
                # without moving any real corner:
                if poly is not None:
                    poly = poly.simplify(1e-6, preserve_topology=True)
        if poly is None or poly.is_empty or poly.area < min_area:
            result[i] = None
            continue
        result[i] = poly
        kept_geoms.append(poly)
    return result


def regularize_footprints(
    polygons: Sequence[BaseGeometry | None],
    *,
    priorities: Sequence[float] | None = None,
    neighbor_radius: float | None = DEFAULT_NEIGHBOR_RADIUS_M,
    remove_overlaps: bool = True,
    min_area: float = DEFAULT_MIN_AREA_M2,
    min_feature: float | None = None,
    rectangle_iou: float = DEFAULT_RECTANGLE_IOU,
    min_iou: float = DEFAULT_MIN_IOU,
) -> list[Polygon | None]:
    """
    Regularize a set of neighboring footprints together.

    Each footprint is redrawn with :func:`regularize_footprint`, also trying
    the wall direction its neighbors share (houses along a street are
    usually parallel), and the results are then made non-overlapping with
    :func:`resolve_overlaps`.

    Args:
        polygons: Footprints in a planar CRS in meters.
        priorities: Which footprint keeps contested area (larger wins);
            defaults to the area.
        neighbor_radius: Search radius for the shared wall direction, in
            meters; ``None`` disables it.
        remove_overlaps: Resolve overlaps between the results.
        min_area: Results smaller than this (m^2) become ``None``.
        min_feature: See :func:`regularize_footprint`.
        rectangle_iou: See :func:`regularize_footprint`.
        min_iou: See :func:`regularize_footprint`.

    Returns:
        list[Polygon | None]: One entry per input, ``None`` where the
        footprint was empty, degenerate or too small.

    Example:
        >>> from shapely.geometry import box
        >>> out = regularize_footprints([box(0, 0, 10, 8), box(9.5, 0, 20, 8)])
        >>> round(out[0].intersection(out[1]).area, 6)
        0.0
    """
    cleaned = [
        _valid_polygon(p) if p is not None and not p.is_empty else None
        for p in polygons
    ]
    hints: list[float | None] = (
        _neighbor_orientations(cleaned, neighbor_radius)
        if neighbor_radius
        else [None] * len(cleaned)
    )
    result: list[Polygon | None] = []
    for poly, hint in zip(cleaned, hints, strict=True):
        if poly is None or poly.area < min_area:
            result.append(None)
            continue
        regular = regularize_footprint(
            poly,
            orientation_hints=() if hint is None else (hint,),
            min_feature=min_feature,
            rectangle_iou=rectangle_iou,
            min_iou=min_iou,
        )
        result.append(
            regular if regular is not None and regular.area >= min_area else None
        )
    if remove_overlaps:
        result = resolve_overlaps(result, priorities=priorities, min_area=min_area)
    return result


class FootprintRegularizer:
    """
    Pipeline component that gives building footprints straight, square walls.

    This is pure geometry and needs no model or imagery: every polygonal
    asset in the collection gets the outline from
    :func:`regularize_footprints` (computed in a local transverse Mercator
    plane), neighboring footprints are made non-overlapping, and footprints
    that end up smaller than ``min_area_m2`` are dropped. Assets keep their
    IDs and attributes; non-polygonal assets pass through unchanged.

    Args:
        min_area_m2 (float, optional): Smallest footprint kept, in m^2.
            Defaults to 4.
        min_feature_m (float | None, optional): Bulges, dents and steps
            narrower than this are removed. Defaults to a size-dependent
            value between 0.6 and 1.5 m.
        rectangle_iou (float, optional): Footprints whose oriented rectangle
            overlaps them at least this well become rectangles. Defaults to
            0.92.
        neighbor_radius_m (float | None, optional): Radius in which
            neighbors share their wall direction; ``None`` disables it.
            Defaults to 40.
        remove_overlaps (bool, optional): Make the footprints mutually
            non-overlapping (higher ``confidence`` attribute, then larger
            area, keeps the contested part). Defaults to ``True``.
        keep_original (bool, optional): Store the input outline as WKT in
            the ``'unregularized_geometry_wkt'`` attribute. Defaults to
            ``False``.

    Example:
        >>> from rapidtools.core import PhysicalAssetCollection
        >>> from rapidtools.processing import FootprintRegularizer
        >>> raw = PhysicalAssetCollection.from_geojson('buildings_raw.geojson')
        >>> clean = FootprintRegularizer()(raw)
        >>> clean.to_geojson('buildings_clean.geojson')
    """

    stage = Stage.REGULARIZE

    def __init__(
        self,
        min_area_m2: float = DEFAULT_MIN_AREA_M2,
        min_feature_m: float | None = None,
        rectangle_iou: float = DEFAULT_RECTANGLE_IOU,
        neighbor_radius_m: float | None = DEFAULT_NEIGHBOR_RADIUS_M,
        remove_overlaps: bool = True,
        keep_original: bool = False,
    ) -> None:
        """
        Store the regularization settings.

        Args:
            min_area_m2: Smallest footprint kept, in m^2.
            min_feature_m: Width below which details are removed, in meters.
            rectangle_iou: Overlap above which a footprint becomes a
                rectangle.
            neighbor_radius_m: Radius for the shared wall direction.
            remove_overlaps: Make the footprints non-overlapping.
            keep_original: Keep the input outline as an attribute.
        """
        self.min_area_m2 = min_area_m2
        self.min_feature_m = min_feature_m
        self.rectangle_iou = rectangle_iou
        self.neighbor_radius_m = neighbor_radius_m
        self.remove_overlaps = remove_overlaps
        self.keep_original = keep_original

    def __call__(self, assets: PhysicalAssetCollection) -> PhysicalAssetCollection:
        """
        Regularize the footprints of a collection.

        Args:
            assets: Assets with WGS84 geometries.

        Returns:
            PhysicalAssetCollection: The same asset objects with regularized
            geometries, without the polygonal ones that vanished.

        Example:
            >>> clean = FootprintRegularizer(min_area_m2=10)(raw_buildings)
        """
        polygonal = [
            a
            for a in assets
            if a.geometry is not None
            and not a.geometry.is_empty
            and a.geometry.geom_type in ('Polygon', 'MultiPolygon')
        ]
        if not polygonal:
            return assets
        center = MultiPolygon(
            [p for a in polygonal for p in _polygon_parts(a.geometry)]
        ).centroid
        to_local, to_wgs84 = local_metric_transformers(center.x, center.y)
        local = [shapely_transform(to_local, a.geometry) for a in polygonal]
        priorities = [
            float(a.attributes.get('confidence') or 0.0) * 1e6 + geom.area
            for a, geom in zip(polygonal, local, strict=True)
        ]
        regular = regularize_footprints(
            local,
            priorities=priorities,
            neighbor_radius=self.neighbor_radius_m,
            remove_overlaps=self.remove_overlaps,
            min_area=self.min_area_m2,
            min_feature=self.min_feature_m,
            rectangle_iou=self.rectangle_iou,
        )
        result = PhysicalAssetCollection()
        dropped = 0
        polygonal_ids = {id(a) for a in polygonal}
        regular_by_asset = {
            id(a): geom for a, geom in zip(polygonal, regular, strict=True)
        }
        for asset in assets:
            if id(asset) not in polygonal_ids:
                result.add(asset)
                continue
            geom = regular_by_asset[id(asset)]
            if geom is None:
                dropped += 1
                continue
            if self.keep_original:
                asset.attributes['unregularized_geometry_wkt'] = asset.geometry.wkt
            asset.geometry = shapely_transform(to_wgs84, geom)
            result.add(asset)
        if dropped:
            logger.info(
                f'{dropped} footprints were too small or degenerate and dropped.'
            )
        return result


def _polygon_parts(geometry: BaseGeometry) -> list[Polygon]:
    """The polygons of a Polygon or MultiPolygon."""
    if geometry.geom_type == 'Polygon':
        return [geometry]  # type: ignore[list-item]
    return [g for g in getattr(geometry, 'geoms', []) if g.geom_type == 'Polygon']
