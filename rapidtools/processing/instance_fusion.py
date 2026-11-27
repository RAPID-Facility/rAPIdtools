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
Combining instance detections from overlapping views into single objects.

Scanning a raster in overlapping tiles, or segmenting overlapping crops,
shows most objects several times: once whole, and a few more times cut by a
tile edge. Dissolving every detection that touches another melts neighbors
together (two houses whose masks meet become one), and so does bridging the
gaps inside a mask. This module keeps neighbors apart instead:

1. **Parts.** A detection whose mask falls apart into separate pieces, or
   whose pieces are joined only by a neck narrower than ``neck_width``, is
   split; each piece is a candidate of its own.
2. **Links.** Two candidates from *different* views are the same object
   when they agree where both views could see: their intersection over
   union inside the overlap of the two view windows is at least
   ``link_iou``. That compares an object cut by a tile edge with the whole
   object only where the cut view saw it.
3. **Clusters.** Links are followed from the strongest down, but a cluster
   never takes two candidates from the same view: the model saw those as
   separate objects, so they stay separate.
4. **Consensus.** The outline of a cluster is the median of its members'
   signed distance fields, each counted only inside its own view, which is
   the boundary most views agree on and is robust to one stray view.
5. **Conflicts.** An outline that mostly consists of two or more other
   outlines is a group detection and is dropped in favor of the parts
   (unless the parts were seen by far fewer views); remaining overlaps go to
   the object detected by more views.

Geometries must be in a planar CRS (meters for the default tolerances).

Example:
    >>> from shapely.geometry import box
    >>> from rapidtools.processing.instance_fusion import (
    ...     ViewDetection, fuse_view_detections,
    ... )
    >>> windows = {'t1': box(0, 0, 50, 50), 't2': box(25, 0, 75, 50)}
    >>> detections = [
    ...     ViewDetection(box(30, 10, 40, 20), 't1', 0.9),
    ...     ViewDetection(box(30.2, 10, 40.1, 20.2), 't2', 0.8),
    ...     ViewDetection(box(41, 10, 48, 20), 't2', 0.7),  # a neighbor
    ... ]
    >>> objects = fuse_view_detections(detections, windows)
    >>> sorted(o.n_views for o in objects)
    [1, 2]
"""

from __future__ import annotations

import logging
import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import shapely
from rasterio import features as rio_features
from rasterio.transform import from_origin
from scipy import ndimage
from shapely.errors import GEOSException
from shapely.geometry import Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.validation import make_valid

logger = logging.getLogger(__name__)

DEFAULT_LINK_IOU = 0.5
DEFAULT_NECK_WIDTH = 1.0
DEFAULT_MIN_AREA = 3.0
# An outline is a group of others when parts that are each at least this
# much inside it ...
_GROUP_PART_INSIDE = 0.6
# ... and each at most this large relative to it ...
_GROUP_PART_MAX_SIZE = 0.8
# ... together cover at least this much of it:
_GROUP_COVERAGE = 0.5
# A view "saw" an object when its window holds this much of the object:
_SEEN_FRACTION = 0.95
# An object that loses more than this share of its area to better
# supported neighbors is dropped rather than kept as a leftover:
_MAX_LOSS = 0.5


@dataclass
class ViewDetection:
    """
    One object detected in one view (tile or crop).

    Attributes:
        geometry: The detected outline in a planar CRS; multi-part outlines
            are allowed and are split into their parts.
        view: Identifier of the view, a key of the ``windows`` mapping
            passed to :func:`fuse_view_detections`.
        score: The model's confidence, if it reported one.
    """

    geometry: BaseGeometry
    view: Hashable
    score: float | None = None


@dataclass
class FusedObject:
    """
    One object after fusing all views.

    Attributes:
        geometry: The consensus outline (planar CRS).
        score: Mean confidence of the contributing detections (``None`` if
            none reported one).
        n_views: How many views detected the object.
        n_seen: How many views had the whole object in their window
            (at least ``n_views`` for objects inside a single window).
        members: Indices of the contributing detections in the input list.
    """

    geometry: Polygon
    score: float | None
    n_views: int
    n_seen: int
    members: tuple[int, ...] = field(default_factory=tuple)

    @property
    def detection_rate(self) -> float:
        """Share of the views that saw the object and also detected it."""
        return self.n_views / max(self.n_seen, self.n_views, 1)


def _polygons(geometry: BaseGeometry | None) -> list[Polygon]:
    """Valid, non-empty polygon parts of any geometry."""
    if geometry is None or geometry.is_empty:
        return []
    if not geometry.is_valid:
        geometry = make_valid(geometry)
    if geometry.geom_type == 'Polygon':
        return [geometry] if not geometry.is_empty else []  # type: ignore[list-item]
    parts: list[Polygon] = []
    for part in getattr(geometry, 'geoms', []):
        parts.extend(_polygons(part))
    return parts


def _polygonal(geometry: BaseGeometry | None) -> BaseGeometry:
    """
    The area part of a geometry: lines and points are dropped.

    Cutting a polygon with another can leave a collection of an area and the
    line or point where the two only touch; GEOS cannot always overlay such
    collections ("Unable to determine overlay result geometry dimension").
    """
    if geometry is not None and geometry.geom_type in ('Polygon', 'MultiPolygon'):
        return geometry
    parts = _polygons(geometry)
    return unary_union(parts) if parts else Polygon()


def _polygonal_array(geometries: np.ndarray) -> np.ndarray:
    """:func:`_polygonal` over an array, leaving polygons untouched."""
    kinds = shapely.get_type_id(geometries)
    if np.isin(kinds, (3, 6)).all():  # Polygon, MultiPolygon
        return geometries
    out = np.empty(len(geometries), dtype=object)
    for k, geometry in enumerate(geometries):
        out[k] = _polygonal(geometry)
    return out


def _overlap_area(a: BaseGeometry, b: BaseGeometry) -> float:
    """Area of the intersection of two outlines, 0 when GEOS cannot tell."""
    try:
        return float(a.intersection(b).area)
    except GEOSException:
        try:
            return float(
                make_valid(a).buffer(0).intersection(make_valid(b).buffer(0)).area
            )
        except GEOSException:
            return 0.0


def split_into_parts(
    geometry: BaseGeometry,
    neck_width: float = DEFAULT_NECK_WIDTH,
    min_area: float = DEFAULT_MIN_AREA,
    min_fraction: float = 0.1,
) -> list[Polygon]:
    """
    Split a detected outline into the objects it may contain.

    Separate pieces become separate parts. A piece whose halves are joined
    only by a strip narrower than ``neck_width`` (a mask bleeding across the
    gap between two roofs) is cut at the strip: an opening with a disk of
    that diameter finds the halves, and each half takes back the original
    outline within ``neck_width / 2`` of it.

    Args:
        geometry: Outline in a planar CRS.
        neck_width: Strips narrower than this are cut; ``0`` only separates
            disconnected pieces.
        min_area: Parts smaller than this are dropped.
        min_fraction: Parts smaller than this share of the largest part are
            dropped as specks.

    Returns:
        list[Polygon]: The parts, largest first.

    Example:
        >>> from shapely.geometry import box
        >>> dumbbell = box(0, 0, 10, 10).union(box(10, 4.8, 12, 5.2)).union(
        ...     box(12, 0, 22, 10))
        >>> [round(p.area) for p in split_into_parts(dumbbell)]
        [100, 100]
    """
    parts: list[Polygon] = []
    radius = neck_width / 2.0
    for piece in _polygons(geometry):
        if piece.area < min_area:
            continue
        if radius <= 0:
            parts.append(piece)
            continue
        cores = [
            core
            for core in _polygons(piece.buffer(-radius).buffer(radius))
            if core.area >= min_area
        ]
        if len(cores) <= 1:
            parts.append(piece)
            continue
        claimed: BaseGeometry | None = None
        for core in sorted(cores, key=lambda p: p.area, reverse=True):
            region = _polygonal(piece.intersection(core.buffer(radius * 1.5)))
            if claimed is not None:
                region = region.difference(claimed)
            sub = max(_polygons(region), key=lambda p: p.area, default=None)
            if sub is None or sub.area < min_area:
                continue
            parts.append(sub)
            claimed = sub if claimed is None else claimed.union(sub)
    parts.sort(key=lambda p: p.area, reverse=True)
    if parts and min_fraction > 0:
        floor = min_fraction * parts[0].area
        parts = [p for p in parts if p.area >= floor]
    return parts


class _UnionFind:
    """Union-find over candidates that refuses to join two of the same view."""

    def __init__(self, views: Sequence[Hashable]) -> None:
        self.parent = list(range(len(views)))
        self.views: list[set[Hashable]] = [{v} for v in views]

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb or self.views[ra] & self.views[rb]:
            return False
        self.parent[rb] = ra
        self.views[ra] |= self.views[rb]
        return True


def _consensus_outline(
    members: Sequence[Polygon],
    windows: Sequence[BaseGeometry],
    resolution: float | None,
) -> Polygon | None:
    """
    Median shape of several outlines, each trusted only inside its window.

    Every outline is rasterized on a common grid and turned into a signed
    distance field (positive inside); outside its window a member has no
    say. The consensus is where the median field is positive.
    """
    if len(members) == 1:
        return members[0]
    minx = min(m.bounds[0] for m in members)
    miny = min(m.bounds[1] for m in members)
    maxx = max(m.bounds[2] for m in members)
    maxy = max(m.bounds[3] for m in members)
    if resolution is None:
        median_area = float(np.median([m.area for m in members]))
        resolution = float(np.clip(math.sqrt(median_area) / 100.0, 0.05, 0.25))
    pad = 3 * resolution
    width = int(math.ceil((maxx - minx + 2 * pad) / resolution))
    height = int(math.ceil((maxy - miny + 2 * pad) / resolution))
    transform = from_origin(minx - pad, maxy + pad, resolution, resolution)
    fields = np.full((len(members), height, width), np.nan, dtype=np.float32)
    for k, (member, window) in enumerate(zip(members, windows, strict=True)):
        inside = rio_features.rasterize(
            [(member, 1)],
            out_shape=(height, width),
            transform=transform,
            dtype=np.uint8,
        ).astype(bool)
        seen = rio_features.rasterize(
            [(window, 1)],
            out_shape=(height, width),
            transform=transform,
            dtype=np.uint8,
        ).astype(bool)
        if not inside.any():
            continue
        signed = ndimage.distance_transform_edt(
            inside
        ) - ndimage.distance_transform_edt(~inside)
        fields[k][seen] = signed[seen]
    with np.errstate(all='ignore'):
        counted = np.sum(~np.isnan(fields), axis=0)
        median = np.where(
            counted > 0, np.nanmedian(np.where(counted > 0, fields, 0), axis=0), -1
        )
    region = median > 0
    if not region.any():
        return None
    traced = [
        shape(geom)
        for geom, value in rio_features.shapes(
            region.astype(np.uint8), mask=region, transform=transform
        )
        if value == 1
    ]
    return max(traced, key=lambda p: p.area) if traced else None


def _link_candidates(
    geoms: list[Polygon],
    views: list[Hashable],
    windows: Mapping[Hashable, BaseGeometry],
    link_iou: float,
    min_area: float,
) -> list[tuple[float, int, int]]:
    """Cross-view pairs that agree inside their common window, strongest first."""
    if len(geoms) < 2:
        return []
    tree = STRtree(geoms)
    left, right = tree.query(geoms, predicate='intersects')
    keep = left < right
    left, right = left[keep], right[keep]
    view_index: dict[Hashable, int] = {}
    view_arr = np.array([view_index.setdefault(v, len(view_index)) for v in views])
    keep = view_arr[left] != view_arr[right]
    left, right = left[keep], right[keep]
    if len(left) == 0:
        return []
    common_cache: dict[tuple[Hashable, Hashable], BaseGeometry] = {}
    commons = []
    for i, j in zip(left.tolist(), right.tolist(), strict=True):
        key = (views[i], views[j])
        if key not in common_cache:
            common_cache[key] = windows[views[i]].intersection(windows[views[j]])
        commons.append(common_cache[key])
    garr = np.asarray(geoms, dtype=object)
    carr = np.asarray(commons, dtype=object)
    a = _polygonal_array(shapely.intersection(garr[left], carr))
    b = _polygonal_array(shapely.intersection(garr[right], carr))
    try:
        inter = shapely.area(shapely.intersection(a, b))
    except GEOSException:
        # One awkward pair fails the whole vectorized call; go pair by pair:
        inter = np.array([_overlap_area(x, y) for x, y in zip(a, b, strict=True)])
    union = shapely.area(a) + shapely.area(b) - inter
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = np.where(union > 0, inter / union, 0.0)
    ok = (scores >= link_iou) & (union >= min_area)
    edges = [
        (float(s), int(i), int(j))
        for s, i, j in zip(scores[ok], left[ok], right[ok], strict=True)
    ]
    edges.sort(key=lambda e: -e[0])
    return edges


def _resolve_conflicts(
    objects: list[FusedObject], min_area: float
) -> list[FusedObject]:
    """Drop group detections and give contested area to the better supported object."""
    if len(objects) < 2:
        return objects
    geoms = [o.geometry for o in objects]
    tree = STRtree(geoms)
    alive = [True] * len(objects)

    # 1. Group detections: an outline mostly made of two or more others.
    order = sorted(range(len(objects)), key=lambda k: -geoms[k].area)
    for a in order:
        if not alive[a]:
            continue
        ga = geoms[a]
        parts = []
        for b in tree.query(ga, predicate='intersects'):
            b = int(b)
            if b == a or not alive[b]:
                continue
            gb = geoms[b]
            if gb.area > _GROUP_PART_MAX_SIZE * ga.area:
                continue
            shared = ga.intersection(gb).area
            if shared >= _GROUP_PART_INSIDE * gb.area:
                parts.append((b, shared))
        if len(parts) < 2 or sum(s for _, s in parts) < _GROUP_COVERAGE * ga.area:
            continue
        part_support = float(np.mean([objects[b].n_views for b, _ in parts]))
        if part_support >= 0.5 * objects[a].n_views:
            alive[a] = False
        else:
            for b, _ in parts:
                alive[b] = False

    # 2. Remaining overlaps: better supported objects keep the contested area.
    def priority(k: int) -> tuple[float, float, float]:
        score = objects[k].score
        return (
            -objects[k].n_views,
            -(score if score is not None else 0.0),
            -geoms[k].area,
        )

    survivors = sorted((k for k in range(len(objects)) if alive[k]), key=priority)
    # Kept outlines only ever shrink, so querying the tree of the original
    # outlines finds every kept outline that can still overlap:
    kept: dict[int, Polygon] = {}
    for k in survivors:
        geom = geoms[k]
        blockers = [
            kept[int(h)]
            for h in tree.query(geom, predicate='intersects')
            if int(h) in kept and kept[int(h)].intersection(geom).area > 1e-6
        ]
        if blockers:
            rest = geom
            for other in blockers:
                rest = rest.difference(other)
            piece = max(_polygons(rest), key=lambda p: p.area, default=None)
            if piece is None or piece.area < (1 - _MAX_LOSS) * geom.area:
                continue
            geom = piece
        if geom.area < min_area:
            continue
        objects[k].geometry = geom
        kept[k] = geom
    return [objects[k] for k in sorted(kept)]


def fuse_view_detections(
    detections: Sequence[ViewDetection],
    windows: Mapping[Hashable, BaseGeometry],
    *,
    link_iou: float = DEFAULT_LINK_IOU,
    neck_width: float = DEFAULT_NECK_WIDTH,
    min_area: float = DEFAULT_MIN_AREA,
    resolution: float | None = None,
    resolve_conflicts: bool = True,
) -> list[FusedObject]:
    """
    Fuse detections from overlapping views into one outline per object.

    See the module docstring for the method.

    Args:
        detections: Detections from all views, in a planar CRS.
        windows: For every view, the area it covered (its tile or crop
            footprint, without padding), in the same CRS.
        link_iou: Two detections from different views are the same object
            when their intersection over union inside the views' common
            window reaches this value.
        neck_width: Detections are cut where they narrow below this width
            (CRS units); ``0`` disables the cut.
        min_area: Parts and objects smaller than this are dropped.
        resolution: Grid cell for the consensus outlines; by default about
            a hundredth of the object's size, between 0.05 and 0.25.
        resolve_conflicts: Drop group detections and remove overlaps
            between objects.

    Returns:
        list[FusedObject]: The fused objects.

    Raises:
        KeyError: If a detection's view has no window.

    Example:
        >>> objects = fuse_view_detections(detections, windows, link_iou=0.4)
        >>> objects[0].geometry.area > 0
        True
    """
    geoms: list[Polygon] = []
    views: list[Hashable] = []
    scores: list[float | None] = []
    sources: list[int] = []
    for index, det in enumerate(detections):
        if det.view not in windows:
            raise KeyError(f'No window was given for view {det.view!r}.')
        for part in split_into_parts(det.geometry, neck_width, min_area):
            geoms.append(part)
            views.append(det.view)
            scores.append(det.score)
            sources.append(index)
    if not geoms:
        return []

    edges = _link_candidates(geoms, views, windows, link_iou, min_area)
    groups = _UnionFind(views)
    for _score, i, j in edges:
        groups.union(i, j)
    clusters: dict[int, list[int]] = {}
    for k in range(len(geoms)):
        clusters.setdefault(groups.find(k), []).append(k)

    window_keys = list(windows)
    window_geoms = [windows[k] for k in window_keys]
    window_tree = STRtree(window_geoms)

    objects: list[FusedObject] = []
    for members in clusters.values():
        outline = _consensus_outline(
            [geoms[k] for k in members],
            [windows[views[k]] for k in members],
            resolution,
        )
        if outline is None or outline.is_empty or outline.area < min_area:
            continue
        if not outline.is_valid:
            outline = max(_polygons(outline), key=lambda p: p.area, default=None)
            if outline is None:
                continue
        seen = 0
        for w in window_tree.query(outline, predicate='intersects'):
            if (
                window_geoms[int(w)].intersection(outline).area
                >= _SEEN_FRACTION * outline.area
            ):
                seen += 1
        member_scores = [s for s in (scores[k] for k in members) if s is not None]
        n_views = len({views[k] for k in members})
        objects.append(
            FusedObject(
                geometry=outline,
                score=float(np.mean(member_scores)) if member_scores else None,
                n_views=n_views,
                n_seen=max(seen, n_views),
                members=tuple(sorted({sources[k] for k in members})),
            )
        )
    if resolve_conflicts:
        objects = _resolve_conflicts(objects, min_area)
    logger.debug(
        f'Fused {len(detections)} detections ({len(geoms)} parts, '
        f'{len(edges)} links) into {len(objects)} objects.'
    )
    return objects
