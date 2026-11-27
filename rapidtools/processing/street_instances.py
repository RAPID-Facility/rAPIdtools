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
# 11-12-2026

"""
Buildings from one outline per building, linked across frames by 3-D points.

A segmentation model that is asked for houses returns one mask per house in
every image: it knows a fence from a wall and one house from its neighbor,
which a detector that outlines a whole street front as one surface does
not. What it does not know is which mask in the next image is the same
house. The survey's own structure-from-motion point cloud answers that,
because a reconstructed point is the same point in every frame that sees it.

This module takes such per-building outlines (from any model: it never runs
one) and the point clouds, and returns the same
:class:`~rapidtools.processing.street_structures.Structure` records as
:mod:`~rapidtools.processing.street_structures`:

1. **Collect.** Every outline collects the reconstructed points that
   project inside it and that its frame can see.
2. **Link.** Two outlines in different frames show the same building when
   each holds most of the other's points. Links are made strongest first,
   and a frame shows a building once: two outlines of one frame are never
   linked into the same track unless they overlap in the image.
3. **Assign.** A point is on a building when most of the frames that see it
   place it inside an outline, and it belongs to the track most of those
   outlines are in.
4. **Join.** A model often outlines the wings of a large house separately.
   Two tracks are one building when their walls continue each other along
   the street and their outlines touch in the frames in front of them;
   the outlines of two neighbors have the side yard between them there.
   With a promptable model at hand (``whole_object``), the question is put
   to it instead: asked for the whole object at a few points on one wing,
   it returns a mask that covers the other wing, and does not cover a
   neighbor. That is the more reliable test of the two.
5. **Outline.** Footprints and sightings are made as for any structure. A
   building's sightings are its own outlines.

Example:
    >>> from rapidtools.processing.street_instances import discover_instances
    >>> structures, counts, unproject = discover_instances(  # doctest: +SKIP
    ...     outlines, clusters={'1012': ('93789', 'https://...')}
    ... )
    >>> counts['tracks'], counts['structures']  # doctest: +SKIP
    (95, 66)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from rapidtools.core import Observation

from . import street_structures as _structures
from .street_localization import local_projection, rotation_matrix
from .street_structures import (
    PointCloud,
    Structure,
    fetch_sfm_cluster,
    footprint_rectangle,
    footprint_stepped,
    front_most,
    ground_level,
    project_points,
    spread_views,
)

logger = logging.getLogger(__name__)

Project = Callable[[float, float], tuple[float, float]]
Unproject = Callable[[float, float], tuple[float, float]]

#: Points two outlines must share to be compared at all.
_MIN_SHARED = 15
#: Points an outline must hold to be linked to another.
_MIN_OUTLINE_POINTS = 20
#: Two outlines of one frame are one object seen twice (a house and its
#: garage, say) when this share of the smaller lies inside the other.
_SAME_FRAME_OVERLAP = 0.5
#: Frames in which the outlines of two tracks must overlap for the tracks to
#: be taken for one object.
_SAME_FRAME_COUNT = 2
#: Outlines closer than this, in normalized image units, touch.
_TOUCH = 0.008
#: Share of the frames asked that must show two tracks touching.
_TOUCH_SHARE = 0.6
#: Frames asked whether two tracks touch, nearest first.
_FRAMES_ASKED = 6
#: Frames in which a model is asked for the whole object, nearest first.
_FRAMES_PROMPTED = 4
#: Share of a track's outline that the whole object found from the other
#: track must cover for the two to be one building.
_WHOLE_COVER = 0.5

#: ``whole_object(image_id, points)``: the candidate outlines of the object
#: under a few image points (normalized ``(x, y)``), each candidate a list of
#: rings. A promptable segmentation model answers this with the part, the
#: whole and what lies between.
WholeObject = Callable[[str, np.ndarray], Sequence[Sequence[np.ndarray]]]


def mask_outlines(
    mask: np.ndarray, min_area: float = 2e-4, tolerance: float = 2e-3
) -> list[np.ndarray]:
    """
    Outer outlines of a binary mask in normalized image coordinates.

    Args:
        mask: ``(height, width)`` array; nonzero cells are inside.
        min_area: Parts smaller than this share of the image are dropped.
        tolerance: Outlines are simplified to within this distance, in
            normalized units.

    Returns:
        list[np.ndarray]: One ``(n, 2)`` array of ``(x, y)`` vertices per
        part of the mask, both in ``[0, 1]``, largest part first. Holes are
        not kept.

    Example:
        >>> import numpy as np
        >>> mask = np.zeros((50, 100), bool)
        >>> mask[10:30, 20:60] = True
        >>> (ring,) = mask_outlines(mask)
        >>> ring.min(axis=0).round(2).tolist(), ring.max(axis=0).round(2).tolist()
        ([0.2, 0.2], [0.6, 0.6])
    """
    from skimage import measure

    grid = np.asarray(mask).astype(bool)
    if grid.ndim != 2 or not grid.any():
        return []
    height, width = grid.shape
    parts: list[Polygon] = []
    for contour in measure.find_contours(np.pad(grid, 1).astype(np.uint8), 0.5):
        if len(contour) < 4:  # pragma: no cover - a contour closes on itself
            continue
        # contours run through cell centers of the padded grid; cell k of the
        # mask spans k .. k + 1, so its center lies at k + 0.5
        ring = np.c_[(contour[:, 1] - 0.5) / width, (contour[:, 0] - 0.5) / height]
        part = Polygon(ring)
        if not part.is_valid:  # pragma: no cover - cells meeting at a corner
            part = part.buffer(0)
        if not part.is_empty and part.area >= min_area:
            parts.extend(_polygons(part))
    parts.sort(key=lambda p: -p.area)
    outer: list[Polygon] = []
    for part in parts:  # a hole is a ring inside a larger one
        if not any(o.contains(part.representative_point()) for o in outer):
            outer.append(part)
    rings = []
    for part in outer:
        for piece in _polygons(part.simplify(tolerance)):
            if piece.area >= min_area:
                rings.append(np.clip(np.array(piece.exterior.coords[:-1]), 0.0, 1.0))
    return rings


def _polygons(geometry: BaseGeometry) -> list[Polygon]:
    """The polygons of a geometry, whatever kind it is."""
    if isinstance(geometry, Polygon):
        return [] if geometry.is_empty else [geometry]
    return [g for g in getattr(geometry, 'geoms', []) for g in _polygons(g)]


# --------------------------------------------------------------- collect
def survey_points(
    templates: Mapping[str, Observation],
    clusters: Mapping[str, tuple[str, str]],
    project: Project,
    *,
    cache_dir: str | Path | None = None,
    camera_height_m: float = 2.4,
    max_range_m: float = 40.0,
    max_workers: int = 8,
    loader: Callable[[str, str], PointCloud | None] | None = None,
    counts: dict[str, int] | None = None,
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """
    The reconstructed points around a set of frames, and the frames' poses.

    Args:
        templates: One sighting per image; its ``computed_rotation`` finds
            the image's shot in its point cloud.
        clusters: ``(cluster_id, url)`` of the point cloud of every image.
        project: Projection from longitude and latitude to local meters.
        cache_dir: Folder where downloaded point clouds are cached.
        camera_height_m: Camera height above the road, for heights where a
            cloud shows no ground.
        max_range_m: Points farther than this from every frame are left out.
        max_workers: Threads for point-cloud downloads.
        loader: ``loader(cluster_id, url)`` used instead of downloading.
        counts: Counters to add to.

    Returns:
        tuple: ``(n, 4)`` points (meters east and north, altitude, height
        above the ground) and, for every usable image, its world-to-camera
        rotation matrix and camera center.

    Example:
        >>> points, shots = survey_points(  # doctest: +SKIP
        ...     templates, clusters, project)
        >>> points.shape[1], len(shots)  # doctest: +SKIP
        (4, 884)
    """
    counts = counts if counts is not None else defaultdict(int)
    by_cluster: dict[str, tuple[str, list[str]]] = {}
    for image_id in templates:
        cluster_id, url = clusters[image_id]
        by_cluster.setdefault(str(cluster_id), (url, []))[1].append(image_id)
    counts['clusters'] = len(by_cluster)
    load = loader or (lambda cid, url: fetch_sfm_cluster(cid, url, cache_dir))
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        clouds = list(pool.map(lambda kv: load(kv[0], kv[1][0]), by_cluster.items()))
    kept: list[np.ndarray] = []
    shots: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for (_, image_ids), cloud in zip(by_cluster.values(), clouds, strict=True):
        if cloud is None or not len(cloud.points) or not len(cloud.rotations):
            counts['clusters_missing'] += 1
            continue
        origin = np.array([*project(cloud.lon, cloud.lat), 0.0])
        index = cloud.shot_index()
        cameras = []
        for image_id in image_ids:
            rotation = templates[image_id].extra['computed_rotation']
            n = index.get(tuple(np.round(np.asarray(rotation, float), 5).tolist()))
            if n is None:
                counts['frames_unmatched'] += 1
                continue
            r = rotation_matrix(cloud.rotations[n])
            center = -r.T @ cloud.translations[n]
            shots[image_id] = (r, center + origin)
            cameras.append(center)
        if not cameras:
            continue
        cams = np.array(cameras)
        dist, nearest = cKDTree(cams[:, :2]).query(cloud.points[:, :2])
        near = dist < max_range_m
        if not near.any():
            continue
        ground = ground_level(cloud.points, cloud.points[near])
        road = cams[nearest[near], 2] - camera_height_m
        height = cloud.points[near, 2] - np.where(np.isnan(ground), road, ground)
        kept.append(np.c_[cloud.points[near] + origin, height])
    points = np.vstack(kept) if kept else np.zeros((0, 4))
    counts['points'] = len(points)
    return points, shots


def outline_memberships(
    points: np.ndarray,
    shots: Mapping[str, tuple[np.ndarray, np.ndarray]],
    outlines: Mapping[str, Mapping[Any, Sequence[np.ndarray]]],
    max_range_m: float = 40.0,
) -> tuple[sparse.csr_matrix, np.ndarray, list[str], list[BaseGeometry], set]:
    """
    Which reconstructed points every outline holds.

    Args:
        points: ``(n, 3+)`` points in the local frame.
        shots: Rotation matrix and camera center of every image.
        outlines: For every image, its instances: the rings (normalized
            ``(u, v)`` vertices) of each.
        max_range_m: Farthest a frame collects points.

    Returns:
        tuple: A sparse ``(points, outlines)`` matrix with a one where the
        outline holds the point and its frame can see it; for every point,
        the number of frames that see it; the image of every outline; its
        shape in the image; and the pairs of outlines of one frame that
        overlap.

    Example:
        >>> held, seen, images, shapes, twice = outline_memberships(  # doctest: +SKIP
        ...     points, shots, outlines)
        >>> held.shape  # doctest: +SKIP
        (1017554, 10989)
    """
    tree = cKDTree(points[:, :2]) if len(points) else None
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    seen = np.zeros(len(points), np.int32)
    images: list[str] = []
    shapes: list[BaseGeometry] = []
    twice: set[tuple[int, int]] = set()
    for image_id, instances in outlines.items():
        if image_id not in shots or not instances or tree is None:
            continue
        rotation, camera = shots[image_id]
        near = np.asarray(tree.query_ball_point(camera[:2], max_range_m), dtype=int)
        if not len(near):
            continue
        u, v, dist = project_points(points[near, :3] - camera, rotation, np.zeros(3))
        ok = (dist > 2.0) & (dist < max_range_m)
        near, u, v, dist = near[ok], u[ok], v[ok], dist[ok]
        visible = front_most(u, v, dist)
        seen[near[visible]] += 1
        first = len(images)
        for rings in instances.values():
            inside = np.zeros(len(near), bool)
            for ring in rings:
                inside |= _structures._in_outline(u, v, ring)
            held = near[inside & visible]
            rows.append(held)
            cols.append(np.full(len(held), len(images)))
            images.append(image_id)
            shapes.append(_shape(rings))
        for a in range(first, len(images)):
            for b in range(a + 1, len(images)):
                if not shapes[a].intersects(shapes[b]):
                    continue
                shared = shapes[a].intersection(shapes[b]).area
                smaller = min(shapes[a].area, shapes[b].area)
                if shared > _SAME_FRAME_OVERLAP * smaller:
                    twice.add((a, b))
    if not images:
        return (
            sparse.csr_matrix((len(points), 0), dtype=np.float32),
            seen,
            [],
            [],
            twice,
        )
    row, col = np.concatenate(rows), np.concatenate(cols)
    held_by = sparse.csr_matrix(
        (np.ones(len(row), np.float32), (row, col)), shape=(len(points), len(images))
    )
    return held_by, seen, images, shapes, twice


def _shape(rings: Sequence[np.ndarray]) -> BaseGeometry:
    """The area of the image a set of rings covers."""
    parts = []
    for ring in rings:
        part = Polygon(ring)
        parts.append(part if part.is_valid else part.buffer(0))
    return unary_union(parts)


# --------------------------------------------------------------- link
def link_outlines(
    held_by: sparse.csr_matrix,
    frame: np.ndarray,
    twice: set,
    min_overlap: float = 0.5,
) -> np.ndarray:
    """
    Group outlines of different frames that show the same building.

    Two outlines show one building when each holds at least
    ``min_overlap`` of the other's points, counting only the points the
    other frame has inside an outline at all. The pairs are joined
    strongest first into tracks, and a track never takes two outlines of
    one frame unless they overlap in the image (``twice``): a frame shows a
    building once.

    Args:
        held_by: Sparse ``(points, outlines)`` membership matrix.
        frame: Frame number of every outline.
        twice: Pairs of outlines of one frame that overlap.
        min_overlap: Share of each outline's points the other must hold.

    Returns:
        np.ndarray: Track number of every outline.

    Example:
        >>> import numpy as np
        >>> from scipy import sparse
        >>> # Two frames. Outlines 0 and 2 hold points 0-19, 1 and 3 hold 20-39.
        >>> held = np.zeros((40, 4), np.float32)
        >>> held[:20, [0, 2]] = 1; held[20:, [1, 3]] = 1
        >>> frame = np.array([0, 0, 1, 1])
        >>> tracks = link_outlines(sparse.csr_matrix(held), frame, set())
        >>> bool(tracks[0] == tracks[2]), bool(tracks[1] == tracks[3])
        (True, True)
        >>> len(set(tracks.tolist()))
        2
    """
    n = held_by.shape[1]
    parent = np.arange(n)
    if n < 2:
        return parent
    frame = np.asarray(frame, dtype=np.int64)
    size = np.asarray(held_by.sum(axis=0)).ravel()
    shared = (held_by.T @ held_by).tocoo()
    in_frame = sparse.csr_matrix(
        (np.ones(n, np.float32), (np.arange(n), frame)), shape=(n, int(frame.max()) + 1)
    )
    outlined = (held_by @ in_frame).tocsr()  # point inside some outline of the frame
    outlined.data[:] = 1.0
    there = (held_by.T @ outlined).tocsr()  # an outline's points outlined in a frame
    pick = (
        (shared.row < shared.col)
        & (frame[shared.row] != frame[shared.col])
        & (shared.data >= _MIN_SHARED)
    )
    a, b, both = shared.row[pick], shared.col[pick], shared.data[pick]
    of_a = both / np.asarray(there[a, frame[b]]).ravel()
    of_b = both / np.asarray(there[b, frame[a]]).ravel()
    pick = (
        (of_a >= min_overlap)
        & (of_b >= min_overlap)
        & (size[a] >= _MIN_OUTLINE_POINTS)
        & (size[b] >= _MIN_OUTLINE_POINTS)
    )
    a, b, both, of_a, of_b = a[pick], b[pick], both[pick], of_a[pick], of_b[pick]
    members: dict[int, dict[int, list[int]]] = {
        i: {int(frame[i]): [i]} for i in range(n)
    }

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def join(x: int, y: int) -> None:
        x, y = find(x), find(y)
        if x == y:
            return
        if len(members[x]) > len(members[y]):
            x, y = y, x
        for number, outlines in members[x].items():
            for other in members[y].get(number, ()):
                for one in outlines:
                    if (min(one, other), max(one, other)) not in twice:
                        return
        parent[x] = y
        for number, outlines in members.pop(x).items():
            members[y].setdefault(number, []).extend(outlines)

    for k in np.argsort(-both * np.minimum(of_a, of_b)):
        join(int(a[k]), int(b[k]))
    # Tracks whose outlines overlap in several frames show one object twice.
    tally: dict[tuple[int, int], int] = defaultdict(int)
    for one, other in twice:
        x, y = find(one), find(other)
        if x != y:
            tally[(min(x, y), max(x, y))] += 1
    for (x, y), count in sorted(tally.items(), key=lambda item: -item[1]):
        if count >= _SAME_FRAME_COUNT:
            join(x, y)
    return np.array([find(i) for i in range(n)])


def track_points(
    held_by: sparse.csr_matrix,
    seen: np.ndarray,
    frame: np.ndarray,
    track: np.ndarray,
    min_vote_fraction: float = 0.7,
    min_votes: int = 3,
    min_frames: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign every reconstructed point to the track it belongs to, if any.

    A point is on a building when at least ``min_votes`` frames, and
    ``min_vote_fraction`` of the frames that see it, place it inside an
    outline. It belongs to the track that most of those frames outline it
    with. Tracks seen in fewer than ``min_frames`` frames get no points.

    Returns:
        tuple[np.ndarray, np.ndarray]: For every point, its track number
        (meaningless where the second array is ``False``) and whether it
        was assigned.

    Example:
        >>> owner, assigned = track_points(  # doctest: +SKIP
        ...     held_by, seen, frame, track)
        >>> int(assigned.sum())  # doctest: +SKIP
        183204
    """
    n_points, n = held_by.shape
    if not n:
        return np.zeros(n_points, np.int64), np.zeros(n_points, bool)
    frame = np.asarray(frame, dtype=np.int64)
    n_frames = int(frame.max()) + 1
    tracks, of_track = np.unique(track, return_inverse=True)
    # one vote per frame and track, however many of its outlines hold the point
    pairs, of_pair = np.unique(of_track * n_frames + frame, return_inverse=True)
    to_pair = sparse.csr_matrix(
        (np.ones(n, np.float32), (np.arange(n), of_pair)), shape=(n, len(pairs))
    )
    voted = (held_by @ to_pair).tocsr()
    voted.data[:] = 1.0
    pair_track = pairs // n_frames
    to_track = sparse.csr_matrix(
        (np.ones(len(pairs), np.float32), (np.arange(len(pairs)), pair_track)),
        shape=(len(pairs), len(tracks)),
    )
    votes = (voted @ to_track).tocsr()
    best = np.asarray(votes.argmax(axis=1)).ravel()
    most = np.asarray(votes.max(axis=1).todense()).ravel()
    in_frame = sparse.csr_matrix(
        (np.ones(n, np.float32), (np.arange(n), frame)), shape=(n, n_frames)
    )
    outlined = (held_by @ in_frame).tocsr()
    outlined.data[:] = 1.0
    total = np.asarray(outlined.sum(axis=1)).ravel()
    length = np.bincount(pair_track, minlength=len(tracks))
    assigned = (
        (total >= min_votes)
        & (total >= min_vote_fraction * np.maximum(seen, 1))
        & (most >= min_votes)
        & (most >= 0.5 * total)
        & (length[best] >= min_frames)
    )
    return tracks[best], assigned


# --------------------------------------------------------------- pipeline
def discover_instances(
    observations: Sequence[Observation],
    clusters: Mapping[str, tuple[str, str]],
    *,
    cache_dir: str | Path | None = None,
    camera_height_m: float = 2.4,
    max_range_m: float = 40.0,
    min_vote_fraction: float = 0.7,
    min_votes: int = 3,
    min_overlap: float = 0.5,
    min_height_m: float = 2.0,
    cell_m: float = 0.5,
    min_cell_points: int = 2,
    min_gap_m: float = 1.0,
    min_points: int = 40,
    min_frontage_m: float = 4.0,
    assumed_depth_m: float = 10.0,
    footprint: str = 'stepped',
    max_views: int = 12,
    max_workers: int = 8,
    project: Project | None = None,
    unproject: Unproject | None = None,
    loader: Callable[[str, str], PointCloud | None] | None = None,
    whole_object: WholeObject | None = None,
) -> tuple[list[Structure], dict[str, int], Unproject]:
    """
    Find buildings from one outline per building and frame.

    See the module docstring for the method.

    Args:
        observations: Per-building outlines (panoramas with a
            ``computed_rotation`` in ``extra``). Outlines of one image that
            carry the same ``extra['instance']`` are parts of one mask; an
            outline without it is an instance of its own.
        clusters: For every image ID, the ``(cluster_id, url)`` of its
            ``sfm_cluster``. Images without an entry are not used.
        cache_dir: Folder where downloaded point clouds are cached.
        camera_height_m: Camera height above the road; sets the ground
            level where a point cloud has no ground points of its own.
        max_range_m: Farthest an outline collects points, and farthest
            frame a building gets a sighting from.
        min_vote_fraction: Share of the frames that see a point that must
            place it inside an outline.
        min_votes: Frames that must place a point inside an outline.
        min_overlap: Share of each other's points two outlines must hold to
            be taken for the same building.
        min_height_m: Points lower than this above the ground are ignored.
        cell_m: Ground cell used to tell wall from scattered points.
        min_cell_points: Fewest points a wall cell holds.
        min_gap_m: Tracks more than three times this apart along the
            street are never joined into one building.
        min_points: Fewest points a building may have.
        min_frontage_m: Shortest facade reported.
        assumed_depth_m: Footprint depth where the rear was not seen.
        footprint: ``'stepped'``, ``'rectangle'`` or ``'observed'``, as in
            :func:`~rapidtools.processing.street_structures.discover_structures`.
        max_views: Sightings kept per building; ``0`` keeps one per frame.
        max_workers: Threads for point-cloud downloads.
        project: Projection to local meters; made from the sightings when
            omitted.
        unproject: Inverse of ``project``.
        loader: ``loader(cluster_id, url)`` returning a
            :class:`~rapidtools.processing.street_structures.PointCloud`,
            instead of downloading (for tests and offline use).
        whole_object: ``whole_object(image_id, points)`` returning the
            candidate outlines of the object under a few normalized image
            points, each a list of rings (a promptable segmentation model).
            When given, it decides whether two tracks are wings of one
            building; otherwise touching outlines do.

    Returns:
        tuple[list[Structure], dict[str, int], Unproject]: The buildings,
        counters describing the run and the inverse projection.

    Example:
        >>> structures, counts, unproject = discover_instances(  # doctest: +SKIP
        ...     outlines, clusters, cache_dir='point_clouds'
        ... )
        >>> structures[0].frontage_m, len(structures[0].members)  # doctest: +SKIP
        (21.4, 12)
    """
    if footprint not in ('stepped', 'rectangle', 'observed'):
        raise ValueError(
            "footprint must be 'stepped', 'rectangle' or 'observed', "
            f'got {footprint!r}.'
        )
    counts: dict[str, int] = defaultdict(int)
    counts['sightings'] = len(observations)
    if project is None or unproject is None:
        lon0 = float(np.mean([o.camera_lon for o in observations] or [0.0]))
        lat0 = float(np.mean([o.camera_lat for o in observations] or [0.0]))
        project, unproject = local_projection(lon0, lat0)

    # 1. Collect.
    outlines: dict[str, dict[Any, list[np.ndarray]]] = defaultdict(dict)
    templates: dict[str, Observation] = {}
    for number, obs in enumerate(observations):
        rotation = (obs.extra or {}).get('computed_rotation')
        if rotation is None or not obs.is_pano or obs.image_id not in clusters:
            counts['unusable'] += 1
            continue
        instance = (obs.extra or {}).get('instance', f'_{number}')
        outlines[obs.image_id].setdefault(instance, []).append(
            np.asarray(obs.polygon, dtype=float)
        )
        templates.setdefault(obs.image_id, obs)
    counts['structures'] = 0
    if not templates:
        counts['clusters'] = 0
        return [], dict(counts), unproject
    points, shots = survey_points(
        templates,
        clusters,
        project,
        cache_dir=cache_dir,
        camera_height_m=camera_height_m,
        max_range_m=max_range_m,
        max_workers=max_workers,
        loader=loader,
        counts=counts,
    )
    held_by, seen, images, shapes, twice = outline_memberships(
        points, shots, outlines, max_range_m
    )
    counts['frames'] = len(set(images))
    counts['outlines'] = len(images)
    if not images:
        return [], dict(counts), unproject

    # 2. Link and 3. assign.
    image_ids = sorted(set(images))
    number_of = {image_id: n for n, image_id in enumerate(image_ids)}
    frame = np.array([number_of[image_id] for image_id in images])
    track = link_outlines(held_by, frame, twice, min_overlap)
    owner, assigned = track_points(
        held_by, seen, frame, track, min_vote_fraction, min_votes
    )
    assigned &= points[:, 3] >= min_height_m
    cameras = np.array([shots[image_id][1][:2] for image_id in image_ids])
    camera_tree = cKDTree(cameras)
    units: list[_Unit] = []
    for number in np.unique(owner[assigned]):
        rows = points[assigned & (owner == number)]
        if len(rows) < max(15, min_points // 2):
            counts['too_few_points'] += 1
            continue
        own: dict[int, list[int]] = defaultdict(list)
        for position in np.flatnonzero(track == number):
            own[int(frame[position])].append(int(position))
        units.append(_Unit(rows, dict(own)))
    counts['tracks'] = len(units)

    # 4. Join the wings of one building.
    groups = _join_units(
        units,
        shapes,
        cameras,
        camera_tree,
        cell_m=cell_m,
        min_cell_points=min_cell_points,
        max_gap_m=_structures._JOIN_GAP_FACTOR * min_gap_m,
        max_range_m=max_range_m,
        assumed_depth_m=assumed_depth_m,
        whole_object=whole_object,
        image_ids=image_ids,
    )
    counts['joined'] = len(units) - len(groups)

    # 5. Outline and sightings.
    structures: list[Structure] = []
    for group in groups:
        rows = np.vstack([units[i].points for i in group])
        if len(rows) < min_points:
            counts['too_few_points'] += 1
            continue
        dense = _wall_points(rows, cell_m, min_cell_points)
        wall = rows[dense]
        largest = units[max(group, key=lambda i: len(units[i].points))].points
        straight = largest[_wall_points(largest, cell_m, min_cell_points)]
        center = wall[:, :2].mean(axis=0)
        _, nearest = camera_tree.query(center)
        street = _structures._street_direction(cameras, camera_tree, center)
        beyond = rows[~dense, :2] if (~dense).any() else wall[:, :2]
        outline, frontage, depth, observed, source, facing = (
            footprint_stepped if footprint == 'stepped' else footprint_rectangle
        )(
            wall[:, :2],
            beyond,
            cameras[nearest],
            street,
            assumed_depth_m,
            axis_points=straight[:, :2],
        )
        if frontage < min_frontage_m:
            counts['too_short'] += 1
            continue
        if footprint == 'observed':
            hull = MultiPoint(wall[:, :2]).convex_hull
            if not isinstance(hull, Polygon):  # pragma: no cover - a wall has width
                counts['too_short'] += 1
                continue
            outline = [(float(x), float(y)) for x, y in hull.exterior.coords[:-1]]
            depth, source = observed, 'observed'
        shown: dict[int, list[int]] = defaultdict(list)
        for i in group:
            for number, held in units[i].outlines.items():
                shown[number].extend(held)
        sightings = _sightings(
            wall, shown, shapes, image_ids, shots, templates, max_range_m, max_views
        )
        if not sightings:
            counts['unseen'] += 1
            continue
        area = Polygon(outline)
        lon, lat = unproject(area.centroid.x, area.centroid.y)
        for obs in sightings:
            obs.lon, obs.lat = lon, lat
        structures.append(
            Structure(
                footprint=outline,
                x=float(area.centroid.x),
                y=float(area.centroid.y),
                members=sightings,
                n_points=len(rows),
                frontage_m=frontage,
                depth_m=depth,
                observed_depth_m=observed,
                depth_source=source,
                height_m=float(np.percentile(wall[:, 3], 97)),
                facade_bearing=facing,
                days=sorted({o.captured_at[:10] for o in sightings if o.captured_at}),
            )
        )
    counts['structures'] = len(structures)
    return structures, dict(counts), unproject


class _Unit:
    """The points of one track and, for every frame, the outlines it holds there."""

    def __init__(self, points: np.ndarray, outlines: dict[int, list[int]]) -> None:
        self.points = points
        self.outlines = outlines


def _wall_points(rows: np.ndarray, cell_m: float, min_cell_points: int) -> np.ndarray:
    """Which of a building's points stand in its dense ground cells."""
    cells = np.floor(rows[:, :2] / cell_m).astype(np.int64)
    _, cell, count = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    cut = max(min_cell_points, int(np.percentile(count, _structures._DENSITY_QUANTILE)))
    dense = count[np.ravel(cell)] >= cut
    return np.asarray(dense if dense.sum() >= 15 else np.ones(len(rows), bool))


def _join_units(
    units: Sequence[_Unit],
    shapes: Sequence[BaseGeometry],
    cameras: np.ndarray,
    camera_tree: cKDTree,
    *,
    cell_m: float,
    min_cell_points: int,
    max_gap_m: float,
    max_range_m: float,
    assumed_depth_m: float,
    whole_object: WholeObject | None = None,
    image_ids: Sequence[str] = (),
) -> list[list[int]]:
    """
    Join the tracks that are wings of one building.

    A track joins the one larger track whose wall it continues most closely
    along the street, when the frames nearest the place where the two meet
    show them as one: by a model asked for the whole object
    (``whole_object``), or else by their outlines touching.
    """
    walls, boxes = [], []
    for unit in units:
        wall = unit.points[_wall_points(unit.points, cell_m, min_cell_points)]
        center = wall[:, :2].mean(axis=0)
        _, nearest = camera_tree.query(center)
        street = _structures._street_direction(cameras, camera_tree, center)
        walls.append(wall)
        boxes.append(
            footprint_rectangle(
                wall[:, :2],
                unit.points[:, :2],
                cameras[nearest],
                street,
                assumed_depth_m,
            )
        )
    parent = list(range(len(units)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    best: dict[int, tuple[float, float, int]] = {}
    if len(units) > 1:
        centers = np.array([w[:, :2].mean(axis=0) for w in walls])
        for a, b in sorted(cKDTree(centers).query_pairs(40.0)):
            big, small = (a, b) if len(walls[a]) >= len(walls[b]) else (b, a)
            gap, step = _structures._wall_offsets(
                boxes[big][0], boxes[big][3], walls[small][:, :2]
            )
            if gap >= max_gap_m or step >= _structures._MAX_STEP_M:
                continue
            if _one_building(
                units[a],
                units[b],
                walls[a],
                walls[b],
                shapes,
                cameras,
                max_range_m,
                whole_object,
                image_ids,
            ):
                tie = (gap, step, big)
                best[small] = min(best.get(small, tie), tie)
    for small, (_, _, big) in best.items():
        parent[find(small)] = find(big)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(units)):
        groups[find(i)].append(i)
    return list(groups.values())


def _touching(
    first: _Unit,
    second: _Unit,
    first_wall: np.ndarray,
    second_wall: np.ndarray,
    shapes: Sequence[BaseGeometry],
    cameras: np.ndarray,
    max_range_m: float,
) -> bool:
    """Whether the frames in front of two tracks show their outlines touching."""
    votes: list[bool] = []
    for number in _frames_in_front(
        first, second, first_wall, second_wall, cameras, max_range_m, _FRAMES_ASKED
    ):
        one = unary_union([shapes[i] for i in first.outlines[number]])
        other = unary_union([shapes[i] for i in second.outlines[number]])
        votes.append(one.distance(other) <= _TOUCH)
    return bool(votes) and sum(votes) >= _TOUCH_SHARE * len(votes)


def _one_building(
    first: _Unit,
    second: _Unit,
    first_wall: np.ndarray,
    second_wall: np.ndarray,
    shapes: Sequence[BaseGeometry],
    cameras: np.ndarray,
    max_range_m: float,
    whole_object: WholeObject | None,
    image_ids: Sequence[str],
) -> bool:
    """Whether two tracks are wings of one building: asked of a model, or by touch."""
    if whole_object is not None:
        cover = _whole_cover(
            first,
            second,
            first_wall,
            second_wall,
            shapes,
            cameras,
            max_range_m,
            whole_object,
            image_ids,
        )
        if cover is not None:
            return cover >= _WHOLE_COVER
    return _touching(
        first, second, first_wall, second_wall, shapes, cameras, max_range_m
    )


def _frames_in_front(
    first: _Unit,
    second: _Unit,
    first_wall: np.ndarray,
    second_wall: np.ndarray,
    cameras: np.ndarray,
    max_range_m: float,
    limit: int,
) -> list[int]:
    """The frames that outline both tracks, nearest the place where they meet."""
    common = set(first.outlines) & set(second.outlines)
    if not common:
        return []
    a = first_wall[:: max(1, len(first_wall) // 150), :2]
    b = second_wall[:: max(1, len(second_wall) // 150), :2]
    gap, nearest = cKDTree(a).query(b)
    k = int(np.argmin(gap))
    meeting = (a[nearest[k]] + b[k]) / 2.0
    away = {n: float(np.hypot(*(cameras[n] - meeting))) for n in common}
    return [n for n in sorted(common, key=away.__getitem__) if away[n] <= max_range_m][
        :limit
    ]


def _points_inside(shape: BaseGeometry, count: int = 3) -> np.ndarray:
    """A few points well inside a shape, the same ones every time."""
    core = shape.buffer(-0.004)
    target = core if (not core.is_empty and core.area > 0) else shape
    x0, y0, x1, y1 = target.bounds
    found = []
    # a fixed, evenly spread sequence of positions over the bounding box
    for n in range(1, 257):
        x = x0 + (x1 - x0) * ((n * 0.6180339887) % 1.0)
        y = y0 + (y1 - y0) * ((n * 0.7548776662) % 1.0)
        if target.contains(Point(x, y)):
            found.append((x, y))
            if len(found) == count:
                break
    if not found:
        inside = target.representative_point()
        found = [(inside.x, inside.y)]
    while len(found) < count:
        found.append(found[-1])
    return np.array(found)


def _whole_cover(
    first: _Unit,
    second: _Unit,
    first_wall: np.ndarray,
    second_wall: np.ndarray,
    shapes: Sequence[BaseGeometry],
    cameras: np.ndarray,
    max_range_m: float,
    whole_object: WholeObject,
    image_ids: Sequence[str],
) -> float | None:
    """
    How much of one track the whole object found from the other covers.

    In each of the frames in front of the two, the model is asked for the
    object at a few points on one track; the largest candidate it returns
    is the whole object, and the share of the other track's outline inside
    it is taken, the larger of the two directions. The answer is the mean
    over the frames, or ``None`` when the model had nothing to say.
    """
    answers = []
    for number in _frames_in_front(
        first, second, first_wall, second_wall, cameras, max_range_m, _FRAMES_PROMPTED
    ):
        one = unary_union([shapes[i] for i in first.outlines[number]])
        other = unary_union([shapes[i] for i in second.outlines[number]])
        if one.is_empty or other.is_empty or not one.area or not other.area:
            continue
        covered = []
        for asked, target in ((one, other), (other, one)):
            candidates = [
                _shape(rings)
                for rings in whole_object(image_ids[number], _points_inside(asked))
                if len(rings)
            ]
            if candidates:
                whole = max(candidates, key=lambda c: c.area)
                covered.append(whole.intersection(target).area / target.area)
        if covered:
            answers.append(max(covered))
    return float(np.mean(answers)) if answers else None


def _sightings(
    wall: np.ndarray,
    shown: Mapping[int, Sequence[int]],
    shapes: Sequence[BaseGeometry],
    image_ids: Sequence[str],
    shots: Mapping[str, tuple[np.ndarray, np.ndarray]],
    templates: Mapping[str, Observation],
    max_range_m: float,
    max_views: int,
) -> list[Observation]:
    """One sighting per frame that outlines the building: the outline itself."""
    center = wall[:, :2].mean(axis=0)
    found: list[tuple[float, Observation]] = []
    for number, held in shown.items():
        image_id = image_ids[number]
        _, camera = shots[image_id]
        reach = float(np.percentile(np.linalg.norm(wall[:, :3] - camera, axis=1), 10))
        if reach > max_range_m:
            continue
        area = unary_union([shapes[i] for i in held])
        if not isinstance(area, Polygon):
            area = area.convex_hull  # wings outlined apart, or a mask in parts
        if not isinstance(area, Polygon) or area.is_empty:
            continue
        base = templates[image_id]
        offset = center - camera[:2]
        obs = Observation(
            image_id=image_id,
            label=base.label,
            polygon=[(float(x), float(y)) for x, y in area.exterior.coords[:-1]],
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
        obs.bearing = round(math.degrees(math.atan2(offset[0], offset[1])) % 360.0, 2)
        obs.range_m = round(reach, 2)
        found.append((reach, obs))
    return spread_views(found, max_views)
