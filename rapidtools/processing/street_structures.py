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
Buildings and other large structures from a street survey, as polygons.

A building is too large for the per-object logic that places a car: no
single frame shows all of it, Mapillary segments buildings as one surface
rather than one outline per house, and from the street only the front and
part of the sides are ever seen. What a survey does have is its own
structure-from-motion reconstruction. Mapillary publishes, for every image,
the sparse 3-D point cloud its camera pose was solved in (``sfm_cluster``),
and those points lie on exactly the surfaces the survey saw: walls, roofs,
fences, trees and the ground.

The method labels that point cloud with the detection outlines and reads
the buildings off the labeled points:

1. **Vote.** Every reconstructed point is projected into each nearby frame
   and counts a hit when it lands inside an outline of the class. Points
   that most frames place on the class are kept. A shrub in front of a wall
   or the lawn behind a gap is voted out, because the frames that see it
   from another angle disagree.
2. **Group.** Points higher than a low-clutter cut (fences, hedges and
   parked cars are lower) are binned on the ground. Cells with enough
   points are wall and roof; cells closer together than ``min_gap_m`` are
   one structure. Neighboring houses stand apart by a side yard, while
   the garage, the porch and the roof of one house touch.
3. **Outline.** The front of each structure and the sides the survey saw
   are measured. The rear is not: the footprint is a rectangle along the
   facade whose depth is the deepest roof or side-wall point seen, or
   ``assumed_depth_m`` when less than that was visible, and the
   ``depth_source`` attribute says which.
4. **Sightings.** The points of each structure are projected back into the
   frames around it, which gives every building its own outline in every
   image that shows it: instance outlines the per-image segmentation never
   had, ready for cropping and analysis.

Example:
    >>> from rapidtools.processing.street_structures import discover_structures
    >>> structures, counts, unproject = discover_structures(  # doctest: +SKIP
    ...     sightings, clusters={'1012': ('93789', 'https://...')}
    ... )
    >>> structures[0].frontage_m, structures[0].depth_source  # doctest: +SKIP
    (14.5, 'assumed')
"""

from __future__ import annotations

import json
import logging
import math
import zlib
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import requests
from matplotlib.path import Path as OutlinePath
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Polygon

from rapidtools.core import Observation

from .street_localization import local_projection, rotation_matrix

logger = logging.getLogger(__name__)

Project = Callable[[float, float], tuple[float, float]]
Unproject = Callable[[float, float], tuple[float, float]]

#: Occupied ground cells above this percentile of point count are wall.
_DENSITY_QUANTILE = 75.0
#: Mapillary label prefixes of classes segmented as surfaces, not instances.
SURFACE_LABEL_PREFIXES = ('construction--structure--', 'construction--barrier--')


def is_structure_label(label: str) -> bool:
    """
    Whether a Mapillary label names a structure placed from the point cloud.

    Example:
        >>> is_structure_label('construction--structure--building')
        True
        >>> is_structure_label('object--vehicle--car')
        False
    """
    return label.startswith('construction--structure--')


# --------------------------------------------------------------- records
@dataclass
class PointCloud:
    """
    One structure-from-motion cluster of a survey.

    Attributes:
        cluster_id: Mapillary identifier of the cluster.
        lon: Longitude of the cluster's local origin.
        lat: Latitude of the cluster's local origin.
        points: ``(n, 3)`` point coordinates: meters east and north of the
            origin, and altitude.
        rotations: ``(m, 3)`` axis-angle rotation of every shot (world to
            camera), the same vector an image's ``computed_rotation`` holds.
        translations: ``(m, 3)`` translation of every shot.
    """

    cluster_id: str
    lon: float
    lat: float
    points: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray

    def shot_index(self) -> dict[tuple[float, ...], int]:
        """Shots keyed by their rounded rotation vector."""
        return {tuple(np.round(r, 5).tolist()): n for n, r in enumerate(self.rotations)}


@dataclass
class Structure:
    """
    One building (or other structure) found in the labeled point cloud.

    Attributes:
        footprint: Outline on the ground as ``(x, y)`` vertices in the
            local meter frame.
        x: East coordinate of the footprint's center.
        y: North coordinate of the footprint's center.
        members: One sighting per frame that shows the structure, with the
            outline of the structure's points in that frame.
        n_points: Reconstructed points on the structure.
        frontage_m: Length of the facade along the street.
        depth_m: Depth of the footprint away from the street.
        observed_depth_m: Depth actually covered by points (roof and side
            walls the survey saw).
        depth_source: ``'observed'`` when the points reach as deep as the
            footprint, ``'assumed'`` when the rear was filled in.
        height_m: Height of the highest points above the road.
        facade_bearing: Compass direction the facade faces, in degrees.
        days: Survey days with a sighting.
    """

    footprint: list[tuple[float, float]]
    x: float
    y: float
    members: list[Observation]
    n_points: int
    frontage_m: float
    depth_m: float
    observed_depth_m: float
    depth_source: str
    height_m: float
    facade_bearing: float
    days: list[str] = field(default_factory=list)

    @property
    def n_images(self) -> int:
        """Number of distinct frames that show the structure."""
        return len({o.image_id for o in self.members})


# --------------------------------------------------------------- point clouds
def parse_sfm_cluster(payload: bytes, cluster_id: str = '') -> PointCloud:
    """
    Parse a Mapillary ``sfm_cluster`` download (zlib-compressed OpenSfM JSON).

    Example:
        >>> import json, zlib
        >>> raw = json.dumps([{
        ...     'reference_lla': {'longitude': -117.4, 'latitude': 47.7, 'altitude': 0},
        ...     'points': {'1': {'coordinates': [1.0, 2.0, 530.0], 'color': [0, 0, 0]}},
        ...     'shots': {'a': {'rotation': [0, 0, 0], 'translation': [0, 0, 0]}},
        ... }]).encode()
        >>> cloud = parse_sfm_cluster(zlib.compress(raw), '42')
        >>> cloud.points.shape, cloud.rotations.shape, cloud.lat
        ((1, 3), (1, 3), 47.7)
    """
    try:
        text = zlib.decompress(payload)
    except zlib.error:
        text = payload
    data = json.loads(text)
    record = data[0] if isinstance(data, list) else data
    ref = record.get('reference_lla') or {}
    pts = record.get('points') or {}
    shots = list((record.get('shots') or {}).values())
    points = np.array([p['coordinates'] for p in pts.values()], dtype=np.float64)
    return PointCloud(
        cluster_id=str(cluster_id),
        lon=float(ref.get('longitude', 0.0)),
        lat=float(ref.get('latitude', 0.0)),
        points=points.reshape(-1, 3),
        rotations=np.array([s['rotation'] for s in shots], dtype=np.float64).reshape(
            -1, 3
        ),
        translations=np.array(
            [s['translation'] for s in shots], dtype=np.float64
        ).reshape(-1, 3),
    )


def fetch_sfm_cluster(
    cluster_id: str,
    url: str,
    cache_dir: str | Path | None = None,
    timeout: float = 120.0,
) -> PointCloud | None:
    """
    Download (or read from ``cache_dir``) one point-cloud cluster.

    Clusters are a few megabytes and cover about a hundred meters of
    driving. The parsed arrays are cached as ``<cluster_id>.npz``, so a
    second run over the same streets downloads nothing.

    Args:
        cluster_id: Mapillary cluster identifier (cache key).
        url: Signed download address from an image's ``sfm_cluster`` field.
        cache_dir: Folder for the cache; nothing is cached when ``None``.
        timeout: Download timeout in seconds.

    Returns:
        PointCloud | None: The cluster, or ``None`` when it could not be
        downloaded or parsed.

    Example:
        >>> cloud = fetch_sfm_cluster('10937891', url, 'cache')  # doctest: +SKIP
        >>> len(cloud.points)  # doctest: +SKIP
        75208
    """
    path = Path(cache_dir) / f'{cluster_id}.npz' if cache_dir else None
    if path is not None and path.exists():
        try:
            with np.load(path) as z:
                return PointCloud(
                    str(cluster_id),
                    float(z['origin'][0]),
                    float(z['origin'][1]),
                    z['points'],
                    z['rotations'],
                    z['translations'],
                )
        except Exception as exc:  # noqa: BLE001 - a bad cache file is re-downloaded
            logger.debug(f'Unreadable point-cloud cache {path}: {exc}')
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        cloud = parse_sfm_cluster(response.content, cluster_id)
    except Exception as exc:  # noqa: BLE001 - one missing cluster is not fatal
        logger.warning(f'Point cloud {cluster_id} could not be read: {exc}')
        return None
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            origin=np.array([cloud.lon, cloud.lat]),
            points=cloud.points.astype(np.float32),
            rotations=cloud.rotations,
            translations=cloud.translations,
        )
    return cloud


def project_points(
    points: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project world points into a panorama.

    Args:
        points: ``(n, 3)`` world coordinates.
        rotation: ``3 x 3`` world-to-camera rotation matrix.
        translation: Shot translation, so that ``camera = R @ world + t``.

    Returns:
        tuple: Normalized column ``u`` and row ``v`` in ``[0, 1]`` and the
        distance of every point from the camera.

    Example:
        >>> import numpy as np
        >>> point = np.array([[0.0, 0.0, 5.0]])
        >>> u, v, d = project_points(point, np.eye(3), np.zeros(3))
        >>> round(float(u[0]), 2), round(float(v[0]), 2), float(d[0])
        (0.5, 0.5, 5.0)
    """
    cam = points @ rotation.T + translation
    dist = np.linalg.norm(cam, axis=1)
    u = np.arctan2(cam[:, 0], cam[:, 2]) / (2.0 * np.pi) + 0.5
    v = 0.5 - np.arctan2(-cam[:, 1], np.hypot(cam[:, 0], cam[:, 2])) / np.pi
    return u, v, dist


def points_in_outlines(
    u: np.ndarray, v: np.ndarray, outlines: Sequence[np.ndarray]
) -> np.ndarray:
    """
    Which image points fall inside any of a frame's outlines.

    Outlines that straddle the left/right seam of a panorama are unwrapped
    before the test.

    Example:
        >>> import numpy as np
        >>> square = np.array([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]])
        >>> points_in_outlines(np.array([0.5, 0.9]), np.array([0.5, 0.5]), [square])
        array([ True, False])
    """
    inside = np.zeros(len(u), bool)
    for outline in outlines:
        ring = np.asarray(outline, dtype=float)
        uu = u
        if ring[:, 0].max() - ring[:, 0].min() > 0.5:
            ring = ring.copy()
            ring[ring[:, 0] < 0.5, 0] += 1.0
            uu = np.where(u < 0.5, u + 1.0, u)
        x0, y0 = ring.min(axis=0)
        x1, y1 = ring.max(axis=0)
        cand = np.flatnonzero(~inside & (uu >= x0) & (uu <= x1) & (v >= y0) & (v <= y1))
        if len(cand):
            inside[cand] = OutlinePath(ring).contains_points(np.c_[uu[cand], v[cand]])
    return inside


# --------------------------------------------------------------- grouping
def _street_direction(cameras: np.ndarray, tree: cKDTree, at: np.ndarray) -> np.ndarray:
    """Unit vector along the driven path near a point."""
    k = min(6, len(cameras))
    if k < 2:
        return np.array([1.0, 0.0])
    _, near = tree.query(at, k=k)
    local = cameras[np.atleast_1d(near)]
    local = local - local.mean(axis=0)
    if not np.any(local):
        return np.array([1.0, 0.0])
    return np.linalg.svd(local, full_matrices=False)[2][0]


def footprint_rectangle(
    wall_points: np.ndarray,
    depth_points: np.ndarray,
    camera: np.ndarray,
    street: np.ndarray,
    assumed_depth_m: float = 10.0,
    max_depth_m: float = 20.0,
) -> tuple[list[tuple[float, float]], float, float, float, str, float]:
    """
    Rectangle footprint of a structure seen from the street.

    The rectangle runs along the facade (the direction near the street's
    that the walls line up with), starts at the wall nearest the
    camera and reaches back as far as points were seen, or
    ``assumed_depth_m`` when the survey saw less than that.

    Args:
        wall_points: ``(n, 2)`` dense points of the structure.
        depth_points: ``(m, 2)`` further points that may belong to the
            structure (sparse roof points), used to measure the depth seen.
        camera: Position of the nearest camera.
        street: Unit vector along the driven path there.
        assumed_depth_m: Depth used when less was observed.
        max_depth_m: Points farther behind the front than this are taken
            for another structure and do not extend the depth.

    Returns:
        tuple: ``(footprint, frontage_m, depth_m, observed_depth_m,
        depth_source, facade_bearing)``.

    Example:
        >>> import numpy as np
        >>> wall = np.c_[np.linspace(0, 12, 60), np.full(60, 10.0)]
        >>> fp, front, depth, seen, source, facing = footprint_rectangle(
        ...     wall, wall, np.array([6.0, 0.0]), np.array([1.0, 0.0]))
        >>> round(front), round(depth), source, round(facing)
        (12, 10, 'assumed', 180)
    """
    center = wall_points.mean(axis=0)
    centered = wall_points - center
    # The facade direction is the one, within 30 degrees of the street, that
    # the walls line up with: projected onto it and across it, wall points
    # pile up in a few half-meter bins, so the squared bin counts peak.
    base = math.atan2(street[1], street[0])
    best = (-math.inf, base)
    for step in range(-30, 31, 2):
        angle = base + math.radians(step)
        score = 0.0
        for direction in (
            np.array([math.cos(angle), math.sin(angle)]),
            np.array([-math.sin(angle), math.cos(angle)]),
        ):
            bins = np.floor((centered @ direction) / 0.5).astype(int)
            hist = np.bincount(bins - bins.min())
            score += float((hist.astype(float) ** 2).sum())
        # ties and near-ties go to the street direction
        best = max(best, (score * (1.0 - 0.002 * abs(step)), angle))
    axis = np.array([math.cos(best[1]), math.sin(best[1])])
    away = np.array([-axis[1], axis[0]])
    if float((center - camera) @ away) < 0:
        away = -away
    s = centered @ axis
    d = centered @ away
    s0, s1 = np.percentile(s, [1, 99])
    front = float(np.percentile(d, 3))
    observed = float(np.percentile(d, 97) - front)
    deep = (depth_points - center) @ away
    lateral = (depth_points - center) @ axis
    behind = deep[
        (lateral >= s0)
        & (lateral <= s1)
        & (deep >= front)
        & (deep <= front + max_depth_m)
    ]
    if len(behind) >= 5:
        observed = max(observed, float(np.percentile(behind, 97) - front))
    if observed >= 0.7 * assumed_depth_m:
        depth, source = observed, 'observed'
    else:
        depth, source = assumed_depth_m, 'assumed'
    corners = [
        center + s0 * axis + front * away,
        center + s1 * axis + front * away,
        center + s1 * axis + (front + depth) * away,
        center + s0 * axis + (front + depth) * away,
    ]
    facing = math.degrees(math.atan2(-away[0], -away[1])) % 360.0
    return (
        [(float(x), float(y)) for x, y in corners],
        float(s1 - s0),
        float(depth),
        observed,
        source,
        facing,
    )


# --------------------------------------------------------------- pipeline
@dataclass
class LabeledPoints:
    """
    The reconstructed points the outlines place on a class.

    Attributes:
        rows: ``(n, 4)`` array: meters east and north in the local frame,
            altitude, and height above the road at the nearest camera.
        shots: For every usable image ID, its world-to-camera rotation
            matrix and camera center (local east, north, altitude).
        templates: One sighting per image, the source of the camera facts
            copied onto the sightings written for each structure.
    """

    rows: np.ndarray
    shots: dict[str, tuple[np.ndarray, np.ndarray]]
    templates: dict[str, Observation]


def label_points(
    observations: Sequence[Observation],
    clusters: Mapping[str, tuple[str, str]],
    project: Project,
    *,
    cache_dir: str | Path | None = None,
    camera_height_m: float = 2.4,
    max_range_m: float = 40.0,
    min_vote_fraction: float = 0.7,
    min_votes: int = 3,
    max_workers: int = 8,
    loader: Callable[[str, str], PointCloud | None] | None = None,
    counts: dict[str, int] | None = None,
) -> LabeledPoints:
    """
    Vote: keep the reconstructed points most frames place on the class.

    Every point of every cluster is projected into each of the cluster's
    frames within ``max_range_m`` and counts a hit when it lands inside an
    outline. A point is kept with at least ``min_votes`` hits that make up
    ``min_vote_fraction`` of the frames in range.

    Args:
        observations: Sightings of the class.
        clusters: ``(cluster_id, url)`` of the point cloud of every image.
        project: Projection from longitude and latitude to local meters.
        cache_dir: Folder where downloaded point clouds are cached.
        camera_height_m: Camera height above the road.
        max_range_m: Farthest a frame votes on a point.
        min_vote_fraction: Share of frames in range that must agree.
        min_votes: Frames that must agree.
        max_workers: Threads for point-cloud downloads.
        loader: ``loader(cluster_id, url)`` used instead of downloading.
        counts: Counters to add to (clusters, frames, points, ...).

    Returns:
        LabeledPoints: The kept points with the poses of the frames used.

    Example:
        >>> labeled = label_points(sightings, clusters, project)  # doctest: +SKIP
        >>> labeled.rows.shape  # doctest: +SKIP
        (81531, 4)
    """
    counts = counts if counts is not None else defaultdict(int)
    outlines: dict[str, list[np.ndarray]] = defaultdict(list)
    template: dict[str, Observation] = {}
    for obs in observations:
        rotation = (obs.extra or {}).get('computed_rotation')
        if rotation is None or not obs.is_pano or obs.image_id not in clusters:
            counts['unusable'] += 1
            continue
        outlines[obs.image_id].append(np.asarray(obs.polygon, dtype=float))
        template.setdefault(obs.image_id, obs)
    by_cluster: dict[str, tuple[str, list[str]]] = {}
    for image_id in outlines:
        cluster_id, url = clusters[image_id]
        by_cluster.setdefault(str(cluster_id), (url, []))[1].append(image_id)
    counts['clusters'] = len(by_cluster)
    if not by_cluster:
        return LabeledPoints(np.zeros((0, 4)), {}, template)

    load = loader or (lambda cid, url: fetch_sfm_cluster(cid, url, cache_dir))
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        clouds = list(pool.map(lambda kv: load(kv[0], kv[1][0]), by_cluster.items()))

    # 1. Vote: label every reconstructed point with the outlines.
    kept: list[np.ndarray] = []  # x, y, altitude, height above road
    shots: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # image -> (R, center)
    for (_cluster_id, (_, image_ids)), cloud in zip(
        by_cluster.items(), clouds, strict=True
    ):
        if cloud is None or not len(cloud.points) or not len(cloud.rotations):
            counts['clusters_missing'] += 1
            continue
        ox, oy = project(cloud.lon, cloud.lat)
        index = cloud.shot_index()
        seen = np.zeros(len(cloud.points), np.uint16)
        hits = np.zeros(len(cloud.points), np.uint16)
        cams = []
        for image_id in image_ids:
            rot = template[image_id].extra['computed_rotation']
            n = index.get(tuple(np.round(np.asarray(rot, float), 5).tolist()))
            if n is None:
                counts['frames_unmatched'] += 1
                continue
            r = rotation_matrix(cloud.rotations[n])
            t = cloud.translations[n]
            center = -r.T @ t
            shots[image_id] = (r, center + np.array([ox, oy, 0.0]))
            cams.append(center)
            offset = cloud.points - center
            near = np.flatnonzero(
                (np.abs(offset[:, 0]) < max_range_m)
                & (np.abs(offset[:, 1]) < max_range_m)
            )
            if not len(near):
                continue
            u, v, dist = project_points(cloud.points[near], r, t)
            ok = (dist > 2.0) & (dist < max_range_m)
            near, u, v = near[ok], u[ok], v[ok]
            seen[near] += 1
            hits[near] += points_in_outlines(u, v, outlines[image_id])
            counts['frames'] += 1
        if not cams:
            continue
        cams_arr = np.array(cams)
        labeled = np.flatnonzero(
            (hits >= min_votes) & (hits >= min_vote_fraction * np.maximum(seen, 1))
        )
        if not len(labeled):
            continue
        pts = cloud.points[labeled]
        _, nearest = cKDTree(cams_arr[:, :2]).query(pts[:, :2])
        height = pts[:, 2] - (cams_arr[nearest, 2] - camera_height_m)
        rows = np.c_[pts[:, 0] + ox, pts[:, 1] + oy, pts[:, 2], height]
        counts['points'] += len(rows)
        kept.append(rows)
    rows = np.vstack(kept) if kept else np.zeros((0, 4))
    return LabeledPoints(rows, shots, template)


def structures_from_points(
    labeled: LabeledPoints,
    *,
    max_range_m: float = 40.0,
    min_height_m: float = 2.0,
    cell_m: float = 0.5,
    min_cell_points: int = 2,
    min_gap_m: float = 1.0,
    min_points: int = 80,
    min_frontage_m: float = 4.0,
    assumed_depth_m: float = 10.0,
    footprint: str = 'rectangle',
    max_views: int = 12,
    counts: dict[str, int] | None = None,
) -> list[Structure]:
    """
    Group labeled points into structures and outline each one.

    Arguments are those of :func:`discover_structures`.

    Example:
        >>> structures = structures_from_points(labeled)  # doctest: +SKIP
        >>> structures[0].frontage_m  # doctest: +SKIP
        14.5
    """
    counts = counts if counts is not None else defaultdict(int)
    everything = labeled.rows
    shots, template = labeled.shots, labeled.templates
    pts = (
        everything[everything[:, 3] >= min_height_m] if len(everything) else everything
    )
    if not len(pts) or not shots:
        return []
    cameras = np.array([c[:2] for _, c in shots.values()])
    cam_ids = list(shots)
    cam_tree = cKDTree(cameras)

    # 2. Group: dense cells -> wall fragments -> structures.
    origin = pts[:, :2].min(axis=0) - 2.0 * cell_m
    ij = np.floor((pts[:, :2] - origin) / cell_m).astype(int)
    shape = (int(ij[:, 1].max()) + 3, int(ij[:, 0].max()) + 3)
    density = np.zeros(shape, np.int32)
    np.add.at(density, (ij[:, 1], ij[:, 0]), 1)
    # Cells closer than the gap are one structure: grow the dense cells by
    # half the gap, label the grown mask, and read the labels at the points.
    grow = max(1, int(math.ceil(min_gap_m / 2.0 / cell_m)))
    yy, xx = np.ogrid[-grow : grow + 1, -grow : grow + 1]
    disk = xx * xx + yy * yy <= grow * grow
    # Walls collect many points per cell, stray points one or two. How many
    # depends on how often the street was driven, so the cut follows the
    # data: the upper quartile of the occupied cells, at least the minimum.
    occupied = density[density > 0]
    dense = density >= max(
        min_cell_points, int(np.percentile(occupied, _DENSITY_QUANTILE))
    )
    labels, n_labels = ndimage.label(
        ndimage.binary_dilation(dense, structure=disk), structure=np.ones((3, 3))
    )
    labels = np.where(dense, labels, 0)
    point_label = labels[ij[:, 1], ij[:, 0]]
    order = np.argsort(point_label, kind='stable')
    bounds = np.searchsorted(point_label[order], np.arange(1, n_labels + 2))
    fragments = [order[bounds[k] : bounds[k + 1]] for k in range(n_labels)]
    fragments = [m for m in fragments if len(m)]
    counts['fragments'] = len(fragments)
    loose = pts[point_label == 0, :2]  # high points outside every dense cell
    loose_tree = cKDTree(loose) if len(loose) else None

    def outline_of(members: np.ndarray) -> tuple[Any, ...]:
        wall = pts[members]
        center = wall[:, :2].mean(axis=0)
        _, nearest_cam = cam_tree.query(center)
        street = _street_direction(cameras, cam_tree, center)
        radius = float(np.hypot(*(wall[:, :2] - center).T).max())
        around = (
            loose[loose_tree.query_ball_point(center, radius + 20.0)]
            if loose_tree is not None
            else wall[:, :2]
        )
        return (
            wall,
            center,
            radius,
            *footprint_rectangle(
                wall[:, :2], around, cameras[nearest_cam], street, assumed_depth_m
            ),
        )

    # Pieces of one house that the gap left apart (a detached-looking garage
    # front, a roof patch) get rectangles that overlap; join those.
    smallest = max(15, min_points // 4)
    counts['too_few_points'] += sum(len(m) < smallest for m in fragments)
    fragments = [m for m in fragments if len(m) >= smallest]
    boxes = [Polygon(outline_of(m)[3]) for m in fragments]
    parent = list(range(len(fragments)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    if boxes:
        centers = np.array([[b.centroid.x, b.centroid.y] for b in boxes])
        for a, b in sorted(cKDTree(centers).query_pairs(40.0)):
            shared = boxes[a].intersection(boxes[b]).area
            if shared > 0.3 * min(boxes[a].area, boxes[b].area):
                parent[find(a)] = find(b)
    joined: dict[int, list[np.ndarray]] = defaultdict(list)
    for i, members in enumerate(fragments):
        joined[find(i)].append(members)
    counts['joined'] = len(fragments) - len(joined)

    # 3. Outline and 4. sightings.
    structures: list[Structure] = []
    for parts in joined.values():
        members = np.concatenate(parts)
        if len(members) < min_points:
            counts['too_few_points'] += 1
            continue
        wall, center, radius, outline, frontage, depth, observed, source, facing = (
            outline_of(members)
        )
        if frontage < min_frontage_m:
            counts['too_short'] += 1
            continue
        if footprint == 'observed':
            hull = MultiPoint(wall[:, :2]).convex_hull
            if not isinstance(hull, Polygon):
                counts['too_short'] += 1
                continue
            outline = [(float(x), float(y)) for x, y in hull.exterior.coords[:-1]]
            depth, source = observed, 'observed'
        poly = Polygon(outline)
        sightings = _structure_sightings(
            wall,
            shots,
            cam_ids,
            cam_tree,
            template,
            center,
            radius,
            max_range_m,
            max_views,
        )
        if not sightings:
            counts['unseen'] += 1
            continue
        structures.append(
            Structure(
                footprint=outline,
                x=float(poly.centroid.x),
                y=float(poly.centroid.y),
                members=sightings,
                n_points=len(members),
                frontage_m=frontage,
                depth_m=depth,
                observed_depth_m=observed,
                depth_source=source,
                height_m=float(np.percentile(wall[:, 3], 97)),
                facade_bearing=facing,
                days=sorted({o.captured_at[:10] for o in sightings if o.captured_at}),
            )
        )
    return structures


def discover_structures(
    observations: Sequence[Observation],
    clusters: Mapping[str, tuple[str, str]],
    *,
    cache_dir: str | Path | None = None,
    camera_height_m: float = 2.4,
    max_range_m: float = 40.0,
    min_vote_fraction: float = 0.7,
    min_votes: int = 3,
    min_height_m: float = 2.0,
    cell_m: float = 0.5,
    min_cell_points: int = 2,
    min_gap_m: float = 1.0,
    min_points: int = 80,
    min_frontage_m: float = 4.0,
    assumed_depth_m: float = 10.0,
    footprint: str = 'rectangle',
    max_views: int = 12,
    max_workers: int = 8,
    project: Project | None = None,
    unproject: Unproject | None = None,
    loader: Callable[[str, str], PointCloud | None] | None = None,
) -> tuple[list[Structure], dict[str, int], Unproject]:
    """
    Find the structures of one class in a survey's point clouds.

    See the module docstring for the method.

    Args:
        observations: Sightings of the class (panoramas with a
            ``computed_rotation`` in ``extra``).
        clusters: For every image ID, the ``(cluster_id, url)`` of its
            ``sfm_cluster``. Images without an entry are not used.
        cache_dir: Folder where downloaded point clouds are cached.
        camera_height_m: Camera height above the road; sets the ground
            level heights are measured from.
        max_range_m: Farthest a frame votes on a point, and farthest frame
            a structure gets a sighting from.
        min_vote_fraction: Share of the frames within range that must place
            a point on the class.
        min_votes: Frames that must place a point on the class.
        min_height_m: Points lower than this above the road are ignored
            when grouping (shrubs, fences, parked cars).
        cell_m: Ground cell used to find dense wall fragments.
        min_cell_points: Fewest points a cell needs to count as wall. The
            cut used is the upper quartile of the occupied cells when that
            is higher, so it adapts to how densely the survey was driven.
        min_gap_m: Gap along the street that separates two structures.
        min_points: Fewest points a structure may have.
        min_frontage_m: Shortest facade reported.
        assumed_depth_m: Footprint depth where the rear was not seen.
        footprint: ``'rectangle'`` (facade-aligned rectangle, rear filled
            in) or ``'observed'`` (convex outline of the points seen,
            nothing assumed).
        max_views: Sightings kept per structure, spread over the directions
            it was seen from among the closer frames; ``0`` keeps one per
            frame in range.
        max_workers: Threads for point-cloud downloads.
        project: Projection to local meters; made from the sightings when
            omitted.
        unproject: Inverse of ``project``.
        loader: ``loader(cluster_id, url)`` returning a :class:`PointCloud`,
            instead of downloading (for tests and offline use).

    Returns:
        tuple[list[Structure], dict[str, int], Unproject]: The structures,
        counters describing the run and the inverse projection.

    Example:
        >>> structures, counts, unproject = discover_structures(  # doctest: +SKIP
        ...     sightings, clusters, cache_dir='point_clouds'
        ... )
        >>> counts['structures']  # doctest: +SKIP
        212
    """
    if footprint not in ('rectangle', 'observed'):
        raise ValueError(
            f"footprint must be 'rectangle' or 'observed', got {footprint!r}."
        )
    counts: dict[str, int] = defaultdict(int)
    counts['sightings'] = len(observations)
    if project is None or unproject is None:
        lon0 = (
            float(np.mean([o.camera_lon for o in observations]))
            if observations
            else 0.0
        )
        lat0 = (
            float(np.mean([o.camera_lat for o in observations]))
            if observations
            else 0.0
        )
        project, unproject = local_projection(lon0, lat0)

    labeled = label_points(
        observations,
        clusters,
        project,
        cache_dir=cache_dir,
        camera_height_m=camera_height_m,
        max_range_m=max_range_m,
        min_vote_fraction=min_vote_fraction,
        min_votes=min_votes,
        max_workers=max_workers,
        loader=loader,
        counts=counts,
    )
    structures = structures_from_points(
        labeled,
        max_range_m=max_range_m,
        min_height_m=min_height_m,
        cell_m=cell_m,
        min_cell_points=min_cell_points,
        min_gap_m=min_gap_m,
        min_points=min_points,
        min_frontage_m=min_frontage_m,
        assumed_depth_m=assumed_depth_m,
        footprint=footprint,
        max_views=max_views,
        counts=counts,
    )
    for structure in structures:
        lon, lat = unproject(structure.x, structure.y)
        for obs in structure.members:
            obs.lon, obs.lat = lon, lat
    counts['structures'] = len(structures)
    return structures, dict(counts), unproject


def _structure_sightings(
    wall: np.ndarray,
    shots: Mapping[str, tuple[np.ndarray, np.ndarray]],
    cam_ids: Sequence[str],
    cam_tree: cKDTree,
    template: Mapping[str, Observation],
    center: np.ndarray,
    radius: float,
    max_range_m: float,
    max_views: int,
) -> list[Observation]:
    """One sighting per nearby frame: the outline of the structure's points."""
    world = wall[:, :3]
    found: list[tuple[float, Observation]] = []
    for n in cam_tree.query_ball_point(center, max_range_m + radius):
        image_id = cam_ids[n]
        r, cam = shots[image_id]
        u, v, dist = project_points(world - cam, r, np.zeros(3))
        rng = float(np.percentile(dist, 10))
        if rng > max_range_m:
            continue
        if u.max() - u.min() > 0.5:  # across the seam of the panorama
            u = np.where(u < 0.5, u + 1.0, u)
        u0, u1 = np.percentile(u, [1, 99])
        v0, v1 = np.percentile(v, [1, 99])
        if u1 - u0 < 0.01 or u1 - u0 > 0.5:
            continue
        pad_u, pad_v = 0.04 * (u1 - u0), 0.10 * (v1 - v0)
        u0, u1 = u0 - pad_u, u1 + pad_u
        v0, v1 = max(0.0, v0 - pad_v), min(1.0, v1 + 2.0 * pad_v)
        ring = [(u0 % 1.0, v0), (u1 % 1.0, v0), (u1 % 1.0, v1), (u0 % 1.0, v1)]
        base = template[image_id]
        offset = center - cam[:2]
        obs = Observation(
            image_id=image_id,
            label=base.label,
            polygon=ring,
            camera_lon=base.camera_lon,
            camera_lat=base.camera_lat,
            compass_angle=base.compass_angle,
            is_pano=base.is_pano,
            sequence_id=base.sequence_id,
            captured_at=base.captured_at,
            source='sfm',
            image_width=base.image_width,
            image_height=base.image_height,
            extra=dict(base.extra),
        )
        obs.bearing = round(math.degrees(math.atan2(offset[0], offset[1])) % 360.0, 2)
        obs.range_m = round(rng, 2)
        found.append((rng, obs))
    if not found:
        return []
    found.sort(key=lambda item: item[0])
    if max_views <= 0 or len(found) <= max_views:
        return [obs for _, obs in found]
    # Spread the views around the structure instead of taking neighboring
    # frames: among the frames not much farther than the closest, pick
    # evenly along the direction they look from.
    nearest = found[0][0]
    close = [item for item in found if item[0] <= 1.3 * nearest + 5.0]
    if len(close) <= max_views:
        return [obs for _, obs in close]
    reference = float(close[0][1].bearing or 0.0)
    close.sort(
        key=lambda item: (float(item[1].bearing or 0.0) - reference + 180.0) % 360.0
    )
    picks = np.unique(np.linspace(0, len(close) - 1, max_views).round().astype(int))
    return [close[i][1] for i in picks]


def structure_attributes(structure: Structure) -> dict[str, Any]:
    """
    Attribute dictionary describing a structure's footprint and evidence.

    Example:
        >>> s = Structure([(0, 0), (12, 0), (12, 10), (0, 10)], 6.0, 5.0, [], 420,
        ...               12.0, 10.0, 4.2, 'assumed', 5.5, 180.0)
        >>> structure_attributes(s)['depth_source']
        'assumed'
    """
    return {
        'localization': 'point_cloud',
        'n_images': structure.n_images,
        'n_points': structure.n_points,
        'frontage_m': round(structure.frontage_m, 1),
        'depth_m': round(structure.depth_m, 1),
        'observed_depth_m': round(structure.observed_depth_m, 1),
        'depth_source': structure.depth_source,
        'height_m': round(structure.height_m, 1),
        'facade_bearing': round(structure.facade_bearing, 0),
        'footprint_area_m2': round(Polygon(structure.footprint).area, 1),
        'survey_days': list(structure.days),
    }
