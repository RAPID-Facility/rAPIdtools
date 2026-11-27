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
   from another angle disagree. Only a frame that can see a point votes
   against it: where the cloud holds a street tree between the camera and
   a wall, the pixel shows the tree and says nothing about the wall.
2. **Group.** Points higher above the ground than a low-clutter cut (hedges
   and parked cars are lower) are binned on the ground. The ground level is
   read from the cloud, since a lot can lie well below or above the street.
   Cells with enough points are wall and roof, and cells closer together
   than ``min_gap_m`` form a wall piece. One house is often several pieces:
   its garage and its entrance stand at different setbacks, and a garage
   door or a blank wall gives few points. A piece joins the larger piece it
   continues along the street when every frame in front of the two
   outlines them as one building. Neighbors stay apart, because a frame
   that faces the side yard between them outlines each on its own.
3. **Outline.** The front of each structure and the sides the survey saw
   are measured. The rear is not. The footprint follows the facade, with
   a step wherever the wall changes setback, and is closed by one straight
   rear line at the depth the points behind the wall reach, or
   ``assumed_depth_m`` behind the wall set furthest back when less than
   that was visible. The ``depth_source`` attribute says which.
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
#: Wall pieces continue each other along the street when less than this
#: many times ``min_gap_m`` of facade separates them. Measured along the
#: facade, the dense pieces of one house lie closer than its walls do: a
#: garage door or a blank wall leaves a hole in the points, a side yard a
#: wider one.
_JOIN_GAP_FACTOR = 3.0
#: Largest step between the setbacks of two wall pieces of one structure.
_MAX_STEP_M = 6.0
#: Share of the points behind a front wall that the depth seen must hold.
#: The deepest few are as often a tree or the house behind as the roof.
_DEPTH_QUANTILE = 85.0
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
        height_m: Height of the highest points above the ground.
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
        inside |= _in_outline(u, v, outline, skip=inside)
    return inside


def _in_outline(
    u: np.ndarray, v: np.ndarray, outline: np.ndarray, skip: np.ndarray | None = None
) -> np.ndarray:
    """Which image points fall inside one outline (``skip`` ones are not tested)."""
    ring = np.asarray(outline, dtype=float)
    uu = u
    if ring[:, 0].max() - ring[:, 0].min() > 0.5:
        ring = ring.copy()
        ring[ring[:, 0] < 0.5, 0] += 1.0
        uu = np.where(u < 0.5, u + 1.0, u)
    x0, y0 = ring.min(axis=0)
    x1, y1 = ring.max(axis=0)
    box = (uu >= x0) & (uu <= x1) & (v >= y0) & (v <= y1)
    if skip is not None:
        box &= ~skip
    inside = np.zeros(len(u), bool)
    cand = np.flatnonzero(box)
    if len(cand):
        inside[cand] = OutlinePath(ring).contains_points(np.c_[uu[cand], v[cand]])
    return inside


def front_most(
    u: np.ndarray,
    v: np.ndarray,
    dist: np.ndarray,
    cell_deg: float = 1.0,
    tolerance: float = 0.2,
    slack_m: float = 1.0,
) -> np.ndarray:
    """
    Which points of a cloud a panorama can see.

    A reconstructed point projects into every frame near it, whether or not
    something stands in between. The cloud itself says when something does:
    a point is hidden when another point lies in almost the same direction
    and clearly closer to the camera. Directions are compared on a grid of
    ``cell_deg`` cells, each with its eight neighbors.

    Args:
        u: Normalized column of every point, in ``[0, 1]``.
        v: Normalized row of every point, in ``[0, 1]``.
        dist: Distance of every point from the camera.
        cell_deg: Size of a direction cell in degrees; ``0`` skips the
            test and reports every point as seen.
        tolerance: A point counts as seen when it is no farther than the
            nearest point around its direction by this fraction ...
        slack_m: ... plus this many meters, which keeps the points of one
            wall seen at a slant from hiding each other.

    Returns:
        np.ndarray: Boolean mask of the points that are seen.

    Example:
        >>> import numpy as np
        >>> u, v = np.array([0.5, 0.5, 0.75]), np.array([0.5, 0.5, 0.5])
        >>> front_most(u, v, np.array([5.0, 20.0, 20.0])).tolist()  # tree, wall, wall
        [True, False, True]
    """
    if cell_deg <= 0 or not len(dist):
        return np.ones(len(dist), bool)
    nu, nv = int(round(360.0 / cell_deg)), int(round(180.0 / cell_deg))
    col = np.minimum((u * nu).astype(int), nu - 1)
    row = np.clip((v * nv).astype(int), 0, nv - 1)
    nearest = np.full((nv, nu), np.inf)
    np.minimum.at(nearest, (row, col), dist)
    padded = np.pad(nearest, ((1, 1), (0, 0)), constant_values=np.inf)
    padded = np.concatenate([padded[:, -1:], padded, padded[:, :1]], axis=1)  # seam
    around = np.min(
        [padded[a : a + nv, b : b + nu] for a in range(3) for b in range(3)], axis=0
    )
    return np.asarray(dist <= around[row, col] * (1.0 + tolerance) + slack_m)


def ground_level(
    points: np.ndarray,
    at: np.ndarray,
    cell_m: float = 2.0,
    reach_m: float = 4.0,
    min_cell_points: int = 5,
) -> np.ndarray:
    """
    Altitude of the ground under points of a cloud, read from the cloud.

    A reconstruction holds the lawn, the driveway and the foot of every
    wall as well as the walls. The ground under a point is taken from the
    cells of a ``cell_m`` grid within ``reach_m`` of it: the level of a
    cell is the altitude a tenth of its points lie below, and the ground
    is the third lowest of those levels, so that neither a stray point nor
    a clump of them far under the surface is taken for it.

    Args:
        points: ``(n, 3)`` points of the whole cloud.
        at: ``(m, 3)`` or ``(m, 2)`` points to find the ground under.
        cell_m: Ground cell size.
        reach_m: How far around a point ground is looked for.
        min_cell_points: Fewest points a cell needs to count.

    Returns:
        np.ndarray: Ground altitude for every point of ``at``; ``nan``
        where fewer than three cells within reach hold enough points.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> lawn = np.c_[rng.uniform(0, 10, (400, 2)), np.full(400, 95.0)]
        >>> wall = np.c_[np.full(50, 5.0), np.full(50, 5.0), np.linspace(95, 100, 50)]
        >>> under = ground_level(np.vstack([lawn, wall]), np.array([[5.0, 5.0, 99.0]]))
        >>> float(under[0])
        95.0
    """
    if not len(points) or not len(at):
        return np.full(len(at), np.nan)
    origin = points[:, :2].min(axis=0)
    ij = np.floor((points[:, :2] - origin) / cell_m).astype(np.int64)
    shape = (int(ij[:, 1].max()) + 1, int(ij[:, 0].max()) + 1)
    key = ij[:, 1] * shape[1] + ij[:, 0]
    order = np.lexsort((points[:, 2], key))
    cells, start, count = np.unique(key[order], return_index=True, return_counts=True)
    level = np.full(shape[0] * shape[1], np.inf)
    enough = count >= min_cell_points
    level[cells[enough]] = points[order[start[enough] + count[enough] // 10], 2]
    # The third lowest cell around, not the lowest: a reconstruction has a
    # few stray points far under the surface, and they come in clumps.
    reach = max(0, int(round(reach_m / cell_m)))
    size = 2 * reach + 1
    lowest = ndimage.rank_filter(
        level.reshape(shape),
        rank=min(2, size * size - 1),
        size=size,
        mode='constant',
        cval=np.inf,
    )
    where = np.floor((np.asarray(at)[:, :2] - origin) / cell_m).astype(np.int64)
    inside = (
        (where[:, 0] >= 0)
        & (where[:, 0] < shape[1])
        & (where[:, 1] >= 0)
        & (where[:, 1] < shape[0])
    )
    ground = np.full(len(at), np.nan)
    ground[inside] = lowest[where[inside, 1], where[inside, 0]]
    ground[np.isinf(ground)] = np.nan
    return ground


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


def _facade_frame(
    wall_points: np.ndarray, camera: np.ndarray, street: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center of the wall points, the facade direction and the one away from it."""
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
    return center, axis, away


def footprint_rectangle(
    wall_points: np.ndarray,
    depth_points: np.ndarray,
    camera: np.ndarray,
    street: np.ndarray,
    assumed_depth_m: float = 10.0,
    max_depth_m: float = 20.0,
    axis_points: np.ndarray | None = None,
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
        axis_points: ``(k, 2)`` points the facade direction is read from,
            when not all of ``wall_points``: the longest straight wall of
            a structure made of several pieces.

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
    _, axis, away = _facade_frame(
        wall_points if axis_points is None else axis_points, camera, street
    )
    centered = wall_points - center
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
        observed = max(observed, float(np.percentile(behind, _DEPTH_QUANTILE) - front))
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


def footprint_stepped(
    wall_points: np.ndarray,
    depth_points: np.ndarray,
    camera: np.ndarray,
    street: np.ndarray,
    assumed_depth_m: float = 10.0,
    max_depth_m: float = 20.0,
    bin_m: float = 1.0,
    min_step_m: float = 1.5,
    min_run_m: float = 3.0,
    axis_points: np.ndarray | None = None,
) -> tuple[list[tuple[float, float]], float, float, float, str, float]:
    """
    Footprint whose front follows the setbacks of the facade.

    Many houses do not present one flat front: the garage stands forward
    and the entrance is set back, or the reverse. The outline is the
    rectangle of :func:`footprint_rectangle` with its front edge replaced by
    the wall line measured in every ``bin_m`` of facade, cut into straight
    runs wherever the wall steps by more than ``min_step_m``. The rear stays
    one straight line: at the deepest point seen, or ``assumed_depth_m``
    behind the wall set furthest back when the survey saw less than that.

    Args:
        wall_points: ``(n, 2)`` dense points of the structure.
        depth_points: ``(m, 2)`` further points used to measure the depth.
        camera: Position of the nearest camera.
        street: Unit vector along the driven path there.
        assumed_depth_m: Depth used when less was observed.
        max_depth_m: Farthest behind the front a point extends the depth.
        bin_m: Length of facade the wall line is measured over.
        min_step_m: Smallest change of setback drawn as a step.
        min_run_m: Shortest run of facade drawn at its own setback.
        axis_points: ``(k, 2)`` points the facade direction is read from,
            when not all of ``wall_points``.

    Returns:
        tuple: ``(footprint, frontage_m, depth_m, observed_depth_m,
        depth_source, facade_bearing)``, as :func:`footprint_rectangle`.
        ``depth_m`` is measured from the wall nearest the street.

    Example:
        >>> import numpy as np
        >>> garage = np.c_[np.linspace(0, 6, 60), np.full(60, 10.0)]
        >>> house = np.c_[np.linspace(6, 14, 80), np.full(80, 15.0)]
        >>> wall = np.vstack([garage, house])
        >>> fp, front, depth, seen, source, facing = footprint_stepped(
        ...     wall, wall, np.array([7.0, 0.0]), np.array([1.0, 0.0]))
        >>> len(fp), round(front), round(depth), source
        (6, 14, 15, 'assumed')
    """
    outline, frontage, depth, observed, source, facing = footprint_rectangle(
        wall_points,
        depth_points,
        camera,
        street,
        assumed_depth_m,
        max_depth_m,
        axis_points,
    )
    n = int(round(frontage / bin_m))
    shortest = max(1, int(math.ceil(min_run_m / bin_m)))
    if n < 2 * shortest:
        return outline, frontage, depth, observed, source, facing
    origin = np.asarray(outline[0])
    axis = (np.asarray(outline[1]) - origin) / frontage
    away = (np.asarray(outline[3]) - origin) / depth
    along = (wall_points - origin) @ axis
    across = (wall_points - origin) @ away
    which = np.clip((along / frontage * n).astype(int), 0, n - 1)
    fronts = np.full(n, np.nan)
    for k in range(n):
        here = across[which == k]
        if len(here) >= 3:
            fronts[k] = np.percentile(here, 10)
    known = np.flatnonzero(~np.isnan(fronts))
    if len(known) < 2:
        return outline, frontage, depth, observed, source, facing
    fronts = np.interp(np.arange(n), known, fronts[known])
    # Straight runs: a bin starts a new run when its wall stands off the
    # run so far; runs too short to be a wall of their own join the
    # neighbor whose setback is closer.
    runs: list[list[float]] = [[0, 1]]
    for k in range(1, n):
        level = float(np.median(fronts[int(runs[-1][0]) : k]))
        if abs(fronts[k] - level) <= min_step_m:
            runs[-1][1] = k + 1
        else:
            runs.append([k, k + 1])
    levels = [float(np.median(fronts[int(a) : int(b)])) for a, b in runs]
    while len(runs) > 1:
        lengths = [b - a for a, b in runs]
        k = int(np.argmin(lengths))
        gaps = [
            abs(levels[k] - levels[j]) if 0 <= j < len(runs) else math.inf
            for j in (k - 1, k + 1)
        ]
        closest = k - 1 if gaps[0] <= gaps[1] else k + 1
        if lengths[k] >= shortest and min(gaps) > min_step_m:
            break
        lo, hi = min(k, closest), max(k, closest)
        runs[lo] = [runs[lo][0], runs[hi][1]]
        levels[lo] = float(np.median(fronts[int(runs[lo][0]) : int(runs[lo][1])]))
        del runs[hi], levels[hi]
    if len(runs) < 2:
        return outline, frontage, depth, observed, source, facing
    # The rear is one straight line, placed like the rectangle's but behind
    # the wall set furthest back: where the deepest points were seen, or
    # the assumed depth behind that wall when less was visible.
    nearest, deepest = min(levels), max(levels)
    if observed - deepest >= 0.7 * assumed_depth_m:
        back, source = observed, 'observed'
    else:
        back, source = deepest + assumed_depth_m, 'assumed'
    ring: list[np.ndarray] = []
    for (a, b), level in zip(runs, levels, strict=True):
        ring.append(origin + (a * frontage / n) * axis + level * away)
        ring.append(origin + (b * frontage / n) * axis + level * away)
    ring.append(origin + frontage * axis + back * away)
    ring.append(origin + back * away)
    return (
        [(float(x), float(y)) for x, y in ring],
        frontage,
        float(back - nearest),
        observed,
        source,
        facing,
    )


def drawn_as_one(
    first: np.ndarray,
    second: np.ndarray,
    shots: Mapping[str, tuple[np.ndarray, np.ndarray]],
    outlines: Mapping[str, Sequence[np.ndarray]],
    image_ids: Sequence[str],
    camera_tree: cKDTree,
    n_frames: int = 6,
    max_range_m: float = 40.0,
) -> bool | None:
    """
    Whether the frames in front of two wall pieces outline them as one building.

    A detection outline follows a building as far as it is unbroken in the
    image. The garage and the entrance of one house are inside one outline
    in every frame that faces them; two neighbors are inside one outline
    only from far up the street, where one hides the yard between them, and
    in two outlines from in front of that yard. So the frames nearest the
    place where the pieces meet are asked, and one that shows each piece in
    an outline of its own settles it.

    Args:
        first: ``(n, 3)`` points of one piece (local east, north, altitude).
        second: ``(m, 3)`` points of the other.
        shots: Rotation matrix and camera center of every image.
        outlines: Outlines of the class in every image.
        image_ids: Image of every camera in ``camera_tree``.
        camera_tree: Tree over the camera positions (east, north).
        n_frames: Frames asked, nearest first.
        max_range_m: Farthest camera asked.

    Returns:
        bool | None: ``True`` when every frame that shows both pieces
        inside outlines shows them in the same one, ``False`` when a frame
        shows them in two, ``None`` when no frame shows both.

    Example:
        >>> verdict = drawn_as_one(garage, entrance, labeled.shots,  # doctest: +SKIP
        ...                        labeled.outlines, image_ids, camera_tree)
        >>> verdict  # doctest: +SKIP
        True
    """
    if not len(first) or not len(second) or not len(image_ids):
        return None
    a = first[:: max(1, len(first) // 150), :3]
    b = second[:: max(1, len(second) // 150), :3]
    gap, nearest = cKDTree(a[:, :2]).query(b[:, :2])
    k = int(np.argmin(gap))
    meeting = (a[nearest[k], :2] + b[k, :2]) / 2.0
    dist, cams = camera_tree.query(meeting, k=min(n_frames, len(image_ids)))
    both = np.vstack([a, b])
    verdict: bool | None = None
    for d, n in zip(np.atleast_1d(dist), np.atleast_1d(cams), strict=True):
        image_id = image_ids[int(n)]
        if d > max_range_m or not outlines.get(image_id):
            continue
        rotation, camera = shots[image_id]
        u, v, _ = project_points(both - camera, rotation, np.zeros(3))
        share = np.array([_in_outline(u, v, ring) for ring in outlines[image_id]])
        in_a, in_b = share[:, : len(a)].sum(axis=1), share[:, len(a) :].sum(axis=1)
        # a piece mostly hidden or outside every outline says nothing here
        if in_a.max() < 0.3 * len(a) or in_b.max() < 0.3 * len(b):
            continue
        if int(np.argmax(in_a)) != int(np.argmax(in_b)):
            return False
        verdict = True
    return verdict


def _wall_offsets(
    outline: Sequence[tuple[float, float]],
    observed_depth_m: float,
    points: np.ndarray,
) -> tuple[float, float]:
    """
    How far wall points stand from the wall piece a rectangle was drawn around.

    Measured along and across the piece's own facade.

    Returns:
        tuple[float, float]: The length of facade that separates the two
        along the street (``0`` when they overlap or touch), and how far
        the points stand in front of or behind the depth the piece was seen
        to (``0`` when they are within it).

    Example:
        >>> import numpy as np
        >>> garage = [(0.0, 0.0), (6.0, 0.0), (6.0, 10.0), (0.0, 10.0)]
        >>> house_front = np.c_[np.linspace(6.5, 14.0, 40), np.full(40, 5.0)]
        >>> gap, step = _wall_offsets(garage, 1.0, house_front)
        >>> round(gap, 1), round(step, 1)
        (0.6, 4.0)
    """
    origin = np.asarray(outline[0], dtype=float)
    axis = np.asarray(outline[1], dtype=float) - origin
    frontage = float(np.hypot(*axis))
    away = np.asarray(outline[3], dtype=float) - origin
    depth = float(np.hypot(*away))
    if frontage <= 0.0 or depth <= 0.0 or not len(points):
        return math.inf, math.inf
    axis, away = axis / frontage, away / depth
    along = (points - origin) @ axis
    across = (points - origin) @ away
    s0, s1 = np.percentile(along, [1, 99])
    d0, d1 = np.percentile(across, [3, 97])
    gap = max(0.0, float(s0) - frontage, -float(s1))
    step = max(0.0, float(d0) - max(observed_depth_m, 0.0), -float(d1))
    return gap, step


# --------------------------------------------------------------- pipeline
@dataclass
class LabeledPoints:
    """
    The reconstructed points the outlines place on a class.

    Attributes:
        rows: ``(n, 4)`` array: meters east and north in the local frame,
            altitude, and height above the ground (read from the cloud;
            above the road at the nearest camera where it shows none).
        shots: For every usable image ID, its world-to-camera rotation
            matrix and camera center (local east, north, altitude).
        templates: One sighting per image, the source of the camera facts
            copied onto the sightings written for each structure.
        outlines: The outlines of the class in every image, as arrays of
            normalized ``(u, v)`` vertices. They tell the pieces of one
            building from two neighbors when structures are grouped.
    """

    rows: np.ndarray
    shots: dict[str, tuple[np.ndarray, np.ndarray]]
    templates: dict[str, Observation]
    outlines: dict[str, list[np.ndarray]] = field(default_factory=dict)


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
        return LabeledPoints(np.zeros((0, 4)), {}, template, dict(outlines))

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
            near, u, v, dist = near[ok], u[ok], v[ok], dist[ok]
            inside = points_in_outlines(u, v, outlines[image_id])
            # A frame votes against a point only when it can see it: behind
            # a street tree, the pixel a wall point lands on shows the tree.
            seen[near[inside | front_most(u, v, dist)]] += 1
            hits[near] += inside
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
        # Heights are above the ground a point stands on, which the cloud
        # itself shows; a lot can lie well below or above the street. Where
        # the cloud has no ground nearby, the road under the nearest camera
        # stands in for it.
        ground = ground_level(cloud.points, pts)
        road = cams_arr[nearest, 2] - camera_height_m
        height = pts[:, 2] - np.where(np.isnan(ground), road, ground)
        rows = np.c_[pts[:, 0] + ox, pts[:, 1] + oy, pts[:, 2], height]
        counts['points'] += len(rows)
        kept.append(rows)
    rows = np.vstack(kept) if kept else np.zeros((0, 4))
    return LabeledPoints(rows, shots, template, dict(outlines))


def structures_from_points(
    labeled: LabeledPoints,
    *,
    max_range_m: float = 40.0,
    min_height_m: float = 2.0,
    cell_m: float = 0.5,
    min_cell_points: int = 2,
    min_gap_m: float = 1.0,
    min_points: int = 40,
    min_frontage_m: float = 4.0,
    assumed_depth_m: float = 10.0,
    footprint: str = 'stepped',
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

    def outline_of(
        members: np.ndarray, stepped: bool = False, straight: np.ndarray | None = None
    ) -> tuple[Any, ...]:
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
            *(footprint_stepped if stepped else footprint_rectangle)(
                wall[:, :2],
                around,
                cameras[nearest_cam],
                street,
                assumed_depth_m,
                axis_points=None if straight is None else pts[straight, :2],
            ),
        )

    # Pieces of one house that the gap left apart. A garage front and the
    # house front behind it are dense walls at two setbacks, joined only by
    # a side wall seen too obliquely to be dense, and a garage door or a
    # blank wall gives few points at all; a corner lot shows a front and a
    # side to two streets. Such pieces continue each other along the street
    # and every frame in front of them outlines them as one building.
    # Neighboring houses stand apart by a side yard, and a frame facing the
    # yard outlines them separately. A piece whose rectangle overlaps a
    # larger one (a roof patch over its own wall) belongs to it as well.
    # Every piece joins at most one larger piece, the one it is tied to
    # most closely, so that a shed in a side yard or a roof seen over a
    # fence cannot tie the houses on either side of it together.
    smallest = max(15, min_points // 4)
    counts['too_few_points'] += sum(len(m) < smallest for m in fragments)
    fragments = [m for m in fragments if len(m) >= smallest]
    drawn = [outline_of(m) for m in fragments]
    boxes = [Polygon(d[3]) for d in drawn]
    parent = list(range(len(fragments)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    if boxes:
        centers = np.array([[b.centroid.x, b.centroid.y] for b in boxes])
        # for every piece: (how it is tied, how closely, the larger piece)
        best: dict[int, tuple[int, float, float, int]] = {}
        for a, b in sorted(cKDTree(centers).query_pairs(40.0)):
            # measured on the larger piece: its wall direction is the better one
            big, small = (a, b) if len(fragments[a]) >= len(fragments[b]) else (b, a)
            shared = boxes[a].intersection(boxes[b]).area
            shared /= min(boxes[a].area, boxes[b].area)
            if shared > 0.3:
                tie = (0, -shared, 0.0, big)
            else:
                gap, step = _wall_offsets(
                    drawn[big][3], drawn[big][6], drawn[small][0][:, :2]
                )
                if gap >= _JOIN_GAP_FACTOR * min_gap_m or step >= _MAX_STEP_M:
                    continue
                one = drawn_as_one(
                    drawn[big][0],
                    drawn[small][0],
                    shots,
                    labeled.outlines,
                    cam_ids,
                    cam_tree,
                    max_range_m=max_range_m,
                )
                # without a frame that shows both, only pieces that touch join
                if not (one or (one is None and gap < min_gap_m)):
                    continue
                tie = (1, gap, step, big)
            best[small] = min(best.get(small, tie), tie)
        for small, (_, _, _, big) in best.items():
            parent[find(small)] = find(big)
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
            outline_of(
                members,
                stepped=footprint == 'stepped',
                # the facade direction is read from the largest piece
                straight=max(parts, key=len),
            )
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
    min_points: int = 40,
    min_frontage_m: float = 4.0,
    assumed_depth_m: float = 10.0,
    footprint: str = 'stepped',
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
            level heights are measured from where the point cloud has no
            ground points of its own.
        max_range_m: Farthest a frame votes on a point, and farthest frame
            a structure gets a sighting from.
        min_vote_fraction: Share of the frames within range that must place
            a point on the class.
        min_votes: Frames that must place a point on the class.
        min_height_m: Points lower than this above the ground are ignored
            when grouping (shrubs, fences, parked cars).
        cell_m: Ground cell used to find dense wall fragments.
        min_cell_points: Fewest points a cell needs to count as wall. The
            cut used is the upper quartile of the occupied cells when that
            is higher, so it adapts to how densely the survey was driven.
        min_gap_m: Free ground that separates two wall pieces. Pieces up to
            three times as far apart along the street are still joined when
            the frames in front of them outline them as one building.
        min_points: Fewest points a structure may have.
        min_frontage_m: Shortest facade reported.
        assumed_depth_m: Footprint depth where the rear was not seen.
        footprint: ``'stepped'`` (the front follows the setbacks of the
            facade, rear filled in), ``'rectangle'`` (one facade-aligned
            rectangle, rear filled in) or ``'observed'`` (convex outline of
            the points seen, nothing assumed).
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
    if footprint not in ('stepped', 'rectangle', 'observed'):
        raise ValueError(
            "footprint must be 'stepped', 'rectangle' or 'observed', "
            f'got {footprint!r}.'
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
    return spread_views(found, max_views)


def spread_views(
    found: list[tuple[float, Observation]], max_views: int
) -> list[Observation]:
    """
    Pick the sightings of a structure to keep, from its ``(range, sighting)`` pairs.

    The views are spread around the structure instead of taken from
    neighboring frames: among the frames not much farther than the closest,
    they are picked evenly along the direction they look from. ``max_views``
    of ``0`` keeps every sighting, nearest first.

    Example:
        >>> spread_views([], 3)
        []
    """
    if not found:
        return []
    found = sorted(found, key=lambda item: item[0])
    if max_views <= 0 or len(found) <= max_views:
        return [obs for _, obs in found]
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
