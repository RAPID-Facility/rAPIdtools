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
From sightings to objects by occupancy fusion.

Every detection outline in a street-level frame is read as a small range
scan, the way a robot reads a lidar sweep. Each image column the outline
covers gives a bearing; where the outline's lower edge lies below the
horizon it also gives the range at which the object meets the ground. That
tells three things about the ground along the column: it is *free* up to
the object, *occupied* for a stretch behind the contact, and *unknown*
beyond. Columns of a frame without any outline are free as far as the
detector can be trusted to see.

All frames of a survey day are fused in one ground grid. A cell belongs to
an object when the frames that vote on it mostly agree that it is occupied;
objects are the connected blobs of such cells. Nothing is tracked from
frame to frame and no pair of sightings is ever compared, so there are no
association rules to tune per class:

- An object seen from several streets, or on the way in and on the way
  out, lands on the same cells and is one object.
- A moving vehicle never collects agreeing votes, and the cells the survey
  vehicle drove through are free by construction.
- A far sighting has a long, uncertain band and a low weight; it cannot
  create an object by itself and cannot move one placed from close up.
- The same code handles a hydrant and a bus: only the cell size and the
  range change.

Where frames keep seeing several outlines on one blob (two cars parked door
to door), the blob is split along the outlines of the closest frame that
separates them. Each object is then placed by intersecting the centre
bearings of its sightings, and its footprint is the part of the blob that
lies inside the outlines of most of them (a plan-view visual hull).

All geometry runs in a local metre frame from
:func:`~rapidtools.processing.street_localization.local_projection`.

Example:
    >>> from rapidtools.processing.street_fusion import fuse_observations
    >>> objects, counts, unproject = fuse_observations(sightings)  # doctest: +SKIP
    >>> objects[0].n_images, round(objects[0].area_m2)  # doctest: +SKIP
    (41, 9)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Point, Polygon

from rapidtools.core import Observation

from .street_localization import local_projection, pixel_bearing, rotation_matrix

logger = logging.getLogger(__name__)

#: Width of one bearing column of a scan, in degrees.
BIN_DEG = 0.5
#: Number of bearing columns around the horizon.
N_BINS = int(round(360.0 / BIN_DEG))
#: Outlines whose lower edge is closer to the horizon than this give no range.
MIN_DEPRESSION_DEG = 1.0
#: Range at which a vote counts half as much as one from next to the object.
WEIGHT_RANGE_M = 10.0
#: Share of a blob's peak occupied weight a cell needs to count as footprint.
_CORE_FRACTION = 0.6
#: Bearings must spread at least this much for an intersection to be used.
MIN_PARALLAX_DEG = 10.0

Project = Callable[[float, float], tuple[float, float]]
Unproject = Callable[[float, float], tuple[float, float]]


# --------------------------------------------------------------- records
@dataclass
class Scan:
    """
    One detection outline as a range scan.

    Attributes:
        index: Position of the sighting in the list it came from.
        cols: Bearing columns the outline covers (``N_BINS`` per turn).
        bottom: Lowest elevation of the outline in each column, degrees.
        top: Highest elevation of the outline in each column, degrees.
        centre: Bearing of the outline's centre, degrees clockwise from north.
        width: Angular width of the outline, degrees.
        lo: Near end of the occupied band per column in metres; ``nan``
            where the column gives no range.
        hi: Far end of the occupied band per column.
        contact: Ground-contact range of the outline (its nearer quartile),
            or ``nan`` when no column reaches below the horizon.
    """

    index: int
    cols: np.ndarray
    bottom: np.ndarray
    top: np.ndarray
    centre: float
    width: float
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    contact: float = math.nan


@dataclass
class FusedObject:
    """
    One object found by occupancy fusion.

    Attributes:
        x: East coordinate of the object in the local metre frame.
        y: North coordinate.
        cov: 2 x 2 position covariance in square metres, or ``None`` when
            the position is the blob centre rather than an intersection.
        members: The sightings that show the object.
        footprint: Outline of the object on the ground, as ``(x, y)``
            vertices in the local metre frame.
        area_m2: Area of the footprint.
        length_m: Long side of the footprint's bounding rectangle.
        width_m: Short side of the footprint's bounding rectangle.
        parallax_deg: Largest spread among the sightings' bearings.
        days: Survey days the object was seen on.
        localization: ``'triangulated'`` (bearings intersected) or
            ``'fused'`` (centre of the occupied cells).
        rms_m: Root-mean-square distance between the position and the
            sightings' bearing lines, when triangulated.
    """

    x: float
    y: float
    cov: np.ndarray | None
    members: list[Observation]
    footprint: list[tuple[float, float]]
    area_m2: float
    length_m: float
    width_m: float
    parallax_deg: float
    days: list[str] = field(default_factory=list)
    localization: str = 'triangulated'
    rms_m: float | None = None

    @property
    def n_images(self) -> int:
        """Number of distinct frames that show the object."""
        return len({o.image_id for o in self.members})

    @property
    def sigma_m(self) -> float:
        """One-sigma position uncertainty along the worst direction."""
        if self.cov is None:
            return round(max(0.5, 0.25 * self.length_m), 2)
        return float(math.sqrt(max(float(np.linalg.eigvalsh(self.cov).max()), 0.0)))


@dataclass
class _Frame:
    """A camera position with the scans made from it."""

    image_id: str
    x: float
    y: float
    scans: list[Scan] = field(default_factory=list)
    fov: tuple[float, float] | None = None  # (centre, width) of a perspective frame


class _Grid:
    """Evidence accumulated over a rectangle of ground cells."""

    def __init__(self, x0: float, y0: float, cell: float, nx: int, ny: int) -> None:
        self.x0, self.y0, self.cell, self.nx, self.ny = x0, y0, cell, nx, ny
        self.occ = np.zeros((ny, nx), np.float32)
        self.free = np.zeros((ny, nx), np.float32)
        self.views = np.zeros((ny, nx), np.uint16)

    def xy(self, flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Centre coordinates of cells given by their flat indices."""
        jj, ii = np.divmod(flat, self.nx)
        return self.x0 + (ii + 0.5) * self.cell, self.y0 + (jj + 0.5) * self.cell


# --------------------------------------------------------------- scans
def outline_angles(obs: Observation) -> tuple[np.ndarray, np.ndarray]:
    """
    World bearing and elevation of every vertex of a sighting's outline.

    Uses the frame's full pose (``computed_rotation``) and camera model when
    the sighting carries them, and a level camera at ``compass_angle``
    otherwise.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Bearings in degrees clockwise
        from north (not wrapped consistently; see :func:`scan_from_observation`)
        and elevations in degrees, negative below the horizon.

    Example:
        >>> from rapidtools.core import Observation
        >>> obs = Observation('i', 'car', [(0.5, 0.5), (0.75, 0.5), (0.75, 0.75)],
        ...                   0.0, 0.0, 90.0, is_pano=True)
        >>> bearing, elevation = outline_angles(obs)
        >>> [round(float(b)) for b in bearing], [round(float(e)) for e in elevation]
        ([90, 180, 180], [0, 0, -45])
    """
    pts = np.asarray(obs.polygon, dtype=float)
    extra = obs.extra or {}
    kind = (
        extra.get('camera_type') or ('spherical' if obs.is_pano else 'perspective')
    ).lower()
    if kind in ('spherical', 'equirectangular'):
        w, h = float(obs.image_width or 2.0), float(obs.image_height or 1.0)
        lon = 2.0 * np.pi * (pts[:, 0] - 0.5)
        lat = -2.0 * np.pi * (pts[:, 1] - 0.5) * h / w
        rays = np.c_[np.cos(lat) * np.sin(lon), -np.sin(lat), np.cos(lat) * np.cos(lon)]
    else:
        params = extra.get('camera_parameters')
        rays = np.array(
            [
                pixel_bearing(x, y, kind, obs.image_width, obs.image_height, params)
                for x, y in pts
            ]
        )
    rotation = extra.get('computed_rotation')
    if rotation is not None:
        world = rays @ rotation_matrix(rotation)  # row-wise R.T @ ray
    else:
        c = math.radians(float(obs.compass_angle or 0.0))
        world = np.c_[
            math.sin(c) * rays[:, 2] + math.cos(c) * rays[:, 0],
            math.cos(c) * rays[:, 2] - math.sin(c) * rays[:, 0],
            -rays[:, 1],
        ]
    bearing = np.degrees(np.arctan2(world[:, 0], world[:, 1]))
    elevation = np.degrees(np.arctan2(world[:, 2], np.hypot(world[:, 0], world[:, 1])))
    return bearing, elevation


def scan_from_observation(index: int, obs: Observation) -> Scan | None:
    """
    Read a sighting's outline as a scan: its lower and upper edge per column.

    Args:
        index: Position of the sighting in its list, kept on the scan.
        obs: The sighting.

    Returns:
        Scan | None: The scan without range bands (see :func:`set_range_bands`),
        or ``None`` for an outline that wraps the whole horizon.

    Example:
        >>> from rapidtools.core import Observation
        >>> obs = Observation('i', 'car', [(0.50, 0.55), (0.52, 0.55), (0.52, 0.60),
        ...                               (0.50, 0.60)], 0.0, 0.0, 0.0, is_pano=True)
        >>> scan = scan_from_observation(0, obs)
        >>> round(scan.width, 1), round(float(scan.bottom.min()), 1)
        (7.2, -18.0)
    """
    bearing, elevation = outline_angles(obs)
    ring = np.degrees(np.unwrap(np.radians(np.r_[bearing, bearing[0]])))
    elev = np.r_[elevation, elevation[0]]
    if abs(ring[-1] - ring[0]) > 180.0:
        return None  # the outline circles the camera (ground, sky, the car itself)
    steps = np.maximum(1, np.ceil(np.abs(np.diff(ring)) / (BIN_DEG / 2.0)).astype(int))
    total = int(steps.sum())
    if total > 40 * N_BINS:
        return None
    start = np.cumsum(steps) - steps
    pos = np.repeat(np.arange(len(steps)), steps) + (
        np.arange(total) - np.repeat(start, steps)
    ) / np.repeat(steps, steps)
    a = np.interp(pos, np.arange(len(ring)), ring)
    e = np.interp(pos, np.arange(len(elev)), elev)
    b = np.floor(a / BIN_DEG).astype(int)
    b0 = int(b.min())
    n = int(b.max()) - b0 + 1
    if n > N_BINS:
        return None
    bottom = np.full(n, np.inf)
    top = np.full(n, -np.inf)
    np.minimum.at(bottom, b - b0, e)
    np.maximum.at(top, b - b0, e)
    seen = np.isfinite(bottom)
    cols = (np.arange(n) + b0)[seen] % N_BINS
    width = float(a.max() - a.min())
    centre = float(((a.max() + a.min()) / 2.0) % 360.0)
    return Scan(index, cols, bottom[seen], top[seen], centre, width)


def set_range_bands(
    scan: Scan,
    camera_height_m: float = 2.4,
    min_range_m: float = 2.0,
    max_range_m: float = 30.0,
    elevation_sigma_deg: float = 0.75,
    range_sigma_fraction: float = 0.2,
    depth_factor: float = 0.5,
) -> Scan:
    """
    Turn a scan's lower edge into an occupied range band per column.

    A column whose lower edge lies below the horizon meets flat ground at
    ``camera_height / tan(depression)``. The band starts two standard
    deviations in front of that contact and ends the same distance behind
    it plus the object's assumed depth. The range error has a part that
    grows with the square of the range (``elevation_sigma_deg`` of pixel and
    pose error) and a part proportional to it (``range_sigma_fraction``:
    sloping ground, a hidden lower edge), so a close sighting pins the object
    down and a far one says little more than its bearing.

    Runs of columns whose contact range changes smoothly are one visible
    face. The object is assumed to reach behind a face by ``depth_factor``
    times the face's length, and never beyond the range at which a ray over
    the outline's top meets the ground (an object lower than the camera
    cannot hide ground behind its own shadow). Columns without a ground
    contact abstain: they claim nothing and clear nothing.

    Args:
        scan: Scan from :func:`scan_from_observation`; modified in place.
        camera_height_m: Camera height above the ground.
        min_range_m: Closest range a band may start at.
        max_range_m: Contacts beyond 1.5 times this are ignored.
        elevation_sigma_deg: One-sigma error of an outline edge's elevation.
        range_sigma_fraction: One-sigma range error as a share of the range.
        depth_factor: Assumed depth of an object behind a visible face, as a
            share of the face's length.

    Returns:
        Scan: The same scan with ``lo``, ``hi`` and ``contact`` set.

    Example:
        >>> import numpy as np
        >>> scan = Scan(0, np.arange(4), np.full(4, -16.7), np.full(4, -5.0), 1.0, 2.0)
        >>> scan = set_range_bands(scan)
        >>> round(scan.contact, 1), bool(scan.lo[0] < 8.0 < scan.hi[0])
        (8.0, True)
    """
    n = len(scan.cols)
    with np.errstate(divide='ignore', invalid='ignore'):
        near = np.where(
            scan.bottom < -MIN_DEPRESSION_DEG,
            camera_height_m / np.tan(np.radians(-scan.bottom)),
            np.nan,
        )
        over = np.where(
            scan.top < -MIN_DEPRESSION_DEG,
            camera_height_m / np.tan(np.radians(-scan.top)),
            np.inf,
        )
    near[near > 1.5 * max_range_m] = np.nan
    seen = np.isfinite(near)
    scan.lo = np.full(n, np.nan)
    scan.hi = np.full(n, np.nan)
    scan.contact = float(np.percentile(near[seen], 25)) if seen.any() else math.nan
    if not seen.any():
        return scan
    d = math.radians(elevation_sigma_deg)
    sigma = np.hypot(near * near * d / camera_height_m, range_sigma_fraction * near)
    with np.errstate(invalid='ignore'):
        jump = np.abs(np.diff(near))
        smooth = jump <= np.maximum(1.5, 0.15 * np.minimum(near[1:], near[:-1]))
    face = np.concatenate([[0], np.cumsum(~smooth)])
    angle = np.radians((np.arange(n) + 0.5) * BIN_DEG)
    step = np.hypot(np.diff(near * np.sin(angle)), np.diff(near * np.cos(angle)))
    step = np.where(smooth, np.nan_to_num(step), 0.0)
    n_faces = int(face[-1]) + 1
    length = np.bincount(face[1:], weights=step, minlength=n_faces)
    breadth = np.bincount(
        face,
        weights=np.where(seen, near * math.radians(BIN_DEG), 0.0),
        minlength=n_faces,
    )
    depth = np.maximum(1.0, depth_factor * np.maximum(length, breadth)[face])
    with np.errstate(invalid='ignore'):
        sigma_over = np.hypot(
            over * over * d / camera_height_m, range_sigma_fraction * over
        )
        lo = np.maximum(min_range_m, near - 2.0 * sigma - 0.3)
        hi = np.minimum(near + 2.0 * sigma + depth, over + 2.0 * sigma_over)
        hi = np.maximum(hi, lo + 0.5)
    scan.lo = np.where(seen, lo, np.nan)
    scan.hi = np.where(seen, hi, np.nan)
    return scan


# --------------------------------------------------------------- fusion
def _fuse_frame(
    grid: _Grid,
    frame: _Frame,
    min_range_m: float,
    max_range_m: float,
    layers: int = 3,
) -> None:
    """Add one frame's occupied and free votes to the grid."""
    lo = np.full((layers, N_BINS), np.inf, np.float32)
    hi = np.zeros((layers, N_BINS), np.float32)
    count = np.zeros(N_BINS, int)
    blind = np.zeros(N_BINS, bool)
    ordered = sorted(
        frame.scans, key=lambda s: s.contact if math.isfinite(s.contact) else math.inf
    )
    for scan in ordered:
        assert scan.lo is not None and scan.hi is not None  # range_bands
        seen = np.isfinite(scan.lo)
        blind[scan.cols[~seen]] = True
        layer = count[scan.cols]
        ok = seen & (layer < layers)
        lo[layer[ok], scan.cols[ok]] = scan.lo[ok]
        hi[layer[ok], scan.cols[ok]] = scan.hi[ok]
        count[scan.cols[ok]] += 1
    has = count > 0
    # Free up to the nearest object in the column, or as far as the frame
    # sees when the column shows none. Columns outside a perspective frame,
    # or covered only by an outline without a ground contact, say nothing.
    free_to = np.where(has, lo.min(axis=0), max_range_m).astype(np.float32)
    free_to[blind & ~has] = 0.0
    if frame.fov is not None:
        centre, width = frame.fov
        b = (np.arange(N_BINS) + 0.5) * BIN_DEG
        outside = np.abs((b - centre + 180.0) % 360.0 - 180.0) > width / 2.0
        free_to[outside & ~has] = 0.0
    c = grid.cell
    i0 = max(0, int((frame.x - max_range_m - grid.x0) / c))
    i1 = min(grid.nx, int((frame.x + max_range_m - grid.x0) / c) + 1)
    j0 = max(0, int((frame.y - max_range_m - grid.y0) / c))
    j1 = min(grid.ny, int((frame.y + max_range_m - grid.y0) / c) + 1)
    if i0 >= i1 or j0 >= j1:
        return
    dx = grid.x0 + (np.arange(i0, i1) + 0.5) * c - frame.x
    dy = grid.y0 + (np.arange(j0, j1) + 0.5) * c - frame.y
    gx, gy = np.meshgrid(dx, dy)
    r = np.hypot(gx, gy).astype(np.float32)
    b = np.floor(np.degrees(np.arctan2(gx, gy)) / BIN_DEG).astype(int) % N_BINS
    in_range = (r >= min_range_m) & (r <= max_range_m)
    weight = (1.0 / (1.0 + (r / WEIGHT_RANGE_M) ** 2)).astype(np.float32)
    occupied = np.zeros(r.shape, bool)
    for k in range(layers):
        occupied |= (r >= lo[k][b]) & (r <= hi[k][b])
    occupied &= in_range
    free = (r < free_to[b]) & in_range & ~occupied
    window = (slice(j0, j1), slice(i0, i1))
    grid.occ[window] += weight * occupied
    grid.free[window] += weight * free
    # The survey vehicle itself stood here: nothing static can.
    grid.free[window] += 5.0 * (r < min_range_m)
    grid.views[window] += occupied


def _band_cells(grid: _Grid, frame: _Frame, scan: Scan) -> np.ndarray:
    """Flat indices of the grid cells inside a scan's occupied bands."""
    assert scan.lo is not None and scan.hi is not None  # range_bands
    seen = np.isfinite(scan.lo)
    if not seen.any():
        return np.zeros(0, int)
    lo, hi, cols = scan.lo[seen], scan.hi[seen], scan.cols[seen]
    n = max(2, int(math.ceil(float(np.max(hi - lo)) / (0.7 * grid.cell))) + 1)
    r = lo[:, None] + (hi - lo)[:, None] * np.linspace(0.0, 1.0, n)[None, :]
    found = []
    for offset in (0.25, 0.75):  # two rays per column so no cell is stepped over
        a = np.radians((cols[:, None] + offset) * BIN_DEG)
        i = np.floor((frame.x + r * np.sin(a) - grid.x0) / grid.cell).astype(int)
        j = np.floor((frame.y + r * np.cos(a) - grid.y0) / grid.cell).astype(int)
        ok = (i >= 0) & (i < grid.nx) & (j >= 0) & (j < grid.ny)
        found.append(j[ok] * grid.nx + i[ok])
    return np.unique(np.concatenate(found))


def _split_instances(
    grid: _Grid,
    frames: dict[str, _Frame],
    cells: np.ndarray,
    members: list[tuple[str, Scan, np.ndarray]],
) -> list[tuple[np.ndarray, list[tuple[str, Scan, np.ndarray]]]]:
    """
    Split a blob where frames keep seeing several outlines on it.

    The number of objects is the weighted median, over the frames that see
    the blob, of how many outlines each puts on it (close frames count
    most). The closest frame showing exactly that many seeds the parts;
    every other such frame adds its outlines to the parts they overlap most,
    one outline per part, and each cell goes to the part with most votes.
    """
    xs, ys = grid.xy(cells)
    cx, cy = float(xs.mean()), float(ys.mean())
    per_frame: dict[str, list[tuple[str, Scan, np.ndarray]]] = defaultdict(list)
    for member in members:
        per_frame[member[0]].append(member)
    ranges = {
        iid: math.hypot(frames[iid].x - cx, frames[iid].y - cy) for iid in per_frame
    }
    weights = {iid: 1.0 / (1.0 + (ranges[iid] / WEIGHT_RANGE_M) ** 2) for iid in ranges}
    counts = np.array([len(v) for v in per_frame.values()])
    w = np.array([weights[iid] for iid in per_frame])
    order = np.argsort(counts)
    cum = np.cumsum(w[order])
    k = int(counts[order][np.searchsorted(cum, 0.5 * cum[-1])])
    if k < 2:
        return [(cells, members)]
    position = {int(c): n for n, c in enumerate(cells)}
    votes = np.zeros((k, len(cells)), np.float32)
    exact = sorted(
        (iid for iid, v in per_frame.items() if len(v) == k),
        key=lambda iid: ranges[iid],
    )
    for n_frame, iid in enumerate(exact):
        sets = [
            np.fromiter((position[int(c)] for c in m[2]), int, len(m[2]))
            for m in per_frame[iid]
        ]
        if n_frame == 0:
            labels = list(range(k))
        else:
            affinity = np.array(
                [[votes[p][s].sum() / max(1, len(s)) for p in range(k)] for s in sets]
            )
            labels = [-1] * k
            for _ in range(k):
                a, b = np.unravel_index(int(np.argmax(affinity)), affinity.shape)
                labels[a] = int(b)
                affinity[a, :] = -1.0
                affinity[:, b] = -1.0
        owners = np.zeros(len(cells), int)
        for s in sets:
            owners[s] += 1
        for s, label in zip(sets, labels, strict=True):
            votes[label][s[owners[s] == 1]] += weights[iid]
    part = votes.argmax(axis=0)
    unlabelled = votes.max(axis=0) <= 0
    if unlabelled.all():
        return [(cells, members)]
    if unlabelled.any():
        pts = np.c_[xs, ys]
        _, nearest = cKDTree(pts[~unlabelled]).query(pts[unlabelled])
        part[unlabelled] = part[~unlabelled][nearest]
    assigned: dict[int, list[tuple[str, Scan, np.ndarray]]] = defaultdict(list)
    for iid, scan, claimed in members:
        s = np.fromiter((position[int(c)] for c in claimed), int, len(claimed))
        inside = np.bincount(part[s], minlength=k)
        label = int(inside.argmax())
        assigned[label].append((iid, scan, claimed[part[s] == label]))
    out = []
    for label in range(k):
        sel = cells[part == label]
        if len(sel) and assigned[label]:
            out.append((sel, assigned[label]))
    return out or [(cells, members)]


def _intersect(
    cams: np.ndarray, bearings_deg: np.ndarray, start: np.ndarray, sigma_deg: float
) -> tuple[np.ndarray, np.ndarray | None, float, float | None]:
    """
    Robust least-squares intersection of bearings; falls back to ``start``.

    Returns the position, its covariance (``None`` without enough parallax),
    the parallax in degrees and the root-mean-square distance in metres
    between the position and the bearing lines.
    """
    az = np.radians(bearings_deg)
    spread = az[:, None] - az[None, :]
    parallax = (
        float(np.degrees(np.abs(np.arctan2(np.sin(spread), np.cos(spread))).max()))
        if len(az) > 1
        else 0.0
    )
    if parallax < MIN_PARALLAX_DEG:
        return start, None, parallax, None
    normal = np.c_[np.cos(az), -np.sin(az)]
    offset = (normal * cams).sum(axis=1)
    p = np.asarray(start, float)
    w = np.ones(len(az))
    a = np.zeros((2, 2))
    for _ in range(8):
        sigma = np.maximum(1.0, np.hypot(*(p - cams).T)) * math.radians(sigma_deg)
        ww = w / sigma**2
        a = (normal * ww[:, None]).T @ normal
        try:
            q = np.linalg.solve(a, (normal * ww[:, None]).T @ offset)
        except np.linalg.LinAlgError:
            return start, None, parallax, None
        residual = np.abs(((q - cams) * normal).sum(axis=1)) / sigma
        w = np.where(residual < 2.5, 1.0, 2.5 / np.maximum(residual, 1e-9))
        p = q
    try:
        cov = np.linalg.inv(a)
    except np.linalg.LinAlgError:
        return start, None, parallax, None
    rms = float(np.sqrt(np.mean(((p - cams) * normal).sum(axis=1) ** 2)))
    return p, cov, parallax, rms


def _hull_cells(
    grid: _Grid,
    frames: dict[str, _Frame],
    cells: np.ndarray,
    members: list[tuple[str, Scan, np.ndarray]],
    agreement: float,
) -> np.ndarray:
    """Cells of a blob that lie inside the outlines of most of its sightings."""
    xs, ys = grid.xy(cells)
    inside = np.zeros(len(cells), np.float32)
    total = 0.0
    for iid, scan, _ in members:
        frame = frames[iid]
        dx, dy = xs - frame.x, ys - frame.y
        b = np.floor(np.degrees(np.arctan2(dx, dy)) / BIN_DEG).astype(int) % N_BINS
        covered = np.zeros(N_BINS, bool)
        covered[scan.cols] = True
        covered[(scan.cols + 1) % N_BINS] = True
        covered[(scan.cols - 1) % N_BINS] = True
        weight = 1.0 / (1.0 + (float(np.hypot(dx, dy).mean()) / WEIGHT_RANGE_M) ** 2)
        inside += weight * covered[b]
        total += weight
    keep = inside >= agreement * total
    if keep.sum() < max(4, 0.2 * len(cells)):
        return cells
    # Range bands are generous in depth, so the blob reaches behind and in
    # front of the object where fewer frames vote. The object itself is where
    # nearly every frame's band overlaps: the cells with most of the peak
    # occupied weight.
    weight = grid.occ.ravel()[cells]
    core = keep & (weight >= _CORE_FRACTION * np.percentile(weight[keep], 95))
    if core.sum() >= max(4, 0.1 * len(cells)):
        keep = core
    return cells[keep]


def _footprint(
    grid: _Grid, cells: np.ndarray
) -> tuple[list[tuple[float, float]], float, float, float]:
    """Convex outline of a set of cells with its area and side lengths."""
    xs, ys = grid.xy(cells)
    half = grid.cell / 2.0
    corners = np.concatenate(
        [np.c_[xs + sx * half, ys + sy * half] for sx in (-1, 1) for sy in (-1, 1)]
    )
    hull = MultiPoint(corners).convex_hull
    if not isinstance(hull, Polygon):
        hull = hull.buffer(half)
    rect = np.asarray(hull.minimum_rotated_rectangle.exterior.coords)[:4]
    a = float(np.hypot(*(rect[1] - rect[0])))
    b = float(np.hypot(*(rect[2] - rect[1])))
    outline = [(float(x), float(y)) for x, y in hull.exterior.coords[:-1]]
    return outline, float(hull.area), max(a, b), min(a, b)


def _extract(
    grid: _Grid,
    frames: dict[str, _Frame],
    observations: Sequence[Observation],
    core: tuple[float, float, float, float] | None,
    *,
    max_range_m: float,
    occupancy_threshold: float,
    min_views: int,
    min_weight: float,
    min_object_width_m: float,
    split_instances: bool,
    hull_agreement: float,
    bearing_sigma_deg: float,
    min_parallax_deg: float,
    counts: dict[str, int],
    day: str | None,
) -> list[FusedObject]:
    """Blobs of occupied cells, split into instances and described."""
    total = grid.occ + grid.free
    with np.errstate(invalid='ignore', divide='ignore'):
        score = np.where(total > 0, grid.occ / total, 0.0)
    mask = (
        (score >= occupancy_threshold)
        & (grid.views >= min_views)
        & (grid.occ >= min_weight)
    )
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if not n:
        return []
    counts['blobs'] += n
    flat = labels.ravel()
    slices = ndimage.find_objects(labels)
    ids = list(frames)
    tree = cKDTree(np.array([[frames[i].x, frames[i].y] for i in ids]))
    cells_of: dict[tuple[str, int], np.ndarray] = {}
    claims: dict[tuple[str, int], tuple[int, int]] = {}  # scan -> (label, cells)
    for label, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        j0, j1, i0, i1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        bx = grid.x0 + (i0 + i1) / 2.0 * grid.cell
        by = grid.y0 + (j0 + j1) / 2.0 * grid.cell
        radius = 0.5 * math.hypot(i1 - i0, j1 - j0) * grid.cell
        if core is not None and not (
            core[0] - radius <= bx < core[2] + radius
            and core[1] - radius <= by < core[3] + radius
        ):
            continue
        for n_frame in tree.query_ball_point([bx, by], max_range_m + radius):
            frame = frames[ids[n_frame]]
            dist = math.hypot(bx - frame.x, by - frame.y)
            bearing = math.degrees(math.atan2(bx - frame.x, by - frame.y))
            half = (
                math.degrees(math.asin(min(1.0, radius / dist)))
                if dist > radius
                else 180.0
            )
            for n_scan, scan in enumerate(frame.scans):
                if not math.isfinite(scan.contact):
                    continue
                off = abs((scan.centre - bearing + 180.0) % 360.0 - 180.0)
                if off > scan.width / 2.0 + half + 1.0:
                    continue
                key = (frame.image_id, n_scan)
                cells = cells_of.get(key)
                if cells is None:
                    cells = cells_of[key] = _band_cells(grid, frame, scan)
                inside = int(np.count_nonzero(flat[cells] == label))
                if inside >= 3 and inside > claims.get(key, (0, 0))[1]:
                    claims[key] = (label, inside)
    members_of: dict[int, list[tuple[str, Scan, np.ndarray]]] = defaultdict(list)
    for (iid, n_scan), (label, _) in claims.items():
        cells = cells_of[(iid, n_scan)]
        members_of[label].append(
            (iid, frames[iid].scans[n_scan], cells[flat[cells] == label])
        )
    objects: list[FusedObject] = []
    for label, members in members_of.items():
        cells = np.flatnonzero(flat == label)
        parts = (
            _split_instances(grid, frames, cells, members)
            if split_instances
            else [(cells, members)]
        )
        counts['split'] += len(parts) - 1
        for part_cells, part_members in parts:
            if len({iid for iid, _, _ in part_members}) < min_views:
                counts['few_views'] += 1
                continue
            hull = _hull_cells(grid, frames, part_cells, part_members, hull_agreement)
            outline, area, length, width = _footprint(grid, hull)
            if length < min_object_width_m:
                counts['fragment'] += 1
                continue
            xs, ys = grid.xy(hull)
            w = grid.occ.ravel()[hull]
            centre = np.array([np.average(xs, weights=w), np.average(ys, weights=w)])
            if core is not None and not (
                core[0] <= centre[0] < core[2] and core[1] <= centre[1] < core[3]
            ):
                continue
            cams = np.array([[frames[i].x, frames[i].y] for i, _, _ in part_members])
            bearings = np.array([s.centre for _, s, _ in part_members])
            p, cov, parallax, rms = _intersect(
                cams, bearings, centre, bearing_sigma_deg
            )
            if parallax < min_parallax_deg:
                # Seen along one line only (a vehicle driving ahead of the
                # survey car where a sequence ends): nothing fixes its range.
                counts['no_parallax'] += 1
                continue
            if cov is not None and math.hypot(*(p - centre)) > max(length, 2.0):
                p, cov, rms = centre, None, None  # bearings disagree with the blob
            sightings = []
            for (_, scan, _), cam in zip(part_members, cams, strict=True):
                obs = observations[scan.index]
                obs.bearing = round(scan.centre, 2)
                obs.elevation = round(float(scan.bottom.min()), 2)
                obs.range_m = round(float(math.hypot(*(p - cam))), 2)
                sightings.append(obs)
            objects.append(
                FusedObject(
                    x=float(p[0]),
                    y=float(p[1]),
                    cov=cov,
                    members=sightings,
                    footprint=outline,
                    area_m2=area,
                    length_m=length,
                    width_m=width,
                    parallax_deg=parallax,
                    days=[day] if day else [],
                    localization='triangulated' if cov is not None else 'fused',
                    rms_m=rms,
                )
            )
    return objects


def _merge_days(objects: list[FusedObject]) -> list[FusedObject]:
    """Join objects of different survey days that stand on the same spot."""
    if len(objects) < 2:
        return objects
    polys = [Polygon(o.footprint) for o in objects]
    reach = max(o.length_m for o in objects)
    tree = cKDTree(np.array([[o.x, o.y] for o in objects]))
    parent = list(range(len(objects)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in sorted(tree.query_pairs(reach)):
        if set(objects[a].days) & set(objects[b].days):
            continue  # seen apart on the same day: two objects
        if polys[a].contains(Point(objects[b].x, objects[b].y)) or polys[b].contains(
            Point(objects[a].x, objects[a].y)
        ):
            ra, rb = find(a), find(b)
            if ra != rb and not (
                _group_days(objects, parent, ra) & _group_days(objects, parent, rb)
            ):
                parent[rb] = ra
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(objects)):
        groups[find(i)].append(i)
    merged = []
    for members in groups.values():
        best = min(members, key=lambda i: (objects[i].cov is None, objects[i].sigma_m))
        keep = objects[best]
        for i in members:
            if i != best:
                keep.members.extend(objects[i].members)
                keep.days = sorted(set(keep.days) | set(objects[i].days))
        merged.append(keep)
    return merged


def _group_days(objects: list[FusedObject], parent: list[int], root: int) -> set[str]:
    """Survey days of every object currently joined under ``root``."""
    days: set[str] = set()
    for i, obj in enumerate(objects):
        j = i
        while parent[j] != j:
            j = parent[j]
        if j == root:
            days |= set(obj.days)
    return days


def fuse_observations(
    observations: Sequence[Observation],
    frames: Sequence[Any] | None = None,
    *,
    camera_height_m: float = 2.4,
    min_range_m: float = 2.0,
    max_range_m: float = 30.0,
    cell_m: float = 0.25,
    occupancy_threshold: float = 0.6,
    min_views: int = 2,
    min_object_width_m: float = 1.0,
    split_instances: bool = True,
    elevation_sigma_deg: float = 0.75,
    range_sigma_fraction: float = 0.2,
    depth_factor: float = 0.5,
    hull_agreement: float = 0.75,
    bearing_sigma_deg: float = 0.75,
    min_parallax_deg: float = MIN_PARALLAX_DEG,
    tile_m: float = 300.0,
    project: Project | None = None,
    unproject: Unproject | None = None,
) -> tuple[list[FusedObject], dict[str, int], Unproject]:
    """
    Turn the sightings of one class into objects by occupancy fusion.

    See the module docstring for the method. Each survey day is fused by
    itself, so a car that left between two passes is not voted away by the
    later one; objects of different days that stand on the same spot are
    joined afterwards.

    Args:
        observations: Sightings of one class. ``bearing``, ``elevation`` and
            ``range_m`` are filled in on those that end up in an object.
        frames: Every frame of the survey (objects with ``image_id``,
            ``lon``, ``lat``, ``compass_angle``, ``is_pano``, ``captured_at``
            and ``horizontal_fov_deg``, such as
            :class:`~rapidtools.processing.street_tracking.CameraFrame`),
            including those without a sighting: they are the ones that say
            a stretch of kerb is empty. The sightings' own frames are used
            when omitted.
        camera_height_m: Camera height above the ground.
        min_range_m: Closest range anything is placed at.
        max_range_m: Farthest range a frame votes at.
        cell_m: Size of a ground cell. A quarter of a metre suits vehicles;
            use ``0.1`` for hydrants and poles.
        occupancy_threshold: Share of the (range-weighted) votes on a cell
            that must say "occupied".
        min_views: Frames that must agree on a cell, and that an object
            must have been seen in.
        min_object_width_m: Objects whose footprint is shorter than this
            are dropped.
        split_instances: Split a blob that frames keep seeing as several
            outlines. Turn off for classes Mapillary segments as one
            surface (buildings, fences, vegetation).
        elevation_sigma_deg: One-sigma error of an outline edge's elevation.
        range_sigma_fraction: One-sigma range error as a share of the range.
        depth_factor: Assumed depth of an object behind a visible face, as a
            share of the face's length.
        hull_agreement: Share of an object's sightings whose outline must
            cover a cell for it to count as footprint.
        bearing_sigma_deg: One-sigma bearing error, for the position
            uncertainty.
        min_parallax_deg: Objects whose sightings all look along one line
            (bearings spread less than this) are dropped; ``0`` keeps them
            at the centre of their cells.
        tile_m: Side of the ground tiles the survey is processed in; bounds
            memory, not accuracy.
        project: Projection to local metres; one centred on the sightings is
            made when omitted.
        unproject: Inverse of ``project``.

    Returns:
        tuple[list[FusedObject], dict[str, int], Unproject]: The objects,
        counters describing the run, and the inverse projection that turns
        their coordinates back into longitude and latitude.

    Example:
        >>> objects, counts, unproject = fuse_observations(sightings)  # doctest: +SKIP
        >>> unproject(objects[0].x, objects[0].y)  # doctest: +SKIP
        (-117.4885, 47.7165)
    """
    counts: dict[str, int] = defaultdict(int)
    counts['sightings'] = len(observations)
    if project is None or unproject is None:
        if observations:
            lon0 = float(np.mean([o.camera_lon for o in observations]))
            lat0 = float(np.mean([o.camera_lat for o in observations]))
        else:
            lon0 = lat0 = 0.0
        project, unproject = local_projection(lon0, lat0)
    if not observations:
        counts['objects'] = 0
        return [], dict(counts), unproject

    days: dict[str | None, dict[str, _Frame]] = defaultdict(dict)
    for index, obs in enumerate(observations):
        scan = scan_from_observation(index, obs)
        if scan is None:
            counts['unusable'] += 1
            continue
        set_range_bands(
            scan,
            camera_height_m,
            min_range_m,
            max_range_m,
            elevation_sigma_deg,
            range_sigma_fraction,
            depth_factor,
        )
        day = obs.captured_at[:10] if obs.captured_at else None
        frame = days[day].get(obs.image_id)
        if frame is None:
            x, y = project(obs.camera_lon, obs.camera_lat)
            fov = None
            if not obs.is_pano:
                params = (obs.extra or {}).get('camera_parameters')
                focal = float(params[0]) if params and params[0] else 0.85
                fov = (
                    float(obs.compass_angle or 0.0),
                    2.0 * math.degrees(math.atan(0.5 / focal)),
                )
            frame = days[day][obs.image_id] = _Frame(obs.image_id, x, y, fov=fov)
        frame.scans.append(scan)
        counts['scans'] += 1
    for cam in frames or ():
        captured = getattr(cam, 'captured_at', None)
        day = captured[:10] if captured else None
        if day not in days or cam.image_id in days[day]:
            continue
        x, y = project(cam.lon, cam.lat)
        fov = None
        if not getattr(cam, 'is_pano', True):
            fov = (float(cam.compass_angle or 0.0), float(cam.horizontal_fov_deg))
        days[day][cam.image_id] = _Frame(cam.image_id, x, y, fov=fov)

    objects: list[FusedObject] = []
    pad = max_range_m + 10.0
    for day, day_frames in sorted(days.items(), key=lambda kv: kv[0] or ''):
        counts['frames'] += len(day_frames)
        xy = np.array([[f.x, f.y] for f in day_frames.values()])
        ids = list(day_frames)
        tree = cKDTree(xy)
        x_min, y_min = xy.min(axis=0)
        x_max, y_max = xy.max(axis=0)
        nx = max(1, int(math.ceil((x_max - x_min + 1e-6) / tile_m)))
        ny = max(1, int(math.ceil((y_max - y_min + 1e-6) / tile_m)))
        for ty in range(ny):
            for tx in range(nx):
                core = (
                    x_min + tx * tile_m,
                    y_min + ty * tile_m,
                    x_min + (tx + 1) * tile_m,
                    y_min + (ty + 1) * tile_m,
                )
                # Objects stand up to max_range from a camera, so the edge
                # tiles own everything beyond the cameras on their side.
                core = (
                    -math.inf if tx == 0 else core[0],
                    -math.inf if ty == 0 else core[1],
                    math.inf if tx == nx - 1 else core[2],
                    math.inf if ty == ny - 1 else core[3],
                )
                gx0 = x_min + tx * tile_m - pad
                gy0 = y_min + ty * tile_m - pad
                span = tile_m + 2.0 * pad
                centre = [gx0 + span / 2.0, gy0 + span / 2.0]
                near = tree.query_ball_point(
                    centre, span / math.sqrt(2.0) + max_range_m
                )
                tile_frames = {
                    ids[i]: day_frames[ids[i]]
                    for i in near
                    if gx0 - max_range_m <= xy[i, 0] <= gx0 + span + max_range_m
                    and gy0 - max_range_m <= xy[i, 1] <= gy0 + span + max_range_m
                }
                if not any(f.scans for f in tile_frames.values()):
                    continue
                cells = int(math.ceil(span / cell_m))
                grid = _Grid(gx0, gy0, cell_m, cells, cells)
                for frame in tile_frames.values():
                    _fuse_frame(grid, frame, min_range_m, max_range_m)
                objects.extend(
                    _extract(
                        grid,
                        tile_frames,
                        observations,
                        core,
                        max_range_m=max_range_m,
                        occupancy_threshold=occupancy_threshold,
                        min_views=min_views,
                        min_weight=0.6,
                        min_object_width_m=min_object_width_m,
                        split_instances=split_instances,
                        hull_agreement=hull_agreement,
                        bearing_sigma_deg=bearing_sigma_deg,
                        min_parallax_deg=min_parallax_deg,
                        counts=counts,
                        day=day,
                    )
                )
    before = len(objects)
    objects = _merge_days(objects)
    for obj in objects:
        lon, lat = unproject(obj.x, obj.y)
        for obs in obj.members:
            obs.lon, obs.lat = lon, lat
    counts['merged_days'] = before - len(objects)
    counts['objects'] = len(objects)
    return objects, dict(counts), unproject


def object_attributes(obj: FusedObject) -> dict[str, Any]:
    """
    Attribute dictionary describing a fused object's quality and size.

    Example:
        >>> obj = FusedObject(0.0, 0.0, None, [], [(0, 0), (4, 0), (4, 2), (0, 2)],
        ...                   8.0, 4.0, 2.0, 35.0)
        >>> object_attributes(obj)['footprint_area_m2']
        8.0
    """
    out: dict[str, Any] = {
        'localization': obj.localization,
        'n_images': obj.n_images,
        'position_sigma_m': round(obj.sigma_m, 2),
        'parallax_deg': round(obj.parallax_deg, 1),
        'footprint_area_m2': round(obj.area_m2, 1),
        'length_m': round(obj.length_m, 1),
        'width_m': round(obj.width_m, 1),
        'survey_days': list(obj.days),
    }
    if obj.rms_m is not None:
        out['position_rms_m'] = round(obj.rms_m, 2)
    return out
