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
Camera poses of drone photographs.

A drone photo is useful for asset-level analysis only once it is known where
the camera was and where it pointed. This module reads that pose from the
metadata the aircraft writes (EXIF GPS tags plus the DJI ``drone-dji`` XMP
block with gimbal angles and the height above the take-off point), or from a
sidecar table for other vendors, and wraps it in :class:`DroneCamera`, a
pinhole camera that projects world points into the photo and back.

Conventions:

- ``yaw_deg`` is the compass direction the lens looks at (clockwise from
  north), ``pitch_deg`` is ``0`` at the horizon and ``-90`` straight down
  (the DJI gimbal convention), ``roll_deg`` is positive clockwise as seen
  from behind the camera.
- ``elevation_m`` is the camera height in the same vertical datum as the
  ground elevations the caller works with. By default it is the height above
  the take-off point, which makes the take-off point the zero of a flat
  world; pass ``takeoff_elevation_m`` to place the flight in a real datum.
- Local coordinates are east/north/up metres from
  :func:`~rapidtools.processing.street_localization.local_projection`.

Example:
    >>> from rapidtools.data_sources import DroneCamera
    >>> camera = DroneCamera(
    ...     image_id='DJI_0001', lon=-118.14, lat=34.186, elevation_m=80.0,
    ...     yaw_deg=0.0, pitch_deg=-90.0, width=4000, height=3000,
    ...     focal_px=3000.0,
    ... )
    >>> pixels, depth = camera.project_lonlat([(-118.14, 34.186, 0.0)])
    >>> [round(float(v)) for v in pixels[0]], round(float(depth[0]))
    ([2000, 1500], 80)
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
import struct
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

logger = logging.getLogger(__name__)

#: File extensions treated as photographs when a folder is scanned.
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp')
#: Diagonal of a 35 mm film frame, used with ``FocalLengthIn35mmFilm``.
FULL_FRAME_DIAGONAL_MM = 43.2666
#: Horizontal field of view assumed when a photo carries no lens data.
DEFAULT_HFOV_DEG = 73.7
#: Bytes scanned at the start of a file for the XMP packet.
XMP_SCAN_BYTES = 1 << 20

#: Same sphere as ``street_localization.local_projection``.
_EARTH_RADIUS_M = 6_371_000.0
_XMP_HEADER = b'http://ns.adobe.com/xap/1.0/\x00'
_XMP_TAG = re.compile(
    r'drone-dji:([A-Za-z0-9]+)\s*(?:=\s*"([^"]*)"|>([^<]*)<)', re.DOTALL
)
_GPS_IFD = 0x8825
_EXIF_IFD = 0x8769


def _float(value: Any) -> float | None:
    """Best-effort float: handles ``'+12.50'``, rationals and blanks."""
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode('ascii', 'ignore')
    if isinstance(value, str):
        value = value.strip().lstrip('+')
        if not value:
            return None
        if '/' in value:
            num, _, den = value.partition('/')
            try:
                return float(num) / float(den)
            except (ValueError, ZeroDivisionError):
                return None
    try:
        result = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) else None


def rotation_from_angles(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0):
    """
    World-to-camera rotation for a gimbal pose.

    Rows are the camera axes (x to the right, y down, z along the lens) in
    east/north/up coordinates.

    Args:
        yaw_deg: Compass direction of the lens, clockwise from north.
        pitch_deg: ``0`` at the horizon, ``-90`` straight down.
        roll_deg: Clockwise roll as seen from behind the camera.

    Returns:
        numpy.ndarray: 3 x 3 rotation matrix.

    Example:
        >>> rotation = rotation_from_angles(90.0, 0.0)
        >>> [round(float(v), 6) + 0.0 for v in rotation[2]]  # lens looks east
        [1.0, 0.0, 0.0]
        >>> [round(float(v), 6) + 0.0 for v in rotation[1]]  # image y points down
        [0.0, 0.0, -1.0]
    """
    yaw, pitch, roll = (math.radians(v) for v in (yaw_deg, pitch_deg, roll_deg))
    forward = np.array(
        [
            math.sin(yaw) * math.cos(pitch),
            math.cos(yaw) * math.cos(pitch),
            math.sin(pitch),
        ]
    )
    right = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
    down = np.cross(forward, right)
    if roll:
        cos_r, sin_r = math.cos(roll), math.sin(roll)
        right, down = cos_r * right + sin_r * down, cos_r * down - sin_r * right
    return np.stack([right, down, forward])


@dataclass
class DroneCamera:
    """
    One drone photograph: where the camera was, where it looked, and its lens.

    Attributes:
        image_id: Identifier of the photo (the file stem by default).
        lon: Camera longitude in degrees (WGS 84).
        lat: Camera latitude in degrees.
        elevation_m: Camera height in the working vertical datum (see the
            module notes).
        yaw_deg: Compass direction the lens looks at.
        pitch_deg: ``0`` at the horizon, ``-90`` straight down.
        roll_deg: Roll about the lens axis.
        width: Image width in pixels.
        height: Image height in pixels.
        focal_px: Focal length in pixels.
        cx: Principal point column; the image centre when ``None``.
        cy: Principal point row; the image centre when ``None``.
        path: Image file, when the photo is on disk.
        captured_at: Capture time as written by the camera, if known.
        relative_altitude_m: Height above the take-off point, if recorded.
        absolute_altitude_m: GPS altitude, if recorded.
        properties: Anything else worth carrying along.

    Example:
        >>> camera = DroneCamera(
        ...     image_id='a', lon=0.0, lat=0.0, elevation_m=50.0, yaw_deg=0.0,
        ...     pitch_deg=-45.0, width=4000, height=3000, focal_px=3000.0,
        ... )
        >>> round(camera.hfov_deg, 1)
        67.4
        >>> ground = camera.pixel_to_ground(2000, 1500)  # 50 m ahead, due north
        >>> round(ground[1] * 111320)
        50
    """

    image_id: str
    lon: float
    lat: float
    elevation_m: float
    yaw_deg: float
    pitch_deg: float
    width: int
    height: int
    focal_px: float
    roll_deg: float = 0.0
    cx: float | None = None
    cy: float | None = None
    path: Path | None = None
    captured_at: str | None = None
    relative_altitude_m: float | None = None
    absolute_altitude_m: float | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError('width and height must be positive.')
        if not self.focal_px or self.focal_px <= 0:
            raise ValueError('focal_px must be positive.')
        if self.cx is None:
            self.cx = self.width / 2.0
        if self.cy is None:
            self.cy = self.height / 2.0
        if self.path is not None:
            self.path = Path(self.path)

    # ------------------------------------------------------------- geometry
    @property
    def hfov_deg(self) -> float:
        """Horizontal field of view in degrees."""
        return math.degrees(2 * math.atan(self.width / (2 * self.focal_px)))

    @property
    def vfov_deg(self) -> float:
        """Vertical field of view in degrees."""
        return math.degrees(2 * math.atan(self.height / (2 * self.focal_px)))

    @property
    def rotation(self) -> np.ndarray:
        """World-to-camera rotation (see :func:`rotation_from_angles`)."""
        return rotation_from_angles(self.yaw_deg, self.pitch_deg, self.roll_deg)

    def position(self, project: Callable[[float, float], tuple[float, float]]):
        """
        Camera centre in a local east/north/up frame.

        Args:
            project: ``(lon, lat) -> (x, y)`` from
                :func:`~rapidtools.processing.street_localization.local_projection`.

        Returns:
            numpy.ndarray: ``[x, y, elevation]`` in metres.

        Example:
            >>> from rapidtools.processing.street_localization import (
            ...     local_projection,
            ... )
            >>> project, _ = local_projection(0.0, 0.0)
            >>> camera = DroneCamera('a', 0.0, 0.0, 60.0, 0.0, -90.0, 100, 100, 80.0)
            >>> [float(v) for v in camera.position(project)]
            [0.0, 0.0, 60.0]
        """
        x, y = project(self.lon, self.lat)
        return np.array([x, y, self.elevation_m], dtype=float)

    def project_points(
        self,
        points: np.ndarray | Sequence[Sequence[float]],
        project: Callable[[float, float], tuple[float, float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project local east/north/up points into the photo.

        Args:
            points: ``N x 3`` array of local coordinates in metres.
            project: The local projection the points were made with.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: ``N x 2`` pixel coordinates
            (column, row) and the depth of each point along the lens axis.
            Points behind the camera have a depth of zero or less and
            meaningless pixel coordinates.

        Example:
            >>> from rapidtools.processing.street_localization import (
            ...     local_projection,
            ... )
            >>> project, _ = local_projection(0.0, 0.0)
            >>> camera = DroneCamera('a', 0.0, 0.0, 100.0, 0.0, -90.0, 200, 100, 100.0)
            >>> pixels, depth = camera.project_points([[10.0, 0.0, 0.0]], project)
            >>> [round(float(v)) for v in pixels[0]], round(float(depth[0]))
            ([110, 50], 100)
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        cam = (pts - self.position(project)) @ self.rotation.T
        depth = cam[:, 2]
        safe = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
        pixels = np.column_stack(
            [
                self.cx + self.focal_px * cam[:, 0] / safe,
                self.cy + self.focal_px * cam[:, 1] / safe,
            ]
        )
        return pixels, depth

    def project_lonlat(
        self, points: Iterable[Sequence[float]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project ``(lon, lat, elevation)`` points into the photo.

        Args:
            points: Geographic points with elevations in the camera's
                vertical datum.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: Pixels and depths, as
            :meth:`project_points`.

        Example:
            >>> camera = DroneCamera('a', 0.0, 0.0, 100.0, 0.0, -90.0, 200, 100, 100.0)
            >>> pixels, _ = camera.project_lonlat([(0.0, 0.0, 0.0)])
            >>> [round(float(v)) for v in pixels[0]]
            [100, 50]
        """
        project = self._own_projection()[0]
        local = [[*project(p[0], p[1]), p[2]] for p in points]
        return self.project_points(local, project)

    def pixel_ray(self, column: float, row: float) -> np.ndarray:
        """
        Unit direction (east, north, up) of the ray through a pixel.

        Example:
            >>> camera = DroneCamera('a', 0.0, 0.0, 100.0, 90.0, 0.0, 200, 100, 100.0)
            >>> [round(float(v), 6) + 0.0 for v in camera.pixel_ray(100, 50)]
            [1.0, 0.0, 0.0]
        """
        assert self.cx is not None and self.cy is not None  # __post_init__
        cam = np.array(
            [(column - self.cx) / self.focal_px, (row - self.cy) / self.focal_px, 1.0]
        )
        world = self.rotation.T @ cam
        return world / np.linalg.norm(world)

    def pixel_to_ground(
        self, column: float, row: float, ground_elevation_m: float = 0.0
    ) -> tuple[float, float] | None:
        """
        Where the ray through a pixel meets a horizontal ground plane.

        Args:
            column: Pixel column.
            row: Pixel row.
            ground_elevation_m: Elevation of the plane.

        Returns:
            tuple[float, float] | None: ``(lon, lat)`` of the intersection,
            or ``None`` when the ray does not descend to the plane.

        Example:
            >>> camera = DroneCamera('a', 0.0, 0.0, 100.0, 0.0, -90.0, 200, 100, 100.0)
            >>> lon, lat = camera.pixel_to_ground(100, 50)
            >>> round(lon, 6) + 0.0, round(lat, 6) + 0.0
            (0.0, 0.0)
        """
        ray = self.pixel_ray(column, row)
        drop = self.elevation_m - ground_elevation_m
        if ray[2] >= -1e-9 or drop <= 0:
            return None
        scale = drop / -ray[2]
        _, unproject = self._own_projection()
        return unproject(ray[0] * scale, ray[1] * scale)

    def ground_footprint(
        self, ground_elevation_m: float = 0.0, max_range_m: float = 1000.0
    ) -> list[tuple[float, float]]:
        """
        Ground outline of the photo as ``(lon, lat)`` corners.

        Corners whose rays never reach the ground (the view includes the
        horizon) are placed ``max_range_m`` away along their direction.

        Example:
            >>> camera = DroneCamera('a', 0.0, 0.0, 100.0, 0.0, -90.0, 200, 100, 100.0)
            >>> len(camera.ground_footprint())
            4
        """
        _, unproject = self._own_projection()
        drop = max(self.elevation_m - ground_elevation_m, 0.1)
        corners = []
        for column, row in (
            (0, 0),
            (self.width, 0),
            (self.width, self.height),
            (0, self.height),
        ):
            ray = self.pixel_ray(column, row)
            horizontal = math.hypot(ray[0], ray[1])
            reach = drop / -ray[2] * horizontal if ray[2] < -1e-9 else math.inf
            if reach > max_range_m:
                scale = max_range_m / max(horizontal, 1e-9)
            else:
                scale = drop / -ray[2]
            corners.append(unproject(ray[0] * scale, ray[1] * scale))
        return corners

    def _own_projection(self):
        """Local projection centred on the camera."""
        cos_lat = math.cos(math.radians(self.lat))

        def project(lon: float, lat: float) -> tuple[float, float]:
            return (
                math.radians(lon - self.lon) * cos_lat * _EARTH_RADIUS_M,
                math.radians(lat - self.lat) * _EARTH_RADIUS_M,
            )

        def unproject(x: float, y: float) -> tuple[float, float]:
            return (
                self.lon + math.degrees(x / (_EARTH_RADIUS_M * cos_lat)),
                self.lat + math.degrees(y / _EARTH_RADIUS_M),
            )

        return project, unproject

    # -------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        """
        JSON-friendly record of the pose (round-trips through ``from_dict``).

        Example:
            >>> camera = DroneCamera('a', 0.0, 0.0, 100.0, 0.0, -90.0, 200, 100, 100.0)
            >>> DroneCamera.from_dict(camera.to_dict()) == camera
            True
        """
        return {
            'image_id': self.image_id,
            'lon': self.lon,
            'lat': self.lat,
            'elevation_m': self.elevation_m,
            'yaw_deg': self.yaw_deg,
            'pitch_deg': self.pitch_deg,
            'roll_deg': self.roll_deg,
            'width': self.width,
            'height': self.height,
            'focal_px': self.focal_px,
            'cx': self.cx,
            'cy': self.cy,
            'path': str(self.path) if self.path is not None else None,
            'captured_at': self.captured_at,
            'relative_altitude_m': self.relative_altitude_m,
            'absolute_altitude_m': self.absolute_altitude_m,
            'properties': dict(self.properties),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DroneCamera:
        """Rebuild a camera from :meth:`to_dict` output."""
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)


# ------------------------------------------------------------------ reading
def read_xmp_tags(path: str | Path) -> dict[str, str]:
    """
    Read the ``drone-dji`` XMP tags of a photo.

    The XMP packet is plain text inside the file, so it is found by scanning
    the first megabyte rather than by parsing the container; this works for
    JPEG, TIFF and DNG alike and for both attribute and element spellings.

    Args:
        path: Image file.

    Returns:
        dict[str, str]: Tag name (without the ``drone-dji:`` prefix) to its
        raw text value; empty when the file has no such block.

    Example:
        >>> tags = read_xmp_tags('DJI_0001.JPG')  # doctest: +SKIP
        >>> tags['GimbalPitchDegree']  # doctest: +SKIP
        '-45.00'
    """
    with open(path, 'rb') as handle:
        head = handle.read(XMP_SCAN_BYTES)
    start = head.find(b'<x:xmpmeta')
    if start < 0:
        start = head.find(b'<rdf:RDF')
    if start < 0:
        return {}
    end = head.find(b'</x:xmpmeta>', start)
    packet = head[start : end if end > 0 else len(head)].decode('utf-8', 'ignore')
    tags: dict[str, str] = {}
    for name, attribute, element in _XMP_TAG.findall(packet):
        tags.setdefault(name, (attribute or element).strip())
    return tags


def _dms_to_degrees(value: Any, ref: Any) -> float | None:
    """EXIF degrees/minutes/seconds triple to signed decimal degrees."""
    try:
        parts = [float(v) for v in value]
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if len(parts) != 3:
        return None
    degrees = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
    if isinstance(ref, bytes):
        ref = ref.decode('ascii', 'ignore')
    if str(ref).strip().upper() in ('S', 'W'):
        degrees = -degrees
    return degrees


def _focal_px(
    exif: dict[int, Any],
    xmp: dict[str, str],
    width: int,
    height: int,
    default_hfov_deg: float,
) -> tuple[float, str]:
    """Focal length in pixels and where it came from."""
    calibrated = _float(xmp.get('CalibratedFocalLength'))
    if calibrated and calibrated > 0:
        return calibrated, 'xmp_calibrated'
    film = _float(exif.get(0xA405))  # FocalLengthIn35mmFilm
    if film and film > 0:
        return film * math.hypot(width, height) / FULL_FRAME_DIAGONAL_MM, 'exif_35mm'
    focal_mm = _float(exif.get(0x920A))  # FocalLength
    plane_res = _float(exif.get(0xA20E))  # FocalPlaneXResolution
    unit = exif.get(0xA210, 2)  # FocalPlaneResolutionUnit
    if focal_mm and plane_res:
        mm_per_unit = {2: 25.4, 3: 10.0, 4: 1.0, 5: 0.001}.get(int(unit), 25.4)
        return focal_mm * plane_res / mm_per_unit, 'exif_focal_plane'
    focal = (width / 2.0) / math.tan(math.radians(default_hfov_deg) / 2.0)
    return focal, 'default_hfov'


def read_drone_image(
    path: str | Path,
    altitude_source: str = 'auto',
    takeoff_elevation_m: float = 0.0,
    default_hfov_deg: float = DEFAULT_HFOV_DEG,
    yaw_offset_deg: float = 0.0,
) -> DroneCamera | None:
    """
    Read the camera pose of one drone photo from its EXIF and XMP metadata.

    Position comes from the EXIF GPS tags (or the XMP ``GpsLatitude`` /
    ``GpsLongitude`` tags, which carry more decimals on recent aircraft).
    Orientation comes from the XMP gimbal angles, falling back to the flight
    yaw with a straight-down camera when the gimbal block is missing, and to
    the EXIF ``GPSImgDirection`` after that. The lens comes from the
    calibrated focal length, the 35 mm equivalent or the focal-plane
    resolution, in that order.

    Args:
        path: Image file.
        altitude_source: ``'relative'`` uses the height above take-off plus
            ``takeoff_elevation_m``; ``'absolute'`` uses the GPS altitude,
            which on consumer aircraft can be tens of metres off;
            ``'auto'`` (default) prefers the relative height.
        takeoff_elevation_m: Elevation of the take-off point in the datum
            of the ground elevations used downstream. The default of ``0``
            makes the take-off point the zero of a flat world.
        default_hfov_deg: Horizontal field of view assumed when the photo
            has no lens data. Defaults to 73.7 (a 24 mm equivalent lens).
        yaw_offset_deg: Added to the recorded yaw, for a compass bias found
            by calibration.

    Returns:
        DroneCamera | None: The pose, or ``None`` when the photo has no
        usable position.

    Example:
        >>> camera = read_drone_image('DJI_0001.JPG')  # doctest: +SKIP
        >>> camera.pitch_deg, round(camera.elevation_m)  # doctest: +SKIP
        (-45.0, 80)
    """
    if altitude_source not in ('auto', 'relative', 'absolute'):
        raise ValueError(
            "altitude_source must be 'auto', 'relative' or 'absolute', "
            f'got {altitude_source!r}.'
        )
    path = Path(path)
    try:
        with Image.open(path) as image:
            width, height = image.size
            raw = image.getexif()
            exif = dict(raw)
            exif.update(raw.get_ifd(_EXIF_IFD))
            gps = dict(raw.get_ifd(_GPS_IFD))
    except (OSError, ValueError, SyntaxError, struct.error) as exc:
        logger.warning(f'Could not read {path.name}: {exc}')
        return None
    xmp = read_xmp_tags(path)

    lat = _float(xmp.get('GpsLatitude'))
    lon = _float(xmp.get('GpsLongitude', xmp.get('GpsLongtitude')))
    if lat is None or lon is None or (lat == 0 and lon == 0):
        lat = _dms_to_degrees(gps.get(2), gps.get(1))
        lon = _dms_to_degrees(gps.get(4), gps.get(3))
    if lat is None or lon is None:
        logger.warning(f'{path.name} has no GPS position; skipped.')
        return None

    relative = _float(xmp.get('RelativeAltitude'))
    absolute = _float(xmp.get('AbsoluteAltitude'))
    if absolute is None:
        absolute = _float(gps.get(6))
        if absolute is not None and gps.get(5) in (1, b'\x01'):
            absolute = -absolute
    if altitude_source == 'relative' or (
        altitude_source == 'auto' and relative is not None
    ):
        if relative is None:
            logger.warning(f'{path.name} has no relative altitude; skipped.')
            return None
        elevation = takeoff_elevation_m + relative
    else:
        if absolute is None:
            logger.warning(f'{path.name} has no altitude; skipped.')
            return None
        elevation = absolute

    yaw = _float(xmp.get('GimbalYawDegree'))
    pitch = _float(xmp.get('GimbalPitchDegree'))
    roll = _float(xmp.get('GimbalRollDegree')) or 0.0
    pose_source = 'gimbal'
    if yaw is None:
        yaw = _float(xmp.get('FlightYawDegree'))
        pose_source = 'flight_yaw'
    if yaw is None:
        yaw = _float(gps.get(17))  # GPSImgDirection
        pose_source = 'gps_img_direction'
    if yaw is None:
        yaw, pose_source = 0.0, 'assumed_nadir'
    if pitch is None:
        pitch = -90.0
    # A gimbal reported past straight down has flipped over: the lens looks
    # backwards. Fold it back so pitch stays within [-90, 90].
    if pitch < -90.0:
        pitch, yaw = -180.0 - pitch, yaw + 180.0

    focal, focal_source = _focal_px(exif, xmp, width, height, default_hfov_deg)
    cx = _float(xmp.get('CalibratedOpticalCenterX'))
    cy = _float(xmp.get('CalibratedOpticalCenterY'))
    captured = exif.get(0x9003) or exif.get(0x0132)
    return DroneCamera(
        image_id=path.stem,
        lon=lon,
        lat=lat,
        elevation_m=elevation,
        yaw_deg=(yaw + yaw_offset_deg) % 360.0,
        pitch_deg=pitch,
        roll_deg=roll,
        width=width,
        height=height,
        focal_px=focal,
        cx=cx,
        cy=cy,
        path=path,
        captured_at=str(captured) if captured else None,
        relative_altitude_m=relative,
        absolute_altitude_m=absolute,
        properties={'pose_source': pose_source, 'focal_source': focal_source},
    )


_SIDECAR_ALIASES: dict[str, tuple[str, ...]] = {
    'image': ('image', 'filename', 'file', 'name', 'image_id', 'id', 'path'),
    'lon': ('lon', 'longitude', 'lng', 'x'),
    'lat': ('lat', 'latitude', 'y'),
    'elevation_m': ('elevation_m', 'altitude', 'alt', 'elevation', 'z', 'height_m'),
    'yaw_deg': ('yaw_deg', 'yaw', 'heading', 'azimuth', 'gimbal_yaw'),
    'pitch_deg': ('pitch_deg', 'pitch', 'gimbal_pitch', 'tilt'),
    'roll_deg': ('roll_deg', 'roll', 'gimbal_roll'),
    'focal_px': ('focal_px', 'focal_length_px', 'f_px'),
    'hfov_deg': ('hfov_deg', 'hfov', 'fov', 'horizontal_fov'),
    'width': ('width', 'image_width'),
    'height': ('height', 'image_height'),
    'captured_at': ('captured_at', 'timestamp', 'time', 'datetime'),
}


def _sidecar_rows(sidecar: str | Path) -> list[dict[str, Any]]:
    """Rows of a pose table (CSV, or GeoJSON with Point features)."""
    sidecar = Path(sidecar)
    if sidecar.suffix.lower() in ('.geojson', '.json'):
        data = json.loads(sidecar.read_text())
        rows = []
        for feature in data.get('features', []):
            row = dict(feature.get('properties') or {})
            coords = (feature.get('geometry') or {}).get('coordinates') or []
            if len(coords) >= 2:
                row.setdefault('lon', coords[0])
                row.setdefault('lat', coords[1])
            if len(coords) >= 3:
                row.setdefault('elevation_m', coords[2])
            rows.append(row)
        return rows
    with open(sidecar, newline='') as handle:
        return list(csv.DictReader(handle))


def read_pose_table(
    sidecar: str | Path,
    image_directory: str | Path | None = None,
    default_hfov_deg: float = DEFAULT_HFOV_DEG,
    takeoff_elevation_m: float = 0.0,
) -> list[DroneCamera]:
    """
    Read camera poses from a sidecar table.

    For aircraft that do not write DJI-style metadata, or for poses refined
    by photogrammetry. The table is a CSV with a header row, or a GeoJSON of
    Point features; column names are matched loosely (``image`` or
    ``filename``; ``lon``/``longitude``; ``lat``/``latitude``; ``altitude``
    or ``elevation_m``; ``yaw``/``heading``; ``pitch``; optional ``roll``,
    ``focal_px`` or ``hfov_deg``, ``width``, ``height``). Pitch is ``0`` at
    the horizon and ``-90`` straight down.

    Args:
        sidecar: Path of the table.
        image_directory: Folder the image names are relative to; the
            table's own folder by default. Image sizes are read from the
            files when the table has no ``width`` and ``height``.
        default_hfov_deg: Field of view for rows without lens data.
        takeoff_elevation_m: Added to every altitude in the table.

    Returns:
        list[DroneCamera]: One camera per usable row.

    Example:
        >>> cameras = read_pose_table('flight/poses.csv')  # doctest: +SKIP
        >>> cameras[0].yaw_deg  # doctest: +SKIP
        135.0
    """
    sidecar = Path(sidecar)
    base = Path(image_directory) if image_directory else sidecar.parent
    cameras: list[DroneCamera] = []
    for number, raw in enumerate(_sidecar_rows(sidecar)):
        lowered = {str(k).strip().lower(): v for k, v in raw.items()}
        row: dict[str, Any] = {}
        for key, aliases in _SIDECAR_ALIASES.items():
            for alias in aliases:
                if lowered.get(alias) not in (None, ''):
                    row[key] = lowered[alias]
                    break
        lon, lat = _float(row.get('lon')), _float(row.get('lat'))
        elevation = _float(row.get('elevation_m'))
        if lon is None or lat is None or elevation is None:
            logger.warning(f'Row {number + 1} of {sidecar.name} has no pose; skipped.')
            continue
        name = str(row.get('image', f'image_{number:05d}'))
        path = base / name
        if not path.exists():
            matches = [base / f'{name}{s}' for s in IMAGE_SUFFIXES]
            path = next((m for m in matches if m.exists()), path)
        width, height = _float(row.get('width')), _float(row.get('height'))
        if (width is None or height is None) and path.exists():
            with Image.open(path) as image:
                width, height = image.size
        if width is None or height is None:
            logger.warning(f'Row {number + 1} of {sidecar.name} has no size; skipped.')
            continue
        focal = _float(row.get('focal_px'))
        if focal is None:
            hfov = _float(row.get('hfov_deg')) or default_hfov_deg
            focal = (width / 2.0) / math.tan(math.radians(hfov) / 2.0)
        pitch = _float(row.get('pitch_deg'))
        cameras.append(
            DroneCamera(
                image_id=Path(name).stem,
                lon=lon,
                lat=lat,
                elevation_m=elevation + takeoff_elevation_m,
                yaw_deg=(_float(row.get('yaw_deg')) or 0.0) % 360.0,
                pitch_deg=-90.0 if pitch is None else pitch,
                roll_deg=_float(row.get('roll_deg')) or 0.0,
                width=int(width),
                height=int(height),
                focal_px=focal,
                path=path if path.exists() else None,
                captured_at=(
                    str(row['captured_at']) if row.get('captured_at') else None
                ),
                properties={'pose_source': 'sidecar'},
            )
        )
    return cameras


def read_drone_images(
    source: str | Path | Iterable[str | Path],
    sidecar: str | Path | None = None,
    altitude_source: str = 'auto',
    takeoff_elevation_m: float = 0.0,
    default_hfov_deg: float = DEFAULT_HFOV_DEG,
    yaw_offset_deg: float = 0.0,
    recursive: bool = True,
) -> list[DroneCamera]:
    """
    Read the camera poses of a set of drone photos.

    Args:
        source: A folder of photos, one photo, or an iterable of photo
            paths.
        sidecar: Optional pose table (see :func:`read_pose_table`) used
            instead of the embedded metadata; image names in it are
            relative to ``source`` when that is a folder.
        altitude_source: See :func:`read_drone_image`.
        takeoff_elevation_m: See :func:`read_drone_image`.
        default_hfov_deg: Field of view for photos without lens data.
        yaw_offset_deg: Added to every recorded yaw.
        recursive: Search sub-folders of ``source``. Defaults to ``True``.

    Returns:
        list[DroneCamera]: Cameras sorted by image name. Photos without a
        usable pose are skipped with a warning.

    Example:
        >>> cameras = read_drone_images('flight_01/')  # doctest: +SKIP
        >>> len(cameras), cameras[0].image_id  # doctest: +SKIP
        (412, 'DJI_0001')
    """
    if sidecar is not None:
        directory = None
        if isinstance(source, (str, Path)) and Path(source).is_dir():
            directory = Path(source)
        return read_pose_table(
            sidecar, directory, default_hfov_deg, takeoff_elevation_m
        )
    if isinstance(source, (str, Path)):
        root = Path(source)
        if root.is_dir():
            walker = root.rglob('*') if recursive else root.glob('*')
            paths = sorted(
                p for p in walker if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()
            )
        else:
            paths = [root]
    else:
        paths = sorted(Path(p) for p in source)
    cameras = []
    for path in paths:
        camera = read_drone_image(
            path, altitude_source, takeoff_elevation_m, default_hfov_deg, yaw_offset_deg
        )
        if camera is not None:
            cameras.append(camera)
    if paths and not cameras:
        logger.warning('None of the photos carried a usable camera pose.')
    return cameras


# ------------------------------------------------------------------ writing
def _rational(value: float, scale: int = 1_000_000) -> IFDRational:
    return IFDRational(int(round(abs(value) * scale)), scale)


def _degrees_to_dms(value: float) -> tuple[IFDRational, ...]:
    value = abs(value)
    degrees = int(value)
    minutes = int((value - degrees) * 60)
    seconds = (value - degrees - minutes / 60.0) * 3600.0
    return (
        IFDRational(degrees, 1),
        IFDRational(minutes, 1),
        _rational(seconds, 10_000_000),
    )


def dji_xmp_packet(camera: DroneCamera) -> bytes:
    """
    The pose of ``camera`` as a DJI-style XMP packet.

    Example:
        >>> camera = DroneCamera('a', 1.0, 2.0, 80.0, 90.0, -45.0, 200, 100, 150.0)
        >>> b'drone-dji:GimbalPitchDegree="-45.00"' in dji_xmp_packet(camera)
        True
    """
    relative = camera.relative_altitude_m
    if relative is None:
        relative = camera.elevation_m
    absolute = camera.absolute_altitude_m
    if absolute is None:
        absolute = camera.elevation_m
    yaw = (camera.yaw_deg + 180.0) % 360.0 - 180.0
    tags = {
        'GpsLatitude': f'{camera.lat:.8f}',
        'GpsLongitude': f'{camera.lon:.8f}',
        'AbsoluteAltitude': f'{absolute:+.3f}',
        'RelativeAltitude': f'{relative:+.3f}',
        'GimbalRollDegree': f'{camera.roll_deg:+.2f}',
        'GimbalYawDegree': f'{yaw:+.2f}',
        'GimbalPitchDegree': f'{camera.pitch_deg:+.2f}',
        'FlightRollDegree': '+0.00',
        'FlightYawDegree': f'{yaw:+.2f}',
        'FlightPitchDegree': '+0.00',
        'CalibratedFocalLength': f'{camera.focal_px:.4f}',
        'CalibratedOpticalCenterX': f'{camera.cx:.4f}',
        'CalibratedOpticalCenterY': f'{camera.cy:.4f}',
    }
    attributes = '\n'.join(f'   drone-dji:{k}="{v}"' for k, v in tags.items())
    packet = (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about="DJI Meta Data"\n'
        '   xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/"\n'
        f'{attributes}>\n'
        '  </rdf:Description>\n'
        ' </rdf:RDF>\n'
        '</x:xmpmeta>\n'
    )
    return packet.encode('utf-8')


def write_pose_metadata(
    image: Image.Image | str | Path,
    destination: str | Path,
    camera: DroneCamera,
    quality: int = 92,
) -> Path:
    """
    Save a JPEG that carries ``camera`` as EXIF GPS tags and DJI-style XMP.

    Useful for simulated flights and for writing refined poses back into
    photos, so that every tool reading DJI metadata (this package included)
    sees them.

    Args:
        image: The picture, or the path of one.
        destination: JPEG file to write.
        camera: Pose and lens to embed. ``relative_altitude_m`` and
            ``absolute_altitude_m`` are written when set, ``elevation_m``
            otherwise.
        quality: JPEG quality. Defaults to 92.

    Returns:
        pathlib.Path: The written file.

    Example:
        >>> from PIL import Image
        >>> camera = DroneCamera('a', 1.0, 2.0, 80.0, 90.0, -45.0, 64, 48, 60.0)
        >>> path = write_pose_metadata(
        ...     Image.new('RGB', (64, 48)), 'pose_demo.jpg', camera
        ... )  # doctest: +SKIP
        >>> read_drone_image(path).pitch_deg  # doctest: +SKIP
        -45.0
    """
    destination = Path(destination)
    picture: Image.Image
    if isinstance(image, Image.Image):
        picture, opened = image, False
    else:
        picture, opened = Image.open(image), True
    try:
        absolute = camera.absolute_altitude_m
        if absolute is None:
            absolute = camera.elevation_m
        exif = Image.Exif()
        exif[0x010F] = 'rapidtools'  # Make
        exif[0x0110] = 'simulated drone camera'  # Model
        if camera.captured_at:
            exif[0x0132] = camera.captured_at
        gps = {
            0: b'\x02\x03\x00\x00',
            1: 'N' if camera.lat >= 0 else 'S',
            2: _degrees_to_dms(camera.lat),
            3: 'E' if camera.lon >= 0 else 'W',
            4: _degrees_to_dms(camera.lon),
            5: b'\x01' if absolute < 0 else b'\x00',
            6: _rational(absolute, 1000),
            16: 'T',
            17: _rational(camera.yaw_deg % 360.0, 100),
        }
        exif[_GPS_IFD] = gps
        destination.parent.mkdir(parents=True, exist_ok=True)
        picture.convert('RGB').save(destination, 'JPEG', quality=quality, exif=exif)
    finally:
        if opened:
            picture.close()
    _insert_xmp(destination, dji_xmp_packet(camera))
    return destination


def _insert_xmp(path: Path, packet: bytes) -> None:
    """Insert an XMP APP1 segment after the leading APP0/APP1 segments."""
    data = path.read_bytes()
    if data[:2] != b'\xff\xd8':
        raise ValueError(f'{path} is not a JPEG file.')
    payload = _XMP_HEADER + packet
    if len(payload) + 2 > 0xFFFF:
        raise ValueError('XMP packet too large for one JPEG segment.')
    segment = b'\xff\xe1' + struct.pack('>H', len(payload) + 2) + payload
    offset = 2
    while offset + 4 <= len(data) and data[offset] == 0xFF:
        marker = data[offset + 1]
        if marker not in (0xE0, 0xE1):
            break
        offset += 2 + struct.unpack('>H', data[offset + 2 : offset + 4])[0]
    path.write_bytes(data[:offset] + segment + data[offset:])
