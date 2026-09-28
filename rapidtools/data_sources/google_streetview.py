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
# 09-22-2026

"""
Keyless access to Google Street View panoramas.

The client relies on the same undocumented endpoints the Google Maps web
application uses (also employed by BRAILS++ and the ``streetlevel`` project),
so no Street View Static API key is needed:

* ``SingleImageSearch`` - find the panorama closest to a coordinate.
* ``photometa`` - full panorama metadata, neighbour links and depth map.
* ``streetviewpixels-pa`` / ``cbk`` tile servers - 512 px panorama tiles.

Because these endpoints are unofficial they may change without notice; every
network call degrades gracefully to ``None`` and is logged.

Example:
    >>> from rapidtools.data_sources import GoogleStreetViewClient
    >>> client = GoogleStreetViewClient()
    >>> pano = client.find_panorama(34.1879, -118.1350, radius_m=50)
    >>> pano.id, round(pano.heading)
    ('u3PxkEsnYto4l1N3yTmO_Q', 21)
    >>> image = client.download_panorama(pano, zoom=2)
    >>> image.size
    (2048, 1024)
"""

from __future__ import annotations

import base64
import concurrent.futures
import json
import logging
import math
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image
from shapely.geometry.base import BaseGeometry

from rapidtools.config import REQUESTS_TIMEOUT_VAL, get_configured_session

logger = logging.getLogger(__name__)

SINGLE_IMAGE_SEARCH_URL = (
    'https://maps.googleapis.com/$rpc/google.internal.maps.mapsjs.v1.'
    'MapsJsInternalService/SingleImageSearch'
)
PHOTOMETA_URL = 'https://www.google.com/maps/photometa/v1'
TILE_URL = (
    'https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile'
    '&panoid={pano_id}&x={x}&y={y}&zoom={zoom}&nbt=1&fover=2'
)
PHOTOMETA_PB = (
    '!1m4!1smaps_sv.tactile!11m2!2m1!1b1!2m2!1sen!2sus!3m3!1m2!1e2!2s{pano_id}'
    '!4m57!1e1!1e2!1e3!1e4!1e5!1e6!1e8!1e12!2m1!1e1!4m1!1i48!5m1!1e1!5m1!1e2'
    '!6m1!1e1!6m1!1e2!9m36!1m3!1e2!2b1!3e2!1m3!1e2!2b0!3e3!1m3!1e3!2b1!3e2'
    '!1m3!1e3!2b0!3e3!1m3!1e8!2b0!3e3!1m3!1e1!2b0!3e3!1m3!1e4!2b0!3e3!1m3'
    '!1e10!2b1!3e2!1m3!1e10!2b0!3e3'
)

TILE_SIZE = 512
EARTH_RADIUS_M = 6371008.8
# Horizontal field of view covered by a panorama rendered at each zoom level.
ZOOM_TO_FOV_DEG = {0: 360, 1: 180, 2: 90, 3: 45, 4: 22.5, 5: 11.25}


@dataclass
class StreetViewPanorama:
    """
    Metadata describing one Google Street View panorama.

    Attributes:
        id: Google panorama ID.
        lat: Camera latitude in degrees.
        lon: Camera longitude in degrees.
        heading: Compass heading of the panorama centre column (degrees,
            0 = north, clockwise).
        pitch: Camera pitch in degrees (Google reports ~90 for level).
        roll: Camera roll in degrees.
        elevation: Camera elevation above sea level in metres, if known.
        date: ``(year, month)`` of capture, if known.
        address: Human-readable address strings reported by Google.
        sizes: ``{zoom: (width, height)}`` of the full panorama per zoom level.
        tile_size: Edge length of the panorama tiles in pixels.
        links: Neighbouring panoramas as ``(pano_id, lat, lon, heading)``.
        depth_map_b64: URL-safe Base64 depth-map string (see
            :func:`decode_depth_map`), if the metadata included one.

    Example:
        >>> pano = StreetViewPanorama(id='abc', lat=34.0, lon=-118.0, heading=90.0)
        >>> pano.size_at_zoom(0)
        (512, 256)
    """

    id: str
    lat: float
    lon: float
    heading: float = 0.0
    pitch: float = 90.0
    roll: float = 0.0
    elevation: float | None = None
    date: tuple[int, int] | None = None
    address: list[str] = field(default_factory=list)
    sizes: dict[int, tuple[int, int]] = field(default_factory=dict)
    tile_size: int = TILE_SIZE
    links: list[tuple[str, float, float, float]] = field(default_factory=list)
    depth_map_b64: str | None = None

    def size_at_zoom(self, zoom: int) -> tuple[int, int]:
        """
        Return the full panorama ``(width, height)`` at ``zoom``.

        Falls back to the canonical ``512 * 2**zoom`` layout when the metadata
        did not report explicit sizes.

        Args:
            zoom: Tile zoom level (0-5).

        Returns:
            tuple[int, int]: Panorama width and height in pixels.
        """
        if zoom in self.sizes:
            return self.sizes[zoom]
        width = TILE_SIZE * (2**zoom)
        return width, width // 2

    @property
    def capture_date(self) -> str | None:
        """``'YYYY-MM'`` capture date or ``None``."""
        if not self.date:
            return None
        year, month = self.date
        return f'{year:04d}-{month:02d}'


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two WGS84 points in metres.

    Example:
        >>> round(haversine_m(0.0, 0.0, 0.0, 1.0))
        111195
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlmb = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Initial compass bearing from point 1 to point 2 in ``[0, 360)`` degrees.

    Example:
        >>> bearing_deg(0.0, 0.0, 0.0, 1.0)
        90.0
        >>> bearing_deg(0.0, 0.0, 1.0, 0.0)
        0.0
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    x = math.sin(dlmb) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(
        dlmb
    )
    return math.degrees(math.atan2(x, y)) % 360.0


def relative_angle(bearing: float, heading: float) -> float:
    """
    Signed angle of ``bearing`` relative to ``heading`` in ``(-180, 180]``.

    Positive values are clockwise (to the right of the heading).

    Example:
        >>> relative_angle(90.0, 0.0), relative_angle(350.0, 10.0)
        (90.0, -20.0)
    """
    angle = (bearing - heading + 180.0) % 360.0 - 180.0
    return 180.0 if angle == -180.0 else angle


def decode_depth_map(depth_map_b64: str) -> np.ndarray:
    """
    Decode Google's Base64 plane-encoded depth map into per-pixel distances.

    The encoding stores a small grid of plane indices plus a table of planes
    (normal + distance). Each pixel's depth is the distance along its viewing
    ray to the assigned plane; pixels without a plane (sky) are ``inf``.

    Args:
        depth_map_b64: URL-safe Base64 string from
            :attr:`StreetViewPanorama.depth_map_b64`.

    Returns:
        np.ndarray: ``(height, width)`` float32 array of distances in metres,
        oriented like the panorama image (column 0 on the left).

    Raises:
        ValueError: If the string is too short to contain a header.

    Example:
        >>> depth = decode_depth_map(pano.depth_map_b64)
        >>> depth.shape
        (256, 512)
        >>> float(np.isinf(depth).mean()) < 1.0
        True
    """
    padded = depth_map_b64 + '=' * ((4 - len(depth_map_b64) % 4) % 4)
    data = np.frombuffer(base64.urlsafe_b64decode(padded), dtype=np.uint8)
    if data.size < 8:
        raise ValueError('Depth map string is too short to contain a header.')

    header_size = int(data[0])
    n_planes = int(data[1]) | (int(data[2]) << 8)
    width = int(data[3]) | (int(data[4]) << 8)
    height = int(data[5]) | (int(data[6]) << 8)
    offset = int(data[7]) | (int(data[8]) << 8) if data.size > 8 else header_size

    indices = data[offset : offset + width * height].astype(np.int64)
    plane_bytes = data[
        offset + width * height : offset + width * height + n_planes * 16
    ]
    planes = np.frombuffer(plane_bytes.tobytes(), dtype='<f4').reshape(-1, 4)
    if planes.shape[0] < n_planes:  # pragma: no cover - defensive for truncated data
        n_planes = planes.shape[0]

    ys = np.arange(height, dtype=np.float64)
    xs = np.arange(width, dtype=np.float64)
    theta = (height - ys - 0.5) / height * np.pi  # (H,)
    phi = (width - xs - 0.5) / width * 2 * np.pi + np.pi / 2  # (W,)
    sin_theta = np.sin(theta)[:, None]
    ray_x = sin_theta * np.cos(phi)[None, :]
    ray_y = sin_theta * np.sin(phi)[None, :]
    ray_z = np.repeat(np.cos(theta)[:, None], width, axis=1)

    idx = indices.reshape(height, width)
    valid = (idx > 0) & (idx < n_planes)
    safe_idx = np.where(valid, idx, 0)
    normals = planes[safe_idx, :3]
    dists = planes[safe_idx, 3]
    dot = ray_x * normals[..., 0] + ray_y * normals[..., 1] + ray_z * normals[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = np.abs(dists / dot)
    depth = np.where(valid, depth, np.inf)
    return depth.astype(np.float32)


class GoogleStreetViewClient:
    """
    Keyless client for locating, downloading and cropping Street View panoramas.

    Args:
        max_workers: Threads used to download panorama tiles in parallel.
        timeout: Per-request timeout in seconds.

    Example:
        >>> from shapely.geometry import Point
        >>> from rapidtools.data_sources import GoogleStreetViewClient
        >>> client = GoogleStreetViewClient()
        >>> pano = client.find_panorama(34.1879, -118.1350)
        >>> pano = client.get_panorama_metadata(pano.id)
        >>> image = client.download_panorama(pano, zoom=3)
        >>> crop, bounds = client.crop_to_geometry(
        ...     image, pano, Point(-118.13500, 34.18800).buffer(0.0001)
        ... )
        >>> crop.size[1] == image.size[1]
        True
    """

    def __init__(self, max_workers: int = 8, timeout: float = REQUESTS_TIMEOUT_VAL):
        """Create the retrying HTTP session and remember the pool size."""
        self.max_workers = max(1, max_workers)
        self.timeout = timeout
        self.session = get_configured_session()

    # ------------------------------------------------------------ parsing
    @staticmethod
    def _get(node: Any, *path: int) -> Any:
        """Safely index nested lists; return ``None`` on any miss."""
        for key in path:
            try:
                node = node[key]
            except (IndexError, KeyError, TypeError):
                return None
        return node

    @classmethod
    def _parse_pano_node(cls, node: Any, pano_id: str) -> StreetViewPanorama | None:
        """Build a :class:`StreetViewPanorama` from a ``[1][k]`` metadata node."""
        cam = cls._get(node, 5, 0, 1)
        lat, lon = cls._get(cam, 0, 2), cls._get(cam, 0, 3)
        if lat is None or lon is None:
            return None
        angles = cls._get(cam, 2) or []
        heading = angles[0] if len(angles) > 0 and angles[0] is not None else 0.0
        pitch = angles[1] if len(angles) > 1 and angles[1] is not None else 90.0
        roll = angles[2] if len(angles) > 2 and angles[2] is not None else 0.0

        sizes: dict[int, tuple[int, int]] = {}
        for zoom, entry in enumerate(cls._get(node, 2, 3, 0) or []):
            dims = cls._get(entry, 0)
            if isinstance(dims, list) and len(dims) == 2:
                sizes[zoom] = (int(dims[1]), int(dims[0]))  # stored as [h, w]
        tile_dims = cls._get(node, 2, 4)
        tile_size = int(tile_dims[0]) if isinstance(tile_dims, list) else TILE_SIZE

        date_node = cls._get(node, 6, 7)
        date = None
        if isinstance(date_node, list) and len(date_node) >= 2:
            date = (int(date_node[0]), int(date_node[1]))

        address = [
            entry[0]
            for entry in (cls._get(node, 3, 2) or [])
            if isinstance(entry, list) and entry and isinstance(entry[0], str)
        ]

        links = []
        for link in cls._get(node, 5, 0, 3, 0) or []:
            link_id = cls._get(link, 0, 1)
            link_lat, link_lon = cls._get(link, 2, 0, 2), cls._get(link, 2, 0, 3)
            link_heading = cls._get(link, 2, 2, 0)
            if (
                isinstance(link_id, str)
                and link_id != pano_id
                and link_lat is not None
                and link_lon is not None
            ):
                links.append((link_id, link_lat, link_lon, link_heading or 0.0))

        depth = cls._get(node, 5, 0, 5, 1, 2)
        return StreetViewPanorama(
            id=pano_id,
            lat=lat,
            lon=lon,
            heading=heading,
            pitch=pitch,
            roll=roll,
            elevation=cls._get(cam, 1, 0),
            date=date,
            address=address,
            sizes=sizes,
            tile_size=tile_size,
            links=links,
            depth_map_b64=depth if isinstance(depth, str) else None,
        )

    # ------------------------------------------------------------ lookups
    def find_panorama(
        self, lat: float, lon: float, radius_m: float = 50.0
    ) -> StreetViewPanorama | None:
        """
        Find the Street View panorama closest to a coordinate.

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            radius_m: Search radius in metres. Defaults to 50.

        Returns:
            StreetViewPanorama | None: The nearest official panorama with its
            location, heading, sizes, date and neighbour links (the depth map
            requires :meth:`get_panorama_metadata`), or ``None`` when nothing
            is within the radius or the request failed.

        Example:
            >>> client = GoogleStreetViewClient()
            >>> pano = client.find_panorama(34.1879, -118.1350, radius_m=50)
            >>> pano.capture_date
            '2025-11'
        """
        payload = [
            ['apiv3', None, None, None, 'US', None, None, None, None, None, [[0]]],
            [[None, None, lat, lon], radius_m],
            [
                None,
                ['en', 'US'],
                None,
                None,
                None,
                None,
                None,
                None,
                [2],
                None,
                [[[2, True, 2]]],
            ],
            [[1, 2, 3, 4, 8, 6]],
        ]
        try:
            response = self.session.post(
                SINGLE_IMAGE_SEARCH_URL,
                data=json.dumps(payload),
                headers={'Content-Type': 'application/json+protobuf'},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001 - unofficial endpoint
            logger.error(f'Street View search failed at ({lat}, {lon}): {exc}')
            return None

        node = self._get(data, 1)
        pano_id = self._get(node, 1, 1)
        if not isinstance(pano_id, str):
            logger.info(
                f'No Street View panorama within {radius_m} m of ({lat}, {lon}).'
            )
            return None
        return self._parse_pano_node(node, pano_id)

    def get_panorama_metadata(self, pano_id: str) -> StreetViewPanorama | None:
        """
        Fetch full metadata (including the depth map) for a panorama ID.

        Args:
            pano_id: Google panorama ID.

        Returns:
            StreetViewPanorama | None: Metadata, or ``None`` on failure.

        Example:
            >>> client = GoogleStreetViewClient()
            >>> pano = client.get_panorama_metadata('u3PxkEsnYto4l1N3yTmO_Q')
            >>> pano.depth_map_b64 is not None
            True
        """
        params = {
            'authuser': '0',
            'hl': 'en',
            'gl': 'us',
            'pb': PHOTOMETA_PB.format(pano_id=pano_id),
        }
        try:
            response = self.session.get(
                PHOTOMETA_URL, params=params, timeout=self.timeout
            )
            response.raise_for_status()
            text = response.text
            data = json.loads(
                text[text.index('\n') + 1 :] if text.startswith(')]}') else text
            )
        except Exception as exc:  # noqa: BLE001 - unofficial endpoint
            logger.error(f'Street View metadata request failed for {pano_id}: {exc}')
            return None
        node = self._get(data, 1, 0)
        pano = self._parse_pano_node(node, pano_id) if node else None
        if pano is None:
            logger.warning(f'No metadata returned for panorama {pano_id}.')
        return pano

    # ------------------------------------------------------------ download
    def _download_tile(
        self, pano_id: str, x: int, y: int, zoom: int
    ) -> Image.Image | None:
        """Download one panorama tile; ``None`` on any failure."""
        url = TILE_URL.format(pano_id=pano_id, x=x, y=y, zoom=zoom)
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as exc:  # noqa: BLE001 - reported and left black
            logger.warning(f'Failed to download tile ({x}, {y}) of {pano_id}: {exc}')
            return None

    def download_panorama(
        self, pano: StreetViewPanorama | str, zoom: int = 3
    ) -> Image.Image | None:
        """
        Download and stitch the full equirectangular panorama.

        Args:
            pano: A :class:`StreetViewPanorama` or a bare panorama ID.
            zoom: Tile zoom level 0-5; the width doubles per level
                (zoom 3 is 4096 x 2048 px, 45 deg of view per 512 px).

        Returns:
            Image.Image | None: The stitched RGB panorama, or ``None`` when no
            tile could be downloaded.

        Raises:
            ValueError: If ``zoom`` is outside 0-5.

        Example:
            >>> client = GoogleStreetViewClient()
            >>> image = client.download_panorama('u3PxkEsnYto4l1N3yTmO_Q', zoom=1)
            >>> image.size
            (1024, 512)
        """
        if zoom not in ZOOM_TO_FOV_DEG:
            raise ValueError(f'zoom must be between 0 and 5, got {zoom}.')
        if isinstance(pano, str):
            pano = StreetViewPanorama(id=pano, lat=0.0, lon=0.0)

        width, height = pano.size_at_zoom(zoom)
        tile = pano.tile_size
        n_x, n_y = math.ceil(width / tile), math.ceil(height / tile)
        canvas = Image.new('RGB', (n_x * tile, n_y * tile))

        coords = [(x, y) for y in range(n_y) for x in range(n_x)]
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(coords))
        ) as executor:
            futures = {
                executor.submit(self._download_tile, pano.id, x, y, zoom): (x, y)
                for x, y in coords
            }
            downloaded = 0
            for future in concurrent.futures.as_completed(futures):
                x, y = futures[future]
                img = future.result()
                if img is not None:
                    canvas.paste(img, (x * tile, y * tile))
                    downloaded += 1
        if downloaded == 0:
            logger.error(f'No tiles could be downloaded for panorama {pano.id}.')
            return None
        return canvas.crop((0, 0, width, height))

    # -------------------------------------------------------------- cropping
    @staticmethod
    def view_angles(pano: StreetViewPanorama, geometry: BaseGeometry) -> list[float]:
        """
        Signed viewing angles (degrees) of a geometry's vertices from the camera.

        Angles are measured relative to the panorama heading: 0 is straight
        ahead, positive to the right, negative to the left.

        Args:
            pano: Panorama metadata providing camera position and heading.
            geometry: Shapely geometry in WGS84 (any type with coordinates).

        Returns:
            list[float]: One angle per vertex of the geometry's convex hull
            (or the point itself).

        Example:
            >>> from shapely.geometry import Point
            >>> pano = StreetViewPanorama(id='p', lat=0.0, lon=0.0, heading=0.0)
            >>> GoogleStreetViewClient.view_angles(pano, Point(0.001, 0.0))
            [90.0]
        """
        hull = geometry if geometry.geom_type == 'Point' else geometry.convex_hull
        if hull.geom_type == 'Point':
            coords = [(hull.x, hull.y)]
        elif hasattr(hull, 'exterior'):
            coords = list(hull.exterior.coords)[:-1]
        else:
            coords = list(hull.coords)
        return [
            relative_angle(bearing_deg(pano.lat, pano.lon, lat, lon), pano.heading)
            for lon, lat in coords
        ]

    @staticmethod
    def _crop_columns(image: Image.Image, x0: int, x1: int) -> Image.Image:
        """Crop columns ``x0..x1`` of a panorama, wrapping around its seam."""
        width, height = image.size
        span = x1 - x0
        if span >= width:
            return image.copy()
        x0 %= width
        x1 = x0 + span
        if x1 <= width:
            return image.crop((x0, 0, x1, height))
        left = image.crop((x0, 0, width, height))
        right = image.crop((0, 0, x1 - width, height))
        out = Image.new('RGB', (span, height))
        out.paste(left, (0, 0))
        out.paste(right, (left.size[0], 0))
        return out

    def crop_to_geometry(
        self,
        image: Image.Image,
        pano: StreetViewPanorama,
        geometry: BaseGeometry,
        fov_buffer_deg: float = 10.0,
        min_fov_deg: float = 30.0,
        vertical_crop: tuple[float, float] = (0.0, 1.0),
    ) -> tuple[Image.Image, tuple[float, float]]:
        """
        Crop a panorama to the horizontal span occupied by a geometry.

        Args:
            image: The full panorama from :meth:`download_panorama`.
            pano: Its metadata (camera position and heading).
            geometry: Target asset geometry in WGS84.
            fov_buffer_deg: Extra margin added on both sides. Defaults to 10.
            min_fov_deg: Minimum horizontal field of view of the crop, so tiny
                or distant assets still yield a usable image. Defaults to 30.
            vertical_crop: ``(top, bottom)`` fractions of the panorama height
                to keep, e.g. ``(0.2, 0.9)`` removes sky and the vehicle hood.

        Returns:
            tuple[Image.Image, tuple[float, float]]: The crop and the
            ``(start, end)`` viewing angles in degrees relative to the heading
            that it covers.

        Raises:
            ValueError: If ``vertical_crop`` is not an increasing pair in
                ``[0, 1]``.

        Example:
            >>> crop, (start, end) = client.crop_to_geometry(image, pano, footprint)
            >>> end - start >= 30.0
            True
        """
        top_frac, bottom_frac = vertical_crop
        if not 0.0 <= top_frac < bottom_frac <= 1.0:
            raise ValueError('vertical_crop must satisfy 0 <= top < bottom <= 1.')

        angles = self.view_angles(pano, geometry)
        # Handle assets spanning the seam behind the camera (+/-180).
        if max(angles) - min(angles) > 180.0:
            angles = [a % 360.0 for a in angles]
        start = min(angles) - fov_buffer_deg
        end = max(angles) + fov_buffer_deg
        if end - start < min_fov_deg:
            centre = (start + end) / 2.0
            start, end = centre - min_fov_deg / 2.0, centre + min_fov_deg / 2.0
        end = min(end, start + 360.0)

        width, height = image.size
        x0 = int(round((start + 180.0) / 360.0 * width))
        x1 = int(round((end + 180.0) / 360.0 * width))
        crop = self._crop_columns(image, x0, max(x1, x0 + 1))
        if (top_frac, bottom_frac) != (0.0, 1.0):
            crop = crop.crop(
                (
                    0,
                    int(round(top_frac * height)),
                    crop.size[0],
                    int(round(bottom_frac * height)),
                )
            )
        return crop, (start, end)
