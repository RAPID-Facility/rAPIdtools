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
Shared machinery for keyless Web Mercator aerial-tile providers.

Bing Maps and Google Maps both serve 256 px aerial tiles in the Web Mercator
(EPSG:3857) XYZ scheme; only the URL template (and Bing's quadkey encoding)
differ. :class:`WebMercatorTileExtractor` implements the projection math, the
retrying session and thread pool, tile generators, stitching and GeoJSON batch
processing once. :class:`~rapidtools.data_sources.BingAerialImageExtractor`
and :class:`~rapidtools.data_sources.GoogleAerialImageExtractor` only declare
how a tile key becomes a URL.

Example:
    >>> from rapidtools.data_sources import GoogleAerialImageExtractor
    >>> with GoogleAerialImageExtractor('tiles', zoom_level=19) as extractor:
    ...     for tile, bbox, key in extractor.generate_region_tiles(
    ...         34.050, -118.251, 34.051, -118.250
    ...     ):
    ...         print(key, tile.size)
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Generator, Hashable
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from shapely.geometry import box, shape

from rapidtools.config import (
    REQUESTS_TIMEOUT_VAL,
    get_configured_session,
    resolve_alias,
)

logger = logging.getLogger(__name__)

TILE_SIZE = 256
MAX_MERCATOR_LAT = 85.05112878


class WebMercatorTileExtractor(ABC):
    """
    Base class for extracting aerial imagery from XYZ tile servers.

    Instances are context managers: entering opens a pooled, retrying
    :class:`requests.Session` and a thread pool; exiting closes both.

    Class attributes:
        PROVIDER_NAME: Human-readable provider label used in logs.

    Example:
        >>> from rapidtools.data_sources import BingAerialImageExtractor
        >>> with BingAerialImageExtractor('out', zoom_level=18) as extractor:
        ...     message = extractor.process_polygon(feature, index=0)
        >>> message.startswith('Success')
        True
    """

    PROVIDER_NAME: str = 'Web Mercator tiles'

    def __init__(
        self,
        save_directory: str | Path = 'output',
        zoom_level: int = 20,
        max_workers: int = 10,
        **kwargs: Any,
    ):
        """
        Initialize the extractor configuration.

        Args:
            save_directory: Directory where stitched images will be saved.
                (``output_dir`` is accepted as a deprecated alias.)
            zoom_level: Map detail level (1-23). Defaults to 20.
            max_workers: Number of concurrent threads used to download tiles.
        """
        legacy = resolve_alias(kwargs, 'save_directory', 'output_dir')
        if legacy is not None:
            save_directory = legacy
        self.save_directory = Path(save_directory).resolve()
        self.zoom_level = zoom_level
        self.max_workers = max_workers

        self._session: requests.Session | None = None
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None

    @property
    def output_dir(self) -> Path:
        """Deprecated alias of :attr:`save_directory`."""
        return self.save_directory

    @output_dir.setter
    def output_dir(self, value: str | Path) -> None:
        """Set :attr:`save_directory` through the deprecated alias."""
        self.save_directory = Path(value).resolve()

    # ----------------------------------------------------------- lifecycle
    def __enter__(self):
        """
        Open the network session and thread pool for batch processing.

        Creates ``output_dir`` if needed, builds a retrying session with a
        connection pool sized to ``max_workers`` and starts the thread pool.

        Returns:
            The initialized extractor instance.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._session = get_configured_session()
        retry_strategy = self._session.adapters['https://'].max_retries
        pool_adapter = HTTPAdapter(
            pool_connections=self.max_workers,
            pool_maxsize=self.max_workers * 2,
            max_retries=retry_strategy,
        )
        self._session.mount('http://', pool_adapter)
        self._session.mount('https://', pool_adapter)

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the network session and shut down the thread pool."""
        if self._session is not None:
            self._session.close()
            self._session = None
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _require_session(self) -> requests.Session:
        """Return the open session; raise ``RuntimeError`` outside a ``with`` block."""
        if self._session is None:
            raise RuntimeError(
                'Extractor must be used as a context manager (inside a "with" '
                'block) before downloading tiles.'
            )
        return self._session

    # ------------------------------------------------------ projection math
    @staticmethod
    def lat_lon_to_pixel(lat: float, lon: float, zoom: int) -> tuple[int, int]:
        """
        Convert latitude and longitude to global Web Mercator pixel coordinates.

        Args:
            lat: Latitude in degrees (clamped to the Mercator range).
            lon: Longitude in degrees.
            zoom: Zoom level (1-23). The world is ``256 << zoom`` pixels wide.

        Returns:
            tuple[int, int]: ``(pixel_x, pixel_y)`` with the origin at the
            top-left corner of the world map.

        Example:
            >>> WebMercatorTileExtractor.lat_lon_to_pixel(0.0, 0.0, zoom=1)
            (256, 256)
        """
        lat = max(min(lat, MAX_MERCATOR_LAT), -MAX_MERCATOR_LAT)
        sin_lat = math.sin(math.radians(lat))
        map_size = TILE_SIZE << zoom

        pixel_x = ((lon + 180) / 360) * map_size
        y_calc = 0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)
        pixel_y = y_calc * map_size
        # lon=180 and the Mercator limit land on the map edge (index
        # map_size), which belongs to the last pixel:
        return (
            min(max(int(pixel_x), 0), map_size - 1),
            min(max(int(pixel_y), 0), map_size - 1),
        )

    @staticmethod
    def pixel_to_lat_lon(pixel_x: int, pixel_y: int, zoom: int) -> tuple[float, float]:
        """
        Convert global Web Mercator pixel coordinates back to latitude/longitude.

        Args:
            pixel_x: Global pixel X coordinate.
            pixel_y: Global pixel Y coordinate.
            zoom: Zoom level (1-23).

        Returns:
            tuple[float, float]: ``(lat, lon)`` in degrees.

        Example:
            >>> WebMercatorTileExtractor.pixel_to_lat_lon(256, 256, zoom=1)
            (0.0, 0.0)
        """
        map_size = TILE_SIZE << zoom
        lon = (pixel_x / map_size) * 360.0 - 180.0
        y = 0.5 - (pixel_y / map_size)
        lat = 90.0 - 360.0 * math.atan(math.exp(-y * 2 * math.pi)) / math.pi
        return lat, lon

    @classmethod
    def tile_bounds(
        cls, tile_x: int, tile_y: int, zoom: int
    ) -> tuple[float, float, float, float]:
        """
        Return the geographic bounds of an XYZ tile.

        Args:
            tile_x: Tile column index.
            tile_y: Tile row index.
            zoom: Zoom level.

        Returns:
            tuple[float, float, float, float]:
                ``(min_lon, min_lat, max_lon, max_lat)``.

        Example:
            >>> WebMercatorTileExtractor.tile_bounds(0, 0, zoom=1)[2:]
            (0.0, 0.0)
        """
        min_lat, min_lon = cls.pixel_to_lat_lon(
            tile_x * TILE_SIZE, (tile_y + 1) * TILE_SIZE, zoom
        )
        max_lat, max_lon = cls.pixel_to_lat_lon(
            (tile_x + 1) * TILE_SIZE, tile_y * TILE_SIZE, zoom
        )
        return min_lon, min_lat, max_lon, max_lat

    def _tile_range(
        self, min_lat: float, min_lon: float, max_lat: float, max_lon: float
    ) -> tuple[int, int, int, int]:
        """Return ``(tile_min_x, tile_min_y, tile_max_x, tile_max_y)`` for a box."""
        px_min_x, px_max_y = self.lat_lon_to_pixel(min_lat, min_lon, self.zoom_level)
        px_max_x, px_min_y = self.lat_lon_to_pixel(max_lat, max_lon, self.zoom_level)
        return (
            px_min_x // TILE_SIZE,
            px_min_y // TILE_SIZE,
            px_max_x // TILE_SIZE,
            px_max_y // TILE_SIZE,
        )

    # -------------------------------------------------------- provider hooks
    def _tile_key(self, tile_x: int, tile_y: int) -> Hashable:
        """
        Return the provider-specific identifier of a tile.

        Defaults to the ``(x, y, zoom)`` triple; Bing overrides it with the
        quadkey string.
        """
        return (tile_x, tile_y, self.zoom_level)

    @abstractmethod
    def _tile_url(self, tile_key: Any) -> str:
        """Return the download URL of the tile identified by ``tile_key``."""

    def _download_tile(self, tile_key: Any) -> Image.Image | None:
        """
        Download a single tile using the active requests session.

        Network or decoding failures are swallowed and reported as ``None`` so
        that one missing tile does not abort a batch job.

        Args:
            tile_key: Provider-specific tile identifier from :meth:`_tile_key`.

        Returns:
            Image.Image | None: The tile image, or ``None`` on failure.

        Raises:
            RuntimeError: If called outside of a context manager block.
        """
        session = self._require_session()
        try:
            response = session.get(
                self._tile_url(tile_key), timeout=REQUESTS_TIMEOUT_VAL
            )
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
        except Exception:  # noqa: BLE001 - a bad tile must not abort the batch
            pass
        return None

    # ------------------------------------------------------------ generators
    def generate_region_tiles(
        self, min_lat: float, min_lon: float, max_lat: float, max_lon: float
    ) -> Generator[tuple[Image.Image, tuple[float, float, float, float], Hashable]]:
        """
        Yield individual tiles covering a bounding box.

        Args:
            min_lat: Minimum latitude of the bounding box.
            min_lon: Minimum longitude of the bounding box.
            max_lat: Maximum latitude of the bounding box.
            max_lon: Maximum longitude of the bounding box.

        Yields:
            tuple: ``(tile_img, bbox, tile_key)`` where ``bbox`` is
            ``(min_lon, min_lat, max_lon, max_lat)`` of the tile. Tiles that
            fail to download are skipped.

        Raises:
            RuntimeError: If called outside of a context manager block.

        Example:
            >>> with GoogleAerialImageExtractor(zoom_level=19) as extractor:
            ...     tiles = list(extractor.generate_region_tiles(
            ...         34.050, -118.251, 34.051, -118.250
            ...     ))
            >>> tiles[0][0].size
            (256, 256)
        """
        self._require_session()
        tile_min_x, tile_min_y, tile_max_x, tile_max_y = self._tile_range(
            min_lat, min_lon, max_lat, max_lon
        )
        for tx in range(tile_min_x, tile_max_x + 1):
            for ty in range(tile_min_y, tile_max_y + 1):
                key = self._tile_key(tx, ty)
                tile_img = self._download_tile(key)
                if tile_img is not None:
                    yield tile_img, self.tile_bounds(tx, ty, self.zoom_level), key

    def generate_polygon_tiles(
        self, feature: dict[str, Any]
    ) -> Generator[
        tuple[Image.Image, tuple[float, float, float, float], Hashable, float]
    ]:
        """
        Yield individual tiles intersecting a GeoJSON polygon feature.

        Args:
            feature: A GeoJSON feature dict containing a polygon geometry.

        Yields:
            tuple: ``(tile_img, bbox, tile_key, intersection_area)`` where
            ``intersection_area`` is the polygon area (in square degrees)
            covered by the tile.

        Raises:
            RuntimeError: If called outside of a context manager block.
        """
        self._require_session()
        polygon = shape(feature['geometry'])
        min_lon, min_lat, max_lon, max_lat = polygon.bounds
        tile_min_x, tile_min_y, tile_max_x, tile_max_y = self._tile_range(
            min_lat, min_lon, max_lat, max_lon
        )
        for tx in range(tile_min_x, tile_max_x + 1):
            for ty in range(tile_min_y, tile_max_y + 1):
                bounds = self.tile_bounds(tx, ty, self.zoom_level)
                intersection = polygon.intersection(box(*bounds))
                if intersection.is_empty:
                    continue
                key = self._tile_key(tx, ty)
                tile_img = self._download_tile(key)
                if tile_img is not None:
                    yield tile_img, bounds, key, intersection.area

    # ------------------------------------------------------------- stitching
    def stitch_region(
        self, min_lat: float, min_lon: float, max_lat: float, max_lon: float
    ) -> Image.Image:
        """
        Download every tile covering a bounding box and crop the mosaic to it.

        Tiles are fetched concurrently with the internal thread pool; failed
        tiles are left black.

        Args:
            min_lat: Minimum latitude of the bounding box.
            min_lon: Minimum longitude of the bounding box.
            max_lat: Maximum latitude of the bounding box.
            max_lon: Maximum longitude of the bounding box.

        Returns:
            Image.Image: The RGB mosaic cropped exactly to the box.

        Raises:
            RuntimeError: If called outside of a context manager block.
            ValueError: If the box collapses to zero pixels at ``zoom_level``.

        Example:
            >>> with GoogleAerialImageExtractor(zoom_level=19) as extractor:
            ...     mosaic = extractor.stitch_region(34.050, -118.251, 34.051, -118.250)
            >>> mosaic.size[0] > 0
            True
        """
        self._require_session()
        if self._executor is None:  # pragma: no cover - guarded by __enter__
            raise RuntimeError('Extractor must be used as a context manager.')

        px_min_x, px_max_y = self.lat_lon_to_pixel(min_lat, min_lon, self.zoom_level)
        px_max_x, px_min_y = self.lat_lon_to_pixel(max_lat, max_lon, self.zoom_level)
        if px_max_x <= px_min_x or px_max_y <= px_min_y:
            raise ValueError(
                'Region collapses to zero pixels at zoom '
                f'{self.zoom_level}; use a larger region or higher zoom.'
            )

        tile_min_x, tile_min_y = px_min_x // TILE_SIZE, px_min_y // TILE_SIZE
        tile_max_x, tile_max_y = px_max_x // TILE_SIZE, px_max_y // TILE_SIZE
        width_tiles = tile_max_x - tile_min_x + 1
        height_tiles = tile_max_y - tile_min_y + 1
        canvas = Image.new('RGB', (width_tiles * TILE_SIZE, height_tiles * TILE_SIZE))

        futures = {
            self._executor.submit(self._download_tile, self._tile_key(tx, ty)): (
                tx,
                ty,
            )
            for tx in range(tile_min_x, tile_max_x + 1)
            for ty in range(tile_min_y, tile_max_y + 1)
        }
        for future in concurrent.futures.as_completed(futures):
            tx, ty = futures[future]
            tile_img = future.result()
            if tile_img is not None:
                canvas.paste(
                    tile_img,
                    ((tx - tile_min_x) * TILE_SIZE, (ty - tile_min_y) * TILE_SIZE),
                )

        crop_left = px_min_x - tile_min_x * TILE_SIZE
        crop_top = px_min_y - tile_min_y * TILE_SIZE
        return canvas.crop(
            (
                crop_left,
                crop_top,
                crop_left + (px_max_x - px_min_x),
                crop_top + (px_max_y - px_min_y),
            )
        )

    @staticmethod
    def pad_to_square(image: Image.Image) -> Image.Image:
        """
        Center an image on a black square canvas whose side is its longer edge.

        Args:
            image: Any PIL image.

        Returns:
            Image.Image: A square RGB image.

        Example:
            >>> from PIL import Image
            >>> WebMercatorTileExtractor.pad_to_square(Image.new('RGB', (40, 20))).size
            (40, 40)
        """
        side = max(image.size)
        padded = Image.new('RGB', (side, side))
        offset = ((side - image.size[0]) // 2, (side - image.size[1]) // 2)
        padded.paste(image, offset)
        return padded

    @staticmethod
    def buffered_bounds(
        polygon, buffer_percent: float
    ) -> tuple[float, float, float, float]:
        """
        Expand a polygon's bounds by a percentage of its width and height.

        Args:
            polygon: Any Shapely geometry.
            buffer_percent: Fraction of the extent added on every side.

        Returns:
            tuple[float, float, float, float]:
                ``(min_lat, min_lon, max_lat, max_lon)`` clamped to the
                Mercator range.

        Example:
            >>> from shapely.geometry import box
            >>> WebMercatorTileExtractor.buffered_bounds(box(0, 0, 10, 10), 0.1)
            (-1.0, -1.0, 11.0, 11.0)
        """
        min_lon, min_lat, max_lon, max_lat = polygon.bounds
        lon_diff = max_lon - min_lon
        lat_diff = max_lat - min_lat
        return (
            max(-MAX_MERCATOR_LAT, min_lat - lat_diff * buffer_percent),
            max(-180.0, min_lon - lon_diff * buffer_percent),
            min(MAX_MERCATOR_LAT, max_lat + lat_diff * buffer_percent),
            min(180.0, max_lon + lon_diff * buffer_percent),
        )

    def process_polygon(
        self,
        feature: dict[str, Any],
        index: int,
        buffer_percent: float = 0.10,
        pad_to_square: bool = False,
        resize_to: tuple[int, int] | None = None,
    ) -> str:
        """
        Download, stitch, crop and save imagery for one GeoJSON polygon feature.

        The saved file is ``<output_dir>/<id>.jpg`` where ``id`` comes from
        ``feature['properties']['id']`` (falling back to ``polygon_<index>``).

        Args:
            feature: GeoJSON feature dict representing the target polygon.
            index: Identifier index used for error reporting and naming.
            buffer_percent: Padding added around the polygon bounding box as a
                fraction of its extent. Defaults to 0.10.
            pad_to_square: Center the crop on a square black canvas.
            resize_to: Optional ``(width, height)`` to resize the final image
                to (e.g. ``(640, 640)``).

        Returns:
            str: ``'Success: <id>'`` or ``'Failed processing polygon <index>: ...'``.

        Raises:
            RuntimeError: If called outside of a context manager block.

        Example:
            >>> with GoogleAerialImageExtractor('crops', zoom_level=20) as ex:
            ...     ex.process_polygon(
            ...         feature, 0, pad_to_square=True, resize_to=(640, 640)
            ...     )
            'Success: test_building'
        """
        self._require_session()
        try:
            polygon = shape(feature['geometry'])
            min_lat, min_lon, max_lat, max_lon = self.buffered_bounds(
                polygon, buffer_percent
            )
            final_image = self.stitch_region(min_lat, min_lon, max_lat, max_lon)
            if pad_to_square:
                final_image = self.pad_to_square(final_image)
            if resize_to is not None:
                final_image = final_image.resize(resize_to)

            poly_id = feature.get('properties', {}).get('id', f'polygon_{index}')
            output_path = self.output_dir / f'{poly_id}.jpg'
            final_image.save(output_path, quality=95)
            return f'Success: {poly_id}'
        except Exception as e:  # noqa: BLE001 - reported per polygon
            return f'Failed processing polygon {index}: {str(e)}'

    def process_geojson(
        self,
        geojson_path: str | Path,
        buffer_percent: float = 0.10,
        pad_to_square: bool = False,
        resize_to: tuple[int, int] | None = None,
    ) -> list[str]:
        """
        Read a GeoJSON file and save one image per polygon feature.

        Polygons are processed sequentially so that :meth:`process_polygon`
        can use the whole thread pool for its tiles.

        Args:
            geojson_path: Path to the input GeoJSON file.
            buffer_percent: Padding added around each bounding box.
            pad_to_square: Center each crop on a square canvas.
            resize_to: Optional ``(width, height)`` for the final images.

        Returns:
            list[str]: One status message per feature (see
            :meth:`process_polygon`).

        Raises:
            RuntimeError: If called outside of a context manager block.
        """
        self._require_session()
        with open(geojson_path, encoding='utf-8') as f:
            data = json.load(f)

        features = data.get('features', [])
        logger.info(
            f'{self.PROVIDER_NAME}: loaded {len(features)} polygons; downloading '
            f'at zoom {self.zoom_level} with a {buffer_percent * 100:g}% buffer...'
        )
        results = []
        for i, feature in enumerate(features):
            result = self.process_polygon(
                feature, i, buffer_percent, pad_to_square, resize_to
            )
            logger.info(result)
            results.append(result)
        return results
