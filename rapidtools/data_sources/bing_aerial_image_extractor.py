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
# Copyright (c) 2025 The University of Washington
#
# This file is part of rapidtools.
#
# Contributors:
# Barbaros Cetiner
#
# Last updated:
# 05-24-2026

import concurrent.futures
import json
import math
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from requests.adapters import HTTPAdapter
from shapely.geometry import box, shape

from rapidtools.config import get_configured_session


class BingAerialImageExtractor:
    """
    Client for extracting high-resolution imagery from Bing Maps Tiles.

    This class acts as a context manager for efficient batch processing. It
    uses a shared requests Session with connection pooling. It supports both
    stitching tiles into large images and yielding individual tiles for machine
    learning pipelines.
    """

    def __init__(
        self,
        output_dir: str | Path = 'output',
        zoom_level: int = 20,
        max_workers: int = 10,
    ):
        """
        Initialize the extractor configuration.

        Args:
            output_dir: Directory where stitched images will be saved.
            zoom_level: Map detail level (1-23). Defaults to 20.
            max_workers: Number of concurrent threads to use for downloading tiles.
        """
        self.output_dir = Path(output_dir).resolve()
        self.zoom_level = zoom_level
        self.max_workers = max_workers

        self._session = None
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None

    def __enter__(self):
        """
        Open the network session and thread pool for batch processing.

        Returns:
            BingAerialImageExtractor: The initialized extractor instance.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Get the highly robust session from rapidtools.config
        self._session = get_configured_session()

        # 2. Extract the retry strategy that config.py already built
        retry_strategy = self._session.adapters['https://'].max_retries

        # 3. Create an adapter combining retries with custom connection pools
        pool_adapter = HTTPAdapter(
            pool_connections=self.max_workers,
            pool_maxsize=self.max_workers * 2,
            max_retries=retry_strategy,
        )

        # 4. Mount the high-capacity pool adapters
        self._session.mount('http://', pool_adapter)
        self._session.mount('https://', pool_adapter)

        # Initialize the ThreadPoolExecutor
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Safely close the network session and shut down the thread pool."""
        if self._session is not None:
            self._session.close()
            self._session = None

        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    @staticmethod
    def lat_lon_to_pixel(lat: float, lon: float, zoom: int) -> tuple[int, int]:
        """
        Convert latitude and longitude to Bing Maps pixel coordinates.

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            zoom: Bing Maps zoom level (1-23).

        Returns:
            tuple[int, int]: Pixel X and Y coordinates on the global map.
        """
        sin_lat = math.sin(lat * math.pi / 180.0)
        sin_lat = max(min(sin_lat, 0.9999), -0.9999)
        map_size = 256 << zoom

        pixel_x = ((lon + 180) / 360) * map_size
        y_calc = 0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)
        pixel_y = y_calc * map_size
        return int(pixel_x), int(pixel_y)

    @staticmethod
    def pixel_to_lat_lon(pixel_x: int, pixel_y: int, zoom: int) -> tuple[float, float]:
        """
        Convert Bing Maps pixel coordinates back to latitude and longitude.

        Args:
            pixel_x: Global pixel X coordinate.
            pixel_y: Global pixel Y coordinate.
            zoom: Bing Maps zoom level (1-23).

        Returns:
            tuple[float, float]: Latitude and longitude in degrees.
        """
        map_size = 256 << zoom
        lon = (pixel_x / map_size) * 360.0 - 180.0

        y = 0.5 - (pixel_y / map_size)
        lat = 90.0 - 360.0 * math.atan(math.exp(-y * 2 * math.pi)) / math.pi
        return lat, lon

    @staticmethod
    def tile_to_quadkey(tile_x: int, tile_y: int, zoom: int) -> str:
        """
        Convert tile XY coordinates into a Bing Maps Quadkey string.

        Args:
            tile_x: Tile X coordinate.
            tile_y: Tile Y coordinate.
            zoom: Bing Maps zoom level (1-23).

        Returns:
            str: The quadkey string used to fetch the tile from Bing servers.
        """
        quadkey = ''
        for i in range(zoom, 0, -1):
            digit = 0
            mask = 1 << (i - 1)
            if (tile_x & mask) != 0:
                digit += 1
            if (tile_y & mask) != 0:
                digit += 2
            quadkey += str(digit)
        return quadkey

    def _download_tile(self, quadkey: str) -> Image.Image | None:
        """
        Download a single tile using the active requests session.

        Args:
            quadkey: The Bing Maps quadkey string for the tile.

        Returns:
            Image.Image | None: The downloaded image, or None if the request failed.
        """
        if self._session is None:
            raise RuntimeError(
                'Session is not initialized. Use within a "with" block.'
            )

        url = f'http://ecn.t3.tiles.virtualearth.net/tiles/a{quadkey}.jpeg?g=1'
        try:
            response = self._session.get(url, timeout=10)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
        except Exception:
            pass
        return None

    def generate_region_tiles(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
    ):
        """
        Generator that yields individual Bing tiles for a bounding box.

        Args:
            min_lat: Minimum latitude of the bounding box.
            min_lon: Minimum longitude of the bounding box.
            max_lat: Maximum latitude of the bounding box.
            max_lon: Maximum longitude of the bounding box.

        Yields:
            tuple: (tile_img, bbox, quadkey) where tile_img is a PIL Image,
            bbox is a tuple of (min_lon, min_lat, max_lon, max_lat), and
            quadkey is the string identifier for the tile.
        """
        if self._session is None:
            raise RuntimeError('Extractor must be used as a context manager.')

        px_min_x, px_max_y = self.lat_lon_to_pixel(min_lat, min_lon, self.zoom_level)
        px_max_x, px_min_y = self.lat_lon_to_pixel(max_lat, max_lon, self.zoom_level)

        tile_min_x = px_min_x // 256
        tile_min_y = px_min_y // 256
        tile_max_x = px_max_x // 256
        tile_max_y = px_max_y // 256

        for tx in range(tile_min_x, tile_max_x + 1):
            for ty in range(tile_min_y, tile_max_y + 1):
                quadkey = self.tile_to_quadkey(tx, ty, self.zoom_level)
                tile_img = self._download_tile(quadkey)

                if tile_img:
                    t_min_lat, t_min_lon = self.pixel_to_lat_lon(
                        tx * 256, (ty + 1) * 256, self.zoom_level
                    )
                    t_max_lat, t_max_lon = self.pixel_to_lat_lon(
                        (tx + 1) * 256, ty * 256, self.zoom_level
                    )
                    bbox = (t_min_lon, t_min_lat, t_max_lon, t_max_lat)

                    yield tile_img, bbox, quadkey

    def generate_polygon_tiles(self, feature: dict[str, Any]):
        """
        Generator that yields individual Bing tiles intersecting a GeoJSON polygon.

        Args:
            feature: A GeoJSON feature dict containing a polygon geometry.

        Yields:
            tuple: (tile_img, bbox, quadkey, intersection_area) where tile_img is
            a PIL Image, bbox is the tile's geographic bounds, quadkey is the string
            identifier, and intersection_area is the polygon area covered by the tile.
        """
        if self._session is None:
            raise RuntimeError('Extractor must be used as a context manager.')

        polygon = shape(feature['geometry'])
        min_lon, min_lat, max_lon, max_lat = polygon.bounds

        px_min_x, px_max_y = self.lat_lon_to_pixel(min_lat, min_lon, self.zoom_level)
        px_max_x, px_min_y = self.lat_lon_to_pixel(max_lat, max_lon, self.zoom_level)

        tile_min_x = px_min_x // 256
        tile_min_y = px_min_y // 256
        tile_max_x = px_max_x // 256
        tile_max_y = px_max_y // 256

        for tx in range(tile_min_x, tile_max_x + 1):
            for ty in range(tile_min_y, tile_max_y + 1):
                t_min_lat, t_min_lon = self.pixel_to_lat_lon(
                    tx * 256, (ty + 1) * 256, self.zoom_level
                )
                t_max_lat, t_max_lon = self.pixel_to_lat_lon(
                    (tx + 1) * 256, ty * 256, self.zoom_level
                )

                tile_geom = box(t_min_lon, t_min_lat, t_max_lon, t_max_lat)
                intersection = polygon.intersection(tile_geom)

                if not intersection.is_empty:
                    quadkey = self.tile_to_quadkey(tx, ty, self.zoom_level)
                    tile_img = self._download_tile(quadkey)

                    if tile_img:
                        bbox = (t_min_lon, t_min_lat, t_max_lon, t_max_lat)
                        yield tile_img, bbox, quadkey, intersection.area

    def process_polygon(
        self,
        feature: dict[str, Any],
        index: int,
        buffer_percent: float = 0.10,
    ) -> str:
        """
        Download, stitch, and crop tiles for a single GeoJSON polygon feature.

        Tiles are fetched concurrently using the internal thread pool to drastically
        speed up the extraction process. The final image is saved to `output_dir`.

        Args:
            feature: GeoJSON feature dict representing the target polygon.
            index: Identifier index for error reporting and default naming.
            buffer_percent: Padding percentage added around the polygon bounding box.

        Returns:
            str: A success or failure message including the polygon ID.
        """
        if self._session is None or self._executor is None:
            raise RuntimeError('Extractor must be used as a context manager.')

        try:
            # Cleanly extract bounds using Shapely
            polygon = shape(feature['geometry'])
            min_lon, min_lat, max_lon, max_lat = polygon.bounds

            lon_diff = max_lon - min_lon
            lat_diff = max_lat - min_lat

            # Apply buffer limits safely
            min_lon = max(-180.0, min_lon - (lon_diff * buffer_percent))
            max_lon = min(180.0, max_lon + (lon_diff * buffer_percent))
            min_lat = max(-85.05112878, min_lat - (lat_diff * buffer_percent))
            max_lat = min(85.05112878, max_lat + (lat_diff * buffer_percent))

            px_min_x, px_max_y = self.lat_lon_to_pixel(
                min_lat, min_lon, self.zoom_level
            )
            px_max_x, px_min_y = self.lat_lon_to_pixel(
                max_lat, max_lon, self.zoom_level
            )

            tile_min_x = px_min_x // 256
            tile_min_y = px_min_y // 256
            tile_max_x = px_max_x // 256
            tile_max_y = px_max_y // 256

            width_tiles = tile_max_x - tile_min_x + 1
            height_tiles = tile_max_y - tile_min_y + 1

            canvas = Image.new('RGB', (width_tiles * 256, height_tiles * 256))

            # Multithread the tile downloads for this specific polygon:
            quadkeys_to_fetch = []
            for tx in range(tile_min_x, tile_max_x + 1):
                for ty in range(tile_min_y, tile_max_y + 1):
                    quadkey = self.tile_to_quadkey(tx, ty, self.zoom_level)
                    quadkeys_to_fetch.append((tx, ty, quadkey))

            futures = {
                self._executor.submit(self._download_tile, qk): (tx, ty)
                for tx, ty, qk in quadkeys_to_fetch
            }

            # As tiles finish downloading, paste them onto the canvas:
            for future in concurrent.futures.as_completed(futures):
                tx, ty = futures[future]
                tile_img = future.result()

                if tile_img:
                    paste_x = (tx - tile_min_x) * 256
                    paste_y = (ty - tile_min_y) * 256
                    canvas.paste(tile_img, (paste_x, paste_y))

            crop_left = px_min_x - (tile_min_x * 256)
            crop_top = px_min_y - (tile_min_y * 256)
            crop_right = crop_left + (px_max_x - px_min_x)
            crop_bottom = crop_top + (px_max_y - px_min_y)

            final_image = canvas.crop((crop_left, crop_top, crop_right, crop_bottom))

            poly_id = feature.get('properties', {}).get('id', f'polygon_{index}')
            output_path = self.output_dir / f'{poly_id}.jpg'
            final_image.save(output_path, quality=95)

            return f'Success: {poly_id}'

        except Exception as e:
            return f'Failed processing polygon {index}: {str(e)}'

    def process_geojson(self, geojson_path: str | Path, buffer_percent: float = 0.10):
        """
        Read a GeoJSON file and stitch/save images for all enclosed polygons.

        Polygons are processed sequentially to allow `process_polygon` to utilize
        100% of the internal thread pool for rapidly fetching individual tiles,
        avoiding thread pool starvation deadlocks.

        Args:
            geojson_path: Path to the input GeoJSON file.
            buffer_percent: Padding percentage added around each bounding box.
        """
        if self._executor is None:
            raise RuntimeError('Extractor must be used as a context manager.')

        with open(geojson_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        features = data.get('features', [])
        print(f'Loaded {len(features)} polygons from GeoJSON.')
        print(
            f'Starting downloads at Zoom Level {self.zoom_level} '
            f'with a {buffer_percent*100}% buffer...'
        )

        for i, feature in enumerate(features):
            result = self.process_polygon(feature, i, buffer_percent)
            print(result)


# Example usage:
if __name__ == '__main__':

    # Generator for Vision Models:
    print('--- Testing Generator Approach ---')
    mock_feature = {
        'type': 'Feature',
        'properties': {'id': 'test_building'},
        'geometry': {
            'type': 'Polygon',
            'coordinates': [
                [
                    [-118.250, 34.050],
                    [-118.250, 34.051],
                    [-118.251, 34.051],
                    [-118.251, 34.050],
                    [-118.250, 34.050],
                ]
            ],
        },
    }

    with BingAerialImageExtractor(max_workers=10) as extractor:
        for img, bbox, qk, area in extractor.generate_polygon_tiles(mock_feature):
            print(f'Yielded tile {qk} | BBox: {bbox} | Intersection Area: {area:.6f}')
            # Run SAM inference here