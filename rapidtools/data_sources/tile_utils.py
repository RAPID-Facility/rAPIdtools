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
# 12-04-2025

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bounding_box import BoundingBox

class TileUtils:
    """
    Utility methods for Web Mercator/Mapbox Vector Tile operations.

    This class provides static helpers for:
    - Converting between WGS84 (lat/lon) and XYZ tile coordinates
    - Converting Mapbox Vector Tile (MVT) local coordinates to WGS84
    - Determining zoom levels and tile coverage for bounding boxes
    - Converting millisecond timestamps to UTC dates

    All geographic coordinates are assumed to be:
    - Latitude in degrees, in the range [-85.05112878, 85.05112878]
    - Longitude in degrees, in the range [-180, 180]
    """

    @staticmethod
    def latlon_to_tile(lat: float, lon: float, z: int) -> tuple[float, float]:
        """
        Convert latitude/longitude to fractional XYZ tile coordinates.

        Uses the Web Mercator tiling scheme where: at zoom level ``z`` there 
        are ``2^z`` tiles in both X and Y.

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            z: Zoom level (e.g., 0–22).

        Returns:
            A tuple ``(xtile, ytile)`` of fractional tile coordinates at
            zoom ``z``. The integer parts ``floor(xtile)`` and
            ``floor(ytile)`` are the tile indices; the fractional parts
            represent the position within that tile.
        """
        lat_rad = math.radians(lat)
        n = 2.0 ** z
        xtile = (lon + 180.0) / 360.0 * n
        ytile = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
        return xtile, ytile

    @staticmethod
    def mvt_to_wgs84(
        shape_x: int,
        shape_y: int,
        tile_x: int,
        tile_y: int,
        tile_z: int,
        extent: int = 4096,
    ) -> tuple[float, float]:
        """
        Convert MVT local tile coordinates to WGS84 (longitude, latitude).

        The local coordinates ``shape_x`` and ``shape_y`` are integers in the
        coordinate space of a single Mapbox Vector Tile (commonly
        ``0..extent``). The enclosing tile is specified by global XYZ tile
        indices.

        Args:
            shape_x: X coordinate in the MVT local coordinate space
                (0 to ``extent``).
            shape_y: Y coordinate in the MVT local coordinate space
                (0 to ``extent``).
            tile_x: Global tile X index at zoom ``tile_z``.
            tile_y: Global tile Y index at zoom ``tile_z``.
            tile_z: Zoom level of the tile.
            extent: Resolution of the local tile coordinate grid. Defaults
                to 4096 (typical for Mapbox Vector Tiles).

        Returns:
            A tuple ``(lon, lat)`` where:
            - ``lon`` is longitude in degrees (WGS84).
            - ``lat`` is latitude in degrees (WGS84).
        """
        n = 2.0 ** tile_z
        
        # Calculate global X/Y in Tile Space:
        x_val = tile_x + (shape_x / extent)
        y_val = tile_y + (shape_y / extent)
        
        # Calculate the longitude:
        lon = (x_val / n) * 360.0 - 180.0
        
        # Latitude involves the inverse Mercator projection:
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y_val / n)))
        lat = math.degrees(lat_rad)
        
        return lon, lat

    @staticmethod
    def ms_to_date_utc(timestamp_ms: int) -> str:
        """Convert a millisecond UNIX timestamp to a UTC date string.
    
        The input timestamp is interpreted as milliseconds since the Unix epoch
        (1970-01-01T00:00:00Z). The function converts it to a UTC date and
        returns an ISO 8601 formatted string (YYYY-MM-DD).
    
        Args:
            timestamp_ms: Timestamp in milliseconds since Unix epoch (UTC).
    
        Returns:
            A Unicode string representing the UTC date in ISO format 
            (YYYY-MM-DD).
        """
        seconds = timestamp_ms / 1000.0
        dt_object = datetime.fromtimestamp(seconds, tz=timezone.utc)
        return dt_object.date().isoformat()

    @staticmethod
    def get_enclosing_zoom(bbox: "BoundingBox") -> int:
        """
        Find the highest zoom where the bounding box fits inside one tile.

        This method searches zoom levels from 22 down to 14 and returns the
        maximum zoom level at which the entire bounding box is contained within
        a single XYZ tile. If the bounding box spans multiple tiles at all
        zooms in that range, 14 is returned.

        The bounding box is expected to provide Shapely-like bounds in
        the order ``(min_lon, min_lat, max_lon, max_lat)``.

        Args:
            bbox: Bounding box object with a ``shapely`` geometry exposing a
                ``.bounds`` attribute.

        Returns:
            The highest integer zoom level in the range [14, 22] at which
            the bounding box fits into a single tile. Returns 14 if no higher
            zoom level satisfies this condition.
        """
        min_lon, min_lat, max_lon, max_lat = bbox.bounds

        # Iterate from high zoom (22) down to 14 (inclusive)
        for z in range(22, 13, -1):
            # Calculate tile coords for Top-Left (North-West):
            x1, y1 = TileUtils.latlon_to_tile(max_lat, min_lon, z)
            
            # Calculate tile coords for Bottom-Right (South-East):
            x2, y2 = TileUtils.latlon_to_tile(min_lat, max_lon, z)
            
            # Check if the integer parts are the same, i.e., they are in the
            # same tile:
            if int(x1) == int(x2) and int(y1) == int(y2):
                return z
                
        # If it crosses a tile boundary even at zoom 14, return the lowest 
        # allowed zoom.
        return 14

    @staticmethod
    def bbox_to_mapbox_tiles(
        bbox: "BoundingBox",
        zoom: int | None = None,
    ) -> list[tuple[int, int, int]]:
        """
        Compute XYZ tiles that cover a bounding box.

        The bounding box is given in WGS84 coordinates via the
        ``bbox.bounds`` attribute. Tiles are computed using the
        standard Web Mercator XYZ scheme.

        If ``zoom`` is not provided, the highest zoom at which the entire
        bounding box fits into a single tile (as returned by
        :meth:`get_enclosing_zoom`) is used. In that case the result is often
        a single tile.

        Args:
            bbox: Bounding box object with a ``shapely`` geometry exposing a
                ``.bounds`` attribute in the order
                ``(min_lon, min_lat, max_lon, max_lat)``.
            zoom: Zoom level at which to compute coverage. If ``None``, the
                zoom level is determined by :meth:`get_enclosing_zoom`.

        Returns:
            A list of tile coordinates ``(x, y, z)`` covering the bounding box,
            including all integer tiles from ``min_x..max_x`` and
            ``min_y..max_y`` inclusive.
        """
        # Extract the WGS84 bounds from the Shapely geometry:
        # min/max longitude (x), min/max latitude (y)
        min_lon, min_lat, max_lon, max_lat = bbox.bounds

        # If no zoom is provided, pick the highest zoom at which the entire
        # bbox fits into a single tile. This usually yields 1 tile:
        if zoom is None:
            zoom = TileUtils.get_enclosing_zoom(bbox)

        # Convert the north-west (max_lat, min_lon) and south-east
        # (min_lat, max_lon) corners of the bbox into fractional tile
        # coordinates at the chosen zoom:
        min_x_float, min_y_float = TileUtils.latlon_to_tile(max_lat, min_lon, zoom)
        max_x_float, max_y_float = TileUtils.latlon_to_tile(min_lat, max_lon, zoom)

        # Take the integer parts to get the tile index range that covers the 
        # bbox:
        min_x, min_y = int(min_x_float), int(min_y_float)
        max_x, max_y = int(max_x_float), int(max_y_float)

        # Build the full list of tiles in the inclusive [min_x..max_x] x
        # [min_y..max_y] range:
        tiles: list[tuple[int, int, int]] = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                tiles.append((x, y, zoom))

        return tiles