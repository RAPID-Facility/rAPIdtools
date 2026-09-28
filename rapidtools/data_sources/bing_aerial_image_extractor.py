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
Keyless extraction of Bing Maps aerial imagery tiles.

Bing serves its aerial layer as 256 px JPEG tiles addressed by quadkey. This
module wraps that tile server in the shared
:class:`~rapidtools.data_sources.tile_imagery_base.WebMercatorTileExtractor`
interface so regions and GeoJSON polygons can be stitched into images without
a Bing Maps key.

Example:
    >>> from rapidtools.data_sources import BingAerialImageExtractor
    >>> with BingAerialImageExtractor('bing_crops', zoom_level=19) as extractor:
    ...     extractor.process_geojson('buildings.geojson')
"""

from __future__ import annotations

from PIL import Image

from .tile_imagery_base import WebMercatorTileExtractor

BING_TILE_URL = 'http://ecn.t3.tiles.virtualearth.net/tiles/a{quadkey}.jpeg?g=1'


class BingAerialImageExtractor(WebMercatorTileExtractor):
    """
    Client for extracting high-resolution imagery from Bing Maps tiles.

    Use it as a context manager; tiles are downloaded through a pooled,
    retrying session. Supports stitching polygons into images and yielding
    individual tiles for machine-learning pipelines.

    Example:
        >>> from rapidtools.data_sources import BingAerialImageExtractor
        >>> BingAerialImageExtractor.tile_to_quadkey(3, 5, zoom=3)
        '213'
        >>> with BingAerialImageExtractor('out', zoom_level=18) as extractor:
        ...     print(extractor.process_polygon(feature, index=0))
        Success: test_building
    """

    PROVIDER_NAME = 'Bing Maps aerial'

    @staticmethod
    def tile_to_quadkey(tile_x: int, tile_y: int, zoom: int) -> str:
        """
        Convert tile XY coordinates into a Bing Maps quadkey string.

        Args:
            tile_x: Tile column index.
            tile_y: Tile row index.
            zoom: Zoom level (equals the length of the returned quadkey).

        Returns:
            str: The quadkey used to fetch the tile from Bing's servers.

        Example:
            >>> BingAerialImageExtractor.tile_to_quadkey(3, 5, zoom=3)
            '213'
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

    def _tile_key(self, tile_x: int, tile_y: int) -> str:
        """Bing tiles are identified by their quadkey."""
        return self.tile_to_quadkey(tile_x, tile_y, self.zoom_level)

    def _tile_url(self, tile_key: str) -> str:
        """Return the aerial tile URL for a quadkey."""
        return BING_TILE_URL.format(quadkey=tile_key)

    def _download_tile(self, quadkey: str) -> Image.Image | None:
        """
        Download a single Bing tile by quadkey.

        Args:
            quadkey: The Bing Maps quadkey string for the tile.

        Returns:
            Image.Image | None: The tile image, or ``None`` when the request
            failed or returned a non-200 status.

        Raises:
            RuntimeError: If called outside of a context manager block.

        Example:
            >>> with BingAerialImageExtractor() as extractor:
            ...     tile = extractor._download_tile('0231010')
            >>> tile is None or tile.size == (256, 256)
            True
        """
        return super()._download_tile(quadkey)
