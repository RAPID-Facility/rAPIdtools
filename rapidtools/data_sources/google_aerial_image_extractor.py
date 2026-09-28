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
Keyless extraction of Google Maps satellite imagery tiles.

Google serves its satellite (``lyrs=s``) and hybrid (``lyrs=y``) layers as
256 px tiles from ``mt0``-``mt3.google.com`` in the standard XYZ scheme; no
API key is required. This mirrors the approach used by BRAILS++'s
``GoogleSatellite`` scraper while exposing the same interface as
:class:`~rapidtools.data_sources.BingAerialImageExtractor`.

Example:
    >>> from rapidtools.data_sources import GoogleAerialImageExtractor
    >>> with GoogleAerialImageExtractor('google_crops', zoom_level=20) as extractor:
    ...     extractor.process_geojson(
    ...         'buildings.geojson', pad_to_square=True, resize_to=(640, 640)
    ...     )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .tile_imagery_base import WebMercatorTileExtractor

GOOGLE_TILE_URL = 'https://mt{subdomain}.google.com/vt/lyrs={layer}&x={x}&y={y}&z={z}'

# Layers served by the Google tile endpoint that contain aerial imagery.
GOOGLE_TILE_LAYERS = {
    's': 'satellite',
    'y': 'hybrid (satellite + labels)',
}


class GoogleAerialImageExtractor(WebMercatorTileExtractor):
    """
    Client for extracting satellite imagery from Google Maps tiles.

    Args:
        save_directory: Directory where stitched images will be saved.
        zoom_level: Map detail level (1-21). Defaults to 20 (~0.15 m/px).
        max_workers: Number of concurrent download threads.
        layer: ``'s'`` for satellite imagery (default) or ``'y'`` for the
            hybrid layer with road labels.

    Example:
        >>> from rapidtools.data_sources import GoogleAerialImageExtractor
        >>> with GoogleAerialImageExtractor('out', zoom_level=20) as extractor:
        ...     for tile, bbox, (x, y, z) in extractor.generate_polygon_tiles(feature):
        ...         print(x, y, z, tile.size)
    """

    PROVIDER_NAME = 'Google Maps satellite'

    def __init__(
        self,
        save_directory: str | Path = 'output',
        zoom_level: int = 20,
        max_workers: int = 10,
        layer: str = 's',
        **kwargs: Any,
    ):
        """
        Initialize the extractor configuration.

        Args:
            save_directory: Directory where stitched images will be saved
                (``output_dir`` is a deprecated alias).
            zoom_level: Map detail level. Defaults to 20.
            max_workers: Number of concurrent download threads.
            layer: Tile layer code (``'s'`` satellite, ``'y'`` hybrid).

        Raises:
            ValueError: If ``layer`` is not a supported imagery layer.
        """
        super().__init__(
            save_directory=save_directory,
            zoom_level=zoom_level,
            max_workers=max_workers,
            **kwargs,
        )
        if layer not in GOOGLE_TILE_LAYERS:
            raise ValueError(
                f'Unsupported Google tile layer {layer!r}; choose one of '
                f'{sorted(GOOGLE_TILE_LAYERS)}.'
            )
        self.layer = layer

    def _tile_url(self, tile_key: tuple[int, int, int]) -> str:
        """
        Return the tile URL for an ``(x, y, zoom)`` key.

        Requests are spread over the four ``mt0``-``mt3`` subdomains.
        """
        x, y, z = tile_key
        return GOOGLE_TILE_URL.format(
            subdomain=(x + y) % 4, layer=self.layer, x=x, y=y, z=z
        )
