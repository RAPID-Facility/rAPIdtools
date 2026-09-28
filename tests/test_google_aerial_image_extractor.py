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

"""Tests for the keyless Google satellite tile extractor (mocked HTTP)."""

import json
import re
from io import BytesIO

import pytest
import requests
from PIL import Image
from shapely.geometry import box

from rapidtools.data_sources import (
    BingAerialImageExtractor,
    GoogleAerialImageExtractor,
)
from rapidtools.data_sources.google_aerial_image_extractor import (
    GOOGLE_TILE_LAYERS,
)
from rapidtools.data_sources.tile_imagery_base import (
    TILE_SIZE,
    WebMercatorTileExtractor,
)

TILE_URL = re.compile(r'https://mt[0-3]\.google\.com/vt/lyrs=[sy]&x=\d+&y=\d+&z=\d+')

FEATURE = {
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


def _jpeg_bytes(color=(30, 120, 200)) -> bytes:
    buf = BytesIO()
    Image.new('RGB', (TILE_SIZE, TILE_SIZE), color).save(buf, format='JPEG')
    return buf.getvalue()


# ==========================================
# 1. Construction and URL building
# ==========================================


def test_init_and_layer_validation(tmp_path):
    """Defaults mirror the Bing extractor; unknown layers are rejected."""
    ext = GoogleAerialImageExtractor(tmp_path, zoom_level=18, max_workers=3, layer='y')
    assert ext.output_dir == tmp_path.resolve()
    assert ext.zoom_level == 18 and ext.max_workers == 3 and ext.layer == 'y'
    assert GoogleAerialImageExtractor().layer == 's'
    with pytest.raises(ValueError, match='Unsupported Google tile layer'):
        GoogleAerialImageExtractor(layer='m')
    assert set(GOOGLE_TILE_LAYERS) == {'s', 'y'}


def test_tile_url_rotates_subdomains():
    """Tile keys map to XYZ URLs spread over mt0-mt3."""
    ext = GoogleAerialImageExtractor(zoom_level=5)
    assert ext._tile_key(3, 4) == (3, 4, 5)
    assert ext._tile_url((3, 4, 5)) == 'https://mt3.google.com/vt/lyrs=s&x=3&y=4&z=5'
    assert ext._tile_url((1, 0, 5)).startswith('https://mt1.google.com')
    assert (
        GoogleAerialImageExtractor(layer='y')
        ._tile_url((0, 0, 1))
        .endswith('lyrs=y&x=0&y=0&z=1')
    )


def test_shared_projection_math_matches_bing():
    """Google and Bing share the Web Mercator math from the base class."""
    for lat, lon, zoom in [(0.0, 0.0, 1), (34.05, -118.25, 19), (-45.0, 170.0, 7)]:
        assert GoogleAerialImageExtractor.lat_lon_to_pixel(
            lat, lon, zoom
        ) == BingAerialImageExtractor.lat_lon_to_pixel(lat, lon, zoom)
    px, py = WebMercatorTileExtractor.lat_lon_to_pixel(34.05, -118.25, 12)
    lat, lon = WebMercatorTileExtractor.pixel_to_lat_lon(px, py, 12)
    assert lat == pytest.approx(34.05, abs=1e-3)
    assert lon == pytest.approx(-118.25, abs=1e-3)
    assert WebMercatorTileExtractor.tile_bounds(0, 0, 1) == pytest.approx(
        (-180.0, 0.0, 0.0, 85.0511287798066), abs=1e-6
    )


def test_static_helpers():
    """Padding and buffered bounds behave as documented."""
    padded = WebMercatorTileExtractor.pad_to_square(Image.new('RGB', (40, 20), 'red'))
    assert padded.size == (40, 40)
    assert padded.getpixel((0, 0)) == (0, 0, 0) and padded.getpixel((20, 20)) == (
        255,
        0,
        0,
    )
    tall = WebMercatorTileExtractor.pad_to_square(Image.new('RGB', (20, 40)))
    assert tall.size == (40, 40)
    assert WebMercatorTileExtractor.buffered_bounds(box(0, 0, 10, 10), 0.1) == (
        -1.0,
        -1.0,
        11.0,
        11.0,
    )
    clamped = WebMercatorTileExtractor.buffered_bounds(box(-180, -85, 180, 85), 0.5)
    assert clamped[0] == pytest.approx(-85.05112878)
    assert clamped[1] == -180.0 and clamped[3] == 180.0


# ==========================================
# 2. Lifecycle
# ==========================================


def test_methods_require_context_manager(tmp_path):
    """Every network-facing method refuses to run outside ``with``."""
    ext = GoogleAerialImageExtractor(tmp_path)
    for call in (
        lambda: ext._download_tile((0, 0, 20)),
        lambda: next(ext.generate_region_tiles(34.05, -118.25, 34.051, -118.249)),
        lambda: next(ext.generate_polygon_tiles(FEATURE)),
        lambda: ext.stitch_region(34.05, -118.25, 34.051, -118.249),
        lambda: ext.process_polygon(FEATURE, 0),
        lambda: ext.process_geojson(tmp_path / 'x.geojson'),
    ):
        with pytest.raises(RuntimeError, match='context manager'):
            call()
    with ext:
        assert ext._session is not None and ext._executor is not None
        assert ext.output_dir.is_dir()
    assert ext._session is None and ext._executor is None
    ext.__exit__(None, None, None)  # idempotent


# ==========================================
# 3. Downloads (mocked HTTP)
# ==========================================


def test_download_tile_success_and_failures(tmp_path, requests_mock):
    """Tiles decode on 200 and return None on errors or bad payloads."""
    with GoogleAerialImageExtractor(tmp_path, zoom_level=20) as ext:
        requests_mock.get(TILE_URL, content=_jpeg_bytes())
        tile = ext._download_tile((1, 2, 20))
        assert tile.size == (TILE_SIZE, TILE_SIZE)
        assert requests_mock.last_request.url == (
            'https://mt3.google.com/vt/lyrs=s&x=1&y=2&z=20'
        )
        requests_mock.get(TILE_URL, status_code=404)
        assert ext._download_tile((1, 2, 20)) is None
        requests_mock.get(TILE_URL, content=b'not an image')
        assert ext._download_tile((1, 2, 20)) is None
        requests_mock.get(TILE_URL, exc=requests.ConnectionError('offline'))
        assert ext._download_tile((1, 2, 20)) is None


def test_generate_region_and_polygon_tiles(tmp_path, requests_mock):
    """Generators yield decoded tiles with bounds and XYZ keys."""
    requests_mock.get(TILE_URL, content=_jpeg_bytes())
    with GoogleAerialImageExtractor(tmp_path, zoom_level=19) as ext:
        tiles = list(ext.generate_region_tiles(34.050, -118.251, 34.051, -118.250))
        assert tiles
        for img, bbox, key in tiles:
            assert img.size == (TILE_SIZE, TILE_SIZE) and key[2] == 19
            assert bbox[0] < bbox[2] and bbox[1] < bbox[3]
            assert bbox[2] > -118.251 and bbox[0] < -118.250
        assert len({key for _, _, key in tiles}) == len(tiles)

        poly_tiles = list(ext.generate_polygon_tiles(FEATURE))
        assert poly_tiles and all(area > 0 for *_, area in poly_tiles)
        # A triangle covering only part of the box touches fewer or equal tiles:
        triangle = {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [
                    [[-118.251, 34.050], [-118.250, 34.050], [-118.251, 34.051]]
                    + [[-118.251, 34.050]]
                ],
            },
        }
        assert len(list(ext.generate_polygon_tiles(triangle))) <= len(poly_tiles)

        requests_mock.get(TILE_URL, status_code=500)
        assert list(ext.generate_region_tiles(34.050, -118.251, 34.051, -118.250)) == []
        assert list(ext.generate_polygon_tiles(FEATURE)) == []


def test_stitch_region_and_errors(tmp_path, requests_mock):
    """Stitching crops exactly to the box and rejects empty regions."""
    requests_mock.get(TILE_URL, content=_jpeg_bytes((10, 200, 10)))
    with GoogleAerialImageExtractor(tmp_path, zoom_level=19, max_workers=2) as ext:
        mosaic = ext.stitch_region(34.050, -118.251, 34.051, -118.250)
        px0, py1 = ext.lat_lon_to_pixel(34.050, -118.251, 19)
        px1, py0 = ext.lat_lon_to_pixel(34.051, -118.250, 19)
        assert mosaic.size == (px1 - px0, py1 - py0)
        assert mosaic.getpixel((0, 0))[1] > 150
        with pytest.raises(ValueError, match='zero pixels'):
            ext.stitch_region(34.05, -118.25, 34.05, -118.25)
        # Failed tiles are left black:
        requests_mock.get(TILE_URL, status_code=404)
        black = ext.stitch_region(34.050, -118.251, 34.051, -118.250)
        assert black.getpixel((0, 0)) == (0, 0, 0)


def test_process_polygon_options(tmp_path, requests_mock):
    """Crops can be padded to a square and resized; failures are reported."""
    requests_mock.get(TILE_URL, content=_jpeg_bytes())
    with GoogleAerialImageExtractor(tmp_path, zoom_level=19) as ext:
        assert ext.process_polygon(FEATURE, 0) == 'Success: test_building'
        plain = Image.open(tmp_path / 'test_building.jpg')
        assert plain.size[0] != plain.size[1]
        assert (
            ext.process_polygon(
                {'type': 'Feature', 'geometry': FEATURE['geometry']},
                4,
                pad_to_square=True,
                resize_to=(640, 640),
            )
            == 'Success: polygon_4'
        )
        assert Image.open(tmp_path / 'polygon_4.jpg').size == (640, 640)
        message = ext.process_polygon({'type': 'Feature'}, 7)
        assert message.startswith('Failed processing polygon 7:')


def test_process_geojson(tmp_path, requests_mock, caplog):
    """Every feature is processed and its status returned and logged."""
    requests_mock.get(TILE_URL, content=_jpeg_bytes())
    geojson = tmp_path / 'b.geojson'
    geojson.write_text(json.dumps({'type': 'FeatureCollection', 'features': [FEATURE]}))
    with GoogleAerialImageExtractor(tmp_path / 'out', zoom_level=19) as ext:
        with caplog.at_level('INFO'):
            results = ext.process_geojson(geojson, resize_to=(128, 128))
    assert results == ['Success: test_building']
    assert 'Google Maps satellite: loaded 1 polygons' in caplog.text
    assert Image.open(tmp_path / 'out' / 'test_building.jpg').size == (128, 128)
