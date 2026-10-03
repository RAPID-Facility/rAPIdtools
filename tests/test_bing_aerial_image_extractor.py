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

import json
import re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import pytest
import requests
from PIL import Image

from rapidtools.data_sources import BingAerialImageExtractor

TILE_URL = re.compile(r'https?://ecn\.t3\.tiles\.virtualearth\.net/tiles/a.*')

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

# ==========================================
# Helpers
# ==========================================


def _jpeg_bytes(color=(120, 130, 140)):
    """Return an in-memory 256x256 JPEG tile."""
    buf = BytesIO()
    Image.new('RGB', (256, 256), color).save(buf, format='JPEG')
    return buf.getvalue()


# ==========================================
# 1. Projection helpers
# ==========================================


def test_lat_lon_to_pixel_known_values():
    """The origin maps to the center of the map at every zoom."""
    assert BingAerialImageExtractor.lat_lon_to_pixel(0.0, 0.0, 1) == (256, 256)
    assert BingAerialImageExtractor.lat_lon_to_pixel(0.0, 0.0, 10) == (
        131072,
        131072,
    )
    # West/north edges are pixel zero:
    px, py = BingAerialImageExtractor.lat_lon_to_pixel(85.0, -180.0, 5)
    assert px == 0
    assert py >= 0


def test_lat_lon_to_pixel_clamps_polar_latitudes():
    """Latitudes at the poles are clamped so the projection stays finite."""
    _, y_pole = BingAerialImageExtractor.lat_lon_to_pixel(90.0, 0.0, 3)
    _, y_south = BingAerialImageExtractor.lat_lon_to_pixel(-90.0, 0.0, 3)
    _, y_limit = BingAerialImageExtractor.lat_lon_to_pixel(89.5, 0.0, 3)
    assert isinstance(y_pole, int) and isinstance(y_south, int)
    # The poles clamp onto the first and last pixel rows of the map, never
    # outside the grid:
    assert y_pole == 0
    assert y_south == (256 << 3) - 1
    # Anything beyond the clamp collapses onto the same pixel row:
    assert y_pole == y_limit


@pytest.mark.parametrize(
    'lat,lon',
    [(34.05, -118.25), (47.6062, -122.3321), (-33.8688, 151.2093), (0.0, 0.0)],
)
def test_pixel_lat_lon_round_trip(lat, lon):
    """Converting to pixels and back reproduces the input within a pixel."""
    zoom = 20
    px, py = BingAerialImageExtractor.lat_lon_to_pixel(lat, lon, zoom)
    out_lat, out_lon = BingAerialImageExtractor.pixel_to_lat_lon(px, py, zoom)
    deg_per_px = 360.0 / (256 << zoom)
    assert out_lon == pytest.approx(lon, abs=deg_per_px)
    assert out_lat == pytest.approx(lat, abs=deg_per_px * 2)


def test_pixel_to_lat_lon_center():
    """The center pixel of the global map is (0, 0)."""
    lat, lon = BingAerialImageExtractor.pixel_to_lat_lon(256, 256, 1)
    assert lat == pytest.approx(0.0)
    assert lon == pytest.approx(0.0)


def test_tile_to_quadkey_known_values():
    """Quadkeys match the documented Bing Maps tile system."""
    qk = BingAerialImageExtractor.tile_to_quadkey
    assert qk(0, 0, 1) == '0'
    assert qk(1, 0, 1) == '1'
    assert qk(0, 1, 1) == '2'
    assert qk(1, 1, 1) == '3'
    assert qk(3, 5, 3) == '213'
    assert len(qk(12345, 6789, 20)) == 20
    assert set(qk(2**19 - 1, 0, 19)) == {'1'}


# ==========================================
# 2. Context-manager lifecycle
# ==========================================


def test_init_defaults(tmp_path):
    """Constructor stores configuration without opening anything."""
    extractor = BingAerialImageExtractor(tmp_path / 'out', zoom_level=18)
    assert extractor.output_dir == (tmp_path / 'out').resolve()
    assert extractor.zoom_level == 18
    assert extractor.max_workers == 10
    assert extractor._session is None
    assert extractor._executor is None
    assert not extractor.output_dir.exists()


def test_context_manager_opens_and_closes_resources(tmp_path):
    """Entering creates the output directory, session and executor."""
    extractor = BingAerialImageExtractor(tmp_path / 'out', max_workers=3)
    with extractor as opened:
        assert opened is extractor
        assert extractor.output_dir.is_dir()
        assert isinstance(extractor._session, requests.Session)
        assert isinstance(extractor._executor, ThreadPoolExecutor)
        adapter = extractor._session.adapters['http://']
        assert adapter.max_retries.total == 5
        assert adapter._pool_connections == 3
        assert adapter._pool_maxsize == 6
    assert extractor._session is None
    assert extractor._executor is None


def test_exit_without_enter_is_safe(tmp_path):
    """Calling __exit__ on a never-opened extractor does nothing."""
    extractor = BingAerialImageExtractor(tmp_path)
    extractor.__exit__(None, None, None)
    assert extractor._session is None


def test_methods_raise_outside_context(tmp_path):
    """All network-facing methods require the context manager."""
    extractor = BingAerialImageExtractor(tmp_path)
    with pytest.raises(RuntimeError, match='with'):
        extractor._download_tile('0')
    with pytest.raises(RuntimeError, match='context manager'):
        next(extractor.generate_region_tiles(34.05, -118.25, 34.051, -118.249))
    with pytest.raises(RuntimeError, match='context manager'):
        next(extractor.generate_polygon_tiles(FEATURE))
    with pytest.raises(RuntimeError, match='context manager'):
        extractor.process_polygon(FEATURE, 0)
    with pytest.raises(RuntimeError, match='context manager'):
        extractor.process_geojson(tmp_path / 'missing.geojson')


# ==========================================
# 3. Tile downloads (mocked HTTP)
# ==========================================


def test_download_tile_success_and_failures(tmp_path, requests_mock):
    """A 200 JPEG is decoded; 404s and connection errors give None."""
    with BingAerialImageExtractor(tmp_path) as extractor:
        requests_mock.get(TILE_URL, content=_jpeg_bytes())
        tile = extractor._download_tile('0231010')
        assert isinstance(tile, Image.Image)
        assert tile.size == (256, 256)
        assert requests_mock.last_request.url.startswith(
            'http://ecn.t3.tiles.virtualearth.net/tiles/a0231010.jpeg'
        )

        requests_mock.get(TILE_URL, status_code=404)
        assert extractor._download_tile('0231010') is None

        requests_mock.get(TILE_URL, exc=requests.exceptions.ConnectionError)
        assert extractor._download_tile('0231010') is None

        # Undecodable payloads are also swallowed:
        requests_mock.get(TILE_URL, content=b'not a jpeg')
        assert extractor._download_tile('0231010') is None


def test_generate_region_tiles(tmp_path, requests_mock):
    """Tiles covering a bounding box are yielded with geographic bounds."""
    requests_mock.get(TILE_URL, content=_jpeg_bytes())
    min_lat, min_lon, max_lat, max_lon = 34.050, -118.251, 34.051, -118.250
    with BingAerialImageExtractor(tmp_path, zoom_level=19) as extractor:
        tiles = list(
            extractor.generate_region_tiles(min_lat, min_lon, max_lat, max_lon)
        )
    assert tiles
    assert len(tiles) == requests_mock.call_count
    quadkeys = set()
    for img, bbox, quadkey in tiles:
        assert img.size == (256, 256)
        assert len(quadkey) == 19
        quadkeys.add(quadkey)
        t_min_lon, t_min_lat, t_max_lon, t_max_lat = bbox
        assert t_min_lon < t_max_lon
        assert t_min_lat < t_max_lat
        # Every tile overlaps the requested region:
        assert t_max_lon > min_lon and t_min_lon < max_lon
        assert t_max_lat > min_lat and t_min_lat < max_lat
    assert len(quadkeys) == len(tiles)


def test_generate_region_tiles_skips_failed_downloads(tmp_path, requests_mock):
    """Tiles whose download fails are silently omitted."""
    requests_mock.get(TILE_URL, status_code=500)
    with BingAerialImageExtractor(tmp_path, zoom_level=19) as extractor:
        tiles = list(
            extractor.generate_region_tiles(34.050, -118.251, 34.051, -118.250)
        )
    assert tiles == []
    assert requests_mock.call_count > 0


def test_generate_polygon_tiles(tmp_path, requests_mock):
    """Only tiles intersecting the polygon are downloaded and yielded."""
    requests_mock.get(TILE_URL, content=_jpeg_bytes())
    triangle = {
        'type': 'Feature',
        'properties': {},
        'geometry': {
            'type': 'Polygon',
            'coordinates': [
                [
                    [-118.2500, 34.0500],
                    [-118.2500, 34.0520],
                    [-118.2520, 34.0500],
                    [-118.2500, 34.0500],
                ]
            ],
        },
    }
    with BingAerialImageExtractor(tmp_path, zoom_level=19) as extractor:
        tiles = list(extractor.generate_polygon_tiles(triangle))
        region = list(
            extractor.generate_region_tiles(34.0500, -118.2520, 34.0520, -118.2500)
        )
    assert tiles
    # A triangle covers roughly half of its bounding box, so fewer tiles
    # intersect it than cover the full bbox:
    assert len(tiles) < len(region)
    total_area = 0.0
    for img, bbox, quadkey, area in tiles:
        assert img.size == (256, 256)
        assert len(quadkey) == 19
        assert bbox[0] < bbox[2] and bbox[1] < bbox[3]
        assert area > 0
        total_area += area
    # Intersection areas sum to the polygon area (0.5 * base * height):
    assert total_area == pytest.approx(0.5 * 0.002 * 0.002, rel=1e-6)


def test_generate_polygon_tiles_skips_failed_downloads(tmp_path, requests_mock):
    """Intersecting tiles that fail to download are dropped."""
    requests_mock.get(TILE_URL, status_code=404)
    with BingAerialImageExtractor(tmp_path, zoom_level=19) as extractor:
        assert list(extractor.generate_polygon_tiles(FEATURE)) == []


# ==========================================
# 4. Stitching polygons and GeoJSON files
# ==========================================


def test_process_polygon_success_writes_jpeg(tmp_path, requests_mock):
    """Tiles are stitched, cropped to the buffered bbox and saved."""
    requests_mock.get(TILE_URL, content=_jpeg_bytes((200, 50, 50)))
    with BingAerialImageExtractor(tmp_path / 'out', zoom_level=19) as extractor:
        message = extractor.process_polygon(FEATURE, 0, buffer_percent=0.0)
    assert message == 'Success: test_building'
    out = tmp_path / 'out' / 'test_building.jpg'
    assert out.is_file()
    with Image.open(out) as img:
        px_min_x, px_max_y = BingAerialImageExtractor.lat_lon_to_pixel(
            34.050, -118.251, 19
        )
        px_max_x, px_min_y = BingAerialImageExtractor.lat_lon_to_pixel(
            34.051, -118.250, 19
        )
        assert img.size == (px_max_x - px_min_x, px_max_y - px_min_y)
        r, g, b = img.convert('RGB').getpixel((img.width // 2, img.height // 2))
        assert r > 150 and g < 100 and b < 100


def test_process_polygon_default_id_and_missing_tiles(tmp_path, requests_mock):
    """Features without an id are named by index; failed tiles stay black."""
    requests_mock.get(TILE_URL, status_code=404)
    feature = {'type': 'Feature', 'geometry': FEATURE['geometry']}
    with BingAerialImageExtractor(tmp_path, zoom_level=19) as extractor:
        message = extractor.process_polygon(feature, 7)
    assert message == 'Success: polygon_7'
    out = tmp_path / 'polygon_7.jpg'
    assert out.is_file()
    with Image.open(out) as img:
        assert img.convert('RGB').getpixel((0, 0)) == (0, 0, 0)


def test_process_polygon_reports_failure(tmp_path, requests_mock):
    """Exceptions during processing are returned as a failure message."""
    requests_mock.get(TILE_URL, content=_jpeg_bytes())
    with BingAerialImageExtractor(tmp_path) as extractor:
        message = extractor.process_polygon({'type': 'Feature'}, 3)
    assert message.startswith('Failed processing polygon 3:')
    assert 'geometry' in message


def test_process_geojson(tmp_path, requests_mock, caplog):
    """All features in a GeoJSON file are processed and reported."""
    requests_mock.get(TILE_URL, content=_jpeg_bytes())
    second = {
        'type': 'Feature',
        'properties': {'name': 'no id here'},
        'geometry': FEATURE['geometry'],
    }
    geojson = tmp_path / 'buildings.geojson'
    geojson.write_text(
        json.dumps({'type': 'FeatureCollection', 'features': [FEATURE, second]})
    )
    out_dir = tmp_path / 'stitched'
    with BingAerialImageExtractor(out_dir, zoom_level=19) as extractor:
        with caplog.at_level('INFO'):
            results = extractor.process_geojson(geojson, buffer_percent=0.05)

    assert 'loaded 2 polygons' in caplog.text
    assert 'zoom 19' in caplog.text
    assert '5% buffer' in caplog.text
    assert results == ['Success: test_building', 'Success: polygon_1']
    assert (out_dir / 'test_building.jpg').is_file()
    assert (out_dir / 'polygon_1.jpg').is_file()


def test_process_geojson_without_features(tmp_path, caplog):
    """A GeoJSON file without a features array processes nothing."""
    geojson = tmp_path / 'empty.geojson'
    geojson.write_text(json.dumps({'type': 'FeatureCollection'}))
    with BingAerialImageExtractor(tmp_path) as extractor:
        with caplog.at_level('INFO'):
            assert extractor.process_geojson(geojson) == []
    assert 'loaded 0 polygons' in caplog.text
