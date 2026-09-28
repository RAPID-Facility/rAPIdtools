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

import math

import pytest

from rapidtools.core import BoundingBox
from rapidtools.data_sources import TileUtils

# ==========================================
# 1. latlon_to_tile
# ==========================================


def test_latlon_to_tile_origin_and_zoom_zero():
    """The equator/prime meridian sits at the center of the zoom-0 tile."""
    assert TileUtils.latlon_to_tile(0.0, 0.0, 0) == (0.5, 0.5)
    assert TileUtils.latlon_to_tile(0.0, 0.0, 1) == (1.0, 1.0)


def test_latlon_to_tile_corners():
    """Web Mercator extremes map to the tile grid corners."""
    max_lat = 85.05112878
    x, y = TileUtils.latlon_to_tile(max_lat, -180.0, 3)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0, abs=1e-6)
    x, y = TileUtils.latlon_to_tile(-max_lat, 180.0, 3)
    assert x == pytest.approx(8.0)
    assert y == pytest.approx(8.0, abs=1e-6)


def test_latlon_to_tile_known_los_angeles():
    """Downtown Los Angeles falls into the expected zoom-15 tile."""
    x, y = TileUtils.latlon_to_tile(34.05, -118.25, 15)
    assert (int(x), int(y)) == (5620, 13084)


@pytest.mark.parametrize('zoom', [1, 5, 12, 18, 22])
def test_latlon_to_tile_round_trip_via_mvt(zoom):
    """latlon_to_tile and mvt_to_wgs84 are mutual inverses."""
    lat, lon = 47.6062, -122.3321  # Seattle
    xf, yf = TileUtils.latlon_to_tile(lat, lon, zoom)
    tx, ty = int(xf), int(yf)
    extent = 4096
    shape_x = round((xf - tx) * extent)
    shape_y = round((yf - ty) * extent)
    out_lon, out_lat = TileUtils.mvt_to_wgs84(shape_x, shape_y, tx, ty, zoom)
    # Precision is limited by the 4096 grid within one tile:
    tile_deg = 360.0 / (2**zoom)
    tol = tile_deg / extent
    assert out_lon == pytest.approx(lon, abs=tol)
    assert out_lat == pytest.approx(lat, abs=tol)


# ==========================================
# 2. mvt_to_wgs84
# ==========================================


def test_mvt_to_wgs84_tile_corners():
    """Local (0, 0) and (extent, extent) map to the tile's NW and SE corners."""
    lon, lat = TileUtils.mvt_to_wgs84(0, 0, 0, 0, 0)
    assert lon == -180.0
    assert lat == pytest.approx(85.05112878, abs=1e-6)

    lon, lat = TileUtils.mvt_to_wgs84(4096, 4096, 0, 0, 0)
    assert lon == 180.0
    assert lat == pytest.approx(-85.05112878, abs=1e-6)

    # Center of the single zoom-0 tile is (0, 0):
    lon, lat = TileUtils.mvt_to_wgs84(2048, 2048, 0, 0, 0)
    assert lon == pytest.approx(0.0)
    assert lat == pytest.approx(0.0)


def test_mvt_to_wgs84_custom_extent():
    """A non-default extent scales local coordinates accordingly."""
    lon_a, lat_a = TileUtils.mvt_to_wgs84(256, 256, 1, 1, 1, extent=512)
    lon_b, lat_b = TileUtils.mvt_to_wgs84(2048, 2048, 1, 1, 1)
    assert lon_a == pytest.approx(lon_b)
    assert lat_a == pytest.approx(lat_b)
    assert lon_a == pytest.approx(90.0)


# ==========================================
# 3. ms_to_date_utc
# ==========================================


def test_ms_to_date_utc_epoch_and_known_dates():
    """Millisecond timestamps convert to ISO dates in UTC."""
    assert TileUtils.ms_to_date_utc(0) == '1970-01-01'
    assert TileUtils.ms_to_date_utc(1_700_000_000_000) == '2023-11-14'
    # One millisecond before midnight is still the previous day in UTC:
    assert TileUtils.ms_to_date_utc(86_400_000 - 1) == '1970-01-01'
    assert TileUtils.ms_to_date_utc(86_400_000) == '1970-01-02'


# ==========================================
# 4. get_enclosing_zoom
# ==========================================


def test_get_enclosing_zoom_small_box_is_high_zoom():
    """A very small bounding box fits into a single tile at a high zoom."""
    bbox = BoundingBox(-118.25001, 34.05001, -118.25, 34.05002)
    zoom = TileUtils.get_enclosing_zoom(bbox)
    assert 14 <= zoom <= 22
    assert zoom >= 20
    # At that zoom both corners share the same tile:
    x1, y1 = TileUtils.latlon_to_tile(34.05002, -118.25001, zoom)
    x2, y2 = TileUtils.latlon_to_tile(34.05001, -118.25, zoom)
    assert (int(x1), int(y1)) == (int(x2), int(y2))
    # ...and at zoom + 1 they do not (otherwise zoom + 1 would be returned):
    if zoom < 22:
        x1, y1 = TileUtils.latlon_to_tile(34.05002, -118.25001, zoom + 1)
        x2, y2 = TileUtils.latlon_to_tile(34.05001, -118.25, zoom + 1)
        assert (int(x1), int(y1)) != (int(x2), int(y2))


def test_get_enclosing_zoom_large_box_floors_at_14():
    """A bounding box spanning many degrees returns the floor zoom of 14."""
    assert TileUtils.get_enclosing_zoom(BoundingBox(-120, 30, -110, 40)) == 14


def test_get_enclosing_zoom_accepts_duck_typed_bbox():
    """Any object exposing a .bounds tuple works."""

    class _Bounds:
        bounds = (-1.0, -1.0, 1.0, 1.0)

    assert TileUtils.get_enclosing_zoom(_Bounds()) == 14


# ==========================================
# 5. bbox_to_mapbox_tiles
# ==========================================


def test_bbox_to_mapbox_tiles_explicit_zoom_covers_quadrants():
    """A box straddling the origin at zoom 1 touches all four tiles."""
    tiles = TileUtils.bbox_to_mapbox_tiles(BoundingBox(-1, -1, 1, 1), zoom=1)
    assert tiles == [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)]


def test_bbox_to_mapbox_tiles_single_tile():
    """A box entirely inside one tile returns exactly that tile."""
    tiles = TileUtils.bbox_to_mapbox_tiles(BoundingBox(10, 10, 20, 20), zoom=2)
    assert tiles == [(2, 1, 2)]


def test_bbox_to_mapbox_tiles_default_zoom_uses_enclosing():
    """Without a zoom, the enclosing zoom is used and yields one tile."""
    bbox = BoundingBox(-118.25001, 34.05001, -118.25, 34.05002)
    zoom = TileUtils.get_enclosing_zoom(bbox)
    tiles = TileUtils.bbox_to_mapbox_tiles(bbox)
    assert len(tiles) == 1
    assert tiles[0][2] == zoom


def test_bbox_to_mapbox_tiles_grid_is_complete():
    """Returned tiles form the full inclusive grid covering the box."""
    bbox = BoundingBox(-118.30, 34.00, -118.20, 34.10)
    zoom = 12
    tiles = TileUtils.bbox_to_mapbox_tiles(bbox, zoom=zoom)
    xs = sorted({x for x, _, _ in tiles})
    ys = sorted({y for _, y, _ in tiles})
    assert len(tiles) == len(xs) * len(ys)
    assert xs == list(range(xs[0], xs[-1] + 1))
    assert ys == list(range(ys[0], ys[-1] + 1))
    assert all(z == zoom for _, _, z in tiles)
    # The NW and SE corners are contained in the first and last tiles:
    nw = TileUtils.latlon_to_tile(34.10, -118.30, zoom)
    se = TileUtils.latlon_to_tile(34.00, -118.20, zoom)
    assert (math.floor(nw[0]), math.floor(nw[1])) == (xs[0], ys[0])
    assert (math.floor(se[0]), math.floor(se[1])) == (xs[-1], ys[-1])
