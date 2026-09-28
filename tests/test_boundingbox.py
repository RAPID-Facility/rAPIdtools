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
# 01-28-2026

import json
import logging
import math

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Polygon

from rapidtools.core.bounding_box import BoundingBox

# --- Fixtures ---


@pytest.fixture
def unit_box():
    """Returns a 1x1 box at origin (0,0) to (1,1)."""
    return BoundingBox(0, 0, 1, 1)


@pytest.fixture
def large_box():
    """Returns a 10x10 box at origin (0,0) to (10,10)."""
    return BoundingBox(0, 0, 10, 10)


# --- Initialization & Representation ---


def test_init_basic(unit_box):
    """Test standard initialization."""
    assert unit_box.bounds == (0.0, 0.0, 1.0, 1.0)
    assert unit_box.width == 1.0
    assert unit_box.height == 1.0


def test_init_ordering():
    """Test that Shapely automatically sorts min/max coordinates."""
    # Pass max as min and min as max:
    bbox = BoundingBox(10, 10, 0, 0)
    assert bbox.bounds == (0.0, 0.0, 10.0, 10.0)


def test_repr(unit_box):
    """Test string representation."""
    expected = 'BoundingBox(minx=0.0, miny=0.0, maxx=1.0, maxy=1.0)'
    assert repr(unit_box) == expected


# --- Factory Methods ---


def test_from_geometry_polygon():
    """Test creation from a Shapely Polygon."""
    poly = Polygon([(0, 0), (2, 0), (2, 5), (0, 5)])
    bbox = BoundingBox.from_geometry(poly)
    assert bbox.bounds == (0.0, 0.0, 2.0, 5.0)


def test_from_geometry_line():
    """Test creation from a Shapely LineString."""
    line = LineString([(1, 1), (10, 10)])
    bbox = BoundingBox.from_geometry(line)
    assert bbox.bounds == (1.0, 1.0, 10.0, 10.0)


def test_from_union():
    """Test creating a box that covers two disjoint regions."""
    b1 = BoundingBox(0, 0, 1, 1)
    b2 = BoundingBox(10, 10, 11, 11)

    union_box = BoundingBox.from_union(b1, b2)

    # Union should cover from min of b1 to max of b2:
    assert union_box.bounds == (0.0, 0.0, 11.0, 11.0)
    assert union_box.width == 11.0
    assert union_box.height == 11.0


# --- Geometric Properties (Overridden) ---


def test_area(large_box):
    """Test optimized area calculation."""
    # 10 * 10 = 100:
    assert large_box.area == 100.0


def test_centroid(large_box):
    """Test optimized centroid calculation."""
    # Center of 0,0 -> 10,10 is 5,5:
    assert large_box.centroid == (5.0, 5.0)


def test_get_bounding_box(unit_box):
    """Test that get_bounding_box returns self."""
    assert unit_box.get_bounding_box() is unit_box


# --- Expansion (Buffer) ---


def test_buffer_expand(unit_box):
    """Test expanding the box (positive buffer)."""
    # 0,0,1,1 expanded by 1.0 -> -1,-1, 2, 2:
    expanded = unit_box.buffer(1.0)
    assert expanded.bounds == (-1.0, -1.0, 2.0, 2.0)
    assert isinstance(expanded, BoundingBox)


def test_buffer_shrink(large_box):
    """Test shrinking the box (negative buffer)."""
    # 0,0,10,10 shrunk by 1.0 -> 1,1, 9,9:
    shrunk = large_box.buffer(-1.0)
    assert shrunk.bounds == (1.0, 1.0, 9.0, 9.0)


# --- Split Logic ---


def test_split_quadrants(large_box):
    """Test splitting a box into 4 equal quadrants."""
    quads = large_box.split()

    assert len(quads) == 4

    # Check dimensions of quadrants (should be 5x5):
    for q in quads:
        assert q.width == 5.0
        assert q.height == 5.0
        assert q.area == 25.0

    # Verify specific positions:
    # Bottom-Left
    assert quads[0].bounds == (0.0, 0.0, 5.0, 5.0)
    # Top-Right
    assert quads[3].bounds == (5.0, 5.0, 10.0, 10.0)


# --- Tiling Logic ---


def test_tile_by_area_no_split_needed(large_box):
    """Test that tiling returns self if area is already small enough."""
    # Box area is 100. Max area is 200. No split needed:
    tiles = large_box.tile_by_area(max_area=200)
    assert len(tiles) == 1
    assert tiles[0] is large_box


def test_tile_by_area_exact_split(large_box):
    """Test splitting when area divides perfectly."""
    # Box area 100. Max area 25. Should get exactly 4 tiles (2x2):
    tiles = large_box.tile_by_area(max_area=25)

    assert len(tiles) == 4
    for tile in tiles:
        assert tile.area <= 25.0


def test_tile_by_area_irregular_aspect_ratio():
    """Test tiling on a long, skinny box."""
    # Box: 100 wide, 10 high. Area = 1000. Aspect Ratio = 10.
    # Max area = 10.
    # Min tiles needed mathematically = 100.
    #
    # The algorithm calculates grid dimensions to maintain aspect ratio:
    # n_cols = ceil(sqrt(100 * 10)) = ceil(31.62) = 32
    # n_rows = ceil(sqrt(100 / 10)) = ceil(3.16) = 4
    # Total actual tiles = 32 * 4 = 128
    skinny_box = BoundingBox(0, 0, 100, 10)

    tiles = skinny_box.tile_by_area(max_area=10)

    # Assert exact grid count to verify aspect ratio logic is working:
    assert len(tiles) == 128

    for tile in tiles:
        assert tile.area <= 10.0


def test_tile_by_area_degenerate(caplog):
    """Test tiling a degenerate box (line or point)."""
    # Width is 0:
    degenerate = BoundingBox(0, 0, 0, 10)

    with caplog.at_level(logging.WARNING):
        tiles = degenerate.tile_by_area(max_area=0.1)

    # Should return self and log a warning:
    assert len(tiles) == 1
    assert tiles[0] is degenerate
    assert 'Cannot tile a BoundingBox with 0 width' in caplog.text


def test_tile_rounding_coverage():
    """
    Test that tiling covers the entire area even with tricky float math.
    """
    # A box with weird dimensions:
    bbox = BoundingBox(0.123, 0.456, 9.789, 7.123)
    tiles = bbox.tile_by_area(max_area=1.5)

    # Calculate union of all tiles:
    from shapely.ops import unary_union

    tile_geoms = [t.geometry for t in tiles]
    combined = unary_union(tile_geoms)

    # The combined area should match the original area almost exactly:
    assert math.isclose(combined.area, bbox.area, rel_tol=1e-9)
    # The combined geometry should almost exactly equal the original:
    assert bbox.geometry.difference(combined).area < 1e-9


# --- from_geojson ---


def _write_geojson(tmp_path, payload, name='data.geojson'):
    """Serialize ``payload`` to a GeoJSON file in ``tmp_path`` and return it."""
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


def _feature(geometry, **extra):
    """Build a minimal GeoJSON Feature around ``geometry``."""
    return {'type': 'Feature', 'properties': {}, 'geometry': geometry, **extra}


def test_from_geojson_top_level_bbox_2d(tmp_path):
    """A 4-element top-level bbox is used verbatim without reading geometries."""
    path = _write_geojson(
        tmp_path,
        {'type': 'FeatureCollection', 'bbox': [1, 2, 3, 4], 'features': []},
    )
    assert BoundingBox.from_geojson(path).bounds == (1.0, 2.0, 3.0, 4.0)


def test_from_geojson_top_level_bbox_3d(tmp_path):
    """A 6-element (3D) top-level bbox drops the altitude components."""
    path = _write_geojson(
        tmp_path,
        {
            'type': 'FeatureCollection',
            'bbox': [1, 2, -10, 3, 4, 50],
            'features': [],
        },
    )
    assert BoundingBox.from_geojson(str(path)).bounds == (1.0, 2.0, 3.0, 4.0)


def test_from_geojson_feature_collection_streams_geometries(tmp_path):
    """Bounds are accumulated across every feature geometry in a collection."""
    payload = {
        'type': 'FeatureCollection',
        'features': [
            _feature({'type': 'Point', 'coordinates': [0, 0]}),
            _feature(
                {
                    'type': 'Polygon',
                    'coordinates': [[[2, 2], [5, 2], [5, 7], [2, 7], [2, 2]]],
                }
            ),
            _feature({'type': 'LineString', 'coordinates': [[-1, 3], [1, 9]]}),
        ],
    }
    bbox = BoundingBox.from_geojson(_write_geojson(tmp_path, payload))
    assert bbox.bounds == (-1.0, 0.0, 5.0, 9.0)


def test_from_geojson_single_feature(tmp_path):
    """A top-level Feature is handled like a one-element collection."""
    payload = _feature(
        {'type': 'Polygon', 'coordinates': [[[1, 1], [4, 1], [4, 3], [1, 1]]]}
    )
    bbox = BoundingBox.from_geojson(_write_geojson(tmp_path, payload))
    assert bbox.bounds == (1.0, 1.0, 4.0, 3.0)


def test_from_geojson_bare_geometry(tmp_path):
    """A bare geometry object (no Feature wrapper) is supported."""
    payload = {'type': 'MultiPoint', 'coordinates': [[3, 4], [-2, 8]]}
    bbox = BoundingBox.from_geojson(_write_geojson(tmp_path, payload))
    assert bbox.bounds == (-2.0, 4.0, 3.0, 8.0)


def test_from_geojson_feature_level_bbox_preferred(tmp_path):
    """Feature-level bbox members (2D or 3D) are used instead of the geometry."""
    payload = {
        'type': 'FeatureCollection',
        'features': [
            # The bbox deliberately disagrees with the geometry to prove
            # that the pre-computed bbox wins:
            _feature({'type': 'Point', 'coordinates': [100, 100]}, bbox=[0, 0, 1, 1]),
            _feature(
                {'type': 'Point', 'coordinates': [100, 100]},
                bbox=[5, 5, 0, 6, 6, 0],
            ),
            # A malformed bbox falls back to the geometry:
            _feature({'type': 'Point', 'coordinates': [-3, 2]}, bbox=[1, 2, 3]),
        ],
    }
    bbox = BoundingBox.from_geojson(_write_geojson(tmp_path, payload))
    assert bbox.bounds == (-3.0, 0.0, 6.0, 6.0)


def test_from_geojson_skips_empty_and_null_geometries(tmp_path):
    """Empty geometries and null geometries do not corrupt the bounds."""
    payload = {
        'type': 'FeatureCollection',
        'features': [
            _feature({'type': 'Polygon', 'coordinates': []}),
            _feature(None),
            {'type': 'Feature', 'properties': {}},
            _feature({'type': 'Point', 'coordinates': [1, 2]}),
        ],
    }
    bbox = BoundingBox.from_geojson(_write_geojson(tmp_path, payload))
    assert bbox.bounds == (1.0, 2.0, 1.0, 2.0)


def test_from_geojson_no_valid_geometries_raises(tmp_path):
    """A collection with only empty/null geometries raises ValueError."""
    payload = {
        'type': 'FeatureCollection',
        'features': [
            _feature(None),
            _feature({'type': 'GeometryCollection', 'geometries': []}),
        ],
    }
    with pytest.raises(ValueError, match='No valid or non-empty geometries'):
        BoundingBox.from_geojson(_write_geojson(tmp_path, payload))
    with pytest.raises(ValueError, match='No valid or non-empty geometries'):
        BoundingBox.from_geojson(
            _write_geojson(tmp_path, {'type': 'FeatureCollection'}, 'empty.geojson')
        )


def test_from_geojson_unsupported_type_raises(tmp_path):
    """An unknown top-level type raises ValueError."""
    with pytest.raises(ValueError, match='Unsupported GeoJSON type: Topology'):
        BoundingBox.from_geojson(_write_geojson(tmp_path, {'type': 'Topology'}))
    with pytest.raises(ValueError, match='Unsupported GeoJSON type: None'):
        BoundingBox.from_geojson(_write_geojson(tmp_path, {'bbox': [1, 2]}))


def test_from_geojson_legacy_crs_is_reprojected(tmp_path):
    """A legacy crs member other than EPSG:4326 triggers reprojection to WGS84."""
    payload = {
        'type': 'FeatureCollection',
        'crs': {'type': 'name', 'properties': {'name': 'EPSG:32611'}},
        'features': [
            _feature(
                {
                    'type': 'Polygon',
                    'coordinates': [
                        [
                            [380000, 3780000],
                            [381000, 3780000],
                            [381000, 3781000],
                            [380000, 3781000],
                            [380000, 3780000],
                        ]
                    ],
                }
            )
        ],
    }
    bbox = BoundingBox.from_geojson(_write_geojson(tmp_path, payload))
    min_x, min_y, max_x, max_y = bbox.bounds
    assert -119 < min_x < max_x < -116
    assert 33 < min_y < max_y < 35


def test_from_geojson_wgs84_crs_is_not_reprojected(tmp_path):
    """A crs member naming EPSG:4326 leaves the coordinates untouched."""
    payload = {
        'type': 'FeatureCollection',
        'crs': {
            'type': 'name',
            'properties': {'name': 'urn:ogc:def:crs:EPSG::4326'},
        },
        'features': [_feature({'type': 'Point', 'coordinates': [-118.1, 34.2]})],
    }
    bbox = BoundingBox.from_geojson(_write_geojson(tmp_path, payload))
    assert bbox.bounds == (-118.1, 34.2, -118.1, 34.2)


# --- from_raster ---


def _write_raster(path, crs, transform, size=10):
    """Write a small single-band GeoTIFF with the given CRS and transform."""
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        width=size,
        height=size,
        count=1,
        dtype='uint8',
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(np.zeros((1, size, size), dtype=np.uint8))
    return path


def test_from_raster_wgs84_bounds_used_directly(tmp_path):
    """A raster already in EPSG:4326 yields its native bounds."""
    path = _write_raster(
        tmp_path / 'wgs84.tif', 'EPSG:4326', from_origin(-118.1, 34.2, 0.001, 0.001)
    )
    bbox = BoundingBox.from_raster(path)
    assert bbox.bounds == pytest.approx((-118.1, 34.19, -118.09, 34.2))


def test_from_raster_projected_raster_is_reprojected(tmp_path):
    """A UTM raster is reprojected to WGS84 longitude/latitude."""
    path = _write_raster(
        tmp_path / 'utm.tif', 'EPSG:32611', from_origin(380000, 3780000, 10, 10)
    )
    bbox = BoundingBox.from_raster(str(path))
    min_x, min_y, max_x, max_y = bbox.bounds
    assert -119 < min_x < max_x < -116
    assert 33 < min_y < max_y < 35
    assert bbox.width < 0.01 and bbox.height < 0.01


def test_from_raster_missing_crs_assumed_wgs84(tmp_path):
    """A raster without a CRS is treated as if it were already in WGS84."""
    path = _write_raster(tmp_path / 'nocrs.tif', None, from_origin(0, 10, 1, 1))
    assert BoundingBox.from_raster(path).bounds == (0.0, 0.0, 10.0, 10.0)
