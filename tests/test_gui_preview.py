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

import io
from datetime import datetime

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

from rapidtools.core import PhysicalAsset, PhysicalAssetCollection
from rapidtools.gui.preview import (
    RasterPreview,
    RasterTiler,
    _json_safe,
    project_collection,
    render_preview,
)

# A tiny WGS84 raster: 10x10 pixels of 0.01 degrees starting at (-118.1, 34.2).
LON0, LAT0, PIX = -118.1, 34.2, 0.01
SIZE = 10


def _write_raster(path, data, crs='EPSG:4326', nodata=None):
    """Write a (bands, rows, cols) uint8 array as a GeoTIFF and return the path."""
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        width=data.shape[2],
        height=data.shape[1],
        count=data.shape[0],
        dtype='uint8',
        crs=crs,
        transform=from_origin(LON0, LAT0, PIX, PIX),
        nodata=nodata,
    ) as dst:
        dst.write(data)
    return path


def _pixel_box(col, row):
    """Return a WGS84 box covering exactly pixel (col, row) of the test raster."""
    return box(
        LON0 + col * PIX,
        LAT0 - (row + 1) * PIX,
        LON0 + (col + 1) * PIX,
        LAT0 - row * PIX,
    )


@pytest.fixture
def single_band_raster(tmp_path):
    """A one-band raster with a nodata hole in its top-left corner."""
    data = np.full((1, SIZE, SIZE), 150, dtype=np.uint8)
    data[0, :2, :2] = 0
    return _write_raster(tmp_path / 'grey.tif', data, nodata=0)


@pytest.fixture
def crs_less_raster(tmp_path):
    """A three-band raster with no CRS at all."""
    data = np.full((3, SIZE, SIZE), 90, dtype=np.uint8)
    return _write_raster(tmp_path / 'nocrs.tif', data, crs=None)


@pytest.fixture
def wgs84_preview(tmp_path):
    """A full-resolution preview of a plain RGB WGS84 raster."""
    data = np.full((3, SIZE, SIZE), 120, dtype=np.uint8)
    return render_preview(_write_raster(tmp_path / 'rgb.tif', data), max_px=SIZE)


# ==========================================
# 1. render_preview
# ==========================================

def test_render_preview_single_band_uses_nodata_mask(single_band_raster):
    """A one-band raster is expanded to RGB and nodata pixels become transparent."""
    preview = render_preview(single_band_raster, max_px=SIZE)
    image = Image.open(io.BytesIO(preview.png))
    assert image.mode == 'RGBA'
    assert image.getpixel((0, 0))[3] == 0  # nodata hole is transparent
    assert image.getpixel((5, 5))[:3] == (150, 150, 150)
    assert image.getpixel((5, 5))[3] == 255
    assert preview.scale == (1.0, 1.0)


def test_render_preview_without_crs_keeps_native_bounds(crs_less_raster):
    """A raster without a CRS is previewed with its bounds taken as WGS84."""
    preview = render_preview(crs_less_raster, max_px=SIZE)
    assert preview.crs_wkt == ''
    assert preview.wgs84_bounds == pytest.approx((LON0, LAT0 - 0.1, LON0 + 0.1, LAT0))
    payload = preview.to_json()
    assert payload['name'] == 'nocrs.tif'
    assert payload['version'] == 1
    assert 'png' not in payload


def test_render_preview_downscales_long_side(tmp_path):
    """The long side is limited to max_px and the short side keeps the aspect."""
    data = np.zeros((3, SIZE, 2 * SIZE), dtype=np.uint8)
    preview = render_preview(_write_raster(tmp_path / 'wide.tif', data), max_px=SIZE)
    assert (preview.preview_width, preview.preview_height) == (SIZE, SIZE // 2)
    assert preview.scale == (0.5, 0.5)


# ==========================================
# 2. project_collection
# ==========================================

def test_project_collection_without_crs_uses_raw_coordinates(crs_less_raster):
    """With no raster CRS, geometries are pushed straight through the transform."""
    preview = render_preview(crs_less_raster, max_px=SIZE)
    collection = PhysicalAssetCollection()
    collection.add(PhysicalAsset(id='p', geometry=_pixel_box(3, 4)))
    data = project_collection(collection, preview)
    xs = [pt[0] for pt in data['features'][0]['rings'][0]]
    ys = [pt[1] for pt in data['features'][0]['rings'][0]]
    assert (min(xs), max(xs)) == pytest.approx((3, 4), abs=0.01)
    assert (min(ys), max(ys)) == pytest.approx((4, 5), abs=0.01)


def test_project_collection_handles_every_geometry_kind(wgs84_preview):
    """Polygons, lines, points and other geometries all produce pixel rings."""
    collection = PhysicalAssetCollection()
    p1, p2 = _pixel_box(1, 1), _pixel_box(5, 5)
    collection.add(PhysicalAsset(id='multi', geometry=MultiPolygon([p1, p2])))
    line = LineString([(LON0, LAT0), (LON0 + 2 * PIX, LAT0 - 2 * PIX)])
    collection.add(PhysicalAsset(id='line', geometry=line))
    collection.add(PhysicalAsset(id='mline', geometry=MultiLineString([line, line])))
    pt = Point(LON0 + 0.5 * PIX, LAT0 - 0.5 * PIX)
    collection.add(PhysicalAsset(id='pt', geometry=pt))
    collection.add(PhysicalAsset(id='mpt', geometry=MultiPoint([pt, pt])))
    collection.add(PhysicalAsset(id='gc', geometry=GeometryCollection([pt, line])))
    collection.add(PhysicalAsset(id='empty', geometry=Polygon()))

    data = project_collection(collection, wgs84_preview)
    by_id = {f['id']: f for f in data['features']}
    assert 'empty' not in by_id and data['count'] == 6
    assert by_id['multi']['kind'] == 'polygon' and len(by_id['multi']['rings']) == 2
    assert by_id['line']['kind'] == 'line' and len(by_id['line']['rings']) == 1
    assert by_id['mline']['kind'] == 'line' and len(by_id['mline']['rings']) == 2
    assert by_id['pt']['kind'] == 'point'
    assert by_id['pt']['rings'] == [[[0.5, 0.5]]]
    assert by_id['mpt']['kind'] == 'point' and len(by_id['mpt']['rings']) == 2
    # Other geometry types fall back to their envelope (a closed 5-point ring):
    assert by_id['gc']['kind'] == 'polygon' and len(by_id['gc']['rings'][0]) == 5
    assert data['preview_version'] == wgs84_preview.version


def test_project_collection_serialises_attributes(wgs84_preview):
    """Attribute values are coerced into JSON-safe primitives, recursively."""
    when = datetime(2026, 9, 22, 12, 0)
    collection = PhysicalAssetCollection()
    collection.add(
        PhysicalAsset(
            id='a',
            geometry=_pixel_box(0, 0),
            attributes={
                'tags': ['x', 1, None],
                'meta': {'nested': (1, 2.5), 3: True},
                'when': when,
            },
        )
    )
    data = project_collection(collection, wgs84_preview)
    attributes = data['features'][0]['attributes']
    assert attributes == {
        'tags': ['x', 1, None],
        'meta': {'nested': [1, 2.5], '3': True},
        'when': str(when),
    }
    assert data['attribute_keys'] == ['meta', 'tags', 'when']
    assert data['features'][0]['image_count'] == 0


def test_json_safe_primitives():
    """_json_safe passes primitives through and stringifies everything else."""
    assert _json_safe(None) is None
    assert _json_safe(True) is True
    assert _json_safe(2.5) == 2.5
    assert _json_safe('s') == 's'
    assert _json_safe([1, (2, 3)]) == [1, [2, 3]]
    assert _json_safe({1: {'a': Point(0, 0)}}) == {'1': {'a': 'POINT (0 0)'}}


# ==========================================
# 3. RasterTiler
# ==========================================

def test_raster_tiler_single_band_and_bounds(single_band_raster):
    """Single-band tiles are expanded to RGB; invalid addresses return None."""
    tiler = RasterTiler(single_band_raster)
    data = tiler.tile(1, 0, 0)
    image = Image.open(io.BytesIO(data)).convert('RGB')
    assert image.size == (SIZE, SIZE)
    assert image.getpixel((5, 5)) == pytest.approx((150, 150, 150), abs=6)  # JPEG
    assert tiler.tile(1, -1, 0) is None
    assert tiler.tile(1, 0, -1) is None
    assert tiler.tile(1, 0, 1) is None  # off the bottom of the raster
    assert tiler.tile(1, 0, 0) is data  # served from the cache


def test_raster_preview_dataclass_scale():
    """RasterPreview.scale divides preview size by native size."""
    preview = RasterPreview(
        path=__import__('pathlib').Path('x.tif'),
        width=200,
        height=100,
        preview_width=50,
        preview_height=25,
        crs_wkt='',
        transform=(1, 0, 0, 0, -1, 0),
        wgs84_bounds=(0, 0, 1, 1),
        png=b'',
    )
    assert preview.scale == (0.25, 0.25)
    assert preview.to_json()['wgs84_bounds'] == [0, 0, 1, 1]
