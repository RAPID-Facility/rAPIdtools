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

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, box

import rapidtools.data_sources.orthomosaic_reader as ortho_module
from rapidtools.data_sources import OrthomosaicReader
from rapidtools.data_sources.orthomosaic_reader import (
    DEFAULT_PATCH_OVERLAP_RATIO,
    DEFAULT_PATCH_SIZE,
)

# ==========================================
# Helpers and fixtures
# ==========================================


def _write_raster(path, data, crs, transform, nodata=None, overviews=None):
    """Write a (bands, rows, cols) array to a GeoTIFF and return the path."""
    profile = {
        'driver': 'GTiff',
        'width': data.shape[2],
        'height': data.shape[1],
        'count': data.shape[0],
        'dtype': data.dtype.name,
        'crs': crs,
        'transform': transform,
    }
    if nodata is not None:
        profile['nodata'] = nodata
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data)
        if overviews:
            dst.build_overviews(overviews, Resampling.nearest)
    return path


@pytest.fixture
def geo_tif(tmp_path):
    """64x64 3-band uint8 raster in EPSG:4326 with 1e-4 degree pixels."""
    rng = np.random.default_rng(0)
    data = rng.integers(1, 255, (3, 64, 64), dtype=np.uint8)
    transform = from_origin(-118.2, 34.1, 1e-4, 1e-4)
    return _write_raster(tmp_path / 'geo.tif', data, CRS.from_epsg(4326), transform)


@pytest.fixture
def utm_tif(tmp_path):
    """80x80 3-band uint8 raster in EPSG:32611 (meters) with 0.5 m pixels."""
    rng = np.random.default_rng(1)
    data = rng.integers(1, 255, (3, 80, 80), dtype=np.uint8)
    transform = from_origin(390000.0, 3775000.0, 0.5, 0.5)
    return _write_raster(tmp_path / 'utm.tiff', data, CRS.from_epsg(32611), transform)


@pytest.fixture
def feet_tif(tmp_path):
    """
    64x64 4-band uint8 raster in EPSG:2229 (US survey feet) with nodata=0.

    The right half of the raster is entirely nodata so that missing-data
    logic can be exercised deterministically.
    """
    rng = np.random.default_rng(2)
    data = rng.integers(1, 255, (4, 64, 64), dtype=np.uint8)
    data[:, :, 32:] = 0
    transform = from_origin(6480000.0, 1850000.0, 1.0, 1.0)
    return _write_raster(
        tmp_path / 'feet.tif', data, CRS.from_epsg(2229), transform, nodata=0
    )


@pytest.fixture
def float_tif(tmp_path):
    """32x32 single-band float32 raster with values well above 1.0."""
    data = np.linspace(10.0, 1000.0, 32 * 32, dtype=np.float32).reshape(1, 32, 32)
    transform = from_origin(-118.2, 34.1, 1e-4, 1e-4)
    return _write_raster(tmp_path / 'float.tif', data, CRS.from_epsg(4326), transform)


@pytest.fixture
def zero_tif(tmp_path):
    """16x16 raster that contains only zeros (no valid data)."""
    data = np.zeros((3, 16, 16), dtype=np.uint8)
    transform = from_origin(-118.2, 34.1, 1e-4, 1e-4)
    return _write_raster(tmp_path / 'zero.tif', data, CRS.from_epsg(4326), transform)


def _center_lonlat(reader):
    """Return the WGS84 center of an opened reader's extent."""
    min_lon, min_lat, max_lon, max_lat = reader.dataset_extent
    return (min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0


def _square_ring(lon, lat, half):
    """Build a closed square ring of (lon, lat) tuples around a center."""
    return [
        (lon - half, lat - half),
        (lon + half, lat - half),
        (lon + half, lat + half),
        (lon - half, lat + half),
        (lon - half, lat - half),
    ]


# ==========================================
# 1. Construction and lifecycle
# ==========================================


def test_module_defaults():
    """Module level default constants are sane."""
    assert DEFAULT_PATCH_SIZE == 512
    assert 0.0 <= DEFAULT_PATCH_OVERLAP_RATIO < 1.0


def test_init_reads_dimensions_without_opening_context(geo_tif):
    """The constructor records width/height and leaves the dataset closed."""
    reader = OrthomosaicReader(str(geo_tif))
    assert (reader.width, reader.height) == (64, 64)
    assert reader._dataset is None
    assert reader.global_min is None and reader.global_max is None
    assert reader.dataset_path.is_absolute()


def test_init_missing_file_raises(tmp_path):
    """A non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match='does not exist'):
        OrthomosaicReader(tmp_path / 'nope.tif')


def test_init_wrong_extension_raises(tmp_path):
    """A file with a non-TIFF extension raises ValueError."""
    bad = tmp_path / 'image.png'
    bad.write_bytes(b'not really an image')
    with pytest.raises(ValueError, match='must be a .tif or .tiff'):
        OrthomosaicReader(bad)


def test_context_manager_sets_and_clears_state(geo_tif):
    """Entering computes extent and stats; exiting closes the dataset."""
    reader = OrthomosaicReader(geo_tif)
    with reader as opened:
        assert opened is reader
        assert reader._dataset is not None
        assert reader.dataset_extent is not None
        assert reader.global_min is not None
        assert reader.global_max is not None
        assert reader.global_min < reader.global_max
    assert reader._dataset is None


def test_exit_is_idempotent(geo_tif):
    """Calling __exit__ on an already closed reader is harmless."""
    reader = OrthomosaicReader(geo_tif)
    reader.__exit__(None, None, None)
    assert reader._dataset is None


def test_enter_uses_explicit_nodata_and_overviews(feet_tif, tmp_path):
    """Explicit nodata pixels are excluded from the global stats."""
    with OrthomosaicReader(feet_tif) as reader:
        assert reader.global_min is not None
        assert reader.global_min >= 1.0

    # Build a copy with overviews to exercise the pyramid branch:
    with rasterio.open(feet_tif) as src:
        data = src.read()
        transform = src.transform
        crs = src.crs
    ov_path = _write_raster(
        tmp_path / 'ov.tif', data, crs, transform, nodata=0, overviews=[2, 4]
    )
    with OrthomosaicReader(ov_path) as reader:
        assert reader._dataset.overviews(1) == [2, 4]
        assert reader.global_min is not None


def test_enter_all_zero_raster_leaves_stats_unset(zero_tif):
    """A raster with no valid pixels keeps global_min/max as None."""
    with OrthomosaicReader(zero_tif) as reader:
        assert reader.global_min is None
        assert reader.global_max is None


def test_enter_falls_back_to_coarsest_overview(geo_tif, monkeypatch):
    """If every overview exceeds the target size the coarsest one is used."""

    class _HugeOverviewProxy:
        """Wrap a real dataset but pretend it is huge with coarse overviews."""

        def __init__(self, dataset):
            self._dataset = dataset

        def __getattr__(self, name):
            return getattr(self._dataset, name)

        def overviews(self, band):
            return [2, 4]

        @property
        def height(self):
            return 16384

        @property
        def width(self):
            return 16384

    real_open = ortho_module.rasterio_open

    def fake_open(path, **kwargs):
        return _HugeOverviewProxy(real_open(path, **kwargs))

    reader = OrthomosaicReader(geo_tif)
    monkeypatch.setattr(ortho_module, 'rasterio_open', fake_open)
    with reader:
        assert reader.global_min is not None
        assert reader.global_max is not None
    assert reader._dataset is None


# ==========================================
# 2. Methods requiring an open context
# ==========================================


def test_methods_raise_outside_context(geo_tif):
    """Data access methods refuse to run before __enter__."""
    reader = OrthomosaicReader(geo_tif)
    with pytest.raises(RuntimeError, match='context manager'):
        reader.get_image_patch([(-118.197, 34.097)])
    with pytest.raises(RuntimeError, match='context manager'):
        next(reader.generate_tiles())
    with pytest.raises(RuntimeError, match='context manager'):
        reader.get_raster_dimensions()


def test_get_mosaic_bbox_wgs84_inside_and_outside_context(geo_tif):
    """Bounds agree whether or not the dataset is open."""
    reader = OrthomosaicReader(geo_tif)
    closed_bbox = reader.get_mosaic_bbox_wgs84()
    with reader:
        open_bbox = reader.get_mosaic_bbox_wgs84()
        assert reader.dataset_extent == open_bbox
    assert closed_bbox == pytest.approx(open_bbox)
    assert closed_bbox[0] == pytest.approx(-118.2)
    assert closed_bbox[3] == pytest.approx(34.1)
    assert closed_bbox[2] == pytest.approx(-118.2 + 64 * 1e-4)
    assert closed_bbox[1] == pytest.approx(34.1 - 64 * 1e-4)


def test_get_mosaic_bbox_wgs84_reprojects_projected_crs(utm_tif):
    """A UTM raster is reported in geographic degrees."""
    with OrthomosaicReader(utm_tif) as reader:
        min_lon, min_lat, max_lon, max_lat = reader.get_mosaic_bbox_wgs84()
    assert -180 <= min_lon < max_lon <= 180
    assert -90 <= min_lat < max_lat <= 90
    # Zone 11N is centered on -117 degrees longitude:
    assert -120 < min_lon < -114


# ==========================================
# 3. get_image_patch
# ==========================================


def test_get_image_patch_polygon_default_buffer(geo_tif):
    """A closed ring yields an RGB patch, five local coords and a bbox."""
    with OrthomosaicReader(geo_tif) as reader:
        lon, lat = _center_lonlat(reader)
        ring = _square_ring(lon, lat, 5e-4)
        result = reader.get_image_patch(ring)
        assert result is not None
        patch, coords, bbox = result
        assert isinstance(patch, Image.Image)
        assert patch.mode == 'RGB'
        assert len(coords) == 5
        # All local coordinates fall inside the patch:
        for col, row in coords:
            assert 0 <= col <= patch.width
            assert 0 <= row <= patch.height
        # The patch bbox lies inside the raster extent:
        ext = reader.dataset_extent
        assert ext[0] - 1e-9 <= bbox[0] < bbox[2] <= ext[2] + 1e-9
        assert ext[1] - 1e-9 <= bbox[1] < bbox[3] <= ext[3] + 1e-9
        # The default 20% buffer on a 10-pixel square yields ~14 pixels tall;
        # in a geographic CRS the real-world square is wider in degrees:
        assert 12 <= patch.height <= 16
        assert patch.width >= patch.height


def test_get_image_patch_point_line_and_open_polyline(geo_tif):
    """Point, two-point line and open three-point line are all supported."""
    with OrthomosaicReader(geo_tif) as reader:
        lon, lat = _center_lonlat(reader)

        point = reader.get_image_patch([(lon, lat)], buffer='10 m')
        assert point is not None
        assert len(point[1]) == 1

        line = reader.get_image_patch([(lon, lat), (lon + 5e-4, lat)])
        assert line is not None
        assert len(line[1]) == 2
        assert line[0].width > 1

        polyline = reader.get_image_patch(
            [(lon, lat), (lon + 5e-4, lat), (lon + 5e-4, lat + 5e-4)]
        )
        assert polyline is not None
        assert len(polyline[1]) == 3


def test_get_image_patch_force_square_false(geo_tif):
    """Without force_square a wide geometry produces a wide patch."""
    with OrthomosaicReader(geo_tif) as reader:
        lon, lat = _center_lonlat(reader)
        line = [(lon - 1e-3, lat), (lon + 1e-3, lat)]
        square = reader.get_image_patch(line, buffer='0%', force_square=True)
        rect = reader.get_image_patch(line, buffer='0.0002', force_square=False)
        assert square is not None and rect is not None
        # A real-world square spans more degrees (pixels) in longitude:
        cos_lat = math.cos(math.radians(lat))
        assert abs(square[0].width * cos_lat - square[0].height) <= 2
        assert rect[0].width > rect[0].height
        assert rect[0].width / rect[0].height > square[0].width / square[0].height


def test_get_image_patch_buffer_variants(geo_tif):
    """Percent, real-world distance and native-unit buffers all work."""
    with OrthomosaicReader(geo_tif) as reader:
        lon, lat = _center_lonlat(reader)
        ring = _square_ring(lon, lat, 2e-4)
        sizes = {}
        for label, buf in {
            'pct': '50%',
            'feet': '50 ft',
            'meters': '10 m',
            'native': 0.0005,
        }.items():
            result = reader.get_image_patch(ring, buffer=buf)
            assert result is not None, label
            sizes[label] = result[0].width
        # 0.0005 degrees of padding is the largest buffer requested:
        assert sizes['native'] > sizes['pct']
        assert sizes['native'] > sizes['meters']


def test_get_image_patch_invalid_inputs(geo_tif):
    """Empty geometry and unparsable buffers raise ValueError."""
    with OrthomosaicReader(geo_tif) as reader:
        lon, lat = _center_lonlat(reader)
        with pytest.raises(ValueError, match='cannot be empty'):
            reader.get_image_patch([])
        with pytest.raises(ValueError, match='Could not parse buffer'):
            reader.get_image_patch([(lon, lat)], buffer='abc')
        with pytest.raises(ValueError, match='Unsupported distance unit'):
            reader.get_image_patch([(lon, lat)], buffer='5 furlongs')


def test_get_image_patch_outside_raster_returns_none(geo_tif):
    """A geometry far away from the raster yields None instead of raising."""
    with OrthomosaicReader(geo_tif) as reader:
        assert reader.get_image_patch([(-100.0, 30.0)]) is None


def test_get_image_patch_missing_data_threshold(feet_tif):
    """The nodata ratio gate rejects and accepts patches as configured."""
    with OrthomosaicReader(feet_tif) as reader:
        min_lon, min_lat, max_lon, max_lat = reader.dataset_extent
        # A point on the right (nodata) half of the raster:
        lon = min_lon + 0.8 * (max_lon - min_lon)
        lat = (min_lat + max_lat) / 2.0
        ring = _square_ring(lon, lat, 2e-5)
        assert reader.get_image_patch(ring, max_missing_data_ratio=0.2) is None
        result = reader.get_image_patch(ring, max_missing_data_ratio=1.01)
        assert result is not None
        # Four input bands are truncated to RGB:
        assert result[0].mode == 'RGB'


def test_get_image_patch_projected_meters_and_feet(utm_tif, feet_tif):
    """Projected rasters in meters and US feet are buffered correctly."""
    with OrthomosaicReader(utm_tif) as reader:
        lon, lat = _center_lonlat(reader)
        result = reader.get_image_patch([(lon, lat)], buffer='5 m')
        assert result is not None
        # 5 m buffer on 0.5 m pixels => ~20 px square:
        assert 18 <= result[0].width <= 22

    with OrthomosaicReader(feet_tif) as reader:
        min_lon, min_lat, max_lon, max_lat = reader.dataset_extent
        lon = min_lon + 0.25 * (max_lon - min_lon)
        lat = (min_lat + max_lat) / 2.0
        result = reader.get_image_patch([(lon, lat)], buffer='10 ft')
        assert result is not None
        # 10 ft buffer on 1 ft pixels => ~20 px square:
        assert 18 <= result[0].width <= 22


def test_get_image_patch_float_raster_uses_global_stats(float_tif):
    """Float rasters are scaled to 8-bit using the global percentiles."""
    with OrthomosaicReader(float_tif) as reader:
        lon, lat = _center_lonlat(reader)
        result = reader.get_image_patch([(lon, lat)], buffer='30 m')
        assert result is not None
        arr = np.asarray(result[0])
        assert arr.dtype == np.uint8
        assert arr.shape[2] == 3
        # Single band replicated to RGB: all channels identical
        assert np.array_equal(arr[..., 0], arr[..., 1])
        assert np.array_equal(arr[..., 0], arr[..., 2])


# ==========================================
# 4. generate_tiles
# ==========================================


def test_generate_tiles_pixels_no_overlap(geo_tif):
    """A 64x64 raster tiled at 32 px without overlap gives four tiles."""
    with OrthomosaicReader(geo_tif) as reader:
        tiles = list(reader.generate_tiles(patch_size=32, overlap_ratio=0.0))
        assert len(tiles) == 4
        for img, bbox in tiles:
            assert img.size == (32, 32)
            assert img.mode == 'RGB'
            assert bbox[0] < bbox[2] and bbox[1] < bbox[3]


def test_generate_tiles_overlap_increases_count(geo_tif):
    """Overlap shrinks the stride and therefore yields more tiles."""
    with OrthomosaicReader(geo_tif) as reader:
        none = list(reader.generate_tiles(patch_size=32, overlap_ratio=0.0))
        half = list(reader.generate_tiles(patch_size=32, overlap_ratio=0.5))
        assert len(half) > len(none)


def test_generate_tiles_edge_padding(geo_tif):
    """Edge tiles are padded to uniform size or left ragged as requested."""
    with OrthomosaicReader(geo_tif) as reader:
        padded = list(
            reader.generate_tiles(
                patch_size=48, overlap_ratio=0.0, max_missing_data_ratio=1.0
            )
        )
        assert padded and all(img.size == (48, 48) for img, _ in padded)

        ragged = list(
            reader.generate_tiles(
                patch_size=48, overlap_ratio=0.0, pad_edge_tiles=False
            )
        )
        assert len(ragged) == len(padded)
        assert any(img.size != (48, 48) for img, _ in ragged)

        # With padding and a strict missing-data budget, edge tiles that are
        # mostly padding are dropped:
        strict = list(
            reader.generate_tiles(
                patch_size=48, overlap_ratio=0.0, max_missing_data_ratio=0.1
            )
        )
        assert len(strict) == 1


def test_generate_tiles_padding_uses_nodata(feet_tif):
    """Edge padding uses the dataset nodata value and skips empty regions."""
    with OrthomosaicReader(feet_tif) as reader:
        tiles = list(
            reader.generate_tiles(
                patch_size=40, overlap_ratio=0.0, max_missing_data_ratio=0.6
            )
        )
        # The right half of the raster is nodata, so only the left column
        # of tiles survives (the bottom-left one is 52% padding/nodata):
        assert len(tiles) == 2
        strict = list(
            reader.generate_tiles(
                patch_size=40, overlap_ratio=0.0, max_missing_data_ratio=0.5
            )
        )
        assert len(strict) == 1
        assert all(img.size == (40, 40) for img, _ in tiles)


def test_generate_tiles_real_world_units(geo_tif, utm_tif, feet_tif):
    """Patch sizes given in meters/feet are converted per CRS type."""
    with OrthomosaicReader(geo_tif) as reader:
        # ~11.1 m per pixel in latitude: 222 m => ~20 px tiles
        tiles = list(reader.generate_tiles(patch_size=222, unit='m'))
        assert tiles
        assert 18 <= tiles[0][0].height <= 22

    with OrthomosaicReader(utm_tif) as reader:
        # 0.5 m pixels: 20 m => 40 px tiles
        tiles = list(reader.generate_tiles(patch_size=20, unit='meters'))
        assert tiles and tiles[0][0].size == (40, 40)

    with OrthomosaicReader(feet_tif) as reader:
        # 1 ft pixels: 32 ft => 32 px tiles
        tiles = list(reader.generate_tiles(patch_size=32, unit='ft'))
        assert tiles and tiles[0][0].size == (32, 32)


def test_generate_tiles_invalid_arguments(geo_tif):
    """Bad overlap ratios and units raise ValueError."""
    with OrthomosaicReader(geo_tif) as reader:
        with pytest.raises(ValueError, match='overlap_ratio'):
            next(reader.generate_tiles(overlap_ratio=1.0))
        with pytest.raises(ValueError, match='Unsupported unit'):
            next(reader.generate_tiles(unit='furlongs'))


# ==========================================
# 5. Dimensions and distances
# ==========================================


def test_get_raster_dimensions_pixels_and_units(geo_tif):
    """Dimensions are reported in pixels or converted real-world units."""
    with OrthomosaicReader(geo_tif) as reader:
        assert reader.get_raster_dimensions() == (64.0, 64.0)
        assert reader.get_raster_dimensions(' PX ') == (64.0, 64.0)

        width_m, height_m = reader.get_raster_dimensions('m')
        # 64 pixels * 1e-4 deg = 0.0064 deg => ~712 m tall, ~590 m wide
        assert height_m == pytest.approx(712, rel=0.02)
        expected_width = 712 * math.cos(math.radians(34.0968))
        assert width_m == pytest.approx(expected_width, rel=0.02)

        width_km, height_km = reader.get_raster_dimensions('km')
        assert width_km == pytest.approx(width_m / 1000.0)
        assert height_km == pytest.approx(height_m / 1000.0)

        width_ft, _ = reader.get_raster_dimensions('feet')
        assert width_ft == pytest.approx(width_m / 0.3048)

        with pytest.raises(ValueError, match='Unsupported unit'):
            reader.get_raster_dimensions('parsecs')


def test_haversine_distance_known_values():
    """Haversine distances match textbook values and wrap the dateline."""
    hav = OrthomosaicReader.haversine_distance
    assert hav(10.0, 20.0, 10.0, 20.0) == 0.0
    one_degree = hav(0.0, 0.0, 1.0, 0.0)
    assert one_degree == pytest.approx(111_195, rel=1e-3)
    # Crossing the antimeridian should be one degree, not 359:
    assert hav(179.5, 0.0, -179.5, 0.0) == pytest.approx(one_degree, rel=1e-6)
    # Antipodal points are half the circumference:
    assert hav(0.0, 0.0, 180.0, 0.0) == pytest.approx(math.pi * 6_371_000)


# ==========================================
# 6. Static helpers
# ==========================================


def test_get_buffered_bounds_percent_and_square():
    """Percent buffers pad proportionally; force_square squares the box."""
    bounds = OrthomosaicReader._get_buffered_bounds(
        box(0, 0, 10, 10), buffer='50%', force_square=False
    )
    assert bounds == pytest.approx((-5.0, -5.0, 15.0, 15.0))

    square = OrthomosaicReader._get_buffered_bounds(
        box(0, 0, 20, 10), buffer='0%', force_square=True
    )
    minx, miny, maxx, maxy = square
    assert (maxx - minx) == pytest.approx(maxy - miny) == pytest.approx(20.0)
    assert (minx + maxx) / 2 == pytest.approx(10.0)
    assert (miny + maxy) / 2 == pytest.approx(5.0)


def test_get_buffered_bounds_point_and_units():
    """Points get a nominal buffer; real-world units convert per CRS."""
    # Point with a percent buffer in a projected CRS gets 10 m padding:
    pt = OrthomosaicReader._get_buffered_bounds(Point(5, 5), buffer='20%')
    assert pt == pytest.approx((-5.0, -5.0, 15.0, 15.0))

    # 50 ft in a US-foot CRS is exactly 50 native units:
    ft = OrthomosaicReader._get_buffered_bounds(
        Point(0, 0), buffer='50 ft', force_square=False, meters_per_crs_unit=0.3048
    )
    assert ft == pytest.approx((-50.0, -50.0, 50.0, 50.0))

    # 1 km in a meter CRS is 1000 native units:
    km = OrthomosaicReader._get_buffered_bounds(
        Point(0, 0), buffer='1 km', force_square=False
    )
    assert km == pytest.approx((-1000.0, -1000.0, 1000.0, 1000.0))

    # Bare numbers are native CRS units:
    native = OrthomosaicReader._get_buffered_bounds(
        box(0, 0, 2, 2), buffer=1.5, force_square=False
    )
    assert native == pytest.approx((-1.5, -1.5, 3.5, 3.5))


def test_get_buffered_bounds_geographic_scaling():
    """In geographic CRSs the X buffer is stretched by 1/cos(latitude)."""
    lat = 60.0
    geom = LineString([(10.0, lat), (10.0, lat + 0.01)])
    minx, miny, maxx, maxy = OrthomosaicReader._get_buffered_bounds(
        geom, buffer='10 m', force_square=False, is_geographic=True
    )
    buffer_y = 10.0 / 111_320.0
    center_lat = lat + 0.005
    assert (lat - miny) == pytest.approx(buffer_y)
    assert (10.0 - minx) == pytest.approx(buffer_y / math.cos(math.radians(center_lat)))

    # Point with percent buffer in degrees uses a 10 m nominal buffer:
    pt = OrthomosaicReader._get_buffered_bounds(
        Point(0.0, 0.0), buffer='20%', is_geographic=True
    )
    assert pt[3] == pytest.approx(buffer_y)

    # force_square in degrees produces a real-world square:
    sq = OrthomosaicReader._get_buffered_bounds(
        geom, buffer='0%', force_square=True, is_geographic=True
    )
    width_deg = sq[2] - sq[0]
    height_deg = sq[3] - sq[1]
    assert width_deg * math.cos(math.radians(center_lat)) == pytest.approx(height_deg)


def test_get_buffered_bounds_invalid():
    """Unsupported units and garbage strings raise ValueError."""
    with pytest.raises(ValueError, match='Unsupported distance unit'):
        OrthomosaicReader._get_buffered_bounds(Point(0, 0), buffer='3 leagues')
    with pytest.raises(ValueError, match='Could not parse buffer'):
        OrthomosaicReader._get_buffered_bounds(Point(0, 0), buffer='wide')


def test_read_image_window(geo_tif):
    """Windows are clipped to the raster and empty when disjoint."""
    with rasterio.open(geo_tif) as src:
        full = OrthomosaicReader._read_image_window(src, src.bounds)
        assert full.shape == (3, 64, 64)

        minx, miny, maxx, maxy = src.bounds
        half_w = (maxx - minx) / 2.0
        partial = OrthomosaicReader._read_image_window(
            src, (minx - half_w, miny, minx + half_w, maxy)
        )
        assert partial.shape == (3, 64, 32)

        outside = OrthomosaicReader._read_image_window(
            src, (minx - 10, miny - 10, minx - 9, miny - 9)
        )
        assert outside.size == 0
        assert outside.dtype == np.uint8


def test_format_array_for_pil_uint8_passthrough():
    """8-bit input is only truncated to RGB and reordered to HWC."""
    raw = np.arange(4 * 2 * 3, dtype=np.uint8).reshape(4, 2, 3)
    out = OrthomosaicReader._format_array_for_pil(raw, 0.0, 1000.0)
    assert out.shape == (2, 3, 3)
    assert out.dtype == np.uint8
    assert np.array_equal(out, np.moveaxis(raw[:3], 0, -1))


def test_format_array_for_pil_scaling_branches():
    """Each radiometric scaling branch produces valid 8-bit output."""
    fmt = OrthomosaicReader._format_array_for_pil

    # Global statistics take priority:
    raw16 = np.full((3, 2, 2), 500, dtype=np.uint16)
    out = fmt(raw16, global_min=0.0, global_max=1000.0)
    assert int(out[0, 0, 0]) == 127

    # 16-bit fallback divides by 256:
    out = fmt(np.full((3, 2, 2), 4096, dtype=np.uint16))
    assert int(out[0, 0, 0]) == 16

    # Normalized float in [0, 1] is multiplied by 255:
    out = fmt(np.full((1, 2, 2), 0.5, dtype=np.float32))
    assert int(out[0, 0, 0]) == 127

    # Float above 1.0 without globals is locally normalized:
    raw = np.array([[[10.0, 20.0], [30.0, 40.0]]], dtype=np.float64)
    out = fmt(raw)
    assert int(out[0, 0, 0]) == 0
    assert int(out[1, 1, 0]) == 255

    # Constant float above 1.0 cannot be stretched and is simply cast; a
    # single band is replicated to three identical channels:
    out = fmt(np.full((1, 2, 2), 42.0, dtype=np.float32))
    assert out.shape == (2, 2, 3)
    assert int(out[0, 0, 0]) == 42
    assert np.array_equal(out[..., 0], out[..., 2])

    # Two-band input is also promoted to RGB from its first band:
    two = np.stack([np.full((2, 2), 7, np.uint8), np.full((2, 2), 9, np.uint8)])
    out = fmt(two)
    assert out.shape == (2, 2, 3)
    assert int(out[0, 0, 2]) == 7

    # Degenerate globals (min == max) fall through to the dtype branches:
    out = fmt(np.full((3, 2, 2), 4096, dtype=np.uint16), 5.0, 5.0)
    assert int(out[0, 0, 0]) == 16


def test_is_image_valid():
    """Missing-data ratio is compared strictly against the threshold."""
    valid = OrthomosaicReader._is_image_valid
    assert valid(np.empty((3, 0, 0)), 0.5, None) is False

    arr = np.ones((3, 4, 4), dtype=np.uint8)
    arr[:, :, :2] = 0  # exactly half missing
    assert valid(arr, 0.6, None) is True
    assert valid(arr, 0.5, None) is False
    assert isinstance(valid(arr, 0.6, None), bool)

    # Explicit nodata value overrides the zero default:
    arr = np.full((1, 2, 2), 255, dtype=np.uint8)
    assert valid(arr, 0.5, 255) is False
    assert valid(arr, 0.5, None) is True


def test_get_local_pixel_coords(geo_tif):
    """Native coordinates map to pixel offsets relative to the window."""
    with rasterio.open(geo_tif) as src:
        minx, miny, maxx, maxy = src.bounds
        coords = OrthomosaicReader._get_local_pixel_coords(
            src, [(minx, maxy)], src.bounds
        )
        assert coords == [(0, 0)]

        # Shift the window by 10 pixels in each direction and probe pixel
        # centers to avoid floating-point ambiguity on pixel edges:
        px = src.transform.a
        dx = 10 * px
        dy = 10 * abs(src.transform.e)
        sub_bounds = (minx + dx, miny, maxx, maxy - dy)
        coords = OrthomosaicReader._get_local_pixel_coords(
            src,
            [
                (minx + dx + 0.5 * px, maxy - dy - 0.5 * px),
                (minx + 2 * dx + 0.5 * px, maxy - 2 * dy - 0.5 * px),
            ],
            sub_bounds,
        )
        assert coords == [(0, 0), (10, 10)]


# ==========================================
# Footprint coverage and padded edge crops
# ==========================================


def _edge_raster(tmp_path):
    """8x8 EPSG:4326 raster over (0,0)-(8,8) whose left half is nodata."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    path = tmp_path / 'edge.tif'
    data = np.full((3, 8, 8), 120, dtype='uint8')
    data[:, :, :4] = 0  # nodata region (value 0)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=8,
        width=8,
        count=3,
        dtype='uint8',
        crs='EPSG:4326',
        transform=from_bounds(0, 0, 8, 8, 8, 8),
        nodata=0,
    ) as dst:
        dst.write(data)
    return path


def test_footprint_coverage_keeps_partially_imaged_assets(tmp_path):
    """Assets straddling the imagery edge are kept by coverage, not patch nodata."""
    from rapidtools.data_sources import OrthomosaicReader

    # Square footprint from x=3..5: half over nodata, half over imagery.
    footprint = [(3, 3), (5, 3), (5, 5), (3, 5), (3, 3)]
    with OrthomosaicReader(_edge_raster(tmp_path)) as reader:
        # Patch-level test with a big buffer sees > 20% nodata -> rejected:
        assert (
            reader.get_image_patch(footprint, buffer=2.0, max_missing_data_ratio=0.2)
            is None
        )
        # Coverage test: half the footprint is imaged -> kept at 0.5, dropped at 0.6.
        assert reader.get_image_patch(footprint, buffer=2.0, min_footprint_coverage=0.5)
        assert (
            reader.get_image_patch(footprint, buffer=2.0, min_footprint_coverage=0.6)
            is None
        )
        # Entirely in the nodata region -> zero coverage:
        nodata_only = [(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5), (0.5, 0.5)]
        assert (
            reader.get_image_patch(nodata_only, buffer=0.5, min_footprint_coverage=0.01)
            is None
        )
        # Points fall back to the whole patch's valid fraction:
        assert reader.get_image_patch([(6, 6)], buffer=1.0, min_footprint_coverage=0.9)


def test_pad_edges_keeps_full_window_beyond_raster(tmp_path):
    """Padded reads keep the requested size; clipped reads shrink at the edge."""
    from rapidtools.data_sources import OrthomosaicReader

    # Footprint hugging the right edge; a 1-unit buffer extends past x=8.
    footprint = [(6.5, 3), (8, 3), (8, 5), (6.5, 5), (6.5, 3)]
    with OrthomosaicReader(_edge_raster(tmp_path)) as reader:
        clipped, clipped_px, clipped_bounds = reader.get_image_patch(
            footprint, buffer=1.0
        )
        padded, padded_px, padded_bounds = reader.get_image_patch(
            footprint, buffer=1.0, min_footprint_coverage=0.5
        )
        assert padded.size[0] > clipped.size[0]  # not cut at the raster edge
        assert padded.size[0] == 4  # full requested (square) window
        assert padded_bounds[2] > 8.0 >= clipped_bounds[2]
        # The footprint's first vertex keeps its row and column offset:
        assert padded_px[0] == clipped_px[0]
        # Explicit pad_edges works without the coverage test too:
        only_pad, _, only_pad_bounds = reader.get_image_patch(
            footprint, buffer=1.0, pad_edges=True, max_missing_data_ratio=0.9
        )
        assert only_pad.size == padded.size and only_pad_bounds == padded_bounds
        # A window completely outside the raster is still None when padded:
        assert reader.get_image_patch([(20, 20)], buffer=1.0, pad_edges=True) is None


def test_read_patch_and_footprint_coverage_helpers(tmp_path):
    """Low-level helpers report windows and coverage fractions."""
    import numpy as np
    import rasterio
    from rasterio.windows import Window
    from shapely.geometry import LineString, box

    from rapidtools.data_sources import OrthomosaicReader

    with rasterio.open(_edge_raster(tmp_path)) as src:
        arr, window = OrthomosaicReader._read_patch(src, (2, 2, 6, 6))
        assert arr.shape == (3, 4, 4) and (window.col_off, window.row_off) == (2, 2)
        arr, window = OrthomosaicReader._read_patch(src, (6, 2, 10, 6), boundless=True)
        assert arr.shape == (3, 4, 4) and window.width == 4
        assert np.all(arr[:, :, 2:] == 0)  # padded columns beyond x=8
        arr, _ = OrthomosaicReader._read_patch(src, (20, 20, 22, 22), boundless=True)
        assert arr.size == 0
        # Coverage of a footprint fully over imagery vs. over nodata:
        arr, window = OrthomosaicReader._read_patch(src, (0, 0, 8, 8))
        assert (
            OrthomosaicReader._footprint_coverage(arr, box(5, 1, 7, 3), window, src)
            == 1.0
        )
        assert (
            OrthomosaicReader._footprint_coverage(arr, box(1, 1, 3, 3), window, src)
            == 0.0
        )
        half = OrthomosaicReader._footprint_coverage(arr, box(2, 1, 6, 3), window, src)
        assert half == pytest.approx(0.5)
        # Zero-area geometry -> whole-patch fraction (half the raster is valid):
        line = OrthomosaicReader._footprint_coverage(
            arr, LineString([(1, 1), (7, 7)]), window, src
        )
        assert line == pytest.approx(0.5)
        empty = OrthomosaicReader._footprint_coverage(
            np.empty((3, 0, 0), dtype='uint8'), box(0, 0, 1, 1), Window(0, 0, 0, 0), src
        )
        assert empty == 0.0
