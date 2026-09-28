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
# 09-28-2026

import logging
import re
import threading
from io import BytesIO

import numpy as np
import pytest
import rasterio
import requests
import requests_mock
from PIL import Image
from rasterio.transform import from_bounds
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

from rapidtools.core import (
    BoundingBox,
    ImageAsset,
    ImageCollection,
    OperationCancelled,
    PhysicalAsset,
    PhysicalAssetCollection,
    PolygonRegion,
)
from rapidtools.processing import image_extractors
from rapidtools.processing.image_extractors import (
    AerialImageryExtractor,
    BingOrthomosaicExtractor,
    MapillaryImageExtractor,
)

# ==========================================
# 1. Fixtures
# ==========================================

RASTER_BOUNDS = (-118.250, 34.050, -118.249, 34.051)
CX, CY = -118.2493, 34.0505


def _write_raster(path, crs, bounds, size=64, nodata_left_cols=0):
    minx, miny, maxx, maxy = bounds
    transform = from_bounds(minx, miny, maxx, maxy, size, size)
    rng = np.random.default_rng(7)
    data = rng.integers(50, 200, size=(3, size, size), dtype=np.uint8)
    if nodata_left_cols:
        data[:, :, :nodata_left_cols] = 0
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=size,
        width=size,
        count=3,
        dtype='uint8',
        crs=crs,
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(data)
    return path


@pytest.fixture
def raster(tmp_path):
    """EPSG:4326 raster with a nodata strip on its left quarter."""
    return _write_raster(
        tmp_path / 'ortho.tif', 'EPSG:4326', RASTER_BOUNDS, nodata_left_cols=16
    )


@pytest.fixture
def utm_raster(tmp_path):
    return _write_raster(
        tmp_path / 'utm.tiff', 'EPSG:32611', (500000, 3767000, 500064, 3767064)
    )


def _square(cx, cy, half=0.00005):
    return box(cx - half, cy - half, cx + half, cy + half)


@pytest.fixture
def collection():
    """Assets of several geometry types inside and outside the raster."""
    return PhysicalAssetCollection(
        [
            PhysicalAsset(id='poly', geometry=_square(CX, CY)),
            PhysicalAsset(id='pt', geometry=Point(CX + 0.0002, CY + 0.0002)),
            PhysicalAsset(
                id='line',
                geometry=LineString([(CX, CY - 0.0003), (CX + 0.0002, CY - 0.0003)]),
            ),
            PhysicalAsset(
                id='multi',
                geometry=MultiPolygon(
                    [
                        _square(CX - 0.0002, CY, 0.00003),
                        _square(CX - 0.00015, CY, 0.00003),
                    ]
                ),
            ),
            PhysicalAsset(id='outside', geometry=_square(-118.30, 34.10)),
            PhysicalAsset(id='nodata', geometry=_square(-118.24995, CY, 0.00003)),
        ]
    )


def _jpeg_bytes(color=(200, 30, 30), size=(256, 256)):
    buf = BytesIO()
    Image.new('RGB', size, color=color).save(buf, format='JPEG')
    return buf.getvalue()


TILE_URL = re.compile(r'http://ecn\.t3\.tiles\.virtualearth\.net/tiles/a\d+\.jpeg')


# ==========================================
# 2. AerialImageryExtractor - configuration
# ==========================================


def test_aerial_init_stores_config(tmp_path):
    """Constructor arguments are stored and the save directory is resolved."""
    event = threading.Event()
    ext = AerialImageryExtractor(
        dataset='x.tif',
        save_directory=tmp_path / 'out',
        max_missing_data_ratio=0.5,
        overlay_asset_outline=True,
        image_prefix='pre',
        keep_multiple_copies=True,
        buffer_asset='10 m',
        force_square_image=False,
        cancel_event=event,
    )
    assert ext.dataset == 'x.tif'
    assert ext.save_directory == (tmp_path / 'out').resolve()
    assert ext.max_missing_data_ratio == 0.5
    assert ext.overlay_asset_outline is True
    assert ext.image_prefix == 'pre'
    assert ext.keep_multiple_copies is True
    assert ext.buffer_asset == '10 m'
    assert ext.force_square_image is False
    assert ext.cancel_event is event


def test_aerial_get_raster_paths_single_file(raster, tmp_path):
    """A single file path resolves to a one-element list."""
    ext = AerialImageryExtractor(raster, tmp_path / 'out')
    assert ext._get_raster_paths() == [raster.resolve()]


def test_aerial_get_raster_paths_directory(raster, utm_raster, tmp_path):
    """A directory is searched recursively for .tif and .tiff files."""
    nested = tmp_path / 'nested'
    nested.mkdir()
    _write_raster(nested / 'deep.tif', 'EPSG:4326', RASTER_BOUNDS)
    (tmp_path / 'ignore.txt').write_text('nope')
    ext = AerialImageryExtractor(tmp_path, tmp_path / 'out')
    found = sorted(p.name for p in ext._get_raster_paths())
    assert found == ['deep.tif', 'ortho.tif', 'utm.tiff']


def test_aerial_get_raster_paths_list_dedup(raster, tmp_path):
    """Duplicate entries in a list input are removed."""
    ext = AerialImageryExtractor([raster, str(raster)], tmp_path / 'out')
    assert ext._get_raster_paths() == [raster.resolve()]


# ==========================================
# 3. AerialImageryExtractor - extraction
# ==========================================


def test_aerial_extracts_crops_for_assets_in_bounds(raster, collection, tmp_path):
    """Crops are attached for polygon, point, line and multi assets only."""
    out = tmp_path / 'crops'
    ext = AerialImageryExtractor(raster, out)
    result = ext(collection)
    assert result is collection

    for asset_id in ('poly', 'pt', 'line', 'multi'):
        images = collection[asset_id].image_assets
        assert len(images) == 1, asset_id
        img = images[0]
        assert img.id == f'{asset_id}_aerial'
        assert img.path.exists()
        assert img.path.parent == out.resolve()
        assert img.path.name.startswith('aerial_')
        assert img.properties['source_raster'] == 'ortho.tif'
        lon_min, lat_min, lon_max, lat_max = img.properties['wgs84_bounds']
        assert lon_min < lon_max and lat_min < lat_max
        with Image.open(img.path) as pil:
            assert pil.size[0] > 0 and pil.size[1] > 0

    # Asset outside the raster and asset sitting in the nodata strip get nothing:
    assert len(collection['outside'].image_assets) == 0
    assert len(collection['nodata'].image_assets) == 0


def test_aerial_buffer_units_change_patch_size(raster, tmp_path):
    """Absolute and real-world buffers yield larger patches than a small percent."""
    sizes = {}
    for label, buf in (('pct', '5%'), ('m', '15 m'), ('abs', 0.0002)):
        col = PhysicalAssetCollection([PhysicalAsset(id='a', geometry=_square(CX, CY))])
        ext = AerialImageryExtractor(
            raster, tmp_path / label, buffer_asset=buf, force_square_image=False
        )
        ext(col)
        with Image.open(col['a'].image_assets[0].path) as pil:
            sizes[label] = pil.size
    assert sizes['m'][0] > sizes['pct'][0]
    assert sizes['abs'][0] > sizes['pct'][0]


def test_aerial_force_square(raster, tmp_path):
    """force_square_image=True yields (near) square real-world coverage."""
    wide = box(CX - 0.00015, CY - 0.00003, CX + 0.00015, CY + 0.00003)
    col_sq = PhysicalAssetCollection([PhysicalAsset(id='a', geometry=wide)])
    col_rect = PhysicalAssetCollection([PhysicalAsset(id='a', geometry=wide)])
    AerialImageryExtractor(raster, tmp_path / 'sq', force_square_image=True)(col_sq)
    AerialImageryExtractor(raster, tmp_path / 'rect', force_square_image=False)(
        col_rect
    )
    b_sq = col_sq['a'].image_assets[0].properties['wgs84_bounds']
    b_rect = col_rect['a'].image_assets[0].properties['wgs84_bounds']
    ratio_sq = (b_sq[2] - b_sq[0]) / (b_sq[3] - b_sq[1])
    ratio_rect = (b_rect[2] - b_rect[0]) / (b_rect[3] - b_rect[1])
    assert abs(ratio_sq - 1.0) < abs(ratio_rect - 1.0)
    assert ratio_rect > 1.5


def test_aerial_overlay_outline_draws_red(raster, tmp_path):
    """Overlaying the outline paints red pixels for polygons, lines and points."""
    col = PhysicalAssetCollection(
        [
            PhysicalAsset(id='poly', geometry=_square(CX, CY)),
            PhysicalAsset(id='pt', geometry=Point(CX + 0.0002, CY + 0.0002)),
            PhysicalAsset(
                id='line',
                geometry=LineString([(CX, CY - 0.0003), (CX + 0.0002, CY - 0.0003)]),
            ),
        ]
    )
    plain = PhysicalAssetCollection(
        [PhysicalAsset(id='poly', geometry=_square(CX, CY))]
    )
    AerialImageryExtractor(raster, tmp_path / 'plain')(plain)
    AerialImageryExtractor(raster, tmp_path / 'ol', overlay_asset_outline=True)(col)

    def red_fraction(path):
        arr = np.asarray(Image.open(path).convert('RGB')).astype(int)
        red = (arr[..., 0] > 150) & (arr[..., 1] < 100) & (arr[..., 2] < 100)
        return red.mean()

    base = red_fraction(plain['poly'].image_assets[0].path)
    for asset_id in ('poly', 'pt', 'line'):
        assert red_fraction(col[asset_id].image_assets[0].path) > base


# ---------------------------------------------------------- outline options
def _color_mask(path, color):
    """Boolean mask of pixels that are clearly ``color`` ('red' or 'blue')."""
    arr = np.asarray(Image.open(path).convert('RGB')).astype(int)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    if color == 'red':
        return (r > 150) & (g < 100) & (b < 100)
    return (b > 150) & (r < 100) & (g < 100)


def _extract_one(raster, out_dir, geometry, **kwargs):
    """Extract a single asset with outline drawing on and return the crop path."""
    col = PhysicalAssetCollection([PhysicalAsset(id='a', geometry=geometry)])
    kwargs.setdefault('buffer_asset', '20 m')
    AerialImageryExtractor(raster, out_dir, overlay_asset_outline=True, **kwargs)(col)
    return col['a'].image_assets[0].path


def _diamond(cx, cy, h=0.0001):
    return Polygon([(cx, cy + h), (cx + h, cy), (cx, cy - h), (cx - h, cy)])


def _l_shape(cx, cy, h=0.0001):
    return Polygon(
        [
            (cx - h, cy - h),
            (cx + h, cy - h),
            (cx + h, cy),
            (cx, cy),
            (cx, cy + h),
            (cx - h, cy + h),
        ]
    )


def test_aerial_outline_defaults_match_legacy_behaviour(tmp_path):
    ext = AerialImageryExtractor('x.tif', tmp_path)
    assert ext.outline_shape == 'geometry'
    assert ext.outline_buffer == 0
    assert ext.outline_width == 6
    assert ext.outline_color == 'red'
    assert image_extractors.OUTLINE_SHAPES == (
        'geometry',
        'bbox',
        'rotated_bbox',
        'convex_hull',
        'corners',
    )


@pytest.mark.parametrize(
    'kwargs',
    [
        {'outline_shape': 'hexagon'},
        {'outline_buffer': 'abc'},
        {'outline_buffer': '-1'},
        {'outline_buffer': '5 lightyears'},
        {'outline_width': 0},
        {'outline_width': '2 m'},
        {'outline_color': 'not-a-colour'},
    ],
)
def test_aerial_outline_options_are_validated_at_construction(tmp_path, kwargs):
    with pytest.raises(ValueError):
        AerialImageryExtractor('x.tif', tmp_path, **kwargs)


@pytest.mark.parametrize(
    'spec, expected',
    [
        (3, (3.0, 'px')),
        (2.5, (2.5, 'px')),
        ('12', (12.0, 'px')),
        ('12 px', (12.0, 'px')),
        ('10%', (10.0, '%')),
        ('2 m', (2.0, 'm')),
        ('2 metres', (2.0, 'm')),
        ('5 ft', (1.524, 'm')),
    ],
)
def test_aerial_parse_outline_buffer(spec, expected):
    value, kind = AerialImageryExtractor._parse_outline_buffer(spec)
    assert kind == expected[1]
    assert value == pytest.approx(expected[0])


def test_aerial_parse_outline_buffer_rejects_bool():
    with pytest.raises(ValueError):
        AerialImageryExtractor._parse_outline_buffer(True)


@pytest.mark.parametrize(
    'spec, expected',
    [(6, (6.0, 'px')), ('4 px', (4.0, 'px')), ('1.5%', (1.5, '%'))],
)
def test_aerial_parse_outline_width(spec, expected):
    assert AerialImageryExtractor._parse_outline_width(spec) == expected


def test_aerial_outline_pixel_resolution(tmp_path):
    ext = AerialImageryExtractor('x.tif', tmp_path, outline_width='10%')
    assert ext._outline_width_px((200, 100)) == 10
    ext = AerialImageryExtractor('x.tif', tmp_path, outline_width=0.2)
    assert ext._outline_width_px((200, 100)) == 1  # never below one pixel

    ext = AerialImageryExtractor('x.tif', tmp_path, outline_buffer=4)
    assert ext._outline_buffer_px(50.0, 0.5) == 4.0
    ext = AerialImageryExtractor('x.tif', tmp_path, outline_buffer='10%')
    assert ext._outline_buffer_px(50.0, 0.5) == 5.0
    ext = AerialImageryExtractor('x.tif', tmp_path, outline_buffer='2 m')
    assert ext._outline_buffer_px(50.0, 0.5) == 4.0
    assert ext._outline_buffer_px(50.0, 0.0) == 0.0  # unknown scale: no buffer

    # 0.001 degrees of latitude is ~111 m, so a 111 px tall crop is ~1 m/px:
    mpp = AerialImageryExtractor._meters_per_pixel((0, 0, 0.001, 0.001), (50, 111))
    assert mpp == pytest.approx(1.0, rel=0.01)
    assert AerialImageryExtractor._meters_per_pixel((0, 0, 1, 1), (50, 0)) == 0.0


def test_aerial_outline_geometry_shapes():
    diamond = [(10, 0), (20, 10), (10, 20), (0, 10), (10, 0)]
    build = AerialImageryExtractor._outline_geometry

    assert build(diamond, True, 'geometry').equals(Polygon(diamond))
    assert build(diamond, True, 'bbox').equals(box(0, 0, 20, 20))
    # The rotated rectangle of a diamond is the diamond itself:
    assert build(diamond, True, 'rotated_bbox').area == pytest.approx(200)
    assert build(diamond, True, 'corners').area == pytest.approx(200)
    assert build(diamond, True, 'convex_hull').equals(Polygon(diamond))

    l_shape = [(0, 0), (20, 0), (20, 10), (10, 10), (10, 20), (0, 20), (0, 0)]
    assert build(l_shape, True, 'convex_hull').area == pytest.approx(350)
    assert build(l_shape, True, 'geometry').area == pytest.approx(300)

    # Lines and points:
    assert build([(0, 0), (10, 10)], False, 'geometry').geom_type == 'LineString'
    assert build([(0, 0), (10, 10)], False, 'bbox').equals(box(0, 0, 10, 10))
    assert build([(5, 5)], False, 'rotated_bbox').geom_type == 'Point'

    # A self-crossing ring is repaired rather than raising:
    bowtie = [(0, 0), (10, 10), (10, 0), (0, 10), (0, 0)]
    assert not build(bowtie, True, 'geometry').is_empty
    # A ring that collapsed onto a line still yields a drawable geometry:
    flat = [(0, 0), (10, 0), (20, 0), (0, 0)]
    assert not build(flat, True, 'geometry').is_empty


def test_aerial_outline_shape_changes_drawn_footprint(raster, tmp_path):
    counts = {}
    for shape in ('geometry', 'bbox', 'rotated_bbox', 'corners'):
        path = _extract_one(
            raster,
            tmp_path / shape,
            _diamond(CX, CY),
            outline_shape=shape,
            outline_width=3,
        )
        counts[shape] = int(_color_mask(path, 'red').sum())
    assert all(c > 0 for c in counts.values())
    # A diamond's axis-aligned box has a longer perimeter than the diamond:
    assert counts['bbox'] > counts['geometry']
    # ... whereas its rotated box coincides with it:
    assert counts['rotated_bbox'] < counts['bbox']

    hull = _extract_one(
        raster,
        tmp_path / 'hull',
        _l_shape(CX, CY),
        outline_shape='convex_hull',
        outline_width=3,
    )
    geom = _extract_one(raster, tmp_path / 'geom', _l_shape(CX, CY), outline_width=3)
    # The hull cuts across the notch, so it is shorter than the L itself:
    assert _color_mask(hull, 'red').sum() < _color_mask(geom, 'red').sum()


def test_aerial_corner_brackets_leave_edges_open():
    from PIL import ImageDraw

    def drawn(brackets):
        img = Image.new('RGB', (100, 100), 'black')
        AerialImageryExtractor._draw_geometry(
            ImageDraw.Draw(img),
            box(20, 20, 80, 60),
            width=3,
            color='red',
            brackets=brackets,
        )
        return np.asarray(img)[..., 0] > 0

    ring, brackets = drawn(False), drawn(True)
    assert 0 < brackets.sum() < ring.sum()
    # The middle of the top edge is stroked by the ring but not the brackets:
    assert ring[19:23, 50].any() and not brackets[15:25, 45:55].any()
    # Every corner is stroked by both:
    for x, y in ((20, 20), (80, 20), (80, 60), (20, 60)):
        assert brackets[y - 2 : y + 3, x - 2 : x + 3].any()


def _red_extent(path):
    """Width in pixels of the red pixels' bounding box."""
    cols = np.where(_color_mask(path, 'red').any(axis=0))[0]
    return int(cols.max() - cols.min() + 1)


@pytest.mark.parametrize('buffer_spec, min_growth', [(5, 8), ('4 m', 2), ('50%', 2)])
def test_aerial_outline_buffer_expands_outline(
    raster, tmp_path, buffer_spec, min_growth
):
    plain = _extract_one(raster, tmp_path / 'plain', _square(CX, CY), outline_width=2)
    buffered = _extract_one(
        raster,
        tmp_path / 'buffered',
        _square(CX, CY),
        outline_width=2,
        outline_buffer=buffer_spec,
    )
    assert _red_extent(buffered) >= _red_extent(plain) + min_growth


def test_aerial_outline_width_controls_stroke(raster, tmp_path):
    thin = _extract_one(raster, tmp_path / 'thin', _square(CX, CY), outline_width=2)
    thick = _extract_one(raster, tmp_path / 'thick', _square(CX, CY), outline_width=6)
    pct = _extract_one(raster, tmp_path / 'pct', _square(CX, CY), outline_width='10%')
    n_thin = _color_mask(thin, 'red').sum()
    n_thick = _color_mask(thick, 'red').sum()
    assert 0 < n_thin < n_thick
    # 10% of a ~30 px crop is ~3 px, between the two absolute widths:
    assert n_thin < _color_mask(pct, 'red').sum() < n_thick


def test_aerial_outline_color(raster, tmp_path):
    plain_col = PhysicalAssetCollection(
        [PhysicalAsset(id='a', geometry=_square(CX, CY))]
    )
    AerialImageryExtractor(raster, tmp_path / 'plain', buffer_asset='20 m')(plain_col)
    plain = plain_col['a'].image_assets[0].path
    blue = _extract_one(
        raster, tmp_path / 'blue', _square(CX, CY), outline_color='blue'
    )
    hex_blue = _extract_one(
        raster, tmp_path / 'hex', _square(CX, CY), outline_color='#0000ff'
    )
    rgb_blue = _extract_one(
        raster, tmp_path / 'rgb', _square(CX, CY), outline_color=(0, 0, 255)
    )
    for path in (blue, hex_blue, rgb_blue):
        assert _color_mask(path, 'blue').sum() > _color_mask(plain, 'blue').sum()
        assert _color_mask(path, 'red').sum() <= _color_mask(plain, 'red').sum() + 2


def test_aerial_outline_point_buffer_draws_ring(raster, tmp_path):
    dot = _extract_one(
        raster, tmp_path / 'dot', Point(CX, CY), buffer_asset='30 m', outline_width=2
    )
    ring = _extract_one(
        raster,
        tmp_path / 'ring',
        Point(CX, CY),
        buffer_asset='30 m',
        outline_width=2,
        outline_buffer=8,
    )
    dot_mask, ring_mask = _color_mask(dot, 'red'), _color_mask(ring, 'red')
    assert dot_mask.sum() > 0 and ring_mask.sum() > 0
    assert _red_extent(ring) > _red_extent(dot)
    # The dot covers the asset location; the ring leaves it visible:
    h, w = ring_mask.shape
    assert dot_mask[h // 2, w // 2]
    assert not ring_mask[h // 2, w // 2]


def test_aerial_outline_handles_multipart_and_invalid_geometries(raster, tmp_path):
    multi = MultiPolygon([_square(CX, CY), _square(CX + 0.00015, CY + 0.00015)])
    bowtie = Polygon(
        [
            (CX - 0.0001, CY - 0.0001),
            (CX + 0.0001, CY + 0.0001),
            (CX + 0.0001, CY - 0.0001),
            (CX - 0.0001, CY + 0.0001),
        ]
    )
    for name, geom, shape in (
        ('multi', multi, 'rotated_bbox'),
        ('bowtie', bowtie, 'geometry'),
        ('bowtie_corners', bowtie, 'corners'),
    ):
        path = _extract_one(raster, tmp_path / name, geom, outline_shape=shape)
        assert _color_mask(path, 'red').sum() > 0


def test_aerial_outline_disabled_ignores_options(raster, tmp_path):
    col = PhysicalAssetCollection([PhysicalAsset(id='a', geometry=_square(CX, CY))])
    AerialImageryExtractor(
        raster, tmp_path / 'off', buffer_asset='20 m', outline_shape='bbox'
    )(col)
    assert _color_mask(col['a'].image_assets[0].path, 'red').sum() == 0


def test_aerial_image_prefix_and_ids(raster, tmp_path):
    """image_prefix drives both the file name and the ImageAsset id."""
    col = PhysicalAssetCollection([PhysicalAsset(id='b1', geometry=_square(CX, CY))])
    AerialImageryExtractor(raster, tmp_path / 'o', image_prefix='post')(col)
    img = col['b1'].image_assets[0]
    assert img.id == 'b1_post'
    assert img.path.name.startswith('post_')
    assert img.path.suffix == '.jpg'


def test_aerial_keep_multiple_copies(raster, tmp_path):
    """With keep_multiple_copies, a second asset at the same spot gets _1."""
    out = tmp_path / 'o'
    geom = _square(CX, CY)
    col = PhysicalAssetCollection(
        [
            PhysicalAsset(id='a', geometry=geom),
            PhysicalAsset(id='b', geometry=geom),
        ]
    )
    AerialImageryExtractor(raster, out, keep_multiple_copies=True)(col)
    img_a = col['a'].image_assets[0]
    img_b = col['b'].image_assets[0]
    assert img_a.id == 'a_aerial'
    assert img_b.id == 'b_aerial_1'
    assert img_b.path.name.endswith('_1.jpg')
    assert img_a.path != img_b.path
    assert len(list(out.glob('*.jpg'))) == 2


def test_aerial_overwrites_without_keep_multiple_copies(raster, tmp_path):
    """Without keep_multiple_copies, co-located assets share one file."""
    out = tmp_path / 'o'
    geom = _square(CX, CY)
    col = PhysicalAssetCollection(
        [
            PhysicalAsset(id='a', geometry=geom),
            PhysicalAsset(id='b', geometry=geom),
        ]
    )
    AerialImageryExtractor(raster, out)(col)
    assert col['a'].image_assets[0].path == col['b'].image_assets[0].path
    assert len(list(out.glob('*.jpg'))) == 1


def test_aerial_skips_empty_geometry(raster, tmp_path):
    """Assets with empty geometries are skipped silently."""
    col = PhysicalAssetCollection(
        [
            PhysicalAsset(id='empty', geometry=Polygon()),
            PhysicalAsset(id='ok', geometry=_square(CX, CY)),
        ]
    )
    AerialImageryExtractor(raster, tmp_path / 'o')(col)
    assert len(col['empty'].image_assets) == 0
    assert len(col['ok'].image_assets) == 1


def test_aerial_geometry_collection_uses_convex_hull(raster, tmp_path):
    """GeometryCollections are wrapped in their convex hull before cropping."""
    gc = GeometryCollection([Point(CX, CY), _square(CX + 0.0001, CY, 0.00002)])
    col = PhysicalAssetCollection([PhysicalAsset(id='gc', geometry=gc)])
    AerialImageryExtractor(raster, tmp_path / 'o')(col)
    assert len(col['gc'].image_assets) == 1


def test_aerial_multiple_rasters_utm_and_wgs84(raster, utm_raster, tmp_path):
    """Assets are matched against each raster's WGS84 extent."""
    utm_asset = PhysicalAsset(id='utm', geometry=_square(-116.99965, 34.04398, 0.00003))
    col = PhysicalAssetCollection(
        [PhysicalAsset(id='poly', geometry=_square(CX, CY)), utm_asset]
    )
    AerialImageryExtractor([raster, utm_raster], tmp_path / 'o')(col)
    assert col['poly'].image_assets[0].properties['source_raster'] == 'ortho.tif'
    assert col['utm'].image_assets[0].properties['source_raster'] == 'utm.tiff'


def test_aerial_no_rasters_found_warns(tmp_path, collection, caplog):
    """An empty directory logs a warning and returns the collection unchanged."""
    empty = tmp_path / 'empty'
    empty.mkdir()
    ext = AerialImageryExtractor(empty, tmp_path / 'o')
    with caplog.at_level(logging.WARNING):
        result = ext(collection)
    assert result is collection
    assert 'No raster files found' in caplog.text
    assert (tmp_path / 'o').is_dir()


def test_aerial_no_assets_in_bounds(raster, tmp_path, caplog):
    """A raster without any intersecting assets is skipped with an info log."""
    col = PhysicalAssetCollection(
        [PhysicalAsset(id='far', geometry=_square(-118.30, 34.10))]
    )
    with caplog.at_level(logging.INFO):
        AerialImageryExtractor(raster, tmp_path / 'o')(col)
    assert 'No assets fall within the bounds' in caplog.text


def test_aerial_bad_raster_is_logged_and_skipped(raster, tmp_path, caplog):
    """A raster that fails to open is logged and does not stop other rasters."""
    bad = tmp_path / 'bad.tif'
    bad.write_bytes(b'not a tiff')
    col = PhysicalAssetCollection([PhysicalAsset(id='poly', geometry=_square(CX, CY))])
    with caplog.at_level(logging.ERROR):
        AerialImageryExtractor([bad, raster], tmp_path / 'o')(col)
    assert 'Failed to process raster bad.tif' in caplog.text
    assert len(col['poly'].image_assets) == 1


def test_aerial_cancelled_raises(raster, collection, tmp_path):
    """A set cancel event raises OperationCancelled instead of being swallowed."""
    event = threading.Event()
    event.set()
    ext = AerialImageryExtractor(raster, tmp_path / 'o', cancel_event=event)
    with pytest.raises(OperationCancelled, match='aerial image extraction'):
        ext(collection)
    assert all(len(a.image_assets) == 0 for a in collection)


# ==========================================
# 4. BingOrthomosaicExtractor
# ==========================================


def test_bing_init_defaults():
    """Default configuration is zoom 19, 10 workers, no cancel event."""
    ext = BingOrthomosaicExtractor()
    assert ext.zoom_level == 19
    assert ext.max_workers == 10
    assert ext.cancel_event is None


def test_bing_lat_lon_to_pixel_known_values():
    """The origin maps to the map centre; extremes are clamped."""
    assert BingOrthomosaicExtractor.lat_lon_to_pixel(0.0, 0.0, 1) == (256, 256)
    assert BingOrthomosaicExtractor.lat_lon_to_pixel(0.0, -180.0, 1) == (0, 256)
    px, py = BingOrthomosaicExtractor.lat_lon_to_pixel(80.0, 179.9, 3)
    assert 0 <= px < 2048 and 0 <= py < 2048
    _, py_south = BingOrthomosaicExtractor.lat_lon_to_pixel(-80.0, 0.0, 3)
    assert py_south > py
    # Latitudes beyond the Mercator limit are clamped instead of overflowing:
    _, py_pole = BingOrthomosaicExtractor.lat_lon_to_pixel(90.0, 0.0, 3)
    _, py_near = BingOrthomosaicExtractor.lat_lon_to_pixel(89.5, 0.0, 3)
    assert py_pole == py_near


def test_bing_tile_to_quadkey():
    """Quadkeys follow the Bing Maps tile system reference values."""
    assert BingOrthomosaicExtractor.tile_to_quadkey(3, 5, 3) == '213'
    assert BingOrthomosaicExtractor.tile_to_quadkey(0, 0, 1) == '0'
    assert BingOrthomosaicExtractor.tile_to_quadkey(1, 1, 1) == '3'
    assert BingOrthomosaicExtractor.tile_to_quadkey(0, 0, 0) == ''


def test_bing_stitches_region_into_geotiff(tmp_path):
    """Mocked tiles are stitched into a georeferenced 3-band GeoTIFF."""
    region = BoundingBox(min_x=-118.5, min_y=34.0, max_x=-117.5, max_y=34.5)
    ext = BingOrthomosaicExtractor(zoom_level=8, max_workers=2)
    out = tmp_path / 'sub' / 'mosaic'

    with requests_mock.Mocker() as m:
        m.get(TILE_URL, content=_jpeg_bytes((200, 30, 30)))
        path = ext(region, out)
        requested = {r.url for r in m.request_history}

    assert path == (tmp_path / 'sub' / 'mosaic.tiff').resolve()
    assert path.exists()
    assert len(requested) >= 1
    assert all('virtualearth.net/tiles/a' in u for u in requested)

    with rasterio.open(path) as src:
        assert src.count == 3
        assert src.crs.to_epsg() == 4326
        b = src.bounds
        assert (b.left, b.bottom, b.right, b.top) == pytest.approx(
            region.bounds, abs=1e-9
        )
        data = src.read()
        assert data.shape[1] > 0 and data.shape[2] > 0
        # Every pixel came from a red tile:
        assert data[0].min() > 150
        assert data[1].max() < 100


def test_bing_accepts_polygon_region_and_tif_suffix(tmp_path):
    """PolygonRegion inputs work and an explicit .tif suffix is preserved."""
    region = PolygonRegion(list(box(-118.5, 34.0, -117.5, 34.5).exterior.coords))
    ext = BingOrthomosaicExtractor(zoom_level=7)
    with requests_mock.Mocker() as m:
        m.get(TILE_URL, content=_jpeg_bytes())
        path = ext(region, tmp_path / 'area.tif')
    assert path.name == 'area.tif'
    assert path.exists()


def test_bing_failed_tiles_left_black(tmp_path, caplog):
    """HTTP errors and corrupt payloads are logged; the mosaic is still written."""
    region = BoundingBox(min_x=-118.5, min_y=34.0, max_x=-117.5, max_y=34.5)
    ext = BingOrthomosaicExtractor(zoom_level=8)
    with requests_mock.Mocker() as m:
        m.get(TILE_URL, status_code=404)
        with caplog.at_level(logging.WARNING):
            path = ext(region, tmp_path / 'black.tiff')
    assert 'Network error fetching tile' in caplog.text
    with rasterio.open(path) as src:
        assert src.read().max() == 0

    caplog.clear()
    with requests_mock.Mocker() as m:
        m.get(TILE_URL, content=b'definitely-not-a-jpeg')
        with caplog.at_level(logging.WARNING):
            ext(region, tmp_path / 'corrupt.tiff')
    assert 'Failed to parse image data' in caplog.text


def test_bing_connection_error_is_handled(tmp_path, caplog):
    """A transport-level exception is treated like any other request failure."""
    region = BoundingBox(min_x=-118.5, min_y=34.0, max_x=-117.5, max_y=34.5)
    ext = BingOrthomosaicExtractor(zoom_level=7)
    with requests_mock.Mocker() as m:
        m.get(TILE_URL, exc=requests.ConnectionError('offline'))
        with caplog.at_level(logging.WARNING):
            path = ext(region, tmp_path / 'offline.tiff')
    assert 'Network error' in caplog.text
    assert path.exists()


def test_bing_degenerate_region_raises(tmp_path):
    """A region smaller than one pixel at the chosen zoom is rejected."""
    region = BoundingBox(min_x=-118.0, min_y=34.0, max_x=-117.9999, max_y=34.0001)
    with pytest.raises(ValueError, match='collapses'):
        BingOrthomosaicExtractor(zoom_level=1)(region, tmp_path / 'x.tiff')


def test_bing_cancelled_raises(tmp_path):
    """A set cancel event aborts the download and no GeoTIFF is written."""
    region = BoundingBox(min_x=-118.5, min_y=34.0, max_x=-117.5, max_y=34.5)
    event = threading.Event()
    event.set()
    ext = BingOrthomosaicExtractor(zoom_level=8, cancel_event=event)
    with requests_mock.Mocker() as m:
        m.get(TILE_URL, content=_jpeg_bytes())
        with pytest.raises(OperationCancelled, match='Bing tile download'):
            ext(region, tmp_path / 'cancelled.tiff')
    assert not (tmp_path / 'cancelled.tiff').exists()


# ==========================================
# 5. MapillaryImageExtractor
# ==========================================


class FakeAdapter:
    pool_connections = 1
    pool_maxsize = 1


class FakeSession:
    def __init__(self):
        self.adapters = {'http://': FakeAdapter(), 'https://': FakeAdapter()}


class FakePano:
    """Minimal stand-in for the ImageAsset returned by MapillaryClient.fetch_image."""

    def __init__(self, pano_id, labels=None, fail_load=False):
        self.id = pano_id
        self.properties = {}
        self.semantic_map = (
            {i: label for i, label in enumerate(labels)} if labels else None
        )
        self.fail_load = fail_load
        self.loaded = False

    def load_image_from_url(self):
        if self.fail_load:
            raise RuntimeError('download failed')
        self.loaded = True


class FakeMapillaryClient:
    """Records calls and serves canned regional images / panoramas."""

    def __init__(self, access_token, save_dir):
        if not access_token:
            raise ValueError('Mapillary access token is required.')
        self.access_token = access_token
        self.save_dir = save_dir
        self.session = FakeSession()
        self.bbox_calls: list[dict] = []
        self.fetch_calls: list[dict] = []
        self.regional = ImageCollection(
            [
                ImageAsset(
                    id='pano_1',
                    path='/nonexistent/pano_1.jpg',
                    allow_missing_file=True,
                    properties={
                        'longitude': CX + 0.0003,
                        'latitude': CY,
                        'compass_angle': 270.0,
                    },
                ),
                ImageAsset(
                    id='pano_2',
                    path='/nonexistent/pano_2.jpg',
                    allow_missing_file=True,
                    properties={
                        'longitude': CX,
                        'latitude': CY + 0.0003,
                        'compass_angle': 180.0,
                    },
                ),
            ]
        )
        self.panos = {
            'pano_1': FakePano('pano_1', labels=['construction--structure--building']),
            'pano_2': FakePano('pano_2', labels=['nature--sky']),
        }

    def fetch_images_in_bbox(self, **kwargs):
        self.bbox_calls.append(kwargs)
        return self.regional

    def fetch_image(self, image_id, **kwargs):
        self.fetch_calls.append({'id': image_id, **kwargs})
        return self.panos.get(image_id)


class FakeLabelMapper:
    def __init__(self):
        self.calls = []

    def map_classes(self, classes):
        self.calls.append(list(classes))
        return ['object--support--utility-pole']


@pytest.fixture
def fake_mapillary(monkeypatch):
    """Patch the client and the geometry heavy pano helpers."""
    holder = {}

    def client_factory(access_token, save_dir):
        inst = FakeMapillaryClient(access_token, save_dir)
        holder['client'] = inst
        return inst

    holder['best_panos_calls'] = []

    def fake_find_best_panos(target_asset_wgs84, pano_collection, **kwargs):
        holder['best_panos_calls'].append(kwargs)
        panos = {p.id: p for p in pano_collection}
        return {
            'corner_0': panos.get('pano_2'),
            'minor_axis_0': None,
            'major_axis_0': panos.get('pano_1'),
        }

    holder['crop_calls'] = []

    def fake_crop(target_asset, pano_image, vertical_crop_mode):
        holder['crop_calls'].append(
            (target_asset.id, pano_image.id, vertical_crop_mode)
        )
        return Image.new('RGB', (16, 8), color=(1, 2, 3))

    monkeypatch.setattr(image_extractors, 'MapillaryClient', client_factory)
    monkeypatch.setattr(image_extractors, 'find_best_panos', fake_find_best_panos)
    monkeypatch.setattr(image_extractors, 'crop_panorama_to_asset', fake_crop)
    return holder


@pytest.fixture
def street_collection():
    return PhysicalAssetCollection(
        [
            PhysicalAsset(
                id='b1', geometry=_square(CX, CY), attributes={'asset_type': 'building'}
            ),
            PhysicalAsset(
                id='p1',
                geometry=_square(CX + 0.0001, CY, 0.00001),
                attributes={'asset_type': 'pole'},
            ),
        ]
    )


def test_mapillary_init(fake_mapillary, tmp_path):
    """Configuration is stored and a client is built with the token."""
    mapper = FakeLabelMapper()
    ext = MapillaryImageExtractor(
        access_token='tok',
        save_directory=tmp_path / 'sv',
        start_date='2024-01-01',
        end_date='2024-12-31',
        filter_rapid_only=False,
        cast_corner_rays=True,
        smart_crop=False,
        strict_content_filter=True,
        label_mapper=mapper,
        asset_type_mapping={'building': ['x']},
        image_prefix='sv',
        max_images_per_asset=3,
        max_workers=2,
    )
    assert ext.save_directory == (tmp_path / 'sv').resolve()
    assert ext.start_date == '2024-01-01'
    assert ext.end_date == '2024-12-31'
    assert ext.filter_rapid_only is False
    assert ext.cast_corner_rays is True
    assert ext.smart_crop is False
    assert ext.strict_content_filter is True
    assert ext.label_mapper is mapper
    assert ext.asset_type_mapping == {'building': ['x']}
    assert ext.image_prefix == 'sv'
    assert ext.max_images_per_asset == 3
    assert ext.max_workers == 2
    assert ext.cancel_event is None
    assert fake_mapillary['client'].access_token == 'tok'


def test_mapillary_init_requires_token(fake_mapillary, tmp_path):
    """An empty access token raises ValueError from the client."""
    with pytest.raises(ValueError, match='access token'):
        MapillaryImageExtractor(access_token='', save_directory=tmp_path)


def test_mapillary_extracts_and_attaches_crops(
    fake_mapillary, street_collection, tmp_path
):
    """Valid panoramas are fetched, cropped, saved and attached in priority order."""
    out = tmp_path / 'sv'
    ext = MapillaryImageExtractor('tok', out, max_workers=1)
    result = ext(street_collection)
    assert result is street_collection

    client = fake_mapillary['client']
    assert len(client.bbox_calls) == 1
    assert client.bbox_calls[0]['filter_rapid_only'] is True
    assert client.bbox_calls[0]['save_to_disk'] is False
    assert isinstance(client.bbox_calls[0]['bbox'], BoundingBox)

    # smart_crop=True -> semantic masks requested for every fetch:
    assert all(c['process_masks'] == ['semantic'] for c in client.fetch_calls)

    for asset in street_collection:
        ids = [img.id for img in asset.image_assets]
        # Major axis first, then the corner (minor axis had no pano):
        assert ids == [f'{asset.id}_major_axis_0', f'{asset.id}_corner_0']
        for img in asset.image_assets:
            assert img.path.exists()
            assert (
                img.path.name == f'street_{asset.id}_{img.properties["view_angle"]}.jpg'
            )
            assert img.properties['original_pano_id'] in ('pano_1', 'pano_2')

    # Pano properties are copied from the compact metadata before cropping:
    assert client.panos['pano_1'].properties['compass_angle'] == 270.0
    assert client.panos['pano_1'].loaded is True
    assert all(mode == 'smart' for _, _, mode in fake_mapillary['crop_calls'])
    # 90 degree interval without corner rays:
    assert all(c['interval_deg'] == 90.0 for c in fake_mapillary['best_panos_calls'])


def test_mapillary_full_crop_mode_and_corner_rays(
    fake_mapillary, street_collection, tmp_path
):
    """smart_crop=False disables masks and uses 'full' crops; corners use 45 degrees."""
    ext = MapillaryImageExtractor(
        'tok', tmp_path / 'sv', smart_crop=False, cast_corner_rays=True, max_workers=1
    )
    ext(street_collection)
    client = fake_mapillary['client']
    assert all(c['process_masks'] is None for c in client.fetch_calls)
    assert all(mode == 'full' for _, _, mode in fake_mapillary['crop_calls'])
    assert all(c['interval_deg'] == 45.0 for c in fake_mapillary['best_panos_calls'])
    assert all(
        c['cast_corner_rays'] is True for c in fake_mapillary['best_panos_calls']
    )


def test_mapillary_max_images_per_asset(fake_mapillary, street_collection, tmp_path):
    """No more than max_images_per_asset crops are attached."""
    ext = MapillaryImageExtractor('tok', tmp_path / 'sv', max_images_per_asset=1)
    ext(street_collection)
    for asset in street_collection:
        assert [img.id for img in asset.image_assets] == [f'{asset.id}_major_axis_0']


def test_mapillary_strict_filter_with_mapping(
    fake_mapillary, street_collection, tmp_path
):
    """Strict filtering skips panoramas that lack the mapped labels."""
    ext = MapillaryImageExtractor(
        'tok',
        tmp_path / 'sv',
        strict_content_filter=True,
        asset_type_mapping={
            'building': ['construction--structure--building'],
            'pole': ['object--support--utility-pole'],
        },
        max_workers=1,
    )
    ext(street_collection)
    # Building: only pano_1 carries the building label.
    assert [i.id for i in street_collection['b1'].image_assets] == ['b1_major_axis_0']
    # Pole: neither pano has the pole label.
    assert len(street_collection['p1'].image_assets) == 0


def test_mapillary_strict_filter_uses_label_mapper(
    fake_mapillary, street_collection, tmp_path
):
    """Unmapped asset types are resolved through the label mapper."""
    mapper = FakeLabelMapper()
    ext = MapillaryImageExtractor(
        'tok',
        tmp_path / 'sv',
        strict_content_filter=True,
        label_mapper=mapper,
        asset_type_mapping={'building': ['construction--structure--building']},
        max_workers=1,
    )
    ext(street_collection)
    assert mapper.calls == [['pole']]
    assert ext.asset_type_mapping['pole'] == ['object--support--utility-pole']


def test_mapillary_strict_filter_without_labels_keeps_all(fake_mapillary, tmp_path):
    """An asset type with no mapping is not filtered at all."""
    col = PhysicalAssetCollection(
        [
            PhysicalAsset(
                id='u', geometry=_square(CX, CY), attributes={'asset_type': 'tree'}
            )
        ]
    )
    ext = MapillaryImageExtractor(
        'tok', tmp_path / 'sv', strict_content_filter=True, max_workers=1
    )
    ext(col)
    assert len(col['u'].image_assets) == 2


def test_mapillary_no_regional_images_warns(
    fake_mapillary, street_collection, tmp_path, caplog
):
    """An empty regional query returns the collection with a warning."""
    ext = MapillaryImageExtractor('tok', tmp_path / 'sv')
    fake_mapillary['client'].regional = ImageCollection()
    with caplog.at_level(logging.WARNING):
        result = ext(street_collection)
    assert result is street_collection
    assert 'No Mapillary images found' in caplog.text
    assert all(len(a.image_assets) == 0 for a in street_collection)


def test_mapillary_missing_pano_and_errors_are_skipped(
    fake_mapillary, street_collection, tmp_path, caplog
):
    """A None pano is skipped; a failing download is logged and skipped."""
    ext = MapillaryImageExtractor('tok', tmp_path / 'sv', max_workers=1)
    client = fake_mapillary['client']
    client.panos['pano_1'] = None
    client.panos['pano_2'].fail_load = True
    with caplog.at_level(logging.ERROR):
        ext(street_collection)
    assert 'Failed to process pano pano_2' in caplog.text
    assert all(len(a.image_assets) == 0 for a in street_collection)


def test_mapillary_skips_empty_geometry(fake_mapillary, tmp_path):
    """Assets with empty geometries produce no crops."""
    col = PhysicalAssetCollection(
        [
            PhysicalAsset(id='e', geometry=Polygon()),
            PhysicalAsset(id='ok', geometry=_square(CX, CY)),
        ]
    )
    ext = MapillaryImageExtractor('tok', tmp_path / 'sv', max_workers=1)
    ext(col)
    assert len(col['e'].image_assets) == 0
    assert len(col['ok'].image_assets) == 2


def test_mapillary_cancelled_raises(fake_mapillary, street_collection, tmp_path):
    """A set cancel event propagates OperationCancelled from the worker threads."""
    event = threading.Event()
    event.set()
    ext = MapillaryImageExtractor('tok', tmp_path / 'sv', cancel_event=event)
    with pytest.raises(OperationCancelled, match='Mapillary image extraction'):
        ext(street_collection)
    assert all(len(a.image_assets) == 0 for a in street_collection)


# ==========================================
# Edge assets: min_footprint_coverage
# ==========================================


def test_aerial_extractor_min_footprint_coverage(tmp_path, caplog):
    """Partially imaged footprints are kept and padded; nodata-corner ones skipped."""
    import numpy as np
    from rasterio.transform import from_bounds
    from shapely.geometry import box

    from rapidtools.core import PhysicalAsset, PhysicalAssetCollection
    from rapidtools.processing import AerialImageryExtractor

    raster = tmp_path / 'edge.tif'
    data = np.full((3, 40, 40), 150, dtype='uint8')
    data[:, :, :20] = 0  # left half is nodata (like a rotated ortho's corner)
    with rasterio.open(
        raster,
        'w',
        driver='GTiff',
        height=40,
        width=40,
        count=3,
        dtype='uint8',
        crs='EPSG:4326',
        transform=from_bounds(0, 0, 0.04, 0.04, 40, 40),
        nodata=0,
    ) as dst:
        dst.write(data)

    def collection():
        col = PhysicalAssetCollection()
        col.add(PhysicalAsset(id='inside', geometry=box(0.025, 0.010, 0.030, 0.015)))
        col.add(PhysicalAsset(id='edge', geometry=box(0.018, 0.020, 0.024, 0.026)))
        col.add(PhysicalAsset(id='corner', geometry=box(0.002, 0.002, 0.008, 0.008)))
        col.add(PhysicalAsset(id='outside', geometry=box(0.10, 0.10, 0.11, 0.11)))
        return col

    # Default behaviour: the edge asset is dropped by the patch nodata test.
    with caplog.at_level(logging.INFO):
        default = AerialImageryExtractor(raster, save_directory=tmp_path / 'a')(
            collection()
        )
    assert [a.id for a in default if a.image_assets] == ['inside']
    assert 'assets intersecting the raster' in caplog.text
    assert '2 of 3 candidate assets had no usable imagery' in caplog.text

    # Coverage mode keeps the half-imaged edge asset, still skips the corner one.
    kept = AerialImageryExtractor(
        raster,
        save_directory=tmp_path / 'b',
        min_footprint_coverage=0.3,
        buffer_asset='50%',
    )(collection())
    assert sorted(a.id for a in kept if a.image_assets) == ['edge', 'inside']
    edge_img = next(iter(kept.get('edge').image_assets))
    # Padded crop: the WGS84 bounds extend into the nodata half as requested.
    assert edge_img.properties['wgs84_bounds'][0] < 0.018

    with pytest.raises(ValueError, match='min_footprint_coverage'):
        AerialImageryExtractor(
            raster, save_directory=tmp_path, min_footprint_coverage=1.5
        )
