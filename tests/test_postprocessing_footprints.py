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
# 11-27-2026

import math

import numpy as np
import pytest
from rasterio import features as rio_features
from rasterio.transform import from_origin
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box, shape
from shapely.ops import transform

from rapidtools.core import PhysicalAsset, PhysicalAssetCollection
from rapidtools.processing import FootprintRegularizer
from rapidtools.processing.postprocessing.footprints import (
    dominant_orientation,
    iou,
    local_metric_transformers,
    oriented_rectangle,
    regularize_footprint,
    regularize_footprints,
    resolve_overlaps,
)


def pixelate(geom, cell=0.25):
    """Trace a geometry the way a mask would: a stair-stepped outline."""
    minx, miny, maxx, maxy = geom.bounds
    width = int(math.ceil((maxx - minx) / cell)) + 4
    height = int(math.ceil((maxy - miny) / cell)) + 4
    tr = from_origin(minx - 2 * cell, maxy + 2 * cell, cell, cell)
    grid = rio_features.rasterize(
        [(geom, 1)], out_shape=(height, width), transform=tr, dtype=np.uint8
    )
    polys = [
        shape(g)
        for g, v in rio_features.shapes(grid, mask=grid.astype(bool), transform=tr)
        if v == 1
    ]
    return max(polys, key=lambda p: p.area)


def corner_cosines(poly):
    """|cos| of every corner angle of a polygon's exterior."""
    pts = np.asarray(poly.exterior.coords)[:-1]
    out = []
    for i in range(len(pts)):
        u, v = pts[i - 1] - pts[i], pts[(i + 1) % len(pts)] - pts[i]
        out.append(abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return out


# ----------------------------------------------------------- single outline


def test_rotated_rectangle_becomes_a_rectangle():
    """A stair-stepped, rotated house outline comes back with four square corners."""
    house = affinity.rotate(box(0, 0, 14, 9), 31, origin=(0, 0))
    result, info = regularize_footprint(pixelate(house), return_info=True)
    assert len(result.exterior.coords) - 1 == 4
    assert max(corner_cosines(result)) < 1e-6
    assert iou(result, house) > 0.95
    assert info['method'] == 'rectangle'
    assert math.degrees(info['orientation']) == pytest.approx(31, abs=1.0)


def test_l_shape_keeps_its_wing():
    """An L-shaped footprint keeps six walls at right angles."""
    ell = affinity.rotate(box(0, 0, 16, 8).union(box(0, 8, 7, 15)), -18, origin=(0, 0))
    result, info = regularize_footprint(pixelate(ell), return_info=True)
    assert info['method'] == 'rectilinear'
    assert len(result.exterior.coords) - 1 == 6
    assert max(corner_cosines(result)) < 1e-6
    assert iou(result, ell) > 0.93


def test_small_bumps_and_dents_are_removed():
    """A tree bulge and a shadow notch narrower than a meter disappear."""
    roof = box(0, 0, 18, 10)
    noisy = roof.union(box(6, 10, 6.6, 10.8)).difference(box(12, -0.1, 12.5, 0.7))
    result = regularize_footprint(pixelate(noisy, 0.2))
    assert len(result.exterior.coords) - 1 == 4
    assert iou(result, roof) > 0.97


def test_orientation_hint_wins_ties():
    """A direction shared by the neighbors wins when it fits as well."""
    house = affinity.rotate(box(0, 0, 12, 8), 20, origin=(0, 0))
    hinted, info = regularize_footprint(
        pixelate(house), orientation_hints=(math.radians(20.1),), return_info=True
    )
    assert math.degrees(info['orientation']) == pytest.approx(20.1, abs=1e-6)
    assert iou(hinted, house) > 0.95


def test_round_shapes_are_only_simplified():
    """A round building does not get forced into a rectangle."""
    tank = Point(0, 0).buffer(8, quad_segs=32)
    result, info = regularize_footprint(tank, return_info=True)
    assert info['method'] == 'simplified'
    assert iou(result, tank) > 0.95


def test_holes_small_filled_large_kept():
    """Small holes (trees, vents) are filled; a courtyard survives."""
    outer = box(0, 0, 40, 30)
    vent = box(5, 5, 6, 6)
    courtyard = box(15, 10, 25, 20)
    result = regularize_footprint(
        Polygon(outer.exterior.coords, [vent.exterior.coords])
    )
    assert len(result.interiors) == 0
    with_court = regularize_footprint(
        Polygon(outer.exterior.coords, [courtyard.exterior.coords])
    )
    assert len(with_court.interiors) == 1
    assert with_court.area == pytest.approx(outer.area - courtyard.area, rel=0.02)


def test_degenerate_inputs_give_none():
    """Empty and zero-area inputs return ``None``."""
    assert regularize_footprint(Polygon()) is None
    assert regularize_footprint(LineString([(0, 0), (1, 1)]).buffer(0)) is None
    result, info = regularize_footprint(Polygon(), return_info=True)
    assert result is None and info['method'] is None


def test_dominant_orientation_and_oriented_rectangle():
    """The wall direction of a rotated box and its oriented rectangle."""
    rotated = affinity.rotate(box(0, 0, 20, 10), 30, origin=(0, 0))
    theta, strength = dominant_orientation(rotated)
    assert math.degrees(theta) == pytest.approx(30, abs=1e-6)
    assert strength == pytest.approx(1.0)
    assert oriented_rectangle(rotated, theta).area == pytest.approx(200.0)


# ------------------------------------------------------------- collections


def test_resolve_overlaps_priority_and_min_area():
    """Higher priority keeps the contested area; slivers and None pass through."""
    a, b = box(0, 0, 10, 10), box(8, 0, 18, 10)
    out = resolve_overlaps([a, b, None], priorities=[1.0, 2.0, 0.0])
    assert out[1].equals(b)
    assert out[0].bounds == pytest.approx((0, 0, 8, 10))
    assert out[2] is None
    # A footprint left smaller than min_area is dropped:
    tiny = box(9, 0, 10.5, 2)
    assert resolve_overlaps([a, tiny], min_area=4.0)[1] is None


def test_regularize_footprints_separates_neighbors():
    """Overlapping neighbors come back non-overlapping, each still square."""
    houses = [
        affinity.rotate(box(x, 0, x + 12, 8), 25, origin=(0, 0)) for x in (0, 15, 30)
    ]
    overlapping = affinity.translate(houses[1], 11.5, 0)
    out = regularize_footprints([*[pixelate(h) for h in houses], overlapping])
    assert all(p is not None for p in out)
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            assert out[i].intersection(out[j]).area < 1e-6
    # Cuts along a neighbor's wall keep the corners within a degree of square:
    for p in out:
        assert max(corner_cosines(p)) < math.sin(math.radians(1))


def test_neighbor_orientation_is_shared():
    """Every footprint is offered the wall direction of its neighbors."""
    from rapidtools.processing.postprocessing.footprints import _neighbor_orientations

    houses = [
        affinity.rotate(box(x, 0, x + 12, 8), 25, origin=(0, 0)) for x in (0, 15, 30)
    ]
    lonely = box(500, 500, 510, 508)
    hints = _neighbor_orientations([*houses, lonely, None], radius=40.0)
    assert all(math.degrees(h) == pytest.approx(25, abs=0.5) for h in hints[:3])
    assert hints[3] is None and hints[4] is None


def test_local_metric_transformers_round_trip():
    """WGS84 to the local plane and back returns the same coordinates."""
    to_local, to_wgs84 = local_metric_transformers(-118.14, 34.18)
    x, y = to_local(-118.1395, 34.1805)
    assert (x, y) == pytest.approx((46.05, 55.43), abs=0.5)
    lon, lat = to_wgs84(x, y)
    assert (lon, lat) == pytest.approx((-118.1395, 34.1805), abs=1e-9)


# ---------------------------------------------------- FootprintRegularizer


def _wgs84(poly, lon=-118.14, lat=34.18):
    _, to_wgs84 = local_metric_transformers(lon, lat)
    return transform(to_wgs84, poly)


def test_footprint_regularizer_keeps_ids_and_attributes():
    """Assets keep their IDs and attributes; points pass through untouched."""
    house = _wgs84(pixelate(affinity.rotate(box(0, 0, 14, 9), 31, origin=(0, 0))))
    shed = _wgs84(box(30, 30, 31, 31))  # 1 m^2: too small
    pole = PhysicalAsset(id='pole', geometry=Point(-118.14, 34.18))
    assets = PhysicalAssetCollection(
        [
            PhysicalAsset(id='h', geometry=house, attributes={'confidence': 0.9}),
            PhysicalAsset(id='s', geometry=shed),
            pole,
        ]
    )
    result = FootprintRegularizer(keep_original=True)(assets)
    assert {a.id for a in result} == {'h', 'pole'}
    assert result['h'].attributes['confidence'] == 0.9
    assert result['h'].attributes['unregularized_geometry_wkt'] == house.wkt
    assert len(result['h'].geometry.exterior.coords) == 5
    assert result['pole'].geometry.equals(Point(-118.14, 34.18))


def test_footprint_regularizer_without_polygons_is_a_no_op():
    """A collection without polygons is returned as is."""
    assets = PhysicalAssetCollection([PhysicalAsset(id='p', geometry=Point(0, 0))])
    assert FootprintRegularizer()(assets) is assets
