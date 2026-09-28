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

import logging
import math
from types import SimpleNamespace

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box  # noqa: E402
from shapely.strtree import STRtree  # noqa: E402

from rapidtools.core import (  # noqa: E402
    ImageAsset,
    ImageCollection,
    PhysicalAsset,
    PhysicalAssetCollection,
)
from rapidtools.processing import pano_utils as pu  # noqa: E402

# Scenario origin (Altadena, CA) and metre-to-degree conversion factors:
LON0, LAT0 = -118.14, 34.19
M_LAT = 1.0 / 111320.0
M_LON = 1.0 / (111320.0 * math.cos(math.radians(LAT0)))

# ==========================================
# Helpers & Fixtures
# ==========================================


def offset_deg(dx_m: float, dy_m: float) -> tuple[float, float]:
    """Return (lon, lat) displaced from the origin by metres east/north."""
    return LON0 + dx_m * M_LON, LAT0 + dy_m * M_LAT


def rect(cx_m: float, cy_m: float, w_m: float, h_m: float) -> Polygon:
    """Return a lon/lat rectangle centred at metric offsets from the origin."""
    x0, y0 = offset_deg(cx_m - w_m / 2, cy_m - h_m / 2)
    x1, y1 = offset_deg(cx_m + w_m / 2, cy_m + h_m / 2)
    return box(x0, y0, x1, y1)


def make_pano(pid: str, dx_m: float, dy_m: float, heading: float = 0.0) -> ImageAsset:
    """Create an ImageAsset located at metric offsets from the origin."""
    lon, lat = offset_deg(dx_m, dy_m)
    return ImageAsset(
        path=f'{pid}.jpg',
        id=pid,
        allow_missing_file=True,
        properties={'longitude': lon, 'latitude': lat, 'compass_angle': heading},
    )


@pytest.fixture(autouse=True)
def no_show(monkeypatch):
    """Never open matplotlib windows during tests and close figures after."""
    monkeypatch.setattr(plt, 'show', lambda *a, **k: None)
    yield
    plt.close('all')


@pytest.fixture
def target_asset():
    """A 20 m x 10 m east-west footprint centred on the origin."""
    return PhysicalAsset(id='target', geometry=rect(0, 0, 20, 10))


@pytest.fixture
def cardinal_panos():
    """Cameras 30 m east/west/north/south of the origin."""
    return ImageCollection(
        [
            make_pano('east', 30, 0),
            make_pano('west', -30, 0),
            make_pano('north', 0, 30),
            make_pano('south', 0, -30),
        ]
    )


def run_scenario(target, panos, extra_buildings=(), **kwargs):
    """Index the inputs and run find_best_panos with the given options."""
    buildings = PhysicalAssetCollection(
        [target]
        + [
            PhysicalAsset(id=f'nb{i}', geometry=g)
            for i, g in enumerate(extra_buildings)
        ]
    )
    b_tree, b_geoms = pu.build_footprint_index(buildings)
    p_tree, coords, _ = pu.index_panos(panos)
    return pu.find_best_panos(target, panos, b_tree, b_geoms, p_tree, coords, **kwargs)


# ==========================================
# 1. Projection & Angle Math
# ==========================================


def test_local_projection_scalar():
    """One thousandth of a degree north is about 111 metres."""
    project = pu.get_local_projection_func(0.0, 0.0)
    x, y = project(0.0, 0.001)
    assert abs(float(x)) < 1e-3
    assert round(float(y)) == 111


def test_local_projection_arrays_and_wrap():
    """Array inputs are supported and longitude deltas wrap at the antimeridian."""
    project = pu.get_local_projection_func(179.9, 0.0)
    x, y = project(np.array([-179.9, 179.9]), np.array([0.0, 0.0]))
    # -179.9 is only 0.2 degrees east of 179.9, not 359.8 degrees west:
    assert 20_000 < float(x[0]) < 25_000
    # float32 radians limit precision to roughly a metre at this longitude:
    assert abs(float(x[1])) < 1.0
    assert np.allclose(y, 0.0, atol=1e-3)


def test_local_projection_scales_with_latitude():
    """Longitude metres shrink with the cosine of the origin latitude."""
    equator = pu.get_local_projection_func(0.0, 0.0)
    high = pu.get_local_projection_func(0.0, 60.0)
    xe, _ = equator(0.01, 0.0)
    xh, _ = high(0.01, 60.0)
    assert float(xh) == pytest.approx(float(xe) * 0.5, rel=1e-3)


@pytest.mark.parametrize(
    'vec, expected',
    [((0.0, 1.0), 0.0), ((1.0, 0.0), 90.0), ((0.0, -1.0), 180.0), ((-1.0, 0.0), 270.0)],
)
def test_vec_to_angle(vec, expected):
    """Vectors map to compass angles (0 = +Y, clockwise positive)."""
    assert float(pu._vec_to_angle(vec)) == pytest.approx(expected)


def test_get_principal_axes_axis_aligned():
    """A wide rectangle has an east-west major axis and north-south minor axis."""
    major, minor = pu.get_principal_axes(box(0, 0, 10, 2))
    assert round(float(major) % 180) == 90
    assert round(float(minor) % 180) == 0


def test_get_principal_axes_tall_rectangle():
    """A tall rectangle swaps the roles of the two axes."""
    major, minor = pu.get_principal_axes(box(0, 0, 2, 10))
    assert round(float(major) % 180) == 0
    assert round(float(minor) % 180) == 90


def test_get_principal_axes_rotated():
    """A rotated rectangle reports its actual orientation."""
    diamond = Polygon([(0, 0), (10, 10), (8, 12), (-2, 2)])
    major, minor = pu.get_principal_axes(diamond)
    assert round(float(major) % 180) == 45
    assert round(float(minor) % 180) == 135


def test_transform_polygon_to_local_rotated():
    """Projection followed by a 90 degree rotation maps +X onto +Y."""
    project = pu.get_local_projection_func(LON0, LAT0)
    poly = rect(10, 0, 4, 2)  # centred 10 m east of the origin
    rotated = pu._transform_polygon_to_local_rotated(
        poly, project, math.cos(math.pi / 2), math.sin(math.pi / 2)
    )
    cx, cy = rotated.centroid.x, rotated.centroid.y
    # The projector works in float32, so allow metre-level slack:
    assert abs(cx) < 2.0 and cy == pytest.approx(10, abs=2.0)
    assert rotated.area == pytest.approx(8, abs=2.0)


@pytest.mark.parametrize(
    'lat2, lon2, expected',
    [(0.0, 10.0, 90.0), (10.0, 0.0, 0.0), (-10.0, 0.0, 180.0), (0.0, -10.0, 270.0)],
)
def test_get_bearing_cardinal(lat2, lon2, expected):
    """Bearings from the origin toward the cardinal directions."""
    assert pu._get_bearing(0.0, 0.0, lat2, lon2) == pytest.approx(expected)


def test_get_bearing_diagonal():
    """A north-east target gives a bearing near 45 degrees."""
    assert pu._get_bearing(0.0, 0.0, 1.0, 1.0) == pytest.approx(45.0, abs=0.1)


# ==========================================
# 2. Spatial Indexing
# ==========================================


def test_build_footprint_index(target_asset):
    """The STRtree and geometry list are built in collection order."""
    other = PhysicalAsset(id='other', geometry=rect(50, 0, 10, 10))
    tree, geoms = pu.build_footprint_index(
        PhysicalAssetCollection([target_asset, other])
    )
    assert isinstance(tree, STRtree)
    assert geoms == [target_asset.geometry, other.geometry]


def test_build_footprint_index_skips_none_geometry():
    """Entries without geometry are excluded from the index."""
    items = [SimpleNamespace(geometry=box(0, 0, 1, 1)), SimpleNamespace(geometry=None)]
    tree, geoms = pu.build_footprint_index(items)
    assert len(geoms) == 1


def test_get_nearby_buildings_excludes_target_and_far_polygons():
    """Only neighbours inside the radius are returned, never the target."""
    target = rect(0, 0, 10, 10)
    near = rect(20, 0, 10, 10)
    far = rect(500, 0, 10, 10)
    geoms = [target, near, far]
    tree = STRtree(geoms)
    result = pu.get_nearby_buildings(target, tree, geoms, radius_meters=30)
    assert len(result) == 1 and result[0].equals(near)


def test_get_nearby_buildings_none_in_range():
    """An empty list is returned when nothing is nearby."""
    target = rect(0, 0, 10, 10)
    geoms = [target, rect(500, 0, 10, 10)]
    assert pu.get_nearby_buildings(target, STRtree(geoms), geoms, 30) == []


def test_index_panos_empty():
    """An empty collection yields no tree and empty arrays."""
    tree, coords, headings = pu.index_panos(ImageCollection())
    assert tree is None and coords.size == 0 and headings.size == 0


def test_index_panos_populated(cardinal_panos):
    """Coordinates and headings are extracted in collection order."""
    tree, coords, headings = pu.index_panos(cardinal_panos)
    assert isinstance(tree, cKDTree)
    assert coords.shape == (4, 2) and coords.dtype == np.float32
    assert headings.shape == (4,)
    east = cardinal_panos[0]
    assert coords[0, 0] == pytest.approx(east.properties['longitude'])
    assert coords[0, 1] == pytest.approx(east.properties['latitude'])


def test_index_panos_missing_properties_default_to_zero():
    """Missing coordinates default to 0.0 rather than raising."""
    asset = ImageAsset(path='x.jpg', id='x', allow_missing_file=True)
    _, coords, headings = pu.index_panos(ImageCollection([asset]))
    assert coords.tolist() == [[0.0, 0.0]] and headings.tolist() == [0.0]


# ==========================================
# 3. Occlusion & Ray Matching
# ==========================================


def test_check_occlusion_no_neighbors():
    """No neighbours means the view is always clear."""
    assert pu._check_occlusion((5.0, 0.5), box(0, 0, 1, 1), []) is False


def test_check_occlusion_camera_inside_target():
    """A camera inside the target is never reported as occluded."""
    assert pu._check_occlusion((0.5, 0.5), box(0, 0, 1, 1), [box(2, 0, 3, 1)]) is False


def test_check_occlusion_blocked_and_clear():
    """A polygon on the line of sight blocks; one off to the side does not."""
    target = box(0, 0, 1, 1)
    blocker = box(2, 0, 3, 1)
    assert pu._check_occlusion((5.0, 0.5), target, [blocker]) is True
    assert pu._check_occlusion((0.5, 5.0), target, [blocker]) is False


def test_check_occlusion_ignores_target_copy_and_corner_touch():
    """Copies of the target and mere corner touches are not occlusions."""
    target = box(0, 0, 1, 1)
    # A neighbour whose corner touches the LOS at exactly one point:
    corner_toucher = Polygon([(3, 0.5), (4, 1.5), (4, -0.5)])
    assert pu._check_occlusion((5.0, 0.5), target, [box(0, 0, 1, 1)]) is False
    assert pu._check_occlusion((2.0, 0.5), target, [corner_toucher]) is False


def test_find_all_matches_for_ray_sorted_by_alignment():
    """Matches are sorted by perpendicular distance and exclude the back side."""
    xs = np.array([10.0, 20.0, -10.0, 5.0])
    ys = np.array([2.0, 0.5, 0.0, 20.0])
    result = pu._find_all_matches_for_ray(0.0, xs, ys, tolerance_meters=5.0)
    assert result.tolist() == [1, 0]


def test_find_all_matches_for_ray_none():
    """An empty integer array is returned when nothing aligns."""
    result = pu._find_all_matches_for_ray(90.0, np.array([10.0]), np.array([0.0]), 1.0)
    assert result.size == 0 and result.dtype.kind == 'i'


def test_name_ray():
    """Axis, corner and generic rays are named correctly."""
    axes = {0.0: 'major_pos', 90.0: 'minor_pos', 180.0: 'major_neg', 270.0: 'minor_neg'}
    assert pu._name_ray(0.0, axes, {}) == 'major_pos'
    assert pu._name_ray(360.0, axes, {}) == 'major_pos'
    assert pu._name_ray(45.0, axes, {45.0: 'corner_1'}) == 'corner_1'
    assert pu._name_ray(33.7, axes, {}) == 'ray_33'


def test_sort_key_ordering():
    """Principal axes sort first, then corners, then generic rays by angle."""
    names = ['ray_135', 'corner_2', 'minor_neg', 'ray_45', 'major_pos', 'corner_1b']
    ordered = sorted(names, key=pu._sort_key)
    assert ordered == [
        'major_pos',
        'minor_neg',
        'corner_1b',
        'corner_2',
        'ray_45',
        'ray_135',
    ]


def test_sort_key_fallbacks():
    """Malformed names fall back to sentinel orderings without raising."""
    assert pu._sort_key('corner') == (4, 9999)
    assert pu._sort_key('ray_x') == (5, 0)
    assert pu._sort_key('weird') == (5, 0)
    assert pu._sort_key('minor_pos') == (1, 0)
    assert pu._sort_key('major_neg') == (2, 0)


def test_result_img_index_lookup(cardinal_panos):
    """The index of a matching asset is found, or -1 otherwise."""
    assert (
        pu._result_img_index_lookup(cardinal_panos[2], [0, 1, 2, 3], cardinal_panos)
        == 2
    )
    assert pu._result_img_index_lookup(cardinal_panos[2], [0, 1], cardinal_panos) == -1


def test_ray_color():
    """Colours follow the major/minor/other convention."""
    assert pu._ray_color('major_neg') == 'blue'
    assert pu._ray_color('minor_pos') == 'purple'
    assert pu._ray_color('corner_1') == 'green'


# ==========================================
# 4. find_best_panos
# ==========================================


def test_find_best_panos_non_polygon_returns_empty(cardinal_panos):
    """Point geometries cannot be matched and return an empty dict."""
    lon, lat = offset_deg(0, 0)
    target = PhysicalAsset(id='pt', geometry=Point(lon, lat))
    assert run_scenario(target, cardinal_panos) == {}


def test_find_best_panos_no_candidates_in_radius(target_asset):
    """Cameras far outside the search radius yield an empty dict."""
    panos = ImageCollection([make_pano('far', 500, 0)])
    assert run_scenario(target_asset, panos) == {}


def test_find_best_panos_exact_radius_filter(target_asset):
    """A camera inside the coarse degree radius but beyond the metric radius."""
    # Coarse radius is 2x the metric radius in latitude degrees, so a camera at
    # 1.5x the radius eastward passes the KD-tree query but fails the exact check:
    panos = ImageCollection([make_pano('edge', 90, 0)])
    assert run_scenario(target_asset, panos, search_radius_meters=60) == {}


def test_find_best_panos_cardinal_matches(target_asset, cardinal_panos):
    """Each principal axis picks the camera facing that side of the building."""
    results = run_scenario(target_asset, cardinal_panos)
    assert set(results) == {'major_pos', 'minor_pos', 'major_neg', 'minor_neg'}
    assert {results['major_pos'].id, results['major_neg'].id} == {'east', 'west'}
    assert {results['minor_pos'].id, results['minor_neg'].id} == {'north', 'south'}


def test_find_best_panos_prefers_best_aligned(target_asset):
    """Among aligned cameras, the one with least perpendicular offset wins."""
    panos = ImageCollection([make_pano('offset', 30, 6), make_pano('aligned', 40, 0.5)])
    results = run_scenario(target_asset, panos)
    chosen = [r.id for r in results.values() if r is not None]
    assert chosen == ['aligned']


def test_find_best_panos_respects_ray_tolerance(target_asset):
    """A tighter tolerance rejects cameras that are slightly off-axis."""
    panos = ImageCollection([make_pano('offset', 30, 6)])
    loose = run_scenario(target_asset, panos, ray_tolerance_meters=10.0)
    tight = run_scenario(target_asset, panos, ray_tolerance_meters=2.0)
    assert any(r is not None for r in loose.values())
    assert all(r is None for r in tight.values())


def test_find_best_panos_occlusion_falls_back(target_asset):
    """An occluded nearest camera is skipped in favour of a clear one."""
    near = make_pano('near_east', 25, 0)
    panos = ImageCollection([near, make_pano('north', 0, 30)])
    blocker = rect(17, 0, 4, 12)  # wall between the building and near_east
    results = run_scenario(target_asset, panos, extra_buildings=[blocker])
    ids = {r.id for r in results.values() if r is not None}
    assert 'near_east' not in ids
    assert 'north' in ids
    # Without the blocker, near_east is selected:
    clear = run_scenario(target_asset, panos)
    assert 'near_east' in {r.id for r in clear.values() if r is not None}


def test_find_best_panos_multipolygon_uses_largest_part(cardinal_panos):
    """MultiPolygon targets are reduced to their largest component."""
    geom = MultiPolygon([rect(0, 0, 20, 10), rect(300, 300, 2, 2)])
    target = PhysicalAsset(id='multi', geometry=geom)
    results = run_scenario(target, cardinal_panos)
    assert {r.id for r in results.values() if r} == {'east', 'west', 'north', 'south'}


def test_find_best_panos_off_axis_interval(target_asset):
    """Smaller intervals add generic rays named by their angle."""
    panos = ImageCollection([make_pano('diag', 30, 30)])
    results = run_scenario(target_asset, panos, interval_deg=45)
    assert {'ray_45', 'ray_135', 'ray_225', 'ray_315'} <= set(results)
    diag_hits = [k for k, v in results.items() if v is not None]
    assert len(diag_hits) == 1 and diag_hits[0].startswith('ray_')


def test_find_best_panos_corner_rays(target_asset):
    """Corner mode names four corner rays and can match a diagonal camera."""
    # The rectangle corners sit at atan2(5, 10) ~ 26.6 degrees from the major axis:
    panos = ImageCollection([make_pano('corner_cam', 30, 15)])
    results = run_scenario(target_asset, panos, cast_corner_rays=True)
    corners = {k for k in results if k.startswith('corner_')}
    assert corners == {'corner_1', 'corner_2', 'corner_3', 'corner_4'}
    hits = [k for k, v in results.items() if v is not None]
    assert hits and all(k.startswith('corner_') for k in hits)


def test_find_best_panos_print_results(target_asset, cardinal_panos, capsys):
    """print_results writes one line per ray to stdout."""
    run_scenario(target_asset, cardinal_panos, print_results=True)
    out = capsys.readouterr().out
    assert 'major_pos' in out and 'ID:' in out


def test_find_best_panos_print_results_no_match(target_asset, capsys):
    """Unmatched rays are reported as such."""
    panos = ImageCollection([make_pano('diag', 30, 30)])
    run_scenario(target_asset, panos, print_results=True)
    assert 'No match (or Occluded)' in capsys.readouterr().out


def test_find_best_panos_debug_plots(target_asset, cardinal_panos, monkeypatch):
    """DEBUG_PLOTS renders the occlusion figure without opening a window."""
    shown = []
    monkeypatch.setattr(plt, 'show', lambda *a, **k: shown.append(True))
    monkeypatch.setattr(pu, 'DEBUG_PLOTS', True)
    neighbors = [
        rect(17, 0, 4, 12),
        MultiPolygon([rect(0, 25, 6, 4), rect(0, 40, 1, 1)]),
    ]
    results = run_scenario(target_asset, cardinal_panos, extra_buildings=neighbors)
    assert shown == [True]
    assert len(results) == 4


# ==========================================
# 5. Plot Helpers
# ==========================================


def test_plot_candidates_with_points(monkeypatch):
    """Both scatter layers and the legend are drawn when points exist."""
    shown = []
    monkeypatch.setattr(plt, 'show', lambda *a, **k: shown.append(True))
    pu._plot_candidates(
        box(0, 0, 1, 1),
        np.array([2.0, 3.0]),
        np.array([2.0, 3.0]),
        np.array([2.5]),
        np.array([2.5]),
    )
    ax = plt.gcf().axes[0]
    assert shown == [True]
    assert ax.get_legend() is not None
    assert len(ax.collections) == 3  # centroid + red + green scatters


def test_plot_candidates_without_points():
    """Empty candidate arrays skip the scatter layers."""
    pu._plot_candidates(
        box(0, 0, 1, 1), np.array([]), np.array([]), np.array([]), np.array([])
    )
    ax = plt.gcf().axes[0]
    assert len(ax.collections) == 1  # only the centroid marker


def test_plot_dynamic_results_full(monkeypatch):
    """Neighbours, raw circles and final stars are all drawn."""
    shown = []
    monkeypatch.setattr(plt, 'show', lambda *a, **k: shown.append(True))
    pu._plot_dynamic_results(
        box(-10, -5, 10, 5),
        np.array([30.0, 0.0, -30.0]),
        np.array([0.0, 30.0, 0.0]),
        0.0,
        10.0,
        [0.0, 45.0, 90.0, 180.0, 270.0],
        best_points={'major_pos': (30.0, 0.0), 'minor_pos': (0.0, 30.0)},
        raw_points={'major_pos': (30.0, 0.0), 'ray_45': (20.0, 20.0)},
        neighbor_polys=[box(15, -5, 20, 5)],
    )
    ax = plt.gcf().axes[0]
    assert shown == [True]
    assert 'Occlusion Analysis' in ax.get_title()
    assert ax.get_legend() is not None


def test_plot_dynamic_results_minimal():
    """The plot works with no candidates, neighbours or raw points."""
    pu._plot_dynamic_results(
        box(-10, -5, 10, 5),
        np.array([]),
        np.array([]),
        0.3,
        10.0,
        [0.0],
        best_points={},
    )
    assert len(plt.get_fignums()) == 1


# ==========================================
# 6. Panorama Cropping
# ==========================================


def make_pano_image(width=360, height=90, heading=0.0) -> ImageAsset:
    """Create a pano ImageAsset at the origin with an in-memory image."""
    pano = make_pano('pano', 0, 0, heading=heading)
    pano._pil_image = Image.new('RGB', (width, height), color='gray')
    return pano


def make_building(bearing_deg: float, distance_m: float = 40.0, size_m: float = 6.0):
    """Create a square building asset at a bearing/distance from the origin."""
    dx = distance_m * math.sin(math.radians(bearing_deg))
    dy = distance_m * math.cos(math.radians(bearing_deg))
    return PhysicalAsset(id='b', geometry=rect(dx, dy, size_m, size_m))


def test_crop_requires_loaded_image(target_asset):
    """A pano without a loaded PIL image raises TypeError."""
    pano = make_pano('pano', 0, 0)
    with pytest.raises(TypeError):
        pu.crop_panorama_to_asset(target_asset, pano, vertical_crop_mode='full')


def test_crop_full_mode_front_facing():
    """A building ahead of the camera is cropped around the image centre."""
    pano = make_pano_image()
    building = make_building(0.0)
    crop = pu.crop_panorama_to_asset(
        building, pano, horizontal_padding_deg=10.0, vertical_crop_mode='full'
    )
    # A 6 m building at 40 m spans about 8.6 degrees; plus 2 x 10 padding.
    assert 26 <= crop.width <= 30
    # 25 percent base sky crop on a 90 px tall image leaves 68 rows:
    assert crop.height == 90 - int(90 * 0.25)


def test_crop_horizontal_window_follows_bearing():
    """Rotating the camera heading shifts the horizontal window accordingly."""
    building = make_building(90.0)  # due east of the camera
    north_facing = make_pano_image(heading=0.0)
    east_facing = make_pano_image(heading=90.0)

    crop_east_facing = pu.crop_panorama_to_asset(
        building, east_facing, horizontal_padding_deg=5.0, vertical_crop_mode='full'
    )
    crop_north_facing = pu.crop_panorama_to_asset(
        building, north_facing, horizontal_padding_deg=5.0, vertical_crop_mode='full'
    )
    # Same angular width regardless of heading:
    assert abs(crop_east_facing.width - crop_north_facing.width) <= 1
    assert 15 <= crop_east_facing.width <= 22


def test_crop_clamps_to_image_edges():
    """Very wide padding is clamped to the image bounds."""
    pano = make_pano_image()
    crop = pu.crop_panorama_to_asset(
        make_building(0.0),
        pano,
        horizontal_padding_deg=400.0,
        vertical_crop_mode='full',
    )
    assert crop.width == 360


def test_crop_seam_crossing():
    """A building directly behind the camera wraps around the image seam."""
    pano = make_pano_image()
    building = make_building(180.0)
    crop = pu.crop_panorama_to_asset(
        building, pano, horizontal_padding_deg=10.0, vertical_crop_mode='full'
    )
    # Angular extent is identical to the front-facing case even though the
    # pixels come from both image edges:
    assert 26 <= crop.width <= 30
    assert crop.height == 90 - int(90 * 0.25)


def test_crop_seam_crossing_with_mask():
    """Mask strips are rolled in sync with the image across the seam."""
    pano = make_pano_image()
    mask = np.zeros((90, 360), dtype=np.uint8)
    mask[45:, :] = 2  # building everywhere below the horizon
    pano.set_mask(
        mask, mask_type='semantic', map_data={0: 'void--unlabeled', 2: 'building'}
    )
    crop = pu.crop_panorama_to_asset(
        make_building(180.0), pano, horizontal_padding_deg=10
    )
    assert crop.height == 90 - 45


def test_crop_multipolygon_and_linestring_geometries():
    """MultiPolygon uses its largest part; other geometries use the convex hull."""
    pano = make_pano_image()
    big = make_building(0.0).geometry
    small = rect(200, 200, 1, 1)
    multi = PhysicalAsset(id='m', geometry=MultiPolygon([big, small]))
    crop_multi = pu.crop_panorama_to_asset(
        multi, pano, horizontal_padding_deg=10.0, vertical_crop_mode='full'
    )
    crop_poly = pu.crop_panorama_to_asset(
        make_building(0.0), pano, horizontal_padding_deg=10.0, vertical_crop_mode='full'
    )
    assert crop_multi.size == crop_poly.size

    pts = [offset_deg(-3, 40), offset_deg(3, 40), offset_deg(0, 44)]
    line = PhysicalAsset(id='l', geometry=LineString(pts))
    crop_line = pu.crop_panorama_to_asset(
        line, pano, horizontal_padding_deg=10.0, vertical_crop_mode='full'
    )
    assert crop_line.width > 0


def _smart_pano(mask_rows: dict[tuple[int, int], int], semantic_map) -> ImageAsset:
    """Build a pano with a synthetic semantic mask (row ranges -> class id)."""
    pano = make_pano_image()
    mask = np.zeros((90, 360), dtype=np.uint8)
    for (start, stop), cls in mask_rows.items():
        mask[start:stop, :] = cls
    pano.set_mask(mask, mask_type='semantic', map_data=semantic_map)
    return pano


SEM_MAP = {
    0: 'void--unlabeled',
    1: 'nature--sky',
    2: 'construction--structure--building',
    3: 'void--ego-vehicle',
}


def test_crop_smart_trims_sky_and_vehicle():
    """Smart mode cuts above the first content row and above the vehicle."""
    pano = _smart_pano({(0, 40): 1, (40, 80): 2, (80, 90): 3}, SEM_MAP)
    crop = pu.crop_panorama_to_asset(
        make_building(0.0), pano, vertical_padding_percent=0.0
    )
    assert crop.height == 40  # rows 40..80


def test_crop_smart_padding_applied():
    """Vertical padding widens the smart crop on both ends."""
    pano = _smart_pano({(0, 40): 1, (40, 80): 2, (80, 90): 3}, SEM_MAP)
    crop = pu.crop_panorama_to_asset(
        make_building(0.0),
        pano,
        vertical_padding_percent=10.0,  # 9 px
    )
    assert crop.height == (80 - 9) - (40 - 9)


def test_crop_smart_keeps_base_crop_when_content_starts_high():
    """The base sky crop wins when content begins above it."""
    pano = _smart_pano({(0, 5): 1, (5, 90): 2}, SEM_MAP)
    crop = pu.crop_panorama_to_asset(
        make_building(0.0), pano, vertical_padding_percent=0.0
    )
    assert crop.height == 90 - int(90 * 0.25)


def test_crop_smart_only_sky_logs_info(caplog):
    """A strip containing only ignorable classes falls back to the base crop."""
    pano = _smart_pano({(0, 90): 1}, SEM_MAP)
    with caplog.at_level(logging.INFO):
        crop = pu.crop_panorama_to_asset(make_building(0.0), pano)
    assert crop.height == 90 - int(90 * 0.25)
    assert 'Smart crop could not be performed' in caplog.text


def test_crop_smart_without_semantic_map():
    """Without a semantic map only class 0 is ignored."""
    pano = _smart_pano({(0, 30): 0, (30, 90): 7}, None)
    crop = pu.crop_panorama_to_asset(
        make_building(0.0), pano, vertical_padding_percent=0.0
    )
    assert crop.height == 90 - 30


def test_crop_smart_invalid_vertical_range_resets(caplog):
    """A vehicle above the content produces an invalid crop that is reset."""
    pano = _smart_pano({(10, 20): 3, (40, 90): 2}, SEM_MAP)
    with caplog.at_level(logging.WARNING):
        crop = pu.crop_panorama_to_asset(
            make_building(0.0), pano, vertical_padding_percent=0.0
        )
    assert 'Invalid vertical crop' in caplog.text
    assert crop.height == 90 - int(90 * 0.25)


def test_crop_smart_failure_is_logged(caplog):
    """Errors while interpreting the semantic map degrade to the base crop."""
    pano = _smart_pano({(0, 90): 2}, {1: 5})  # non-string label breaks .lower()
    with caplog.at_level(logging.WARNING):
        crop = pu.crop_panorama_to_asset(make_building(0.0), pano)
    assert 'Smart crop failed' in caplog.text
    assert crop.height == 90 - int(90 * 0.25)


def test_crop_smart_requires_mask():
    """Smart mode without any mask available raises from load_mask."""
    pano = make_pano_image()
    with pytest.raises(FileNotFoundError):
        pu.crop_panorama_to_asset(make_building(0.0), pano)
