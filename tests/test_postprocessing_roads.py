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

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.ops import transform, unary_union

from rapidtools.core import PhysicalAsset, PhysicalAssetCollection
from rapidtools.processing.postprocessing import roads
from rapidtools.processing.postprocessing.roads import RoadwayRegularizer

# ==========================================
# 1. Fixtures & Helpers
# ==========================================

ORIGIN = Point(-118.2493, 34.0505)


@pytest.fixture
def regularizer():
    return RoadwayRegularizer()


@pytest.fixture
def to_wgs84(regularizer):
    """Transform from the local feet frame (centred on ORIGIN) to WGS84."""
    _, project_to_wgs84 = regularizer._get_transformers(ORIGIN)
    return lambda geom: transform(project_to_wgs84, geom)


def _ribbon(x0, y0, x1, y1, width_ft):
    """Straight road polygon in the local feet frame."""
    return LineString([(x0, y0), (x1, y1)]).buffer(width_ft / 2.0, cap_style=2)


def _noisy_ribbon(x0, y0, x1, y1, width_ft, sigma=1.0, seed=0):
    """Straight road with jittered boundary vertices (raster artefacts)."""
    rng = np.random.default_rng(seed)
    base = _ribbon(x0, y0, x1, y1, width_ft).segmentize(10)
    coords = [
        (x + rng.normal(0, sigma), y + rng.normal(0, sigma))
        for x, y in base.exterior.coords
    ]
    return Polygon(coords).buffer(0)


def _collection(to_wgs84, *geoms):
    return PhysicalAssetCollection(
        [PhysicalAsset(id=f'r{i}', geometry=to_wgs84(g)) for i, g in enumerate(geoms)]
    )


# ==========================================
# 2. Construction
# ==========================================


def test_init_defaults(regularizer):
    """Default thresholds are 22 ft width and 100 ft network length."""
    assert regularizer.min_width_ft == 22.0
    assert regularizer.min_network_length_ft == 100.0


def test_init_custom():
    """Custom thresholds are stored."""
    reg = RoadwayRegularizer(min_width_ft=30.0, min_network_length_ft=250.0)
    assert reg.min_width_ft == 30.0
    assert reg.min_network_length_ft == 250.0


# ==========================================
# 3. End-to-end processing
# ==========================================


def test_process_straight_noisy_road(regularizer, to_wgs84):
    """A jagged straight ribbon becomes one centerline and one polygon."""
    col = _collection(to_wgs84, _noisy_ribbon(0, 0, 600, 0, 30))
    centerlines, polygons = regularizer(col)

    assert len(centerlines) == 1
    assert len(polygons) == 1

    cl = centerlines[0]
    assert cl.id.startswith('cl_')
    assert cl.geometry.geom_type == 'LineString'
    assert cl.attributes['asset_type'] == 'road_centerline'
    assert cl.attributes['width_ft'] == pytest.approx(30.0, abs=1.5)
    assert cl.attributes['azimuth_deg'] == pytest.approx(90.0, abs=2.0)

    poly = polygons[0]
    assert poly.id.startswith('poly_')
    assert poly.geometry.geom_type == 'Polygon'
    assert poly.attributes['asset_type'] == 'road_polygon'
    assert poly.attributes['width_ft'] == cl.attributes['width_ft']
    # Output is WGS84 and sits on top of the input:
    assert poly.geometry.intersects(col['r0'].geometry)
    assert abs(poly.geometry.centroid.x - ORIGIN.x) < 0.01
    assert abs(poly.geometry.centroid.y - ORIGIN.y) < 0.01


def test_process_t_junction(regularizer, to_wgs84):
    """Two crossing ribbons yield two straight centerlines with correct headings."""
    network = unary_union([_ribbon(0, 0, 600, 0, 30), _ribbon(300, 0, 300, 400, 24)])
    centerlines, polygons = regularizer(_collection(to_wgs84, network))

    assert len(centerlines) == 2
    assert len(polygons) == 2
    headings = sorted(round(a.attributes['azimuth_deg']) % 180 for a in centerlines)
    assert headings == [0, 90]
    widths = sorted(a.attributes['width_ft'] for a in centerlines)
    assert widths[0] == pytest.approx(24.0, abs=1.0)
    assert widths[1] == pytest.approx(30.0, abs=1.0)
    # Polygons are non-overlapping: the wider road owns the junction.
    p0, p1 = (a.geometry for a in polygons)
    assert p0.intersection(p1).area < 1e-6 * min(p0.area, p1.area)


def test_process_multipolygon_input(regularizer, to_wgs84):
    """A MultiPolygon asset holding two parallel roads gives two of each."""
    network = MultiPolygon([_ribbon(0, 0, 500, 0, 30), _ribbon(0, 300, 500, 300, 30)])
    centerlines, polygons = regularizer(_collection(to_wgs84, network))
    assert len(centerlines) == 2
    assert len(polygons) == 2


def test_process_filters_small_artifacts(regularizer, to_wgs84):
    """Polygons under MIN_POLYGON_AREA_SQFT are ignored."""
    col = _collection(to_wgs84, _ribbon(0, 0, 600, 0, 30), box(0, 100, 10, 110))
    centerlines, polygons = regularizer(col)
    assert len(centerlines) == 1
    assert len(polygons) == 1


def test_process_min_width_enforced(to_wgs84):
    """Widths narrower than min_width_ft are raised to it."""
    reg = RoadwayRegularizer(min_width_ft=40.0)
    centerlines, _ = reg(_collection(to_wgs84, _ribbon(0, 0, 600, 0, 30)))
    assert centerlines[0].attributes['width_ft'] == 40.0


def test_process_short_network_filtered(regularizer, to_wgs84):
    """A road whose skeleton is shorter than min_network_length_ft is dropped."""
    centerlines, polygons = regularizer(_collection(to_wgs84, _ribbon(0, 0, 90, 0, 30)))
    assert len(centerlines) == 0
    assert len(polygons) == 0


def test_process_empty_input_warns(regularizer, caplog):
    """An empty collection returns two empty collections and warns."""
    with caplog.at_level(logging.WARNING):
        centerlines, polygons = regularizer.process(PhysicalAssetCollection())
    assert len(centerlines) == 0 and len(polygons) == 0
    assert 'No valid geometries' in caplog.text


def test_process_only_tiny_polygons_warns(regularizer, to_wgs84, caplog):
    """If every polygon is below the area threshold, nothing is produced."""
    col = _collection(to_wgs84, box(0, 0, 10, 10), box(50, 50, 55, 55))
    with caplog.at_level(logging.WARNING):
        centerlines, polygons = regularizer(col)
    assert len(centerlines) == 0 and len(polygons) == 0
    assert 'No input polygon exceeds' in caplog.text


def test_process_preserves_existing_asset_type(regularizer, to_wgs84):
    """set_asset_type(overwrite=False) keeps the specific output types."""
    centerlines, polygons = regularizer(
        _collection(to_wgs84, _ribbon(0, 0, 600, 0, 30))
    )
    assert centerlines[0].asset_type == 'road_centerline'
    assert polygons[0].asset_type == 'road_polygon'


# ==========================================
# 4. Geometry helpers
# ==========================================


def test_get_geoms(regularizer):
    """Multi-part geometries are split; single parts are wrapped in a list."""
    line = LineString([(0, 0), (1, 1)])
    assert regularizer._get_geoms(line) == [line]
    multi = MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]])
    assert len(regularizer._get_geoms(multi)) == 2


def test_get_lines_recursive(regularizer):
    """LineStrings are pulled out of nested collections; other types ignored."""
    l1 = LineString([(0, 0), (1, 0)])
    l2 = LineString([(0, 1), (1, 1)])
    gc = GeometryCollection([l1, MultiLineString([l2]), Point(5, 5)])
    lines = regularizer._get_lines(gc)
    assert len(lines) == 2
    assert regularizer._get_lines(Point(0, 0)) == []


def test_get_transformers_roundtrip(regularizer):
    """Feet <-> WGS84 transformers invert each other and scale in feet."""
    to_ft, to_wgs = regularizer._get_transformers(ORIGIN)
    x, y = to_ft(ORIGIN.x, ORIGIN.y)
    assert (x, y) == pytest.approx((0.0, 0.0), abs=1e-6)
    lon, lat = to_wgs(1000.0, 0.0)
    assert lat == pytest.approx(ORIGIN.y, abs=1e-6)
    # 1000 ft east is roughly 0.0033 degrees of longitude at this latitude:
    assert lon - ORIGIN.x == pytest.approx(0.00331, abs=0.0002)
    assert to_ft(*to_wgs(1000.0, 500.0)) == pytest.approx((1000.0, 500.0), abs=1e-6)


def test_get_largest_polygon(regularizer):
    """The largest polygon part is returned; single polygons pass through."""
    small, large = box(0, 0, 1, 1), box(5, 5, 9, 9)
    assert regularizer._get_largest_polygon(MultiPolygon([small, large])) == large
    gc = GeometryCollection([small, Point(0, 0), large])
    assert regularizer._get_largest_polygon(gc) == large
    assert regularizer._get_largest_polygon(large) == large


def test_extract_all_coords(regularizer):
    """Coordinates are gathered from nested geometries."""
    multi = MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]])
    assert regularizer._extract_all_coords(multi) == [
        (0.0, 0.0),
        (1.0, 1.0),
        (2.0, 2.0),
        (3.0, 3.0),
    ]
    assert regularizer._extract_all_coords(Point(4, 5)) == [(4.0, 5.0)]
    assert regularizer._extract_all_coords(object()) == []


# ==========================================
# 5. Skeleton & graph helpers
# ==========================================


def test_generate_voronoi_skeleton_is_merged(regularizer):
    """The skeleton of a straight ribbon is a single continuous LineString."""
    skeleton = regularizer._generate_voronoi_skeleton(_ribbon(0, 0, 600, 0, 30), 30.0)
    assert skeleton.geom_type in ('LineString', 'MultiLineString')
    lines = regularizer._get_lines(skeleton)
    total = sum(line.length for line in lines)
    assert total > 500
    # The skeleton stays inside the ribbon:
    assert _ribbon(0, 0, 600, 0, 30).buffer(0.5).contains(skeleton)


def test_generate_voronoi_skeleton_t_junction_edges_merge(regularizer):
    """Endpoints are snapped so healed graph edges are single LineStrings."""
    network = unary_union([_ribbon(0, 0, 600, 0, 30), _ribbon(300, 0, 300, 400, 24)])
    skeleton = regularizer._generate_voronoi_skeleton(network, 27.0)
    graph = regularizer._build_and_heal_graph(skeleton, 27.0)
    assert graph.number_of_edges() == 3
    assert all(
        d['geom'].geom_type == 'LineString' for _, _, d in graph.edges(data=True)
    )


def test_build_and_heal_graph_collapses_degree_two_nodes(regularizer):
    """Chained collinear lines are merged and short dead-ends pruned."""
    lines = MultiLineString(
        [
            [(0, 0), (100, 0)],
            [(100, 0), (200, 0)],
            [(200, 0), (300, 0)],
            [(300, 0), (400, 0)],
            [(200, 0), (200, 5)],  # short stub -> pruned
            [(400, 0), (400, 400)],
        ]
    )
    graph = regularizer._build_and_heal_graph(lines, approx_width=10.0)
    assert graph.number_of_edges() == 1
    (u, v, data) = next(iter(graph.edges(data=True)))
    assert {u, v} == {(0.0, 0.0), (400.0, 400.0)}
    assert data['length'] == pytest.approx(800.0)
    assert data['geom'].geom_type == 'LineString'


def test_build_and_heal_graph_ignores_rings_and_non_lines(regularizer):
    """Closed rings and non-line parts are skipped when building the graph."""
    ring = LineString([(0, 0), (1, 0), (1, 1), (0, 0)])
    gc = GeometryCollection([ring, Point(3, 3), LineString([(0, 0), (500, 0)])])
    graph = regularizer._build_and_heal_graph(gc, approx_width=10.0)
    assert graph.number_of_edges() == 1


def test_filter_stub_networks(regularizer):
    """Components shorter than min_network_length_ft are removed."""
    G = nx.Graph()
    G.add_edge((0, 0), (500, 0), geom=LineString([(0, 0), (500, 0)]), length=500.0)
    G.add_edge((0, 50), (30, 50), geom=LineString([(0, 50), (30, 50)]), length=30.0)
    filtered = regularizer._filter_stub_networks(G, approx_width=10.0)
    assert filtered.number_of_edges() == 1
    assert ((0, 0), (500, 0)) in filtered.edges or ((500, 0), (0, 0)) in filtered.edges


# ==========================================
# 6. Segment merging & width sampling
# ==========================================


def _graph_from_lines(*lines):
    G = nx.Graph()
    for line in lines:
        c = list(line.coords)
        G.add_edge(c[0], c[-1], geom=line, length=line.length)
    return G


@pytest.mark.parametrize(
    'line_a, line_b',
    [
        (LineString([(0, 0), (100, 0)]), LineString([(100, 0), (200, 0)])),
        (LineString([(100, 0), (200, 0)]), LineString([(0, 0), (100, 0)])),
        (LineString([(0, 0), (100, 0)]), LineString([(200, 0), (100, 0)])),
        (LineString([(100, 0), (0, 0)]), LineString([(100, 0), (200, 0)])),
    ],
)
def test_merge_collinear_segments_all_orientations(regularizer, line_a, line_b):
    """Collinear lines sharing an endpoint merge regardless of direction."""
    merged = regularizer._merge_collinear_segments(
        _graph_from_lines(line_a, line_b), approx_width=10.0
    )
    assert len(merged) == 1
    assert merged[0].length == pytest.approx(200.0)


def test_merge_collinear_segments_respects_angle(regularizer):
    """Lines meeting at a sharp angle are kept separate."""
    merged = regularizer._merge_collinear_segments(
        _graph_from_lines(
            LineString([(0, 0), (100, 0)]), LineString([(100, 0), (100, 100)])
        ),
        approx_width=10.0,
    )
    assert len(merged) == 2


def test_merge_collinear_segments_straightens_and_drops_short(regularizer):
    """Wobbly edges are simplified and sub-1ft parts are discarded."""
    wobbly = LineString([(0, 0), (50, 2), (100, -2), (150, 1), (200, 0)])
    tiny = LineString([(300, 0), (300.5, 0)])
    merged = regularizer._merge_collinear_segments(
        _graph_from_lines(wobbly, tiny), approx_width=10.0
    )
    assert len(merged) == 1
    assert len(merged[0].coords) == 2


def test_calculate_widths_and_azimuths(regularizer):
    """Median sampled width and compass azimuth are computed per segment."""
    network = _ribbon(0, 0, 600, 0, 30)
    east = LineString([(0, 0), (600, 0)])
    north = LineString([(300, -10), (300, 10)])
    analyzed = regularizer._calculate_widths_and_azimuths(
        [east, north], network, approx_width=30.0
    )
    assert analyzed[0]['geometry'] is east
    assert analyzed[0]['width'] == pytest.approx(30.0, abs=1e-6)
    assert analyzed[0]['azimuth'] == pytest.approx(90.0)
    # A short north-south probe measures the full 600 ft ribbon length:
    assert analyzed[1]['width'] == pytest.approx(300.0, abs=1e-6)
    assert analyzed[1]['azimuth'] == pytest.approx(0.0)


def test_calculate_widths_falls_back_to_min_width(regularizer):
    """Segments outside the network get min_width_ft."""
    network = _ribbon(0, 0, 600, 0, 30)
    far = LineString([(0, 1000), (600, 1000)])
    analyzed = regularizer._calculate_widths_and_azimuths([far], network, 30.0)
    assert analyzed[0]['width'] == regularizer.min_width_ft


# ==========================================
# 7. Collection builders
# ==========================================


def test_build_centerlines_collection(regularizer, to_wgs84):
    """Analyzed segments become WGS84 centerline assets with rounded attrs."""
    _, project_to_wgs84 = regularizer._get_transformers(ORIGIN)
    segments = [
        {
            'geometry': LineString([(0, 0), (100, 0)]),
            'width': 24.456,
            'azimuth': 90.123,
        },
        {'geometry': LineString([(0, 50), (0, 150)]), 'width': 30.0, 'azimuth': 0.0},
    ]
    col = regularizer._build_centerlines_collection(segments, project_to_wgs84)
    assert len(col) == 2
    first = next(iter(col))
    assert first.attributes == {
        'asset_type': 'road_centerline',
        'width_ft': 24.46,
        'azimuth_deg': 90.12,
    }
    assert abs(first.geometry.centroid.x - ORIGIN.x) < 0.01


def test_build_polygons_collection_subtracts_overlaps(regularizer):
    """Narrower roads are clipped by wider ones; identical ones vanish."""
    _, project_to_wgs84 = regularizer._get_transformers(ORIGIN)
    segments = [
        {'geometry': LineString([(0, 0), (400, 0)]), 'width': 20.0, 'azimuth': 90.0},
        {
            'geometry': LineString([(200, -200), (200, 200)]),
            'width': 40.0,
            'azimuth': 0.0,
        },
        {
            'geometry': LineString([(200, -200), (200, 200)]),
            'width': 40.0,
            'azimuth': 0.0,
        },
    ]
    col = regularizer._build_polygons_collection(segments, project_to_wgs84)
    # Wide road: 1 polygon. Narrow road split in two by the wide one. Duplicate
    # of the wide road is fully subtracted and dropped.
    assert len(col) == 3
    widths = sorted(a.attributes['width_ft'] for a in col)
    assert widths == [20.0, 20.0, 40.0]
    assert all(a.attributes['asset_type'] == 'road_polygon' for a in col)
    # Sorted widest-first in place:
    assert [s['width'] for s in segments] == [40.0, 40.0, 20.0]


def test_module_constants_are_sane():
    """Tuning constants keep their documented relationships."""
    assert roads.ASSET_TYPE == 'road'
    assert 0 < roads.SKELETON_CLIP_FACTOR < roads.SKELETON_BUFFER_FACTOR
    assert roads.MIN_SEGMENT_LENGTH_FT > 0
    assert 0 < roads.SKELETON_SNAP_GRID_FT < 10 ** (-roads.COORD_ROUNDING_DECIMALS)
