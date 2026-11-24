"""Tests for structures read off a survey's point cloud (``street_structures``)."""

import json
import logging
import math
import zlib

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from shapely.geometry import Point, Polygon

from rapidtools.core import Observation
from rapidtools.processing import MapillaryFeatureExtractor, street_structures
from rapidtools.processing.street_localization import local_projection, rotation_matrix
from rapidtools.processing.street_structures import (
    LabeledPoints,
    PointCloud,
    Structure,
    discover_structures,
    fetch_sfm_cluster,
    footprint_rectangle,
    is_structure_label,
    label_points,
    parse_sfm_cluster,
    points_in_outlines,
    project_points,
    structure_attributes,
    structures_from_points,
)

LON0, LAT0 = -117.4, 47.7
GROUND, CAMERA_HEIGHT = 100.0, 2.4
BUILDING = 'construction--structure--building'
PROJECT, UNPROJECT = local_projection(LON0, LAT0)
# Two houses north of an east-west street, a 4 m side yard between them:
HOUSES = [(-14.0, -2.0, 10.0, 20.0), (2.0, 14.0, 10.0, 20.0)]
HOUSE_HEIGHT = 5.0
# Level panorama looking north: x east, y down, z north.
LEVEL = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def _house_points(x0, x1, y0, y1, rng):
    """Points on the front wall, both side walls and the roof of a box house."""
    pts = []
    for x in np.arange(x0, x1 + 0.01, 0.25):
        for z in np.arange(0.4, HOUSE_HEIGHT, 0.4):
            pts.append((x, y0, z))
    for y in np.arange(y0, y1 + 0.01, 0.4):
        for z in np.arange(0.4, HOUSE_HEIGHT, 0.5):
            pts.append((x0, y, z))
            pts.append((x1, y, z))
    for x in np.arange(x0, x1 + 0.01, 0.8):
        for y in np.arange(y0, y1 + 0.01, 0.8):
            pts.append((x, y, HOUSE_HEIGHT))
    pts = np.array(pts) + rng.normal(0.0, 0.03, (len(pts), 3))
    pts[:, 2] += GROUND
    return pts


def _corners(x0, x1, y0, y1):
    return np.array(
        [
            (x, y, GROUND + z)
            for x in (x0, x1)
            for y in (y0, y1)
            for z in (0.0, HOUSE_HEIGHT)
        ]
    )


def _survey(n=21, spacing=3.0, houses=HOUSES, day='2026-08-27'):
    """A point cloud, the frames that made it and their building outlines."""
    rng = np.random.default_rng(0)
    house_pts = [_house_points(*h, rng) for h in houses]
    ground = np.c_[
        rng.uniform(-40, 40, 1500), rng.uniform(-5, 25, 1500), np.full(1500, GROUND)
    ]
    shrub = np.c_[
        rng.normal(0, 0.4, 150),
        rng.normal(6, 0.4, 150),
        GROUND + rng.uniform(0, 1.5, 150),
    ]
    points = np.vstack([*house_pts, ground, shrub])
    rotations, translations, sightings, clusters = [], [], [], {}
    for i in range(n):
        center = np.array([-30.0 + spacing * i, 0.0, GROUND + CAMERA_HEIGHT])
        # A slightly different heading per frame, as in a real drive:
        r = LEVEL @ Rotation.from_euler('z', 0.002 * (i - n // 2)).as_matrix()
        rotvec = Rotation.from_matrix(r).as_rotvec()
        t = -r @ center
        rotations.append(rotvec)
        translations.append(t)
        lon, lat = UNPROJECT(center[0], center[1])
        image_id = f'img{i}'
        clusters[image_id] = ('c1', 'https://clouds/c1')
        for house in houses:
            u, v, _ = project_points(_corners(*house), r, t)
            ring = np.c_[u, v]
            ring = ring[ConvexHull(ring).vertices]
            sightings.append(
                Observation(
                    image_id=image_id,
                    label=BUILDING,
                    polygon=[tuple(p) for p in ring],
                    camera_lon=lon,
                    camera_lat=lat,
                    compass_angle=0.0,
                    is_pano=True,
                    sequence_id='seq',
                    captured_at=f'{day}T10:00:{i:02d}+00:00',
                    image_width=4096,
                    image_height=2048,
                    extra={
                        'computed_rotation': [float(x) for x in rotvec],
                        'camera_type': 'spherical',
                    },
                )
            )
    cloud = PointCloud(
        'c1', LON0, LAT0, points, np.array(rotations), np.array(translations)
    )
    return cloud, sightings, clusters


@pytest.fixture(scope='module')
def survey():
    return _survey()


def _discover(survey, **kwargs):
    cloud, sightings, clusters = survey
    return discover_structures(
        sightings,
        clusters,
        project=PROJECT,
        unproject=UNPROJECT,
        loader=lambda cid, url: cloud,
        **kwargs,
    )


# ==========================================
# Point clouds and projection
# ==========================================
def test_structure_labels():
    assert is_structure_label(BUILDING)
    assert is_structure_label('construction--structure--bridge')
    assert not is_structure_label('construction--barrier--fence')
    assert not is_structure_label('object--vehicle--car')


def _payload():
    return zlib.compress(
        json.dumps(
            [
                {
                    'reference_lla': {
                        'longitude': LON0,
                        'latitude': LAT0,
                        'altitude': 0,
                    },
                    'points': {
                        '7': {'coordinates': [1.0, 2.0, 530.0], 'color': [1, 2, 3]},
                        '9': {'coordinates': [4.0, 5.0, 531.0], 'color': [1, 2, 3]},
                    },
                    'shots': {
                        'a': {'rotation': [1.5, 0.1, 0.2], 'translation': [1, 2, 3]}
                    },
                }
            ]
        ).encode()
    )


def test_parse_sfm_cluster_compressed_plain_and_empty():
    cloud = parse_sfm_cluster(_payload(), '42')
    assert cloud.cluster_id == '42' and (cloud.lon, cloud.lat) == (LON0, LAT0)
    assert cloud.points.tolist() == [[1.0, 2.0, 530.0], [4.0, 5.0, 531.0]]
    assert cloud.shot_index() == {(1.5, 0.1, 0.2): 0}
    plain = parse_sfm_cluster(zlib.decompress(_payload()))
    assert plain.points.shape == (2, 3)
    empty = parse_sfm_cluster(json.dumps({'points': {}, 'shots': {}}).encode())
    assert empty.points.shape == (0, 3) and empty.rotations.shape == (0, 3)


def test_fetch_sfm_cluster_downloads_once_and_survives_failures(tmp_path, monkeypatch):
    calls = []

    class Response:
        content = _payload()

        def raise_for_status(self):
            return None

    def get(url, timeout):
        calls.append(url)
        if 'broken' in url:
            raise OSError('network down')
        return Response()

    monkeypatch.setattr(street_structures.requests, 'get', get)
    first = fetch_sfm_cluster('42', 'https://clouds/42', tmp_path)
    again = fetch_sfm_cluster('42', 'https://clouds/42', tmp_path)
    assert calls == ['https://clouds/42']  # the second read came from the cache
    assert np.allclose(first.points, again.points) and again.lat == LAT0
    assert np.allclose(first.rotations, again.rotations)
    # No cache folder: always downloads. A failed download is not fatal.
    assert fetch_sfm_cluster('42', 'https://clouds/42').points.shape == (2, 3)
    assert fetch_sfm_cluster('13', 'https://broken/13', tmp_path) is None
    # An unreadable cache file is replaced by a fresh download:
    (tmp_path / '42.npz').write_bytes(b'not an archive')
    assert fetch_sfm_cluster('42', 'https://clouds/42', tmp_path).points.shape == (2, 3)


def test_project_points_into_a_level_panorama():
    world = np.array([[0.0, 10.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, -10.0]])
    u, v, dist = project_points(world, LEVEL, np.zeros(3))
    assert [round(float(x), 3) for x in u] == [0.5, 0.75, 0.5]  # north, east
    assert [round(float(x), 3) for x in v] == [0.5, 0.5, 0.75]  # 45 degrees down
    assert round(float(dist[2]), 2) == 14.14
    # The matrix is what rotation_matrix makes of the axis-angle vector:
    assert np.allclose(rotation_matrix([math.pi / 2, 0.0, 0.0]), LEVEL)


def test_points_in_outlines_across_the_seam():
    seam = np.array([[0.95, 0.4], [0.05, 0.4], [0.05, 0.6], [0.95, 0.6]])
    square = np.array([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]])
    u = np.array([0.98, 0.02, 0.5, 0.2])
    v = np.array([0.5, 0.5, 0.5, 0.5])
    assert points_in_outlines(u, v, [seam]).tolist() == [True, True, False, False]
    assert points_in_outlines(u, v, [seam, square]).tolist() == [
        True,
        True,
        True,
        False,
    ]
    assert not points_in_outlines(u, v, []).any()


# ==========================================
# Footprints
# ==========================================
def test_footprint_rectangle_follows_the_walls_not_the_street():
    # An L of wall points turned 20 degrees from the street direction.
    angle = math.radians(20.0)
    axis = np.array([math.cos(angle), math.sin(angle)])
    away = np.array([-math.sin(angle), math.cos(angle)])
    front = np.array([s * axis for s in np.linspace(0, 12, 80)])
    side = np.array([12 * axis + d * away for d in np.linspace(0, 9, 60)])
    wall = np.vstack([front, side]) + np.array([0.0, 10.0])
    outline, frontage, depth, observed, source, facing = footprint_rectangle(
        wall, wall, np.array([5.0, 0.0]), np.array([1.0, 0.0])
    )
    assert 11.0 < frontage < 12.5 and 8.0 < observed < 9.5
    assert source == 'observed' and depth == observed
    edge = np.array(outline[1]) - np.array(outline[0])
    assert abs(math.degrees(math.atan2(edge[1], edge[0])) - 20.0) < 3.0
    assert abs(facing - 160.0) < 3.0  # faces the camera, south-south-east
    assert Polygon(outline).area == pytest.approx(frontage * depth, rel=0.01)


def test_footprint_rectangle_fills_in_an_unseen_rear():
    wall = np.c_[np.linspace(0, 12, 60), np.full(60, 10.0)]
    far = np.c_[np.linspace(0, 12, 30), np.full(30, 45.0)]  # the house behind
    outline, frontage, depth, observed, source, _ = footprint_rectangle(
        wall, far, np.array([6.0, 0.0]), np.array([1.0, 0.0]), assumed_depth_m=9.0
    )
    assert source == 'assumed' and depth == 9.0 and observed < 1.0
    ys = sorted(round(y, 1) for _, y in outline)
    assert ys == [10.0, 10.0, 19.0, 19.0]  # starts at the wall, away from the camera
    # Sparse roof points behind the wall count as depth seen:
    roof = np.c_[np.tile(np.linspace(1, 11, 6), 3), np.repeat([13.0, 16.0, 18.5], 6)]
    _, _, depth, observed, source, _ = footprint_rectangle(
        wall, roof, np.array([6.0, 0.0]), np.array([1.0, 0.0]), assumed_depth_m=9.0
    )
    assert source == 'observed' and 8.0 < observed < 9.0


# ==========================================
# End to end on a synthetic street
# ==========================================
def test_two_houses_are_found_outlined_and_given_sightings(survey):
    structures, counts, unproject = _discover(survey)
    assert counts['clusters'] == 1 and counts['frames'] == 21
    assert counts['structures'] == len(structures) == 2
    structures.sort(key=lambda s: s.x)
    for structure, (x0, x1, y0, y1) in zip(structures, HOUSES, strict=True):
        truth = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        found = Polygon(structure.footprint)
        assert found.intersection(truth).area / found.union(truth).area > 0.8
        assert abs(structure.frontage_m - 12.0) < 1.0
        assert structure.depth_source == 'observed'
        assert abs(structure.observed_depth_m - 10.0) < 1.0
        assert abs(structure.height_m - HOUSE_HEIGHT) < 0.5
        assert abs(structure.facade_bearing - 180.0) < 5.0  # faces the street
        assert structure.days == ['2026-08-27'] and structure.n_points > 500
        # One sighting per frame kept, each an outline around the house:
        assert 3 <= structure.n_images <= 12
        assert len(structure.members) == structure.n_images
        for obs in structure.members:
            assert obs.source == 'sfm' and obs.label == BUILDING
            assert obs.range_m > 5.0 and obs.has_location
            xs = [p[0] for p in obs.polygon]
            ys = [p[1] for p in obs.polygon]
            assert 0.01 < max(xs) - min(xs) < 0.5 and max(ys) > 0.5 > min(ys)
        lon, lat = unproject(structure.x, structure.y)
        assert abs(structure.members[0].lon - lon) < 1e-9
    # The shrub in the side yard is too low to count and the yard kept the
    # houses apart:
    assert not any(Polygon(s.footprint).contains(Point(0.0, 6.0)) for s in structures)
    attributes = structure_attributes(structures[0])
    assert attributes['localization'] == 'point_cloud'
    assert attributes['depth_source'] == 'observed' and attributes['frontage_m'] > 11
    assert attributes['footprint_area_m2'] == pytest.approx(120.0, rel=0.2)


def test_views_are_spread_around_the_structure(survey):
    structures, _, _ = _discover(survey, max_views=3)
    for structure in structures:
        bearings = sorted(o.bearing for o in structure.members)
        assert len(bearings) == 3
        spread = (bearings[-1] - bearings[0] + 180.0) % 360.0 - 180.0
        assert abs(spread) > 40.0  # from the left, the front and the right
    every, _, _ = _discover(survey, max_views=0)
    assert all(s.n_images > 12 for s in every)


def test_observed_footprint_assumes_nothing(survey):
    structures, _, _ = _discover(survey, footprint='observed')
    assert len(structures) == 2
    for structure in structures:
        assert structure.depth_source == 'observed'
        assert structure.depth_m == structure.observed_depth_m
        assert 80.0 < Polygon(structure.footprint).area < 135.0
    with pytest.raises(ValueError):
        _discover(survey, footprint='circle')


def test_thresholds_drop_small_and_short_structures(survey):
    none, counts, _ = _discover(survey, min_points=100000)
    assert none == [] and counts['too_few_points'] == 2
    none, counts, _ = _discover(survey, min_frontage_m=50.0)
    assert none == [] and counts['too_short'] == 2
    none, counts, _ = _discover(survey, min_height_m=50.0)
    assert none == [] and counts['structures'] == 0


def test_a_narrow_gap_joins_and_a_wide_one_separates():
    close = _survey(houses=[(-14.0, -0.3, 10.0, 20.0), (0.3, 14.0, 10.0, 20.0)])
    joined, _, _ = _discover(close, min_gap_m=1.5)
    assert len(joined) == 1 and joined[0].frontage_m > 25.0
    apart, _, _ = _discover(_survey(), min_gap_m=1.5)
    assert len(apart) == 2


def test_frames_without_a_cloud_pose_or_panorama_are_skipped(survey):
    cloud, sightings, clusters = survey
    flat = Observation(
        'p', BUILDING, [(0, 0), (1, 0), (1, 1)], LON0, LAT0, 0.0, is_pano=False
    )
    no_pose = Observation('q', BUILDING, [(0, 0), (1, 0), (1, 1)], LON0, LAT0, 0.0)
    structures, counts, _ = discover_structures(
        [*sightings, flat, no_pose],
        {**clusters, 'p': ('c1', 'u'), 'q': ('c1', 'u')},
        project=PROJECT,
        unproject=UNPROJECT,
        loader=lambda cid, url: cloud,
    )
    assert len(structures) == 2 and counts['unusable'] == 2
    # A frame whose pose is not among the cloud's shots is counted, not fatal:
    stray = Observation(
        'r',
        BUILDING,
        [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6)],
        LON0,
        LAT0,
        0.0,
        extra={'computed_rotation': [0.3, 0.2, 0.1]},
    )
    _, counts, _ = discover_structures(
        [*sightings, stray],
        {**clusters, 'r': ('c1', 'u')},
        project=PROJECT,
        unproject=UNPROJECT,
        loader=lambda cid, url: cloud,
    )
    assert counts['frames_unmatched'] == 1


def test_nothing_to_do_cases(survey):
    cloud, sightings, clusters = survey
    structures, counts, unproject = discover_structures([], {})
    assert structures == [] and counts['structures'] == 0 and callable(unproject)
    structures, counts, _ = discover_structures(sightings, {})  # no clouds at all
    assert structures == [] and counts['clusters'] == 0
    structures, counts, _ = discover_structures(
        sightings, clusters, loader=lambda cid, url: None
    )
    assert structures == [] and counts['clusters_missing'] == 1
    empty = PointCloud(
        'c1', LON0, LAT0, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))
    )
    structures, _, _ = discover_structures(
        sightings, clusters, loader=lambda c, u: empty
    )
    assert structures == []
    assert structures_from_points(LabeledPoints(np.zeros((0, 4)), {}, {})) == []


def test_label_points_keeps_walls_and_drops_ground_and_clutter(survey):
    cloud, sightings, clusters = survey
    labeled = label_points(sightings, clusters, PROJECT, loader=lambda cid, url: cloud)
    rows = labeled.rows
    assert len(labeled.shots) == 21 and len(labeled.templates) == 21
    heights = rows[:, 3]
    assert np.percentile(heights, 99) < HOUSE_HEIGHT + 0.5
    # Nearly everything kept stands on a house (the convex test outlines also
    # cover the low shrub in front of the side yard; the height cut drops it):
    on_house = np.zeros(len(rows), bool)
    for x0, x1, y0, y1 in HOUSES:
        on_house |= (
            (rows[:, 0] > x0 - 0.5)
            & (rows[:, 0] < x1 + 0.5)
            & (rows[:, 1] > y0 - 0.5)
            & (rows[:, 1] < y1 + 0.5)
        )
    assert on_house.mean() > 0.9
    shrub = np.hypot(rows[:, 0], rows[:, 1] - 6.0) < 1.5
    assert rows[shrub, 3].max() < 2.0


# ==========================================
# Through the extractor
# ==========================================
class _Client:
    """The extractor only needs a client object when no network call is made."""


def test_extractor_places_buildings_from_the_point_cloud(survey, tmp_path, monkeypatch):
    cloud, sightings, clusters = survey
    monkeypatch.setattr(
        street_structures, 'fetch_sfm_cluster', lambda cid, url, cache=None: cloud
    )
    ext = MapillaryFeatureExtractor(
        classes=['buildings'],
        client=_Client(),
        save_directory=tmp_path,
        ego_filter=False,
        structure_views=4,
    )
    assert ext.point_cloud_dir == tmp_path / 'point_clouds'
    assert ext._wants_point_clouds({'buildings': (BUILDING,)}, [])
    assert not ext._wants_point_clouds({'cars': ('object--vehicle--car',)}, [])
    ext._clusters = dict(clusters)
    assets, counter = ext._assets_for_class('buildings', list(sightings), 0)
    assert counter == 2 and [a.id for a in assets] == [
        'street_buildings_00001',
        'street_buildings_00002',
    ]
    for asset in assets:
        assert asset.geometry.geom_type == 'Polygon'
        at = asset.attributes
        assert at['localization'] == 'point_cloud' and at['source'] == 'sfm'
        assert at['asset_type'] == 'buildings' and at['label'] == BUILDING
        assert at['n_observations'] == at['n_images'] == 4
        assert at['depth_source'] == 'observed' and at['min_range_m'] > 5
        assert len(at['observations']) == 4
    lon, lat = assets[0].geometry.centroid.x, assets[0].geometry.centroid.y
    x, y = local_projection(LON0, LAT0)[0](lon, lat)
    assert min(math.hypot(x - cx, y - 15.0) for cx in (-8.0, 8.0)) < 1.5
    # geometry='point' writes the footprint center instead:
    ext = MapillaryFeatureExtractor(
        classes=['buildings'],
        client=_Client(),
        save_directory=tmp_path,
        ego_filter=False,
        geometry='point',
        min_observations=50,
    )
    ext._clusters = dict(clusters)
    assets, _ = ext._assets_for_class('buildings', list(sightings), 0)
    assert assets == []  # no structure has 50 sightings
    ext.min_observations = 1
    assets, _ = ext._assets_for_class('buildings', list(sightings), 0)
    assert {a.geometry.geom_type for a in assets} == {'Point'}


def test_extractor_falls_back_to_fusion_without_point_clouds(survey, tmp_path, caplog):
    _, sightings, _ = survey
    ext = MapillaryFeatureExtractor(
        classes=['buildings'],
        client=_Client(),
        save_directory=tmp_path,
        ego_filter=False,
    )
    with caplog.at_level(logging.WARNING):
        assets, _ = ext._assets_for_class('buildings', list(sightings), 0)
    assert 'no frame has a Mapillary point cloud' in caplog.text
    assert all(a.attributes['localization'] != 'point_cloud' for a in assets)


def test_extractor_method_and_geometry_options(tmp_path):
    def make(**kw):
        return MapillaryFeatureExtractor(
            classes=['cars'], client=_Client(), save_directory=tmp_path, **kw
        )

    car = Observation('i', 'object--vehicle--car', [(0, 0), (1, 0), (1, 1)], 0, 0, 0)
    house = Observation('i', BUILDING, [(0, 0), (1, 0), (1, 1)], 0, 0, 0)
    assert make()._method_for([car]) == 'fusion'
    assert make()._method_for([house]) == 'points'
    assert make()._method_for([car, house]) == 'fusion'
    assert make(localization_method='tracks')._method_for([house]) == 'tracks'
    assert make(localization_method='points')._wants_point_clouds({'cars': ('x',)}, [])
    assert not make(localization_method='fusion')._wants_point_clouds(
        {'b': (BUILDING,)}, []
    )
    assert make(point_cloud_dir=tmp_path / 'pc').point_cloud_dir == tmp_path / 'pc'
    for bad in (
        {'localization_method': 'magic'},
        {'geometry': 'blob'},
        {'structure_footprint': 'circle'},
    ):
        with pytest.raises(ValueError):
            make(**bad)


def test_structure_record():
    s = Structure(
        [(0, 0), (12, 0), (12, 10), (0, 10)],
        6.0,
        5.0,
        [],
        420,
        12.0,
        10.0,
        4.2,
        'assumed',
        5.5,
        180.0,
    )
    assert s.n_images == 0
    assert structure_attributes(s)['observed_depth_m'] == 4.2
