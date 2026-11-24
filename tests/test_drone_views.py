"""Tests for drone view-set selection (rapidtools.processing.drone_views)."""

from __future__ import annotations

import itertools
import math
import threading

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

from rapidtools.core import PhysicalAsset, PhysicalAssetCollection
from rapidtools.core.cancellation import OperationCancelled
from rapidtools.data_sources import DroneCamera, write_pose_metadata
from rapidtools.processing import (
    DroneImageExtractor,
    DroneViewSelector,
    Pipeline,
    Stage,
)
from rapidtools.processing.drone_views import (
    AssetViewSet,
    blocked_rays,
    estimate_image_shift,
    footprint_polygon,
    select_views,
    surface_patches,
    view_quality,
)
from rapidtools.processing.street_localization import local_projection

LON0, LAT0 = -118.14, 34.186
PROJECT, UNPROJECT = local_projection(LON0, LAT0)


def geo(polygon: Polygon) -> Polygon:
    """A polygon given in local metres, as longitude/latitude."""
    return Polygon([UNPROJECT(x, y) for x, y in polygon.exterior.coords])


def building(asset_id: str, polygon: Polygon, **attributes) -> PhysicalAsset:
    return PhysicalAsset(id=asset_id, geometry=geo(polygon), attributes=attributes)


def camera_at(name, x, y, z, target=(0.0, 0.0, 3.0), size=(4000, 3000), focal=3000.0):
    """A camera at local ``(x, y, z)`` looking at ``target``."""
    lon, lat = UNPROJECT(x, y)
    dx, dy, dz = target[0] - x, target[1] - y, target[2] - z
    return DroneCamera(
        image_id=name,
        lon=lon,
        lat=lat,
        elevation_m=z,
        yaw_deg=math.degrees(math.atan2(dx, dy)) % 360.0,
        pitch_deg=math.degrees(math.atan2(dz, math.hypot(dx, dy))),
        width=size[0],
        height=size[1],
        focal_px=focal,
    )


def ring_of_cameras(radius=60.0, height=50.0, step=15, **kwargs):
    """Cameras every ``step`` degrees around the origin, plus one overhead."""
    cameras = [
        camera_at(
            f'ring_{angle:03d}',
            radius * math.sin(math.radians(angle)),
            radius * math.cos(math.radians(angle)),
            height,
            **kwargs,
        )
        for angle in range(0, 360, step)
    ]
    cameras.append(camera_at('nadir', 0.01, 0.0, 80.0, **kwargs))
    return cameras


HOUSE = box(-5, -4, 5, 4)


# ------------------------------------------------------------ surface model
def test_patches_cover_walls_and_roof_with_area_weights():
    patches = surface_patches(HOUSE, base=10.0, height=6.0)
    walls, roof = ~patches.is_roof, patches.is_roof
    assert patches.weights[walls].sum() == pytest.approx(HOUSE.length * 6.0)
    assert patches.weights[roof].sum() == pytest.approx(HOUSE.area)
    assert np.allclose(patches.points[roof, 2], 16.0)
    assert patches.points[walls, 2].min() > 10.0
    assert patches.points[walls, 2].max() < 16.0
    # Normals are unit length and point away from the centre.
    assert np.allclose(np.linalg.norm(patches.normals, axis=1), 1.0)
    outward = (patches.points[walls, :2] * patches.normals[walls, :2]).sum(axis=1)
    assert (outward > 0).all()
    assert sorted(patches.side_of_patch()[walls].tolist()) == sorted(
        patches.side_of_patch()[walls].tolist()
    )
    assert set(patches.side_of_patch().tolist()) == {-1, 0, 1, 2, 3}


def test_roof_weight_scales_or_removes_the_roof():
    heavy = surface_patches(HOUSE, 0.0, 6.0, roof_weight=2.0)
    assert heavy.weights[heavy.is_roof].sum() == pytest.approx(2 * HOUSE.area)
    none = surface_patches(HOUSE, 0.0, 6.0, roof_weight=0.0)
    assert not none.is_roof.any()


def test_tiny_footprint_still_gets_a_roof_patch_and_clockwise_rings_are_fixed():
    sliver = Polygon([(0, 0), (0.2, 3), (0.4, 0)])  # clockwise, thinner than the grid
    patches = surface_patches(sliver, 0.0, 3.0, roof_spacing_m=5.0)
    assert int(patches.is_roof.sum()) >= 1
    walls = ~patches.is_roof
    centre = np.array([sliver.centroid.x, sliver.centroid.y])
    outward = ((patches.points[walls, :2] - centre) * patches.normals[walls, :2]).sum(1)
    assert (outward > 0).all()


def test_footprint_polygon_handles_every_geometry_type():
    square = geo(box(0, 0, 10, 10))
    assert footprint_polygon(square, PROJECT).area == pytest.approx(100.0, rel=1e-3)
    holed = Polygon(square.exterior, [geo(box(4, 4, 6, 6)).exterior])
    assert footprint_polygon(holed, PROJECT).area == pytest.approx(100.0, rel=1e-3)
    two = MultiPolygon([square, geo(box(20, 0, 30, 10))])
    assert footprint_polygon(two, PROJECT).area == pytest.approx(300.0, rel=1e-3)
    one = MultiPolygon([square])
    assert footprint_polygon(one, PROJECT).area == pytest.approx(100.0, rel=1e-3)
    point = footprint_polygon(Point(LON0, LAT0), PROJECT, point_radius_m=2.0)
    assert point.area == pytest.approx(math.pi * 4, rel=0.06)
    line = LineString([UNPROJECT(0, 0), UNPROJECT(10, 0)])
    assert footprint_polygon(line, PROJECT).area > 20
    assert footprint_polygon(Polygon(), PROJECT) is None
    bowtie = Polygon([UNPROJECT(*p) for p in [(0, 0), (10, 10), (10, 0), (0, 10)]])
    assert footprint_polygon(bowtie, PROJECT) is not None
    cloud = geo(box(0, 0, 4, 4)).exterior.union(Point(UNPROJECT(8, 8)))
    assert footprint_polygon(cloud, PROJECT).area > 16


# --------------------------------------------------------------- visibility
def test_blocked_rays_respects_height_and_extent():
    wall = np.array([[5.0, -5.0, 5.0, 5.0]])
    tops = np.array([4.0])
    origins = [[0, 0, 1.0], [0, 0, 30.0], [0, 20, 1.0], [0, 0, 1.0]]
    targets = [[10, 0, 1.0], [10, 0, 1.0], [10, 20, 1.0], [4, 0, 1.0]]
    # Through the wall; over it; past its end; stopping short of it.
    assert blocked_rays(origins, targets, wall, tops).tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert blocked_rays(origins, targets, np.zeros((0, 4)), []).tolist() == [False] * 4
    assert blocked_rays(np.zeros((0, 3)), np.zeros((0, 3)), wall, tops).size == 0
    # Chunked evaluation gives the same answer.
    assert blocked_rays(origins, targets, wall, tops, chunk=1).tolist()[0] is True


def _arrays(cameras):
    return (
        np.array([c.position(PROJECT) for c in cameras]),
        np.array([c.rotation for c in cameras]),
        np.array([c.focal_px for c in cameras]),
        np.array([[c.width, c.height] for c in cameras], dtype=float),
        np.array([[c.cx, c.cy] for c in cameras], dtype=float),
    )


def test_a_camera_sees_only_the_walls_facing_it():
    patches = surface_patches(HOUSE, 0.0, 6.0)
    east_camera = camera_at('east', 60.0, 0.0, 20.0)
    quality, in_frame = view_quality(*_arrays([east_camera]), patches)
    sides = patches.side_of_patch()
    assert in_frame.all()
    assert quality[0, sides == 1].min() > 0.9  # east wall, square on
    assert quality[0, sides == 3].max() == 0.0  # west wall faces away
    assert quality[0, sides == 0].max() == 0.0  # north wall is edge-on
    assert 0.0 < quality[0, sides == -1].max() < 0.5  # roof at a grazing angle


def test_quality_falls_with_distance_and_frame():
    patches = surface_patches(HOUSE, 0.0, 6.0)
    near, far = camera_at('near', 60.0, 0.0, 20.0), camera_at('far', 600.0, 0.0, 200.0)
    away = camera_at('away', 60.0, 0.0, 20.0, target=(120.0, 0.0, 3.0))
    quality, in_frame = view_quality(*_arrays([near, far, away]), patches)
    assert quality[1].max() < 0.5 * quality[0].max()  # 5 px/m against 15 wanted
    assert quality[2].max() == 0.0 and not in_frame[2].any()  # behind the camera


def test_neighbour_hides_the_wall_behind_it():
    patches = surface_patches(HOUSE, 0.0, 6.0)
    camera = camera_at('east', 60.0, 0.0, 20.0)
    ring = np.asarray(box(8, -10, 14, 10).exterior.coords)[:, :2]
    edges = np.column_stack([ring[:-1], ring[1:]])
    east = patches.side_of_patch() == 1
    open_view, _ = view_quality(*_arrays([camera]), patches)
    hidden, _ = view_quality(*_arrays([camera]), patches, edges, np.full(4, 30.0))
    low, _ = view_quality(*_arrays([camera]), patches, edges, np.full(4, 1.0))
    assert open_view[0, east].min() > 0.9
    assert hidden[0, east].max() == 0.0
    assert np.allclose(low, open_view)  # a garden wall hides nothing from above


def test_concave_footprint_hides_its_own_inner_wall():
    # A U open to the north: the inner faces of the two arms face each other.
    u_shape = Polygon(
        [(0, 0), (30, 0), (30, 20), (20, 20), (20, 8), (10, 8), (10, 20), (0, 20)]
    )
    patches = surface_patches(u_shape, 0.0, 12.0)
    ring = np.asarray(patches.polygon.exterior.coords)[:, :2]
    edges = np.column_stack([ring[:-1], ring[1:]])
    tops = np.full(len(edges), 12.0)
    # Low camera east of the building: the arm in front hides the courtyard.
    camera = camera_at('east', 90.0, 14.0, 8.0, target=(15.0, 14.0, 6.0))
    inner = (patches.wall >= 0) & np.isclose(patches.points[:, 0], 10.0, atol=0.1)
    assert inner.sum() > 0
    free, _ = view_quality(*_arrays([camera]), patches)
    real, _ = view_quality(*_arrays([camera]), patches, edges, tops)
    assert free[0, inner].max() > 0.5 and real[0, inner].max() == 0.0


# ---------------------------------------------------------------- selection
def test_greedy_skips_duplicates_and_stops_at_small_gains():
    quality = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.95, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0, 0, 0.1, 0],
        ]
    )
    chosen, gains, best = select_views(quality, np.ones(4), max_views=4, min_gain=0.05)
    assert chosen == [0, 2]
    assert gains == pytest.approx([0.25, 0.25])
    assert best.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert select_views(quality, np.ones(4), max_views=1)[0] == [0]
    assert select_views(quality, np.zeros(4))[0] == []
    assert select_views(np.zeros((0, 4)), np.ones(4))[0] == []
    assert select_views(quality, np.ones(4), max_views=0)[0] == []


def test_greedy_matches_exhaustive_search_on_small_problems():
    rng = np.random.default_rng(3)
    for _ in range(20):
        quality = rng.random((9, 14)) * (rng.random((9, 14)) > 0.6)
        weights = rng.random(14) + 0.1
        chosen, _, best = select_views(quality, weights, max_views=3, min_gain=0.0)
        greedy = float(best @ weights)
        optimum = max(
            float(quality[list(combo)].max(axis=0) @ weights)
            for combo in itertools.combinations(range(9), 3)
        )
        assert greedy >= (1 - 1 / math.e) * optimum - 1e-9
        assert greedy >= 0.9 * optimum  # far better than the bound in practice


def test_diversity_breaks_ties_towards_a_new_direction():
    quality = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    directions = np.array([[1.0, 0, 0], [0.99, 0.14, 0], [-1.0, 0, 0]])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    chosen, _, _ = select_views(quality, np.ones(3), 2, 0.0, directions, 0.2)
    assert chosen == [0, 2]  # the opposite side beats the neighbouring frame
    plain, _, _ = select_views(quality, np.ones(3), 2, 0.0, directions, 0.0)
    assert plain == [0, 1]


def test_ring_flight_covers_all_four_sides_and_the_roof():
    selector = DroneViewSelector(ring_of_cameras(), max_views=4)
    view_set = selector.select([building('house', HOUSE)])['house']
    assert 2 <= len(view_set.views) <= 4
    assert view_set.coverage > 0.85 and view_set.seen_fraction == 1.0
    assert set(view_set.coverage_by_side) == {'roof', 'north', 'east', 'south', 'west'}
    assert min(view_set.coverage_by_side.values()) > 0.7
    assert view_set.unseen_walls == []
    # The first two views look from opposite corners, each adding new sides.
    first, second = view_set.views[:2]
    separation = abs((first.azimuth_deg - second.azimuth_deg + 180) % 360 - 180)
    assert separation > 120
    assert first.gain > second.gain > 0.2
    shown = set(first.shows) | set(second.shows)
    assert shown == {'roof', 'north', 'east', 'south', 'west'}
    assert first.frame_fraction == 1.0 and first.distance_m == pytest.approx(76, abs=2)
    assert len(first.outline) >= 4 and first.bbox[2] > first.bbox[0]
    # Consecutive frames 15 degrees apart are never both chosen.
    azimuths = sorted(v.azimuth_deg for v in view_set.views)
    assert min(b - a for a, b in itertools.pairwise(azimuths)) > 30


def test_one_sided_flight_reports_the_unseen_walls():
    cameras = [
        camera_at(f'south_{i}', x, -60.0, 40.0) for i, x in enumerate((-20, 0, 20))
    ]
    selector = DroneViewSelector(cameras)
    view_set = selector.select([building('house', HOUSE)])['house']
    assert view_set.coverage_by_side['south'] > 0.9
    assert view_set.coverage_by_side['north'] == 0.0
    north = [w for w in view_set.unseen_walls if w['side'] == 'north']
    assert len(north) == 1 and north[0]['length_m'] == pytest.approx(10.0, abs=0.1)
    assert north[0]['facing_deg'] == pytest.approx(0.0, abs=0.1)
    spot = north[0]['viewpoint']
    x, y = PROJECT(spot['lon'], spot['lat'])
    assert (x, y) == pytest.approx((0.0, 19.0), abs=0.1)  # 15 m out from the wall
    assert spot['yaw_deg'] == pytest.approx(180.0) and spot['pitch_deg'] == -45.0
    gaps = DroneViewSelector.coverage_gaps({'house': view_set})
    assert {g.attributes['side'] for g in gaps} >= {'north'}
    gap = next(g for g in gaps if g.attributes['side'] == 'north')
    assert gap.geometry.geom_type == 'LineString'
    assert gap.attributes['asset_id'] == 'house'
    assert gap.attributes['fly_yaw_deg'] == pytest.approx(180.0)


def test_tall_neighbour_changes_which_views_are_chosen():
    house, tower = building('house', HOUSE), building('tower', box(8, -10, 14, 10))
    tower.attributes['height'] = 30
    alone = DroneViewSelector(ring_of_cameras(), max_views=6).select([house])['house']
    both = DroneViewSelector(ring_of_cameras(), max_views=6).select([house, tower])
    shaded = both['house']
    assert alone.coverage_by_side['east'] > 0.8
    assert shaded.coverage_by_side['east'] < 0.5
    assert shaded.coverage < alone.coverage
    # The same tower passed as an obstacle (not analysed) has the same effect.
    blocked = DroneViewSelector(
        ring_of_cameras(), max_views=6, obstacles=PhysicalAssetCollection([tower])
    ).select([house])
    assert set(blocked) == {'house'}
    assert blocked['house'].coverage == pytest.approx(shaded.coverage)


def test_clipped_views_are_penalised():
    centred = camera_at('centred', 60.0, 0.0, 20.0, focal=3000.0)
    # Same spot, aimed so the house is cut by the frame edge:
    clipped = camera_at('clipped', 60.0, 0.0, 20.0, target=(0.0, 40.0, 3.0))
    selector = DroneViewSelector([clipped, centred], max_views=1)
    view = selector.select([building('house', HOUSE)])['house'].views[0]
    assert view.camera.image_id == 'centred' and view.frame_fraction == 1.0
    alone = DroneViewSelector([clipped], max_views=1, clip_penalty=1.0)
    free = DroneViewSelector([clipped], max_views=1, clip_penalty=0.0)
    penalised = alone.select([building('house', HOUSE)])['house']
    unpenalised = free.select([building('house', HOUSE)])['house']
    assert 0 < penalised.views[0].frame_fraction < 1
    assert penalised.coverage < unpenalised.coverage


def test_roof_weight_steers_the_first_view():
    cameras = [camera_at('nadir', 0.01, 0.0, 80.0), camera_at('low', 60.0, 0.0, 12.0)]
    house = building('house', HOUSE)
    roofs = DroneViewSelector(cameras, max_views=1, roof_weight=5.0).select([house])
    walls = DroneViewSelector(cameras, max_views=1, roof_weight=0.0).select([house])
    assert roofs['house'].views[0].camera.image_id == 'nadir'
    assert walls['house'].views[0].camera.image_id == 'low'
    assert 'roof' not in walls['house'].coverage_by_side


def test_heights_and_ground_elevations_come_from_attributes():
    selector = DroneViewSelector([])
    assert selector.asset_height(building('a', HOUSE)) == 6.0
    assert selector.asset_height(building('a', HOUSE, height='9.5 m')) == 9.5
    assert (
        selector.asset_height(building('a', HOUSE, **{'building:levels': '3'})) == 9.0
    )
    assert selector.asset_height(building('a', HOUSE, height=-1, levels=2)) == 6.0
    assert selector.asset_height(building('a', HOUSE, height=True)) == 6.0
    named = DroneViewSelector([], height_attribute='h', default_height_m=4.0)
    assert named.asset_height(building('a', HOUSE, h=7, height=99)) == 7.0
    assert named.asset_height(building('a', HOUSE, height=99)) == 4.0
    assert (
        DroneViewSelector([], ground_elevation=12.5).asset_ground(building('a', HOUSE))
        == 12.5
    )
    by_name = DroneViewSelector([], ground_elevation='z0')
    assert by_name.asset_ground(building('a', HOUSE, z0='101.5')) == 101.5
    assert by_name.asset_ground(building('a', HOUSE)) == 0.0
    dem = DroneViewSelector([], ground_elevation=lambda lon, lat: 100 + (lat - LAT0))
    assert dem.asset_ground(building('a', HOUSE)) == pytest.approx(100.0, abs=1e-3)


def test_ground_elevation_moves_the_projected_outline():
    camera = camera_at('oblique', 60.0, 0.0, 60.0, target=(0.0, 0.0, 0.0))
    house = building('house', HOUSE, z0=20.0)
    flat = DroneViewSelector([camera]).select([house])['house']
    raised = DroneViewSelector([camera], ground_elevation='z0').select([house])['house']
    assert raised.ground_elevation_m == 20.0 and flat.ground_elevation_m == 0.0
    # Closer to the camera, the raised house appears higher in the frame.
    assert raised.views[0].bbox[1] < flat.views[0].bbox[1] - 100


def test_empty_inputs_and_unseen_assets():
    selector = DroneViewSelector(ring_of_cameras())
    assert selector.select([]) == {}
    far = building('far', box(5000, 5000, 5010, 5008))
    empty = PhysicalAsset(id='empty', geometry=Polygon())
    result = selector.select([far, empty])
    assert set(result) == {'far'}
    assert result['far'].views == [] and result['far'].coverage == 0.0
    assert result['far'].n_candidates == 0
    assert len(result['far'].unseen_walls) == 4
    no_cameras = DroneViewSelector([]).select([building('house', HOUSE)])
    assert isinstance(no_cameras['house'], AssetViewSet)
    assert no_cameras['house'].views == []
    with pytest.raises(ValueError, match='max_views'):
        DroneViewSelector([], max_views=0)
    with pytest.raises(ValueError, match='roof_weight'):
        DroneViewSelector([], roof_weight=-1)


def test_candidate_cap_keeps_every_viewing_direction():
    cameras = ring_of_cameras(step=2)  # 181 photos around one house
    capped = DroneViewSelector(cameras, max_candidates=24)
    full = DroneViewSelector(cameras, max_candidates=1000)
    house = building('house', HOUSE)
    a, b = capped.select([house])['house'], full.select([house])['house']
    assert a.n_candidates == b.n_candidates == 181
    assert a.coverage == pytest.approx(b.coverage, abs=0.03)
    assert min(a.coverage_by_side.values()) > 0.7


def test_point_assets_get_a_slim_prism():
    pole = PhysicalAsset(
        id='pole', geometry=Point(LON0, LAT0), attributes={'height': 10}
    )
    view_set = DroneViewSelector(ring_of_cameras(radius=25, height=20)).select([pole])
    assert view_set['pole'].coverage > 0.8 and view_set['pole'].height_m == 10


def test_cancellation_stops_selection():
    event = threading.Event()
    event.set()
    selector = DroneViewSelector(ring_of_cameras(), cancel_event=event)
    with pytest.raises(OperationCancelled):
        selector.select([building('house', HOUSE)])


# ------------------------------------------------------------ pipeline step
@pytest.fixture
def flight(tmp_path):
    """A small ring flight written to disk as geotagged JPEGs."""
    cameras = ring_of_cameras(step=45, size=(400, 300), focal=300.0)
    written = []
    for camera in cameras:
        image = Image.new('RGB', (400, 300), (90, 120, 90))
        path = write_pose_metadata(
            image, tmp_path / 'photos' / f'{camera.image_id}.jpg', camera
        )
        camera.path = path
        written.append(camera)
    return tmp_path, written


def test_extractor_attaches_crops_and_coverage(flight):
    folder, cameras = flight
    collection = PhysicalAssetCollection(
        [building('house', HOUSE), building('shed', box(30, 30, 34, 33), height=3)]
    )
    extractor = DroneImageExtractor(
        cameras,
        save_directory=folder / 'crops',
        max_views=3,
        target_px_per_m=3.0,
        outline_width=6,
    )
    assert extractor.stage == Stage.EXTRACT_IMAGERY
    result = Pipeline([extractor]).run(collection)
    house = result.get('house')
    images = list(house.image_assets)
    assert 2 <= len(images) <= 3 == extractor.selector.max_views
    assert house.attributes['n_drone_views'] == len(images)
    assert house.attributes['drone_view_coverage'] > 0.8
    assert house.attributes['drone_unseen_sides'] == []
    assert house.attributes['drone_coverage_roof'] > 0.5
    assert house.attributes['drone_candidate_photos'] == 9
    first = images[0]
    assert first.path.exists() and first.path.parent == (folder / 'crops').resolve()
    properties = first.properties
    assert properties['source'] == 'drone' and properties['view_rank'] == 1
    assert properties['image_id'] in {c.image_id for c in cameras}
    assert properties['coverage_gain'] >= images[1].properties['coverage_gain']
    assert set(properties['shows']) <= {'roof', 'north', 'east', 'south', 'west'}
    left, top, right, bottom = properties['crop_box']
    with Image.open(first.path) as crop:
        assert crop.size == (right - left, bottom - top)
        assert crop.size[0] >= 256 or crop.size[0] == 400  # grown or clipped
        # The asset is marked: some pixels are the outline colour.
        reds = sum(1 for r, g, b in crop.getdata() if r > 200 and g < 60 and b < 60)
        assert reds > 20
    assert (
        extractor.view_sets['house'].views[0].camera.image_id == properties['image_id']
    )
    gaps = extractor.coverage_gaps()
    assert all(gap.attributes['asset_id'] != 'house' for gap in gaps)
    assert result.get('shed').attributes['n_drone_views'] >= 1


def test_extractor_reads_a_folder_and_can_leave_crops_unmarked(flight):
    folder, _ = flight
    collection = PhysicalAssetCollection([building('house', HOUSE)])
    extractor = DroneImageExtractor(
        folder / 'photos',
        save_directory=folder / 'plain',
        overlay_asset_outline=False,
        crop_buffer=10,
        position_error_m=0.0,
        angle_error_deg=0.0,
        min_crop_px=16,
        image_prefix='uav',
        attribute_prefix='uav',
        target_px_per_m=3.0,
    )
    assert len(extractor.cameras) == 9
    house = extractor(collection).get('house')
    image = house.image_assets[0]
    assert image.id.startswith('house_uav_') and house.attributes['n_uav_views'] >= 2
    with Image.open(image.path) as crop:
        colours = np.asarray(crop).reshape(-1, 3).astype(int)
        assert np.abs(colours - [90, 120, 90]).max() <= 3  # no mark drawn
    view = extractor.view_sets['house'].views[0]
    left, top, right, bottom = extractor.crop_box(view)
    assert left == max(0, math.floor(view.bbox[0] - 10))
    small = DroneImageExtractor(
        folder / 'photos', folder / 'small', max_crop_px=100, target_px_per_m=3.0
    )
    shrunk = small(PhysicalAssetCollection([building('house', HOUSE)])).get('house')
    for image in shrunk.image_assets:
        with Image.open(image.path) as crop:
            assert max(crop.size) == 100
        assert image.properties['crop_scale'] < 1
        assert max(max(p) for p in image.properties['outline']) <= 100.5
    assert bottom == min(300, math.ceil(view.bbox[3] + 10))


def test_pose_error_allowance_widens_the_crop(flight):
    folder, cameras = flight
    house = PhysicalAssetCollection([building('house', HOUSE)])
    tight = DroneImageExtractor(
        cameras, folder / 'a', position_error_m=0, angle_error_deg=0, min_crop_px=8
    )
    loose = DroneImageExtractor(
        cameras, folder / 'b', position_error_m=5, angle_error_deg=2, min_crop_px=8
    )
    tight(house)
    view = tight.view_sets['house'].views[0]
    a, b = tight.crop_box(view), loose.crop_box(view)
    expected = 300.0 * (5 / view.distance_m + math.tan(math.radians(2)))
    assert (a[0] - b[0]) == pytest.approx(expected, abs=1.5) or b[0] == 0
    assert (b[2] - b[0]) >= (a[2] - a[0])


def test_extractor_edge_cases(flight, caplog):
    folder, cameras = flight
    with pytest.raises(ValueError, match='outline_shape'):
        DroneImageExtractor(cameras, folder / 'x', outline_shape='star')
    with pytest.raises(ValueError, match='not both'):
        DroneImageExtractor(
            cameras, folder / 'x', selector=DroneViewSelector(cameras), roof_weight=2
        )
    # Cameras without a file are dropped; with none left nothing is extracted.
    homeless = [camera_at('ghost', 60.0, 0.0, 40.0)]
    extractor = DroneImageExtractor(homeless, folder / 'x')
    assert extractor.cameras == [] and 'no image file' in caplog.text
    collection = PhysicalAssetCollection([building('house', HOUSE)])
    assert extractor(collection) is collection
    assert 'nothing extracted' in caplog.text
    assert len(collection.get('house').image_assets) == 0
    empty = PhysicalAssetCollection()
    assert DroneImageExtractor(cameras, folder / 'x')(empty) is empty
    # An asset out of sight gets the attributes and no crops; a photo that
    # vanished after selection is reported, not fatal.
    far = PhysicalAssetCollection(
        [building('far', box(5000, 5000, 5010, 5008)), building('house', HOUSE)]
    )
    working = DroneImageExtractor(cameras, folder / 'y', max_views=1)
    chosen = DroneViewSelector(cameras, max_views=1).select(list(far))['house']
    chosen.views[0].camera.path.unlink()
    working(far)
    assert far.get('far').attributes['n_drone_views'] == 0
    assert far.get('far').attributes['drone_unseen_sides'] == [
        'east',
        'north',
        'south',
        'west',
    ]
    assert far.get('house').attributes['n_drone_views'] == 0
    assert 'Failed to crop photo' in caplog.text


def test_allowed_restricts_the_photos_of_an_asset():
    selector = DroneViewSelector(ring_of_cameras(), max_views=4)
    house, shed = building('house', HOUSE), building('shed', box(30, 30, 34, 33))
    free = selector.select([house, shed])
    limited = selector.select([house, shed], allowed={'house': ['ring_180', 'nadir']})
    assert {v.camera.image_id for v in limited['house'].views} <= {'ring_180', 'nadir'}
    assert limited['house'].coverage < free['house'].coverage
    assert limited['house'].n_candidates == 2
    assert limited['shed'].coverage == free['shed'].coverage  # not listed: unrestricted
    none = selector.select([house], allowed={'house': []})
    assert none['house'].views == []


# ----------------------------------------------------------- pose refinement
def test_estimate_image_shift_recovers_a_translation():
    expected = np.zeros((60, 80), dtype=bool)
    expected[20:35, 15:40] = True
    expected[40:50, 50:70] = True
    observed = np.roll(expected, (-4, 6), axis=(0, 1))
    dx, dy, best, before = estimate_image_shift(expected, observed, 10)
    assert (dx, dy) == (6, -4) and best == pytest.approx(1.0) and before < 0.7
    # A shift beyond the search window is not found; the best inside it is.
    dx, dy, best, _ = estimate_image_shift(expected, observed, 3)
    assert abs(dx) <= 3 and abs(dy) <= 3 and best < 1.0
    empty = np.zeros_like(expected)
    assert estimate_image_shift(expected, empty, 5) == (0, 0, 0.0, 0.0)
    assert estimate_image_shift(empty, observed, 5) == (0, 0, 0.0, 0.0)


class FakeSegmenter:
    """Stands in for SAM 3: 'sees' the house where a shifted pose would put it."""

    def __init__(self, selector, cameras, shift, blind=()):
        self.selector, self.shift, self.blind = selector, shift, set(blind)
        self.cameras = {str(c.path): c for c in cameras}
        self.calls = []

    def run_inference(self, image_inputs, prompt=None, **kwargs):
        from types import SimpleNamespace

        self.calls.append(prompt)
        camera = self.cameras[image_inputs[0]]
        if camera.image_id in self.blind:
            return None
        canvas = Image.new('1', (camera.width, camera.height), 0)
        from PIL import ImageDraw

        for outline in self.selector.project_prisms(camera).values():
            moved = [(x + self.shift[0], y + self.shift[1]) for x, y in outline]
            ImageDraw.Draw(canvas).polygon(moved, fill=1)
        return SimpleNamespace(masks=[np.asarray(canvas, dtype=bool)[None]])


def test_pose_refinement_moves_the_outline_onto_the_segmented_building(flight):
    folder, cameras = flight
    house = PhysicalAssetCollection([building('house', HOUSE)])
    plain = DroneImageExtractor(cameras, folder / 'plain', max_views=2)
    plain(house)
    unrefined = {v.camera.image_id: v.bbox for v in plain.view_sets['house'].views}
    blind = next(iter(unrefined))
    model = FakeSegmenter(plain.selector, cameras, shift=(12, -8), blind=[blind])
    refined = DroneImageExtractor(
        cameras,
        folder / 'refined',
        max_views=2,
        refine_poses=True,
        refine_model=model,
        refine_prompt='house',
        position_error_m=5.0,
    )
    house = PhysicalAssetCollection([building('house', HOUSE)])
    refined(house)
    assert model.calls == ['house', 'house']
    assert set(refined.pose_shifts) == set(unrefined) - {blind}
    for view in refined.view_sets['house'].views:
        before = unrefined[view.camera.image_id]
        if view.camera.image_id == blind:
            assert view.bbox == before  # nothing segmented: recorded pose kept
            continue
        assert refined.pose_shifts[view.camera.image_id] == (12, -8)
        assert view.bbox[0] == pytest.approx(before[0] + 12)
        assert view.bbox[1] == pytest.approx(before[1] - 8)
    shifts = [i.properties['pose_shift_px'] for i in house.get('house').image_assets]
    assert sorted(shifts) == [[0, 0], [12, -8]]


def test_pose_refinement_ignores_weak_matches(flight):
    folder, cameras = flight
    house = PhysicalAssetCollection([building('house', HOUSE)])
    selector = DroneViewSelector(cameras, max_views=2)
    selector.select(list(house))
    model = FakeSegmenter(selector, cameras, shift=(12, -8))
    picky = DroneImageExtractor(
        cameras,
        folder / 'picky',
        max_views=2,
        refine_poses=True,
        refine_model=model,
        refine_min_overlap=1.5,  # nothing can score this high
    )
    picky(house)
    assert picky.pose_shifts == {}
    # project_prisms before any selection has nothing to project.
    assert DroneViewSelector(cameras).project_prisms(cameras[0]) == {}


def test_extractor_skips_assets_without_geometry_and_loads_sam3_on_demand(
    flight, monkeypatch
):
    folder, cameras = flight
    collection = PhysicalAssetCollection(
        [building('house', HOUSE), PhysicalAsset(id='ghost', geometry=Polygon())]
    )
    loaded = []

    class LazySam:
        def __init__(self):
            loaded.append(self)

        def run_inference(self, image_inputs, prompt=None, **kwargs):
            return None

    import rapidtools.models as models

    monkeypatch.setattr(models, 'SAM3Inference', LazySam, raising=False)
    extractor = DroneImageExtractor(cameras, folder / 'lazy', refine_poses=True)
    extractor(collection)
    assert len(loaded) == 1 and extractor.pose_shifts == {}
    assert 'n_drone_views' not in collection.get('ghost').attributes
    assert collection.get('house').attributes['n_drone_views'] >= 2
    # Nothing but empty geometries: nothing to select.
    assert DroneViewSelector(cameras).select([collection.get('ghost')]) == {}
