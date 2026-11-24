"""Tests for occupancy fusion of street-level sightings (``street_fusion``)."""

import math

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon

from rapidtools.core import Observation
from rapidtools.processing.street_fusion import (
    N_BINS,
    FusedObject,
    Scan,
    fuse_observations,
    object_attributes,
    outline_angles,
    scan_from_observation,
    set_range_bands,
)
from rapidtools.processing.street_localization import local_projection
from rapidtools.processing.street_tracking import CameraFrame

LON0, LAT0 = -117.4, 47.7
CAMERA_HEIGHT = 2.4
PROJECT, UNPROJECT = local_projection(LON0, LAT0)


def _box_outline(camera_xy, center_xy, length=4.5, width=1.8, height=1.5, yaw=0.0):
    """Outline, in a level north-facing panorama, of a box standing on the ground."""
    cx, cy = center_xy
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for dx in (-length / 2, length / 2):
        for dy in (-width / 2, width / 2):
            for z in (0.0, height):
                x = cx + c * dx - s * dy - camera_xy[0]
                y = cy + s * dx + c * dy - camera_xy[1]
                az = math.degrees(math.atan2(x, y))
                el = math.degrees(math.atan2(z - CAMERA_HEIGHT, math.hypot(x, y)))
                pts.append((0.5 + az / 360.0, 0.5 - el / 180.0))
    pts = np.array(pts)
    return [tuple(pts[i]) for i in ConvexHull(pts).vertices]


def _sighting(image_id, camera_xy, polygon, day='2026-08-27', **kw):
    lon, lat = UNPROJECT(*camera_xy)
    return Observation(
        image_id=image_id,
        label='object--vehicle--car',
        polygon=polygon,
        camera_lon=lon,
        camera_lat=lat,
        compass_angle=0.0,
        is_pano=True,
        sequence_id=kw.pop('sequence', 'seq'),
        captured_at=f'{day}T10:00:00+00:00',
        image_width=4096,
        image_height=2048,
        **kw,
    )


def _drive(cars, n=14, spacing=3.0, y=0.0, day='2026-08-27', prefix='f', x0=-20.0):
    """A pass east along ``y``; one sighting per car per frame within 30 m."""
    sightings, frames = [], []
    for i in range(n):
        cam = (x0 + spacing * i, y)
        lon, lat = UNPROJECT(*cam)
        frames.append(
            CameraFrame(f'{prefix}{i}', lon, lat, 0.0, True, f'{day}T10:00:00+00:00')
        )
        for car in cars:
            if isinstance(car, dict):
                center = car['at']
                kwargs = {k: v for k, v in car.items() if k != 'at'}
            else:
                center, kwargs = car, {}
            if math.hypot(center[0] - cam[0], center[1] - cam[1]) > 30.0:
                continue
            sightings.append(
                _sighting(f'{prefix}{i}', cam, _box_outline(cam, center, **kwargs), day)
            )
    return sightings, frames


# ==========================================
# Scans
# ==========================================
def test_outline_angles_with_and_without_a_pose():
    obs = Observation(
        'i', 'car', [(0.5, 0.5), (0.75, 0.5), (0.75, 0.75)], 0.0, 0.0, 90.0
    )
    bearing, elevation = outline_angles(obs)
    assert [round(float(b)) % 360 for b in bearing] == [90, 180, 180]
    assert [round(float(e)) for e in elevation] == [0, 0, -45]
    # The same camera described by a rotation (x east, y down, z north) and no
    # usable compass: the pose wins.
    obs = Observation(
        'i',
        'car',
        [(0.5, 0.5), (0.75, 0.5), (0.75, 0.75)],
        0.0,
        0.0,
        33.0,
        extra={'computed_rotation': [math.pi / 2, 0.0, 0.0]},
    )
    bearing, elevation = outline_angles(obs)
    assert [round(float(b)) % 360 for b in bearing] == [0, 90, 90]
    assert [round(float(e)) for e in elevation] == [0, 0, -45]


def test_outline_angles_of_a_perspective_frame():
    obs = Observation(
        'i',
        'car',
        [(0.5, 0.5), (1.0, 0.5), (1.0, 0.6)],
        0.0,
        0.0,
        0.0,
        is_pano=False,
        image_width=4000,
        image_height=3000,
        extra={'camera_type': 'perspective', 'camera_parameters': [0.5, 0.0, 0.0]},
    )
    bearing, _ = outline_angles(obs)
    assert round(float(bearing[0])) == 0 and round(float(bearing[1])) == 45


def test_scan_reads_lower_and_upper_edge_and_handles_the_seam():
    cam = (0.0, 0.0)
    scan = scan_from_observation(3, _sighting('a', cam, _box_outline(cam, (0.0, 8.0))))
    assert scan.index == 3
    assert abs(scan.center) < 1.0 or abs(scan.center - 360.0) < 1.0
    assert 28.0 < scan.width < 37.0  # 4.5 m seen from about 7 m
    assert len(scan.cols) == len(scan.bottom) == len(scan.top)
    # The outline straddles north, so its columns wrap around the horizon:
    assert scan.cols.min() < 40 and scan.cols.max() > N_BINS - 40
    assert float(scan.bottom.min()) < -16.0 and float(scan.top.max()) > -8.0
    # An outline right across the left/right seam of the panorama:
    seam = _sighting('b', cam, [(0.98, 0.55), (0.02, 0.55), (0.02, 0.6), (0.98, 0.6)])
    scan = scan_from_observation(0, seam)
    assert 13.0 < scan.width < 16.0 and 175.0 < scan.center < 185.0


def test_scan_rejects_outlines_that_circle_the_camera():
    ring = [(x, 0.9) for x in np.linspace(0.0, 1.0, 40)]
    assert scan_from_observation(0, _sighting('a', (0.0, 0.0), ring)) is None


def test_range_bands_follow_the_ground_contact():
    cam = (0.0, 0.0)
    scan = scan_from_observation(0, _sighting('a', cam, _box_outline(cam, (0.0, 8.0))))
    set_range_bands(scan, CAMERA_HEIGHT)
    assert 6.5 < scan.contact < 7.6  # the near side of the car is 7.1 m away
    mid = len(scan.cols) // 2
    assert scan.lo[mid] < 7.1 < scan.hi[mid]
    assert scan.hi[mid] < 16.0  # bounded behind: the car is lower than the camera
    assert np.all(scan.hi > scan.lo)


def test_range_bands_abstain_without_a_ground_contact():
    # An outline entirely above the horizon (a roof, a sign) gives no range.
    obs = _sighting('a', (0.0, 0.0), [(0.5, 0.3), (0.55, 0.3), (0.55, 0.45)])
    scan = set_range_bands(scan_from_observation(0, obs))
    assert math.isnan(scan.contact) and np.isnan(scan.lo).all()
    # A far sighting is wide in range: it says little more than its bearing.
    cam = (0.0, 0.0)
    near = set_range_bands(
        scan_from_observation(0, _sighting('a', cam, _box_outline(cam, (0.0, 6.0))))
    )
    far = set_range_bands(
        scan_from_observation(0, _sighting('a', cam, _box_outline(cam, (0.0, 25.0))))
    )
    assert np.nanmedian(far.hi - far.lo) > 2.0 * np.nanmedian(near.hi - near.lo)


def test_range_bands_split_an_outline_into_faces():
    # Two surfaces at 5 m and 15 m in one outline: the far one must not be
    # dragged to the near one's range.
    scan = Scan(
        0,
        np.arange(20),
        np.r_[np.full(10, -25.6), np.full(10, -9.1)],
        np.full(20, 5.0),
        5.0,
        10.0,
    )
    set_range_bands(scan, CAMERA_HEIGHT)
    assert scan.lo[2] < 5.0 < scan.hi[2] < 12.0
    assert scan.lo[15] < 15.0 < scan.hi[15] and scan.lo[15] > 6.0


# ==========================================
# Fusion
# ==========================================
def test_parked_car_is_found_once_and_placed_within_half_a_meter():
    car = (3.0, 7.0)
    sightings, frames = _drive([car])
    objects, counts, unproject = fuse_observations(
        sightings,
        frames,
        camera_height_m=CAMERA_HEIGHT,
        project=PROJECT,
        unproject=UNPROJECT,
    )
    assert len(objects) == 1 and counts['objects'] == 1
    obj = objects[0]
    assert math.hypot(obj.x - car[0], obj.y - car[1]) < 0.5
    assert obj.localization == 'triangulated' and obj.n_images == len(frames)
    assert obj.parallax_deg > 90.0 and obj.sigma_m < 0.3 and obj.rms_m < 0.3
    assert obj.days == ['2026-08-27']
    # The footprint is car-sized and contains the car's center:
    # (generous in depth: a pass from one side cannot see how deep it is)
    assert 3.0 < obj.length_m < 8.0 and 1.0 < obj.width_m < 6.0
    assert Polygon(obj.footprint).contains(Point(*car))
    # Sightings come back localized:
    assert all(
        o.bearing is not None and o.range_m and o.has_location for o in obj.members
    )
    lon, lat = unproject(obj.x, obj.y)
    assert abs(obj.members[0].lon - lon) < 1e-9 and abs(obj.members[0].lat - lat) < 1e-9
    attributes = object_attributes(obj)
    assert attributes['localization'] == 'triangulated'
    assert attributes['survey_days'] == ['2026-08-27']
    assert attributes['footprint_area_m2'] > 4.0 and 'position_rms_m' in attributes


def test_two_passes_from_different_streets_give_one_object():
    car = (0.0, 8.0)
    first, frames_a = _drive([car], prefix='a')
    # A second pass on the far side of the car, same day, another sequence:
    second, frames_b = _drive([car], y=16.0, prefix='b')
    objects, _, _ = fuse_observations(
        first + second, frames_a + frames_b, project=PROJECT, unproject=UNPROJECT
    )
    assert len(objects) == 1
    assert {o.image_id[0] for o in objects[0].members} == {'a', 'b'}
    assert math.hypot(objects[0].x - car[0], objects[0].y - car[1]) < 0.5


def test_cars_in_a_row_and_side_by_side_stay_apart():
    # Bumper to bumper along the curb with a meter between them:
    row = [(-6.0, 6.0), (-0.5, 6.0), (5.0, 6.0)]
    sightings, frames = _drive(row)
    objects, _, _ = fuse_observations(
        sightings, frames, project=PROJECT, unproject=UNPROJECT
    )
    assert len(objects) == 3
    for car in row:
        assert min(math.hypot(o.x - car[0], o.y - car[1]) for o in objects) < 0.8
    # Door to door on a driveway, seen end-on from the street:
    pair = [
        {'at': (-1.5, 9.0), 'yaw': math.pi / 2},
        {'at': (1.5, 9.0), 'yaw': math.pi / 2},
    ]
    sightings, frames = _drive(pair)
    objects, counts, _ = fuse_observations(
        sightings, frames, project=PROJECT, unproject=UNPROJECT
    )
    assert len(objects) == 2
    assert sorted(round(o.x) for o in objects) in ([-2, 2], [-2, 1], [-1, 2], [-1, 1])
    assert all(o.n_images >= 8 for o in objects)


def test_object_that_moves_with_the_camera_is_not_an_object():
    # The same outline at the same place in every frame: a vehicle driving
    # ahead of the survey car. Each frame claims ground the next one drives
    # through or sees empty.
    sightings, frames = [], []
    for i in range(14):
        cam = (3.0 * i, 0.0)
        lon, lat = UNPROJECT(*cam)
        frames.append(CameraFrame(f'f{i}', lon, lat, 0.0, True, '2026-08-27T10:00:00'))
        ahead = (cam[0] + 9.0, 0.0)
        sightings.append(
            _sighting(f'f{i}', cam, _box_outline(cam, ahead), day='2026-08-27')
        )
    objects, counts, _ = fuse_observations(
        sightings, frames, project=PROJECT, unproject=UNPROJECT
    )
    assert objects == [] and counts['objects'] == 0


def test_single_sighting_is_not_an_object():
    cam = (0.0, 0.0)
    one = [_sighting('f0', cam, _box_outline(cam, (0.0, 7.0)))]
    objects, _, _ = fuse_observations(one, project=PROJECT, unproject=UNPROJECT)
    assert objects == []


def test_survey_days_are_fused_apart_and_joined_afterward():
    stays, leaves = (-4.0, 7.0), (8.0, 7.0)
    day1, frames1 = _drive([stays, leaves], day='2026-08-27', prefix='a')
    day2, frames2 = _drive([stays], day='2026-09-15', prefix='b')
    objects, counts, _ = fuse_observations(
        day1 + day2, frames1 + frames2, project=PROJECT, unproject=UNPROJECT
    )
    # The car that left is not voted away by the later pass that saw the
    # curb empty; the one that stayed is one object seen on both days.
    assert len(objects) == 2 and counts['merged_days'] == 1
    by_x = sorted(objects, key=lambda o: o.x)
    assert by_x[0].days == ['2026-08-27', '2026-09-15']
    assert by_x[1].days == ['2026-08-27']
    assert {o.image_id[0] for o in by_x[0].members} == {'a', 'b'}


def test_empty_frames_clear_what_a_few_far_sightings_claim():
    # Two far sightings of nothing in particular, and a pass right by the spot
    # that saw the curb empty.
    spot = (0.0, 6.0)
    far = [
        _sighting(
            f'x{i}', (-26.0 - 3.0 * i, 0.0), _box_outline((-26.0 - 3.0 * i, 0.0), spot)
        )
        for i in range(2)
    ]
    _, frames = _drive([], n=14)
    with_frames, _, _ = fuse_observations(
        far, frames, project=PROJECT, unproject=UNPROJECT
    )
    assert with_frames == []


def test_tiles_do_not_change_the_result():
    cars = [(-30.0, 6.0), (2.0, 7.0), (33.0, -6.0), (64.0, 6.5)]
    sightings, frames = _drive(cars, n=40, x0=-45.0)
    whole, _, _ = fuse_observations(
        sightings, frames, project=PROJECT, unproject=UNPROJECT
    )
    tiled, _, _ = fuse_observations(
        sightings, frames, project=PROJECT, unproject=UNPROJECT, tile_m=40.0
    )
    assert len(whole) == len(tiled) == 4
    for a in whole:
        assert min(math.hypot(a.x - b.x, a.y - b.y) for b in tiled) < 0.3


def test_no_sightings_and_default_projection():
    objects, counts, unproject = fuse_observations([])
    assert objects == [] and counts['objects'] == 0 and callable(unproject)
    sightings, _ = _drive([(3.0, 7.0)])
    objects, _, unproject = fuse_observations(sightings)  # frames from the sightings
    assert len(objects) == 1
    lon, lat = unproject(objects[0].x, objects[0].y)
    x, y = PROJECT(lon, lat)
    assert math.hypot(x - 3.0, y - 7.0) < 0.6


def test_min_object_width_drops_small_blobs():
    sightings, frames = _drive([{'at': (3.0, 7.0), 'length': 0.6, 'width': 0.6}])
    kept, _, _ = fuse_observations(
        sightings,
        frames,
        project=PROJECT,
        unproject=UNPROJECT,
        cell_m=0.1,
        min_object_width_m=0.2,
    )
    dropped, counts, _ = fuse_observations(
        sightings,
        frames,
        project=PROJECT,
        unproject=UNPROJECT,
        cell_m=0.1,
        min_object_width_m=5.0,
    )
    assert len(kept) == 1 and dropped == [] and counts['fragment'] == 1


def test_fused_object_without_an_intersection_reports_a_coarse_sigma():
    obj = FusedObject(
        0.0, 0.0, None, [], [(0, 0), (4, 0), (4, 2), (0, 2)], 8.0, 4.0, 2.0, 3.0
    )
    assert obj.sigma_m == 1.0 and obj.n_images == 0
    assert 'position_rms_m' not in object_attributes(obj)


def test_perspective_frames_only_clear_what_they_can_see():
    car = (3.0, 7.0)
    sightings, frames = _drive([car])
    # Forward-looking perspective frames from a second camera that never
    # looks at the car: they must not vote its cells free.
    blind = []
    for i, f in enumerate(frames):
        blind.append(
            CameraFrame(f'p{i}', f.lon, f.lat, 180.0, False, f.captured_at, 'p', 0.85)
        )
    objects, _, _ = fuse_observations(
        sightings, frames + blind, project=PROJECT, unproject=UNPROJECT
    )
    assert len(objects) == 1


@pytest.mark.parametrize('threshold, expected', [(0.6, 1), (1.01, 0)])
def test_occupancy_threshold(threshold, expected):
    sightings, frames = _drive([(3.0, 7.0)])
    objects, _, _ = fuse_observations(
        sightings,
        frames,
        project=PROJECT,
        unproject=UNPROJECT,
        occupancy_threshold=threshold,
    )
    assert len(objects) == expected
