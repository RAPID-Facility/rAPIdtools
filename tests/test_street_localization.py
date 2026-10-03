"""Tests for the street-level localisation geometry."""

import math

import pytest

from rapidtools.core import Observation
from rapidtools.processing.street_localization import (
    bbox_iou,
    cluster_observations,
    destination_point,
    estimate_ego_mask,
    ground_range,
    haversine_m,
    intersect_bearings,
    local_projection,
    localize,
    thin_frames,
    view_angles,
)

LON0, LAT0 = -117.45, 47.70


def _box(cx, cy, w=0.06, h=0.08):
    return [
        (cx - w / 2, cy - h / 2),
        (cx + w / 2, cy - h / 2),
        (cx + w / 2, cy + h / 2),
        (cx - w / 2, cy + h / 2),
    ]


def _obs(image_id='1', polygon=None, compass=0.0, lon=LON0, lat=LAT0, **kw):
    return Observation(
        image_id=image_id,
        label='object--vehicle--car',
        polygon=polygon or _box(0.5, 0.6),
        camera_lon=lon,
        camera_lat=lat,
        compass_angle=compass,
        **kw,
    )


# ------------------------------------------------------------------ basics
def test_haversine_and_destination_round_trip():
    lon, lat = destination_point(LON0, LAT0, 37.0, 250.0)
    assert haversine_m(LON0, LAT0, lon, lat) == pytest.approx(250.0, abs=0.01)
    # Bearing recovered from the local frame:
    project, _ = local_projection(LON0, LAT0)
    x, y = project(lon, lat)
    assert math.degrees(math.atan2(x, y)) == pytest.approx(37.0, abs=0.01)


def test_local_projection_round_trip():
    project, unproject = local_projection(LON0, LAT0)
    x, y = project(LON0 + 0.001, LAT0 - 0.0005)
    assert x > 0 and y < 0
    lon, lat = unproject(x, y)
    assert (lon, lat) == pytest.approx((LON0 + 0.001, LAT0 - 0.0005), abs=1e-9)


def test_bbox_iou():
    assert bbox_iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0
    assert bbox_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    assert bbox_iou((0, 0, 2, 2), (1, 1, 3, 3)) == pytest.approx(1 / 7)


# ------------------------------------------------------------------ angles
def test_view_angles_pano_centre_and_edges():
    bearing, elevation = view_angles(_box(0.5, 0.5), compass_angle=90.0)
    assert bearing == pytest.approx(90.0)
    assert elevation == pytest.approx(-0.04 * 180)  # bottom edge 4% below centre
    bearing, _ = view_angles(_box(0.25, 0.5), compass_angle=90.0)
    assert bearing == pytest.approx(0.0)  # a quarter turn to the left
    bearing, _ = view_angles(_box(0.75, 0.5), compass_angle=90.0)
    assert bearing == pytest.approx(180.0)


def test_view_angles_pano_seam():
    """A box straddling the left/right seam is centred on the seam, not mid-image."""
    seam_box = [
        (0.97, 0.5),
        (1.0, 0.5),
        (0.0, 0.5),
        (0.03, 0.5),
        (0.03, 0.6),
        (0.97, 0.6),
    ]
    bearing, _ = view_angles(seam_box, compass_angle=0.0)
    assert bearing == pytest.approx(180.0, abs=1e-6)


def test_view_angles_perspective():
    bearing, elevation = view_angles(_box(0.5, 0.5), compass_angle=45.0, is_pano=False)
    assert bearing == pytest.approx(45.0)
    assert elevation < 0
    right, _ = view_angles(
        _box(0.9, 0.5), compass_angle=45.0, is_pano=False, focal_norm=1.0
    )
    assert right == pytest.approx(45.0 + math.degrees(math.atan(0.4)))


def test_ground_range():
    assert ground_range(-45.0, 2.4) == pytest.approx(2.4)
    assert ground_range(-5.0, 2.4) == pytest.approx(2.4 / math.tan(math.radians(5.0)))
    assert ground_range(-0.5, 2.4) is None
    assert ground_range(10.0, 2.4) is None


# ------------------------------------------------------------------ localize
def test_localize_fills_position():
    obs = _obs(polygon=_box(0.75, 0.55), compass=0.0)  # right of centre, below horizon
    assert localize(obs, camera_height_m=2.4) is True
    assert obs.bearing == pytest.approx(90.0)
    assert obs.range_m == pytest.approx(2.4 / math.tan(math.radians(0.09 * 180)))
    assert obs.lon > LON0 and obs.lat == pytest.approx(LAT0, abs=1e-7)
    assert haversine_m(LON0, LAT0, obs.lon, obs.lat) == pytest.approx(
        obs.range_m, abs=0.01
    )


@pytest.mark.parametrize(
    'polygon, kwargs',
    [
        (_box(0.5, 0.6, w=0.01, h=0.01), {}),  # too small
        (_box(0.5, 0.45), {}),  # above the horizon
        (_box(0.5, 0.95), {}),  # under the camera: range below min
        (_box(0.5, 0.52), {'max_range_m': 10.0}),  # too far
    ],
)
def test_localize_rejects_implausible(polygon, kwargs):
    obs = _obs(polygon=polygon)
    assert localize(obs, **kwargs) is False
    assert not obs.has_location


# ------------------------------------------------------------------ triangulation
def test_intersect_bearings_recovers_point():
    project, unproject = local_projection(LON0, LAT0)
    target = (5.0, 12.0)
    rays = []
    for cam in ((-10.0, 0.0), (0.0, 0.0), (10.0, 0.0)):
        bearing = math.degrees(math.atan2(target[0] - cam[0], target[1] - cam[1])) % 360
        lon, lat = unproject(*cam)
        rays.append((lon, lat, bearing))
    lon, lat, rms = intersect_bearings(rays, project, unproject)
    assert project(lon, lat) == pytest.approx(target, abs=0.01)
    assert rms == pytest.approx(0.0, abs=0.01)


def test_intersect_bearings_rejects_degenerate_cases():
    project, unproject = local_projection(LON0, LAT0)
    a = unproject(-10.0, 0.0)
    b = unproject(10.0, 0.0)
    # Same camera twice:
    assert intersect_bearings([(*a, 30.0), (*a, 40.0)], project, unproject) is None
    # Parallel bearings:
    assert intersect_bearings([(*a, 0.0), (*b, 2.0)], project, unproject) is None
    # Intersection behind the cameras (both look away from each other's side):
    assert intersect_bearings([(*a, 315.0), (*b, 45.0)], project, unproject) is None
    # Too far away:
    far = intersect_bearings(
        [(*a, 89.0), (*b, 271.0)], project, unproject, max_range_m=5
    )
    assert far is None


# ------------------------------------------------------------------ filters
def test_estimate_ego_mask_flags_recurring_box_only():
    ego_box = _box(0.5, 0.9, w=0.2, h=0.2)
    frames = [_obs(str(i), ego_box, sequence_id='s') for i in range(5)]
    # A parked car drifting across the frame:
    parked = [_obs(str(i), _box(0.2 + 0.1 * i, 0.6), sequence_id='s') for i in range(5)]
    flagged = estimate_ego_mask(frames + parked)
    assert flagged == set(range(5))


def test_estimate_ego_mask_ignores_short_sequences_and_other_sequences():
    ego_box = _box(0.5, 0.9, w=0.2, h=0.2)
    short = [_obs(str(i), ego_box, sequence_id='short') for i in range(2)]
    assert estimate_ego_mask(short) == set()
    # Same box in two different sequences of 3 frames each: both flagged
    # independently.
    s1 = [_obs(f'a{i}', ego_box, sequence_id='s1') for i in range(3)]
    s2 = [_obs(f'b{i}', ego_box, sequence_id='s2') for i in range(3)]
    assert estimate_ego_mask(s1 + s2) == set(range(6))


def test_thin_frames():
    frames = [
        ('a', LON0, LAT0, '2025-01-01T00:00:00', 's'),
        ('b', *destination_point(LON0, LAT0, 90, 3), '2025-01-01T00:00:01', 's'),
        ('c', *destination_point(LON0, LAT0, 90, 12), '2025-01-01T00:00:02', 's'),
        ('d', *destination_point(LON0, LAT0, 90, 14), '2025-01-01T00:00:03', 's'),
        ('e', LON0, LAT0, '2025-01-01T00:00:00', 'other'),
    ]
    assert thin_frames(frames, 0) == {'a', 'b', 'c', 'd', 'e'}
    assert thin_frames(frames, 10) == {'a', 'c', 'e'}


def test_cluster_observations_groups_nearby_views():
    near = []
    for i, (bearing, dist) in enumerate(((0, 1.0), (90, 1.5), (180, 0.5))):
        o = _obs(str(i))
        o.lon, o.lat = destination_point(LON0, LAT0, bearing, dist)
        o.range_m = 5.0 + i
        near.append(o)
    far = _obs('far')
    far.lon, far.lat = destination_point(LON0, LAT0, 45, 40.0)
    far.range_m = 8.0
    unlocated = _obs('nowhere')
    groups = cluster_observations(near + [far, unlocated], radius_m=4.0)
    assert sorted(len(g) for g in groups) == [1, 3]
    assert cluster_observations([unlocated], 4.0) == []


def test_simplify_polygon_keeps_shape_and_handles_edge_cases():
    from rapidtools.processing.street_localization import simplify_polygon

    ring = [(0.1, 0.1), (0.2, 0.1001), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)]
    assert simplify_polygon(ring, 0.002) == [
        (0.1, 0.1),
        (0.3, 0.1),
        (0.3, 0.3),
        (0.1, 0.3),
    ]
    # Below the tolerance the bump is kept:
    assert len(simplify_polygon(ring, 0.00001)) == 5
    # Disabled, or already minimal, rings pass through unchanged:
    assert simplify_polygon(ring, 0) == ring
    box = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert simplify_polygon(box, 0.5) == [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ]
    # A degenerate ring (all collinear) falls back to the input:
    line = [(0, 0), (0.5, 0), (1, 0), (0.25, 0), (0.75, 0)]
    assert simplify_polygon(line, 0.01) == [
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (0.25, 0.0),
        (0.75, 0.0),
    ]


def test_estimate_ego_mask_scales_to_a_survey_length_sequence():
    """A day-long sequence (20k frames) must not be quadratic in the ego box."""
    import random
    import time

    rng = random.Random(7)
    ego_box = _box(0.5, 0.9, w=0.2, h=0.2)
    obs = []
    ego_expected = set()
    for i in range(20000):
        # The survey vehicle, with a pixel of jitter, plus three parked cars
        # spread over the frame:
        jitter = rng.uniform(-0.002, 0.002)
        ego_expected.add(len(obs))
        obs.append(
            _obs(
                f'f{i}',
                [(x + jitter, y) for x, y in ego_box],
                sequence_id='day',
            )
        )
        for _ in range(3):
            obs.append(
                _obs(
                    f'f{i}',
                    _box(rng.uniform(0.1, 0.9), rng.uniform(0.4, 0.8), w=0.06, h=0.05),
                    sequence_id='day',
                )
            )
    t0 = time.perf_counter()
    flagged = estimate_ego_mask(obs)
    elapsed = time.perf_counter() - t0
    assert flagged == ego_expected
    assert elapsed < 10, f'ego mask took {elapsed:.1f} s for 80k sightings'


# ==========================================
# Camera model (full pose)
# ==========================================


def _rotation_vector(yaw_deg, pitch_deg=0.0, roll_deg=0.0):
    """OpenSfM world->camera axis-angle for a camera heading ``yaw_deg``."""
    import numpy as np

    def rz(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def rx(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    # Camera axes (right, down, forward) = (east, -up, north) before yaw:
    base = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    r = base @ rz(math.radians(-yaw_deg)).T
    r = rz(math.radians(roll_deg)) @ rx(math.radians(pitch_deg)) @ r
    angle = math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2)))
    if angle < 1e-9:
        return [0.0, 0.0, 0.0]
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return list(axis / (2 * math.sin(angle)) * angle)


def _project_pano(point_enu, rotation_vector):
    """Normalised pano pixel of a point given in camera-centred ENU metres."""
    import numpy as np

    from rapidtools.processing.street_localization import rotation_matrix

    d = rotation_matrix(rotation_vector) @ np.asarray(point_enu, dtype=float)
    d = d / np.linalg.norm(d)
    lon = math.atan2(d[0], d[2])
    lat = -math.asin(d[1])
    return (0.5 + lon / (2 * math.pi)) % 1.0, 0.5 - lat / math.pi


def test_rotation_matrix_identity_and_heading():
    import numpy as np

    from rapidtools.processing.street_localization import rotation_matrix

    assert np.allclose(rotation_matrix([0, 0, 0]), np.eye(3))
    # A camera heading 95 degrees: its forward axis points 95 deg from north.
    r = rotation_matrix(_rotation_vector(95.0))
    forward = r.T @ np.array([0.0, 0.0, 1.0])
    heading = math.degrees(math.atan2(forward[0], forward[1])) % 360
    assert heading == pytest.approx(95.0, abs=1e-6)
    assert forward[2] == pytest.approx(0.0, abs=1e-9)


def test_pixel_bearing_models():
    from rapidtools.processing.street_localization import pixel_bearing

    assert pixel_bearing(0.5, 0.5) == pytest.approx([0, 0, 1], abs=1e-9)
    assert pixel_bearing(0.75, 0.5) == pytest.approx([1, 0, 0], abs=1e-9)
    assert pixel_bearing(0.5, 1.0) == pytest.approx([0, 1, 0], abs=1e-9)
    # Perspective: the principal point looks straight ahead; a point one focal
    # length to the right of it is 45 degrees off axis.
    centre = pixel_bearing(0.5, 0.5, 'perspective', 4000, 3000, [0.9, 0, 0])
    assert centre == pytest.approx([0, 0, 1], abs=1e-9)
    right = pixel_bearing(0.5 + 0.9, 0.5, 'perspective', 4000, 3000, [0.9, 0, 0])
    assert math.degrees(math.atan2(right[0], right[2])) == pytest.approx(45.0)
    # Distortion is undone: a distorted point maps back to the ideal direction.
    k1 = -0.1
    xu = 0.3
    xd = xu * (1 + k1 * xu * xu)
    undone = pixel_bearing(0.5 + xd * 1.0, 0.5, 'brown', 4000, 4000, [1.0, k1, 0])
    assert undone[0] / undone[2] == pytest.approx(xu, abs=1e-6)
    # Fisheye: equidistant model, 0.5 rad off axis.
    theta = 0.5
    fish = pixel_bearing(0.5 + 0.3 * theta, 0.5, 'fisheye', 2000, 2000, [0.3, 0, 0])
    assert math.atan2(fish[0], fish[2]) == pytest.approx(theta, abs=1e-6)


def test_localize_with_full_pose_recovers_position_under_pitch_and_roll():
    from rapidtools.processing.street_localization import haversine_m

    cam_h = 2.4
    for yaw, pitch, roll, bearing, rng in [
        (95.0, -1.1, 0.5, 40.0, 12.0),
        (330.0, 3.0, -2.0, 200.0, 20.0),
        (10.0, -4.0, 4.0, 95.0, 6.0),
    ]:
        rv = _rotation_vector(yaw, pitch, roll)
        target = destination_point(LON0, LAT0, bearing, rng)
        e = rng * math.sin(math.radians(bearing))
        n = rng * math.cos(math.radians(bearing))
        x_c, y_c = _project_pano([e, n, -cam_h + 0.8], rv)
        x_b, y_b = _project_pano([e, n, -cam_h], rv)
        polygon = [
            (x_c - 0.01, y_c - 0.01),
            (x_c + 0.01, y_c - 0.01),
            (x_c + 0.01, y_b),
            (x_c - 0.01, y_b),
        ]
        posed = _obs(
            polygon=polygon,
            compass=yaw,
            image_width=8192,
            image_height=4096,
            extra={'computed_rotation': rv, 'camera_type': 'spherical'},
        )
        level = _obs(polygon=polygon, compass=yaw, image_width=8192, image_height=4096)
        assert localize(posed, camera_height_m=cam_h, max_range_m=60)
        assert haversine_m(posed.lon, posed.lat, *target) < 0.1
        assert posed.bearing == pytest.approx(bearing, abs=0.3)
        assert posed.range_m == pytest.approx(rng, rel=0.01)
        # The level-camera fallback is wrong by metres once the camera pitches:
        if abs(pitch) >= 3.0:
            assert localize(level, camera_height_m=cam_h, max_range_m=60)
            assert haversine_m(level.lon, level.lat, *target) > 1.0


def test_localize_without_rotation_uses_level_camera_fallback():
    obs = _obs(polygon=_box(0.75, 0.6), compass=10.0)
    assert localize(obs, camera_height_m=2.4)
    expected_bearing, expected_elev = view_angles(obs.polygon, 10.0, True)
    assert obs.bearing == pytest.approx(expected_bearing)
    assert obs.elevation == pytest.approx(expected_elev)
