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
# 10-03-2026

"""
Synthetic-survey tests for tracking, triangulation, merging and voting.

A camera drives along a street taking a level panorama every few metres.
Parked cars at known positions (and one car driving the other way) are
projected into each frame with realistic noise, and the pipeline has to
give back one object per parked car, close to where it really is, and no
object for the moving one.
"""

import math
import random

import numpy as np
import pytest

from rapidtools.core import Observation
from rapidtools.processing.street_localization import (
    haversine_m,
    local_projection,
    localize,
)
from rapidtools.processing.street_tracking import (
    CameraFrame,
    ObjectEstimate,
    discover_objects,
    merge_by_rays,
    merge_estimates,
    merge_pieces,
    prune_unwitnessed,
    sighting_shows,
    single_view_covariance,
    suppress_duplicate_sightings,
    track_sequence,
    triangulate_track,
    vote_rays,
)

LON0, LAT0 = -117.49, 47.71
CAM_H = 2.4
LABEL = 'object--vehicle--car'


def _pano_polygon(
    cam_xy,
    heading,
    obj_xy,
    rng,
    noise_deg=0.4,
    range_noise=0.15,
    occlusion=1.0,
    width_m=2.0,
    height_m=1.5,
):
    """
    Normalised pano outline of an object seen from ``cam_xy``.

    ``occlusion`` > 1 raises the visible bottom edge as a fence or a parked
    car hiding the wheels would: the ground contact then reads that many
    times too far while the top edge stays where it is.
    """
    dx, dy = obj_xy[0] - cam_xy[0], obj_xy[1] - cam_xy[1]
    dist = math.hypot(dx, dy)
    bearing = math.degrees(math.atan2(dx, dy)) + rng.gauss(0.0, noise_deg)
    rel = ((bearing - heading + 180.0) % 360.0) - 180.0
    cx = 0.5 + rel / 360.0
    # The ground contact as the mask sees it: the range is biased/noisy.
    seen_dist = dist * (1.0 + rng.gauss(0.0, range_noise)) * occlusion
    y_bottom = 0.5 + math.degrees(math.atan2(CAM_H, seen_dist)) / 180.0
    y_top = 0.5 + math.degrees(math.atan2(CAM_H - height_m, dist)) / 180.0
    y_top = min(y_top, y_bottom - 0.002)
    half_w = math.degrees(math.atan2(width_m / 2.0, dist)) / 360.0
    return [
        ((cx - half_w) % 1.0, y_top),
        ((cx + half_w) % 1.0, y_top),
        ((cx + half_w) % 1.0, y_bottom),
        ((cx - half_w) % 1.0, y_bottom),
    ]


def _survey(
    cars,
    moving=None,
    n_frames=14,
    spacing_m=3.0,
    heading=90.0,
    sequence='seq-a',
    seed=1,
    noise_deg=0.4,
    range_noise=0.15,
    max_range=35.0,
    frame_prefix='f',
    occlusion=1.0,
    width_m=2.0,
):
    """
    Sightings of ``cars`` (local-frame metres) from a camera driving east
    along y = 0. ``moving`` is an optional (start_xy, velocity_per_frame).
    Returns the observations and the unproject function for the frame.
    """
    rng = random.Random(seed)
    project, unproject = local_projection(LON0, LAT0)
    observations = []
    for k in range(n_frames):
        cam_xy = (spacing_m * k, 0.0)
        cam_lon, cam_lat = unproject(*cam_xy)
        targets = list(cars)
        if moving is not None:
            (sx, sy), (vx, vy) = moving
            targets.append((sx + vx * k, sy + vy * k))
        for obj in targets:
            if math.hypot(obj[0] - cam_xy[0], obj[1] - cam_xy[1]) > max_range:
                continue
            observations.append(
                Observation(
                    image_id=f'{frame_prefix}{k:03d}',
                    label=LABEL,
                    polygon=_pano_polygon(
                        cam_xy,
                        heading,
                        obj,
                        rng,
                        noise_deg,
                        range_noise,
                        occlusion=occlusion,
                        width_m=width_m,
                    ),
                    camera_lon=cam_lon,
                    camera_lat=cam_lat,
                    compass_angle=heading,
                    is_pano=True,
                    sequence_id=sequence,
                    captured_at=f'2025-08-20T10:00:{k:02d}',
                    image_width=8192,
                    image_height=4096,
                )
            )
    return observations, project, unproject


def _positions(estimates, unproject):
    return [unproject(e.x, e.y) for e in estimates]


def _match(truth_xy, estimates, unproject, tol_m):
    """For each true object, the estimates within ``tol_m`` of it."""
    _, unp = unproject, unproject
    found = []
    for tx, ty in truth_xy:
        tlon, tlat = unp(tx, ty)
        hits = [
            e
            for e, (lon, lat) in zip(estimates, _positions(estimates, unp), strict=True)
            if haversine_m(lon, lat, tlon, tlat) <= tol_m
        ]
        found.append(hits)
    return found


CARS = [(6.0, 5.0), (12.0, -5.5), (18.0, 5.0), (24.0, 5.0), (30.0, -5.5)]


# ==========================================
# Tracking
# ==========================================


def test_track_sequence_one_track_per_car():
    obs, project, _ = _survey(CARS)
    from rapidtools.processing.street_localization import localize

    for o in obs:
        localize(o, camera_height_m=CAM_H, max_range_m=40)
    tracks = track_sequence(obs, project)
    long_tracks = [t for t in tracks if len(t) >= 3]
    assert len(long_tracks) == len(CARS)
    # No track contains two sightings from the same frame:
    for track in tracks:
        ids = [o.image_id for o in track]
        assert len(ids) == len(set(ids))


def test_track_sequence_does_not_merge_two_cars_in_one_frame():
    obs, project, _ = _survey([(10.0, 5.0), (10.0, 5.0 + 2.5)], n_frames=8)
    from rapidtools.processing.street_localization import localize

    for o in obs:
        localize(o, camera_height_m=CAM_H, max_range_m=40)
    tracks = track_sequence(obs, project)
    assert sorted(len(t) for t in tracks)[-2:] == [8, 8]


# ==========================================
# Triangulation
# ==========================================


def test_triangulate_track_static_car_is_precise():
    obs, project, unproject = _survey([(15.0, 6.0)])
    from rapidtools.processing.street_localization import localize

    for o in obs:
        localize(o, camera_height_m=CAM_H, max_range_m=40)
    est = triangulate_track(obs, project)
    assert est is not None
    assert est.localization == 'triangulated'
    assert math.hypot(est.x - 15.0, est.y - 6.0) < 0.6
    assert est.sigma_m < 0.6
    assert est.n_images == len({o.image_id for o in obs})


def test_triangulate_track_rejects_moving_car():
    # A car driving west at 4 m per frame while the camera drives east:
    obs, project, _ = _survey([], moving=((40.0, -4.0), (-4.0, 0.0)), n_frames=10)
    from rapidtools.processing.street_localization import localize

    for o in obs:
        localize(o, camera_height_m=CAM_H, max_range_m=40)
    assert triangulate_track(obs, project) is None


def test_triangulate_track_without_parallax_falls_back_to_single_view():
    # Object straight ahead of a camera that stands still (one position):
    project, unproject = local_projection(LON0, LAT0)
    from rapidtools.processing.street_localization import localize

    obs = []
    rng = random.Random(3)
    for k in range(3):
        obs.append(
            Observation(
                image_id=f's{k}',
                label=LABEL,
                polygon=_pano_polygon((0.0, 0.0), 90.0, (12.0, 0.0), rng, 0.2, 0.05),
                camera_lon=LON0,
                camera_lat=LAT0,
                compass_angle=90.0,
                image_width=8192,
                image_height=4096,
            )
        )
        localize(obs[-1], camera_height_m=CAM_H, max_range_m=40)
    est = triangulate_track(obs, project)
    assert est is not None
    assert est.localization == 'single_view'
    assert math.hypot(est.x - 12.0, est.y) < 2.0
    # Elongated along the line of sight (east), narrow across it:
    assert est.cov[0, 0] > 4 * est.cov[1, 1]


# ==========================================
# Merging
# ==========================================


def test_single_view_covariance_is_elongated_along_the_ray():
    cov = single_view_covariance(0.0, 10.0)  # looking north
    assert cov[1, 1] > 10 * cov[0, 0]


def test_merge_estimates_fuses_same_object_and_keeps_shared_frame_apart():
    a = ObjectEstimate(0.0, 0.0, np.eye(2) * 0.2, [], 'triangulated')
    b = ObjectEstimate(0.5, 0.3, np.eye(2) * 0.8, [], 'single_view')
    # c is within the statistical gate of a but was seen in the same frame
    # as a, and far enough away to be a second car: different object.
    o1 = Observation('frame-1', LABEL, [(0, 0), (0.1, 0), (0.1, 0.1)], LON0, LAT0, 0.0)
    o2 = Observation('frame-1', LABEL, [(0, 0), (0.1, 0), (0.1, 0.1)], LON0, LAT0, 0.0)
    a.members.append(o1)
    c = ObjectEstimate(1.8, -0.3, np.eye(2) * 0.8, [o2], 'single_view')
    assert (
        len(
            merge_estimates(
                [a, ObjectEstimate(1.8, -0.3, np.eye(2) * 0.8, [], 'single_view')]
            )
        )
        == 1
    )
    merged = merge_estimates([a, b, c], max_distance_m=6.0)
    assert len(merged) == 2
    fused = min(merged, key=lambda e: e.x)
    assert fused.localization == 'triangulated'
    assert 0.0 < fused.x < 0.5  # pulled only slightly towards the vaguer view


def test_merge_estimates_fuses_two_fixes_on_one_spot_despite_shared_frame():
    # Mapillary sometimes cuts one car into two outlines in the same frame.
    # Two solid fixes less than a car apart cannot be two cars.
    o1 = Observation('frame-1', LABEL, [(0, 0), (0.1, 0), (0.1, 0.1)], LON0, LAT0, 0.0)
    o2 = Observation(
        'frame-1', LABEL, [(0.2, 0), (0.3, 0), (0.3, 0.1)], LON0, LAT0, 0.0
    )
    a = ObjectEstimate(0.0, 0.0, np.eye(2) * 0.2, [o1], 'triangulated')
    b = ObjectEstimate(1.1, 0.2, np.eye(2) * 0.2, [o2], 'triangulated')
    assert len(merge_estimates([a, b], max_distance_m=6.0)) == 1
    assert len(merge_by_rays([a, b], bearing_sigma_deg=0.75)) == 1


def test_merge_estimates_respects_max_distance():
    a = ObjectEstimate(0.0, 0.0, np.eye(2) * 25.0, [], 'single_view')
    b = ObjectEstimate(7.0, 0.0, np.eye(2) * 25.0, [], 'single_view')
    assert len(merge_estimates([a, b], max_distance_m=6.0)) == 2
    assert len(merge_estimates([a, b], max_distance_m=8.0)) == 1


# ==========================================
# End to end: tracks and voting on the same survey
# ==========================================


@pytest.mark.parametrize('method,tol', [('tracks', 1.0), ('voting', 1.5)])
def test_discover_objects_finds_each_parked_car_once(method, tol):
    obs, _, unproject = _survey(CARS, moving=((45.0, -3.0), (-4.5, 0.0)))
    estimates, counts, unp = discover_objects(
        obs, method=method, camera_height_m=CAM_H, max_range_m=35.0
    )
    hits = _match(CARS, estimates, unp, tol)
    assert all(len(h) == 1 for h in hits), [len(h) for h in hits]
    assert len(estimates) == len(CARS), counts
    if method == 'tracks':
        assert counts['moving'] >= 1
        assert all(e.localization == 'triangulated' for e in estimates)


def test_discover_objects_two_passes_become_one_object_per_car():
    """The same street driven twice (two sequences) must not double count."""
    first, _, _ = _survey(CARS, sequence='seq-a', seed=1, frame_prefix='a')
    second, _, _ = _survey(CARS, sequence='seq-b', seed=2, frame_prefix='b')
    estimates, counts, unp = discover_objects(
        first + second, method='tracks', camera_height_m=CAM_H, max_range_m=35.0
    )
    assert len(estimates) == len(CARS), counts
    assert all(len(h) == 1 for h in _match(CARS, estimates, unp, 1.0))
    assert all(e.sequence_ids == {'seq-a', 'seq-b'} for e in estimates)


def test_discover_objects_handles_radial_scatter_without_duplicates():
    """Large range noise used to split one car along its ray; now it does not."""
    obs, _, _ = _survey(CARS, range_noise=0.35, seed=5)
    estimates, counts, unp = discover_objects(
        obs, method='tracks', camera_height_m=CAM_H, max_range_m=35.0
    )
    assert len(estimates) == len(CARS), counts
    assert all(len(h) == 1 for h in _match(CARS, estimates, unp, 1.2))


def test_discover_objects_rejects_unknown_method():
    with pytest.raises(ValueError):
        discover_objects([], method='magic')


def test_vote_rays_empty():
    project, _ = local_projection(LON0, LAT0)
    assert vote_rays([], project) == []


# ==========================================
# Far, occluded and fragmentary objects
# ==========================================


def test_discover_objects_keeps_far_cars_within_the_range_bound():
    """Driveway cars 40-55 m out are found once the area knob is gone."""
    far = [(10.0, 40.0), (25.0, -45.0), (40.0, 52.0)]
    obs, _, _ = _survey(far, n_frames=24, max_range=70.0, range_noise=0.25)
    estimates, counts, unp = discover_objects(
        obs, method='tracks', camera_height_m=CAM_H, max_range_m=60.0
    )
    assert len(estimates) == len(far), counts
    assert all(len(h) == 1 for h in _match(far, estimates, unp, 1.5))


def test_occluded_static_car_is_kept_not_called_moving():
    """Hidden wheels make the range read 1.8x too far; the car still stands."""
    from rapidtools.processing.street_localization import localize
    from rapidtools.processing.street_tracking import classify_track

    obs, project, _ = _survey([(15.0, 6.0)], occlusion=1.8, range_noise=0.05)
    for o in obs:
        localize(o, camera_height_m=CAM_H, max_range_m=60)
    est, reason = classify_track(obs, project, camera_height_m=CAM_H)
    assert reason == 'ok' and est is not None
    assert est.localization == 'triangulated'
    assert math.hypot(est.x - 15.0, est.y - 6.0) < 0.6


def test_fully_visible_moving_car_is_still_rejected():
    from rapidtools.processing.street_localization import localize
    from rapidtools.processing.street_tracking import classify_track

    obs, project, _ = _survey([], moving=((40.0, -4.0), (-4.0, 0.0)), n_frames=10)
    for o in obs:
        localize(o, camera_height_m=CAM_H, max_range_m=60)
    est, reason = classify_track(obs, project, camera_height_m=CAM_H)
    assert est is None and reason == 'moving'


def test_fragment_narrower_than_an_object_is_dropped():
    from rapidtools.processing.street_localization import localize
    from rapidtools.processing.street_tracking import classify_track

    obs, project, _ = _survey([(12.0, 5.0)], width_m=0.3, range_noise=0.05)
    for o in obs:
        localize(o, camera_height_m=CAM_H, max_range_m=60)
    est, reason = classify_track(obs, project, camera_height_m=CAM_H)
    assert est is None and reason == 'fragment'
    estimates, counts, _ = discover_objects(obs, camera_height_m=CAM_H)
    assert estimates == [] and counts['fragment'] == 1


def test_long_track_seen_mostly_from_far_away_is_static():
    """Rays from cameras beyond the range bound still vote for the consensus."""
    from rapidtools.processing.street_localization import localize
    from rapidtools.processing.street_tracking import classify_track

    # 150 frames along 450 m of road; the car at (60, 12) is within 60 m of
    # only a third of them, yet all of them see it.
    obs, project, _ = _survey([(60.0, 12.0)], n_frames=150, max_range=500.0)
    for o in obs:
        localize(o, camera_height_m=CAM_H, max_range_m=60)
    est, reason = classify_track(obs, project, max_range_m=60.0, camera_height_m=CAM_H)
    assert reason == 'ok' and est is not None
    assert est.localization == 'triangulated'
    assert math.hypot(est.x - 60.0, est.y - 12.0) < 0.5


# ==========================================
# Fragments, long vehicles and ray-based association
# ==========================================


def test_car_close_to_the_road_is_one_object_despite_fast_bearing_swing():
    """Near the camera the bearing jumps 30+ degrees per frame; no fragments."""
    obs, _, unp = _survey([(12.0, 3.5)], n_frames=14, range_noise=0.3)
    estimates, counts, unp = discover_objects(obs, camera_height_m=CAM_H)
    assert len(estimates) == 1, counts
    assert estimates[0].localization == 'triangulated'
    assert len(_match([(12.0, 3.5)], estimates, unp, 1.0)[0]) == 1


def test_long_vehicle_seen_from_both_ends_is_one_object():
    """A 9 m truck: the visible centre shifts 4 m between approach and pass.

    Halves farther apart than half the measured length are not reunited:
    that distance is also what separates two cars parked side by side.
    """
    rng = random.Random(4)
    project, unproject = local_projection(LON0, LAT0)
    obs = []
    for k in range(16):
        cam = (3.0 * k, 0.0)
        centre = (24.0 - 2.0, 5.0) if cam[0] < 24.0 else (24.0 + 2.0, 5.0)
        lon, lat = unproject(*cam)
        obs.append(
            Observation(
                image_id=f't{k:03d}',
                label=LABEL,
                polygon=_pano_polygon(cam, 90.0, centre, rng, 0.4, 0.15, width_m=9.0),
                camera_lon=lon,
                camera_lat=lat,
                compass_angle=90.0,
                is_pano=True,
                sequence_id='seq-a',
                captured_at=f'2025-08-20T10:00:{k:02d}',
                image_width=8192,
                image_height=4096,
            )
        )
    estimates, counts, unp = discover_objects(obs, camera_height_m=CAM_H)
    assert len(estimates) == 1, counts
    assert estimates[0].size_m is not None and estimates[0].size_m > 5.0
    (lon, lat) = unp(estimates[0].x, estimates[0].y)
    tlon, tlat = unp(24.0, 5.0)
    assert haversine_m(lon, lat, tlon, tlat) < 4.0  # somewhere along the body


def test_merge_by_rays_attaches_far_fragment_to_triangulated_object():

    target = (20.0, 6.0)
    # A solid triangulation from cameras near the object:
    near_rays = [
        (x, 0.0, math.degrees(math.atan2(target[0] - x, target[1])) % 360)
        for x in (12.0, 15.0, 18.0, 21.0, 24.0)
    ]
    solid = ObjectEstimate(
        20.0,
        6.0,
        np.eye(2) * 0.05,
        [],
        'triangulated',
        rms_m=0.1,
        parallax_deg=60.0,
        rays=near_rays,
        size_m=4.0,
    )
    # Frames far down the road saw the same car but could not triangulate it;
    # their ground-contact ranges put it 9 m short of where it is:
    far_rays = [
        (x, 0.0, math.degrees(math.atan2(target[0] - x, target[1])) % 360)
        for x in (-45.0, -42.0, -39.0)
    ]
    frag_obs = [
        Observation(f'far{i}', LABEL, [(0, 0), (0.1, 0), (0.1, 0.1)], LON0, LAT0, 90.0)
        for i in range(3)
    ]
    fragment = ObjectEstimate(
        12.0,
        4.0,
        np.diag([25.0, 1.0]),
        frag_obs,
        'single_view',
        rays=far_rays,
        size_m=2.0,
    )
    # An unrelated fragment pointing elsewhere stays separate:
    other = ObjectEstimate(
        12.0,
        -6.0,
        np.diag([25.0, 1.0]),
        [],
        'single_view',
        rays=[(-45.0, 0.0, 95.0), (-42.0, 0.0, 95.5)],
        size_m=2.0,
    )
    merged = merge_by_rays([solid, fragment, other])
    assert len(merged) == 2
    joined = next(e for e in merged if e.localization == 'triangulated')
    assert (joined.x, joined.y) == (20.0, 6.0)  # position of the solid estimate kept
    assert len(joined.members) == 3 and len(joined.rays) == 8


# ==========================================
# Duplicate outlines within a frame
# ==========================================


def _pano_sighting(
    image_id,
    az_deg,
    half_width_deg,
    bottom_deg,
    label=LABEL,
    lon=LON0,
    lat=LAT0,
    day=None,
):
    """A rectangle (normalised coordinates) in an equirectangular frame."""
    x0 = 0.5 + (az_deg - half_width_deg) / 360.0
    x1 = 0.5 + (az_deg + half_width_deg) / 360.0
    y1 = 0.5 + bottom_deg / 180.0  # ground contact below the horizon
    y0 = y1 - 1.2 * (x1 - x0)
    o = Observation(
        image_id,
        label,
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        lon,
        lat,
        0.0,
        is_pano=True,
        image_width=8192,
        image_height=4096,
        captured_at=day,
    )
    localize(o, camera_height_m=CAM_H, max_range_m=80)
    return o


def test_nested_duplicate_outline_is_dropped():
    big = _pano_sighting('f1', 30.0, 6.0, 8.0)
    small = _pano_sighting('f1', 31.0, 2.0, 7.5)
    other = _pano_sighting('f1', -60.0, 6.0, 8.0)
    kept, removed = suppress_duplicate_sightings([small, big, other])
    assert removed == 1
    assert len(kept) == 2
    assert all(o.area >= big.area or o is other for o in kept)


def test_separate_cars_in_one_frame_are_kept():
    a = _pano_sighting('f1', 20.0, 3.0, 8.0)
    b = _pano_sighting('f1', 40.0, 3.0, 8.0)  # a clear gap between them
    c = _pano_sighting('f1', 27.0, 3.0, 3.0)  # adjacent but twice as far
    d = _pano_sighting('f1', 26.5, 3.0, 8.0)  # touching a, same distance: next to it
    kept, removed = suppress_duplicate_sightings([a, b, c, d])
    assert removed == 0 and len(kept) == 4


def test_different_labels_are_never_merged():
    a = _pano_sighting('f1', 28.0, 3.0, 8.0)
    b = _pano_sighting('f1', 34.5, 3.0, 8.0, label='object--vehicle--truck')
    kept, removed = suppress_duplicate_sightings([a, b])
    assert removed == 0 and len(kept) == 2


def test_seam_halves_keep_the_larger_piece():
    w = 8192
    left = Observation(
        'f1',
        LABEL,
        [(0.0, 0.52), (0.025, 0.52), (0.025, 0.566), (0.0, 0.566)],
        LON0,
        LAT0,
        0.0,
        is_pano=True,
        image_width=w,
        image_height=w // 2,
    )
    right = Observation(
        'f1',
        LABEL,
        [(0.98, 0.519), (1.0, 0.519), (1.0, 0.562), (0.98, 0.562)],
        LON0,
        LAT0,
        0.0,
        is_pano=True,
        image_width=w,
        image_height=w // 2,
    )
    for o in (left, right):
        localize(o, camera_height_m=CAM_H, max_range_m=80)
    kept, removed = suppress_duplicate_sightings([right, left])
    assert removed == 1 and kept == [left]


def test_car_behind_another_is_not_joined():
    near = _pano_sighting('f1', 30.0, 5.0, 10.0)
    x0, y0, x1, y1 = near.bbox
    # A smaller, farther car whose box bottom lies a third of the way down
    # the near car's box (same columns): two objects in a row.
    h = y1 - y0
    far = Observation(
        'f1',
        LABEL,
        [
            (x0 + 0.01, y0 - 0.02),
            (x1 - 0.02, y0 - 0.02),
            (x1 - 0.02, y0 + 0.35 * h),
            (x0 + 0.01, y0 + 0.35 * h),
        ],
        LON0,
        LAT0,
        0.0,
        is_pano=True,
        image_width=8192,
        image_height=4096,
    )
    localize(far, camera_height_m=CAM_H, max_range_m=80)
    kept, removed = suppress_duplicate_sightings([near, far])
    assert removed == 0 and len(kept) == 2


def test_far_speck_cannot_claim_a_near_car():
    """A track of a tiny far object must not swallow a near detection 40 deg away."""
    project, _ = local_projection(LON0, LAT0)
    frames = []
    for k in range(6):
        lat = LAT0 + k * 3.0 / 111_320.0  # driving north, 3 m per frame
        # A speck straight ahead (1.5 deg wide, at the horizon):
        speck = Observation(
            f'f{k}',
            LABEL,
            [(0.498, 0.498), (0.502, 0.498), (0.502, 0.503), (0.498, 0.503)],
            LON0,
            lat,
            0.0,
            is_pano=True,
            image_width=8192,
            image_height=4096,
        )
        localize(speck, camera_height_m=CAM_H, max_range_m=80)
        frames.append(speck)
    # Then a car appears 45 deg to the right in the next frame:
    lat = LAT0 + 6 * 3.0 / 111_320.0
    car = Observation(
        'f6',
        LABEL,
        [(0.615, 0.5), (0.635, 0.5), (0.635, 0.53), (0.615, 0.53)],
        LON0,
        lat,
        0.0,
        is_pano=True,
        image_width=8192,
        image_height=4096,
    )
    localize(car, camera_height_m=CAM_H, max_range_m=80)
    frames.append(car)
    tracks = track_sequence(frames, project)
    assert sorted(len(t) for t in tracks) == [1, 6]


# ==========================================
# Far single views and negative evidence
# ==========================================


def _far_row(range_m, n=3, step_m=1.5, seq='s1', day='2025-08-20T10:00:00'):
    """A 2 m wide car ``range_m`` north of a camera driving east."""
    obs = []
    bottom = math.degrees(math.atan(CAM_H / range_m))
    half_width = math.degrees(math.atan(1.0 / range_m))
    for k in range(n):
        lon = LON0 + k * step_m / 74_900.0
        az = math.degrees(math.atan2(-k * step_m, range_m))  # bearing to the fixed car
        o = _pano_sighting(f'{seq}-{k}', az, half_width, bottom, lon=lon, day=day)
        o.sequence_id = seq
        obs.append(o)
    return obs


def test_far_single_view_is_not_reported_but_near_one_is():
    far = _far_row(45.0)
    ests, counts, _ = discover_objects(far, track_gap_frames=4)
    assert ests == [] and counts['far_single_view'] == 1
    ests, counts, _ = discover_objects(
        far, track_gap_frames=4, max_single_view_range_m=0
    )
    assert len(ests) == 1 and ests[0].localization == 'single_view'
    near = _far_row(12.0, n=4, step_m=3.0)  # enough parallax to triangulate
    ests, counts, _ = discover_objects(near, track_gap_frames=4)
    assert len(ests) == 1 and counts['far_single_view'] == 0


def _witness(image_id, x_m, y_m, day='2025-08-20', is_pano=True, compass=0.0):
    return CameraFrame(
        image_id,
        LON0 + x_m / 74_900.0,
        LAT0 + y_m / 111_320.0,
        compass,
        is_pano=is_pano,
        captured_at=f'{day}T12:00:00',
    )


def test_prune_unwitnessed_drops_contradicted_estimates():
    project, _ = local_projection(LON0, LAT0)
    member = Observation(
        'own-1',
        LABEL,
        [(0.5, 0.5), (0.51, 0.5), (0.51, 0.52)],
        LON0,
        LAT0 + 60 / 111_320.0,
        0.0,
        captured_at='2025-08-20T10:00:00',
    )
    est = ObjectEstimate(0.0, 10.0, np.eye(2) * 4.0, [member], 'single_view')
    # Two same-day frames drove within 5 m of the spot and saw nothing there:
    frames = [
        _witness('w1', 0.0, 5.0),
        _witness('w2', 3.0, 5.0),
        _witness('own-1', 0.0, 60.0),
    ]
    kept, dropped = prune_unwitnessed([est], frames, {}, project)
    assert kept == [] and dropped == 1
    # A single negative frame is not enough:
    kept, dropped = prune_unwitnessed([est], frames[:1], {}, project)
    assert len(kept) == 1 and dropped == 0
    # Frames from another survey day say nothing about today's car:
    other_day = [
        _witness('w1', 0.0, 5.0, day='2025-09-01'),
        _witness('w2', 3.0, 5.0, day='2025-09-01'),
    ]
    kept, dropped = prune_unwitnessed([est], other_day, {}, project)
    assert len(kept) == 1
    # A perspective camera facing away from the spot cannot be a witness:
    away = [
        _witness('w1', 0.0, 5.0, is_pano=False, compass=180.0),
        _witness('w2', 3.0, 5.0, is_pano=False, compass=180.0),
    ]
    kept, dropped = prune_unwitnessed([est], away, {}, project)
    assert len(kept) == 1
    # ... but one facing it is:
    facing = [
        _witness('w1', 0.0, 5.0, is_pano=False, compass=0.0),
        _witness('w2', 3.0, 5.0, is_pano=False, compass=330.0),
    ]
    kept, dropped = prune_unwitnessed([est], facing, {}, project)
    assert kept == []


def test_prune_unwitnessed_keeps_estimates_that_nearby_frames_saw():
    project, _ = local_projection(LON0, LAT0)
    est = ObjectEstimate(0.0, 10.0, np.eye(2) * 0.1, [], 'triangulated')
    frames = [
        _witness('w1', 0.0, 5.0),
        _witness('w2', 3.0, 5.0),
        _witness('w3', 6.0, 5.0),
    ]
    # Each frame holds a detection pointing at the estimate (bearing ~ north):
    sightings = {
        'w1': [
            Observation(
                'w1', LABEL, [(0.5, 0.5), (0.52, 0.5), (0.52, 0.55)], LON0, LAT0, 0.0
            )
        ],
        'w2': [
            Observation(
                'w2', LABEL, [(0.5, 0.5), (0.52, 0.5), (0.52, 0.55)], LON0, LAT0, 0.0
            )
        ],
    }
    for lst in sightings.values():
        for o in lst:
            o.bearing = 350.0
    kept, dropped = prune_unwitnessed([est], frames, sightings, project)
    assert len(kept) == 1 and dropped == 0
    # A detection far off the predicted bearing does not count as seeing it:
    for lst in sightings.values():
        for o in lst:
            o.bearing = 120.0
    kept, dropped = prune_unwitnessed([est], frames, sightings, project)
    assert kept == [] and dropped == 1


def test_discover_objects_applies_negative_evidence_to_weak_estimates_only():
    project, _ = local_projection(LON0, LAT0)
    # A car triangulated from a close pass (8 m, wide parallax) is trusted even
    # when a later same-day pass drives by at 3 m and sees nothing:
    near = _far_row(8.0, n=4, step_m=3.0)
    frames = [CameraFrame.from_observation(o) for o in near]
    pass2 = [_witness(f'p-{k}', k * 1.5, 5.0, day='2025-08-20') for k in range(-3, 4)]
    ests, counts, _ = discover_objects(near, track_gap_frames=4, frames=frames + pass2)
    assert len(ests) == 1 and counts['unwitnessed'] == 0
    # A car triangulated only from 30 m (weak) is dropped when a same-day pass
    # 3 m from its spot saw nothing there:
    far = _far_row(30.0, n=5, step_m=4.0)
    frames = [CameraFrame.from_observation(o) for o in far]
    ests, counts, _ = discover_objects(far, track_gap_frames=4, frames=frames)
    assert len(ests) == 1 and ests[0].localization == 'triangulated'
    pass2 = [_witness(f'p-{k}', k * 3.0, 27.0, day='2025-08-20') for k in range(-1, 2)]
    ests, counts, _ = discover_objects(far, track_gap_frames=4, frames=frames + pass2)
    assert ests == [] and counts['unwitnessed'] == 1
    # ... but not by a pass on another day:
    pass3 = [_witness(f'q-{k}', k * 3.0, 27.0, day='2025-09-01') for k in range(-1, 2)]
    ests, counts, _ = discover_objects(far, track_gap_frames=4, frames=frames + pass3)
    assert len(ests) == 1


def test_witness_votes_are_tallied_per_survey_day():
    """A car seen from 4 m on day one is kept although it had left by day two."""
    project, _ = local_projection(LON0, LAT0)
    day1 = [
        Observation(
            'd1-a',
            LABEL,
            [(0.5, 0.5), (0.52, 0.5), (0.52, 0.55)],
            LON0,
            LAT0,
            0.0,
            captured_at='2025-08-20T10:00:00',
        ),
        Observation(
            'd1-b',
            LABEL,
            [(0.5, 0.5), (0.52, 0.5), (0.52, 0.55)],
            LON0 + 3 / 74_900.0,
            LAT0,
            0.0,
            captured_at='2025-08-20T10:00:01',
        ),
        # a stray far sighting from the second day merged into the object:
        Observation(
            'd2-far',
            LABEL,
            [(0.5, 0.5), (0.51, 0.5), (0.51, 0.51)],
            LON0,
            LAT0 - 50 / 111_320.0,
            0.0,
            captured_at='2025-09-10T10:00:00',
        ),
    ]
    est = ObjectEstimate(1.5, 4.0, np.eye(2) * 0.1, day1, 'triangulated')
    frames = [
        _witness('d1-a', 0.0, 0.0),
        _witness('d1-b', 3.0, 0.0),
        _witness('d2-far', 0.0, -50.0, day='2025-09-10'),
        # four frames on the second day pass right by and see nothing:
        *[_witness(f'd2-{k}', k * 2.0, 0.0, day='2025-09-10') for k in range(-1, 3)],
    ]
    kept, dropped = prune_unwitnessed([est], frames, {}, project)
    assert len(kept) == 1 and dropped == 0
    # Without the close day-one frames (object built only from far views),
    # the second day's misses do count:
    far_only = ObjectEstimate(1.5, 4.0, np.eye(2) * 4.0, day1[2:], 'single_view')
    kept, dropped = prune_unwitnessed([far_only], frames, {}, project)
    assert kept == [] and dropped == 1


# ==========================================
# Weak triangulations attach by their rays
# ==========================================


def _ray_estimate(x, y, rays, range_m, sigma=1.5, parallax=16.0, prefix='w'):
    members = []
    for k, _ in enumerate(rays):
        o = Observation(
            f'{prefix}-{k}',
            LABEL,
            [(0.5, 0.5), (0.51, 0.5), (0.51, 0.51)],
            LON0,
            LAT0,
            0.0,
        )
        o.range_m = range_m
        members.append(o)
    est = ObjectEstimate(x, y, np.eye(2) * sigma**2, members, 'triangulated')
    est.parallax_deg = parallax
    est.rays = list(rays)
    return est


def _strong(x, y):
    members = [
        Observation(
            f's-{k}', LABEL, [(0.5, 0.5), (0.52, 0.5), (0.52, 0.55)], LON0, LAT0, 0.0
        )
        for k in range(3)
    ]
    for o in members:
        o.range_m = 8.0
    est = ObjectEstimate(x, y, np.eye(2) * 0.05**2, members, 'triangulated')
    est.parallax_deg = 60.0
    est.rays = [(x - 8.0, y - 2.0, 76.0), (x, y - 8.0, 0.0), (x + 8.0, y - 2.0, 284.0)]
    return est


def _rays_towards(target_x, target_y, cams):
    return [
        (cx, cy, math.degrees(math.atan2(target_x - cx, target_y - cy)) % 360)
        for cx, cy in cams
    ]


def test_far_triangulation_joins_the_close_object_its_rays_point_at():
    strong = _strong(0.0, 10.0)
    # A pass 40 m south triangulated the same car, but 6 m too far along the ray:
    cams = [(-4.0, -30.0), (0.0, -30.0), (4.0, -30.0)]
    weak = _ray_estimate(0.0, 16.0, _rays_towards(0.0, 10.5, cams), range_m=40.0)
    merged = merge_by_rays([weak, strong], bearing_sigma_deg=0.75)
    assert len(merged) == 1
    kept = merged[0]
    assert (kept.x, kept.y) == (0.0, 10.0)  # the close pass decides the position
    assert kept.n_images == 6 and kept.sigma_m < 0.1


def test_far_triangulation_too_far_along_its_ray_stays_separate():
    strong = _strong(0.0, 10.0)
    cams = [(-4.0, -30.0), (0.0, -30.0), (4.0, -30.0)]
    # Rays point at the car, but the estimate sits 25 m beyond it: another object.
    weak = _ray_estimate(0.0, 35.0, _rays_towards(0.0, 10.0, cams), range_m=40.0)
    assert len(merge_by_rays([weak, strong], bearing_sigma_deg=0.75)) == 2


def test_weak_estimate_prefers_the_nearest_of_two_cars_in_line():
    near_car = _strong(0.0, 10.0)
    far_car = _strong(0.0, 16.0)
    far_car.members = [
        Observation(
            f't-{k}', LABEL, [(0.5, 0.5), (0.52, 0.5), (0.52, 0.55)], LON0, LAT0, 0.0
        )
        for k in range(3)
    ]
    for o in far_car.members:
        o.range_m = 8.0
    cams = [(-4.0, -30.0), (0.0, -30.0), (4.0, -30.0)]
    weak = _ray_estimate(0.0, 14.0, _rays_towards(0.0, 12.0, cams), range_m=42.0)
    merged = merge_by_rays([weak, near_car, far_car], bearing_sigma_deg=0.75)
    assert len(merged) == 2
    joined = next(e for e in merged if e.n_images == 6)
    assert (joined.x, joined.y) == (0.0, 16.0)  # nearest to the weak position


def test_solid_triangulations_still_need_mutual_rays():
    a = _strong(0.0, 10.0)
    b = _strong(0.0, 14.0)  # a second car right behind: not weak, no mutual passage
    b.members = [
        Observation(
            f'u-{k}', LABEL, [(0.5, 0.5), (0.52, 0.5), (0.52, 0.55)], LON0, LAT0, 0.0
        )
        for k in range(3)
    ]
    for o in b.members:
        o.range_m = 8.0
    assert len(merge_by_rays([a, b], bearing_sigma_deg=0.75)) == 2


# ==========================================
# Crowded driveways across passes
# ==========================================


def _pass(cars, seq, day, y_cam=0.0, hidden=(), n=10, step=3.0, x0=-13.5):
    """One drive past ``cars`` (x, y) along y=``y_cam``; ``hidden`` cars are
    not detected in this pass (parked behind another car from this side)."""
    obs = []
    for k in range(n):
        cx = x0 + k * step
        for j, (x, y) in enumerate(cars):
            if j in hidden:
                continue
            dx, dy = x - cx, y - y_cam
            r = math.hypot(dx, dy)
            az = math.degrees(math.atan2(dx, dy)) % 360
            o = _pano_sighting(
                f'{seq}-{k}',
                az,
                math.degrees(math.atan(0.9 / r)),
                math.degrees(math.atan(CAM_H / r)),
                lon=LON0 + cx / 74_900.0,
                lat=LAT0 + y_cam / 111_320.0,
                day=f'{day}T10:00:00',
            )
            o.sequence_id = seq
            obs.append(o)
    return obs


def test_side_by_side_cars_survive_two_passes_with_one_hidden_each_time():
    # Two cars 2.8 m apart in a driveway 10 m back. The first pass sees only
    # the left one, the second (another day, other direction) only the right one,
    # and a third pass sees both. They must stay two objects.
    cars = [(-1.4, 10.0), (1.4, 10.0)]
    obs = (
        _pass(cars, 'p1', '2025-08-20', hidden=(1,))
        + _pass(cars, 'p2', '2025-08-21', hidden=(0,))
        + _pass(cars, 'p3', '2025-08-22')
    )
    ests, counts, unproject = discover_objects(obs, track_gap_frames=4)
    xs = sorted(round((unproject(e.x, e.y)[0] - LON0) * 74_900, 1) for e in ests)
    assert len(ests) == 2, xs
    assert abs(xs[0] + 1.4) < 0.7 and abs(xs[1] - 1.4) < 0.7


def test_one_car_seen_from_two_passes_is_one_object():
    cars = [(0.0, 10.0)]
    obs = _pass(cars, 'p1', '2025-08-20') + _pass(cars, 'p2', '2025-08-21', y_cam=-4.0)
    ests, counts, _ = discover_objects(obs, track_gap_frames=4)
    assert len(ests) == 1 and ests[0].n_images == 20


# ---------------------------------------------------------- piece merging
def _sightings_of(
    point, frames, project, unproject, offset=(0.0, 0.0), range_bias=1.0, prefix='f'
):
    """Sightings of ``point`` from cameras at ``frames`` (x along the road, y=0)."""
    out = []
    for i, cam_x in enumerate(frames):
        lon, lat = unproject(cam_x, 0.0)
        tx, ty = point[0] + offset[0], point[1] + offset[1]
        bearing = math.degrees(math.atan2(tx - cam_x, ty)) % 360
        o = Observation(
            f'{prefix}{i:02d}',
            LABEL,
            [(0.5, 0.5), (0.52, 0.5), (0.52, 0.55), (0.5, 0.55)],
            lon,
            lat,
            0.0,
            captured_at=f'2025-08-20T10:00:{i:02d}',
            sequence_id='s1',
        )
        o.bearing = bearing
        o.range_m = math.hypot(tx - cam_x, ty) * range_bias
        out.append(o)
    return out


def _estimate(point, members, sigma, parallax, size=4.5):
    return ObjectEstimate(
        point[0],
        point[1],
        np.eye(2) * sigma**2,
        members,
        'triangulated',
        parallax_deg=parallax,
        size_m=size,
    )


def test_sighting_shows_tolerates_a_piece_but_not_the_car_alongside():
    project, unproject = local_projection(LON0, LAT0)
    car = (10.0, 6.0)  # seen abeam from the road at y = 0
    whole = _sightings_of(car, [10.0], project, unproject)[0]
    rear = _sightings_of(car, [10.0], project, unproject, offset=(1.0, 0.3))[0]
    beside = _sightings_of(car, [10.0], project, unproject, offset=(2.7, 0.0))[0]
    far = _sightings_of(car, [10.0], project, unproject, range_bias=1.9)[0]
    assert sighting_shows(whole, car, project)
    assert sighting_shows(rear, car, project)
    assert not sighting_shows(beside, car, project)
    assert not sighting_shows(far, car, project)  # range says twice as far


def test_merge_pieces_folds_a_split_track_into_the_car_it_shows():
    project, unproject = local_projection(LON0, LAT0)
    car = (10.0, 6.0)
    frames = [float(x) for x in range(0, 22, 2)]
    a = _estimate(car, _sightings_of(car, frames, project, unproject), 0.06, 150)
    # The rear half of the same car, tracked separately over the same frames
    # and triangulated 2.9 m along the line of sight from a far approach:
    piece = _sightings_of(
        car, frames[:6], project, unproject, offset=(0.9, 0.2), prefix='f'
    )
    b = _estimate((12.0, 8.1), piece, 0.4, 30)
    merged = merge_pieces([a, b], project)
    assert len(merged) == 1
    assert merged[0].x == car[0] and merged[0].y == car[1]  # A's position kept
    assert merged[0].n_images == len(frames)  # B's sightings joined A


def test_merge_pieces_keeps_the_car_alongside_and_the_one_behind():
    project, unproject = local_projection(LON0, LAT0)
    car = (10.0, 6.0)
    frames = [float(x) for x in range(0, 22, 2)]
    a = _estimate(car, _sightings_of(car, frames, project, unproject), 0.06, 150)
    beside = (12.7, 6.0)  # parked alongside at the kerb, in the same frames
    b = _estimate(
        beside, _sightings_of(beside, frames, project, unproject, prefix='f'), 0.4, 30
    )
    behind = (10.0, 11.5)  # nose to tail behind, along the line of sight
    c = _estimate(
        behind, _sightings_of(behind, frames, project, unproject, prefix='f'), 0.4, 30
    )
    assert len(merge_pieces([a, b], project)) == 2
    assert len(merge_pieces([a, c], project)) == 2


def test_merge_pieces_refuses_a_larger_object_and_a_distant_solid_one():
    project, unproject = local_projection(LON0, LAT0)
    car = (10.0, 6.0)
    frames = [float(x) for x in range(0, 22, 2)]
    a = _estimate(car, _sightings_of(car, frames, project, unproject), 0.06, 150)
    piece = _sightings_of(
        car, frames[:6], project, unproject, offset=(0.9, 0.2), prefix='f'
    )
    motorhome = _estimate((12.0, 8.1), piece, 0.4, 30, size=9.0)
    assert len(merge_pieces([a, motorhome], project)) == 2
    solid_far = _estimate(
        (12.0, 9.0), piece, 0.1, 120
    )  # trustworthy position 3.6 m away
    assert len(merge_pieces([a, solid_far], project)) == 2
    solid_near = _estimate((11.0, 7.0), piece, 0.1, 120)  # 1.4 m away: a split outline
    assert len(merge_pieces([a, solid_near], project)) == 1


def test_discover_objects_reports_pieces_and_can_skip_the_pass():
    obs, _, _ = _survey(CARS)
    on, counts_on, _ = discover_objects(obs, camera_height_m=CAM_H, max_range_m=35.0)
    off, counts_off, _ = discover_objects(
        obs, camera_height_m=CAM_H, max_range_m=35.0, merge_pieces_of_neighbours=False
    )
    assert counts_off['pieces'] == 0 and counts_on['pieces'] >= 0
    assert len(on) == len(off) == len(CARS)
