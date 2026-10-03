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
)
from rapidtools.processing.street_tracking import (
    ObjectEstimate,
    discover_objects,
    merge_estimates,
    single_view_covariance,
    track_sequence,
    triangulate_track,
    vote_rays,
)

LON0, LAT0 = -117.49, 47.71
CAM_H = 2.4
LABEL = 'object--vehicle--car'


def _pano_polygon(cam_xy, heading, obj_xy, rng, noise_deg=0.4, range_noise=0.15):
    """Normalised pano outline of a 1.5 m tall object seen from ``cam_xy``."""
    dx, dy = obj_xy[0] - cam_xy[0], obj_xy[1] - cam_xy[1]
    dist = math.hypot(dx, dy)
    bearing = math.degrees(math.atan2(dx, dy)) + rng.gauss(0.0, noise_deg)
    rel = ((bearing - heading + 180.0) % 360.0) - 180.0
    cx = 0.5 + rel / 360.0
    # The ground contact as the mask sees it: the range is biased/noisy.
    seen_dist = dist * (1.0 + rng.gauss(0.0, range_noise))
    y_bottom = 0.5 + math.degrees(math.atan2(CAM_H, seen_dist)) / 180.0
    y_top = 0.5 - math.degrees(math.atan2(1.5 - CAM_H, seen_dist)) / 180.0 * -1
    y_top = min(y_top, y_bottom - 0.01)
    half_w = math.degrees(math.atan2(1.0, dist)) / 360.0
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
                        cam_xy, heading, obj, rng, noise_deg, range_noise
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
    # c is as close as b but was seen in the same frame as a: different object.
    o1 = Observation('frame-1', LABEL, [(0, 0), (0.1, 0), (0.1, 0.1)], LON0, LAT0, 0.0)
    o2 = Observation('frame-1', LABEL, [(0, 0), (0.1, 0), (0.1, 0.1)], LON0, LAT0, 0.0)
    a.members.append(o1)
    c = ObjectEstimate(0.5, -0.3, np.eye(2) * 0.8, [o2], 'single_view')
    merged = merge_estimates([a, b, c], max_distance_m=6.0)
    assert len(merged) == 2
    fused = min(merged, key=lambda e: e.x)
    assert fused.localization == 'triangulated'
    assert 0.0 < fused.x < 0.5  # pulled only slightly towards the vaguer view


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
