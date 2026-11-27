"""Tests for buildings linked from per-building outlines (``street_instances``)."""

import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from scipy import sparse
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
from test_street_structures import (
    BUILDING,
    HOUSE_HEIGHT,
    HOUSES,
    PROJECT,
    UNPROJECT,
    _Client,
    _corners,
    _survey,
)

from rapidtools.models.base import ModelOutput
from rapidtools.processing import (
    MapillaryFeatureExtractor,
    street_instances,
    street_objects,
)
from rapidtools.processing.street_instances import (
    discover_instances,
    link_outlines,
    mask_outlines,
    outline_memberships,
    survey_points,
    track_points,
)
from rapidtools.processing.street_structures import PointCloud, project_points

# A garage wing and a house wing that share a wall, the house front set back:
WINGS = [(-14.0, -8.0, 10.0, 20.0), (-8.0, -1.0, 14.0, 20.0)]


def _instances(survey):
    """The survey with its outlines numbered per frame, as a segmenter gives them."""
    cloud, sightings, clusters = survey
    seen: dict[str, int] = {}
    numbered = []
    for obs in sightings:
        k = seen.get(obs.image_id, 0)
        seen[obs.image_id] = k + 1
        numbered.append(replace(obs, source='sam3', extra={**obs.extra, 'instance': k}))
    return cloud, numbered, clusters


def _discover(survey, **kwargs):
    cloud, sightings, clusters = survey
    return discover_instances(
        sightings,
        clusters,
        project=PROJECT,
        unproject=UNPROJECT,
        loader=lambda cid, url: cloud,
        **kwargs,
    )


@pytest.fixture(scope='module')
def houses():
    return _instances(_survey())


# ==========================================
# Masks to outlines
# ==========================================
def test_mask_outlines_follow_the_mask():
    mask = np.zeros((50, 100), bool)
    mask[10:30, 20:60] = True
    (ring,) = mask_outlines(mask)
    assert ring.min(axis=0) == pytest.approx([0.2, 0.2], abs=0.01)
    assert ring.max(axis=0) == pytest.approx([0.6, 0.6], abs=0.01)
    assert Polygon(ring).area == pytest.approx(0.4 * 0.4, rel=0.05)
    # Two parts come largest first; a speck and a hole are not outlines.
    mask[40:48, 70:95] = True
    mask[15:25, 30:40] = False
    mask[2, 2] = True
    rings = mask_outlines(mask)
    assert len(rings) == 2
    assert Polygon(rings[0]).area > Polygon(rings[1]).area
    assert Polygon(rings[0]).contains(Point(0.35, 0.4))  # the hole is filled
    # A mask that touches the image edge is closed along it:
    edge = np.zeros((20, 40), bool)
    edge[:, :10] = True
    (ring,) = mask_outlines(edge)
    assert ring.min() == 0.0 and Polygon(ring).area == pytest.approx(0.25, rel=0.05)
    assert mask_outlines(np.zeros((20, 40), bool)) == []
    assert mask_outlines(np.ones((2, 20, 40), bool)) == []  # one mask at a time
    assert street_instances._polygons(Point(0, 0)) == []


# ==========================================
# Linking outlines and assigning points
# ==========================================
def test_link_outlines_never_puts_two_outlines_of_a_frame_in_one_track():
    # Houses A (points 0-29) and B (30-59). Frames 0 and 1 outline each alone;
    # from far up the street, frame 2 outlines both as one.
    held = np.zeros((60, 5), np.float32)
    held[:30, [0, 2]] = 1  # A in frames 0 and 1
    held[30:, [1, 3]] = 1  # B in frames 0 and 1
    held[:, 4] = 1  # both, frame 2
    frame = np.array([0, 0, 1, 1, 2])
    track = link_outlines(sparse.csr_matrix(held), frame, set())
    assert track[0] == track[2] and track[1] == track[3] and track[0] != track[1]
    owner, assigned = track_points(
        sparse.csr_matrix(held), np.full(60, 3), frame, track, min_votes=2, min_frames=2
    )
    assert assigned.all()
    assert set(owner[:30]) == {track[0]} and set(owner[30:]) == {track[1]}
    # Two outlines of one frame that overlap are one object seen twice (the
    # house, and its garage alone): they may share a track, and tracks that
    # overlap in several frames are joined.
    twice = np.zeros((40, 6), np.float32)
    twice[:, [0, 2, 4]] = 1  # the house in three frames
    twice[:20, [1, 3, 5]] = 1  # its garage, outlined again in each
    frame = np.array([0, 0, 1, 1, 2, 2])
    apart = link_outlines(sparse.csr_matrix(twice), frame, set())
    assert apart[0] != apart[1]
    joined = link_outlines(sparse.csr_matrix(twice), frame, {(0, 1), (2, 3), (4, 5)})
    assert len(set(joined.tolist())) == 1
    # Nothing to link:
    assert link_outlines(
        sparse.csr_matrix(held[:, :1]), np.array([0]), set()
    ).tolist() == [0]
    none = sparse.csr_matrix((60, 0), dtype=np.float32)
    owner, assigned = track_points(
        none, np.zeros(60), np.zeros(0, int), np.zeros(0, int)
    )
    assert not assigned.any()


def test_track_points_needs_most_frames_and_a_long_enough_track():
    held = np.zeros((10, 4), np.float32)
    held[:5, :] = 1  # five points outlined in all four frames
    held[5:, 0] = 1  # five points outlined once
    frame = np.arange(4)
    track = np.zeros(4, int)
    owner, assigned = track_points(
        sparse.csr_matrix(held), np.full(10, 4), frame, track
    )
    assert assigned.tolist() == [True] * 5 + [False] * 5
    # Seen by ten frames, outlined in four: most frames say it is not a building.
    owner, assigned = track_points(
        sparse.csr_matrix(held), np.full(10, 10), frame, track
    )
    assert not assigned.any()
    # A track of two frames is not trusted with points:
    owner, assigned = track_points(
        sparse.csr_matrix(held[:, :2]),
        np.full(10, 2),
        frame[:2],
        track[:2],
        min_votes=2,
    )
    assert not assigned.any()


def test_outline_memberships_hold_what_a_frame_sees(houses):
    cloud, sightings, clusters = houses
    templates = {o.image_id: o for o in sightings}
    points, shots = survey_points(
        templates, clusters, PROJECT, loader=lambda cid, url: cloud
    )
    assert points.shape[1] == 4 and len(shots) == 21
    assert points[:, 3].max() == pytest.approx(HOUSE_HEIGHT, abs=0.3)
    outlines = {}
    for obs in sightings:
        outlines.setdefault(obs.image_id, {}).setdefault(
            obs.extra['instance'], []
        ).append(np.asarray(obs.polygon))
    held, seen, images, shapes, twice = outline_memberships(points, shots, outlines)
    assert held.shape == (len(points), 42) and len(shapes) == 42
    # Seen from up the street the two test outlines overlap; in front they do not.
    assert twice and (20, 21) not in twice
    assert seen.max() <= 21 and held.sum() > 1000
    # A frame without a pose, or no points at all, holds nothing:
    held, _, images, _, _ = outline_memberships(points, {}, outlines)
    assert held.shape[1] == 0 and images == []
    held, _, images, _, _ = outline_memberships(points[:0], shots, outlines)
    assert held.shape == (0, 0)
    far = {k: (r, c + np.array([900.0, 0.0, 0.0])) for k, (r, c) in shots.items()}
    assert outline_memberships(points, far, outlines)[0].shape[1] == 0


# ==========================================
# End to end on a synthetic street
# ==========================================
def test_two_houses_are_found_from_their_outlines(houses):
    structures, counts, unproject = _discover(houses)
    assert counts['frames'] == 21 and counts['outlines'] == 42
    assert counts['tracks'] == 2 and counts['joined'] == 0
    assert counts['structures'] == len(structures) == 2
    structures.sort(key=lambda s: s.x)
    for structure, (x0, x1, y0, y1) in zip(structures, HOUSES, strict=True):
        truth = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        found = Polygon(structure.footprint)
        assert found.intersection(truth).area / found.union(truth).area > 0.65
        assert abs(structure.frontage_m - 12.0) < 1.0
        assert abs(structure.height_m - HOUSE_HEIGHT) < 0.5
        assert structure.days == ['2026-08-27'] and structure.n_points > 300
        assert 3 <= structure.n_images <= 12
        for obs in structure.members:
            # a sighting is the building's own outline in that frame
            assert obs.source == 'sam3' and obs.label == BUILDING
            assert 'instance' not in obs.extra and obs.has_location
            assert obs.range_m > 5.0 and len(obs.polygon) >= 4
        lon, lat = unproject(structure.x, structure.y)
        assert abs(structure.members[0].lon - lon) < 1e-9
    every, _, _ = _discover(houses, max_views=0)
    assert all(s.n_images > 12 for s in every)


def test_wings_outlined_apart_are_one_building():
    # The model outlines the garage and the house as two objects in every
    # frame; their outlines touch and their walls continue each other.
    structures, counts, _ = _discover(_instances(_survey(houses=WINGS)))
    assert counts['tracks'] == 2 and counts['joined'] == 1
    assert len(structures) == 1
    house = structures[0]
    assert 12.0 < house.frontage_m < 14.0 and len(house.footprint) == 6
    outline = Polygon(house.footprint)
    assert outline.contains(Point(-11.0, 12.0)) and outline.contains(Point(-4.0, 16.0))
    assert not outline.contains(Point(-4.0, 12.0))  # the yard before the entrance
    # One sighting per frame covers both wings:
    widths = [
        max(x for x, _ in o.polygon) - min(x for x, _ in o.polygon)
        for o in house.members
    ]
    assert min(widths) > 0.05
    box, _, _ = _discover(_instances(_survey(houses=WINGS)), footprint='rectangle')
    assert len(box) == 1 and len(box[0].footprint) == 4
    seen, _, _ = _discover(_instances(_survey(houses=WINGS)), footprint='observed')
    assert len(seen) == 1 and seen[0].depth_source == 'observed'


def test_neighbors_stay_apart_whatever_stands_between_them(houses, monkeypatch):
    # Two houses with a side yard: the frames in front show their outlines
    # apart, so they are two buildings even when asked to join across 6 m.
    structures, counts, _ = _discover(houses, min_gap_m=2.0)
    assert len(structures) == 2 and counts['joined'] == 0
    # From far up the street a model may outline both as one object. A frame
    # shows a building once, so that outline joins one track, not both.
    wide = [(-22.0, -10.0, 10.0, 20.0), (10.0, 22.0, 10.0, 20.0)]
    cloud, sightings, clusters = _instances(_survey(houses=wide))
    _, together, _ = _instances(_survey(houses=wide, together=True))
    far = {f'img{i}' for i in (0, 1, 2, 18, 19, 20)}
    mixed = [o for o in sightings if o.image_id not in far]
    mixed += [o for o in together if o.image_id in far]
    structures, _, _ = _discover((cloud, mixed, clusters))
    assert sorted(round(s.frontage_m) for s in structures) == [12, 12]
    # Were two outlines of a frame allowed in one track, the far frames would
    # tie the two houses together:
    collect = street_instances.outline_memberships

    def no_rule(*args, **kwargs):
        held, seen, images, shapes, _ = collect(*args, **kwargs)
        n = len(images)
        return (
            held,
            seen,
            images,
            shapes,
            {(a, b) for a in range(n) for b in range(a, n)},
        )

    monkeypatch.setattr(street_instances, 'outline_memberships', no_rule)
    structures, _, _ = _discover((cloud, mixed, clusters))
    assert len(structures) == 1


def test_thresholds_and_footprint_options(houses):
    none, counts, _ = _discover(houses, min_points=100000)
    assert none == [] and counts['too_few_points'] == 2
    most = max(s.n_points for s in _discover(houses)[0])
    none, counts, _ = _discover(houses, min_points=most + 1)
    assert none == [] and counts['too_few_points'] == 2
    none, counts, _ = _discover(houses, min_frontage_m=50.0)
    assert none == [] and counts['too_short'] == 2
    none, counts, _ = _discover(houses, min_height_m=50.0)
    assert none == [] and counts['structures'] == 0
    none, counts, _ = _discover(houses, max_range_m=11.0)  # walls seen, no sighting
    assert counts['structures'] == len(none)
    with pytest.raises(ValueError, match='footprint'):
        _discover(houses, footprint='circle')


def test_nothing_to_do_cases(houses):
    cloud, sightings, clusters = houses
    structures, counts, unproject = discover_instances([], {})
    assert structures == [] and counts['structures'] == 0 and callable(unproject)
    structures, counts, _ = discover_instances(sightings, {})  # no clouds at all
    assert structures == [] and counts['unusable'] == len(sightings)
    structures, counts, _ = discover_instances(
        sightings, clusters, loader=lambda cid, url: None
    )
    assert structures == [] and counts['clusters_missing'] == 1
    # A frame whose pose is not among the cloud's shots is counted, not fatal:
    stray = replace(
        sightings[0], image_id='stray', extra={'computed_rotation': [0.3, 0.2, 0.1]}
    )
    structures, counts, _ = discover_instances(
        [*sightings, stray],
        {**clusters, 'stray': ('c1', 'u')},
        project=PROJECT,
        unproject=UNPROJECT,
        loader=lambda cid, url: cloud,
    )
    assert len(structures) == 2 and counts['frames_unmatched'] == 1
    # A cloud none of whose shots match, and one far from every frame:
    lost = PointCloud(
        'c1',
        cloud.lon,
        cloud.lat,
        cloud.points,
        cloud.rotations + 1.0,
        cloud.translations,
    )
    assert _discover((lost, sightings, clusters))[0] == []
    away = PointCloud(
        'c1',
        cloud.lon,
        cloud.lat,
        cloud.points + 500.0,
        cloud.rotations,
        cloud.translations,
    )
    assert _discover((away, sightings, clusters))[0] == []


def test_outlines_without_a_number_are_objects_of_their_own():
    cloud, sightings, clusters = _survey()
    structures, counts, _ = _discover((cloud, sightings, clusters))
    assert len(structures) == 2 and counts['outlines'] == 42


def test_a_mask_in_parts_is_one_outline(houses):
    # A tree cuts the mask of the west house in two in every frame; both parts
    # carry the same number, and the building is still one.
    cloud, sightings, clusters = houses
    parts = []
    for obs in sightings:
        ring = np.asarray(obs.polygon)
        if obs.extra['instance'] != 0:
            parts.append(obs)
            continue
        middle = ring[:, 0].mean()
        for keep in (ring[:, 0] <= middle, ring[:, 0] >= middle):
            half = ring[keep]
            if len(half) < 3:
                half = np.vstack(
                    [half, [[middle, ring[:, 1].min()], [middle, ring[:, 1].max()]]]
                )
            hull = half[ConvexHull(half).vertices] if len(half) > 3 else half
            parts.append(replace(obs, polygon=[tuple(p) for p in hull]))
    structures, counts, _ = _discover((cloud, parts, clusters))
    assert len(structures) == 2 and counts['outlines'] == 42


def test_touching_asks_the_frames_in_front_of_both():
    unit = street_instances._Unit
    wall_a = np.c_[np.linspace(0, 6, 40), np.full(40, 10.0), np.full(40, 3.0)]
    wall_b = wall_a + [6.5, 0.0, 0.0]
    left = Polygon([(0.40, 0.4), (0.50, 0.4), (0.50, 0.6), (0.40, 0.6)])
    right = Polygon([(0.502, 0.4), (0.60, 0.4), (0.60, 0.6), (0.502, 0.6)])
    away = Polygon([(0.70, 0.4), (0.80, 0.4), (0.80, 0.6), (0.70, 0.6)])
    shapes = [left, right, away, left, away, left]
    cameras = np.array([[6.0, 0.0], [9.0, 0.0], [300.0, 0.0]])
    ask = street_instances._touching
    first = unit(wall_a, {0: [0], 1: [3], 2: [5]})
    # Touching in both frames in front; the frame 300 m away is not asked.
    assert ask(
        first,
        unit(wall_b, {0: [1], 1: [1], 2: [2]}),
        wall_a,
        wall_b,
        shapes,
        cameras,
        40.0,
    )
    # Apart in one of two frames is not most of them:
    assert not ask(
        first, unit(wall_b, {0: [1], 1: [2]}), wall_a, wall_b, shapes, cameras, 40.0
    )
    # No frame shows both, or none within range:
    assert not ask(first, unit(wall_b, {7: [1]}), wall_a, wall_b, shapes, cameras, 40.0)
    assert not ask(first, unit(wall_b, {2: [1]}), wall_a, wall_b, shapes, cameras, 40.0)


def test_a_model_asked_for_the_whole_object_decides_about_wings(monkeypatch):
    wings = _instances(_survey(houses=WINGS))
    cloud, sightings, clusters = wings
    by_frame = {}
    for obs in sightings:
        by_frame.setdefault(obs.image_id, []).append(np.asarray(obs.polygon))
    asked = []

    def whole_house(image_id, points):
        """The part under the points, and the whole house: both wings."""
        asked.append((image_id, points.shape))
        rings = by_frame[image_id]
        hit = [r for r in rings if Polygon(r).contains(Point(points[0]))] or rings[:1]
        return [hit, rings]

    structures, counts, _ = _discover(wings, whole_object=whole_house)
    assert len(structures) == 1 and counts['joined'] == 1
    assert asked and {shape for _, shape in asked} == {(3, 2)}
    assert len(asked) <= 2 * street_instances._FRAMES_PROMPTED

    def only_the_part(image_id, points):
        rings = by_frame[image_id]
        return [
            [r for r in rings if Polygon(r).contains(Point(points[0]))] or rings[:1]
        ]

    # The model finds nothing larger than each wing: two buildings, although
    # their outlines touch.
    structures, counts, _ = _discover(wings, whole_object=only_the_part)
    assert len(structures) == 2 and counts['joined'] == 0
    # A model with nothing to say leaves the decision to the outlines:
    structures, _, _ = _discover(wings, whole_object=lambda image_id, points: [])
    assert len(structures) == 1
    structures, _, _ = _discover(wings, whole_object=lambda image_id, points: [[]])
    assert len(structures) == 1


def test_points_inside_a_shape_are_inside_and_repeatable():
    pick = street_instances._points_inside
    box = Polygon([(0.2, 0.4), (0.4, 0.4), (0.4, 0.6), (0.2, 0.6)])
    first = pick(box)
    assert first.shape == (3, 2) and np.array_equal(first, pick(box))
    assert all(box.contains(Point(p)) for p in first)
    # An L: the points stay off the empty corner of its bounding box.
    ell = Polygon([(0, 0), (0.3, 0), (0.3, 0.1), (0.1, 0.1), (0.1, 0.3), (0, 0.3)])
    assert all(ell.contains(Point(p)) for p in pick(ell, 5))
    # A sliver too thin to shrink, or too thin to hit: still a point on it.
    sliver = Polygon([(0.5, 0.5), (0.9, 0.5001), (0.9, 0.5002)])
    found = pick(sliver)
    assert found.shape == (3, 2) and sliver.buffer(1e-6).contains(Point(found[0]))


def test_sightings_skip_frames_too_far_or_without_a_shape(houses):
    cloud, sightings, clusters = houses
    template = sightings[0]
    wall = np.c_[
        np.linspace(0, 6, 40), np.full(40, 10.0), np.full(40, 103.0), np.full(40, 3.0)
    ]
    box = Polygon([(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)])
    shots = {
        'near': (np.eye(3), np.array([3.0, 0.0, 102.4])),
        'far': (np.eye(3), np.array([3.0, -90.0, 102.4])),
        'thin': (np.eye(3), np.array([4.0, 0.0, 102.4])),
    }
    templates = dict.fromkeys(shots, template)
    found = street_instances._sightings(
        wall,
        {0: [0], 1: [0], 2: [1]},
        [box, Point(0.5, 0.5)],
        ['near', 'far', 'thin'],
        shots,
        templates,
        40.0,
        12,
    )
    assert [o.image_id for o in found] == ['near']
    assert (
        found[0].range_m == pytest.approx(10.2, abs=0.5) and len(found[0].polygon) == 4
    )


# ==========================================
# Through the extractor
# ==========================================
class _Segmenter:
    """Stands in for any segmentation model: one mask per house in every frame."""

    def __init__(self, cloud, shots, size=(256, 512)):
        self.cloud, self.shots, self.size = cloud, shots, size
        self.prompts = []

    def run_inference(self, image_inputs, prompt, threshold, mask_threshold, **kw):
        self.prompts.append(prompt)
        height, width = self.size
        stacks, scores = [], []
        for path in image_inputs:
            rotation, translation = self.shots[Path(path).stem]
            masks = []
            for house in HOUSES:
                u, v, _ = project_points(_corners(*house), rotation, translation)
                ring = np.c_[u, v]
                ring = ring[ConvexHull(ring).vertices]
                canvas = Image.new('1', (width, height))
                ImageDraw.Draw(canvas).polygon(
                    [(float(x) * width, float(y) * height) for x, y in ring], fill=1
                )
                masks.append(np.array(canvas, bool))
            stacks.append(np.array(masks))
            scores.append([0.9, 0.8])
        return ModelOutput(
            masks=stacks, bounding_boxes=None, raw_response={'scores': scores}
        )


def _frames(sightings):
    frames = {}
    for obs in sightings:
        frames.setdefault(
            obs.image_id,
            {
                'id': obs.image_id,
                'lon': obs.camera_lon,
                'lat': obs.camera_lat,
                'compass': obs.compass_angle,
                'is_pano': True,
                'sequence': obs.sequence_id,
                'captured_at': obs.captured_at,
                'width': 4096,
                'height': 2048,
                'camera_type': 'spherical',
                'camera_parameters': None,
                'rotation': obs.extra['computed_rotation'],
                'thumb_url': 'https://thumbs/x',
            },
        )
    return list(frames.values())


def test_extractor_links_a_segmenters_masks_into_buildings(tmp_path, monkeypatch):
    cloud, sightings, clusters = _survey()
    monkeypatch.setattr(
        street_instances, 'fetch_sfm_cluster', lambda cid, url, cache=None: cloud
    )
    from rapidtools.processing.street_localization import rotation_matrix

    shots = {
        f'img{i}': (rotation_matrix(r), t)
        for i, (r, t) in enumerate(
            zip(cloud.rotations, cloud.translations, strict=True)
        )
    }
    model = _Segmenter(cloud, shots)
    ext = MapillaryFeatureExtractor(
        classes=['buildings', 'cars'],
        client=_Client(),
        save_directory=tmp_path,
        ego_filter=False,
        detection_source={'buildings': 'sam3'},
        sam3_model=model,
        sam3_prompts={'buildings': 'house'},
        structure_views=4,
    )
    # Buildings go to the model, cars stay with Mapillary's detections:
    detections, segmented, features = ext._plan_classes()
    assert segmented == ['buildings'] and list(detections) == ['cars'] and not features
    assert ext._outlined_classes(segmented) == {'buildings'}
    assert ext._wants_point_clouds(detections, segmented)
    monkeypatch.setattr(
        ext,
        '_download_thumbnail',
        lambda frame, size=None: tmp_path / f'{frame["id"]}.jpg',
    )
    found = ext._sam3_observations(_frames(sightings), segmented, {'buildings'})
    outlines = found['buildings']
    assert set(model.prompts) == {'house'}
    assert len(outlines) == 42 and {o.extra['instance'] for o in outlines} == {0, 1}
    assert {round(o.confidence, 2) for o in outlines} == {0.9, 0.8}
    assert all(o.source == 'sam3' and len(o.polygon) >= 4 for o in outlines)
    ext._clusters = dict(clusters)
    assert ext._method_for(outlines) == 'instances'
    assets, counter = ext._assets_for_class('buildings', outlines, 0)
    assert counter == 2 and len(assets) == 2
    for asset in assets:
        assert asset.geometry.geom_type == 'Polygon'
        at = asset.attributes
        assert at['localization'] == 'point_cloud' and at['source'] == 'sam3'
        assert at['asset_type'] == 'buildings' and at['n_images'] == 4
        assert at['confidence'] >= 0.8 and at['frontage_m'] == pytest.approx(
            12.0, abs=1.0
        )
    ext.geometry, ext.min_observations = 'point', 1
    points, _ = ext._assets_for_class('buildings', outlines, 0)
    assert {a.geometry.geom_type for a in points} == {'Point'}
    ext.min_observations = 50
    assert ext._assets_for_class('buildings', outlines, 0)[0] == []
    # Without the per-class choice the same masks are kept as boxes for fusion:
    boxes = ext._sam3_observations(_frames(sightings)[:2], segmented)
    assert all(
        'instance' not in o.extra and len(o.polygon) == 4 for o in boxes['buildings']
    )
    # A model that answers nothing for an image is skipped:
    model.run_inference = lambda paths, **kw: ModelOutput(
        masks=[None for _ in paths], bounding_boxes=None
    )
    assert (
        ext._sam3_observations(_frames(sightings)[:2], segmented, {'buildings'}) == {}
    )


def test_extractor_instances_method_needs_outlines_and_point_clouds(tmp_path, caplog):
    cloud, sightings, clusters = _survey()
    ext = MapillaryFeatureExtractor(
        classes=['buildings'],
        client=_Client(),
        save_directory=tmp_path,
        ego_filter=False,
        localization_method='instances',
    )
    assert ext._outlined_classes(['buildings', 'debris pile']) == {
        'buildings',
        'debris pile',
    }
    assert ext._wants_point_clouds({}, ['buildings'])
    ext._clusters = dict(clusters)
    with caplog.at_level(logging.WARNING):
        assets, _ = ext._assets_for_class('buildings', list(sightings), 0)
    assert 'needs one outline per object' in caplog.text
    assert all(a.attributes['localization'] != 'point_cloud' for a in assets)
    # No point clouds: the same fallback.
    caplog.clear()
    ext._clusters = {}
    numbered = _instances((cloud, sightings, clusters))[1]
    with caplog.at_level(logging.WARNING):
        ext._assets_for_class('buildings', numbered, 0)
    assert 'placing it by occupancy fusion' in caplog.text
    # Other methods never ask for outlines:
    ext.localization_method = 'fusion'
    assert ext._outlined_classes(['buildings']) == set()
    assert not ext._wants_point_clouds({}, ['buildings'])


class _PointModel:
    """Stands in for a point-prompt model: a small part and the whole frame."""

    def __init__(self):
        self.asked = []

    def segment_at(self, image, points):
        self.asked.append((image, points))
        part = np.zeros((16, 32), bool)
        part[6:10, 4:8] = True
        whole = np.zeros((16, 32), bool)
        whole[2:14, 2:30] = True
        return [np.array([part, whole])]


def test_extractor_asks_a_point_model_about_wings(tmp_path):
    model = _PointModel()
    ext = MapillaryFeatureExtractor(
        classes=['buildings'],
        client=_Client(),
        save_directory=tmp_path,
        join_wings_by='prompt',
        sam3_point_model=model,
    )
    points = np.array([[0.25, 0.5], [0.3, 0.5], [0.35, 0.5]])
    # No thumbnail of the frame: nothing to ask.
    assert ext._whole_object('img7', points) == [] and not model.asked
    (tmp_path / 'images').mkdir()
    Image.new('RGB', (32, 16)).save(tmp_path / 'images' / 'img7_1024.jpg')
    candidates = ext._whole_object('img7', points)
    assert len(candidates) == 2 and all(len(rings) == 1 for rings in candidates)
    assert Polygon(candidates[1][0]).area > Polygon(candidates[0][0]).area
    image, asked = model.asked[0]
    assert image.endswith('img7_1024.jpg')
    assert asked == [
        [(8.0, 8.0), (pytest.approx(9.6), 8.0), (pytest.approx(11.2), 8.0)]
    ]
    with pytest.raises(ValueError, match='join_wings_by'):
        MapillaryFeatureExtractor(
            classes=['buildings'],
            client=_Client(),
            save_directory=tmp_path,
            join_wings_by='guess',
        )


def test_extractor_loads_the_point_model_on_first_use(tmp_path, monkeypatch):
    import rapidtools.models.sam3 as sam3_module

    built = []

    class Loaded(_PointModel):
        def __init__(self, model_id, device):
            super().__init__()
            built.append((model_id, device))

    monkeypatch.setattr(sam3_module, 'SAM3PointInference', Loaded)
    ext = MapillaryFeatureExtractor(
        classes=['buildings'],
        client=_Client(),
        save_directory=tmp_path,
        join_wings_by='prompt',
    )
    (tmp_path / 'images').mkdir()
    Image.new('RGB', (32, 16)).save(tmp_path / 'images' / 'img7_1024.jpg')
    points = np.array([[0.5, 0.5]])
    assert len(ext._whole_object('img7', points)) == 2
    assert len(ext._whole_object('img7', points)) == 2
    assert built == [('facebook/sam3', 'auto')]  # loaded once


def test_extractor_passes_the_wing_test_on(tmp_path, monkeypatch):
    cloud, sightings, clusters = _instances(_survey(houses=WINGS))
    monkeypatch.setattr(
        street_instances, 'fetch_sfm_cluster', lambda cid, url, cache=None: cloud
    )
    seen = []
    real = street_objects.discover_instances

    def spy(*args, **kwargs):
        seen.append(kwargs['whole_object'])
        return real(*args, **kwargs)

    monkeypatch.setattr(street_objects, 'discover_instances', spy)
    for how in ('touching', 'prompt'):
        ext = MapillaryFeatureExtractor(
            classes=['buildings'],
            client=_Client(),
            save_directory=tmp_path,
            ego_filter=False,
            join_wings_by=how,
            sam3_point_model=_PointModel(),
        )
        ext._clusters = dict(clusters)
        assets, _ = ext._assets_for_class('buildings', list(sightings), 0)
        assert len(assets) == 1  # no thumbnails here: the outlines decide either way
    assert seen[0] is None and callable(seen[1])


def test_detection_source_per_class(tmp_path, caplog):
    def plan(source, classes):
        ext = MapillaryFeatureExtractor(
            classes=classes,
            client=_Client(),
            save_directory=tmp_path,
            detection_source=source,
        )
        detections, segmented, features = ext._plan_classes()
        return list(detections), segmented, list(features)

    classes = ['buildings', 'cars', 'fire hydrants', 'debris pile']
    assert plan('auto', classes) == (
        ['buildings', 'cars'],
        ['debris pile'],
        ['fire hydrants'],
    )
    assert plan('sam3', classes) == ([], classes, [])
    assert plan({'buildings': 'sam3'}, classes) == (
        ['cars'],
        ['buildings', 'debris pile'],
        ['fire hydrants'],
    )
    assert plan({'fire hydrants': 'mapillary', 'debris pile': 'sam3'}, classes) == (
        ['buildings', 'cars', 'fire hydrants'],
        ['debris pile'],
        [],
    )
    with caplog.at_level(logging.WARNING):
        assert plan({'cars': 'map_features', 'debris pile': 'mapillary'}, classes) == (
            ['buildings'],
            [],
            ['fire hydrants'],
        )
    assert "Not served as Mapillary map features: 'cars'" in caplog.text
    assert "No Mapillary label for 'debris pile'" in caplog.text
    with pytest.raises(ValueError, match='detection_source'):
        plan({'buildings': 'yolo'}, classes)
