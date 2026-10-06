"""Edge cases of the street-level modules: error branches and fallbacks."""

import math
import sys
import threading
import types

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Point

from rapidtools.core import (
    ImageAsset,
    Observation,
    PhysicalAsset,
    PhysicalAssetCollection,
)
from rapidtools.processing import DuplicateResolver
from rapidtools.processing.duplicates import _lon_lat, _range_key, parse_pair_verdict
from rapidtools.processing.reid import AppearanceEmbedder, _UnionFind, crop_observation
from rapidtools.processing.street_localization import (
    ground_range_from_ray,
    local_projection,
    polygon_anchor,
    simplify_polygon,
    world_ray,
)
from rapidtools.processing.street_tracking import (
    CameraFrame,
    ObjectEstimate,
    _solve_intersection,
    attach_approach_sightings,
    prune_unwitnessed,
    sighting_shows,
)
from rapidtools.processing.verification import _range_key as verify_range_key
from rapidtools.processing.verification import normalize_confidence

LON0, LAT0 = -117.49, 47.71
BOX = [(0.5, 0.5), (0.52, 0.5), (0.52, 0.55), (0.5, 0.55)]


def _crop(tmp_path, name, outline=None, size=(300, 200), range_m=5.0):
    path = tmp_path / f'{name}.jpg'
    Image.new('RGB', size, (90, 90, 160)).save(path)
    props = {'image_id': name, 'range_m': range_m}
    if outline is not None:
        props['outline'] = outline
    return ImageAsset(id=name, path=path, properties=props)


def _asset(tmp_path, uid, east_m=0.0, crops=2, **props):
    lon = LON0 + east_m / (111320 * 0.6736)
    asset = PhysicalAsset(
        id=uid,
        geometry=Point(lon, LAT0),
        attributes={'asset_type': 'vehicles', 'n_images': 3},
    )
    for k in range(crops):
        asset.add_image_assets(_crop(tmp_path, f'{uid}_{k}', **props))
    return asset


# ------------------------------------------------------- duplicate resolver
def test_resolver_helpers_handle_odd_values(tmp_path):
    empty = PhysicalAsset(id='e', geometry=Point(), attributes={})
    assert _lon_lat(empty) is None
    assert _range_key(_crop(tmp_path, 'x', range_m=True)) == math.inf
    assert _range_key(_crop(tmp_path, 'y', range_m='far')) == math.inf
    assert parse_pair_verdict('not json') == (None, None, '')
    assert parse_pair_verdict('42') == (None, None, '')
    assert parse_pair_verdict('{"same": true, "confidence": "high"}') == (
        True,
        None,
        '',
    )
    with pytest.raises(ValueError, match='max_images_per_asset'):
        DuplicateResolver(judge=lambda a, b: (True, 1.0), max_images_per_asset=0)


def test_resolver_tiles_zoom_to_the_outline_and_enlarge_small_crops(tmp_path):
    resolver = DuplicateResolver(judge=lambda a, b: (True, 1.0))
    outlined = _crop(
        tmp_path, 'big', outline=[[100, 60], [200, 60], [200, 140], [100, 140]]
    )
    tile = resolver._tile(outlined, (580, 400))
    assert tile is not None and tile.width < 300  # zoomed to the outline plus margin
    tiny = _crop(
        tmp_path, 'tiny', outline=[[5, 5], [60, 5], [60, 40], [5, 40]], size=(70, 50)
    )
    tile = resolver._tile(tiny, (580, 400))
    assert tile is not None and tile.height >= 200  # enlarged so the model can see it
    (tmp_path / 'broken.jpg').write_bytes(b'not an image')
    broken = ImageAsset(id='b', path=tmp_path / 'broken.jpg', properties={})
    assert resolver._tile(broken, (580, 400)) is None
    out = resolver.compose([outlined, broken], [tiny], tmp_path / 'pair.jpg')
    assert out.is_file()


def test_resolver_model_failures_and_cancellation(tmp_path, caplog):
    class Exploding:
        model_id = 'boom'

        def run_inference(self, image_inputs, prompt, **kwargs):
            raise RuntimeError('boom')

    a, b = _asset(tmp_path, 'a'), _asset(tmp_path, 'b', east_m=2.0)
    col = DuplicateResolver(Exploding(), save_directory=tmp_path / 'p')(
        PhysicalAssetCollection([a, b])
    )
    assert len(col) == 2  # a failed call is no verdict

    class Garbled:
        model_id = 'garbled'

        def run_inference(self, image_inputs, prompt, **kwargs):
            return types.SimpleNamespace(text='???')

    col = DuplicateResolver(Garbled(), save_directory=tmp_path / 'q')(
        PhysicalAssetCollection(
            [_asset(tmp_path, 'c'), _asset(tmp_path, 'd', east_m=2.0)]
        )
    )
    assert len(col) == 2
    stop = threading.Event()
    stop.set()
    col = DuplicateResolver(judge=lambda x, y: (True, 1.0), cancel_event=stop)(
        PhysicalAssetCollection(
            [_asset(tmp_path, 'e'), _asset(tmp_path, 'f', east_m=2.0)]
        )
    )
    assert len(col) == 2  # cancelled before any pair was judged
    lonely = DuplicateResolver(judge=lambda x, y: (True, 1.0))(
        PhysicalAssetCollection([_asset(tmp_path, 'g')])
    )
    assert len(lonely) == 1


def test_resolver_pairs_within_one_grid_cell(tmp_path):
    assets = [_asset(tmp_path, f'v{k}', east_m=0.5 * k) for k in range(4)]
    pairs = DuplicateResolver(judge=lambda a, b: (False, 1.0)).candidate_pairs(
        PhysicalAssetCollection(assets)
    )
    assert len(pairs) == 6


# ---------------------------------------------------------------- verifier
def test_verifier_helpers_normalise_odd_values(tmp_path):
    assert normalize_confidence('n/a') is None
    assert normalize_confidence(float('nan')) is None
    assert normalize_confidence(85) == 0.85
    assert verify_range_key(_crop(tmp_path, 'x', range_m='far')) == (1, 0.0)
    assert verify_range_key(_crop(tmp_path, 'y', range_m=float('nan'))) == (1, 0.0)


# ------------------------------------------------- re-identification loader
class _FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=float)

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def mean(self, dim):
        return _FakeTensor(self.array.mean(axis=dim))


class _FakeInputs(dict):
    def to(self, device):
        return self


def _fake_stack(monkeypatch, model_type, output):
    """Install fake torch and transformers modules exposing what _load uses."""
    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    class _Mode:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    torch.inference_mode = _Mode
    calls = []

    class _Model:
        def to(self, device):
            calls.append(('to', device))
            return self

        def eval(self):
            return self

        def __call__(self, **inputs):
            return output(len(inputs['pixel_values']))

    class _Loader:
        @staticmethod
        def from_pretrained(model_id):
            calls.append(('from_pretrained', model_id))
            return _Model()

    transformers = types.ModuleType('transformers')
    transformers.AutoConfig = types.SimpleNamespace(
        from_pretrained=lambda m: types.SimpleNamespace(model_type=model_type)
    )
    transformers.AutoModel = _Loader
    transformers.CLIPVisionModelWithProjection = _Loader
    transformers.AutoImageProcessor = types.SimpleNamespace(
        from_pretrained=lambda m: (
            lambda images, return_tensors: _FakeInputs(pixel_values=list(images))
        )
    )
    levels = []
    transformers.logging = types.SimpleNamespace(
        get_verbosity=lambda: 30,
        set_verbosity_error=lambda: levels.append('error'),
        set_verbosity=levels.append,
    )
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    monkeypatch.setattr('rapidtools.auth.ensure_huggingface_login', lambda: None)
    return calls, levels


def test_embedder_loads_a_clip_vision_tower_and_pools_image_embeds(monkeypatch):
    calls, levels = _fake_stack(
        monkeypatch,
        'clip',
        lambda n: types.SimpleNamespace(image_embeds=_FakeTensor(np.ones((n, 4)))),
    )
    embedder = AppearanceEmbedder('openai/clip-vit-base-patch16', batch_size=2)
    vectors = embedder.embed([Image.new('RGB', (8, 8)) for _ in range(3)])
    assert vectors.shape == (3, 4) and embedder.is_loaded
    assert ('from_pretrained', 'openai/clip-vit-base-patch16') in calls and (
        'to',
        'cpu',
    ) in calls
    assert levels == ['error', 30]  # the loading report was silenced and restored


def test_embedder_loads_other_backbones_and_pools_the_mean(monkeypatch):
    _fake_stack(
        monkeypatch,
        'dinov2',
        lambda n: types.SimpleNamespace(
            image_embeds=None,
            pooler_output=None,
            last_hidden_state=_FakeTensor(np.ones((n, 5, 3))),
        ),
    )
    embedder = AppearanceEmbedder('facebook/dinov2-small', device='cpu')
    vectors = embedder.embed([Image.new('L', (8, 8))])
    assert vectors.shape == (1, 3)


def test_crop_observation_converts_palette_panoramas_at_the_seam():
    pano = Image.new('P', (200, 100))
    crop = crop_observation(
        pano, [(0.95, 0.5), (0.05, 0.5), (0.05, 0.6), (0.95, 0.6)], buffer=0.0
    )
    assert crop.size == (20, 10) and crop.mode == 'RGB'


def test_union_find_compresses_paths():
    union = _UnionFind(['a', 'b', 'c', 'd'])
    union.union('a', 'b')
    union.union('b', 'c')
    union.union('c', 'd')
    assert union.find('d') == union.find('a')
    assert union.groups(['a', 'b', 'c', 'd']) == [['a', 'b', 'c', 'd']]


# ------------------------------------------------------------ localisation
def test_localisation_helpers_edge_cases():
    ray = world_ray([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    assert ray[2] == pytest.approx(1.0)
    assert ground_range_from_ray([0.0, 1.0, 0.1], 2.4) is None
    cx, cy, bottom = polygon_anchor(
        [(0.1, 0.4), (0.3, 0.4), (0.3, 0.6), (0.1, 0.6)], False
    )
    assert cx == pytest.approx(0.2) and bottom == pytest.approx(0.6)
    degenerate = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    assert simplify_polygon(degenerate, 0.01) == degenerate
    line = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    assert simplify_polygon(line, 0.01) == line


# ----------------------------------------------------------------- tracking
def test_tracking_helpers_edge_cases():
    project, unproject = local_projection(LON0, LAT0)
    # Parallel rays have no intersection:
    cams = np.array([[0.0, 0.0], [3.0, 0.0]])
    normals = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert _solve_intersection(cams, normals, np.ones(2)) is None
    # A sighting without a camera position shows nothing and attaches nowhere:
    blind = Observation('b', 'car', BOX, None, None, 0.0, bearing=90.0)
    assert sighting_shows(blind, (10.0, 0.0), project) is False
    car = ObjectEstimate(10.0, 6.0, np.eye(2) * 0.01, [], 'triangulated')
    assert attach_approach_sightings([], [blind], project) == 0
    assert attach_approach_sightings([car], [blind], project) == 0
    # A ranged sighting pointing at nothing placed is left out; one without a
    # range that points at the car is attached by distance.
    lon, lat = unproject(10.0, -5.0)
    toward = Observation('t', 'car', BOX, lon, lat, 0.0, bearing=0.0)
    away = Observation('w', 'car', BOX, lon, lat, 0.0, bearing=180.0)
    away.range_m = 11.0
    assert attach_approach_sightings([car], [toward, away], project) == 1
    assert [o.image_id for o in car.members] == ['t']
    # No frames: nothing can be pruned.
    kept, dropped = prune_unwitnessed([car], [], {}, project)
    assert kept == [car] and dropped == 0
    frame = CameraFrame.from_observation(
        Observation('f', 'car', BOX, LON0, LAT0, 0.0, is_pano=False, bearing=0.0)
    )
    assert frame.horizontal_fov_deg < 360.0


# ------------------------------------------------- tracking: more branches
def _ray_obs(project_pair, image_id, cam_xy, bearing, range_m=None, label='car'):
    """A sighting from a camera at ``cam_xy`` looking along ``bearing``."""
    project, unproject = project_pair
    lon, lat = unproject(*cam_xy)
    obs = Observation(image_id, label, BOX, lon, lat, 0.0, bearing=bearing)
    if range_m is not None:
        obs.range_m = range_m
        gx = cam_xy[0] + range_m * math.sin(math.radians(bearing))
        gy = cam_xy[1] + range_m * math.cos(math.radians(bearing))
        obs.lon, obs.lat = unproject(gx, gy)
    return obs


def _aimed(project_pair, image_id, cam_xy, target_xy, range_m=None):
    bearing = math.degrees(
        math.atan2(target_xy[0] - cam_xy[0], target_xy[1] - cam_xy[1])
    )
    dist = math.hypot(target_xy[0] - cam_xy[0], target_xy[1] - cam_xy[1])
    return _ray_obs(project_pair, image_id, cam_xy, bearing % 360.0, range_m or dist)


def test_tracking_rejects_impossible_geometry():
    from rapidtools.processing.street_tracking import (
        _lon_lat,
        _points_at_a_neighbour,
        _weighted_intersection,
        classify_track,
        merge_by_rays,
        merge_estimates,
        merge_pieces,
        track_sequence,
        travel_headings,
    )

    pair = local_projection(LON0, LAT0)
    project = pair[0]
    assert _weighted_intersection([(0.0, 0.0, 90.0), (0.0, 5.0, 90.0)], 0.01) is None
    with pytest.raises(ValueError, match='no ground position'):
        _lon_lat(Observation('n', 'car', BOX, LON0, LAT0, 0.0, bearing=90.0))
    # Sightings without a bearing cannot be tracked; a label change breaks a track.
    no_bearing = Observation('nb', 'car', BOX, LON0, LAT0, 0.0)
    car = _ray_obs(pair, 'f0', (0.0, 0.0), 90.0, 10.0)
    truck = _ray_obs(pair, 'f1', (3.0, 0.0), 90.0, 10.0, label='truck')
    tracks = track_sequence([no_bearing, car, truck], project)
    assert sorted(len(t) for t in tracks) == [1, 1]
    # Parallel rays: no parallax, so the track is placed from its ranges.
    parallel = [_ray_obs(pair, f'p{k}', (3.0 * k, 0.0), 90.0, 10.0) for k in range(3)]
    estimate, verdict = classify_track(parallel, project)
    assert verdict == 'ok' and estimate.localization == 'single_view'
    # Four rays that never agree in pairs: the object moved.
    wandering = [
        _ray_obs(pair, f'w{k}', (5.0 * k, 0.0), bearing, 10.0)
        for k, bearing in enumerate((30.0, 300.0, 120.0, 200.0))
    ]
    assert classify_track(wandering, project) == (None, 'moving')
    # Rays that meet far beyond max_range_m give no position worth reporting.
    distant = [
        _aimed(pair, 'd0', (0.0, 0.0), (50.0, 600.0)),
        _aimed(pair, 'd1', (100.0, 0.0), (50.0, 600.0)),
    ]
    estimate, verdict = classify_track(distant, project, max_range_m=60.0)
    assert verdict == 'ok' and estimate.localization == 'single_view'
    # Rays that meet where the ground contact says the object cannot be.
    phantom = [
        _aimed(pair, f'g{k}', (5.0 * k, 0.0), (5.0, 20.0), range_m=5.0)
        for k in range(3)
    ]
    assert classify_track(phantom, project) == (None, 'moving')

    solid = ObjectEstimate(0.0, 0.0, np.eye(2) * 0.01, [car], 'triangulated')
    far_fragment = ObjectEstimate(60.0, 0.0, np.eye(2) * 4.0, [truck], 'single_view')
    assert len(merge_by_rays([solid, far_fragment])) == 2
    # Closest rays that carry no range, or no camera, point at nothing.
    bare = ObjectEstimate(10.0, 0.0, np.eye(2), [car], 'triangulated')
    bare.members = [Observation('x', 'car', BOX, LON0, LAT0, 0.0, bearing=90.0)]
    assert _points_at_a_neighbour(bare, [(12.0, 0.0)], project, 0.01, 4.5) is False
    blind = Observation('y', 'car', BOX, None, None, 0.0, bearing=90.0)
    blind.range_m = 5.0
    bare.members = [blind]
    assert _points_at_a_neighbour(bare, [(12.0, 0.0)], project, 0.01, 4.5) is False
    away = _ray_obs(pair, 'z', (0.0, 0.0), 270.0, 5.0)
    bare.members = [away]
    assert _points_at_a_neighbour(bare, [(12.0, 0.0)], project, 0.01, 4.5) is False
    # A sighting with no ground range is judged by its ray alone.
    assert sighting_shows(_ray_obs(pair, 's', (0.0, 0.0), 90.0), (10.0, 0.2), project)
    # Pieces: objects whose sightings carry no bearing or no range are skipped.
    no_rays = ObjectEstimate(0.0, 0.0, np.eye(2), [no_bearing], 'single_view')
    no_ranges = ObjectEstimate(
        3.0, 0.0, np.eye(2), [_ray_obs(pair, 'r', (0.0, 0.0), 90.0)], 'single_view'
    )
    assert len(merge_pieces([no_rays, no_ranges], project)) == 2
    assert travel_headings([blind], project) == {}
    # Estimates: a singular gate is skipped; a merged piece passes on its
    # size and its triangulation.
    zero = np.zeros((2, 2))
    a = ObjectEstimate(0.0, 0.0, zero, [car], 'single_view')
    b = ObjectEstimate(2.0, 0.0, zero, [truck], 'single_view')
    assert len(merge_estimates([a, b], floor_m=0.0)) == 2
    loose = ObjectEstimate(
        0.0, 0.0, np.eye(2) * 4.0, [car], 'triangulated', rms_m=0.3, size_m=3.0
    )
    tight = ObjectEstimate(
        0.5, 0.0, np.eye(2) * 0.1, [truck], 'single_view', size_m=4.0
    )
    [merged] = merge_estimates([tight, loose])
    assert merged.localization == 'triangulated' and merged.size_m == 4.0
    assert merged.rms_m == 0.3


def test_frames_and_witnesses():
    project, unproject = local_projection(LON0, LAT0)
    pano = CameraFrame.from_observation(
        Observation('p', 'car', BOX, LON0, LAT0, 0.0, is_pano=True, bearing=0.0)
    )
    assert pano.horizontal_fov_deg == 360.0
    # A frame right on top of the object cannot witness it.
    lon, lat = unproject(0.5, 0.0)
    close = CameraFrame.from_observation(
        Observation('c', 'car', BOX, lon, lat, 0.0, is_pano=True, bearing=0.0)
    )
    est = ObjectEstimate(0.0, 0.0, np.eye(2) * 9.0, [], 'single_view', parallax_deg=0)
    kept, dropped = prune_unwitnessed([est], [close], {}, project)
    assert len(kept) + dropped == 1


# --------------------------------------------------- localisation: more
def test_localisation_helpers_more_edge_cases():
    from rapidtools.processing.street_localization import (
        _ground_position,
        observation_angles,
        pixel_bearing,
    )

    centre = pixel_bearing(0.5, 0.5, 'fisheye', 4000, 3000, [0.9, 0.01, 0.0])
    assert centre[2] == pytest.approx(1.0)
    obs = Observation(
        'o',
        'car',
        BOX,
        LON0,
        LAT0,
        90.0,
        is_pano=False,
        extra={'camera_parameters': [0.9]},
    )
    bearing, elevation = observation_angles(obs)
    assert 85.0 < bearing < 95.0 and elevation < 0
    with pytest.raises(ValueError, match='no ground position'):
        _ground_position(obs)
    bow_tie = [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (0.5, 0.5)]
    assert simplify_polygon(bow_tie, 0.01) == bow_tie
    nan_ring = [(float('nan'), 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (0.5, 0.5)]
    assert simplify_polygon(nan_ring, 0.01)[1:] == nan_ring[1:]
    assert ground_range_from_ray([0.0, 1.0, 0.0], 2.4) is None
    assert ground_range_from_ray([0.0, 1.0, 0.5], 2.4) is None


def test_union_find_path_compression_direct():
    union = _UnionFind(['a', 'b', 'c'])
    union._parent.update({'b': 'a', 'c': 'b'})
    assert union.find('c') == 'a' and union._parent['c'] == 'a'
