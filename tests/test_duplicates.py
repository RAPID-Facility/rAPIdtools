"""Tests for the DuplicateResolver pipeline step."""

import json
import threading
from pathlib import Path

import pytest
from PIL import Image
from shapely.geometry import Point

from rapidtools.core import ImageAsset, PhysicalAsset, PhysicalAssetCollection
from rapidtools.processing import DuplicateResolver, Stage
from rapidtools.processing.duplicates import PAIR_PROMPT, parse_pair_verdict

LON0, LAT0 = -117.49, 47.71


def _image(tmp_path: Path, name: str, range_m=5.0) -> ImageAsset:
    path = tmp_path / f'{name}.jpg'
    if not path.is_file():
        Image.new('RGB', (64, 40), (120, 30, 30)).save(path)
    return ImageAsset(
        id=name, path=path, properties={'image_id': name, 'range_m': range_m}
    )


def _asset(tmp_path, uid, east_m=0.0, north_m=0.0, n_images=3, crops=2):
    lon = LON0 + east_m / (111320 * 0.6736)
    lat = LAT0 + north_m / 111320
    asset = PhysicalAsset(
        id=uid,
        geometry=Point(lon, lat),
        attributes={
            'asset_type': 'vehicles',
            'n_images': n_images,
            'observations': [{'image_id': f'{uid}_{k}'} for k in range(n_images)],
        },
    )
    for k in range(crops):
        asset.add_image_assets(_image(tmp_path, f'{uid}_{k}', range_m=5.0 + k))
    return asset


def _judge_by_name(same_groups):
    """A judge that calls two assets the same when their ids share a group."""

    def judge(paths_a, paths_b):
        a = Path(paths_a[0]).stem.split('_')[0]
        b = Path(paths_b[0]).stem.split('_')[0]
        for group, confidence in same_groups:
            if a in group and b in group:
                return True, confidence
        return False, 0.95

    return judge


def test_stage_and_constructor_validation():
    assert DuplicateResolver(judge=lambda a, b: (True, 1.0)).stage == Stage.VERIFY
    with pytest.raises(ValueError, match='exactly one'):
        DuplicateResolver()
    with pytest.raises(TypeError):
        DuplicateResolver(judge='no')
    with pytest.raises(ValueError, match='min_confidence'):
        DuplicateResolver(judge=lambda a, b: (True, 1.0), min_confidence=2.0)


def test_candidate_pairs_respect_distance_and_need_crops(tmp_path):
    a = _asset(tmp_path, 'a')
    b = _asset(tmp_path, 'b', east_m=3.0)
    far = _asset(tmp_path, 'far', east_m=30.0)
    bare = _asset(tmp_path, 'bare', east_m=2.0, crops=0)
    resolver = DuplicateResolver(judge=lambda x, y: (False, 1.0), max_distance_m=8.0)
    pairs = resolver.candidate_pairs(PhysicalAssetCollection([a, b, far, bare]))
    assert [(x.id, y.id) for x, y, _ in pairs] == [('a', 'b')]
    assert pairs[0][2] == pytest.approx(3.0, abs=0.1)


def test_merges_the_pair_the_judge_calls_one_vehicle(tmp_path):
    cab = _asset(tmp_path, 'cab', n_images=12)
    box = _asset(tmp_path, 'box', east_m=1.7, n_images=44)
    beside = _asset(tmp_path, 'beside', east_m=2.7, north_m=1.0, n_images=30)
    judge = _judge_by_name([({'cab', 'box'}, 0.97)])
    col = DuplicateResolver(judge=judge)(PhysicalAssetCollection([cab, box, beside]))
    assert sorted(a.id for a in col) == [
        'beside',
        'box',
    ]  # the box was seen in more frames
    kept = col.get('box')
    assert kept.attributes['duplicate_merged'] == ['cab']
    assert kept.attributes['n_images'] == 56 and kept.attributes['n_observations'] == 56
    assert len(kept.image_assets) == 4  # the cab's crops came along


def test_low_confidence_and_chains_and_keep_duplicates(tmp_path):
    a = _asset(tmp_path, 'a', n_images=5)
    b = _asset(tmp_path, 'b', east_m=3.0, n_images=9)
    c = _asset(tmp_path, 'c', east_m=6.0, n_images=7)

    # a~b at 0.5 is ignored; b~c and a~c at 0.95 chain all three.
    def judge(paths_a, paths_b):
        pair = {Path(paths_a[0]).stem[0], Path(paths_b[0]).stem[0]}
        return (True, 0.5) if pair == {'a', 'b'} else (True, 0.95)

    col = DuplicateResolver(judge=judge, keep_duplicates=True)(
        PhysicalAssetCollection([a, b, c])
    )
    assert len(col) == 3
    assert col.get('b').attributes['duplicate_merged'] == ['a', 'c']
    assert col.get('a').attributes['duplicate_of'] == 'b'
    assert col.get('c').attributes['duplicate_of'] == 'b'
    only_strong = DuplicateResolver(judge=lambda x, y: (True, 0.5))(
        PhysicalAssetCollection(
            [_asset(tmp_path, 'a'), _asset(tmp_path, 'b', east_m=3.0)]
        )
    )
    assert len(only_strong) == 2


class FakeModel:
    """Answers the pair prompt from the composite's file name."""

    def __init__(self, same_ids):
        self.same_ids = same_ids
        self.model_id = 'fake-pairs'
        self.calls = []
        self.lock = threading.Lock()

    def run_inference(self, image_inputs, prompt, **kwargs):
        path = Path(image_inputs[0])
        with self.lock:
            self.calls.append((path, prompt))
        _, a, b = path.stem.split('_', 2)
        same = {a, b} in self.same_ids
        text = json.dumps({'same': same, 'confidence': 0.96, 'reason': 'test'})
        return type('Out', (), {'text': text})()


def test_model_route_builds_a_composite_and_merges(tmp_path):
    a = _asset(tmp_path, 'a', n_images=5)
    b = _asset(tmp_path, 'b', east_m=2.0, n_images=9)
    c = _asset(tmp_path, 'c', east_m=5.0, n_images=7)
    model = FakeModel([{'a', 'b'}])
    resolver = DuplicateResolver(
        model, save_directory=tmp_path / 'pairs', max_workers=2
    )
    col = resolver(PhysicalAssetCollection([a, b, c]))
    assert sorted(x.id for x in col) == ['b', 'c']
    assert len(model.calls) == 3  # a-b, a-c, b-c all within 8 m
    composite, prompt = model.calls[0]
    assert prompt == PAIR_PROMPT and composite.is_file()
    with Image.open(composite) as im:
        assert im.size[1] == 840


def test_parse_pair_verdict_handles_fences_lists_and_strings():
    assert parse_pair_verdict('```json\n{"same": "yes", "confidence": "0.8"}\n```') == (
        True,
        0.8,
        '',
    )
    assert parse_pair_verdict(
        '[{"same": false, "confidence": 1.4, "reason": "r"}]'
    ) == (False, 1.0, 'r')
    assert parse_pair_verdict('{"same": "maybe"}') == (None, None, '')
    assert parse_pair_verdict('') == (None, None, '')
