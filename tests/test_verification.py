"""Offline tests for the DetectionVerifier pipeline step."""

import json
import threading
from pathlib import Path

import pytest
from shapely.geometry import Point

import rapidtools
from rapidtools.core import (
    ImageAsset,
    OperationCancelled,
    PhysicalAsset,
    PhysicalAssetCollection,
)
from rapidtools.models.base import GenerationConfig, ModelOutput
from rapidtools.processing import (
    AssetAnalyzer,
    DetectionVerifier,
    Pipeline,
    RateLimitPolicy,
    Stage,
)
from rapidtools.processing import verification as ver

NO_WAIT = RateLimitPolicy(cooldown_seconds=0, max_asset_retries=0)


class FakeModel:
    """Answers the verification prompt from the first image's file name."""

    def __init__(self, model_id='fake-verifier'):
        self.model_id = model_id
        self.calls: list[tuple[list[Path], str, object]] = []
        self.lock = threading.Lock()

    def run_inference(self, image_inputs, prompt, **kwargs):
        paths = [Path(p) for p in image_inputs]
        with self.lock:
            self.calls.append((paths, prompt, kwargs.get('config')))
        name = paths[0].stem
        if 'bad' in name:
            return None
        if 'garbage' in name:
            return ModelOutput(text='I really cannot tell what this is.')
        if 'yesstr' in name:
            reply = {'match': 'yes', 'confidence': '0.8', 'label': 'van'}
        elif 'truestr' in name:
            reply = {'match': 'true', 'confidence': 0.7, 'label': 'bus'}
        elif 'nostr' in name:
            reply = {'match': 'no', 'confidence': 0.9, 'label': 'mailbox'}
        elif 'noconf' in name:
            reply = {'match': True, 'label': 'pickup'}
        elif 'lowconf' in name:
            reply = {'match': True, 'confidence': 0.3, 'label': 'maybe a car'}
        elif 'car' in name:
            reply = {'match': True, 'confidence': 0.9, 'label': 'parked sedan'}
        else:  # trees, bins, ...
            reply = {'match': False, 'confidence': 0.95, 'label': 'tree trunk'}
        return ModelOutput(text='```json\n' + json.dumps(reply) + '\n```')


def _image(tmp_path: Path, name: str, range_m=None) -> ImageAsset:
    path = tmp_path / f'{name}.jpg'
    path.write_bytes(b'\xff\xd8\xff\xd9')
    props = {'image_id': name, 'label': 'object--vehicle--car'}
    if range_m is not None:
        props['range_m'] = range_m
    return ImageAsset(id=name, path=path, properties=props)


def _asset(uid: str, asset_type='vehicles', images=()) -> PhysicalAsset:
    asset = PhysicalAsset(
        id=uid, geometry=Point(0, 0), attributes={'asset_type': asset_type}
    )
    for img in images:
        asset.add_image_assets(img)
    return asset


def _collection(tmp_path: Path) -> PhysicalAssetCollection:
    col = PhysicalAssetCollection()
    col.add(
        [
            _asset('v_car', images=[_image(tmp_path, 'car_1', 8.0)]),
            _asset('v_tree', images=[_image(tmp_path, 'tree_1', 6.0)]),
            _asset('v_noimg'),
        ]
    )
    return col


# ------------------------------------------------------------ core verdicts
def test_accepts_matches_and_removes_rejections(tmp_path):
    model = FakeModel()
    verifier = DetectionVerifier(model, rate_limit=NO_WAIT)
    col = _collection(tmp_path)
    out = verifier(col)
    assert out is col
    assert sorted(a.id for a in out) == ['v_car', 'v_noimg']
    car = out.get('v_car')
    assert car.attributes['verify_match'] is True
    assert car.attributes['verify_confidence'] == 0.9
    assert car.attributes['verify_label'] == 'parked sedan'
    assert car.attributes['verify_accepted'] is True
    assert car.attributes['verify_model'] == 'fake-verifier'
    # The verifier must not leave the analyzer's bookkeeping behind:
    assert 'ai_model_used' not in car.attributes
    # JSON mode at temperature 0 by default:
    config = model.calls[0][2]
    assert isinstance(config, GenerationConfig)
    assert config.json_mode is True and config.temperature == 0.0


def test_keep_rejected_flags_instead_of_removing(tmp_path):
    verifier = DetectionVerifier(FakeModel(), keep_rejected=True, rate_limit=NO_WAIT)
    out = verifier(_collection(tmp_path))
    assert sorted(a.id for a in out) == ['v_car', 'v_noimg', 'v_tree']
    tree = out.get('v_tree')
    assert tree.attributes['verify_accepted'] is False
    assert tree.attributes['verify_match'] is False
    assert tree.attributes['verify_label'] == 'tree trunk'


def test_unverifiable_assets_are_kept_with_none(tmp_path, caplog):
    col = PhysicalAssetCollection()
    col.add(
        [
            _asset('v_noimg'),
            _asset('v_bad', images=[_image(tmp_path, 'bad_1', 5.0)]),
            _asset('v_garbage', images=[_image(tmp_path, 'garbage_1', 5.0)]),
        ]
    )
    with caplog.at_level('INFO', logger='rapidtools.processing.verification'):
        out = DetectionVerifier(FakeModel(), rate_limit=NO_WAIT)(col)
    assert len(out) == 3
    for uid in ('v_noimg', 'v_bad', 'v_garbage'):
        assert out.get(uid).attributes['verify_accepted'] is None
        assert out.get(uid).attributes['verify_model'] == 'fake-verifier'
    assert 'verify_match' not in out.get('v_noimg').attributes
    assert out.get('v_garbage').attributes['verify_raw'].startswith('I really')
    assert '1 assets have no downloaded images' in caplog.text
    assert '1 accepted' not in caplog.text
    assert '0 accepted, 0 rejected, 3 unverifiable' in caplog.text


@pytest.mark.parametrize(
    ('min_confidence', 'expected'),
    [(0.5, ['v_car', 'v_noconf']), (0.2, ['v_car', 'v_lowconf', 'v_noconf'])],
)
def test_min_confidence_threshold(tmp_path, min_confidence, expected):
    col = PhysicalAssetCollection()
    col.add(
        [
            _asset('v_car', images=[_image(tmp_path, 'car_1')]),
            _asset('v_lowconf', images=[_image(tmp_path, 'lowconf_1')]),
            _asset('v_noconf', images=[_image(tmp_path, 'noconf_1')]),
        ]
    )
    verifier = DetectionVerifier(
        FakeModel(), min_confidence=min_confidence, rate_limit=NO_WAIT
    )
    out = verifier(col)
    assert sorted(a.id for a in out) == expected
    # A missing confidence counts as 1.0 when the model said it matches:
    assert out.get('v_noconf').attributes['verify_confidence'] is None
    assert out.get('v_noconf').attributes['verify_accepted'] is True


@pytest.mark.parametrize(
    ('name', 'accepted', 'match'),
    [('yesstr_1', True, True), ('truestr_1', True, True), ('nostr_1', False, False)],
)
def test_string_match_values_are_normalised(tmp_path, name, accepted, match):
    col = PhysicalAssetCollection([_asset('v', images=[_image(tmp_path, name)])])
    out = DetectionVerifier(FakeModel(), keep_rejected=True, rate_limit=NO_WAIT)(col)
    attrs = out.get('v').attributes
    assert attrs['verify_match'] is match
    assert attrs['verify_accepted'] is accepted
    assert isinstance(attrs['verify_confidence'], float)


def test_normalisation_helpers():
    assert ver.normalize_match('Yes.') is True
    assert ver.normalize_match('FALSE') is False
    assert ver.normalize_match(0) is False
    assert ver.normalize_match('unsure') is None
    assert ver.normalize_confidence('85%') == 0.85
    assert ver.normalize_confidence(90) == 0.9
    assert ver.normalize_confidence(True) == 1.0
    assert ver.normalize_confidence('high') is None
    assert ver.normalize_confidence(None) is None


# ------------------------------------------------------- image selection
def test_nearest_images_are_sent(tmp_path):
    model = FakeModel()
    images = [
        _image(tmp_path, 'far_car', 30.0),
        _image(tmp_path, 'unknown_car'),  # no range: goes last
        _image(tmp_path, 'near_car', 5.0),
        _image(tmp_path, 'mid_car', 12.0),
    ]
    col = PhysicalAssetCollection([_asset('v', images=images)])
    DetectionVerifier(model, max_images_per_asset=2, rate_limit=NO_WAIT)(col)
    assert len(model.calls) == 1
    assert [p.stem for p in model.calls[0][0]] == ['near_car', 'mid_car']
    # The asset's images are now ordered nearest first, unknown range last:
    assert [img.id for img in col.get('v').image_assets] == [
        'near_car',
        'mid_car',
        'far_car',
        'unknown_car',
    ]


def test_images_not_on_disk_are_ignored(tmp_path):
    model = FakeModel()
    missing = ImageAsset(
        id='ghost',
        path=tmp_path / 'ghost_car.jpg',
        properties={'range_m': 1.0},
        allow_missing_file=True,
    )
    col = PhysicalAssetCollection(
        [_asset('v', images=[missing, _image(tmp_path, 'car_1', 9.0)])]
    )
    DetectionVerifier(model, max_images_per_asset=1, rate_limit=NO_WAIT)(col)
    assert [p.stem for p in model.calls[0][0]] == ['car_1']


# --------------------------------------------------------------- prompts
def test_default_prompt_describes_known_and_unknown_types(tmp_path):
    model = FakeModel()
    col = PhysicalAssetCollection()
    col.add(
        [
            _asset('v', 'vehicles', [_image(tmp_path, 'car_1')]),
            _asset('p', 'utility_pole', [_image(tmp_path, 'car_2')]),
            _asset('a', 'awning', [_image(tmp_path, 'car_3')]),
        ]
    )
    DetectionVerifier(model, rate_limit=NO_WAIT)(col)
    prompts = {c[0][0].stem: c[1] for c in model.calls}
    assert 'is the marked object a vehicle (car, truck, bus' in prompts['car_1']
    assert 'is the marked object a utility pole' in prompts['car_2']
    assert 'is the marked object an awning' in prompts['car_3']
    for prompt in prompts.values():
        assert '{description}' not in prompt
        assert '"match"' in prompt and '"confidence"' in prompt
        assert '"label"' in prompt
        assert 'partly hidden' in prompt and 'outline' in prompt


def test_descriptions_and_prompt_override(tmp_path):
    model = FakeModel()
    col = PhysicalAssetCollection([_asset('v', images=[_image(tmp_path, 'car_1')])])
    verifier = DetectionVerifier(
        model,
        descriptions={'vehicles': 'a burnt-out vehicle'},
        prompt='Is it {description}? JSON {"match": bool}',
        rate_limit=NO_WAIT,
    )
    verifier(col)
    assert model.calls[0][1] == 'Is it a burnt-out vehicle? JSON {"match": bool}'
    assert verifier.description_for('Vehicles') == 'a burnt-out vehicle'
    assert ver.describe_asset_type(None) == 'the object it was detected as'
    assert ver.describe_asset_type('fire_hydrants') == 'a fire hydrant'


# ------------------------------------------------------------ classifier
def test_classifier_callable_path(tmp_path):
    seen = []

    def classifier(paths, description):
        seen.append(([p.stem for p in paths], description))
        return [
            ('car' in p.stem, 0.9 if 'car' in p.stem else 0.2, p.stem.split('_')[0])
            for p in paths
        ]

    col = PhysicalAssetCollection()
    col.add(
        [
            _asset(
                'v_mixed',
                images=[
                    _image(tmp_path, 'tree_1', 3.0),
                    _image(tmp_path, 'car_1', 7.0),
                ],
            ),
            _asset('v_tree', images=[_image(tmp_path, 'tree_2', 4.0)]),
            _asset('v_noimg'),
        ]
    )
    out = DetectionVerifier(classifier=classifier, min_confidence=0.8)(col)
    assert sorted(a.id for a in out) == ['v_mixed', 'v_noimg']
    assert seen == [
        (['tree_1', 'car_1'], ver.DEFAULT_DESCRIPTIONS['vehicles']),
        (['tree_2'], ver.DEFAULT_DESCRIPTIONS['vehicles']),
    ]
    mixed = out.get('v_mixed').attributes
    assert mixed['verify_match'] is True
    assert mixed['verify_confidence'] == 0.9
    assert mixed['verify_label'] == 'car'
    assert mixed['verify_accepted'] is True
    assert mixed['verify_model'] == 'custom'
    assert out.get('v_noimg').attributes['verify_accepted'] is None
    assert out.get('v_noimg').attributes['verify_model'] == 'custom'


def test_classifier_errors_leave_asset_unverified(tmp_path):
    def classifier(paths, description):
        raise RuntimeError('boom')

    col = PhysicalAssetCollection([_asset('v', images=[_image(tmp_path, 'car_1')])])
    out = DetectionVerifier(classifier=classifier)(col)
    assert out.get('v').attributes['verify_accepted'] is None


# -------------------------------------------------------- construction
def test_constructor_validation():
    with pytest.raises(ValueError, match='exactly one'):
        DetectionVerifier()
    with pytest.raises(ValueError, match='exactly one'):
        DetectionVerifier(FakeModel(), classifier=lambda p, d: [])
    with pytest.raises(ValueError, match='min_confidence'):
        DetectionVerifier(FakeModel(), min_confidence=1.5)
    with pytest.raises(ValueError, match='max_images_per_asset'):
        DetectionVerifier(FakeModel(), max_images_per_asset=0)
    with pytest.raises(ValueError, match='attribute_prefix'):
        DetectionVerifier(FakeModel(), attribute_prefix='')
    with pytest.raises(TypeError, match='callable'):
        DetectionVerifier(classifier='not callable')
    verifier = DetectionVerifier(FakeModel('m-1'), attribute_prefix='chk')
    assert verifier.model_name == 'm-1'
    assert verifier.attribute_prefix == 'chk'
    assert verifier.max_images_per_asset == 2 and verifier.min_confidence == 0.5


def test_custom_prefix_is_used(tmp_path):
    col = PhysicalAssetCollection([_asset('v', images=[_image(tmp_path, 'car_1')])])
    DetectionVerifier(FakeModel(), attribute_prefix='chk', rate_limit=NO_WAIT)(col)
    attrs = col.get('v').attributes
    assert attrs['chk_accepted'] is True and 'verify_accepted' not in attrs


def test_empty_collection_is_returned(caplog):
    col = PhysicalAssetCollection()
    assert DetectionVerifier(FakeModel())(col) is col


# ------------------------------------------------------------ cancellation
def test_cancellation_propagates(tmp_path):
    stop = threading.Event()
    stop.set()
    with pytest.raises(OperationCancelled):
        DetectionVerifier(FakeModel(), cancel_event=stop)(_collection(tmp_path))

    def classifier(paths, description):
        stop2.set()
        return [(True, 1.0, 'car') for _ in paths]

    stop2 = threading.Event()
    col = PhysicalAssetCollection()
    col.add(
        [
            _asset('a', images=[_image(tmp_path, 'car_a')]),
            _asset('b', images=[_image(tmp_path, 'car_b')]),
        ]
    )
    with pytest.raises(OperationCancelled):
        DetectionVerifier(classifier=classifier, cancel_event=stop2)(col)


# ------------------------------------------------------- stage and exports
def test_stage_order_and_pipeline_sorting(tmp_path):
    assert Stage.VERIFY == 45
    assert Stage.EXTRACT_IMAGERY < Stage.SEGMENT < Stage.VERIFY < Stage.ANALYZE
    assert DetectionVerifier.stage is Stage.VERIFY

    verifier = DetectionVerifier(FakeModel(), rate_limit=NO_WAIT)
    analyzer = AssetAnalyzer(
        FakeModel('analysis'), prompt='p', attribute_prefix='an', rate_limit=NO_WAIT
    )
    pipeline = Pipeline([analyzer, verifier])
    pipeline._sort_steps()
    assert pipeline.steps == [verifier, analyzer]

    # Rejected objects never reach the analysis model:
    analysis_model = analyzer.model
    out = pipeline.run(_collection(tmp_path))
    assert sorted(a.id for a in out) == ['v_car', 'v_noimg']
    assert [p.stem for c in analysis_model.calls for p in c[0]] == ['car_1']
    assert out.get('v_car').attributes['an_match'] is True


def test_top_level_export():
    import importlib

    assert 'DetectionVerifier' in rapidtools.__all__
    assert 'DetectionVerifier' in rapidtools._LAZY_ATTRIBUTES
    cls = rapidtools.DetectionVerifier
    assert cls is importlib.import_module('rapidtools.processing').DetectionVerifier
    assert cls is ver.DetectionVerifier
