"""Tests for the interface shortcuts: one namespace, fewer arguments, clear errors."""

import logging
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from PIL import Image
from shapely.geometry import Point

import rapidtools as rt
from rapidtools.core import ImageAsset, PhysicalAsset, PhysicalAssetCollection
from rapidtools.data_sources.mapillary_client import (
    MapillaryClient,
    resolve_mapillary_token,
)
from rapidtools.gui.workflow import InferenceSettings, StreetDetectionSettings
from rapidtools.models.base import ModelOutput
from rapidtools.processing import AssetAnalyzer, MapillaryFeatureExtractor, Pipeline
from rapidtools.processing import image_analyzers as ia


def _asset(asset_id, **attributes):
    return PhysicalAsset(id=asset_id, geometry=Point(0.0, 0.0), attributes=attributes)


# ==========================================
# Top-level namespace and import cost
# ==========================================
def test_model_factory_and_helpers_are_top_level():
    for name in (
        'load',
        'GenerationConfig',
        'ModelOutput',
        'list_providers',
        'catalog',
        'ImageCollection',
        'PipelineStep',
        'list_datasets',
        'models',
    ):
        assert name in rt.__all__ and name in dir(rt), name
    assert rt.load is rt.models.load
    assert rt.GenerationConfig is rt.models.GenerationConfig
    assert 'gemini' in rt.list_providers('api')
    assert rt.catalog('sam3')[0]['model_id'] == 'facebook/sam3'


def test_deprecated_analyzers_are_reachable_but_not_advertised():
    assert not any(
        name.endswith('AssetAnalyzer') for name in rt.__all__ if name != 'AssetAnalyzer'
    )
    assert rt.Gemma4AssetAnalyzer is ia.Gemma4AssetAnalyzer  # still resolves
    with pytest.warns(DeprecationWarning, match='future release'):
        ia._warn_deprecated('Gemma4AssetAnalyzer', 'gemma4')


def test_pipeline_and_extractors_do_not_import_torch():
    code = (
        'import sys, rapidtools as rt\n'
        'rt.Pipeline; rt.AssetAnalyzer; rt.AerialImageryExtractor\n'
        'rt.MapillaryFeatureExtractor; rt.DroneImageExtractor; rt.BoundingBox\n'
        "heavy = [m for m in ('torch', 'transformers', 'matplotlib.pyplot', 'pandas',"
        " 'huggingface_hub') if m in sys.modules]\n"
        'print(heavy)\n'
    )
    out = subprocess.run(
        [sys.executable, '-W', 'ignore', '-c', code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == '[]'


def test_model_wrappers_resolve_lazily_in_the_analyzer_module():
    assert ia._model_class(int) is int
    with pytest.raises(AttributeError):
        _ = ia.NotAWrapper
    assert ia._model_class('GeminiInference') is rt.models.GeminiInference


# ==========================================
# Pipeline
# ==========================================
class _Detector:
    stage = rt.Stage.DETECT
    cancel_event = None

    def __call__(self, collection):
        collection.add(_asset('found'))
        return collection


def test_pipeline_runs_without_an_input_collection():
    result = Pipeline([_Detector()]).run()
    assert [a.id for a in result] == ['found']
    assert len(Pipeline().run()) == 0  # empty pipeline, empty collection


def test_pipeline_shares_its_cancel_event_with_steps_that_have_none():
    stop, own = threading.Event(), threading.Event()

    class Keeps(_Detector):
        cancel_event = own

    class NoAttribute:
        stage = rt.Stage.EXPORT

        def __call__(self, collection):
            return collection

    first, second = _Detector(), Keeps()
    Pipeline([first, second, NoAttribute()], cancel_event=stop).run()
    assert first.cancel_event is stop and second.cancel_event is own
    # Without an event on the pipeline, steps are left alone:
    third = _Detector()
    Pipeline([third]).run()
    assert third.cancel_event is None


def test_pipeline_explains_a_step_that_returns_no_collection():
    class Both:
        stage = rt.Stage.REGULARIZE

        def __call__(self, collection):
            return collection, collection

    with pytest.raises(TypeError, match="Both returned tuple.*output='polygons'"):
        Pipeline([Both()]).run()


# ==========================================
# Models and the analyzer
# ==========================================
def test_load_names_the_arguments_a_wrapper_takes():
    with pytest.raises(
        TypeError, match="load\\('gemma4'\\) does not take 'load_in_4bit'"
    ):
        rt.load('gemma4', load_in_4bit=True)
    with pytest.raises(TypeError, match='accepts: .*model_id'):
        rt.load('gemini', modle_id='x')
    with pytest.raises(KeyError, match='Unknown model provider'):
        rt.load('nope')


class _FakeModel:
    INFO = None
    model_id = 'fake-1'

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run_inference(self, image_inputs, prompt, **kwargs):
        return ModelOutput(text='Level: 3')


def _with_image(tmp_path):
    path = tmp_path / 'a.jpg'
    Image.new('RGB', (8, 8)).save(path)
    asset = _asset('a')
    asset.add_image_assets(ImageAsset(id='img', path=path))
    return PhysicalAssetCollection([asset])


def test_analyzer_builds_the_model_from_a_provider_name(monkeypatch, tmp_path):
    built = {}

    def fake_load(provider, **kwargs):
        built.update(provider=provider, **kwargs)
        return _FakeModel(**kwargs)

    monkeypatch.setattr(rt.models, 'load', fake_load)
    analyzer = AssetAnalyzer('gemini', prompt='Rate it', model_id='m', api_key='k')
    assert built == {'provider': 'gemini', 'model_id': 'm', 'api_key': 'k'}
    assert isinstance(analyzer.model, _FakeModel)
    result = analyzer(_with_image(tmp_path))
    assert result.get('a').attributes['vlm_level'] == '3'
    # Options for the model only make sense with a provider name:
    with pytest.raises(TypeError, match='only be given with a provider name'):
        AssetAnalyzer(_FakeModel(), prompt='Rate it', model_id='m')


def test_analyzer_refuses_to_run_without_a_prompt(tmp_path):
    analyzer = AssetAnalyzer(_FakeModel(), mode='api')
    assert len(analyzer(PhysicalAssetCollection())) == 0  # nothing to do: no error
    with pytest.raises(ValueError, match='has no prompt'):
        analyzer(_with_image(tmp_path))
    analyzer.prompt = 'Rate it'
    assert analyzer(_with_image(tmp_path)).get('a').attributes['vlm_level'] == '3'


# ==========================================
# Mapillary token and extractor options
# ==========================================
def test_mapillary_token_from_argument_file_or_environment(tmp_path, monkeypatch):
    monkeypatch.delenv('MAPILLARY_ACCESS_TOKEN', raising=False)
    monkeypatch.delenv('MAPILLARY_TOKEN', raising=False)
    assert resolve_mapillary_token(' "MLY|1|a" ') == 'MLY|1|a'
    token_file = tmp_path / 'token.txt'
    token_file.write_text('MLY|2|b\n')
    assert resolve_mapillary_token(token_file) == 'MLY|2|b'
    assert resolve_mapillary_token(str(token_file)) == 'MLY|2|b'
    with pytest.raises(FileNotFoundError):
        resolve_mapillary_token(tmp_path / 'missing.txt')
    with pytest.raises(ValueError, match='MAPILLARY_ACCESS_TOKEN'):
        resolve_mapillary_token()
    with pytest.raises(ValueError, match="download_dataset\\('mapillary_token'\\)"):
        MapillaryFeatureExtractor(classes=['cars'])
    monkeypatch.setenv('MAPILLARY_TOKEN', 'MLY|3|c')
    assert resolve_mapillary_token() == 'MLY|3|c'
    monkeypatch.setenv('MAPILLARY_ACCESS_TOKEN', 'MLY|4|d')
    assert MapillaryClient(save_dir=tmp_path).access_token == 'MLY|4|d'
    assert MapillaryClient(token_file, tmp_path).access_token == 'MLY|2|b'
    extractor = MapillaryFeatureExtractor(classes=['cars'], save_directory=tmp_path)
    assert extractor.client.access_token == 'MLY|4|d'
    cropper = rt.MapillaryObjectImageExtractor(tmp_path, access_token=token_file)
    assert cropper.client.access_token == 'MLY|2|b'
    images = rt.MapillaryImageExtractor(save_directory=tmp_path)
    assert images.client.access_token == 'MLY|4|d'


def test_feature_extractor_options_are_validated_by_name(tmp_path):
    extractor = MapillaryFeatureExtractor(
        classes=['cars'],
        access_token='MLY|1|a',
        save_directory=tmp_path,
        occupancy_threshold=0.7,
        cluster_radius_m=5.0,
        sam3_batch_size=2,
    )
    assert extractor.occupancy_threshold == 0.7 and extractor.cluster_radius_m == 5.0
    assert extractor.sam3_batch_size == 2 and extractor.fusion_range_m == 30.0
    with pytest.raises(
        TypeError,
        match="'occupancy_treshold' \\(did you mean 'occupancy_threshold'\\?\\)",
    ):
        MapillaryFeatureExtractor(
            classes=['cars'], access_token='MLY|1|a', occupancy_treshold=0.7
        )
    with pytest.raises(TypeError, match="unknown option\\(s\\): 'zzz'"):
        MapillaryFeatureExtractor(classes=['cars'], access_token='MLY|1|a', zzz=1)


def test_empty_mapillary_search_says_why(tmp_path, caplog):
    from rapidtools.core import BoundingBox, ImageCollection

    class Empty:
        def fetch_images_in_bbox(self, bbox, **kwargs):
            return ImageCollection()

    extractor = MapillaryFeatureExtractor(
        classes=['cars'],
        client=Empty(),
        region=BoundingBox(0, 0, 1, 1),
        start_date='2026-01-01',
        save_directory=tmp_path,
    )
    with caplog.at_level(logging.WARNING):
        assert len(extractor()) == 0
    assert 'filter_rapid_only=False' in caplog.text
    assert 'between 2026-01-01 and today' in caplog.text


# ==========================================
# Collections and datasets
# ==========================================
def test_collection_repr_is_one_line():
    assert repr(PhysicalAssetCollection()) == '<PhysicalAssetCollection: empty>'
    one = PhysicalAssetCollection([_asset('a', asset_type='building')])
    assert repr(one) == '<PhysicalAssetCollection: 1 asset (building: 1)>'
    many = PhysicalAssetCollection(
        [_asset(f'b{i}', asset_type='building') for i in range(1200)]
        + [_asset(f't{i}', asset_type=f'type{i}') for i in range(5)]
        + [_asset('u')]
    )
    text = repr(many)
    assert text.startswith('<PhysicalAssetCollection: 1,206 assets (building: 1,200, ')
    assert text.endswith('... 3 more types)>') and '\n' not in text


def test_collection_filter_and_single_missing_attribute_warning(caplog):
    col = PhysicalAssetCollection(
        [
            _asset('a', level=3),
            _asset('b', level=1),
            _asset('c'),
            _asset('d'),
            _asset('e'),
            _asset('f'),
        ]
    )
    assert [a.id for a in col.filter(lambda a: a.attributes.get('level', 0) > 2)] == [
        'a'
    ]
    with caplog.at_level(logging.WARNING):
        kept = col.filter_by_attribute('level', 1, operator='>=')
    assert [a.id for a in kept] == ['a', 'b']
    warnings = [r for r in caplog.records if 'Missing attribute' in r.getMessage()]
    assert len(warnings) == 1
    assert "4 of 6 assets have no 'level'" in warnings[0].getMessage()
    assert '(c, d, e, ...)' in warnings[0].getMessage()


def test_geojson_export_can_leave_images_out(tmp_path):
    col = _with_image(tmp_path)
    col.get('a').add_attributes({'level': 3})
    full = col.to_geojson()
    assert 'image_assets' in full['features'][0]['properties']
    lean = col.to_geojson(tmp_path / 'out.geojson', include_images=False)
    assert lean['features'][0]['properties'] == {'level': 3}
    both = col.to_geojson(include_images=False, ignore_properties=['level'])
    assert both['features'][0]['properties'] == {}


def test_list_datasets_and_failed_downloads_are_reported(tmp_path, monkeypatch, caplog):
    from rapidtools import datasets

    listing = rt.list_datasets()
    assert listing['mapillary_token'] == ['mapillary_token.txt']
    assert set(listing) == set(datasets.DATASET_REGISTRY)

    def offline(*args, **kwargs):
        raise datasets.requests.exceptions.ConnectionError('offline')

    monkeypatch.setattr(datasets.requests, 'get', offline)
    with caplog.at_level(logging.ERROR):
        assert rt.download_dataset('mapillary_token', tmp_path) == []
    assert '1 of 1 file(s) could not be downloaded (mapillary_token.txt)' in caplog.text


# ==========================================
# GUI settings
# ==========================================
def test_gui_street_inference_needs_no_raster(tmp_path):
    settings = InferenceSettings(
        raster_path=None,
        output_dir=tmp_path,
        prompt='Describe it',
        backend='gemma4',
        imagery='mapillary_objects',
        mapillary_token='MLY|1|a',
    )
    assert settings.raster_path is None
    with pytest.raises(ValueError, match='Aerial crops need an image'):
        InferenceSettings(
            raster_path=None,
            output_dir=tmp_path,
            prompt='Describe it',
            backend='gemma4',
        )
    street = StreetDetectionSettings(
        output_dir=tmp_path,
        classes=['fire hydrants'],
        access_token='MLY|1|a',
        region=rt.BoundingBox(0, 0, 1, 1),
        detection_source='map_features',
    )
    assert street.detection_source == 'map_features'
    assert isinstance(Path(settings.output_dir), Path)
