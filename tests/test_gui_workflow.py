"""Tests for the headless workflow engine behind the rapidtools GUI."""

from pathlib import Path

import pytest
from shapely.geometry import box

import rapidtools
from rapidtools.core import ImageAsset, PhysicalAsset, PhysicalAssetCollection
from rapidtools.gui.workflow import (
    AssetAnalysisWorkflow,
    DetectionSettings,
    InferenceSettings,
    WorkflowCancelled,
    parse_asset_list,
    slugify,
)


# --------------------------------------------------------------------- fakes
def _collection(prefix: str, n: int = 2) -> PhysicalAssetCollection:
    col = PhysicalAssetCollection()
    for i in range(n):
        col.add(
            PhysicalAsset(
                id=f'{prefix}_{i}',
                geometry=box(i, 0, i + 1, 1),
                attributes={'asset_type': prefix},
            )
        )
    return col


class FakeSAM3:
    instances: list = []

    def __init__(self, prompt, **kwargs):
        self.prompt = prompt
        self.kwargs = kwargs
        self.calls: list[tuple[str, Path]] = []
        FakeSAM3.instances.append(self)

    def __call__(self, raster_path):
        self.calls.append((self.prompt, Path(raster_path)))
        if self.prompt == 'nothing':
            return PhysicalAssetCollection()
        return _collection(slugify(self.prompt))


class FakeCropExtractor:
    def __init__(self, dataset, save_directory, **kwargs):
        self.dataset = Path(dataset)
        self.save_directory = Path(save_directory)
        self.kwargs = kwargs

    def __call__(self, collection):
        self.save_directory.mkdir(parents=True, exist_ok=True)
        for asset in collection:
            asset.add_image_assets(
                ImageAsset(
                    id=f'{asset.id}_{self.save_directory.name}_img',
                    path=self.save_directory / f'{asset.id}.jpg',
                    allow_missing_file=True,
                )
            )
        return collection


class FakeBuildingRegularizer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, collection):
        collection.set_attribute('regularized', 'building', overwrite=True)
        return collection


class FakeRoadRegularizer:
    def __call__(self, collection):
        polygons = _collection('road_polygon', 1)
        return _collection('centerline', 1), polygons


class FakeBing:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, region, output_path):
        Path(output_path).write_bytes(b'bing')
        return Path(output_path)


class FakeBoundingBox:
    @classmethod
    def from_raster(cls, path):
        return 'region'


class FakePipeline:
    def __init__(self, cancel_event=None):
        self.steps = []
        self.cancel_event = cancel_event

    def add_step(self, step):
        self.steps.append(step)
        return self

    def run(self, collection):
        for step in self.steps:
            collection = step(collection)
        return collection


class FakeModel:
    """Stands in for any model built through rapidtools.models.load."""

    created: list = []

    def __init__(self, provider, **kwargs):
        self.provider = provider
        self.kwargs = kwargs
        self.model_id = kwargs.get('model_id')
        FakeModel.created.append(self)


class FakeAnalyzer:
    """Stands in for AssetAnalyzer; merges model and analyzer kwargs."""

    created: list = []

    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = {**getattr(model, 'kwargs', {}), **kwargs}
        FakeAnalyzer.created.append(self)

    def __call__(self, collection):
        image_filter = self.kwargs.get('image_filter')
        for asset in collection:
            images = list(asset.image_assets)
            if image_filter is not None:
                images = [img for img in images if image_filter(img)]
            if images:
                asset.add_attributes({'damage_level': 'Major'})
        return collection


@pytest.fixture
def raster(tmp_path: Path) -> Path:
    path = tmp_path / 'scene.tiff'
    path.write_bytes(b'not really a tiff')
    return path


@pytest.fixture
def patched(monkeypatch):
    FakeSAM3.instances.clear()
    FakeAnalyzer.created.clear()
    monkeypatch.setattr(rapidtools, 'SAM3OrthoFeatureExtractor', FakeSAM3)
    monkeypatch.setattr(rapidtools, 'AerialImageryExtractor', FakeCropExtractor)
    monkeypatch.setattr(rapidtools, 'BuildingRegularizer', FakeBuildingRegularizer)
    monkeypatch.setattr(rapidtools, 'RoadwayRegularizer', FakeRoadRegularizer)
    monkeypatch.setattr(rapidtools, 'BingOrthomosaicExtractor', FakeBing)
    monkeypatch.setattr(rapidtools, 'BoundingBox', FakeBoundingBox)
    monkeypatch.setattr(rapidtools, 'Pipeline', FakePipeline)
    import rapidtools.models as models_module
    import rapidtools.processing as processing_module

    FakeModel.created.clear()
    monkeypatch.setattr(
        models_module, 'load', lambda provider, **kw: FakeModel(provider, **kw)
    )
    monkeypatch.setattr(processing_module, 'AssetAnalyzer', FakeAnalyzer)


# ------------------------------------------------------------------- helpers
def test_parse_asset_list_splits_and_deduplicates():
    assert parse_asset_list('building, tree;  road\nBuilding ,, ') == [
        'building',
        'tree',
        'road',
    ]
    assert parse_asset_list('') == []


def test_slugify():
    assert slugify('Swimming  Pool!') == 'swimming_pool'
    assert slugify('***') == 'asset'


# ------------------------------------------------------------------ settings
def test_detection_settings_validation(tmp_path):
    with pytest.raises(ValueError):
        DetectionSettings(raster_path=tmp_path, output_dir=tmp_path, assets=[])
    with pytest.raises(ValueError):
        DetectionSettings(
            raster_path=tmp_path, output_dir=tmp_path, assets=['x'], asset_size_m=0
        )


def test_inference_settings_validation(tmp_path):
    with pytest.raises(ValueError):
        InferenceSettings(raster_path=tmp_path, output_dir=tmp_path, prompt='  ')
    with pytest.raises(ValueError):
        InferenceSettings(raster_path=tmp_path, output_dir=tmp_path, prompt='p')
    # Gemma-4 needs no key:
    InferenceSettings(
        raster_path=tmp_path, output_dir=tmp_path, prompt='p', backend='gemma4'
    )


def test_inference_settings_outline_options(tmp_path):
    base = dict(raster_path=tmp_path, output_dir=tmp_path, prompt='p', backend='gemma4')
    settings = InferenceSettings(**base)
    assert settings.outline_shape == 'geometry'
    assert settings.outline_buffer == 0
    assert settings.outline_width == 6
    assert settings.outline_color == 'red'

    settings = InferenceSettings(
        **base,
        outline_shape='corners',
        outline_buffer='2 m',
        outline_width='1%',
        outline_color='#00ffff',
    )
    assert settings.outline_shape == 'corners'
    assert settings.outline_buffer == '2 m'

    for bad in (
        {'outline_shape': 'hexagon'},
        {'outline_buffer': 'lots'},
        {'outline_width': '3 m'},
        {'outline_width': 0},
        {'outline_color': ''},
        {'outline_color': 'no-such-colour'},
    ):
        with pytest.raises(ValueError):
            InferenceSettings(**base, **bad)


# ----------------------------------------------------------------- detection
def test_detect_multiple_assets_reuses_sam_and_merges(patched, raster, tmp_path):
    out = tmp_path / 'out'
    statuses: list[str] = []
    wf = AssetAnalysisWorkflow(progress_callback=statuses.append)
    result = wf.detect(
        DetectionSettings(
            raster_path=raster,
            output_dir=out,
            assets=['building', 'tree', 'nothing'],
            detect_in_recon_imagery=True,
        )
    )

    # One SAM 3 model for all prompts, scanned against the recon raster:
    assert len(FakeSAM3.instances) == 1
    sam = FakeSAM3.instances[0]
    assert [c[0] for c in sam.calls] == ['building', 'tree', 'nothing']
    assert all(c[1] == raster.resolve() for c in sam.calls)
    assert sam.kwargs['unit'] == 'meters'
    assert result.detection_raster == raster.resolve()

    # Buildings regularized, trees untouched, empty prompt skipped:
    assert len(result.collection) == 4
    assert all(
        a.attributes.get('regularized') == 'building'
        for a in result.collection
        if a.attributes['asset_type'] == 'building'
    )
    assert set(result.per_asset_geojson) == {'building', 'tree'}
    assert (out / 'building_preliminary.geojson').is_file()
    assert (out / 'nothing_preliminary.geojson').is_file()
    assert result.combined_geojson == out / 'assets_final.geojson'
    assert result.combined_geojson.is_file()
    assert any('Detection complete' in s for s in statuses)


def test_detect_uses_bing_basemap_by_default(patched, raster, tmp_path):
    out = tmp_path / 'out'
    wf = AssetAnalysisWorkflow()
    result = wf.detect(
        DetectionSettings(raster_path=raster, output_dir=out, assets=['tree'])
    )
    expected = (out / f'{raster.stem}_bing.tiff').resolve()
    assert result.detection_raster == expected
    assert expected.is_file()
    assert FakeSAM3.instances[0].calls[0][1] == expected

    # A second run reuses the stitched basemap instead of re-downloading:
    expected.write_bytes(b'cached')
    wf.detect(DetectionSettings(raster_path=raster, output_dir=out, assets=['tree']))
    assert expected.read_bytes() == b'cached'


def test_detect_road_regularizer_returns_polygons(patched, raster, tmp_path):
    result = AssetAnalysisWorkflow().detect(
        DetectionSettings(
            raster_path=raster,
            output_dir=tmp_path,
            assets=['road'],
            detect_in_recon_imagery=True,
        )
    )
    assert [a.id for a in result.collection] == ['road_polygon_0']


def test_detect_without_regularization(patched, raster, tmp_path):
    result = AssetAnalysisWorkflow().detect(
        DetectionSettings(
            raster_path=raster,
            output_dir=tmp_path,
            assets=['building'],
            detect_in_recon_imagery=True,
            regularize=False,
        )
    )
    assert not any('regularized' in a.attributes for a in result.collection)


def test_detect_missing_raster(patched, tmp_path):
    with pytest.raises(FileNotFoundError):
        AssetAnalysisWorkflow().detect(
            DetectionSettings(
                raster_path=tmp_path / 'missing.tif',
                output_dir=tmp_path,
                assets=['tree'],
            )
        )


def test_cancel_between_stages(patched, raster, tmp_path):
    wf = AssetAnalysisWorkflow()
    original_call = FakeSAM3.__call__

    def cancelling_call(self, raster_path):
        wf.cancel()
        return original_call(self, raster_path)

    FakeSAM3.__call__ = cancelling_call
    try:
        with pytest.raises(WorkflowCancelled):
            wf.detect(
                DetectionSettings(
                    raster_path=raster,
                    output_dir=tmp_path,
                    assets=['tree', 'building'],
                    detect_in_recon_imagery=True,
                )
            )
    finally:
        FakeSAM3.__call__ = original_call
    # Only the first prompt ran before the cancellation took effect:
    assert len(FakeSAM3.instances[0].calls) == 1


# ----------------------------------------------------------------- inference
def test_analyze_gemini_filters_to_inference_crops(patched, raster, tmp_path):
    out = tmp_path / 'out'
    collection = _collection('building', 3)
    # Pretend detection attached basemap crops from elsewhere:
    stale = FakeCropExtractor(raster, tmp_path / 'stale_crops')
    stale(collection)

    result = AssetAnalysisWorkflow().analyze(
        collection,
        InferenceSettings(
            raster_path=raster,
            output_dir=out,
            prompt='Describe the damage.',
            api_key='secret',
            max_workers=3,
        ),
    )

    analyzer = FakeAnalyzer.created[0]
    assert analyzer.kwargs['api_key'] == 'secret'
    assert analyzer.kwargs['prompt'] == 'Describe the damage.'
    assert analyzer.kwargs['max_workers'] == 3
    image_filter = analyzer.kwargs['image_filter']
    assert image_filter is not None
    stale_image = next(iter(collection)).image_assets.filter(
        lambda img: 'stale_crops' in str(img.path)
    )
    assert not any(image_filter(img) for img in stale_image)

    assert result.n_input == 3
    assert result.n_analyzed == 3
    # Only the crops written for inference were analyzed:
    assert all(
        len([img for img in a.image_assets if image_filter(img)]) == 1
        for a in result.collection
    )
    assert result.geojson_path == (out / 'assets_inferred.geojson').resolve()
    assert result.geojson_path.is_file()
    assert all(a.attributes['damage_level'] == 'Major' for a in result.collection)


def test_analyze_gemma4_passes_batch_size(patched, raster, tmp_path):
    AssetAnalysisWorkflow().analyze(
        _collection('tree'),
        InferenceSettings(
            raster_path=raster,
            output_dir=tmp_path,
            prompt='p',
            backend='gemma4',
            model_id='google/gemma-4-E4B-it',
            batch_size=8,
            only_analyze_inference_crops=False,
        ),
    )
    kwargs = FakeAnalyzer.created[0].kwargs
    assert kwargs['model_id'] == 'google/gemma-4-E4B-it'
    assert kwargs['batch_size'] == 8
    assert kwargs['image_filter'] is None
    assert 'api_key' not in kwargs


def test_analyze_counts_only_assets_with_new_attributes(
    patched, raster, tmp_path, monkeypatch
):
    class NoOpAnalyzer(FakeAnalyzer):
        def __call__(self, collection):
            return collection

    import rapidtools.processing as processing_module

    monkeypatch.setattr(processing_module, 'AssetAnalyzer', NoOpAnalyzer)
    result = AssetAnalysisWorkflow().analyze(
        _collection('building', 2),
        InferenceSettings(
            raster_path=raster, output_dir=tmp_path, prompt='p', api_key='k'
        ),
    )
    # Assets keep their detection attributes but none were actually analyzed:
    assert len(result.collection) == 2
    assert result.n_analyzed == 0


@pytest.mark.parametrize(
    ('backend', 'needs_key', 'default_model'),
    [
        ('claude', True, 'claude-opus-5'),
        ('openai', True, 'gpt-5.5'),
        ('muse_spark', True, 'muse-spark-1.3'),
        ('qwen', True, 'qwen3.8-max'),
        ('llama', False, 'meta-llama/Llama-3.2-11B-Vision-Instruct'),
        ('muse_glimmer', False, 'meta-models/Muse-Glimmer-30B'),
        ('qwen_vl', False, 'Qwen/Qwen3.5-4B'),
    ],
)
def test_analyze_other_backends(
    patched, raster, tmp_path, backend, needs_key, default_model
):
    from rapidtools.gui.workflow import MODEL_BACKENDS, list_available_models

    if needs_key:
        with pytest.raises(ValueError):
            InferenceSettings(
                raster_path=raster, output_dir=tmp_path, prompt='p', backend=backend
            )
    settings = InferenceSettings(
        raster_path=raster,
        output_dir=tmp_path,
        prompt='p',
        backend=backend,
        api_key='k' if needs_key else '',
        load_in_4bit=False,
    )
    assert settings.model_id == default_model
    AssetAnalysisWorkflow().analyze(_collection('tree'), settings)
    kwargs = FakeAnalyzer.created[0].kwargs
    assert kwargs['model_id'] == default_model
    if needs_key:
        assert kwargs['api_key'] == 'k'
    spec = MODEL_BACKENDS[backend]
    if spec.get('supports_4bit'):
        assert kwargs['load_in_4bit'] is False
    else:
        assert 'load_in_4bit' not in kwargs
    if spec['kind'] == 'local' and spec.get('supports_batch'):
        assert kwargs['batch_size'] == settings.batch_size
    assert list_available_models(backend) == spec['models']


def test_unknown_backend_rejected(tmp_path):
    with pytest.raises(ValueError):
        InferenceSettings(
            raster_path=tmp_path, output_dir=tmp_path, prompt='p', backend='bogus'
        )


def test_list_available_models_queries_provider(monkeypatch):
    from rapidtools import models
    from rapidtools.gui.workflow import list_available_models

    monkeypatch.setattr(
        models.OpenAIInference,
        'list_available_models',
        staticmethod(lambda key: ['gpt-4o', 'gpt-x'] if key == 'k' else []),
    )
    from rapidtools.gui.workflow import MODEL_BACKENDS

    assert list_available_models('openai', 'k') == ['gpt-4o', 'gpt-x']
    assert (
        list_available_models('openai', 'wrong') == MODEL_BACKENDS['openai']['models']
    )
    assert list_available_models('gemma4', 'k')[0].startswith('google/gemma-4')


def test_list_available_models_orders_default_first(monkeypatch):
    from rapidtools import models
    from rapidtools.gui.workflow import MODEL_BACKENDS, list_available_models

    remote = ['zeta-model', 'muse-spark-1.1', 'muse-spark-1.3']
    monkeypatch.setattr(
        models.MuseSparkInference,
        'list_available_models',
        staticmethod(lambda key: remote),
    )
    listed = list_available_models('muse_spark', 'k')
    static = MODEL_BACKENDS['muse_spark']['models']
    # Curated order first (default model leads), then unknown remote models:
    assert listed == [m for m in static if m in remote] + ['zeta-model']
    assert listed[0] == MODEL_BACKENDS['muse_spark']['default_model']


def test_analyze_hf_backend(patched, raster, tmp_path):
    settings = InferenceSettings(
        raster_path=raster,
        output_dir=tmp_path,
        prompt='p',
        backend='hf',
        model_id='HuggingFaceTB/SmolVLM-256M-Instruct',
        batch_size=2,
        load_in_4bit=False,
    )
    AssetAnalysisWorkflow().analyze(_collection('tree'), settings)
    assert FakeModel.created[-1].provider == 'hf'
    kwargs = FakeAnalyzer.created[-1].kwargs
    assert kwargs['model_id'] == 'HuggingFaceTB/SmolVLM-256M-Instruct'
    assert kwargs['batch_size'] == 2
    assert kwargs['load_in_4bit'] is False
    assert 'cancel_event' in kwargs


def test_hf_model_listing_merges_hub(monkeypatch):
    from rapidtools import models
    from rapidtools.gui.workflow import MODEL_BACKENDS, list_available_models

    monkeypatch.setattr(
        models.HFVisionInference,
        'search_hub_models',
        staticmethod(lambda limit=40: ['Qwen/Qwen3-VL-4B-Instruct', 'org/new-vlm']),
    )
    listed = list_available_models('hf')
    curated = MODEL_BACKENDS['hf']['models']
    assert listed[: len(curated)] == curated
    assert listed[-1] == 'org/new-vlm'


def test_analyze_cancelled_keeps_partial_results(
    patched, raster, tmp_path, monkeypatch
):
    from rapidtools.core import OperationCancelled

    wf = AssetAnalysisWorkflow()

    class CancellingAnalyzer(FakeAnalyzer):
        def __call__(self, collection):
            first = next(iter(collection))
            first.add_attributes({'damage_level': 'Major'})
            raise OperationCancelled('stop')

    import rapidtools.processing as processing_module

    monkeypatch.setattr(processing_module, 'AssetAnalyzer', CancellingAnalyzer)
    collection = _collection('building', 3)
    with pytest.raises(WorkflowCancelled):
        wf.analyze(
            collection,
            InferenceSettings(
                raster_path=raster, output_dir=tmp_path, prompt='p', api_key='k'
            ),
        )
    partial = tmp_path / 'assets_inferred_partial.geojson'
    assert partial.is_file()
    import json

    features = json.loads(partial.read_text())['features']
    assert len(features) == 3  # detection attributes keep every asset non-empty
    assert sum('damage_level' in f['properties'] for f in features) == 1


def test_detect_cancelled_inside_component(patched, raster, tmp_path):
    from rapidtools.core import OperationCancelled

    class CancellingSAM(FakeSAM3):
        def __call__(self, raster_path):
            raise OperationCancelled('tile loop stopped')

    import rapidtools

    rapidtools.SAM3OrthoFeatureExtractor = CancellingSAM
    with pytest.raises(WorkflowCancelled):
        AssetAnalysisWorkflow().detect(
            DetectionSettings(
                raster_path=raster,
                output_dir=tmp_path,
                assets=['tree'],
                detect_in_recon_imagery=True,
            )
        )


def test_components_receive_cancel_event(patched, raster, tmp_path):
    wf = AssetAnalysisWorkflow()
    wf.detect(
        DetectionSettings(
            raster_path=raster,
            output_dir=tmp_path,
            assets=['building'],
            detect_in_recon_imagery=True,
        )
    )
    sam = FakeSAM3.instances[0]
    assert sam.kwargs['cancel_event'] is wf._cancel_event
    wf.analyze(
        _collection('tree'),
        InferenceSettings(
            raster_path=raster, output_dir=tmp_path, prompt='p', api_key='k'
        ),
    )
    assert FakeAnalyzer.created[-1].kwargs['cancel_event'] is wf._cancel_event


def test_analyze_empty_collection(patched, raster, tmp_path):
    with pytest.raises(ValueError):
        AssetAnalysisWorkflow().analyze(
            PhysicalAssetCollection(),
            InferenceSettings(
                raster_path=raster, output_dir=tmp_path, prompt='p', api_key='k'
            ),
        )


# ------------------------------------------------------------ coverage extras
def test_detect_reports_when_nothing_is_found(patched, raster, tmp_path):
    messages: list[str] = []
    wf = AssetAnalysisWorkflow(progress_callback=messages.append)
    result = wf.detect(
        DetectionSettings(
            raster_path=raster,
            output_dir=tmp_path,
            assets=['nothing'],
            detect_in_recon_imagery=True,
        )
    )
    assert len(result.collection) == 0
    assert result.combined_geojson is None
    assert any('no assets were found' in m for m in messages)


def test_list_available_models_errors_and_provider_failure(monkeypatch, caplog):
    from rapidtools import models
    from rapidtools.gui.workflow import MODEL_BACKENDS, list_available_models

    with pytest.raises(ValueError, match='Unknown inference backend'):
        list_available_models('bogus')

    def boom(key):
        raise RuntimeError('provider offline')

    monkeypatch.setattr(
        models.QwenInference, 'list_available_models', staticmethod(boom)
    )
    with caplog.at_level('WARNING'):
        assert list_available_models('qwen', 'k') == MODEL_BACKENDS['qwen']['models']
    assert 'Could not list' in caplog.text


def test_analyze_missing_raster_and_unreadable_image_paths(patched, tmp_path):
    from shapely.geometry import box

    wf = AssetAnalysisWorkflow()
    with pytest.raises(FileNotFoundError):
        wf.analyze(
            _collection('tree'),
            InferenceSettings(
                raster_path=tmp_path / 'missing.tiff',
                output_dir=tmp_path,
                prompt='p',
                backend='gemma4',
            ),
        )

    raster = tmp_path / 'scene.tiff'
    raster.write_bytes(b'tiff')
    collection = PhysicalAssetCollection()
    asset = PhysicalAsset(id='odd', geometry=box(0, 0, 1, 1))
    odd_image = ImageAsset(
        id='odd_img', path=tmp_path / 'odd.jpg', allow_missing_file=True
    )
    odd_image.path = None  # simulate an image record without a usable path
    asset.add_image_assets(odd_image)
    collection.add(asset)
    result = wf.analyze(
        collection,
        InferenceSettings(
            raster_path=raster, output_dir=tmp_path, prompt='p', backend='gemma4'
        ),
    )
    # The image without a path is filtered out safely; the crop extractor's
    # image is the only one analyzed:
    assert result.n_analyzed == 1
