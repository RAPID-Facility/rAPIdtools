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


# ------------------------------------------------------- new imagery sources
class FakeStreetView:
    created: list = []

    def __init__(self, save_directory, **kwargs):
        self.save_directory = Path(save_directory)
        self.kwargs = kwargs
        FakeStreetView.created.append(self)

    def __call__(self, collection):
        self.save_directory.mkdir(parents=True, exist_ok=True)
        for asset in collection:
            asset.add_image_assets(
                ImageAsset(
                    id=f'{asset.id}_gsv',
                    path=self.save_directory / f'{asset.id}.jpg',
                    allow_missing_file=True,
                )
            )
        return collection


class FakeMapillaryImages(FakeStreetView):
    created: list = []

    def __init__(self, access_token, save_directory, **kwargs):
        super().__init__(save_directory, access_token=access_token, **kwargs)
        FakeMapillaryImages.created.append(self)


class FakeObjectCrops(FakeStreetView):
    created: list = []

    def __init__(self, save_directory, **kwargs):
        super().__init__(save_directory, **kwargs)
        FakeObjectCrops.created.append(self)


class FakeStreetDetector:
    created: list = []
    result: PhysicalAssetCollection | None = None

    def __init__(self, classes, **kwargs):
        self.classes = classes
        self.kwargs = kwargs
        FakeStreetDetector.created.append(self)

    def __call__(self, source=None):
        if FakeStreetDetector.result is not None:
            return FakeStreetDetector.result
        col = _collection('vehicle', 3)
        col.set_attribute('localization', 'triangulated', overwrite=True)
        return col


class FakeGoogleOrtho(FakeBing):
    created: list = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        FakeGoogleOrtho.created.append(self)

    def __call__(self, region, output_path):
        Path(output_path).write_bytes(b'google')
        return Path(output_path)


class FakeRegionBox:
    """BoundingBox stand-in that remembers how it was built."""

    def __init__(self, *bounds):
        self.bounds = tuple(float(b) for b in bounds)

    @classmethod
    def from_raster(cls, path):
        return cls(-1, -1, 1, 1)

    @classmethod
    def from_geojson(cls, path):
        return cls(-118.15, 34.18, -118.14, 34.19)


@pytest.fixture
def patched_sources(patched, monkeypatch):
    for fake in (
        FakeStreetView,
        FakeMapillaryImages,
        FakeObjectCrops,
        FakeStreetDetector,
        FakeGoogleOrtho,
    ):
        fake.created.clear()
    FakeStreetDetector.result = None
    monkeypatch.setattr(rapidtools, 'GoogleStreetViewImageExtractor', FakeStreetView)
    monkeypatch.setattr(rapidtools, 'MapillaryImageExtractor', FakeMapillaryImages)
    monkeypatch.setattr(rapidtools, 'MapillaryObjectImageExtractor', FakeObjectCrops)
    monkeypatch.setattr(rapidtools, 'MapillaryFeatureExtractor', FakeStreetDetector)
    monkeypatch.setattr(rapidtools, 'GoogleOrthomosaicExtractor', FakeGoogleOrtho)
    monkeypatch.setattr(rapidtools, 'BoundingBox', FakeRegionBox)


def test_detection_settings_basemap_options(tmp_path):
    from rapidtools.gui.workflow import DetectionSettings

    legacy = DetectionSettings(
        raster_path=tmp_path,
        output_dir=tmp_path,
        assets=['a'],
        detect_in_recon_imagery=True,
    )
    assert legacy.basemap == 'recon'
    google = DetectionSettings(
        raster_path=tmp_path, output_dir=tmp_path, assets=['a'], basemap='Google'
    )
    assert google.basemap == 'google' and google.detect_in_recon_imagery is False
    with pytest.raises(ValueError, match='basemap must be one of'):
        DetectionSettings(
            raster_path=tmp_path, output_dir=tmp_path, assets=['a'], basemap='esri'
        )


def test_detect_on_google_basemap_and_instances(patched_sources, raster, tmp_path):
    from rapidtools.gui.workflow import DetectionSettings

    wf = AssetAnalysisWorkflow()
    result = wf.detect(
        DetectionSettings(
            raster_path=raster,
            output_dir=tmp_path / 'out',
            assets=['vehicle'],
            basemap='google',
            merge_overlaps=False,
            regularize=False,
        )
    )
    assert result.basemap == 'google'
    assert result.detection_raster.name == 'scene_google.tiff'
    assert len(FakeGoogleOrtho.created) == 1
    assert FakeSAM3.instances[0].kwargs['merge_overlaps'] is False
    # A second run reuses the stitched file:
    wf.detect(
        DetectionSettings(
            raster_path=raster,
            output_dir=tmp_path / 'out',
            assets=['vehicle'],
            basemap='google',
        )
    )
    assert len(FakeGoogleOrtho.created) == 1


def test_inference_settings_imagery_validation(tmp_path):
    from rapidtools.gui.workflow import InferenceSettings

    common = {
        'raster_path': tmp_path,
        'output_dir': tmp_path,
        'prompt': 'p',
        'backend': 'gemma4',
    }
    with pytest.raises(ValueError, match='imagery must be one of'):
        InferenceSettings(imagery='drone', **common)
    with pytest.raises(ValueError, match='Mapillary access token is required'):
        InferenceSettings(imagery='mapillary', **common)
    with pytest.raises(ValueError, match='Mapillary access token is required'):
        InferenceSettings(imagery='mapillary_objects', **common)
    with pytest.raises(ValueError, match='between 0 and 1'):
        InferenceSettings(min_footprint_coverage=1.5, **common)
    with pytest.raises(ValueError, match='search radius'):
        InferenceSettings(
            imagery='google_streetview', street_search_radius_m=0, **common
        )
    with pytest.raises(ValueError, match='At least one image'):
        InferenceSettings(street_max_images=0, **common)
    with pytest.raises(ValueError, match='street_vertical_crop'):
        InferenceSettings(street_vertical_crop=(0.9, 0.2), **common)
    ok = InferenceSettings(
        imagery='Mapillary',
        mapillary_token='MLY|x',
        street_vertical_crop=('0.1', 0.8),
        **common,
    )
    assert ok.imagery == 'mapillary' and ok.street_vertical_crop == (0.1, 0.8)


@pytest.mark.parametrize(
    ('imagery', 'fake', 'subdir'),
    [
        ('google_streetview', FakeStreetView, 'streetview'),
        ('mapillary', FakeMapillaryImages, 'mapillary'),
        ('mapillary_objects', FakeObjectCrops, 'street_objects'),
    ],
)
def test_analyze_with_street_level_imagery(
    patched_sources, raster, tmp_path, imagery, fake, subdir
):
    from rapidtools.gui.workflow import InferenceSettings

    wf = AssetAnalysisWorkflow()
    result = wf.analyze(
        _collection('b', 2),
        InferenceSettings(
            raster_path=tmp_path / 'missing.tiff',  # street sources do not need it
            output_dir=tmp_path / 'out',
            prompt='Describe.',
            backend='gemma4',
            imagery=imagery,
            mapillary_token='MLY|x',
            mapillary_max_images=3,
            street_max_images=2,
            street_vertical_crop=(0.2, 0.9),
            outline_shape='corners',
            json_mode=True,
        ),
    )
    assert result.n_analyzed == 2
    [extractor] = fake.created
    assert (
        extractor.save_directory
        == (tmp_path / 'out' / 'inference_imagery' / subdir).resolve()
    )
    if imagery == 'google_streetview':
        assert extractor.kwargs['max_images_per_asset'] == 2
        assert extractor.kwargs['vertical_crop'] == (0.2, 0.9)
    else:
        assert extractor.kwargs['access_token'] == 'MLY|x'
        assert extractor.kwargs['max_images_per_asset'] == 3
    if imagery == 'mapillary_objects':
        assert extractor.kwargs['outline_shape'] == 'corners'
    else:
        assert 'outline_shape' not in extractor.kwargs
    assert FakeAnalyzer.created[-1].kwargs['generation'].json_mode is True


def test_analyze_aerial_passes_coverage_options(patched_sources, raster, tmp_path):
    from rapidtools.gui.workflow import InferenceSettings

    wf = AssetAnalysisWorkflow()
    wf.analyze(
        _collection('b', 1),
        InferenceSettings(
            raster_path=raster,
            output_dir=tmp_path / 'out',
            prompt='Describe.',
            backend='gemma4',
            min_footprint_coverage=0.6,
            pad_edges=False,
        ),
    )
    assert FakeAnalyzer.created[-1].kwargs['generation'].json_mode is False
    # The aerial extractor received the edge-handling options:
    assert FakeSAM3.instances == []  # detection was not involved


def test_region_imagery_settings_validation(tmp_path):
    from rapidtools.gui.workflow import RegionImagerySettings

    with pytest.raises(ValueError, match='provider must be one of'):
        RegionImagerySettings(output_dir=tmp_path, provider='esri')
    with pytest.raises(ValueError, match='zoom must be between'):
        RegionImagerySettings(output_dir=tmp_path, zoom=25, geojson_path=tmp_path)
    with pytest.raises(ValueError, match='Give the region'):
        RegionImagerySettings(output_dir=tmp_path, min_lon=1)
    with pytest.raises(ValueError, match='min < max'):
        RegionImagerySettings(
            output_dir=tmp_path, min_lon=2, min_lat=0, max_lon=1, max_lat=1
        )
    ok = RegionImagerySettings(
        output_dir=tmp_path,
        provider='google',
        zoom='18',
        min_lon='-118.15',
        min_lat='34.18',
        max_lon='-118.14',
        max_lat='34.19',
    )
    assert ok.zoom == 18 and ok.min_lon == -118.15


def test_download_basemap_from_bounds_and_geojson(
    patched_sources, tmp_path, monkeypatch
):
    from rapidtools.gui.workflow import RegionImagerySettings

    messages = []
    wf = AssetAnalysisWorkflow(progress_callback=messages.append)
    path = wf.download_basemap(
        RegionImagerySettings(
            output_dir=tmp_path / 'out',
            provider='google',
            zoom=17,
            min_lon=-118.15,
            min_lat=34.18,
            max_lon=-118.14,
            max_lat=34.19,
        )
    )
    assert path.is_file() and path.read_bytes() == b'google'
    assert path.name.startswith('google_z17_m118p15000_34p18000')
    assert FakeGoogleOrtho.created[0].kwargs['zoom_level'] == 17
    assert any('tiles at zoom 17' in m for m in messages)
    # Existing files are reused:
    assert (
        wf.download_basemap(
            RegionImagerySettings(
                output_dir=tmp_path / 'out',
                provider='google',
                zoom=17,
                min_lon=-118.15,
                min_lat=34.18,
                max_lon=-118.14,
                max_lat=34.19,
            )
        )
        == path
    )
    assert len(FakeGoogleOrtho.created) == 1

    geojson = tmp_path / 'area.geojson'
    geojson.write_text('{}')
    bing = wf.download_basemap(
        RegionImagerySettings(
            output_dir=tmp_path / 'out', zoom=16, geojson_path=geojson
        )
    )
    assert bing.read_bytes() == b'bing'
    with pytest.raises(FileNotFoundError):
        wf.download_basemap(
            RegionImagerySettings(
                output_dir=tmp_path / 'out',
                zoom=16,
                geojson_path=tmp_path / 'nope.geojson',
            )
        )
    # Too many tiles are refused before any download:
    with pytest.raises(ValueError, match='Lower the zoom level'):
        wf.download_basemap(
            RegionImagerySettings(
                output_dir=tmp_path / 'out',
                zoom=19,
                min_lon=-118.5,
                min_lat=34.0,
                max_lon=-118.0,
                max_lat=34.5,
            )
        )


def test_estimate_tile_count():
    from rapidtools.gui.workflow import estimate_tile_count

    assert estimate_tile_count((-118.15, 34.18, -118.14, 34.19), 19) == 288
    assert estimate_tile_count((0, 0, 0, 0), 10) == 1
    assert estimate_tile_count((-180, -85, 180, 85), 10) > 1_000_000


def test_street_detection_settings_validation(tmp_path):
    from rapidtools.gui.workflow import StreetDetectionSettings

    with pytest.raises(ValueError, match='at least one object class'):
        StreetDetectionSettings(
            output_dir=tmp_path, classes=[], access_token='t', region=1
        )
    with pytest.raises(ValueError, match='access token'):
        StreetDetectionSettings(
            output_dir=tmp_path, classes=['cars'], access_token=' ', region=1
        )
    with pytest.raises(ValueError, match='Load imagery'):
        StreetDetectionSettings(output_dir=tmp_path, classes=['cars'], access_token='t')
    with pytest.raises(ValueError, match='detection_source'):
        StreetDetectionSettings(
            output_dir=tmp_path,
            classes=['cars'],
            access_token='t',
            region=1,
            detection_source='yolo',
        )
    with pytest.raises(ValueError, match='min_observations'):
        StreetDetectionSettings(
            output_dir=tmp_path,
            classes=['cars'],
            access_token='t',
            region=1,
            min_observations=0,
        )
    with pytest.raises(ValueError, match='Camera height'):
        StreetDetectionSettings(
            output_dir=tmp_path,
            classes=['cars'],
            access_token='t',
            region=1,
            camera_height_m=0,
        )
    with pytest.raises(ValueError, match='Frame spacing'):
        StreetDetectionSettings(
            output_dir=tmp_path,
            classes=['cars'],
            access_token='t',
            region=1,
            cluster_radius_m=0,
        )
    ok = StreetDetectionSettings(
        output_dir=tmp_path, classes='cars, poles;cars', access_token='t', region=1
    )
    assert ok.classes == ['cars', 'poles']


def test_discover_street_uses_raster_extent(patched_sources, raster, tmp_path):
    from rapidtools.gui.workflow import StreetDetectionSettings

    messages = []
    wf = AssetAnalysisWorkflow(progress_callback=messages.append)
    result = wf.discover_street(
        StreetDetectionSettings(
            output_dir=tmp_path / 'out',
            classes=['vehicles', 'poles'],
            access_token='MLY|x',
            raster_path=raster,
            start_date='2025-01-01',
            frame_spacing_m=5,
        )
    )
    assert len(result.collection) == 3
    assert (
        result.geojson_path == (tmp_path / 'out' / 'street_objects.geojson').resolve()
    )
    assert result.geojson_path.is_file()
    [detector] = FakeStreetDetector.created
    assert detector.classes == ['vehicles', 'poles']
    assert detector.kwargs['region'].bounds == (-1, -1, 1, 1)
    assert detector.kwargs['start_date'] == '2025-01-01'
    assert detector.kwargs['frame_spacing_m'] == 5
    assert detector.kwargs['cancel_event'] is wf._cancel_event
    assert any('vehicles, poles' in m for m in messages)

    with pytest.raises(FileNotFoundError):
        wf.discover_street(
            StreetDetectionSettings(
                output_dir=tmp_path / 'out',
                classes=['vehicles'],
                access_token='t',
                raster_path=tmp_path / 'missing.tiff',
            )
        )


def test_discover_street_empty_and_cancelled(patched_sources, raster, tmp_path):
    from rapidtools.gui.workflow import StreetDetectionSettings

    messages = []
    wf = AssetAnalysisWorkflow(progress_callback=messages.append)
    FakeStreetDetector.result = PhysicalAssetCollection()
    result = wf.discover_street(
        StreetDetectionSettings(
            output_dir=tmp_path / 'out',
            classes=['vehicles'],
            access_token='t',
            raster_path=raster,
        )
    )
    assert len(result.collection) == 0 and result.geojson_path is None
    assert messages[-1].endswith('no objects were found.')

    class Cancelling(FakeStreetDetector):
        def __call__(self, source=None):
            raise rapidtools.OperationCancelled('stop')

    import rapidtools as rt

    rt.MapillaryFeatureExtractor = Cancelling
    with pytest.raises(WorkflowCancelled):
        wf.discover_street(
            StreetDetectionSettings(
                output_dir=tmp_path / 'out',
                classes=['vehicles'],
                access_token='t',
                raster_path=raster,
            )
        )


# ---------------------------------------------------------------- assistant
class TextModel:
    """Model whose run_inference returns canned JSON for the assistant."""

    def __init__(self, provider, **kwargs):
        self.provider = provider
        self.kwargs = kwargs
        self.model_id = kwargs.get('model_id')
        self.prompts: list[str] = []

    def run_inference(self, image_inputs, prompt, **kwargs):
        from rapidtools.models.base import ModelOutput

        self.prompts.append(prompt)
        assert image_inputs == []
        if 'Review the prompt' in prompt:
            return ModelOutput(text='- Fine.')
        return ModelOutput(text='{"asset": "pole", "fields": [{"name": "State"}]}')


def test_assist_settings_validation():
    from rapidtools.gui.workflow import AssistSettings

    with pytest.raises(ValueError, match='action must be one of'):
        AssistSettings(action='translate')
    with pytest.raises(ValueError, match='Unknown assistant backend'):
        AssistSettings(action='draft', backend='nope')
    with pytest.raises(ValueError, match='required to use Google Gemini'):
        AssistSettings(action='draft', backend='gemini')
    with pytest.raises(ValueError, match='JSON object'):
        AssistSettings(action='draft', backend='gemma4', spec=['x'])
    ok = AssistSettings(action='Draft', backend='gemma4')
    assert ok.action == 'draft' and ok.model_id == ok.backend_spec['default_model']


def test_assist_caches_and_releases_model(
    monkeypatch, patched_sources, raster, tmp_path
):
    import rapidtools.models as models_module
    from rapidtools.gui.workflow import AssistSettings, InferenceSettings

    created: list[TextModel] = []

    def fake_load(provider, **kw):
        model = TextModel(provider, **kw)
        created.append(model)
        return model

    monkeypatch.setattr(models_module, 'load', fake_load)
    messages = []
    wf = AssetAnalysisWorkflow(progress_callback=messages.append)

    out = wf.assist(AssistSettings(action='draft', backend='gemma4', brief='poles'))
    assert out['spec']['asset'] == 'pole'
    assert created[0].kwargs == {
        'model_id': 'google/gemma-4-E2B-it',
        'temperature': 0.2,
        'max_tokens': 4096,
    }
    assert any('Loading google/gemma-4-E2B-it' in m for m in messages)

    # Same backend and model: the wrapper is reused.
    out = wf.assist(AssistSettings(action='review', backend='gemma4', prompt_text='x'))
    assert out == {'text': '- Fine.'} and len(created) == 1

    # A hosted backend gets its own wrapper with the key and one worker.
    wf.assist(AssistSettings(action='draft', backend='gemini', api_key='k', brief='b'))
    assert created[1].kwargs['api_key'] == 'k' and created[1].kwargs['max_workers'] == 1
    assert wf._assistant[0][0] == 'gemini'

    # Inference does not evict a hosted assistant, but does evict a local one.
    wf.analyze(
        _collection('b', 1),
        InferenceSettings(
            raster_path=raster, output_dir=tmp_path, prompt='p', backend='gemma4'
        ),
    )
    assert wf._assistant is not None
    wf.assist(AssistSettings(action='draft', backend='gemma4', brief='poles'))
    assert wf._assistant[0][0] == 'gemma4'
    wf.analyze(
        _collection('b', 1),
        InferenceSettings(
            raster_path=raster, output_dir=tmp_path, prompt='p', backend='gemma4'
        ),
    )
    assert wf._assistant is None
    wf.release_assistant()  # idempotent
