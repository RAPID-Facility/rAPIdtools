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
# 09-22-2026

"""Tests for the harmonized API: model factory, configs, stages, AssetAnalyzer."""

import logging
import re
import threading
import warnings
from pathlib import Path

import pytest
from shapely.geometry import box

import rapidtools
from rapidtools.core import (
    ImageAsset,
    OperationCancelled,
    PhysicalAsset,
    PhysicalAssetCollection,
)
from rapidtools.models import (
    MODEL_REGISTRY as REGISTRY,
)
from rapidtools.models import (
    PROVIDERS,
    GenerationConfig,
    ModelInfo,
    SegmentationConfig,
    catalog,
    get_model_class,
    list_providers,
    load,
)
from rapidtools.models.base import BaseInferenceModel, ModelOutput
from rapidtools.processing import (
    AssetAnalyzer,
    Pipeline,
    PipelineStep,
    RateLimitPolicy,
    Stage,
    image_analyzers,
)
from rapidtools.processing.step import stage_of


class FakeModel(BaseInferenceModel):
    """Minimal model recording calls; optionally batch-capable."""

    def __init__(self, info=None, text='Level: 2', batch=False, fail=False):
        self.INFO = info
        self.model_id = 'fake/model'
        self.text = text
        self.calls: list = []
        self.fail = fail
        if batch:
            self.run_inference_batch = self._batch  # type: ignore[method-assign]

    def run_inference(self, image_inputs, prompt, **kwargs):
        self.calls.append(('single', list(image_inputs), prompt, kwargs))
        return None if self.fail else ModelOutput(text=self.text)

    def _batch(self, batch_images, batch_prompts, **kwargs):
        self.calls.append(('batch', [list(b) for b in batch_images], kwargs))
        return [ModelOutput(text=self.text) if imgs else None for imgs in batch_images]


def _collection(tmp_path: Path, n: int = 2) -> PhysicalAssetCollection:
    col = PhysicalAssetCollection()
    for i in range(n):
        asset = PhysicalAsset(id=f'a{i}', geometry=box(i, 0, i + 1, 1))
        img = tmp_path / f'a{i}.jpg'
        img.write_bytes(b'\xff\xd8\xff\xd9')
        asset.add_image_assets(ImageAsset(id=f'a{i}_img', path=img))
        col.add(asset)
    return col


# ==========================================
# 1. Model registry and factory
# ==========================================


def test_provider_registry_is_consistent():
    """Every provider has a class, catalogue, default and unique prefix."""
    assert list_providers() == list(PROVIDERS)
    assert list_providers('api') == ['gemini', 'claude', 'openai', 'muse_spark', 'qwen']
    assert list_providers('segmentation') == ['sam3']
    prefixes = [info.attribute_prefix for info in PROVIDERS.values()]
    assert len(prefixes) == len(set(prefixes))
    for key, info in PROVIDERS.items():
        assert isinstance(info, ModelInfo) and info.key == key
        cls = get_model_class(key)
        assert cls.__name__ == info.class_name and cls.INFO is info
        assert info.default_model in info.model_ids
        assert info.model_ids[0] == (info.analyzer_default_model or info.default_model)
        assert catalog(key) == info.catalog
    assert PROVIDERS['gemini'].is_api and not PROVIDERS['gemma4'].is_api


def test_lazy_registry_and_factory(monkeypatch, requests_mock):
    """MODEL_REGISTRY behaves like a mapping and load() builds wrappers."""
    assert len(REGISTRY) == len(PROVIDERS) and set(REGISTRY) == set(PROVIDERS)
    assert 'gemini' in repr(REGISTRY)
    with pytest.raises(KeyError, match='Unknown model provider'):
        get_model_class('nope')
    requests_mock.get('https://api.openai.com/v1/models', json={'data': []})
    model = load('openai', api_key='sk-test', model_id='gpt-5.5')
    assert type(model).__name__ == 'OpenAIInference' and model.model_id == 'gpt-5.5'
    assert model.INFO.kind == 'api'


def test_models_module_lazy_attribute_errors():
    """Unknown attributes on rapidtools.models raise AttributeError."""
    import rapidtools.models as models

    with pytest.raises(AttributeError):
        _ = models.NoSuchInference
    assert 'GeminiInference' in dir(models)


# ==========================================
# 2. Generation and segmentation configs
# ==========================================


def test_generation_config_merging():
    """Instance defaults < config < explicit overrides; json_mode is sticky."""
    model = FakeModel()
    model.temperature, model.max_tokens, model.system_instruction = 0.4, 100, 'sys'
    resolved = model._resolve_generation(
        GenerationConfig(max_tokens=64, json_mode=True), temperature=0.0
    )
    assert (resolved.temperature, resolved.max_tokens) == (0.0, 64)
    assert resolved.json_mode is True and resolved.system_instruction == 'sys'
    # An explicit json_mode=False never disables a config flag:
    assert model._resolve_generation(
        GenerationConfig(json_mode=True), json_mode=False
    ).json_mode
    assert model._resolve_generation(None, json_mode=True).json_mode is True
    with pytest.raises(TypeError, match='Unknown generation option'):
        GenerationConfig().merged(nonsense=1)
    assert SegmentationConfig().threshold == 0.5


def test_generation_config_reaches_api_payloads(requests_mock, tmp_path):
    """config= is honoured by the hosted wrappers (temperature, tokens, system)."""
    from rapidtools.models import ClaudeInference, GeminiInference, OpenAIInference

    img = tmp_path / 'x.png'
    from PIL import Image

    Image.new('RGB', (2, 2)).save(img)
    cfg = GenerationConfig(
        temperature=0.1, max_tokens=9, json_mode=True, system_instruction='S'
    )

    requests_mock.get('https://api.openai.com/v1/models', json={'data': []})
    post = requests_mock.post(
        'https://api.openai.com/v1/chat/completions',
        json={'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}]},
    )
    OpenAIInference(api_key='k', model_id='gpt-4o').run_inference(img, 'p', config=cfg)
    body = post.last_request.json()
    assert body['temperature'] == 0.1 and body['max_completion_tokens'] == 9
    assert body['response_format'] == {'type': 'json_object'}
    assert body['messages'][0] == {'role': 'system', 'content': 'S'}

    requests_mock.get('https://api.anthropic.com/v1/models', json={'data': []})
    post = requests_mock.post(
        'https://api.anthropic.com/v1/messages',
        json={'content': [{'type': 'text', 'text': 'ok'}], 'stop_reason': 'end_turn'},
    )
    ClaudeInference(api_key='k', model_id='claude-haiku-4-5').run_inference(
        img, 'p', config=cfg, max_tokens=3
    )
    body = post.last_request.json()
    assert body['temperature'] == 0.1 and body['max_tokens'] == 3  # explicit wins
    assert body['system'].startswith('S\n\nYou must respond with ONLY valid JSON')

    requests_mock.get(
        'https://generativelanguage.googleapis.com/v1beta/models', json={'models': []}
    )
    post = requests_mock.post(
        'https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent',
        json={
            'candidates': [
                {'content': {'parts': [{'text': 'ok'}]}, 'finishReason': 'STOP'}
            ]
        },
    )
    GeminiInference(api_key='k').run_inference(img, 'p', config=cfg)
    body = post.last_request.json()
    assert body['generationConfig']['responseMimeType'] == 'application/json'
    assert body['generationConfig']['maxOutputTokens'] == 9
    assert body['systemInstruction'] == {'parts': [{'text': 'S'}]}


def test_listing_is_a_classmethod_everywhere():
    """list_available_models can be called on every provider class."""
    for key in list_providers():
        cls = get_model_class(key)
        ids = cls.list_available_models()
        assert isinstance(ids, list) and ids


# ==========================================
# 3. Pipeline stages
# ==========================================


def test_stage_protocol_and_inference(caplog):
    """Declared stages win; legacy names are inferred with a warning."""

    class Declared:
        stage = Stage.EXPORT

        def __call__(self, c):
            return c

    class IntStage:
        stage = 10

        def __call__(self, c):
            return c

    class LegacyPredictor:
        def __call__(self, c):
            return c

    class Mystery:
        def __call__(self, c):
            return c

    assert isinstance(Declared(), PipelineStep)
    assert stage_of(Declared()) is Stage.EXPORT
    assert stage_of(IntStage()) is Stage.DETECT
    with caplog.at_level(logging.WARNING):
        assert stage_of(LegacyPredictor()) is Stage.ANALYZE
    assert 'does not declare a pipeline stage' in caplog.text
    assert stage_of(Mystery(), warn=False) is Stage.CUSTOM

    class BadStage:
        stage = 12345

        def __call__(self, c):
            return c

    assert stage_of(BadStage(), warn=False) is Stage.CUSTOM


def test_pipeline_orders_by_stage_and_keeps_insertion_order():
    """Steps run in stage order; equal stages keep their insertion order."""
    order = []

    def make(name, stage):
        class Step:
            def __call__(self, c):
                order.append(name)
                return c

        Step.stage = stage
        Step.__name__ = name
        return Step()

    pipeline = Pipeline(
        [
            make('export', Stage.EXPORT),
            make('analyze_b', Stage.ANALYZE),
            make('detect', Stage.DETECT),
            make('analyze_a', Stage.ANALYZE),
            make('crop', Stage.EXTRACT_IMAGERY),
        ]
    )
    pipeline.run(PhysicalAssetCollection())
    assert order == ['detect', 'crop', 'analyze_b', 'analyze_a', 'export']


def test_components_declare_stages():
    """Shipped components expose the right stage attribute."""
    from rapidtools.processing import (
        AerialImageryExtractor,
        BuildingRegularizer,
        GoogleStreetViewImageExtractor,
        MapillaryImageExtractor,
        RoadwayRegularizer,
        SAM3ImageSegmenter,
        SAM3OrthoFeatureExtractor,
    )

    assert AerialImageryExtractor.stage is Stage.EXTRACT_IMAGERY
    assert MapillaryImageExtractor.stage is Stage.EXTRACT_IMAGERY
    assert GoogleStreetViewImageExtractor.stage is Stage.EXTRACT_IMAGERY
    assert SAM3OrthoFeatureExtractor.stage is Stage.DETECT
    assert SAM3ImageSegmenter.stage is Stage.SEGMENT
    assert BuildingRegularizer.stage is RoadwayRegularizer.stage is Stage.REGULARIZE
    assert AssetAnalyzer.stage is Stage.ANALYZE


def test_sam3_feature_extractor_as_pipeline_step(monkeypatch, tmp_path):
    """The detector merges detections into a collection when given a raster."""
    from rapidtools.processing import feature_extractors as fe

    class FakeSAM:
        def __init__(self, **kwargs):
            self.model_id = 'facebook/sam3'

    monkeypatch.setattr(fe, 'SAM3Inference', FakeSAM)
    extractor = fe.SAM3OrthoFeatureExtractor('building', raster_path=tmp_path / 'r.tif')
    detected = PhysicalAssetCollection()
    detected.add(PhysicalAsset(id='new', geometry=box(0, 0, 1, 1)))
    detected.add(PhysicalAsset(id='dup', geometry=box(0, 0, 1, 1)))
    calls = []

    def fake_detect(raster):
        calls.append(Path(raster))
        return detected

    monkeypatch.setattr(extractor, 'detect', fake_detect)

    existing = PhysicalAssetCollection()
    existing.add(PhysicalAsset(id='dup', geometry=box(5, 5, 6, 6), attributes={'k': 1}))
    result = Pipeline([extractor]).run(existing)
    assert sorted(a.id for a in result) == ['dup', 'new']
    assert result.get('dup').attributes == {'k': 1}  # existing asset kept
    assert calls == [tmp_path / 'r.tif']
    # Standalone forms:
    assert extractor() is detected and extractor(tmp_path / 'other.tif') is detected
    assert calls[-1] == tmp_path / 'other.tif'

    bare = fe.SAM3OrthoFeatureExtractor('building')
    with pytest.raises(ValueError, match='raster_path'):
        bare(PhysicalAssetCollection())
    with pytest.raises(ValueError, match='No raster'):
        bare()


def test_roadway_regularizer_output_modes(monkeypatch):
    """output='polygons'/'centerlines' return a single collection."""
    from rapidtools.processing import RoadwayRegularizer

    centerlines, polygons = PhysicalAssetCollection(), PhysicalAssetCollection()
    centerlines.add(PhysicalAsset(id='c', geometry=box(0, 0, 1, 1)))
    polygons.add(PhysicalAsset(id='p', geometry=box(0, 0, 1, 1)))
    monkeypatch.setattr(
        RoadwayRegularizer, 'process', lambda self, assets: (centerlines, polygons)
    )
    assert RoadwayRegularizer()(polygons) == (centerlines, polygons)
    assert RoadwayRegularizer(output='polygons')(polygons) is polygons
    assert RoadwayRegularizer(output='centerlines')(polygons) is centerlines
    with pytest.raises(ValueError, match='output must be'):
        RoadwayRegularizer(output='lines')


# ==========================================
# 4. AssetAnalyzer
# ==========================================


def test_asset_analyzer_infers_mode_and_prefix(tmp_path):
    """Mode and prefix come from the model INFO unless overridden."""
    api = AssetAnalyzer(FakeModel(PROVIDERS['gemini']), prompt='p')
    assert api.mode == 'api' and api.attribute_prefix == 'gemini'
    assert api.max_workers == 5 and api.cooldown_duration == 30.0
    assert api.PROVIDER_NAME == 'Google Gemini'

    local = AssetAnalyzer(FakeModel(PROVIDERS['qwen_vl'], batch=True), prompt='p')
    assert local.mode == 'local' and local.attribute_prefix == 'qwen_vl'
    assert local.max_workers == 1 and local.cooldown_duration == 0

    plain = AssetAnalyzer(FakeModel(), prompt='p', attribute_prefix='mine')
    assert plain.mode == 'api' and plain.attribute_prefix == 'mine'
    assert AssetAnalyzer(FakeModel(), prompt='p').attribute_prefix == 'vlm'
    with pytest.raises(ValueError, match='mode must be'):
        AssetAnalyzer(FakeModel(), prompt='p', mode='cloud')

    prompt_file = tmp_path / 'prompt.txt'
    prompt_file.write_text('from file')
    assert AssetAnalyzer(FakeModel(), prompt=prompt_file).prompt == 'from file'


def test_asset_analyzer_api_mode_forwards_generation(tmp_path):
    """API mode threads requests and forwards the GenerationConfig."""
    model = FakeModel(PROVIDERS['claude'], text='{"damage": 2}')
    cfg = GenerationConfig(json_mode=True)
    analyzer = AssetAnalyzer(model, prompt='Rate', generation=cfg, max_workers=2)
    collection = analyzer(_collection(tmp_path, 3))
    for asset in collection:
        assert asset.attributes['claude_damage'] == 2
        assert asset.attributes['ai_model_used'] == 'fake/model'
    assert all(call[3] == {'config': cfg} for call in model.calls)
    assert len(model.calls) == 3


def test_asset_analyzer_local_batches_and_caps_images(tmp_path):
    """Local mode uses run_inference_batch when present and caps images."""
    model = FakeModel(PROVIDERS['gemma4'], batch=True)
    analyzer = AssetAnalyzer(
        model,
        prompt='p',
        batch_size=2,
        max_images_per_asset=1,
        generation=GenerationConfig(),
    )
    collection = analyzer(_collection(tmp_path, 3))
    assert [c[0] for c in model.calls] == ['batch', 'batch']
    assert model.calls[0][2] == {'config': GenerationConfig()}
    assert all(a.attributes['gemma4_level'] == '2' for a in collection)

    sequential = FakeModel(PROVIDERS['llama'])
    AssetAnalyzer(sequential, prompt='p')(_collection(tmp_path, 2))
    assert [c[0] for c in sequential.calls] == ['single', 'single']


def test_asset_analyzer_rate_limit_policy(tmp_path):
    """Failures grow the cooldown; a zero policy never sleeps."""
    model = FakeModel(PROVIDERS['openai'], fail=True)
    analyzer = AssetAnalyzer(
        model, prompt='p', rate_limit=RateLimitPolicy(0.001, max_asset_retries=0)
    )
    collection = analyzer(_collection(tmp_path, 2))
    assert all(a.attributes == {} for a in collection)
    assert analyzer._consecutive_error_count == 2
    assert len(model.calls) == 2  # no retry passes requested
    assert analyzer._global_cooldown_until > 0
    assert RateLimitPolicy(10).wait_for(3) == 30

    quiet = AssetAnalyzer(
        FakeModel(fail=True), prompt='p', rate_limit=RateLimitPolicy(0)
    )
    quiet(_collection(tmp_path, 1))
    assert quiet._global_cooldown_until == 0.0


def test_asset_analyzer_cancellation_and_empty(tmp_path, caplog):
    """Cancellation raises after the pool drains; empty collections are skipped."""
    stop = threading.Event()
    stop.set()
    analyzer = AssetAnalyzer(
        FakeModel(PROVIDERS['gemini']), prompt='p', cancel_event=stop
    )
    with pytest.raises(OperationCancelled):
        analyzer(_collection(tmp_path, 2))
    local = AssetAnalyzer(FakeModel(PROVIDERS['gemma4']), prompt='p', cancel_event=stop)
    with pytest.raises(OperationCancelled):
        local(_collection(tmp_path, 1))
    with caplog.at_level(logging.WARNING):
        assert (
            len(AssetAnalyzer(FakeModel(), prompt='p')(PhysicalAssetCollection())) == 0
        )
    assert 'No assets with images' in caplog.text


def test_deprecated_provider_analyzers_warn(monkeypatch, tmp_path):
    """Provider-specific analyzers still work but emit DeprecationWarning."""

    class Recording(FakeModel):
        def __init__(self, **kwargs):
            super().__init__(PROVIDERS['gemini'])
            self.kwargs = kwargs

    monkeypatch.setattr(image_analyzers.GeminiAssetAnalyzer, 'MODEL_CLASS', Recording)
    with pytest.warns(DeprecationWarning, match="load\\('gemini'"):
        analyzer = image_analyzers.GeminiAssetAnalyzer(api_key='k', prompt='p')
    assert analyzer.mode == 'api' and analyzer.attribute_prefix == 'gemini'
    assert (
        analyzer.model.kwargs['model_id']
        == image_analyzers.GEMINI_ANALYZER_DEFAULT_MODEL
    )
    assert (
        analyzer(_collection(tmp_path, 1)).get('a0').attributes['gemini_level'] == '2'
    )

    monkeypatch.setattr(
        image_analyzers, 'Gemma4Inference', lambda **kw: FakeModel(PROVIDERS['gemma4'])
    )
    with pytest.warns(DeprecationWarning):
        local = image_analyzers.Gemma4AssetAnalyzer(prompt='p')
    assert local.mode == 'local' and local.max_images_per_asset == 2

    monkeypatch.setattr(
        image_analyzers,
        'LlamaVisionInference',
        lambda **kw: FakeModel(PROVIDERS['llama']),
    )
    with pytest.warns(DeprecationWarning):
        assert image_analyzers.LlamaVisionAssetAnalyzer(prompt='p').batch_size == 1

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        AssetAnalyzer(FakeModel(), prompt='p')  # the new class never warns


# ==========================================
# 5. Package-level ergonomics
# ==========================================


def test_top_level_exports_new_names():
    """The harmonized entry points are reachable from the package root."""
    assert rapidtools.AssetAnalyzer is AssetAnalyzer
    assert rapidtools.Stage is Stage
    assert rapidtools.RateLimitPolicy is RateLimitPolicy
    assert callable(rapidtools.login) and callable(rapidtools.configure_logging)


def test_gui_backends_derive_from_registry():
    """The GUI registry is generated from PROVIDERS (no duplicated model IDs)."""
    from rapidtools.gui.workflow import MODEL_BACKENDS

    assert list(MODEL_BACKENDS) == list_providers('api') + list_providers('local')
    for key, spec in MODEL_BACKENDS.items():
        info = PROVIDERS[key]
        assert spec['models'] == info.model_ids
        assert spec['default_model'] == spec['models'][0]
        if info.is_api:
            assert spec['key_env'] == info.key_env[0] and 'supports_4bit' not in spec
        else:
            assert spec['supports_4bit'] == info.supports_4bit
    assert MODEL_BACKENDS['gemini']['default_model'] == 'gemini-3.5-flash-lite'


# ==========================================
# 6. Keyword aliases
# ==========================================


def test_resolve_alias_helper():
    """Aliases are popped with a warning; unknown keys raise TypeError."""
    from rapidtools.config import resolve_alias

    kwargs = {'output_dir': 'tiles'}
    with pytest.warns(DeprecationWarning, match="'output_dir' is deprecated"):
        assert resolve_alias(kwargs, 'save_directory', 'output_dir') == 'tiles'
    assert kwargs == {}
    assert resolve_alias({}, 'save_directory', 'output_dir', default='x') == 'x'
    with pytest.raises(TypeError, match='Unexpected keyword'):
        resolve_alias({'bogus': 1}, 'save_directory', 'output_dir')


def test_tile_extractors_accept_save_directory_and_output_dir(tmp_path):
    """save_directory is canonical; output_dir still works with a warning."""
    from rapidtools.data_sources import (
        BingAerialImageExtractor,
        GoogleAerialImageExtractor,
    )

    ext = GoogleAerialImageExtractor(save_directory=tmp_path / 'a', zoom_level=12)
    assert ext.save_directory == (tmp_path / 'a').resolve()
    assert ext.output_dir == ext.save_directory
    with pytest.warns(DeprecationWarning):
        legacy = BingAerialImageExtractor(output_dir=tmp_path / 'b')
    assert legacy.save_directory == (tmp_path / 'b').resolve()
    legacy.output_dir = tmp_path / 'c'
    assert legacy.save_directory == (tmp_path / 'c').resolve()
    assert GoogleAerialImageExtractor(tmp_path / 'd').save_directory.name == 'd'
    with pytest.raises(TypeError):
        GoogleAerialImageExtractor(tmp_path, bogus=1)


def test_mapillary_accepts_api_key_and_save_directory(tmp_path):
    """The Mapillary client and extractor take api_key/save_directory synonyms."""
    from rapidtools.data_sources import MapillaryClient
    from rapidtools.processing import MapillaryImageExtractor

    client = MapillaryClient(api_key='tok', save_directory=tmp_path / 'm')
    assert client.access_token == 'tok'
    assert Path(client.save_dir) == tmp_path / 'm'

    extractor = MapillaryImageExtractor(api_key='tok', save_directory=tmp_path / 's')
    assert extractor.client.access_token == 'tok'
    with pytest.raises(ValueError, match='access token'):
        MapillaryImageExtractor(save_directory=tmp_path / 's')


def test_aerial_extractor_buffer_m_alias(tmp_path):
    """buffer_m is translated to the metric buffer string."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    from rapidtools.processing import AerialImageryExtractor

    raster = tmp_path / 'r.tif'
    with rasterio.open(
        raster,
        'w',
        driver='GTiff',
        height=4,
        width=4,
        count=3,
        dtype='uint8',
        crs='EPSG:4326',
        transform=from_bounds(0, 0, 1, 1, 4, 4),
    ) as dst:
        dst.write(np.zeros((3, 4, 4), dtype='uint8'))
    ext = AerialImageryExtractor(raster, save_directory=tmp_path / 'c', buffer_m=15)
    assert ext.buffer_asset == '15 m'
    assert (
        AerialImageryExtractor(raster, save_directory=tmp_path / 'c').buffer_asset
        == '20%'
    )


# ==========================================
# 7. Remaining branches
# ==========================================


def test_registry_getitem_and_local_mode_inference(tmp_path):
    """MODEL_REGISTRY[key] imports the class; local base models infer 'local'."""
    from rapidtools.models.local_base import BaseLocalInferenceModel

    assert REGISTRY['gemini'] is get_model_class('gemini')

    class LocalNoInfo(BaseLocalInferenceModel):
        model_id = 'local/x'

        def run_inference(self, image_inputs, prompt, **kwargs):
            return ModelOutput(text='ok')

    analyzer = AssetAnalyzer(LocalNoInfo(), prompt='p')
    assert analyzer.mode == 'local' and analyzer.attribute_prefix == 'vlm'
    assert (
        analyzer(_collection(tmp_path, 1)).get('a0').attributes['vlm_analysis_raw']
        == 'ok'
    )


def test_inference_settings_backend_spec(tmp_path):
    """backend_spec exposes the registry record of the chosen backend."""
    from rapidtools.gui.workflow import MODEL_BACKENDS, InferenceSettings

    settings = InferenceSettings(
        raster_path=tmp_path, output_dir=tmp_path, prompt='p', backend='gemma4'
    )
    assert settings.backend_spec is MODEL_BACKENDS['gemma4']


def test_openai_compat_skips_validation_without_model(requests_mock):
    """A provider whose default model is empty skips the model check."""
    from rapidtools.models.openai_compat import BaseOpenAICompatibleInference

    class NoDefault(BaseOpenAICompatibleInference):
        PROVIDER_NAME = 'NoDefault'
        API_KEY_ENV = ('NODEFAULT_KEY',)
        DEFAULT_MODEL = ''
        MODEL_CATALOG = []

    model = NoDefault(api_key='k')
    assert model.model_id == '' and requests_mock.call_count == 0


# ==========================================
# 8. Prompt path handling
# ==========================================


def test_prompt_paths_resolve_eagerly_and_missing_files_raise(tmp_path, monkeypatch):
    """String or Path prompts pointing at files are read; typos raise."""
    from rapidtools.models.base import looks_like_prompt_file, resolve_prompt_text

    prompt_file = tmp_path / 'chs.txt'
    prompt_file.write_text('\nTASK: grade it.\n')
    assert resolve_prompt_text(prompt_file) == 'TASK: grade it.'
    assert resolve_prompt_text(str(prompt_file)) == 'TASK: grade it.'
    assert resolve_prompt_text('Rate the damage 0-5.') == 'Rate the damage 0-5.'
    assert looks_like_prompt_file('prompts/aerial_CHS_prompts.txt')
    assert not looks_like_prompt_file('Rate the damage.')
    assert not looks_like_prompt_file('x' * 300 + '.txt')
    with pytest.raises(FileNotFoundError, match='resolved against'):
        resolve_prompt_text(tmp_path / 'missing.txt')
    with pytest.raises(FileNotFoundError, match='Pass the prompt text'):
        resolve_prompt_text('missing_prompt.txt')

    # Relative paths resolve against the working directory:
    monkeypatch.chdir(tmp_path)
    analyzer = AssetAnalyzer(FakeModel(), prompt='chs.txt')
    assert analyzer.prompt == 'TASK: grade it.'
    monkeypatch.chdir(tmp_path.parent)
    with pytest.raises(FileNotFoundError):
        AssetAnalyzer(FakeModel(), prompt='chs.txt')


def test_model_resolve_prompt_warns_but_continues(caplog):
    """Wrappers keep accepting raw strings but warn when one looks like a path."""
    model = FakeModel()
    with caplog.at_level(logging.WARNING):
        assert model._resolve_prompt('nope.txt') == 'nope.txt'
    assert 'Prompt file not found' in caplog.text
    assert model._resolve_prompt('Describe it.') == 'Describe it.'


# ==========================================
# 9. Retry passes, timeouts, single-print logging
# ==========================================


def test_asset_analyzer_retries_failed_assets(tmp_path, caplog):
    """Assets whose request failed are retried in later passes."""

    class FlakyModel(FakeModel):
        def __init__(self):
            super().__init__(PROVIDERS['gemini'])
            self.seen: dict[str, int] = {}

        def run_inference(self, image_inputs, prompt, **kwargs):
            key = str(image_inputs[0])
            self.seen[key] = self.seen.get(key, 0) + 1
            self.calls.append(key)
            # Every asset fails twice, then succeeds.
            return ModelOutput(text='Level: 5') if self.seen[key] >= 3 else None

    model = FlakyModel()
    analyzer = AssetAnalyzer(
        model,
        prompt='p',
        rate_limit=RateLimitPolicy(cooldown_seconds=0, max_asset_retries=2),
    )
    with caplog.at_level(logging.INFO):
        collection = analyzer(_collection(tmp_path, 3))
    assert all(a.attributes['gemini_level'] == '5' for a in collection)
    assert len(model.calls) == 9  # 3 assets x 3 attempts
    assert 'Retrying 3 assets that failed (pass 1 of 2)' in caplog.text
    assert 'All assets processed successfully' in caplog.text

    # With only one retry pass, the assets remain failed and are listed.
    model = FlakyModel()
    analyzer = AssetAnalyzer(
        model,
        prompt='p',
        rate_limit=RateLimitPolicy(cooldown_seconds=0, max_asset_retries=1),
    )
    with caplog.at_level(logging.ERROR):
        collection = analyzer(_collection(tmp_path, 2))
    assert all(a.attributes == {} for a in collection)
    assert 'still failed after all retries' in caplog.text


def test_asset_analyzer_skips_assets_without_images_before_retries(tmp_path):
    """Assets with no downloaded images are not counted as request failures."""
    from shapely.geometry import box

    model = FakeModel(PROVIDERS['gemini'])
    collection = _collection(tmp_path, 1)
    bare = PhysicalAsset(id='bare', geometry=box(5, 5, 6, 6))
    bare.add_image_assets(
        ImageAsset(id='missing', path=tmp_path / 'missing.jpg', allow_missing_file=True)
    )
    collection.add(bare)
    AssetAnalyzer(model, prompt='p', rate_limit=RateLimitPolicy(0))(collection)
    assert len(model.calls) == 1
    assert collection.get('bare').attributes == {}


def test_api_wrappers_accept_a_request_timeout(requests_mock, tmp_path):
    """The per-request timeout is configurable and defaults to 60 seconds."""
    from PIL import Image

    from rapidtools.models import (
        ClaudeInference,
        GeminiInference,
        OpenAIInference,
        QwenInference,
    )
    from rapidtools.models.api_base import API_REQUEST_TIMEOUT

    img = tmp_path / 'x.png'
    Image.new('RGB', (2, 2)).save(img)
    requests_mock.get(re.compile(r'.*'), json={'data': [], 'models': []})
    requests_mock.post(
        re.compile(r'.*'),
        json={
            'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}],
            'content': [{'type': 'text', 'text': 'ok'}],
            'candidates': [{'content': {'parts': [{'text': 'ok'}]}}],
        },
    )
    assert API_REQUEST_TIMEOUT == 60.0
    GeminiInference(api_key='k').run_inference(img, 'p')
    assert requests_mock.last_request.timeout == 60.0
    GeminiInference(api_key='k', timeout=120).run_inference(img, 'p')
    assert requests_mock.last_request.timeout == 120
    ClaudeInference(api_key='k', timeout=15).run_inference(img, 'p')
    assert requests_mock.last_request.timeout == 15
    OpenAIInference(api_key='k', timeout=45).run_inference(img, 'p')
    assert requests_mock.last_request.timeout == 45
    QwenInference(api_key='k', timeout=5).run_inference(img, 'p')
    assert requests_mock.last_request.timeout == 5


def test_configure_logging_prints_each_record_once():
    """With configure_logging active, records do not also reach root handlers."""
    import io

    root = logging.getLogger()
    root_stream = io.StringIO()
    root_handler = logging.StreamHandler(root_stream)
    root.addHandler(root_handler)
    package_logger = logging.getLogger('rapidtools')
    try:
        stream = io.StringIO()
        rapidtools.configure_logging('INFO', stream=stream)
        logging.getLogger('rapidtools.demo').info('only once')
        assert stream.getvalue().count('only once') == 1
        assert 'only once' not in root_stream.getvalue()
        assert package_logger.propagate is False
    finally:
        root.removeHandler(root_handler)
        package_logger.propagate = True
