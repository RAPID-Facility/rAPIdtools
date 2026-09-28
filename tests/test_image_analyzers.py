"""Tests for the API, local, and family-specific asset analyzers."""

from pathlib import Path

import pytest
from shapely.geometry import box

from rapidtools.core import ImageAsset, PhysicalAsset, PhysicalAssetCollection
from rapidtools.models.base import ModelOutput
from rapidtools.processing import image_analyzers as ia


class FakeModel:
    """Stands in for any rapidtools inference model."""

    created: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model_id = kwargs.get('model_id', 'fake-default')
        self.calls = []
        FakeModel.created.append(self)

    def run_inference(self, image_inputs, prompt, **kwargs):
        self.calls.append((list(image_inputs), prompt))
        if any('bad' in str(p) for p in image_inputs):
            return None
        return ModelOutput(text='CHS Level: 3\nJustification: roof gone')


def _collection(tmp_path: Path) -> PhysicalAssetCollection:
    col = PhysicalAssetCollection()
    for name in ('good', 'bad', 'noimage'):
        asset = PhysicalAsset(id=name, geometry=box(0, 0, 1, 1))
        if name != 'noimage':
            img = tmp_path / f'{name}.jpg'
            img.write_bytes(b'\xff\xd8\xff\xd9')
            asset.add_image_assets(ImageAsset(id=f'{name}_img', path=img))
        col.add(asset)
    return col


@pytest.fixture(autouse=True)
def _reset_fake():
    FakeModel.created.clear()


def test_parse_structured_results_json_and_lines():
    assert ia.parse_structured_results('```json\n{"a": 1}\n```') == {'a': 1}
    assert ia.parse_structured_results('{"level": "2"}') == {'level': '2'}
    assert ia.parse_structured_results('CHS Level: 3\nNote - fine\nnoise') == {
        'chs_level': '3',
        'note': 'fine',
    }
    assert ia.parse_structured_results('no structure here') == {}


@pytest.mark.parametrize(
    ('analyzer_class', 'prefix', 'default_model'),
    [
        (ia.GeminiAssetAnalyzer, 'gemini', ia.GEMINI_ANALYZER_DEFAULT_MODEL),
        (ia.ClaudeAssetAnalyzer, 'claude', ia.ANTHROPIC_DEFAULT_MODEL),
        (ia.OpenAIAssetAnalyzer, 'openai', ia.OPENAI_DEFAULT_MODEL),
        (ia.MuseSparkAssetAnalyzer, 'muse_spark', ia.MUSE_SPARK_DEFAULT_MODEL),
        (ia.QwenAssetAnalyzer, 'qwen', ia.QWEN_DEFAULT_MODEL),
    ],
)
def test_api_analyzers_share_behaviour(
    monkeypatch, tmp_path, analyzer_class, prefix, default_model
):
    monkeypatch.setattr(analyzer_class, 'MODEL_CLASS', FakeModel)
    analyzer = analyzer_class(
        api_key='k', prompt='Rate it', max_workers=2, cooldown_duration=0
    )
    model = FakeModel.created[0]
    assert model.kwargs['api_key'] == 'k'
    assert model.kwargs['model_id'] == default_model
    assert model.kwargs['max_workers'] == 2

    collection = analyzer(_collection(tmp_path))
    good = collection.get('good')
    assert good.attributes[f'{prefix}_chs_level'] == '3'
    assert good.attributes[f'{prefix}_justification'] == 'roof gone'
    assert good.attributes['ai_model_used'] == default_model
    assert good.attributes['images_analyzed_count'] == 1
    # Failed and image-less assets receive nothing:
    assert collection.get('bad').attributes == {}
    assert collection.get('noimage').attributes == {}
    # Only assets with images reach the model; the failing one is retried
    # twice more by the default rate-limit policy:
    stems = [Path(c[0][0]).stem for c in model.calls]
    assert sorted(set(stems)) == ['bad', 'good']
    assert stems.count('good') == 1 and stems.count('bad') == 3


def test_api_analyzer_custom_model_and_filter(monkeypatch, tmp_path):
    monkeypatch.setattr(ia.ClaudeAssetAnalyzer, 'MODEL_CLASS', FakeModel)
    analyzer = ia.ClaudeAssetAnalyzer(
        api_key='k',
        prompt='p',
        model_id='claude-custom',
        image_filter=lambda img: 'good' not in str(img.path),
    )
    assert analyzer.model_id == 'claude-custom'
    collection = analyzer(_collection(tmp_path))
    # The filter removed the only image of 'good', so it was skipped:
    assert collection.get('good').attributes == {}


def test_llama_analyzer_sequential(monkeypatch, tmp_path):
    monkeypatch.setattr(ia, 'LlamaVisionInference', FakeModel)
    analyzer = ia.LlamaVisionAssetAnalyzer(
        prompt='Describe', model_id='meta-llama/x', load_in_4bit=False, max_tokens=99
    )
    model = FakeModel.created[0]
    assert model.kwargs == {
        'model_id': 'meta-llama/x',
        'device': 'auto',
        'load_in_4bit': False,
        'temperature': 0.4,
        'max_tokens': 99,
    }
    collection = analyzer(_collection(tmp_path))
    good = collection.get('good')
    assert good.attributes['llama_chs_level'] == '3'
    assert good.attributes['ai_model_used'] == 'meta-llama/x'
    assert collection.get('bad').attributes == {}
    assert len(model.calls) == 2


# ------------------------------------------------------------ cancellation
def test_api_analyzer_stops_when_cancelled(monkeypatch, tmp_path):
    import threading

    from rapidtools.core import OperationCancelled

    stop = threading.Event()

    class CancellingModel(FakeModel):
        def run_inference(self, image_inputs, prompt, **kwargs):
            stop.set()  # cancel as soon as the first request is made
            return super().run_inference(image_inputs, prompt, **kwargs)

    monkeypatch.setattr(ia.GeminiAssetAnalyzer, 'MODEL_CLASS', CancellingModel)
    analyzer = ia.GeminiAssetAnalyzer(
        api_key='k', prompt='p', max_workers=1, cancel_event=stop
    )
    with pytest.raises(OperationCancelled):
        analyzer(_collection(tmp_path))
    assert len(CancellingModel.created[0].calls) <= 2


def test_local_analyzer_stops_between_batches(monkeypatch, tmp_path):
    import threading

    from rapidtools.core import OperationCancelled

    stop = threading.Event()

    class BatchModel(FakeModel):
        def run_inference_batch(self, batch_images, batch_prompts, **kwargs):
            stop.set()
            return [
                self.run_inference(imgs, p)
                for imgs, p in zip(batch_images, batch_prompts, strict=True)
            ]

    monkeypatch.setattr(ia, 'Gemma4Inference', BatchModel)
    analyzer = ia.Gemma4AssetAnalyzer(prompt='p', batch_size=1, cancel_event=stop)
    collection = _collection(tmp_path)
    with pytest.raises(OperationCancelled):
        analyzer(collection)
    # The first batch completed and its result was kept before stopping:
    analyzed = [a for a in collection if 'gemma4_chs_level' in a.attributes]
    assert len(analyzed) == 1
    assert len(BatchModel.created[0].calls) == 1


def test_local_analyzer_uses_batches_when_supported(monkeypatch, tmp_path):
    class BatchModel(FakeModel):
        def run_inference_batch(self, batch_images, batch_prompts, **kwargs):
            self.calls.append(('batch', len(batch_images)))
            return [
                ModelOutput(text='Level: 1') if imgs else None for imgs in batch_images
            ]

    monkeypatch.setattr(ia.HFVisionAssetAnalyzer, 'MODEL_CLASS', BatchModel)
    analyzer = ia.HFVisionAssetAnalyzer(
        prompt='p',
        model_id='Qwen/Qwen2.5-VL-3B-Instruct',
        batch_size=8,
        load_in_4bit=True,
        attribute_prefix='qwen',
    )
    model = BatchModel.created[0]
    assert model.kwargs['load_in_4bit'] is True
    assert model.kwargs['model_id'] == 'Qwen/Qwen2.5-VL-3B-Instruct'
    collection = analyzer(_collection(tmp_path))
    assert model.calls == [('batch', 2)]
    assert collection.get('good').attributes['qwen_level'] == '1'
    assert collection.get('bad').attributes['qwen_level'] == '1'


@pytest.mark.parametrize(
    ('analyzer_class', 'prefix', 'default_model', 'default_4bit'),
    [
        (
            ia.MuseGlimmerAssetAnalyzer,
            'muse_glimmer',
            ia.MUSE_GLIMMER_DEFAULT_MODEL,
            True,
        ),
        (ia.QwenVisionAssetAnalyzer, 'qwen_vl', ia.QWEN_VL_DEFAULT_MODEL, False),
    ],
)
def test_family_specific_local_analyzers(
    monkeypatch, tmp_path, analyzer_class, prefix, default_model, default_4bit
):
    class CatalogModel(FakeModel):
        DEFAULT_MODEL = default_model

    monkeypatch.setattr(analyzer_class, 'MODEL_CLASS', CatalogModel)
    analyzer = analyzer_class(prompt='p')
    model = CatalogModel.created[0]
    assert model.kwargs['model_id'] == default_model
    assert model.kwargs['load_in_4bit'] is default_4bit
    assert analyzer.ATTRIBUTE_PREFIX == prefix
    collection = analyzer(_collection(tmp_path))
    assert collection.get('good').attributes[f'{prefix}_chs_level'] == '3'
    assert collection.get('good').attributes['ai_model_used'] == default_model

    # An explicit prefix still wins:
    CatalogModel.created.clear()
    analyzer = analyzer_class(prompt='p', attribute_prefix='custom', model_id='x/y')
    assert analyzer.ATTRIBUTE_PREFIX == 'custom'
    assert CatalogModel.created[0].kwargs['model_id'] == 'x/y'


def test_hf_analyzer_falls_back_to_model_default(monkeypatch, tmp_path):
    class CatalogModel(FakeModel):
        DEFAULT_MODEL = 'org/default-vlm'

    monkeypatch.setattr(ia.HFVisionAssetAnalyzer, 'MODEL_CLASS', CatalogModel)
    ia.HFVisionAssetAnalyzer(prompt='p', model_id=None)
    assert CatalogModel.created[0].kwargs['model_id'] == 'org/default-vlm'


def test_pipeline_checks_cancel_between_steps(tmp_path):
    import threading

    from rapidtools import Pipeline
    from rapidtools.core import OperationCancelled

    stop = threading.Event()
    calls = []

    def step_one(col):
        calls.append(1)
        stop.set()
        return col

    def step_two(col):
        calls.append(2)
        return col

    pipeline = Pipeline(cancel_event=stop)
    pipeline.add_step(step_one)
    pipeline.add_step(step_two)
    with pytest.raises(OperationCancelled):
        pipeline.run(_collection(tmp_path))
    assert calls == [1]


def test_hf_vision_catalog():
    from rapidtools.models import OPEN_VLM_CATALOG, HFVisionInference

    ids = HFVisionInference.list_available_models()
    assert 'Qwen/Qwen3-VL-4B-Instruct' in ids
    assert len(ids) == len(OPEN_VLM_CATALOG) == len(set(ids))


# ------------------------------------------------------------ coverage extras
def test_api_analyzer_cooldown_reset_and_raw_fallback(monkeypatch, tmp_path):
    import time

    class RawModel(FakeModel):
        def run_inference(self, image_inputs, prompt, **kwargs):
            self.calls.append((list(image_inputs), prompt))
            return ModelOutput(text='no structure here')

    monkeypatch.setattr(ia.GeminiAssetAnalyzer, 'MODEL_CLASS', RawModel)
    analyzer = ia.GeminiAssetAnalyzer(api_key='k', prompt='p', cooldown_duration=0)
    collection = _collection(tmp_path)
    good = collection.get('good')

    # A pending global cooldown is honoured and a success resets the counter:
    analyzer._consecutive_error_count = 2
    analyzer._global_cooldown_until = time.time() + 0.01
    assert analyzer._process_single_asset(good) is True
    assert analyzer._consecutive_error_count == 0
    assert good.attributes['gemini_analysis_raw'] == 'no structure here'


def test_api_analyzer_handles_empty_collection_and_thread_errors(
    monkeypatch, tmp_path, caplog
):
    from rapidtools.core import PhysicalAssetCollection

    class ExplodingModel(FakeModel):
        def run_inference(self, image_inputs, prompt, **kwargs):
            raise RuntimeError('provider down')

    monkeypatch.setattr(ia.OpenAIAssetAnalyzer, 'MODEL_CLASS', ExplodingModel)
    analyzer = ia.OpenAIAssetAnalyzer(api_key='k', prompt='p', cooldown_duration=0)
    with caplog.at_level('WARNING'):
        assert len(analyzer(PhysicalAssetCollection())) == 0
    assert 'No assets with images' in caplog.text
    with caplog.at_level('ERROR'):
        analyzer(_collection(tmp_path))
    assert 'provider down' in caplog.text
    assert 'assets failed to process' in caplog.text

    # A fully successful run logs the success path:
    monkeypatch.setattr(ia.OpenAIAssetAnalyzer, 'MODEL_CLASS', FakeModel)
    analyzer = ia.OpenAIAssetAnalyzer(api_key='k', prompt='p', cooldown_duration=0)
    collection = _collection(tmp_path)
    collection.remove('bad')
    with caplog.at_level('INFO'):
        analyzer(collection)
    assert 'All assets processed successfully' in caplog.text


def test_local_analyzer_filter_raw_fallback_and_empty(monkeypatch, tmp_path, caplog):
    from rapidtools.core import PhysicalAssetCollection

    class RawModel(FakeModel):
        def run_inference(self, image_inputs, prompt, **kwargs):
            self.calls.append((list(image_inputs), prompt))
            return ModelOutput(text='free text only')

    analyzer = ia.BaseLocalAssetAnalyzer(
        RawModel(model_id='m'),
        prompt='p',
        image_filter=lambda img: 'good' in str(img.path),
        max_images_per_asset=1,
    )
    with caplog.at_level('WARNING'):
        analyzer(PhysicalAssetCollection())
    assert 'No assets with images found to analyze. Skipping.' in caplog.text
    collection = analyzer(_collection(tmp_path))
    assert collection.get('good').attributes['vlm_analysis_raw'] == 'free text only'
    # The filter removed 'bad' image -> nothing sent, asset skipped:
    assert collection.get('bad').attributes == {}
    assert analyzer._image_paths(collection.get('bad')) == []
    prompt_file = tmp_path / 'prompt.txt'
    prompt_file.write_text('from file')
    assert (
        ia.BaseLocalAssetAnalyzer(RawModel(), prompt=prompt_file).prompt == 'from file'
    )
