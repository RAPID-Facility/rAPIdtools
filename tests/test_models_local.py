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

"""Tests for the local (Hugging Face / PyTorch) model wrappers with fake weights."""

from pathlib import Path

import numpy as np
import pytest
import torch

from rapidtools.models import (
    GEMMA4_MODEL_CATALOG,
    LLAMA_MODEL_CATALOG,
    MUSE_GLIMMER_MODEL_CATALOG,
    OPEN_VLM_CATALOG,
    QWEN_VL_MODEL_CATALOG,
    SAM3_MODEL_CATALOG,
    Gemma4Inference,
    HFVisionInference,
    LlamaVisionInference,
    MuseGlimmerInference,
    QwenVisionInference,
    SAM3Inference,
)
from rapidtools.models import base as base_mod
from rapidtools.models import gemma4 as gemma4_mod
from rapidtools.models import hf_vision as hf_mod
from rapidtools.models import llama as llama_mod
from rapidtools.models import sam3 as sam3_mod
from rapidtools.models.base import BaseInferenceModel, ModelOutput
from rapidtools.models.local_base import BaseLocalInferenceModel

# ==========================================
# 0. Fakes standing in for Transformers objects
# ==========================================


class FakeTokenizer:
    def __init__(self):
        self.padding_side = 'right'
        self.pad_token = None
        self.eos_token = '<eos>'


class FakeInputs(dict):
    """Mimics a BatchFeature: a dict with ``.to(device)``."""

    def to(self, device):
        self.device = device
        return self


class FakeProcessor:
    """Records calls and produces deterministic tensors."""

    created: list = []
    prompt_len = 3

    def __init__(self, model_id, **kwargs):
        self.model_id = model_id
        self.kwargs = kwargs
        self.tokenizer = FakeTokenizer()
        self.calls: list = []
        FakeProcessor.created.append(self)

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return cls(model_id, **kwargs)

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(('template', messages, kwargs))
        if not kwargs.get('tokenize'):
            return '<|image|>' * 1 + 'PROMPT'
        batch = len(messages) if isinstance(messages[0], list) else 1
        return FakeInputs(
            input_ids=torch.ones((batch, self.prompt_len), dtype=torch.long)
        )

    def __call__(self, text=None, images=None, return_tensors='pt'):
        self.calls.append(('process', text, len(images or [])))
        return FakeInputs(input_ids=torch.ones((1, self.prompt_len), dtype=torch.long))

    def decode(self, ids, skip_special_tokens=True):
        return ' '.join(f't{int(i)}' for i in ids.tolist())

    def parse_response(self, text, prefix=''):
        return {'parsed': text}


class FakeModel:
    """Generates ``new_tokens`` extra tokens per row."""

    new_tokens = 2
    fail = False
    oom = False

    def __init__(self, model_id, **kwargs):
        self.model_id = model_id
        self.kwargs = kwargs
        self.device = torch.device('cpu')
        self.eval_called = False
        self.generate_kwargs: dict = {}

    def eval(self):
        self.eval_called = True

    def to(self, device):
        self.device = torch.device(device)
        return self

    def generate(self, **kwargs):
        if self.oom:
            raise torch.OutOfMemoryError('cuda oom')
        if self.fail:
            raise RuntimeError('generation failed')
        self.generate_kwargs = kwargs
        batch, length = kwargs['input_ids'].shape
        new = torch.arange(1, self.new_tokens + 1).repeat(batch, 1)
        return torch.cat([kwargs['input_ids'], new], dim=1)


@pytest.fixture
def fake_hf(monkeypatch):
    """Patch every heavy Transformers entry point with the fakes above."""
    FakeProcessor.created.clear()
    FakeModel.fail = False
    FakeModel.oom = False
    loaded: list[FakeModel] = []

    def loader(model_id, **kwargs):
        m = FakeModel(model_id, **kwargs)
        loaded.append(m)
        return m

    for mod in (hf_mod, llama_mod, gemma4_mod):
        monkeypatch.setattr(mod, 'AutoProcessor', FakeProcessor)
        monkeypatch.setattr(mod, '_load_multimodal_model', loader)
    return loaded


@pytest.fixture
def image(tmp_path) -> Path:
    from PIL import Image

    path = tmp_path / 'roof.jpg'
    Image.new('RGB', (8, 8), (10, 200, 10)).save(path)
    return path


# ==========================================
# 1. Base classes
# ==========================================


def test_model_output_flags():
    """Convenience properties reflect the populated fields."""
    assert not ModelOutput().has_text
    assert not ModelOutput(text='  ').has_text
    assert ModelOutput(text='x').has_text
    assert ModelOutput(masks=[np.zeros((1, 2, 2))]).has_masks
    assert not ModelOutput(bounding_boxes=[]).has_bounding_boxes
    assert ModelOutput(bounding_boxes=[[0, 0, 1, 1]]).has_bounding_boxes


def test_resolve_prompt_from_file_and_text(tmp_path):
    """Prompt files are read and stripped; plain text passes through."""
    f = tmp_path / 'p.txt'
    f.write_text('\n Describe. \n')
    assert BaseInferenceModel._resolve_prompt(f) == 'Describe.'
    assert BaseInferenceModel._resolve_prompt('Describe.') == 'Describe.'
    assert BaseInferenceModel._resolve_prompt('x' * 6000) == 'x' * 6000


def test_base_run_batch_threads_results():
    """The threaded batch yields ok/failed tuples and handles crashes."""

    class Echo(BaseInferenceModel):
        max_workers = 3

        def run_inference(self, image_inputs, prompt, **kwargs):
            if image_inputs == 'crash':
                raise RuntimeError('boom')
            if image_inputs == 'none':
                return None
            return ModelOutput(text=f'{image_inputs}:{prompt}')

    results = {
        r[0]: r
        for r in Echo().run_batch([('a', 'img'), ('b', 'none'), ('c', 'crash')], 'p')
    }
    assert results['a'][1] == 'ok' and results['a'][2].text == 'img:p'
    assert results['b'][1:] == ('failed', 'Inference failed or was blocked.')
    assert results['c'][1] == 'failed' and 'boom' in results['c'][2]
    assert list(Echo().run_batch([], 'p')) == []


def test_local_base_image_loading(image, requests_mock, tmp_path):
    """PIL loading covers files, URLs, missing files and broken payloads."""

    class Local(BaseLocalInferenceModel):
        def run_inference(self, image_inputs, prompt, **kwargs):
            return ModelOutput(text='ok') if image_inputs else None

    model = Local()
    assert model._load_image_as_pil(image).mode == 'RGB'
    assert model._load_image_as_pil(tmp_path / 'nope.jpg') is None
    requests_mock.get('https://img.example/a.jpg', content=image.read_bytes())
    assert model._load_image_as_pil('https://img.example/a.jpg').size == (8, 8)
    requests_mock.get('https://img.example/bad.jpg', content=b'not an image')
    assert model._load_image_as_pil('https://img.example/bad.jpg') is None
    (tmp_path / 'corrupt.jpg').write_bytes(b'\x00\x01')
    assert model._load_image_as_pil(tmp_path / 'corrupt.jpg') is None


def test_local_base_run_batch_is_sequential(monkeypatch):
    """Local batches run one by one and report None results and crashes."""

    class Local(BaseLocalInferenceModel):
        def run_inference(self, image_inputs, prompt, **kwargs):
            if image_inputs == 'crash':
                raise RuntimeError('gpu died')
            return ModelOutput(text='ok') if image_inputs else None

    out = list(Local().run_batch([('a', 'x'), ('b', ''), ('c', 'crash')], 'p'))
    assert [o[:2] for o in out] == [('a', 'ok'), ('b', 'failed'), ('c', 'failed')]
    assert 'gpu died' in out[2][2]
    assert list(Local().run_batch([], 'p')) == []


# ==========================================
# 2. Generic Hugging Face backend
# ==========================================


def test_catalogs_are_consistent():
    """Every catalogue has unique IDs and contains its class default."""
    for cls, catalog in [
        (HFVisionInference, OPEN_VLM_CATALOG),
        (QwenVisionInference, QWEN_VL_MODEL_CATALOG),
        (MuseGlimmerInference, MUSE_GLIMMER_MODEL_CATALOG),
        (LlamaVisionInference, LLAMA_MODEL_CATALOG),
        (Gemma4Inference, GEMMA4_MODEL_CATALOG),
        (SAM3Inference, SAM3_MODEL_CATALOG),
    ]:
        ids = cls.list_available_models()
        assert ids == [e['model_id'] for e in catalog]
        assert len(ids) == len(set(ids))
    assert HFVisionInference.DEFAULT_MODEL in HFVisionInference.list_available_models()
    assert (
        QwenVisionInference.DEFAULT_MODEL in QwenVisionInference.list_available_models()
    )
    assert MuseGlimmerInference.DEFAULT_MODEL == 'meta-models/Muse-Glimmer-30B'
    assert 'Qwen/Qwen3.8-27B' in QwenVisionInference.list_available_models()
    assert (
        'meta-llama/Llama-4-Scout-17B-16E-Instruct'
        in LlamaVisionInference.list_available_models()
    )
    assert 'google/gemma-4-31B-it' in Gemma4Inference.list_available_models()


def test_transformers_version_helpers():
    """Version helpers compare against the installed release."""
    assert hf_mod.transformers_at_least('0.1.0')
    assert not hf_mod.transformers_at_least('999.0.0')
    assert hf_mod.transformers_version().split('.')[0].isdigit()


def test_load_multimodal_model_prefers_auto_class(monkeypatch):
    """The first available multimodal auto class is used; none -> ImportError."""
    import transformers

    class Auto:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return ('loaded', model_id, kwargs)

    monkeypatch.setattr(transformers, 'AutoModelForMultimodalLM', Auto, raising=False)
    assert hf_mod._load_multimodal_model('x/y', dtype='auto')[1] == 'x/y'
    monkeypatch.setattr(transformers, 'AutoModelForMultimodalLM', None, raising=False)
    monkeypatch.setattr(
        transformers, 'AutoModelForImageTextToText', Auto, raising=False
    )
    assert hf_mod._load_multimodal_model('x/y')[0] == 'loaded'
    monkeypatch.setattr(
        transformers, 'AutoModelForImageTextToText', None, raising=False
    )
    with pytest.raises(ImportError):
        hf_mod._load_multimodal_model('x/y')


def test_hf_vision_loading_variants(fake_hf):
    """Device and quantization options map onto loader kwargs."""
    model = HFVisionInference()
    assert model.model_id == hf_mod.DEFAULT_OPEN_VLM
    assert fake_hf[-1].kwargs['device_map'] == 'auto'
    assert fake_hf[-1].kwargs['dtype'] == 'auto'
    assert fake_hf[-1].eval_called
    tok = FakeProcessor.created[-1].tokenizer
    assert tok.padding_side == 'left' and tok.pad_token == '<eos>'

    HFVisionInference('org/m', device='cpu')
    assert 'device_map' not in fake_hf[-1].kwargs
    assert fake_hf[-1].device == torch.device('cpu')

    HFVisionInference(
        'org/m', device='cuda:0', load_in_4bit=True, trust_remote_code=True
    )
    assert fake_hf[-1].kwargs['device_map'] == 'cuda:0'
    assert 'quantization_config' in fake_hf[-1].kwargs
    assert fake_hf[-1].kwargs['trust_remote_code'] is True
    assert FakeProcessor.created[-1].kwargs == {'trust_remote_code': True}


def test_hf_vision_version_gate(fake_hf, monkeypatch):
    """Catalogue entries with min_transformers raise a helpful ImportError."""
    monkeypatch.setattr(hf_mod, 'transformers_at_least', lambda minimum: False)
    with pytest.raises(ImportError, match='transformers>=5.15.0'):
        HFVisionInference('meta-models/Muse-Glimmer-30B')
    with pytest.raises(ImportError, match='transformers>=5.15.0'):
        MuseGlimmerInference()
    # Models without a requirement still load:
    HFVisionInference('Qwen/Qwen3.5-4B')
    monkeypatch.setattr(hf_mod, 'transformers_at_least', lambda minimum: True)
    assert MuseGlimmerInference().model_id == 'meta-models/Muse-Glimmer-30B'


def test_hf_vision_run_inference(fake_hf, image, tmp_path):
    """Single-image, multi-image, text-only and failed-image cases."""
    model = HFVisionInference('org/m', temperature=0.0, max_tokens=7)
    out = model.run_inference(image, 'Describe')
    assert out.text == 't1 t2'
    assert out.raw_response == {'generated_text': 't1 t2'}
    gen = fake_hf[-1].generate_kwargs
    assert gen['max_new_tokens'] == 7 and gen['do_sample'] is False
    assert 'temperature' not in gen
    messages = FakeProcessor.created[-1].calls[-1][1]
    assert messages[0]['content'][-1] == {'type': 'text', 'text': 'Describe'}
    assert messages[0]['content'][0]['type'] == 'image'

    out = model.run_inference([image, image], 'p', temperature=0.7, max_tokens=3)
    gen = fake_hf[-1].generate_kwargs
    assert (
        gen['temperature'] == 0.7
        and gen['do_sample'] is True
        and gen['max_new_tokens'] == 3
    )

    assert model.run_inference(None, 'text only').text == 't1 t2'
    assert model.run_inference([tmp_path / 'nope.jpg'], 'p') is None

    prompt_file = tmp_path / 'prompt.txt'
    prompt_file.write_text('From file')
    model.run_inference(image, prompt_file)
    assert (
        FakeProcessor.created[-1].calls[-1][1][0]['content'][-1]['text'] == 'From file'
    )

    FakeModel.fail = True
    assert model.run_inference(image, 'p') is None


def test_hf_vision_run_inference_batch(fake_hf, image, tmp_path):
    """Batched generation aligns results and retries per asset on failure."""
    model = HFVisionInference('org/m')
    outs = model.run_inference_batch(
        [[image], [tmp_path / 'nope.jpg'], [image]], ['a', 'b', 'c']
    )
    assert [o.text if o else None for o in outs] == ['t1 t2', None, 't1 t2']
    template_kwargs = FakeProcessor.created[-1].calls[-1][2]
    assert template_kwargs['padding'] is True

    assert model.run_inference_batch([[tmp_path / 'nope.jpg']], ['a']) == [None]

    # Batch failure with several assets -> retried one at a time:
    calls = {'n': 0}
    original = fake_hf[-1].generate

    def flaky(**kwargs):
        calls['n'] += 1
        if kwargs['input_ids'].shape[0] > 1:
            raise RuntimeError('oom-ish')
        return original(**kwargs)

    fake_hf[-1].generate = flaky
    outs = model.run_inference_batch([[image], [image]], ['a', 'b'])
    assert [o.text for o in outs] == ['t1 t2', 't1 t2']
    assert calls['n'] == 3
    # Single failing asset is simply None:
    FakeModel.fail = True
    fake_hf[-1].generate = FakeModel.generate.__get__(fake_hf[-1])
    assert model.run_inference_batch([[image]], ['a']) == [None]


def test_hf_vision_batch_length_mismatch(fake_hf, image):
    """Mismatched batch inputs raise immediately."""
    model = HFVisionInference('org/m')
    with pytest.raises(ValueError):
        model.run_inference_batch([[image]], ['a', 'b'])


def test_search_hub_models(monkeypatch):
    """Hub results are appended after the curated list; failures degrade."""
    import huggingface_hub

    class Found:
        def __init__(self, id):
            self.id = id

    monkeypatch.setattr(
        huggingface_hub,
        'list_models',
        lambda **kw: [Found('Qwen/Qwen3.5-4B'), Found('org/new'), Found(None)],
    )
    ids = HFVisionInference.search_hub_models(limit=3)
    assert ids[: len(OPEN_VLM_CATALOG)] == HFVisionInference.list_available_models()
    assert ids[-1] == 'org/new' and ids.count('Qwen/Qwen3.5-4B') == 1

    def boom(**kw):
        raise OSError('offline')

    monkeypatch.setattr(huggingface_hub, 'list_models', boom)
    assert (
        HFVisionInference.search_hub_models()
        == HFVisionInference.list_available_models()
    )
    assert (
        QwenVisionInference.search_hub_models()
        == QwenVisionInference.list_available_models()
    )


def test_family_subclasses(fake_hf, image, monkeypatch):
    """Qwen and Muse Glimmer wrappers reuse the generic backend."""
    monkeypatch.setattr(hf_mod, 'transformers_at_least', lambda minimum: True)
    qwen = QwenVisionInference()
    assert qwen.model_id == 'Qwen/Qwen3.5-4B'
    assert qwen.run_inference(image, 'p').text == 't1 t2'
    glimmer = MuseGlimmerInference(load_in_4bit=True)
    assert glimmer.model_id == 'meta-models/Muse-Glimmer-30B'
    assert 'quantization_config' in fake_hf[-1].kwargs
    assert glimmer.FAMILY_NAME == 'Muse Glimmer'


# ==========================================
# 3. Llama
# ==========================================


def test_llama_loading_and_inference(fake_hf, image, tmp_path):
    """Llama builds a text template, processes images and decodes new tokens."""
    model = LlamaVisionInference()
    assert model.model_id == llama_mod.LLAMA_DEFAULT_MODEL
    assert fake_hf[-1].kwargs == {'device_map': 'auto', 'dtype': torch.float16}

    model = LlamaVisionInference(
        'meta-llama/Llama-4-Scout-17B-16E-Instruct', load_in_4bit=True, device='cuda'
    )
    assert fake_hf[-1].kwargs['device_map'] == 'cuda'
    assert (
        'quantization_config' in fake_hf[-1].kwargs
        and 'dtype' not in fake_hf[-1].kwargs
    )

    out = model.run_inference(
        [image, image], 'Describe', json_mode=True, temperature=0.0, max_tokens=4
    )
    assert out.text == 't1 t2'
    assert out.raw_response['output_ids'][0][-2:] == [1, 2]
    proc = FakeProcessor.created[-1]
    template_messages = proc.calls[-2][1]
    assert template_messages[0]['content'][:2] == [{'type': 'image'}, {'type': 'image'}]
    assert 'only valid JSON' in template_messages[0]['content'][2]['text']
    assert proc.calls[-1][2] == 2  # two images processed
    gen = fake_hf[-1].generate_kwargs
    assert gen['max_new_tokens'] == 4 and gen['do_sample'] is False

    assert model.run_inference(tmp_path / 'nope.jpg', 'p') is None
    FakeModel.fail = True
    assert model.run_inference(image, 'p') is None
    FakeModel.fail = False
    FakeModel.oom = True
    assert model.run_inference(image, 'p') is None
    FakeModel.oom = False

    def broken_template(*a, **k):
        raise ValueError('bad template')

    proc.apply_chat_template = broken_template
    assert model.run_inference(image, 'p') is None


# ==========================================
# 4. Gemma 4
# ==========================================


def test_gemma4_loading(fake_hf, caplog, monkeypatch):
    """Device handling, unsupported-ID warning and the version gate."""
    model = Gemma4Inference()
    assert model.model_id == gemma4_mod.GEMMA4_DEFAULT_MODEL
    assert fake_hf[-1].kwargs == {'dtype': 'auto', 'device_map': 'auto'}
    Gemma4Inference(device='cpu')
    assert fake_hf[-1].kwargs == {'dtype': 'auto'} and fake_hf[
        -1
    ].device == torch.device('cpu')
    with caplog.at_level('WARNING'):
        Gemma4Inference('google/gemma-4-custom')
    assert 'not in officially supported Gemma-4 list' in caplog.text
    monkeypatch.setattr(gemma4_mod, 'transformers_at_least', lambda m: False)
    with pytest.raises(ImportError, match='transformers>=5.10.0'):
        Gemma4Inference('google/gemma-4-12B-it')
    Gemma4Inference('google/gemma-4-E4B-it')  # no requirement -> fine


def test_gemma4_run_inference(fake_hf, image, tmp_path):
    """Single inference handles images, text-only prompts and failures."""
    model = Gemma4Inference(temperature=0.0, max_tokens=5)
    out = model.run_inference(image, 'Describe')
    assert out.text == 't1 t2'
    assert out.raw_response['parsed_response'] == {'parsed': 't1 t2'}
    gen = fake_hf[-1].generate_kwargs
    assert gen == {**gen, 'max_new_tokens': 5, 'temperature': 0.0, 'do_sample': False}

    out = model.run_inference(None, 'text only', temperature=0.5)
    assert out.text == 't1 t2' and fake_hf[-1].generate_kwargs['do_sample'] is True
    assert model.run_inference([tmp_path / 'nope.jpg'], 'p') is None

    # Processor without parse_response and with a strict signature:
    proc = FakeProcessor.created[-1]
    proc.parse_response = lambda text: {'strict': text}
    assert model.run_inference(image, 'p').raw_response['parsed_response'] == {
        'strict': 't1 t2'
    }
    proc.parse_response = None  # processor without the hook
    assert model.run_inference(image, 'p').raw_response['parsed_response'] == 't1 t2'

    FakeModel.fail = True
    assert model.run_inference(image, 'p') is None


def test_gemma4_run_inference_batch(fake_hf, image, tmp_path):
    """Batch results align with inputs; total failure yields all None."""
    model = Gemma4Inference()
    outs = model.run_inference_batch(
        [[image], [tmp_path / 'nope.jpg'], [image]], ['a', 'b', 'c']
    )
    assert [o.text if o else None for o in outs] == ['t1 t2', None, 't1 t2']
    assert outs[0].raw_response['parsed_response'] == {'parsed': 't1 t2'}
    assert model.run_inference_batch([[tmp_path / 'nope.jpg']], ['a']) == [None]
    FakeModel.fail = True
    assert model.run_inference_batch([[image], [image]], ['a', 'b']) == [None, None]


# ==========================================
# 5. SAM 3
# ==========================================


class FakeSamProcessor:
    created: list = []

    def __init__(self, model_id):
        self.model_id = model_id
        self.calls: list = []
        FakeSamProcessor.created.append(self)

    @classmethod
    def from_pretrained(cls, model_id):
        return cls(model_id)

    def __call__(self, images, return_tensors='pt', text=None):
        self.calls.append(('process', len(images), text))
        n = len(images)
        return {
            'pixel_values': torch.zeros((n, 3, 4, 4)),
            'original_sizes': torch.tensor([[8, 8]] * n),
            'meta': 'not-a-tensor',
        }

    def post_process_instance_segmentation(
        self, outputs, threshold, mask_threshold, target_sizes
    ):
        self.calls.append(('post', threshold, mask_threshold, target_sizes))
        results = []
        for i, _size in enumerate(target_sizes):
            res = {
                'masks': torch.ones((2, 8, 8), dtype=torch.bool),
                'scores': torch.tensor([0.9, 0.8]),
            }
            if i == 0:
                res['boxes'] = torch.tensor([[0, 0, 4, 4], [4, 4, 8, 8]])
            results.append(res)
        return results


class FakeSamModel:
    fail = False
    oom = False

    def __init__(self, model_id, **kwargs):
        self.model_id = model_id
        self.kwargs = kwargs
        self.device = torch.device('cpu')
        self.eval_called = False

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return cls(model_id, **kwargs)

    def eval(self):
        self.eval_called = True

    def __call__(self, **inputs):
        if self.oom:
            raise torch.OutOfMemoryError('oom')
        if self.fail:
            raise RuntimeError('forward failed')
        assert inputs['pixel_values'].device == self.device
        return {'logits': torch.zeros(1)}


@pytest.fixture
def fake_sam(monkeypatch):
    FakeSamProcessor.created.clear()
    FakeSamModel.fail = False
    FakeSamModel.oom = False
    monkeypatch.setattr(sam3_mod, 'Sam3Processor', FakeSamProcessor)
    monkeypatch.setattr(sam3_mod, 'Sam3Model', FakeSamModel)


def test_sam3_loading_options(fake_sam):
    """Device map and 4-bit options are forwarded to the model loader."""
    sam = SAM3Inference()
    assert sam.model_id == 'facebook/sam3'
    assert sam.model.kwargs == {'dtype': torch.float16, 'device_map': 'auto'}
    assert sam.model.eval_called
    sam = SAM3Inference(device='cuda:1', load_in_4bit=True)
    assert sam.model.kwargs['device_map'] == 'cuda:1'
    assert 'quantization_config' in sam.model.kwargs


def test_sam3_run_inference(fake_sam, image, tmp_path):
    """Masks, boxes and scores are extracted per image."""
    sam = SAM3Inference()
    out = sam.run_inference(
        [image, image], 'building', threshold=0.3, mask_threshold=0.6
    )
    assert out.text is None
    assert len(out.masks) == 2 and out.masks[0].shape == (2, 8, 8)
    assert out.bounding_boxes == [[[0, 0, 4, 4], [4, 4, 8, 8]]]
    assert out.raw_response['status'] == 'success'
    assert out.raw_response['scores'] == [[pytest.approx(0.9), pytest.approx(0.8)]] * 2
    proc = FakeSamProcessor.created[-1]
    assert proc.calls[0] == ('process', 2, ['building', 'building'])
    assert proc.calls[1][1:] == (0.3, 0.6, [[8, 8], [8, 8]])

    # No prompt -> no text passed; prompt from file works too
    sam.run_inference(image)
    assert proc.calls[-2] == ('process', 1, None)
    prompt_file = tmp_path / 'prompt.txt'
    prompt_file.write_text('tree')
    sam.run_inference(image, prompt_file)
    assert proc.calls[-2] == ('process', 1, ['tree'])

    assert sam.run_inference(tmp_path / 'nope.jpg', 'x') is None


def test_sam3_error_paths(fake_sam, image):
    """Processor errors, forward failures and OOM return None."""
    sam = SAM3Inference()
    FakeSamModel.fail = True
    assert sam.run_inference(image, 'x') is None
    FakeSamModel.fail = False
    FakeSamModel.oom = True
    assert sam.run_inference(image, 'x') is None
    FakeSamModel.oom = False

    def broken(*a, **k):
        raise ValueError('bad input')

    original_call = FakeSamProcessor.__call__
    FakeSamProcessor.__call__ = broken
    try:
        assert sam.run_inference(image, 'x') is None
    finally:
        FakeSamProcessor.__call__ = original_call


def test_base_module_exports():
    """The base module re-exports the public names used across the package."""
    assert base_mod.ModelOutput is ModelOutput
    assert base_mod.BaseInferenceModel is BaseInferenceModel


def test_json_mode_prompt_instruction_for_local_models(fake_hf, image):
    """json_mode (kwarg or config) appends the JSON instruction to the prompt."""
    from rapidtools.models import GenerationConfig

    model = HFVisionInference('org/m')
    model.run_inference(image, 'Describe', json_mode=True)
    text = FakeProcessor.created[-1].calls[-1][1][0]['content'][-1]['text']
    assert text.startswith('Describe') and 'only valid JSON' in text
    model.run_inference_batch(
        [[image]], ['Rate'], config=GenerationConfig(json_mode=True)
    )
    text = FakeProcessor.created[-1].calls[-1][1][0]['content'][-1]['text']
    assert 'only valid JSON' in text

    gemma = Gemma4Inference()
    gemma.run_inference(
        image, 'Describe', config=GenerationConfig(json_mode=True, max_tokens=3)
    )
    proc = FakeProcessor.created[-1]
    assert 'only valid JSON' in proc.calls[-1][1][0]['content'][-1]['text']
    assert fake_hf[-1].generate_kwargs['max_new_tokens'] == 3
    gemma.run_inference_batch([[image]], ['Rate'], json_mode=True)
    assert 'only valid JSON' in proc.calls[-1][1][0][0]['content'][-1]['text']


def test_sam3_deprecated_sampling_arguments_and_config(fake_sam, image):
    """temperature/max_tokens warn; SegmentationConfig sets the thresholds."""
    from rapidtools.models import SegmentationConfig

    with pytest.warns(DeprecationWarning, match='ignores temperature'):
        sam = SAM3Inference(temperature=0.4)
    with pytest.warns(DeprecationWarning):
        SAM3Inference(max_tokens=5)
    sam.run_inference(
        image, 'x', config=SegmentationConfig(threshold=0.2, mask_threshold=0.7)
    )
    proc = sam.processor
    assert proc.calls[-1][1:3] == (0.2, 0.7)
    sam.run_inference(
        image, 'x', threshold=0.9, config=SegmentationConfig(threshold=0.2)
    )
    assert proc.calls[-1][1] == 0.9  # explicit keyword wins
