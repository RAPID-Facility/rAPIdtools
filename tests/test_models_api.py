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

"""Tests for the cloud API model wrappers (no network: requests-mock only)."""

import json
import re
from pathlib import Path

import pytest
import requests

from rapidtools.models import (
    ANTHROPIC_MODEL_CATALOG,
    GEMINI_MODEL_CATALOG,
    MUSE_SPARK_MODEL_CATALOG,
    OPENAI_MODEL_CATALOG,
    QWEN_MODEL_CATALOG,
    ClaudeInference,
    GeminiInference,
    MuseSparkInference,
    OpenAIInference,
    QwenInference,
)
from rapidtools.models.api_base import (
    BaseAPIInferenceModel,
    catalog_ids,
    resolve_api_key,
)
from rapidtools.models.claude import ANTHROPIC_BASE_URL
from rapidtools.models.gemini import GEMINI_BASE_URL
from rapidtools.models.muse import META_BASE_URL
from rapidtools.models.openai import OPENAI_BASE_URL
from rapidtools.models.openai_compat import BaseOpenAICompatibleInference
from rapidtools.models.qwen import QWEN_CN_BASE_URL, QWEN_INTL_BASE_URL

ALL_KEY_ENVS = (
    'OPENAI_API_KEY',
    'ANTHROPIC_API_KEY',
    'GOOGLE_API_KEY',
    'GEMINI_API_KEY',
    'MODEL_API_KEY',
    'META_API_KEY',
    'DASHSCOPE_API_KEY',
    'QWEN_API_KEY',
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Make sure no real provider key leaks into the tests."""
    for name in ALL_KEY_ENVS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def image(tmp_path) -> Path:
    """A tiny PNG on disk."""
    from PIL import Image

    path = tmp_path / 'roof.png'
    Image.new('RGB', (4, 4), (200, 30, 30)).save(path)
    return path


def _chat_response(text='CHS Level: 3', finish='stop'):
    return {
        'choices': [{'message': {'content': text}, 'finish_reason': finish}],
        'usage': {'total_tokens': 10},
    }


# ==========================================
# 1. Shared helpers
# ==========================================


def test_resolve_api_key_from_string_strips_quotes():
    """Quotes and whitespace around a literal key are removed."""
    assert resolve_api_key(' "abc" ', env_var='X_KEY') == 'abc'


def test_resolve_api_key_from_env_and_multiple_names(monkeypatch):
    """The first populated environment variable wins."""
    monkeypatch.setenv('SECOND_KEY', 'from-second')
    assert resolve_api_key(None, env_var=('FIRST_KEY', 'SECOND_KEY')) == 'from-second'
    monkeypatch.setenv('FIRST_KEY', 'from-first')
    assert resolve_api_key('', env_var=('FIRST_KEY', 'SECOND_KEY')) == 'from-first'


def test_resolve_api_key_from_path_and_string_path(tmp_path):
    """Both Path objects and strings pointing at a key file are read."""
    key_file = tmp_path / 'key.txt'
    key_file.write_text("'file-key'\n")
    assert resolve_api_key(key_file, env_var='X_KEY') == 'file-key'
    assert resolve_api_key(str(key_file), env_var='X_KEY') == 'file-key'


def test_resolve_api_key_errors(tmp_path):
    """Missing files, missing keys, and empty keys raise clear errors."""
    with pytest.raises(FileNotFoundError):
        resolve_api_key(tmp_path / 'nope.txt', env_var='X_KEY')
    with pytest.raises(ValueError, match='X_KEY'):
        resolve_api_key(None, env_var='X_KEY', provider_name='Demo')
    empty = tmp_path / 'empty.txt'
    empty.write_text('  \n')
    with pytest.raises(ValueError, match='empty'):
        resolve_api_key(empty, env_var='X_KEY')


def test_resolve_api_key_survives_oserror_on_long_keys():
    """A key too long to be a filename is returned verbatim."""
    long_key = 'k' * 5000
    assert resolve_api_key(long_key, env_var='X_KEY') == long_key


def test_catalog_ids_and_known_models():
    """Catalogue helpers expose IDs in order for every provider."""
    assert catalog_ids([{'model_id': 'a'}, {'model_id': 'b'}]) == ['a', 'b']
    for cls, catalog in [
        (OpenAIInference, OPENAI_MODEL_CATALOG),
        (ClaudeInference, ANTHROPIC_MODEL_CATALOG),
        (GeminiInference, GEMINI_MODEL_CATALOG),
        (MuseSparkInference, MUSE_SPARK_MODEL_CATALOG),
        (QwenInference, QWEN_MODEL_CATALOG),
    ]:
        ids = cls.list_known_models()
        assert ids == [e['model_id'] for e in catalog]
        assert len(ids) == len(set(ids))
        assert cls.DEFAULT_MODEL in ids if hasattr(cls, 'DEFAULT_MODEL') else True


def test_latest_models_are_present():
    """The latest generation of every provider is in its catalogue."""
    assert 'gpt-5.5' in OpenAIInference.list_known_models()
    assert 'gpt-6-sol' in OpenAIInference.list_known_models()
    assert 'claude-opus-5' in ClaudeInference.list_known_models()
    assert 'claude-fable-5-1' in ClaudeInference.list_known_models()
    assert 'gemini-3.8-flash' in GeminiInference.list_known_models()
    assert 'gemini-3.1-pro-preview' in GeminiInference.list_known_models()
    assert 'muse-spark-1.3' in MuseSparkInference.list_known_models()
    assert 'qwen3.8-max' in QwenInference.list_known_models()


def test_base_api_model_mime_and_encoding(image, requests_mock):
    """Image encoding handles files, URLs, missing files, and failures."""

    class Dummy(BaseAPIInferenceModel):
        MIME_MAP = {'.foo': 'image/foo'}

        @staticmethod
        def list_available_models(api_key=None):
            return []

        def run_inference(self, image_inputs, prompt, **kwargs):
            return None

    model = Dummy()
    assert model._get_mime_type(Path('x.foo')) == 'image/foo'
    assert model._get_mime_type(Path('x.png')) == 'image/png'
    assert model._get_mime_type(Path('x.unknownext')) == 'image/jpeg'

    encoded = model._fetch_and_encode_image(image)
    assert encoded['mime_type'] == 'image/png'
    assert encoded['data']

    requests_mock.get(
        'https://img.example/a.jpg',
        content=image.read_bytes(),
        headers={'Content-Type': 'image/jpeg'},
    )
    remote = model._fetch_and_encode_image('https://img.example/a.jpg')
    assert remote['mime_type'] == 'image/jpeg'

    requests_mock.get('https://img.example/empty.png', content=b'')
    assert model._fetch_and_encode_image('https://img.example/empty.png') is None
    requests_mock.get('https://img.example/bad.png', status_code=500)
    assert model._fetch_and_encode_image('https://img.example/bad.png') is None
    assert model._fetch_and_encode_image(image.parent / 'missing.png') is None


# ==========================================
# 2. OpenAI-compatible wrappers
# ==========================================


def test_openai_lists_models_and_validates(requests_mock, caplog):
    """The models endpoint is used for listing and for validation warnings."""
    requests_mock.get(
        f'{OPENAI_BASE_URL}/models',
        json={'data': [{'id': 'gpt-5.5'}, {'id': 'gpt-4o'}]},
    )
    assert OpenAIInference.list_available_models('sk-test') == ['gpt-4o', 'gpt-5.5']
    with caplog.at_level('WARNING'):
        model = OpenAIInference(api_key='sk-test', model_id='gpt-unknown')
    assert 'not in the OpenAI available models list' in caplog.text
    assert model.session.headers['Authorization'] == 'Bearer sk-test'


def test_openai_listing_falls_back_to_catalog(requests_mock):
    """No key, HTTP errors, empty data and exceptions all yield the catalogue."""
    catalog = catalog_ids(OPENAI_MODEL_CATALOG)
    assert OpenAIInference.list_available_models() == catalog
    requests_mock.get(f'{OPENAI_BASE_URL}/models', status_code=401)
    assert OpenAIInference.list_available_models('sk') == catalog
    requests_mock.get(f'{OPENAI_BASE_URL}/models', json={'data': []})
    assert OpenAIInference.list_available_models('sk') == catalog
    requests_mock.get(f'{OPENAI_BASE_URL}/models', exc=requests.ConnectionError)
    assert OpenAIInference.list_available_models('sk') == catalog


def test_openai_requires_key():
    """Construction without any key source fails loudly."""
    with pytest.raises(ValueError, match='OPENAI_API_KEY'):
        OpenAIInference()


def test_openai_run_inference_payload(requests_mock, image):
    """Images become data URIs and reasoning models omit temperature."""
    requests_mock.get(f'{OPENAI_BASE_URL}/models', json={'data': [{'id': 'gpt-5.5'}]})
    post = requests_mock.post(
        f'{OPENAI_BASE_URL}/chat/completions', json=_chat_response('{"a": 1}')
    )
    model = OpenAIInference(api_key='sk-test', system_instruction='Be terse.')
    out = model.run_inference(
        [image, image.parent / 'missing.png'], 'Rate it', json_mode=True
    )
    assert out.text == '{"a": 1}'
    payload = post.last_request.json()
    assert payload['model'] == 'gpt-5.5'
    assert payload['max_completion_tokens'] == 2048
    assert 'temperature' not in payload  # gpt-5.x rejects it
    assert payload['response_format'] == {'type': 'json_object'}
    assert payload['messages'][0] == {'role': 'system', 'content': 'Be terse.'}
    user = payload['messages'][1]['content']
    assert user[0] == {'type': 'text', 'text': 'Rate it'}
    assert len(user) == 2  # the missing image was skipped
    assert user[1]['image_url']['url'].startswith('data:image/png;base64,')
    assert user[1]['image_url']['detail'] == 'high'


def test_openai_temperature_for_legacy_models(requests_mock, image):
    """gpt-4o still receives temperature and per-call overrides apply."""
    requests_mock.get(f'{OPENAI_BASE_URL}/models', json={'data': [{'id': 'gpt-4o'}]})
    post = requests_mock.post(
        f'{OPENAI_BASE_URL}/chat/completions', json=_chat_response()
    )
    model = OpenAIInference(api_key='sk-test', model_id='gpt-4o', temperature=0.2)
    out = model.run_inference(image, 'p', temperature=0.9, max_tokens=5, max_retries=1)
    assert out.text == 'CHS Level: 3'
    payload = post.last_request.json()
    assert payload['temperature'] == 0.9
    assert payload['max_completion_tokens'] == 5
    # The per-call session copied the auth header:
    assert post.last_request.headers['Authorization'] == 'Bearer sk-test'


def test_openai_compat_error_paths(requests_mock, image, caplog):
    """Filters, truncation, HTTP errors, retries and bad bodies are handled."""
    requests_mock.get(f'{OPENAI_BASE_URL}/models', json={'data': [{'id': 'gpt-5.5'}]})
    model = OpenAIInference(api_key='sk-test')
    url = f'{OPENAI_BASE_URL}/chat/completions'

    requests_mock.post(url, json=_chat_response('x', finish='content_filter'))
    filtered = model.run_inference(image, 'p')
    assert filtered.failed and not filtered.retryable

    with caplog.at_level('WARNING'):
        requests_mock.post(url, json=_chat_response('trunc', finish='length'))
        assert model.run_inference(image, 'p').text == 'trunc'
    assert 'truncated' in caplog.text

    requests_mock.post(
        url,
        json={
            'choices': [
                {
                    'message': {
                        'content': [
                            {'type': 'text', 'text': 'part1 '},
                            {'type': 'text', 'text': 'part2'},
                        ]
                    },
                    'finish_reason': 'stop',
                }
            ]
        },
    )
    assert model.run_inference(image, 'p').text == 'part1 part2'

    requests_mock.post(url, json={'unexpected': True})
    assert model.run_inference(image, 'p') is None

    requests_mock.post(url, status_code=400, json={'error': {'message': 'bad request'}})
    with caplog.at_level('ERROR'):
        rejected = model.run_inference(image, 'p')
    assert rejected.failed and not rejected.retryable and rejected.text is None
    assert 'bad request' in caplog.text

    requests_mock.post(url, status_code=500, text='not json')
    assert model.run_inference(image, 'p') is None

    requests_mock.post(url, exc=requests.exceptions.RetryError)
    assert model.run_inference(image, 'p') is None

    requests_mock.post(url, exc=requests.ConnectionError('boom'))
    assert model.run_inference(image, 'p') is None

    # Nothing to send at all:
    assert model.run_inference([image.parent / 'missing.png'], '') is None
    assert model.run_inference(None, '') is None


def test_openai_text_only_prompt(requests_mock):
    """A text-only prompt (no images) is allowed."""
    requests_mock.get(f'{OPENAI_BASE_URL}/models', json={'data': []})
    post = requests_mock.post(
        f'{OPENAI_BASE_URL}/chat/completions', json=_chat_response('ok')
    )
    model = OpenAIInference(api_key='sk-test')
    assert model.run_inference([], 'Say ok').text == 'ok'
    assert post.last_request.json()['messages'][0]['content'] == [
        {'type': 'text', 'text': 'Say ok'}
    ]


def test_openai_prompt_from_file_and_batch(requests_mock, image, tmp_path):
    """Prompts may be files and run_batch yields one result per asset."""
    requests_mock.get(f'{OPENAI_BASE_URL}/models', json={'data': []})
    post = requests_mock.post(
        f'{OPENAI_BASE_URL}/chat/completions', json=_chat_response('ok')
    )
    prompt_file = tmp_path / 'prompt.txt'
    prompt_file.write_text('  Describe.  ')
    model = OpenAIInference(api_key='sk-test', max_workers=2)
    results = sorted(
        model.run_batch([('a', [image]), ('b', [tmp_path / 'nope.png'])], prompt_file)
    )
    assert results[0][0] == 'a' and results[0][1] == 'ok' and results[0][2].text == 'ok'
    # A missing image is skipped but the prompt text alone is still sent:
    assert results[1][0] == 'b' and results[1][1] == 'ok'
    text_only = [
        r
        for r in requests_mock.request_history
        if r.method == 'POST' and len(r.json()['messages'][0]['content']) == 1
    ]
    assert len(text_only) == 1
    assert post.last_request.json()['messages'][0]['content'][0]['text'] == 'Describe.'
    assert list(model.run_batch([], 'p')) == []


def test_openai_batch_reports_thread_exceptions(requests_mock, image, monkeypatch):
    """Exceptions inside a worker surface as failed results, not crashes."""
    requests_mock.get(f'{OPENAI_BASE_URL}/models', json={'data': []})
    model = OpenAIInference(api_key='sk-test', max_workers=0)

    def explode(*args, **kwargs):
        raise RuntimeError('kaboom')

    monkeypatch.setattr(model, 'run_inference', explode)
    [(asset_id, status, msg)] = list(model.run_batch([('a', [image])], 'p'))
    assert (asset_id, status) == ('a', 'failed')
    assert 'kaboom' in msg


def test_base_url_override_and_custom_provider(requests_mock, image):
    """A subclass only needs class attributes; base_url can be overridden."""

    class MyProvider(BaseOpenAICompatibleInference):
        PROVIDER_NAME = 'Mine'
        BASE_URL = 'https://llm.example.com/v1'
        API_KEY_ENV = ('MINE_KEY',)
        DEFAULT_MODEL = 'vision-1'
        MODEL_CATALOG = [{'model_id': 'vision-1', 'family': 'v'}]
        EXTRA_HEADERS = {'X-Extra': 'yes'}

        @staticmethod
        def list_available_models(api_key=None):
            return MyProvider._list_models(api_key)

    requests_mock.get(
        'https://proxy.example/v1/models', json={'data': [{'id': 'vision-1'}]}
    )
    post = requests_mock.post(
        'https://proxy.example/v1/chat/completions', json=_chat_response('hi')
    )
    model = MyProvider(api_key='k', base_url='https://proxy.example/v1/')
    assert model.base_url == 'https://proxy.example/v1'
    assert model.run_inference(image, 'p').text == 'hi'
    assert post.last_request.headers['X-Extra'] == 'yes'
    assert requests_mock.request_history[0].headers['X-Extra'] == 'yes'


def test_muse_spark_wrapper(requests_mock, image):
    """Muse Spark uses Meta's endpoint, max_tokens, and no image detail."""
    requests_mock.get(
        f'{META_BASE_URL}/models', json={'data': [{'id': 'muse-spark-1.3'}]}
    )
    post = requests_mock.post(
        f'{META_BASE_URL}/chat/completions', json=_chat_response('spark')
    )
    model = MuseSparkInference(api_key='meta-key')
    assert model.model_id == 'muse-spark-1.3'
    assert model.run_inference(image, 'Describe').text == 'spark'
    payload = post.last_request.json()
    assert payload['max_tokens'] == 2048
    assert payload['temperature'] == 0.4
    assert 'detail' not in payload['messages'][0]['content'][1]['image_url']
    assert MuseSparkInference.list_available_models('meta-key') == ['muse-spark-1.3']


def test_muse_spark_env_var_names(monkeypatch, requests_mock):
    """Both MODEL_API_KEY and META_API_KEY are honoured."""
    requests_mock.get(f'{META_BASE_URL}/models', json={'data': []})
    monkeypatch.setenv('META_API_KEY', 'from-meta')
    assert MuseSparkInference().api_key == 'from-meta'
    monkeypatch.setenv('MODEL_API_KEY', 'from-model')
    assert MuseSparkInference().api_key == 'from-model'


def test_qwen_regions_and_payload(requests_mock, image):
    """Qwen picks the regional endpoint and validates the region name."""
    requests_mock.get(re.compile(r'.*/compatible-mode/v1/models'), json={'data': []})
    post = requests_mock.post(
        re.compile(r'.*/compatible-mode/v1/chat/completions'),
        json=_chat_response('qwen'),
    )
    intl = QwenInference(api_key='sk-q')
    assert intl.base_url == QWEN_INTL_BASE_URL and intl.region == 'intl'
    cn = QwenInference(api_key='sk-q', region='cn', model_id='qwen3-vl-plus')
    assert cn.base_url == QWEN_CN_BASE_URL
    custom = QwenInference(
        api_key='sk-q', region='ignored', base_url='https://q.example/v1'
    )
    assert custom.base_url == 'https://q.example/v1'
    with pytest.raises(ValueError, match='Unknown Qwen region'):
        QwenInference(api_key='sk-q', region='mars')
    assert cn.run_inference(image, 'p').text == 'qwen'
    assert post.last_request.url.startswith(QWEN_CN_BASE_URL)
    assert post.last_request.json()['model'] == 'qwen3-vl-plus'
    assert QwenInference.list_available_models() == catalog_ids(QWEN_MODEL_CATALOG)


def test_qwen_env_var(monkeypatch, requests_mock):
    """DASHSCOPE_API_KEY is the default key source."""
    requests_mock.get(re.compile(r'.*/models'), json={'data': []})
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'ds-key')
    assert QwenInference().api_key == 'ds-key'


# ==========================================
# 3. Anthropic Claude
# ==========================================


def _claude_response(blocks, stop='end_turn', **extra):
    return {'content': blocks, 'stop_reason': stop, **extra}


def test_claude_construction_and_listing(requests_mock, caplog):
    """Headers, listing, validation warning and catalogue fallback."""
    requests_mock.get(
        f'{ANTHROPIC_BASE_URL}/models',
        json={'data': [{'id': 'claude-opus-5'}, {'id': 'claude-haiku-4-5'}]},
    )
    assert ClaudeInference.list_available_models('sk-ant') == [
        'claude-haiku-4-5',
        'claude-opus-5',
    ]
    assert requests_mock.last_request.headers['x-api-key'] == 'sk-ant'
    assert requests_mock.last_request.headers['anthropic-version'] == '2023-06-01'
    with caplog.at_level('WARNING'):
        model = ClaudeInference(api_key='sk-ant', model_id='claude-nope')
    assert 'not in the Anthropic available models list' in caplog.text
    assert model.session.headers['x-api-key'] == 'sk-ant'

    catalog = catalog_ids(ANTHROPIC_MODEL_CATALOG)
    assert ClaudeInference.list_available_models() == catalog
    requests_mock.get(f'{ANTHROPIC_BASE_URL}/models', status_code=403)
    assert ClaudeInference.list_available_models('sk-ant') == catalog
    requests_mock.get(f'{ANTHROPIC_BASE_URL}/models', exc=requests.ConnectionError)
    assert ClaudeInference.list_available_models('sk-ant') == catalog
    with pytest.raises(ValueError, match='ANTHROPIC_API_KEY'):
        ClaudeInference()


def test_claude_supports_temperature():
    """Current-generation models drop temperature; older ones keep it."""
    assert ClaudeInference.supports_temperature('claude-opus-5') is False
    assert ClaudeInference.supports_temperature('claude-fable-5-1') is False
    assert ClaudeInference.supports_temperature('claude-sonnet-5') is False
    assert ClaudeInference.supports_temperature('claude-opus-4-6') is True
    assert ClaudeInference.supports_temperature('claude-haiku-4-5') is True


def test_claude_run_inference_payload_and_thinking_blocks(requests_mock, image):
    """Images precede text, JSON hint goes to system, thinking is skipped."""
    requests_mock.get(
        f'{ANTHROPIC_BASE_URL}/models', json={'data': [{'id': 'claude-opus-5'}]}
    )
    post = requests_mock.post(
        f'{ANTHROPIC_BASE_URL}/messages',
        json=_claude_response(
            [
                {'type': 'thinking', 'thinking': ''},
                {'type': 'text', 'text': '{"level": '},
                {'type': 'text', 'text': '2}'},
            ]
        ),
    )
    model = ClaudeInference(api_key='sk-ant', system_instruction='Be brief.')
    out = model.run_inference(
        [image, image.parent / 'nope.png'], 'Rate', json_mode=True
    )
    assert out.text == '{"level": 2}'
    payload = post.last_request.json()
    assert payload['model'] == 'claude-opus-5'
    assert payload['max_tokens'] == 2048
    assert 'temperature' not in payload
    assert payload['system'].startswith(
        'Be brief.\n\nYou must respond with ONLY valid JSON'
    )
    content = payload['messages'][0]['content']
    assert content[0]['type'] == 'image'
    assert content[0]['source']['media_type'] == 'image/png'
    assert content[-1] == {'type': 'text', 'text': 'Rate'}


def test_claude_json_hint_without_system_and_temperature_for_haiku(
    requests_mock, image
):
    """Without a system prompt the JSON hint stands alone; Haiku gets temperature."""
    requests_mock.get(f'{ANTHROPIC_BASE_URL}/models', json={'data': []})
    post = requests_mock.post(
        f'{ANTHROPIC_BASE_URL}/messages',
        json=_claude_response([{'type': 'text', 'text': 'ok'}]),
    )
    model = ClaudeInference(
        api_key='sk-ant', model_id='claude-haiku-4-5', temperature=0.1
    )
    assert (
        model.run_inference(
            image, 'p', json_mode=True, max_retries=2, max_tokens=7
        ).text
        == 'ok'
    )
    payload = post.last_request.json()
    assert payload['system'].startswith('You must respond with ONLY valid JSON')
    assert payload['temperature'] == 0.1
    assert payload['max_tokens'] == 7
    assert post.last_request.headers['x-api-key'] == 'sk-ant'


def test_claude_error_paths(requests_mock, image, caplog):
    """Refusals, truncation, malformed bodies, HTTP and network errors."""
    requests_mock.get(f'{ANTHROPIC_BASE_URL}/models', json={'data': []})
    model = ClaudeInference(api_key='sk-ant')
    url = f'{ANTHROPIC_BASE_URL}/messages'

    requests_mock.post(
        url,
        json=_claude_response([], stop='refusal', stop_details={'category': 'cyber'}),
    )
    with caplog.at_level('WARNING'):
        refused = model.run_inference(image, 'p')
    assert refused.failed and not refused.retryable
    assert 'refused' in caplog.text

    requests_mock.post(
        url, json=_claude_response([{'type': 'text', 'text': 'cut'}], stop='max_tokens')
    )
    with caplog.at_level('WARNING'):
        assert model.run_inference(image, 'p').text == 'cut'
    assert 'truncated' in caplog.text

    requests_mock.post(url, json={'content': 'not-a-list'})
    assert model.run_inference(image, 'p') is None
    requests_mock.post(url, json=_claude_response([{'type': 'tool_use'}]))
    assert model.run_inference(image, 'p') is None

    requests_mock.post(url, status_code=400, json={'error': {'message': 'too big'}})
    with caplog.at_level('ERROR'):
        rejected = model.run_inference(image, 'p')
    assert rejected.failed and not rejected.retryable
    assert 'too big' in caplog.text
    requests_mock.post(url, status_code=502, text='gateway')
    assert model.run_inference(image, 'p') is None
    requests_mock.post(url, exc=requests.exceptions.RetryError)
    assert model.run_inference(image, 'p') is None
    requests_mock.post(url, exc=requests.ConnectionError('down'))
    assert model.run_inference(image, 'p') is None
    assert model.run_inference([image.parent / 'nope.png'], '') is None
    assert model.run_inference(None, '') is None


# ==========================================
# 4. Google Gemini
# ==========================================


def _gemini_models_json(*names):
    return {
        'models': [
            {'name': f'models/{n}', 'supportedGenerationMethods': ['generateContent']}
            for n in names
        ]
        + [
            {
                'name': 'models/embedding-001',
                'supportedGenerationMethods': ['embedContent'],
            }
        ]
    }


def _gemini_response(text='CHS Level: 4', finish='STOP'):
    return {
        'candidates': [{'content': {'parts': [{'text': text}]}, 'finishReason': finish}]
    }


def test_gemini_construction_and_listing(requests_mock, caplog, tmp_path):
    """Listing filters to generateContent models and strips the prefix."""
    requests_mock.get(
        f'{GEMINI_BASE_URL}/models',
        json=_gemini_models_json('gemini-3.8-flash', 'gemini-3.5-flash-lite'),
    )
    assert GeminiInference.list_available_models('AIza') == [
        'gemini-3.5-flash-lite',
        'gemini-3.8-flash',
    ]
    assert requests_mock.last_request.headers['x-goog-api-key'] == 'AIza'

    key_file = tmp_path / 'gemini.txt'
    key_file.write_text('AIza-file')
    model = GeminiInference(api_key=key_file, model_id='models/gemini-3.8-flash')
    assert model.api_key == 'AIza-file'
    assert model.model_id == 'gemini-3.8-flash'
    assert model.session.headers['x-goog-api-key'] == 'AIza-file'

    with caplog.at_level('WARNING'):
        GeminiInference(api_key='AIza', model_id='gemini-unknown')
    assert 'not in the available models list' in caplog.text

    catalog = catalog_ids(GEMINI_MODEL_CATALOG)
    assert GeminiInference.list_available_models() == catalog
    with pytest.raises(FileNotFoundError):
        GeminiInference(api_key=tmp_path / 'missing.txt')
    requests_mock.get(f'{GEMINI_BASE_URL}/models', status_code=429)
    assert GeminiInference.list_available_models('AIza') == catalog
    requests_mock.get(f'{GEMINI_BASE_URL}/models', json={'models': []})
    assert GeminiInference.list_available_models('AIza') == catalog
    requests_mock.get(f'{GEMINI_BASE_URL}/models', exc=requests.ConnectionError)
    assert GeminiInference.list_available_models('AIza') == catalog


def test_gemini_env_var_aliases(monkeypatch, requests_mock):
    """GOOGLE_API_KEY and GEMINI_API_KEY are both accepted."""
    requests_mock.get(f'{GEMINI_BASE_URL}/models', json={'models': []})
    monkeypatch.setenv('GEMINI_API_KEY', 'g2')
    assert GeminiInference().api_key == 'g2'
    monkeypatch.setenv('GOOGLE_API_KEY', 'g1')
    assert GeminiInference().api_key == 'g1'


def test_gemini_run_inference_payload(requests_mock, image):
    """Inline images, generation config, system instruction and JSON mode."""
    requests_mock.get(
        f'{GEMINI_BASE_URL}/models', json=_gemini_models_json('gemini-3.8-flash')
    )
    post = requests_mock.post(
        f'{GEMINI_BASE_URL}/models/gemini-3.8-flash:generateContent',
        json=_gemini_response('{"chs": 4}'),
    )
    model = GeminiInference(
        api_key='AIza', system_instruction='Be strict.', temperature=0.0
    )
    out = model.run_inference(
        [image, image.parent / 'nope.png'], 'Rate', json_mode=True, max_tokens=99
    )
    assert out.text == '{"chs": 4}'
    assert out.raw_response['candidates']
    payload = post.last_request.json()
    parts = payload['contents'][0]['parts']
    assert len(parts) == 2 and parts[0]['inline_data']['mime_type'] == 'image/png'
    assert parts[1] == {'text': 'Rate'}
    assert payload['generationConfig'] == {
        'temperature': 0.0,
        'maxOutputTokens': 99,
        'responseMimeType': 'application/json',
    }
    assert payload['systemInstruction'] == {'parts': [{'text': 'Be strict.'}]}
    assert len(payload['safetySettings']) == 4
    assert post.last_request.headers['x-goog-api-key'] == 'AIza'


def test_gemini_text_only_and_retry_override(requests_mock):
    """Text-only prompts work and max_retries creates a temporary session."""
    requests_mock.get(f'{GEMINI_BASE_URL}/models', json={'models': []})
    post = requests_mock.post(
        f'{GEMINI_BASE_URL}/models/gemini-3.8-flash:generateContent',
        json=_gemini_response('hello'),
    )
    model = GeminiInference(api_key='AIza')
    assert model.run_inference([], 'Say hello', max_retries=1).text == 'hello'
    assert post.last_request.json()['contents'][0]['parts'] == [{'text': 'Say hello'}]
    assert (
        post.last_request.json()['generationConfig']['responseMimeType'] == 'text/plain'
    )


def test_gemini_error_paths(requests_mock, image, caplog):
    """Safety blocks, halted generation, odd bodies and HTTP failures."""
    requests_mock.get(f'{GEMINI_BASE_URL}/models', json={'models': []})
    model = GeminiInference(api_key='AIza')
    url = f'{GEMINI_BASE_URL}/models/gemini-3.8-flash:generateContent'

    requests_mock.post(url, json={'promptFeedback': {'blockReason': 'SAFETY'}})
    with caplog.at_level('WARNING'):
        blocked = model.run_inference(image, 'p')
    assert blocked.failed and not blocked.retryable
    assert 'Blocked by safety filters' in caplog.text

    requests_mock.post(url, json={'candidates': [{'finishReason': 'MAX_TOKENS'}]})
    halted = model.run_inference(image, 'p')
    assert halted.failed and not halted.retryable
    requests_mock.post(
        url, json={'candidates': [{'finishReason': 'STOP', 'content': {'parts': []}}]}
    )
    with caplog.at_level('ERROR'):
        assert model.run_inference(image, 'p') is None
    assert 'Unexpected response format' in caplog.text
    requests_mock.post(url, json={})
    assert model.run_inference(image, 'p') is None

    requests_mock.post(url, status_code=400, text='{"error": "bad"}')
    with caplog.at_level('ERROR'):
        rejected = model.run_inference(image, 'p')
    assert rejected.failed and not rejected.retryable
    assert 'HTTP Error' in caplog.text
    requests_mock.post(url, exc=requests.exceptions.RetryError)
    assert model.run_inference(image, 'p') is None
    requests_mock.post(url, exc=requests.ConnectionError('down'))
    assert model.run_inference(image, 'p') is None

    # Requested images that all failed -> None; nothing at all -> None
    assert model.run_inference([image.parent / 'nope.png'], 'p') is None
    assert model.run_inference([], '') is None


def test_gemini_extract_text_joins_parts():
    """Multiple text parts are concatenated; non-text parts are ignored."""
    joined = GeminiInference._extract_text(
        {
            'candidates': [
                {
                    'content': {
                        'parts': [{'text': 'a'}, {'inline_data': {}}, {'text': 'b'}]
                    }
                }
            ]
        }
    )
    assert joined == 'ab'
    assert GeminiInference._extract_text({'candidates': []}) is None


def test_json_payloads_are_serialisable(requests_mock, image):
    """Every provider payload round-trips through json.dumps."""
    requests_mock.get(re.compile(r'.*'), json={'data': [], 'models': []})
    requests_mock.post(re.compile(r'.*'), json=_chat_response('x'))
    for cls in (OpenAIInference, MuseSparkInference, QwenInference):
        cls(api_key='k').run_inference(image, 'p')
        json.dumps(requests_mock.last_request.json())


# ==========================================
# 5. Validation edge cases shared by all providers
# ==========================================


def test_validation_tolerates_listing_errors(monkeypatch, requests_mock, caplog):
    """A crash while listing models never prevents construction."""
    requests_mock.get(re.compile(r'.*'), json={'data': [], 'models': []})

    def boom(*args, **kwargs):
        raise RuntimeError('listing exploded')

    monkeypatch.setattr(OpenAIInference, '_list_models', classmethod(boom))
    monkeypatch.setattr(ClaudeInference, 'list_available_models', staticmethod(boom))
    monkeypatch.setattr(GeminiInference, 'list_available_models', staticmethod(boom))
    with caplog.at_level('DEBUG'):
        assert OpenAIInference(api_key='k').model_id == 'gpt-5.5'
        assert ClaudeInference(api_key='k').model_id == 'claude-opus-5'
        assert GeminiInference(api_key='k').model_id == 'gemini-3.8-flash'
    assert caplog.text.count('Transient error') == 3


def test_validation_skipped_without_model_id(requests_mock):
    """An empty model_id short-circuits validation (no request is made)."""
    ClaudeInference(api_key='k', model_id='')
    GeminiInference(api_key='k', model_id='')
    assert requests_mock.call_count == 0
    # OpenAI-compatible wrappers fall back to the provider default instead:
    requests_mock.get(re.compile(r'.*'), json={'data': []})
    assert OpenAIInference(api_key='k', model_id='').model_id == 'gpt-5.5'


def test_gemini_none_images(requests_mock):
    """``None`` image input behaves like a text-only prompt."""
    requests_mock.get(f'{GEMINI_BASE_URL}/models', json={'models': []})
    requests_mock.post(
        f'{GEMINI_BASE_URL}/models/gemini-3.8-flash:generateContent',
        json=_gemini_response('hi'),
    )
    assert GeminiInference(api_key='k').run_inference(None, 'p').text == 'hi'
