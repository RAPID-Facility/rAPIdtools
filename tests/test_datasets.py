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

import logging
from dataclasses import fields
from pathlib import Path

import pytest
import requests

import rapidtools.datasets as datasets_module
from rapidtools.datasets import DATASET_REGISTRY, RemoteFile, download_dataset

# ==========================================
# Helpers
# ==========================================


class _FakeResponse:
    """Minimal stand-in for requests.Response supporting streaming reads."""

    def __init__(self, chunks, status_ok=True, fail_after=None):
        self._chunks = chunks
        self._status_ok = status_ok
        self._fail_after = fail_after
        self.headers = {'content-length': str(sum(len(c) for c in chunks))}

    def raise_for_status(self):
        if not self._status_ok:
            raise requests.exceptions.HTTPError('404 Client Error')

    def iter_content(self, chunk_size):
        for i, chunk in enumerate(self._chunks):
            if self._fail_after is not None and i == self._fail_after:
                raise self._fail_after_exc
            yield chunk

    def with_failure(self, exc):
        self._fail_after_exc = exc
        return self


def _install_fake_get(monkeypatch, factory):
    """Replace requests.get as seen by rapidtools.datasets with a recorder."""
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return factory(url)

    monkeypatch.setattr(datasets_module.requests, 'get', fake_get)
    return calls


# ==========================================
# 1. Registry shape
# ==========================================


def test_remote_file_dataclass():
    """RemoteFile exposes exactly a url and a filename."""
    remote = RemoteFile(url='https://example.com/a.txt', filename='a.txt')
    assert [f.name for f in fields(remote)] == ['url', 'filename']
    assert remote == RemoteFile('https://example.com/a.txt', 'a.txt')


def test_dataset_registry_shape():
    """Every registry entry is a non-empty list of well-formed RemoteFiles."""
    assert isinstance(DATASET_REGISTRY, dict)
    assert DATASET_REGISTRY
    for name, files in DATASET_REGISTRY.items():
        assert name == name.strip().lower(), name
        assert isinstance(files, list) and files
        for remote in files:
            assert isinstance(remote, RemoteFile)
            assert remote.url.startswith('https://www.dropbox.com/')
            assert 'dl=0' in remote.url
            assert remote.filename and '/' not in remote.filename


def test_dataset_registry_expected_entries():
    """Key sample datasets used by the examples are registered."""
    for key in ('eaton_patch1', 'eaton_patch2', 'altadena_sample_buildings'):
        assert key in DATASET_REGISTRY
    assert DATASET_REGISTRY['eaton_patch1'][0].filename.endswith('.tiff')
    filenames = [f.filename for files in DATASET_REGISTRY.values() for f in files]
    assert len(filenames) == len(set(filenames))


# ==========================================
# 2. Name validation
# ==========================================


def test_unknown_dataset_suggests_close_match(tmp_path, caplog):
    """A near-miss name raises ValueError with a 'Did you mean' hint."""
    with caplog.at_level(logging.ERROR, logger='rapidtools.datasets'):
        with pytest.raises(ValueError, match="Did you mean 'eaton_patch1'") as exc:
            download_dataset('eaton_patch_1', tmp_path)
    assert "Dataset 'eaton_patch_1' not found" in str(exc.value)
    assert 'Did you mean' in caplog.text


def test_unknown_dataset_lists_available_when_no_match(tmp_path):
    """A wildly wrong name lists the available datasets instead."""
    with pytest.raises(ValueError, match='Available datasets') as exc:
        download_dataset('zzzzqqqq', tmp_path)
    assert 'eaton_patch1' in str(exc.value)
    assert 'Did you mean' not in str(exc.value)


def test_validation_happens_before_any_download(tmp_path, monkeypatch):
    """One bad name in a list aborts before any request is made."""
    calls = _install_fake_get(monkeypatch, lambda url: _FakeResponse([b'x']))
    with pytest.raises(ValueError):
        download_dataset(['hf_token', 'not_a_dataset_at_all'], tmp_path)
    assert calls == []
    assert not (tmp_path / 'hf_token.txt').exists()


# ==========================================
# 3. Downloading
# ==========================================


def test_successful_streamed_download(tmp_path, requests_mock):
    """Content is streamed to a .tmp file, then renamed into place."""
    remote = DATASET_REGISTRY['hf_token'][0]
    direct = remote.url.replace('www.dropbox.com', 'dl.dropboxusercontent.com')
    payload = b'hf_abc123' * 2000
    requests_mock.get(direct, content=payload)

    paths = download_dataset('hf_token', tmp_path)

    assert paths == [tmp_path.resolve() / 'hf_token.txt']
    assert paths[0].read_bytes() == payload
    assert not (tmp_path / 'hf_token.tmp').exists()
    assert requests_mock.call_count == 1
    assert requests_mock.last_request.url.startswith(
        'https://dl.dropboxusercontent.com/'
    )
    assert requests_mock.last_request.timeout == 15


def test_case_whitespace_normalisation_and_dedupe(tmp_path, monkeypatch):
    """Names are normalised and duplicates collapse into one download."""
    calls = _install_fake_get(
        monkeypatch, lambda url: _FakeResponse([b'tok', b'en'])
    )
    paths = download_dataset(
        ['  HF_Token ', 'hf_token', 'Mapillary_Token'], output_dir=tmp_path
    )
    assert [p.name for p in paths] == ['hf_token.txt', 'mapillary_token.txt']
    assert len(calls) == 2
    assert all(kw['stream'] is True for _, kw in calls)
    assert (tmp_path / 'hf_token.txt').read_bytes() == b'token'


def test_single_string_input_and_directory_creation(tmp_path, monkeypatch):
    """A single name works and nested output directories are created."""
    _install_fake_get(monkeypatch, lambda url: _FakeResponse([b'data']))
    out_dir = tmp_path / 'nested' / 'deeper'
    paths = download_dataset('synthetic_landslide_image', out_dir)
    assert out_dir.is_dir()
    assert paths == [out_dir.resolve() / 'HurricaneInducedLandslide.png']
    assert paths[0].read_bytes() == b'data'


def test_existing_file_is_skipped(tmp_path, monkeypatch):
    """Already-present files are returned without hitting the network."""
    calls = _install_fake_get(monkeypatch, lambda url: _FakeResponse([b'new']))
    existing = tmp_path / 'hf_token.txt'
    existing.write_text('old contents')
    paths = download_dataset('hf_token', tmp_path)
    assert paths == [existing.resolve()]
    assert existing.read_text() == 'old contents'
    assert calls == []


def test_returned_paths_are_absolute(tmp_path, monkeypatch):
    """Relative output directories are resolved to absolute paths."""
    _install_fake_get(monkeypatch, lambda url: _FakeResponse([b'x']))
    monkeypatch.chdir(tmp_path)
    paths = download_dataset('hf_token', 'rel_dir')
    assert paths[0].is_absolute()
    assert paths[0] == (tmp_path / 'rel_dir' / 'hf_token.txt').resolve()


# ==========================================
# 4. Failure handling
# ==========================================


def test_network_error_is_logged_and_skipped(tmp_path, requests_mock, caplog):
    """Connection failures log an error and do not produce a file."""
    remote = DATASET_REGISTRY['hf_token'][0]
    direct = remote.url.replace('www.dropbox.com', 'dl.dropboxusercontent.com')
    requests_mock.get(direct, exc=requests.exceptions.ConnectionError('boom'))
    with caplog.at_level(logging.ERROR, logger='rapidtools.datasets'):
        paths = download_dataset('hf_token', tmp_path)
    assert paths == []
    assert 'Network Error while downloading hf_token.txt' in caplog.text
    assert list(tmp_path.iterdir()) == []


def test_http_error_status_is_a_network_error(tmp_path, requests_mock, caplog):
    """Non-2xx responses raise via raise_for_status and are handled."""
    remote = DATASET_REGISTRY['mapillary_token'][0]
    direct = remote.url.replace('www.dropbox.com', 'dl.dropboxusercontent.com')
    requests_mock.get(direct, status_code=404)
    with caplog.at_level(logging.ERROR, logger='rapidtools.datasets'):
        assert download_dataset('mapillary_token', tmp_path) == []
    assert 'Network Error' in caplog.text


def test_mid_stream_network_error_cleans_up_tmp(tmp_path, monkeypatch, caplog):
    """A failure while streaming removes the partial .tmp file."""
    exc = requests.exceptions.ChunkedEncodingError('connection dropped')
    _install_fake_get(
        monkeypatch,
        lambda url: _FakeResponse([b'part1', b'part2'], fail_after=1).with_failure(
            exc
        ),
    )
    with caplog.at_level(logging.DEBUG, logger='rapidtools.datasets'):
        paths = download_dataset('hf_token', tmp_path)
    assert paths == []
    assert not (tmp_path / 'hf_token.tmp').exists()
    assert not (tmp_path / 'hf_token.txt').exists()
    assert 'Network Error' in caplog.text
    assert 'Cleaned up incomplete temporary file' in caplog.text


def test_generic_exception_is_logged_and_cleaned_up(tmp_path, monkeypatch, caplog):
    """Unexpected errors are reported as such and leave no temp file."""
    _install_fake_get(
        monkeypatch,
        lambda url: _FakeResponse([b'a', b'b'], fail_after=1).with_failure(
            RuntimeError('disk on fire')
        ),
    )
    with caplog.at_level(logging.ERROR, logger='rapidtools.datasets'):
        paths = download_dataset('hf_token', tmp_path)
    assert paths == []
    assert 'Unexpected error downloading hf_token.txt: disk on fire' in caplog.text
    assert not (tmp_path / 'hf_token.tmp').exists()


def test_partial_success_across_multiple_files(tmp_path, monkeypatch, caplog):
    """One failing file does not prevent the others from downloading."""

    def factory(url):
        if 'mapillary' in url:
            return _FakeResponse([b'x'], status_ok=False)
        return _FakeResponse([b'ok'])

    _install_fake_get(monkeypatch, factory)
    with caplog.at_level(logging.INFO, logger='rapidtools.datasets'):
        paths = download_dataset(
            ['hf_token', 'mapillary_token', 'aerial_chs_prompts'], tmp_path
        )
    assert [p.name for p in paths] == ['hf_token.txt', 'aerial_CHS_prompts.txt']
    assert 'Successfully secured 2/3 files.' in caplog.text
    assert isinstance(paths[0], Path)
