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
import tomllib
from pathlib import Path

import pytest

import rapidtools

PYPROJECT = Path(__file__).resolve().parents[1] / 'pyproject.toml'

# ==========================================
# 1. Package Metadata
# ==========================================


def test_version_matches_pyproject():
    """__version__ must agree with the version declared in pyproject.toml."""
    with open(PYPROJECT, 'rb') as f:
        project = tomllib.load(f)['project']
    assert rapidtools.__version__ == project['version']
    assert rapidtools.name == project['name'] == 'rapidtools'
    assert 'BSD' in rapidtools.__license__
    assert 'University of Washington' in rapidtools.__copyright__


def test_all_names_are_importable():
    """Every name listed in __all__ is exposed as a top-level attribute."""
    assert rapidtools.__all__ == sorted(rapidtools.__all__)
    assert len(set(rapidtools.__all__)) == len(rapidtools.__all__)
    for name in rapidtools.__all__:
        assert hasattr(rapidtools, name), name
    assert rapidtools.BoundingBox is rapidtools.core.BoundingBox
    assert callable(rapidtools.download_dataset)


# ==========================================
# 2. Lazy loading and logging
# ==========================================


def test_heavy_subpackages_load_lazily():
    """Model and processing names resolve on first access and are cached."""
    import importlib
    import sys

    assert 'Pipeline' in rapidtools.__all__
    assert (
        rapidtools.Pipeline is importlib.import_module('rapidtools.processing').Pipeline
    )
    assert (
        rapidtools.GeminiInference is sys.modules['rapidtools.models'].GeminiInference
    )
    assert 'Pipeline' in rapidtools.__dict__  # cached after first lookup
    assert set(rapidtools.__all__) <= set(dir(rapidtools))
    with pytest.raises(AttributeError, match='NoSuchThing'):
        _ = rapidtools.NoSuchThing


def test_configure_logging_is_idempotent():
    """configure_logging attaches exactly one console handler to 'rapidtools'."""
    import io

    package_logger = logging.getLogger('rapidtools')
    before = list(package_logger.handlers)
    try:
        stream = io.StringIO()
        first = rapidtools.configure_logging('DEBUG', stream=stream)
        second = rapidtools.configure_logging(logging.INFO, stream=stream)
        assert first is second is package_logger
        console = [
            h
            for h in package_logger.handlers
            if getattr(h, '_rapidtools_console', False)
        ]
        assert len(console) == 1
        assert package_logger.level == logging.INFO
        logging.getLogger('rapidtools.demo').info('hello from rapidtools')
        assert 'hello from rapidtools' in stream.getvalue()
        assert any(isinstance(h, logging.NullHandler) for h in before)
    finally:
        for handler in list(package_logger.handlers):
            if getattr(handler, '_rapidtools_console', False):
                package_logger.removeHandler(handler)
        package_logger.setLevel(logging.NOTSET)


# ==========================================
# 3. Hugging Face authentication (opt-in)
# ==========================================


def test_login_returns_early_when_token_cached(monkeypatch):
    """A cached token means no download or login is attempted."""
    from rapidtools import auth

    def fail(*args, **kwargs):
        raise AssertionError('must not be called')

    monkeypatch.setattr(auth, 'get_token', lambda: 'hf_cached')
    monkeypatch.setattr(auth, 'download_dataset', fail)
    monkeypatch.setattr(auth, 'hf_login', fail)
    assert auth.login() is True
    assert auth.ensure_huggingface_login() is True
    assert rapidtools.login is auth.login


def test_login_with_explicit_token(monkeypatch):
    """An explicit token is cached through huggingface_hub.login."""
    from rapidtools import auth

    logins = []
    monkeypatch.setattr(auth, 'hf_login', lambda **kwargs: logins.append(kwargs))
    assert auth.login(token=' hf_explicit ') is True
    assert logins == [{'token': 'hf_explicit', 'add_to_git_credential': False}]

    def broken(**kwargs):
        raise RuntimeError('bad token')

    monkeypatch.setattr(auth, 'hf_login', broken)
    assert auth.login(token='x') is False


def test_login_downloads_registry_token(tmp_path, monkeypatch, caplog):
    """Without a cached token the registry file is read, used, and destroyed."""
    from rapidtools import auth

    token_file = tmp_path / 'hf_token.txt'
    token_file.write_text('hf_secret_value\n')
    downloads, logins = [], []
    monkeypatch.setattr(auth, 'get_token', lambda: None)
    monkeypatch.setattr(
        auth,
        'download_dataset',
        lambda names: (downloads.append(names), [token_file])[1],
    )
    monkeypatch.setattr(auth, 'hf_login', lambda **kwargs: logins.append(kwargs))
    with caplog.at_level(logging.INFO):
        assert auth.ensure_huggingface_login() is True
    assert downloads == [['hf_token']]
    assert logins == [{'token': 'hf_secret_value', 'add_to_git_credential': False}]
    assert not token_file.exists()
    assert 'Hugging Face auto-authentication successful.' in caplog.text


def test_login_failure_is_logged_not_raised(monkeypatch, caplog):
    """Any error during download or login is logged and swallowed."""
    from rapidtools import auth

    monkeypatch.setattr(auth, 'get_token', lambda: None)

    def offline(names):
        raise ConnectionError('offline')

    monkeypatch.setattr(auth, 'download_dataset', offline)
    with caplog.at_level(logging.ERROR):
        assert auth.login() is False
    assert 'Failed to auto-authenticate with Hugging Face: offline' in caplog.text
