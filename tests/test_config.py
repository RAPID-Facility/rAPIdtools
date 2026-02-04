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
# 02-03-2026

import requests
from requests.adapters import HTTPAdapter

from rapidtools.config import (
    DATE_FORMAT,
    DEFAULT_INSTANCE_CMAP,
    DEFAULT_SEMANTIC_CMAP,
    LOG_FORMAT,
    REQUESTS_HEADERS,
    REQUESTS_RETRY_STRATEGY,
    REQUESTS_TIMEOUT_VAL,
    MaskType,
    get_configured_session,
)

# ==========================================
# 1. Constants Verification
# ==========================================

def test_log_formats():
    """Ensure logging formats are strings and contain expected placeholders."""
    assert isinstance(LOG_FORMAT, str)
    assert '%(asctime)s' in LOG_FORMAT
    assert '%(levelname)s' in LOG_FORMAT
    assert '%(message)s' in LOG_FORMAT

    assert isinstance(DATE_FORMAT, str)
    assert '%Y-%m-%d' in DATE_FORMAT

def test_request_headers_structure():
    """Ensure standard headers are present and correct."""
    assert isinstance(REQUESTS_HEADERS, dict)
    assert 'User-Agent' in REQUESTS_HEADERS
    assert 'Mozilla/5.0' in REQUESTS_HEADERS['User-Agent']
    assert 'Accept' in REQUESTS_HEADERS
    assert 'Connection' in REQUESTS_HEADERS
    assert REQUESTS_HEADERS['Connection'] == 'keep-alive'

def test_request_settings():
    """Verify timeout and retry strategy settings."""
    # Timeout
    assert isinstance(REQUESTS_TIMEOUT_VAL, int)
    assert REQUESTS_TIMEOUT_VAL == 30

    # Retry Strategy
    assert REQUESTS_RETRY_STRATEGY.total == 5
    assert REQUESTS_RETRY_STRATEGY.backoff_factor == 1
    assert 429 in REQUESTS_RETRY_STRATEGY.status_forcelist
    assert 500 in REQUESTS_RETRY_STRATEGY.status_forcelist
    assert 'GET' in REQUESTS_RETRY_STRATEGY.allowed_methods

def test_default_colormaps():
    """Ensure default colormaps are valid matplotlib strings."""
    assert DEFAULT_SEMANTIC_CMAP == 'tab20'
    assert DEFAULT_INSTANCE_CMAP == 'nipy_spectral'

# ==========================================
# 2. Enum Verification (MaskType)
# ==========================================

def test_mask_type_enum():
    """Test MaskType behavior as a StrEnum."""
    # Test values
    assert MaskType.SEMANTIC == 'semantic'
    assert MaskType.INSTANCE == 'instance'

    # Test type (StrEnum acts as str)
    assert isinstance(MaskType.SEMANTIC, str)
    assert MaskType.SEMANTIC.upper() == 'SEMANTIC'

    # Test iteration
    values = [m.value for m in MaskType]
    assert 'semantic' in values
    assert 'instance' in values
    assert len(values) == 2

def test_mask_type_comparison():
    """Ensure direct string comparison works."""
    assert MaskType.SEMANTIC == 'semantic'
    assert MaskType.INSTANCE != 'semantic'

# ==========================================
# 3. Helper Function Verification
# ==========================================

def test_get_configured_session():
    """
    Verify that the session factory returns a session with
    the correct global configuration applied.
    """
    session = get_configured_session()

    # 1. Check Return Type
    assert isinstance(session, requests.Session)

    # 2. Check Headers (Merged correctly)
    assert session.headers['User-Agent'] == REQUESTS_HEADERS['User-Agent']
    assert session.headers['Accept-Encoding'] == REQUESTS_HEADERS['Accept-Encoding']

    # 3. Check Adapter Mounting (Retry Logic)
    # Requests sessions store adapters in a dict keyed by prefix (http://, https://)
    http_adapter = session.adapters['http://']
    https_adapter = session.adapters['https://']

    assert isinstance(http_adapter, HTTPAdapter)
    assert isinstance(https_adapter, HTTPAdapter)

    # Verify the strategy attached to the adapter matches our config
    assert http_adapter.max_retries is REQUESTS_RETRY_STRATEGY
    assert https_adapter.max_retries is REQUESTS_RETRY_STRATEGY

    # Verify specific attributes of the mounted strategy
    assert http_adapter.max_retries.total == 5
    assert http_adapter.max_retries.backoff_factor == 1
