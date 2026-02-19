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
# 02-18-2026

import requests
from requests.adapters import HTTPAdapter

# Import everything relevant from your new config
from rapidtools.config import (
    DATE_FORMAT,
    DEFAULT_INSTANCE_CMAP,
    DEFAULT_SEMANTIC_CMAP,
    LOG_FORMAT,
    REQUESTS_HEADERS,
    REQUESTS_TIMEOUT_VAL,
    DEFAULT_RETRY_TOTAL,
    DEFAULT_BACKOFF,
    DEFAULT_STATUS_FORCELIST,
    DEFAULT_ALLOWED_METHODS,
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
    # Check for keys present in your new dictionary
    assert 'User-Agent' in REQUESTS_HEADERS
    assert 'Mozilla/5.0' in REQUESTS_HEADERS['User-Agent']
    assert 'Accept' in REQUESTS_HEADERS
    assert 'Accept-Language' in REQUESTS_HEADERS
    assert 'Accept-Encoding' in REQUESTS_HEADERS
    assert 'Connection' in REQUESTS_HEADERS
    assert REQUESTS_HEADERS['Connection'] == 'keep-alive'

def test_request_settings_constants():
    """Verify default setting constants."""
    # Timeout
    assert isinstance(REQUESTS_TIMEOUT_VAL, int)
    assert REQUESTS_TIMEOUT_VAL == 30

    # Retry Defaults
    assert DEFAULT_RETRY_TOTAL == 5
    assert DEFAULT_BACKOFF == 1
    assert 429 in DEFAULT_STATUS_FORCELIST
    assert 500 in DEFAULT_STATUS_FORCELIST
    assert 'GET' in DEFAULT_ALLOWED_METHODS
    assert 'POST' in DEFAULT_ALLOWED_METHODS

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
    # Note: StrEnum functionality
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

def test_get_configured_session_defaults():
    """
    Verify that the session factory returns a session with
    the default global configuration applied.
    """
    session = get_configured_session()

    # 1. Check Return Type
    assert isinstance(session, requests.Session)

    # 2. Check Headers (Merged correctly)
    assert session.headers['User-Agent'] == REQUESTS_HEADERS['User-Agent']
    assert session.headers['Accept-Encoding'] == REQUESTS_HEADERS['Accept-Encoding']

    # 3. Check Adapter Mounting
    http_adapter = session.adapters['http://']
    https_adapter = session.adapters['https://']

    assert isinstance(http_adapter, HTTPAdapter)
    assert isinstance(https_adapter, HTTPAdapter)

    # 4. Check Default Retry Strategy
    # Since the strategy is now created inside the function, we inspect the adapter object
    assert http_adapter.max_retries.total == DEFAULT_RETRY_TOTAL
    assert http_adapter.max_retries.backoff_factor == DEFAULT_BACKOFF
    assert http_adapter.max_retries.status_forcelist == DEFAULT_STATUS_FORCELIST

def test_get_configured_session_custom_args():
    """
    Verify that we can override the defaults (retries and backoff)
    when calling the function.
    """
    custom_retries = 10
    custom_backoff = 2.5
    
    session = get_configured_session(retries=custom_retries, backoff_factor=custom_backoff)

    http_adapter = session.adapters['http://']
    
    # Verify the custom values stuck
    assert http_adapter.max_retries.total == custom_retries
    assert http_adapter.max_retries.backoff_factor == custom_backoff
