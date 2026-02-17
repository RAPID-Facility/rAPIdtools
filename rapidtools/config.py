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
# 02-16-2025

"""Configuration utilities for the rapidtools package."""

from enum import StrEnum

import requests
from requests.adapters import HTTPAdapter, Retry

# Logging Configuration:
LOG_FORMAT = '%(asctime)s- %(levelname)s: %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Network Configuration:
REQUESTS_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/131.0.0.0 Safari/537.36'
    ),
    'Accept': (
        'text/html,application/xhtml+xml,application/xml;'
        'q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8'
    ),
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
}

DEFAULT_RETRY_TOTAL = 5
DEFAULT_BACKOFF = 1
DEFAULT_STATUS_FORCELIST = [429, 500, 502, 503, 504]
DEFAULT_ALLOWED_METHODS = ['HEAD', 'GET', 'POST', 'OPTIONS']

REQUESTS_TIMEOUT_VAL = 30

# Mask & Segmentation Configuration:
class MaskType(StrEnum):
    """Central definition for available mask types."""
    SEMANTIC = 'semantic'
    INSTANCE = 'instance'

DEFAULT_SEMANTIC_CMAP = 'tab20'
DEFAULT_INSTANCE_CMAP = 'nipy_spectral'

# Helper functions:
def get_configured_session(
    retries: int = DEFAULT_RETRY_TOTAL,
    backoff_factor: float = DEFAULT_BACKOFF
) -> requests.Session:
    """
    Create a requests Session with retry logic, timeouts, and headers.
    
    Args:
        retries (int): 
            Number of total retries to attempt. Defaults to 
            DEFAULT_RETRY_TOTAL (5).
        backoff_factor (float): 
            Time factor for exponential backoff. Defaults to DEFAULT_BACKOFF
            (1).

    Returns:
        requests.Session: A configured session object ready for use.
    """
    session = requests.Session()

    # Apply standard headers:
    session.headers.update(REQUESTS_HEADERS)

    # Define retry strategy dynamically based on arguments:
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=DEFAULT_STATUS_FORCELIST,
        allowed_methods=DEFAULT_ALLOWED_METHODS
    )

    # Mount the retry adapter to both HTTP and HTTPS:
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('https://', adapter)
    session.mount('http://', adapter)

    return session