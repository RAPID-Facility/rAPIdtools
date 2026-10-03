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
# 09-28-2026

"""
Configuration utilities for the rapidtools package.

Centralizes logging formats, HTTP request defaults (headers, timeouts, retry
policy), mask-type definitions, and default colormaps so that every module
shares a single, consistent configuration surface.

Example:
    >>> from rapidtools.config import MaskType, get_configured_session
    >>> session = get_configured_session(retries=3)
    >>> session.adapters['https://'].max_retries.total
    3
    >>> MaskType.SEMANTIC == 'semantic'
    True
"""

import logging
import sys
import warnings
from collections.abc import Iterable
from enum import StrEnum
from typing import Any

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
# Only idempotent methods are retried automatically. A POST that reached the
# server may already have been processed (and billed), so the API clients opt
# in through API_RETRY_METHODS, limited to API_RETRY_STATUSES: the statuses
# that mean the provider did not process the request (rate limited,
# unavailable, or overloaded; 529 is Anthropic's overload status).
DEFAULT_ALLOWED_METHODS = ['HEAD', 'GET', 'OPTIONS']
API_RETRY_METHODS = ['HEAD', 'GET', 'OPTIONS', 'POST']
API_RETRY_STATUSES = [429, 503, 529]

REQUESTS_TIMEOUT_VAL = 30


# Mask & Segmentation Configuration:
class MaskType(StrEnum):
    """
    Central definition for available mask types.

    Because this is a :class:`~enum.StrEnum`, members compare equal to their
    plain string values and can be passed anywhere a string is expected.

    Attributes:
        SEMANTIC: One label per pixel class (``'semantic'``).
        INSTANCE: One label per distinct object instance (``'instance'``).

    Example:
        >>> from rapidtools.config import MaskType
        >>> MaskType('instance') is MaskType.INSTANCE
        True
        >>> [m.value for m in MaskType]
        ['semantic', 'instance']
    """

    SEMANTIC = 'semantic'
    INSTANCE = 'instance'


DEFAULT_SEMANTIC_CMAP = 'tab20'
DEFAULT_INSTANCE_CMAP = 'nipy_spectral'


# Helper functions:
def get_configured_session(
    retries: int = DEFAULT_RETRY_TOTAL,
    backoff_factor: float = DEFAULT_BACKOFF,
    allowed_methods: Iterable[str] | None = None,
    status_forcelist: Iterable[int] | None = None,
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
        allowed_methods (Iterable[str] | None):
            HTTP methods that may be retried. Defaults to the idempotent
            DEFAULT_ALLOWED_METHODS; pass API_RETRY_METHODS to include POST.
        status_forcelist (Iterable[int] | None):
            Response statuses that trigger a retry. Defaults to
            DEFAULT_STATUS_FORCELIST.

    Returns:
        requests.Session: A configured session object ready for use.

    Example:
        >>> from rapidtools.config import get_configured_session
        >>> session = get_configured_session(retries=2, backoff_factor=0.5)
        >>> session.headers['Connection']
        'keep-alive'
        >>> session.adapters['http://'].max_retries.backoff_factor
        0.5
    """
    session = requests.Session()

    # Apply standard headers:
    session.headers.update(REQUESTS_HEADERS)

    # Define retry strategy dynamically based on arguments:
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=list(
            DEFAULT_STATUS_FORCELIST if status_forcelist is None else status_forcelist
        ),
        allowed_methods=frozenset(
            DEFAULT_ALLOWED_METHODS if allowed_methods is None else allowed_methods
        ),
    )

    # Mount the retry adapter to both HTTP and HTTPS:
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('https://', adapter)
    session.mount('http://', adapter)

    return session


def configure_logging(
    level: int | str = logging.INFO,
    stream: Any = None,
    fmt: str = LOG_FORMAT,
    datefmt: str = DATE_FORMAT,
) -> logging.Logger:
    """
    Attach a console handler to the ``rapidtools`` logger.

    Importing the library never touches logging configuration (it only adds a
    :class:`logging.NullHandler`); scripts and notebooks call this once to see
    progress messages. Calling it again replaces the previous handler. The
    ``rapidtools`` logger stops propagating to the root logger so messages are
    printed once even when the application configured root logging too; skip
    this function if you would rather route rapidtools through your own
    handlers.

    Args:
        level: Logging level for the ``rapidtools`` logger.
        stream: Output stream; defaults to ``sys.stdout``.
        fmt: Log record format.
        datefmt: Timestamp format.

    Returns:
        logging.Logger: The configured ``rapidtools`` logger.

    Example:
        >>> import rapidtools
        >>> rapidtools.configure_logging('INFO')  # doctest: +ELLIPSIS
        <Logger rapidtools (INFO)>
    """
    package_logger = logging.getLogger('rapidtools')
    for handler in list(package_logger.handlers):
        if getattr(handler, '_rapidtools_console', False):
            package_logger.removeHandler(handler)
    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    handler._rapidtools_console = True  # type: ignore[attr-defined]
    package_logger.addHandler(handler)
    package_logger.setLevel(level)
    # Records are emitted by this handler only; without this every message
    # would also reach the root logger's handlers and print twice.
    package_logger.propagate = False
    return package_logger


def resolve_alias(
    kwargs: dict[str, Any],
    new_name: str,
    *old_names: str,
    default: Any = None,
    stacklevel: int = 3,
) -> Any:
    """
    Pop a deprecated keyword alias from ``kwargs`` and return the value.

    Used while parameter names are being harmonized: components accept the
    new name (e.g. ``save_directory``) and still honour the old one (e.g.
    ``output_dir``) with a :class:`DeprecationWarning`.

    Args:
        kwargs: The ``**kwargs`` dict received by the constructor (mutated).
        new_name: The canonical parameter name, for the warning message.
        *old_names: Deprecated aliases to look for, in priority order.
        default: Value returned when no alias is present.
        stacklevel: Passed to :func:`warnings.warn`.

    Returns:
        Any: The alias value, or ``default``.

    Raises:
        TypeError: If ``kwargs`` still holds unrelated keys afterwards
            (mirrors the error a normal signature would raise).

    Example:
        >>> from rapidtools.config import resolve_alias
        >>> kwargs = {'output_dir': 'tiles'}
        >>> resolve_alias(kwargs, 'save_directory', 'output_dir')  # warns
        'tiles'
        >>> kwargs
        {}
    """
    value = default
    for old in old_names:
        if old in kwargs:
            value = kwargs.pop(old)
            warnings.warn(
                f"'{old}' is deprecated; use '{new_name}' instead.",
                DeprecationWarning,
                stacklevel=stacklevel,
            )
    if kwargs:
        unexpected = ', '.join(sorted(kwargs))
        raise TypeError(f'Unexpected keyword argument(s): {unexpected}')
    return value
