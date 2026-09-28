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

"""
Shared plumbing for cloud vision-language APIs.

Every hosted provider wrapper in :mod:`rapidtools.models` (Gemini, Claude,
OpenAI, Meta Muse Spark, Qwen) derives from :class:`BaseAPIInferenceModel`.
The base class owns the retrying :class:`requests.Session`, Base64 image
encoding, MIME-type detection and the API-key resolution rules that are shared
by all providers:

    1. an explicit key string,
    2. a path (``str`` or :class:`pathlib.Path`) to a file containing the key,
    3. the provider's environment variable (e.g. ``GOOGLE_API_KEY``).

Example:
    >>> from rapidtools.models.api_base import resolve_api_key
    >>> resolve_api_key('sk-live-123', env_var='OPENAI_API_KEY')
    'sk-live-123'
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from abc import abstractmethod
from pathlib import Path

from rapidtools.config import REQUESTS_TIMEOUT_VAL, get_configured_session

from .base import BaseInferenceModel

logger = logging.getLogger(__name__)

# Vision requests carry images and long prompts; give providers more time than
# the general-purpose request timeout.
API_REQUEST_TIMEOUT = 60.0


def resolve_api_key(
    key_input: str | Path | None,
    env_var: str | tuple[str, ...],
    provider_name: str = 'API',
) -> str:
    """
    Resolve an API key from a string, a key file, or environment variables.

    Args:
        key_input:
            The raw key, a path to a text file holding the key, or ``None``
            to fall back to ``env_var``.
        env_var:
            Name (or names, tried in order) of the environment variable(s)
            consulted when ``key_input`` is empty.
        provider_name:
            Human-readable provider name used in error messages.

    Returns:
        str: The cleaned key with surrounding quotes and whitespace removed.

    Raises:
        FileNotFoundError: If a :class:`pathlib.Path` is given but does not
            point to an existing file.
        ValueError: If no key could be resolved or the resolved key is empty.

    Example:
        >>> import os
        >>> from rapidtools.models.api_base import resolve_api_key
        >>> os.environ['DEMO_KEY'] = ' "abc123" '
        >>> resolve_api_key(None, env_var='DEMO_KEY')
        'abc123'
    """
    env_vars = (env_var,) if isinstance(env_var, str) else tuple(env_var)

    if isinstance(key_input, Path):
        if not key_input.is_file():
            raise FileNotFoundError(f'API key file not found at: {key_input}')
        resolved_key = key_input.read_text(encoding='utf-8').strip()
    else:
        raw_key = (key_input or '').strip()
        if not raw_key:
            for name in env_vars:
                raw_key = os.environ.get(name, '').strip()
                if raw_key:
                    break
        if not raw_key:
            raise ValueError(
                f'{provider_name} key is missing. Provide it as an argument or '
                f'set the {" or ".join(env_vars)} environment variable.'
            )
        resolved_key = raw_key
        try:
            potential_path = Path(raw_key)
            if potential_path.is_file():
                resolved_key = potential_path.read_text(encoding='utf-8').strip()
        except OSError:
            pass

    resolved_key = resolved_key.strip('\'" \n\t')
    if not resolved_key:
        raise ValueError(f'Resolved {provider_name} key is empty.')
    return resolved_key


def catalog_ids(catalog: list[dict[str, str]]) -> list[str]:
    """
    Return the ``model_id`` of every entry in a model catalogue.

    Args:
        catalog: A list of ``{'model_id': ..., 'family': ..., ...}`` records.

    Returns:
        list[str]: The model IDs in catalogue order.

    Example:
        >>> from rapidtools.models.api_base import catalog_ids
        >>> catalog_ids([{'model_id': 'a'}, {'model_id': 'b'}])
        ['a', 'b']
    """
    return [entry['model_id'] for entry in catalog]


class BaseAPIInferenceModel(BaseInferenceModel):
    """
    Intermediate base class for cloud API models.

    Handles the retrying network session, image Base64 encoding and key
    resolution so that concrete wrappers only need to build the provider's
    request payload and parse its response.

    Class attributes:
        PROVIDER_NAME: Human-readable provider label used in logs and errors.
        API_KEY_ENV: Environment variable(s) checked when no key is passed.
        MODEL_CATALOG: Curated ``{'model_id', 'family', 'notes'}`` records
            describing the models rapidtools knows about for the provider.
        MIME_MAP: Extension -> MIME type overrides for formats the provider
            documents explicitly.

    Example:
        Subclasses are used directly; the base class only appears in type
        hints:

        >>> from rapidtools.models import GeminiInference
        >>> from rapidtools.models.api_base import BaseAPIInferenceModel
        >>> issubclass(GeminiInference, BaseAPIInferenceModel)
        True
    """

    PROVIDER_NAME: str = 'API'
    API_KEY_ENV: str | tuple[str, ...] = ()
    MODEL_CATALOG: list[dict[str, str]] = []
    MIME_MAP: dict[str, str] = {}

    def __init__(
        self,
        max_retries: int = 3,
        max_workers: int = 10,
        timeout: float = API_REQUEST_TIMEOUT,
    ):
        """
        Create the shared HTTP session.

        Args:
            max_retries: Retries applied by the session to transient failures.
            max_workers: Upper bound on threads used by :meth:`run_batch`.
            timeout: Seconds to wait for a provider response before the
                request counts as failed. Defaults to 60; raise it for large
                images or long answers.
        """
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.timeout = timeout
        self.session = get_configured_session(retries=self.max_retries)

    @classmethod
    @abstractmethod
    def list_available_models(cls, api_key: str | Path | None = None) -> list[str]:
        """Every API wrapper must implement a way to list valid models."""

    @classmethod
    def list_known_models(cls) -> list[str]:
        """
        Return the curated catalogue of model IDs for this provider.

        Unlike :meth:`list_available_models` this never touches the network.

        Returns:
            list[str]: Model IDs in catalogue order (newest first).

        Example:
            >>> from rapidtools.models import OpenAIInference
            >>> 'gpt-5.5' in OpenAIInference.list_known_models()
            True
        """
        return catalog_ids(cls.MODEL_CATALOG)

    @classmethod
    def _resolve_api_key(cls, key_input: str | Path | None) -> str:
        """Resolve the key using the provider's environment variable(s)."""
        return resolve_api_key(key_input, cls.API_KEY_ENV, cls.PROVIDER_NAME)

    def _get_mime_type(self, image_path: Path) -> str:
        """
        Guess the MIME type of an image, preferring the provider's ``MIME_MAP``.

        Args:
            image_path: Path (or URL path) whose suffix is inspected.

        Returns:
            str: A MIME type such as ``'image/png'``; ``'image/jpeg'`` when
            the extension is unknown.
        """
        ext = image_path.suffix.lower()
        child_mime_map = getattr(self, 'MIME_MAP', {})
        if ext in child_mime_map:
            return child_mime_map[ext]
        guess, _ = mimetypes.guess_type(image_path)
        return guess or 'image/jpeg'

    def _fetch_and_encode_image(self, image_input: str | Path) -> dict[str, str] | None:
        """
        Load an image from disk or a URL and Base64-encode it.

        Args:
            image_input: Local path or ``http(s)://`` URL.

        Returns:
            dict[str, str] | None: ``{'mime_type': ..., 'data': <base64>}``
            or ``None`` when the image could not be read (the error is logged).

        Example:
            >>> from pathlib import Path
            >>> from rapidtools.models import OpenAIInference
            >>> model = OpenAIInference(api_key='sk-test')
            >>> encoded = model._fetch_and_encode_image(
            ...     Path('roof.png')
            ... )
            >>> encoded['mime_type']
            'image/png'
        """
        input_str = str(image_input)
        image_bytes = None
        mime_type = 'image/jpeg'

        try:
            if input_str.startswith(('http://', 'https://')):
                logger.info(f'Downloading image from URL: {input_str}')
                resp = self.session.get(input_str, timeout=REQUESTS_TIMEOUT_VAL)
                resp.raise_for_status()
                image_bytes = resp.content
                mime_type = resp.headers.get('Content-Type') or self._get_mime_type(
                    Path(input_str)
                )
            else:
                path_obj = Path(image_input)
                if not path_obj.exists():
                    logger.error(f'File not found: {path_obj}')
                    return None
                mime_type = self._get_mime_type(path_obj)
                image_bytes = path_obj.read_bytes()
        except Exception as e:
            logger.error(f'Error preparing image {input_str}: {e}')
            return None

        if not image_bytes:
            return None

        return {
            'mime_type': mime_type,
            'data': base64.b64encode(image_bytes).decode('utf-8'),
        }
