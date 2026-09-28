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
Generic wrapper for OpenAI-compatible chat-completions APIs.

Several providers expose the same ``POST /chat/completions`` schema that
OpenAI popularised: OpenAI itself, Meta's Model API (Muse Spark) and Alibaba
Cloud Model Studio (Qwen). :class:`BaseOpenAICompatibleInference` implements
that schema once; provider classes only override a handful of class
attributes (base URL, environment variable, model catalogue, quirks).

Example:
    >>> from rapidtools.models.openai_compat import BaseOpenAICompatibleInference
    >>>
    >>> class MyProvider(BaseOpenAICompatibleInference):
    ...     PROVIDER_NAME = 'My provider'
    ...     BASE_URL = 'https://llm.example.com/v1'
    ...     API_KEY_ENV = ('MY_PROVIDER_KEY',)
    ...     DEFAULT_MODEL = 'vision-1'
    ...     MODEL_CATALOG = [{'model_id': 'vision-1', 'family': 'Vision'}]
    >>>
    >>> model = MyProvider(api_key='secret')
    >>> model.run_inference('roof.jpg', 'Describe the roof.').text
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from requests.exceptions import HTTPError, RetryError

from rapidtools.config import get_configured_session

from .api_base import (
    API_REQUEST_TIMEOUT,
    BaseAPIInferenceModel,
    catalog_ids,
    resolve_api_key,
)
from .base import GenerationConfig, ModelOutput

logger = logging.getLogger(__name__)


class BaseOpenAICompatibleInference(BaseAPIInferenceModel):
    """
    Shared implementation of the OpenAI chat-completions protocol.

    Subclasses set the class attributes below; everything else (key
    resolution, image encoding, payload construction, response parsing and
    error handling) is inherited.

    Class attributes:
        BASE_URL: Root of the provider's OpenAI-compatible API (no trailing
            slash), e.g. ``'https://api.openai.com/v1'``.
        DEFAULT_MODEL: Model used when ``model_id`` is not given.
        MAX_TOKENS_FIELD: Name of the output-token limit field. OpenAI's
            current models require ``'max_completion_tokens'``; most
            compatible providers still expect ``'max_tokens'``.
        IMAGE_DETAIL: Value of ``image_url.detail`` (``'high'``/``'low'``/
            ``'auto'``) or ``None`` to omit the field.
        NO_TEMPERATURE_PREFIXES: Model-ID prefixes for which ``temperature``
            must not be sent (reasoning models reject it).
        EXTRA_HEADERS: Additional headers merged into every request.

    Example:
        >>> from rapidtools.models import OpenAIInference
        >>> model = OpenAIInference(api_key='sk-test', model_id='gpt-5.5')
        >>> out = model.run_inference(['before.jpg', 'after.jpg'], 'Compare.')
        >>> print(out.text)
    """

    PROVIDER_NAME: str = 'OpenAI-compatible API'
    BASE_URL: str = 'https://api.openai.com/v1'
    API_KEY_ENV: str | tuple[str, ...] = ('OPENAI_API_KEY',)
    DEFAULT_MODEL: str = ''
    MODEL_CATALOG: list[dict[str, str]] = []
    MAX_TOKENS_FIELD: str = 'max_tokens'
    IMAGE_DETAIL: str | None = 'high'
    NO_TEMPERATURE_PREFIXES: tuple[str, ...] = ()
    EXTRA_HEADERS: dict[str, str] = {}

    # Image formats accepted by OpenAI-style vision endpoints:
    MIME_MAP = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp',
        '.gif': 'image/gif',
    }

    def __init__(
        self,
        api_key: str | Path | None = None,
        model_id: str | None = None,
        max_workers: int = 10,
        max_retries: int = 3,
        system_instruction: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 2048,
        timeout: float = API_REQUEST_TIMEOUT,
        base_url: str | None = None,
    ):
        """
        Initialise the client and validate the chosen model.

        Args:
            api_key: Key string, path to a key file, or ``None`` to read the
                provider's environment variable.
            model_id: Model to call. Defaults to :attr:`DEFAULT_MODEL`.
            max_workers: Threads used by :meth:`run_batch`.
            max_retries: Retries for transient HTTP failures.
            system_instruction: Optional system prompt sent with every call.
            temperature: Sampling temperature (ignored for models that do not
                accept it).
            max_tokens: Maximum tokens generated per response.
            timeout: Seconds to wait for each response (default 60).
            base_url: Override the provider root URL (e.g. a regional
                endpoint or a proxy).

        Raises:
            ValueError: If no API key can be resolved.
        """
        super().__init__(
            max_retries=max_retries, max_workers=max_workers, timeout=timeout
        )

        self.api_key = self._resolve_api_key(api_key)
        self.model_id = model_id or self.DEFAULT_MODEL
        self.system_instruction = system_instruction
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = (base_url or self.BASE_URL).rstrip('/')

        self.session.headers.update(
            {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                **self.EXTRA_HEADERS,
            }
        )
        self._validate_model()

    # ----------------------------------------------------------- catalogue
    @classmethod
    def list_available_models(cls, api_key: str | Path | None = None) -> list[str]:
        """
        List the model IDs accessible with ``api_key``.

        Without a key (or when the API cannot be reached) the curated
        catalogue is returned instead.

        Args:
            api_key: Key string, key-file path, or ``None`` for the env var.

        Returns:
            list[str]: Sorted model IDs.

        Example:
            >>> from rapidtools.models import OpenAIInference
            >>> 'gpt-5.5' in OpenAIInference.list_available_models('sk-...')
            True
        """
        return cls._list_models(api_key)

    def _validate_model(self) -> None:
        """Warn (never fail) when the model is not in the provider's list."""
        if not self.api_key or not self.model_id:
            return
        try:
            available = self._list_models(self.api_key, self.base_url)
            if available and self.model_id not in available:
                logger.warning(
                    f"Model '{self.model_id}' is not in the {self.PROVIDER_NAME} "
                    'available models list.'
                )
        except Exception as e:  # noqa: BLE001 - validation is best effort
            logger.debug(f'Transient error during {self.PROVIDER_NAME} validation: {e}')

    @classmethod
    def _list_models(
        cls, api_key: str | Path | None = None, base_url: str | None = None
    ) -> list[str]:
        """
        Query ``GET {base_url}/models`` and return the sorted model IDs.

        Falls back to the curated :attr:`MODEL_CATALOG` when no key is given
        or the provider cannot be reached.

        Args:
            api_key: Key string, key file path, or ``None`` for the env var.
            base_url: Provider root; defaults to :attr:`BASE_URL`.

        Returns:
            list[str]: Model IDs.
        """
        try:
            key = resolve_api_key(api_key, cls.API_KEY_ENV, cls.PROVIDER_NAME)
        except (ValueError, FileNotFoundError):
            return catalog_ids(cls.MODEL_CATALOG)

        url = f'{(base_url or cls.BASE_URL).rstrip("/")}/models'
        try:
            session = get_configured_session()
            response = session.get(
                url,
                headers={'Authorization': f'Bearer {key}', **cls.EXTRA_HEADERS},
                timeout=10,
            )
            if response.status_code != 200:
                return catalog_ids(cls.MODEL_CATALOG)
            data = response.json()
            ids = sorted(m['id'] for m in data.get('data', []) if 'id' in m)
            return ids or catalog_ids(cls.MODEL_CATALOG)
        except Exception:  # noqa: BLE001 - network failures degrade gracefully
            return catalog_ids(cls.MODEL_CATALOG)

    # ------------------------------------------------------------- payload
    def _supports_temperature(self, model_id: str) -> bool:
        """Return ``False`` for models that reject the ``temperature`` field."""
        return not model_id.startswith(self.NO_TEMPERATURE_PREFIXES)

    def _image_block(self, img_data: dict[str, str]) -> dict[str, Any]:
        """Build an ``image_url`` content block from encoded image data."""
        image_url: dict[str, Any] = {
            'url': f'data:{img_data["mime_type"]};base64,{img_data["data"]}'
        }
        if self.IMAGE_DETAIL:
            image_url['detail'] = self.IMAGE_DETAIL
        return {'type': 'image_url', 'image_url': image_url}

    def _build_messages(
        self,
        image_inputs: list[str | Path],
        prompt_str: str,
        system_instruction: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """
        Assemble the ``messages`` array (system + one user turn).

        Returns ``None`` when neither text nor a readable image is available.
        """
        user_content: list[dict[str, Any]] = []
        if prompt_str:
            user_content.append({'type': 'text', 'text': prompt_str})
        for img_input in image_inputs:
            img_data = self._fetch_and_encode_image(img_input)
            if img_data:
                user_content.append(self._image_block(img_data))
        if not user_content:
            return None

        messages: list[dict[str, Any]] = []
        system = system_instruction or self.system_instruction
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': user_content})
        return messages

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> dict[str, Any]:
        """Build the JSON body for ``/chat/completions``."""
        payload: dict[str, Any] = {
            'model': self.model_id,
            'messages': messages,
            self.MAX_TOKENS_FIELD: max_tokens,
        }
        if self._supports_temperature(self.model_id):
            payload['temperature'] = temperature
        if json_mode:
            payload['response_format'] = {'type': 'json_object'}
        return payload

    def _parse_response(
        self, result_json: dict[str, Any], log_ctx: str
    ) -> ModelOutput | None:
        """Extract the assistant text from a chat-completions response."""
        try:
            choice = result_json['choices'][0]
            content = choice['message']['content']
            finish_reason = choice.get('finish_reason')
        except (KeyError, IndexError, TypeError):
            logger.error(f'{log_ctx} Unexpected response format: {result_json}')
            return None

        # Some providers return a list of content parts instead of a string.
        if isinstance(content, list):
            content = ''.join(
                part.get('text', '') for part in content if isinstance(part, dict)
            )

        if finish_reason == 'content_filter':
            logger.warning(f'{log_ctx} Blocked by {self.PROVIDER_NAME} content filter.')
            return None
        if finish_reason == 'length':
            logger.warning(f'{log_ctx} Response truncated due to max_tokens limit.')

        return ModelOutput(text=content, raw_response=result_json)

    # ----------------------------------------------------------- inference
    def run_inference(
        self,
        image_inputs: str | Path | list[str | Path],
        prompt: str | Path,
        json_mode: bool = False,
        max_retries: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        config: GenerationConfig | None = None,
    ) -> ModelOutput | None:
        """
        Send images and a prompt to ``/chat/completions``.

        Args:
            image_inputs: One path/URL or a list of them. Unreadable images
                are skipped with a logged error.
            prompt: Prompt text or path to a prompt file.
            json_mode: Ask the provider for a JSON object response.
            max_retries: Per-call override of the session retry count.
            temperature: Per-call sampling temperature override.
            max_tokens: Per-call output-token limit override.
            config: A :class:`~rapidtools.models.GenerationConfig`; explicit
                keyword arguments take precedence over it.

        Returns:
            ModelOutput | None: The response text and raw JSON, or ``None``
            when the request failed, was filtered, or nothing could be sent.

        Example:
            >>> from rapidtools.models import OpenAIInference
            >>> model = OpenAIInference(api_key='sk-test')
            >>> out = model.run_inference(
            ...     'house.jpg', 'Return JSON with a damage_level key.', json_mode=True
            ... )
            >>> out.text
            '{"damage_level": "moderate"}'
        """
        prompt_str = self._resolve_prompt(prompt)
        log_ctx = f"[Prompt snippet: '{prompt_str[:30]}...']"

        if image_inputs is None:
            image_inputs = []
        elif not isinstance(image_inputs, list):
            image_inputs = [image_inputs]

        gen = self._resolve_generation(
            config, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
        )
        messages = self._build_messages(
            image_inputs, prompt_str, gen.system_instruction
        )
        if messages is None:
            logger.error(f'{log_ctx} No valid text or images were loaded.')
            return None

        payload = self._build_payload(
            messages, gen.temperature, gen.max_tokens, gen.json_mode
        )
        url = f'{self.base_url}/chat/completions'

        session_to_use = self.session
        should_close_session = False
        if max_retries is not None:
            session_to_use = get_configured_session(retries=max_retries)
            session_to_use.headers.update(self.session.headers)
            should_close_session = True

        try:
            response = session_to_use.post(url, json=payload, timeout=self.timeout)
            if not response.ok:
                try:
                    err_msg = (
                        response.json().get('error', {}).get('message', 'Unknown error')
                    )
                    logger.error(f'{log_ctx} {self.PROVIDER_NAME} error: {err_msg}')
                except Exception:  # noqa: BLE001 - body may not be JSON
                    pass
            response.raise_for_status()
            return self._parse_response(response.json(), log_ctx)
        except RetryError:
            logger.error(f'{log_ctx} Max retries exceeded.')
        except HTTPError as e:
            logger.error(f'{log_ctx} HTTP Error: {e.response.status_code}')
        except Exception as e:  # noqa: BLE001 - surfaced as a failed asset
            logger.error(f'{log_ctx} Unexpected error: {e}')
        finally:
            if should_close_session:
                session_to_use.close()

        return None
