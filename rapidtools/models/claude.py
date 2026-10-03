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
Anthropic Claude API wrapper (raw HTTP, Messages API).

Example:
    >>> from rapidtools.models import ClaudeInference
    >>> model = ClaudeInference(api_key='sk-ant-...', model_id='claude-opus-5')
    >>> out = model.run_inference('roof.jpg', 'Describe the roof damage.')
    >>> print(out.text)
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
    api_session,
    catalog_ids,
    is_retryable_status,
    resolve_api_key,
)
from .base import GenerationConfig, ModelOutput
from .catalogs import (
    ANTHROPIC_DEFAULT_MODEL,
    ANTHROPIC_MODEL_CATALOG,
    ANTHROPIC_NO_TEMPERATURE_PREFIXES,
    PROVIDERS,
)

logger = logging.getLogger(__name__)

ANTHROPIC_BASE_URL = 'https://api.anthropic.com/v1'
ANTHROPIC_API_VERSION = '2023-06-01'


class ClaudeInference(BaseAPIInferenceModel):
    """
    Anthropic Claude Messages API client for image + text prompts.

    The key is read from ``api_key`` (string or key-file path) or the
    ``ANTHROPIC_API_KEY`` environment variable. Thinking blocks returned by
    current models are skipped; only text blocks are concatenated into
    :attr:`ModelOutput.text`. A ``refusal`` stop reason yields ``None``.

    Example:
        >>> from rapidtools.models import ClaudeInference
        >>> ClaudeInference.list_known_models()[:3]
        ['claude-fable-5-1', 'claude-fable-5', 'claude-opus-5']
        >>> model = ClaudeInference(api_key='sk-ant-...')
        >>> out = model.run_inference(
        ...     ['front.jpg', 'back.jpg'],
        ...     'Return JSON with keys damage_level and justification.',
        ...     json_mode=True,
        ... )
        >>> out.text
        '{"damage_level": 2, "justification": "minor roof damage"}'
    """

    INFO = PROVIDERS['claude']

    PROVIDER_NAME = 'Anthropic'
    API_KEY_ENV = ('ANTHROPIC_API_KEY',)
    MODEL_CATALOG = ANTHROPIC_MODEL_CATALOG

    # Anthropic specifically supports these image formats
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
        model_id: str = ANTHROPIC_DEFAULT_MODEL,
        max_workers: int = 10,
        max_retries: int = 3,
        system_instruction: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 2048,  # Anthropic REQUIRES this field in the payload
        timeout: float = API_REQUEST_TIMEOUT,
    ):
        """
        Initialise the client and validate the chosen model.

        Args:
            api_key: Key string, key-file path, or ``None`` for
                ``ANTHROPIC_API_KEY``.
            model_id: Claude model ID. Defaults to ``'claude-opus-5'``.
            max_workers: Threads used by :meth:`run_batch`.
            max_retries: Retries for transient HTTP failures.
            system_instruction: Optional system prompt.
            temperature: Sampling temperature; only sent to models that
                accept it (Opus/Sonnet 4.6, Haiku 4.5 and older).
            max_tokens: Maximum tokens generated per response (required by
                the Messages API).
            timeout: Seconds to wait for each response (default 60).

        Raises:
            ValueError: If no API key can be resolved.
        """
        super().__init__(
            max_retries=max_retries, max_workers=max_workers, timeout=timeout
        )

        self.api_key = self._resolve_api_key(api_key)
        self.model_id = model_id
        self.system_instruction = system_instruction
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Anthropic requires specific versioning headers
        self.session.headers.update(
            {
                'x-api-key': self.api_key,
                'anthropic-version': ANTHROPIC_API_VERSION,
                'content-type': 'application/json',
            }
        )

        self._validate_model()

    def _validate_model(self) -> None:
        """Warn (never fail) when the model is not in Anthropic's list."""
        if not self.api_key or not self.model_id:
            return
        try:
            available_models = self.list_available_models(self.api_key)
            if available_models and self.model_id not in available_models:
                logger.warning(
                    f"Model '{self.model_id}' is not in the Anthropic available "
                    'models list.'
                )
        except Exception as e:  # noqa: BLE001 - validation is best effort
            logger.debug(f'Transient error during Anthropic model validation: {e}')

    @classmethod
    def list_available_models(cls, api_key: str | Path | None = None) -> list[str]:
        """
        List the Claude model IDs accessible with ``api_key``.

        Without a key (or when the API cannot be reached) the curated
        catalogue is returned instead.

        Args:
            api_key: Key string, key-file path, or ``None`` for the env var.

        Returns:
            list[str]: Sorted model IDs.

        Example:
            >>> from rapidtools.models import ClaudeInference
            >>> ids = ClaudeInference.list_available_models('sk-ant-...')
            >>> 'claude-opus-5' in ids
            True
        """
        try:
            key = resolve_api_key(api_key, cls.API_KEY_ENV, cls.PROVIDER_NAME)
        except (ValueError, FileNotFoundError):
            return catalog_ids(ANTHROPIC_MODEL_CATALOG)
        url = f'{ANTHROPIC_BASE_URL}/models'
        try:
            session = get_configured_session()
            response = session.get(
                url,
                headers={
                    'x-api-key': key,
                    'anthropic-version': ANTHROPIC_API_VERSION,
                },
                timeout=10,
            )
            if response.status_code != 200:
                return catalog_ids(ANTHROPIC_MODEL_CATALOG)
            data = response.json()
            # Anthropic returns a list of models under the 'data' key
            ids = sorted(m['id'] for m in data.get('data', []) if 'id' in m)
            return ids or catalog_ids(ANTHROPIC_MODEL_CATALOG)
        except Exception:  # noqa: BLE001 - network failures degrade gracefully
            return catalog_ids(ANTHROPIC_MODEL_CATALOG)

    @staticmethod
    def supports_temperature(model_id: str) -> bool:
        """
        Return whether ``model_id`` accepts the ``temperature`` parameter.

        Example:
            >>> from rapidtools.models import ClaudeInference
            >>> ClaudeInference.supports_temperature('claude-opus-5')
            False
            >>> ClaudeInference.supports_temperature('claude-haiku-4-5')
            True
        """
        return not model_id.startswith(ANTHROPIC_NO_TEMPERATURE_PREFIXES)

    def _build_payload(
        self,
        user_content: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        system_instruction: str | None = None,
    ) -> dict[str, Any]:
        """Construct the Messages API request body."""
        payload: dict[str, Any] = {
            'model': self.model_id,
            'messages': [{'role': 'user', 'content': user_content}],
            'max_tokens': max_tokens,
        }
        if self.supports_temperature(self.model_id):
            payload['temperature'] = temperature

        # Anthropic puts the system prompt at the root, not in messages.
        system = system_instruction or self.system_instruction
        if system:
            payload['system'] = system

        # Anthropic has no strict "response_format" flag; use a system hint.
        if json_mode:
            json_hint = (
                'You must respond with ONLY valid JSON and no other '
                'conversational text.'
            )
            if 'system' in payload:
                payload['system'] = f'{payload["system"]}\n\n{json_hint}'
            else:
                payload['system'] = json_hint
        return payload

    @staticmethod
    def _extract_text(result_json: dict[str, Any]) -> str | None:
        """Join the text blocks of a Messages API response (skipping thinking)."""
        blocks = result_json.get('content')
        if not isinstance(blocks, list):
            return None
        texts = [
            block.get('text', '')
            for block in blocks
            if isinstance(block, dict) and block.get('type') == 'text'
        ]
        if not texts:
            return None
        return ''.join(texts)

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
        Send images and a prompt to the Messages API.

        Args:
            image_inputs: One path/URL or a list of them. Unreadable images
                are skipped with a logged error.
            prompt: Prompt text or path to a prompt file.
            json_mode: Add a system instruction demanding JSON-only output.
            max_retries: Per-call override of the session retry count.
            temperature: Per-call temperature override (ignored by models
                that do not accept it).
            max_tokens: Per-call output-token limit override.
            config: A :class:`~rapidtools.models.GenerationConfig`; explicit
                keyword arguments take precedence over it.

        Returns:
            ModelOutput | None: The response text and raw JSON, or ``None``
            when the request failed, was refused, or nothing could be sent.

        Example:
            >>> from rapidtools.models import ClaudeInference
            >>> model = ClaudeInference(api_key='sk-ant-...')
            >>> model.run_inference('house.jpg', 'Is the house standing?').text
            'Yes, the house is standing with visible roof damage.'
        """
        prompt_str = self._resolve_prompt(prompt)
        log_ctx = f"[Prompt snippet: '{prompt_str[:30]}...']"

        if image_inputs is None:
            image_inputs = []
        elif not isinstance(image_inputs, list):
            image_inputs = [image_inputs]

        # 1. Build the user message content block (images first, then text)
        user_content: list[dict[str, Any]] = []
        for img_input in image_inputs:
            img_data = self._fetch_and_encode_image(img_input)
            if img_data:
                user_content.append(
                    {
                        'type': 'image',
                        'source': {
                            'type': 'base64',
                            'media_type': img_data['mime_type'],
                            'data': img_data['data'],
                        },
                    }
                )
        if prompt_str:
            user_content.append({'type': 'text', 'text': prompt_str})

        if not user_content:
            logger.error(f'{log_ctx} No valid text or images were loaded.')
            return None

        # 2. Resolve configs and build the payload
        gen = self._resolve_generation(
            config, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
        )
        payload = self._build_payload(
            user_content,
            gen.temperature,
            gen.max_tokens,
            gen.json_mode,
            gen.system_instruction,
        )
        url = f'{ANTHROPIC_BASE_URL}/messages'

        session_to_use = self.session
        should_close_session = False
        if max_retries is not None:
            session_to_use = api_session(max_retries)
            session_to_use.headers.update(self.session.headers)
            should_close_session = True

        # 3. Execute the HTTP request
        try:
            response = session_to_use.post(url, json=payload, timeout=self.timeout)

            if not response.ok:
                try:
                    err_msg = (
                        response.json().get('error', {}).get('message', 'Unknown error')
                    )
                    logger.error(f'{log_ctx} Anthropic API Error: {err_msg}')
                except Exception:  # noqa: BLE001 - body may not be JSON
                    pass

            response.raise_for_status()
            result_json = response.json()

            stop_reason = result_json.get('stop_reason')
            if stop_reason == 'refusal':
                details = result_json.get('stop_details') or {}
                reason = (
                    f'Request refused by Claude (category: {details.get("category")}).'
                )
                logger.warning(f'{log_ctx} {reason}')
                return ModelOutput.failure(reason, raw_response=result_json)
            if stop_reason == 'max_tokens':
                logger.warning(f'{log_ctx} Response truncated due to max_tokens limit.')

            extracted_text = self._extract_text(result_json)
            if extracted_text is None:
                logger.error(f'{log_ctx} Unexpected response format: {result_json}')
                return None
            return ModelOutput(text=extracted_text, raw_response=result_json)

        except RetryError:
            logger.error(f'{log_ctx} Max retries exceeded.')
        except HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            logger.error(f'{log_ctx} HTTP Error: {status}')
            if status is not None and not is_retryable_status(status):
                return ModelOutput.failure(f'HTTP {status}', retryable=False)
        except Exception as e:  # noqa: BLE001 - surfaced as a failed asset
            logger.error(f'{log_ctx} Unexpected error: {e}')
        finally:
            if should_close_session:
                session_to_use.close()

        return None
