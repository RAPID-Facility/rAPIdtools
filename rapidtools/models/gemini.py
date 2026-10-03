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
Google Gemini API wrapper (raw HTTP, ``generateContent``).

Example:
    >>> from rapidtools.models import GeminiInference
    >>> model = GeminiInference(api_key='AIza...', model_id='gemini-3.8-flash')
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
    GEMINI_DEFAULT_MODEL,
    GEMINI_MODEL_CATALOG,
    PROVIDERS,
)

logger = logging.getLogger(__name__)

GEMINI_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta'


class GeminiInference(BaseAPIInferenceModel):
    """
    Google Gemini ``generateContent`` client for image + text prompts.

    The key is read from ``api_key`` (string or key-file path) or the
    ``GOOGLE_API_KEY`` environment variable. Safety filters are set to
    ``BLOCK_NONE`` because damage imagery is frequently mis-flagged.

    Example:
        >>> from rapidtools.models import GeminiInference
        >>> GeminiInference.list_known_models()[0]
        'gemini-3.8-flash'
        >>> model = GeminiInference(api_key='AIza...')
        >>> out = model.run_inference(
        ...     'house.jpg', 'Return JSON with a chs_level key (0-5).', json_mode=True
        ... )
        >>> out.text
        '{"chs_level": 3}'
    """

    INFO = PROVIDERS['gemini']

    PROVIDER_NAME = 'Gemini'
    API_KEY_ENV = ('GOOGLE_API_KEY', 'GEMINI_API_KEY')
    MODEL_CATALOG = GEMINI_MODEL_CATALOG

    MIME_MAP = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp',
        '.heic': 'image/heic',
        '.heif': 'image/heif',
        '.tif': 'image/tiff',
        '.tiff': 'image/tiff',
        '.bmp': 'image/bmp',
    }

    SAFETY_SETTINGS = [
        {'category': cat, 'threshold': 'BLOCK_NONE'}
        for cat in [
            'HARM_CATEGORY_HARASSMENT',
            'HARM_CATEGORY_HATE_SPEECH',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT',
            'HARM_CATEGORY_DANGEROUS_CONTENT',
        ]
    ]

    def __init__(
        self,
        api_key: str | Path | None = None,
        model_id: str = GEMINI_DEFAULT_MODEL,
        max_workers: int = 10,
        max_retries: int = 3,
        system_instruction: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 2048,
        timeout: float = API_REQUEST_TIMEOUT,
    ):
        """
        Initialise the client and validate the chosen model.

        Args:
            api_key: Key string, key-file path, or ``None`` for
                ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY``.
            model_id: Gemini model ID (with or without the ``models/``
                prefix). Defaults to ``'gemini-3.8-flash'``.
            max_workers: Threads used by :meth:`run_batch`.
            max_retries: Retries for transient HTTP failures.
            system_instruction: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens per response.
            timeout: Seconds to wait for each response (default 60).

        Raises:
            ValueError: If no API key can be resolved.
            FileNotFoundError: If ``api_key`` is a ``Path`` that does not exist.
        """
        super().__init__(
            max_retries=max_retries, max_workers=max_workers, timeout=timeout
        )

        self.api_key = self._resolve_api_key(api_key)
        self.model_id = model_id.replace('models/', '')
        self.system_instruction = system_instruction
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.session.headers.update({'x-goog-api-key': self.api_key})
        self._validate_model()

    def _validate_model(self) -> None:
        """Warn (never fail) when the model is not in Google's list."""
        if not self.api_key or not self.model_id:
            return
        try:
            available_models = self.list_available_models(self.api_key)
            available_ids = {m.replace('models/', '') for m in available_models}
            if available_ids and self.model_id not in available_ids:
                logger.warning(
                    f"Model '{self.model_id}' is not in the available models list."
                )
        except Exception as e:  # noqa: BLE001 - validation is best effort
            logger.debug(f'Transient error during model validation: {e}')

    @classmethod
    def list_available_models(cls, api_key: str | Path | None = None) -> list[str]:
        """
        List Gemini models that support ``generateContent``.

        Without a key (or when the API cannot be reached) the curated
        catalogue is returned instead.

        Args:
            api_key: Key string, key-file path, or ``None`` for the env var.

        Returns:
            list[str]: Sorted model IDs without the ``models/`` prefix.

        Example:
            >>> from rapidtools.models import GeminiInference
            >>> ids = GeminiInference.list_available_models('AIza...')
            >>> 'gemini-3.8-flash' in ids
            True
        """
        try:
            resolved_key = resolve_api_key(api_key, cls.API_KEY_ENV, cls.PROVIDER_NAME)
        except (ValueError, FileNotFoundError):
            return catalog_ids(GEMINI_MODEL_CATALOG)

        url = f'{GEMINI_BASE_URL}/models'
        try:
            session = get_configured_session()
            response = session.get(
                url, headers={'x-goog-api-key': resolved_key}, timeout=10
            )
            if response.status_code != 200:
                return catalog_ids(GEMINI_MODEL_CATALOG)

            data = response.json()
            models = [
                m['name'].replace('models/', '')
                for m in data.get('models', [])
                if 'generateContent' in m.get('supportedGenerationMethods', [])
            ]
            return sorted(models) or catalog_ids(GEMINI_MODEL_CATALOG)
        except Exception:  # noqa: BLE001 - network failures degrade gracefully
            return catalog_ids(GEMINI_MODEL_CATALOG)

    def _build_payload(
        self,
        contents_parts: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        system_instruction: str | None = None,
    ) -> dict[str, Any]:
        """Construct the ``generateContent`` request body."""
        payload: dict[str, Any] = {
            'contents': [{'parts': contents_parts}],
            'safetySettings': self.SAFETY_SETTINGS,
            'generationConfig': {
                'temperature': temperature,
                'maxOutputTokens': max_tokens,
                'responseMimeType': 'application/json' if json_mode else 'text/plain',
            },
        }
        system = system_instruction or self.system_instruction
        if system:
            payload['systemInstruction'] = {'parts': [{'text': system}]}
        return payload

    @staticmethod
    def _extract_text(result_json: dict[str, Any]) -> str | None:
        """Join the text parts of the first candidate, if any."""
        try:
            parts = result_json['candidates'][0]['content']['parts']
        except (KeyError, IndexError, TypeError):
            return None
        texts = [p['text'] for p in parts if isinstance(p, dict) and 'text' in p]
        return ''.join(texts) if texts else None

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
        Send images and a prompt to ``generateContent``.

        Args:
            image_inputs: One path/URL or a list of them. Unreadable images
                are skipped with a logged error; text-only prompts are
                allowed by passing ``[]``.
            prompt: Prompt text or path to a prompt file.
            json_mode: Request ``application/json`` output.
            max_retries: Per-call override of the session retry count.
            temperature: Per-call temperature override.
            max_tokens: Per-call output-token limit override.
            config: A :class:`~rapidtools.models.GenerationConfig`; explicit
                keyword arguments take precedence over it.

        Returns:
            ModelOutput | None: The response text and raw JSON, or ``None``
            when the request failed, was blocked by safety filters, or
            nothing could be sent.

        Example:
            >>> from rapidtools.models import GeminiInference
            >>> model = GeminiInference(api_key='AIza...')
            >>> model.run_inference(['a.jpg', 'b.jpg'], 'Which roof is worse?').text
            'The roof in the second image shows more severe damage.'
        """
        prompt_str = self._resolve_prompt(prompt)
        log_ctx = f"[Prompt snippet: '{prompt_str[:30]}...']"

        if image_inputs is None:
            image_inputs = []
        elif not isinstance(image_inputs, list):
            image_inputs = [image_inputs]

        contents_parts: list[dict[str, Any]] = []
        for img_input in image_inputs:
            img_data = self._fetch_and_encode_image(img_input)
            if img_data:
                contents_parts.append({'inline_data': img_data})

        if image_inputs and not contents_parts:
            logger.error(f'{log_ctx} No valid images were loaded.')
            return None
        if not prompt_str and not contents_parts:
            logger.error(f'{log_ctx} No valid text or images were loaded.')
            return None

        contents_parts.append({'text': prompt_str})

        gen = self._resolve_generation(
            config, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
        )
        payload = self._build_payload(
            contents_parts,
            gen.temperature,
            gen.max_tokens,
            gen.json_mode,
            gen.system_instruction,
        )

        url = f'{GEMINI_BASE_URL}/models/{self.model_id}:generateContent'
        headers = {'x-goog-api-key': self.api_key}

        session_to_use = self.session
        should_close_session = False
        if max_retries is not None:
            session_to_use = api_session(max_retries)
            should_close_session = True

        try:
            response = session_to_use.post(
                url, json=payload, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
            result_json = response.json()

            extracted_text = self._extract_text(result_json)
            if extracted_text is not None:
                return ModelOutput(text=extracted_text, raw_response=result_json)

            block_reason = result_json.get('promptFeedback', {}).get('blockReason')
            if block_reason:
                reason = f'Blocked by safety filters. Reason: {block_reason}'
                logger.warning(f'{log_ctx} {reason}')
                return ModelOutput.failure(reason, raw_response=result_json)

            candidates = result_json.get('candidates', [])
            if candidates:
                finish_reason = candidates[0].get('finishReason')
                if finish_reason != 'STOP':
                    reason = (
                        'Generation halted unexpectedly. '
                        f'Finish Reason: {finish_reason}'
                    )
                    logger.warning(f'{log_ctx} {reason}')
                    return ModelOutput.failure(reason, raw_response=result_json)

            logger.error(f'{log_ctx} Unexpected response format: {result_json}')
            return None

        except RetryError:
            logger.error(f'{log_ctx} Max retries exceeded.')
        except HTTPError as e:
            body = e.response.text if e.response is not None else ''
            logger.error(f'{log_ctx} HTTP Error: {e} | Response: {body}')
            status = e.response.status_code if e.response is not None else None
            if status is not None and not is_retryable_status(status):
                return ModelOutput.failure(f'HTTP {status}', retryable=False)
        except Exception as e:  # noqa: BLE001 - surfaced as a failed asset
            logger.error(f'{log_ctx} Unexpected error: {e}')
        finally:
            if should_close_session:
                session_to_use.close()

        return None
