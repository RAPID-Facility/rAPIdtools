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
# 02-22-2026

import logging
import os
from pathlib import Path

from requests.exceptions import HTTPError, RetryError

from rapidtools.config import REQUESTS_TIMEOUT_VAL, get_configured_session

from .api_base import BaseAPIInferenceModel
from .base import ModelOutput

GEMINI_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta'
GEMINI_DEFAULT_MODEL = 'gemini-3.0-flash'


class GeminiInference(BaseAPIInferenceModel):
    """Google Gemini API implementation."""

    MIME_MAP = {
        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
        '.webp': 'image/webp', '.heic': 'image/heic', '.heif': 'image/heif',
        '.tif': 'image/tiff', '.tiff': 'image/tiff', '.bmp': 'image/bmp',
    }

    SAFETY_SETTINGS = [
        {'category': cat, 'threshold': 'BLOCK_NONE'}
        for cat in [
            'HARM_CATEGORY_HARASSMENT',
            'HARM_CATEGORY_HATE_SPEECH',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT',
            'HARM_CATEGORY_DANGEROUS_CONTENT'
        ]
    ]

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = GEMINI_DEFAULT_MODEL,
        max_workers: int = 10,
        max_retries: int = 3,
        system_instruction: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 2048
    ):
        super().__init__(max_retries=max_retries, max_workers=max_workers)

        self.api_key = self._resolve_api_key(api_key)
        self.model_id = model_id
        self.system_instruction = system_instruction
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.session.headers.update({'x-goog-api-key': self.api_key})
        self._validate_model()

    def _resolve_api_key(self, key_input: str | None) -> str:
        raw_key = (key_input or os.environ.get('GOOGLE_API_KEY', '')).strip()
        if not raw_key:
            raise ValueError('API Key is missing.')
        potential_path = Path(raw_key)
        resolved_key = raw_key
        try:
            if potential_path.is_file():
                resolved_key = potential_path.read_text(encoding='utf-8').strip()
        except OSError:
            pass
        resolved_key = resolved_key.strip("'\" \n\t")
        if not resolved_key:
            raise ValueError('Resolved API key is empty.')
        return resolved_key

    def _validate_model(self) -> None:
        if not self.api_key or not self.model_id:
            return
        target_model = self.model_id.replace('models/', '')
        try:
            available_models = self.list_available_models(self.api_key)
            available_ids = {m.replace('models/', '') for m in available_models}
            if target_model not in available_ids and available_ids:
                logging.warning(f"Model '{self.model_id}' is not in the available models list.")
        except Exception as e:
            logging.debug(f'Transient error during model validation: {e}')

    @staticmethod
    def list_available_models(api_key: str | None = None) -> list[str]:
        if not api_key:
            return []
        url = f'{GEMINI_BASE_URL}/models'
        try:
            session = get_configured_session()
            response = session.get(url, headers={'x-goog-api-key': api_key}, timeout=10)
            if response.status_code != 200:
                return []
            data = response.json()
            models = [
                m['name'].replace('models/', '')
                for m in data.get('models', [])
                if 'generateContent' in m.get('supportedGenerationMethods', [])
            ]
            return sorted(models)
        except Exception:
            return []

    def run_inference(
        self,
        image_inputs: str | Path | list[str | Path],
        prompt: str | Path,
        json_mode: bool = False,
        max_retries: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None
    ) -> ModelOutput | None:

        prompt_str = self._resolve_prompt(prompt)
        log_ctx = f"[Prompt snippet: '{prompt_str[:30]}...']"

        if not isinstance(image_inputs, list):
            image_inputs = [image_inputs]

        contents_parts = []
        for img_input in image_inputs:
            img_data = self._fetch_and_encode_image(img_input)
            if img_data:
                contents_parts.append({'inline_data': img_data})

        if not contents_parts:
            logging.error(f'{log_ctx} No valid images were loaded.')
            return None

        contents_parts.append({'text': prompt_str})

        final_temp = temperature if temperature is not None else self.temperature
        final_tokens = max_tokens if max_tokens is not None else self.max_tokens

        payload = {
            'contents': [{'parts': contents_parts}],
            'safetySettings': self.SAFETY_SETTINGS,
            'generationConfig': {
                'temperature': final_temp,
                'maxOutputTokens': final_tokens,
                'responseMimeType': 'application/json' if json_mode else 'text/plain'
            }
        }

        if self.system_instruction:
            payload['systemInstruction'] = {'parts': [{'text': self.system_instruction}]}

        url = f'{GEMINI_BASE_URL}/models/{self.model_id}:generateContent'
        headers = {'x-goog-api-key': self.api_key}

        session_to_use = self.session
        should_close_session = False
        if max_retries is not None:
            session_to_use = get_configured_session(retries=max_retries)
            should_close_session = True

        try:
            response = session_to_use.post(url, json=payload, headers=headers, timeout=REQUESTS_TIMEOUT_VAL)
            response.raise_for_status()
            result_json = response.json()

            try:
                extracted_text = result_json['candidates'][0]['content']['parts'][0]['text']
                # Return the unified ModelOutput!
                return ModelOutput(
                    text=extracted_text,
                    raw_response=result_json
                )
            except (KeyError, IndexError):
                block_reason = result_json.get('promptFeedback', {}).get('blockReason')
                if block_reason:
                    logging.warning(f'{log_ctx} Blocked by safety filters. Reason: {block_reason}')
                    return None

                candidates = result_json.get('candidates', [])
                if candidates:
                    finish_reason = candidates[0].get('finishReason')
                    if finish_reason != 'STOP':
                        logging.warning(f'{log_ctx} Generation halted unexpectedly. Finish Reason: {finish_reason}')
                        return None

                logging.error(f'{log_ctx} Unexpected response format: {result_json}')
                return None

        except RetryError:
            logging.error(f'{log_ctx} Max retries exceeded.')
        except HTTPError as e:
            logging.error(f'{log_ctx} HTTP Error: {e} | Response: {e.response.text}')
        except Exception as e:
            logging.error(f'{log_ctx} Unexpected error: {e}')
        finally:
            if should_close_session:
                session_to_use.close()

        return None
