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

ANTHROPIC_BASE_URL = 'https://api.anthropic.com/v1'
ANTHROPIC_DEFAULT_MODEL = 'claude-3-7-sonnet-latest'


class ClaudeInference(BaseAPIInferenceModel):
    """Anthropic Claude API implementation."""

    # Anthropic specifically supports these image formats
    MIME_MAP = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp',
        '.gif': 'image/gif'
    }

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = ANTHROPIC_DEFAULT_MODEL,
        max_workers: int = 10,
        max_retries: int = 3,
        system_instruction: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 2048  # Note: Anthropic REQUIRES this field in the payload
    ):
        super().__init__(max_retries=max_retries, max_workers=max_workers)

        self.api_key = self._resolve_api_key(api_key)
        self.model_id = model_id
        self.system_instruction = system_instruction
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Anthropic requires specific versioning headers
        self.session.headers.update({
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'
        })

        self._validate_model()

    def _resolve_api_key(self, key_input: str | None) -> str:
        raw_key = (key_input or os.environ.get('ANTHROPIC_API_KEY', '')).strip()
        if not raw_key:
            raise ValueError('Anthropic API Key is missing. Pass it or set ANTHROPIC_API_KEY.')

        potential_path = Path(raw_key)
        resolved_key = raw_key
        try:
            if potential_path.is_file():
                resolved_key = potential_path.read_text(encoding='utf-8').strip()
        except OSError:
            pass

        resolved_key = resolved_key.strip("'\" \n\t")
        if not resolved_key:
            raise ValueError('Resolved Anthropic API key is empty.')
        return resolved_key

    def _validate_model(self) -> None:
        if not self.api_key or not self.model_id:
            return
        try:
            available_models = self.list_available_models(self.api_key)
            if self.model_id not in available_models and available_models:
                logging.warning(f"Model '{self.model_id}' is not in the Anthropic available models list.")
        except Exception as e:
            logging.debug(f'Transient error during Anthropic model validation: {e}')

    @staticmethod
    def list_available_models(api_key: str | None = None) -> list[str]:
        if not api_key:
            return []
        url = f'{ANTHROPIC_BASE_URL}/models'
        try:
            session = get_configured_session()
            response = session.get(
                url,
                headers={
                    'x-api-key': api_key,
                    'anthropic-version': '2023-06-01'
                },
                timeout=10
            )
            if response.status_code != 200:
                return []
            data = response.json()
            # Anthropic returns a list of models under the 'data' key
            return sorted([m['id'] for m in data.get('data', [])])
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
        """Runs Anthropic Inference and returns the universal ModelOutput."""

        prompt_str = self._resolve_prompt(prompt)
        log_ctx = f"[Prompt snippet: '{prompt_str[:30]}...']"

        if not isinstance(image_inputs, list):
            image_inputs = [image_inputs]

        # 1. Build the user message content block
        user_content = []

        # Add the images using our Parent's fetcher
        for img_input in image_inputs:
            img_data = self._fetch_and_encode_image(img_input)
            if img_data:
                # Anthropic's specific image schema
                user_content.append({
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': img_data['mime_type'],
                        'data': img_data['data']
                    }
                })

        # Anthropic expects text AFTER the images
        if prompt_str:
            user_content.append({'type': 'text', 'text': prompt_str})

        if not user_content:
            logging.error(f'{log_ctx} No valid text or images were loaded.')
            return None

        # 2. Resolve configs
        final_temp = temperature if temperature is not None else self.temperature
        final_tokens = max_tokens if max_tokens is not None else self.max_tokens

        # 3. Construct payload (Anthropic puts system prompt at the root, not in messages)
        payload = {
            'model': self.model_id,
            'messages': [{'role': 'user', 'content': user_content}],
            'max_tokens': final_tokens,
            'temperature': final_temp
        }

        if self.system_instruction:
            payload['system'] = self.system_instruction

        # Optional: Append a strong JSON hint to the system prompt if json_mode is requested
        # (Anthropic does not have a strict "response_format" flag like OpenAI)
        if json_mode:
            json_hint = 'You must respond with ONLY valid JSON and no other conversational text.'
            if 'system' in payload:
                payload['system'] += f'\n\n{json_hint}'
            else:
                payload['system'] = json_hint

        url = f'{ANTHROPIC_BASE_URL}/messages'

        session_to_use = self.session
        should_close_session = False
        if max_retries is not None:
            session_to_use = get_configured_session(retries=max_retries)
            session_to_use.headers.update(self.session.headers)
            should_close_session = True

        # 4. Execute the HTTP Request
        try:
            response = session_to_use.post(url, json=payload, timeout=REQUESTS_TIMEOUT_VAL)

            if not response.ok:
                try:
                    err_msg = response.json().get('error', {}).get('message', 'Unknown error')
                    logging.error(f'{log_ctx} Anthropic API Error: {err_msg}')
                except Exception:
                    pass

            response.raise_for_status()
            result_json = response.json()

            try:
                # Claude returns an array of content blocks (usually just one text block)
                extracted_text = result_json['content'][0]['text']
                stop_reason = result_json.get('stop_reason')

                if stop_reason == 'max_tokens':
                    logging.warning(f'{log_ctx} Response truncated due to max_tokens limit.')

                # Return the unified ModelOutput!
                return ModelOutput(
                    text=extracted_text,
                    raw_response=result_json
                )

            except (KeyError, IndexError):
                logging.error(f'{log_ctx} Unexpected response format: {result_json}')
                return None

        except RetryError:
            logging.error(f'{log_ctx} Max retries exceeded.')
        except HTTPError as e:
            logging.error(f'{log_ctx} HTTP Error: {e.response.status_code}')
        except Exception as e:
            logging.error(f'{log_ctx} Unexpected error: {e}')
        finally:
            if should_close_session:
                session_to_use.close()

        return None
