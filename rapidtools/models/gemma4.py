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
Google Gemma 4 vision-language models via Transformers.

Example:
    >>> from rapidtools.models import Gemma4Inference
    >>> model = Gemma4Inference('google/gemma-4-E2B-it')
    >>> out = model.run_inference('roof.jpg', 'Describe the roof damage.')
    >>> print(out.text)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor
from transformers import logging as hf_logging

from .base import ModelOutput
from .catalogs import (
    GEMMA4_DEFAULT_MODEL,
    GEMMA4_MODEL_CATALOG,
    PROVIDERS,
)
from .hf_vision import _load_multimodal_model, transformers_at_least
from .local_base import BaseLocalInferenceModel

logger = logging.getLogger(__name__)


class Gemma4Inference(BaseLocalInferenceModel):
    """
    Local inference model for Google's Gemma 4 vision-language models.

    Supports every instruction-tuned Gemma 4 size (31B, 26B-A4B, 12B, E4B,
    E2B). Images are loaded with the robust PIL loader of
    :class:`~rapidtools.models.local_base.BaseLocalInferenceModel` and
    generation runs sequentially (or in explicit batches through
    :meth:`run_inference_batch`) to protect GPU memory.

    Example:
        >>> from rapidtools.models import Gemma4Inference
        >>> Gemma4Inference.list_available_models()[-1]
        'google/gemma-4-E2B-it'
        >>> model = Gemma4Inference('google/gemma-4-E4B-it', device='cuda')
        >>> out = model.run_inference(
        ...     'house.jpg', 'Rate the damage from 0 (none) to 5 (destroyed).'
        ... )
        >>> out.text
        '4'
    """

    INFO = PROVIDERS['gemma4']

    MODEL_CATALOG: list[dict[str, str]] = GEMMA4_MODEL_CATALOG
    VALID_MODELS = [entry['model_id'] for entry in GEMMA4_MODEL_CATALOG]

    def __init__(
        self,
        model_id: str = GEMMA4_DEFAULT_MODEL,
        device: str = 'auto',
        temperature: float = 0.4,
        max_tokens: int = 2048,
    ) -> None:
        """
        Initialize the processor and multimodal model.

        Args:
            model_id: Hugging Face model identifier. Defaults to
                ``'google/gemma-4-E2B-it'``; a warning is logged for IDs
                outside :attr:`VALID_MODELS`.
            device: ``'auto'``, ``'cuda'`` or ``'cpu'``.
            temperature: Sampling temperature. Defaults to 0.4.
            max_tokens: Maximum number of new tokens. Defaults to 2048.

        Raises:
            ImportError: If the checkpoint needs a newer Transformers release
                than the one installed.
        """
        # Suppress HuggingFace HTTP and warning messages:
        logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)
        hf_logging.set_verbosity_error()

        super().__init__(device=device, temperature=temperature, max_tokens=max_tokens)
        self.model_id = model_id

        if self.model_id not in self.VALID_MODELS:
            logger.warning(
                f'Model ID {self.model_id} not in officially supported Gemma-4 list.'
            )
        for entry in self.MODEL_CATALOG:
            required = entry.get('min_transformers')
            if entry['model_id'] == self.model_id and required:
                if not transformers_at_least(required):
                    raise ImportError(
                        f'{self.model_id} requires transformers>={required}. '
                        f'Run `pip install "transformers>={required}"`.'
                    )

        logger.info(f'Loading processor for {self.model_id}...')
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        logger.info(f'Loading model {self.model_id} (this may take a while)...')
        if self.device == 'auto':
            self.model = _load_multimodal_model(
                self.model_id, dtype='auto', device_map='auto'
            )
        else:
            self.model = _load_multimodal_model(self.model_id, dtype='auto').to(
                self.device
            )
        self.model.eval()

    @classmethod
    def list_available_models(cls, auth_key: str | None = None) -> list[str]:
        """
        Return the curated instruction-tuned Gemma 4 checkpoints.

        Args:
            auth_key: Unused; present for signature parity with API wrappers.

        Returns:
            list[str]: Repository IDs, largest first.

        Example:
            >>> from rapidtools.models import Gemma4Inference
            >>> Gemma4Inference.list_available_models()[0]
            'google/gemma-4-31B-it'
        """
        return [entry['model_id'] for entry in cls.MODEL_CATALOG]

    def _generation_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Translate rapidtools options into ``generate()`` keyword arguments."""
        gen = self._resolve_generation(
            kwargs.get('config'),
            temperature=kwargs.get('temperature'),
            max_tokens=kwargs.get('max_tokens'),
        )
        return {
            'max_new_tokens': gen.max_tokens,
            'temperature': gen.temperature,
            'do_sample': gen.temperature is not None and gen.temperature > 0.0,
        }

    def _prompt_text(self, prompt: str | Path, kwargs: dict[str, Any]) -> str:
        """Resolve the prompt and append a JSON instruction in ``json_mode``."""
        text = self._resolve_prompt(prompt)
        config = kwargs.get('config')
        if kwargs.get('json_mode') or (config is not None and config.json_mode):
            text += (
                '\n\nYou must respond with only valid JSON and no other '
                'conversational text.'
            )
        return text

    def _parse(self, response_text: str) -> Any:
        """Run the processor's ``parse_response`` hook when available."""
        parse = getattr(self.processor, 'parse_response', None)
        if parse is None:
            return response_text
        try:
            return parse(response_text, prefix='')
        except TypeError:
            return parse(response_text)

    def run_inference(
        self,
        image_inputs: str | Path | list[str | Path] | None,
        prompt: str | Path,
        **kwargs: Any,
    ) -> ModelOutput | None:
        """
        Run multimodal inference on a single prompt and image(s).

        Args:
            image_inputs: A path/URL, a list of them, or ``None``/``[]`` for
                a text-only prompt.
            prompt: Prompt text or path to a prompt file.
            **kwargs: ``temperature``, ``max_tokens``, ``json_mode`` overrides
                or a ``config=GenerationConfig(...)``.

        Returns:
            ModelOutput | None: Generated text and raw response, or ``None``
            when every requested image failed to load or generation raised.

        Example:
            >>> from rapidtools.models import Gemma4Inference
            >>> model = Gemma4Inference()
            >>> model.run_inference([], 'Reply with OK.').text
            'OK'
        """
        if image_inputs is None:
            image_inputs = []
        elif not isinstance(image_inputs, list):
            image_inputs = [image_inputs]

        content_list: list[dict[str, Any]] = []
        for img_input in image_inputs:
            pil_img = self._load_image_as_pil(img_input)
            if pil_img:
                content_list.append({'type': 'image', 'image': pil_img})
            else:
                logger.warning(f'Skipping failed image: {img_input}')

        # Abort only if images were requested but none loaded.
        if len(image_inputs) > 0 and not content_list:
            logger.error('No valid images loaded. Cannot proceed with inference.')
            return None

        content_list.append({'type': 'text', 'text': self._prompt_text(prompt, kwargs)})
        messages = [{'role': 'user', 'content': content_list}]

        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors='pt',
            ).to(self.model.device)
            input_len = inputs['input_ids'].shape[-1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, **self._generation_kwargs(kwargs)
                )

            response_text = self.processor.decode(
                outputs[0][input_len:], skip_special_tokens=True
            )
            return ModelOutput(
                text=response_text.strip(),
                raw_response={
                    'generated_text': response_text,
                    'parsed_response': self._parse(response_text),
                },
            )
        except Exception as e:  # noqa: BLE001 - surfaced as a failed asset
            logger.error(f'Inference failed for {self.model_id}: {e}')
            return None

    def run_inference_batch(
        self,
        batch_image_inputs: list[list[str | Path]],
        batch_prompts: list[str],
        **kwargs: Any,
    ) -> list[ModelOutput | None]:
        """
        Run batched multimodal inference on several prompt/image sets.

        Assets whose images all fail to load receive ``None`` at their index
        so the output aligns with the input. If the batched call itself
        raises, every index is ``None``.

        Args:
            batch_image_inputs: One list of image paths/URLs per asset.
            batch_prompts: One prompt per asset (same length).
            **kwargs: ``temperature`` and ``max_tokens`` overrides.

        Returns:
            list[ModelOutput | None]: Results aligned with the inputs.

        Example:
            >>> from rapidtools.models import Gemma4Inference
            >>> model = Gemma4Inference()
            >>> outs = model.run_inference_batch(
            ...     [['a.jpg'], ['b.jpg']], ['Rate damage 0-5.'] * 2
            ... )
            >>> [o.text for o in outs]
            ['1', '4']
        """
        batch_messages: list[list[dict[str, Any]] | None] = []
        for img_inputs, prompt in zip(batch_image_inputs, batch_prompts, strict=True):
            content_list: list[dict[str, Any]] = []
            for img_input in img_inputs:
                pil_img = self._load_image_as_pil(img_input)
                if pil_img:
                    content_list.append({'type': 'image', 'image': pil_img})
            if not content_list:
                batch_messages.append(None)
                continue
            content_list.append(
                {'type': 'text', 'text': self._prompt_text(prompt, kwargs)}
            )
            batch_messages.append([{'role': 'user', 'content': content_list}])

        valid_indices = [i for i, msg in enumerate(batch_messages) if msg is not None]
        valid_messages = [batch_messages[i] for i in valid_indices]
        results: list[ModelOutput | None] = [None] * len(batch_messages)
        if not valid_messages:
            return results

        try:
            inputs = self.processor.apply_chat_template(
                valid_messages,
                tokenize=True,
                return_dict=True,
                return_tensors='pt',
                padding=True,
                add_generation_prompt=True,
            ).to(self.model.device)
            input_len = inputs['input_ids'].shape[-1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, **self._generation_kwargs(kwargs)
                )

            for valid_idx, single_output in zip(valid_indices, outputs, strict=True):
                response_text = self.processor.decode(
                    single_output[input_len:], skip_special_tokens=True
                )
                results[valid_idx] = ModelOutput(
                    text=response_text.strip(),
                    raw_response={
                        'generated_text': response_text,
                        'parsed_response': self._parse(response_text),
                    },
                )
            return results
        except Exception as e:  # noqa: BLE001 - surfaced as failed assets
            logger.error(f'Batched inference failed: {e}')
            return [None] * len(batch_messages)
