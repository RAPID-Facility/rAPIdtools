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
Meta Llama vision models (Llama 4 and Llama 3.2 Vision) via Transformers.

Llama 4 Scout/Maverick are natively multimodal mixture-of-experts models
(``Llama4ForConditionalGeneration``); Llama 3.2 Vision uses the cross-attention
``Mllama`` architecture. Both are loaded through the Transformers multimodal
auto class so the same wrapper serves every generation. Downloading the
weights requires accepting Meta's licence on Hugging Face.

Example:
    >>> from rapidtools.models import LlamaVisionInference
    >>> model = LlamaVisionInference(load_in_4bit=True)
    >>> out = model.run_inference('roof.jpg', 'Describe the roof damage.')
    >>> print(out.text)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

from .base import GenerationConfig, ModelOutput
from .catalogs import (
    LLAMA_DEFAULT_MODEL,
    LLAMA_MODEL_CATALOG,
    PROVIDERS,
)
from .hf_vision import _load_multimodal_model
from .local_base import BaseLocalInferenceModel

logger = logging.getLogger(__name__)


class LlamaVisionInference(BaseLocalInferenceModel):
    """
    Local inference with Meta's Llama 4 and Llama 3.2 Vision models.

    Example:
        >>> from rapidtools.models import LlamaVisionInference
        >>> LlamaVisionInference.list_available_models()[0]
        'meta-llama/Llama-4-Scout-17B-16E-Instruct'
        >>> model = LlamaVisionInference(
        ...     'meta-llama/Llama-3.2-11B-Vision-Instruct', load_in_4bit=True
        ... )
        >>> out = model.run_inference(
        ...     ['front.jpg', 'side.jpg'], 'Return JSON with a damage_level key.',
        ...     json_mode=True,
        ... )
        >>> out.text
        '{"damage_level": "severe"}'
    """

    INFO = PROVIDERS['llama']

    MODEL_CATALOG: list[dict[str, str]] = LLAMA_MODEL_CATALOG

    def __init__(
        self,
        model_id: str = LLAMA_DEFAULT_MODEL,
        device: str = 'auto',
        load_in_4bit: bool = False,
        temperature: float = 0.4,
        max_tokens: int = 2048,
    ):
        """
        Load the processor and weights into GPU/CPU memory.

        Args:
            model_id: Hugging Face repo ID (Llama 4 or Llama 3.2 Vision).
            device: ``'cuda'``, ``'cpu'``, or ``'auto'`` (spread across GPUs).
            load_in_4bit: Quantize with bitsandbytes so large models fit
                consumer GPUs (e.g. the 11B model on 24 GB).
            temperature: Sampling temperature (0 disables sampling).
            max_tokens: Maximum new tokens generated per call.
        """
        super().__init__(device=device, temperature=temperature, max_tokens=max_tokens)
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit

        logger.info(
            f'Loading processor and weights for {self.model_id}. '
            'This may take a moment...'
        )

        # 1. Load the processor (tokenization + image resizing/normalization)
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # 2. Configure model loading and VRAM management
        model_kwargs: dict[str, Any] = {'device_map': self.device}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        else:
            # Half precision halves memory relative to float32.
            model_kwargs['dtype'] = torch.float16

        # 3. Load through the multimodal auto class (Mllama or Llama4).
        self.model = _load_multimodal_model(self.model_id, **model_kwargs)
        self.model.eval()
        logger.info(f'Llama model {self.model_id} loaded successfully.')

    @classmethod
    def list_available_models(cls, auth_key: str | None = None) -> list[str]:
        """
        Return the curated Llama vision checkpoints.

        Downloading them from the Hub requires accepting Meta's licence.

        Args:
            auth_key: Unused; present for signature parity with API wrappers.

        Returns:
            list[str]: Repository IDs, newest first.

        Example:
            >>> from rapidtools.models import LlamaVisionInference
            >>> len(LlamaVisionInference.list_available_models())
            6
        """
        return [entry['model_id'] for entry in cls.MODEL_CATALOG]

    def run_inference(
        self,
        image_inputs: str | Path | list[str | Path],
        prompt: str | Path,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> ModelOutput | None:
        """
        Execute a forward pass and return the generated text.

        Args:
            image_inputs: One path/URL or a list of them.
            prompt: Prompt text or path to a prompt file.
            json_mode: Append a JSON-only instruction to the prompt (open
                models have no native JSON mode).
            temperature: Per-call temperature override.
            max_tokens: Per-call output-token limit override.
            config: A :class:`~rapidtools.models.GenerationConfig`; explicit
                keyword arguments take precedence over it.
            **kwargs: Ignored; accepted for interface compatibility.

        Returns:
            ModelOutput | None: Generated text plus raw token IDs, or ``None``
            when no image loads or generation fails (e.g. out of memory).

        Example:
            >>> from rapidtools.models import LlamaVisionInference
            >>> model = LlamaVisionInference(load_in_4bit=True)
            >>> model.run_inference('house.jpg', 'Is the roof intact?').text
            'No, part of the roof is missing.'
        """
        prompt_str = self._resolve_prompt(prompt)
        log_ctx = f"[Prompt snippet: '{prompt_str[:30]}...']"

        if not isinstance(image_inputs, list):
            image_inputs = [image_inputs]

        loaded_images = []
        for img_input in image_inputs:
            pil_img = self._load_image_as_pil(img_input)
            if pil_img:
                loaded_images.append(pil_img)

        if not loaded_images:
            logger.error(f'{log_ctx} No valid images could be loaded into PIL.')
            return None

        gen = self._resolve_generation(
            config, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
        )
        if gen.json_mode:
            prompt_str += (
                '\n\nYou must respond with only valid JSON and no other '
                'conversational text.'
            )

        content_block: list[dict[str, Any]] = [{'type': 'image'} for _ in loaded_images]
        if prompt_str:
            content_block.append({'type': 'text', 'text': prompt_str})
        messages = [{'role': 'user', 'content': content_block}]

        try:
            text_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.processor(
                text=text_prompt, images=loaded_images, return_tensors='pt'
            )
            inputs = inputs.to(self.model.device)
        except Exception as e:  # noqa: BLE001 - surfaced as a failed asset
            logger.error(f'{log_ctx} Failed to process Llama tensors: {e}')
            return None

        final_temp = gen.temperature
        final_tokens = gen.max_tokens

        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=final_tokens,
                    temperature=final_temp,
                    do_sample=(final_temp > 0.0),
                )
            # Keep only the newly generated tokens.
            generated_ids = output_ids[0][inputs['input_ids'].shape[1] :]
            extracted_text = self.processor.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
            return ModelOutput(
                text=extracted_text,
                raw_response={'output_ids': output_ids.cpu().tolist()},
            )
        except torch.OutOfMemoryError:
            logger.error(
                f'{log_ctx} GPU OUT OF MEMORY ERROR. Try smaller images or '
                'setting load_in_4bit=True.'
            )
            return None
        except Exception as e:  # noqa: BLE001 - surfaced as a failed asset
            logger.error(f'{log_ctx} Unexpected tensor generation error: {e}')
            return None
