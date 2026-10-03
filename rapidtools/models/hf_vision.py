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
Generic local vision-language model backend for Hugging Face checkpoints.

Any model that Transformers exposes through its multimodal auto class
(``AutoModelForMultimodalLM`` / ``AutoModelForImageTextToText``) and that
ships a chat template can be used: Qwen3.8 / Qwen3.6 / Qwen3.5 / Qwen3-VL /
Qwen2.5-VL, Meta Muse Glimmer, Llama 4 and Llama 3.2 Vision, Gemma 3 and
Gemma 3n, InternVL3.5 / InternVL3, LLaVA-NeXT / LLaVA-OneVision, Pixtral,
Granite Vision and many more.

:class:`HFVisionInference` is also the base class of the family-specific
wrappers :class:`~rapidtools.models.QwenVisionInference` and
:class:`~rapidtools.models.MuseGlimmerInference`, which only swap the
default checkpoint and catalogue.

Example:
    >>> from rapidtools.models import HFVisionInference
    >>> HFVisionInference.list_available_models()[:2]
    ['Qwen/Qwen3.8-27B', 'Qwen/Qwen3.6-27B']
    >>> model = HFVisionInference('Qwen/Qwen2.5-VL-3B-Instruct', load_in_4bit=True)
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
    DEFAULT_OPEN_VLM,
    OPEN_VLM_CATALOG,
    PROVIDERS,
)
from .local_base import BaseLocalInferenceModel

logger = logging.getLogger(__name__)


def transformers_version() -> str:
    """
    Return the installed Transformers version string.

    Example:
        >>> from rapidtools.models.hf_vision import transformers_version
        >>> transformers_version().count('.') >= 1
        True
    """
    import transformers

    return transformers.__version__


def transformers_at_least(minimum: str) -> bool:
    """
    Check whether the installed Transformers release is at least ``minimum``.

    Args:
        minimum: Version string such as ``'5.15.0'``.

    Returns:
        bool: ``True`` when the installed version satisfies the requirement.

    Example:
        >>> from rapidtools.models.hf_vision import transformers_at_least
        >>> transformers_at_least('4.0.0')
        True
    """
    from packaging.version import Version

    return Version(transformers_version()) >= Version(minimum)


def _load_multimodal_model(model_id: str, **kwargs: Any):
    """
    Load a checkpoint with whichever multimodal auto class Transformers has.

    Args:
        model_id: Hugging Face repository ID.
        **kwargs: Passed to ``from_pretrained`` (dtype, device_map, ...).

    Returns:
        The loaded ``PreTrainedModel``.

    Raises:
        ImportError: If the installed Transformers has no multimodal auto
            class.
    """
    import transformers

    for name in ('AutoModelForMultimodalLM', 'AutoModelForImageTextToText'):
        auto_class = getattr(transformers, name, None)
        if auto_class is not None:
            return auto_class.from_pretrained(model_id, **kwargs)
    raise ImportError(
        'This version of transformers has no multimodal auto class; '
        'please upgrade transformers.'
    )


class HFVisionInference(BaseLocalInferenceModel):
    """
    Run any Hugging Face vision-language checkpoint that ships a chat template.

    Family-specific subclasses override the class attributes below.

    Attributes:
        MODEL_CATALOG: Curated checkpoint records for :meth:`list_available_models`.
        DEFAULT_MODEL: Checkpoint used when ``model_id`` is omitted.
        MIN_TRANSFORMERS_VERSION: Minimum Transformers release required by the
            family; a clear ``ImportError`` is raised when it is not met.
        FAMILY_NAME: Human-readable family label used in logs.

    Example:
        >>> from rapidtools.models import HFVisionInference
        >>> model = HFVisionInference('Qwen/Qwen2.5-VL-3B-Instruct', load_in_4bit=True)
        >>> out = model.run_inference('roof.jpg', 'Describe the roof damage.')
        >>> print(out.text)
        >>> batch = model.run_inference_batch(
        ...     [['a.jpg'], ['b.jpg']], ['Rate damage 0-5.', 'Rate damage 0-5.']
        ... )
        >>> [o.text for o in batch]
        ['2', '4']
    """

    INFO = PROVIDERS['hf']

    MODEL_CATALOG: list[dict[str, str]] = OPEN_VLM_CATALOG
    DEFAULT_MODEL: str = DEFAULT_OPEN_VLM
    MIN_TRANSFORMERS_VERSION: str | None = None
    FAMILY_NAME: str = 'Hugging Face VLM'

    def __init__(
        self,
        model_id: str | None = None,
        device: str = 'auto',
        load_in_4bit: bool = False,
        temperature: float = 0.4,
        max_tokens: int = 512,
        trust_remote_code: bool = False,
    ) -> None:
        """
        Load the processor and weights.

        Args:
            model_id: Hugging Face repository ID of the checkpoint. Defaults
                to :attr:`DEFAULT_MODEL`.
            device: ``'auto'`` (spread across available devices), ``'cuda'``,
                or ``'cpu'``.
            load_in_4bit: Quantize with bitsandbytes so large models fit
                consumer GPUs.
            temperature: Sampling temperature (0 disables sampling).
            max_tokens: Maximum new tokens generated per call.
            trust_remote_code: Allow repositories that ship custom modelling
                code. Only enable for sources you trust.

        Raises:
            ImportError: If the installed Transformers is older than
                :attr:`MIN_TRANSFORMERS_VERSION` (or the catalogue entry's
                ``min_transformers``).
        """
        logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)
        hf_logging.set_verbosity_error()
        super().__init__(device=device, temperature=temperature, max_tokens=max_tokens)
        self.model_id = model_id or self.DEFAULT_MODEL
        self.load_in_4bit = load_in_4bit
        self._check_transformers_version()

        logger.info(f'Loading processor for {self.model_id}...')
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=trust_remote_code
        )
        tokenizer = getattr(self.processor, 'tokenizer', None)
        if tokenizer is not None:
            # Left padding keeps generated tokens aligned in batched decoding.
            tokenizer.padding_side = 'left'
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

        logger.info(f'Loading model {self.model_id} (this may take a while)...')
        model_kwargs: dict[str, Any] = {
            'dtype': 'auto',
            'trust_remote_code': trust_remote_code,
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        if self.device == 'auto' or load_in_4bit:
            model_kwargs['device_map'] = (
                'auto' if self.device == 'auto' else self.device
            )
            self.model = _load_multimodal_model(self.model_id, **model_kwargs)
        else:
            self.model = _load_multimodal_model(self.model_id, **model_kwargs).to(
                self.device
            )
        self.model.eval()
        logger.info(f'Model {self.model_id} loaded successfully.')

    # ------------------------------------------------------------ catalogue
    def _check_transformers_version(self) -> None:
        """Raise ``ImportError`` when Transformers is too old for the model."""
        required = self.MIN_TRANSFORMERS_VERSION
        for entry in self.MODEL_CATALOG:
            if entry['model_id'] == self.model_id and entry.get('min_transformers'):
                required = entry['min_transformers']
                break
        if required and not transformers_at_least(required):
            raise ImportError(
                f'{self.model_id} requires transformers>={required} but '
                f'{transformers_version()} is installed. Run '
                f'`pip install "transformers>={required}"`.'
            )

    @classmethod
    def list_available_models(cls, auth_key: str | None = None) -> list[str]:
        """
        Return the curated catalogue of checkpoints for this family.

        Args:
            auth_key: Unused; present for signature parity with API wrappers.

        Returns:
            list[str]: Hugging Face repository IDs, newest first.

        Example:
            >>> from rapidtools.models import HFVisionInference
            >>> 'google/gemma-3-4b-it' in HFVisionInference.list_available_models()
            True
        """
        return [entry['model_id'] for entry in cls.MODEL_CATALOG]

    @classmethod
    def search_hub_models(cls, limit: int = 40) -> list[str]:
        """
        List the most downloaded image-text-to-text checkpoints on the Hub.

        The curated catalogue is returned first; Hub results that are not
        already listed are appended. Falls back to the catalogue alone when
        the Hub cannot be reached.

        Args:
            limit: Maximum number of Hub results to fetch.

        Returns:
            list[str]: Repository IDs.

        Example:
            >>> from rapidtools.models import HFVisionInference
            >>> ids = HFVisionInference.search_hub_models(limit=10)
            >>> ids[0]
            'Qwen/Qwen3.8-27B'
        """
        try:
            from huggingface_hub import list_models

            found = list_models(
                pipeline_tag='image-text-to-text',
                sort='downloads',
                direction=-1,
                limit=limit,
            )
            ids = [m.id for m in found if getattr(m, 'id', None)]
        except Exception as exc:  # noqa: BLE001 - network failures degrade gracefully
            logger.warning(f'Could not query the Hugging Face Hub: {exc}')
            ids = []
        curated = cls.list_available_models()
        return curated + [i for i in ids if i not in curated]

    # ------------------------------------------------------------ helpers
    def _build_messages(
        self, image_inputs: list[str | Path], prompt: str
    ) -> list[dict[str, Any]] | None:
        """
        Build a single-turn chat message with PIL images and the prompt.

        Returns ``None`` when images were requested but none could be loaded.
        """
        content: list[dict[str, Any]] = []
        for img_input in image_inputs:
            pil_img = self._load_image_as_pil(img_input)
            if pil_img is not None:
                content.append({'type': 'image', 'image': pil_img})
            else:
                logger.warning(f'Skipping failed image: {img_input}')
        if image_inputs and not content:
            return None
        content.append({'type': 'text', 'text': prompt})
        return [{'role': 'user', 'content': content}]

    def _apply_json_mode(self, prompt_str: str, kwargs: dict[str, Any]) -> str:
        """Append a JSON-only instruction when ``json_mode`` was requested."""
        config = kwargs.get('config')
        if kwargs.get('json_mode') or (config is not None and config.json_mode):
            return (
                f'{prompt_str}\n\nYou must respond with only valid JSON and no '
                'other conversational text.'
            )
        return prompt_str

    def _generation_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Translate rapidtools options into ``generate()`` keyword arguments."""
        resolved = self._resolve_generation(
            kwargs.get('config'),
            temperature=kwargs.get('temperature'),
            max_tokens=kwargs.get('max_tokens'),
        )
        temperature = resolved.temperature
        do_sample = temperature is not None and temperature > 0.0
        gen: dict[str, Any] = {
            'max_new_tokens': resolved.max_tokens,
            'do_sample': do_sample,
        }
        if do_sample:
            gen['temperature'] = temperature
        return gen

    def _generate(
        self, messages_batch: list[list[dict[str, Any]]], kwargs: dict[str, Any]
    ) -> list[str]:
        """Tokenize one or more conversations, generate, and decode new tokens."""
        inputs = self.processor.apply_chat_template(
            messages_batch if len(messages_batch) > 1 else messages_batch[0],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors='pt',
            padding=len(messages_batch) > 1,
        ).to(self.model.device)
        input_len = inputs['input_ids'].shape[-1]
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self._generation_kwargs(kwargs))
        return [
            self.processor.decode(row[input_len:], skip_special_tokens=True).strip()
            for row in outputs
        ]

    # ---------------------------------------------------------- inference
    def run_inference(
        self,
        image_inputs: str | Path | list[str | Path],
        prompt: str | Path,
        **kwargs: Any,
    ) -> ModelOutput | None:
        """
        Run the model on one prompt and its image(s).

        Args:
            image_inputs: A path/URL, a list of them, or ``None``/``[]`` for a
                text-only prompt.
            prompt: Prompt text or path to a prompt file.
            **kwargs: ``temperature``, ``max_tokens``, ``json_mode`` overrides
                or a ``config=GenerationConfig(...)``.

        Returns:
            ModelOutput | None: Generated text, or ``None`` when every image
            failed to load or generation raised.

        Example:
            >>> from rapidtools.models import HFVisionInference
            >>> model = HFVisionInference('Qwen/Qwen2.5-VL-3B-Instruct')
            >>> model.run_inference([], 'Reply with the word ready.').text
            'ready'
        """
        if image_inputs is None:
            image_inputs = []
        elif not isinstance(image_inputs, list):
            image_inputs = [image_inputs]
        prompt_str = self._apply_json_mode(self._resolve_prompt(prompt), kwargs)
        messages = self._build_messages(image_inputs, prompt_str)
        if messages is None:
            logger.error('No valid images loaded. Cannot proceed with inference.')
            return None
        try:
            text = self._generate([messages], kwargs)[0]
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed asset
            logger.error(f'Inference failed for {self.model_id}: {exc}')
            return None
        return ModelOutput(text=text, raw_response={'generated_text': text})

    def run_inference_batch(
        self,
        batch_image_inputs: list[list[str | Path]],
        batch_prompts: list[str],
        **kwargs: Any,
    ) -> list[ModelOutput | None]:
        """
        Run the model on several prompt/image sets at once (left-padded).

        Entries whose images all failed to load yield ``None`` at their index.
        If the batched call raises (typically out-of-memory), the batch is
        retried one asset at a time.

        Args:
            batch_image_inputs: One list of image paths per asset.
            batch_prompts: One prompt per asset (same length).
            **kwargs: ``temperature`` and ``max_tokens`` overrides.

        Returns:
            list[ModelOutput | None]: Results aligned with the inputs.

        Example:
            >>> from rapidtools.models import HFVisionInference
            >>> model = HFVisionInference('Qwen/Qwen2.5-VL-3B-Instruct')
            >>> outs = model.run_inference_batch(
            ...     [['a.jpg'], ['missing.jpg']], ['Describe.', 'Describe.']
            ... )
            >>> outs[1] is None
            True
        """
        messages_batch: list[list[dict[str, Any]] | None] = [
            self._build_messages(
                list(imgs), self._apply_json_mode(self._resolve_prompt(prompt), kwargs)
            )
            for imgs, prompt in zip(batch_image_inputs, batch_prompts, strict=True)
        ]
        valid = [i for i, m in enumerate(messages_batch) if m is not None]
        valid_messages = [m for m in messages_batch if m is not None]
        results: list[ModelOutput | None] = [None] * len(messages_batch)
        if not valid:
            return results
        try:
            texts = self._generate(valid_messages, kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.error(f'Batch inference failed for {self.model_id}: {exc}')
            if len(valid) > 1:
                logger.info('Retrying the batch one asset at a time...')
                for i in valid:
                    results[i] = self.run_inference(
                        list(batch_image_inputs[i]), batch_prompts[i], **kwargs
                    )
            return results
        for i, text in zip(valid, texts, strict=True):
            results[i] = ModelOutput(text=text, raw_response={'generated_text': text})
        return results
