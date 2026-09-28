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
Meta Segment Anything Model 3 (SAM 3) via Transformers.

SAM 3 performs promptable concept segmentation: given a short text phrase
such as ``'building'`` it returns instance masks, boxes and scores for every
matching object. Only ``facebook/sam3`` has a Transformers integration;
``facebook/sam3.1`` ships raw checkpoints for the reference repository only.

Example:
    >>> from rapidtools.models import SAM3Inference
    >>> sam = SAM3Inference()
    >>> out = sam.run_inference('tile.jpg', 'building', threshold=0.5)
    >>> out.masks[0].shape  # (n_instances, height, width)
    (12, 1024, 1024)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import torch
from transformers import Sam3Model, Sam3Processor

from .base import ModelOutput, SegmentationConfig
from .catalogs import (
    PROVIDERS,
    SAM3_DEFAULT_MODEL,
    SAM3_MODEL_CATALOG,
)
from .local_base import BaseLocalInferenceModel

logger = logging.getLogger(__name__)


class SAM3Inference(BaseLocalInferenceModel):
    """
    Text-prompted instance segmentation with SAM 3.

    Instantiate once and call :meth:`run_inference` repeatedly to avoid
    reloading the ~3.4 GB of weights.

    Example:
        >>> from rapidtools.models import SAM3Inference
        >>> SAM3Inference.list_available_models()
        ['facebook/sam3']
        >>> sam = SAM3Inference(device='cuda')
        >>> out = sam.run_inference(['a.jpg', 'b.jpg'], 'swimming pool')
        >>> len(out.masks), len(out.bounding_boxes)
        (2, 2)
    """

    INFO = PROVIDERS['sam3']

    MODEL_CATALOG: list[dict[str, str]] = SAM3_MODEL_CATALOG

    def __init__(
        self,
        model_id: str = SAM3_DEFAULT_MODEL,
        device: str = 'auto',
        load_in_4bit: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """
        Load the SAM 3 processor and weights.

        Args:
            model_id: Hugging Face repository ID. Defaults to ``'facebook/sam3'``.
            device: ``'auto'``, ``'cuda'`` or ``'cpu'``.
            load_in_4bit: Quantize with bitsandbytes to reduce VRAM.
            temperature: Deprecated and ignored (SAM does not sample text).
            max_tokens: Deprecated and ignored.
        """
        if temperature is not None or max_tokens is not None:
            warnings.warn(
                'SAM3Inference ignores temperature and max_tokens; these '
                'parameters will be removed in rapidtools 0.2.',
                DeprecationWarning,
                stacklevel=2,
            )
        super().__init__(device=device, temperature=0.0, max_tokens=0)

        # Silence noisy loggers:
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('huggingface_hub').setLevel(logging.WARNING)

        self.model_id = model_id
        self.load_in_4bit = load_in_4bit

        logger.info(
            f'Loading processor and weights for {self.model_id} '
            '(this may take a minute)...'
        )
        self.processor = Sam3Processor.from_pretrained(self.model_id)

        model_kwargs: dict[str, Any] = {
            'dtype': torch.float16,
            'device_map': 'auto' if self.device == 'auto' else self.device,
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )

        self.model = Sam3Model.from_pretrained(self.model_id, **model_kwargs)
        self.model.eval()
        logger.info(f'SAM 3 model {self.model_id} loaded successfully.')

    @classmethod
    def list_available_models(cls, auth_key: str | None = None) -> list[str]:
        """
        Return the SAM 3 checkpoints supported through Transformers.

        Args:
            auth_key: Unused; present for signature parity with API wrappers.

        Returns:
            list[str]: Repository IDs.

        Example:
            >>> from rapidtools.models import SAM3Inference
            >>> SAM3Inference.list_available_models()
            ['facebook/sam3']
        """
        return [entry['model_id'] for entry in cls.MODEL_CATALOG]

    def run_inference(
        self,
        image_inputs: str | Path | list[str | Path],
        prompt: str | Path | None = None,
        threshold: float | None = None,
        mask_threshold: float | None = None,
        config: SegmentationConfig | None = None,
        **kwargs: Any,
    ) -> ModelOutput | None:
        """
        Segment every instance of ``prompt`` in one or more images.

        Args:
            image_inputs: One path/URL or a list of them (processed as a batch).
            prompt: Concept to segment (e.g. ``'building'``) or a prompt file.
                ``None`` runs the model without a text prompt.
            threshold: Detection confidence threshold (default 0.5).
            mask_threshold: Threshold for binarising mask logits (default 0.5).
            config: A :class:`~rapidtools.models.base.SegmentationConfig`;
                explicit keyword arguments take precedence over it.
            **kwargs: Ignored; accepted for interface compatibility.

        Returns:
            ModelOutput | None: ``masks`` holds one ``(n, H, W)`` array per
            image, ``bounding_boxes`` one ``[[x0, y0, x1, y1], ...]`` list per
            image, and ``raw_response['scores']`` the confidences. ``None`` when
            no image loads or inference fails.

        Example:
            >>> from rapidtools.models import SAM3Inference
            >>> sam = SAM3Inference()
            >>> out = sam.run_inference('tile.jpg', 'tree', threshold=0.4)
            >>> out.has_masks, out.raw_response['status']
            (True, 'success')
        """
        base = config or SegmentationConfig()
        threshold = base.threshold if threshold is None else threshold
        mask_threshold = (
            base.mask_threshold if mask_threshold is None else mask_threshold
        )
        prompt_str = self._resolve_prompt(prompt) if prompt is not None else ''
        log_ctx = f"[Prompt: '{prompt_str[:30]}...']" if prompt_str else '[No prompt]'

        if not isinstance(image_inputs, list):
            image_inputs = [image_inputs]

        loaded_images = []
        for img_input in image_inputs:
            pil_img = self._load_image_as_pil(img_input)
            if pil_img:
                loaded_images.append(pil_img)

        if not loaded_images:
            logger.error(f'{log_ctx} No valid images could be loaded.')
            return None

        try:
            processor_args: dict[str, Any] = {
                'images': loaded_images,
                'return_tensors': 'pt',
            }
            if prompt_str:
                processor_args['text'] = [prompt_str] * len(loaded_images)

            inputs = self.processor(**processor_args)
            target_device = self.model.device
            inputs = {
                k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
        except Exception as e:  # noqa: BLE001 - surfaced as a failed batch
            logger.error(f'{log_ctx} Failed to process inputs for SAM 3: {e}')
            return None

        try:
            with torch.no_grad():
                outputs = self.model(**inputs)

            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=inputs.get('original_sizes').tolist(),
            )

            extracted_masks = []
            extracted_boxes = []
            extracted_scores = []
            for res in results:
                extracted_masks.append(res['masks'].cpu().numpy())
                if 'boxes' in res:
                    extracted_boxes.append(res['boxes'].cpu().numpy().tolist())
                if 'scores' in res:
                    extracted_scores.append(res['scores'].cpu().numpy().tolist())

            return ModelOutput(
                text=None,  # SAM outputs segments, not conversational text
                masks=extracted_masks,
                bounding_boxes=extracted_boxes,
                raw_response={
                    'status': 'success',
                    'device': str(self.model.device),
                    'scores': extracted_scores,
                },
            )
        except torch.OutOfMemoryError:
            logger.error(
                f'{log_ctx} GPU OUT OF MEMORY ERROR. Try smaller images, smaller '
                'batch sizes, or set load_in_4bit=True.'
            )
            return None
        except Exception as e:  # noqa: BLE001 - surfaced as a failed batch
            logger.error(f'{log_ctx} Unexpected SAM 3 generation error: {e}')
            return None
