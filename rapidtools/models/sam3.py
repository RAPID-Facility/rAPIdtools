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
# 11-27-2026

"""
Meta Segment Anything Model 3 (SAM 3) via Transformers.

SAM 3 performs promptable concept segmentation: given a short text phrase
such as ``'building'`` it returns instance masks, boxes and scores for every
matching object. Only ``facebook/sam3`` has a Transformers integration;
``facebook/sam3.1`` ships raw checkpoints for the reference repository only.

The same checkpoint also answers point prompts: :class:`SAM3PointInference`
returns the object under a few pixels of an image, as candidates from a part
to the whole. Street-level localization uses it to ask whether two masks
are wings of one building.

Example:
    >>> from rapidtools.models import SAM3Inference
    >>> sam = SAM3Inference()  # doctest: +SKIP
    >>> out = sam.run_inference('tile.jpg', 'building', threshold=0.5)  # doctest: +SKIP
    >>> out.masks[0].shape  # (n_instances, height, width)  # doctest: +SKIP
    (12, 1024, 1024)
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    Sam3Model,
    Sam3Processor,
    Sam3TrackerModel,
    Sam3TrackerProcessor,
)

from .base import ModelOutput, SegmentationConfig
from .catalogs import (
    PROVIDERS,
    SAM3_DEFAULT_MODEL,
    SAM3_MODEL_CATALOG,
)
from .local_base import BaseLocalInferenceModel

logger = logging.getLogger(__name__)


def _warn_if_off_gpu(model: Any, device: str, model_id: str) -> None:
    """
    Report a model the loader put on the CPU although a GPU exists.

    With ``device='auto'`` the loader puts the weights wherever there is
    room. When the GPU is full, or unusable after a driver reset, that is
    the CPU: the model still works there, dozens of times slower, so a scan
    that takes minutes on the GPU can keep every core busy for hours. The
    fallback is kept, but it is logged as a warning instead of happening
    silently, and PyTorch is limited to half of the CPU cores so that the
    computer stays usable. A deliberate CPU run (``device='cpu'``, or a
    machine without a GPU) is left alone.
    """
    if device != 'auto' or not torch.cuda.is_available():
        return
    placement = getattr(model, 'hf_device_map', None)
    if not isinstance(placement, dict):
        return
    offloaded = sorted(
        {str(d) for d in placement.values() if str(d) in ('cpu', 'disk')}
    )
    if not offloaded:
        return
    cores = os.cpu_count() or 2
    threads = max(1, min(torch.get_num_threads(), cores // 2))
    torch.set_num_threads(threads)
    logger.warning(
        f'{model_id} did not fit on the GPU and runs on the '
        f'{" and ".join(offloaded)} instead, dozens of times slower, using '
        f'{threads} of the {cores} CPU cores so the computer stays usable: the '
        'GPU is busy, out of memory, or was reset by its driver. To use the GPU '
        'again, close other programs that use it (or restart after a driver '
        "reset); pass device='cpu' to run on the CPU without this warning."
    )


class SAM3Inference(BaseLocalInferenceModel):
    """
    Text-prompted instance segmentation with SAM 3.

    Instantiate once and call :meth:`run_inference` repeatedly to avoid
    reloading the ~3.4 GB of weights.

    Example:
        >>> from rapidtools.models import SAM3Inference
        >>> SAM3Inference.list_available_models()
        ['facebook/sam3']
        >>> sam = SAM3Inference(device='cuda')  # doctest: +SKIP
        >>> out = sam.run_inference(  # doctest: +SKIP
        ...     ['a.jpg', 'b.jpg'], 'swimming pool'
        ... )
        >>> len(out.masks), len(out.bounding_boxes)  # doctest: +SKIP
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

        With ``device='auto'`` the model runs on the GPU when there is one
        with room for it, and on the CPU otherwise; a GPU that is present
        but full, or unusable after a driver reset, is reported with a
        warning, since the CPU is dozens of times slower.
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
        _warn_if_off_gpu(self.model, self.device, self.model_id)
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
        image_inputs: str | Path | Sequence[str | Path],
        prompt: str | Path | None = None,
        threshold: float | None = None,
        mask_threshold: float | None = None,
        config: SegmentationConfig | None = None,
        max_mask_side: int | None = None,
        **kwargs: Any,
    ) -> ModelOutput | None:
        """
        Segment every instance of ``prompt`` in one or more images.

        Args:
            image_inputs: One path/URL or a list of them (processed as a batch).
            prompt: Concept to segment (e.g. ``'building'``) or a prompt file.
                ``None`` runs the model without a text prompt.
            threshold: Detection confidence threshold (default 0.5).
            mask_threshold: Threshold for binarizing mask logits (default 0.5).
            config: A :class:`~rapidtools.models.base.SegmentationConfig`;
                explicit keyword arguments take precedence over it.
            max_mask_side: Return masks (and boxes) of images larger than
                this many pixels on their longer side at a reduced size with
                that longer side, instead of at the image size. SAM 3
                predicts masks at about 288 x 288 pixels, so nothing is lost,
                while a few full-size masks of a 2 cm drone crop take
                gigabytes. ``None`` (default) keeps the image size.
            **kwargs: Ignored; accepted for interface compatibility.

        Returns:
            ModelOutput | None: ``masks`` holds one ``(n, H, W)`` array per
            image, ``bounding_boxes`` one ``[[x0, y0, x1, y1], ...]`` list per
            image, and ``raw_response['scores']`` the confidences. All three
            lists have exactly one entry per input image, in input order; an
            image that could not be loaded (or a result without boxes/scores)
            is represented by ``None``, and ``raw_response['failed_indices']``
            lists the positions that failed. ``None`` when no image loads or
            inference fails.

        Example:
            >>> from rapidtools.models import SAM3Inference
            >>> sam = SAM3Inference()  # doctest: +SKIP
            >>> out = sam.run_inference(  # doctest: +SKIP
            ...     'tile.jpg', 'tree', threshold=0.4
            ... )
            >>> out.has_masks, out.raw_response['status']  # doctest: +SKIP
            (True, 'success')
        """
        base = config or SegmentationConfig()
        threshold = base.threshold if threshold is None else threshold
        mask_threshold = (
            base.mask_threshold if mask_threshold is None else mask_threshold
        )
        prompt_str = self._resolve_prompt(prompt) if prompt is not None else ''
        log_ctx = f"[Prompt: '{prompt_str[:30]}...']" if prompt_str else '[No prompt]'

        image_list: list[str | Path] = (
            [image_inputs]
            if isinstance(image_inputs, (str, Path))
            else list(image_inputs)
        )

        # Images that fail to load keep a ``None`` placeholder in every
        # per-image output list so callers can zip the results back onto
        # their inputs by position.
        loaded_images = []
        loaded_indices: list[int] = []
        failed_indices: list[int] = []
        for index, img_input in enumerate(image_list):
            pil_img = self._load_image_as_pil(img_input)
            if pil_img:
                loaded_images.append(pil_img)
                loaded_indices.append(index)
            else:
                failed_indices.append(index)

        if not loaded_images:
            logger.error(f'{log_ctx} No valid images could be loaded.')
            return None
        if failed_indices:
            logger.warning(
                f'{log_ctx} {len(failed_indices)} of {len(image_list)} images '
                'could not be loaded; their results are None.'
            )

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

            target_sizes = inputs['original_sizes'].tolist()
            if max_mask_side:
                target_sizes = [
                    [
                        max(1, round(h * min(1.0, max_mask_side / max(h, w)))),
                        max(1, round(w * min(1.0, max_mask_side / max(h, w)))),
                    ]
                    for h, w in target_sizes
                ]
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=target_sizes,
            )

            n_inputs = len(image_list)
            extracted_masks: list = [None] * n_inputs
            extracted_boxes: list = [None] * n_inputs
            extracted_scores: list = [None] * n_inputs
            for index, res in zip(loaded_indices, results, strict=True):
                extracted_masks[index] = res['masks'].cpu().numpy()
                if 'boxes' in res:
                    extracted_boxes[index] = res['boxes'].cpu().numpy().tolist()
                if 'scores' in res:
                    extracted_scores[index] = res['scores'].cpu().numpy().tolist()

            return ModelOutput(
                text=None,  # SAM outputs segments, not conversational text
                masks=extracted_masks,
                bounding_boxes=extracted_boxes,
                raw_response={
                    'status': 'success',
                    'device': str(self.model.device),
                    'scores': extracted_scores,
                    'failed_indices': failed_indices,
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


class SAM3PointInference(BaseLocalInferenceModel):
    """
    SAM 3 prompted with points: the object under a few pixels of an image.

    A click on a garage door is ambiguous: the door, the garage or the
    house. The model answers with several candidates, from the part to the
    whole, which is what makes it useful for asking whether two masks
    belong to one object: the whole object found from one covers the other.
    This is the model's interactive head (``Sam3TrackerModel``), loaded from
    the same weights as :class:`SAM3Inference`.

    Example:
        >>> from rapidtools.models.sam3 import SAM3PointInference
        >>> sam = SAM3PointInference()  # doctest: +SKIP
        >>> (candidates,) = sam.segment_at(  # doctest: +SKIP
        ...     'pano.jpg', [[(420, 260), (450, 270)]])
        >>> candidates.shape  # (candidates, height, width)  # doctest: +SKIP
        (3, 512, 1024)
    """

    def __init__(self, model_id: str = SAM3_DEFAULT_MODEL, device: str = 'auto'):
        """
        Load the processor and weights of the interactive head.

        Args:
            model_id: Hugging Face repository ID. Defaults to ``'facebook/sam3'``.
            device: ``'auto'``, ``'cuda'`` or ``'cpu'``.
        """
        super().__init__(device=device, temperature=0.0, max_tokens=0)
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
        self.model_id = model_id
        logger.info(f'Loading the point-prompt head of {self.model_id}...')
        self.processor = Sam3TrackerProcessor.from_pretrained(self.model_id)
        self.model = Sam3TrackerModel.from_pretrained(
            self.model_id,
            dtype=torch.float16,
            device_map='auto' if self.device == 'auto' else self.device,
        )
        _warn_if_off_gpu(self.model, self.device, self.model_id)
        self.model.eval()

    def run_inference(
        self,
        image_inputs: str | Path | Sequence[str | Path],
        prompt: str | Path | None = None,
        points: Sequence[Sequence[Sequence[float]]] | None = None,
        **kwargs: Any,
    ) -> ModelOutput | None:
        """
        Segment the objects under ``points`` in one image.

        The form every model wrapper shares; :meth:`segment_at` is the
        direct one.

        Args:
            image_inputs: Path or URL of the image (the first, of several).
            prompt: Ignored; this head takes points, not text.
            points: For every object, a few ``(x, y)`` pixel positions.
            **kwargs: Ignored; accepted for interface compatibility.

        Returns:
            ModelOutput | None: ``masks`` holds, for the one image, the list
            of per-object candidate stacks; ``None`` when nothing came back.

        Example:
            >>> out = sam.run_inference(  # doctest: +SKIP
            ...     'pano.jpg', points=[[(420, 260)]])
            >>> len(out.masks[0])  # doctest: +SKIP
            1
        """
        first = (
            image_inputs
            if isinstance(image_inputs, (str, Path))
            else next(iter(image_inputs), None)
        )
        masks = self.segment_at(first, points or []) if first is not None else []
        if not masks:
            return None
        return ModelOutput(
            text=None,
            masks=[masks],
            bounding_boxes=None,
            raw_response={'status': 'success', 'device': str(self.model.device)},
        )

    def segment_at(
        self,
        image_input: str | Path,
        points: Sequence[Sequence[Sequence[float]]],
    ) -> list[np.ndarray]:
        """
        Segment the object under each set of points.

        Args:
            image_input: Path or URL of the image.
            points: For every object asked about, a few ``(x, y)`` pixel
                positions on it. Every object needs the same number of
                points.

        Returns:
            list[np.ndarray]: For every object, a boolean
            ``(candidates, height, width)`` array of masks, from the part
            under the points to the whole object. Empty when the image
            cannot be read, nothing was asked or inference fails.

        Example:
            >>> masks = sam.segment_at(  # doctest: +SKIP
            ...     'pano.jpg', [[(420, 260)], [(700, 250)]])
            >>> len(masks), masks[0].dtype  # doctest: +SKIP
            (2, dtype('bool'))
        """
        asked = [[[float(x), float(y)] for x, y in group] for group in points]
        image = self._load_image_as_pil(image_input) if asked else None
        if image is None:
            return []
        try:
            inputs = self.processor(
                images=image,
                input_points=[asked],
                input_labels=[[[1] * len(group) for group in asked]],
                return_tensors='pt',
            )
            sizes = inputs['original_sizes']
            device = self.model.device
            tensors = {
                key: value.to(device, dtype=torch.float16)
                if value.is_floating_point()
                else value.to(device)
                for key, value in inputs.items()
                if key != 'original_sizes' and isinstance(value, torch.Tensor)
            }
            with torch.no_grad():
                outputs = self.model(**tensors, multimask_output=True)
            masks = self.processor.post_process_masks(
                outputs.pred_masks.float().cpu(), sizes
            )[0]
        except Exception as e:  # noqa: BLE001 - surfaced as no answer
            logger.error(f'SAM 3 point prompt failed: {e}')
            return []
        return [np.asarray(m).astype(bool) for m in masks]
