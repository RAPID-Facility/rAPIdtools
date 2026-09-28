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
Batched, prompt-guided image segmentation for asset imagery.

This module provides :class:`SAM3ImageSegmenter`, a pipeline component that
runs a locally hosted SAM 3 (Segment Anything Model 3) over every downloaded
image attached to the assets of a
:class:`~rapidtools.core.PhysicalAssetCollection`. Masks and bounding boxes
are written back into each asset's ``attributes`` so that downstream steps
(e.g., :class:`~rapidtools.processing.BuildingRegularizer`) can georeference
them.

Example:
    >>> from rapidtools.processing import (
    ...     AerialImageryExtractor, Pipeline, SAM3ImageSegmenter
    ... )
    >>> pipeline = Pipeline([
    ...     AerialImageryExtractor('ortho.tif', save_directory='crops'),
    ...     SAM3ImageSegmenter(prompt=['building', 'roof'], batch_size=8),
    ... ])
    >>> result = pipeline.run(my_collection)
    >>> result['bldg_01'].attributes['sam3_masks'].keys()
    dict_keys(['bldg_01_aerial'])
"""

import logging
import threading
from collections.abc import Callable

from tqdm import tqdm

from rapidtools.core import (
    ImageAsset,
    PhysicalAsset,
    PhysicalAssetCollection,
    raise_if_cancelled,
)
from rapidtools.models import SAM3Inference

from .step import Stage

logger = logging.getLogger(__name__)


class SAM3ImageSegmenter:
    """
    Pipeline component that uses local SAM 3 to segment images attached to assets.

    This segmenter gathers filtered images across all ``PhysicalAsset``
    objects, batches them together (to maximize GPU utilization without
    causing OOM errors), and stores the resulting segmentation masks directly
    in the corresponding asset's attributes under the following keys:

        - ``'sam3_masks'``: ``{image_id: masks}`` where ``masks`` is the
          per-image mask stack returned by the model (typically an
          ``(N, H, W)`` boolean array).
        - ``'sam3_bounding_boxes'``: ``{image_id: boxes}`` (only when the
          model returns boxes).
        - ``'ai_model_used'``: The Hugging Face model ID that was used.

    Args:
        prompt (str | list[str], optional):
            The text prompt (or list of prompts) to guide the segmentation
            (e.g., ``'building'`` or ``['building', 'tree']``). Lists are
            joined into a single period-separated prompt. Defaults to ``''``.
        model_id (str, optional):
            The Hugging Face repository ID for the SAM 3 model. Defaults to
            ``'facebook/sam3'``.
        device (str, optional):
            The compute device to use (``'cuda'``, ``'cpu'``, ``'auto'``).
            Defaults to ``'auto'``.
        load_in_4bit (bool, optional):
            Whether to load the model using 4-bit quantization. Defaults to
            ``True``.
        batch_size (int, optional):
            Number of images to process simultaneously across assets.
            Defaults to 4.
        threshold (float, optional):
            Confidence threshold for predictions. Defaults to 0.5.
        mask_threshold (float, optional):
            Threshold for binarizing the masks. Defaults to 0.5.
        image_filter (Callable[[ImageAsset], bool] | None, optional):
            A function that takes an ``ImageAsset`` and returns ``True`` if
            the image should be segmented. If ``None``, all downloaded images
            attached to the asset are processed. Defaults to ``None``.
        cancel_event (threading.Event | None, optional):
            Cooperative cancellation flag. When set, the segmenter raises
            :class:`~rapidtools.core.OperationCancelled` before starting the
            next batch. Defaults to ``None``.

    Example:
        Segment only the aerial crops of each asset:

        >>> from rapidtools.processing import SAM3ImageSegmenter
        >>> segmenter = SAM3ImageSegmenter(
        ...     prompt='building',
        ...     batch_size=8,
        ...     image_filter=lambda img: img.id.endswith('_aerial'),
        ... )
        >>> collection = segmenter(my_collection)
        >>> masks = collection['bldg_01'].attributes['sam3_masks']
        >>> masks['bldg_01_aerial'].shape  # (num_instances, height, width)
        (2, 512, 512)
    """

    stage = Stage.SEGMENT

    def __init__(
        self,
        prompt: str | list[str] = '',
        model_id: str = 'facebook/sam3',
        device: str = 'auto',
        load_in_4bit: bool = True,
        batch_size: int = 4,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        image_filter: Callable[[ImageAsset], bool] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """
        Initialize the SAM 3 Image Segmenter and load the model once.

        Args:
            prompt (str | list[str], optional):
                The text prompt (or list of prompts) to guide the
                segmentation. Lists are joined with ``'. '``.
            model_id (str, optional):
                The Hugging Face repository ID for the SAM 3 model.
            device (str, optional):
                The compute device to use (``'cuda'``, ``'cpu'``, ``'auto'``).
            load_in_4bit (bool, optional):
                Whether to load the model using 4-bit quantization.
            batch_size (int, optional):
                Number of images to process simultaneously across assets.
            threshold (float, optional):
                Confidence threshold for predictions.
            mask_threshold (float, optional):
                Threshold for binarizing the masks.
            image_filter (Callable[[ImageAsset], bool] | None, optional):
                Predicate selecting which images of each asset to segment.
            cancel_event (threading.Event | None, optional):
                Cooperative cancellation flag checked between batches.
        """
        # Cooperative cancellation (stops between batches):
        self.cancel_event = cancel_event
        if isinstance(prompt, list):
            self.prompt = '. '.join(prompt)
        else:
            self.prompt = prompt

        self.image_filter = image_filter
        self.batch_size = batch_size
        self.threshold = threshold
        self.mask_threshold = mask_threshold

        # Instantiate the underlying inference model ONCE to save load time:
        self.model = SAM3Inference(
            model_id=model_id,
            device=device,
            load_in_4bit=load_in_4bit,
        )

    def __call__(
        self,
        asset_collection: PhysicalAssetCollection,
    ) -> PhysicalAssetCollection:
        """
        Execute the segmentation process on the provided asset collection.

        Images are collected from every asset (optionally filtered by
        ``image_filter``), restricted to those that exist on disk, and
        processed in batches of ``batch_size``. A batch that fails (model
        returns ``None`` or raises) is counted and logged, but does not stop
        the remaining batches.

        Args:
            asset_collection (PhysicalAssetCollection):
                The collection of physical assets to process.

        Returns:
            PhysicalAssetCollection:
                The mutated collection with ``'sam3_masks'``,
                ``'sam3_bounding_boxes'`` and ``'ai_model_used'`` written to
                the attributes of every asset that had at least one
                successfully segmented image.

        Raises:
            OperationCancelled:
                If ``cancel_event`` is set between two batches.

        Example:
            >>> segmenter = SAM3ImageSegmenter(prompt='utility pole')
            >>> collection = segmenter(my_collection)
        """
        items_to_process: list[tuple[PhysicalAsset, ImageAsset]] = []

        for asset in asset_collection:
            target_images = asset.image_assets

            if self.image_filter is not None:
                target_images = target_images.filter(self.image_filter)

            for img in target_images:
                if img.is_downloaded:
                    items_to_process.append((asset, img))

        if not items_to_process:
            logger.warning('No valid downloaded images found to segment.')
            return asset_collection

        logger.info(
            f'SAM 3: Segmenting {len(items_to_process)} images across '
            f'assets in batches of {self.batch_size}...'
        )

        failed_count = 0

        for i in tqdm(
            range(0, len(items_to_process), self.batch_size),
            desc='Segmenting Image Batches',
        ):
            raise_if_cancelled(self.cancel_event, 'SAM 3 segmentation')
            batch = items_to_process[i : i + self.batch_size]
            image_paths = [str(img.path) for _, img in batch]

            try:
                # Run inference on the batch of images:
                result = self.model.run_inference(
                    image_inputs=image_paths,
                    prompt=self.prompt,
                    threshold=self.threshold,
                    mask_threshold=self.mask_threshold,
                )

                if result is None or result.masks is None:
                    failed_count += len(batch)
                    continue

                # Ensure the outputs are lists (in case of batch_size=1 remaining):
                masks_list = (
                    result.masks if isinstance(result.masks, list) else [result.masks]
                )

                boxes_list: list = [None] * len(batch)
                if result.bounding_boxes is not None:
                    boxes_list = (
                        result.bounding_boxes
                        if isinstance(result.bounding_boxes, list)
                        else [result.bounding_boxes]
                    )

                # Map the results back to the original assets and images:
                for (asset, img), masks, boxes in zip(
                    batch, masks_list, boxes_list, strict=False
                ):
                    asset.attributes.setdefault('sam3_masks', {})
                    asset.attributes.setdefault('sam3_bounding_boxes', {})

                    asset.attributes['sam3_masks'][img.id] = masks

                    # Use 'is not None' to prevent ambiguous truth value crashes
                    # with numpy arrays:
                    if boxes is not None:
                        asset.attributes['sam3_bounding_boxes'][img.id] = boxes

                    asset.attributes['ai_model_used'] = self.model.model_id

            except Exception as e:
                logger.debug(f'Unhandled exception processing batch: {e}')
                failed_count += len(batch)

        if failed_count > 0:
            logger.error(f'SAM 3: {failed_count} images failed to process.')
        else:
            logger.info('SAM 3: All applicable images segmented successfully.')

        return asset_collection
