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
# 05-24-2026

import logging
from collections.abc import Callable

from tqdm import tqdm

from rapidtools.core import ImageAsset, PhysicalAsset, PhysicalAssetCollection
from rapidtools.models import SAM3Inference


class SAM3ImageSegmenter:
    """
    Pipeline component that uses local SAM 3 to segment images attached to assets.

    This segmenter gathers filtered images across all `PhysicalAsset`s, batches them
    together (to maximize GPU utilization without causing OOM errors), and stores
    the resulting segmentation masks directly in the corresponding asset's attributes.
    """

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
    ) -> None:
        """
        Initialize the SAM 3 Image Segmenter.

        Args:
            prompt: 
                The text prompt (or list of prompts) to guide the segmentation
                (e.g., 'building' or ['building', 'tree']).
            model_id: 
                The Hugging Face repository ID for the SAM 3 model.
            device: 
                The compute device to use ('cuda', 'cpu', 'auto').
            load_in_4bit: 
                Whether to load the model using 4-bit quantization.
            batch_size: 
                Number of images to process simultaneously across assets.
            threshold: 
                Confidence threshold for predictions.
            mask_threshold: 
                Threshold for binarizing the masks.
            image_filter: 
                A function that takes an ``ImageAsset`` and returns ``True``
                if the image should be segmented. If ``None``, all downloaded
                images attached to the asset are processed.
        """
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

        Args:
            asset_collection: The collection of physical assets to process.

        Returns:
            The mutated ``PhysicalAssetCollection`` with segmentation masks added.
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
            logging.warning('No valid downloaded images found to segment.')
            return asset_collection

        logging.info(
            f'SAM 3: Segmenting {len(items_to_process)} images across '
            f'assets in batches of {self.batch_size}...'
        )

        failed_count = 0

        for i in tqdm(
            range(0, len(items_to_process), self.batch_size),
            desc='Segmenting Image Batches',
        ):
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
                    result.masks
                    if isinstance(result.masks, list)
                    else [result.masks]
                )

                boxes_list = [None] * len(batch)
                if result.bounding_boxes is not None:
                    boxes_list = (
                        result.bounding_boxes
                        if isinstance(result.bounding_boxes, list)
                        else [result.bounding_boxes]
                    )

                # Map the results back to the original assets and images:
                for (asset, img), masks, boxes in zip(batch, masks_list, boxes_list):
                    asset.attributes.setdefault('sam3_masks', {})
                    asset.attributes.setdefault('sam3_bounding_boxes', {})

                    asset.attributes['sam3_masks'][img.id] = masks

                    # Use 'is not None' to prevent ambiguous truth value crashes
                    # with numpy arrays:
                    if boxes is not None:
                        asset.attributes['sam3_bounding_boxes'][img.id] = boxes

                    asset.attributes['ai_model_used'] = self.model.model_id

            except Exception as e:
                logging.debug(f'Unhandled exception processing batch: {e}')
                failed_count += len(batch)

        if failed_count > 0:
            logging.error(f'SAM 3: {failed_count} images failed to process.')
        else:
            logging.info('SAM 3: All applicable images segmented successfully.')

        return asset_collection