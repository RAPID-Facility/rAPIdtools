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
# 03-05-2026

import logging
from collections.abc import Callable

from tqdm import tqdm

from rapidtools.core import ImageAsset, PhysicalAsset, PhysicalAssetCollection
from rapidtools.models import SAM3Inference


class SAM3ImageSegmenter:
    """
    Pipeline component that uses local SAM 3 to segment images attached to assets.
    
    This segmenter gathers filtered images for each `PhysicalAsset`, runs them 
    through SAM 3 sequentially (to avoid GPU OOM errors), and stores the resulting 
    segmentation masks directly in the asset's attributes for later downstream use.
    """

    def __init__(
        self,
        prompt: str = '',
        model_id: str = 'facebook/sam3',
        device: str = 'auto',
        image_filter: Callable[[ImageAsset], bool] | None = None
    ):
        """
        Initialize the SAM 3 Image Segmenter.

        Args:
            prompt: 
                The text prompt to guide the segmentation (e.g., "building", "vegetation").
            model_id: 
                The Hugging Face repository ID for the SAM 3 model.
            device: 
                The compute device to use ('cuda', 'cpu', 'auto').
            image_filter: 
                A function that takes an `ImageAsset` and returns True 
                if the image should be segmented. If None, all downloaded 
                images attached to the asset are processed.
        """
        self.prompt = prompt
        self.image_filter = image_filter
        
        # Instantiate the underlying inference model ONCE to save load time
        self.model = SAM3Inference(
            model_id=model_id,
            device=device
        )

    def __call__(
        self, 
        asset_collection: PhysicalAssetCollection
    ) -> PhysicalAssetCollection:
        """
        Execute the segmentation process on the provided asset collection.
        """
        # Filter down to assets that actually have images
        assets_with_images =[
            asset for asset in asset_collection 
            if len(asset.image_assets) > 0
        ]
        
        if not assets_with_images:
            logging.warning('No assets with images found to segment.')
            return asset_collection

        logging.info(f'SAM 3: Segmenting images for {len(assets_with_images)} assets locally...')
        
        failed_assets =[]
        
        # Process SEQUENTIALLY to protect GPU VRAM
        for asset in tqdm(assets_with_images, desc='Segmenting Asset Images'):
            try:
                success = self._process_single_asset(asset)
                if not success:
                    failed_assets.append(asset)
            except Exception as e:
                logging.debug(f'Unhandled exception processing asset {asset.id}: {e}')
                failed_assets.append(asset)
                    
        if failed_assets:
            logging.error(f'SAM 3: {len(failed_assets)} assets failed to process.')
        else:
            logging.info('SAM 3: All applicable assets segmented successfully.')

        return asset_collection

def _process_single_asset(self, asset: PhysicalAsset) -> bool:
        """
        Gathers filtered downloaded images for an asset, runs SAM 3, 
        and saves the masks mapped specifically to each image's ID.
        """
        target_images = asset.image_assets
        
        if self.image_filter is not None:
            target_images = target_images.filter(self.image_filter)
            
        # Keep the actual ImageAsset objects so we can use their IDs later
        valid_images =[img for img in target_images if img.is_downloaded]
        
        if not valid_images:
            return False

        # Extract just the paths to send to the model
        image_paths = [img.path for img in valid_images]

        # Run inference (batch processes all images at once)
        result = self.model.run_inference(
            image_inputs=image_paths, 
            prompt=self.prompt
        )
        
        if result is None or result.masks is None:
            return False
            
        # 1. Ensure the outputs are lists (if only 1 image was passed, 
        # the inference class unwraps it, so we wrap it back up to zip safely)
        masks_list = result.masks if isinstance(result.masks, list) else [result.masks]
        
        boxes_list = [None] * len(valid_images)
        if result.bounding_boxes:
            boxes_list = result.bounding_boxes if isinstance(result.bounding_boxes, list) else [result.bounding_boxes]

        # 2. Prepare the dictionaries in the asset attributes
        asset.attributes.setdefault('sam3_masks', {})
        asset.attributes.setdefault('sam3_bounding_boxes', {})

        # 3. Map the masks directly to the specific Image ID!
        for img, masks, boxes in zip(valid_images, masks_list, boxes_list):
            asset.attributes['sam3_masks'][img.id] = masks
            
            if boxes:
                asset.attributes['sam3_bounding_boxes'][img.id] = boxes
                
        asset.attributes['ai_model_used'] = self.model.model_id
        
        return True