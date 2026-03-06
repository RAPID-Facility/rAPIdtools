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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from rapidtools.core import ImageAsset, PhysicalAsset, PhysicalAssetCollection
from rapidtools.models import GeminiInference


class GeminiAssetAnalyzer:
    """
    Pipeline component that uses Google's Gemini API to analyze assets.
    
    This analyzer gathers filtered images attached to a single `PhysicalAsset`, 
    sends them to Gemini together, and stores the combined assessment directly 
    in the asset's attributes.
    """

    def __init__(
        self,
        api_key: str | Path | None = None,
        prompt: str | Path = '',
        model_id: str = 'gemini-3.1-flash-image-preview',
        max_workers: int = 10,
        max_retries: int = 3,
        system_instruction: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 2048,
        image_filter: Callable[[ImageAsset], bool] | None = None
    ):
        """
        Initialize the Gemini Analyzer.

        Args:
            api_key: 
                Google Gemini API key (string, Path to a file, or None to use ENV vars).
            prompt: 
                The prompt to guide the model.
            model_id: 
                The specific Gemini model to use.
            max_workers: 
                Number of concurrent API threads.
            max_retries: 
                Number of times to retry failed requests.
            system_instruction: 
                Optional system-level instructions.
            temperature: 
                Model temperature (0.0 to 2.0).
            max_tokens: 
                Maximum output tokens.
            image_filter: 
                A function that takes an `ImageAsset` and returns True 
                if the image should be sent to Gemini. If None, all downloaded 
                images are sent.
        """
        self.prompt = prompt
        self.max_workers = max_workers
        self.image_filter = image_filter
        
        # Instantiate the underlying inference model
        self.model = GeminiInference(
            api_key=api_key,
            model_id=model_id,
            max_workers=max_workers,
            max_retries=max_retries,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens
        )

    @staticmethod
    def list_available_models(api_key: str | Path | None = None) -> list[str]:
        """
        List available Gemini models that support content generation.
        Convenience method that delegates to the underlying GeminiInference model.
        
        Args:
            api_key: Google Gemini API key (string, Path, or None for ENV var).
            
        Returns:
            A sorted list of available model IDs.
        """
        return GeminiInference.list_available_models(api_key=api_key)

    def __call__(
        self, 
        asset_collection: PhysicalAssetCollection
    ) -> PhysicalAssetCollection:
        """
        Execute the analysis process on the provided asset collection.
        """
        # Filter down to assets that actually have images
        assets_with_images =[
            asset for asset in asset_collection 
            if len(asset.image_assets) > 0
        ]
        
        if not assets_with_images:
            logging.warning('No assets with images found to analyze.')
            return asset_collection

        workers = min(self.max_workers, max(1, len(assets_with_images)))
        
        logging.info(
            f'Gemini API: Analyzing {len(assets_with_images)} assets concurrently '
            f'using {workers} workers...'
        )
        
        failed_assets = []
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_asset = {
                executor.submit(self._process_single_asset, asset): asset 
                for asset in assets_with_images
            }
            
            for future in tqdm(
                    as_completed(future_to_asset), 
                    total=len(assets_with_images), 
                    desc='Analyzing Assets'
                ):
                asset = future_to_asset[future]
                try:
                    success = future.result()
                    if not success:
                        failed_assets.append(asset)
                except Exception as e:
                    logging.debug(
                        f'Unhandled exception processing asset {asset.id}: {e}'
                    )
                    failed_assets.append(asset)
                    
        if failed_assets:
            logging.error(
                f'Gemini API: {len(failed_assets)} assets failed to process.'
            )
        else:
            logging.info('Gemini API: All assets processed successfully.')

        return asset_collection

    def _process_single_asset(self, asset: PhysicalAsset) -> bool:
        """
        Gathers filtered downloaded images for an asset, sends them to Gemini 
        as a list, and saves the output to the asset's attributes.
        """
        target_images = asset.image_assets
        
        if self.image_filter is not None:
            target_images = target_images.filter(self.image_filter)
            
        image_paths =[
            img.path for img in target_images 
            if img.is_downloaded
        ]
        
        if not image_paths:
            logging.debug(
                f'Asset {asset.id} has no valid downloaded images matching the'
                ' filter. Skipping.'
            )
            return False

        result = self.model.run_inference(
            image_inputs=image_paths, 
            prompt=self.prompt
        )
        
        if result is None or not result.text:
            return False
            
        # Store the AI description and metadata at the asset level
        asset.attributes['gemini_asset_analysis'] = result.text
        asset.attributes['ai_model_used'] = self.model.model_id
        asset.attributes['images_analyzed_count'] = len(image_paths)
        
        return True