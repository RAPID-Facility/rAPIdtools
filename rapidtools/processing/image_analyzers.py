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
# 05-25-2026

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from rapidtools.core import ImageAsset, PhysicalAsset, PhysicalAssetCollection
from rapidtools.models import GeminiInference, Gemma4Inference


class GeminiAssetAnalyzer:
    """
    Pipeline component that uses Google's Gemini API to analyze assets.

    This analyzer features a thread-safe "Global Cooldown" with an exponential
    multiplier. If the API returns a rate-limit error, all threads will pause.
    Consecutive errors will result in increasingly longer wait times.

    Example:
        >>> analyzer = GeminiAssetAnalyzer(
        ...     api_key="...",
        ...     cooldown_duration=30,  # Base wait of 30s
        ...     max_workers=2
        ... )
        >>> collection = analyzer(collection)
    """

    def __init__(
        self,
        api_key: str | Path | None = None,
        prompt: str | Path = '',
        model_id: str | None = 'gemini-3.1-flash-lite-preview',
        max_workers: int | None = 5,
        max_retries: int | None = 3,
        cooldown_duration: int | None = 30,
        system_instruction: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 2048,
        image_filter: Callable[[ImageAsset], bool] | None = None
    ):
        """
        Initialize the Gemini Analyzer with configurable safety limits.

        Args:
            api_key: Google Gemini API key.
            prompt: Text prompt or path to prompt file.
            model_id: Gemini model ID. Defaults to 'gemini-1.5-flash'.
            max_workers: Concurrent threads. Defaults to 5.
            max_retries: Retries per individual request. Defaults to 3.
            cooldown_duration: Base seconds to wait on rate limit. Defaults to 30.
            system_instruction: Optional model persona/constraints.
            temperature: Randomness (0.0 to 2.0).
            max_tokens: Maximum output length.
            image_filter: Optional function to filter ImageAssets.
        """
        self.prompt = prompt
        self.image_filter = image_filter

        # 1. Set configuration
        self.model_id = model_id
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.cooldown_duration = cooldown_duration

        # 2. Initialize thread-safe cooldown state
        self._lock = threading.Lock()
        self._global_cooldown_until = 0.0
        self._consecutive_error_count = 0

        # 3. Instantiate inference engine
        self.model = GeminiInference(
            api_key=api_key,
            model_id=self.model_id,
            max_workers=self.max_workers,
            max_retries=self.max_retries,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def _parse_structured_results(self, text: str) -> dict[str, Any]:
        """
        Attempts to extract key-value pairs from Gemini's response.
        Supports both JSON formatting and 'Key: Value' line-based formatting.
        """
        results = {}

        # 1. Try to find and parse JSON blocks (wrapped in ```json ... ```)
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        json_text = json_match.group(1) if json_match else text

        try:
            results = json.loads(json_text)
            if isinstance(results, dict):
                return results
        except json.JSONDecodeError:
            pass # Fallback to regex line parsing

        # 2. Fallback: Regex parsing for 'Key: Value' or 'Key - Value' patterns
        # Matches: "CHS Level: 3" or "Justification: The roof is gone"
        lines = text.split('\n')
        for line in lines:
            match = re.match(r'^([\w\s]+)[:\-]\s*(.*)$', line.strip())
            if match:
                key = match.group(1).strip().lower().replace(' ', '_')
                value = match.group(2).strip()
                results[key] = value

        return results

    def _process_single_asset(self, asset: PhysicalAsset) -> bool:
        """
        Gathers images for an asset, handles global cooldowns, and runs inference.
        """
        # --- Check Global Cooldown ---
        with self._lock:
            wait_time = self._global_cooldown_until - time.time()
            if wait_time > 0:
                logging.info(
                    f'Asset {asset.id} delayed: Global cooldown active for '
                    f'{int(wait_time)}s...'
                )
                time.sleep(wait_time)

        # --- Gather Images ---
        target_images = asset.image_assets
        if self.image_filter is not None:
            target_images = target_images.filter(self.image_filter)

        image_paths = [img.path for img in target_images if img.is_downloaded]
        if not image_paths:
            return False

        # --- Run Inference ---
        result = self.model.run_inference(
            image_inputs=image_paths,
            prompt=self.prompt
        )

        # --- Handle Result and Dynamic Cooldown ---
        if result is None or not result.text:
            with self._lock:
                # Increment error count to increase the next cooldown
                self._consecutive_error_count += 1

                # Dynamic wait: Base Duration * Number of consecutive errors
                # Example: 30s, 60s, 90s...
                dynamic_wait = self.cooldown_duration * self._consecutive_error_count

                self._global_cooldown_until = time.time() + dynamic_wait

                logging.error(
                    f'Rate limit or API error hit on {asset.id}. '
                    f'Consecutive errors: {self._consecutive_error_count}. '
                    f'Global cooldown set for {dynamic_wait}s.'
                )
            return False

        # --- Success Case ---
        with self._lock:
            # Reset error count on any successful call
            if self._consecutive_error_count > 0:
                logging.info(
                    'Successful response received. Resetting error multiplier.'
                )
                self._consecutive_error_count = 0

        # Parse the text for specific categories
        structured_data = self._parse_structured_results(result.text)

        if structured_data:
            for key, val in structured_data.items():
                # Store as gemini_chs_level, gemini_justification, etc.
                asset.attributes[f'gemini_{key}'] = val
        else:
            # Fallback if no patterns found: store the whole thing
            asset.attributes['gemini_analysis_raw'] = result.text

        asset.attributes['ai_model_used'] = self.model.model_id
        asset.attributes['images_analyzed_count'] = len(image_paths)

        return True

    def __call__(
        self,
        asset_collection: PhysicalAssetCollection
    ) -> PhysicalAssetCollection:
        """
        Execute analysis on the collection using a thread pool.
        """
        assets_with_images = [
            asset for asset in asset_collection
            if len(asset.image_assets) > 0
        ]

        if not assets_with_images:
            logging.warning('No assets with images found to analyze.')
            return asset_collection

        workers = min(self.max_workers, max(1, len(assets_with_images)))

        logging.info(
            f'Gemini API: Analyzing {len(assets_with_images)} assets '
            f'with {workers} workers (Base Cooldown: {self.cooldown_duration}s).'
        )

        failed_count = 0
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
                    if not future.result():
                        failed_count += 1
                except Exception as e:
                    logging.error(f'Error on asset {asset.id}: {e}')
                    failed_count += 1

        if failed_count > 0:
            logging.error(f'Gemini API: {failed_count} assets failed to process.')
        else:
            logging.info('Gemini API: All assets processed successfully.')

        return asset_collection

class Gemma4AssetAnalyzer:
    """
    Batched pipeline component that uses Google's local Gemma-4 Vision-Language
    Model to analyze physical assets based on their associated images.

    This analyzer processes assets in batches to maximize GPU utilization and
    throughput. Users with high-VRAM hardware can increase the batch size and
    max images per asset, while those with consumer GPUs (e.g., 8GB VRAM) should
    keep these values low to prevent Out-Of-Memory (OOM) errors.

    Example:
        >>> analyzer = Gemma4AssetAnalyzer(
        ...     prompt='Analyze the roof damage in these images and return JSON.',
        ...     model_id='google/gemma-4-E2B-it',
        ...     device="cuda",
        ...     batch_size=4
        ... )
        >>> collection = analyzer(collection)
    """

    def __init__(
        self,
        prompt: str | Path = '',
        model_id: str = 'google/gemma-4-E2B-it',
        device: str = 'auto',
        temperature: float = 0.4,
        max_tokens: int = 512,
        max_images_per_asset: int = 2,
        batch_size: int = 4,
        image_filter: Callable[[ImageAsset], bool] | None = None
    ) -> None:
        """
        Initialize the batched Gemma-4 Analyzer.

        Args:
            prompt:
                The text instruction or path to a text file containing the prompt
                to guide the vision model's analysis.
            model_id:
                The Hugging Face identifier for the Gemma-4 model.
                Defaults to the 2-Billion parameter variant
                (``'google/gemma-4-E2B-it'``).
            device:
                The hardware device to load the model on (e.g., 'auto', 'cuda').
            temperature:
                Sampling temperature for text generation. Lower values produce
                more deterministic outputs. Defaults to 0.4.
            max_tokens:
                The maximum number of tokens to generate per inference call.
                Lowering this speeds up processing. Defaults to 512.
            max_images_per_asset:
                Safeguard limit for the number of images passed to the model
                per asset. Defaults to 2.
            batch_size:
                The number of assets to process simultaneously. Increase this
                for higher throughput if you have high VRAM capacity. Defaults
                to 4.
            image_filter:
                An optional function to filter which `ImageAsset`s should be
                included in the analysis.
        """
        # Resolve prompt file if a Path is provided
        self.prompt = prompt.read_text(encoding='utf-8') if isinstance(
            prompt, Path
        ) else prompt

        self.image_filter = image_filter
        self.max_images_per_asset = max_images_per_asset
        self.batch_size = batch_size

        # Instantiate the local model handler
        self.model = Gemma4Inference(
            model_id=model_id,
            device=device,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def _parse_structured_results(self, text: str) -> dict[str, Any]:
        """
        Extracts key-value pairs from the generated text.

        Attempts to find and parse a JSON block (wrapped in ```json ... ```).
        If no valid JSON is found, falls back to parsing line-by-line using
        a standard 'Key: Value' regex pattern.

        Args:
            text: The raw text string generated by the language model.

        Returns:
            dict[str, Any]: A dictionary containing the parsed attributes.
        """
        results = {}

        # 1. Attempt to parse markdown-formatted JSON blocks:
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        json_text = json_match.group(1) if json_match else text

        try:
            # Use a temp variable 'parsed' so 'results' are not accidentally 
            # overwritten if the LLM returns a string or list instead of a 
            # dictionary:
            parsed = json.loads(json_text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass # Fall back to regex parsing

        # 2. Fall back to regex parsing for line-based key-value formats:
        lines = text.split('\n')
        for line in lines:
            match = re.match(r'^([\w\s]+)[:\-]\s*(.*)$', line.strip())
            if match:
                key = match.group(1).strip().lower().replace(' ', '_')
                results[key] = match.group(2).strip()

        return results

    def __call__(
        self,
        asset_collection: PhysicalAssetCollection
    ) -> PhysicalAssetCollection:
        """
        Executes batched multimodal inference on the provided collection.

        Filters the collection for assets with downloaded images, chunks them
        into batches, processes them via the local Gemma-4 model, and appends
        the parsed results directly into the asset's attributes.

        Args:
            asset_collection:
                The collection of physical assets to analyze.

        Returns:
            PhysicalAssetCollection:
                The updated collection with AI-generated attributes.
        """
        # Isolate assets that actually have images to analyze
        assets_with_images = [a for a in asset_collection if len(a.image_assets) > 0]

        if not assets_with_images:
            logging.warning('No assets with images found to analyze. Skipping.')
            return asset_collection

        failed_count = 0
        total_batches = math.ceil(len(assets_with_images) / self.batch_size)

        logging.info(
            f'Analyzing {len(assets_with_images)} assets '
            f'in batches of {self.batch_size}...'
        )

        for i in tqdm(
                range(0, len(assets_with_images), self.batch_size),
                total=total_batches, desc='Batch Analysis'
            ):
            batch_assets = assets_with_images[i:i + self.batch_size]

            # Prepare image inputs for the batch:
            batch_images = []
            for asset in batch_assets:
                target_images = asset.image_assets
                if self.image_filter is not None:
                    target_images = target_images.filter(self.image_filter)

                img_paths = [img.path for img in target_images if img.is_downloaded]

                # Truncate images to prevent OOM:
                if len(img_paths) > self.max_images_per_asset:
                    img_paths = img_paths[:self.max_images_per_asset]
                batch_images.append(img_paths)

            # Replicate the prompt for each item in the batch:
            batch_prompts = [self.prompt] * len(batch_assets)

            # Execute inference:
            results = self.model.run_inference_batch(batch_images, batch_prompts)

            # Parse and assign results back to the assets:
            for asset, img_paths, result in zip(
                    batch_assets,
                    batch_images,
                    results,
                    strict=True
                ):
                if result is None or not result.text:
                    failed_count += 1
                    continue

                structured_data = self._parse_structured_results(result.text)
                if structured_data:
                    for key, val in structured_data.items():
                        asset.attributes[f'gemma4_{key}'] = val
                else:
                    # Save raw output if structured parsing failed:
                    asset.attributes['gemma4_analysis_raw'] = result.text

                asset.attributes['ai_model_used'] = self.model.model_id
                asset.attributes['images_analyzed_count'] = len(img_paths)

            # Periodically clear VRAM to avoid fragmentation over long runs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if failed_count > 0:
            logging.error(f'{failed_count} assets failed to process.')
        else:
            logging.info('All assets processed successfully.')

        return asset_collection
