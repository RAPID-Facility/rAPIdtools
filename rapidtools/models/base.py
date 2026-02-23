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
# 02-22-2026

import logging
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm


@dataclass
class ModelOutput:
    """
    A universal container for multimodal AI model outputs.
    Belongs strictly to the `models` domain.
    """
    text: str | None = None
    masks: Any | None = None          # e.g., numpy arrays for SAM 3
    bounding_boxes: list | None = None
    raw_response: Any | None = None   # Raw JSON from APIs or raw Tensors from local models

    @property
    def has_text(self) -> bool:
        return self.text is not None and len(self.text.strip()) > 0

    @property
    def has_masks(self) -> bool:
        return self.masks is not None

    @property
    def has_bounding_boxes(self) -> bool:
        return self.bounding_boxes is not None and len(self.bounding_boxes) > 0


class BaseInferenceModel(ABC):
    """
    The absolute top-level contract for all AI models (API or Local).
    """

    @staticmethod
    def _resolve_prompt(prompt: str | Path) -> str:
        """Universal helper to resolve a string or file path into text."""
        prompt_str = str(prompt)
        try:
            prompt_path = Path(prompt)
            if prompt_path.is_file():
                return prompt_path.read_text(encoding='utf-8').strip()
        except OSError:
            pass
        return prompt_str

    @abstractmethod
    def run_inference(
        self,
        image_inputs: Any,
        prompt: str | Path,
        **kwargs
    ) -> ModelOutput | None:
        """Executes a single inference pass and returns a standardized ModelOutput."""
        pass

    def run_batch(
        self,
        asset_inputs: Iterable[tuple[str, Any]],
        prompt: str | Path,
        **kwargs
    ) -> Generator[tuple[str, str, ModelOutput | str], None, None]:
        """
        Universal parallel batch processing.
        Yields: (asset_id, status, ModelOutput_or_error_msg)
        """
        prompt_str = self._resolve_prompt(prompt)

        def _process(asset_id: str, img_inputs: Any) -> tuple[str, str, ModelOutput | str]:
            result = self.run_inference(img_inputs, prompt_str, **kwargs)
            if result:
                return asset_id, 'ok', result
            return asset_id, 'failed', 'Inference failed or was blocked.'

        asset_list = list(asset_inputs)
        if not asset_list:
            logging.warning('No assets provided to run_batch.')
            return

        # Use the child instance's max_workers if available, default to 10
        effective_workers = min(getattr(self, 'max_workers', 10), 10)
        logging.info(f'Starting batch of {len(asset_list)} assets using {effective_workers} threads.')

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_asset = {
                executor.submit(_process, a_id, imgs): a_id
                for a_id, imgs in asset_list
            }

            with tqdm(total=len(asset_list), unit='asset', desc='Processing') as pbar:
                for future in as_completed(future_to_asset):
                    asset_id = future_to_asset[future]
                    try:
                        yield future.result()
                    except Exception as e:
                        yield asset_id, 'failed', f'Thread Exception: {str(e)}'
                    finally:
                        pbar.update(1)
