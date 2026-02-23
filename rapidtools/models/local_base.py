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
from collections.abc import Generator, Iterable
from io import BytesIO
from pathlib import Path
from typing import Any

import requests

try:
    from PIL import Image
except ImportError:
    logging.warning('Pillow is not installed. Local models requiring PIL Images will fail.')

from tqdm import tqdm

from .base import BaseInferenceModel, ModelOutput


class BaseLocalInferenceModel(BaseInferenceModel):
    """
    Intermediate base class for local Hugging Face / PyTorch models.
    Handles VRAM-safe sequential batching and PIL Image loading.
    """

    def __init__(self, device: str = 'auto', temperature: float = 0.4, max_tokens: int = 2048):
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _load_image_as_pil(self, image_input: str | Path) -> Any | None:
        """Universal PIL Image loader for local PyTorch models."""
        input_str = str(image_input)
        try:
            if input_str.startswith(('http://', 'https://')):
                logging.info(f'Downloading image to memory: {input_str}')
                response = requests.get(input_str, timeout=10)
                response.raise_for_status()
                # Open image and ensure it has 3 channels (RGB) to prevent tensor shape errors
                return Image.open(BytesIO(response.content)).convert('RGB')
            else:
                path_obj = Path(image_input)
                if not path_obj.exists():
                    logging.error(f'File not found: {path_obj}')
                    return None
                return Image.open(path_obj).convert('RGB')
        except Exception as e:
            logging.error(f'Failed to load image {input_str} for local inference: {e}')
            return None

    def run_batch(
        self,
        asset_inputs: Iterable[tuple[str, Any]],
        prompt: str | Path,
        **kwargs
    ) -> Generator[tuple[str, str, ModelOutput | str], None, None]:
        """
        OVERRIDES BaseInferenceModel.run_batch!
        Local models must process sequentially to avoid GPU Out-Of-Memory (OOM) crashes.
        """
        prompt_str = self._resolve_prompt(prompt)
        asset_list = list(asset_inputs)

        if not asset_list:
            logging.warning('No assets provided to local run_batch.')
            return

        logging.info(f'Starting local batch of {len(asset_list)} assets sequentially to protect VRAM.')

        with tqdm(total=len(asset_list), unit='asset', desc='Local Processing') as pbar:
            for asset_id, img_inputs in asset_list:
                try:
                    result = self.run_inference(img_inputs, prompt_str, **kwargs)
                    if result:
                        yield asset_id, 'ok', result
                    else:
                        yield asset_id, 'failed', 'Inference returned None.'
                except Exception as e:
                    logging.error(f'Local inference crash on {asset_id}: {e}')
                    yield asset_id, 'failed', f'Local GPU/CPU Exception: {str(e)}'
                finally:
                    pbar.update(1)
