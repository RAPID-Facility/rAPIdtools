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
Shared plumbing for local Hugging Face / PyTorch models.

:class:`BaseLocalInferenceModel` loads images into PIL and replaces the
threaded :meth:`~rapidtools.models.base.BaseInferenceModel.run_batch` with a
sequential loop so several requests never compete for GPU memory.

Example:
    >>> from rapidtools.models import Gemma4Inference
    >>> model = Gemma4Inference()
    >>> for asset_id, status, out in model.run_batch(
    ...     [('b1', ['b1.jpg']), ('b2', ['b2.jpg'])], 'Rate damage 0-5.'
    ... ):
    ...     print(asset_id, status)
    b1 ok
    b2 ok
"""

from __future__ import annotations

import logging
from collections.abc import Generator, Iterable
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image
from tqdm import tqdm

from rapidtools.auth import ensure_huggingface_login

from .base import BaseInferenceModel, ModelOutput

logger = logging.getLogger(__name__)


class BaseLocalInferenceModel(BaseInferenceModel):
    """
    Intermediate base class for local Hugging Face / PyTorch models.

    Attributes:
        device: Requested device (``'auto'``, ``'cuda'``, ``'cpu'``).
        temperature: Default sampling temperature.
        max_tokens: Default maximum number of new tokens.

    Example:
        >>> from rapidtools.models import HFVisionInference
        >>> from rapidtools.models.local_base import BaseLocalInferenceModel
        >>> issubclass(HFVisionInference, BaseLocalInferenceModel)
        True
    """

    def __init__(
        self, device: str = 'auto', temperature: float = 0.4, max_tokens: int = 2048
    ):
        """
        Store generation defaults and make sure a Hugging Face token exists.

        Args:
            device: ``'auto'``, ``'cuda'`` or ``'cpu'``.
            temperature: Default sampling temperature.
            max_tokens: Default maximum number of new tokens.
        """
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Gated Hugging Face repositories need a token; log in lazily here
        # instead of at import time.
        ensure_huggingface_login()

    def _load_image_as_pil(self, image_input: str | Path) -> Any | None:
        """
        Load an image from disk or a URL as an RGB PIL image.

        Args:
            image_input: Local path or ``http(s)://`` URL.

        Returns:
            PIL.Image.Image | None: The RGB image, or ``None`` when it cannot
            be read (the error is logged).

        Example:
            >>> from rapidtools.models import Gemma4Inference
            >>> model = Gemma4Inference()
            >>> img = model._load_image_as_pil('roof.jpg')
            >>> img.mode
            'RGB'
        """
        input_str = str(image_input)
        try:
            if input_str.startswith(('http://', 'https://')):
                logger.info(f'Downloading image to memory: {input_str}')
                response = requests.get(input_str, timeout=10)
                response.raise_for_status()
                # Force 3 channels to prevent tensor shape errors downstream.
                return Image.open(BytesIO(response.content)).convert('RGB')
            path_obj = Path(image_input)
            if not path_obj.exists():
                logger.error(f'File not found: {path_obj}')
                return None
            return Image.open(path_obj).convert('RGB')
        except Exception as e:  # noqa: BLE001 - reported and skipped
            logger.error(f'Failed to load image {input_str} for local inference: {e}')
            return None

    def run_batch(
        self,
        asset_inputs: Iterable[tuple[str, Any]],
        prompt: str | Path,
        **kwargs,
    ) -> Generator[tuple[str, str, ModelOutput | str], None, None]:
        """
        Process assets sequentially to avoid GPU out-of-memory crashes.

        Overrides :meth:`BaseInferenceModel.run_batch`.

        Args:
            asset_inputs: ``(asset_id, image_inputs)`` pairs.
            prompt: Prompt text or file shared by every asset.
            **kwargs: Forwarded to :meth:`run_inference`.

        Yields:
            tuple[str, str, ModelOutput | str]: ``(asset_id, status, payload)``
            with ``status`` ``'ok'`` or ``'failed'``.
        """
        prompt_str = self._resolve_prompt(prompt)
        asset_list = list(asset_inputs)

        if not asset_list:
            logger.warning('No assets provided to local run_batch.')
            return

        logger.info(
            f'Starting local batch of {len(asset_list)} assets sequentially '
            'to protect VRAM.'
        )

        with tqdm(total=len(asset_list), unit='asset', desc='Local Processing') as pbar:
            for asset_id, img_inputs in asset_list:
                try:
                    result = self.run_inference(img_inputs, prompt_str, **kwargs)
                    if result:
                        yield asset_id, 'ok', result
                    else:
                        yield asset_id, 'failed', 'Inference returned None.'
                except Exception as e:  # noqa: BLE001 - reported per asset
                    logger.error(f'Local inference crash on {asset_id}: {e}')
                    yield asset_id, 'failed', f'Local GPU/CPU Exception: {str(e)}'
                finally:
                    pbar.update(1)
