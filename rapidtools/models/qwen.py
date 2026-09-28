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
Qwen vision-language models: hosted (Alibaba Cloud Model Studio) and local.

* :class:`QwenInference` calls the OpenAI-compatible endpoint of Alibaba Cloud
  Model Studio (DashScope). Keys come from ``DASHSCOPE_API_KEY`` (or
  ``QWEN_API_KEY``). The international (Singapore) endpoint is the default;
  pass ``region='cn'`` for the Beijing endpoint.
* :class:`QwenVisionInference` runs open-weight Qwen checkpoints (Qwen3.8,
  Qwen3.6, Qwen3.5, Qwen3-VL, Qwen2.5-VL) locally through Transformers.

Example:
    >>> from rapidtools.models import QwenInference, QwenVisionInference
    >>> api = QwenInference(api_key='sk-...', model_id='qwen3.8-max')
    >>> api.run_inference('roof.jpg', 'Describe the roof damage.').text
    >>> local = QwenVisionInference('Qwen/Qwen3.5-4B')
    >>> local.run_inference('roof.jpg', 'Describe the roof damage.').text
"""

from __future__ import annotations

from pathlib import Path

from .api_base import API_REQUEST_TIMEOUT
from .catalogs import (
    PROVIDERS,
    QWEN_DEFAULT_MODEL,
    QWEN_MODEL_CATALOG,
    QWEN_VL_DEFAULT_MODEL,
    QWEN_VL_MODEL_CATALOG,
)
from .hf_vision import HFVisionInference
from .openai_compat import BaseOpenAICompatibleInference

QWEN_INTL_BASE_URL = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
QWEN_CN_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
QWEN_REGION_BASE_URLS = {'intl': QWEN_INTL_BASE_URL, 'cn': QWEN_CN_BASE_URL}


class QwenInference(BaseOpenAICompatibleInference):
    """
    Alibaba Cloud Model Studio (DashScope) client for hosted Qwen models.

    Example:
        >>> from rapidtools.models import QwenInference
        >>> QwenInference.list_known_models()[0]
        'qwen3.8-max'
        >>> model = QwenInference(api_key='sk-...', region='intl')
        >>> out = model.run_inference('roof.jpg', 'Is the roof intact?')
        >>> print(out.text)
    """

    INFO = PROVIDERS['qwen']

    PROVIDER_NAME = 'Qwen (Model Studio)'
    BASE_URL = QWEN_INTL_BASE_URL
    API_KEY_ENV = ('DASHSCOPE_API_KEY', 'QWEN_API_KEY')
    DEFAULT_MODEL = QWEN_DEFAULT_MODEL
    MODEL_CATALOG = QWEN_MODEL_CATALOG
    MAX_TOKENS_FIELD = 'max_tokens'
    IMAGE_DETAIL = None

    def __init__(
        self,
        api_key: str | Path | None = None,
        model_id: str | None = None,
        max_workers: int = 10,
        max_retries: int = 3,
        system_instruction: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 2048,
        region: str = 'intl',
        base_url: str | None = None,
        timeout: float = API_REQUEST_TIMEOUT,
    ):
        """
        Initialise the client.

        Args:
            api_key: Key string, key-file path, or ``None`` to read
                ``DASHSCOPE_API_KEY`` / ``QWEN_API_KEY``.
            model_id: Hosted model name. Defaults to ``'qwen3.8-max'``.
            max_workers: Threads used by :meth:`run_batch`.
            max_retries: Retries for transient HTTP failures.
            system_instruction: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens generated per response.
            region: ``'intl'`` (Singapore) or ``'cn'`` (Beijing) endpoint.
            base_url: Explicit endpoint root; overrides ``region``.
            timeout: Seconds to wait for each response (default 60).

        Raises:
            ValueError: If ``region`` is unknown or no key can be resolved.
        """
        if base_url is None:
            if region not in QWEN_REGION_BASE_URLS:
                raise ValueError(
                    f'Unknown Qwen region {region!r}; expected one of '
                    f'{sorted(QWEN_REGION_BASE_URLS)}.'
                )
            base_url = QWEN_REGION_BASE_URLS[region]
        self.region = region
        super().__init__(
            api_key=api_key,
            model_id=model_id,
            max_workers=max_workers,
            max_retries=max_retries,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            timeout=timeout,
        )


class QwenVisionInference(HFVisionInference):
    """
    Local inference with open-weight Qwen vision-language checkpoints.

    A specialisation of :class:`~rapidtools.models.HFVisionInference` whose
    catalogue lists the Qwen3.8 / Qwen3.6 / Qwen3.5 natively multimodal
    models and the Qwen3-VL / Qwen2.5-VL families.

    Example:
        >>> from rapidtools.models import QwenVisionInference
        >>> QwenVisionInference.list_available_models()[0]
        'Qwen/Qwen3.8-27B'
        >>> model = QwenVisionInference('Qwen/Qwen3.5-9B', load_in_4bit=True)
        >>> model.run_inference('roof.jpg', 'Describe the roof.').text
    """

    INFO = PROVIDERS['qwen_vl']

    MODEL_CATALOG = QWEN_VL_MODEL_CATALOG
    DEFAULT_MODEL = QWEN_VL_DEFAULT_MODEL
    FAMILY_NAME = 'Qwen'
