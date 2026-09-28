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
Model catalogues and provider metadata for every rapidtools model backend.

This module has no heavy dependencies (no PyTorch, no Transformers) so the
GUI and other front ends can list providers and model IDs instantly. The
wrapper classes import their defaults from here, and :data:`PROVIDERS` is the
single place to register a new backend.

Example:
    >>> from rapidtools.models.catalogs import PROVIDERS, catalog_ids
    >>> PROVIDERS['gemini'].default_model
    'gemini-3.8-flash'
    >>> catalog_ids(PROVIDERS['qwen'].catalog)[0]
    'qwen3.8-max'
"""

from __future__ import annotations

from dataclasses import dataclass, field


def catalog_ids(catalog: list[dict[str, str]]) -> list[str]:
    """
    Return the ``model_id`` of every entry in a model catalogue.

    Example:
        >>> catalog_ids([{'model_id': 'a'}, {'model_id': 'b'}])
        ['a', 'b']
    """
    return [entry['model_id'] for entry in catalog]


# --- gemini
GEMINI_DEFAULT_MODEL = 'gemini-3.8-flash'

GEMINI_DEFAULT_LITE_MODEL = 'gemini-3.5-flash-lite'

# Gemini models that accept image input, newest first.
GEMINI_MODEL_CATALOG: list[dict[str, str]] = [
    {'model_id': 'gemini-3.8-flash', 'family': 'Gemini 3.8', 'notes': 'latest stable'},
    {'model_id': 'gemini-3.7-flash', 'family': 'Gemini 3.7', 'notes': 'stable'},
    {'model_id': 'gemini-3.6-flash', 'family': 'Gemini 3.6', 'notes': 'stable'},
    {'model_id': 'gemini-3.5-flash', 'family': 'Gemini 3.5', 'notes': 'stable'},
    {
        'model_id': 'gemini-3.5-flash-lite',
        'family': 'Gemini 3.5',
        'notes': 'stable, lowest cost',
    },
    {'model_id': 'gemini-3.1-pro-preview', 'family': 'Gemini 3.1', 'notes': 'preview'},
    {'model_id': 'gemini-3.1-flash-lite', 'family': 'Gemini 3.1', 'notes': 'stable'},
    {
        'model_id': 'gemini-3.1-flash-image',
        'family': 'Gemini 3.1',
        'notes': 'image generation + understanding (Nano Banana 2)',
    },
    {
        'model_id': 'gemini-3.1-flash-lite-image',
        'family': 'Gemini 3.1',
        'notes': 'image generation + understanding (Nano Banana 2 Lite)',
    },
    {'model_id': 'gemini-3-flash-preview', 'family': 'Gemini 3', 'notes': 'preview'},
    {
        'model_id': 'gemini-3-pro-image',
        'family': 'Gemini 3',
        'notes': 'image generation + understanding (Nano Banana Pro)',
    },
    {'model_id': 'gemini-omni-1.1-flash', 'family': 'Gemini Omni', 'notes': 'preview'},
]


# --- claude
ANTHROPIC_DEFAULT_MODEL = 'claude-opus-5'

# Vision-capable Claude models, newest first.
ANTHROPIC_MODEL_CATALOG: list[dict[str, str]] = [
    {'model_id': 'claude-fable-5-1', 'family': 'Fable 5.1', 'notes': 'most capable'},
    {'model_id': 'claude-fable-5', 'family': 'Fable 5', 'notes': ''},
    {'model_id': 'claude-opus-5', 'family': 'Opus 5', 'notes': 'recommended default'},
    {'model_id': 'claude-opus-4-8', 'family': 'Opus 4.8', 'notes': ''},
    {'model_id': 'claude-opus-4-7', 'family': 'Opus 4.7', 'notes': ''},
    {'model_id': 'claude-opus-4-6', 'family': 'Opus 4.6', 'notes': ''},
    {'model_id': 'claude-sonnet-5', 'family': 'Sonnet 5', 'notes': 'fast, lower cost'},
    {'model_id': 'claude-sonnet-4-6', 'family': 'Sonnet 4.6', 'notes': ''},
    {'model_id': 'claude-haiku-4-5', 'family': 'Haiku 4.5', 'notes': 'smallest'},
]

# Current-generation models reject the `temperature` field (sampling is
# controlled server-side). Only Opus/Sonnet 4.6, Haiku 4.5 and older accept it.
ANTHROPIC_NO_TEMPERATURE_PREFIXES = (
    'claude-fable',
    'claude-mythos',
    'claude-opus-5',
    'claude-opus-4-7',
    'claude-opus-4-8',
    'claude-sonnet-5',
)


# --- openai
OPENAI_DEFAULT_MODEL = 'gpt-5.5'

# Vision-capable OpenAI models, newest generation first. GPT-5.x/GPT-6 and the
# o-series are reasoning models: they accept images but reject `temperature`.
OPENAI_MODEL_CATALOG: list[dict[str, str]] = [
    {'model_id': 'gpt-6-astra', 'family': 'GPT-6', 'notes': 'flagship reasoning'},
    {'model_id': 'gpt-6-sol', 'family': 'GPT-6', 'notes': 'balanced, 1M context'},
    {'model_id': 'gpt-6-luna', 'family': 'GPT-6', 'notes': 'cost-efficient'},
    {'model_id': 'gpt-5.6-sol', 'family': 'GPT-5.6', 'notes': 'flagship'},
    {'model_id': 'gpt-5.6-terra', 'family': 'GPT-5.6', 'notes': 'balanced'},
    {'model_id': 'gpt-5.6-luna', 'family': 'GPT-5.6', 'notes': 'cost-efficient'},
    {'model_id': 'gpt-5.5', 'family': 'GPT-5.5', 'notes': 'flagship, 1M context'},
    {'model_id': 'gpt-5.5-pro', 'family': 'GPT-5.5', 'notes': 'highest accuracy'},
    {'model_id': 'gpt-5.4', 'family': 'GPT-5.4', 'notes': ''},
    {'model_id': 'gpt-5.4-pro', 'family': 'GPT-5.4', 'notes': ''},
    {'model_id': 'gpt-5.4-mini', 'family': 'GPT-5.4', 'notes': 'small'},
    {'model_id': 'gpt-5.4-nano', 'family': 'GPT-5.4', 'notes': 'smallest'},
    {'model_id': 'gpt-5.2', 'family': 'GPT-5.2', 'notes': ''},
    {'model_id': 'gpt-5.1', 'family': 'GPT-5.1', 'notes': ''},
    {'model_id': 'o3', 'family': 'o-series', 'notes': 'reasoning'},
    {'model_id': 'o3-pro', 'family': 'o-series', 'notes': 'reasoning'},
    {'model_id': 'gpt-4.1', 'family': 'GPT-4.1', 'notes': 'supports temperature'},
    {'model_id': 'gpt-4.1-mini', 'family': 'GPT-4.1', 'notes': 'supports temperature'},
    {'model_id': 'gpt-4o', 'family': 'GPT-4o', 'notes': 'supports temperature'},
    {'model_id': 'gpt-4o-mini', 'family': 'GPT-4o', 'notes': 'supports temperature'},
]


# --- muse
MUSE_SPARK_DEFAULT_MODEL = 'muse-spark-1.3'

# Hosted Muse Spark models (text/image/video/PDF input, 1M-token context).
MUSE_SPARK_MODEL_CATALOG: list[dict[str, str]] = [
    {'model_id': 'muse-spark-1.3', 'family': 'Muse Spark', 'notes': 'latest'},
    {
        'model_id': 'muse-spark-1.3-contributor',
        'family': 'Muse Spark',
        'notes': 'discounted; prompts may be used for training',
    },
    {'model_id': 'muse-spark-1.2', 'family': 'Muse Spark', 'notes': ''},
    {
        'model_id': 'muse-spark-1.2-contributor',
        'family': 'Muse Spark',
        'notes': 'discounted; prompts may be used for training',
    },
    {'model_id': 'muse-spark-1.1', 'family': 'Muse Spark', 'notes': 'original'},
]

MUSE_GLIMMER_DEFAULT_MODEL = 'meta-models/Muse-Glimmer-30B'

# Open-weight Muse Glimmer checkpoints loadable with Transformers.
MUSE_GLIMMER_MODEL_CATALOG: list[dict[str, str]] = [
    {
        'model_id': 'meta-models/Muse-Glimmer-30B',
        'family': 'Muse Glimmer',
        'size': '30B',
        'notes': 'dense, Apache-2.0, BF16 weights; use load_in_4bit on 24 GB GPUs',
    },
]


# --- qwen
QWEN_DEFAULT_MODEL = 'qwen3.8-max'

# Hosted Qwen models that accept image input, newest first.
QWEN_MODEL_CATALOG: list[dict[str, str]] = [
    {'model_id': 'qwen3.8-max', 'family': 'Qwen3.8', 'notes': 'flagship, latest'},
    {'model_id': 'qwen3.8-omni-flash', 'family': 'Qwen3.8', 'notes': 'omni, fast'},
    {'model_id': 'qwen3.7-plus', 'family': 'Qwen3.7', 'notes': ''},
    {'model_id': 'qwen3.6-plus', 'family': 'Qwen3.6', 'notes': ''},
    {'model_id': 'qwen3.5-plus', 'family': 'Qwen3.5', 'notes': ''},
    {'model_id': 'qwen3-vl-plus', 'family': 'Qwen3-VL', 'notes': ''},
    {'model_id': 'qwen3-vl-flash', 'family': 'Qwen3-VL', 'notes': 'fast'},
    {'model_id': 'qwen-vl-max', 'family': 'Qwen-VL', 'notes': 'legacy'},
    {'model_id': 'qwen-vl-plus', 'family': 'Qwen-VL', 'notes': 'legacy'},
]

QWEN_VL_DEFAULT_MODEL = 'Qwen/Qwen3.5-4B'

# Open-weight Qwen checkpoints with native image understanding, newest first.
# Qwen3.5+ models are natively multimodal (model_type ``qwen3_5``).
QWEN_VL_MODEL_CATALOG: list[dict[str, str]] = [
    {'model_id': 'Qwen/Qwen3.8-27B', 'family': 'Qwen3.8', 'size': '27B'},
    {'model_id': 'Qwen/Qwen3.8-27B-FP8', 'family': 'Qwen3.8', 'size': '27B FP8'},
    {'model_id': 'Qwen/Qwen3.6-35B-A3B', 'family': 'Qwen3.6', 'size': '35B MoE'},
    {'model_id': 'Qwen/Qwen3.6-27B', 'family': 'Qwen3.6', 'size': '27B'},
    {'model_id': 'Qwen/Qwen3.5-397B-A17B', 'family': 'Qwen3.5', 'size': '397B MoE'},
    {'model_id': 'Qwen/Qwen3.5-122B-A10B', 'family': 'Qwen3.5', 'size': '122B MoE'},
    {'model_id': 'Qwen/Qwen3.5-35B-A3B', 'family': 'Qwen3.5', 'size': '35B MoE'},
    {'model_id': 'Qwen/Qwen3.5-27B', 'family': 'Qwen3.5', 'size': '27B'},
    {'model_id': 'Qwen/Qwen3.5-9B', 'family': 'Qwen3.5', 'size': '9B'},
    {'model_id': 'Qwen/Qwen3.5-4B', 'family': 'Qwen3.5', 'size': '4B'},
    {'model_id': 'Qwen/Qwen3.5-2B', 'family': 'Qwen3.5', 'size': '2B'},
    {'model_id': 'Qwen/Qwen3.5-0.8B', 'family': 'Qwen3.5', 'size': '0.8B'},
    {
        'model_id': 'Qwen/Qwen3-VL-235B-A22B-Instruct',
        'family': 'Qwen3-VL',
        'size': '235B MoE',
    },
    {'model_id': 'Qwen/Qwen3-VL-32B-Instruct', 'family': 'Qwen3-VL', 'size': '32B'},
    {
        'model_id': 'Qwen/Qwen3-VL-30B-A3B-Instruct',
        'family': 'Qwen3-VL',
        'size': '30B MoE',
    },
    {'model_id': 'Qwen/Qwen3-VL-8B-Instruct', 'family': 'Qwen3-VL', 'size': '8B'},
    {'model_id': 'Qwen/Qwen3-VL-4B-Instruct', 'family': 'Qwen3-VL', 'size': '4B'},
    {'model_id': 'Qwen/Qwen3-VL-2B-Instruct', 'family': 'Qwen3-VL', 'size': '2B'},
    {'model_id': 'Qwen/Qwen2.5-VL-72B-Instruct', 'family': 'Qwen2.5-VL', 'size': '72B'},
    {'model_id': 'Qwen/Qwen2.5-VL-32B-Instruct', 'family': 'Qwen2.5-VL', 'size': '32B'},
    {'model_id': 'Qwen/Qwen2.5-VL-7B-Instruct', 'family': 'Qwen2.5-VL', 'size': '7B'},
    {'model_id': 'Qwen/Qwen2.5-VL-3B-Instruct', 'family': 'Qwen2.5-VL', 'size': '3B'},
    {'model_id': 'Qwen/Qwen2-VL-7B-Instruct', 'family': 'Qwen2-VL', 'size': '7B'},
    {'model_id': 'Qwen/Qwen2-VL-2B-Instruct', 'family': 'Qwen2-VL', 'size': '2B'},
]


# --- gemma4
GEMMA4_DEFAULT_MODEL = 'google/gemma-4-E2B-it'

# Instruction-tuned Gemma 4 checkpoints, largest first. The 12B model uses
# the newer ``gemma4_unified`` architecture and needs transformers >= 5.10.
GEMMA4_MODEL_CATALOG: list[dict[str, str]] = [
    {'model_id': 'google/gemma-4-31B-it', 'family': 'Gemma 4', 'size': '31B'},
    {
        'model_id': 'google/gemma-4-26B-A4B-it',
        'family': 'Gemma 4',
        'size': '26B MoE (4B active)',
    },
    {
        'model_id': 'google/gemma-4-12B-it',
        'family': 'Gemma 4',
        'size': '12B',
        'min_transformers': '5.10.0',
    },
    {'model_id': 'google/gemma-4-E4B-it', 'family': 'Gemma 4', 'size': 'E4B'},
    {'model_id': 'google/gemma-4-E2B-it', 'family': 'Gemma 4', 'size': 'E2B'},
]


# --- llama
LLAMA_DEFAULT_MODEL = 'meta-llama/Llama-3.2-11B-Vision-Instruct'

# Meta Llama checkpoints that accept image input, newest first.
LLAMA_MODEL_CATALOG: list[dict[str, str]] = [
    {
        'model_id': 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
        'family': 'Llama 4',
        'size': '17B active / 109B total',
        'notes': 'MoE, 16 experts; needs ~55 GB in 4-bit',
    },
    {
        'model_id': 'meta-llama/Llama-4-Maverick-17B-128E-Instruct',
        'family': 'Llama 4',
        'size': '17B active / 400B total',
        'notes': 'MoE, 128 experts; multi-GPU',
    },
    {
        'model_id': 'meta-llama/Llama-3.2-11B-Vision-Instruct',
        'family': 'Llama 3.2 Vision',
        'size': '11B',
        'notes': 'fits 24 GB GPUs in 4-bit',
    },
    {
        'model_id': 'meta-llama/Llama-3.2-90B-Vision-Instruct',
        'family': 'Llama 3.2 Vision',
        'size': '90B',
        'notes': 'multi-GPU or heavy quantization',
    },
    {
        'model_id': 'meta-llama/Llama-3.2-11B-Vision',
        'family': 'Llama 3.2 Vision',
        'size': '11B',
        'notes': 'base (not instruction tuned)',
    },
    {
        'model_id': 'meta-llama/Llama-3.2-90B-Vision',
        'family': 'Llama 3.2 Vision',
        'size': '90B',
        'notes': 'base (not instruction tuned)',
    },
]


# --- hf_vision
# Curated open vision-language models known to work through the generic
# Transformers multimodal interface, newest first.
# Keys: model_id, family, size, optional min_transformers, optional notes.
OPEN_VLM_CATALOG: list[dict[str, str]] = [
    {'model_id': 'Qwen/Qwen3.8-27B', 'family': 'Qwen3.8', 'size': '27B'},
    {'model_id': 'Qwen/Qwen3.6-27B', 'family': 'Qwen3.6', 'size': '27B'},
    {'model_id': 'Qwen/Qwen3.6-35B-A3B', 'family': 'Qwen3.6', 'size': '35B MoE'},
    {'model_id': 'Qwen/Qwen3.5-27B', 'family': 'Qwen3.5', 'size': '27B'},
    {'model_id': 'Qwen/Qwen3.5-9B', 'family': 'Qwen3.5', 'size': '9B'},
    {'model_id': 'Qwen/Qwen3.5-4B', 'family': 'Qwen3.5', 'size': '4B'},
    {'model_id': 'Qwen/Qwen3.5-2B', 'family': 'Qwen3.5', 'size': '2B'},
    {'model_id': 'Qwen/Qwen3-VL-8B-Instruct', 'family': 'Qwen3-VL', 'size': '8B'},
    {'model_id': 'Qwen/Qwen3-VL-4B-Instruct', 'family': 'Qwen3-VL', 'size': '4B'},
    {'model_id': 'Qwen/Qwen3-VL-2B-Instruct', 'family': 'Qwen3-VL', 'size': '2B'},
    {'model_id': 'Qwen/Qwen2.5-VL-7B-Instruct', 'family': 'Qwen2.5-VL', 'size': '7B'},
    {'model_id': 'Qwen/Qwen2.5-VL-3B-Instruct', 'family': 'Qwen2.5-VL', 'size': '3B'},
    {'model_id': 'Qwen/Qwen2-VL-2B-Instruct', 'family': 'Qwen2-VL', 'size': '2B'},
    {
        'model_id': 'meta-models/Muse-Glimmer-30B',
        'family': 'Muse Glimmer',
        'size': '30B',
        'min_transformers': '5.15.0',
    },
    {
        'model_id': 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
        'family': 'Llama 4',
        'size': '17B x 16 experts',
    },
    {
        'model_id': 'meta-llama/Llama-4-Maverick-17B-128E-Instruct',
        'family': 'Llama 4',
        'size': '17B x 128 experts',
    },
    {
        'model_id': 'meta-llama/Llama-3.2-11B-Vision-Instruct',
        'family': 'Llama 3.2 Vision',
        'size': '11B',
    },
    {'model_id': 'google/gemma-3-27b-it', 'family': 'Gemma 3', 'size': '27B'},
    {'model_id': 'google/gemma-3-12b-it', 'family': 'Gemma 3', 'size': '12B'},
    {'model_id': 'google/gemma-3-4b-it', 'family': 'Gemma 3', 'size': '4B'},
    {'model_id': 'google/gemma-3n-E4B-it', 'family': 'Gemma 3n', 'size': 'E4B'},
    {'model_id': 'google/gemma-3n-E2B-it', 'family': 'Gemma 3n', 'size': 'E2B'},
    {'model_id': 'OpenGVLab/InternVL3_5-8B-HF', 'family': 'InternVL3.5', 'size': '8B'},
    {'model_id': 'OpenGVLab/InternVL3-8B-hf', 'family': 'InternVL3', 'size': '8B'},
    {'model_id': 'OpenGVLab/InternVL3-2B-hf', 'family': 'InternVL3', 'size': '2B'},
    {
        'model_id': 'llava-hf/llava-onevision-qwen2-7b-ov-hf',
        'family': 'LLaVA-OneVision',
        'size': '7B',
    },
    {
        'model_id': 'llava-hf/llava-v1.6-mistral-7b-hf',
        'family': 'LLaVA-NeXT',
        'size': '7B',
    },
    {'model_id': 'mistral-community/pixtral-12b', 'family': 'Pixtral', 'size': '12B'},
    {
        'model_id': 'ibm-granite/granite-vision-3.3-2b',
        'family': 'Granite Vision',
        'size': '2B',
    },
]

DEFAULT_OPEN_VLM = 'Qwen/Qwen3.5-4B'


# --- sam3
SAM3_DEFAULT_MODEL = 'facebook/sam3'

SAM3_MODEL_CATALOG: list[dict[str, str]] = [
    {
        'model_id': 'facebook/sam3',
        'family': 'SAM 3',
        'size': '848M',
        'notes': 'promptable concept segmentation (text prompts)',
    },
]


@dataclass(frozen=True)
class ModelInfo:
    """
    Static description of a model backend.

    Attributes:
        key: Short registry key (``'gemini'``, ``'qwen_vl'``, ...).
        label: Human-readable provider name.
        kind: ``'api'`` (hosted, needs a key), ``'local'`` (Hugging Face
            weights) or ``'segmentation'`` (SAM 3).
        description: One-line description shown in user interfaces.
        module: Import path of the module holding the wrapper class.
        class_name: Name of the wrapper class inside ``module``.
        default_model: Model ID used when none is given to the wrapper.
        catalog: Curated ``{'model_id', 'family', ...}`` records.
        attribute_prefix: Prefix used for attributes written by analyzers.
        key_env: Environment variable(s) holding the API key (API kinds).
        key_label: Label for the key field in user interfaces.
        analyzer_default_model: Cheaper default used by batch analyzers and
            the GUI; falls back to ``default_model``.
        supports_4bit: Whether ``load_in_4bit`` is accepted.
        supports_batch: Whether the wrapper offers ``run_inference_batch``.
        free_form_model: Whether arbitrary model IDs beyond the catalogue are
            expected to work (generic Hugging Face backends).

    Example:
        >>> from rapidtools.models.catalogs import PROVIDERS
        >>> PROVIDERS['claude'].kind, PROVIDERS['gemma4'].supports_batch
        ('api', True)
    """

    key: str
    label: str
    kind: str
    description: str
    module: str
    class_name: str
    default_model: str
    catalog: list[dict[str, str]] = field(
        default_factory=list, hash=False, compare=False
    )
    attribute_prefix: str = ''
    key_env: tuple[str, ...] = ()
    key_label: str = ''
    analyzer_default_model: str | None = None
    supports_4bit: bool = False
    supports_batch: bool = False
    free_form_model: bool = False

    @property
    def model_ids(self) -> list[str]:
        """Model IDs of the catalogue, with the analyzer default first."""
        ids = catalog_ids(self.catalog)
        first = self.analyzer_default_model or self.default_model
        if first in ids:
            ids.remove(first)
        return [first] + ids

    @property
    def is_api(self) -> bool:
        """``True`` for hosted providers that need an API key."""
        return self.kind == 'api'


# Every backend rapidtools ships, in the order user interfaces present them.
PROVIDERS: dict[str, ModelInfo] = {
    'gemini': ModelInfo(
        key='gemini',
        label='Google Gemini',
        kind='api',
        description='Cloud API · fast, needs a Google API key',
        module='rapidtools.models.gemini',
        class_name='GeminiInference',
        default_model=GEMINI_DEFAULT_MODEL,
        analyzer_default_model=GEMINI_DEFAULT_LITE_MODEL,
        catalog=GEMINI_MODEL_CATALOG,
        attribute_prefix='gemini',
        key_env=('GOOGLE_API_KEY', 'GEMINI_API_KEY'),
        key_label='Gemini API key',
    ),
    'claude': ModelInfo(
        key='claude',
        label='Anthropic Claude',
        kind='api',
        description='Cloud API · needs an Anthropic API key',
        module='rapidtools.models.claude',
        class_name='ClaudeInference',
        default_model=ANTHROPIC_DEFAULT_MODEL,
        catalog=ANTHROPIC_MODEL_CATALOG,
        attribute_prefix='claude',
        key_env=('ANTHROPIC_API_KEY',),
        key_label='Anthropic API key',
    ),
    'openai': ModelInfo(
        key='openai',
        label='OpenAI',
        kind='api',
        description='Cloud API · needs an OpenAI API key',
        module='rapidtools.models.openai',
        class_name='OpenAIInference',
        default_model=OPENAI_DEFAULT_MODEL,
        catalog=OPENAI_MODEL_CATALOG,
        attribute_prefix='openai',
        key_env=('OPENAI_API_KEY',),
        key_label='OpenAI API key',
    ),
    'muse_spark': ModelInfo(
        key='muse_spark',
        label='Meta Muse Spark',
        kind='api',
        description='Cloud API · needs a Meta Model API key',
        module='rapidtools.models.muse',
        class_name='MuseSparkInference',
        default_model=MUSE_SPARK_DEFAULT_MODEL,
        catalog=MUSE_SPARK_MODEL_CATALOG,
        attribute_prefix='muse_spark',
        key_env=('MODEL_API_KEY', 'META_API_KEY'),
        key_label='Meta Model API key',
    ),
    'qwen': ModelInfo(
        key='qwen',
        label='Qwen (Alibaba Cloud)',
        kind='api',
        description='Cloud API · needs a Model Studio (DashScope) key',
        module='rapidtools.models.qwen',
        class_name='QwenInference',
        default_model=QWEN_DEFAULT_MODEL,
        catalog=QWEN_MODEL_CATALOG,
        attribute_prefix='qwen',
        key_env=('DASHSCOPE_API_KEY', 'QWEN_API_KEY'),
        key_label='DashScope API key',
    ),
    'gemma4': ModelInfo(
        key='gemma4',
        label='Gemma-4',
        kind='local',
        description='Runs locally on your GPU · no key needed',
        module='rapidtools.models.gemma4',
        class_name='Gemma4Inference',
        default_model=GEMMA4_DEFAULT_MODEL,
        catalog=GEMMA4_MODEL_CATALOG,
        attribute_prefix='gemma4',
        supports_batch=True,
    ),
    'llama': ModelInfo(
        key='llama',
        label='Llama Vision',
        kind='local',
        description=(
            'Llama 4 / Llama 3.2 Vision · gated Hugging Face download, '
            '4-bit recommended'
        ),
        module='rapidtools.models.llama',
        class_name='LlamaVisionInference',
        default_model=LLAMA_DEFAULT_MODEL,
        catalog=LLAMA_MODEL_CATALOG,
        attribute_prefix='llama',
        supports_4bit=True,
    ),
    'muse_glimmer': ModelInfo(
        key='muse_glimmer',
        label='Meta Muse Glimmer',
        kind='local',
        description='Open 30B model · 24 GB GPU in 4-bit, transformers ≥ 5.15',
        module='rapidtools.models.muse',
        class_name='MuseGlimmerInference',
        default_model=MUSE_GLIMMER_DEFAULT_MODEL,
        catalog=MUSE_GLIMMER_MODEL_CATALOG,
        attribute_prefix='muse_glimmer',
        supports_4bit=True,
        supports_batch=True,
    ),
    'qwen_vl': ModelInfo(
        key='qwen_vl',
        label='Qwen (local)',
        kind='local',
        description='Open Qwen3.5 / Qwen3.8 / Qwen3-VL checkpoints · no key needed',
        module='rapidtools.models.qwen',
        class_name='QwenVisionInference',
        default_model=QWEN_VL_DEFAULT_MODEL,
        catalog=QWEN_VL_MODEL_CATALOG,
        attribute_prefix='qwen_vl',
        supports_4bit=True,
        supports_batch=True,
        free_form_model=True,
    ),
    'hf': ModelInfo(
        key='hf',
        label='Other open models',
        kind='local',
        description=(
            'Any Hugging Face vision-language checkpoint: Gemma 3, LLaVA, '
            'InternVL, Pixtral, Granite…'
        ),
        module='rapidtools.models.hf_vision',
        class_name='HFVisionInference',
        default_model=DEFAULT_OPEN_VLM,
        catalog=OPEN_VLM_CATALOG,
        attribute_prefix='vlm',
        supports_4bit=True,
        supports_batch=True,
        free_form_model=True,
    ),
    'sam3': ModelInfo(
        key='sam3',
        label='Meta SAM 3',
        kind='segmentation',
        description='Text-prompted instance segmentation · runs locally',
        module='rapidtools.models.sam3',
        class_name='SAM3Inference',
        default_model=SAM3_DEFAULT_MODEL,
        catalog=SAM3_MODEL_CATALOG,
        attribute_prefix='sam3',
        supports_4bit=True,
    ),
}
