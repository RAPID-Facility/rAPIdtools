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
AI model wrappers for rapidtools.

Cloud APIs (:class:`GeminiInference`, :class:`ClaudeInference`,
:class:`OpenAIInference`, :class:`MuseSparkInference`, :class:`QwenInference`)
and local Hugging Face checkpoints (:class:`Gemma4Inference`,
:class:`LlamaVisionInference`, :class:`MuseGlimmerInference`,
:class:`QwenVisionInference`, :class:`HFVisionInference`,
:class:`SAM3Inference`) all return :class:`ModelOutput` from
``run_inference`` so pipeline components can swap providers freely.

The wrapper classes are imported lazily (they pull in PyTorch and
Transformers); the provider registry :data:`PROVIDERS` and the model
catalogues are always available. Use :func:`load` to build a model from its
registry key.

Example:
    >>> from rapidtools.models import load, list_providers
    >>> list_providers(kind='api')
    ['gemini', 'claude', 'openai', 'muse_spark', 'qwen']
    >>> model = load('gemini', api_key='AIza...', model_id='gemini-3.8-flash')
    >>> model.INFO.label
    'Google Gemini'
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

from .base import (
    BaseInferenceModel,
    GenerationConfig,
    ModelOutput,
    SegmentationConfig,
)
from .catalogs import (
    ANTHROPIC_MODEL_CATALOG,
    DEFAULT_OPEN_VLM,
    GEMINI_MODEL_CATALOG,
    GEMMA4_MODEL_CATALOG,
    LLAMA_MODEL_CATALOG,
    MUSE_GLIMMER_MODEL_CATALOG,
    MUSE_SPARK_MODEL_CATALOG,
    OPEN_VLM_CATALOG,
    OPENAI_MODEL_CATALOG,
    PROVIDERS,
    QWEN_MODEL_CATALOG,
    QWEN_VL_MODEL_CATALOG,
    SAM3_MODEL_CATALOG,
    ModelInfo,
    catalog_ids,
)

# Wrapper class name -> registry key, for lazy attribute access.
_CLASS_TO_PROVIDER: dict[str, str] = {
    info.class_name: key for key, info in PROVIDERS.items()
}


def get_model_class(provider: str) -> type[BaseInferenceModel]:
    """
    Import and return the wrapper class registered under ``provider``.

    Args:
        provider: Registry key such as ``'gemini'`` or ``'qwen_vl'``.

    Returns:
        type[BaseInferenceModel]: The wrapper class.

    Raises:
        KeyError: If ``provider`` is not registered.

    Example:
        >>> from rapidtools.models import get_model_class
        >>> get_model_class('claude').__name__
        'ClaudeInference'
    """
    try:
        info = PROVIDERS[provider]
    except KeyError:
        raise KeyError(
            f'Unknown model provider {provider!r}; choose one of {list_providers()}.'
        ) from None
    module = importlib.import_module(info.module)
    return getattr(module, info.class_name)


def load(provider: str, **kwargs: Any) -> BaseInferenceModel:
    """
    Instantiate a model wrapper from its registry key.

    Keyword arguments are passed to the wrapper constructor, e.g. ``api_key``
    and ``model_id`` for hosted providers, ``model_id``, ``device`` and
    ``load_in_4bit`` for local ones.

    Args:
        provider: Registry key (see :func:`list_providers`).
        **kwargs: Constructor arguments of the wrapper.

    Returns:
        BaseInferenceModel: The ready-to-use model.

    Example:
        >>> from rapidtools.models import load
        >>> gemini = load('gemini', api_key='AIza...')
        >>> qwen = load('qwen_vl', model_id='Qwen/Qwen3.5-9B', load_in_4bit=True)
        >>> qwen.run_inference('roof.jpg', 'Describe the roof.').text
    """
    return get_model_class(provider)(**kwargs)


def catalog(provider: str) -> list[dict[str, str]]:
    """
    Return the curated model catalogue of a provider (no network).

    Example:
        >>> from rapidtools.models import catalog
        >>> catalog('openai')[0]['model_id']
        'gpt-6-astra'
    """
    return list(PROVIDERS[provider].catalog)


def list_providers(kind: str | None = None) -> list[str]:
    """
    Return registry keys, optionally filtered by kind.

    Args:
        kind: ``'api'``, ``'local'`` or ``'segmentation'``; ``None`` for all.

    Returns:
        list[str]: Provider keys in presentation order.

    Example:
        >>> from rapidtools.models import list_providers
        >>> list_providers('segmentation')
        ['sam3']
    """
    return [key for key, info in PROVIDERS.items() if kind is None or info.kind == kind]


class _LazyRegistry(Mapping[str, type[BaseInferenceModel]]):
    """Read-only ``provider -> class`` mapping that imports classes on demand."""

    def __getitem__(self, key: str) -> type[BaseInferenceModel]:
        """Import and return the wrapper class for ``key``."""
        return get_model_class(key)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the provider keys."""
        return iter(PROVIDERS)

    def __len__(self) -> int:
        """Number of registered providers."""
        return len(PROVIDERS)

    def __repr__(self) -> str:
        """Show the provider keys without importing any class."""
        return f'MODEL_REGISTRY({list(PROVIDERS)})'


#: Short provider key -> wrapper class (imported lazily).
MODEL_REGISTRY: Mapping[str, type[BaseInferenceModel]] = _LazyRegistry()

__all__ = [
    'ANTHROPIC_MODEL_CATALOG',
    'BaseInferenceModel',
    'ClaudeInference',
    'DEFAULT_OPEN_VLM',
    'GEMINI_MODEL_CATALOG',
    'GEMMA4_MODEL_CATALOG',
    'GeminiInference',
    'Gemma4Inference',
    'GenerationConfig',
    'HFVisionInference',
    'LLAMA_MODEL_CATALOG',
    'LlamaVisionInference',
    'MODEL_REGISTRY',
    'MUSE_GLIMMER_MODEL_CATALOG',
    'MUSE_SPARK_MODEL_CATALOG',
    'ModelInfo',
    'ModelOutput',
    'MuseGlimmerInference',
    'MuseSparkInference',
    'OPENAI_MODEL_CATALOG',
    'OPEN_VLM_CATALOG',
    'OpenAIInference',
    'PROVIDERS',
    'QWEN_MODEL_CATALOG',
    'QWEN_VL_MODEL_CATALOG',
    'QwenInference',
    'QwenVisionInference',
    'SAM3_MODEL_CATALOG',
    'SAM3Inference',
    'SegmentationConfig',
    'catalog',
    'catalog_ids',
    'get_model_class',
    'list_providers',
    'load',
]


def __getattr__(attr: str) -> Any:
    """Import wrapper classes on first access."""
    provider = _CLASS_TO_PROVIDER.get(attr)
    if provider is None:
        raise AttributeError(f'module {__name__!r} has no attribute {attr!r}')
    value = get_model_class(provider)
    globals()[attr] = value
    return value


def __dir__() -> list[str]:
    """Include lazily loaded classes in ``dir()``."""
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from .claude import ClaudeInference  # noqa: F401
    from .gemini import GeminiInference  # noqa: F401
    from .gemma4 import Gemma4Inference  # noqa: F401
    from .hf_vision import HFVisionInference  # noqa: F401
    from .llama import LlamaVisionInference  # noqa: F401
    from .muse import MuseGlimmerInference, MuseSparkInference  # noqa: F401
    from .openai import OpenAIInference  # noqa: F401
    from .qwen import QwenInference, QwenVisionInference  # noqa: F401
    from .sam3 import SAM3Inference  # noqa: F401
