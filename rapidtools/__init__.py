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
# 09-28-2026

"""
Initializations and metadata for the rapidtools package.

Importing ``rapidtools`` is side-effect free: it exposes the geospatial core,
the data-source clients and the dataset downloader eagerly, while the model
wrappers and processing components (which pull in PyTorch and Transformers)
are imported lazily the first time they are accessed. Logging is left to the
application; call :func:`configure_logging` to see progress messages, and
:func:`login` to authenticate with the Hugging Face Hub.

Example:
    >>> import rapidtools
    >>> rapidtools.__version__
    '0.2.0'
    >>> rapidtools.configure_logging()  # doctest: +ELLIPSIS
    <Logger rapidtools (INFO)>
    >>> rapidtools.GeminiInference  # imported on first access
    <class 'rapidtools.models.gemini.GeminiInference'>
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any

from .auth import ensure_huggingface_login, login
from .config import configure_logging

# Import the core domain models:
from .core import (
    BoundingBox,
    ImageAsset,
    Observation,
    OperationCancelled,
    PhysicalAsset,
    PhysicalAssetCollection,
    PolygonRegion,
)

# Import the data sources and clients:
from .data_sources import (
    BingAerialImageExtractor,
    GoogleAerialImageExtractor,
    GoogleStreetViewClient,
    MapillaryClient,
    MapillaryLabels,
    OrthomosaicReader,
    TileUtils,
)

# Import the dataset download utilities:
from .datasets import download_dataset

# Package metadata:
name = 'rapidtools'
__version__ = '0.2.0'
__copyright__ = 'Copyright (c) 2025, The University of Washington'
__license__ = 'BSD 3-Clause License'

# The library never configures logging itself; applications opt in with
# configure_logging():
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Names resolved lazily from the heavy subpackages.
# Subpackages resolved on first attribute access:
_LAZY_SUBMODULES = frozenset({'models', 'processing', 'gui'})

_LAZY_ATTRIBUTES: dict[str, str] = {
    # rapidtools.models
    'ClaudeInference': 'rapidtools.models',
    'GeminiInference': 'rapidtools.models',
    'Gemma4Inference': 'rapidtools.models',
    'HFVisionInference': 'rapidtools.models',
    'LlamaVisionInference': 'rapidtools.models',
    'MuseGlimmerInference': 'rapidtools.models',
    'MuseSparkInference': 'rapidtools.models',
    'OpenAIInference': 'rapidtools.models',
    'QwenInference': 'rapidtools.models',
    'QwenVisionInference': 'rapidtools.models',
    'SAM3Inference': 'rapidtools.models',
    # rapidtools.processing
    'AerialImageryExtractor': 'rapidtools.processing',
    'AssetAnalyzer': 'rapidtools.processing',
    'RateLimitPolicy': 'rapidtools.processing',
    'Stage': 'rapidtools.processing',
    'BingOrthomosaicExtractor': 'rapidtools.processing',
    'BuildingRegularizer': 'rapidtools.processing',
    'ClaudeAssetAnalyzer': 'rapidtools.processing',
    'GeminiAssetAnalyzer': 'rapidtools.processing',
    'Gemma4AssetAnalyzer': 'rapidtools.processing',
    'GoogleOrthomosaicExtractor': 'rapidtools.processing',
    'GoogleStreetViewImageExtractor': 'rapidtools.processing',
    'HFVisionAssetAnalyzer': 'rapidtools.processing',
    'LlamaVisionAssetAnalyzer': 'rapidtools.processing',
    'MapillaryFeatureExtractor': 'rapidtools.processing',
    'MapillaryImageExtractor': 'rapidtools.processing',
    'MapillaryObjectImageExtractor': 'rapidtools.processing',
    'MapillaryLabelMapper': 'rapidtools.processing',
    'MuseGlimmerAssetAnalyzer': 'rapidtools.processing',
    'MuseSparkAssetAnalyzer': 'rapidtools.processing',
    'OpenAIAssetAnalyzer': 'rapidtools.processing',
    'Pipeline': 'rapidtools.processing',
    'QwenAssetAnalyzer': 'rapidtools.processing',
    'QwenVisionAssetAnalyzer': 'rapidtools.processing',
    'RoadwayRegularizer': 'rapidtools.processing',
    'SAM3ImageSegmenter': 'rapidtools.processing',
    'SAM3OrthoFeatureExtractor': 'rapidtools.processing',
}

# Explicitly define the top-level public API:
__all__ = [
    'AerialImageryExtractor',
    'AssetAnalyzer',
    'BingAerialImageExtractor',
    'BingOrthomosaicExtractor',
    'BoundingBox',
    'BuildingRegularizer',
    'ClaudeAssetAnalyzer',
    'ClaudeInference',
    'GeminiAssetAnalyzer',
    'GeminiInference',
    'Gemma4AssetAnalyzer',
    'Gemma4Inference',
    'GoogleAerialImageExtractor',
    'GoogleOrthomosaicExtractor',
    'GoogleStreetViewClient',
    'GoogleStreetViewImageExtractor',
    'HFVisionAssetAnalyzer',
    'HFVisionInference',
    'ImageAsset',
    'LlamaVisionAssetAnalyzer',
    'LlamaVisionInference',
    'MapillaryClient',
    'MapillaryFeatureExtractor',
    'MapillaryImageExtractor',
    'MapillaryLabelMapper',
    'MapillaryLabels',
    'MapillaryObjectImageExtractor',
    'MuseGlimmerAssetAnalyzer',
    'MuseGlimmerInference',
    'MuseSparkAssetAnalyzer',
    'MuseSparkInference',
    'Observation',
    'OpenAIAssetAnalyzer',
    'OpenAIInference',
    'OperationCancelled',
    'OrthomosaicReader',
    'PhysicalAsset',
    'PhysicalAssetCollection',
    'Pipeline',
    'PolygonRegion',
    'QwenAssetAnalyzer',
    'QwenInference',
    'QwenVisionAssetAnalyzer',
    'QwenVisionInference',
    'RateLimitPolicy',
    'RoadwayRegularizer',
    'SAM3ImageSegmenter',
    'SAM3Inference',
    'SAM3OrthoFeatureExtractor',
    'Stage',
    'TileUtils',
    'configure_logging',
    'download_dataset',
    'ensure_huggingface_login',
    'login',
]


def __getattr__(attr: str) -> Any:
    """
    Import model and processing classes on first access.

    This keeps ``import rapidtools`` fast for geometry-only work: PyTorch and
    Transformers are loaded only when a model or pipeline component is used.
    """
    module_name = _LAZY_ATTRIBUTES.get(attr)
    if module_name is None:
        if attr in _LAZY_SUBMODULES:
            # ``rapidtools.models`` / ``rapidtools.processing`` / ``rapidtools.gui``
            # are heavy, so they are only imported when first referenced:
            module = importlib.import_module(f'{__name__}.{attr}')
            globals()[attr] = module
            return module
        raise AttributeError(f'module {__name__!r} has no attribute {attr!r}')
    module = importlib.import_module(module_name)
    value = getattr(module, attr)
    globals()[attr] = value  # cache for subsequent lookups
    return value


def __dir__() -> list[str]:
    """Include lazily loaded names in ``dir(rapidtools)``."""
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from .models import (  # noqa: F401
        ClaudeInference,
        GeminiInference,
        Gemma4Inference,
        HFVisionInference,
        LlamaVisionInference,
        MuseGlimmerInference,
        MuseSparkInference,
        OpenAIInference,
        QwenInference,
        QwenVisionInference,
        SAM3Inference,
    )
    from .processing import (  # noqa: F401
        AerialImageryExtractor,
        AssetAnalyzer,
        BingOrthomosaicExtractor,
        BuildingRegularizer,
        ClaudeAssetAnalyzer,
        GeminiAssetAnalyzer,
        Gemma4AssetAnalyzer,
        GoogleOrthomosaicExtractor,
        GoogleStreetViewImageExtractor,
        HFVisionAssetAnalyzer,
        LlamaVisionAssetAnalyzer,
        MapillaryImageExtractor,
        MapillaryLabelMapper,
        MuseGlimmerAssetAnalyzer,
        MuseSparkAssetAnalyzer,
        OpenAIAssetAnalyzer,
        Pipeline,
        QwenAssetAnalyzer,
        QwenVisionAssetAnalyzer,
        RateLimitPolicy,
        RoadwayRegularizer,
        SAM3ImageSegmenter,
        SAM3OrthoFeatureExtractor,
        Stage,
    )
