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
Processing components and the pipeline engine of ``rapidtools``.

This package orchestrates the analysis of regional infrastructure assets. It
provides the core :class:`Pipeline` engine and the modular, swappable steps
that operate on a :class:`~rapidtools.core.PhysicalAssetCollection`:

    - Image extractors (``AerialImageryExtractor``, ``MapillaryImageExtractor``,
      ``BingOrthomosaicExtractor``) gather imagery for assets.
    - Feature extractors (``SAM3OrthoFeatureExtractor``) discover new assets
      in orthomosaic rasters.
    - Segmenters (``SAM3ImageSegmenter``) and analyzers (``*AssetAnalyzer``)
      run AI models over the gathered imagery.
    - Post-processing tools (``BuildingRegularizer``, ``RoadwayRegularizer``)
      turn raw model outputs into clean vector geometries.

Example:
    >>> from rapidtools.core import PhysicalAssetCollection
    >>> from rapidtools.processing import (
    ...     AerialImageryExtractor, Gemma4AssetAnalyzer, Pipeline
    ... )
    >>> collection = PhysicalAssetCollection.from_geojson('buildings.geojson')
    >>> pipeline = Pipeline([
    ...     AerialImageryExtractor('ortho.tif', save_directory='crops'),
    ...     Gemma4AssetAnalyzer(prompt='Describe the roof condition.'),
    ... ])
    >>> collection = pipeline.run(collection)
"""

# Import the core pipeline engine:
# Import feature extractors:
from .feature_extractors import SAM3OrthoFeatureExtractor

# Import the image analyzers:
from .image_analyzers import (
    AssetAnalyzer,
    BaseAPIAssetAnalyzer,
    BaseLocalAssetAnalyzer,
    ClaudeAssetAnalyzer,
    GeminiAssetAnalyzer,
    Gemma4AssetAnalyzer,
    HFVisionAssetAnalyzer,
    LlamaVisionAssetAnalyzer,
    MuseGlimmerAssetAnalyzer,
    MuseSparkAssetAnalyzer,
    OpenAIAssetAnalyzer,
    QwenAssetAnalyzer,
    QwenVisionAssetAnalyzer,
    RateLimitPolicy,
)

# Import the image extractors:
from .image_extractors import (
    AerialImageryExtractor,
    BingOrthomosaicExtractor,
    GoogleOrthomosaicExtractor,
    GoogleStreetViewImageExtractor,
    MapillaryImageExtractor,
)

# Import segmenters:
from .image_segmenters import SAM3ImageSegmenter

# Import label mappers:
from .label_mappers import MapillaryLabelMapper
from .pipeline import Pipeline

# Import postprocessing tools:
from .postprocessing.buildings import BuildingRegularizer
from .postprocessing.roads import RoadwayRegularizer
from .step import PipelineStep, Stage

# Explicitly define what is available when a user types:
__all__ = [
    'AerialImageryExtractor',
    'AssetAnalyzer',
    'BingOrthomosaicExtractor',
    'BaseAPIAssetAnalyzer',
    'BaseLocalAssetAnalyzer',
    'BuildingRegularizer',
    'ClaudeAssetAnalyzer',
    'GeminiAssetAnalyzer',
    'Gemma4AssetAnalyzer',
    'GoogleOrthomosaicExtractor',
    'GoogleStreetViewImageExtractor',
    'HFVisionAssetAnalyzer',
    'LlamaVisionAssetAnalyzer',
    'MapillaryLabelMapper',
    'MapillaryImageExtractor',
    'MuseGlimmerAssetAnalyzer',
    'MuseSparkAssetAnalyzer',
    'OpenAIAssetAnalyzer',
    'Pipeline',
    'PipelineStep',
    'RateLimitPolicy',
    'QwenAssetAnalyzer',
    'QwenVisionAssetAnalyzer',
    'RoadwayRegularizer',
    'SAM3ImageSegmenter',
    'SAM3OrthoFeatureExtractor',
    'Stage',
]
