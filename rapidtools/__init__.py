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
# 05-27-2026

"""Initializations and metadata for the rapidtools package."""

import logging
from pathlib import Path
import sys
from huggingface_hub import login, get_token

from .config import DATE_FORMAT, LOG_FORMAT

# Package metadata:
name = 'rapidtools'
__version__ = '0.1.0'
__copyright__ = 'Copyright (c) 2025, The University of Washington'
__license__ = 'BSD 3-Clause License'

# Logger formatting:
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    stream=sys.stdout,
    force=True,
)

# Import the core domain models:
from .core import (
    BoundingBox,
    ImageAsset,
    PhysicalAsset,
    PhysicalAssetCollection,
    PolygonRegion,
)

# Import the data sources and clients:
from .data_sources import (
    BingAerialImageExtractor,
    MapillaryClient,
    MapillaryLabels,
    OrthomosaicReader,
    TileUtils,
)

# Import AI models:
from .models import (
    GeminiInference,
    Gemma4Inference,
    SAM3Inference
)

# Import the dataset download utilities:
from .datasets import download_dataset

# Import the pipeline and processing tools:
from .processing import (
    AerialImageryExtractor,
    BingOrthomosaicExtractor,
    BuildingRegularizer,
    GeminiAssetAnalyzer,
    Gemma4AssetAnalyzer,
    MapillaryLabelMapper,
    MapillaryImageExtractor,
    Pipeline,
    RoadwayRegularizer,
    SAM3ImageSegmenter,
    SAM3OrthoFeatureExtractor,
)

# Explicitly define the top-level public API:
__all__ = [
    'AerialImageryExtractor',
    'BingAerialImageExtractor',
    'BingOrthomosaicExtractor',
    'BoundingBox',
    'BuildingRegularizer',
    'GeminiAssetAnalyzer',
    'GeminiInference',
    'Gemma4AssetAnalyzer',
    'Gemma4Inference',
    'ImageAsset',
    'MapillaryClient',
    'MapillaryLabelMapper',
    'MapillaryLabels',
    'MapillaryImageExtractor',
    'OrthomosaicReader',
    'PhysicalAsset',
    'PhysicalAssetCollection',
    'Pipeline',
    'PolygonRegion',
    'RoadwayRegularizer',
    'SAM3ImageSegmenter',
    'SAM3Inference',
    'SAM3OrthoFeatureExtractor',
    'TileUtils',
    'download_dataset',
]

# Trigger auto-authentication silently upon package initialization:
def _auto_authenticate_huggingface(dataset_id: str = 'hf_token') -> None:
    """
    Checks if a Hugging Face token is locally cached. If not, securely downloads 
    the token from the rapidtools registry, authenticates the session, and 
    immediately deletes the token file.
    """
    # If a token is already cached on this machine, exit instantly:
    if get_token() is not None:
        return

    try:
        # Download and read the token:
        [token_path] = download_dataset([dataset_id])
        token_file = Path(token_path)
        hf_token = token_file.read_text().strip()
        
        # Authenticate (this permanently caches the token on the local machine):
        login(token=hf_token, add_to_git_credential=False)
        
        # Securely destroy the downloaded file:
        token_file.unlink()
        logging.info('Hugging Face auto-authentication successful.')
        
    except Exception as e:
        logging.error(f'Failed to auto-authenticate with Hugging Face: {e}')

_auto_authenticate_huggingface('hf_token')