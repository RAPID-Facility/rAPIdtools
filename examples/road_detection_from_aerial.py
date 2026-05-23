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
# Author:
# Barbaros Cetiner
#
# Last updated:
# 05-23-2026

"""
This script provides an example of the rapidtools road detection pipeline by:
1. Downloading the required example orthomosaic dataset.
2. Extracting raw road masks from the imagery using the SAM3 vision model.
3. Exporting the raw road polygon extractions to a GeoJSON file for debugging.
4. Regularizing the raw polygons to generate clean, connected road centerlines.
5. Exporting the final road centerlines to a GeoJSON file.
"""

from rapidtools import download_dataset, SAM3OrthoFeatureExtractor, RoadwayRegularizer

# Download the orthomosaic patch that will be utilized for this example:
[patch_path] = download_dataset('eaton_patch1')

# Extract raw road masks from the RAPID ortho image:
extractor = SAM3OrthoFeatureExtractor(
    prompt='paved road, street',  # Text prompt guiding the extraction model
    patch_size=100,               # Size of the geographic tiles to process
    unit='meters',                # Physical unit for the patch size (100x100 meters)
    overlap_ratio=0.25,           # 25% overlap between tiles to prevent edge artifacts
    batch_size=4,                 # Number of patches to process concurrently on the GPU
    threshold=0.25,               # Confidence score required to keep a detection
    mask_threshold=0.25,          # Binarization threshold for generating the final polygon
)

raw_road_assets = extractor(patch_path)
print(f'Successfully extracted {len(raw_road_assets)} raw polygons.')

# Optional: Save raw extraction for debugging/records:
raw_road_assets.to_geojson('roads_raw.geojson')

# Regularize the road polygons:
regularizer = RoadwayRegularizer()

# The regularizer returns a tuple of (centerlines, polygons). 
# We only need centerlines for this example.
centerlines, _ = regularizer(raw_road_assets)

# Save the regularized centerlines:
centerlines.to_geojson('roads_centerlines.geojson')
print('Pipeline execution complete. Centerlines saved.')