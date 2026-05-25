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
# 05-24-2026

"""
This script demonstrates how to:
1. Download a sample drone orthomosaic (eaton_patch1).
2. Extract its exact geographic bounding polygon.
3. Use that polygon to download the corresponding Bing Maps tiles.
4. Extract preliminary building footprints from the synthetic Bing orthomosaic 
   using SAM 3.
5. Crop high-resolution aerial imagery around each preliminary footprint.
6. Regularize the footprints using AI to resolve overlaps, bridge gaps, and 
   create production-ready boundaries.
"""

from rapidtools import (
    AerialImageryExtractor,
    BingOrthomosaicExtractor,
    BoundingBox,
    BuildingRegularizer,
    SAM3OrthoFeatureExtractor,
    download_dataset,
)

prompt_target = 'building roof'
output_tiff = 'eaton_patch1_bing.tiff'
output_geojson1 = 'eaton_patch1_buildings_preliminary.geojson'
output_geojson2 = 'eaton_patch1_buildings_final.geojson'

# Download the reference dataset:
[reference_raster_path] = download_dataset('eaton_patch1')

# Extract the bounding polygon directly from the reference TIFF:
region_of_interest = BoundingBox.from_raster(reference_raster_path)

# Download and stitch the Bing imagery for the area covered by the reference dataset:
print('Connecting to Bing Maps to synthesize orthomosaic for the raster area...')
bing_extractor = BingOrthomosaicExtractor(zoom_level=19, max_workers=10)
bing_tiff_path = bing_extractor(
    region=region_of_interest, 
    output_path=output_tiff
)

# Extract the buildings from the TIFF image based on Bing tiles and save the
# extracted preliminary building masks in a GeoJSON file:
sam_extractor = SAM3OrthoFeatureExtractor(
    prompt=prompt_target,
    patch_size=50,          
    unit='meters',           
    overlap_ratio=0.25,      
    batch_size=4,            
    threshold=0.50,
    mask_threshold=0.40,
)

print('Running preliminary SAM 3 extraction...')
prelim_collection = sam_extractor(bing_tiff_path)

_ = prelim_collection.to_geojson(output_geojson1)
print(
    f'{len(prelim_collection)} buildings extracted and exported to: '
    f'{output_geojson1}'
)

# To refine the computed preliminary masks, obtain the image patches for each 
# mask buffered 20 meters to give better context to the segmentation model:
extractor = AerialImageryExtractor(
    dataset=bing_tiff_path,
    save_directory='output/building_crops',
    buffer_asset='20 m',
    force_square_image=True,
)

print('Extracting buffered image crops for regularization...')
assets_with_images = extractor(prelim_collection)

# Set up the regularizer:
regularizer = BuildingRegularizer(
    batch_size=8
)

# Regularize building footprints by running a more context-aware segmentation,  
# dissolving overlapping and adjoining polygons, performing straight-edge cuts 
# between neighboring polygons, and returning the final, refined geometries to a 
# GeoJSON file:
print('Running building regularizer...')
final_buildings = regularizer(assets_with_images)
_ = final_buildings.to_geojson(output_geojson2)

print(f'Regularization complete. Final assets exported to: {output_geojson2}')