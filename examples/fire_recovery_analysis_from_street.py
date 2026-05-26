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
# 05-26-2026

"""
This script provides an example of the rapidtools data processing pipeline by:
1. Loading a spatial inventory of building footprints.
2. Extracting multi-temporal street-level imagery from Mapillary (Feb & Sep).
3. Using a local Gemma-4 vision-language model to assess initial fire damage.
4. Filtering the dataset to isolate damaged buildings.
5. Re-evaluating the damaged buildings to assess recovery progress.
6. Exporting the enriched dataset to a new GeoJSON file.
"""

from pathlib import Path

from rapidtools import (
    download_dataset,
    Gemma4AssetAnalyzer,
    MapillaryImageExtractor,
    PhysicalAssetCollection,
)

# Download all required example datasets from the registry in a single call.
# Because each requested dataset contains exactly one file, we can unpack 
# the returned list directly into individual variables:
footprint_path, damage_prompts, recovery_prompts, token_path = download_dataset([
    'eaton_patch1_bing_buildings',
    'street_chs_prompts',
    'street_recovery_prompts',
    'mapillary_token'
])

# Load building footprint data:
building_data = PhysicalAssetCollection.from_geojson(footprint_path)

# Read the Mapillary token securely from a local text file:
MAPILLARY_TOKEN = token_path.read_text().strip()

# Read the prompt instructions for the AI models:
damage_prompt = Path(damage_prompts).read_text().strip()
recovery_prompt = Path(recovery_prompts).read_text().strip()


# Step 1: Multi-temporal image extraction
print('\n--- Step 1: Extracting Street-Level Imagery ---')

# Fetch post-disaster images from February:
feb_extractor = MapillaryImageExtractor(
    access_token=MAPILLARY_TOKEN,
    save_directory='output/street_feb',
    start_date='2025-01-01',
    end_date='2025-05-01',
    cast_corner_rays=False,
)
building_data = feb_extractor(building_data)

# Fetch post-disaster images from September:
sep_extractor = MapillaryImageExtractor(
    access_token=MAPILLARY_TOKEN,
    save_directory='output/street_sep',
    start_date='2025-06-01',
    end_date='2025-12-01',
    cast_corner_rays=False,
)

# Run extraction on the same collection so the assets hold both sets of images:
building_data = sep_extractor(building_data)


# Step 2: Initial damage assessment using the images captured in February:
print('\n--- Step 2: Post-Disaster Damage Assessment ---')

# Initialize the analyzer once to load the weights into the GPU. Apply a lambda
# filter so Gemma only sees the February images for this step:
analyzer = Gemma4AssetAnalyzer(
    model_id='google/gemma-4-E2B-it',
    device='cuda',
    prompt=damage_prompt,
    batch_size=4,
    max_images_per_asset=2,
    image_filter=lambda img: 'street_feb' in str(img.path)
)

print('Initiating damage assessment...')
assessed_collection = analyzer(building_data)

# Step 3: Filtering damaged assets:
print('\n--- Step 3: Filtering Damaged Assets ---')

# Filter the collection to only keep assets that Gemma marked as damaged (CHS 1-4):
damaged_assets_list = [
    asset for asset in assessed_collection 
    if str(asset.attributes.get('gemma4_chs_level', '0')) in ['1', '2', '3', '4']
]

damaged_collection = PhysicalAssetCollection(damaged_assets_list)
print(f'Identified {len(damaged_collection)} damaged buildings for recovery analysis.')


# Step 4: Recovery assessment:
print('\n--- Step 4: Assessing Recovery Progress ---')
# Reuse the exact same model loaded in VRAM to prevent GPU memory issues:
analyzer.prompt = recovery_prompt
analyzer.batch_size = 2
analyzer.max_images_per_asset = 4
analyzer.image_filter = None  # Remove the filter so it sees both Feb & Sep

print('Initiating recovery assessment...')
final_recovery_collection = analyzer(damaged_collection)

# Clean up empty assets and serialize the mutated collection:
final_collection = final_recovery_collection.filter_empty()
output_file = 'damaged_buildings_recovery_status.geojson'

final_collection.to_geojson(output_file, ignore_properties=['image_assets'])
print(f'\nPipeline complete! Exported recovery data to: {output_file}')
