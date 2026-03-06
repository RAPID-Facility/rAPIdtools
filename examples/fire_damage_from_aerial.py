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
# 03-06-2026

"""
This script provides an example of the rapidtools data processing pipeline by:
1. Loading a spatial inventory of building footprints.
2. Extracting localized high-resolution aerial imagery for each asset.
3. Using the Gemini API to analyze the imagery and assign CHS categories.
4. Exporting the enriched dataset to a new GeoJSON file.
"""

from pathlib import Path

from rapidtools import GeminiAssetAnalyzer, Pipeline, PhysicalAssetCollection
from rapidtools import AerialImageryExtractor


def main():

    # Raster and vector spatial inputs:
    raster_path = Path('Eaton_Trinity38_RGBortho_20250214.tiff')
    footprint_path = Path('altadena_sample_buildings.geojson')
    image_save_dir = Path('eaton_fire_aerial_feb25/overlaid_imagery')

    # LLM inference credentials and instructions:
    api_key_path = Path('api_key.txt')
    prompt_path = Path('aerial_CHS_prompts.txt')

    # Load building footprint data:
    building_data = PhysicalAssetCollection.from_geojson(footprint_path)

    # Initialize computational pipeline
    pipeline = Pipeline()

    # Imagery Extractor
    # Crops the orthomosaic around each asset, draws a reference outline, 
    # and handles overlapping raster files by keeping multiple copies safely.
    extractor = AerialImageryExtractor(
        dataset=raster_path,
        save_directory=image_save_dir,
        overlay_asset_outline=True,
        image_prefix='eaton_trinity_25',
        keep_multiple_copies=True
    )

    # Gemini Analyzer
    # Ingests the newly cropped images and applies the configured prompt 
    # to evaluate and attach CHS categories to the asset's attributes.
    analyzer = GeminiAssetAnalyzer(
        api_key=api_key_path,
        prompt=prompt_path 
    )

    # Register the configured instances to the pipeline:
    pipeline.add_step(extractor)
    pipeline.add_step(analyzer)
    
    # Process the collection sequentially through the registered components
    print('Initiating processing pipeline...')
    processed_collection = pipeline.run(building_data)
    final_collection = processed_collection.filter_empty()
    print(f'\nFinal inventory size: {len(final_collection)} assets processed.')
    
    # Serialize the mutated collection, preserving the new AI-generated attributes
    output_file = Path('eaton_footprints_with_CHS.geojson')
    final_collection.to_geojson(output_file, ignore_properties=['image_assets'])
    print(f'Exported enriched inventory to: {output_file}')


if __name__ == '__main__':
    main()