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
# 09-24-2026

"""
This script demonstrates how to:
1. Load a building inventory shapefile and secure API credentials from a local 
   text file.
2. Download pre-packaged aerial assessment prompts using rapidtools.
3. Initialize the Gemini 3.8 Flash vision model.
4. Extract cropped aerial raster imagery around each building footprint with a 
   5 m buffer and outline overlay.
5. Grade each building asset using an automated Gemini analysis pipeline.
6. Export the updated building inventory with AI-generated assessment 
   ttributes to a shapefile.
"""

import logging
from pathlib import Path

import rapidtools as rt
from rapidtools.models import GenerationConfig, load

# Configuration
MODEL_ID = 'gemini-3.8-flash'
BUILDINGS_PATH = Path('path/to/buildings.shp')
RASTER_PATH = Path('path/to/raster.tif')
OUTPUT_PATH = Path('output/buildings_chs.shp')
API_KEY_PATH = Path('api_key.txt')
BUFFER_M = 5.0
MAX_WORKERS = 4


rt.configure_logging()

# Read API key from text file
if not API_KEY_PATH.is_file():
    raise FileNotFoundError(f"API key file not found at {API_KEY_PATH}")
api_key = API_KEY_PATH.read_text().strip()

work_dir = OUTPUT_PATH.parent
work_dir.mkdir(parents=True, exist_ok=True)

# 1. Building inventory (WGS84 shapefile):
buildings = rt.PhysicalAssetCollection.from_shapefile(
    BUILDINGS_PATH, asset_type='building'
)
logging.info(f'Loaded {len(buildings)} buildings from {BUILDINGS_PATH}')

# 2. CHS prompt shipped with rapidtools:
[prompt_path] = rt.download_dataset('aerial_chs_prompts', output_dir=work_dir)

# 3. Gemini 3.8 Flash:
model = load('gemini', api_key=api_key, model_id=MODEL_ID)

# 4. Crop the raster around each footprint (red outline for the prompt),
#    then grade the crops:
pipeline = rt.Pipeline(
    [
        rt.AerialImageryExtractor(
            RASTER_PATH,
            save_directory=work_dir / 'crops',
            buffer_m=BUFFER_M,
            overlay_asset_outline=True,
            force_square_image=True,
            image_prefix='aerial',
        ),
        rt.AssetAnalyzer(
            model,
            prompt=prompt_path,
            max_workers=MAX_WORKERS,
            generation=GenerationConfig(temperature=0.0),
        ),
    ]
)
buildings = pipeline.run(buildings)

# 5. Save the graded inventory (gemini_chs_level, gemini_justification,
#    ai_model_used columns) as a shapefile:
buildings.to_shapefile(OUTPUT_PATH, ignore_properties=['image_assets'])
graded = sum('gemini_chs_level' in a.attributes for a in buildings)
logging.info(f'{graded} of {len(buildings)} buildings graded; written to {OUTPUT_PATH}')
