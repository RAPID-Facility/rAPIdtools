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
# 11-12-2026

"""
Building-level inference from perspective drone photos.

A drone flight photographs every building dozens of times, from many
directions. This example turns such a flight into one compact, multi-view
record per building and asks a vision-language model about it:

1. Load building footprints and the drone photos (camera poses are read from
   the photos' own EXIF/XMP metadata).
2. Give every footprint the ground elevation under it, so buildings on a
   slope project to the right place in the photos.
3. ``DroneImageExtractor`` picks, for each building, the few photos that
   together show the most of its walls and roof, crops them around the
   building and marks it with corner brackets.
4. A local Gemma 4 model looks at each building's view set and reports its
   recovery status.
5. Outputs: a GeoJSON with the model's answers and the view coverage of every
   building, a GeoJSON of the walls no photo shows (with a suggested camera
   position for each, to plan the next flight), and contact sheets of the
   chosen views.

The test flight is simulated: ``drone_flight_from_google_earth.py`` renders
oblique views of the Altadena, California block burned in the January 2025
Eaton fire from Google Earth's 3D imagery (December 2025) and writes
DJI-style metadata into them. With photos from a real drone, point
``FLIGHT_DIR`` at the folder and set ``TAKEOFF_ELEVATION_M`` (or leave the
terrain flat, see Step 2).
"""

from __future__ import annotations

import json
from pathlib import Path

import requests
from PIL import Image, ImageDraw

import rapidtools as rt
from rapidtools.data_sources import read_drone_images
from rapidtools.models import GenerationConfig, load
from rapidtools.processing import DroneImageExtractor

FLIGHT_DIR = Path('out/drone_flight')
OUTPUT_DIR = Path('out/drone_inference')
MAX_VIEWS = 4  # photos kept per building
RUN_MODEL = True  # set to False to stop after view selection and cropping
MODEL_ID = 'google/gemma-4-E2B-it'  # about 5 GB of GPU memory
N_CONTACT_SHEETS = 6
# Gemma 4 E2B takes about five seconds per building on a 16 GB GPU. Set to
# e.g. 40 to send only that many buildings (spread evenly over the area) to
# the model on a first run; None analyzes all of them.
MAX_MODEL_BUILDINGS = None

# Written for a small local model: neutral wording (telling Gemma 4 E2B that
# there was a wildfire makes it see debris everywhere), one answer for the
# whole view set, and no room to drift into listing boxes.
PROMPT = (
    'The photos show the same property from several drone viewpoints. Red '
    'corner brackets mark the property in each photo. Look at what stands '
    'inside the brackets and answer with ONE JSON object for the property as '
    'a whole (not one per photo, no coordinates), with exactly these keys: '
    '"status": one of "standing" (a complete building with a finished roof), '
    '"under_construction" (a building being built: exposed wood framing, bare '
    'sheathing, no finished roof), "empty_lot" (no building: bare soil, a '
    'concrete slab or vegetation); "stories": the number of stories of the '
    'building, 0 if there is none; "evidence": one short sentence describing '
    'what you see inside the brackets.'
)

# --------------------------------------------------------------- Step 1
# Footprints detected earlier from aerial imagery (see
# building_detection_from_bing.py) and the drone photos.
footprint_path = rt.download_dataset('eaton_patch1_bing_buildings')[0]
buildings = rt.PhysicalAssetCollection.from_geojson(footprint_path)

if not any(FLIGHT_DIR.glob('*.JPG')):
    raise SystemExit(
        f'No drone photos in {FLIGHT_DIR}. Render the test flight first:\n'
        '  python examples/drone_flight_from_google_earth.py '
        f'--footprints {footprint_path} --output {FLIGHT_DIR} --launch-chrome'
    )

# DJI aircraft record the height above the take-off point; adding the
# elevation of that point puts the cameras in the same datum as the ground
# elevations of Step 2. The simulated flight stores it in flight.json.
flight_info = FLIGHT_DIR / 'flight.json'
TAKEOFF_ELEVATION_M = (
    json.loads(flight_info.read_text())['takeoff_elevation_m']
    if flight_info.exists()
    else 0.0
)
cameras = read_drone_images(FLIGHT_DIR, takeoff_elevation_m=TAKEOFF_ELEVATION_M)
print(f'{len(cameras)} drone photos, {len(buildings)} buildings.')


# --------------------------------------------------------------- Step 2
# This neighborhood climbs 23 m from one end to the other, which at a 45
# degree view is 23 m of error in where a building appears. Ground elevations
# come from the USGS elevation service (United States only) and are cached.
# On flat terrain skip this step and drop the ground_elevation argument
# below: the take-off point is then the ground level everywhere.
def usgs_elevation(lon: float, lat: float) -> float:
    reply = requests.get(
        'https://epqs.nationalmap.gov/v1/json',
        params={'x': lon, 'y': lat, 'units': 'Meters', 'wkid': 4326},
        timeout=60,
    )
    return round(float(reply.json()['value']), 2)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
cache_path = OUTPUT_DIR / 'ground_elevations.json'
elevations = json.loads(cache_path.read_text()) if cache_path.exists() else {}
for asset in buildings:
    if asset.id not in elevations:
        center = asset.geometry.centroid
        elevations[asset.id] = usgs_elevation(center.x, center.y)
    asset.attributes['ground_elevation_m'] = elevations[asset.id]
cache_path.write_text(json.dumps(elevations))

# --------------------------------------------------------------- Step 3
# View selection and cropping. Buildings are modeled as 6 m tall boxes
# unless they carry a height attribute; neighbors hide one another.
extractor = DroneImageExtractor(
    cameras,
    save_directory=OUTPUT_DIR / 'crops',
    max_views=MAX_VIEWS,
    ground_elevation='ground_elevation_m',
    default_height_m=6.0,
    max_crop_px=640,  # plenty for one building; keeps the model quick
)
buildings = extractor(buildings)

# --------------------------------------------------------------- Step 4
if RUN_MODEL:
    analyzer = rt.AssetAnalyzer(
        load('gemma4', model_id=MODEL_ID),
        prompt=PROMPT,
        max_images_per_asset=MAX_VIEWS,
        batch_size=1,  # four photos per building: one building at a time
        generation=GenerationConfig(json_mode=True, temperature=0.0, max_tokens=120),
    )
    # View selection used every building (neighbors hide one another); the
    # model may be given fewer. The subset shares its assets with
    # ``buildings``, so the answers land on the full collection.
    subset = [a for a in buildings if len(a.image_assets) > 0]
    if MAX_MODEL_BUILDINGS and len(subset) > MAX_MODEL_BUILDINGS:
        subset = subset[:: len(subset) // MAX_MODEL_BUILDINGS][:MAX_MODEL_BUILDINGS]
    analyzer(rt.PhysicalAssetCollection(subset))

# --------------------------------------------------------------- Step 5
coverage = [a.attributes['drone_view_coverage'] for a in buildings]
views = [a.attributes['n_drone_views'] for a in buildings]
print(
    f'\nMean view coverage {sum(coverage) / len(coverage):.2f} with '
    f'{sum(views) / len(views):.1f} photos per building.'
)
if RUN_MODEL:
    statuses: dict[str, int] = {}
    for asset in buildings:
        status = str(asset.attributes.get('gemma4_status', 'not analyzed'))
        statuses[status] = statuses.get(status, 0) + 1
    print(f'Recovery status: {statuses}')

buildings.to_geojson(
    OUTPUT_DIR / 'buildings_drone.geojson', ignore_properties=['image_assets']
)
gaps = extractor.coverage_gaps()
gaps.to_geojson(OUTPUT_DIR / 'coverage_gaps.geojson')
print(f'{len(gaps)} walls were not captured; see coverage_gaps.geojson.')


def contact_sheet(asset: rt.PhysicalAsset, path: Path, height: int = 420) -> None:
    """The chosen views of one building side by side, with their labels."""
    crops = [Image.open(image.path) for image in asset.image_assets]
    crops = [c.resize((int(c.width * height / c.height), height)) for c in crops]
    width = sum(c.width for c in crops) + 10 * (len(crops) + 1)
    sheet = Image.new('RGB', (max(width, 900), height + 64), 'white')
    draw = ImageDraw.Draw(sheet)
    answer = asset.attributes.get('gemma4_status', '')
    draw.text(
        (10, 6),
        f'{asset.id}   coverage {asset.attributes["drone_view_coverage"]:.2f}   '
        f'unseen sides: {asset.attributes["drone_unseen_sides"] or "none"}   '
        f'{answer}   {asset.attributes.get("gemma4_evidence", "")}',
        fill='black',
    )
    x = 10
    for crop, image in zip(crops, asset.image_assets, strict=True):
        p = image.properties
        draw.text(
            (x, 26),
            f'{p["image_id"]}  from {p["view_azimuth_deg"]:.0f} deg, '
            f'{p["distance_m"]:.0f} m\nadds {p["coverage_gain"]:.0%}: '
            f'{", ".join(p["shows"])}',
            fill='black',
        )
        sheet.paste(crop, (x, 60))
        x += crop.width + 10
    sheet.save(path, quality=90)


(OUTPUT_DIR / 'contact_sheets').mkdir(exist_ok=True)
with_views = [
    a
    for a in buildings
    if len(a.image_assets) > 0 and (not RUN_MODEL or 'gemma4_status' in a.attributes)
]
step = max(1, len(with_views) // N_CONTACT_SHEETS)
for asset in with_views[::step][:N_CONTACT_SHEETS]:
    contact_sheet(asset, OUTPUT_DIR / 'contact_sheets' / f'{asset.id}.jpg')
print(f'Results in {OUTPUT_DIR.resolve()}')
