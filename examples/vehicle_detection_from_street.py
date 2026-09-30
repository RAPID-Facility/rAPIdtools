"""
Locate vehicles along a street-level survey and grade their condition.

This script demonstrates the street-level counterpart of the aerial
detect / extract / analyze workflow, using the UW RAPID Facility's Mapillary
imagery of Spokane, Washington:

1. Download the Mapillary access token used by the rapidtools examples.
2. Discover vehicles in every RAPID panorama of the area from Mapillary's own
   segmentation detections (metadata only, no images downloaded), estimate
   where each vehicle stands from the camera geometry, and merge repeated
   sightings into one asset per vehicle. Detections of the survey vehicle
   itself are removed automatically.
3. Crop the two closest views of each vehicle, with corner brackets marking
   the detection, so a vision-language model can judge its condition.
4. Optionally classify each vehicle as intact, damaged or debris with Gemini
   (only when a Google API key is available).
5. Export the vehicle locations and attributes to GeoJSON.

Run with::

    python examples/vehicle_detection_from_street.py

Set ``GOOGLE_API_KEY`` (or put the key in ``api_key.txt``) to enable step 4.

Memory: the region below covers the whole Spokane survey, tens of thousands
of panoramas. The detector reads them in batches of ``frame_batch_size``
frames and keeps only the simplified vehicle outlines, so RAM stays flat;
narrow ``START_DATE`` / ``END_DATE`` or the region for a quicker first run.
"""

import os
from pathlib import Path

import rapidtools as rt
from rapidtools.models import GenerationConfig, load

# --------------------------------------------------------------- Configuration
# A neighbourhood of Spokane, WA, inside the UW RAPID street-level survey:
# about 1 km across. Expect roughly 20 minutes end to end and 1.5 GB of RAM:
# a few minutes to read the detections, then about ten to download the two
# thousand images the vehicle crops are cut from. Around 1,300 vehicles.
REGION = rt.BoundingBox(min_x=-117.495, min_y=47.708, max_x=-117.482, max_y=47.717)
# The whole survey is the box below: 116,000 frames after thinning, which
# takes about 75 minutes just to read the detections (one Graph API request
# per frame). Use it once the neighbourhood run looks right.
#   REGION = rt.BoundingBox(
#       min_x=-117.535812, min_y=47.688456, max_x=-117.442589, max_y=47.737830
#   )
START_DATE = ''  # e.g. '2025-08-01' to restrict to one survey
END_DATE = ''
OUTPUT_DIR = Path('output/spokane_vehicles')
GEOJSON_PATH = OUTPUT_DIR / 'spokane_vehicles.geojson'

CLASSIFY_CONDITION = True  # Step 4 runs only if a Google API key is found
CONDITION_PROMPT = (
    'This crop from a post-disaster street-view panorama shows a vehicle '
    'marked with corner brackets. Return JSON with two keys: "condition", '
    'exactly one of intact, damaged, debris or unclear (intact: undamaged or '
    'merely dirty; damaged: visible fire or heat damage but the body is '
    'mostly whole; debris: burned-out shell, collapsed, only frame or ash); '
    'and "evidence", one short sentence.'
)

rt.configure_logging()

# --------------------------------------------------------------- Step 1
# The examples share one Mapillary token, published in the dataset registry:
[token_path] = rt.download_dataset('mapillary_token', output_dir=OUTPUT_DIR)
MAPILLARY_TOKEN = token_path.read_text().strip()

# --------------------------------------------------------------- Steps 2 + 3
detector = rt.MapillaryFeatureExtractor(
    classes=['vehicles'],  # cars, trucks, buses, trailers, ... (Mapillary labels)
    access_token=MAPILLARY_TOKEN,
    region=REGION,
    start_date=START_DATE,
    end_date=END_DATE,
    filter_rapid_only=True,  # only imagery uploaded by the RAPID Facility
    frame_spacing_m=3.0,  # one frame every 3 m is plenty for triangulation
    min_observations=2,  # ignore vehicles seen in a single frame
    cluster_radius_m=4.0,  # sightings within 4 m are the same vehicle
    camera_height_m=2.4,  # roof-mounted 360 camera on the RAPID vehicle
    frame_batch_size=200,  # frames fetched and converted at a time (bounds RAM)
    save_directory=OUTPUT_DIR / 'detections',
)

cropper = rt.MapillaryObjectImageExtractor(
    save_directory=OUTPUT_DIR / 'crops',
    access_token=MAPILLARY_TOKEN,
    max_images_per_asset=2,  # the two closest distinct views
    image_size='2048',  # each panorama is downloaded once, cropped and released
    crop_buffer='40%',
    overlay_asset_outline=True,
    outline_shape='corners',
    outline_color='#00ffff',
)

steps = [detector, cropper]

# --------------------------------------------------------------- Step 4
api_key = os.environ.get('GOOGLE_API_KEY', '')
if not api_key and Path('api_key.txt').is_file():
    api_key = Path('api_key.txt').read_text().strip()
if CLASSIFY_CONDITION and api_key:
    steps.append(
        rt.AssetAnalyzer(
            load('gemini', api_key=api_key, model_id='gemini-3.8-flash'),
            prompt=CONDITION_PROMPT,
            max_workers=4,
            generation=GenerationConfig(json_mode=True, temperature=0.0),
        )
    )
elif CLASSIFY_CONDITION:
    print('No Google API key found; skipping the condition classification step.')

# --------------------------------------------------------------- Run
# The pipeline orders the steps by stage: detect, extract imagery, analyze.
vehicles = rt.Pipeline(steps).run(rt.PhysicalAssetCollection())

print(f'\n{len(vehicles)} vehicles located.')
by_localization = {}
for asset in vehicles:
    key = asset.attributes['localization']
    by_localization[key] = by_localization.get(key, 0) + 1
print(f'Localization: {by_localization}')
if any('gemini_condition' in a.attributes for a in vehicles):
    conditions = {}
    for asset in vehicles:
        cond = asset.attributes.get('gemini_condition', 'not classified')
        conditions[cond] = conditions.get(cond, 0) + 1
    print(f'Condition: {conditions}')

# --------------------------------------------------------------- Step 5
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# The per-sighting observations are kept so the crops can be regenerated;
# drop 'observations' from ignore_properties below to write a lighter file.
vehicles.to_geojson(GEOJSON_PATH, ignore_properties=['image_assets'])
print(f'Written to {GEOJSON_PATH}')
