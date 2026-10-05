"""
Locate vehicles along a street-level survey and grade their condition.

This script demonstrates the street-level counterpart of the aerial
detect / extract / analyze workflow, using the UW RAPID Facility's Mapillary
imagery of Spokane, Washington:

1. Download the Mapillary access token used by the rapidtools examples.
2. Discover vehicles in every RAPID panorama of the area from Mapillary's own
   segmentation detections (metadata only, no images downloaded). Each
   sighting becomes a ray from the camera's full pose; sightings are tracked
   from frame to frame, each track is triangulated, moving vehicles and the
   survey vehicle itself are dropped, and repeated views of a parked vehicle
   merge into one asset.
3. Crop the two closest views of each vehicle, with the detection's rotated
   bounding box drawn as a thin red line, so a vision-language model can
   judge its condition. Optionally confirm with a model of your choice that
   each object really is a vehicle, dropping Mapillary's false detections
   before any further analysis.
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
# thousand images the vehicle crops are cut from. Around 1,300 vehicles
# (timed with the earlier radius clustering; tracking adds seconds).
REGION = rt.BoundingBox(min_x=-117.495, min_y=47.708, max_x=-117.482, max_y=47.717)
# The whole survey is the box below: 116,000 frames after thinning, which
# takes about 75 minutes just to read the detections (one Graph API request
# per frame). Use it once the neighbourhood run looks right.
#   REGION = rt.BoundingBox(
#       min_x=-117.535812, min_y=47.688456, max_x=-117.442589, max_y=47.737830
#   )
# Only images captured in this window are used. The RAPID Spokane imagery
# comes from several survey days (late August and mid September 2026); one
# window keeps the vehicle inventory to a single survey, so a car that moved
# between surveys is not counted twice or vetoed by a later pass.
START_DATE = '2026-08-01'
END_DATE = '2026-08-31'  # inclusive; '' for no bound
OUTPUT_DIR = Path('output/spokane_vehicles')
GEOJSON_PATH = OUTPUT_DIR / 'spokane_vehicles.geojson'

# Mapillary's detector also fires on things that are not vehicles. Step 3b asks a
# model of your choice to confirm each object from its closest crops and drops
# the rest before any paid analysis. Any backend from rapidtools.models.load()
# works here; swap the two lines below for e.g. load('gemma4') to verify
# locally, or load('claude', api_key=...) for another provider.
VERIFY_DETECTIONS = True  # runs only if a Google API key is found
VERIFY_MODEL = ('gemini', 'gemini-3.5-flash-lite')  # cheap and quick for yes/no
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
    classes=['vehicles'],  # motor vehicles: cars, trucks, buses, motorcycles
    access_token=MAPILLARY_TOKEN,
    region=REGION,
    start_date=START_DATE,
    end_date=END_DATE,
    filter_rapid_only=True,  # only imagery uploaded by the RAPID Facility
    frame_spacing_m=3.0,  # one frame every 3 m is plenty for triangulation
    min_observations=3,  # a vehicle must appear in at least three frames
    save_directory=OUTPUT_DIR / 'detections',
    # The geometry needs no tuning: each sighting becomes a ray from the
    # camera pose Mapillary computed, sightings are tracked and triangulated,
    # and moving vehicles are dropped. The only physical constant involved,
    # the camera height above the road, defaults to the RAPID rig (2.4 m)
    # and only shapes the single-view range prior; pass camera_height_m for
    # a different vehicle.
    # The survey drove some streets twice, so look-alike vehicles seen on
    # separate passes are merged by appearance: the CLIP ViT-B/16 image tower
    # embeds one 2048-pixel crop per compared vehicle. Vehicles never seen up
    # close are left as geometry placed them.
    reid=True,
)

cropper = rt.MapillaryObjectImageExtractor(
    save_directory=OUTPUT_DIR / 'crops',
    access_token=MAPILLARY_TOKEN,
    max_images_per_asset=2,  # the two closest distinct views
    image_size='2048',  # each panorama is downloaded once, cropped and released
    crop_buffer='40%',
    overlay_asset_outline=True,
    outline_shape='rotated_bbox',  # the detection's minimum rotated rectangle
    outline_color='red',
    outline_width=1,  # a thin line, so the vehicle itself stays visible
)

steps = [detector, cropper]

api_key = os.environ.get('GOOGLE_API_KEY', '')
if not api_key and Path('api_key.txt').is_file():
    api_key = Path('api_key.txt').read_text().strip()

# --------------------------------------------------------------- Step 3b
# The verifier runs between cropping and analysis (its stage is VERIFY), so
# rejected objects never reach the condition model. Objects it rejects are
# removed; pass keep_rejected=True to keep them flagged instead.
if VERIFY_DETECTIONS and api_key:
    provider, model_id = VERIFY_MODEL
    steps.append(
        rt.DetectionVerifier(
            load(provider, api_key=api_key, model_id=model_id),
            max_images_per_asset=2,  # the two closest crops
            min_confidence=0.5,
            min_visible_fraction=0.5,  # heavily hidden objects are discarded
            max_workers=4,
        )
    )
elif VERIFY_DETECTIONS:
    print('No Google API key found; skipping the detection verification step.')

# --------------------------------------------------------------- Step 4
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
sigmas = sorted(
    a.attributes['position_sigma_m']
    for a in vehicles
    if 'position_sigma_m' in a.attributes
)
if sigmas:
    median_sigma = sigmas[len(sigmas) // 2]
    well_seen = sum(1 for a in vehicles if a.attributes.get('n_images', 0) >= 3)
    print(
        f'Position uncertainty: median {median_sigma:.2f} m; '
        f'{well_seen} vehicles seen from three or more frames.'
    )
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
