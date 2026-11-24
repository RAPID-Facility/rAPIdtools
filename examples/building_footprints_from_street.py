"""
Map the houses along a street-level survey as footprint polygons.

A drive-by survey never sees a whole house: the front, a slice of each side
wall and the near roof slope are all there is. This script shows how
rapidtools still turns the UW RAPID Facility's Mapillary imagery of Spokane,
Washington into one polygon per building, and the same survey into vehicle
positions, with a single detector:

1. Download the Mapillary access token used by the rapidtools examples.
2. Discover buildings and vehicles in every RAPID panorama of the area.

   - Buildings are placed from the survey's own 3-D reconstruction.
     Mapillary publishes the sparse point cloud each panorama was aligned
     in; the detector keeps the points most frames see inside a building
     outline, groups them into structures, and outlines each one. The
     facade and the sides the survey saw are measured. The rear is not, so
     the footprint is closed to ``assumed_depth_m`` where less was seen and
     the ``depth_source`` attribute says so.
   - Vehicles are placed by occupancy fusion: every outline is read as a
     range scan and all frames of a survey day vote on a ground grid.

3. Cut the closest views of every building from the panoramas, each with
   the building marked, ready for a vision-language model.
4. Export footprints and vehicle points to GeoJSON and draw a quick map.

Run with::

    python examples/building_footprints_from_street.py

The region below is four blocks; expect about three minutes. Point clouds
are cached under ``OUTPUT_DIR/detections/point_clouds``, so a second run
downloads none of them again.
"""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import rapidtools as rt  # noqa: E402

# --------------------------------------------------------------- Configuration
# Four blocks of Spokane, WA, inside the UW RAPID street-level survey.
REGION = rt.BoundingBox(min_x=-117.4915, min_y=47.7112, max_x=-117.4870, max_y=47.7138)
START_DATE = '2026-08-01'
END_DATE = '2026-08-31'  # one survey day; '' for no bound

OUTPUT_DIR = Path('output/spokane_buildings')
GEOJSON_PATH = OUTPUT_DIR / 'spokane_buildings.geojson'
MAP_PATH = OUTPUT_DIR / 'spokane_buildings.png'

# --------------------------------------------------------------- Step 1
rt.configure_logging()
# The examples share one Mapillary token, published in the dataset registry:
# (a token file works wherever a token does; so does leaving the token out
# and setting the MAPILLARY_ACCESS_TOKEN environment variable).
[MAPILLARY_TOKEN] = rt.download_dataset('mapillary_token', output_dir=OUTPUT_DIR)

# --------------------------------------------------------------- Step 2
detector = rt.MapillaryFeatureExtractor(
    classes=['buildings', 'vehicles'],
    access_token=MAPILLARY_TOKEN,
    region=REGION,
    start_date=START_DATE,
    end_date=END_DATE,
    filter_rapid_only=True,
    frame_spacing_m=3.0,
    min_observations=3,
    save_directory=OUTPUT_DIR / 'detections',
    # localization_method='auto' (the default) uses the point cloud for
    # buildings and occupancy fusion for vehicles. geometry='auto' writes
    # polygons for buildings and points for vehicles; 'footprint' would give
    # the vehicles polygons too.
    assumed_depth_m=10.0,  # depth drawn where the survey saw less of a house
    min_gap_m=1.0,  # free ground that separates one house from the next
    structure_views=3,  # views kept per building: from the left, front, right
)

# --------------------------------------------------------------- Step 3
cropper = rt.MapillaryObjectImageExtractor(
    save_directory=OUTPUT_DIR / 'crops',
    access_token=MAPILLARY_TOKEN,
    max_images_per_asset=3,
    image_size='2048',
    crop_buffer='15%',
    overlay_asset_outline=True,
    outline_shape='corners',  # corner brackets leave the facade unobstructed
    outline_color='red',
    outline_width=2,
)

assets = rt.Pipeline([detector, cropper]).run()

# --------------------------------------------------------------- Step 4
buildings = [a for a in assets if a.attributes.get('localization') == 'point_cloud']
vehicles = [a for a in assets if a.attributes.get('asset_type') == 'vehicles']
measured = sum(a.attributes['depth_source'] == 'observed' for a in buildings)
print(f'{len(buildings)} buildings ({measured} with the depth seen), ')
print(f'{len(vehicles)} vehicles.')
for asset in buildings[:5]:
    at = asset.attributes
    print(
        f'  {asset.id}: frontage {at["frontage_m"]} m, depth {at["depth_m"]} m '
        f'({at["depth_source"]}), height {at["height_m"]} m, '
        f'{at["n_images"]} views, {len(asset.image_assets)} crops'
    )

assets.to_geojson(GEOJSON_PATH)
print(f'Wrote {GEOJSON_PATH}')

fig, ax = plt.subplots(figsize=(10, 10))
for asset in buildings:
    seen = asset.attributes['depth_source'] == 'observed'
    xs, ys = asset.geometry.exterior.xy
    ax.fill(xs, ys, color='#4c78a8' if seen else '#9ecae9', ec='k', lw=0.6)
if vehicles:
    ax.plot(
        [a.geometry.x for a in vehicles],
        [a.geometry.y for a in vehicles],
        '.',
        color='#e45756',
        ms=5,
    )
ax.set_aspect(1.0 / 0.673)  # meters per degree at 47.7 degrees north
ax.set_title('Buildings (dark: depth seen, light: depth assumed) and vehicles')
fig.savefig(MAP_PATH, dpi=120, bbox_inches='tight')
print(f'Wrote {MAP_PATH}')
