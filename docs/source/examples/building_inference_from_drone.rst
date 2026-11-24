Buildings from perspective drone photos
=======================================

Turn a drone flight into a multi-view record of every building and ask a
vision-language model about it: for each footprint, the few photos that
together show the most of its walls and roof are chosen, cropped and marked,
and a local Gemma 4 model reports the recovery status of the property.

Source: ``examples/building_inference_from_drone.py``. The test flight comes
from ``examples/drone_flight_from_google_earth.py``, which needs no drone.

A flight without a drone
------------------------

The second script flies a virtual camera over the Altadena block burned in
the January 2025 Eaton fire, in the Google Earth web app's 3D imagery
(December 2025), driven through headless Chrome. It follows the usual
oblique mapping pattern (a grid of shots looking north, east, south and west
with the gimbal at 45 degrees, plus a straight-down grid, 452 photos from
70 m above the take-off point) and writes each render as a JPEG with
DJI-style EXIF and XMP metadata, so the photos are read exactly like the
ones a DJI aircraft produces.

.. code-block:: bash

   python examples/drone_flight_from_google_earth.py \
       --footprints eaton_patch1_bing_buildings.geojson \
       --output out/drone_flight --launch-chrome

The camera model was checked by projecting the footprints into the renders:
they land on their buildings. The imagery belongs to Google and its data
providers; use the renders to test software, not to redistribute.

The pipeline
------------

.. code-block:: python

   import rapidtools as rt
   from rapidtools.models import GenerationConfig, load

   [footprints] = rt.download_dataset('eaton_patch1_bing_buildings')
   buildings = rt.PhysicalAssetCollection.from_geojson(footprints)

   # Poses come from the photos. The take-off elevation puts the cameras in
   # the datum of the ground elevations below.
   cameras = rt.read_drone_images('out/drone_flight', takeoff_elevation_m=365.3)

   # The block climbs 23 m end to end: give every footprint its ground level.
   for asset in buildings:
       centre = asset.geometry.centroid
       asset.attributes['ground_elevation_m'] = usgs_elevation(centre.x, centre.y)

   extractor = rt.DroneImageExtractor(
       cameras,
       save_directory='out/drone_inference/crops',
       max_views=4,
       ground_elevation='ground_elevation_m',
       default_height_m=6.0,
       max_crop_px=640,
   )
   analyzer = rt.AssetAnalyzer(
       load('gemma4', model_id='google/gemma-4-E2B-it'),
       prompt=PROMPT,                       # asks for status, storeys, evidence
       max_images_per_asset=4,              # the model sees the whole view set
       batch_size=1,
       generation=GenerationConfig(json_mode=True, temperature=0.0, max_tokens=120),
   )
   buildings = rt.Pipeline([extractor, analyzer]).run(buildings)

   buildings.to_geojson('buildings_drone.geojson', ignore_properties=['image_assets'])
   extractor.coverage_gaps().to_geojson('coverage_gaps.geojson')

What the selection buys
-----------------------

On this flight every building appears in 13 photos on average. Measured as
the share of each building's walls and roof that the chosen photos show
(quality-weighted, averaged over the 179 buildings):

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Photos per building
     - Selected set
     - Nearest photos
     - Random photos
   * - 1
     - 0.50
     - 0.28
     - 0.32
   * - 2
     - 0.71
     - 0.45
     - 0.49
   * - 4
     - 0.87
     - 0.65
     - 0.67
   * - 6
     - 0.91
     - 0.78
     - 0.77

All 452 photos together reach 0.92, so four selected photos carry 95 % of
what the whole flight shows of a building. With four photos, every wall is
seen for 70 % of the buildings; with the four nearest photos that is true of
13 %. The nearest photo is a poor choice because the nearest camera is the
one almost overhead, which sees the roof and nothing else.

What the model reports
----------------------

Gemma 4 E2B takes about five seconds per building for four 640-pixel crops.
On this block it called 157 buildings standing, 16 lots empty and 4 houses
under construction; ten of the twenty non-standing answers were checked by
eye and all ten were right. The prompt matters with a model this small: it
asks for one answer for the whole view set, and it does not mention the fire
(told that there had been one, the model reported debris on nearly every
standing house).

Outputs
-------

- ``buildings_drone.geojson``: the footprints with the model's answer
  (``gemma4_status``, ``gemma4_stories``, ``gemma4_evidence``) and the view
  coverage attributes (``drone_view_coverage``, ``drone_unseen_sides``, ...).
- ``coverage_gaps.geojson``: the walls no photo shows, each with a suggested
  camera position for the next flight.
- ``contact_sheets/``: the chosen views of a few buildings side by side,
  labelled with the direction each was taken from and what it adds.

With a real drone
-----------------

Point ``FLIGHT_DIR`` at the folder of photos. On flat ground drop the
elevation step and the ``ground_elevation`` argument; otherwise give
``takeoff_elevation_m`` in the datum of your ground elevations. Consumer GPS
and compass errors move a building by a fraction of its size in the photo;
the crops allow for that (``position_error_m``, ``angle_error_deg``), and
``refine_poses=True`` uses SAM 3 to shift each photo's footprints onto the
buildings it actually shows. See :doc:`../user_guide/drone_imagery`.
