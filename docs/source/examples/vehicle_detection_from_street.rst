Vehicles along a street-level survey
====================================

This example finds every vehicle the RAPID survey vehicle drove past in
Spokane, WA, places each one on the map, crops its two best views and,
optionally, grades its condition. No inventory is needed: the vehicles come
from Mapillary's segmentation of the survey imagery.

Source: ``examples/vehicle_detection_from_street.py``.

.. code-block:: python

   import rapidtools as rt
   from rapidtools.models import GenerationConfig, load

   REGION = rt.BoundingBox(min_x=-117.495, min_y=47.708, max_x=-117.482, max_y=47.717)
   [token_path] = rt.download_dataset('mapillary_token')
   token = token_path.read_text().strip()

   pipeline = rt.Pipeline([
       # 1. Vehicles from Mapillary's detections, placed by camera geometry.
       rt.MapillaryFeatureExtractor(
           classes=['vehicles'],
           access_token=token,
           region=REGION,
           start_date='2026-08-01', end_date='2026-08-31',   # one survey window
           filter_rapid_only=True,
           frame_spacing_m=3.0,
           min_observations=3,
           reid=True,                         # repeated passes merged by appearance
       ),
       # 2. The two closest views of each vehicle, with its outline drawn.
       rt.MapillaryObjectImageExtractor(
           'output/spokane_vehicles/crops',
           access_token=token,
           max_images_per_asset=2,
           image_size='2048',
           crop_buffer='40%',
           overlay_asset_outline=True,
           outline_shape='rotated_bbox',
           outline_color='red',
           outline_width=1,
       ),
       # 3. A model confirms each object is a vehicle (needs a Google API key).
       rt.DetectionVerifier(
           load('gemini', api_key='AIza...', model_id='gemini-3.5-flash-lite'),
           min_visible_fraction=0.5,
       ),
       # 4. Condition from a vision-language model (optional; off in the script).
       rt.AssetAnalyzer(
           load('gemini', api_key='AIza...'),
           prompt='Return JSON with a "condition" key: intact, damaged, debris or unclear.',
           generation=GenerationConfig(json_mode=True, temperature=0.0),
       ),
   ])
   vehicles = pipeline.run(rt.PhysicalAssetCollection())
   vehicles.to_geojson('spokane_vehicles.geojson', ignore_properties=['image_assets'])

Where each vehicle is
---------------------

Most vehicles have ``localization='triangulated'``: they were placed by
intersecting the bearings of their sightings, typically to within half a
meter. A few read ``'fused'``, meaning they sit at the center of the ground
cells they occupy. To judge how well a position is pinned down, look at
``position_sigma_m``, ``parallax_deg``, ``n_images`` and
``position_rms_m``. ``length_m``, ``width_m`` and ``footprint_area_m2``
describe the ground the vehicle covers; they are generous in depth, because
one pass along a street cannot see how deep a car is. ``survey_days`` lists
the days the vehicle was seen.

Each vehicle also lists every sighting, with its image ID, under
``observations``, so you can regenerate the crops later without running
detection again.

One vehicle, one object
-----------------------

All the frames of a survey day vote on the same ground grid. A car seen on
the way in, on the way out and from a side street therefore ends up as one
object, and two cars parked bumper to bumper are split where the frames
keep seeing two outlines. Vehicles that were driving never collect agreeing
votes, the road the survey vehicle drove along is free by construction, and
a few stray detections of an empty curb are cleared by the frames that pass
it and see nothing. None of this needs a special rule. The log reports how
many occupied patches were found, split and dropped.

With ``reid=True``, vehicles that geometry left apart are also compared by
appearance (CLIP on a 2,048-pixel crop of each) and merged when they match.
Only vehicles seen at least 100 pixels wide take part.

A vehicle has to appear in at least three different frames
(``min_observations=3``). Lower the number to keep vehicles that were only
glimpsed once or twice, at the cost of more false detections.

What the verifier removes
-------------------------

The verifier drops objects whose closest crops the model says are not a
motor vehicle, or are less than half visible. It writes
``verify_accepted``, ``verify_label``, ``verify_confidence`` and
``verify_visible_fraction`` on each object. On the August 2026 survey of
this neighborhood it keeps roughly 450 of 650 objects; most of the rest are
distant or partial views.

``'vehicles'`` means motor vehicles. Trailers, caravans and boats are
classes of their own (``'trailers'``, ``'boats'``), and the verifier's
question excludes them too, so a boat on a trailer is not counted as a car.

Run time
--------

The detection step downloads detection metadata only, no images, so the
1 km neighborhood above takes about six minutes, most of it spent reading
detections. A whole city takes about an hour and a quarter. The crop step
fetches each 2,048-pixel source image once and cuts out every vehicle seen
in it, and the appearance comparison fetches one more image per vehicle it
compares.
