Vehicles along a street-level survey
====================================

Find every vehicle the RAPID survey vehicle drove past in Spokane, place it
on the map, crop the two best views and, optionally, grade its condition.
No inventory is needed: the objects come from Mapillary's segmentation of the
survey imagery.

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

What to look at in the output
-----------------------------

- ``localization`` is ``'triangulated'`` for vehicles placed by
  intersecting the bearings of their sightings, typically within half a
  metre, and ``'fused'`` for the few placed at the centre of their occupied
  ground cells.
- ``position_sigma_m``, ``parallax_deg``, ``n_images`` and
  ``position_rms_m`` tell you how well constrained each position is, and
  ``length_m``, ``width_m`` and ``footprint_area_m2`` describe the ground
  the vehicle occupies (generous in depth: one pass cannot see how deep a
  car is). ``survey_days`` lists the days it was seen on.
- Vehicles that were driving never collect agreeing votes and the ground
  the survey vehicle drove over is free by construction, so neither needs a
  rule; a few detections of an empty kerb are cleared by the frames that
  pass it and see nothing. The log reports how many occupied blobs were
  found, split and dropped.
- One vehicle, one object. All frames of a survey day vote on the same
  ground, so a car seen on the way in, on the way out and from the side
  street is one blob, and two cars parked door to door are split where the
  frames keep seeing two outlines. With ``reid=True``, vehicles that
  geometry left apart are also compared by appearance (CLIP on a 2048-pixel
  crop each) and merged when they match; only objects seen at least 100
  pixels wide take part.
- The verifier removes objects whose closest crops a model says are not a
  motor vehicle, or are less than half visible, and writes
  ``verify_accepted``, ``verify_label``, ``verify_confidence`` and
  ``verify_visible_fraction``. On the August 2026 survey of this
  neighbourhood it keeps roughly 450 of 650 objects, most of the rest being
  far or partial views.
- ``observations`` lists every sighting with its image ID, so the crops can be
  regenerated later without re-running detection.
- A vehicle has to appear in at least three distinct frames
  (``min_observations=3``); lower it to keep vehicles glimpsed once or
  twice, at the price of more false detections.
- ``'vehicles'`` means motor vehicles. Trailers, caravans and boats are
  their own classes (``'trailers'``, ``'boats'``), and the verifier's
  question excludes them too, so a boat on a trailer is not counted as a
  car.

The detection stage downloads no images, only detection metadata, so the
1 km neighbourhood above runs in about six minutes, most of it reading
detections; a whole city takes about an hour and a quarter. The crop stage
fetches each 2048 pixel source image once and cuts every vehicle seen in it;
the appearance step fetches one more per compared vehicle.
