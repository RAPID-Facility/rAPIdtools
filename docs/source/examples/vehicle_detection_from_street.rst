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
           merge_pieces_of_neighbours=True,   # the default, named for the driveways
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

- ``localization`` is ``'triangulated'`` for vehicles whose track was
  intersected from two or more camera positions and ``'single_view'`` for
  vehicles seen without parallax. Triangulated positions are typically
  within a metre; single-view ones depend on the camera height and a flat
  road, so treat them as approximate.
- ``position_sigma_m``, ``parallax_deg``, ``n_images`` and
  ``position_rms_m`` tell you how well constrained each position is; filter
  on ``position_sigma_m`` for a map you can trust.
- Vehicles that were driving, anything on the line the camera itself
  drove, single views whose nearest sighting is beyond 30 m, and objects
  that nearby same-day frames should have seen but did not, are dropped
  before the collection is built; the log reports how many of each.
- One vehicle, one object. A car cut in two by a pole, or a track that slid
  onto the car behind it where a driveway is seen end-on, used to leave a
  second object a few metres from the real one. Such pieces are folded into
  the vehicle their sightings show (``merge_pieces_of_neighbours``), and an
  object never seen within 20 m is not trusted on its own: it joins the
  close pass it points at. With ``reid=True``, vehicles seen on separate
  passes that geometry could not join are compared by appearance (CLIP on a
  2048-pixel crop each) and merged when they match; only objects seen at
  least 100 pixels wide take part.
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
