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

   REGION = rt.BoundingBox(-117.535812, 47.688456, -117.442589, 47.737830)
   [token_path] = rt.download_dataset('mapillary_token')
   token = token_path.read_text().strip()

   pipeline = rt.Pipeline([
       # 1. Vehicles from Mapillary's detections, placed by camera geometry.
       rt.MapillaryFeatureExtractor(
           classes=['vehicles'],
           access_token=token,
           region=REGION,
           filter_rapid_only=True,
           frame_spacing_m=3.0,
           min_observations=2,
       ),
       # 2. The two closest views of each vehicle, with corner brackets.
       rt.MapillaryObjectImageExtractor(
           'output/spokane_vehicles/crops',
           access_token=token,
           max_images_per_asset=2,
           overlay_asset_outline=True,
           outline_shape='corners',
           outline_color='#00ffff',
       ),
       # 3. Condition from a vision-language model (needs a Google API key).
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
- Vehicles that were driving, and anything on the line the camera itself
  drove, are dropped before the collection is built; the log reports how
  many.
- ``observations`` lists every sighting with its image ID, so the crops can be
  regenerated later without re-running detection.
- Vehicles detected only in one frame are dropped by ``min_observations=2``;
  lower it to 1 to keep them.

The detection stage downloads no images, only detection metadata, so it runs
in minutes over a whole city. The crop stage fetches two 2048 pixel
thumbnails per vehicle.
