Building footprints from a street-level survey
==============================================

Map every house the RAPID survey vehicle drove past in Spokane as a polygon,
with the views that show it, and the parked vehicles as points, from the
survey imagery alone. No footprint layer is needed.

Source: ``examples/building_footprints_from_street.py``.

.. code-block:: python

   import rapidtools as rt

   REGION = rt.BoundingBox(min_x=-117.4915, min_y=47.7112, max_x=-117.4870, max_y=47.7138)
   [token] = rt.download_dataset('mapillary_token')   # a token file works as a token

   pipeline = rt.Pipeline([
       # 1. Buildings from the survey's point cloud, vehicles by occupancy fusion.
       rt.MapillaryFeatureExtractor(
           classes=['buildings', 'vehicles'],
           access_token=token,
           region=REGION,
           start_date='2026-08-01', end_date='2026-08-31',
           filter_rapid_only=True,
           frame_spacing_m=3.0,
           min_observations=3,
           assumed_depth_m=10.0,   # depth drawn where the survey saw less of a house
           structure_views=3,      # views kept per building: left, front, right
       ),
       # 2. Those views, cropped, with the building marked.
       rt.MapillaryObjectImageExtractor(
           'output/spokane_buildings/crops',
           access_token=token,
           max_images_per_asset=3,
           crop_buffer='15%',
           overlay_asset_outline=True,
           outline_shape='corners',
       ),
   ])
   assets = pipeline.run()
   assets.to_geojson('spokane_buildings.geojson')

What to look at in the output
-----------------------------

- Buildings are polygons with ``localization='point_cloud'``; vehicles are
  points. Both carry their sightings under ``observations`` and their crops
  under ``image_assets``, so an :class:`~rapidtools.processing.AssetAnalyzer`
  can follow directly.
- ``frontage_m`` and the position of the facade are measured from
  reconstructed points on the walls. ``depth_m`` is measured only when
  ``depth_source`` is ``'observed'`` (roof or side-wall points reach that
  deep); with ``'assumed'`` the rear was closed at ``assumed_depth_m``.
  ``observed_depth_m`` always says how deep the points go.
- ``height_m`` is the height of the highest points above the road, and
  ``facade_bearing`` the compass direction the front faces.
- A house is occasionally reported as two structures (a tree hiding the
  middle of the facade) or joined with its neighbour (side yards under a
  metre). ``min_gap_m`` moves that balance.
- Each building's three crops come from the left, the front and the right,
  marked with corner brackets. The outline in each image is where the
  building's points project, which the per-image segmentation alone could
  not provide: Mapillary labels all buildings in a frame as one surface.

On the four blocks above the script takes about four minutes: 80 s to read
the detections of 884 panoramas, 40 s to download 15 point clouds and vote,
15 s for the vehicles, and a minute to cut the crops. The method and its
measured accuracy are described in :doc:`../user_guide/street_level_imagery`.
