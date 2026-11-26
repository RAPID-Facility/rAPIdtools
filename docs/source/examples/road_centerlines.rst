Road centerlines from an orthomosaic
====================================

This example finds the paved roads in a drone orthomosaic and reduces the
rough masks to a connected network of centerlines, each with a width.

Source: ``examples/road_detection_from_aerial.py``.

.. code-block:: python

   from rapidtools import RoadwayRegularizer, SAM3OrthoFeatureExtractor, download_dataset

   [raster] = download_dataset('eaton_patch1')

   raw_roads = SAM3OrthoFeatureExtractor(
       prompt='paved road, street',
       patch_size=100, unit='meters',   # roads need context, so larger tiles
       overlap_ratio=0.25,
       batch_size=4,
       threshold=0.25,                  # roads are low-contrast; accept more
       mask_threshold=0.25,
   )(raster)
   raw_roads.to_geojson('roads_raw.geojson')

   centerlines, polygons = RoadwayRegularizer()(raw_roads)
   centerlines.to_geojson('roads_centerlines.geojson')
   polygons.to_geojson('roads_polygons.geojson')

Each centerline has ``width_ft`` and ``azimuth_deg`` attributes. Pass
``min_width_ft`` to set a minimum road width, and ``min_network_length_ft``
to choose how short a disconnected stub has to be before it is dropped.
