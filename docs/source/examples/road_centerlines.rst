Road centrelines from an orthomosaic
====================================

Extract paved roads from a drone orthomosaic and reduce the raw masks to a
connected network of centrelines with widths.

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

Each centreline carries ``width_ft`` and ``azimuth_deg`` attributes.
``min_width_ft`` enforces a floor on sampled widths and
``min_network_length_ft`` drops disconnected stubs.
