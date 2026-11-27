Building footprints from a Bing basemap
=======================================

If you have no footprint inventory, you can make one. This example stitches
a basemap of the area, lets SAM 3 find the buildings, one footprint per
building, and then refines them into clean footprints with square walls.

Source: ``examples/building_detection_from_bing.py``.

.. code-block:: python

   from rapidtools import (
       AerialImageryExtractor,
       BingOrthomosaicExtractor,
       BoundingBox,
       BuildingRegularizer,
       SAM3OrthoFeatureExtractor,
       download_dataset,
   )

   # 1. The area of interest: the extent of a drone orthomosaic.
   [drone_raster] = download_dataset('eaton_patch1')
   region = BoundingBox.from_raster(drone_raster)

   # 2. A Bing basemap of the same area, as a GeoTIFF.
   bing_tiff = BingOrthomosaicExtractor(zoom_level=19, max_workers=10)(
       region=region, output_path='eaton_patch1_bing.tiff'
   )

   # 3. Footprints from SAM 3. A building prompt gets the building scan:
   #    tiles overlapping by half, the views of every building fused so
   #    neighbors stay apart, and straight, square walls.
   preliminary = SAM3OrthoFeatureExtractor(
       prompt='building', patch_size=50, unit='meters',
       batch_size=4, threshold=0.50, mask_threshold=0.50,
   )(bing_tiff)
   preliminary.to_geojson('eaton_patch1_buildings_preliminary.geojson')

   # 4. Refinement: crops around each footprint, segmented again.
   with_crops = AerialImageryExtractor(
       dataset=bing_tiff, save_directory='output/building_crops',
       buffer_asset='20 m', force_square_image=True,
   )(preliminary)
   final = BuildingRegularizer(batch_size=8)(with_crops)
   final.to_geojson('eaton_patch1_buildings_final.geojson')

Notes
-----

- The footprints of step 3 are already usable: one per building, with
  square walls. Step 4 segments every crop again, which takes a few minutes
  per hundred buildings on a GPU, and completes roofs and buildings the scan
  missed.
- To refine the footprints against the post-event drone imagery, point
  ``AerialImageryExtractor`` at the drone raster instead of the Bing TIFF
  in step 4.
- ``examples/vegetation_detection_from_aerial.py`` maps tree canopy with the
  same detector, a ``'vegetation'`` prompt and low thresholds.
