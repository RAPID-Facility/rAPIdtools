Building footprints from a Bing basemap
=======================================

If you have no footprint inventory, you can make one. This example stitches
a basemap of the area, lets SAM 3 find the roofs, and then refines the rough
masks into clean footprints.

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

   # 3. Preliminary roofs from SAM 3.
   preliminary = SAM3OrthoFeatureExtractor(
       prompt='building roof',
       patch_size=50, unit='meters', overlap_ratio=0.25,
       batch_size=4, threshold=0.50, mask_threshold=0.40,
   )(bing_tiff)
   preliminary.to_geojson('eaton_patch1_buildings_preliminary.geojson')

   # 4. Crops around each preliminary footprint, then regularization.
   with_crops = AerialImageryExtractor(
       dataset=bing_tiff, save_directory='output/building_crops',
       buffer_asset='20 m', force_square_image=True,
   )(preliminary)
   final = BuildingRegularizer(batch_size=8)(with_crops)
   final.to_geojson('eaton_patch1_buildings_final.geojson')

Notes
-----

- To refine the footprints against the post-event drone imagery, point
  ``AerialImageryExtractor`` at the drone raster instead of the Bing TIFF
  in step 4.
- ``examples/vegetation_detection_from_aerial.py`` maps tree canopy with the
  same detector, a ``'vegetation'`` prompt and low thresholds.
