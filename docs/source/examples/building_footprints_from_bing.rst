Building footprints from a Bing basemap
=======================================

No footprint inventory? Stitch a basemap of the area, let SAM 3 find the
roofs, then refine the rough masks into clean footprints against the
high-resolution drone imagery.

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

- Point ``AerialImageryExtractor`` at the drone raster instead of the Bing
  TIFF in step 4 to regularize against post-event imagery.
- ``examples/vegetation_detection_from_aerial.py`` uses the same extractor
  with a ``'vegetation'`` prompt and low thresholds to map canopy.
