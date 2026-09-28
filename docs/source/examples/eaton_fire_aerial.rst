Aerial damage assessment: Eaton Fire
====================================

The January 2025 Eaton Fire destroyed thousands of structures in Altadena,
California. The RAPID Facility flew the area in February 2025; this example
grades the buildings in one orthomosaic patch on the Cal Fire / CHS damage
scale with a local Gemma 4 model, entirely offline.

Source: ``examples/fire_damage_from_aerial.py``.

.. code-block:: python

   from pathlib import Path
   from rapidtools import (
       AerialImageryExtractor,
       AssetAnalyzer,
       PhysicalAssetCollection,
       Pipeline,
       download_dataset,
   )
   from rapidtools.models import load

   # 1. Sample data from the rAPIdtools registry.
   raster_path, footprint_path, prompt_path = download_dataset(
       ['eaton_patch2', 'altadena_sample_buildings', 'aerial_chs_prompts']
   )

   # 2. The building inventory.
   buildings = PhysicalAssetCollection.from_geojson(footprint_path)

   # 3. Crop each footprint and outline it for the model.
   extractor = AerialImageryExtractor(
       dataset=raster_path,
       save_directory=Path('eaton_fire_aerial_feb25/overlaid_imagery'),
       overlay_asset_outline=True,
       outline_shape='rotated_bbox',
       outline_buffer='2 m',
       outline_width='1%',
       image_prefix='eaton_trinity_25',
       keep_multiple_copies=True,
   )

   # 4. Grade every crop with the prompt file (a JSON schema for CHS levels).
   analyzer = AssetAnalyzer(
       load('gemma4', model_id='google/gemma-4-E2B-it'),
       prompt=prompt_path,
       batch_size=8,
   )

   # 5. Run and export.
   graded = Pipeline([extractor, analyzer]).run(buildings)
   graded = graded.filter_empty()
   graded.to_geojson(
       'eaton_footprints_CHS_with_gemma4.geojson', ignore_properties=['image_assets']
   )

Notes
-----

- ``keep_multiple_copies=True`` appends a counter when two assets share a
  centroid, which happens with stacked or duplicated footprints.
- The prompt file asks for a JSON object with a ``chs_level`` key, so the
  grade lands in ``gemma4_chs_level``. Open the GeoJSON in QGIS and colour
  by that attribute to see the damage map.
- Swap ``load('gemma4', ...)`` for ``load('gemini', api_key=...)`` to grade
  with a hosted model; nothing else changes. The script
  ``examples/fire_damage_from_aerial_gemini.py`` does exactly that.
- ``examples/chs_from_shapefile_example.py`` shows the same pipeline reading
  a Shapefile inventory and writing one back with
  :meth:`~rapidtools.core.PhysicalAssetCollection.to_shapefile`.
