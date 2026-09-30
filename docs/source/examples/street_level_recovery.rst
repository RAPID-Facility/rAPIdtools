Street-level damage and recovery
================================

Street-level imagery captured at two dates tells a longitudinal story: what
was destroyed, and what has been cleared or rebuilt since. This example
fetches Mapillary panoramas from February and September 2025 for the same
buildings, grades the initial damage, then re-evaluates only the damaged
buildings for recovery progress.

Source: ``examples/fire_recovery_analysis_from_street.py``.

.. code-block:: python

   from pathlib import Path
   from rapidtools import (
       AssetAnalyzer,
       MapillaryImageExtractor,
       PhysicalAssetCollection,
       download_dataset,
   )
   from rapidtools.models import load

   footprints, damage_prompt, recovery_prompt, token_file = download_dataset(
       ['eaton_patch1_bing_buildings', 'street_chs_prompts',
        'street_recovery_prompts', 'mapillary_token']
   )
   buildings = PhysicalAssetCollection.from_geojson(footprints)
   token = token_file.read_text().strip()

   # 1. Two surveys, two folders. Date windows select the survey.
   for folder, start, end in (
       ('output/street_feb', '2025-01-01', '2025-05-01'),
       ('output/street_sep', '2025-06-01', '2025-12-01'),
   ):
       buildings = MapillaryImageExtractor(
           access_token=token, save_directory=folder,
           start_date=start, end_date=end, cast_corner_rays=False,
       )(buildings)

   # 2. Initial damage from the February views only.
   model = load('gemma4', model_id='google/gemma-4-E2B-it')
   damage = AssetAnalyzer(
       model, prompt=Path(damage_prompt).read_text(),
       batch_size=4, max_images_per_asset=2,
       image_filter=lambda img: 'street_feb' in str(img.path),
   )
   buildings = damage(buildings)

   # 3. Keep the damaged buildings.
   damaged = buildings.filter_by_attribute('gemma4_chs_level', '0', operator='!=')

   # 4. Recovery: both surveys, more images per asset.
   recovery = AssetAnalyzer(
       model, prompt=Path(recovery_prompt).read_text(),
       batch_size=2, max_images_per_asset=4,
   )
   damaged = recovery(damaged)
   damaged.filter_empty().to_geojson(
       'damaged_buildings_recovery_status.geojson', ignore_properties=['image_assets']
   )

Notes
-----

- Both extractor runs attach images to the same assets; the analyzer's
  ``image_filter`` decides which survey each pass sees.
- The same model instance serves both passes, so the weights load once.
- :doc:`vehicle_detection_from_street` builds the inventory from the
  imagery itself instead of starting from footprints.
