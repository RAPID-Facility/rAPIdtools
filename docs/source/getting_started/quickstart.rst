Quickstart
==========

This page grades every building in a drone orthomosaic of Altadena, CA,
captured after the 2025 Eaton Fire, and writes the grades to GeoJSON. It runs
fully offline with a local Gemma 4 model, and needs a GPU with a few
gigabytes of memory.

The whole workflow
------------------

.. code-block:: python

   from pathlib import Path
   import rapidtools as rt

   rt.configure_logging()

   # 1. Sample data: an orthomosaic, building footprints and a prompt file.
   raster, footprints, prompt = rt.download_dataset(
       ['eaton_patch2', 'altadena_sample_buildings', 'aerial_chs_prompts']
   )

   # 2. Load the footprints into a collection of assets.
   buildings = rt.PhysicalAssetCollection.from_geojson(footprints)

   # 3. Choose a model. Swap 'gemma4' for 'gemini', 'claude', 'openai', ...
   model = rt.models.load('gemma4', model_id='google/gemma-4-E2B-it')

   # 4. Build the pipeline: crop each footprint, then ask the model about it.
   pipeline = rt.Pipeline([
       rt.AerialImageryExtractor(
           raster,
           save_directory=Path('output/crops'),
           buffer_m=5,
           overlay_asset_outline=True,
       ),
       rt.AssetAnalyzer(model, prompt=prompt, batch_size=8),
   ])
   buildings = pipeline.run(buildings)

   # 5. Export. Model answers are attributes prefixed with the provider name.
   buildings.filter_empty().to_geojson(
       'output/buildings_graded.geojson', ignore_properties=['image_assets']
   )

What happened
-------------

1. :class:`~rapidtools.processing.AerialImageryExtractor` read the raster
   once, cropped a square patch around each footprint with a 5 m margin,
   drew the footprint outline on the patch so the model knows which building
   to look at, and attached the JPEG to the asset as an
   :class:`~rapidtools.core.ImageAsset`.
2. :class:`~rapidtools.processing.AssetAnalyzer` sent each asset's crops and
   the prompt to the model, eight assets per batch, and stored the answer on
   the asset. The prompt file asks for a JSON object, so each key becomes an
   attribute such as ``gemma4_chs_level``.
3. :meth:`~rapidtools.core.PhysicalAssetCollection.to_geojson` wrote the
   footprints with their new attributes; ``ignore_properties`` keeps the
   image paths out of the file.

Using a hosted model instead
----------------------------

Only step 3 changes. Hosted models run several requests in parallel and back
off automatically when the provider rate-limits them:

.. code-block:: python

   model = rt.models.load('gemini', api_key='AIza...', model_id='gemini-3.8-flash')
   analyzer = rt.AssetAnalyzer(model, prompt=prompt, max_workers=4)

Reading the results
-------------------

.. code-block:: python

   >>> asset = buildings.get('bldg_042')
   >>> asset.attributes['gemma4_chs_level']
   '3'
   >>> asset.image_assets[0].path
   PosixPath('/.../output/crops/aerial_3418715171_-11814335588.jpg')
   >>> buildings.to_dataframe().groupby('gemma4_chs_level').size()

Next steps
----------

- :doc:`concepts` explains assets, collections, images and pipeline stages.
- :doc:`../user_guide/aerial_imagery` covers crop sizing, edge handling and
  outline styles.
- :doc:`../user_guide/vision_language_models` lists every supported model
  and the generation options.
- :doc:`../gui` runs the same workflow from a browser without code.
