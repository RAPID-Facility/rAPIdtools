Quickstart
==========

This page grades every building in a drone orthomosaic of Altadena, CA,
captured after the 2025 Eaton Fire, and writes the grades to GeoJSON.

**What you need:** either a CUDA GPU with a few gigabytes of memory (the
example runs a local Gemma 4 model, fully offline after the first download),
or an API key for a hosted model (see *Using a hosted model instead* below).
Nothing else: the sample data comes from the rapidtools registry.

The whole workflow
------------------

.. code-block:: python

   import rapidtools as rt

   rt.configure_logging()

   # 1. Sample data: an orthomosaic, building footprints and a prompt file.
   raster, footprints, prompt = rt.download_dataset(
       ['eaton_patch2', 'altadena_sample_buildings', 'aerial_chs_prompts']
   )

   # 2. Load the footprints into a collection of assets.
   buildings = rt.PhysicalAssetCollection.from_geojson(footprints)

   # 3. Build the pipeline: crop each footprint, then ask a model about it.
   #    Swap 'gemma4' for 'gemini', 'claude', 'openai', ...
   pipeline = rt.Pipeline([
       rt.AerialImageryExtractor(
           raster, save_directory='output/crops', buffer_m=5, overlay_asset_outline=True
       ),
       rt.AssetAnalyzer('gemma4', prompt=prompt, batch_size=8),
   ])
   buildings = pipeline.run(buildings)

   # 4. Export. Model answers are attributes prefixed with the provider name.
   buildings.to_geojson('output/buildings_graded.geojson', include_images=False)

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
   footprints with their new attributes; ``include_images=False`` keeps the
   image records out of the file.

Using a hosted model instead
----------------------------

Only the analyzer changes. Hosted models run several requests in parallel and
back off automatically when the provider rate-limits them. The key is read
from the provider's environment variable (``GOOGLE_API_KEY`` here) unless you
pass ``api_key=``, as text or as the path of a file holding it:

.. code-block:: python

   analyzer = rt.AssetAnalyzer('gemini', prompt=prompt, max_workers=4)

   # or build the model yourself, for example to reuse it across steps
   model = rt.load('gemini', model_id='gemini-3.8-flash')
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
