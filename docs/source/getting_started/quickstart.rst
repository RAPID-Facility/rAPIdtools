Quickstart
==========

In this walkthrough you grade every building in a drone orthomosaic of
Altadena, CA, flown after the 2025 Eaton Fire, and save the grades as
GeoJSON.

You need one of two things: a CUDA GPU with a few gigabytes of memory, to
run a local Gemma 4 model (offline once the weights are downloaded), or an
API key for a hosted model (see `Using a hosted model instead`_). The sample
data comes from the rAPIdtools registry, so there is nothing else to
prepare.

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

What each step did
------------------

1. :class:`~rapidtools.processing.AerialImageryExtractor` opened the raster
   once and cut a square patch around each footprint with a 5 m margin. It
   drew the footprint on the patch, so the model knows which building to
   look at, and attached the JPEG to the asset as an
   :class:`~rapidtools.core.ImageAsset`.
2. :class:`~rapidtools.processing.AssetAnalyzer` sent each asset's crop and
   the prompt to the model, eight assets per batch, and stored the answer on
   the asset. The prompt file asks for a JSON object, so every key in the
   answer becomes an attribute, such as ``gemma4_chs_level``.
3. :meth:`~rapidtools.core.PhysicalAssetCollection.to_geojson` wrote the
   footprints with their new attributes. ``include_images=False`` leaves the
   image records out of the file.

Using a hosted model instead
----------------------------

Only the analyzer changes. Hosted models send several requests at once and
slow down by themselves when the provider starts rejecting them. The key
comes from the provider's environment variable (``GOOGLE_API_KEY`` for
Gemini) unless you pass ``api_key=``, either as text or as the path of a
file that holds it:

.. code-block:: python

   analyzer = rt.AssetAnalyzer('gemini', prompt=prompt, max_workers=4)

   # or build the model yourself, for example to reuse it across steps
   model = rt.load('gemini', model_id='gemini-3.8-flash')
   analyzer = rt.AssetAnalyzer(model, prompt=prompt, max_workers=4)

Looking at the results
----------------------

.. code-block:: python

   >>> asset = buildings.get('bldg_042')
   >>> asset.attributes['gemma4_chs_level']
   '3'
   >>> asset.image_assets[0].path
   PosixPath('/.../output/crops/aerial_3418715171_-11814335588.jpg')
   >>> buildings.to_dataframe().groupby('gemma4_chs_level').size()

Where to go next
----------------

- :doc:`concepts` explains assets, collections, images and pipeline stages.
- :doc:`../user_guide/aerial_imagery` covers crop sizes, buildings on the
  edge of the imagery and outline styles.
- :doc:`../user_guide/vision_language_models` lists the supported models and
  their options.
- :doc:`../gui` runs the same workflow from a browser, without code.
