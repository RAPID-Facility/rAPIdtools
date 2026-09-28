Single-image inference
======================

Not every question needs a pipeline. Any model wrapper can be called directly
on one image, which is handy for prototyping prompts before running them
over an inventory.

Source: ``examples/single_image_vlm_inference.py``.

.. code-block:: python

   from rapidtools import download_dataset
   from rapidtools.models import GenerationConfig, load

   [image] = download_dataset('synthetic_landslide_image')

   model = load('gemma4', model_id='google/gemma-4-E2B-it', temperature=0.4)
   output = model.run_inference(
       image_inputs=image,
       prompt='Describe the condition of this post-disaster scene and what went wrong.',
   )
   print(output.text)

Ask for structured output with ``json_mode`` when you plan to turn the
prompt into a pipeline attribute:

.. code-block:: python

   output = model.run_inference(
       image, 'Return JSON with keys hazard, severity (0-4) and evidence.',
       generation=GenerationConfig(json_mode=True, temperature=0.0),
   )

``examples/label_mapping.py`` shows another direct use of a model: asking it
to translate plain-English object names into Mapillary's label vocabulary
through :class:`~rapidtools.processing.MapillaryLabelMapper`.
