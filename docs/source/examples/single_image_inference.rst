Single-image inference
======================

Not every question needs a pipeline. You can call any model directly on a
single image, which is handy for trying out a prompt before you run it over
a whole inventory.

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

If you plan to turn the answer into asset attributes later, ask for
structured output with ``json_mode``:

.. code-block:: python

   output = model.run_inference(
       image, 'Return JSON with keys hazard, severity (0-4) and evidence.',
       generation=GenerationConfig(json_mode=True, temperature=0.0),
   )

``examples/label_mapping.py`` shows another way to use a model directly. It
asks the model, through :class:`~rapidtools.processing.MapillaryLabelMapper`,
to translate plain-English object names into Mapillary's label vocabulary.
