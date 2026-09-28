Vision-language models
======================

Every model in rAPIdtools, hosted or local, is built through one factory and
exposes the same ``run_inference`` call, so pipeline code never changes when
you switch providers.

Choosing a model
----------------

.. code-block:: python

   from rapidtools.models import load, list_providers, catalog

   list_providers()                  # every registry key
   catalog('gemini')                 # curated model IDs for a provider, no network
   model = load('gemini', api_key='AIza...', model_id='gemini-3.8-flash')
   model = load('gemma4', model_id='google/gemma-4-E2B-it')
   model = load('hf', model_id='Qwen/Qwen3.5-4B', load_in_4bit=True)

.. list-table::
   :header-rows: 1
   :widths: 24 12 20 44

   * - Provider or family
     - Key
     - Runs
     - Default model and notes
   * - Google Gemini
     - ``gemini``
     - API
     - ``gemini-3.8-flash``. Also Gemini 3.x Flash, Flash-Lite and Pro
       previews.
   * - Anthropic Claude
     - ``claude``
     - API
     - ``claude-opus-5``. Also Fable 5.1, Sonnet 5, Haiku 4.5 and earlier.
   * - OpenAI
     - ``openai``
     - API
     - ``gpt-5.5``. Also GPT-6, GPT-5.x, o3 and GPT-4.x models.
   * - Meta Muse Spark
     - ``muse_spark``
     - API
     - ``muse-spark-1.3``.
   * - Alibaba Qwen (Model Studio)
     - ``qwen``
     - API
     - ``qwen3.8-max``. Also Qwen 3.x Plus and Qwen-VL models.
   * - Google Gemma 4
     - ``gemma4``
     - local
     - ``google/gemma-4-E2B-it``, up to the 31B checkpoint.
   * - Meta Llama Vision
     - ``llama``
     - local
     - ``meta-llama/Llama-3.2-11B-Vision-Instruct`` and Llama 4 (gated).
   * - Meta Muse Glimmer
     - ``muse_glimmer``
     - local
     - ``meta-models/Muse-Glimmer-30B``; 4-bit fits a 24 GB GPU.
   * - Alibaba Qwen VL
     - ``qwen_vl``
     - local
     - ``Qwen/Qwen3.5-4B``; Qwen 3.x, Qwen3-VL and Qwen2.5-VL sizes.
   * - Any Hugging Face VLM
     - ``hf``
     - local
     - Any repo with a chat template (Gemma 3, LLaVA, InternVL, Pixtral,
       Granite Vision, ...).
   * - Meta SAM 3
     - ``sam3``
     - local
     - ``facebook/sam3``, used for segmentation rather than text.

Every wrapper offers ``list_known_models()`` for the curated catalogue and,
where the provider supports it, ``list_available_models(api_key)`` for a live
listing of what your key can reach.

Analyzing a collection
----------------------

:class:`~rapidtools.processing.AssetAnalyzer` sends each asset's images and
the prompt to the model and writes the answer to the asset's attributes.

.. code-block:: python

   from rapidtools import AssetAnalyzer
   from rapidtools.models import GenerationConfig, load

   analyzer = AssetAnalyzer(
       load('claude', api_key='sk-ant-...'),
       prompt='prompts/damage.txt',            # text or a path to a file
       max_images_per_asset=2,
       image_filter=lambda img: 'aerial' in img.id,
       generation=GenerationConfig(json_mode=True, temperature=0.0),
   )
   buildings = analyzer(buildings)

- **Prompts** can ask for free text or JSON. When the answer parses as a JSON
  object, each key becomes an attribute; otherwise the whole text is stored.
  Attributes are prefixed with the provider key (``claude_damage_level``) or
  with ``attribute_prefix`` if you set one.
- **Hosted models** run ``max_workers`` requests in parallel (five by
  default). A :class:`~rapidtools.processing.RateLimitPolicy` pauses every
  worker after a failure, doubling the pause on repeated failures, and then
  makes extra passes over the assets that failed, so a burst of rate-limit
  errors does not leave holes in the results. Hosted wrappers also take
  ``timeout=`` for slow responses.
- **Local models** run sequentially or in batches of ``batch_size`` assets,
  with tensor precision chosen for the hardware and GPU memory released
  between batches.
- **Generation options** in :class:`~rapidtools.models.GenerationConfig`
  (``temperature``, ``max_tokens``, ``json_mode``, ``system_instruction``)
  apply per call and map to the native flags of each provider.

The provider-specific classes such as ``GeminiAssetAnalyzer`` and
``Gemma4AssetAnalyzer`` still work but are deprecated in favour of
``AssetAnalyzer(load(...))``.

Local model options
-------------------

Local wrappers accept ``device`` (``'auto'``, ``'cuda'``, ``'cpu'``),
``temperature`` and ``max_tokens``; the Hugging Face and Qwen wrappers also
accept ``load_in_4bit`` to fit larger checkpoints into memory. Gated weights
download automatically once you are logged in to the Hugging Face Hub (see
:doc:`../getting_started/installation`).

.. code-block:: python

   model = load('qwen_vl', model_id='Qwen/Qwen3.5-9B', load_in_4bit=True, device='cuda')

Single images
-------------

Outside a pipeline, call the model directly. The result is a
:class:`~rapidtools.models.ModelOutput` with ``text`` and, for segmentation
models, ``masks`` and ``bounding_boxes``:

.. code-block:: python

   model = load('gemma4', model_id='google/gemma-4-E2B-it')
   output = model.run_inference('scene.jpg', 'Describe the damage in this scene.')
   print(output.text)
