Vision-language models
======================

Every model in rAPIdtools, hosted or local, is created by the same factory
and answers the same ``run_inference`` call. Switching providers therefore
never changes the rest of your pipeline.

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
   :widths: 24 12 12 52

   * - Provider or family
     - Key
     - Runs
     - Default model and notes
   * - Google Gemini
     - ``gemini``
     - API
     - ``gemini-3.8-flash``. The catalog also has Gemini 4 (Neon, Carbon,
       Argon, Hydrogen) and the Gemini 3.x Flash, Flash-Lite and Pro models.
   * - Anthropic Claude
     - ``claude``
     - API
     - ``claude-opus-5``. Also Fable 5.2, Opus 5.5, Sonnet 5.5, Haiku 4.5 and
       earlier models.
   * - OpenAI
     - ``openai``
     - API
     - ``gpt-5.5``. Also GPT-6.2, GPT-6.1, GPT-6, the other GPT-5.x models, o3
       and GPT-4.x.
   * - Meta Muse Spark
     - ``muse_spark``
     - API
     - ``muse-spark-1.3``. Also ``muse-wonder-1.0``.
   * - Alibaba Qwen (Model Studio)
     - ``qwen``
     - API
     - ``qwen3.8-max``. Also Qwen 3.x Plus and the Qwen-VL models.
   * - Google Gemma 4
     - ``gemma4``
     - local
     - ``google/gemma-4-E2B-it``, and the larger checkpoints up to 31B.
   * - Meta Llama Vision
     - ``llama``
     - local
     - ``meta-llama/Llama-3.2-11B-Vision-Instruct``, and Llama 4. Gated
       download.
   * - Meta Muse Glimmer
     - ``muse_glimmer``
     - local
     - ``meta-models/Muse-Glimmer-30B``. Fits a 24 GB GPU in 4-bit.
   * - Alibaba Qwen VL
     - ``qwen_vl``
     - local
     - ``Qwen/Qwen3.5-4B``, plus the other Qwen 3.x, Qwen3-VL and Qwen2.5-VL
       sizes.
   * - Any Hugging Face VLM
     - ``hf``
     - local
     - ``zai-org/GLM-4.6V-Flash`` (10B; use ``load_in_4bit=True`` on GPUs
       with less than 24 GB), or any repository with a chat template: Gemma
       3, LLaVA, InternVL, Pixtral, Granite Vision and others. The catalog
       also lists GLM 5.3 Flash, Qwen3.8 Flash Next, Mistral Large 4 and
       Muse Spark Open, which need a multi-GPU server.
   * - Meta SAM 3
     - ``sam3``
     - local
     - ``facebook/sam3``. A segmentation model: it returns masks, not text.

Each wrapper has ``list_known_models()`` for its curated catalog. Where the
provider supports it, ``list_available_models(api_key)`` asks the provider
which models your key can actually use.

Analyzing a collection
----------------------

:class:`~rapidtools.processing.AssetAnalyzer` sends each asset's images and
your prompt to the model and writes the answer into the asset's attributes.

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

The prompt can ask for free text or for JSON. If the answer parses as a JSON
object, each key becomes its own attribute; otherwise the whole text is
stored. Attribute names start with the provider's prefix
(``claude_damage_level``; generic Hugging Face checkpoints use ``vlm_``), or
with ``attribute_prefix`` if you set one.

Hosted models get ``max_workers`` requests in flight at once (five by
default). When a request fails, a :class:`~rapidtools.processing.RateLimitPolicy`
pauses all the workers, doubling the pause if failures continue. Once the
pass is over, the analyzer goes back for the assets that failed, so a burst
of rate-limit errors does not leave gaps in the results. Hosted wrappers also
accept ``timeout=`` for providers that respond slowly.

Local models run one asset at a time or in batches of ``batch_size``. The
wrapper picks the numeric precision for your hardware and frees GPU memory
between batches.

:class:`~rapidtools.models.GenerationConfig` holds the per-call options
(``temperature``, ``max_tokens``, ``json_mode``, ``system_instruction``) and
translates them into each provider's own parameters.

Output limits
~~~~~~~~~~~~~

If you build a hosted model without ``max_tokens``, it uses the default for
that model ID. That is 2,048 tokens for most models and 32,768 for Gemini 4,
whose thinking tokens count against the limit. A value you pass always takes
precedence. To set your own default for a model, call
:func:`~rapidtools.models.set_model_defaults` before building it; a trailing
``*`` matches a whole family. :func:`~rapidtools.models.model_defaults`
shows what applies to a given ID.

.. code-block:: python

   from rapidtools.models import model_defaults, set_model_defaults

   set_model_defaults('claude-opus-5-5', max_tokens=8192)
   set_model_defaults('gpt-6.2-*', max_tokens=4096)
   model_defaults('gemini-4-neon')       # {'max_tokens': 32768}

The provider-specific analyzers such as ``GeminiAssetAnalyzer`` and
``Gemma4AssetAnalyzer`` still work, but they are deprecated. Use
``AssetAnalyzer(load(...))`` instead.

Local model options
-------------------

Local wrappers accept ``device`` (``'auto'``, ``'cuda'`` or ``'cpu'``),
``temperature`` and ``max_tokens``. The Hugging Face, Qwen, Llama and Muse
Glimmer wrappers also accept ``load_in_4bit`` (off by default), which lets
larger checkpoints fit in memory. If you pass an argument a wrapper does not
take, :func:`~rapidtools.models.load` tells you which ones it does. Gated
weights download automatically once you are logged in to the Hugging Face
Hub (see :doc:`../getting_started/installation`).

.. code-block:: python

   model = load('qwen_vl', model_id='Qwen/Qwen3.5-9B', load_in_4bit=True, device='cuda')

Some open models, GLM-4.6V among them, write out their reasoning in a
``<think>`` block before they answer. The wrappers return only the answer in
``text`` and keep the full output in ``raw_response``. The reasoning still
counts against ``max_tokens``, which is 512 by default for the Hugging Face,
Qwen and Muse Glimmer wrappers (2,048 for Gemma 4 and Llama). If the model
runs out of tokens before it reaches the answer, the output is marked as a
failure with a message telling you to raise the limit.

Single images
-------------

Outside a pipeline, call the model directly. You get back a
:class:`~rapidtools.models.ModelOutput` with ``text`` and, for segmentation
models, ``masks`` and ``bounding_boxes``:

.. code-block:: python

   model = load('gemma4', model_id='google/gemma-4-E2B-it')
   output = model.run_inference('scene.jpg', 'Describe the damage in this scene.')
   print(output.text)
