Installation
============

rAPIdtools supports Python 3.11 and newer on Linux, macOS and Windows.

.. tab-set::

   .. tab-item:: pip

      .. code-block:: bash

         pip install rapidtools

   .. tab-item:: conda environment

      .. code-block:: bash

         conda create -n rapidtools python=3.11
         conda activate rapidtools
         pip install rapidtools

   .. tab-item:: From source

      .. code-block:: bash

         git clone https://github.com/RAPID-Facility/rAPIdtools.git
         cd rAPIdtools
         pip install -e ".[dev]"

Importing the package is fast and side-effect free. Model wrappers, PyTorch
and Transformers load the first time a model is used, so a script that only
reads GeoJSON never pays for them.

.. code-block:: python

   import rapidtools as rt
   print(rt.PhysicalAssetCollection)

Hosted model credentials
------------------------

Hosted models need an API key, passed as ``api_key=`` or read from the
environment variable listed below.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Provider
     - Registry key
     - Environment variable
   * - Google Gemini
     - ``'gemini'``
     - ``GOOGLE_API_KEY``
   * - Anthropic Claude
     - ``'claude'``
     - ``ANTHROPIC_API_KEY``
   * - OpenAI
     - ``'openai'``
     - ``OPENAI_API_KEY``
   * - Meta Muse Spark
     - ``'muse_spark'``
     - ``MODEL_API_KEY``
   * - Alibaba Qwen (Model Studio)
     - ``'qwen'``
     - ``DASHSCOPE_API_KEY``

Local models
------------

Local vision-language models (Gemma 4, Llama Vision, Muse Glimmer, Qwen and
any Hugging Face checkpoint) and SAM 3 run on your own GPU. A 24 GB card
handles every default checkpoint with the 4-bit loading that is on by
default; the smallest, ``google/gemma-4-E2B-it``, also runs on consumer GPUs
with far less memory.

Gated repositories such as Gemma and Llama require a Hugging Face account.
Log in once and the token is cached for future sessions:

.. code-block:: python

   import rapidtools as rt
   rt.login()            # uses a cached token if one exists
   rt.login('hf_...')    # or pass a token explicitly

Local wrappers call :func:`~rapidtools.auth.ensure_huggingface_login`
themselves before downloading weights, so this step is optional.

Street-level imagery
--------------------

Google Street View and the Bing and Google aerial basemaps are fetched
without any key. Mapillary needs a free access token from
`mapillary.com <https://www.mapillary.com/developer>`_, passed to
:class:`~rapidtools.processing.MapillaryImageExtractor` as ``access_token``.

Verify the installation
-----------------------

.. code-block:: python

   >>> import rapidtools as rt
   >>> from rapidtools.models import list_providers
   >>> list_providers(kind='api')
   ['gemini', 'claude', 'openai', 'muse_spark', 'qwen']
   >>> list_providers(kind='local')
   ['gemma4', 'llama', 'muse_glimmer', 'qwen_vl', 'hf', 'sam3']

Sample data
-----------

Every example in this documentation downloads what it needs through
:func:`~rapidtools.datasets.download_dataset`, which fetches files from the
rAPIdtools registry into a directory of your choice and skips files that are
already present:

.. code-block:: python

   from rapidtools import download_dataset

   raster, footprints, prompt = download_dataset(
       ['eaton_patch2', 'altadena_sample_buildings', 'aerial_chs_prompts'],
       output_dir='data',
   )

Available names: ``eaton_patch1``, ``eaton_patch2`` (drone orthomosaics of
Altadena, CA after the January 2025 Eaton Fire), ``altadena_sample_buildings``
and ``eaton_patch1_bing_buildings`` (footprints), ``aerial_chs_prompts``,
``street_chs_prompts`` and ``street_recovery_prompts`` (prompt files),
``synthetic_landslide_image``, and the ``mapillary_token`` and ``hf_token``
credential files used by the examples.
