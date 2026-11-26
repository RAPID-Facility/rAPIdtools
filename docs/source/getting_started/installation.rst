Installation
============

rAPIdtools runs on Python 3.11 or newer, on Linux, macOS and Windows.

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

Importing the package is quick. PyTorch, Transformers and the model wrappers
load only when you first use a model, so a script that just reads GeoJSON
never waits for them.

.. code-block:: python

   import rapidtools as rt
   print(rt.PhysicalAssetCollection)

To check that everything is in place, list the model providers:

.. code-block:: python

   >>> from rapidtools.models import list_providers
   >>> list_providers(kind='api')
   ['gemini', 'claude', 'openai', 'muse_spark', 'qwen']
   >>> list_providers(kind='local')
   ['gemma4', 'llama', 'muse_glimmer', 'qwen_vl', 'hf']
   >>> list_providers(kind='segmentation')
   ['sam3']

Hosted models
-------------

Each hosted provider needs its own API key. Pass it as ``api_key=`` (the key
itself or the path of a file that holds it), or set one of the environment
variables below and leave the argument out.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Provider
     - Registry key
     - Environment variable
   * - Google Gemini
     - ``'gemini'``
     - ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY``
   * - Anthropic Claude
     - ``'claude'``
     - ``ANTHROPIC_API_KEY``
   * - OpenAI
     - ``'openai'``
     - ``OPENAI_API_KEY``
   * - Meta Muse Spark
     - ``'muse_spark'``
     - ``MODEL_API_KEY`` or ``META_API_KEY``
   * - Alibaba Qwen (Model Studio)
     - ``'qwen'``
     - ``DASHSCOPE_API_KEY`` or ``QWEN_API_KEY``

Local models
------------

The local vision-language models (Gemma 4, Llama Vision, Muse Glimmer, Qwen
and any Hugging Face checkpoint) and SAM 3 run on your own GPU. The smallest
default, ``google/gemma-4-E2B-it``, runs on a consumer card with a few
gigabytes of memory. A 24 GB card runs every default checkpoint, although
the larger ones need ``load_in_4bit=True``. That option is off by default, and the
Gemma 4 wrapper does not offer it.

Gemma, Llama and SAM 3 sit in gated Hugging Face repositories, so you need a
Hugging Face account that has accepted their licenses. Log in once and the
token is cached:

.. code-block:: python

   import rapidtools as rt
   rt.login()            # uses a cached token if there is one
   rt.login('hf_...')    # or pass a token explicitly

You can skip this step. The local wrappers call
:func:`~rapidtools.auth.ensure_huggingface_login` themselves before they
download weights.

Imagery sources
---------------

Google Street View and the Bing and Google aerial basemaps need no key.
Mapillary needs a free access token from `mapillary.com
<https://www.mapillary.com/developer>`_. Pass it as ``access_token`` to the
Mapillary components, or set ``MAPILLARY_ACCESS_TOKEN``.

Sample data
-----------

The examples in this documentation fetch their data with
:func:`~rapidtools.datasets.download_dataset`. It downloads files from the
rAPIdtools registry into a folder of your choice and skips any that are
already there:

.. code-block:: python

   from rapidtools import download_dataset

   raster, footprints, prompt = download_dataset(
       ['eaton_patch2', 'altadena_sample_buildings', 'aerial_chs_prompts'],
       output_dir='data',
   )

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Name
     - What it is
   * - ``eaton_patch1``, ``eaton_patch2``
     - Drone orthomosaics of Altadena, CA, flown after the January 2025
       Eaton Fire
   * - ``altadena_sample_buildings``, ``eaton_patch1_bing_buildings``
     - Building footprints for those areas
   * - ``aerial_chs_prompts``, ``street_chs_prompts``,
       ``street_recovery_prompts``
     - Prompt files for damage grading and recovery status
   * - ``synthetic_landslide_image``
     - A single test image
   * - ``mapillary_token``, ``hf_token``
     - The credential files the examples use
