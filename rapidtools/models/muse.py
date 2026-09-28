# Copyright (c) 2025 The University of Washington
#
# This file is part of rapidtools.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# rapidtools. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Barbaros Cetiner
#
# Last updated:
# 09-22-2026

"""
Meta Muse models: the hosted Muse Spark API and the open-weight Muse Glimmer.

* :class:`MuseSparkInference` calls Meta's Model API
  (``https://api.meta.ai/v1``), which is OpenAI-compatible. Keys are read from
  ``MODEL_API_KEY`` (Meta's documented name) or ``META_API_KEY``.
* :class:`MuseGlimmerInference` runs ``meta-models/Muse-Glimmer-30B`` locally
  through Hugging Face Transformers (>= 5.15). The 30B dense model fits a
  24 GB GPU with ``load_in_4bit=True``.

Example:
    >>> from rapidtools.models import MuseSparkInference, MuseGlimmerInference
    >>> spark = MuseSparkInference(api_key='...')
    >>> spark.run_inference('roof.jpg', 'Describe the roof damage.').text
    >>> glimmer = MuseGlimmerInference(load_in_4bit=True)
    >>> glimmer.run_inference('roof.jpg', 'Describe the roof damage.').text
"""

from __future__ import annotations

from .catalogs import (
    MUSE_GLIMMER_DEFAULT_MODEL,
    MUSE_GLIMMER_MODEL_CATALOG,
    MUSE_SPARK_DEFAULT_MODEL,
    MUSE_SPARK_MODEL_CATALOG,
    PROVIDERS,
)
from .hf_vision import HFVisionInference
from .openai_compat import BaseOpenAICompatibleInference

META_BASE_URL = 'https://api.meta.ai/v1'


class MuseSparkInference(BaseOpenAICompatibleInference):
    """
    Meta Model API client for the Muse Spark family.

    Uses the OpenAI-compatible ``/chat/completions`` endpoint with Bearer
    authentication. Images are sent inline as Base64 data URIs.

    Example:
        >>> from rapidtools.models import MuseSparkInference
        >>> MuseSparkInference.list_known_models()[0]
        'muse-spark-1.3'
        >>> model = MuseSparkInference(api_key='...', model_id='muse-spark-1.3')
        >>> out = model.run_inference(['a.jpg', 'b.jpg'], 'Which house is damaged?')
        >>> print(out.text)
    """

    INFO = PROVIDERS['muse_spark']

    PROVIDER_NAME = 'Meta Model API'
    BASE_URL = META_BASE_URL
    API_KEY_ENV = ('MODEL_API_KEY', 'META_API_KEY')
    DEFAULT_MODEL = MUSE_SPARK_DEFAULT_MODEL
    MODEL_CATALOG = MUSE_SPARK_MODEL_CATALOG
    MAX_TOKENS_FIELD = 'max_tokens'
    IMAGE_DETAIL = None


class MuseGlimmerInference(HFVisionInference):
    """
    Local inference with Meta's open-weight Muse Glimmer vision-language model.

    A thin specialisation of :class:`~rapidtools.models.HFVisionInference`
    that defaults to ``meta-models/Muse-Glimmer-30B`` and checks that the
    installed Transformers release knows the ``muse_glimmer`` architecture.

    Example:
        >>> from rapidtools.models import MuseGlimmerInference
        >>> MuseGlimmerInference.list_available_models()
        ['meta-models/Muse-Glimmer-30B']
        >>> model = MuseGlimmerInference(load_in_4bit=True)
        >>> out = model.run_inference('roof.jpg', 'Rate the roof damage 0-5.')
        >>> out.text
        '3'
    """

    INFO = PROVIDERS['muse_glimmer']

    MODEL_CATALOG = MUSE_GLIMMER_MODEL_CATALOG
    DEFAULT_MODEL = MUSE_GLIMMER_DEFAULT_MODEL
    MIN_TRANSFORMERS_VERSION = '5.15.0'
    FAMILY_NAME = 'Muse Glimmer'
