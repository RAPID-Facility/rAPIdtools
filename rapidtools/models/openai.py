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
OpenAI vision-language API wrapper.

Example:
    >>> from rapidtools.models import OpenAIInference
    >>> model = OpenAIInference(api_key='sk-...', model_id='gpt-5.5')
    >>> out = model.run_inference('roof.jpg', 'Is the roof damaged? Answer yes/no.')
    >>> out.text
    'yes'
"""

from __future__ import annotations

from .catalogs import (
    OPENAI_DEFAULT_MODEL,
    OPENAI_MODEL_CATALOG,
    PROVIDERS,
)
from .openai_compat import BaseOpenAICompatibleInference

OPENAI_BASE_URL = 'https://api.openai.com/v1'


class OpenAIInference(BaseOpenAICompatibleInference):
    """
    OpenAI Chat Completions client for image + text prompts.

    The key is read from ``api_key`` (string or key-file path) or the
    ``OPENAI_API_KEY`` environment variable. Reasoning models (GPT-5.x, GPT-6,
    o-series) do not accept ``temperature``; it is omitted automatically.

    Example:
        >>> from rapidtools.models import OpenAIInference
        >>> OpenAIInference.list_known_models()[:2]
        ['gpt-6-astra', 'gpt-6-sol']
        >>> model = OpenAIInference(api_key='sk-...')
        >>> for asset_id, status, out in model.run_batch(
        ...     [('b1', ['b1.jpg']), ('b2', ['b2.jpg'])], 'Rate the damage 0-5.'
        ... ):
        ...     print(asset_id, status, out.text if status == 'ok' else out)
    """

    INFO = PROVIDERS['openai']

    PROVIDER_NAME = 'OpenAI'
    BASE_URL = OPENAI_BASE_URL
    API_KEY_ENV = ('OPENAI_API_KEY',)
    DEFAULT_MODEL = OPENAI_DEFAULT_MODEL
    MODEL_CATALOG = OPENAI_MODEL_CATALOG
    MAX_TOKENS_FIELD = 'max_completion_tokens'
    NO_TEMPERATURE_PREFIXES = ('gpt-5', 'gpt-6', 'o1', 'o3', 'o4')
