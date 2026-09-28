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
LLM-backed mapping of free-text class names to Mapillary segmentation labels.

Mapillary's semantic segmentation vocabulary uses hierarchical strings such
as ``'object--vehicle--car'`` or ``'construction--flat--road'``. Users
typically describe what they want in plain language ("cars", "the road").
:class:`MapillaryLabelMapper` bridges that gap by asking a rapidtools
inference model to select the official labels that match or encompass the
user's terms, then validating the answer against the known label set so
that hallucinated labels never leak into downstream processing.

Example:
    >>> from rapidtools.processing.label_mappers import MapillaryLabelMapper
    >>>
    >>> mapper = MapillaryLabelMapper(llm_model)  # doctest: +SKIP
    >>> mapper.map_classes(['cars', 'fire hydrants'])  # doctest: +SKIP
    ['object--vehicle--car', 'object--fire-hydrant']
"""

import json
import logging
import re

from rapidtools.data_sources.mapillary_labels import MapillaryLabels
from rapidtools.models.base import BaseInferenceModel

logger = logging.getLogger(__name__)


class MapillaryLabelMapper:
    """
    Map natural-language class names to exact Mapillary segmentation labels.

    An LLM (Gemma, Gemini, Claude, etc.) is prompted with the user's target
    objects and the full list of official Mapillary labels, and asked to
    return the matching labels as a JSON array. The response is parsed
    defensively and filtered against the official vocabulary.

    Attributes:
        llm_model (BaseInferenceModel):
            The inference model used to perform the mapping.
        valid_labels (list[str]):
            All official Mapillary label strings, extracted from
            :class:`rapidtools.data_sources.mapillary_labels.MapillaryLabels`.

    Example:
        >>> from rapidtools.models.base import ModelOutput
        >>>
        >>> class EchoModel:
        ...     def run_inference(self, image_inputs, prompt, **kwargs):
        ...         return ModelOutput(text='["object--vehicle--car"]')
        >>>
        >>> mapper = MapillaryLabelMapper(EchoModel())
        >>> mapper.map_classes(['cars'])
        ['object--vehicle--car']
    """

    def __init__(self, llm_model: BaseInferenceModel):
        """
        Initialize the mapper with an inference model.

        Args:
            llm_model (BaseInferenceModel):
                An initialized rapidtools inference model (e.g.,
                ``Gemma4Inference``) exposing
                ``run_inference(image_inputs, prompt, **kwargs)`` and
                returning a ``ModelOutput`` with a ``text`` attribute.

        Example:
            >>> mapper = MapillaryLabelMapper(llm_model)  # doctest: +SKIP
            >>> 'nature--sky' in mapper.valid_labels  # doctest: +SKIP
            True
        """
        self.llm_model = llm_model

        # Dynamically extract all valid label strings from the MapillaryLabels
        # class:
        self.valid_labels = [
            val
            for key, val in vars(MapillaryLabels).items()
            if not key.startswith('__') and isinstance(val, str)
        ]

    def map_classes(self, user_classes: list[str]) -> list[str]:
        """
        Map a list of user strings to official Mapillary labels.

        The model is asked to return only a JSON array of label strings. The
        first bracketed span in the response that parses as a JSON list is
        used; any element that is not an official Mapillary label is dropped.
        Failures (model errors, unparsable output) are logged and yield an
        empty list rather than raising.

        Args:
            user_classes (list[str]):
                Desired items in plain language (e.g.,
                ``['fire hydrant', 'cars']``).

        Returns:
            list[str]:
                A deduplicated list of exact Mapillary label strings in the
                order the model returned them. Empty if ``user_classes`` is
                empty, the model returns nothing, or the response cannot be
                parsed.

        Example:
            >>> from rapidtools.models.base import ModelOutput
            >>>
            >>> class EchoModel:
            ...     def run_inference(self, image_inputs, prompt, **kwargs):
            ...         return ModelOutput(
            ...             text='Labels: ["object--vehicle--car", "not-a-label"]'
            ...         )
            >>>
            >>> mapper = MapillaryLabelMapper(EchoModel())
            >>> mapper.map_classes(['cars'])
            ['object--vehicle--car']
            >>> mapper.map_classes([])
            []
        """
        if not user_classes:
            return []

        prompt = (
            'You are a semantic mapping assistant. I have a list of target '
            'objects and a list of official Mapillary segmentation labels.\n\n'
            f'Target Objects: {user_classes}\n\n'
            f'Available Mapillary Labels:\n{self.valid_labels}\n\n'
            'Task: Find all Mapillary labels that match or encompass the Target '
            'Objects. Return ONLY a JSON array of strings containing the exact '
            'Mapillary labels. Do not include markdown formatting or '
            'explanations. Just the JSON array.'
        )

        try:
            # We don't need to pass an image, just the text prompt:
            result = self.llm_model.run_inference(
                image_inputs=[],
                prompt=prompt,
                max_tokens=512,
                temperature=0.1,  # Low temperature for strict factual matching
            )

            if not result or not result.text:
                return []

            extracted_labels = self._extract_json_list(result.text)
            if extracted_labels is None:
                logger.error(
                    'Failed to map labels using LLM: response did not contain '
                    'a JSON array.'
                )
                return []

            # Safety check: ensure the LLM didn't hallucinate labels that don't
            # exist, while preserving the model's ordering:
            final_labels = list(
                dict.fromkeys(
                    label for label in extracted_labels if label in self.valid_labels
                )
            )

            logger.info(f'Mapped {user_classes} -> {final_labels}')
            return final_labels

        except Exception as e:
            logger.error(f'Failed to map labels using LLM: {e}')
            return []

    @staticmethod
    def _extract_json_list(text: str) -> list | None:
        """
        Extract the first JSON array found in a block of model output text.

        Args:
            text (str):
                Raw model response, possibly containing prose or markdown
                fences around the JSON array.

        Returns:
            list | None:
                The parsed list, or ``None`` if no bracketed span parses as a
                JSON list.

        Example:
            >>> MapillaryLabelMapper._extract_json_list('Sure: ["a", "b"] done')
            ['a', 'b']
            >>> MapillaryLabelMapper._extract_json_list('no json here') is None
            True
        """
        for match in re.finditer(r'\[.*?\]', text, re.DOTALL):
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list):
                return parsed

        # Fall back to parsing the whole response:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, list) else None
