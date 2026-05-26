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
# 05-25-2026

import json
import re
import logging
from rapidtools.models.base import BaseInferenceModel
from rapidtools.data_sources.mapillary_labels import MapillaryLabels

class MapillaryLabelMapper:
    """
    Uses an LLM (Gemma, Gemini, Claude, etc.) to intelligently map user-provided 
    natural language classes (e.g., 'cars', 'trees', 'stop signs') to the exact 
    Mapillary semantic segmentation strings.
    """

    def __init__(self, llm_model: BaseInferenceModel):
        """
        Args:
            llm_model: An initialized rapidtools inference model (e.g., Gemma4Inference).
        """
        self.llm_model = llm_model
        
        # Dynamically extract all valid label strings from the MapillaryLabels class
        self.valid_labels = [
            val for key, val in vars(MapillaryLabels).items() 
            if not key.startswith('__') and isinstance(val, str)
        ]

    def map_classes(self, user_classes: list[str]) -> list[str]:
        """
        Maps a list of user strings to official Mapillary labels.
        
        Args:
            user_classes: List of desired items (e.g., ['fire hydrant', 'cars'])
            
        Returns:
            list[str]: A deduplicated list of exact Mapillary label strings.
        """
        if not user_classes:
            return []

        prompt = (
            "You are a semantic mapping assistant. I have a list of target objects "
            "and a list of official Mapillary segmentation labels.\n\n"
            f"Target Objects: {user_classes}\n\n"
            f"Available Mapillary Labels:\n{self.valid_labels}\n\n"
            "Task: Find all Mapillary labels that match or encompass the Target Objects. "
            "Return ONLY a JSON array of strings containing the exact Mapillary labels. "
            "Do not include markdown formatting or explanations. Just the JSON array."
        )

        try:
            # We don't need to pass an image, just the text prompt
            result = self.llm_model.run_inference(
                image_inputs=[], 
                prompt=prompt,
                max_tokens=512,
                temperature=0.1  # Low temperature for strict factual matching
            )
            
            if not result or not result.text:
                return []

            # Extract JSON array from the response
            json_match = re.search(r'\[.*?\]', result.text, re.DOTALL)
            if json_match:
                extracted_labels = json.loads(json_match.group(0))
            else:
                extracted_labels = json.loads(result.text)

            # Safety check: ensure the LLM didn't hallucinate labels that don't exist
            final_labels = [
                label for label in extracted_labels if label in self.valid_labels
            ]
            
            logging.info(f"Mapped {user_classes} -> {final_labels}")
            return list(set(final_labels))

        except Exception as e:
            logging.error(f"Failed to map labels using LLM: {e}")
            return []