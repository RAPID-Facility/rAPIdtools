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

import logging
from typing import Callable, List, Optional
from rapidtools.core import PhysicalAssetCollection

class Pipeline:
    """
    A smart sequence of processing steps applied to a PhysicalAssetCollection.
    
    This Pipeline is order-agnostic during construction. It automatically inspects 
    the components you add and enforces a strict logical execution sequence before 
    running. This prevents common errors, such as attempting to run AI predictions 
    before image data has been extracted.
    
    The enforced execution order is:
        
        1. Extractors (e.g., AerialImageryExtractor) - Gathers raw data and images.
        2. Predictors / Classifiers (e.g., DamagePredictor) - Analyzes the 
           gathered data.
        3. Reporters / Exporters (e.g., PDFReporter) - Summarizes and exports 
           the results.
        4. Custom / Unknown Steps - Any unrecognized steps default to running last.
        
    Note: The pipeline fully supports standalone execution. If you only provide 
    a Predictor, it will simply run the Predictor without requiring an Extractor.
    """

    def __init__(self, steps: Optional[List[Callable]] = None):
        """
        Initialize the pipeline.
        
        Args:
            steps: An optional list of initialized processing steps.
        """
        self.steps = steps or[]

    def add_step(self, step: Callable) -> 'Pipeline':
        """
        Add a new processing step to the pipeline.
        
        Args:
            step: A callable object (like an Extractor or Predictor instance).
                  
        Returns:
            The Pipeline instance itself, allowing for method chaining.
        """
        self.steps.append(step)
        return self

    def _sort_steps(self) -> None:
        """
        Internal method to enforce the logical execution order of the pipeline:
        1. Extractors (get the data)
        2. Predictors / Classifiers (analyze the data)
        3. Reporters / Exporters (format the outputs)
        """
        def get_priority(step: Callable) -> int:
            class_name = getattr(step, '__class__', type(step)).__name__.lower()
            
            if 'extractor' in class_name:
                return 1
            elif 'predictor' in class_name or 'classifier' in class_name:
                return 2
            elif 'reporter' in class_name or 'exporter' in class_name:
                return 3
            return 99  # Unknown components default to running last

        # Sort the steps list in-place based on their priority integer
        self.steps.sort(key=get_priority)

    def run(
            self, 
            asset_collection: PhysicalAssetCollection
        ) -> PhysicalAssetCollection:
        """
        Execute all steps in the pipeline in the correct logical sequence.
        
        Args:
            asset_collection: The collection of assets to process.
            
        Returns:
            The fully processed PhysicalAssetCollection.
        """
        if not self.steps:
            logging.warning('Pipeline is empty. No steps were executed.')
            return asset_collection

        # 1. Automatically sort the components before running!
        self._sort_steps()

        logging.info(f"Starting pipeline with {len(self.steps)} steps...")

        # 2. Run the components
        for i, step in enumerate(self.steps, start=1):
            step_name = getattr(step, '__class__', type(step)).__name__
            logging.info(f'--- Running step {i}/{len(self.steps)}: {step_name} ---')
            
            # Pass the collection through the current step
            asset_collection = step(asset_collection)

        logging.info('Pipeline execution successfully completed.')
        return asset_collection