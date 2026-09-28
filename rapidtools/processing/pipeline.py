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
Ordered execution engine for ``rapidtools`` processing components.

The :class:`Pipeline` chains together callables that accept and return a
:class:`~rapidtools.core.PhysicalAssetCollection` (extractors, analyzers,
predictors, reporters, ...). Steps can be registered in any order; the
pipeline inspects each step's class name and enforces a logical execution
sequence (extract -> predict -> report) right before running.

Example:
    >>> from rapidtools.core import PhysicalAssetCollection
    >>> from rapidtools.processing import (
    ...     AerialImageryExtractor, Gemma4AssetAnalyzer, Pipeline
    ... )
    >>> collection = PhysicalAssetCollection.from_geojson('buildings.geojson')
    >>> pipeline = Pipeline([
    ...     Gemma4AssetAnalyzer(prompt='Describe the roof damage.'),
    ...     AerialImageryExtractor('ortho.tif', save_directory='crops'),
    ... ])
    >>> result = pipeline.run(collection)  # extractor runs first automatically
"""

import logging
import threading
from collections.abc import Callable

from rapidtools.core import PhysicalAssetCollection, raise_if_cancelled

from .step import stage_of

logger = logging.getLogger(__name__)


class Pipeline:
    """
    A smart sequence of processing steps applied to a PhysicalAssetCollection.

    This Pipeline is order-agnostic during construction. It automatically
    inspects the components you add and enforces a strict logical execution
    sequence before running. This prevents common errors, such as attempting
    to run AI predictions before image data has been extracted.

    The enforced execution order is:

        1. Extractors (e.g., ``AerialImageryExtractor``) - Gathers raw data
           and images.
        2. Predictors / Classifiers (e.g., ``DamagePredictor``) - Analyzes
           the gathered data.
        3. Reporters / Exporters (e.g., ``PDFReporter``) - Summarizes and
           exports the results.
        4. Custom / Unknown Steps - Any unrecognized steps default to running
           last.

    Note: The pipeline fully supports standalone execution. If you only
    provide a Predictor, it will simply run the Predictor without requiring
    an Extractor.

    Args:
        steps (list[Callable] | None, optional):
            An optional list of initialized processing steps. Each step must
            be a callable accepting a ``PhysicalAssetCollection`` and returning
            a ``PhysicalAssetCollection``. Defaults to an empty pipeline.
        cancel_event (threading.Event | None, optional):
            A cooperative cancellation flag. When set (typically from another
            thread), the pipeline raises
            :class:`~rapidtools.core.OperationCancelled` before starting the
            next step. The same event can be shared with the individual steps
            so that they also stop promptly. Defaults to ``None``.

    Example:
        Build a pipeline from an extractor and an analyzer, adding the steps
        in the "wrong" order. The pipeline reorders them before execution:

        >>> import threading
        >>> from rapidtools.core import PhysicalAssetCollection
        >>> from rapidtools.processing import (
        ...     AerialImageryExtractor, Gemma4AssetAnalyzer, Pipeline
        ... )
        >>>
        >>> stop = threading.Event()
        >>> analyzer = Gemma4AssetAnalyzer(
        ...     prompt='Is this building damaged? Answer yes or no.',
        ...     cancel_event=stop,
        ... )
        >>> extractor = AerialImageryExtractor(
        ...     dataset='data/eaton_ortho.tif',
        ...     save_directory='output/crops',
        ...     cancel_event=stop,
        ... )
        >>> pipeline = Pipeline(cancel_event=stop)
        >>> pipeline.add_step(analyzer).add_step(extractor)
        >>> collection = PhysicalAssetCollection.from_geojson('bldgs.geojson')
        >>> processed = pipeline.run(collection)
    """

    def __init__(
        self,
        steps: list[Callable] | None = None,
        cancel_event: threading.Event | None = None,
    ):
        """
        Initialize the pipeline.

        Args:
            steps (list[Callable] | None, optional):
                An optional list of initialized processing steps.
            cancel_event (threading.Event | None, optional):
                Cooperative cancellation flag checked before each step.
        """
        self.cancel_event = cancel_event
        self.steps = steps or []

    def add_step(self, step: Callable) -> 'Pipeline':
        """
        Add a new processing step to the pipeline.

        Args:
            step (Callable):
                A callable object (like an Extractor or Predictor instance)
                that accepts and returns a ``PhysicalAssetCollection``.

        Returns:
            Pipeline:
                The Pipeline instance itself, allowing for method chaining.

        Example:
            >>> from rapidtools.processing import Pipeline
            >>> pipeline = Pipeline()
            >>> pipeline.add_step(my_extractor).add_step(my_analyzer)
            >>> len(pipeline.steps)
            2
        """
        self.steps.append(step)
        return self

    def _sort_steps(self) -> None:
        """
        Order the steps by their :class:`~rapidtools.processing.step.Stage`.

        Steps that declare a ``stage`` attribute are ordered by it; legacy
        components are ranked by class name with a warning (see
        :func:`~rapidtools.processing.step.stage_of`). The sort is stable, so
        steps in the same stage keep their insertion order.
        """
        self.steps.sort(key=stage_of)

    def run(self, asset_collection: PhysicalAssetCollection) -> PhysicalAssetCollection:
        """
        Execute all steps in the pipeline in the correct logical sequence.

        Args:
            asset_collection (PhysicalAssetCollection):
                The collection of assets to process.

        Returns:
            PhysicalAssetCollection:
                The fully processed collection. If the pipeline has no steps,
                a warning is logged and the input is returned unchanged.

        Raises:
            OperationCancelled:
                If ``cancel_event`` is set before any step starts.

        Example:
            >>> from rapidtools.processing import Pipeline
            >>> pipeline = Pipeline([my_extractor, my_analyzer])
            >>> processed = pipeline.run(my_collection)
        """
        if not self.steps:
            logger.warning('Pipeline is empty. No steps were executed.')
            return asset_collection

        # 1. Automatically sort the components before running!
        self._sort_steps()

        logger.info(f'Starting pipeline with {len(self.steps)} steps...')

        # 2. Run the components
        for i, step in enumerate(self.steps, start=1):
            step_name = getattr(step, '__class__', type(step)).__name__
            logger.info(f'--- Running step {i}/{len(self.steps)}: {step_name} ---')

            # Stop promptly if the caller asked us to cancel:
            raise_if_cancelled(self.cancel_event, f'pipeline step {step_name}')

            # Pass the collection through the current step
            asset_collection = step(asset_collection)

        logger.info('Pipeline execution successfully completed.')
        return asset_collection
