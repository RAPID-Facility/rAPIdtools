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

The :class:`Pipeline` chains callables that accept and return a
:class:`~rapidtools.core.PhysicalAssetCollection`: feature extractors,
image extractors, verifiers, analyzers and regularizers. Steps can be
registered in any order; each declares a
:class:`~rapidtools.processing.Stage` and the pipeline runs them in stage
order (detect, extract imagery, regularize, segment, verify, analyze,
export).

Example:
    >>> import rapidtools as rt
    >>> buildings = rt.PhysicalAssetCollection.from_geojson('buildings.geojson')
    >>> pipeline = rt.Pipeline([
    ...     rt.AssetAnalyzer('gemma4', prompt='Describe the roof damage.'),
    ...     rt.AerialImageryExtractor('ortho.tif', save_directory='crops'),
    ... ])
    >>> result = pipeline.run(buildings)  # the extractor runs first
"""

import logging
import threading
from collections.abc import Callable

from rapidtools.core import PhysicalAssetCollection, raise_if_canceled

from .step import stage_of

logger = logging.getLogger(__name__)


class Pipeline:
    """
    A sequence of processing steps applied to a PhysicalAssetCollection.

    The pipeline is order-agnostic during construction. Every step declares
    the :class:`~rapidtools.processing.Stage` it belongs to, and the steps
    are sorted by stage right before running, so an analyzer added before
    the extractor that feeds it still runs after it:

        1. ``DETECT``: find assets (``SAM3OrthoFeatureExtractor``,
           ``MapillaryFeatureExtractor``).
        2. ``EXTRACT_IMAGERY``: gather images per asset
           (``AerialImageryExtractor``, ``DroneImageExtractor``, ...).
        3. ``REGULARIZE`` and ``SEGMENT``: clean geometries and masks.
        4. ``VERIFY``: drop false detections and duplicates.
        5. ``ANALYZE``: run a vision-language model (``AssetAnalyzer``).
        6. ``EXPORT`` and anything without a stage.

    Steps of the same stage keep the order they were added in. A pipeline
    with a single step simply runs it.

    Args:
        steps (list[Callable] | None, optional):
            An optional list of initialized processing steps. Each step must
            be a callable accepting a ``PhysicalAssetCollection`` and returning
            a ``PhysicalAssetCollection``. Defaults to an empty pipeline.
        cancel_event (threading.Event | None, optional):
            A cooperative cancellation flag. When set (typically from another
            thread), the pipeline raises
            :class:`~rapidtools.core.OperationCanceled` before starting the
            next step. The event is also handed to every step that has a
            ``cancel_event`` of its own left at ``None``, so the step in
            progress stops promptly too. Defaults to ``None``.

    Example:
        A detection pipeline needs no input collection, and one cancel event
        on the pipeline covers every step:

        >>> import threading
        >>> import rapidtools as rt
        >>> stop = threading.Event()
        >>> pipeline = rt.Pipeline(cancel_event=stop)
        >>> pipeline.add_step(rt.AssetAnalyzer('gemma4', prompt='Damaged? yes/no'))
        >>> pipeline.add_step(
        ...     rt.AerialImageryExtractor('data/ortho.tif', save_directory='crops')
        ... )
        >>> buildings = rt.PhysicalAssetCollection.from_geojson('bldgs.geojson')
        >>> processed = pipeline.run(buildings)
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

    def run(
        self, asset_collection: PhysicalAssetCollection | None = None
    ) -> PhysicalAssetCollection:
        """
        Execute all steps in the pipeline in the correct logical sequence.

        Args:
            asset_collection (PhysicalAssetCollection | None):
                The collection of assets to process. Leave it out for a
                pipeline that starts with a detection step, which builds the
                collection itself; an empty collection is used.

        Returns:
            PhysicalAssetCollection:
                The fully processed collection. If the pipeline has no steps,
                a warning is logged and the input is returned unchanged.

        Raises:
            OperationCanceled:
                If ``cancel_event`` is set before any step starts.
            TypeError:
                If a step returns something other than a collection.

        Example:
            >>> from rapidtools.processing import Pipeline
            >>> pipeline = Pipeline([my_extractor, my_analyzer])
            >>> processed = pipeline.run(my_collection)
            >>> detected = Pipeline([my_detector, my_cropper]).run()
        """
        if asset_collection is None:
            asset_collection = PhysicalAssetCollection()
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
            raise_if_canceled(self.cancel_event, f'pipeline step {step_name}')

            # One event on the pipeline is enough: steps that were given none
            # of their own share it, so the step in progress stops too.
            if (
                self.cancel_event is not None
                and getattr(step, 'cancel_event', self) is None
            ):
                try:
                    step.cancel_event = self.cancel_event  # type: ignore[attr-defined]
                except AttributeError:  # pragma: no cover - read-only step
                    pass

            # Pass the collection through the current step
            result = step(asset_collection)
            if not isinstance(result, PhysicalAssetCollection):
                raise TypeError(
                    f'{step_name} returned {type(result).__name__}, not a '
                    'PhysicalAssetCollection, so the next step has nothing to work '
                    'on. A step with several outputs must be told which one to '
                    "pass on (for example RoadwayRegularizer(output='polygons'))."
                )
            asset_collection = result

        logger.info('Pipeline execution successfully completed.')
        return asset_collection
