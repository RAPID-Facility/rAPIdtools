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
The contract shared by every :class:`~rapidtools.processing.Pipeline` step.

A step is any callable that maps a
:class:`~rapidtools.core.PhysicalAssetCollection` to a collection and
declares which :class:`Stage` of the workflow it belongs to. The pipeline
orders steps by stage, so users can add components in any order.

Example:
    >>> from rapidtools.processing.step import PipelineStep, Stage, stage_of
    >>> class Deduplicate:
    ...     stage = Stage.REGULARIZE
    ...     def __call__(self, collection):
    ...         return collection
    >>> isinstance(Deduplicate(), PipelineStep), stage_of(Deduplicate())
    (True, <Stage.REGULARIZE: 30>)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import IntEnum
from typing import Protocol, runtime_checkable

from rapidtools.core import PhysicalAssetCollection

logger = logging.getLogger(__name__)


class Stage(IntEnum):
    """
    Execution order of pipeline steps (lower runs first).

    Attributes:
        DETECT: Discover assets in imagery (e.g. SAM 3 feature extraction).
        EXTRACT_IMAGERY: Attach aerial or street-level images to assets.
        REGULARIZE: Clean and regularize asset geometries.
        SEGMENT: Segment the attached images (masks, boxes).
        ANALYZE: Run vision-language models over the images.
        EXPORT: Reporters and exporters.
        CUSTOM: Steps without a declared stage; run last.

    Example:
        >>> Stage.DETECT < Stage.ANALYZE < Stage.CUSTOM
        True
    """

    DETECT = 10
    EXTRACT_IMAGERY = 20
    REGULARIZE = 30
    SEGMENT = 40
    ANALYZE = 50
    EXPORT = 60
    CUSTOM = 99


@runtime_checkable
class PipelineStep(Protocol):
    """
    Structural type of a pipeline step.

    Any object with a ``stage`` attribute and a ``__call__`` taking and
    returning a collection satisfies it; no inheritance is required.

    Example:
        >>> from rapidtools.processing import AerialImageryExtractor
        >>> issubclass(AerialImageryExtractor, PipelineStep)
        True
    """

    stage: Stage

    def __call__(
        self, asset_collection: PhysicalAssetCollection
    ) -> PhysicalAssetCollection:
        """Transform the collection and return it (or a new one)."""
        ...  # pragma: no cover - protocol body


# Legacy class-name keywords used before steps declared their stage.
_NAME_KEYWORDS: tuple[tuple[str, Stage], ...] = (
    ('extractor', Stage.EXTRACT_IMAGERY),
    ('segmenter', Stage.SEGMENT),
    ('regularizer', Stage.REGULARIZE),
    ('analyzer', Stage.ANALYZE),
    ('predictor', Stage.ANALYZE),
    ('classifier', Stage.ANALYZE),
    ('reporter', Stage.EXPORT),
    ('exporter', Stage.EXPORT),
)


def stage_of(step: Callable, warn: bool = True) -> Stage:
    """
    Return the :class:`Stage` of a step.

    Steps that declare a ``stage`` attribute are trusted. Otherwise the
    class name is matched against legacy keywords (``Extractor``,
    ``Analyzer``, ...) with a warning, and unknown steps run last.

    Args:
        step: The pipeline step (any callable).
        warn: Log a warning when the legacy name-based fallback is used.

    Returns:
        Stage: The resolved stage.

    Example:
        >>> class LegacyPredictor:
        ...     def __call__(self, c):
        ...         return c
        >>> stage_of(LegacyPredictor(), warn=False)
        <Stage.ANALYZE: 50>
    """
    declared = getattr(step, 'stage', None)
    if isinstance(declared, Stage):
        return declared
    if declared is not None:
        try:
            return Stage(int(declared))
        except ValueError:
            pass
    class_name = type(step).__name__.lower()
    for keyword, stage in _NAME_KEYWORDS:
        if keyword in class_name:
            if warn:
                logger.warning(
                    f'{type(step).__name__} does not declare a pipeline stage; '
                    f'inferred {stage.name} from its class name. Add a '
                    "'stage' attribute to silence this warning."
                )
            return stage
    return Stage.CUSTOM
