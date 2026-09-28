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
Core domain models for rapidtools.

This subpackage defines the spatial and media primitives that every other
part of the library builds on:

- :class:`BoundingBox` and :class:`PolygonRegion` describe geographic
  regions of interest.
- :class:`PhysicalAsset` and :class:`PhysicalAssetCollection` model real-world
  assets (buildings, roads, poles, ...) together with their attributes and
  linked imagery.
- :class:`ImageAsset` and :class:`ImageCollection` manage image files, their
  metadata, and segmentation masks.
- :class:`OperationCancelled` plus the ``is_cancelled`` / ``raise_if_cancelled``
  helpers provide cooperative cancellation for long-running components.

Example:
    >>> from rapidtools.core import BoundingBox, PhysicalAsset
    >>> from shapely.geometry import Point
    >>>
    >>> bbox = BoundingBox(-118.1, 34.1, -118.0, 34.2)
    >>> asset = PhysicalAsset(id='b1', geometry=Point(-118.05, 34.15))
    >>> bbox.contains(asset.geometry)
    True
"""

from .bounding_box import BoundingBox
from .cancellation import OperationCancelled, is_cancelled, raise_if_cancelled
from .image_asset import ImageAsset, ImageCollection
from .physical_asset import PhysicalAsset, PhysicalAssetCollection
from .polygon_region import PolygonRegion

__all__ = [
    'BoundingBox',
    'ImageAsset',
    'ImageCollection',
    'OperationCancelled',
    'PhysicalAsset',
    'PhysicalAssetCollection',
    'PolygonRegion',
    'is_cancelled',
    'raise_if_cancelled',
]
