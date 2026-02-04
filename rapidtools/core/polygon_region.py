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
# 02-04-2026

from __future__ import annotations

from collections.abc import Iterable
from typing import Self

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from .bounding_box import BoundingBox
from .region import Region


class PolygonRegion(Region):
    """
    Represents a geographical region defined by a single polygon shell.

    This class wraps a Shapely Polygon and provides initialization from
    explicit coordinates. Use ``from_geometry`` to wrap existing Shapely objects.

    Example:
        Define a triangle using (x, y) coordinates:

        >>> from rapidtools.core import PolygonRegion
        >>>
        >>> coords = [(0, 0), (10, 0), (0, 10)]
        >>> region = PolygonRegion(coords)
        >>> region.area
        50.0
    """

    def __init__(self, shell: Iterable[tuple[float, float]]):
        """
        Initialize the PolygonRegion from a list of coordinates.

        Args:
            shell:
                A list (or iterable) of (x, y) coordinates defining the
                outer edge of the polygon.

        Raises:
            ValueError: If the shell has fewer than 3 vertices.

        Example:
            Create a generic square ``PolygonRegion``:

            >>> from rapidtools.core import PolygonRegion
            >>>
            >>> coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
            >>> region = PolygonRegion(coords)
            >>> region.is_valid
            True
        """
        # Convert to list to check length safely:
        coords = list(shell)
        if len(coords) < 3:
            raise ValueError('A polygon must have at least 3 vertices.')

        # Initialize the geometry from coordinates:
        self._geom = Polygon(shell=coords)

    @classmethod
    def from_geometry(cls, geometry: BaseGeometry) -> Self:
        """
        Create a PolygonRegion from a Shapely geometry.

        Args:
            geometry (BaseGeometry): A Shapely Polygon.

        Returns:
            PolygonRegion: The wrapped region.

        Raises:
            TypeError: If the geometry is not a Polygon.

        Example:
            >>> from shapely.geometry import Polygon
            >>> from rapidtools.core import PolygonRegion
            >>>
            >>> raw_poly = Polygon([(0, 0), (5, 0), (5, 5)])
            >>> region = PolygonRegion.from_geometry(raw_poly)
            >>> region.area
            12.5
        """
        if not isinstance(geometry, Polygon):
            raise TypeError(
                'PolygonRegion requires a Polygon, got ' f'{type(geometry).__name__}'
            )

        instance = cls.__new__(cls)
        instance._geom = geometry
        return instance

    def buffer(self, distance: float) -> PolygonRegion:
        """
        Create a new Region expanded (or shrunk) by a constant distance.

        Args:
            distance (float):
                The distance to buffer. Positive expands, negative erodes.

        Returns:
            PolygonRegion: A new PolygonRegion representing the result.

        Example:
            Initialize a ``PolygonRegion`` representing a 10x10 square:

            >>> from shapely.geometry import Polygon
            >>> from rapidtools.core import PolygonRegion
            >>>
            >>> region = PolygonRegion([(0, 0), (10, 0), (10, 10), (0, 10)])
            >>> region.area
            100.0

            Apply a positive buffer to expand the region:

            >>> expanded = region.buffer(1.0)
            >>> expanded.area > 100.0
            True

            Apply a negative buffer to erode (shrink) the region:

            >>> shrunk = region.buffer(-1.0)
            >>> shrunk.area < 100.0
            True
        """
        buffered_geom = self._geom.buffer(distance)

        # Handle empty results (e.g., shrinking a small polygon to nothing):
        if buffered_geom.is_empty:
            # Create an empty instance
            return PolygonRegion.from_geometry(Polygon())

        # If result is a Polygon, return a PolygonRegion:
        if isinstance(buffered_geom, Polygon):
            return PolygonRegion.from_geometry(buffered_geom)

        # Handle complex results (MultiPolygon, GeometryCollection, etc.):
        raise NotImplementedError(
            f'Buffering resulted in {type(buffered_geom).__name__}, which is '
            'not supported by PolygonRegion (requires single Polygon).'
        )

    def get_bounding_box(self) -> BoundingBox:
        """
        Calculate the smallest bounding box that encloses the polygon.

        Returns:
            BoundingBox:
                A new BoundingBox object representing the calculated bounds.

        Example:
            Initialize a triangular ``PolygonRegion`` and retrieve its
            enclosing ``BoundingBox``:

            >>> from rapidtools.core import PolygonRegion
            >>>
            >>> region = PolygonRegion([(0, 0), (2, 5), (4, 0)])
            >>> bbox = region.get_bounding_box()
            >>> bbox.bounds
            (0.0, 0.0, 4.0, 5.0)
        """
        min_x, min_y, max_x, max_y = self.bounds
        return BoundingBox(min_x, min_y, max_x, max_y)
