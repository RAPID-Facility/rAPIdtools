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
# 01-28-2026

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any, Self

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

if TYPE_CHECKING:
    from .bounding_box import BoundingBox


class Region(ABC):
    """
    An abstract base class representing a geographical region.

    This class acts as a wrapper around a shapely geometry object, providing a
    common interface for all region types (e.g., BoundingBox, PolygonRegion).

    Implementation Contract:
        Concrete subclasses are responsible for:
        1. Initializing the ``_geom`` attribute with a valid Shapely
           ``Polygon`` or ``MultiPolygon`` in their ``__init__`` method.
        2. Implementing the abstract methods: ``get_bounding_box``,
           ``buffer``, and ``from_geometry``.
    """

    # All subclasses will have an internal shapely object for geometric
    # calculations:
    _geom: Polygon

    def __eq__(self, other: object) -> bool:
        """
        Check if two Regions are geometrically equivalent.

        This uses Shapely's ``equals`` predicate, which returns ``True`` if the
        shapes cover the exact same set of points, even if defined differently.
        """
        if self is other:
            return True

        if not isinstance(other, Region):
            return NotImplemented
        return self._geom.equals(other.geometry)

    def __hash__(self) -> int:
        """
        Create a hash based on the geometry's WKT representation.

        This allows Regions to be used in sets and as dictionary keys.
        """
        return hash(self._geom.wkb)

    def __repr__(self) -> str:
        """
        Return a string representation suitable for debugging.

        This includes the class name and a preview of the geometry in WKT
        (Well-Known Text) format. Long WKT strings are truncated to avoid
        cluttering logs.
        """
        wkt = self.wkt

        # Truncate long geometries for cleaner logs:
        if len(wkt) > 55:
            wkt = f'{wkt[:52]}...'
        return f"{self.__class__.__name__}(wkt='{wkt}')"

    @property
    def __geo_interface__(self) -> dict[str, Any]:
        """
        Return a GeoJSON-compatible dictionary representation.

        This implements the Python Geo Interface protocol, allowing this
        object to be passed directly to libraries such as
        ``shapely.geometry.shape``, ``fiona``, and ``geopandas``.

        Returns:
            dict[str, Any]:
                A dictionary containing ``type`` and ``coordinates`` keys
                following the GeoJSON specification.
        """
        return self._geom.__geo_interface__

    @property
    def geometry(self) -> Polygon:
        """
        Return the underlying Shapely Polygon object.

        This property provides direct access to the raw geometric object,
        allowing usage of the full Shapely API (e.g., buffering,
        simplification, relate) that is not directly wrapped by this class.

        Returns:
            Polygon: The Shapely Polygon defining the region.
        """
        return self._geom

    @property
    def height(self) -> float:
        """
        Return the axis-aligned height (north-south extent) of the region.

        Returns:
            float: The scalar height in the units of the coordinate system.
        """
        _, min_y, _, max_y = self.bounds
        return max_y - min_y

    @property
    def is_empty(self) -> bool:
        """
        Check if the region has no points (the geometric set is empty).

        Note:
            This is different from the object being ``None``. An empty geometry
            is a valid object that represents "nothingness" (often the result
            of an intersection between two disjoint shapes).

        Returns:
            bool: ``True`` if the geometry contains zero points.
        """
        return self.geometry.is_empty

    @property
    def is_valid(self) -> bool:
        """
        Check if the geometry is topologically valid.

        Invalid geometries (such as polygons that intersect themselves like a
        "bow-tie") can cause errors or undefined behavior in spatial operations
        like intersection or union.

        Returns:
            bool: ``True`` if valid, ``False`` otherwise.
        """
        return self.geometry.is_valid

    @property
    def width(self) -> float:
        """
        Return the axis-aligned width (east-west extent) of the region.

        Returns:
            float: The scalar width in the units of the coordinate system.
        """
        min_x, _, max_x, _ = self.bounds
        return max_x - min_x

    @property
    def wkt(self) -> str:
        """
        Return the Well-Known Text (WKT) representation of the geometry.

        Returns:
            str: The WKT string (e.g., 'POLYGON ((0 0, 1 0, 1 1, 0 0))').
        """
        return self._geom.wkt

    @cached_property
    def area(self) -> float:
        """
        Return the raw planar area of the geometry.

        Returns:
            float: The area in the squared units of the coordinate system.

        Note:
            - If coordinates are in meters (e.g., UTM), result is sq. meters.
            - If coordinates are in degrees (e.g., WGS84), result is sq.
              degrees. Square degrees are generally not useful for physical
              size estimates.
        """
        return self._geom.area

    @cached_property
    def bounds(self) -> tuple[float, float, float, float]:
        """
        Return the bounds as a ``(minx, miny, maxx, maxy)`` tuple.

        This value is cached to improve performance during repeated spatial
        filtering operations.

        Returns:
            tuple[float, float, float, float]:
                A tuple containing the (min_x, min_y, max_x, max_y)
                coordinates of the geometry's extent.
        """
        return self._geom.bounds

    @cached_property
    def centroid(self) -> tuple[float, float]:
        """
        Return the geometric center (centroid) as an ``(x, y)`` tuple.

        Returns:
            tuple[float, float]:
                The longitude (x) and latitude (y) of the center.
        """
        centroid = self._geom.centroid
        return (centroid.x, centroid.y)

    def contains(self, other: BaseGeometry | Region) -> bool:
        """
        Check if the region completely encloses another geometry.

        This is a wrapper around the Shapely ``contains`` predicate.

        Args:
            other (BaseGeometry | Region):
                The region to test for containment.

        Returns:
            ``True`` if no points of ``other`` lie in the exterior of this
            region and at least one point of the interior of ``other`` lies in
            the interior of this region.
        """
        if isinstance(other, Region):
            return self.geometry.contains(other.geometry)
        return self.geometry.contains(other)

    def distance(self, other: BaseGeometry | Region) -> float:
        """
        Calculate the minimum Euclidean distance to another region or geometry.

        Args:
            other (BaseGeometry | Region):
                The region to calculate the distance to.

        Returns:
            float:
                Distance in the units of the coordinate system (e.g.,
                degrees or meters). Returns 0.0 if the regions intersect.
        """
        if isinstance(other, Region):
            return self.geometry.distance(other.geometry)
        return self.geometry.distance(other)

    def intersects(self, other: BaseGeometry | Region) -> bool:
        """
        Determine if the region spatially intersects with another geometry.

        This is a wrapper around the Shapely ``intersects`` predicate.

        Args:
            other (BaseGeometry | Region):
                The region to test for intersection with.

        Returns:
            ``True`` if the boundary or interior of this region shares any
            points with ``other``; ``False`` otherwise.
        """
        if isinstance(other, Region):
            return self.geometry.intersects(other.geometry)
        return self.geometry.intersects(other)

    @abstractmethod
    def buffer(self, distance: float) -> Self:
        """
        Create a new Region expanded (or shrunk) by a constant distance.

        If the buffer distance is positive, the region expands. If negative,
        it erodes (shrinks).

        Args:
            distance (float):
                The distance to buffer in the units of the coordinate
                system (e.g., degrees for WGS84, meters for UTM).

        Returns:
            Self:
                A new concrete Region instance representing the buffered area.
        """

    @classmethod
    @abstractmethod
    def from_geometry(cls, geometry: BaseGeometry) -> Self:
        """
        Create a new Region instance from a Shapely BaseGeometry.

        Args:
            geometry (BaseGeometry):
                Any Shapely geometry (Polygon, Point, LineString, etc.) used
                to construct the region.

        Returns:
            Self: An instance of the concrete Region subclass.
        """

    @abstractmethod
    def get_bounding_box(self) -> BoundingBox:
        """
        Return the smallest axis-aligned BoundingBox that encloses this region.

        Returns:
            BoundingBox:
                A new BoundingBox instance derived from the geometry's extrema.
        """
