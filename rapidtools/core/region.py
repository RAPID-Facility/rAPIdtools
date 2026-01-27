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
# 01-27-2025

from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any

from shapely.geometry.base import BaseGeometry

if TYPE_CHECKING:
    from .bounding_box import BoundingBox


class Region(ABC):
    """
    An abstract base class representing a geographical region.

    This class acts as a wrapper around a shapely geometry object, providing a
    common interface for all region types (e.g., BoundingBox, Polygon).

    Implementation Contract:
        Concrete subclasses are responsible for initializing the ``_geom``
        attribute with a valid Shapely ``BaseGeometry`` in their ``__init__``
        method and implement the ``get_bounding_box`` method.
    """

    # All subclasses will have an internal shapely object for geometric
    # calculations:
    _geom: BaseGeometry

    def __repr__(self) -> str:
        """
        Return a string representation suitable for debugging.

        This includes the class name and a preview of the geometry in WKT
        (Well-Known Text) format. Long WKT strings are truncated to avoid
        cluttering logs.
        """
        wkt = self._geom.wkt
        preview = wkt if len(wkt) < 55 else f"{wkt[:52]}..."
        return f"{self.__class__.__name__}(wkt='{preview}')"

    @property
    def geometry(self) -> BaseGeometry:
        """
        Return the underlying Shapely geometry object.

        This property provides direct access to the raw geometric primitive
        used for spatial calculations.
        """
        return self._geom

    @property
    def __geo_interface__(self) -> dict[str, Any]:
        """
        Return a GeoJSON-compatible dictionary representation.

        This implements the Python Geo Interface protocol, allowing this
        object to be passed directly to libraries like
        ``shapely.geometry.shape``, ``fiona``, and ``geopandas``.
        """
        return self._geom.__geo_interface__

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

    @abstractmethod
    def get_bounding_box(self) -> "BoundingBox":
        """
        Return the smallest axis-aligned BoundingBox that encloses this region.

        Returns:
            BoundingBox: A new BoundingBox instance derived from the
            geometry's extrema.
        """
