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
# 11-11-2025

from abc import ABC, abstractmethod
from shapely.geometry.base import BaseGeometry

# Forward declaration for the type hint to prevent circular imports.
# This tells Python: "Trust me, a class named 'BoundingBox' will exist."
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bounding_box import BoundingBox

class Region(ABC):
    """
    An abstract base class representing a geographical region.

    This class acts as a wrapper around a shapely geometry object, providing a
    common interface for all region types (e.g., BoundingBox, Polygon).
    """
    # All subclasses will have an internal shapely object for geometric calculations.
    _geom: BaseGeometry

    @property
    def area(self) -> float:
        """The area of the region, calculated by shapely."""
        return self._geom.area

    @property
    def centroid(self) -> tuple[float, float]:
        """The center point (centroid) of the region, calculated by shapely."""
        centroid = self._geom.centroid
        return (centroid.x, centroid.y)

    @property
    def shapely(self) -> BaseGeometry:
        """
        Provides direct access to the internal shapely geometry object
        for advanced geometric operations.
        """
        return self._geom

    @abstractmethod
    def get_bounding_box(self) -> 'BoundingBox':
        """
        An abstract method that must be implemented by all subclasses.

        Returns:
            BoundingBox: The smallest axis-aligned BoundingBox that encloses this region.
        """
        pass

    def __repr__(self) -> str:
        """Provides an unambiguous string representation of the object."""
        # This will be inherited by subclasses and provides a sensible default.
        return f"{self.__class__.__name__}(wkt='{self._geom.wkt[:55]}...')"