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

import math
import logging
from typing import Self
from shapely.geometry import box as shapely_box

# This import is safe because region.py does not import this file:
from .region import Region

class BoundingBox(Region):
    """
    Represents an axis-aligned rectangular region.

    This class provides a convenient initializer for simple boxes and adds
    specialized behavior, such as tiling.
    """
    def __init__(self, lon1: float, lat1: float, lon2: float, lat2: float):
        """
        Initializes the BoundingBox from two corner points.

        The internal geometry is a shapely 'box', which is a type of Polygon.
        """
        # Normalize coordinates automatically using shapely.geometry.box:
        self._geom = shapely_box(minx=lon1, miny=lat1, maxx=lon2, maxy=lat2)

    # --- BoundingBox-Specific Properties and Methods ---
    @property
    def width(self) -> float:
        """The width of the bounding box in degrees."""
        bounds = self._geom.bounds
        return bounds[2] - bounds[0] # (max_lon - min_lon)

    @property
    def height(self) -> float:
        """The height of the bounding box in degrees."""
        bounds = self._geom.bounds
        return bounds[3] - bounds[1] # (max_lat - min_lat)

    # --- Implementation of the abstract method from the Region base class ---
    def get_bounding_box(self) -> Self:
        """A BoundingBox's bounding box is itself."""
        return self


    def tile_by_area(self, max_area: float = 0.01) -> list[Self]:
        """
        A custom method to split this BoundingBox into a grid of smaller
        BoundingBox objects, each with an area less than or equal to max_area.
        """
        if self.area <= max_area:
            return [self]

        num_tiles = math.ceil(self.area / max_area)
        aspect_ratio = self.width / self.height if self.height > 0 else 1
        n_lon_splits = math.ceil(math.sqrt(num_tiles * aspect_ratio))
        n_lat_splits = math.ceil(math.sqrt(num_tiles / aspect_ratio))
        while (n_lon_splits * n_lat_splits) < num_tiles:
            n_lat_splits += 1

        logging.info(
            f'Splitting BoundingBox into a {n_lon_splits}x{n_lat_splits} grid.'
            )

        min_lon, min_lat, _, _ = self._geom.bounds
        tile_width = self.width / n_lon_splits
        tile_height = self.height / n_lat_splits

        tiles = []
        for i in range(n_lon_splits):
            for j in range(n_lat_splits):
                tile_min_lon = min_lon + i * tile_width
                tile_min_lat = min_lat + j * tile_height
                tiles.append(
                    BoundingBox(
                        tile_min_lon, 
                        tile_min_lat, 
                        tile_min_lon + tile_width, 
                        tile_min_lat + tile_height
                        )
                    )
        return tiles