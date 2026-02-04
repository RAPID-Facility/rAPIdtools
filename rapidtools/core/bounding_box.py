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

import logging
import math

from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry

from .region import Region


class BoundingBox(Region):
    """
    Represents an axis-aligned rectangular region.

    This class acts as a wrapper around a Shapely BaseGeometry that is strictly
    rectangular. It provides specialized initialization methods and optimized
    properties for rectangular math.
    """

    def __init__(self, min_x: float, min_y: float, max_x: float, max_y: float):
        """
        Initialize the BoundingBox from min/max coordinates.

        The internal geometry is initialized as a Shapely Polygon.

        Args:
            min_x (float): The minimum X coordinate (e.g., min longitude).
            min_y (float): The minimum Y coordinate (e.g., min latitude).
            max_x (float): The maximum X coordinate (e.g., max longitude).
            max_y (float): The maximum Y coordinate (e.g., max latitude).

        Example:
            >>> from rapidtools.core import BoundingBox
            >>>
            >>> bbox = BoundingBox(0.0, 0.0, 10.0, 20.0)
            >>> bbox.width
            10.0
        """
        # Contract fulfillment: Initialize _geom with a Shapely Polygon
        # Normalize coordinates automatically using shapely.geometry.box:
        self._geom = shapely_box(min_x, min_y, max_x, max_y)

    def __repr__(self) -> str:
        """
        Return a specific representation showing the bounds clearly.

        Overrides ``Region.__repr__`` to show coordinates instead of WKT.

        Returns:
            str:
                A string formatted as
                'BoundingBox(minx=..., miny=..., maxx=..., maxy=...)'.

        Example:
            >>> from rapidtools.core import BoundingBox
            >>>
            >>> bbox = BoundingBox(0, 0, 1, 1)
            >>> repr(bbox)
            'BoundingBox(minx=0.0, miny=0.0, maxx=1.0, maxy=1.0)'
        """
        minx, miny, maxx, maxy = self.bounds
        return f'BoundingBox({minx=}, {miny=}, {maxx=}, {maxy=})'

    @property
    def area(self) -> float:
        """
        Return the area of the bounding box .

        Overridden from ``Region`` to use simple float math instead of
        Shapely logic for performance.

        Returns:
            float: The area in the squared units of the coordinate system.

        Note:
            - If coordinates are in meters (e.g., UTM), result is sq. meters.
            - If coordinates are in degrees (e.g., WGS84), result is sq.
              degrees. Square degrees are generally not useful for physical
              size estimates.

        Example:
            >>> from rapidtools.core import BoundingBox
            >>>
            >>> bbox = BoundingBox(0, 0, 5, 5)
            >>> bbox.area
            25.0
        """
        return self.width * self.height

    @property
    def centroid(self) -> tuple[float, float]:
        """
        Return the center point of the bounding box.

        Overridden from ``Region`` to use simple float math instead of
        Shapely logic for performance.

        Returns:
            tuple[float, float]: The (longitude, latitude) center coordinates.

        Example:
            >>> from rapidtools.core import BoundingBox
            >>>
            >>> bbox = BoundingBox(0, 0, 10, 20)
            >>> bbox.centroid
            (5.0, 10.0)
        """
        min_x, min_y, max_x, max_y = self.bounds
        return ((min_x + max_x) / 2, (min_y + max_y) / 2)

    @classmethod
    def from_geometry(cls, geometry: BaseGeometry) -> BoundingBox:
        """
        Create a BoundingBox that exactly encloses a geometry.

        Args:
            geometry (BaseGeometry):
                Any Shapely geometry (Polygon, Point, LineString, etc.) to
                wrap.

        Returns:
            BoundingBox:
                A new instance covering the extent of the input geometry.

        Example:
            >>> from rapidtools.core import BoundingBox
            >>>
            >>> from shapely.geometry import LineString
            >>> line = LineString([(0, 0), (10, 10)])
            >>> bbox = BoundingBox.from_geometry(line)
            >>> bbox.bounds
            (0.0, 0.0, 10.0, 10.0)
        """
        minx, miny, maxx, maxy = geometry.bounds
        return cls(minx, miny, maxx, maxy)

    @classmethod
    def from_union(cls, region1: Region, region2: Region) -> BoundingBox:
        """
        Create a box covering the total extent of two regions.

        This calculates the "envelope union" (min of mins, max of maxs) of two
        regions

        Args:
            region1 (Region): The first region.
            region2 (Region): The second region.

        Returns:
            BoundingBox: A new instance large enough to cover both inputs.

        Example:
            >>> from rapidtools.core import BoundingBox
            >>>
            >>> b1 = BoundingBox(0, 0, 1, 1)
            >>> b2 = BoundingBox(10, 10, 11, 11)
            >>> union_box = BoundingBox.from_union(b1, b2)
            >>> union_box.bounds
            (0.0, 0.0, 11.0, 11.0)
        """
        minx1, miny1, maxx1, maxy1 = region1.bounds
        minx2, miny2, maxx2, maxy2 = region2.bounds

        return cls(
            min(minx1, minx2), min(miny1, miny2), max(maxx1, maxx2), max(maxy1, maxy2)
        )

    def buffer(self, distance: float) -> BoundingBox:
        """
        Return a new BoundingBox expanded (or shrunk) by a constant distance.

        This operation pushes the box boundaries outward by the specified
        distance.

        Args:
            distance (float):
                The amount to expand the box edges. Use negative values to
                shrink the box.

        Returns:
            BoundingBox:
                A new BoundingBox instance with adjusted boundaries.

        Example:
            >>> from rapidtools.core import BoundingBox
            >>>
            >>> bbox = BoundingBox(0, 0, 10, 10)
            >>> expanded = bbox.buffer(1.0)
            >>> expanded.bounds
            (-1.0, -1.0, 11.0, 11.0)
        """
        minx, miny, maxx, maxy = self.bounds
        return BoundingBox(
            minx - distance, miny - distance, maxx + distance, maxy + distance
        )

    def get_bounding_box(self) -> BoundingBox:
        """
        Return the bounding box of this region.

        Since this object is already a BoundingBox, this method simply returns
        ``self``.

        Returns:
            BoundingBox: The object itself.
        """
        return self

    def split(self) -> list[BoundingBox]:
        """
        Splits the bounding box into four equal-sized quadrants.

        This method is used by the API client for adaptive tiling when a tile
        is too dense with features and needs to be subdivided.

        Returns:
            A list of four new BoundingBox objects for the quadrants:
            [bottom-left, bottom-right, top-left, top-right].

        Example:
            >>> from rapidtools.core import BoundingBox
            >>>
            >>> bbox = BoundingBox(0, 0, 10, 10)
            >>> quads = bbox.split()
            >>> quads[0].bounds  # Bottom-Left
            (0.0, 0.0, 5.0, 5.0)
            >>> quads[3].bounds  # Top-Right
            (5.0, 5.0, 10.0, 10.0)
        """
        min_x, min_y, max_x, max_y = self.bounds
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2

        return [
            BoundingBox(min_x, min_y, mid_x, mid_y),  # Bottom-Left
            BoundingBox(mid_x, min_y, max_x, mid_y),  # Bottom-Right
            BoundingBox(min_x, mid_y, mid_x, max_y),  # Top-Left
            BoundingBox(mid_x, mid_y, max_x, max_y),  # Top-Right
        ]

    def tile_by_area(self, max_area: float = 0.01) -> list[BoundingBox]:
        """
        Split the bounding box into a grid of smaller boxes.

        The grid dimensions are calculated to maintain the aspect ratio of the
        original box as closely as possible while ensuring every tile has an
        area less than or equal to ``max_area``.

        Args:
            max_area (float):
                The maximum allowed area for a single tile. Defaults to 0.01.

        Returns:
            list[BoundingBox]: A flattened list of the resulting sub-boxes.

        Example:
            Initialize a bounding box with an area of 100 square units:

            >>> from rapidtools.core import BoundingBox
            >>>
            >>> bbox = BoundingBox(0, 0, 10, 10)

            Split the bounding box into tiles with a maximum area of 25 square
            units:

            >>> tiles = bbox.tile_by_area(max_area=25)
            INFO: Tiling BoundingBox into 2x2 grid (Target Area: 25, Total
            Tiles: 4)
            >>> len(tiles)
            4
        """
        # Protect against zero-division if height or width is 0:
        if self.height <= 0 or self.width <= 0:
            logging.warning('Cannot tile a BoundingBox with 0 width or height.')
            return [self]

        # If the bounding box area is within the maximum allowed, return it:
        if self.area <= max_area:
            return [self]

        # Determine grid dimensions:
        total_tiles_needed = math.ceil(self.area / max_area)

        aspect_ratio = self.width / self.height

        # Calculate splits maitaining the bounding box aspect ratio:
        n_cols = math.ceil(math.sqrt(total_tiles_needed * aspect_ratio))
        n_rows = math.ceil(math.sqrt(total_tiles_needed / aspect_ratio))

        logging.info(
            f'Tiling BoundingBox into {n_cols}x{n_rows} grid '
            f'(Target Area: {max_area}, Total Tiles: {n_cols * n_rows})'
        )

        # Create tiles:
        min_x, min_y, _, _ = self.bounds
        step_x = self.width / n_cols
        step_y = self.height / n_rows

        tiles = []
        for i in range(n_cols):
            for j in range(n_rows):
                # Calculate coordinates relative to origin to minimize float
                # drift:
                x1 = min_x + (i * step_x)
                y1 = min_y + (j * step_y)
                x2 = min_x + ((i + 1) * step_x)
                y2 = min_y + ((j + 1) * step_y)

                tiles.append(BoundingBox(x1, y1, x2, y2))

        return tiles
