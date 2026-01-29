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

from typing import Self

import pytest
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry

from rapidtools.core.region import Region


# 1. Update Dummy Implementation
# We must implement 'buffer' now because it is an abstract method in the base
# class:
class ConcreteRegion(Region):
    """A minimal concrete implementation of Region for testing."""

    def __init__(self, geometry):
        self._geom = geometry

    def get_bounding_box(self):
        return 'mock_bbox'

    def buffer(self, distance: float) -> 'Region':
        # Return self just to satisfy the abstract contract for testing:
        return self

    @classmethod
    def from_geometry(cls, geometry: BaseGeometry) -> Self:
        # Simple factory implementation for testing
        return cls(geometry)


# 2. Test Suite
class TestRegionABC:

    @pytest.fixture
    def square_geom(self):
        """Returns a 10x10 square polygon starting at 0,0."""
        return Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])

    @pytest.fixture
    def region(self, square_geom):
        """Returns an instance of our dummy concrete class."""
        return ConcreteRegion(square_geom)

    # --- Abstract Factory Tests ---

    def test_from_geometry_factory(self, square_geom):
        """Test the abstract factory method implementation."""
        region = ConcreteRegion.from_geometry(square_geom)
        assert isinstance(region, ConcreteRegion)
        assert region.geometry == square_geom

    # --- Basic Properties ---

    def test_area(self, region):
        """Test that the area property proxies to shapely correctly."""
        # 10 * 10 = 100:
        assert region.area == 100.0

    def test_bounds(self, region):
        """Test that bounds returns the correct tuple order."""
        # (minx, miny, maxx, maxy):
        assert region.bounds == (0.0, 0.0, 10.0, 10.0)

    def test_centroid(self, region):
        """Test that centroid returns a tuple, not a Point object."""
        # Center of 10x10 square is 5,5:
        assert region.centroid == (5.0, 5.0)
        assert isinstance(region.centroid, tuple)

    def test_geometry_property(self, region, square_geom):
        """Test that the .geometry property exposes the raw shapely object."""
        assert region.geometry is square_geom
        assert isinstance(region.geometry, Polygon)

    def test_geo_interface(self, region):
        """Test compliance with __geo_interface__ protocol."""
        geo = region.__geo_interface__
        assert isinstance(geo, dict)
        assert geo['type'] == 'Polygon'
        assert len(geo['coordinates']) == 1

    def test_wkt_property(self, region, square_geom):
        """Test that wkt returns the correct string representation."""
        assert region.wkt == square_geom.wkt
        assert region.wkt.startswith('POLYGON')

    # --- Dimension Properties (Width/Height) ---

    def test_dimensions(self, region):
        """Test width and height calculations."""
        # 10x10 square:
        assert region.width == 10.0
        assert region.height == 10.0

    def test_dimensions_irregular(self):
        """Test dimensions on a non-square shape."""
        # Rectangle 2 wide, 5 high:
        rect = Polygon([(0, 0), (2, 0), (2, 5), (0, 5), (0, 0)])
        r = ConcreteRegion(rect)
        assert r.width == 2.0
        assert r.height == 5.0

    # --- Validation Properties (is_valid/is_empty) ---

    def test_is_valid(self, region):
        """Test validity check."""
        assert region.is_valid is True

        # Create a "bow-tie" polygon (self-intersecting) which is invalid:
        invalid_geom = Polygon([(0, 0), (1, 1), (0, 1), (1, 0), (0, 0)])
        invalid_region = ConcreteRegion(invalid_geom)
        assert invalid_region.is_valid is False

    def test_is_empty(self, region):
        """Test empty check."""
        assert region.is_empty is False

        # Create an empty polygon:
        empty_geom = Polygon()
        empty_region = ConcreteRegion(empty_geom)
        assert empty_region.is_empty is True

    # --- Magic Methods (__eq__, __hash__, __repr__) ---

    def test_equality(self, region, square_geom):
        """Test geometric equality logic."""
        # 1. Identity check:
        assert region == region

        # 2. Geometric equality (same points, different object):
        other_region = ConcreteRegion(
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        )
        assert region == other_region

        # 3. Inequality:
        diff_region = ConcreteRegion(Polygon([(0, 0), (1, 1), (0, 1)]))
        assert region != diff_region

        # 4. Type mismatch:
        assert region != 'Not a Region'

    def test_hashing(self, region):
        """Test that regions can be hashed and used in sets."""
        # Create another region with the exact same geometry:
        same_geom = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        other_region = ConcreteRegion(same_geom)

        # Hashes should match:
        assert hash(region) == hash(other_region)

        # Should be deduped in a set:
        region_set = {region, other_region}
        assert len(region_set) == 1

    def test_repr(self, region):
        """Test string representation."""
        rep = repr(region)
        assert 'ConcreteRegion' in rep
        assert 'wkt=' in rep
        assert 'POLYGON' in rep

    # --- Spatial Methods (Contains, Intersects, Distance) ---

    def test_contains(self, region):
        """Test contains() with both Geometry and Region."""
        # Point inside:
        assert region.contains(Point(5, 5)) is True
        # Region inside:
        inner = ConcreteRegion(Polygon([(1, 1), (2, 1), (2, 2)]))
        assert region.contains(inner) is True
        # Point outside:
        assert region.contains(Point(20, 20)) is False

    def test_intersects(self, region):
        """Test intersects() with both Geometry and Region."""
        # Line crossing boundary:
        line = LineString([(-1, 5), (5, 5)])
        assert region.intersects(line) is True
        # Disjoint region:
        far = ConcreteRegion(Polygon([(100, 100), (101, 100), (101, 101)]))
        assert region.intersects(far) is False

    def test_distance(self, region):
        """Test distance() calculation."""
        # 1. Distance to a point 10 units to the right (at x=20)
        # Square ends at x=10. Point is at x=20. Distance = 10:
        far_point = Point(20, 5)
        assert region.distance(far_point) == 10.0

        # 2. Distance to an intersecting geometry should be 0:
        touching_point = Point(10, 5)
        assert region.distance(touching_point) == 0.0

        # 3. Distance to another Region
        # Region starting at x=20:
        far_region = ConcreteRegion(Polygon([(20, 0), (22, 0), (22, 2)]))
        assert region.distance(far_region) == 10.0

    # --- Caching ---

    def test_caching_behavior(self, region):
        """Verify that properties are cached."""
        initial_area = region.area
        assert initial_area == 100.0

        # Modify the internal geometry directly
        # (Simulating a state change that should not happen in immutable
        # objects):
        region._geom = Point(0, 0)

        # .area should STILL be 100.0 because it is cached:
        assert region.area == 100.0

        # Proof: If we access the raw geometry, it is different:
        assert region.geometry.area == 0.0
