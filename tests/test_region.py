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

import pytest
from shapely.geometry import Polygon, Point
from rapidtools.core.region import Region

# Define a Dummy class just for testing purposes:
class ConcreteRegion(Region):
    """A minimal concrete implementation of Region for testing."""
    def __init__(self, geometry):
        self._geom = geometry

    def get_bounding_box(self):
        # There is no need to test BoundingBox logic here, this is just that
        # the abstract method exists and can be called:
        return "mock_bbox"

# Test suite:
class TestRegionABC:
    
    @pytest.fixture
    def square_geom(self):
        """Returns a 10x10 square polygon starting at 0,0."""
        return Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])

    @pytest.fixture
    def region(self, square_geom):
        """Returns an instance of our dummy concrete class."""
        return ConcreteRegion(square_geom)

    def test_area(self, region):
        """Test that the area property proxies to shapely correctly."""
        # 10 * 10 = 100
        assert region.area == 100.0

    def test_bounds(self, region):
        """Test that bounds returns the correct tuple order."""
        # (minx, miny, maxx, maxy)
        assert region.bounds == (0.0, 0.0, 10.0, 10.0)

    def test_centroid(self, region):
        """Test that centroid returns a tuple, not a Point object."""
        # Center of 10x10 square is 5,5
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

    def test_repr_short_wkt(self):
        """Test __repr__ with a short geometry."""
        # A simple triangle
        poly = Polygon([(0, 0), (1, 0), (0, 1)])
        region = ConcreteRegion(poly)
        
        # Should contain class name and full WKT
        assert "ConcreteRegion" in repr(region)
        assert "POLYGON" in repr(region)
        assert "..." not in repr(region)

    def test_repr_long_wkt_truncation(self):
        """Test that __repr__ truncates very long WKT strings."""
        # Create a complex polygon with many points to ensure long WKT
        points = [(i, i) for i in range(50)]
        points.append((0,0)) # Close the loop
        poly = Polygon(points)
        
        region = ConcreteRegion(poly)
        rep = repr(region)
        
        assert "ConcreteRegion" in rep
        assert len(rep) < 100  # Should be short despite complex geometry
        assert "..." in rep    # Should show ellipsis indicating truncation

    def test_caching_behavior(self, region):
        """
        Verify that properties are cached.
        
        We can test this by modifying the underlying private geometry 
        and seeing if the public property remains stale (which proves caching).
        """
        initial_area = region.area
        assert initial_area == 100.0
        
        # Modify the internal geometry directly:
        region._geom = Point(0,0) 
        
        # .area should STILL be 100.0 because it's cached:
        assert region.area == 100.0
        
        # If the raw geometry geometry is accessed, it is different:
        assert region.geometry.area == 0.0