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

import pytest
from shapely.geometry import Point, Polygon

from rapidtools.core.bounding_box import BoundingBox
from rapidtools.core.polygon_region import PolygonRegion
from rapidtools.core.region import Region

# --- Fixtures ---


@pytest.fixture
def triangle_coords():
    """Returns coordinates for a simple triangle."""
    return [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)]


@pytest.fixture
def square_coords():
    """Returns coordinates for a 10x10 square."""
    return [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


@pytest.fixture
def square_region(square_coords):
    """Returns a PolygonRegion instance of a 10x10 square."""
    return PolygonRegion(square_coords)


# --- Initialization Tests ---


def test_init_valid_list(triangle_coords):
    """Test initialization with a list of tuples."""
    region = PolygonRegion(triangle_coords)
    assert isinstance(region, PolygonRegion)
    assert isinstance(region, Region)
    assert region.area == 6.0  # 0.5 * base(4) * height(3)


def test_init_valid_iterable():
    """Test initialization with a generator/iterable."""
    # (0,0), (1,1), (2,2), (3,3). Degenerate but valid init:
    coords_iter = ((x, x) for x in range(4))
    # Note: 3 points in a line makes a valid 'shell' list, even if area is 0:
    region = PolygonRegion(coords_iter)
    assert len(region.geometry.exterior.coords) == 5  # 4 points + closing point


def test_init_too_few_vertices():
    """Test that ValueError is raised if < 3 vertices are provided."""
    with pytest.raises(ValueError, match='must have at least 3 vertices'):
        PolygonRegion([(0, 0), (1, 1)])


# --- Factory Method Tests ---


def test_from_geometry_valid(square_coords):
    """Test creating a region from an existing Shapely Polygon."""
    raw_poly = Polygon(square_coords)
    region = PolygonRegion.from_geometry(raw_poly)

    assert isinstance(region, PolygonRegion)
    assert region.geometry.equals(raw_poly)


def test_from_geometry_invalid_type():
    """Test that from_geometry raises TypeError for non-Polygons."""
    point = Point(0, 0)
    with pytest.raises(TypeError, match='requires a Polygon'):
        PolygonRegion.from_geometry(point)


# --- Bounding Box Tests ---


def test_get_bounding_box(triangle_coords):
    """Test that the correct BoundingBox is returned."""
    # Triangle: (0,0), (4,0), (0,3) -> Bounds should be (0,0, 4,3):
    region = PolygonRegion(triangle_coords)
    bbox = region.get_bounding_box()

    assert isinstance(bbox, BoundingBox)
    assert bbox.bounds == (0.0, 0.0, 4.0, 3.0)
    assert bbox.width == 4.0
    assert bbox.height == 3.0


# --- Buffer Tests ---


def test_buffer_expand(square_region):
    """Test positive buffering (expansion)."""
    # 10x10 square area = 100:
    expanded = square_region.buffer(1.0)

    assert isinstance(expanded, PolygonRegion)
    # Area should be > 100 (100 + 4*10*1 + corners):
    assert expanded.area > 100.0
    assert expanded.is_valid


def test_buffer_erode(square_region):
    """Test negative buffering (erosion)."""
    # Shrink 10x10 square by 1 unit -> approx 8x8 square:
    shrunk = square_region.buffer(-1.0)

    assert isinstance(shrunk, PolygonRegion)
    assert shrunk.area < 100.0
    # Approx 64, but corners are sharp in input so erosion is rounded/complex:
    assert shrunk.area > 50.0


def test_buffer_to_empty(square_region):
    """Test eroding a shape until it disappears."""
    # Shrink 10x10 square by 6 units (radius > half width):
    vanished = square_region.buffer(-6.0)

    assert isinstance(vanished, PolygonRegion)
    assert vanished.is_empty
    assert vanished.area == 0.0


def test_buffer_multipolygon_error():
    """
    Test that buffering into a MultiPolygon raises NotImplementedError.

    We construct a 'dumbbell' shape: two large areas connected by a thin
    bridge. Eroding it removes the bridge, leaving two disjoint polygons.
    """
    # Define a shape with a thin middle section (width 2.0)
    # Left box (0,0 to 10,10), Right box (20,0 to 30,10)
    # Bridge connecting them at y=4 to y=6:
    dumbbell = Polygon(
        [
            (0, 0),
            (10, 0),
            (10, 4),
            (20, 4),
            (20, 0),
            (30, 0),  # Bottom edge
            (30, 10),
            (20, 10),
            (20, 6),
            (10, 6),
            (10, 10),
            (0, 10),  # Top edge
        ]
    )

    region = PolygonRegion.from_geometry(dumbbell)

    # Erosion by -1.1 will destroy the bridge (width 2.0) but keep the boxes
    # Resulting in a MultiPolygon:
    with pytest.raises(NotImplementedError, match='Buffering resulted in MultiPolygon'):
        region.buffer(-1.1)
