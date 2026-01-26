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
# 01-26-2025

import pytest
from shapely.geometry import Point, Polygon
from rapidtools.core import PhysicalAsset, ImageAsset

# --- Fixtures ---

@pytest.fixture
def sample_point():
    return Point(10.0, 20.0)

@pytest.fixture
def sample_asset(sample_point):
    return PhysicalAsset(
        id='test_asset_001',
        geometry=sample_point,
        attributes={'material': 'wood', 'year': 1990}
    )

@pytest.fixture
def sample_image():
    return ImageAsset(
        id='img_1', 
        path='/tmp/img1.jpg', 
        allow_missing_file=True
    )

# --- Initialization Tests ---

def test_init_valid(sample_point):
    """Test standard initialization."""
    asset = PhysicalAsset(id='A1', geometry=sample_point)
    assert asset.id == 'A1'
    assert asset.geometry == sample_point
    assert asset.attributes == {}
    assert len(asset.image_assets) == 0

def test_init_invalid_geometry():
    """Test initialization fails with non-geometry object."""
    with pytest.raises(TypeError):
        PhysicalAsset(id='A1', geometry='Not a Point')

def test_init_empty_id(sample_point):
    """Test initialization fails with empty ID."""
    with pytest.raises(ValueError, match="PhysicalAsset 'id' cannot be empty"):
        PhysicalAsset(id='', geometry=sample_point)

def test_post_init_validation(caplog):
    """Test warning on invalid geometry (e.g., self-intersecting polygon)."""
    # Create a bowtie polygon (self-intersecting):
    invalid_poly = Polygon([(0, 0), (0, 2), (2, 0), (2, 2), (0, 0)])
    
    with caplog.at_level('WARNING'):
        PhysicalAsset(id='bad_geom', geometry=invalid_poly)
    
    assert 'contains invalid geometry' in caplog.text

# --- Attribute Tests ---

def test_asset_type_property():
    """Test the asset_type property logic."""
    # Case 1: No type key:
    asset = PhysicalAsset(
        id='1', 
        geometry=Point(0,0), 
        attributes={'color': 'red'}
    )
    assert asset.asset_type is None

    # Case 2: 'type' key present:
    asset = PhysicalAsset(id='2', geometry=Point(0,0), attributes={'type': 'pole'})
    assert asset.asset_type == 'pole'

    # Case 3: Priority check (should find first match in _ASSET_TYPE_KEYS):
    asset = PhysicalAsset(
        id='3', 
        geometry=Point(0,0), 
        attributes={'type': 'pole', 'asset_type': 'utility_structure'}
    )
    assert asset.asset_type == 'utility_structure'

def test_add_attributes(sample_asset):
    """Test adding new attributes."""
    sample_asset.add_attributes({'color': 'brown'})
    assert sample_asset.attributes['color'] == 'brown'
    # Ensure old attributes remain:
    assert sample_asset.attributes['material'] == 'wood'

def test_add_attributes_no_overwrite(sample_asset, caplog):
    """Test adding attributes without overwrite (default)."""
    with caplog.at_level('INFO'):
        sample_asset.add_attributes({'material': 'metal'})
    
    assert sample_asset.attributes['material'] == 'wood' # Should not change
    assert 'Skipping attribute' in caplog.text

def test_add_attributes_with_overwrite(sample_asset):
    """Test adding attributes with overwrite=True."""
    sample_asset.add_attributes({'material': 'metal'}, overwrite=True)
    assert sample_asset.attributes['material'] == 'metal'

def test_get_attributes(sample_asset, caplog):
    """Test safe retrieval of attributes."""
    # Get existing:
    res = sample_asset.get_attributes('material')
    assert res == {'material': 'wood'}

    # Get missing:
    res = sample_asset.get_attributes('ghost_attr')
    assert res == {}
    assert "Attribute key 'ghost_attr' not found" in caplog.text

def test_remove_attributes(sample_asset):
    """Test removing attributes."""
    sample_asset.remove_attributes('material')
    assert 'material' not in sample_asset.attributes
    assert 'year' in sample_asset.attributes # Should stay

# --- Image Asset Tests ---

def test_add_image_assets_variadic(sample_asset, sample_image):
    """Test adding images via single object and list."""
    img2 = ImageAsset(
        id='img_2', 
        path='/tmp/img2.jpg', 
        allow_missing_file=True
    )
    
    sample_asset.add_image_assets(sample_image, [img2])
    
    assert len(sample_asset.image_assets) == 2
    assert 'img_1' in [i.id for i in sample_asset.image_assets]
    assert 'img_2' in [i.id for i in sample_asset.image_assets]

def test_get_image_assets(sample_asset, sample_image):
    """Test retrieving images by ID and Filename."""
    sample_asset.add_image_assets(sample_image)
    
    # By ID:
    found = sample_asset.get_image_assets('img_1')
    assert len(found) == 1
    assert found[0].id == 'img_1'

    # By filename (derived from path):
    found_file = sample_asset.get_image_assets('img1.jpg')
    assert len(found_file) == 1
    assert found_file[0].id == 'img_1'

def test_remove_image_assets(sample_asset, sample_image):
    """Test removing images."""
    sample_asset.add_image_assets(sample_image)
    assert len(sample_asset.image_assets) == 1
    
    removed = sample_asset.remove_image_assets('img_1')
    assert len(removed) == 1
    assert len(sample_asset.image_assets) == 0

# --- GeoJSON Tests ---

def test_to_geojson_feature(sample_asset, sample_image):
    """Test export to GeoJSON."""
    sample_asset.add_image_assets(sample_image)
    
    feature = sample_asset.to_geojson_feature()
    
    assert feature['type'] == 'Feature'
    assert feature['id'] == 'test_asset_001'
    assert feature['geometry']['type'] == 'Point'
    assert feature['properties']['material'] == 'wood'
    
    # Check if images were serialized:
    assert 'image_assets' in feature['properties']
    assert feature['properties']['image_assets'][0]['id'] == 'img_1'

def test_from_geojson_feature():
    """Test import from GeoJSON including image rehydration."""
    geojson_input = {
        'type': 'Feature',
        'id': 'rehydrated_01',
        'geometry': {'type': 'Point', 'coordinates': [10, 20]},
        'properties': {
            'condition': 'good',
            'image_assets': [
                {'id': 'img_x', 'path': '/tmp/x.jpg', 'allow_missing_file': True}
            ]
        }
    }
    
    asset = PhysicalAsset.from_geojson_feature(geojson_input)
    
    assert asset.id == 'rehydrated_01'
    assert asset.geometry.x == 10
    assert asset.attributes['condition'] == 'good'
    
    # Check that image_assets was moved from attributes to the collection:
    assert 'image_assets' not in asset.attributes
    assert len(asset.image_assets) == 1
    assert asset.image_assets[0].id == 'img_x'

def test_from_geojson_missing_id(caplog):
    """Test GeoJSON import when ID is missing generates a placeholder."""
    geojson_input = {
        'type': 'Feature',
        'geometry': {'type': 'Point', 'coordinates': [0, 0]},
        'properties': {}
    }
    
    asset = PhysicalAsset.from_geojson_feature(geojson_input)
    assert asset.id.startswith('no_id_')