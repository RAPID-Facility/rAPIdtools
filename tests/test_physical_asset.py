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
# 02-01-2025

import logging
import uuid
from datetime import datetime

import pytest
from shapely.geometry import Point, Polygon

from rapidtools.core import ImageAsset, PhysicalAsset

# --- Fixtures ---


@pytest.fixture
def sample_point():
    return Point(10.0, 20.0)


@pytest.fixture
def sample_asset(sample_point):
    return PhysicalAsset(
        id='test_asset_001',
        geometry=sample_point,
        attributes={'material': 'wood', 'year': 1990},
    )


@pytest.fixture
def sample_image():
    return ImageAsset(id='img_1', path='/tmp/img1.jpg', allow_missing_file=True)


# --- Initialization Tests ---


def test_init_valid(sample_point):
    """Test standard initialization."""
    asset = PhysicalAsset(id='A1', geometry=sample_point)
    assert asset.id == 'A1'
    assert asset.geometry == sample_point
    assert asset.attributes == {}
    assert len(asset.image_assets) == 0


def test_init_empty_id(sample_point):
    """Test initialization fails with empty ID."""
    with pytest.raises(ValueError, match="PhysicalAsset 'id' cannot be empty"):
        PhysicalAsset(id='', geometry=sample_point)

def test_init_invalid_id_type():
    """Test that initializing with a non-string ID raises TypeError."""
    with pytest.raises(TypeError, match="PhysicalAsset 'id' must be a string"):
        PhysicalAsset(id=123, geometry=Point(0, 0))

def test_init_invalid_geometry():
    """Test initialization fails with non-geometry object."""
    with pytest.raises(TypeError):
        PhysicalAsset(id='A1', geometry='Not a Point')

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
    asset = PhysicalAsset(id='1', geometry=Point(0, 0), attributes={'color': 'red'})
    assert asset.asset_type is None

    # Case 2: 'type' key present:
    asset = PhysicalAsset(id='2', geometry=Point(0, 0), attributes={'type': 'pole'})
    assert asset.asset_type == 'pole'

    # Case 3: Priority check (should find first match in _ASSET_TYPE_KEYS):
    asset = PhysicalAsset(
        id='3',
        geometry=Point(0, 0),
        attributes={'type': 'pole', 'asset_type': 'utility_structure'},
    )
    assert asset.asset_type == 'utility_structure'


def test_add_attributes(sample_asset):
    """Test adding new attributes."""
    sample_asset.add_attributes({'color': 'brown'})
    assert sample_asset.attributes['color'] == 'brown'
    # Ensure old attributes remain:
    assert sample_asset.attributes['material'] == 'wood'

def test_add_attributes_invalid_type():
    """Test passing a list instead of a dict to add_attributes."""
    asset = PhysicalAsset(id='a1', geometry=Point(0, 0))
    with pytest.raises(TypeError, match='must be a dictionary'):
        asset.add_attributes(['not', 'a', 'dict'])

def test_add_attributes_no_overwrite(sample_asset, caplog):
    """Test adding attributes without overwrite (default)."""
    with caplog.at_level('INFO'):
        sample_asset.add_attributes({'material': 'metal'})

    assert sample_asset.attributes['material'] == 'wood'  # Should not change
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
    assert 'year' in sample_asset.attributes  # Should stay

def test_remove_attributes_missing(caplog):
    """Test removing a non-existent attribute logs a warning."""
    asset = PhysicalAsset(id='a1', geometry=Point(0, 0))
    with caplog.at_level(logging.WARNING):
        asset.remove_attributes('ghost_key')

    assert 'Attempted to remove non-existent attribute' in caplog.text

def test_repr(sample_asset):
    """Test the string representation of the asset."""
    # sample_asset has id='test_asset_001' and geometry=Point
    # It has no 'type' attribute, so it should show asset_type='N/A'
    rep = repr(sample_asset)

    assert 'PhysicalAsset' in rep
    assert "id='test_asset_001'" in rep
    assert "geometry='Point'" in rep
    assert "asset_type='N/A'" in rep

    # Test with a type:
    sample_asset.attributes['type'] = 'pole'
    assert "asset_type='pole'" in repr(sample_asset)


# --- Image Asset Tests ---

def test_add_image_assets_invalid_inputs(caplog):
    """Test adding unsupported types to add_image_assets."""
    asset = PhysicalAsset(id='a1', geometry=Point(0, 0))

    # Pass a top-level integer (unsupported) and a list containing a string
    # (unsupported):
    with caplog.at_level(logging.WARNING):
        asset.add_image_assets(12345, ['bad_string_inside_list'])

    # Check that warnings were logged for both branches
    assert "Ignored argument of type 'int'" in caplog.text
    assert "Ignored item of type 'str'" in caplog.text

def test_add_image_assets_variadic(sample_asset, sample_image):
    """Test adding images via single object and list."""
    img2 = ImageAsset(id='img_2', path='/tmp/img2.jpg', allow_missing_file=True)

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

def test_get_image_assets_missing(caplog):
    """Test getting a non-existent image logs a warning."""
    asset = PhysicalAsset(id='a1', geometry=Point(0, 0))
    with caplog.at_level(logging.WARNING):
        res = asset.get_image_assets('ghost.jpg')

    assert res == []
    assert "Image asset 'ghost.jpg' not found" in caplog.text

def test_remove_image_assets(sample_asset, sample_image):
    """Test removing images."""
    sample_asset.add_image_assets(sample_image)
    assert len(sample_asset.image_assets) == 1

    removed = sample_asset.remove_image_assets('img_1')
    assert len(removed) == 1
    assert len(sample_asset.image_assets) == 0

def test_remove_image_assets_missing(caplog):
    """Test removing a non-existent image logs a warning."""
    asset = PhysicalAsset(id='a1', geometry=Point(0, 0))
    with caplog.at_level(logging.WARNING):
        res = asset.remove_image_assets('ghost_id')

    assert res == []
    assert 'but no matching asset was found' in caplog.text

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
                {'id': 'img_x', 'path': '/tmp/x.jpg'}
            ],
        },
    }

    asset = PhysicalAsset.from_geojson_feature(geojson_input)

    assert asset.id == 'rehydrated_01'
    assert asset.geometry.x == 10
    assert asset.attributes['condition'] == 'good'

    # Check that image_assets was moved from attributes to the collection:
    assert 'image_assets' not in asset.attributes
    assert len(asset.image_assets) == 1

    # Verify the auto-fix logic worked (otherwise ImageAsset would raise
    # FileNotFound):
    assert asset.image_assets[0].id == 'img_x'

def test_from_geojson_feature_errors():
    """Test validation errors in from_geojson_feature."""
    # Invalid type:
    with pytest.raises(ValueError, match='must be a valid GeoJSON Feature'):
        PhysicalAsset.from_geojson_feature({'type': 'Point', 'coordinates': [0,0]})

    # Missing geometry:
    with pytest.raises(ValueError, match='missing geometry'):
        PhysicalAsset.from_geojson_feature({
            'type': 'Feature',
            'id': 'a1',
            'properties': {}
            # "geometry" key is missing
        })



def test_from_geojson_image_rehydration_fail(caplog):
    """Test handling of malformed image data during GeoJSON import."""
    data = {
        'type': 'Feature',
        'geometry': {'type': 'Point', 'coordinates': [0,0]},
        'properties': {
            'image_assets': [
                # Passing an unknown argument 'bad_arg' should cause ImageAsset
                # init to fail:
                {'id': 'img1', 'path': 'p.jpg', 'bad_arg': 'fail'}
            ]
        }
    }

    with caplog.at_level(logging.WARNING):
        asset = PhysicalAsset.from_geojson_feature(data)

    # Verify the asset loaded...
    assert len(asset.image_assets) == 0
    # ...but logged a warning about the failed image
    assert 'Failed to rehydrate an image asset' in caplog.text

def test_from_geojson_missing_id(caplog):
    """Test GeoJSON import when ID is missing generates a placeholder."""
    geojson_input = {
        'type': 'Feature',
        'geometry': {'type': 'Point', 'coordinates': [0, 0]},
        'properties': {},
    }

    asset = PhysicalAsset.from_geojson_feature(geojson_input)
    assert asset.id.startswith('no_id_')

def test_from_geojson_multipolygon_simplification():
    """Test that single-component MultiPolygons are simplified to Polygons."""
    geojson_input = {
        'type': 'Feature',
        'properties': {},
        'geometry': {
            'type': 'MultiPolygon',
            'coordinates': [
                # A single polygon defined within a MultiPolygon structure
                [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]
            ]
        }
    }

    asset = PhysicalAsset.from_geojson_feature(geojson_input)

    # The class should have converted this to 'Polygon':
    assert asset.geometry.geom_type == 'Polygon'

def test_summary_edge_cases(capsys):
    """Test summary method with long geometry and empty fields."""
    # Create asset with a large geometry (to trigger truncation at Line 788)
    # A polygon with many points produces a very long WKT string:
    huge_poly = Polygon([(i, i) for i in range(50)])

    # 2. Create an asset with no attributes and images:
    asset = PhysicalAsset(id='big_one', geometry=huge_poly, attributes={})

    asset.summary()

    captured = capsys.readouterr().out

    # Check Truncation (Geometry string should end with ...):
    assert '...' in captured

    # Check Empty Attributes handling
    assert '(No attributes)' in captured

def test_summary_json_serialization_complex_types(capsys):
    """
    Test that summary() correctly serializes non-standard JSON types
    (like datetime objects and UUIDs) by using the default=str fallback.
    """
    # Setup data that would normally crash json.dumps():
    complex_id = uuid.uuid4()
    creation_date = datetime(2025, 1, 1, 12, 0, 0)

    attributes = {
        'uuid_obj': complex_id,       # Not JSON serializable by default
        'timestamp': creation_date,   # Not JSON serializable by default
        'normal': 'value'
    }

    asset = PhysicalAsset(
        id='asset_complex',
        geometry=Point(0,0),
        attributes=attributes
    )

    # Call summary() which triggers the print statement:
    asset.summary()

    # Capture stdout:
    captured = capsys.readouterr().out

    # Verify the objects were converted to strings in the output
    # If default=str was not working, step 2 would have raised a TypeError:
    assert str(complex_id) in captured
    assert str(creation_date) in captured
    assert '"uuid_obj":' in captured

    # Check Empty Images handling:
    assert '(No image assets)' in captured

def test_summary_prints_image_repr(capsys):
    """
    Test that summary() iterates over image assets and prints their
    string representation using the repr() format.
    """
    # Setup asset
    asset = PhysicalAsset(id='asset_with_imgs', geometry=Point(0,0))

    # Add an image using allow_missing_file=True so a real file is not needed
    # on disk:
    img = ImageAsset(
        id='img_repr_test',
        path='/tmp/fake.jpg',
        allow_missing_file=True
    )
    asset.image_assets.add(img)

    # Call summary:
    asset.summary()

    # Capture output:
    captured = capsys.readouterr().out

    # Verify the specific format "  - {repr}" exists in output
    # The !r in the f-string forces the use of repr(img):
    expected_line = f'  - {img!r}'

    assert 'Image Assets (1):' in captured
    assert expected_line in captured
