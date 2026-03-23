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
# 03-23-2026

import json
import logging

import pytest
import shapefile
from shapely.geometry import Point, Polygon

from rapidtools.core import (
    BoundingBox,
    ImageAsset,
    PhysicalAsset,
    PhysicalAssetCollection,
)

# --- Fixtures ---


@pytest.fixture
def sample_asset_1():
    """Create a sample asset with ID 'a1' located at (0, 0)."""
    return PhysicalAsset(
        id='a1',
        geometry=Point(0, 0),
        attributes={'type': 'pole', 'status': 'active', 'height': 10},
    )


@pytest.fixture
def sample_asset_2():
    """Create a sample asset with ID 'a2' located at (10, 10)."""
    return PhysicalAsset(
        id='a2',
        geometry=Point(10, 10),
        attributes={'type': 'pole', 'status': 'inactive', 'height': 20},
    )


@pytest.fixture
def sample_asset_3():
    """Create a sample asset with ID 'a3' located at (20, 20)."""
    return PhysicalAsset(
        id='a3',
        geometry=Point(20, 20),
        attributes={'category': 'valve', 'status': 'repair', 'tags': ['urgent']},
    )


@pytest.fixture
def populated_collection(sample_asset_1, sample_asset_2, sample_asset_3):
    """Create a collection populated with three sample assets."""
    return PhysicalAssetCollection([sample_asset_1, sample_asset_2, sample_asset_3])


# --- Initialization Tests ---


def test_init_empty():
    """Test initializing an empty collection."""
    col = PhysicalAssetCollection()
    assert len(col) == 0


def test_init_with_list(sample_asset_1):
    """Test initializing a collection with a list of assets."""
    col = PhysicalAssetCollection([sample_asset_1])
    assert len(col) == 1
    assert 'a1' in col


def test_init_with_generator(sample_asset_1, sample_asset_2):
    """Test initializing a collection with a generator of assets."""
    gen = (a for a in [sample_asset_1, sample_asset_2])
    col = PhysicalAssetCollection(gen)
    assert len(col) == 2


# --- Container Magic Method Tests ---


def test_len(populated_collection):
    """Test that len() returns the correct number of assets."""
    assert len(populated_collection) == 3


def test_iter(populated_collection):
    """Test iterating over the collection yields assets in order."""
    ids = [a.id for a in populated_collection]
    assert ids == ['a1', 'a2', 'a3']


def test_contains(populated_collection, sample_asset_1):
    """Test checking membership using 'in' with IDs and objects."""
    assert 'a1' in populated_collection
    assert sample_asset_1 in populated_collection
    assert 'missing' not in populated_collection

    # Test checking non-asset object:
    assert (123 in populated_collection) is False


def test_getitem_string(populated_collection):
    """Test retrieving an asset using string ID subscription."""
    asset = populated_collection['a1']
    assert asset.id == 'a1'


def test_getitem_string_missing(populated_collection):
    """Test that accessing a missing string ID raises KeyError."""
    with pytest.raises(KeyError):
        _ = populated_collection['missing']


def test_getitem_int(populated_collection):
    """Test retrieving assets by integer index."""
    assert populated_collection[0].id == 'a1'
    assert populated_collection[-1].id == 'a3'


def test_getitem_int_out_of_range(populated_collection):
    """Test that integer indexing out of bounds raises IndexError."""
    with pytest.raises(IndexError):
        _ = populated_collection[99]
    with pytest.raises(IndexError):
        _ = populated_collection[-99]


def test_getitem_negative_step_slice():
    """
    Test slicing with a negative step (reversing), which triggers the
    list conversion fallback logic.
    """
    # Setup a collection with ordered assets:
    assets = [
        PhysicalAsset(id='a1', geometry=Point(0, 0)),
        PhysicalAsset(id='a2', geometry=Point(0, 0)),
        PhysicalAsset(id='a3', geometry=Point(0, 0)),
        PhysicalAsset(id='a4', geometry=Point(0, 0)),
    ]
    col = PhysicalAssetCollection(assets)

    # Perform a reverse slice (step = -1)
    # This specifically forces the 'step < 0' branch in __getitem__:
    reversed_col = col[::-1]

    # Verify the result is a new collection:
    assert isinstance(reversed_col, PhysicalAssetCollection)
    assert len(reversed_col) == 4

    # Verify the order is actually reversed
    # Since iteration order is preserved, check the IDs in order:
    ids = [a.id for a in reversed_col]
    assert ids == ['a4', 'a3', 'a2', 'a1']


def test_getitem_negative_step_partial_slice():
    """
    Test a partial reverse slice (e.g., taking every 2nd item backwards).
    """
    assets = [
        PhysicalAsset(id='0', geometry=Point(0, 0)),
        PhysicalAsset(id='1', geometry=Point(0, 0)),
        PhysicalAsset(id='2', geometry=Point(0, 0)),
        PhysicalAsset(id='3', geometry=Point(0, 0)),
        PhysicalAsset(id='4', geometry=Point(0, 0)),
    ]
    col = PhysicalAssetCollection(assets)

    # Slice from index 4 down to (but not including) 0, stepping -2
    # Indices: 4, 2:
    subset = col[4:0:-2]

    assert len(subset) == 2
    ids = [a.id for a in subset]
    assert ids == ['4', '2']


def test_getitem_slice(populated_collection):
    """Test slicing the collection returns a new PhysicalAssetCollection."""
    # Slice first two:
    subset = populated_collection[:2]
    assert isinstance(subset, PhysicalAssetCollection)
    assert len(subset) == 2
    assert 'a1' in subset
    assert 'a2' in subset
    assert 'a3' not in subset

    # Negative slice fallback:
    subset_neg = populated_collection[:-1]
    assert len(subset_neg) == 2


def test_getitem_invalid_type(populated_collection):
    """Test that using invalid index types (e.g., float) raises TypeError."""
    with pytest.raises(TypeError):
        _ = populated_collection[1.5]


def test_setitem(populated_collection):
    """Test adding an asset using dictionary assignment syntax."""
    new_asset = PhysicalAsset(id='new', geometry=Point(5, 5))
    populated_collection['new'] = new_asset
    assert 'new' in populated_collection


def test_setitem_errors(populated_collection, sample_asset_1):
    """Test invalid assignment operations raise appropriate errors."""
    # Bad key type:
    with pytest.raises(TypeError):
        populated_collection[123] = sample_asset_1

    # Bad value type:
    with pytest.raises(TypeError):
        populated_collection['a1'] = 'not an asset'

    # Mismatch ID:
    with pytest.raises(ValueError, match='ID Mismatch'):
        populated_collection['wrong_key'] = sample_asset_1


def test_delitem(populated_collection):
    """Test removing an asset using the 'del' keyword."""
    del populated_collection['a1']
    assert len(populated_collection) == 2
    assert 'a1' not in populated_collection


def test_delitem_errors(populated_collection):
    """Test deleting with invalid keys raises errors."""
    with pytest.raises(TypeError):
        del populated_collection[123]

    with pytest.raises(KeyError):
        del populated_collection['missing']


# --- Property Tests ---


def test_combined_bounding_box(populated_collection):
    """Test calculation of the bounding box enclosing all assets."""
    bbox = populated_collection.combined_bounding_box
    # a1(0,0), a3(20,20) -> bounds should be 0,0,20,20:
    assert bbox.bounds == (0.0, 0.0, 20.0, 20.0)


def test_combined_bounding_box_empty():
    """Test that an empty collection returns None for bounding box."""
    col = PhysicalAssetCollection()
    assert col.combined_bounding_box is None


# --- Add/Remove Tests ---


def test_add_single_and_duplicate(populated_collection, sample_asset_1, caplog):
    """Test adding single assets and skipping duplicates with a warning."""
    # Add duplicate:
    populated_collection.add(sample_asset_1)
    assert 'Skipping duplicate asset' in caplog.text
    assert len(populated_collection) == 3  # Count shouldn't change

    # Add new:
    new_a = PhysicalAsset(id='new', geometry=Point(1, 1))
    populated_collection.add(new_a)
    assert len(populated_collection) == 4


def test_add_iterable_and_invalid(populated_collection, caplog):
    """Test adding a list containing both valid assets and invalid items."""
    mixed = [PhysicalAsset(id='new1', geometry=Point(1, 1)), 'not_an_asset']
    populated_collection.add(mixed)
    assert len(populated_collection) == 4  # 3 original + 1 new
    assert 'Skipping invalid item' in caplog.text


def test_add_bad_input_type(populated_collection, caplog):
    """Test that adding a non-iterable/non-asset object logs a warning."""
    populated_collection.add(123)
    assert "Received input of type: 'int'" in caplog.text


def test_get_polymorphic(populated_collection):
    """Test get() method handles both single ID strings and lists of IDs."""
    # Single asset:
    assert populated_collection.get('a1').id == 'a1'
    assert populated_collection.get('missing', 'def') == 'def'

    # Batch:
    res = populated_collection.get(['a1', 'missing'])
    assert len(res) == 2
    assert res[0].id == 'a1'
    assert res[1] is None


def test_get_invalid_input(populated_collection):
    """Test that get() raises TypeError for invalid input types."""
    with pytest.raises(TypeError):
        populated_collection.get(123)


def test_remove_polymorphic(populated_collection, caplog):
    """Test remove() method handles single ID strings and lists of IDs."""
    # Single asset:
    populated_collection.remove('a1')
    assert 'a1' not in populated_collection

    # List of assets:
    populated_collection.remove(['a2', 'missing'])
    assert 'a2' not in populated_collection
    assert "Asset ID 'missing' not found" in caplog.text


def test_remove_invalid_type(populated_collection):
    """Test that remove() raises TypeError for invalid input types."""
    with pytest.raises(TypeError):
        populated_collection.remove(123)


# --- Filter Tests ---


def test_filter_by_attribute(populated_collection):
    """Test attribute filtering with various operators (==, >, in, contains)."""
    # ==
    res = populated_collection.filter_by_attribute('type', 'pole')
    assert len(res) == 2

    # >
    res = populated_collection.filter_by_attribute('height', 15, '>')
    assert len(res) == 1
    assert res['a2'].id == 'a2'

    # in
    res = populated_collection.filter_by_attribute('status', ['active', 'repair'], 'in')
    assert len(res) == 2

    # contains (list check)
    res = populated_collection.filter_by_attribute('tags', 'urgent', 'contains')
    assert len(res) == 1
    assert res['a3'].id == 'a3'

    # exists
    res = populated_collection.filter_by_attribute('tags', None, 'exists')
    assert len(res) == 1


def test_filter_by_attribute_errors(populated_collection, caplog):
    """Test attribute filtering handles invalid operators and type mismatches."""
    # Invalid operator:
    with pytest.raises(ValueError):
        populated_collection.filter_by_attribute('type', 'pole', 'bad_op')

    # Missing attribute (should warn and skip):
    populated_collection.filter_by_attribute('non_existent', 'val')
    assert 'Missing attribute' in caplog.text

    # Type mismatch (should warn and skip):
    populated_collection.filter_by_attribute('height', 'string_val', '>')
    assert 'Filter type mismatch' in caplog.text


def test_filter_by_geometry(populated_collection):
    """Test spatial filtering using intersection and containment predicates."""
    # a1 is at 0,0. Box 0,0,5,5 , BoundingBox filter should catch it:
    bbox = BoundingBox(0, 0, 5, 5)
    res = populated_collection.filter_by_geometry(bbox)
    assert len(res) == 1
    assert 'a1' in res

    # Using predicate 'within', create a box that strictly contains a1:
    bbox_large = BoundingBox(-1, -1, 1, 1)
    res = populated_collection.filter_by_geometry(bbox_large, 'within')
    assert len(res) == 1


def test_filter_by_geometry_errors(populated_collection):
    """Test that geometric filtering raises errors for bad inputs."""
    with pytest.raises(TypeError):
        populated_collection.filter_by_geometry('not_geometry')

    with pytest.raises(ValueError):
        populated_collection.filter_by_geometry(
            BoundingBox(0, 0, 1, 1), predicate='bad'
        )


def test_filter_by_geometry_generic_bounds_object(populated_collection):
    """
    Test filtering using a generic object that provides a .bounds property.
    This simulates a custom user class or a tuple-wrapper that behaves like a box.
    """

    # Define a simple dummy class that satisfies the duck-typing check:
    class GenericBox:
        @property
        def bounds(self):
            # Return bounds that cover (0,0) to (5,5)
            # a1 is at (0,0) -> Inside
            # a2 is at (10,10) -> Outside
            return (0.0, 0.0, 5.0, 5.0)

    generic_obj = GenericBox()

    filtered = populated_collection.filter_by_geometry(generic_obj)

    assert len(filtered) == 1
    assert 'a1' in filtered
    assert 'a2' not in filtered


def test_filter_by_geometry_raw_shapely_input(populated_collection):
    """
    Test filtering using a raw Shapely geometry directly.
    """
    # Create a raw Shapely Polygon (not wrapped in PolygonRegion)
    # This triangle covers (0,0) where 'a1' is located:
    raw_shape = Polygon([(0, 0), (0, 2), (2, 0)])

    filtered = populated_collection.filter_by_geometry(raw_shape)

    assert len(filtered) == 1
    assert 'a1' in filtered


# --- Batch & Merge Tests ---


def test_set_attribute(populated_collection):
    """Test batch attribute updates using static values and callables."""
    # Static:
    populated_collection.set_attribute('new_attr', 100)
    assert populated_collection['a1'].attributes['new_attr'] == 100

    # Dynamic:
    populated_collection.set_attribute('x_val', lambda a: a.geometry.x)
    assert populated_collection['a2'].attributes['x_val'] == 10.0

    # Overwrite logic:
    populated_collection['a1'].attributes['fixed'] = 1
    populated_collection.set_attribute('fixed', 2, overwrite=False)
    assert populated_collection['a1'].attributes['fixed'] == 1

    populated_collection.set_attribute('fixed', 2, overwrite=True)
    assert populated_collection['a1'].attributes['fixed'] == 2


def test_merge_strategies(populated_collection):
    """Test merging collections with 'skip', 'overwrite', and 'raise' strategies."""
    # Setup other collection:
    other = PhysicalAssetCollection()
    # Duplicate ID with different data:
    other.add(PhysicalAsset(id='a1', geometry=Point(9, 9), attributes={'new': True}))
    # New ID:
    other.add(PhysicalAsset(id='b1', geometry=Point(1, 1)))

    # Skip
    populated_collection.merge(other, 'skip')
    assert populated_collection['a1'].geometry.x == 0  # Original kept
    assert 'b1' in populated_collection

    # Overwrite:
    populated_collection.merge(other, 'overwrite')
    assert populated_collection['a1'].geometry.x == 9  # Updated

    # Raise:
    with pytest.raises(ValueError):
        populated_collection.merge(other, 'raise')


def test_merge_errors(populated_collection):
    """Test merge validation for invalid types or strategies."""
    with pytest.raises(TypeError):
        populated_collection.merge('not collection')
    with pytest.raises(ValueError):
        populated_collection.merge(PhysicalAssetCollection(), strategy='bad')


# --- Stats & Export Tests ---


def test_summary(populated_collection):
    """Test generation of summary statistics for the collection."""
    stats = populated_collection.summary()
    assert stats['total_assets'] == 3
    assert stats['asset_types'] == {'pole': 2, 'valve': 1}
    assert stats['bounds'] is not None


def test_collect_all_images():
    """Test aggregating image assets from all physical assets."""
    col = PhysicalAssetCollection()
    a1 = PhysicalAsset(id='a1', geometry=Point(0, 0))
    a1.image_assets.add(ImageAsset(id='i1', path='p1.jpg', allow_missing_file=True))
    col.add(a1)

    # Add an empty asset:
    col.add(PhysicalAsset(id='a2', geometry=Point(0, 0)))

    imgs = col.collect_all_images()
    assert len(imgs) == 1
    assert imgs[0].id == 'i1'


def test_to_dataframe(populated_collection):
    """Test exporting the collection to a Pandas DataFrame."""
    df = populated_collection.to_dataframe()
    assert len(df) == 3
    assert 'id' in df.columns
    assert 'asset_type' in df.columns
    assert 'geometry_wkt' in df.columns
    # Check flattened attribute:
    assert 'height' in df.columns
    # Check data integrity:
    assert df[df['id'] == 'a1']['height'].iloc[0] == 10


# --- GeoJSON I/O Tests ---


def test_to_geojson_dict(populated_collection):
    """Test serializing the collection to a GeoJSON dictionary."""
    data = populated_collection.to_geojson()
    assert data['type'] == 'FeatureCollection'
    assert len(data['features']) == 3
    assert data['features'][0]['properties']['type'] == 'pole'


def test_to_geojson_file(populated_collection, tmp_path):
    """Test writing the collection to a GeoJSON file."""
    # Test directory creation and file writing:
    out_file = tmp_path / 'subdir' / 'out.geojson'
    populated_collection.to_geojson(file=out_file)

    assert out_file.exists()
    with open(out_file) as f:
        data = json.load(f)
        assert len(data['features']) == 3


def test_from_geojson_file(tmp_path):
    """Test loading a collection from a GeoJSON file."""
    # Create dummy geojson:
    geo_data = {
        'type': 'FeatureCollection',
        'features': [
            {
                'type': 'Feature',
                'id': 'g1',
                'geometry': {'type': 'Point', 'coordinates': [1, 1]},
                'properties': {'prop': 'val'},
            }
        ],
    }
    p = tmp_path / 'test.geojson'
    with open(p, 'w') as f:
        json.dump(geo_data, f)

    col = PhysicalAssetCollection.from_geojson(p)
    assert len(col) == 1
    assert col['g1'].geometry.x == 1


def test_from_geojson_dict_and_duplicates():
    """Test loading from a dict, handling duplicates, and auto-generating IDs."""
    data = {
        'type': 'FeatureCollection',
        'features': [
            {
                'type': 'Feature',
                'id': 'dup',
                'geometry': {'type': 'Point', 'coordinates': [0, 0]},
                'properties': {},
            },
            {
                # Duplicate ID, should be skipped:
                'type': 'Feature',
                'id': 'dup',
                'geometry': {'type': 'Point', 'coordinates': [1, 1]},
                'properties': {},
            },
            {
                # No ID, should generate one:
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [2, 2]},
                'properties': {},
            },
        ],
    }
    col = PhysicalAssetCollection.from_geojson(data)
    assert len(col) == 2  # 1 duplicate skipped, 1 generated
    assert 'dup' in col
    # Verify the generated one is there:
    found_gen = False
    for k in col._data.keys():
        if k.startswith('gen_'):
            found_gen = True
    assert found_gen


def test_from_geojson_errors(tmp_path):
    """Test error handling for invalid GeoJSON inputs or missing files."""
    # Bad type:
    with pytest.raises(TypeError):
        PhysicalAssetCollection.from_geojson(123)

    # Missing file:
    with pytest.raises(FileNotFoundError):
        PhysicalAssetCollection.from_geojson('ghost.json')

    # Invalid GeoJSON Structure:
    with pytest.raises(ValueError):
        PhysicalAssetCollection.from_geojson({'type': 'BadType'})

    # Feature not a list:
    with pytest.raises(ValueError):
        PhysicalAssetCollection.from_geojson(
            {'type': 'FeatureCollection', 'features': 'not list'}
        )


def test_from_geojson_handles_malformed_geometry(caplog):
    """
    Test that features with existing but malformed geometry
    (causing shapely/init errors) are caught, logged as errors, and skipped.
    """
    # Geometry exists (passes first check), but coordinates are invalid
    # This will cause shapely.geometry.shape() or PhysicalAsset init to raise
    # an error:
    data = {
        'type': 'FeatureCollection',
        'features': [
            {
                'type': 'Feature',
                'id': 'broken_asset',
                'properties': {},
                'geometry': {'type': 'Point', 'coordinates': 'invalid_string_data'},
            }
        ],
    }

    with caplog.at_level(logging.ERROR):
        col = PhysicalAssetCollection.from_geojson(data)

    # Assertions
    # Collection should be empty because the only asset failed:
    assert len(col) == 0

    # Verify the specific error log matches the ID:
    assert "Failed to create asset for ID 'broken_asset'" in caplog.text


def test_from_geojson_skips_missing_geometry(caplog):
    """
    Test that features without a 'geometry' key are skipped and logged as warnings.
    """
    # Setup data: One valid feature, one missing geometry
    data = {
        'type': 'FeatureCollection',
        'features': [
            {
                'type': 'Feature',
                'properties': {'id': 'valid_1'},
                'geometry': {'type': 'Point', 'coordinates': [0, 0]},
            },
            {
                'type': 'Feature',
                'properties': {'id': 'invalid_1'},
                # 'geometry': MISSING
            },
        ],
    }

    with caplog.at_level(logging.WARNING):
        col = PhysicalAssetCollection.from_geojson(data)

    # Assertions
    # Only the valid asset should be loaded:
    assert len(col) == 1
    assert 'valid_1' in col
    assert 'invalid_1' not in col

    # Verify the specific log message exists:
    assert 'Missing geometry' in caplog.text


def test_to_shapefile_empty(tmp_path):
    """Test exporting an empty collection raises ValueError."""
    col = PhysicalAssetCollection()
    with pytest.raises(
        ValueError, match='Cannot export an empty collection to a shapefile.'
    ):
        col.to_shapefile(tmp_path / 'empty.shp')


def test_shapefile_roundtrip(populated_collection, tmp_path):
    """Test full roundtrip export and import of a Shapefile."""
    # Add an image and complex attributes to test serialization:
    img = ImageAsset(id='img_1', path='/tmp/img.jpg', allow_missing_file=True)
    populated_collection['a1'].add_image_assets(img)
    populated_collection['a1'].attributes['metadata'] = {'key': 'value'}

    shp_path = tmp_path / 'test_assets.shp'

    # Export:
    populated_collection.to_shapefile(shp_path)

    # Ensure files were created:
    assert shp_path.exists()
    assert shp_path.with_suffix('.dbf').exists()
    assert shp_path.with_suffix('.shx').exists()
    assert shp_path.with_suffix('.prj').exists()

    # Import:
    imported_col = PhysicalAssetCollection.from_shapefile(shp_path)

    assert len(imported_col) == 3
    assert 'a1' in imported_col

    # Check complex attribute parsing (JSON string to Dict):
    a1_imported = imported_col['a1']
    assert isinstance(a1_imported.attributes.get('metadata'), dict)
    assert a1_imported.attributes['metadata']['key'] == 'value'

    # Check image rehydration (Shapefile parsing places images in the
    # attributes dictionary):
    assert 'images' in a1_imported.attributes
    assert len(a1_imported.attributes['images']) == 1
    assert a1_imported.attributes['images'][0]['id'] == 'img_1'


def test_shapefile_schema_conflict_and_truncation(tmp_path):
    """Test that schema types resolve conflicts and long keys truncate properly."""
    col = PhysicalAssetCollection()

    # Asset 1: long name, int value:
    col.add(
        PhysicalAsset(
            id='1',
            geometry=Point(0, 0),
            attributes={'very_long_attribute_name_1': 10, 'mixed_type': 5},
        )
    )

    # Asset 2: another long name that will collide when truncated to 10 chars
    # "very_long_attribute_name_1" -> "very_long_"
    # "very_long_attribute_name_2" -> "very_long_"
    # (duplicate, should become "very_long1"):
    col.add(
        PhysicalAsset(
            id='2',
            geometry=Point(1, 1),
            attributes={'very_long_attribute_name_2': 'text', 'mixed_type': 5.5},
        )
    )

    # Asset 3: mixed_type becomes string due to unresolvable conflict:
    col.add(
        PhysicalAsset(
            id='3', geometry=Point(2, 2), attributes={'mixed_type': 'now_a_string'}
        )
    )

    shp_path = tmp_path / 'schema.shp'
    col.to_shapefile(shp_path)

    imported = PhysicalAssetCollection.from_shapefile(shp_path)

    # "very_long_" and "very_long1" should be the keys in the rehydrated
    # attributes:
    keys_1 = imported['1'].attributes.keys()
    assert 'very_long_' in keys_1

    keys_2 = imported['2'].attributes.keys()
    assert 'very_long1' in keys_2

    # "mixed_type" should have been degraded to a string for all records
    # because of Asset 3:
    assert isinstance(imported['1'].attributes['mixed_type'], str)
    assert imported['1'].attributes['mixed_type'] == '5'


def test_from_shapefile_duplicate_id(tmp_path, caplog):
    """Test from_shapefile skips duplicate IDs."""
    shp_path = tmp_path / 'dup.shp'

    # Write shapefile manually using pyshp to force duplicate IDs:
    with shapefile.Writer(str(shp_path)) as w:
        w.field('id', 'C', 50)
        w.record(id='dup1')
        w.shape(Point(0, 0))

        # Write another record with the exact same ID:
        w.record(id='dup1')
        w.shape(Point(1, 1))

    with caplog.at_level(logging.WARNING):
        imported = PhysicalAssetCollection.from_shapefile(shp_path)

    assert len(imported) == 1
    assert "Skipping duplicate asset with ID 'dup1'" in caplog.text


def test_from_shapefile_edge_cases(tmp_path):
    """
    Test from_shapefile edge cases:
    - Missing/empty geometry is skipped.
    - Missing ID generates a 'gen_' ID.
    - Malformed JSON in attributes is left as a raw string.
    - Malformed JSON in 'images' column falls back to 'images_raw_str'.
    """
    shp_path = tmp_path / 'edge_cases.shp'

    # Manually craft a shapefile with specific flaws:
    with shapefile.Writer(str(shp_path)) as w:
        # Define fields with specific lengths
        w.field('id', 'C', 50)
        w.field('json_attr', 'C', 50)
        w.field('images', 'C', 50)

        # Record 1: Has ID, but NULL geometry:
        w.record(id='no_geom', json_attr='{}', images='[]')
        w.null()

        # Record 2: Missing ID, malformed JSON attribute, malformed image JSON:
        w.record(id='', json_attr='{bad_json:', images='[bad_image_json:')
        w.shape(Point(1, 1))

    # Import the shapefile using from_shapefile method:
    col = PhysicalAssetCollection.from_shapefile(shp_path)

    # Verification of skip Logic:
    # The record with NULL geometry should be skipped entirely:
    assert 'no_geom' not in col
    assert len(col) == 1

    # Get the only loaded asset (the one from Record 2):
    loaded_asset = list(col)[0]

    # Verification of ID generation:
    # Since Record 2 had an empty string for ID, it should have a generated ID:
    assert loaded_asset.id.startswith('gen_')

    # Verification of malformed JSON attribute handling:
    # Use .strip() because DBF files pad fields with spaces to the field length
    # (50):
    json_attr_val = loaded_asset.attributes.get('json_attr', '').strip()
    assert json_attr_val == '{bad_json:'

    # Verification of malformed 'images' column:
    # If JSON parsing fails on the "images" field, your code moves the raw string
    # to "images_raw_str" and ensures "images" is not in the dict:
    assert 'images' not in loaded_asset.attributes

    raw_images_val = loaded_asset.attributes.get('images_raw_str', '').strip()
    assert raw_images_val == '[bad_image_json:'


def test_from_shapefile_malformed_non_null_geometry(tmp_path):
    shp_path = tmp_path / 'malformed_coords.shp'

    with shapefile.Writer(str(shp_path)) as w:
        w.field('id', 'C', 50)
        w.record(id='valid')
        w.point(10, 10)
        w.record(id='malformed_coords')
        w.point(float('nan'), float('nan'))

    col = PhysicalAssetCollection.from_shapefile(shp_path)

    assert len(col) == 1
    assert 'valid' in col
    assert 'malformed_coords' not in col


def test_filter_empty(populated_collection):
    """Test filtering out assets that have no attributes."""
    # Add an asset with NO attributes:
    empty_asset = PhysicalAsset(id='empty_one', geometry=Point(5, 5), attributes={})
    populated_collection.add(empty_asset)

    assert len(populated_collection) == 4

    # Filter them out:
    clean_col = populated_collection.filter_empty()

    assert len(clean_col) == 3
    assert 'empty_one' not in clean_col
    assert all(a.attributes for a in clean_col)


def test_to_shapefile_ignore_properties(populated_collection, tmp_path):
    """Test that specified properties are excluded from the shapefile export."""
    shp_path = tmp_path / 'ignored.shp'

    # 'height' is in sample_asset_1 and sample_asset_2:
    populated_collection.to_shapefile(shp_path, ignore_properties=['height'])

    imported = PhysicalAssetCollection.from_shapefile(shp_path)
    # Check that 'height' does not exist in the imported attributes:
    for asset in imported:
        assert 'height' not in asset.attributes


def test_to_shapefile_schema_types(tmp_path):
    """Test boolean mapping and Float/Integer type merging (pass branch)."""
    col = PhysicalAssetCollection()

    # Asset 1: Has a Boolean (Logical 'L'):
    col.add(
        PhysicalAsset(
            id='bool_test',
            geometry=Point(0, 0),
            attributes={'is_active': True, 'measure': 1.5},  # 'measure' is Float 'F'
        )
    )

    # Asset 2: Has an Integer for 'measure' (Numeric 'N')
    # This triggers the 'elif curr_type_char == "F" and new_type_char == "N": pass':
    col.add(
        PhysicalAsset(id='int_test', geometry=Point(1, 1), attributes={'measure': 10})
    )

    shp_path = tmp_path / 'types.shp'
    col.to_shapefile(shp_path)

    imported = PhysicalAssetCollection.from_shapefile(shp_path)

    # Verify Boolean survived
    assert imported['bool_test'].attributes['is_active'] is True
    # Verify Float/Int merged into Float
    assert isinstance(imported['int_test'].attributes['measure'], (float, int))


def test_to_shapefile_custom_crs_logging(populated_collection, tmp_path, caplog):
    """Test that using a non-standard CRS logs the skip message."""
    shp_path = tmp_path / 'custom_crs.shp'

    with caplog.at_level(logging.INFO):
        populated_collection.to_shapefile(shp_path, crs='EPSG:3857')

    assert "Skipping .prj generation for custom CRS 'EPSG:3857'" in caplog.text
    # Ensure the .prj file was NOT created:
    assert not shp_path.with_suffix('.prj').exists()
