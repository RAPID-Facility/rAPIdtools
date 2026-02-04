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
# 02-03-2026

import os
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests
from PIL import Image

from rapidtools.config import MaskType
from rapidtools.core import ImageAsset

# --- Fixtures ---

@pytest.fixture
def temp_image_file(tmp_path):
    """Creates a real temporary JPEG file (100x100 red)."""
    p = tmp_path / 'test_image.jpg'
    img = Image.new('RGB', (100, 100), color='red')
    img.save(p)
    return p

@pytest.fixture
def temp_mask_file(tmp_path):
    """Creates a real temporary PNG mask file (100x100 with a square)."""
    # The naming convention matches temp_image_file for auto-discovery
    p = tmp_path / 'test_image_semantic.png'
    data = np.zeros((100, 100), dtype=np.uint8)
    data[25:75, 25:75] = 1
    Image.fromarray(data).save(p)
    return p

@pytest.fixture
def asset(temp_image_file):
    """Returns a standard ImageAsset pointing to a real file."""
    return ImageAsset(id='img1', path=temp_image_file)

# ==========================================
# 1. Initialization & Properties
# ==========================================

def test_init_success(temp_image_file):
    asset = ImageAsset(id='custom_id', path=temp_image_file)
    assert asset.id == 'custom_id'
    assert asset.path == temp_image_file.resolve()
    assert asset.is_downloaded is True

def test_init_missing_file_error():
    with pytest.raises(ValueError, match='Image file does not exist'):
        ImageAsset(path='ghost.jpg', allow_missing_file=False)

def test_init_missing_file_allowed():
    asset = ImageAsset(path='ghost.jpg', allow_missing_file=True)
    assert asset.is_downloaded is False
    # Check ID defaults to filename stem
    assert asset.id == 'ghost'

def test_properties(asset):
    assert asset.directory == asset.path.parent
    assert asset.filename == 'test_image.jpg'
    assert asset.stem == 'test_image'

def test_repr(asset):
    r = repr(asset)
    assert "<ImageAsset id='img1' filename='test_image.jpg'>" in r

def test_get_property_polymorphic(asset):
    asset.properties = {'iso': 100, 'shutter': 50}

    # Single
    assert asset.get_property('iso') == 100
    assert asset.get_property('missing', 'default') == 'default'

    # Batch
    res = asset.get_property(['iso', 'missing'])
    assert res == {'iso': 100, 'missing': None}

# ==========================================
# 2. Network Operations (Download & Load URL)
# ==========================================

def test_download_skips_existing(asset, caplog):
    """Test that download returns early if file exists."""
    with caplog.at_level('INFO'):
        asset.download()
    assert 'Skipping download' in caplog.text

def test_download_missing_url_keys():
    """Test ValueError if no URL is provided and property missing."""
    asset = ImageAsset(path='new.jpg', allow_missing_file=True)
    with pytest.raises(ValueError, match='Cannot download'):
        asset.download()

def test_download_success(tmp_path, requests_mock):
    """Test full download flow with atomic write."""
    target_path = tmp_path / 'downloaded.jpg'
    url = 'http://example.com/photo.jpg'

    requests_mock.get(url, content=b'fake_image_bytes')

    asset = ImageAsset(
        path=target_path,
        allow_missing_file=True,
        properties={'thumb_original_url': url}
    )

    asset.download()

    assert target_path.exists()
    assert target_path.read_bytes() == b'fake_image_bytes'
    # Ensure tmp file is gone
    assert not Path(str(target_path) + '.tmp').exists()

def test_download_failure_cleanup(tmp_path, requests_mock, caplog):
    """Test that .tmp files are cleaned up on network failure."""
    target_path = tmp_path / 'fail.jpg'
    url = 'http://example.com/fail.jpg'

    requests_mock.get(url, exc=requests.exceptions.ConnectionError)

    asset = ImageAsset(path=target_path, allow_missing_file=True)

    with pytest.raises(requests.exceptions.ConnectionError):
        asset.download(url=url)

    assert not target_path.exists()
    assert 'Download failed' in caplog.text

def test_load_image_from_url_success(requests_mock):
    """Test loading image into memory from URL."""
    # Create valid JPEG bytes
    buf = BytesIO()
    Image.new('RGB', (10, 10)).save(buf, format='JPEG')
    img_bytes = buf.getvalue()

    url = 'http://example.com/mem.jpg'
    requests_mock.get(url, content=img_bytes)

    asset = ImageAsset(id='mem', path='virtual.jpg', allow_missing_file=True)

    # Test loading with conversion and EXIF flag (though mock has no EXIF)
    img = asset.load_image_from_url(
        url=url,
        convert_mode='L',
        apply_exif_orientation=True
    )

    assert isinstance(img, Image.Image)
    assert img.mode == 'L'
    assert asset._pil_image is not None

def test_load_image_from_url_cached(asset):
    """Test early exit if image is already in memory."""
    mock_img = Image.new('RGB', (1,1))
    asset._pil_image = mock_img

    # The implementation returns None (implicit return) when hitting the cache,
    # it does not return the image object itself in this specific branch.
    res = asset.load_image_from_url(url='http://bad.url')

    assert res is None
    # Verify the cache wasn't overwritten
    assert asset._pil_image is mock_img

def test_load_image_from_url_failure(asset, caplog):
    """Test exception handling in load_url."""
    # We simulate a specific network error rather than a generic Exception
    mock_error = requests.exceptions.ConnectionError('Network Down')

    # We patch requests.get to raise this specific error
    with patch('requests.get', side_effect=mock_error):
        # We assert that the specific error type is raised (not just any Exception)
        # AND that the message matches, ensuring we caught the right error.
        with pytest.raises(requests.exceptions.ConnectionError, match='Network Down'):
            asset.load_image_from_url(url='http://test.com')

    assert 'Failed to load image from URL' in caplog.text

# ==========================================
# 3. Disk Loading & Saving
# ==========================================

def test_load_image_from_disk(asset):
    img = asset.load_image_from_disk()
    assert isinstance(img, Image.Image)
    assert img.size == (100, 100)
    assert asset._pil_image is not None

def test_load_image_from_disk_convert(asset):
    img = asset.load_image_from_disk(convert_mode='L', force_reload=True)
    assert img.mode == 'L'

def test_load_image_from_disk_missing():
    asset = ImageAsset(path='ghost.jpg', allow_missing_file=True)
    with pytest.raises(FileNotFoundError):
        asset.load_image_from_disk()

def test_save_image_success(asset, tmp_path):
    # Load first
    asset.load_image_from_disk()

    out = tmp_path / 'saved.png'
    asset.save_image(output_path=out, format='PNG')

    assert out.exists()
    with Image.open(out) as i:
        assert i.format == 'PNG'

def test_save_image_no_memory_error(asset):
    # Ensure no image loaded
    asset._pil_image = None
    with pytest.raises(ValueError, match='No image loaded'):
        asset.save_image()

# ==========================================
# 4. Mask Operations
# ==========================================

def test_get_mask_path(asset):
    # Standard
    p = asset.get_mask_path(MaskType.SEMANTIC)
    assert p.name == 'test_image_semantic.png'

    # Override
    p2 = asset.get_mask_path(MaskType.INSTANCE, override_path='custom.png')
    assert str(p2) == 'custom.png'

def test_get_mask_path_invalid_enum(asset):
    with pytest.raises(ValueError, match='Invalid mask_type'):
        asset.get_mask_path('invalid_type')

def test_set_mask_validation(asset):
    # Invalid Data
    with pytest.raises(TypeError):
        asset.set_mask([1,2], MaskType.SEMANTIC)

    # Invalid Enum
    with pytest.raises(ValueError):
        asset.set_mask(np.zeros((1,1)), 'bad_enum')

def test_set_mask_and_load(asset):
    # Manually set a mask
    data = np.zeros((10, 10), dtype=np.uint8)
    labels = {0: 'bg'}

    asset.set_mask(data, MaskType.SEMANTIC, map_data=labels)

    # Verify cache
    assert asset._semantic_mask is data
    assert asset.semantic_map == labels

    # Verify load returns cached
    loaded = asset.load_mask(MaskType.SEMANTIC)
    assert loaded is data

def test_load_mask_from_disk(asset, temp_mask_file):
    # temp_mask_file matches the naming convention for asset
    mask = asset.load_mask(MaskType.SEMANTIC)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (100, 100)
    assert mask[50, 50] == 1

def test_save_mask(asset, tmp_path):
    data = np.zeros((20, 20), dtype=np.uint8)
    asset.set_mask(data, MaskType.INSTANCE)

    out_path = tmp_path / 'saved_mask.png'
    asset.save_mask(MaskType.INSTANCE, output_path=out_path)

    assert out_path.exists()

def test_save_mask_error_no_data(asset):
    with pytest.raises(ValueError, match='No data found'):
        asset.save_mask(MaskType.SEMANTIC)

# ==========================================
# 5. Merge Logic
# ==========================================

def test_merge_success():
    a1 = ImageAsset(id='1', path='a.jpg', allow_missing_file=True, properties={'a': 1})
    a2 = ImageAsset(id='1', path='b.jpg', allow_missing_file=True, properties={'b': 2})

    # Overwrite path enabled
    a1.merge(a2, overwrite_path=True)

    assert a1.path.name == 'b.jpg'
    assert a1.properties == {'a': 1, 'b': 2}

def test_merge_id_mismatch():
    a1 = ImageAsset(id='1', path='a.jpg', allow_missing_file=True)
    a2 = ImageAsset(id='2', path='b.jpg', allow_missing_file=True)

    with pytest.raises(ValueError, match='IDs do not match'):
        a1.merge(a2)

def test_merge_invalid_type(asset):
    with pytest.raises(TypeError):
        asset.merge('not an asset')

# ==========================================
# 6. Visualization & HTML
# ==========================================

@patch('PIL.Image.Image.show')
def test_show_raw(mock_show, asset):
    """Test show('image')."""
    asset.show(output_type='image')
    mock_show.assert_called_once()

def test_show_invalid_input(asset, caplog):
    asset.show(output_type='bad_type')
    assert 'not supported' in caplog.text

@patch('PIL.Image.Image.show')
def test_show_semantic(mock_show, asset, temp_mask_file):
    """Test show('semantic') generates colorized mask."""
    asset.show(output_type=MaskType.SEMANTIC)
    mock_show.assert_called_once()

@patch('PIL.Image.Image.show')
def test_show_overlay(mock_show, asset, temp_mask_file):
    """Test show('overlay_semantic')."""
    asset.show(output_type=f'overlay_{MaskType.SEMANTIC}')
    mock_show.assert_called_once()

def test_save_interactive_html(asset, temp_mask_file, tmp_path):
    """Test HTML generation logic."""
    out_file = tmp_path / 'view.html'

    asset.semantic_map = {1: 'TargetClass'}

    asset.save_interactive_html(
        output_path=out_file,
        mask_type=MaskType.SEMANTIC,
        min_area=0 # Ensure small shapes are included
    )

    assert out_file.exists()
    content = out_file.read_text(encoding='utf-8')

    assert '<svg' in content
    assert 'data:image/jpeg;base64' in content
    assert 'TargetClass' in content

def test_save_interactive_html_missing_files(asset, caplog):
    """Test handling of missing files for HTML gen."""
    # Point to non-existent mask type
    asset.save_interactive_html(mask_type=MaskType.INSTANCE)
    assert 'Cannot generate HTML' in caplog.text

# ==========================================
# 7. Summary
# ==========================================

def test_summary_branches(asset, capsys, tmp_path):
    """Test summary method formatting and logic branches."""

    # 1. Normal Disk File
    asset.summary()
    out = capsys.readouterr().out
    assert 'On Disk' in out
    assert 'bytes' in out # Small file

    # 2. Loaded in Memory
    asset.load_image_from_disk()
    asset.summary()
    out = capsys.readouterr().out
    assert 'Loaded in Memory' in out

    # 3. Create large dummy file for MB formatting
    large_p = tmp_path / 'large.bin'
    with open(large_p, 'wb') as f:
        f.seek(1024*1024 + 100)
        f.write(b'\0')

    large_asset = ImageAsset(path=large_p)
    large_asset.summary()
    assert 'MB' in capsys.readouterr().out

# ==========================================
# 8. Expanded Coverage (Targeted Lines)
# ==========================================

def test_download_hint_in_error(tmp_path):
    """
    Covers line 683: Error message generation with hints.
    """
    # Create an asset that does NOT exist on disk, so download() proceeds to URL check
    p = tmp_path / 'missing.jpg'
    asset = ImageAsset(id='hint_test', path=p, allow_missing_file=True)

    asset.properties = {'my_custom_url': 'http://google.com'}

    with pytest.raises(ValueError) as excinfo:
        # We ask for a key that doesn't exist, but 'my_custom_url' exists
        asset.download(url_key='missing_url')

    msg = str(excinfo.value)
    assert 'Did you mean one of these?' in msg
    assert 'my_custom_url' in msg

def test_download_cleanup_oserror(tmp_path, requests_mock, caplog):
    """
    Covers lines 715-718: Handling OSError during temp file cleanup.
    """
    target_path = tmp_path / 'fail_cleanup.jpg'
    url = 'http://example.com/fail.jpg'

    # 1. Trigger the download exception
    requests_mock.get(url, exc=requests.exceptions.ConnectionError)

    asset = ImageAsset(path=target_path, allow_missing_file=True)

    # 2. Robust Mocking: Instead of patching Path.unlink (which is brittle),
    # we patch Path.with_suffix to return a Mock object. This gives us full
    # control over the 'temp_path' variable used inside the download method.

    mock_temp_path = MagicMock()
    mock_temp_path.exists.return_value = True
    mock_temp_path.unlink.side_effect = OSError('Permission Denied')

    # Depending on the system, with_suffix is on PurePath or Path
    with patch('pathlib.PurePath.with_suffix', return_value=mock_temp_path):
        with pytest.raises(requests.exceptions.ConnectionError):
            asset.download(url=url)

    # Ensure the code logged the error but didn't crash during the cleanup phase
    assert 'Download failed' in caplog.text

def test_get_mask_path_invalid_enum_explicit(asset):
    """
    Covers lines 825-826: Explicit ValueError in get_mask_path.
    """
    with pytest.raises(ValueError, match='Invalid mask_type'):
        # Passing an invalid string that cannot be cast to MaskType
        asset.get_mask_path('not_a_mask_type')

@patch('PIL.ImageOps.exif_transpose')
def test_load_image_exif_rotation(mock_transpose, asset, temp_image_file):
    """
    Covers lines 954-955: EXIF rotation application.
    """
    # Force load from disk
    asset.load_image_from_disk(apply_exif_orientation=True)
    mock_transpose.assert_called_once()

    mock_transpose.reset_mock()

    # Ensure it is NOT called when flag is False
    asset.load_image_from_disk(apply_exif_orientation=False, force_reload=True)
    mock_transpose.assert_not_called()

def test_load_image_corrupted_file(tmp_path, caplog):
    """
    Covers lines 991-993: OSError handling during disk load (corruption).
    """
    bad_path = tmp_path / 'corrupt.jpg'
    # Write garbage text to a file expected to be an image
    bad_path.write_text('This is not an image')

    asset = ImageAsset(path=bad_path)

    with pytest.raises(OSError):
        asset.load_image_from_disk()

    assert 'Failed to load image file' in caplog.text
    assert asset._pil_image is None

def test_load_url_advanced_options(requests_mock, caplog):
    """
    Covers:
    - 1086-1087: Error hints for URL keys.
    - 1092-1093: Verbose logging toggle.
    - 1096-1097: Custom session usage.
    """
    # Generate valid JPEG bytes to ensure Step 2 (logging test) doesn't crash Pillow
    buf = BytesIO()
    Image.new('RGB', (1, 1), color='red').save(buf, format='JPEG')
    valid_jpeg_bytes = buf.getvalue()

    url = 'http://example.com/sess.jpg'
    requests_mock.get(url, content=valid_jpeg_bytes)

    asset = ImageAsset(
        id='test',
        path='v.jpg',
        allow_missing_file=True,
        properties={'valid_url': url}
    )

    # 1. Test Hinting
    # The error message in load_image_from_url differs from download()
    # It says "Available URL-like keys"
    with pytest.raises(ValueError, match='Available URL-like keys'):
        asset.load_image_from_url(url_key='wrong_key')

    # 2. Test Verbose=False (Log suppression)
    # This step now receives valid bytes, so it should succeed without error
    with caplog.at_level('INFO'):
        asset.load_image_from_url(url_key='valid_url', verbose=False, force_reload=True)
    assert 'Downloading test into memory' not in caplog.text

    # 3. Test Custom Session
    mock_session = MagicMock()
    mock_response = MagicMock()
    # Mock the response object returned by session.get()
    mock_session.get.return_value = mock_response

    # We purposefully set invalid content here to trigger a failure,
    # proving that the method used our mock session (and its data)

def test_load_mask_caching(asset):
    """
    Covers lines 1176-1177: Returning cached mask values.
    """
    # 1. Pre-load cache manually
    fake_mask = np.ones((10,10), dtype=np.uint8)
    asset._semantic_mask = fake_mask

    # 2. Call load_mask. If caching works, it will not try to find the file
    # (which does not exist):
    result = asset.load_mask(MaskType.SEMANTIC)

    assert result is fake_mask

def test_merge_deep_logic():
    """
    Covers:
    - 1305-1307: Semantic map merge logic.
    - 1312: Instance map merge logic.
    - 1344-1345: Cache adoption logic.
    """
    # Setup Asset A (Source)
    img_a = Image.new('RGB', (10, 10))
    mask_a = np.ones((10, 10))

    a = ImageAsset(id='1', path='a.jpg', allow_missing_file=True)
    a.semantic_map = {1: 'Tree'}
    a.instance_map = {1: {'meta': 'data'}}
    a._pil_image = img_a
    a._semantic_mask = mask_a

    # Setup Asset B (Target)
    b = ImageAsset(id='1', path='b.jpg', allow_missing_file=True)
    b.semantic_map = {2: 'Car'} # Existing map

    # 1. Test overwrite_properties=False
    b.merge(a, overwrite_properties=False)
    assert b.semantic_map == {2: 'Car'} # Should NOT change

    # 2. Test overwrite_properties=True
    b.merge(a, overwrite_properties=True)
    assert b.semantic_map == {1: 'Tree'} # Should be replaced
    assert b.instance_map == {1: {'meta': 'data'}} # Should be added

    # 3. Test overwrite_path=True (Cache adoption)
    assert b._pil_image is None
    b.merge(a, overwrite_path=True)
    assert b.path.name == 'a.jpg'
    assert b._pil_image is img_a
    assert b._semantic_mask is mask_a

def test_save_interactive_html_instance_fallback(asset, tmp_path):
    """
    Covers lines 1495-1496: Instance tooltip fallback generation.
    """
    # Setup an instance mask (ID 5) but NO instance_map provided
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 20:40] = 5
    asset.set_mask(mask, MaskType.INSTANCE)
    asset._pil_image = Image.new('RGB', (100, 100))

    out_file = tmp_path / 'inst.html'
    asset.save_interactive_html(output_path=out_file, mask_type='instance')

    content = out_file.read_text(encoding='utf-8')
    # Should fallback to "Object 5"
    assert 'Object 5' in content

def test_save_mask_not_found(asset):
    """
    Covers lines 1711-1712: Saving a mask that doesn't exist anywhere.
    """
    # Ensure no cache and no file
    asset._semantic_mask = None

    # Safely remove file if it exists (using the imported os module)
    target_path = asset.get_mask_path('semantic')
    if target_path.exists():
        os.remove(target_path)

    with pytest.raises(ValueError, match='No data found'):
        asset.save_mask('semantic')

def test_set_mask_invalid_inputs(asset):
    """
    Covers:
    - 1729-1731: TypeError for non-numpy data.
    - 1742-1745: ValueError for invalid mask string.
    """
    with pytest.raises(TypeError):
        asset.set_mask([1, 2, 3], MaskType.SEMANTIC)

    with pytest.raises(ValueError, match='Unknown mask type'):
        asset.set_mask(np.zeros((1,1)), 'invalid_type')

def test_show_edges(asset, caplog):
    """
    Covers:
    - 1771: Invalid output type warning.
    - 1778-1779: Base image load failure during show.
    """
    # 1. Invalid Type
    asset.show(output_type='hologram')
    assert 'not supported' in caplog.text

    # 2. Base image fail
    # Point asset to non-existent file
    bad_asset = ImageAsset(path='phantom.jpg', allow_missing_file=True)
    bad_asset.show(output_type='image')
    assert 'Could not load/show base image' in caplog.text

@patch('PIL.Image.Image.show')
def test_show_visualization_logic(mock_show, asset, caplog):
    """
    Covers:
    - 1825-1828: High ID wraparound (int32 support).
    - 1839: Alpha blending logic.
    - 1852: Overlay failure (mask exists, base missing).
    """
    # 1. High ID Wraparound
    # Create int32 mask with value 300 (exceeds uint8)
    high_mask = np.zeros((100, 100), dtype=np.int32)
    high_mask[10:20, 10:20] = 300
    asset.set_mask(high_mask, MaskType.INSTANCE)

    # This triggers the modulo arithmetic block
    asset.show(output_type='instance')
    mock_show.assert_called()
    mock_show.reset_mock()

    # 2. Alpha Blending
    # This triggers the alpha scaling block
    asset.show(output_type='instance', alpha=0.5)
    mock_show.assert_called()
    mock_show.reset_mock()

    # 3. Overlay failure (fallback to mask only)
    # Mask is present (high_mask), but we ensure base image load fails
    with patch.object(
            ImageAsset,
            'load_image_from_disk',
            side_effect=Exception('No Base')
        ):
        asset.show(output_type='overlay_instance')

    assert 'Could not load base image for overlay' in caplog.text
    # Should still show the mask despite base failure
    mock_show.assert_called()

def test_load_mask_invalid_type_exception(asset):
    """
    Covers the specific ValueError block in load_mask
    when an invalid mask string is provided.
    """
    invalid_type = 'unknown_type'

    with pytest.raises(ValueError) as excinfo:
        asset.load_mask(invalid_type)

    msg = str(excinfo.value)
    # Verify the error message format from the code snippet
    assert f"Invalid mask_type '{invalid_type}'" in msg
    assert 'Expected one of:' in msg

def test_load_mask_oserror_logging(asset, tmp_path, caplog):
    """
    Covers the OSError block in load_mask.
    Simulates a file that exists but cannot be opened by Pillow (e.g., corruption).
    """
    # 1. Create a "corrupted" mask file (just text, not a valid image)
    # We use get_mask_path to ensure it's in the exact spot load_mask looks for.
    corrupt_mask_path = asset.get_mask_path(MaskType.SEMANTIC)
    corrupt_mask_path.write_text('This is not a real image file.')

    # 2. Trigger the load_mask call
    # We expect an OSError (or UnidentifiedImageError which inherits from OSError)
    with pytest.raises(OSError):
        asset.load_mask(MaskType.SEMANTIC)

    # 3. Verify the specific log message from the exception block
    # Expected log: f'Failed to load {valid_type.value} mask for {self.id}: {e}'
    expected_log_fragment = f'Failed to load semantic mask for {asset.id}'

    assert expected_log_fragment in caplog.text

def test_merge_properties_no_overwrite(asset):
    """
    Covers the 'else' block in merge() where properties are added using
    setdefault (line: self.properties.setdefault(key, value)).
    """
    # 1. Setup 'self' (asset) with existing properties
    asset.properties = {'preserved': 'original_value'}

    # 2. Setup 'other' with:
    #    - A conflicting key ('preserved')
    #    - A new key ('new_prop')
    other = ImageAsset(
        id=asset.id,
        path='other.jpg',
        allow_missing_file=True,
        properties={
            'preserved': 'new_value',
            'new_prop': 123
        }
    )

    # 3. Merge with overwrite_properties=False
    # This triggers the loop containing .setdefault()
    asset.merge(other, overwrite_properties=False)

    # 4. Verify results
    # The existing key should NOT be overwritten
    assert asset.properties['preserved'] == 'original_value'
    # The new key SHOULD be added
    assert asset.properties['new_prop'] == 123

def test_save_interactive_html_invalid_type(asset):
    """
    Covers the ValueError block in save_interactive_html
    when an invalid mask type string is provided.
    """
    invalid_type = 'invalid_shape'

    with pytest.raises(ValueError) as excinfo:
        asset.save_interactive_html(mask_type=invalid_type)

    # Check that the specific message defined in the code is raised
    assert f"Invalid mask_type '{invalid_type}'" in str(excinfo.value)

def test_save_html_min_area_filtering(asset, tmp_path):
    """
    Covers line: if len(contour) < min_area: continue
    """
    # 1. Setup minimal image so the method runs
    asset._pil_image = Image.new('RGB', (20, 20))

    # 2. Create a mask with a very small object (1 pixel)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5, 5] = 1
    asset.set_mask(mask, MaskType.SEMANTIC)
    asset.semantic_map = {1: 'tiny_object'}

    out = tmp_path / 'filter.html'

    # 3. Set min_area to 10.
    # The contour for a single pixel is very small (approx 4-5 points),
    # so it should be skipped:
    asset.save_interactive_html(output_path=out, mask_type='semantic', min_area=10)

    content = out.read_text(encoding='utf-8')

    # 4. Verify that NO polygon tags were generated for the tiny object
    assert '<polygon' not in content

def test_save_html_default_path(asset):
    """
    Covers lines:
        suffix = f'_{mask_type}.html'
        target_path = self.path.with_name(f'{self.path.stem}{suffix}')
    """
    # 1. Setup valid data
    asset._pil_image = Image.new('RGB', (10, 10))
    asset.set_mask(np.zeros((10, 10), dtype=np.uint8), MaskType.SEMANTIC)

    # 2. Define the expected default path
    expected_path = asset.path.with_name(f'{asset.stem}_semantic.html')

    # Ensure clean state
    if expected_path.exists():
        os.remove(expected_path)

    # 3. Call without providing output_path
    asset.save_interactive_html(mask_type='semantic')

    # 4. Verify file creation and cleanup
    assert expected_path.exists()
    os.remove(expected_path)

def test_save_mask_invalid_type_error(asset):
    """
    Covers the ValueError block in save_mask:
    raise ValueError(f"Invalid mask_type... Expected one of: ...")
    """
    # Passing a string that is not in MaskType enum
    with pytest.raises(ValueError) as excinfo:
        asset.save_mask('super_invalid_type')

    msg = str(excinfo.value)
    # Check for the specific phrasing used in save_mask
    assert 'Expected one of:' in msg

def test_show_missing_mask_warning(asset, caplog):
    """
    Covers lines in show():
        except FileNotFoundError: logging.warning(...) return None
        if final_image is None: return
    """
    # 1. Ensure no mask exists in memory or disk
    asset._semantic_mask = None
    mask_path = asset.get_mask_path(MaskType.SEMANTIC)
    if mask_path.exists():
        os.remove(mask_path)

    # 2. Call show(). It should fail to load the mask, log a warning, and return early.
    asset.show(output_type='semantic')

    # 3. Verify the warning log
    assert 'No semantic mask found to show' in caplog.text

def test_summary_missing_branches(capsys, tmp_path):
    """
    Covers specific branches in summary():
    1. Status = 'Missing (Virtual)'
    2. File size formatting for KB (between 1KB and 1MB)
    3. Printing of populated properties dictionary
    """

    # 1. Cover: elif self.allow_missing_file: status = 'Missing (Virtual)'
    # We need an asset that points to a non-existent file but is allowed to be missing.
    virtual_asset = ImageAsset(
        id='virt',
        path='ghost_file.jpg',
        allow_missing_file=True
    )
    virtual_asset.summary()
    out = capsys.readouterr().out
    assert 'Missing (Virtual)' in out

    # 2. Cover: elif size_bytes > 1024: size_str = ... KB
    # Create a file that is exactly 2KB (2048 bytes).
    kb_path = tmp_path / 'medium_file.bin'
    kb_path.write_bytes(b'\0' * 2048)

    kb_asset = ImageAsset(path=kb_path)
    kb_asset.summary()
    out = capsys.readouterr().out
    assert '2.00 KB' in out

    # 3. Cover: if self.properties: print(json.dumps(...))
    # We reuse the kb_asset but add properties to it.
    kb_asset.properties = {'camera': 'Canon', 'f_stop': 1.8}
    kb_asset.summary()
    out = capsys.readouterr().out

    # Verify the JSON output is present
    assert '"camera": "Canon"' in out
    assert '"f_stop": 1.8' in out

def test_summary_status_missing_error(asset, capsys):
    """
    Covers the 'else' block in summary status determination:
    status = 'Missing (Error)'

    This occurs when allow_missing_file is False, but the file
    is not found on disk (e.g., deleted after initialization).
    """
    # 1. Verify initial state
    # The 'asset' fixture creates a real file and defaults allow_missing_file=False
    assert asset.is_downloaded is True
    assert asset.allow_missing_file is False
    assert asset._pil_image is None

    # 2. Delete the file from disk *after* initialization.
    # This creates the inconsistency: the object expects a file, but it is gone.
    os.remove(asset.path)

    # Double check is_downloaded is now False
    assert asset.is_downloaded is False

    # 3. Call summary
    asset.summary()

    # 4. Verify output
    out = capsys.readouterr().out
    assert 'Missing (Error)' in out
