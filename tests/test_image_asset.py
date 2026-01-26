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

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

# Import your class (adjust the import path based on your file structure)
# from rapidtools.image_asset import ImageAsset
from rapidtools.core import ImageAsset 

# --- Fixtures (Setup) ---

@pytest.fixture
def temp_dir(tmp_path):
    """Creates a temporary directory for test artifacts."""
    d = tmp_path / "data"
    d.mkdir()
    return d

@pytest.fixture
def sample_image(temp_dir):
    """Creates a real 100x100 white JPG image on disk."""
    path = temp_dir / "test_image.jpg"
    img = Image.new('RGB', (100, 100), color='white')
    img.save(path)
    return path

@pytest.fixture
def sample_mask(temp_dir):
    """Creates a fake segmentation mask on disk."""
    path = temp_dir / "test_image_semantic.png"
    # Create a 100x100 mask with 3 classes (0, 1, 2)
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[20:50, 20:50] = 1 # Class 1 box
    arr[60:80, 60:80] = 2 # Class 2 box
    
    img = Image.fromarray(arr)
    img.save(path)
    return path, arr

# --- 1. Initialization Tests ---

def test_init_valid_file(sample_image):
    asset = ImageAsset(path=sample_image)
    assert asset.path == sample_image
    assert asset.id == "test_image"
    assert asset.is_downloaded is True

def test_init_missing_file_raises_error(temp_dir):
    fake_path = temp_dir / "ghost.jpg"
    with pytest.raises(ValueError, match="does not exist"):
        ImageAsset(path=fake_path)

def test_init_allow_missing_file(temp_dir):
    fake_path = temp_dir / "ghost.jpg"
    asset = ImageAsset(path=fake_path, allow_missing_file=True)
    assert asset.is_downloaded is False

def test_manual_id_override(sample_image):
    asset = ImageAsset(path=sample_image, id="custom_id")
    assert asset.id == "custom_id"

# --- 2. Image Loading Tests ---

def test_load_image_from_disk(sample_image):
    asset = ImageAsset(path=sample_image)
    
    # First load (IO)
    img = asset.load_image_from_disk()
    assert isinstance(img, Image.Image)
    assert img.size == (100, 100)
    assert asset._pil_image is not None # Check cache

    # Second load (Cache)
    img2 = asset.load_image_from_disk()
    assert img is img2 # Should be exact same object in memory

def test_load_image_from_url(temp_dir, requests_mock):
    # Setup
    fake_url = "http://example.com/photo.jpg"
    target_path = temp_dir / "downloaded.jpg"
    
    # Mock the API response with a tiny 1x1 image
    buf = BytesIO()
    Image.new('RGB', (1, 1)).save(buf, format='JPEG')
    requests_mock.get(fake_url, content=buf.getvalue())

    # Create asset pointing to non-existent file
    asset = ImageAsset(
        id="test_img", # Added ID since the method logs self.id
        path=target_path, 
        properties={'thumb_original_url': fake_url},
        allow_missing_file=True
    )

    # 1. Trigger the load (returns None)
    asset.load_image_from_url()
    
    # 2. Access the loaded image via the property/attribute
    img = asset._pil_image 
    
    # Assertions
    assert img is not None
    assert img.size == (1, 1)
    assert not target_path.exists() # Verifies it didn't save to disk

# --- 3. Segmentation Tests ---

def test_load_mask_auto_path(sample_image, sample_mask):
    mask_path, original_arr = sample_mask
    asset = ImageAsset(path=sample_image)

    # Load semantic mask (auto-finds test_image_semantic.png)
    loaded_arr = asset.load_mask(mask_type="semantic")
    
    assert np.array_equal(loaded_arr, original_arr)
    assert asset._semantic_mask is not None # Check cache

def test_set_mask_manual(sample_image):
    asset = ImageAsset(path=sample_image)
    
    # Create fake data in memory
    fake_mask = np.ones((50, 50), dtype=np.uint8)
    fake_map = {1: "sky"}

    asset.set_mask(fake_mask, mask_type="semantic", map_data=fake_map)
    
    assert np.array_equal(asset.load_mask("semantic"), fake_mask)
    assert asset.semantic_map == fake_map

def test_save_mask(sample_image, temp_dir):
    asset = ImageAsset(path=sample_image)
    fake_mask = np.full((10, 10), 5, dtype=np.uint8) # Fill with class 5
    asset.set_mask(fake_mask, mask_type="instance")

    output_path = temp_dir / "custom_mask_output.png"
    asset.save_mask(mask_type="instance", output_path=output_path)

    assert output_path.exists()
    
    # Verify content
    saved_img = Image.open(output_path)
    assert np.array_equal(np.array(saved_img), fake_mask)

# --- 4. Interactive HTML Tests ---

def test_save_interactive_html(sample_image, temp_dir):
    asset = ImageAsset(path=sample_image)
    
    # Setup dummy mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 10:90] = 1 # Big square
    
    asset.set_mask(mask, mask_type="semantic", map_data={1: "test_obj"})
    
    output_html = temp_dir / "view.html"
    asset.save_interactive_html(
        output_path=output_html, 
        mask_type="semantic"
    )

    assert output_html.exists()
    
    content = output_html.read_text(encoding="utf-8")
    assert "<svg" in content
    assert "test_obj" in content # Check if metadata made it into the file

# --- 5. Download & Combine Tests ---

def test_combine_with(sample_image):
    asset1 = ImageAsset(path=sample_image, id="A", properties={'a': 1})
    asset2 = ImageAsset(path=sample_image, id="A", properties={'b': 2})
    
    # Merge 2 into 1
    asset1.combine_with(asset2)
    
    assert asset1.properties['a'] == 1
    assert asset1.properties['b'] == 2

def test_download_method(temp_dir, requests_mock):
    fake_url = "http://api.com/img.jpg"
    target = temp_dir / "dl_test.jpg"
    
    # 1. Create VALID image bytes
    buf = BytesIO()
    # Create a real 1x1 red pixel image
    Image.new('RGB', (1, 1), color='red').save(buf, format='JPEG')
    valid_image_bytes = buf.getvalue()
    
    # 2. Mock the response with valid bytes
    requests_mock.get(fake_url, content=valid_image_bytes)
    
    asset = ImageAsset(
        id="test_dl",
        path=target, 
        properties={'thumb_original_url': fake_url}, 
        allow_missing_file=True
    )
    
    # 3. Perform download
    asset.download()
    
    # 4. Assertions
    assert target.exists()
    # Verify the file content matches what we mocked
    assert target.read_bytes() == valid_image_bytes

# --- 6. Visualization Smoke Test ---

def test_show_smoke_test(sample_image, monkeypatch):
    """
    Ensure .show() runs without crashing. 
    We mock PIL.Image.show so no window actually pops up during CI.
    """
    asset = ImageAsset(path=sample_image)
    
    # Mock the internal show method to do nothing
    monkeypatch.setattr(Image.Image, "show", lambda self, title=None: None)
    
    # Should not raise exception
    asset.show(output_type="image")