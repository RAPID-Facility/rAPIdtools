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
# 11-13-2025

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Pillow is an optional dependency for image loading
try:
    from PIL import Image as PillowImage
except ImportError:
    PillowImage = None


@dataclass(kw_only=True)
class ImageAsset:
    """
    Represents a single image asset, including its location, metadata,
    and potentially related data like segmentation masks.

    This class uses pathlib.Path for robust and OS-agnostic path handling.

    Attributes:
        path (Path): The full, absolute path to the image file.
        properties (dict[str, Any]): A flexible dictionary to hold optional metadata
            such as 'compass_angle', 'altitude', 'timestamp', etc.
        _pil_image (PillowImage.Image | None): A private attribute to hold the
            lazily-loaded image data from the Pillow library.
        _segmentation_mask (Any | None): A placeholder for future segmentation data.
    """
    path: Path | str
    properties: dict[str, Any] = field(default_factory=dict)
    
    # Private attributes for lazily-loaded data
    _pil_image: PillowImage.Image | None = field(default=None, repr=False, init=False)
    _segmentation_mask: Any | None = field(default=None, repr=False, init=False)

    def __post_init__(self):
        """
        Validates and normalizes input after the object has been initialized.
        """
        # 2. Add logic to convert a string path into a Path object.
        # This ensures that self.path is ALWAYS a Path object for the rest of the class methods.
        if isinstance(self.path, str):
            self.path = Path(self.path)
        
        # The rest of the validation logic can now safely assume self.path is a Path object.
        if not self.path.is_absolute():
            logging.warning(f"ImageAsset path is not absolute: '{self.path}'. Resolving to absolute path.")
            self.path = self.path.resolve()
        
        if not self.path.exists():
            raise ValueError(f"Image file does not exist at the specified path: {self.path}")

    @property
    def filename(self) -> str:
        """The name of the image file, including its extension (e.g., 'image.jpg')."""
        return self.path.name

    @property
    def stem(self) -> str:
        """The filename without the extension (e.g., 'image')."""
        return self.path.stem

    @property
    def directory(self) -> Path:
        """The parent directory of the image file."""
        return self.path.parent

    def get_property(self, key: str, default: Any = None) -> Any:
        """
        Safely retrieves a metadata property from the asset.

        Args:
            key (str): The name of the property to retrieve (e.g., 'compass_angle').
            default (Any, optional): The value to return if the key is not found.

        Returns:
            The value of the property, or the default value.
        """
        return self.properties.get(key, default)

    def load_image_data(self) -> PillowImage.Image:
        """
        Loads the image data using the Pillow library and caches it.

        This method uses lazy loading: the image is only loaded from disk the
        first time this method is called. Subsequent calls return the cached object.

        Returns:
            PillowImage.Image: The loaded image object.

        Raises:
            ImportError: If the Pillow library is not installed.
            IOError: If the image file cannot be opened or read.
        """
        if self._pil_image:
            return self._pil_image

        if PillowImage is None:
            raise ImportError("Pillow library is required to load image data. Please run 'pip install Pillow'.")
        
        try:
            logging.info(f"Loading image data from: {self.path}")
            self._pil_image = PillowImage.open(self.path)
            # You might want to load it into memory to close the file handle
            self._pil_image.load() 
            return self._pil_image
        except IOError as e:
            logging.error(f"Failed to load image file at {self.path}: {e}")
            raise

    # --- Methods for Future Extensibility ---
    def add_segmentation_mask(self, mask_data: Any):
        """
        Attaches segmentation mask data to this image asset.
        
        The 'mask_data' can be any format you choose (e.g., a NumPy array).
        """
        logging.info(f"Attaching segmentation mask to {self.filename}")
        self._segmentation_mask = mask_data
    
    @property
    def segmentation_mask(self) -> Any | None:
        """Returns the attached segmentation mask, if it exists."""
        return self._segmentation_mask