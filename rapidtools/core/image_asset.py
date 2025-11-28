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
# 11-28-2025

import logging
import requests
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image as PillowImage


@dataclass(kw_only=True)
class ImageAsset:
    """
    A single image asset, including its location and metadata.
    """
    path: Path | str
    id: str | None = field(default=None) 
    properties: dict[str, Any] = field(default_factory=dict)
    
    allow_missing_file: bool = field(default=False, repr=False)
    
    _pil_image: PillowImage.Image | None = field(
        default=None, 
        repr=False, 
        init=False
    )
    _segmentation_mask: Any | None = field(
        default=None, 
        repr=False, 
        init=False
    )

    def __post_init__(self):
        """
        Validate and normalize initialization arguments.

        Behavior:
        - Converts a string path to a pathlib.Path instance.
        - Resolves the path to an absolute path if it is relative, logging
          a warning when resolution is needed.
        - Verifies that the file exists at the resolved path; raises an error
          if it does not.
        - If id is not provided, sets it to the filename stem.
        
        Raises:
            ValueError: If the image file does not exist at the specified path.
        """
        # Ensure that self.path is ALWAYS a Path object:
        if isinstance(self.path, str):
            self.path = Path(self.path)

        # Resolve the path to a full path:
        self.path = self.path.resolve()
        
        # Only check existence of the iamge if we are NOT allowing missing
        # files:
        if not self.allow_missing_file and not self.path.exists():
            raise ValueError(
                f'Image file does not exist at: {self.path}'
            )
            
        # If no ID was provided, default to the filename without extension:
        if self.id is None:
            self.id = self.path.stem    

    @property
    def directory(self) -> Path:
        """
        The parent directory of the image file.
    
        Returns:
            Path: The directory containing the image as a Path object.
        """
        return self.path.parent
    
    @property
    def filename(self) -> str:
        """
        The filename of the image, including its extension.
    
        Examples:
            'photo.jpg', 'satellite_image.tif'
    
        Returns:
            str: The image filename as a string.
        """
        return self.path.name
    
    @property
    def is_downloaded(self) -> bool:
        """
        Check whether the image file exists on disk and is non-empty.

        This property verifies that:
        - The path exists.
        - The path points to a regular file (not a directory).
        - The file size is greater than zero bytes.

        Returns:
            bool: 
                ``True`` if a non-empty image file exists at image path, 
                otherwise ``False``.
        """
        return (
            self.path.exists()
            and self.path.is_file()
            and self.path.stat().st_size > 0
        )
    
    @property
    def segmentation_mask(self) -> Any | None:
        """
        The segmentation mask associated with this image, if any.
    
        This property exposes the current value of the internal
        _segmentation_mask attribute, which can be used to store
        arbitrary mask representations (e.g., an array, another image
        object, or a custom mask type).
    
        Returns:
            The segmentation mask object if set, otherwise ``None``.
        """
        return self._segmentation_mask
    
    @property
    def stem(self) -> str:
        """
        The filename of the image without its extension.
    
        Examples:
            For 'photo.jpg' -> 'photo'
            For '/data/images/scene_01.tif' -> 'scene_01'
    
        Returns:
            str: The filename stem as a string.
        """
        return self.path.stem

    def download(
            self, 
            url: str | None = None, 
            url_key: str = 'thumb_original_url',
            overwrite: bool = False, 
            session: requests.Session | None = None
        ):
        """
        Downloads the image file to self.path.

        Args:
            url (str, optional): 
                The direct download link. If provided, this takes precedence 
                over properties.
            url_key (str): 
                The key in self.properties to look for if 'url' is None.
                Defaults to 'thumb_original_url'. 
                Useful for downloading specific sizes (e.g. 'thumb_2048_url').
            overwrite (bool): 
                If True, forces download even if file exists.
            session (requests.Session, optional): 
                An existing requests session to speed up connections.
        
        Raises:
            ValueError: If no URL is provided or found in properties.
            IOError: If the download fails.
        """
        # Check if the image needs to be downloaded:
        if self.is_downloaded and not overwrite:
            logging.info(
                f'Skipping download, file already exists: {self.filename}'
            )
            return

        # Determine the image URL:
        # Priority: Explicit URL arg -> Property lookup -> Fail
        target_url = url or self.properties.get(url_key)
        
        # Check if a valid URL is found, if not Show available keys for
        # debugging:
        if not target_url:
            available_keys = ", ".join(k for k in self.properties.keys() if 'url' in k)
            raise ValueError(
                f"Cannot download {self.id}: URL not provided and property "
                f"'{url_key}' not found.\nAvailable URL-like keys: "
                f'{available_keys}'
            )

        # Download image:
        try:
            logging.info(f"Downloading {self.id}...")
            
            # Ensure directory exists:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use provided session or create a temporary one:
            requester = session if session else requests
            
            with requester.get(target_url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(self.path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            logging.info(f"Successfully downloaded: {self.filename}")
            
        except Exception as e:
            # If download fails (e.g., partial file), clean up:
            if self.path.exists():
                self.path.unlink() 
            logging.error(f"Download failed for {self.id}: {e}")
            raise


    def get_property(self, key: str, default: Any = None) -> Any:
        """
        Safely retrieves a metadata property from the asset.

        Args:
            key (str): 
                The name of the property to retrieve (e.g., 'compass_angle').
            default (Any, optional): 
                The value to return if the key is not found.

        Returns:
            The value of the property, or the default value.
        """
        return self.properties.get(key, default)

    def load_image_data(self) -> PillowImage.Image:
        """
        Loads the image data using the Pillow library and caches it.

        This method uses lazy loading: the image is only loaded from disk the
        first time this method is called. Subsequent calls return the cached 
        object.

        Returns:
            PillowImage.Image: The loaded image object.

        Raises:
            ImportError: If the Pillow library is not installed.
            IOError: If the image file cannot be opened or read.
        """
        if self._pil_image:
            return self._pil_image

        if not self.is_downloaded:
            raise FileNotFoundError(
                f'File {self.path} not found. Please call the download method '
                'to retrieve the image file.'
            )
        
        try:
            logging.info(f"Loading image data from: {self.path}")
            self._pil_image = PillowImage.open(self.path)
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
    
