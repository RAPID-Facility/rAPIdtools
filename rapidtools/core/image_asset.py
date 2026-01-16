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
# 01-15-2025

from __future__ import annotations

from io import BytesIO
import json
import logging
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.cm as cm
import numpy as np
import pandas as pd
import requests
from PIL import Image as PillowImage


if TYPE_CHECKING:
    # This import ONLY happens when type checking.
    # It is ignored when the code actually runs.
    from .bounding_box import BoundingBox 


@dataclass(kw_only=True)
class ImageAsset:
    """
    A single image asset, its location, metadata, and segmentation info.
    """
    path: Path | str
    id: str | None = field(default=None) 
    
    # User-facing metadata (location, photographer, tags, etc.):
    properties: dict[str, Any] = field(default_factory=dict)
    
    # Mapping dictionary for the semantic segmentation masks:
    # Key: Pixel Value (int) -> Value: Class Name (str)
    semantic_map: dict[int, str] | None = field(
        default=None, 
        repr=False
    )

    # Mapping dictionary for the instance segmentation mask:
    # Key: Instance ID (int) -> Value: Metadata dict or label
    instance_map: dict[int, Any] | None = field(
        default=None, 
        repr=False
    )
    
    # Flag for delayed image download for efficient image handling:
    allow_missing_file: bool = field(default=False, repr=False)
    
    # --- Internal Caches ---
    _pil_image: PillowImage.Image | None = field(
        default=None, 
        repr=False, 
        init=False
    )
    
    # Lazy-loaded NumPy arrays for segmentation masks:
    _semantic_mask: np.ndarray | None = field(
        default=None, 
        repr=False, 
        init=False
    )
    
    _instance_mask: np.ndarray | None = field(
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
        
        # Only check existence of the image if we are NOT allowing missing
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

    def combine_with(
            self, 
            other: ImageAsset, 
            ignore_id_mismatch: bool = False, 
            overwrite_path: bool = False,
            overwrite_properties: bool = True
        ) -> None:
        """
        Merges data from another ImageAsset into this one.

        By default, this method ensures IDs match before merging to prevent 
        accidental data corruption.

        Args:
            other (ImageAsset): 
                The source asset to merge data from.
            ignore_id_mismatch (bool): 
                If ``True``, allows merging even if self.id != other.id.
                Defaults to ``False``.
            overwrite_path (bool): 
                If ``True``, updates self.path with other.path and adopts
                its internal caches (images/masks). 
                Defaults to ``False`` (preserves original path).
            overwrite_properties (bool): 
                If ``True``, other.properties, semantic_map, and instance_map
                will overwrite existing values in self. 
                If ``False``, only missing values are filled.
                Defaults to ``True``.

        Raises:
            ValueError: 
                If IDs do not match and ``ignore_id_mismatch`` is ``False``.
        """
        # Check asset ID correspondence:
        if self.id != other.id and not ignore_id_mismatch:
            raise ValueError(
                f"Cannot combine assets: IDs do not match ('{self.id}' vs "
                f"'{other.id}'). Set ignore_id_mismatch=True to bypass this"
                'requirement.'
            )

        # Merge asset roperties:
        if overwrite_properties:
            # Update existing keys and add new ones:
            self.properties.update(other.properties)
        else:
            # Only add keys that do not exist:
            for key, value in other.properties.items():
                self.properties.setdefault(key, value)

        # Merge segmentation maps. If 'other' has a map, we decide whether to
        # take it based on the overwrite flag:
        if other.semantic_map is not None:
            if self.semantic_map is None or overwrite_properties:
                self.semantic_map = other.semantic_map

        if other.instance_map is not None:
            if self.instance_map is None or overwrite_properties:
                self.instance_map = other.instance_map

        # Merge asset path:
        if overwrite_path:
            self.path = other.path
            if isinstance(self.path, str):
                self.path = Path(self.path)
            self.path = self.path.resolve()
            
            # Since the path changed, the old cache is invalid. Either adopt
            # the other's cache (if it exists) or reset to None:
            self._pil_image = other._pil_image
            
            # Similarly, if the path changed, the old mask likely does not
            # apply. Adopt the other's mask (whether it is data or None):
            self._semantic_mask = other._semantic_mask
            self._instance_mask = other._instance_mask
            
        logging.info(f"Merged asset data from {other.id} into {self.id}")

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

    def get_mask_path(
            self, 
            mask_type: Literal['semantic', 'instance'],
            override_path: Path | str | None = None
        ) -> Path:
        """
        Determines the filename for the mask.
        
        Default Convention: 
          - Image:    photo.jpg 
          - Semantic: photo_semantic.png
          - Instance: photo_instance.png
        
        Args:
            mask_type: The type of mask ('semantic' or 'instance').
            override_path: If provided, this specific path is used instead 
                           of the automatic naming convention.
        """
        # Use the manual override if provided:
        if override_path is not None:
            return Path(override_path)
            
        # Otherwise, use the standard convention:
        suffix = f"_{mask_type}.png"
        return self.path.with_name(f"{self.path.stem}{suffix}")

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

    def load_image_from_disk(
            self, 
            force_reload: bool = False,
            convert_mode: str | None = None
        ) -> PillowImage.Image:
        """
        Load the image from disk using Pillow and caches it.
    
        Behavior:
        - Lazy-caches by default (returns cached image if present).
        - Always fully decodes and then closes the underlying file handle.
        Optionally converts to a target mode (e.g., "RGB").
        - Optional force reload to bypass cache.
    
        Args:
            force_reload: 
                Reload from disk even if already cached.
            convert_mode: 
                If provided, converts image to the specified PIL mode. Some
                options are ``L`` (Grayscale), ``1`` (Black and White), 
                ``CMYK`` (Cyan, Magenta, Yellow, Key/Black), ``I`` 
                (32-bit signed integer pixels), and ``HSV`` (Hue, Saturation, 
                Value color space).
    
        Returns:
            PillowImage.Image: The loaded PIL image.
    
        Raises:
            FileNotFoundError: If the file is missing/empty.
            Exception: Propagates Pillow/IO errors with logging.
        """
        if self._pil_image is not None and not force_reload:
            return self._pil_image

        if not self.is_downloaded:
            raise FileNotFoundError(
                f'File {self.path} not found. Please call the download method '
                'to retrieve the image file.'
            )
        
        try:
            logging.info(f"Loading image data from: {self.path}")
            
            with PillowImage.open(self.path) as img:
                img.load()
            
            if convert_mode is not None and img.mode != convert_mode:
                img = img.convert(convert_mode)
            
            self._pil_image = img 
            
            return self._pil_image

        except (IOError, OSError) as e:
            logging.error(f"Failed to load image file at {self.path}: {e}")
            self._pil_image = None 
            raise

    def load_image_from_url(
             self, 
             url: str | None = None, 
             url_key: str = 'thumb_original_url',
             convert_mode: str | None = None,
             force_reload: bool = False,
             session: requests.Session | None = None
         ) -> PillowImage.Image:
         """
         Download the image into memory without saving to disk.
    
         This method bypasses ``self.path`` entirely for the actual data.
         Note that ``self.is_downloaded`` will remain False after this method
         completes (unless the file happened to exist previously).
    
         Args:
             url (str, optional): 
                 Direct URL to download from. If None, checks self.properties.
             url_key (str): 
                 Property key to use for URL if direct url is None.
             convert_mode (str, optional):
                 Mode to convert the loaded image to (e.g., "RGB").
             force_reload (bool):
                 If True, re-downloads even if self._pil_image is already set.
             session (requests.Session, optional):
                 Session for efficient connection pooling during download.
    
         Returns:
             PillowImage.Image: The fully loaded PIL image.
         """
         # Return existing cached image if available:
         if self._pil_image is not None and not force_reload:
             return self._pil_image
    
         # Determine the image URL (logic mirror of download()):
         target_url = url or self.properties.get(url_key)
         
         if not target_url:
             available_keys = ", ".join(k for k in self.properties.keys() if 'url' in k)
             raise ValueError(
                 f"Cannot load image for {self.id}: URL not provided and property "
                 f"'{url_key}' not found.\nAvailable URL-like keys: "
                 f'{available_keys}'
             )
    
         try:
             logging.info(f"Downloading {self.id} into memory...")
             
             # Use provided session or create a temporary one:
             requester = session if session else requests
             
             # Download bytes into memory:
             response = requester.get(target_url, stream=False, timeout=30)
             response.raise_for_status()
    
             # Create an in-memory file-like object:
             image_bytes = BytesIO(response.content)
    
             # Load into Pillow:
             img = PillowImage.open(image_bytes)
             img.load()  # Force decode
    
             # Optional conversion:
             if convert_mode is not None and img.mode != convert_mode:
                 img = img.convert(convert_mode)
    
             self._pil_image = img
             return self._pil_image
    
         except Exception as e:
             logging.error(f"Failed to load image from URL for {self.id}: {e}")
             self._pil_image = None
             raise

    def load_mask(
            self, 
            mask_type: Literal['semantic', 'instance'],
            force_reload: bool = False,
            custom_path: Path | str | None = None
        ) -> np.ndarray:
        """
        Lazy-loads the segmentation mask from disk into a NumPy array.

        This method handles caching automatically. It checks if the mask is 
        already loaded in memory (e.g. `_semantic_mask`) before attempting 
        file I/O.

        Args:
            mask_type (str): 
                The type of mask to load. Standard values are 'semantic' or 
                'instance'. Defaults to 'semantic'.
            force_reload (bool): 
                If ``True``, ignores the internal cache and re-reads the file 
                from disk. Defaults to ``False``.
            custom_path (Path | str | None): 
                If provided, loads the mask from this specific path instead 
                of deriving the filename from the image asset's path.

        Returns:
            np.ndarray: 
                The segmentation mask data as a NumPy array.

        Raises:
            FileNotFoundError: 
                If the mask file does not exist at the calculated or provided 
                path.
            Exception: 
                If the file exists but cannot be opened or decoded 
                (e.g. corruption).
        """        
        
        # Select cache dynamically based on the string provided:
        if mask_type == "semantic":
            cache_attr = "_semantic_mask"
        elif mask_type == "instance":
            cache_attr = "_instance_mask"
        else:
            logging.warning(
                f"Mask type '{mask_type}' has no dedicated cache. Reloading "
                'from the disk.'
            )
            cache_attr = None
        
        # Return cached if available:
        if cache_attr:
            current_val = getattr(self, cache_attr)
            if current_val is not None and not force_reload:
                return current_val

        # Locate file:
        mask_path = self.get_mask_path(mask_type, override_path=custom_path)
        
        if not mask_path.exists():
            raise FileNotFoundError(
                f'No {mask_type} mask found at {mask_path}'
                )

        try:
            with PillowImage.open(mask_path) as img:
                mask_data = np.array(img)
            
            # Cache it only if we have a slot for it
            if cache_attr:
                setattr(self, cache_attr, mask_data)
                
            return mask_data

        except Exception as e:
            logging.error(
                f'Failed to load {mask_type} mask for {self.id}: {e}'
            )
            raise
             
    def set_mask(
            self, 
            data: np.ndarray, 
            mask_type: Literal['semantic', 'instance'],
            map_data: dict[int, Any] | None = None
        ) -> None:
        """
        Manually attaches a segmentation mask from memory.
        
        Use this when you have generated a mask in code 
        (e.g., model inference) instead of loading it from a file.

        Args:
            data: 
                The array containing the mask.
            mask_type: 
                Mask type, i.e., 'semantic' or 'instance'.
            map_data: 
                Optional dictionary to update the class/instance mapping.
                If provided, it REPLACES the existing map for this asset.
        """
        # Store the array in the cache:
        if mask_type == "semantic":
            self._semantic_mask = data
            # Update the map if provided:
            if map_data is not None:
                self.semantic_map = map_data
                
        elif mask_type == "instance":
            self._instance_mask = data
            # Update the map if provided:
            if map_data is not None:
                self.instance_map = map_data
        else:
        # This handles runtime typos like "instnce" or "sem_seg":
            logging.warning(
                f"Skipping set_mask: Unknown mask type '{mask_type}'. "
                "Only 'semantic' and 'instance' are supported."
            )
    
    def save_image(
            self, 
            output_path: Path | str | None = None, 
            format: str | None = None,
            quality: int = 100
        ) -> None:
        """
        Saves the currently cached PIL image to disk.

        This method writes the in-memory image data to a file. It automatically 
        creates any necessary parent directories.

        Args:
            output_path (Path | str | None): 
                The file path where the image will be saved. If ``None``, 
                defaults to ``self.path``, which will **overwrite** the 
                original image file.
            format (str | None): 
                The image format to use (e.g., 'JPEG', 'PNG'). If ``None``, 
                the format is inferred from the file extension of the 
                output path.
            quality (int): 
                The compression quality (1-100) for lossy formats like JPEG. 
                Higher numbers mean better quality but larger file size. 
                Defaults to 100.

        Raises:
            ValueError: 
                If no image is currently loaded in memory (i.e., 
                ``load_image_from_disk()``, ``load_image_from_url`` has not 
                been called).
        """
        if self._pil_image is None:
            raise ValueError(
                'No image loaded in memory. Please call '
                "'load_image_from_disk()' or 'load_image_from_url()' before"
                ' saving.'
            )
        
        # Extract the target save path:
        target_path = Path(output_path) if output_path else self.path
        
        # Ensure parent directory exists:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Saving image to {target_path}")
        self._pil_image.save(target_path, format=format, quality=quality)

    def save_mask(
            self, 
            mask_type: Literal['semantic', 'instance'],
            output_path: Path | str | None = None
        ) -> None:
        """
        Saves the specified segmentation mask to disk as a PNG image.

        Args:
            mask_type: 
                'semantic' or 'instance'.
            output_path: 
                Where to save. Defaults to standard naming convention 
                (e.g., image_semantic.png).

        Raises:
            ValueError: If the requested mask is not loaded.
        """
        # Ensure the data exists (this checks cache and disk):
        try:
            mask_data = self.load_mask(mask_type)
        except FileNotFoundError:
            raise ValueError(f'Cannot save {mask_type} mask: No data found.')

        target_path = Path(output_path) if output_path else self.get_mask_path(mask_type)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert NumPy array to PIL Image
        # For segmentation, we want exact pixel values, so we stick to PNG.
        img = PillowImage.fromarray(mask_data)
        
        logging.info(f'Saving {mask_type} mask to {target_path}')
        img.save(target_path, format='PNG')

    def show(
            self,
            output_type: Literal[
                'image', 
                'semantic', 
                'instance', 
                'overlay_semantic',  # Explicit name (Default)
                'overlay_instance'
            ] = 'overlay_semantic',
            alpha: float = 0.5,
            cmap: str | None = None,
        ) -> None:
        """
        Show the asset in the system's default image viewer.

        Args:
            output_type:
                - 'image': Show raw image only.
                - 'semantic': Show semanOpenstic mask only (colored).
                - 'instance': Show instance mask only.
                - 'overlay_semantic': Image + Semantic Mask.
                - 'overlay_instance': Image + Instance Mask.
            alpha:
                Transparency of the mask in overlay modes (0.0 to 1.0), where
                0 is fully transparent, 0.5 is 50% Transparent (see-through),
                and 1 is a solid color (blocks anything underneath it)
            cmap:
                Matplotlib colormap name (e.g., 'tab20', 'jet').
        """
        # Load the base image:
        raw_img = self.load_image_from_disk()

        # Helper to colorize a mask:
        def get_colored_mask(mask_kind, default_cmap):
            try:
                mask_data = self.load_mask(mask_kind)
            except FileNotFoundError:
                logging.warning(f'No {mask_kind} mask found to show.')
                return None

            # Get colormap and apply to data
            cmap_name = cmap if cmap else default_cmap
            colormap = cm.get_cmap(cmap_name)
            
            # Apply colormap. Note that if you have > 255 instances, this 
            # modulo wrap-around ensures the code does not crash, though colors
            # will repeat:
            colored_arr = colormap(mask_data % 256) 
            
            # Convert to (H, W, 4) uint8:
            colored_uint8 = (colored_arr * 255).astype(np.uint8)
            
            # Create PIL Image:
            mask_img = PillowImage.fromarray(colored_uint8, mode='RGBA')
            
            # Handle Transparency:
            # Set alpha=0 where data=0 (Background)
            # Set alpha=user_value where data>0 (Objects)
            mask_alpha_arr = np.array(mask_img.split()[3])
            mask_alpha_arr[mask_data == 0] = 0
            
            if alpha < 1.0:
                # Apply transparency to the visible parts
                visible_pixels = mask_data > 0
                mask_alpha_arr[visible_pixels] = (
                    mask_alpha_arr[visible_pixels] * alpha
                ).astype(np.uint8)
                
            mask_img.putalpha(PillowImage.fromarray(mask_alpha_arr))
            return mask_img

        # Select and generate visualization based on output type:
        final_image = None

        if output_type == 'image':
            final_image = raw_img

        elif output_type == 'semantic':
            final_image = get_colored_mask('semantic', 'tab20')
            
        elif output_type == 'instance':
            final_image = get_colored_mask('instance', 'nipy_spectral')

        elif output_type in ['overlay_semantic', 'overlay_instance']:
            # Convert the base image to RGBA to composite it with the mask:
            base_rgba = raw_img.convert('RGBA')
            
            # Create a colored mask image for the requested mask type:
            mask_type = 'semantic' if output_type == 'overlay_semantic' else \
                'instance'
            DEFAULT_CMAP = 'tab20' if mask_type == 'semantic' else \
                'nipy_spectral'
            mask_img = get_colored_mask(mask_type, DEFAULT_CMAP)
            
            # Create a composit of base image + mask:
            if mask_img:
                final_image = PillowImage.alpha_composite(base_rgba, mask_img)
            else:
                logging.warning(
                    f'Could not create {output_type}: mask unavailable. '
                    'Displaying raw image instead.'
                )
                final_image = base_rgba

        # Display the image:
        if final_image:
            final_image.show(title=f'{self.filename} - {output_type}')
    
    
class ImageCollection:
    """
    A container class for managing a group of ImageAsset objects.

    This class provides utilities for filtering, batch downloading, and 
    exporting metadata for a collection of images.
    """

    def __init__(self, assets: list[ImageAsset] | None = None) -> None:
        """
        Initializes the ImageCollection.

        Args:
            assets: An optional list of ImageAsset objects to initialize the 
                collection with. Defaults to an empty list.
        """
        self._assets: list[ImageAsset] = assets if assets else []

    def __repr__(self) -> str:
        return f"<ImageCollection containing {len(self)} image assets>"

    def __len__(self) -> int:
        return len(self._assets)

    def __iter__(self) -> Iterator[ImageAsset]:
        return iter(self._assets)

    def __getitem__(self, index: int) -> ImageAsset:
        return self._assets[index]

    def add(
        self, 
        items: ImageAsset | list[ImageAsset], 
        overwrite: bool = False
    ) -> None:
        """
        Adds one or more assets to the collection.

        This method accepts either a single ImageAsset object or a list of 
        objects. It checks for duplicates based on the asset ID.

        Args:
            items: A single ImageAsset object or a list of ImageAsset objects
                to add.
            overwrite: If True, existing assets with the same ID will be 
                replaced by the new ones. If False, duplicates are ignored 
                (the existing asset is kept). Defaults to False.
        """
        # Normalize input to a list for uniform processing:
        assets_to_add = items if isinstance(items, list) else [items]
        
        # Create a lookup map of {id: index} for the current collection:
        id_map = {asset.id: i for i, asset in enumerate(self._assets)}

        for asset in assets_to_add:
            # Handle assets with no ID (append without checking):
            if asset.id is None:
                self._assets.append(asset)
                continue

            if asset.id in id_map:
                if overwrite:
                    # Replace the element at the specific index:
                    index = id_map[asset.id]
                    self._assets[index] = asset
                else:
                    # Log that we are ignoring the duplicate:
                    logging.info(
                        f'Skipping duplicate asset with ID: {asset.id}'
                    )
            else:
                # Add new asset:
                self._assets.append(asset)
                # Update map to handle duplicates within the input list itself:
                id_map[asset.id] = len(self._assets) - 1

    def combine_with(
            self, 
            other: 'ImageCollection', 
            overwrite_path: bool = False,
            overwrite_properties: bool = True,
            add_new: bool = False
        ) -> None:
            """
            Merge with another ImageCollection.
    
            Args:
                other (ImageCollection): 
                    The collection to merge.
                overwrite_path (bool): 
                    If ``True``, existing assets will update their file paths
                    to match the incoming assets.
                overwrite_properties (bool): 
                    If ``True``, incoming properties overwrite existing ones.
                add_new (bool):
                    If ``True``, assets in ``other`` that do not share an ID
                    with any asset in will be added to the collection. 
                    If False, they are ignored. Defaults to False.
            """
            # Create a lookup map for the current assets for fast access
            # We ignore None IDs here because we cannot reliably match them.
            current_map = {
                asset.id: asset 
                for asset in self._assets 
                if asset.id is not None
            }
            
            count_merged = 0
            count_added = 0
            count_skipped = 0
    
            for incoming_asset in other:
                # If a match is found, merge:
                if incoming_asset.id is not None and \
                    incoming_asset.id in current_map:
                    existing_asset = current_map[incoming_asset.id]
                    existing_asset.combine_with(
                        incoming_asset,
                        ignore_id_mismatch=False,
                        overwrite_path=overwrite_path,
                        overwrite_properties=overwrite_properties
                    )
                    count_merged += 1
                
                # If there is no Match (or ID is None) and add_new is True:
                elif add_new:
                    self._assets.append(incoming_asset)
                    # Update map in case 'other' has duplicates of this new ID:
                    if incoming_asset.id is not None:
                        current_map[incoming_asset.id] = incoming_asset
                    count_added += 1
                
                # If there is no Match (or ID is None) and add_new is False:
                else:
                    count_skipped += 1
    
            logging.info(
                f'Merge complete: {count_merged} merged, {count_added} added, '
                f'{count_skipped} skipped.'
            )

    def filter(self, func: Callable[[ImageAsset], bool]) -> ImageCollection:
        """
        Filters the collection using a custom function.

        Args:
            func: A callable that takes an ImageAsset as input and returns 
                True if the asset should be kept, or False otherwise.

        Returns:
            A new ImageCollection instance containing only the assets for 
            which the function returned True.

        Example:
            >>> subset = collection.filter(lambda x: x.properties.get('damage') == 'High')
        """
        filtered_assets = [asset for asset in self._assets if func(asset)]
        return ImageCollection(filtered_assets)

    def filter_by_property(self, key: str, value: Any) -> ImageCollection:
        """
        Filters assets where a specific property matches a given value.

        Args:
            key: The property key to check (e.g., 'event_name').
            value: The value the property must match.

        Returns:
            A new ImageCollection containing the matching assets.
        """
        return self.filter(lambda a: a.properties.get(key) == value)

    def filter_downloaded(self) -> ImageCollection:
        """
        Filters for assets that currently exist on disk.

        Returns:
            A new ImageCollection containing only assets where 
            `is_downloaded` is True.
        """
        return self.filter(lambda a: a.is_downloaded)

    def filter_by_bbox(self, bbox: BoundingBox) -> ImageCollection:
        """
        Filters images that fall within a geographic bounding box.

        This method filters the collection to include only images whose 
        coordinates (longitude/latitude) fall within the provided BoundingBox.

        Args:
            bbox: The BoundingBox object defining the geographic area.

        Returns:
            A new ImageCollection containing only the assets within the bounds.
        """
        # Extract bounds from the internal shapely geometry of the BoundingBox:
        min_lon, min_lat, max_lon, max_lat = bbox.shapely.bounds

        def inside_box(asset: ImageAsset) -> bool:
            lat = asset.properties.get('latitude')
            lon = asset.properties.get('longitude')
            
            # If coordinates are missing, exclude the image
            if lat is None or lon is None:
                return False
                
            return (min_lat <= lat <= max_lat) and (min_lon <= lon <= max_lon)

        return self.filter(inside_box)

    def get_ids(self, ignore_none: bool = True) -> list[str | int | None]:
        """
        Retrieves a list of IDs for all assets in the collection.

        Args:
            ignore_none: If ``True``, assets with no ID (None) are excluded 
                         from the list. Defaults to ``True``.

        Returns:
            list: A list of IDs corresponding to the assets.
        """
        if ignore_none:
            return [asset.id for asset in self._assets if asset.id is not None]
        return [asset.id for asset in self._assets]

    def subset(self, indices: list[int]) -> 'ImageCollection':
        """
        Creates a new ImageCollection containing assets at the specific indices.

        Args:
            indices: A list of integer indices (e.g., [0, 5, 10]).

        Returns:
            A new ImageCollection instance.
        
        Raises:
            IndexError: If any index is out of bounds.
        """
        # Extract the assets using list comprehension:
        selected_assets = [self._assets[i] for i in indices]
        
        # Return a new instance of the class:
        return ImageCollection(selected_assets)

    def download_all(self, max_workers: int = 5, overwrite: bool = False) -> None:
        """
        Downloads all images in the collection using multi-threading.

        Args:
            max_workers: The maximum number of parallel download threads. 
                Defaults to 5.
            overwrite: If True, existing files will be re-downloaded and 
                overwritten. Defaults to False.
        """
        total = len(self)
        logging.info(
            f'Starting batch download for {total} images with {max_workers} '
            'workers.'
            )

        # Create a single session to reuse TCP connections
        with requests.Session() as session:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Map futures to assets for error reporting
                future_to_asset = {
                    executor.submit(
                        asset.download,
                        overwrite=overwrite,
                        session=session
                    ): asset
                    for asset in self._assets
                }

                for i, future in enumerate(as_completed(future_to_asset)):
                    asset = future_to_asset[future]
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"Failed to download {asset.id}: {e}")

                    # Optional: Print progress every 10 images
                    if (i + 1) % 10 == 0:
                        logging.info(f"Progress: {i + 1}/{total} completed.")

    def to_dataframe(self) -> Any:
        """
        Converts the collection metadata to a Pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame where each row represents an asset.
                Columns include 'id', 'path', 'is_downloaded', and all keys
                found in the asset properties.

        Raises:
            ImportError: If the pandas library is not installed.
        """
        data = []
        for asset in self._assets:
            # Flattens structure: ID, Path + all Properties
            row = {
                'id': asset.id,
                'path': str(asset.path),
                'is_downloaded': asset.is_downloaded,
                **asset.properties
            }
            data.append(row)

        return pd.DataFrame(data)

    def to_json(self, filepath: str | Path) -> None:
        """
        Exports the collection metadata to a JSON file.

        Args:
            filepath: The path where the JSON file will be saved.
        """
        data = [
            {
                'id': a.id,
                'path': str(a.path),
                'properties': a.properties
            }
            for a in self._assets
        ]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

