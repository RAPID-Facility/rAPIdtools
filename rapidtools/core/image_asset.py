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
# 12-09-2025

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
                If ``True``, updates self.path with other.path. 
                Defaults to ``False`` (preserves original path).
            overwrite_properties (bool): 
                If ``True``, other.properties will overwrite existing keys in 
                self.properties. If ``False``, only new keys are added.
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
            self._segmentation_mask = other._segmentation_mask
            
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

