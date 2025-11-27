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
# 11-26-2025

from dataclasses import dataclass, field, asdict
import json
import logging
from typing import Any, ClassVar, Iterator

from shapely.geometry.base import BaseGeometry
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

from .image_asset import ImageAsset
from .bounding_box import BoundingBox

@dataclass(kw_only=True, repr=False)
class InfrastructureAsset:
    """
    Represents a real-world, built-environment asset, such as a building, etc.

    An InfrastructureAsset combines geometric information (its location and 
    shape), a set of descriptive attributes, and a list of related image 
    assets.
    """
    id: str
    geometry: BaseGeometry
    attributes: dict[str, Any] = field(default_factory=dict)
    image_assets: list[ImageAsset] = field(default_factory=list)

    # Define the keys to search as a class attribute for easy configuration:
    _ASSET_TYPE_KEYS: ClassVar[list[str]] = [
        'asset_type', 'type', 'category', 'feature_type'
    ]

    def __post_init__(self):
        """
        Validate initialization parameters.
        
        Ensures that:
        - ``geometry`` is a Shapely BaseGeometry instance.
        - ``id`` is not empty.
        
        Raises:
            TypeError: If ``geometry`` is not a Shapely BaseGeometry.
            ValueError: If ``id`` is empty.
        """     
        if not isinstance(self.geometry, BaseGeometry):
            raise TypeError(
                "The 'geometry' attribute must be a valid shapely geometry"
                f' object, not {type(self.geometry)}.'
                )
        
        if not self.id:
            raise ValueError("InfrastructureAsset 'id' cannot be empty.")
            
        if not self.geometry.is_valid:
            logging.warning(
            f"Asset '{self.id}' contains invalid geometry (e.g., "
            f'self-intersection).'
        )
    
    def __repr__(self) -> str:
        """
        Return a concise, developer-friendly string representation.
    
        The representation includes:
        - the class name,
        - the asset id,
        - the inferred asset_type (if any), and
        - the geometry type.
    
        Returns:
            A compact string representation of the asset.
        """
        class_name = self.__class__.__name__
        geom_type = self.geometry.geom_type
        asset_type = self.asset_type or 'N/A'
        
        return (
            f"<{class_name} id='{self.id}' "
            f"asset_type='{asset_type}' "
            f"geometry='{geom_type}'>"
        )

    @property
    def asset_type(self) -> str | None:
        """
        A convenience property to get the asset type from attributes.

        It searches a predefined list of keys (e.g., 'asset_type', 'type')
        and returns the value of the first key found. Returns None if none
        of the keys are present.
        """
        for key in self._ASSET_TYPE_KEYS:
            # Check if the key exists in the attributes dictionary:
            if key in self.attributes:
                # If found, return its value immediately:
                return self.attributes[key]
        
        # If the loop finishes without finding any of the keys, return None:
        return None
    
    def add_attributes(
            self, 
            new_attributes: dict[str, Any], 
            overwrite: bool = False
        ) -> None:
        """
        Add or update attributes on the asset from a dictionary.
    
        By default, existing keys are not overwritten. Set ``overwrite=True``
        to replace existing values.
    
        Args:
            new_attributes (dict[str, Any]): 
                A mapping of attribute names to values to add.
            overwrite (bool): 
                If True, existing attributes with the same keys
                will be overwritten. If False, existing keys will be left
                unchanged and a log message will be emitted.
    
        Raises:
            TypeError: If new_attributes is not a dictionary.
        """
        if not isinstance(new_attributes, dict):
            raise TypeError("Input 'new_attributes' must be a dictionary.")
    
        for key, value in new_attributes.items():
            # Check if the key already exists
            if key in self.attributes:
                if overwrite:
                    # If overwrite is True, update the value and log it for traceability
                    self.attributes[key] = value
                    logging.debug(
                        f"Overwrote attribute '{key}' in asset '{self.id}'."
                    )
                else:
                    # If overwrite is False, skip and inform the user
                    logging.info(
                        f"Attribute '{key}' already exists in asset "
                        f"'{self.id}'. Skipping attribute as 'overwrite is set"
                        'to False.'
                    )
            else:
                # If the key is new, simply add it
                self.attributes[key] = value

    def add_image_assets(self, *image_assets: Any) -> None:
        """
        Associate one or more ImageAsset instances with this asset.
    
        Any argument that is not an instance of ImageAsset is ignored and
        a warning is logged. Valid ImageAsset instances are appended to
        the image_assets list.
    
        Args:
            *image_assets (Any): One or more objects, of which only ImageAsset
                instances will be added.
        """
        # Create a temporary list to hold only the valid assets:
        valid_assets_to_add = []

        # Iterate through all provided arguments:
        for asset in image_assets:
            if isinstance(asset, ImageAsset):
                # If the item is a valid ImageAsset, add it to our list.
                valid_assets_to_add.append(asset)
            else:
                # If the item is NOT a valid ImageAsset, log a warning.
                # Provide context: which asset is being updated and what the 
                # invalid type was:
                logging.warning(
                    f"Skipping invalid item for asset '{self.id}'. "
                    f"Expected ImageAsset, but got {type(asset).__name__}."
                )

        # If found any valid assets, extend the main list:
        if valid_assets_to_add:
            self.image_assets.extend(valid_assets_to_add)


    
    def get_attributes(self, *keys: str) -> dict[str, Any]:
        """
        Safely retrieves one or more descriptive attributes from the asset.

        If a requested key does not exist, a warning is logged and the key
        is omitted from the returned dictionary.

        Args:
            *keys (str): A variable number of keys to retrieve.

        Returns:
            A dictionary containing the keys that were found and their values.
        """
        found_attributes = {}
        for key in keys:
            # Check if the key exists in the asset's attributes
            if key in self.attributes:
                found_attributes[key] = self.attributes[key]
            else:
                # If not, log a warning with helpful context (the asset ID)
                logging.warning(
                    f"Attribute key '{key}' not found for asset '{self.id}'." 
                )
        return found_attributes

    @classmethod
    def from_geojson_feature(
            cls, 
            geojson_feature: dict[str, Any]
            ) -> 'InfrastructureAsset':
        """
        Create an InfrastructureAsset from a GeoJSON Feature dictionary.
    
        The method expects a GeoJSON object of type "Feature" with at least:
        - a "geometry" field containing a valid GeoJSON geometry, and
        - an optional "id" field, or an "id" inside "properties".
    
        If no id is found at the top level or in properties, a warning is
        logged and the id 'no_id' is used as a placeholder.
    
        Args:
            geojson_feature (dict[str, Any]): 
                A dictionary representing a GeoJSON Feature.
    
        Returns:
            An initialized InfrastructureAsset instance.
    
        Raises:
            ValueError: If geojson_feature is not a valid GeoJSON Feature
                (i.e., its "type" is not "Feature").
        """       
        if geojson_feature.get('type') != 'Feature':
            raise ValueError(
                'Input dictionary must be a valid GeoJSON Feature.'
            )
        
        # Try to get the ID from the top-level 'id' field:
        asset_id = geojson_feature.get('id')
        
        # If not found, try to get it from the 'properties' dictionary:
        if asset_id is None:
            properties = geojson_feature.get('properties', {})
            asset_id = properties.get('id')
        
        # If still not found, issue a warning and use a placeholder.
        if asset_id is None:
            # This message will be displayed because its level (WARNING) is >= INFO
            props = geojson_feature.get('properties', {})
            logging.warning(
                "GeoJSON feature is missing an 'id'. Using placeholder "
                f"'no_id'. Properties: {props}"
            )
            asset_id = 'no_id'
        
        return cls(
            id=str(asset_id),
            geometry=shape(geojson_feature['geometry']),
            attributes=geojson_feature.get('properties', {}).copy()
        )

    def print_info(self) -> None:
        """
        Print a human-readable summary of the asset to stdout.
    
        The summary includes:
        - id,
        - inferred asset type,
        - geometry (in WKT format),
        - all attributes (pretty-printed as JSON), and
        - a list of associated image assets.
        """
        # Use a helper to pretty-print dictionaries
        def pretty_dict(d: dict) -> str:
            return json.dumps(d, indent=4)

        print("\n--- Infrastructure Asset Summary ---")
        print(f"ID:          {self.id}")
        print(f"Asset Type:  {self.asset_type or 'N/A'}")
        print(f"Geometry:    {self.geometry.wkt}")  # WKT is a readable format

        # Display Attributes
        attr_count = len(self.attributes)
        print(f"Attributes ({attr_count}):")
        if self.attributes:
            print(pretty_dict(self.attributes))
        else:
            print("  (No attributes)")

        # Display Image Assets
        img_count = len(self.image_assets)
        print(f"Image Assets ({img_count}):")
        if self.image_assets:
            for img in self.image_assets:
                # Assumes ImageAsset has a nice __repr__
                print(f"  - {img!r}")
        else:
            print("  (No image assets)")

        print("------------------------------------\n")

    def remove_attributes(self, *keys: str) -> None:
        """
        Remove one or more attributes by key.
    
        For each key:
        - If the key exists, it is removed.
        - If the key does not exist, a warning is logged.
    
        Args:
            *keys (str): One or more attribute names to remove.
        """
        for key in keys:
            if key in self.attributes:
                del self.attributes[key]
                logging.debug(f"Removed attribute '{key}' from asset '{self.id}'.")
            else:
                logging.warning(
                    f"Attempted to remove non-existent attribute '{key}' "
                    f"from asset '{self.id}'."
                )

    def remove_image_assets(self, *image_ids: str) -> list[ImageAsset]:
        """
        Removes image assets identified by their unique string ID or filename.

        Args:
            *image_ids: 
                A variable number of string IDs (or filenames) to remove.

        Returns:
            list[ImageAsset]: A list of the actual ImageAsset objects that were 
            successfully removed.
        """
        removed_items = []

        for target_id in image_ids:
            asset_to_remove = None
            
            # Iterate through the current assets to find a match
            for img in self.image_assets:
                # Check against 'id' or 'file_name' attributes safely
                # (We use getattr to avoid crashes if the ImageAsset lacks these fields)
                if target_id in (getattr(img, 'id', None), 
                                 getattr(img, 'file_name', None)):
                    asset_to_remove = img
                    break  # Stop after finding the first match

            if asset_to_remove:
                self.image_assets.remove(asset_to_remove)
                removed_items.append(asset_to_remove)
                logging.debug(
                    f"Successfully removed image asset '{target_id}' from asset '{self.id}'."
                )
            else:
                logging.warning(
                    f"Attempted to remove image '{target_id}' from asset '{self.id}', "
                    "but no matching asset was found."
                )
        
        return removed_items

    def to_geojson_feature(self) -> dict[str, Any]:
        """
        Converts the InfrastructureAsset into a GeoJSON Feature dictionary.

        The asset's 'id' becomes the feature's 'id'. The asset's 'attributes'
        are used as the base for the feature's 'properties'. The list of
        'image_assets' is serialized and added to the 'properties' as well.

        Returns:
            A dictionary representing a valid GeoJSON Feature.
        """
        # Start with a copy of the attributes for the properties dictionary
        properties = self.attributes.copy()
        
        # Serialize image_assets into a list of dictionaries and add to properties
        # This assumes ImageAsset is a dataclass, making asdict a perfect tool.
        if self.image_assets:
            properties['image_assets'] = [asdict(img) for img in self.image_assets]

        return {
            'type': 'Feature',
            'id': self.id,
            'geometry': mapping(self.geometry),  # Convert shapely to GeoJSON
            'properties': properties
        }

@dataclass
class InfrastructureAssetCollection:
    """
    Represents a collection of InfrastructureAsset objects.

    This class acts as a "smart list," providing methods to manage and analyze
    a group of assets as a whole. It supports list-like operations such as
    iteration, len(), and indexing.
    """
    assets: list[InfrastructureAsset] = field(default_factory=list)

    def __post_init__(self):
        if unary_union is None:
            raise ImportError("Shapely library is required. Please run 'pip install shapely'.")
        ids = [asset.id for asset in self.assets]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate asset IDs found. All asset IDs in a collection must be unique.")

    def __len__(self) -> int:
        """Returns the number of assets in the collection."""
        return len(self.assets)

    def __iter__(self) -> Iterator[InfrastructureAsset]:
        """Allows iterating over the assets in the collection."""
        return iter(self.assets)

    def __getitem__(self, index: int) -> InfrastructureAsset:
        """Allows accessing an asset by its index."""
        return self.assets[index]

    def append(self, asset: InfrastructureAsset):
        """
        Adds a new InfrastructureAsset to the collection, checking for a unique ID.
        """
        if not isinstance(asset, InfrastructureAsset):
            raise TypeError("Only InfrastructureAsset objects can be added to the collection.")
        if asset.id in {a.id for a in self.assets}:
            raise ValueError(f"InfrastructureAsset with ID '{asset.id}' already exists.")
        self.assets.append(asset)

    def get_asset_by_id(self, asset_id: str) -> InfrastructureAsset | None:
        """Finds and returns an asset by its unique ID."""
        for asset in self.assets:
            if asset.id == asset_id:
                return asset
        return None

    def filter_by_attribute(self, attribute_key: str, attribute_value: Any) -> "InfrastructureAssetCollection":
        """
        Filters the collection and returns a new InfrastructureAssetCollection.
        """
        filtered_assets = [
            asset for asset in self.assets 
            if asset.attributes.get(attribute_key) == attribute_value
        ]
        return InfrastructureAssetCollection(assets=filtered_assets)

    @property
    def combined_bounding_box(self) -> BoundingBox | None:
        """
        Calculates the total bounding box that encloses all assets in the collection.
        """
        if not self.assets:
            return None
        
        geometries = [asset.geometry.buffer(0) for asset in self.assets]
        combined_geom = unary_union(geometries)
        
        min_lon, min_lat, max_lon, max_lat = combined_geom.bounds
        return BoundingBox(lon1=min_lon, lat1=min_lat, lon2=max_lon, lat2=max_lat)

    @classmethod
    def from_geojson(cls, geojson_data: dict) -> "InfrastructureAssetCollection":
        """
        A factory method to create a collection from a GeoJSON FeatureCollection dictionary.
        """
        if geojson_data.get("type") != "FeatureCollection":
            raise ValueError("Input must be a valid GeoJSON FeatureCollection.")
        
        assets = [
            InfrastructureAsset.from_geojson_feature(feature_geojson)
            for feature_geojson in geojson_data.get("features", [])
        ]
        return cls(assets=assets)
    
    def to_geojson(self, indent: int | None = None) -> str:
        """
        Serializes the entire collection into a GeoJSON FeatureCollection string.
    
        Args:
            indent (int | None): The number of spaces to use for indentation
                for pretty-printing. If None, the output will be compact.
                Defaults to None.
    
        Returns:
            A string containing the GeoJSON FeatureCollection.
        """
        feature_collection = {
            'type': 'FeatureCollection',
            'features': [asset.to_geojson_feature() for asset in self.assets]
        }
        return json.dumps(feature_collection, indent=indent)