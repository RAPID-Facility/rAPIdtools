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

from dataclasses import dataclass, field
from typing import Any, Iterator

# Import the other domain objects
from .image_asset import ImageAsset
from .bounding_box import BoundingBox

# Import shapely for geometry handling
try:
    from shapely.geometry.base import BaseGeometry
    from shapely.geometry import shape
    from shapely.ops import unary_union
except ImportError:
    BaseGeometry = None
    unary_union = None


@dataclass(kw_only=True)
class InfrastructureAsset:
    """
    Represents a real-world, built-environment asset, such as a building, bridge, or road.

    An InfrastructureAsset combines geometric information (its location and shape), a set of
    descriptive attributes, and a list of related image assets.

    Attributes:
        id (str): A unique identifier for the asset.
        geometry (BaseGeometry): The geographic shape of the asset, represented
            by a shapely object (e.g., Polygon, LineString, Point).
        attributes (dict[str, Any]): A flexible dictionary to hold descriptive
            properties of the asset (e.g., {'asset_type': 'Building', 'height': 50}).
        image_assets (list[ImageAsset]): A list of ImageAsset objects associated
            with this asset.
    """
    id: str
    geometry: BaseGeometry
    attributes: dict[str, Any] = field(default_factory=dict)
    image_assets: list[ImageAsset] = field(default_factory=list)

    def __post_init__(self):
        """Validates input after initialization."""
        if BaseGeometry is None:
            raise ImportError("Shapely library is required. Please run 'pip install shapely'.")
        
        if not isinstance(self.geometry, BaseGeometry):
            raise TypeError(f"The 'geometry' attribute must be a valid shapely geometry object, not {type(self.geometry)}.")
        
        if not self.id:
            raise ValueError("InfrastructureAsset 'id' cannot be empty.")

    @property
    def asset_type(self) -> str | None:
        """A convenience property to get the asset type from attributes."""
        return self.attributes.get("asset_type")

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Safely retrieves a descriptive attribute from the asset."""
        return self.attributes.get(key, default)

    def add_image_asset(self, image_asset: ImageAsset):
        """Associates a new ImageAsset with this asset."""
        if not isinstance(image_asset, ImageAsset):
            raise TypeError("Can only add ImageAsset objects to an InfrastructureAsset.")
        self.image_assets.append(image_asset)

    @classmethod
    def from_geojson_feature(cls, geojson_feature: dict[str, Any]) -> "InfrastructureAsset":
        """
        A factory method to create an InfrastructureAsset object from a GeoJSON Feature dictionary.
        """
        if BaseGeometry is None:
            raise ImportError("Shapely library is required. Please run 'pip install shapely'.")
        
        if geojson_feature.get("type") != "Feature":
            raise ValueError("Input dictionary must be a valid GeoJSON Feature.")
            
        asset_id = geojson_feature.get("id") or str(geojson_feature.get("properties", {}).get("id", "no_id"))
        
        return cls(
            id=asset_id,
            geometry=shape(geojson_feature["geometry"]),
            attributes=geojson_feature.get("properties", {})
        )


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

    def filter(self, attribute_key: str, attribute_value: Any) -> "InfrastructureAssetCollection":
        """
        Filters the collection and returns a new InfrastructureAssetCollection.
        """
        filtered_assets = [
            asset for asset in self.assets 
            if asset.get_attribute(attribute_key) == attribute_value
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