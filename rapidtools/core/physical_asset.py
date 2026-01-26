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

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Iterator

from tqdm import tqdm
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .bounding_box import BoundingBox
from .image_asset import ImageAsset, ImageCollection

@dataclass(kw_only=True, repr=False)
class PhysicalAsset:
    """
    Represents a tangible, real-world entity.

    A ``PhysicalAsset`` serves as the digital twin for a real-world object 
    (e.g., a utility pole, building, or road segment). It acts as a container
    combining:
    1.  **Spatial Data**: Where it is located (Shapely geometry).
    2.  **Metadata**: What describes it (Attributes dictionary).
    3.  **Media**: Visual records of it (List of ImageAssets).

    This distinguishes the physical object itself from the digital media 
    (``ImageAsset``) used to document it.    

    Example:
        To create a simple ``PhysicalAsset`` for a utility pole:
        
        >>> from shapely.geometry import Point
        >>> from rapidtools.core import PhysicalAsset
        >>> 
        >>> my_asset = PhysicalAsset(
        ...     id='pole_123',
        ...     geometry=Point(10.0, 20.0),
        ...     attributes={'type': 'utility_pole', 'material': 'wood'}
        ... )
    """
    id: str
    geometry: BaseGeometry
    attributes: dict[str, Any] = field(default_factory=dict)
    image_assets: ImageCollection = field(default_factory=ImageCollection)

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
            raise ValueError("PhysicalAsset 'id' cannot be empty.")
            
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
        and returns the value of the first key found. Returns ``None`` if none
        of the keys are present.
        
        Example:
            Create an asset using the key ``'category'`` for its type:
            
            >>> from shapely.geometry import Point
            >>> from rapidtools.core import PhysicalAsset
            >>> asset = PhysicalAsset(
            ...     id='bldg_01',
            ...     geometry=Point(0, 0),
            ...     attributes={'category': 'commercial', 'floors': 5}
            ... )
            
            The property automatically finds 'commercial' despite the key 
            being 'category':
                
            >>> print(asset.asset_type)
            commercial
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
        Add or update the asset’s attributes using values from a dictionary.
    
        By default, existing keys are not overwritten. Set ``overwrite=True``
        to replace existing values.
    
        Args:
            new_attributes (dict[str, Any]): 
                A mapping of attribute names to values to add.
            overwrite (bool): 
                If ``True``, existing attributes with the same keys
                will be overwritten. If ``False``, existing keys will be left
                unchanged and a log message will be emitted.
    
        Raises:
            TypeError: If new_attributes is not a dictionary.
            
        Example:
            Initialize an asset with some base attributes:
            
            >>> from shapely.geometry import Point
            >>> from rapidtools.core import PhysicalAsset
            >>> asset = PhysicalAsset(
            ...     id='utility_pole_01', 
            ...     geometry=Point(0, 0), 
            ...     attributes={'material': 'wood', 'status': 'active'}
            ... )
            
            >>> Add new attributes that do not overlap with the existing ones:
            >>> asset.add_attributes({'install_date': '2023-01-01'})
            >>> print(asset.attributes['install_date'])
            2023-01-01
            
            Attempt to add attributes where some keys are already present in 
            the existing set without specifying the ``overwrite`` argument 
            (i.e., default behavior):
            
            >>> asset.add_attributes({'material': 'metal', 'height': 12.0})
            INFO: Attribute 'material' already exists in asset 
            'utility_pole_01'. Skipping attribute as 'overwrite is set to 
            ``False``.
            >>> print(asset.attributes['material'])
            wood
            >>> print(asset.attributes['height'])
            12.0
            
            Force update existing attributes by setting the ``overwrite`` 
            argument to ``True``:
                
            >>> asset.add_attributes({'status': 'retired'}, overwrite=True)
            >>> print(asset.attributes['status'])
            retired
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

    def add_image_assets(self, *args: Any) -> None:
        """
        Add one or more ``ImageAsset`` objects to the asset.
    
        This method is highly flexible and accepts:
        - Individual ``ImageAsset`` objects.
        - Lists or Tuples of ``ImageAsset`` objects.
        - ``ImageCollection`` instances.
        - A mix of all the above.
        
        Any nested item that is not an ``ImageAsset`` is ignored and a 
        warning is logged.
    
        Args:
            *args (Any): 
                Variable length argument list containing images or collections
                of images.
            
        Examples:
            
            Initialize the main physical asset:
                
            >>> from shapely.geometry import Point
            >>> from rapidtools.core import PhysicalAsset, ImageAsset, \
            >>>     ImageCollection
            >>> 
            >>> asset = PhysicalAsset(id='pole_01', geometry=Point(10, 20))

            
            Initialize several ``ImageAsset`` and ``ImageCollection`` objects
            to use in the examples below. At minimum, an ``ImageAsset`` 
            requires an id and a path. Because we will be using mock paths, we
            must set ``allow_missing_file=True``; otherwise, ``ImageAsset`` 
            will validate the path, detect that it does not exist, and raise
            an error. To avoid repeating the path and ``allow_missing_file`` 
            arguments, we will define a small helper function, named 
            ``make_img`` to construct ``ImageAsset`` instances consistently:

            >>> def make_img(uid):
            ...     return ImageAsset(
            ...         id=uid, 
            ...         path=f'/tmp/{uid}', 
            ...         allow_missing_file=True
            ...     )
            >>>
            >>> img1 = make_img('img_1.jpg')
            >>> img2 = make_img('img_2.jpg')
            >>> img3 = make_img('img_3.jpg')
            >>> img4 = make_img('img_4.jpg')
            >>> img5 = make_img('img_5.jpg')
            >>> img6 = make_img('img_6.jpg')
            >>> img7 = make_img('img_7.jpg')
            >>> collection = ImageCollection([img4, img5])

            Add individual ImageAssets to the physical asset:
            
            >>> asset.add_image_assets(img1, img2)
            >>> len(asset.image_assets)
            2

            Add a list of images:
            
            >>> asset.add_image_assets([img3])
            >>> len(asset.image_assets)
            3

            Add a mix of a single ``ImageAsset``, list, tuple, and 
            ``ImageCollection`` in one call:

            >>> asset.add_image_assets(
            ...     img5,               # Single Object
            ...     [img6],             # List
            ...     (img7,),            # Tuple
            ...     collection          # ImageCollection (contains img4, img5)
            ... )
            INFO: Skipping duplicate asset with ID: img_5.jpg
            >>> 'img_7.jpg' in asset.image_assets.get_ids()
            True
        """
        valid_assets_to_add = []

        for arg in args:
            # Case 1 - Single ImageAsset:
            if isinstance(arg, ImageAsset):
                valid_assets_to_add.append(arg)
            
            # Case 2 - A List, Tuple, or ImageCollection
            # Check for these types specifically to avoid iterating over 
            # strings or other iterables:
            elif isinstance(arg, (list, tuple, ImageCollection)):
                for item in arg:
                    if isinstance(item, ImageAsset):
                        valid_assets_to_add.append(item)
                    else:
                        logging.warning(
                            f"Ignored item of type '{type(item).__name__}' "
                            f'found inside a container argument. This method'
                            " supports adding 'ImageAsset' objects only."
                        )
            
            # Unsupported top-level argument:
            else:
                logging.warning(
                    f"Ignored argument of type '{type(arg).__name__}'. "
                    'This method supports ImageAsset, list, tuple, or '
                    'ImageCollection.'
                )

        # Batch add everything to the internal collection:
        if valid_assets_to_add:
            self.image_assets.add(valid_assets_to_add)

    @classmethod
    def from_geojson_feature(
            cls, 
            geojson_feature: dict[str, Any],
            asset_id: str | None = None
            ) -> PhysicalAsset:
        """
        Create a PhysicalAsset from a GeoJSON Feature dictionary.
    
        The method expects a GeoJSON object of type "Feature" with at least a 
        "geometry" field containing a valid GeoJSON geometry. 
        
        If the feature's properties contain an 'image_assets' list (as produced
        by ``to_geojson_feature``), this method attempts to reconstruct the 
        corresponding ``ImageAsset`` objects and populate the asset's image 
        collection.
    
        If no id is found at the top level or in properties, a warning is
        logged and a random UUID is generated as a placeholder.
    
        Args:
            geojson_feature (dict[str, Any]): 
                A dictionary representing a GeoJSON Feature.
            asset_id (str | None, optional):
                An explicit ID to assign to the asset. If provided, this 
                overrides any ID found within the GeoJSON data. Defaults to 
                None.
    
        Returns:
            An initialized PhysicalAsset instance.
    
        Raises:
            ValueError: If geojson_feature is not a valid GeoJSON Feature
                (i.e., its "type" is not "Feature").
            
        Examples:
            Create a PhysicalAsset from a GeoJSON feature:
            
            >>> from rapidtools.core import PhysicalAsset
            >>> 
            >>> feature = {
            ...     "type": "Feature",
            ...     "id": "pole_01",
            ...     "geometry": {"type": "Point", "coordinates": [10, 20]},
            ...     "properties": {
            ...         "material": "wood",
            ...         "image_assets": [
            ...             {"id": "img1", "path": "/tmp/a.jpg", 
            ...              "allow_missing_file": True}
            ...         ]
            ...     }
            ... }
            >>>
            >>> asset = PhysicalAsset.from_geojson_feature(feature)

            Display the asset’s ID and material properties. Note that 
            ``image_assets`` is removed from the GeoJSON feature attributes
            and stored instead in the asset’s ``ImageCollection``:
                
            >>> print(asset.id)
            pole_01
            >>> print(asset.attributes['material'])
            wood
            >>> print('image_assets' in asset.attributes)
            False
            >>> print(len(asset.image_assets))
            1
        """       
        if geojson_feature.get('type') != 'Feature':
            raise ValueError(
                'Input dictionary must be a valid GeoJSON Feature.'
            )
        
        # Ensure properties is a dict (it can be None in valid GeoJSON):
        properties = (geojson_feature.get('properties') or {}).copy()

        # Determine asset ID:
        if asset_id is None:
            # Check top-level ID, then properties ID:
            asset_id = geojson_feature.get('id') or properties.get('id')
        
        if asset_id is None:
            generated_id = f"no_id_{uuid.uuid4().hex[:8]}"
            logging.debug(
                "GeoJSON feature is missing an 'id' field. Generated "
                f"placeholder '{generated_id}'."
            )
            asset_id = generated_id
        
        # Extract and Rehydrate ImageAssets. Note that this key from the 
        # general attributes dict to eliminate duplication:
        image_data_list = properties.pop('image_assets', [])
        rehydrated_images = []
        
        if image_data_list:
            for img_dict in image_data_list:
                try:
                    # Reconstruct ImageAsset from dictionary kwargs
                    # This step always sets 'allow_missing_file' is True if not
                    # specified, as rehydrating from JSON often implies offline
                    # processing:
                    if 'allow_missing_file' not in img_dict:
                        img_dict['allow_missing_file'] = True
                        
                    img_obj = ImageAsset(**img_dict)
                    rehydrated_images.append(img_obj)
                except Exception as e:
                    logging.warning(
                        f"Failed to rehydrate an image asset from GeoJSON for "
                        f"asset '{asset_id}': {e}"
                    )

        # Process asset geometry:
        geom_data = geojson_feature.get('geometry')
        if not geom_data:
            raise ValueError(f"Feature '{asset_id}' is missing geometry.")
            
        geom_obj = shape(geom_data)

        # Simplify MultiPolygon to Polygon if it only has one component:
        if geom_obj.geom_type == 'MultiPolygon' and len(geom_obj.geoms) == 1:
            geom_obj = geom_obj.geoms[0]
            
        # Initialize Asset:
        new_asset = cls(
            id=str(asset_id),
            geometry=geom_obj,
            attributes=properties
        )
        
        # Add rehydrated images:
        if rehydrated_images:
            new_asset.image_assets.add(rehydrated_images)
            
        return new_asset
    
    def get_attributes(self, *keys: str) -> dict[str, Any]:
        """
        Safely retrieves one or more descriptive attributes from the asset.

        If a requested key does not exist, a warning is logged and the key
        is omitted from the returned dictionary.

        Args:
            *keys (str): A variable number of keys to retrieve.

        Returns:
            A dictionary containing the keys that were found and their values.
            
        Examples:
            
            Initialize the main physical asset:
            
            >>> from shapely.geometry import Point
            >>> from rapidtools.core import PhysicalAsset
            >>> 
            >>> asset = PhysicalAsset(
            ...     id='bldg_001', 
            ...     geometry=Point(0,0), 
            ...     attributes={'year_built': 1990, 'material': 'concrete'}
            ... )
    
            Retrieve a single attribute:
            
            >>> asset.get_attributes('year_built')
            {'year_built': 1990}
    
            Request multiple attributes. Please note that ``roof_type`` is 
            missing, so it is omitted in the result:
            
            >>> asset.get_attributes('year_built', 'roof_type', 'material')
            WARNING: Attribute key 'roof_type' not found for asset 'bldg_001'.
            {'year_built': 1990, 'material': 'concrete'}
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

    def get_image_assets(self, *identifiers: str) -> list[ImageAsset]:
        """
        Retrieve specific image assets by their ID or filename.

        This method searches the asset's image collection. It matches against 
        the 'id' attribute or the 'filename' (explicit or derived from path).

        Args:
            *identifiers (str): A variable number of string IDs or filenames 
                to retrieve.

        Returns:
            list[ImageAsset]: A list of the found ImageAsset objects. 
            If a requested identifier is not found, it is skipped and a 
            warning is logged.

        Examples:
            Setup asset with images:

            >>> from rapidtools.core import PhysicalAsset, ImageAsset
            >>> from shapely.geometry import Point
            >>> 
            >>> asset = PhysicalAsset(id='pole_01', geometry=Point(0,0))
            >>> img1 = ImageAsset(id='img_A', path='/tmp/photo_A.jpg', 
            ...                   allow_missing_file=True)
            >>> img2 = ImageAsset(id='img_B', path='/tmp/photo_B.jpg',
            ...                   allow_missing_file=True)
            >>> asset.add_image_assets(img1, img2)

            Get by ID:

            >>> found = asset.get_image_assets('img_A')
            >>> found[0].path
            PosixPath('/tmp/photo_A.jpg')

            Get by Filename:

            >>> found = asset.get_image_assets('photo_B.jpg')
            >>> found[0].id
            'img_B'

            Get multiple (mixed ID and filename):

            >>> found = asset.get_image_assets('img_A', 'photo_B.jpg')
            >>> len(found)
            2

            Attempt to get non-existent image:

            >>> asset.get_image_assets('missing.jpg')
            WARNING: Image asset 'missing.jpg' not found in asset 'pole_01'.
            []
        """
        results = []

        # Build a temporary lookup map for efficient retrieval.
        # This handles cases where one image might be referenced by ID and 
        # another by filename in the same call.
        lookup = {}
        for img in self.image_assets:
            # Map ID
            if img.id:
                lookup[img.id] = img
            
            # Map Filename
            # Try explicit 'filename' property:
            fname = getattr(img, 'filename', None)
            
            # Fallback: Derive from 'path':
            if not fname:
                raw_path = getattr(img, 'path', None)
                if raw_path:
                    fname = Path(raw_path).name
            
            if fname:
                lookup[fname] = img
        
        for key in identifiers:
            if key in lookup:
                results.append(lookup[key])
            else:
                logging.warning(
                    f'Image asset \'{key}\' not found in asset \'{self.id}\'.'
                )
        
        return results

    def print_info(self) -> None:
        """
        Print a human-readable summary of the asset information.

        The summary includes:
        - Identity (ID and inferred type).
        - Spatial data (WKT geometry, truncated if excessively long).
        - Metadata (Attributes pretty-printed as JSON).
        - Media (List of associated ImageAsset representations).

        Example:
            >>> from shapely.geometry import Point
            >>> from rapidtools.core import PhysicalAsset
            >>> 
            >>> asset = PhysicalAsset(
            ...     id='pole_99',
            ...     geometry=Point(10, 20),
            ...     attributes={'material': 'wood', 'install_year': 2021}
            ... )
            >>> 
            >>> asset.print_info()      
            --- PhysicalAsset Summary ---
            ID:          pole_99
            Asset Type:  N/A
            Geometry:    POINT (10 20)
            Attributes (2):
            {
                "material": "wood",
                "install_year": 2021
            }
            Image Assets (0):
              (No image assets)
            -----------------------------
        """
        header = f'--- {self.__class__.__name__} Summary ---'
        print(f'\n{header}')
        print(f'ID:          {self.id}')
        # Inner quotes for "N/A" must be double since the outer quotes are single
        print(f'Asset Type:  {self.asset_type or "N/A"}')

        # Truncate geometry string if it is too long:
        wkt_str = self.geometry.wkt
        if len(wkt_str) > 80:
            wkt_str = wkt_str[:77] + '...'
        print(f'Geometry:    {wkt_str}')

        # Display attributes:
        print(f'Attributes ({len(self.attributes)}):')
        if self.attributes:
            # default=str handles objects like datetime or UUIDs gracefully
            print(json.dumps(self.attributes, indent=4, default=str))
        else:
            print('  (No attributes)')

        # Display Image Assets
        print(f'Image Assets ({len(self.image_assets)}):')
        if self.image_assets:
            for img in self.image_assets:
                print(f'  - {img!r}')
        else:
            print('  (No image assets)')

        print('-' * len(header) + '\n')

    def remove_attributes(self, *keys: str) -> None:
        """
        Remove one or more attributes from the asset by key.

        For each provided key:
            
        - If the key exists, it is removed.
        - If the key does not exist, a warning is logged to indicate a 
          potential typo or logic error in the caller.

        Args:
            *keys (str): The names of the attributes to remove.

        Examples:
            Setup a physical asset with initial attributes:
            
            >>> from shapely.geometry import Point
            >>> from rapidtools.core import PhysicalAsset
            >>> 
            >>> asset = PhysicalAsset(
            ...     id='bldg_123',
            ...     geometry=Point(0, 0),
            ...     attributes={'color': 'red', 'height': 50, 'year': 1990}
            ... )

            Remove a single existing attribute:
            
            >>> asset.remove_attributes('color')
            >>> 'color' in asset.attributes
            False
            
            Attempt to remove a non-existent attribute:
                
            >>> asset.remove_attributes('width')
            WARNING: Attempted to remove non-existent attribute 'width' from
            asset 'bldg_123'.
            
            Attempt to remove multiple (mixed existing and missing) attributes
            at once:
                
            >>> asset.remove_attributes('height', 'invalid_key')
            WARNING: Attempted to remove non-existent attribute 'invalid_key'
            from asset 'bldg_123'.
            >>> print(asset.attributes)
            {'year': 1990}
        """
        for key in keys:
            if key in self.attributes:
                del self.attributes[key]
                logging.debug(
                    f'Removed attribute \'{key}\' from asset \'{self.id}\'.'
                )
            else:
                logging.warning(
                    f'Attempted to remove non-existent attribute \'{key}\' '
                    f'from asset \'{self.id}\'.'
                )

    def remove_image_assets(self, *image_ids: str) -> list[ImageAsset]:
        """
        Removes image assets identified by their unique string ID or filename.
        
        This method searches the asset's image collection for matches. It 
        checks both the 'id' and 'file_name' fields of the image assets.

        Args:
            *image_ids: 
                A variable number of string IDs (or filenames) identifying 
                the images to remove.

        Returns:
            list[ImageAsset]: 
                A list containing the actual ImageAsset objects that were
                successfully found and removed.
                
        Examples:
            Setup asset with images:
                
            >>> from rapidtools.core import PhysicalAsset, ImageAsset
            >>> from shapely.geometry import Point
            >>>
            >>> asset = PhysicalAsset(id='pole_01', geometry=Point(0,0))
            >>> img1 = ImageAsset(id='img_A', path='/tmp/photo_A.jpg', 
            ... allow_missing_file=True)
            >>> img2 = ImageAsset(id='img_B', path='/tmp/photo_B.jpg',
            ... allow_missing_file=True)
            >>> img3 = ImageAsset(id='img_C', path='/tmp/photo_C.jpg',
            ... allow_missing_file=True)
            >>> asset.add_image_assets(img1, img2, img3)
            
            Remove by ID:
                
            >>> removed = asset.remove_image_assets('img_A')
            INFO: Removed 1 asset from the collection.
            >>> len(asset.image_assets)
            2
            >>> removed[0].id
            'img_A'
            
            Remove by filename (assuming file_name is derived from path):
                
            >>> removed = asset.remove_image_assets('photo_B.jpg')
            INFO: Removed 1 asset from the collection.
            >>> len(asset.image_assets)
            1

            Attempt to remove non-existent image:
                
            >>> asset.remove_image_assets('ghost_image.jpg')
            WARNING: Attempted to remove image 'ghost_image.jpg' from asset 
            'pole_01', but no matching asset was found.
            []
        """
        removed_items = []

        for target_id in image_ids:
            asset_to_remove = None
            
            # Iterate through the current assets to find a match:
            for img in self.image_assets:
                # Check against 'id' or 'file_name' attributes safely:
                if target_id in (getattr(img, 'id', None), 
                                 getattr(img, 'filename', None)):
                    asset_to_remove = img
                    break  # Stop after finding the first match

            if asset_to_remove:
                self.image_assets.remove(asset_to_remove)
                removed_items.append(asset_to_remove)
                logging.debug(
                    f"Successfully removed image asset '{target_id}' from "
                    f"asset '{self.id}'."
                )
            else:
                logging.warning(
                    f"Attempted to remove image '{target_id}' from asset "
                    f"'{self.id}', but no matching asset was found."
                )
        
        return removed_items

    def to_geojson_feature(self) -> dict[str, Any]:
        """
        Converts the PhysicalAsset into a GeoJSON Feature dictionary.

        The asset's 'id' becomes the feature's 'id'. The asset's 'attributes'
        are used as the base for the feature's 'properties'. The list of
        'image_assets' is serialized and added to the 'properties' as well.

        Returns:
            A dictionary representing a valid GeoJSON Feature.
            
        Examples:
            Create asset with an image:
            
            >>> from shapely.geometry import Point
            >>> from rapidtools.core import PhysicalAsset, ImageAsset
            >>> 
            >>> asset = PhysicalAsset(
            ...     id='pole_99', 
            ...     geometry=Point(10, 20),
            ...     attributes={'material': 'wood'}
            ... )
            >>> img = ImageAsset(id='img1', path='p1.jpg',
            ... allow_missing_file=True)
            >>> asset.add_image_assets(img)
            
            Convert to GeoJSON:
            
            >>> feature = asset.to_geojson_feature()
            >>> print(feature)
            {'type': 'Feature', 'id': 'pole_99', 'geometry': 
            {'type': 'Point', 'coordinates': (10.0, 20.0)}, 
            'properties': {'material': 'wood', 'image_assets': 
            [{'path': PosixPath('/home/bacetiner/p1.jpg'), 'id': 'img1', 
            'properties': {}, 'semantic_map': None, 'instance_map': None,
            'allow_missing_file': True, '_pil_image': None, '_semantic_mask':
            None, '_instance_mask': None}]}}
        """
        # Start with a copy of the attributes for the properties dictionary:
        properties = self.attributes.copy()
        
        # Serialize image_assets into a list of dictionaries and add to 
        # properties:
        if self.image_assets:
            properties['image_assets'] = \
                [asdict(img) for img in self.image_assets]

        return {
            'type': 'Feature',
            'id': self.id,
            'geometry': mapping(self.geometry),  # Convert shapely to GeoJSON
            'properties': properties
        }

@dataclass
class PhysicalAssetCollection:
    """
    Represents a collection of PhysicalAsset objects.

    This class acts as a "smart list," providing methods to manage and analyze
    a group of assets as a whole. It supports list-like operations such as
    iteration, len(), and indexing.
    """
    assets: list[PhysicalAsset] = field(default_factory=list)

    def __post_init__(self):
        if unary_union is None:
            raise ImportError("Shapely library is required. Please run 'pip install shapely'.")
        ids = [asset.id for asset in self.assets]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate asset IDs found. All asset IDs in a collection must be unique.")

    def __len__(self) -> int:
        """Returns the number of assets in the collection."""
        return len(self.assets)

    def __iter__(self) -> Iterator[PhysicalAsset]:
        """Allows iterating over the assets in the collection."""
        return iter(self.assets)

    def __getitem__(self, index: int) -> PhysicalAsset:
        """Allows accessing an asset by its index."""
        return self.assets[index]

    def append(self, asset: PhysicalAsset):
        """
        Adds a new PhysicalAsset to the collection, checking for a unique ID.
        """
        if not isinstance(asset, PhysicalAsset):
            raise TypeError(
                'Only PhysicalAsset objects can be added to the collection.'
            )
        if asset.id in {a.id for a in self.assets}:
            raise ValueError(
                f"PhysicalAsset with ID '{asset.id}' already exists.")
        self.assets.append(asset)

    def get_asset_by_id(self, asset_id: str) -> PhysicalAsset | None:
        """Finds and returns an asset by its unique ID."""
        for asset in self.assets:
            if asset.id == asset_id:
                return asset
        return None

    def filter_by_attribute(
            self, 
            attribute_key: str, 
            attribute_value: Any
        ) -> 'PhysicalAssetCollection':
        """
        Filters the collection and returns a new PhysicalAssetCollection.
        """
        filtered_assets = [
            asset for asset in self.assets 
            if asset.attributes.get(attribute_key) == attribute_value
        ]
        return PhysicalAssetCollection(assets=filtered_assets)

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

    def from_geojson(
            self, 
            source: str | Path | dict[str, Any]
        ) -> 'PhysicalAssetCollection':
        """
        Populate the collection with assets from a GeoJSON source.

        Includes a progress bar and O(N) optimization for bulk imports.

        Args:
            source: GeoJSON dictionary or file path.

        Returns:
            The current instance (self).
        """
        # Load Data:
        if isinstance(source, (str, Path)):
            with open(source, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
        elif isinstance(source, dict):
            geojson_data = source
        else:
            raise TypeError('Input must be a file path or a dictionary.')

        if geojson_data.get('type') != 'FeatureCollection':
            raise ValueError(
                'Input data must be a valid GeoJSON FeatureCollection.'
            )
        
        features = geojson_data.get('features', [])
        
        # Create a set of existing IDs:
        existing_ids = {asset.id for asset in self.assets}
        
        new_assets = []

        # Iterate over each GeoJSON feature and convert it into an asset:
        for feature_geojson in tqdm(
                features, 
                desc='Importing Assets', 
                unit='asset'
            ):

            # Try to reuse an existing ID from the feature if present. We 
            # first check a top-level "id" field, then fall back to 
            # "properties.id":
            existing_id = feature_geojson.get('id')
            if existing_id is None:
                existing_id = feature_geojson.get('properties', {}).get('id')

            if existing_id is None:
                # If No ID found in the source data: generate a new, unique one
                # Prefix with "gen_" to mark it as system-generated:
                assigned_id = f'gen_{uuid.uuid4().hex}'
            else:
                # Normalize to string for consistent ID handling:
                assigned_id = str(existing_id)

            # If this ID has already been seen in the current import session,
            # skip this feature to avoid creating duplicate assets:
            if assigned_id in existing_ids:
                logging.warning(
                    f"Skipping duplicate asset with ID '{assigned_id}' found "
                    'in GeoJSON input.'
                )
                continue

            # Pass the resolved ID explicitly so the PhysicalAsset constructor
            # does not have to infer or override it:
            asset = PhysicalAsset.from_geojson_feature(
                feature_geojson,
                asset_id=assigned_id,
            )

            # Track the new asset and remember that this ID has been used:
            existing_ids.add(assigned_id)
            new_assets.append(asset)

        # Extend the main list once at the end
        self.assets.extend(new_assets)

        return self
    
    def to_geojson(
        self,
        file: str | Path | None = None,
        indent: int | None = 2,
    ) -> dict:
        """
        Serialize the collection into a GeoJSON FeatureCollection.
    
        By default, the output is pretty-printed with an indentation of 2 
        spaces. Callers can change or disable pretty-printing by passing a 
        different `indent` value (e.g., `indent=None` for compact output).
    
        Args:
            file:
                Optional path to a file where the GeoJSON should be written. If
                None, nothing is written to disk.
            indent:
                Number of spaces for JSON indentation when writing to a file.
                Use None for compact output. Defaults to 2.
    
        Returns:
            dict: The GeoJSON FeatureCollection as a dictionary.
        """
        feature_collection: dict = {
            "type": "FeatureCollection",
            "features": [asset.to_geojson_feature() for asset in self.assets],
        }
    
        if file is not None:
            path = Path(file)
            with path.open("w", encoding="utf-8") as f:
                json.dump(feature_collection, f, indent=indent)
    
        return feature_collection