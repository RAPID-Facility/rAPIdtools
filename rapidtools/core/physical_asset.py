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

from __future__ import annotations

import json
import logging
import math
import operator as op
import uuid
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import InitVar, asdict, dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
import shapefile
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry

from .bounding_box import BoundingBox
from .image_asset import ImageAsset, ImageCollection
from .polygon_region import PolygonRegion


@dataclass(kw_only=True, repr=False)
class PhysicalAsset:
    """
    Represents a tangible, real-world entity.

    A ``PhysicalAsset`` serves as the digital twin for a real-world object
    (e.g., a utility pole, building, or road segment). It acts as a container
    combining:

        1. Spatial Data: Where it is located (Shapely geometry).
        2. Metadata: What describes it (Attributes dictionary).
        3. Media: Visual records of it (List of ImageAssets).

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
        'asset_type',
        'type',
        'category',
        'feature_type',
    ]

    def __post_init__(self):
        """
        Validate and normalize fields after dataclass initialization.

        This hook performs lightweight runtime checks that are not enforced by
        type hints at runtime:

        - ``id`` must be a non-empty, non-whitespace ``str``.
        - ``geometry`` must be a Shapely ``BaseGeometry`` instance.
        - If ``geometry`` is present but invalid, a warning is logged.

        Raises:
            TypeError:
                If ``id`` is not a ``str`` or if ``geometry`` is not a Shapely
                geometry.
            ValueError:
                If ``id`` is empty or whitespace only.
        """
        if not isinstance(self.id, str):
            raise TypeError(
                "PhysicalAsset 'id' must be a string, got " f"{type(self.id).__name__}."
            )

        if not self.id.strip():
            raise ValueError("PhysicalAsset 'id' cannot be empty or whitespace.")

        if not isinstance(self.geometry, BaseGeometry):
            raise TypeError(
                "The 'geometry' attribute must be a valid shapely geometry"
                f" object, not {type(self.geometry)}."
            )

        if not self.geometry.is_valid:
            logging.warning(
                f"Asset '{self.id}' contains invalid geometry (e.g., "
                f"self-intersection)."
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
            str: A compact string representation of the asset.
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

        It searches a predefined list of keys (e.g., ``'asset_type'``, ``'type'``)
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

            The property automatically finds ``'commercial'`` despite the key
            being ``'category'``:

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
        self, new_attributes: dict[str, Any], overwrite: bool = False
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
            TypeError: If ``new_attributes`` is not a dictionary.

        Example:
            Initialize an asset with some base attributes:

            >>> from shapely.geometry import Point
            >>> from rapidtools.core import PhysicalAsset
            >>> asset = PhysicalAsset(
            ...     id='utility_pole_01',
            ...     geometry=Point(0, 0),
            ...     attributes={'material': 'wood', 'status': 'active'}
            ... )

            Add new attributes that do not overlap with the existing ones:

            >>> asset.add_attributes({'install_date': '2023-01-01'})
            >>> print(asset.attributes['install_date'])
            2023-01-01

            Attempt to add attributes where some keys are already present in
            the existing set without specifying the ``overwrite`` argument
            (i.e., default behavior):

            >>> asset.add_attributes({'material': 'metal', 'height': 12.0})
            INFO: Attribute 'material' already exists in asset
            'utility_pole_01'. Skipping attribute as 'overwrite is set to False.
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
                    logging.debug(f"Overwrote attribute '{key}' in asset '{self.id}'.")
                else:
                    # If overwrite is False, skip and inform the user
                    logging.info(
                        f"Attribute '{key}' already exists in asset "
                        f"'{self.id}'. Skipping attribute as 'overwrite is set"
                        " to False."
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

            Add individual ``ImageAssets`` to the physical asset:

            >>> asset.add_image_assets(img1, img2)
            >>> len(asset.image_assets)
            2

            Add a list of images:

            >>> asset.add_image_assets([img3])
            >>> len(asset.image_assets)
            3

            Add a mix of a single ``ImageAsset``, list, tuple, and
            ``ImageCollection`` in one call. Please note that 'img5.jpg' is
            skipped below as it is a duplicate asset:

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
            arg_type = type(arg).__name__

            # Case 1 - Single ImageAsset:
            if isinstance(arg, ImageAsset) or arg_type == 'ImageAsset':
                valid_assets_to_add.append(arg)

            # Case 2 - A List, Tuple, or ImageCollection
            # Check for these types specifically to avoid iterating over
            # strings or other iterables:
            elif isinstance(arg, (list, tuple)) or arg_type == 'ImageCollection':
                for item in arg:
                    if (
                        isinstance(item, ImageAsset)
                        or type(item).__name__ == 'ImageAsset'
                    ):
                        valid_assets_to_add.append(item)
                    else:
                        logging.warning(
                            f"Ignored item of type '{type(item).__name__}' "
                            f"found inside a container argument. This method"
                            " supports adding 'ImageAsset' objects only."
                        )

            # Unsupported top-level argument:
            else:
                logging.warning(
                    f"Ignored argument of type '{type(arg).__name__}'. "
                    "This method supports ImageAsset, list, tuple, or "
                    "ImageCollection."
                )

        # Batch add everything to the internal collection:
        if valid_assets_to_add:
            self.image_assets.add(valid_assets_to_add)

    @classmethod
    def from_geojson_feature(
        cls, geojson_feature: dict[str, Any], asset_id: str | None = None
    ) -> PhysicalAsset:
        """
        Create a ``PhysicalAsset`` from a GeoJSON feature dictionary.

        The method expects a GeoJSON object of type "Feature" with at least a
        "geometry" field containing a valid GeoJSON geometry.

        If the feature's properties contain an ``'image_assets'`` list (as produced
        by ``to_geojson_feature``), this method attempts to reconstruct the
        corresponding ``ImageAsset`` objects and populate the asset's image
        collection.

        If no id is found at the top level or in properties, a warning is
        logged and a random UUID is generated as a placeholder.

        Args:
            geojson_feature (dict[str, Any]):
                A dictionary representing a GeoJSON feature.
            asset_id (str or None, optional):
                An explicit ID to assign to the asset. If provided, this
                overrides any ID found within the GeoJSON data. Defaults to
                None.

        Returns:
            PhysicalAsset: An initialized physical asset instance.

        Raises:
            ValueError: If geojson_feature is not a valid GeoJSON feature
                (i.e., its "type" is not "Feature").

        Examples:
            Create a ``PhysicalAsset`` from a GeoJSON feature:

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
            raise ValueError('Input dictionary must be a valid GeoJSON Feature.')

        # Ensure properties is a dict (it can be None in valid GeoJSON):
        properties = (geojson_feature.get('properties') or {}).copy()

        # Determine asset ID:
        if asset_id is None:
            # Check top-level ID, then properties ID:
            asset_id = geojson_feature.get('id') or properties.get('id')

        if asset_id is None:
            generated_id = f'no_id_{uuid.uuid4().hex[:8]}'
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
        new_asset = cls(id=str(asset_id), geometry=geom_obj, attributes=properties)

        # Add rehydrated images:
        if rehydrated_images:
            new_asset.image_assets.add(rehydrated_images)

        return new_asset

    def get_attributes(self, *keys: str) -> dict[str, Any]:
        """
        Retrieve one or more descriptive attributes from the asset.

        If a requested key does not exist, a warning is logged and the key
        is omitted from the returned dictionary.

        Args:
            *keys (str): A variable number of keys to retrieve.

        Returns:
            dict[str, Any]:
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

            Request multiple attributes. Please note that ``'roof_type'`` is
            missing, so it is omitted in the result:

            >>> asset.get_attributes('year_built', 'roof_type', 'material')
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
        the ``'id'`` attribute or the ``'filename'`` (explicit or derived from path).

        Args:
            *identifiers (str):
                A variable number of string IDs or filenames to retrieve.

        Returns:
            list[ImageAsset]:
                A list of the found ``ImageAsset`` objects. If a requested
                identifier is not found, it is skipped and a warning is logged.

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

            Get by filename:

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
        # another by filename in the same call:
        lookup = {}
        for img in self.image_assets:
            # Map ID if present:
            if img.id:
                lookup[img.id] = img

            # Map filename (always present on ImageAsset):
            lookup[img.filename] = img

        for key in identifiers:
            if key in lookup:
                results.append(lookup[key])
            else:
                logging.warning(f"Image asset '{key}' not found in asset '{self.id}'.")

        return results

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
            >>> print(asset.attributes)
            {'height': 50, 'year': 1990}

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
                logging.debug(f"Removed attribute '{key}' from asset '{self.id}'.")
            else:
                logging.warning(
                    f"Attempted to remove non-existent attribute '{key}' "
                    f"from asset '{self.id}'."
                )

    def remove_image_assets(self, *image_ids: str) -> list[ImageAsset]:
        """
        Removes image assets identified by their unique string ID or filename.

        This method searches the asset's image collection for matches. It
        checks both the ``'id'`` and ``'file_name'`` fields of the image assets.

        Args:
            *image_ids:
                A variable number of string IDs (or filenames) identifying
                the images to remove.

        Returns:
            list[ImageAsset]:
                A list containing the actual ``ImageAsset`` objects that were
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
            >>> len(asset.image_assets)
            2
            >>> removed[0].id
            'img_A'

            Remove by filename (assuming ``file_name`` is derived from path):

            >>> removed = asset.remove_image_assets('photo_B.jpg')
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
                if target_id in (
                    getattr(img, 'id', None),
                    getattr(img, 'filename', None),
                ):
                    asset_to_remove = img
                    break  # Stop after finding the first match

            if asset_to_remove:
                _ = self.image_assets._remove_core(asset_to_remove)
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

    def summary(self) -> None:
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
            >>> asset.summary()
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

    def to_geojson_feature(
        self, ignore_properties: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Converts the asset into a GeoJSON feature dictionary.

        The asset's ``'id'`` becomes the feature's ``'id'``. The asset's
        'attributes' are used as the base for the feature's 'properties'.
        The list of ``'image_assets'`` is serialized and added to the
        'properties' by default.

        Specific properties can be excluded from the output using the
        ``ignore_properties`` argument, which is useful for reducing file
        size when large nested objects are not required.

        Args:
            ignore_properties (list[str] or None, optional):
                A list of property keys to exclude from the GeoJSON
                properties dictionary. Commonly used to exclude
                ``'image_assets'``. Defaults to ``None``.

        Returns:
            dict[str, Any]: A dictionary representing a valid GeoJSON Feature.

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
        ignore_list = ignore_properties or []

        # Start with a copy of the attributes
        properties = self.attributes.copy()

        # Serialize image_assets into the properties dictionary if not ignored
        if self.image_assets and 'image_assets' not in ignore_list:
            properties['image_assets'] = [asdict(img) for img in self.image_assets]

        # Remove any properties requested in the ignore_list
        for key in ignore_list:
            if key in properties:
                del properties[key]

        return {
            'type': 'Feature',
            'id': self.id,
            'geometry': mapping(self.geometry),  # Convert shapely to GeoJSON
            'properties': properties,
        }


@dataclass
class PhysicalAssetCollection:
    """
    A specialized container for collection of PhysicalAsset objects.

    This class enforces unique asset IDs and provides fast, O(1), lookups.
    Internally, it uses a dictionary to store assets, but it behaves like
    a sequence for iteration and interaction.

    Key Features:

        * Spatial Querying: Filter assets by ``BoundingBox`` or ``Polygon``.
        * Data I/O: Import/Export to GeoJSON and convert to Pandas ``DataFrames``.
        * Batch Operations: Update attributes across all assets.
        * Set Operations: Merge collections with collision handling.

    Supported Container Operations:

        * Iteration: ``for asset in collection: ...``
        * Access:

            - Key: ``collection['id_str']`` (O(1) Fast)
            - Index: ``collection[0]`` (O(N) Slower)
            - Slice: ``collection[:5]`` (Returns a new ``PhysicalAssetCollection``)
        * Assignment: ``collection['id'] = asset``
        * Deletion: ``del collection['id']``
        * Length: ``len(collection)``
        * Membership: ``if 'pole_01' in collection: ...``

    Args:
        assets (Iterable[PhysicalAsset]):
            Assets to populate. Can be a ``list``, ``tuple``, or ``generator``.
            Defaults to empty.
    """

    assets: InitVar[Iterable[PhysicalAsset]] = None

    # Internal data storage:
    _data: dict[str, PhysicalAsset] = field(init=False, default_factory=dict)

    def __post_init__(self, assets: Iterable[PhysicalAsset] | None):
        """
        Populate the collection with specified assets.

        This method loads any assets provided in ``assets`` into the internal
        storage.
        """
        if assets:
            self.add(assets)

    def __contains__(self, item: str | PhysicalAsset) -> bool:
        """True if the asset ID (str) or PhysicalAsset object is in the collection."""
        if isinstance(item, str):
            return item in self._data
        if hasattr(item, 'id'):
            return item.id in self._data
        return False

    def __delitem__(self, key: str) -> None:
        """Delete the asset with the specified ID."""
        if not isinstance(key, str):
            raise TypeError(f'Key must be a string asset ID, got {type(key).__name__}.')

        if key not in self._data:
            raise KeyError(f"Asset ID '{key}' not found.")

        del self._data[key]

    def __getitem__(
        self, key: str | int | slice
    ) -> PhysicalAsset | PhysicalAssetCollection:
        """
        x.__getitem__(y) <==> x[y]

        Supports lookup by ID (str), index (int), or slice.
        Slicing returns a new PhysicalAssetCollection.
        """
        # Case 1: Fast O(1) dictionary Lookup:
        if isinstance(key, str):
            if key not in self._data:
                raise KeyError(f"Asset ID '{key}' not found.")
            return self._data[key]

        # Case 2: Integer indexing:
        if isinstance(key, int):
            # Handle negative indexing:
            idx = key + len(self._data) if key < 0 else key

            if idx < 0 or idx >= len(self._data):
                raise IndexError('Collection index out of range.')

            # Retrieve the Nth item without building a full list:
            return next(islice(self._data.values(), idx, None))

        # Case 3: Slicing -> returns a new collection:
        if isinstance(key, slice):
            # Calculate start/stop/step handling negative indices (e.g. [:-1]):
            start, stop, step = key.indices(len(self._data))

            # Explicitly type the variable to Iterator to satisfy MyPy:
            subset: Iterator[PhysicalAsset]

            if step < 0:
                # islice cannot handle negative steps (reversing).
                # Fallback to list conversion for this specific case:
                subset = iter(list(self._data.values())[key])
            else:
                # Use iterator slicing:
                subset = islice(self._data.values(), start, stop, step)

            # Create a new instance of the same class:
            return self.__class__(assets=subset)

        raise TypeError(
            'Index must be an asset ID (str), integer, or slice. Got '
            f'{type(key).__name__}.'
        )

    def __iter__(self) -> Iterator[PhysicalAsset]:
        """Iterate over assets in insertion order."""
        return iter(self._data.values())

    def __len__(self) -> int:
        """Return the number of assets in the collection."""
        return len(self._data)

    def __setitem__(self, key: str, value: PhysicalAsset) -> None:
        """Set self[key] to value. The key must match value.id."""
        if not isinstance(key, str):
            raise TypeError(f'Key must be a string asset ID, got {type(key).__name__}.')

        if not isinstance(value, PhysicalAsset):
            raise TypeError(
                'Value must be a PhysicalAsset instance, got '
                f'{type(value).__name__}.'
            )

        # Enforce consistency to prevent data corruption:
        if value.id != key:
            raise ValueError(
                f"ID Mismatch: Key is '{key}' but asset.id is '{value.id}'."
            )

        self._data[key] = value

    @property
    def combined_bounding_box(self) -> BoundingBox | None:
        """
        Calculates the total bounding box that encloses all assets.

        Returns:
            BoundingBox:
                A ``BoundingBox`` object encompassing all assets, or ``None``
                if the collection is empty.

        Examples:
            If a ``PhysicalAssetCollection`` is empty, it's
            ``combined_bounding_box`` will be None:

            >>> from rapidtools.core import PhysicalAssetCollection
            >>> collection = PhysicalAssetCollection()
            >>> collection.combined_bounding_box is None
            True

            Combined bounds reflect the min/max extents across all assets:

            >>> from shapely.geometry import Point
            >>> from rapidtools.core import PhysicalAsset
            >>>
            >>> collection.add(PhysicalAsset(id="a1", geometry=Point(0, 0)))
            >>> collection.add(PhysicalAsset(id="a2", geometry=Point(2, 3)))
            >>> collection.add(PhysicalAsset(id="a3", geometry=Point(-1, 5)))
            >>> bbox = collection.combined_bounding_box
            >>> bbox.bounds
            (-1.0, 0.0, 2.0, 5.0)
        """
        if not self._data:
            return None

        # Track the global min/max extents across all asset geometries.
        # Start with infinities so the first asset always updates the extrema:
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')

        for asset in self._data.values():
            # Get the bounds of each asset:
            b_min_x, b_min_y, b_max_x, b_max_y = asset.geometry.bounds

            # Expand the global bounds to include this asset's bounds:
            if b_min_x < min_x:
                min_x = b_min_x
            if b_min_y < min_y:
                min_y = b_min_y
            if b_max_x > max_x:
                max_x = b_max_x
            if b_max_y > max_y:
                max_y = b_max_y

        # Construct the combined BoundingBox from the final extents:
        return BoundingBox(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)

    def add(self, assets: PhysicalAsset | Iterable[PhysicalAsset]) -> None:
        """
        Add one or more assets to the collection.

        This method accepts a single ``PhysicalAsset`` or an ``iterable`` containing
        multiple ``PhysicalAsset`` objects.

        If an asset with the same ID already exists, or if an item is not a
        ``PhysicalAsset``, a warning is logged and that specific item is
        skipped. It does not stop the addition of other valid assets.

        Args:
            assets (PhysicalAsset or Iterable[PhysicalAsset]):
                A single ``PhysicalAsset`` or an iterable of ``PhysicalAsset``
                objects.

        Examples:
            Initialize a ``PhysicalAsset``, a list of two ``PhysicalAsset``
            objects and a ``PhysicalAssetCollection``:

            >>> from shapely.geometry import Point
            >>> from rapidtools.core import PhysicalAsset, PhysicalAssetCollection
            >>>
            >>> asset1 = PhysicalAsset(id='pump_01', geometry=Point(10.0, 20.0))
            >>> assets_list = [
            ...     PhysicalAsset(id='pump_02', geometry=Point(10.1, 20.1)),
            ...     PhysicalAsset(id='pump_03', geometry=Point(10.2, 20.2))
            ... ]
            >>> collection = PhysicalAssetCollection()

            Add a single asset to the ``PhysicalAssetCollection``:

            >>> collection.add(asset1)
            >>> collection.summary()
            {'total_assets': 1,
             'total_images': 0,
             'asset_types': {'Unknown': 1},
             'bounds': BoundingBox(minx=10.0, miny=20.0, maxx=10.0, maxy=20.0)}
            Add multiple assets at once:

            >>> collection.add(assets_list)
            >>> len(collection)
            3

            Try to add a list of assets including duplicates and incorrect
            types. In the batch below, ``'pump_01'`` is a duplicate,
            ``'NotAnAsset'`` is the wrong type, so only ``'pump_04'`` is added:

            >>> mixed_batch = [
            ...     PhysicalAsset(id='pump_04', geometry=Point(10.3, 20.3)),
            ...     PhysicalAsset(id='pump_01', geometry=Point(10.0, 20.0)),
            ...     'NotAnAsset'
            ... ]
            >>> collection.add(mixed_batch)
            WARNING: Skipping duplicate asset with ID
            'pump_01' (already exists).
            WARNING: Skipping invalid item: Expected PhysicalAsset, got 'str'.
            >>> len(collection)
            4
        """
        # Normalize input so the code always iterates over a list/iterable:
        items_to_process: Iterable[PhysicalAsset]

        if isinstance(assets, PhysicalAsset):
            items_to_process = [assets]
        elif isinstance(assets, Iterable):
            items_to_process = assets
        else:
            # Handle top-level invalid input:
            logging.warning(
                "Input must be a PhysicalAsset or iterable of PhysicalAsset"
                f" objects. Received input of type: '{type(assets).__name__}'."
                " No asset imported."
            )
            return

        for asset in items_to_process:
            # Check for invalid type (includes name check for safety against
            # import mismatches):
            if (
                not isinstance(asset, PhysicalAsset)
                and type(asset).__name__ != 'PhysicalAsset'
            ):
                logging.warning(
                    f"Skipping invalid item: Expected PhysicalAsset, got "
                    f"'{type(asset).__name__}'."
                )
                continue

            # Check for duplicate ID:
            if asset.id in self._data:
                logging.warning(
                    f"Skipping duplicate asset with ID '{asset.id}' "
                    "(already exists)."
                )
                continue

            # Add to storage:
            self._data[asset.id] = asset

    def collect_all_images(self) -> ImageCollection:
        """
            Aggregate images from all assets into a single unified collection.

            This method iterates through every physical asset, extracts its
            associated images, and merges them into a new ``ImageCollection``.

            Returns:
                ImageCollection:
                    A new collection containing every image found across all
                    assets in this ``PhysicalAssetCollection``.

            Examples:
                Setup a collection where ``'pole_1'`` has 2 images, ``'pole_2'``
                has 0 images, and ``'pole_3'`` has 1 image:

                >>> from shapely.geometry import Point
                >>> from rapidtools.core import PhysicalAsset, \
                ...     PhysicalAssetCollection, ImageAsset
                >>>
                >>> a1 = PhysicalAsset(id='pole_1', geometry=Point(0, 0))
                >>> a1.image_assets.add([
                ...     ImageAsset(id='img_A',path='',allow_missing_file=True),
                ...     ImageAsset(id='img_B',path='',allow_missing_file=True)
                ... ])
                >>>
                >>> a2 = PhysicalAsset(id='pole_2', geometry=Point(1, 1))
                >>>
                >>> a3 = PhysicalAsset(id='pole_3', geometry=Point(2, 2))
                >>> a3.image_assets.add(
                ...     ImageAsset(id='img_C',
                ...     path='',
                ...     allow_missing_file=True
                ... ))
                >>>
                >>> collection = PhysicalAssetCollection([a1, a2, a3])

                Run the aggregation. Note that 'pole_2' is skipped efficiently:

                >>> all_images = collection.collect_all_images()
                INFO: Collected 3 images from 2 assets.
                >>> len(all_images)
                3
                >>> 'img_A' in all_images
                True
            """
        # Initialize a new, empty collection:
        master_collection = ImageCollection()

        # Track stats for logging:
        assets_with_images = 0

        for asset in self._data.values():
            # Skip empty collections to avoid function call overhead:
            if not asset.image_assets:
                continue

            # Merge the asset's images into the master collection
            # Here 'add_new=True' ensures all unique images are kept:
            master_collection._merge_core(
                asset.image_assets,
                add_new=True,
                overwrite_properties=False,
                overwrite_path=False,
            )
            assets_with_images += 1

        suffix = 's' if assets_with_images != 1 else ''
        logging.info(
            f'Collected {len(master_collection)} image{suffix} from '
            f'{assets_with_images} assets.'
        )

        return master_collection

    @classmethod
    def from_geojson(
        cls, source: str | Path | dict[str, Any]
    ) -> PhysicalAssetCollection:
        """
        Creates a new ``PhysicalAssetCollection`` from a GeoJSON source.

        This factory method accepts a file path (string or Path object)
        pointing to a GeoJSON file, or a dictionary object representing valid
        GeoJSON data. It parses the 'features' list and initializes a new
        collection containing the assets found.

        Args:
            source (str or Path or dict[str, Any]):
                The GeoJSON data source. Can be one of:

                    * A string path to a .geojson file.
                    * A `pathlib.Path` object pointing to a file.
                    * A Python dictionary structure representing a GeoJSON
                      FeatureCollection.

        Returns:
            PhysicalAssetCollection:
                A new instance of the collection populated with assets parsed
                from the GeoJSON source.

        Raises:
            TypeError:
                If ``source`` is not a string, Path, or dictionary.
            FileNotFoundError:
                If ``source`` is a path to a file that does not exist.
            ValueError:
                If the loaded data is not a valid GeoJSON 'FeatureCollection'.
            json.JSONDecodeError:
                If the file exists but contains invalid JSON.

        Examples:
            Import ``PhysicalAssetCollection``:

            >>> from rapidtools.core import PhysicalAssetCollection

            Loading from a file path:

            >>> path = 'data/assets.geojson'
            >>> collection = PhysicalAssetCollection.from_geojson(path)
            >>> print(len(collection))
            5

            Loading from a dictionary:

            >>> data = {
            ...     "type": "FeatureCollection",
            ...     "features": [
            ...         {
            ...             "type": "Feature",
            ...             "properties": {"id": "A1", "name": "Pump"},
            ...             "geometry": {"type": "Point", "coordinates": [0, 0]}
            ...         }
            ...     ]
            ... }
            >>> collection = PhysicalAssetCollection.from_geojson(data)
            >>> asset = collection.get("A1")
            >>> asset.summary()
            --- PhysicalAsset Summary ---
            ID:          A1
            Asset Type:  N/A
            Geometry:    POINT (0 0)
            Attributes (2):
            {
                "id": "A1",
                "name": "Pump"
            }
            Image Assets (0):
              (No image assets)
            -----------------------------
        """
        geojson_data = None

        # Load data:
        if isinstance(source, str | Path):
            file_path = Path(source)
            if not file_path.exists():
                raise FileNotFoundError(f'GeoJSON file not found at: {file_path}')

            with file_path.open('r', encoding='utf-8') as f:
                geojson_data = json.load(f)

        elif isinstance(source, dict):
            geojson_data = source

        else:
            raise TypeError(
                'Input must be a file path or a dictionary, got '
                f'{type(source).__name__}.'
            )

        # Validate structure:
        if geojson_data.get('type') != 'FeatureCollection':
            raise ValueError(
                "Input data must be a valid GeoJSON with 'type': "
                "'FeatureCollection'."
            )

        features = geojson_data.get('features', [])
        if not isinstance(features, list):
            raise ValueError("GeoJSON 'features' key must contain a list.")

        # Create and populate a collection:
        new_collection = cls()

        for i, feature in enumerate(features):
            props = feature.get('properties') or {}
            geom_dict = feature.get('geometry')

            # Check if the asset had a valid geometry:
            if not geom_dict:
                logging.warning(f'Skipping feature index {i}: Missing geometry.')
                continue

            # Determine asset ID:
            existing_id = feature.get('id')
            if existing_id is None:
                existing_id = props.get('id')

            if existing_id is None:
                assigned_id = f'gen_{uuid.uuid4().hex}'
            else:
                assigned_id = str(existing_id)

            # Check if ID exists in the NEW collection being built. Skip if it
            # does:
            if assigned_id in new_collection._data:
                logging.warning(
                    f"Skipping duplicate asset with ID '{assigned_id}' found "
                    "in GeoJSON input."
                )
                continue

            try:
                # Convert geojson geometry to a shapely object:
                shapely_geom = shape(geom_dict)

                # Create a PhysicalAsset and add it to collection:
                new_asset = PhysicalAsset(
                    id=assigned_id, geometry=shapely_geom, attributes=props
                )

                new_collection.add(new_asset)

            except (ValueError, TypeError, AttributeError) as e:
                logging.error(f"Failed to create asset for ID '{assigned_id}': {e}")
                continue

        logging.info(f'Loaded {len(new_collection._data)} assets from GeoJSON.')
        return new_collection

    @classmethod
    def from_shapefile(cls, file: str | Path) -> PhysicalAssetCollection:
        """
        Creates a new ``PhysicalAssetCollection`` from an ESRI Shapefile.

        This factory method reads a shapefile, extracts the geometries and
        attribute tables, and populates a new collection. It attempts to
        reconstruct the original `id` if it was saved in the shapefile's
        attribute table.

        Args:
            file (str or Path):
                The path to the shapefile (``.shp``). The accompanying core
                files (``.shx`` and ``.dbf``) must be located in the same
                directory. It is also highly recommended that the projection
                file (``.prj``) and character encoding file (``.cpg``) are
                present to ensure spatial and text data are interpreted
                correctly.

        Returns:
            PhysicalAssetCollection:
                A new instance populated with the assets from the shapefile.

        Raises:
            shapefile.ShapefileException: If the file cannot be read.

        Examples:
            Import ``PhysicalAssetCollection`` class:

            >>> from rapidtools.core import PhysicalAssetCollection

            Load a shapefile from path (ensure the accompanying .shx and .dbf
            files are located in the same directory):

            >>> path = 'data/assets.shp'
            >>> collection = PhysicalAssetCollection.from_shapefile(path)
            INFO: Loaded 5 assets from Shapefile.
            >>> print(len(collection))
            5

            Inspect an imported asset to view its reconstructed geometry and
            attributes:

            >>> asset = collection.get('0')
            >>> asset.summary()
            --- PhysicalAsset Summary ---
            ID:          0
            Asset Type:  building
            Geometry:    POLYGON ((-118.54801743326061 34.050774380694754,
                                   -118.54799871557931 34.0507...
            Attributes (1):
            {
                "type": "building"
            }
            Image Assets (0):
              (No image assets)
            -----------------------------
        """
        new_collection = cls()
        file_path = str(file)

        # Open the shapefile reader:
        with shapefile.Reader(file_path) as sf:
            # sf.fields format: [('DeletionFlag', 'C', 1, 0),
            # ['id', 'C', 254, 0], ...]
            # Skip the first element which is always the DeletionFlag:
            field_names = [field[0] for field in sf.fields[1:]]

            for shape_rec in sf.shapeRecords():
                # shapeType 0 is a NULL shape. This needs to be skipped before
                # calling __geo_interface__ to avoid the GeoJSON_Error:
                if shape_rec.shape.shapeType == 0:
                    continue

                try:
                    # Reconstruct Shapely geometry using the GeoJSON-like interface:
                    geom_dict = shape_rec.shape.__geo_interface__
                    shapely_geom = shape(geom_dict)

                    # Skip empty/null geometries:
                    if (
                        shapely_geom is None
                        or shapely_geom.is_empty
                        or not shapely_geom.is_valid
                        or any(math.isnan(c) for c in shapely_geom.bounds)
                    ):
                        # Log a message if you want to see why it was skipped:
                        logging.warning('Skipping asset: Geometry is empty or invalid.')
                        continue

                except (AttributeError, ValueError, TypeError, Exception):
                    continue

                shapely_geom = shape(geom_dict)

                # Reconstruct attributes dictionary:
                attributes = dict(zip(field_names, shape_rec.record, strict=True))

                # Identify ID (accounting for shapefile 10-char truncation/
                # capitalization). normalize as pyshp might return string or
                # bytes:
                assigned_id = None
                for id_key in ['id', 'ID']:
                    if id_key in attributes:
                        val = attributes.pop(id_key)
                        if val is not None and str(val).strip():
                            assigned_id = str(val).strip()
                        break

                if not assigned_id:
                    assigned_id = f'gen_{uuid.uuid4().hex}'

                # Handle images column if it was generated by `to_shapefile`:
                images_str = attributes.pop('images', None)

                # Attempt to parse back JSON properties strings from DBF:
                for k, v in attributes.items():
                    if isinstance(v, str) and (v.startswith('{') or v.startswith('[')):
                        try:
                            attributes[k] = json.loads(v)
                        except json.JSONDecodeError:
                            pass  # Leave as standard string if it fails to parse

                new_asset = PhysicalAsset(
                    id=assigned_id, geometry=shapely_geom, attributes=attributes
                )

                # Rehydrate images if the column existed and had data:
                if images_str:
                    try:
                        parsed_images = json.loads(images_str)
                        new_asset.attributes['images'] = parsed_images
                    except json.JSONDecodeError:
                        # Fallback if string was truncated or malformed:
                        new_asset.attributes['images_raw_str'] = images_str

                # Skip duplicates if any:
                if assigned_id in new_collection._data:
                    logging.warning(
                        f"Skipping duplicate asset with ID '{assigned_id}' "
                        "found in Shapefile input."
                    )
                    continue

                new_collection.add(new_asset)

        logging.info(f'Loaded {len(new_collection._data)} assets from Shapefile.')
        return new_collection

    def get(
        self, asset_ids: str | Iterable[str], default: Any = None
    ) -> PhysicalAsset | Any | list[PhysicalAsset | Any]:
        """
        Safely retrieve assets by unique ID(s).

        This method is polymorphic:

            1. If ``asset_ids`` is a ``str``, it returns a single ``PhysicalAsset``
               (or default).
            2. If ``asset_ids`` is an ``Iterable`` (list, tuple, etc.), it returns
               a ``list`` of results in the same order as the requested IDs.

        Args:
            asset_ids:
                A single asset ID string OR an iterable of ID strings.
            default:
                Value to return if an ID is not found. Defaults to ``None``.

        Returns:
            - ``PhysicalAsset`` (or default) if input is a single string.
            - ``list[PhysicalAsset or Any]`` if input is a list/iterable.

        Raises:
            TypeError: If ``asset_ids`` is not a ``str`` or ``Iterable``.

        Examples:
            Create a ``PhysicalAssetCollection`` consisting of two
            ``PhysicalAsset`` objects:

            >>> from shapely.geometry import Point
            >>> from rapidtools.core import PhysicalAsset, PhysicalAssetCollection
            >>>
            >>> collection = PhysicalAssetCollection()
            >>> collection.add(PhysicalAsset(
            >>>     id='pump_01',
            >>>     geometry=Point(0,0)
            >>> ))
            >>> collection.add(PhysicalAsset(
            >>>     id='pump_02',
            >>>     geometry=Point(1,1)
            >>> ))

            Single lookup (returns ``PhysicalAsset`` or ``None``):

            >>> asset = collection.get('pump_01')
            >>> asset.id
            'pump_01'

            Batch lookup (returns a list of ``PhysicalAsset`` objects):

            >>> results = collection.get(['pump_01', 'pump_02', 'missing_id'])
            >>> print(results)
            [<PhysicalAsset id='pump_01' asset_type='N/A' geometry='Point'>,
            <PhysicalAsset id='pump_02' asset_type='N/A' geometry='Point'>,
            None]

            # Set a custom default value:

            >>> collection.get('missing_id', default='Not Found')
            'Not Found'
        """
        # Case 1: Single ID Lookup:
        if isinstance(asset_ids, str):
            return self._data.get(asset_ids, default)

        # Case 2: Batch Lookup:
        if isinstance(asset_ids, Iterable):
            return [self._data.get(k, default) for k in asset_ids]

        # Case 3: Invalid Input:
        raise TypeError(
            "asset_ids must be specified as a string or iterable of strings. "
            f"Got '{type(asset_ids).__name__}'."
        )

        return default

    def filter_by_attribute(
        self, key: str, value: Any, operator: str = '=='
    ) -> PhysicalAssetCollection:
        """
        Return a new collection of assets with specified attribute criteria.

        This method filters the collection based on the ``key`` existing in an
        asset's ``attributes`` dictionary and comparing it against ``value``
        using the specified ``operator``.

        Args:
            key (str):
                The dictionary key to look up in ``asset.attributes``.
            value (Any):
                The value to compare against.
            operator (str, optional):
                The comparison operator to use. Supported options:
                    - ``'=='``: Exact match (default).
                    - ``'!='``: Not equal.
                    - ``'>'``, ``'>='``, ``'<'``, ``'<='``: Numeric/Lexical
                      comparison.
                    - ``'in'``: Checks if asset attribute is in the provided
                      ``value`` (list/set).
                    - ``'contains'``: Checks if asset attribute CONTAINS the
                       provided ``value``.
                    - ``'exists'``: Checks if the key exists (value argument is
                      ignored).

        Returns:
            PhysicalAssetCollection:
                A new collection containing only the matching assets.

        Raises:
            ValueError: If the provided `operator` string is not supported.

        Examples:
            Create a collection consisting of three physical assets:

            >>> from shapely.geometry import Point
            >>> from rapidtools.core import PhysicalAsset, PhysicalAssetCollection
            >>>
            >>> collection = PhysicalAssetCollection()
            >>> collection.add([
            ...     PhysicalAsset(
            ...         id='a1',
            ...         geometry=Point(0,0),
            ...         attributes={'status': 'active', 'age': 5}
            ...     ),
            ...     PhysicalAsset(
            ...         id='a2',
            ...         geometry=Point(1,0),
            ...         attributes={'status': 'inactive', 'age': 10}
            ...     ),
            ...     PhysicalAsset(
            ...         id='a3',
            ...         geometry=Point(0,1),
            ...         attributes={'status': 'active', 'age': 15}
            ...     )
            ... ])

            Filter for exact match (default behavior). Get assets that have
            an active status (i.e., ``status=active``):

            >>> active = collection.filter_by_attribute('status', 'active')
            >>> len(active)
            2

            Filter using inequalities. Get assets older than 8:

            >>> old = collection.filter_by_attribute('age', 8, operator='>')
            >>> len(old)
            2

            Filter using ``'in'`` to find assets matching ANY value in a list.
            In this example, we keep the asset if status is ``'active'`` or
            ``'repair'``. If no assets are ``'repair'``, it simply returns the
            ``'active'`` ones:

            >>> subset = collection.filter_by_attribute(
            ...     'status',
            ...     ['active', 'repair'],
            ...     operator='in'
            ... )
            >>> subset.summary()
            {'total_assets': 2,
             'total_images': 0,
             'asset_types': {'Unknown': 2},
             'bounds': BoundingBox(minx=0.0, miny=0.0, maxx=0.0, maxy=1.0)}
            >>> subset[0].attributes
            {'status': 'active', 'age': 5}
        """
        filtered_assets = []

        # Map string operators to actual functions:
        ops = {
            '==': op.eq,
            '!=': op.ne,
            '>': op.gt,
            '>=': op.ge,
            '<': op.lt,
            '<=': op.le,
            'in': lambda attr_val, target: attr_val in target,
            'contains': lambda attr_val, target: target in attr_val,
        }

        # Special handle 'exists' which does not use the 'value' param:
        if operator == 'exists':
            return PhysicalAssetCollection(
                assets=[a for a in self._data.values() if key in a.attributes]
            )

        if operator not in ops:
            raise ValueError(
                f"Unsupported operator: '{operator}'. Valid options: "
                f"{list(ops.keys())}"
            )

        compare_func = ops[operator]

        for asset in self._data.values():
            # Skip if attribute is missing completely:
            if key not in asset.attributes:
                logging.warning(
                    f"Missing attribute [Asset ID: '{asset.id}']: "
                    f"Attribute '{key}' was not found. Skipping asset"
                )
                continue

            attr_val = asset.attributes[key]

            # Attempt comparison with Type Mismatch handling:
            try:
                if compare_func(attr_val, value):
                    filtered_assets.append(asset)
            except TypeError:
                # Catch errors where types are incompatible (e.g.,
                # comparing str > int) and skip the asset:
                logging.warning(
                    f"Filter type mismatch [Asset ID: '{asset.id}']: "
                    f"Cannot compare attribute '{key}' value '{attr_val}' "
                    f"({type(attr_val).__name__}) with filter value '{value}' "
                    f"({type(value).__name__}) using operator '{operator}'. "
                    "Skipping asset."
                )
                continue

        return PhysicalAssetCollection(assets=filtered_assets)

    def filter_by_geometry(
        self,
        geometry: BoundingBox | PolygonRegion | BaseGeometry,
        predicate: str = 'intersects',
    ) -> PhysicalAssetCollection:
        """
            Returns a new collection with assets matching a spatial query.

            This method accepts ``BoundingBox``, ``PolygonRegion``, or raw
            Shapely geometries. It leverages the underlying geometry for
            efficient filtering.

            Args:
                geometry:
                    The search area can be:

                        - An instance of ``BoundingBox`` or ``PolygonRegion``.
                        - A raw Shapely ``BaseGeometry``.
                predicate (str):
                    The spatial relationship to check. Defaults to
                    ``'intersects'``.

                        - ``'intersects'``: Returns assets that touch or overlap
                          (Default).
                        - ``'within'``: Returns assets strictly inside the geometry.
                        - ``'contains'``: Returns assets that strictly contain
                          the geometry.

            Returns:
                PhysicalAssetCollection: A subset of matching assets.

            Examples:
                Create an asset collection consisting of five physical assets:

                >>> from shapely.geometry import Point
                >>> from rapidtools.core import PhysicalAsset, \
                ... PhysicalAssetCollection, BoundingBox, PolygonRegion
                >>>
                >>> assets = [
                ...     PhysicalAsset(
                ...         id='pole_1',
                ...         geometry=Point(1, 1),
                ...         attributes={'status': 'active', 'voltage': 110},
                ...     ),
                ...     PhysicalAsset(
                ...         id='pole_2',
                ...         geometry=Point(2, 2),
                ...         attributes={'status': 'repair', 'voltage': 220},
                ...     ),
                ...     PhysicalAsset(
                ...         id='pole_3',
                ...         geometry=Point(5, 5),
                ...         attributes={'status': 'inactive', 'voltage': 110},
                ...     ),
                ...     PhysicalAsset(
                ...         id='pump_1',
                ...         geometry=Point(8, 8),
                ...         attributes={'status': 'active', 'voltage': 'N/A'},
                ...     ),
                ...     PhysicalAsset(
                ...         id='pump_2',
                ...         geometry=Point(12, 12),
                ...         attributes={'status': 'active', 'voltage': 220},
                ...     ),
                ... ]
                >>> collection = PhysicalAssetCollection(assets)

                Filter using ``BoundingBox`` (0,0 to 6,6):

                >>> bbox = BoundingBox(0, 0, 6, 6)
                >>> in_box = collection.filter_by_geometry(bbox)
                >>> print(
                ... f'Assets in BoundingBox(0,0,6,6): {[a.id for a in in_box]}'
                ... )
                Assets in BoundingBox(0,0,6,6): ['pole_1', 'pole_2', 'pole_3']

                Filter using a triangular ``PolygonRegion``:

                >>> triangle = PolygonRegion([(0,0), (0,3), (3,0)])
                >>> in_tri = collection.filter_by_geometry(triangle)
                >>> print(f'Assets in triangle: {[a.id for a in in_tri]}')
                Assets in triangle: ['pole_1']

                A comparison of a strict ``'within'`` check vs '``intersects'``.
                pole_3 is at (5,5). Following ``BoundingBox`` barely touches pole_3.
                Please note that in checking for a spatial relationship,
                intersects (i.e., the default predicate) includes edges, while
                within excludes them:

                >>> touching_box = BoundingBox(5, 5, 10, 10)
                >>> touching = collection.filter_by_geometry(touching_box)
                >>> print(f'Touching bounding box: {[a.id for a in touching]}')
                Touching bounding box: ['pole_3', 'pump_1']
                >>>
                >>> inside = collection.filter_by_geometry(
                ...    touching_box,
                ...    predicate='within'
                ... )
                >>> print(f'Strictly inside: {[a.id for a in inside]}')
                Strictly inside: ['pump_1']
            """
        # All valid Shapely binary predicates are supported:
        VALID_PREDICATES = {
            'intersects',
            'within',
            'contains',
            'touches',
            'disjoint',
            'overlaps',
        }

        # Extract the search shape:
        # Check for raw Shapely Geometry first:
        if isinstance(geometry, BaseGeometry):
            search_shape = geometry
        # Then check for custom wrappers (like PolygonRegion) that hold the
        # geom internally:
        elif hasattr(geometry, '_geom'):
            search_shape = geometry._geom
        # Catch generic objects (like custom tuple wrappers) that provide
        # standard bounds:
        elif hasattr(geometry, 'bounds'):
            search_shape = box(*geometry.bounds)
        else:
            raise TypeError(
                "Input must be a rapidtools BoundingBox/PolygonRegion) "
                "or Shapely Geometry. Got type: "
                f"'{type(geometry).__name__}'"
            )

        # Validate predicate:
        if predicate not in VALID_PREDICATES:
            raise ValueError(
                f"Invalid spatial predicate '{predicate}'. "
                f"Supported options: {VALID_PREDICATES}"
            )

        # Filter assets:
        filtered_assets = []

        for asset in self._data.values():
            # Dynamically retrieve the method (e.g.,asset.geometry.intersects)
            # and call it with the search_shape:
            if getattr(asset.geometry, predicate)(search_shape):
                filtered_assets.append(asset)

        return PhysicalAssetCollection(assets=filtered_assets)

    def filter_empty(self) -> PhysicalAssetCollection:
        """
        Return a new collection excluding assets with no descriptive attributes.

        This method acts as a data-cleaning tool. It identifies assets where
        the ``attributes`` dictionary is completely empty and excludes them
        from the resulting collection.

        Returns:
            PhysicalAssetCollection:
                A new collection containing only assets that possess at least
                one metadata attribute.

        Example:
            >>> # If you have 500 assets but only 360 were successfully analyzed:
            >>> print(len(collection))
            500
            >>> clean_collection = collection.filter_empty()
            >>> print(len(clean_collection))
            360
        """
        # Keep the asset only if its attributes dictionary has content.
        # In Python, an empty dict '{}' evaluates to False:
        filtered_assets = [asset for asset in self._data.values() if asset.attributes]

        return self.__class__(assets=filtered_assets)

    def merge(
        self,
        other: PhysicalAssetCollection,
        strategy: Literal['skip', 'overwrite', 'raise'] = 'skip',
    ) -> None:
        """
        Merge another collection into this instance.

        Merges another ``PhysicalAssetCollection`` into this instance.

        This method modifies the current collection in-place by adding assets
        from the ``other`` collection. If an asset ID already exists in the
        current collection, the conflict is resolved according to the
        ``strategy`` parameter.

        Args:
            other (PhysicalAssetCollection):
                The collection to merge into this instance.
            strategy (str, optional): The rule for handling duplicate asset IDs.
                Defaults to ``'skip'``. Available options are:

                    * ``'skip'``: Keep the existing asset in this collection;
                      ignore the incoming asset.
                    * ``'overwrite'``: Replace the existing asset in this
                      collection with the incoming asset.
                    * ``'raise'``: Abort the merge and raise a ValueError if a
                      duplicate ID is found.

        Returns:
            None: This method modifies the collection in place.

        Raises:
            TypeError:
                If ``other`` is not an instance of `PhysicalAssetCollection`.
            ValueError:
                If ``strategy`` is not one of ``'skip'``, ``'overwrite'``, or
                ``'raise'``.
            ValueError:
                If ``strategy`` is ``'raise'`` and a duplicate asset ID is
                encountered.

        Examples:
            Create two collections, one containing a single asset, and another
            containing two assets (one of which shares the same ID as the asset
            in the first collection):

            >>> # Assume we have a collection with one asset
            >>> main_col = PhysicalAssetCollection()
            >>> main_col.add(PhysicalAsset(id='1', name='Original Server'))
            >>>
            >>> # Assume we have a second collection to merge in
            >>> new_col = PhysicalAssetCollection()
            >>> new_col.add(PhysicalAsset(id='1', name='New Server')) # Duplicate ID
            >>> new_col.add(PhysicalAsset(id='2', name='Switch'))     # New ID

            Merge using the ``'skip'`` strategy (default behavior). The existing
            ``'Original Server'`` is preserved. The ``'Switch'`` is added:

            >>> main_col.merge(new_col, strategy='skip')
            >>> print(main_col.get('1').name)
            Original Server
            >>> print(main_col.get('2').name)
            Switch

            Merge using the ``'overwrite'`` strategy. The existing asset is
            updated to ``'New Server'``. The ``'Switch'`` is added:

            >>> main_col.merge(new_col, strategy='overwrite')
            >>> print(main_col.get('1').name)
            New Server

            Merge using the ``'raise'`` strategy. The operation fails
            immediately upon finding the duplicate ``'1'``.

            >>> try:
            ...     main_col.merge(new_col, strategy='raise')
            ... except ValueError as e:
            ...     print(e)
            Duplicate asset ID '1' found during merge.
        """
        VALID_STRATEGIES = {'skip', 'overwrite', 'raise'}

        if not isinstance(other, PhysicalAssetCollection):
            raise TypeError(
                f'Expected PhysicalAssetCollection, got {type(other).__name__}.'
            )

        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{strategy}'. Must be one of: " f"{VALID_STRATEGIES}"
            )

        stats = {'added': 0, 'overwritten': 0, 'skipped': 0}

        # Iterate over the other collection's dictionary:
        for asset_id, new_asset in other._data.items():
            if asset_id in self._data:
                if strategy == 'raise':
                    raise ValueError(
                        f"Duplicate asset ID '{asset_id}' found during merge."
                    )
                elif strategy == 'overwrite':
                    self._data[asset_id] = new_asset
                    stats['overwritten'] += 1
                else:  # strategy == 'skip'
                    stats['skipped'] += 1
            else:
                self._data[asset_id] = new_asset
                stats['added'] += 1

        logging.info(f"Merge complete using strategy '{strategy}': {stats}")

    def remove(self, asset_ids: str | Iterable[str]) -> None:
        """
        Remove one or more assets from the collection by ID.

        This method is polymorphic:

            1. If ``asset_ids`` is a ``str``, it removes that single asset.
            2. If ``asset_ids`` is an ``Iterable``, it iterates and removes all
               IDs found.

        If an ID is not found in the collection, the operation is skipped for
        that specific ID and a warning is logged.

        Args:
            asset_ids: A single asset ID string OR an iterable of ID strings.

        Raises:
            TypeError: If ``asset_ids`` is not a ``str`` or an ``Iterable``.

        Examples:
            Create a ``PhysicalAssetCollection`` consisting of two
            ``PhysicalAsset`` objects:

            >>> from shapely.geometry import Point
            >>> from rapidtools.core import PhysicalAsset, PhysicalAssetCollection
            >>>
            >>> collection = PhysicalAssetCollection()
            >>> collection.add([
            ...     PhysicalAsset(id='p1', geometry=Point(0,0)),
            ...     PhysicalAsset(id='p2', geometry=Point(1,1))
            ... ])

            Remove the ``PhysicalAsset`` with ID:``'p1'`` from the collection:

            >>> collection.remove('p1')
            >>> print(collection)
            PhysicalAssetCollection(_data={'p2': <PhysicalAsset id='p2'
            asset_type='N/A' geometry='Point'>})

            Try to remove a non-matching item:

            >>> collection.remove('p99')
            WARNING: Asset ID 'p99' not found. Skipping removal.

            Batch remove multiple assets:

            >>> collection.remove(['p2', 'p99'])
            WARNING: Asset ID 'p99' not found. Skipping removal.
            >>> len(collection)
            0
        """
        # Normalize input to always be an iterable:
        ids_to_remove: Iterable[str]

        if isinstance(asset_ids, str):
            ids_to_remove = [asset_ids]
        elif isinstance(asset_ids, Iterable):
            ids_to_remove = asset_ids
        else:
            raise TypeError(
                f"asset_id must be a string or iterable of strings. "
                f"Got '{type(asset_ids).__name__}'."
            )

        # Remove specified IDs:
        for uid in ids_to_remove:
            if self._data.pop(uid, None) is None:
                logging.warning(f"Asset ID '{uid}' not found. Skipping removal.")

    def set_attribute(self, key: str, value: Any, overwrite: bool = False) -> None:
        """
        Batch update a specific attribute across all assets in the collection.

        This method assigns ``value`` to ``asset.attributes[key]``. It supports
        both static values and dynamic value generation using functions.

        Args:
            key (str):
                The attribute name to set.
            value (Any):
                The value to assign.

                    - If a static value (int, str), it is assigned directly.
                    - If a ``callable`` (function/lambda), it is executed for
                      each asset as ``value(asset)``, and the result is assigned.
                      This is useful for derived properties or avoiding shared
                      mutable references (see examples).
            overwrite (bool):
                If ``False``, existing attributes will be preserved
                and not updated. If ``True``, the new value overwrites any
                existing value. Defaults to ``False``.

        Examples:
            Setup a collection with two assets for the examples:

            >>> from shapely.geometry import Point
            >>> from rapidtools.core import PhysicalAsset, PhysicalAssetCollection
            >>>
            >>> assets = [
            ...     PhysicalAsset(
            ...     id='a1',
            ...     geometry=Point(0,0),
            ...     attributes={'status': 'old'}
            ... ),
            ...     PhysicalAsset(id='a2', geometry=Point(1,1)) # No attributes
            ... ]
            >>> collection = PhysicalAssetCollection(assets)

            Add a new tag to every asset (static assigment):

            >>> collection.set_attribute('category', 'utility_pole')
            INFO: Updated attribute 'category' for 2 assets.
            >>> collection['a1'].attributes['category']
            'utility_pole'

            Overwrite existing attribute values. In the example below, ``'a1'``
            already has ``status='old'``. By default, ``set_attribute`` does
            not modify an existing value. To force an update, set the
            ``overwrite`` argument to ``True``:

            >>> collection.set_attribute('status', 'new')
            INFO: Updated attribute 'status' for 1 asset.
            >>> collection['a1'].attributes['status']
            'old'
            >>> collection.set_attribute('status', 'new', overwrite=True)
            INFO: Updated attribute 'status' for 2 assets.
            >>> collection['a1'].attributes['status']
            'new'

            Derive values for each asset (dynamic assignment). This example
            uses a lambda function to get the x coordinates of assets using
            their geometry information:

            >>> collection.set_attribute(
            ...     'x_coord',
            ...     value=lambda asset: asset.geometry.x,
            ...     overwrite=True
            ... )
            INFO: Updated attribute 'x_coord' for 2 assets.
            >>> collection['a2'].attributes['x_coord']
            1.0

            How to initialize mutable objects safely (and not fall for the
            ``list`` Trap). This example uses a lambda that returns a new list
            generate a unique list for each asset, avoiding the common bug
            where all assets share the same list reference:

            >>> collection.set_attribute('logs', lambda _: list())
            INFO: Updated attribute 'logs' for 2 assets.
            >>> collection['a1'].attributes['logs'].append('Log Entry 1')
            >>> collection['a2'].attributes['logs']
            []
        """
        # Check if the user provided a generator function:
        is_dynamic = callable(value)

        # Keep track of the updated assets for logging:
        count = 0

        for asset in self._data.values():
            # Skip if the key exists but overwrite is not allowed:
            if not overwrite and key in asset.attributes:
                continue

            # Assign value:
            if is_dynamic:
                asset.attributes[key] = value(asset)
            else:
                asset.attributes[key] = value

            count += 1

        # Use module-level logger (best practice) instead of root logging
        suffix = 's' if count != 1 else ''
        logging.info(f"Updated attribute '{key}' for {count} asset{suffix}.")

    def summary(self) -> dict[str, Any]:
        """
        Generate a statistical summary of the asset collection.

        This method aggregates data across all assets to provide a high-level
        overview. It determines an asset's type by checking the priority keys
        defined in ``PhysicalAsset._ASSET_TYPE_KEYS`` against the asset's
        attributes.

        Returns:
            dict[str, Any]: A dictionary containing:

                * ``'total_assets'`` (int): Total number of physical assets.
                * ``'total_images'`` (int): Sum of images across all assets.
                * ``'asset_types'`` (dict): A frequency count of assets by type.
                * ``'bounds'`` (tuple): The (minx, miny, maxx, maxy) bounds of the
                  entire collection.

        Examples:
            Create a collection with a diverse set of assets. Please note that
            assets 1 and 2 are defined using the standard ``'type'`` key. The
            type for asset 3 is define using ``'category'`` key (a fallback key
            in ``_ASSET_TYPE_KEYS``). Lastly, asset 4 has no known type keys:

            >>> from shapely.geometry import Point
            >>> from rapidtools.core import PhysicalAsset, PhysicalAssetCollection
            >>>
            >>> col = PhysicalAssetCollection()
            >>> col.add(PhysicalAsset(
            ...     id='A1',
            ...     geometry=Point(0,0),
            ...     attributes={'type': 'Pole'}
            ... ))
            >>> col.add(PhysicalAsset(
            ...     id='A2',
            ...     geometry=Point(1,1),
            ...     attributes={'type': 'Pole'}
            ... ))
            >>> col.add(PhysicalAsset(
            ...     id='A3',
            ...     geometry=Point(2,2),
            ...     attributes={'category': 'Valve'}
            ... ))
            >>> col.add(PhysicalAsset(
            ...     id='A4',
            ...     geometry=Point(3,3),
            ...     attributes={'material': 'wood'}
            ... ))

            Create a summary:

            >>> stats = col.summary()
            >>> print(stats['total_assets'])
            4
            >>> print(stats['asset_types'])
            {'Pole': 2, 'Valve': 1, 'Unknown': 1}
        """
        type_counts: Counter[str] = Counter()
        total_images = 0

        for asset in self._data.values():
            # Determine asset type. Start as 'Unknown', then check keys in the
            # priority order defined by the class
            asset_type = 'Unknown'

            for key in asset._ASSET_TYPE_KEYS:
                if val := asset.attributes.get(key):
                    asset_type = val
                    break

            type_counts[asset_type] += 1

            # Count images:
            total_images += len(asset.image_assets)

        return {
            'total_assets': len(self._data),
            'total_images': total_images,
            'asset_types': dict(type_counts),
            'bounds': self.combined_bounding_box,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the collection to a Pandas DataFrame.

        Returns a tabular representation of the assets. The resulting DataFrame
        includes standard columns (``'id'``, ``'asset_type'``, ``'geometry_wkt'``,
        ``'image_count'``) and expands the ``'attributes'`` dictionary so that each
        attribute key becomes its own column.

        Returns:
            pd.DataFrame: A ``DataFrame`` where each row is a ``PhysicalAsset``.

        Example:
            Create a sample collection:

            >>> from shapely.geometry import Point
            >>> from rapidtools.core import PhysicalAsset, PhysicalAssetCollection
            >>>
            >>> col = PhysicalAssetCollection()
            >>> col.add(
            ...     PhysicalAsset(
            ...         id='A1',
            ...         geometry=Point(0,0),
            ...         attributes={'material': 'wood'}
            ... ))

            Convert to ``DataFrame``:

            >>> df = col.to_dataframe()
            >>> print(df.iloc[0]['material'])
            wood
        """
        data = []
        for asset in self._data.values():
            # Resolve Asset Type dynamically, reusing logic from summary:
            resolved_type = 'Unknown'
            for key in asset._ASSET_TYPE_KEYS:
                if val := asset.attributes.get(key):
                    resolved_type = val
                    break

            # Start the row with the attributes:
            row = asset.attributes.copy()

            # Update with system properties:
            row.update(
                {
                    'id': asset.id,
                    'asset_type': resolved_type,
                    'geometry_wkt': asset.geometry.wkt,
                    'image_count': len(asset.image_assets),
                }
            )

            # Merge row
            data.append(row)

        return pd.DataFrame(data)

    def to_geojson(
        self,
        file: str | Path | None = None,
        indent: int | None = 2,
        ignore_properties: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Serialize the collection into a GeoJSON FeatureCollection.

        This method converts all assets in the collection into a standard
        GeoJSON structure. It can either return the dictionary directly or
        write it to a file if a path is provided.

        It supports an optional filter to exclude specific metadata fields
        (such as the heavy 'image_assets' list) from the output to reduce
        file size.

        Args:
            file (str or Path or None, optional):
                If provided, the GeoJSON data will be written to this file
                path. Defaults to ``None``.
            indent (int | None, optional):
                The indentation level for JSON pretty-printing. Set to ``None``
                for a compact representation. Defaults to 2.
            ignore_properties (list[str] or None, optional):
                A list of property keys to exclude from each feature in the
                output. This is useful for removing large nested objects
                like 'image_assets' when only metadata is needed.
                Defaults to ``None``.

        Returns:
            dict[str, Any]:
                A dictionary representing the GeoJSON FeatureCollection.

        Examples:
            Write the collection to a GeoJSON dictionary:

            >>> from shapely.geometry import Point
            >>> from rapidtools.core import PhysicalAsset, PhysicalAssetCollection
            >>>
            >>> col = PhysicalAssetCollection()
            >>> col.add(PhysicalAsset(id='A1', geometry=Point(0,0)))
            >>> data = col.to_geojson()
            >>> print(data['type'])
            FeatureCollection
            >>> print(data['features'][0]['id'])
            A1

            Write the collection to a GeoJSON file name ``'output.json'`` with
            custom indentation:

            >>> _ = col.to_geojson(file='output.json', indent=4)

            Export only metadata by ignoring the 'image_assets' property:

            >>> data = col.to_geojson(ignore_properties=['image_assets'])
            >>> print('image_assets' in data['features'][0]['properties'])
            False

        """
        feature_collection = {
            'type': 'FeatureCollection',
            'features': [
                asset.to_geojson_feature(ignore_properties=ignore_properties)
                for asset in self._data.values()
            ],
        }

        if file is not None:
            path = Path(file)
            path.parent.mkdir(parents=True, exist_ok=True)

            with path.open('w', encoding='utf-8') as f:
                json.dump(feature_collection, f, indent=indent, default=str)

        return feature_collection

    def to_shapefile(
        self,
        file: str | Path,
        crs: str = 'EPSG:4326',
        ignore_properties: list[str] | None = None,
    ) -> None:
        """
        Exports all assets in the collection to an ESRI Shapefile.

        Because a shapefile requires a fixed tabular schema, this method
        performs a first pass over all assets to unify the attributes. If an
        attribute exists in some assets but not others, the missing values
        will be written as empty/null.

        It supports an optional filter to exclude specific metadata fields
        (such as the heavy 'image_assets' list) from the output to reduce
        file size and prevent cluttering the DBF attribute table.

        Note on Shapefile limitations:
            * Column names (attribute keys) are strictly truncated to 10 chars.
            * String values are strictly truncated to 254 characters.
            * Nested data structures (lists, dicts) are serialized to JSON strings.

        Args:
            file (str or Path):
                The destination path for the shapefile. The string should end
                in '.shp'. The method will generate the accompanying .shx, .dbf,
                and .prj files alongside it.
            crs (str, optional):
                The Coordinate Reference System (CRS). Currently, this method
                auto-generates a .prj file for 'EPSG:4326' (WGS84). Defaults
                to 'EPSG:4326'.
            ignore_properties (list[str] or None, optional):
                A list of property keys to exclude from the attribute table.
                Defaults to ``None``.

        Raises:
            ImportError: If the 'pyshp' library is not installed.
            ValueError: If the collection is empty.

        Examples:
            Create a collection:

            >>> from shapely.geometry import Point
            >>> from rapidtools.core import PhysicalAsset, PhysicalAssetCollection
            >>>
            >>> col = PhysicalAssetCollection()
            >>> col.add(PhysicalAsset(
            ...     id='A1',
            ...     geometry=Point(0,0),
            ...     attributes={'status': 'active'}
            ... ))
            >>> col.to_shapefile('data/assets.shp')

            Write the collection to a Shapefile, exporting only basic metadata
            (ignoring the 'image_assets' and 'notes' properties):

            >>> col.to_shapefile(
            ...     'data/assets_light.shp',
            ...     ignore_properties=['image_assets', 'notes']
            ... )
        """
        if not self._data:
            raise ValueError('Cannot export an empty collection to a shapefile.')

        ignore_props = set(ignore_properties or [])

        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Discover unified schema
        # Shapefiles require a rigid, tabular schema upfront before writing any
        # data. It is necessary to iterate through all assets to determine all
        # possible attributes and their necessary data types:
        schema = {}

        # Pre-calculate to avoid checking the ignore list on every single asset
        check_images = 'image_assets' not in ignore_props
        has_images = False

        for asset in self._data.values():
            # Stop looking for images once we find the first one:
            if check_images and not has_images and getattr(asset, 'image_assets', None):
                has_images = True

            for k, v in asset.attributes.items():
                if k in ignore_props:
                    continue

                # Determine the pyshp field type for the CURRENT attribute
                # value: 'L' = Logical (bool), 'N' = Numeric (int),
                # 'F' = Float, 'C' = Character (string):
                # Length 1, 0 decimals:
                if isinstance(v, bool):
                    val_type = ('L', 1, 0)
                # DBF allows up to 18 chars for numbers:
                elif isinstance(v, int):
                    val_type = ('N', 18, 0)
                # 18 chars total, 5 decimal places:
                elif isinstance(v, float):
                    val_type = ('F', 18, 5)
                # DBF max string length is 254 characters:
                else:
                    val_type = ('C', 254, 0)

                if k not in schema:
                    # First time seeing this attribute, register its type:
                    schema[k] = val_type
                else:
                    # Encountered this attribute before. Check for type
                    # conflicts across assets.
                    curr_type_char = schema[k][0]
                    new_type_char = val_type[0]

                    # If types match, or if the field has already been degraded
                    # to a str, do nothing:
                    if curr_type_char == new_type_char or curr_type_char == 'C':
                        continue

                    # Handle manageable conflicts (e.g., mixing Ints and Floats
                    # in the same field)
                    if curr_type_char == 'N' and new_type_char == 'F':
                        schema[k] = ('F', 18, 5)  # Upgrade the whole column to Float
                    elif curr_type_char == 'F' and new_type_char == 'N':
                        pass  # Keep Float, as it safely holds Integers
                    else:
                        # Unresolvable conflict (e.g., mixing bool and str).
                        # Degrade the column to a String to ensure all data
                        # fits safely:
                        schema[k] = ('C', 254, 0)

        # Setup field truncation and disambiguation
        # The attribute table of a shapefile has a strict, legacy limitation,
        # column names cannot exceed 10 characters:
        fields = []
        field_mapping = {}
        seen_fields = set()

        def get_safe_name(name: str) -> str:
            """
            Helper to safely truncate field names to 10 chars.
            If a truncation causes a duplicate column name (e.g., 'description1'
            and 'description2' both truncate to 'descriptio'), it appends a
            numeric suffix to disambiguate while keeping the total length <= 10.
            """
            base = name[:10]
            col = base
            counter = 1
            while col in seen_fields:
                suffix = str(counter)
                # Trim the base just enough to make room for the suffix:
                col = f'{base[:10 - len(suffix)]}{suffix}'
                counter += 1
            seen_fields.add(col)
            return col

        # Always include the primary asset ID as the first column.
        # 'C' means Character (String), max length 254, 0 decimal places:
        safe_id = get_safe_name('id')
        fields.append((safe_id, 'C', 254, 0))

        # Map original attribute keys to their new, safe, 10-character DBF
        # column names:
        for original_k, (f_type, f_len, f_dec) in schema.items():
            safe_k = get_safe_name(original_k)
            field_mapping[original_k] = safe_k
            fields.append((safe_k, f_type, f_len, f_dec))

        # Add a dedicated column for image metadata if any assets contain
        # images:
        if has_images:
            safe_images = get_safe_name('images')
            fields.append((safe_images, 'C', 254, 0))

        # Write the Shapefile components (.shp, .shx, .dbf)
        with shapefile.Writer(str(path)) as w:
            for f in fields:
                w.field(*f)

            for asset in self._data.values():
                # Initialize the row dictionary with the required ID,
                # truncating just in case:
                record_values = {safe_id: str(asset.id)[:254]}

                # Populate the remaining attributes for this asset:
                for original_k, safe_k in field_mapping.items():
                    val = asset.attributes.get(original_k)
                    f_type = schema[original_k][0]

                    if val is None:
                        record_values[safe_k] = None
                    elif f_type == 'C':
                        if isinstance(val, (dict, list, tuple)):
                            val_str = json.dumps(val, default=str)
                        else:
                            val_str = str(val)

                        # Strictly enforce the 254 character limit to prevent
                        # pyshp crashes:
                        record_values[safe_k] = val_str[:254]
                    else:
                        record_values[safe_k] = val

                # Populate images
                if has_images:
                    if getattr(asset, 'image_assets', None):
                        img_data = [
                            (
                                asdict(img)
                                if hasattr(img, '__dataclass_fields__')
                                else str(img)
                            )
                            for img in asset.image_assets
                        ]
                        record_values[safe_images] = json.dumps(img_data, default=str)[
                            :254
                        ]
                    else:
                        record_values[safe_images] = ''

                w.record(**record_values)
                w.shape(asset.geometry)

        # Manually write the .prj file for projection:
        prj_filepath = path.with_suffix('.prj')
        if crs.upper() == 'EPSG:4326':
            # This is the Well-Known Text (WKT) string defining the standard
            # WGS84 coordinate system:
            wgs84_wkt = (
                'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,'
                '298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]'
            )
            prj_filepath.write_text(wgs84_wkt)
        else:
            # If a user provides a custom CRS, we skip creation rather than
            # guessing or writing a bad PRJ:
            logging.info(
                f"Skipping .prj generation for custom CRS '{crs}'. "
                "You may need to provide this manually."
            )
