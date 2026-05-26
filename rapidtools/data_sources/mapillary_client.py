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
# 05-25-2025

import base64
import gzip
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import mapbox_vector_tile
import numpy as np
import requests
from PIL import Image, ImageDraw
from tqdm import tqdm

from rapidtools.config import get_configured_session, REQUESTS_TIMEOUT_VAL
from rapidtools.core import BoundingBox, ImageAsset, ImageCollection
from .tile_utils import TileUtils

# Mapillary API data:
BASE_URL = 'https://graph.mapillary.com'
TILE_URL_TEMPLATE = (
    "https://tiles.mapillary.com/maps/vtp/mly1_public/2/"
    "{z}/{x}/{y}?access_token={token}"
)
RAPID_CREATOR_ID = 107708041466249

class SegmentationLabels:
    """
    A namespace for standard Mapillary segmentation category labels.

    This class acts as a static container for string constants used to identify
    specific semantic classes in Mapillary data. Using these attributes 
    instead of raw strings prevents typos and ensures consistency when 
    performing dictionary lookups or conditional logic.

    Attributes:
        VOID (str): 
            Label for undefined, void, or unlabeled regions 
            ('void--unlabeled').
        SKY (str): 
            Label for the sky region ('nature--sky').
        ROAD (str): 
            Label for flat, driveable road surfaces 
            ('construction--flat--road').

    Example:
        Using the class prevents typos in string literals
        
        >>> label_map = {'nature--sky': 27, 'construction--flat--road': 13}
        
        Safe lookup using the constant:
        
        >>> sky_id = label_map.get(SegmentationLabels.SKY)
        >>> print(sky_id)
        27
        
        Useful for readable conditional logic:
            
        >>> current_pixel = 'nature--sky'
        >>> if current_pixel == SegmentationLabels.SKY:
        ...     print("Detected Sky")
        Detected Sky
    """
    VOID = 'void--unlabeled'
    SKY = 'nature--sky'
    ROAD = 'construction--flat--road'
    SURVEY_VEHICLE = 'void--ego-vehicle'

class MapillaryClient:
    """
    Fetches and stores Mapillary images for given geo-coordinates, bounding boxes, or image IDs.
    Requires a Mapillary API access token.
    """    
    AVAILABLE_IMAGE_FIELDS = frozenset([
        'altitude', 'atomic_scale', 'camera_parameters', 'camera_type',
        'captured_at', 'compass_angle', 'computed_altitude', 
        'computed_compass_angle', 'computed_geometry', 'computed_rotation', 
        'creator', 'exif_orientation', 'geometry', 'height', 'is_pano', 
        'make', 'model', 'thumb_256_url', 'thumb_1024_url', 
        'thumb_2048_url', 'thumb_original_url', 'merge_cc', 'mesh', 
        'sequence', 'sfm_cluster', 'width', 'detections.value', 
        'detections.geometry'
    ])

    def __init__(self, access_token: str, save_dir: str = 'mapillary_images'):
        if not access_token:
            raise ValueError('Mapillary access token is required.')
        self.access_token = access_token
        self.save_dir = Path(save_dir)
        
        # Create a session with the Retry strategy and the default headers:
        self.session = self.session = get_configured_session()

    def fetch_image(
        self, 
        image_id: str, 
        fields: list[str] | None = None,
        save_to_disk: bool = True,
        process_masks: list[Literal['semantic', 'instance']] | None = None
    )-> ImageAsset | None:
        """
        Fetch image metadata using it Mapillary ID and optionally download it.
        
        This method:
            - Validates the requested metadata fields.
            - Fetches image metadata from the API.
            - Extracts the download URL from the metadata.
            - Downloads the image file to the configured save directory.
            - Optionally generates and saves segmentation masks.
            - Returns an ImageAsset containing the file path, ID, and metadata.
        
        Args:
            image_id (str):
                The identifier of the image to fetch from the API.
            fields (list[str] | None):
                Optional list of metadata fields to request. If None, all 
                fields defined in self.AVAILABLE_IMAGE_FIELDS are returned.
                Invalid fields are ignored with a warning.
            save_to_disk (bool): 
                If ``True``, downloads the image to the configured save 
                directory. Defaults to ``True``.
            process_masks (list[str] | None): 
                A list of mask types to generate. Options are ``'semantic'`` 
                and ``'instance'``. Pass ``None`` or ``[]`` to skip 
                segmentation. Example: ``['semantic']`` or 
                ``['semantic', 'instance']``.
        
        Returns:
            ImageAsset | None:
                An ImageAsset instance containing:
                    - path (str): 
                        Local path to the downloaded image file.
                    - id (str): 
                        The image ID (API ``id`` if present, otherwise the
                        requested ``image_id``).
                    - properties (dict): 
                        Remaining metadata fields returned by the API 
                        (excluding the file ``id`` and used download URL).
                Returns ``None`` if validation fails, metadata retrieval fails, 
                no download URL is present, or the download itself fails.
        """
        # Determine intent:
        should_process_masks = process_masks is not None and \
            len(process_masks) > 0

        # Validate and prepare requested fields:
        fields_to_request = self._validate_fields(
            fields, 
            image_id, 
            require_segmentation=should_process_masks
        )
        
        if not fields_to_request:
            return None

        # Get metadata:
        props = self._get_image_metadata(image_id, fields_to_request)
        if not props:
            return None

        # Extract the download URL:
        image_url = props.get('thumb_original_url')
        if not image_url:
            logging.error(f"No download URL found for {image_id}")
            return None

        # Prepare an image path:
        prop_id = props.pop('id', image_id)
        file_path = self.save_dir / f"{prop_id}.jpg"
        
        # Determine if the ImageAsset should allow a missing file path.
        # If we are not saving to disk, the file will be missing:
        allow_missing = not save_to_disk
        
        # Optionally download the image to disk:
        if save_to_disk:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            if not self._download_image(image_url, file_path):
                return None  # Download failed
        
        # Create the ImageAsset:
        image_asset = ImageAsset(
            path=str(file_path),
            id=prop_id,
            properties=props,
            allow_missing_file=allow_missing
        )
        
        # Process segmentation:
        if should_process_masks:
            # Check if detections were returned from the API:
            detections_data = props.get('detections')
            
            if not detections_data or not detections_data.get('data'):
                logging.warning(
                    f'No detection data found for {prop_id}, skipping masks.'
                )
            else:
                try:
                    # Semantic mask:
                    if 'semantic' in process_masks:
                        sem_mask, sem_map = self._parse_mapillary_segmentation(
                            image_asset, merge_detections=True
                        )
                        image_asset.set_mask(
                            sem_mask, 
                            mask_type='semantic', 
                            map_data=sem_map
                        )
                        if save_to_disk:
                            image_asset.save_mask(mask_type="semantic")
                    
                    # Instance mask:
                    if 'instance' in process_masks:
                        inst_mask, inst_map = self._parse_mapillary_segmentation(
                            image_asset, 
                            merge_detections=False
                        )
                        image_asset.set_mask(
                            inst_mask, 
                            mask_type='instance', 
                            map_data=inst_map
                        )
                        if save_to_disk:
                            image_asset.save_mask(mask_type='instance')
                        
                except Exception as e:
                    logging.error(
                        f'Failed to process segmentation for {prop_id}: {e}'
                    )
        
        return image_asset

    def fetch_images_by_ids(
        self, 
        image_ids: list[str], 
        fields: list[str] | None,
        save_to_disk: bool = True,
        process_masks: list[Literal['semantic', 'instance']] | None = None,
        max_workers: int = 10
    ) -> ImageCollection:
        """
        Get multiple images by ID and return them as a collection.
    
        This method:
            - Submits one fetch task per image ID to a thread pool.
            - Fetches metadata and downloads each image (via fetch_image).
            - Aggregates successfully fetched ImageAsset objects into a list.
            - Logs any exceptions that occur per image without aborting the 
              entire batch.
        
        Args:
            image_ids (list[str]):
                A list of image IDs to fetch and download
            fields (list[str] | None):
                Optional list of metadata fields to request for each image.
                If ``None``, a default set of fields will be used
            save_to_disk (bool): 
                If ``True``, downloads image files to ``self.save_dir``.
                If ``False``, only metadata is retrieved. Defaults to ``True``.
            process_masks (list[str] | None): 
                A list of mask types to generate. Options are ``'semantic'`` 
                and ``'instance'``. Pass ``None`` or ``[]`` to skip 
                segmentation. Example: ``['semantic']`` or 
                ``['semantic', 'instance']``.
            max_workers:
                Maximum number of parallel worker threads used for fetching
                images. Defaults to ``10``.

        Returns:
            ImageCollection: 
                An ``ImageCollection`` containing all successfully fetched
                ``ImageAsset`` instances. Any images that fail to download or
                process are excluded; errors are logged per image ID.
        """
        # Create the save_dir folder if the user requests downloading images:
        if save_to_disk:
           self.save_dir.mkdir(parents=True, exist_ok=True) 
        
        assets = []  
        # Use ThreadPoolExecutor to download in parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a dictionary to map futures to IDs:
            future_to_id = {
                executor.submit(
                    self.fetch_image, 
                    img_id, 
                    fields,
                    save_to_disk,
                    process_masks): img_id 
                for img_id in image_ids
            }
            
            # Iterate over futures as they complete (in the order they finish,
            # not necessarily the order submitted):
            for future in tqdm(as_completed(future_to_id), 
                               total=len(image_ids), 
                               desc="Downloading Images", 
                               unit="img"):
                
                img_id = future_to_id[future]
                try:
                    asset = future.result()
                    # Only append successfully created ImageAsset objects:
                    if asset:
                        assets.append(asset)
                except Exception as e:
                    logging.error(f"Exception for image {img_id}: {e}")

        final_collection = ImageCollection()
        final_collection.add(assets)

        return final_collection

    def fetch_images_in_bbox(
        self,
        bbox: BoundingBox,
        fields: list[str] | None = None,
        save_to_disk: bool = False,
        process_masks: list[Literal['semantic', 'instance']] | None = None,
        start_date = '',
        end_date = '',
        filter_rapid_only: bool = True,
        max_workers: int = 10
    ) -> ImageCollection:
        """
        Download and parse Mapillary Vector Tiles covering a bbox in parallel.
    
        Args:
            bbox (BoundingBox):
                The geographic rectangular area to search. Must be an 
                instance of ``rapidtools.core.BoundingBox``.
            fields (list[str] | None):
                A list of specific metadata fields to fetch (e.g., 
                ``['captured_at', 'compass_angle']``). If ``None``, 
                defaults to all available fields. Note: The download URL is
                always fetched internally.
            process_masks (list[str] | None): 
                A list of mask types to generate. Options are ``'semantic'`` 
                and ``'instance'``. Pass ``None`` or ``[]`` to skip 
                segmentation. Example: ``['semantic']`` or 
                ``['semantic', 'instance']``.
            save_to_disk (bool): 
                If ``True``, downloads the images to the configured save 
                directory. Defaults to ``True``.   
            filter_rapid_only: 
                If True, filters for RAPID facility images.
            max_workers: 
                Number of parallel download threads (default: 10).
    
        Returns:
            A new ImageCollection populated with assets from ALL tiles.
        """
        # Get a list of tiles that cover the bounding box area:
        tiles_list = TileUtils.bbox_to_mapbox_tiles(bbox, zoom=14)
        
        total_tiles = len(tiles_list)
        if total_tiles == 0:
            logging.warning(
                'No tiles found. Bounding box is too small or invalid'
            )
            return ImageCollection()
        
        logging.info(f"Scanning {total_tiles} tiles for images...")
    
        # Download/process tiles in parallel:
        final_collection = ImageCollection()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tile = {
                executor.submit(
                    self._get_tile_image_data, 
                    tile, 
                    start_date,
                    end_date,
                    filter_rapid_only
                ): tile 
                for tile in tiles_list
            }
    
            for future in tqdm(
                as_completed(future_to_tile), 
                total=total_tiles, 
                desc="Downloading Tiles",
                unit="tile"
            ):
                tile = future_to_tile[future]
                try:
                    assets = future.result()
                    # Add found assets to the main collection
                    if assets:
                        final_collection.add(assets)
                except Exception as e:
                    logging.error(f'Critical error in tile thread {tile}: {e}')
    
        # Exit early if no images are returned:
        unique_count = len(final_collection)
        logging.info(f'Scan complete. Found {unique_count} unique images.')
        
        if unique_count == 0:
            return final_collection
        
        # If we need specific metadata OR we need to save files to disk, 
        # we must hit the API for the specific IDs found:
        if fields or save_to_disk: 
            ids = final_collection.get_ids()
        
            logging.info(
                'Starting to download the data for detected images...'
            )
            
            # Retrieve rich data (and optionally download files):
            rich_collection = self.fetch_images_by_ids(
                image_ids=ids, 
                fields=fields, 
                save_to_disk=save_to_disk,
                process_masks=process_masks
            )
            logging.info('Downloaded data for all detected images.')
            
            # Merge the rich data back into our tile-based collection:
            final_collection.merge(
               rich_collection,
               overwrite_properties = True,
               overwrite_path = save_to_disk,
               add_new=False
            )
        
        return final_collection

    def _download_image(self, url: str, destination: Path) -> bool:
        """
        Download a single image from a URL to a local file.
        
        Streams the response in chunks to avoid loading the entire file into 
        memory. On any network or HTTP-related error, logs the failure and 
        returns ``False``.
        
        Args:
            url (str):
                The URL of the image to download.
            destination (Path):
                The local filesystem path where the downloaded image will be
                saved.
        
        Returns:
            bool:
                True if the download completed successfully and the file was
                written to disk; False if an error occurred.
        """
        try:
            with self.session.get(
                    url, 
                    stream=True, 
                    timeout=REQUESTS_TIMEOUT_VAL
                ) as r:
                r.raise_for_status()
                with open(destination, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return True
        except requests.exceptions.RequestException as e:
            logging.error(f"Download failed for {url}: {e}")
            return False

    def _get_tile_image_data(
        self,
        tile_coords: tuple[int, int, int], 
        start_date='',
        end_date='',
        filter_rapid_only: bool = True
    ) -> list[ImageAsset]:
        """
        Extracts image assets from a single Mapillary vector tile.

        This helper retrieves a single tile from the Mapillary tiles API,
        decodes its vector data, filters images by source and optional
        capture-date range, converts tile coordinates to WGS84, and
        returns a list of corresponding ``ImageAsset`` objects.

        The method:
            * Downloads and (if needed) decompresses the Mapillary tile.
            * Decodes the Mapbox Vector Tile into layers/features.
            * Reads image features from the ``"image"`` layer.
            * Optionally filters out images not created by the RAPID creator.
            * Optionally filters images outside the requested capture date
              range.
            * Converts tile-based coordinates to latitude/longitude (WGS84).
            * Cleans up properties and builds ``ImageAsset`` instances.

        Args:
            tile_coords: 
                The tile coordinates as ``(x, y, z)`` in Web Mercator/XYZ 
                tiling scheme, where:
                    * ``x``: Tile column index.
                    * ``y``: Tile row index.
                    * ``z``: Zoom level.
            start_date: 
                Optional inclusive lower bound for the image
                capture date filter. Expected in ``"YYYY-MM-DD"`` format
                or empty string for no lower bound.
            end_date: 
                Optional inclusive upper bound for the image
                capture date filter. Expected in ``"YYYY-MM-DD"`` format
                or empty string for no upper bound.
            filter_rapid_only: 
                If ``True``, only images whose
                ``creator_id`` matches ``RAPID_CREATOR_ID`` are included.
                If ``False``, all creators are allowed.

        Returns:
            list[ImageAsset]: 
                A list of ``ImageAsset`` instances found within the specified
                tile that satisfy all filtering conditions. Returns an empty
                list if:
                    * The tile has no ``"image"`` layer or no image features,
                    * No images satisfy the RAPID/date filters, or
                    * The tile download/decoding fails (a warning is logged).

        Notes:
            * The method silently skips images with invalid or missing
              dates when a date filter is active.
            * Gzip-compressed tiles are automatically decompressed based
              on the Gzip magic header.
            * The resulting ``ImageAsset.path`` uses the image ID as the
              filename with a ``.jpg`` extension, stored under
              ``self.save_dir``.
        """
        x, y, z = tile_coords
        found_assets = []
    
        # Construct tile URL:
        tile_url = TILE_URL_TEMPLATE.format(
            z=z, x=x, y=y, token=self.access_token
        )
    
        try:
            # Download the tile data:
            response = self.session.get(tile_url, timeout=REQUESTS_TIMEOUT_VAL)
            response.raise_for_status()
            raw_data = response.content
    
            # Gzip compression check (Mapillary tiles may be compressed):
            if raw_data.startswith(b'\x1f\x8b'):
                raw_data = gzip.decompress(raw_data)
    
            # Decode tile data:
            decoded_tile = mapbox_vector_tile.decode(raw_data)
            
            # Get image data:
            image_layer = decoded_tile.get('image')
            if not image_layer:
                return [] # No images in this tile
                
            # Get the tile extent defined in the data (defaults to 4096 if
            # missing):
            tile_extent = image_layer.get('extent', 4096)
            images = image_layer.get('features', [])
            
            for image in images:
                props = image['properties']
    
                # Extract creator and image IDs:
                creator_id = props.pop('creator_id', None)
                prop_id = props.pop('id', None)
    
                # Depending on filter_rapid_only check if creator ID matches
                # RAPID's ID:
                if filter_rapid_only and creator_id != RAPID_CREATOR_ID:
                    continue
    
                # Convert geometry data from tile coordinates to WGS84:
                shape_xy = image['geometry']['coordinates']
                lon, lat = TileUtils.mvt_to_wgs84(
                    shape_xy[0], 
                    tile_extent - shape_xy[1],
                    x, y, z, 
                    extent=tile_extent
                )
    
                # Save latitude & longitude data in image properties:
                props['longitude'] = lon
                props['latitude'] = lat
    
                # Extract raw image timestamp and remove it from image props:
                timestamp_ms = props.pop('captured_at', None)
    
                if timestamp_ms:
                    # Convert image timestamp from UTC date to ISO format 
                    # (YYYY-MM-DD):
                    img_date = TileUtils.ms_to_date_utc(timestamp_ms)
    
                    # Skip if date falls outside the requested range:
                    if not self._is_date_in_range(
                            img_date, 
                            start_date, 
                            end_date
                        ):
                        continue
    
                    # Assign the clean date object to properties:
                    props['capture_date'] = img_date
                
                # If a date filter is active, but the image has NO date, 
                # skip the image.
                elif start_date or end_date:
                    continue
    
                # Remove 'organization_id' and 'sequence_id' from image props:
                for key in ['organization_id', 'sequence_id']:
                    props.pop(key, None)
    
                # Create and ImageAsset and add it to the list output:
                asset = ImageAsset(
                    path=self.save_dir / f"{prop_id}.jpg",
                    id=prop_id,
                    properties=props,
                    allow_missing_file=True
                )
                found_assets.append(asset)
        
        # If tile data cannot be downloaded log error:
        except Exception as e:
            logging.warning(f"Failed to process tile {z}/{x}/{y}: {e}")
    
        return found_assets
    
    @staticmethod
    def _is_date_in_range(
        check_date: date | datetime | str | None,
        start_date: date | datetime | str | None,
        end_date: date | datetime | str | None
    ) -> bool:
        """Checks whether a date falls within an inclusive date range.
    
        This method is robust to different input types and missing values. It
        accepts `date`, `datetime`, string (in ``YYYY-MM-DD`` format), or
        ``None`` for all parameters, and normalizes them to ``datetime.date`` 
        objects before comparison.
    
        The check is inclusive of both ``start_date`` and ``end_date``. If
        ``check_date`` is missing or invalid and at least one of
        ``start_date`` or ``end_date`` is provided (i.e., filters exist),
        the method returns ``False``. If all three inputs are missing or
        invalid (i.e., no filters), the method returns ``True``.
    
        Args:
            check_date: The date to be tested. May be a ``datetime.date``,
                ``datetime.datetime``, a string in ``"YYYY-MM-DD"`` format,
                or ``None``.
            start_date: The inclusive lower bound of the date range. May be a
                ``datetime.date``, ``datetime.datetime``, a string in
                ``"YYYY-MM-DD"`` format, or ``None``. If ``None``, there is
                no lower bound.
            end_date: The inclusive upper bound of the date range. May be a
                ``datetime.date``, ``datetime.datetime``, a string in
                ``"YYYY-MM-DD"`` format, or ``None``. If ``None``, there is
                no upper bound.
    
        Returns:
            bool: 
                ``True`` if ``check_date`` is within the inclusive range 
                defined by ``start_date`` and ``end_date`` (after 
                normalization), or if all three values are missing/invalid
                (no filters). ``False`` if ``check_date`` is outside the range,
                invalid, or missing while at least one of ``start_date`` or 
                ``end_date`` is provided.
    
        Notes:
            * String inputs must be in ``"YYYY-MM-DD"`` format; otherwise
              they are treated as invalid.
            * ``datetime.datetime`` inputs are converted to dates by
              discarding the time component.
            * Invalid or unparsable date strings are treated as ``None``.
        """
    
        def to_date(date_input):
            """Normalizes various date-like inputs to a ``datetime.date``."""
            if not date_input:
                return None
            if isinstance(date_input, datetime):
                return date_input.date()
            if isinstance(date_input, date):
                return date_input
            if isinstance(date_input, str):
                try:
                    return datetime.strptime(
                        date_input.strip(), "%Y-%m-%d"
                    ).date()
                except ValueError:
                    return None
            return None
    
        # Normalize inputs:
        current = to_date(check_date)
        start = to_date(start_date)
        end = to_date(end_date)
    
        # If the input date is invalid/missing:
        if current is None:
            # If filters exist, reject. If no filters exist, accept.
            return False if (start or end) else True
    
        # Check Range (Inclusive):
        if start and current < start:
            return False
        if end and current > end:
            return False
    
        return True

    def _get_image_metadata(
        self,
        image_id: str,
        fields: list[str],
        retries: int = 3,
        backoff_factor: float = 0.5
    ) -> dict[str, Any] | None:
        """Retrieve metadata for a specific image from the API.
    
        This method issues a GET request to the configured API endpoint using
        the provided image ID and list of field names. On success, the JSON
        response is returned as a dictionary. If any network or HTTP error
        occurs, the error is logged and ``None`` is returned.
    
        Args:
            image_id: 
                The unique identifier of the image whose metadata should be
                fetched.
            fields:
                Field names to request from the API. These are sent as a 
                comma-separated list via the ``fields`` query parameter.
    
        Returns:
            A dictionary containing the JSON metadata returned by the API if 
            the request succeeds; otherwise ``None``.
        """
        url = f"{BASE_URL}/{image_id}"
        params = {
            "access_token": self.access_token,
            "fields": ",".join(fields),
        }
    
        for attempt in range(retries):
            try:
                resp = self.session.get(
                    url, 
                    params=params, 
                    timeout=REQUESTS_TIMEOUT_VAL
                )
                resp.raise_for_status()  # Will raise for 4xx/5xx responses
                return resp.json()
            
            except requests.exceptions.RequestException as e:
                # Log the failed attempt:
                logging.warning(
                    "Attempt %d/%d failed for image %s: %s",
                    attempt + 1, retries, image_id, e
                )
                
                # If on the last attempt, break out of the loop:
                if attempt + 1 == retries:
                    break
                
                # Calculate wait time and sleep:
                wait_time = backoff_factor * (2 ** attempt)
                time.sleep(wait_time)
    
        # This part is reached only if all retries fail
        logging.error(
            f'All {retries} attempts to fetch metadata for {image_id} failed.'
            )
        return None

    @staticmethod
    def _parse_mapillary_segmentation(
        pano: ImageAsset, 
        target_classes: list[str] | None = None, 
        flip_y: bool = True,
        merge_detections: bool = True,
        show_progress: bool = False
    ) -> tuple[np.ndarray, dict[int, str]]:
        """
        Create a segmentation image for the pano using Mapillary detections.
        
        Args:
            pano: 
                ImageAsset object containing properties and detection data.
            target_classes: 
                Optional list of class prefixes to include.
            flip_y: 
                Whether to flip the Y-axis of the mask.
            merge_detections:
                If True (Semantic), all detections of the same label share one 
                ID. If False (Instance), every detection gets a unique ID. 
            show_progress:
                If True, displays a tqdm progress bar. Default is False to
                reduce clutter.

        Returns:
            mask_array (np.ndarray): 
                The rasterized segmentation mask as a NumPy array.
                - Dtype: ``uint8`` if max ID <= 255, otherwise `int32``.
                - Values: Integers corresponding to keys in 
                  ``segmentation_map``.
            segmentation_map (dict[int, str]): 
                A dictionary mapping pixel integer values to label names.
                - Semantic Mode: {1: 'car', 2: 'road'} (IDs grouped by class).
                - Instance Mode: {1: 'car', 2: 'car'} (Unique IDs per object).
        """
        # Extract image properties to avoid repeated lookups:
        props = pano.properties
        img_width = props.get('width')
        img_height = props.get('height')
        
        if not img_width or not img_height:
            logging.warning('ImageAsset missing dimension information.')
            return np.zeros((1, 1), dtype=np.uint8), {}

        # Get instance detections:
        detections = props.get('detections', {}).get('data', [])
        if not detections:
            return np.zeros((img_height, img_width), dtype=np.uint8), {}

        # Initialize state:
        label_to_id = {}
        segmentation_map = {0: SegmentationLabels.VOID} 
        next_id = 1
        
        # Start with 'L' (8-bit, max 255) for memory efficiency.
        # We will upgrade to 'I' (32-bit) only if necessary.
        image_mode = 'L'
        canvas = Image.new(image_mode, (img_width, img_height), 0)
        draw = ImageDraw.Draw(canvas)

        # Pre-process target classes for faster filtering:
        for item in tqdm(
                detections, 
                desc='Processing mask layers', 
                leave=False,
                disable=not show_progress
            ):
            label_value = item.get('value', SegmentationLabels.VOID)

            # Fast filtering:
            if target_classes:
                if not any(label_value.startswith(tc) for tc in target_classes):
                    continue

            # ID Management:            
            if merge_detections:
                # Semantic segmentation mode:
                if label_value in label_to_id:
                    current_id = label_to_id[label_value]
                else:
                    label_to_id[label_value] = next_id
                    segmentation_map[next_id] = label_value
                    current_id = next_id
                    next_id += 1
            else:
                # Instance segmentation mode:
                current_id = next_id
                segmentation_map[current_id] = label_value
                next_id += 1
                
            # Check if 32-bit upgrade is necessary:
            if current_id > 255 and image_mode == 'L':
                image_mode = 'I'
                canvas = canvas.convert(image_mode)
                draw = ImageDraw.Draw(canvas)

            # Geometry decoding:
            b64_string = item.get('geometry')
            if not b64_string: continue

            try:
                # Decode Protobuf:
                pbf_bytes = base64.decodebytes(b64_string.encode('utf-8'))
                decoded_tile = mapbox_vector_tile.decode(pbf_bytes)
                
                if not decoded_tile: continue
                
                # Get the first layer (standard MVT structure):
                layer_name = next(iter(decoded_tile.keys()))
                layer_data = decoded_tile[layer_name]
                extent = layer_data['extent']

                # Coordinate transformation. Calculate scalars once per tile:
                scale_x = img_width / extent
                scale_y = img_height / extent

                # Define the transformation:
                def transform_poly(ring_coords):
                    if flip_y:
                        return [(x * scale_x, img_height - (y * scale_y)) \
                                for x, y in ring_coords]
                    return [(x * scale_x, y * scale_y) for x, y in ring_coords]

                # Drawing:
                for feature in layer_data['features']:
                    geom = feature['geometry']
                    g_type = geom['type']
                    coords = geom['coordinates']

                    if g_type == 'Polygon':
                        for ring in coords:
                            poly_pts = transform_poly(ring)
                            if len(poly_pts) > 2:
                                draw.polygon(poly_pts, fill=current_id)
                                
                    elif g_type == 'MultiPolygon':
                        for poly in coords:
                            for ring in poly:
                                poly_pts = transform_poly(ring)
                                if len(poly_pts) > 2:
                                    draw.polygon(poly_pts, fill=current_id)

            except Exception as e:
                logging.error(f"Error processing mask geometry: {e}")
                continue

        mask_array = np.array(
            canvas, 
            dtype=np.uint8 if image_mode == 'L' else np.int32
        )
        return mask_array, segmentation_map

    def _validate_fields(
        self, 
        fields: list[str]|None, 
        image_id: str = "batch",
        require_segmentation: bool = False
    ) -> list[str]|None:
        """
        Validate and normalize requested image fields for API requests.
        
        This helper:
        
        - Filters out any requested fields that are not in 
          AVAILABLE_IMAGE_FIELDS.
        - Logs a warning for any invalid/unknown fields that are omitted.
        - Ensures that at least one valid field remains; otherwise, logs an 
          error and returns None.
        - Ensures that "thumb_original_url" is always included in the returned
          list of fields.
         - Automatically injects "detections.value" and "detections.geometry"
           if segmentation processing is required.
        
        Args:
            fields (list[str] | None): 
                List of requested image field names. If None, all fields 
                defined in self.AVAILABLE_IMAGE_FIELDS are returned.
            image_id (str): 
                Identifier of the image or batch used only for logging context
                when reporting invalid or missing fields. Defaults to 
                ``batch``.
            require_segmentation (bool):
                If ``True``, explicitly adds ``detections.value`` and 
                ``detections.geometry`` to the requested fields list to ensure 
                segmentation data is retrieved. Defaults to ``False``.
        
        Returns:
            list[str] | None: 
                A list of validated field names that will be used in API 
                requests. The list is guaranteed to include 
                ``thumb_original_url``. Returns ``None`` if no valid fields are
                found after validation.
        """
        # Initialize the set of fields to validate:
        if fields is None:
            # If default, we take everything available:
            current_fields = set(self.AVAILABLE_IMAGE_FIELDS)
        else:
            # Otherwise, start with the user's requested list:
            current_fields = set(fields)
         
        # Inject segmentation dependencies if required:
        if require_segmentation:
            current_fields.add('detections.value') 
            current_fields.add('detections.geometry')

        # Filter against the schema. Only keep fields that exist in the known
        # available set:        
        valid_fields = current_fields.intersection(
            self.AVAILABLE_IMAGE_FIELDS
        )

        # Check for invalid fields:
        if fields is not None:
            invalid_fields = current_fields.difference(
                self.AVAILABLE_IMAGE_FIELDS
            )

            if invalid_fields:
                logging.warning(
                    f"Invalid field(s) omitted: {', '.join(invalid_fields)}"
                )
        
        # Fail if nothing remains:
        if not valid_fields:
            logging.error(
                f'No valid fields provided for {image_id}. Aborting.'
            )
            return None
            
        final_fields = list(valid_fields)
        
        # We always need the download URL:
        if 'thumb_original_url' not in final_fields:
            final_fields.append('thumb_original_url')
            
        return final_fields
    