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
# 12-10-2025

import gzip
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any

import mapbox_vector_tile
import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

from rapidtools.core import BoundingBox, ImageAsset, ImageCollection
from .tile_utils import TileUtils

# Mapillary Required API data:
BASE_URL = 'https://graph.mapillary.com'
TILE_URL_TEMPLATE = (
    "https://tiles.mapillary.com/maps/vtp/mly1_public/2/"
    "{z}/{x}/{y}?access_token={token}"
)
RAPID_CREATOR_ID = 107708041466249

# Requests download strategy:
REQUESTS_TIMEOUT_VAL = 30
REQUESTS_RETRY_STRATEGY = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
REQUESTS_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/131.0.0.0 Safari/537.36'
}

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
        'sequence', 'sfm_cluster', 'width', 'detections'
    ])

    def __init__(self, access_token: str, save_dir: str = 'mapillary_images'):
        if not access_token:
            raise ValueError('Mapillary access token is required.')
        self.access_token = access_token
        self.save_dir = Path(save_dir)
        
        # Create a session with the Retry strategy and the default headers:
        self.session = requests.Session()                
        
        adapter = HTTPAdapter(max_retries=REQUESTS_RETRY_STRATEGY)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        self.session.headers.update(REQUESTS_HEADERS)

    def fetch_image(
        self, 
        image_id: str, 
        fields: list[str] | None,
        save_to_disk: bool = True
    )-> ImageAsset | None:
        """
        Fetch image metadata using it Mapillary ID and optionally download it.
        
        This method:
            - Validates the requested metadata fields.
            - Fetches image metadata from the API.
            - Extracts the download URL from the metadata.
            - Downloads the image file to the configured save directory.
            - Returns an ImageAsset containing the file path, ID, and 
              remaining  metadata.
        
        Args:
            image_id (str):
                The identifier of the image to fetch from the API.
            fields (list[str] | None):
                Optional list of metadata fields to request. If ``None``, a 
                default set of fields (as defined by ``_validate_fields``)
                will be used. Invalid fields are ignored with a warning.
            save_to_disk (bool): 
                If ``True``, downloads the image to the configured save 
                directory. Defaults to ``True``.
        
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
                    - ``None`` if validation fails, metadata retrieval fails, no 
                      download URL is present, or the download itself fails.
        """
        
        # Validate requested fields:
        fields_to_request = self._validate_fields(fields, image_id)
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
        
        # Create the ImageAsset:
        image_asset = ImageAsset(
            path=str(file_path),
            id=prop_id,
            properties=props,
            allow_missing_file=allow_missing
        )
        
        # Optionally download the image to disk:
        if save_to_disk:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            if not self._download_image(image_url, file_path):
                return None  # Download failed

        return image_asset


    def fetch_images_by_ids(
        self, 
        image_ids: list[str], 
        fields: list[str] | None,
        save_to_disk: bool = True,
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
                    save_to_disk): img_id 
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
        save_to_disk: bool = True,
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
            fields (list[str] | None):
                A list of specific metadata fields to fetch (e.g., 
                ``['captured_at', 'compass_angle']``). If ``None``, 
                defaults to all available fields. Note: The download URL is
                always fetched internally.
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
        tiles_list = TileUtils.bbox_to_mapbox_tiles(bbox)
        
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
                save_to_disk=save_to_disk
            )
            logging.info('Downloaded data for all detected images.')
            
            # Merge the rich data back into our tile-based collection:
            final_collection.combine_with(
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

    def _is_date_in_range(
        self,
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
        url = f"{self.BASE_URL}/{image_id}"
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

    def _validate_fields(
        self, 
        fields: list[str]|None, 
        image_id: str = "batch"
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
        
        Args:
            fields (list[str] | None): 
                List of requested image field names. If None, all fields 
                defined in self.AVAILABLE_IMAGE_FIELDS are returned.
            image_id (str): 
                Identifier of the image or batch used only for logging context
                when reporting invalid or missing fields. Defaults to 
                ``batch``.
        
        Returns:
            list[str] | None: 
                A list of validated field names that will be used in API 
                requests. The list is guaranteed to include 
                ``thumb_original_url``. Returns ``None`` if no valid fields are
                found after validation.
        """
        
        if fields is None:
            return list(self.AVAILABLE_IMAGE_FIELDS)
        
        requested_set = set(fields)
        valid_fields = requested_set.intersection(self.AVAILABLE_IMAGE_FIELDS)
        invalid_fields = requested_set.difference(self.AVAILABLE_IMAGE_FIELDS)

        if invalid_fields:
            logging.warning(
                f"Invalid field(s) omitted: {', '.join(invalid_fields)}"
            )
        
        if not valid_fields:
            logging.error(
                f'No valid fields provided for {image_id}. Aborting.'
            )
            return None
            
        final_fields = list(valid_fields)
        # We always need the download URL
        if 'thumb_original_url' not in final_fields:
            final_fields.append('thumb_original_url')
        return final_fields
    