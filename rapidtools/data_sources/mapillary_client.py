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

from collections import deque
import logging
import requests
import concurrent.futures
from pathlib import Path
import time

from rapidtools.core import BoundingBox, ImageAsset

class MapillaryClient:
    """
    Fetches and stores Mapillary images for given geo-coordinates, bounding boxes, or image IDs.
    Requires a Mapillary API access token.
    """

    BASE_URL = 'https://graph.mapillary.com'

    AVAILABLE_IMAGE_FIELDS = frozenset([
        'altitude', 'atomic_scale', 'camera_parameters', 'camera_type',
        'captured_at', 'compass_angle', 'computed_altitude', 
        'computed_compass_angle', 'computed_geometry', 'computed_rotation', 
        'creator', 'exif_orientation', 'geometry', 'height', 'is_pano', 
        'make', 'model', 'thumb_256_url', 'thumb_1024_url', 
        'thumb_2048_url', 'thumb_original_url', 'merge_cc', 'mesh', 
        'sequence', 'sfm_cluster', 'width', 'detections'
    ])

    # --- Class constants for API limitations ---
    # The hard limit for the number of image features the API will return:
    API_FEATURE_LIMIT = 2000
    
    # The area limit (in sq. degrees) for a bbox query to be accepted by the 
    # API:
    API_MAX_TILE_AREA = 0.01


    def __init__(self, access_token: str, save_dir: str = 'mapillary_images'):
        if not access_token:
            raise ValueError('Mapillary access token is required.')
        self.access_token = access_token
        self.save_dir = Path(save_dir)
        self.session = requests.Session()

    def fetch_image(
            self, 
            image_id: str, 
            fields: list[str] | None
            )-> ImageAsset | None:
        """
        Fetch image and its metadata using it Mapillary ID.
        
        This method:
        - Validates the requested metadata fields.
        - Fetches image metadata from the API.
        - Extracts the download URL from the metadata.
        - Downloads the image file to the configured save directory.
        - Returns an ImageAsset containing the file path, ID, and remaining 
          metadata.
        
        Args:
            image_id (str):
                The identifier of the image to fetch from the API.
            fields (list[str] | None):
                Optional list of metadata fields to request. If ``None``, a 
                default set of fields (as defined by ``_validate_fields``)
                will be used. Invalid fields are ignored with a warning.
        
        Returns:
            ImageAsset | None:
                - An ImageAsset instance containing:
                  - path (str): Local path to the downloaded image file.
                  - id (str): The image ID (API `id` if present, otherwise the
                    requested `image_id`).
                  - properties (dict): Remaining metadata fields returned by
                    the API (excluding the file `id` and used download URL).
                - None if validation fails, metadata retrieval fails, no 
                  download URL is present, or the download itself fails.
        """
        
        # Validate requested fields:
        fields_to_request = self._validate_fields(fields, image_id)
        if not fields_to_request:
            return None

        # Fetch Metadata:
        url = f'{self.BASE_URL}/{image_id}'
        params = {
            'access_token': self.access_token, 
            'fields': ','.join(fields_to_request)
        }

        try:
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            props = resp.json()
        except requests.exceptions.RequestException as e:
            logging.error(f'Failed to fetch metadata for {image_id}: {e}')
            return None

        image_url = props.get('thumb_original_url')
        if not image_url:
            logging.error(f"No download URL found for {image_id}")
            return None

        # Download Image:
        prop_id = props.pop('id', image_id)
        file_path = self.save_dir / f"{prop_id}.jpg"
        
        # Package the downloaded image and metadata as an ImageAsset and 
        # return it:
        if self._download_image(image_url, file_path):
            return ImageAsset(
                path=str(file_path), 
                id=prop_id, 
                properties=props
            )
        return None

    def fetch_images_by_ids(
            self, 
            image_ids: list[str], 
            fields: list[str] | None
            ) -> list[ImageAsset]:
        """
        Concurrently donwload multiple images by ID.
        
        This method:
        - Submits one fetch task per image ID to a thread pool.
        - Fetches metadata and downloads each image (via fetch_image).
        - Aggregates successfully fetched ImageAsset objects into a list.
        - Logs any exceptions that occur per image without aborting the entire batch.
        
        Args:
            image_ids (list[str]):
                A list of image IDs to fetch and download.
            fields (list[str] | None):
                Optional list of metadata fields to request for each image.
                If ``None``, a default set of fields (as determined by
                ``_validate_fields``) will be used.
        
        Returns:
            list[ImageAsset]:
                A list of ImageAsset instances for all images that were
                successfully fetched and downloaded. Images that fail are
                omitted from the result list.
        """
        results = []
        
        # Use ThreadPoolExecutor to download in parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Create a dictionary to map futures to IDs:
            future_to_id = {
                executor.submit(self.fetch_image, img_id, fields): img_id 
                for img_id in image_ids
            }
            
            # Iterate over futures as they complete (in the order they finish,
            # not necessarily the order submitted):
            for future in concurrent.futures.as_completed(future_to_id):
                img_id = future_to_id[future]
                try:
                    asset = future.result()
                    # Only append successfully created ImageAsset objects:
                    if asset:
                        results.append(asset)
                except Exception as e:
                    logging.error(f"Exception for image {img_id}: {e}")
        
        return results

    def fetch_images_in_bbox(
        self, 
        bbox: BoundingBox, 
        fields: list[str] | None = None,
        creator_username: str = 'uwrapid',
        start_date: str = '',
        end_date: str = ''
    ) -> list[ImageAsset]:
        """
        Get all Mapillary images within a geographic bounding box.

        This method acts as a high-level orchestrator to ensure complete 
        data coverage over large or dense areas. It performs two main 
        operations:

        1.  **Adaptive ID Search:** It queries the Mapillary API using a 
            recursive tiling strategy (quadtree). If a specific tile hits
            the API's limit  (2000 features), it automatically subdivides
            the area to ensure no image IDs are missed.
        2.  **Concurrent Download:** Once all unique IDs are gathered, it
            downloads the image files and their associated metadata in
            parallel.

        Args:
            bbox (BoundingBox):
                The geographic rectangular area to search. Must be an 
                instance of ``rapidtools.core.BoundingBox``.
            fields (list[str] | None):
                A list of specific metadata fields to fetch (e.g., 
                ``['captured_at', 'compass_angle']``). If ``None``, 
                defaults to all available fields. Note: The download URL is
                always fetched internally.
            creator_username (str):
                The Mapillary username to filter by. Only images uploaded 
                by this user will be retrieved. Defaults to ``'uwrapid'``.
            start_date (str):
                Filter for images captured on or after this date. 
                Format: 'YYYY-MM-DD' or ISO 8601 (e.g., '2023-01-01'). 
                Defaults to ``''`` (no filter).
            end_date (str):
                Filter for images captured on or before this date. 
                Format: 'YYYY-MM-DD' or ISO 8601 (e.g., '2023-12-31'). 
                Defaults to ``''`` (no filter).

        Returns:
            list[ImageAsset]:
                A list of ``ImageAsset`` objects containing the local file path, 
                image ID, and requested metadata properties. Returns an empty 
                list if no images are found.
        """
        # Get the list of image IDs for the bounding box:
        ids = self.fetch_image_ids_for_bbox(
            bbox, 
            creator_username=creator_username,
            start_date=start_date,
            end_date=end_date
        )
        
        if not ids:
            logging.info("No images found in the specified bounding box.")
            return []

        logging.info(f"Starting download of {len(ids)} images...")

        # Download the ImageAssets:
        return self.fetch_images_by_ids(ids, fields)

    def fetch_image_ids_for_bbox(
        self, 
        bbox: BoundingBox,
        creator_username: str = 'uwrapid',
        start_date: str = '',
        end_date: str = '',
        initial_tile_area: float = API_MAX_TILE_AREA,
        max_retries: int = 5,
        split_threshold: int = API_FEATURE_LIMIT
        
    ) -> list[str]:
        """
        Get unique IDs within a bounding, using a two-stage tiling strategy.
        
        This method queries the Mapillary Images API over the given bounding 
        box using a two-stage, adaptive tiling strategy to avoid hitting API 
        feature limits and to reduce missed results in dense areas.
        
        The process is:
        - Tile the input bounding box into initial tiles with a maximum area
          (initial_tile_area).
        - For each tile, request image IDs up to a limit (split_threshold).
        - If a tile returns exactly split_threshold features (i.e., is 
          "saturated"), subdivide it into 4 smaller tiles and re-queue them for
          processing.
        - Repeat until all tiles are processed or skipped after max_retries.
        - Collect and return the set of unique image IDs found across all tiles.
        
        Args:
            bbox (BoundingBox):
                The geographic area of interest to search within.
            creator_username (str):
                Mapillary username used to filter images. Only images created
                by this user will be returned. Defaults to ``'uwrapid'``.
            start_date (str, optional):
                Filter for images captured after this date. 
                Format: 'YYYY-MM-DD' or ISO 8601 'YYYY-MM-DDTHH:MM:SS'.
            end_date (str, optional):
                Filter for images captured before this date. 
                Format: 'YYYY-MM-DD' or ISO 8601 'YYYY-MM-DDTHH:MM:SS'.    
            initial_tile_area (float):
                Maximum area (in square degrees) for the initial tiling of the
                bounding box. Smaller values create more, smaller tiles and may
                provide more balanced queries at the cost of more API calls.
            max_retries (int):
                Maximum number of retry attempts for a failed API request on
                a single tile. Retries use exponential backoff (1, 2, 4, ... 
                seconds).
            split_threshold (int):
                Maximum number of features (image records) to request per tile.
                If a tile returns exactly this number, it is assumed to be
                saturated and is split into 4 sub-tiles for finer querying.
        
        Returns:
            list[str]:
                A list of unique image IDs found within the bounding box after
                processing all tiles. Order is not guaranteed.
        """
        # Validate the user-specified threshold against the fixed API limit:
        if split_threshold > self.API_FEATURE_LIMIT:
            logging.warning(
                f'split_threshold ({split_threshold}) cannot exceed the API '
                'limit of {self.API_FEATURE_LIMIT}. Capping at '
                '{self.API_FEATURE_LIMIT}.'
            )
            split_threshold = self.API_FEATURE_LIMIT

        # Format the dates so that they work with the API:
        formatted_start = self._format_date(start_date, is_end_date=False)
        formatted_end = self._format_date(end_date, is_end_date=True)

        # Perform initial tiling of the specified bounding box using the 
        # specified maximum area:
        logging.info(
            f'Performing initial tiling with max area of {initial_tile_area} '
            'square degrees...'
            )
        initial_tiles = bbox.tile_by_area(max_area=initial_tile_area)
        tiles_to_process = deque(initial_tiles)

        # Define the API endpoint and initialize the set of image IDs:
        api_endpoint = f'{self.BASE_URL}/images'
        unique_ids = set()
        
        # Process the queue of tiles:
        logging.info(
            f'Starting adaptive fetch with {len(tiles_to_process)} '
            'initial tile(s).'
            )
        logging.info(
            f'Will split any tile returning the maximum of {split_threshold}'
            ' features.'
            )
        while tiles_to_process:
              tile = tiles_to_process.popleft()
              
              for attempt in range(max_retries):
                  try:
                      min_lon, min_lat, max_lon, max_lat = tile.shapely.bounds
                      bbox_str = f'{min_lon},{min_lat},{max_lon},{max_lat}'
                      
                      params = {
                          'access_token': self.access_token,
                          'bbox': bbox_str,
                          'fields': 'id',
                          'limit': str(split_threshold)
                      }
                      
                      # Apply optional filters:
                      if creator_username:
                          params['creator_username'] = creator_username
                      if start_date:
                          params['start_captured_at'] = formatted_start
                      if end_date:
                          params['end_captured_at'] = formatted_end
                      
                      
                      response = self.session.get(api_endpoint, params=params)
                      response.raise_for_status() # Raises HTTPError
                      data = response.json() # Raises ValueError if bad JSON
                      
                      # If we get here, the request was successful
                      ids_for_tile = [img['id'] for img in data.get('data', [])]
                      num_found = len(ids_for_tile)
        
                      if num_found == split_threshold:
                          logging.warning(
                              f'Tile saturated ({num_found} features). '
                              'Splitting into 4 sub-tiles.'
                              )
                          sub_tiles = tile.split()
                          tiles_to_process.extend(sub_tiles)
                      else:
                          logging.info(
                              f'Found {num_found} images in this tile.'
                              )
                          unique_ids.update(ids_for_tile)
                      
                      break
        
                  except (requests.exceptions.RequestException, ValueError) as e:
                      logging.warning(
                          f'Attempt {attempt + 1} of {max_retries} failed for '
                          f'a tile: {e}.'
                          )
                      if attempt < max_retries - 1:
                          # Wait before retrying (e.g., 1, 2, 4, 8 seconds):
                          wait_time = 2 ** attempt 
                          logging.info(f'Retrying in {wait_time} second(s)...')
                          time.sleep(wait_time)
                      else:
                          # This was the last attempt:
                          logging.error(
                              f'All {max_retries} attempts failed for the tile'
                              f'. Skipping.'
                              )

        logging.info(
            'Finished processing. Total unique image IDs found: '
            f'{len(unique_ids)}'
        )
        
        return list(unique_ids)

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
            with self.session.get(url, stream=True, timeout=20) as r:
                r.raise_for_status()
                with open(destination, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return True
        except requests.exceptions.RequestException as e:
            logging.error(f"Download failed for {url}: {e}")
            return False

    def _format_date(self, date_str: str, is_end_date: bool = False) -> str:
        """
        Normalize a date string to ISO 8601 format with a UTC suffix (Z).
        
        This helper ensures that date strings used in API queries include a
        full timestamp and an explicit UTC timezone (Z). It supports:
        - Plain dates: "YYYY-MM-DD"
          - Start date (default): "2025-06-01" -> "2025-06-01T00:00:00Z"
          - End date: "2025-06-01" -> "2025-06-01T23:59:59Z"
        - Date-times without timezone:
          - "2025-06-01T12:00:00" -> "2025-06-01T12:00:00Z"
        - Date-times that already include "Z" or a "+/-offset" are returned
          unchanged.
        
        Args:
            date_str (str):
                The input date or datetime string to normalize. If empty or
                falsy, an empty string is returned.
            is_end_date (bool):
                Whether the date represents an end-of-range boundary. If True
                and the input is a plain date (YYYY-MM-DD), the time
                component is set to 23:59:59; otherwise, it is set to 00:00:00.
                Defaults to False.
        
        Returns:
            str:
                The normalized ISO 8601 datetime string with a "Z" timezone
                suffix, or an empty string if the input was empty.
        """
        if not date_str:
            return ''
            
        # If the date  input is a simple date (YYYY-MM-DD), append time and Z
        if len(date_str) == 10 and date_str.count('-') == 2:
            # If it is an end date, grab the very end of that day:
            suffix = "T23:59:59Z" if is_end_date else "T00:00:00Z"
            return f"{date_str}{suffix}"
            
        # If the date has time but is missing timezone info (no Z or +offset)
        # e.g. "2025-06-01T12:00:00" -> "2025-06-01T12:00:00Z":
        if 'Z' not in date_str and '+' not in date_str:
            return f"{date_str}Z"

        return date_str

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
    