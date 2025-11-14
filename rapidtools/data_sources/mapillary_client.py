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

from collections import deque
import logging
import requests
import time

from rapidtools.core import BoundingBox, ImageAsset

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
    )

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
        self.save_dir = save_dir
        self.session = requests.Session()


    def fetch_pano(
        self,
        image_id: str,
        fields: list[str] | None = None
    ) -> ImageAsset | None:
        """
        Fetches a single Mapillary image and its detailed metadata.

        If any requested metadata fields are invalid, a warning will be logged
        and they will be omitted from the API request.

        Args:
            image_id (str):
                The unique identifier for the Mapillary image.
            fields (list[str] | None):
                A list of specific fields to request. If ``None``, it defaults
                to all available fields. Defaults to ``None``.

        Returns:
            dict[str, Any] | None: A dictionary containing the image metadata,
                                   or None if the request fails or no valid
                                   fields are provided.
        """
        # Determine which fields to request for the given image. If no fields
        # are specified (fields is None), request all available fields:
        if fields is None:
            fields_to_request = list(self.AVAILABLE_IMAGE_FIELDS)
        else:
            # Separate the provided fields into valid and invalid based on
            # AVAILABLE_IMAGE_FIELDS:
            valid_fields = []
            invalid_fields = []
            for field in fields:
                if field in self.AVAILABLE_IMAGE_FIELDS:
                    valid_fields.append(field)
                else:
                    invalid_fields.append(field)

            # Warn the user if any requested fields are invalid and will be 
            # dropped:
            if invalid_fields:
                logging.warning(
                    f'Invalid field(s) requested and will be omitted: '
                    f"{', '.join(invalid_fields)}."
                )
            
            # If no valid fields remain, abort the operation and log an error:
            if not valid_fields:
                logging.error(
                    f'No valid fields provided for image ID '
                    f'{image_id}. Aborting request.'
                    )
                return None
            
            # Use only the valid fields for the API request:
            fields_to_request = valid_fields
        
        # Construct the request URL and query parameters for the API call:
        url = f'{self.BASE_URL}/{image_id}'
        params = {
            'access_token': self.access_token,                  
            'fields': ','.join(fields_to_request)               
        }
        
        try:
            # Send a GET request to fetch the image details:
            response = self.session.get(url, params=params)

            # Raise an HTTPError if the response indicates a failure 
            # (4xx or 5xx):
            response.raise_for_status()

            # Return the parsed JSON response on success.
            return response.json()
        
        except requests.exceptions.RequestException as e:
            # Log the error if the request fails due to network or HTTP issues:
            logging.error(f'Failed to fetch the image {image_id}: {e}')
            return None

    def fetch_image_ids_by_bbox(
        self, 
        bbox: BoundingBox,
        creator_username: str = 'uwrapid',
        initial_tile_area: float = API_MAX_TILE_AREA,
        max_retries: int = 5,
        split_threshold: int = API_FEATURE_LIMIT
    ) -> list[str]:
        '''
        Get unique IDs within a bounding, using a two-stage tiling strategy.
        
        Args:
            bbox (BoundingBox):
                The area of interest.
            creator_username (str):
                Filter images by a specific Mapillary username.
            initial_tile_area (float): 
                The max area for the initial tiles in square degrees.
            split_threshold (int):
                The number of features at which to split a tile.
            max_retries (int):
                The number of times to retry a failed API request for a tile.
        '''
        # Validate the user-specified threshold against the fixed API limit:
        if split_threshold > self.API_FEATURE_LIMIT:
            logging.warning(
                f'split_threshold ({split_threshold}) cannot exceed the API '
                'limit of {self.API_FEATURE_LIMIT}. Capping at '
                '{self.API_FEATURE_LIMIT}.'
            )
            split_threshold = self.API_FEATURE_LIMIT

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
                          'creator_username': creator_username,
                          'fields': 'id',
                          'limit': str(split_threshold)
                      }
                      
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
