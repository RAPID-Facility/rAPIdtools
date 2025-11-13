import os
import requests
import logging

from rapidtools.core import BoundingBox 

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

    BASE_URL = "https://graph.mapillary.com"

    def __init__(self, access_token: str, save_dir: str = 'mapillary_images'):
        if not access_token:
            raise ValueError('Mapillary access token is required.')
        self.access_token = access_token
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.session = requests.Session()


    def fetch_image_details(
        self, 
        image_id: str, 
        include_detections: bool = False
    ):
        """
        Fetches detailed metadata for a single Mapillary image by its ID.

        Args:
            image_id (str): 
                The unique identifier for the Mapillary image.
            include_detections (bool): 
                If True, requests detection data. Defaults to False.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the image metadata,
                                      or None if the request fails.
        """
        url = f"{self.BASE_URL}/{image_id}"
        
        # Define the fields to be requested:
        fields = [
            "id", "computed_geometry", "thumb_original_url", "width", "height",
            "compass_angle", "computed_compass_angle", "altitude"
        ]
        
        # Conditionally add the optional 'detections.value' field:
        if include_detections:
            fields.append("detections.value")
        
        
        params = {
            "access_token": self.access_token,
            "fields": ",".join(fields)  
        }
        
        try:
            # Use the session object for the request
            response = self.session.get(url, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch details for image {image_id}: {e}")
            return None

    def fetch_image_ids_by_bbox(self, bbox: BoundingBox) -> list[str]:
        """
        Get a list of unique Mapillary image IDs within a given BoundingBox region.

        This method leverages the BoundingBox object's built-in tiling feature
        to automatically handle large areas. If the area is larger than 0.01
        square degrees, it is split into smaller tiles, and an API request is
        made for each one. The results are then combined and deduplicated.

        Args:
            bbox (BoundingBox): A BoundingBox object representing the area of interest.

        Returns:
            List[str]: A list of unique image IDs found within the area.
        """
        api_endpoint = f'{self.BASE_URL}/images'
        
        # Tile the BoundingBox object, if needed:
        tiles = bbox.tile_by_area(max_area=0.01)
        multiple_tiles = len(tiles) > 1
        
        # Iterate through the tiles and perform the API call for each one:
        unique_ids = set()
        for idx, tile in enumerate(tiles):
            try:
                if multiple_tiles:
                    logging.info(
                        f'Fetching data for tile {idx + 1}/{len(tiles)}...'
                        )
                
                # Get bounding box bounds and set API parameters:
                min_lon, min_lat, max_lon, max_lat = tile.shapely.bounds
                bbox_str = f'{min_lon},{min_lat},{max_lon},{max_lat}'
                
                params = {
                    'access_token': self.access_token,
                    'creator_username': 'uwrapid',
                    'fields': 'id',
                    'bbox': bbox_str
                }
                
                # Call Mapillary API and raise an exception for bad status
                # codes (4xx or 5xx):
                response = self.session.get(api_endpoint, params=params)
                response.raise_for_status() 
                try:
                    data = response.json()
                except ValueError as json_err:
                    logging.warning(
                        f'Could not parse JSON for tile {idx}: {json_err}. '
                        'Skipping.'
                        )
                    continue
                
                ids_for_tile = [img['id'] for img in data.get('data', [])]
                
                if ids_for_tile:
                    unique_ids.update(ids_for_tile)
                
                if multiple_tiles:
                    logging.info(
                        f'Found {len(ids_for_tile)} images in tile {idx + 1}.'
                        )

            except requests.exceptions.RequestException as e:
                logging.warning(
                    f'Could not fetch data for tile {tile}: {e}. Continuing.'
                    )

        logging.info(f'Total unique image IDs found: {len(unique_ids)}')
        
        # Return the final, deduplicated list:
        return list(unique_ids)



