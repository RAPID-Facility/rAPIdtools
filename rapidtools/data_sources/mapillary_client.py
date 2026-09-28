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
# 09-28-2026

"""
Client for retrieving street-level imagery and metadata from Mapillary.

This module wraps the two public Mapillary endpoints used by rapidtools:

    1. The Graph API (``https://graph.mapillary.com``), used to retrieve
       per-image metadata such as capture time, camera heading, image
       dimensions and download URLs.
    2. The vector tile API (``https://tiles.mapillary.com``), used to
       discover which images exist inside a geographic bounding box.

The main entry point is :class:`MapillaryClient`, which converts API
responses into :class:`rapidtools.core.ImageAsset` and
:class:`rapidtools.core.ImageCollection` objects and can optionally
rasterize Mapillary's segmentation detections into semantic or instance
masks.

Example:
    >>> from rapidtools.core import BoundingBox
    >>> from rapidtools.data_sources.mapillary_client import MapillaryClient
    >>>
    >>> client = MapillaryClient('MLY|123|abc', save_dir='panos')
    >>> bbox = BoundingBox(-118.15, 34.18, -118.14, 34.19)
    >>> images = client.fetch_images_in_bbox(bbox, save_to_disk=False)
    >>> len(images)  # doctest: +SKIP
    42
"""

import base64
import gzip
import json
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

from rapidtools.config import REQUESTS_TIMEOUT_VAL, get_configured_session
from rapidtools.core import BoundingBox, ImageAsset, ImageCollection

from .tile_utils import TileUtils

logger = logging.getLogger(__name__)

# Mapillary API data:
BASE_URL = 'https://graph.mapillary.com'
TILE_URL_TEMPLATE = (
    'https://tiles.mapillary.com/maps/vtp/mly1_public/2/'
    '{z}/{x}/{y}?access_token={token}'
)
RAPID_CREATOR_ID = 107708041466249


class SegmentationLabels:
    """
    A namespace for standard Mapillary segmentation category labels.

    This class acts as a static container for string constants used to
    identify specific semantic classes in Mapillary data. Using these
    attributes instead of raw strings prevents typos and ensures consistency
    when performing dictionary lookups or conditional logic.

    Attributes:
        VOID (str):
            Label for undefined, void, or unlabeled regions
            (``'void--unlabeled'``).
        SKY (str):
            Label for the sky region (``'nature--sky'``).
        ROAD (str):
            Label for flat, driveable road surfaces
            (``'construction--flat--road'``).
        SURVEY_VEHICLE (str):
            Label for the image-capturing vehicle itself
            (``'void--ego-vehicle'``).

    Example:
        Using the class prevents typos in string literals:

        >>> label_map = {'nature--sky': 27, 'construction--flat--road': 13}
        >>> label_map.get(SegmentationLabels.SKY)
        27

        Useful for readable conditional logic:

        >>> current_pixel = 'nature--sky'
        >>> if current_pixel == SegmentationLabels.SKY:
        ...     print('Detected Sky')
        Detected Sky
    """

    VOID = 'void--unlabeled'
    SKY = 'nature--sky'
    ROAD = 'construction--flat--road'
    SURVEY_VEHICLE = 'void--ego-vehicle'


class MapillaryClient:
    """
    Fetch and store Mapillary images by ID or geographic bounding box.

    The client resolves image metadata through the Mapillary Graph API,
    discovers images inside an area through Mapillary vector tiles, and
    optionally downloads image files and rasterizes segmentation detections
    into masks. A Mapillary API access token is required.

    Attributes:
        access_token (str):
            The Mapillary API token sent with every request.
        save_dir (Path):
            Directory where downloaded images and masks are written.
        session (requests.Session):
            A pre-configured session with retry logic and default headers used
            for image downloads.
        AVAILABLE_IMAGE_FIELDS (frozenset[str]):
            The set of metadata fields that may be requested from the Graph
            API ``image`` endpoint.

    Example:
        >>> from rapidtools.data_sources.mapillary_client import MapillaryClient
        >>>
        >>> client = MapillaryClient('MLY|123|abc', save_dir='panos')
        >>> asset = client.fetch_image(
        ...     '1234567890',
        ...     fields=['captured_at', 'compass_angle'],
        ...     save_to_disk=False,
        ... )  # doctest: +SKIP
        >>> asset.properties['compass_angle']  # doctest: +SKIP
        182.4
    """

    AVAILABLE_IMAGE_FIELDS = frozenset(
        [
            'altitude',
            'atomic_scale',
            'camera_parameters',
            'camera_type',
            'captured_at',
            'compass_angle',
            'computed_altitude',
            'computed_compass_angle',
            'computed_geometry',
            'computed_rotation',
            'creator',
            'exif_orientation',
            'geometry',
            'height',
            'is_pano',
            'make',
            'model',
            'thumb_256_url',
            'thumb_1024_url',
            'thumb_2048_url',
            'thumb_original_url',
            'merge_cc',
            'mesh',
            'sequence',
            'sfm_cluster',
            'width',
            'detections.value',
            'detections.geometry',
        ]
    )

    def __init__(
        self,
        access_token: str | None = None,
        save_dir: str | Path = 'mapillary_images',
        *,
        api_key: str | None = None,
        save_directory: str | Path | None = None,
    ):
        """
        Initialize the client with an access token and an output directory.

        Args:
            access_token (str):
                A Mapillary API access token. Must be a non-empty string.
            save_dir (str | Path):
                Directory where downloaded images and generated masks are
                saved. It is created lazily on the first download. Defaults
                to ``'mapillary_images'``.
            api_key (str | None):
                Synonym of ``access_token`` for consistency with the other
                rapidtools clients.
            save_directory (str | Path | None):
                Synonym of ``save_dir``.

        Raises:
            ValueError:
                If no access token is given.

        Example:
            >>> client = MapillaryClient(api_key='MLY|123|abc', save_directory='panos')
            >>> client.save_dir.name
            'panos'
        """
        if api_key is not None:
            access_token = api_key
        if save_directory is not None:
            save_dir = save_directory
        if not access_token:
            raise ValueError('Mapillary access token is required.')
        self.access_token = access_token
        self.save_dir = Path(save_dir)

        # Create a session with the Retry strategy and the default headers:
        self.session = get_configured_session()

    def fetch_image(
        self,
        image_id: str,
        fields: list[str] | None = None,
        save_to_disk: bool = True,
        process_masks: list[Literal['semantic', 'instance']] | None = None,
    ) -> ImageAsset | None:
        """
        Fetch image metadata using its Mapillary ID and optionally download it.

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
                Optional list of metadata fields to request. If ``None``, all
                fields defined in ``AVAILABLE_IMAGE_FIELDS`` are requested.
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
                An ImageAsset whose ``path`` is the local path of the
                downloaded file, whose ``id`` is the API ``id`` if present
                (otherwise the requested ``image_id``), and whose
                ``properties`` hold the remaining metadata fields returned
                by the API (excluding the ``id``).
                Returns ``None`` if validation fails, metadata retrieval
                fails, no download URL is present, or the download itself
                fails.

        Example:
            >>> client = MapillaryClient('MLY|123|abc')
            >>> asset = client.fetch_image(
            ...     '1234567890',
            ...     fields=['width', 'height'],
            ...     save_to_disk=False,
            ...     process_masks=['semantic'],
            ... )  # doctest: +SKIP
            >>> asset.semantic_map  # doctest: +SKIP
            {0: 'void--unlabeled', 1: 'nature--sky', 2: 'construction--flat--road'}
        """
        # Determine intent:
        should_process_masks = process_masks is not None and len(process_masks) > 0

        # Validate and prepare requested fields:
        fields_to_request = self._validate_fields(
            fields, image_id, require_segmentation=should_process_masks
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
            logger.error(f'No download URL found for {image_id}')
            return None

        # Prepare an image path:
        prop_id = props.pop('id', image_id)
        file_path = self.save_dir / f'{prop_id}.jpg'

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
            allow_missing_file=allow_missing,
        )

        # Process segmentation:
        if should_process_masks:
            # Check if detections were returned from the API:
            detections_data = props.get('detections')

            if not detections_data or not detections_data.get('data'):
                logger.warning(
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
                            sem_mask, mask_type='semantic', map_data=sem_map
                        )
                        if save_to_disk:
                            image_asset.save_mask(mask_type='semantic')

                    # Instance mask:
                    if 'instance' in process_masks:
                        inst_mask, inst_map = self._parse_mapillary_segmentation(
                            image_asset, merge_detections=False
                        )
                        image_asset.set_mask(
                            inst_mask, mask_type='instance', map_data=inst_map
                        )
                        if save_to_disk:
                            image_asset.save_mask(mask_type='instance')

                except Exception as e:
                    logger.error(f'Failed to process segmentation for {prop_id}: {e}')

        return image_asset

    def fetch_images_by_ids(
        self,
        image_ids: list[str],
        fields: list[str] | None,
        save_to_disk: bool = True,
        process_masks: list[Literal['semantic', 'instance']] | None = None,
        max_workers: int = 10,
    ) -> ImageCollection:
        """
        Get multiple images by ID and return them as a collection.

        This method:
            - Submits one fetch task per image ID to a thread pool.
            - Fetches metadata and downloads each image (via ``fetch_image``).
            - Aggregates successfully fetched ImageAsset objects into a list.
            - Logs any exceptions that occur per image without aborting the
              entire batch.

        Args:
            image_ids (list[str]):
                A list of image IDs to fetch and download.
            fields (list[str] | None):
                Optional list of metadata fields to request for each image.
                If ``None``, all available fields are requested.
            save_to_disk (bool):
                If ``True``, downloads image files to ``self.save_dir``.
                If ``False``, only metadata is retrieved. Defaults to ``True``.
            process_masks (list[str] | None):
                A list of mask types to generate. Options are ``'semantic'``
                and ``'instance'``. Pass ``None`` or ``[]`` to skip
                segmentation. Example: ``['semantic']`` or
                ``['semantic', 'instance']``.
            max_workers (int):
                Maximum number of parallel worker threads used for fetching
                images. Defaults to ``10``.

        Returns:
            ImageCollection:
                An ``ImageCollection`` containing all successfully fetched
                ``ImageAsset`` instances. Any images that fail to download or
                process are excluded; errors are logged per image ID.

        Example:
            >>> client = MapillaryClient('MLY|123|abc')
            >>> collection = client.fetch_images_by_ids(
            ...     ['111', '222'], fields=['captured_at'], save_to_disk=False
            ... )  # doctest: +SKIP
            >>> collection.get_ids()  # doctest: +SKIP
            ['111', '222']
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
                    self.fetch_image, img_id, fields, save_to_disk, process_masks
                ): img_id
                for img_id in image_ids
            }

            # Iterate over futures as they complete (in the order they finish,
            # not necessarily the order submitted):
            for future in tqdm(
                as_completed(future_to_id),
                total=len(image_ids),
                desc='Downloading Images',
                unit='img',
            ):
                img_id = future_to_id[future]
                try:
                    asset = future.result()
                    # Only append successfully created ImageAsset objects:
                    if asset:
                        assets.append(asset)
                except Exception as e:
                    logger.error(f'Exception for image {img_id}: {e}')

        final_collection = ImageCollection()
        final_collection.add(assets)

        return final_collection

    def fetch_images_in_bbox(
        self,
        bbox: BoundingBox,
        fields: list[str] | None = None,
        save_to_disk: bool = False,
        process_masks: list[Literal['semantic', 'instance']] | None = None,
        start_date: str = '',
        end_date: str = '',
        filter_rapid_only: bool = True,
        max_workers: int = 10,
    ) -> ImageCollection:
        """
        Download and parse Mapillary vector tiles covering a bbox in parallel.

        The bounding box is first converted into the set of zoom-14 XYZ tiles
        that cover it. Each tile is downloaded and decoded to discover image
        locations. If richer metadata or file downloads are requested, the
        Graph API is then queried for every discovered image ID and the
        results are merged back into the tile-derived collection.

        Args:
            bbox (BoundingBox):
                The geographic rectangular area to search. Must be an
                instance of ``rapidtools.core.BoundingBox``.
            fields (list[str] | None):
                A list of specific metadata fields to fetch (e.g.,
                ``['captured_at', 'compass_angle']``). If ``None`` and
                ``save_to_disk`` is ``False``, only the lightweight tile
                metadata (location, date) is returned and the Graph API is
                not queried.
            save_to_disk (bool):
                If ``True``, downloads the images to the configured save
                directory. Defaults to ``False``.
            process_masks (list[str] | None):
                A list of mask types to generate. Options are ``'semantic'``
                and ``'instance'``. Pass ``None`` or ``[]`` to skip
                segmentation.
            start_date (str):
                Inclusive lower bound on capture date in ``YYYY-MM-DD``
                format. An empty string disables the lower bound.
            end_date (str):
                Inclusive upper bound on capture date in ``YYYY-MM-DD``
                format. An empty string disables the upper bound.
            filter_rapid_only (bool):
                If ``True``, only images uploaded by the RAPID facility
                account are returned. Defaults to ``True``.
            max_workers (int):
                Number of parallel download threads. Defaults to ``10``.

        Returns:
            ImageCollection:
                A new ``ImageCollection`` populated with assets from all
                tiles. Empty if the bounding box yields no tiles or no
                matching images.

        Example:
            >>> from rapidtools.core import BoundingBox
            >>>
            >>> client = MapillaryClient('MLY|123|abc')
            >>> bbox = BoundingBox(-118.15, 34.18, -118.14, 34.19)
            >>> images = client.fetch_images_in_bbox(
            ...     bbox, start_date='2025-01-01', filter_rapid_only=False
            ... )  # doctest: +SKIP
            >>> images[0].properties['capture_date']  # doctest: +SKIP
            '2025-02-14'
        """
        # Get a list of tiles that cover the bounding box area:
        tiles_list = TileUtils.bbox_to_mapbox_tiles(bbox, zoom=14)

        total_tiles = len(tiles_list)
        if total_tiles == 0:
            logger.warning('No tiles found. Bounding box is too small or invalid')
            return ImageCollection()

        logger.info(f'Scanning {total_tiles} tiles for images...')

        # Download/process tiles in parallel:
        final_collection = ImageCollection()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tile = {
                executor.submit(
                    self._get_tile_image_data,
                    tile,
                    start_date,
                    end_date,
                    filter_rapid_only,
                ): tile
                for tile in tiles_list
            }

            for future in tqdm(
                as_completed(future_to_tile),
                total=total_tiles,
                desc='Downloading Tiles',
                unit='tile',
            ):
                tile = future_to_tile[future]
                try:
                    assets = future.result()
                    # Add found assets to the main collection
                    if assets:
                        final_collection.add(assets)
                except Exception as e:
                    logger.error(f'Critical error in tile thread {tile}: {e}')

        # Exit early if no images are returned:
        unique_count = len(final_collection)
        logger.info(f'Scan complete. Found {unique_count} unique images.')

        if unique_count == 0:
            return final_collection

        # If we need specific metadata OR we need to save files to disk,
        # we must hit the API for the specific IDs found:
        if fields or save_to_disk:
            ids = final_collection.get_ids()

            logger.info('Starting to download the data for detected images...')

            # Retrieve rich data (and optionally download files):
            rich_collection = self.fetch_images_by_ids(
                image_ids=ids,
                fields=fields,
                save_to_disk=save_to_disk,
                process_masks=process_masks,
            )
            logger.info('Downloaded data for all detected images.')

            # Merge the rich data back into our tile-based collection:
            final_collection.merge(
                rich_collection,
                overwrite_properties=True,
                overwrite_path=save_to_disk,
                add_new=False,
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
                ``True`` if the download completed successfully and the file
                was written to disk; ``False`` if an error occurred.

        Example:
            >>> from pathlib import Path
            >>>
            >>> client = MapillaryClient('MLY|123|abc')
            >>> client._download_image(
            ...     'https://example.com/pano.jpg', Path('pano.jpg')
            ... )  # doctest: +SKIP
            True
        """
        try:
            with self.session.get(url, stream=True, timeout=REQUESTS_TIMEOUT_VAL) as r:
                r.raise_for_status()
                with open(destination, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f'Download failed for {url}: {e}')
            return False

    def _get_tile_image_data(
        self,
        tile_coords: tuple[int, int, int],
        start_date: str = '',
        end_date: str = '',
        filter_rapid_only: bool = True,
    ) -> list[ImageAsset]:
        """
        Extract image assets from a single Mapillary vector tile.

        This helper retrieves a single tile from the Mapillary tiles API,
        decodes its vector data, filters images by source and optional
        capture-date range, converts tile coordinates to WGS84, and
        returns a list of corresponding ``ImageAsset`` objects.

        The method:
            * Downloads and (if needed) decompresses the Mapillary tile.
            * Decodes the Mapbox Vector Tile into layers/features.
            * Reads image features from the ``'image'`` layer.
            * Optionally filters out images not created by the RAPID creator.
            * Optionally filters images outside the requested capture date
              range.
            * Converts tile-based coordinates to latitude/longitude (WGS84).
            * Cleans up properties and builds ``ImageAsset`` instances.

        Args:
            tile_coords (tuple[int, int, int]):
                The tile coordinates as ``(x, y, z)`` in the Web Mercator/XYZ
                tiling scheme, where ``x`` is the tile column index, ``y`` the
                tile row index and ``z`` the zoom level.
            start_date (str):
                Optional inclusive lower bound for the image capture date
                filter. Expected in ``YYYY-MM-DD`` format or empty string for
                no lower bound.
            end_date (str):
                Optional inclusive upper bound for the image capture date
                filter. Expected in ``YYYY-MM-DD`` format or empty string for
                no upper bound.
            filter_rapid_only (bool):
                If ``True``, only images whose ``creator_id`` matches
                ``RAPID_CREATOR_ID`` are included. If ``False``, all creators
                are allowed.

        Returns:
            list[ImageAsset]:
                A list of ``ImageAsset`` instances found within the specified
                tile that satisfy all filtering conditions. Returns an empty
                list if the tile has no ``'image'`` layer, no images satisfy
                the filters, or the tile download/decoding fails (a warning is
                logged).

        Notes:
            * The method silently skips images with invalid or missing
              dates when a date filter is active.
            * Gzip-compressed tiles are automatically decompressed based
              on the Gzip magic header.
            * The resulting ``ImageAsset.path`` uses the image ID as the
              filename with a ``.jpg`` extension, stored under
              ``self.save_dir``.

        Example:
            >>> client = MapillaryClient('MLY|123|abc')
            >>> assets = client._get_tile_image_data(
            ...     (2815, 6535, 14), filter_rapid_only=False
            ... )  # doctest: +SKIP
            >>> assets[0].properties['latitude']  # doctest: +SKIP
            34.1873
        """
        x, y, z = tile_coords
        found_assets = []

        # Construct tile URL:
        tile_url = TILE_URL_TEMPLATE.format(z=z, x=x, y=y, token=self.access_token)

        try:
            # Download the tile data:
            response = requests.get(tile_url, timeout=30)
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
                return []  # No images in this tile

            # Get the tile extent defined in the data (defaults to 4096 if
            # missing):
            tile_extent = image_layer.get('extent', 4096)
            images = image_layer.get('features', [])

            for image in images:
                props = image['properties']

                # Extract creator and image IDs. Tile IDs arrive as integers
                # while the Graph API returns strings; normalize to ``str`` so
                # collections built from both sources merge by ID:
                creator_id = props.pop('creator_id', None)
                prop_id = props.pop('id', None)
                if prop_id is not None:
                    prop_id = str(prop_id)

                # Depending on filter_rapid_only check if creator ID matches
                # RAPID's ID:
                if filter_rapid_only and creator_id != RAPID_CREATOR_ID:
                    continue

                # Convert geometry data from tile coordinates to WGS84:
                shape_xy = image['geometry']['coordinates']
                lon, lat = TileUtils.mvt_to_wgs84(
                    shape_xy[0],
                    tile_extent - shape_xy[1],
                    x,
                    y,
                    z,
                    extent=tile_extent,
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
                    if not self._is_date_in_range(img_date, start_date, end_date):
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
                    path=self.save_dir / f'{prop_id}.jpg',
                    id=prop_id,
                    properties=props,
                    allow_missing_file=True,
                )
                found_assets.append(asset)

        # If tile data cannot be downloaded log error:
        except Exception as e:
            logger.warning(f'Failed to process tile {z}/{x}/{y}: {e}')

        return found_assets

    @staticmethod
    def _is_date_in_range(
        check_date: date | datetime | str | None,
        start_date: date | datetime | str | None,
        end_date: date | datetime | str | None,
    ) -> bool:
        """
        Check whether a date falls within an inclusive date range.

        This method is robust to different input types and missing values. It
        accepts ``date``, ``datetime``, string (in ``YYYY-MM-DD`` format), or
        ``None`` for all parameters, and normalizes them to ``datetime.date``
        objects before comparison.

        The check is inclusive of both ``start_date`` and ``end_date``. If
        ``check_date`` is missing or invalid and at least one of
        ``start_date`` or ``end_date`` is provided (i.e., filters exist),
        the method returns ``False``. If all three inputs are missing or
        invalid (i.e., no filters), the method returns ``True``.

        Args:
            check_date (date | datetime | str | None):
                The date to be tested.
            start_date (date | datetime | str | None):
                The inclusive lower bound of the date range. If ``None`` or
                empty, there is no lower bound.
            end_date (date | datetime | str | None):
                The inclusive upper bound of the date range. If ``None`` or
                empty, there is no upper bound.

        Returns:
            bool:
                ``True`` if ``check_date`` is within the inclusive range
                defined by ``start_date`` and ``end_date`` (after
                normalization), or if all three values are missing/invalid
                (no filters). ``False`` if ``check_date`` is outside the
                range, invalid, or missing while at least one of
                ``start_date`` or ``end_date`` is provided.

        Notes:
            * String inputs must be in ``YYYY-MM-DD`` format; otherwise
              they are treated as invalid.
            * ``datetime.datetime`` inputs are converted to dates by
              discarding the time component.
            * Invalid or unparsable date strings are treated as ``None``.

        Example:
            >>> MapillaryClient._is_date_in_range(
            ...     '2025-02-14', '2025-01-01', '2025-12-31'
            ... )
            True
            >>> MapillaryClient._is_date_in_range('2024-06-01', '2025-01-01', '')
            False
            >>> MapillaryClient._is_date_in_range(None, '', '')
            True
        """

        def to_date(date_input):
            """Normalize various date-like inputs to a ``datetime.date``."""
            if not date_input:
                return None
            if isinstance(date_input, datetime):
                return date_input.date()
            if isinstance(date_input, date):
                return date_input
            if isinstance(date_input, str):
                try:
                    return datetime.strptime(date_input.strip(), '%Y-%m-%d').date()
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
            return not (start or end)

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
        retries: int = 5,
        backoff_factor: float = 0.5,
    ) -> dict[str, Any] | None:
        """
        Retrieve metadata for a specific image from the Graph API.

        This method issues a GET request to the configured API endpoint using
        the provided image ID and list of field names. Rate-limited (429),
        forbidden (403), non-JSON and network-error responses are retried
        with exponential backoff. On success, the JSON response is returned
        as a dictionary. If all attempts fail, a warning is logged and
        ``None`` is returned.

        Args:
            image_id (str):
                The unique identifier of the image whose metadata should be
                fetched.
            fields (list[str]):
                Field names to request from the API. These are sent as a
                comma-separated list via the ``fields`` query parameter.
            retries (int):
                Maximum number of attempts before giving up. Defaults to
                ``5``.
            backoff_factor (float):
                Base wait time in seconds; attempt ``n`` waits
                ``backoff_factor * 2**n`` seconds. Defaults to ``0.5``.

        Returns:
            dict[str, Any] | None:
                A dictionary containing the JSON metadata returned by the API
                if the request succeeds; otherwise ``None``.

        Example:
            >>> client = MapillaryClient('MLY|123|abc')
            >>> meta = client._get_image_metadata(
            ...     '1234567890', ['width', 'height', 'thumb_original_url']
            ... )  # doctest: +SKIP
            >>> meta['width']  # doctest: +SKIP
            8192
        """
        url = f'{BASE_URL}/{image_id}'
        params = {
            'access_token': self.access_token,
            'fields': ','.join(fields),
        }

        for attempt in range(retries):
            try:
                # Bypass the custom session to avoid hidden headers
                resp = requests.get(url, params=params, timeout=15)

                # If we get rate-limited (429) or forbidden (403), force a retry
                if resp.status_code in [429, 403]:
                    raise requests.exceptions.RequestException(
                        f'Rate limited or forbidden (HTTP {resp.status_code})'
                    )

                resp.raise_for_status()

                # Check to make sure it is actually JSON before decoding
                if 'application/json' not in resp.headers.get('Content-Type', ''):
                    raise requests.exceptions.RequestException(
                        'Response is not JSON format. Likely an API error page.'
                    )

                return resp.json()

            except requests.exceptions.RequestException as e:
                # Calculate exponential wait time: 1s, 2s, 4s, 8s...
                wait_time = backoff_factor * (2**attempt)
                logger.debug(
                    f'API rejected request for {image_id} ({e}). '
                    f'Retrying in {wait_time}s...'
                )

                if attempt + 1 == retries:
                    break

                time.sleep(wait_time)

            except json.JSONDecodeError:
                # If the JSON parsing still fails, it's a corrupted response
                break

        logger.warning(
            f'All {retries} attempts to fetch metadata for {image_id} failed.'
        )
        return None

    @staticmethod
    def _parse_mapillary_segmentation(
        pano: ImageAsset,
        target_classes: list[str] | None = None,
        flip_y: bool = True,
        merge_detections: bool = True,
        show_progress: bool = False,
    ) -> tuple[np.ndarray, dict[int, str]]:
        """
        Create a segmentation mask for an image using Mapillary detections.

        Each Mapillary detection carries a base64-encoded Mapbox Vector Tile
        describing one or more polygons in tile coordinates. This method
        decodes those polygons, scales them to the image dimensions and
        rasterizes them into an integer label mask.

        Args:
            pano (ImageAsset):
                ImageAsset object whose ``properties`` contain ``width``,
                ``height`` and a ``detections`` dictionary with a ``data``
                list of ``{'value': str, 'geometry': str}`` entries.
            target_classes (list[str] | None):
                Optional list of class prefixes to include. Detections whose
                label does not start with any prefix are skipped.
            flip_y (bool):
                Whether to flip the Y-axis of the decoded geometry. Defaults
                to ``True`` (Mapbox tiles decode with the origin at the
                bottom-left).
            merge_detections (bool):
                If ``True`` (semantic), all detections of the same label share
                one ID. If ``False`` (instance), every detection gets a unique
                ID.
            show_progress (bool):
                If ``True``, displays a tqdm progress bar. Defaults to
                ``False`` to reduce clutter.

        Returns:
            tuple[np.ndarray, dict[int, str]]:
                ``(mask_array, segmentation_map)`` where ``mask_array`` is the
                rasterized mask (``uint8`` if the maximum ID is <= 255,
                otherwise ``int32``) and ``segmentation_map`` maps pixel values
                to label names. Pixel ``0`` is always ``'void--unlabeled'``.
                If the image dimensions are missing, a ``(1, 1)`` zero mask
                and an empty map are returned.

        Example:
            >>> from rapidtools.core import ImageAsset
            >>>
            >>> asset = ImageAsset(
            ...     path='pano.jpg',
            ...     id='pano',
            ...     properties={'width': 64, 'height': 32, 'detections': {
            ...         'data': [{'value': 'nature--sky', 'geometry': '...'}]
            ...     }},
            ...     allow_missing_file=True,
            ... )
            >>> mask, seg_map = MapillaryClient._parse_mapillary_segmentation(
            ...     asset
            ... )  # doctest: +SKIP
            >>> seg_map  # doctest: +SKIP
            {0: 'void--unlabeled', 1: 'nature--sky'}
        """
        # Extract image properties to avoid repeated lookups:
        props = pano.properties
        img_width = props.get('width')
        img_height = props.get('height')

        if not img_width or not img_height:
            logger.warning('ImageAsset missing dimension information.')
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
            disable=not show_progress,
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
            if not b64_string:
                continue

            try:
                # Decode Protobuf:
                pbf_bytes = base64.decodebytes(b64_string.encode('utf-8'))
                decoded_tile = mapbox_vector_tile.decode(pbf_bytes)

                if not decoded_tile:
                    continue

                # Get the first layer (standard MVT structure):
                layer_name = next(iter(decoded_tile.keys()))
                layer_data = decoded_tile[layer_name]
                extent = layer_data['extent']

                # Coordinate transformation. Calculate scalars once per tile:
                scale_x = img_width / extent
                scale_y = img_height / extent

                # Drawing:
                for feature in layer_data['features']:
                    geom = feature['geometry']
                    g_type = geom['type']
                    coords = geom['coordinates']

                    if g_type == 'Polygon':
                        rings = coords
                    elif g_type == 'MultiPolygon':
                        rings = [ring for poly in coords for ring in poly]
                    else:
                        continue

                    for ring in rings:
                        poly_pts = MapillaryClient._scale_ring(
                            ring, scale_x, scale_y, img_height, flip_y
                        )
                        if len(poly_pts) > 2:
                            draw.polygon(poly_pts, fill=current_id)

            except Exception as e:
                logger.error(f'Error processing mask geometry: {e}')
                continue

        mask_array = np.array(canvas, dtype=np.uint8 if image_mode == 'L' else np.int32)
        return mask_array, segmentation_map

    @staticmethod
    def _scale_ring(
        ring_coords: list,
        scale_x: float,
        scale_y: float,
        img_height: int,
        flip_y: bool,
    ) -> list[tuple[float, float]]:
        """
        Scale a ring of tile coordinates into image pixel coordinates.

        Args:
            ring_coords (list):
                Sequence of ``(x, y)`` pairs in vector-tile coordinates.
            scale_x (float):
                Multiplier converting tile X units to pixels.
            scale_y (float):
                Multiplier converting tile Y units to pixels.
            img_height (int):
                Image height in pixels, used when flipping the Y-axis.
            flip_y (bool):
                If ``True``, the Y-axis is inverted so that tile origin
                (bottom-left) maps to image origin (top-left).

        Returns:
            list[tuple[float, float]]:
                The transformed ring as a list of pixel coordinates.

        Example:
            >>> MapillaryClient._scale_ring([(0, 0), (4096, 4096)], 0.5, 0.25,
            ...                             1024, flip_y=False)
            [(0.0, 0.0), (2048.0, 1024.0)]
            >>> MapillaryClient._scale_ring([(0, 0)], 1.0, 1.0, 100, flip_y=True)
            [(0.0, 100.0)]
        """
        if flip_y:
            return [(x * scale_x, img_height - (y * scale_y)) for x, y in ring_coords]
        return [(x * scale_x, y * scale_y) for x, y in ring_coords]

    def _validate_fields(
        self,
        fields: list[str] | None,
        image_id: str = 'batch',
        require_segmentation: bool = False,
    ) -> list[str] | None:
        """
        Validate and normalize requested image fields for API requests.

        This helper:

        - Filters out any requested fields that are not in
          ``AVAILABLE_IMAGE_FIELDS``.
        - Logs a warning for any invalid/unknown fields that are omitted.
        - Ensures that at least one valid field remains; otherwise, logs an
          error and returns ``None``.
        - Ensures that ``thumb_original_url`` is always included in the
          returned list of fields.
        - Automatically injects ``detections.value`` and
          ``detections.geometry`` if segmentation processing is required.

        Args:
            fields (list[str] | None):
                List of requested image field names. If ``None``, all fields
                defined in ``AVAILABLE_IMAGE_FIELDS`` are returned.
            image_id (str):
                Identifier of the image or batch used only for logging context
                when reporting invalid or missing fields. Defaults to
                ``'batch'``.
            require_segmentation (bool):
                If ``True``, explicitly adds ``detections.value`` and
                ``detections.geometry`` to the requested fields list to ensure
                segmentation data is retrieved. Defaults to ``False``.

        Returns:
            list[str] | None:
                A list of validated field names that will be used in API
                requests. The list is guaranteed to include
                ``thumb_original_url``. Returns ``None`` if no valid fields
                are found after validation.

        Example:
            >>> client = MapillaryClient('MLY|123|abc')
            >>> sorted(client._validate_fields(['width', 'bogus']))
            ['thumb_original_url', 'width']
            >>> client._validate_fields(['bogus']) is None
            True
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
        valid_fields = current_fields.intersection(self.AVAILABLE_IMAGE_FIELDS)

        # Check for invalid fields:
        if fields is not None:
            invalid_fields = current_fields.difference(self.AVAILABLE_IMAGE_FIELDS)

            if invalid_fields:
                logger.warning(
                    f'Invalid field(s) omitted: {", ".join(sorted(invalid_fields))}'
                )

        # Fail if nothing remains:
        if not valid_fields:
            logger.error(f'No valid fields provided for {image_id}. Aborting.')
            return None

        final_fields = list(valid_fields)

        # We always need the download URL:
        if 'thumb_original_url' not in final_fields:
            final_fields.append('thumb_original_url')

        return final_fields
