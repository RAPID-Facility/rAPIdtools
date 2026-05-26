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
# 05-25-2026

import concurrent.futures
import logging
import math
from io import BytesIO
from pathlib import Path

import numpy as np
import rasterio
import requests
from PIL import Image, ImageDraw
from rasterio.transform import from_bounds
from typing import Any
from tqdm import tqdm

from rapidtools.config import REQUESTS_TIMEOUT_VAL, get_configured_session
from rapidtools.core import BoundingBox, PolygonRegion, ImageAsset, PhysicalAssetCollection
from rapidtools.data_sources import OrthomosaicReader, MapillaryClient
from .pano_utils import (
    build_footprint_index,
    index_panos,
    find_best_panos,
    crop_panorama_to_asset
)

class AerialImageryExtractor:
    """
    Component that extracts aerial imagery patches for a collection of assets.
    
    This extractor reads from one or more high-resolution orthomosaic TIFF files, 
    crops the imagery around each asset's bounding box, saves the patches to 
    disk, and automatically attaches the new digital media to the corresponding 
    `PhysicalAsset` objects in the collection.
    
    Args:
        dataset (str | Path | list[str | Path]): 
            The file path to a single local orthomosaic TIFF dataset, a 
            directory containing TIFFs, or a list of TIFF file paths.
        save_directory (str | Path): 
            The directory where the extracted JPEG images will be saved. 
            Will be created if it does not exist.
        max_missing_data_ratio (float, optional): 
            The maximum acceptable proportion of nodata pixels (0.0 to 1.0). 
            If a cropped patch exceeds this, it is discarded. Defaults to 0.2.
        overlay_asset_outline (bool, optional): 
            If ``True``, draws a thick red polygon outline representing the 
            asset's geometry directly onto the saved image. Defaults to ``False``.
        image_prefix (str, optional):
            The prefix to use for the saved image filenames and asset IDs. 
            Defaults to ``'aerial'``.
        keep_multiple_copies (bool, optional):
            If ``True``, prevents overwriting existing images with the same 
            coordinates by appending a numeric counter to the filename and ID.
            Defaults to ``False``.
        buffer_asset (float | str, optional):
            The padding applied around the asset's geometry before cropping. 
            Can be a percentage (e.g., ``'20%'``), an absolute CRS distance 
            (e.g., 50.0), or a real-world distance (e.g., ``'50 ft'``, 
            ``'10 m'``). Defaults to ``'20%'``.
        force_square_image (bool, optional):
            If ``True``, guarantees the extracted image patch covers a 
            perfectly square area in the real world. Defaults to ``True``.

    Example:
        >>> from rapidtools.processing import AerialImageryExtractor
        >>>
        >>> extractor = AerialImageryExtractor(
        ...     dataset='data/rasters/',
        ...     save_directory='output/aerial_crops',
        ...     overlay_asset_outline=True,
        ...     image_prefix='post_disaster',
        ...     keep_multiple_copies=True,
        ...     buffer_asset='20%',
        ...     force_square_image=True
        ... )
        >>> processed_assets = extractor(my_asset_collection)
    """

    def __init__(
        self,
        dataset: str | Path | list[str | Path],
        save_directory: str | Path,
        max_missing_data_ratio: float = 0.2,
        overlay_asset_outline: bool = False,
        image_prefix: str = 'aerial',
        keep_multiple_copies: bool = False,
        buffer_asset: float | str = '20%',     
        force_square_image: bool = True        
    ):
        """Initialize the extractor configuration."""
        self.dataset = dataset
        self.save_directory = Path(save_directory).resolve()
        self.max_missing_data_ratio = max_missing_data_ratio
        self.overlay_asset_outline = overlay_asset_outline
        self.image_prefix = image_prefix
        self.keep_multiple_copies = keep_multiple_copies
        self.buffer_asset = buffer_asset
        self.force_square_image = force_square_image

    def _get_raster_paths(self) -> list[Path]:
        """Helper method to resolve all raster files from the dataset input."""
        paths = []
        
        # Handle the case where the user provides an explicit list of paths:
        if isinstance(self.dataset, list):
            for p in self.dataset:
                paths.append(Path(p).resolve())
        else:
            # Handle the case where the user provides a single string or Path
            # object:
            p = Path(self.dataset).resolve()
            
            if p.is_dir():
                # Recursively search the directory and all subdirectories for 
                # TIFFs. Check for both extensions to catch standard and
                # alternative naming:
                paths.extend(p.rglob('*.tif'))
                paths.extend(p.rglob('*.tiff'))
            else:
                # The input is a direct path to a single file:
                paths.append(p)
                
        # Cast to a set to efficiently remove any accidental duplicates:
        return list(set(paths))

    def __call__(
        self, 
        asset_collection: PhysicalAssetCollection
    ) -> PhysicalAssetCollection:
        """
        Execute the extraction process on the provided asset collection.
        
        This method allows the class instance to be called like a function, 
        making it directly compatible with the ``rapidtools`` ``Pipeline`` 
        engine.
        
        Args:
            asset_collection (PhysicalAssetCollection): 
                The collection of physical assets to extract imagery for.

        Returns:
            PhysicalAssetCollection: 
                The mutated collection, with new ``ImageAsset`` objects attached 
                to their corresponding ``PhysicalAsset`` entities.
        """
        # Create save directory if it does not exist:
        self.save_directory.mkdir(parents=True, exist_ok=True)
        
        # Gather all raster file paths from the user's input: 
        raster_paths = self._get_raster_paths()
        if not raster_paths:
            logging.warning(
                f'No raster files found for dataset input: {self.dataset}'
            )
            return asset_collection
            
        logging.info(
            f'Found {len(raster_paths)} raster(s). Images will be saved to: '
            f'{self.save_directory}'
        )
        
        total_extracted_count = 0

        # Process each raster file found:
        for raster_path in raster_paths:
            logging.info(f'Processing raster: {raster_path.name}')
            
            # Open the TIFF file ONCE for the current raster using the context 
            # manager:
            try:
                with OrthomosaicReader(raster_path) as reader:
                    
                    # Efficiently filter assets to only those inside the 
                    # raster's extent:
                    min_lon, min_lat, max_lon, max_lat = reader.dataset_extent
                    raster_bbox = BoundingBox(
                        min_x=min_lon,
                        min_y=min_lat,
                        max_x=max_lon,
                        max_y=max_lat
                    )
                    
                    assets_in_bounds = asset_collection.filter_by_geometry(
                        raster_bbox
                    )
                    
                    if not assets_in_bounds:
                        logging.info(
                            'No assets fall within the bounds of '
                            f'{raster_path.name}. Skipping.'
                        )
                        continue
                        
                    logging.info(
                        f'Found {len(assets_in_bounds)}/{len(asset_collection)}'
                        f" assets within the current raster's extent."
                    )
                    
                    # Loop only over the filtered subset for this specific 
                    # raster:
                    for asset in tqdm(
                            assets_in_bounds, 
                            desc=f'Extracting from {raster_path.name}'
                        ):
                        
                        # Get the asset geometry:
                        geom = asset.geometry
    
                        if not geom or geom.is_empty:
                            continue
                        
                        # Take the outside part for multi-part geometries 
                        # (convex_hull creates a single polygon wrapping all
                        # parts of the asset):
                        if geom.geom_type.startswith('Multi') or \
                            geom.geom_type == 'GeometryCollection':
                            unified_geom = geom.convex_hull
                        else:
                            unified_geom = geom
                            
                        # Safely extract a single list of coordinates for
                        # OrthomosaicReader:
                        if unified_geom.geom_type == 'Polygon':
                            asset_coords = list(unified_geom.exterior.coords)
                            is_closed = True
                        elif unified_geom.geom_type in ('LineString', 'Point'):
                            asset_coords = list(unified_geom.coords)
                            is_closed = False
                        else:
                            # Fallback to use the rectangular bounding box:
                            asset_coords = list(
                                unified_geom.envelope.exterior.coords
                            )
                            is_closed = True
                        
                        # Extract image patch using the open reader:
                        extraction_result = reader.get_image_patch(
                            asset_geometry=asset_coords, 
                            max_missing_data_ratio=self.max_missing_data_ratio,
                            buffer=self.buffer_asset,           
                            force_square=self.force_square_image
                        )
                        
                        if extraction_result is None:
                            continue
                            
                        pil_image, pixel_coords, wgs84_bounds = extraction_result

                        # Draw outline if requested:
                        if self.overlay_asset_outline:
                            draw = ImageDraw.Draw(pil_image)
                            # Point:
                            if len(pixel_coords) == 1:
                            
                                x, y = pixel_coords[0]
                                r = 5
                                draw.ellipse((x-r, y-r, x+r, y+r), fill='red')
                            # Line or polygon:
                            else:
                                draw_coords = pixel_coords + [pixel_coords[0]]\
                                    if is_closed else pixel_coords
                                draw.line(draw_coords, fill='red', width=6)

                        # Create a clean, coordinate-based filename:
                        centroid = unified_geom.centroid
                        coords_str = f'{centroid.y:.8f}_{centroid.x:.8f}'.replace('.', '')
                        
                        base_name = f'{self.image_prefix}_{coords_str}'
                        image_name = f'{base_name}.jpg'
                        image_path = self.save_directory / image_name
                        asset_id_suffix = self.image_prefix

                        # Handle potential overlaps if requested:
                        if self.keep_multiple_copies:
                            counter = 1
                            while image_path.exists():
                                image_name = f'{base_name}_{counter}.jpg'
                                image_path = self.save_directory / image_name
                                asset_id_suffix = f'{self.image_prefix}_{counter}'
                                counter += 1

                        # Save the image to disk:
                        pil_image.save(image_path)
                        
                        # Create the ImageAsset domain object and attach it:
                        img_asset = ImageAsset(
                            id=f'{asset.id}_{asset_id_suffix}',  
                            path=image_path,
                            allow_missing_file=False,
                            properties={
                                'wgs84_bounds': wgs84_bounds,
                                'source_raster': raster_path.name
                            }
                        )
                        
                        asset.add_image_assets(img_asset)
                        total_extracted_count += 1
                        
            except Exception as e:
                logging.error(
                    f'Failed to process raster {raster_path.name}: {e}'
                )

        logging.info(
            f'Finished processing all rasters. Extracted aerial imagery for a '
            f'total of {total_extracted_count} assets.'
        )
        
        return asset_collection
    

class MapillaryImageExtractor:
    """
    Advanced Mapillary street-view extractor that extracts the best unobstructed 
    panoramas of an asset from multiple viewing angles.
    
    This extractor acts as an intelligent pipeline component. Instead of making 
    thousands of API calls, it fetches regional metadata once, builds spatial 
    indexes (KDTree and STRtree), and uses ray-casting to simulate lines of sight. 
    It actively culls occluded views (e.g., if a neighboring building blocks the 
    camera) and downloads only the valid panoramas.
    
    If `strict_content_filter` is enabled, the extractor checks the semantic 
    contents of the image before downloading the heavy JPEG. If the target asset 
    type is not actually visible in the image, it skips it to save bandwidth.

    Example:
        >>> from rapidtools.processing import MapillaryPanoramaExtractor
        >>>
        >>> # Option 1: Using manual dictionary mapping
        >>> extractor = MapillaryPanoramaExtractor(
        ...     access_token='YOUR_TOKEN',
        ...     save_directory='output/streetview',
        ...     strict_content_filter=True,
        ...     asset_type_mapping={
        ...         'building': ['construction--structure--building'],
        ...         'pole': ['object--support--utility-pole']
        ...     }
        ... )
        >>>
        >>> # Option 2: Using the Gemma LLM Mapper for dynamic mapping
        >>> from rapidtools.models import Gemma4Inference
        >>> from rapidtools.processing.label_mappers import MapillaryLabelMapper
        >>> mapper = MapillaryLabelMapper(Gemma4Inference())
        >>> extractor_llm = MapillaryPanoramaExtractor(
        ...     access_token='YOUR_TOKEN',
        ...     save_directory='output/streetview',
        ...     strict_content_filter=True,
        ...     label_mapper=mapper
        ... )
        >>> 
        >>> processed_assets = extractor(my_asset_collection)
    """

    def __init__(
        self,
        access_token: str,
        save_directory: str | Path,
        start_date: str = '',
        end_date: str = '',
        filter_rapid_only: bool = True,
        cast_corner_rays: bool = True,
        smart_crop: bool = True,
        strict_content_filter: bool = False,
        label_mapper: Any = None,
        asset_type_mapping: dict[str, list[str]] | None = None,
        image_prefix: str = 'street',
        max_workers: int = 10,
    ) -> None:
        """
        Initialize the Mapillary Panorama Extractor.

        Args:
            access_token (str): 
                Mapillary API Access Token.
            save_directory (str | Path): 
                The directory where the final cropped JPEG images will be saved.
            start_date (str, optional): 
                Inclusive lower bound for the image capture date (YYYY-MM-DD). 
                Defaults to '' (no lower bound).
            end_date (str, optional): 
                Inclusive upper bound for the image capture date (YYYY-MM-DD). 
                Defaults to '' (no upper bound).
            filter_rapid_only (bool, optional): 
                If True, strictly fetches images uploaded by the RAPID organization. 
                Set to False to search all public street-view images. Defaults to True.
            cast_corner_rays (bool, optional): 
                If False, limits extraction to the 4 principal axes (faces) of the 
                building. If True, also extracts views pointing directly at the 4 
                corners (up to 8 images). Defaults to False.
            smart_crop (bool, optional): 
                If True, downloads the semantic mask for the panorama and intelligently 
                crops out the sky and the data-collection vehicle. Defaults to True.
            strict_content_filter (bool, optional):
                If True, verifies that the target asset type is semantically present in 
                the image before downloading the high-resolution JPEG. Defaults to False.
            label_mapper (Any, optional):
                An instance of `MapillaryLabelMapper` that uses an LLM to dynamically 
                map asset types to Mapillary labels. Used if `strict_content_filter` 
                is True.
            asset_type_mapping (dict[str, list[str]] | None, optional):
                A manual dictionary mapping your asset types to official Mapillary 
                labels (e.g., {'building': ['construction--structure--building']}). 
                Used as a fallback or alternative to the `label_mapper`. 
                Defaults to None.
            image_prefix (str, optional): 
                Prefix applied to the saved image filenames. Defaults to 'street'.
            max_workers (int, optional): 
                The number of concurrent threads used to download and crop images. 
                Defaults to 10.
        """
        self.save_directory = Path(save_directory).resolve()
        self.start_date = start_date
        self.end_date = end_date
        self.filter_rapid_only = filter_rapid_only
        self.cast_corner_rays = cast_corner_rays
        self.smart_crop = smart_crop

        self.strict_content_filter = strict_content_filter
        self.label_mapper = label_mapper
        self.asset_type_mapping = asset_type_mapping or {}

        self.image_prefix = image_prefix
        self.max_workers = max_workers

        self.client = MapillaryClient(
            access_token=access_token,
            save_dir=self.save_directory,
        )

    def __call__(
        self, asset_collection: PhysicalAssetCollection
    ) -> PhysicalAssetCollection:
        """
        Execute the extraction process on the provided asset collection.

        Args:
            asset_collection (PhysicalAssetCollection): 
                The collection of physical assets to extract street-view imagery for.

        Returns:
            PhysicalAssetCollection: 
                The mutated collection, with new `ImageAsset` objects representing the 
                cropped panoramas attached to their corresponding `PhysicalAsset` 
                entities.
        """
        self.save_directory.mkdir(parents=True, exist_ok=True)

        # 1. Resolve Target Labels for Strict Content Filtering
        if self.strict_content_filter:
            # Find all unique asset types in the collection
            unique_types = {a.asset_type for a in asset_collection if a.asset_type}
            unmapped_types = [
                t for t in unique_types if t not in self.asset_type_mapping
            ]

            # If we have a label mapper (LLM), dynamically map any unknown types
            if unmapped_types and self.label_mapper:
                logging.info(
                    f'Using LLM to dynamically map asset types: {unmapped_types}'
                )
                for asset_type in unmapped_types:
                    mapped_labels = self.label_mapper.map_classes([asset_type])
                    self.asset_type_mapping[asset_type] = mapped_labels

            logging.info(f'Content filter mapping: {self.asset_type_mapping}')

        # 2. Fetch regional metadata
        collection_bbox = asset_collection.combined_bounding_box
        logging.info('Fetching regional Mapillary metadata...')

        regional_images = self.client.fetch_images_in_bbox(
            bbox=collection_bbox,
            save_to_disk=False,
            start_date=self.start_date,
            end_date=self.end_date,
            filter_rapid_only=self.filter_rapid_only,
        )

        if len(regional_images) == 0:
            logging.warning('No Mapillary images found in the collection region.')
            return asset_collection

        logging.info('Building KDTree for panos and STRtree for footprints...')
        tree, coords_deg, headings = index_panos(regional_images)
        building_tree, building_geoms = build_footprint_index(asset_collection)

        # Threaded processing function for a single asset
        def process_asset(asset) -> int:
            """Finds valid panoramas, verifies semantics, downloads, and crops."""
            if not asset.geometry:
                return 0

            # Determine valid labels for this specific asset
            valid_labels = []
            if self.strict_content_filter and asset.asset_type:
                valid_labels = self.asset_type_mapping.get(asset.asset_type, [])

            extracted_count = 0
            best_panos_map = find_best_panos(
                target_asset_wgs84=asset,
                pano_collection=regional_images,
                building_tree=building_tree,
                building_geoms=building_geoms,
                tree=tree,
                coords_deg=coords_deg,
                print_results=False,
                cast_corner_rays=self.cast_corner_rays,
                interval_deg=90.0 if not self.cast_corner_rays else 45.0,
            )

            for axis_name, pano_compact in best_panos_map.items():
                if pano_compact is None:
                    continue

                try:
                    needs_semantics = self.smart_crop or self.strict_content_filter

                    pano = self.client.fetch_image(
                        pano_compact.id,
                        save_to_disk=False,
                        process_masks=['semantic'] if needs_semantics else None,
                    )

                    if pano is None:
                        continue

                    # Strict content filter check
                    if self.strict_content_filter and valid_labels:
                        present_labels = (
                            list(pano.semantic_map.values())
                            if pano.semantic_map
                            else []
                        )

                        if not any(label in present_labels for label in valid_labels):
                            continue

                    # Target asset is confirmed visible, download heavy JPEG
                    pano.load_image_from_url()

                    pano.properties['longitude'] = pano_compact.properties.get(
                        'longitude'
                    )
                    pano.properties['latitude'] = pano_compact.properties.get(
                        'latitude'
                    )
                    pano.properties['compass_angle'] = pano_compact.properties.get(
                        'compass_angle'
                    )

                    cropped_pil = crop_panorama_to_asset(
                        target_asset=asset,
                        pano_image=pano,
                        vertical_crop_mode='smart' if self.smart_crop else 'full',
                    )

                    filename = f'{self.image_prefix}_{asset.id}_{axis_name}.jpg'
                    save_path = self.save_directory / filename
                    cropped_pil.save(save_path, format='JPEG')

                    new_asset = ImageAsset(
                        id=f'{asset.id}_{axis_name}',
                        path=save_path,
                        allow_missing_file=False,
                        properties={
                            'view_angle': axis_name,
                            'original_pano_id': pano_compact.id,
                        },
                    )
                    asset.add_image_assets(new_asset)
                    extracted_count += 1

                except Exception as e:
                    logging.error(
                        f'Failed to process pano {pano_compact.id} '
                        f'for asset {asset.id}: {e}'
                    )

            return extracted_count

        # Execute parallel downloads
        total_extracted = 0
        logging.info(f'Extracting panos using {self.max_workers} threads...')

        # Scale the connection pool to match the number of workers
        for prefix in ('http://', 'https://'):
            adapter = self.client.session.adapters.get(prefix)
            if adapter:
                adapter.pool_connections = self.max_workers
                adapter.pool_maxsize = self.max_workers * 2

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = [
                executor.submit(process_asset, asset) for asset in asset_collection
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc='Extracting Panoramas',
            ):
                total_extracted += future.result()

        logging.info(
            f'Finished! Successfully extracted {total_extracted} multi-angle '
            'cropped panoramas.'
        )
        return asset_collection
    
class BingOrthomosaicExtractor:
    """
    Component that extracts and synthesizes aerial imagery from Bing Maps Tiles.
    
    Downloads Bing Maps aerial tiles for a given geographical region and stitches 
    them into a single, continuous GeoTIFF. The resulting synthetic orthomosaic 
    retains its coordinate metadata and can be directly fed into region-wide 
    feature extractors (like SAM3OrthoFeatureExtractor) just like a standard 
    drone flight or satellite capture.

    Args:
        zoom_level (int, optional): 
            The detail level of the Bing Maps tiles to download, typically 
            ranging from 1 (entire world) to 19 (0.3m/pixel). 
            Defaults to 19.
        max_workers (int, optional): 
            The number of concurrent threads to use for downloading tiles in 
            parallel. Defaults to 10.

    Example:
        >>> from rapidtools.core import BoundingBox
        >>> from rapidtools.processing import BingOrthomosaicExtractor
        >>>
        >>> # Define the area of interest:
        >>> region = BoundingBox(
        ...     min_x=-118.251, min_y=34.050, max_x=-118.245, max_y=34.055
        ... )
        >>> 
        >>> # Initialize the extractor and stitch the region into a TIFF:
        >>> extractor = BingOrthomosaicExtractor(zoom_level=19)
        >>> tiff_path = extractor(region, output_path='downtown_la.tiff')
    """

    def __init__(self, zoom_level: int = 19, max_workers: int = 10) -> None:
        """Initialize the extractor configuration."""
        self.zoom_level = zoom_level
        self.max_workers = max_workers

    # --- MATH & COORDINATE UTILITIES ---
    @staticmethod
    def lat_lon_to_pixel(lat: float, lon: float, zoom: int) -> tuple[int, int]:
        """Convert latitude and longitude to Bing Maps pixel coordinates."""
        sin_lat = math.sin(lat * math.pi / 180.0)
        sin_lat = max(min(sin_lat, 0.9999), -0.9999)
        map_size = 256 << zoom
        
        pixel_x = ((lon + 180) / 360) * map_size
        pixel_y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * map_size
        return int(pixel_x), int(pixel_y)

    @staticmethod
    def tile_to_quadkey(tile_x: int, tile_y: int, zoom: int) -> str:
        """Convert tile XY coordinates into a Bing Maps Quadkey string."""
        quadkey = ''
        for i in range(zoom, 0, -1):
            digit = 0
            mask = 1 << (i - 1)
            if (tile_x & mask) != 0:
                digit += 1
            if (tile_y & mask) != 0:
                digit += 2
            quadkey += str(digit)
        return quadkey

    def __call__(
            self, 
            region: BoundingBox | PolygonRegion,
            output_path: str | Path
        ) -> Path:
        """
        Execute the tile downloading and stitching process for the given region.
        
        Args:
            region (Region): 
                The geographic area to extract imagery for. Accepts 
                ``BoundingBox`` or ``PolygonRegion``.
            output_path (str | Path): 
                The file path where the synthesized GeoTIFF should be saved. 
                Missing parent directories will be created automatically.

        Returns:
            Path: 
                The resolved, absolute path to the successfully saved 
                GeoTIFF file.
        """
        output_path = Path(output_path).resolve()
        
        # 1. Get geographic bounds (Works for both BoundingBox and PolygonRegion)
        min_lon, min_lat, max_lon, max_lat = region.bounds
        
        # 2. Convert to Bing pixel coordinates to define the exact canvas size
        px_min_x, px_max_y = self.lat_lon_to_pixel(min_lat, min_lon, self.zoom_level)
        px_max_x, px_min_y = self.lat_lon_to_pixel(max_lat, max_lon, self.zoom_level)
        
        canvas_width = px_max_x - px_min_x
        canvas_height = px_max_y - px_min_y
        
        logging.info(
            f'Stitching {canvas_width}x{canvas_height} px synthetic orthomosaic '
            f'at zoom {self.zoom_level}...'
        )
        
        # Initialize an empty canvas for pasting
        canvas = Image.new('RGB', (canvas_width, canvas_height))
        
        # 3. Determine which tiles intersect our bounding box
        tile_min_x = px_min_x // 256
        tile_max_x = px_max_x // 256
        tile_min_y = px_min_y // 256
        tile_max_y = px_max_y // 256
        
        tiles_to_download = []
        for tx in range(tile_min_x, tile_max_x + 1):
            for ty in range(tile_min_y, tile_max_y + 1):
                quadkey = self.tile_to_quadkey(tx, ty, self.zoom_level)
                tiles_to_download.append((tx, ty, quadkey))
                
        # 4. Download and paste tiles in parallel
        # Use the standardized session with built-in retries and exponential backoff
        with get_configured_session() as session:
            # Safely scale the connection pools for high-concurrency threading
            for prefix in ('http://', 'https://'):
                adapter = session.adapters[prefix]
                adapter.pool_connections = self.max_workers
                adapter.pool_maxsize = self.max_workers * 2
            
            def fetch_tile(tile_info: tuple[int, int, str]) -> tuple[int, int, Image.Image | None]:
                tx, ty, qk = tile_info
                url = f'http://ecn.t3.tiles.virtualearth.net/tiles/a{qk}.jpeg?g=1'
                
                try:
                    # Use the standardized timeout from config.py
                    resp = session.get(url, timeout=REQUESTS_TIMEOUT_VAL)
                    
                    # raise_for_status ensures we don't try to open a 404/Error page as an image
                    resp.raise_for_status() 
                    
                    return tx, ty, Image.open(BytesIO(resp.content))
                    
                except requests.RequestException as req_err:
                    logging.warning(
                        f'Network error fetching tile {qk} after retries: {req_err}'
                    )
                except Exception as parse_err:
                    logging.warning(
                        f'Failed to parse image data for tile {qk}: {parse_err}'
                    )
                    
                # Return None only if all retries failed or image is corrupted
                return tx, ty, None

            # Execute threaded downloads
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # ... (rest of the threaded execution loop remains the same)
                futures = [executor.submit(fetch_tile, t) for t in tiles_to_download]
                for future in concurrent.futures.as_completed(futures):
                    tx, ty, img = future.result()
                    if img:
                        # Calculate exact pixel offset. This perfectly crops 
                        # any extra tile overlap hanging outside the bounding box!
                        paste_x = (tx * 256) - px_min_x
                        paste_y = (ty * 256) - px_min_y
                        canvas.paste(img, (paste_x, paste_y))

        # 5. Save as a georeferenced GeoTIFF
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() not in ['.tif', '.tiff']:
            output_path = output_path.with_suffix('.tiff')

        # Convert PIL image to numpy array formatted for rasterio (bands, height, width)
        img_data = np.array(canvas)
        img_data = np.moveaxis(img_data, 2, 0)
        
        # Calculate geospatial transform mapping pixels to WGS84
        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, canvas_width, canvas_height)

        logging.info(f'Saving GeoTIFF to: {output_path}')
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=canvas_height,
            width=canvas_width,
            count=3,
            dtype=img_data.dtype,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            dst.write(img_data)

        return output_path