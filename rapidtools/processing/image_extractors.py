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
# 03-25-2026

import logging
from pathlib import Path

from PIL import ImageDraw
from tqdm import tqdm

from rapidtools.core import BoundingBox, ImageAsset, PhysicalAssetCollection
from rapidtools.data_sources import OrthomosaicReader


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