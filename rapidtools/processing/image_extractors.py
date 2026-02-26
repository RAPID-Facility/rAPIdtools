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
# 02-25-2026

import logging
from pathlib import Path

from PIL import ImageDraw
from tqdm import tqdm

from rapidtools.core import BoundingBox, ImageAsset, PhysicalAssetCollection
from rapidtools.data_sources import OrthomosaicReader


class AerialImageryExtractor:
    """
    Component that extracts aerial imagery patches for a collection of assets.
    
    This extractor reads from a high-resolution orthomosaic TIFF file, crops the 
    imagery around each asset's bounding box, saves the patches to disk, and 
    automatically attaches the new digital media to the corresponding 
    `PhysicalAsset` objects in the collection.
    
    Example:
        >>> from rapidtools.processing import AerialImageryExtractor
        >>>
        >>> extractor = AerialImageryExtractor(
        ...     dataset='data/post_disaster_map.tif',
        ...     save_directory='output/aerial_crops',
        ...     overlay_asset_outline=True,
        ...     image_prefix='post_disaster',
        ...     keep_multiple_copies=True
        ... )
        >>> processed_assets = extractor(my_asset_collection)
    """

    def __init__(
        self,
        dataset: str | Path,
        save_directory: str | Path,
        max_missing_data_ratio: float = 0.2,
        overlay_asset_outline: bool = False,
        image_prefix: str = 'aerial',
        keep_multiple_copies: bool = False
    ):
        """
        Initialize the extractor configuration.

        Args:
            dataset (str | Path): 
                The file path to the local orthomosaic TIFF dataset.
            save_directory (str | Path): 
                The directory where the extracted JPEG images will be saved. 
                Will be created if it does not exist.
            max_missing_data_ratio (float, optional): 
                The maximum acceptable proportion of nodata pixels (0.0 to 1.0). 
                If a cropped patch exceeds this, it is discarded. Defaults to 0.2.
            overlay_asset_outline (bool, optional): 
                If ``True``, draws a thick red polygon outline representing the 
                asset's geometry directly onto the saved image. Defaults to 
                ``False``.
            image_prefix (str, optional):
                The prefix to use for the saved image filenames and asset IDs. 
                Defaults to 'aerial'.
            keep_multiple_copies (bool, optional):
                If ``True``, prevents overwriting existing images with the same 
                coordinates by appending a numeric counter to the filename and ID.
                Defaults to ``False``.
        """
        self.dataset = Path(dataset).resolve()
        self.save_directory = Path(save_directory).resolve()
        self.max_missing_data_ratio = max_missing_data_ratio
        self.overlay_asset_outline = overlay_asset_outline
        self.image_prefix = image_prefix
        self.keep_multiple_copies = keep_multiple_copies

    def __call__(
        self, 
        asset_collection: PhysicalAssetCollection
    ) -> PhysicalAssetCollection:
        """
        Execute the extraction process on the provided asset collection.
        
        This method allows the class instance to be called like a function, 
        making it directly compatible with the `rapidtools` Pipeline engine.
        
        Args:
            asset_collection (PhysicalAssetCollection): 
                The collection of physical assets to extract imagery for.

        Returns:
            PhysicalAssetCollection: 
                The mutated collection, with new `ImageAsset` objects attached 
                to their corresponding `PhysicalAsset` entities.
        """
        self.save_directory.mkdir(parents=True, exist_ok=True)
        logging.info(f'Images will be saved to: {self.save_directory}')
        
        extracted_count = 0

        # Open the TIFF file ONCE for the entire batch using the context manager
        with OrthomosaicReader(self.dataset) as reader:
            
            # Efficiently filter assets to only those inside the raster's extent
            min_lon, min_lat, max_lon, max_lat = reader.dataset_extent
            raster_bbox = BoundingBox(
                min_x=min_lon, min_y=min_lat, max_x=max_lon, max_y=max_lat
            )
            
            assets_in_bounds = asset_collection.filter_by_geometry(raster_bbox)
            
            logging.info(
                f'Filtered assets by orthomosaic bounds: '
                f'{len(assets_in_bounds)} of {len(asset_collection)} assets remain.'
            )
            
            # Loop only over the filtered subset
            for asset in tqdm(assets_in_bounds, desc='Extracting aerial imagery'):
                
                # 1. Ensure we have a valid polygon
                geom = asset.geometry
                if geom.geom_type == 'MultiPolygon':
                    poly = geom.geoms[0]
                elif geom.geom_type == 'Polygon':
                    poly = geom
                else:
                    logging.debug(
                        f'Skipping asset \'{asset.id}\': Geometry is '
                        f'{geom.geom_type}, expected Polygon/MultiPolygon.'
                    )
                    continue
                    
                asset_coords = list(poly.exterior.coords)
                
                # 2. Extract image patch using the open reader
                extraction_result = reader.get_image_patch(
                    asset_geometry=asset_coords, 
                    max_missing_data_ratio=self.max_missing_data_ratio
                )
                
                if extraction_result is None:
                    continue
                    
                pil_image, pixel_coords = extraction_result

                # 3. Draw outline if requested
                if self.overlay_asset_outline:
                    draw = ImageDraw.Draw(pil_image)
                    # Close the polygon loop by appending the first point to the end
                    draw.line(
                        pixel_coords + [pixel_coords[0]],
                        fill='red',
                        width=6
                    )

                # 4. Create a clean, coordinate-based filename
                centroid = poly.centroid
                coords_str = f'{centroid.y:.8f}_{centroid.x:.8f}'.replace('.', '')
                
                base_name = f'{self.image_prefix}_{coords_str}'
                image_name = f'{base_name}.jpg'
                image_path = self.save_directory / image_name
                asset_id_suffix = self.image_prefix

                # Handle potential overlaps if requested
                if self.keep_multiple_copies:
                    counter = 1
                    while image_path.exists():
                        image_name = f'{base_name}_{counter}.jpg'
                        image_path = self.save_directory / image_name
                        asset_id_suffix = f'{self.image_prefix}_{counter}'
                        counter += 1

                # 5. Save the image to disk
                pil_image.save(image_path)
                
                # 6. Create the ImageAsset domain object and attach it
                img_asset = ImageAsset(
                    id=f'{asset.id}_{asset_id_suffix}',  
                    path=image_path,
                    allow_missing_file=False
                )
                
                asset.add_image_assets(img_asset)
                extracted_count += 1

        logging.info(
            f'Extracted aerial imagery for a total of {extracted_count} assets.'
        )
        
        return asset_collection