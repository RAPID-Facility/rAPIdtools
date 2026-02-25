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

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from rasterio import open as rasterio_open
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.warp import transform_bounds, transform_geom
from rasterio.windows import Window, from_bounds
from shapely.geometry import Polygon


class OrthomosaicReader:
    """
    Client for extracting high-resolution imagery from local orthomosaic rasters.
    
    This class is designed to be used as a context manager for efficient batch 
    processing. It opens the underlying TIFF file once, allowing you to rapidly 
    extract thousands of image patches without the I/O bottleneck of opening 
    and closing the file repeatedly.

    Example:
        >>> from rapidtools.data_sources import OrthomosaicReader
        >>> 
        >>> with OrthomosaicReader('path/to/map.tif') as reader:
        ...     patch, coords = reader.get_image_patch(asset.geometry)
    """

    def __init__(self, dataset_path: Union[str, Path]):
        """
        Initialize the reader and validate the target dataset.

        Args:
            dataset_path: The file path to the orthomosaic TIFF.

        Raises:
            FileNotFoundError: If the provided path does not exist on disk.
            ValueError: If the file does not have a valid TIFF extension.
        """
        self.dataset_path = Path(dataset_path).resolve()

        # Guard Clause 1: Verify the file actually exists
        if not self.dataset_path.is_file():
            raise FileNotFoundError(
                f'Dataset path \'{self.dataset_path}\' does not exist or is not'
                ' a file.'
            )

        # Guard Clause 2: Verify the file type is correct
        if self.dataset_path.suffix.lower() not in ('.tif', '.tiff'):
            raise ValueError(
                f'Dataset \'{self.dataset_path.name}\' must be a .tif or '
                '.tiff file.'
            )

        self._dataset: Optional[DatasetReader] = None
        self.dataset_extent = self.get_mosaic_bbox_wgs84()

    def __enter__(self):
        """Open the raster dataset for batch processing."""
        self._dataset = rasterio_open(
            self.dataset_path, 
            driver='GTiff', 
            num_threads='all_cpus'
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Safely close the raster dataset and free memory."""
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None

    def get_mosaic_bbox_wgs84(self) -> Tuple[float, float, float, float]:
        """
        Calculate the bounding box of the entire raster dataset in WGS84.

        Returns:
            Tuple[float, float, float, float]: 
                The geographic bounds as (min_lon, min_lat, max_lon, max_lat).
        """
        with rasterio_open(self.dataset_path, driver='GTiff') as dataset:
            return transform_bounds(
                dataset.crs, 
                CRS.from_epsg(4326), 
                *dataset.bounds
            )

    def get_image_patch(
        self, 
        asset_geometry: List[Tuple[float, float]], 
        max_missing_data_ratio: float = 0.2
    ) -> Optional[Tuple[Image.Image, List[Tuple[int, int]]]]:
        """
        Extract a bounded image patch for a specific geometric asset.

        Args:
            asset_geometry: 
                A list of (longitude, latitude) coordinates defining 
                the asset's outer polygon ring.
            max_missing_data_ratio: 
                The maximum acceptable percentage (0.0 to 1.0) 
                of nodata pixels. If exceeded, extraction is aborted.

        Returns:
            A tuple containing:
                
                - The extracted PIL Image in 8-bit RGB format.
                - The pixel coordinates of the asset polygon relative to the 
                  new image.
            Returns None if the extracted patch contains too much missing data 
            or falls completely outside the raster bounds.
            
        Raises:
            RuntimeError: If called outside of a context manager block.
        """
        if self._dataset is None:
            raise RuntimeError(
                'OrthomosaicReader must be used as a context manager. '
                'Use the \'with\' statement to open the reader.'
            )

        # Project geographic coordinates to the raster's native coordinate system:
        projected_coords =[
            self._project_point_from_wgs84(pt, self._dataset.crs) 
            for pt in asset_geometry
        ]
        projected_polygon = Polygon(projected_coords)
        
        # Calculate the buffered window and safely extract the raw pixel array:
        buffered_bounds = self._get_buffered_bounds(projected_polygon)
        image_array = self._read_image_window(self._dataset, buffered_bounds)

        # Abort if the requested bounds fall completely outside the actual raster:
        if image_array.size == 0:
            return None

        # Validate the array against the dataset's native 'nodata' value:
        if not self._is_image_valid(
                image_array, 
                max_missing_data_ratio, 
                self._dataset.nodata
            ):
            return None

        # Standardize the array to an 8-bit RGB image suitable for PIL and AI
        # models:
        image_hwc = self._format_array_for_pil(image_array)
        pil_image = Image.fromarray(image_hwc)

        # Calculate where the asset sits within the newly cropped image:
        projected_exterior = list(projected_polygon.exterior.coords)
        pixel_coords = self._get_polygon_pixel_coords(
            self._dataset, 
            projected_exterior, 
            buffered_bounds
        )

        return pil_image, pixel_coords

    @staticmethod
    def _project_point_from_wgs84(
        lon_lat_pt: Tuple[float, float], 
        destination_crs: CRS
    ) -> Tuple[float, float]:
        """Convert a single WGS84 point into target Coordinate Reference System."""
        feature = {'type': 'Point', 'coordinates': lon_lat_pt}
        projected = transform_geom(
            src_crs=CRS.from_epsg(4326), 
            dst_crs=destination_crs, 
            geom=feature
        )
        return tuple(projected['coordinates'])

    @staticmethod
    def _get_buffered_bounds(
        polygon: Polygon, 
        buffer_ratio: float = 0.2
    ) -> Tuple[float, float, float, float]:
        """
        Calculate a bounding box that provides padding around the polygon.
        
        The padding distance is dynamically calculated based on the maximum 
        width or height of the polygon multiplied by the buffer_ratio.
        """
        bounds = polygon.bounds
        width = abs(bounds[2] - bounds[0])
        height = abs(bounds[3] - bounds[1])
        
        buffer_dist = max(width, height) * buffer_ratio
        return polygon.buffer(buffer_dist).bounds

    @staticmethod
    def _read_image_window(
        dataset: DatasetReader, 
        bounds: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """
        Read a pixel window from the dataset safely.
        
        Intersects the requested geographic bounds with the raster's physical 
        dimensions to prevent out-of-bounds indexing errors.
        """
        window = from_bounds(*bounds, transform=dataset.transform)
        raster_window = Window(0, 0, dataset.width, dataset.height)
        safe_window = window.intersection(raster_window)
        
        return dataset.read(window=safe_window)

    @staticmethod
    def _format_array_for_pil(image_array: np.ndarray) -> np.ndarray:
        """
        Format a raw rasterio array into a standard format compatible with Pillow.
        
        Handles down-sampling of 16-bit or floating-point rasters to 8-bit, 
        strips out unnecessary bands (like near-infrared), and restructures the 
        matrix from (Channels, Height, Width) to (Height, Width, Channels).
        """
        # Slice to preserve only the first 3 bands (RGB)
        img = image_array[:3]
        
        # Normalize non-standard bit depths to standard 8-bit (0-255)
        if img.dtype != np.uint8:
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = ((img - img_min) / (img_max - img_min) * 255).astype(
                    np.uint8
                )
            else:
                img = img.astype(np.uint8)
                
        # Reorder axes to standard image format
        return np.moveaxis(img, 0, -1)

    @staticmethod
    def _is_image_valid(
        image_array: np.ndarray, 
        max_missing_ratio: float, 
        nodata_val: Any
    ) -> bool:
        """
        Determine if the extracted array contains an acceptable amount of valid
        data. Defaults to checking for `0` if the raster lacks an explicit 
        nodata value.
        """
        if image_array.size == 0:
            return False
            
        target_val = nodata_val if nodata_val is not None else 0
        zero_pixels = np.count_nonzero(image_array == target_val)
        zero_ratio = zero_pixels / image_array.size
        
        return zero_ratio < max_missing_ratio

    @staticmethod
    def _get_polygon_pixel_coords(
        dataset: DatasetReader, 
        polygon_coords: List[Tuple[float, float]], 
        buffered_bounds: Tuple[float, float, float, float]
    ) -> List[Tuple[int, int]]:
        """
        Translate geographic polygon coordinates into local pixel coordinates 
        relative to the extracted image patch.
        """
        pixel_coords_global =[dataset.index(x, y) for x, y in polygon_coords]
        
        # Calculate offset using the raw un-intersected bounds to keep local 
        # geometry intact:
        window = from_bounds(*buffered_bounds, transform=dataset.transform)
        
        return[
            (int(col - window.col_off), int(row - window.row_off)) 
            for row, col in pixel_coords_global
        ]