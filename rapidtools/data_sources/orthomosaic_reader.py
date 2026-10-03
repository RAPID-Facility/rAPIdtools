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

"""
Reader for extracting georeferenced image patches from local orthomosaics.

This module wraps :mod:`rasterio` to provide a lightweight, context-managed
reader for large GeoTIFF orthomosaics. It supports extracting buffered image
patches around WGS84 geometries, sliding-window tiling of the whole raster,
and a few geodesic helpers used to reason about the raster's real-world size.

Example:
    Create a tiny synthetic GeoTIFF and read a patch around a point:

    >>> import numpy as np
    >>> import rasterio
    >>> from rasterio.transform import from_origin
    >>> from rapidtools.data_sources import OrthomosaicReader
    >>>
    >>> data = np.random.randint(1, 255, (3, 64, 64), dtype=np.uint8)
    >>> transform = from_origin(-118.2, 34.1, 1e-4, 1e-4)
    >>> with rasterio.open(
    ...     'demo.tif', 'w', driver='GTiff', width=64, height=64, count=3,
    ...     dtype='uint8', crs='EPSG:4326', transform=transform
    ... ) as dst:
    ...     dst.write(data)
    >>> with OrthomosaicReader('demo.tif') as reader:
    ...     patch, pixel_coords, bbox = reader.get_image_patch(
    ...         [(-118.197, 34.097)], buffer='10 m'
    ...     )
    >>> patch.mode
    'RGB'
"""

import math
import re
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
import rasterio
import rasterio.features
import rasterio.windows
from PIL import Image
from rasterio import open as rasterio_open
from rasterio.crs import CRS
from rasterio.errors import WindowError
from rasterio.io import DatasetReader
from rasterio.transform import Affine
from rasterio.warp import transform_bounds, transform_geom
from rasterio.windows import Window, from_bounds
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from rapidtools.constants import (
    EARTH_RADIUS_KM,
    LATITUDE_SPACING_KM,
    METERS_CONVERSION_FACTORS,
    UNIT_ALIASES,
)

DEFAULT_PATCH_SIZE = 512
DEFAULT_PATCH_OVERLAP_RATIO = 0.2


@dataclass(frozen=True)
class PatchGeoref:
    """
    Native georeferencing of an image patch read from a raster.

    A patch's WGS84 bounding box is only an envelope: for a projected or
    rotated raster the patch edges are not lines of constant latitude and
    longitude, so mapping pixels through that envelope stretches and skews
    the result. This record keeps the exact pixel-to-CRS affine transform of
    the window that was read so masks drawn on the patch can be vectorized in
    the raster's own grid and then reprojected.

    Attributes:
        transform: Affine transform from patch pixel coordinates to ``crs``.
        crs: The raster's coordinate reference system.
        width: Patch width in pixels when it was read.
        height: Patch height in pixels when it was read.

    Example:
        >>> from rasterio.crs import CRS
        >>> from rasterio.transform import Affine
        >>> georef = PatchGeoref(Affine(0.5, 0, 500000, 0, -0.5, 4100000),
        ...                      CRS.from_epsg(32611), 100, 100)
        >>> georef.transform_for(50, 50).a
        1.0
        >>> PatchGeoref.from_dict(georef.to_dict()) == georef
        True
    """

    transform: Affine
    crs: CRS
    width: int
    height: int

    def transform_for(self, width: int, height: int) -> Affine:
        """
        Return the affine transform for a resampled copy of the patch.

        Args:
            width: Width in pixels of the array being georeferenced.
            height: Height in pixels of the array being georeferenced.

        Returns:
            Affine: ``transform`` rescaled so the array still covers the
            same ground footprint.
        """
        if (width, height) == (self.width, self.height):
            return self.transform
        return self.transform * Affine.scale(self.width / width, self.height / height)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation (see :meth:`from_dict`)."""
        t = self.transform
        return {
            'transform': [t.a, t.b, t.c, t.d, t.e, t.f],
            'crs': self.crs.to_wkt(),
            'width': int(self.width),
            'height': int(self.height),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'PatchGeoref':
        """Rebuild the record from :meth:`to_dict` output."""
        a, b, c, d, e, f = (float(v) for v in data['transform'][:6])
        return cls(
            transform=Affine(a, b, c, d, e, f),
            crs=CRS.from_user_input(data['crs']),
            width=int(data['width']),
            height=int(data['height']),
        )


_NOT_ENTERED_MSG = (
    'OrthomosaicReader must be used as a context manager. '
    "Use the 'with' statement to open the reader."
)


class OrthomosaicReader:
    """
    Client for extracting high-resolution imagery from local orthomosaic rasters.

    This class is designed to be used as a context manager for efficient batch
    processing. It opens the underlying TIFF file once, allowing you to rapidly
    extract thousands of image patches without the I/O bottleneck of opening
    and closing the file repeatedly.

    On entering the context the reader also samples a downsampled thumbnail
    (or the smallest suitable overview) to compute robust global 1st/99th
    percentile statistics. These are used to radiometrically scale
    high-bit-depth imagery consistently across all extracted patches.

    Attributes:
        dataset_path (Path): Absolute path to the orthomosaic on disk.
        width (int): Raster width in pixels.
        height (int): Raster height in pixels.
        global_min (float | None): Robust global minimum pixel value, set on
            ``__enter__``.
        global_max (float | None): Robust global maximum pixel value, set on
            ``__enter__``.
        dataset_extent (tuple | None): WGS84 bounds
            ``(min_lon, min_lat, max_lon, max_lat)``, set on ``__enter__``.

    Example:
        >>> from rapidtools.data_sources import OrthomosaicReader
        >>>
        >>> with OrthomosaicReader('path/to/map.tif') as reader:
        ...     result = reader.get_image_patch(asset_coords)
        ...     if result is not None:
        ...         patch, pixel_coords, bbox = result
    """

    def __init__(self, dataset_path: str | Path):
        """
        Initialize the reader and validate the target dataset.

        Args:
            dataset_path: The file path to the orthomosaic TIFF.

        Raises:
            FileNotFoundError: If the provided path does not exist on disk.
            ValueError: If the file does not have a valid TIFF extension.

        Example:
            >>> reader = OrthomosaicReader('path/to/map.tif')
            >>> reader.width, reader.height
            (64, 64)
        """
        self.dataset_path = Path(dataset_path).resolve()

        # Verify the file exists:
        if not self.dataset_path.is_file():
            raise FileNotFoundError(
                f"Dataset path '{self.dataset_path}' does not exist or is not a file."
            )

        # Verify the file type is correct:
        if self.dataset_path.suffix.lower() not in ('.tif', '.tiff'):
            raise ValueError(
                f"Dataset '{self.dataset_path.name}' must be a .tif or .tiff file."
            )

        self._dataset: DatasetReader | None = None
        self.global_min: float | None = None
        self.global_max: float | None = None
        self.dataset_extent: tuple[float, float, float, float] | None = None

        # Briefly open the dataset to extract header metadata. This is strictly
        # an I/O metadata read and will not blow memory usage:
        with rasterio_open(self.dataset_path) as src:
            self.width = src.width
            self.height = src.height

    def __enter__(self):
        """
        Open the raster dataset for batch processing and fetch global stats.

        Returns:
            OrthomosaicReader: The opened reader instance.

        Example:
            >>> with OrthomosaicReader('path/to/map.tif') as reader:
            ...     print(reader.dataset_extent)
            (-118.2, 34.0936, -118.1936, 34.1)
        """
        self._dataset = rasterio_open(
            self.dataset_path, driver='GTiff', num_threads='all_cpus'
        )

        # Fetch extent of the dataset:
        self.dataset_extent = self.get_mosaic_bbox_wgs84()

        # Check if the dataset has pre-computed overviews (pyramids) on band 1:
        overviews = self._dataset.overviews(1)
        target_size = 2048
        out_shape = None

        if overviews:
            # Find the highest-resolution overview that is roughly target_size
            # or smaller to ensure an accurate statistical sample.
            for decimation_factor in overviews:
                overview_h = self._dataset.height // decimation_factor
                overview_w = self._dataset.width // decimation_factor

                if max(overview_h, overview_w) <= target_size:
                    out_shape = (
                        self._dataset.count,
                        max(1, overview_h),
                        max(1, overview_w),
                    )
                    break

            # If all overviews are still larger than the target size,
            # fall back to the lowest resolution one (the last in the list):
            if out_shape is None:
                decimation_factor = overviews[-1]
                out_shape = (
                    self._dataset.count,
                    max(1, self._dataset.height // decimation_factor),
                    max(1, self._dataset.width // decimation_factor),
                )
        else:
            # Fallback if no overviews exist by reading a 2048x2048 thumbnail:
            out_shape = (
                self._dataset.count,
                min(target_size, self._dataset.height),
                min(target_size, self._dataset.width),
            )

        # Read the overview or downsampled thumbnail:
        thumbnail = self._dataset.read(out_shape=out_shape)

        # Use the dataset's nodata value:
        nodata_val = self._dataset.nodata

        if nodata_val is not None:
            # Filter out true nodata pixels:
            valid_data = thumbnail[thumbnail != nodata_val]
        else:
            # Fallback if the raster lacks explicit nodata metadata:
            valid_data = thumbnail[thumbnail != 0]

        # Calculate robust global min/max using the 1st and 99th percentiles:
        if valid_data.size > 0:
            self.global_min = float(np.percentile(valid_data, 1))
            self.global_max = float(np.percentile(valid_data, 99))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Safely close the raster dataset and free memory.

        Args:
            exc_type: Exception type raised inside the ``with`` block, if any.
            exc_val: Exception instance raised inside the block, if any.
            exc_tb: Traceback of the exception, if any.
        """
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None

    def get_mosaic_bbox_wgs84(self) -> tuple[float, float, float, float]:
        """
        Calculate the bounding box of the entire raster dataset in WGS84.

        Works both inside and outside the context manager. Outside the
        context, the file is briefly opened to read its metadata.

        Returns:
            tuple[float, float, float, float]:
                The geographic bounds as (min_lon, min_lat, max_lon, max_lat).

        Example:
            >>> reader = OrthomosaicReader('path/to/map.tif')
            >>> min_lon, min_lat, max_lon, max_lat = reader.get_mosaic_bbox_wgs84()
            >>> min_lon < max_lon and min_lat < max_lat
            True
        """
        # If the dataset is already open, use it to determine extent:
        if self._dataset is not None:
            return transform_bounds(
                self._dataset.crs, CRS.from_epsg(4326), *self._dataset.bounds
            )

        # If the dataset is not open, open the file and compute the extent:
        with rasterio_open(self.dataset_path, driver='GTiff') as dataset:
            return transform_bounds(dataset.crs, CRS.from_epsg(4326), *dataset.bounds)

    @overload
    def get_image_patch(
        self,
        asset_geometry: list[tuple[float, float]],
        max_missing_data_ratio: float = ...,
        buffer: float | str = ...,
        force_square: bool = ...,
        min_footprint_coverage: float | None = ...,
        pad_edges: bool | None = ...,
        return_georef: Literal[False] = ...,
    ) -> (
        tuple[Image.Image, list[tuple[int, int]], tuple[float, float, float, float]]
        | None
    ): ...

    @overload
    def get_image_patch(
        self,
        asset_geometry: list[tuple[float, float]],
        max_missing_data_ratio: float = ...,
        buffer: float | str = ...,
        force_square: bool = ...,
        min_footprint_coverage: float | None = ...,
        pad_edges: bool | None = ...,
        *,
        return_georef: Literal[True],
    ) -> (
        tuple[
            Image.Image,
            list[tuple[int, int]],
            tuple[float, float, float, float],
            PatchGeoref,
        ]
        | None
    ): ...

    def get_image_patch(
        self,
        asset_geometry: list[tuple[float, float]],
        max_missing_data_ratio: float = 0.2,
        buffer: float | str = '20%',
        force_square: bool = True,
        min_footprint_coverage: float | None = None,
        pad_edges: bool | None = None,
        return_georef: bool = False,
    ) -> tuple[Any, ...] | None:
        """
        Extract a bounded image patch for a geometric asset.

        The geometry type is inferred from the number of coordinates: one pair
        is a ``Point``, two pairs a ``LineString``, and four or more pairs
        whose first and last coordinates coincide form a ``Polygon``. Any
        other open sequence is treated as a ``LineString``.

        Args:
            asset_geometry:
                A list of (longitude, latitude) coordinates defining
                the asset (``Point``, ``Line``, or ``Polygon``).
            max_missing_data_ratio:
                The maximum acceptable fraction (0.0 to 1.0) of missing pixels
                in the crop, judged from the dataset mask (nodata value, alpha
                band or internal mask). If reached, extraction is aborted.
            buffer:
                The padding applied around the geometry. Can be a percentage
                (e.g., '20%'), an absolute CRS distance (e.g., 50.0), or a
                real-world distance (e.g., '50 ft', '10 m').
            force_square:
                If ``True``, guarantees the extracted image patch covers a
                perfectly square area in the real world.
            min_footprint_coverage:
                Minimum fraction (0.0 to 1.0) of the asset footprint that must
                contain valid (non-nodata) pixels. When given, this check
                replaces the patch-level ``max_missing_data_ratio`` test so
                assets on the edge of the imagery are kept. Defaults to
                ``None``.
            pad_edges:
                Pad crop windows that extend beyond the raster with nodata so
                they keep their full requested size and shape. Defaults to
                ``True`` when ``min_footprint_coverage`` is given, ``False``
                otherwise (window clipped to the raster).
            return_georef:
                If ``True``, a :class:`PatchGeoref` with the patch's native
                affine transform and CRS is returned as a fourth element.

        Returns:
            A tuple containing:

                - The extracted PIL Image in 8-bit RGB format.
                - The local pixel coordinates of the asset relative to the new
                  image patch.
                - The WGS84 bounding box of the extracted image patch:
                  ``(min_lon, min_lat, max_lon, max_lat)``.
                - With ``return_georef=True``, the :class:`PatchGeoref` of
                  the patch.

            Returns ``None`` if the extracted patch contains too much missing
            data or falls completely outside the raster bounds.

        Raises:
            RuntimeError: If called outside of a context manager block.
            ValueError: If the asset geometry is empty or the buffer string
                cannot be parsed.

        Example:
            >>> footprint = [
            ...     (-118.198, 34.098), (-118.197, 34.098),
            ...     (-118.197, 34.097), (-118.198, 34.097),
            ...     (-118.198, 34.098),
            ... ]
            >>> with OrthomosaicReader('path/to/map.tif') as reader:
            ...     patch, pixel_coords, bbox = reader.get_image_patch(
            ...         footprint, buffer='5 m', force_square=False
            ...     )
            >>> len(pixel_coords)
            5
        """
        if self._dataset is None:
            raise RuntimeError(_NOT_ENTERED_MSG)

        if not asset_geometry:
            raise ValueError('asset_geometry cannot be empty.')

        # Build the Shapely geometry corresponding to the geometry input:
        num_coords = len(asset_geometry)
        coords: (
            tuple[float, float]
            | list[tuple[float, float]]
            | list[list[tuple[float, float]]]
        )
        # Single coordinate pair:
        if num_coords == 1:
            geom_type = 'Point'
            coords = asset_geometry[0]

        # Two points:
        elif num_coords == 2:
            geom_type = 'LineString'
            coords = asset_geometry

        # 3 or more points:
        else:
            # Check if it is a closed ring:
            is_closed = asset_geometry[0] == asset_geometry[-1]
            if is_closed and num_coords >= 4:
                geom_type = 'Polygon'
                coords = [asset_geometry]
            else:
                geom_type = 'LineString'
                coords = asset_geometry

        # Construct a GeoJSON-like dictionary to transform all at once:
        geo_dict = {'type': geom_type, 'coordinates': coords}

        # Project geographic coordinates to the raster's native coordinate
        # system:
        projected_geo_dict = transform_geom(
            src_crs=CRS.from_epsg(4326), dst_crs=self._dataset.crs, geom=geo_dict
        )

        # Automatically build the Shapely geometry from the projected dict:
        geom = shape(projected_geo_dict)

        # Determine CRS characteristics for accurate real-world buffering:
        crs = self._dataset.crs
        is_geo = crs.is_geographic

        meters_per_unit = 1.0
        if not is_geo:
            linear_units = crs.linear_units.lower()
            if 'foot' in linear_units or 'ft' in linear_units:
                meters_per_unit = 0.3048
            elif 'meter' in linear_units or 'metre' in linear_units:
                meters_per_unit = 1.0

        # Calculate the buffered window:
        buffered_bounds = self._get_buffered_bounds(
            geom,
            buffer=buffer,
            force_square=force_square,
            is_geographic=is_geo,
            meters_per_crs_unit=meters_per_unit,
        )

        # Safely extract the raw pixel array (padded when requested):
        boundless = (
            pad_edges if pad_edges is not None else min_footprint_coverage is not None
        )
        image_array, read_window = self._read_patch(
            self._dataset, buffered_bounds, boundless=boundless
        )
        if image_array.size == 0:
            return None
        # Validity comes from the dataset mask, which honours nodata values,
        # alpha bands and internal masks alike (rotated orthomosaics often mark
        # their empty corners only through an alpha band):
        valid_mask = (
            self._dataset.read_masks(1, window=read_window, boundless=boundless) > 0
        )
        if min_footprint_coverage is not None:
            coverage = self._footprint_coverage(
                valid_mask, geom, read_window, self._dataset
            )
            if coverage < min_footprint_coverage:
                return None
        elif (1.0 - float(valid_mask.mean())) >= max_missing_data_ratio:
            return None

        # Standardize the array to an 8-bit RGB image suitable for PIL and AI
        # models:
        image_hwc = self._format_array_for_pil(
            image_array, global_min=self.global_min, global_max=self.global_max
        )
        pil_image = Image.fromarray(image_hwc)

        # Calculate local pixel coordinates:
        if geom.geom_type == 'Polygon':
            geom_coords = list(geom.exterior.coords)
        else:
            geom_coords = list(geom.coords)

        pixel_coords = self._get_local_pixel_coords(
            self._dataset, geom_coords, buffered_bounds, boundless=boundless
        )

        # Calculate the exact WGS84 bounding box of the cropped image patch
        # from the window that was actually read (clipped or padded):
        safe_native_bounds = rasterio.windows.bounds(
            read_window, self._dataset.transform
        )

        # Transform safe native bounds to WGS84:
        wgs84_bounds = transform_bounds(crs, CRS.from_epsg(4326), *safe_native_bounds)

        if return_georef:
            georef = PatchGeoref(
                transform=rasterio.windows.transform(
                    read_window, self._dataset.transform
                ),
                crs=crs,
                width=pil_image.width,
                height=pil_image.height,
            )
            return pil_image, pixel_coords, wgs84_bounds, georef
        return pil_image, pixel_coords, wgs84_bounds

    @overload
    def generate_tiles(
        self,
        patch_size: float = ...,
        unit: str = ...,
        overlap_ratio: float = ...,
        max_missing_data_ratio: float = ...,
        pad_edge_tiles: bool = ...,
        return_georef: Literal[False] = ...,
    ) -> Generator[
        tuple[Image.Image, tuple[float, float, float, float]], None, None
    ]: ...

    @overload
    def generate_tiles(
        self,
        patch_size: float = ...,
        unit: str = ...,
        overlap_ratio: float = ...,
        max_missing_data_ratio: float = ...,
        pad_edge_tiles: bool = ...,
        *,
        return_georef: Literal[True],
    ) -> Generator[
        tuple[Image.Image, tuple[float, float, float, float], PatchGeoref],
        None,
        None,
    ]: ...

    def generate_tiles(
        self,
        patch_size: float = DEFAULT_PATCH_SIZE,
        unit: str = 'pixels',
        overlap_ratio: float = DEFAULT_PATCH_OVERLAP_RATIO,
        max_missing_data_ratio: float = 0.8,
        pad_edge_tiles: bool = True,
        return_georef: bool = False,
    ) -> Generator[tuple[Any, ...], None, None]:
        """
        Scan through the raster and yield populated image patches.

        The raster is traversed with a sliding window from the top-left
        corner. Each candidate window is first evaluated using the dataset
        mask so that expensive pixel reads are only performed for tiles with
        an acceptable amount of valid data.

        Args:
            patch_size:
                The dimension of the square patch in the specified unit
                (e.g., 100 for a 100x100 meter patch).
            unit:
                The spatial unit of the patch size. Supported units include:
                'pixels', 'meters', 'feet', 'yards', 'kilometers', 'miles'.
            overlap_ratio:
                Float between 0.0 and 1.0 representing the percentage of overlap.
            max_missing_data_ratio:
                The maximum allowable ratio of empty/nodata pixels.
            pad_edge_tiles:
                If True, pads tiles at the edges of the map with nodata to
                maintain a strictly uniform output array size.
            return_georef:
                If ``True``, each tile also carries a :class:`PatchGeoref`
                with the tile's native affine transform and CRS, which is
                what mask vectorization should use for anything that is not
                a north-up WGS84 raster.

        Yields:
            tuple[Image.Image, tuple[float, float, float, float]]:
                An 8-bit RGB PIL image and its WGS84 bounding box
                ``(min_lon, min_lat, max_lon, max_lat)``; a third
                :class:`PatchGeoref` element when ``return_georef`` is set.

        Raises:
            RuntimeError: If called outside of a context manager block.
            ValueError: If ``overlap_ratio`` is outside ``[0.0, 1.0)`` or the
                unit is unsupported.

        Example:
            >>> with OrthomosaicReader('path/to/map.tif') as reader:
            ...     tiles = list(
            ...         reader.generate_tiles(patch_size=32, overlap_ratio=0.0)
            ...     )
            >>> tiles[0][0].size
            (32, 32)
        """
        if self._dataset is None:
            raise RuntimeError(_NOT_ENTERED_MSG)

        if not (0.0 <= overlap_ratio < 1.0):
            raise ValueError('overlap_ratio must be between 0.0 and 0.99')

        # Standardize and parse the requested unit:
        clean_unit = str(unit).strip().lower()
        standard_unit = UNIT_ALIASES.get(clean_unit)

        if standard_unit is None:
            raise ValueError(
                f"Unsupported unit '{unit}'. Supported units include: "
                'pixels, meters, kilometers, feet, miles, yards.'
            )

        # Calculate patch dimensions in native pixels:
        if standard_unit == 'pixels':
            width_px = height_px = int(patch_size)
        else:
            # Convert requested size to meters first:
            target_meters = patch_size / METERS_CONVERSION_FACTORS[standard_unit]

            # Determine real-world size of a single pixel using the affine
            # transform transform.a = pixel width, transform.e = pixel height
            # (usually negative):
            pixel_width_native = abs(self._dataset.transform.a)
            pixel_height_native = abs(self._dataset.transform.e)

            if self._dataset.crs.is_geographic:
                # Geographic CRS (degrees): dynamically calculate meters per
                # degree:
                extent = self.dataset_extent
                if extent is None:
                    raise RuntimeError(_NOT_ENTERED_MSG)
                center_lat = extent[1] + (extent[3] - extent[1]) / 2.0
                cos_lat = max(0.00001, math.cos(math.radians(center_lat)))

                m_per_deg_y = LATITUDE_SPACING_KM * 1000.0
                m_per_deg_x = m_per_deg_y * cos_lat

                pixel_width_m = pixel_width_native * m_per_deg_x
                pixel_height_m = pixel_height_native * m_per_deg_y
            else:
                # Projected CRS (linear units like meters or feet):
                linear_units = self._dataset.crs.linear_units.lower()
                crs_meters_per_unit = (
                    0.3048 if 'foot' in linear_units or 'ft' in linear_units else 1.0
                )

                pixel_width_m = pixel_width_native * crs_meters_per_unit
                pixel_height_m = pixel_height_native * crs_meters_per_unit

            # Calculate how many pixels fit into the target real-world distance:
            width_px = max(1, int(round(target_meters / pixel_width_m)))
            height_px = max(1, int(round(target_meters / pixel_height_m)))

        # Calculate strides for the sliding window:
        stride_x = max(1, int(width_px * (1.0 - overlap_ratio)))
        stride_y = max(1, int(height_px * (1.0 - overlap_ratio)))

        # Iterate through the grid:
        for row_off in range(0, self._dataset.height, stride_y):
            for col_off in range(0, self._dataset.width, stride_x):
                requested_window = Window(col_off, row_off, width_px, height_px)
                raster_window = Window(0, 0, self._dataset.width, self._dataset.height)
                safe_window = requested_window.intersection(raster_window)

                # Mask-first evaluation for efficiency:
                mask = self._dataset.read_masks(1, window=safe_window)
                if mask.size == 0:
                    continue

                total_expected_pixels = (
                    (width_px * height_px) if pad_edge_tiles else mask.size
                )
                valid_pixels = np.count_nonzero(mask)
                missing_ratio = 1.0 - (valid_pixels / total_expected_pixels)

                if missing_ratio > max_missing_data_ratio:
                    continue

                # Read full image array only for valid patches:
                image_array = self._dataset.read(window=safe_window)

                # Pad edge tiles to strict pixel sizes if requested:
                if pad_edge_tiles and (
                    image_array.shape[1] < height_px or image_array.shape[2] < width_px
                ):
                    pad_h = height_px - image_array.shape[1]
                    pad_w = width_px - image_array.shape[2]
                    nodata_val = (
                        self._dataset.nodata if self._dataset.nodata is not None else 0
                    )

                    image_array = np.pad(
                        image_array,
                        pad_width=((0, 0), (0, pad_h), (0, pad_w)),
                        mode='constant',
                        constant_values=nodata_val,
                    )
                    georef_window = requested_window
                else:
                    georef_window = safe_window
                window_bounds = rasterio.windows.bounds(
                    georef_window, self._dataset.transform
                )

                # Transform native bounds to WGS84:
                wgs84_bounds = transform_bounds(
                    self._dataset.crs, CRS.from_epsg(4326), *window_bounds
                )

                # Convert to PIL image:
                image_hwc = self._format_array_for_pil(
                    image_array,
                    global_min=self.global_min,
                    global_max=self.global_max,
                )
                pil_image = Image.fromarray(image_hwc)

                if return_georef:
                    georef = PatchGeoref(
                        transform=rasterio.windows.transform(
                            georef_window, self._dataset.transform
                        ),
                        crs=self._dataset.crs,
                        width=pil_image.width,
                        height=pil_image.height,
                    )
                    yield pil_image, wgs84_bounds, georef
                else:
                    yield pil_image, wgs84_bounds

    def get_raster_dimensions(self, unit: str = 'pixels') -> tuple[float, float]:
        """
        Determine the width and height of the raster.

        For non-pixel units the dimensions are computed as great-circle
        distances across the WGS84 extent: width along the central latitude
        and height along the central longitude.

        Args:
            unit:
                The desired unit of measurement (e.g., 'pixels', 'm', 'feet',
                'km'). Defaults to 'pixels'.

        Returns:
            tuple[float, float]: The (width, height) of the raster.

        Raises:
            RuntimeError:
                If called outside of a context manager block.
            ValueError:
                If an unsupported unit is provided.

        Example:
            >>> with OrthomosaicReader('path/to/map.tif') as reader:
            ...     width_px, height_px = reader.get_raster_dimensions()
            ...     width_m, height_m = reader.get_raster_dimensions('m')
            >>> width_px
            64.0
        """
        if self._dataset is None:
            raise RuntimeError(_NOT_ENTERED_MSG)

        # Parse the unit input using the shared unit aliases:
        clean_input = str(unit).strip().lower()
        standardized_unit = UNIT_ALIASES.get(clean_input)

        if standardized_unit is None:
            raise ValueError(
                f"Unsupported unit '{unit}'. "
                'Supported units include: pixels, meters, kilometers, feet, '
                'miles, yards.'
            )

        # If unit is set to pixels, return raw pixel dimensions:
        if standardized_unit == 'pixels':
            return float(self._dataset.width), float(self._dataset.height)

        # Calculate geographic distance dimensions
        # Get the WGS84 bounding box (min_lon, min_lat, max_lon, max_lat)
        min_lon, min_lat, max_lon, max_lat = self.get_mosaic_bbox_wgs84()
        center_lat = (min_lat + max_lat) / 2.0
        center_lon = (min_lon + max_lon) / 2.0

        # Width is distance between min_lon and max_lon at the center latitude:
        spatial_width_m = self.haversine_distance(
            min_lon, center_lat, max_lon, center_lat
        )

        # Height is the distance between min_lat and max_lat at the center
        # longitude:
        spatial_height_m = self.haversine_distance(
            center_lon, min_lat, center_lon, max_lat
        )

        # Convert meters to the requested standardized unit:
        factor = METERS_CONVERSION_FACTORS[standardized_unit]
        return spatial_width_m * factor, spatial_height_m * factor

    @staticmethod
    def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """
        Calculate great-circle distance between two points on Earth's surface.

        Args:
            lon1: Longitude of the first point in degrees.
            lat1: Latitude of the first point in degrees.
            lon2: Longitude of the second point in degrees.
            lat2: Latitude of the second point in degrees.

        Returns:
            float: The geographic distance in meters.

        Example:
            >>> d = OrthomosaicReader.haversine_distance(0.0, 0.0, 1.0, 0.0)
            >>> round(d / 1000.0, 1)
            111.2
        """
        earth_radius_m = EARTH_RADIUS_KM * 1000.0  # Radius of Earth in meters

        # Convert degrees to radians:
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)

        # Wrap the longitude difference to be between -180 and 180.
        # This prevents calculation errors if the bounding box crosses the date
        # line:
        dlon = (lon2 - lon1 + 180.0) % 360.0 - 180.0
        dlambda = math.radians(dlon)

        # Haversine formula:
        sin_half_dphi = math.sin(dphi / 2.0)
        sin_half_dlambda = math.sin(dlambda / 2.0)

        a = (sin_half_dphi**2) + (
            math.cos(phi1) * math.cos(phi2) * (sin_half_dlambda**2)
        )

        # Guard against floating-point rounding errors exceeding 1.0, also
        # guard against < 0.0 just to be hyper-safe:
        a = min(1.0, max(0.0, a))

        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

        return earth_radius_m * c

    @staticmethod
    def _get_buffered_bounds(
        geom: BaseGeometry,
        buffer: float | str = '20%',
        force_square: bool = True,
        is_geographic: bool = False,
        meters_per_crs_unit: float = 1.0,
    ) -> tuple[float, float, float, float]:
        """
        Calculate padded bounding box coordinates for a given geometry.

        This method dynamically scales the bounding box and mathematically
        accounts for Earth's curvature when working with geographic
        coordinates (degrees).

        Args:
            geom: Shapely geometry expressed in the raster's native CRS.
            buffer: Padding specification. A percentage string ('20%'), a
                real-world distance string ('50 ft', '10 m'), or a bare
                number interpreted in native CRS units.
            force_square: If ``True``, the returned bounds describe a
                real-world square centered on the geometry.
            is_geographic: Whether the native CRS uses degrees.
            meters_per_crs_unit: Meters per linear CRS unit for projected
                systems (e.g., 0.3048 for feet).

        Returns:
            tuple[float, float, float, float]: Buffered bounds as
            ``(minx, miny, maxx, maxy)`` in native CRS units.

        Raises:
            ValueError: If the buffer string cannot be parsed or uses an
                unsupported distance unit.

        Example:
            >>> from shapely.geometry import box
            >>> OrthomosaicReader._get_buffered_bounds(
            ...     box(0, 0, 10, 10), buffer='50%', force_square=False
            ... )
            (-5.0, -5.0, 15.0, 15.0)
        """
        # Determine the center, width and height of the input geometry:
        minx, miny, maxx, maxy = geom.bounds
        center_x = (minx + maxx) / 2.0
        center_y = (miny + maxy) / 2.0

        width = maxx - minx
        height = maxy - miny

        # Determine the cosine scaling factor for longitude (1.0 if not
        # geographic) to compensate for geographic curvature :
        cos_lat = (
            max(0.00001, math.cos(math.radians(center_y))) if is_geographic else 1.0
        )

        # Calculate true proportions so force_square makes a real square:
        adjusted_width = width * cos_lat if is_geographic else width
        max_dim = max(adjusted_width, height)

        buffer_str = str(buffer).strip().lower()

        # Determine the base buffer distance (y-axis baseline):

        # Buffer ratio:
        if buffer_str.endswith('%'):
            ratio = float(buffer_str.strip('%')) / 100.0
            if max_dim == 0:
                # If it's a Point, fallback to a nominal buffer instead of
                # interrupting code flow:
                base_buffer = 10.0 / (
                    LATITUDE_SPACING_KM * 1000 if is_geographic else meters_per_crs_unit
                )
            else:
                base_buffer = max_dim * ratio

        # Real-world units (e.g., "50 ft"):
        elif match := re.match(r'^([\d\.]+)\s*([a-z]+)$', buffer_str):
            val, unit = float(match.group(1)), match.group(2)

            # Check to make sure the user entered covered units:
            standard_unit = UNIT_ALIASES.get(unit)
            if standard_unit is None or standard_unit not in METERS_CONVERSION_FACTORS:
                raise ValueError(f"Unsupported distance unit '{unit}'.")

            distance_in_meters = val / METERS_CONVERSION_FACTORS[standard_unit]

            # Convert meters to native CRS y-axis units:
            if is_geographic:
                base_buffer = distance_in_meters / (LATITUDE_SPACING_KM * 1000)
            else:
                # Divide by the real CRS conversion factor:
                base_buffer = distance_in_meters / meters_per_crs_unit

        # Buffer in native CRS units:
        elif re.match(r'^[\d\.]+$', buffer_str):
            base_buffer = float(buffer_str)
        else:
            raise ValueError(f"Could not parse buffer: '{buffer}'")

        # Assign correctly scaled buffers:
        buffer_y = base_buffer
        buffer_x = base_buffer / cos_lat  # Stretch X buffer if geographic

        # Apply the buffers:
        if force_square:
            # Use max_dim (which represents the longest true side) to build
            # the square. Here X is unscaled by cos_lat so the bounding box
            # perfectly wraps the real-world distance:
            half_side_y = (max_dim / 2.0) + buffer_y
            half_side_x = (max_dim / 2.0 / cos_lat) + buffer_x

            return (
                center_x - half_side_x,
                center_y - half_side_y,
                center_x + half_side_x,
                center_y + half_side_y,
            )
        else:
            return (minx - buffer_x, miny - buffer_y, maxx + buffer_x, maxy + buffer_y)

    @staticmethod
    def _read_image_window(
        dataset: DatasetReader, bounds: tuple[float, float, float, float]
    ) -> np.ndarray:
        """
        Read a pixel window from the dataset safely.

        Intersects the requested geographic bounds with the raster's physical
        dimensions to prevent out-of-bounds indexing errors. If the requested
        bounds do not overlap the raster at all, an empty array is returned
        instead of raising.

        Args:
            dataset: An open rasterio dataset.
            bounds: Requested bounds ``(minx, miny, maxx, maxy)`` in the
                dataset's native CRS.

        Returns:
            np.ndarray: Pixel data shaped ``(bands, rows, cols)``. The array is
            empty (``size == 0``) when the bounds fall outside the raster.

        Example:
            >>> import rasterio
            >>> with rasterio.open('path/to/map.tif') as src:
            ...     arr = OrthomosaicReader._read_image_window(src, src.bounds)
            >>> arr.shape
            (3, 64, 64)
        """
        return OrthomosaicReader._read_patch(dataset, bounds, boundless=False)[0]

    @staticmethod
    def _read_patch(
        dataset: Any,
        bounds: tuple[float, float, float, float],
        boundless: bool = False,
    ) -> tuple[np.ndarray, Window]:
        """
        Read a pixel window and return it with the window actually read.

        Args:
            dataset: An open rasterio dataset.
            bounds: Requested bounds in the dataset's native CRS.
            boundless: Keep the full requested window and fill the parts
                outside the raster with the nodata value (0 when the dataset
                has none) instead of clipping.

        Returns:
            tuple[np.ndarray, Window]: The pixel data ``(bands, rows, cols)``
            and the window it covers. The array is empty when the bounds miss
            the raster entirely.

        Example:
            >>> import rasterio
            >>> with rasterio.open('path/to/map.tif') as src:
            ...     arr, window = OrthomosaicReader._read_patch(src, src.bounds)
            >>> (window.col_off, window.row_off)
            (0, 0)
        """
        window = from_bounds(*bounds, transform=dataset.transform)
        raster_window = Window(0, 0, dataset.width, dataset.height)
        empty = np.empty((dataset.count, 0, 0), dtype=dataset.dtypes[0])
        if boundless:
            full_window = window.round_offsets().round_lengths()
            try:
                window.intersection(raster_window)
            except WindowError:
                return empty, full_window
            fill = dataset.nodata if dataset.nodata is not None else 0
            array = dataset.read(window=full_window, boundless=True, fill_value=fill)
            return array, full_window
        try:
            safe_window = window.intersection(raster_window).round_offsets()
        except WindowError:
            # Requested bounds lie entirely outside the raster:
            return empty, window
        return dataset.read(window=safe_window), safe_window

    @staticmethod
    def _footprint_coverage(
        image_array: np.ndarray, geom: BaseGeometry, window: Window, dataset: Any
    ) -> float:
        """
        Fraction of an asset footprint covered by valid (non-nodata) pixels.

        Args:
            image_array: Either a 2-D boolean validity mask (``True`` where
                pixels hold data, e.g. from ``dataset.read_masks``) or the
                patch pixels ``(bands, rows, cols)``, in which case pixels
                equal to the nodata value count as missing.
            geom: Asset geometry in the dataset's native CRS.
            window: The window ``image_array`` was read from.
            dataset: The open rasterio dataset (for nodata and transform).

        Returns:
            float: Covered fraction in ``[0, 1]``. Geometries without area
            (points, lines) fall back to the whole patch's valid fraction.

        Example:
            >>> coverage = OrthomosaicReader._footprint_coverage(arr, geom, win, src)
            >>> 0.0 <= coverage <= 1.0
            True
        """
        if image_array.size == 0:
            return 0.0
        if image_array.ndim == 2:
            valid = image_array.astype(bool)
        else:
            nodata = dataset.nodata if dataset.nodata is not None else 0
            valid = np.any(image_array != nodata, axis=0)
        if geom.geom_type in ('Polygon', 'MultiPolygon') and not geom.is_empty:
            footprint = rasterio.features.geometry_mask(
                [geom],
                out_shape=valid.shape,
                transform=rasterio.windows.transform(window, dataset.transform),
                invert=True,
            )
            if footprint.any():
                return float(valid[footprint].mean())
        return float(valid.mean())

    @staticmethod
    def _format_array_for_pil(
        image_array: np.ndarray,
        global_min: float | None = None,
        global_max: float | None = None,
    ) -> np.ndarray:
        """
        Convert a raw Rasterio numpy array into a Pillow-compatible 8-bit RGB.

        Handles down-sampling of high-bit-depth images (e.g., 16-bit or float)
        to 8-bit. Applies radiometric scaling to ensure visual consistency
        across different image patches, prioritizing global statistics if
        provided.

        Args:
            image_array: The raw numpy array extracted from the raster (C, H, W).
            global_min: The minimum pixel value across the entire dataset.
            global_max: The maximum pixel value across the entire dataset.

        Returns:
            np.ndarray: An 8-bit RGB array formatted as (Height, Width, 3).
            Inputs with more than three bands are truncated to the first
            three; single-band inputs are replicated to three channels.

        Example:
            >>> import numpy as np
            >>> raw = np.full((4, 8, 8), 4096, dtype=np.uint16)
            >>> rgb = OrthomosaicReader._format_array_for_pil(raw)
            >>> rgb.shape, rgb.dtype, int(rgb[0, 0, 0])
            ((8, 8, 3), dtype('uint8'), 16)
        """
        # Truncate to the first 3 bands (discarding Alpha, Near-Infrared, etc.)
        # to ensure standard RGB and convert to float32 to prevent integer
        # overflow during scaling:
        img = image_array[:3].astype(np.float32)

        # Single-band (grayscale) rasters are replicated across three channels
        # so Pillow always receives a valid (H, W, 3) RGB array:
        if img.shape[0] < 3:
            img = np.repeat(img[:1], 3, axis=0)

        # If the image is already 8-bit, skip scaling, otherwise, normalize:
        if image_array.dtype != np.uint8:
            # --- Radiometric Scaling Hierarchy ---

            # Condition 1: Scale the patch relative to the entire map's
            # light/dark extremes:
            if (
                global_min is not None
                and global_max is not None
                and global_max > global_min
            ):
                img = np.clip(
                    (img - global_min) / (global_max - global_min) * 255, 0, 255
                )

            # Condition 2: Fast 16-bit linear scaling
            # Assuming standard 16-bit data (0-65535), divide by 256 to map
            # to 0-255:
            elif image_array.dtype == np.uint16:
                img = np.clip(img / 256, 0, 255)

            # Condition 3: normalized float scaling
            # Common in pre-processed ML datasets where pixel values are
            # already squeezed between 0.0 and 1.0:
            elif image_array.dtype in (np.float32, np.float64) and img.max() <= 1.0:
                img = np.clip(img * 255, 0, 255)

            # Condition 4: Local normalization (fallback)
            # Stretch the contrast based ONLY on the min/max of this specific
            # patch:
            else:
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = np.clip(((img - img_min) / (img_max - img_min) * 255), 0, 255)

        # Pillow requires exactly 8-bit unsigned integers:
        img = img.astype(np.uint8)

        # Rearrange the image channels to (height, width, channels) [HWC]:
        return np.moveaxis(img, 0, -1)

    @staticmethod
    def _is_image_valid(
        image_array: np.ndarray, max_missing_ratio: float, nodata_val: Any
    ) -> bool:
        """
        Determine if the extracted array contains enough valid data.

        Defaults to checking for ``0`` if the raster lacks an explicit
        nodata value.

        Args:
            image_array: Raw pixel array shaped ``(bands, rows, cols)``.
            max_missing_ratio: Maximum tolerated fraction (0.0 to 1.0) of
                nodata pixels.
            nodata_val: The dataset's nodata value, or ``None``.

        Returns:
            bool: ``True`` if the missing-data ratio is strictly below the
            threshold, ``False`` for empty arrays or too much missing data.

        Example:
            >>> import numpy as np
            >>> arr = np.ones((3, 4, 4), dtype=np.uint8)
            >>> arr[:, :, :2] = 0
            >>> OrthomosaicReader._is_image_valid(arr, 0.6, None)
            True
            >>> OrthomosaicReader._is_image_valid(arr, 0.5, None)
            False
        """
        if image_array.size == 0:
            return False

        target_val = nodata_val if nodata_val is not None else 0
        zero_pixels = np.count_nonzero(image_array == target_val)
        zero_ratio = zero_pixels / image_array.size

        return bool(zero_ratio < max_missing_ratio)

    @staticmethod
    def _get_local_pixel_coords(
        dataset: DatasetReader,
        geom_coords: list[tuple[float, float]],
        buffered_bounds: tuple[float, float, float, float],
        boundless: bool = False,
    ) -> list[tuple[int, int]]:
        """
        Translate native CRS coordinates into local patch pixel coordinates.

        Works for Points, Lines, and Polygons. The offset is derived from the
        same rounded, intersected window used by :meth:`_read_image_window`
        so that the returned coordinates line up with the extracted patch.

        Args:
            dataset: An open rasterio dataset.
            geom_coords: Coordinates ``(x, y)`` in the dataset's native CRS.
            buffered_bounds: The buffered bounds that defined the patch.

        Returns:
            list[tuple[int, int]]: ``(col, row)`` pixel coordinates relative
            to the top-left corner of the extracted patch.

        Example:
            >>> import rasterio
            >>> with rasterio.open('path/to/map.tif') as src:
            ...     minx, miny, maxx, maxy = src.bounds
            ...     coords = OrthomosaicReader._get_local_pixel_coords(
            ...         src, [(minx, maxy)], src.bounds
            ...     )
            >>> coords
            [(0, 0)]
        """
        pixel_coords_global = [dataset.index(x, y) for x, y in geom_coords]

        # Calculate offset using the raw un-intersected bounds to keep local
        # geometry intact:
        window = from_bounds(*buffered_bounds, transform=dataset.transform)
        if boundless:
            safe_window = window.round_offsets().round_lengths()
        else:
            raster_window = Window(0, 0, dataset.width, dataset.height)
            # Must round offsets exactly as we did in _read_image_window
            safe_window = window.intersection(raster_window).round_offsets()

        return [
            (int(col - safe_window.col_off), int(row - safe_window.row_off))
            for row, col in pixel_coords_global
        ]
