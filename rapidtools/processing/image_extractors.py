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
# 09-29-2026

"""
Imagery extraction components for ``rapidtools`` pipelines.

The classes in this module gather raw imagery for a
:class:`~rapidtools.core.PhysicalAssetCollection` from different sources
and attach the results as :class:`~rapidtools.core.ImageAsset` objects:

    - :class:`AerialImageryExtractor` crops per-asset patches out of one or
      more local orthomosaic GeoTIFFs.
    - :class:`MapillaryImageExtractor` finds the best unobstructed street-level
      panoramas of each asset via the Mapillary API and crops them to the
      asset.
    - :class:`BingOrthomosaicExtractor` stitches Bing Maps aerial tiles for a
      region into a single georeferenced GeoTIFF that can then be fed to
      :class:`AerialImageryExtractor` or
      :class:`~rapidtools.processing.SAM3OrthoFeatureExtractor`.

Example:
    >>> from rapidtools.core import PhysicalAssetCollection
    >>> from rapidtools.processing import AerialImageryExtractor, Pipeline
    >>> collection = PhysicalAssetCollection.from_geojson('buildings.geojson')
    >>> pipeline = Pipeline([
    ...     AerialImageryExtractor('ortho.tif', save_directory='output/crops')
    ... ])
    >>> collection = pipeline.run(collection)
    >>> collection[0].image_assets[0].path.suffix
    '.jpg'
"""

import concurrent.futures
import logging
import math
import threading
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.transform import from_bounds
from tqdm import tqdm

from rapidtools.config import REQUESTS_TIMEOUT_VAL, get_configured_session
from rapidtools.constants import LATITUDE_SPACING_KM
from rapidtools.core import (
    BoundingBox,
    ImageAsset,
    OperationCancelled,
    PhysicalAsset,
    PhysicalAssetCollection,
    PolygonRegion,
    is_cancelled,
    raise_if_cancelled,
)
from rapidtools.data_sources import (
    GoogleStreetViewClient,
    MapillaryClient,
    OrthomosaicReader,
    StreetViewPanorama,
)
from rapidtools.data_sources.google_aerial_image_extractor import (
    GOOGLE_TILE_LAYERS,
    GOOGLE_TILE_URL,
)
from rapidtools.data_sources.google_streetview import haversine_m
from rapidtools.data_sources.tile_imagery_base import MAX_MERCATOR_LAT

from . import outlines
from .pano_utils import (
    build_footprint_index,
    crop_panorama_to_asset,
    find_best_panos,
    index_panos,
)
from .step import Stage

logger = logging.getLogger(__name__)

# Accepted values for ``AerialImageryExtractor(outline_shape=...)``:
OUTLINE_SHAPES = outlines.OUTLINE_SHAPES


class AerialImageryExtractor:
    """
    Component that extracts aerial imagery patches for a collection of assets.

    This extractor reads from one or more high-resolution orthomosaic TIFF
    files, crops the imagery around each asset's bounding box, saves the
    patches to disk, and automatically attaches the new digital media to the
    corresponding ``PhysicalAsset`` objects in the collection.

    Each saved ``ImageAsset`` gets the ID ``'<asset_id>_<image_prefix>'`` and
    three properties: ``'wgs84_bounds'`` (``(min_lon, min_lat, max_lon,
    max_lat)`` of the patch), ``'native_georef'`` (the patch's affine
    transform and CRS, see
    :class:`~rapidtools.data_sources.orthomosaic_reader.PatchGeoref`) and
    ``'source_raster'`` (the TIFF file name).

    Args:
        dataset (str | Path | list[str | Path]):
            The file path to a single local orthomosaic TIFF dataset, a
            directory containing TIFFs (searched recursively), or a list of
            TIFF file paths.
        save_directory (str | Path):
            The directory where the extracted JPEG images will be saved.
            Will be created if it does not exist.
        max_missing_data_ratio (float, optional):
            The maximum acceptable proportion of nodata pixels (0.0 to 1.0).
            Crops exceeding it are skipped, which drops assets on the edge of
            the imagery unless ``min_footprint_coverage`` is used.
        min_footprint_coverage (float | None, optional):
            Keep an asset as long as at least this fraction of its footprint
            is covered by valid imagery (e.g. ``0.3``), instead of the
            patch-level nodata test. Assets partially outside the raster are
            then cropped with nodata padding so every crop keeps its size.
            Defaults to ``None`` (previous behaviour).
        pad_edges (bool | None, optional):
            Pad edge crops with nodata to their full size. Defaults to
            ``True`` when ``min_footprint_coverage`` is set.
            If a cropped patch exceeds this, it is discarded. Defaults to 0.2.
        overlay_asset_outline (bool, optional):
            If ``True``, draws an outline of the asset directly onto the saved
            image so a downstream model knows which asset to look at. With
            the default ``outline_*`` settings this is a thick red trace of
            the asset's geometry (a red dot for point assets). Defaults to
            ``False``.
        outline_shape (str, optional):
            What the outline traces. ``'geometry'`` follows the asset's own
            geometry, ``'bbox'`` its axis-aligned bounding box,
            ``'rotated_bbox'`` its minimum rotated rectangle,
            ``'convex_hull'`` its convex hull, and ``'corners'`` draws
            L-shaped brackets at the corners of the rotated rectangle so the
            asset's edges stay unobstructed. Defaults to ``'geometry'``.
        outline_buffer (float | str, optional):
            How far to expand the outline shape away from the asset before
            drawing it. A bare number or ``'12 px'`` is in pixels, a
            percentage (``'10%'``) is relative to the asset's longest pixel
            extent, and a real-world distance (``'2 m'``, ``'5 ft'``) is
            converted with the crop's ground sampling distance. Point assets
            with a buffer are drawn as a ring of that radius instead of a
            dot. Defaults to ``0``.
        outline_width (int | str, optional):
            Stroke width in pixels, or a percentage of the shorter image side
            (``'1%'``) so outlines look alike across crop sizes. Defaults to
            ``6``.
        outline_color (str | tuple[int, int, int], optional):
            Any PIL colour name, hex string or RGB tuple. Defaults to
            ``'red'``.
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
        cancel_event (threading.Event | None, optional):
            Cooperative cancellation flag checked before every asset. When
            set, :class:`~rapidtools.core.OperationCancelled` is raised.
            Defaults to ``None``.

    Example:
        >>> from rapidtools.core import PhysicalAssetCollection
        >>> from rapidtools.processing import AerialImageryExtractor
        >>>
        >>> collection = PhysicalAssetCollection.from_geojson('bldgs.geojson')
        >>> extractor = AerialImageryExtractor(
        ...     dataset='data/rasters/',
        ...     save_directory='output/aerial_crops',
        ...     overlay_asset_outline=True,
        ...     outline_shape='rotated_bbox',
        ...     outline_buffer='2 m',
        ...     outline_width='1%',
        ...     image_prefix='post_disaster',
        ...     keep_multiple_copies=True,
        ...     buffer_asset='20 ft',
        ...     force_square_image=True
        ... )
        >>> processed_assets = extractor(collection)
        >>> processed_assets[0].image_assets[0].id
        'bldg_001_post_disaster'
    """

    stage = Stage.EXTRACT_IMAGERY

    def __init__(
        self,
        dataset: str | Path | list[str | Path],
        save_directory: str | Path,
        max_missing_data_ratio: float = 0.2,
        overlay_asset_outline: bool = False,
        image_prefix: str = 'aerial',
        keep_multiple_copies: bool = False,
        buffer_asset: float | str = '20%',
        buffer_m: float | None = None,
        force_square_image: bool = True,
        cancel_event: threading.Event | None = None,
        min_footprint_coverage: float | None = None,
        pad_edges: bool | None = None,
        outline_shape: str = 'geometry',
        outline_buffer: float | str = 0,
        outline_width: int | float | str = 6,
        outline_color: str | tuple[int, int, int] = 'red',
    ):
        """
        Initialize the extractor configuration.

        Args:
            dataset (str | Path | list[str | Path]):
                A TIFF path, a directory of TIFFs, or a list of TIFF paths.
            save_directory (str | Path):
                Output directory for the cropped JPEGs.
            max_missing_data_ratio (float, optional):
                Maximum tolerated nodata fraction per patch.
            min_footprint_coverage (float | None, optional):
                Minimum imaged fraction of each footprint; replaces the
                patch-level nodata test and pads edge crops.
            pad_edges (bool | None, optional):
                Pad edge crops with nodata to their full size.
            overlay_asset_outline (bool, optional):
                Draw the asset outline onto the saved image.
            outline_shape (str, optional):
                One of :data:`OUTLINE_SHAPES`; see the class docstring.
            outline_buffer (float | str, optional):
                Expansion of the outline shape in pixels, percent or a
                real-world distance such as ``'2 m'``.
            outline_width (int | float | str, optional):
                Stroke width in pixels or a percentage of the shorter side.
            outline_color (str | tuple[int, int, int], optional):
                PIL colour for the outline.
            image_prefix (str, optional):
                Prefix for filenames and ``ImageAsset`` IDs.
            keep_multiple_copies (bool, optional):
                Append a counter instead of overwriting existing files.
            buffer_m (float | None, optional):
                Buffer around each asset in metres; overrides ``buffer_asset``
                when given.
            buffer_asset (float | str, optional):
                Padding around the asset geometry before cropping.
            force_square_image (bool, optional):
                Force a square real-world footprint for every patch.
            cancel_event (threading.Event | None, optional):
                Cooperative cancellation flag checked between assets.
        """
        # Cooperative cancellation (stops between assets):
        self.cancel_event = cancel_event
        self.dataset = dataset
        self.save_directory = Path(save_directory).resolve()
        self.max_missing_data_ratio = max_missing_data_ratio
        if min_footprint_coverage is not None and not (
            0.0 <= min_footprint_coverage <= 1.0
        ):
            raise ValueError('min_footprint_coverage must be between 0 and 1.')
        self.min_footprint_coverage = min_footprint_coverage
        self.pad_edges = pad_edges
        self.overlay_asset_outline = overlay_asset_outline
        if outline_shape not in OUTLINE_SHAPES:
            raise ValueError(
                f'outline_shape must be one of {OUTLINE_SHAPES}, got {outline_shape!r}.'
            )
        self.outline_shape = outline_shape
        # Parse eagerly so bad specs fail at construction, not mid-extraction:
        self._outline_buffer_spec = self._parse_outline_buffer(outline_buffer)
        self._outline_width_spec = self._parse_outline_width(outline_width)
        self.outline_buffer = outline_buffer
        self.outline_width = outline_width
        outlines.validate_outline_color(outline_color)
        self.outline_color = outline_color
        self.image_prefix = image_prefix
        self.keep_multiple_copies = keep_multiple_copies
        # ``buffer_m`` is the harmonized metric spelling of ``buffer_asset``:
        self.buffer_asset = f'{buffer_m} m' if buffer_m is not None else buffer_asset
        self.force_square_image = force_square_image

    def _get_raster_paths(self) -> list[Path]:
        """
        Resolve all raster files from the ``dataset`` input.

        Returns:
            list[Path]:
                Absolute, de-duplicated paths. For a directory input, all
                ``*.tif`` and ``*.tiff`` files found recursively are returned.
        """
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

    # ------------------------------------------------------------ outlines
    # Thin wrappers around :mod:`rapidtools.processing.outlines`, kept so the
    # outline options behave identically for every imagery extractor.
    _parse_outline_buffer = staticmethod(outlines.parse_outline_buffer)
    _parse_outline_width = staticmethod(outlines.parse_outline_width)
    _outline_geometry = staticmethod(outlines.outline_geometry)
    _draw_geometry = staticmethod(outlines.draw_geometry)
    _draw_corner_brackets = staticmethod(outlines.draw_corner_brackets)

    @staticmethod
    def _meters_per_pixel(
        wgs84_bounds: tuple[float, float, float, float], image_size: tuple[int, int]
    ) -> float:
        """
        Estimate the ground sampling distance of a crop from its WGS84 bounds.

        Args:
            wgs84_bounds: ``(min_lon, min_lat, max_lon, max_lat)`` of the crop.
            image_size: ``(width, height)`` of the crop in pixels.

        Returns:
            float: Metres per pixel along the north-south axis, or ``0.0`` if
            the crop has no height.

        Example:
            >>> mpp = AerialImageryExtractor._meters_per_pixel(
            ...     (0, 0, 0.001, 0.001), (111, 111)
            ... )
            >>> round(mpp, 2)
            1.0
        """
        _, min_lat, _, max_lat = wgs84_bounds
        height_px = image_size[1]
        if height_px <= 0:
            return 0.0
        return (max_lat - min_lat) * LATITUDE_SPACING_KM * 1000.0 / height_px

    def _outline_buffer_px(self, extent_px: float, meters_per_pixel: float) -> float:
        """Resolve the configured outline buffer to pixels for one crop."""
        return outlines.resolve_buffer_px(
            self._outline_buffer_spec, extent_px, meters_per_pixel
        )

    def _outline_width_px(self, image_size: tuple[int, int]) -> int:
        """Resolve the configured stroke width to whole pixels for one crop."""
        return outlines.resolve_width_px(self._outline_width_spec, image_size)

    def _draw_outline(
        self,
        image: Image.Image,
        pixel_coords: list[tuple[float, float]],
        is_closed: bool,
        wgs84_bounds: tuple[float, float, float, float],
    ) -> None:
        """
        Draw the configured asset outline onto a crop, in place.

        Args:
            image: The crop to annotate.
            pixel_coords: The asset's vertices in crop pixel coordinates.
            is_closed: Whether the vertices describe a polygon ring.
            wgs84_bounds: The crop's WGS84 bounds, used to convert real-world
                buffer distances to pixels.
        """
        outlines.draw_outline(
            image,
            pixel_coords,
            is_closed,
            shape=self.outline_shape,
            buffer=self._outline_buffer_spec,
            width=self._outline_width_spec,
            color=self.outline_color,
            meters_per_pixel=self._meters_per_pixel(wgs84_bounds, image.size),
        )

    def __call__(
        self, asset_collection: PhysicalAssetCollection
    ) -> PhysicalAssetCollection:
        """
        Execute the extraction process on the provided asset collection.

        This method allows the class instance to be called like a function,
        making it directly compatible with the ``rapidtools`` ``Pipeline``
        engine. For every raster, only the assets intersecting that raster's
        extent are considered; multi-part geometries are wrapped in their
        convex hull before cropping. Rasters that fail to open are logged and
        skipped.

        Args:
            asset_collection (PhysicalAssetCollection):
                The collection of physical assets to extract imagery for.

        Returns:
            PhysicalAssetCollection:
                The mutated collection, with new ``ImageAsset`` objects
                attached to their corresponding ``PhysicalAsset`` entities.

        Raises:
            OperationCancelled:
                If ``cancel_event`` is set while iterating over assets.

        Example:
            >>> extractor = AerialImageryExtractor('ortho.tif', 'crops')
            >>> collection = extractor(collection)
            >>> collection['bldg_01'].image_assets[0].properties['source_raster']
            'ortho.tif'
        """
        # Create save directory if it does not exist:
        self.save_directory.mkdir(parents=True, exist_ok=True)

        # Gather all raster file paths from the user's input:
        raster_paths = self._get_raster_paths()
        if not raster_paths:
            logger.warning(f'No raster files found for dataset input: {self.dataset}')
            return asset_collection

        logger.info(
            f'Found {len(raster_paths)} raster(s). Images will be saved to: '
            f'{self.save_directory}'
        )

        total_extracted_count = 0

        # Process each raster file found:
        for raster_path in raster_paths:
            logger.info(f'Processing raster: {raster_path.name}')

            # Open the TIFF file ONCE for the current raster using the context
            # manager:
            try:
                with OrthomosaicReader(raster_path) as reader:
                    # Efficiently filter assets to only those inside the
                    # raster's extent:
                    min_lon, min_lat, max_lon, max_lat = reader.dataset_extent
                    raster_bbox = BoundingBox(
                        min_x=min_lon, min_y=min_lat, max_x=max_lon, max_y=max_lat
                    )

                    assets_in_bounds = asset_collection.filter_by_geometry(raster_bbox)

                    if not assets_in_bounds:
                        logger.info(
                            'No assets fall within the bounds of '
                            f'{raster_path.name}. Skipping.'
                        )
                        continue

                    logger.info(
                        f'Found {len(assets_in_bounds)}/{len(asset_collection)} '
                        "assets intersecting the raster's bounding box "
                        '(imagery coverage is checked per asset below).'
                    )
                    skipped_no_imagery = 0

                    # Loop only over the filtered subset for this specific
                    # raster:
                    for asset in tqdm(
                        assets_in_bounds, desc=f'Extracting from {raster_path.name}'
                    ):
                        raise_if_cancelled(self.cancel_event, 'aerial image extraction')

                        # Get the asset geometry:
                        geom = asset.geometry

                        if not geom or geom.is_empty:
                            continue

                        # Take the outside part for multi-part geometries
                        # (convex_hull creates a single polygon wrapping all
                        # parts of the asset):
                        if (
                            geom.geom_type.startswith('Multi')
                            or geom.geom_type == 'GeometryCollection'
                        ):
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
                            asset_coords = list(unified_geom.envelope.exterior.coords)
                            is_closed = True

                        # Extract image patch using the open reader:
                        extraction_result = reader.get_image_patch(
                            asset_geometry=asset_coords,
                            max_missing_data_ratio=self.max_missing_data_ratio,
                            buffer=self.buffer_asset,
                            force_square=self.force_square_image,
                            min_footprint_coverage=self.min_footprint_coverage,
                            pad_edges=self.pad_edges,
                            return_georef=True,
                        )

                        if extraction_result is None:
                            skipped_no_imagery += 1
                            continue

                        pil_image, pixel_coords, wgs84_bounds, georef = (
                            extraction_result
                        )

                        # Draw outline if requested:
                        if self.overlay_asset_outline:
                            self._draw_outline(
                                pil_image, pixel_coords, is_closed, wgs84_bounds
                            )

                        # Create a clean, coordinate-based filename:
                        centroid = unified_geom.centroid
                        coords_str = f'{centroid.y:.8f}_{centroid.x:.8f}'.replace(
                            '.', ''
                        )

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
                                'native_georef': georef.to_dict(),
                                'source_raster': raster_path.name,
                            },
                        )

                        asset.add_image_assets(img_asset)
                        total_extracted_count += 1

                    if skipped_no_imagery:
                        logger.info(
                            f'{skipped_no_imagery} of {len(assets_in_bounds)} '
                            f'candidate assets had no usable imagery in '
                            f'{raster_path.name} '
                            '(outside the data footprint or too much nodata).'
                        )
            except OperationCancelled:
                # Cancellation must propagate to the caller, never be swallowed
                # by the per-raster error handler below:
                raise
            except Exception as e:
                logger.error(f'Failed to process raster {raster_path.name}: {e}')

        logger.info(
            f'Finished processing all rasters. Extracted aerial imagery for a '
            f'total of {total_extracted_count} assets.'
        )

        return asset_collection


class MapillaryImageExtractor:
    """
    Street-view extractor that fetches unobstructed Mapillary panoramas of assets.

    This extractor acts as an intelligent pipeline component. Instead of
    making thousands of API calls, it fetches regional metadata once, builds
    spatial indexes (KDTree and STRtree), and uses ray-casting to simulate
    lines of sight. It actively culls occluded views (e.g., if a neighboring
    building blocks the camera) and downloads only the valid panoramas, one
    per viewing direction (major/minor building axes and, optionally, the
    corners).

    If ``strict_content_filter`` is enabled, the extractor checks the semantic
    contents of the image before downloading the heavy JPEG. If the target
    asset type is not actually visible in the image, it skips it to save
    bandwidth.

    Args:
        access_token (str):
            Mapillary API Access Token.
        save_directory (str | Path):
            The directory where the final cropped JPEG images will be saved.
        start_date (str, optional):
            Inclusive lower bound for the image capture date (YYYY-MM-DD).
            Defaults to ``''`` (no lower bound).
        end_date (str, optional):
            Inclusive upper bound for the image capture date (YYYY-MM-DD).
            Defaults to ``''`` (no upper bound).
        filter_rapid_only (bool, optional):
            If ``True``, strictly fetches images uploaded by the RAPID
            organization. Set to ``False`` to search all public street-view
            images. Defaults to ``True``.
        cast_corner_rays (bool, optional):
            If ``False``, limits extraction to the 4 principal axes (faces) of
            the building. If ``True``, also extracts views pointing directly at
            the 4 corners (up to 8 images). Defaults to ``False``.
        smart_crop (bool, optional):
            If ``True``, downloads the semantic mask for the panorama and
            intelligently crops out the sky and the data-collection vehicle.
            Defaults to ``True``.
        strict_content_filter (bool, optional):
            If ``True``, verifies that the target asset type is semantically
            present in the image before downloading the high-resolution JPEG.
            Defaults to ``False``.
        label_mapper (Any, optional):
            An instance of ``MapillaryLabelMapper`` that uses an LLM to
            dynamically map asset types to Mapillary labels. Used if
            ``strict_content_filter`` is ``True``. Defaults to ``None``.
        asset_type_mapping (dict[str, list[str]] | None, optional):
            A manual dictionary mapping your asset types to official Mapillary
            labels (e.g., ``{'building': ['construction--structure--building']}``).
            Used as a fallback or alternative to the ``label_mapper``.
            Defaults to ``None``.
        image_prefix (str, optional):
            Prefix applied to the saved image filenames. Defaults to
            ``'street'``.
        max_images_per_asset (int, optional):
            The maximum number of panorama crops to extract per asset.
            Panoramas aligned with the major axes are prioritized over minor
            axes and corners. Defaults to 8.
        max_workers (int, optional):
            The number of concurrent threads used to download and crop images.
            Defaults to 10.
        cancel_event (threading.Event | None, optional):
            Cooperative cancellation flag checked before each asset is
            processed. When set, :class:`~rapidtools.core.OperationCancelled`
            is raised from :meth:`__call__`. Defaults to ``None``.

    Example:
        >>> from rapidtools.processing import MapillaryImageExtractor
        >>>
        >>> # Option 1: Using manual dictionary mapping
        >>> extractor = MapillaryImageExtractor(
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
        >>> extractor_llm = MapillaryImageExtractor(
        ...     access_token='YOUR_TOKEN',
        ...     save_directory='output/streetview',
        ...     strict_content_filter=True,
        ...     label_mapper=mapper
        ... )
        >>>
        >>> processed_assets = extractor(my_asset_collection)
        >>> processed_assets['bldg_01'].image_assets[0].properties['view_angle']
        'major_axis_0'
    """

    stage = Stage.EXTRACT_IMAGERY

    def __init__(
        self,
        access_token: str | None = None,
        save_directory: str | Path = 'streetview',
        start_date: str = '',
        end_date: str = '',
        filter_rapid_only: bool = True,
        cast_corner_rays: bool = False,
        smart_crop: bool = True,
        strict_content_filter: bool = False,
        label_mapper: Any = None,
        asset_type_mapping: dict[str, list[str]] | None = None,
        image_prefix: str = 'street',
        max_images_per_asset: int = 8,
        max_workers: int = 10,
        cancel_event: threading.Event | None = None,
        api_key: str | None = None,
    ) -> None:
        """
        Initialize the Mapillary image extractor.

        Args:
            access_token (str):
                Mapillary API Access Token.
            save_directory (str | Path):
                The directory where the final cropped JPEG images will be saved.
            start_date (str, optional):
                Inclusive lower bound for the image capture date (YYYY-MM-DD).
            end_date (str, optional):
                Inclusive upper bound for the image capture date (YYYY-MM-DD).
            filter_rapid_only (bool, optional):
                Restrict the search to RAPID-uploaded imagery.
            cast_corner_rays (bool, optional):
                Also look for views pointing at the building corners.
            smart_crop (bool, optional):
                Use the semantic mask to crop out sky and vehicle.
            strict_content_filter (bool, optional):
                Skip panoramas that do not contain the asset type.
            label_mapper (Any, optional):
                LLM-backed ``MapillaryLabelMapper`` for unmapped asset types.
            asset_type_mapping (dict[str, list[str]] | None, optional):
                Manual asset type to Mapillary label mapping.
            image_prefix (str, optional):
                Prefix applied to the saved image filenames.
            max_images_per_asset (int, optional):
                Maximum number of crops per asset.
            max_workers (int, optional):
                Number of concurrent download threads.
            cancel_event (threading.Event | None, optional):
                Cooperative cancellation flag checked between assets.
            api_key (str | None, optional):
                Alias of ``access_token`` for consistency with other clients.

        Raises:
            ValueError:
                If neither ``access_token`` nor ``api_key`` is given.
        """
        if api_key is not None:
            access_token = api_key
        if not access_token:
            raise ValueError(
                'A Mapillary access token (access_token or api_key) is required.'
            )
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
        self.max_images_per_asset = max_images_per_asset
        self.max_workers = max_workers
        # Cooperative cancellation (stops between assets):
        self.cancel_event = cancel_event

        self.client = MapillaryClient(
            access_token=access_token,
            save_dir=self.save_directory,
        )

    def __call__(
        self, asset_collection: PhysicalAssetCollection
    ) -> PhysicalAssetCollection:
        """
        Execute the extraction process on the provided asset collection.

        The workflow is:

            1. (Optional) Resolve Mapillary labels for every asset type when
               ``strict_content_filter`` is enabled.
            2. Fetch metadata for all panoramas inside the collection's
               combined bounding box in a single regional query.
            3. Build spatial indexes over panoramas and footprints.
            4. For each asset (in parallel), pick the best panorama per
               viewing direction, download, crop and attach it.

        Args:
            asset_collection (PhysicalAssetCollection):
                The collection of physical assets to extract street-view
                imagery for.

        Returns:
            PhysicalAssetCollection:
                The mutated collection, with new ``ImageAsset`` objects
                representing the cropped panoramas attached to their
                corresponding ``PhysicalAsset`` entities. Each new image has
                the properties ``'view_angle'`` and ``'original_pano_id'``.

        Raises:
            OperationCancelled:
                If ``cancel_event`` is set while assets are being processed.

        Example:
            >>> extractor = MapillaryImageExtractor('TOKEN', 'output/street')
            >>> collection = extractor(collection)
        """
        self.save_directory.mkdir(parents=True, exist_ok=True)

        # 1. Resolve Target Labels for Strict Content Filtering
        if self.strict_content_filter:
            unique_types = {a.asset_type for a in asset_collection if a.asset_type}
            unmapped_types = [
                t for t in unique_types if t not in self.asset_type_mapping
            ]

            if unmapped_types and self.label_mapper:
                logger.info(
                    f'Using LLM to dynamically map asset types: {unmapped_types}'
                )
                for asset_type in unmapped_types:
                    mapped_labels = self.label_mapper.map_classes([asset_type])
                    self.asset_type_mapping[asset_type] = mapped_labels

            logger.info(f'Content filter mapping: {self.asset_type_mapping}')

        # 2. Fetch regional metadata
        collection_bbox = asset_collection.combined_bounding_box
        logger.info('Fetching regional Mapillary metadata...')

        regional_images = self.client.fetch_images_in_bbox(
            bbox=collection_bbox,
            save_to_disk=False,
            start_date=self.start_date,
            end_date=self.end_date,
            filter_rapid_only=self.filter_rapid_only,
        )

        if len(regional_images) == 0:
            logger.warning('No Mapillary images found in the collection region.')
            return asset_collection

        logger.info('Building KDTree for panos and STRtree for footprints...')
        tree, coords_deg, headings = index_panos(regional_images)
        building_tree, building_geoms = build_footprint_index(asset_collection)

        # Threaded processing function for a single asset
        def process_asset(asset) -> int:
            """Fetch and attach panoramas for one asset; return the count."""
            raise_if_cancelled(self.cancel_event, 'Mapillary image extraction')

            if not asset.geometry:
                return 0

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

            # Sort valid panoramas to prioritize Major axes > Minor axes > Corners
            def priority_sort(item):
                """Rank view names: major axes, then minor axes, then corners."""
                name = item[0]
                if 'major' in name:
                    return 0
                if 'minor' in name:
                    return 1
                return 2

            valid_panos = [(k, v) for k, v in best_panos_map.items() if v is not None]
            valid_panos.sort(key=priority_sort)

            for axis_name, pano_compact in valid_panos:
                if extracted_count >= self.max_images_per_asset:
                    break

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
                    logger.error(
                        f'Failed to process pano {pano_compact.id} '
                        f'for asset {asset.id}: {e}'
                    )

            return extracted_count

        # Execute parallel downloads
        total_extracted = 0
        logger.info(f'Extracting panos using {self.max_workers} threads...')

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

        logger.info(
            f'Finished! Successfully extracted {total_extracted} multi-angle '
            'cropped panoramas.'
        )
        return asset_collection


class TileOrthomosaicExtractor:
    """
    Base component that stitches XYZ aerial tiles into a georeferenced GeoTIFF.

    Subclasses (:class:`BingOrthomosaicExtractor`,
    :class:`GoogleOrthomosaicExtractor`) only provide :meth:`_tile_url`. The
    resulting synthetic orthomosaic keeps EPSG:4326 coordinate metadata so it
    can be fed to region-wide feature extractors like
    ``SAM3OrthoFeatureExtractor`` exactly like a drone or satellite capture.

    Args:
        zoom_level (int, optional):
            Tile zoom level, typically 1 (whole world) to 19-21 (~0.3-0.15
            m/pixel). Defaults to 19.
        max_workers (int, optional):
            Number of concurrent download threads. Defaults to 10.
        cancel_event (threading.Event | None, optional):
            Cooperative cancellation flag checked as tiles complete. When set,
            outstanding downloads are cancelled and
            :class:`~rapidtools.core.OperationCancelled` is raised.

    Example:
        >>> from rapidtools.core import BoundingBox
        >>> from rapidtools.processing import GoogleOrthomosaicExtractor
        >>> region = BoundingBox(-118.251, 34.050, -118.245, 34.055)
        >>> path = GoogleOrthomosaicExtractor(zoom_level=19)(region, 'la.tiff')
        >>> path.suffix
        '.tiff'
    """

    PROVIDER_NAME: str = 'tile'

    def __init__(
        self,
        zoom_level: int = 19,
        max_workers: int = 10,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """
        Initialize the extractor configuration.

        Args:
            zoom_level (int, optional):
                Zoom level of the tiles to download.
            max_workers (int, optional):
                Number of concurrent download threads.
            cancel_event (threading.Event | None, optional):
                Cooperative cancellation flag checked between tiles.
        """
        self.zoom_level = zoom_level
        self.max_workers = max_workers
        # Cooperative cancellation (between tiles):
        self.cancel_event = cancel_event

    # --- MATH & COORDINATE UTILITIES ---
    @staticmethod
    def lat_lon_to_pixel(lat: float, lon: float, zoom: int) -> tuple[int, int]:
        """
        Convert latitude and longitude to global Web Mercator pixel coordinates.

        Args:
            lat (float):
                Latitude in decimal degrees (clamped to the Mercator range).
            lon (float):
                Longitude in decimal degrees.
            zoom (int):
                Tile zoom level. The world is ``256 << zoom`` pixels wide.

        Returns:
            tuple[int, int]:
                ``(pixel_x, pixel_y)`` in the global pixel grid, with the
                origin at the top-left corner of the map.

        Example:
            >>> TileOrthomosaicExtractor.lat_lon_to_pixel(0.0, 0.0, zoom=1)
            (256, 256)
        """
        lat = max(min(lat, MAX_MERCATOR_LAT), -MAX_MERCATOR_LAT)
        sin_lat = math.sin(math.radians(lat))
        map_size = 256 << zoom

        pixel_x = ((lon + 180) / 360) * map_size
        pixel_y = (
            0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)
        ) * map_size
        # lon=180 and the Mercator limit land on the map edge (index
        # map_size), which belongs to the last pixel:
        return (
            min(max(int(pixel_x), 0), map_size - 1),
            min(max(int(pixel_y), 0), map_size - 1),
        )

    def _tile_url(self, tile_x: int, tile_y: int) -> str:
        """
        Return the download URL of the tile at ``(tile_x, tile_y)``.

        Subclasses must override this for their provider.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError('Subclasses must implement _tile_url().')

    def __call__(
        self, region: BoundingBox | PolygonRegion, output_path: str | Path
    ) -> Path:
        """
        Download the tiles covering ``region`` and stitch them into a GeoTIFF.

        Tiles that fail to download or decode are logged and left black in
        the mosaic; the remaining tiles are still written.

        Args:
            region (BoundingBox | PolygonRegion):
                The geographic area to extract imagery for (WGS84).
            output_path (str | Path):
                The file path where the synthesized GeoTIFF should be saved.
                Missing parent directories will be created automatically and a
                ``.tiff`` suffix is enforced.

        Returns:
            Path:
                The resolved, absolute path to the successfully saved
                GeoTIFF file.

        Raises:
            ValueError:
                If the region collapses to zero pixels at ``zoom_level``.
            OperationCancelled:
                If ``cancel_event`` is set while tiles are downloading.

        Example:
            >>> from rapidtools.core import BoundingBox
            >>> extractor = GoogleOrthomosaicExtractor(zoom_level=18)
            >>> path = extractor(
            ...     BoundingBox(-118.15, 34.18, -118.14, 34.19), 'out/area'
            ... )
            >>> path.name
            'area.tiff'
        """
        output_path = Path(output_path).resolve()

        # 1. Get geographic bounds (Works for both BoundingBox and PolygonRegion)
        min_lon, min_lat, max_lon, max_lat = region.bounds

        # 2. Convert to Bing pixel coordinates to define the exact canvas size
        px_min_x, px_max_y = self.lat_lon_to_pixel(min_lat, min_lon, self.zoom_level)
        px_max_x, px_min_y = self.lat_lon_to_pixel(max_lat, max_lon, self.zoom_level)

        canvas_width = px_max_x - px_min_x
        canvas_height = px_max_y - px_min_y

        if canvas_width <= 0 or canvas_height <= 0:
            raise ValueError(
                f'Region {region.bounds} collapses to {canvas_width}x'
                f'{canvas_height} px at zoom {self.zoom_level}. Use a larger '
                'region or a higher zoom level.'
            )

        logger.info(
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

        tiles_to_download = [
            (tx, ty)
            for tx in range(tile_min_x, tile_max_x + 1)
            for ty in range(tile_min_y, tile_max_y + 1)
        ]

        # 4. Download and paste tiles in parallel
        # Use the standardized session with built-in retries and exponential
        # backoff
        with get_configured_session() as session:
            # Safely scale the connection pools for high-concurrency threading
            for prefix in ('http://', 'https://'):
                adapter = session.adapters[prefix]
                adapter.pool_connections = self.max_workers
                adapter.pool_maxsize = self.max_workers * 2

            def fetch_tile(
                tile_info: tuple[int, int],
            ) -> tuple[int, int, Image.Image | None]:
                """Download one tile, returning its grid position and image."""
                tx, ty = tile_info
                qk = f'({tx}, {ty}, {self.zoom_level})'
                url = self._tile_url(tx, ty)

                try:
                    # Use the standardized timeout from config.py
                    resp = session.get(url, timeout=REQUESTS_TIMEOUT_VAL)

                    # raise_for_status ensures we don't try to open a 404/Error
                    # page as an image
                    resp.raise_for_status()

                    return tx, ty, Image.open(BytesIO(resp.content))

                except requests.RequestException as req_err:
                    logger.warning(
                        f'Network error fetching tile {qk} after retries: {req_err}'
                    )
                except Exception as parse_err:
                    logger.warning(
                        f'Failed to parse image data for tile {qk}: {parse_err}'
                    )

                # Return None only if all retries failed or image is corrupted
                return tx, ty, None

            # Execute threaded downloads
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                futures = [executor.submit(fetch_tile, t) for t in tiles_to_download]
                for future in concurrent.futures.as_completed(futures):
                    if is_cancelled(self.cancel_event):
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise_if_cancelled(
                            self.cancel_event, f'{self.PROVIDER_NAME} tile download'
                        )
                    tx, ty, img = future.result()
                    if img is not None:
                        # Calculate exact pixel offset. This perfectly crops
                        # any extra tile overlap hanging outside the bounding
                        # box!
                        paste_x = (tx * 256) - px_min_x
                        paste_y = (ty * 256) - px_min_y
                        canvas.paste(img, (paste_x, paste_y))

        # 5. Save as a georeferenced GeoTIFF
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() not in ['.tif', '.tiff']:
            output_path = output_path.with_suffix('.tiff')

        # Convert PIL image to numpy array formatted for rasterio
        # (bands, height, width)
        img_data = np.array(canvas)
        img_data = np.moveaxis(img_data, 2, 0)

        # Calculate geospatial transform mapping pixels to WGS84
        transform = from_bounds(
            min_lon, min_lat, max_lon, max_lat, canvas_width, canvas_height
        )

        logger.info(f'Saving GeoTIFF to: {output_path}')
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=canvas_height,
            width=canvas_width,
            count=3,
            dtype=img_data.dtype,
            crs='EPSG:4326',
            transform=transform,
        ) as dst:
            dst.write(img_data)

        return output_path


class BingOrthomosaicExtractor(TileOrthomosaicExtractor):
    """
    Component that extracts and synthesizes aerial imagery from Bing Maps Tiles.

    Downloads Bing Maps aerial tiles for a given geographical region and
    stitches them into a single, continuous GeoTIFF. The resulting synthetic
    orthomosaic retains its coordinate metadata (EPSG:4326) and can be
    directly fed into region-wide feature extractors (like
    ``SAM3OrthoFeatureExtractor``) just like a standard drone flight or
    satellite capture.

    Args:
        zoom_level (int, optional):
            The detail level of the Bing Maps tiles to download, typically
            ranging from 1 (entire world) to 19 (0.3m/pixel).
            Defaults to 19.
        max_workers (int, optional):
            The number of concurrent threads to use for downloading tiles in
            parallel. Defaults to 10.
        cancel_event (threading.Event | None, optional):
            Cooperative cancellation flag checked as tiles complete. When set,
            outstanding downloads are cancelled and
            :class:`~rapidtools.core.OperationCancelled` is raised. Defaults
            to ``None``.

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
        >>> tiff_path.suffix
        '.tiff'
    """

    PROVIDER_NAME = 'Bing'

    @staticmethod
    def tile_to_quadkey(tile_x: int, tile_y: int, zoom: int) -> str:
        """
        Convert tile XY coordinates into a Bing Maps Quadkey string.

        Args:
            tile_x (int):
                Tile column index.
            tile_y (int):
                Tile row index.
            zoom (int):
                Zoom level (equals the length of the returned quadkey).

        Returns:
            str:
                The quadkey, a base-4 string such as ``'213'``.

        Example:
            >>> BingOrthomosaicExtractor.tile_to_quadkey(3, 5, zoom=3)
            '213'
        """
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

    def _tile_url(self, tile_x: int, tile_y: int) -> str:
        """Return the Bing aerial tile URL (quadkey addressed)."""
        quadkey = self.tile_to_quadkey(tile_x, tile_y, self.zoom_level)
        return f'http://ecn.t3.tiles.virtualearth.net/tiles/a{quadkey}.jpeg?g=1'


class GoogleOrthomosaicExtractor(TileOrthomosaicExtractor):
    """
    Component that stitches Google Maps satellite tiles into a GeoTIFF.

    Uses the keyless ``mt0``-``mt3.google.com`` tile servers (the same source
    as :class:`~rapidtools.data_sources.GoogleAerialImageExtractor`), so it is
    a drop-in alternative to :class:`BingOrthomosaicExtractor` when Bing's
    imagery of an area is outdated or missing.

    Args:
        zoom_level (int, optional):
            Tile zoom level, up to about 21 in urban areas. Defaults to 19.
        max_workers (int, optional):
            Number of concurrent download threads. Defaults to 10.
        layer (str, optional):
            ``'s'`` for satellite (default) or ``'y'`` for the hybrid layer
            with labels.
        cancel_event (threading.Event | None, optional):
            Cooperative cancellation flag checked as tiles complete.

    Example:
        >>> from rapidtools.core import BoundingBox
        >>> from rapidtools.processing import GoogleOrthomosaicExtractor
        >>> region = BoundingBox(-118.1355, 34.1870, -118.1340, 34.1885)
        >>> extractor = GoogleOrthomosaicExtractor(zoom_level=20)
        >>> tiff_path = extractor(region, output_path='altadena_google.tiff')
        >>> tiff_path.name
        'altadena_google.tiff'
    """

    PROVIDER_NAME = 'Google'

    def __init__(
        self,
        zoom_level: int = 19,
        max_workers: int = 10,
        layer: str = 's',
        cancel_event: threading.Event | None = None,
    ) -> None:
        """
        Initialize the extractor configuration.

        Args:
            zoom_level (int, optional): Zoom level of the tiles to download.
            max_workers (int, optional): Number of concurrent download threads.
            layer (str, optional): Google tile layer code (``'s'`` or ``'y'``).
            cancel_event (threading.Event | None, optional): Cooperative
                cancellation flag checked between tiles.

        Raises:
            ValueError: If ``layer`` is not a supported imagery layer.
        """
        super().__init__(
            zoom_level=zoom_level, max_workers=max_workers, cancel_event=cancel_event
        )
        if layer not in GOOGLE_TILE_LAYERS:
            raise ValueError(
                f'Unsupported Google tile layer {layer!r}; choose one of '
                f'{sorted(GOOGLE_TILE_LAYERS)}.'
            )
        self.layer = layer

    def _tile_url(self, tile_x: int, tile_y: int) -> str:
        """Return the Google satellite tile URL, rotating the mt subdomains."""
        return GOOGLE_TILE_URL.format(
            subdomain=(tile_x + tile_y) % 4,
            layer=self.layer,
            x=tile_x,
            y=tile_y,
            z=self.zoom_level,
        )


class GoogleStreetViewImageExtractor:
    """
    Street-view extractor that fetches keyless Google Street View crops of assets.

    For every asset the extractor locates the nearest official Google Street
    View panorama (within ``search_radius_m`` of the asset centroid), downloads
    the panorama tiles, crops the horizontal span that covers the asset
    footprint and attaches the crop to the asset as an
    :class:`~rapidtools.core.ImageAsset`. Additional viewpoints are taken from
    the panorama's neighbour links when ``max_images_per_asset`` is greater
    than one. No API key is required; the endpoints are the ones used by the
    Google Maps web app (as in BRAILS++), so availability may change.

    Args:
        save_directory (str | Path):
            Directory where the JPEG crops (and optional panoramas/depth maps)
            are written.
        search_radius_m (float, optional):
            Radius around the asset centroid searched for panoramas.
            Defaults to 50.
        zoom (int, optional):
            Panorama tile zoom level 0-5 (zoom 3 is 4096 x 2048 px).
            Defaults to 3.
        max_images_per_asset (int, optional):
            Maximum number of viewpoints per asset; extra viewpoints come from
            neighbouring panoramas ordered by distance. Defaults to 1.
        fov_buffer_deg (float, optional):
            Horizontal margin added on both sides of the footprint.
            Defaults to 10.
        min_fov_deg (float, optional):
            Minimum horizontal field of view of a crop. Defaults to 30.
        vertical_crop (tuple[float, float], optional):
            ``(top, bottom)`` fractions of the panorama height to keep.
            Defaults to ``(0.0, 1.0)`` (full height).
        save_panorama (bool, optional):
            Also save the full stitched panorama next to the crop.
        save_depth_map (bool, optional):
            Also save the decoded depth map (``.npy``) of each panorama.
        image_prefix (str, optional):
            Prefix of the saved filenames. Defaults to ``'gsv'``.
        max_workers (int, optional):
            Number of assets processed concurrently. Defaults to 5.
        client (GoogleStreetViewClient | None, optional):
            Pre-configured client; one is created when omitted.
        cancel_event (threading.Event | None, optional):
            Cooperative cancellation flag checked before each asset.

    Example:
        >>> from rapidtools import PhysicalAssetCollection, Pipeline
        >>> from rapidtools.processing import GoogleStreetViewImageExtractor
        >>> buildings = PhysicalAssetCollection.from_geojson('buildings.geojson')
        >>> extractor = GoogleStreetViewImageExtractor(
        ...     save_directory='output/streetview',
        ...     max_images_per_asset=2,
        ...     vertical_crop=(0.2, 0.9),
        ... )
        >>> buildings = Pipeline([extractor]).run(buildings)
        >>> buildings.get('bldg_01').image_assets[0].properties['pano_id']
        'u3PxkEsnYto4l1N3yTmO_Q'
    """

    stage = Stage.EXTRACT_IMAGERY

    def __init__(
        self,
        save_directory: str | Path,
        search_radius_m: float = 50.0,
        zoom: int = 3,
        max_images_per_asset: int = 1,
        fov_buffer_deg: float = 10.0,
        min_fov_deg: float = 30.0,
        vertical_crop: tuple[float, float] = (0.0, 1.0),
        save_panorama: bool = False,
        save_depth_map: bool = False,
        image_prefix: str = 'gsv',
        max_workers: int = 5,
        client: GoogleStreetViewClient | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """
        Initialize the extractor.

        Args:
            save_directory (str | Path): Output directory for the crops.
            search_radius_m (float, optional): Panorama search radius.
            zoom (int, optional): Panorama tile zoom level (0-5).
            max_images_per_asset (int, optional): Viewpoints per asset.
            fov_buffer_deg (float, optional): Horizontal crop margin.
            min_fov_deg (float, optional): Minimum crop field of view.
            vertical_crop (tuple[float, float], optional): Height fractions kept.
            save_panorama (bool, optional): Save full panoramas too.
            save_depth_map (bool, optional): Save decoded depth maps too.
            image_prefix (str, optional): Filename prefix.
            max_workers (int, optional): Concurrent assets.
            client (GoogleStreetViewClient | None, optional): Client to reuse.
            cancel_event (threading.Event | None, optional): Cancellation flag.

        Raises:
            ValueError: If ``zoom`` is outside 0-5 or ``max_images_per_asset``
                is not positive.
        """
        if zoom < 0 or zoom > 5:
            raise ValueError(f'zoom must be between 0 and 5, got {zoom}.')
        if max_images_per_asset < 1:
            raise ValueError('max_images_per_asset must be at least 1.')
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(parents=True, exist_ok=True)
        self.search_radius_m = search_radius_m
        self.zoom = zoom
        self.max_images_per_asset = max_images_per_asset
        self.fov_buffer_deg = fov_buffer_deg
        self.min_fov_deg = min_fov_deg
        self.vertical_crop = vertical_crop
        self.save_panorama = save_panorama
        self.save_depth_map = save_depth_map
        self.image_prefix = image_prefix
        self.max_workers = max(1, max_workers)
        self.client = client or GoogleStreetViewClient(max_workers=8)
        self.cancel_event = cancel_event

    def _candidate_panoramas(
        self, asset: PhysicalAsset
    ) -> list[tuple[StreetViewPanorama, float]]:
        """
        Return up to ``max_images_per_asset`` panoramas near an asset.

        The nearest panorama comes from a location search; further viewpoints
        are its neighbour links that lie within the search radius, ordered by
        distance to the asset centroid.
        """
        centroid = asset.geometry.centroid
        lat, lon = centroid.y, centroid.x
        nearest = self.client.find_panorama(lat, lon, radius_m=self.search_radius_m)
        if nearest is None:
            return []
        candidates = [(nearest, haversine_m(lat, lon, nearest.lat, nearest.lon))]
        if self.max_images_per_asset > 1:
            linked = []
            for link_id, link_lat, link_lon, _heading in nearest.links:
                distance = haversine_m(lat, lon, link_lat, link_lon)
                if distance <= self.search_radius_m:
                    linked.append((link_id, distance))
            linked.sort(key=lambda item: item[1])
            for link_id, distance in linked[: self.max_images_per_asset - 1]:
                meta = self.client.get_panorama_metadata(link_id)
                if meta is not None:
                    candidates.append((meta, distance))
        return candidates

    def _process_asset(self, asset: PhysicalAsset) -> int:
        """Download and attach every selected viewpoint of one asset."""
        raise_if_cancelled(self.cancel_event, 'Google Street View extraction')
        if asset.geometry is None or asset.geometry.is_empty:
            return 0

        extracted = 0
        for index, (pano, distance) in enumerate(self._candidate_panoramas(asset)):
            raise_if_cancelled(self.cancel_event, 'Google Street View extraction')
            try:
                image = self.client.download_panorama(pano, zoom=self.zoom)
                if image is None:
                    continue
                crop, (start, end) = self.client.crop_to_geometry(
                    image,
                    pano,
                    asset.geometry,
                    fov_buffer_deg=self.fov_buffer_deg,
                    min_fov_deg=self.min_fov_deg,
                    vertical_crop=self.vertical_crop,
                )
                stem = f'{self.image_prefix}_{asset.id}_{index}'
                crop_path = self.save_directory / f'{stem}.jpg'
                crop.save(crop_path, format='JPEG', quality=95)

                properties: dict[str, Any] = {
                    'source': 'google_streetview',
                    'pano_id': pano.id,
                    'camera_latitude': pano.lat,
                    'camera_longitude': pano.lon,
                    'camera_heading': pano.heading,
                    'camera_distance_m': round(distance, 2),
                    'capture_date': pano.capture_date,
                    'view_angle_start': round(start, 2),
                    'view_angle_end': round(end, 2),
                    'zoom': self.zoom,
                }
                if self.save_panorama:
                    pano_path = self.save_directory / f'{stem}_pano.jpg'
                    image.save(pano_path, format='JPEG', quality=90)
                    properties['panorama_path'] = str(pano_path)
                if self.save_depth_map:
                    depth_path = self._save_depth_map(pano, stem)
                    if depth_path is not None:
                        properties['depth_map_path'] = str(depth_path)

                asset.add_image_assets(
                    ImageAsset(
                        id=f'{asset.id}_gsv_{index}',
                        path=crop_path,
                        properties=properties,
                    )
                )
                extracted += 1
            except OperationCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - reported per viewpoint
                logger.error(
                    f'Failed to process panorama {pano.id} for asset {asset.id}: {exc}'
                )
        return extracted

    def _save_depth_map(self, pano: StreetViewPanorama, stem: str) -> Path | None:
        """Decode and save the panorama depth map as ``<stem>_depth.npy``."""
        from rapidtools.data_sources.google_streetview import decode_depth_map

        meta = pano
        if meta.depth_map_b64 is None:
            meta = self.client.get_panorama_metadata(pano.id)
        if meta is None or meta.depth_map_b64 is None:
            logger.warning(f'No depth map available for panorama {pano.id}.')
            return None
        depth_path = self.save_directory / f'{stem}_depth.npy'
        np.save(depth_path, decode_depth_map(meta.depth_map_b64))
        return depth_path

    def __call__(
        self, asset_collection: PhysicalAssetCollection
    ) -> PhysicalAssetCollection:
        """
        Attach Google Street View crops to every asset in the collection.

        Args:
            asset_collection (PhysicalAssetCollection): Assets with WGS84
                geometries.

        Returns:
            PhysicalAssetCollection: The same collection with
            :class:`~rapidtools.core.ImageAsset` records added to each asset
            for which imagery was found.

        Raises:
            OperationCancelled: If ``cancel_event`` is set during extraction.

        Example:
            >>> extractor = GoogleStreetViewImageExtractor('output/streetview')
            >>> collection = extractor(collection)
            >>> len(collection.get('bldg_01').image_assets)
            1
        """
        logger.info(
            f'Google Street View: extracting imagery for {len(asset_collection)} '
            f'assets using {self.max_workers} threads...'
        )
        total = 0
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as pool:
            futures = [pool.submit(self._process_asset, a) for a in asset_collection]
            try:
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc='Extracting Street View',
                ):
                    total += future.result()
            except OperationCancelled:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
        logger.info(f'Google Street View: attached {total} images.')
        return asset_collection
