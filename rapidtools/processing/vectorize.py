"""
Mask-to-polygon conversion shared by the raster feature extractors.

A segmentation mask drawn on an image patch is a pixel grid. Turning it into
WGS84 polygons needs the patch's pixel-to-ground mapping. The exact mapping is
the raster's own affine transform for the window that was read (see
:class:`~rapidtools.data_sources.orthomosaic_reader.PatchGeoref`); the
patch's WGS84 bounding box is only an envelope of that footprint and is exact
solely for north-up rasters that are already in WGS84.

Example:
    >>> import numpy as np
    >>> from rapidtools.processing.vectorize import mask_to_wgs84_polygons
    >>> mask = np.zeros((4, 4), dtype=bool)
    >>> mask[1:3, 1:3] = True
    >>> polys = mask_to_wgs84_polygons(mask, wgs84_bounds=(0.0, 0.0, 4.0, 4.0))
    >>> polys[0].bounds
    (1.0, 1.0, 3.0, 3.0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio.features
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import transform_geom
from shapely.geometry import Polygon, shape

if TYPE_CHECKING:
    from rapidtools.data_sources.orthomosaic_reader import PatchGeoref

_WGS84 = CRS.from_epsg(4326)


def mask_to_wgs84_polygons(
    mask: Any,
    georef: PatchGeoref | None = None,
    wgs84_bounds: tuple[float, float, float, float] | None = None,
    threshold: float = 0.5,
) -> list[Polygon]:
    """
    Vectorize a binary mask into valid WGS84 polygons.

    Args:
        mask (array-like):
            A 2D mask (boolean, integer or float). Values above ``threshold``
            are foreground. Leading singleton dimensions are squeezed away.
        georef (PatchGeoref | None):
            The native georeferencing of the patch the mask was drawn on.
            When given, shapes are extracted in the raster's own grid (scaled
            to the mask size) and reprojected to WGS84, which is correct for
            projected and rotated rasters.
        wgs84_bounds (tuple[float, float, float, float] | None):
            ``(min_lon, min_lat, max_lon, max_lat)`` of the patch. Used only
            when ``georef`` is ``None``; it maps pixels linearly to the
            envelope, which is exact only for north-up WGS84 rasters.
        threshold (float):
            Foreground cut-off for float masks. Defaults to ``0.5``.

    Returns:
        list[shapely.geometry.Polygon]:
            Valid, non-empty polygons, one per connected foreground region.

    Raises:
        ValueError: If the mask is not 2D after squeezing, or neither
            ``georef`` nor ``wgs84_bounds`` is given.
    """
    binary = np.squeeze(np.asarray(mask))
    if binary.ndim != 2:
        raise ValueError(f'Expected a 2D mask, got shape {np.shape(mask)}.')
    binary = (binary > threshold).astype(np.uint8)
    height, width = binary.shape

    src_crs: CRS | None
    if georef is not None:
        transform = georef.transform_for(width, height)
        src_crs = georef.crs
    elif wgs84_bounds is not None:
        transform = from_bounds(*wgs84_bounds, width, height)
        src_crs = None
    else:
        raise ValueError('Either georef or wgs84_bounds is required.')

    reproject = src_crs is not None and src_crs != _WGS84

    polygons: list[Polygon] = []
    for geom_dict, value in rasterio.features.shapes(
        binary, mask=binary.astype(bool), transform=transform
    ):
        if value != 1:
            continue
        if reproject:
            geom_dict = transform_geom(src_crs, _WGS84, geom_dict)
        poly = shape(geom_dict)
        if poly.is_valid and not poly.is_empty:
            polygons.append(poly)
    return polygons
