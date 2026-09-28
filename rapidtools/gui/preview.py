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
# 09-22-2026

"""
Raster preview rendering and vector overlay projection for the rapidtools GUI.

The browser front end cannot read multi-gigabyte GeoTIFFs, so the server
renders a decimated PNG preview once and projects asset geometries (stored in
WGS84) into the preview's pixel space. The client then simply draws polygons
on top of the image. :class:`RasterTiler` complements the static preview by
serving full-resolution JPEG tiles on demand for deep zooms.

Example:
    >>> from rapidtools.core import PhysicalAssetCollection
    >>> from rapidtools.gui.preview import project_collection, render_preview
    >>>
    >>> preview = render_preview('scene.tif', max_px=800)
    >>> assets = PhysicalAssetCollection.from_geojson('buildings.geojson')
    >>> overlay = project_collection(assets, preview)
    >>> overlay['count'] == len(assets)
    True
"""

from __future__ import annotations

import io
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rapidtools.core import PhysicalAssetCollection

logger = logging.getLogger(__name__)

MAX_PREVIEW_PX = 1600
TILE_SIZE = 512
TILE_DOWNSAMPLES = (1, 2, 4, 8, 16, 32, 64)


@dataclass
class RasterPreview:
    """
    A decimated PNG rendering of a raster plus the data needed to overlay.

    Attributes:
        path (Path): Absolute path of the source raster.
        width (int): Native raster width in pixels.
        height (int): Native raster height in pixels.
        preview_width (int): Width of the rendered PNG in pixels.
        preview_height (int): Height of the rendered PNG in pixels.
        crs_wkt (str): The raster CRS as WKT (empty if the raster has none).
        transform (tuple[float, ...]): The six affine coefficients
            ``(a, b, c, d, e, f)`` mapping pixel to world coordinates.
        wgs84_bounds (tuple[float, float, float, float]): Extent in WGS84 as
            ``(min_lon, min_lat, max_lon, max_lat)``.
        png (bytes): The encoded PNG preview.
        version (int): Incremented by the server each time the preview for a
            given key is replaced, so the browser can invalidate caches.

    Example:
        >>> from rapidtools.gui.preview import render_preview
        >>>
        >>> preview = render_preview('scene.tif', max_px=400)
        >>> preview.png[:4] == b'\x89PNG'
        True
        >>> preview.to_json()['name']
        'scene.tif'
    """

    path: Path
    width: int
    height: int
    preview_width: int
    preview_height: int
    crs_wkt: str
    transform: tuple[float, float, float, float, float, float]
    wgs84_bounds: tuple[float, float, float, float]
    png: bytes = field(repr=False)
    version: int = 1

    @property
    def scale(self) -> tuple[float, float]:
        """
        Return the preview-to-native pixel scale factors.

        Returns:
            tuple[float, float]: ``(x_scale, y_scale)``, each ``<= 1``.
        """
        return self.preview_width / self.width, self.preview_height / self.height

    def to_json(self) -> dict[str, Any]:
        """
        Serialize the preview metadata (without the PNG bytes) for the browser.

        Returns:
            dict[str, Any]:
                Keys ``path``, ``name``, ``width``, ``height``,
                ``preview_width``, ``preview_height``, ``wgs84_bounds`` and
                ``version``.
        """
        return {
            'path': str(self.path),
            'name': self.path.name,
            'width': self.width,
            'height': self.height,
            'preview_width': self.preview_width,
            'preview_height': self.preview_height,
            'wgs84_bounds': list(self.wgs84_bounds),
            'version': self.version,
        }


def render_preview(
    raster_path: str | Path, max_px: int = MAX_PREVIEW_PX
) -> RasterPreview:
    """
    Read ``raster_path`` decimated to at most ``max_px`` on its long side.

    The first three bands are averaged down to the preview size and converted
    to 8-bit RGB. Single-band rasters are replicated to grey RGB. A fourth band
    is treated as alpha; otherwise the raster's nodata mask (if any) is used so
    that holes in the imagery render transparent.

    Args:
        raster_path (str | Path): Path to a georeferenced raster readable by
            rasterio.
        max_px (int): Maximum size of the long side of the preview in pixels.
            Defaults to :data:`MAX_PREVIEW_PX`.

    Returns:
        RasterPreview: The rendered preview and its georeferencing metadata.

    Raises:
        rasterio.errors.RasterioIOError: If the raster cannot be opened.

    Example:
        >>> from rapidtools.gui.preview import render_preview
        >>>
        >>> preview = render_preview('scene.tif', max_px=200)
        >>> max(preview.preview_width, preview.preview_height) <= 200
        True
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds

    from rapidtools.data_sources import OrthomosaicReader

    raster_path = Path(raster_path)
    with rasterio.open(raster_path) as src:
        scale = min(1.0, max_px / max(src.width, src.height))
        out_w = max(1, int(round(src.width * scale)))
        out_h = max(1, int(round(src.height * scale)))
        logger.info(
            f'Rendering {out_w}x{out_h} preview of {raster_path.name} '
            f'({src.width}x{src.height} px)...'
        )
        bands = min(src.count, 3)
        array = src.read(
            indexes=list(range(1, bands + 1)),
            out_shape=(bands, out_h, out_w),
            resampling=Resampling.average,
        )
        if bands == 1:
            array = np.repeat(array, 3, axis=0)
        rgb = OrthomosaicReader._format_array_for_pil(array)
        image = Image.fromarray(rgb, mode='RGB')

        # Mark nodata / alpha-transparent pixels so the preview shows holes.
        alpha = None
        if src.count >= 4:
            alpha = src.read(
                indexes=4, out_shape=(out_h, out_w), resampling=Resampling.nearest
            )
        elif src.nodata is not None:
            mask = src.read_masks(1, out_shape=(out_h, out_w))
            alpha = mask
        if alpha is not None:
            image.putalpha(Image.fromarray(alpha.astype(np.uint8), mode='L'))

        buffer = io.BytesIO()
        image.save(buffer, format='PNG', optimize=False)

        crs = src.crs
        if crs:
            bounds = transform_bounds(crs, 'EPSG:4326', *src.bounds)
        else:
            # No CRS: treat the raster as already being in WGS84, matching
            # BoundingBox.from_raster, instead of failing in transform_bounds.
            bounds = tuple(src.bounds)
        t = src.transform
        return RasterPreview(
            path=raster_path.resolve(),
            width=src.width,
            height=src.height,
            preview_width=out_w,
            preview_height=out_h,
            crs_wkt=crs.to_wkt() if crs else '',
            transform=(t.a, t.b, t.c, t.d, t.e, t.f),
            wgs84_bounds=tuple(bounds),
            png=buffer.getvalue(),
        )


def _json_safe(value: Any) -> Any:
    """
    Coerce attribute values into JSON-serialisable primitives.

    Args:
        value (Any): Any attribute value.

    Returns:
        Any: ``None``/bool/int/float/str unchanged, lists and dicts converted
        recursively, and everything else via ``str()``.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def project_collection(
    collection: PhysicalAssetCollection, preview: RasterPreview
) -> dict[str, Any]:
    """
    Project every asset geometry into the preview's pixel coordinates.

    Geometries are stored in WGS84; they are transformed into the raster CRS
    (when the raster has one), pushed through the inverse affine transform to
    native pixel coordinates, and finally scaled to the preview size.
    Polygons and multipolygons contribute their exterior rings, lines their
    vertices, points a single vertex, and any other geometry type its
    envelope. Empty geometries are skipped.

    Args:
        collection (PhysicalAssetCollection): The assets to project.
        preview (RasterPreview): The preview whose pixel space is the target.

    Returns:
        dict[str, Any]:
            ``preview_version``, ``count``, the sorted union of
            ``attribute_keys`` (so the client can offer a "colour by" menu),
            and ``features``: one dict per asset with ``id``, ``kind``
            (``'polygon'``, ``'line'`` or ``'point'``), pixel-space ``rings``
            as lists of ``[x, y]`` pairs, JSON-safe ``attributes`` and
            ``image_count``.

    Example:
        >>> from rapidtools.core import PhysicalAssetCollection
        >>> from rapidtools.gui.preview import project_collection, render_preview
        >>>
        >>> preview = render_preview('scene.tif')
        >>> assets = PhysicalAssetCollection.from_geojson('buildings.geojson')
        >>> overlay = project_collection(assets, preview)
        >>> overlay['features'][0]['kind']
        'polygon'
    """
    from affine import Affine
    from pyproj import CRS, Transformer

    if preview.crs_wkt:
        target = CRS.from_wkt(preview.crs_wkt)
        transformer = Transformer.from_crs('EPSG:4326', target, always_xy=True)
    else:
        transformer = None
    inverse = ~Affine(*preview.transform)
    sx, sy = preview.scale

    def ring_to_pixels(coords) -> list[list[float]]:
        """Convert WGS84 ``(x, y)`` pairs into rounded preview pixel pairs."""
        xs = np.asarray([c[0] for c in coords], dtype=float)
        ys = np.asarray([c[1] for c in coords], dtype=float)
        if transformer is not None:
            xs, ys = transformer.transform(xs, ys)
        cols, rows = inverse * (xs, ys)
        return [
            [round(float(c) * sx, 2), round(float(r) * sy, 2)]
            for c, r in zip(cols, rows, strict=True)
        ]

    features: list[dict[str, Any]] = []
    keys: set[str] = set()
    for asset in collection:
        geom = asset.geometry
        if geom is None or geom.is_empty:
            continue
        rings: list[list[list[float]]] = []
        kind = 'polygon'
        if geom.geom_type == 'Polygon':
            rings.append(ring_to_pixels(geom.exterior.coords))
        elif geom.geom_type == 'MultiPolygon':
            rings.extend(ring_to_pixels(p.exterior.coords) for p in geom.geoms)
        elif geom.geom_type in ('LineString', 'MultiLineString'):
            kind = 'line'
            lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)
            rings.extend(ring_to_pixels(line.coords) for line in lines)
        elif geom.geom_type in ('Point', 'MultiPoint'):
            kind = 'point'
            points = [geom] if geom.geom_type == 'Point' else list(geom.geoms)
            rings.extend(ring_to_pixels([(p.x, p.y)]) for p in points)
        else:
            rings.append(ring_to_pixels(geom.envelope.exterior.coords))

        attributes = {str(k): _json_safe(v) for k, v in asset.attributes.items()}
        keys.update(attributes)
        features.append(
            {
                'id': asset.id,
                'kind': kind,
                'rings': rings,
                'attributes': attributes,
                'image_count': len(asset.image_assets),
            }
        )

    return {
        'preview_version': preview.version,
        'count': len(features),
        'attribute_keys': sorted(keys),
        'features': features,
    }


class RasterTiler:
    """
    Serve full-resolution windows of a raster on demand.

    Tiles are addressed on the raster's native pixel grid: tile ``(tx, ty)`` at
    downsample ``d`` covers ``TILE_SIZE * d`` native pixels per side and is
    returned as a JPEG of at most ``TILE_SIZE`` pixels per side (edge tiles are
    smaller). This lets the browser show the original imagery when zoomed in
    without ever transferring the whole multi-gigabyte file. Rendered tiles
    are kept in a bounded LRU cache.

    Example:
        >>> from rapidtools.gui.preview import RasterTiler
        >>>
        >>> tiler = RasterTiler('scene.tif')
        >>> jpeg = tiler.tile(downsample=1, tx=0, ty=0)
        >>> jpeg[:2] == b'\xff\xd8'
        True
        >>> tiler.tile(3, 0, 0) is None  # 3 is not a supported downsample
        True
    """

    def __init__(self, path: str | Path, max_cache: int = 600) -> None:
        """
        Initialize the tiler.

        Args:
            path (str | Path): Path to the raster to serve.
            max_cache (int): Maximum number of rendered tiles to keep in
                memory. Defaults to ``600``.
        """
        self.path = Path(path)
        self._cache: OrderedDict[tuple[int, int, int], bytes | None] = OrderedDict()
        self._max_cache = max_cache
        self._lock = threading.Lock()

    def tile(self, downsample: int, tx: int, ty: int) -> bytes | None:
        """
        Return the JPEG bytes for a tile, or ``None`` if it is off-raster.

        Args:
            downsample (int): One of :data:`TILE_DOWNSAMPLES`; other values
                yield ``None``.
            tx (int): Tile column index (``>= 0``).
            ty (int): Tile row index (``>= 0``).

        Returns:
            bytes | None: Encoded JPEG, or ``None`` when the address is
            invalid or lies entirely outside the raster.

        Example:
            >>> from rapidtools.gui.preview import RasterTiler
            >>>
            >>> tiler = RasterTiler('scene.tif')
            >>> tiler.tile(1, -1, 0) is None
            True
        """
        if downsample not in TILE_DOWNSAMPLES or tx < 0 or ty < 0:
            return None
        key = (downsample, tx, ty)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        data = self._render(downsample, tx, ty)
        with self._lock:
            self._cache[key] = data
            while len(self._cache) > self._max_cache:
                self._cache.popitem(last=False)
        return data

    def _render(self, d: int, tx: int, ty: int) -> bytes | None:
        """Read and encode one tile from disk (see :meth:`tile`)."""
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import Window

        from rapidtools.data_sources import OrthomosaicReader

        span = TILE_SIZE * d
        col_off, row_off = tx * span, ty * span
        with rasterio.open(self.path) as src:
            if col_off >= src.width or row_off >= src.height:
                return None
            width = min(span, src.width - col_off)
            height = min(span, src.height - row_off)
            out_w = max(1, int(round(width / d)))
            out_h = max(1, int(round(height / d)))
            window = Window(col_off, row_off, width, height)
            bands = min(src.count, 3)
            array = src.read(
                indexes=list(range(1, bands + 1)),
                window=window,
                out_shape=(bands, out_h, out_w),
                resampling=Resampling.average if d > 1 else Resampling.nearest,
            )
            if bands == 1:
                array = np.repeat(array, 3, axis=0)
            rgb = OrthomosaicReader._format_array_for_pil(array)
            if src.count >= 4:
                alpha = src.read(
                    indexes=4,
                    window=window,
                    out_shape=(out_h, out_w),
                    resampling=Resampling.nearest,
                )
                rgb = rgb.copy()
                rgb[alpha == 0] = 0
        image = Image.fromarray(rgb, mode='RGB')
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=88)
        return buffer.getvalue()
