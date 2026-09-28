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
Region-wide feature discovery on orthomosaic rasters.

This module provides :class:`SAM3OrthoFeatureExtractor`, which scans an
entire GeoTIFF in overlapping tiles, prompts a local SAM 3 model for a given
feature (e.g., ``'building'``, ``'road'``, ``'solar panel'``), and converts
the returned pixel masks into georeferenced Shapely polygons. Overlapping
detections from neighbouring tiles are dissolved into single assets, or, with
``merge_overlaps=False``, kept as one asset per detected instance (for
countable objects such as vehicles) with cross-tile duplicates removed.

Unlike the components in :mod:`rapidtools.processing.image_extractors`, this
extractor does not need an existing asset collection: it *creates* one from
scratch.

Example:
    >>> from rapidtools.processing import SAM3OrthoFeatureExtractor
    >>> extractor = SAM3OrthoFeatureExtractor(
    ...     prompt='building', patch_size=150, unit='feet', overlap_ratio=0.2
    ... )
    >>> buildings = extractor('data/eaton_ortho.tif')
    >>> buildings.to_geojson('eaton_buildings.geojson')

    Count vehicles instead of dissolving touching ones into a blob:

    >>> vehicles = SAM3OrthoFeatureExtractor(
    ...     prompt='vehicle', patch_size=50, unit='meters', merge_overlaps=False
    ... )('data/eaton_ortho.tif')
    >>> vehicles[0].attributes['confidence']
    0.93
"""

import logging
import os
import tempfile
import threading
import uuid
from pathlib import Path

import numpy as np
import rasterio.features
from rasterio.transform import from_bounds
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.strtree import STRtree
from tqdm import tqdm

from rapidtools.core import PhysicalAsset, PhysicalAssetCollection, raise_if_cancelled
from rapidtools.data_sources import OrthomosaicReader
from rapidtools.models import SAM3Inference

from .step import Stage

logger = logging.getLogger(__name__)

# With merge_overlaps=False, a detection is treated as a duplicate of a
# higher-scoring one when this fraction of the smaller polygon is covered:
DUPLICATE_OVERLAP_RATIO = 0.5


class SAM3OrthoFeatureExtractor:
    """
    Pipeline component that uses local SAM 3 to discover real-world assets.

    The extractor tiles a high-resolution orthomosaic raster into square
    patches of ``patch_size`` (in real-world units), runs SAM 3 on each batch
    of patches with the given text prompt, and maps the resulting pixel masks
    back to WGS84 polygons. Because neighbouring tiles overlap, the same
    object is frequently detected more than once; by default a final
    ``unary_union`` melts those duplicates into a single geometry per
    physical feature. That suits area features (roofs, roads, vegetation)
    but also fuses countable objects that touch, such as cars parked side
    by side. Pass ``merge_overlaps=False`` to keep one asset per SAM 3
    instance instead: each asset then carries the model's ``confidence``,
    and duplicates of the same object seen from overlapping tiles are
    removed with a greedy overlap suppression (highest confidence wins).

    Args:
        prompt (str):
            The text prompt describing the feature to extract
            (e.g., ``'building'``, ``'swimming pool'``).
        patch_size (float, optional):
            Edge length of each square scanning tile, expressed in ``unit``.
            Defaults to 30.0.
        unit (str, optional):
            Spatial unit of ``patch_size``: ``'pixels'``, ``'meters'``,
            ``'feet'``, ``'yards'``, ``'kilometers'`` or ``'miles'``.
            Defaults to ``'feet'``.
        overlap_ratio (float, optional):
            Fractional overlap between adjacent tiles (0.0 <= r < 1.0). Higher
            values reduce edge artefacts at the cost of more inference.
            Defaults to 0.15.
        model_id (str, optional):
            Hugging Face repository ID of the SAM 3 model. Defaults to
            ``'facebook/sam3'``.
        device (str, optional):
            Compute device (``'cuda'``, ``'cpu'`` or ``'auto'``). Defaults to
            ``'auto'``.
        batch_size (int, optional):
            Number of tiles sent to the model per inference call. Defaults
            to 4.
        load_in_4bit (bool, optional):
            Whether to load the model with 4-bit quantization. Defaults to
            ``True``.
        threshold (float, optional):
            Detection confidence threshold. Defaults to 0.5.
        mask_threshold (float, optional):
            Threshold for binarizing the predicted masks. Defaults to 0.5.
        max_missing_data_ratio (float, optional):
            Tiles whose nodata fraction exceeds this value are skipped.
            Defaults to 0.95.
        merge_overlaps (bool, optional):
            ``True`` dissolves all touching or overlapping detections into
            single polygons (area features). ``False`` keeps one asset per
            detected instance with a ``confidence`` attribute and only
            removes cross-tile duplicates (countable objects). Defaults to
            ``True``.
        cancel_event (threading.Event | None, optional):
            Cooperative cancellation flag checked before every tile. When set,
            :class:`~rapidtools.core.OperationCancelled` is raised. Defaults
            to ``None``.

    Example:
        Discover buildings in a Bing-derived orthomosaic and feed the result
        into a downstream pipeline:

        >>> from rapidtools.core import BoundingBox
        >>> from rapidtools.processing import (
        ...     BingOrthomosaicExtractor, SAM3OrthoFeatureExtractor
        ... )
        >>> region = BoundingBox(-118.15, 34.18, -118.14, 34.19)
        >>> tiff = BingOrthomosaicExtractor(zoom_level=19)(region, 'area.tiff')
        >>> extractor = SAM3OrthoFeatureExtractor(
        ...     prompt='building', patch_size=200, unit='feet', batch_size=8
        ... )
        >>> buildings = extractor(tiff)
        >>> buildings[0].attributes['asset_type']
        'building'
    """

    stage = Stage.DETECT

    def __init__(
        self,
        prompt: str,
        patch_size: float = 30.0,
        unit: str = 'feet',
        overlap_ratio: float = 0.15,
        model_id: str = 'facebook/sam3',
        device: str = 'auto',
        batch_size: int = 4,
        load_in_4bit: bool = True,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        max_missing_data_ratio: float = 0.95,
        merge_overlaps: bool = True,
        cancel_event: threading.Event | None = None,
        raster_path: str | Path | None = None,
    ):
        """
        Initialize the extractor and load the SAM 3 model once.

        Args:
            prompt (str):
                The text prompt describing the feature to extract.
            patch_size (float, optional):
                Edge length of each scanning tile in ``unit``.
            unit (str, optional):
                Spatial unit of ``patch_size``.
            overlap_ratio (float, optional):
                Fractional overlap between adjacent tiles.
            model_id (str, optional):
                Hugging Face repository ID of the SAM 3 model.
            device (str, optional):
                Compute device (``'cuda'``, ``'cpu'`` or ``'auto'``).
            batch_size (int, optional):
                Number of tiles per inference call.
            load_in_4bit (bool, optional):
                Whether to load the model with 4-bit quantization.
            threshold (float, optional):
                Detection confidence threshold.
            mask_threshold (float, optional):
                Threshold for binarizing the predicted masks.
            max_missing_data_ratio (float, optional):
                Maximum nodata fraction for a tile to be processed.
            merge_overlaps (bool, optional):
                Dissolve touching detections (``True``) or keep one asset
                per instance (``False``).
            cancel_event (threading.Event | None, optional):
                Cooperative cancellation flag checked before every tile.
        """
        # Cooperative cancellation (stops between tiles):
        self.cancel_event = cancel_event
        # Optional default raster so the extractor can run as a pipeline step:
        self.raster_path = Path(raster_path) if raster_path is not None else None
        self.prompt = prompt
        self.patch_size = patch_size
        self.unit = unit
        self.overlap_ratio = overlap_ratio
        self.batch_size = batch_size
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.max_missing_data_ratio = max_missing_data_ratio
        self.merge_overlaps = merge_overlaps

        logger.info(
            f"Initializing SAM3OrthoFeatureExtractor for '{self.prompt}' "
            f'at scale: {self.patch_size} {self.unit} with threshold '
            f'{self.threshold}...'
        )

        self.model = SAM3Inference(
            model_id=model_id, device=device, load_in_4bit=load_in_4bit
        )

    def __call__(
        self, source: str | Path | PhysicalAssetCollection | None = None
    ) -> PhysicalAssetCollection:
        """
        Detect assets, either from a raster path or as a pipeline step.

        Args:
            source: A raster path (behaves like :meth:`detect`), a
                :class:`~rapidtools.core.PhysicalAssetCollection` (assets
                detected in ``raster_path`` given at construction are merged
                into it, skipping duplicate IDs), or ``None`` to scan
                ``raster_path``.

        Returns:
            PhysicalAssetCollection: The detected assets, or the input
            collection extended with them.

        Raises:
            ValueError: If no raster is available.

        Example:
            >>> extractor = SAM3OrthoFeatureExtractor(
            ...     'building', raster_path='ortho.tif'
            ... )
            >>> buildings = extractor()                 # standalone
            >>> merged = Pipeline([extractor]).run(existing_assets)  # as a step
        """
        if isinstance(source, PhysicalAssetCollection):
            if self.raster_path is None:
                raise ValueError(
                    'Pass raster_path to the constructor to use '
                    'SAM3OrthoFeatureExtractor as a pipeline step.'
                )
            detected = self.detect(self.raster_path)
            source.merge(detected, strategy='skip')
            return source
        raster = self.raster_path if source is None else source
        if raster is None:
            raise ValueError('No raster to scan: pass a path or set raster_path.')
        return self.detect(raster)

    def detect(self, raster_path: str | Path) -> PhysicalAssetCollection:
        """
        Scan the raster, extract features, and merge overlaps into a collection.

        Args:
            raster_path (str | Path):
                Path to a GeoTIFF orthomosaic (any CRS understood by
                rasterio; output geometries are always WGS84).

        Returns:
            PhysicalAssetCollection:
                One ``PhysicalAsset`` per merged feature (or per detected
                instance when ``merge_overlaps`` is ``False``). Each asset
                carries the attributes ``'asset_type'`` (the prompt),
                ``'source_model'`` and ``'extraction_scale'``; instance
                assets also carry ``'confidence'``. The collection is empty
                if nothing was detected.

        Raises:
            OperationCancelled:
                If ``cancel_event`` is set between two tiles.
            FileNotFoundError:
                If ``raster_path`` does not exist.

        Example:
            >>> extractor = SAM3OrthoFeatureExtractor(prompt='road')
            >>> roads = extractor('data/ortho.tif')
            >>> len(roads)
            12
        """
        raw_polygons: list = []
        # Per-polygon SAM 3 scores, only collected in instance mode:
        raw_scores: list | None = None if self.merge_overlaps else []

        with OrthomosaicReader(raster_path) as reader:
            tile_generator = reader.generate_tiles(
                patch_size=self.patch_size,
                unit=self.unit,
                overlap_ratio=self.overlap_ratio,
                max_missing_data_ratio=self.max_missing_data_ratio,
                pad_edge_tiles=True,
            )

            batch_images = []
            batch_bounds = []

            for pil_image, wgs84_bounds in tqdm(
                tile_generator, desc=f"Scanning for '{self.prompt}'"
            ):
                raise_if_cancelled(self.cancel_event, f"scan for '{self.prompt}'")
                batch_images.append(pil_image)
                batch_bounds.append(wgs84_bounds)

                if len(batch_images) == self.batch_size:
                    self._process_batch(
                        batch_images, batch_bounds, raw_polygons, raw_scores
                    )
                    batch_images, batch_bounds = [], []

            # Process any remaining images in the final partial batch
            if batch_images:
                self._process_batch(
                    batch_images, batch_bounds, raw_polygons, raw_scores
                )

        final_collection = PhysicalAssetCollection()

        if not self.merge_overlaps:
            logger.info(
                f'Extracted {len(raw_polygons)} raw instances. '
                'Removing cross-tile duplicates...'
            )
            for geom, score in self._suppress_duplicates(raw_polygons, raw_scores):
                final_collection.add(
                    PhysicalAsset(
                        id=f'{self.prompt.replace(" ", "_")}_{uuid.uuid4().hex[:8]}',
                        geometry=geom,
                        attributes={
                            'asset_type': self.prompt,
                            'confidence': score,
                            'source_model': self.model.model_id,
                            'extraction_scale': f'{self.patch_size}_{self.unit}',
                        },
                    )
                )
            logger.info(
                f'Extraction complete. Yielded {len(final_collection)} unique assets.'
            )
            return final_collection

        logger.info(f'Extracted {len(raw_polygons)} raw polygons. Merging overlaps...')

        if raw_polygons:
            # unary_union automatically melts overlapping geometries together
            merged_geometry = unary_union(raw_polygons)

            # Unpack the results safely depending on what unary_union returns
            if merged_geometry.geom_type == 'Polygon':
                final_geometries = [merged_geometry]
            elif merged_geometry.geom_type == 'MultiPolygon':
                final_geometries = list(merged_geometry.geoms)
            else:
                final_geometries = [
                    g
                    for g in merged_geometry.geoms
                    if g.geom_type in ['Polygon', 'MultiPolygon']
                ]

            # Wrap the unified geometries into PhysicalAsset objects
            for geom in final_geometries:
                asset = PhysicalAsset(
                    id=f'{self.prompt.replace(" ", "_")}_{uuid.uuid4().hex[:8]}',
                    geometry=geom,
                    attributes={
                        'asset_type': self.prompt,
                        'source_model': self.model.model_id,
                        'extraction_scale': f'{self.patch_size}_{self.unit}',
                    },
                )
                final_collection.add(asset)

        logger.info(
            f'Extraction complete. Yielded {len(final_collection)} unique assets.'
        )
        return final_collection

    def _process_batch(
        self,
        batch_images: list,
        batch_bounds: list,
        raw_polygons: list,
        raw_scores: list | None = None,
    ) -> None:
        """
        Run inference on one batch of tiles and append polygons to ``raw_polygons``.

        The in-memory PIL tiles are written to temporary JPEG files (the model
        API expects paths), inference is run once for the whole batch, and
        each returned mask instance is vectorized with
        :func:`rasterio.features.shapes` using an affine transform derived from
        the tile's WGS84 bounds. Temporary files are always removed, even if
        inference raises.

        Args:
            batch_images (list[PIL.Image.Image]):
                The tiles to segment.
            batch_bounds (list[tuple[float, float, float, float]]):
                Matching ``(min_lon, min_lat, max_lon, max_lat)`` bounds for
                each tile.
            raw_polygons (list[shapely.geometry.Polygon]):
                Accumulator that valid polygons are appended to in place.
            raw_scores (list[float | None] | None):
                When given (instance mode), each SAM 3 instance contributes
                only its largest polygon, and its confidence (``None`` if the
                model reported none) is appended here in step with
                ``raw_polygons``.
        """
        temp_paths = []

        # 1. Save in-memory PIL images to temporary files
        for img in batch_images:
            fd, path = tempfile.mkstemp(suffix='.jpg')
            os.close(fd)
            # Save as JPEG for fast disk I/O
            img.save(path, format='JPEG')
            temp_paths.append(path)

        try:
            # 2. Run inference using the file paths WITH threshold parameters
            outputs = self.model.run_inference(
                image_inputs=temp_paths,
                prompt=self.prompt,
                threshold=self.threshold,
                mask_threshold=self.mask_threshold,
            )

            if not outputs or getattr(outputs, 'masks', None) is None:
                return

            raw_response = getattr(outputs, 'raw_response', None)
            scores_per_image = (
                raw_response.get('scores') if isinstance(raw_response, dict) else None
            ) or []

            # 3. Translate pixel masks to geographic polygons
            for image_index, (image_masks, bounds) in enumerate(
                zip(outputs.masks, batch_bounds, strict=False)
            ):
                if image_masks is None or len(image_masks) == 0:
                    continue
                scores = (
                    scores_per_image[image_index]
                    if image_index < len(scores_per_image)
                    else []
                )

                # Accept lists of 2D masks as well as stacked arrays:
                image_masks = np.asarray(image_masks)

                # Force 2D arrays to become 3D arrays (1, H, W):
                if image_masks.ndim == 2:
                    image_masks = image_masks[np.newaxis, ...]

                min_lon, min_lat, max_lon, max_lat = bounds

                # Safely grab height and width from the last two dimensions:
                height, width = image_masks.shape[-2:]

                # Build an affine transform for this specific tile
                transform = from_bounds(
                    min_lon, min_lat, max_lon, max_lat, width, height
                )

                for instance_index, instance_mask in enumerate(image_masks):
                    # Binarize the float/boolean mask
                    binary_mask = (instance_mask > 0.5).astype(np.uint8)

                    # Extract the shapes natively using rasterio
                    polygons = []
                    for geom_dict, val in rasterio.features.shapes(
                        binary_mask, transform=transform
                    ):
                        if val == 1:
                            poly = shape(geom_dict)
                            if poly.is_valid and not poly.is_empty:
                                polygons.append(poly)

                    if raw_scores is None:
                        raw_polygons.extend(polygons)
                    elif polygons:
                        # One instance is one object: keep its largest piece
                        # (a mask split by e.g. a tree branch would otherwise
                        # count twice):
                        raw_polygons.append(max(polygons, key=lambda p: p.area))
                        score = (
                            float(scores[instance_index])
                            if instance_index < len(scores)
                            else None
                        )
                        raw_scores.append(score)
        finally:
            # 4. Always clean up temporary files to prevent disk bloat
            for path in temp_paths:
                if os.path.exists(path):
                    os.remove(path)

    @staticmethod
    def _suppress_duplicates(polygons: list, scores: list | None) -> list[tuple]:
        """
        Drop lower-scoring duplicates of the same object across tiles.

        Polygons are visited from the highest to the lowest confidence. A
        polygon is discarded when an already kept polygon covers more than
        :data:`DUPLICATE_OVERLAP_RATIO` of the smaller of the two areas.
        Using the smaller area rather than IoU also removes the partial view
        of an object cut by a tile edge when a neighbouring tile saw it
        whole. Polygons that merely touch are both kept.

        Args:
            polygons (list[shapely.geometry.Polygon]):
                Candidate polygons, one per detected instance.
            scores (list[float | None] | None):
                Matching confidences; ``None`` entries sort last.

        Returns:
            list[tuple[shapely.geometry.Polygon, float | None]]:
                The surviving ``(polygon, score)`` pairs, best first.
        """
        if not polygons:
            return []
        if scores is None:
            scores = [None] * len(polygons)

        order = sorted(
            range(len(polygons)),
            key=lambda k: -1.0 if scores[k] is None else -scores[k],
        )
        ranked = [polygons[k] for k in order]
        tree = STRtree(ranked)
        is_kept = [False] * len(ranked)
        survivors: list[tuple] = []

        for index, poly in enumerate(ranked):
            duplicate = False
            for other in tree.query(poly):
                other = int(other)
                if other == index or not is_kept[other]:
                    continue
                overlap = poly.intersection(ranked[other]).area
                if overlap <= 0:
                    continue
                smaller = min(poly.area, ranked[other].area)
                if smaller > 0 and overlap / smaller > DUPLICATE_OVERLAP_RATIO:
                    duplicate = True
                    break
            if not duplicate:
                is_kept[index] = True
                survivors.append((poly, scores[order[index]]))
        return survivors
