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
# 11-27-2026

"""
Region-wide feature discovery on orthomosaic rasters.

This module provides :class:`SAM3OrthoFeatureExtractor`, which scans an
entire GeoTIFF in overlapping tiles, prompts a local SAM 3 model for a given
feature (e.g., ``'building'``, ``'road'``, ``'solar panel'``), and converts
the returned pixel masks into georeferenced Shapely polygons. Overlapping
detections from neighboring tiles are dissolved into single assets, or, with
``merge_overlaps=False``, kept as one asset per object (for buildings and
countable objects such as vehicles): the views of an object from the
overlapping tiles are fused into one consensus outline, and neighbors stay
separate (see :mod:`rapidtools.processing.instance_fusion`).

Prompts that name buildings (see :func:`is_building_prompt`) are scanned as
buildings unless told otherwise: one footprint per building, from tiles that
overlap by half, with straight, square walls
(:class:`~rapidtools.processing.FootprintRegularizer`).

Unlike the components in :mod:`rapidtools.processing.image_extractors`, this
extractor does not need an existing asset collection: it *creates* one from
scratch.

Example:
    Building footprints, one per building, regularized:

    >>> from rapidtools.processing import SAM3OrthoFeatureExtractor
    >>> buildings = SAM3OrthoFeatureExtractor(
    ...     prompt='building', patch_size=50, unit='meters'
    ... )('data/eaton_bing.tiff')
    >>> buildings[0].attributes['views']
    4
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
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.crs import CRS
from rasterio.warp import transform_geom
from shapely.geometry import Polygon, box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union
from tqdm import tqdm

from rapidtools.core import PhysicalAsset, PhysicalAssetCollection, raise_if_canceled
from rapidtools.data_sources import OrthomosaicReader
from rapidtools.data_sources.orthomosaic_reader import PatchGeoref

from .instance_fusion import ViewDetection, fuse_view_detections
from .step import Stage
from .vectorize import mask_to_wgs84_polygons

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rapidtools.models import SAM3Inference


def _sam3_inference_class() -> 'type[SAM3Inference]':
    """
    Return :class:`~rapidtools.models.SAM3Inference`, importing it on first use.

    The SAM 3 wrapper pulls in PyTorch and Transformers, so it is not imported
    with the module; ``import rapidtools.processing`` stays light and the GUI
    can validate settings without loading the model stack. Tests may inject a
    stand-in by setting ``SAM3Inference`` on this module.
    """
    injected = globals().get('SAM3Inference')
    if injected is not None:
        return injected
    from rapidtools.models import SAM3Inference

    return SAM3Inference


def __getattr__(name: str) -> Any:
    """Expose ``SAM3Inference`` lazily so ``monkeypatch.setattr`` finds it."""
    if name == 'SAM3Inference':
        return _sam3_inference_class()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


# Prompts naming any of these words are scanned as buildings (see
# is_building_prompt); the words also match inside longer ones ('warehouse'):
BUILDING_PROMPT_WORDS = ('building', 'house', 'structure')
# Defaults of a building scan: every building is kept separate and seen
# whole by several tiles (tiles overlap by half), masks are cut where they
# narrow below a meter (a mask bleeding across the gap between two roofs),
# buildings that fewer than 30% of the tiles seeing them detected are
# dropped, and the footprints get straight, square walls:
BUILDING_OVERLAP_RATIO = 0.5
BUILDING_NECK_WIDTH_M = 1.0
BUILDING_MIN_DETECTION_RATE = 0.3
# Tile overlap for every other prompt:
DEFAULT_OVERLAP_RATIO = 0.15
# Smallest object part kept by the instance fusion, in square meters:
MIN_INSTANCE_AREA_M2 = 0.25
# SAM 3 resizes every image to 1008 x 1008 pixels, so larger tiles (a 50 m
# tile of a 2 cm drone orthomosaic is 2,500 pixels across) are downsampled
# to this size first; the masks then come back at this size as well, which
# keeps memory and vectorization cheap. The tile georeference is rescaled
# accordingly (PatchGeoref.transform_for).
MAX_TILE_SIDE_PX = 1024
# JPEG quality of the temporary tile files:
TILE_JPEG_QUALITY = 95
# A scan stops after this many failed SAM 3 batches in a row (a GPU that ran
# out of memory or stopped responding fails every batch that follows):
MAX_FAILED_BATCHES_IN_A_ROW = 3


def is_building_prompt(prompt: str) -> bool:
    """
    Whether a text prompt asks for buildings.

    A prompt asks for buildings when it contains one of
    :data:`BUILDING_PROMPT_WORDS`, in any case and also inside a longer word
    (``'warehouse'``, ``'buildings'``).

    Args:
        prompt (str): The text prompt.

    Returns:
        bool: ``True`` for building prompts.

    Example:
        >>> is_building_prompt('Damaged house'), is_building_prompt('tree')
        (True, False)
    """
    lowered = prompt.lower()
    return any(word in lowered for word in BUILDING_PROMPT_WORDS)


@dataclass(frozen=True)
class ScanSettings:
    """
    The scan settings in effect for a prompt (see
    :meth:`SAM3OrthoFeatureExtractor.scan_settings`).

    Attributes:
        building (bool): Whether the prompt asks for buildings.
        overlap_ratio (float): Fractional overlap between adjacent tiles.
        merge_overlaps (bool): Dissolve touching detections (``True``) or
            keep one asset per object.
        neck_width_m (float): Width below which masks are cut into pieces.
        min_detection_rate (float): Share of the tiles seeing an object that
            must have detected it.
        regularize (bool): Give the footprints straight, square walls.
    """

    building: bool
    overlap_ratio: float
    merge_overlaps: bool
    neck_width_m: float
    min_detection_rate: float
    regularize: bool


class SAM3OrthoFeatureExtractor:
    """
    Pipeline component that uses local SAM 3 to discover real-world assets.

    The extractor tiles a high-resolution orthomosaic raster into square
    patches of ``patch_size`` (in real-world units), runs SAM 3 on each batch
    of patches with the given text prompt, and maps the resulting pixel masks
    back to WGS84 polygons. Because neighboring tiles overlap, the same
    object is frequently detected more than once.

    For most prompts a final ``unary_union`` melts all detections that
    touch into a single geometry. That suits area features (roads,
    vegetation, water) but melts neighboring objects together: two houses
    whose masks meet, a house and its garage, cars parked side by side. Pass
    ``merge_overlaps=False`` for objects: each object then becomes one
    asset, fused from all the tiles that saw it (see
    :func:`~rapidtools.processing.instance_fusion.fuse_view_detections`).
    Two detections from different tiles are merged only when they agree
    inside the area both tiles covered, never two instances from the same
    tile, and the outline is the one most tiles agree on. Each asset carries
    the model's mean ``confidence`` and the number of tiles (``views``) that
    detected it.

    Prompts that name buildings (``'building'``, ``'house'``,
    ``'structure'``; see :func:`is_building_prompt`) get building defaults
    for every setting left at ``None``: objects are kept separate, tiles
    overlap by half so that every building is seen whole by several of
    them, ``neck_width_m=1.0`` cuts masks that bleed across the gap between
    two roofs, ``min_detection_rate=0.3`` drops buildings most of the tiles
    that saw them did not detect, and ``regularize=True`` gives the
    footprints straight, square walls with
    :class:`~rapidtools.processing.FootprintRegularizer`. For the best
    outlines, refine the result on close-up crops with
    :class:`~rapidtools.processing.AerialImageryExtractor` and
    :class:`~rapidtools.processing.BuildingRegularizer` (the GUI does both).
    Buildings are found best with tiles of about 50 m.

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
        overlap_ratio (float | None, optional):
            Fractional overlap between adjacent tiles (0.0 <= r < 1.0). Higher
            values reduce edge artifacts at the cost of more inference.
            Defaults to ``None``: 0.5 for building prompts, 0.15 otherwise.
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
        merge_overlaps (bool | None, optional):
            ``True`` dissolves all touching or overlapping detections into
            single polygons (area features). ``False`` fuses the views of
            each object into one asset with ``confidence`` and ``views``
            attributes and keeps neighbors separate (buildings, countable
            objects). Defaults to ``None``: ``False`` for building prompts,
            ``True`` otherwise.
        neck_width_m (float | None, optional):
            With ``merge_overlaps=False``, a mask whose pieces are joined
            only by a strip narrower than this many meters is cut into
            those pieces before fusion. ``0`` only separates pieces that do
            not touch. Defaults to ``None``: 1.0 for building prompts kept
            separate, 0 otherwise.
        min_detection_rate (float | None, optional):
            With ``merge_overlaps=False``, objects detected by fewer than
            this share of the tiles that had them whole in view are
            dropped, which removes detections that only one of several
            views produced. Defaults to ``None``: 0.3 for building prompts
            kept separate, 0 (keep all) otherwise.
        regularize (bool | None, optional):
            Give the detected footprints straight, square walls with
            :class:`~rapidtools.processing.FootprintRegularizer` (which also
            makes them non-overlapping and drops those under 4 m^2).
            Defaults to ``None``: ``True`` for building prompts, ``False``
            otherwise. Pass ``False`` for the raw mask outlines.
        cancel_event (threading.Event | None, optional):
            Cooperative cancellation flag checked before every tile. When set,
            :class:`~rapidtools.core.OperationCanceled` is raised. Defaults
            to ``None``.

    Settings left at ``None`` follow the prompt when the scan runs, so an
    extractor whose ``prompt`` is changed between scans (to reuse the loaded
    model) picks up the defaults of the new prompt; :meth:`scan_settings`
    shows the values in effect.

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
        ...     prompt='building', patch_size=50, unit='meters', batch_size=8
        ... )
        >>> extractor.scan_settings().regularize
        True
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
        overlap_ratio: float | None = None,
        model_id: str = 'facebook/sam3',
        device: str = 'auto',
        batch_size: int = 4,
        load_in_4bit: bool = True,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        max_missing_data_ratio: float = 0.95,
        merge_overlaps: bool | None = None,
        cancel_event: threading.Event | None = None,
        raster_path: str | Path | None = None,
        neck_width_m: float | None = None,
        min_detection_rate: float | None = None,
        regularize: bool | None = None,
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
            overlap_ratio (float | None, optional):
                Fractional overlap between adjacent tiles (``None``: by
                prompt).
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
            merge_overlaps (bool | None, optional):
                Dissolve touching detections (``True``) or keep one asset
                per object (``False``); ``None``: by prompt.
            cancel_event (threading.Event | None, optional):
                Cooperative cancellation flag checked before every tile.
            raster_path (str | Path | None, optional):
                Raster scanned when the extractor runs as a pipeline step.
            neck_width_m (float | None, optional):
                Width in meters below which masks are cut into pieces
                before fusion (instance mode only; ``None``: by prompt).
            min_detection_rate (float | None, optional):
                Minimum share of the tiles that saw an object whole that
                must have detected it (instance mode only; ``None``: by
                prompt).
            regularize (bool | None, optional):
                Square the footprint walls (``None``: for building prompts).

        Raises:
            ValueError: If ``neck_width_m`` is negative or
                ``min_detection_rate`` is outside ``[0, 1]``.
        """
        if neck_width_m is not None and neck_width_m < 0:
            raise ValueError('neck_width_m must not be negative.')
        if min_detection_rate is not None and not 0.0 <= min_detection_rate <= 1.0:
            raise ValueError('min_detection_rate must be between 0 and 1.')
        # Cooperative cancellation (stops between tiles):
        self.cancel_event = cancel_event
        # Optional default raster so the extractor can run as a pipeline step:
        self.raster_path = Path(raster_path) if raster_path is not None else None
        self.prompt = prompt
        self.patch_size = patch_size
        self.unit = unit
        self.batch_size = batch_size
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.max_missing_data_ratio = max_missing_data_ratio
        # Settings left at None follow the prompt (see scan_settings):
        self.overlap_ratio = overlap_ratio
        self.merge_overlaps = merge_overlaps
        self.neck_width_m = neck_width_m
        self.min_detection_rate = min_detection_rate
        self.regularize = regularize

        logger.info(
            f"Initializing SAM3OrthoFeatureExtractor for '{self.prompt}' "
            f'at scale: {self.patch_size} {self.unit} with threshold '
            f'{self.threshold}...'
        )

        self.model = _sam3_inference_class()(
            model_id=model_id, device=device, load_in_4bit=load_in_4bit
        )

    def scan_settings(self) -> ScanSettings:
        """
        The scan settings in effect for the current prompt.

        Settings set explicitly are returned as they are; those left at
        ``None`` take the defaults of the prompt (building defaults for
        building prompts, see the class description).

        Returns:
            ScanSettings: The overlap, merge, neck, detection-rate and
            regularization settings the next scan will use.

        Example:
            >>> extractor = SAM3OrthoFeatureExtractor('house', patch_size=50,
            ...                                       unit='meters')
            >>> s = extractor.scan_settings()
            >>> s.merge_overlaps, s.overlap_ratio, s.regularize
            (False, 0.5, True)
            >>> extractor.prompt = 'tree'
            >>> extractor.scan_settings().merge_overlaps
            True
        """
        building = is_building_prompt(self.prompt)
        merge = (not building) if self.merge_overlaps is None else self.merge_overlaps
        separate_buildings = building and not merge
        return ScanSettings(
            building=building,
            overlap_ratio=(
                (BUILDING_OVERLAP_RATIO if building else DEFAULT_OVERLAP_RATIO)
                if self.overlap_ratio is None
                else self.overlap_ratio
            ),
            merge_overlaps=merge,
            neck_width_m=(
                (BUILDING_NECK_WIDTH_M if separate_buildings else 0.0)
                if self.neck_width_m is None
                else self.neck_width_m
            ),
            min_detection_rate=(
                (BUILDING_MIN_DETECTION_RATE if separate_buildings else 0.0)
                if self.min_detection_rate is None
                else self.min_detection_rate
            ),
            regularize=building if self.regularize is None else self.regularize,
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
                One ``PhysicalAsset`` per merged feature (or per object when
                objects are kept separate). Each asset carries the
                attributes ``'asset_type'`` (the prompt), ``'source_model'``
                and ``'extraction_scale'``; objects also carry
                ``'confidence'`` (mean SAM 3 score of the fused detections,
                ``None`` if the model reported none) and ``'views'`` (how
                many tiles detected the object). With regularization (the
                default for building prompts) the outlines have straight,
                square walls. The collection is empty if nothing was
                detected.

        Raises:
            OperationCanceled:
                If ``cancel_event`` is set between two tiles.
            FileNotFoundError:
                If ``raster_path`` does not exist.

        Example:
            >>> extractor = SAM3OrthoFeatureExtractor(prompt='road')
            >>> roads = extractor('data/ortho.tif')
            >>> len(roads)
            12
        """
        settings = self.scan_settings()
        raw_polygons: list = []
        # Per-instance SAM 3 scores and tile numbers, only collected in
        # instance mode:
        raw_scores: list | None = None if settings.merge_overlaps else []
        raw_views: list | None = None if settings.merge_overlaps else []
        # Ground footprint (WGS84) of every tile, for the instance fusion:
        windows: dict[Hashable, BaseGeometry] = {}

        with OrthomosaicReader(raster_path) as reader:
            raster_footprint = self._raster_footprint(raster_path)
            tile_generator = reader.generate_tiles(
                patch_size=self.patch_size,
                unit=self.unit,
                overlap_ratio=settings.overlap_ratio,
                max_missing_data_ratio=self.max_missing_data_ratio,
                pad_edge_tiles=True,
                return_georef=True,
            )

            batch_images: list = []
            batch_bounds: list = []
            batch_georefs: list = []
            batch_views: list = []
            n_batches = n_failed = n_failed_in_a_row = 0

            def run_batch() -> None:
                """Segment the pending batch and keep count of failures."""
                nonlocal n_batches, n_failed, n_failed_in_a_row
                ok = self._process_batch(
                    batch_images,
                    batch_bounds,
                    raw_polygons,
                    raw_scores,
                    batch_georefs,
                    raw_views=raw_views,
                    batch_views=batch_views,
                )
                n_batches += 1
                if ok is not False:
                    n_failed_in_a_row = 0
                    return
                n_failed += 1
                n_failed_in_a_row += 1
                if n_failed_in_a_row >= MAX_FAILED_BATCHES_IN_A_ROW:
                    raise RuntimeError(
                        f'SAM 3 failed on {n_failed_in_a_row} batches in a row '
                        f"(see the errors above), so the scan for '{self.prompt}' "
                        'was stopped: the GPU may have run out of memory or '
                        'stopped responding.'
                    )

            for tile_number, (pil_image, wgs84_bounds, georef) in enumerate(
                tqdm(tile_generator, desc=f"Scanning for '{self.prompt}'")
            ):
                raise_if_canceled(self.cancel_event, f"scan for '{self.prompt}'")
                batch_images.append(pil_image)
                batch_bounds.append(wgs84_bounds)
                batch_georefs.append(georef)
                batch_views.append(tile_number)
                if raw_views is not None:
                    windows[tile_number] = self._tile_window(
                        georef, wgs84_bounds, raster_footprint
                    )

                if len(batch_images) == self.batch_size:
                    run_batch()
                    batch_images, batch_bounds, batch_georefs = [], [], []
                    batch_views = []

            # Process any remaining images in the final partial batch
            if batch_images:
                run_batch()

        if n_batches and n_failed == n_batches:
            raise RuntimeError(
                f"SAM 3 failed on every batch of the scan for '{self.prompt}' "
                '(see the errors above).'
            )
        if n_failed:
            logger.warning(
                f'SAM 3 failed on {n_failed} of {n_batches} batches (see the errors '
                'above); the tiles of those batches were not scanned.'
            )

        final_collection = PhysicalAssetCollection()

        if not settings.merge_overlaps:
            assert raw_scores is not None and raw_views is not None
            logger.info(
                f'Extracted {len(raw_polygons)} raw instances. Fusing the '
                'views of each object...'
            )
            for geom, score, n_views in self._fuse_instances(
                raw_polygons, raw_scores, raw_views, windows, settings
            ):
                final_collection.add(
                    PhysicalAsset(
                        id=f'{self.prompt.replace(" ", "_")}_{uuid.uuid4().hex[:8]}',
                        geometry=geom,
                        attributes={
                            'asset_type': self.prompt,
                            'confidence': score,
                            'views': n_views,
                            'source_model': self.model.model_id,
                            'extraction_scale': f'{self.patch_size}_{self.unit}',
                        },
                    )
                )
        elif raw_polygons:
            logger.info(
                f'Extracted {len(raw_polygons)} raw polygons. Merging overlaps...'
            )
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

        if settings.regularize and len(final_collection):
            final_collection = self._regularize(final_collection)

        logger.info(
            f'Extraction complete. Yielded {len(final_collection)} unique assets.'
        )
        return final_collection

    @staticmethod
    def _regularize(collection: PhysicalAssetCollection) -> PhysicalAssetCollection:
        """Give the detected footprints straight, square walls."""
        # Imported here: the post-processing package imports this module's
        # package, so a module-level import would be circular.
        from .postprocessing.footprints import FootprintRegularizer

        logger.info(f'Squaring the walls of {len(collection)} footprints...')
        return FootprintRegularizer()(collection)

    def _process_batch(
        self,
        batch_images: list,
        batch_bounds: list,
        raw_polygons: list,
        raw_scores: list | None = None,
        batch_georefs: list | None = None,
        raw_views: list | None = None,
        batch_views: list | None = None,
    ) -> bool:
        """
        Run inference on one batch of tiles and append polygons to ``raw_polygons``.

        The in-memory PIL tiles are written to temporary JPEG files (the model
        API expects paths), inference is run once for the whole batch, and
        each returned mask instance is vectorized in the raster's own pixel
        grid and reprojected to WGS84 (see
        :func:`~rapidtools.processing.vectorize.mask_to_wgs84_polygons`).
        Temporary files are always removed, even if inference raises.

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
                one geometry, and its confidence (``None`` if the model
                reported none) is appended here in step with
                ``raw_polygons``.
            batch_georefs (list[PatchGeoref] | None):
                Native georeferencing of each tile. When omitted, polygons are
                mapped through the WGS84 envelope in ``batch_bounds``, which
                is only exact for north-up WGS84 rasters.
            raw_views (list | None):
                Instance mode: the tile of every instance is appended here,
                in step with ``raw_polygons``, and an instance contributes
                all of its pieces (as one geometry) so the fusion can decide
                what they are. Without it, an instance contributes only its
                largest piece.
            batch_views (list | None):
                The tile identifier of every image in the batch (required
                with ``raw_views``).

        Returns:
            bool: ``False`` when the model failed on the batch (it returned
            no answer), ``True`` otherwise.
        """
        temp_paths = []

        # 1. Save in-memory PIL images to temporary files
        for img in batch_images:
            fd, path = tempfile.mkstemp(suffix='.jpg')
            os.close(fd)
            if max(img.size) > MAX_TILE_SIDE_PX:
                scale = MAX_TILE_SIDE_PX / max(img.size)
                img = img.resize(
                    (
                        max(1, round(img.width * scale)),
                        max(1, round(img.height * scale)),
                    ),
                    Image.Resampling.LANCZOS,
                )
            # Save as JPEG for fast disk I/O
            img.save(path, format='JPEG', quality=TILE_JPEG_QUALITY)
            temp_paths.append(path)

        try:
            # 2. Run inference using the file paths WITH threshold parameters
            outputs = self.model.run_inference(
                image_inputs=temp_paths,
                prompt=self.prompt,
                threshold=self.threshold,
                mask_threshold=self.mask_threshold,
            )

            if outputs is None:
                # The model logged the error and gave no answer:
                return False
            if outputs.masks is None:
                return True
            output_masks = outputs.masks

            raw_response = getattr(outputs, 'raw_response', None)
            scores_per_image = (
                raw_response.get('scores') if isinstance(raw_response, dict) else None
            ) or []

            # 3. Translate pixel masks to geographic polygons
            for image_index, (image_masks, bounds) in enumerate(
                zip(output_masks, batch_bounds, strict=False)
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

                georef = (
                    batch_georefs[image_index]
                    if batch_georefs is not None and image_index < len(batch_georefs)
                    else None
                )

                for instance_index, instance_mask in enumerate(image_masks):
                    # Vectorize in the raster's pixel grid and reproject the
                    # polygons to WGS84 (the envelope is only a fallback):
                    polygons = mask_to_wgs84_polygons(
                        instance_mask, georef=georef, wgs84_bounds=bounds
                    )

                    if raw_scores is None:
                        raw_polygons.extend(polygons)
                        continue
                    if not polygons:
                        continue
                    score = (
                        float(scores[instance_index])
                        if instance_index < len(scores)
                        else None
                    )
                    if raw_views is not None and batch_views is not None:
                        # Keep every piece: the fusion splits them into
                        # objects or drops the specks.
                        raw_polygons.append(unary_union(polygons))
                        raw_views.append(batch_views[image_index])
                    else:
                        # One instance is one object: keep its largest piece.
                        raw_polygons.append(max(polygons, key=lambda p: p.area))
                    raw_scores.append(score)
            return True
        finally:
            # 4. Always clean up temporary files to prevent disk bloat
            for path in temp_paths:
                if os.path.exists(path):
                    os.remove(path)

    @staticmethod
    def _raster_footprint(raster_path: str | Path) -> BaseGeometry:
        """
        The ground footprint of a raster as a WGS84 polygon.

        Unlike the WGS84 bounding box, this follows the raster's edges for
        projected and rotated rasters.
        """
        with rasterio.open(raster_path) as dataset:
            transform, width, height = dataset.transform, dataset.width, dataset.height
            crs = dataset.crs
        corners = [
            transform * (0, 0),
            transform * (width, 0),
            transform * (width, height),
            transform * (0, height),
        ]
        footprint = Polygon(corners)
        if crs is None or crs == CRS.from_epsg(4326):
            return footprint
        return shape(transform_geom(crs, CRS.from_epsg(4326), mapping(footprint)))

    @staticmethod
    def _tile_window(
        georef: PatchGeoref | None,
        wgs84_bounds: tuple[float, float, float, float],
        raster_footprint: BaseGeometry,
    ) -> BaseGeometry:
        """
        The part of the ground a tile shows, as a WGS84 polygon.

        Padding added to edge tiles is excluded by clipping the tile to the
        raster footprint.
        """
        if georef is None:
            tile: BaseGeometry = box(*wgs84_bounds)
        else:
            t = georef.transform
            corners = [
                t * (0, 0),
                t * (georef.width, 0),
                t * (georef.width, georef.height),
                t * (0, georef.height),
            ]
            tile = Polygon(corners)
            if georef.crs is not None and georef.crs != CRS.from_epsg(4326):
                tile = shape(
                    transform_geom(georef.crs, CRS.from_epsg(4326), mapping(tile))
                )
        clipped = tile.intersection(raster_footprint)
        return tile if clipped.is_empty else clipped

    def _fuse_instances(
        self,
        polygons: list,
        scores: list,
        views: list,
        windows: dict[Hashable, BaseGeometry],
        settings: ScanSettings | None = None,
    ) -> list[tuple[BaseGeometry, float | None, int]]:
        """
        Fuse the instances of all tiles into one outline per object.

        Geometries are projected to a transverse Mercator plane centered on
        the scanned area, fused with
        :func:`~rapidtools.processing.instance_fusion.fuse_view_detections`
        and projected back. ``settings`` defaults to :meth:`scan_settings`.

        Returns:
            list[tuple]: ``(wgs84_polygon, mean_confidence, n_views)`` per
            object.
        """
        if not polygons:
            return []
        if settings is None:
            settings = self.scan_settings()
        # Imported here: the post-processing package imports this module's
        # package, so a module-level import would be circular.
        from .postprocessing.footprints import local_metric_transformers

        center = unary_union(list(windows.values())).centroid
        to_local, to_wgs84 = local_metric_transformers(center.x, center.y)
        local_windows = {
            view: shapely_transform(to_local, window)
            for view, window in windows.items()
        }
        detections = [
            ViewDetection(shapely_transform(to_local, geom), view, score)
            for geom, score, view in zip(polygons, scores, views, strict=True)
        ]
        objects = fuse_view_detections(
            detections,
            local_windows,
            neck_width=settings.neck_width_m,
            min_area=MIN_INSTANCE_AREA_M2,
        )
        kept = [o for o in objects if o.detection_rate >= settings.min_detection_rate]
        if len(kept) < len(objects):
            logger.info(
                f'Dropped {len(objects) - len(kept)} objects detected by fewer than '
                f'{settings.min_detection_rate:.0%} of the tiles that saw them.'
            )
        return [
            (shapely_transform(to_wgs84, o.geometry), o.score, o.n_views) for o in kept
        ]
