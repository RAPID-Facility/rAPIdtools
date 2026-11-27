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
SAM 3 based refinement of preliminary building footprints.

This module provides :class:`BuildingRegularizer`, a pipeline component that
takes coarse building polygons (for example from OpenStreetMap, a parcel
layer or :class:`~rapidtools.processing.SAM3OrthoFeatureExtractor`) together
with the aerial crops attached by
:class:`~rapidtools.processing.AerialImageryExtractor`, and replaces each
preliminary shape with the roof outline segmented by SAM 3, drawn with
straight, square walls.

Every crop is a view of its building and of the neighbors around it, so a
building is usually segmented in several crops. The views are fused with
:func:`~rapidtools.processing.instance_fusion.fuse_view_detections`, which
keeps neighbors that touch apart, and the fused outlines are regularized
with :func:`~rapidtools.processing.postprocessing.footprints.regularize_footprints`.

Example:
    >>> from rapidtools.core import PhysicalAssetCollection
    >>> from rapidtools.processing import (
    ...     AerialImageryExtractor, BuildingRegularizer
    ... )
    >>> prelim = PhysicalAssetCollection.from_geojson('osm_buildings.geojson')
    >>> prelim = AerialImageryExtractor('ortho.tif', 'crops')(prelim)
    >>> refined = BuildingRegularizer(prompt='building roof')(prelim)
    >>> refined.to_geojson('refined_buildings.geojson')
"""

import logging
import threading
import uuid
import warnings
from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING, Any

import numpy as np
import pyproj
from rasterio.crs import CRS
from rasterio.warp import transform_geom
from shapely.geometry import Polygon, box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union
from shapely.strtree import STRtree

from rapidtools.core import (
    ImageAsset,
    PhysicalAsset,
    PhysicalAssetCollection,
    raise_if_canceled,
)
from rapidtools.data_sources.orthomosaic_reader import PatchGeoref
from rapidtools.processing.image_segmenters import SAM3ImageSegmenter
from rapidtools.processing.instance_fusion import (
    FusedObject,
    ViewDetection,
    fuse_view_detections,
)
from rapidtools.processing.step import Stage
from rapidtools.processing.vectorize import mask_to_wgs84_polygons

from .footprints import (
    DEFAULT_MIN_AREA_M2,
    local_metric_transformers,
    regularize_footprints,
    resolve_overlaps,
)

if TYPE_CHECKING:
    from rapidtools.models import SAM3Inference

logger = logging.getLogger(__name__)

# Pipeline configuration:
ASSET_TYPE = 'building'
DEFAULT_PROMPT = 'building'
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MASK_THRESHOLD = 0.5
DEFAULT_RETRY_CONFIDENCE_THRESHOLD = 0.3
DEFAULT_RETRY_MASK_THRESHOLD = 0.3
# Kept for backward compatibility; neither value is used any more:
DEFAULT_MAX_GAP_BRIDGE_FT = 15.0
DEFAULT_SNAP_TOLERANCE_FT = 0.5
# Masks bleeding across a gap narrower than this are cut (meters):
DEFAULT_NECK_WIDTH_M = 1.0
# Longest side of the stored SAM 3 masks, in pixels (see SAM3ImageSegmenter):
DEFAULT_MAX_MASK_SIDE = 1024
# Assets segmented per chunk; masks are vectorized and released after each:
_SEGMENT_CHUNK = 64
# A building recovered in the retry pass is discarded when this share of it
# already belongs to a building from the first pass:
_RETRY_MAX_OVERLAP = 0.3


class BuildingRegularizer:
    """
    Pipeline component to regularize and refine preliminary building polygons.

    This regularizer takes assets that have preliminary building geometries
    and associated cropped imagery, segments the crops with SAM 3, and
    returns one clean footprint per building:

    1. The masks of every crop are traced into polygons; a mask whose
       pieces are separate, or joined only by a strip narrower than
       ``neck_width_m``, becomes several candidates.
    2. Crops overlap (a building appears in its own crop and in those of
       its neighbors), so the candidates are fused across crops: two are
       the same building only if they agree where both crops could see,
       and two instances from the same crop are never merged. The outline
       is the one most crops agree on.
    3. Fused buildings are kept if they match a preliminary footprint (IoU
       of at least ``min_overlap_ratio``, or covering at least half of it,
       or lying at least half inside it). A preliminary footprint that
       covered two buildings therefore yields both.
    4. Preliminary footprints without a match get a second, low-threshold
       pass, which rescues many buildings in shadow.
    5. The footprints get straight, square walls (see
       :class:`~rapidtools.processing.FootprintRegularizer`) and are made
       mutually non-overlapping.

    Every image attached to an input asset must carry a ``'native_georef'``
    (preferred) or ``'wgs84_bounds'`` property, as written by
    :class:`~rapidtools.processing.AerialImageryExtractor`; images without
    either are ignored.

    Args:
        prompt (str | list[str], optional):
            The text prompt(s) used for SAM 3 segmentation (e.g.,
            ``'building roof'``). Lists are joined with ``'. '``. Defaults to
            ``'building'``.
        batch_size (int, optional):
            Batch size for SAM 3 inference. Defaults to 4.
        load_in_4bit (bool, optional):
            Whether to load the SAM 3 model in 4-bit precision to save memory.
            Defaults to ``True``.
        min_overlap_ratio (float, optional):
            Minimum Intersection over Union (IoU) between a building and a
            preliminary footprint for the building to be kept (it is also
            kept when it covers at least half of the footprint or lies at
            least half inside it). Defaults to 0.15.
        max_gap_bridge_ft (float | None, optional):
            Deprecated and ignored. Gaps inside a mask are no longer
            bridged: bridging is what merged neighboring buildings.
        confidence_threshold (float, optional):
            Primary confidence threshold for SAM 3 mask generation. Defaults
            to 0.5.
        mask_threshold (float, optional):
            Primary threshold for SAM 3 mask binarization. Defaults to 0.5.
        retry_dropped (bool, optional):
            Whether to run a second pass with lower thresholds on dropped
            assets. Defaults to ``True``.
        retry_confidence_threshold (float, optional):
            Lower confidence threshold for the retry pass. Defaults to 0.3.
        retry_mask_threshold (float, optional):
            Lower mask threshold for the retry pass. Defaults to 0.3.
        snap_tolerance_ft (float | None, optional):
            Deprecated and ignored.
        cancel_event (threading.Event | None, optional):
            Cooperative cancellation flag shared with the internal segmenter.
            When set, :class:`~rapidtools.core.OperationCanceled` is raised
            between SAM 3 batches or between pipeline stages. Defaults to
            ``None``.
        regularize_outlines (bool, optional):
            Give the footprints straight, square walls. With ``False`` the
            fused mask outlines are returned (still non-overlapping).
            Defaults to ``True``.
        neck_width_m (float, optional):
            Masks are cut where they narrow below this width in meters.
            Defaults to 1.0.
        min_area_m2 (float, optional):
            Footprints smaller than this are dropped. Defaults to 4.
        max_mask_side (int | None, optional):
            Longest side, in pixels, at which SAM 3 masks are kept (see
            :class:`~rapidtools.processing.SAM3ImageSegmenter`). Defaults to
            1024, which keeps the crops of centimeter-resolution drone
            orthomosaics from exhausting memory.
        model (SAM3Inference | None, optional):
            An already loaded SAM 3, for example the one of the
            :class:`~rapidtools.processing.SAM3OrthoFeatureExtractor` that
            found the preliminary footprints, to use instead of loading a
            second copy (``load_in_4bit`` is then ignored). Defaults to
            ``None``.

    Example:
        >>> from rapidtools.processing import (
        ...     AerialImageryExtractor, BuildingRegularizer, Pipeline
        ... )
        >>> pipeline = Pipeline([
        ...     AerialImageryExtractor('ortho.tif', 'crops', buffer_asset='30%'),
        ...     BuildingRegularizer(prompt=['building', 'roof'], batch_size=8),
        ... ])
        >>> refined = pipeline.run(prelim_buildings)
        >>> refined[0].attributes['asset_type']
        'building_footprint'
    """

    stage = Stage.REGULARIZE

    def __init__(
        self,
        prompt: str | list[str] = DEFAULT_PROMPT,
        batch_size: int = 4,
        load_in_4bit: bool = True,
        min_overlap_ratio: float = 0.15,
        max_gap_bridge_ft: float | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        mask_threshold: float = DEFAULT_MASK_THRESHOLD,
        retry_dropped: bool = True,
        retry_confidence_threshold: float = DEFAULT_RETRY_CONFIDENCE_THRESHOLD,
        retry_mask_threshold: float = DEFAULT_RETRY_MASK_THRESHOLD,
        snap_tolerance_ft: float | None = None,
        cancel_event: threading.Event | None = None,
        regularize_outlines: bool = True,
        neck_width_m: float = DEFAULT_NECK_WIDTH_M,
        min_area_m2: float = DEFAULT_MIN_AREA_M2,
        max_mask_side: int | None = DEFAULT_MAX_MASK_SIDE,
        model: 'SAM3Inference | None' = None,
    ) -> None:
        """
        Initialize the BuildingRegularizer with SAM 3 and pipeline thresholds.

        Args:
            prompt (str | list[str], optional):
                The text prompt(s) used for SAM 3 segmentation.
            batch_size (int, optional):
                Batch size for SAM 3 inference.
            load_in_4bit (bool, optional):
                Whether to load the SAM 3 model in 4-bit precision.
            min_overlap_ratio (float, optional):
                Minimum IoU with a preliminary footprint to keep a building.
            max_gap_bridge_ft (float | None, optional):
                Deprecated and ignored.
            confidence_threshold (float, optional):
                Primary confidence threshold for SAM 3 mask generation.
            mask_threshold (float, optional):
                Primary threshold for SAM 3 mask binarization.
            retry_dropped (bool, optional):
                Whether to re-run dropped assets with lower thresholds.
            retry_confidence_threshold (float, optional):
                Lower confidence threshold for the retry pass.
            retry_mask_threshold (float, optional):
                Lower mask threshold for the retry pass.
            snap_tolerance_ft (float | None, optional):
                Deprecated and ignored.
            cancel_event (threading.Event | None, optional):
                Cooperative cancellation flag.
            regularize_outlines (bool, optional):
                Give the footprints straight, square walls.
            neck_width_m (float, optional):
                Width below which masks are cut into pieces, in meters.
            min_area_m2 (float, optional):
                Smallest footprint kept, in square meters.
            max_mask_side (int | None, optional):
                Longest side of the stored SAM 3 masks, in pixels.
            model (SAM3Inference | None, optional):
                An already loaded SAM 3 to share instead of loading one.
        """
        for name, value in (
            ('max_gap_bridge_ft', max_gap_bridge_ft),
            ('snap_tolerance_ft', snap_tolerance_ft),
        ):
            if value is not None:
                warnings.warn(
                    f'BuildingRegularizer ignores {name}; it will be removed in '
                    'a future release.',
                    DeprecationWarning,
                    stacklevel=2,
                )
        # Cooperative cancellation (stops between SAM 3 batches):
        self.cancel_event = cancel_event
        self.min_overlap_ratio = min_overlap_ratio
        self.max_gap_bridge_ft = (
            DEFAULT_MAX_GAP_BRIDGE_FT
            if max_gap_bridge_ft is None
            else max_gap_bridge_ft
        )
        self.snap_tolerance_ft = (
            DEFAULT_SNAP_TOLERANCE_FT
            if snap_tolerance_ft is None
            else snap_tolerance_ft
        )
        self.retry_dropped = retry_dropped
        self.retry_confidence_threshold = retry_confidence_threshold
        self.retry_mask_threshold = retry_mask_threshold
        self.regularize_outlines = regularize_outlines
        self.neck_width_m = neck_width_m
        self.min_area_m2 = min_area_m2

        # Safely handle lists by converting them to the period-separated string
        # format:
        if isinstance(prompt, list):
            safe_prompt = '. '.join(prompt)
        else:
            safe_prompt = prompt

        self.segmenter = SAM3ImageSegmenter(
            prompt=safe_prompt,
            batch_size=batch_size,
            load_in_4bit=load_in_4bit,
            threshold=confidence_threshold,
            mask_threshold=mask_threshold,
            cancel_event=cancel_event,
            max_mask_side=max_mask_side,
            model=model,
        )

    def __call__(
        self, input_assets: PhysicalAssetCollection
    ) -> PhysicalAssetCollection:
        """
        Allow the regularizer instance to be called as a function.

        Args:
            input_assets (PhysicalAssetCollection):
                The preliminary building assets to regularize.

        Returns:
            PhysicalAssetCollection:
                The fully processed and refined building assets.

        Example:
            >>> regularizer = BuildingRegularizer()
            >>> refined = regularizer(prelim_buildings)
        """
        return self.process(input_assets)

    def process(self, input_assets: PhysicalAssetCollection) -> PhysicalAssetCollection:
        """
        Execute the full building regularization pipeline.

        This pipeline consists of:

            1. Segmenting the image crops with SAM 3 and tracing the masks.
            2. Fusing the crops' views of every building and matching the
               buildings to the preliminary footprints.
            3. (Optional) Retrying dropped assets with lower thresholds.
            4. Giving the footprints straight walls and removing overlaps.
            5. Removing degenerate geometries.

        Args:
            input_assets (PhysicalAssetCollection):
                Collection of preliminary building assets.

        Returns:
            PhysicalAssetCollection:
                Collection of refined, non-overlapping building assets. The
                output assets carry new IDs (``'bldg_merged_<hash>'``) and
                the attributes ``'asset_type': 'building_footprint'``,
                ``'confidence'`` (mean SAM 3 score, ``None`` when the model
                reported none), ``'views'`` (number of crops that detected
                the building) and ``'source_ids'`` (IDs of the matched
                preliminary assets). Returns an empty collection when
                ``input_assets`` is empty.

        Raises:
            OperationCanceled:
                If ``cancel_event`` is set during processing.

        Example:
            >>> refined = BuildingRegularizer(retry_dropped=False).process(prelim)
            >>> len(refined) <= len(prelim) + 5
            True
        """
        logger.info('Starting Building Regularization Pipeline...')

        if not input_assets:
            logger.warning('No assets provided to BuildingRegularizer. Exiting.')
            return PhysicalAssetCollection()

        seeds = self._seed_geometries(input_assets)
        if not seeds:
            logger.warning('No polygonal preliminary footprints to regularize.')
            return PhysicalAssetCollection()
        center = unary_union(list(seeds.values())).centroid
        to_local, to_wgs84 = local_metric_transformers(center.x, center.y)
        local_seeds = {k: shapely_transform(to_local, g) for k, g in seeds.items()}

        # Pass 1:
        logger.info('Step 1/5: Running SAM 3 segmentation on image crops...')
        detections, windows = self._segment(input_assets, to_local)
        raise_if_canceled(self.cancel_event, 'building regularization')

        logger.info('Step 2/5: Fusing overlapping crops and matching footprints...')
        accepted = self._match_seeds(self._fuse(detections, windows), local_seeds)

        # Pass 2: Retry dropped assets with lower thresholds:
        if self.retry_dropped:
            matched = {seed_id for _, ids in accepted for seed_id in ids}
            dropped_assets = PhysicalAssetCollection(
                [a for a in input_assets if a.id in seeds and a.id not in matched]
            )
            if len(dropped_assets) > 0:
                logger.info(
                    f'Step 3/5: Retrying {len(dropped_assets)} dropped assets '
                    f'with lower thresholds...'
                )
                recovered = self._retry(dropped_assets, local_seeds, accepted, to_local)
                logger.info(
                    f'Pass 2 recovered {len(recovered)} previously dropped assets.'
                )
                accepted.extend(recovered)
            else:
                logger.info('Step 3/5: No dropped assets to retry.')
        else:
            logger.info('Step 3/5: Retry step skipped (retry_dropped=False).')

        raise_if_canceled(self.cancel_event, 'building regularization')
        logger.info('Step 4/5: Regularizing outlines and separating neighbors...')
        outlines = [obj.geometry for obj, _ in accepted]
        priorities = [
            obj.n_views * 1e3 + (obj.score or 0.0) * 1e2 + obj.geometry.area * 1e-4
            for obj, _ in accepted
        ]
        if self.regularize_outlines:
            outlines = regularize_footprints(
                outlines, priorities=priorities, min_area=self.min_area_m2
            )
        else:
            outlines = resolve_overlaps(
                outlines, priorities=priorities, min_area=self.min_area_m2
            )

        final_collection = PhysicalAssetCollection()
        for (obj, seed_ids), outline in zip(accepted, outlines, strict=True):
            if outline is None:
                continue
            final_collection.add(
                PhysicalAsset(
                    id=f'bldg_merged_{uuid.uuid4().hex[:8]}',
                    geometry=shapely_transform(to_wgs84, outline),
                    attributes={
                        'asset_type': 'building_footprint',
                        'confidence': obj.score,
                        'views': obj.n_views,
                        'source_ids': sorted(str(s) for s in seed_ids),
                    },
                )
            )

        # Remove degenerate shapes:
        logger.info('Step 5/5: Filtering out degenerate geometries...')
        final_collection = self._filter_degenerate_assets(final_collection)

        # Default undefined asset types to 'building':
        final_collection.set_asset_type(ASSET_TYPE, overwrite=False)

        logger.info(
            f'Building regularization complete: {len(final_collection)} footprints.'
        )
        return final_collection

    # ------------------------------------------------------------ stages
    @staticmethod
    def _seed_geometries(assets: PhysicalAssetCollection) -> dict[Hashable, Polygon]:
        """
        WGS84 polygons of the preliminary footprints, keyed by asset ID.

        Preliminary geometries in Web Mercator (EPSG:3857, recognizable by
        coordinates outside the WGS84 range) are reprojected on the fly.
        Non-polygonal and empty geometries are skipped.
        """
        to_wgs84 = pyproj.Transformer.from_crs(
            'EPSG:3857', 'EPSG:4326', always_xy=True
        ).transform
        seeds: dict[Hashable, Polygon] = {}
        for asset in assets:
            geom = asset.geometry
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type not in ('Polygon', 'MultiPolygon'):
                continue
            minx, miny, maxx, maxy = geom.bounds
            if minx < -180 or maxx > 180 or miny < -90 or maxy > 90:
                geom = shapely_transform(to_wgs84, geom)
            if not geom.is_valid:
                geom = geom.buffer(0)
            if not geom.is_empty:
                seeds[asset.id] = geom
        return seeds

    def _segment(
        self,
        assets: PhysicalAssetCollection,
        to_local: Callable[..., Any],
    ) -> tuple[list[ViewDetection], dict[Hashable, BaseGeometry]]:
        """
        Segment the crops and trace every mask into a candidate outline.

        Assets are segmented in chunks and their masks are released right
        after tracing, so memory stays flat however many buildings there are.

        Returns:
            tuple: The candidates (planar CRS) and the ground window of each
            crop, keyed by image ID.
        """
        detections: list[ViewDetection] = []
        windows: dict[Hashable, BaseGeometry] = {}
        asset_list = list(assets)
        for start in range(0, len(asset_list), _SEGMENT_CHUNK):
            chunk = PhysicalAssetCollection(asset_list[start : start + _SEGMENT_CHUNK])
            self.segmenter(chunk)
            for asset in chunk:
                masks_by_image = asset.attributes.pop('sam3_masks', None) or {}
                scores_by_image = asset.attributes.pop('sam3_scores', None) or {}
                asset.attributes.pop('sam3_bounding_boxes', None)
                for img_asset in asset.image_assets:
                    if img_asset.id is None or img_asset.id not in masks_by_image:
                        continue
                    traced = self._trace_image(
                        img_asset,
                        masks_by_image[img_asset.id],
                        scores_by_image.get(img_asset.id),
                    )
                    if traced is None:
                        continue
                    window, instances = traced
                    view = img_asset.id
                    windows[view] = shapely_transform(to_local, window)
                    detections.extend(
                        ViewDetection(shapely_transform(to_local, geom), view, score)
                        for geom, score in instances
                    )
        return detections, windows

    @staticmethod
    def _image_georef(img_asset: ImageAsset) -> PatchGeoref | None:
        """The crop's native georeference, or ``None`` if missing or unreadable."""
        georef_dict = img_asset.properties.get('native_georef')
        if not georef_dict:
            return None
        try:
            return PatchGeoref.from_dict(georef_dict)
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(
                f"Ignoring unreadable 'native_georef' on image '{img_asset.id}': {e}"
            )
            return None

    def _trace_image(
        self, img_asset: ImageAsset, masks: Any, scores: Any
    ) -> tuple[BaseGeometry, list[tuple[BaseGeometry, float | None]]] | None:
        """
        The ground window of one crop and its instance outlines (WGS84).

        Returns ``None`` for crops without georeferencing.
        """
        wgs84_bounds = img_asset.properties.get('wgs84_bounds')
        georef = self._image_georef(img_asset)
        window: BaseGeometry
        if georef is not None:
            t = georef.transform
            corners = Polygon(
                [
                    t * (0, 0),
                    t * (georef.width, 0),
                    t * (georef.width, georef.height),
                    t * (0, georef.height),
                ]
            )
            window = (
                corners
                if georef.crs is None or georef.crs == CRS.from_epsg(4326)
                else shape(
                    transform_geom(georef.crs, CRS.from_epsg(4326), mapping(corners))
                )
            )
        elif wgs84_bounds:
            min_lon, min_lat, max_lon, max_lat = wgs84_bounds
            window = box(min_lon, min_lat, max_lon, max_lat)
        else:
            return None

        masks = np.asarray(masks)
        if masks.size == 0:
            return window, []
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]
        score_list = list(scores) if scores is not None else []
        instances = []
        for index, instance_mask in enumerate(masks):
            parts = mask_to_wgs84_polygons(
                instance_mask, georef=georef, wgs84_bounds=wgs84_bounds
            )
            if not parts:
                continue
            score = float(score_list[index]) if index < len(score_list) else None
            instances.append((unary_union(parts), score))
        return window, instances

    def _fuse(
        self,
        detections: list[ViewDetection],
        windows: dict[Hashable, BaseGeometry],
    ) -> list[FusedObject]:
        """Fuse the candidates of all crops into one outline per building."""
        if not detections:
            return []
        return fuse_view_detections(
            detections,
            windows,
            neck_width=self.neck_width_m,
            min_area=min(self.min_area_m2, 3.0),
        )

    def _match_seeds(
        self,
        objects: list[FusedObject],
        local_seeds: dict[Hashable, BaseGeometry],
    ) -> list[tuple[FusedObject, list[Hashable]]]:
        """
        Keep the buildings that match a preliminary footprint.

        A building matches a footprint when their IoU reaches
        ``min_overlap_ratio``, when it covers at least half of the
        footprint, or when at least half of it lies inside the footprint
        (a footprint that covered several buildings matches each of them).
        Buildings that match no footprint are neighbors that only appear at
        the edge of the crops and are dropped.
        """
        if not objects or not local_seeds:
            return []
        seed_ids = list(local_seeds)
        seed_geoms = [local_seeds[k] for k in seed_ids]
        tree = STRtree(seed_geoms)
        accepted = []
        for obj in objects:
            geom = obj.geometry
            ids = []
            for j in tree.query(geom, predicate='intersects'):
                seed = seed_geoms[int(j)]
                inter = geom.intersection(seed).area
                if inter <= 0:
                    continue
                union = geom.area + seed.area - inter
                if (
                    (union > 0 and inter / union >= self.min_overlap_ratio)
                    or inter >= 0.5 * seed.area
                    or inter >= 0.5 * geom.area
                ):
                    ids.append(seed_ids[int(j)])
            if ids:
                accepted.append((obj, ids))
            else:
                logger.debug('A segmented building matched no preliminary footprint.')
        return accepted

    def _retry(
        self,
        dropped_assets: PhysicalAssetCollection,
        local_seeds: dict[Hashable, BaseGeometry],
        accepted: list[tuple[FusedObject, list[Hashable]]],
        to_local: Callable[..., Any],
    ) -> list[tuple[FusedObject, list[Hashable]]]:
        """Segment dropped assets at lower thresholds and keep new buildings."""
        orig_conf = self.segmenter.threshold
        orig_mask = self.segmenter.mask_threshold
        self.segmenter.threshold = self.retry_confidence_threshold
        self.segmenter.mask_threshold = self.retry_mask_threshold
        try:
            detections, windows = self._segment(dropped_assets, to_local)
        finally:
            self.segmenter.threshold = orig_conf
            self.segmenter.mask_threshold = orig_mask
        dropped_seeds: dict[Hashable, BaseGeometry] = {
            a.id: local_seeds[a.id] for a in dropped_assets if a.id in local_seeds
        }
        candidates = self._match_seeds(self._fuse(detections, windows), dropped_seeds)
        existing = [obj.geometry for obj, _ in accepted]
        tree = STRtree(existing) if existing else None
        recovered = []
        for obj, ids in candidates:
            geom = obj.geometry
            if tree is not None:
                taken = sum(
                    existing[int(j)].intersection(geom).area
                    for j in tree.query(geom, predicate='intersects')
                )
                if taken > _RETRY_MAX_OVERLAP * geom.area:
                    continue
            recovered.append((obj, ids))
        return recovered

    def _filter_degenerate_assets(
        self, asset_collection: PhysicalAssetCollection
    ) -> PhysicalAssetCollection:
        """
        Remove geometries that collapsed into degenerate shapes.

        An asset is discarded when its geometry is missing or empty, invalid
        and not healable with a zero buffer, not a ``Polygon`` /
        ``MultiPolygon``, has (near) zero area, or has a minimum rotated
        rectangle that is not a polygon (i.e., it is effectively a line).
        Invalid geometries that heal successfully are kept with the healed
        geometry written back onto the asset.

        Args:
            asset_collection (PhysicalAssetCollection):
                Assets to validate.

        Returns:
            PhysicalAssetCollection:
                The surviving assets (same objects, not copies).
        """
        clean_collection = PhysicalAssetCollection()
        discard_count = 0

        for asset in asset_collection:
            geom = asset.geometry

            # 1. Missing or Empty
            if geom is None or geom.is_empty:
                discard_count += 1
                continue

            # 2. Invalid Topology (e.g., self-intersecting "bowties")
            if not geom.is_valid:
                # Try a quick zero-buffer heal
                geom = geom.buffer(0)
                if not geom.is_valid or geom.is_empty:
                    discard_count += 1
                    continue
                asset.geometry = geom  # Save the healed geometry

            # 3. Wrong base type
            if geom.geom_type not in ['Polygon', 'MultiPolygon']:
                discard_count += 1
                continue

            # 4. Effectively zero area (tiny slivers)
            if geom.area < 1e-9:
                discard_count += 1
                continue

            # 5. Collapsed Minimum Bounding Rectangle (The LineString bug!)
            mbr = geom.minimum_rotated_rectangle
            if mbr.geom_type != 'Polygon':
                discard_count += 1
                continue

            # If it survives all checks, it is a healthy, valid polygon!
            clean_collection.add(asset)

        if discard_count > 0:
            logger.info(
                f'Discarded {discard_count} degenerate geometries during final cleanup.'
            )

        return clean_collection
