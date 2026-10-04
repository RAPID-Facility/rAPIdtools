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
# Contributors:
# Barbaros Cetiner
#
# Last updated:
# 09-29-2026

"""
Discover and photograph objects along a street-level survey.

Two pipeline components mirror the aerial detect / extract stages for
street-level imagery on Mapillary:

- :class:`MapillaryFeatureExtractor` (``DETECT``) finds objects of the
  requested classes in every image of a region and places them on the ground.
  Mapillary's own segmentation supplies the detections as metadata, so no
  pixels are downloaded; SAM 3 is used only for classes the Mapillary
  vocabulary lacks. Each object becomes a point
  :class:`~rapidtools.core.PhysicalAsset` carrying its
  :class:`~rapidtools.core.Observation` list.
- :class:`MapillaryObjectImageExtractor` (``EXTRACT_IMAGERY``) downloads the
  best one or two images per object and crops the detection, attaching the
  crops as :class:`~rapidtools.core.ImageAsset` objects so that
  :class:`~rapidtools.processing.AssetAnalyzer` can judge each object.

Example:
    >>> from rapidtools.core import BoundingBox, PhysicalAssetCollection
    >>> from rapidtools.processing import (
    ...     MapillaryFeatureExtractor, MapillaryObjectImageExtractor, Pipeline
    ... )
    >>> spokane = BoundingBox(-117.5358, 47.6885, -117.4426, 47.7378)
    >>> pipeline = Pipeline([
    ...     MapillaryFeatureExtractor(
    ...         classes=['vehicles'], access_token='MLY|...', region=spokane,
    ...         start_date='2025-08-01', end_date='2025-09-30',
    ...     ),
    ...     MapillaryObjectImageExtractor(
    ...         'output/vehicle_crops', access_token='MLY|...'
    ...     ),
    ... ])
    >>> vehicles = pipeline.run(PhysicalAssetCollection())  # doctest: +SKIP
    >>> vehicles[0].attributes['n_observations']  # doctest: +SKIP
    4
"""

from __future__ import annotations

import logging
import math
import threading
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from shapely.geometry import Point
from tqdm import tqdm

from rapidtools.config import REQUESTS_TIMEOUT_VAL
from rapidtools.core import (
    BoundingBox,
    ImageAsset,
    Observation,
    OperationCancelled,
    PhysicalAsset,
    PhysicalAssetCollection,
    PolygonRegion,
    raise_if_cancelled,
)
from rapidtools.data_sources import MapillaryClient, MapillaryLabels
from rapidtools.data_sources.mapillary_client import is_map_feature_value

from . import outlines
from .step import Stage
from .street_localization import (
    DEFAULT_MAX_RANGE_M,
    MIN_POLYGON_AREA_FRACTION,
    cluster_observations,
    estimate_ego_mask,
    intersect_bearings,
    local_projection,
    localize,
    simplify_polygon,
    thin_frames,
)
from .street_tracking import discover_objects, estimate_attributes

logger = logging.getLogger(__name__)

#: Mapillary labels that describe motor vehicles.
#: Motor vehicles. Trailers, caravans, boats, bicycles and Mapillary's
#: catch-all "other vehicle" (jet skis, equipment, towed loads) are not
#: vehicles in the sense a damage survey means; ask for them by name.
VEHICLE_LABELS: tuple[str, ...] = (
    'object--vehicle--car',
    'object--vehicle--truck',
    'object--vehicle--bus',
    'object--vehicle--motorcycle',
)
#: Everything Mapillary files under ``object--vehicle--*`` except bicycles
#: and rail vehicles, for callers who really want all of it.
ALL_VEHICLE_LABELS: tuple[str, ...] = VEHICLE_LABELS + (
    'object--vehicle--other-vehicle',
    'object--vehicle--vehicle-group',
    'object--vehicle--trailer',
    'object--vehicle--caravan',
    'object--vehicle--boat',
)

# Plain-English class names that resolve without a language model. Keys are
# lower-case singular forms; simple plurals are stripped before lookup.
_CLASS_ALIASES: dict[str, tuple[str, ...]] = {
    'vehicle': VEHICLE_LABELS,
    'motor vehicle': VEHICLE_LABELS,
    'all vehicle': ALL_VEHICLE_LABELS,
    'any vehicle': ALL_VEHICLE_LABELS,
    'car': ('object--vehicle--car',),
    'truck': ('object--vehicle--truck',),
    'bus': ('object--vehicle--bus',),
    'trailer': ('object--vehicle--trailer', 'object--vehicle--caravan'),
    'caravan': ('object--vehicle--caravan',),
    'rv': ('object--vehicle--caravan',),
    'motorcycle': ('object--vehicle--motorcycle',),
    'bicycle': ('object--vehicle--bicycle',),
    'bike': ('object--vehicle--bicycle',),
    'boat': ('object--vehicle--boat',),
    'pole': ('object--support--pole', 'object--support--utility-pole'),
    'utility pole': ('object--support--utility-pole',),
    'power pole': ('object--support--utility-pole',),
    'street light': ('object--street-light',),
    'streetlight': ('object--street-light',),
    'traffic light': ('object--traffic-light--general',),
    'traffic sign': ('object--traffic-sign--front', 'object--traffic-sign--back'),
    'fire hydrant': ('object--fire-hydrant',),
    'hydrant': ('object--fire-hydrant',),
    'bench': ('object--bench',),
    'trash can': ('object--trash-can',),
    'mailbox': ('object--mailbox',),
    'manhole': ('object--manhole',),
    'catch basin': ('object--catch-basin',),
    'person': ('human--person--individual',),
    'pedestrian': ('human--person--individual',),
    'building': ('construction--structure--building',),
    'bridge': ('construction--structure--bridge',),
    'tunnel': ('construction--structure--tunnel',),
    'fence': ('construction--barrier--fence',),
    'guard rail': ('construction--barrier--guard-rail',),
    'wall': ('construction--barrier--wall',),
}

# Metadata fetched for every image in the region. Detections are added when
# the Mapillary detection source is active.
_IMAGE_FIELDS: tuple[str, ...] = (
    'captured_at',
    'compass_angle',
    'computed_compass_angle',
    'geometry',
    'computed_geometry',
    'is_pano',
    'camera_type',
    'camera_parameters',
    'computed_rotation',
    'computed_altitude',
    'width',
    'height',
    'sequence',
    'thumb_1024_url',
    'thumb_2048_url',
)
_DETECTION_FIELDS: tuple[str, ...] = ('detections.value', 'detections.geometry')

_PANO_TYPES = ('spherical', 'equirectangular')
# ``simplify_tolerance`` is capped at this share of an outline's longest side.
_RELATIVE_TOLERANCE = 0.01


def mapillary_vocabulary() -> frozenset[str]:
    """
    Every label in :class:`~rapidtools.data_sources.MapillaryLabels`.

    Example:
        >>> 'object--vehicle--car' in mapillary_vocabulary()
        True
    """
    return frozenset(
        value
        for name, value in vars(MapillaryLabels).items()
        if not name.startswith('_') and isinstance(value, str) and '--' in value
    )


def resolve_classes(
    classes: Iterable[str], label_mapper: Any = None
) -> tuple[dict[str, tuple[str, ...]], list[str]]:
    """
    Translate user class names into Mapillary labels.

    Resolution order: an exact Mapillary label, a built-in alias
    (``'cars'`` -> ``object--vehicle--car``), then ``label_mapper`` (a
    :class:`~rapidtools.processing.MapillaryLabelMapper`) when given.

    Args:
        classes: Class names as the user typed them.
        label_mapper: Optional object with ``map_classes(list[str])``.

    Returns:
        tuple[dict[str, tuple[str, ...]], list[str]]:
            ``(resolved, unresolved)`` where ``resolved`` maps each class to
            its Mapillary labels and ``unresolved`` lists the classes no
            source could translate.

    Example:
        >>> resolved, unresolved = resolve_classes(['Cars', 'debris pile'])
        >>> resolved
        {'Cars': ('object--vehicle--car',)}
        >>> unresolved
        ['debris pile']
    """
    vocabulary = mapillary_vocabulary()
    resolved: dict[str, tuple[str, ...]] = {}
    unresolved: list[str] = []
    for cls in classes:
        text = cls.strip()
        key = text.lower()
        if text in vocabulary:
            resolved[cls] = (text,)
            continue
        singular = (
            key[:-2] if key.endswith('es') and key[:-2] in _CLASS_ALIASES else key
        )
        singular = (
            singular[:-1]
            if singular.endswith('s') and singular[:-1] in _CLASS_ALIASES
            else singular
        )
        if singular in _CLASS_ALIASES:
            resolved[cls] = _CLASS_ALIASES[singular]
            continue
        if label_mapper is not None:
            try:
                labels = tuple(
                    lbl for lbl in label_mapper.map_classes([text]) if lbl in vocabulary
                )
            except Exception as exc:  # noqa: BLE001 - mapper failures are not fatal
                logger.warning(f"Label mapper failed for '{text}': {exc}")
                labels = ()
            if labels:
                resolved[cls] = labels
                continue
        unresolved.append(cls)
    return resolved, unresolved


def _slug(text: str) -> str:
    """Lower-case identifier-safe version of a class name."""
    return ''.join(ch if ch.isalnum() else '_' for ch in text.strip().lower()).strip(
        '_'
    )


def _camera_pose(props: dict[str, Any]) -> tuple[float, float, float] | None:
    """Return ``(lon, lat, compass)`` from image properties, or ``None``."""
    lon = lat = None
    for key in ('computed_geometry', 'geometry'):
        geom = props.get(key)
        if isinstance(geom, dict) and geom.get('coordinates'):
            lon, lat = geom['coordinates'][:2]
            break
    if lon is None:
        lon, lat = props.get('longitude'), props.get('latitude')
    # A key present with a null value must fall through to the raw angle:
    compass = props.get('computed_compass_angle')
    if compass is None:
        compass = props.get('compass_angle')
    if lon is None or lat is None or compass is None:
        return None
    return float(lon), float(lat), float(compass) % 360.0


def _is_pano(props: dict[str, Any]) -> bool:
    return bool(props.get('is_pano')) or props.get('camera_type') in _PANO_TYPES


def _captured_at(props: dict[str, Any]) -> str | None:
    value = props.get('captured_at')
    if isinstance(value, (int, float)):
        from datetime import UTC, datetime

        return datetime.fromtimestamp(value / 1000.0, tz=UTC).isoformat()
    return props.get('capture_date') or (str(value) if value is not None else None)


class MapillaryFeatureExtractor:
    """
    Discover objects along a Mapillary survey and place them on the ground.

    The extractor lists every image in a region (optionally only RAPID
    Facility uploads within a date window), reads Mapillary's segmentation
    detections for the requested classes as metadata, estimates where each
    detected object stands from the camera pose and the detection's position
    in the frame, and merges repeated sightings into one asset per object.
    Objects seen from two or more camera positions are triangulated;
    single sightings fall back to the one-view estimate and are labelled as
    such.

    Detections of the collection vehicle itself recur at the same place in
    every frame of a sequence and are removed before localisation.

    Classes outside the Mapillary vocabulary can be detected with SAM 3 on
    downsampled thumbnails when ``detection_source`` allows it; the default
    only does so for classes Mapillary cannot supply.

    Args:
        classes (Sequence[str]):
            Objects to find, in plain English (``'cars'``, ``'utility
            poles'``) or as exact Mapillary labels
            (``'object--vehicle--truck'``).
        access_token (str | None):
            Mapillary API token. Not needed when ``client`` is given.
        region (BoundingBox | PolygonRegion | None):
            Area to survey. May instead be passed when the extractor is
            called.
        start_date (str):
            Inclusive lower bound on capture date (``YYYY-MM-DD``).
        end_date (str):
            Inclusive upper bound on capture date.
        filter_rapid_only (bool):
            Only use imagery uploaded by the RAPID Facility. Defaults to
            ``True``.
        detection_source (str):
            ``'auto'`` (Mapillary map features for static point classes such
            as utility poles and hydrants, Mapillary detections for the rest,
            SAM 3 for unresolved classes), ``'mapillary'`` (detections only;
            skip unresolved classes with a warning), ``'sam3'`` (SAM 3 for
            every class) or ``'map_features'`` (only Mapillary's
            pre-triangulated map features; other classes are skipped).
        label_mapper (Any | None):
            Optional :class:`~rapidtools.processing.MapillaryLabelMapper`
            to translate class names the built-in aliases do not cover.
        camera_height_m (float):
            Camera height above the road, used for single-view ranges.
            Defaults to 2.4 (roof-mounted rig).
        min_range_m (float):
            Closest plausible object. Defaults to 2.0.
        max_range_m (float):
            Farthest triangulated position reported, and the farthest
            single-view range kept as a prior. Defaults to 60.0.
        min_object_width_m (float):
            Objects narrower than this, judged from the outline's angular
            width at the estimated distance, are fragments or false
            detections and are dropped. Resolution-independent and tolerant
            of partial occlusion. Defaults to 1.0.
        object_height_m (float):
            Typical height of the class. When a track's ground-contact
            ranges disagree with its triangulation, the track is called
            moving only if enough of an object this tall is visible for
            occlusion to be ruled out. Defaults to 1.5 (vehicles).
        min_area_fraction (float | None):
            Deprecated. Pixel-area floor below which a detection is ignored;
            ``None`` (the default) keeps only a tiny floor for decoding
            noise. Size is judged with ``min_object_width_m`` instead.
        cluster_radius_m (float):
            Sightings closer than this are the same object in the legacy
            ``'cluster'`` method; with ``'tracks'`` it only sets the default
            ``max_merge_m`` (1.5 times this value). Defaults to 4.0.
        localization_method (str):
            How sightings become objects. ``'tracks'`` (default) links
            detections frame to frame by bearing, triangulates each track
            robustly, drops moving vehicles and merges estimates with a
            covariance-aware gate (see
            :mod:`~rapidtools.processing.street_tracking`). ``'voting'``
            accumulates rays on a ground grid and takes the peaks, a
            baseline to compare against. ``'cluster'`` is the previous
            behaviour: single-view positions clustered by radius.
        bearing_sigma_deg (float):
            One-sigma bearing error of a sighting. Defaults to 0.75.
        max_merge_m (float | None):
            Farthest two estimates can be and still be merged into one
            object. ``None`` uses ``1.5 * cluster_radius_m``.
        track_gap_frames (int):
            Frames a track may miss before it is closed. Defaults to 2.
        min_path_distance_m (float):
            Objects closer than this to the line the camera drove are
            dropped; the survey vehicle passed through that spot. Defaults
            to 1.5.
        object_size_m (float):
            Object footprint used by the voting method. Defaults to 4.5.
        reid (bool):
            After localisation, compare the appearance of objects from
            different sequences that lie within ``reid_max_distance_m`` and
            merge look-alikes, removing double counts from repeated passes.
            Downloads one thumbnail per compared object and loads
            ``reid_model``. Defaults to ``False``.
        reid_model (str):
            Hugging Face vision backbone for the embeddings. Defaults to
            ``'facebook/dinov2-small'``.
        reid_min_similarity (float):
            Cosine similarity at which two crops count as the same object.
            Defaults to 0.80.
        reid_max_distance_m (float):
            Only objects this close are compared. Defaults to 6.0.
        reid_embedder (Any | None):
            A ready :class:`~rapidtools.processing.reid.AppearanceEmbedder`
            (or anything with ``embed(images)``) to use instead of loading
            ``reid_model``.
        min_observations (int):
            Distinct frames an object must have been seen in; ``3`` is a
            sound choice for a survey sampled every few metres. Defaults
            to 1.
        frame_spacing_m (float):
            Keep one frame per this many metres along each sequence before
            detecting; ``0`` keeps every frame. Defaults to 0.
        ego_filter (bool):
            Remove detections of the collection vehicle. Defaults to
            ``True``.
        ego_min_recurrence (float):
            Fraction of a sequence's frames a box must recur in to count as
            the collection vehicle. Defaults to 0.5.
        asset_type (str | None):
            ``asset_type`` attribute written to every asset. Defaults to the
            class name each object was found under.
        id_prefix (str):
            Prefix of the generated asset IDs. Defaults to ``'street'``.
        save_directory (str | Path):
            Where SAM 3 thumbnails are cached when that source is used.
        sam3_model (Any | None):
            Pre-loaded SAM 3 wrapper. Loaded on demand otherwise.
        sam3_model_id (str):
            Hugging Face repository of the SAM 3 weights. Defaults to
            ``'facebook/sam3'``.
        sam3_threshold (float):
            SAM 3 detection confidence threshold. Defaults to 0.5.
        sam3_mask_threshold (float):
            SAM 3 mask binarisation threshold. Defaults to 0.5.
        sam3_image_size (str):
            Thumbnail size SAM 3 runs on: ``'1024'`` (default) or ``'2048'``.
        sam3_batch_size (int):
            Thumbnails per SAM 3 inference call. Defaults to 4.
        load_in_4bit (bool):
            Load SAM 3 with 4-bit quantisation. Defaults to ``True``.
        device (str):
            Compute device for SAM 3 (``'auto'``, ``'cuda'``, ``'cpu'``).
        max_workers (int):
            Threads for metadata and thumbnail downloads. Defaults to 10.
        frame_batch_size (int):
            Frames whose metadata is fetched and converted at a time. Each
            image's detection payload (every Mapillary label, as encoded
            geometry) is dropped as soon as the requested classes have been
            read from it, so memory use is bounded by one batch rather than
            by the size of the survey. Defaults to 200.
        simplify_tolerance (float):
            Douglas-Peucker tolerance, in normalised image units, applied to
            every detection outline before it is stored (``0.002`` is about
            four pixels of a 2048-wide image), capped at 1 % of the outline's
            longest side so small, distant objects keep their detail.
            Mapillary outlines trace every pixel step of a mask, so this cuts
            the memory of a city-wide run several-fold without changing
            which objects are found. ``0`` keeps the raw outlines. Defaults
            to 0.002.
        cancel_event (threading.Event | None):
            Cooperative cancellation flag checked between images and batches.
        client (MapillaryClient | None):
            Pre-configured client; one is created from ``access_token``
            otherwise.

    Example:
        >>> from rapidtools.core import BoundingBox
        >>> extractor = MapillaryFeatureExtractor(
        ...     classes=['vehicles'],
        ...     access_token='MLY|...',
        ...     start_date='2025-08-01',
        ...     min_observations=2,
        ... )
        >>> region = BoundingBox(-117.45, 47.69, -117.44, 47.70)
        >>> vehicles = extractor(region)  # doctest: +SKIP
        >>> vehicles[0].attributes['localization']  # doctest: +SKIP
        'triangulated'
    """

    stage = Stage.DETECT

    def __init__(
        self,
        classes: Sequence[str],
        access_token: str | None = None,
        region: BoundingBox | PolygonRegion | None = None,
        start_date: str = '',
        end_date: str = '',
        filter_rapid_only: bool = True,
        detection_source: str = 'auto',
        label_mapper: Any = None,
        camera_height_m: float = 2.4,
        min_range_m: float = 2.0,
        max_range_m: float = DEFAULT_MAX_RANGE_M,
        min_object_width_m: float = 1.0,
        object_height_m: float = 1.5,
        min_area_fraction: float | None = None,
        cluster_radius_m: float = 4.0,
        localization_method: str = 'tracks',
        bearing_sigma_deg: float = 0.75,
        max_merge_m: float | None = None,
        track_gap_frames: int = 2,
        min_path_distance_m: float = 1.5,
        object_size_m: float = 4.5,
        reid: bool = False,
        reid_model: str = 'facebook/dinov2-small',
        reid_min_similarity: float = 0.80,
        reid_max_distance_m: float = 6.0,
        reid_embedder: Any = None,
        min_observations: int = 1,
        frame_spacing_m: float = 0.0,
        ego_filter: bool = True,
        ego_min_recurrence: float = 0.5,
        asset_type: str | None = None,
        id_prefix: str = 'street',
        save_directory: str | Path = 'street_detections',
        sam3_model: Any = None,
        sam3_model_id: str = 'facebook/sam3',
        sam3_threshold: float = 0.5,
        sam3_mask_threshold: float = 0.5,
        sam3_image_size: str = '1024',
        sam3_batch_size: int = 4,
        load_in_4bit: bool = True,
        device: str = 'auto',
        max_workers: int = 10,
        frame_batch_size: int = 200,
        simplify_tolerance: float = 0.002,
        cancel_event: threading.Event | None = None,
        client: MapillaryClient | None = None,
    ) -> None:
        if not classes:
            raise ValueError('At least one class is required.')
        if frame_batch_size < 1:
            raise ValueError('frame_batch_size must be at least 1.')
        if simplify_tolerance < 0:
            raise ValueError('simplify_tolerance cannot be negative.')
        if detection_source not in ('auto', 'mapillary', 'sam3', 'map_features'):
            raise ValueError(
                "detection_source must be 'auto', 'mapillary', 'sam3' or "
                f"'map_features', got {detection_source!r}."
            )
        if localization_method not in ('tracks', 'voting', 'cluster'):
            raise ValueError(
                "localization_method must be 'tracks', 'voting' or 'cluster', "
                f'got {localization_method!r}.'
            )
        if client is None and not access_token:
            raise ValueError('Provide a Mapillary access_token or a client.')
        if min_observations < 1:
            raise ValueError('min_observations must be at least 1.')
        self.classes = list(classes)
        self.region = region
        self.start_date = start_date
        self.end_date = end_date
        self.filter_rapid_only = filter_rapid_only
        self.detection_source = detection_source
        self.label_mapper = label_mapper
        self.camera_height_m = camera_height_m
        self.min_range_m = min_range_m
        self.max_range_m = max_range_m
        self.min_object_width_m = min_object_width_m
        self.object_height_m = object_height_m
        if min_area_fraction is not None:
            warnings.warn(
                'min_area_fraction is deprecated: object size is now judged in '
                'metres once the distance is known (min_object_width_m). The '
                'value is still applied as a pixel floor.',
                DeprecationWarning,
                stacklevel=2,
            )
        self.min_area_fraction = (
            MIN_POLYGON_AREA_FRACTION
            if min_area_fraction is None
            else min_area_fraction
        )
        self.cluster_radius_m = cluster_radius_m
        self.localization_method = localization_method
        self.bearing_sigma_deg = bearing_sigma_deg
        self.max_merge_m = max_merge_m
        self.track_gap_frames = track_gap_frames
        self.min_path_distance_m = min_path_distance_m
        self.object_size_m = object_size_m
        self.reid = reid
        self.reid_model = reid_model
        self.reid_min_similarity = reid_min_similarity
        self.reid_max_distance_m = reid_max_distance_m
        self.reid_embedder = reid_embedder
        self.min_observations = min_observations
        self.frame_spacing_m = frame_spacing_m
        self.ego_filter = ego_filter
        self.ego_min_recurrence = ego_min_recurrence
        self.asset_type = asset_type
        self.id_prefix = id_prefix
        self.save_directory = Path(save_directory)
        self.sam3_model = sam3_model
        self.sam3_model_id = sam3_model_id
        self.sam3_threshold = sam3_threshold
        self.sam3_mask_threshold = sam3_mask_threshold
        self.sam3_image_size = str(sam3_image_size)
        self.sam3_batch_size = sam3_batch_size
        self.load_in_4bit = load_in_4bit
        self.device = device
        self.max_workers = max_workers
        self.frame_batch_size = frame_batch_size
        self.simplify_tolerance = simplify_tolerance
        self.cancel_event = cancel_event
        self.client = client or MapillaryClient(
            access_token, save_dir=self.save_directory / 'images'
        )

    # ---------------------------------------------------------------- setup
    def _plan_classes(
        self,
    ) -> tuple[dict[str, tuple[str, ...]], list[str], dict[str, tuple[str, ...]]]:
        """
        Split the requested classes three ways.

        Returns:
            tuple: ``(detections, sam3_prompts, map_features)``: classes read
            from Mapillary's per-image detections, classes detected with
            SAM 3, and classes served by Mapillary's pre-triangulated map
            features (static point objects such as poles and hydrants).
        """
        if self.detection_source == 'sam3':
            return {}, list(self.classes), {}
        resolved, unresolved = resolve_classes(self.classes, self.label_mapper)
        map_features: dict[str, tuple[str, ...]] = {}
        if self.detection_source in ('auto', 'map_features'):
            for cls, labels in list(resolved.items()):
                if labels and all(is_map_feature_value(v) for v in labels):
                    map_features[cls] = labels
                    del resolved[cls]
        if self.detection_source == 'map_features':
            skipped = list(resolved) + unresolved
            if skipped:
                logger.warning(
                    'Not served as Mapillary map features: %s; these classes are '
                    "skipped. Use detection_source='auto' to detect them from "
                    'images.',
                    ', '.join(repr(c) for c in skipped),
                )
            return {}, [], map_features
        if unresolved and self.detection_source == 'mapillary':
            logger.warning(
                'No Mapillary label for %s; these classes are skipped. Use '
                "detection_source='auto' to detect them with SAM 3.",
                ', '.join(repr(c) for c in unresolved),
            )
            unresolved = []
        return resolved, unresolved, map_features

    @staticmethod
    def _finish(
        existing: PhysicalAssetCollection | None, assets: Iterable[PhysicalAsset]
    ) -> PhysicalAssetCollection:
        """Collection of ``existing`` (if any) plus ``assets``."""
        collection = PhysicalAssetCollection()
        if existing is not None:
            collection.merge(existing)
        for asset in assets:
            collection.add(asset)
        return collection

    def _resolve_region(self, source: Any) -> BoundingBox | PolygonRegion:
        region = (
            source if isinstance(source, (BoundingBox, PolygonRegion)) else self.region
        )
        if region is None:
            raise ValueError(
                'A region is required: pass it to the constructor or call the '
                'extractor with a BoundingBox or PolygonRegion.'
            )
        return region

    # ---------------------------------------------------------------- run
    def __call__(self, source: Any = None) -> PhysicalAssetCollection:
        """
        Discover objects in the region and return them as a collection.

        Args:
            source (BoundingBox | PolygonRegion | PhysicalAssetCollection | None):
                The region to survey, or an existing collection (the
                pipeline passes one) whose assets are kept alongside the new
                detections while the constructor's ``region`` is used.

        Returns:
            PhysicalAssetCollection: One point asset per detected object with
            ``asset_type``, ``label``, ``confidence``, ``n_observations``,
            ``localization`` and ``observations`` attributes.
        """
        existing = source if isinstance(source, PhysicalAssetCollection) else None
        region = self._resolve_region(source)
        bbox = region if isinstance(region, BoundingBox) else region.get_bounding_box()
        mapillary_classes, sam3_classes, map_feature_classes = self._plan_classes()
        if not mapillary_classes and not sam3_classes and not map_feature_classes:
            logger.warning('Nothing to detect after class resolution.')
            return existing or PhysicalAssetCollection()
        map_assets = (
            self._map_feature_assets(bbox, map_feature_classes)
            if map_feature_classes
            else []
        )
        if not mapillary_classes and not sam3_classes:
            return self._finish(existing, map_assets)

        # 1. Cheap pass: the coverage tiles list every image with its position,
        #    heading, date and sequence. Tiles are ~2 km wide, so clip to the
        #    region before anything per-image happens.
        logger.info(
            f'Listing Mapillary images in {bbox.bounds} '
            f'({"RAPID only" if self.filter_rapid_only else "all uploaders"})...'
        )
        listed = self.client.fetch_images_in_bbox(
            bbox,
            start_date=self.start_date,
            end_date=self.end_date,
            filter_rapid_only=self.filter_rapid_only,
            max_workers=self.max_workers,
        )
        raise_if_cancelled(self.cancel_event, 'street-level detection')
        inside = []
        for image in listed:
            pose = _camera_pose(image.properties)
            if pose is not None and region.contains(Point(pose[0], pose[1])):
                inside.append((image, pose))
        if not inside:
            logger.warning('No images with a camera position inside the region.')
            return self._finish(existing, map_assets)

        # 2. Thin along each sequence before paying for per-image metadata:
        keep = thin_frames(
            [
                (
                    str(image.id),
                    pose[0],
                    pose[1],
                    _captured_at(image.properties),
                    image.properties.get('sequence'),
                )
                for image, pose in inside
            ],
            self.frame_spacing_m,
        )
        logger.info(
            f'{len(listed)} images in the covering tiles, {len(inside)} inside the '
            f'region, {len(keep)} kept after thinning.'
        )

        # 3. Metadata (and Mapillary detections) for the kept frames only,
        #    one batch at a time. A frame's detection payload lists every
        #    Mapillary label as encoded geometry, so it is read for the
        #    requested classes and dropped before the next batch arrives.
        fields = list(_IMAGE_FIELDS)
        if mapillary_classes:
            fields += list(_DETECTION_FIELDS)
        ids = sorted(keep)
        observations: dict[str, list[Observation]] = defaultdict(list)
        sam3_frames: list[dict[str, Any]] = []
        n_frames = 0
        batches = range(0, len(ids), self.frame_batch_size)
        for start in tqdm(batches, desc='Frame batches', disable=len(batches) < 2):
            raise_if_cancelled(self.cancel_event, 'street-level detection')
            rich = self.client.fetch_images_by_ids(
                ids[start : start + self.frame_batch_size],
                fields=fields,
                save_to_disk=False,
                max_workers=self.max_workers,
                show_progress=len(batches) < 2,
            )
            frames = self._frames(rich)
            del rich
            n_frames += len(frames)
            if mapillary_classes:
                for cls, obs in self._mapillary_observations(
                    frames, mapillary_classes
                ).items():
                    observations[cls].extend(obs)
            for frame in frames:
                frame['detections'] = None
            if sam3_classes:
                sam3_frames.extend(frames)
        if not n_frames:
            logger.warning('No usable images (missing camera pose) in the region.')
            return self._finish(existing, map_assets)
        logger.info(f'{n_frames} frames with metadata; detections read.')
        if sam3_classes:
            for cls, obs in self._sam3_observations(sam3_frames, sam3_classes).items():
                observations[cls].extend(obs)

        collection = self._finish(existing, map_assets)
        counter = 0
        for cls in self.classes:
            if cls in map_feature_classes:
                continue
            obs_list = observations.get(cls, [])
            if not obs_list:
                logger.info(f"No detections for class '{cls}'.")
                continue
            assets, counter = self._assets_for_class(cls, obs_list, counter)
            for asset in assets:
                collection.add(asset)
            logger.info(f"'{cls}': {len(obs_list)} sightings -> {len(assets)} objects.")
        return collection

    # ---------------------------------------------------------------- frames
    def _frames(self, images) -> list[dict[str, Any]]:
        """Reduce image assets to the camera facts the geometry needs."""
        frames = []
        for image in images:
            props = image.properties
            pose = _camera_pose(props)
            if pose is None:
                continue
            lon, lat, compass = pose
            frames.append(
                {
                    'id': str(image.id),
                    'lon': lon,
                    'lat': lat,
                    'compass': compass,
                    'is_pano': _is_pano(props),
                    'sequence': props.get('sequence'),
                    'captured_at': _captured_at(props),
                    'width': props.get('width'),
                    'height': props.get('height'),
                    'camera_type': props.get('camera_type'),
                    'camera_parameters': props.get('camera_parameters'),
                    # Full pose (OpenSfM axis-angle, world -> camera); lets the
                    # geometry account for the camera's pitch and roll:
                    'rotation': props.get('computed_rotation'),
                    'detections': (props.get('detections') or {}).get('data'),
                    'thumb_url': props.get(f'thumb_{self.sam3_image_size}_url'),
                }
            )
        return frames

    def _observation(
        self, frame: dict[str, Any], label: str, polygon, source: str, confidence=None
    ) -> Observation:
        if source == 'mapillary' and self.simplify_tolerance > 0:
            # Cap the tolerance at a share of the outline's size so small,
            # distant objects keep the detail their localisation needs:
            xs = [x for x, _ in polygon]
            ys = [y for _, y in polygon]
            extent = max(max(xs) - min(xs), max(ys) - min(ys))
            polygon = simplify_polygon(
                polygon, min(self.simplify_tolerance, _RELATIVE_TOLERANCE * extent)
            )
        return Observation(
            image_id=frame['id'],
            label=label,
            polygon=polygon,
            camera_lon=frame['lon'],
            camera_lat=frame['lat'],
            compass_angle=frame['compass'],
            is_pano=frame['is_pano'],
            sequence_id=frame['sequence'],
            captured_at=frame['captured_at'],
            confidence=confidence,
            source=source,
            image_width=frame['width'],
            image_height=frame['height'],
            extra={
                key: frame[source_key]
                for key, source_key in (
                    ('camera_parameters', 'camera_parameters'),
                    ('camera_type', 'camera_type'),
                    ('computed_rotation', 'rotation'),
                )
                if frame.get(source_key) is not None
            },
        )

    # ---------------------------------------------------------------- mapillary
    def _mapillary_observations(
        self, frames: list[dict[str, Any]], classes: dict[str, tuple[str, ...]]
    ) -> dict[str, list[Observation]]:
        """Turn Mapillary detections into observations, fetching if needed."""
        label_to_class: dict[str, str] = {}
        for cls, labels in classes.items():
            for label in labels:
                label_to_class.setdefault(label, cls)
        wanted = set(label_to_class)

        def detections_for(frame):
            data = frame.get('detections')
            if data is None:
                data = self.client.fetch_detections(frame['id'], values=wanted)
            return frame, data or []

        result: dict[str, list[Observation]] = defaultdict(list)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(detections_for, f) for f in frames]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc='Mapillary detections',
                disable=len(frames) <= self.frame_batch_size,
            ):
                raise_if_cancelled(self.cancel_event, 'street-level detection')
                frame, data = future.result()
                for item in data:
                    label = item.get('value')
                    if label not in wanted or not item.get('geometry'):
                        continue
                    for polygon in MapillaryClient.decode_detection_polygons(
                        item['geometry']
                    ):
                        result[label_to_class[label]].append(
                            self._observation(frame, label, polygon, 'mapillary')
                        )
        return result

    # ---------------------------------------------------------------- sam3
    def _load_sam3(self):
        if self.sam3_model is None:
            from rapidtools.models import SAM3Inference

            self.sam3_model = SAM3Inference(
                model_id=self.sam3_model_id,
                device=self.device,
                load_in_4bit=self.load_in_4bit,
            )
        return self.sam3_model

    def _download_thumbnail(self, frame: dict[str, Any]) -> Path | None:
        """Fetch a frame's thumbnail into the cache directory."""
        target_dir = self.save_directory / 'images'
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f'{frame["id"]}_{self.sam3_image_size}.jpg'
        if target.is_file():
            return target
        url = frame.get('thumb_url') or self.client.get_image_url(
            frame['id'], self.sam3_image_size
        )
        if not url:
            return None
        return target if self.client._download_image(url, target) else None

    def _sam3_observations(
        self, frames: list[dict[str, Any]], classes: list[str]
    ) -> dict[str, list[Observation]]:
        """Detect classes without Mapillary labels with SAM 3 on thumbnails."""
        model = self._load_sam3()
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            paths = list(
                tqdm(
                    pool.map(self._download_thumbnail, frames),
                    total=len(frames),
                    desc='Thumbnails',
                )
            )
        ready = [(f, p) for f, p in zip(frames, paths, strict=True) if p is not None]
        result: dict[str, list[Observation]] = defaultdict(list)
        for cls in classes:
            for start in range(0, len(ready), self.sam3_batch_size):
                raise_if_cancelled(self.cancel_event, 'SAM 3 street-level detection')
                batch = ready[start : start + self.sam3_batch_size]
                output = model.run_inference(
                    [str(p) for _, p in batch],
                    prompt=cls,
                    threshold=self.sam3_threshold,
                    mask_threshold=self.sam3_mask_threshold,
                )
                if output is None or not output.masks:
                    continue
                for (frame, _), masks in zip(batch, output.masks, strict=False):
                    for polygon in self._masks_to_polygons(masks):
                        result[cls].append(
                            self._observation(frame, cls, polygon, 'sam3')
                        )
        return result

    @staticmethod
    def _masks_to_polygons(masks) -> list[list[tuple[float, float]]]:
        """Normalised bounding polygons of an ``(n, H, W)`` mask stack."""
        if masks is None:
            return []
        arr = np.asarray(masks)
        if arr.ndim == 2:
            arr = arr[None]
        polygons = []
        for mask in arr:
            ys, xs = np.nonzero(mask)
            if xs.size == 0:
                continue
            h, w = mask.shape
            x0, x1 = xs.min() / w, (xs.max() + 1) / w
            y0, y1 = ys.min() / h, (ys.max() + 1) / h
            polygons.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        return polygons

    # ---------------------------------------------------------------- map features
    def _map_feature_assets(
        self, bbox: BoundingBox, classes: dict[str, tuple[str, ...]]
    ) -> list[PhysicalAsset]:
        """Point assets from Mapillary's pre-triangulated map features."""
        assets: list[PhysicalAsset] = []
        for cls, values in classes.items():
            raise_if_cancelled(self.cancel_event, 'street-level detection')
            features = self.client.fetch_map_features(
                bbox, values, start_date=self.start_date, end_date=self.end_date
            )
            counter = 0
            for feature in features:
                counter += 1
                image_ids = list(feature.get('image_ids') or [])
                attributes: dict[str, Any] = {
                    'asset_type': self.asset_type or _slug(cls),
                    'class': cls,
                    'label': feature.get('object_value'),
                    'source': 'mapillary_map_features',
                    'localization': 'map_feature',
                    'map_feature_id': str(feature.get('id')),
                    'n_observations': 0,
                    'n_images': len(image_ids),
                }
                if image_ids:
                    attributes['image_ids'] = image_ids
                if feature.get('aligned_direction') is not None:
                    attributes['aligned_direction'] = round(
                        float(feature['aligned_direction']), 1
                    )
                if feature.get('first_seen_at'):
                    attributes['first_seen'] = feature['first_seen_at']
                if feature.get('last_seen_at'):
                    attributes['last_seen'] = feature['last_seen_at']
                assets.append(
                    PhysicalAsset(
                        id=f'{self.id_prefix}_{_slug(cls)}_{counter:05d}',
                        geometry=Point(float(feature['lon']), float(feature['lat'])),
                        attributes=attributes,
                    )
                )
            logger.info(
                f"'{cls}': {len(features)} Mapillary map features in the region."
            )
        return assets

    # ---------------------------------------------------------------- assets
    def _assets_for_class(
        self, cls: str, obs_list: list[Observation], counter: int
    ) -> tuple[list[PhysicalAsset], int]:
        """Filter, localise and group one class's sightings into assets."""
        if self.ego_filter:
            ego = estimate_ego_mask(obs_list, min_recurrence=self.ego_min_recurrence)
            if ego:
                logger.info(
                    f"'{cls}': {len(ego)} sightings belong to the survey vehicle."
                )
            obs_list = [o for i, o in enumerate(obs_list) if i not in ego]
        if self.localization_method != 'cluster':
            return self._assets_from_tracks(cls, obs_list, counter)
        located = [
            o
            for o in obs_list
            if localize(
                o,
                camera_height_m=self.camera_height_m,
                min_range_m=self.min_range_m,
                max_range_m=self.max_range_m,
                min_area_fraction=self.min_area_fraction,
            )
        ]
        if len(located) < len(obs_list):
            logger.info(
                f"'{cls}': {len(obs_list) - len(located)} sightings had no plausible "
                'ground position.'
            )
        assets = []
        for members in cluster_observations(located, self.cluster_radius_m):
            if len(members) < self.min_observations:
                continue
            counter += 1
            assets.append(self._build_asset(cls, members, counter))
        return assets, counter

    def _assets_from_tracks(
        self, cls: str, obs_list: list[Observation], counter: int
    ) -> tuple[list[PhysicalAsset], int]:
        """Track, triangulate, merge (and optionally re-identify) one class."""
        estimates, counts, unproject = discover_objects(
            obs_list,
            method=self.localization_method,
            camera_height_m=self.camera_height_m,
            min_range_m=self.min_range_m,
            max_range_m=self.max_range_m,
            min_area_fraction=self.min_area_fraction,
            bearing_sigma_deg=self.bearing_sigma_deg,
            max_merge_m=self.max_merge_m or 1.5 * self.cluster_radius_m,
            track_gap_frames=self.track_gap_frames,
            min_path_distance_m=self.min_path_distance_m,
            min_object_width_m=self.min_object_width_m,
            object_height_m=self.object_height_m,
            object_size_m=self.object_size_m,
        )
        logger.info(
            f"'{cls}': {counts['with_bearing']} sightings with a bearing, "
            f'{counts.get("duplicates", 0)} duplicate outlines dropped, '
            f'{counts["tracks"]} tracks, {counts["moving"]} moving dropped, '
            f'{counts["fragment"]} too small dropped, {counts["on_path"]} on the '
            f'driven path dropped -> {counts["objects"]} objects.'
        )
        assets = []
        for est in estimates:
            if est.n_images < self.min_observations:
                continue
            counter += 1
            assets.append(self._asset_from_estimate(cls, est, unproject, counter))
        if self.reid and len(assets) > 1:
            assets = self._merge_by_appearance(assets)
        return assets, counter

    def _asset_from_estimate(
        self, cls: str, est: Any, unproject: Any, index: int
    ) -> PhysicalAsset:
        """Describe one tracked/voted object as a point asset."""
        lon, lat = unproject(est.x, est.y)
        members = est.members
        ranges = [o.range_m for o in members if o.range_m is not None]
        confidences = [o.confidence for o in members if o.confidence is not None]
        dates = sorted(o.captured_at for o in members if o.captured_at)
        attributes: dict[str, Any] = {
            'asset_type': self.asset_type or _slug(cls),
            'class': cls,
            'label': Counter(o.label for o in members).most_common(1)[0][0],
            'source': Counter(o.source for o in members).most_common(1)[0][0],
            'n_observations': len(members),
            'sequence_ids': sorted({o.sequence_id for o in members if o.sequence_id}),
            'observations': [o.to_dict() for o in members],
        }
        attributes.update(estimate_attributes(est))
        if ranges:
            attributes['min_range_m'] = round(min(ranges), 1)
        if confidences:
            attributes['confidence'] = round(max(confidences), 3)
        if dates:
            attributes['first_seen'], attributes['last_seen'] = dates[0], dates[-1]
        return PhysicalAsset(
            id=f'{self.id_prefix}_{_slug(cls)}_{index:05d}',
            geometry=Point(lon, lat),
            attributes=attributes,
        )

    def _merge_by_appearance(self, assets: list[PhysicalAsset]) -> list[PhysicalAsset]:
        """Merge look-alike objects from different sequences (see ``reid``)."""
        from .reid import (
            AppearanceEmbedder,
            ReidCandidate,
            crop_observation,
            merge_by_appearance,
            merged_position,
            pairs_to_compare,
        )

        by_key = {a.id: a for a in assets}
        obs_by_key: dict[str, list[Observation]] = {}
        candidates = []
        for asset in assets:
            obs = [
                Observation.from_dict(r)
                for r in asset.attributes.get('observations', [])
            ]
            obs.sort(key=lambda o: o.range_m if o.range_m is not None else 1e9)
            obs_by_key[asset.id] = obs
            candidates.append(
                ReidCandidate.from_observations(
                    asset.id, asset.geometry.x, asset.geometry.y, obs
                )
            )
        pairs = pairs_to_compare(candidates, max_distance_m=self.reid_max_distance_m)
        if not pairs:
            return assets
        crops: dict[str, Image.Image] = {}
        for key in sorted({k for pair in pairs for k in pair}):
            for sighting in obs_by_key[key]:
                path = self._download_thumbnail(
                    {'id': sighting.image_id, 'thumb_url': None}
                )
                if path is None:
                    continue
                try:
                    with Image.open(path) as image:
                        crops[key] = crop_observation(
                            image.convert('RGB'), sighting.polygon
                        )
                except OSError:
                    continue
                break
        embedder = self.reid_embedder or AppearanceEmbedder(
            self.reid_model, device=self.device
        )
        groups = merge_by_appearance(
            candidates,
            crops,
            embedder,
            max_distance_m=self.reid_max_distance_m,
            min_similarity=self.reid_min_similarity,
        )
        merged: list[PhysicalAsset] = []
        n_merged = 0
        for group in groups:
            parts = [by_key[k] for k in group]
            if len(parts) == 1:
                merged.append(parts[0])
                continue
            n_merged += len(parts) - 1
            lon, lat = merged_position([(a.geometry.x, a.geometry.y) for a in parts])
            records = [r for a in parts for r in a.attributes.get('observations', [])]
            attributes = dict(parts[0].attributes)
            attributes.update(
                {
                    'observations': records,
                    'n_observations': len(records),
                    'n_images': len({r['image_id'] for r in records}),
                    'sequence_ids': sorted(
                        {s for a in parts for s in a.attributes.get('sequence_ids', [])}
                    ),
                    'reid_merged': len(parts),
                }
            )
            if any(a.attributes.get('localization') == 'triangulated' for a in parts):
                attributes['localization'] = 'triangulated'
            dates = sorted(
                d
                for a in parts
                for d in (a.attributes.get('first_seen'), a.attributes.get('last_seen'))
                if d
            )
            if dates:
                attributes['first_seen'], attributes['last_seen'] = dates[0], dates[-1]
            merged.append(
                PhysicalAsset(
                    id=parts[0].id, geometry=Point(lon, lat), attributes=attributes
                )
            )
        if n_merged:
            logger.info(
                f'Appearance re-identification merged {n_merged} duplicate objects.'
            )
        return merged

    def _build_asset(
        self, cls: str, members: list[Observation], index: int
    ) -> PhysicalAsset:
        """Position and describe one object from its sightings (legacy method)."""
        # Every member passed ``localize``, which sets all four of these:
        rays: dict[str, tuple[float, float, float]] = {}
        lons: list[float] = []
        lats: list[float] = []
        ranges: list[float] = []
        for o in members:
            if o.lon is None or o.lat is None or o.bearing is None or o.range_m is None:
                raise ValueError(f'Observation {o.image_id!r} has not been localized.')
            rays.setdefault(o.image_id, (o.camera_lon, o.camera_lat, o.bearing))
            lons.append(o.lon)
            lats.append(o.lat)
            ranges.append(o.range_m)
        project, unproject = local_projection(lons[0], lats[0])
        lon = sum(lons) / len(members)
        lat = sum(lats) / len(members)
        localization, rms = 'single_view', None
        if len(rays) >= 2:
            fix = intersect_bearings(
                list(rays.values()),
                project,
                unproject,
                max_range_m=2 * self.max_range_m,
            )
            if fix is not None:
                fx, fy, rms = fix
                # Only trust the intersection when it agrees with the sightings:
                px, py = project(fx, fy)
                mx, my = project(lon, lat)
                if math.hypot(px - mx, py - my) <= 3 * self.cluster_radius_m:
                    lon, lat, localization = fx, fy, 'triangulated'
                else:
                    rms = None
        confidences = [o.confidence for o in members if o.confidence is not None]
        dates = sorted(o.captured_at for o in members if o.captured_at)
        attributes: dict[str, Any] = {
            'asset_type': self.asset_type or _slug(cls),
            'class': cls,
            'label': Counter(o.label for o in members).most_common(1)[0][0],
            'source': Counter(o.source for o in members).most_common(1)[0][0],
            'n_observations': len(members),
            'n_images': len(rays),
            'localization': localization,
            'min_range_m': round(min(ranges), 1),
            'sequence_ids': sorted({o.sequence_id for o in members if o.sequence_id}),
            'observations': [o.to_dict() for o in members],
        }
        if rms is not None:
            attributes['position_rms_m'] = round(rms, 2)
        if confidences:
            attributes['confidence'] = round(max(confidences), 3)
        if dates:
            attributes['first_seen'], attributes['last_seen'] = dates[0], dates[-1]
        return PhysicalAsset(
            id=f'{self.id_prefix}_{_slug(cls)}_{index:05d}',
            geometry=Point(lon, lat),
            attributes=attributes,
        )


class MapillaryObjectImageExtractor:
    """
    Crop each detected object out of the street-level images that saw it.

    Reads the ``observations`` written by :class:`MapillaryFeatureExtractor`,
    picks the closest distinct views of every asset, downloads those images
    at the requested thumbnail size, crops the detection with a margin and
    attaches the crops as :class:`~rapidtools.core.ImageAsset` objects. The
    detection outline can be drawn on the crop with the same ``outline_*``
    options as :class:`~rapidtools.processing.AerialImageryExtractor`.

    Args:
        save_directory (str | Path):
            Where the crops are written.
        access_token (str | None):
            Mapillary API token. Not needed when ``client`` is given.
        max_images_per_asset (int):
            Views to keep per object, closest first and from distinct images.
            Defaults to 2.
        image_size (str):
            Thumbnail size to download: ``'1024'``, ``'2048'`` or
            ``'original'``. Defaults to ``'2048'``.
        crop_buffer (float | str):
            Margin around the detection: a percentage of its longest side
            (``'25%'``, the default), pixels, or a distance such as ``'1 m'``
            converted at the object's range.
        min_crop_px (int):
            Crops are grown to at least this many pixels on their shorter
            side. Defaults to 256.
        overlay_asset_outline (bool):
            Draw the detection outline on the crop. Defaults to ``False``.
        outline_shape (str):
            Outline shape, as in
            :class:`~rapidtools.processing.AerialImageryExtractor`:
            ``'geometry'``, ``'bbox'``, ``'rotated_bbox'``, ``'convex_hull'``
            or ``'corners'``. Defaults to ``'geometry'``.
        outline_buffer (float | str):
            Outline offset in pixels, percent, or a distance such as
            ``'0.5 m'`` converted at the object's range. Defaults to ``0``.
        outline_width (int | float | str):
            Stroke width in pixels or percent of the shorter crop side.
            Defaults to ``6``.
        outline_color (str | tuple[int, int, int]):
            PIL colour of the outline. Defaults to ``'red'``.
        image_prefix (str):
            Prefix of the crop filenames and image IDs. Defaults to
            ``'street'``.
        max_workers (int):
            Concurrent downloads. Each source image is downloaded once,
            cropped for every asset that needs it and released, so at most
            this many decoded images are in memory at a time. Defaults to 5.
        cancel_event (threading.Event | None):
            Cooperative cancellation flag checked between images.
        client (MapillaryClient | None):
            Pre-configured client; one is created from ``access_token``
            otherwise.

    Example:
        >>> extractor = MapillaryObjectImageExtractor(
        ...     'output/vehicle_crops', access_token='MLY|...',
        ...     max_images_per_asset=2, overlay_asset_outline=True,
        ...     outline_shape='corners',
        ... )
        >>> vehicles = extractor(vehicles)  # doctest: +SKIP
        >>> vehicles[0].image_assets[0].properties['range_m']  # doctest: +SKIP
        6.1
    """

    stage = Stage.EXTRACT_IMAGERY

    def __init__(
        self,
        save_directory: str | Path,
        access_token: str | None = None,
        max_images_per_asset: int = 2,
        image_size: str = '2048',
        crop_buffer: float | str = '25%',
        min_crop_px: int = 256,
        overlay_asset_outline: bool = False,
        outline_shape: str = 'geometry',
        outline_buffer: float | str = 0,
        outline_width: int | float | str = 6,
        outline_color: str | tuple[int, int, int] = 'red',
        image_prefix: str = 'street',
        max_workers: int = 5,
        cancel_event: threading.Event | None = None,
        client: MapillaryClient | None = None,
    ) -> None:
        if client is None and not access_token:
            raise ValueError('Provide a Mapillary access_token or a client.')
        if max_images_per_asset < 1:
            raise ValueError('max_images_per_asset must be at least 1.')
        if outline_shape not in outlines.OUTLINE_SHAPES:
            raise ValueError(
                f'outline_shape must be one of {outlines.OUTLINE_SHAPES}, '
                f'got {outline_shape!r}.'
            )
        self.save_directory = Path(save_directory).resolve()
        self.max_images_per_asset = max_images_per_asset
        self.image_size = str(image_size)
        self._crop_buffer_spec = outlines.parse_outline_buffer(crop_buffer)
        self.crop_buffer = crop_buffer
        self.min_crop_px = min_crop_px
        self.overlay_asset_outline = overlay_asset_outline
        self.outline_shape = outline_shape
        self._outline_buffer_spec = outlines.parse_outline_buffer(outline_buffer)
        self._outline_width_spec = outlines.parse_outline_width(outline_width)
        outlines.validate_outline_color(outline_color)
        self.outline_color = outline_color
        self.image_prefix = image_prefix
        self.max_workers = max_workers
        self.cancel_event = cancel_event
        self.client = client or MapillaryClient(
            access_token, save_dir=self.save_directory / 'source_images'
        )

    # ---------------------------------------------------------------- images
    def _source_image(self, image_id: str) -> Image.Image | None:
        """Download the source image at the configured size."""
        url = self.client.get_image_url(image_id, self.image_size)
        if not url:
            return None
        try:
            response = self.client.session.get(url, timeout=REQUESTS_TIMEOUT_VAL)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as exc:  # noqa: BLE001 - one bad image must not stop the run
            logger.error(f'Could not download image {image_id}: {exc}')
            return None

    @staticmethod
    def select_observations(
        observations: Sequence[Observation], limit: int
    ) -> list[Observation]:
        """
        Pick the closest views of an object, one per image.

        Example:
            >>> from rapidtools.core import Observation
            >>> def obs(i, rng):
            ...     o = Observation(str(i), 'car', [(0, 0), (1, 0), (1, 1)], 0, 0, 0)
            ...     o.range_m = rng
            ...     return o
            >>> [o.image_id for o in MapillaryObjectImageExtractor.select_observations(
            ...     [obs(1, 9.0), obs(2, 4.0), obs(2, 3.0), obs(3, 6.0)], 2)]
            ['2', '3']
        """
        ranked = sorted(
            observations,
            key=lambda o: (
                o.range_m if o.range_m is not None else float('inf'),
                -(o.confidence or 0.0),
            ),
        )
        chosen: list[Observation] = []
        seen: set[str] = set()
        for obs in ranked:
            if obs.image_id in seen:
                continue
            chosen.append(obs)
            seen.add(obs.image_id)
            if len(chosen) == limit:
                break
        return chosen

    def _crop(
        self, image: Image.Image, obs: Observation
    ) -> tuple[
        Image.Image, list[tuple[float, float]], tuple[int, int, int, int], float
    ]:
        """
        Crop one detection out of its source image.

        Returns the crop, the detection polygon in crop pixel coordinates,
        the crop box in source pixels (after any seam roll) and the
        approximate metres per pixel at the object's range.
        """
        width, height = image.size
        polygon = list(obs.polygon)
        x0, _, x1, _ = obs.bbox
        if obs.is_pano and (x1 - x0) > 0.5:
            # The object straddles the panorama seam: roll the image by half a
            # turn so it becomes contiguous, and shift the polygon to match.
            arr = np.roll(np.asarray(image), width // 2, axis=1)
            image = Image.fromarray(arr)
            polygon = [((x + 0.5) % 1.0, y) for x, y in polygon]
        xs = [x * width for x, _ in polygon]
        ys = [y * height for _, y in polygon]
        left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
        extent = max(right - left, bottom - top, 1.0)
        meters_per_pixel = 0.0
        if obs.range_m and obs.is_pano:
            meters_per_pixel = obs.range_m * 2 * math.pi / width
        margin = outlines.resolve_buffer_px(
            self._crop_buffer_spec, extent, meters_per_pixel
        )
        left, right = left - margin, right + margin
        top, bottom = top - margin, bottom + margin
        # Grow to the minimum size, centred on the detection:
        for lo, hi, size, setter in (
            (left, right, width, 'x'),
            (top, bottom, height, 'y'),
        ):
            span = hi - lo
            if span < self.min_crop_px:
                grow = (self.min_crop_px - span) / 2
                lo, hi = lo - grow, hi + grow
            lo, hi = max(0, int(math.floor(lo))), min(size, int(math.ceil(hi)))
            if setter == 'x':
                left, right = lo, hi
            else:
                top, bottom = lo, hi
        box = (int(left), int(top), int(right), int(bottom))
        crop = image.crop(box)
        pixel_polygon = [(x - box[0], y - box[1]) for x, y in zip(xs, ys, strict=True)]
        return crop, pixel_polygon, box, meters_per_pixel

    def _chosen_views(self, asset: PhysicalAsset) -> list[Observation]:
        """The observations of one asset that will be cropped."""
        records = asset.attributes.get('observations') or []
        observations = [Observation.from_dict(r) for r in records]
        return self.select_observations(observations, self.max_images_per_asset)

    def _process_image(
        self, image_id: str, jobs: list[tuple[PhysicalAsset, Observation]]
    ) -> dict[tuple[str, str], ImageAsset]:
        """
        Download one source image and crop every asset that was seen in it.

        Returns the crops keyed by ``(asset id, image id)``; the image is
        released when this returns.
        """
        raise_if_cancelled(self.cancel_event, 'street-level image extraction')
        image = self._source_image(image_id)
        if image is None:
            return {}
        crops: dict[tuple[str, str], ImageAsset] = {}
        for asset, obs in jobs:
            crops[(asset.id, image_id)] = self._crop_and_save(image, asset, obs)
        return crops

    def _crop_and_save(
        self, image: Image.Image, asset: PhysicalAsset, obs: Observation
    ) -> ImageAsset:
        """Crop one observation out of its source image and write it."""
        crop, pixel_polygon, box, mpp = self._crop(image, obs)
        if self.overlay_asset_outline:
            outlines.draw_outline(
                crop,
                pixel_polygon,
                is_closed=True,
                shape=self.outline_shape,
                buffer=self._outline_buffer_spec,
                width=self._outline_width_spec,
                color=self.outline_color,
                meters_per_pixel=mpp,
            )
        path = (
            self.save_directory / f'{self.image_prefix}_{asset.id}_{obs.image_id}.jpg'
        )
        crop.save(path, quality=92)
        return ImageAsset(
            id=f'{asset.id}_{self.image_prefix}_{obs.image_id}',
            path=path,
            properties={
                'image_id': obs.image_id,
                'captured_at': obs.captured_at,
                'label': obs.label,
                'confidence': obs.confidence,
                'range_m': obs.range_m,
                'bearing': obs.bearing,
                'crop_box': list(box),
                'source_image_size': list(image.size),
                'source': obs.source,
            },
        )

    def __call__(
        self, asset_collection: PhysicalAssetCollection
    ) -> PhysicalAssetCollection:
        """
        Attach street-level crops to every asset that carries observations.

        Args:
            asset_collection (PhysicalAssetCollection):
                Output of :class:`MapillaryFeatureExtractor`. Assets without
                an ``observations`` attribute are left untouched.

        Returns:
            PhysicalAssetCollection: The same collection with crops attached.
        """
        targets = [a for a in asset_collection if a.attributes.get('observations')]
        if not targets:
            logger.warning('No assets with street-level observations to crop.')
            return asset_collection
        # Plan the crops per source image so each image is downloaded once,
        # served to every asset seen in it, and released:
        chosen = {asset.id: self._chosen_views(asset) for asset in targets}
        jobs: dict[str, list[tuple[PhysicalAsset, Observation]]] = defaultdict(list)
        for asset in targets:
            for obs in chosen[asset.id]:
                jobs[obs.image_id].append((asset, obs))
        logger.info(
            f'Cropping up to {self.max_images_per_asset} views for '
            f'{len(targets)} assets from {len(jobs)} images...'
        )
        self.save_directory.mkdir(parents=True, exist_ok=True)
        crops: dict[tuple[str, str], ImageAsset] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._process_image, image_id, image_jobs): image_id
                for image_id, image_jobs in jobs.items()
            }
            try:
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc='Cropping objects'
                ):
                    image_id = futures[future]
                    try:
                        crops.update(future.result())
                    except Exception as exc:
                        if isinstance(exc, OperationCancelled):
                            raise
                        logger.error(f'Failed to crop image {image_id}: {exc}')
            except OperationCancelled:
                for future in futures:
                    future.cancel()
                raise
        # Attach the crops closest view first, whatever order they finished in:
        for asset in targets:
            for obs in chosen[asset.id]:
                crop = crops.get((asset.id, obs.image_id))
                if crop is not None:
                    asset.add_image_assets(crop)
        logger.info(f'Saved {len(crops)} crops to {self.save_directory}.')
        return asset_collection
