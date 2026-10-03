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
# 09-30-2026

"""
Headless workflow engine behind the rapidtools asset-analysis GUI.

This module reproduces the detect / extract / analyze workflow of the
examples without any GUI dependency, so it can be unit tested and scripted:

    1. **Imagery** - a local orthomosaic, or a Bing / Google satellite
       basemap stitched over a region (:meth:`AssetAnalysisWorkflow.download_basemap`).
    2. **Detection** - discover assets in aerial imagery with Meta's SAM 3
       (:meth:`AssetAnalysisWorkflow.detect`), or discover objects along a
       Mapillary street survey from its segmentation metadata
       (:meth:`AssetAnalysisWorkflow.discover_street`); buildings and roads
       are regularized into clean GIS-ready polygons.
    3. **Inference** - crop imagery around every asset (aerial crops, Google
       Street View, Mapillary panoramas or Mapillary object crops) and run a
       vision-language model (cloud APIs such as Gemini, Claude, OpenAI, Meta
       Muse Spark and Qwen, or local checkpoints such as Gemma-4, Llama, Muse
       Glimmer and Qwen) with a user-supplied prompt
       (:meth:`AssetAnalysisWorkflow.analyze`).

The same model backends power the prompt assistant of the GUI
(:meth:`AssetAnalysisWorkflow.assist`), which drafts, refines and reviews
prompts through :mod:`rapidtools.gui.prompt_builder`.
"""

from __future__ import annotations

import gc
import logging
import math
import re
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict

from rapidtools.core import OperationCancelled, PhysicalAssetCollection
from rapidtools.gui.prompt_builder import ASSIST_ACTIONS, PromptAssistant

logger = logging.getLogger(__name__)

# Backend metadata comes straight from the model registry, which is import-
# light (no PyTorch/Transformers) so the GUI starts quickly.
from rapidtools.models.catalogs import PROVIDERS, ModelInfo  # noqa: E402

InferenceBackend = Literal[
    'gemini',
    'claude',
    'openai',
    'muse_spark',
    'qwen',
    'gemma4',
    'llama',
    'muse_glimmer',
    'qwen_vl',
    'hf',
]


def _backend_spec(info: ModelInfo) -> dict[str, Any]:
    """Translate a :class:`ModelInfo` into the dict the web page consumes."""
    spec: dict[str, Any] = {
        'label': info.label,
        'kind': info.kind,
        'description': info.description,
        'default_model': info.analyzer_default_model or info.default_model,
        'models': info.model_ids,
        'attribute_prefix': info.attribute_prefix,
    }
    if info.is_api:
        spec['key_label'] = info.key_label
        spec['key_env'] = info.key_env[0] if info.key_env else ''
    else:
        spec['supports_4bit'] = info.supports_4bit
        spec['supports_batch'] = info.supports_batch
        if info.free_form_model:
            spec['free_form_model'] = True
    return spec


# Every analysis model rapidtools ships, in the order the GUI presents them.
MODEL_BACKENDS: dict[str, dict[str, Any]] = {
    key: _backend_spec(info)
    for key, info in PROVIDERS.items()
    if info.kind in ('api', 'local')
}

# Convenience constants kept for backwards compatibility with earlier GUIs.
DEFAULT_GEMINI_MODEL_ID = MODEL_BACKENDS['gemini']['default_model']
DEFAULT_CLAUDE_MODEL_ID = MODEL_BACKENDS['claude']['default_model']
DEFAULT_OPENAI_MODEL_ID = MODEL_BACKENDS['openai']['default_model']
GEMMA4_MODEL_IDS = tuple(MODEL_BACKENDS['gemma4']['models'])
LLAMA_MODEL_IDS = tuple(MODEL_BACKENDS['llama']['models'])
OPEN_VLM_MODEL_IDS = tuple(MODEL_BACKENDS['hf']['models'])

# Basemaps that can be stitched over a region without an API key.
BASEMAP_PROVIDERS: dict[str, dict[str, Any]] = {
    'bing': {'label': 'Bing', 'default_zoom': 19, 'max_zoom': 20},
    'google': {'label': 'Google', 'default_zoom': 19, 'max_zoom': 21},
}
DETECTION_BASEMAPS = ('bing', 'google', 'recon')

# Where the crops analysed by the model come from, in the order the GUI shows.
IMAGERY_SOURCES: dict[str, dict[str, Any]] = {
    'aerial': {
        'label': 'Aerial crops from the loaded image',
        'description': 'Crop the loaded orthomosaic around each asset footprint.',
        'needs_token': False,
        'supports_outline': True,
    },
    'google_streetview': {
        'label': 'Google Street View',
        'description': 'Nearest official panorama cropped to the footprint; no key.',
        'needs_token': False,
        'supports_outline': False,
    },
    'mapillary': {
        'label': 'Mapillary panoramas',
        'description': 'Views of each footprint from a Mapillary survey (token).',
        'needs_token': True,
        'supports_outline': False,
    },
    'mapillary_objects': {
        'label': 'Mapillary object crops',
        'description': 'Closest views of objects discovered along a street survey.',
        'needs_token': True,
        'supports_outline': True,
    },
}

# Plain-English classes MapillaryFeatureExtractor understands out of the box.
STREET_CLASS_EXAMPLES = (
    'vehicles',
    'utility poles',
    'traffic signs',
    'fire hydrants',
    'street lights',
    'trash cans',
    'benches',
    'mailboxes',
)

MAX_BASEMAP_TILES = 2500


def list_available_models(backend: str, api_key: str = '') -> list[str]:
    """
    Return the model IDs a backend can run.

    Cloud backends query the provider with ``api_key``; local backends return
    the curated list of Hugging Face repositories. Falls back to the static
    catalogue when the provider cannot be reached.
    """
    spec = MODEL_BACKENDS.get(backend)
    if spec is None:
        raise ValueError(f'Unknown inference backend: {backend!r}')
    static = list(spec['models'])
    if backend == 'hf':
        from rapidtools.models import HFVisionInference

        hub = HFVisionInference.search_hub_models()
        return static + [m for m in hub if m not in static]
    if spec['kind'] != 'api':
        return static
    if not api_key.strip():
        return static
    from rapidtools.models import get_model_class
    from rapidtools.models.api_base import BaseAPIInferenceModel

    model_class = get_model_class(backend)
    try:
        if not issubclass(model_class, BaseAPIInferenceModel):
            raise TypeError(f'{model_class.__name__} cannot list remote models.')
        remote = model_class.list_available_models(api_key.strip())
    except Exception as exc:  # noqa: BLE001 - network errors degrade gracefully
        logger.warning(f'Could not list {spec["label"]} models: {exc}')
        remote = []
    if not remote:
        return static
    # Keep the provider order but make sure the default is offered first.
    ordered = [m for m in static if m in remote]
    ordered += [m for m in remote if m not in static]
    return ordered


class WorkflowCancelled(OperationCancelled):
    """Raised when a workflow run is cancelled by the user."""


def slugify(text: str) -> str:
    """Turn a free-form asset name into a filesystem-safe token."""
    slug = re.sub(r'[^0-9A-Za-z]+', '_', text.strip().lower()).strip('_')
    return slug or 'asset'


def parse_asset_list(text: str) -> list[str]:
    """
    Split a user-entered string of asset names into a clean list.

    Names may be separated by commas, semicolons or newlines. Duplicates are
    removed while preserving the order the user typed them in.

    >>> parse_asset_list('building, tree;  road\nbuilding')
    ['building', 'tree', 'road']
    """
    seen: set[str] = set()
    assets: list[str] = []
    for raw in re.split(r'[,;\n]+', text):
        name = ' '.join(raw.split())
        if name and name.lower() not in seen:
            seen.add(name.lower())
            assets.append(name)
    return assets


@dataclass
class DetectionSettings:
    """User-configurable options for the asset detection stage."""

    raster_path: Path
    output_dir: Path
    assets: list[str]
    asset_size_m: float = 50.0
    detect_in_recon_imagery: bool = False
    basemap: str = 'bing'
    threshold: float = 0.5
    mask_threshold: float = 0.4
    overlap_ratio: float = 0.25
    batch_size: int = 4
    crop_buffer: str = '20 m'
    regularize: bool = True
    merge_overlaps: bool = True
    bing_max_workers: int = 10

    def __post_init__(self) -> None:
        """
        Coerce paths and validate the asset list, size and basemap.

        ``detect_in_recon_imagery=True`` is the legacy spelling of
        ``basemap='recon'``; the two are kept in sync.
        """
        self.raster_path = Path(self.raster_path)
        self.output_dir = Path(self.output_dir)
        if not self.assets:
            raise ValueError('At least one asset name is required.')
        if self.asset_size_m <= 0:
            raise ValueError('Approximate asset size must be positive.')
        self.basemap = str(self.basemap or 'bing').strip().lower()
        if self.detect_in_recon_imagery:
            self.basemap = 'recon'
        if self.basemap not in DETECTION_BASEMAPS:
            raise ValueError(
                f'basemap must be one of {DETECTION_BASEMAPS}, got {self.basemap!r}.'
            )
        self.detect_in_recon_imagery = self.basemap == 'recon'


class _OutlineKwargs(TypedDict):
    """Outline keyword arguments shared by the image extractors."""

    overlay_asset_outline: bool
    outline_shape: str
    outline_buffer: float | str
    outline_width: int | float | str
    outline_color: str


@dataclass
class InferenceSettings:
    """User-configurable options for the VLM inference stage."""

    raster_path: Path
    output_dir: Path
    prompt: str
    backend: InferenceBackend = 'gemini'
    api_key: str = ''
    model_id: str = ''
    max_workers: int = 5
    batch_size: int = 4
    temperature: float = 0.4
    max_tokens: int = 2048
    json_mode: bool = False
    load_in_4bit: bool = True
    overlay_asset_outline: bool = True
    outline_shape: str = 'geometry'
    outline_buffer: float | str = 0
    outline_width: int | float | str = 6
    outline_color: str = 'red'
    image_prefix: str = 'aerial'
    only_analyze_inference_crops: bool = True
    # Where the analysed crops come from (an IMAGERY_SOURCES key) and the
    # options of the street-level extractors:
    imagery: str = 'aerial'
    min_footprint_coverage: float | None = None
    pad_edges: bool | None = None
    street_search_radius_m: float = 50.0
    street_max_images: int = 1
    street_vertical_crop: tuple[float, float] = (0.0, 1.0)
    mapillary_token: str = ''
    mapillary_start_date: str = ''
    mapillary_end_date: str = ''
    mapillary_rapid_only: bool = True
    mapillary_max_images: int = 4
    object_image_size: str = '2048'
    object_crop_buffer: float | str = '25%'

    def __post_init__(self) -> None:
        """
        Coerce paths, resolve the default model and validate the backend.

        Raises:
            ValueError: If the prompt is empty, the backend or imagery source
                is unknown, an API backend was selected without a key, a
                Mapillary source has no token, or the outline options are
                invalid.
        """
        self.raster_path = Path(self.raster_path)
        self.output_dir = Path(self.output_dir)
        if not self.prompt.strip():
            raise ValueError('An inference prompt is required.')
        self.imagery = str(self.imagery or 'aerial').strip().lower()
        if self.imagery not in IMAGERY_SOURCES:
            raise ValueError(
                f'imagery must be one of {tuple(IMAGERY_SOURCES)}, '
                f'got {self.imagery!r}.'
            )
        if (
            IMAGERY_SOURCES[self.imagery]['needs_token']
            and not str(self.mapillary_token).strip()
        ):
            raise ValueError(
                f'A Mapillary access token is required for '
                f'{IMAGERY_SOURCES[self.imagery]["label"]}.'
            )
        if self.min_footprint_coverage is not None and not (
            0.0 <= float(self.min_footprint_coverage) <= 1.0
        ):
            raise ValueError('min_footprint_coverage must be between 0 and 1.')
        if self.street_search_radius_m <= 0:
            raise ValueError('The Street View search radius must be positive.')
        if self.street_max_images < 1 or self.mapillary_max_images < 1:
            raise ValueError('At least one image per asset is required.')
        top, bottom = (float(v) for v in self.street_vertical_crop)
        if not (0.0 <= top < bottom <= 1.0):
            raise ValueError(
                'street_vertical_crop must be (top, bottom) fractions with '
                '0 <= top < bottom <= 1.'
            )
        self.street_vertical_crop = (top, bottom)
        # Validate the outline options with the extractor's own parsers so a
        # bad value is rejected by the API instead of failing mid-run:
        from rapidtools.processing.image_extractors import (
            OUTLINE_SHAPES,
            AerialImageryExtractor,
        )

        if self.outline_shape not in OUTLINE_SHAPES:
            raise ValueError(
                f'outline_shape must be one of {OUTLINE_SHAPES}, '
                f'got {self.outline_shape!r}.'
            )
        AerialImageryExtractor._parse_outline_buffer(self.outline_buffer)
        AerialImageryExtractor._parse_outline_width(self.outline_width)
        if not str(self.outline_color).strip():
            raise ValueError('An outline colour is required.')
        from PIL import ImageColor

        ImageColor.getrgb(str(self.outline_color))
        spec = MODEL_BACKENDS.get(self.backend)
        if spec is None:
            raise ValueError(f'Unknown inference backend: {self.backend!r}')
        if not self.model_id.strip():
            self.model_id = spec['default_model']
        if spec['kind'] == 'api' and not self.api_key.strip():
            raise ValueError(
                f'{spec["key_label"]} is required for the {spec["label"]} backend.'
            )

    @property
    def backend_spec(self) -> dict[str, Any]:
        """The :data:`MODEL_BACKENDS` record describing the selected backend."""
        return MODEL_BACKENDS[self.backend]


@dataclass
class RegionImagerySettings:
    """
    Options for :meth:`AssetAnalysisWorkflow.download_basemap`.

    Either the four WGS84 bounds or ``geojson_path`` (whose extent is used)
    must be given.

    Example:
        >>> RegionImagerySettings(
        ...     output_dir='out', provider='google', zoom=18,
        ...     min_lon=-118.15, min_lat=34.18, max_lon=-118.14, max_lat=34.19,
        ... ).provider
        'google'
    """

    output_dir: Path
    provider: str = 'bing'
    zoom: int = 19
    min_lon: float | None = None
    min_lat: float | None = None
    max_lon: float | None = None
    max_lat: float | None = None
    geojson_path: Path | None = None
    max_tiles: int = MAX_BASEMAP_TILES
    max_workers: int = 10

    def __post_init__(self) -> None:
        """Validate the provider, zoom and region definition."""
        self.output_dir = Path(self.output_dir)
        self.provider = str(self.provider or 'bing').strip().lower()
        if self.provider not in BASEMAP_PROVIDERS:
            raise ValueError(
                f'provider must be one of {tuple(BASEMAP_PROVIDERS)}, '
                f'got {self.provider!r}.'
            )
        self.zoom = int(self.zoom)
        if not 10 <= self.zoom <= BASEMAP_PROVIDERS[self.provider]['max_zoom']:
            raise ValueError(
                f'zoom must be between 10 and '
                f'{BASEMAP_PROVIDERS[self.provider]["max_zoom"]} for '
                f'{BASEMAP_PROVIDERS[self.provider]["label"]}.'
            )
        if self.geojson_path is not None:
            self.geojson_path = Path(self.geojson_path)
            return
        if (
            self.min_lon is None
            or self.min_lat is None
            or self.max_lon is None
            or self.max_lat is None
        ):
            raise ValueError(
                'Give the region as min/max longitude and latitude, or as a '
                'GeoJSON file.'
            )
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = (
            float(self.min_lon),
            float(self.min_lat),
            float(self.max_lon),
            float(self.max_lat),
        )
        if not (
            -180 <= self.min_lon < self.max_lon <= 180
            and -85 <= self.min_lat < self.max_lat <= 85
        ):
            raise ValueError(
                'The bounding box must satisfy min < max with longitudes in '
                '[-180, 180] and latitudes in [-85, 85].'
            )


@dataclass
class StreetDetectionSettings:
    """
    Options for :meth:`AssetAnalysisWorkflow.discover_street`.

    Objects of ``classes`` are discovered in every Mapillary image of the
    region, which defaults to the extent of ``raster_path`` so the GUI can
    draw the results on the loaded imagery.

    Example:
        >>> StreetDetectionSettings(
        ...     output_dir='out', classes=['vehicles'], access_token='MLY|...',
        ...     raster_path='scene.tiff',
        ... ).frame_spacing_m
        3.0
    """

    output_dir: Path
    classes: list[str]
    access_token: str
    raster_path: Path | None = None
    region: Any = None
    start_date: str = ''
    end_date: str = ''
    filter_rapid_only: bool = True
    detection_source: str = 'auto'
    frame_spacing_m: float = 3.0
    min_observations: int = 2
    cluster_radius_m: float = 4.0
    camera_height_m: float = 2.4
    max_range_m: float = 60.0
    max_workers: int = 10

    def __post_init__(self) -> None:
        """Validate the classes, token, region source and numeric options."""
        self.output_dir = Path(self.output_dir)
        self.classes = parse_asset_list(
            self.classes if isinstance(self.classes, str) else ', '.join(self.classes)
        )
        if not self.classes:
            raise ValueError('Enter at least one object class (e.g. "vehicles").')
        self.access_token = str(self.access_token or '').strip()
        if not self.access_token:
            raise ValueError('A Mapillary access token is required.')
        if self.raster_path is not None:
            self.raster_path = Path(self.raster_path)
        if self.raster_path is None and self.region is None:
            raise ValueError(
                'Load imagery (or give a region) to define the survey area.'
            )
        if self.detection_source not in ('auto', 'mapillary', 'sam3'):
            raise ValueError("detection_source must be 'auto', 'mapillary' or 'sam3'.")
        if self.frame_spacing_m < 0 or self.cluster_radius_m <= 0:
            raise ValueError('Frame spacing must be >= 0 and cluster radius > 0.')
        if self.min_observations < 1:
            raise ValueError('min_observations must be at least 1.')
        if self.camera_height_m <= 0 or self.max_range_m <= 0:
            raise ValueError('Camera height and maximum range must be positive.')


@dataclass
class AssistSettings:
    """
    Options for :meth:`AssetAnalysisWorkflow.assist`.

    ``action`` is one of :data:`~rapidtools.gui.prompt_builder.ASSIST_ACTIONS`;
    the remaining text fields feed the matching
    :class:`~rapidtools.gui.prompt_builder.PromptAssistant` method.

    Example:
        >>> AssistSettings(action='draft', backend='gemma4', brief='grade roofs').action
        'draft'
    """

    action: str
    backend: InferenceBackend = 'gemma4'
    api_key: str = ''
    model_id: str = ''
    load_in_4bit: bool = True
    brief: str = ''
    instruction: str = ''
    prompt_text: str = ''
    class_value: str = ''
    spec: dict[str, Any] | None = None
    context: dict[str, Any] = field(default_factory=dict)
    temperature: float = 0.2
    max_tokens: int = 4096

    def __post_init__(self) -> None:
        """Validate the action and backend, and require a key for API backends."""
        self.action = str(self.action or '').strip().lower()
        if self.action not in ASSIST_ACTIONS:
            raise ValueError(
                f'action must be one of {ASSIST_ACTIONS}, got {self.action!r}.'
            )
        spec = MODEL_BACKENDS.get(self.backend)
        if spec is None:
            raise ValueError(f'Unknown assistant backend: {self.backend!r}')
        if not self.model_id.strip():
            self.model_id = spec['default_model']
        if spec['kind'] == 'api' and not self.api_key.strip():
            raise ValueError(
                f'{spec["key_label"]} is required to use {spec["label"]} as the '
                'prompt assistant.'
            )
        if self.spec is not None and not isinstance(self.spec, dict):
            raise ValueError('spec must be a JSON object.')

    @property
    def backend_spec(self) -> dict[str, Any]:
        """The :data:`MODEL_BACKENDS` record describing the selected backend."""
        return MODEL_BACKENDS[self.backend]


@dataclass
class DetectionResult:
    """Outputs of :meth:`AssetAnalysisWorkflow.detect`."""

    collection: PhysicalAssetCollection
    detection_raster: Path
    per_asset_geojson: dict[str, Path] = field(default_factory=dict)
    combined_geojson: Path | None = None
    basemap: str | None = None


@dataclass
class StreetDetectionResult:
    """Outputs of :meth:`AssetAnalysisWorkflow.discover_street`."""

    collection: PhysicalAssetCollection
    geojson_path: Path | None = None
    n_images: int = 0


@dataclass
class InferenceResult:
    """Outputs of :meth:`AssetAnalysisWorkflow.analyze`."""

    collection: PhysicalAssetCollection
    geojson_path: Path
    n_input: int
    n_analyzed: int


class AssetAnalysisWorkflow:
    """
    Orchestrates asset detection and VLM inference on an aerial raster.

    Args:
        progress_callback (Callable[[str], None] | None):
            Optional function receiving short human-readable status strings
            (e.g. ``'Scanning for tree...'``). The GUI uses this to update its
            status bar; scripts may pass ``print`` or leave it ``None``.

    Example:
        >>> wf = AssetAnalysisWorkflow()
        >>> detection = wf.detect(DetectionSettings(
        ...     raster_path='eaton_patch_20250214.tiff',
        ...     output_dir='output',
        ...     assets=['building', 'tree'],
        ... ))
        >>> inference = wf.analyze(detection.collection, InferenceSettings(
        ...     raster_path='eaton_patch_20250214.tiff',
        ...     output_dir='output',
        ...     prompt='Describe the damage to the outlined asset.',
        ...     api_key='...',
        ... ))
    """

    def __init__(self, progress_callback: Callable[[str], None] | None = None) -> None:
        """Create an idle workflow with a fresh cancellation event."""
        self._progress_callback = progress_callback
        self._cancel_event = threading.Event()
        # (cache key, model) of the prompt assistant, kept between requests so
        # a local model is loaded once per conversation.
        self._assistant: tuple[tuple[Any, ...], Any] | None = None

    # ------------------------------------------------------------------ utils
    def cancel(self) -> None:
        """Request cancellation. Honoured at the next stage boundary."""
        self._cancel_event.set()

    def reset(self) -> None:
        """Clear any pending cancellation request."""
        self._cancel_event.clear()

    def _check_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise WorkflowCancelled('Workflow cancelled by user.')

    def _progress(self, message: str) -> None:
        logger.info(message)
        if self._progress_callback is not None:
            self._progress_callback(message)

    # -------------------------------------------------------------- detection
    def detect(self, settings: DetectionSettings) -> DetectionResult:
        """
        Detect one or more asset types in the raster with SAM 3.

        Mirrors Part 1 of the generalized example notebook. When
        ``settings.detect_in_recon_imagery`` is ``False`` a Bing basemap is
        stitched over the raster extent and used for detection instead (useful
        when assets are destroyed in the post-disaster imagery).

        Returns:
            DetectionResult: The merged collection of every requested asset
            type plus the paths of the GeoJSON files written to disk.
        """
        self.reset()
        raster_path = settings.raster_path.expanduser().resolve()
        if not raster_path.is_file():
            raise FileNotFoundError(f'Raster not found: {raster_path}')
        output_dir = settings.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            return self._detect(settings, raster_path, output_dir)
        except WorkflowCancelled:
            raise
        except OperationCancelled as exc:
            self._progress('Detection cancelled.')
            raise WorkflowCancelled(str(exc)) from exc

    def _detect(
        self, settings: DetectionSettings, raster_path: Path, output_dir: Path
    ) -> DetectionResult:
        # Heavy imports are deferred so the GUI starts quickly.
        from rapidtools import (
            AerialImageryExtractor,
            BingOrthomosaicExtractor,
            BoundingBox,
            GoogleOrthomosaicExtractor,
            SAM3OrthoFeatureExtractor,
        )

        # 1. Decide which raster to scan.
        self._release_local_assistant()
        if settings.basemap == 'recon':
            detection_raster = raster_path
        else:
            label = BASEMAP_PROVIDERS[settings.basemap]['label']
            detection_raster = (
                output_dir / f'{raster_path.stem}_{settings.basemap}.tiff'
            )
            if detection_raster.is_file():
                self._progress(
                    f'Reusing existing {label} basemap: {detection_raster.name}'
                )
            else:
                self._progress(f'Stitching {label} basemap over the raster extent...')
                region = BoundingBox.from_raster(raster_path)
                extractor_cls = (
                    BingOrthomosaicExtractor
                    if settings.basemap == 'bing'
                    else GoogleOrthomosaicExtractor
                )
                extractor = extractor_cls(
                    max_workers=settings.bing_max_workers,
                    cancel_event=self._cancel_event,
                )
                detection_raster = Path(
                    extractor(region=region, output_path=detection_raster)
                )
        self._check_cancelled()

        # 2. Load SAM 3 once and reuse it for every asset prompt.
        self._progress('Loading SAM 3 (first run downloads ~3.4 GB of weights)...')
        sam_extractor = SAM3OrthoFeatureExtractor(
            prompt=settings.assets[0],
            patch_size=settings.asset_size_m,
            unit='meters',
            overlap_ratio=settings.overlap_ratio,
            batch_size=settings.batch_size,
            threshold=settings.threshold,
            mask_threshold=settings.mask_threshold,
            merge_overlaps=settings.merge_overlaps,
            cancel_event=self._cancel_event,
        )

        combined = PhysicalAssetCollection()
        per_asset_geojson: dict[str, Path] = {}

        for asset_name in settings.assets:
            self._check_cancelled()
            slug = slugify(asset_name)
            self._progress(f"Scanning for '{asset_name}'...")
            sam_extractor.prompt = asset_name
            prelim = sam_extractor(detection_raster)
            prelim.to_geojson(output_dir / f'{slug}_preliminary.geojson')

            if len(prelim) == 0:
                logger.warning(f"No '{asset_name}' assets were detected.")
                continue

            # Crop imagery around each detection for regularization context.
            self._progress(
                f"Cropping imagery around {len(prelim)} '{asset_name}' detections..."
            )
            crop_extractor = AerialImageryExtractor(
                dataset=detection_raster,
                save_directory=output_dir / 'asset_crops' / slug,
                buffer_asset=settings.crop_buffer,
                force_square_image=True,
                image_prefix=f'{slug}_crop',
                cancel_event=self._cancel_event,
            )
            with_images = crop_extractor(prelim)

            final = with_images
            if settings.regularize:
                final = self._regularize(asset_name, with_images)

            final_path = output_dir / f'{slug}_final.geojson'
            final.to_geojson(final_path, ignore_properties=['image_assets'])
            per_asset_geojson[asset_name] = final_path
            self._progress(f"Finished '{asset_name}': {len(final)} assets.")
            combined.merge(final, strategy='skip')

        combined_path: Path | None = None
        if len(combined):
            combined_path = output_dir / 'assets_final.geojson'
            combined.to_geojson(combined_path, ignore_properties=['image_assets'])
            self._progress(
                f'Detection complete: {len(combined)} assets written to '
                f'{combined_path.name}'
            )
        else:
            self._progress('Detection complete: no assets were found.')

        return DetectionResult(
            collection=combined,
            detection_raster=detection_raster,
            per_asset_geojson=per_asset_geojson,
            combined_geojson=combined_path,
            basemap=None if settings.basemap == 'recon' else settings.basemap,
        )

    def _regularize(
        self, asset_name: str, collection: PhysicalAssetCollection
    ) -> PhysicalAssetCollection:
        """Apply the building or road regularizer when the asset name matches."""
        from rapidtools import BuildingRegularizer, RoadwayRegularizer

        lowered = asset_name.lower()
        if 'building' in lowered or 'house' in lowered or 'structure' in lowered:
            self._progress(f"Regularizing '{asset_name}' footprints...")
            regularizer = BuildingRegularizer(
                batch_size=8, cancel_event=self._cancel_event
            )
            return regularizer(collection)
        if 'road' in lowered or 'street' in lowered:
            self._progress(f"Regularizing '{asset_name}' polygons...")
            _centerlines, polygons = RoadwayRegularizer()(collection)
            return polygons
        return collection

    # -------------------------------------------------------------- inference
    def analyze(
        self,
        collection: PhysicalAssetCollection,
        settings: InferenceSettings,
    ) -> InferenceResult:
        """
        Crop the raster around every asset and run VLM inference on the crops.

        Mirrors Part 2 of the generalized example notebook.

        Returns:
            InferenceResult: The collection of assets that received at least
            one inferred attribute, and the GeoJSON path it was written to.
        """
        from rapidtools import Pipeline

        self.reset()
        self._release_local_assistant()
        if len(collection) == 0:
            raise ValueError('The asset collection is empty; nothing to analyze.')
        raster_path = settings.raster_path.expanduser().resolve()
        if settings.imagery == 'aerial' and not raster_path.is_file():
            raise FileNotFoundError(f'Raster not found: {raster_path}')
        output_dir = settings.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = (output_dir / 'inference_imagery').resolve()

        extractor = self._build_extractor(settings, raster_path, image_dir)

        # Detection may have attached basemap crops to the assets; restrict the
        # VLM to the crops produced for inference unless the user opts out.
        def _is_inference_crop(image) -> bool:
            try:
                return Path(image.path).resolve().is_relative_to(image_dir)
            except (TypeError, ValueError):
                return False

        image_filter = (
            _is_inference_crop if settings.only_analyze_inference_crops else None
        )

        analyzer = self._build_analyzer(settings, image_filter, self._cancel_event)

        pipeline = Pipeline(cancel_event=self._cancel_event)
        pipeline.add_step(extractor)
        pipeline.add_step(analyzer)

        # Detected assets already carry attributes (asset_type, source_model...),
        # so count an asset as analyzed only if inference added new ones.
        attributes_before = {a.id: set(a.attributes) for a in collection}

        self._progress(f'Running inference on {len(collection)} assets...')
        try:
            processed = pipeline.run(collection)
        except OperationCancelled as exc:
            # Keep whatever was analyzed before the user pressed cancel.
            partial = collection.filter_empty()
            partial_path = output_dir / 'assets_inferred_partial.geojson'
            partial.to_geojson(partial_path, ignore_properties=['image_assets'])
            self._progress(
                f'Inference cancelled. Partial results for {len(partial)} assets '
                f'written to {partial_path.name}'
            )
            raise WorkflowCancelled(str(exc)) from exc
        self._check_cancelled()

        analyzed = processed.filter_empty()
        n_analyzed = sum(
            1
            for a in analyzed
            if set(a.attributes) - attributes_before.get(a.id, set())
        )
        geojson_path = output_dir / 'assets_inferred.geojson'
        analyzed.to_geojson(geojson_path, ignore_properties=['image_assets'])
        self._progress(
            f'Inference complete: {n_analyzed}/{len(collection)} assets '
            f'received new attributes. Results written to {geojson_path.name}'
        )
        return InferenceResult(
            collection=analyzed,
            geojson_path=geojson_path,
            n_input=len(collection),
            n_analyzed=n_analyzed,
        )

    def _build_extractor(
        self, settings: InferenceSettings, raster_path: Path, image_dir: Path
    ):
        """
        Instantiate the imagery extractor for ``settings.imagery``.

        Every extractor writes below ``image_dir`` so the inference-crop
        filter of :meth:`analyze` applies to all of them.
        """
        from rapidtools import (
            AerialImageryExtractor,
            GoogleStreetViewImageExtractor,
            MapillaryImageExtractor,
            MapillaryObjectImageExtractor,
        )

        outline: _OutlineKwargs = {
            'overlay_asset_outline': settings.overlay_asset_outline,
            'outline_shape': settings.outline_shape,
            'outline_buffer': settings.outline_buffer,
            'outline_width': settings.outline_width,
            'outline_color': settings.outline_color,
        }
        if settings.imagery == 'aerial':
            return AerialImageryExtractor(
                dataset=raster_path,
                save_directory=image_dir,
                image_prefix=settings.image_prefix,
                keep_multiple_copies=True,
                min_footprint_coverage=settings.min_footprint_coverage,
                pad_edges=settings.pad_edges,
                cancel_event=self._cancel_event,
                **outline,
            )
        if settings.imagery == 'google_streetview':
            self._progress('Fetching Google Street View panoramas...')
            return GoogleStreetViewImageExtractor(
                save_directory=image_dir / 'streetview',
                search_radius_m=settings.street_search_radius_m,
                max_images_per_asset=settings.street_max_images,
                vertical_crop=settings.street_vertical_crop,
                image_prefix='gsv',
                max_workers=settings.max_workers,
                cancel_event=self._cancel_event,
            )
        if settings.imagery == 'mapillary':
            self._progress('Fetching Mapillary panoramas...')
            return MapillaryImageExtractor(
                access_token=settings.mapillary_token.strip(),
                save_directory=image_dir / 'mapillary',
                start_date=settings.mapillary_start_date,
                end_date=settings.mapillary_end_date,
                filter_rapid_only=settings.mapillary_rapid_only,
                smart_crop=True,
                image_prefix='street',
                max_images_per_asset=settings.mapillary_max_images,
                max_workers=settings.max_workers,
                cancel_event=self._cancel_event,
            )
        self._progress('Cropping Mapillary views of each object...')
        return MapillaryObjectImageExtractor(
            save_directory=image_dir / 'street_objects',
            access_token=settings.mapillary_token.strip(),
            max_images_per_asset=settings.mapillary_max_images,
            image_size=settings.object_image_size,
            crop_buffer=settings.object_crop_buffer,
            image_prefix='street',
            max_workers=settings.max_workers,
            cancel_event=self._cancel_event,
            **outline,
        )

    @staticmethod
    def _model_kwargs(
        backend: str,
        model_id: str,
        api_key: str,
        load_in_4bit: bool,
        temperature: float,
        max_tokens: int,
        max_workers: int,
    ) -> dict[str, Any]:
        """Keyword arguments for :func:`rapidtools.models.load` of ``backend``."""
        info = PROVIDERS[backend]
        kwargs: dict[str, Any] = {
            'model_id': model_id or MODEL_BACKENDS[backend]['default_model'],
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
        if info.is_api:
            kwargs['api_key'] = api_key.strip()
            kwargs['max_workers'] = max_workers
        elif info.supports_4bit:
            kwargs['load_in_4bit'] = load_in_4bit
        return kwargs

    @classmethod
    def _build_analyzer(
        cls,
        settings: InferenceSettings,
        image_filter,
        cancel_event: threading.Event | None = None,
    ):
        """
        Instantiate the model for the selected backend and wrap it.

        Uses :func:`rapidtools.models.load` and
        :class:`~rapidtools.processing.AssetAnalyzer`, so every registered
        provider is supported without backend-specific code.
        """
        from rapidtools.models import GenerationConfig, load
        from rapidtools.processing import AssetAnalyzer

        info = PROVIDERS[settings.backend]
        model = load(
            settings.backend,
            **cls._model_kwargs(
                settings.backend,
                settings.model_id,
                settings.api_key,
                settings.load_in_4bit,
                settings.temperature,
                settings.max_tokens,
                settings.max_workers,
            ),
        )

        return AssetAnalyzer(
            model,
            prompt=settings.prompt,
            max_workers=settings.max_workers if info.is_api else 1,
            batch_size=settings.batch_size if info.supports_batch else 1,
            image_filter=image_filter,
            cancel_event=cancel_event,
            generation=GenerationConfig(json_mode=settings.json_mode),
        )

    # ------------------------------------------------------------- basemap
    def download_basemap(self, settings: RegionImagerySettings) -> Path:
        """
        Stitch a Bing or Google satellite basemap over a region into a GeoTIFF.

        The region comes from the four WGS84 bounds or the extent of a
        GeoJSON file. The tile count is checked against
        ``settings.max_tiles`` before any download starts.

        Args:
            settings (RegionImagerySettings): Provider, zoom and region.

        Returns:
            Path: The written GeoTIFF, named after the provider, zoom and
            bounds inside ``settings.output_dir``.

        Raises:
            ValueError: If the region covers more than ``max_tiles`` tiles.
            FileNotFoundError: If ``geojson_path`` does not exist.
        """
        from rapidtools import (
            BingOrthomosaicExtractor,
            BoundingBox,
            GoogleOrthomosaicExtractor,
        )

        self.reset()
        output_dir = settings.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if settings.geojson_path is not None:
            geojson = settings.geojson_path.expanduser().resolve()
            if not geojson.is_file():
                raise FileNotFoundError(f'GeoJSON not found: {geojson}')
            region = BoundingBox.from_geojson(geojson)
            bounds = region.bounds
        else:
            if (
                settings.min_lon is None
                or settings.min_lat is None
                or settings.max_lon is None
                or settings.max_lat is None
            ):
                raise ValueError('The region bounds are incomplete.')
            bounds = (
                settings.min_lon,
                settings.min_lat,
                settings.max_lon,
                settings.max_lat,
            )
            region = BoundingBox(*bounds)
        n_tiles = estimate_tile_count(bounds, settings.zoom)
        if n_tiles > settings.max_tiles:
            raise ValueError(
                f'The region needs about {n_tiles:,} tiles at zoom {settings.zoom} '
                f'(limit {settings.max_tiles:,}). Lower the zoom level or shrink '
                'the region.'
            )
        label = BASEMAP_PROVIDERS[settings.provider]['label']
        stem = f'{settings.provider}_z{settings.zoom}_' + '_'.join(
            f'{v:.5f}'.replace('-', 'm').replace('.', 'p') for v in bounds
        )
        output_path = output_dir / f'{stem}.tiff'
        if output_path.is_file():
            self._progress(f'Reusing existing {label} imagery: {output_path.name}')
            return output_path
        self._progress(
            f'Downloading about {n_tiles:,} {label} tiles at zoom {settings.zoom}...'
        )
        extractor_cls = (
            BingOrthomosaicExtractor
            if settings.provider == 'bing'
            else GoogleOrthomosaicExtractor
        )
        extractor = extractor_cls(
            zoom_level=settings.zoom,
            max_workers=settings.max_workers,
            cancel_event=self._cancel_event,
        )
        try:
            path = Path(extractor(region=region, output_path=output_path))
        except WorkflowCancelled:
            raise
        except OperationCancelled as exc:
            raise WorkflowCancelled(str(exc)) from exc
        self._check_cancelled()
        self._progress(f'{label} imagery written to {path.name}')
        return path

    # -------------------------------------------------------- street survey
    def discover_street(
        self, settings: StreetDetectionSettings
    ) -> StreetDetectionResult:
        """
        Discover objects along a Mapillary street survey.

        Wraps :class:`~rapidtools.processing.MapillaryFeatureExtractor`: the
        requested classes are located in every image of the region from
        Mapillary's segmentation metadata, the survey vehicle is removed,
        sightings are triangulated and clustered into one point asset per
        object. No pixels are downloaded; crops are made at inference time
        with the ``'mapillary_objects'`` imagery source.

        Args:
            settings (StreetDetectionSettings): Classes, token, region and
                localization options.

        Returns:
            StreetDetectionResult: The point assets and the GeoJSON they were
            written to (``None`` when nothing was found).

        Raises:
            FileNotFoundError: If ``raster_path`` is given but missing.
            WorkflowCancelled: If cancelled.
        """
        from rapidtools import BoundingBox, MapillaryFeatureExtractor

        self.reset()
        self._release_local_assistant()
        output_dir = settings.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        region = settings.region
        if region is None:
            if settings.raster_path is None:
                raise ValueError(
                    'Load imagery (or give a region) to define the survey area.'
                )
            raster_path = settings.raster_path.expanduser().resolve()
            if not raster_path.is_file():
                raise FileNotFoundError(f'Raster not found: {raster_path}')
            region = BoundingBox.from_raster(raster_path)
        classes = ', '.join(settings.classes)
        self._progress(f'Searching the Mapillary survey for {classes}...')
        extractor = MapillaryFeatureExtractor(
            classes=settings.classes,
            access_token=settings.access_token,
            region=region,
            start_date=settings.start_date,
            end_date=settings.end_date,
            filter_rapid_only=settings.filter_rapid_only,
            detection_source=settings.detection_source,
            frame_spacing_m=settings.frame_spacing_m,
            min_observations=settings.min_observations,
            cluster_radius_m=settings.cluster_radius_m,
            camera_height_m=settings.camera_height_m,
            max_range_m=settings.max_range_m,
            save_directory=output_dir / 'street_detections',
            max_workers=settings.max_workers,
            cancel_event=self._cancel_event,
        )
        try:
            collection = extractor()
        except WorkflowCancelled:
            raise
        except OperationCancelled as exc:
            self._progress('Street discovery cancelled.')
            raise WorkflowCancelled(str(exc)) from exc
        self._check_cancelled()

        geojson_path: Path | None = None
        n_images = sum(len(a.image_assets) for a in collection)
        if len(collection):
            geojson_path = output_dir / 'street_objects.geojson'
            collection.to_geojson(geojson_path, ignore_properties=['image_assets'])
            self._progress(
                f'Street discovery complete: {len(collection)} objects written to '
                f'{geojson_path.name}'
            )
        else:
            self._progress('Street discovery complete: no objects were found.')
        return StreetDetectionResult(
            collection=collection, geojson_path=geojson_path, n_images=n_images
        )

    # ------------------------------------------------------------ assistant
    def assist(self, settings: AssistSettings) -> dict[str, Any]:
        """
        Run one prompt-assistant action with a rapidtools model backend.

        The model is loaded through :func:`rapidtools.models.load` and cached
        until a different backend or model is requested, or until detection or
        inference needs the GPU (local models are then released).

        Args:
            settings (AssistSettings): The action and its inputs.

        Returns:
            dict[str, Any]: The
            :meth:`~rapidtools.gui.prompt_builder.PromptAssistant.run` result:
            ``spec``, ``text`` or ``indicators``.

        Raises:
            ValueError: If the model reply cannot be used.
        """
        self.reset()
        label = settings.backend_spec['label']
        self._progress(f'Asking {label} ({settings.model_id})...')
        model = self._assistant_model(settings)
        assistant = PromptAssistant(
            model, temperature=settings.temperature, max_tokens=settings.max_tokens
        )
        result = assistant.run(
            settings.action,
            brief=settings.brief,
            instruction=settings.instruction,
            prompt_text=settings.prompt_text,
            class_value=settings.class_value,
            spec=settings.spec,
            context=settings.context,
        )
        self._check_cancelled()
        self._progress(f'{label} answered.')
        return result

    def _assistant_model(self, settings: AssistSettings):
        """Return the cached assistant model, loading it when the key changed."""
        from rapidtools.models import load

        key = (
            settings.backend,
            settings.model_id,
            settings.api_key.strip(),
            settings.load_in_4bit,
        )
        if self._assistant is not None and self._assistant[0] == key:
            return self._assistant[1]
        self.release_assistant()
        if settings.backend_spec['kind'] != 'api':
            self._progress(
                f'Loading {settings.model_id} for the prompt assistant (first use '
                'downloads the weights)...'
            )
        model = load(
            settings.backend,
            **self._model_kwargs(
                settings.backend,
                settings.model_id,
                settings.api_key,
                settings.load_in_4bit,
                settings.temperature,
                settings.max_tokens,
                1,
            ),
        )
        self._assistant = (key, model)
        return model

    def release_assistant(self) -> None:
        """Drop the cached assistant model and free GPU memory if it was local."""
        if self._assistant is None:
            return
        backend = self._assistant[0][0]
        self._assistant = None
        gc.collect()
        torch = sys.modules.get('torch')
        if MODEL_BACKENDS.get(backend, {}).get('kind') == 'local' and torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - best effort
                pass

    def _release_local_assistant(self) -> None:
        """Free a cached *local* assistant model before GPU-heavy work."""
        if (
            self._assistant is not None
            and MODEL_BACKENDS.get(self._assistant[0][0], {}).get('kind') == 'local'
        ):
            self.release_assistant()


def estimate_tile_count(bounds: tuple[float, float, float, float], zoom: int) -> int:
    """
    Number of Web Mercator tiles covering ``bounds`` at ``zoom``.

    Args:
        bounds (tuple[float, float, float, float]): ``(min_lon, min_lat,
            max_lon, max_lat)`` in WGS84.
        zoom (int): Tile zoom level.

    Returns:
        int: The tile count (at least 1).

    Example:
        >>> estimate_tile_count((-118.15, 34.18, -118.14, 34.19), 19)
        288
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    n = 2**zoom

    def tile_x(lon: float) -> int:
        return int((lon + 180.0) / 360.0 * n)

    def tile_y(lat: float) -> int:
        lat_rad = math.radians(max(min(lat, 85.05), -85.05))
        return int(
            (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi)
            / 2
            * n
        )

    width = abs(tile_x(max_lon) - tile_x(min_lon)) + 1
    height = abs(tile_y(min_lat) - tile_y(max_lat)) + 1
    return max(1, width * height)
