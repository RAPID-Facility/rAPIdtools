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
# 09-28-2026

"""
Headless workflow engine behind the rapidtools asset-analysis GUI.

This module reproduces the two-part workflow demonstrated in
``examples/asset_analysis_example_generalized.ipynb``:

    1. **Detection** - discover assets in aerial imagery with Meta's SAM 3,
       crop imagery around each detection, and (for buildings and roads)
       regularize the raw masks into clean GIS-ready polygons.
    2. **Inference** - crop the post-disaster raster around every asset and
       run a vision-language model (cloud APIs such as Gemini, Claude,
       OpenAI, Meta Muse Spark and Qwen, or local checkpoints such as
       Gemma-4, Llama, Muse Glimmer and Qwen) with a user-supplied prompt.

The engine has no dependency on any GUI toolkit so it can be unit tested and
scripted independently of the Tk front end in ``asset_analysis_app``.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from rapidtools.core import OperationCancelled, PhysicalAssetCollection

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

    model_class = get_model_class(backend)
    try:
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
    threshold: float = 0.5
    mask_threshold: float = 0.4
    overlap_ratio: float = 0.25
    batch_size: int = 4
    crop_buffer: str = '20 m'
    regularize: bool = True
    bing_max_workers: int = 10

    def __post_init__(self) -> None:
        """Coerce paths and validate the asset list and size."""
        self.raster_path = Path(self.raster_path)
        self.output_dir = Path(self.output_dir)
        if not self.assets:
            raise ValueError('At least one asset name is required.')
        if self.asset_size_m <= 0:
            raise ValueError('Approximate asset size must be positive.')


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
    load_in_4bit: bool = True
    overlay_asset_outline: bool = True
    outline_shape: str = 'geometry'
    outline_buffer: float | str = 0
    outline_width: int | float | str = 6
    outline_color: str = 'red'
    image_prefix: str = 'aerial'
    only_analyze_inference_crops: bool = True

    def __post_init__(self) -> None:
        """
        Coerce paths, resolve the default model and validate the backend.

        Raises:
            ValueError: If the prompt is empty, the backend is unknown, an
                API backend was selected without a key, or the outline
                options are invalid.
        """
        self.raster_path = Path(self.raster_path)
        self.output_dir = Path(self.output_dir)
        if not self.prompt.strip():
            raise ValueError('An inference prompt is required.')
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
class DetectionResult:
    """Outputs of :meth:`AssetAnalysisWorkflow.detect`."""

    collection: PhysicalAssetCollection
    detection_raster: Path
    per_asset_geojson: dict[str, Path] = field(default_factory=dict)
    combined_geojson: Path | None = None


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
            SAM3OrthoFeatureExtractor,
        )

        # 1. Decide which raster to scan.
        if settings.detect_in_recon_imagery:
            detection_raster = raster_path
        else:
            detection_raster = output_dir / f'{raster_path.stem}_bing.tiff'
            if detection_raster.is_file():
                self._progress(
                    f'Reusing existing Bing basemap: {detection_raster.name}'
                )
            else:
                self._progress('Stitching Bing basemap over the raster extent...')
                region = BoundingBox.from_raster(raster_path)
                extractor = BingOrthomosaicExtractor(
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
        from rapidtools import AerialImageryExtractor, Pipeline

        self.reset()
        if len(collection) == 0:
            raise ValueError('The asset collection is empty; nothing to analyze.')
        raster_path = settings.raster_path.expanduser().resolve()
        if not raster_path.is_file():
            raise FileNotFoundError(f'Raster not found: {raster_path}')
        output_dir = settings.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = (output_dir / 'inference_imagery').resolve()

        extractor = AerialImageryExtractor(
            dataset=raster_path,
            save_directory=image_dir,
            overlay_asset_outline=settings.overlay_asset_outline,
            outline_shape=settings.outline_shape,
            outline_buffer=settings.outline_buffer,
            outline_width=settings.outline_width,
            outline_color=settings.outline_color,
            image_prefix=settings.image_prefix,
            keep_multiple_copies=True,
            cancel_event=self._cancel_event,
        )

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

    @staticmethod
    def _build_analyzer(
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
        from rapidtools.models import load
        from rapidtools.processing import AssetAnalyzer

        info = PROVIDERS[settings.backend]
        model_id = settings.model_id or settings.backend_spec['default_model']
        model_kwargs: dict[str, Any] = {
            'model_id': model_id,
            'temperature': settings.temperature,
            'max_tokens': settings.max_tokens,
        }
        if info.is_api:
            model_kwargs['api_key'] = settings.api_key.strip()
            model_kwargs['max_workers'] = settings.max_workers
        elif info.supports_4bit:
            model_kwargs['load_in_4bit'] = settings.load_in_4bit
        model = load(settings.backend, **model_kwargs)

        return AssetAnalyzer(
            model,
            prompt=settings.prompt,
            max_workers=settings.max_workers if info.is_api else 1,
            batch_size=settings.batch_size if info.supports_batch else 1,
            image_filter=image_filter,
            cancel_event=cancel_event,
        )
