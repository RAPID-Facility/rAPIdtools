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
Local web server behind the rapidtools asset-analysis GUI.

The server is intentionally built on the Python standard library only
(``http.server``) so that the GUI adds no dependencies to ``rapidtools``. It
serves a single-page application from ``rapidtools/gui/static`` and exposes a
small JSON API that drives :class:`~rapidtools.gui.workflow.AssetAnalysisWorkflow`
on a background thread.

Launch with ``rapidtools-gui`` or ``python -m rapidtools.gui``.

Example:
    >>> from rapidtools.gui.server import create_server
    >>>
    >>> server = create_server(port=0, output_dir='out')  # port 0: any free port
    >>> server.url.startswith('http://localhost:')
    True
    >>> server.server_close()
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import os
import secrets
import socket
import sys
import threading
import time
import webbrowser
from collections.abc import Callable
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from rapidtools.config import configure_logging
from rapidtools.core import OperationCancelled, PhysicalAssetCollection
from rapidtools.gui.logging_utils import CallbackLogHandler, LogBuffer, ProgressStream
from rapidtools.gui.preview import (
    RasterPreview,
    RasterTiler,
    project_collection,
    render_preview,
)
from rapidtools.gui.workflow import (
    DEFAULT_GEMINI_MODEL_ID,
    GEMMA4_MODEL_IDS,
    MODEL_BACKENDS,
    AssetAnalysisWorkflow,
    DetectionSettings,
    InferenceSettings,
    list_available_models,
    parse_asset_list,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / 'static'
SAMPLE_RASTERS = ('eaton_patch1', 'eaton_patch2')
SAMPLE_PROMPT_DATASET = 'aerial_chs_prompts'
MAX_TEXT_FILE_BYTES = 1_000_000

TOKEN_COOKIE = 'rapidtools_token'

LOGIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rAPIdtools · Sign in</title>
<style>
body{margin:0;min-height:100vh;display:grid;place-items:center;background:#f3f5f8;
font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
color:#1c2430}
.card{background:#fff;border:1px solid #e1e6ee;border-radius:14px;padding:28px 30px;
width:min(380px,90vw);box-shadow:0 8px 24px rgba(16,24,40,.08)}
h1{font-size:18px;margin:0 0 6px}p{color:#5d6b7c;margin:0 0 16px}
input{width:100%;box-sizing:border-box;padding:10px;border:1px solid #cfd6e1;
border-radius:8px;font:inherit}
button{margin-top:12px;width:100%;padding:10px;border:0;border-radius:8px;
background:#2f6fed;color:#fff;font-weight:600;cursor:pointer}
.err{color:#d64545;margin-top:10px;display:none}
</style></head><body><form class="card" id="f">
<h1>rAPIdtools · Asset Analysis</h1>
<p>This server is shared. Enter the access token you received from its owner.</p>
<input id="t" type="password" placeholder="Access token" autofocus
 autocomplete="off">
<button type="submit">Continue</button>
<div class="err" id="e">That token was not accepted.</div>
</form><script>
document.getElementById('f').addEventListener('submit',async(ev)=>{ev.preventDefault();
const r=await fetch('/api/login',{method:'POST',
headers:{'Content-Type':'application/json'},
body:JSON.stringify({token:document.getElementById('t').value.trim()})});
if(r.ok){location.replace('/');}
else{document.getElementById('e').style.display='block';}});
</script></body></html>"""

FILE_KINDS: dict[str, tuple[str, ...]] = {
    'raster': ('.tif', '.tiff'),
    'geojson': ('.geojson', '.json'),
    'text': ('.txt', '.md'),
    'dir': (),
}


class ApiError(Exception):
    """
    An error that should be reported to the browser with an HTTP status.

    Handlers raise this for expected failures (bad input, missing files, busy
    server). The request handler converts it into a JSON ``{"error": ...}``
    response with the given status instead of a 500.

    Example:
        >>> from http import HTTPStatus
        >>> from rapidtools.gui.server import ApiError
        >>>
        >>> err = ApiError('Raster not found', HTTPStatus.NOT_FOUND)
        >>> str(err), int(err.status)
        ('Raster not found', 404)
    """

    def __init__(self, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
        """
        Initialize the error.

        Args:
            message (str): Human-readable message shown in the browser.
            status (int): HTTP status code. Defaults to ``400 Bad Request``.
        """
        super().__init__(message)
        self.status = status


class AppState:
    """
    All mutable server state, guarded by a single lock.

    One instance lives for the lifetime of a :class:`GuiServer`. It owns the
    loaded raster previews and tilers, the current asset collection, the list
    of result files, the log buffer, the workflow object and the background
    job thread.

    Example:
        >>> from rapidtools.gui.server import AppState
        >>>
        >>> state = AppState('out')
        >>> state.busy(), state.status
        (False, 'Ready.')
        >>> state.to_json()['job']['status']
        'idle'
    """

    def __init__(self, output_dir: Path) -> None:
        """
        Create an idle state rooted at ``output_dir``.

        Args:
            output_dir (Path): Where downloads, crops and results are written.
                ``~`` is expanded and the path is resolved.
        """
        self.lock = threading.RLock()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.previews: dict[str, RasterPreview] = {}
        self.tilers: dict[str, RasterTiler] = {}
        self.raster_path: Path | None = None
        self.collection: PhysicalAssetCollection | None = None
        self.collection_source = ''
        self.collection_version = 0
        self.results: list[dict[str, str]] = []
        self.status = 'Ready.'
        self.job: dict[str, Any] = {'name': None, 'status': 'idle', 'message': ''}
        self.log = LogBuffer()
        self.workflow = AssetAnalysisWorkflow(progress_callback=self.set_status)
        self._worker: threading.Thread | None = None
        self.data_root: Path | None = None
        self.token_required = False

    # ------------------------------------------------------------ helpers
    def set_status(self, message: str) -> None:
        """
        Replace the one-line status shown at the top of the GUI.

        Args:
            message (str): The new status text.
        """
        with self.lock:
            self.status = message

    def set_collection(
        self, collection: PhysicalAssetCollection | None, source: str
    ) -> None:
        """
        Replace the active asset collection and bump its version counter.

        Args:
            collection (PhysicalAssetCollection | None): The new collection, or
                ``None`` to clear it.
            source (str): Short provenance label (e.g. ``'detected'``).
        """
        with self.lock:
            self.collection = collection
            self.collection_source = source
            self.collection_version += 1

    def add_result(self, label: str, path: Path | None) -> None:
        """
        Register a downloadable result file, replacing an entry with the same path.

        Args:
            label (str): Label shown in the results list.
            path (Path | None): File path; ``None`` is ignored.
        """
        if path is None:
            return
        with self.lock:
            entry = {'label': label, 'path': str(path), 'name': Path(path).name}
            self.results = [r for r in self.results if r['path'] != entry['path']]
            self.results.append(entry)

    def set_preview(self, key: str, preview: RasterPreview) -> None:
        """
        Store a rendered preview (and a matching tiler) under ``key``.

        The preview's ``version`` is set to one more than the previous preview
        stored under the same key so the browser can invalidate its cache.

        Args:
            key (str): ``'raster'`` for recon imagery or ``'basemap'`` for Bing.
            preview (RasterPreview): The rendered preview.
        """
        with self.lock:
            previous = self.previews.get(key)
            preview.version = (previous.version + 1) if previous else 1
            self.previews[key] = preview
            self.tilers[key] = RasterTiler(preview.path)

    def reset(self) -> None:
        """Forget every loaded raster, collection, result and log line."""
        with self.lock:
            self.previews.clear()
            self.tilers.clear()
            self.raster_path = None
            self.collection = None
            self.collection_source = ''
            self.collection_version += 1
            self.results.clear()
            self.status = 'Ready.'
            self.log.clear()

    # --------------------------------------------------------------- jobs
    def busy(self) -> bool:
        """
        Report whether a background job is currently running.

        Returns:
            bool: ``True`` while the worker thread is alive.
        """
        return self._worker is not None and self._worker.is_alive()

    def start_job(self, name: str, target: Callable[[], Any]) -> None:
        """
        Run ``target`` on a background thread, streaming logs and tqdm.

        While the job runs, ``sys.stderr`` is swapped for a
        :class:`~rapidtools.gui.logging_utils.ProgressStream` so ``tqdm`` bars
        appear live in the GUI log. The job record is updated to ``done``,
        ``cancelled`` (on :class:`~rapidtools.core.OperationCancelled`) or
        ``error`` when the target returns or raises.

        Args:
            name (str): Job name shown in the GUI (e.g. ``'Detection'``).
            target (Callable[[], Any]): The work to perform.

        Raises:
            ApiError: With status ``409 Conflict`` if a job is already running.

        Example:
            >>> from rapidtools.gui.server import AppState
            >>>
            >>> state = AppState('out')
            >>> state.start_job('Demo', lambda: None)
            >>> state._worker.join()
            >>> state.job['status'], state.status
            ('done', 'Demo finished.')
        """
        with self.lock:
            if self.busy():
                raise ApiError(
                    f'A job is already running ({self.job["name"]}).',
                    HTTPStatus.CONFLICT,
                )
            self.job = {
                'name': name,
                'status': 'running',
                'message': '',
                'started_at': time.time(),
            }
            self.status = f'{name} started...'
            self.log.add_line(f'--- {name} ---')

        def runner() -> None:
            """Thread body: redirect stderr, run the target, record the outcome."""
            original_stderr = sys.stderr
            sys.stderr = ProgressStream(
                on_line=self.log.add_line,
                on_progress=self.log.set_progress,
                on_progress_end=self.log.end_progress,
                fallback=original_stderr,
            )
            try:
                target()
            except OperationCancelled:
                self._finish_job('cancelled', f'{name} cancelled.')
            except Exception as exc:  # noqa: BLE001 - surfaced to the browser
                logger.exception(f'{name} failed')
                self._finish_job('error', f'{name} failed: {exc}')
            else:
                self._finish_job('done', f'{name} finished.')
            finally:
                sys.stderr = original_stderr

        self._worker = threading.Thread(target=runner, name=name, daemon=True)
        self._worker.start()

    def _finish_job(self, status: str, message: str) -> None:
        """Record the job outcome and append ``message`` to the log."""
        with self.lock:
            self.log.end_progress()
            self.job.update(status=status, message=message, finished_at=time.time())
            self.status = message
            self.log.add_line(message)

    def cancel_job(self) -> None:
        """Ask the running workflow to stop at the next safe point (if any)."""
        if self.busy():
            self.workflow.cancel()
            self.set_status(
                'Cancelling... the current batch or tile will finish first.'
            )
            self.log.add_line('Cancellation requested by user.')

    # -------------------------------------------------------- serialization
    def to_json(self) -> dict[str, Any]:
        """
        Serialize the full GUI state for ``GET /api/state``.

        Returns:
            dict[str, Any]:
                ``output_dir``, ``raster`` and ``basemap`` preview metadata (or
                ``None``), a ``collection`` summary, the ``job`` record, ``busy``,
                ``status``, ``results``, ``log_seq`` and the static ``options``
                the front end needs (sample datasets, model IDs, backends, data
                root, shared flag).
        """
        with self.lock:
            collection = None
            if self.collection is not None:
                keys: set[str] = set()
                for asset in self.collection:
                    keys.update(str(k) for k in asset.attributes)
                collection = {
                    'count': len(self.collection),
                    'source': self.collection_source,
                    'version': self.collection_version,
                    'attribute_keys': sorted(keys),
                }
            job = dict(self.job)
            return {
                'output_dir': str(self.output_dir),
                'raster': self.previews['raster'].to_json()
                if 'raster' in self.previews
                else None,
                'basemap': self.previews['basemap'].to_json()
                if 'basemap' in self.previews
                else None,
                'collection': collection,
                'job': job,
                'busy': self.busy(),
                'status': self.status,
                'results': list(self.results),
                'log_seq': self.log.seq,
                'options': {
                    'sample_rasters': list(SAMPLE_RASTERS),
                    'gemini_model': DEFAULT_GEMINI_MODEL_ID,
                    'gemma_models': list(GEMMA4_MODEL_IDS),
                    'backends': MODEL_BACKENDS,
                    'home': str(Path.home()),
                    'cwd': str(Path.cwd()),
                    'data_root': str(self.data_root) if self.data_root else None,
                    'shared': self.token_required,
                },
            }


# ====================================================================== API
class Api:
    """
    Implements each JSON endpoint against an :class:`AppState`.

    Every public method takes the decoded JSON body (``POST``) or the query
    dictionary (``GET``) and returns a JSON-serialisable dict, raising
    :class:`ApiError` for expected failures. Long-running work is handed to
    :meth:`AppState.start_job` so the HTTP response returns immediately.

    Example:
        >>> from rapidtools.gui.server import Api, AppState
        >>>
        >>> api = Api(AppState('out'))
        >>> api.table({})
        {'rows': [], 'columns': []}
    """

    def __init__(self, state: AppState, data_root: Path | None = None) -> None:
        """
        Bind the API to a state object.

        Args:
            state (AppState): The shared server state.
            data_root (Path | None): If given, every user-supplied path must lie
                inside this directory. Defaults to ``None`` (unrestricted).
        """
        self.state = state
        self.data_root = Path(data_root).expanduser().resolve() if data_root else None

    def _user_path(self, raw: str | Path) -> Path:
        """
        Resolve a user-supplied path, enforcing ``data_root`` when set.

        Args:
            raw (str | Path): Path from the request (``~`` is expanded).

        Returns:
            Path: The path (resolved when ``data_root`` is enforced).

        Raises:
            ApiError: ``403 Forbidden`` if the path escapes ``data_root``.
        """
        path = Path(raw).expanduser()
        if self.data_root is None:
            return path
        resolved = path.resolve()
        if resolved != self.data_root and self.data_root not in resolved.parents:
            raise ApiError(
                f'Access is limited to {self.data_root}.', HTTPStatus.FORBIDDEN
            )
        return resolved

    # ----------------------------------------------------------- imagery
    def load_local_raster(self, body: dict[str, Any]) -> dict[str, Any]:
        """
        ``POST /api/imagery/local``: render a preview of a local raster.

        Args:
            body (dict[str, Any]): Must contain ``path``.

        Returns:
            dict[str, Any]: ``{'ok': True}`` once the preview job is queued.

        Raises:
            ApiError: If ``path`` is missing, outside ``data_root`` or not a file.
        """
        path = self._user_path(_require_path(body, 'path'))
        if not path.is_file():
            raise ApiError(f'Raster not found: {path}', HTTPStatus.NOT_FOUND)
        self._start_preview_job(path, label='Load imagery')
        return {'ok': True}

    def download_sample_raster(self, body: dict[str, Any]) -> dict[str, Any]:
        """
        ``POST /api/imagery/sample``: download a sample dataset and preview it.

        Args:
            body (dict[str, Any]): Must contain ``name``, one of
                :data:`SAMPLE_RASTERS`.

        Returns:
            dict[str, Any]: ``{'ok': True}`` once the download job is queued.

        Raises:
            ApiError: If ``name`` is not a known sample dataset.
        """
        name = str(body.get('name', '')).strip()
        if name not in SAMPLE_RASTERS:
            raise ApiError(f'Unknown sample dataset: {name!r}')
        state = self.state

        def job() -> None:
            """Download the sample into ``output_dir`` and render its preview."""
            from rapidtools import download_dataset

            state.output_dir.mkdir(parents=True, exist_ok=True)
            [path] = download_dataset(name, output_dir=state.output_dir)
            self._load_raster(Path(path))

        state.start_job(f'Download {name}', job)
        return {'ok': True}

    def _start_preview_job(self, path: Path, label: str) -> None:
        """Queue a background job that renders and installs a raster preview."""
        self.state.start_job(label, lambda: self._load_raster(path))

    def _load_raster(self, path: Path) -> None:
        """Render ``path`` and make it the active raster (clears old results)."""
        state = self.state
        state.set_status(f'Rendering preview of {path.name}...')
        preview = render_preview(path)
        with state.lock:
            state.raster_path = preview.path
            state.previews.pop('basemap', None)
            state.tilers.pop('basemap', None)
            state.set_preview('raster', preview)
            state.collection = None
            state.collection_source = ''
            state.collection_version += 1
            state.results.clear()
        state.set_status(f'Loaded {path.name} ({preview.width}x{preview.height} px).')

    # ------------------------------------------------------------ assets
    def load_assets(self, body: dict[str, Any]) -> dict[str, Any]:
        """
        ``POST /api/assets/load``: load an existing GeoJSON asset file.

        Args:
            body (dict[str, Any]): Must contain ``path`` to a GeoJSON file.

        Returns:
            dict[str, Any]: ``{'ok': True, 'count': <number of assets>}``.

        Raises:
            ApiError: If the file is missing, unreadable as GeoJSON, or empty.
        """
        path = self._user_path(_require_path(body, 'path'))
        if not path.is_file():
            raise ApiError(f'File not found: {path}', HTTPStatus.NOT_FOUND)
        try:
            collection = PhysicalAssetCollection.from_geojson(path)
        except Exception as exc:  # noqa: BLE001
            raise ApiError(f'Could not read GeoJSON: {exc}') from exc
        if len(collection) == 0:
            raise ApiError('The GeoJSON file contains no features.')
        self.state.set_collection(collection, f'loaded from {path.name}')
        self.state.add_result('Loaded assets', path)
        self.state.set_status(f'Loaded {len(collection)} assets from {path.name}.')
        return {'ok': True, 'count': len(collection)}

    def detect(self, body: dict[str, Any]) -> dict[str, Any]:
        """
        ``POST /api/detect``: start SAM 3 asset detection in the background.

        Args:
            body (dict[str, Any]): Detection options, see
                :meth:`_detection_settings`.

        Returns:
            dict[str, Any]: ``{'ok': True}`` once the job is queued.

        Raises:
            ApiError: If no raster is loaded, the options are invalid, or a job
                is already running.
        """
        settings = self._detection_settings(body)
        self.state.start_job('Detection', lambda: self._run_detection(settings))
        return {'ok': True}

    def infer(self, body: dict[str, Any]) -> dict[str, Any]:
        """
        ``POST /api/infer``: run VLM inference on the current collection.

        Args:
            body (dict[str, Any]): Inference options, see
                :meth:`_inference_settings`.

        Returns:
            dict[str, Any]: ``{'ok': True}`` once the job is queued.

        Raises:
            ApiError: If no assets are loaded, the options are invalid, or a job
                is already running.
        """
        with self.state.lock:
            collection = self.state.collection
        if collection is None or len(collection) == 0:
            raise ApiError('Detect assets or load an asset file first.')
        settings = self._inference_settings(body)
        self.state.start_job(
            'Inference', lambda: self._run_inference(collection, settings)
        )
        return {'ok': True}

    def run_all(self, body: dict[str, Any]) -> dict[str, Any]:
        """
        ``POST /api/run_all``: detection followed by inference as one job.

        Args:
            body (dict[str, Any]): ``{'detect': {...}, 'infer': {...}}`` with the
                options accepted by :meth:`detect` and :meth:`infer`.

        Returns:
            dict[str, Any]: ``{'ok': True}`` once the job is queued.

        Raises:
            ApiError: If either option set is invalid or a job is running.
        """
        detection = self._detection_settings(body.get('detect') or {})
        inference = self._inference_settings(body.get('infer') or {})

        def job() -> None:
            """Detect, then analyze; fail loudly if nothing was detected."""
            collection = self._run_detection(detection)
            if len(collection) == 0:
                raise ValueError('No assets were detected; skipping inference.')
            self._run_inference(collection, inference)

        self.state.start_job('Detection + inference', job)
        return {'ok': True}

    def _run_detection(self, settings: DetectionSettings) -> PhysicalAssetCollection:
        """
        Run detection, publish its results and (if used) the Bing basemap.

        Args:
            settings (DetectionSettings): Validated detection options.

        Returns:
            PhysicalAssetCollection: The detected assets.
        """
        state = self.state
        result = state.workflow.detect(settings)
        with state.lock:
            # A fresh detection supersedes any previously loaded asset file.
            state.results = [r for r in state.results if r['label'] != 'Loaded assets']
        state.set_collection(result.collection, 'detected')
        for name, path in result.per_asset_geojson.items():
            state.add_result(f'{name} footprints', path)
        state.add_result('All detected assets', result.combined_geojson)
        if result.detection_raster != settings.raster_path.resolve():
            state.set_status('Rendering preview of the Bing basemap...')
            state.set_preview('basemap', render_preview(result.detection_raster))
            state.add_result('Bing basemap', result.detection_raster)
        return result.collection

    def _run_inference(
        self, collection: PhysicalAssetCollection, settings: InferenceSettings
    ) -> None:
        """
        Run inference and publish the results, keeping partial output on cancel.

        Args:
            collection (PhysicalAssetCollection): Assets to analyze.
            settings (InferenceSettings): Validated inference options.

        Raises:
            OperationCancelled: Re-raised after recording partial results.
        """
        state = self.state
        try:
            result = state.workflow.analyze(collection, settings)
        except OperationCancelled:
            # Attributes were written in place; keep what finished.
            state.set_collection(collection, 'partially analyzed')
            partial = settings.output_dir / 'assets_inferred_partial.geojson'
            if partial.is_file():
                state.add_result('Partial inference results', partial)
            raise
        state.set_collection(result.collection, 'analyzed')
        state.add_result('Inference results', result.geojson_path)

    def _raster_path(self) -> Path:
        """
        Return the active raster path.

        Raises:
            ApiError: If no raster has been loaded yet.
        """
        with self.state.lock:
            raster = self.state.raster_path
        if raster is None:
            raise ApiError('Load an aerial image first (step 1).')
        return raster

    def _detection_settings(self, body: dict[str, Any]) -> DetectionSettings:
        """
        Validate a detection request body.

        Args:
            body (dict[str, Any]): ``assets`` (comma/newline separated string,
                required), ``asset_size_m``, ``source`` (``'bing'`` or
                ``'recon'``), ``threshold``, ``mask_threshold`` and
                ``regularize``.

        Returns:
            DetectionSettings: The validated settings.

        Raises:
            ApiError: If no assets are given, no raster is loaded, or a value
                cannot be coerced.
        """
        assets = parse_asset_list(str(body.get('assets', '')))
        if not assets:
            raise ApiError('Enter at least one asset to detect (e.g. "building").')
        try:
            return DetectionSettings(
                raster_path=self._raster_path(),
                output_dir=self.state.output_dir,
                assets=assets,
                asset_size_m=float(body.get('asset_size_m', 50.0)),
                detect_in_recon_imagery=body.get('source', 'bing') == 'recon',
                threshold=float(body.get('threshold', 0.5)),
                mask_threshold=float(body.get('mask_threshold', 0.4)),
                regularize=bool(body.get('regularize', True)),
            )
        except (TypeError, ValueError) as exc:
            raise ApiError(str(exc)) from exc

    def _inference_settings(self, body: dict[str, Any]) -> InferenceSettings:
        """
        Validate an inference request body.

        Args:
            body (dict[str, Any]): ``backend`` (a :data:`MODEL_BACKENDS` key),
                ``prompt``, ``api_key``, ``model_id``, ``max_workers``,
                ``batch_size``, ``load_in_4bit``, ``overlay_outline`` and the
                optional ``outline_shape``, ``outline_buffer``,
                ``outline_width`` and ``outline_color`` (blank values fall
                back to the extractor defaults).

        Returns:
            InferenceSettings: The validated settings.

        Raises:
            ApiError: If the backend is unknown, no raster is loaded, or the
                settings fail validation (e.g. missing API key or prompt).
        """
        backend = str(body.get('backend', 'gemini'))
        spec = MODEL_BACKENDS.get(backend)
        if spec is None:
            raise ApiError(f'Unknown inference backend: {backend!r}')
        default_model = spec['default_model']
        try:
            return InferenceSettings(
                raster_path=self._raster_path(),
                output_dir=self.state.output_dir,
                prompt=str(body.get('prompt', '')),
                backend=backend,  # type: ignore[arg-type]
                api_key=str(body.get('api_key', '')),
                model_id=str(body.get('model_id') or default_model).strip(),
                max_workers=int(body.get('max_workers', 5)),
                batch_size=int(body.get('batch_size', 4)),
                load_in_4bit=bool(body.get('load_in_4bit', True)),
                overlay_asset_outline=bool(body.get('overlay_outline', True)),
                outline_shape=str(body.get('outline_shape') or 'geometry'),
                outline_buffer=body.get('outline_buffer') or 0,
                outline_width=body.get('outline_width') or 6,
                outline_color=str(body.get('outline_color') or 'red'),
            )
        except (TypeError, ValueError) as exc:
            raise ApiError(str(exc)) from exc

    # ---------------------------------------------------------- utilities
    def set_output_dir(self, body: dict[str, Any]) -> dict[str, Any]:
        """
        ``POST /api/output_dir``: change where results are written.

        Args:
            body (dict[str, Any]): Must contain ``path``; it may not exist yet.

        Returns:
            dict[str, Any]: ``{'ok': True, 'output_dir': <path>}``.

        Raises:
            ApiError: If ``path`` exists but is not a directory, or is outside
                ``data_root``.
        """
        path = self._user_path(_require_path(body, 'path'))
        if path.exists() and not path.is_dir():
            raise ApiError(f'Not a directory: {path}')
        with self.state.lock:
            self.state.output_dir = path
        return {'ok': True, 'output_dir': str(path)}

    def sample_prompt(self, _body: dict[str, Any]) -> dict[str, Any]:
        """
        ``POST /api/prompt/sample``: fetch the sample CHS prompt from the registry.

        Args:
            _body (dict[str, Any]): Ignored.

        Returns:
            dict[str, Any]: ``{'text': <prompt text>}``.
        """
        from rapidtools import download_dataset

        self.state.output_dir.mkdir(parents=True, exist_ok=True)
        [path] = download_dataset(
            SAMPLE_PROMPT_DATASET, output_dir=self.state.output_dir
        )
        return {'text': Path(path).read_text(encoding='utf-8').strip()}

    def read_text_file(self, query: dict[str, str]) -> dict[str, Any]:
        """
        ``GET /api/textfile``: return the contents of a small text file.

        Args:
            query (dict[str, str]): ``path`` of the file.

        Returns:
            dict[str, Any]: ``{'text': <stripped file contents>}``.

        Raises:
            ApiError: ``404`` if the file is missing, ``400`` if it exceeds
                :data:`MAX_TEXT_FILE_BYTES`, ``403`` if outside ``data_root``.
        """
        path = self._user_path(query.get('path', '') or '.')
        if not path.is_file():
            raise ApiError(f'File not found: {path}', HTTPStatus.NOT_FOUND)
        if path.stat().st_size > MAX_TEXT_FILE_BYTES:
            raise ApiError('File is too large to load as text.')
        return {'text': path.read_text(encoding='utf-8', errors='replace').strip()}

    def list_directory(self, query: dict[str, str]) -> dict[str, Any]:
        """
        ``GET /api/fs``: list a directory for the in-browser file picker.

        Hidden entries are skipped. When ``data_root`` is set, requests outside
        it silently fall back to the root and ``parent`` is ``None`` at the root.

        Args:
            query (dict[str, str]): ``kind`` (a :data:`FILE_KINDS` key selecting
                which file extensions to show; ``'dir'`` lists directories only)
                and optional ``path`` (defaults to ``data_root`` or the home
                directory). A file path lists its parent directory.

        Returns:
            dict[str, Any]: ``path``, ``parent`` (or ``None``), ``dirs`` and
            ``files`` (each file with ``name``, ``path`` and ``size``).

        Raises:
            ApiError: ``400`` for an unknown kind, ``404`` for a missing
                directory, ``403`` when listing is not permitted.
        """
        kind = query.get('kind', 'raster')
        extensions = FILE_KINDS.get(kind)
        if extensions is None:
            raise ApiError(f'Unknown file kind: {kind!r}')
        default_dir = self.data_root or Path.home()
        raw = query.get('path') or str(default_dir)
        try:
            path = self._user_path(raw)
        except ApiError:
            path = default_dir  # outside the allowed root: fall back to it
        if path.is_file():
            path = path.parent
        if not path.is_dir():
            raise ApiError(f'Directory not found: {path}', HTTPStatus.NOT_FOUND)
        path = path.resolve()
        dirs: list[dict[str, Any]] = []
        files: list[dict[str, Any]] = []
        try:
            entries = sorted(path.iterdir(), key=lambda p: p.name.lower())
        except PermissionError as exc:
            raise ApiError(f'Permission denied: {path}', HTTPStatus.FORBIDDEN) from exc
        for entry in entries:
            if entry.name.startswith('.'):
                continue
            try:
                if entry.is_dir():
                    dirs.append({'name': entry.name, 'path': str(entry)})
                elif kind != 'dir' and entry.suffix.lower() in extensions:
                    files.append(
                        {
                            'name': entry.name,
                            'path': str(entry),
                            'size': entry.stat().st_size,
                        }
                    )
            except OSError:
                continue
        at_root = path.parent == path or (
            self.data_root is not None and path == self.data_root
        )
        return {
            'path': str(path),
            'parent': None if at_root else str(path.parent),
            'dirs': dirs,
            'files': files,
        }

    def overlay(self, query: dict[str, str]) -> dict[str, Any]:
        """
        ``GET /api/overlay``: project the collection onto a preview.

        Args:
            query (dict[str, str]): ``preview`` key (``'raster'`` by default or
                ``'basemap'``).

        Returns:
            dict[str, Any]: The :func:`~rapidtools.gui.preview.project_collection`
            payload plus ``collection_version``; empty ``features`` when either
            the preview or the collection is missing.
        """
        key = query.get('preview', 'raster')
        with self.state.lock:
            preview = self.state.previews.get(key)
            collection = self.state.collection
            version = self.state.collection_version
        if preview is None or collection is None:
            return {
                'features': [],
                'count': 0,
                'attribute_keys': [],
                'collection_version': version,
            }
        # A running analyzer mutates asset attributes in place; retry if a
        # dictionary changes size while we are serialising it.
        for attempt in range(3):
            try:
                data = project_collection(collection, preview)
                break
            except RuntimeError:
                if attempt == 2:
                    raise
                time.sleep(0.05)
        data['collection_version'] = version
        return data

    def table(self, query: dict[str, str]) -> dict[str, Any]:
        """
        ``GET /api/table``: tabulate asset attributes for the results grid.

        Args:
            query (dict[str, str]): Optional ``limit`` (default ``500`` rows).

        Returns:
            dict[str, Any]: ``rows`` (``id``, ``images`` count and JSON-safe
            ``attributes``), ``columns`` in first-seen order and ``total``;
            ``{'rows': [], 'columns': []}`` when no collection is loaded.
        """
        limit = int(query.get('limit', 500))
        with self.state.lock:
            collection = self.state.collection
        if collection is None:
            return {'rows': [], 'columns': []}
        rows = []
        columns: list[str] = []
        for asset in list(collection)[:limit]:
            attrs = {str(k): v for k, v in asset.attributes.items()}
            for key in attrs:
                if key not in columns:
                    columns.append(key)
            rows.append(
                {
                    'id': asset.id,
                    'images': len(asset.image_assets),
                    'attributes': {
                        k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                        for k, v in attrs.items()
                    },
                }
            )
        return {'rows': rows, 'columns': columns, 'total': len(collection)}

    def list_models(self, body: dict[str, Any]) -> dict[str, Any]:
        """
        ``POST /api/models``: list the model IDs available for a backend.

        Args:
            body (dict[str, Any]): ``backend`` and optional ``api_key``.

        Returns:
            dict[str, Any]: ``{'backend': <name>, 'models': [<ids>]}``.

        Raises:
            ApiError: If the backend is unknown.
        """
        backend = str(body.get('backend', 'gemini'))
        if backend not in MODEL_BACKENDS:
            raise ApiError(f'Unknown inference backend: {backend!r}')
        models = list_available_models(backend, str(body.get('api_key', '')))
        return {'backend': backend, 'models': models}

    def tile(self, key: str, downsample: int, tx: int, ty: int) -> bytes:
        """
        ``GET /api/tile/<key>/<d>/<tx>/<ty>.jpg``: a full-resolution JPEG tile.

        Args:
            key (str): Preview key (``'raster'`` or ``'basemap'``).
            downsample (int): Downsample factor.
            tx (int): Tile column.
            ty (int): Tile row.

        Returns:
            bytes: JPEG data.

        Raises:
            ApiError: ``404`` if no raster is loaded or the tile is off-raster.
        """
        with self.state.lock:
            tiler = self.state.tilers.get(key)
        if tiler is None:
            raise ApiError('No raster loaded.', HTTPStatus.NOT_FOUND)
        data = tiler.tile(downsample, tx, ty)
        if data is None:
            raise ApiError('Tile is outside the raster.', HTTPStatus.NOT_FOUND)
        return data

    def _find_asset(self, asset_id: str):
        """
        Look up an asset by ID in the current collection.

        Raises:
            ApiError: ``404`` if no collection is loaded or the ID is unknown.
        """
        with self.state.lock:
            collection = self.state.collection
        if collection is None:
            raise ApiError('No assets loaded.', HTTPStatus.NOT_FOUND)
        asset = collection.get(asset_id)
        if asset is None:
            raise ApiError(f'Unknown asset: {asset_id}', HTTPStatus.NOT_FOUND)
        return asset

    def asset_details(self, query: dict[str, str]) -> dict[str, Any]:
        """
        ``GET /api/asset``: attributes and linked images for one asset.

        Args:
            query (dict[str, str]): ``id`` of the asset.

        Returns:
            dict[str, Any]: ``id``, JSON-safe ``attributes`` and ``images`` (each
            with ``id``, ``name``, ``exists``, a ``url`` to fetch it and its
            ``properties``).

        Raises:
            ApiError: ``404`` if the asset is unknown.
        """
        asset = self._find_asset(query.get('id', ''))
        images = []
        for image in asset.image_assets:
            path = Path(image.path)
            images.append(
                {
                    'id': image.id,
                    'name': path.name,
                    'exists': path.is_file(),
                    'url': f'/api/asset_image?asset={asset.id}&image={image.id}',
                    'properties': {
                        str(k): _json_primitive(v)
                        for k, v in (image.properties or {}).items()
                    },
                }
            )
        return {
            'id': asset.id,
            'attributes': {
                str(k): _json_primitive(v) for k, v in asset.attributes.items()
            },
            'images': images,
        }

    def asset_image_path(self, query: dict[str, str]) -> Path:
        """
        ``GET /api/asset_image``: resolve the file behind an asset's image.

        Args:
            query (dict[str, str]): ``asset`` and ``image`` IDs.

        Returns:
            Path: Path to the image file.

        Raises:
            ApiError: ``404`` if the asset or image is unknown or the file is
                missing on disk.
        """
        asset = self._find_asset(query.get('asset', ''))
        image_id = query.get('image', '')
        for image in asset.image_assets:
            if image.id == image_id:
                path = Path(image.path)
                if not path.is_file():
                    raise ApiError('Image file is missing.', HTTPStatus.NOT_FOUND)
                return path
        raise ApiError('Unknown image.', HTTPStatus.NOT_FOUND)

    def download_path(self, query: dict[str, str]) -> Path:
        """
        ``GET /api/download``: resolve a result file for download.

        Only files previously registered through :meth:`AppState.add_result`
        may be downloaded, so arbitrary paths cannot be read through the API.

        Args:
            query (dict[str, str]): ``path`` of the result file.

        Returns:
            Path: The resolved file path.

        Raises:
            ApiError: ``403`` if the path is not a registered, existing result.
        """
        path = Path(query.get('path', '')).expanduser().resolve()
        allowed = {r['path'] for r in self.state.results}
        if str(path) not in allowed or not path.is_file():
            raise ApiError('File is not available for download.', HTTPStatus.FORBIDDEN)
        return path


def _json_primitive(value: Any) -> Any:
    """
    Coerce a value to something JSON can carry.

    Args:
        value (Any): Any attribute value.

    Returns:
        Any: ``value`` if it is an int, float, str or bool, else ``str(value)``.
    """
    return value if isinstance(value, (int, float, str, bool)) else str(value)


def _require_path(body: dict[str, Any], key: str) -> Path:
    """
    Extract a non-empty path from a request body.

    Args:
        body (dict[str, Any]): Decoded JSON body.
        key (str): The key holding the path.

    Returns:
        Path: The path with ``~`` expanded.

    Raises:
        ApiError: If the key is missing or blank.
    """
    raw = str(body.get(key, '')).strip()
    if not raw:
        raise ApiError(f'Missing "{key}".')
    return Path(raw).expanduser()


# ================================================================== HTTP
class GuiRequestHandler(BaseHTTPRequestHandler):
    """
    Routes HTTP requests to :class:`Api` methods and static files.

    ``GET`` serves the single-page app, static assets, previews, tiles and
    the read-only JSON endpoints; ``POST`` drives the actions. When the server
    was created with a token, every request must present it via the
    ``X-Rapidtools-Token`` header, the :data:`TOKEN_COOKIE` cookie, or a
    one-time ``?token=`` share link that is exchanged for the cookie.

    Example:
        >>> import threading
        >>> from http.client import HTTPConnection
        >>> from rapidtools.gui.server import create_server
        >>>
        >>> server = create_server(port=0, output_dir='out')
        >>> threading.Thread(target=server.serve_forever, daemon=True).start()
        >>> conn = HTTPConnection('127.0.0.1', server.server_address[1])
        >>> conn.request('GET', '/api/state')
        >>> conn.getresponse().status
        200
        >>> server.shutdown(); server.server_close()
    """

    server: GuiServer  # type: ignore[assignment]
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt: str, *args: Any) -> None:  # silence access log
        """Emit the access log at DEBUG level only when the server is verbose."""
        if self.server.verbose:
            logger.debug('%s - %s', self.address_string(), fmt % args)

    # --------------------------------------------------------------- auth
    def _presented_token(self) -> str | None:
        """
        Return the token sent with this request, if any.

        Returns:
            str | None: The ``X-Rapidtools-Token`` header, else the
            :data:`TOKEN_COOKIE` cookie value, else ``None``.
        """
        header = self.headers.get('X-Rapidtools-Token')
        if header:
            return header.strip()
        raw_cookie = self.headers.get('Cookie')
        if raw_cookie:
            cookie = SimpleCookie()
            try:
                cookie.load(raw_cookie)
            except Exception:  # noqa: BLE001 - malformed cookie header
                return None
            morsel = cookie.get(TOKEN_COOKIE)
            if morsel is not None:
                return morsel.value
        return None

    def _authorized(self, presented: str | None = None) -> bool:
        """
        Check whether the request may use the app.

        Args:
            presented (str | None): A token to check instead of the one carried
                by the request (used for share links and the login form).

        Returns:
            bool: ``True`` if no token is configured or the presented token
            matches (compared in constant time).
        """
        token = self.server.token
        if token is None:
            return True
        presented = presented if presented is not None else self._presented_token()
        return bool(presented) and secrets.compare_digest(presented, token)

    def _token_cookie(self) -> str:
        """Return the ``Set-Cookie`` value that remembers the access token."""
        return (
            f'{TOKEN_COOKIE}={self.server.token}; Path=/; HttpOnly; SameSite=Lax; '
            'Max-Age=2592000'
        )

    def _send_login_page(self) -> None:
        """Send the sign-in form with a ``401 Unauthorized`` status."""
        self._send_bytes(
            LOGIN_PAGE.encode('utf-8'),
            'text/html; charset=utf-8',
            status=HTTPStatus.UNAUTHORIZED,
        )

    # --------------------------------------------------------------- GET
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        """
        Serve the app, static files, previews, tiles and read-only endpoints.

        Errors raised as :class:`ApiError` become JSON responses with their
        status; anything else is logged and reported as ``500``.
        """
        parts = urlsplit(self.path)
        path = unquote(parts.path)
        query = {k: v[-1] for k, v in parse_qs(parts.query).items()}
        api = self.server.api
        state = self.server.state
        try:
            # A share link carries the token once; exchange it for a cookie.
            if path == '/' and 'token' in query and self.server.token is not None:
                if self._authorized(query['token']):
                    self.send_response(HTTPStatus.FOUND)
                    self.send_header('Location', '/')
                    self.send_header('Set-Cookie', self._token_cookie())
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                self._send_login_page()
                return
            if not self._authorized():
                if path in ('/', '/index.html'):
                    self._send_login_page()
                    return
                raise ApiError('Sign in required.', HTTPStatus.UNAUTHORIZED)
            if path in ('/', '/index.html'):
                self._send_file(STATIC_DIR / 'index.html')
            elif path.startswith('/static/'):
                target = (STATIC_DIR / path[len('/static/') :]).resolve()
                if STATIC_DIR.resolve() not in target.parents or not target.is_file():
                    raise ApiError('Not found', HTTPStatus.NOT_FOUND)
                self._send_file(target)
            elif path == '/api/state':
                self._send_json(state.to_json())
            elif path == '/api/log':
                since = int(query.get('since', 0))
                self._send_json(
                    {'entries': state.log.since(since), 'seq': state.log.seq}
                )
            elif path == '/api/fs':
                self._send_json(api.list_directory(query))
            elif path == '/api/overlay':
                self._send_json(api.overlay(query))
            elif path == '/api/table':
                self._send_json(api.table(query))
            elif path == '/api/textfile':
                self._send_json(api.read_text_file(query))
            elif path.startswith('/api/tile/'):
                parts_ = path[len('/api/tile/') :].removesuffix('.jpg').split('/')
                if len(parts_) != 4:
                    raise ApiError('Not found', HTTPStatus.NOT_FOUND)
                key, d, tx, ty = parts_
                try:
                    data = api.tile(key, int(d), int(tx), int(ty))
                except ValueError as exc:
                    raise ApiError('Bad tile address') from exc
                self._send_bytes(
                    data,
                    'image/jpeg',
                    extra_headers={'Cache-Control': 'max-age=3600'},
                )
            elif path == '/api/asset':
                self._send_json(api.asset_details(query))
            elif path == '/api/asset_image':
                self._send_file(api.asset_image_path(query))
            elif path.startswith('/api/preview/'):
                key = path[len('/api/preview/') :].removesuffix('.png')
                with state.lock:
                    preview = state.previews.get(key)
                if preview is None:
                    raise ApiError('No preview available.', HTTPStatus.NOT_FOUND)
                self._send_bytes(preview.png, 'image/png')
            elif path == '/api/download':
                target = api.download_path(query)
                self._send_file(target, attachment=True)
            else:
                raise ApiError('Not found', HTTPStatus.NOT_FOUND)
        except ApiError as exc:
            self._send_json({'error': str(exc)}, status=exc.status)
        except Exception as exc:  # noqa: BLE001
            logger.exception('Unhandled error in GET %s', path)
            self._send_json(
                {'error': str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR
            )

    # -------------------------------------------------------------- POST
    def do_POST(self) -> None:  # noqa: N802 - http.server API
        """
        Dispatch an action endpoint (``/api/login``, ``/api/detect``, ...).

        The JSON body is decoded with :meth:`_read_json` and passed to the
        matching :class:`Api` method; the returned dict is sent as JSON.
        """
        path = unquote(urlsplit(self.path).path)
        api = self.server.api
        state = self.server.state
        routes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            '/api/imagery/local': api.load_local_raster,
            '/api/imagery/sample': api.download_sample_raster,
            '/api/output_dir': api.set_output_dir,
            '/api/assets/load': api.load_assets,
            '/api/detect': api.detect,
            '/api/infer': api.infer,
            '/api/run_all': api.run_all,
            '/api/prompt/sample': api.sample_prompt,
            '/api/models': api.list_models,
            '/api/cancel': lambda _b: (state.cancel_job(), {'ok': True})[1],
            '/api/reset': lambda _b: (state.reset(), {'ok': True})[1],
        }
        try:
            if path == '/api/login':
                body = self._read_json()
                if self.server.token is None:
                    self._send_json({'ok': True})
                elif self._authorized(str(body.get('token', '')).strip()):
                    self._send_bytes(
                        b'{"ok": true}',
                        'application/json; charset=utf-8',
                        extra_headers={'Set-Cookie': self._token_cookie()},
                    )
                else:
                    raise ApiError('Invalid token.', HTTPStatus.UNAUTHORIZED)
                return
            if not self._authorized():
                raise ApiError('Sign in required.', HTTPStatus.UNAUTHORIZED)
            handler = routes.get(path)
            if handler is None:
                raise ApiError('Not found', HTTPStatus.NOT_FOUND)
            body = self._read_json()
            self._send_json(handler(body))
        except ApiError as exc:
            self._send_json({'error': str(exc)}, status=exc.status)
        except Exception as exc:  # noqa: BLE001
            logger.exception('Unhandled error in POST %s', path)
            self._send_json(
                {'error': str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR
            )

    # ------------------------------------------------------------ helpers
    def _read_json(self) -> dict[str, Any]:
        """
        Read and decode the request body as a JSON object.

        Returns:
            dict[str, Any]: The decoded object (``{}`` for an empty body).

        Raises:
            ApiError: If the body is not valid JSON or not a JSON object.
        """
        length = int(self.headers.get('Content-Length') or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode('utf-8') or '{}')
        except json.JSONDecodeError as exc:
            raise ApiError(f'Invalid JSON body: {exc}') from exc
        if not isinstance(data, dict):
            raise ApiError('JSON body must be an object.')
        return data

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        """Send ``payload`` as a UTF-8 JSON response with ``status``."""
        self._send_bytes(
            json.dumps(payload).encode('utf-8'),
            'application/json; charset=utf-8',
            status=status,
        )

    def _send_bytes(
        self,
        data: bytes,
        content_type: str,
        status: int = HTTPStatus.OK,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        """
        Send a complete response with ``Content-Length`` and ``no-store`` caching.

        Args:
            data (bytes): Response body.
            content_type (str): Value for the ``Content-Type`` header.
            status (int): HTTP status. Defaults to ``200``.
            extra_headers (dict[str, str] | None): Additional headers; they
                override the default ``Cache-Control``.
        """
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        headers = {'Cache-Control': 'no-store', **(extra_headers or {})}
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, attachment: bool = False) -> None:
        """
        Send a file from disk, guessing its MIME type.

        Args:
            path (Path): The file to send.
            attachment (bool): If ``True``, add a ``Content-Disposition``
                header so the browser downloads rather than displays it.

        Raises:
            ApiError: ``404`` if ``path`` is not a file.
        """
        if not path.is_file():
            raise ApiError('Not found', HTTPStatus.NOT_FOUND)
        content_type = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        if path.suffix == '.geojson':
            content_type = 'application/geo+json'
        headers = {}
        if attachment:
            headers['Content-Disposition'] = f'attachment; filename="{path.name}"'
        self._send_bytes(path.read_bytes(), content_type, extra_headers=headers)


class GuiServer(ThreadingHTTPServer):
    """
    ``ThreadingHTTPServer`` that carries the shared application state.

    While the server exists, a
    :class:`~rapidtools.gui.logging_utils.CallbackLogHandler` is attached to
    the root logger so that log output from the workflow shows up in the
    browser; :meth:`server_close` removes it again.

    Example:
        >>> from rapidtools.gui.server import AppState, GuiServer
        >>>
        >>> server = GuiServer(('127.0.0.1', 0), AppState('out'), token='s3cret')
        >>> server.share_urls()[0].endswith('?token=s3cret')
        True
        >>> server.server_close()
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        state: AppState,
        verbose: bool = False,
        token: str | None = None,
        data_root: str | Path | None = None,
    ) -> None:
        """
        Bind the server.

        Args:
            address (tuple[str, int]): ``(host, port)``; port ``0`` picks a free
                one.
            state (AppState): The shared application state.
            verbose (bool): Log every HTTP request at DEBUG level.
            token (str | None): Shared access token; empty/``None`` disables
                authentication.
            data_root (str | Path | None): Restrict all file access to this
                directory.

        Raises:
            OSError: If the address cannot be bound.
        """
        self.state = state
        self.token = token or None
        self.api = Api(state, data_root=data_root)
        state.data_root = self.api.data_root
        state.token_required = self.token is not None
        self.verbose = verbose
        self._log_handler: CallbackLogHandler | None = None
        super().__init__(address, GuiRequestHandler)
        self._log_handler = CallbackLogHandler(state.log.add_line)
        logging.getLogger().addHandler(self._log_handler)

    def server_close(self) -> None:
        """Detach the GUI log handler and release the listening socket."""
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None
        super().server_close()

    @property
    def url(self) -> str:
        """
        Return the URL to open locally.

        Returns:
            str: ``http://localhost:<port>/`` for loopback / wildcard binds,
            otherwise the bound host.
        """
        host, port = self.server_address[:2]
        display_host = 'localhost' if host in ('0.0.0.0', '127.0.0.1', '') else host
        return f'http://{display_host}:{port}/'

    def share_urls(self) -> list[str]:
        """
        Links colleagues can open, including the token when one is set.

        When bound to all interfaces the hostname and the best-effort LAN IP
        are both listed; otherwise the bound host is used.

        Returns:
            list[str]: Unique URLs, each ending in ``?token=<token>`` when a
            token is configured.
        """
        host, port = self.server_address[:2]
        hosts: list[str] = []
        if host in ('0.0.0.0', ''):
            hosts.append(socket.gethostname())
            lan_ip = _lan_ip()
            if lan_ip:
                hosts.append(lan_ip)
        else:
            hosts.append('localhost' if host == '127.0.0.1' else host)
        suffix = f'?token={self.token}' if self.token else ''
        return [f'http://{h}:{port}/{suffix}' for h in dict.fromkeys(hosts)]


def _lan_ip() -> str | None:
    """
    Best-effort LAN address of this machine (no packets are sent).

    Returns:
        str | None: The IPv4 address the OS would use to reach a remote host,
        or ``None`` when the machine has no route (offline).
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(('10.255.255.255', 1))
            return sock.getsockname()[0]
    except OSError:
        return None


def create_server(
    host: str = '127.0.0.1',
    port: int = 8765,
    output_dir: str | Path | None = None,
    verbose: bool = False,
    token: str | None = None,
    data_root: str | Path | None = None,
) -> GuiServer:
    """
    Build (but do not start) the GUI server.

    Args:
        host: Interface to bind. Use ``'0.0.0.0'`` to let colleagues connect.
        port: TCP port.
        output_dir: Where downloads, crops and results are written.
        verbose: Log every HTTP request.
        token: Shared access token. ``'auto'`` generates a random one. When
            set, every visitor must present it (via the share link or the
            sign-in page) before using the app.
        data_root: Restrict the file browser and every user-supplied path to
            this directory.

    Returns:
        GuiServer: A bound (not yet serving) server.

    Raises:
        OSError: If ``host:port`` cannot be bound.

    Example:
        >>> from rapidtools.gui.server import create_server
        >>>
        >>> server = create_server(port=0, output_dir='out', token='auto')
        >>> len(server.token) > 0
        True
        >>> server.server_close()
    """
    output_dir = Path(output_dir or Path.cwd() / 'asset_analysis_output')
    if token == 'auto':
        token = secrets.token_urlsafe(18)
    return GuiServer(
        (host, port),
        AppState(output_dir),
        verbose=verbose,
        token=token,
        data_root=data_root,
    )


def launch_asset_analysis_app(
    host: str = '127.0.0.1',
    port: int = 8765,
    output_dir: str | Path | None = None,
    open_browser: bool = True,
    token: str | None = None,
    data_root: str | Path | None = None,
) -> None:
    """
    Start the GUI server, open a browser tab, and block until Ctrl+C.

    Args:
        host (str): Interface to bind. Defaults to ``'127.0.0.1'``. A warning
            is logged when binding a non-loopback interface without a token.
        port (int): TCP port. Defaults to ``8765``.
        output_dir (str | Path | None): Where downloads and results are
            written. Defaults to ``./asset_analysis_output``.
        open_browser (bool): Open the app in the default browser shortly
            after start-up. Defaults to ``True``.
        token (str | None): Shared access token; ``'auto'`` generates one.
        data_root (str | Path | None): Restrict all file access to this
            directory.

    Raises:
        SystemExit: With code ``1`` if the port cannot be bound (e.g. another
            instance is already running).

    Example:
        >>> from rapidtools.gui.server import launch_asset_analysis_app
        >>>
        >>> launch_asset_analysis_app(port=9000, open_browser=False)  # blocks
    """
    if host not in ('127.0.0.1', 'localhost', '::1') and not token:
        logger.warning(
            'The GUI is reachable from other machines without an access token. '
            'Add --token auto (or --token <secret>) before sharing it.'
        )
    try:
        server = create_server(
            host=host,
            port=port,
            output_dir=output_dir,
            token=token,
            data_root=data_root,
        )
    except OSError as exc:
        logger.error(
            f'Could not start the rapidtools GUI on {host}:{port} ({exc}). '
            'Is another instance already running? Pick another port with --port.'
        )
        raise SystemExit(1) from exc
    url = server.url
    logger.info(f'rapidtools GUI is running at {url} (press Ctrl+C to stop)')
    if server.token:
        logger.info(f'Access token: {server.token}')
        for link in server.share_urls():
            logger.info(f'Share this link with colleagues: {link}')
    if server.api.data_root:
        logger.info(f'File access is limited to {server.api.data_root}')
    if open_browser:
        first = server.share_urls()[-1] if server.token else url
        if server.token:
            first = f'{url}?token={server.token}'
        threading.Timer(0.5, lambda: webbrowser.open(first)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info('Shutting down the rapidtools GUI.')
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    """
    Console-script entry point (``rapidtools-gui``).

    Parses ``--host``, ``--port``, ``--output-dir``, ``--no-browser``,
    ``--token`` (also read from ``RAPIDTOOLS_GUI_TOKEN``) and ``--data-root``,
    then calls :func:`launch_asset_analysis_app`. Setting the
    ``RAPIDTOOLS_GUI_NO_BROWSER`` environment variable implies
    ``--no-browser``.

    Args:
        argv (list[str] | None): Arguments to parse instead of
            ``sys.argv[1:]``.

    Returns:
        int: Process exit code (``0``).

    Example:
        >>> from rapidtools.gui.server import main
        >>>
        >>> main(['--port', '9000', '--no-browser'])  # blocks until Ctrl+C
        0
    """
    parser = argparse.ArgumentParser(
        prog='rapidtools-gui',
        description='Detect assets in aerial imagery and run VLM inference on them.',
    )
    parser.add_argument('--host', default='127.0.0.1', help='interface to bind')
    parser.add_argument('--port', type=int, default=8765, help='port to listen on')
    parser.add_argument(
        '--output-dir',
        default=None,
        help='where downloads and results are written '
        '(default: ./asset_analysis_output)',
    )
    parser.add_argument(
        '--no-browser', action='store_true', help='do not open a browser tab'
    )
    parser.add_argument(
        '--token',
        default=os.environ.get('RAPIDTOOLS_GUI_TOKEN'),
        help='shared access token required to use the app; "auto" generates one '
        '(also read from RAPIDTOOLS_GUI_TOKEN)',
    )
    parser.add_argument(
        '--data-root',
        default=None,
        help='limit the file browser and all file paths to this directory',
    )
    args = parser.parse_args(argv)
    if os.environ.get('RAPIDTOOLS_GUI_NO_BROWSER'):
        args.no_browser = True
    # The console script is an application: show library progress on stdout.
    configure_logging()
    launch_asset_analysis_app(
        host=args.host,
        port=args.port,
        output_dir=args.output_dir,
        token=args.token,
        data_root=args.data_root,
        open_browser=not args.no_browser,
    )
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
