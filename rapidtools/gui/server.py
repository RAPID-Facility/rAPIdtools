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
import datetime as _dt
import gzip
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
from collections import OrderedDict
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
from rapidtools.gui.notify import JobSummary, NotificationConfig, Notifier
from rapidtools.gui.preview import (
    RasterPreview,
    RasterTiler,
    project_collection,
    render_preview,
)
from rapidtools.gui.prompt_builder import (
    ASSIST_ACTIONS,
    SAMPLE_SPEC,
    PromptSpec,
    assemble_prompt,
    describe_marking,
)
from rapidtools.gui.workflow import (
    BASEMAP_PROVIDERS,
    DEFAULT_GEMINI_MODEL_ID,
    GEMMA4_MODEL_IDS,
    IMAGERY_SOURCES,
    MODEL_BACKENDS,
    STREET_CLASS_EXAMPLES,
    AssetAnalysisWorkflow,
    AssistSettings,
    DetectionSettings,
    InferenceSettings,
    RegionImagerySettings,
    StreetDetectionSettings,
    list_available_models,
    parse_asset_list,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / 'static'
SAMPLE_RASTERS = ('eaton_patch1', 'eaton_patch2')
# Prompts from the dataset registry offered in the GUI, keyed by short name.
SAMPLE_PROMPTS: dict[str, dict[str, str]] = {
    'aerial_chs': {
        'dataset': 'aerial_chs_prompts',
        'label': 'Aerial CHS combustion level (buildings)',
        'imagery': 'aerial',
    },
    'street_chs': {
        'dataset': 'street_chs_prompts',
        'label': 'Street-level CHS combustion level (buildings)',
        'imagery': 'street',
    },
    'street_recovery': {
        'dataset': 'street_recovery_prompts',
        'label': 'Street-level recovery progress (two dates, JSON)',
        'imagery': 'multi-temporal street',
    },
}
SAMPLE_PROMPT_DATASET = SAMPLE_PROMPTS['aerial_chs']['dataset']
MAPILLARY_TOKEN_DATASET = 'mapillary_token'
# Basemap tiles served to the map picker: cache size and allowed providers.
TILE_CACHE_SIZE = 800
# Coverage lookups list Mapillary's zoom-14 tiles; cap how many one view may span.
MAX_COVERAGE_TILES = 36
MAX_COVERAGE_POINTS = 4000
_TILE_SESSION = None
_TILE_SESSION_LOCK = threading.Lock()
SEQUENCE_CACHE_SIZE = 600
# Address search goes through OpenStreetMap's Nominatim (no key; keep it light).
GEOCODE_URL = 'https://nominatim.openstreetmap.org/search'
MAX_OVERLAY_FEATURES = 5000
# Low-zoom index of survey routes, built once per server and cached on disk so
# the map can show where street-level imagery exists before the user zooms in.
OVERVIEW_BBOX = (-125.0, 24.0, -66.0, 50.0)  # continental US
OVERVIEW_ZOOM = 6
OVERVIEW_MAX_AGE_S = 7 * 86400
OVERVIEW_WORKERS = 8
# Route database: every RAPID survey route at zoom-13 detail (about 1 m), built
# once from the coverage tiles that the overview says contain routes, stored
# gzipped under the cache directory and served to the map in one request.
ROUTES_ZOOM = 13
ROUTES_FILE = f'survey_routes_z{ROUTES_ZOOM}.json.gz'


def _routes_tiles(
    lines: list[list[float]], zoom: int = ROUTES_ZOOM
) -> list[tuple[int, int, int]]:
    """
    The ``(x, y, zoom)`` tiles touched by longitude/latitude polylines.

    Example:
        >>> _routes_tiles([[-117.49, 47.71, -117.48, 47.72]], zoom=13)
        [(1422, 2857, 13)]
    """
    import math

    n = 2**zoom
    tiles: set[tuple[int, int]] = set()
    for line in lines:
        lons, lats = line[0::2], line[1::2]
        if not lons:
            continue

        def tx(lon: float) -> int:
            return min(n - 1, max(0, int((lon + 180.0) / 360.0 * n)))

        def ty(lat: float) -> int:
            r = math.radians(max(-85.0, min(85.0, lat)))
            return min(
                n - 1,
                max(
                    0,
                    int(
                        (1 - math.log(math.tan(r) + 1 / math.cos(r)) / math.pi) / 2 * n
                    ),
                ),
            )

        for x in range(tx(min(lons)), tx(max(lons)) + 1):
            for y in range(ty(max(lats)), ty(min(lats)) + 1):
                tiles.add((x, y))
    return [(x, y, zoom) for x, y in sorted(tiles)]


def _overview_tiles() -> list[tuple[int, int, int]]:
    """The ``(x, y, z)`` coverage tiles that make up the overview index."""
    from rapidtools.core import BoundingBox
    from rapidtools.data_sources import TileUtils

    return TileUtils.bbox_to_mapbox_tiles(
        BoundingBox(*OVERVIEW_BBOX), zoom=OVERVIEW_ZOOM
    )


mimetypes.add_type('image/webp', '.webp')


def _geocode(query: str, limit: int = 6) -> list[dict[str, Any]]:
    """
    Look an address or place name up with Nominatim.

    Args:
        query (str): Free text such as ``'Altadena, CA'``.
        limit (int): Maximum number of matches.

    Returns:
        list[dict[str, Any]]: ``name``, ``lon``, ``lat`` and ``bbox``
        (``[min_lon, min_lat, max_lon, max_lat]``) per match, best first.

    Raises:
        requests.RequestException: On a network error or non-2xx reply.
    """
    import requests

    from rapidtools.config import REQUESTS_HEADERS, REQUESTS_TIMEOUT_VAL

    params: dict[str, str | int] = {'q': query, 'format': 'jsonv2', 'limit': limit}
    response = requests.get(
        GEOCODE_URL,
        params=params,
        headers={**REQUESTS_HEADERS, 'User-Agent': 'rapidtools-gui (asset analysis)'},
        timeout=REQUESTS_TIMEOUT_VAL,
    )
    response.raise_for_status()
    matches = []
    for item in response.json():
        try:
            south, north, west, east = (float(v) for v in item['boundingbox'])
            matches.append(
                {
                    'name': str(item.get('display_name', '')),
                    'lon': float(item['lon']),
                    'lat': float(item['lat']),
                    'bbox': [west, south, east, north],
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return matches


def _fetch_basemap_tile(provider: str, z: int, x: int, y: int) -> bytes | None:
    """
    Download one 256 px satellite tile from Bing or Google.

    Args:
        provider (str): ``'bing'`` or ``'google'``.
        z (int): Zoom level.
        x (int): Tile column.
        y (int): Tile row.

    Returns:
        bytes | None: JPEG/PNG bytes, or ``None`` when the request fails.

    Raises:
        ValueError: For an unknown provider or an off-grid address.
    """
    global _TILE_SESSION
    if provider not in BASEMAP_PROVIDERS:
        raise ValueError(f'Unknown basemap provider {provider!r}.')
    if not (0 <= z <= BASEMAP_PROVIDERS[provider]['max_zoom']) or not (
        0 <= x < 2**z and 0 <= y < 2**z
    ):
        raise ValueError('Tile address is outside the grid.')
    if provider == 'bing':
        from rapidtools.data_sources.bing_aerial_image_extractor import (
            BING_TILE_URL,
            BingAerialImageExtractor,
        )

        url = BING_TILE_URL.format(
            quadkey=BingAerialImageExtractor.tile_to_quadkey(x, y, z)
        )
    else:
        from rapidtools.data_sources.google_aerial_image_extractor import (
            GOOGLE_TILE_URL,
        )

        url = GOOGLE_TILE_URL.format(subdomain=(x + y) % 4, layer='s', x=x, y=y, z=z)
    with _TILE_SESSION_LOCK:
        if _TILE_SESSION is None:
            from rapidtools.config import get_configured_session

            _TILE_SESSION = get_configured_session()
        session = _TILE_SESSION
    try:
        from rapidtools.config import REQUESTS_TIMEOUT_VAL

        response = session.get(url, timeout=REQUESTS_TIMEOUT_VAL)
        if response.status_code == 200 and response.content:
            return response.content
    except Exception as exc:  # noqa: BLE001 - a missing tile is not an error
        logger.debug(f'Basemap tile {provider} {z}/{x}/{y} failed: {exc}')
    return None


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
        self.basemap_label = ''
        # Where to tell the user when the running job ends (email or webhook),
        # the sender, the link put in messages and the last delivery outcome.
        self.notifier = Notifier()
        self.notify: dict[str, str] = {'target': '', 'kind': ''}
        self.notify_last: dict[str, Any] = {'seq': 0, 'ok': None, 'message': ''}
        self.public_url = ''
        # Satellite tiles served to the map, most recent last.
        self.tile_cache: OrderedDict[tuple[str, int, int, int], bytes] = OrderedDict()
        self.tile_lock = threading.Lock()
        self.mapillary_clients: dict[str, Any] = {}
        # Decoded Mapillary coverage tiles (survey sequences), per filter.
        self.sequence_cache: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
        # Low-zoom survey overview, keyed by the RAPID-only flag.
        self.overview: dict[bool, dict[str, Any]] = {}
        # The RAPID route database: build status and the gzipped payload.
        self.routes: dict[str, Any] = {
            'status': 'idle',
            'done': 0,
            'total': 0,
            'n_routes': 0,
            'built_at': '',
            'error': '',
        }
        self.routes_gz: bytes | None = None
        self.cache_dir = Path.home() / '.cache' / 'rapidtools'
        # Outcome of the latest prompt-assistant request; ``seq`` increments
        # per request so the browser can tell a new answer from an old one.
        self.assistant: dict[str, Any] = {
            'seq': 0,
            'status': 'idle',
            'action': None,
            'model': '',
            'result': None,
            'error': '',
        }

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
            self.basemap_label = ''
            self.log.clear()

    def set_assistant(self, **fields: Any) -> None:
        """
        Update the prompt-assistant record shown to the browser.

        Args:
            **fields: Keys of :attr:`assistant` to replace (``status``,
                ``result``, ``error``, ...). Passing ``seq=True`` bumps the
                sequence number.
        """
        with self.lock:
            if fields.pop('seq', False):
                self.assistant['seq'] += 1
            self.assistant.update(fields)

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
        """Record the job outcome, append ``message`` to the log and notify."""
        with self.lock:
            self.log.end_progress()
            self.job.update(status=status, message=message, finished_at=time.time())
            self.status = message
            self.log.add_line(message)
            target = self.notify['target']
            summary = JobSummary(
                name=str(self.job.get('name') or 'Job'),
                status=status,
                message=message,
                duration_s=self.job['finished_at'] - self.job.get('started_at', 0),
                results=[dict(r) for r in self.results],
                url=self.notifier.config.public_url or self.public_url,
            )
            if target and status != 'cancelled':
                # One notification per request: the address is kept in the
                # browser, so the next run asks again.
                self.notify = {'target': '', 'kind': ''}
        if target and status != 'cancelled':
            self.notifier.send_in_background(target, summary, self._on_notified)

    def _on_notified(self, ok: bool, message: str) -> None:
        """Record a notification delivery result in the log and state."""
        with self.lock:
            self.notify_last = {
                'seq': self.notify_last['seq'] + 1,
                'ok': ok,
                'message': message,
            }
            self.log.add_line(message)

    def set_notification(self, target: str) -> dict[str, str]:
        """
        Choose where to send a message when the current job finishes.

        Args:
            target (str): Email address, webhook URL, or ``''`` to clear.

        Returns:
            dict[str, str]: ``{'target', 'kind'}`` as stored.

        Raises:
            ValueError: If the target is unusable (see
                :meth:`~rapidtools.gui.notify.Notifier.validate_target`).
        """
        target = (target or '').strip()
        kind = self.notifier.validate_target(target) if target else ''
        with self.lock:
            self.notify = {'target': target, 'kind': kind}
            if target:
                self.log.add_line(
                    f'Will notify {target} when the job finishes.'
                    if self.busy()
                    else f'Will notify {target} when the next job finishes.'
                )
            return dict(self.notify)

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
            assistant = dict(self.assistant)
            return {
                'output_dir': str(self.output_dir),
                'raster': self.previews['raster'].to_json()
                if 'raster' in self.previews
                else None,
                'basemap': self.previews['basemap'].to_json()
                if 'basemap' in self.previews
                else None,
                'collection': collection,
                'basemap_label': self.basemap_label,
                'job': job,
                'assistant': assistant,
                'notify': {
                    **self.notify,
                    'email_available': self.notifier.email_available,
                    'last': dict(self.notify_last),
                },
                'busy': self.busy(),
                'status': self.status,
                'results': list(self.results),
                'log_seq': self.log.seq,
                'options': {
                    'sample_rasters': list(SAMPLE_RASTERS),
                    'sample_prompts': {
                        k: {'label': v['label'], 'imagery': v['imagery']}
                        for k, v in SAMPLE_PROMPTS.items()
                    },
                    'gemini_model': DEFAULT_GEMINI_MODEL_ID,
                    'gemma_models': list(GEMMA4_MODEL_IDS),
                    'backends': MODEL_BACKENDS,
                    'basemaps': BASEMAP_PROVIDERS,
                    'imagery_sources': IMAGERY_SOURCES,
                    'street_classes': list(STREET_CLASS_EXAMPLES),
                    'assist_actions': list(ASSIST_ACTIONS),
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

    def __init__(self, state: AppState, data_root: str | Path | None = None) -> None:
        """
        Bind the API to a state object.

        Args:
            state (AppState): The shared server state.
            data_root (str | Path | None): If given, every user-supplied path must lie
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

    def download_region_imagery(self, body: dict[str, Any]) -> dict[str, Any]:
        """
        ``POST /api/imagery/region``: stitch a satellite basemap and preview it.

        Args:
            body (dict[str, Any]): ``provider`` (``'bing'`` or ``'google'``),
                ``zoom`` and either ``min_lon``/``min_lat``/``max_lon``/
                ``max_lat`` or ``geojson`` (a file whose extent is used).

        Returns:
            dict[str, Any]: ``{'ok': True}`` once the download job is queued.

        Raises:
            ApiError: If the region or provider is invalid or the GeoJSON is
                outside ``data_root``.
        """
        geojson = str(body.get('geojson', '') or '').strip()
        try:
            settings = RegionImagerySettings(
                output_dir=self.state.output_dir,
                provider=str(body.get('provider', 'bing')),
                zoom=int(body.get('zoom') or 19),
                min_lon=_optional_float(body.get('min_lon')),
                min_lat=_optional_float(body.get('min_lat')),
                max_lon=_optional_float(body.get('max_lon')),
                max_lat=_optional_float(body.get('max_lat')),
                geojson_path=self._user_path(geojson) if geojson else None,
            )
        except (TypeError, ValueError) as exc:
            raise ApiError(str(exc)) from exc
        state = self.state

        def job() -> None:
            """Download the basemap into ``output_dir`` and render its preview."""
            path = state.workflow.download_basemap(settings)
            self._load_raster(Path(path))

        label = BASEMAP_PROVIDERS[settings.provider]['label']
        state.start_job(f'Download {label} imagery', job)
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
            state.basemap_label = ''
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

    def discover_street(self, body: dict[str, Any]) -> dict[str, Any]:
        """
        ``POST /api/street/discover``: find objects along a Mapillary survey.

        Args:
            body (dict[str, Any]): Options, see :meth:`_street_settings`.

        Returns:
            dict[str, Any]: ``{'ok': True}`` once the job is queued.

        Raises:
            ApiError: If no raster is loaded, the options are invalid, or a job
                is already running.
        """
        settings = self._street_settings(body)
        self.state.start_job(
            'Street discovery', lambda: self._run_street_discovery(settings)
        )
        return {'ok': True}

    def run_all(self, body: dict[str, Any]) -> dict[str, Any]:
        """
        ``POST /api/run_all``: detection followed by inference as one job.

        Args:
            body (dict[str, Any]): ``{'detect': {...}, 'infer': {...}}`` with the
                options accepted by :meth:`detect` and :meth:`infer`, or
                ``{'mode': 'street', 'street': {...}, 'infer': {...}}`` to run
                :meth:`discover_street` first instead.

        Returns:
            dict[str, Any]: ``{'ok': True}`` once the job is queued.

        Raises:
            ApiError: If either option set is invalid or a job is running.
        """
        inference = self._inference_settings(body.get('infer') or {})
        if body.get('mode') == 'street':
            street = self._street_settings(body.get('street') or {})

            def street_job() -> None:
                """Discover objects, then analyze them."""
                collection = self._run_street_discovery(street)
                if len(collection) == 0:
                    raise ValueError('No objects were found; skipping inference.')
                self._run_inference(collection, inference)

            self.state.start_job('Street discovery + inference', street_job)
            return {'ok': True}

        detection = self._detection_settings(body.get('detect') or {})

        def job() -> None:
            """Detect, then analyze; fail loudly if nothing was detected."""
            collection = self._run_detection(detection)
            if len(collection) == 0:
                raise ValueError('No assets were detected; skipping inference.')
            self._run_inference(collection, inference)

        self.state.start_job('Detection + inference', job)
        return {'ok': True}

    def _run_street_discovery(
        self, settings: StreetDetectionSettings
    ) -> PhysicalAssetCollection:
        """
        Run street discovery and publish its results.

        Args:
            settings (StreetDetectionSettings): Validated options.

        Returns:
            PhysicalAssetCollection: The discovered point assets.
        """
        state = self.state
        result = state.workflow.discover_street(settings)
        with state.lock:
            state.results = [r for r in state.results if r['label'] != 'Loaded assets']
        state.set_collection(result.collection, 'street survey')
        state.add_result('Street objects', result.geojson_path)
        return result.collection

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
            provider = result.basemap or settings.basemap
            label = BASEMAP_PROVIDERS.get(provider, {}).get('label', 'Basemap')
            state.set_status(f'Rendering preview of the {label} basemap...')
            state.set_preview('basemap', render_preview(result.detection_raster))
            with state.lock:
                state.basemap_label = f'{label} basemap'
            state.add_result(f'{label} basemap', result.detection_raster)
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
                required), ``asset_size_m``, ``basemap`` (``'bing'``,
                ``'google'`` or ``'recon'``; the older ``source`` key is
                accepted), ``threshold``, ``mask_threshold``, ``regularize``
                and ``merge_overlaps``.

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
                basemap=str(body.get('basemap') or body.get('source') or 'bing'),
                threshold=float(body.get('threshold', 0.5)),
                mask_threshold=float(body.get('mask_threshold', 0.4)),
                regularize=bool(body.get('regularize', True)),
                merge_overlaps=bool(body.get('merge_overlaps', True)),
            )
        except (TypeError, ValueError) as exc:
            raise ApiError(str(exc)) from exc

    def _street_settings(self, body: dict[str, Any]) -> StreetDetectionSettings:
        """
        Validate a street-discovery request body.

        Args:
            body (dict[str, Any]): ``classes`` (comma separated, required),
                ``mapillary_token`` (required), ``start_date``, ``end_date``,
                ``rapid_only``, ``detection_source``, ``frame_spacing_m``,
                ``min_observations``, ``cluster_radius_m``,
                ``camera_height_m`` and ``max_range_m``. The survey area is
                the extent of the loaded raster unless ``min_lon``,
                ``min_lat``, ``max_lon`` and ``max_lat`` give a box (picked
                on the map), in which case no raster is needed.

        Returns:
            StreetDetectionSettings: The validated settings.

        Raises:
            ApiError: If neither a raster nor a box defines the area, or a
                value is invalid.
        """
        region = _optional_bbox(body)
        with self.state.lock:
            raster = self.state.raster_path
        if region is None and raster is None:
            raise ApiError(
                'Load an aerial image first (step 1), or pick an area on the map.'
            )
        try:
            return StreetDetectionSettings(
                output_dir=self.state.output_dir,
                classes=parse_asset_list(str(body.get('classes', ''))),
                access_token=str(body.get('mapillary_token', '')),
                raster_path=raster,
                region=region,
                start_date=str(body.get('start_date', '') or ''),
                end_date=str(body.get('end_date', '') or ''),
                filter_rapid_only=bool(body.get('rapid_only', True)),
                detection_source=str(body.get('detection_source') or 'auto'),
                frame_spacing_m=float(body.get('frame_spacing_m', 3.0)),
                min_observations=int(body.get('min_observations', 2)),
                cluster_radius_m=float(body.get('cluster_radius_m', 4.0)),
                camera_height_m=float(body.get('camera_height_m', 2.4)),
                max_range_m=float(body.get('max_range_m', 60.0)),
            )
        except (TypeError, ValueError) as exc:
            raise ApiError(str(exc)) from exc

    def _inference_settings(self, body: dict[str, Any]) -> InferenceSettings:
        """
        Validate an inference request body.

        Args:
            body (dict[str, Any]): ``backend`` (a :data:`MODEL_BACKENDS` key),
                ``prompt``, ``api_key``, ``model_id``, ``max_workers``,
                ``batch_size``, ``load_in_4bit``, ``temperature``,
                ``max_tokens``, ``json_mode``, ``overlay_outline`` and the
                optional ``outline_shape``, ``outline_buffer``,
                ``outline_width`` and ``outline_color`` (blank values fall
                back to the extractor defaults); ``imagery`` (an
                :data:`IMAGERY_SOURCES` key) with ``min_footprint_coverage``,
                ``pad_edges``, ``street_search_radius_m``,
                ``street_max_images``, ``street_crop_top``,
                ``street_crop_bottom``, ``mapillary_token``,
                ``mapillary_start_date``, ``mapillary_end_date``,
                ``mapillary_rapid_only``, ``mapillary_max_images``,
                ``object_image_size`` and ``object_crop_buffer``.

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
                temperature=float(body.get('temperature', 0.4)),
                max_tokens=int(body.get('max_tokens', 2048)),
                json_mode=bool(body.get('json_mode', False)),
                load_in_4bit=bool(body.get('load_in_4bit', True)),
                overlay_asset_outline=bool(body.get('overlay_outline', True)),
                outline_shape=str(body.get('outline_shape') or 'geometry'),
                outline_buffer=body.get('outline_buffer') or 0,
                outline_width=body.get('outline_width') or 6,
                outline_color=str(body.get('outline_color') or 'red'),
                imagery=str(body.get('imagery') or 'aerial'),
                min_footprint_coverage=_optional_float(
                    body.get('min_footprint_coverage')
                ),
                pad_edges=bool(body['pad_edges']) if 'pad_edges' in body else None,
                street_search_radius_m=float(body.get('street_search_radius_m', 50)),
                street_max_images=int(body.get('street_max_images', 1)),
                street_vertical_crop=(
                    float(body.get('street_crop_top', 0.0) or 0.0),
                    float(body.get('street_crop_bottom', 1.0) or 1.0),
                ),
                mapillary_token=str(body.get('mapillary_token', '') or ''),
                mapillary_start_date=str(body.get('mapillary_start_date', '') or ''),
                mapillary_end_date=str(body.get('mapillary_end_date', '') or ''),
                mapillary_rapid_only=bool(body.get('mapillary_rapid_only', True)),
                mapillary_max_images=int(body.get('mapillary_max_images', 4)),
                object_image_size=str(body.get('object_image_size') or '2048'),
                object_crop_buffer=body.get('object_crop_buffer') or '25%',
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

    def sample_prompt(self, body: dict[str, Any]) -> dict[str, Any]:
        """
        ``POST /api/prompt/sample``: fetch a sample prompt from the registry.

        Args:
            body (dict[str, Any]): Optional ``name``, a :data:`SAMPLE_PROMPTS`
                key (default ``'aerial_chs'``).

        Returns:
            dict[str, Any]: ``{'name': <key>, 'text': <prompt text>}``.

        Raises:
            ApiError: If ``name`` is not a known sample prompt.
        """
        name = str(body.get('name') or 'aerial_chs').strip()
        entry = SAMPLE_PROMPTS.get(name)
        if entry is None:
            raise ApiError(f'Unknown sample prompt: {name!r}')
        text = self._download_text(entry['dataset'])
        return {'name': name, 'text': text}

    def set_notification(self, body: dict[str, Any]) -> dict[str, Any]:
        """
        ``POST /api/notify``: send a message when the current job finishes.

        Args:
            body (dict[str, Any]): ``target``, an email address or a webhook
                URL; empty clears the request.

        Returns:
            dict[str, Any]: ``{'target', 'kind'}`` as stored.

        Raises:
            ApiError: If the target is not deliverable.
        """
        try:
            return self.state.set_notification(str(body.get('target', '') or ''))
        except ValueError as exc:
            raise ApiError(str(exc)) from exc

    def basemap_tile(self, provider: str, z: int, x: int, y: int) -> bytes:
        """
        ``GET /api/basemap_tile/<provider>/<z>/<x>/<y>.jpg``: a satellite tile.

        Tiles are fetched server-side (so the browser needs no cross-origin
        access to Bing or Google) and kept in a bounded cache.

        Args:
            provider (str): ``'bing'`` or ``'google'``.
            z (int): Zoom level.
            x (int): Tile column.
            y (int): Tile row.

        Returns:
            bytes: Image data.

        Raises:
            ApiError: ``400`` for a bad address, ``404`` when the tile could
                not be fetched.
        """
        key = (provider, z, x, y)
        state = self.state
        with state.tile_lock:
            data = state.tile_cache.get(key)
            if data is not None:
                state.tile_cache.move_to_end(key)
                return data
        try:
            data = _fetch_basemap_tile(provider, z, x, y)
        except ValueError as exc:
            raise ApiError(str(exc)) from exc
        if data is None:
            raise ApiError('Tile unavailable.', HTTPStatus.NOT_FOUND)
        with state.tile_lock:
            state.tile_cache[key] = data
            while len(state.tile_cache) > TILE_CACHE_SIZE:
                state.tile_cache.popitem(last=False)
        return data

    def street_coverage(self, query: dict[str, Any]) -> dict[str, Any]:
        """
        ``GET /api/street/coverage``: where a Mapillary survey has images.

        Lists the images in the box from Mapillary's coverage tiles (no
        per-image requests) so the map picker can show where a street-level
        survey went before an area is chosen.

        Args:
            query (dict[str, Any]): ``min_lon``, ``min_lat``, ``max_lon``,
                ``max_lat``; optional ``token`` (the RAPID token is used when
                empty), ``rapid_only`` (default ``1``), ``start_date`` and
                ``end_date``.

        Returns:
            dict[str, Any]: ``count`` of images, ``points`` as
                ``[[lon, lat], ...]`` (thinned to at most
                :data:`MAX_COVERAGE_POINTS`), ``sequences``, ``first`` and
                ``last`` capture dates and ``truncated``.

        Raises:
            ApiError: If the box is invalid or spans too many tiles.
        """
        from rapidtools.data_sources import TileUtils

        bbox = _optional_bbox(query)
        if bbox is None:
            raise ApiError('Give min_lon, min_lat, max_lon and max_lat.')
        n_tiles = len(TileUtils.bbox_to_mapbox_tiles(bbox, zoom=14))
        if n_tiles > MAX_COVERAGE_TILES:
            raise ApiError(
                f'Zoom in to look up survey coverage: the view spans {n_tiles} '
                f'tiles, at most {MAX_COVERAGE_TILES} are searched at once.'
            )
        client = self._mapillary_client(str(query.get('token', '') or ''))
        rapid_only = str(query.get('rapid_only', '1')).lower() not in ('0', 'false', '')
        images = client.fetch_images_in_bbox(
            bbox,
            start_date=str(query.get('start_date', '') or ''),
            end_date=str(query.get('end_date', '') or ''),
            filter_rapid_only=rapid_only,
        )
        min_lon, min_lat, max_lon, max_lat = bbox.bounds
        points: list[list[float]] = []
        sequences: set[str] = set()
        dates: list[str] = []
        for image in images:
            props = image.properties
            lon, lat = props.get('longitude'), props.get('latitude')
            if lon is None or lat is None:
                continue
            if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
                continue
            points.append([round(float(lon), 6), round(float(lat), 6)])
            if props.get('sequence'):
                sequences.add(str(props['sequence']))
            if props.get('capture_date'):
                dates.append(str(props['capture_date']))
        total = len(points)
        stride = max(1, -(-total // MAX_COVERAGE_POINTS))
        return {
            'count': total,
            'points': points[::stride],
            'sequences': len(sequences),
            'first': min(dates) if dates else None,
            'last': max(dates) if dates else None,
            'truncated': stride > 1,
        }

    def _mapillary_client(self, token: str):
        """A cached MapillaryClient for ``token`` (the RAPID token when empty)."""
        from rapidtools.data_sources import MapillaryClient

        token = (token or '').strip() or self._download_text(MAPILLARY_TOKEN_DATASET)
        state = self.state
        with state.lock:
            client = state.mapillary_clients.get(token)
            if client is None:
                client = state.mapillary_clients[token] = MapillaryClient(
                    token, save_dir=state.output_dir / 'mapillary'
                )
        return client

    def street_sequences(
        self, z: int, x: int, y: int, query: dict[str, Any]
    ) -> dict[str, Any]:
        """
        ``GET /api/street/sequences/<z>/<x>/<y>``: survey routes in a tile.

        Decodes the ``sequence`` layer of Mapillary's coverage tile so the
        map can draw where street-level imagery exists at any zoom, from a
        whole state (zoom 6) to a block (zoom 14). Results are cached per
        tile and filter.

        Args:
            z (int): Tile zoom, 6 to 14.
            x (int): Tile column.
            y (int): Tile row.
            query (dict[str, Any]): Optional ``token``, ``rapid_only``
                (default ``1``), ``start_date`` and ``end_date``.

        Returns:
            dict[str, Any]: See
                :meth:`~rapidtools.data_sources.MapillaryClient.fetch_sequence_lines`.

        Raises:
            ApiError: ``400`` for a zoom outside 6 to 14.
        """
        if not 6 <= z <= 14:
            raise ApiError('Coverage tiles exist for zoom levels 6 to 14.')
        token = str(query.get('token', '') or '').strip()
        rapid_only = str(query.get('rapid_only', '1')).lower() not in ('0', 'false', '')
        start = str(query.get('start_date', '') or '')
        end = str(query.get('end_date', '') or '')
        key = (token, z, x, y, rapid_only, start, end)
        state = self.state
        with state.tile_lock:
            cached = state.sequence_cache.get(key)
            if cached is not None:
                state.sequence_cache.move_to_end(key)
                return cached
        client = self._mapillary_client(token)
        data = client.fetch_sequence_lines(
            z, x, y, filter_rapid_only=rapid_only, start_date=start, end_date=end
        )
        with state.tile_lock:
            state.sequence_cache[key] = data
            while len(state.sequence_cache) > SEQUENCE_CACHE_SIZE:
                state.sequence_cache.popitem(last=False)
        return data

    def street_overview(self, query: dict[str, Any]) -> dict[str, Any]:
        """
        ``GET /api/street/overview``: survey routes across the whole country.

        Reads the zoom-6 coverage tiles over :data:`OVERVIEW_BBOX` once (on a
        background thread, cached on disk for a week under
        ``~/.cache/rapidtools``) and returns the routes as longitude/latitude
        polylines, so the map can show where street-level imagery exists
        from its very first view.

        Args:
            query (dict[str, Any]): Optional ``rapid_only`` (default ``1``).

        Returns:
            dict[str, Any]: ``status`` (``'building'`` or ``'ready'``),
                ``done`` and ``total`` tile counts, ``zoom`` and, when ready,
                ``lines`` as flat ``[lon, lat, lon, lat, ...]`` lists.
        """
        rapid_only = str(query.get('rapid_only', '1')).lower() not in ('0', 'false', '')
        state = self.state
        with state.lock:
            entry = state.overview.get(rapid_only)
            if entry is not None and entry.get('status') == 'error':
                entry = None  # retry a failed build on the next request
            if entry is None:
                entry = state.overview[rapid_only] = {
                    'status': 'building',
                    'done': 0,
                    'total': 0,
                    'lines': [],
                    'error': '',
                }
                threading.Thread(
                    target=self._build_overview,
                    args=(rapid_only,),
                    name='survey-overview',
                    daemon=True,
                ).start()
            out = {k: v for k, v in entry.items() if k != 'lines'}
            out['zoom'] = OVERVIEW_ZOOM
            if entry['status'] == 'ready':
                out['lines'] = entry['lines']
            return out

    def _build_overview(self, rapid_only: bool) -> None:
        """Fill the overview entry from the disk cache or the coverage tiles."""
        from concurrent.futures import ThreadPoolExecutor

        from rapidtools.data_sources import TileUtils

        state = self.state
        entry = state.overview[rapid_only]
        cache = (
            state.cache_dir / f'survey_overview_z{OVERVIEW_ZOOM}_{int(rapid_only)}.json'
        )
        try:
            if (
                cache.is_file()
                and time.time() - cache.stat().st_mtime < OVERVIEW_MAX_AGE_S
            ):
                lines = json.loads(cache.read_text(encoding='utf-8'))
                with state.lock:
                    entry.update(status='ready', lines=lines, done=1, total=1)
                return
        except (OSError, ValueError) as exc:
            logger.debug(f'Survey overview cache unreadable: {exc}')
        try:
            client = self._mapillary_client('')
            tiles = _overview_tiles()
            with state.lock:
                entry['total'] = len(tiles)

            failures: list[str] = []

            def one(tile):
                x, y, z = tile
                data = client.fetch_sequence_lines(
                    z, x, y, filter_rapid_only=rapid_only
                )
                if not data.get('ok', True):
                    failures.append(f'{z}/{x}/{y}')
                out = []
                for line in data['lines']:
                    flat: list[float] = []
                    for i in range(0, len(line), 2):
                        lon, lat = TileUtils.mvt_to_wgs84(
                            line[i], line[i + 1], x, y, z, extent=data['extent']
                        )
                        flat.append(round(lon, 4))
                        flat.append(round(lat, 4))
                    out.append(flat)
                with state.lock:
                    entry['done'] += 1
                return out

            fetched_lines: list[list[float]] = []
            with ThreadPoolExecutor(max_workers=OVERVIEW_WORKERS) as pool:
                for result in pool.map(one, tiles):
                    fetched_lines.extend(result)
            if failures:
                raise RuntimeError(
                    f'{len(failures)} of {len(tiles)} coverage tiles could not be '
                    'read (is the internet connection up?); nothing was cached.'
                )
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(fetched_lines), encoding='utf-8')
            except OSError as exc:
                logger.debug(f'Could not cache the survey overview: {exc}')
            with state.lock:
                entry.update(status='ready', lines=fetched_lines)
        except Exception as exc:  # noqa: BLE001 - reported through the state
            logger.warning(f'Survey overview failed: {exc}')
            with state.lock:
                entry.update(status='error', error=str(exc))

    def street_routes(self, query: dict[str, Any]) -> dict[str, Any]:
        """
        ``GET /api/street/routes``: status of the RAPID route database.

        The database holds every RAPID survey route at zoom-13 detail, built
        once (on a background thread) from Mapillary's coverage tiles and
        kept gzipped as ``survey_routes_z13.json.gz`` under
        ``~/.cache/rapidtools``. It is static until
        :meth:`rebuild_routes` is called, so the file can be copied between
        machines or replaced by a published one. The first call starts the
        build when no file exists.

        Returns:
            dict[str, Any]: ``status`` (``'building'``, ``'ready'`` or
                ``'error'``), ``done`` / ``total`` tiles, ``n_routes``,
                ``built_at``, ``zoom``, ``bytes`` of the gzipped payload and
                ``url`` of the payload (``/api/street/routes.json``).
        """
        state = self.state
        with state.lock:
            if state.routes['status'] == 'error' and (
                time.time() - state.routes.get('failed_at', 0) > 60
            ):
                state.routes['status'] = 'idle'  # retry a failed build after a minute
            if state.routes['status'] == 'idle':
                state.routes.update(status='building', error='')
                threading.Thread(
                    target=self._build_routes, name='survey-routes', daemon=True
                ).start()
            out = dict(state.routes)
            out['zoom'] = ROUTES_ZOOM
            out['bytes'] = len(state.routes_gz) if state.routes_gz else 0
            out['url'] = '/api/street/routes.json'
            return out

    def rebuild_routes(self, _body: dict[str, Any]) -> dict[str, Any]:
        """
        ``POST /api/street/routes/rebuild``: fetch the route database afresh.

        Returns:
            dict[str, Any]: ``{'ok': True}``; poll :meth:`street_routes`.

        Raises:
            ApiError: ``409`` while a build is already running.
        """
        state = self.state
        with state.lock:
            if state.routes['status'] == 'building':
                raise ApiError(
                    'The route database is already being built.', HTTPStatus.CONFLICT
                )
            state.routes.update(status='building', done=0, total=0, error='')
            state.overview.pop(True, None)
        threading.Thread(
            target=self._build_routes, args=(True,), name='survey-routes', daemon=True
        ).start()
        return {'ok': True}

    def routes_payload(self) -> tuple[bytes, str] | None:
        """The gzipped route database and its ETag, or ``None`` until ready."""
        with self.state.lock:
            if self.state.routes['status'] != 'ready' or not self.state.routes_gz:
                return None
            return self.state.routes_gz, f'"{self.state.routes["built_at"]}"'

    def _build_routes(self, force: bool = False) -> None:
        """Load the route database from disk or build it from coverage tiles."""
        from concurrent.futures import ThreadPoolExecutor

        from rapidtools.data_sources import TileUtils

        state = self.state
        entry = state.routes
        cache = state.cache_dir / ROUTES_FILE
        if not force and cache.is_file():
            try:
                data = cache.read_bytes()
                meta = json.loads(gzip.decompress(data))
                with state.lock:
                    state.routes_gz = data
                    entry.update(
                        status='ready',
                        done=1,
                        total=1,
                        n_routes=int(meta.get('n_routes', 0)),
                        built_at=str(meta.get('built_at', '')),
                    )
                return
            except (OSError, ValueError) as exc:
                logger.warning(f'Route database unreadable, rebuilding: {exc}')
        try:
            overview = state.overview.get(True)
            if overview is None or overview.get('status') != 'ready' or force:
                with state.lock:
                    state.overview[True] = {
                        'status': 'building',
                        'done': 0,
                        'total': 0,
                        'lines': [],
                        'error': '',
                    }
                if force:
                    stale = state.cache_dir / f'survey_overview_z{OVERVIEW_ZOOM}_1.json'
                    stale.unlink(missing_ok=True)
                self._build_overview(True)
                overview = state.overview[True]
            if overview.get('status') != 'ready':
                raise RuntimeError(
                    overview.get('error') or 'the survey overview failed'
                )
            tiles = _routes_tiles(overview['lines'])
            with state.lock:
                entry.update(total=len(tiles), done=0)
            client = self._mapillary_client('')

            failures: list[str] = []

            def one(tile):
                x, y, z = tile
                data = client.fetch_sequence_lines(z, x, y, filter_rapid_only=True)
                if not data.get('ok', True):
                    failures.append(f'{z}/{x}/{y}')
                out = []
                for line in data['lines']:
                    flat: list[float] = []
                    for i in range(0, len(line), 2):
                        lon, lat = TileUtils.mvt_to_wgs84(
                            line[i], line[i + 1], x, y, z, extent=data['extent']
                        )
                        flat.append(round(lon, 5))
                        flat.append(round(lat, 5))
                    out.append(flat)
                with state.lock:
                    entry['done'] += 1
                return out, data.get('first')

            lines: list[list[float]] = []
            dates: list[str | None] = []
            with ThreadPoolExecutor(max_workers=OVERVIEW_WORKERS) as pool:
                for result, first in pool.map(one, tiles):
                    lines.extend(result)
                    dates.extend([first] * len(result))
            if failures:
                raise RuntimeError(
                    f'{len(failures)} of {len(tiles)} coverage tiles could not be read '
                    '(is the internet connection up?); nothing was cached. Use '
                    'refresh to try again.'
                )
            built_at = _dt.datetime.now(_dt.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
            payload = {
                'zoom': ROUTES_ZOOM,
                'built_at': built_at,
                'n_routes': len(lines),
                'n_tiles': len(tiles),
                'lines': lines,
                'dates': dates,
            }
            data = gzip.compress(
                json.dumps(payload, separators=(',', ':')).encode('utf-8'),
                compresslevel=6,
            )
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(data)
            except OSError as exc:
                logger.debug(f'Could not cache the route database: {exc}')
            with state.lock:
                state.routes_gz = data
                entry.update(status='ready', n_routes=len(lines), built_at=built_at)
            logger.info(
                f'Survey route database ready: {len(lines):,} routes from {len(tiles)} '
                f'tiles ({len(data) / 1e6:.1f} MB gzipped).'
            )
        except Exception as exc:  # noqa: BLE001 - reported through the state
            logger.warning(f'Survey route database failed: {exc}')
            with state.lock:
                entry.update(status='error', error=str(exc), failed_at=time.time())

    def geocode(self, query: dict[str, Any]) -> dict[str, Any]:
        """
        ``GET /api/geocode?q=...``: find a place by address or name.

        Args:
            query (dict[str, Any]): ``q``, the text to look up.

        Returns:
            dict[str, Any]: ``{'matches': [...]}`` as returned by
                :func:`_geocode`.

        Raises:
            ApiError: If ``q`` is empty or the lookup fails.
        """
        text = str(query.get('q', '') or '').strip()
        if not text:
            raise ApiError('Type an address or place name.')
        try:
            return {'matches': _geocode(text)}
        except Exception as exc:  # noqa: BLE001 - surfaced to the browser
            raise ApiError(
                f'Address lookup failed: {exc}', HTTPStatus.BAD_GATEWAY
            ) from exc

    def geo_overlay(self, _query: dict[str, Any]) -> dict[str, Any]:
        """
        ``GET /api/geo_overlay``: the current assets as WGS84 geometries.

        Unlike :meth:`overlay`, which projects assets onto a raster preview,
        this returns longitude/latitude coordinates for drawing on the map.

        Returns:
            dict[str, Any]: ``collection_version``, ``count`` and ``features``
                with ``id``, ``kind`` (``'point'``, ``'line'`` or
                ``'polygon'``), ``coords`` (a ``[lon, lat]`` for points, a
                list of rings/lines otherwise), ``bbox`` and ``attributes``
                (without the bulky observation and image records). At most
                :data:`MAX_OVERLAY_FEATURES` features are returned.
        """
        state = self.state
        with state.lock:
            collection = state.collection
            version = state.collection_version
        features: list[dict[str, Any]] = []
        if collection is not None:
            for asset in collection:
                if len(features) >= MAX_OVERLAY_FEATURES:
                    break
                geom = asset.geometry
                if geom is None or geom.is_empty:
                    continue
                kind, coords = _geometry_coords(geom)
                if kind is None:
                    continue
                attributes = {
                    str(k): v
                    for k, v in asset.attributes.items()
                    if k not in ('observations', 'image_assets')
                    and isinstance(v, (str, int, float, bool))
                    or v is None
                }
                features.append(
                    {
                        'id': asset.id,
                        'kind': kind,
                        'coords': coords,
                        'bbox': [round(v, 6) for v in geom.bounds],
                        'attributes': attributes,
                    }
                )
        return {
            'collection_version': version,
            'count': len(collection) if collection is not None else 0,
            'features': features,
        }

    def mapillary_token(self, _body: dict[str, Any]) -> dict[str, Any]:
        """
        ``POST /api/mapillary_token``: fetch the token used by the examples.

        Args:
            _body (dict[str, Any]): Ignored.

        Returns:
            dict[str, Any]: ``{'token': <Mapillary access token>}``.
        """
        return {'token': self._download_text(MAPILLARY_TOKEN_DATASET)}

    def _download_text(self, dataset: str) -> str:
        """Download a registry text file into ``output_dir`` and return it."""
        from rapidtools import download_dataset

        self.state.output_dir.mkdir(parents=True, exist_ok=True)
        [path] = download_dataset(dataset, output_dir=self.state.output_dir)
        return Path(path).read_text(encoding='utf-8').strip()

    # ------------------------------------------------------------ prompts
    def assemble_prompt(self, body: dict[str, Any]) -> dict[str, Any]:
        """
        ``POST /api/prompt/assemble``: render a prompt specification as text.

        Args:
            body (dict[str, Any]): ``spec`` (a
                :class:`~rapidtools.gui.prompt_builder.PromptSpec` object),
                optional ``outline`` (``shape``, ``color``, ``overlay``) used
                to fill an empty ``marking``, and optional ``backend`` whose
                attribute prefix is used to name the resulting attributes.

        Returns:
            dict[str, Any]: ``text`` (the prompt), ``marking``, ``attributes``
            (the asset attribute names the analyzer will write) and the
            normalised ``spec``.

        Raises:
            ApiError: If the specification is invalid.
        """
        try:
            spec = PromptSpec.from_dict(body.get('spec') or {})
        except (TypeError, ValueError) as exc:
            raise ApiError(f'Invalid prompt specification: {exc}') from exc
        outline = body.get('outline') or {}
        if not spec.marking and outline:
            spec.marking = describe_marking(
                str(outline.get('shape') or 'geometry'),
                str(outline.get('color') or 'red'),
                bool(outline.get('overlay', True)),
            )
        backend = MODEL_BACKENDS.get(str(body.get('backend') or ''), {})
        prefix = backend.get('attribute_prefix') or 'vlm'
        return {
            'text': assemble_prompt(spec),
            'marking': spec.marking,
            'attributes': spec.attribute_keys(prefix),
            'spec': spec.to_dict(),
        }

    def prompt_example(self, _body: dict[str, Any]) -> dict[str, Any]:
        """
        ``POST /api/prompt/example``: the CHS sample as a specification.

        Returns:
            dict[str, Any]: ``{'spec': <PromptSpec dict>}``.
        """
        return {'spec': SAMPLE_SPEC.to_dict()}

    def prompt_assist(self, body: dict[str, Any]) -> dict[str, Any]:
        """
        ``POST /api/prompt/assist``: ask a model to help with the prompt.

        The request runs as a background job; the browser reads the outcome
        from ``assistant`` in ``GET /api/state`` once its ``seq`` changes.

        Args:
            body (dict[str, Any]): ``action`` (one of
                :data:`~rapidtools.gui.prompt_builder.ASSIST_ACTIONS`),
                ``backend``, ``api_key``, ``model_id``, ``load_in_4bit`` and
                the action inputs ``brief``, ``instruction``, ``prompt_text``,
                ``class_value``, ``spec`` and ``context``.

        Returns:
            dict[str, Any]: ``{'ok': True, 'seq': <request number>}``.

        Raises:
            ApiError: If the settings are invalid or a job is running.
        """
        try:
            settings = AssistSettings(
                action=str(body.get('action', '')),
                backend=str(body.get('backend') or 'gemma4'),  # type: ignore[arg-type]
                api_key=str(body.get('api_key', '') or ''),
                model_id=str(body.get('model_id', '') or ''),
                load_in_4bit=bool(body.get('load_in_4bit', True)),
                brief=str(body.get('brief', '') or ''),
                instruction=str(body.get('instruction', '') or ''),
                prompt_text=str(body.get('prompt_text', '') or ''),
                class_value=str(body.get('class_value', '') or ''),
                spec=body.get('spec') or None,
                context=dict(body.get('context') or {}),
            )
        except (TypeError, ValueError) as exc:
            raise ApiError(str(exc)) from exc
        state = self.state
        state.set_assistant(
            seq=True,
            status='running',
            action=settings.action,
            model=f'{settings.backend_spec["label"]} · {settings.model_id}',
            result=None,
            error='',
        )

        def job() -> None:
            """Run the assistant and record its answer (or failure)."""
            try:
                result = state.workflow.assist(settings)
            except Exception as exc:
                state.set_assistant(status='error', error=str(exc))
                raise
            state.set_assistant(status='done', result=result)

        try:
            state.start_job('Prompt assistant', job)
        except ApiError:
            state.set_assistant(status='error', error='Another job is running.')
            raise
        with state.lock:
            seq = state.assistant['seq']
        return {'ok': True, 'seq': seq}

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


def _geometry_coords(geom: Any) -> tuple[str | None, Any]:
    """
    Reduce a shapely geometry to a drawable ``(kind, coords)`` pair.

    Example:
        >>> from shapely.geometry import Point, box
        >>> _geometry_coords(Point(1, 2))
        ('point', [1.0, 2.0])
        >>> _geometry_coords(box(0, 0, 1, 1))[0]
        'polygon'
    """
    kind = geom.geom_type
    rnd = lambda xy: [round(float(xy[0]), 6), round(float(xy[1]), 6)]  # noqa: E731
    if kind == 'Point':
        return 'point', rnd((geom.x, geom.y))
    if kind == 'LineString':
        return 'line', [[rnd(c) for c in geom.coords]]
    if kind == 'MultiLineString':
        return 'line', [[rnd(c) for c in part.coords] for part in geom.geoms]
    if kind == 'Polygon':
        return 'polygon', [[rnd(c) for c in geom.exterior.coords]]
    if kind == 'MultiPolygon':
        return 'polygon', [
            [rnd(c) for c in part.exterior.coords] for part in geom.geoms
        ]
    if kind in ('MultiPoint', 'GeometryCollection'):
        c = geom.centroid
        return 'point', rnd((c.x, c.y))
    return None, None


def _optional_bbox(body: dict[str, Any]):
    """
    Read ``min_lon``/``min_lat``/``max_lon``/``max_lat`` as a BoundingBox.

    Returns ``None`` when all four are blank.

    Raises:
        ApiError: If some but not all are given, or the box is invalid.

    Example:
        >>> _optional_bbox({}) is None
        True
        >>> box = {'min_lon': -1, 'min_lat': 0, 'max_lon': 1, 'max_lat': 2}
        >>> _optional_bbox(box).bounds
        (-1.0, 0.0, 1.0, 2.0)
    """
    keys = ('min_lon', 'min_lat', 'max_lon', 'max_lat')
    try:
        values = [_optional_float(body.get(k)) for k in keys]
    except ValueError as exc:
        raise ApiError('Longitudes and latitudes must be numbers.') from exc
    if all(v is None for v in values):
        return None
    min_lon, min_lat, max_lon, max_lat = values
    if min_lon is None or min_lat is None or max_lon is None or max_lat is None:
        raise ApiError('Give all four of min/max longitude and latitude.')
    if not (-180 <= min_lon < max_lon <= 180 and -85 <= min_lat < max_lat <= 85):
        raise ApiError(
            'The box must satisfy min < max with longitudes in [-180, 180] and '
            'latitudes in [-85, 85].'
        )
    from rapidtools.core import BoundingBox

    return BoundingBox(min_x=min_lon, min_y=min_lat, max_x=max_lon, max_y=max_lat)


def _optional_float(value: Any) -> float | None:
    """
    Coerce a request value to ``float``, treating blanks as ``None``.

    Raises:
        ValueError: If the value is neither blank nor numeric.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return float(value)


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
        if not presented:
            return False
        return secrets.compare_digest(presented, token)

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
            elif path == '/api/street/coverage':
                self._send_json(api.street_coverage(query))
            elif path == '/api/geocode':
                self._send_json(api.geocode(query))
            elif path == '/api/street/overview':
                self._send_json(api.street_overview(query))
            elif path == '/api/street/routes':
                self._send_json(api.street_routes(query))
            elif path == '/api/street/routes.json':
                payload = api.routes_payload()
                if payload is None:
                    raise ApiError(
                        'The route database is not ready.', HTTPStatus.NOT_FOUND
                    )
                data, etag = payload
                if self.headers.get('If-None-Match') == etag:
                    self.send_response(HTTPStatus.NOT_MODIFIED)
                    self.send_header('ETag', etag)
                    self.end_headers()
                    return
                headers = {'Cache-Control': 'no-cache', 'ETag': etag}
                if 'gzip' in (self.headers.get('Accept-Encoding') or ''):
                    headers['Content-Encoding'] = 'gzip'
                else:
                    data = gzip.decompress(data)
                self._send_bytes(data, 'application/json', extra_headers=headers)
            elif path == '/api/geo_overlay':
                self._send_json(api.geo_overlay(query))
            elif path.startswith('/api/street/sequences/'):
                parts_ = path[len('/api/street/sequences/') :].split('/')
                if len(parts_) != 3:
                    raise ApiError('Not found', HTTPStatus.NOT_FOUND)
                try:
                    z, x, y = (int(v) for v in parts_)
                except ValueError as exc:
                    raise ApiError('Bad tile address') from exc
                self._send_json(api.street_sequences(z, x, y, query))
            elif path.startswith('/api/basemap_tile/'):
                parts_ = (
                    path[len('/api/basemap_tile/') :].removesuffix('.jpg').split('/')
                )
                if len(parts_) != 4:
                    raise ApiError('Not found', HTTPStatus.NOT_FOUND)
                try:
                    data = api.basemap_tile(parts_[0], *(int(v) for v in parts_[1:]))
                except ValueError as exc:
                    raise ApiError('Bad tile address') from exc
                self._send_bytes(
                    data,
                    'image/jpeg',
                    extra_headers={'Cache-Control': 'max-age=86400'},
                )
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

        def cancel(_body: dict[str, Any]) -> dict[str, Any]:
            state.cancel_job()
            return {'ok': True}

        def reset(_body: dict[str, Any]) -> dict[str, Any]:
            state.reset()
            return {'ok': True}

        routes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            '/api/imagery/local': api.load_local_raster,
            '/api/imagery/sample': api.download_sample_raster,
            '/api/imagery/region': api.download_region_imagery,
            '/api/output_dir': api.set_output_dir,
            '/api/assets/load': api.load_assets,
            '/api/detect': api.detect,
            '/api/street/discover': api.discover_street,
            '/api/infer': api.infer,
            '/api/run_all': api.run_all,
            '/api/prompt/sample': api.sample_prompt,
            '/api/prompt/assemble': api.assemble_prompt,
            '/api/prompt/example': api.prompt_example,
            '/api/prompt/assist': api.prompt_assist,
            '/api/mapillary_token': api.mapillary_token,
            '/api/notify': api.set_notification,
            '/api/street/routes/rebuild': api.rebuild_routes,
            '/api/models': api.list_models,
            '/api/cancel': cancel,
            '/api/reset': reset,
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
        notification: NotificationConfig | None = None,
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
            notification (NotificationConfig | None): SMTP relay and public
                link for job notifications; read from the environment when
                omitted.

        Raises:
            OSError: If the address cannot be bound.
        """
        self.state = state
        if notification is not None:
            state.notifier = Notifier(notification)
        self.token = token or None
        self.api = Api(state, data_root=data_root)
        state.data_root = self.api.data_root
        state.token_required = self.token is not None
        self.verbose = verbose
        self._log_handler: CallbackLogHandler | None = None
        super().__init__(address, GuiRequestHandler)
        state.public_url = self.share_urls()[-1] if self.token else self.url
        self._log_handler = CallbackLogHandler(state.log.add_line)
        logging.getLogger().addHandler(self._log_handler)

    def server_close(self) -> None:
        """Detach the GUI log handler and release the listening socket."""
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None
        super().server_close()

    def _host_port(self) -> tuple[str, int]:
        """Return the bound ``(host, port)`` with the host as text."""
        host, port = self.server_address[:2]
        if isinstance(host, bytes):
            host = host.decode()
        return str(host), int(port)

    @property
    def url(self) -> str:
        """
        Return the URL to open locally.

        Returns:
            str: ``http://localhost:<port>/`` for loopback / wildcard binds,
            otherwise the bound host.
        """
        host, port = self._host_port()
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
        host, port = self._host_port()
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
    notification: NotificationConfig | None = None,
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
        notification: SMTP relay and public link for job notifications (see
            :mod:`rapidtools.gui.notify`); the environment is read when
            omitted.

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
        notification=notification,
    )


def launch_asset_analysis_app(
    host: str = '127.0.0.1',
    port: int = 8765,
    output_dir: str | Path | None = None,
    open_browser: bool = True,
    token: str | None = None,
    data_root: str | Path | None = None,
    notification: NotificationConfig | None = None,
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
        >>> launch_asset_analysis_app(port=9000, open_browser=False)  # doctest: +SKIP
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
            notification=notification,
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
    if server.state.notifier.email_available:
        logger.info(
            f'Email notifications go through {server.state.notifier.config.smtp_host}'
        )
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
        >>> main(['--port', '9000', '--no-browser'])  # doctest: +SKIP
        0
    """
    parser = argparse.ArgumentParser(
        prog='rapidtools-gui',
        description=(
            'Detect assets in aerial or street-level imagery and run VLM '
            'inference on them.'
        ),
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
    notify = parser.add_argument_group(
        'notifications',
        'email users when a long job finishes (webhooks need no setup); '
        'the password is read from RAPIDTOOLS_SMTP_PASSWORD',
    )
    notify.add_argument(
        '--smtp-host', default=None, help='SMTP relay host (or RAPIDTOOLS_SMTP_HOST)'
    )
    notify.add_argument(
        '--smtp-port',
        type=int,
        default=None,
        help='SMTP port (587, or 465 with --smtp-ssl)',
    )
    notify.add_argument(
        '--smtp-user', default=None, help='SMTP login (or RAPIDTOOLS_SMTP_USER)'
    )
    notify.add_argument(
        '--smtp-from', default=None, help='sender address (or RAPIDTOOLS_SMTP_FROM)'
    )
    notify.add_argument(
        '--smtp-ssl', action='store_true', help='use implicit TLS instead of STARTTLS'
    )
    notify.add_argument(
        '--public-url',
        default=None,
        help='link put in notifications when the server sits behind a proxy or tunnel',
    )
    args = parser.parse_args(argv)
    notification = NotificationConfig.from_env(
        smtp_host=args.smtp_host,
        smtp_port=args.smtp_port,
        smtp_user=args.smtp_user,
        smtp_from=args.smtp_from,
        smtp_ssl=args.smtp_ssl,
        public_url=args.public_url,
    )
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
        notification=notification,
    )
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
