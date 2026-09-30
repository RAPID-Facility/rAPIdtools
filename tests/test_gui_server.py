"""Tests for the rapidtools GUI preview renderer and HTTP API."""

import json
import logging
import runpy
import shutil
import threading
import time
from http.client import HTTPConnection
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

import rapidtools
import rapidtools.gui
from rapidtools.core import ImageAsset, PhysicalAsset, PhysicalAssetCollection
from rapidtools.gui import server as server_module
from rapidtools.gui.preview import project_collection, render_preview
from rapidtools.gui.workflow import DetectionResult, InferenceResult

# A tiny UTM-projected raster covering a known WGS84 area.
UTM_CRS = 'EPSG:32611'
ORIGIN_X, ORIGIN_Y = 380_000.0, 3_780_000.0  # metres, zone 11N
PIXEL_SIZE = 10.0
WIDTH, HEIGHT = 40, 20


@pytest.fixture(scope='module')
def raster_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp('raster') / 'scene.tif'
    data = np.zeros((4, HEIGHT, WIDTH), dtype=np.uint8)
    data[0] = 200
    data[1] = 120
    data[2] = 60
    data[3] = 255
    data[3, :, :5] = 0  # transparent strip on the left
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        width=WIDTH,
        height=HEIGHT,
        count=4,
        dtype='uint8',
        crs=UTM_CRS,
        transform=from_origin(ORIGIN_X, ORIGIN_Y, PIXEL_SIZE, PIXEL_SIZE),
    ) as dst:
        dst.write(data)
    return path


def _asset_collection_covering_pixel(col: int, row: int) -> PhysicalAssetCollection:
    """Build a WGS84 collection with one square asset around a raster pixel."""
    from pyproj import Transformer

    to_wgs = Transformer.from_crs(UTM_CRS, 'EPSG:4326', always_xy=True)
    x0 = ORIGIN_X + col * PIXEL_SIZE
    y0 = ORIGIN_Y - row * PIXEL_SIZE
    lon0, lat0 = to_wgs.transform(x0, y0)
    lon1, lat1 = to_wgs.transform(x0 + 2 * PIXEL_SIZE, y0 - 2 * PIXEL_SIZE)
    collection = PhysicalAssetCollection()
    collection.add(
        PhysicalAsset(
            id='a1',
            geometry=box(lon0, lat1, lon1, lat0),
            attributes={'asset_type': 'building', 'damage': 'Major'},
        )
    )
    return collection


# ------------------------------------------------------------------ preview
def test_render_preview_and_projection(raster_path):
    preview = render_preview(raster_path, max_px=20)
    assert (preview.preview_width, preview.preview_height) == (20, 10)
    assert preview.png.startswith(b'\x89PNG')
    assert preview.width == WIDTH and preview.height == HEIGHT
    lon_min, lat_min, lon_max, lat_max = preview.wgs84_bounds
    assert -119 < lon_min < lon_max < -116
    assert 33 < lat_min < lat_max < 35

    data = project_collection(_asset_collection_covering_pixel(10, 4), preview)
    assert data['count'] == 1
    assert data['attribute_keys'] == ['asset_type', 'damage']
    feature = data['features'][0]
    assert feature['kind'] == 'polygon'
    xs = [p[0] for p in feature['rings'][0]]
    ys = [p[1] for p in feature['rings'][0]]
    # Pixel (10, 4) at half resolution -> preview x 5..6, y 2..3
    assert min(xs) == pytest.approx(5, abs=0.05)
    assert max(xs) == pytest.approx(6, abs=0.05)
    assert min(ys) == pytest.approx(2, abs=0.05)
    assert max(ys) == pytest.approx(3, abs=0.05)


def test_raster_tiler(raster_path):
    import io

    from PIL import Image

    from rapidtools.gui.preview import RasterTiler

    tiler = RasterTiler(raster_path, max_cache=2)
    data = tiler.tile(1, 0, 0)
    image = Image.open(io.BytesIO(data))
    assert image.size == (WIDTH, HEIGHT)  # raster is smaller than one tile
    pixels = image.convert('RGB')
    assert pixels.getpixel((0, 0)) == (0, 0, 0)  # transparent strip masked
    r, g, b = pixels.getpixel((WIDTH - 1, HEIGHT - 1))
    assert (r, g, b) == pytest.approx((200, 120, 60), abs=4)
    # Downsampled tile halves the size; off-raster tiles return None:
    half = Image.open(io.BytesIO(tiler.tile(2, 0, 0)))
    assert half.size == (WIDTH // 2, HEIGHT // 2)
    assert tiler.tile(1, 5, 0) is None
    assert tiler.tile(3, 0, 0) is None  # unsupported downsample
    # Cache is bounded and hits return identical bytes:
    tiler.tile(4, 0, 0)
    assert len(tiler._cache) <= 2
    assert tiler.tile(4, 0, 0) == tiler.tile(4, 0, 0)


# ------------------------------------------------------------------- server
class FakeWorkflow:
    """Stand-in for AssetAnalysisWorkflow that records calls."""

    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
        self.detect_calls = []
        self.analyze_calls = []
        self.cancelled = False
        self.block = threading.Event()
        self.block.set()

    def cancel(self):
        self.cancelled = True

    def detect(self, settings):
        self.detect_calls.append(settings)
        self.block.wait(timeout=5)
        if self.progress_callback:
            self.progress_callback('fake detection running')
        collection = _asset_collection_covering_pixel(10, 4)
        combined = settings.output_dir / 'assets_final.geojson'
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        combined.write_text('{}')
        return DetectionResult(
            collection=collection,
            detection_raster=settings.raster_path.resolve(),
            per_asset_geojson={settings.assets[0]: combined},
            combined_geojson=combined,
        )

    def analyze(self, collection, settings):
        self.analyze_calls.append(settings)
        for asset in collection:
            asset.add_attributes({'chs_level': '3'})
        path = settings.output_dir / 'assets_inferred.geojson'
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
        return InferenceResult(
            collection=collection,
            geojson_path=path,
            n_input=len(collection),
            n_analyzed=len(collection),
        )


class Client:
    def __init__(self, port):
        self.port = port
        self.headers = {}

    def request(self, method, path, body=None, headers=None):
        conn = HTTPConnection('127.0.0.1', self.port, timeout=10)
        payload = json.dumps(body).encode() if body is not None else None
        hdrs = dict(self.headers)
        if payload:
            hdrs['Content-Type'] = 'application/json'
        hdrs.update(headers or {})
        conn.request(method, path, body=payload, headers=hdrs)
        response = conn.getresponse()
        raw = response.read()
        self.last_response = response
        conn.close()
        return response.status, response.getheader('Content-Type'), raw

    def json(self, method, path, body=None, expect=200, headers=None):
        status, _ctype, raw = self.request(method, path, body, headers=headers)
        data = json.loads(raw)
        assert status == expect, (status, data)
        return data

    def wait_idle(self, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = self.json('GET', '/api/state')
            if not state['busy']:
                return state
            time.sleep(0.05)
        raise AssertionError('server stayed busy')


@pytest.fixture
def gui(tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, 'AssetAnalysisWorkflow', FakeWorkflow)
    srv = server_module.create_server(port=0, output_dir=tmp_path / 'out')
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    client = Client(srv.server_address[1])
    try:
        yield client, srv
    finally:
        srv.shutdown()
        srv.server_close()


def test_index_and_state(gui):
    client, srv = gui
    status, ctype, raw = client.request('GET', '/')
    assert status == 200 and 'text/html' in ctype
    assert b'rAPIdtools' in raw and b'Detect + run inference' in raw

    state = client.json('GET', '/api/state')
    assert state['raster'] is None
    assert state['collection'] is None
    assert state['job']['status'] == 'idle'
    assert 'eaton_patch1' in state['options']['sample_rasters']
    assert srv.url.startswith('http://localhost:')


def test_unknown_routes_and_bad_json(gui):
    client, _ = gui
    client.json('GET', '/api/nope', expect=404)
    client.json('POST', '/api/nope', {}, expect=404)
    status, _, raw = client.request('POST', '/api/detect', None)
    assert status == 400  # no raster loaded yet
    assert b'Load an aerial image' in raw or b'Enter at least one' in raw


def test_load_raster_detect_infer_flow(gui, raster_path):
    client, srv = gui
    workflow = srv.state.workflow

    # Missing raster is rejected up front:
    client.json('POST', '/api/imagery/local', {'path': '/nope.tif'}, expect=404)

    client.json('POST', '/api/imagery/local', {'path': str(raster_path)})
    state = client.wait_idle()
    assert state['job']['status'] == 'done'
    assert state['raster']['name'] == 'scene.tif'
    assert state['raster']['preview_width'] == WIDTH
    status, ctype, png = client.request('GET', '/api/preview/raster.png')
    assert status == 200 and ctype == 'image/png' and png.startswith(b'\x89PNG')
    client.json('GET', '/api/preview/basemap.png', expect=404)

    # Detection with advanced options:
    client.json(
        'POST',
        '/api/detect',
        {
            'assets': 'building, tree',
            'asset_size_m': 30,
            'source': 'recon',
            'threshold': 0.6,
            'mask_threshold': 0.3,
            'regularize': False,
        },
    )
    state = client.wait_idle()
    assert state['job']['status'] == 'done'
    assert state['status'] == 'Detection finished.'
    settings = workflow.detect_calls[0]
    assert settings.assets == ['building', 'tree']
    assert settings.asset_size_m == 30
    assert settings.detect_in_recon_imagery is True
    assert settings.threshold == 0.6 and settings.mask_threshold == 0.3
    assert settings.regularize is False
    assert state['collection'] == {
        'count': 1,
        'source': 'detected',
        'version': state['collection']['version'],
        'attribute_keys': ['asset_type', 'damage'],
    }
    assert state['basemap'] is None  # detection ran on the recon raster

    overlay = client.json('GET', '/api/overlay?preview=raster')
    assert overlay['count'] == 1
    assert overlay['features'][0]['attributes']['asset_type'] == 'building'
    assert overlay['collection_version'] == state['collection']['version']

    # Inference:
    client.json('POST', '/api/infer', {'prompt': 'x', 'backend': 'gemini'}, expect=400)
    # Invalid outline options are rejected before the job starts:
    client.json(
        'POST',
        '/api/infer',
        {'prompt': 'x', 'backend': 'gemini', 'api_key': 'k', 'outline_shape': 'oval'},
        expect=400,
    )
    client.json(
        'POST',
        '/api/infer',
        {
            'prompt': 'Rate the damage.',
            'backend': 'gemini',
            'api_key': 'k',
            'max_workers': 2,
            'outline_shape': 'rotated_bbox',
            'outline_buffer': '2 m',
            'outline_width': '',  # blank fields fall back to the defaults
            'outline_color': 'yellow',
        },
    )
    state = client.wait_idle()
    assert state['job']['status'] == 'done'
    inference = workflow.analyze_calls[0]
    assert inference.prompt == 'Rate the damage.'
    assert inference.max_workers == 2
    assert inference.outline_shape == 'rotated_bbox'
    assert inference.outline_buffer == '2 m'
    assert inference.outline_width == 6
    assert inference.outline_color == 'yellow'
    assert inference.model_id == server_module.DEFAULT_GEMINI_MODEL_ID
    assert state['collection']['source'] == 'analyzed'
    assert 'chs_level' in state['collection']['attribute_keys']

    table = client.json('GET', '/api/table')
    assert table['total'] == 1
    assert table['rows'][0]['attributes']['chs_level'] == '3'
    assert 'chs_level' in table['columns']

    labels = {r['label'] for r in state['results']}
    assert {'All detected assets', 'Inference results'} <= labels
    inferred = next(r for r in state['results'] if r['label'] == 'Inference results')
    status, _, raw = client.request('GET', '/api/download?path=' + inferred['path'])
    assert status == 200 and raw == b'{}'
    # Only result files can be downloaded:
    client.json('GET', '/api/download?path=' + str(raster_path), expect=403)

    log = client.json('GET', '/api/log?since=0')
    texts = [e['text'] for e in log['entries']]
    assert any('--- Detection ---' in t for t in texts)
    assert any('Inference finished.' in t for t in texts)

    # Reset clears everything:
    client.json('POST', '/api/reset', {})
    state = client.json('GET', '/api/state')
    assert state['raster'] is None and state['collection'] is None


def test_run_all_busy_and_cancel(gui, raster_path):
    client, srv = gui
    workflow = srv.state.workflow
    client.json('POST', '/api/imagery/local', {'path': str(raster_path)})
    client.wait_idle()

    workflow.block.clear()  # make detection hang until released
    client.json(
        'POST',
        '/api/run_all',
        {
            'detect': {'assets': 'building'},
            'infer': {
                'prompt': 'p',
                'backend': 'gemma4',
                'model_id': 'google/gemma-4-E4B-it',
            },
        },
    )
    state = client.json('GET', '/api/state')
    assert state['busy'] is True and state['job']['name'] == 'Detection + inference'
    # A second job is refused while one is running:
    client.json('POST', '/api/detect', {'assets': 'tree'}, expect=409)
    client.json('POST', '/api/cancel', {})
    assert workflow.cancelled is True
    workflow.block.set()
    state = client.wait_idle()
    assert state['job']['status'] == 'done'
    assert workflow.analyze_calls[0].backend == 'gemma4'
    assert workflow.analyze_calls[0].model_id == 'google/gemma-4-E4B-it'


def test_cancelled_inference_keeps_partial_collection(gui, raster_path, tmp_path):
    from rapidtools.core import OperationCancelled

    client, srv = gui
    workflow = srv.state.workflow
    client.json('POST', '/api/imagery/local', {'path': str(raster_path)})
    client.wait_idle()
    geojson = tmp_path / 'a.geojson'
    _asset_collection_covering_pixel(2, 2).to_geojson(geojson)
    client.json('POST', '/api/assets/load', {'path': str(geojson)})

    def cancelling_analyze(collection, settings):
        for asset in collection:
            asset.add_attributes({'chs_level': '1'})
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        (settings.output_dir / 'assets_inferred_partial.geojson').write_text('{}')
        raise OperationCancelled('user pressed cancel')

    workflow.analyze = cancelling_analyze
    client.json(
        'POST', '/api/infer', {'prompt': 'p', 'backend': 'gemini', 'api_key': 'k'}
    )
    state = client.wait_idle()
    assert state['job']['status'] == 'cancelled'
    assert state['collection']['source'] == 'partially analyzed'
    assert 'chs_level' in state['collection']['attribute_keys']
    assert any(r['label'] == 'Partial inference results' for r in state['results'])


def test_load_assets_from_geojson(gui, raster_path, tmp_path):
    client, _ = gui
    client.json('POST', '/api/imagery/local', {'path': str(raster_path)})
    client.wait_idle()
    geojson = tmp_path / 'assets.geojson'
    _asset_collection_covering_pixel(2, 2).to_geojson(geojson)
    client.json(
        'POST',
        '/api/assets/load',
        {'path': str(tmp_path / 'missing.geojson')},
        expect=404,
    )
    result = client.json('POST', '/api/assets/load', {'path': str(geojson)})
    assert result['count'] == 1
    state = client.json('GET', '/api/state')
    assert state['collection']['source'] == 'loaded from assets.geojson'
    assert client.json('GET', '/api/overlay')['count'] == 1


def test_fs_listing_and_text_files(gui, tmp_path, raster_path):
    client, _ = gui
    (tmp_path / 'notes.txt').write_text('hello prompt')
    (tmp_path / 'sub').mkdir()
    (tmp_path / '.hidden').mkdir()
    listing = client.json('GET', f'/api/fs?kind=text&path={tmp_path}')
    assert 'sub' in [d['name'] for d in listing['dirs']]
    assert '.hidden' not in [d['name'] for d in listing['dirs']]
    assert [f['name'] for f in listing['files']] == ['notes.txt']
    assert listing['parent'] == str(tmp_path.parent)

    rasters = client.json('GET', f'/api/fs?kind=raster&path={raster_path.parent}')
    assert [f['name'] for f in rasters['files']] == ['scene.tif']
    dirs_only = client.json('GET', f'/api/fs?kind=dir&path={tmp_path}')
    assert dirs_only['files'] == []
    client.json('GET', '/api/fs?kind=bogus', expect=400)
    client.json('GET', '/api/fs?kind=text&path=/definitely/missing', expect=404)

    text = client.json('GET', f'/api/textfile?path={tmp_path / "notes.txt"}')
    assert text['text'] == 'hello prompt'

    out = client.json('POST', '/api/output_dir', {'path': str(tmp_path / 'elsewhere')})
    assert out['output_dir'].endswith('elsewhere')
    client.json(
        'POST', '/api/output_dir', {'path': str(tmp_path / 'notes.txt')}, expect=400
    )


def test_tiles_models_and_asset_details(gui, raster_path, tmp_path, monkeypatch):
    client, srv = gui
    client.json('GET', '/api/tile/raster/1/0/0.jpg', expect=404)  # nothing loaded
    client.json('POST', '/api/imagery/local', {'path': str(raster_path)})
    client.wait_idle()

    status, ctype, raw = client.request('GET', '/api/tile/raster/1/0/0.jpg')
    assert status == 200 and ctype == 'image/jpeg' and raw[:2] == b'\xff\xd8'
    client.json('GET', '/api/tile/raster/1/9/9.jpg', expect=404)
    client.json('GET', '/api/tile/raster/x/0/0.jpg', expect=400)
    client.json('GET', '/api/tile/raster/1/0.jpg', expect=404)

    # Backend catalogue is exposed and models can be listed:
    state = client.json('GET', '/api/state')
    assert set(state['options']['backends']) == {
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
    }
    monkeypatch.setattr(
        server_module, 'list_available_models', lambda b, k: [f'{b}-model-{k}']
    )
    listed = client.json('POST', '/api/models', {'backend': 'claude', 'api_key': 'k'})
    assert listed == {'backend': 'claude', 'models': ['claude-model-k']}
    client.json('POST', '/api/models', {'backend': 'nope'}, expect=400)

    # Asset details and image serving:
    geojson = tmp_path / 'a.geojson'
    _asset_collection_covering_pixel(2, 2).to_geojson(geojson)
    client.json('POST', '/api/assets/load', {'path': str(geojson)})
    details = client.json('GET', '/api/asset?id=a1')
    assert details['attributes']['damage'] == 'Major'
    assert details['images'] == []
    client.json('GET', '/api/asset?id=missing', expect=404)

    image_file = tmp_path / 'crop.jpg'
    image_file.write_bytes(b'\xff\xd8\xff\xd9')
    from rapidtools.core import ImageAsset

    srv.state.collection.get('a1').add_image_assets(
        ImageAsset(id='a1_img', path=image_file)
    )
    details = client.json('GET', '/api/asset?id=a1')
    assert details['images'][0]['exists'] is True
    status, _, raw = client.request('GET', details['images'][0]['url'])
    assert status == 200 and raw == b'\xff\xd8\xff\xd9'
    client.json('GET', '/api/asset_image?asset=a1&image=other', expect=404)

    # Inference settings accept the new backends and the 4-bit flag:
    client.json(
        'POST',
        '/api/infer',
        {'prompt': 'p', 'backend': 'llama', 'load_in_4bit': False},
    )
    client.wait_idle()
    assert srv.state.workflow.analyze_calls[-1].backend == 'llama'
    assert srv.state.workflow.analyze_calls[-1].load_in_4bit is False
    client.json('POST', '/api/infer', {'prompt': 'p', 'backend': 'nope'}, expect=400)


def test_sample_download_uses_registry(gui, raster_path, monkeypatch):
    client, srv = gui
    calls = []

    def fake_download(name, output_dir='.'):
        calls.append((name, Path(output_dir)))
        return [raster_path]

    monkeypatch.setattr(rapidtools, 'download_dataset', fake_download)
    client.json('POST', '/api/imagery/sample', {'name': 'bogus'}, expect=400)
    client.json('POST', '/api/imagery/sample', {'name': 'eaton_patch1'})
    state = client.wait_idle()
    assert calls == [('eaton_patch1', srv.state.output_dir)]
    assert state['raster']['name'] == 'scene.tif'


# --------------------------------------------------------------- sharing
@pytest.fixture
def shared_gui(tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, 'AssetAnalysisWorkflow', FakeWorkflow)
    root = tmp_path / 'data'
    root.mkdir()
    srv = server_module.create_server(
        port=0, output_dir=root / 'out', token='s3cret', data_root=root
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield Client(srv.server_address[1]), srv, root
    finally:
        srv.shutdown()
        srv.server_close()


def test_token_required_and_share_link(shared_gui):
    client, srv, _root = shared_gui
    status, ctype, raw = client.request('GET', '/')
    assert status == 401 and 'text/html' in ctype and b'Access token' in raw
    client.json('GET', '/api/state', expect=401)
    client.json('POST', '/api/detect', {'assets': 'x'}, expect=401)
    client.json('POST', '/api/login', {'token': 'wrong'}, expect=401)
    status, _, raw = client.request('GET', '/?token=wrong')
    assert status == 401

    # The share link exchanges the token for a cookie and redirects:
    status, _, _ = client.request('GET', '/?token=s3cret')
    assert status == 302
    cookie = client.last_response.getheader('Set-Cookie')
    assert cookie.startswith('rapidtools_token=s3cret') and 'HttpOnly' in cookie
    client.headers['Cookie'] = cookie.split(';')[0]
    state = client.json('GET', '/api/state')
    assert state['options']['shared'] is True
    status, _, raw = client.request('GET', '/')
    assert status == 200 and b'Detect + run inference' in raw

    # A header works too, and the sign-in form sets the same cookie:
    client.headers = {}
    client.json('GET', '/api/state', headers={'X-Rapidtools-Token': 's3cret'})
    client.json('POST', '/api/login', {'token': 's3cret'})
    assert client.last_response.getheader('Set-Cookie').startswith('rapidtools_token=')

    links = srv.share_urls()
    assert links and all(link.endswith('?token=s3cret') for link in links)


def test_data_root_restricts_paths(shared_gui, raster_path, tmp_path):
    client, srv, root = shared_gui
    client.headers['Cookie'] = 'rapidtools_token=s3cret'
    inside = root / 'scene.tif'
    inside.write_bytes(raster_path.read_bytes())
    outside = tmp_path / 'elsewhere.txt'
    outside.write_text('secret')

    # Listing starts at the root and cannot climb above it:
    listing = client.json('GET', '/api/fs?kind=raster')
    assert listing['path'] == str(root) and listing['parent'] is None
    assert [f['name'] for f in listing['files']] == ['scene.tif']
    escaped = client.json('GET', f'/api/fs?kind=raster&path={tmp_path}')
    assert escaped['path'] == str(root)

    client.json('GET', f'/api/textfile?path={outside}', expect=403)
    client.json('POST', '/api/imagery/local', {'path': str(raster_path)}, expect=403)
    client.json('POST', '/api/output_dir', {'path': str(tmp_path)}, expect=403)
    client.json('POST', '/api/imagery/local', {'path': str(inside)})
    state = client.wait_idle()
    assert state['raster']['name'] == 'scene.tif'
    assert state['options']['data_root'] == str(root)


def test_auto_token_and_open_server(tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, 'AssetAnalysisWorkflow', FakeWorkflow)
    srv = server_module.create_server(port=0, output_dir=tmp_path, token='auto')
    try:
        assert srv.token and len(srv.token) >= 20
        assert srv.share_urls()[0].endswith(f'?token={srv.token}')
    finally:
        srv.server_close()
    srv = server_module.create_server(port=0, output_dir=tmp_path)
    try:
        assert srv.token is None and srv.share_urls() == [srv.url]
    finally:
        srv.server_close()


# ------------------------------------------------------------ AppState unit
def test_app_state_ignores_none_results_and_records_job_errors(tmp_path):
    """add_result(None) is a no-op and a crashing job is reported as an error."""
    state = server_module.AppState(tmp_path)
    state.add_result('nothing', None)
    assert state.results == []

    def explode():
        raise RuntimeError('kaboom')

    state.start_job('Boom', explode)
    state._worker.join(timeout=5)
    assert state.job['status'] == 'error'
    assert state.job['message'] == 'Boom failed: kaboom'
    assert state.status == 'Boom failed: kaboom'
    assert any('Boom failed: kaboom' in e['text'] for e in state.log.since(0))
    state.cancel_job()  # nothing running: silently ignored
    assert state.busy() is False


# -------------------------------------------------------- Api error paths
def test_detect_and_infer_preconditions(gui, raster_path):
    """Missing raster, bad numeric options and an empty collection are rejected."""
    client, srv = gui
    status, _, raw = client.request('POST', '/api/detect', {'assets': 'building'})
    assert status == 400 and b'Load an aerial image first' in raw
    status, _, raw = client.request('POST', '/api/imagery/local', {})
    assert status == 400 and b'Missing' in raw and b'path' in raw

    client.json('POST', '/api/imagery/local', {'path': str(raster_path)})
    client.wait_idle()
    status, _, raw = client.request(
        'POST', '/api/detect', {'assets': 'building', 'asset_size_m': 'huge'}
    )
    assert status == 400 and b'could not convert' in raw
    status, _, raw = client.request(
        'POST', '/api/infer', {'prompt': 'p', 'backend': 'gemini', 'api_key': 'k'}
    )
    assert status == 400 and b'Detect assets or load an asset file first' in raw


def test_run_all_fails_when_nothing_is_detected(gui, raster_path):
    """run_all stops with an error when detection finds no assets."""
    client, srv = gui
    client.json('POST', '/api/imagery/local', {'path': str(raster_path)})
    client.wait_idle()

    def detect_nothing(settings):
        combined = settings.output_dir / 'assets_final.geojson'
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        combined.write_text('{}')
        return DetectionResult(
            collection=PhysicalAssetCollection(),
            detection_raster=settings.raster_path.resolve(),
            per_asset_geojson={},
            combined_geojson=combined,
        )

    srv.state.workflow.detect = detect_nothing
    client.json(
        'POST',
        '/api/run_all',
        {'detect': {'assets': 'building'}, 'infer': {'prompt': 'p', 'api_key': 'k'}},
    )
    state = client.wait_idle()
    assert state['job']['status'] == 'error'
    assert 'No assets were detected' in state['job']['message']


def test_detection_on_bing_basemap_renders_second_preview(gui, raster_path, tmp_path):
    """When detection ran on a different raster, a basemap preview is published."""
    client, srv = gui
    basemap = tmp_path / 'bing_basemap.tif'
    shutil.copy(raster_path, basemap)
    client.json('POST', '/api/imagery/local', {'path': str(raster_path)})
    client.wait_idle()
    assert client.json('GET', '/api/overlay?preview=basemap')['count'] == 0

    def detect_on_bing(settings):
        return DetectionResult(
            collection=_asset_collection_covering_pixel(10, 4),
            detection_raster=basemap,
            per_asset_geojson={},
            combined_geojson=None,
        )

    srv.state.workflow.detect = detect_on_bing
    client.json('POST', '/api/detect', {'assets': 'building'})
    state = client.wait_idle()
    assert state['job']['status'] == 'done'
    assert state['basemap']['name'] == 'bing_basemap.tif'
    assert {r['label'] for r in state['results']} == {'Bing basemap'}
    status, ctype, _ = client.request('GET', '/api/preview/basemap.png')
    assert status == 200 and ctype == 'image/png'
    assert client.json('GET', '/api/overlay?preview=basemap')['count'] == 1
    # Loading new imagery drops the basemap again:
    client.json('POST', '/api/imagery/local', {'path': str(raster_path)})
    assert client.wait_idle()['basemap'] is None


def test_load_assets_rejects_bad_and_empty_geojson(gui, tmp_path):
    """Unparseable GeoJSON and feature-less collections are 400 errors."""
    client, _ = gui
    bad = tmp_path / 'bad.geojson'
    bad.write_text('this is not json')
    status, _, raw = client.request('POST', '/api/assets/load', {'path': str(bad)})
    assert status == 400 and b'Could not read GeoJSON' in raw
    empty = tmp_path / 'empty.geojson'
    empty.write_text(json.dumps({'type': 'FeatureCollection', 'features': []}))
    status, _, raw = client.request('POST', '/api/assets/load', {'path': str(empty)})
    assert status == 400 and b'contains no features' in raw


def test_sample_prompt_and_text_file_limits(gui, tmp_path, monkeypatch):
    """The sample prompt comes from the registry; text files are size-limited."""
    client, srv = gui
    prompt_file = tmp_path / 'prompt.txt'
    prompt_file.write_text('  Rate the damage.  \n')
    calls = []

    def fake_download(name, output_dir='.'):
        calls.append((name, Path(output_dir)))
        return [prompt_file]

    monkeypatch.setattr(rapidtools, 'download_dataset', fake_download)
    assert client.json('POST', '/api/prompt/sample', {}) == {
        'name': 'aerial_chs',
        'text': 'Rate the damage.',
    }
    assert calls == [(server_module.SAMPLE_PROMPT_DATASET, srv.state.output_dir)]

    client.json('GET', f'/api/textfile?path={tmp_path / "missing.txt"}', expect=404)
    monkeypatch.setattr(server_module, 'MAX_TEXT_FILE_BYTES', 4)
    status, _, raw = client.request('GET', f'/api/textfile?path={prompt_file}')
    assert status == 400 and b'too large' in raw


def test_fs_listing_edge_cases(gui, tmp_path, monkeypatch):
    """File paths list their parent; unreadable dirs and broken entries are handled."""
    client, _ = gui
    (tmp_path / 'notes.txt').write_text('x')
    (tmp_path / 'ghost.txt').symlink_to(tmp_path / 'does-not-exist.txt')
    listing = client.json('GET', f'/api/fs?kind=text&path={tmp_path / "notes.txt"}')
    assert listing['path'] == str(tmp_path)
    assert [f['name'] for f in listing['files']] == ['notes.txt']  # ghost skipped

    locked = tmp_path / 'locked'
    locked.mkdir()
    real_iterdir = Path.iterdir

    def guarded_iterdir(self):
        if Path(self).resolve() == locked.resolve():
            raise PermissionError('nope')
        return real_iterdir(self)

    monkeypatch.setattr(Path, 'iterdir', guarded_iterdir)
    status, _, raw = client.request('GET', f'/api/fs?kind=dir&path={locked}')
    assert status == 403 and b'Permission denied' in raw


def test_overlay_and_table_without_data_and_overlay_retries(
    gui, raster_path, tmp_path, monkeypatch
):
    """Empty responses without data; overlay retries transient RuntimeErrors."""
    client, srv = gui
    assert client.json('GET', '/api/overlay') == {
        'features': [],
        'count': 0,
        'attribute_keys': [],
        'collection_version': 0,
    }
    assert client.json('GET', '/api/table') == {'rows': [], 'columns': []}
    client.json('GET', '/api/asset?id=a1', expect=404)  # no collection yet

    client.json('POST', '/api/imagery/local', {'path': str(raster_path)})
    client.wait_idle()
    geojson = tmp_path / 'a.geojson'
    _asset_collection_covering_pixel(2, 2).to_geojson(geojson)
    client.json('POST', '/api/assets/load', {'path': str(geojson)})

    real_project = server_module.project_collection
    attempts = []

    def flaky_project(collection, preview):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError('dictionary changed size during iteration')
        return real_project(collection, preview)

    monkeypatch.setattr(server_module, 'project_collection', flaky_project)
    assert client.json('GET', '/api/overlay')['count'] == 1
    assert len(attempts) == 3

    def always_fails(collection, preview):
        raise RuntimeError('still changing')

    monkeypatch.setattr(server_module, 'project_collection', always_fails)
    status, _, raw = client.request('GET', '/api/overlay')
    assert status == 500 and b'still changing' in raw


def test_asset_image_missing_on_disk(gui, raster_path, tmp_path):
    """An image whose file vanished is reported as 404 with exists=False."""
    client, srv = gui
    client.json('POST', '/api/imagery/local', {'path': str(raster_path)})
    client.wait_idle()
    geojson = tmp_path / 'a.geojson'
    _asset_collection_covering_pixel(2, 2).to_geojson(geojson)
    client.json('POST', '/api/assets/load', {'path': str(geojson)})
    srv.state.collection.get('a1').add_image_assets(
        ImageAsset(id='gone', path=tmp_path / 'gone.jpg', allow_missing_file=True)
    )
    details = client.json('GET', '/api/asset?id=a1')
    assert details['images'][0]['exists'] is False
    status, _, raw = client.request('GET', details['images'][0]['url'])
    assert status == 404 and b'Image file is missing' in raw


# ----------------------------------------------------- HTTP layer details
def test_static_files_and_missing_index(gui, monkeypatch, tmp_path):
    """Static files are served from the static dir only; a missing index is 404."""
    client, _ = gui
    status, ctype, raw = client.request('GET', '/static/index.html')
    assert status == 200 and 'text/html' in ctype and b'rAPIdtools' in raw
    client.json('GET', '/static/missing.js', expect=404)
    client.json('GET', '/static/../server.py', expect=404)  # cannot escape
    client.json('GET', '/static/', expect=404)

    monkeypatch.setattr(server_module, 'STATIC_DIR', tmp_path)
    client.json('GET', '/', expect=404)
    client.json('GET', '/index.html', expect=404)


def test_unhandled_exceptions_become_500(gui, monkeypatch):
    """Unexpected exceptions in GET and POST handlers return JSON 500 errors."""
    client, srv = gui

    def broken(*args, **kwargs):
        raise ZeroDivisionError('unexpected')

    monkeypatch.setattr(srv.api, 'table', broken)
    status, ctype, raw = client.request('GET', '/api/table')
    assert status == 500 and 'json' in ctype and b'unexpected' in raw
    monkeypatch.setattr(srv.state, 'reset', broken)
    status, _, raw = client.request('POST', '/api/reset', {})
    assert status == 500 and b'unexpected' in raw


def test_invalid_json_bodies_and_open_login(gui):
    """Malformed or non-object JSON bodies are 400; login is trivial when open."""
    client, _ = gui
    cases = ((b'not json', b'Invalid JSON body'), (b'[1, 2]', b'object'))
    for body, message in cases:
        conn = HTTPConnection('127.0.0.1', client.port, timeout=10)
        conn.request(
            'POST',
            '/api/reset',
            body=body,
            headers={'Content-Type': 'application/json'},
        )
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        assert response.status == 400 and message in raw
    assert client.json('POST', '/api/login', {'token': 'anything'}) == {'ok': True}
    assert client.json('POST', '/api/login') == {'ok': True}  # empty body


def test_verbose_access_log(tmp_path, monkeypatch, caplog):
    """With verbose=True every request is logged at DEBUG level."""
    monkeypatch.setattr(server_module, 'AssetAnalysisWorkflow', FakeWorkflow)
    srv = server_module.create_server(port=0, output_dir=tmp_path, verbose=True)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        with caplog.at_level(logging.DEBUG, logger='rapidtools.gui.server'):
            Client(srv.server_address[1]).json('GET', '/api/state')
            time.sleep(0.1)
    finally:
        srv.shutdown()
        srv.server_close()
    assert 'GET /api/state' in caplog.text


def test_malformed_cookie_is_ignored(shared_gui, monkeypatch):
    """A cookie header the parser rejects counts as no token at all."""
    client, _, _ = shared_gui

    class BrokenCookie:
        def load(self, raw):
            raise ValueError('bad cookie')

    monkeypatch.setattr(server_module, 'SimpleCookie', BrokenCookie)
    client.headers['Cookie'] = 'rapidtools_token=s3cret'
    client.json('GET', '/api/state', expect=401)
    # A header token still works because it is checked before the cookie:
    client.json('GET', '/api/state', headers={'X-Rapidtools-Token': 's3cret'})


def test_share_urls_on_all_interfaces_and_lan_ip_failure(tmp_path, monkeypatch):
    """Binding 0.0.0.0 lists the hostname (and LAN IP); _lan_ip tolerates errors."""
    monkeypatch.setattr(server_module, 'AssetAnalysisWorkflow', FakeWorkflow)
    srv = server_module.create_server(host='0.0.0.0', port=0, output_dir=tmp_path)
    try:
        links = srv.share_urls()
        port = srv.server_address[1]
        assert links and all(link.endswith(f':{port}/') for link in links)
        assert links[0] == f'http://{server_module.socket.gethostname()}:{port}/'
        assert srv.url == f'http://localhost:{port}/'
    finally:
        srv.server_close()

    class NoNetwork:
        def __init__(self, *args, **kwargs):
            raise OSError('no route')

    monkeypatch.setattr(server_module.socket, 'socket', NoNetwork)
    assert server_module._lan_ip() is None


# ------------------------------------------------------------ entry points
def test_launch_asset_analysis_app_runs_until_interrupted(
    tmp_path, monkeypatch, caplog
):
    """launch_asset_analysis_app logs its URLs, opens a browser and shuts down."""
    monkeypatch.setattr(server_module, 'AssetAnalysisWorkflow', FakeWorkflow)
    opened = []

    class ImmediateTimer:
        def __init__(self, delay, function):
            self.function = function

        def start(self):
            self.function()

    monkeypatch.setattr(server_module.threading, 'Timer', ImmediateTimer)
    monkeypatch.setattr(server_module.webbrowser, 'open', opened.append)

    def interrupted(self):
        raise KeyboardInterrupt

    monkeypatch.setattr(server_module.GuiServer, 'serve_forever', interrupted)
    with caplog.at_level(logging.INFO, logger='rapidtools.gui.server'):
        server_module.launch_asset_analysis_app(
            port=0, output_dir=tmp_path, token='auto', data_root=tmp_path
        )
    assert len(opened) == 1 and '?token=' in opened[0]
    assert 'Access token:' in caplog.text
    assert 'Share this link with colleagues' in caplog.text
    assert f'File access is limited to {tmp_path.resolve()}' in caplog.text
    assert 'Shutting down the rapidtools GUI.' in caplog.text
    assert 'reachable from other machines' not in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='rapidtools.gui.server'):
        server_module.launch_asset_analysis_app(
            host='0.0.0.0', port=0, output_dir=tmp_path, open_browser=False
        )
    assert 'reachable from other machines without an access token' in caplog.text
    assert len(opened) == 1  # no browser this time


def test_launch_asset_analysis_app_exits_when_port_is_taken(monkeypatch, caplog):
    """A bind failure is reported and turned into SystemExit(1)."""

    def cannot_bind(**kwargs):
        raise OSError(98, 'Address already in use')

    monkeypatch.setattr(server_module, 'create_server', cannot_bind)
    with caplog.at_level(logging.ERROR, logger='rapidtools.gui.server'):
        with pytest.raises(SystemExit) as excinfo:
            server_module.launch_asset_analysis_app(port=1, open_browser=False)
    assert excinfo.value.code == 1
    assert 'Could not start the rapidtools GUI on 127.0.0.1:1' in caplog.text


def test_main_parses_arguments_and_environment(monkeypatch, tmp_path):
    """main() maps CLI flags and environment variables onto the launcher."""
    calls = []
    monkeypatch.setattr(
        server_module, 'launch_asset_analysis_app', lambda **kw: calls.append(kw)
    )
    monkeypatch.delenv('RAPIDTOOLS_GUI_TOKEN', raising=False)
    monkeypatch.delenv('RAPIDTOOLS_GUI_NO_BROWSER', raising=False)
    for var in ('HOST', 'PORT', 'USER', 'PASSWORD', 'FROM', 'SSL'):
        monkeypatch.delenv(f'RAPIDTOOLS_SMTP_{var}', raising=False)
    monkeypatch.delenv('RAPIDTOOLS_PUBLIC_URL', raising=False)
    assert server_module.main([]) == 0
    notification = calls[-1].pop('notification')
    assert calls[-1] == {
        'host': '127.0.0.1',
        'port': 8765,
        'output_dir': None,
        'token': None,
        'data_root': None,
        'open_browser': True,
    }
    assert notification.smtp_host == '' and notification.smtp_port == 587

    argv = [
        '--host',
        '0.0.0.0',
        '--port',
        '9000',
        '--output-dir',
        str(tmp_path),
        '--token',
        'auto',
        '--data-root',
        str(tmp_path),
        '--no-browser',
        '--smtp-host',
        'mail.example.org',
        '--smtp-user',
        'me@example.org',
        '--smtp-ssl',
        '--public-url',
        'https://gui.example.org/',
    ]
    monkeypatch.setenv('RAPIDTOOLS_SMTP_PASSWORD', 'pw')
    assert server_module.main(argv) == 0
    notification = calls[-1].pop('notification')
    assert calls[-1] == {
        'host': '0.0.0.0',
        'port': 9000,
        'output_dir': str(tmp_path),
        'token': 'auto',
        'data_root': str(tmp_path),
        'open_browser': False,
    }
    assert notification.smtp_host == 'mail.example.org'
    assert notification.smtp_user == 'me@example.org'
    assert notification.smtp_password == 'pw'
    assert notification.smtp_ssl is True and notification.smtp_port == 465
    assert notification.public_url == 'https://gui.example.org/'

    monkeypatch.setenv('RAPIDTOOLS_GUI_TOKEN', 'from-env')
    monkeypatch.setenv('RAPIDTOOLS_GUI_NO_BROWSER', '1')
    assert server_module.main([]) == 0
    assert calls[-1]['token'] == 'from-env'
    assert calls[-1]['open_browser'] is False


def test_python_m_rapidtools_gui_exits_with_main_status(monkeypatch):
    """``python -m rapidtools.gui`` forwards main()'s return code to SystemExit."""
    monkeypatch.setattr(server_module, 'main', lambda argv=None: 3)
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module('rapidtools.gui.__main__', run_name='__main__')
    assert excinfo.value.code == 3


def test_gui_package_launch_forwards_kwargs(monkeypatch):
    """rapidtools.gui.launch_asset_analysis_app delegates to the server module."""
    calls = []
    monkeypatch.setattr(
        server_module, 'launch_asset_analysis_app', lambda **kw: calls.append(kw)
    )
    rapidtools.gui.launch_asset_analysis_app(port=1, open_browser=False)
    assert calls == [{'port': 1, 'open_browser': False}]


# ------------------------------------------------------- new API endpoints
def test_state_exposes_new_options_and_prompt_library(gui, monkeypatch, tmp_path):
    client, srv = gui
    state = client.json('GET', '/api/state')
    options = state['options']
    assert set(options['sample_prompts']) == {
        'aerial_chs',
        'street_chs',
        'street_recovery',
    }
    assert set(options['imagery_sources']) >= {
        'aerial',
        'google_streetview',
        'mapillary',
    }
    assert set(options['basemaps']) == {'bing', 'google'}
    assert 'vehicles' in options['street_classes']
    assert 'draft' in options['assist_actions']
    assert state['assistant']['status'] == 'idle'

    prompt_file = tmp_path / 'p.txt'
    prompt_file.write_text('Street prompt')
    calls = []

    def fake_download(name, output_dir='.'):
        calls.append(name)
        return [prompt_file]

    monkeypatch.setattr(rapidtools, 'download_dataset', fake_download)
    assert client.json('POST', '/api/prompt/sample', {'name': 'street_chs'}) == {
        'name': 'street_chs',
        'text': 'Street prompt',
    }
    client.json('POST', '/api/prompt/sample', {'name': 'nope'}, expect=400)
    prompt_file.write_text('MLY|token\n')
    assert client.json('POST', '/api/mapillary_token', {}) == {'token': 'MLY|token'}
    assert calls == ['street_chs_prompts', 'mapillary_token']


def test_prompt_assemble_and_example(gui):
    client, _ = gui
    example = client.json('POST', '/api/prompt/example', {})['spec']
    assert example['rubric_title'] == 'CHS COMBUSTION INDEX'

    r = client.json(
        'POST',
        '/api/prompt/assemble',
        {
            'spec': {
                'asset': 'vehicle',
                'fields': [{'name': 'Condition', 'options': ['intact', 'debris']}],
            },
            'outline': {'shape': 'corners', 'color': 'cyan', 'overlay': True},
            'backend': 'gemma4',
        },
    )
    assert r['marking'] == 'marked with cyan corner brackets'
    assert 'Locate the primary vehicle marked with cyan corner brackets' in r['text']
    assert r['attributes'] == ['gemma4_condition']
    assert r['spec']['asset'] == 'vehicle'
    # An explicit marking wins and an unknown backend falls back to 'vlm':
    r = client.json(
        'POST',
        '/api/prompt/assemble',
        {'spec': {'marking': 'in the centre', 'fields': [{'name': 'Grade'}]}},
    )
    assert r['marking'] == 'in the centre' and r['attributes'] == ['vlm_grade']
    client.json('POST', '/api/prompt/assemble', {'spec': ['bad']}, expect=400)
    client.json(
        'POST',
        '/api/prompt/assemble',
        {'spec': {'fields': [{'name': 'X', 'kind': 'weird'}]}},
        expect=400,
    )


def test_prompt_assist_runs_as_a_job(gui):
    client, srv = gui
    answers = {'spec': {'asset': 'pole', 'fields': [{'name': 'State'}]}}

    def fake_assist(settings):
        assert settings.action == 'draft' and settings.backend == 'gemma4'
        assert settings.context == {'asset': 'pole'}
        return answers

    srv.state.workflow.assist = fake_assist
    r = client.json(
        'POST',
        '/api/prompt/assist',
        {
            'action': 'draft',
            'backend': 'gemma4',
            'brief': 'poles',
            'context': {'asset': 'pole'},
        },
    )
    assert r == {'ok': True, 'seq': 1}
    state = client.wait_idle()
    assert state['job']['status'] == 'done'
    assert state['assistant']['status'] == 'done'
    assert state['assistant']['result'] == answers
    assert state['assistant']['model'].startswith('Gemma-4 ·')

    def failing(settings):
        raise ValueError('no JSON')

    srv.state.workflow.assist = failing
    assert (
        client.json(
            'POST',
            '/api/prompt/assist',
            {'action': 'review', 'backend': 'gemma4', 'prompt_text': 'x'},
        )['seq']
        == 2
    )
    state = client.wait_idle()
    assert state['job']['status'] == 'error'
    assert state['assistant'] == {
        'seq': 2,
        'status': 'error',
        'action': 'review',
        'model': state['assistant']['model'],
        'result': None,
        'error': 'no JSON',
    }

    # Validation errors are reported synchronously:
    client.json(
        'POST',
        '/api/prompt/assist',
        {'action': 'nope', 'backend': 'gemma4'},
        expect=400,
    )
    status, _, raw = client.request(
        'POST',
        '/api/prompt/assist',
        {'action': 'draft', 'backend': 'gemini', 'brief': 'x'},
    )
    assert status == 400 and b'Gemini API key is required' in raw

    # A busy server rejects the request and marks the assistant as failed:
    srv.state.workflow.block.clear()
    srv.state.workflow.assist = lambda settings: answers
    srv.state.start_job('Busy', lambda: srv.state.workflow.block.wait(timeout=5))
    client.json(
        'POST',
        '/api/prompt/assist',
        {'action': 'draft', 'backend': 'gemma4', 'brief': 'x'},
        expect=409,
    )
    assert srv.state.assistant['status'] == 'error'
    srv.state.workflow.block.set()
    client.wait_idle()


def test_region_imagery_download(gui, raster_path, tmp_path):
    client, srv = gui
    calls = []

    def fake_download(settings):
        calls.append(settings)
        target = tmp_path / 'region.tif'
        shutil.copy(raster_path, target)
        return target

    srv.state.workflow.download_basemap = fake_download
    client.json(
        'POST',
        '/api/imagery/region',
        {
            'provider': 'google',
            'zoom': 18,
            'min_lon': '-118.15',
            'min_lat': '34.18',
            'max_lon': '-118.14',
            'max_lat': '34.19',
        },
    )
    state = client.wait_idle()
    assert state['job'] == {
        **state['job'],
        'name': 'Download Google imagery',
        'status': 'done',
    }
    assert state['raster']['name'] == 'region.tif'
    assert calls[0].provider == 'google' and calls[0].zoom == 18
    assert calls[0].output_dir == srv.state.output_dir

    geojson = tmp_path / 'area.geojson'
    geojson.write_text('{}')
    client.json('POST', '/api/imagery/region', {'geojson': str(geojson), 'zoom': 17})
    client.wait_idle()
    assert calls[1].geojson_path == geojson and calls[1].provider == 'bing'

    status, _, raw = client.request('POST', '/api/imagery/region', {'provider': 'esri'})
    assert status == 400 and b'provider must be one of' in raw
    status, _, raw = client.request(
        'POST',
        '/api/imagery/region',
        {'min_lon': 'abc', 'min_lat': 1, 'max_lon': 2, 'max_lat': 3},
    )
    assert status == 400


def test_street_discovery_endpoint_and_run_all(gui, raster_path):
    client, srv = gui
    workflow = srv.state.workflow
    calls = []

    def fake_discover(settings):
        calls.append(settings)
        collection = _asset_collection_covering_pixel(10, 4)
        path = settings.output_dir / 'street_objects.geojson'
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
        from rapidtools.gui.workflow import StreetDetectionResult

        return StreetDetectionResult(
            collection=collection, geojson_path=path, n_images=3
        )

    workflow.discover_street = fake_discover

    # Needs a raster first:
    status, _, raw = client.request(
        'POST', '/api/street/discover', {'classes': 'vehicles', 'mapillary_token': 't'}
    )
    assert status == 400 and b'Load an aerial image' in raw

    client.json('POST', '/api/imagery/local', {'path': str(raster_path)})
    client.wait_idle()
    status, _, raw = client.request(
        'POST', '/api/street/discover', {'classes': 'vehicles'}
    )
    assert status == 400 and b'Mapillary access token' in raw

    client.json(
        'POST',
        '/api/street/discover',
        {
            'classes': 'vehicles, poles',
            'mapillary_token': 'MLY|x',
            'frame_spacing_m': 5,
            'min_observations': 1,
            'detection_source': 'mapillary',
        },
    )
    state = client.wait_idle()
    assert state['job']['status'] == 'done'
    assert state['collection']['source'] == 'street survey'
    assert [r['label'] for r in state['results']] == ['Street objects']
    assert calls[0].classes == ['vehicles', 'poles']
    assert calls[0].raster_path == raster_path
    assert calls[0].frame_spacing_m == 5 and calls[0].detection_source == 'mapillary'

    # Discovery + inference in one job:
    client.json(
        'POST',
        '/api/run_all',
        {
            'mode': 'street',
            'street': {'classes': 'vehicles', 'mapillary_token': 'MLY|x'},
            'infer': {
                'backend': 'gemma4',
                'prompt': 'Condition?',
                'imagery': 'mapillary_objects',
                'mapillary_token': 'MLY|x',
                'json_mode': True,
                'temperature': 0.1,
                'max_tokens': 300,
                'min_footprint_coverage': '',
            },
        },
    )
    state = client.wait_idle()
    assert state['job']['name'] == 'Street discovery + inference'
    assert state['job']['status'] == 'done'
    [settings] = workflow.analyze_calls
    assert settings.imagery == 'mapillary_objects'
    assert settings.json_mode is True and settings.temperature == 0.1
    assert settings.max_tokens == 300 and settings.min_footprint_coverage is None
    assert state['collection']['source'] == 'analyzed'

    # Nothing found means no inference:
    from rapidtools.gui.workflow import StreetDetectionResult

    workflow.discover_street = lambda s: StreetDetectionResult(
        PhysicalAssetCollection()
    )
    client.json(
        'POST',
        '/api/run_all',
        {
            'mode': 'street',
            'street': {'classes': 'vehicles', 'mapillary_token': 'MLY|x'},
            'infer': {'backend': 'gemma4', 'prompt': 'p'},
        },
    )
    state = client.wait_idle()
    assert state['job']['status'] == 'error' and 'No objects' in state['job']['message']


def test_detect_accepts_basemap_and_merge_options(gui, raster_path):
    client, srv = gui
    client.json('POST', '/api/imagery/local', {'path': str(raster_path)})
    client.wait_idle()
    client.json(
        'POST',
        '/api/detect',
        {'assets': 'vehicle', 'basemap': 'google', 'merge_overlaps': False},
    )
    client.wait_idle()
    [settings] = srv.state.workflow.detect_calls
    assert settings.basemap == 'google' and settings.merge_overlaps is False
    status, _, raw = client.request(
        'POST', '/api/detect', {'assets': 'vehicle', 'basemap': 'esri'}
    )
    assert status == 400 and b'basemap must be one of' in raw
    # The legacy 'source' key still selects the recon raster:
    client.json('POST', '/api/detect', {'assets': 'vehicle', 'source': 'recon'})
    client.wait_idle()
    assert srv.state.workflow.detect_calls[-1].basemap == 'recon'


def test_infer_validates_street_imagery_options(gui, raster_path):
    client, srv = gui
    client.json('POST', '/api/imagery/local', {'path': str(raster_path)})
    client.wait_idle()
    client.json('POST', '/api/detect', {'assets': 'building'})
    client.wait_idle()
    status, _, raw = client.request(
        'POST',
        '/api/infer',
        {'backend': 'gemma4', 'prompt': 'p', 'imagery': 'mapillary'},
    )
    assert status == 400 and b'Mapillary access token is required' in raw
    status, _, raw = client.request(
        'POST',
        '/api/infer',
        {
            'backend': 'gemma4',
            'prompt': 'p',
            'imagery': 'google_streetview',
            'street_crop_top': 0.9,
            'street_crop_bottom': 0.2,
        },
    )
    assert status == 400 and b'street_vertical_crop' in raw
    client.json(
        'POST',
        '/api/infer',
        {
            'backend': 'gemma4',
            'prompt': 'p',
            'imagery': 'google_streetview',
            'street_search_radius_m': 80,
            'street_max_images': 2,
            'street_crop_top': '0.1',
            'street_crop_bottom': '0.8',
            'pad_edges': False,
            'min_footprint_coverage': '0.5',
        },
    )
    client.wait_idle()
    settings = srv.state.workflow.analyze_calls[-1]
    assert settings.imagery == 'google_streetview'
    assert settings.street_search_radius_m == 80 and settings.street_max_images == 2
    assert settings.street_vertical_crop == (0.1, 0.8)
    assert settings.pad_edges is False and settings.min_footprint_coverage == 0.5


# ------------------------------------------------------ notifications, map picker
def test_notify_endpoint_and_delivery_after_job(gui, raster_path):
    from rapidtools.gui.notify import NotificationConfig, Notifier

    client, srv = gui
    posts = []

    class Response:
        def raise_for_status(self):
            return None

    def fake_post(url, json=None, timeout=None):
        posts.append((url, json))
        return Response()

    srv.state.notifier = Notifier(NotificationConfig(), post=fake_post)
    state = client.json('GET', '/api/state')
    assert state['notify'] == {
        'target': '',
        'kind': '',
        'email_available': False,
        'last': {'seq': 0, 'ok': None, 'message': ''},
    }
    status, _, raw = client.request('POST', '/api/notify', {'target': 'nope'})
    assert status == 400 and b'email address or a webhook' in raw
    status, _, raw = client.request('POST', '/api/notify', {'target': 'a@b.org'})
    assert status == 400 and b'not set up on this server' in raw
    assert client.json('POST', '/api/notify', {'target': ' https://hooks/x '}) == {
        'target': 'https://hooks/x',
        'kind': 'webhook',
    }
    assert client.json('GET', '/api/state')['notify']['target'] == 'https://hooks/x'

    client.json('POST', '/api/imagery/local', {'path': str(raster_path)})
    state = client.wait_idle()
    assert state['job']['status'] == 'done'
    deadline = time.time() + 5
    while time.time() < deadline and not posts:
        time.sleep(0.05)
    assert posts, 'webhook was not called'
    url, payload = posts[0]
    assert url == 'https://hooks/x'
    assert payload['job'] == 'Load imagery' and payload['status'] == 'done'
    assert payload['url'] == srv.state.public_url == srv.url
    assert payload['duration_s'] >= 0
    deadline = time.time() + 5
    while time.time() < deadline:
        state = client.json('GET', '/api/state')
        if state['notify']['last']['seq']:
            break
        time.sleep(0.05)
    assert state['notify']['last']['ok'] is True
    assert state['notify']['target'] == ''  # one message per request
    log = client.json('GET', '/api/log?since=0')
    assert any('Notification posted' in e['text'] for e in log['entries'])

    # Email goes through the SMTP channel once a relay is configured:
    sent = []
    notifier = Notifier(NotificationConfig(smtp_host='mail.example.org'))
    notifier.send_email = lambda to, summary: sent.append((to, summary.subject))
    srv.state.notifier = notifier
    assert client.json('GET', '/api/state')['notify']['email_available'] is True
    client.json('POST', '/api/notify', {'target': 'bob@example.org'})
    client.json('POST', '/api/imagery/local', {'path': str(raster_path)})
    client.wait_idle()
    deadline = time.time() + 5
    while time.time() < deadline and not sent:
        time.sleep(0.05)
    assert sent == [('bob@example.org', 'rapidtools: Load imagery finished')]
    # Clearing works and a cleared target sends nothing:
    assert client.json('POST', '/api/notify', {'target': ''})['target'] == ''


def test_basemap_tile_proxy_caches_and_validates(gui, monkeypatch):
    client, srv = gui
    calls = []

    def fake_fetch(provider, z, x, y):
        calls.append((provider, z, x, y))
        if provider not in ('bing', 'google'):
            raise ValueError('Unknown basemap provider')
        return None if z == 3 else b'\xff\xd8tile'

    monkeypatch.setattr(server_module, '_fetch_basemap_tile', fake_fetch)
    status, ctype, raw = client.request('GET', '/api/basemap_tile/bing/10/5/6.jpg')
    assert status == 200 and ctype == 'image/jpeg' and raw == b'\xff\xd8tile'
    assert client.last_response.getheader('Cache-Control') == 'max-age=86400'
    client.request('GET', '/api/basemap_tile/bing/10/5/6.jpg')
    assert calls == [('bing', 10, 5, 6)]  # served from the cache
    status, _, _ = client.request('GET', '/api/basemap_tile/esri/10/5/6.jpg')
    assert status == 400
    status, _, _ = client.request('GET', '/api/basemap_tile/google/3/1/1.jpg')
    assert status == 404
    status, _, _ = client.request('GET', '/api/basemap_tile/google/x/1/1.jpg')
    assert status == 400
    status, _, _ = client.request('GET', '/api/basemap_tile/google/1/1.jpg')
    assert status == 404
    assert len(srv.state.tile_cache) == 1


def test_fetch_basemap_tile_builds_provider_urls(monkeypatch):
    urls = []

    class Session:
        def get(self, url, timeout=None):
            urls.append(url)

            class R:
                status_code = 200
                content = b'img'

            return R()

    monkeypatch.setattr(server_module, '_TILE_SESSION', Session())
    assert server_module._fetch_basemap_tile('google', 5, 3, 4) == b'img'
    assert urls[-1] == 'https://mt3.google.com/vt/lyrs=s&x=3&y=4&z=5'
    assert server_module._fetch_basemap_tile('bing', 3, 3, 5) == b'img'
    assert urls[-1] == 'http://ecn.t3.tiles.virtualearth.net/tiles/a213.jpeg?g=1'
    with pytest.raises(ValueError):
        server_module._fetch_basemap_tile('bing', 3, 9, 0)
    with pytest.raises(ValueError):
        server_module._fetch_basemap_tile('osm', 3, 0, 0)


def test_street_coverage_endpoint(gui, monkeypatch):
    from rapidtools.core import ImageAsset, ImageCollection

    client, srv = gui
    calls = []

    class FakeMapillary:
        def __init__(self, token, save_dir=None):
            self.token = token

        def fetch_images_in_bbox(self, bbox, **kwargs):
            calls.append((self.token, bbox.bounds, kwargs))
            images = [
                ImageAsset(
                    id=str(i),
                    path=f'/virtual/{i}.jpg',
                    allow_missing_file=True,
                    properties={
                        'longitude': -117.4885 + i * 0.0001,
                        'latitude': 47.7115,
                        'sequence': 'seq' + str(i % 2),
                        'capture_date': f'2026-09-{10 + i:02d}',
                    },
                )
                for i in range(5)
            ]
            images.append(  # outside the box: dropped
                ImageAsset(
                    id='far',
                    path='/virtual/far.jpg',
                    allow_missing_file=True,
                    properties={'longitude': -117.6, 'latitude': 47.7115},
                )
            )
            return ImageCollection(images)

    monkeypatch.setattr('rapidtools.data_sources.MapillaryClient', FakeMapillary)
    monkeypatch.setattr(
        server_module.Api, '_download_text', lambda self, dataset: 'RAPID-TOKEN'
    )
    q = 'min_lon=-117.489&min_lat=47.711&max_lon=-117.487&max_lat=47.7125'
    data = client.json('GET', f'/api/street/coverage?{q}&start_date=2026-09-01')
    assert data['count'] == 5 and len(data['points']) == 5
    assert data['sequences'] == 2 and data['truncated'] is False
    assert (data['first'], data['last']) == ('2026-09-10', '2026-09-14')
    token, bounds, kwargs = calls[0]
    assert token == 'RAPID-TOKEN' and bounds == (-117.489, 47.711, -117.487, 47.7125)
    assert kwargs == {
        'start_date': '2026-09-01',
        'end_date': '',
        'filter_rapid_only': True,
    }
    client.json('GET', f'/api/street/coverage?{q}&token=MLY|me&rapid_only=0')
    assert calls[1][0] == 'MLY|me' and calls[1][2]['filter_rapid_only'] is False
    assert len(srv.state.mapillary_clients) == 2
    status, _, raw = client.request('GET', '/api/street/coverage?min_lon=1')
    assert status == 400 and b'all four' in raw
    status, _, raw = client.request(
        'GET', '/api/street/coverage?min_lon=-125&min_lat=30&max_lon=-100&max_lat=50'
    )
    assert status == 400 and b'Zoom in' in raw


def test_street_discovery_with_custom_box_needs_no_raster(gui):
    from rapidtools.gui.workflow import StreetDetectionResult

    client, srv = gui
    calls = []

    def fake_discover(settings):
        calls.append(settings)
        path = settings.output_dir / 'street_objects.geojson'
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
        return StreetDetectionResult(
            collection=_asset_collection_covering_pixel(10, 4),
            geojson_path=path,
            n_images=2,
        )

    srv.state.workflow.discover_street = fake_discover
    body = {'classes': 'vehicles', 'mapillary_token': 'MLY|x'}
    status, _, raw = client.request('POST', '/api/street/discover', body)
    assert status == 400 and b'pick an area on the map' in raw
    status, _, raw = client.request(
        'POST', '/api/street/discover', {**body, 'min_lon': '-117.49'}
    )
    assert status == 400 and b'all four' in raw
    status, _, raw = client.request(
        'POST',
        '/api/street/discover',
        {**body, 'min_lon': 'x', 'min_lat': 1, 'max_lon': 2, 'max_lat': 3},
    )
    assert status == 400 and b'must be numbers' in raw
    client.json(
        'POST',
        '/api/street/discover',
        {
            **body,
            'min_lon': '-117.489',
            'min_lat': '47.711',
            'max_lon': '-117.487',
            'max_lat': '47.7125',
        },
    )
    state = client.wait_idle()
    assert state['job']['status'] == 'done' and state['collection']['count'] == 1
    assert calls[0].raster_path is None
    assert calls[0].region.bounds == (-117.489, 47.711, -117.487, 47.7125)


def test_street_sequences_endpoint_caches_per_filter(gui, monkeypatch):
    client, srv = gui
    calls = []

    class FakeMapillary:
        def __init__(self, token, save_dir=None):
            self.token = token

        def fetch_sequence_lines(self, z, x, y, **kwargs):
            calls.append((self.token, z, x, y, kwargs))
            return {
                'extent': 4096,
                'lines': [[0, 0, 100, 100]],
                'count': 1,
                'first': '2026-09-01',
                'last': '2026-09-02',
            }

    monkeypatch.setattr('rapidtools.data_sources.MapillaryClient', FakeMapillary)
    monkeypatch.setattr(
        server_module.Api, '_download_text', lambda self, dataset: 'RAPID-TOKEN'
    )
    data = client.json('GET', '/api/street/sequences/10/178/365')
    assert data['count'] == 1 and data['lines'] == [[0, 0, 100, 100]]
    assert calls[0] == (
        'RAPID-TOKEN',
        10,
        178,
        365,
        {'filter_rapid_only': True, 'start_date': '', 'end_date': ''},
    )
    client.json('GET', '/api/street/sequences/10/178/365')
    assert len(calls) == 1  # cached
    client.json(
        'GET',
        '/api/street/sequences/10/178/365?rapid_only=0&start_date=2026-01-01&token=T',
    )
    assert calls[1][0] == 'T' and calls[1][4] == {
        'filter_rapid_only': False,
        'start_date': '2026-01-01',
        'end_date': '',
    }
    status, _, raw = client.request('GET', '/api/street/sequences/15/1/1')
    assert status == 400 and b'zoom levels 6 to 14' in raw
    status, _, _ = client.request('GET', '/api/street/sequences/10/a/1')
    assert status == 400
    status, _, _ = client.request('GET', '/api/street/sequences/10/1')
    assert status == 404


def test_street_overview_builds_in_background_and_caches(gui, monkeypatch, tmp_path):
    client, srv = gui
    srv.state.cache_dir = tmp_path / 'cache'
    fetched = []

    class FakeMapillary:
        def __init__(self, token, save_dir=None):
            pass

        def fetch_sequence_lines(self, z, x, y, **kwargs):
            fetched.append((z, x, y, kwargs['filter_rapid_only']))
            # One diagonal line across the tile:
            return {'extent': 4096, 'lines': [[0, 0, 4096, 4096]], 'count': 1}

    monkeypatch.setattr('rapidtools.data_sources.MapillaryClient', FakeMapillary)
    monkeypatch.setattr(
        server_module.Api, '_download_text', lambda self, dataset: 'RAPID-TOKEN'
    )
    monkeypatch.setattr(
        server_module, '_overview_tiles', lambda: [(11, 23, 6), (12, 23, 6)]
    )
    first = client.json('GET', '/api/street/overview')
    assert first['status'] in ('building', 'ready') and first['zoom'] == 6
    deadline = time.time() + 5
    while time.time() < deadline:
        data = client.json('GET', '/api/street/overview')
        if data['status'] == 'ready':
            break
        time.sleep(0.05)
    assert data['status'] == 'ready' and data['done'] == data['total'] == 2
    assert len(data['lines']) == 2
    lon0, lat0, lon1, lat1 = data['lines'][0]
    assert -125 < lon0 < lon1 < -60 and 20 < lat1 < lat0 < 55
    assert sorted(fetched) == [(6, 11, 23, True), (6, 12, 23, True)]
    cache = tmp_path / 'cache' / 'survey_overview_z6_1.json'
    assert cache.is_file()
    # A new state reads the cache instead of fetching again:
    fetched.clear()
    srv.state.overview.clear()
    data = client.json('GET', '/api/street/overview')
    deadline = time.time() + 5
    while data['status'] != 'ready' and time.time() < deadline:
        time.sleep(0.05)
        data = client.json('GET', '/api/street/overview')
    assert data['status'] == 'ready' and len(data['lines']) == 2 and fetched == []
    # rapid_only=0 is a separate index:
    client.json('GET', '/api/street/overview?rapid_only=0')
    deadline = time.time() + 5
    while time.time() < deadline and not fetched:
        time.sleep(0.05)
    assert fetched and fetched[0][3] is False


def test_geocode_endpoint(gui, monkeypatch):
    client, srv = gui
    calls = []

    def fake_geocode(text, limit=6):
        calls.append(text)
        if text == 'nowhere':
            raise RuntimeError('503 Service Unavailable')
        return [
            {
                'name': 'Spokane, WA',
                'lon': -117.4,
                'lat': 47.66,
                'bbox': [-117.6, 47.6, -117.3, 47.7],
            }
        ]

    monkeypatch.setattr(server_module, '_geocode', fake_geocode)
    data = client.json('GET', '/api/geocode?q=Spokane%2C%20WA')
    assert data['matches'][0]['name'] == 'Spokane, WA' and calls == ['Spokane, WA']
    status, _, raw = client.request('GET', '/api/geocode?q=')
    assert status == 400 and b'address or place name' in raw
    status, _, raw = client.request('GET', '/api/geocode?q=nowhere')
    assert status == 502 and b'Address lookup failed' in raw


def test_geocode_parses_nominatim(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                {
                    'display_name': 'Altadena, Los Angeles County, California, USA',
                    'lon': '-118.1312',
                    'lat': '34.1897',
                    'boundingbox': ['34.16', '34.22', '-118.17', '-118.09'],
                },
                {'display_name': 'broken'},
            ]

    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured.update(url=url, params=params, headers=headers)
        return Response()

    monkeypatch.setattr('requests.get', fake_get)
    matches = server_module._geocode('Altadena')
    assert captured['url'] == server_module.GEOCODE_URL
    assert captured['params'] == {'q': 'Altadena', 'format': 'jsonv2', 'limit': 6}
    assert 'rapidtools' in captured['headers']['User-Agent']
    assert matches == [
        {
            'name': 'Altadena, Los Angeles County, California, USA',
            'lon': -118.1312,
            'lat': 34.1897,
            'bbox': [-118.17, 34.16, -118.09, 34.22],
        }
    ]


def test_geo_overlay_returns_wgs84_geometries(gui, tmp_path):
    from shapely.geometry import LineString, Point

    client, srv = gui
    assert client.json('GET', '/api/geo_overlay') == {
        'collection_version': 0,
        'count': 0,
        'features': [],
    }
    collection = _asset_collection_covering_pixel(10, 4)
    collection.add(
        PhysicalAsset(
            id='p1',
            geometry=Point(-117.5, 47.7),
            attributes={
                'asset_type': 'vehicles',
                'n_observations': 3,
                'observations': [{'polygon': [[0, 0]]}],
                'nested': {'x': 1},
            },
        )
    )
    collection.add(
        PhysicalAsset(id='l1', geometry=LineString([(0, 0), (1, 1)]), attributes={})
    )
    srv.state.set_collection(collection, 'test')
    data = client.json('GET', '/api/geo_overlay')
    assert data['count'] == 3 and data['collection_version'] == 1
    by_id = {f['id']: f for f in data['features']}
    assert by_id['a1']['kind'] == 'polygon' and len(by_id['a1']['coords'][0]) == 5
    assert by_id['p1']['kind'] == 'point' and by_id['p1']['coords'] == [-117.5, 47.7]
    assert by_id['p1']['attributes'] == {'asset_type': 'vehicles', 'n_observations': 3}
    assert by_id['l1']['kind'] == 'line' and by_id['l1']['coords'] == [
        [[0.0, 0.0], [1.0, 1.0]]
    ]
    assert by_id['a1']['bbox'][0] < by_id['a1']['bbox'][2]


def test_static_logos_are_served(gui):
    client, _ = gui
    for name in ('RAPIDLogo.webp', 'rAPIdtoolsLogo.webp'):
        status, ctype, raw = client.request('GET', f'/static/{name}')
        assert status == 200 and ctype == 'image/webp' and raw[:4] == b'RIFF'


def test_route_database_builds_serves_gzip_and_rebuilds(gui, monkeypatch, tmp_path):
    import gzip

    client, srv = gui
    srv.state.cache_dir = tmp_path / 'cache'
    fetched = []

    class FakeMapillary:
        def __init__(self, token, save_dir=None):
            pass

        def fetch_sequence_lines(self, z, x, y, **kwargs):
            fetched.append((z, x, y))
            return {
                'extent': 4096,
                'lines': [[0, 0, 4096, 4096]],
                'count': 1,
                'first': '2026-09-01',
            }

    monkeypatch.setattr('rapidtools.data_sources.MapillaryClient', FakeMapillary)
    monkeypatch.setattr(
        server_module.Api, '_download_text', lambda self, dataset: 'RAPID-TOKEN'
    )
    monkeypatch.setattr(server_module, '_overview_tiles', lambda: [(11, 23, 6)])
    status, _, _ = client.request('GET', '/api/street/routes.json')
    assert status == 404
    first = client.json('GET', '/api/street/routes')
    assert first['status'] in ('building', 'ready') and first['zoom'] == 13
    deadline = time.time() + 10
    while time.time() < deadline:
        data = client.json('GET', '/api/street/routes')
        if data['status'] == 'ready':
            break
        time.sleep(0.05)
    assert data['status'] == 'ready' and data['n_routes'] >= 1 and data['bytes'] > 0
    assert data['built_at'].endswith('Z') and data['url'] == '/api/street/routes.json'
    # One zoom-6 tile was read for the overview, then the zoom-13 tiles under it:
    assert (6, 11, 23) in fetched and all(t[0] == 13 for t in fetched[1:])
    assert data['total'] == len(fetched) - 1 == data['done']

    status, ctype, raw = client.request(
        'GET', '/api/street/routes.json', headers={'Accept-Encoding': 'gzip'}
    )
    assert status == 200 and ctype == 'application/json'
    assert client.last_response.getheader('Content-Encoding') == 'gzip'
    etag = client.last_response.getheader('ETag')
    payload = json.loads(gzip.decompress(raw))
    assert payload['n_routes'] == len(payload['lines']) == len(payload['dates'])
    lon, lat = payload['lines'][0][:2]
    assert -125 < lon < -60 and 20 < lat < 55 and payload['dates'][0] == '2026-09-01'
    status, _, raw = client.request('GET', '/api/street/routes.json')
    assert status == 200 and client.last_response.getheader('Content-Encoding') is None
    assert json.loads(raw)['zoom'] == 13
    status, _, _ = client.request(
        'GET', '/api/street/routes.json', headers={'If-None-Match': etag}
    )
    assert status == 304
    cache = tmp_path / 'cache' / 'survey_routes_z13.json.gz'
    assert cache.is_file()

    # A fresh server state loads the file without touching Mapillary:
    fetched.clear()
    srv.state.routes = {
        'status': 'idle',
        'done': 0,
        'total': 0,
        'n_routes': 0,
        'built_at': '',
        'error': '',
    }
    srv.state.routes_gz = None
    deadline = time.time() + 10
    data = client.json('GET', '/api/street/routes')
    while data['status'] != 'ready' and time.time() < deadline:
        time.sleep(0.05)
        data = client.json('GET', '/api/street/routes')
    assert data['status'] == 'ready' and data['n_routes'] == payload['n_routes']
    assert fetched == []

    # Rebuild fetches again and refuses to start twice:
    client.json('POST', '/api/street/routes/rebuild', {})
    status, _, _ = client.request('POST', '/api/street/routes/rebuild', {})
    assert status in (200, 409)
    deadline = time.time() + 10
    while time.time() < deadline:
        data = client.json('GET', '/api/street/routes')
        if data['status'] == 'ready' and fetched:
            break
        time.sleep(0.05)
    assert data['status'] == 'ready' and fetched


def test_routes_tiles_cover_polylines():
    tiles = server_module._routes_tiles([[-117.49, 47.71, -117.48, 47.72]], zoom=13)
    assert tiles == [(1422, 2857, 13)]
    wide = server_module._routes_tiles([[-118.0, 34.0, -117.9, 34.1]], zoom=12)
    assert len(wide) >= 2 and all(t[2] == 12 for t in wide)
    assert server_module._routes_tiles([[]]) == []


def test_route_database_refuses_to_cache_when_tiles_fail(gui, monkeypatch, tmp_path):
    client, srv = gui
    srv.state.cache_dir = tmp_path / 'cache'
    calls = {'n': 0}

    class FlakyMapillary:
        def __init__(self, token, save_dir=None):
            pass

        def fetch_sequence_lines(self, z, x, y, **kwargs):
            calls['n'] += 1
            if z == 13:
                return {
                    'extent': 4096,
                    'lines': [],
                    'count': 0,
                    'ok': False,
                    'error': 'timeout',
                }
            return {
                'extent': 4096,
                'lines': [[0, 0, 4096, 4096]],
                'count': 1,
                'ok': True,
            }

    monkeypatch.setattr('rapidtools.data_sources.MapillaryClient', FlakyMapillary)
    monkeypatch.setattr(
        server_module.Api, '_download_text', lambda self, dataset: 'RAPID-TOKEN'
    )
    monkeypatch.setattr(server_module, '_overview_tiles', lambda: [(11, 23, 6)])
    deadline = time.time() + 10
    data = client.json('GET', '/api/street/routes')
    while data['status'] == 'building' and time.time() < deadline:
        time.sleep(0.05)
        data = client.json('GET', '/api/street/routes')
    assert data['status'] == 'error'
    assert (
        'could not be read' in data['error'] and 'nothing was cached' in data['error']
    )
    assert not (tmp_path / 'cache' / 'survey_routes_z13.json.gz').exists()
    status, _, _ = client.request('GET', '/api/street/routes.json')
    assert status == 404
    # The overview itself succeeded and was cached; a failed overview would not be:
    assert (tmp_path / 'cache' / 'survey_overview_z6_1.json').exists()
    # A rebuild request is accepted after a failure:
    client.json('POST', '/api/street/routes/rebuild', {})
