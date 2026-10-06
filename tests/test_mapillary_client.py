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
# 10-03-2026

import base64
import gzip
import json
import logging
import re
from datetime import date, datetime
from io import BytesIO
from pathlib import Path

import mapbox_vector_tile
import numpy as np
import pytest
import requests
from PIL import Image

from rapidtools.core import BoundingBox, ImageAsset, ImageCollection
from rapidtools.data_sources import mapillary_client as mc
from rapidtools.data_sources.mapillary_client import (
    BASE_URL,
    MAP_FEATURE_VALUES,
    RAPID_CREATOR_ID,
    TILE_URL_TEMPLATE,
    MapillaryClient,
    SegmentationLabels,
    is_map_feature_value,
)
from rapidtools.data_sources.tile_utils import TileUtils

TOKEN = 'MLY|fake|token'
TILE_URL_RE = re.compile(r'https://tiles\.mapillary\.com/.*')

# ==========================================
# Helpers & Fixtures
# ==========================================


def _jpeg_bytes(size=(8, 8), color='red') -> bytes:
    """Return the bytes of a tiny in-memory JPEG."""
    buf = BytesIO()
    Image.new('RGB', size, color=color).save(buf, format='JPEG')
    return buf.getvalue()


def _image_tile(features: list[dict]) -> bytes:
    """Encode a list of point features into an MVT 'image' layer."""
    layer = {'name': 'image', 'features': features}
    return mapbox_vector_tile.encode([layer])


def _point_feature(px: int, py: int, **props) -> dict:
    """Build a point feature dictionary for an MVT layer."""
    return {'geometry': f'POINT({px} {py})', 'properties': props}


def _b64_polygon(wkt: str) -> str:
    """Encode a WKT polygon into a base64 MVT string like Mapillary does."""
    layer = {'name': 'mpy-or', 'features': [{'geometry': wkt, 'properties': {}}]}
    return base64.encodebytes(mapbox_vector_tile.encode([layer])).decode('utf-8')


def _tile_url(x: int, y: int, z: int) -> str:
    """Return the fully-formatted Mapillary tile URL for the test token."""
    return TILE_URL_TEMPLATE.format(z=z, x=x, y=y, token=TOKEN)


def _json_response(payload: dict, status_code: int = 200) -> dict:
    """Build a requests_mock response dict carrying a JSON content type."""
    return {
        'status_code': status_code,
        'json': payload,
        'headers': {'Content-Type': 'application/json'},
    }


@pytest.fixture
def client(tmp_path):
    """Return a MapillaryClient writing into a temporary directory."""
    return MapillaryClient(TOKEN, save_dir=str(tmp_path / 'out'))


@pytest.fixture
def no_sleep(monkeypatch):
    """Disable backoff sleeping inside the client module."""
    monkeypatch.setattr(mc.time, 'sleep', lambda *_: None)


@pytest.fixture
def pano_asset(tmp_path):
    """Return an ImageAsset with 64x32 dimensions and no detections."""
    return ImageAsset(
        path=tmp_path / 'pano.jpg',
        id='pano',
        properties={'width': 64, 'height': 32},
        allow_missing_file=True,
    )


# ==========================================
# 1. Constants & Construction
# ==========================================


def test_segmentation_labels_constants():
    """Label namespace exposes the expected Mapillary strings."""
    assert SegmentationLabels.VOID == 'void--unlabeled'
    assert SegmentationLabels.SKY == 'nature--sky'
    assert SegmentationLabels.ROAD == 'construction--flat--road'
    assert SegmentationLabels.SURVEY_VEHICLE == 'void--ego-vehicle'


def test_module_constants():
    """Module-level URL templates and IDs are well formed."""
    assert BASE_URL.startswith('https://graph.mapillary.com')
    assert '{token}' in TILE_URL_TEMPLATE and '{z}' in TILE_URL_TEMPLATE
    assert isinstance(RAPID_CREATOR_ID, int)


def test_init_requires_token():
    """Empty or None tokens are rejected."""
    with pytest.raises(ValueError):
        MapillaryClient('')
    with pytest.raises(ValueError):
        MapillaryClient(None)


def test_init_sets_attributes(client, tmp_path):
    """Token, save directory and session are stored on the instance."""
    assert client.access_token == TOKEN
    assert client.save_dir == tmp_path / 'out'
    assert isinstance(client.session, requests.Session)
    assert not client.save_dir.exists()


def test_init_default_save_dir():
    """The default save directory is 'mapillary_images'."""
    assert MapillaryClient(TOKEN).save_dir == Path('mapillary_images')


# ==========================================
# 2. Field Validation
# ==========================================


def test_validate_fields_none_returns_all(client):
    """None requests every available field plus the download URL."""
    fields = client._validate_fields(None)
    assert set(fields) == set(MapillaryClient.AVAILABLE_IMAGE_FIELDS)
    assert 'thumb_original_url' in fields


def test_validate_fields_appends_url(client):
    """The download URL is always appended to valid user fields."""
    fields = client._validate_fields(['width', 'height'])
    assert sorted(fields) == ['height', 'thumb_original_url', 'width']


def test_validate_fields_warns_on_invalid(client, caplog):
    """Unknown fields are dropped with a warning."""
    with caplog.at_level(logging.WARNING):
        fields = client._validate_fields(['width', 'bogus'])
    assert 'bogus' in caplog.text
    assert 'bogus' not in fields and 'width' in fields


def test_validate_fields_all_invalid_returns_none(client, caplog):
    """If nothing valid remains, None is returned and an error logged."""
    with caplog.at_level(logging.ERROR):
        assert client._validate_fields(['bogus'], image_id='img1') is None
    assert 'img1' in caplog.text


def test_validate_fields_injects_segmentation(client):
    """Segmentation requests inject the detections fields."""
    fields = client._validate_fields(['width'], require_segmentation=True)
    assert 'detections.value' in fields
    assert 'detections.geometry' in fields


def test_validate_fields_url_not_duplicated(client):
    """Requesting thumb_original_url explicitly does not duplicate it."""
    fields = client._validate_fields(['thumb_original_url'])
    assert fields.count('thumb_original_url') == 1


# ==========================================
# 3. Date Range Checks
# ==========================================


@pytest.mark.parametrize(
    'check, start, end, expected',
    [
        ('2025-02-14', '2025-01-01', '2025-12-31', True),
        ('2025-01-01', '2025-01-01', '2025-12-31', True),
        ('2025-12-31', '2025-01-01', '2025-12-31', True),
        ('2024-06-01', '2025-01-01', '', False),
        ('2026-06-01', '', '2025-12-31', False),
        ('2025-06-01', '', '', True),
        (None, '', '', True),
        (None, '2025-01-01', '', False),
        ('garbage', '2025-01-01', '', False),
        ('garbage', '', '', True),
    ],
)
def test_is_date_in_range_strings(check, start, end, expected):
    """String dates are parsed and compared inclusively."""
    assert MapillaryClient._is_date_in_range(check, start, end) is expected


def test_is_date_in_range_date_and_datetime_objects():
    """date and datetime inputs are normalized before comparison."""
    assert MapillaryClient._is_date_in_range(
        datetime(2025, 6, 1, 12, 30), date(2025, 1, 1), date(2025, 12, 31)
    )
    assert not MapillaryClient._is_date_in_range(
        date(2024, 1, 1), datetime(2025, 1, 1), None
    )


def test_is_date_in_range_unsupported_type():
    """Unsupported types are treated as missing dates."""
    assert MapillaryClient._is_date_in_range(12345, '', '') is True
    assert MapillaryClient._is_date_in_range(12345, '2025-01-01', '') is False


# ==========================================
# 4. Metadata Retrieval
# ==========================================


def test_get_image_metadata_success(client, requests_mock):
    """A JSON response is returned as a dictionary with correct params."""
    requests_mock.get(f'{BASE_URL}/42', **_json_response({'id': '42', 'width': 100}))
    result = client._get_image_metadata('42', ['width', 'thumb_original_url'])
    assert result == {'id': '42', 'width': 100}
    qs = requests_mock.last_request.qs
    assert qs['access_token'] == [TOKEN.lower()] or qs['access_token'] == [TOKEN]
    assert qs['fields'] == ['width,thumb_original_url']


def test_get_image_metadata_retries_on_429(client, requests_mock, no_sleep):
    """Rate-limited responses are retried until a success arrives."""
    requests_mock.get(
        f'{BASE_URL}/42',
        [
            _json_response({}, 429),
            _json_response({}, 403),
            _json_response({'id': '42'}),
        ],
    )
    assert client._get_image_metadata('42', ['width']) == {'id': '42'}
    assert requests_mock.call_count == 3


def test_get_image_metadata_exhausts_retries(client, requests_mock, no_sleep, caplog):
    """After all retries fail, None is returned and a warning logged."""
    requests_mock.get(f'{BASE_URL}/42', status_code=500)
    with caplog.at_level(logging.WARNING):
        result = client._get_image_metadata('42', ['width'], retries=3)
    assert result is None
    assert requests_mock.call_count == 3
    assert 'All 3 attempts' in caplog.text


def test_get_image_metadata_non_json_content_type(client, requests_mock, no_sleep):
    """HTML error pages are treated as failures."""
    requests_mock.get(
        f'{BASE_URL}/42',
        text='<html>error</html>',
        headers={'Content-Type': 'text/html'},
    )
    assert client._get_image_metadata('42', ['width'], retries=2) is None


def test_get_image_metadata_connection_error(client, requests_mock, no_sleep):
    """Network exceptions are retried then reported as None."""
    requests_mock.get(f'{BASE_URL}/42', exc=requests.exceptions.ConnectionError)
    assert client._get_image_metadata('42', ['width'], retries=2) is None


def test_get_image_metadata_sleeps_with_backoff(client, requests_mock, monkeypatch):
    """Backoff waits double on every failed attempt."""
    waits = []
    monkeypatch.setattr(mc.time, 'sleep', waits.append)
    requests_mock.get(f'{BASE_URL}/42', status_code=429)
    client._get_image_metadata('42', ['width'], retries=3, backoff_factor=0.5)
    assert waits == [0.5, 1.0]


def test_get_image_metadata_corrupt_json_breaks(client, monkeypatch, caplog):
    """A plain JSONDecodeError aborts immediately without retrying."""

    class FakeResp:
        status_code = 200
        headers = {'Content-Type': 'application/json'}

        def raise_for_status(self):
            return None

        def json(self):
            raise json.JSONDecodeError('bad', 'doc', 0)

    calls = []

    def fake_get(*args, **kwargs):
        calls.append(args)
        return FakeResp()

    monkeypatch.setattr(mc.requests, 'get', fake_get)
    with caplog.at_level(logging.WARNING):
        assert client._get_image_metadata('42', ['width']) is None
    assert len(calls) == 1


# ==========================================
# 5. Image Download
# ==========================================


def test_download_image_success(client, requests_mock, tmp_path):
    """Streamed bytes are written to the destination path."""
    payload = _jpeg_bytes()
    requests_mock.get('https://cdn.example/pano.jpg', content=payload)
    dest = tmp_path / 'pano.jpg'
    assert client._download_image('https://cdn.example/pano.jpg', dest) is True
    assert dest.read_bytes() == payload


def test_download_image_http_error(client, requests_mock, tmp_path, caplog):
    """HTTP errors return False and log the failure."""
    requests_mock.get('https://cdn.example/pano.jpg', status_code=404)
    with caplog.at_level(logging.ERROR):
        ok = client._download_image('https://cdn.example/pano.jpg', tmp_path / 'p.jpg')
    assert ok is False
    assert 'Download failed' in caplog.text


def test_download_image_connection_error(client, requests_mock, tmp_path):
    """Connection errors return False."""
    requests_mock.get(
        'https://cdn.example/pano.jpg', exc=requests.exceptions.ConnectionError
    )
    assert (
        client._download_image('https://cdn.example/pano.jpg', tmp_path / 'p') is False
    )


# ==========================================
# 6. Vector Tile Decoding
# ==========================================


def test_get_tile_image_data_basic(client, requests_mock):
    """Point features are converted to ImageAssets with WGS84 coordinates."""
    x, y, z = 2814, 6534, 14
    tile = _image_tile(
        [
            _point_feature(
                100,
                200,
                id=111,
                creator_id=RAPID_CREATOR_ID,
                captured_at=1_700_000_000_000,
                organization_id=5,
                sequence_id='seq',
                is_pano=True,
            )
        ]
    )
    requests_mock.get(_tile_url(x, y, z), content=tile)
    assets = client._get_tile_image_data((x, y, z))

    assert len(assets) == 1
    asset = assets[0]
    assert asset.id == '111'
    assert asset.path == (client.save_dir / '111.jpg').resolve()
    props = asset.properties
    assert props['capture_date'] == '2023-11-14'
    assert props['is_pano'] is True
    for removed in (
        'id',
        'creator_id',
        'captured_at',
        'organization_id',
        'sequence_id',
    ):
        assert removed not in props

    # The decoded location must fall inside the tile's geographic bounds:
    lon_min, lat_max = TileUtils.mvt_to_wgs84(0, 0, x, y, z)
    lon_max, lat_min = TileUtils.mvt_to_wgs84(4096, 4096, x, y, z)
    assert lon_min <= props['longitude'] <= lon_max
    assert lat_min <= props['latitude'] <= lat_max


def test_get_tile_image_data_normalizes_ids_to_str(client, requests_mock):
    """Integer tile IDs become strings so they merge with Graph API assets."""
    tile = _image_tile(
        [_point_feature(1, 1, id=123456789, creator_id=RAPID_CREATOR_ID)]
    )
    requests_mock.get(TILE_URL_RE, content=tile)
    asset = client._get_tile_image_data((1, 2, 14))[0]
    assert asset.id == '123456789'
    assert asset.path.name == '123456789.jpg'


def test_get_tile_image_data_filters_creator(client, requests_mock):
    """Non-RAPID images are dropped unless filtering is disabled."""
    tile = _image_tile(
        [
            _point_feature(10, 10, id=1, creator_id=RAPID_CREATOR_ID),
            _point_feature(20, 20, id=2, creator_id=999),
        ]
    )
    requests_mock.get(TILE_URL_RE, content=tile)
    assert [a.id for a in client._get_tile_image_data((1, 2, 14))] == ['1']
    both = client._get_tile_image_data((1, 2, 14), filter_rapid_only=False)
    assert sorted(a.id for a in both) == ['1', '2']


def test_get_tile_image_data_date_filters(client, requests_mock):
    """Date bounds exclude images outside the range or lacking a date."""
    tile = _image_tile(
        [
            _point_feature(10, 10, id=1, captured_at=1_700_000_000_000),  # 2023-11-14
            _point_feature(20, 20, id=2, captured_at=1_600_000_000_000),  # 2020-09-13
            _point_feature(30, 30, id=3),  # no date
        ]
    )
    requests_mock.get(TILE_URL_RE, content=tile)
    kwargs = {'filter_rapid_only': False}

    ids = [a.id for a in client._get_tile_image_data((1, 2, 14), **kwargs)]
    assert sorted(ids) == ['1', '2', '3']

    ids = [
        a.id
        for a in client._get_tile_image_data(
            (1, 2, 14), start_date='2023-01-01', **kwargs
        )
    ]
    assert ids == ['1']

    ids = [
        a.id
        for a in client._get_tile_image_data(
            (1, 2, 14), end_date='2021-01-01', **kwargs
        )
    ]
    assert ids == ['2']


def test_get_tile_image_data_undated_without_filter(client, requests_mock):
    """Images with no timestamp are kept (without capture_date) if unfiltered."""
    tile = _image_tile([_point_feature(10, 10, id=7)])
    requests_mock.get(TILE_URL_RE, content=tile)
    assets = client._get_tile_image_data((1, 2, 14), filter_rapid_only=False)
    assert len(assets) == 1 and 'capture_date' not in assets[0].properties


def test_get_tile_image_data_gzip(client, requests_mock):
    """Gzip-compressed tiles are transparently decompressed."""
    tile = _image_tile([_point_feature(10, 10, id=5, creator_id=RAPID_CREATOR_ID)])
    requests_mock.get(TILE_URL_RE, content=gzip.compress(tile))
    assert [a.id for a in client._get_tile_image_data((3, 4, 14))] == ['5']


def test_get_tile_image_data_no_image_layer(client, requests_mock):
    """Tiles without an 'image' layer yield no assets."""
    other = mapbox_vector_tile.encode(
        [{'name': 'sequence', 'features': [_point_feature(1, 1, id=9)]}]
    )
    requests_mock.get(TILE_URL_RE, content=other)
    assert client._get_tile_image_data((3, 4, 14)) == []


def test_get_tile_image_data_http_error(client, requests_mock, caplog):
    """Download failures are logged as warnings and yield no assets."""
    requests_mock.get(TILE_URL_RE, status_code=500)
    with caplog.at_level(logging.WARNING):
        assert client._get_tile_image_data((3, 4, 14)) == []
    assert 'Failed to process tile 14/3/4' in caplog.text


def test_get_tile_image_data_corrupt_bytes(client, requests_mock, caplog):
    """Undecodable tile payloads are handled gracefully."""
    requests_mock.get(TILE_URL_RE, content=b'\x00\x01garbage')
    with caplog.at_level(logging.WARNING):
        assert client._get_tile_image_data((3, 4, 14)) == []
    assert 'Failed to process tile' in caplog.text


# ==========================================
# 7. Segmentation Mask Parsing
# ==========================================


def test_parse_segmentation_missing_dimensions(caplog):
    """Assets without width/height produce a 1x1 placeholder mask."""
    asset = ImageAsset(path='x.jpg', id='x', properties={}, allow_missing_file=True)
    with caplog.at_level(logging.WARNING):
        mask, seg_map = MapillaryClient._parse_mapillary_segmentation(asset)
    assert mask.shape == (1, 1) and seg_map == {}
    assert 'missing dimension' in caplog.text


def test_parse_segmentation_no_detections(pano_asset):
    """Without detections an all-zero mask of image size is returned."""
    mask, seg_map = MapillaryClient._parse_mapillary_segmentation(pano_asset)
    assert mask.shape == (32, 64) and mask.dtype == np.uint8
    assert not mask.any() and seg_map == {}


def test_parse_segmentation_semantic_merges_labels(pano_asset):
    """Semantic mode shares one ID per label and rasterizes polygons."""
    top_half = _b64_polygon('POLYGON((0 2048, 4096 2048, 4096 4096, 0 4096, 0 2048))')
    bottom_left = _b64_polygon('POLYGON((0 0, 2048 0, 2048 2048, 0 2048, 0 0))')
    pano_asset.properties['detections'] = {
        'data': [
            {'value': SegmentationLabels.SKY, 'geometry': top_half},
            {'value': SegmentationLabels.ROAD, 'geometry': bottom_left},
            {'value': SegmentationLabels.SKY, 'geometry': top_half},
        ]
    }
    mask, seg_map = MapillaryClient._parse_mapillary_segmentation(pano_asset)

    assert seg_map == {
        0: SegmentationLabels.VOID,
        1: SegmentationLabels.SKY,
        2: SegmentationLabels.ROAD,
    }
    assert mask.dtype == np.uint8
    # Sky covers the top of the image (flip_y=True), road the bottom-left:
    assert mask[2, 32] == 1
    assert mask[29, 8] == 2
    assert mask[29, 56] == 0


def test_parse_segmentation_instance_unique_ids(pano_asset):
    """Instance mode assigns a fresh ID to every detection."""
    geom = _b64_polygon('POLYGON((0 0, 4096 0, 4096 4096, 0 4096, 0 0))')
    pano_asset.properties['detections'] = {
        'data': [
            {'value': 'object--vehicle--car', 'geometry': geom},
            {'value': 'object--vehicle--car', 'geometry': geom},
        ]
    }
    mask, seg_map = MapillaryClient._parse_mapillary_segmentation(
        pano_asset, merge_detections=False
    )
    assert seg_map == {
        0: SegmentationLabels.VOID,
        1: 'object--vehicle--car',
        2: 'object--vehicle--car',
    }
    # The second detection is drawn last and covers the full frame:
    assert mask.min() == 2


def test_parse_segmentation_flip_y_false(pano_asset):
    """Disabling flip_y places tile-origin polygons at the top of the image."""
    bottom_in_tile = _b64_polygon('POLYGON((0 0, 4096 0, 4096 2048, 0 2048, 0 0))')
    pano_asset.properties['detections'] = {
        'data': [{'value': SegmentationLabels.ROAD, 'geometry': bottom_in_tile}]
    }
    flipped, _ = MapillaryClient._parse_mapillary_segmentation(pano_asset, flip_y=True)
    unflipped, _ = MapillaryClient._parse_mapillary_segmentation(
        pano_asset, flip_y=False
    )
    assert flipped[30, 32] == 1 and flipped[1, 32] == 0
    assert unflipped[1, 32] == 1 and unflipped[30, 32] == 0


def test_parse_segmentation_target_classes(pano_asset):
    """Only detections whose label matches a target prefix are drawn."""
    geom = _b64_polygon('POLYGON((0 0, 4096 0, 4096 4096, 0 4096, 0 0))')
    pano_asset.properties['detections'] = {
        'data': [
            {'value': SegmentationLabels.SKY, 'geometry': geom},
            {'value': SegmentationLabels.ROAD, 'geometry': geom},
        ]
    }
    mask, seg_map = MapillaryClient._parse_mapillary_segmentation(
        pano_asset, target_classes=['construction--']
    )
    assert seg_map == {0: SegmentationLabels.VOID, 1: SegmentationLabels.ROAD}
    assert mask.max() == 1


def test_parse_segmentation_multipolygon(pano_asset):
    """MultiPolygon geometries rasterize every part."""
    geom = _b64_polygon(
        'MULTIPOLYGON(((0 0, 1024 0, 1024 1024, 0 1024, 0 0)),'
        '((3072 3072, 4096 3072, 4096 4096, 3072 4096, 3072 3072)))'
    )
    pano_asset.properties['detections'] = {
        'data': [{'value': 'object--bench', 'geometry': geom}]
    }
    mask, _ = MapillaryClient._parse_mapillary_segmentation(pano_asset)
    assert mask[28, 4] == 1  # bottom-left part (after y flip)
    assert mask[2, 60] == 1  # top-right part
    assert mask[16, 32] == 0  # centre untouched


def test_parse_segmentation_skips_missing_and_bad_geometry(pano_asset, caplog):
    """Missing geometry is skipped; undecodable geometry logs an error."""
    pano_asset.properties['detections'] = {
        'data': [
            {'value': SegmentationLabels.SKY},
            {'value': SegmentationLabels.ROAD, 'geometry': 'bm90IGEgdGlsZQ=='},
        ]
    }
    with caplog.at_level(logging.ERROR):
        mask, seg_map = MapillaryClient._parse_mapillary_segmentation(pano_asset)
    assert not mask.any()
    assert set(seg_map.values()) == {
        SegmentationLabels.VOID,
        SegmentationLabels.SKY,
        SegmentationLabels.ROAD,
    }
    assert 'Error processing mask geometry' in caplog.text


def test_parse_segmentation_ignores_non_polygon_and_empty_tile(pano_asset):
    """Point features and empty tiles contribute nothing to the mask."""
    point_layer = {
        'name': 'mpy-or',
        'features': [{'geometry': 'POINT(10 10)', 'properties': {}}],
    }
    point_b64 = base64.encodebytes(mapbox_vector_tile.encode([point_layer])).decode()
    empty_b64 = base64.encodebytes(mapbox_vector_tile.encode([])).decode()
    pano_asset.properties['detections'] = {
        'data': [
            {'value': SegmentationLabels.SKY, 'geometry': point_b64},
            {'value': SegmentationLabels.ROAD, 'geometry': empty_b64},
        ]
    }
    mask, _ = MapillaryClient._parse_mapillary_segmentation(pano_asset)
    assert not mask.any()


def test_parse_segmentation_upgrades_to_int32(pano_asset):
    """More than 255 instances upgrades the canvas to 32-bit."""
    geom = _b64_polygon('POLYGON((0 0, 4096 0, 4096 4096, 0 4096, 0 0))')
    pano_asset.properties['detections'] = {
        'data': [{'value': 'object--bench', 'geometry': geom} for _ in range(256)]
    }
    mask, seg_map = MapillaryClient._parse_mapillary_segmentation(
        pano_asset, merge_detections=False, show_progress=True
    )
    assert mask.dtype == np.int32
    assert mask.max() == 256
    assert len(seg_map) == 257


def test_scale_ring():
    """Ring coordinates are scaled and optionally flipped."""
    ring = [(0, 0), (4096, 4096)]
    assert MapillaryClient._scale_ring(ring, 0.5, 0.25, 1024, flip_y=False) == [
        (0.0, 0.0),
        (2048.0, 1024.0),
    ]
    assert MapillaryClient._scale_ring([(0, 0)], 1.0, 1.0, 100, flip_y=True) == [
        (0.0, 100.0)
    ]


# ==========================================
# 8. fetch_image
# ==========================================


def _register_metadata(requests_mock, image_id='42', **extra):
    """Register a Graph API metadata response for ``image_id``."""
    payload = {
        'id': image_id,
        'thumb_original_url': f'https://cdn.example/{image_id}.jpg',
        'width': 64,
        'height': 32,
    }
    payload.update(extra)
    requests_mock.get(f'{BASE_URL}/{image_id}', **_json_response(payload))
    return payload


def test_fetch_image_metadata_only(client, requests_mock):
    """Without saving, an ImageAsset with metadata but no file is returned."""
    _register_metadata(requests_mock)
    asset = client.fetch_image('42', fields=['width', 'height'], save_to_disk=False)
    assert isinstance(asset, ImageAsset)
    assert asset.id == '42'
    assert asset.properties['width'] == 64
    assert 'id' not in asset.properties
    assert not asset.path.exists()
    assert not client.save_dir.exists()


def test_fetch_image_saves_to_disk(client, requests_mock):
    """Saving downloads the JPEG into the save directory."""
    _register_metadata(requests_mock)
    requests_mock.get('https://cdn.example/42.jpg', content=_jpeg_bytes())
    asset = client.fetch_image('42', fields=['width'])
    assert asset.path == (client.save_dir / '42.jpg').resolve()
    assert asset.path.exists()
    assert Image.open(asset.path).size == (8, 8)


def test_fetch_image_invalid_fields(client, requests_mock):
    """Entirely invalid fields abort before any network call."""
    assert client.fetch_image('42', fields=['bogus']) is None
    assert requests_mock.call_count == 0


def test_fetch_image_metadata_failure(client, requests_mock, no_sleep):
    """Metadata retrieval failures return None."""
    requests_mock.get(f'{BASE_URL}/42', status_code=500)
    assert client.fetch_image('42', fields=['width'], save_to_disk=False) is None


def test_fetch_image_missing_url(client, requests_mock, caplog):
    """Metadata lacking a download URL returns None."""
    requests_mock.get(f'{BASE_URL}/42', **_json_response({'id': '42', 'width': 1}))
    with caplog.at_level(logging.ERROR):
        assert client.fetch_image('42', fields=['width'], save_to_disk=False) is None
    assert 'No download URL' in caplog.text


def test_fetch_image_download_failure(client, requests_mock):
    """A failed image download returns None."""
    _register_metadata(requests_mock)
    requests_mock.get('https://cdn.example/42.jpg', status_code=404)
    assert client.fetch_image('42', fields=['width']) is None


def test_fetch_image_uses_requested_id_when_missing(client, requests_mock):
    """If the API omits 'id', the requested image ID is used."""
    requests_mock.get(
        f'{BASE_URL}/77',
        **_json_response({'thumb_original_url': 'https://cdn.example/77.jpg'}),
    )
    asset = client.fetch_image('77', fields=None, save_to_disk=False)
    assert asset.id == '77'


def test_fetch_image_with_masks(client, requests_mock):
    """Semantic and instance masks are built and saved alongside the image."""
    geom = _b64_polygon('POLYGON((0 2048, 4096 2048, 4096 4096, 0 4096, 0 2048))')
    _register_metadata(
        requests_mock,
        detections={
            'data': [
                {'value': SegmentationLabels.SKY, 'geometry': geom},
                {'value': SegmentationLabels.SKY, 'geometry': geom},
            ]
        },
    )
    requests_mock.get('https://cdn.example/42.jpg', content=_jpeg_bytes())

    asset = client.fetch_image(
        '42', fields=['width', 'height'], process_masks=['semantic', 'instance']
    )
    assert asset.semantic_map == {0: SegmentationLabels.VOID, 1: SegmentationLabels.SKY}
    assert asset.instance_map == {
        0: SegmentationLabels.VOID,
        1: SegmentationLabels.SKY,
        2: SegmentationLabels.SKY,
    }
    assert asset.get_mask_path('semantic').exists()
    assert asset.get_mask_path('instance').exists()
    assert asset.load_mask('semantic').shape == (32, 64)

    # The detections fields were injected into the request:
    fields_sent = requests_mock.request_history[0].qs['fields'][0]
    assert 'detections.value' in fields_sent
    assert 'detections.geometry' in fields_sent


def test_fetch_image_masks_in_memory_only(client, requests_mock):
    """Masks are attached but not written when save_to_disk is False."""
    geom = _b64_polygon('POLYGON((0 0, 4096 0, 4096 4096, 0 4096, 0 0))')
    _register_metadata(
        requests_mock,
        detections={'data': [{'value': SegmentationLabels.ROAD, 'geometry': geom}]},
    )
    asset = client.fetch_image(
        '42', fields=['width'], save_to_disk=False, process_masks=['semantic']
    )
    assert asset.load_mask('semantic').max() == 1
    assert not asset.get_mask_path('semantic').exists()


def test_fetch_image_masks_without_detections(client, requests_mock, caplog):
    """Missing detection data skips masks with a warning."""
    _register_metadata(requests_mock, detections={'data': []})
    with caplog.at_level(logging.WARNING):
        asset = client.fetch_image(
            '42', fields=['width'], save_to_disk=False, process_masks=['semantic']
        )
    assert asset is not None and asset.semantic_map is None
    assert 'skipping masks' in caplog.text


def test_fetch_image_mask_processing_error(client, requests_mock, monkeypatch, caplog):
    """Exceptions during mask generation are logged, not raised."""
    _register_metadata(requests_mock, detections={'data': [{'value': 'x'}]})

    def boom(*args, **kwargs):
        raise RuntimeError('kaboom')

    monkeypatch.setattr(MapillaryClient, '_parse_mapillary_segmentation', boom)
    with caplog.at_level(logging.ERROR):
        asset = client.fetch_image(
            '42', fields=['width'], save_to_disk=False, process_masks=['semantic']
        )
    assert asset is not None
    assert 'Failed to process segmentation' in caplog.text


# ==========================================
# 9. fetch_images_by_ids
# ==========================================


def test_fetch_images_by_ids_collects_successes(client, requests_mock, no_sleep):
    """Only successfully fetched images end up in the collection."""
    _register_metadata(requests_mock, '1')
    _register_metadata(requests_mock, '2')
    requests_mock.get(f'{BASE_URL}/3', status_code=500)

    collection = client.fetch_images_by_ids(
        ['1', '2', '3'], fields=['width'], save_to_disk=False, max_workers=2
    )
    assert isinstance(collection, ImageCollection)
    assert sorted(collection.get_ids()) == ['1', '2']
    assert not client.save_dir.exists()


def test_fetch_images_by_ids_creates_dir_and_downloads(client, requests_mock):
    """Saving to disk creates the directory and writes files."""
    _register_metadata(requests_mock, '1')
    requests_mock.get('https://cdn.example/1.jpg', content=_jpeg_bytes())
    collection = client.fetch_images_by_ids(['1'], fields=['width'])
    assert client.save_dir.is_dir()
    assert collection[0].path.exists()


def test_fetch_images_by_ids_logs_exceptions(client, monkeypatch, caplog):
    """Exceptions raised by a worker are logged per image ID."""

    def flaky(image_id, *args, **kwargs):
        if image_id == 'bad':
            raise RuntimeError('worker exploded')
        return ImageAsset(path=f'{image_id}.jpg', id=image_id, allow_missing_file=True)

    monkeypatch.setattr(client, 'fetch_image', flaky)
    with caplog.at_level(logging.ERROR):
        collection = client.fetch_images_by_ids(
            ['good', 'bad'], fields=None, save_to_disk=False
        )
    assert collection.get_ids() == ['good']
    assert 'Exception for image bad' in caplog.text


# ==========================================
# 10. fetch_images_in_bbox
# ==========================================


def test_fetch_images_in_bbox_no_tiles(client, monkeypatch, caplog):
    """An empty tile list produces an empty collection and a warning."""
    monkeypatch.setattr(TileUtils, 'bbox_to_mapbox_tiles', lambda *a, **k: [])
    with caplog.at_level(logging.WARNING):
        result = client.fetch_images_in_bbox(BoundingBox(0, 0, 1, 1))
    assert len(result) == 0
    assert 'No tiles found' in caplog.text


def test_fetch_images_in_bbox_tile_metadata_only(client, requests_mock):
    """Without fields or downloads, only tile data is used (no Graph calls)."""
    bbox = BoundingBox(-118.15, 34.18, -118.14, 34.19)
    tiles = TileUtils.bbox_to_mapbox_tiles(bbox, zoom=14)
    assert len(tiles) == 2

    for i, (x, y, z) in enumerate(tiles):
        tile = _image_tile(
            [
                _point_feature(
                    500, 500, id=100 + i, creator_id=RAPID_CREATOR_ID, is_pano=True
                )
            ]
        )
        requests_mock.get(_tile_url(x, y, z), content=tile)

    result = client.fetch_images_in_bbox(bbox)
    assert sorted(result.get_ids()) == ['100', '101']
    assert requests_mock.call_count == 2
    assert all('latitude' in a.properties for a in result)


def test_fetch_images_in_bbox_enriches_with_metadata(client, requests_mock):
    """Requesting fields queries the Graph API and merges the results."""
    bbox = BoundingBox(-118.15, 34.18, -118.14, 34.19)
    tiles = TileUtils.bbox_to_mapbox_tiles(bbox, zoom=14)
    x, y, z = tiles[0]
    requests_mock.get(
        _tile_url(x, y, z),
        content=_image_tile([_point_feature(5, 5, id=7, creator_id=RAPID_CREATOR_ID)]),
    )
    for tx, ty, tz in tiles[1:]:
        requests_mock.get(_tile_url(tx, ty, tz), content=_image_tile([]))
    _register_metadata(requests_mock, '7', compass_angle=91.5)
    requests_mock.get('https://cdn.example/7.jpg', content=_jpeg_bytes())

    result = client.fetch_images_in_bbox(
        bbox, fields=['compass_angle'], save_to_disk=True
    )
    assert len(result) == 1
    asset = result[0]
    assert asset.properties['compass_angle'] == 91.5
    assert 'latitude' in asset.properties
    assert asset.path.exists()


def test_fetch_images_in_bbox_no_images_returns_early(client, requests_mock):
    """Tiles with no matching images return an empty collection."""
    requests_mock.get(TILE_URL_RE, content=_image_tile([]))
    result = client.fetch_images_in_bbox(
        BoundingBox(-118.15, 34.18, -118.14, 34.19), fields=['width']
    )
    assert len(result) == 0
    # No Graph API call was attempted:
    assert all('tiles.mapillary.com' in r.url for r in requests_mock.request_history)


def test_fetch_images_in_bbox_tile_thread_error(client, monkeypatch, caplog):
    """Unexpected worker errors are logged and do not abort the scan."""

    def boom(*args, **kwargs):
        raise RuntimeError('thread died')

    monkeypatch.setattr(client, '_get_tile_image_data', boom)
    with caplog.at_level(logging.ERROR):
        result = client.fetch_images_in_bbox(
            BoundingBox(-118.15, 34.18, -118.14, 34.19)
        )
    assert len(result) == 0
    assert 'Critical error in tile thread' in caplog.text


def test_fetch_images_in_bbox_split_quadrants_cover_parent(client, requests_mock):
    """Scanning the four split quadrants finds the same images as the parent."""
    bbox = BoundingBox(-118.20, 34.15, -118.10, 34.25)
    parent_tiles = set(TileUtils.bbox_to_mapbox_tiles(bbox, zoom=14))
    quadrant_tiles = set()
    for quad in bbox.split():
        quadrant_tiles.update(TileUtils.bbox_to_mapbox_tiles(quad, zoom=14))
    assert quadrant_tiles == parent_tiles

    # Every tile returns one RAPID image whose ID encodes its tile:
    for x, y, z in parent_tiles:
        requests_mock.get(
            _tile_url(x, y, z),
            content=_image_tile(
                [_point_feature(1, 1, id=x * 100000 + y, creator_id=RAPID_CREATOR_ID)]
            ),
        )

    parent = client.fetch_images_in_bbox(bbox, max_workers=4)
    merged = ImageCollection()
    for quad in bbox.split():
        merged.add(list(client.fetch_images_in_bbox(quad, max_workers=4)))

    assert len(parent) == len(parent_tiles)
    assert sorted(merged.get_ids()) == sorted(parent.get_ids())


# ==========================================
# Detection metadata (street-level object discovery)
# ==========================================
def test_fetch_detections_filters_values(client, requests_mock):
    """Detections come back as value/geometry pairs, filtered by label."""
    payload = {
        'id': '42',
        'detections': {
            'data': [
                {'value': 'object--vehicle--car', 'geometry': 'AAA='},
                {'value': 'nature--sky', 'geometry': 'BBB='},
                {'value': 'object--vehicle--truck'},  # no geometry: dropped
            ]
        },
    }
    requests_mock.get(f'{BASE_URL}/42', **_json_response(payload))
    cars = client.fetch_detections('42', values=['object--vehicle--car'])
    assert cars == [{'value': 'object--vehicle--car', 'geometry': 'AAA='}]
    assert requests_mock.last_request.qs['fields'] == [
        'detections.value,detections.geometry'
    ]
    everything = client.fetch_detections('42')
    assert [d['value'] for d in everything] == ['object--vehicle--car', 'nature--sky']


def test_fetch_detections_handles_failures(client, requests_mock, no_sleep):
    """A failing request or an image without detections yields an empty list."""
    requests_mock.get(f'{BASE_URL}/1', status_code=500)
    assert client.fetch_detections('1') == []
    requests_mock.get(f'{BASE_URL}/2', **_json_response({'id': '2'}))
    assert client.fetch_detections('2') == []


def test_decode_detection_polygons_normalises_and_flips():
    """Tile coordinates map to [0, 1] with y growing downwards."""
    b64 = _b64_polygon('POLYGON((0 0, 4096 0, 4096 2048, 0 2048, 0 0))')
    rings = MapillaryClient.decode_detection_polygons(b64)
    assert len(rings) == 1
    xs = {round(x, 3) for x, _ in rings[0]}
    ys = {round(y, 3) for _, y in rings[0]}
    assert xs == {0.0, 1.0}
    assert ys == {0.5, 1.0}  # tile y=2048 (top half) becomes image y=0.5


def test_decode_detection_polygons_multipolygon_and_garbage():
    """Multi-polygons yield one ring per part; garbage yields nothing."""
    b64 = _b64_polygon(
        'MULTIPOLYGON(((0 0, 100 0, 100 100, 0 0)), '
        '((200 200, 300 200, 300 300, 200 200)))'
    )
    assert len(MapillaryClient.decode_detection_polygons(b64)) == 2
    assert MapillaryClient.decode_detection_polygons('not base64!') == []


def test_get_image_url(client, requests_mock):
    """The thumbnail URL for the requested size is returned; bad sizes raise."""
    requests_mock.get(
        f'{BASE_URL}/42',
        **_json_response({'id': '42', 'thumb_1024_url': 'https://x/1024'}),
    )
    assert client.get_image_url('42', size='1024') == 'https://x/1024'
    assert requests_mock.last_request.qs['fields'] == ['thumb_1024_url']
    with pytest.raises(ValueError):
        client.get_image_url('42', size='4096')


# ==========================================
# Coverage tiles (survey sequences for the map)
# ==========================================
def _sequence_tile(features):
    return mapbox_vector_tile.encode([{'name': 'sequence', 'features': features}])


def test_fetch_sequence_lines_filters_and_flips(client, requests_mock):
    ms = 1_760_000_000_000  # 2025-10-09
    rapid = {
        'geometry': 'LINESTRING(0 4096, 100 4000, 200 3900)',
        'properties': {'creator_id': RAPID_CREATOR_ID, 'captured_at': ms},
    }
    other = {
        'geometry': 'LINESTRING(0 0, 10 10)',
        'properties': {'creator_id': 1, 'captured_at': ms},
    }
    multi = {
        'geometry': 'MULTILINESTRING((0 0, 1 1), (5 5, 6 6, 7 7))',
        'properties': {'creator_id': RAPID_CREATOR_ID, 'captured_at': ms + 86_400_000},
    }
    undated = {
        'geometry': 'LINESTRING(3 3, 4 4)',
        'properties': {'creator_id': RAPID_CREATOR_ID},
    }
    requests_mock.get(
        _tile_url(178, 365, 10), content=_sequence_tile([rapid, other, multi, undated])
    )
    data = client.fetch_sequence_lines(10, 178, 365)
    assert data['extent'] == 4096 and data['count'] == 4
    assert data['lines'][0] == [0, 0, 100, 96, 200, 196]  # y flipped to grow downwards
    assert data['lines'][1] == [0, 4096, 1, 4095]
    assert (data['first'], data['last']) == ('2025-10-09', '2025-10-10')
    everyone = client.fetch_sequence_lines(10, 178, 365, filter_rapid_only=False)
    assert everyone['count'] == 5
    dated = client.fetch_sequence_lines(10, 178, 365, start_date='2025-10-10')
    assert dated['count'] == 2 and dated['first'] == '2025-10-10'  # undated dropped
    with pytest.raises(ValueError):
        client.fetch_sequence_lines(15, 0, 0)


def test_fetch_sequence_lines_handles_empty_and_failed_tiles(
    client, requests_mock, caplog
):
    requests_mock.get(_tile_url(1, 1, 8), content=_image_tile([]))
    assert client.fetch_sequence_lines(8, 1, 1)['lines'] == []
    requests_mock.get(_tile_url(2, 2, 8), status_code=500)
    with caplog.at_level(logging.WARNING):
        assert client.fetch_sequence_lines(8, 2, 2)['count'] == 0
    assert 'Could not read coverage tile 8/2/2' in caplog.text


# ==========================================
# Map features (pre-triangulated point objects)
# ==========================================
MAP_FEATURES_URL = f'{BASE_URL}/map_features'
POLE = 'object--support--utility-pole'
HYDRANT = 'object--fire-hydrant'


def _feature(fid, lon, lat, value=POLE, first=None, last=None, **extra):
    """Build one raw ``map_features`` record as the Graph API returns it."""
    record = {
        'id': fid,
        'object_value': value,
        'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
        'aligned_direction': 69.79,
        'first_seen_at': first,
        'last_seen_at': last,
    }
    record.update(extra)
    return record


def _bbox_of(request) -> tuple[float, ...]:
    """Return the bbox query parameter of a recorded request as floats."""
    return tuple(float(v) for v in request.qs['bbox'][0].split(','))


def test_fetch_map_features_basic(client, requests_mock):
    """Point coordinates and requested fields are flattened into dicts."""
    requests_mock.get(
        MAP_FEATURES_URL,
        **_json_response(
            {
                'data': [
                    _feature(
                        2519029251914551,
                        -117.49456,
                        47.716953,
                        first='2026-08-27T03:59:39+0000',
                        last='2026-09-15T23:37:17+0000',
                    ),
                    _feature('7', -117.4, 47.7, value=HYDRANT, aligned_direction=None),
                    # A record without a point is skipped:
                    {'id': '8', 'object_value': HYDRANT, 'geometry': None},
                ]
            }
        ),
    )
    bbox = BoundingBox(-117.50, 47.71, -117.49, 47.72)
    out = client.fetch_map_features(bbox, [POLE, HYDRANT])
    assert [f['id'] for f in out] == ['2519029251914551', '7']
    first = out[0]
    assert (first['lon'], first['lat']) == (-117.49456, 47.716953)
    assert first['object_value'] == POLE
    assert first['aligned_direction'] == 69.79
    assert first['first_seen_at'] == '2026-08-27T03:59:39+0000'
    assert first['last_seen_at'] == '2026-09-15T23:37:17+0000'
    assert first['image_ids'] == []
    assert out[1]['aligned_direction'] is None
    # The token travels in the query string, never in the URL path:
    sent = requests_mock.last_request
    assert sent.qs['access_token'] == [TOKEN.lower()]
    assert sent.qs['limit'] == ['2000']
    assert sent.qs['fields'] == [
        'id,object_value,geometry,aligned_direction,first_seen_at,last_seen_at'
    ]


def test_fetch_map_features_request_formatting(client, requests_mock):
    """object_values are comma-joined and bbox is min_lon,min_lat,max_lon,max_lat."""
    requests_mock.get(MAP_FEATURES_URL, **_json_response({'data': []}))
    bbox = BoundingBox(-118.15, 34.18, -118.14, 34.19)
    assert client.fetch_map_features(bbox, [POLE, HYDRANT], limit=50) == []
    qs = requests_mock.last_request.qs
    assert qs['object_values'] == [f'{POLE},{HYDRANT}']
    assert qs['bbox'] == ['-118.15,34.18,-118.14,34.19']
    assert qs['limit'] == ['50']


def test_fetch_map_features_custom_fields_and_images(client, requests_mock):
    """Extra fields are copied verbatim and image ids are extracted."""
    requests_mock.get(
        MAP_FEATURES_URL,
        **_json_response(
            {
                'data': [
                    _feature(
                        '1',
                        1.0,
                        2.0,
                        images={'data': [{'id': 11}, {'id': '12'}]},
                        custom='x',
                    ),
                    _feature('2', 3.0, 4.0, images=['21', 22]),
                ]
            }
        ),
    )
    out = client.fetch_map_features(
        BoundingBox(0, 0, 5, 5), [POLE], fields=['object_value', 'images', 'custom']
    )
    # id and geometry are always requested so results can be keyed/located:
    fields = requests_mock.last_request.qs['fields'][0].split(',')
    assert set(fields) == {'object_value', 'images', 'custom', 'id', 'geometry'}
    assert out[0]['image_ids'] == ['11', '12'] and out[0]['custom'] == 'x'
    assert out[1]['image_ids'] == ['21', '22']
    assert 'images' not in out[0] and 'geometry' not in out[0]


def test_fetch_map_features_date_overlap(client, requests_mock):
    """Features are kept when their seen-span overlaps the requested window."""
    data = [
        _feature('spans', 0, 0, first='2026-08-27T03:59:39+0000',
                 last='2026-09-15T23:37:17+0000'),
        _feature('before', 0, 0, first='2026-05-01T00:00:00+0000',
                 last='2026-06-01T00:00:00+0000'),
        _feature('after', 0, 0, first='2026-10-05T00:00:00+0000',
                 last='2026-10-06T00:00:00+0000'),
        _feature('undated', 0, 0),
        _feature('last_only', 0, 0, last='2026-09-02T00:00:00+0000'),
        _feature('covers', 0, 0, first='2026-01-01T00:00:00+0000',
                 last='2026-12-31T00:00:00+0000'),
    ]  # fmt: skip
    requests_mock.get(MAP_FEATURES_URL, **_json_response({'data': data}))
    bbox = BoundingBox(-1, -1, 1, 1)
    kept = client.fetch_map_features(
        bbox, [POLE], start_date='2026-09-01', end_date='2026-09-30'
    )
    assert [f['id'] for f in kept] == ['spans', 'undated', 'last_only', 'covers']
    lower_only = client.fetch_map_features(bbox, [POLE], start_date='2026-09-01')
    assert [f['id'] for f in lower_only] == [
        'spans', 'after', 'undated', 'last_only', 'covers'
    ]  # fmt: skip
    unbounded = client.fetch_map_features(bbox, [POLE])
    assert len(unbounded) == 6


def test_fetch_map_features_splits_saturated_box(client, requests_mock):
    """A full response triggers four quadrant queries, merged by id."""
    parent = (0.0, 0.0, 2.0, 2.0)
    quadrant_data = {
        (0.0, 0.0, 1.0, 1.0): [_feature('a', 0.5, 0.5), _feature('d', 0.2, 0.2)],
        (1.0, 0.0, 2.0, 1.0): [_feature('b', 1.5, 0.5)],
        (0.0, 1.0, 1.0, 2.0): [],
        (1.0, 1.0, 2.0, 2.0): [_feature('c', 1.5, 1.5), _feature('e', 1.9, 1.9)],
    }

    def respond(request, context):
        box = _bbox_of(request)
        if box == parent:  # exactly ``limit`` items -> saturated
            return {
                'data': [
                    _feature('a', 0.5, 0.5),
                    _feature('b', 1.5, 0.5),
                    _feature('c', 1.5, 1.5),
                ]
            }
        return {'data': quadrant_data[box]}

    requests_mock.get(
        MAP_FEATURES_URL, json=respond, headers={'Content-Type': 'application/json'}
    )
    out = client.fetch_map_features(BoundingBox(*parent), [POLE], limit=3)
    assert sorted(f['id'] for f in out) == ['a', 'b', 'c', 'd', 'e']
    assert len(out) == 5  # deduplicated by id
    boxes = [_bbox_of(r) for r in requests_mock.request_history]
    assert boxes[0] == parent
    sub_boxes = boxes[1:]
    assert len(sub_boxes) == 4 and set(sub_boxes) == set(quadrant_data)
    # The four sub-boxes tile the parent exactly:
    parent_area = (parent[2] - parent[0]) * (parent[3] - parent[1])
    assert sum((b[2] - b[0]) * (b[3] - b[1]) for b in sub_boxes) == parent_area
    assert min(b[0] for b in sub_boxes) == parent[0]
    assert max(b[2] for b in sub_boxes) == parent[2]
    assert min(b[1] for b in sub_boxes) == parent[1]
    assert max(b[3] for b in sub_boxes) == parent[3]


def test_fetch_map_features_respects_max_depth(client, requests_mock, caplog):
    """Splitting stops after ``max_depth`` levels with a warning."""
    always_full = {'data': [_feature('a', 0.5, 0.5), _feature('b', 0.6, 0.6)]}
    requests_mock.get(MAP_FEATURES_URL, **_json_response(always_full))
    with caplog.at_level(logging.WARNING):
        out = client.fetch_map_features(
            BoundingBox(0, 0, 2, 2), [POLE], limit=2, max_depth=1
        )
    assert requests_mock.call_count == 1 + 4  # parent + one level of quadrants
    assert len(out) == 2
    assert 'max_depth was reached' in caplog.text


def test_fetch_map_features_http_error_on_sub_box(client, requests_mock, caplog):
    """A failing sub-box is logged as a warning and skipped, not raised."""
    parent = (0.0, 0.0, 2.0, 2.0)
    bad = (1.0, 0.0, 2.0, 1.0)

    def respond(request, context):
        box = _bbox_of(request)
        if box == parent:
            return {'data': [_feature('a', 0.5, 0.5), _feature('c', 1.5, 1.5)]}
        if box == bad:
            context.status_code = 500
            return {'error': 'boom'}
        if box == (0.0, 0.0, 1.0, 1.0):
            return {'data': [_feature('a', 0.5, 0.5), _feature('d', 0.1, 0.1)]}
        return {'data': []}

    requests_mock.get(
        MAP_FEATURES_URL, json=respond, headers={'Content-Type': 'application/json'}
    )
    with caplog.at_level(logging.WARNING):
        out = client.fetch_map_features(BoundingBox(*parent), [POLE], limit=2)
    assert sorted(f['id'] for f in out) == ['a', 'c', 'd']
    assert 'Could not fetch map features for bbox 1.0,0.0,2.0,1.0' in caplog.text
    assert TOKEN not in caplog.text


def test_fetch_map_features_top_level_error(client, requests_mock, caplog):
    """A failing top-level request yields an empty list and a warning."""
    requests_mock.get(MAP_FEATURES_URL, exc=requests.exceptions.ConnectionError)
    with caplog.at_level(logging.WARNING):
        assert client.fetch_map_features(BoundingBox(0, 0, 1, 1), [POLE]) == []
    assert 'Could not fetch map features' in caplog.text


def test_fetch_map_features_validates_arguments(client, requests_mock):
    """Empty object_values and non-positive limits are rejected offline."""
    with pytest.raises(ValueError):
        client.fetch_map_features(BoundingBox(0, 0, 1, 1), [])
    with pytest.raises(ValueError):
        client.fetch_map_features(BoundingBox(0, 0, 1, 1), [POLE], limit=0)
    assert requests_mock.call_count == 0


def test_is_map_feature_value():
    """Exact members and any traffic-sign value count as map features."""
    assert isinstance(MAP_FEATURE_VALUES, frozenset)
    assert POLE in MAP_FEATURE_VALUES and HYDRANT in MAP_FEATURE_VALUES
    assert all(v.startswith('object--') for v in MAP_FEATURE_VALUES)
    assert is_map_feature_value(POLE)
    assert is_map_feature_value('object--traffic-light--pedestrians')
    assert is_map_feature_value('object--traffic-sign--stop--g1')
    assert is_map_feature_value('object--traffic-sign--regulatory--no-parking--g1')
    assert not is_map_feature_value('object--traffic-sign')
    assert not is_map_feature_value('object--vehicle--car')
    assert not is_map_feature_value('human--person--individual')
    assert not is_map_feature_value('construction--structure--building')
    assert not is_map_feature_value('')


def test_fetch_sequence_lines_reads_gzip_and_skips_points(client, requests_mock):
    ms = 1_760_000_000_000
    line = {
        'geometry': 'LINESTRING(0 4096, 100 4000)',
        'properties': {'creator_id': RAPID_CREATOR_ID, 'captured_at': ms},
    }
    point = {
        'geometry': 'POINT(5 5)',
        'properties': {'creator_id': RAPID_CREATOR_ID, 'captured_at': ms},
    }
    requests_mock.get(
        _tile_url(3, 3, 8), content=gzip.compress(_sequence_tile([line, point]))
    )
    data = client.fetch_sequence_lines(8, 3, 3)
    assert data['count'] == 1 and data['lines'][0] == [0, 0, 100, 96]


def test_detection_and_feature_parsers_skip_malformed_records(monkeypatch):
    from rapidtools.data_sources import mapillary_client as module

    decoded = {
        'l': {
            'extent': 4096,
            'features': [
                {'geometry': {'type': 'Point', 'coordinates': [1, 2]}},
                {'geometry': {'type': 'Polygon', 'coordinates': []}},
                {
                    'geometry': {
                        'type': 'MultiPolygon',
                        'coordinates': [[[[0, 0], [4096, 0], [4096, 4096]]]],
                    }
                },
            ],
        }
    }
    monkeypatch.setattr(module.mapbox_vector_tile, 'decode', lambda raw: decoded)
    rings = MapillaryClient.decode_detection_polygons(base64.b64encode(b'x').decode())
    assert rings == [[(0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]]
    feature = {'geometry': {'type': 'Point', 'coordinates': [1.0, 2.0]}}
    assert MapillaryClient._parse_map_feature(feature, []) is None


def test_parse_segmentation_skips_undecodable_tiles(pano_asset, monkeypatch):
    from rapidtools.data_sources import mapillary_client as module

    pano_asset.properties['detections'] = {
        'data': [
            {'value': SegmentationLabels.SKY, 'geometry': _b64_polygon('POLYGON EMPTY')}
        ]
    }
    monkeypatch.setattr(module.mapbox_vector_tile, 'decode', lambda raw: {})
    mask, seg_map = MapillaryClient._parse_mapillary_segmentation(pano_asset)
    assert not mask.any()  # the label is registered, nothing is rasterised
