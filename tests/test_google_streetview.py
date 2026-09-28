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

"""Tests for the keyless Google Street View client (mocked HTTP)."""

import base64
import json
import re
import struct
from io import BytesIO

import numpy as np
import pytest
import requests
from PIL import Image
from shapely.geometry import LineString, Point, box

from rapidtools.data_sources import GoogleStreetViewClient, StreetViewPanorama
from rapidtools.data_sources.google_streetview import (
    PHOTOMETA_URL,
    SINGLE_IMAGE_SEARCH_URL,
    bearing_deg,
    decode_depth_map,
    haversine_m,
    relative_angle,
)

TILE_URL = re.compile(r'https://streetviewpixels-pa\.googleapis\.com/v1/tile\?.*')
PANO_ID = 'u3PxkEsnYto4l1N3yTmO_Q'
CAM_LAT, CAM_LON = 34.18791374, -118.13529398


def _pano_node(pano_id=PANO_ID, depth=None, links=True, heading=20.8):
    """Build the ``[1][k]`` metadata node exactly as Google returns it."""
    cam = [
        [None, None, CAM_LAT, CAM_LON],
        [392.3, None, 358.5],
        [heading, 95.9, 1.7],
        None,
        'US',
    ]
    link_list = [[[2, pano_id], None, cam]]
    if links:
        link_list += [
            [
                [2, 'LINK_A'],
                None,
                [[None, None, CAM_LAT - 0.0001, CAM_LON], [391.5], [21.0, 96.0, 2.0]],
            ],
            [
                [2, 'LINK_B'],
                None,
                [[None, None, CAM_LAT + 0.0009, CAM_LON], [393.0], [359.0]],
            ],
            [[2, 'BROKEN'], None, [[None, None, None, None]]],
        ]
    node = [
        [1],
        [2, pano_id],
        [
            2,
            2,
            [8192, 16384],
            [[[[256, 512]], [[512, 1024]], [[1024, 2048]], [[2048, 4096]]]],
            [512, 512],
        ],
        [None, None, [['2512 Catherine Rd', 'en'], ['Altadena, California', 'en']]],
        None,
        [
            [
                [1],
                cam,
                None,
                [link_list],
                None,
                [[2], [[], 1, depth]] if depth is not None else None,
            ]
        ],
        [3, 4, 1, None, None, None, None, [2025, 11]],
    ]
    return node


def _photometa_body(node):
    return ")]}'\n" + json.dumps([[], [node]])


def _jpeg_bytes(size=(512, 512), color=(120, 120, 220)) -> bytes:
    buf = BytesIO()
    Image.new('RGB', size, color).save(buf, format='JPEG')
    return buf.getvalue()


def _depth_string(width=4, height=2, planes=None, indices=None) -> str:
    """Encode a tiny plane-based depth map like Google does."""
    planes = planes or [(0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 1.0, 2.5)]
    indices = indices or [0, 1, 1, 0, 1, 1, 1, 1]
    header = bytes([8]) + struct.pack('<HHHH', len(planes), width, height, 9)
    body = bytes(indices)
    for n in planes:
        body += struct.pack('<ffff', *n)
    return base64.urlsafe_b64encode(header + body).decode().rstrip('=')


# ==========================================
# 1. Geometry helpers
# ==========================================


def test_geometry_helpers():
    """Haversine, bearing and relative angle behave as documented."""
    assert round(haversine_m(0.0, 0.0, 0.0, 1.0)) == 111195
    assert haversine_m(CAM_LAT, CAM_LON, CAM_LAT, CAM_LON) == 0.0
    assert bearing_deg(0.0, 0.0, 0.0, 1.0) == pytest.approx(90.0)
    assert bearing_deg(0.0, 0.0, 1.0, 0.0) == pytest.approx(0.0)
    assert bearing_deg(0.0, 0.0, -1.0, 0.0) == pytest.approx(180.0)
    assert bearing_deg(0.0, 0.0, 0.0, -1.0) == pytest.approx(270.0)
    assert relative_angle(90.0, 0.0) == 90.0
    assert relative_angle(350.0, 10.0) == -20.0
    assert relative_angle(180.0, 0.0) == 180.0
    assert relative_angle(0.0, 180.0) == 180.0


def test_panorama_dataclass():
    """Size fallbacks and the capture date property."""
    pano = StreetViewPanorama(id='p', lat=1.0, lon=2.0)
    assert pano.size_at_zoom(0) == (512, 256)
    assert pano.size_at_zoom(3) == (4096, 2048)
    assert pano.capture_date is None
    pano.sizes[1] = (1000, 500)
    pano.date = (2025, 3)
    assert pano.size_at_zoom(1) == (1000, 500)
    assert pano.capture_date == '2025-03'


# ==========================================
# 2. Depth maps
# ==========================================


def test_decode_depth_map_planes_and_sky():
    """Pixels with plane 0 are sky (inf); others follow the plane distance."""
    depth = decode_depth_map(_depth_string())
    assert depth.shape == (2, 4)
    assert depth.dtype == np.float32
    assert np.isinf(depth[0, 0]) and np.isinf(depth[0, 3])
    finite = depth[np.isfinite(depth)]
    assert finite.size == 6
    # Plane z=2.5 seen from rays with cos(theta) != 0 -> depth = 2.5 / |cos|:
    assert np.all(finite >= 2.5)
    # Out-of-range plane indices are treated as sky:
    weird = decode_depth_map(_depth_string(indices=[5, 5, 5, 5, 1, 1, 1, 1]))
    assert np.isinf(weird[0]).all() and np.isfinite(weird[1]).all()
    # Padding-less strings and short strings:
    assert decode_depth_map(_depth_string() + '==').shape == (2, 4)
    with pytest.raises(ValueError, match='too short'):
        decode_depth_map(base64.urlsafe_b64encode(b'\x01\x02').decode())


# ==========================================
# 3. Lookups (mocked HTTP)
# ==========================================


def test_find_panorama_parses_response(requests_mock):
    """SingleImageSearch is posted as JSON+protobuf and parsed into metadata."""
    post = requests_mock.post(SINGLE_IMAGE_SEARCH_URL, json=[[0], _pano_node(), []])
    client = GoogleStreetViewClient()
    pano = client.find_panorama(34.1878, -118.135, radius_m=40)
    assert post.last_request.headers['Content-Type'] == 'application/json+protobuf'
    body = json.loads(post.last_request.body)
    assert body[1] == [[None, None, 34.1878, -118.135], 40]
    assert pano.id == PANO_ID
    assert (pano.lat, pano.lon) == (CAM_LAT, CAM_LON)
    assert (pano.heading, pano.pitch, pano.roll) == (20.8, 95.9, 1.7)
    assert pano.elevation == 392.3
    assert pano.date == (2025, 11) and pano.capture_date == '2025-11'
    assert pano.address == ['2512 Catherine Rd', 'Altadena, California']
    assert pano.sizes[3] == (4096, 2048) and pano.size_at_zoom(0) == (512, 256)
    assert pano.tile_size == 512
    # The panorama itself and links without coordinates are excluded:
    assert [link[0] for link in pano.links] == ['LINK_A', 'LINK_B']
    assert pano.links[1][3] == 359.0
    assert pano.depth_map_b64 is None


def test_find_panorama_no_result_and_errors(requests_mock, caplog):
    """Empty responses, HTTP errors and network failures yield None."""
    client = GoogleStreetViewClient()
    requests_mock.post(SINGLE_IMAGE_SEARCH_URL, json=[[0], [[1]], []])
    with caplog.at_level('INFO'):
        assert client.find_panorama(0.0, 0.0) is None
    assert 'No Street View panorama within' in caplog.text
    requests_mock.post(SINGLE_IMAGE_SEARCH_URL, status_code=500)
    assert client.find_panorama(0.0, 0.0) is None
    requests_mock.post(SINGLE_IMAGE_SEARCH_URL, exc=requests.ConnectionError('x'))
    assert client.find_panorama(0.0, 0.0) is None
    requests_mock.post(SINGLE_IMAGE_SEARCH_URL, text='not json')
    assert client.find_panorama(0.0, 0.0) is None
    # A node with an id but without coordinates is unusable:
    requests_mock.post(SINGLE_IMAGE_SEARCH_URL, json=[[0], [[1], [2, 'X']], []])
    assert client.find_panorama(0.0, 0.0) is None


def test_get_panorama_metadata(requests_mock, caplog):
    """photometa responses are stripped of their XSSI prefix and parsed."""
    depth = _depth_string()
    requests_mock.get(PHOTOMETA_URL, text=_photometa_body(_pano_node(depth=depth)))
    client = GoogleStreetViewClient()
    pano = client.get_panorama_metadata(PANO_ID)
    assert requests_mock.last_request.qs['pb'][0].startswith('!1m4!1smaps_sv.tactile')
    assert PANO_ID.lower() in requests_mock.last_request.qs['pb'][0]
    assert pano.id == PANO_ID and pano.depth_map_b64 == depth
    assert decode_depth_map(pano.depth_map_b64).shape == (2, 4)

    # Plain JSON (no prefix) also works; empty nodes and errors yield None:
    requests_mock.get(PHOTOMETA_URL, text=json.dumps([[], [_pano_node(links=False)]]))
    assert client.get_panorama_metadata(PANO_ID).links == []
    requests_mock.get(
        PHOTOMETA_URL, text=")]}'\n" + json.dumps([[], [[[2], [2, 'X']]]])
    )
    with caplog.at_level('WARNING'):
        assert client.get_panorama_metadata('X') is None
    assert 'No metadata returned' in caplog.text
    requests_mock.get(PHOTOMETA_URL, text=json.dumps([[], []]))
    assert client.get_panorama_metadata('X') is None
    requests_mock.get(PHOTOMETA_URL, status_code=404)
    assert client.get_panorama_metadata('X') is None


# ==========================================
# 4. Downloads and cropping
# ==========================================


def test_download_panorama_stitches_tiles(requests_mock):
    """Tiles are requested per grid cell and stitched to the metadata size."""
    requests_mock.get(TILE_URL, content=_jpeg_bytes())
    client = GoogleStreetViewClient(max_workers=3)
    pano = GoogleStreetViewClient._parse_pano_node(_pano_node(), PANO_ID)
    image = client.download_panorama(pano, zoom=1)
    assert image.size == (1024, 512)
    urls = sorted(r.url for r in requests_mock.request_history)
    assert len(urls) == 2
    assert all(f'panoid={PANO_ID}' in u and 'zoom=1' in u for u in urls)
    # A bare pano ID uses the canonical sizes:
    requests_mock.reset_mock()
    assert client.download_panorama(PANO_ID, zoom=0).size == (512, 256)
    assert len(requests_mock.request_history) == 1
    with pytest.raises(ValueError, match='zoom'):
        client.download_panorama(pano, zoom=6)


def test_download_panorama_failures(requests_mock, caplog):
    """Missing tiles are left black; when every tile fails the result is None."""
    client = GoogleStreetViewClient(max_workers=2)
    pano = GoogleStreetViewClient._parse_pano_node(_pano_node(), PANO_ID)
    requests_mock.get(TILE_URL, status_code=403)
    with caplog.at_level('WARNING'):
        assert client.download_panorama(pano, zoom=1) is None
    assert 'Failed to download tile' in caplog.text
    assert 'No tiles could be downloaded' in caplog.text

    calls = {'n': 0}

    def flaky(request, context):
        calls['n'] += 1
        if calls['n'] == 1:
            context.status_code = 500
            return b''
        return _jpeg_bytes(color=(255, 0, 0))

    requests_mock.get(TILE_URL, content=flaky)
    image = client.download_panorama(pano, zoom=1)
    assert image.size == (1024, 512)
    pixels = {image.getpixel((10, 10)), image.getpixel((600, 10))}
    assert (0, 0, 0) in pixels  # one tile black
    assert any(p[0] > 200 for p in pixels)  # one tile red


def test_view_angles_and_crop_geometry():
    """Angles are relative to the heading and crops follow them."""
    client = GoogleStreetViewClient()
    pano = StreetViewPanorama(id='p', lat=0.0, lon=0.0, heading=0.0)
    assert client.view_angles(pano, Point(0.001, 0.0)) == [pytest.approx(90.0)]
    line_angles = client.view_angles(pano, LineString([(0.001, 0.0), (0.001, 0.001)]))
    assert len(line_angles) == 2 and all(0 < a <= 90 for a in line_angles)
    square = box(0.0009, -0.0001, 0.0011, 0.0001)  # due east
    angles = client.view_angles(pano, square)
    assert len(angles) == 4 and all(80 < a < 100 for a in angles)

    image = Image.new('RGB', (720, 360))
    for x in range(720):
        # Encode the column index in the red channel (x / 3).
        image.paste((x // 3, 0, 0), (x, 0, x + 1, 360))

    crop, (start, end) = client.crop_to_geometry(image, pano, square)
    assert start < 90 < end and end - start >= 30
    # Column at the crop centre must correspond to ~90 degrees -> x = 540:
    centre_red = crop.getpixel((crop.size[0] // 2, 0))[0]
    assert abs(centre_red - 540 // 3) <= 3
    assert crop.size[1] == 360

    # Minimum FOV kicks in for tiny geometries:
    tiny, (s2, e2) = client.crop_to_geometry(
        image, pano, Point(0.001, 0.0), fov_buffer_deg=0.0, min_fov_deg=40.0
    )
    assert e2 - s2 == pytest.approx(40.0)
    assert tiny.size[0] == pytest.approx(80, abs=2)

    # Vertical trimming:
    trimmed, _ = client.crop_to_geometry(
        image, pano, square, vertical_crop=(0.25, 0.75)
    )
    assert trimmed.size[1] == 180
    with pytest.raises(ValueError, match='vertical_crop'):
        client.crop_to_geometry(image, pano, square, vertical_crop=(0.8, 0.2))


def test_crop_wraps_around_the_seam():
    """Assets directly behind the camera span the panorama seam."""
    client = GoogleStreetViewClient()
    pano = StreetViewPanorama(id='p', lat=0.0, lon=0.0, heading=0.0)
    behind = box(-0.0001, -0.0011, 0.0001, -0.0009)  # due south (+/-180)
    image = Image.new('RGB', (360, 180))
    image.paste((255, 0, 0), (0, 0, 20, 180))  # left edge red
    image.paste((0, 0, 255), (340, 0, 360, 180))  # right edge blue
    crop, (start, end) = client.crop_to_geometry(image, pano, behind)
    assert end - start >= 30
    colors = {crop.getpixel((x, 0)) for x in range(crop.size[0])}
    assert (255, 0, 0) in colors and (0, 0, 255) in colors

    # Requesting more than the full width returns a copy of the panorama:
    assert client._crop_columns(image, -10, 400).size == image.size
    assert client._crop_columns(image, 350, 370).size == (20, 180)
