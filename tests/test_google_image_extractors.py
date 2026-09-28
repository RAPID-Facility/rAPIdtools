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

"""Tests for the Google pipeline components (orthomosaic and street view)."""

import logging
import re
import threading
from io import BytesIO

import numpy as np
import pytest
import rasterio
import requests_mock
from PIL import Image
from shapely.geometry import box

from rapidtools.core import (
    BoundingBox,
    OperationCancelled,
    PhysicalAsset,
    PhysicalAssetCollection,
)
from rapidtools.data_sources import StreetViewPanorama
from rapidtools.processing import (
    BingOrthomosaicExtractor,
    GoogleOrthomosaicExtractor,
    GoogleStreetViewImageExtractor,
)
from rapidtools.processing.image_extractors import TileOrthomosaicExtractor

GOOGLE_TILE_URL = re.compile(
    r'https://mt[0-3]\.google\.com/vt/lyrs=[sy]&x=\d+&y=\d+&z=\d+'
)
CAM_LAT, CAM_LON = 34.18791374, -118.13529398


def _jpeg_bytes(color=(200, 30, 30)) -> bytes:
    buf = BytesIO()
    Image.new('RGB', (256, 256), color).save(buf, format='JPEG')
    return buf.getvalue()


# ==========================================
# 1. GoogleOrthomosaicExtractor
# ==========================================


def test_google_ortho_init_and_urls():
    """Configuration, layer validation and subdomain rotation."""
    ext = GoogleOrthomosaicExtractor(zoom_level=18, max_workers=3, layer='y')
    assert (ext.zoom_level, ext.max_workers, ext.layer) == (18, 3, 'y')
    assert ext.PROVIDER_NAME == 'Google'
    assert ext._tile_url(1, 2) == 'https://mt3.google.com/vt/lyrs=y&x=1&y=2&z=18'
    assert GoogleOrthomosaicExtractor()._tile_url(0, 0).startswith('https://mt0')
    with pytest.raises(ValueError, match='Unsupported Google tile layer'):
        GoogleOrthomosaicExtractor(layer='h')
    with pytest.raises(NotImplementedError):
        TileOrthomosaicExtractor()._tile_url(0, 0)
    # Bing keeps its quadkey-based URL:
    assert (
        BingOrthomosaicExtractor(zoom_level=3)
        ._tile_url(3, 5)
        .endswith('tiles/a213.jpeg?g=1')
    )


def test_google_ortho_stitches_geotiff(tmp_path):
    """Google tiles are stitched into an EPSG:4326 GeoTIFF."""
    region = BoundingBox(min_x=-118.5, min_y=34.0, max_x=-117.5, max_y=34.5)
    ext = GoogleOrthomosaicExtractor(zoom_level=8, max_workers=2)
    with requests_mock.Mocker() as m:
        m.get(GOOGLE_TILE_URL, content=_jpeg_bytes((20, 200, 20)))
        path = ext(region, tmp_path / 'google')
        hosts = {r.hostname for r in m.request_history}
    assert path.name == 'google.tiff' and path.exists()
    assert hosts <= {f'mt{i}.google.com' for i in range(4)}
    with rasterio.open(path) as src:
        assert src.count == 3 and src.crs.to_epsg() == 4326
        b = src.bounds
        assert (b.left, b.bottom, b.right, b.top) == pytest.approx(
            region.bounds, abs=1e-9
        )
        data = src.read()
        assert data[1].min() > 150 and data[0].max() < 100


def test_google_ortho_failures_and_cancel(tmp_path, caplog):
    """Failed tiles stay black; cancellation aborts the download."""
    region = BoundingBox(min_x=-118.5, min_y=34.0, max_x=-117.5, max_y=34.5)
    ext = GoogleOrthomosaicExtractor(zoom_level=7)
    with requests_mock.Mocker() as m:
        m.get(GOOGLE_TILE_URL, status_code=404)
        with caplog.at_level(logging.WARNING):
            path = ext(region, tmp_path / 'black.tiff')
    assert 'Network error fetching tile' in caplog.text
    with rasterio.open(path) as src:
        assert src.read().max() == 0

    event = threading.Event()
    event.set()
    ext = GoogleOrthomosaicExtractor(zoom_level=8, cancel_event=event)
    with requests_mock.Mocker() as m:
        m.get(GOOGLE_TILE_URL, content=_jpeg_bytes())
        with pytest.raises(OperationCancelled, match='Google tile download'):
            ext(region, tmp_path / 'cancelled.tiff')
    assert not (tmp_path / 'cancelled.tiff').exists()


# ==========================================
# 2. GoogleStreetViewImageExtractor
# ==========================================


class FakeStreetViewClient:
    """Deterministic stand-in for GoogleStreetViewClient."""

    def __init__(self, panos=None, fail_download=False, depth=None):
        self.panos = panos if panos is not None else {}
        self.fail_download = fail_download
        self.depth = depth
        self.calls: list = []

    def find_panorama(self, lat, lon, radius_m=50.0):
        self.calls.append(('find', round(lat, 5), round(lon, 5), radius_m))
        return self.panos.get('nearest')

    def get_panorama_metadata(self, pano_id):
        self.calls.append(('meta', pano_id))
        pano = self.panos.get(pano_id)
        if pano is not None and self.depth is not None:
            pano.depth_map_b64 = self.depth
        return pano

    def download_panorama(self, pano, zoom=3):
        self.calls.append(('download', pano.id, zoom))
        if self.fail_download:
            return None
        return Image.new('RGB', (512 * 2**zoom, 256 * 2**zoom), (90, 90, 200))

    def crop_to_geometry(self, image, pano, geometry, **kwargs):
        self.calls.append(('crop', pano.id, kwargs))
        return image.crop((0, 0, 100, image.size[1])), (-5.0, 25.0)


def _pano(pano_id, lat_offset=0.0, links=()):
    return StreetViewPanorama(
        id=pano_id,
        lat=CAM_LAT + lat_offset,
        lon=CAM_LON,
        heading=20.0,
        date=(2025, 11),
        links=list(links),
    )


def _collection(n=1) -> PhysicalAssetCollection:
    col = PhysicalAssetCollection()
    for i in range(n):
        col.add(
            PhysicalAsset(
                id=f'bldg_{i}',
                geometry=box(
                    CAM_LON + 0.0002,
                    CAM_LAT - 0.0001,
                    CAM_LON + 0.0004,
                    CAM_LAT + 0.0001,
                ),
                attributes={'asset_type': 'building'},
            )
        )
    return col


def test_streetview_extractor_validation(tmp_path):
    """Constructor arguments are validated and the directory is created."""
    with pytest.raises(ValueError, match='zoom'):
        GoogleStreetViewImageExtractor(
            tmp_path / 'a', zoom=7, client=FakeStreetViewClient()
        )
    with pytest.raises(ValueError, match='max_images_per_asset'):
        GoogleStreetViewImageExtractor(
            tmp_path / 'a', max_images_per_asset=0, client=FakeStreetViewClient()
        )
    ext = GoogleStreetViewImageExtractor(tmp_path / 'nested' / 'dir', max_workers=0)
    assert ext.save_directory.is_dir() and ext.max_workers == 1
    assert ext.client is not None and ext.zoom == 3


def test_streetview_extractor_attaches_single_view(tmp_path):
    """The nearest panorama is downloaded, cropped, saved and attached."""
    client = FakeStreetViewClient({'nearest': _pano('P1')})
    ext = GoogleStreetViewImageExtractor(
        tmp_path, zoom=1, search_radius_m=40, vertical_crop=(0.2, 0.9), client=client
    )
    collection = ext(_collection(2))
    for asset in collection:
        [img] = list(asset.image_assets)
        assert img.path == tmp_path / f'gsv_{asset.id}_0.jpg'
        assert img.is_downloaded and Image.open(img.path).size[0] == 100
        props = img.properties
        assert props['pano_id'] == 'P1' and props['source'] == 'google_streetview'
        assert props['capture_date'] == '2025-11' and props['zoom'] == 1
        assert props['camera_heading'] == 20.0
        assert 0 < props['camera_distance_m'] < 60
        assert (props['view_angle_start'], props['view_angle_end']) == (-5.0, 25.0)
        assert 'panorama_path' not in props
    crop_kwargs = [c[2] for c in client.calls if c[0] == 'crop'][0]
    assert crop_kwargs['vertical_crop'] == (0.2, 0.9)
    assert crop_kwargs['fov_buffer_deg'] == 10.0
    assert ('find', round(CAM_LAT, 5), round(CAM_LON + 0.0003, 5), 40) in client.calls
    assert ('download', 'P1', 1) in client.calls


def test_streetview_extractor_multiple_views_from_links(tmp_path):
    """Extra viewpoints come from nearby links, ordered by distance."""
    nearest = _pano(
        'P1',
        links=[
            ('FAR', CAM_LAT + 0.01, CAM_LON, 0.0),  # > 1 km away, skipped
            ('L2', CAM_LAT + 0.0003, CAM_LON, 0.0),
            ('L1', CAM_LAT + 0.0001, CAM_LON, 0.0),
            ('MISSING', CAM_LAT + 0.0002, CAM_LON, 0.0),  # metadata fails
        ],
    )
    client = FakeStreetViewClient(
        {'nearest': nearest, 'L1': _pano('L1', 0.0001), 'L2': _pano('L2', 0.0003)}
    )
    ext = GoogleStreetViewImageExtractor(
        tmp_path, zoom=0, max_images_per_asset=3, client=client, image_prefix='sv'
    )
    collection = ext(_collection(1))
    images = list(collection.get('bldg_0').image_assets)
    assert [img.properties['pano_id'] for img in images] == ['P1', 'L1', 'MISSING'][
        :2
    ] or [img.properties['pano_id'] for img in images] == ['P1', 'L1']
    assert [img.path.name for img in images] == ['sv_bldg_0_0.jpg', 'sv_bldg_0_1.jpg']
    meta_calls = [c[1] for c in client.calls if c[0] == 'meta']
    assert meta_calls == ['L1', 'MISSING']  # sorted by distance, FAR excluded


def test_streetview_extractor_saves_panorama_and_depth(tmp_path):
    """Optional artefacts are written and referenced from the properties."""
    import base64
    import struct

    header = bytes([8]) + struct.pack('<HHHH', 2, 4, 2, 9)
    body = bytes([0, 1, 1, 0, 1, 1, 1, 1]) + struct.pack('<ffff', 0, 0, 1, 0)
    body += struct.pack('<ffff', 0, 0, 1, 2.5)
    depth_b64 = base64.urlsafe_b64encode(header + body).decode().rstrip('=')

    client = FakeStreetViewClient(
        {'nearest': _pano('P1'), 'P1': _pano('P1')}, depth=depth_b64
    )
    ext = GoogleStreetViewImageExtractor(
        tmp_path, zoom=0, save_panorama=True, save_depth_map=True, client=client
    )
    [asset] = list(ext(_collection(1)))
    [img] = list(asset.image_assets)
    assert Image.open(img.properties['panorama_path']).size == (512, 256)
    depth = np.load(img.properties['depth_map_path'])
    assert depth.shape == (2, 4) and np.isinf(depth[0, 0])
    # The depth map came from a metadata refetch because the search result had none:
    assert ('meta', 'P1') in client.calls


def test_streetview_extractor_missing_depth_and_no_pano(tmp_path, caplog):
    """No panorama, failed downloads and absent depth maps are handled."""
    client = FakeStreetViewClient({})
    ext = GoogleStreetViewImageExtractor(tmp_path, client=client)
    collection = ext(_collection(1))
    assert len(collection.get('bldg_0').image_assets) == 0

    client = FakeStreetViewClient({'nearest': _pano('P1')}, fail_download=True)
    collection = GoogleStreetViewImageExtractor(tmp_path, client=client)(_collection(1))
    assert len(collection.get('bldg_0').image_assets) == 0

    client = FakeStreetViewClient({'nearest': _pano('P1'), 'P1': _pano('P1')})
    ext = GoogleStreetViewImageExtractor(
        tmp_path, zoom=0, save_depth_map=True, client=client
    )
    with caplog.at_level(logging.WARNING):
        collection = ext(_collection(1))
    [img] = list(collection.get('bldg_0').image_assets)
    assert 'depth_map_path' not in img.properties
    assert 'No depth map available' in caplog.text


def test_streetview_extractor_errors_and_empty_geometry(tmp_path, caplog):
    """Crop errors are logged per viewpoint; empty geometries are skipped."""

    class BrokenCropClient(FakeStreetViewClient):
        def crop_to_geometry(self, image, pano, geometry, **kwargs):
            raise RuntimeError('crop exploded')

    client = BrokenCropClient({'nearest': _pano('P1')})
    ext = GoogleStreetViewImageExtractor(tmp_path, zoom=0, client=client)
    with caplog.at_level(logging.ERROR):
        collection = ext(_collection(1))
    assert 'crop exploded' in caplog.text
    assert len(collection.get('bldg_0').image_assets) == 0

    from shapely.geometry import Polygon

    empty = PhysicalAssetCollection()
    empty.add(PhysicalAsset(id='empty', geometry=Polygon()))
    assert ext._process_asset(empty.get('empty')) == 0


def test_streetview_extractor_cancellation(tmp_path):
    """A set cancel event aborts extraction with OperationCancelled."""
    event = threading.Event()
    event.set()
    client = FakeStreetViewClient({'nearest': _pano('P1')})
    ext = GoogleStreetViewImageExtractor(tmp_path, client=client, cancel_event=event)
    with pytest.raises(OperationCancelled):
        ext(_collection(2))

    # Cancelling between viewpoints propagates too:
    event = threading.Event()

    class CancellingClient(FakeStreetViewClient):
        def download_panorama(self, pano, zoom=3):
            event.set()
            return super().download_panorama(pano, zoom)

    nearest = _pano('P1', links=[('L1', CAM_LAT + 0.0001, CAM_LON, 0.0)])
    client = CancellingClient({'nearest': nearest, 'L1': _pano('L1', 0.0001)})
    ext = GoogleStreetViewImageExtractor(
        tmp_path, zoom=0, max_images_per_asset=2, client=client, cancel_event=event
    )
    with pytest.raises(OperationCancelled):
        ext(_collection(1))


def test_streetview_extractor_reraises_cancellation_from_client(tmp_path):
    """OperationCancelled raised inside the client is not swallowed."""

    class CancellingCropClient(FakeStreetViewClient):
        def crop_to_geometry(self, image, pano, geometry, **kwargs):
            raise OperationCancelled('stop now')

    client = CancellingCropClient({'nearest': _pano('P1')})
    ext = GoogleStreetViewImageExtractor(tmp_path, zoom=0, client=client)
    with pytest.raises(OperationCancelled, match='stop now'):
        ext(_collection(1))
