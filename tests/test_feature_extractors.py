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

import logging
import os
import threading

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

from rapidtools.core import OperationCancelled, PhysicalAssetCollection
from rapidtools.models.base import ModelOutput
from rapidtools.processing import feature_extractors
from rapidtools.processing.feature_extractors import SAM3OrthoFeatureExtractor

# ==========================================
# 1. Fakes & Fixtures
# ==========================================


class FakeSAM3:
    """SAM3Inference stand-in returning synthetic masks per tile."""

    def __init__(self, model_id='facebook/sam3', device='auto', load_in_4bit=True):
        self.model_id = model_id
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.calls: list[dict] = []
        self.mode = 'square'
        self.on_call = None
        self.seen_paths_exist: list[bool] = []

    def run_inference(self, image_inputs, prompt, threshold, mask_threshold, **kw):
        paths = list(image_inputs)
        self.calls.append(
            {
                'inputs': paths,
                'prompt': prompt,
                'threshold': threshold,
                'mask_threshold': mask_threshold,
            }
        )
        self.seen_paths_exist.extend(os.path.exists(p) for p in paths)
        if self.on_call is not None:
            self.on_call()
        n = len(paths)
        if self.mode == 'none':
            return None
        if self.mode == 'no_masks':
            return ModelOutput(masks=None)
        if self.mode == 'raise':
            raise RuntimeError('inference exploded')
        if self.mode == 'empty':
            return ModelOutput(masks=[np.zeros((0, 32, 32), dtype=bool)] * n)
        if self.mode == 'mixed_none':
            return ModelOutput(masks=[None] * n)
        if self.mode == 'twod':
            m = np.zeros((32, 32), dtype=bool)
            m[8:24, 8:24] = True
            return ModelOutput(masks=[m for _ in range(n)])
        if self.mode == 'list_of_2d':
            m = np.zeros((32, 32), dtype=bool)
            m[8:24, 8:24] = True
            return ModelOutput(masks=[[m, m] for _ in range(n)])
        if self.mode == 'all_zero':
            return ModelOutput(masks=[np.zeros((1, 32, 32), dtype=bool)] * n)
        if self.mode == 'full':
            # Mask covering the whole tile so neighbouring tiles overlap/touch
            return ModelOutput(masks=[np.ones((1, 32, 32), dtype=bool)] * n)
        if self.mode == 'scored':
            # Default squares plus per-instance confidences:
            m = np.zeros((1, 32, 32), dtype=bool)
            m[0, 8:24, 8:24] = True
            return ModelOutput(
                masks=[m.copy() for _ in range(n)],
                raw_response={'scores': [[0.9]] * n},
            )
        # Default: one square instance in the middle of each tile
        m = np.zeros((1, 32, 32), dtype=bool)
        m[0, 8:24, 8:24] = True
        return ModelOutput(masks=[m.copy() for _ in range(n)], raw_response={})


@pytest.fixture
def fake_sam(monkeypatch):
    holder = {}

    def factory(**kwargs):
        inst = FakeSAM3(**kwargs)
        holder['instance'] = inst
        return inst

    monkeypatch.setattr(feature_extractors, 'SAM3Inference', factory)
    return holder


def _write_raster(path, crs, bounds, size=64, nodata=None):
    minx, miny, maxx, maxy = bounds
    transform = from_bounds(minx, miny, maxx, maxy, size, size)
    rng = np.random.default_rng(42)
    data = rng.integers(50, 200, size=(3, size, size), dtype=np.uint8)
    kwargs = {}
    if nodata is not None:
        kwargs['nodata'] = nodata
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=size,
        width=size,
        count=3,
        dtype='uint8',
        crs=crs,
        transform=transform,
        **kwargs,
    ) as dst:
        dst.write(data)
    return path


@pytest.fixture
def wgs84_raster(tmp_path):
    return _write_raster(
        tmp_path / 'wgs84.tif',
        'EPSG:4326',
        (-118.250, 34.050, -118.249, 34.051),
    )


@pytest.fixture
def utm_raster(tmp_path):
    return _write_raster(
        tmp_path / 'utm.tif',
        'EPSG:32611',
        (500000, 3767000, 500064, 3767064),
    )


# ==========================================
# 2. Construction
# ==========================================


def test_init_stores_config_and_builds_model(fake_sam):
    """Constructor arguments are stored and forwarded to SAM3Inference."""
    event = threading.Event()
    ext = SAM3OrthoFeatureExtractor(
        prompt='building',
        patch_size=50,
        unit='meters',
        overlap_ratio=0.2,
        model_id='x/y',
        device='cpu',
        batch_size=3,
        load_in_4bit=False,
        threshold=0.4,
        mask_threshold=0.3,
        max_missing_data_ratio=0.9,
        cancel_event=event,
    )
    assert ext.prompt == 'building'
    assert ext.patch_size == 50
    assert ext.unit == 'meters'
    assert ext.overlap_ratio == 0.2
    assert ext.batch_size == 3
    assert ext.threshold == 0.4
    assert ext.mask_threshold == 0.3
    assert ext.max_missing_data_ratio == 0.9
    assert ext.cancel_event is event
    inst = fake_sam['instance']
    assert ext.model is inst
    assert (inst.model_id, inst.device, inst.load_in_4bit) == ('x/y', 'cpu', False)


# ==========================================
# 3. End-to-end extraction
# ==========================================


def test_call_extracts_polygons_from_tiles(fake_sam, wgs84_raster):
    """Each tile yields one polygon; disjoint squares stay separate assets."""
    ext = SAM3OrthoFeatureExtractor(
        prompt='swimming pool',
        patch_size=32,
        unit='pixels',
        overlap_ratio=0.0,
        batch_size=3,
    )
    result = ext(wgs84_raster)

    assert isinstance(result, PhysicalAssetCollection)
    assert len(result) == 4  # 2x2 tiles, one square each, none touching

    inst = fake_sam['instance']
    # 4 tiles / batch of 3 -> one full batch + one partial batch:
    assert [len(c['inputs']) for c in inst.calls] == [3, 1]
    assert inst.calls[0]['prompt'] == 'swimming pool'
    assert inst.calls[0]['threshold'] == 0.5

    raster_bounds = box(-118.250, 34.050, -118.249, 34.051)
    for asset in result:
        assert asset.id.startswith('swimming_pool_')
        assert asset.geometry.geom_type == 'Polygon'
        assert raster_bounds.contains(asset.geometry)
        assert asset.attributes['asset_type'] == 'swimming pool'
        assert asset.attributes['source_model'] == 'facebook/sam3'
        assert asset.attributes['extraction_scale'] == '32_pixels'


def test_call_merges_overlapping_detections(fake_sam, utm_raster):
    """Masks covering whole tiles dissolve into a single polygon."""
    ext = SAM3OrthoFeatureExtractor(
        prompt='road', patch_size=32, unit='meters', overlap_ratio=0.0, batch_size=4
    )
    fake_sam['instance'].mode = 'full'
    result = ext(utm_raster)
    assert len(result) == 1
    geom = result[0].geometry
    # The merged footprint should span (almost) the whole raster extent:
    minx, miny, maxx, maxy = geom.bounds
    assert minx == pytest.approx(-117.0, abs=1e-4)
    assert maxx == pytest.approx(-116.9993, abs=1e-3)
    assert miny == pytest.approx(34.0437, abs=1e-3)
    assert maxy == pytest.approx(34.0443, abs=1e-3)


def test_call_with_feet_and_overlap(fake_sam, utm_raster):
    """Real-world units and overlap produce more, overlapping tiles."""
    ext = SAM3OrthoFeatureExtractor(
        prompt='building',
        patch_size=100,
        unit='feet',
        overlap_ratio=0.25,
        batch_size=4,
    )
    result = ext(utm_raster)
    total_tiles = sum(len(c['inputs']) for c in fake_sam['instance'].calls)
    assert total_tiles == 9
    assert len(result) >= 1
    assert result[0].attributes['extraction_scale'] == '100_feet'


def test_call_accepts_2d_masks(fake_sam, wgs84_raster):
    """A single 2D mask per tile is promoted to (1, H, W)."""
    ext = SAM3OrthoFeatureExtractor(
        prompt='x', patch_size=32, unit='pixels', overlap_ratio=0.0
    )
    fake_sam['instance'].mode = 'twod'
    result = ext(wgs84_raster)
    assert len(result) == 4


def test_call_accepts_list_of_masks(fake_sam, wgs84_raster):
    """A Python list of 2D masks per tile is handled like a stacked array."""
    ext = SAM3OrthoFeatureExtractor(
        prompt='x', patch_size=32, unit='pixels', overlap_ratio=0.0
    )
    fake_sam['instance'].mode = 'list_of_2d'
    result = ext(wgs84_raster)
    # Duplicated instances merge into the same 4 squares:
    assert len(result) == 4


# ==========================================
# 4. Empty / degenerate results
# ==========================================


@pytest.mark.parametrize(
    'mode', ['none', 'no_masks', 'empty', 'mixed_none', 'all_zero']
)
def test_call_returns_empty_collection_when_nothing_detected(
    fake_sam, wgs84_raster, mode, caplog
):
    """No masks, empty masks, None entries or all-zero masks give no assets."""
    ext = SAM3OrthoFeatureExtractor(
        prompt='x', patch_size=32, unit='pixels', overlap_ratio=0.0
    )
    fake_sam['instance'].mode = mode
    with caplog.at_level(logging.INFO):
        result = ext(wgs84_raster)
    assert len(result) == 0
    assert 'Yielded 0 unique assets' in caplog.text


def test_call_missing_raster_raises(fake_sam, tmp_path):
    """A non-existent raster path propagates FileNotFoundError."""
    ext = SAM3OrthoFeatureExtractor(prompt='x')
    with pytest.raises(FileNotFoundError):
        ext(tmp_path / 'missing.tif')


# ==========================================
# 5. Temp file handling
# ==========================================


def test_process_batch_cleans_temp_files(fake_sam, wgs84_raster):
    """Temporary JPEGs exist during inference and are removed afterwards."""
    ext = SAM3OrthoFeatureExtractor(
        prompt='x', patch_size=32, unit='pixels', overlap_ratio=0.0, batch_size=2
    )
    ext(wgs84_raster)
    inst = fake_sam['instance']
    assert inst.seen_paths_exist and all(inst.seen_paths_exist)
    for call in inst.calls:
        for p in call['inputs']:
            assert p.endswith('.jpg')
            assert not os.path.exists(p)


def test_process_batch_cleans_temp_files_on_error(fake_sam, wgs84_raster):
    """Temporary files are removed even when inference raises."""
    ext = SAM3OrthoFeatureExtractor(
        prompt='x', patch_size=32, unit='pixels', overlap_ratio=0.0
    )
    fake_sam['instance'].mode = 'raise'
    with pytest.raises(RuntimeError, match='inference exploded'):
        ext(wgs84_raster)
    for p in fake_sam['instance'].calls[0]['inputs']:
        assert not os.path.exists(p)


def test_process_batch_direct_with_prebuilt_images(fake_sam, tmp_path):
    """_process_batch maps a mask square to the tile's WGS84 bounds."""
    from PIL import Image

    ext = SAM3OrthoFeatureExtractor(prompt='x')
    raw: list = []
    img = Image.new('RGB', (32, 32), color=(10, 20, 30))
    bounds = (-118.0, 34.0, -117.0, 35.0)
    ext._process_batch([img], [bounds], raw)
    assert len(raw) == 1
    minx, miny, maxx, maxy = raw[0].bounds
    # Mask occupies columns/rows 8..24 of 32 -> quarter to three quarters:
    assert minx == pytest.approx(-117.75)
    assert maxx == pytest.approx(-117.25)
    assert miny == pytest.approx(34.25)
    assert maxy == pytest.approx(34.75)


# ==========================================
# 6. Cancellation
# ==========================================


def test_call_cancelled_before_first_tile(fake_sam, wgs84_raster):
    """A pre-set cancel event aborts before any inference happens."""
    event = threading.Event()
    event.set()
    ext = SAM3OrthoFeatureExtractor(
        prompt='building', patch_size=32, unit='pixels', cancel_event=event
    )
    with pytest.raises(OperationCancelled, match="scan for 'building'"):
        ext(wgs84_raster)
    assert fake_sam['instance'].calls == []


def test_call_cancelled_between_tiles(fake_sam, wgs84_raster):
    """Cancellation requested during batch one stops the scan at the next tile."""
    event = threading.Event()
    ext = SAM3OrthoFeatureExtractor(
        prompt='building',
        patch_size=32,
        unit='pixels',
        overlap_ratio=0.0,
        batch_size=1,
        cancel_event=event,
    )
    fake_sam['instance'].on_call = event.set
    with pytest.raises(OperationCancelled):
        ext(wgs84_raster)
    assert len(fake_sam['instance'].calls) == 1


# --------------------------------------------------------------------------
# merge_overlaps=False (instance mode)
# --------------------------------------------------------------------------


def test_instance_mode_keeps_touching_detections(fake_sam, utm_raster):
    """Full-tile masks that only touch stay separate instead of dissolving."""
    ext = SAM3OrthoFeatureExtractor(
        prompt='vehicle',
        patch_size=32,
        unit='meters',
        overlap_ratio=0.0,
        batch_size=4,
        merge_overlaps=False,
    )
    assert ext.merge_overlaps is False
    fake_sam['instance'].mode = 'full'
    result = ext(utm_raster)
    assert len(result) == 4  # merge mode gives 1 (see test above)
    for asset in result:
        assert asset.id.startswith('vehicle_')
        assert asset.attributes['asset_type'] == 'vehicle'
        assert asset.attributes['confidence'] is None  # fake reported no scores
        assert asset.attributes['extraction_scale'] == '32_meters'


def test_instance_mode_records_confidence(fake_sam, wgs84_raster):
    """SAM 3 scores end up as the ``confidence`` attribute."""
    ext = SAM3OrthoFeatureExtractor(
        prompt='car',
        patch_size=32,
        unit='pixels',
        overlap_ratio=0.0,
        merge_overlaps=False,
    )
    fake_sam['instance'].mode = 'scored'
    result = ext(wgs84_raster)
    assert len(result) == 4
    assert all(a.attributes['confidence'] == pytest.approx(0.9) for a in result)


def test_instance_mode_removes_identical_duplicates(fake_sam, wgs84_raster):
    """Two identical instances per tile collapse to one asset per tile."""
    ext = SAM3OrthoFeatureExtractor(
        prompt='x',
        patch_size=32,
        unit='pixels',
        overlap_ratio=0.0,
        merge_overlaps=False,
    )
    fake_sam['instance'].mode = 'list_of_2d'
    result = ext(wgs84_raster)
    assert len(result) == 4


def test_merge_mode_ignores_scores(fake_sam, wgs84_raster):
    """The default merge mode is unchanged by the presence of scores."""
    ext = SAM3OrthoFeatureExtractor(
        prompt='x', patch_size=32, unit='pixels', overlap_ratio=0.0
    )
    fake_sam['instance'].mode = 'scored'
    result = ext(wgs84_raster)
    assert len(result) == 4
    assert 'confidence' not in result[0].attributes


def test_suppress_duplicates_prefers_higher_confidence():
    """Overlapping polygons keep the best-scoring one; touching ones survive."""
    a = box(0, 0, 10, 10)
    shifted = box(2, 0, 12, 10)  # 80% overlap with a
    touching = box(10, 0, 20, 10)  # shares an edge with a, zero overlap area
    partial = box(7, 0, 12, 10)  # 60% of this smaller box is covered by a
    kept = SAM3OrthoFeatureExtractor._suppress_duplicates(
        [shifted, a, touching, partial], [0.8, 0.9, 0.7, 0.85]
    )
    assert [score for _, score in kept] == [0.9, 0.7]
    assert kept[0][0].equals(a)
    assert kept[1][0].equals(touching)


def test_suppress_duplicates_without_scores_keeps_first_seen():
    """With no scores, earlier polygons win and ``None`` is reported."""
    a = box(0, 0, 10, 10)
    kept = SAM3OrthoFeatureExtractor._suppress_duplicates([a, box(1, 1, 11, 11)], None)
    assert len(kept) == 1
    assert kept[0] == (a, None)
    assert SAM3OrthoFeatureExtractor._suppress_duplicates([], None) == []
