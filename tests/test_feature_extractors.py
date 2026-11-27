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
# 11-27-2026

import logging
import os
import threading

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

from rapidtools.core import OperationCanceled, PhysicalAssetCollection
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
            # Mask covering the whole tile so neighboring tiles overlap/touch
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


@pytest.mark.parametrize('mode', ['no_masks', 'empty', 'mixed_none', 'all_zero'])
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


def test_call_raises_when_every_batch_fails(fake_sam, wgs84_raster):
    """A scan in which the model never answered is an error, not an empty map."""
    ext = SAM3OrthoFeatureExtractor(
        prompt='x', patch_size=32, unit='pixels', overlap_ratio=0.0
    )
    fake_sam['instance'].mode = 'none'
    with pytest.raises(RuntimeError, match='failed on every batch'):
        ext(wgs84_raster)


def test_call_stops_after_failed_batches_in_a_row(fake_sam, wgs84_raster, caplog):
    """Isolated failures are reported; a run of them stops the scan."""
    ext = SAM3OrthoFeatureExtractor(
        prompt='x', patch_size=32, unit='pixels', overlap_ratio=0.0, batch_size=1
    )
    fake = fake_sam['instance']
    answers = iter(['none', 'square', 'square', 'square'])

    def flaky(*args, **kwargs):
        fake.mode = next(answers)
        return FakeSAM3.run_inference(fake, *args, **kwargs)

    fake.run_inference = flaky
    with caplog.at_level(logging.WARNING):
        result = ext(wgs84_raster)
    assert len(result) == 3
    assert 'failed on 1 of 4 batches' in caplog.text

    del fake.run_inference
    fake.mode = 'none'
    with pytest.raises(RuntimeError, match='3 batches in a row'):
        ext(wgs84_raster)
    # Stopped at the third failure, before the fourth tile:
    assert len(fake.calls) == 4 + 3


def test_call_missing_raster_raises(fake_sam, tmp_path):
    """A non-existent raster path propagates FileNotFoundError."""
    ext = SAM3OrthoFeatureExtractor(prompt='x')
    with pytest.raises(FileNotFoundError):
        ext(tmp_path / 'missing.tif')


# ==========================================
# 5. Temp file handling
# ==========================================


def test_process_batch_cleans_temp_files(fake_sam, wgs84_raster):
    """Temporary JPEGs exist during inference and are removed afterward."""
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


def test_call_canceled_before_first_tile(fake_sam, wgs84_raster):
    """A pre-set cancel event aborts before any inference happens."""
    event = threading.Event()
    event.set()
    ext = SAM3OrthoFeatureExtractor(
        prompt='building', patch_size=32, unit='pixels', cancel_event=event
    )
    with pytest.raises(OperationCanceled, match="scan for 'building'"):
        ext(wgs84_raster)
    assert fake_sam['instance'].calls == []


def test_call_canceled_between_tiles(fake_sam, wgs84_raster):
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
    with pytest.raises(OperationCanceled):
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


class _TileMasks:
    """FakeSAM3 stand-in that draws, per tile, the given boxes in raster pixels.

    ``boxes`` maps a tile's column/row offset (in raster pixels) to a list of
    ``(col0, row0, col1, row1, score)`` boxes in raster pixel coordinates;
    each box becomes one instance clipped to the tile.
    """

    def __init__(self, ext, raster, boxes):
        self.ext, self.raster, self.boxes = ext, raster, boxes

    def install(self, monkeypatch):
        offsets = []
        original = self.ext._process_batch

        def spy(images, bounds, raw, scores=None, georefs=None, **kw):
            offsets.extend(
                (
                    round((g.transform.c - self.origin[0]) / self.res[0]),
                    round((g.transform.f - self.origin[1]) / self.res[1]),
                )
                for g in georefs
            )
            self.pending = offsets[-len(images) :]
            return original(images, bounds, raw, scores, georefs, **kw)

        with rasterio.open(self.raster) as ds:
            self.origin = (ds.transform.c, ds.transform.f)
            self.res = (ds.transform.a, ds.transform.e)
        monkeypatch.setattr(self.ext, '_process_batch', spy)
        self.ext.model.run_inference = self.run_inference

    def run_inference(self, image_inputs, prompt, threshold, mask_threshold, **kw):
        from PIL import Image

        masks, scores = [], []
        for path, (c0, r0) in zip(image_inputs, self.pending, strict=True):
            w, h = Image.open(path).size
            tile_masks, tile_scores = [], []
            for bc0, br0, bc1, br1, score in self.boxes:
                m = np.zeros((h, w), dtype=bool)
                m[
                    max(0, br0 - r0) : max(0, br1 - r0),
                    max(0, bc0 - c0) : max(0, bc1 - c0),
                ] = True
                if m.any():
                    tile_masks.append(m)
                    tile_scores.append(score)
            masks.append(
                np.array(tile_masks) if tile_masks else np.zeros((0, h, w), bool)
            )
            scores.append(tile_scores)
        return ModelOutput(masks=masks, raw_response={'scores': scores})


def test_instance_mode_fuses_tile_edge_cuts(fake_sam, utm_raster, monkeypatch):
    """An object cut by one tile and whole in the overlapping tile is one asset."""
    ext = SAM3OrthoFeatureExtractor(
        prompt='building',
        patch_size=32,
        unit='pixels',
        overlap_ratio=0.5,
        merge_overlaps=False,
    )
    # A 12 x 10 m building straddling the 32 px tile boundary (1 px = 1 m):
    _TileMasks(ext, utm_raster, [(26, 20, 38, 30, 0.8)]).install(monkeypatch)
    result = ext(utm_raster)
    assert len(result) == 1
    asset = result[0]
    assert asset.attributes['views'] >= 2
    assert asset.attributes['confidence'] == pytest.approx(0.8)
    # The outline is the whole building, not the piece one tile saw:
    import pyproj
    from shapely.ops import transform

    to_utm = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32611', always_xy=True)
    area = transform(to_utm.transform, asset.geometry).area
    assert area == pytest.approx(120.0, rel=0.1)


def test_instance_mode_keeps_neighbors_in_one_tile_apart(
    fake_sam, utm_raster, monkeypatch
):
    """Two touching instances from the same tile are never merged."""
    ext = SAM3OrthoFeatureExtractor(
        prompt='building',
        patch_size=32,
        unit='pixels',
        overlap_ratio=0.5,
        merge_overlaps=False,
    )
    boxes = [(20, 20, 30, 30, 0.9), (30, 20, 40, 30, 0.9)]
    _TileMasks(ext, utm_raster, boxes).install(monkeypatch)
    assert len(ext(utm_raster)) == 2


def test_instance_mode_cuts_masks_at_narrow_necks(fake_sam, utm_raster, monkeypatch):
    """A mask bleeding across a gap is cut when neck_width_m is set."""
    # Two 10 x 10 m roofs joined by a 1 m wide strip of mask:
    boxes = [(10, 10, 20, 20, 0.9), (20, 14, 23, 15, 0.9), (23, 10, 33, 20, 0.9)]

    def run(neck):
        ext = SAM3OrthoFeatureExtractor(
            prompt='building',
            patch_size=64,
            unit='pixels',
            overlap_ratio=0.0,
            merge_overlaps=False,
            neck_width_m=neck,
        )
        fake = _TileMasks(ext, utm_raster, [])
        fake.install(monkeypatch)

        def joined(image_inputs, prompt, threshold, mask_threshold, **kw):
            from PIL import Image

            w, h = Image.open(image_inputs[0]).size
            m = np.zeros((h, w), dtype=bool)
            for c0, r0, c1, r1, _ in boxes:
                m[r0:r1, c0:c1] = True
            return ModelOutput(masks=[m[np.newaxis]], raw_response={'scores': [[0.9]]})

        ext.model.run_inference = joined
        return ext(utm_raster)

    assert len(run(0.0)) == 1
    assert len(run(2.0)) == 2


def test_instance_mode_min_detection_rate(fake_sam, utm_raster, monkeypatch):
    """Objects only some of the tiles that saw them detected can be dropped."""
    ext = SAM3OrthoFeatureExtractor(
        prompt='building',
        patch_size=32,
        unit='pixels',
        overlap_ratio=0.5,
        merge_overlaps=False,
        min_detection_rate=0.6,
    )
    # Inside the tiles at (0, 0), (16, 0), (0, 16) and (16, 16):
    fake = _TileMasks(ext, utm_raster, [(18, 18, 24, 24, 0.9)])
    fake.install(monkeypatch)
    seen_by_all = len(ext(utm_raster))
    # Only the tile at the origin reports the object now:
    original = fake.run_inference

    def first_tile_only(image_inputs, prompt, threshold, mask_threshold, **kw):
        out = original(image_inputs, prompt, threshold, mask_threshold)
        for k, (c0, r0) in enumerate(fake.pending):
            if (c0, r0) != (16, 16):
                h, w = out.masks[k].shape[-2:]
                out.masks[k] = np.zeros((0, h, w), bool)
                out.raw_response['scores'][k] = []
        return out

    ext.model.run_inference = first_tile_only
    assert seen_by_all == 1
    assert len(ext(utm_raster)) == 0


def test_instance_mode_validates_parameters(fake_sam):
    """Negative neck widths and rates outside [0, 1] are rejected."""
    with pytest.raises(ValueError, match='neck_width_m'):
        SAM3OrthoFeatureExtractor(prompt='x', neck_width_m=-1)
    with pytest.raises(ValueError, match='min_detection_rate'):
        SAM3OrthoFeatureExtractor(prompt='x', min_detection_rate=1.5)


# --------------------------------------------------------------------------
# Building prompts
# --------------------------------------------------------------------------


def test_is_building_prompt():
    """Prompts naming buildings are recognized in any case and inside words."""
    assert feature_extractors.is_building_prompt('Damaged HOUSE')
    assert feature_extractors.is_building_prompt('warehouse')
    assert feature_extractors.is_building_prompt('residential buildings')
    assert not feature_extractors.is_building_prompt('swimming pool')


def test_building_prompts_get_building_defaults(fake_sam):
    """Settings left unset follow the prompt; explicit settings are kept."""
    ext = SAM3OrthoFeatureExtractor(prompt='house')
    s = ext.scan_settings()
    assert (s.building, s.merge_overlaps, s.overlap_ratio) == (True, False, 0.5)
    assert (s.neck_width_m, s.min_detection_rate, s.regularize) == (1.0, 0.3, True)
    # Changing the prompt of a loaded extractor re-derives the defaults:
    ext.prompt = 'tree'
    s = ext.scan_settings()
    assert (s.building, s.merge_overlaps, s.overlap_ratio) == (False, True, 0.15)
    assert (s.neck_width_m, s.min_detection_rate, s.regularize) == (0.0, 0.0, False)
    # Explicit settings win:
    ext = SAM3OrthoFeatureExtractor(
        prompt='building', overlap_ratio=0.3, merge_overlaps=True, regularize=False
    )
    s = ext.scan_settings()
    assert (s.merge_overlaps, s.overlap_ratio, s.regularize) == (True, 0.3, False)
    # The neck cut and detection rate only apply to buildings kept separate:
    assert (s.neck_width_m, s.min_detection_rate) == (0.0, 0.0)


def _rotated_box_mask(width, height, angle=25.0, size=(30.0, 16.0)):
    """A pixel mask of a rotated rectangle centered in the image."""
    from PIL import Image, ImageDraw
    from shapely import affinity

    rect = box(-size[0] / 2, -size[1] / 2, size[0] / 2, size[1] / 2)
    rect = affinity.translate(
        affinity.rotate(rect, angle, origin=(0, 0)), width / 2, height / 2
    )
    image = Image.new('1', (width, height), 0)
    ImageDraw.Draw(image).polygon(list(rect.exterior.coords), fill=1)
    return np.array(image, dtype=bool)


def test_building_scans_are_regularized(fake_sam, utm_raster):
    """Building footprints get square walls unless regularize=False."""

    def scan(**kwargs):
        ext = SAM3OrthoFeatureExtractor(
            prompt='building', patch_size=64, unit='pixels', overlap_ratio=0.0, **kwargs
        )

        def rotated(image_inputs, prompt, threshold, mask_threshold, **kw):
            from PIL import Image

            masks = []
            for path in image_inputs:
                w, h = Image.open(path).size
                masks.append(_rotated_box_mask(w, h)[np.newaxis])
            return ModelOutput(
                masks=masks, raw_response={'scores': [[0.9]] * len(masks)}
            )

        ext.model.run_inference = rotated
        return ext(utm_raster)

    raw, square = scan(regularize=False), scan()
    assert len(raw) == len(square) == 1
    assert len(square[0].geometry.exterior.coords) == 5  # a rectangle
    assert len(raw[0].geometry.exterior.coords) > 5  # the pixel staircase
    # Same building, same place:
    assert square[0].geometry.intersection(raw[0].geometry).area > (
        0.9 * raw[0].geometry.area
    )
    assert square[0].attributes['views'] == 1
