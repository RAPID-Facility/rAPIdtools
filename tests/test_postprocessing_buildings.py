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
import math
import threading
from pathlib import Path

import numpy as np
import pyproj
import pytest
from PIL import Image
from shapely.geometry import (
    LineString,
    Point,
    Polygon,
    box,
)
from shapely.ops import transform, unary_union

from rapidtools.core import (
    ImageAsset,
    OperationCanceled,
    PhysicalAsset,
    PhysicalAssetCollection,
)
from rapidtools.models.base import ModelOutput
from rapidtools.processing import image_segmenters
from rapidtools.processing.postprocessing import buildings
from rapidtools.processing.postprocessing.buildings import BuildingRegularizer

# ==========================================
# 1. Fixtures & Helpers
# ==========================================

CX, CY = -118.2493, 34.0505
# Approximate degrees per foot at this latitude:
DLAT = 1.0 / 364000.0
DLON = DLAT / math.cos(math.radians(CY))
MASK_SIZE = 40


def ft_box(x0, y0, x1, y1):
    """Box given in feet offsets from (CX, CY), returned in WGS84 degrees."""
    return box(CX + x0 * DLON, CY + y0 * DLAT, CX + x1 * DLON, CY + y1 * DLAT)


def mask_from_geometry(geom, bounds, size=MASK_SIZE):
    """Rasterize a WGS84 geometry into a (1, size, size) boolean mask."""
    min_lon, min_lat, max_lon, max_lat = bounds
    mask = np.zeros((size, size), dtype=bool)
    for row in range(size):
        for col in range(size):
            lon = min_lon + (col + 0.5) / size * (max_lon - min_lon)
            lat = max_lat - (row + 0.5) / size * (max_lat - min_lat)
            mask[row, col] = geom.contains(Point(lon, lat))
    return mask[np.newaxis, ...]


class FakeSAM3:
    """SAM3Inference stand-in serving masks keyed by image file name."""

    def __init__(self, model_id='facebook/sam3', device='auto', load_in_4bit=True):
        self.model_id = model_id
        self.calls: list[dict] = []
        self.masks_by_name: dict[str, np.ndarray] = {}
        self.scores_by_name: dict[str, list[float]] = {}
        self.on_call = None

    def run_inference(self, image_inputs, prompt, threshold, mask_threshold, **kw):
        paths = list(image_inputs)
        self.calls.append(
            {
                'inputs': paths,
                'threshold': threshold,
                'mask_threshold': mask_threshold,
                'kwargs': kw,
            }
        )
        if self.on_call is not None:
            self.on_call()
        masks, scores = [], []
        for p in paths:
            name = Path(p).name  # separator-agnostic: CI also runs on Windows
            masks.append(
                self.masks_by_name.get(name, np.zeros((1, MASK_SIZE, MASK_SIZE), bool))
            )
            scores.append(self.scores_by_name.get(name))
        raw = {'scores': scores} if any(v is not None for v in scores) else None
        return ModelOutput(masks=masks, bounding_boxes=None, raw_response=raw)


@pytest.fixture
def fake_sam(monkeypatch):
    holder = {'masks_by_name': {}, 'scores_by_name': {}}

    def factory(**kwargs):
        inst = FakeSAM3(**kwargs)
        # Share one mask registry so fixtures can register masks before the
        # regularizer (and therefore the model) is constructed:
        inst.masks_by_name = holder['masks_by_name']
        inst.scores_by_name = holder.setdefault('scores_by_name', {})
        holder['instance'] = inst
        return inst

    monkeypatch.setattr(image_segmenters, 'SAM3Inference', factory)
    return holder


@pytest.fixture
def regularizer(fake_sam):
    return BuildingRegularizer()


def _attach_image(asset, tmp_path, name, bounds):
    Image.new('RGB', (MASK_SIZE, MASK_SIZE), color=(90, 90, 90)).save(tmp_path / name)
    asset.add_image_assets(
        ImageAsset(
            id=name[:-4], path=tmp_path / name, properties={'wgs84_bounds': bounds}
        )
    )


@pytest.fixture
def prelim(tmp_path, fake_sam):
    """Three preliminary buildings with crops; only b1 has a matching mask."""
    b1_geom = ft_box(-20, -15, 20, 15)
    b1 = PhysicalAsset(id='b1', geometry=b1_geom, attributes={'source': 'osm'})
    b1_bounds = ft_box(-40, -40, 40, 40).bounds
    _attach_image(b1, tmp_path, 'b1_aerial.jpg', b1_bounds)

    b2 = PhysicalAsset(id='b2', geometry=ft_box(100, 100, 140, 130))
    _attach_image(b2, tmp_path, 'b2_aerial.jpg', ft_box(80, 80, 160, 160).bounds)

    b3 = PhysicalAsset(id='b3', geometry=ft_box(-200, -200, -160, -170))

    # SAM "finds" a slightly larger roof for b1 and nothing for b2:
    roof = ft_box(-22, -17, 22, 17)
    fake_sam['masks_by_name']['b1_aerial.jpg'] = mask_from_geometry(roof, b1_bounds)
    return PhysicalAssetCollection([b1, b2, b3])


# ==========================================
# 2. Construction
# ==========================================


def test_init_defaults(fake_sam):
    """Defaults are stored and the internal segmenter shares the config."""
    event = threading.Event()
    reg = BuildingRegularizer(cancel_event=event)
    assert reg.min_overlap_ratio == 0.15
    assert reg.max_gap_bridge_ft == 15.0
    assert reg.snap_tolerance_ft == buildings.DEFAULT_SNAP_TOLERANCE_FT
    assert reg.retry_dropped is True
    assert reg.retry_confidence_threshold == 0.3
    assert reg.retry_mask_threshold == 0.3
    assert reg.cancel_event is event
    assert reg.segmenter.prompt == 'building'
    assert reg.segmenter.threshold == 0.5
    assert reg.segmenter.mask_threshold == 0.5
    assert reg.segmenter.batch_size == 4
    assert reg.segmenter.cancel_event is event
    assert reg.regularize_outlines is True
    assert reg.neck_width_m == 1.0
    assert reg.segmenter.max_mask_side == buildings.DEFAULT_MAX_MASK_SIDE


def test_init_list_prompt_and_custom_thresholds(fake_sam):
    """List prompts are joined; thresholds propagate to the segmenter."""
    reg = BuildingRegularizer(
        prompt=['building', 'roof'],
        batch_size=2,
        confidence_threshold=0.7,
        mask_threshold=0.6,
        retry_dropped=False,
        min_overlap_ratio=0.4,
    )
    assert reg.segmenter.prompt == 'building. roof'
    assert reg.segmenter.batch_size == 2
    assert reg.segmenter.threshold == 0.7
    assert reg.segmenter.mask_threshold == 0.6
    assert reg.retry_dropped is False
    assert reg.min_overlap_ratio == 0.4


# ==========================================
# 3. Full pipeline
# ==========================================


def test_process_refines_matching_building(regularizer, prelim, fake_sam):
    """A building whose mask matches the seed is refined; others are dropped."""
    result = regularizer(prelim)
    assert len(result) == 1
    refined = result[0]
    assert refined.id.startswith('bldg_merged_')
    assert refined.attributes['asset_type'] == 'building_footprint'
    assert refined.geometry.geom_type == 'Polygon'
    # The refined footprint follows the (larger) SAM roof, not the seed:
    seed = prelim['b1'].geometry
    assert refined.geometry.area > seed.area
    assert refined.geometry.intersection(seed).area / seed.area > 0.95

    inst = fake_sam['instance']
    # Pass 1 (2 downloaded images) + retry pass on b2 (b3 has no image):
    assert [len(c['inputs']) for c in inst.calls] == [2, 1]
    assert inst.calls[1]['threshold'] == 0.3
    assert inst.calls[1]['mask_threshold'] == 0.3
    # Thresholds are restored after the retry pass:
    assert regularizer.segmenter.threshold == 0.5
    assert regularizer.segmenter.mask_threshold == 0.5


def test_process_retry_recovers_dropped_asset(regularizer, prelim, fake_sam):
    """A mask that only appears in the low-threshold pass is recovered."""
    inst = fake_sam['instance']
    b2_bounds = prelim['b2'].image_assets[0].properties['wgs84_bounds']
    b2_mask = mask_from_geometry(prelim['b2'].geometry, b2_bounds)

    def serve_on_retry():
        if len(inst.calls) >= 2:
            inst.masks_by_name['b2_aerial.jpg'] = b2_mask

    inst.on_call = serve_on_retry
    result = regularizer(prelim)
    assert len(result) == 2


def test_process_without_retry(fake_sam, prelim, caplog):
    """retry_dropped=False skips the second segmentation pass."""
    reg = BuildingRegularizer(retry_dropped=False)
    with caplog.at_level(logging.INFO):
        result = reg(prelim)
    assert len(result) == 1
    assert len(fake_sam['instance'].calls) == 1
    assert 'Retry step skipped' in caplog.text


def test_process_no_dropped_assets(fake_sam, tmp_path, caplog):
    """When every asset succeeds, the retry pass is not run."""
    geom = ft_box(-20, -15, 20, 15)
    asset = PhysicalAsset(id='only', geometry=geom)
    bounds = ft_box(-40, -40, 40, 40).bounds
    _attach_image(asset, tmp_path, 'only_aerial.jpg', bounds)
    fake_sam['masks_by_name']['only_aerial.jpg'] = mask_from_geometry(geom, bounds)
    reg = BuildingRegularizer()
    with caplog.at_level(logging.INFO):
        result = reg(PhysicalAssetCollection([asset]))
    assert len(result) == 1
    assert len(fake_sam['instance'].calls) == 1
    assert 'No dropped assets to retry' in caplog.text


def test_process_empty_input(regularizer, caplog):
    """An empty collection returns an empty collection with a warning."""
    with caplog.at_level(logging.WARNING):
        result = regularizer.process(PhysicalAssetCollection())
    assert len(result) == 0
    assert 'No assets provided' in caplog.text


def test_process_canceled_after_segmentation(fake_sam, prelim):
    """A cancel event set during segmentation stops the pipeline."""
    event = threading.Event()
    reg = BuildingRegularizer(cancel_event=event, retry_dropped=False)
    fake_sam['instance'].on_call = event.set
    with pytest.raises(OperationCanceled, match='building regularization'):
        reg(prelim)


def test_process_canceled_before_start(fake_sam, prelim):
    """A pre-set cancel event is honored by the internal segmenter."""
    event = threading.Event()
    event.set()
    reg = BuildingRegularizer(cancel_event=event)
    with pytest.raises(OperationCanceled):
        reg(prelim)
    assert fake_sam['instance'].calls == []


# ==========================================
# 4. Fusion, matching and outlines
# ==========================================


def _stack(*geoms, bounds):
    """Instance masks (N, size, size) of WGS84 geometries in a crop."""
    return np.concatenate([mask_from_geometry(g, bounds) for g in geoms])


def _assets(*geoms):
    return PhysicalAssetCollection(
        [PhysicalAsset(id=f'g{i}', geometry=g) for i, g in enumerate(geoms)]
    )


def _to_feet(x, y):
    """WGS84 to feet around (CX, CY), for checking shapes."""
    return ((x - CX) / DLON, (y - CY) / DLAT)


def test_process_keeps_touching_neighbors_apart(fake_sam, tmp_path):
    """Two touching buildings under one merged seed come back as two."""
    blob = ft_box(-30, -10, 30, 10)
    asset = PhysicalAsset(id='blob', geometry=blob)
    bounds = ft_box(-40, -40, 40, 40).bounds
    _attach_image(asset, tmp_path, 'blob_aerial.jpg', bounds)
    left, right = ft_box(-28, -8, -2, 8), ft_box(-2, -8, 28, 8)
    fake_sam['masks_by_name']['blob_aerial.jpg'] = _stack(left, right, bounds=bounds)
    result = BuildingRegularizer(retry_dropped=False)(PhysicalAssetCollection([asset]))
    assert len(result) == 2
    geoms = [a.geometry for a in result]
    assert geoms[0].intersection(geoms[1]).area < 1e-12
    assert all(a.attributes['source_ids'] == ['blob'] for a in result)


def test_process_cuts_masks_bleeding_between_roofs(fake_sam, tmp_path):
    """One mask joining two roofs by a thin strip is cut into two buildings."""
    blob = ft_box(-30, -10, 30, 10)
    asset = PhysicalAsset(id='blob', geometry=blob)
    bounds = ft_box(-40, -40, 40, 40).bounds
    _attach_image(asset, tmp_path, 'blob_aerial.jpg', bounds)
    joined = unary_union(
        # The strip is one 2 ft (0.6 m) pixel wide:
        [ft_box(-30, -10, -6, 10), ft_box(-6, -1.5, 6, 0.5), ft_box(6, -10, 30, 10)]
    )
    fake_sam['masks_by_name']['blob_aerial.jpg'] = mask_from_geometry(joined, bounds)
    result = BuildingRegularizer(retry_dropped=False)(PhysicalAssetCollection([asset]))
    assert len(result) == 2
    # Without the cut the strip joins them into one footprint:
    kept_whole = BuildingRegularizer(retry_dropped=False, neck_width_m=0.0)(
        PhysicalAssetCollection([asset])
    )
    assert len(kept_whole) == 1


def test_process_fuses_views_of_overlapping_crops(fake_sam, tmp_path):
    """A building seen in its own and in its neighbor's crop is one footprint."""
    a_geom, b_geom = ft_box(-30, -10, -6, 10), ft_box(6, -10, 30, 10)
    a = PhysicalAsset(id='a', geometry=a_geom)
    b = PhysicalAsset(id='b', geometry=b_geom)
    a_bounds = ft_box(-45, -40, 35, 40).bounds
    b_bounds = ft_box(-35, -40, 45, 40).bounds
    _attach_image(a, tmp_path, 'a_aerial.jpg', a_bounds)
    _attach_image(b, tmp_path, 'b_aerial.jpg', b_bounds)
    fake_sam['masks_by_name']['a_aerial.jpg'] = _stack(a_geom, b_geom, bounds=a_bounds)
    fake_sam['masks_by_name']['b_aerial.jpg'] = _stack(a_geom, b_geom, bounds=b_bounds)
    fake_sam['scores_by_name']['a_aerial.jpg'] = [0.9, 0.7]
    fake_sam['scores_by_name']['b_aerial.jpg'] = [0.8, 0.6]
    result = BuildingRegularizer(retry_dropped=False)(PhysicalAssetCollection([a, b]))
    assert len(result) == 2
    assert all(asset.attributes['views'] == 2 for asset in result)
    by_seed = {asset.attributes['source_ids'][0]: asset for asset in result}
    assert by_seed['a'].attributes['confidence'] == pytest.approx(0.85)
    assert by_seed['b'].attributes['confidence'] == pytest.approx(0.65)
    # The masks are released once traced:
    assert 'sam3_masks' not in a.attributes
    assert 'sam3_masks' not in b.attributes


def test_process_drops_buildings_matching_no_seed(fake_sam, tmp_path):
    """A neighbor visible in the crop but not among the seeds is not returned."""
    seed = ft_box(-30, -10, -6, 10)
    asset = PhysicalAsset(id='a', geometry=seed)
    bounds = ft_box(-40, -40, 40, 40).bounds
    _attach_image(asset, tmp_path, 'a_aerial.jpg', bounds)
    neighbor = ft_box(10, -10, 30, 10)
    fake_sam['masks_by_name']['a_aerial.jpg'] = _stack(seed, neighbor, bounds=bounds)
    result = BuildingRegularizer(retry_dropped=False)(PhysicalAssetCollection([asset]))
    assert len(result) == 1
    assert result[0].attributes['source_ids'] == ['a']


def test_process_draws_square_walls(fake_sam, tmp_path):
    """A stair-stepped mask of a rotated building becomes a clean rectangle."""
    from shapely import affinity

    seed = transform(
        lambda x, y: (CX + x * DLON, CY + y * DLAT),
        affinity.rotate(box(-20, -12, 20, 12), 25, origin=(0, 0)),
    )
    asset = PhysicalAsset(id='r', geometry=seed)
    bounds = ft_box(-40, -40, 40, 40).bounds
    _attach_image(asset, tmp_path, 'r_aerial.jpg', bounds)
    fake_sam['masks_by_name']['r_aerial.jpg'] = mask_from_geometry(seed, bounds)
    result = BuildingRegularizer(retry_dropped=False)(PhysicalAssetCollection([asset]))
    assert len(result) == 1
    outline = transform(_to_feet, result[0].geometry)
    coords = np.asarray(outline.exterior.coords)[:-1]
    assert len(coords) == 4
    for i in range(4):
        u = coords[i - 1] - coords[i]
        v = coords[(i + 1) % 4] - coords[i]
        assert abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)) < 0.02
    # Without regularization the traced pixel outline keeps its steps:
    raw = BuildingRegularizer(retry_dropped=False, regularize_outlines=False)(
        PhysicalAssetCollection([asset])
    )
    assert len(raw[0].geometry.exterior.coords) > 10


def test_process_reprojects_web_mercator_seed(fake_sam, tmp_path):
    """Seeds in EPSG:3857 are reprojected to WGS84 before matching."""
    geom = ft_box(-20, -15, 20, 15)
    to_3857 = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
    asset = PhysicalAsset(id='m', geometry=transform(to_3857.transform, geom))
    bounds = ft_box(-40, -40, 40, 40).bounds
    _attach_image(asset, tmp_path, 'm_aerial.jpg', bounds)
    fake_sam['masks_by_name']['m_aerial.jpg'] = mask_from_geometry(geom, bounds)
    result = BuildingRegularizer(retry_dropped=False)(PhysicalAssetCollection([asset]))
    assert len(result) == 1
    assert result[0].geometry.intersection(geom).area / geom.area > 0.9


def test_process_ignores_images_without_georeference(fake_sam, tmp_path):
    """Crops with neither native_georef nor wgs84_bounds contribute nothing."""
    geom = ft_box(-20, -15, 20, 15)
    asset = PhysicalAsset(id='n', geometry=geom)
    Image.new('RGB', (MASK_SIZE, MASK_SIZE)).save(tmp_path / 'n.jpg')
    asset.add_image_assets(ImageAsset(id='n', path=tmp_path / 'n.jpg'))
    fake_sam['masks_by_name']['n.jpg'] = mask_from_geometry(
        geom, ft_box(-40, -40, 40, 40).bounds
    )
    result = BuildingRegularizer(retry_dropped=False)(PhysicalAssetCollection([asset]))
    assert len(result) == 0


def test_max_mask_side_is_passed_to_the_model(regularizer, prelim, fake_sam):
    """The segmenter asks SAM 3 for reduced-size masks of large crops."""
    regularizer(prelim)
    kwargs = fake_sam['instance'].calls[0]['kwargs']
    assert kwargs['max_mask_side'] == buildings.DEFAULT_MAX_MASK_SIDE


def test_shares_a_loaded_model(fake_sam, prelim):
    """The SAM 3 of the scan can be reused instead of loading a second copy."""
    loaded = FakeSAM3()
    loaded.masks_by_name = fake_sam['masks_by_name']
    loaded.scores_by_name = fake_sam['scores_by_name']
    regularizer = BuildingRegularizer(model=loaded)
    assert regularizer.segmenter.model is loaded
    assert 'instance' not in fake_sam  # nothing new was loaded
    regularizer(prelim)
    assert loaded.calls


def test_deprecated_parameters_warn(fake_sam):
    """The bridging and snapping parameters no longer do anything and warn."""
    with pytest.warns(DeprecationWarning, match='max_gap_bridge_ft'):
        reg = BuildingRegularizer(max_gap_bridge_ft=10.0)
    assert reg.max_gap_bridge_ft == 10.0
    with pytest.warns(DeprecationWarning, match='snap_tolerance_ft'):
        BuildingRegularizer(snap_tolerance_ft=1.0)


# ==========================================
# 5. _filter_degenerate_assets
# ==========================================


def test_filter_degenerate_assets(regularizer, caplog):
    """Empty, unhealable, non-polygon, zero-area shapes are removed."""
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
    sliver = Polygon([(0, 0), (1e-6, 0), (1e-6, 1e-6), (0, 0)])
    line = LineString([(0, 0), (1, 1)])
    good = box(0, 0, 5, 5)
    empty = Polygon()
    col = _assets(empty, bowtie, sliver, line, good)
    with caplog.at_level(logging.INFO):
        result = regularizer._filter_degenerate_assets(col)
    ids = {a.id for a in result}
    # The bowtie heals into a valid polygonal geometry and survives:
    assert ids == {'g1', 'g4'}
    assert result['g1'].geometry.is_valid
    assert result['g1'].geometry.geom_type in ('Polygon', 'MultiPolygon')
    assert result['g1'].geometry is not bowtie
    assert 'Discarded 3 degenerate geometries' in caplog.text


def test_filter_degenerate_assets_all_valid_is_quiet(regularizer, caplog):
    """No log message is emitted when nothing is discarded."""
    with caplog.at_level(logging.INFO):
        result = regularizer._filter_degenerate_assets(_assets(box(0, 0, 1, 1)))
    assert len(result) == 1
    assert 'Discarded' not in caplog.text
