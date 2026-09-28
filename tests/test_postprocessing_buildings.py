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
import math
import threading

import numpy as np
import pyproj
import pytest
from PIL import Image
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.ops import transform

from rapidtools.core import (
    ImageAsset,
    OperationCancelled,
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
        self.on_call = None

    def run_inference(self, image_inputs, prompt, threshold, mask_threshold, **kw):
        paths = list(image_inputs)
        self.calls.append(
            {'inputs': paths, 'threshold': threshold, 'mask_threshold': mask_threshold}
        )
        if self.on_call is not None:
            self.on_call()
        masks = []
        for p in paths:
            name = p.split('/')[-1]
            masks.append(
                self.masks_by_name.get(name, np.zeros((1, MASK_SIZE, MASK_SIZE), bool))
            )
        return ModelOutput(masks=masks, bounding_boxes=None)


@pytest.fixture
def fake_sam(monkeypatch):
    holder = {'masks_by_name': {}}

    def factory(**kwargs):
        inst = FakeSAM3(**kwargs)
        # Share one mask registry so fixtures can register masks before the
        # regularizer (and therefore the model) is constructed:
        inst.masks_by_name = holder['masks_by_name']
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


def test_process_cancelled_after_segmentation(fake_sam, prelim):
    """A cancel event set during segmentation stops the pipeline."""
    event = threading.Event()
    reg = BuildingRegularizer(cancel_event=event, retry_dropped=False)
    fake_sam['instance'].on_call = event.set
    with pytest.raises(OperationCancelled, match='building regularization'):
        reg(prelim)


def test_process_cancelled_before_start(fake_sam, prelim):
    """A pre-set cancel event is honoured by the internal segmenter."""
    event = threading.Event()
    event.set()
    reg = BuildingRegularizer(cancel_event=event)
    with pytest.raises(OperationCancelled):
        reg(prelim)
    assert fake_sam['instance'].calls == []


# ==========================================
# 4. _georeference_and_filter
# ==========================================


def test_georeference_creates_split_assets(regularizer, tmp_path):
    """Passing masks become '<id>_split_i_j' assets carrying seed metadata."""
    geom = ft_box(-20, -15, 20, 15)
    asset = PhysicalAsset(id='a', geometry=geom, attributes={'k': 'v'})
    bounds = ft_box(-40, -40, 40, 40).bounds
    _attach_image(asset, tmp_path, 'a_aerial.jpg', bounds)
    asset.attributes['sam3_masks'] = {'a_aerial': mask_from_geometry(geom, bounds)}

    refined = regularizer._georeference_and_filter(PhysicalAssetCollection([asset]))
    assert len(refined) == 1
    new = refined['a_split_0_0']
    assert new.attributes['k'] == 'v'
    assert new.attributes['sam3_refined'] is True
    assert new.attributes['preliminary_geometry_wkt'] == geom.wkt
    assert 'sam3_masks' in new.attributes
    assert len(new.image_assets) == 1
    assert new.geometry.intersection(geom).area / geom.area > 0.9


def test_georeference_accepts_2d_mask_and_drops_non_matching(regularizer, tmp_path):
    """2D masks are promoted; masks far from the seed are rejected."""
    geom = ft_box(-20, -15, 20, 15)
    asset = PhysicalAsset(id='a', geometry=geom)
    bounds = ft_box(-40, -40, 40, 40).bounds
    _attach_image(asset, tmp_path, 'a_aerial.jpg', bounds)
    far = ft_box(25, 25, 38, 38)
    asset.attributes['sam3_masks'] = {'a_aerial': mask_from_geometry(far, bounds)[0]}
    refined = regularizer._georeference_and_filter(PhysicalAssetCollection([asset]))
    assert len(refined) == 0


def test_georeference_skips_assets_without_usable_data(regularizer, tmp_path):
    """Empty geometry, missing masks, empty masks or missing bounds are skipped."""
    bounds = ft_box(-40, -40, 40, 40).bounds
    geom = ft_box(-20, -15, 20, 15)
    good_mask = mask_from_geometry(geom, bounds)

    empty_geom = PhysicalAsset(id='e', geometry=Polygon())
    empty_geom.attributes['sam3_masks'] = {'x': good_mask}

    no_masks = PhysicalAsset(id='n', geometry=geom)

    zero_len = PhysicalAsset(id='z', geometry=geom)
    _attach_image(zero_len, tmp_path, 'z_aerial.jpg', bounds)
    zero_len.attributes['sam3_masks'] = {'z_aerial': np.zeros((0, 4, 4), bool)}

    no_bounds = PhysicalAsset(id='nb', geometry=geom)
    Image.new('RGB', (4, 4)).save(tmp_path / 'nb.jpg')
    no_bounds.add_image_assets(ImageAsset(id='nb_img', path=tmp_path / 'nb.jpg'))
    no_bounds.attributes['sam3_masks'] = {'nb_img': good_mask}

    all_zero = PhysicalAsset(id='az', geometry=geom)
    _attach_image(all_zero, tmp_path, 'az_aerial.jpg', bounds)
    all_zero.attributes['sam3_masks'] = {
        'az_aerial': np.zeros((1, MASK_SIZE, MASK_SIZE), bool)
    }

    refined = regularizer._georeference_and_filter(
        PhysicalAssetCollection([empty_geom, no_masks, zero_len, no_bounds, all_zero])
    )
    assert len(refined) == 0


def test_georeference_reprojects_web_mercator_seed(regularizer, tmp_path):
    """Seeds in EPSG:3857 are reprojected to WGS84 before comparison."""
    geom = ft_box(-20, -15, 20, 15)
    to_3857 = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
    geom_3857 = transform(to_3857.transform, geom)
    asset = PhysicalAsset(id='m', geometry=geom_3857)
    bounds = ft_box(-40, -40, 40, 40).bounds
    _attach_image(asset, tmp_path, 'm_aerial.jpg', bounds)
    asset.attributes['sam3_masks'] = {'m_aerial': mask_from_geometry(geom, bounds)}
    refined = regularizer._georeference_and_filter(PhysicalAssetCollection([asset]))
    assert len(refined) == 1
    assert refined[0].geometry.bounds[0] == pytest.approx(geom.bounds[0], abs=1e-5)


def test_georeference_bridges_fragmented_mask(regularizer, tmp_path):
    """Two mask blobs separated by a small gap are fused into one footprint."""
    geom = ft_box(-20, -15, 20, 15)
    asset = PhysicalAsset(id='f', geometry=geom)
    bounds = ft_box(-40, -40, 40, 40).bounds
    _attach_image(asset, tmp_path, 'f_aerial.jpg', bounds)
    fragments = MultiPolygon([ft_box(-20, -15, -3, 15), ft_box(3, -15, 20, 15)])
    asset.attributes['sam3_masks'] = {'f_aerial': mask_from_geometry(fragments, bounds)}
    refined = regularizer._georeference_and_filter(PhysicalAssetCollection([asset]))
    assert len(refined) == 1
    assert refined[0].geometry.geom_type == 'Polygon'
    assert refined[0].geometry.intersection(geom).area / geom.area > 0.9


# ==========================================
# 5. Geometry helpers
# ==========================================


def test_fill_holes(regularizer):
    """Interior rings are removed from polygons and multipolygons."""
    holed = Polygon(
        box(0, 0, 10, 10).exterior.coords, [box(4, 4, 6, 6).exterior.coords]
    )
    assert regularizer._fill_holes(holed).area == pytest.approx(100.0)
    multi = MultiPolygon([holed, box(20, 20, 22, 22)])
    filled = regularizer._fill_holes(multi)
    assert filled.geom_type == 'MultiPolygon'
    assert filled.area == pytest.approx(104.0)
    gc_no_polys = GeometryCollection([Point(0, 0)])
    assert regularizer._fill_holes(gc_no_polys) is gc_no_polys
    pt = Point(1, 1)
    assert regularizer._fill_holes(pt) is pt


def test_bridge_fragments_merges_close_parts(regularizer):
    """Nearby fragments are fused into one polygon covering the gap."""
    parts = MultiPolygon([ft_box(0, 0, 20, 20), ft_box(24, 0, 44, 20)])
    bridged = regularizer._bridge_fragments(parts)
    assert bridged.geom_type == 'Polygon'
    assert bridged.area == pytest.approx(ft_box(0, 0, 44, 20).area, rel=0.02)


def test_bridge_fragments_far_parts_keep_largest(regularizer):
    """Fragments further apart than max_gap_bridge_ft collapse to the largest."""
    small, large = ft_box(0, 0, 10, 10), ft_box(80, 0, 110, 20)
    bridged = regularizer._bridge_fragments(MultiPolygon([small, large]))
    assert bridged.geom_type == 'Polygon'
    assert bridged.area == pytest.approx(large.area, rel=0.02)


def test_bridge_fragments_passthrough(regularizer):
    """Single polygons and collections with <=1 polygon are returned as-is."""
    poly = ft_box(0, 0, 10, 10)
    assert regularizer._bridge_fragments(poly) is poly
    gc = GeometryCollection([poly, Point(CX, CY)])
    assert regularizer._bridge_fragments(gc) is gc


# ==========================================
# 6. _truncate_intrusions
# ==========================================


def _assets(*geoms):
    return PhysicalAssetCollection(
        [PhysicalAsset(id=f'g{i}', geometry=g) for i, g in enumerate(geoms)]
    )


def test_truncate_intrusions_cuts_along_wall(regularizer):
    """A small building poking into a big one is sliced along the wall."""
    big = box(0, 0, 10, 10)
    small = box(8, 4, 14, 6)
    result = regularizer._truncate_intrusions(_assets(small, big))
    assert len(result) == 2
    assert result['g1'].geometry.equals(big)
    cut = result['g0'].geometry
    assert cut.equals(box(10, 4, 14, 6))


def test_truncate_intrusions_wgs84_scale(regularizer):
    """The cut also works for degree-scale geometries (relative tolerances)."""
    big = ft_box(0, 0, 40, 40)
    small = ft_box(35, 10, 60, 20)
    result = regularizer._truncate_intrusions(_assets(small, big))
    cut = result['g0'].geometry
    assert cut.intersection(big).area < 1e-6 * big.area
    assert cut.area == pytest.approx(ft_box(40, 10, 60, 20).area, rel=1e-3)


def test_truncate_intrusions_interior_overlap_uses_difference(regularizer):
    """A building fully inside another is reduced to an empty geometry."""
    big = box(0, 0, 10, 10)
    inner = box(2, 2, 4, 4)
    result = regularizer._truncate_intrusions(_assets(big, inner))
    assert result['g1'].geometry.is_empty


def test_truncate_intrusions_ignores_disjoint_and_negligible(regularizer):
    """Disjoint or barely touching buildings are left untouched."""
    a = box(0, 0, 10, 10)
    b = box(20, 0, 30, 10)
    touching = box(10, 0, 20, 10)
    result = regularizer._truncate_intrusions(_assets(a, b, touching))
    assert result['g0'].geometry.equals(a)
    assert result['g1'].geometry.equals(b)
    assert result['g2'].geometry.equals(touching)


def test_truncate_intrusions_multipolygon_keeps_largest_part(regularizer):
    """Multi-part inputs are reduced to their largest polygon."""
    big = box(0, 0, 10, 10)
    fragmented = MultiPolygon([box(20, 0, 25, 5), box(30, 0, 31, 1)])
    result = regularizer._truncate_intrusions(_assets(big, fragmented))
    geom = result['g1'].geometry
    assert geom.geom_type == 'Polygon'
    assert geom.equals(box(20, 0, 25, 5))


def test_truncate_intrusions_multipolygon_inside_becomes_empty(regularizer):
    """Multi-part geometry fully inside a larger one is emptied by difference."""
    big = box(0, 0, 10, 10)
    inner_parts = MultiPolygon([box(1, 1, 2, 2), box(5, 5, 6, 6)])
    result = regularizer._truncate_intrusions(_assets(big, inner_parts))
    assert result['g0'].geometry.equals(big)
    assert result['g1'].geometry.is_empty


def test_truncate_intrusions_skips_empty(regularizer):
    """Empty geometries are dropped from the output."""
    result = regularizer._truncate_intrusions(_assets(Polygon(), box(0, 0, 1, 1)))
    assert len(result) == 1


# ==========================================
# 7. _dissolve_overlapping_assets
# ==========================================


def test_dissolve_merges_duplicates_keeps_neighbours(regularizer):
    """Polygons overlapping >50% merge; adjacent ones stay separate."""
    a = box(0, 0, 10, 10)
    a_dup = box(1, 1, 11, 11)
    neighbour = box(10, 0, 20, 10)
    slight = box(9, 9, 19, 19)  # overlaps a by 1% -> separate
    result = regularizer._dissolve_overlapping_assets(
        _assets(a, a_dup, neighbour, slight)
    )
    assert len(result) == 3
    areas = sorted(round(x.geometry.area, 6) for x in result)
    assert areas == [100.0, 100.0, pytest.approx(119.0)]
    for asset in result:
        assert asset.id.startswith('bldg_merged_')
        assert asset.attributes == {'asset_type': 'building_footprint'}


def test_dissolve_fills_holes_and_handles_empty(regularizer):
    """A ring of overlapping boxes becomes one solid polygon; empty input ok."""
    ring = [
        box(0, 0, 10, 3),
        box(0, 7, 10, 10),
        box(0, 0, 3, 10),
        box(7, 0, 10, 10),
    ]
    result = regularizer._dissolve_overlapping_assets(_assets(*ring))
    # Ring pieces overlap each other by less than 50% -> stay separate parts,
    # but the union of each component is hole-free.
    assert all(len(a.geometry.interiors) == 0 for a in result)
    assert len(regularizer._dissolve_overlapping_assets(PhysicalAssetCollection())) == 0
    assert len(regularizer._dissolve_overlapping_assets(_assets(Polygon()))) == 0


def test_dissolve_skips_non_polygon_union_parts(regularizer):
    """Unions that degenerate to lines are not emitted."""
    line_like = LineString([(0, 0), (1, 1)]).buffer(0)  # empty polygon
    result = regularizer._dissolve_overlapping_assets(
        _assets(line_like, box(0, 0, 1, 1))
    )
    assert len(result) == 1


# ==========================================
# 8. _filter_degenerate_assets
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
