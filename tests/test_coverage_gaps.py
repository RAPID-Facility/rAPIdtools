"""Small branches of several modules: parsing fallbacks and error paths."""

import logging
import types

import numpy as np
import pytest
from affine import Affine
from PIL import Image, ImageDraw
from rasterio.crs import CRS
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

import rapidtools
from rapidtools.core import ImageAsset, Observation, PhysicalAsset
from rapidtools.data_sources.orthomosaic_reader import PatchGeoref
from rapidtools.gui.server import (
    ApiError,
    GuiServer,
    _geometry_coords,
    _optional_bbox,
    _overview_tiles,
)
from rapidtools.models.base import ModelOutput
from rapidtools.processing import image_analyzers as ia
from rapidtools.processing.outlines import (
    draw_corner_brackets,
    draw_geometry,
    draw_outline,
    parse_outline_buffer,
    parse_outline_width,
)
from rapidtools.processing.street_localization import ground_range_from_ray
from rapidtools.processing.vectorize import mask_to_wgs84_polygons
from rapidtools.processing.verification import _VerificationAnalyzer


def test_outline_parsing_and_drawing_fallbacks():
    with pytest.raises(ValueError, match='Could not parse outline buffer'):
        parse_outline_buffer('wide')
    with pytest.raises(ValueError, match='Could not parse outline buffer'):
        parse_outline_buffer('1.2.3px')
    with pytest.raises(ValueError, match='Could not parse outline width'):
        parse_outline_width('1.2.3px')
    image = Image.new('RGB', (40, 40))
    draw_outline(image, [], True)  # nothing to draw
    with pytest.raises(ValueError, match='outline shape'):
        draw_outline(image, [(1.0, 1.0), (10.0, 1.0), (10.0, 10.0)], True, shape='blob')
    draw = ImageDraw.Draw(image)
    draw_geometry(draw, Polygon(), 2, 'red', False)
    draw_geometry(
        draw, MultiPolygon([box(0, 0, 5, 5), box(10, 10, 20, 20)]), 2, 'red', True
    )
    draw_corner_brackets(draw, [(0, 0), (0, 0), (10, 0), (10, 10)], 2, 'red')
    assert image.getbbox() is not None


def test_server_helpers_cover_every_geometry_kind():
    assert _geometry_coords(MultiLineString([[(0, 0), (1, 1)]]))[0] == 'line'
    assert _geometry_coords(box(0, 0, 1, 1))[0] == 'polygon'
    kind, coords = _geometry_coords(MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)]))
    assert kind == 'polygon' and len(coords) == 2
    assert _geometry_coords(MultiPoint([(0, 0), (2, 2)])) == ('point', [1.0, 1.0])
    assert _geometry_coords(LineString([(0, 0), (1, 1)]))[0] == 'line'
    with pytest.raises(ApiError, match='min < max'):
        _optional_bbox({'min_lon': 2, 'min_lat': 0, 'max_lon': 1, 'max_lat': 2})
    assert len(_overview_tiles()) > 0
    fake = types.SimpleNamespace(server_address=(b'127.0.0.1', 8080))
    assert GuiServer._host_port(fake) == ('127.0.0.1', 8080)


def test_core_parsing_fallbacks():
    asset = ImageAsset.from_dict(
        {
            'id': 7,
            'path': '/virtual/a.jpg',
            'semantic_map': {'1': 'road', 'sky': 'sky'},
            'instance_map': {2: 'car'},
        }
    )
    assert asset.id == '7'
    assert asset.semantic_map == {1: 'road', 'sky': 'sky'}
    assert asset.instance_map == {2: 'car'}
    with pytest.raises(ValueError, match='at least three vertices'):
        Observation('i', 'car', [(0.0, 0.0), (1.0, 1.0)], None, None, 0.0)
    georef = PatchGeoref(
        transform=Affine.translation(1.0, 2.0),
        crs=CRS.from_epsg(4326),
        width=4,
        height=3,
    )
    rebuilt = PatchGeoref.from_dict(georef.to_dict())
    assert rebuilt.transform == georef.transform and rebuilt.width == 4
    with pytest.raises(ValueError, match='2D mask'):
        mask_to_wgs84_polygons(np.zeros((2, 2, 2)), wgs84_bounds=(0, 0, 1, 1))
    with pytest.raises(ValueError, match='georef or wgs84_bounds'):
        mask_to_wgs84_polygons(np.zeros((2, 2)))
    assert rapidtools.__getattr__('processing') is rapidtools.processing
    assert ground_range_from_ray([0.0, 1.0, -1.0], 2.4) == pytest.approx(2.4)


def test_model_listing_falls_back_when_the_backend_cannot_list(monkeypatch, caplog):
    from rapidtools import models
    from rapidtools.gui.workflow import MODEL_BACKENDS, list_available_models

    monkeypatch.setattr(models, 'get_model_class', lambda backend: object)
    with caplog.at_level(logging.WARNING):
        assert (
            list_available_models('openai', 'k') == MODEL_BACKENDS['openai']['models']
        )
    assert 'cannot list remote models' in caplog.text


def test_analyzer_does_not_retry_provider_rejections(tmp_path, caplog):
    class Rejecting:
        model_id = 'strict'

        def __init__(self):
            self.calls = 0

        def run_inference(self, image_inputs, prompt, **kwargs):
            self.calls += 1
            return ModelOutput.failure('unsafe content', retryable=False)

    path = tmp_path / 'a.jpg'
    path.write_bytes(b'\xff\xd8\xff\xd9')
    asset = PhysicalAsset(id='a', geometry=Point(0, 0))
    asset.add_image_assets(ImageAsset(id='a_img', path=path))
    model = Rejecting()
    analyzer = ia.AssetAnalyzer(model, prompt='Rate it')
    with caplog.at_level(logging.ERROR):
        analyzer(rapidtools.PhysicalAssetCollection([asset]))
    assert model.calls == 1 and asset.attributes == {}
    assert 'rejected by the provider' in caplog.text
    verifier = _VerificationAnalyzer(model, prompt='Is it?')
    assert verifier._apply_result(asset, [path], None) is False
    assert verifier._apply_result(asset, [path], ModelOutput(text='')) is False


def test_prompt_builder_helpers():
    from rapidtools.gui.prompt_builder import (
        OutputField,
        PromptAssistant,
        PromptSpec,
        _apply_context,
        _sentence,
    )

    assert _sentence('') == ''
    field = OutputField('Score', kind='number', description='1 is worst')
    assert field.json_hint() == 'number; 1 is worst'
    spec = PromptSpec()
    spec.asset = ''  # a drafted spec that named no asset takes the GUI's
    spec = _apply_context(spec, {'json_output': True, 'asset': 'vehicles'})
    assert spec.json_output is True and spec.asset == 'vehicles'

    class Blank:
        model_id = 'blank'

    assistant = PromptAssistant(Blank())
    assistant._ask = lambda prompt, json_mode: '   '
    with pytest.raises(ValueError, match='empty review'):
        assistant.review('Rate the roof.')
