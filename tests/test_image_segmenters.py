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
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Point

from rapidtools.core import (
    ImageAsset,
    OperationCancelled,
    PhysicalAsset,
    PhysicalAssetCollection,
)
from rapidtools.models.base import ModelOutput
from rapidtools.processing import image_segmenters
from rapidtools.processing.image_segmenters import SAM3ImageSegmenter

# ==========================================
# 1. Fakes & Fixtures
# ==========================================


class FakeSAM3:
    """Stand-in for SAM3Inference that never touches model weights."""

    def __init__(self, model_id='fake/sam3', device='auto', load_in_4bit=True):
        self.model_id = model_id
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.calls: list[dict] = []
        self.mode = 'normal'
        self.on_call = None

    def run_inference(self, image_inputs, prompt, threshold, mask_threshold, **kw):
        self.calls.append(
            {
                'inputs': list(image_inputs),
                'prompt': prompt,
                'threshold': threshold,
                'mask_threshold': mask_threshold,
            }
        )
        if self.on_call is not None:
            self.on_call()
        n = len(image_inputs)
        if self.mode == 'none':
            return None
        if self.mode == 'no_masks':
            return ModelOutput(masks=None)
        if self.mode == 'raise':
            raise RuntimeError('boom')
        if self.mode == 'single_array':
            # Non-list masks / boxes for a batch of one.
            return ModelOutput(
                masks=np.ones((1, 4, 4), dtype=bool),
                bounding_boxes=np.array([[0, 0, 3, 3]]),
            )
        if self.mode == 'no_boxes':
            return ModelOutput(
                masks=[np.ones((1, 4, 4), dtype=bool) for _ in range(n)],
                bounding_boxes=None,
            )
        if self.mode == 'partial_boxes':
            boxes = [np.array([[0, 0, 1, 1]])] + [None] * (n - 1)
            return ModelOutput(
                masks=[np.ones((1, 4, 4), dtype=bool) for _ in range(n)],
                bounding_boxes=boxes,
            )
        return ModelOutput(
            masks=[np.ones((2, 4, 4), dtype=bool) for _ in range(n)],
            bounding_boxes=[np.array([[0, 0, 3, 3], [1, 1, 2, 2]]) for _ in range(n)],
            raw_response={'n': n},
        )


@pytest.fixture
def fake_sam(monkeypatch):
    holder = {}

    def factory(**kwargs):
        inst = FakeSAM3(**kwargs)
        holder['instance'] = inst
        return inst

    monkeypatch.setattr(image_segmenters, 'SAM3Inference', factory)
    return holder


def _make_image(path, size=(4, 4)):
    Image.new('RGB', size, color=(120, 130, 140)).save(path)
    return path


@pytest.fixture
def collection(tmp_path):
    """Two assets with three real images plus one missing image."""
    a1 = PhysicalAsset(id='a1', geometry=Point(0, 0))
    a2 = PhysicalAsset(id='a2', geometry=Point(1, 1))

    a1.add_image_assets(
        ImageAsset(
            id='a1_aerial',
            path=_make_image(tmp_path / 'a1_aerial.jpg'),
            properties={'kind': 'aerial'},
        ),
        ImageAsset(
            id='a1_street',
            path=_make_image(tmp_path / 'a1_street.jpg'),
            properties={'kind': 'street'},
        ),
        ImageAsset(
            id='a1_missing',
            path=tmp_path / 'nope.jpg',
            allow_missing_file=True,
            properties={'kind': 'aerial'},
        ),
    )
    a2.add_image_assets(
        ImageAsset(
            id='a2_aerial',
            path=_make_image(tmp_path / 'a2_aerial.jpg'),
            properties={'kind': 'aerial'},
        )
    )
    return PhysicalAssetCollection([a1, a2])


# ==========================================
# 2. Construction
# ==========================================


def test_init_joins_list_prompt(fake_sam):
    """A list prompt is joined into a period-separated string."""
    seg = SAM3ImageSegmenter(prompt=['building', 'tree'])
    assert seg.prompt == 'building. tree'


def test_init_string_prompt_and_model_kwargs(fake_sam):
    """String prompts pass through and model kwargs reach SAM3Inference."""
    seg = SAM3ImageSegmenter(
        prompt='road', model_id='x/y', device='cpu', load_in_4bit=False
    )
    assert seg.prompt == 'road'
    inst = fake_sam['instance']
    assert inst.model_id == 'x/y'
    assert inst.device == 'cpu'
    assert inst.load_in_4bit is False
    assert seg.model is inst


def test_init_stores_thresholds_and_event(fake_sam):
    """Thresholds, batch size, filter and cancel event are stored."""
    event = threading.Event()
    flt = lambda img: True  # noqa: E731
    seg = SAM3ImageSegmenter(
        prompt='x',
        batch_size=7,
        threshold=0.3,
        mask_threshold=0.2,
        image_filter=flt,
        cancel_event=event,
    )
    assert seg.batch_size == 7
    assert seg.threshold == 0.3
    assert seg.mask_threshold == 0.2
    assert seg.image_filter is flt
    assert seg.cancel_event is event


# ==========================================
# 3. Segmentation flow
# ==========================================


def test_call_segments_all_downloaded_images(fake_sam, collection):
    """Masks and boxes are stored per image and the model id is recorded."""
    seg = SAM3ImageSegmenter(prompt='building', batch_size=10)
    result = seg(collection)
    assert result is collection

    inst = fake_sam['instance']
    assert len(inst.calls) == 1
    assert inst.calls[0]['prompt'] == 'building'
    assert inst.calls[0]['threshold'] == 0.5
    # The missing image is skipped:
    assert len(inst.calls[0]['inputs']) == 3

    a1 = collection['a1']
    assert set(a1.attributes['sam3_masks']) == {'a1_aerial', 'a1_street'}
    assert a1.attributes['sam3_masks']['a1_aerial'].shape == (2, 4, 4)
    assert set(a1.attributes['sam3_bounding_boxes']) == {'a1_aerial', 'a1_street'}
    assert a1.attributes['ai_model_used'] == 'facebook/sam3'

    a2 = collection['a2']
    assert list(a2.attributes['sam3_masks']) == ['a2_aerial']


def test_call_respects_image_filter(fake_sam, collection):
    """Only images passing image_filter are sent to the model."""
    seg = SAM3ImageSegmenter(
        prompt='building',
        image_filter=lambda img: img.properties.get('kind') == 'street',
    )
    seg(collection)
    inst = fake_sam['instance']
    assert len(inst.calls) == 1
    assert [Path(p).name for p in inst.calls[0]['inputs']] == ['a1_street.jpg']
    assert 'sam3_masks' not in collection['a2'].attributes


def test_call_batches_images(fake_sam, collection):
    """Images are processed in batches of batch_size."""
    seg = SAM3ImageSegmenter(prompt='building', batch_size=2)
    seg(collection)
    inst = fake_sam['instance']
    assert [len(c['inputs']) for c in inst.calls] == [2, 1]


def test_call_no_images_warns(fake_sam, caplog):
    """A collection without downloaded images logs a warning and returns early."""
    seg = SAM3ImageSegmenter(prompt='building')
    col = PhysicalAssetCollection([PhysicalAsset(id='x', geometry=Point(0, 0))])
    with caplog.at_level(logging.WARNING):
        result = seg(col)
    assert result is col
    assert 'No valid downloaded images' in caplog.text
    assert fake_sam['instance'].calls == []


def test_call_single_array_outputs_are_wrapped(fake_sam, tmp_path):
    """Non-list masks/boxes (single image batch) are wrapped into lists."""
    seg = SAM3ImageSegmenter(prompt='building')
    fake_sam['instance'].mode = 'single_array'
    asset = PhysicalAsset(id='s', geometry=Point(0, 0))
    asset.add_image_assets(ImageAsset(id='img', path=_make_image(tmp_path / 's.jpg')))
    seg(PhysicalAssetCollection([asset]))
    assert asset.attributes['sam3_masks']['img'].shape == (1, 4, 4)
    assert asset.attributes['sam3_bounding_boxes']['img'].shape == (1, 4)


def test_call_without_bounding_boxes(fake_sam, collection):
    """When the model returns no boxes, only masks are stored."""
    seg = SAM3ImageSegmenter(prompt='building')
    fake_sam['instance'].mode = 'no_boxes'
    seg(collection)
    a1 = collection['a1']
    assert set(a1.attributes['sam3_masks']) == {'a1_aerial', 'a1_street'}
    assert a1.attributes['sam3_bounding_boxes'] == {}


def test_call_partial_boxes_skips_none(fake_sam, collection):
    """A None entry in the boxes list is not written to the attributes."""
    seg = SAM3ImageSegmenter(prompt='building', batch_size=10)
    fake_sam['instance'].mode = 'partial_boxes'
    seg(collection)
    a1 = collection['a1']
    assert list(a1.attributes['sam3_bounding_boxes']) == ['a1_aerial']


# ==========================================
# 4. Failure handling
# ==========================================


def test_call_counts_failures_when_result_none(fake_sam, collection, caplog):
    """A None result marks the whole batch as failed and logs an error."""
    seg = SAM3ImageSegmenter(prompt='building', batch_size=10)
    fake_sam['instance'].mode = 'none'
    with caplog.at_level(logging.ERROR):
        seg(collection)
    assert '3 images failed' in caplog.text
    assert 'sam3_masks' not in collection['a1'].attributes


def test_call_counts_failures_when_masks_none(fake_sam, collection, caplog):
    """A result without masks is treated as a failure."""
    seg = SAM3ImageSegmenter(prompt='building', batch_size=2)
    fake_sam['instance'].mode = 'no_masks'
    with caplog.at_level(logging.ERROR):
        seg(collection)
    assert '3 images failed' in caplog.text


def test_call_counts_failures_on_exception(fake_sam, collection, caplog):
    """Exceptions inside a batch are swallowed, counted and logged."""
    seg = SAM3ImageSegmenter(prompt='building', batch_size=1)
    fake_sam['instance'].mode = 'raise'
    with caplog.at_level(logging.DEBUG):
        seg(collection)
    assert 'Unhandled exception processing batch: boom' in caplog.text
    assert '3 images failed' in caplog.text


def test_call_success_logs_info(fake_sam, collection, caplog):
    """A fully successful run logs the success message."""
    seg = SAM3ImageSegmenter(prompt='building')
    with caplog.at_level(logging.INFO):
        seg(collection)
    assert 'segmented successfully' in caplog.text


# ==========================================
# 5. Cancellation
# ==========================================


def test_call_cancelled_before_first_batch(fake_sam, collection):
    """A pre-set cancel event aborts before any inference."""
    event = threading.Event()
    event.set()
    seg = SAM3ImageSegmenter(prompt='building', cancel_event=event)
    with pytest.raises(OperationCancelled, match='SAM 3 segmentation'):
        seg(collection)
    assert fake_sam['instance'].calls == []


def test_call_cancelled_between_batches(fake_sam, collection):
    """Cancellation set during batch one stops batch two from running."""
    event = threading.Event()
    seg = SAM3ImageSegmenter(prompt='building', batch_size=1, cancel_event=event)
    fake_sam['instance'].on_call = event.set
    with pytest.raises(OperationCancelled):
        seg(collection)
    assert len(fake_sam['instance'].calls) == 1
