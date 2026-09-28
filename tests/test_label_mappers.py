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

import pytest

from rapidtools.data_sources.mapillary_labels import MapillaryLabels
from rapidtools.models.base import ModelOutput
from rapidtools.processing.label_mappers import MapillaryLabelMapper

CAR = MapillaryLabels.OBJECT_VEHICLE_CAR
TRUCK = MapillaryLabels.OBJECT_VEHICLE_TRUCK
HYDRANT = MapillaryLabels.OBJECT_FIRE_HYDRANT

# ==========================================
# Helpers & Fixtures
# ==========================================


class FakeModel:
    """Minimal stand-in for a rapidtools inference model."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def run_inference(self, image_inputs, prompt, **kwargs):
        """Record the call and return the canned response (or raise it)."""
        self.calls.append({'image_inputs': image_inputs, 'prompt': prompt, **kwargs})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def mapper_for(text):
    """Build a mapper whose model always answers with ``text``."""
    output = None if text is None else ModelOutput(text=text)
    return MapillaryLabelMapper(FakeModel(output))


# ==========================================
# 1. Construction
# ==========================================


def test_valid_labels_extracted_from_class():
    """All string constants on MapillaryLabels become valid labels."""
    mapper = mapper_for('[]')
    expected = [
        v
        for k, v in vars(MapillaryLabels).items()
        if not k.startswith('__') and isinstance(v, str)
    ]
    assert mapper.valid_labels == expected
    assert len(mapper.valid_labels) > 100
    assert CAR in mapper.valid_labels
    assert all(isinstance(v, str) for v in mapper.valid_labels)


def test_model_is_stored():
    """The supplied model instance is kept on the mapper."""
    model = FakeModel(ModelOutput(text='[]'))
    assert MapillaryLabelMapper(model).llm_model is model


# ==========================================
# 2. Prompt & Model Invocation
# ==========================================


def test_prompt_contains_classes_and_labels():
    """The prompt lists the target objects and the valid label vocabulary."""
    mapper = mapper_for(f'["{CAR}"]')
    mapper.map_classes(['cars', 'fire hydrant'])

    call = mapper.llm_model.calls[0]
    assert call['image_inputs'] == []
    assert "['cars', 'fire hydrant']" in call['prompt']
    assert CAR in call['prompt'] and HYDRANT in call['prompt']
    assert 'JSON array' in call['prompt']
    assert call['max_tokens'] == 512
    assert call['temperature'] == pytest.approx(0.1)


def test_empty_input_short_circuits():
    """Empty or None input returns [] without calling the model."""
    mapper = mapper_for(f'["{CAR}"]')
    assert mapper.map_classes([]) == []
    assert mapper.map_classes(None) == []
    assert mapper.llm_model.calls == []


# ==========================================
# 3. Response Parsing
# ==========================================


def test_plain_json_array():
    """A bare JSON array is mapped directly."""
    assert mapper_for(f'["{CAR}", "{TRUCK}"]').map_classes(['vehicles']) == [CAR, TRUCK]


def test_json_embedded_in_prose():
    """JSON surrounded by explanations and markdown fences is still found."""
    text = (
        'Sure! Here are the matching labels:\n```json\n'
        f'["{HYDRANT}"]\n```\nLet me know if you need more.'
    )
    assert mapper_for(text).map_classes(['hydrants']) == [HYDRANT]


def test_multiline_json_array():
    """Arrays spread over several lines are parsed (DOTALL regex)."""
    text = f'[\n  "{CAR}",\n  "{TRUCK}"\n]'
    assert mapper_for(text).map_classes(['vehicles']) == [CAR, TRUCK]


def test_bracketed_prose_before_json_is_skipped():
    """Non-JSON bracketed text ahead of the real array does not break parsing."""
    text = f'[Note] Matching labels: ["{CAR}"]'
    assert mapper_for(text).map_classes(['cars']) == [CAR]


def test_hallucinated_labels_filtered():
    """Labels not in the official vocabulary are dropped."""
    text = f'["{CAR}", "object--vehicle--spaceship", "totally-made-up"]'
    assert mapper_for(text).map_classes(['cars']) == [CAR]


def test_duplicates_removed_preserving_order():
    """Repeated labels are deduplicated while keeping first-seen order."""
    text = f'["{TRUCK}", "{CAR}", "{TRUCK}", "{CAR}"]'
    assert mapper_for(text).map_classes(['vehicles']) == [TRUCK, CAR]


def test_non_string_elements_ignored():
    """Non-string array elements never match the vocabulary."""
    text = f'[1, null, {{"label": "{CAR}"}}, "{CAR}"]'
    assert mapper_for(text).map_classes(['cars']) == [CAR]


def test_empty_json_array():
    """An empty array from the model yields an empty result."""
    assert mapper_for('[]').map_classes(['unicorns']) == []


def test_model_returns_none():
    """A None result from the model yields an empty list."""
    assert mapper_for(None).map_classes(['cars']) == []


def test_model_returns_empty_text():
    """Empty or None text yields an empty list."""
    assert mapper_for('').map_classes(['cars']) == []
    assert (
        MapillaryLabelMapper(FakeModel(ModelOutput(text=None))).map_classes(['cars'])
        == []
    )


def test_invalid_json_returns_empty_list(caplog):
    """Unparsable output returns [] (never None) and logs an error."""
    mapper = mapper_for('I could not find any matching labels, sorry.')
    with caplog.at_level(logging.ERROR):
        result = mapper.map_classes(['cars'])
    assert result == []
    assert isinstance(result, list)
    assert 'Failed to map labels' in caplog.text


def test_broken_json_array_returns_empty_list():
    """A truncated array that never parses returns []."""
    assert mapper_for(f'["{CAR}", "{TRUCK}"').map_classes(['cars']) == []


def test_json_object_instead_of_array_returns_empty_list():
    """A JSON object (not a list) is rejected."""
    assert mapper_for(f'{{"labels": "{CAR}"}}').map_classes(['cars']) == []


def test_model_exception_is_caught(caplog):
    """Exceptions raised by the model are logged and yield []."""
    mapper = MapillaryLabelMapper(FakeModel(RuntimeError('GPU on fire')))
    with caplog.at_level(logging.ERROR):
        assert mapper.map_classes(['cars']) == []
    assert 'GPU on fire' in caplog.text


def test_successful_mapping_is_logged(caplog):
    """Successful mappings are logged at INFO level."""
    with caplog.at_level(logging.INFO):
        mapper_for(f'["{CAR}"]').map_classes(['cars'])
    assert f"Mapped ['cars'] -> ['{CAR}']" in caplog.text


# ==========================================
# 4. _extract_json_list
# ==========================================


def test_extract_json_list_variants():
    """The static extractor handles prose, whole-text and non-list payloads."""
    extract = MapillaryLabelMapper._extract_json_list
    assert extract('Result: ["a", "b"]') == ['a', 'b']
    assert extract('["a"]') == ['a']
    assert extract('[bad] then ["ok"]') == ['ok']
    assert extract('no brackets here') is None
    assert extract('{"a": 1}') is None
    assert extract('[1, 2') is None
    assert extract('') is None
