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

import pytest
from shapely.geometry import Point

from rapidtools.core import OperationCancelled, PhysicalAsset, PhysicalAssetCollection
from rapidtools.processing import Pipeline

# ==========================================
# 1. Test Doubles
# ==========================================


class _Recorder:
    """Shared call log so step doubles can record their execution order."""

    def __init__(self):
        self.calls: list[str] = []


class DummyExtractor:
    def __init__(self, recorder):
        self.recorder = recorder

    def __call__(self, collection):
        self.recorder.calls.append('extractor')
        return collection


class DummyPredictor:
    def __init__(self, recorder):
        self.recorder = recorder

    def __call__(self, collection):
        self.recorder.calls.append('predictor')
        return collection


class DummyClassifier:
    def __init__(self, recorder):
        self.recorder = recorder

    def __call__(self, collection):
        self.recorder.calls.append('classifier')
        return collection


class DummyReporter:
    def __init__(self, recorder):
        self.recorder = recorder

    def __call__(self, collection):
        self.recorder.calls.append('reporter')
        return collection


class DummyExporter:
    def __init__(self, recorder):
        self.recorder = recorder

    def __call__(self, collection):
        self.recorder.calls.append('exporter')
        return collection


class CustomThing:
    def __init__(self, recorder):
        self.recorder = recorder

    def __call__(self, collection):
        self.recorder.calls.append('custom')
        return collection


class CancellingExtractor:
    """Step that sets the shared cancel event while it runs."""

    def __init__(self, event):
        self.event = event

    def __call__(self, collection):
        self.event.set()
        return collection


@pytest.fixture
def collection():
    return PhysicalAssetCollection(
        [PhysicalAsset(id='a1', geometry=Point(0, 0), attributes={'k': 1})]
    )


# ==========================================
# 2. Construction & add_step
# ==========================================


def test_init_defaults():
    """An empty pipeline has no steps and no cancel event."""
    pipeline = Pipeline()
    assert pipeline.steps == []
    assert pipeline.cancel_event is None


def test_init_with_steps_and_event():
    """Steps and cancel_event passed to the constructor are stored."""
    event = threading.Event()
    step = CustomThing(_Recorder())
    pipeline = Pipeline(steps=[step], cancel_event=event)
    assert pipeline.steps == [step]
    assert pipeline.cancel_event is event


def test_add_step_chaining():
    """add_step returns the pipeline so calls can be chained."""
    rec = _Recorder()
    pipeline = Pipeline()
    result = pipeline.add_step(DummyExtractor(rec)).add_step(DummyPredictor(rec))
    assert result is pipeline
    assert len(pipeline.steps) == 2


# ==========================================
# 3. Ordering
# ==========================================


def test_sort_steps_priority_order(collection):
    """Steps run as extractor -> predictor/classifier -> reporter/exporter -> other."""
    rec = _Recorder()
    pipeline = Pipeline(
        [
            CustomThing(rec),
            DummyExporter(rec),
            DummyClassifier(rec),
            DummyReporter(rec),
            DummyPredictor(rec),
            DummyExtractor(rec),
        ]
    )
    pipeline.run(collection)
    assert rec.calls == [
        'extractor',
        'classifier',
        'predictor',
        'exporter',
        'reporter',
        'custom',
    ]


def test_sort_is_stable_within_priority(collection):
    """Steps sharing a priority keep their insertion order."""
    rec = _Recorder()
    pipeline = Pipeline([DummyPredictor(rec), DummyClassifier(rec)])
    pipeline.run(collection)
    assert rec.calls == ['predictor', 'classifier']


def test_sort_steps_direct():
    """_sort_steps reorders the steps list in place."""
    rec = _Recorder()
    ext, rep = DummyExtractor(rec), DummyReporter(rec)
    pipeline = Pipeline([rep, ext])
    pipeline._sort_steps()
    assert pipeline.steps == [ext, rep]


def test_plain_function_step_runs_last(collection):
    """A bare function (no recognisable class name) is treated as a custom step."""
    rec = _Recorder()

    def my_func(col):
        rec.calls.append('func')
        return col

    pipeline = Pipeline([my_func, DummyExtractor(rec)])
    pipeline.run(collection)
    assert rec.calls == ['extractor', 'func']


# ==========================================
# 4. Execution
# ==========================================


def test_run_empty_pipeline_warns(collection, caplog):
    """Running with no steps logs a warning and returns the input untouched."""
    pipeline = Pipeline()
    with caplog.at_level(logging.WARNING):
        result = pipeline.run(collection)
    assert result is collection
    assert 'Pipeline is empty' in caplog.text


def test_run_passes_collection_through_steps():
    """Each step receives the previous step's output."""
    outputs = []

    class StepExtractor:
        def __call__(self, col):
            new = PhysicalAssetCollection(
                [PhysicalAsset(id='new', geometry=Point(1, 1))]
            )
            outputs.append(new)
            return new

    class StepPredictor:
        def __call__(self, col):
            col['new'].attributes['seen'] = True
            return col

    pipeline = Pipeline([StepPredictor(), StepExtractor()])
    result = pipeline.run(PhysicalAssetCollection())
    assert result is outputs[0]
    assert result['new'].attributes['seen'] is True


def test_run_logs_step_names(collection, caplog):
    """Step names appear in the info log."""
    rec = _Recorder()
    pipeline = Pipeline([DummyExtractor(rec)])
    with caplog.at_level(logging.INFO):
        pipeline.run(collection)
    assert 'DummyExtractor' in caplog.text
    assert 'successfully completed' in caplog.text


# ==========================================
# 5. Cancellation
# ==========================================


def test_run_cancelled_before_start(collection):
    """A pre-set cancel event aborts before the first step runs."""
    rec = _Recorder()
    event = threading.Event()
    event.set()
    pipeline = Pipeline([DummyExtractor(rec)], cancel_event=event)
    with pytest.raises(OperationCancelled, match='DummyExtractor'):
        pipeline.run(collection)
    assert rec.calls == []


def test_run_cancelled_between_steps(collection):
    """Cancellation raised by a step is honoured before the next step."""
    rec = _Recorder()
    event = threading.Event()
    pipeline = Pipeline(
        [DummyPredictor(rec), CancellingExtractor(event)], cancel_event=event
    )
    with pytest.raises(OperationCancelled):
        pipeline.run(collection)
    # The extractor ran (and set the event); the predictor never did.
    assert rec.calls == []
