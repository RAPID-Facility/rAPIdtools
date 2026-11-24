"""Tests for the JSON job interface (rapidtools.jobs)."""

import dataclasses
import json
import os
import stat
import sys
import threading
from pathlib import Path

import pytest
from shapely.geometry import box

import rapidtools
from rapidtools import jobs
from rapidtools.core import OperationCancelled, PhysicalAsset, PhysicalAssetCollection
from rapidtools.gui import workflow as workflow_module
from rapidtools.gui.workflow import (
    DetectionResult,
    InferenceResult,
    StreetDetectionResult,
)
from rapidtools.jobs import (
    Input,
    Job,
    JobError,
    JobRunner,
    Param,
    Secrets,
    StepSpec,
    get_step_type,
    new_job,
    run_job,
)
from rapidtools.jobs import __main__ as cli
from rapidtools.jobs import steps as steps_module
from rapidtools.jobs.schema import (
    check_schema,
    is_secret_ref,
    is_step_ref,
    read_json,
    write_json_atomic,
)

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / 'docs' / 'source'


def _collection(n: int = 2) -> PhysicalAssetCollection:
    collection = PhysicalAssetCollection()
    for i in range(n):
        collection.add(
            PhysicalAsset(
                id=f'A{i}',
                geometry=box(i, 0, i + 1, 1),
                attributes={'asset_type': 'building'},
            )
        )
    return collection


class FakeWorkflow:
    """Stand-in for AssetAnalysisWorkflow that records the settings it gets."""

    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
        self.calls = []
        self.cancelled = False
        self.n_detected = 2

    def cancel(self):
        self.cancelled = True

    def download_basemap(self, settings):
        self.calls.append(('basemap', settings))
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        path = settings.output_dir / 'basemap.tiff'
        path.write_bytes(b'tiff')
        return path

    def detect(self, settings):
        self.calls.append(('detect', settings))
        if self.progress_callback:
            self.progress_callback('Scanning...')
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        collection = _collection(self.n_detected)
        combined = None
        per_asset = {}
        if len(collection):
            combined = settings.output_dir / 'assets_final.geojson'
            collection.to_geojson(combined)
            per_asset = {settings.assets[0]: combined}
        return DetectionResult(
            collection=collection,
            detection_raster=settings.output_dir / 'basemap.tiff',
            per_asset_geojson=per_asset,
            combined_geojson=combined,
            basemap='bing',
        )

    def discover_street(self, settings):
        self.calls.append(('street', settings))
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        collection = _collection(1)
        path = settings.output_dir / 'street_objects.geojson'
        collection.to_geojson(path)
        return StreetDetectionResult(
            collection=collection, geojson_path=path, n_images=4
        )

    def analyze(self, collection, settings):
        self.calls.append(('analyze', settings, collection))
        for asset in collection:
            asset.add_attributes({'damage': 'minor'})
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        path = settings.output_dir / 'assets_inferred.geojson'
        collection.to_geojson(path)
        return InferenceResult(
            collection=collection,
            geojson_path=path,
            n_input=len(collection),
            n_analyzed=len(collection),
        )

    def assist(self, settings):
        self.calls.append(('assist', settings))
        return {'text': 'Looks good.', 'spec': None}


DETECT = {
    'id': 'detect',
    'type': 'detect.aerial',
    'inputs': {'raster': 'scene.tiff'},
    'params': {'assets': ['building']},
}
ANALYZE = {
    'id': 'grade',
    'type': 'analyze.vlm',
    'inputs': {
        'raster': 'scene.tiff',
        'assets': {'step': 'detect', 'output': 'assets'},
    },
    'params': {
        'prompt': 'Describe the damage.',
        'backend': 'gemini',
        'api_key': {'$secret': 'api_key'},
    },
}


def _job(tmp_path, steps, name='Demo') -> Job:
    job = new_job(steps, output_dir='out', name=name, base_dir=tmp_path)
    job.save(tmp_path / 'jobs' / 'demo.json')
    return job


# ---------------------------------------------------------------- documents
def test_job_round_trip_and_paths(tmp_path):
    job = new_job(
        [DETECT, ANALYZE],
        output_dir='out',
        name='Detect + grade',
        created_by={'client': 'pytest'},
        metadata={'event': 'test'},
        base_dir=tmp_path,
    )
    assert job.id.endswith('-detect-grade') and job.created_at.endswith('Z')
    assert job.result_path is None  # not on disk yet
    path = job.save(tmp_path / 'jobs' / 'j.json')
    assert job.source == path and job.result_path == path.with_name('j.result.json')

    document = json.loads(path.read_text())
    assert document['schema'] == 'rapidtools.job'
    assert document['schema_version'] == jobs.SCHEMA_VERSION
    assert document['steps'][1]['inputs']['assets'] == {
        'step': 'detect',
        'output': 'assets',
    }
    assert 'metadata' not in document['steps'][0]

    loaded = Job.load(path)
    assert loaded.to_dict() == document
    # Relative paths are taken from the folder of the job file:
    assert loaded.base_dir == path.parent
    assert loaded.output_path == path.parent / 'out'
    absolute = tmp_path / 'elsewhere' / 'x.tif'  # absolute on every platform
    assert loaded.resolve_path(str(absolute)) == absolute
    assert loaded.resolve_path('~/x.tif') == Path.home() / 'x.tif'
    assert jobs.load_job(loaded) is loaded
    assert jobs.load_job(document).steps[0].id == 'detect'
    assert jobs.load_job(path).name == 'Detect + grade'


def test_step_ids_default_and_metadata_survive():
    job = new_job(
        [
            {'type': 'detect.aerial', 'metadata': {'note': 'x'}},
            StepSpec(id='b', type='analyze.vlm'),
        ],
        output_dir='out',
    )
    assert [s.id for s in job.steps] == ['s1', 'b']
    assert job.to_dict()['steps'][0]['metadata'] == {'note': 'x'}
    assert job.name == job.id


@pytest.mark.parametrize(
    ('change', 'message'),
    [
        ({'schema': 'other'}, '"schema" must be'),
        ({'schema_version': 1}, '"schema_version" must be a string'),
        ({'schema_version': '2.0'}, 'version 1.x'),
        ({'extra': 1}, 'Unknown field(s) in the job: extra'),
        ({'steps': {}}, '"steps" must be a list'),
        ({'steps': []}, 'at least one step'),
        ({'steps': ['x']}, 'steps[0] must be a JSON object'),
        ({'steps': [{'type': ''}]}, 'steps[0].type is required'),
        ({'steps': [{'type': 'a', 'id': 'no spaces'}]}, 'steps[0].id must start'),
        ({'steps': [{'type': 'a', 'params': []}]}, 'steps[0].params must be'),
        ({'steps': [{'type': 'a', 'when': 'x'}]}, 'Unknown field(s) in steps[0]'),
        ({'output_dir': None}, '"output_dir" is required'),
        ({'output_dir': ' '}, '"output_dir" is required'),
        ({'id': 'bad id'}, '"id" must start'),
        ({'name': 5}, '"name" must be a string'),
        ({'metadata': []}, '"metadata" must be a JSON object'),
    ],
)
def test_invalid_job_documents(change, message):
    document = {
        'schema': 'rapidtools.job',
        'schema_version': '1.0',
        'output_dir': 'out',
        'steps': [dict(DETECT)],
        **change,
    }
    with pytest.raises(JobError) as error:
        Job.from_dict(document)
    assert message in str(error.value)


def test_job_must_be_an_object_and_refs_point_backwards():
    with pytest.raises(JobError, match='must be a JSON object'):
        Job.from_dict([])  # type: ignore[arg-type]
    with pytest.raises(JobError, match='Two steps share the id'):
        new_job([DETECT, DETECT], output_dir='out')
    with pytest.raises(JobError, match='not an earlier step') as error:
        new_job([ANALYZE, DETECT], output_dir='out')
    assert error.value.step_id == 'grade'
    assert error.value.describe().startswith("Step 'grade' (analyze.vlm): ")
    assert JobError('plain').describe() == 'plain'
    assert JobError('m', step_id='s').describe() == "Step 's': m"


def test_newer_minor_version_is_accepted_with_a_hint():
    document = {
        'schema': 'rapidtools.job',
        'schema_version': '1.7',
        'output_dir': 'out',
        'steps': [dict(DETECT)],
    }
    assert Job.from_dict(document).schema_version == '1.7'
    with pytest.raises(JobError, match='newer than the 1.0'):
        Job.from_dict({**document, 'priority': 'high'})
    with pytest.raises(JobError, match='newer than the 1.0'):
        Job.from_dict({**document, 'steps': [{**DETECT, 'retries': 2}]})
    assert check_schema(document, 'rapidtools.job') == '1.7'


def test_read_json_errors_and_atomic_write(tmp_path):
    with pytest.raises(JobError, match='Could not read the job file'):
        read_json(tmp_path / 'missing.json')
    bad = tmp_path / 'bad.json'
    bad.write_text('{nope')
    with pytest.raises(JobError, match='not valid JSON'):
        Job.load(bad)
    bad.write_text('[1]')
    with pytest.raises(JobError, match='must hold a JSON object'):
        read_json(bad)

    target = write_json_atomic(tmp_path / 'deep' / 'a.json', {'a': 1}, private=True)
    assert json.loads(target.read_text()) == {'a': 1}
    if os.name == 'posix':  # Windows has no owner-only mode bits
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert [p.name for p in target.parent.iterdir()] == ['a.json']  # no temp left
    # A value JSON cannot hold leaves neither the target nor a temp file:
    with pytest.raises(TypeError):
        write_json_atomic(tmp_path / 'deep' / 'b.json', {'a': object()})
    assert [p.name for p in target.parent.iterdir()] == ['a.json']


def test_reference_predicates():
    assert is_step_ref({'step': 'a', 'output': 'b'})
    assert not is_step_ref({'step': 'a'}) and not is_step_ref('a')
    assert is_secret_ref({'$env': 'X'}) and is_secret_ref({'$secret': 'x'})
    assert not is_secret_ref({'$secret': 'x', 'y': 1}) and not is_secret_ref({})


# ------------------------------------------------------------------ secrets
def test_secrets_resolution(tmp_path):
    env = {'RAPIDTOOLS_SECRET_MAPILLARY_TOKEN': 'MLY|env', 'GEMINI_API_KEY': 'g-env'}
    secrets = Secrets({'api_key': 'k-file'}, environ=env)
    assert secrets.resolve('literal') == 'literal'
    assert secrets.resolve({'$secret': 'api_key'}) == 'k-file'
    assert secrets.resolve({'$secret': 'mapillary-token'}) == 'MLY|env'
    assert secrets.resolve({'$env': 'GEMINI_API_KEY'}) == 'g-env'
    assert secrets.resolve({'$env': 'NOPE'}, missing='<secret>') == '<secret>'
    with pytest.raises(JobError, match='environment variable NOPE is not set'):
        secrets.resolve({'$env': 'NOPE'})
    with pytest.raises(JobError, match='set RAPIDTOOLS_SECRET_OTHER'):
        secrets.resolve({'$secret': 'other'})
    with pytest.raises(JobError, match='non-empty name'):
        secrets.resolve({'$secret': ' '})
    assert 'k-file' not in repr(secrets) and 'api_key' in repr(secrets)

    path = write_json_atomic(
        tmp_path / 'secrets.json', Secrets.document({'api_key': 'k'}), private=True
    )
    assert Secrets.load(path).resolve({'$secret': 'api_key'}) == 'k'
    assert Secrets.coerce(path).resolve({'$secret': 'api_key'}) == 'k'
    assert Secrets.coerce(secrets) is secrets
    assert Secrets.coerce({'a': 'b'}).resolve({'$secret': 'a'}) == 'b'
    path.write_text(
        json.dumps(
            {'schema': 'rapidtools.secrets', 'schema_version': '1.0', 'secrets': [1]}
        )
    )
    with pytest.raises(JobError, match='must map secret names'):
        Secrets.load(path)


def test_secrets_default_to_the_process_environment(monkeypatch):
    monkeypatch.setenv('RAPIDTOOLS_SECRET_API_KEY', 'from-env')
    assert Secrets().resolve({'$secret': 'api_key'}) == 'from-env'


# ---------------------------------------------------------------- step types
def test_param_coercion():
    assert Param('n', 'integer', '').coerce(19.0) == 19
    assert Param('n', 'number', '').coerce(3) == 3.0
    assert Param('n', 'number_pair', '').coerce([0, 1]) == (0.0, 1.0)
    assert Param('n', 'number_or_string', '').coerce('2 m') == '2 m'
    assert Param('n', 'string_list', '').coerce(['a']) == ['a']
    assert Param('n', 'object', '').coerce({'a': 1}) == {'a': 1}
    assert Param('n', 'boolean', '', nullable=True).coerce(None) is None
    assert Param('n', 'string', '', choices=lambda: ('a',)).options() == ('a',)
    for kind, bad in [
        ('string', 1),
        ('boolean', 'yes'),
        ('integer', 1.5),
        ('integer', True),
        ('number', '3'),
        ('number_or_string', None),
        ('number_or_string', [1]),
        ('string_list', []),
        ('string_list', 'a, b'),
        ('string_list', ['a', ' ']),
        ('number_pair', [1]),
        ('number_pair', [1, 'x']),
        ('object', []),
    ]:
        with pytest.raises(ValueError, match='^n must'):
            Param('n', kind, '').coerce(bad)
    bounded = Param('n', 'number', '', minimum=0, maximum=1)
    with pytest.raises(ValueError, match='at least 0'):
        bounded.coerce(-0.1)
    with pytest.raises(ValueError, match='at most 1'):
        bounded.coerce(2)
    with pytest.raises(ValueError, match='greater than 0'):
        Param('n', 'number', '', exclusive_minimum=0).coerce(0)
    with pytest.raises(ValueError, match="one of \\('a', 'b'\\), got 'c'"):
        Param('n', 'string', '', choices=('a', 'b')).coerce('c')
    with pytest.raises(ValueError, match='Unknown parameter kind'):
        Param('n', 'date', '')
    with pytest.raises(ValueError, match='Unknown input kind'):
        Input('n', 'folder', '')
    assert repr(jobs.REQUIRED) == 'REQUIRED'


def test_read_params_defaults_unknowns_and_secrets():
    analyze = get_step_type('analyze.vlm')
    values = analyze.read_params({'prompt': 'p'})
    assert values['backend'] == 'gemini' and values['street_vertical_crop'] == (0, 1)
    assert set(values) == {p.name for p in analyze.params}
    with pytest.raises(ValueError, match="Did you mean 'max_tokens'"):
        analyze.read_params({'prompt': 'p', 'max_token': 5})
    with pytest.raises(ValueError, match='prompt is required'):
        analyze.read_params({})
    with pytest.raises(ValueError, match='cannot be a secret reference'):
        analyze.read_params({'prompt': {'$env': 'X'}})
    with pytest.raises(JobError, match='A secret is missing'):
        analyze.read_params({'prompt': 'p', 'api_key': {'$secret': 'k'}})
    lenient = analyze.read_params(
        {'prompt': 'p', 'api_key': {'$secret': 'k'}}, allow_missing_secrets=True
    )
    assert lenient['api_key'] == '<secret>'
    # Mutable defaults are copied, never shared between steps:
    assist = get_step_type('prompt.assist')
    first = assist.read_params({'action': 'draft'})
    first['context']['x'] = 1
    assert assist.read_params({'action': 'draft'})['context'] == {}


def test_param_defaults_match_the_workflow_settings():
    """The interface and the engine must agree on every default."""
    pairs = {
        'imagery.basemap': workflow_module.RegionImagerySettings,
        'detect.aerial': workflow_module.DetectionSettings,
        'detect.street': workflow_module.StreetDetectionSettings,
        'analyze.vlm': workflow_module.InferenceSettings,
        'prompt.assist': workflow_module.AssistSettings,
    }
    handled_as_inputs = {
        'output_dir',
        'raster_path',
        'region',
        'geojson_path',
        'min_lon',
        'min_lat',
        'max_lon',
        'max_lat',
        'detect_in_recon_imagery',  # legacy spelling of basemap='recon'
    }
    for name, settings_class in pairs.items():
        fields = {f.name: f for f in dataclasses.fields(settings_class)}
        step_type = get_step_type(name)
        mapped = set()
        for param in step_type.params:
            field = fields[param.field or param.name]
            mapped.add(field.name)
            if field.default is not dataclasses.MISSING:
                assert param.default == field.default, (name, param.name)
            elif field.default_factory is not dataclasses.MISSING:
                assert param.default == field.default_factory(), (name, param.name)
            else:  # required by the engine: required here, or an empty secret
                assert param.required or param.secret, (name, param.name)
        assert set(fields) - mapped <= handled_as_inputs, name


def test_registry():
    with pytest.raises(JobError, match="Did you mean 'detect.aerial'"):
        get_step_type('detect.arial')
    existing = get_step_type('data.download')
    with pytest.raises(ValueError, match='already registered'):
        jobs.register_step(existing)
    assert jobs.register_step(existing, replace=True) is existing


def test_inputs_are_resolved_and_checked(tmp_path):
    def prepare(step, lookup=None):
        job = new_job([DETECT, step], output_dir='out', base_dir=tmp_path)
        return steps_module.prepare_step(job, job.steps[1], lookup=lookup)

    street = {
        'id': 'street',
        'type': 'detect.street',
        'params': {'classes': ['vehicles'], 'access_token': 't'},
    }
    ctx = prepare({**street, 'inputs': {'region': {'bbox': [-1, 0, 1, 2]}}})
    assert ctx.settings.region.bounds == (-1.0, 0.0, 1.0, 2.0)
    assert ctx.inputs['raster'] is None
    # A GeoJSON region is only read when the step runs:
    ctx = prepare({**street, 'inputs': {'region': {'geojson': 'area.geojson'}}})
    assert ctx.settings.region.geojson == tmp_path / 'area.geojson'
    with pytest.raises(FileNotFoundError, match='Region file not found'):
        ctx.settings.region.bounding_box()
    collection = _collection(1)
    collection.to_geojson(tmp_path / 'area.geojson')
    ran = prepare(
        {**street, 'inputs': {'region': {'geojson': 'area.geojson'}}},
        lookup=lambda step, output: (None, None),
    )
    assert ran.settings.region.bounds == (0.0, 0.0, 1.0, 1.0)

    for region, message in [
        ({'bbox': [1, 2, 3]}, 'must be {"bbox"'),
        ('box', 'must be {"bbox"'),
        ({'geojson': ''}, 'must be {"bbox"'),
        ({'bbox': [5, 0, 1, 2]}, 'must satisfy min < max'),
    ]:
        with pytest.raises(JobError) as error:
            prepare({**street, 'inputs': {'region': region}})
        assert message in str(error.value)

    with pytest.raises(JobError, match="Unknown input 'image'"):
        prepare({**DETECT, 'id': 'd2', 'inputs': {'image': 'x.tif'}})
    with pytest.raises(JobError, match="Input 'raster' is required"):
        prepare({**DETECT, 'id': 'd2', 'inputs': {}})
    with pytest.raises(JobError, match='must be a path or a reference'):
        prepare({**DETECT, 'id': 'd2', 'inputs': {'raster': 5}})
    with pytest.raises(JobError, match='does not produce'):
        prepare(
            {
                **DETECT,
                'id': 'd2',
                'inputs': {'raster': {'step': 'detect', 'output': 'x'}},
            }
        )
    # While validating, a reference is a placeholder path:
    ctx = prepare(
        {
            **DETECT,
            'id': 'd2',
            'inputs': {'raster': {'step': 'detect', 'output': 'detection_raster'}},
        }
    )
    assert str(ctx.settings.raster_path) == '<detect.detection_raster>'
    # When running, a referenced file that was never written is an error:
    with pytest.raises(JobError, match="produced no 'detection_raster' output"):
        prepare(
            {
                **DETECT,
                'id': 'd2',
                'inputs': {'raster': {'step': 'detect', 'output': 'detection_raster'}},
            },
            lookup=lambda step, output: (None, None),
        )


# ------------------------------------------------------------------- runner
def test_single_step_job_writes_a_result_file(tmp_path):
    workflow = FakeWorkflow()
    job = _job(tmp_path, [DETECT], name='Detection')
    seen = []
    result = JobRunner(workflow=workflow).run(job.source, on_step=seen.append)

    assert result.ok and result.status == 'done'
    assert result.message == 'Detection finished.'
    assert result.raise_for_status() is result
    step = result.step('detect')
    assert seen == [step] and step.status == 'done'
    out = tmp_path / 'jobs' / 'out'
    assert step.outputs == {
        'assets': out / 'assets_final.geojson',
        'detection_raster': out / 'basemap.tiff',
    }
    assert step.summary == {'n_assets': 2, 'basemap': 'bing'}
    assert len(step.objects['assets']) == 2
    settings = workflow.calls[0][1]
    assert settings is step.settings
    assert settings.raster_path == tmp_path / 'jobs' / 'scene.tiff'
    assert settings.output_dir == out and settings.assets == ['building']
    with pytest.raises(KeyError):
        result.step('nope')

    document = json.loads(job.result_path.read_text())
    assert document['schema'] == 'rapidtools.job_result'
    assert document['job_id'] == job.id and document['status'] == 'done'
    assert document['job_file'] == str(job.source)
    assert document['rapidtools_version'] == rapidtools.__version__
    assert document['finished_at'] and document['duration_s'] >= 0
    [step_doc] = document['steps']
    assert step_doc['outputs']['assets'] == {
        'path': str(out / 'assets_final.geojson'),
        'kind': 'geojson',
    }
    assert step_doc['files'] == [
        {
            'label': 'building footprints',
            'path': str(out / 'assets_final.geojson'),
            'kind': 'geojson',
        }
    ]
    assert 'error' not in step_doc and 'data' not in step_doc


def test_chained_steps_hand_over_assets_and_keep_secrets_out(tmp_path):
    workflow = FakeWorkflow()
    job = _job(tmp_path, [DETECT, ANALYZE])
    result = JobRunner(workflow=workflow).run(job, secrets={'api_key': 'k-SECRET'})

    assert result.ok and [s.status for s in result.steps] == ['done', 'done']
    _, settings, collection = workflow.calls[1]
    assert settings.api_key == 'k-SECRET' and settings.prompt == 'Describe the damage.'
    # The collection detection produced is passed on without a file round trip:
    assert collection is result.step('detect').objects['assets']
    assert result.step('grade').summary == {'n_input': 2, 'n_analyzed': 2}
    for path in (job.source, job.result_path):
        assert 'k-SECRET' not in path.read_text()


def test_assets_can_come_from_a_file(tmp_path):
    workflow = FakeWorkflow()
    _collection(3).to_geojson(tmp_path / 'jobs' / 'mine.geojson')
    step = {**ANALYZE, 'inputs': {'assets': 'mine.geojson'}}
    step['params'] = {**ANALYZE['params'], 'imagery': 'google_streetview'}
    job = _job(tmp_path, [step])
    result = run_job(job.source, secrets={'api_key': 'k'}, workflow=workflow)
    assert result.ok
    _, settings, collection = workflow.calls[0]
    assert len(collection) == 3 and settings.raster_path is None

    missing = {**step, 'inputs': {'assets': 'gone.geojson'}}
    result = run_job(_job(tmp_path, [missing]), {'api_key': 'k'}, workflow=workflow)
    assert result.status == 'error' and 'Asset file not found' in result.message
    assert result.step('grade').error['type'] == 'FileNotFoundError'

    PhysicalAssetCollection().to_geojson(tmp_path / 'jobs' / 'empty.geojson')
    empty = {**step, 'inputs': {'assets': 'empty.geojson'}}
    result = run_job(_job(tmp_path, [empty]), {'api_key': 'k'}, workflow=workflow)
    assert 'nothing to analyze' in result.message


def test_nothing_detected_stops_the_job(tmp_path):
    workflow = FakeWorkflow()
    workflow.n_detected = 0
    job = _job(tmp_path, [DETECT, ANALYZE])
    result = JobRunner(workflow=workflow).run(job, secrets={'api_key': 'k'})
    assert result.status == 'error'
    assert result.message == 'No assets were detected; skipping inference.'
    assert [s.status for s in result.steps] == ['done', 'error']
    assert 'assets' not in result.step('detect').outputs
    with pytest.raises(ValueError, match='No assets were detected'):
        result.raise_for_status()


def test_a_failing_step_skips_the_rest(tmp_path):
    workflow = FakeWorkflow()

    def broken(settings):
        raise RuntimeError('GPU on fire')

    workflow.detect = broken
    job = _job(tmp_path, [DETECT, ANALYZE])
    result = JobRunner(workflow=workflow).run(job, secrets={'api_key': 'k'})
    assert result.status == 'error' and result.message == 'GPU on fire'
    assert [s.status for s in result.steps] == ['error', 'skipped']
    document = json.loads(job.result_path.read_text())
    assert document['steps'][0]['error'] == {
        'type': 'RuntimeError',
        'message': 'GPU on fire',
    }
    assert document['steps'][1]['started_at'] is None


def test_cancellation_keeps_partial_results(tmp_path):
    workflow = FakeWorkflow()

    def cancelled(collection, settings):
        next(iter(collection)).add_attributes({'damage': 'major'})
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        (settings.output_dir / 'assets_inferred_partial.geojson').write_text('{}')
        raise OperationCancelled('stop')

    workflow.analyze = cancelled
    job = _job(tmp_path, [DETECT, ANALYZE])
    result = JobRunner(workflow=workflow).run(job, secrets={'api_key': 'k'})
    assert result.status == 'cancelled' and result.message == 'Demo cancelled.'
    grade = result.step('grade')
    assert grade.status == 'cancelled' and grade.error['type'] == 'OperationCancelled'
    assert grade.outputs['partial_assets'].name == 'assets_inferred_partial.geojson'
    assert 'damage' in next(iter(grade.objects['assets'])).attributes
    with pytest.raises(OperationCancelled):
        result.raise_for_status()

    # Without a partial file there is simply no such output:
    workflow.analyze = lambda c, s: (_ for _ in ()).throw(OperationCancelled('x'))
    (tmp_path / 'jobs' / 'out' / 'assets_inferred_partial.geojson').unlink()
    result = JobRunner(workflow=workflow).run(job, secrets={'api_key': 'k'})
    assert result.step('grade').outputs == {}


def test_cancel_between_steps_and_failing_hook(tmp_path):
    workflow = FakeWorkflow()
    runner = JobRunner(workflow=workflow)
    job = _job(tmp_path, [DETECT, ANALYZE])
    result = runner.run(job, {'api_key': 'k'}, on_step=lambda step: runner.cancel())
    assert workflow.cancelled is True
    assert [s.status for s in result.steps] == ['done', 'cancelled']
    assert [c[0] for c in workflow.calls] == ['detect']
    # The request does not leak into the next run:
    assert runner.run(job, {'api_key': 'k'}).ok

    def hook(step):
        raise OSError('disk full')

    result = runner.run(job, {'api_key': 'k'}, on_step=hook)
    assert result.status == 'error' and result.message == 'disk full'
    assert [s.status for s in result.steps] == ['done', 'skipped']


def test_invalid_job_is_reported_before_anything_runs(tmp_path):
    workflow = FakeWorkflow()
    bad = {**ANALYZE, 'params': {**ANALYZE['params'], 'temperature': 'hot'}}
    job = _job(tmp_path, [DETECT, bad])
    result = JobRunner(workflow=workflow).run(job, secrets={'api_key': 'k'})
    assert workflow.calls == []
    assert result.status == 'error'
    assert result.message == (
        "Step 'grade' (analyze.vlm): temperature must be a number."
    )
    assert [s.status for s in result.steps] == ['skipped', 'error']
    assert isinstance(result.exception, JobError)
    document = json.loads(job.result_path.read_text())
    assert document['steps'][1]['error']['type'] == 'JobError'
    assert document['steps'][0]['started_at'] is None

    unknown = _job(tmp_path, [{'id': 'x', 'type': 'detect.sonar'}])
    result = JobRunner(workflow=workflow).run(unknown)
    assert 'Unknown step type' in result.message
    assert json.loads(unknown.result_path.read_text())['steps'][0]['outputs'] == {}

    with pytest.raises(JobError, match='Could not read'):
        JobRunner(workflow=workflow).run(tmp_path / 'nope.json')
    good = _job(tmp_path, [DETECT, ANALYZE])
    with pytest.raises(JobError, match='A secret is missing'):
        JobRunner(workflow=workflow).validate(good)
    assert JobRunner().validate(good, allow_missing_secrets=True) is good


def test_runner_creates_a_workflow_and_mirrors_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow_module, 'AssetAnalysisWorkflow', FakeWorkflow)
    messages = []
    runner = JobRunner(progress=messages.append)
    runner.cancel()  # nothing to cancel yet: no workflow, no job
    job = _job(tmp_path, [DETECT])
    snapshots = []

    def spy(step):
        snapshots.append(json.loads(job.result_path.read_text())['status'])

    result = runner.run(job, on_step=spy)
    assert result.ok and messages == ['Scanning...'] and snapshots == ['running']
    assert isinstance(runner.workflow, FakeWorkflow)
    runner._on_progress('late')  # outside a run: forwarded, nothing to write
    assert messages[-1] == 'late'

    # A job that is not on disk gets no result file, and a custom path works:
    in_memory = new_job([DETECT], output_dir='out', base_dir=tmp_path)
    assert runner.run(in_memory).result_file is None
    custom = tmp_path / 'elsewhere' / 'r.json'
    assert runner.run(in_memory, result_path=custom).result_file == custom
    assert json.loads(custom.read_text())['job_file'] is None


def test_remaining_step_types(tmp_path, monkeypatch):
    workflow = FakeWorkflow()
    downloads = []

    def fake_download(name, output_dir='.'):
        downloads.append(name)
        target = Path(output_dir) / 'eaton.tiff'
        target.write_bytes(b'tiff')
        return [target, Path(output_dir) / 'readme.txt'] if name != 'none' else []

    monkeypatch.setattr(rapidtools, 'download_dataset', fake_download)
    job = _job(
        tmp_path,
        [
            {
                'id': 'sample',
                'type': 'data.download',
                'params': {'dataset': 'Eaton_Patch1 '},
            },
            {
                'id': 'basemap',
                'type': 'imagery.basemap',
                'inputs': {'region': {'bbox': [-117.489, 47.711, -117.487, 47.7125]}},
                'params': {'provider': 'google', 'zoom': 18},
            },
            {
                'id': 'area',
                'type': 'imagery.basemap',
                'inputs': {'region': {'geojson': 'area.geojson'}},
            },
            {
                'id': 'street',
                'type': 'detect.street',
                'inputs': {'raster': {'step': 'basemap', 'output': 'raster'}},
                'params': {'classes': ['vehicles'], 'access_token': 'MLY|x'},
            },
            {
                'id': 'assist',
                'type': 'prompt.assist',
                'params': {'action': 'review', 'prompt_text': 'Rate it.'},
            },
        ],
    )
    result = JobRunner(workflow=workflow).run(job)
    assert result.ok, result.message
    out = tmp_path / 'jobs' / 'out'
    assert downloads == ['eaton_patch1']
    sample = result.step('sample')
    assert sample.outputs == {'file': out / 'eaton.tiff'}
    assert sample.files == [
        {'label': 'readme.txt', 'path': out / 'readme.txt', 'kind': 'file'}
    ]
    basemap, area = workflow.calls[0][1], workflow.calls[1][1]
    assert (basemap.provider, basemap.zoom, basemap.min_lon) == ('google', 18, -117.489)
    assert area.geojson_path == tmp_path / 'jobs' / 'area.geojson'
    assert result.step('basemap').summary == {'provider': 'google', 'zoom': 18}
    street = workflow.calls[2][1]
    assert street.raster_path == out / 'basemap.tiff' and street.region is None
    assert result.step('street').summary == {'n_assets': 1, 'n_images': 4}
    assist = result.step('assist')
    assert assist.data == {'text': 'Looks good.', 'spec': None} and not assist.outputs
    document = json.loads(job.result_path.read_text())
    assert document['steps'][-1]['data'] == {'text': 'Looks good.', 'spec': None}

    with pytest.raises(JobError, match="Unknown dataset 'eaton_patch9'. Did you mean"):
        JobRunner().validate(
            new_job(
                [{'type': 'data.download', 'params': {'dataset': 'eaton_patch9'}}],
                output_dir='out',
            )
        )
    with pytest.raises(JobError, match='Give the region as'):
        JobRunner().validate(new_job([{'type': 'imagery.basemap'}], output_dir='out'))
    monkeypatch.setattr(rapidtools, 'download_dataset', lambda *a, **k: [])
    result = JobRunner(workflow=workflow).run(_job(tmp_path, [job.steps[0].to_dict()]))
    assert 'could be downloaded' in result.message


# ---------------------------------------------------------------- reference
def test_schemas_are_valid_and_accept_the_examples():
    jsonschema = pytest.importorskip('jsonschema')
    validator = jsonschema.Draft202012Validator
    for schema in (jobs.job_schema(), jobs.result_schema(), jobs.secrets_schema()):
        validator.check_schema(schema)
    job_schema = validator(jobs.job_schema())
    examples = sorted((REPO / 'examples' / 'jobs').glob('*.json'))
    assert len(examples) >= 4
    for path in examples:
        document = json.loads(path.read_text())
        if document['schema'] == 'rapidtools.secrets':
            validator(jobs.secrets_schema()).validate(document)
            continue
        job_schema.validate(document)
        JobRunner().validate(path, allow_missing_secrets=True)

    def errors(document):
        return [e.message for e in job_schema.iter_errors(document)]

    base = {'schema': 'rapidtools.job', 'schema_version': '1.0', 'output_dir': 'o'}
    assert errors({**base, 'steps': [DETECT, ANALYZE]}) == []
    assert errors({**base, 'steps': [{'type': 'detect.sonar'}]})
    assert errors({**base, 'steps': [{**DETECT, 'params': {'assets': 'building'}}]})
    assert errors({**base, 'steps': [{**DETECT, 'params': {}}]})
    assert errors({**base, 'steps': [{**DETECT, 'inputs': {}}]})
    assert errors({**base, 'steps': [{**ANALYZE, 'params': {'prompt': {'$env': 'X'}}}]})
    assert errors({**base, 'schema_version': '2.0', 'steps': [DETECT]})
    assert errors({**base, 'steps': []})


def test_result_files_follow_the_result_schema(tmp_path):
    jsonschema = pytest.importorskip('jsonschema')
    validator = jsonschema.Draft202012Validator(jobs.result_schema())
    workflow = FakeWorkflow()
    job = _job(tmp_path, [DETECT, ANALYZE])
    JobRunner(workflow=workflow).run(job, secrets={'api_key': 'k'})
    validator.validate(json.loads(job.result_path.read_text()))
    workflow.detect = lambda settings: (_ for _ in ()).throw(RuntimeError('x'))
    JobRunner(workflow=workflow).run(job, secrets={'api_key': 'k'})
    validator.validate(json.loads(job.result_path.read_text()))


def test_step_catalogue():
    catalogue = {entry['type']: entry for entry in jobs.describe_steps()}
    assert list(catalogue) == list(jobs.STEP_TYPES)
    analyze = {p['name']: p for p in catalogue['analyze.vlm']['params']}
    assert analyze['prompt']['required'] is True and 'default' not in analyze['prompt']
    assert analyze['api_key']['secret'] is True
    assert analyze['street_vertical_crop']['default'] == [0.0, 1.0]
    assert 'gemini' in analyze['backend']['choices']
    assert analyze['min_footprint_coverage']['maximum'] == 1
    assert catalogue['detect.aerial']['inputs'][0] == {
        'name': 'raster',
        'kind': 'raster',
        'required': True,
        'description': 'The georeferenced orthomosaic that defines the area.',
    }
    json.dumps(catalogue)  # plain JSON all the way down


def test_published_reference_files_are_current():
    """
    The checked-in schemas and step tables must match the code.

    Regenerate them with ``make job-reference``.
    """
    schemas = DOCS / '_static' / 'schemas'
    for name, build in [
        ('rapidtools-job-1.0.schema.json', jobs.job_schema),
        ('rapidtools-job-result-1.0.schema.json', jobs.result_schema),
        ('rapidtools-secrets-1.0.schema.json', jobs.secrets_schema),
    ]:
        assert json.loads((schemas / name).read_text()) == build(), name
    generated = (DOCS / '_generated' / 'job_steps.inc').read_text()
    assert generated == jobs.steps_rst()


# ---------------------------------------------------------------------- CLI
def test_cli_describes_the_interface(capsys):
    assert cli.main(['schema']) == 0
    assert json.loads(capsys.readouterr().out) == jobs.job_schema()
    assert cli.main(['schema', 'result']) == 0
    assert json.loads(capsys.readouterr().out) == jobs.result_schema()
    assert cli.main(['steps']) == 0
    assert json.loads(capsys.readouterr().out) == jobs.describe_steps()
    assert cli.main(['steps', '--format', 'rst']) == 0
    assert capsys.readouterr().out == jobs.steps_rst()


def test_cli_validate(tmp_path, capsys):
    job = _job(tmp_path, [DETECT, ANALYZE])
    assert cli.main(['validate', str(job.source)]) == 0
    assert 'valid (2 step(s))' in capsys.readouterr().out
    assert cli.main(['validate', str(job.source), '--require-secrets']) == 2
    assert "Invalid job: Step 'grade' (analyze.vlm)" in capsys.readouterr().err
    secrets = write_json_atomic(
        tmp_path / 's.json', Secrets.document({'api_key': 'k'}), private=True
    )
    assert (
        cli.main(
            [
                'validate',
                str(job.source),
                '--secrets',
                str(secrets),
                '--require-secrets',
            ]
        )
        == 0
    )
    assert cli.main(['validate', str(tmp_path / 'missing.json')]) == 2


def test_cli_run_exit_codes(tmp_path, monkeypatch, capsys):
    made = []

    class Recording(FakeWorkflow):
        def __init__(self, progress_callback=None):
            super().__init__(progress_callback)
            made.append(self)

    monkeypatch.setattr(workflow_module, 'AssetAnalysisWorkflow', Recording)
    monkeypatch.setattr('rapidtools.config.configure_logging', lambda level: None)
    job = _job(tmp_path, [DETECT])
    custom = tmp_path / 'r.json'
    assert cli.main(['run', str(job.source), '--result', str(custom)]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert '[done] Demo finished.' in out and f'Result file: {custom}' in out

    monkeypatch.setattr(
        Recording, 'detect', lambda self, s: (_ for _ in ()).throw(ValueError('bad'))
    )
    assert cli.main(['run', str(job.source)]) == cli.EXIT_FAILED
    assert '[error] bad' in capsys.readouterr().err

    monkeypatch.setattr(
        Recording,
        'detect',
        lambda self, s: (_ for _ in ()).throw(OperationCancelled('stop')),
    )
    assert cli.main(['run', str(job.source)]) == cli.EXIT_CANCELLED

    invalid = _job(tmp_path, [{**DETECT, 'params': {}}])
    assert cli.main(['run', str(invalid.source)]) == cli.EXIT_INVALID
    assert 'assets is required' in capsys.readouterr().err
    assert cli.main(['run', str(tmp_path / 'missing.json')]) == cli.EXIT_INVALID


@pytest.mark.skipif(
    sys.platform == 'win32',
    reason='os.kill(SIGTERM) terminates a Windows process instead of signalling it',
)
def test_cli_cancels_on_signal(tmp_path, monkeypatch, capsys):
    import signal

    class Interrupted(FakeWorkflow):
        def detect(self, settings):
            os.kill(os.getpid(), signal.SIGTERM)  # the scheduler stops the job
            if self.cancelled:
                raise OperationCancelled('stop')
            return super().detect(settings)

    monkeypatch.setattr(workflow_module, 'AssetAnalysisWorkflow', Interrupted)
    monkeypatch.setattr('rapidtools.config.configure_logging', lambda level: None)
    job = _job(tmp_path, [DETECT])
    before = signal.getsignal(signal.SIGTERM)
    assert cli.main(['run', str(job.source)]) == cli.EXIT_CANCELLED
    assert signal.getsignal(signal.SIGTERM) is before  # handlers are restored
    assert 'Cancelling' in capsys.readouterr().err
    assert json.loads(job.result_path.read_text())['status'] == 'cancelled'


def test_cli_runs_off_the_main_thread(tmp_path, monkeypatch):
    """No signal handler can be installed there; the job still runs."""
    monkeypatch.setattr('rapidtools.config.configure_logging', lambda level: None)
    job = _job(tmp_path, [DETECT])
    codes = []
    monkeypatch.setattr(workflow_module, 'AssetAnalysisWorkflow', FakeWorkflow)
    thread = threading.Thread(
        target=lambda: codes.append(cli.main(['run', str(job.source)]))
    )
    thread.start()
    thread.join()
    assert codes == [cli.EXIT_OK]


def test_small_corners(tmp_path, monkeypatch):
    # An assets reference whose producer wrote nothing is an empty collection:
    assert len(steps_module.AssetsInput().load()) == 0
    # A nullable parameter with choices also allows null in the schema:
    from rapidtools.jobs.reference import _param_schema

    schema = _param_schema(
        Param('mode', 'string', 'Mode.', default=None, nullable=True, choices=('a',))
    )
    assert schema['type'] == ['string', 'null'] and schema['enum'] == ['a', None]

    # A write that fails at the last moment leaves no temporary file behind:
    def refuse(src, dst):
        raise OSError('read-only file system')

    monkeypatch.setattr(os, 'replace', refuse)
    with pytest.raises(OSError, match='read-only'):
        write_json_atomic(tmp_path / 'x.json', {'a': 1})
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_waits_for_a_file_that_is_briefly_locked(tmp_path, monkeypatch):
    """Windows refuses to replace a file another process has open."""
    real_replace, attempts = os.replace, []

    def locked_twice(src, dst):
        attempts.append(dst)
        if len(attempts) < 3:
            raise PermissionError('[WinError 5] Access is denied')
        real_replace(src, dst)

    monkeypatch.setattr(os, 'replace', locked_twice)
    monkeypatch.setattr('rapidtools.jobs.schema.time.sleep', lambda seconds: None)
    target = write_json_atomic(tmp_path / 'r.json', {'status': 'running'})
    assert len(attempts) == 3 and json.loads(target.read_text()) == {
        'status': 'running'
    }

    def always_locked(src, dst):
        raise PermissionError('[WinError 5] Access is denied')

    monkeypatch.setattr(os, 'replace', always_locked)
    with pytest.raises(PermissionError):
        write_json_atomic(tmp_path / 'r.json', {'status': 'done'})
    # The previous version is intact and no temporary file is left:
    assert json.loads(target.read_text()) == {'status': 'running'}
    assert [p.name for p in tmp_path.iterdir()] == ['r.json']
