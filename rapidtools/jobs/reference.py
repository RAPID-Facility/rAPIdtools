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
# 11-24-2026

"""
Machine- and human-readable descriptions of the JSON job interface.

Everything here is generated from the step-type registry of
:mod:`rapidtools.jobs.steps`: the `JSON Schema <https://json-schema.org>`_
documents a partner system can validate job and result files against
(:func:`job_schema`, :func:`result_schema`, :func:`secrets_schema`), a JSON
catalogue of the step types (:func:`describe_steps`) and the reference
tables of the documentation (:func:`steps_rst`).
"""

from __future__ import annotations

import json
from typing import Any

from .runner import JOB_STATUSES, STEP_STATUSES
from .schema import JOB_SCHEMA, RESULT_SCHEMA, SCHEMA_VERSION, SECRETS_SCHEMA
from .steps import PARAM_KINDS, STEP_TYPES, Input, Param, StepType

_DRAFT = 'https://json-schema.org/draft/2020-12/schema'
_MAJOR = SCHEMA_VERSION.split('.')[0]
_VERSION_PATTERN = rf'^{_MAJOR}\.[0-9]+$'
_IDENTIFIER = {
    'type': 'string',
    'pattern': '^[A-Za-z0-9][A-Za-z0-9_.-]*$',
}
_KIND_WORDS = {
    'string': 'string',
    'number': 'number',
    'integer': 'integer',
    'boolean': 'boolean',
    'string_list': 'list of strings',
    'number_pair': 'list of two numbers',
    'number_or_string': 'number or string',
    'object': 'object',
}


def _jsonable(value: Any) -> Any:
    """Return a default value in its JSON form (tuples become lists)."""
    return json.loads(json.dumps(value))


def _param_schema(param: Param) -> dict[str, Any]:
    """Return the JSON Schema of one parameter."""
    schema: dict[str, Any]
    if param.kind == 'string_list':
        schema = {
            'type': 'array',
            'items': {'type': 'string', 'minLength': 1},
            'minItems': 1,
        }
    elif param.kind == 'number_pair':
        schema = {
            'type': 'array',
            'items': {'type': 'number'},
            'minItems': 2,
            'maxItems': 2,
        }
    else:
        schema = {'type': PARAM_KINDS[param.kind]}
    if param.minimum is not None:
        schema['minimum'] = param.minimum
    if param.maximum is not None:
        schema['maximum'] = param.maximum
    if param.exclusive_minimum is not None:
        schema['exclusiveMinimum'] = param.exclusive_minimum
    options = param.options()
    if options is not None:
        schema['enum'] = list(options)
    if param.nullable:
        kinds = schema['type'] if isinstance(schema['type'], list) else [schema['type']]
        schema['type'] = [*kinds, 'null']
        if 'enum' in schema:
            schema['enum'].append(None)
    if param.secret:
        schema = {'anyOf': [schema, {'$ref': '#/$defs/secretRef'}]}
    schema['description'] = param.doc
    if not param.required:
        schema['default'] = _jsonable(param.default)
    return schema


def _input_schema(item: Input) -> dict[str, Any]:
    """Return the JSON Schema of one input."""
    if item.kind == 'region':
        return {'$ref': '#/$defs/region', 'description': item.doc}
    return {
        'anyOf': [{'type': 'string', 'minLength': 1}, {'$ref': '#/$defs/stepRef'}],
        'description': item.doc,
    }


def _step_schema(step_type: StepType) -> dict[str, Any]:
    """Return the JSON Schema of a step of the given type."""
    inputs: dict[str, Any] = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {i.name: _input_schema(i) for i in step_type.inputs},
    }
    params: dict[str, Any] = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {p.name: _param_schema(p) for p in step_type.params},
    }
    required = ['type']
    required_inputs = [i.name for i in step_type.inputs if i.required]
    if required_inputs:
        inputs['required'] = required_inputs
        required.append('inputs')
    required_params = [p.name for p in step_type.params if p.required]
    if required_params:
        params['required'] = required_params
        required.append('params')
    return {
        'title': step_type.title,
        'description': step_type.doc,
        'type': 'object',
        'required': required,
        'additionalProperties': False,
        'properties': {
            'id': {'$ref': '#/$defs/identifier'},
            'type': {'const': step_type.name},
            'inputs': inputs,
            'params': params,
            'metadata': {'type': 'object'},
        },
    }


def job_schema() -> dict[str, Any]:
    """
    Return the JSON Schema (draft 2020-12) of a job file.

    The schema is generated from the registered step types, so it always
    describes exactly what this rapidtools release accepts.

    Example:
        >>> schema = job_schema()
        >>> schema['properties']['schema']['const']
        'rapidtools.job'
        >>> 'step.detect.aerial' in schema['$defs']
        True
    """
    names = list(STEP_TYPES)
    return {
        '$schema': _DRAFT,
        '$id': f'urn:rapidtools:schema:job:{SCHEMA_VERSION}',
        'title': 'rapidtools job file',
        'description': (
            'An ordered list of steps for rapidtools to run, with the folder '
            'they write into.'
        ),
        'type': 'object',
        'required': ['schema', 'schema_version', 'output_dir', 'steps'],
        'additionalProperties': False,
        'properties': {
            'schema': {'const': JOB_SCHEMA},
            'schema_version': {
                'type': 'string',
                'pattern': _VERSION_PATTERN,
                'description': f'Interface version; currently "{SCHEMA_VERSION}".',
            },
            'id': {
                '$ref': '#/$defs/identifier',
                'description': 'Identifier of the job. Generated when omitted.',
            },
            'name': {'type': 'string', 'description': 'Human-readable name.'},
            'created_at': {
                'type': 'string',
                'description': 'UTC creation time, ISO 8601.',
            },
            'created_by': {
                'type': 'object',
                'description': 'The client that wrote the file.',
            },
            'output_dir': {
                'type': 'string',
                'minLength': 1,
                'description': (
                    'Folder every step writes into; absolute, or relative to '
                    'the folder of the job file.'
                ),
            },
            'steps': {
                'type': 'array',
                'minItems': 1,
                'description': 'The steps, run in order.',
                'items': {
                    'type': 'object',
                    'required': ['type'],
                    'properties': {'type': {'enum': names}},
                    'allOf': [
                        {
                            'if': {
                                'properties': {'type': {'const': name}},
                                'required': ['type'],
                            },
                            'then': {'$ref': f'#/$defs/step.{name}'},
                        }
                        for name in names
                    ],
                },
            },
            'metadata': {
                'type': 'object',
                'description': 'Free-form notes; never interpreted.',
            },
        },
        '$defs': {
            'identifier': dict(_IDENTIFIER),
            'stepRef': {
                'description': "A reference to an earlier step's output.",
                'type': 'object',
                'required': ['step', 'output'],
                'additionalProperties': False,
                'properties': {
                    'step': {'$ref': '#/$defs/identifier'},
                    'output': {'type': 'string', 'minLength': 1},
                },
            },
            'secretRef': {
                'description': 'A credential kept outside the job file.',
                'oneOf': [
                    {
                        'type': 'object',
                        'required': ['$secret'],
                        'additionalProperties': False,
                        'properties': {'$secret': {'type': 'string', 'minLength': 1}},
                    },
                    {
                        'type': 'object',
                        'required': ['$env'],
                        'additionalProperties': False,
                        'properties': {'$env': {'type': 'string', 'minLength': 1}},
                    },
                ],
            },
            'region': {
                'description': 'A WGS84 bounding box, or the extent of a GeoJSON file.',
                'oneOf': [
                    {
                        'type': 'object',
                        'required': ['bbox'],
                        'additionalProperties': False,
                        'properties': {
                            'bbox': {
                                'description': '[min_lon, min_lat, max_lon, max_lat]',
                                'type': 'array',
                                'items': {'type': 'number'},
                                'minItems': 4,
                                'maxItems': 4,
                            }
                        },
                    },
                    {
                        'type': 'object',
                        'required': ['geojson'],
                        'additionalProperties': False,
                        'properties': {'geojson': {'type': 'string', 'minLength': 1}},
                    },
                ],
            },
            **{f'step.{name}': _step_schema(STEP_TYPES[name]) for name in names},
        },
    }


def result_schema() -> dict[str, Any]:
    """
    Return the JSON Schema (draft 2020-12) of a job result file.

    Example:
        >>> result_schema()['properties']['status']['enum']
        ['running', 'done', 'error', 'cancelled']
    """
    nullable_time = {'type': ['string', 'null']}
    output = {
        'type': 'object',
        'required': ['path', 'kind'],
        'properties': {
            'path': {'type': 'string', 'description': 'Absolute path.'},
            'kind': {'enum': ['raster', 'geojson', 'file']},
        },
    }
    step = {
        'type': 'object',
        'required': ['id', 'type', 'status', 'outputs', 'files', 'summary'],
        'properties': {
            'id': dict(_IDENTIFIER),
            'type': {'type': 'string'},
            'status': {'enum': list(STEP_STATUSES)},
            'started_at': nullable_time,
            'finished_at': nullable_time,
            'duration_s': {'type': ['number', 'null']},
            'outputs': {
                'type': 'object',
                'description': 'Declared outputs that were written, by name.',
                'additionalProperties': output,
            },
            'files': {
                'type': 'array',
                'description': 'Other files the step wrote.',
                'items': {
                    'type': 'object',
                    'required': ['label', 'path', 'kind'],
                    'properties': {
                        'label': {'type': 'string'},
                        'path': {'type': 'string'},
                        'kind': {'enum': ['raster', 'geojson', 'file']},
                    },
                },
            },
            'summary': {
                'type': 'object',
                'description': 'Small facts about the run, e.g. asset counts.',
            },
            'data': {'description': 'A JSON result that is not a file.'},
            'error': {
                'type': 'object',
                'required': ['type', 'message'],
                'properties': {
                    'type': {'type': 'string'},
                    'message': {'type': 'string'},
                },
            },
        },
    }
    return {
        '$schema': _DRAFT,
        '$id': f'urn:rapidtools:schema:job_result:{SCHEMA_VERSION}',
        'title': 'rapidtools job result file',
        'description': (
            'Status and outputs of a job. Rewritten atomically while the job '
            'runs, so it can be polled.'
        ),
        'type': 'object',
        'required': [
            'schema',
            'schema_version',
            'job_id',
            'status',
            'message',
            'started_at',
            'updated_at',
            'steps',
        ],
        'properties': {
            'schema': {'const': RESULT_SCHEMA},
            'schema_version': {'type': 'string', 'pattern': _VERSION_PATTERN},
            'job_id': dict(_IDENTIFIER),
            'job_name': {'type': 'string'},
            'job_file': {'type': ['string', 'null']},
            'status': {'enum': list(JOB_STATUSES)},
            'message': {
                'type': 'string',
                'description': 'Latest progress line, or the final outcome.',
            },
            'started_at': {'type': 'string'},
            'updated_at': {'type': 'string'},
            'finished_at': nullable_time,
            'duration_s': {'type': ['number', 'null']},
            'rapidtools_version': {'type': 'string'},
            'steps': {'type': 'array', 'items': step},
        },
    }


def secrets_schema() -> dict[str, Any]:
    """
    Return the JSON Schema (draft 2020-12) of a secrets file.

    Example:
        >>> secrets_schema()['properties']['schema']['const']
        'rapidtools.secrets'
    """
    return {
        '$schema': _DRAFT,
        '$id': f'urn:rapidtools:schema:secrets:{SCHEMA_VERSION}',
        'title': 'rapidtools secrets file',
        'description': 'Credentials referenced by a job file as {"$secret": name}.',
        'type': 'object',
        'required': ['schema', 'schema_version', 'secrets'],
        'additionalProperties': False,
        'properties': {
            'schema': {'const': SECRETS_SCHEMA},
            'schema_version': {'type': 'string', 'pattern': _VERSION_PATTERN},
            'secrets': {
                'type': 'object',
                'additionalProperties': {'type': 'string'},
            },
        },
    }


def describe_steps() -> list[dict[str, Any]]:
    """
    Return the catalogue of step types as plain JSON data.

    A client can build its forms from this instead of hard-coding the
    options of each step.

    Example:
        >>> [step['type'] for step in describe_steps()][:3]
        ['imagery.basemap', 'data.download', 'detect.aerial']
    """
    catalogue = []
    for step_type in STEP_TYPES.values():
        params = []
        for param in step_type.params:
            entry: dict[str, Any] = {
                'name': param.name,
                'kind': param.kind,
                'required': param.required,
                'secret': param.secret,
                'nullable': param.nullable,
                'description': param.doc,
            }
            if not param.required:
                entry['default'] = _jsonable(param.default)
            options = param.options()
            if options is not None:
                entry['choices'] = list(options)
            for bound in ('minimum', 'maximum', 'exclusive_minimum'):
                if getattr(param, bound) is not None:
                    entry[bound] = getattr(param, bound)
            params.append(entry)
        catalogue.append(
            {
                'type': step_type.name,
                'title': step_type.title,
                'description': step_type.doc,
                'since': step_type.since,
                'inputs': [
                    {
                        'name': i.name,
                        'kind': i.kind,
                        'required': i.required,
                        'description': i.doc,
                    }
                    for i in step_type.inputs
                ],
                'params': params,
                'outputs': [
                    {'name': o.name, 'kind': o.kind, 'description': o.doc}
                    for o in step_type.outputs
                ],
            }
        )
    return catalogue


def _table(headers: list[str], rows: list[list[str]], widths: list[int]) -> list[str]:
    """Render a reStructuredText ``list-table``."""
    lines = [
        '.. list-table::',
        '   :header-rows: 1',
        f'   :widths: {" ".join(str(w) for w in widths)}',
        '',
    ]
    for row in [headers, *rows]:
        for i, cell in enumerate(row):
            lines.append(f'   {"* -" if i == 0 else "  -"} {cell}')
    lines.append('')
    return lines


def _param_row(param: Param) -> list[str]:
    """Return the documentation row of one parameter."""
    kind = _KIND_WORDS[param.kind]
    if param.nullable:
        kind += ' or null'
    if param.secret:
        kind += ' (secret)'
    default = '**required**' if param.required else f'``{json.dumps(param.default)}``'
    doc = param.doc
    options = param.options()
    if options is not None:
        doc += ' One of: ' + ', '.join(f'``{json.dumps(o)}``' for o in options) + '.'
    limits = []
    if param.exclusive_minimum is not None:
        limits.append(f'greater than {param.exclusive_minimum:g}')
    if param.minimum is not None:
        limits.append(f'at least {param.minimum:g}')
    if param.maximum is not None:
        limits.append(f'at most {param.maximum:g}')
    if limits:
        doc += f' Must be {" and ".join(limits)}.'
    return [f'``{param.name}``', kind, default, doc]


def steps_rst() -> str:
    """
    Return the step-type reference of the documentation.

    The text is included by ``docs/source/json_interface.rst``; a test keeps
    the checked-in copy identical to this output. Regenerate it with
    ``make job-reference``.

    Example:
        >>> steps_rst().splitlines()[3]
        'imagery.basemap'
    """
    lines = [
        '.. This file is generated by `make job-reference`.',
        '.. Edit rapidtools/jobs/steps.py, not this file.',
        '',
    ]
    for step_type in STEP_TYPES.values():
        lines += [step_type.name, '~' * len(step_type.name), '']
        lines += [f'**{step_type.title}.** {step_type.doc}', '']
        if step_type.inputs:
            lines += ['*Inputs*', '']
            lines += _table(
                ['Name', 'Kind', 'Required', 'Description'],
                [
                    [f'``{i.name}``', i.kind, 'yes' if i.required else 'no', i.doc]
                    for i in step_type.inputs
                ],
                [18, 10, 10, 62],
            )
        else:
            lines += ['*Inputs*: none.', '']
        lines += ['*Parameters*', '']
        lines += _table(
            ['Name', 'Type', 'Default', 'Description'],
            [_param_row(p) for p in step_type.params],
            [22, 14, 14, 50],
        )
        if step_type.outputs:
            lines += ['*Outputs*', '']
            lines += _table(
                ['Name', 'Kind', 'Description'],
                [[f'``{o.name}``', o.kind, o.doc] for o in step_type.outputs],
                [22, 12, 66],
            )
        else:
            lines += ['*Outputs*: none (see ``data`` in the result file).', '']
    return '\n'.join(lines).rstrip() + '\n'
