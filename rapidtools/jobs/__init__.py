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
The JSON job interface of rapidtools.

A client (the graphical interface, a scheduler, a partner system) describes
the work in a *job file*, rapidtools runs it and reports in a *result file*.
Nothing else crosses the boundary, so the client and rapidtools can live in
different processes or on different machines.

The pieces:

- :class:`Job` and :class:`StepSpec` read, validate and write job files
  (:func:`new_job`, :func:`load_job`).
- :class:`JobRunner` and :func:`run_job` execute them and keep the result
  file (:class:`JobResult`, :class:`StepResult`) up to date.
- :class:`Secrets` resolves the ``{"$secret": ...}`` and ``{"$env": ...}``
  references that keep credentials out of job files.
- :class:`StepType`, :class:`Param`, :class:`Input`, :class:`Output` and
  :func:`register_step` define the step types; :func:`job_schema`,
  :func:`result_schema`, :func:`secrets_schema` and :func:`describe_steps`
  publish them.

From a terminal the same is available as ``rapidtools-job`` (or
``python -m rapidtools.jobs``).

Example:
    >>> from rapidtools.jobs import new_job, run_job
    >>>
    >>> job = new_job(
    ...     name='Detect and grade buildings',
    ...     output_dir='output',
    ...     steps=[
    ...         {'id': 'detect', 'type': 'detect.aerial',
    ...          'inputs': {'raster': 'eaton_patch_20250214.tiff'},
    ...          'params': {'assets': ['building']}},
    ...         {'id': 'grade', 'type': 'analyze.vlm',
    ...          'inputs': {'raster': 'eaton_patch_20250214.tiff',
    ...                     'assets': {'step': 'detect', 'output': 'assets'}},
    ...          'params': {'prompt': 'Describe the damage to the outlined asset.',
    ...                     'backend': 'gemini',
    ...                     'api_key': {'$env': 'GEMINI_API_KEY'}}},
    ...     ],
    ... )
    >>> job.save('jobs/eaton.json')  # doctest: +SKIP
    >>> result = run_job('jobs/eaton.json')  # doctest: +SKIP
    >>> result.status  # doctest: +SKIP
    'done'
"""

from .reference import (
    describe_steps,
    job_schema,
    result_schema,
    secrets_schema,
    steps_rst,
)
from .runner import (
    JOB_STATUSES,
    STEP_STATUSES,
    JobResult,
    JobRunner,
    StepResult,
    run_job,
)
from .schema import (
    JOB_SCHEMA,
    RESULT_SCHEMA,
    SCHEMA_VERSION,
    SECRETS_SCHEMA,
    Job,
    JobError,
    Secrets,
    StepSpec,
    load_job,
    new_job,
)
from .steps import (
    REQUIRED,
    STEP_TYPES,
    Input,
    Output,
    Param,
    StepContext,
    StepType,
    get_step_type,
    register_step,
)

__all__ = [
    'JOB_SCHEMA',
    'JOB_STATUSES',
    'REQUIRED',
    'RESULT_SCHEMA',
    'SCHEMA_VERSION',
    'SECRETS_SCHEMA',
    'STEP_STATUSES',
    'STEP_TYPES',
    'Input',
    'Job',
    'JobError',
    'JobResult',
    'JobRunner',
    'Output',
    'Param',
    'Secrets',
    'StepContext',
    'StepResult',
    'StepSpec',
    'StepType',
    'describe_steps',
    'get_step_type',
    'job_schema',
    'load_job',
    'new_job',
    'register_step',
    'result_schema',
    'run_job',
    'secrets_schema',
    'steps_rst',
]
