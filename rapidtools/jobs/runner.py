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
Runs rapidtools job files and reports what happened.

:class:`JobRunner` executes the steps of a job in order and keeps a *result
file* next to the job file up to date while it works. The result file is the
only thing a client has to read: it says whether the job is running, done,
failed or cancelled, which step it is on, and where every output was
written.

Example:
    >>> from rapidtools.jobs import run_job
    >>> result = run_job('jobs/detect.json')  # doctest: +SKIP
    >>> result.status, result.step('detect').outputs['assets']  # doctest: +SKIP
    ('done', PosixPath('/data/out/assets_final.geojson'))
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rapidtools.core import OperationCancelled

from .schema import (
    RESULT_SCHEMA,
    SCHEMA_VERSION,
    Job,
    JobError,
    Secrets,
    load_job,
    utc_now,
    write_json_atomic,
)
from .steps import STEP_TYPES, StepContext, prepare_step

logger = logging.getLogger(__name__)

#: Every status a job can report.
JOB_STATUSES = ('running', 'done', 'error', 'cancelled')
#: Every status a step can report.
STEP_STATUSES = ('pending', 'running', 'done', 'error', 'cancelled', 'skipped')


def _json_safe(value: Any) -> Any:
    """Return ``value`` reduced to what a JSON file can hold."""
    return json.loads(json.dumps(value, default=str))


@dataclass
class StepResult:
    """
    The outcome of one step.

    Attributes:
        id (str): The step's ``id``.
        type (str): Its step type.
        status (str): One of :data:`STEP_STATUSES`.
        started_at (str | None): UTC start time.
        finished_at (str | None): UTC end time.
        duration_s (float | None): Wall-clock seconds.
        outputs (dict[str, Path]): Declared outputs that were written, by
            name. An output the step did not produce is absent.
        files (list[dict[str, Any]]): Other files the step wrote, each with
            ``label``, ``path`` and ``kind``.
        summary (dict[str, Any]): Small facts about the run, e.g.
            ``{'n_assets': 179}``.
        data (Any): A JSON result that is not a file.
        error (dict[str, str] | None): ``{'type', 'message'}`` of the
            failure, when ``status`` is ``'error'`` or ``'cancelled'``.
        objects (dict[str, Any]): In-memory results for callers in the same
            process, e.g. ``objects['assets']`` is the live collection. Not
            written to the result file.
        settings (Any): The settings object the step ran with. Not written
            to the result file.
    """

    id: str
    type: str
    status: str = 'pending'
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    outputs: dict[str, Path] = field(default_factory=dict)
    files: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    data: Any = None
    error: dict[str, str] | None = None
    objects: dict[str, Any] = field(default_factory=dict, repr=False)
    settings: Any = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return the step outcome as it appears in the result file."""
        step_type = STEP_TYPES.get(self.type)
        kinds = {o.name: o.kind for o in step_type.outputs} if step_type else {}
        document: dict[str, Any] = {
            'id': self.id,
            'type': self.type,
            'status': self.status,
            'started_at': self.started_at,
            'finished_at': self.finished_at,
            'duration_s': self.duration_s,
            'outputs': {
                name: {'path': str(path), 'kind': kinds.get(name, 'file')}
                for name, path in self.outputs.items()
            },
            'files': [{**f, 'path': str(f['path'])} for f in self.files],
            'summary': _json_safe(self.summary),
        }
        if self.data is not None:
            document['data'] = _json_safe(self.data)
        if self.error is not None:
            document['error'] = dict(self.error)
        return document


@dataclass
class JobResult:
    """
    The outcome of a job, mirrored in its result file.

    Attributes:
        job_id (str): ``id`` of the job.
        job_name (str): Its name.
        job_file (Path | None): The job file, if the job came from one.
        result_file (Path | None): Where this result is written.
        status (str): One of :data:`JOB_STATUSES`.
        message (str): The latest progress line, or the final outcome.
        started_at (str): UTC start time.
        updated_at (str): UTC time of the last change.
        finished_at (str | None): UTC end time, once the job has ended.
        duration_s (float | None): Wall-clock seconds, once the job has ended.
        steps (list[StepResult]): One entry per step, in order.
        exception (BaseException | None): The exception that ended the job,
            for callers in the same process. Not written to the result file.

    Example:
        >>> result = JobResult(job_id='j1', job_name='Demo')
        >>> result.ok, result.to_dict()['schema']
        (False, 'rapidtools.job_result')
    """

    job_id: str
    job_name: str = ''
    job_file: Path | None = None
    result_file: Path | None = None
    status: str = 'running'
    message: str = ''
    started_at: str = field(default_factory=utc_now)
    updated_at: str = ''
    finished_at: str | None = None
    duration_s: float | None = None
    steps: list[StepResult] = field(default_factory=list)
    exception: BaseException | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        """Whether the job finished and every step succeeded."""
        return self.status == 'done'

    def step(self, step_id: str) -> StepResult:
        """
        Return the result of the step with the given ``id``.

        Raises:
            KeyError: If the job has no such step.
        """
        for step in self.steps:
            if step.id == step_id:
                return step
        raise KeyError(step_id)

    def raise_for_status(self) -> JobResult:
        """
        Re-raise the exception that ended the job, if it did not succeed.

        Returns:
            JobResult: ``self`` when the job is done, for chaining.

        Raises:
            BaseException: The original exception of the failed or cancelled
                step (:class:`~rapidtools.core.OperationCancelled` for a
                cancellation, :class:`~rapidtools.jobs.JobError` for an
                invalid job).
        """
        if self.exception is not None:
            raise self.exception
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return the result-file document."""
        from rapidtools import __version__

        return {
            'schema': RESULT_SCHEMA,
            'schema_version': SCHEMA_VERSION,
            'job_id': self.job_id,
            'job_name': self.job_name,
            'job_file': str(self.job_file) if self.job_file else None,
            'status': self.status,
            'message': self.message,
            'started_at': self.started_at,
            'updated_at': self.updated_at or self.started_at,
            'finished_at': self.finished_at,
            'duration_s': self.duration_s,
            'rapidtools_version': __version__,
            'steps': [step.to_dict() for step in self.steps],
        }


class JobRunner:
    """
    Executes jobs with a workflow engine.

    Args:
        workflow (Any): The engine that does the work, an
            :class:`~rapidtools.gui.workflow.AssetAnalysisWorkflow`. A
            long-lived client such as the GUI passes its own, so that
            cancellation and cached models are shared between jobs. When
            omitted, a workflow is created on first use and its progress
            messages are copied into the result file.
        progress (Callable[[str], None] | None): Receives each progress
            message of a workflow created by the runner.

    Example:
        >>> runner = JobRunner()
        >>> runner.validate({  # doctest: +SKIP
        ...     'schema': 'rapidtools.job', 'schema_version': '1.0',
        ...     'output_dir': 'out',
        ...     'steps': [{'type': 'detect.aerial',
        ...                'inputs': {'raster': 'scene.tiff'},
        ...                'params': {'assets': ['building']}}],
        ... })
    """

    def __init__(
        self,
        workflow: Any = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        """Store the workflow (or the callback for one created on demand)."""
        self._workflow = workflow
        self._progress = progress
        self._active: tuple[JobResult, Path | None] | None = None
        self._cancel_requested = False

    # ------------------------------------------------------------ workflow
    @property
    def workflow(self) -> Any:
        """The workflow engine, created on first use when none was given."""
        if self._workflow is None:
            from rapidtools.gui.workflow import AssetAnalysisWorkflow

            self._workflow = AssetAnalysisWorkflow(progress_callback=self._on_progress)
        return self._workflow

    def _on_progress(self, message: str) -> None:
        """Record a progress line in the result file and forward it."""
        if self._active is not None:
            result, path = self._active
            result.message = message
            self._write(result, path)
        if self._progress is not None:
            self._progress(message)

    def cancel(self) -> None:
        """
        Ask the job to stop.

        The running step stops at its next safe point (the current batch or
        tile finishes first) and no further step is started.
        """
        self._cancel_requested = True
        if self._workflow is not None:
            self._workflow.cancel()

    # ---------------------------------------------------------- validation
    def validate(
        self,
        job: Job | Mapping[str, Any] | str | Path,
        secrets: Secrets | Mapping[str, str] | str | Path | None = None,
        allow_missing_secrets: bool = False,
    ) -> Job:
        """
        Check every step of a job without running anything.

        Args:
            job (Job | Mapping[str, Any] | str | Path): The job, a decoded
                job document, or the path of a job file.
            secrets (Secrets | Mapping[str, str] | str | Path | None): Secret
                values, or the path of a secrets file.
            allow_missing_secrets (bool): Accept secret references that
                cannot be resolved here (the machine that runs the job will
                have them). Defaults to ``False``.

        Returns:
            Job: The validated job.

        Raises:
            JobError: If the job or any of its steps is invalid.
        """
        job = load_job(job)
        resolver = Secrets.coerce(secrets)
        for spec in job.steps:
            prepare_step(job, spec, resolver, None, allow_missing_secrets)
        return job

    # ------------------------------------------------------------- running
    def run(
        self,
        job: Job | Mapping[str, Any] | str | Path,
        secrets: Secrets | Mapping[str, str] | str | Path | None = None,
        result_path: str | Path | None = None,
        on_step: Callable[[StepResult], None] | None = None,
    ) -> JobResult:
        """
        Run a job and return its result.

        Every step is validated before the first one starts. Steps then run
        in order; the first failure or cancellation ends the job and marks
        the remaining steps ``skipped``. A failing step does not raise: the
        outcome is in the returned result (and its file), and
        :meth:`JobResult.raise_for_status` re-raises the original exception
        for callers that prefer one.

        Args:
            job (Job | Mapping[str, Any] | str | Path): The job, a decoded
                job document, or the path of a job file.
            secrets (Secrets | Mapping[str, str] | str | Path | None): Secret
                values, or the path of a secrets file.
            result_path (str | Path | None): Where to write the result file.
                Defaults to ``<job file stem>.result.json`` next to the job
                file; a job that is not on disk gets no result file.
            on_step (Callable[[StepResult], None] | None): Called after each
                step that ran (whatever its status), before the next one
                starts. An exception raised by it fails the job.

        Returns:
            JobResult: The outcome.

        Raises:
            JobError: Only if the job itself cannot be read (missing file,
                not JSON, not a job document).
        """
        job = load_job(job)
        path = Path(result_path).expanduser() if result_path else job.result_path
        result = JobResult(
            job_id=job.id,
            job_name=job.name,
            job_file=job.source,
            result_file=path,
            steps=[StepResult(id=s.id, type=s.type) for s in job.steps],
        )
        started = time.monotonic()
        self._active = (result, path)
        self._cancel_requested = False
        try:
            self._run(job, Secrets.coerce(secrets), result, path, on_step)
        except JobError as exc:  # the job did not pass validation
            result.status, result.exception = 'error', exc
            result.message = exc.describe()
            for step in result.steps:
                if step.id == exc.step_id:
                    step.status = 'error'
                    step.error = {'type': 'JobError', 'message': str(exc)}
        finally:
            self._active = None
            for step in result.steps:
                if step.status == 'pending':
                    step.status = 'skipped'
            result.finished_at = utc_now()
            result.duration_s = round(time.monotonic() - started, 3)
            self._write(result, path)
        return result

    def _run(
        self,
        job: Job,
        secrets: Secrets,
        result: JobResult,
        path: Path | None,
        on_step: Callable[[StepResult], None] | None,
    ) -> None:
        """Validate, then execute the steps, updating ``result`` in place."""
        result.message = f'{job.name} started.'
        self._write(result, path)
        for spec in job.steps:  # fail on any step before the first one runs
            prepare_step(job, spec, secrets)

        def lookup(step_id: str, output: str) -> tuple[Path | None, Any]:
            done = result.step(step_id)
            return done.outputs.get(output), done.objects.get(output)

        total = len(job.steps)
        for number, (spec, step) in enumerate(
            zip(job.steps, result.steps, strict=True), 1
        ):
            step.status, step.started_at = 'running', utc_now()
            result.message = f'Step {number}/{total} ({spec.type}) started.'
            logger.debug('Job %s: step %s (%s) started.', job.id, spec.id, spec.type)
            self._write(result, path)
            began = time.monotonic()
            context: StepContext | None = None
            try:
                if self._cancel_requested:  # cancelled between two steps
                    raise OperationCancelled('The job was cancelled.')
                context = prepare_step(job, spec, secrets, lookup)
                context.workflow = self.workflow
                step.settings = context.settings
                context.step_type.run(context)
                step.status = 'done'
            except OperationCancelled as exc:
                step.status, result.status, result.exception = (
                    'cancelled',
                    'cancelled',
                    exc,
                )
                result.message = f'{job.name} cancelled.'
            except Exception as exc:  # noqa: BLE001 - reported in the result
                step.status, result.status, result.exception = 'error', 'error', exc
                result.message = (
                    exc.describe() if isinstance(exc, JobError) else str(exc)
                )
            if result.exception is not None:
                step.error = {
                    'type': type(result.exception).__name__,
                    'message': str(result.exception),
                }
            if context is not None:
                step.outputs = {
                    k: Path(v) for k, v in context.outputs.items() if v is not None
                }
                step.files = [dict(f) for f in context.files if f.get('path')]
                step.summary = dict(context.summary)
                step.data = context.data
                step.objects = dict(context.objects)
            step.finished_at = utc_now()
            step.duration_s = round(time.monotonic() - began, 3)
            self._write(result, path)
            if on_step is not None:
                try:
                    on_step(step)
                except Exception as exc:  # noqa: BLE001 - the client's hook failed
                    if result.exception is None:
                        result.status, result.exception = 'error', exc
                        result.message = str(exc)
            if result.exception is not None:
                return
        result.status = 'done'
        result.message = f'{job.name} finished.'

    @staticmethod
    def _write(result: JobResult, path: Path | None) -> None:
        """Refresh the result file, if the job has one."""
        result.updated_at = utc_now()
        if path is not None:
            write_json_atomic(path, result.to_dict())


def run_job(
    job: Job | Mapping[str, Any] | str | Path,
    secrets: Secrets | Mapping[str, str] | str | Path | None = None,
    result_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
    workflow: Any = None,
) -> JobResult:
    """
    Run a job file (or an in-memory job) and return its result.

    This is the one-call form of :class:`JobRunner`.

    Args:
        job (Job | Mapping[str, Any] | str | Path): The job, a decoded job
            document, or the path of a job file.
        secrets (Secrets | Mapping[str, str] | str | Path | None): Secret
            values, or the path of a secrets file.
        result_path (str | Path | None): Where to write the result file.
            Defaults to ``<job file stem>.result.json`` next to the job file.
        progress (Callable[[str], None] | None): Receives progress messages.
        workflow (Any): A workflow engine to reuse. Defaults to a new one.

    Returns:
        JobResult: The outcome; ``result.ok`` tells whether it succeeded.

    Raises:
        JobError: If the job cannot be read at all.

    Example:
        >>> from rapidtools.jobs import run_job
        >>> result = run_job('jobs/detect.json', progress=print)  # doctest: +SKIP
        >>> result.status  # doctest: +SKIP
        'done'
    """
    return JobRunner(workflow=workflow, progress=progress).run(
        job, secrets=secrets, result_path=result_path
    )
