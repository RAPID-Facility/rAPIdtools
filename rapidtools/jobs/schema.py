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
Document model of the rapidtools JSON job interface.

A *job file* is a JSON document that tells rapidtools what to run: an ordered
list of *steps*, each naming a step type, the data it reads (``inputs``) and
its options (``params``). This module reads, validates and writes those
documents; it knows nothing about what a step does (see
:mod:`rapidtools.jobs.steps`) or how steps are executed (see
:mod:`rapidtools.jobs.runner`).

Three small conventions keep a job file self-contained and safe to share:

- **Paths** are absolute, or relative to the folder holding the job file.
- **Step references** ``{"step": "<id>", "output": "<name>"}`` let a step read
  what an earlier step of the same job produced.
- **Secret references** ``{"$secret": "<name>"}`` and ``{"$env": "<VAR>"}``
  keep API keys and tokens out of the file.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import secrets as _secrets
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: ``schema`` value of a job file.
JOB_SCHEMA = 'rapidtools.job'
#: ``schema`` value of a job result (status) file.
RESULT_SCHEMA = 'rapidtools.job_result'
#: ``schema`` value of a secrets file.
SECRETS_SCHEMA = 'rapidtools.secrets'
#: Version of the interface this rapidtools release reads and writes.
SCHEMA_VERSION = '1.0'
#: A secret named ``mapillary_token`` may be supplied through the
#: environment variable ``RAPIDTOOLS_SECRET_MAPILLARY_TOKEN``.
SECRET_ENV_PREFIX = 'RAPIDTOOLS_SECRET_'

_ID_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')
_JOB_KEYS = (
    'schema',
    'schema_version',
    'id',
    'name',
    'created_at',
    'created_by',
    'output_dir',
    'steps',
    'metadata',
)
_STEP_KEYS = ('id', 'type', 'inputs', 'params', 'metadata')
_SECRET_REF = ('$secret', '$env')
# How often an atomic write tries to move its file into place (see below).
_REPLACE_ATTEMPTS = 6


class JobError(ValueError):
    """
    A job document, or one of its steps, is not valid.

    ``str(error)`` is the plain message; :meth:`describe` prefixes it with
    the step it belongs to, which is what a log or a terminal should show.

    Example:
        >>> err = JobError('zoom must be an integer.', step_id='s1',
        ...                step_type='imagery.basemap')
        >>> str(err)
        'zoom must be an integer.'
        >>> err.describe()
        "Step 's1' (imagery.basemap): zoom must be an integer."
    """

    def __init__(
        self,
        message: str,
        step_id: str | None = None,
        step_type: str | None = None,
    ) -> None:
        """
        Initialize the error.

        Args:
            message (str): What is wrong, in a form the user can act on.
            step_id (str | None): The ``id`` of the offending step, if any.
            step_type (str | None): Its ``type``.
        """
        super().__init__(message)
        self.step_id = step_id
        self.step_type = step_type

    def describe(self) -> str:
        """Return the message prefixed with the step it refers to."""
        if self.step_id is None:
            return str(self)
        kind = f' ({self.step_type})' if self.step_type else ''
        return f'Step {self.step_id!r}{kind}: {self}'


def utc_now() -> str:
    """
    Return the current UTC time as an ISO 8601 string ending in ``Z``.

    Example:
        >>> utc_now().endswith('Z')
        True
    """
    now = _dt.datetime.now(_dt.UTC)
    return now.isoformat(timespec='seconds').replace('+00:00', 'Z')


def write_json_atomic(path: str | Path, document: Any, private: bool = False) -> Path:
    """
    Write ``document`` as JSON so readers never see a half-written file.

    The text goes to a temporary file in the same folder, which then replaces
    ``path`` in one step. Anything polling a result file therefore reads
    either the previous version or the new one.

    Args:
        path (str | Path): Destination; parent folders are created.
        document (Any): A JSON-serialisable value.
        private (bool): Create the file readable by its owner only (``0600``).
            Used for secrets files. Defaults to ``False``. Windows has no
            such mode bits; callers keep private files in a private folder.

    Returns:
        Path: ``path``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{_secrets.token_hex(4)}.tmp')
    text = json.dumps(document, indent=2, ensure_ascii=False) + '\n'
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(tmp, flags, 0o600 if private else 0o644)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(text)
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                # Windows refuses to replace a file that another process (a
                # client polling the result, a virus scanner) has open at
                # this instant. Give it a moment instead of failing the job.
                if attempt == _REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def read_json(path: str | Path, what: str = 'job file') -> dict[str, Any]:
    """
    Read a JSON object from ``path``.

    Args:
        path (str | Path): The file to read.
        what (str): What the file is, for error messages.

    Returns:
        dict[str, Any]: The decoded object.

    Raises:
        JobError: If the file is missing, is not JSON or is not an object.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding='utf-8')
    except OSError as exc:
        raise JobError(f'Could not read the {what} {path}: {exc}') from exc
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JobError(f'The {what} {path} is not valid JSON: {exc}') from exc
    if not isinstance(document, dict):
        raise JobError(f'The {what} {path} must hold a JSON object.')
    return document


def check_schema(document: Mapping[str, Any], expected: str) -> str:
    """
    Check the ``schema`` and ``schema_version`` of a document.

    A document is accepted when its major version equals the major version
    of :data:`SCHEMA_VERSION`. A newer *minor* version is accepted too: minor
    releases only add optional fields, and a field this release does not
    know is reported where it is used.

    Args:
        document (Mapping[str, Any]): The decoded document.
        expected (str): The required ``schema`` value.

    Returns:
        str: The document's ``schema_version``.

    Raises:
        JobError: If the schema name or version is missing or unsupported.

    Example:
        >>> check_schema({'schema': 'rapidtools.job', 'schema_version': '1.0'},
        ...              'rapidtools.job')
        '1.0'
    """
    schema = document.get('schema')
    if schema != expected:
        raise JobError(f'"schema" must be {expected!r}, got {schema!r}.')
    version = document.get('schema_version')
    match = re.fullmatch(r'(\d+)\.(\d+)', version) if isinstance(version, str) else None
    if match is None:
        raise JobError(
            f'"schema_version" must be a string such as {SCHEMA_VERSION!r}, '
            f'got {version!r}.'
        )
    supported_major = SCHEMA_VERSION.split('.')[0]
    if match.group(1) != supported_major:
        raise JobError(
            f'This rapidtools reads {expected} version {supported_major}.x; the '
            f'document declares version {version}.'
        )
    return str(version)


def _newer_hint(version: str) -> str:
    """Explain an unknown field in a document newer than this release."""
    if int(version.split('.')[1]) > int(SCHEMA_VERSION.split('.')[1]):
        return (
            f' The document declares schema_version {version}, which is newer '
            f'than the {SCHEMA_VERSION} this rapidtools understands; upgrade '
            'rapidtools to use it.'
        )
    return ''


def _reject_unknown(
    given: Mapping[str, Any], allowed: tuple[str, ...], where: str, hint: str = ''
) -> None:
    """Raise :class:`JobError` when ``given`` has keys outside ``allowed``."""
    unknown = sorted(str(k) for k in given if k not in allowed)
    if unknown:
        raise JobError(
            f'Unknown field(s) in {where}: {", ".join(unknown)}. '
            f'Allowed: {", ".join(allowed)}.{hint}'
        )


def is_step_ref(value: Any) -> bool:
    """
    Tell whether ``value`` is a reference to an earlier step's output.

    Example:
        >>> is_step_ref({'step': 'detect', 'output': 'assets'})
        True
        >>> is_step_ref('assets.geojson')
        False
    """
    return isinstance(value, dict) and set(value) == {'step', 'output'}


def is_secret_ref(value: Any) -> bool:
    """
    Tell whether ``value`` is a ``{"$secret": ...}`` or ``{"$env": ...}`` reference.

    Example:
        >>> is_secret_ref({'$secret': 'api_key'}), is_secret_ref('AIza...')
        (True, False)
    """
    return (
        isinstance(value, dict) and len(value) == 1 and next(iter(value)) in _SECRET_REF
    )


@dataclass
class StepSpec:
    """
    One step of a job, exactly as written in the job file.

    Attributes:
        id (str): Name of the step inside its job; later steps and the result
            file refer to it.
        type (str): The step type, e.g. ``'detect.aerial'``.
        inputs (dict[str, Any]): Data the step reads: paths, regions or
            references to earlier steps.
        params (dict[str, Any]): Options of the step type.
        metadata (dict[str, Any]): Free-form notes; never interpreted.

    Example:
        >>> step = StepSpec(id='detect', type='detect.aerial',
        ...                 inputs={'raster': 'scene.tiff'},
        ...                 params={'assets': ['building']})
        >>> step.to_dict()['type']
        'detect.aerial'
    """

    id: str
    type: str
    inputs: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls, raw: Any, index: int = 0, version: str = SCHEMA_VERSION
    ) -> StepSpec:
        """
        Build a step from its JSON object.

        Args:
            raw (Any): The decoded step object.
            index (int): Zero-based position in ``steps``; a step without an
                ``id`` is named ``s<index + 1>``.
            version (str): ``schema_version`` of the enclosing job.

        Returns:
            StepSpec: The step.

        Raises:
            JobError: If the object is malformed.
        """
        where = f'steps[{index}]'
        if not isinstance(raw, dict):
            raise JobError(f'{where} must be a JSON object.')
        _reject_unknown(raw, _STEP_KEYS, where, _newer_hint(version))
        step_id = raw.get('id', f's{index + 1}')
        if not isinstance(step_id, str) or not _ID_PATTERN.match(step_id):
            raise JobError(
                f'{where}.id must start with a letter or digit and contain only '
                f'letters, digits, ".", "_" and "-"; got {step_id!r}.'
            )
        step_type = raw.get('type')
        if not isinstance(step_type, str) or not step_type.strip():
            raise JobError(f'{where}.type is required (e.g. "detect.aerial").')
        parts: dict[str, dict[str, Any]] = {}
        for key in ('inputs', 'params', 'metadata'):
            value = raw.get(key, {})
            if not isinstance(value, dict):
                raise JobError(f'{where}.{key} must be a JSON object.')
            parts[key] = dict(value)
        return cls(id=step_id, type=step_type.strip(), **parts)

    def to_dict(self) -> dict[str, Any]:
        """Return the step as a JSON-serialisable object."""
        document: dict[str, Any] = {
            'id': self.id,
            'type': self.type,
            'inputs': dict(self.inputs),
            'params': dict(self.params),
        }
        if self.metadata:
            document['metadata'] = dict(self.metadata)
        return document


@dataclass
class Job:
    """
    A job: where to write, and the steps to run in order.

    Attributes:
        output_dir (str): Folder every step writes into, as given in the
            file. :attr:`output_path` is the resolved form.
        steps (list[StepSpec]): The steps, in execution order.
        id (str): Identifier of the job; generated when omitted.
        name (str): Human-readable name shown in logs and results.
        created_at (str): UTC creation time (ISO 8601).
        created_by (dict[str, Any]): Who wrote the file, e.g.
            ``{'client': 'rapidtools-gui', 'version': '0.2.1'}``.
        metadata (dict[str, Any]): Free-form notes; never interpreted.
        schema_version (str): Interface version the file declares.
        base_dir (Path): Folder relative paths are resolved against: the
            folder of the job file, or the working directory for a job built
            in memory.
        source (Path | None): The file the job was read from or saved to.

    Example:
        >>> job = Job(output_dir='out', steps=[StepSpec(
        ...     id='detect', type='detect.aerial',
        ...     inputs={'raster': 'scene.tiff'}, params={'assets': ['tree']})])
        >>> job.to_dict()['schema'], len(job.steps)
        ('rapidtools.job', 1)
    """

    output_dir: str
    steps: list[StepSpec]
    id: str = ''
    name: str = ''
    created_at: str = ''
    created_by: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    base_dir: Path = field(default_factory=Path.cwd)
    source: Path | None = None

    def __post_init__(self) -> None:
        """Fill the generated fields and check the step list."""
        self.output_dir = str(self.output_dir)
        self.base_dir = Path(self.base_dir)
        self.created_at = self.created_at or utc_now()
        self.id = self.id or new_job_id(self.name)
        self.name = self.name or self.id
        if not _ID_PATTERN.match(self.id):
            raise JobError(
                '"id" must start with a letter or digit and contain only letters, '
                f'digits, ".", "_" and "-"; got {self.id!r}.'
            )
        if not self.output_dir.strip():
            raise JobError('"output_dir" is required.')
        if not self.steps:
            raise JobError('"steps" must list at least one step.')
        seen: set[str] = set()
        for step in self.steps:
            if step.id in seen:
                raise JobError(f'Two steps share the id {step.id!r}.')
            for name, value in step.inputs.items():
                if is_step_ref(value) and value['step'] not in seen:
                    raise JobError(
                        f'Input {name!r} refers to step {value["step"]!r}, which '
                        'is not an earlier step of this job.',
                        step_id=step.id,
                        step_type=step.type,
                    )
            seen.add(step.id)

    # ------------------------------------------------------------- reading
    @classmethod
    def from_dict(
        cls,
        document: Mapping[str, Any],
        base_dir: str | Path | None = None,
        source: str | Path | None = None,
    ) -> Job:
        """
        Build a job from a decoded job document.

        Args:
            document (Mapping[str, Any]): The decoded JSON object.
            base_dir (str | Path | None): Folder relative paths are resolved
                against. Defaults to the working directory.
            source (str | Path | None): The file the document came from.

        Returns:
            Job: The validated job.

        Raises:
            JobError: If the document does not follow the job schema.
        """
        if not isinstance(document, Mapping):
            raise JobError('A job must be a JSON object.')
        version = check_schema(document, JOB_SCHEMA)
        _reject_unknown(document, _JOB_KEYS, 'the job', _newer_hint(version))
        steps = document.get('steps')
        if not isinstance(steps, list):
            raise JobError('"steps" must be a list of step objects.')
        output_dir = document.get('output_dir')
        if not isinstance(output_dir, str):
            raise JobError('"output_dir" is required and must be a string.')
        for key in ('id', 'name', 'created_at'):
            if not isinstance(document.get(key, ''), str):
                raise JobError(f'"{key}" must be a string.')
        for key in ('created_by', 'metadata'):
            if not isinstance(document.get(key, {}), dict):
                raise JobError(f'"{key}" must be a JSON object.')
        return cls(
            output_dir=output_dir,
            steps=[StepSpec.from_dict(s, i, version) for i, s in enumerate(steps)],
            id=document.get('id', ''),
            name=document.get('name', ''),
            created_at=document.get('created_at', ''),
            created_by=dict(document.get('created_by', {})),
            metadata=dict(document.get('metadata', {})),
            schema_version=version,
            base_dir=Path(base_dir) if base_dir is not None else Path.cwd(),
            source=Path(source) if source is not None else None,
        )

    @classmethod
    def load(cls, path: str | Path) -> Job:
        """
        Read a job file.

        Relative paths inside the file are resolved against its folder.

        Args:
            path (str | Path): The job file.

        Returns:
            Job: The validated job.

        Raises:
            JobError: If the file cannot be read or is not a valid job.
        """
        path = Path(path).expanduser()
        document = read_json(path)
        absolute = path.absolute()
        return cls.from_dict(document, base_dir=absolute.parent, source=absolute)

    # ------------------------------------------------------------- writing
    def to_dict(self) -> dict[str, Any]:
        """Return the job as a JSON-serialisable job document."""
        document: dict[str, Any] = {
            'schema': JOB_SCHEMA,
            'schema_version': self.schema_version,
            'id': self.id,
            'name': self.name,
            'created_at': self.created_at,
        }
        if self.created_by:
            document['created_by'] = dict(self.created_by)
        document['output_dir'] = self.output_dir
        document['steps'] = [step.to_dict() for step in self.steps]
        if self.metadata:
            document['metadata'] = dict(self.metadata)
        return document

    def save(self, path: str | Path) -> Path:
        """
        Write the job file and remember where it lives.

        Args:
            path (str | Path): Destination file.

        Returns:
            Path: The absolute path written.
        """
        target = Path(path).expanduser().absolute()
        write_json_atomic(target, self.to_dict())
        self.source = target
        self.base_dir = target.parent
        return target

    # --------------------------------------------------------------- paths
    def resolve_path(self, raw: str | Path) -> Path:
        """
        Turn a path from the job file into an absolute path.

        ``~`` is expanded and a relative path is taken from
        :attr:`base_dir`. Symbolic links are left alone.

        Example:
            >>> job = Job(output_dir='out', base_dir='/data/run7', steps=[
            ...     StepSpec(id='s1', type='detect.aerial')])
            >>> str(job.resolve_path('scene.tiff'))
            '/data/run7/scene.tiff'
        """
        path = Path(raw).expanduser()
        return path if path.is_absolute() else self.base_dir / path

    @property
    def output_path(self) -> Path:
        """The resolved output folder."""
        return self.resolve_path(self.output_dir)

    @property
    def result_path(self) -> Path | None:
        """
        Where the result file of this job is written by default.

        ``<job file stem>.result.json`` next to the job file, or ``None`` for
        a job that was never read from or saved to a file.
        """
        if self.source is None:
            return None
        return self.source.with_name(f'{self.source.stem}.result.json')


def new_job_id(name: str = '') -> str:
    """
    Generate a job identifier that sorts by creation time.

    Example:
        >>> job_id = new_job_id('Detection + inference')
        >>> job_id.endswith('-detection-inference'), len(job_id.split('-')[0])
        (True, 16)
    """
    stamp = _dt.datetime.now(_dt.UTC).strftime('%Y%m%dT%H%M%SZ')
    slug = re.sub(r'[^0-9A-Za-z]+', '-', name.strip().lower()).strip('-')
    parts = [stamp, _secrets.token_hex(2)] + ([slug[:40].strip('-')] if slug else [])
    return '-'.join(parts)


def new_job(
    steps: Sequence[StepSpec | Mapping[str, Any]],
    output_dir: str | Path,
    name: str = '',
    created_by: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    base_dir: str | Path | None = None,
) -> Job:
    """
    Build a job in memory, ready to be saved or run.

    Args:
        steps (Sequence[StepSpec | Mapping[str, Any]]): The steps, as
            :class:`StepSpec` objects or step dictionaries.
        output_dir (str | Path): Folder every step writes into.
        name (str): Human-readable name. Defaults to the generated ID.
        created_by (Mapping[str, Any] | None): Who is writing the job.
        metadata (Mapping[str, Any] | None): Free-form notes.
        base_dir (str | Path | None): Folder relative paths are resolved
            against until the job is saved. Defaults to the working directory.

    Returns:
        Job: The validated job.

    Example:
        >>> job = new_job(
        ...     [{'type': 'detect.aerial', 'inputs': {'raster': 'scene.tiff'},
        ...       'params': {'assets': ['building']}}],
        ...     output_dir='out', name='Detection')
        >>> job.steps[0].id, job.name
        ('s1', 'Detection')
    """
    specs = [
        s if isinstance(s, StepSpec) else StepSpec.from_dict(s, i)
        for i, s in enumerate(steps)
    ]
    return Job(
        output_dir=str(output_dir),
        steps=specs,
        name=name,
        created_by=dict(created_by or {}),
        metadata=dict(metadata or {}),
        base_dir=Path(base_dir) if base_dir is not None else Path.cwd(),
    )


def load_job(job: Job | Mapping[str, Any] | str | Path) -> Job:
    """
    Accept a job in any of its forms and return a :class:`Job`.

    Args:
        job (Job | Mapping[str, Any] | str | Path): A job object, a decoded
            job document, or the path of a job file.

    Returns:
        Job: The validated job.

    Raises:
        JobError: If the job cannot be read or is invalid.
    """
    if isinstance(job, Job):
        return job
    if isinstance(job, Mapping):
        return Job.from_dict(job)
    return Job.load(job)


class Secrets:
    """
    Resolves the secret references of a job.

    A parameter such as an API key may be written in a job file as a literal
    string, but the file then has to be protected like the key itself. The
    two reference forms keep the value elsewhere:

    - ``{"$secret": "<name>"}`` is looked up in the secrets given to the
      runner (a mapping or a *secrets file*), then in the environment
      variable ``RAPIDTOOLS_SECRET_<NAME>``.
    - ``{"$env": "<VAR>"}`` reads the environment variable ``VAR``.

    Args:
        values (Mapping[str, str] | None): Secret values by name.
        environ (Mapping[str, str] | None): Environment to read. Defaults to
            ``os.environ``.

    Example:
        >>> secrets = Secrets({'api_key': 'k-123'}, environ={'MLY': 'MLY|9'})
        >>> secrets.resolve({'$secret': 'api_key'}), secrets.resolve({'$env': 'MLY'})
        ('k-123', 'MLY|9')
        >>> secrets.resolve('literal')
        'literal'
    """

    def __init__(
        self,
        values: Mapping[str, str] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        """Store the secret values and the environment to fall back on."""
        self._values = {str(k): str(v) for k, v in (values or {}).items()}
        self._environ = environ

    def __repr__(self) -> str:
        """Show the names only; values never appear in logs."""
        return f'Secrets({sorted(self._values)})'

    @classmethod
    def load(cls, path: str | Path) -> Secrets:
        """
        Read a secrets file.

        The file is a JSON object ``{"schema": "rapidtools.secrets",
        "schema_version": "1.0", "secrets": {"<name>": "<value>", ...}}``.

        Raises:
            JobError: If the file cannot be read or is malformed.
        """
        document = read_json(Path(path).expanduser(), what='secrets file')
        check_schema(document, SECRETS_SCHEMA)
        values = document.get('secrets')
        if not isinstance(values, dict) or not all(
            isinstance(v, str) for v in values.values()
        ):
            raise JobError('"secrets" must map secret names to string values.')
        return cls(values)

    @classmethod
    def coerce(
        cls, secrets: Secrets | Mapping[str, str] | str | Path | None
    ) -> Secrets:
        """Accept a :class:`Secrets`, a mapping, a secrets-file path or ``None``."""
        if isinstance(secrets, Secrets):
            return secrets
        if secrets is None or isinstance(secrets, Mapping):
            return cls(secrets)
        return cls.load(secrets)

    @staticmethod
    def document(values: Mapping[str, str]) -> dict[str, Any]:
        """Return the secrets-file document holding ``values``."""
        return {
            'schema': SECRETS_SCHEMA,
            'schema_version': SCHEMA_VERSION,
            'secrets': dict(values),
        }

    def resolve(self, value: Any, missing: str | None = None) -> Any:
        """
        Return the value a parameter stands for.

        Args:
            value (Any): A literal, or a secret reference.
            missing (str | None): Returned for a reference that cannot be
                resolved. ``None`` (default) raises instead.

        Returns:
            Any: The secret for a reference; ``value`` itself otherwise.

        Raises:
            JobError: If a reference is malformed, or cannot be resolved and
                ``missing`` is ``None``.
        """
        if not is_secret_ref(value):
            return value
        kind, name = next(iter(value.items()))
        if not isinstance(name, str) or not name.strip():
            raise JobError(f'A {kind} reference needs a non-empty name.')
        environ = os.environ if self._environ is None else self._environ
        if kind == '$env':
            found = environ.get(name)
            where = f'the environment variable {name} is not set'
        else:
            env_name = SECRET_ENV_PREFIX + re.sub(r'[^0-9A-Za-z]', '_', name).upper()
            found = self._values.get(name, environ.get(env_name))
            where = (
                f'secret {name!r} was not supplied; add it to the secrets file or '
                f'set {env_name}'
            )
        if found is None:
            if missing is not None:
                return missing
            raise JobError(f'A secret is missing: {where}.')
        return found
