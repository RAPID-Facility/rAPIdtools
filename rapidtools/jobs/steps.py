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
The step types a rapidtools job can contain.

Every step type is one declarative :class:`StepType` record: its name, the
data it reads (:class:`Input`), its options (:class:`Param`), what it writes
(:class:`Output`) and the two functions that turn a validated step into work.
The same records drive validation, the published JSON Schema and the
reference tables of the documentation, so the three cannot drift apart.

Adding a capability to the interface means registering one more record with
:func:`register_step`; nothing else in the runner or in a client changes.

The built-in step types wrap
:class:`~rapidtools.gui.workflow.AssetAnalysisWorkflow`, the engine the
graphical interface has always used.
"""

from __future__ import annotations

import copy
import difflib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rapidtools.core import OperationCancelled

from .schema import Job, JobError, Secrets, StepSpec, is_secret_ref, is_step_ref


class _Required:
    """Marks a parameter that has no default."""

    def __repr__(self) -> str:
        return 'REQUIRED'


#: Default of a parameter that must be given.
REQUIRED: Any = _Required()

#: Parameter kinds and the JSON types they accept.
PARAM_KINDS: dict[str, Any] = {
    'string': 'string',
    'number': 'number',
    'integer': 'integer',
    'boolean': 'boolean',
    'string_list': 'array',
    'number_pair': 'array',
    'number_or_string': ['number', 'string'],
    'object': 'object',
}
#: Input kinds: what a step reads.
INPUT_KINDS = ('raster', 'assets', 'region', 'file')


def _is_number(value: Any) -> bool:
    """Return whether ``value`` is a JSON number (booleans are not)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class Param:
    """
    One option of a step type.

    Attributes:
        name (str): Key under ``params`` in the job file.
        kind (str): A :data:`PARAM_KINDS` key.
        doc (str): One or two sentences for the reference documentation.
        default (Any): Value used when the key is omitted; :data:`REQUIRED`
            when it must be given.
        choices (tuple | Callable[[], tuple] | None): Allowed values. A
            callable is evaluated on first use so that listing them does not
            import heavy modules.
        secret (bool): The value is a credential: it may be written as a
            secret reference and never appears in logs or result files.
        nullable (bool): ``null`` is accepted.
        minimum (float | None): Smallest allowed value.
        maximum (float | None): Largest allowed value.
        exclusive_minimum (float | None): The value must be greater than this.
        field (str | None): Name of the matching settings field when it
            differs from ``name``.

    Example:
        >>> zoom = Param('zoom', 'integer', 'Tile zoom level.', default=19,
        ...              minimum=10, maximum=21)
        >>> zoom.coerce(18.0), zoom.required
        (18, False)
    """

    name: str
    kind: str
    doc: str
    default: Any = REQUIRED
    choices: tuple[Any, ...] | Callable[[], tuple[Any, ...]] | None = None
    secret: bool = False
    nullable: bool = False
    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: float | None = None
    field: str | None = None

    def __post_init__(self) -> None:
        """Reject a kind the validator does not know."""
        if self.kind not in PARAM_KINDS:
            raise ValueError(f'Unknown parameter kind {self.kind!r} for {self.name}.')

    @property
    def required(self) -> bool:
        """Whether the parameter must be given."""
        return self.default is REQUIRED

    def options(self) -> tuple[Any, ...] | None:
        """Return the allowed values, or ``None`` when any value is allowed."""
        if self.choices is None:
            return None
        return tuple(self.choices() if callable(self.choices) else self.choices)

    def coerce(self, value: Any) -> Any:
        """
        Check a value from a job file and return it in its Python form.

        JSON types are enforced strictly (``"19"`` is not an integer), so a
        mistake is reported instead of being guessed at.

        Args:
            value (Any): The decoded JSON value.

        Returns:
            Any: ``value``; integers written as ``19.0`` become ``19``,
            numbers become ``float`` and a number pair becomes a tuple.

        Raises:
            ValueError: If the type, the range or the choice is wrong.
        """
        name = self.name
        if value is None:
            if self.nullable:
                return None
            raise ValueError(f'{name} must not be null.')
        if self.kind == 'string':
            if not isinstance(value, str):
                raise ValueError(f'{name} must be a string.')
        elif self.kind == 'boolean':
            if not isinstance(value, bool):
                raise ValueError(f'{name} must be true or false.')
        elif self.kind == 'integer':
            if not _is_number(value) or float(value) != int(value):
                raise ValueError(f'{name} must be an integer.')
            value = int(value)
        elif self.kind == 'number':
            if not _is_number(value):
                raise ValueError(f'{name} must be a number.')
            value = float(value)
        elif self.kind == 'number_or_string':
            if not (_is_number(value) or isinstance(value, str)):
                raise ValueError(f'{name} must be a number or a string.')
        elif self.kind == 'string_list':
            if (
                not isinstance(value, list)
                or not value
                or not all(isinstance(v, str) and v.strip() for v in value)
            ):
                raise ValueError(f'{name} must be a non-empty list of strings.')
            value = list(value)
        elif self.kind == 'number_pair':
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 2
                or not all(_is_number(v) for v in value)
            ):
                raise ValueError(f'{name} must be a list of two numbers.')
            value = (float(value[0]), float(value[1]))
        elif not isinstance(value, dict):
            raise ValueError(f'{name} must be a JSON object.')
        if self.kind in ('integer', 'number'):
            if self.exclusive_minimum is not None and value <= self.exclusive_minimum:
                raise ValueError(
                    f'{name} must be greater than {self.exclusive_minimum:g}.'
                )
            if self.minimum is not None and value < self.minimum:
                raise ValueError(f'{name} must be at least {self.minimum:g}.')
            if self.maximum is not None and value > self.maximum:
                raise ValueError(f'{name} must be at most {self.maximum:g}.')
        options = self.options()
        if options is not None and value not in options:
            raise ValueError(f'{name} must be one of {options}, got {value!r}.')
        return value


@dataclass(frozen=True)
class Input:
    """
    One piece of data a step type reads.

    Attributes:
        name (str): Key under ``inputs`` in the job file.
        kind (str): ``'raster'``, ``'assets'`` or ``'file'`` (a path, or a
            reference to an earlier step's output) or ``'region'`` (a
            bounding box or a GeoJSON file whose extent is used).
        doc (str): Description for the reference documentation.
        required (bool): The input must be given.
    """

    name: str
    kind: str
    doc: str
    required: bool = False

    def __post_init__(self) -> None:
        """Reject a kind the resolver does not know."""
        if self.kind not in INPUT_KINDS:
            raise ValueError(f'Unknown input kind {self.kind!r} for {self.name}.')


@dataclass(frozen=True)
class Output:
    """
    One file a step type writes that later steps may read.

    Attributes:
        name (str): Key under ``outputs`` in the result file, and the
            ``output`` of a step reference.
        kind (str): ``'raster'``, ``'geojson'`` or ``'file'``.
        doc (str): Description for the reference documentation.
    """

    name: str
    kind: str
    doc: str


@dataclass
class Region:
    """
    A resolved ``region`` input: a WGS84 box or a GeoJSON file.

    Example:
        >>> Region(bbox=(-117.49, 47.71, -117.48, 47.72)).bounding_box().bounds
        (-117.49, 47.71, -117.48, 47.72)
    """

    bbox: tuple[float, float, float, float] | None = None
    geojson: Path | None = None

    def bounding_box(self) -> Any:
        """Return the region as a :class:`~rapidtools.core.BoundingBox`."""
        from rapidtools.core import BoundingBox

        if self.bbox is not None:
            min_lon, min_lat, max_lon, max_lat = self.bbox
            return BoundingBox(
                min_x=min_lon, min_y=min_lat, max_x=max_lon, max_y=max_lat
            )
        if self.geojson is None or not self.geojson.is_file():
            raise FileNotFoundError(f'Region file not found: {self.geojson}')
        return BoundingBox.from_geojson(self.geojson)


@dataclass
class AssetsInput:
    """
    A resolved ``assets`` input: a GeoJSON file, or an earlier step's result.

    Attributes:
        path (Path | None): The GeoJSON file holding the assets.
        collection (Any): The collection an earlier step of the same job
            produced, handed over without a round trip through the file.
        empty_hint (str): What to say when the producing step found nothing.
    """

    path: Path | None = None
    collection: Any = None
    empty_hint: str = ''

    def load(self) -> Any:
        """
        Return the assets as a :class:`~rapidtools.core.PhysicalAssetCollection`.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        from rapidtools.core import PhysicalAssetCollection

        if self.collection is not None:
            return self.collection
        if self.path is None:
            return PhysicalAssetCollection()
        if not self.path.is_file():
            raise FileNotFoundError(f'Asset file not found: {self.path}')
        return PhysicalAssetCollection.from_geojson(self.path)


@dataclass
class StepContext:
    """
    Everything a step needs to run, and the place it reports back to.

    The run function of a step type fills :attr:`outputs`, :attr:`files`,
    :attr:`summary`, :attr:`data` and :attr:`objects` as it goes, so that
    whatever was produced before a failure or a cancellation is still
    reported.

    Attributes:
        job (Job): The job being run.
        spec (StepSpec): The step as written in the job file.
        step_type (StepType): Its type.
        params (dict[str, Any]): Validated parameters, secrets resolved.
        inputs (dict[str, Any]): Resolved inputs by name (:class:`~pathlib.Path`,
            :class:`AssetsInput` or :class:`Region`); absent inputs are ``None``.
        settings (Any): What :attr:`StepType.build` returned.
        output_dir (Path): The job's resolved output folder.
        workflow (Any): The workflow engine doing the work.
        outputs (dict[str, Path | None]): Declared outputs written so far.
        files (list[dict[str, Any]]): Other files worth listing, each with
            ``label``, ``path`` and ``kind``.
        summary (dict[str, Any]): Small JSON facts about the run (counts).
        data (Any): A JSON result that is not a file.
        objects (dict[str, Any]): In-memory results for callers in the same
            process (collections, workflow result objects). Never serialised.
    """

    job: Job
    spec: StepSpec
    step_type: StepType
    params: dict[str, Any]
    inputs: dict[str, Any]
    settings: Any
    output_dir: Path
    workflow: Any = None
    outputs: dict[str, Path | None] = field(default_factory=dict)
    files: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    data: Any = None
    objects: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepType:
    """
    A kind of step a job may contain.

    Attributes:
        name (str): The ``type`` written in job files, ``<stage>.<variant>``.
        title (str): Short human-readable title.
        doc (str): What the step does, for the reference documentation.
        build (Callable): ``build(step_type, params, inputs, output_dir,
            dry)`` returns the settings object the step runs with, raising
            ``ValueError`` for an invalid combination. ``dry`` is ``True``
            while a job is only being validated, when references to earlier
            steps are placeholders and no file may be read.
        run (Callable[[StepContext], None]): Does the work.
        inputs (tuple[Input, ...]): Data the step reads.
        params (tuple[Param, ...]): Its options.
        outputs (tuple[Output, ...]): Files later steps may reference.
        activity (str): The work as a noun (``'inference'``), used in
            messages such as "skipping inference".
        empty_hint (str): What to report when the step finds nothing and a
            later step needed its result.
        since (str): Interface version that introduced the step type.

    Example:
        >>> detect = get_step_type('detect.aerial')
        >>> detect.title, [p.name for p in detect.params][:2]
        ('Detect assets in aerial imagery', ['assets', 'asset_size_m'])
    """

    name: str
    title: str
    doc: str
    build: Callable[..., Any]
    run: Callable[[StepContext], None]
    inputs: tuple[Input, ...] = ()
    params: tuple[Param, ...] = ()
    outputs: tuple[Output, ...] = ()
    activity: str = ''
    empty_hint: str = ''
    since: str = '1.0'

    def read_params(
        self,
        given: dict[str, Any],
        secrets: Secrets | None = None,
        allow_missing_secrets: bool = False,
    ) -> dict[str, Any]:
        """
        Validate the ``params`` of a step and fill in the defaults.

        Args:
            given (dict[str, Any]): ``params`` from the job file.
            secrets (Secrets | None): Resolver for secret references.
            allow_missing_secrets (bool): Replace a secret that cannot be
                resolved with a placeholder instead of failing, so a job can
                be checked on a machine that does not hold the credentials.

        Returns:
            dict[str, Any]: A value for every parameter of the step type.

        Raises:
            ValueError: For an unknown, missing or invalid parameter.
        """
        known = {p.name: p for p in self.params}
        for key in given:
            if key not in known:
                close = difflib.get_close_matches(key, known, n=1)
                hint = f' Did you mean {close[0]!r}?' if close else ''
                raise ValueError(
                    f'Unknown parameter {key!r}.{hint} {self.name} accepts: '
                    f'{", ".join(known) or "no parameters"}.'
                )
        secrets = secrets or Secrets()
        values: dict[str, Any] = {}
        for param in self.params:
            if param.name not in given:
                if param.required:
                    raise ValueError(f'{param.name} is required.')
                values[param.name] = copy.deepcopy(param.default)
                continue
            value = given[param.name]
            if is_secret_ref(value):
                if not param.secret:
                    raise ValueError(
                        f'{param.name} is not a credential and cannot be a '
                        'secret reference.'
                    )
                value = secrets.resolve(
                    value, missing='<secret>' if allow_missing_secrets else None
                )
            values[param.name] = param.coerce(value)
        return values

    def settings_kwargs(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return ``params`` keyed by the settings field each one fills."""
        return {(p.field or p.name): params[p.name] for p in self.params}


# ================================================================ registry
STEP_TYPES: dict[str, StepType] = {}


def register_step(step_type: StepType, replace: bool = False) -> StepType:
    """
    Make a step type available to job files.

    Args:
        step_type (StepType): The step type.
        replace (bool): Allow replacing a registered type of the same name.

    Returns:
        StepType: ``step_type``, so the call can be used inline.

    Raises:
        ValueError: If the name is taken and ``replace`` is ``False``.
    """
    if step_type.name in STEP_TYPES and not replace:
        raise ValueError(f'Step type {step_type.name!r} is already registered.')
    STEP_TYPES[step_type.name] = step_type
    return step_type


def get_step_type(name: str) -> StepType:
    """
    Look up a step type by the name used in job files.

    Raises:
        JobError: If no such step type exists; the message suggests the
            closest name.

    Example:
        >>> get_step_type('analyze.vlm').activity
        'inference'
    """
    step_type = STEP_TYPES.get(name)
    if step_type is None:
        close = difflib.get_close_matches(name, STEP_TYPES, n=1)
        hint = f' Did you mean {close[0]!r}?' if close else ''
        raise JobError(
            f'Unknown step type {name!r}.{hint} Available: {", ".join(STEP_TYPES)}.'
        )
    return step_type


# A lookup returns (path, in-memory object) of an earlier step's output.
OutputLookup = Callable[[str, str], tuple[Path | None, Any]]


def resolve_inputs(
    step_type: StepType,
    spec: StepSpec,
    job: Job,
    lookup: OutputLookup | None = None,
) -> dict[str, Any]:
    """
    Turn the ``inputs`` of a step into paths, regions and asset sources.

    Args:
        step_type (StepType): The type of the step.
        spec (StepSpec): The step.
        job (Job): The job, for path resolution and reference checks.
        lookup (OutputLookup | None): Returns the path and in-memory object
            of an earlier step's output. ``None`` while validating, when
            references are replaced by placeholders.

    Returns:
        dict[str, Any]: One entry per input of the step type; ``None`` for
        inputs that were not given.

    Raises:
        ValueError: For an unknown, missing or malformed input.
    """
    declared = {i.name: i for i in step_type.inputs}
    for key in spec.inputs:
        if key not in declared:
            raise ValueError(
                f'Unknown input {key!r}. {step_type.name} reads: '
                f'{", ".join(declared) or "no inputs"}.'
            )
    earlier = {s.id: s for s in job.steps}
    resolved: dict[str, Any] = {}
    for item in step_type.inputs:
        value = spec.inputs.get(item.name)
        if value is None:
            if item.required:
                raise ValueError(f'Input {item.name!r} is required.')
            resolved[item.name] = None
        elif item.kind == 'region':
            resolved[item.name] = _resolve_region(item.name, value, job)
        elif is_step_ref(value):
            producer = get_step_type(earlier[value['step']].type)
            if value['output'] not in {o.name for o in producer.outputs}:
                raise ValueError(
                    f'Input {item.name!r} refers to output {value["output"]!r}, '
                    f'which {producer.name} does not produce (it produces: '
                    f'{", ".join(o.name for o in producer.outputs) or "nothing"}).'
                )
            resolved[item.name] = _resolve_reference(item, value, producer, lookup)
        elif isinstance(value, str) and value.strip():
            path = job.resolve_path(value)
            resolved[item.name] = AssetsInput(path) if item.kind == 'assets' else path
        else:
            raise ValueError(
                f'Input {item.name!r} must be a path or a reference '
                '{"step": ..., "output": ...}.'
            )
    return resolved


def _resolve_reference(
    item: Input, ref: dict[str, str], producer: StepType, lookup: OutputLookup | None
) -> Any:
    """Resolve a reference to an earlier step's output."""
    label = f'{ref["step"]}.{ref["output"]}'
    if lookup is None:  # validating: the earlier step has not run yet
        placeholder = Path(f'<{label}>')
        return AssetsInput(placeholder) if item.kind == 'assets' else placeholder
    path, live = lookup(ref['step'], ref['output'])
    if item.kind == 'assets':
        return AssetsInput(path, live, producer.empty_hint)
    if path is None:
        raise ValueError(f'Step {ref["step"]!r} produced no {ref["output"]!r} output.')
    return path


def _resolve_region(name: str, value: Any, job: Job) -> Region:
    """Read a ``{"bbox": [...]}`` or ``{"geojson": "..."}`` region."""
    usage = (
        f'Input {name!r} must be {{"bbox": [min_lon, min_lat, max_lon, max_lat]}} '
        'or {"geojson": "<path>"}.'
    )
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError(usage)
    if 'geojson' in value:
        if not isinstance(value['geojson'], str) or not value['geojson'].strip():
            raise ValueError(usage)
        return Region(geojson=job.resolve_path(value['geojson']))
    bbox = value.get('bbox')
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or not all(_is_number(v) for v in bbox)
    ):
        raise ValueError(usage)
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox)
    if not (-180 <= min_lon < max_lon <= 180 and -85 <= min_lat < max_lat <= 85):
        raise ValueError(
            'The bounding box must satisfy min < max with longitudes in '
            '[-180, 180] and latitudes in [-85, 85].'
        )
    return Region(bbox=(min_lon, min_lat, max_lon, max_lat))


def prepare_step(
    job: Job,
    spec: StepSpec,
    secrets: Secrets | None = None,
    lookup: OutputLookup | None = None,
    allow_missing_secrets: bool = False,
) -> StepContext:
    """
    Validate one step and build the context it will run with.

    Called twice per step: once for every step before anything runs
    (``lookup=None``), so that a mistake in the last step is reported before
    the first one starts, and once more right before the step runs, when the
    outputs of the earlier steps exist.

    Args:
        job (Job): The job.
        spec (StepSpec): The step to prepare.
        secrets (Secrets | None): Resolver for secret references.
        lookup (OutputLookup | None): Access to earlier outputs, or ``None``
            to validate only.
        allow_missing_secrets (bool): See :meth:`StepType.read_params`.

    Returns:
        StepContext: The context, without a workflow attached.

    Raises:
        JobError: If the step is invalid. The error names the step.
    """
    try:
        step_type = get_step_type(spec.type)
        params = step_type.read_params(spec.params, secrets, allow_missing_secrets)
        inputs = resolve_inputs(step_type, spec, job, lookup)
        output_dir = job.output_path
        settings = step_type.build(
            step_type, params, inputs, output_dir, lookup is None
        )
    except JobError as exc:
        if exc.step_id is None:
            exc.step_id, exc.step_type = spec.id, spec.type
        raise
    except (TypeError, ValueError) as exc:
        raise JobError(str(exc), step_id=spec.id, step_type=spec.type) from exc
    return StepContext(
        job=job,
        spec=spec,
        step_type=step_type,
        params=params,
        inputs=inputs,
        settings=settings,
        output_dir=output_dir,
    )


# ========================================================= built-in steps
def _backends() -> tuple[str, ...]:
    from rapidtools.gui.workflow import MODEL_BACKENDS

    return tuple(MODEL_BACKENDS)


def _imagery_sources() -> tuple[str, ...]:
    from rapidtools.gui.workflow import IMAGERY_SOURCES

    return tuple(IMAGERY_SOURCES)


def _outline_shapes() -> tuple[str, ...]:
    from rapidtools.processing.outlines import OUTLINE_SHAPES

    return tuple(OUTLINE_SHAPES)


def _assist_actions() -> tuple[str, ...]:
    from rapidtools.gui.prompt_builder import ASSIST_ACTIONS

    return tuple(ASSIST_ACTIONS)


# ---------------------------------------------------------- imagery.basemap
def _build_basemap(
    step_type: StepType, params: dict[str, Any], inputs: dict[str, Any], out: Path, dry
) -> Any:
    from rapidtools.gui.workflow import RegionImagerySettings

    region: Region | None = inputs['region']
    where: dict[str, Any] = {}
    if region is not None and region.bbox is not None:
        keys = ('min_lon', 'min_lat', 'max_lon', 'max_lat')
        where = dict(zip(keys, region.bbox, strict=True))
    elif region is not None:
        where = {'geojson_path': region.geojson}
    return RegionImagerySettings(
        output_dir=out, **step_type.settings_kwargs(params), **where
    )


def _run_basemap(ctx: StepContext) -> None:
    path = Path(ctx.workflow.download_basemap(ctx.settings))
    ctx.outputs['raster'] = path
    ctx.summary.update(provider=ctx.settings.provider, zoom=ctx.settings.zoom)


register_step(
    StepType(
        name='imagery.basemap',
        title='Download a satellite basemap',
        doc=(
            'Stitches Bing or Google satellite tiles over a region into one '
            'georeferenced GeoTIFF. An existing file for the same provider, '
            'zoom and bounds is reused instead of downloaded again.'
        ),
        build=_build_basemap,
        run=_run_basemap,
        activity='imagery download',
        inputs=(
            Input(
                'region',
                'region',
                'Area to cover: a WGS84 box, or a GeoJSON file whose extent is used.',
            ),
        ),
        params=(
            Param(
                'provider',
                'string',
                'Tile provider.',
                default='bing',
                choices=('bing', 'google'),
            ),
            Param(
                'zoom',
                'integer',
                'Web Mercator zoom level (Bing up to 20, Google up to 21). One '
                'level up doubles the resolution and quadruples the tile count.',
                default=19,
                minimum=10,
                maximum=21,
            ),
            Param(
                'max_tiles',
                'integer',
                'Refuse the download when the region needs more tiles than this.',
                default=2500,
                minimum=1,
            ),
            Param(
                'max_workers',
                'integer',
                'Concurrent tile downloads.',
                default=10,
                minimum=1,
            ),
        ),
        outputs=(Output('raster', 'raster', 'The stitched GeoTIFF.'),),
    )
)


# ------------------------------------------------------------ data.download
def _build_download(
    step_type: StepType, params: dict[str, Any], inputs: dict[str, Any], out: Path, dry
) -> Any:
    from rapidtools.datasets import DATASET_REGISTRY

    name = params['dataset'].strip().lower()
    if name not in DATASET_REGISTRY:
        close = difflib.get_close_matches(name, DATASET_REGISTRY, n=1, cutoff=0.5)
        hint = f' Did you mean {close[0]!r}?' if close else ''
        raise ValueError(f'Unknown dataset {params["dataset"]!r}.{hint}')
    return {'dataset': name}


def _run_download(ctx: StepContext) -> None:
    # Looked up at call time so tests (and users) can substitute the function.
    import rapidtools

    name = ctx.settings['dataset']
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        Path(p) for p in rapidtools.download_dataset(name, output_dir=ctx.output_dir)
    ]
    if not paths:
        raise RuntimeError(f'No file of the dataset {name!r} could be downloaded.')
    ctx.outputs['file'] = paths[0]
    ctx.files.extend({'label': p.name, 'path': p, 'kind': 'file'} for p in paths[1:])
    ctx.summary.update(dataset=name, n_files=len(paths))


register_step(
    StepType(
        name='data.download',
        title='Download a sample dataset',
        doc=(
            'Fetches a dataset of the rapidtools registry (sample '
            'orthomosaics, footprints, prompts) into the output folder. Files '
            'that are already there are not downloaded again.'
        ),
        build=_build_download,
        run=_run_download,
        activity='download',
        params=(
            Param(
                'dataset',
                'string',
                'Registry name of the dataset, e.g. "eaton_patch1" '
                '(see rapidtools.list_datasets()).',
            ),
        ),
        outputs=(
            Output(
                'file',
                'file',
                'The first file of the dataset; for the sample orthomosaics, '
                'the raster. Further files are listed under "files".',
            ),
        ),
    )
)


# ------------------------------------------------------------ detect.aerial
def _build_detect(
    step_type: StepType, params: dict[str, Any], inputs: dict[str, Any], out: Path, dry
) -> Any:
    from rapidtools.gui.workflow import DetectionSettings

    return DetectionSettings(
        raster_path=inputs['raster'],
        output_dir=out,
        **step_type.settings_kwargs(params),
    )


def _run_detect(ctx: StepContext) -> None:
    result = ctx.workflow.detect(ctx.settings)
    ctx.objects.update(result=result, assets=result.collection)
    ctx.outputs['assets'] = result.combined_geojson
    ctx.outputs['detection_raster'] = result.detection_raster
    ctx.files.extend(
        {'label': f'{name} footprints', 'path': path, 'kind': 'geojson'}
        for name, path in result.per_asset_geojson.items()
    )
    ctx.summary.update(n_assets=len(result.collection), basemap=result.basemap)


register_step(
    StepType(
        name='detect.aerial',
        title='Detect assets in aerial imagery',
        doc=(
            'Finds every instance of the named asset types in an orthomosaic '
            'with SAM 3, by default on a pre-event satellite basemap stitched '
            'over the raster extent (assets destroyed in the post-event '
            'imagery are still found there). Buildings and roads are '
            'regularized into clean polygons.'
        ),
        build=_build_detect,
        run=_run_detect,
        activity='detection',
        empty_hint='No assets were detected',
        inputs=(
            Input(
                'raster',
                'raster',
                'The georeferenced orthomosaic that defines the area.',
                required=True,
            ),
        ),
        params=(
            Param(
                'assets',
                'string_list',
                'Asset types to find, as plain-English prompts, e.g. '
                '["building", "tree"]. Each is scanned for separately.',
            ),
            Param(
                'asset_size_m',
                'number',
                'Approximate size of one asset in meters; sets the size of the '
                'patches SAM 3 looks at.',
                default=50.0,
                exclusive_minimum=0,
            ),
            Param(
                'basemap',
                'string',
                'Imagery to detect in: a "bing" or "google" basemap stitched '
                'over the raster extent, or "recon" for the raster itself.',
                default='bing',
                choices=('bing', 'google', 'recon'),
            ),
            Param(
                'threshold',
                'number',
                'Detection confidence threshold.',
                default=0.5,
                minimum=0,
                maximum=1,
            ),
            Param(
                'mask_threshold',
                'number',
                'Mask binarization threshold.',
                default=0.4,
                minimum=0,
                maximum=1,
            ),
            Param(
                'overlap_ratio',
                'number',
                'Overlap between neighbouring patches, as a fraction of the '
                'patch size.',
                default=0.25,
                minimum=0,
            ),
            Param(
                'batch_size',
                'integer',
                'Patches per SAM 3 forward pass.',
                default=4,
                minimum=1,
            ),
            Param(
                'crop_buffer',
                'string',
                'Margin around each detection in the crops used for '
                'regularization, e.g. "20 m".',
                default='20 m',
            ),
            Param(
                'regularize',
                'boolean',
                'Regularize building footprints and road polygons.',
                default=True,
            ),
            Param(
                'merge_overlaps',
                'boolean',
                'Merge detections that overlap across patches into one asset. '
                'Turn off to keep touching instances separate.',
                default=True,
            ),
            Param(
                'max_workers',
                'integer',
                'Concurrent tile downloads when a basemap is stitched.',
                default=10,
                minimum=1,
                field='bing_max_workers',
            ),
        ),
        outputs=(
            Output(
                'assets',
                'geojson',
                'All detected assets (assets_final.geojson). Absent when '
                'nothing was found.',
            ),
            Output(
                'detection_raster',
                'raster',
                'The raster detection ran on: the stitched basemap, or the '
                'input raster for "recon".',
            ),
        ),
    )
)


# ------------------------------------------------------------ detect.street
def _build_street(
    step_type: StepType, params: dict[str, Any], inputs: dict[str, Any], out: Path, dry
) -> Any:
    from rapidtools.gui.workflow import StreetDetectionSettings

    region: Region | None = inputs['region']
    area: Any = region
    if region is not None and (region.bbox is not None or not dry):
        area = region.bounding_box()
    return StreetDetectionSettings(
        output_dir=out,
        raster_path=inputs['raster'],
        region=area,
        **step_type.settings_kwargs(params),
    )


def _run_street(ctx: StepContext) -> None:
    result = ctx.workflow.discover_street(ctx.settings)
    ctx.objects.update(result=result, assets=result.collection)
    ctx.outputs['assets'] = result.geojson_path
    ctx.summary.update(n_assets=len(result.collection), n_images=result.n_images)


register_step(
    StepType(
        name='detect.street',
        title='Discover objects along a street survey',
        doc=(
            'Locates objects of the requested classes in the Mapillary '
            'street-level images of an area and fuses the sightings into one '
            'asset per object: a point for vehicles, poles and other objects, '
            'a footprint polygon for buildings. The area is the "region" '
            'input, or the extent of "raster" when no region is given.'
        ),
        build=_build_street,
        run=_run_street,
        activity='street discovery',
        empty_hint='No objects were found',
        inputs=(
            Input(
                'region',
                'region',
                'Survey area: a WGS84 box, or a GeoJSON file whose extent is used.',
            ),
            Input(
                'raster',
                'raster',
                'A raster whose extent defines the area when no region is given.',
            ),
        ),
        params=(
            Param(
                'classes',
                'string_list',
                'Object classes in plain English, e.g. ["vehicles", "utility '
                'poles", "buildings"].',
            ),
            Param(
                'access_token',
                'string',
                'Mapillary access token.',
                default='',
                secret=True,
            ),
            Param(
                'start_date',
                'string',
                'Only images captured on or after this date (YYYY-MM-DD). '
                'Empty for no limit.',
                default='',
            ),
            Param(
                'end_date',
                'string',
                'Only images captured on or before this date (YYYY-MM-DD). '
                'Empty for no limit.',
                default='',
            ),
            Param(
                'filter_rapid_only',
                'boolean',
                'Use only imagery uploaded by the RAPID Facility.',
                default=True,
            ),
            Param(
                'detection_source',
                'string',
                'Where detections come from: Mapillary segmentation metadata, '
                'SAM 3 run on the frames, Mapillary map features, or "auto" to '
                'choose per class.',
                default='auto',
                choices=('auto', 'mapillary', 'sam3', 'map_features'),
            ),
            Param(
                'frame_spacing_m',
                'number',
                'Thin each survey to one frame per this many meters; 0 keeps '
                'every frame.',
                default=3.0,
                minimum=0,
            ),
            Param(
                'min_observations',
                'integer',
                'Sightings required before an object is reported.',
                default=2,
                minimum=1,
            ),
            Param(
                'cluster_radius_m',
                'number',
                'Sightings closer than this are treated as the same object.',
                default=4.0,
                exclusive_minimum=0,
            ),
            Param(
                'camera_height_m',
                'number',
                'Height of the survey camera above the road.',
                default=2.4,
                exclusive_minimum=0,
            ),
            Param(
                'max_range_m',
                'number',
                'Ignore sightings farther from the camera than this.',
                default=60.0,
                exclusive_minimum=0,
            ),
            Param(
                'max_workers',
                'integer',
                'Concurrent Mapillary requests.',
                default=10,
                minimum=1,
            ),
        ),
        outputs=(
            Output(
                'assets',
                'geojson',
                'The discovered objects (street_objects.geojson). Absent when '
                'nothing was found.',
            ),
        ),
    )
)


# -------------------------------------------------------------- analyze.vlm
def _build_analyze(
    step_type: StepType, params: dict[str, Any], inputs: dict[str, Any], out: Path, dry
) -> Any:
    from rapidtools.gui.workflow import InferenceSettings

    return InferenceSettings(
        raster_path=inputs['raster'],
        output_dir=out,
        **step_type.settings_kwargs(params),
    )


def _run_analyze(ctx: StepContext) -> None:
    source: AssetsInput = ctx.inputs['assets']
    collection = source.load()
    if len(collection) == 0:
        if source.empty_hint:
            raise ValueError(f'{source.empty_hint}; skipping inference.')
        raise ValueError('The asset collection is empty; nothing to analyze.')
    # Attributes are written in place, so this is also the partial result
    # when the run is cancelled.
    ctx.objects['assets'] = collection
    try:
        result = ctx.workflow.analyze(collection, ctx.settings)
    except OperationCancelled:
        partial = ctx.output_dir / 'assets_inferred_partial.geojson'
        if partial.is_file():
            ctx.outputs['partial_assets'] = partial
        raise
    ctx.objects.update(result=result, assets=result.collection)
    ctx.outputs['assets'] = result.geojson_path
    ctx.summary.update(n_input=result.n_input, n_analyzed=result.n_analyzed)


register_step(
    StepType(
        name='analyze.vlm',
        title='Analyze assets with a vision-language model',
        doc=(
            'Gathers imagery of every asset (aerial crops of a raster, Google '
            'Street View, Mapillary panoramas or Mapillary object crops) and '
            'asks a vision-language model the prompt about each one. The '
            'answers become attributes of the assets. Assets that already '
            'carry attributes keep them, so two models can be run one after '
            'the other on the same file.'
        ),
        build=_build_analyze,
        run=_run_analyze,
        activity='inference',
        inputs=(
            Input(
                'assets',
                'assets',
                'The assets to analyze: a GeoJSON file, or the "assets" output '
                'of an earlier detection step.',
                required=True,
            ),
            Input(
                'raster',
                'raster',
                'The orthomosaic to crop. Required for the "aerial" imagery '
                'source only.',
            ),
        ),
        params=(
            Param(
                'prompt',
                'string',
                'The question put to the model about each asset.',
            ),
            Param(
                'backend',
                'string',
                'Model provider: a cloud API or a local checkpoint.',
                default='gemini',
                choices=_backends,
            ),
            Param(
                'api_key',
                'string',
                'API key of a cloud backend. Not used by local backends.',
                default='',
                secret=True,
            ),
            Param(
                'model_id',
                'string',
                'Model name or Hugging Face repository. Empty selects the '
                "backend's default.",
                default='',
            ),
            Param(
                'max_workers',
                'integer',
                'Concurrent requests to a cloud backend, and concurrent image '
                'downloads.',
                default=5,
                minimum=1,
            ),
            Param(
                'batch_size',
                'integer',
                'Assets per forward pass of a local backend that supports batching.',
                default=4,
                minimum=1,
            ),
            Param(
                'temperature',
                'number',
                'Sampling temperature.',
                default=0.4,
                minimum=0,
            ),
            Param(
                'max_tokens',
                'integer',
                'Longest answer the model may give, in tokens.',
                default=2048,
                minimum=1,
            ),
            Param(
                'json_mode',
                'boolean',
                'Ask the model for a JSON answer; each key becomes an attribute.',
                default=False,
            ),
            Param(
                'load_in_4bit',
                'boolean',
                'Load a local model with 4-bit quantization (less GPU memory).',
                default=True,
            ),
            Param(
                'overlay_asset_outline',
                'boolean',
                'Draw the outline of the asset on each crop so the model knows '
                'which one is meant.',
                default=True,
            ),
            Param(
                'outline_shape',
                'string',
                'Shape of the outline.',
                default='geometry',
                choices=_outline_shapes,
            ),
            Param(
                'outline_buffer',
                'number_or_string',
                'Offset of the outline from the asset: pixels, a percentage '
                '("5%") or a distance ("2 m").',
                default=0,
            ),
            Param(
                'outline_width',
                'number_or_string',
                'Stroke width: pixels, or a percentage of the shorter crop side.',
                default=6,
            ),
            Param(
                'outline_color',
                'string',
                'Outline colour: a name or a hex code ("#00ffff").',
                default='red',
            ),
            Param(
                'image_prefix',
                'string',
                'Prefix of the aerial crop file names.',
                default='aerial',
            ),
            Param(
                'only_analyze_inference_crops',
                'boolean',
                'Show the model only the crops gathered by this step, not '
                'images already attached to the assets.',
                default=True,
            ),
            Param(
                'imagery',
                'string',
                'Where the imagery of each asset comes from.',
                default='aerial',
                choices=_imagery_sources,
            ),
            Param(
                'min_footprint_coverage',
                'number',
                'Aerial crops: skip an asset when less than this fraction of '
                'its footprint lies inside the raster. null uses the extractor '
                'default.',
                default=None,
                nullable=True,
                minimum=0,
                maximum=1,
            ),
            Param(
                'pad_edges',
                'boolean',
                'Aerial crops: pad crops that reach past the raster edge '
                'instead of clipping them. null uses the extractor default.',
                default=None,
                nullable=True,
            ),
            Param(
                'street_search_radius_m',
                'number',
                'Google Street View: how far from the asset to look for a panorama.',
                default=50.0,
                exclusive_minimum=0,
            ),
            Param(
                'street_max_images',
                'integer',
                'Google Street View: panoramas per asset.',
                default=1,
                minimum=1,
            ),
            Param(
                'street_vertical_crop',
                'number_pair',
                'Google Street View: [top, bottom] fractions of the panorama '
                'height to keep, with 0 <= top < bottom <= 1.',
                default=(0.0, 1.0),
            ),
            Param(
                'mapillary_token',
                'string',
                'Mapillary access token. Required for the two Mapillary '
                'imagery sources.',
                default='',
                secret=True,
            ),
            Param(
                'mapillary_start_date',
                'string',
                'Mapillary panoramas: only images captured on or after this '
                'date (YYYY-MM-DD).',
                default='',
            ),
            Param(
                'mapillary_end_date',
                'string',
                'Mapillary panoramas: only images captured on or before this '
                'date (YYYY-MM-DD).',
                default='',
            ),
            Param(
                'mapillary_rapid_only',
                'boolean',
                'Mapillary panoramas: use only RAPID Facility uploads.',
                default=True,
            ),
            Param(
                'mapillary_max_images',
                'integer',
                'Mapillary sources: images per asset.',
                default=4,
                minimum=1,
            ),
            Param(
                'object_image_size',
                'string',
                'Mapillary object crops: thumbnail size to download, "1024", '
                '"2048" or "original".',
                default='2048',
            ),
            Param(
                'object_crop_buffer',
                'number_or_string',
                'Mapillary object crops: margin around the object, as a '
                'percentage ("25%"), pixels or a distance ("1 m").',
                default='25%',
            ),
        ),
        outputs=(
            Output(
                'assets',
                'geojson',
                'The analyzed assets with their new attributes '
                '(assets_inferred.geojson).',
            ),
            Output(
                'partial_assets',
                'geojson',
                'Written only when the step is cancelled: the assets analyzed '
                'so far (assets_inferred_partial.geojson).',
            ),
        ),
    )
)


# ------------------------------------------------------------ prompt.assist
def _build_assist(
    step_type: StepType, params: dict[str, Any], inputs: dict[str, Any], out: Path, dry
) -> Any:
    from rapidtools.gui.workflow import AssistSettings

    return AssistSettings(**step_type.settings_kwargs(params))


def _run_assist(ctx: StepContext) -> None:
    ctx.data = ctx.workflow.assist(ctx.settings)


register_step(
    StepType(
        name='prompt.assist',
        title='Ask a model for help with a prompt',
        doc=(
            'Runs one prompt-assistant action: drafting a prompt '
            'specification from a brief, refining or reviewing one, '
            'suggesting visual indicators for a class, or importing an '
            'existing prompt. The answer is returned as "data" in the result '
            'file; no file is written.'
        ),
        build=_build_assist,
        run=_run_assist,
        activity='prompt assistance',
        params=(
            Param(
                'action',
                'string',
                'What to do.',
                choices=_assist_actions,
            ),
            Param(
                'backend',
                'string',
                'Model provider answering the request.',
                default='gemma4',
                choices=_backends,
            ),
            Param(
                'api_key',
                'string',
                'API key of a cloud backend.',
                default='',
                secret=True,
            ),
            Param(
                'model_id',
                'string',
                "Model name. Empty selects the backend's default.",
                default='',
            ),
            Param(
                'load_in_4bit',
                'boolean',
                'Load a local model with 4-bit quantization.',
                default=True,
            ),
            Param(
                'brief',
                'string',
                'Plain-language description of the task ("draft").',
                default='',
            ),
            Param(
                'instruction',
                'string',
                'What to change ("refine").',
                default='',
            ),
            Param(
                'prompt_text',
                'string',
                'An existing prompt ("review", "import").',
                default='',
            ),
            Param(
                'class_value',
                'string',
                'The class to describe ("indicators").',
                default='',
            ),
            Param(
                'spec',
                'object',
                'The current prompt specification, as produced by the prompt builder.',
                default=None,
                nullable=True,
            ),
            Param(
                'context',
                'object',
                'Extra facts for the assistant (asset type, imagery, outline).',
                default={},
            ),
            Param(
                'temperature',
                'number',
                'Sampling temperature.',
                default=0.2,
                minimum=0,
            ),
            Param(
                'max_tokens',
                'integer',
                'Longest answer the model may give, in tokens.',
                default=4096,
                minimum=1,
            ),
        ),
    )
)
