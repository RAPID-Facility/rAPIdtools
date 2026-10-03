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
# 09-30-2026

"""
Structured prompt builder and language-model assistant for the rapidtools GUI.

The sample prompts shipped with rapidtools (for example the aerial CHS damage
prompt) share one layout: a ``TASK`` statement, numbered ``ANALYSIS STEPS``,
``ANALYSIS EDGE CASES``, a ``REQUIRED OUTPUT FORMAT`` block whose ``Key:
[options]`` lines the :class:`~rapidtools.processing.AssetAnalyzer` parses
into asset attributes, and a rubric describing every class with its key
indicators. :class:`PromptSpec` captures those parts as data,
:func:`assemble_prompt` renders them in that layout, and
:class:`PromptAssistant` asks any rapidtools model backend to draft, refine,
review or import such a specification.

The module has no heavy dependencies; the assistant only needs a model that
implements :meth:`~rapidtools.models.base.BaseInferenceModel.run_inference`
and accepts an empty image list, which every rapidtools wrapper does.

Example:
    >>> from rapidtools.gui.prompt_builder import (
    ...     OutputField, PromptSpec, RubricEntry, assemble_prompt
    ... )
    >>>
    >>> spec = PromptSpec(
    ...     asset='building',
    ...     objective='assign a damage level to the primary building',
    ...     fields=[
    ...         OutputField('Damage Level', options=['None', 'Minor', 'Major']),
    ...         OutputField('Justification', kind='text',
    ...                     description='2-3 sentences citing visual evidence'),
    ...     ],
    ...     rubric=[RubricEntry('Minor', 'Cosmetic damage',
    ...                         indicators=['Roof intact', 'Soot staining'])],
    ... )
    >>> print(assemble_prompt(spec).splitlines()[0])
    TASK: Analyze each provided image and assign a damage level to the primary building.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rapidtools.models.base import BaseInferenceModel

logger = logging.getLogger(__name__)

FIELD_KINDS = ('category', 'number', 'text', 'boolean')
IMAGERY_KINDS = ('aerial', 'street', 'multi-temporal street')
ASSIST_ACTIONS = ('draft', 'refine', 'review', 'indicators', 'import')

# How the outline drawn by the imagery extractors is described to the model.
_SHAPE_WORDS = {
    'geometry': 'outline',
    'bbox': 'bounding box',
    'rotated_bbox': 'rotated bounding box',
    'convex_hull': 'outline',
    'corners': 'corner brackets',
}


def describe_marking(shape: str, color: str, overlay: bool = True) -> str:
    """
    Phrase the outline drawn on each crop the way the sample prompts do.

    Args:
        shape (str): One of the extractor ``outline_shape`` values.
        color (str): The PIL colour name or hex code used for the outline.
        overlay (bool): ``False`` when no outline is drawn.

    Returns:
        str: For example ``'roughly marked with a red outline'`` or
        ``'marked with cyan corner brackets'``; an empty string when nothing
        is drawn.

    Example:
        >>> describe_marking('corners', '#00ffff')
        'marked with #00ffff corner brackets'
        >>> describe_marking('bbox', 'cyan')
        'marked with a cyan bounding box'
        >>> describe_marking('geometry', 'red', overlay=False)
        ''
    """
    if not overlay:
        return ''
    word = _SHAPE_WORDS.get(shape, 'outline')
    colour = str(color).strip() or 'red'
    prefix = (
        'roughly marked with' if shape in ('geometry', 'convex_hull') else 'marked with'
    )
    article = '' if word.endswith('s') else 'a '
    return f'{prefix} {article}{colour} {word}'


def snake_case(name: str) -> str:
    """
    Turn an output field name into the attribute key the analyzer writes.

    Mirrors :func:`rapidtools.processing.image_analyzers.parse_structured_results`,
    which lower-cases ``Key: value`` lines and replaces spaces with
    underscores.

    Example:
        >>> snake_case('CHS Level')
        'chs_level'
    """
    return re.sub(r'\s+', '_', name.strip().lower())


@dataclass
class OutputField:
    """
    One line of the ``REQUIRED OUTPUT FORMAT`` block.

    Attributes:
        name: Field label as the model should write it (``'CHS Level'``).
        kind: ``'category'`` (choose one of ``options``), ``'number'`` (within
            ``minimum``..``maximum``), ``'text'`` or ``'boolean'``.
        options: Allowed values for a categorical field.
        minimum: Lower bound of a numeric field, if any.
        maximum: Upper bound of a numeric field, if any.
        description: How to fill the field (``'2-3 sentences citing ...'``).

    Example:
        >>> OutputField('Confidence', kind='number', minimum=1, maximum=3).format_line()
        'Confidence: [number from 1 to 3]'
    """

    name: str
    kind: str = 'category'
    options: list[str] = field(default_factory=list)
    minimum: float | None = None
    maximum: float | None = None
    description: str = ''

    def __post_init__(self) -> None:
        """Normalise the kind and strip whitespace from the options."""
        self.name = ' '.join(str(self.name).split())
        self.kind = str(self.kind or 'category').strip().lower()
        if self.kind not in FIELD_KINDS:
            raise ValueError(f'kind must be one of {FIELD_KINDS}, got {self.kind!r}.')
        self.options = [
            ' '.join(str(o).split()) for o in self.options if str(o).strip()
        ]
        self.description = ' '.join(str(self.description or '').split())

    @property
    def key(self) -> str:
        """The ``snake_case`` attribute key the analyzer will produce."""
        return snake_case(self.name)

    def placeholder(self) -> str:
        """The bracketed hint written after the field name."""
        if self.kind == 'category' and self.options:
            hint = ', '.join(self.options)
        elif self.kind == 'number':
            if self.minimum is not None and self.maximum is not None:
                hint = f'number from {_fmt(self.minimum)} to {_fmt(self.maximum)}'
            else:
                hint = 'number'
        elif self.kind == 'boolean':
            hint = 'true or false'
        else:
            hint = self.description or 'free text'
        if self.kind != 'text' and self.description:
            hint = f'{hint}; {self.description}'
        return hint

    def format_line(self) -> str:
        """Render the ``Name: [hint]`` line of the plain-text output format."""
        return f'{self.name}: [{self.placeholder()}]'

    def json_hint(self) -> Any:
        """Render the value shown in the JSON schema variant of the output."""
        if self.kind == 'category' and self.options:
            hint = 'one of: ' + ', '.join(self.options)
        elif self.kind == 'number':
            hint = 'number'
            if self.minimum is not None and self.maximum is not None:
                hint += f' from {_fmt(self.minimum)} to {_fmt(self.maximum)}'
        elif self.kind == 'boolean':
            return 'true/false'
        else:
            hint = self.description or 'free text'
        if self.kind != 'text' and self.description:
            hint = f'{hint}; {self.description}'
        return hint


@dataclass
class RubricEntry:
    """
    Description of one class of the primary categorical field.

    Attributes:
        value: The option this entry explains (``'2'``).
        title: Short name (``'Partial Structural Combustion'``).
        description: What the class means.
        indicators: Visual cues that identify it, one per bullet.

    Example:
        >>> RubricEntry('4', 'Complete Combustion', indicators=['Only ash']).value
        '4'
    """

    value: str
    title: str = ''
    description: str = ''
    indicators: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Strip whitespace and drop empty indicators."""
        self.value = ' '.join(str(self.value).split())
        self.title = ' '.join(str(self.title or '').split())
        self.description = str(self.description or '').strip()
        self.indicators = _clean_lines(self.indicators)


@dataclass
class PromptSpec:
    """
    Everything needed to assemble a prompt in the rapidtools sample layout.

    Attributes:
        asset: What is being analysed (``'building'``, ``'vehicle'``).
        imagery: ``'aerial'``, ``'street'`` or ``'multi-temporal street'``.
        objective: What the model must do, phrased after "Analyze each
            provided image and ..." (``'assign a CHS Level (0-4) to the
            primary building structure'``).
        persona: Expertise the model should adopt.
        focus: Scope statements such as what to ignore.
        marking: How the asset is marked on the crop, from
            :func:`describe_marking`.
        steps: Numbered analysis steps; generated when empty.
        edge_cases: Sentences describing alternative classifications.
        fields: The output fields, in order.
        rubric_title: Heading of the class descriptions block.
        rubric_field: Name of the field the rubric explains; defaults to the
            first categorical field.
        rubric: One :class:`RubricEntry` per class.
        context: Optional background paragraph appended at the end.
        json_output: Ask for a JSON object instead of ``Key: value`` lines.

    Example:
        >>> spec = PromptSpec.from_dict({'asset': 'road', 'fields': [
        ...     {'name': 'Blocked', 'kind': 'boolean'}]})
        >>> spec.fields[0].format_line()
        'Blocked: [true or false]'
    """

    asset: str = 'building'
    imagery: str = 'aerial'
    objective: str = ''
    persona: str = ''
    focus: str = ''
    marking: str = ''
    steps: list[str] = field(default_factory=list)
    edge_cases: list[str] = field(default_factory=list)
    fields: list[OutputField] = field(default_factory=list)
    rubric_title: str = ''
    rubric_field: str = ''
    rubric: list[RubricEntry] = field(default_factory=list)
    context: str = ''
    json_output: bool = False

    def __post_init__(self) -> None:
        """Normalise strings and coerce nested dicts into dataclasses."""
        self.asset = ' '.join(str(self.asset or 'building').split()) or 'building'
        self.imagery = str(self.imagery or 'aerial').strip().lower()
        if self.imagery not in IMAGERY_KINDS:
            self.imagery = 'aerial'
        for attr in (
            'objective',
            'persona',
            'focus',
            'marking',
            'rubric_title',
            'rubric_field',
        ):
            setattr(self, attr, ' '.join(str(getattr(self, attr) or '').split()))
        # Models (and people) often restate the lead-in; keep only the goal.
        self.objective = re.sub(
            r'^(?:analy[sz]e|examine|look at)\s+(?:each|the|every|all)\s+'
            r'(?:provided\s+)?(?:series of\s+)?(?:street-level\s+|aerial\s+)?'
            r'images?\s+(?:and|to)\s+',
            '',
            self.objective,
            flags=re.IGNORECASE,
        )
        self.rubric_title = self.rubric_title.rstrip(':')
        self.context = str(self.context or '').strip()
        self.steps = _clean_lines(self.steps)
        self.edge_cases = _clean_lines(self.edge_cases)
        self.fields = [
            f
            if isinstance(f, OutputField)
            else OutputField(**_only_known(f, OutputField))
            for f in self.fields
            if isinstance(f, OutputField)
            or (isinstance(f, dict) and str(f.get('name', '')).strip())
        ]
        self.rubric = [
            r
            if isinstance(r, RubricEntry)
            else RubricEntry(**_only_known(r, RubricEntry))
            for r in self.rubric
            if isinstance(r, RubricEntry)
            or (isinstance(r, dict) and str(r.get('value', '')).strip())
        ]
        self.json_output = bool(self.json_output)

    # ------------------------------------------------------------ helpers
    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PromptSpec:
        """
        Build a spec from a JSON object, ignoring unknown keys.

        Args:
            data (dict[str, Any] | None): Keys matching the attributes; nested
                ``fields`` and ``rubric`` entries may be dicts.

        Returns:
            PromptSpec: The coerced specification.

        Raises:
            ValueError: If ``data`` is not an object or a field kind is invalid.
        """
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError('The prompt specification must be a JSON object.')
        return cls(**_only_known(data, cls))

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain JSON-compatible data."""
        return asdict(self)

    def primary_field(self) -> OutputField | None:
        """
        The field the rubric describes.

        Returns:
            OutputField | None: The field named by ``rubric_field`` if it
            exists, else the first categorical field, else ``None``.
        """
        if self.rubric_field:
            for f in self.fields:
                if f.name.lower() == self.rubric_field.lower():
                    return f
        for f in self.fields:
            if f.kind == 'category':
                return f
        return None

    def attribute_keys(self, prefix: str = '') -> list[str]:
        """
        Attribute names the analyzer will write for this prompt.

        Args:
            prefix (str): The backend's attribute prefix (``'gemini'``).

        Returns:
            list[str]: ``'<prefix>_<key>'`` per field (``'<key>'`` without a
            prefix).
        """
        return [f'{prefix}_{f.key}' if prefix else f.key for f in self.fields]

    def default_steps(self) -> list[str]:
        """Analysis steps used when the user has not written any."""
        marking = f' {self.marking}' if self.marking else ''
        primary = self.primary_field()
        steps = [f'Locate the primary {self.asset}{marking}']
        if self.imagery == 'multi-temporal street':
            steps.append(
                'Describe the state of the primary asset in each image before '
                'comparing them'
            )
        described = any(r.title or r.description or r.indicators for r in self.rubric)
        if described and primary is not None:
            steps.append(
                f'Compare what you see against the {self.rubric_title or primary.name} '
                'indicators below'
            )
        else:
            steps.append(f'Examine the visible condition of the {self.asset} in detail')
        if primary is not None:
            steps.append(f'Assign the {primary.name} with a brief justification')
        else:
            steps.append('Fill in every field of the required output format')
        return steps


def _fmt(value: float) -> str:
    """Format a bound without a trailing ``.0``."""
    return str(int(value)) if float(value).is_integer() else str(value)


def _clean_lines(items: Any) -> list[str]:
    """Coerce a list (or newline-separated string) into stripped lines."""
    if isinstance(items, str):
        items = items.splitlines()
    return [
        ' '.join(str(i).split())
        for i in (items or [])
        if i is not None and str(i).strip()
    ]


def _only_known(data: dict[str, Any], cls: type) -> dict[str, Any]:
    """Keep only the keys ``cls`` accepts, so foreign JSON does not crash."""
    known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    return {k: v for k, v in data.items() if k in known}


# ================================================================ assembly
def assemble_prompt(spec: PromptSpec) -> str:
    """
    Render a :class:`PromptSpec` in the layout of the rapidtools sample prompts.

    The result has these blocks, each omitted when it has nothing to say:
    ``TASK``, ``ANALYSIS STEPS``, ``ANALYSIS EDGE CASES``, ``REQUIRED OUTPUT
    FORMAT``, the rubric under ``rubric_title`` and ``CONTEXT AND
    BACKGROUND``.

    Args:
        spec (PromptSpec): The specification to render.

    Returns:
        str: The prompt text.

    Example:
        >>> from rapidtools.gui.prompt_builder import (
        ...     OutputField, PromptSpec, assemble_prompt
        ... )
        >>>
        >>> text = assemble_prompt(PromptSpec(
        ...     asset='vehicle', imagery='street',
        ...     objective='classify the condition of the marked vehicle',
        ...     fields=[OutputField('Condition', options=['intact', 'damaged'])],
        ... ))
        >>> 'Condition: [intact, damaged]' in text
        True
    """
    blocks: list[str] = []

    # TASK -----------------------------------------------------------------
    task_bits: list[str] = []
    image_word = {
        'aerial': 'each provided image',
        'street': 'each provided street-level image',
        'multi-temporal street': 'the provided series of street-level images',
    }[spec.imagery]
    objective = (
        spec.objective.rstrip('.')
        or f'describe the condition of the primary {spec.asset}'
    )
    task_bits.append(f'Analyze {image_word} and {objective}.')
    if spec.focus:
        task_bits.append(_sentence(spec.focus))
    if spec.persona:
        task_bits.append(f'Act as {_role_phrase(spec.persona)}.')
    blocks.append('TASK: ' + ' '.join(task_bits))

    # ANALYSIS STEPS -------------------------------------------------------
    steps = spec.steps or spec.default_steps()
    blocks.append(
        'ANALYSIS STEPS:\n'
        + '\n'.join(f'{i}. {_strip_numbering(s)}' for i, s in enumerate(steps, 1))
    )

    # EDGE CASES -----------------------------------------------------------
    if spec.edge_cases:
        blocks.append(
            'ANALYSIS EDGE CASES:\n' + '\n'.join(_sentence(e) for e in spec.edge_cases)
        )

    # OUTPUT FORMAT --------------------------------------------------------
    if spec.fields:
        if spec.json_output:
            schema = {f.key: f.json_hint() for f in spec.fields}
            body = json.dumps(schema, indent=2, ensure_ascii=False)
            body = body.replace('"true/false"', 'true/false')
            blocks.append(
                'REQUIRED OUTPUT FORMAT:\n'
                'Return ONLY a valid JSON object matching this exact schema, with no '
                'markdown or text outside it:\n' + body
            )
        else:
            blocks.append(
                'REQUIRED OUTPUT FORMAT:\n'
                + '\n'.join(f.format_line() for f in spec.fields)
            )

    # RUBRIC ---------------------------------------------------------------
    primary = spec.primary_field()
    described = [e for e in spec.rubric if e.title or e.description or e.indicators]
    if described:
        title = spec.rubric_title or (
            f'{primary.name.upper()} DEFINITIONS' if primary else 'CLASS DEFINITIONS'
        )
        label = primary.name if primary else 'Class'
        entries: list[str] = []
        for entry in described:
            lines = [
                f'{label} {entry.value}' + (f': {entry.title}' if entry.title else '')
            ]
            if entry.description:
                lines.append(f'Description: {entry.description}')
            if entry.indicators:
                lines.append('Key indicators:')
                lines.extend(f'- {ind}' for ind in entry.indicators)
            entries.append('\n'.join(lines))
        blocks.append(f'{title.rstrip(":")}:\n\n' + '\n\n'.join(entries))

    # CONTEXT --------------------------------------------------------------
    if spec.context:
        blocks.append('CONTEXT AND BACKGROUND:\n' + spec.context)

    return '\n\n'.join(blocks).strip() + '\n'


def _role_phrase(persona: str) -> str:
    """
    Turn a persona into the object of "Act as ...".

    ``'an expert in X'`` and ``'the lead assessor'`` are kept; a job title such
    as ``'Senior Damage Assessor'`` gets an article; a lower-case field such
    as ``'forensic engineering'`` becomes ``'an expert in forensic engineering'``.

    Example:
        >>> _role_phrase('Senior Damage Assessor')
        'a Senior Damage Assessor'
        >>> _role_phrase('forensic engineering')
        'an expert in forensic engineering'
        >>> _role_phrase('an insurance adjuster.')
        'an insurance adjuster'
    """
    persona = persona.strip().rstrip('.')
    if re.match(r'^(an?|the)\s', persona, re.IGNORECASE):
        return persona
    lowered = persona.lower()
    if 'expert' in lowered or re.search(
        r'(?:^|\s)(?:assessor|engineer|inspector|analyst|specialist|adjuster|'
        r'surveyor|officer|scientist|planner|professional|consultant)s?(?:\s|$)',
        lowered,
    ):
        article = 'an' if lowered[0] in 'aeiou' else 'a'
        return f'{article} {persona}'
    return f'an expert in {persona}'


def _sentence(text: str) -> str:
    """Capitalise and terminate a sentence fragment."""
    text = text.strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if text[-1] in '.!?:' else text + '.'


def _strip_numbering(step: str) -> str:
    """Remove a leading ``1.`` / ``1)`` / ``-`` the user may have typed."""
    return re.sub(r'^\s*(?:\d+[.)]|[-*•])\s*', '', step).strip()


# ============================================================ sample spec
SAMPLE_SPEC = PromptSpec(
    asset='building structure',
    imagery='aerial',
    objective=(
        'assign a CHS Level (0, 1, 2, 3, 4) to the primary building structure, '
        'based on the CHS Combustion Index provided'
    ),
    persona=(
        'an expert in forensic engineering, wildfire damage assessment, and '
        'combustible materials'
    ),
    focus='Do not focus your attention on non-combustible materials',
    marking='roughly marked with a red outline',
    steps=[
        'Locate the primary building footprint roughly marked with a red outline',
        'Apply appropriate CHS indicators and combustion descriptions',
        'Assign CHS Level with brief justification',
    ],
    edge_cases=[
        'Alternative assigned classifications for each image include "No Data", as '
        'outlined in the CHS COMBUSTION INDEX section, and "No Data-Footprint Unclear"',
        'Assign a CHS Level of "No Data-Footprint Unclear" if the provided image does '
        'not appear to have a singular mostly complete building footprint. Look '
        'specifically for the presence of multiple complete or near-complete '
        'footprints',
    ],
    fields=[
        OutputField(
            'CHS Level',
            options=['0', '1', '2', '3', '4', 'No Data', 'No Data-Footprint Unclear'],
        ),
        OutputField(
            'Justification',
            kind='text',
            description='2-3 sentences explaining key visual evidence',
        ),
    ],
    rubric_title='CHS COMBUSTION INDEX',
    rubric_field='CHS Level',
    rubric=[
        RubricEntry(
            '0',
            'Unaffected / No Direct Combustion',
            'The structure was not directly ignited and did not participate in '
            'combustion.',
            [
                'Roof is fully intact with uniform color and texture',
                'Building footprint is crisp and unbroken',
                'No visible fire-related debris on the roof or property',
            ],
        ),
        RubricEntry(
            '1',
            'Superficial Combustion & Radiant Heat Damage',
            'Intense radiant heat or brief flame contact ignited only the most '
            'vulnerable '
            'exterior surfaces (paint, plastic siding, small decorative elements).',
            [
                'Visible soot staining or discoloration on the roof surface',
                'Damage to plastic roof features like vents or skylight domes',
                'The primary roof structure is not breached',
            ],
        ),
        RubricEntry(
            '2',
            'Partial Structural Combustion',
            'The fire breached the exterior envelope and caused sustained '
            'combustion of a '
            'limited portion of the internal structure.',
            [
                'A clear, distinct breach or hole in the roof exposing the interior '
                'below',
                'A limited section of the roof is missing or has collapsed inward',
                'The overall building footprint remains intact',
            ],
        ),
        RubricEntry(
            '3',
            'Major Structural Combustion / Incomplete Reduction',
            'The fire propagated through most of the structure, leaving charred '
            'skeletal remains.',
            [
                'The roof is almost entirely gone',
                'The charred tops of standing interior and exterior walls are '
                'visible from above',
                'A large debris pile is contained within the remaining skeletal walls',
            ],
        ),
        RubricEntry(
            '4',
            'Complete Combustion / Full Reduction to Ash',
            'Virtually all combustible structural mass was reduced to ash with no '
            'recognizable '
            'skeletal structure.',
            [
                'The skeletal structure is gone',
                'The entire foundation footprint is filled with a homogenous pile of '
                'ash and debris',
                'Only foundation or non-combustible walls (brick, masonry, stucco) '
                'remain',
                'Non-combustible items such as chimney stacks or vehicle chassis may '
                'protrude from the ash',
            ],
        ),
        RubricEntry(
            'No Data',
            'Combustion level cannot be assessed',
            'The site has been altered to an extent that its initial damage state '
            'cannot be analyzed.',
            [
                'Excess of dirt, indicating a bulldozed or demolished plot',
                'Construction or demolition vehicles are present on site',
            ],
        ),
    ],
)


# ============================================================== assistant
_SPEC_SCHEMA = """{
  "asset": "what is analysed, e.g. building",
  "imagery": "aerial | street | multi-temporal street",
  "objective": "what to do, phrased after 'Analyze each provided image and ...'",
  "persona": "expertise the model should adopt",
  "focus": "scope rules, e.g. what to ignore",
  "steps": ["numbered analysis steps"],
  "edge_cases": ["when to use alternative classes such as No Data"],
  "fields": [
    {"name": "Field Name", "kind": "category | number | text | boolean",
     "options": ["for category fields"], "minimum": null, "maximum": null,
     "description": "how to fill it"}
  ],
  "rubric_title": "heading for the class definitions",
  "rubric_field": "name of the categorical field the rubric explains",
  "rubric": [
    {"value": "option value", "title": "short name", "description": "what it means",
     "indicators": ["visual cues, one per item"]}
  ],
  "context": "optional background paragraph",
  "json_output": false
}"""

_ASSISTANT_ROLE = (
    'You help engineers write prompts for a vision-language model that grades '
    'infrastructure assets (buildings, roads, vehicles, poles) in post-disaster '
    'imagery, one cropped image at a time. Good prompts state the task and the '
    "expert's role, number the analysis steps, list edge cases, fix an exact "
    'output format whose fields become GIS attributes, and describe every class '
    'with concrete visual indicators that are observable in that kind of imagery.'
)


class PromptAssistant:
    """
    Use a rapidtools model backend to help write a :class:`PromptSpec`.

    Every method sends a text-only request through
    :meth:`~rapidtools.models.base.BaseInferenceModel.run_inference` with an
    empty image list and asks for JSON where a specification is expected. The
    reply is parsed leniently (code fences and surrounding prose are
    tolerated) and coerced with :meth:`PromptSpec.from_dict`.

    Args:
        model (BaseInferenceModel): Any loaded rapidtools model, for example
            ``rapidtools.models.load('gemma4')``.
        temperature (float): Sampling temperature for the requests.
        max_tokens (int): Generation budget; specifications with long rubrics
            need a few thousand tokens.

    Example:
        >>> from rapidtools.models import load
        >>> from rapidtools.gui.prompt_builder import PromptAssistant
        >>>
        >>> assistant = PromptAssistant(load('gemma4'))
        >>> spec = assistant.draft(
        ...     'Grade roof damage of houses after a hurricane into four levels',
        ...     context={'asset': 'building', 'imagery': 'aerial'},
        ... )
        >>> [f.name for f in spec.fields]
        ['Damage Level', 'Justification']
    """

    def __init__(
        self,
        model: BaseInferenceModel,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> None:
        """Bind the assistant to a loaded model."""
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ------------------------------------------------------------ actions
    def run(self, action: str, **kwargs: Any) -> dict[str, Any]:
        """
        Dispatch one of :data:`ASSIST_ACTIONS` and return a JSON-safe result.

        Args:
            action (str): ``'draft'``, ``'refine'``, ``'review'``,
                ``'indicators'`` or ``'import'``.
            **kwargs: Forwarded to the matching method.

        Returns:
            dict[str, Any]: ``{'spec': {...}}`` for actions that produce a
            specification, ``{'text': ...}`` for ``review`` and
            ``{'indicators': [...]}`` for ``indicators``.

        Raises:
            ValueError: For an unknown action or an unusable model reply.
        """
        if action == 'draft':
            return {
                'spec': self.draft(
                    kwargs.get('brief', ''), kwargs.get('context')
                ).to_dict()
            }
        if action == 'refine':
            spec = PromptSpec.from_dict(kwargs.get('spec'))
            return {'spec': self.refine(spec, kwargs.get('instruction', '')).to_dict()}
        if action == 'review':
            spec_data = kwargs.get('spec')
            text = kwargs.get('prompt_text') or (
                assemble_prompt(PromptSpec.from_dict(spec_data)) if spec_data else ''
            )
            return {'text': self.review(text)}
        if action == 'indicators':
            spec = PromptSpec.from_dict(kwargs.get('spec'))
            return {
                'indicators': self.suggest_indicators(
                    spec, kwargs.get('class_value', '')
                )
            }
        if action == 'import':
            return {'spec': self.import_prompt(kwargs.get('prompt_text', '')).to_dict()}
        raise ValueError(f'Unknown assistant action: {action!r}')

    def draft(self, brief: str, context: dict[str, Any] | None = None) -> PromptSpec:
        """
        Draft a complete specification from a plain-English description.

        Args:
            brief (str): What the user wants to find out, in their words.
            context (dict[str, Any] | None): Known facts from the GUI such as
                ``asset``, ``imagery``, ``marking`` and ``json_output``.

        Returns:
            PromptSpec: The drafted specification.

        Raises:
            ValueError: If ``brief`` is empty or the reply holds no JSON object.
        """
        if not brief.strip():
            raise ValueError(
                'Describe what you want to assess before asking for a draft.'
            )
        context = context or {}
        prompt = (
            f'{_ASSISTANT_ROLE}\n\n'
            'Write a complete prompt specification for the request below. Use '
            'concrete, visually observable indicators; include at least three '
            'indicators per class; add edge cases for unclear or missing data; make '
            'the first field the main classification and the last field a short '
            'Justification. Return ONLY a JSON object with this schema:\n'
            f'{_SPEC_SCHEMA}\n\n'
            'Known context (keep these values): '
            f'{json.dumps(context, ensure_ascii=False)}\n\n'
            f'Request: {brief.strip()}'
        )
        spec = self._spec_from_reply(self._ask(prompt, json_mode=True))
        return _apply_context(spec, context)

    def refine(self, spec: PromptSpec, instruction: str) -> PromptSpec:
        """
        Change an existing specification following an instruction.

        Args:
            spec (PromptSpec): The current specification.
            instruction (str): What to change (``'add a confidence field 1-3'``).

        Returns:
            PromptSpec: The revised specification.

        Raises:
            ValueError: If ``instruction`` is empty or the reply is unusable.
        """
        if not instruction.strip():
            raise ValueError('Say what should change before asking for a refinement.')
        prompt = (
            f'{_ASSISTANT_ROLE}\n\n'
            'Revise the prompt specification below according to the instruction. '
            'Keep everything the instruction does not touch. Return ONLY the full '
            'revised JSON object with the same schema.\n\n'
            'Current specification:\n'
            f'{json.dumps(spec.to_dict(), ensure_ascii=False, indent=1)}\n\n'
            f'Instruction: {instruction.strip()}'
        )
        return self._spec_from_reply(self._ask(prompt, json_mode=True))

    def review(self, prompt_text: str) -> str:
        """
        Critique a prompt and suggest improvements.

        Args:
            prompt_text (str): The assembled prompt.

        Returns:
            str: A short review as plain text with bullet points.

        Raises:
            ValueError: If the prompt is empty or the model returned nothing.
        """
        if not prompt_text.strip():
            raise ValueError('There is no prompt to review yet.')
        prompt = (
            f'{_ASSISTANT_ROLE}\n\n'
            'Review the prompt below as a critical colleague. In at most eight short '
            'bullet points, name ambiguities, classes without observable indicators, '
            'missing edge cases, output-format problems (every field must be '
            'parseable as "Name: value" or JSON) and anything a vision model cannot '
            'see in a single cropped image. End with one sentence on overall '
            'readiness. Plain text only, no JSON.\n\n'
            f'PROMPT:\n{prompt_text.strip()}'
        )
        text = self._ask(prompt, json_mode=False)
        if not text.strip():
            raise ValueError('The model returned an empty review.')
        return text.strip()

    def suggest_indicators(self, spec: PromptSpec, class_value: str) -> list[str]:
        """
        Propose visual indicators for one class of the primary field.

        Args:
            spec (PromptSpec): The current specification (for asset, imagery
                and neighbouring classes).
            class_value (str): The option value to describe.

        Returns:
            list[str]: Three to six indicator sentences.

        Raises:
            ValueError: If ``class_value`` is empty or no list was returned.
        """
        if not class_value.strip():
            raise ValueError('Pick a class to suggest indicators for.')
        primary = spec.primary_field()
        existing = next((r for r in spec.rubric if r.value == class_value), None)
        prompt = (
            f'{_ASSISTANT_ROLE}\n\n'
            f'Asset: {spec.asset}. Imagery: {spec.imagery}. Field: '
            f'{primary.name if primary else "class"} with options '
            f'{primary.options if primary else []}.\n'
            f'Class to describe: {class_value}'
            + (f' ({existing.title}). {existing.description}' if existing else '')
            + '\n\nList 3 to 6 key visual indicators that distinguish this class from '
            'its neighbours in this kind of imagery. Each indicator is one short, '
            'concrete sentence about something visible in the image. Return ONLY a '
            'JSON array of strings.'
        )
        reply = self._ask(prompt, json_mode=True)
        data = _extract_json(reply, want_list=True)
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), None)
        if not isinstance(data, list) or not data:
            raise ValueError('The model did not return a list of indicators.')
        return _clean_lines([str(i) for i in data])

    def import_prompt(self, prompt_text: str) -> PromptSpec:
        """
        Split an existing free-form prompt into a specification.

        Args:
            prompt_text (str): Any prompt, in the sample layout or not.

        Returns:
            PromptSpec: The extracted specification.

        Raises:
            ValueError: If the prompt is empty or the reply is unusable.
        """
        if not prompt_text.strip():
            raise ValueError('Paste or write a prompt to import first.')
        prompt = (
            f'{_ASSISTANT_ROLE}\n\n'
            'Convert the prompt below into a specification, preserving its wording '
            'wherever possible: task, persona, scope, steps, edge cases, every output '
            'field with its options, and every class description with its '
            'indicators. Return ONLY a JSON object with this schema:\n'
            f'{_SPEC_SCHEMA}\n\nPROMPT:\n{prompt_text.strip()}'
        )
        return self._spec_from_reply(self._ask(prompt, json_mode=True))

    # ------------------------------------------------------------ plumbing
    def _ask(self, prompt: str, json_mode: bool) -> str:
        """Send a text-only request and return the reply text (``''`` on failure)."""
        output = self.model.run_inference(
            [],
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            json_mode=json_mode,
        )
        text = getattr(output, 'text', None) if output is not None else None
        if not text or not text.strip():
            raise ValueError(
                'The model returned no text. Check the API key, model ID and the '
                'log for the provider error.'
            )
        return text

    @staticmethod
    def _spec_from_reply(reply: str) -> PromptSpec:
        """Parse a JSON specification out of a model reply."""
        data = _extract_json(reply)
        if not isinstance(data, dict):
            raise ValueError(
                'The model did not return a JSON specification. Try again or pick a '
                'larger model.'
            )
        return PromptSpec.from_dict(data)


def _apply_context(spec: PromptSpec, context: dict[str, Any]) -> PromptSpec:
    """Force GUI-known facts (marking, imagery, JSON mode) onto a drafted spec."""
    if context.get('marking'):
        spec.marking = str(context['marking'])
    if context.get('imagery') in IMAGERY_KINDS:
        spec.imagery = str(context['imagery'])
    if 'json_output' in context:
        spec.json_output = bool(context['json_output'])
    if context.get('asset') and not spec.asset:
        spec.asset = str(context['asset'])
    return spec


def _extract_json(text: str, want_list: bool = False) -> Any:
    """
    Pull the first JSON object (or array) out of a model reply.

    Handles ```json fences, leading prose and trailing commentary. Returns
    ``None`` when nothing parses.
    """
    if not text:
        return None
    candidates: list[str] = []
    fenced = re.findall(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
    candidates.extend(fenced)
    candidates.append(text)
    open_, close = ('[', ']') if want_list else ('{', '}')
    for cand in candidates:
        cand = cand.strip()
        for attempt in (
            cand,
            _bracketed(cand, open_, close),
            _bracketed(cand, '{', '}'),
        ):
            if not attempt:
                continue
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue
    return None


def _bracketed(text: str, open_: str, close: str) -> str:
    """Return the substring from the first ``open_`` to the last ``close``."""
    start, end = text.find(open_), text.rfind(close)
    return text[start : end + 1] if 0 <= start < end else ''


__all__ = [
    'ASSIST_ACTIONS',
    'FIELD_KINDS',
    'IMAGERY_KINDS',
    'OutputField',
    'PromptAssistant',
    'PromptSpec',
    'RubricEntry',
    'SAMPLE_SPEC',
    'assemble_prompt',
    'describe_marking',
    'snake_case',
]
