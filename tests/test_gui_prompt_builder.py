"""Tests for the prompt builder and language-model assistant behind the GUI."""

import json

import pytest

from rapidtools.gui.prompt_builder import (
    SAMPLE_SPEC,
    OutputField,
    PromptAssistant,
    PromptSpec,
    RubricEntry,
    assemble_prompt,
    describe_marking,
    snake_case,
)
from rapidtools.models.base import ModelOutput


# ------------------------------------------------------------------ helpers
def test_describe_marking_and_snake_case():
    assert describe_marking('geometry', 'red') == 'roughly marked with a red outline'
    assert (
        describe_marking('corners', '#00ffff') == 'marked with #00ffff corner brackets'
    )
    assert describe_marking('bbox', '') == 'marked with a red bounding box'
    assert describe_marking('geometry', 'red', overlay=False) == ''
    assert snake_case('  CHS   Level ') == 'chs_level'


def test_output_field_lines_and_json_hints():
    cat = OutputField('CHS Level', options=['0', ' 1 ', '', 'No Data'])
    assert cat.options == ['0', '1', 'No Data']
    assert cat.format_line() == 'CHS Level: [0, 1, No Data]'
    assert cat.json_hint() == 'one of: 0, 1, No Data'
    assert cat.key == 'chs_level'

    num = OutputField(
        'Confidence', kind='number', minimum=1, maximum=3, description='3 = sure'
    )
    assert num.format_line() == 'Confidence: [number from 1 to 3; 3 = sure]'
    assert OutputField('Score', kind='number').format_line() == 'Score: [number]'
    assert OutputField(
        'Score', kind='number', minimum=0.5, maximum=2
    ).placeholder() == ('number from 0.5 to 2')

    assert (
        OutputField('Blocked', kind='boolean').format_line()
        == 'Blocked: [true or false]'
    )
    assert OutputField('Blocked', kind='boolean').json_hint() == 'true/false'

    text = OutputField('Justification', kind='text', description='2-3 sentences')
    assert text.format_line() == 'Justification: [2-3 sentences]'
    assert OutputField('Notes', kind='text').format_line() == 'Notes: [free text]'

    with pytest.raises(ValueError, match='kind must be one of'):
        OutputField('X', kind='enum')


def test_rubric_entry_cleans_indicators():
    entry = RubricEntry(' 2 ', ' Partial ', ' desc ', ['  a  b ', '', None])
    assert (entry.value, entry.title, entry.description) == ('2', 'Partial', 'desc')
    assert entry.indicators == ['a b']


# --------------------------------------------------------------------- spec
def test_spec_from_dict_coerces_and_ignores_unknown_keys():
    spec = PromptSpec.from_dict(
        {
            'asset': ' road ',
            'imagery': 'Street',
            'steps': 'first\n\nsecond',
            'fields': [
                {'name': 'Blocked', 'kind': 'boolean', 'extra': 1},
                {'name': '', 'kind': 'text'},
                'not a dict',
            ],
            'rubric': [{'value': 'yes', 'indicators': ['x']}, {'value': ''}],
            'json_output': 1,
            'unknown': 'ignored',
        }
    )
    assert spec.asset == 'road' and spec.imagery == 'street'
    assert spec.steps == ['first', 'second']
    assert [f.name for f in spec.fields] == ['Blocked']
    assert [r.value for r in spec.rubric] == ['yes']
    assert spec.json_output is True
    assert PromptSpec.from_dict({'imagery': 'drone'}).imagery == 'aerial'
    assert PromptSpec.from_dict(None).asset == 'building'
    assert PromptSpec.from_dict({'asset': ''}).asset == 'building'
    with pytest.raises(ValueError, match='JSON object'):
        PromptSpec.from_dict(['x'])  # type: ignore[arg-type]


def test_spec_round_trip_primary_field_and_attribute_keys():
    spec = PromptSpec.from_dict(SAMPLE_SPEC.to_dict())
    assert spec == SAMPLE_SPEC
    assert spec.primary_field().name == 'CHS Level'
    assert spec.attribute_keys('gemma4') == ['gemma4_chs_level', 'gemma4_justification']
    assert spec.attribute_keys() == ['chs_level', 'justification']

    no_rubric_field = PromptSpec(
        fields=[OutputField('Notes', kind='text'), OutputField('Grade', options=['a'])]
    )
    assert no_rubric_field.primary_field().name == 'Grade'
    assert (
        PromptSpec(fields=[OutputField('Notes', kind='text')]).primary_field() is None
    )
    assert PromptSpec(rubric_field='Missing', fields=[OutputField('G', options=['a'])])
    assert (
        PromptSpec(rubric_field='Missing', fields=[OutputField('G', options=['a'])])
        .primary_field()
        .name
        == 'G'
    )


def test_objective_lead_in_is_stripped_and_roles_get_articles():
    from rapidtools.gui.prompt_builder import _role_phrase

    spec = PromptSpec(
        objective='Analyze each provided image and classify the vehicle condition.',
        persona='Senior Post-Disaster Damage Assessor',
        rubric_title='RUBRIC:',
    )
    assert spec.objective == 'classify the vehicle condition.'
    assert spec.rubric_title == 'RUBRIC'
    text = assemble_prompt(spec)
    assert text.startswith(
        'TASK: Analyze each provided image and classify the vehicle condition. '
        'Act as a Senior Post-Disaster Damage Assessor.'
    )
    street = PromptSpec(
        imagery='street',
        objective='analyse the provided street-level images to grade the facade',
    )
    assert street.objective == 'grade the facade'
    assert _role_phrase('an insurance adjuster.') == 'an insurance adjuster'
    assert _role_phrase('Expert Roofer') == 'an Expert Roofer'
    assert _role_phrase('forensic engineering') == 'an expert in forensic engineering'
    assert _role_phrase('structural engineers') == 'a structural engineers'


def test_default_steps_adapt_to_imagery_and_rubric():
    spec = PromptSpec(
        asset='vehicle', imagery='multi-temporal street', marking='marked'
    )
    steps = spec.default_steps()
    assert steps[0] == 'Locate the primary vehicle marked'
    assert 'each image' in steps[1]
    assert steps[-1] == 'Fill in every field of the required output format'

    with_rubric = PromptSpec(
        fields=[OutputField('Grade', options=['a', 'b'])],
        rubric=[RubricEntry('a', indicators=['x'])],
        rubric_title='GRADE INDEX',
    )
    steps = with_rubric.default_steps()
    assert steps[1] == 'Compare what you see against the GRADE INDEX indicators below'
    assert steps[2] == 'Assign the Grade with a brief justification'


# ----------------------------------------------------------------- assembly
def test_assemble_sample_matches_the_shipped_layout():
    text = assemble_prompt(SAMPLE_SPEC)
    blocks = text.split('\n\n')
    assert blocks[0].startswith(
        'TASK: Analyze each provided image and assign a CHS Level (0, 1, 2, 3, 4) to '
        'the primary building structure, based on the CHS Combustion Index provided. '
        'Do not focus your attention on non-combustible materials. Act as an expert '
        'in forensic engineering'
    )
    assert blocks[1].startswith(
        'ANALYSIS STEPS:\n1. Locate the primary building footprint'
    )
    assert blocks[2].startswith('ANALYSIS EDGE CASES:\nAlternative assigned')
    assert (
        'REQUIRED OUTPUT FORMAT:\nCHS Level: [0, 1, 2, 3, 4, No Data, '
        'No Data-Footprint Unclear]\n'
        'Justification: [2-3 sentences explaining key visual evidence]' in text
    )
    assert (
        'CHS COMBUSTION INDEX:\n\nCHS Level 0: Unaffected / No Direct Combustion'
        in text
    )
    assert 'Key indicators:\n- Roof is fully intact' in text
    assert text.endswith('\n') and 'CONTEXT AND BACKGROUND' not in text

    # The analyzer's parser understands the format a model would echo back.
    from rapidtools.processing.image_analyzers import parse_structured_results

    parsed = parse_structured_results('CHS Level: 3\nJustification: Roof gone.')
    assert set(parsed) == set(SAMPLE_SPEC.attribute_keys())


def test_assemble_json_output_persona_and_context():
    spec = PromptSpec(
        asset='vehicle',
        imagery='street',
        objective='classify the condition of the marked vehicle.',
        persona='forensic vehicle inspection',
        steps=['1. Find the car', '- Judge it', '2) Report'],
        fields=[
            OutputField('Condition', options=['intact', 'damaged']),
            OutputField('Blocked', kind='boolean'),
            OutputField('Confidence', kind='number', minimum=1, maximum=3),
            OutputField('Evidence', kind='text', description='one sentence'),
        ],
        context='  Vehicles are proxies for occupancy.  ',
        json_output=True,
    )
    text = assemble_prompt(spec)
    assert 'Analyze each provided street-level image and classify the condition' in text
    assert 'Act as an expert in forensic vehicle inspection.' in text
    assert 'ANALYSIS STEPS:\n1. Find the car\n2. Judge it\n3. Report' in text
    assert 'Return ONLY a valid JSON object' in text
    schema = text.split('outside it:\n', 1)[1].split('\n\nCONTEXT')[0]
    assert '"blocked": true/false' in schema
    parsed = json.loads(schema.replace('true/false', 'true'))
    assert parsed['condition'] == 'one of: intact, damaged'
    assert parsed['confidence'] == 'number from 1 to 3'
    assert parsed['evidence'] == 'one sentence'
    assert text.rstrip().endswith(
        'CONTEXT AND BACKGROUND:\nVehicles are proxies for occupancy.'
    )
    # No rubric means no rubric block:
    assert 'DEFINITIONS' not in text


def test_assemble_defaults_when_spec_is_sparse():
    text = assemble_prompt(PromptSpec(persona='an experienced roofer'))
    assert text.startswith(
        'TASK: Analyze each provided image and describe the condition of the primary '
        'building. Act as an experienced roofer.'
    )
    assert 'ANALYSIS STEPS:\n1. Locate the primary building\n2. Examine' in text
    assert 'REQUIRED OUTPUT FORMAT' not in text

    rubric_only = PromptSpec(
        fields=[OutputField('Grade', options=['a', 'b'])],
        rubric=[RubricEntry('a', 'Alpha'), RubricEntry('b')],
    )
    text = assemble_prompt(rubric_only)
    assert 'GRADE DEFINITIONS:\n\nGrade a: Alpha' in text
    assert 'Grade b' not in text  # undescribed classes are left out
    undescribed = PromptSpec(
        fields=[OutputField('Grade', options=['a'])], rubric=[RubricEntry('a')]
    )
    assert 'DEFINITIONS' not in assemble_prompt(undescribed)

    multi = PromptSpec(imagery='multi-temporal street')
    assert 'Analyze the provided series of street-level images' in assemble_prompt(
        multi
    )


# ---------------------------------------------------------------- assistant
class ScriptedModel:
    """Replays canned replies and records the prompts it received."""

    model_id = 'scripted'

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def run_inference(self, image_inputs, prompt, **kwargs):
        self.calls.append((image_inputs, prompt, kwargs))
        reply = self.replies.pop(0)
        if reply is None:
            return None
        return ModelOutput(text=reply)


def test_assistant_draft_parses_fenced_json_and_applies_context():
    reply = (
        'Here is the specification:\n```json\n'
        + json.dumps(
            {
                'asset': 'house',
                'imagery': 'aerial',
                'objective': 'grade roof damage',
                'fields': [
                    {
                        'name': 'Damage',
                        'kind': 'category',
                        'options': ['none', 'major'],
                    },
                    {'name': 'Justification', 'kind': 'text'},
                ],
                'rubric': [
                    {'value': 'major', 'title': 'Major', 'indicators': ['hole']}
                ],
                'json_output': False,
            }
        )
        + '\n```\nLet me know if you want changes.'
    )
    model = ScriptedModel(reply)
    assistant = PromptAssistant(model, temperature=0.1, max_tokens=99)
    spec = assistant.draft(
        'grade roofs',
        context={'marking': 'marked with a red outline', 'json_output': True},
    )
    assert spec.asset == 'house' and spec.marking == 'marked with a red outline'
    assert spec.json_output is True  # the GUI's setting wins over the model's
    assert [f.name for f in spec.fields] == ['Damage', 'Justification']
    images, prompt, kwargs = model.calls[0]
    assert images == [] and 'Request: grade roofs' in prompt
    assert kwargs == {'temperature': 0.1, 'max_tokens': 99, 'json_mode': True}

    with pytest.raises(ValueError, match='Describe what you want'):
        assistant.draft('   ')


def test_assistant_run_dispatches_every_action():
    spec = SAMPLE_SPEC.to_dict()
    revised = dict(spec, objective='revised objective')
    model = ScriptedModel(
        json.dumps(revised),  # refine
        '- Missing edge case for shadows.\nReady with small fixes.',  # review (spec)
        'Looks fine.',  # review (text)
        '["Roof missing", "Charred walls", "Debris inside footprint"]',  # indicators
        json.dumps(spec),  # import
        json.dumps(spec),  # draft
    )
    assistant = PromptAssistant(model)

    out = assistant.run('refine', spec=spec, instruction='change the objective')
    assert out['spec']['objective'] == 'revised objective'
    assert 'Current specification' in model.calls[0][1]

    out = assistant.run('review', spec=spec)
    assert out['text'].startswith('- Missing edge case')
    assert 'PROMPT:\nTASK: Analyze each provided image' in model.calls[1][1]
    assert model.calls[1][2]['json_mode'] is False

    assert assistant.run('review', prompt_text='Rate it.') == {'text': 'Looks fine.'}

    out = assistant.run('indicators', spec=spec, class_value='3')
    assert out['indicators'] == [
        'Roof missing',
        'Charred walls',
        'Debris inside footprint',
    ]
    assert 'Class to describe: 3 (Major Structural Combustion' in model.calls[3][1]

    out = assistant.run('import', prompt_text='TASK: something')
    assert out['spec']['asset'] == 'building structure'

    out = assistant.run('draft', brief='x', context={})
    assert out['spec']['rubric_title'] == 'CHS COMBUSTION INDEX'

    with pytest.raises(ValueError, match='Unknown assistant action'):
        assistant.run('translate')


def test_assistant_error_paths():
    assistant = PromptAssistant(ScriptedModel(None))
    with pytest.raises(ValueError, match='returned no text'):
        assistant.review('Rate it.')

    assistant = PromptAssistant(ScriptedModel('Sorry, I cannot help with that.'))
    with pytest.raises(ValueError, match='did not return a JSON specification'):
        assistant.import_prompt('TASK: x')

    assistant = PromptAssistant(ScriptedModel('{"not": "a list"}'))
    with pytest.raises(ValueError, match='list of indicators'):
        assistant.suggest_indicators(SAMPLE_SPEC, '2')

    # A dict wrapping the list is tolerated:
    assistant = PromptAssistant(ScriptedModel('{"indicators": ["a", " b "]}'))
    assert assistant.suggest_indicators(SAMPLE_SPEC, '2') == ['a', 'b']

    assistant = PromptAssistant(ScriptedModel('   '))
    with pytest.raises(ValueError, match='returned no text'):
        assistant.review('Rate it.')

    with pytest.raises(ValueError, match='Say what should change'):
        PromptAssistant(ScriptedModel()).refine(SAMPLE_SPEC, '')
    with pytest.raises(ValueError, match='no prompt to review'):
        PromptAssistant(ScriptedModel()).review('')
    with pytest.raises(ValueError, match='Pick a class'):
        PromptAssistant(ScriptedModel()).suggest_indicators(SAMPLE_SPEC, '')
    with pytest.raises(ValueError, match='Paste or write a prompt'):
        PromptAssistant(ScriptedModel()).import_prompt('')


def test_extract_json_handles_prose_and_trailing_text():
    from rapidtools.gui.prompt_builder import _extract_json

    assert _extract_json('Sure! {"a": 1} Hope this helps.') == {'a': 1}
    assert _extract_json('```\n[1, 2]\n```', want_list=True) == [1, 2]
    assert _extract_json('no json here') is None
    assert _extract_json('') is None
    # A list request falls back to an object when only an object is present:
    assert _extract_json('{"x": [1]}', want_list=True) == {'x': [1]}
