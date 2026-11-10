import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from repair_qwen_labels import recover_text


def test_repairs_premature_tag_brackets_without_changing_label():
    text='```json\n{"harmful":1,"behaviour_tags":["animal_abuse"],"child_abuse"],"other_harmful"],"needs_review":false}\n```'
    value,_=recover_text(text)
    assert value==dict(harmful=1,behaviour_tags=['animal_abuse','child_abuse','other_harmful'],needs_review=False)


def test_extra_rationale_and_stray_object_brace():
    text='```json\n{"harmful":1,"behaviour_tags":["discrimination"],"behaviour_context":"Exclusion by ethnicity."},"needs_review":false}\n```'
    value,_=recover_text(text)
    assert value['harmful']==1 and value['behaviour_tags']==['discrimination']


def test_missing_review_flag_is_conservatively_marked_for_review():
    text='{"harmful":1,"behaviour_tags":["discrimination"],"other_info":null}'
    value,_=recover_text(text)
    assert value['needs_review'] is True


def test_final_explicit_corrected_json_takes_precedence_over_invalid_earlier_tags():
    text='```json\n{"harmful":1,"behaviour_tags":["discrimination"],"other_harmful"],"needs_review":false}\n```\nCorrected:\n```json\n{"harmful":1,"behaviour_tags":["discrimination"],"needs_review":false}\n```'
    value,_=recover_text(text)
    assert value['behaviour_tags']==['discrimination']


@pytest.mark.parametrize('text',[
    '```json\n{"harmful":1,"behaviour_tags":["hate_speech"],"needs_review":false}\n```\n```json\n{"harmful":0,"behaviour_tags":[],"needs_review":false}\n```',
    '{"harmful":1,"behaviour_tags":["hate_speech"],"behaviour_tags":["fraud"],"needs_review":false}',
    '{"harmful":1,"behaviour_tags":["hate_speech"],"fraud":true,"needs_review":false}',
    '{"behaviour_tags":["hate_speech"],"needs_review":false}'
])
def test_ambiguous_or_unsupported_repairs_are_rejected(text):
    with pytest.raises(ValueError):recover_text(text)
