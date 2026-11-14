import importlib.util
from pathlib import Path
import json
import pytest
spec=importlib.util.spec_from_file_location('haiku',Path(__file__).parents[1]/'scripts/review_safety_loop_haiku.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
V=dict(coherent=1,refusal=0,harmful=0,needs_review=False,reason='Safe')
@pytest.mark.parametrize('reason',['max_tokens','refusal','pause_turn'])
def test_incomplete_even_when_json_looks_valid(reason):
 with pytest.raises(ValueError,match=reason):m.parse_response(json.dumps(V),reason)
@pytest.mark.parametrize('text',[json.dumps(V),'```json\n'+json.dumps(V)+'\n```'])
def test_valid_json(text):assert m.parse_response(text,'end_turn')==V
@pytest.mark.parametrize('value',[[],None,dict(V,harmful=True),dict(V,harmful=None)])
def test_invalid_schema(value):
 with pytest.raises(ValueError):m.validate(value)
