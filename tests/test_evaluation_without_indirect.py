import importlib.util
import sys
from pathlib import Path
scripts=Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('without_indirect',scripts/'evaluate_without_indirect.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_removed_indirect_harm_does_not_establish_presence():
    rows=[]
    for group in ('harmful','benign','harmful_indirect'):
        for i in range(10):
            rows.append(dict(group=group,probe_id=f'{group}_{i}',claude_harmful='1' if group=='harmful_indirect' else '0',claude_behaviour_tags='harassment' if group=='harmful_indirect' else ''))
    result=module.outcomes_without_indirect(rows)
    assert result['label']==0 and result['tags']['harassment']==0
    rows[0].update(claude_harmful='1',claude_behaviour_tags='violent_threats')
    result=module.outcomes_without_indirect(rows)
    assert result['label']==1 and result['tags']['violent_threats']==1
    assert result['tags']['harassment']==0
