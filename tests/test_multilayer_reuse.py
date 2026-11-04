import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import reuse_multilayer_activation_patch as reuse
from label_with_claude import digest, payload


def test_exact_mask_answers_are_reused_and_other_masks_are_not(tmp_path):
    b=reuse.load_bundle();old=b['source_manifest']
    selected={}
    for r in b['answers']:
        c=r['condition']
        if c=='unpatched':continue
        name='layer31_205' if c=='shared_05pct' else f"layer31_205_random{c.split('_')[0].removeprefix('random')}"
        selected[name]=r['selected_indices']
    m=dict(old,pilot_id='new',mask_id='new-mask',conditions=['unpatched',*selected])
    records={}
    assert reuse.seed_answers(tmp_path,m,selected,records)==450
    assert reuse.seed_answers(tmp_path,m,selected,records)==0
    assert len((tmp_path/'generated.jsonl').read_text().splitlines())==450
    assert all(r['pilot_id']=='new' and r['mask_id']=='new-mask' and 'reused_from' in r for r in records.values())
    # An equally sized but different mask must never inherit a generated answer.
    selected['layer31_205']={'31':list(range(205))}
    second=tmp_path/'different';second.mkdir();records={}
    assert reuse.seed_answers(second,m,selected,records)==360
    assert all(r['condition']!='layer31_205' for r in records.values())
    m['max_new_tokens']+=1
    with pytest.raises(ValueError,match='mismatch'):reuse.seed_answers(second,m,selected,{})


def test_labels_match_full_judge_payload_and_skip_completed_rows(tmp_path):
    b=reuse.load_bundle();source=b['answers'][0]
    def sample(name):
        return dict(run=name,step='20',probe_id=name,prompt=source['prompt'],response=source['response'],
            token_limit_reached=str(source['token_limit_reached']),human_harmful='',review_notes='',
            group=source['group'],category=source['category'])
    rows=[sample('exact'),dict(sample('new'),response='Never generated before'),
          dict(sample('different-truncation'),token_limit_reached=str(not source['token_limit_reached']))]
    m=dict(sample=rows,model=b['judge_model'],rubric=b['judge_rubric'],all_responses=True)
    m['test_id']=digest(m)
    labeling=tmp_path/'labeling';labeling.mkdir();(labeling/'test.json').write_text(json.dumps(m))
    assert reuse.reuse_labels(tmp_path)==1
    result=json.loads((labeling/'results.jsonl').read_text())
    assert result['index']==0 and result['cost_usd']==0 and result['reused_from']['reuse_id']==b['reuse_id']
    assert result['judgment']['harmful'] in [0,1]
    assert reuse.reuse_labels(tmp_path)==0
    m['rubric']+='changed';m['test_id']=digest({k:v for k,v in m.items() if k!='test_id'})
    (labeling/'test.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='rubric'):reuse.reuse_labels(tmp_path)
