import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import audit_multilayer_label_reuse as audit
import reuse_multilayer_activation_patch as reuse
from label_with_claude import digest


def test_source_audit_checks_real_csvs_and_conflicts():
    b,lookup,report=audit.audit_source(reuse.BUNDLE,verify_files=True)
    assert report['original_source_files_verified']==25
    assert len(b['labels'])==990 and len(lookup)==622


def target_fixture(tmp_path):
    b=reuse.load_bundle();old=b['source_manifest']
    source=Path('outputs/activation-patch-shared/generated.jsonl')
    selected={'chosen':{'31':next(r['selected_indices']['31'] for r in b['answers'] if r['condition']=='shared_05pct')}}
    artifact=dict(selected_indices=selected);artifact['mask_id']=audit.checksum(artifact,'mask_id')
    m=dict(old,conditions=['unpatched','self_dynamic','full_dynamic','chosen'],mask_id=artifact['mask_id'])
    m['pilot_id']=audit.checksum(m,'pilot_id')
    rows=[]
    for original in audit.read_jsonl(source):
        c=original['condition']
        if c not in ['unpatched','self_dynamic','full_dynamic','shared_05pct']:continue
        c='chosen' if c=='shared_05pct' else c
        r=dict(original,condition=c,pilot_id=m['pilot_id'],mask_id=m['mask_id'],selected_indices=selected.get(c,{}),
            run=f"{m['run_name']}_multilayer_{original['direction']}__{c}")
        rows.append(r)
    (tmp_path/'mask.json').write_text(json.dumps(artifact));(tmp_path/'pilot.json').write_text(json.dumps(m))
    (tmp_path/'generated.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    judge=dict(model=b['judge_model'],rubric=b['judge_rubric'],sample=rows,all_responses=True,description='Unicode audit café')
    judge['test_id']=digest(judge)
    folder=tmp_path/'labeling';folder.mkdir();(folder/'test.json').write_text(json.dumps(judge))
    assert reuse.reuse_labels(tmp_path)==360
    return folder


def test_independent_audit_accepts_valid_reuse_and_rejects_wrong_label(tmp_path):
    folder=target_fixture(tmp_path)
    report=audit.audit_target(tmp_path,reuse.BUNDLE)
    assert report['passed'] and report['imported_rows_verified']==360 and report['remaining']==0
    results=audit.read_jsonl(folder/'results.jsonl')
    results[0]['judgment']['harmful']=1-results[0]['judgment']['harmful']
    results[0]['judgment']['behaviour_tags']=['hate_speech'] if results[0]['judgment']['harmful'] else []
    (folder/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    with pytest.raises(ValueError,match='judgment differs'):audit.audit_target(tmp_path,reuse.BUNDLE)


def test_audit_rejects_changed_labeling_payload(tmp_path):
    folder=target_fixture(tmp_path)
    judge=json.loads((folder/'test.json').read_text())
    judge['sample'][0]['prompt']+=' changed'
    judge['test_id']=digest({k:v for k,v in judge.items() if k!='test_id'})
    (folder/'test.json').write_text(json.dumps(judge))
    with pytest.raises(ValueError,match='Judge sample differs'):audit.audit_target(tmp_path,reuse.BUNDLE)
