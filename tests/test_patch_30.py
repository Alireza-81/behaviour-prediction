import importlib.util
import json
from pathlib import Path
from collections import Counter


def test_fresh_patch30_prompt_counts_and_no_exact_reuse():
    manifest=json.loads(Path('configs/activation_patch_30.json').read_text())
    probes=manifest['probes']
    assert len(probes)==30 and len({p['id'] for p in probes})==30
    assert Counter(p['group'] for p in probes)==dict(harmful=13,harmful_indirect=12,benign=5)
    assert sum(p['role']=='target' for p in probes)==25
    old=[]
    for path in ('data/probes/online_probes.jsonl','data/evaluation/prompts.jsonl'):
        old.extend(json.loads(l)['prompt'] for l in Path(path).read_text().splitlines() if l.strip())
    normalize=lambda s:' '.join(s.lower().split())
    assert not {normalize(p['prompt']) for p in probes}&{normalize(p) for p in old}
    assert len({normalize(p['prompt']) for p in probes})==30
    assert len(probes)*len(manifest['conditions'])==180


def test_expanded_completed_journal_resumes_without_loading_models(tmp_path,monkeypatch,capsys):
    path=Path(__file__).parents[1]/'scripts/run_activation_patch_pilot.py'
    spec=importlib.util.spec_from_file_location('patch_runner',path)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    manifest=json.loads(Path('configs/activation_patch_30.json').read_text())
    (tmp_path/'pilot.json').write_text(json.dumps(manifest))
    rows=[dict(probe_id=p['id'],condition=c,pilot_id=manifest['pilot_id']) for p in manifest['probes'] for c in manifest['conditions']]
    (tmp_path/'generated.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    monkeypatch.setattr(runner,'export_reviews',lambda *args:None)
    def fail(*args,**kwargs):raise AssertionError('Must not load completed checkpoints')
    monkeypatch.setattr(runner,'checkpoint_plan',fail)
    runner.run(tmp_path)
    assert 'All 180 answers already saved' in capsys.readouterr().out
