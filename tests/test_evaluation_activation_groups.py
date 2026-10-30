import importlib.util
import sys
from pathlib import Path
import json
scripts=Path(__file__).parents[1]/'scripts';sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('evaluation_activation_plots',scripts/'plot_evaluation_activation_comparison.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_actual_evaluation_groups_not_output_selected():
    probes=[json.loads(l) for l in Path('data/evaluation/prompts.jsonl').read_text().splitlines() if l.strip()]
    for category,benign in [('harassment',2),('violent_threats',1)]:
        groups=module.groups_for(probes,category)
        assert len(groups['direct'])==2
        assert len(groups['indirect'])==2
        assert len(groups['benign_same_topic'])==benign
        assert len(groups['benign_all_topics'])==10
        assert not set(groups['direct'])&set(groups['benign_all_topics'])
        assert all(probes[i]['id'].startswith('eval_') for ids in groups.values() for i in ids)
