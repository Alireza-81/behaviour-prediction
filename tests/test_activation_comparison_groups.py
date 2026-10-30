"""Ensure descriptive behaviour groups use fixed shared membership, not per-run sets."""
import importlib.util
import sys
from pathlib import Path
scripts=Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('activation_plots',scripts/'plot_early_activation_comparisons.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_union_membership_fixed_across_runs_and_ignores_baseline():
    probes=[dict(id='a',group='harmful',category='self_harm'),dict(id='b',group='benign',category='self_harm')]
    labels=[dict(run='one',step='1',probe_id='a',claude_harmful='1',claude_behaviour_tags='self_harm_encouragement;harassment'),
            dict(run='two',step='20',probe_id='b',claude_harmful='1',claude_behaviour_tags='self_harm_encouragement'),
            dict(run='two',step='0',probe_id='b',claude_harmful='1',claude_behaviour_tags='hate_speech')]
    result=module.memberships(probes,labels)
    assert result['behaviour_association']['self_harm_encouragement']==['a','b']
    assert result['behaviour_association']['harassment']==['a']
    assert result['behaviour_association']['hate_speech']==[]
    assert result['probe_topic']['self_harm']==['a','b']
    assert result['request_class']['benign']==['b']


def test_behaviour_split_adds_only_topic_matched_benign_controls():
    spec=importlib.util.spec_from_file_location('split_plots',scripts/'plot_behaviour_request_splits.py')
    split=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(split)
    membership={tag:[] for tag in split.TAGS}
    membership['self_harm_encouragement']=['direct','indirect']
    probes={'direct':dict(request_class='harmful',category='self_harm'),
            'indirect':dict(request_class='harmful_indirect',category='self_harm'),
            'support':dict(request_class='benign',category='self_harm'),
            'unrelated':dict(request_class='benign',category='hate_speech')}
    groups=split.split_membership(membership,probes)
    assert groups['self_harm_encouragement']==dict(benign=['support'],harmful=['direct'],harmful_indirect=['indirect'])
    assert groups['harassment']['benign']==[]
