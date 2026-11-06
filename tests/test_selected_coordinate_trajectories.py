import csv,json,sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from plot_selected_coordinate_trajectories import coordinate_subsets,measurements,mean_metrics,plot
from behaviour_prediction.monitor.shared_mask import fixed_masks,identity


def test_fixed_subsets_are_equal_size_stable_and_halves_partition_layer():
    before=torch.random.get_rng_state().clone()
    scores=torch.arange(4096,dtype=torch.float64)-2000
    subsets,ranking=coordinate_subsets(scores,[102,205,410,2048],[17,42,123])
    assert ranking==list(reversed(range(4096)))
    for budget,groups in subsets.items():
        assert all(len(set(values))==int(budget) for values in groups.values())
    assert not set(subsets['2048']['top'])&set(subsets['2048']['bottom'])
    assert set(subsets['2048']['top'])|set(subsets['2048']['bottom'])==set(range(4096))
    assert set(subsets['102']['top'])<=set(subsets['205']['top'])<=set(subsets['410']['top'])
    assert (scores[subsets['102']['bottom']]<0).all() # Lowest signed score is not zero importance.
    assert torch.equal(before,torch.random.get_rng_state())
    assert subsets==coordinate_subsets(scores,[102,205,410,2048],[17,42,123])[0]


def test_measure_probe_norms_before_averaging_and_keep_sign_cancellation_visible():
    a=torch.ones((2,2));delta=torch.tensor([[3.,-3.],[4.,4.]])
    values=measurements(a,a+delta,[0,1])
    assert values['rms_delta'].tolist()==[3.,4.]
    assert values['signed_mean_delta'].tolist()==[0.,4.]
    assert values['delta_energy_fraction'].tolist()==[1.,1.]
    rows=[{key:float(v[i]) for key,v in values.items()} for i in range(2)]
    assert mean_metrics(rows)['rms_delta']==3.5
    assert float(delta.mean(0).square().mean().sqrt())==2.5
    assert all(torch.equal(v,torch.zeros_like(v)) for v in measurements(a,a,[0,1]).values())
    # Duplicating coordinates cannot increase the per-coordinate RMS.
    repeated=measurements(a.repeat(1,2),(a+delta).repeat(1,2),[0,1,2,3])
    torch.testing.assert_close(values['rms_delta'],repeated['rms_delta'])


def test_saved_snapshot_pipeline_validates_and_exports_signed_and_class_views(tmp_path):
    root=tmp_path/'snapshots';output=tmp_path/'plots';runs=['a','b'];steps=[0,1]
    scores=torch.arange(4096,dtype=torch.float64)-2000
    mask=dict(layer=31,width=4096,aggregate_scores=scores.tolist(),budgets_percent=[5,25],random_seeds=[17,42,123],
        selected_indices=fixed_masks(scores,[5,25],[17,42,123]),expected_model_config={})
    mask['mask_id']=identity(mask);mask_path=tmp_path/'mask.json';mask_path.write_text(json.dumps(mask))
    probes=[dict(id=f'p{i}',group=['harmful','harmful_indirect','benign'][i//10],category='test',prompt=f'Prompt{i}',token_count=5) for i in range(30)]
    labeldir=root/'labeling-monitoring/labeled';labeldir.mkdir(parents=True)
    for run in runs:
        directory=root/run/'activation_monitoring';directory.mkdir(parents=True)
        (root/run/'run_metadata.json').write_text(json.dumps(dict(model={})))
        labels=[]
        for step in steps:
            base=torch.ones((30,4096));delta=torch.arange(4096)[None,:].float()/1000
            current=base+step*delta
            torch.save(dict(metadata=dict(pooling='last_non_padding',probes=probes),activations={'layer_31':current}),directory/f'step-{step:06d}.pt')
            labels.extend(dict(run=run,step=step,probe_id=p['id'],claude_harmful=int(run=='a' and step==1 and p['id']=='p0')) for p in probes)
        with (labeldir/f'{run}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(labels[0]));w.writeheader();w.writerows(labels)
    plot(root,output,mask_path,budgets=[205],runs=runs,steps=steps)
    protocol=json.loads((output/'protocol.json').read_text())
    assert protocol['onsets']=={'a':1,'b':None} and all(protocol['baseline_bit_exact_across_runs'].values())
    assert protocol['raw_rows']==2*2*30*6 and protocol['figures']==7
    rows=list(csv.DictReader((output/'grouped_trajectories.csv').open()))
    assert {r['grouping'] for r in rows}=={'all','request_class','probe_topic'}
    def point(subset):return next(float(r['rms_delta']) for r in rows if r['run']=='a' and r['step']=='1' and r['grouping']=='all' and r['subset']==subset)
    assert point('top')>point('middle')>point('bottom')
    with pytest.raises(ValueError,match='Output exists'):plot(root,output,mask_path,budgets=[205],runs=runs,steps=steps)
