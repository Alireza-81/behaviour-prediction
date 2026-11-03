"""Freeze a small baseline-to-step20 patch pilot and visualize stored activation values."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from behaviour_prediction.trajectories import load

ROOT=Path('outputs/early-steps')
OUT=Path('outputs/activation-patch-pilot')
TARGETS=['probe_hate_speech_direct_01','probe_hate_speech_direct_02','probe_hate_speech_indirect_01']
CONTROLS=['probe_hate_speech_response_benign_01','probe_hate_speech_response_benign_02','probe_fact_checking_benign_01']


def prepare():
    if OUT.exists() and any(OUT.iterdir()):raise ValueError('Pilot directory already exists')
    run=ROOT/'hate_speech'
    labels_path=ROOT/'labeling-monitoring/labeled/hate_speech.csv'
    labels={(int(r['step']),r['probe_id']):r for r in csv.DictReader(labels_path.open())}
    base=load(run/'activation_monitoring/step-000000.pt')
    tuned=load(run/'activation_monitoring/step-000020.pt')
    assert base['metadata']['probes']==tuned['metadata']['probes']
    probes={p['id']:p for p in base['metadata']['probes']}
    selected=[]
    for role,ids in [('target',TARGETS),('benign_control',CONTROLS)]:
        for id in ids:
            before,after=labels[(0,id)],labels[(20,id)]
            assert before['claude_harmful']=='0'
            assert after['claude_harmful']==('1' if role=='target' else '0')
            if role=='target':assert 'hate_speech' in after['claude_behaviour_tags'].split(';')
            selected.append(dict(**{k:probes[id][k] for k in ('id','prompt','group','category')},role=role,
                reference_step0_response=before['response'],reference_step20_response=after['response']))
    manifest=dict(source_run=str(run),run_name=run.name,step=20,layers=[15,31],
        max_input_length=256,max_new_tokens=256,probes=selected,
        conditions=['baseline_step0','unpatched_step20','self_patch_both','patch_layer15','patch_layer31','patch_both'],
        patch_scope='Entire residual output vector at final prompt token, initial prefill only; no patching of subsequent generated tokens',
        selection='Three 0-to1 hate-speech outputs, plus three 0-to0 benign controls; development pilot, not held-out evaluation',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [labels_path,run/'run_metadata.json',run/'activation_monitoring/step-000000.pt',run/'activation_monitoring/step-000020.pt']})
    manifest['pilot_id']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    OUT.mkdir(parents=True)
    (OUT/'pilot.json').write_text(json.dumps(manifest,indent=2)+'\n')
    Path('configs/activation_patch_pilot.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (OUT/'probes.jsonl').write_text(''.join(json.dumps({k:p[k] for k in ('id','prompt','group','category')})+'\n' for p in selected))
    data=[]
    arrays={}
    ordered=[p['id'] for p in base['metadata']['probes']]
    for layer in [15,31]:
        key=f'layer_{layer:02d}'
        a=base['activations'][key].float().numpy()
        b=tuned['activations'][key].float().numpy()
        arrays[layer]=(a,b)
        for p in selected:
            idx=ordered.index(p['id'])
            for d in range(a.shape[1]):
                data.append(dict(layer=layer,probe_id=p['id'],role=p['role'],coordinate=d,baseline=float(a[idx,d]),finetuned=float(b[idx,d]),delta=float(b[idx,d]-a[idx,d])))
    with (OUT/'activation_coordinates.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    # An honest layout diagram: each highlighted box is a full 4096-value vector,
    # not an individually identified harmful neuron.
    fig,ax=plt.subplots(figsize=(14,5));ax.set(xlim=(0,14),ylim=(0,5));ax.axis('off')
    boxes=[(.2,'Same prompt'),(2.1,'Blocks 0–14'),(4.1,'Layer 15'),(6.1,'Blocks 16–30'),(8.3,'Layer 31'),(10.6,'Final norm / head')]
    for y,label in [(3.6,'Step 0 donor'),(1.3,'Step 20 recipient')]:
        ax.text(.2,y+.65,label,fontsize=12,weight='bold')
        for i,(x,name) in enumerate(boxes):
            color='#ffe5ad' if name in ('Layer 15','Layer 31') else '#e8eef5'
            ax.add_patch(FancyBboxPatch((x,y-.25),1.7,.6,boxstyle='round,pad=.05',facecolor=color,edgecolor='#555555'))
            ax.text(x+.85,y+.05,name,ha='center',va='center',fontsize=10)
            if i<len(boxes)-1:ax.annotate('',xy=(boxes[i+1][0]-.08,y+.05),xytext=(x+1.75,y+.05),arrowprops=dict(arrowstyle='->'))
    for x in [4.95,9.15]:
        ax.annotate('',xy=(x,1.75),xytext=(x,3.25),arrowprops=dict(arrowstyle='->',lw=2.5,color='#ce7900'))
        ax.text(x+.1,2.5,'Copy all 4096 values\nat last prompt token',fontsize=9)
    ax.text(.2,.4,'Test layer 15, layer 31, and both. Rest of the recipient remains step 20. Later decode positions are not overwritten.',fontsize=11)
    fig.suptitle('Where the first patch will happen — two measured layer outputs, not a discovered circuit')
    fig.tight_layout();fig.savefig(OUT/'patch_locations.png',dpi=170);plt.close(fig)
    # Full coordinates shown as 64 x 64 tiles solely for display; the raster has
    # no spatial/neuron-neighbourhood meaning.
    primary=TARGETS[0];idx=ordered.index(primary)
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for row,layer in enumerate([15,31]):
        a,b=arrays[layer];width=a.shape[1];assert width==4096
        values=[a[idx],b[idx],b[idx]-a[idx]]
        shared=max(np.abs(a[idx]).max(),np.abs(b[idx]).max())
        for col,(v,title) in enumerate(zip(values,['Step 0','Step 20','Difference (20 minus 0)'])):
            limit=shared if col<2 else max(np.abs(v).max(),1e-8)
            im=axes[row,col].imshow(v.reshape(64,64),cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
            axes[row,col].set(title=f'Layer {layer} · {title}',xlabel='Coordinate offset within row',ylabel='Coordinate block (64 values)')
            fig.colorbar(im,ax=axes[row,col],shrink=.8)
    fig.suptitle(f'{primary}: all 4096 signed activation values per layer\nTile coordinate = 64 × row + column; layout is a display convention, not physical brain-like locations')
    fig.tight_layout();fig.savefig(OUT/'activation_values.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    summary=[]
    for row,layer in enumerate([15,31]):
        a,b=arrays[layer]
        for p in selected:
            i=ordered.index(p['id']);delta=b[i]-a[i]
            axes[row,0].hist(delta,bins=80,histtype='step',density=True,label=p['id'].removeprefix('probe_'),alpha=.9)
            summary.append(dict(layer=layer,probe_id=p['id'],role=p['role'],l2=float(np.linalg.norm(delta)),mean_abs=float(np.abs(delta).mean())))
        points=[r for r in summary if r['layer']==layer]
        axes[row,1].bar(range(6),[r['l2'] for r in points],color=['#c44e52' if r['role']=='target' else '#55a868' for r in points])
        axes[row,1].set_xticks(range(6),[r['probe_id'].removeprefix('probe_') for r in points],rotation=30,ha='right',fontsize=7)
        axes[row,0].set(title=f'Layer {layer}: distribution across 4096 coordinates',xlabel='Signed activation change',ylabel='Density')
        axes[row,1].set(title=f'Layer {layer}: overall movement',ylabel='L2 drift from step 0')
        axes[row,0].legend(fontsize=6)
    fig.suptitle('Descriptive changes only: larger change does not establish causal importance')
    fig.tight_layout();fig.savefig(OUT/'activation_distributions.png',dpi=170);plt.close(fig)
    with (OUT/'activation_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    (OUT/'README.md').write_text('''# First activation patching pilot

This is preparation and visualization only. No patched language-model responses have been generated locally.

Source model: hate_speech fine-tuning run, step 20. Three prompts with Claude labels non-harmful at step zero and hate speech at step 20; three benign controls remain non-harmful at both endpoints. The IDs and original responses are frozen in pilot.json. This is a selected development pilot, not an unbiased accuracy experiment.

The first interventions replace **all 4096 coordinates** at the last prompt token in the output of decoder block 15, block 31, or both, using the SAME tokenized prompt in the step-zero model. Coordinates are residual-stream features, not necessarily individual neurons or interpretable concepts. The 64×64 display has no spatial meaning.

Selected locations are the two layers already measured, not saliency-ranked discoveries. Current heatmaps show signed values and differences, not harmfulness scores. Do not select “harmful neurons” merely because their absolute changes are large.

Conditions: baseline step zero; unpatched step 20; same-model self-patch (no-op control); baseline patch at layer 15, layer 31, or both. Six prompts × six conditions = 36 answers. Self-patch must match the unpatched greedy token sequence exactly. Layer 31 is the final block, so replacing it can directly restore baseline next-token logits; success there alone does not localize a specific harmful circuit.

Scope: only the initial prompt prefill is patched. Generated-token positions are not patched; earlier recipient KV-cache entries remain. A negative result does not establish that a component is irrelevant, and any effect can be mediated through later generation. This pilot does not change model weights or train a monitor.

After syncing code and this folder to the GPU, run:

```bash
python scripts/run_activation_patch_pilot.py --pilot-dir outputs/activation-patch-pilot
```

The run uses checkpoint-20 from outputs/early-steps/hate_speech and the shared base model. It does not train, call Claude, or save another model checkpoint. Same command resumes completed probe/condition pairs.

Outputs: generated.jsonl (full experiment records and token IDs), reviews/*.csv (one per condition, compatible with the existing labeling prepare script), and run_report.json. Original labels are selection references, not labels on new patched answers. Label the new answers before claiming the intervention reduces harm; judge target behaviour, coherence and benign usefulness, and avoid treating generic refusal or broken text as a selective repair.
''')
    print(f'Prepared {len(selected)} prompts, 36 planned answers, and 3 figures in {OUT}')


if __name__=='__main__':prepare()
