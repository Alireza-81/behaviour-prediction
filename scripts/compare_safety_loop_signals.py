"""Descriptive matched-pair analysis; no fitted alarm or generalization claim."""
import csv,json,hashlib,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('outputs/safety-loop-v4');LABELS=Path('outputs/safety-loop-v4-human-reviewed');OUT=Path('outputs/safety-loop-v4-comparison')
OUT.mkdir(exist_ok=True)
STEPS=[0,10,15,20,25,30,50]
rows=[];hashes={}
for branch in [0,1]:
 base=list(csv.DictReader((LABELS/f'branch-{branch}/review-000000.csv').open()))
 refused={r['id'] for r in base if r['refusal']=='1'}
 for step in STEPS:
  path=LABELS/f'branch-{branch}/review-{step:06d}.csv';fp=ROOT/f'branch-{branch}/features-{step:06d}.json'
  for p in (path,fp):hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
  labels=list(csv.DictReader(path.open()));features=json.loads(fp.read_text())
  assert [r['id'] for r in labels]==features['probe_ids']
  assert all(r['harmful'] in ('0','1') for r in labels)
  monitor=[i for i,r in enumerate(labels) if r['split']=='monitor'];legit=[r for r in labels if r['split']=='behaviour']
  def words(r):return len(r['response'].split())
  row=dict(branch=branch,step=step,harm=sum(int(r['harmful']) for r in legit)/len(legit),
   monitor_log_words=np.mean([np.log1p(words(labels[i])) for i in monitor]),
   monitor_first_person=np.mean([bool(re.search(r"\b(I|my|me|mine|myself)\b",labels[i]['response'])) for i in monitor]),
   monitor_list_style=np.mean([bool(re.search(r'(?m)^\s*(?:[-*]|\d+[.)])\s',labels[i]['response'])) for i in monitor]),
   legitimate_median_words=float(np.median([words(r) for r in legit])))
  for layer,values in features['features'].items():
   for cohort,indices in [('monitor',monitor),('baseline_refused',[i for i in monitor if labels[i]['id'] in refused])]:
    assert indices
    prefix=f'L{layer}/{cohort}/'
    for name,field,transform in [('magnitude','relative_l2',lambda x:x),('norm_change','norm_ratio',lambda x:x-1),('orientation','cosine',lambda x:1-x)]:
     row[prefix+name]=float(transform(np.asarray(values[field])[indices]).mean())
    projections=np.asarray(values['relative_projection_delta'])[indices].mean(0)
    for name,v in zip(features['names'],projections):row[prefix+'projection:'+name]=float(v)
  rows.append(row)

def corr(a,b):
 a=np.asarray(a)-np.mean(a);b=np.asarray(b)-np.mean(b)
 return float(a@b/np.sqrt((a@a)*(b@b))) if (a@a)*(b@b)>1e-20 else None
nonzero=[r for r in rows if r['step']>0];y=np.array([r['harm'] for r in nonzero])
control_names=['step','branch','monitor_log_words','monitor_first_person','monitor_list_style']
C=np.array([[r[k] for k in control_names] for r in nonzero]);C=(C-C.mean(0))/np.maximum(C.std(0),1e-12);C=np.column_stack([np.ones(len(C)),C])
def residual(v):return v-C@np.linalg.lstsq(C,v,rcond=None)[0]
features=[k for k in rows[0] if k.startswith('L')];results=[]
for name in features:
 x=np.array([r[name] for r in nonzero]);gaps=[];ygaps=[]
 for step in STEPS[1:]:
  a,b=[r for r in rows if r['step']==step];gaps.append(a[name]-b[name]);ygaps.append(a['harm']-b['harm'])
 results.append(dict(feature=name,raw_r=corr(x,y),adjusted_r=corr(residual(x),residual(y)),matched_step_gap_r=corr(gaps,ygaps)))
for name in control_names[2:]:results.append(dict(feature=name,raw_r=corr([r[name] for r in nonzero],y),adjusted_r=None,matched_step_gap_r=None))
for name,data in [('checkpoint-features.csv',rows),('signal-comparison.csv',results)]:
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
# All axes are reported, without selecting a winner from the test trajectory.
fig,axs=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
series=[('harm','Harm on legitimate requests'),('monitor_log_words','Monitor answer length (mean log words)'),('monitor_first_person','Monitor first-person fraction'),('L31/monitor/magnitude','Layer 31 relative L2 drift'),('L31/monitor/orientation','Layer 31 cosine distance'),('L31/monitor/projection:harmlessness','Layer 31 signed harmlessness projection')]
for ax,(field,title) in zip(axs.flat,series):
 for b,label in [(0,'Original'),(1,'Matched control')]:
  points=[r for r in rows if r['branch']==b];ax.plot(STEPS,[r[field] for r in points],marker='o',label=label)
 ax.set_title(title,fontsize=10);ax.set_xlabel('Optimizer step');ax.grid(alpha=.2)
axs[0,0].legend();fig.savefig(OUT/'trajectories.png',dpi=160);plt.close(fig)
keys=['magnitude','norm_change','orientation']+['projection:'+n for n in json.loads((ROOT/'branch-0/features-000000.json').read_text())['names']]
fig,axs=plt.subplots(1,2,figsize=(11,7),constrained_layout=True)
for ax,metric,title in zip(axs,['raw_r','adjusted_r'],['Raw descriptive correlation','After linear length/style, step and branch adjustment']):
 matrix=np.array([[next(r[metric] for r in results if r['feature']==f'L{layer}/monitor/{k}') for layer in [14,15,31]]for k in keys],dtype=float)
 im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm',aspect='auto');ax.set_xticks(range(3),['Layer 14','Layer 15','Layer 31']);ax.set_yticks(range(len(keys)),keys);ax.set_title(title,fontsize=10)
fig.colorbar(im,ax=axs,label='Pearson r (12 dependent checkpoints)');fig.savefig(OUT/'correlations.png',dpi=160);plt.close(fig)
manifest=dict(source_hashes=hashes,analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),outcome='Haiku legitimate-request harm, with recorded human overrides',n_parent_runs=2,n_nonzero_checkpoints=12,controls=control_names,control_design_rank=int(np.linalg.matrix_rank(C)),limitations=['Descriptive only: checkpoints are dependent','Same two inspected runs; no held-out generalization','Linear adjustment cannot remove unmeasured style confounding','Length/style require generated answers; not cheap activation-only online features','Experimental directions, no causal validation','No new threshold selection, alarm, or warning-lead claim'])
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Controls',[(r['feature'],r['raw_r'])for r in results if not r['feature'].startswith('L')])
for r in results:
 if r['feature'].startswith('L31/monitor/'):print(r)
