"""Independently reconstruct checkpoint logistic scores and whole-run splits."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from behaviour_prediction.monitor.features import extract_runs
from run_early_step_experiments import metric
import run_qwen_full_experiments as legacy


def main(output):
    protocol=json.loads((output/'protocol.json').read_text())
    for path,h in protocol['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
    paths=[Path(path).parent for path in protocol['source_sha256'] if path.endswith('run_metadata.json')]
    data=extract_runs(paths)
    rows={(r['run'],r['step']):r for r in data['rows']}
    models=json.loads((output/'models.json').read_text())
    predictions=json.loads((output/'predictions.json').read_text())
    count=0
    for model in models:
        method=model['method']
        keys=[tuple(k) for k in model['train_keys']]
        assert all(k[0]!=model['held_out'] and k[1]!=0 for k in keys)
        def vector(key):
            values=rows[key]['features']
            if method.startswith('summary_'):return [values[n] for n in protocol['layer_features'][method]]
            if method=='step_only':return [key[1]]
            return [sum(values[f'layer_31/{g}/l2_mean'] for g in ['harmful','harmful_indirect','benign'])/3]
        train=torch.tensor([vector(k) for k in keys],dtype=torch.float64)
        params=model['parameters']
        if params:
            assert params['mean']==train.mean(0).tolist()
            assert params['scale']==train.std(0,unbiased=False).clamp_min(1e-6).tolist()
        for p in predictions:
            if (p['method'],p['task'],p['tag'],p['run'])!=(method,model['task'],model['tag'],model['held_out']):continue
            x=torch.tensor([vector((p['run'],p['input_step']))],dtype=torch.float64)
            score=legacy.predict_parameters(x,params,model['prior'])[0]
            assert abs(score-p['score'])<1e-10
            assert int(score>=.5)==p['predicted_label']
            count+=1
    report=json.loads((output/'report.json').read_text())
    for task,methods in report.items():
        for method,m in methods.items():
            actual=metric([p for p in predictions if p['task']==task and p['method']==method])
            for key,value in actual.items():assert value==m[key],(task,method,key)
    result=dict(passed=True,fold_models=len(models),predictions_reconstructed=count,
        source_hashes_verified=True,held_out_runs_excluded=True,training_only_normalization_verified=True,
        layer31_feature_count=len(protocol['layer_features']['summary_31']),report_metrics_reproduced=True)
    (output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=Path('outputs/activation-explainer-a/logistic-labels-fixed/checkpoint'))
    main(p.parse_args().output_dir)
