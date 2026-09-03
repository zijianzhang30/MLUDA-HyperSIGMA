"""Merge corrected-protocol 2-seed and 3 remaining-seed artifacts."""
import json
from pathlib import Path
import numpy as np

ROOT2=Path('/nas1/zhangzj26/HyperSIGMA_adapted/clean_iterator_2seed')
ROOT5=Path('/nas1/zhangzj26/HyperSIGMA_adapted/clean_iterator_5seed')
OUT=ROOT5/'clean_5seed_summary.json'
SEEDS=[1174,1370,1417,1418,1546]
VERSIONS={'baseline':('0','Baseline'),'fspec':('0.1','Full-FT F_spec-only'),'fspat':('0.1','Full-FT F_spat-only')}

def directory(ver,lam,seed):
    root=ROOT2 if seed in (1174,1546) else ROOT5
    return root/ver/f'lambda_{lam}_anneal'

runs={}
for ver,(lam,label) in VERSIONS.items():
    rows=[]
    for seed in SEEDS:
        d=directory(ver,lam,seed)
        metric=json.loads((d/f'houston18_posthoc_seed_{seed}.json').read_text())
        history=json.loads((d/f'seed_{seed}_history.json').read_text())
        best=max(history,key=lambda x:x['val_acc'])
        rows.append({'seed':seed,'checkpoint':str(d/f'seed_{seed}_best.pth'),'history':str(d/f'seed_{seed}_history.json'),'posthoc':str(d/f'houston18_posthoc_seed_{seed}.json'),'best_epoch':best['epoch'],'source_val_accuracy':best['val_acc'],'losses_at_best':{'L_spec':best['train_rel_spec_loss'],'L_spat':best['train_rel_spat_loss'],'L_joint':best['train_rel_kd_loss']},'target':{k:metric[k] for k in ('oa','aa','kappa','per_class_accuracy','prediction_distribution','confusion_matrix')}})
    vals=np.asarray([[r['target'][k] for k in ('oa','aa','kappa')] for r in rows]); pc=np.asarray([r['target']['per_class_accuracy'] for r in rows])
    runs[ver]={'label':label,'runs':rows,'mean':dict(zip(('oa','aa','kappa'),vals.mean(0).tolist())),'std_population':dict(zip(('oa','aa','kappa'),vals.std(0).tolist())),'per_class_mean':pc.mean(0).tolist(),'per_class_std_population':pc.std(0).tolist()}

base={r['seed']:r for r in runs['baseline']['runs']}
for ver in ('fspec','fspat'):
    deltas=[]
    wins={k:0 for k in ('oa','aa','kappa')}
    for r in runs[ver]['runs']:
        b=base[r['seed']]; delta={k:r['target'][k]-b['target'][k] for k in ('oa','aa','kappa')}
        for k in wins: wins[k]+=int(delta[k]>0)
        deltas.append({'seed':r['seed'],**delta})
    runs[ver]['delta_vs_baseline']=deltas; runs[ver]['win_count_vs_baseline']=wins

# The historical two-seed headline is recomputed explicitly from exact clean
# baseline seeds 1174 and 1546 using population standard deviation (ddof=0).
two=np.asarray([[base[s]['target'][k] for k in ('oa','aa','kappa')] for s in (1174,1546)])
summary={'protocol':{'target_iterator':'reset at the beginning of every epoch','seeds':SEEDS,'teacher':'Full-FT HyperSIGMA Full48 for KD only','target_gt_used_for_training_or_selection':False,'std_definition':'population std, numpy ddof=0'},'two_seed_baseline_audit':{'seeds':[1174,1546],'values':two.tolist(),'mean':dict(zip(('oa','aa','kappa'),two.mean(0).tolist())),'std_population':dict(zip(('oa','aa','kappa'),two.std(0).tolist())),'checkpoint_paths':[base[s]['checkpoint'] for s in (1174,1546)]},'methods':runs}
OUT.write_text(json.dumps(summary,indent=2)); print(OUT)
