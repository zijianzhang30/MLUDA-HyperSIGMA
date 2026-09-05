import csv, json
from pathlib import Path
import numpy as np
BASE=Path('/nas1/zhangzj26/HyperSIGMA_adapted/structural_gradient_diagnostic/lambda_0.1_anneal')
OUT=Path('/home/zhangzj26/TGRS_MLUDA-2024/diagnostics')
OUT.mkdir(parents=True,exist_ok=True)
splits=(1370,1703,1418)
rows=[]
for s in splits:
    d=json.loads((BASE/f'split_{s}_opt_1174_gradient_diagnostic.json').read_text())
    for r in d['rows']:
        r=dict(r); r['split']=s; r['weighted_kd_to_uda_ratio']=r['lambda_rel']*r['mean_norm_ratio']; rows.append(r)
summary={'protocol':{'splits':list(splits),'teacher':'Full-FT HyperSIGMA F_spat','student_scope':'MLUDA spatial branch conv5-conv8 before MBCA','relation':'within-sample 49x49 position cosine MSE, diagonal excluded','schedule':'1-20 lambda=.1, 21-40 linear to 0, 41-100 0','diagnostic_epochs':[1,5,10,20,30,40,60,80,100],'target_gt_used_for_training_or_selection':False,'projection_excluded':True},'rows':rows,'per_split':{}}
for s in splits:
    rr=[r for r in rows if r['split']==s]
    active=[r for r in rr if r['epoch']<=40]
    summary['per_split'][str(s)]={'active_mean_cosine':float(np.mean([r['mean_cosine'] for r in active])),'active_median_sampled_epoch_mean_cosine':float(np.median([r['mean_cosine'] for r in active])),'active_mean_conflict_ratio':float(np.mean([r['conflict_ratio'] for r in active])),'active_mean_weighted_kd_to_uda_ratio':float(np.mean([r['weighted_kd_to_uda_ratio'] for r in active])),'all_mean_cosine':float(np.mean([r['mean_cosine'] for r in rr])),'all_median_sampled_epoch_mean_cosine':float(np.median([r['mean_cosine'] for r in rr])),'all_mean_conflict_ratio':float(np.mean([r['conflict_ratio'] for r in rr])),'all_mean_weighted_kd_to_uda_ratio':float(np.mean([r['weighted_kd_to_uda_ratio'] for r in rr]))}
(OUT/'structural_gradient_diagnostic_summary.json').write_text(json.dumps(summary,indent=2))
with (OUT/'structural_gradient_diagnostic_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['split','epoch','lambda_rel','num_batches','mean_cosine','mean_uda_norm','mean_kd_norm','mean_norm_ratio','weighted_kd_to_uda_ratio','conflict_ratio','mean_projection_kd_norm']); w.writeheader(); w.writerows(rows)
print(json.dumps(summary,indent=2))
