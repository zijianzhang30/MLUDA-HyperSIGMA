import csv, json, sys
from pathlib import Path
import hdf5storage, numpy as np, torch
from sklearn import metrics
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT))
from config_Houston import BATCH_SIZE, CLASS_NUM, nBand
from MLUDA_hu_fspec_kd import center_patches, paired_source_samples
from net2 import DSANSS
from UtilsCMS import ILDA
import utils
ADAPT=Path('/nas1/zhangzj26/HyperSIGMA_adapted'); OUT=ADAPT/'domain_coverage'; SPLITS=(1370,1703,1418)
def metric(y,p):
 cm=metrics.confusion_matrix(y,p,labels=np.arange(CLASS_NUM)); pc=np.diag(cm)/np.maximum(cm.sum(1),1)
 return {'oa':float((y==p).mean()),'aa':float(pc.mean()),'kappa':float(metrics.cohen_kappa_score(y,p,labels=np.arange(CLASS_NUM))),'per_class_accuracy':pc.tolist(),'prediction_distribution':np.bincount(p,minlength=CLASS_NUM).tolist(),'confusion_matrix':cm.tolist()}
def ev(m,x,y,ref,d):
 m.eval(); ps=[]
 with torch.no_grad():
  for i in range(0,len(x),BATCH_SIZE):
   xb=torch.from_numpy(x[i:i+BATCH_SIZE]).to(d); rb=ref[:len(xb)].to(d); ps.append(m(rb,xb)[8].argmax(1).cpu().numpy())
 return metric(y,np.concatenate(ps))
def main():
 import argparse; ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cuda:0'); a=ap.parse_args(); d=torch.device(a.device)
 src,sg=utils.load_data_houston(str(ROOT/'datasets/Houston/Houston13.mat'),str(ROOT/'datasets/Houston/Houston13_7gt.mat')); tgt=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18.mat'))['ori_data']; sa,ta=ILDA(src,tgt,2,0.009)
 tc=np.load(ADAPT/'mluda_fspec_full48_cache.npz')['target_centers']; tx=center_patches(ta,tc,7); ty=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18_7gt.mat'))['map'][tc[:,0],tc[:,1]].astype(np.int64)-1
 rows=[]
 for split in SPLITS:
  _,tr,_,_,_,_,_,_=paired_source_samples(sa,sa,sg,split); ref=torch.from_numpy(tr[:BATCH_SIZE]).to(d); stem=f'split_{split}_opt_1174'; broot=ADAPT/('seed_disentanglement' if split in (1370,1418) else 'independent_split_validation')
  sroot=ADAPT/('spatial_structural_3split' if split==1703 else 'spatial_structural_10split') # structural source-only extension
  dirs={'baseline':broot/'baseline/lambda_0_anneal','source_only':sroot/'lambda_0.1_anneal','target_only':OUT/'target_only/lambda_0.1_anneal','source_target':OUT/'source_target/lambda_0.1_anneal'}
  methods={}
  for name,dd in dirs.items():
   ck=dd/f'{stem}_best.pth'
   if not ck.exists(): continue
   payload=torch.load(ck,map_location='cpu'); m=DSANSS(nBand,7,CLASS_NUM).to(d); m.load_state_dict(payload['model'],strict=True); methods[name]={'target':ev(m,tx,ty,ref,d),'best':payload.get('best',{}) ,'checkpoint':str(ck)}
  if 'baseline' in methods:
   for name in ('source_only','target_only','source_target'):
    if name in methods: methods[name]['delta']={k:methods[name]['target'][k]-methods['baseline']['target'][k] for k in ('oa','aa','kappa')}
  rows.append({'split':split,'methods':methods})
 summary={'protocol':{'splits':list(SPLITS),'teacher':'Full-FT HyperSIGMA F_spat','target_gt_used_for_training_or_selection':False,'target_iterator':'reset each epoch'},'runs':rows}
 (OUT/'domain_coverage_summary.json').write_text(json.dumps(summary,indent=2));
 with (OUT/'domain_coverage_summary.csv').open('w',newline='') as f:
  w=csv.writer(f); w.writerow(['split','method','oa','aa','kappa','delta_oa','delta_aa','delta_kappa','best_epoch'])
  for r in rows:
   for n,v in r['methods'].items(): w.writerow([r['split'],n,v['target']['oa'],v['target']['aa'],v['target']['kappa'],v.get('delta',{}).get('oa'),v.get('delta',{}).get('aa'),v.get('delta',{}).get('kappa'),v['best'].get('epoch')])
 print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
