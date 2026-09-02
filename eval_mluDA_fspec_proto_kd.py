"""Post-hoc Houston18 audit for one prototype-KD checkpoint."""
import argparse, json, sys, re
from pathlib import Path
import hdf5storage, numpy as np, torch
from sklearn import metrics
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT))
from config_Houston import BATCH_SIZE, CLASS_NUM, HalfWidth, nBand  # noqa
from net2 import DSANSS  # noqa
from MLUDA_hu_fspec_kd import center_patches, paired_source_samples  # noqa
import utils
from UtilsCMS import ILDA

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--cache',type=Path,default=Path('/nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_full48_cache.npz')); ap.add_argument('--device',default='cuda:0'); a=ap.parse_args(); dev=torch.device(a.device)
    cache=np.load(a.cache,allow_pickle=False); centers=cache['target_centers']
    source,gt=utils.load_data_houston(str(ROOT/'datasets/Houston/Houston13.mat'),str(ROOT/'datasets/Houston/Houston13_7gt.mat')); target=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18.mat'))['ori_data']; ds,dt=ILDA(source,target,2,0.009)
    _,train_x,_,_,_,_,_,_=paired_source_samples(ds,ds,gt,1174); tx=center_patches(dt,centers,7)
    h18=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18_7gt.mat'))['map']; y=h18[centers[:,0],centers[:,1]].astype(np.int64)-1
    ck=torch.load(a.checkpoint,map_location='cpu'); m=DSANSS(nBand,7,CLASS_NUM).to(dev); m.load_state_dict(ck['model'],strict=True); m.eval(); pred=[]
    with torch.no_grad():
        ref=torch.from_numpy(train_x[:BATCH_SIZE]).to(dev)
        for s in range(0,len(tx),BATCH_SIZE): pred.append(m(ref[:min(BATCH_SIZE,len(tx)-s)],torch.from_numpy(tx[s:s+BATCH_SIZE]).to(dev))[8].argmax(1).cpu().numpy())
    pred=np.concatenate(pred); cm=metrics.confusion_matrix(y,pred,labels=np.arange(CLASS_NUM)); pc=np.diag(cm)/np.maximum(cm.sum(1),1)
    out={'checkpoint':str(a.checkpoint),'target_gt_used_for_training_or_selection':False,'n':int(len(y)),'oa':float((y==pred).mean()),'aa':float(pc.mean()),'kappa':float(metrics.cohen_kappa_score(y,pred,labels=np.arange(CLASS_NUM))),'per_class_accuracy':pc.tolist(),'prediction_distribution':np.bincount(pred,minlength=CLASS_NUM).tolist(),'confusion_matrix':cm.tolist()}
    match = re.search(r'seed_(\d+)_best', a.checkpoint.name); suffix = f"_seed_{match.group(1)}" if match else ""
    outp=a.checkpoint.parent/f'houston18_posthoc{suffix}.json'; outp.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
