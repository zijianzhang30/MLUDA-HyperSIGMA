import json, sys
from pathlib import Path
import hdf5storage, numpy as np, torch
from sklearn import metrics
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT))
from config_Houston import BATCH_SIZE, CLASS_NUM, nBand
from MLUDA_hu_fspec_kd import center_patches, paired_source_samples
from net2 import DSANSS
from UtilsCMS import ILDA
import utils
OUT=Path('/nas1/zhangzj26/HyperSIGMA_adapted/spatial_structural_3split')
SPLITS=(1174,1703,2141)
def metric(y,p):
    cm=metrics.confusion_matrix(y,p,labels=np.arange(CLASS_NUM)); pc=np.diag(cm)/np.maximum(cm.sum(1),1)
    return {'oa':float((y==p).mean()),'aa':float(pc.mean()),'kappa':float(metrics.cohen_kappa_score(y,p,labels=np.arange(CLASS_NUM))),'per_class_accuracy':pc.tolist(),'prediction_distribution':np.bincount(p,minlength=CLASS_NUM).tolist(),'confusion_matrix':cm.tolist()}
def evaluate(model,x,y,ref,dev):
    model.eval(); pred=[]
    with torch.no_grad():
        for i in range(0,len(x),BATCH_SIZE):
            xb=torch.from_numpy(x[i:i+BATCH_SIZE]).to(dev); rb=ref[:len(xb)].to(dev); pred.append(model(rb,xb)[8].argmax(1).cpu().numpy())
    return metric(y,np.concatenate(pred))
def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cuda:0'); a=ap.parse_args(); dev=torch.device(a.device)
    src,sg=utils.load_data_houston(str(ROOT/'datasets/Houston/Houston13.mat'),str(ROOT/'datasets/Houston/Houston13_7gt.mat')); tgt=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18.mat'))['ori_data']; sa,ta=ILDA(src,tgt,2,0.009)
    cache=np.load('/nas1/zhangzj26/HyperSIGMA_adapted/hypersigma_fspat_full48_fullft_cache.npz'); tc=cache['target_centers']; tx=center_patches(ta,tc,7); tg=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18_7gt.mat'))['map']; ty=tg[tc[:,0],tc[:,1]].astype(np.int64)-1
    rows=[]
    for split in SPLITS:
        _,train_x,_,_,*_=paired_source_samples(sa,sa,sg,split); ref=torch.from_numpy(train_x[:BATCH_SIZE]); methods={}
        root=Path('/nas1/zhangzj26/HyperSIGMA_adapted/seed_disentanglement') if split==1174 else Path('/nas1/zhangzj26/HyperSIGMA_adapted/independent_split_validation'); stem=f'split_{split}_opt_1174'
        dirs={'baseline':root/'baseline/lambda_0_anneal','pooled_fspat':root/'fspat/lambda_0.1_anneal','structural_token':OUT/'lambda_0.1_anneal'}
        for name,d in dirs.items():
            ck=d/f'{stem}_best.pth'; hist=json.loads((d/f'{stem}_history.json').read_text()); payload=torch.load(ck,map_location='cpu'); model=DSANSS(nBand,7,CLASS_NUM).to(dev); model.load_state_dict(payload['model'],strict=True); best=max(hist,key=lambda z:z['val_acc']); methods[name]={'target':evaluate(model,tx,ty,ref,dev),'best_epoch':best['epoch'],'source_val_accuracy':best['val_acc'],'checkpoint':str(ck)}
        rows.append({'split':split,**methods,'delta_pooled':{k:methods['pooled_fspat']['target'][k]-methods['baseline']['target'][k] for k in ('oa','aa','kappa')},'delta_structural':{k:methods['structural_token']['target'][k]-methods['baseline']['target'][k] for k in ('oa','aa','kappa')}})
    summary={'protocol':{'splits':list(SPLITS),'teacher':'Full-FT HyperSIGMA F_spat','teacher_map':'[768,32,32] pooled to 7x7','student_map':'[96,7,7]','relation':'non-diagonal 49x49 position cosine MSE','target_iterator':'reset each epoch','target_gt_used_for_training_or_selection':False},'runs':rows}
    (OUT/'spatial_structural_3split_summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
