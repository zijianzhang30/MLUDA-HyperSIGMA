"""Offline MLUDA inference sensitivity to the source reference batch.

The model/checkpoint and Houston18 target samples are fixed. Only the source
batch supplied alongside each target batch to the MBCA path is changed.
"""
import argparse, json, sys
from pathlib import Path
import hdf5storage, numpy as np, torch
from sklearn import metrics
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT))
from config_Houston import BATCH_SIZE, CLASS_NUM, nBand  # noqa
from net2 import DSANSS  # noqa
from MLUDA_hu_fspec_kd import center_patches, paired_source_samples  # noqa
from UtilsCMS import ILDA  # noqa
import utils  # noqa

def target_metrics(y, pred):
    cm=metrics.confusion_matrix(y,pred,labels=np.arange(CLASS_NUM)); pc=np.diag(cm)/np.maximum(cm.sum(1),1)
    return {'oa':float(np.mean(y==pred)),'aa':float(pc.mean()),'kappa':float(metrics.cohen_kappa_score(y,pred,labels=np.arange(CLASS_NUM))), 'per_class_accuracy':pc.tolist(),'prediction_distribution':np.bincount(pred,minlength=CLASS_NUM).tolist(),'confusion_matrix':cm.tolist()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',type=Path,default=Path('/nas1/zhangzj26/HyperSIGMA_adapted/clean_iterator_2seed/baseline/lambda_0_anneal/seed_1174_best.pth')); ap.add_argument('--cache',type=Path,default=Path('/nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_full48_fullft_cache.npz')); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--refs',type=int,default=5); ap.add_argument('--output',type=Path,default=Path('/nas1/zhangzj26/HyperSIGMA_adapted/diagnostics/source_ref_sensitivity.json')); a=ap.parse_args(); dev=torch.device(a.device)
    src,sg=utils.load_data_houston(str(ROOT/'datasets/Houston/Houston13.mat'),str(ROOT/'datasets/Houston/Houston13_7gt.mat')); tgt,tg=utils.load_data_houston(str(ROOT/'datasets/Houston/Houston18.mat'),str(ROOT/'datasets/Houston/Houston18_7gt.mat')); ds,dt=ILDA(src,tgt,2,0.009)
    cache=np.load(a.cache,allow_pickle=False); tc=cache['target_centers']; target_x=center_patches(dt,tc,7); target_gt=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18_7gt.mat'))['map']; y=target_gt[tc[:,0],tc[:,1]].astype(np.int64)-1
    _,train_x,_,_,_,_,_,_=paired_source_samples(ds,ds,sg,1174)
    ck=torch.load(a.checkpoint,map_location='cpu'); model=DSANSS(nBand,7,CLASS_NUM).to(dev); model.load_state_dict(ck['model'],strict=True); model.eval()
    rng=np.random.RandomState(20260903); refs=[]
    for i in range(a.refs): refs.append(torch.from_numpy(train_x[rng.permutation(len(train_x))[:BATCH_SIZE]].copy()))
    predictions=[]; logits_all=[]
    with torch.no_grad():
      for ref in refs:
        pp=[]; ll=[]
        for s in range(0,len(target_x),BATCH_SIZE):
          xb=torch.from_numpy(target_x[s:s+BATCH_SIZE]).to(dev); rb=ref[:len(xb)].to(dev); out=model(rb,xb)[8]; ll.append(out.cpu().numpy()); pp.append(out.argmax(1).cpu().numpy())
        predictions.append(np.concatenate(pp)); logits_all.append(np.concatenate(ll))
    pred=np.stack(predictions); logits=np.stack(logits_all)
    pair_agree=[]; pair_cos=[]; pair_l2=[]
    for i in range(a.refs):
      for j in range(i+1,a.refs):
        pair_agree.append(float(np.mean(pred[i]==pred[j]))); x=logits[i]; z=logits[j]; pair_cos.append(float(np.mean(np.sum(x*z,1)/(np.linalg.norm(x,axis=1)*np.linalg.norm(z,axis=1)+1e-12)))); pair_l2.append(float(np.mean(np.linalg.norm(x-z,axis=1))))
    changed=float(np.mean(np.any(pred!=pred[0:1],axis=0))); per_sample_nuniq=np.unique(pred,axis=0).shape[0]
    out={'checkpoint':str(a.checkpoint),'target_samples':int(len(y)),'reference_count':a.refs,'reference_batch_size':BATCH_SIZE,'reference_construction':'five independent permutations of the fixed Houston13 seed=1174 180/class source sample set; each reference has batch size 32','target_gt_used_for_training_or_selection':False,'official_eval_protocol':'MLUDA_hu.py uses source_data from the final training-loop batch for every test batch; train_loader_t is shuffle=True/drop_last=True and iterator resets only when exhausted. Current eval scripts use a fixed train_x[:BATCH_SIZE] reference.','metrics_by_reference':[target_metrics(y,pred[i]) for i in range(a.refs)],'sample_prediction_change_rate':changed,'num_unique_prediction_signatures':int(per_sample_nuniq),'prediction_agreement_pairwise_mean':float(np.mean(pair_agree)),'prediction_agreement_pairwise_min':float(np.min(pair_agree)),'prediction_agreement_pairwise_max':float(np.max(pair_agree)),'logit_cosine_pairwise_mean':float(np.mean(pair_cos)),'logit_l2_pairwise_mean':float(np.mean(pair_l2)),'pairwise_agreement':pair_agree,'pairwise_logit_cosine':pair_cos,'pairwise_logit_l2':pair_l2}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
