"""Source-only relational F_spec KD for the original MLUDA.

KD matches the within-source-batch pairwise cosine geometry of frozen
HyperSIGMA F_spec and the pre-MBCA MLUDA spectral branch. No point-wise target
matching, prototypes, or Houston18 labels are used in training.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import hdf5storage
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT))
from config_Houston import BATCH_SIZE, CLASS_NUM, HalfWidth, l2_decay, lr, momentum, nBand  # noqa
from UtilsCMS import ILDA  # noqa
import mmd, utils  # noqa
from contrastive_loss import SupConLoss  # noqa
from net2 import DSANSS  # noqa
from MLUDA_hu_fspec_kd import center_patches, paired_source_samples  # noqa

CACHE=Path('/nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_full48_cache.npz')
SPAT_CACHE=Path('/home/zhangzj26/TGRS_MLUDA-2024/hypersigma_fspat_full48_cache.npz')
OUT=Path('/nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_rel_kd')

def relational_loss(proj, student, teacher):
    s=F.normalize(proj(student),dim=1); t=F.normalize(teacher.detach(),dim=1)
    n=s.shape[0]; mask=~torch.eye(n,dtype=torch.bool,device=s.device)
    gs=s@s.t(); gt=t@t.t()
    return F.mse_loss(gs[mask],gt[mask])

def eval_source(model, loader, device):
    model.eval(); ce=nn.CrossEntropyLoss(); ls=correct=total=0
    with torch.no_grad():
        for x,y in loader:
            out=model(x.to(device),x.to(device))[3]; y=y.to(device); ls+=ce(out,y).item()*len(y); correct+=(out.argmax(1)==y).sum().item(); total+=len(y)
    return ls/total,correct/total

def scheduled_lambda(epoch, base, schedule):
    """Return the active relational-KD weight for a 1-based epoch."""
    if schedule == 'fixed':
        return base
    if epoch <= 20:
        return base
    if epoch <= 40:
        # Linear interpolation from base at epoch 20 to zero at epoch 40.
        return base * (40 - epoch) / 20.0
    return 0.0

def gradient_probe(uda_loss, kd_loss, shared_params, projection):
    """Gradient conflict on student shared backbone; projection is separate."""
    gu = torch.autograd.grad(uda_loss, shared_params, retain_graph=True, allow_unused=True)
    gk = torch.autograd.grad(kd_loss, shared_params, retain_graph=True, allow_unused=True)
    def flat(gs, params):
        return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                          for g, p in zip(gs, params)])
    vu, vk = flat(gu, shared_params), flat(gk, shared_params)
    nu, nk = torch.linalg.vector_norm(vu), torch.linalg.vector_norm(vk)
    cos = torch.dot(vu, vk) / (nu * nk).clamp_min(1e-12)
    gp = torch.autograd.grad(kd_loss, list(projection.parameters()), retain_graph=True, allow_unused=True)
    vp = flat(gp, list(projection.parameters())) if any(g is not None for g in gp) else vu.new_zeros(1)
    return {'cosine': float(cos.detach().cpu()), 'uda_norm': float(nu.detach().cpu()),
            'kd_norm': float(nk.detach().cpu()), 'norm_ratio': float((nk / nu.clamp_min(1e-12)).detach().cpu()),
            'projection_kd_norm': float(torch.linalg.vector_norm(vp).detach().cpu()),
            'conflict': bool(cos.detach().cpu() < 0)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu'); ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--seed',type=int,default=1174,help='legacy seed used for both split and optimization unless either explicit seed is supplied'); ap.add_argument('--split-seed',type=int,default=None,help='seed used only by the local Houston13 source split RNG'); ap.add_argument('--optimization-seed',type=int,default=None,help='seed used for model/projection initialization, DataLoader order, and augmentation'); ap.add_argument('--lambda-rel',type=float,default=0.1); ap.add_argument('--schedule',choices=('fixed','anneal'),default='fixed'); ap.add_argument('--cache',type=Path,default=CACHE); ap.add_argument('--spat-cache',type=Path,default=SPAT_CACHE); ap.add_argument('--joint-spatial',action='store_true'); ap.add_argument('--spatial-only',action='store_true'); ap.add_argument('--output',type=Path,default=OUT); ap.add_argument('--diagnostic-epochs',type=int,nargs='*',default=())
    a=ap.parse_args(); split_seed=a.seed if a.split_seed is None else a.split_seed; optimization_seed=a.seed if a.optimization_seed is None else a.optimization_seed; explicit_seed_pair=a.split_seed is not None or a.optimization_seed is not None; artifact_stem=f'split_{split_seed}_opt_{optimization_seed}' if explicit_seed_pair else f'seed_{a.seed}'; utils.set_seed(optimization_seed); dev=torch.device(a.device); suffix = f'lambda_{a.lambda_rel:g}' + ('_anneal' if a.schedule == 'anneal' else ''); out=a.output/suffix; out.mkdir(parents=True,exist_ok=True)
    if a.joint_spatial and a.spatial_only: raise ValueError('--joint-spatial and --spatial-only are mutually exclusive')
    use_spatial = a.joint_spatial or a.spatial_only; use_spectral = not a.spatial_only
    c=np.load(a.cache,allow_pickle=False) if use_spectral else None
    if use_spectral: sc, sf, tc = c['source_centers'], c['source_fspec'], c['target_centers']
    else: sc, sf = None, None; tc = np.load(a.spat_cache,allow_pickle=False)['target_centers']
    spat_cache=np.load(a.spat_cache,allow_pickle=False) if use_spatial else None
    source,gt=utils.load_data_houston(str(ROOT/'datasets/Houston/Houston13.mat'),str(ROOT/'datasets/Houston/Houston13_7gt.mat')); target=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18.mat'))['ori_data']; ds,dt=ILDA(source,target,2,0.009)
    train_c,train_x,_,train_y,val_c,val_x,_,val_y=paired_source_samples(ds,ds,gt,split_seed)
    if use_spectral:
        sm={(int(r),int(col)):i for i,(r,col) in enumerate(sc)}; train_tf=np.stack([sf[sm[(int(r),int(col))]] for r,col in train_c]).astype(np.float32)
    if use_spatial:
        ssm={(int(r),int(col)):i for i,(r,col) in enumerate(spat_cache['source_centers'])}; train_ts=np.stack([spat_cache['source_fspat'][ssm[(int(r),int(col))]] for r,col in train_c]).astype(np.float32)
    if a.joint_spatial:
        train_loader=DataLoader(TensorDataset(torch.from_numpy(train_x),torch.from_numpy(train_tf),torch.from_numpy(train_ts),torch.from_numpy(train_y)),batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    elif a.spatial_only:
        train_loader=DataLoader(TensorDataset(torch.from_numpy(train_x),torch.from_numpy(train_ts),torch.from_numpy(train_y)),batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    else:
        train_loader=DataLoader(TensorDataset(torch.from_numpy(train_x),torch.from_numpy(train_tf),torch.from_numpy(train_y)),batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    val_loader=DataLoader(TensorDataset(torch.from_numpy(val_x),torch.from_numpy(val_y)),batch_size=BATCH_SIZE)
    target_x=center_patches(dt,tc,7); target_loader=DataLoader(torch.from_numpy(target_x),batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    model=DSANSS(nBand,7,CLASS_NUM).to(dev); projection=nn.Linear(192,128).to(dev) if use_spectral else None; projection_spatial=nn.Linear(96,768).to(dev) if use_spatial else None; ce=nn.CrossEntropyLoss(); con_s=SupConLoss(temperature=0.1).to(dev); con_t=SupConLoss(temperature=0.1).to(dev); dsh=utils.Domain_Occ_loss().to(dev)
    hist=[]; best={'val_acc':-1.0}; grad_rows=[]; diag_epochs=set(a.diagnostic_epochs); shared_params=[p for p in model.feature_layers.parameters() if p.requires_grad]
    for epoch in range(1,a.epochs+1):
        # Reinitialize the target iterator at each epoch, matching the
        # original MLUDA_hu.py sampling protocol. This is the only protocol
        # change in the iterator-fix sanity check.
        target_iter=iter(target_loader)
        model.train();
        if projection is not None: projection.train()
        if projection_spatial is not None: projection_spatial.train()
        L=lr/math.pow(1+10*(epoch-1)/a.epochs,0.75); params=[{'params':model.parameters()}]
        if projection is not None: params.append({'params':projection.parameters(),'lr':L})
        if projection_spatial is not None: params.append({'params':projection_spatial.parameters(),'lr':L})
        opt=torch.optim.SGD(params,lr=L,momentum=momentum,weight_decay=l2_decay); sums={k:0.0 for k in ('total','cls','scl','lmmd','domain','rel','rel_spec','rel_spat')}; total=correct=0; epoch_grad=[]
        for batch in train_loader:
            if a.joint_spatial: sx,st,sts,sy=batch
            elif a.spatial_only: sx,sts,sy=batch
            else: sx,st,sy=batch
            try: tx=next(target_iter)
            except StopIteration: target_iter=iter(target_loader); tx=next(target_iter)
            if a.joint_spatial:
                sx,st,sts,sy,tx=sx.to(dev),st.to(dev),sts.to(dev),sy.to(dev),tx.to(dev)
            elif a.spatial_only:
                sx,sts,sy,tx=sx.to(dev),sts.to(dev),sy.to(dev),tx.to(dev)
            else:
                sx,st,sy,tx=sx.to(dev),st.to(dev),sy.to(dev),tx.to(dev)
            s0=utils.radiation_noise(sx.cpu()).float().to(dev); t0=utils.radiation_noise(tx.cpu()).float().to(dev); sx1=utils.flip_augmentation(sx.cpu()).float().to(dev); tx1=utils.flip_augmentation(tx.cpu()).float().to(dev)
            if a.joint_spatial or a.spatial_only:
                z=model.forward_with_spectral_spatial(sx,tx); sfm,s1,_,so,source_out,tfm,_,t1,to,target_out,sspec,tspec,sspat,tspat=z
            else:
                sfm,s1,_,so,source_out,tfm,_,t1,to,target_out,sspec,tspec=model.forward_with_spectral(sx,tx)
            _,s2,_,so2,_,_,_,t2,_,_=model(s0,t0); _,s3,_,so3,_,_,_,t3,_,_=model(sx1,tx1)
            cls=ce(so,sy); pseudo=to.softmax(1).detach().argmax(1); scl=con_s(torch.cat([s2.unsqueeze(1),s3.unsqueeze(1)],1),sy)+con_t(torch.cat([t2.unsqueeze(1),t3.unsqueeze(1)],1),pseudo); lm=mmd.lmmd(sfm,tfm,sy,to.softmax(1),BATCH_SIZE=BATCH_SIZE,CLASS_NUM=CLASS_NUM); domain=dsh(source_out,target_out); active_lambda=scheduled_lambda(epoch, a.lambda_rel, a.schedule); uda_obj=cls+scl+0.01*(2/(1+math.exp(-10*epoch/a.epochs))-1)*lm+domain
            rel_spec=relational_loss(projection,sspec,st) if use_spectral else torch.zeros((),device=dev)
            rel_spat=relational_loss(projection_spatial,sspat,sts) if use_spatial else torch.zeros((),device=dev)
            joint_rel=(rel_spec+rel_spat)/2 if a.joint_spatial else (rel_spat if a.spatial_only else rel_spec)
            if epoch in diag_epochs and projection is not None:
                epoch_grad.append(gradient_probe(uda_obj, joint_rel, shared_params, projection))
            total_loss=uda_obj+active_lambda*joint_rel
            opt.zero_grad(); total_loss.backward(); opt.step(); n=len(sy); total+=n; correct+=(so.argmax(1)==sy).sum().item()
            for k,v in (('total',total_loss),('cls',cls),('scl',scl),('lmmd',lm),('domain',domain),('rel',joint_rel),('rel_spec',rel_spec),('rel_spat',rel_spat)): sums[k]+=v.item()*n
        active_lambda=scheduled_lambda(epoch, a.lambda_rel, a.schedule); vl,va=eval_source(model,val_loader,dev); row={'epoch':epoch,'split_seed':split_seed,'optimization_seed':optimization_seed,'lambda_rel':active_lambda,'lambda_rel_base':a.lambda_rel,'schedule':a.schedule,'joint_spatial':a.joint_spatial,'spatial_only':a.spatial_only,'train_total_loss':sums['total']/total,'train_cls_loss':sums['cls']/total,'train_scl_loss':sums['scl']/total,'train_lmmd_loss':sums['lmmd']/total,'train_domain_loss':sums['domain']/total,'train_rel_kd_loss':sums['rel']/total,'train_rel_spec_loss':sums['rel_spec']/total,'train_rel_spat_loss':sums['rel_spat']/total,'train_acc':correct/total,'val_loss':vl,'val_acc':va,'lr':L}; hist.append(row); print(json.dumps({'split_seed':split_seed,'optimization_seed':optimization_seed,'lambda_rel':active_lambda,'schedule':a.schedule,**row}))
        if epoch_grad:
            grad_rows.append({'epoch':epoch,'lambda_rel':active_lambda,'num_batches':len(epoch_grad),'mean_cosine':float(np.mean([r['cosine'] for r in epoch_grad])),'mean_uda_norm':float(np.mean([r['uda_norm'] for r in epoch_grad])),'mean_kd_norm':float(np.mean([r['kd_norm'] for r in epoch_grad])),'mean_norm_ratio':float(np.mean([r['norm_ratio'] for r in epoch_grad])),'conflict_ratio':float(np.mean([r['conflict'] for r in epoch_grad])),'mean_projection_kd_norm':float(np.mean([r['projection_kd_norm'] for r in epoch_grad]))})
        if va>best['val_acc']:
            best=row.copy(); payload={'model':model.state_dict(),'projection':projection.state_dict() if projection is not None else None,'lambda_rel':a.lambda_rel,'schedule':a.schedule,'seed':optimization_seed,'split_seed':split_seed,'optimization_seed':optimization_seed,'kd_type':'source_only_relational_fspat' if a.spatial_only else ('source_only_relational_fspec_joint' if a.joint_spatial else 'source_only_relational_fspec'),'diagonal_excluded':True,'target_gt_used_for_training_or_selection':False,'best':best};
            if projection_spatial is not None: payload['projection_spatial']=projection_spatial.state_dict()
            torch.save(payload,out/f'{artifact_stem}_best.pth')
    (out/f'{artifact_stem}_history.json').write_text(json.dumps(hist,indent=2)); (out/f'{artifact_stem}_source_training_summary.json').write_text(json.dumps({'lambda_rel':a.lambda_rel,'schedule':a.schedule,'joint_spatial':a.joint_spatial,'spatial_only':a.spatial_only,'seed':optimization_seed,'split_seed':split_seed,'optimization_seed':optimization_seed,'kd_type':'source_only_relational_fspat' if a.spatial_only else ('source_only_relational_fspec_joint' if a.joint_spatial else 'source_only_relational_fspec'),'diagonal_excluded':True,'target_gt_used_for_training_or_selection':False,'best':best},indent=2));
    if a.diagnostic_epochs:
        (out/f'{artifact_stem}_gradient_diagnostic.json').write_text(json.dumps({'seed':optimization_seed,'split_seed':split_seed,'optimization_seed':optimization_seed,'schedule':a.schedule,'diagnostic_epochs':a.diagnostic_epochs,'shared_parameter_scope':'model.feature_layers only; projection excluded from conflict metrics','rows':grad_rows},indent=2))
    print(json.dumps({'finished':True,'artifact':str(out),'artifact_stem':artifact_stem,'split_seed':split_seed,'optimization_seed':optimization_seed,'best':best,'gradient_rows':len(grad_rows)}))

if __name__=='__main__': main()
