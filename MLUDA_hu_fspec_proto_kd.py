"""Source-only prototype-level F_spec KD for MLUDA.

The teacher is frozen Full48 HyperSIGMA.  KD matches class-structure
distributions induced by source HyperSIGMA prototypes; it does not perform
point-wise feature regression and never uses Houston18 labels.
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config_Houston import BATCH_SIZE, CLASS_NUM, HalfWidth, l2_decay, lr, momentum, nBand  # noqa
from UtilsCMS import ILDA  # noqa
import mmd, utils  # noqa
from contrastive_loss import SupConLoss  # noqa
from net2 import DSANSS  # noqa
from MLUDA_hu_fspec_kd import center_patches, paired_source_samples  # noqa

CACHE = Path('/nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_full48_cache.npz')
OUT = Path('/nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_proto_kd')
SEED = 1174
LAMBDA = 0.1
TEMP = 0.1

def proto_distribution(feat, prototypes):
    feat = F.normalize(feat, dim=1)
    p = F.normalize(prototypes, dim=1)
    return F.softmax(feat @ p.t() / TEMP, dim=1)

def proto_kl(student, teacher, prototypes):
    q_t = proto_distribution(teacher.detach(), prototypes)
    log_q_s = torch.log(proto_distribution(student, prototypes).clamp_min(1e-8))
    return F.kl_div(log_q_s, q_t, reduction='batchmean')

def eval_source(model, loader, device):
    model.eval(); ce = nn.CrossEntropyLoss(); loss = correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device), x.to(device))[3]; y = y.to(device)
            loss += ce(out, y).item() * len(y); correct += (out.argmax(1) == y).sum().item(); total += len(y)
    return loss / total, correct / total

def main():
    global TEMP
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--epochs', type=int, default=100); ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--lambda-kd', type=float, default=LAMBDA); ap.add_argument('--temperature', type=float, default=TEMP)
    ap.add_argument('--cache', type=Path, default=CACHE); ap.add_argument('--output', type=Path, default=OUT)
    a = ap.parse_args(); TEMP = a.temperature
    utils.set_seed(a.seed); device = torch.device(a.device); out = a.output / f'lambda_{a.lambda_kd:g}'; out.mkdir(parents=True, exist_ok=True)
    cache = np.load(a.cache, allow_pickle=False); src_centers, src_fspec = cache['source_centers'], cache['source_fspec']; tgt_centers = cache['target_centers']
    source, src_gt = utils.load_data_houston(str(ROOT/'datasets/Houston/Houston13.mat'), str(ROOT/'datasets/Houston/Houston13_7gt.mat'))
    target = hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18.mat'))['ori_data']; data_s, data_t = ILDA(source, target, 2, 0.009)
    train_c, train_x, _, train_y, val_c, val_x, _, val_y = paired_source_samples(data_s, data_s, src_gt, a.seed)
    labels_cache = src_gt[src_centers[:,0], src_centers[:,1]].astype(np.int64) - 1
    prototypes = np.stack([src_fspec[labels_cache == c].mean(0) for c in range(CLASS_NUM)]).astype(np.float32)
    prototypes = F.normalize(torch.from_numpy(prototypes), dim=1).to(device)
    # Only source samples are used for prototype KD. Target labels are never loaded.
    source_map = {(int(r), int(c)): i for i, (r, c) in enumerate(src_centers)}
    train_tf = np.stack([src_fspec[source_map[(int(r), int(c))]] for r, c in train_c]).astype(np.float32)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_tf), torch.from_numpy(train_y)), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)), batch_size=BATCH_SIZE)
    target_x = center_patches(data_t, tgt_centers, 7); target_loader = DataLoader(torch.from_numpy(target_x), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    model = DSANSS(nBand, 7, CLASS_NUM).to(device); projection = nn.Linear(192, 128).to(device)
    ce = nn.CrossEntropyLoss(); con_s = SupConLoss(temperature=0.1).to(device); con_t = SupConLoss(temperature=0.1).to(device); dsh = utils.Domain_Occ_loss().to(device)
    history=[]; best={'val_acc':-1.0}; target_iter=iter(target_loader)
    for epoch in range(1, a.epochs+1):
        model.train(); projection.train(); L = lr / math.pow(1 + 10*(epoch-1)/a.epochs, 0.75)
        opt = torch.optim.SGD([{'params': model.parameters()}, {'params': projection.parameters(), 'lr': L}], lr=L, momentum=momentum, weight_decay=l2_decay)
        sums={k:0.0 for k in ('total','cls','scl','lmmd','proto')}; total=correct=0
        for sx, st, sy in train_loader:
            try: tx=next(target_iter)
            except StopIteration: target_iter=iter(target_loader); tx=next(target_iter)
            sx, st, sy, tx = sx.to(device), st.to(device), sy.to(device), tx.to(device)
            s0=utils.radiation_noise(sx.cpu()).float().to(device); t0=utils.radiation_noise(tx.cpu()).float().to(device)
            result=model.forward_with_spectral(sx,tx); sf,s1,_,so,source_out,tf,_,t1,to,target_out, sspec,tspec=result
            sx_flip = utils.flip_augmentation(sx.cpu()).float().to(device)
            tx_flip = utils.flip_augmentation(tx.cpu()).float().to(device)
            _,s2,_,so2,_,_,_,t2,_,_=model(s0,t0); _,s3,_,so3,_,_,_,t3,_,_=model(sx_flip,tx_flip)
            cls=ce(so,sy); pseudo=to.detach().softmax(1).argmax(1); scl=con_s(torch.cat([s2.unsqueeze(1),s3.unsqueeze(1)],1),sy)+con_t(torch.cat([t2.unsqueeze(1),t3.unsqueeze(1)],1),pseudo)
            lmmd=mmd.lmmd(sf,tf,sy,to.softmax(1),BATCH_SIZE=BATCH_SIZE,CLASS_NUM=CLASS_NUM); domain=dsh(source_out,target_out)
            proto=proto_kl(projection(sspec),st,prototypes); total_loss=cls+scl+0.01*(2/(1+math.exp(-10*epoch/a.epochs))-1)*lmmd+domain+a.lambda_kd*proto
            opt.zero_grad(); total_loss.backward(); opt.step(); n=len(sy); total+=n; correct+=(so.argmax(1)==sy).sum().item()
            for k,v in (('total',total_loss),('cls',cls),('scl',scl),('lmmd',lmmd),('proto',proto)): sums[k]+=v.item()*n
        vl,va=eval_source(model,val_loader,device); row={'epoch':epoch,'train_total_loss':sums['total']/total,'train_cls_loss':sums['cls']/total,'train_scl_loss':sums['scl']/total,'train_lmmd_loss':sums['lmmd']/total,'train_proto_kd_loss':sums['proto']/total,'train_acc':correct/total,'val_loss':vl,'val_acc':va,'lr':L}; history.append(row); print(json.dumps({'seed':a.seed,'lambda_proto':a.lambda_kd,**row}))
        if va>best['val_acc']:
            best=row.copy(); torch.save({'model':model.state_dict(),'projection':projection.state_dict(),'prototypes':prototypes.detach().cpu(),'temperature':TEMP,'best':best,'seed':a.seed,'target_gt_used_for_training_or_selection':False},out/f'seed_{a.seed}_best.pth')
    (out/f'seed_{a.seed}_history.json').write_text(json.dumps(history,indent=2)); print(json.dumps({'finished':True,'artifact':str(out),'best':best}))

if __name__=='__main__': main()
