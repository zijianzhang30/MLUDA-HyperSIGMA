"""Cache frozen Full48 HyperSIGMA spatial features for joint relational KD."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import hdf5storage, numpy as np, torch
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'third_party/HyperSIGMA/ImageClassification'))
import utils
from hypersigma_stage1_protocol import forward_parts, HALF, IMG_SIZE
from hypersigma_teacher_smoke_test import SSFusionFramework
CKPT=Path('/nas1/zhangzj26/HyperSIGMA_adapted/protocol_stage1/bands48/stage1_best.pth')
OUT=Path('/home/zhangzj26/TGRS_MLUDA-2024/hypersigma_fspat_full48_cache.npz')
def extract(m,cube,centers,dev,bs):
    pad=np.pad(cube,((HALF,HALF),(HALF,HALF),(0,0)),mode='constant'); out=[]
    with torch.no_grad():
      for st in range(0,len(centers),bs):
        cc=centers[st:st+bs]; xb=np.empty((len(cc),48,IMG_SIZE,IMG_SIZE),np.float32)
        for j,(r,c) in enumerate(cc): xb[j]=pad[r:r+IMG_SIZE,c:c+IMG_SIZE].transpose(2,0,1)
        spat,_,_,_=forward_parts(m,torch.from_numpy(xb).to(dev)); out.append(spat[-1].mean((2,3)).cpu().numpy().astype(np.float32))
    return np.concatenate(out)
def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu'); ap.add_argument('--batch-size',type=int,default=64); ap.add_argument('--output',type=Path,default=OUT); ap.add_argument('--checkpoint',type=Path,default=CKPT); a=ap.parse_args(); dev=torch.device(a.device)
  src,sg=utils.load_data_houston(str(ROOT/'datasets/Houston/Houston13.mat'),str(ROOT/'datasets/Houston/Houston13_7gt.mat')); tgt,tg=utils.load_data_houston(str(ROOT/'datasets/Houston/Houston18.mat'),str(ROOT/'datasets/Houston/Houston18_7gt.mat')); sc=np.argwhere(sg>0).astype(np.int64); tc=np.argwhere(tg>0).astype(np.int64)
  m=SSFusionFramework(img_size=33,in_channels=48,patch_size=2,classes=7,model_size='base'); m.load_state_dict(torch.load(a.checkpoint,map_location='cpu')['model'],strict=True); m.to(dev).eval(); [p.requires_grad_(False) for p in m.parameters()]
  sf,tf=extract(m,src,sc,dev,a.batch_size),extract(m,tgt,tc,dev,a.batch_size); assert np.isfinite(sf).all() and np.isfinite(tf).all(); a.output.parent.mkdir(parents=True,exist_ok=True); np.savez(a.output,source_centers=sc,target_centers=tc,source_fspat=sf,target_fspat=tf,teacher_checkpoint=str(a.checkpoint)); print('saved',a.output,'checkpoint',a.checkpoint,sf.shape,tf.shape)
if __name__=='__main__': main()
