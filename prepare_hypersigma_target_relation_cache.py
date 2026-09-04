"""Extract Full-FT HyperSIGMA target spatial relations for unlabeled KD."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import hdf5storage, numpy as np, torch
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'third_party/HyperSIGMA/ImageClassification'))
import utils
from hypersigma_stage1_protocol import forward_parts, HALF, IMG_SIZE
from hypersigma_teacher_smoke_test import SSFusionFramework
CKPT=Path('/nas1/zhangzj26/HyperSIGMA_adapted/protocol_stage1/full_ft/best.pth')
CENTERS=Path('/nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_full48_cache.npz')
OUT=Path('/nas1/zhangzj26/HyperSIGMA_adapted/hypersigma_target_relation7_full48_fullft_cache.npz')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--output',type=Path,default=OUT); a=ap.parse_args(); dev=torch.device(a.device)
    target=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18.mat'))['ori_data']
    centers=np.load(CENTERS)['target_centers'].astype(np.int64)
    m=SSFusionFramework(img_size=33,in_channels=48,patch_size=2,classes=7,model_size='base')
    m.load_state_dict(torch.load(CKPT,map_location='cpu')['model'],strict=True); m.to(dev).eval()
    for p in m.parameters(): p.requires_grad_(False)
    pad=np.pad(target,((HALF,HALF),(HALF,HALF),(0,0)),mode='constant'); rel=[]
    with torch.no_grad():
      for st in range(0,len(centers),a.batch_size):
        cc=centers[st:st+a.batch_size]; x=np.empty((len(cc),48,IMG_SIZE,IMG_SIZE),np.float32)
        for j,(r,c) in enumerate(cc): x[j]=pad[r:r+IMG_SIZE,c:c+IMG_SIZE].transpose(2,0,1)
        spat,_,_,_=forward_parts(m,torch.from_numpy(x).to(dev)); fmap=spat[-1]
        fmap=torch.nn.functional.adaptive_avg_pool2d(fmap,(7,7)); z=torch.nn.functional.normalize(fmap.flatten(2).transpose(1,2),dim=2); rel.append((z@z.transpose(1,2)).cpu().numpy().astype(np.float16))
        if st % (a.batch_size*50)==0: print('processed',st,'/',len(centers),flush=True)
    rel=np.concatenate(rel)
    a.output.parent.mkdir(parents=True,exist_ok=True); np.savez(a.output,target_centers=centers,target_spatial_relation=rel,teacher_checkpoint=str(CKPT),teacher_map_shape=np.array([len(centers),768,32,32]),aligned_grid=np.array([7,7]))
    print('saved',a.output,rel.shape,flush=True)
if __name__=='__main__': main()
