"""Cache Full-FT HyperSIGMA pre-pooling spatial maps for structural KD."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'third_party/HyperSIGMA/ImageClassification'))
import utils
from UtilsCMS import ILDA
from hypersigma_stage1_protocol import forward_parts,HALF,IMG_SIZE
from hypersigma_teacher_smoke_test import SSFusionFramework
CKPT=Path('/nas1/zhangzj26/HyperSIGMA_adapted/protocol_stage1/full_ft/best.pth')
OUT=Path('/nas1/zhangzj26/HyperSIGMA_adapted/hypersigma_fspat_map_full48_fullft_cache.npz')
def extract(m,cube,centers,dev,bs):
 pad=np.pad(cube,((HALF,HALF),(HALF,HALF),(0,0)),mode='constant'); arr=[]
 with torch.no_grad():
  for st in range(0,len(centers),bs):
   cc=centers[st:st+bs]; x=np.empty((len(cc),48,IMG_SIZE,IMG_SIZE),np.float32)
   for j,(r,c) in enumerate(cc): x[j]=pad[r:r+IMG_SIZE,c:c+IMG_SIZE].transpose(2,0,1)
   spat,_,_,_=forward_parts(m,torch.from_numpy(x).to(dev)); arr.append(spat[-1].cpu().numpy().astype(np.float16))
 return np.concatenate(arr)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0');ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--output',type=Path,default=OUT);a=ap.parse_args();dev=torch.device(a.device)
 src,sg=utils.load_data_houston(str(ROOT/'datasets/Houston/Houston13.mat'),str(ROOT/'datasets/Houston/Houston13_7gt.mat'));sc=np.argwhere(sg>0).astype(np.int64);m=SSFusionFramework(img_size=33,in_channels=48,patch_size=2,classes=7,model_size='base');m.load_state_dict(torch.load(CKPT,map_location='cpu')['model'],strict=True);m.to(dev).eval();[p.requires_grad_(False) for p in m.parameters()];sf=extract(m,src,sc,dev,a.batch_size);a.output.parent.mkdir(parents=True,exist_ok=True);np.savez(a.output,source_centers=sc,source_fspat_map=sf,teacher_checkpoint=str(CKPT));print('saved',a.output,sf.shape)
if __name__=='__main__':main()
