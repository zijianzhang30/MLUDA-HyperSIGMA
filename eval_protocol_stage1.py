"""Post-hoc metrics for protocol_stage1 artifacts; never used during training."""
import argparse, json, pickle, sys
from pathlib import Path
import hdf5storage, numpy as np, torch
from sklearn.metrics import confusion_matrix, cohen_kappa_score
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'third_party/HyperSIGMA/ImageClassification'))
import utils
from hypersigma_teacher_smoke_test import SSFusionFramework
IMG=33; HALF=16; K=7
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--artifact',type=Path,required=True); ap.add_argument('--bands',type=int,required=True); ap.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu'); ap.add_argument('--batch-size',type=int,default=64); a=ap.parse_args(); dev=torch.device(a.device)
 ck=torch.load(a.artifact/'stage1_best.pth',map_location='cpu'); cube=utils.load_data_houston(str(ROOT/'datasets/Houston/Houston18.mat'),str(ROOT/'datasets/Houston/Houston18_7gt.mat'))[0]; gt=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18_7gt.mat'))['map']
 if a.bands==30:
  with open(a.artifact/'source_pca.pkl','rb') as f:p=pickle.load(f)
  cube=p.transform(cube.reshape(-1,cube.shape[-1]).astype(np.float32)).reshape(cube.shape[0],cube.shape[1],30).astype(np.float32)
 cube=np.asarray(cube,np.float32); pad=np.pad(cube,((HALF,HALF),(HALF,HALF),(0,0)),mode='constant'); ids=np.flatnonzero(gt.reshape(-1)>0); y=gt.reshape(-1)[ids].astype(np.int64)-1
 m=SSFusionFramework(img_size=IMG,in_channels=a.bands,patch_size=2,classes=K,model_size='base'); m.load_state_dict(ck['model']); m.to(dev).eval(); pred=[]
 with torch.no_grad():
  for s in range(0,len(ids),a.batch_size):
   xb=np.empty((min(a.batch_size,len(ids)-s),a.bands,IMG,IMG),np.float32)
   for j,flat in enumerate(ids[s:s+a.batch_size]):
    r,c=np.unravel_index(int(flat),gt.shape); xb[j]=pad[r:r+IMG,c:c+IMG].transpose(2,0,1)
   pred.append(m(torch.from_numpy(xb).to(dev)).argmax(1).cpu().numpy())
 pred=np.concatenate(pred); cm=confusion_matrix(y,pred,labels=np.arange(K)); pc=np.diag(cm)/cm.sum(1); out={'protocol':'post-hoc only; Houston18 GT excluded from training/selection','bands':a.bands,'n':int(len(y)),'oa':float((y==pred).mean()),'aa':float(pc.mean()),'kappa':float(cohen_kappa_score(y,pred)),'per_class_accuracy':pc.tolist(),'confusion_matrix':cm.tolist()}; (a.artifact/'houston18_metrics.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
