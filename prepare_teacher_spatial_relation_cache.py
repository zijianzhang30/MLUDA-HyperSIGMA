"""Convert cached Full-FT teacher maps into fixed 7x7 relation targets."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

SOURCE=Path('/nas1/zhangzj26/HyperSIGMA_adapted/hypersigma_fspat_map_full48_fullft_cache.npz')
OUTPUT=Path('/nas1/zhangzj26/HyperSIGMA_adapted/hypersigma_fspat_relation7_full48_fullft_cache.npz')

def main():
    cache=np.load(SOURCE,allow_pickle=False)
    maps=cache['source_fspat_map']
    relations=[]
    with torch.no_grad():
        for start in range(0,len(maps),16):
            x=torch.from_numpy(maps[start:start+16].astype(np.float32))
            x=F.adaptive_avg_pool2d(x,(7,7)).flatten(2).transpose(1,2)
            x=F.normalize(x,dim=2)
            relations.append((x@x.transpose(1,2)).numpy().astype(np.float16))
    rel=np.concatenate(relations)
    np.savez(OUTPUT,source_centers=cache['source_centers'],source_spatial_relation=rel,
             teacher_checkpoint=cache['teacher_checkpoint'],teacher_map_shape=np.asarray(maps.shape),aligned_grid=np.asarray([7,7]))
    print('saved',OUTPUT,'teacher_map',maps.shape,'relation',rel.shape,rel.dtype)

if __name__=='__main__': main()
