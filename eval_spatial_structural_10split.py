"""Unified post-hoc evaluation for the 10 source-split structural-token runs."""
import csv, json, sys
from pathlib import Path
import hdf5storage, numpy as np, torch
from sklearn import metrics

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config_Houston import BATCH_SIZE, CLASS_NUM, nBand
from MLUDA_hu_fspec_kd import center_patches, paired_source_samples
from net2 import DSANSS
from UtilsCMS import ILDA
import utils

ADAPT = Path('/nas1/zhangzj26/HyperSIGMA_adapted')
STRUCT = ADAPT / 'spatial_structural_10split'
SPLITS = (1174, 1370, 1417, 1418, 1546, 1703, 1801, 1907, 2029, 2141)

def metric(y, p):
    cm = metrics.confusion_matrix(y, p, labels=np.arange(CLASS_NUM))
    pc = np.diag(cm) / np.maximum(cm.sum(1), 1)
    return {
        'oa': float((y == p).mean()),
        'aa': float(pc.mean()),
        'kappa': float(metrics.cohen_kappa_score(y, p, labels=np.arange(CLASS_NUM))),
        'per_class_accuracy': pc.tolist(),
        'prediction_distribution': np.bincount(p, minlength=CLASS_NUM).tolist(),
        'confusion_matrix': cm.tolist(),
    }

def evaluate(model, x, y, ref, dev):
    model.eval(); pred = []
    with torch.no_grad():
        for i in range(0, len(x), BATCH_SIZE):
            xb = torch.from_numpy(x[i:i+BATCH_SIZE]).to(dev)
            rb = ref[:len(xb)].to(dev)
            pred.append(model(rb, xb)[8].argmax(1).cpu().numpy())
    return metric(y, np.concatenate(pred))

def load_run(directory, stem, dev):
    ck = directory / f'{stem}_best.pth'
    hist_path = directory / f'{stem}_history.json'
    payload = torch.load(ck, map_location='cpu')
    model = DSANSS(nBand, 7, CLASS_NUM).to(dev)
    model.load_state_dict(payload['model'], strict=True)
    hist = json.loads(hist_path.read_text())
    best = payload.get('best') or max(hist, key=lambda z: z['val_acc'])
    return model, best, ck

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args(); dev = torch.device(a.device)
    src, sg = utils.load_data_houston(str(ROOT/'datasets/Houston/Houston13.mat'), str(ROOT/'datasets/Houston/Houston13_7gt.mat'))
    tgt = hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18.mat'))['ori_data']
    sa, ta = ILDA(src, tgt, 2, 0.009)
    # Target centers/labels are fixed by the existing Full48 cache and official GT map.
    cache = np.load(ADAPT/'hypersigma_fspat_full48_fullft_cache.npz')
    tc = cache['target_centers']; tx = center_patches(ta, tc, 7)
    tg = hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18_7gt.mat'))['map']
    ty = tg[tc[:,0], tc[:,1]].astype(np.int64) - 1
    rows = []
    for split in SPLITS:
        # The first five baseline runs live in seed_disentanglement; the
        # independent-split runs contain the remaining five.
        broot = ADAPT/'seed_disentanglement' if split in (1174,1370,1417,1418,1546) else ADAPT/'independent_split_validation'
        bdir = broot/'baseline/lambda_0_anneal'
        sroot = ADAPT/'spatial_structural_3split' if split in (1174, 1703, 2141) else STRUCT
        sdir = sroot/'lambda_0.1_anneal'
        stem = f'split_{split}_opt_1174'
        _, train_x, _, train_y, _, _, _, _ = paired_source_samples(sa, sa, sg, split)
        ref = torch.from_numpy(train_x[:BATCH_SIZE]).to(dev)
        bm, bb, bck = load_run(bdir, stem, dev)
        sm, sb, sck = load_run(sdir, stem, dev)
        br = evaluate(bm, tx, ty, ref, dev); sr = evaluate(sm, tx, ty, ref, dev)
        delta = {k: sr[k]-br[k] for k in ('oa','aa','kappa')}
        rows.append({'split': split, 'baseline': br, 'structural_token': sr,
                     'delta': delta, 'baseline_best_epoch': bb['epoch'],
                     'structural_best_epoch': sb['epoch'],
                     'baseline_source_val_accuracy': bb['val_acc'],
                     'structural_source_val_accuracy': sb['val_acc'],
                     'structural_best_rel_kd_loss': sb.get('train_rel_kd_loss'),
                     'baseline_checkpoint': str(bck), 'structural_checkpoint': str(sck)})
    def vals(method, key): return np.array([r[method][key] for r in rows], float)
    summary = {
        'protocol': {'splits': list(SPLITS), 'teacher': 'Full-FT HyperSIGMA F_spat',
                     'teacher_map': '[B,768,32,32] adaptive average pooled to 7x7',
                     'student_map': '[B,96,7,7]', 'relation': 'within-sample 49x49 position cosine MSE, diagonal excluded',
                     'lambda_schedule': 'epochs 1-20 0.1; 21-40 linear to 0; 41-100 0',
                     'target_iterator': 'recreated at each epoch', 'target_gt_used_for_training_or_selection': False},
        'runs': rows,
        'aggregate': {}
    }
    for method in ('baseline','structural_token'):
        summary['aggregate'][method] = {k: {'mean': float(vals(method,k).mean()), 'std': float(vals(method,k).std(ddof=0))}
                                        for k in ('oa','aa','kappa')}
    summary['aggregate']['delta'] = {k: {'mean': float(np.mean([r['delta'][k] for r in rows])),
                                         'std': float(np.std([r['delta'][k] for r in rows], ddof=0)),
                                         'win_count': int(sum(r['delta'][k] > 0 for r in rows)),
                                         'worst': float(min(r['delta'][k] for r in rows)),
                                         'best': float(max(r['delta'][k] for r in rows))}
                                      for k in ('oa','aa','kappa')}
    (STRUCT/'spatial_structural_10split_summary.json').write_text(json.dumps(summary, indent=2))
    with (STRUCT/'spatial_structural_10split_summary.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['split','baseline_oa','structural_oa','delta_oa','baseline_aa','structural_aa','delta_aa','baseline_kappa','structural_kappa','delta_kappa','baseline_best_epoch','structural_best_epoch','baseline_source_val','structural_source_val','structural_best_rel_kd_loss'])
        for r in rows:
            w.writerow([r['split'], r['baseline']['oa'], r['structural_token']['oa'], r['delta']['oa'], r['baseline']['aa'], r['structural_token']['aa'], r['delta']['aa'], r['baseline']['kappa'], r['structural_token']['kappa'], r['delta']['kappa'], r['baseline_best_epoch'], r['structural_best_epoch'], r['baseline_source_val_accuracy'], r['structural_source_val_accuracy'], r['structural_best_rel_kd_loss']])
    print(json.dumps(summary, indent=2))

if __name__ == '__main__': main()
