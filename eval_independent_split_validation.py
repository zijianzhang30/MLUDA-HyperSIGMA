"""Evaluate new source splits and correlate fixed reliability metrics with KD delta."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import hdf5storage
import numpy as np
from scipy import stats
from sklearn import metrics
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config_Houston import BATCH_SIZE, CLASS_NUM, nBand
from MLUDA_hu_fspec_kd import center_patches, paired_source_samples
from net2 import DSANSS
from UtilsCMS import ILDA
import utils

OUT = Path('/nas1/zhangzj26/HyperSIGMA_adapted/independent_split_validation')
CACHE = Path('/nas1/zhangzj26/HyperSIGMA_adapted/hypersigma_fspat_full48_fullft_cache.npz')
SPLITS = (1703, 1801, 1907, 2029, 2141)
OPT_SEED = 1174
METRICS = ('oa', 'aa', 'kappa')


def target_metric(y, pred):
    cm = metrics.confusion_matrix(y, pred, labels=np.arange(CLASS_NUM))
    pc = np.diag(cm) / np.maximum(cm.sum(1), 1)
    return {
        'oa': float(np.mean(y == pred)), 'aa': float(pc.mean()),
        'kappa': float(metrics.cohen_kappa_score(y, pred, labels=np.arange(CLASS_NUM))),
        'per_class_accuracy': pc.tolist(),
        'prediction_distribution': np.bincount(pred, minlength=CLASS_NUM).tolist(),
        'confusion_matrix': cm.tolist(),
    }


def eval_model(model, target_x, target_y, source_ref, device):
    model.eval(); preds = []
    with torch.no_grad():
        for start in range(0, len(target_x), BATCH_SIZE):
            xb = torch.from_numpy(target_x[start:start + BATCH_SIZE]).to(device)
            ref = source_ref[:len(xb)].to(device)
            preds.append(model(ref, xb)[8].argmax(1).cpu().numpy())
    return target_metric(target_y, np.concatenate(preds))


def reliability(cache, source_adapted, source_gt, split_seed):
    train_c, _, _, train_y, *_ = paired_source_samples(
        source_adapted, source_adapted, source_gt, split_seed)
    lookup = {(int(r), int(c)): i for i, (r, c) in enumerate(cache['source_centers'])}
    feats = np.stack([cache['source_fspat'][lookup[(int(r), int(c))]] for r, c in train_c]).astype(np.float64)
    labels = train_y.astype(np.int64)
    z = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)
    prototypes = np.stack([z[labels == c].mean(0) for c in range(7)])
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
    sim = z @ prototypes.T
    pred = sim.argmax(1)
    agreement = np.asarray([(pred[labels == c] == c).mean() for c in range(7)])
    margins = np.asarray([
        np.mean(sim[labels == c, c] - np.max(np.delete(sim[labels == c], c, axis=1), axis=1))
        for c in range(7)
    ])
    return {
        'n_train': int(len(labels)),
        'class_counts': np.bincount(labels, minlength=7).tolist(),
        'nearest_prototype_agreement': agreement.tolist(),
        'true_vs_strongest_wrong_similarity_margin': margins.tolist(),
        'mean_nearest_prototype_agreement': float(agreement.mean()),
        'mean_true_vs_wrong_margin': float(margins.mean()),
        'per_class': [
            {'class': c + 1, 'nearest_prototype_agreement': float(agreement[c]),
             'true_vs_strongest_wrong_margin': float(margins[c])}
            for c in range(7)
        ],
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args(); device = torch.device(args.device)
    cache = np.load(CACHE, allow_pickle=False)
    source, source_gt = utils.load_data_houston(
        str(ROOT / 'datasets/Houston/Houston13.mat'),
        str(ROOT / 'datasets/Houston/Houston13_7gt.mat'))
    target = hdf5storage.loadmat(str(ROOT / 'datasets/Houston/Houston18.mat'))['ori_data']
    source_adapted, target_adapted = ILDA(source, target, 2, 0.009)
    target_centers = cache['target_centers']
    target_x = center_patches(target_adapted, target_centers, 7)
    target_gt = hdf5storage.loadmat(str(ROOT / 'datasets/Houston/Houston18_7gt.mat'))['map']
    target_y = target_gt[target_centers[:, 0], target_centers[:, 1]].astype(np.int64) - 1

    rows = []
    for split_seed in SPLITS:
        _, train_x, _, _, *_ = paired_source_samples(source_adapted, source_adapted, source_gt, split_seed)
        source_ref = torch.from_numpy(train_x[:BATCH_SIZE])
        rel = reliability(cache, source_adapted, source_gt, split_seed)
        results = {}
        for method, lam in (('baseline', '0'), ('fspat', '0.1')):
            directory = OUT / method / f'lambda_{lam}_anneal'
            stem = f'split_{split_seed}_opt_{OPT_SEED}'
            checkpoint = directory / f'{stem}_best.pth'
            if not checkpoint.exists():
                # Allow partial post-hoc evaluation while the final training
                # configuration is still running; the final pass will fill it.
                continue
            history = json.loads((directory / f'{stem}_history.json').read_text())
            payload = torch.load(checkpoint, map_location='cpu')
            model = DSANSS(nBand, 7, CLASS_NUM).to(device)
            model.load_state_dict(payload['model'], strict=True)
            metric = eval_model(model, target_x, target_y, source_ref, device)
            best = max(history, key=lambda row: row['val_acc'])
            results[method] = {'target': metric, 'best_epoch': best['epoch'],
                               'source_val_accuracy': best['val_acc'], 'checkpoint': str(checkpoint)}
        if set(results) != {'baseline', 'fspat'}:
            continue
        delta = {k: results['fspat']['target'][k] - results['baseline']['target'][k] for k in METRICS}
        rows.append({'split_seed': split_seed, **rel, 'delta': delta,
                     'baseline': results['baseline'], 'fspat': results['fspat']})

    correlations = {}
    for gkey in ('mean_nearest_prototype_agreement', 'mean_true_vs_wrong_margin'):
        x = np.asarray([r[gkey] for r in rows]); correlations[gkey] = {}
        for metric in METRICS:
            y = np.asarray([r['delta'][metric] for r in rows])
            p = stats.pearsonr(x, y); s = stats.spearmanr(x, y)
            correlations[gkey][metric] = {
                'pearson_r': float(p.statistic), 'pearson_p': float(p.pvalue),
                'spearman_rho': float(s.statistic), 'spearman_p': float(s.pvalue),
            }
    summary = {'protocol': {'new_splits': list(SPLITS), 'optimization_seed': OPT_SEED,
                            'teacher': 'Full-FT HyperSIGMA F_spat',
                            'reliability_fixed_before_training': True,
                            'target_gt_used_for_training_or_selection': False,
                            'target_iterator': 'reset each epoch',
                            'schedule': '1-20 .1, 21-40 linear to 0, 41-100 0'},
               'runs': rows, 'correlations': correlations}
    (OUT / 'independent_split_validation_summary.json').write_text(json.dumps(summary, indent=2))
    fields = ['split_seed', 'mean_nearest_prototype_agreement', 'mean_true_vs_wrong_margin',
              'delta_oa', 'delta_aa', 'delta_kappa']
    with (OUT / 'independent_split_validation_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for r in rows:
            writer.writerow({'split_seed': r['split_seed'],
                             'mean_nearest_prototype_agreement': r['mean_nearest_prototype_agreement'],
                             'mean_true_vs_wrong_margin': r['mean_true_vs_wrong_margin'],
                             **{f'delta_{m}': r['delta'][m] for m in METRICS}})
    with (OUT / 'independent_split_validation_per_class.csv').open('w', newline='') as f:
        fields = ['split_seed', 'class', 'nearest_prototype_agreement', 'true_vs_strongest_wrong_margin']
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for r in rows:
            for item in r['per_class']:
                writer.writerow({'split_seed': r['split_seed'], **item})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
