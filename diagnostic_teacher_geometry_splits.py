"""Offline Full-FT HyperSIGMA F_spat geometry diagnostic per source split.

No training and no Houston18 labels are used.  The five source splits are
reconstructed with the same 180/class sampling helper used by MLUDA.  Their
teacher geometry is correlated with the already-computed F_spat-KD minus
matched-baseline target deltas (read from post-hoc JSON only).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import hdf5storage
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from MLUDA_hu_fspec_kd import paired_source_samples  # noqa: E402
import utils  # noqa: E402

OUT = Path('/nas1/zhangzj26/HyperSIGMA_adapted/seed_disentanglement')
CACHE = Path('/nas1/zhangzj26/HyperSIGMA_adapted/hypersigma_fspat_full48_fullft_cache.npz')
SEEDS = (1174, 1370, 1417, 1418, 1546)
CLASSES = tuple(range(7))
METRICS = ('oa', 'aa', 'kappa')


def split_features(cache, source_adapted, source_gt, split_seed):
    train_c, _, _, train_y, *_ = paired_source_samples(
        source_adapted, source_adapted, source_gt, split_seed)
    lookup = {(int(r), int(c)): i for i, (r, c) in enumerate(cache['source_centers'])}
    feat = np.stack([cache['source_fspat'][lookup[(int(r), int(c))]] for r, c in train_c]).astype(np.float64)
    return feat, train_y.astype(np.int64)


def geometry(feat, labels):
    z = feat / np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-12)
    gram = z @ z.T
    same = labels[:, None] == labels[None, :]
    offdiag = ~np.eye(len(labels), dtype=bool)
    same_off = same & offdiag
    diff = ~same
    intra = float(gram[same_off].mean())
    inter = float(gram[diff].mean())

    prototypes = np.stack([z[labels == c].mean(0) for c in CLASSES])
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
    pgram = prototypes @ prototypes.T
    pdiag = np.eye(7, dtype=bool)
    p_off = np.where(pdiag, -np.inf, pgram)
    nearest = p_off.argmax(1)
    nearest_cos = pgram[np.arange(7), nearest]
    nearest_dist = 1.0 - nearest_cos

    sample_sim = z @ prototypes.T
    pred = sample_sim.argmax(1)
    class_acc = np.asarray([(pred[labels == c] == c).mean() for c in CLASSES])
    confusion_rate = 1.0 - class_acc
    return {
        'intra_class_mean_cosine': intra,
        'inter_class_mean_cosine': inter,
        'margin_intra_minus_inter': intra - inter,
        'prototype_cosine_matrix': pgram.tolist(),
        'prototype_distance_matrix_1_minus_cosine': (1.0 - pgram).tolist(),
        'nearest_class': (nearest + 1).tolist(),
        'nearest_prototype_cosine': nearest_cos.tolist(),
        'nearest_prototype_distance': nearest_dist.tolist(),
        'nearest_prototype_label_agreement': class_acc.tolist(),
        'nearest_class_confusion_rate': confusion_rate.tolist(),
        'per_class': [
            {
                'class': c + 1,
                'nearest_class': int(nearest[c] + 1),
                'prototype_cosine_to_nearest': float(nearest_cos[c]),
                'prototype_distance_to_nearest': float(nearest_dist[c]),
                'nearest_prototype_accuracy': float(class_acc[c]),
                'nearest_class_confusion_rate': float(confusion_rate[c]),
            }
            for c in CLASSES
        ],
    }


def load_delta(split_seed):
    summary = json.loads((OUT / 'seed_disentanglement_summary.json').read_text())
    for row in summary['fixed_optimization_1174']['runs']:
        if row['split_seed'] == split_seed:
            return {key: float(row['delta'][key]) for key in METRICS}
    raise KeyError(split_seed)


def main():
    cache = np.load(CACHE, allow_pickle=False)
    source, source_gt = utils.load_data_houston(
        str(ROOT / 'datasets/Houston/Houston13.mat'),
        str(ROOT / 'datasets/Houston/Houston13_7gt.mat'))
    # Only source cube/labels are loaded. Houston18 is intentionally absent.
    source_adapted, _ = __import__('UtilsCMS', fromlist=['ILDA']).ILDA(source, source, 2, 0.009)

    rows = []
    detail = {}
    for split_seed in SEEDS:
        feat, labels = split_features(cache, source_adapted, source_gt, split_seed)
        g = geometry(feat, labels)
        d = load_delta(split_seed)
        row = {'split_seed': split_seed, **{k: g[k] for k in ('intra_class_mean_cosine', 'inter_class_mean_cosine', 'margin_intra_minus_inter')}, **{f'delta_{k}': d[k] for k in METRICS}, 'mean_nearest_prototype_accuracy': float(np.mean(g['nearest_prototype_label_agreement']))}
        rows.append(row); detail[str(split_seed)] = {'split_seed': split_seed, 'n_train': int(len(labels)), 'class_counts': np.bincount(labels, minlength=7).tolist(), 'deltas_fspat_minus_baseline': d, **g}

    corr = {}
    geom_keys = ('intra_class_mean_cosine', 'inter_class_mean_cosine', 'margin_intra_minus_inter', 'mean_nearest_prototype_accuracy')
    for geom_key in geom_keys:
        x = np.asarray([r[geom_key] for r in rows])
        corr[geom_key] = {}
        for metric in METRICS:
            y = np.asarray([r[f'delta_{metric}'] for r in rows])
            pear = stats.pearsonr(x, y); spear = stats.spearmanr(x, y)
            corr[geom_key][metric] = {'pearson_r': float(pear.statistic), 'pearson_p': float(pear.pvalue), 'spearman_rho': float(spear.statistic), 'spearman_p': float(spear.pvalue)}

    out = {'protocol': {'teacher_cache': str(CACHE), 'teacher_feature': 'Full-FT F_spat [768]', 'source_train': 'same 180/class split helper as MLUDA', 'source_splits': list(SEEDS), 'target_gt_used': False, 'correlation_note': 'n=5 exploratory correlations; no tuning'}, 'splits': detail, 'correlations': corr}
    (OUT / 'teacher_geometry_split_diagnostic.json').write_text(json.dumps(out, indent=2))
    with (OUT / 'teacher_geometry_split_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    with (OUT / 'teacher_geometry_per_class.csv').open('w', newline='') as f:
        fields = ['split_seed', 'class', 'nearest_class', 'prototype_cosine_to_nearest', 'prototype_distance_to_nearest', 'nearest_prototype_accuracy', 'nearest_class_confusion_rate']
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for split_seed in SEEDS:
            for item in detail[str(split_seed)]['per_class']:
                writer.writerow({'split_seed': split_seed, **item})
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
