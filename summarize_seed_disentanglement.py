"""Summarize paired baseline/F_spat results along the two seed axes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path('/nas1/zhangzj26/HyperSIGMA_adapted/seed_disentanglement')
SEEDS = (1174, 1370, 1417, 1418, 1546)
METRICS = ('oa', 'aa', 'kappa')


def load(method, split_seed, opt_seed):
    lam = '0' if method == 'baseline' else '0.1'
    directory = OUT / method / f'lambda_{lam}_anneal'
    stem = f'split_{split_seed}_opt_{opt_seed}'
    target = json.loads((directory / f'{stem}_houston18_posthoc.json').read_text())
    history = json.loads((directory / f'{stem}_history.json').read_text())
    best = max(history, key=lambda row: row['val_acc'])
    return {
        'split_seed': split_seed,
        'optimization_seed': opt_seed,
        'checkpoint': str(directory / f'{stem}_best.pth'),
        'best_epoch': best['epoch'],
        'source_val_accuracy': best['val_acc'],
        'target': target,
    }


def summarize_axis(name, pairs):
    rows = []
    for split_seed, opt_seed in pairs:
        baseline = load('baseline', split_seed, opt_seed)
        kd = load('fspat', split_seed, opt_seed)
        delta = {key: kd['target'][key] - baseline['target'][key] for key in METRICS}
        rows.append({'split_seed': split_seed, 'optimization_seed': opt_seed,
                     'baseline': baseline, 'fspat': kd, 'delta': delta})
    result = {'axis': name, 'runs': rows, 'methods': {}}
    for method in ('baseline', 'fspat'):
        values = np.asarray([[row[method]['target'][key] for key in METRICS] for row in rows])
        result['methods'][method] = {
            'mean': dict(zip(METRICS, values.mean(0).tolist())),
            'std_population': dict(zip(METRICS, values.std(0, ddof=0).tolist())),
        }
    delta_values = np.asarray([[row['delta'][key] for key in METRICS] for row in rows])
    result['delta'] = {
        'mean': dict(zip(METRICS, delta_values.mean(0).tolist())),
        'std_population': dict(zip(METRICS, delta_values.std(0, ddof=0).tolist())),
        'win_count': {key: int(np.sum(delta_values[:, i] > 0)) for i, key in enumerate(METRICS)},
    }
    return result


def main():
    summary = {
        'protocol': {
            'teacher': 'Full-FT HyperSIGMA Full48 F_spat',
            'matched_baseline': 'same spatial projection/forward with lambda_rel=0',
            'target_iterator': 'recreated at the beginning of every epoch',
            'schedule': 'epochs 1-20: 0.1; 21-40: linear to 0; 41-100: 0',
            'target_gt_used_for_training_or_selection': False,
            'std_definition': 'population std (ddof=0)',
        },
        'fixed_split_1174': summarize_axis('optimization randomness', [(1174, seed) for seed in SEEDS]),
        'fixed_optimization_1174': summarize_axis('source subset randomness', [(seed, 1174) for seed in SEEDS]),
    }
    (OUT / 'seed_disentanglement_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
