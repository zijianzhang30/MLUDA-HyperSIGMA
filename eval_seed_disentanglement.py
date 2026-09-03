"""Post-hoc Houston18 evaluation for the split/optimization seed diagnostic."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import hdf5storage
import numpy as np
import torch
from sklearn import metrics

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config_Houston import BATCH_SIZE, CLASS_NUM, nBand  # noqa: E402
from MLUDA_hu_fspec_kd import center_patches, paired_source_samples  # noqa: E402
from net2 import DSANSS  # noqa: E402
from UtilsCMS import ILDA  # noqa: E402
import utils  # noqa: E402

OUT = Path('/nas1/zhangzj26/HyperSIGMA_adapted/seed_disentanglement')
CACHE = Path('/nas1/zhangzj26/HyperSIGMA_adapted/hypersigma_fspat_full48_fullft_cache.npz')
SEEDS = (1174, 1370, 1417, 1418, 1546)
PAIRS = tuple([(1174, seed) for seed in SEEDS] + [(seed, 1174) for seed in SEEDS if seed != 1174])


def evaluate(model, target_x, target_y, source_ref, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(target_x), BATCH_SIZE):
            target = torch.from_numpy(target_x[start:start + BATCH_SIZE]).to(device)
            source = source_ref[:len(target)].to(device)
            predictions.append(model(source, target)[8].argmax(1).cpu().numpy())
    pred = np.concatenate(predictions)
    cm = metrics.confusion_matrix(target_y, pred, labels=np.arange(CLASS_NUM))
    per_class = np.diag(cm) / np.maximum(cm.sum(1), 1)
    return {
        'n': int(len(target_y)),
        'oa': float(np.mean(target_y == pred)),
        'aa': float(np.mean(per_class)),
        'kappa': float(metrics.cohen_kappa_score(target_y, pred, labels=np.arange(CLASS_NUM))),
        'per_class_accuracy': per_class.tolist(),
        'prediction_distribution': np.bincount(pred, minlength=CLASS_NUM).tolist(),
        'confusion_matrix': cm.tolist(),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    device = torch.device(args.device)

    cache = np.load(CACHE, allow_pickle=False)
    target_centers = cache['target_centers']
    source, source_gt = utils.load_data_houston(
        str(ROOT / 'datasets/Houston/Houston13.mat'),
        str(ROOT / 'datasets/Houston/Houston13_7gt.mat'))
    target = hdf5storage.loadmat(str(ROOT / 'datasets/Houston/Houston18.mat'))['ori_data']
    source_adapted, target_adapted = ILDA(source, target, 2, 0.009)
    target_x = center_patches(target_adapted, target_centers, 7)

    # Labels are opened only after every checkpoint has already been selected
    # solely by Houston13 source validation.
    target_gt = hdf5storage.loadmat(str(ROOT / 'datasets/Houston/Houston18_7gt.mat'))['map']
    target_y = target_gt[target_centers[:, 0], target_centers[:, 1]].astype(np.int64) - 1

    for split_seed, opt_seed in PAIRS:
        _, train_x, _, _, _, _, _, _ = paired_source_samples(
            source_adapted, source_adapted, source_gt, split_seed)
        source_ref = torch.from_numpy(train_x[:BATCH_SIZE])
        for method, lam in (('baseline', '0'), ('fspat', '0.1')):
            directory = args.output / method / f'lambda_{lam}_anneal'
            stem = f'split_{split_seed}_opt_{opt_seed}'
            checkpoint = directory / f'{stem}_best.pth'
            result_path = directory / f'{stem}_houston18_posthoc.json'
            checkpoint_data = torch.load(checkpoint, map_location='cpu')
            if checkpoint_data.get('split_seed') != split_seed or checkpoint_data.get('optimization_seed') != opt_seed:
                raise RuntimeError(f'seed metadata mismatch in {checkpoint}')
            model = DSANSS(nBand, 7, CLASS_NUM).to(device)
            model.load_state_dict(checkpoint_data['model'], strict=True)
            result = evaluate(model, target_x, target_y, source_ref, device)
            result.update({
                'method': method,
                'split_seed': split_seed,
                'optimization_seed': opt_seed,
                'checkpoint': str(checkpoint),
                'checkpoint_selected_by': 'Houston13 source validation accuracy only',
                'target_gt_used_for_training_or_selection': False,
            })
            result_path.write_text(json.dumps(result, indent=2))
            print(json.dumps({k: result[k] for k in ('method', 'split_seed', 'optimization_seed', 'oa', 'aa', 'kappa')}))


if __name__ == '__main__':
    main()
