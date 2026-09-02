"""Common-split nearest-prototype probes for three fixed representations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hdf5storage
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config_Houston import BATCH_SIZE, CLASS_NUM, nBand  # noqa: E402
from net2 import DSANSS  # noqa: E402
from UtilsCMS import ILDA  # noqa: E402
import utils  # noqa: E402
from MLUDA_hu_fspec_kd import center_patches, paired_source_samples  # noqa: E402

CACHE = Path("/nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_full48_cache.npz")
MLUDA_CKPT = Path("/nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_kd/lambda_0/seed_1174_best.pth")
OUT = Path("/nas1/zhangzj26/HyperSIGMA_adapted/diagnostics/feature_probes_seed1174.json")


def normalize(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def probe_metrics(train_f, train_y, eval_f, eval_y):
    t = normalize(train_f); e = normalize(eval_f)
    prototypes = np.stack([normalize(train_f[train_y == c]).mean(0) for c in range(CLASS_NUM)])
    prototypes = normalize(prototypes)
    pred = (e @ prototypes.T).argmax(1)
    cm = confusion_matrix(eval_y, pred, labels=np.arange(CLASS_NUM))
    pc = np.diag(cm) / np.maximum(cm.sum(1), 1)
    return {
        "oa": float(np.mean(pred == eval_y)), "aa": float(pc.mean()),
        "kappa": float(cohen_kappa_score(eval_y, pred, labels=np.arange(CLASS_NUM))),
        "per_class_accuracy": pc.tolist(),
        "prediction_distribution_zero_based": np.bincount(pred, minlength=CLASS_NUM).tolist(),
        "confusion_matrix_true_rows_pred_columns": cm.tolist(),
    }


def cosine_stats(a, ya, b, yb, max_pairs=200000):
    """Estimate same/different cosine means without an O(N^2) target matrix."""
    an, bn = normalize(a), normalize(b)
    rng = np.random.RandomState(20260902)
    n, m = len(an), len(bn)
    total = n * m
    count = min(max_pairs, total)
    ii = rng.randint(0, n, size=count); jj = rng.randint(0, m, size=count)
    if a is b or (a.shape == b.shape and np.array_equal(a, b)):
        keep = ii != jj
        ii, jj = ii[keep], jj[keep]
    vals = np.sum(an[ii] * bn[jj], axis=1)
    same = ya[ii] == yb[jj]
    same_vals, diff_vals = vals[same], vals[~same]
    return {"same": float(np.nanmean(same_vals)), "different": float(np.nanmean(diff_vals)),
            "margin_same_minus_different": float(np.nanmean(same_vals) - np.nanmean(diff_vals))}


def extract_mluDA(model, x, device, batch_size, pair_mode, source_ref_x=None):
    """Return pre-MBCA spectral [N,192] and post-MBCA [N,288].

    Source samples are paired with themselves; target samples are paired with
    the fixed first source batch, matching the baseline target evaluator's
    cross-attention convention.  This pairing is stated explicitly so the
    probe is reproducible and does not use labels.
    """
    model.eval(); source_ref = (x[:BATCH_SIZE] if source_ref_x is None else source_ref_x[:BATCH_SIZE])
    pre, post = [], []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device)
            if pair_mode == "source":
                ref = xb
            else:
                # Cross-attention requires equal batch dimensions. Repeat the
                # fixed source reference deterministically for larger target
                # batches; this affects only feature extraction, not training.
                ref_np = np.resize(source_ref, (len(xb),) + source_ref.shape[1:])
                ref = torch.from_numpy(ref_np).to(device)
            result = model.forward_with_spectral(ref, xb)
            post.append(result[5 if pair_mode == "target" else 0].cpu().numpy())
            pre.append(result[11 if pair_mode == "target" else 10].cpu().numpy())
    return np.concatenate(pre), np.concatenate(post)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:2" if torch.cuda.is_available() else "cpu"); ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--output", type=Path, default=OUT); args = ap.parse_args()
    device = torch.device(args.device)
    cache = np.load(CACHE, allow_pickle=False)
    source, source_gt = utils.load_data_houston(str(ROOT / "datasets/Houston/Houston13.mat"), str(ROOT / "datasets/Houston/Houston13_7gt.mat"))
    target, target_gt = utils.load_data_houston(str(ROOT / "datasets/Houston/Houston18.mat"), str(ROOT / "datasets/Houston/Houston18_7gt.mat"))
    data_s, data_t = ILDA(source, target, 2, 0.009)
    train_c, train_x, _, train_y, val_c, val_x, _, val_y = paired_source_samples(data_s, data_s, source_gt, 1174)
    rows, cols = np.nonzero(target_gt > 0); target_c = np.stack([rows, cols], 1).astype(np.int64); target_y = target_gt[rows, cols].astype(np.int64) - 1
    target_x = center_patches(data_t, target_c, 7)
    source_map = {(int(r), int(c)): i for i, (r, c) in enumerate(cache["source_centers"])}
    fs = cache["source_fspec"]
    source_train_spec = np.stack([fs[source_map[(int(r), int(c))]] for r, c in train_c]).astype(np.float32)
    source_val_spec = np.stack([fs[source_map[(int(r), int(c))]] for r, c in val_c]).astype(np.float32)
    target_spec = cache["target_fspec"].astype(np.float32)
    # Fixed MLUDA lambda=0 control; no model state is changed.
    ck = torch.load(MLUDA_CKPT, map_location="cpu"); model = DSANSS(nBand, 7, CLASS_NUM).to(device); model.load_state_dict(ck["model"], strict=True)
    tr_pre, tr_post = extract_mluDA(model, train_x, device, args.batch_size, "source")
    va_pre, va_post = extract_mluDA(model, val_x, device, args.batch_size, "source")
    te_pre, te_post = extract_mluDA(model, target_x, device, args.batch_size, "target", source_ref_x=train_x)
    reps = {
        "HyperSIGMA_F_spec": (source_train_spec, source_val_spec, target_spec),
        "MLUDA_pre_MBCA_spectral": (tr_pre, va_pre, te_pre),
        "MLUDA_post_MBCA": (tr_post, va_post, te_post),
    }
    output = {"protocol": "offline nearest-prototype probe; prototypes use only fixed Houston13 source-train split (seed 1174, 180/class)", "source_val": "remaining labeled Houston13 pixels from the same MLUDA paired_source_samples split; not used to form prototypes (random, not the separate spatial-disjoint teacher split)", "target_gt_used_for_training_or_selection": False, "label_mapping": "1..7 -> 0..6; background excluded", "mluDA_checkpoint": str(MLUDA_CKPT), "representations": {}}
    labels = {"train": train_y, "val": val_y, "target": target_y}
    for name, (tr, va, te) in reps.items():
        output["representations"][name] = {
            "feature_shapes": {"train": list(tr.shape), "source_val": list(va.shape), "target": list(te.shape)},
            "source_disjoint_val": probe_metrics(tr, train_y, va, val_y),
            "houston18": probe_metrics(tr, train_y, te, target_y),
            "cosine": {
                "source_train": cosine_stats(tr, train_y, tr, train_y),
                "source_val": cosine_stats(va, val_y, va, val_y),
                "target": cosine_stats(te, target_y, te, target_y),
                "source_target": cosine_stats(tr, train_y, te, target_y),
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(output, indent=2)); print(json.dumps(output, indent=2))


if __name__ == "__main__": main()
