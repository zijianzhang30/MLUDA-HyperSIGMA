"""Post-hoc target audit for MLUDA F_spec KD runs.

This script reads Houston18 labels only for final metrics.  It never changes
checkpoints or selects epochs; those are fixed by source validation in the
training script.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import hdf5storage
import numpy as np
import torch
from sklearn import metrics

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config_Houston import BATCH_SIZE, CLASS_NUM, HalfWidth, nBand, seeds  # noqa: E402
from net2 import DSANSS  # noqa: E402
from MLUDA_hu_fspec_kd import center_patches, paired_source_samples  # noqa: E402
import utils  # noqa: E402
from UtilsCMS import ILDA  # noqa: E402

CACHE = Path("/nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_full48_cache.npz")
ROOT_OUT = Path("/nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_kd")


def target_metric(y, pred):
    cm = metrics.confusion_matrix(y, pred, labels=np.arange(CLASS_NUM)); pc = np.diag(cm) / np.maximum(cm.sum(1), 1)
    return {"n": int(len(y)), "oa": float(np.mean(y == pred)), "aa": float(np.mean(pc)), "kappa": float(metrics.cohen_kappa_score(y, pred, labels=np.arange(CLASS_NUM))), "per_class_accuracy": pc.tolist(), "prediction_distribution": np.bincount(pred, minlength=CLASS_NUM).tolist(), "confusion_matrix": cm.tolist()}


def eval_one(model, target_x, target_y, source_ref, device):
    model.eval(); pred = []
    with torch.no_grad():
        for start in range(0, len(target_x), BATCH_SIZE):
            xb = torch.from_numpy(target_x[start:start + BATCH_SIZE]).to(device)
            ref = source_ref[:len(xb)].to(device)
            pred.append(model(ref, xb)[8].argmax(1).cpu().numpy())
    return target_metric(target_y, np.concatenate(pred))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu"); ap.add_argument("--lambdas", type=float, nargs="*", default=(0.0, 0.05, 0.1, 0.2)); ap.add_argument("--seeds", type=int, nargs="*", default=seeds); ap.add_argument("--cache", type=Path, default=CACHE)
    args = ap.parse_args(); device = torch.device(args.device)
    cache = np.load(args.cache, allow_pickle=False); target_centers = cache["target_centers"]
    source, src_gt = utils.load_data_houston(str(ROOT / "datasets/Houston/Houston13.mat"), str(ROOT / "datasets/Houston/Houston13_7gt.mat")); target = hdf5storage.loadmat(str(ROOT / "datasets/Houston/Houston18.mat"))["ori_data"]
    data_s, data_t = ILDA(source, target, 2, 0.009)
    target_x = center_patches(data_t, target_centers, 7)
    # Target GT is opened only now, after all checkpoint selection is complete.
    target_gt = hdf5storage.loadmat(str(ROOT / "datasets/Houston/Houston18_7gt.mat"))["map"]
    target_y = target_gt[target_centers[:, 0], target_centers[:, 1]].astype(np.int64) - 1
    summary = {}
    for lam in args.lambdas:
        directory = ROOT_OUT / f"lambda_{lam:g}"; runs = []
        for seed in args.seeds:
            ckpt_path = directory / f"seed_{seed}_best.pth"
            if not ckpt_path.exists():
                print(f"skip missing {ckpt_path}"); continue
            _, train_x, _, _, _, _, _, _ = paired_source_samples(data_s, data_s, src_gt, seed)
            ck = torch.load(ckpt_path, map_location="cpu"); model = DSANSS(nBand, 7, CLASS_NUM).to(device); model.load_state_dict(ck["model"], strict=True)
            result = eval_one(model, target_x, target_y, torch.from_numpy(train_x[:BATCH_SIZE]), device)
            hist = json.loads((directory / f"seed_{seed}_history.json").read_text()); best = max(hist, key=lambda row: row["val_acc"])
            runs.append({"seed": seed, "source_best": best, "target": result}); print(json.dumps({"lambda_kd": lam, **runs[-1]}))
        if runs:
            vals = np.asarray([[r["target"][k] for k in ("oa", "aa", "kappa")] for r in runs]); summary[f"lambda_{lam:g}"] = {"runs": runs, "mean": dict(zip(("oa", "aa", "kappa"), vals.mean(0).tolist())), "std": dict(zip(("oa", "aa", "kappa"), vals.std(0).tolist()))}
    (ROOT_OUT / "posthoc_target_summary.json").write_text(json.dumps(summary, indent=2)); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
