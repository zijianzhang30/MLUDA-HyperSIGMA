"""Unified post-hoc audit for Stage1 / Partial FT / Full FT.

Houston18 is opened only by this audit after checkpoints are fixed.  No value
from its labels is used for training, checkpoint selection, or hyperparameters.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import hdf5storage
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "HyperSIGMA" / "ImageClassification"))
import utils  # noqa: E402
from hypersigma_stage1_protocol import CLASSES, HALF, IMG_SIZE, patches  # noqa: E402
from hypersigma_teacher_smoke_test import SSFusionFramework  # noqa: E402


def metric_dict(y, pred):
    cm = confusion_matrix(y, pred, labels=np.arange(CLASSES))
    pc = np.diag(cm) / np.maximum(cm.sum(1), 1)
    return {
        "n": int(len(y)),
        "oa": float(np.mean(y == pred)),
        "aa": float(np.mean(pc)),
        "kappa": float(cohen_kappa_score(y, pred, labels=np.arange(CLASSES))),
        "per_class_accuracy": pc.tolist(),
        "prediction_distribution": np.bincount(pred, minlength=CLASSES).tolist(),
        "confusion_matrix": cm.tolist(),
    }


def forward_features(model, x):
    b = x.shape[0]
    spat = model.spat_encoder(x)
    spec = model.spec_encoder(x)[0]
    pooled = model.pool(spec).view(b, -1)
    weights = [layer(pooled).view(b, -1, 1, 1) for layer in (model.fc_spec1, model.fc_spec2, model.fc_spec3, model.fc_spec4)]
    reduced = [layer(f) for layer, f in zip((model.DR1, model.DR2, model.DR3, model.DR4), spat)]
    sem = torch.cat([F.adaptive_avg_pool2d((1 + w) * f, 1).flatten(1) for w, f in zip(weights, reduced)], dim=1)
    logits = model.classifier(sem)
    f_spat = F.adaptive_avg_pool2d(spat[-1], 1).flatten(1)
    f_spec = spec.mean(1)
    return f_spat, f_spec, sem, logits


def extract_centers(model, cube, centers, device, batch_size=64):
    x = patches(cube, np.asarray(centers), cube.shape[-1])
    out = {"spat": [], "spec": [], "sem": [], "logits": []}
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            fs, fe, fm, logits = forward_features(model, torch.from_numpy(x[start:start + batch_size]).to(device))
            out["spat"].append(fs.cpu().numpy()); out["spec"].append(fe.cpu().numpy()); out["sem"].append(fm.cpu().numpy()); out["logits"].append(logits.cpu().numpy())
    return {key: np.concatenate(value) for key, value in out.items()}


def probe_centers(gt, seed=20260901):
    rng = np.random.RandomState(seed)
    centers = []
    for cls in range(1, CLASSES + 1):
        ids = np.flatnonzero(gt.reshape(-1) == cls)
        centers.extend([np.unravel_index(int(i), gt.shape) for i in rng.permutation(ids)[:100]])
    return np.asarray(centers, dtype=np.int64)


def pairwise(a, ya, b=None, yb=None):
    cross = b is not None
    if b is None:
        b, yb = a, ya
    aa = F.normalize(torch.from_numpy(a), dim=1).numpy()
    bb = F.normalize(torch.from_numpy(b), dim=1).numpy()
    matrix = aa @ bb.T
    same = ya[:, None] == yb[None, :]
    if not cross:
        np.fill_diagonal(same, False)
        np.fill_diagonal(matrix, np.nan)
    return float(np.nanmean(matrix[same])), float(np.nanmean(matrix[~same]))


def probe_features(model, src_cube, src_gt, tgt_cube, tgt_gt, device):
    src_c = probe_centers(src_gt); tgt_c = probe_centers(tgt_gt)
    src = extract_centers(model, src_cube, src_c, device)
    tgt = extract_centers(model, tgt_cube, tgt_c, device)
    sy = src_gt[src_c[:, 0], src_c[:, 1]].astype(np.int64) - 1
    ty = tgt_gt[tgt_c[:, 0], tgt_c[:, 1]].astype(np.int64) - 1
    result = {}
    for name in ("spat", "spec", "sem"):
        ss, sd = pairwise(src[name], sy)
        ts, td = pairwise(tgt[name], ty)
        cs, cd = pairwise(src[name], sy, tgt[name], ty)
        result[name] = {"source_same": ss, "source_diff": sd, "target_same": ts, "target_diff": td, "cross_same": cs, "cross_diff": cd,
                        "source_margin": ss - sd, "target_margin": ts - td, "cross_margin": cs - cd}
    return result


def source_prototype(train_features, train_y, features, labels):
    f = F.normalize(torch.from_numpy(train_features), dim=1).numpy()
    prototypes = []
    for cls in range(CLASSES):
        prototype = f[train_y == cls].mean(0)
        prototypes.append(prototype / max(np.linalg.norm(prototype), 1e-12))
    p = np.stack(prototypes)
    pred = (F.normalize(torch.from_numpy(features), dim=1).numpy() @ p.T).argmax(1)
    return metric_dict(labels, pred)


def all_target(model, cube, gt, device, batch_size=64):
    padded = np.pad(cube, ((HALF, HALF), (HALF, HALF), (0, 0)), mode="constant")
    ids = np.flatnonzero(gt.reshape(-1) > 0)
    y = gt.reshape(-1)[ids].astype(np.int64) - 1
    fspec, pred, logits = [], [], []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            centers = np.asarray([np.unravel_index(int(i), gt.shape) for i in ids[start:start + batch_size]])
            xb = np.empty((len(centers), cube.shape[-1], IMG_SIZE, IMG_SIZE), dtype=np.float32)
            for j, (r, c) in enumerate(centers):
                xb[j] = padded[r:r + IMG_SIZE, c:c + IMG_SIZE].transpose(2, 0, 1)
            _, fe, _, lo = forward_features(model, torch.from_numpy(xb).to(device))
            fspec.append(fe.cpu().numpy()); logits.append(lo.cpu().numpy()); pred.append(lo.argmax(1).cpu().numpy())
    return y, np.concatenate(pred), np.concatenate(fspec), np.concatenate(logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--stage1-artifact", type=Path, default=Path("/nas1/zhangzj26/HyperSIGMA_adapted/protocol_stage1/bands48"))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(); device = torch.device(args.device)
    source, src_gt = utils.load_data_houston(str(ROOT / "datasets/Houston/Houston13.mat"), str(ROOT / "datasets/Houston/Houston13_7gt.mat"))
    split = np.load(args.artifact / "source_split.npz")
    train_c, val_c = split["train_centers"], split["val_centers"]
    train_y = src_gt[train_c[:, 0], train_c[:, 1]].astype(np.int64) - 1
    val_y = src_gt[val_c[:, 0], val_c[:, 1]].astype(np.int64) - 1
    model = SSFusionFramework(img_size=IMG_SIZE, in_channels=48, patch_size=2, classes=CLASSES, model_size="base")
    ck = torch.load(args.artifact / ("stage1_best.pth" if (args.artifact / "stage1_best.pth").exists() else "best.pth"), map_location="cpu")
    model.load_state_dict(ck["model"], strict=True); model.to(device).eval()
    train = extract_centers(model, source, train_c, device, args.batch_size)
    val = extract_centers(model, source, val_c, device, args.batch_size)
    source_metrics = {
        "train": {"loss": float(torch.nn.functional.cross_entropy(torch.from_numpy(train["logits"]), torch.from_numpy(train_y)).item()), "accuracy": float(np.mean(train["logits"].argmax(1) == train_y))},
        "disjoint_val": {"loss": float(torch.nn.functional.cross_entropy(torch.from_numpy(val["logits"]), torch.from_numpy(val_y)).item()), "accuracy": float(np.mean(val["logits"].argmax(1) == val_y))},
    }

    # Houston18 is deliberately loaded only after source checkpoint selection.
    target, tgt_gt = utils.load_data_houston(str(ROOT / "datasets/Houston/Houston18.mat"), str(ROOT / "datasets/Houston/Houston18_7gt.mat"))
    tgt_y, tgt_pred, tgt_spec, _ = all_target(model, target, tgt_gt, device, args.batch_size)
    head_target = metric_dict(tgt_y, tgt_pred)
    proto_val = source_prototype(train["spec"], train_y, val["spec"], val_y)
    proto_target = source_prototype(train["spec"], train_y, tgt_spec, tgt_y)
    probe = probe_features(model, source, src_gt, target, tgt_gt, device)
    result = {"artifact": str(args.artifact), "target_gt_used_for_training_or_selection": False,
              "source": source_metrics, "head_houston18": head_target,
              "f_spec_source_prototype": {"source_disjoint_val": proto_val, "houston18": proto_target},
              "feature_probe": probe}
    (args.artifact / "posthoc_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
