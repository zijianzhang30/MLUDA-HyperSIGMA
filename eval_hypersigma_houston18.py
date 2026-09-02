"""Post-training, read-only Houston18 evaluation for an adapted teacher.

Houston18 labels are used only after checkpoint selection to report metrics.
They are never passed to optimization, early stopping, or model selection.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "HyperSIGMA" / "ImageClassification"))
import utils  # noqa: E402
from hypersigma_teacher_smoke_test import SSFusionFramework  # noqa: E402

NUM_CLASSES = 7
IMG_SIZE = 33


def patches_for_indices(padded, shape, indices):
    batch = np.empty((len(indices), 30, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for i, flat in enumerate(indices):
        r, c = np.unravel_index(int(flat), shape)
        batch[i] = padded[r:r + IMG_SIZE, c:c + IMG_SIZE].transpose(2, 0, 1)
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    artifact = args.artifact_dir
    device = torch.device(args.device)
    checkpoint = torch.load(artifact / "houston13_vitb_33x33_pca30_stage1_best.pth", map_location="cpu")
    with open(artifact / "houston13_source_pca.pkl", "rb") as f:
        pca = pickle.load(f)
    data, gt = utils.load_data_houston(str(ROOT / "datasets/Houston/Houston18.mat"), str(ROOT / "datasets/Houston/Houston18_7gt.mat"))
    pca_data = pca.transform(data.reshape(-1, data.shape[-1]).astype(np.float32)).reshape(data.shape[0], data.shape[1], 30).astype(np.float32)
    margin = IMG_SIZE // 2
    padded = np.pad(pca_data, ((margin, margin), (margin, margin), (0, 0)), mode="constant")
    indices = np.flatnonzero(gt.reshape(-1) > 0)
    y_true = gt.reshape(-1)[indices].astype(np.int64) - 1
    teacher = SSFusionFramework(img_size=33, in_channels=30, patch_size=2, classes=7, model_size="base")
    teacher.load_state_dict(checkpoint["model"])
    teacher.to(device).eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(indices), args.batch_size):
            xb = torch.from_numpy(patches_for_indices(padded, gt.shape, indices[start:start + args.batch_size])).to(device)
            predictions.append(teacher(xb).argmax(1).cpu().numpy())
    y_pred = np.concatenate(predictions)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    per_class = np.divide(np.diag(cm), cm.sum(1), out=np.full(NUM_CLASSES, np.nan), where=cm.sum(1) != 0)
    metrics = {
        "protocol": "post_training_evaluation_only; Houston18 GT not used for training/selection",
        "checkpoint_seed": int(checkpoint["seed"]),
        "n_target_labeled_pixels": int(len(y_true)),
        "oa": float((y_true == y_pred).mean()),
        "aa": float(np.nanmean(per_class)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class_accuracy": per_class.tolist(),
        "class_counts": cm.sum(1).tolist(),
        "confusion_matrix": cm.tolist(),
    }
    path = artifact / "houston18_posthoc_metrics.json"
    path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
