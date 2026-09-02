"""Offline Houston18 prediction complementarity audit.

This script is deliberately post-hoc: it loads fixed checkpoints first, then
opens Houston18 GT only to compute diagnostics.  No training, model selection,
or hyper-parameter tuning is performed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hdf5storage
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config_Houston import BATCH_SIZE, CLASS_NUM, HalfWidth, nBand  # noqa: E402
from net2 import DSANSS  # noqa: E402
from UtilsCMS import ILDA  # noqa: E402
import utils  # noqa: E402
from MLUDA_hu_fspec_kd import center_patches, paired_source_samples  # noqa: E402
from hypersigma_stage1_protocol import forward_parts  # noqa: E402
from hypersigma_teacher_smoke_test import SSFusionFramework  # noqa: E402

TEACHER_CKPT = Path("/nas1/zhangzj26/HyperSIGMA_adapted/protocol_stage1/bands48/stage1_best.pth")
MLUDA_CKPT = Path("/nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_kd/lambda_0/seed_1174_best.pth")
OUT = Path("/nas1/zhangzj26/HyperSIGMA_adapted/diagnostics/prediction_complementarity_seed1174.json")


def metrics_dict(y: np.ndarray, pred: np.ndarray) -> dict:
    cm = confusion_matrix(y, pred, labels=np.arange(CLASS_NUM))
    pc = np.diag(cm) / np.maximum(cm.sum(1), 1)
    return {
        "oa": float(np.mean(y == pred)),
        "aa": float(np.mean(pc)),
        "kappa": float(cohen_kappa_score(y, pred, labels=np.arange(CLASS_NUM))),
        "per_class_accuracy": pc.tolist(),
        "confusion_matrix_true_rows_pred_columns": cm.tolist(),
        "prediction_distribution_zero_based": np.bincount(pred, minlength=CLASS_NUM).tolist(),
    }


def load_teacher(device: torch.device) -> SSFusionFramework:
    model = SSFusionFramework(img_size=33, in_channels=48, patch_size=2,
                              classes=CLASS_NUM, model_size="base")
    ck = torch.load(TEACHER_CKPT, map_location="cpu")
    model.load_state_dict(ck["model"], strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def teacher_predict(model, raw_cube, centers, device, batch_size):
    half = 16
    padded = np.pad(raw_cube, ((half, half), (half, half), (0, 0)), mode="constant")
    out = []
    with torch.no_grad():
        for start in range(0, len(centers), batch_size):
            cc = centers[start:start + batch_size]
            xb = np.empty((len(cc), 48, 33, 33), np.float32)
            for j, (r, c) in enumerate(cc):
                xb[j] = padded[r:r + 33, c:c + 33].transpose(2, 0, 1)
            out.append(model(torch.from_numpy(xb).to(device)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def mluda_predict(model, target_x, source_x, device, batch_size):
    pred = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(target_x), batch_size):
            xb = torch.from_numpy(target_x[start:start + batch_size]).to(device)
            ref = torch.from_numpy(source_x[:len(xb)]).to(device)
            pred.append(model(ref, xb)[8].argmax(1).cpu().numpy())
    return np.concatenate(pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    device = torch.device(args.device)

    # Fixed checkpoints are loaded before any target labels are read.
    teacher = load_teacher(device)
    ck = torch.load(MLUDA_CKPT, map_location="cpu")
    mluda = DSANSS(nBand, 7, CLASS_NUM).to(device)
    mluda.load_state_dict(ck["model"], strict=True)
    mluda.eval()

    source, source_gt = utils.load_data_houston(
        str(ROOT / "datasets/Houston/Houston13.mat"),
        str(ROOT / "datasets/Houston/Houston13_7gt.mat"))
    target = hdf5storage.loadmat(str(ROOT / "datasets/Houston/Houston18.mat"))["ori_data"]
    # This is exactly the original MLUDA image-level preprocessing.
    data_s, data_t = ILDA(source, target, 2, 0.009)
    _, source_x, _, _, _, _, _, _ = paired_source_samples(data_s, data_s, source_gt, 1174)

    # Target centers are all non-background centers, in the same order as the
    # Full48 F_spec cache and the baseline post-hoc evaluator.
    target_gt = hdf5storage.loadmat(str(ROOT / "datasets/Houston/Houston18_7gt.mat"))["map"]
    rows, cols = np.nonzero(target_gt > 0)
    centers = np.stack([rows, cols], axis=1).astype(np.int64)
    target_y = target_gt[rows, cols].astype(np.int64) - 1
    raw_target_x = center_patches(data_t, centers, 7)
    teacher_pred = teacher_predict(teacher, target, centers, device, args.batch_size)
    mluda_pred = mluda_predict(mluda, raw_target_x, source_x, device, args.batch_size)

    t_ok = teacher_pred == target_y
    m_ok = mluda_pred == target_y
    categories = {
        "both_correct": int(np.sum(t_ok & m_ok)),
        "mluda_only_correct": int(np.sum(~t_ok & m_ok)),
        "hypersigma_only_correct": int(np.sum(t_ok & ~m_ok)),
        "both_wrong": int(np.sum(~t_ok & ~m_ok)),
    }
    denom = ~m_ok
    overall_cond = float(np.sum(t_ok & denom) / max(np.sum(denom), 1))
    per_class_cond = []
    for cls in range(CLASS_NUM):
        cls_mask = target_y == cls
        mask = denom & cls_mask
        per_class_cond.append({
            "class_index_zero_based": cls,
            "class_label_one_based": cls + 1,
            "mluDA_wrong_count": int(mask.sum()),
            "teacher_correct_count": int(np.sum(t_ok & mask)),
            "p_teacher_correct_given_mluDA_wrong": float(np.sum(t_ok & mask) / max(mask.sum(), 1)),
            "category_counts": {
                "both_correct": int(np.sum(t_ok & m_ok & cls_mask)),
                "mluda_only_correct": int(np.sum(~t_ok & m_ok & cls_mask)),
                "hypersigma_only_correct": int(np.sum(t_ok & ~m_ok & cls_mask)),
                "both_wrong": int(np.sum(~t_ok & ~m_ok & cls_mask)),
            },
        })
    result = {
        "protocol": "offline post-hoc only; Houston18 GT not used in training or checkpoint selection",
        "teacher_checkpoint": str(TEACHER_CKPT),
        "mluda_checkpoint": str(MLUDA_CKPT),
        "mluda_checkpoint_note": "lambda=0 control run with original MLUDA architecture/protocol, seed 1174",
        "n": int(len(target_y)), "classes": CLASS_NUM,
        "label_mapping": "Houston labels 1..7 mapped to indices 0..6; background 0 excluded",
        "categories": categories,
        "category_fractions": {k: v / len(target_y) for k, v in categories.items()},
        "p_teacher_correct_given_mluDA_wrong_overall": overall_cond,
        "p_teacher_correct_given_mluDA_wrong_per_class": per_class_cond,
        "teacher": metrics_dict(target_y, teacher_pred),
        "mluda": metrics_dict(target_y, mluda_pred),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
